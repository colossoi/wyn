//! Tests for `analyze_entry` — the tail-position walk that finds the
//! top-level SOAC of a compute entry point.
//!
//! The 11 tests exercise creative ways a SOAC can be "hidden" at the
//! tail of an entry body: wrapped in Lambdas, Force markers, Let
//! chains, alias lets (`let x = soac in x`), chained aliases, nested
//! let-in-RHS, and rejections for non-tail positions (If, Loop, App).
//!
//! T11 additionally verifies `required_params` — the subset of outer
//! Lambda params that the restructured body actually references.

use super::VarRef;
use super::analyze_entry;
use crate::ast::{self, Span, TypeName};
use crate::tlc::{
    ArrayExpr, Def, DefMeta, Lambda, LoopKind, SoacBody, SoacDestination, SoacOp, Term, TermId,
    TermIdSource, TermKind,
};
use crate::{BindingRef, SymbolId, SymbolTable};
use polytype::Type;

// ---------- type helpers ----------

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn bool_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Bool, vec![])
}

fn arr_i32_ty() -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            i32_ty(),
            Type::Constructed(TypeName::Size(8), vec![]),
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
        ],
    )
}

// ---------- term helpers ----------

struct B {
    symbols: SymbolTable,
    ids: TermIdSource,
}

impl B {
    fn new() -> Self {
        B {
            symbols: SymbolTable::new(),
            ids: TermIdSource::new(),
        }
    }

    fn sym(&mut self, name: &str) -> SymbolId {
        self.symbols.alloc(name.to_string())
    }

    fn id(&mut self) -> TermId {
        self.ids.next_id()
    }

    fn term(&mut self, kind: TermKind, ty: Type<TypeName>) -> Term {
        Term {
            id: self.id(),
            ty,
            span: Span::dummy(),
            kind,
        }
    }

    fn int_lit(&mut self, v: i64) -> Term {
        self.term(TermKind::IntLit(v.to_string()), i32_ty())
    }

    fn var(&mut self, sym: SymbolId, ty: Type<TypeName>) -> Term {
        self.term(TermKind::Var(VarRef::Symbol(sym)), ty)
    }

    fn let_(&mut self, name: SymbolId, name_ty: Type<TypeName>, rhs: Term, body: Term) -> Term {
        let body_ty = body.ty.clone();
        self.term(
            TermKind::Let {
                name,
                name_ty,
                rhs: Box::new(rhs),
                body: Box::new(body),
            },
            body_ty,
        )
    }

    /// Build `map(|x| x, 0..<8)` producing `[8]i32`.
    fn trivial_map(&mut self) -> Term {
        let x = self.sym("x");
        let body = self.var(x, i32_ty());
        let lam = Lambda {
            params: vec![(x, i32_ty())],
            body: Box::new(body),
            ret_ty: i32_ty(),
        };
        let start = self.int_lit(0);
        let len = self.int_lit(8);
        let input = ArrayExpr::Range {
            start: Box::new(start),
            len: Box::new(len),
            step: None,
        };
        self.term(
            TermKind::Soac(SoacOp::Map {
                lam: SoacBody {
                    lam,
                    captures: vec![],
                },
                inputs: vec![input],
                destination: SoacDestination::Fresh,
            }),
            arr_i32_ty(),
        )
    }

    /// Wrap `body` in a single flat Lambda with the given params,
    /// mirroring what `transform_entry` produces for entry points.
    fn entry_def(&mut self, name: SymbolId, params: Vec<(SymbolId, Type<TypeName>)>, body: Term) -> Def {
        let body_ty = body.ty.clone();
        let lam_term = self.term(
            TermKind::Lambda(Lambda {
                params,
                body: Box::new(body),
                ret_ty: body_ty.clone(),
            }),
            body_ty.clone(),
        );
        Def {
            name,
            ty: body_ty,
            body: lam_term,
            meta: DefMeta::Function,
            arity: 1,
        }
    }
}

// ---------- tests ----------

/// T1: Lambda + Soac (baseline).
#[test]
fn t1_naked_lambda_soac() {
    let mut b = B::new();
    let p = b.sym("p");
    let body = b.trivial_map();
    let name = b.sym("entry");
    let def = b.entry_def(name, vec![(p, i32_ty())], body);
    let a = analyze_entry(&def, &b.symbols).expect("should find SOAC");
    assert!(a.prefix_lets.is_empty());
    assert!(matches!(a.soac.original, SoacOp::Map { .. }));
}

/// T2: Lambda + deep Let chain + Soac. All lets should appear in
/// prefix_lets, in insertion order.
#[test]
fn t2_deep_let_chain() {
    let mut b = B::new();
    let p = b.sym("p");
    let a_sym = b.sym("a");
    let bb_sym = b.sym("b");
    let c_sym = b.sym("c");

    let soac = b.trivial_map();
    let rhs_a = b.int_lit(1);
    let rhs_b = b.int_lit(2);
    let rhs_c = b.int_lit(3);
    let inner_c = b.let_(c_sym, i32_ty(), rhs_c, soac);
    let inner_b = b.let_(bb_sym, i32_ty(), rhs_b, inner_c);
    let body = b.let_(a_sym, i32_ty(), rhs_a, inner_b);
    let name = b.sym("entry");
    let def = b.entry_def(name, vec![(p, i32_ty())], body);
    let a = analyze_entry(&def, &b.symbols).expect("should find SOAC");
    assert_eq!(a.prefix_lets.len(), 3);
    assert_eq!(a.prefix_lets[0].0, a_sym);
    assert_eq!(a.prefix_lets[1].0, bb_sym);
    assert_eq!(a.prefix_lets[2].0, c_sym);
}

/// T3: `let x = soac in x` — the alias let is consumed by Var-follow,
/// leaving no prefix lets.
#[test]
fn t3_alias_let() {
    let mut b = B::new();
    let p = b.sym("p");
    let x = b.sym("x");

    let soac = b.trivial_map();
    let tail = b.var(x, arr_i32_ty());
    let body = b.let_(x, arr_i32_ty(), soac, tail);
    let name = b.sym("entry");
    let def = b.entry_def(name, vec![(p, i32_ty())], body);
    let a = analyze_entry(&def, &b.symbols).expect("should find SOAC");
    assert!(a.prefix_lets.is_empty(), "alias let should be consumed");
    assert!(matches!(a.soac.original, SoacOp::Map { .. }));
}

/// T4: `let x = soac in let y = x in y` — both aliases consumed.
#[test]
fn t4_chained_aliases() {
    let mut b = B::new();
    let p = b.sym("p");
    let x = b.sym("x");
    let y = b.sym("y");

    let soac = b.trivial_map();
    let x_ref = b.var(x, arr_i32_ty());
    let y_ref = b.var(y, arr_i32_ty());
    let inner = b.let_(y, arr_i32_ty(), x_ref, y_ref);
    let body = b.let_(x, arr_i32_ty(), soac, inner);
    let name = b.sym("entry");
    let def = b.entry_def(name, vec![(p, i32_ty())], body);
    let a = analyze_entry(&def, &b.symbols).expect("should find SOAC");
    assert!(a.prefix_lets.is_empty());
}

/// T5: `let x = (let y = 1 in soac) in x` — x consumed by Var-follow,
/// inner `y` surfaces as a prefix.
#[test]
fn t5_nested_let_in_rhs() {
    let mut b = B::new();
    let p = b.sym("p");
    let x = b.sym("x");
    let y = b.sym("y");

    let soac = b.trivial_map();
    let rhs_y = b.int_lit(1);
    let inner = b.let_(y, i32_ty(), rhs_y, soac);
    let x_ref = b.var(x, arr_i32_ty());
    let body = b.let_(x, arr_i32_ty(), inner, x_ref);
    let name = b.sym("entry");
    let def = b.entry_def(name, vec![(p, i32_ty())], body);
    let a = analyze_entry(&def, &b.symbols).expect("should find SOAC");
    assert_eq!(a.prefix_lets.len(), 1);
    assert_eq!(a.prefix_lets[0].0, y);
}

/// T7: SOAC inside an If arm — If is the tail, reject.
#[test]
fn t7_if_arm_rejected() {
    let mut b = B::new();
    let p = b.sym("p");
    let soac = b.trivial_map();
    let cond = b.term(TermKind::BoolLit(true), bool_ty());
    let zero = b.int_lit(0);
    let body = b.term(
        TermKind::If {
            cond: Box::new(cond),
            then_branch: Box::new(soac),
            else_branch: Box::new(zero),
        },
        arr_i32_ty(),
    );
    let name = b.sym("entry");
    let def = b.entry_def(name, vec![(p, bool_ty())], body);
    assert!(analyze_entry(&def, &b.symbols).is_none());
}

/// T8: SOAC inside a Loop body — Loop is the tail, reject.
#[test]
fn t8_loop_body_rejected() {
    let mut b = B::new();
    let p = b.sym("p");
    let acc = b.sym("acc");
    let i = b.sym("i");
    let soac = b.trivial_map();
    let init = b.int_lit(0);
    let bound = b.int_lit(3);
    let body = b.term(
        TermKind::Loop {
            loop_var: acc,
            loop_var_ty: i32_ty(),
            init: Box::new(init),
            init_bindings: vec![],
            kind: LoopKind::ForRange {
                var: i,
                var_ty: i32_ty(),
                bound: Box::new(bound),
            },
            body: Box::new(soac),
        },
        arr_i32_ty(),
    );
    let name = b.sym("entry");
    let def = b.entry_def(name, vec![(p, i32_ty())], body);
    assert!(analyze_entry(&def, &b.symbols).is_none());
}

/// T9: SOAC as an App argument — App is the tail, reject.
#[test]
fn t9_app_arg_rejected() {
    let mut b = B::new();
    let p = b.sym("p");
    let f = b.sym("someFn");
    let soac = b.trivial_map();
    let arg2 = b.var(p, i32_ty());
    let func = b.var(f, i32_ty()); // type doesn't matter for rejection
    let body = b.term(
        TermKind::App {
            func: Box::new(func),
            args: vec![soac, arg2],
        },
        arr_i32_ty(),
    );
    let name = b.sym("entry");
    let def = b.entry_def(name, vec![(p, i32_ty())], body);
    assert!(analyze_entry(&def, &b.symbols).is_none());
}

/// T10: Plain literal at tail — not a SOAC.
#[test]
fn t10_literal_tail_rejected() {
    let mut b = B::new();
    let p = b.sym("p");
    let body = b.int_lit(42);
    let name = b.sym("entry");
    let def = b.entry_def(name, vec![(p, i32_ty())], body);
    assert!(analyze_entry(&def, &b.symbols).is_none());
}

/// T11: SOAC lambda references an entry param — required_params
/// must surface exactly that param.
#[test]
fn t11_required_params_closure() {
    let mut b = B::new();
    let p = b.sym("p"); // unused
    let q = b.sym("q"); // referenced by the map's lambda
    let x = b.sym("x");

    // Inner lambda: |x: i32| x + q
    let x_ref = b.var(x, i32_ty());
    let q_ref = b.var(q, u32_ty());
    let plus = b.term(
        TermKind::BinOp(ast::BinaryOp { op: "+".to_string() }),
        Type::Constructed(
            TypeName::Arrow,
            vec![
                i32_ty(),
                Type::Constructed(TypeName::Arrow, vec![i32_ty(), i32_ty()]),
            ],
        ),
    );
    let plus_body = b.term(
        TermKind::App {
            func: Box::new(plus),
            args: vec![x_ref, q_ref],
        },
        i32_ty(),
    );
    let lam = Lambda {
        params: vec![(x, i32_ty())],
        body: Box::new(plus_body),
        ret_ty: i32_ty(),
    };
    let start = b.int_lit(0);
    let len = b.int_lit(8);
    let input = ArrayExpr::Range {
        start: Box::new(start),
        len: Box::new(len),
        step: None,
    };
    let body = b.term(
        TermKind::Soac(SoacOp::Map {
            lam: SoacBody {
                lam,
                captures: vec![],
            },
            inputs: vec![input],
            destination: SoacDestination::Fresh,
        }),
        arr_i32_ty(),
    );
    let name = b.sym("entry");
    let def = b.entry_def(name, vec![(p, arr_i32_ty()), (q, u32_ty())], body);
    let a = analyze_entry(&def, &b.symbols).expect("should find SOAC");
    assert!(a.prefix_lets.is_empty());
    assert_eq!(
        a.required_params.len(),
        1,
        "only q is referenced; p should be filtered out"
    );
    assert_eq!(a.required_params[0].0, q);
}

// ----------------------------------------------------------------------
// Binding registry tests
// ----------------------------------------------------------------------

use super::collect_all_used_bindings;
use crate::tlc::Program;
use std::collections::HashMap;

/// Build a minimal `Program` wrapping a single def body, for binding
/// registry tests. No uniforms or user-declared storage.
fn program_wrapping_body(b: &mut B, body: Term) -> Program {
    let name = b.sym("entry");
    let def = b.entry_def(name, vec![], body);
    Program {
        defs: vec![def],
        symbols: std::mem::replace(&mut b.symbols, SymbolTable::new()),
        def_syms: HashMap::new(),
    }
}

/// Binding registry finds `ArrayExpr::StorageBuffer` nested inside a
/// SOAC input. The scan/reduce allocator must see these so it doesn't
/// pick a colliding (set, binding) for its phase buffers.
#[test]
fn binding_registry_finds_storage_buffer_in_soac_input() {
    let mut b = B::new();
    let u32_ty_v = u32_ty();

    // Construct: map(identity, StorageBuffer{set=0, binding=3, offset=0, len=16, elem_ty=i32})
    let x = b.sym("x");
    let lam = Lambda {
        params: vec![(x, i32_ty())],
        body: Box::new(b.var(x, i32_ty())),
        ret_ty: i32_ty(),
    };
    let offset = b.term(TermKind::IntLit("0".into()), u32_ty_v.clone());
    let len = b.term(TermKind::IntLit("16".into()), u32_ty_v);
    let input = ArrayExpr::StorageView(crate::tlc::StorageView {
        binding: BindingRef::new(0, 3),
        offset: Box::new(offset),
        len: Box::new(len),
        elem_ty: i32_ty(),
    });
    let soac = b.term(
        TermKind::Soac(SoacOp::Map {
            lam: SoacBody {
                lam,
                captures: vec![],
            },
            inputs: vec![input],
            destination: SoacDestination::Fresh,
        }),
        arr_i32_ty(),
    );

    let program = program_wrapping_body(&mut b, soac);
    let used = collect_all_used_bindings(&program);
    assert!(
        used.contains(&BindingRef::new(0, 3)),
        "expected set=0,binding=3 in used bindings; got {:?}",
        used
    );
}

/// Binding registry also sees `StorageBuffer` bindings threaded through
/// nested `ArrayExpr::Zip` / `ArrayExpr::Ref` wrappers, and through
/// nested `Soac` inside the SOAC (Redomap + inner Scan case).
#[test]
fn binding_registry_finds_nested_storage_buffers() {
    let mut b = B::new();
    let u32_ty_v = u32_ty();

    let mk_sb = |b: &mut B, binding: u32| -> ArrayExpr {
        let offset = b.term(TermKind::IntLit("0".into()), u32_ty_v.clone());
        let len = b.term(TermKind::IntLit("8".into()), u32_ty_v.clone());
        ArrayExpr::StorageView(crate::tlc::StorageView {
            binding: BindingRef::new(0, binding),
            offset: Box::new(offset),
            len: Box::new(len),
            elem_ty: i32_ty(),
        })
    };

    let x = b.sym("x");
    let y = b.sym("y");
    let inner_lam = Lambda {
        params: vec![(x, i32_ty()), (y, i32_ty())],
        body: Box::new(b.var(x, i32_ty())),
        ret_ty: i32_ty(),
    };
    let sb_a = mk_sb(&mut b, 5);
    let sb_b = mk_sb(&mut b, 7);
    let zip = ArrayExpr::Zip(vec![sb_a, sb_b]);
    let soac = b.term(
        TermKind::Soac(SoacOp::Map {
            lam: SoacBody {
                lam: inner_lam,
                captures: vec![],
            },
            inputs: vec![zip],
            destination: SoacDestination::Fresh,
        }),
        arr_i32_ty(),
    );

    let program = program_wrapping_body(&mut b, soac);
    let used = collect_all_used_bindings(&program);
    assert!(
        used.contains(&BindingRef::new(0, 5)),
        "missing set=0,binding=5: {:?}",
        used
    );
    assert!(
        used.contains(&BindingRef::new(0, 7)),
        "missing set=0,binding=7: {:?}",
        used
    );
}

// =============================================================================
// pick_workgroup_size — bucket policy for #[size_hint(N)]
// =============================================================================

#[test]
fn pick_workgroup_size_no_hint_is_64() {
    assert_eq!(super::pick_workgroup_size(None), (64, 1, 1));
}

#[test]
fn pick_workgroup_size_tiny_rounds_up_to_power_of_two() {
    let nz = |n| std::num::NonZeroU32::new(n).unwrap();
    // 1 → next_power_of_two(1) = 1
    assert_eq!(super::pick_workgroup_size(Some(nz(1))), (1, 1, 1));
    // 3 → 4
    assert_eq!(super::pick_workgroup_size(Some(nz(3))), (4, 1, 1));
    // 16 → 16
    assert_eq!(super::pick_workgroup_size(Some(nz(16))), (16, 1, 1));
    // 63 → 64
    assert_eq!(super::pick_workgroup_size(Some(nz(63))), (64, 1, 1));
}

#[test]
fn pick_workgroup_size_medium_stays_64() {
    let nz = |n| std::num::NonZeroU32::new(n).unwrap();
    assert_eq!(super::pick_workgroup_size(Some(nz(64))), (64, 1, 1));
    assert_eq!(super::pick_workgroup_size(Some(nz(1024))), (64, 1, 1));
    assert_eq!(super::pick_workgroup_size(Some(nz(65_536))), (64, 1, 1));
}

#[test]
fn pick_workgroup_size_large_picks_256() {
    let nz = |n| std::num::NonZeroU32::new(n).unwrap();
    assert_eq!(super::pick_workgroup_size(Some(nz(65_537))), (256, 1, 1));
    assert_eq!(super::pick_workgroup_size(Some(nz(1_000_000))), (256, 1, 1));
}

// =============================================================================
// Pipeline-shape tests — exercise nested SOACs and multi-entry generation
// =============================================================================
//
// These compile small Wyn snippets through the TLC pipeline up to
// `parallelize_soacs` and assert on the resulting `PipelineDescriptor` and
// the number of compute entry-point defs in the program. They use the
// public `compile_thru_tlc` path plus the parallelize milestone directly
// (no EGIR/SSA needed — every fact is on the TLC side once parallelize ran).

use crate::interface::Attribute;
use crate::pipeline_descriptor::{DispatchSize, MultiComputePipeline, Pipeline};

/// Run `src` through the canonical TLC pipeline including `parallelize_soacs`,
/// returning the parallelized program + pipeline descriptor.
fn parallelize_src(
    src: &str,
) -> (
    crate::tlc::Program,
    crate::pipeline_descriptor::PipelineDescriptor,
) {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let parsed = crate::Compiler::parse(src, &mut node_counter).expect("parse");
    let type_checked = parsed
        .resolve(&module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    let tlc = type_checked
        .to_tlc(&module_manager, false)
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .apply_ownership()
        .expect("apply_ownership")
        .normalize_outputs()
        .expect("normalize_outputs")
        .lift_gathers()
        .defunctionalize()
        .monomorphize()
        .buffer_specialize()
        .fold_generated_lambdas()
        .inline_small()
        .parallelize_soacs(false)
        .expect("parallelize_soacs");
    (tlc.tlc.clone(), tlc.pipeline.clone())
}

/// Count compute entry-point defs in the program.
fn compute_entry_count(program: &crate::tlc::Program) -> usize {
    program
        .defs
        .iter()
        .filter(|d| matches!(&d.meta, DefMeta::EntryPoint(e) if matches!(e.entry_type, Attribute::Compute)))
        .count()
}

/// Stage count of the first pipeline matching `entry_name`, if it's a
/// multi-compute pipeline.
fn multi_stage_count(
    desc: &crate::pipeline_descriptor::PipelineDescriptor,
    entry_name: &str,
) -> Option<usize> {
    desc.pipelines.iter().find_map(|p| match p {
        Pipeline::MultiCompute(MultiComputePipeline { stages, .. })
            if stages.iter().any(|s| s.entry_point == entry_name) =>
        {
            Some(stages.len())
        }
        _ => None,
    })
}

// ----- Nested SOACs -----

/// `map(g, map(f, xs))` fuses into a single Map at TLC `fuse_maps`, so it
/// parallelizes as one compute pipeline with one entry-point def.
#[test]
fn nested_map_of_map_fuses_into_single_pipeline() {
    let src = r#"
        #[compute]
        entry e(xs: []i32) []i32 = map(|y: i32| y * 2, map(|x: i32| x + 1, xs))
    "#;
    let (program, desc) = parallelize_src(src);
    assert_eq!(
        compute_entry_count(&program),
        1,
        "map-of-map should fuse to one entry"
    );
    assert_eq!(desc.pipelines.len(), 1, "one fused pipeline");
    assert!(
        matches!(desc.pipelines[0], Pipeline::Compute(_)),
        "single-dispatch compute"
    );
}

/// `reduce(+, 0, map(f, xs))` becomes a redomap, which lowers to a two-phase
/// (phase1 chunked + phase2 single-workgroup tree) multi-compute pipeline.
#[test]
fn nested_reduce_of_map_becomes_two_phase_multi_compute() {
    let src = r#"
        #[compute]
        entry e(xs: []i32) i32 =
            reduce(|a: i32, b: i32| a + b, 0, map(|x: i32| x * 2, xs))
    "#;
    let (_program, desc) = parallelize_src(src);
    assert_eq!(desc.pipelines.len(), 1);
    let stages = multi_stage_count(&desc, "e_phase1_chunks")
        .or_else(|| multi_stage_count(&desc, "e_phase2_combine"))
        .expect("reduce(map(...)) should produce a multi-compute pipeline");
    assert_eq!(stages, 2, "two-phase reduce: phase1 + phase2");
}

/// `scan(+, 0, map(f, xs))` is a parallelizable scan: 3 stages
/// (phase1 chunks + phase2 sums + phase3 add offsets).
#[test]
fn nested_scan_of_map_yields_three_stage_multi_compute() {
    let src = r#"
        #[compute]
        entry e(xs: []i32) []i32 =
            scan(|a: i32, b: i32| a + b, 0, map(|x: i32| x * 2, xs))
    "#;
    let (_program, desc) = parallelize_src(src);
    let stages = multi_stage_count(&desc, "e").expect("scan should produce a multi-compute pipeline");
    assert_eq!(stages, 3, "scan: phase1 + phase2 + phase3");
}

/// A SOAC inside a non-tail position — here, a `let` binding whose RHS is a
/// reduce, with a final Map that consumes it — exercises whether the tail
/// SOAC analysis sees through the let-prefix.
#[test]
fn let_bound_reduce_then_map_parallelizes_the_tail_map() {
    let src = r#"
        #[compute]
        entry e(xs: []i32) []i32 =
            let s = reduce(|a: i32, b: i32| a + b, 0, xs) in
            map(|x: i32| x + s, xs)
    "#;
    let (_program, desc) = parallelize_src(src);
    // The tail SOAC is the map. Whether the let-bound reduce gets its own
    // pipeline depends on whether `lift_graphical_invariant_soacs` /
    // `lift_gathers` hoists it. Either way the descriptor must include at
    // least one pipeline for the tail map.
    assert!(
        desc.pipelines.iter().any(|p| matches!(p, Pipeline::Compute(_) | Pipeline::MultiCompute(_))),
        "tail map should produce a pipeline"
    );
}

// ----- Multi-entry generation -----

/// Two independent top-level entries are preserved through parallelize as
/// two separate pipelines (one per entry).
#[test]
fn two_user_entries_yield_two_pipelines() {
    let src = r#"
        #[compute]
        entry inc(xs: []i32) []i32 = map(|x: i32| x + 1, xs)

        #[compute]
        entry dbl(ys: []i32) []i32 = map(|y: i32| y * 2, ys)
    "#;
    let (program, desc) = parallelize_src(src);
    assert_eq!(compute_entry_count(&program), 2);
    assert_eq!(desc.pipelines.len(), 2);
}

/// A `gather` — `map(|i| arr[idx_expr(i)], iota(N))` where `arr` is itself a
/// SOAC result — is hoisted by `lift_gathers` into a compute pre-pass entry
/// that materializes `arr`, leaving the consumer to read the materialized
/// buffer. One source entry → two compute entries + two pipelines.
#[test]
fn gather_yields_prepass_plus_consumer_entries() {
    let src = r#"
        #[compute]
        entry gen(bh: []vec4f32) []i32 =
            let counts = map(|h: vec4f32| 4 + 5 * (if h.x > 4.0 then 3 else 1), bh) in
            map(|i: i32| counts[i % 256], iota(6144))
    "#;
    let (program, desc) = parallelize_src(src);
    assert!(
        compute_entry_count(&program) >= 2,
        "gather should produce a pre-pass entry"
    );
    assert!(
        desc.pipelines.len() >= 2,
        "gather → pre-pass + consumer pipelines"
    );
}

/// Chaining `map → scan → gather` exercises both fusion and gather-lifting:
/// `fuse_maps` folds the producer map into the scan, so a single source
/// entry produces (i) a `MultiCompute` scan-of-map pipeline (3 stages) and
/// (ii) a `Compute` gather-consumer pipeline. Net: 2 compute entries, 2
/// pipelines, 4 total dispatch stages — demonstrating that fusion can
/// collapse N SOACs into <N entries even as gather-lifting splits them out.
#[test]
fn chained_map_scan_gather_yields_two_pipelines_four_stages() {
    let src = r#"
        #[compute]
        entry gen(bh: []vec4f32) []i32 =
            let counts = map(|h: vec4f32| 4 + 5 * (if h.x > 4.0 then 3 else 1), bh) in
            let offsets = scan(|a: i32, b: i32| a + b, 0, counts) in
            map(|i: i32| offsets[i % 256], iota(6144))
    "#;
    let (program, desc) = parallelize_src(src);
    assert_eq!(compute_entry_count(&program), 2, "scan-of-map + gather consumer");
    assert_eq!(desc.pipelines.len(), 2);
    let total_stages: usize = desc
        .pipelines
        .iter()
        .map(|p| match p {
            Pipeline::Compute(_) => 1,
            Pipeline::MultiCompute(mc) => mc.stages.len(),
            Pipeline::Graphics(_) => 0,
        })
        .sum();
    assert_eq!(total_stages, 4, "1 (consumer compute) + 3 (scan-of-map phases)");
}

/// Every parallel-Map's `Pipeline::Compute` resolves a real `DispatchLen`
/// (no `Fixed{count: 0}` placeholder leaks through). Asserts the
/// option-D resolution applies uniformly across the multi-entry test corpus.
#[test]
fn parallel_pipelines_carry_resolved_dispatch_lens() {
    let srcs = [
        r#"#[compute] entry a(xs: []i32) []i32 = map(|x: i32| x * 2, xs)"#,
        r#"#[compute] entry b() []i32 = map(|x: i32| x + 1, iota(64))"#,
        r#"#[compute] entry c(xs: []i32) []i32 = map(|i: i32| xs[i % 8], iota(length(xs)))"#,
    ];
    for src in srcs {
        let (_, desc) = parallelize_src(src);
        for p in &desc.pipelines {
            match p {
                Pipeline::Compute(cp) => {
                    if let DispatchSize::DerivedFrom { len, .. } = &cp.dispatch_size {
                        assert!(
                            !matches!(len, crate::pipeline_descriptor::DispatchLen::Fixed { count: 0 }),
                            "placeholder Fixed{{0}} leaked: {src:?}"
                        );
                    }
                }
                Pipeline::MultiCompute(mc) => {
                    for s in &mc.stages {
                        if let DispatchSize::DerivedFrom { len, .. } = &s.dispatch_size {
                            assert!(
                                !matches!(len, crate::pipeline_descriptor::DispatchLen::Fixed { count: 0 }),
                                "placeholder Fixed{{0}} leaked in stage {:?}: {src:?}",
                                s.entry_point
                            );
                        }
                    }
                }
                Pipeline::Graphics(_) => {}
            }
        }
    }
}

// =============================================================================
// Stage-count stress tests — see how high we can push pipeline stage counts
// =============================================================================
//
// These pin upper-bound behavior: how many `ComputeStage`s + single `Compute`
// dispatches the descriptor accumulates from various source-program shapes.
// Each test prints the actual breakdown via assert messages so a future
// regression that changes stage counts surfaces the new shape, not just a
// pass/fail.

/// Sum the dispatch stage count across every pipeline in `desc`. A single
/// `Compute` counts as 1 stage; a `MultiCompute` contributes
/// `stages.len()`; graphics pipelines don't count toward compute stages.
fn total_compute_stages(desc: &crate::pipeline_descriptor::PipelineDescriptor) -> usize {
    desc.pipelines
        .iter()
        .map(|p| match p {
            Pipeline::Compute(_) => 1,
            Pipeline::MultiCompute(mc) => mc.stages.len(),
            Pipeline::Graphics(_) => 0,
        })
        .sum()
}

/// Single entry with two independent scans of the same input, both
/// gathered in the tail. Each scan should materialize into its own
/// pipeline (lift_gathers can't fold them together because they're
/// distinct SOAC results). Probes whether two parallel-but-independent
/// scans within one entry both get hoisted.
#[test]
fn single_entry_two_independent_scans_then_gather_yields_many_stages() {
    let src = r#"
        #[compute]
        entry e(xs: []i32) []i32 =
            let a = scan(|x: i32, y: i32| x + y, 0, xs) in
            let b = scan(|x: i32, y: i32| x * y, 1, xs) in
            map(|i: i32| a[i % 256] + b[i % 256], iota(2048))
    "#;
    let (program, desc) = parallelize_src(src);
    let stages = total_compute_stages(&desc);
    let kinds: Vec<_> = desc
        .pipelines
        .iter()
        .map(|p| match p {
            Pipeline::Compute(cp) => format!("Compute({})", cp.entry_point),
            Pipeline::MultiCompute(mc) => format!("MultiCompute(stages={})", mc.stages.len()),
            Pipeline::Graphics(_) => "Graphics".into(),
        })
        .collect();
    assert_eq!(
        desc.pipelines.len(),
        3,
        "1 source entry → 3 pipelines (scan_a + scan_b + gather consumer); got {kinds:?}, \
         compute entries={}",
        compute_entry_count(&program)
    );
    assert_eq!(
        stages, 7,
        "3 + 3 + 1 = 7 stages from one source entry; got {kinds:?}"
    );
}

/// Single entry with a reduce + scan + tail map, where the tail gathers
/// the scan but consumes the reduce as a scalar. **Finding:** only the
/// scan gets hoisted (its array result is gathered); the reduce stays
/// inline in the consumer because `lift_gathers` only extracts SOACs
/// whose results are *indexed*, and a reduce's scalar result isn't.
/// Pins the asymmetry: parallel hoisting follows gather sites, not
/// SOAC-in-let-RHS structure.
#[test]
fn single_entry_only_gathered_soacs_get_hoisted_not_scalar_reduces() {
    let src = r#"
        #[compute]
        entry e(xs: []i32) []i32 =
            let s = reduce(|x: i32, y: i32| x + y, 0, xs) in
            let prefix = scan(|x: i32, y: i32| x + y, 0, xs) in
            map(|i: i32| prefix[i % 256] + s, iota(2048))
    "#;
    let (program, desc) = parallelize_src(src);
    let kinds: Vec<_> = desc
        .pipelines
        .iter()
        .map(|p| match p {
            Pipeline::Compute(cp) => format!("Compute({})", cp.entry_point),
            Pipeline::MultiCompute(mc) => format!("MultiCompute(stages={})", mc.stages.len()),
            Pipeline::Graphics(_) => "Graphics".into(),
        })
        .collect();
    assert_eq!(
        desc.pipelines.len(),
        2,
        "scan hoisted (gathered), reduce stays inline (scalar consumer): {kinds:?}, \
         compute entries={}",
        compute_entry_count(&program)
    );
    assert_eq!(
        total_compute_stages(&desc),
        4,
        "3 (scan) + 1 (consumer with inline reduce) = 4; {kinds:?}"
    );
}

/// Single entry where the tail map gathers *the same* materialized scan
/// twice via two different indexing expressions. If lift_gathers
/// deduplicates the producer, we get one scan pipeline + one consumer
/// (4 stages). If it generates a separate pipeline per gather site, we
/// get two scan pipelines + one consumer (7 stages). Pins which.
#[test]
fn single_entry_two_gathers_of_same_scan_share_one_producer() {
    let src = r#"
        #[compute]
        entry e(xs: []i32) []i32 =
            let prefix = scan(|x: i32, y: i32| x + y, 0, xs) in
            map(|i: i32| prefix[i % 256] + prefix[i % 128], iota(2048))
    "#;
    let (program, desc) = parallelize_src(src);
    let kinds: Vec<_> = desc
        .pipelines
        .iter()
        .map(|p| match p {
            Pipeline::Compute(cp) => format!("Compute({})", cp.entry_point),
            Pipeline::MultiCompute(mc) => format!("MultiCompute(stages={})", mc.stages.len()),
            Pipeline::Graphics(_) => "Graphics".into(),
        })
        .collect();
    assert_eq!(
        desc.pipelines.len(),
        2,
        "two gathers of the same `prefix` should share one producer pipeline; \
         got {kinds:?}, compute entries={}",
        compute_entry_count(&program)
    );
    assert_eq!(
        total_compute_stages(&desc),
        4,
        "3 (scan) + 1 (consumer) = 4 stages; got {kinds:?}"
    );
}

/// Single entry with three independent reduces in let-RHS, all consumed
/// as scalars in the tail map. **Finding:** none of the reduces hoist
/// — the entire entry compiles as a *single non-parallelized* Compute
/// pipeline that loops the reduces inline. Because `lift_gathers` only
/// extracts SOACs feeding gather sites and reduces produce scalars (no
/// gather), three reduces in scalar position get serialized into the
/// kernel body. Concretely demonstrates the limitation: today, scalar
/// SOAC results in let-RHS aren't hoisted to their own pipelines, even
/// when they're independent and would benefit from parallel execution.
#[test]
fn single_entry_scalar_reduces_in_let_rhs_stay_inline_not_hoisted() {
    let src = r#"
        #[compute]
        entry e(xs: []i32) []i32 =
            let s = reduce(|x: i32, y: i32| x + y, 0, xs) in
            let p = reduce(|x: i32, y: i32| x * y, 1, xs) in
            let m = reduce(|x: i32, y: i32| if x > y then x else y, 0, xs) in
            map(|i: i32| s + p + m + i, iota(2048))
    "#;
    let (program, desc) = parallelize_src(src);
    let kinds: Vec<_> = desc
        .pipelines
        .iter()
        .map(|p| match p {
            Pipeline::Compute(cp) => format!("Compute({})", cp.entry_point),
            Pipeline::MultiCompute(mc) => format!("MultiCompute(stages={})", mc.stages.len()),
            Pipeline::Graphics(_) => "Graphics".into(),
        })
        .collect();
    assert_eq!(
        desc.pipelines.len(),
        1,
        "scalar reduces in let-RHS don't hoist; whole entry stays in one pipeline: {kinds:?}, \
         compute entries={}",
        compute_entry_count(&program)
    );
    assert_eq!(
        total_compute_stages(&desc),
        1,
        "single Compute, inline reduces: {kinds:?}"
    );
}

/// `let counts = map(...); let s1 = scan(counts); let s2 = scan(map(|i|
/// s1[i%N], iota(M))); map(|i| s2[i%K], iota(P))` — four nominally
/// stageable SOACs. `fuse_maps` folds `counts` into `s1` (one consumer),
/// then `lift_gathers` lifts both `s1` and `s2` into their own gather
/// pre-passes because each is randomly indexed downstream. From one source
/// entry we get three pipelines (two scan pre-passes + the consumer)
/// totalling stages = `s1`'s MultiCompute phases + `s2`'s serial Compute
/// (its input is `iota` — a Range, not a buffer — so the parallel scan
/// can't route, falling through to a single-thread Compute) + the consumer.
#[test]
fn chained_scans_within_one_entry_collapse_to_three_pipelines() {
    let src = r#"
        #[compute]
        entry gen(bh: []vec4f32) []i32 =
            let counts = map(|h: vec4f32| 4 + 5 * (if h.x > 4.0 then 3 else 1), bh) in
            let s1 = scan(|a: i32, b: i32| a + b, 0, counts) in
            let s2 = scan(|a: i32, b: i32| a + b, 0,
                map(|i: i32| s1[i % 256], iota(6144))) in
            map(|i: i32| s2[i % 128], iota(2048))
    "#;
    let (program, desc) = parallelize_src(src);
    let kinds: Vec<_> = desc
        .pipelines
        .iter()
        .map(|p| match p {
            Pipeline::Compute(cp) => format!("Compute({})", cp.entry_point),
            Pipeline::MultiCompute(mc) => format!("MultiCompute(stages={})", mc.stages.len()),
            Pipeline::Graphics(_) => "Graphics".into(),
        })
        .collect();
    assert_eq!(
        desc.pipelines.len(),
        3,
        "expected 3 pipelines (s1 pre-pass + s2 pre-pass + consumer); got {kinds:?}, \
         compute entries={}",
        compute_entry_count(&program)
    );
}

/// Two independent source entries, each chaining a gather over a scan.
/// Combines per-entry expansion (gather → pre-pass + consumer) with
/// multi-entry summation. Each source entry yields multiple compute
/// entries; the descriptor stages add up across the program.
#[test]
fn two_entries_each_with_scan_then_gather_yield_many_stages() {
    let src = r#"
        #[compute]
        entry a(bh: []vec4f32) []i32 =
            let counts = map(|h: vec4f32| 4 + 5 * (if h.x > 4.0 then 3 else 1), bh) in
            let offsets = scan(|x: i32, y: i32| x + y, 0, counts) in
            map(|i: i32| offsets[i % 256], iota(6144))

        #[compute]
        entry b(bh: []vec4f32) []i32 =
            let counts = map(|h: vec4f32| 2 + 3 * (if h.y > 1.0 then 2 else 1), bh) in
            let offsets = scan(|x: i32, y: i32| x + y, 0, counts) in
            map(|i: i32| offsets[i % 128], iota(2048))
    "#;
    let (program, desc) = parallelize_src(src);
    let stages = total_compute_stages(&desc);
    // Each entry, per `chained_map_scan_gather_yields_two_pipelines_four_stages`,
    // produces 2 pipelines and 4 stages. Two such entries: 4 pipelines, 8 stages.
    assert_eq!(desc.pipelines.len(), 4, "2 entries × 2 pipelines");
    assert_eq!(
        stages,
        8,
        "2 entries × 4 stages each; program has {} compute entries",
        compute_entry_count(&program)
    );
}

// =============================================================================
// Aspirational tests — pin limitations as `#[ignore]`d failing tests
// =============================================================================
//
// These describe behavior we'd *like* parallelize to have but doesn't today.
// Marked `#[ignore]` with a `reason` so they don't fail CI, but `cargo test
// -- --ignored` runs them. When someone teaches the parallelizer to do
// what these expect, the assertions will pass and the `#[ignore]` can come
// off — turning each into a regression guard.

/// **Aspiration:** a scalar SOAC result (here, `reduce`) in let-RHS,
/// consumed inside the tail SOAC, should be hoisted into its own
/// parallel pipeline rather than serialized inline in the consumer.
///
/// **Today:** `lift_gathers` only extracts SOACs that feed *array*
/// indexing (`arr[i]`). A reduce returns a scalar, so it's never gathered
/// → stays inline in the consumer's kernel as a single-thread sequential
/// loop (effectively O(N) per consumer thread; for a map of M threads
/// over a reduce of N inputs, this is M×N work for N total work).
///
/// **Ideal shape:** 2 pipelines (reduce 2-stage + map 1-stage), 3 stages.
#[test]
#[ignore = "scalar reduce in let-RHS is not hoisted — quadratic blowup; needs lift_scalar_soacs (or extension to lift_gathers)"]
fn aspiration_scalar_reduce_in_let_rhs_should_hoist() {
    let src = r#"
        #[compute]
        entry e(xs: []i32) []i32 =
            let s = reduce(|x: i32, y: i32| x + y, 0, xs) in
            map(|i: i32| i + s, iota(2048))
    "#;
    let (_, desc) = parallelize_src(src);
    assert_eq!(
        desc.pipelines.len(),
        2,
        "reduce should hoist into its own pipeline"
    );
    assert_eq!(
        total_compute_stages(&desc),
        3,
        "2 (reduce phase1+phase2) + 1 (consumer map) = 3"
    );
}

/// **Aspiration:** two independent scalar reduces in let-RHS, both
/// consumed in the tail, should each hoist into their own pipeline.
///
/// **Today:** neither reduce hoists; the whole entry compiles as a
/// single non-parallelized Compute that runs both reduces inline.
///
/// **Ideal shape:** 3 pipelines (reduce₁ + reduce₂ + consumer), 5 stages.
#[test]
#[ignore = "multiple scalar reduces in let-RHS stay inline; needs hoisting of non-gathered SOAC results"]
fn aspiration_two_scalar_reduces_in_let_rhs_should_each_hoist() {
    let src = r#"
        #[compute]
        entry e(xs: []i32) []i32 =
            let s = reduce(|x: i32, y: i32| x + y, 0, xs) in
            let p = reduce(|x: i32, y: i32| x * y, 1, xs) in
            map(|i: i32| i + s + p, iota(2048))
    "#;
    let (_, desc) = parallelize_src(src);
    assert_eq!(
        desc.pipelines.len(),
        3,
        "each reduce should hoist into its own pipeline"
    );
    assert_eq!(
        total_compute_stages(&desc),
        5,
        "2 + 2 (reduces) + 1 (consumer) = 5"
    );
}
