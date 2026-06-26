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

use super::analyze_entry;
use super::VarRef;
use crate::ast::{self, Span, TypeName};
use crate::tlc::{
    ArrayExpr, Def, DefMeta, Lambda, LoopKind, ScremaAccumulator, ScremaAccumulatorSpec, ScremaLane,
    SoacBody, SoacDestination, SoacOp, Term, TermId, TermIdSource, TermKind,
};
use crate::{SymbolId, SymbolTable};
use polytype::Type;

// ---------- type helpers ----------

fn input_ae(boxed: Box<crate::tlc::Term>) -> crate::tlc::ArrayExpr {
    use crate::tlc::{ArrayExpr, TermKind};
    let t = *boxed;
    match t.kind {
        TermKind::Var(vr) => ArrayExpr::Var(vr, t.ty),
        TermKind::ArrayExpr(ae) => ae,
        other => panic!("test SOAC input must be a variable or array expr, got {other:?}"),
    }
}

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
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            Type::Constructed(TypeName::Size(8), vec![]),
            crate::types::no_region(),
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

    fn pointwise_screma(&mut self) -> Term {
        let x = self.sym("x");
        let map_body = self.var(x, i32_ty());
        let map_lam = Lambda {
            params: vec![(x, i32_ty())],
            body: Box::new(map_body),
            ret_ty: i32_ty(),
        };
        let start = self.int_lit(0);
        let len = self.int_lit(8);
        let input = ArrayExpr::Range {
            start: Box::new(start),
            len: Box::new(len),
            step: None,
        };
        let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![arr_i32_ty()]);
        self.term(
            TermKind::Soac(SoacOp::Screma {
                lanes: vec![ScremaLane {
                    lam: SoacBody {
                        lam: map_lam,
                        captures: vec![],
                    },
                    input_indices: vec![0],
                }],
                accumulators: vec![],
                inputs: vec![input],
            }),
            tuple_ty,
        )
    }

    fn trivial_screma(&mut self) -> Term {
        let x = self.sym("x");
        let map_body = self.var(x, i32_ty());
        let map_lam = Lambda {
            params: vec![(x, i32_ty())],
            body: Box::new(map_body),
            ret_ty: i32_ty(),
        };

        let acc = self.sym("acc");
        let elem = self.sym("elem");
        let step_body = self.var(acc, i32_ty());
        let step_lam = Lambda {
            params: vec![(acc, i32_ty()), (elem, i32_ty())],
            body: Box::new(step_body.clone()),
            ret_ty: i32_ty(),
        };

        let start = self.int_lit(0);
        let len = self.int_lit(8);
        let input = ArrayExpr::Range {
            start: Box::new(start),
            len: Box::new(len),
            step: None,
        };
        let ne = self.int_lit(0);
        let tuple_ty = Type::Constructed(TypeName::Tuple(2), vec![arr_i32_ty(), i32_ty()]);
        self.term(
            TermKind::Soac(SoacOp::Screma {
                lanes: vec![ScremaLane {
                    lam: SoacBody {
                        lam: map_lam,
                        captures: vec![],
                    },
                    input_indices: vec![0],
                }],
                accumulators: vec![ScremaAccumulatorSpec {
                    kind: ScremaAccumulator::Reduce,
                    step_lam: SoacBody {
                        lam: step_lam.clone(),
                        captures: vec![],
                    },
                    reduce_op: SoacBody {
                        lam: step_lam,
                        captures: vec![],
                    },
                    ne: Box::new(ne),
                }],
                inputs: vec![input],
            }),
            tuple_ty,
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

#[test]
fn t1b_tail_pointwise_screma_is_parallel_planned() {
    let mut b = B::new();
    let p = b.sym("p");
    let body = b.pointwise_screma();
    let name = b.sym("entry");
    let def = b.entry_def(name, vec![(p, i32_ty())], body);
    let a = analyze_entry(&def, &b.symbols).expect("pointwise Screma should plan like multi-output map");
    assert!(a.prefix_lets.is_empty());
    assert!(matches!(a.soac.original, SoacOp::Screma { accumulators, .. } if accumulators.is_empty()));
}

#[test]
fn t1c_tail_mixed_screma_is_recognised_by_analysis() {
    let mut b = B::new();
    let p = b.sym("p");
    let body = b.trivial_screma();
    let name = b.sym("entry");
    let def = b.entry_def(name, vec![(p, i32_ty())], body);
    let a = analyze_entry(&def, &b.symbols)
        .expect("mixed Screma (map + reduce accumulator) should now be recognised by the analyzer");
    assert!(a.prefix_lets.is_empty());
    assert!(matches!(
        a.soac.original,
        SoacOp::Screma { ref accumulators, .. } if !accumulators.is_empty()
    ));
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
    assert_eq!(a.required_params[0].sym, q);
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
use crate::pipeline_descriptor::{ComputePipeline, DispatchSize, Pipeline};

/// Run `src` through the canonical TLC pipeline including `parallelize_soacs`,
/// returning the parallelized program + pipeline descriptor.
fn parallelize_src(
    src: &str,
) -> (
    crate::tlc::Program,
    crate::pipeline_descriptor::PipelineDescriptor,
) {
    let tlc = crate::test_pipeline::compile_to_reachable(src, false);
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
        Pipeline::Compute(ComputePipeline { stages, .. })
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

#[test]
fn ordered_clear_scatter_suffix_serializes_entry_until_multi_dispatch_scheduler() {
    let src = r#"
        #[compute]
        entry tick(xs: []i32, fb: *[]i32) []i32 =
            let out = map(|x: i32| x + 1, xs) in
            let cleared = map(|_p: i32| 0, fb) in
            let _ = scatter(cleared, [0i32], [1i32]) in
            out
    "#;
    let (program, desc) = parallelize_src(src);
    assert_eq!(
        compute_entry_count(&program),
        3,
        "clear + scatter are extracted as ordered dispatches before the tail map"
    );
    let stages = desc
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(mc) if mc.stages.iter().any(|s| s.entry_point == "tick") => Some(&mc.stages),
            _ => None,
        })
        .expect("tick should be scheduled as an ordered multi-dispatch pipeline");
    assert_eq!(stages.len(), 3, "clear, scatter, then tail map");
    assert!(
        matches!(stages[0].dispatch_size, DispatchSize::DerivedFrom { .. }),
        "the clear map should dispatch over the framebuffer"
    );
    assert_eq!(
        stages[1].dispatch_size,
        DispatchSize::Fixed { x: 1, y: 1, z: 1 },
        "scatter is still serial until the parallel scatter lowering lands"
    );
    assert!(
        matches!(stages[2].dispatch_size, DispatchSize::DerivedFrom { .. }),
        "the tail map should remain lane-parallel"
    );
}

#[test]
fn ordered_clear_scatter_suffix_serializes_reduce_until_multi_dispatch_scheduler() {
    let src = r#"
        #[compute]
        entry tick(xs: []i32, fb: *[]i32) i32 =
            let sum = reduce(|a: i32, b: i32| a + b, 0, xs) in
            let cleared = map(|_p: i32| 0, fb) in
            let _ = scatter(cleared, [0i32], [1i32]) in
            sum
    "#;
    let (_program, desc) = parallelize_src(src);
    let stages = desc
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(mc) if mc.stages.iter().any(|s| s.entry_point == "tick") => Some(&mc.stages),
            _ => None,
        })
        .expect("tick should be scheduled as ordered dispatches plus reduce phases");
    assert_eq!(stages.len(), 4, "clear, scatter, reduce phase 1, reduce phase 2");
    assert!(matches!(
        stages[0].dispatch_size,
        DispatchSize::DerivedFrom { .. }
    ));
    assert_eq!(stages[1].dispatch_size, DispatchSize::Fixed { x: 1, y: 1, z: 1 });
    assert!(stages[2].entry_point == "tick" || stages[2].entry_point.contains("phase1"));
    assert!(stages[3].entry_point.contains("phase2"));
}

#[test]
fn ordered_prefix_partitions_pre_tail_and_post_tail_dispatches() {
    let src = r#"
        #[compute]
        entry tick(xs: []i32, fb: *[]i32) []i32 =
            let out = map(|x: i32| x + 1, xs) in
            let cleared = map(|_p: i32| 0, fb) in
            let idxs = map(|p: i32| p, out) in
            let vals = map(|p: i32| p, out) in
            let _ = scatter(cleared, idxs, vals) in
            out
    "#;
    let (_program, desc) = parallelize_src(src);
    let stages = desc
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(mc) if mc.stages.iter().any(|s| s.entry_point == "tick") => Some(&mc.stages),
            _ => None,
        })
        .expect("tick should be an ordered multi-dispatch pipeline");
    assert_eq!(stages.len(), 3, "clear, tail map, post-tail scatter");
    assert_ne!(stages[0].entry_point, "tick");
    assert_eq!(stages[1].entry_point, "tick");
    assert_ne!(stages[2].entry_point, "tick");
    assert!(
        matches!(stages[0].dispatch_size, DispatchSize::DerivedFrom { .. }),
        "pre-tail clear dispatches over fb"
    );
    assert!(
        matches!(stages[1].dispatch_size, DispatchSize::DerivedFrom { .. }),
        "tail map dispatches over xs"
    );
    assert!(
        matches!(stages[2].dispatch_size, DispatchSize::DerivedFrom { .. }),
        "post-tail scatter dispatches over the materialized tail output"
    );
}

#[test]
fn unsupported_scan_fallback_dispatches_single_lane() {
    let src = r#"
        def add_with(k: i32, xs: []i32) []i32 =
            scan(|a: i32, b: i32| a + b + k, 0, xs)

        #[compute]
        entry tick(xs: []i32, k: i32) []i32 =
            add_with(k, xs)
    "#;
    let (_program, desc) = parallelize_src(src);
    let tick = desc
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(cp) if cp.stages.iter().any(|s| s.entry_point == "tick") => Some(cp),
            _ => None,
        })
        .expect("capturing scan should stay as the original compute entry");
    let stage = tick.stages.iter().find(|s| s.entry_point == "tick").expect("tick stage");
    assert_eq!(
        stage.dispatch_size,
        DispatchSize::Fixed { x: 1, y: 1, z: 1 },
        "a serial scan fallback must not run once per input element"
    );
}

/// `reduce(+, 0, map(f, xs))` becomes a fused map→reduce (a single-accumulator
/// Screma), which lowers to a two-phase
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

/// A shared producer feeding both a map and a reduce fuses to a Screma
/// (map outputs + 1 reduce accumulator). After Step 2 the Screma path
/// emits the same two-stage shape as a plain reduce — phase 1 chunks
/// the input and the map output view + writes the reduce partial to
/// `partials[tid]`, phase 2 tree-reduces partials into the result.
#[test]
fn screma_fused_map_and_reduce_becomes_two_phase_multi_compute() {
    let src = r#"
        #[compute]
        entry gen(xs: []i32) ([]i32, [1]i32) =
          let b = map(|x: i32| x + 1, xs) in
          let c = map(|y: i32| y * 2, b) in
          let d = reduce(|acc: i32, z: i32| acc + z, 0, b) in
          (c, [d])
    "#;
    let (_program, desc) = parallelize_src(src);
    let stages = multi_stage_count(&desc, "gen")
        .or_else(|| multi_stage_count(&desc, "gen_phase2_combine"))
        .expect("fused map+reduce Screma should produce a multi-compute pipeline");
    assert_eq!(stages, 2, "Screma reduce-accumulator: phase1 + phase2");
}

/// A shared producer feeding both a map and a scan fuses to a Screma
/// (map outputs + 1 scan accumulator). After Step 3 the Screma path
/// emits the same three-stage shape as the plain Scan path — phase 1
/// chunks the input + map output view + scan output view + writes the
/// per-thread final accumulator to `block_sums[tid]`; phase 2
/// sequentially scans block_sums into block_offsets; phase 3 applies
/// each chunk's offset back over the scan output.
#[test]
fn screma_fused_map_and_scan_becomes_three_stage_multi_compute() {
    let src = r#"
        #[compute]
        entry gen(xs: []i32) ([]i32, []i32) =
          let b = map(|x: i32| x + 1, xs) in
          let c = map(|y: i32| y * 2, b) in
          let d = scan(|acc: i32, z: i32| acc + z, 0, b) in
          (c, d)
    "#;
    let (_program, desc) = parallelize_src(src);
    let stages = multi_stage_count(&desc, "gen")
        .or_else(|| multi_stage_count(&desc, "gen_phase2_scan_sums"))
        .or_else(|| multi_stage_count(&desc, "gen_phase3_add_offsets"))
        .expect("fused map+scan Screma should produce a multi-compute pipeline");
    assert_eq!(stages, 3, "Screma scan-accumulator: phase1 + phase2 + phase3");
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
        desc.pipelines.iter().any(|p| matches!(p, Pipeline::Compute(_))),
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
            Pipeline::Compute(cp) => cp.stages.len(),
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
                    for s in &cp.stages {
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

/// Sum the dispatch stage count across every pipeline in `desc`.
/// Each `Compute` contributes `stages.len()`; graphics pipelines don't
/// count toward compute stages.
fn total_compute_stages(desc: &crate::pipeline_descriptor::PipelineDescriptor) -> usize {
    desc.pipelines
        .iter()
        .map(|p| match p {
            Pipeline::Compute(cp) => cp.stages.len(),
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
            Pipeline::Compute(cp) => format!("Compute(stages={})", cp.stages.len()),
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

/// Single entry with a reduce + scan + tail map, where the tail gathers the
/// scan and consumes the reduce as a captured scalar. Both hoist: the scan via
/// `lift_gathers` (its array result is indexed), the reduce via
/// `lift_compute_scalar_reduces` (the planner marks the captured scalar
/// `ScalarBroadcast`). 3 pipelines (scan + reduce + consumer), 6 stages
/// (3 scan + 2 reduce + 1 consumer).
#[test]
fn single_entry_scan_and_scalar_reduce_both_hoist() {
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
            Pipeline::Compute(cp) => format!("Compute(stages={})", cp.stages.len()),
            Pipeline::Graphics(_) => "Graphics".into(),
        })
        .collect();
    assert_eq!(
        desc.pipelines.len(),
        3,
        "scan hoisted (gathered) + reduce hoisted (ScalarBroadcast) + consumer: {kinds:?}, \
         compute entries={}",
        compute_entry_count(&program)
    );
    assert_eq!(
        total_compute_stages(&desc),
        6,
        "3 (scan) + 2 (reduce) + 1 (consumer) = 6; {kinds:?}"
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
            Pipeline::Compute(cp) => format!("Compute(stages={})", cp.stages.len()),
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

/// Single entry with three independent reduces in let-RHS, all consumed as
/// scalars in the tail map. Each reduce is captured into the consumer lambda
/// (the lane var `i` is out of scope at the lets), so the planner marks all
/// three `ScalarBroadcast` and `lift_compute_scalar_reduces` hoists each into
/// its own two-phase pre-pass: 4 pipelines (3 reduces + consumer), 7 stages
/// (2 + 2 + 2 + 1).
#[test]
fn single_entry_scalar_reduces_in_let_rhs_each_hoist() {
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
            Pipeline::Compute(cp) => format!("Compute(stages={})", cp.stages.len()),
            Pipeline::Graphics(_) => "Graphics".into(),
        })
        .collect();
    assert_eq!(
        desc.pipelines.len(),
        4,
        "3 reduce pre-passes + consumer map: {kinds:?}, compute entries={}",
        compute_entry_count(&program)
    );
    assert_eq!(
        total_compute_stages(&desc),
        7,
        "2 + 2 + 2 (reduces) + 1 (consumer) = 7: {kinds:?}"
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
            Pipeline::Compute(cp) => format!("Compute(stages={})", cp.stages.len()),
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
// Scalar-reduce hoisting (producer-consumer planner ScalarBroadcast)
// =============================================================================
//
// A scalar SOAC result (`reduce` or fused `map→reduce`) bound in a top-level let and
// consumed *per element* inside the tail SOAC's operator lambda is marked
// `StoragePrepass(ScalarBroadcast)` by the producer-consumer planner and
// hoisted by `lift_compute_scalar_reduces` into its own two-phase reduce
// pre-pass — instead of being serialized inline in the consumer kernel (which
// would run the O(N) reduce once per consumer thread).

/// One captured scalar reduce → 2 pipelines (reduce 2-stage + consumer map
/// 1-stage), 3 stages. The lane variable `i` is the consumer lambda's
/// parameter, out of scope at `let s`, so the reduce is invariant across the
/// map and the hoist is sound.
#[test]
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

/// Two independent captured scalar reduces → each hoists into its own
/// two-phase pre-pass: 3 pipelines (reduce₁ + reduce₂ + consumer), 5 stages.
#[test]
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

// ----- TLC-side two-phase reduce synthesis -----

/// Compile `src` end-to-end through SPIR-V via the canonical TLC + EGIR
/// + SSA pipeline. Used by tests that exercise the TLC-side two-phase
/// reduce synthesis path — the bugs there surface as EGIR-conversion
/// errors that `parallelize_src` alone wouldn't observe.
fn compile_to_spirv(src: &str) -> crate::error::Result<Vec<u32>> {
    crate::compile_thru_spirv(src)
        .map(|l| l.spirv)
        .map_err(|e| crate::error::CompilerError::FlatteningError(format!("{e:?}"), None))
}

/// Single-output reduce whose combiner captures a uniform — forces the
/// TLC-side two-phase synthesis (`can_route = false` because
/// `captures` is non-empty). The synthesized phase entries must
/// re-attach the original storage / uniform attributes to the captured
/// params; otherwise EGIR's push-constant fallback rejects the
/// runtime-sized `xs` array.
///
/// Regression: prior to the fix, `make_entry_def` set
/// `param_bindings: vec![None; n]` and emitted bare patterns with no
/// `#[storage]` / `#[uniform]` attribute. Compilation failed with
/// `push-constant param 'xs' has no static byte layout`.
#[test]
fn two_phase_reduce_with_capture_preserves_param_attributes() {
    let src = r#"
#[compute]
entry foo(#[storage(set=1, binding=0)] xs: []i32,
          #[uniform(set=2, binding=0)] k: i32) i32 =
    reduce(|a:i32, b:i32| a + b + k, 0, xs)
"#;
    compile_to_spirv(src).expect("two-phase reduce with capture must compile end-to-end");
}

/// Demonstration: a multi-output two-phase reduce must preserve every
/// declared output slot through phase synthesis. `analyze_entry`'s
/// slot-0 short-circuit at `parallelize.rs` used to descend into slot
/// 0's value and discard the rest of the let-chain (which carries
/// slots 1..N's `OutputSlotStore` terms). `build_two_phase_entries`
/// then built phase1/phase2 from an `EntryAnalysis` that knew nothing
/// about extra slots, the original entry was removed from
/// `program.defs`, and the replacement phase entries declared no
/// output for slot 1.
///
/// The original entry declares 2 outputs (the reduce result + `42`).
/// After parallelize, the program's compute entries should collectively
/// declare 2 `StorageRole::Output` bindings — one per slot. Pre-fix
/// they declared exactly 1.
#[test]
fn two_phase_reduce_preserves_extra_slot_value() {
    let src = r#"
#[compute]
entry foo(#[storage(set=1, binding=0)] xs: []i32,
          #[uniform(set=2, binding=0)] k: i32) (i32, i32) =
    (reduce(|a:i32, b:i32| a + b + k, 0, xs), 42)
"#;
    let (program, _desc) = parallelize_src(src);

    let total_outputs: usize = program
        .defs
        .iter()
        .filter_map(|d| match &d.meta {
            DefMeta::EntryPoint(e) if e.entry_type.is_compute() => Some(
                e.storage_bindings
                    .iter()
                    .filter(|b| matches!(b.role, crate::interface::StorageRole::Output))
                    .count(),
            ),
            _ => None,
        })
        .sum();
    assert_eq!(
        total_outputs, 2,
        "multi-output entry must produce one Output binding per slot \
         (regression: two-phase synthesis drops extra slot stores)"
    );

    // End-to-end: slot 1's literal 42 must survive into SPIR-V as an
    // `OpConstant` (opcode 43) with a word-count of 4 and a literal
    // value 42. The 4-word `OpConstant` instruction starts with
    // `(4 << 16) | 43 = 0x0004002B`, so scan for any window of 4 words
    // where the first is that encoding and the last is 42.
    let spirv = compile_to_spirv(src).expect("multi-output two-phase reduce must lower to SPIR-V");
    let op_constant_42_header: u32 = (4u32 << 16) | 43;
    let has_constant_42 = spirv.windows(4).any(|w| w[0] == op_constant_42_header && w[3] == 42);
    assert!(
        has_constant_42,
        "expected OpConstant 42 in SPIR-V (slot 1's literal 42 missing)"
    );
}

/// Regression: a compute entry that mixes an explicit `#[storage]`
/// view-array param with a `#[storage_image]` param must dispatch
/// per-element of the view-array, not per-texel of the image. The
/// auto-allocator skips host-wired storage params; before the fix
/// the dispatch then fell through to the storage_image and ran
/// W*H threads instead of N.
#[test]
fn explicit_storage_view_array_drives_dispatch_over_storage_image() {
    use crate::pipeline_descriptor::DispatchLen;
    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] prev: []vec4f32,
                   #[storage_image(set=0, binding=0, format=rgba8unorm, access=write_only)] img: storage_image)
          []vec4f32 =
            map(|p: vec4f32|
                  let _ = image_store(img, @[i32.f32(p.x), i32.f32(p.y)], @[1.0, 1.0, 1.0, 1.0]) in
                  p,
                prev)
    "#;
    let (_program, desc) = parallelize_src(src);
    let len = desc.pipelines.iter().find_map(|p| match p {
        Pipeline::Compute(cp) => cp.stages.iter().find_map(|s| {
            if s.entry_point == "tick" {
                if let DispatchSize::DerivedFrom { len, .. } = &s.dispatch_size {
                    return Some(len.clone());
                }
            }
            None
        }),
        _ => None,
    });
    assert_eq!(
        len,
        Some(DispatchLen::InputBinding {
            set: 2,
            binding: 0,
            elem_bytes: 16,
        }),
        "tick's dispatch must derive from prev (set=2, binding=0), not the storage_image"
    );
}

// ----- Source buffer names survive parallelization -----
//
// The parallel path names storage inputs positionally (`input_0`, …); the
// relabel pass in `to_egraph` restores the source-parameter names so viz's
// `--feedback`/`--buffer-init` can reference buffers by the names the user
// wrote. These go through `compile_thru_ssa` (the real mainline) because the
// relabel only runs at the EGIR boundary, not at the TLC `parallelize_src`
// stage above.

/// Every `Input` storage buffer in the descriptor as `((set, binding), name)`.
fn input_storage_names(desc: &crate::pipeline_descriptor::PipelineDescriptor) -> Vec<((u32, u32), String)> {
    use crate::pipeline_descriptor::{Binding, BufferUsage};
    desc.pipelines
        .iter()
        .flat_map(|p| match p {
            Pipeline::Compute(cp) => cp.bindings.iter().collect::<Vec<_>>(),
            Pipeline::Graphics(gp) => gp.bindings.iter().collect::<Vec<_>>(),
        })
        .filter_map(|b| match b {
            Binding::StorageBuffer {
                set,
                binding,
                usage: BufferUsage::Input,
                name,
                ..
            } => Some(((*set, *binding), name.clone())),
            _ => None,
        })
        .collect()
}

#[test]
fn parallelized_compute_keeps_source_input_buffer_name() {
    let src = r#"
        #[compute]
        entry sim(#[storage(set=2, binding=0, access=read)] prev_pos: []f32) []f32 =
            map(|p: f32| p + 1.0, prev_pos)
    "#;
    let converted = crate::compile_thru_ssa(src).expect("compile thru ssa");
    let names = input_storage_names(&converted.pipeline);
    assert!(
        names.contains(&((2, 0), "prev_pos".to_string())),
        "input (2,0) must be named `prev_pos`, got {names:?}"
    );
    assert!(
        !names.iter().any(|(_, n)| n.starts_with("input_")),
        "no fabricated `input_N` names should survive, got {names:?}"
    );
}

#[test]
fn parallelized_compute_keeps_auto_set_input_buffer_name() {
    // No explicit `#[storage]` attribute: the param lands at the auto storage
    // set (0), binding 0. Its source name `xs` must still survive.
    let src = r#"
        #[compute]
        entry e(xs: []i32) []i32 = map(|x: i32| x + 1, xs)
    "#;
    let converted = crate::compile_thru_ssa(src).expect("compile thru ssa");
    let names = input_storage_names(&converted.pipeline);
    assert!(
        names.contains(&((0, 0), "xs".to_string())),
        "input (0,0) must be named `xs`, got {names:?}"
    );
    assert!(
        !names.iter().any(|(_, n)| n.starts_with("input_")),
        "no fabricated `input_N` names should survive, got {names:?}"
    );
}

#[test]
fn parallelized_compute_keeps_all_source_input_names() {
    // Particles' `sim` shape: several named storage inputs read element-wise
    // through the parallel path. Each keeps its source name (none positional).
    let src = r#"
        #[compute]
        entry sim(#[storage(set=2, binding=0, access=read)] prev_pos: []f32,
                  #[storage(set=2, binding=1, access=read)] seed: []f32) []f32 =
            let prev_pos = prev_pos[0..4] in
            let seed = seed[0..4] in
            map(|i: i32| prev_pos[i] + seed[i], 0i32..<4)
    "#;
    let converted = crate::compile_thru_ssa(src).expect("compile thru ssa");
    let names = input_storage_names(&converted.pipeline);
    assert!(
        names.contains(&((2, 0), "prev_pos".to_string())),
        "input (2,0) must be named `prev_pos`, got {names:?}"
    );
    assert!(
        names.contains(&((2, 1), "seed".to_string())),
        "input (2,1) must be named `seed`, got {names:?}"
    );
    assert!(
        !names.iter().any(|(_, n)| n.starts_with("input_")),
        "no fabricated `input_N` names should survive, got {names:?}"
    );
}

#[test]
fn ordered_schedule_output_uses_entry_output_name() {
    // The ordered (clear + scatter + tail-map) schedule names its host-facing
    // output `{entry}_output`, matching the non-parallel convention — not the
    // generic `tail_output`. `tick` returns the tail map's result.
    use crate::pipeline_descriptor::{Binding, BufferUsage};
    let src = r#"
        #[compute]
        entry tick(xs: []i32, fb: *[]i32) []i32 =
            let out = map(|x: i32| x + 1, xs) in
            let cleared = map(|_p: i32| 0, fb) in
            let _ = scatter(cleared, [0i32], [1i32]) in
            out
    "#;
    let converted = crate::compile_thru_ssa(src).expect("compile thru ssa");
    let outputs: Vec<String> = converted
        .pipeline
        .pipelines
        .iter()
        .flat_map(|p| match p {
            Pipeline::Compute(cp) => cp.bindings.iter().collect::<Vec<_>>(),
            Pipeline::Graphics(gp) => gp.bindings.iter().collect::<Vec<_>>(),
        })
        .filter_map(|b| match b {
            Binding::StorageBuffer {
                usage: BufferUsage::Output,
                name,
                ..
            } => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert!(
        outputs.contains(&"tick_output".to_string()),
        "ordered tail output must be `tick_output`, got {outputs:?}"
    );
    assert!(
        !outputs.iter().any(|n| n == "tail_output"),
        "the generic `tail_output` name must not survive, got {outputs:?}"
    );
}

// ---------- domain-key / partition helpers ----------
//
// `soac_domain_key` + `partition_by_domain` identify which output slots
// share an iteration domain (and so may be horizontally fused). The split
// in `make_multidomain_map_plan` is correct regardless of partitioning;
// these helpers drive the equal-domain fusion that collapses same-domain
// stages into one guarded kernel.

fn size_ty(n: usize) -> Type<TypeName> {
    Type::Constructed(TypeName::Size(n), vec![])
}

#[test]
fn partition_groups_equal_size_terms() {
    let keys = vec![Some(size_ty(8)), Some(size_ty(8)), Some(size_ty(4))];
    assert_eq!(super::partition_by_domain(&keys), vec![vec![0, 1], vec![2]]);
}

#[test]
fn partition_distinguishes_size_variables_by_id() {
    let keys = vec![
        Some(Type::Variable(1)),
        Some(Type::Variable(2)),
        Some(Type::Variable(1)),
    ];
    assert_eq!(super::partition_by_domain(&keys), vec![vec![0, 2], vec![1]]);
}

#[test]
fn partition_treats_unknown_domain_as_singleton() {
    let keys = vec![Some(size_ty(8)), None, Some(size_ty(8)), None];
    assert_eq!(
        super::partition_by_domain(&keys),
        vec![vec![0, 2], vec![1], vec![3]]
    );
}

#[test]
fn soac_domain_key_reads_primary_input_size() {
    // `map(|x| x, ref: [8]i32)` — the domain key is the input's outer size.
    let mut b = B::new();
    let x = b.sym("x");
    let body = b.var(x, i32_ty());
    let arr = b.sym("arr");
    let input_ref = b.var(arr, arr_i32_ty());
    let map = SoacOp::Map {
        lam: SoacBody {
            lam: Lambda {
                params: vec![(x, i32_ty())],
                body: Box::new(body),
                ret_ty: i32_ty(),
            },
            captures: vec![],
        },
        inputs: vec![input_ae(Box::new(input_ref))],
        destination: SoacDestination::Fresh,
    };
    assert_eq!(super::soac_domain_key(&map), Some(size_ty(8)));
}
