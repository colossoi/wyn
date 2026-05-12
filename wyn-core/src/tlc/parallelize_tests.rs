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
use crate::tlc::SoacBody;
use crate::tlc::{ArrayExpr, Def, DefMeta, Lambda, LoopKind, SoacOp, Term, TermId, TermIdSource, TermKind};
use crate::{SymbolId, SymbolTable};
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
                consumes_input: false,
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

/// T6: Force wrapper — transparent.
#[test]
fn t6_force_wrapper() {
    let mut b = B::new();
    let p = b.sym("p");
    let soac = b.trivial_map();
    let soac_ty = soac.ty.clone();
    let body = b.term(TermKind::Force(Box::new(soac)), soac_ty);
    let name = b.sym("entry");
    let def = b.entry_def(name, vec![(p, i32_ty())], body);
    let a = analyze_entry(&def, &b.symbols).expect("should find SOAC through Force");
    assert!(a.prefix_lets.is_empty());
    assert!(matches!(a.soac.original, SoacOp::Map { .. }));
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
            consumes_input: false,
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
        uniforms: vec![],
        storage: vec![],
        symbols: std::mem::replace(&mut b.symbols, SymbolTable::new()),
        def_syms: HashMap::new(),
    }
}

/// Binding registry finds `ArrayExpr::StorageBuffer` nested inside a
/// SOAC input — the case `collect_program_resource_bindings` misses
/// and that caused the scan/reduce allocator to collide with input
/// buffers before PLAN_scan_stage_b.md was written.
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
    let input = ArrayExpr::StorageBuffer {
        set: 0,
        binding: 3,
        offset: Box::new(offset),
        len: Box::new(len),
        elem_ty: i32_ty(),
    };
    let soac = b.term(
        TermKind::Soac(SoacOp::Map {
            lam: SoacBody {
                lam,
                captures: vec![],
            },
            inputs: vec![input],
            consumes_input: false,
        }),
        arr_i32_ty(),
    );

    let program = program_wrapping_body(&mut b, soac);
    let used = collect_all_used_bindings(&program);
    assert!(
        used.contains(&(0, 3)),
        "expected (0, 3) in used bindings; got {:?}",
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
        ArrayExpr::StorageBuffer {
            set: 0,
            binding,
            offset: Box::new(offset),
            len: Box::new(len),
            elem_ty: i32_ty(),
        }
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
            consumes_input: false,
        }),
        arr_i32_ty(),
    );

    let program = program_wrapping_body(&mut b, soac);
    let used = collect_all_used_bindings(&program);
    assert!(used.contains(&(0, 5)), "missing (0, 5): {:?}", used);
    assert!(used.contains(&(0, 7)), "missing (0, 7): {:?}", used);
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
