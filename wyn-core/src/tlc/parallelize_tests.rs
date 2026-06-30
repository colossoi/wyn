//! Recognition tests for `analyze_entry` — the tail-position walk that
//! finds the top-level SOAC of a compute entry point and returns its
//! recognized `SoacAnalysis`.
//!
//! The tests exercise the ways a SOAC can sit at the tail of an entry
//! body: a naked Map/Screma, through Let chains, alias lets
//! (`let x = soac in x`), chained aliases, and a nested let in the RHS,
//! plus a lambda closing over an entry param. They also pin the
//! rejections for non-tail positions (If, Loop, App) and a plain
//! literal tail, where `analyze_entry` returns `None`.

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
    assert!(matches!(
        a.soac.original,
        SoacOp::Screma { ref accumulators, .. } if !accumulators.is_empty()
    ));
}

/// T2: Lambda + deep Let chain + Soac. The SOAC is still recognised at the
/// tail through a chain of let bindings.
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
    assert!(matches!(a.soac.original, SoacOp::Map { .. }));
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
    assert!(matches!(a.soac.original, SoacOp::Map { .. }));
}

/// T5: `let x = (let y = 1 in soac) in x` — the SOAC is recognised through
/// the alias and the nested let in the RHS.
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

/// T11: SOAC lambda references an entry param — the map is still recognised
/// at the tail when its lambda captures an outer param.
#[test]
fn t11_closure_over_entry_param() {
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
    assert!(matches!(a.soac.original, SoacOp::Map { .. }));
}
