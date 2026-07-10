//! Pre-defunctionalization normalization for runtime-indexed nested producers.
//!
//! Static-index fusion handles `map(f, xs)[3]` by computing one scalar element.
//! The runtime-index counterpart has no scalar fused form, so make it look like
//! the ordinary gather shape while lambdas are still local:
//!
//! ```text
//! map(|i| (map(f, xs))[i], is)
//!   -> let _runtime_gather = map(f, xs) in
//!      map(|i| _runtime_gather[i], is)
//! ```
//!
//! This is deliberately before defunctionalization. After defunc the index may
//! live in a generated operator def while the producer survives only as a SOAC
//! capture, which is exactly the interprocedural rewrite this pass avoids.

use crate::LookupSet;

use polytype::Type;

use crate::ast::TypeName;
use crate::SymbolId;

use super::{ArrayExpr, Lambda, Program, SoacBody, SoacOp, Term, TermIdSource, TermKind, VarRef};

#[derive(Debug)]
struct Binding {
    name: SymbolId,
    name_ty: Type<TypeName>,
    rhs: Term,
}

pub fn run(program: &mut Program) {
    let mut ids = TermIdSource::new();
    let blocked = LookupSet::new();

    for idx in 0..program.defs.len() {
        let body = program.defs[idx].body.clone();
        let (floats, body) = float_term(body, &blocked, &mut ids, &mut program.symbols, false);
        program.defs[idx].body = wrap_lets(floats, body, &mut ids);
    }

    super::anf::debug_check(&program, "runtime_index_producers");
}

fn float_term(
    term: Term,
    blocked: &LookupSet<SymbolId>,
    ids: &mut TermIdSource,
    symbols: &mut crate::SymbolTable,
    collect: bool,
) -> (Vec<Binding>, Term) {
    let Term { id, ty, span, kind } = term;
    match kind {
        TermKind::Lambda(lam) => {
            let mut inner_blocked = blocked.clone();
            for (sym, _) in &lam.params {
                inner_blocked.insert(*sym);
            }
            let (floats, body) = float_term(*lam.body, &inner_blocked, ids, symbols, false);
            let body = wrap_lets(floats, body, ids);
            (
                vec![],
                Term {
                    id,
                    ty,
                    span,
                    kind: TermKind::Lambda(Lambda {
                        params: lam.params,
                        body: Box::new(body),
                        ret_ty: lam.ret_ty,
                    }),
                },
            )
        }
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            let (mut rhs_floats, rhs) = float_term(*rhs, blocked, ids, symbols, collect);
            let mut inner_blocked = blocked.clone();
            inner_blocked.insert(name);
            let (mut body_floats, body) = float_term(*body, &inner_blocked, ids, symbols, collect);

            if collect {
                rhs_floats.append(&mut body_floats);
                (
                    rhs_floats,
                    Term {
                        id,
                        ty,
                        span,
                        kind: TermKind::Let {
                            name,
                            name_ty,
                            rhs: Box::new(rhs),
                            body: Box::new(body),
                        },
                    },
                )
            } else {
                (
                    vec![],
                    Term {
                        id,
                        ty,
                        span,
                        kind: TermKind::Let {
                            name,
                            name_ty,
                            rhs: Box::new(wrap_lets(rhs_floats, rhs, ids)),
                            body: Box::new(wrap_lets(body_floats, body, ids)),
                        },
                    },
                )
            }
        }
        TermKind::OutputSlotStore { slot_index, value } => {
            let (floats, value) = float_term(*value, blocked, ids, symbols, true);
            (
                vec![],
                Term {
                    id,
                    ty,
                    span,
                    kind: TermKind::OutputSlotStore {
                        slot_index,
                        value: Box::new(wrap_lets(floats, value, ids)),
                    },
                },
            )
        }
        TermKind::Soac(soac) => {
            let (floats, soac) = float_soac(soac, blocked, ids, symbols);
            let soac_term = Term {
                id,
                ty,
                span,
                kind: TermKind::Soac(soac),
            };
            finish(floats, soac_term, collect, ids)
        }
        TermKind::ArrayExpr(ae) => {
            let (floats, ae) = float_array_expr(ae, blocked, ids, symbols);
            let ae_term = Term {
                id,
                ty,
                span,
                kind: TermKind::ArrayExpr(ae),
            };
            finish(floats, ae_term, collect, ids)
        }
        TermKind::Index { array, index } => {
            let (mut index_floats, index) = float_term(*index, blocked, ids, symbols, collect);

            if !is_int_lit(&index)
                && is_runtime_sized_array(&array.ty)
                && is_liftable_array_producer(&array)
                && !references_any(&array, blocked)
            {
                let name = symbols.alloc("_runtime_gather".to_string());
                let name_ty = array.ty.clone();
                let array_span = array.span;
                let var = Term {
                    id: ids.next_id(),
                    ty: name_ty.clone(),
                    span: array_span,
                    kind: TermKind::Var(VarRef::Symbol(name)),
                };
                let indexed = Term {
                    id,
                    ty,
                    span,
                    kind: TermKind::Index {
                        array: Box::new(var),
                        index: Box::new(index),
                    },
                };
                index_floats.push(Binding {
                    name,
                    name_ty,
                    rhs: *array,
                });
                return finish(index_floats, indexed, collect, ids);
            }

            let (mut array_floats, array) = float_term(*array, blocked, ids, symbols, collect);
            index_floats.append(&mut array_floats);
            let indexed = Term {
                id,
                ty,
                span,
                kind: TermKind::Index {
                    array: Box::new(array),
                    index: Box::new(index),
                },
            };
            finish(index_floats, indexed, collect, ids)
        }
        other => {
            let mut floats = Vec::new();
            let mapped = Term {
                id,
                ty,
                span,
                kind: other,
            }
            .map_children(&mut |child| {
                let (mut child_floats, child) = float_term(child, blocked, ids, symbols, true);
                floats.append(&mut child_floats);
                child
            });
            finish(floats, mapped, collect, ids)
        }
    }
}

fn float_soac(
    soac: SoacOp,
    blocked: &LookupSet<SymbolId>,
    ids: &mut TermIdSource,
    symbols: &mut crate::SymbolTable,
) -> (Vec<Binding>, SoacOp) {
    match soac {
        SoacOp::Map {
            lam,
            inputs,
            destination,
        } => {
            let (mut floats, lam) = float_soac_body(lam, blocked, ids, symbols);
            let inputs = inputs
                .into_iter()
                .map(|input| {
                    let (mut input_floats, input) = float_array_expr(input, blocked, ids, symbols);
                    floats.append(&mut input_floats);
                    input
                })
                .collect();
            (
                floats,
                SoacOp::Map {
                    lam,
                    inputs,
                    destination,
                },
            )
        }
        SoacOp::Reduce { op, ne, input } => {
            let (mut floats, op) = float_soac_body(op, blocked, ids, symbols);
            let (mut ne_floats, ne) = float_term(*ne, blocked, ids, symbols, true);
            let (mut input_floats, input) = float_array_expr(input, blocked, ids, symbols);
            floats.append(&mut ne_floats);
            floats.append(&mut input_floats);
            (
                floats,
                SoacOp::Reduce {
                    op,
                    ne: Box::new(ne),
                    input,
                },
            )
        }
        SoacOp::Screma {
            lanes,
            accumulators,
            inputs,
        } => {
            let mut floats = Vec::new();
            let lanes = lanes
                .into_iter()
                .map(|lane| {
                    let (mut body_floats, lam) = float_soac_body(lane.lam, blocked, ids, symbols);
                    floats.append(&mut body_floats);
                    super::ScremaLane {
                        lam,
                        input_indices: lane.input_indices,
                    }
                })
                .collect();
            let accumulators = accumulators
                .into_iter()
                .map(|acc| {
                    let (mut step_floats, step_lam) = float_soac_body(acc.step_lam, blocked, ids, symbols);
                    let (mut op_floats, reduce_op) = float_soac_body(acc.reduce_op, blocked, ids, symbols);
                    let (mut ne_floats, ne) = float_term(*acc.ne, blocked, ids, symbols, true);
                    floats.append(&mut step_floats);
                    floats.append(&mut op_floats);
                    floats.append(&mut ne_floats);
                    super::ScremaAccumulatorSpec {
                        kind: acc.kind,
                        step_lam,
                        reduce_op,
                        ne: Box::new(ne),
                    }
                })
                .collect();
            let inputs = inputs
                .into_iter()
                .map(|input| {
                    let (mut input_floats, input) = float_array_expr(input, blocked, ids, symbols);
                    floats.append(&mut input_floats);
                    input
                })
                .collect();
            (
                floats,
                SoacOp::Screma {
                    lanes,
                    accumulators,
                    inputs,
                },
            )
        }
        SoacOp::Scan {
            op,
            reduce_op,
            ne,
            input,
            destination,
        } => {
            let (mut floats, op) = float_soac_body(op, blocked, ids, symbols);
            let (mut reduce_floats, reduce_op) = float_soac_body(reduce_op, blocked, ids, symbols);
            let (mut ne_floats, ne) = float_term(*ne, blocked, ids, symbols, true);
            let (mut input_floats, input) = float_array_expr(input, blocked, ids, symbols);
            floats.append(&mut reduce_floats);
            floats.append(&mut ne_floats);
            floats.append(&mut input_floats);
            (
                floats,
                SoacOp::Scan {
                    op,
                    reduce_op,
                    ne: Box::new(ne),
                    input,
                    destination,
                },
            )
        }
        SoacOp::Filter {
            map_lam,
            pred,
            input,
            destination,
        } => {
            let (mut floats, map_lam) = match map_lam {
                Some(ml) => {
                    let (f, ml) = float_soac_body(ml, blocked, ids, symbols);
                    (f, Some(ml))
                }
                None => (Vec::new(), None),
            };
            let (mut pred_floats, pred) = float_soac_body(pred, blocked, ids, symbols);
            floats.append(&mut pred_floats);
            let (mut input_floats, input) = float_array_expr(input, blocked, ids, symbols);
            floats.append(&mut input_floats);
            (
                floats,
                SoacOp::Filter {
                    map_lam,
                    pred,
                    input,
                    destination,
                },
            )
        }
        SoacOp::Scatter { dest, lam, inputs } => {
            let (mut floats, lam) = float_soac_body(lam, blocked, ids, symbols);
            let new_inputs = inputs
                .into_iter()
                .map(|ae| {
                    let (mut f, ae) = float_array_expr(ae, blocked, ids, symbols);
                    floats.append(&mut f);
                    ae
                })
                .collect();
            (
                floats,
                SoacOp::Scatter {
                    dest,
                    lam,
                    inputs: new_inputs,
                },
            )
        }
        SoacOp::ReduceByIndex {
            dest,
            op,
            ne,
            indices,
            values,
        } => {
            let (mut floats, op) = float_soac_body(op, blocked, ids, symbols);
            let (mut ne_floats, ne) = float_term(*ne, blocked, ids, symbols, true);
            let (mut index_floats, indices) = float_array_expr(indices, blocked, ids, symbols);
            let (mut value_floats, values) = float_array_expr(values, blocked, ids, symbols);
            floats.append(&mut ne_floats);
            floats.append(&mut index_floats);
            floats.append(&mut value_floats);
            (
                floats,
                SoacOp::ReduceByIndex {
                    dest,
                    op,
                    ne: Box::new(ne),
                    indices,
                    values,
                },
            )
        }
    }
}

fn float_soac_body(
    body: SoacBody,
    blocked: &LookupSet<SymbolId>,
    ids: &mut TermIdSource,
    symbols: &mut crate::SymbolTable,
) -> (Vec<Binding>, SoacBody) {
    let mut lambda_blocked = blocked.clone();
    for (sym, _) in &body.lam.params {
        lambda_blocked.insert(*sym);
    }

    let (mut floats, lam_body) = float_term(*body.lam.body, &lambda_blocked, ids, symbols, true);
    let mut captures = Vec::with_capacity(body.captures.len());
    for (sym, ty, term) in body.captures {
        let (mut capture_floats, term) = float_term(term, blocked, ids, symbols, true);
        floats.append(&mut capture_floats);
        captures.push((sym, ty, term));
    }

    (
        floats,
        SoacBody {
            lam: Lambda {
                params: body.lam.params,
                body: Box::new(lam_body),
                ret_ty: body.lam.ret_ty,
            },
            captures,
        },
    )
}

fn float_array_expr(
    ae: ArrayExpr,
    blocked: &LookupSet<SymbolId>,
    ids: &mut TermIdSource,
    symbols: &mut crate::SymbolTable,
) -> (Vec<Binding>, ArrayExpr) {
    match ae {
        // A named input has no producer to float.
        ArrayExpr::Var(vr, ty) => (vec![], ArrayExpr::Var(vr, ty)),
        ArrayExpr::Zip(children) => {
            let mut floats = Vec::new();
            let children = children
                .into_iter()
                .map(|child| {
                    let (mut child_floats, child) = float_array_expr(child, blocked, ids, symbols);
                    floats.append(&mut child_floats);
                    child
                })
                .collect();
            (floats, ArrayExpr::Zip(children))
        }
        ArrayExpr::Literal(terms) => {
            let mut floats = Vec::new();
            let terms = terms
                .into_iter()
                .map(|term| {
                    let (mut term_floats, term) = float_term(term, blocked, ids, symbols, true);
                    floats.append(&mut term_floats);
                    term
                })
                .collect();
            (floats, ArrayExpr::Literal(terms))
        }
        ArrayExpr::Range { start, len, step } => {
            let (mut floats, start) = float_term(*start, blocked, ids, symbols, true);
            let (mut len_floats, len) = float_term(*len, blocked, ids, symbols, true);
            floats.append(&mut len_floats);
            let step = step.map(|step| {
                let (mut step_floats, step) = float_term(*step, blocked, ids, symbols, true);
                floats.append(&mut step_floats);
                Box::new(step)
            });
            (
                floats,
                ArrayExpr::Range {
                    start: Box::new(start),
                    len: Box::new(len),
                    step,
                },
            )
        }
        ArrayExpr::StorageView(sv) => {
            let (mut floats, offset) = float_term(*sv.offset, blocked, ids, symbols, true);
            let (mut len_floats, len) = float_term(*sv.len, blocked, ids, symbols, true);
            floats.append(&mut len_floats);
            (
                floats,
                ArrayExpr::StorageView(super::StorageView {
                    binding: sv.binding,
                    offset: Box::new(offset),
                    len: Box::new(len),
                    elem_ty: sv.elem_ty,
                }),
            )
        }
    }
}

fn finish(floats: Vec<Binding>, term: Term, collect: bool, ids: &mut TermIdSource) -> (Vec<Binding>, Term) {
    if collect {
        (floats, term)
    } else {
        (vec![], wrap_lets(floats, term, ids))
    }
}

fn wrap_lets(bindings: Vec<Binding>, mut body: Term, ids: &mut TermIdSource) -> Term {
    for binding in bindings.into_iter().rev() {
        let body_ty = body.ty.clone();
        body = Term {
            id: ids.next_id(),
            ty: body_ty,
            span: binding.rhs.span,
            kind: TermKind::Let {
                name: binding.name,
                name_ty: binding.name_ty,
                rhs: Box::new(binding.rhs),
                body: Box::new(body),
            },
        };
    }
    body
}

fn is_liftable_array_producer(term: &Term) -> bool {
    match &term.kind {
        TermKind::Let { body, .. } => is_liftable_array_producer(body),
        TermKind::Soac(SoacOp::Map { .. } | SoacOp::Scan { .. }) => true,
        _ => false,
    }
}

fn is_runtime_sized_array(ty: &Type<TypeName>) -> bool {
    crate::types::array_size(ty)
        .map(|s| {
            matches!(
                s,
                Type::Variable(_) | Type::Constructed(TypeName::SizePlaceholder, _)
            )
        })
        .unwrap_or(false)
}

fn is_int_lit(term: &Term) -> bool {
    matches!(term.kind, TermKind::IntLit(_))
}

fn references_any(term: &Term, blocked: &LookupSet<SymbolId>) -> bool {
    let mut found = false;
    term.for_each_child(&mut |child| {
        if !found {
            found = references_any(child, blocked);
        }
    });
    found || matches!(&term.kind, TermKind::Var(VarRef::Symbol(sym)) if blocked.contains(sym))
}

#[cfg(test)]
#[path = "runtime_index_producers_tests.rs"]
mod runtime_index_producers_tests;
