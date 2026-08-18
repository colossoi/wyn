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

use crate::ast;
use crate::map_in_place;
use crate::op;
use crate::types;
use crate::LookupSet;
use crate::SymbolTable;

use polytype::Type;

use crate::ast::TypeName;
use crate::SymbolId;

use super::data::Empty;
use super::{
    wrap_let_bindings, ArrayExpr, Def, Lambda, LetBinding, SoacBody, SoacOp, Term, TermIdSource, TermKind,
    VarRef,
};

#[derive(Debug, Clone, Copy)]
pub enum RuntimeIndexProducersFloatedTag {}
pub type RuntimeIndexProducersFloated = super::Program<
    RuntimeIndexProducersFloatedTag,
    super::monomorphize::Monomorphic,
    super::context::RewriteGlobal,
>;

pub fn float_runtime_index_nested_producers(
    mut program: super::stage::SoacsAnfNormalized,
) -> RuntimeIndexProducersFloated {
    let ids = &mut program.term_ids;
    let blocked = LookupSet::new();

    map_in_place(&mut program.defs, |def| {
        let body = def.body;
        let (floats, body) = float_term(body, &blocked, ids, &mut program.symbols, false);
        Def {
            body: wrap_let_bindings(floats, body, ids),
            ..def
        }
    });

    super::anf::debug_check(&program, "runtime_index_producers");
    program.retag()
}

fn float_term(
    term: Term<Empty, Empty>,
    blocked: &LookupSet<SymbolId>,
    ids: &mut TermIdSource,
    symbols: &mut SymbolTable,
    collect: bool,
) -> (Vec<LetBinding<Empty, Empty>>, Term<Empty, Empty>) {
    let Term { ty, span, kind, .. } = term;
    let id = ids.next_id();
    match kind {
        TermKind::Lambda(lam) => {
            let mut inner_blocked = blocked.clone();
            for (sym, _) in &lam.params {
                inner_blocked.insert(*sym);
            }
            let (floats, body) = float_term(*lam.body, &inner_blocked, ids, symbols, false);
            let body = wrap_let_bindings(floats, body, ids);
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

            if let Some((mut lifted, bucket)) = fuse_ranked_bucket_map(name, &rhs, &body, ids, symbols) {
                rhs_floats.append(&mut lifted);
                rhs_floats.append(&mut body_floats);
                return finish(rhs_floats, bucket, collect, ids);
            }

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
                            rhs: Box::new(wrap_let_bindings(rhs_floats, rhs, ids)),
                            body: Box::new(wrap_let_bindings(body_floats, body, ids)),
                        },
                    },
                )
            }
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
                let var = Term::fresh(
                    ids,
                    name_ty.clone(),
                    array_span,
                    TermKind::Var(VarRef::Symbol(name)),
                );
                let indexed = Term {
                    id,
                    ty,
                    span,
                    kind: TermKind::Index {
                        array: Box::new(var),
                        index: Box::new(index),
                    },
                };
                index_floats.push(LetBinding {
                    name,
                    name_ty,
                    rhs: *array,
                    span: array_span,
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
            let fresh_id = ids.next_id();
            let mapped = Term {
                id,
                ty,
                span,
                kind: other,
            }
            .map_children(fresh_id, &mut |child| {
                let (mut child_floats, child) = float_term(child, blocked, ids, symbols, true);
                floats.append(&mut child_floats);
                child
            });
            finish(floats, mapped, collect, ids)
        }
    }
}

struct RankedMapFusion {
    params: Vec<(SymbolId, Type<TypeName>)>,
    inputs: Vec<ArrayExpr<Empty, Empty>>,
    input_dimensions: Vec<Vec<u8>>,
    lifted: Vec<LetBinding<Empty, Empty>>,
    leaf: Option<Term<Empty, Empty>>,
}

/// Compose a rectangular nest of generated maps directly into a ranked bucket
/// scatter. The source-level `items` array then remains a semantic producer:
/// no intermediate ranked array is allocated or materialized.
fn fuse_ranked_bucket_map(
    binding: SymbolId,
    rhs: &Term<Empty, Empty>,
    body: &Term<Empty, Empty>,
    ids: &mut TermIdSource,
    symbols: &mut SymbolTable,
) -> Option<(Vec<LetBinding<Empty, Empty>>, Term<Empty, Empty>)> {
    let TermKind::Soac(SoacOp::BucketScatter {
        dest,
        inputs,
        input_dimensions,
        domain_rank,
        ..
    }) = &body.kind
    else {
        return None;
    };
    if inputs.len() != 1
        || input_dimensions.as_slice() != [(0..*domain_rank).collect::<Vec<_>>()]
        || !matches!(&inputs[0], ArrayExpr::Var(VarRef::Symbol(symbol), _) if *symbol == binding)
    {
        return None;
    }

    let mut fusion = RankedMapFusion {
        params: Vec::new(),
        inputs: Vec::new(),
        input_dimensions: Vec::new(),
        lifted: Vec::new(),
        leaf: None,
    };
    let mut bound = LookupSet::new();
    extract_ranked_map_level(rhs, 0, *domain_rank, &mut bound, &mut fusion, ids)?;
    let leaf = guard_bucket_leaf(fusion.leaf?, ids, symbols)?;
    let ret_ty = leaf.ty.clone();
    let lam = SoacBody {
        lam: Lambda {
            params: fusion.params,
            body: Box::new(leaf),
            ret_ty,
        },
        data: (),
    };
    let bucket = Term {
        id: body.id,
        ty: body.ty.clone(),
        span: body.span,
        kind: TermKind::Soac(SoacOp::BucketScatter {
            dest: dest.clone(),
            lam,
            inputs: fusion.inputs,
            input_dimensions: fusion.input_dimensions,
            domain_rank: *domain_rank,
        }),
    };
    Some((fusion.lifted, bucket))
}

/// Adapt a generated `(key, value)` leaf to the canonical guarded histogram
/// ABI `(active, key, value)`. The pair is let-bound so an expensive generated
/// leaf is evaluated once even though its key feeds both the guard and output.
fn guard_bucket_leaf(
    pair: Term<Empty, Empty>,
    ids: &mut TermIdSource,
    symbols: &mut SymbolTable,
) -> Option<Term<Empty, Empty>> {
    let Type::Constructed(TypeName::Tuple(2), fields) = &pair.ty else {
        return None;
    };
    let key_ty = fields[0].clone();
    let value_ty = fields[1].clone();
    let pair_ty = pair.ty.clone();
    let span = pair.span;
    let binding = symbols.alloc("_w_bucket_emission".into());
    let mut project = |index: usize, ty: Type<TypeName>| {
        let tuple = Term::fresh(ids, pair_ty.clone(), span, TermKind::Var(VarRef::Symbol(binding)));
        Term::fresh(
            ids,
            ty.clone(),
            span,
            TermKind::TupleProj {
                tuple: Box::new(tuple),
                idx: index,
            },
        )
    };
    let guard_key = project(0, key_ty.clone());
    let key = project(0, key_ty.clone());
    let value = project(1, value_ty.clone());
    let zero = Term::fresh(ids, key_ty.clone(), span, TermKind::IntLit("0".into()));
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let operator_ty = Type::Constructed(
        TypeName::Arrow,
        vec![
            key_ty.clone(),
            Type::Constructed(TypeName::Arrow, vec![key_ty.clone(), bool_ty.clone()]),
        ],
    );
    let operator = Term::fresh(
        ids,
        operator_ty,
        span,
        TermKind::BinOp(ast::BinaryOp {
            op: op::BinaryOperator::GreaterEqual,
        }),
    );
    let active = Term::fresh(
        ids,
        bool_ty.clone(),
        span,
        TermKind::App {
            func: Box::new(operator),
            args: vec![guard_key, zero],
        },
    );
    let emission_ty = Type::Constructed(TypeName::Tuple(3), vec![bool_ty, key_ty, value_ty]);
    let emission = Term::fresh(
        ids,
        emission_ty.clone(),
        span,
        TermKind::Tuple(vec![active, key, value]),
    );
    Some(Term::fresh(
        ids,
        emission_ty,
        span,
        TermKind::Let {
            name: binding,
            name_ty: pair_ty,
            rhs: Box::new(pair),
            body: Box::new(emission),
        },
    ))
}

fn extract_ranked_map_level(
    term: &Term<Empty, Empty>,
    depth: u8,
    rank: u8,
    bound: &mut LookupSet<SymbolId>,
    fusion: &mut RankedMapFusion,
    ids: &mut TermIdSource,
) -> Option<()> {
    if depth >= rank {
        return None;
    }
    let (term, scoped) = peel_ranked_map_lets(term, bound, &mut fusion.lifted);
    bound.extend(scoped.iter().map(|binding| binding.name));
    let TermKind::Soac(SoacOp::Map { lam, inputs, .. }) = &term.kind else {
        return None;
    };
    if inputs.len() != lam.lam.params.len()
        || inputs.iter().any(|input| array_expr_references_any(input, bound))
    {
        return None;
    }

    fusion.params.extend(lam.lam.params.iter().cloned());
    fusion.inputs.extend(inputs.iter().cloned());
    fusion.input_dimensions.extend((0..inputs.len()).map(|_| vec![depth]));
    bound.extend(lam.lam.params.iter().map(|(symbol, _)| *symbol));

    let fused = if depth + 1 == rank {
        fusion.leaf = Some((*lam.lam.body).clone());
        Some(())
    } else {
        extract_ranked_map_level(&lam.lam.body, depth + 1, rank, bound, fusion, ids)
    };
    fused?;
    if !scoped.is_empty() {
        let leaf = fusion.leaf.take()?;
        fusion.leaf = Some(wrap_let_bindings(scoped, leaf, ids));
    }
    Some(())
}

/// Hoist map-prefix lets that do not depend on outer logical coordinates.
/// Coordinate-dependent scalar lets stay inside the fused leaf. Array inputs
/// that depend on one of those scoped names are still rejected: representing
/// them requires a true coordinate-addressed producer input rather than a
/// value wrapper.
fn peel_ranked_map_lets<'a>(
    mut term: &'a Term<Empty, Empty>,
    bound: &LookupSet<SymbolId>,
    lifted: &mut Vec<LetBinding<Empty, Empty>>,
) -> (&'a Term<Empty, Empty>, Vec<LetBinding<Empty, Empty>>) {
    let mut scoped = Vec::new();
    let mut scope_bound = bound.clone();
    while let TermKind::Let {
        name,
        name_ty,
        rhs,
        body,
    } = &term.kind
    {
        let binding = LetBinding {
            name: *name,
            name_ty: name_ty.clone(),
            rhs: (**rhs).clone(),
            span: term.span,
        };
        if references_any(rhs, &scope_bound) {
            scope_bound.insert(*name);
            scoped.push(binding);
        } else {
            lifted.push(binding);
        }
        term = body;
    }
    (term, scoped)
}

fn array_expr_references_any(ae: &ArrayExpr<Empty, Empty>, blocked: &LookupSet<SymbolId>) -> bool {
    match ae {
        ArrayExpr::Var(VarRef::Symbol(symbol), _) => blocked.contains(symbol),
        ArrayExpr::Var(VarRef::Builtin { .. }, _) => false,
        ArrayExpr::Zip(children) => children.iter().any(|child| array_expr_references_any(child, blocked)),
        ArrayExpr::Literal(terms) => terms.iter().any(|term| references_any(term, blocked)),
        ArrayExpr::Range { start, len, step } => {
            references_any(start, blocked)
                || references_any(len, blocked)
                || step.as_deref().is_some_and(|step| references_any(step, blocked))
        }
    }
}

fn float_soac(
    soac: SoacOp<Empty, Empty>,
    blocked: &LookupSet<SymbolId>,
    ids: &mut TermIdSource,
    symbols: &mut SymbolTable,
) -> (Vec<LetBinding<Empty, Empty>>, SoacOp<Empty, Empty>) {
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
        SoacOp::Scan {
            op,
            ne,
            input,
            destination,
        } => {
            let (mut floats, op) = float_soac_body(op, blocked, ids, symbols);
            let (mut ne_floats, ne) = float_term(*ne, blocked, ids, symbols, true);
            let (mut input_floats, input) = float_array_expr(input, blocked, ids, symbols);
            floats.append(&mut ne_floats);
            floats.append(&mut input_floats);
            (
                floats,
                SoacOp::Scan {
                    op,
                    ne: Box::new(ne),
                    input,
                    destination,
                },
            )
        }
        SoacOp::Filter {
            pred,
            input,
            destination,
        } => {
            let (mut floats, pred) = float_soac_body(pred, blocked, ids, symbols);
            let (mut input_floats, input) = float_array_expr(input, blocked, ids, symbols);
            floats.append(&mut input_floats);
            (
                floats,
                SoacOp::Filter {
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
        SoacOp::BucketScatter {
            dest,
            lam,
            inputs,
            input_dimensions,
            domain_rank,
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
                SoacOp::BucketScatter {
                    dest,
                    lam,
                    inputs,
                    input_dimensions,
                    domain_rank,
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
    body: SoacBody<Empty, Empty>,
    blocked: &LookupSet<SymbolId>,
    ids: &mut TermIdSource,
    symbols: &mut SymbolTable,
) -> (Vec<LetBinding<Empty, Empty>>, SoacBody<Empty, Empty>) {
    let mut lambda_blocked = blocked.clone();
    for (sym, _) in &body.lam.params {
        lambda_blocked.insert(*sym);
    }

    let (floats, lam_body) = float_term(*body.lam.body, &lambda_blocked, ids, symbols, true);
    (
        floats,
        SoacBody {
            lam: Lambda {
                params: body.lam.params,
                body: Box::new(lam_body),
                ret_ty: body.lam.ret_ty,
            },
            data: (),
        },
    )
}

fn float_array_expr(
    ae: ArrayExpr<Empty, Empty>,
    blocked: &LookupSet<SymbolId>,
    ids: &mut TermIdSource,
    symbols: &mut SymbolTable,
) -> (Vec<LetBinding<Empty, Empty>>, ArrayExpr<Empty, Empty>) {
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
    }
}

fn finish(
    floats: Vec<LetBinding<Empty, Empty>>,
    term: Term<Empty, Empty>,
    collect: bool,
    ids: &mut TermIdSource,
) -> (Vec<LetBinding<Empty, Empty>>, Term<Empty, Empty>) {
    if collect {
        (floats, term)
    } else {
        (vec![], wrap_let_bindings(floats, term, ids))
    }
}

fn is_liftable_array_producer(term: &Term<Empty, Empty>) -> bool {
    match &term.kind {
        TermKind::Let { body, .. } => is_liftable_array_producer(body),
        TermKind::Soac(SoacOp::Map { .. } | SoacOp::Scan { .. }) => true,
        _ => false,
    }
}

fn is_runtime_sized_array(ty: &Type<TypeName>) -> bool {
    types::TypeExt::is_runtime_sized_array(ty)
}

fn is_int_lit(term: &Term<Empty, Empty>) -> bool {
    matches!(term.kind, TermKind::IntLit(_))
}

fn references_any(term: &Term<Empty, Empty>, blocked: &LookupSet<SymbolId>) -> bool {
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
