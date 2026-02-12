//! Map fusion pass for TLC.
//!
//! Fuses consecutive `map` operations to eliminate intermediate arrays:
//!
//! ```text
//! let b = map(f, a) in map(g, b)
//!   =>  map(g∘f, a)
//! ```
//!
//! Operates before defunctionalization — lambdas are full expressions
//! and captures are empty, making composition straightforward.

use crate::ast::{Span, TypeName};
use crate::{SymbolId, SymbolTable};
use polytype::Type;

use super::{ArrayExpr, Def, Lambda, LoopKind, Place, Program, SoacOp, Term, TermIdSource, TermKind};

// =============================================================================
// Public entry point
// =============================================================================

/// Fuse consecutive map operations in a TLC program.
pub fn fuse_maps(program: Program) -> Program {
    let mut symbols = program.symbols;
    let mut term_ids = TermIdSource::new();

    let defs = program
        .defs
        .into_iter()
        .map(|def| {
            let body = transform_term(def.body, &mut symbols, &mut term_ids);
            Def { body, ..def }
        })
        .collect();

    Program {
        defs,
        uniforms: program.uniforms,
        storage: program.storage,
        symbols,
    }
}

// =============================================================================
// Term traversal (bottom-up)
// =============================================================================

/// Recursively transform a term bottom-up, fusing maps where possible.
fn transform_term(term: Term, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> Term {
    let kind = match term.kind {
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            // Transform children first (bottom-up)
            let rhs = transform_term(*rhs, symbols, term_ids);
            let body = transform_term(*body, symbols, term_ids);

            // Try to fuse
            if let Some(fused) = try_fuse_let(name, &name_ty, &rhs, body.clone(), symbols, term_ids) {
                return fused;
            }

            TermKind::Let {
                name,
                name_ty,
                rhs: Box::new(rhs),
                body: Box::new(body),
            }
        }

        TermKind::App { func, arg } => TermKind::App {
            func: Box::new(transform_term(*func, symbols, term_ids)),
            arg: Box::new(transform_term(*arg, symbols, term_ids)),
        },

        TermKind::Lambda(lam) => TermKind::Lambda(transform_lambda(lam, symbols, term_ids)),

        TermKind::If {
            cond,
            then_branch,
            else_branch,
        } => TermKind::If {
            cond: Box::new(transform_term(*cond, symbols, term_ids)),
            then_branch: Box::new(transform_term(*then_branch, symbols, term_ids)),
            else_branch: Box::new(transform_term(*else_branch, symbols, term_ids)),
        },

        TermKind::Loop {
            loop_var,
            loop_var_ty,
            init,
            init_bindings,
            kind,
            body,
        } => {
            let init = transform_term(*init, symbols, term_ids);
            let init_bindings = init_bindings
                .into_iter()
                .map(|(n, ty, e)| (n, ty, transform_term(e, symbols, term_ids)))
                .collect();
            let kind = transform_loop_kind(kind, symbols, term_ids);
            let body = transform_term(*body, symbols, term_ids);
            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init: Box::new(init),
                init_bindings,
                kind,
                body: Box::new(body),
            }
        }

        TermKind::Soac(soac) => TermKind::Soac(transform_soac(soac, symbols, term_ids)),

        TermKind::ArrayExpr(ae) => TermKind::ArrayExpr(transform_array_expr(ae, symbols, term_ids)),

        TermKind::Force(inner) => TermKind::Force(Box::new(transform_term(*inner, symbols, term_ids))),

        TermKind::Pack {
            exists_ty,
            dims,
            value,
        } => TermKind::Pack {
            exists_ty,
            dims,
            value: Box::new(transform_term(*value, symbols, term_ids)),
        },

        TermKind::Unpack {
            scrut,
            dim_binders,
            value_binder,
            body,
        } => TermKind::Unpack {
            scrut: Box::new(transform_term(*scrut, symbols, term_ids)),
            dim_binders,
            value_binder,
            body: Box::new(transform_term(*body, symbols, term_ids)),
        },

        // Leaves — no children to transform
        TermKind::Var(_)
        | TermKind::BinOp(_)
        | TermKind::UnOp(_)
        | TermKind::IntLit(_)
        | TermKind::FloatLit(_)
        | TermKind::BoolLit(_)
        | TermKind::StringLit(_)
        | TermKind::Extern(_) => term.kind,
    };

    Term {
        id: term.id,
        ty: term.ty,
        span: term.span,
        kind,
    }
}

fn transform_lambda(lam: Lambda, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> Lambda {
    Lambda {
        params: lam.params,
        body: Box::new(transform_term(*lam.body, symbols, term_ids)),
        ret_ty: lam.ret_ty,
        captures: lam
            .captures
            .into_iter()
            .map(|(s, ty, t)| (s, ty, transform_term(t, symbols, term_ids)))
            .collect(),
    }
}

fn transform_soac(soac: SoacOp, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> SoacOp {
    match soac {
        SoacOp::Map { lam, inputs } => {
            // First: recurse into children (bottom-up)
            let lam = transform_lambda(lam, symbols, term_ids);
            let inputs: Vec<_> =
                inputs.into_iter().map(|ae| transform_array_expr(ae, symbols, term_ids)).collect();

            // Then: fuse any inline nested Maps from inputs
            fuse_inline_map_inputs(lam, inputs, symbols, term_ids)
        }
        SoacOp::Reduce { op, ne, input, props } => SoacOp::Reduce {
            op: transform_lambda(op, symbols, term_ids),
            ne: Box::new(transform_term(*ne, symbols, term_ids)),
            input: transform_array_expr(input, symbols, term_ids),
            props,
        },
        SoacOp::Scan { op, ne, input } => SoacOp::Scan {
            op: transform_lambda(op, symbols, term_ids),
            ne: Box::new(transform_term(*ne, symbols, term_ids)),
            input: transform_array_expr(input, symbols, term_ids),
        },
        SoacOp::Filter { pred, input } => SoacOp::Filter {
            pred: transform_lambda(pred, symbols, term_ids),
            input: transform_array_expr(input, symbols, term_ids),
        },
        SoacOp::Scatter {
            dest,
            indices,
            values,
        } => SoacOp::Scatter {
            dest: transform_place(dest, symbols, term_ids),
            indices: transform_array_expr(indices, symbols, term_ids),
            values: transform_array_expr(values, symbols, term_ids),
        },
        SoacOp::ReduceByIndex {
            dest,
            op,
            ne,
            indices,
            values,
            props,
        } => SoacOp::ReduceByIndex {
            dest: transform_place(dest, symbols, term_ids),
            op: transform_lambda(op, symbols, term_ids),
            ne: Box::new(transform_term(*ne, symbols, term_ids)),
            indices: transform_array_expr(indices, symbols, term_ids),
            values: transform_array_expr(values, symbols, term_ids),
            props,
        },
    }
}

fn transform_array_expr(
    ae: ArrayExpr,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> ArrayExpr {
    match ae {
        ArrayExpr::Ref(t) => ArrayExpr::Ref(Box::new(transform_term(*t, symbols, term_ids))),
        ArrayExpr::Zip(exprs) => {
            ArrayExpr::Zip(exprs.into_iter().map(|e| transform_array_expr(e, symbols, term_ids)).collect())
        }
        ArrayExpr::Soac(op) => ArrayExpr::Soac(Box::new(transform_soac(*op, symbols, term_ids))),
        ArrayExpr::Generate {
            shape,
            index_fn,
            elem_ty,
        } => ArrayExpr::Generate {
            shape,
            index_fn: transform_lambda(index_fn, symbols, term_ids),
            elem_ty,
        },
        ArrayExpr::Literal(terms) => {
            ArrayExpr::Literal(terms.into_iter().map(|t| transform_term(t, symbols, term_ids)).collect())
        }
        ArrayExpr::Range { start, len } => ArrayExpr::Range {
            start: Box::new(transform_term(*start, symbols, term_ids)),
            len: Box::new(transform_term(*len, symbols, term_ids)),
        },
    }
}

fn transform_loop_kind(kind: LoopKind, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> LoopKind {
    match kind {
        LoopKind::For { var, var_ty, iter } => LoopKind::For {
            var,
            var_ty,
            iter: Box::new(transform_term(*iter, symbols, term_ids)),
        },
        LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
            var,
            var_ty,
            bound: Box::new(transform_term(*bound, symbols, term_ids)),
        },
        LoopKind::While { cond } => LoopKind::While {
            cond: Box::new(transform_term(*cond, symbols, term_ids)),
        },
    }
}

fn transform_place(place: Place, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> Place {
    match place {
        Place::BufferSlice {
            base,
            offset,
            shape,
            elem_ty,
        } => Place::BufferSlice {
            base: Box::new(transform_term(*base, symbols, term_ids)),
            offset: Box::new(transform_term(*offset, symbols, term_ids)),
            shape,
            elem_ty,
        },
        Place::LocalArray { id, shape, elem_ty } => Place::LocalArray { id, shape, elem_ty },
    }
}

// =============================================================================
// Inline fusion (nested Maps in ArrayExpr::Ref inputs)
// =============================================================================

/// Fuse inline nested Maps from a Map's inputs.
///
/// For each input that is `ArrayExpr::Ref(Soac(Map{inner_lam, inner_inputs}))`:
/// - Splice `inner_lam.params` into the outer lambda at that input's position
/// - Splice `inner_inputs` into the outer map's inputs at that position
/// - Inline `inner_lam.body` via a let-binding, substituting the outer param
///
/// This handles both sole-input (`map(f, map(g, a))`) and zip-fused cases
/// (`map(f, zip(map(g, a), b))`) uniformly.
fn fuse_inline_map_inputs(
    lam: Lambda,
    inputs: Vec<ArrayExpr>,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> SoacOp {
    // Check if any input is a Ref wrapping a Soac Map
    let has_fusible = inputs.iter().any(
        |input| matches!(input, ArrayExpr::Ref(t) if matches!(t.kind, TermKind::Soac(SoacOp::Map { .. }))),
    );

    if !has_fusible {
        return SoacOp::Map { lam, inputs };
    }

    let mut new_params = Vec::new();
    let mut new_inputs = Vec::new();
    let mut body = *lam.body;
    let span = body.span;

    for (i, input) in inputs.into_iter().enumerate() {
        // Check if this input is Ref(Soac(Map{...}))
        let inner_map = match input {
            ArrayExpr::Ref(ref t) => match &t.kind {
                TermKind::Soac(SoacOp::Map { .. }) => true,
                _ => false,
            },
            _ => false,
        };

        if inner_map {
            // Extract the inner map (consume the input)
            let inner_term = match input {
                ArrayExpr::Ref(t) => *t,
                _ => unreachable!(),
            };
            let (inner_lam, inner_inputs) = match inner_term.kind {
                TermKind::Soac(SoacOp::Map { lam, inputs }) => (lam, inputs),
                _ => unreachable!(),
            };

            // Fresh symbol for the intermediate (inner map's result per-element)
            let fresh = symbols.alloc("_fused".to_string());
            let outer_param = lam.params[i].0;

            // Substitute the outer param with fresh in the body
            body = substitute_sym(body, outer_param, fresh, term_ids);

            // Wrap: let fresh = inner_lam.body in body
            body = Term {
                id: term_ids.next_id(),
                ty: body.ty.clone(),
                span,
                kind: TermKind::Let {
                    name: fresh,
                    name_ty: inner_lam.ret_ty,
                    rhs: inner_lam.body,
                    body: Box::new(body),
                },
            };

            // Splice inner params and inputs at this position
            new_params.extend(inner_lam.params);
            new_inputs.extend(inner_inputs);
        } else {
            // Keep non-fusible inputs as-is
            new_params.push(lam.params[i].clone());
            new_inputs.push(input);
        }
    }

    SoacOp::Map {
        lam: Lambda {
            params: new_params,
            body: Box::new(body),
            ret_ty: lam.ret_ty,
            captures: vec![],
        },
        inputs: new_inputs,
    }
}

// =============================================================================
// Let-based fusion
// =============================================================================

/// Try to fuse a Let binding where the RHS is a producer Map and the body
/// contains a consumer Map that uses the intermediate as its sole input.
///
/// Returns `Some(fused_term)` if fusion succeeds, `None` otherwise.
fn try_fuse_let(
    name: SymbolId,
    _name_ty: &Type<TypeName>,
    rhs: &Term,
    body: Term,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    // 1. Check that the RHS is a Map SOAC (the producer)
    let (producer_lam, producer_inputs) = match &rhs.kind {
        TermKind::Soac(SoacOp::Map { lam, inputs }) => (lam, inputs),
        _ => return None,
    };

    // 2. Check that `name` is used exactly once in the body
    if count_uses(&body, name) != 1 {
        return None;
    }

    // 3. Find the consumer: the single use must be as the sole input to a Map
    let consumer_info = find_consumer_map(&body, name)?;

    // 4. Compose the lambdas: g ∘ f
    let composed = compose_lambdas(
        producer_lam.clone(),
        consumer_info.consumer_lam.clone(),
        rhs.span,
        symbols,
        term_ids,
    );

    // 5. Build the fused Map
    let fused_map = SoacOp::Map {
        lam: composed,
        inputs: producer_inputs.clone(),
    };

    let fused_term = Term {
        id: term_ids.next_id(),
        ty: consumer_info.consumer_ty.clone(),
        span: rhs.span,
        kind: TermKind::Soac(fused_map),
    };

    // 6. Replace the consumer map in the body with the fused map
    Some(replace_consumer_map(&body, name, fused_term, term_ids))
}

/// Information about a consumer Map found in the body.
struct ConsumerInfo {
    consumer_lam: Lambda,
    consumer_ty: Type<TypeName>,
}

/// Find a consumer Map in the body that uses `name` as its sole input.
/// Returns the consumer lambda and type if found.
fn find_consumer_map(term: &Term, name: SymbolId) -> Option<ConsumerInfo> {
    match &term.kind {
        // Direct consumer: the body itself is a Map using `name`
        TermKind::Soac(SoacOp::Map { lam, inputs }) => {
            if is_sole_ref_to(inputs, name) {
                Some(ConsumerInfo {
                    consumer_lam: lam.clone(),
                    consumer_ty: term.ty.clone(),
                })
            } else {
                None
            }
        }

        // The consumer might be nested in a Let body
        TermKind::Let {
            name: inner_name,
            rhs,
            body,
            ..
        } => {
            // Check if the use is in the rhs
            if count_uses(rhs, name) == 1 {
                find_consumer_map(rhs, name)
            } else if *inner_name != name && count_uses(body, name) == 1 {
                // Use is in the body (and not shadowed)
                find_consumer_map(body, name)
            } else {
                None
            }
        }

        _ => None,
    }
}

/// Check if the inputs list is a single `ArrayExpr::Ref` pointing to `name`.
fn is_sole_ref_to(inputs: &[ArrayExpr], name: SymbolId) -> bool {
    if inputs.len() != 1 {
        return false;
    }
    match &inputs[0] {
        ArrayExpr::Ref(t) => matches!(&t.kind, TermKind::Var(sym) if *sym == name),
        _ => false,
    }
}

/// Replace the consumer Map (which uses `name` as sole input) with `replacement`.
/// This traverses the body to find and replace the exact consumer.
fn replace_consumer_map(
    term: &Term,
    name: SymbolId,
    replacement: Term,
    term_ids: &mut TermIdSource,
) -> Term {
    match &term.kind {
        TermKind::Soac(SoacOp::Map { inputs, .. }) => {
            if is_sole_ref_to(inputs, name) {
                // This is the consumer — replace it
                replacement
            } else {
                term.clone()
            }
        }

        TermKind::Let {
            name: inner_name,
            name_ty,
            rhs,
            body,
        } => {
            if count_uses(rhs, name) == 1 {
                // The consumer is in the rhs
                Term {
                    id: term_ids.next_id(),
                    ty: term.ty.clone(),
                    span: term.span,
                    kind: TermKind::Let {
                        name: *inner_name,
                        name_ty: name_ty.clone(),
                        rhs: Box::new(replace_consumer_map(rhs, name, replacement, term_ids)),
                        body: body.clone(),
                    },
                }
            } else if *inner_name != name && count_uses(body, name) == 1 {
                // The consumer is in the body
                Term {
                    id: term_ids.next_id(),
                    ty: term.ty.clone(),
                    span: term.span,
                    kind: TermKind::Let {
                        name: *inner_name,
                        name_ty: name_ty.clone(),
                        rhs: rhs.clone(),
                        body: Box::new(replace_consumer_map(body, name, replacement, term_ids)),
                    },
                }
            } else {
                term.clone()
            }
        }

        _ => term.clone(),
    }
}

// =============================================================================
// Lambda composition
// =============================================================================

/// Compose two lambdas: `compose(f, g)` where `f: A→B`, `g: B→C` produces `A→C`.
///
/// ```text
/// Lambda {
///     params: f.params,
///     body: let fresh = f.body in g.body[g.param → fresh],
///     ret_ty: g.ret_ty,
///     captures: [],
/// }
/// ```
fn compose_lambdas(
    f: Lambda,
    g: Lambda,
    span: Span,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Lambda {
    // Fresh symbol for the intermediate result of f
    let fresh_sym = symbols.alloc("_fused".to_string());

    // The type of the intermediate is f's return type
    let intermediate_ty = f.ret_ty.clone();

    // Substitute g's first parameter with the fresh symbol in g's body
    let g_param = g.params[0].0;
    let g_body_substituted = substitute_sym(*g.body, g_param, fresh_sym, term_ids);

    // Build: let fresh = f.body in g.body[g.param → fresh]
    let composed_body = Term {
        id: term_ids.next_id(),
        ty: g.ret_ty.clone(),
        span,
        kind: TermKind::Let {
            name: fresh_sym,
            name_ty: intermediate_ty,
            rhs: f.body,
            body: Box::new(g_body_substituted),
        },
    };

    Lambda {
        params: f.params,
        body: Box::new(composed_body),
        ret_ty: g.ret_ty,
        captures: vec![],
    }
}

// =============================================================================
// Use counting
// =============================================================================

/// Count free occurrences of `sym` in a term.
fn count_uses(term: &Term, sym: SymbolId) -> usize {
    match &term.kind {
        TermKind::Var(s) => {
            if *s == sym {
                1
            } else {
                0
            }
        }

        TermKind::BinOp(_)
        | TermKind::UnOp(_)
        | TermKind::IntLit(_)
        | TermKind::FloatLit(_)
        | TermKind::BoolLit(_)
        | TermKind::StringLit(_)
        | TermKind::Extern(_) => 0,

        TermKind::App { func, arg } => count_uses(func, sym) + count_uses(arg, sym),

        TermKind::Lambda(lam) => {
            if lam.params.iter().any(|(p, _)| *p == sym) {
                0
            } else {
                count_uses_lambda(lam, sym)
            }
        }

        TermKind::Let { name, rhs, body, .. } => {
            let rhs_count = count_uses(rhs, sym);
            if *name == sym { rhs_count } else { rhs_count + count_uses(body, sym) }
        }

        TermKind::If {
            cond,
            then_branch,
            else_branch,
        } => count_uses(cond, sym) + count_uses(then_branch, sym) + count_uses(else_branch, sym),

        TermKind::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body,
            ..
        } => {
            let init_count = count_uses(init, sym);

            let shadows = *loop_var == sym || init_bindings.iter().any(|(n, _, _)| *n == sym);

            let bindings_count = if shadows {
                0
            } else {
                init_bindings.iter().map(|(_, _, e)| count_uses(e, sym)).sum::<usize>()
            };

            let kind_count = match kind {
                LoopKind::For { var, iter, .. } => {
                    let iter_count = count_uses(iter, sym);
                    if *var == sym {
                        iter_count
                    } else if shadows {
                        iter_count
                    } else {
                        iter_count
                    }
                }
                LoopKind::ForRange { bound, .. } => count_uses(bound, sym),
                LoopKind::While { cond } => {
                    if shadows {
                        0
                    } else {
                        count_uses(cond, sym)
                    }
                }
            };

            let body_count = if shadows { 0 } else { count_uses(body, sym) };

            init_count + bindings_count + kind_count + body_count
        }

        TermKind::Soac(soac) => count_uses_soac(soac, sym),

        TermKind::ArrayExpr(ae) => count_uses_array_expr(ae, sym),

        TermKind::Force(inner) => count_uses(inner, sym),

        TermKind::Pack { value, .. } => count_uses(value, sym),

        TermKind::Unpack {
            scrut,
            value_binder,
            body,
            ..
        } => {
            let scrut_count = count_uses(scrut, sym);
            if *value_binder == sym { scrut_count } else { scrut_count + count_uses(body, sym) }
        }
    }
}

fn count_uses_lambda(lam: &Lambda, sym: SymbolId) -> usize {
    let body_count = count_uses(&lam.body, sym);
    let captures_count: usize = lam.captures.iter().map(|(_, _, t)| count_uses(t, sym)).sum();
    body_count + captures_count
}

fn count_uses_soac(soac: &SoacOp, sym: SymbolId) -> usize {
    match soac {
        SoacOp::Map { lam, inputs } => {
            count_uses_lambda(lam, sym)
                + inputs.iter().map(|ae| count_uses_array_expr(ae, sym)).sum::<usize>()
        }
        SoacOp::Reduce { op, ne, input, .. } => {
            count_uses_lambda(op, sym) + count_uses(ne, sym) + count_uses_array_expr(input, sym)
        }
        SoacOp::Scan { op, ne, input } => {
            count_uses_lambda(op, sym) + count_uses(ne, sym) + count_uses_array_expr(input, sym)
        }
        SoacOp::Filter { pred, input } => count_uses_lambda(pred, sym) + count_uses_array_expr(input, sym),
        SoacOp::Scatter {
            dest,
            indices,
            values,
        } => {
            count_uses_place(dest, sym)
                + count_uses_array_expr(indices, sym)
                + count_uses_array_expr(values, sym)
        }
        SoacOp::ReduceByIndex {
            dest,
            op,
            ne,
            indices,
            values,
            ..
        } => {
            count_uses_place(dest, sym)
                + count_uses_lambda(op, sym)
                + count_uses(ne, sym)
                + count_uses_array_expr(indices, sym)
                + count_uses_array_expr(values, sym)
        }
    }
}

fn count_uses_array_expr(ae: &ArrayExpr, sym: SymbolId) -> usize {
    match ae {
        ArrayExpr::Ref(t) => count_uses(t, sym),
        ArrayExpr::Zip(exprs) => exprs.iter().map(|e| count_uses_array_expr(e, sym)).sum(),
        ArrayExpr::Soac(op) => count_uses_soac(op, sym),
        ArrayExpr::Generate { index_fn, .. } => count_uses_lambda(index_fn, sym),
        ArrayExpr::Literal(terms) => terms.iter().map(|t| count_uses(t, sym)).sum(),
        ArrayExpr::Range { start, len } => count_uses(start, sym) + count_uses(len, sym),
    }
}

fn count_uses_place(place: &Place, sym: SymbolId) -> usize {
    match place {
        Place::BufferSlice { base, offset, .. } => count_uses(base, sym) + count_uses(offset, sym),
        Place::LocalArray { .. } => 0,
    }
}

// =============================================================================
// Symbol substitution
// =============================================================================

/// Substitute all free occurrences of `old` with `Var(new)` in a term,
/// respecting shadowing.
fn substitute_sym(term: Term, old: SymbolId, new: SymbolId, term_ids: &mut TermIdSource) -> Term {
    match term.kind {
        TermKind::Var(s) if s == old => Term {
            id: term_ids.next_id(),
            ty: term.ty,
            span: term.span,
            kind: TermKind::Var(new),
        },

        TermKind::Var(_)
        | TermKind::BinOp(_)
        | TermKind::UnOp(_)
        | TermKind::IntLit(_)
        | TermKind::FloatLit(_)
        | TermKind::BoolLit(_)
        | TermKind::StringLit(_)
        | TermKind::Extern(_) => term,

        TermKind::App { func, arg } => Term {
            id: term_ids.next_id(),
            ty: term.ty,
            span: term.span,
            kind: TermKind::App {
                func: Box::new(substitute_sym(*func, old, new, term_ids)),
                arg: Box::new(substitute_sym(*arg, old, new, term_ids)),
            },
        },

        TermKind::Lambda(ref lam) if lam.params.iter().any(|(p, _)| *p == old) => term,

        TermKind::Lambda(lam) => Term {
            id: term_ids.next_id(),
            ty: term.ty,
            span: term.span,
            kind: TermKind::Lambda(substitute_sym_lambda(lam, old, new, term_ids)),
        },

        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            let new_rhs = substitute_sym(*rhs, old, new, term_ids);
            let new_body = if name == old { *body } else { substitute_sym(*body, old, new, term_ids) };
            Term {
                id: term_ids.next_id(),
                ty: term.ty,
                span: term.span,
                kind: TermKind::Let {
                    name,
                    name_ty,
                    rhs: Box::new(new_rhs),
                    body: Box::new(new_body),
                },
            }
        }

        TermKind::If {
            cond,
            then_branch,
            else_branch,
        } => Term {
            id: term_ids.next_id(),
            ty: term.ty,
            span: term.span,
            kind: TermKind::If {
                cond: Box::new(substitute_sym(*cond, old, new, term_ids)),
                then_branch: Box::new(substitute_sym(*then_branch, old, new, term_ids)),
                else_branch: Box::new(substitute_sym(*else_branch, old, new, term_ids)),
            },
        },

        TermKind::Loop {
            loop_var,
            loop_var_ty,
            init,
            init_bindings,
            kind,
            body,
        } => {
            let shadows = loop_var == old || init_bindings.iter().any(|(n, _, _)| *n == old);

            let new_init = substitute_sym(*init, old, new, term_ids);

            let new_init_bindings: Vec<_> = init_bindings
                .into_iter()
                .map(|(n, ty, e)| {
                    let new_e = if shadows { e } else { substitute_sym(e, old, new, term_ids) };
                    (n, ty, new_e)
                })
                .collect();

            let new_kind = match kind {
                LoopKind::For { var, var_ty, iter } => LoopKind::For {
                    var,
                    var_ty,
                    iter: Box::new(substitute_sym(*iter, old, new, term_ids)),
                },
                LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                    var,
                    var_ty,
                    bound: Box::new(substitute_sym(*bound, old, new, term_ids)),
                },
                LoopKind::While { cond } => {
                    let new_cond = if shadows { *cond } else { substitute_sym(*cond, old, new, term_ids) };
                    LoopKind::While {
                        cond: Box::new(new_cond),
                    }
                }
            };

            let new_body = if shadows { *body } else { substitute_sym(*body, old, new, term_ids) };

            Term {
                id: term_ids.next_id(),
                ty: term.ty,
                span: term.span,
                kind: TermKind::Loop {
                    loop_var,
                    loop_var_ty,
                    init: Box::new(new_init),
                    init_bindings: new_init_bindings,
                    kind: new_kind,
                    body: Box::new(new_body),
                },
            }
        }

        TermKind::Soac(soac) => Term {
            id: term_ids.next_id(),
            ty: term.ty,
            span: term.span,
            kind: TermKind::Soac(substitute_sym_soac(soac, old, new, term_ids)),
        },

        TermKind::ArrayExpr(ae) => Term {
            id: term_ids.next_id(),
            ty: term.ty,
            span: term.span,
            kind: TermKind::ArrayExpr(substitute_sym_array_expr(ae, old, new, term_ids)),
        },

        TermKind::Force(inner) => Term {
            id: term_ids.next_id(),
            ty: term.ty,
            span: term.span,
            kind: TermKind::Force(Box::new(substitute_sym(*inner, old, new, term_ids))),
        },

        TermKind::Pack {
            exists_ty,
            dims,
            value,
        } => Term {
            id: term_ids.next_id(),
            ty: term.ty,
            span: term.span,
            kind: TermKind::Pack {
                exists_ty,
                dims,
                value: Box::new(substitute_sym(*value, old, new, term_ids)),
            },
        },

        TermKind::Unpack {
            scrut,
            dim_binders,
            value_binder,
            body,
        } => {
            let new_scrut = substitute_sym(*scrut, old, new, term_ids);
            let new_body =
                if value_binder == old { *body } else { substitute_sym(*body, old, new, term_ids) };
            Term {
                id: term_ids.next_id(),
                ty: term.ty,
                span: term.span,
                kind: TermKind::Unpack {
                    scrut: Box::new(new_scrut),
                    dim_binders,
                    value_binder,
                    body: Box::new(new_body),
                },
            }
        }
    }
}

fn substitute_sym_lambda(lam: Lambda, old: SymbolId, new: SymbolId, term_ids: &mut TermIdSource) -> Lambda {
    Lambda {
        params: lam.params,
        body: Box::new(substitute_sym(*lam.body, old, new, term_ids)),
        ret_ty: lam.ret_ty,
        captures: lam
            .captures
            .into_iter()
            .map(|(s, ty, t)| (s, ty, substitute_sym(t, old, new, term_ids)))
            .collect(),
    }
}

fn substitute_sym_soac(soac: SoacOp, old: SymbolId, new: SymbolId, term_ids: &mut TermIdSource) -> SoacOp {
    match soac {
        SoacOp::Map { lam, inputs } => SoacOp::Map {
            lam: substitute_sym_lambda(lam, old, new, term_ids),
            inputs: inputs
                .into_iter()
                .map(|ae| substitute_sym_array_expr(ae, old, new, term_ids))
                .collect(),
        },
        SoacOp::Reduce { op, ne, input, props } => SoacOp::Reduce {
            op: substitute_sym_lambda(op, old, new, term_ids),
            ne: Box::new(substitute_sym(*ne, old, new, term_ids)),
            input: substitute_sym_array_expr(input, old, new, term_ids),
            props,
        },
        SoacOp::Scan { op, ne, input } => SoacOp::Scan {
            op: substitute_sym_lambda(op, old, new, term_ids),
            ne: Box::new(substitute_sym(*ne, old, new, term_ids)),
            input: substitute_sym_array_expr(input, old, new, term_ids),
        },
        SoacOp::Filter { pred, input } => SoacOp::Filter {
            pred: substitute_sym_lambda(pred, old, new, term_ids),
            input: substitute_sym_array_expr(input, old, new, term_ids),
        },
        SoacOp::Scatter {
            dest,
            indices,
            values,
        } => SoacOp::Scatter {
            dest,
            indices: substitute_sym_array_expr(indices, old, new, term_ids),
            values: substitute_sym_array_expr(values, old, new, term_ids),
        },
        SoacOp::ReduceByIndex {
            dest,
            op,
            ne,
            indices,
            values,
            props,
        } => SoacOp::ReduceByIndex {
            dest,
            op: substitute_sym_lambda(op, old, new, term_ids),
            ne: Box::new(substitute_sym(*ne, old, new, term_ids)),
            indices: substitute_sym_array_expr(indices, old, new, term_ids),
            values: substitute_sym_array_expr(values, old, new, term_ids),
            props,
        },
    }
}

fn substitute_sym_array_expr(
    ae: ArrayExpr,
    old: SymbolId,
    new: SymbolId,
    term_ids: &mut TermIdSource,
) -> ArrayExpr {
    match ae {
        ArrayExpr::Ref(t) => ArrayExpr::Ref(Box::new(substitute_sym(*t, old, new, term_ids))),
        ArrayExpr::Zip(exprs) => ArrayExpr::Zip(
            exprs.into_iter().map(|e| substitute_sym_array_expr(e, old, new, term_ids)).collect(),
        ),
        ArrayExpr::Soac(op) => ArrayExpr::Soac(Box::new(substitute_sym_soac(*op, old, new, term_ids))),
        ArrayExpr::Generate {
            shape,
            index_fn,
            elem_ty,
        } => ArrayExpr::Generate {
            shape,
            index_fn: substitute_sym_lambda(index_fn, old, new, term_ids),
            elem_ty,
        },
        ArrayExpr::Literal(terms) => {
            ArrayExpr::Literal(terms.into_iter().map(|t| substitute_sym(t, old, new, term_ids)).collect())
        }
        ArrayExpr::Range { start, len } => ArrayExpr::Range {
            start: Box::new(substitute_sym(*start, old, new, term_ids)),
            len: Box::new(substitute_sym(*len, old, new, term_ids)),
        },
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;

    fn dummy_span() -> Span {
        Span::new(0, 0, 0, 0)
    }

    fn mk_term(kind: TermKind, ty: Type<TypeName>, term_ids: &mut TermIdSource) -> Term {
        Term {
            id: term_ids.next_id(),
            ty,
            span: dummy_span(),
            kind,
        }
    }

    fn i32_ty() -> Type<TypeName> {
        Type::Constructed(TypeName::Int(32), vec![])
    }

    fn f32_ty() -> Type<TypeName> {
        Type::Constructed(TypeName::Float(32), vec![])
    }

    fn array_ty(elem: Type<TypeName>) -> Type<TypeName> {
        Type::Constructed(
            TypeName::Array,
            vec![
                elem,
                Type::Variable(0), // size
                Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            ],
        )
    }

    /// Build a simple map: `map(lam, [input])`
    fn mk_map(lam: Lambda, input: Term, result_ty: Type<TypeName>, term_ids: &mut TermIdSource) -> Term {
        mk_term(
            TermKind::Soac(SoacOp::Map {
                lam,
                inputs: vec![ArrayExpr::Ref(Box::new(input))],
            }),
            result_ty,
            term_ids,
        )
    }

    /// Build a lambda with one parameter
    fn mk_lambda1(param: SymbolId, param_ty: Type<TypeName>, body: Term, ret_ty: Type<TypeName>) -> Lambda {
        Lambda {
            params: vec![(param, param_ty)],
            body: Box::new(body),
            ret_ty,
            captures: vec![],
        }
    }

    // -------------------------------------------------------------------------
    // Test: simple map(g, map(f, a)) → map(g∘f, a)
    // -------------------------------------------------------------------------
    #[test]
    fn test_simple_map_fusion() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        // Symbols
        let a_sym = symbols.alloc("a".to_string());
        let b_sym = symbols.alloc("b".to_string());
        let x_sym = symbols.alloc("x".to_string()); // f's param
        let y_sym = symbols.alloc("y".to_string()); // g's param

        // f: i32 → i32 (identity-like, just returns x)
        let f_body = mk_term(TermKind::Var(x_sym), i32_ty(), &mut term_ids);
        let f = mk_lambda1(x_sym, i32_ty(), f_body, i32_ty());

        // g: i32 → i32 (identity-like, just returns y)
        let g_body = mk_term(TermKind::Var(y_sym), i32_ty(), &mut term_ids);
        let g = mk_lambda1(y_sym, i32_ty(), g_body, i32_ty());

        // a: [i32]
        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);

        // Producer: map(f, a)
        let producer = mk_map(f, a, array_ty(i32_ty()), &mut term_ids);

        // Consumer: map(g, b)
        let b_ref = mk_term(TermKind::Var(b_sym), array_ty(i32_ty()), &mut term_ids);
        let consumer = mk_map(g, b_ref, array_ty(i32_ty()), &mut term_ids);

        // let b = map(f, a) in map(g, b)
        let program_body = mk_term(
            TermKind::Let {
                name: b_sym,
                name_ty: array_ty(i32_ty()),
                rhs: Box::new(producer),
                body: Box::new(consumer),
            },
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: program_body,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let fused = fuse_maps(program);

        // The result should be a single Map (no Let binding)
        match &fused.defs[0].body.kind {
            TermKind::Soac(SoacOp::Map { lam, inputs }) => {
                // Input should be 'a' (the original array)
                assert_eq!(inputs.len(), 1);
                match &inputs[0] {
                    ArrayExpr::Ref(t) => match &t.kind {
                        TermKind::Var(s) => assert_eq!(*s, a_sym),
                        other => panic!("Expected Var(a), got {:?}", other),
                    },
                    other => panic!("Expected Ref, got {:?}", other),
                }

                // Lambda should have f's param (x)
                assert_eq!(lam.params.len(), 1);
                assert_eq!(lam.params[0].0, x_sym);

                // Body should be: let _fused = x in _fused
                // (g's body is y, substituted to _fused; f's body is x)
                match &lam.body.kind {
                    TermKind::Let { rhs, body, .. } => {
                        // rhs is f's body (Var(x))
                        assert!(matches!(&rhs.kind, TermKind::Var(s) if *s == x_sym));
                        // body should be Var(_fused) — the fresh symbol
                        assert!(matches!(&body.kind, TermKind::Var(_)));
                    }
                    other => panic!("Expected Let (composed body), got {:?}", other),
                }
            }
            other => panic!("Expected fused Soac(Map), got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Test: chain of three maps fused
    // -------------------------------------------------------------------------
    #[test]
    fn test_chain_of_three_maps() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        let a_sym = symbols.alloc("a".to_string());
        let b_sym = symbols.alloc("b".to_string());
        let c_sym = symbols.alloc("c".to_string());
        let x_sym = symbols.alloc("x".to_string());
        let y_sym = symbols.alloc("y".to_string());
        let z_sym = symbols.alloc("z".to_string());

        // f, g, h: i32 → i32
        let f = mk_lambda1(
            x_sym,
            i32_ty(),
            mk_term(TermKind::Var(x_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let g = mk_lambda1(
            y_sym,
            i32_ty(),
            mk_term(TermKind::Var(y_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let h = mk_lambda1(
            z_sym,
            i32_ty(),
            mk_term(TermKind::Var(z_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );

        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);
        let producer = mk_map(f, a, array_ty(i32_ty()), &mut term_ids);

        let b_ref = mk_term(TermKind::Var(b_sym), array_ty(i32_ty()), &mut term_ids);
        let middle = mk_map(g, b_ref, array_ty(i32_ty()), &mut term_ids);

        let c_ref = mk_term(TermKind::Var(c_sym), array_ty(i32_ty()), &mut term_ids);
        let consumer = mk_map(h, c_ref, array_ty(i32_ty()), &mut term_ids);

        // let b = map(f, a) in let c = map(g, b) in map(h, c)
        let inner_let = mk_term(
            TermKind::Let {
                name: c_sym,
                name_ty: array_ty(i32_ty()),
                rhs: Box::new(middle),
                body: Box::new(consumer),
            },
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let outer_let = mk_term(
            TermKind::Let {
                name: b_sym,
                name_ty: array_ty(i32_ty()),
                rhs: Box::new(producer),
                body: Box::new(inner_let),
            },
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: outer_let,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let fused = fuse_maps(program);

        // Should be a single Map with a's input (all three fused)
        match &fused.defs[0].body.kind {
            TermKind::Soac(SoacOp::Map { inputs, lam }) => {
                assert_eq!(inputs.len(), 1);
                match &inputs[0] {
                    ArrayExpr::Ref(t) => assert!(matches!(&t.kind, TermKind::Var(s) if *s == a_sym)),
                    other => panic!("Expected Ref(a), got {:?}", other),
                }
                // Lambda param should be f's original param (x)
                assert_eq!(lam.params[0].0, x_sym);
            }
            other => panic!("Expected fully fused Map, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Test: multi-use intermediate (no fusion)
    // -------------------------------------------------------------------------
    #[test]
    fn test_multi_use_no_fusion() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        let a_sym = symbols.alloc("a".to_string());
        let b_sym = symbols.alloc("b".to_string());
        let x_sym = symbols.alloc("x".to_string());
        let y_sym = symbols.alloc("y".to_string());

        let f = mk_lambda1(
            x_sym,
            i32_ty(),
            mk_term(TermKind::Var(x_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let g = mk_lambda1(
            y_sym,
            i32_ty(),
            mk_term(TermKind::Var(y_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );

        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);
        let producer = mk_map(f, a, array_ty(i32_ty()), &mut term_ids);

        // Consumer uses b
        let b_ref1 = mk_term(TermKind::Var(b_sym), array_ty(i32_ty()), &mut term_ids);
        let consumer = mk_map(g, b_ref1, array_ty(i32_ty()), &mut term_ids);

        // Second use of b (in an App, making count_uses == 2)
        let b_ref2 = mk_term(TermKind::Var(b_sym), array_ty(i32_ty()), &mut term_ids);

        // Body: App(consumer, b) — artificial but creates two uses
        let body = mk_term(
            TermKind::App {
                func: Box::new(consumer),
                arg: Box::new(b_ref2),
            },
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program_body = mk_term(
            TermKind::Let {
                name: b_sym,
                name_ty: array_ty(i32_ty()),
                rhs: Box::new(producer),
                body: Box::new(body),
            },
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: program_body,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let result = fuse_maps(program);

        // Should still be a Let (no fusion because b is used twice)
        assert!(matches!(&result.defs[0].body.kind, TermKind::Let { .. }));
    }

    // -------------------------------------------------------------------------
    // Test: zip-fused producer: map(g, map(f, zip(a,b)))
    // -------------------------------------------------------------------------
    #[test]
    fn test_zip_fused_producer() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        let a_sym = symbols.alloc("a".to_string());
        let b_sym = symbols.alloc("b".to_string());
        let intermediate_sym = symbols.alloc("inter".to_string());
        let x1_sym = symbols.alloc("x1".to_string());
        let x2_sym = symbols.alloc("x2".to_string());
        let y_sym = symbols.alloc("y".to_string());

        let _tuple_ty = Type::Constructed(TypeName::Tuple(2), vec![i32_ty(), f32_ty()]);

        // f: (i32, f32) → i32 — takes two params (zip-fused)
        let f = Lambda {
            params: vec![(x1_sym, i32_ty()), (x2_sym, f32_ty())],
            body: Box::new(mk_term(TermKind::Var(x1_sym), i32_ty(), &mut term_ids)),
            ret_ty: i32_ty(),
            captures: vec![],
        };

        // Producer: map(f, [a, b]) with zip-fused inputs
        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);
        let b = mk_term(TermKind::Var(b_sym), array_ty(f32_ty()), &mut term_ids);
        let producer = mk_term(
            TermKind::Soac(SoacOp::Map {
                lam: f,
                inputs: vec![ArrayExpr::Ref(Box::new(a)), ArrayExpr::Ref(Box::new(b))],
            }),
            array_ty(i32_ty()),
            &mut term_ids,
        );

        // g: i32 → i32
        let g = mk_lambda1(
            y_sym,
            i32_ty(),
            mk_term(TermKind::Var(y_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );

        // Consumer: map(g, inter) with single input
        let inter_ref = mk_term(TermKind::Var(intermediate_sym), array_ty(i32_ty()), &mut term_ids);
        let consumer = mk_map(g, inter_ref, array_ty(i32_ty()), &mut term_ids);

        // let inter = map(f, [a, b]) in map(g, inter)
        let program_body = mk_term(
            TermKind::Let {
                name: intermediate_sym,
                name_ty: array_ty(i32_ty()),
                rhs: Box::new(producer),
                body: Box::new(consumer),
            },
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: program_body,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let fused = fuse_maps(program);

        // Should be a Map with [a, b] inputs (producer's multi-inputs preserved)
        match &fused.defs[0].body.kind {
            TermKind::Soac(SoacOp::Map { lam, inputs }) => {
                assert_eq!(inputs.len(), 2);
                // Lambda should have f's params (x1, x2)
                assert_eq!(lam.params.len(), 2);
                assert_eq!(lam.params[0].0, x1_sym);
                assert_eq!(lam.params[1].0, x2_sym);
            }
            other => panic!("Expected fused Map with 2 inputs, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Test: consumer with multiple inputs (no fusion)
    // -------------------------------------------------------------------------
    #[test]
    fn test_consumer_multi_input_no_fusion() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        let a_sym = symbols.alloc("a".to_string());
        let b_sym = symbols.alloc("b".to_string());
        let other_sym = symbols.alloc("other".to_string());
        let x_sym = symbols.alloc("x".to_string());
        let y1_sym = symbols.alloc("y1".to_string());
        let y2_sym = symbols.alloc("y2".to_string());

        let f = mk_lambda1(
            x_sym,
            i32_ty(),
            mk_term(TermKind::Var(x_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );

        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);
        let producer = mk_map(f, a, array_ty(i32_ty()), &mut term_ids);

        // g: (i32, i32) → i32, takes two inputs
        let g = Lambda {
            params: vec![(y1_sym, i32_ty()), (y2_sym, i32_ty())],
            body: Box::new(mk_term(TermKind::Var(y1_sym), i32_ty(), &mut term_ids)),
            ret_ty: i32_ty(),
            captures: vec![],
        };

        // Consumer: map(g, [b, other]) — b plus another array
        let b_ref = mk_term(TermKind::Var(b_sym), array_ty(i32_ty()), &mut term_ids);
        let other = mk_term(TermKind::Var(other_sym), array_ty(i32_ty()), &mut term_ids);
        let consumer = mk_term(
            TermKind::Soac(SoacOp::Map {
                lam: g,
                inputs: vec![ArrayExpr::Ref(Box::new(b_ref)), ArrayExpr::Ref(Box::new(other))],
            }),
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program_body = mk_term(
            TermKind::Let {
                name: b_sym,
                name_ty: array_ty(i32_ty()),
                rhs: Box::new(producer),
                body: Box::new(consumer),
            },
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: program_body,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let result = fuse_maps(program);

        // Should NOT fuse — consumer has multiple inputs
        assert!(matches!(&result.defs[0].body.kind, TermKind::Let { .. }));
    }

    // -------------------------------------------------------------------------
    // Test: inline map(f, map(g, a)) — no Let binding
    // -------------------------------------------------------------------------
    #[test]
    fn test_inline_map_fusion() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        let a_sym = symbols.alloc("a".to_string());
        let x_sym = symbols.alloc("x".to_string());
        let y_sym = symbols.alloc("y".to_string());

        // Inner: map(g, a) where g(x) = x
        let g = mk_lambda1(
            x_sym,
            i32_ty(),
            mk_term(TermKind::Var(x_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);
        let inner_map = mk_map(g, a, array_ty(i32_ty()), &mut term_ids);

        // Outer: map(f, inner_map) where f(y) = y — inner_map is inline (Ref)
        let f = mk_lambda1(
            y_sym,
            i32_ty(),
            mk_term(TermKind::Var(y_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let outer = mk_term(
            TermKind::Soac(SoacOp::Map {
                lam: f,
                inputs: vec![ArrayExpr::Ref(Box::new(inner_map))],
            }),
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: outer,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let fused = fuse_maps(program);

        // Should be a single Map with a's input, param x (inner g's param)
        match &fused.defs[0].body.kind {
            TermKind::Soac(SoacOp::Map { lam, inputs }) => {
                assert_eq!(inputs.len(), 1);
                match &inputs[0] {
                    ArrayExpr::Ref(t) => assert!(matches!(&t.kind, TermKind::Var(s) if *s == a_sym)),
                    other => panic!("Expected Ref(a), got {:?}", other),
                }
                // Lambda should have g's param (x), not f's param (y)
                assert_eq!(lam.params.len(), 1);
                assert_eq!(lam.params[0].0, x_sym);
            }
            other => panic!("Expected fused Map, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Test: inline chain map(f, map(g, map(h, a))) — no Let bindings
    // -------------------------------------------------------------------------
    #[test]
    fn test_inline_chain_of_three() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        let a_sym = symbols.alloc("a".to_string());
        let x_sym = symbols.alloc("x".to_string());
        let y_sym = symbols.alloc("y".to_string());
        let z_sym = symbols.alloc("z".to_string());

        // h(x) = x
        let h = mk_lambda1(
            x_sym,
            i32_ty(),
            mk_term(TermKind::Var(x_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);
        let inner = mk_map(h, a, array_ty(i32_ty()), &mut term_ids);

        // g(y) = y
        let g = mk_lambda1(
            y_sym,
            i32_ty(),
            mk_term(TermKind::Var(y_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let middle = mk_term(
            TermKind::Soac(SoacOp::Map {
                lam: g,
                inputs: vec![ArrayExpr::Ref(Box::new(inner))],
            }),
            array_ty(i32_ty()),
            &mut term_ids,
        );

        // f(z) = z
        let f = mk_lambda1(
            z_sym,
            i32_ty(),
            mk_term(TermKind::Var(z_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let outer = mk_term(
            TermKind::Soac(SoacOp::Map {
                lam: f,
                inputs: vec![ArrayExpr::Ref(Box::new(middle))],
            }),
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: outer,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let fused = fuse_maps(program);

        // All three fused into one Map over a
        match &fused.defs[0].body.kind {
            TermKind::Soac(SoacOp::Map { lam, inputs }) => {
                assert_eq!(inputs.len(), 1);
                match &inputs[0] {
                    ArrayExpr::Ref(t) => assert!(matches!(&t.kind, TermKind::Var(s) if *s == a_sym)),
                    other => panic!("Expected Ref(a), got {:?}", other),
                }
                // Innermost param (h's param x) should be the lambda param
                assert_eq!(lam.params.len(), 1);
                assert_eq!(lam.params[0].0, x_sym);
            }
            other => panic!("Expected fully fused Map, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Test: zip-fused consumer — map(f, zip(map(g, a), b))
    // One input is an inline nested map, the other is not
    // -------------------------------------------------------------------------
    #[test]
    fn test_zip_fused_consumer_inline() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        let a_sym = symbols.alloc("a".to_string());
        let b_sym = symbols.alloc("b".to_string());
        let x_sym = symbols.alloc("x".to_string());
        let y1_sym = symbols.alloc("y1".to_string());
        let y2_sym = symbols.alloc("y2".to_string());

        // Inner: map(g, a) where g(x) = x
        let g = mk_lambda1(
            x_sym,
            i32_ty(),
            mk_term(TermKind::Var(x_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);
        let inner_map = mk_map(g, a, array_ty(i32_ty()), &mut term_ids);

        // Outer: map(f, zip(map(g, a), b)) — zip already absorbed into multi-input
        // f(y1, y2) = y1  (takes two params from zip)
        let f = Lambda {
            params: vec![(y1_sym, i32_ty()), (y2_sym, f32_ty())],
            body: Box::new(mk_term(TermKind::Var(y1_sym), i32_ty(), &mut term_ids)),
            ret_ty: i32_ty(),
            captures: vec![],
        };
        let b = mk_term(TermKind::Var(b_sym), array_ty(f32_ty()), &mut term_ids);
        let outer = mk_term(
            TermKind::Soac(SoacOp::Map {
                lam: f,
                inputs: vec![
                    ArrayExpr::Ref(Box::new(inner_map)), // will be fused
                    ArrayExpr::Ref(Box::new(b)),         // stays as-is
                ],
            }),
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: outer,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let fused = fuse_maps(program);

        // The inner map should be fused: y1's slot replaced by g's param x,
        // input[0] is now Ref(a), input[1] is Ref(b)
        match &fused.defs[0].body.kind {
            TermKind::Soac(SoacOp::Map { lam, inputs }) => {
                assert_eq!(inputs.len(), 2);
                // First input: a (was map(g, a), now fused)
                match &inputs[0] {
                    ArrayExpr::Ref(t) => assert!(matches!(&t.kind, TermKind::Var(s) if *s == a_sym)),
                    other => panic!("Expected Ref(a), got {:?}", other),
                }
                // Second input: b (unchanged)
                match &inputs[1] {
                    ArrayExpr::Ref(t) => assert!(matches!(&t.kind, TermKind::Var(s) if *s == b_sym)),
                    other => panic!("Expected Ref(b), got {:?}", other),
                }
                // Params: x (from g), y2 (from f — kept)
                assert_eq!(lam.params.len(), 2);
                assert_eq!(lam.params[0].0, x_sym);
                assert_eq!(lam.params[1].0, y2_sym);
            }
            other => panic!("Expected fused Map with 2 inputs, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Test: map-zip-map — map(f, zip(map(g, a), map(h, b)))
    // Both zip inputs are inline maps, both should be fused
    // -------------------------------------------------------------------------
    #[test]
    fn test_map_zip_map() {
        let mut symbols = SymbolTable::default();
        let mut term_ids = TermIdSource::new();

        let a_sym = symbols.alloc("a".to_string());
        let b_sym = symbols.alloc("b".to_string());
        let x_sym = symbols.alloc("x".to_string()); // g's param
        let w_sym = symbols.alloc("w".to_string()); // h's param
        let y1_sym = symbols.alloc("y1".to_string()); // f's param 1
        let y2_sym = symbols.alloc("y2".to_string()); // f's param 2

        // g(x) = x : i32 → i32
        let g = mk_lambda1(
            x_sym,
            i32_ty(),
            mk_term(TermKind::Var(x_sym), i32_ty(), &mut term_ids),
            i32_ty(),
        );
        let a = mk_term(TermKind::Var(a_sym), array_ty(i32_ty()), &mut term_ids);
        let map_g_a = mk_map(g, a, array_ty(i32_ty()), &mut term_ids);

        // h(w) = w : f32 → f32
        let h = mk_lambda1(
            w_sym,
            f32_ty(),
            mk_term(TermKind::Var(w_sym), f32_ty(), &mut term_ids),
            f32_ty(),
        );
        let b = mk_term(TermKind::Var(b_sym), array_ty(f32_ty()), &mut term_ids);
        let map_h_b = mk_map(h, b, array_ty(f32_ty()), &mut term_ids);

        // f(y1: i32, y2: f32) = y1
        let f = Lambda {
            params: vec![(y1_sym, i32_ty()), (y2_sym, f32_ty())],
            body: Box::new(mk_term(TermKind::Var(y1_sym), i32_ty(), &mut term_ids)),
            ret_ty: i32_ty(),
            captures: vec![],
        };

        // map(f, zip(map(g, a), map(h, b))) — zip absorbed
        let outer = mk_term(
            TermKind::Soac(SoacOp::Map {
                lam: f,
                inputs: vec![
                    ArrayExpr::Ref(Box::new(map_g_a)),
                    ArrayExpr::Ref(Box::new(map_h_b)),
                ],
            }),
            array_ty(i32_ty()),
            &mut term_ids,
        );

        let program = Program {
            defs: vec![Def {
                name: symbols.alloc("main".to_string()),
                ty: array_ty(i32_ty()),
                body: outer,
                meta: super::super::DefMeta::Function,
                arity: 0,
            }],
            uniforms: vec![],
            storage: vec![],
            symbols,
        };

        let fused = fuse_maps(program);

        match &fused.defs[0].body.kind {
            TermKind::Soac(SoacOp::Map { lam, inputs }) => {
                // Both intermediates eliminated: inputs are [a, b]
                assert_eq!(inputs.len(), 2);
                match &inputs[0] {
                    ArrayExpr::Ref(t) => assert!(matches!(&t.kind, TermKind::Var(s) if *s == a_sym)),
                    other => panic!("Expected Ref(a), got {:?}", other),
                }
                match &inputs[1] {
                    ArrayExpr::Ref(t) => assert!(matches!(&t.kind, TermKind::Var(s) if *s == b_sym)),
                    other => panic!("Expected Ref(b), got {:?}", other),
                }
                // Params: x (from g), w (from h)
                assert_eq!(lam.params.len(), 2);
                assert_eq!(lam.params[0].0, x_sym);
                assert_eq!(lam.params[1].0, w_sym);
            }
            other => panic!("Expected fused Map, got {:?}", other),
        }
    }
}
