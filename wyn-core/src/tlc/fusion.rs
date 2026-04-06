//! SOAC fusion pass for TLC.
//!
//! Fuses consecutive SOAC operations to eliminate intermediate arrays:
//!
//! ```text
//! let b = map(f, a) in map(g, b)         =>  map(g∘f, a)
//! let b = map(f, a) in reduce(op, ne, b) =>  reduce(op∘f, ne, a)
//! ```
//!
//! Operates before defunctionalization — lambdas are full expressions
//! and captures are empty, making composition straightforward.

use crate::ast::{Span, TypeName};
use crate::{SymbolId, SymbolTable};
use polytype::Type;

use std::collections::HashMap;

use super::fusion_summary::DefSummary;
use super::{
    ArrayExpr, Def, Lambda, LoopKind, Place, Program, ReduceProps, SoacOp, Term, TermIdSource, TermKind,
};

type Summaries = HashMap<SymbolId, DefSummary>;

// =============================================================================
// Public entry point
// =============================================================================

/// Fuse consecutive SOAC operations in a TLC program.
///
/// 1. Normalizes (lifts SOACs into let bindings)
/// 2. Computes function summaries for interprocedural fusion
/// 3. Fuses map-map, map-reduce chains within and across function bodies
pub fn fuse_maps(program: Program) -> Program {
    use super::fusion_summary::{DefSummary, propagate_summaries, summarize_program};

    // Normalize: lift SOACs out of nested positions into let bindings
    let program = super::normalize::normalize(program);

    // Compute function summaries for interprocedural fusion
    let mut summaries = summarize_program(&program);
    propagate_summaries(&program, &mut summaries);

    let mut symbols = program.symbols;
    let mut term_ids = TermIdSource::new();

    let defs = program
        .defs
        .into_iter()
        .map(|def| {
            let body = transform_term_s(def.body, &mut symbols, &mut term_ids, &summaries);
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
fn transform_term_s(
    term: Term,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
    sums: &Summaries,
) -> Term {
    let kind = match term.kind {
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            // Transform children first (bottom-up)
            let rhs = transform_term_s(*rhs, symbols, term_ids, sums);
            let body = transform_term_s(*body, symbols, term_ids, sums);

            // Try to fuse
            if let Some(fused) = try_fuse_let(name, &name_ty, &rhs, body.clone(), symbols, term_ids, sums) {
                return fused;
            }

            TermKind::Let {
                name,
                name_ty,
                rhs: Box::new(rhs),
                body: Box::new(body),
            }
        }

        TermKind::App { func, args } => TermKind::App {
            func: Box::new(transform_term_s(*func, symbols, term_ids, sums)),
            args: args.into_iter().map(|a| transform_term_s(a, symbols, term_ids, sums)).collect(),
        },

        TermKind::Lambda(lam) => TermKind::Lambda(transform_lambda(lam, symbols, term_ids, sums)),

        TermKind::If {
            cond,
            then_branch,
            else_branch,
        } => TermKind::If {
            cond: Box::new(transform_term_s(*cond, symbols, term_ids, sums)),
            then_branch: Box::new(transform_term_s(*then_branch, symbols, term_ids, sums)),
            else_branch: Box::new(transform_term_s(*else_branch, symbols, term_ids, sums)),
        },

        TermKind::Loop {
            loop_var,
            loop_var_ty,
            init,
            init_bindings,
            kind,
            body,
        } => {
            let init = transform_term_s(*init, symbols, term_ids, sums);
            let init_bindings = init_bindings
                .into_iter()
                .map(|(n, ty, e)| (n, ty, transform_term_s(e, symbols, term_ids, sums)))
                .collect();
            let kind = transform_loop_kind(kind, symbols, term_ids, sums);
            let body = transform_term_s(*body, symbols, term_ids, sums);
            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init: Box::new(init),
                init_bindings,
                kind,
                body: Box::new(body),
            }
        }

        TermKind::Soac(soac) => TermKind::Soac(transform_soac(soac, symbols, term_ids, sums)),

        TermKind::ArrayExpr(ae) => TermKind::ArrayExpr(transform_array_expr(ae, symbols, term_ids, sums)),

        TermKind::Force(inner) => {
            TermKind::Force(Box::new(transform_term_s(*inner, symbols, term_ids, sums)))
        }

        TermKind::Pack {
            exists_ty,
            dims,
            value,
        } => TermKind::Pack {
            exists_ty,
            dims,
            value: Box::new(transform_term_s(*value, symbols, term_ids, sums)),
        },

        TermKind::Unpack {
            scrut,
            dim_binders,
            value_binder,
            body,
        } => TermKind::Unpack {
            scrut: Box::new(transform_term_s(*scrut, symbols, term_ids, sums)),
            dim_binders,
            value_binder,
            body: Box::new(transform_term_s(*body, symbols, term_ids, sums)),
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

fn transform_lambda(
    lam: Lambda,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
    sums: &Summaries,
) -> Lambda {
    Lambda {
        params: lam.params,
        body: Box::new(transform_term_s(*lam.body, symbols, term_ids, sums)),
        ret_ty: lam.ret_ty,
        captures: lam
            .captures
            .into_iter()
            .map(|(s, ty, t)| (s, ty, transform_term_s(t, symbols, term_ids, sums)))
            .collect(),
    }
}

fn transform_soac(
    soac: SoacOp,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
    sums: &Summaries,
) -> SoacOp {
    match soac {
        SoacOp::Map { lam, inputs } => {
            // First: recurse into children (bottom-up)
            let lam = transform_lambda(lam, symbols, term_ids, sums);
            let inputs: Vec<_> =
                inputs.into_iter().map(|ae| transform_array_expr(ae, symbols, term_ids, sums)).collect();

            // Then: fuse any inline nested Maps from inputs
            fuse_inline_map_inputs(lam, inputs, symbols, term_ids, sums)
        }
        SoacOp::Reduce { op, ne, input, props } => SoacOp::Reduce {
            op: transform_lambda(op, symbols, term_ids, sums),
            ne: Box::new(transform_term_s(*ne, symbols, term_ids, sums)),
            input: transform_array_expr(input, symbols, term_ids, sums),
            props,
        },
        SoacOp::Scan { op, ne, input } => SoacOp::Scan {
            op: transform_lambda(op, symbols, term_ids, sums),
            ne: Box::new(transform_term_s(*ne, symbols, term_ids, sums)),
            input: transform_array_expr(input, symbols, term_ids, sums),
        },
        SoacOp::Filter { pred, input } => SoacOp::Filter {
            pred: transform_lambda(pred, symbols, term_ids, sums),
            input: transform_array_expr(input, symbols, term_ids, sums),
        },
        SoacOp::Scatter {
            dest,
            indices,
            values,
        } => SoacOp::Scatter {
            dest: transform_place(dest, symbols, term_ids, sums),
            indices: transform_array_expr(indices, symbols, term_ids, sums),
            values: transform_array_expr(values, symbols, term_ids, sums),
        },
        SoacOp::ReduceByIndex {
            dest,
            op,
            ne,
            indices,
            values,
            props,
        } => SoacOp::ReduceByIndex {
            dest: transform_place(dest, symbols, term_ids, sums),
            op: transform_lambda(op, symbols, term_ids, sums),
            ne: Box::new(transform_term_s(*ne, symbols, term_ids, sums)),
            indices: transform_array_expr(indices, symbols, term_ids, sums),
            values: transform_array_expr(values, symbols, term_ids, sums),
            props,
        },
    }
}

fn transform_array_expr(
    ae: ArrayExpr,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
    sums: &Summaries,
) -> ArrayExpr {
    match ae {
        ArrayExpr::Ref(t) => ArrayExpr::Ref(Box::new(transform_term_s(*t, symbols, term_ids, sums))),
        ArrayExpr::Zip(exprs) => ArrayExpr::Zip(
            exprs.into_iter().map(|e| transform_array_expr(e, symbols, term_ids, sums)).collect(),
        ),
        ArrayExpr::Soac(op) => ArrayExpr::Soac(Box::new(transform_soac(*op, symbols, term_ids, sums))),
        ArrayExpr::Generate {
            shape,
            index_fn,
            elem_ty,
        } => ArrayExpr::Generate {
            shape,
            index_fn: transform_lambda(index_fn, symbols, term_ids, sums),
            elem_ty,
        },
        ArrayExpr::Literal(terms) => ArrayExpr::Literal(
            terms.into_iter().map(|t| transform_term_s(t, symbols, term_ids, sums)).collect(),
        ),
        ArrayExpr::Range { start, len } => ArrayExpr::Range {
            start: Box::new(transform_term_s(*start, symbols, term_ids, sums)),
            len: Box::new(transform_term_s(*len, symbols, term_ids, sums)),
        },
        ArrayExpr::StorageBuffer { .. } => unreachable!("StorageBuffer introduced after fusion"),
    }
}

fn transform_loop_kind(
    kind: LoopKind,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
    sums: &Summaries,
) -> LoopKind {
    match kind {
        LoopKind::For { var, var_ty, iter } => LoopKind::For {
            var,
            var_ty,
            iter: Box::new(transform_term_s(*iter, symbols, term_ids, sums)),
        },
        LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
            var,
            var_ty,
            bound: Box::new(transform_term_s(*bound, symbols, term_ids, sums)),
        },
        LoopKind::While { cond } => LoopKind::While {
            cond: Box::new(transform_term_s(*cond, symbols, term_ids, sums)),
        },
    }
}

fn transform_place(
    place: Place,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
    sums: &Summaries,
) -> Place {
    match place {
        Place::BufferSlice {
            base,
            offset,
            shape,
            elem_ty,
        } => Place::BufferSlice {
            base: Box::new(transform_term_s(*base, symbols, term_ids, sums)),
            offset: Box::new(transform_term_s(*offset, symbols, term_ids, sums)),
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
    sums: &Summaries,
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
    sums: &Summaries,
) -> Option<Term> {
    // 1. Check that the RHS is a Map SOAC (the producer), either directly
    //    or via a function call with a Map summary.
    let (producer_lam, producer_inputs) = match &rhs.kind {
        TermKind::Soac(SoacOp::Map { lam, inputs }) => (lam.clone(), inputs.clone()),

        // Interprocedural: function call where callee has a Map or ProducesMap summary
        TermKind::App { ref func, ref args } => {
            if let TermKind::Var(callee_sym) = &func.kind {
                match sums.get(callee_sym) {
                    Some(DefSummary::Map { lam, param_idx }) if *param_idx < args.len() => {
                        let input_term = args[*param_idx].clone();
                        let input_expr = ArrayExpr::Ref(Box::new(input_term));
                        (lam.clone(), vec![input_expr])
                    }
                    Some(DefSummary::ProducesMap {
                        lam,
                        inputs,
                        callee_params,
                    }) => {
                        if args.len() != callee_params.len() {
                            return None;
                        }
                        let mut result_lam = lam.clone();
                        let mut result_inputs = inputs.clone();
                        for (i, (param_sym, _)) in callee_params.iter().enumerate() {
                            let arg_sym = match &args[i].kind {
                                TermKind::Var(s) => *s,
                                _ => return None,
                            };
                            result_lam.body = Box::new(substitute_sym(
                                *result_lam.body,
                                *param_sym,
                                arg_sym,
                                term_ids,
                            ));
                            result_inputs = result_inputs
                                .into_iter()
                                .map(|inp| {
                                    substitute_sym_array_expr(inp, *param_sym, arg_sym, term_ids)
                                })
                                .collect();
                        }
                        (result_lam, result_inputs)
                    }
                    _ => return None,
                }
            } else {
                return None;
            }
        }

        _ => return None,
    };

    // 2. Check that `name` is used exactly once in the body
    if count_uses(&body, name) != 1 {
        return None;
    }

    // 3. Find the consumer: the single use must be as the sole input to a Map or Reduce
    let consumer_info = find_fusion_consumer(&body, name, sums)?;

    // 4. Build the fused SOAC based on consumer kind
    let (fused_soac, result_ty) = match &consumer_info.kind {
        ConsumerKind::Map { lam, result_ty } => {
            let composed = compose_lambdas(producer_lam.clone(), lam.clone(), rhs.span, symbols, term_ids);
            (
                SoacOp::Map {
                    lam: composed,
                    inputs: producer_inputs.clone(),
                },
                result_ty.clone(),
            )
        }
        ConsumerKind::Reduce {
            op,
            ne,
            props,
            result_ty,
        } => {
            // Phase 1 restrictions:
            // - only fuse single-input map producers
            // - reduce op must have exactly 2 params (acc, elem)
            if producer_inputs.len() != 1 || op.params.len() != 2 {
                return None;
            }
            let composed_op =
                compose_map_reduce(producer_lam.clone(), op.clone(), rhs.span, symbols, term_ids);
            (
                SoacOp::Reduce {
                    op: composed_op,
                    ne: Box::new(ne.clone()),
                    input: producer_inputs[0].clone(),
                    props: props.clone(),
                },
                result_ty.clone(),
            )
        }
    };

    let fused_term = Term {
        id: term_ids.next_id(),
        ty: result_ty,
        span: rhs.span,
        kind: TermKind::Soac(fused_soac),
    };

    // 5. Replace the consumer in the body with the fused SOAC
    Some(replace_fusion_consumer(&body, name, fused_term, term_ids))
}

/// The kind of SOAC consumer found in the body.
enum ConsumerKind {
    Map {
        lam: Lambda,
        result_ty: Type<TypeName>,
    },
    Reduce {
        op: Lambda,
        ne: Term,
        props: ReduceProps,
        result_ty: Type<TypeName>,
    },
}

/// Information about a consumer SOAC found in the body.
struct ConsumerInfo {
    kind: ConsumerKind,
}

/// Find a consumer SOAC (Map, Reduce, or function call with summary) in the body
/// that uses `name` as its sole input.
fn find_fusion_consumer(term: &Term, name: SymbolId, sums: &Summaries) -> Option<ConsumerInfo> {
    match &term.kind {
        // Map consumer
        TermKind::Soac(SoacOp::Map { lam, inputs }) => {
            if is_sole_ref_to(inputs, name) {
                Some(ConsumerInfo {
                    kind: ConsumerKind::Map {
                        lam: lam.clone(),
                        result_ty: term.ty.clone(),
                    },
                })
            } else {
                None
            }
        }

        // Reduce consumer
        TermKind::Soac(SoacOp::Reduce { op, ne, input, props }) => {
            if is_sole_ref_to_single(input, name) {
                Some(ConsumerInfo {
                    kind: ConsumerKind::Reduce {
                        op: op.clone(),
                        ne: (**ne).clone(),
                        props: props.clone(),
                        result_ty: term.ty.clone(),
                    },
                })
            } else {
                None
            }
        }

        // Interprocedural: function call where callee has a Map or Reduce summary
        // and `name` is passed as the summarized parameter
        TermKind::App { ref func, ref args } => {
            if let TermKind::Var(callee_sym) = &func.kind {
                match sums.get(callee_sym) {
                    Some(DefSummary::Map { lam, param_idx }) if *param_idx < args.len() => {
                        if let TermKind::Var(sym) = &args[*param_idx].kind {
                            if *sym == name {
                                return Some(ConsumerInfo {
                                    kind: ConsumerKind::Map {
                                        lam: lam.clone(),
                                        result_ty: term.ty.clone(),
                                    },
                                });
                            }
                        }
                    }
                    Some(DefSummary::Reduce {
                        op,
                        ne,
                        param_idx,
                        props,
                    }) if *param_idx < args.len() => {
                        if let TermKind::Var(sym) = &args[*param_idx].kind {
                            if *sym == name {
                                if op.params.len() >= 2 {
                                    return Some(ConsumerInfo {
                                        kind: ConsumerKind::Reduce {
                                            op: op.clone(),
                                            ne: ne.clone(),
                                            props: props.clone(),
                                            result_ty: term.ty.clone(),
                                        },
                                    });
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            // Interprocedural check didn't match — recurse into the arg that uses name
            for arg in args {
                if count_uses(arg, name) == 1 {
                    return find_fusion_consumer(arg, name, sums);
                }
            }
            None
        }

        // The consumer might be nested in a Let body
        TermKind::Let {
            name: inner_name,
            rhs,
            body,
            ..
        } => {
            if count_uses(rhs, name) == 1 {
                find_fusion_consumer(rhs, name, sums)
            } else if *inner_name != name && count_uses(body, name) == 1 {
                find_fusion_consumer(body, name, sums)
            } else {
                None
            }
        }

        _ => None,
    }
}

/// Check if a single ArrayExpr is a Ref pointing to `name`.
fn is_sole_ref_to_single(input: &ArrayExpr, name: SymbolId) -> bool {
    match input {
        ArrayExpr::Ref(t) => matches!(&t.kind, TermKind::Var(sym) if *sym == name),
        _ => false,
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

/// Replace the consumer SOAC (which uses `name` as sole input) with `replacement`.
/// This traverses the body to find and replace the exact consumer.
fn replace_fusion_consumer(
    term: &Term,
    name: SymbolId,
    replacement: Term,
    term_ids: &mut TermIdSource,
) -> Term {
    match &term.kind {
        TermKind::Soac(SoacOp::Map { inputs, .. }) => {
            if is_sole_ref_to(inputs, name) {
                replacement
            } else {
                term.clone()
            }
        }

        TermKind::Soac(SoacOp::Reduce { input, .. }) => {
            if is_sole_ref_to_single(input, name) {
                replacement
            } else {
                term.clone()
            }
        }

        // Interprocedural consumer: function call that uses `name` as an argument
        TermKind::App { .. } => {
            // If this App uses `name` somewhere in its args, it might be the consumer
            if count_uses(term, name) == 1 { replacement } else { term.clone() }
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
                        rhs: Box::new(replace_fusion_consumer(rhs, name, replacement, term_ids)),
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
                        body: Box::new(replace_fusion_consumer(body, name, replacement, term_ids)),
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

/// Compose a map lambda with a reduce operator:
/// `map_lam: A → B` and `reduce_op: (Acc, B) → Acc` produces `(Acc, A) → Acc`.
///
/// ```text
/// Lambda {
///     params: [reduce_op.params[0], map_lam.params[0]],
///     body: let fresh = map_lam.body in reduce_op.body[elem → fresh],
///     ret_ty: reduce_op.ret_ty,
///     captures: [],
/// }
/// ```
fn compose_map_reduce(
    map_lam: Lambda,
    reduce_op: Lambda,
    span: Span,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Lambda {
    let fresh_sym = symbols.alloc("_fused".to_string());
    let intermediate_ty = map_lam.ret_ty.clone();

    // The reduce operator has params [acc, elem].
    // Substitute elem with fresh in the reduce body.
    let elem_param = reduce_op.params[1].0;
    let op_body_substituted = substitute_sym(*reduce_op.body, elem_param, fresh_sym, term_ids);

    // Build: let fresh = map_lam.body in reduce_op.body[elem → fresh]
    let composed_body = Term {
        id: term_ids.next_id(),
        ty: reduce_op.ret_ty.clone(),
        span,
        kind: TermKind::Let {
            name: fresh_sym,
            name_ty: intermediate_ty,
            rhs: map_lam.body,
            body: Box::new(op_body_substituted),
        },
    };

    // New params: [acc (from reduce), input_elem (from map)]
    Lambda {
        params: vec![reduce_op.params[0].clone(), map_lam.params[0].clone()],
        body: Box::new(composed_body),
        ret_ty: reduce_op.ret_ty,
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

        TermKind::App { func, args } => {
            count_uses(func, sym) + args.iter().map(|a| count_uses(a, sym)).sum::<usize>()
        }

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
        ArrayExpr::StorageBuffer { .. } => unreachable!("StorageBuffer introduced after fusion"),
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

        TermKind::App { func, args } => Term {
            id: term_ids.next_id(),
            ty: term.ty,
            span: term.span,
            kind: TermKind::App {
                func: Box::new(substitute_sym(*func, old, new, term_ids)),
                args: args.into_iter().map(|a| substitute_sym(a, old, new, term_ids)).collect(),
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
        ArrayExpr::StorageBuffer { .. } => unreachable!("StorageBuffer introduced after fusion"),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[path = "fusion_tests.rs"]
mod fusion_tests;
