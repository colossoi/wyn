//! Graph-driven SOAC fusion pass for TLC.
//!
//! Fuses consecutive SOAC operations to eliminate intermediate arrays.
//! Uses the ProducerGraph to find producer-consumer pairs and
//! ArraySemantics to determine fusibility.
//!
//! ```text
//! let b = map(f, a) in map(g, b)         =>  map(g∘f, a)
//! let b = map(f, a) in reduce(op, ne, b) =>  reduce(op∘f, ne, a)
//! ```

use crate::ast::{Span, TypeName};
use crate::{SymbolId, SymbolTable};
use polytype::Type;

use std::collections::HashMap;

use super::array_semantics::{ArraySemantics, FunctionSummary, FusionKind, can_fuse, summarize_program};
use super::producer_graph::{self, ProducerEdge};
use super::{ArrayExpr, Def, Lambda, Program, SoacOp, Term, TermIdSource, TermKind, extract_lambda_params};

type Summaries = HashMap<SymbolId, FunctionSummary>;

// =============================================================================
// Public entry point
// =============================================================================

/// Fuse SOAC operations in a TLC program.
///
/// 1. Normalizes (lifts SOACs into let bindings)
/// 2. Computes function summaries for interprocedural analysis
/// 3. Builds producer graphs and fuses producer-consumer pairs
/// 4. Repeats until fixpoint
pub fn run(program: Program) -> Program {
    // Normalize: lift SOACs out of nested positions into let bindings
    let mut program = super::normalize::normalize(program);

    let mut changed = true;
    while changed {
        changed = false;

        let summaries = summarize_program(&program);

        let mut symbols = program.symbols;
        let def_syms = program.def_syms;
        let mut term_ids = TermIdSource::new();

        let defs = program
            .defs
            .into_iter()
            .map(|def| {
                // Bottom-up: fuse children first, then try graph-driven fusion
                let new_body = fuse_term(def.body, &summaries, &mut symbols, &mut term_ids, &def_syms);
                let (new_body, did_fuse) =
                    fuse_def_body(new_body, &summaries, &mut symbols, &mut term_ids, &def_syms);
                if did_fuse {
                    changed = true;
                }
                // Inline map fusion: map(f, map(g, a)) → map(f∘g, a)
                let new_body = fuse_inline_maps(new_body, &mut symbols, &mut term_ids);
                Def {
                    body: new_body,
                    ..def
                }
            })
            .collect();

        program = Program {
            defs,
            uniforms: program.uniforms,
            storage: program.storage,
            symbols,
            def_syms: def_syms.clone(),
        };
    }

    program
}

// =============================================================================
// Per-def fusion
// =============================================================================

/// Bottom-up: recurse into children, applying graph-driven fusion at each
/// sub-expression that contains a Let chain with SOAC producers/consumers.
fn fuse_term(
    term: Term,
    summaries: &Summaries,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
    def_syms: &HashMap<String, SymbolId>,
) -> Term {
    let term = term.map_children(&mut |child| fuse_term(child, summaries, symbols, term_ids, def_syms));

    if matches!(term.kind, TermKind::Let { .. }) {
        let (fused, _) = fuse_def_body(term, summaries, symbols, term_ids, def_syms);
        return fused;
    }

    term
}

/// Fuse within a body term. Returns the new body and whether any fusion happened.
fn fuse_def_body(
    body: Term,
    summaries: &Summaries,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
    def_syms: &HashMap<String, SymbolId>,
) -> (Term, bool) {
    let (params, inner) = extract_lambda_params(&body);
    let param_syms: Vec<SymbolId> = params.iter().map(|(s, _)| *s).collect();

    let graph = producer_graph::build_producer_graph(&inner, &param_syms, summaries, symbols, def_syms);

    if graph.node_count() < 2 {
        return (body, false);
    }

    // Find fusible edges
    let fusible: Vec<ProducerEdge> = graph
        .edges()
        .iter()
        .filter(|e| {
            let p = graph.node(e.producer);
            let c = graph.node(e.consumer);
            let fk = can_fuse(&p.semantics, &c.semantics);
            if fk == FusionKind::NotFusible {
                return false;
            }
            if p.use_count != 1 || p.binding.is_none() {
                return false;
            }
            if fk == FusionKind::ComposeElementwise {
                if let ArraySemantics::Elementwise { inputs, .. } = &c.semantics {
                    if inputs.len() != 1 {
                        return false;
                    }
                }
            }
            true
        })
        .cloned()
        .collect();

    if fusible.is_empty() {
        return (body, false);
    }

    // Apply the first fusible edge (then the outer fixpoint loop reruns)
    let edge = &fusible[0];
    let producer = graph.node(edge.producer);
    let consumer = graph.node(edge.consumer);
    let prod_sym = producer.binding.unwrap();
    let cons_sym = consumer.binding; // None if tail expression
    let fusion_kind = can_fuse(&producer.semantics, &consumer.semantics);

    // Build the fused SOAC term from semantics
    let fused_soac_term = build_fused_from_semantics(
        &producer.semantics,
        &consumer.semantics,
        &fusion_kind,
        consumer.ty.clone(),
        inner.span,
        symbols,
        term_ids,
    );
    let fused_soac_term = match fused_soac_term {
        Some(t) => t,
        None => return (body, false),
    };

    // Rewrite the Let chain: remove producer, replace consumer with fused
    if let Some(fused_inner) = rewrite_let_chain(&inner, prod_sym, cons_sym, fused_soac_term, term_ids) {
        // Rebuild lambda wrapper
        let result = if params.is_empty() {
            fused_inner
        } else {
            let ret_ty = fused_inner.ty.clone();
            let mut lam_ty = ret_ty.clone();
            for (_, param_ty) in params.iter().rev() {
                lam_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), lam_ty]);
            }
            Term {
                id: term_ids.next_id(),
                ty: lam_ty,
                span: fused_inner.span,
                kind: TermKind::Lambda(Lambda {
                    params,
                    body: Box::new(fused_inner),
                    ret_ty,
                }),
            }
        };
        (result, true)
    } else {
        (body, false)
    }
}

// =============================================================================
// Semantic fusion — build fused SOAC from ArraySemantics
// =============================================================================

/// Build a fused SOAC Term from producer and consumer ArraySemantics.
/// ArraySemantics stores ArrayExpr directly, so types are preserved.
fn build_fused_from_semantics(
    producer: &ArraySemantics,
    consumer: &ArraySemantics,
    fusion_kind: &FusionKind,
    consumer_ty: Type<TypeName>,
    span: Span,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    // Fusion runs pre-defunctionalize, so SoacBody.captures is always
    // empty here — extract the inner Lambda for composition.
    let (prod_lam, input_exprs) = match producer {
        ArraySemantics::Elementwise { body, inputs } => (&body.lam, inputs.clone()),
        _ => return None,
    };

    match (fusion_kind, consumer) {
        (FusionKind::ComposeElementwise, ArraySemantics::Elementwise { body: cons_body, .. }) => {
            let composed =
                compose_lambdas(prod_lam.clone(), cons_body.lam.clone(), span, symbols, term_ids);
            Some(Term {
                id: term_ids.next_id(),
                ty: consumer_ty,
                span,
                kind: TermKind::Soac(SoacOp::Map {
                    lam: super::SoacBody {
                        lam: composed,
                        captures: vec![],
                    },
                    inputs: input_exprs,
                    consumes_input: false,
                }),
            })
        }

        (FusionKind::MapIntoReduce, ArraySemantics::Reduction { op, init, props, .. }) => {
            if op.lam.params.len() != 2 {
                return None;
            }
            let composed_op = compose_map_reduce(prod_lam.clone(), op.lam.clone(), span, symbols, term_ids);
            Some(Term {
                id: term_ids.next_id(),
                ty: consumer_ty,
                span,
                kind: TermKind::Soac(SoacOp::Redomap {
                    op: super::SoacBody {
                        lam: composed_op,
                        captures: vec![],
                    },
                    reduce_op: op.clone(),
                    ne: init.clone(),
                    inputs: input_exprs,
                    props: props.clone(),
                }),
            })
        }

        (FusionKind::MapIntoScan, ArraySemantics::PrefixScan { op, init, .. }) => {
            if input_exprs.len() != 1 {
                return None;
            }
            let composed_op = compose_map_reduce(prod_lam.clone(), op.lam.clone(), span, symbols, term_ids);
            Some(Term {
                id: term_ids.next_id(),
                ty: consumer_ty,
                span,
                kind: TermKind::Soac(SoacOp::Scan {
                    op: super::SoacBody {
                        lam: composed_op,
                        captures: vec![],
                    },
                    ne: init.clone(),
                    input: input_exprs[0].clone(),
                }),
            })
        }

        _ => None,
    }
}

// =============================================================================
// Let-chain rewriting
// =============================================================================

/// Rewrite the Let chain: remove the producer Let, replace the consumer
/// (Let-bound or tail expression) with the fused SOAC.
fn rewrite_let_chain(
    term: &Term,
    prod_sym: SymbolId,
    cons_sym: Option<SymbolId>,
    fused: Term,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    match &term.kind {
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            if *name == prod_sym {
                // Skip the producer Let, replace consumer in body
                return replace_consumer(body, cons_sym, fused, term_ids);
            }
            // Recurse
            let new_body = rewrite_let_chain(body, prod_sym, cons_sym, fused, term_ids)?;
            Some(Term {
                id: term_ids.next_id(),
                ty: term.ty.clone(),
                span: term.span,
                kind: TermKind::Let {
                    name: *name,
                    name_ty: name_ty.clone(),
                    rhs: rhs.clone(),
                    body: Box::new(new_body),
                },
            })
        }
        _ => None,
    }
}

/// Replace the consumer in the body with the fused SOAC.
/// If cons_sym is None, the body itself is the consumer (tail expression).
fn replace_consumer(
    body: &Term,
    cons_sym: Option<SymbolId>,
    fused: Term,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    match cons_sym {
        None => {
            // Tail expression — replace entirely, but keep the consumer's type
            Some(Term {
                ty: body.ty.clone(),
                ..fused
            })
        }
        Some(target) => {
            match &body.kind {
                TermKind::Let {
                    name,
                    name_ty,
                    rhs,
                    body: inner,
                } => {
                    if *name == target {
                        // Replace this Let's RHS with the fused SOAC
                        Some(Term {
                            id: term_ids.next_id(),
                            ty: body.ty.clone(),
                            span: body.span,
                            kind: TermKind::Let {
                                name: *name,
                                name_ty: name_ty.clone(),
                                rhs: Box::new(Term {
                                    ty: rhs.ty.clone(),
                                    ..fused
                                }),
                                body: inner.clone(),
                            },
                        })
                    } else {
                        let new_inner = replace_consumer(inner, cons_sym, fused, term_ids)?;
                        Some(Term {
                            id: term_ids.next_id(),
                            ty: body.ty.clone(),
                            span: body.span,
                            kind: TermKind::Let {
                                name: *name,
                                name_ty: name_ty.clone(),
                                rhs: rhs.clone(),
                                body: Box::new(new_inner),
                            },
                        })
                    }
                }
                _ => None,
            }
        }
    }
}

// =============================================================================
// Inline map fusion: map(f, map(g, a)) → map(f∘g, a)
// =============================================================================

/// Bottom-up pass: fuse nested Maps in SOAC inputs.
fn fuse_inline_maps(term: Term, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> Term {
    // Recurse into children first (bottom-up)
    let term = term.map_children(&mut |child| fuse_inline_maps(child, symbols, term_ids));

    // Check if this is a Map with any Map inputs
    if let TermKind::Soac(SoacOp::Map { .. }) = &term.kind {
        let TermKind::Soac(SoacOp::Map {
            lam,
            inputs,
            consumes_input,
        }) = term.kind
        else {
            unreachable!()
        };

        let has_fusible = inputs.iter().any(|input| {
            matches!(input, ArrayExpr::Ref(t) if matches!(t.kind, TermKind::Soac(SoacOp::Map { .. })))
        });

        if has_fusible {
            let fused = fuse_inline_map_inputs(lam, inputs, symbols, term_ids);
            return Term {
                kind: TermKind::Soac(fused),
                ..term
            };
        }

        return Term {
            kind: TermKind::Soac(SoacOp::Map {
                lam,
                inputs,
                consumes_input,
            }),
            ..term
        };
    }

    term
}

/// Fuse inline nested Maps from a Map's inputs.
fn fuse_inline_map_inputs(
    lam: super::SoacBody,
    inputs: Vec<ArrayExpr>,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> SoacOp {
    // Pre-defunc, captures are empty — operate on the inner Lambda.
    let lam = lam.lam;
    let mut new_params = Vec::new();
    let mut new_inputs = Vec::new();
    let mut body = *lam.body;
    let span = body.span;

    for (i, input) in inputs.into_iter().enumerate() {
        let is_inner_map = matches!(
            &input,
            ArrayExpr::Ref(t) if matches!(t.kind, TermKind::Soac(SoacOp::Map { .. }))
        );

        if is_inner_map {
            let inner_term = match input {
                ArrayExpr::Ref(t) => *t,
                _ => unreachable!(),
            };
            let (inner_sb, inner_inputs) = match inner_term.kind {
                TermKind::Soac(SoacOp::Map { lam, inputs, .. }) => (lam, inputs),
                _ => unreachable!(),
            };
            let inner_lam = inner_sb.lam;

            let fresh = symbols.alloc("_fused".to_string());
            let outer_param = lam.params[i].0;

            body = substitute_sym(body, outer_param, fresh, term_ids);

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

            new_params.extend(inner_lam.params);
            new_inputs.extend(inner_inputs);
        } else {
            new_params.push(lam.params[i].clone());
            new_inputs.push(input);
        }
    }

    SoacOp::Map {
        lam: super::SoacBody {
            lam: Lambda {
                params: new_params,
                body: Box::new(body),
                ret_ty: lam.ret_ty,
            },
            captures: vec![],
        },
        inputs: new_inputs,
        consumes_input: false,
    }
}

// =============================================================================
// Lambda composition
// =============================================================================

/// Compose two lambdas: f then g, producing g∘f.
/// f: A→B, g: B→C → composed: A→C
fn compose_lambdas(
    f: Lambda,
    g: Lambda,
    span: Span,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Lambda {
    let fresh_sym = symbols.alloc("_fused".to_string());
    let intermediate_ty = f.ret_ty.clone();

    let g_param = g.params[0].0;
    let g_body_substituted = substitute_sym(*g.body, g_param, fresh_sym, term_ids);

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
    }
}

/// Compose map lambda into reduce/scan operator.
/// map_lam: A→B, op: (Acc,B)→Acc → composed: (Acc,A)→Acc
fn compose_map_reduce(
    map_lam: Lambda,
    reduce_op: Lambda,
    span: Span,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Lambda {
    let fresh_sym = symbols.alloc("_fused".to_string());
    let intermediate_ty = map_lam.ret_ty.clone();

    let elem_param = reduce_op.params[1].0;
    let op_body_substituted = substitute_sym(*reduce_op.body, elem_param, fresh_sym, term_ids);

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

    // Combined params: (acc, x1, ..., xN) where x1..xN are the map lambda's params
    let mut params = vec![reduce_op.params[0].clone()];
    params.extend(map_lam.params.iter().cloned());

    Lambda {
        params,
        body: Box::new(composed_body),
        ret_ty: reduce_op.ret_ty,
    }
}

// =============================================================================
// Symbol substitution
// =============================================================================

/// Substitute all free occurrences of `old` with `Var(new)` in a term,
/// respecting shadowing by Let names, Lambda params, and Loop vars.
pub fn substitute_sym(term: Term, old: SymbolId, new: SymbolId, term_ids: &mut TermIdSource) -> Term {
    match term.kind {
        TermKind::Var(crate::tlc::VarRef::Symbol(s)) if s == old => Term {
            id: term_ids.next_id(),
            kind: TermKind::Var(crate::tlc::VarRef::Symbol(new)),
            ..term
        },

        TermKind::Var(crate::tlc::VarRef::Symbol(_))
        | TermKind::BinOp(_)
        | TermKind::UnOp(_)
        | TermKind::IntLit(_)
        | TermKind::FloatLit(_)
        | TermKind::BoolLit(_)
        | TermKind::Extern(_) => term,

        // Lambda: stop if param shadows old
        TermKind::Lambda(ref lam) if lam.params.iter().any(|(p, _)| *p == old) => term,

        // Let: substitute in rhs, stop in body if name shadows
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
                kind: TermKind::Let {
                    name,
                    name_ty,
                    rhs: Box::new(new_rhs),
                    body: Box::new(new_body),
                },
                ..term
            }
        }

        // Everything else: recurse via map_children
        _ => term.map_children(&mut |child| substitute_sym(child, old, new, term_ids)),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[path = "fusion_tests.rs"]
mod fusion_tests;
