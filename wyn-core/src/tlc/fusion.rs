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

use super::VarRef;
use crate::ast::{Span, TypeName};
use crate::{SymbolId, SymbolTable};
use polytype::Type;

use std::collections::HashMap;

use super::array_semantics::{ArraySemantics, FunctionSummary, FusionKind, can_fuse, summarize_program};
use super::producer_graph::{self, ProducerEdge};
use super::{
    ArrayExpr, Def, Lambda, Program, SoacDestination, SoacOp, Term, TermIdSource, TermKind,
    extract_lambda_params,
};

type Summaries = HashMap<SymbolId, FunctionSummary>;

/// Pass-local context threaded through every fusion-internal call.
/// Holds artifacts that are expensive to recompute per call site:
/// - `summaries`: borrowed from the per-outer-iteration analysis
/// - `sym_to_def`: rebuilt once per outer iteration (previously
///   `producer_graph::build_producer_graph` re-scanned the entire
///   symbol table on every call).
struct FusionContext<'a> {
    summaries: &'a Summaries,
    sym_to_def: HashMap<SymbolId, SymbolId>,
}

/// Build the `Var-symbol → def-symbol` lookup once. The fusion pass
/// allocates fresh let-binding / lambda symbols during a sweep, but
/// none of those name a def, so the map stays valid across the sweep.
pub(crate) fn build_sym_to_def(
    symbols: &SymbolTable,
    def_syms: &HashMap<String, SymbolId>,
) -> HashMap<SymbolId, SymbolId> {
    let mut sym_to_def: HashMap<SymbolId, SymbolId> = HashMap::new();
    for (sym, name) in symbols.iter() {
        if let Some(&def_sym) = def_syms.get(name) {
            sym_to_def.insert(*sym, def_sym);
        }
    }
    sym_to_def
}

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

        // Precompute the Var-symbol → def-symbol map once per outer
        // iteration. Each `build_producer_graph` call used to rebuild
        // this from a full symbol-table scan; with N defs containing
        // M let chains, that's N·M scans of the (growing) symbol table
        // per outer iteration. Hoisting it makes producer-graph
        // construction O(graph size) instead of O(symbol table size).
        let sym_to_def = build_sym_to_def(&symbols, &def_syms);
        let ctx = FusionContext {
            summaries: &summaries,
            sym_to_def,
        };

        let defs = program
            .defs
            .into_iter()
            .map(|def| {
                // Bottom-up: fuse children first, then try graph-driven fusion
                let new_body = fuse_term(def.body, &ctx, &mut symbols, &mut term_ids);
                let (new_body, did_fuse) = fuse_screma_groups(new_body, &mut symbols, &mut term_ids);
                if did_fuse {
                    changed = true;
                }
                let (new_body, did_fuse) =
                    fuse_map_into_screma_consumer(new_body, &mut symbols, &mut term_ids);
                if did_fuse {
                    changed = true;
                }
                let (new_body, did_fuse) = fuse_def_body(new_body, &ctx, &mut symbols, &mut term_ids);
                if did_fuse {
                    changed = true;
                }
                // Inline producer→consumer fusion for SOACs nested directly
                // as another SOAC's input (map→map, map→reduce, map→scan).
                let new_body = fuse_inline_soac_inputs(new_body, &mut symbols, &mut term_ids);
                Def {
                    body: new_body,
                    ..def
                }
            })
            .collect();

        program = Program {
            defs,
            symbols,
            def_syms: def_syms.clone(),
            ..program
        };
    }

    program
}

// =============================================================================
// Per-def fusion
// =============================================================================

#[derive(Clone)]
struct LetBinding {
    name: SymbolId,
    name_ty: Type<TypeName>,
    rhs: Term,
    span: Span,
}

struct ScremaRewrite {
    insert_at: usize,
    skip_indices: Vec<usize>,
    tuple_sym: SymbolId,
    tuple_ty: Type<TypeName>,
    fused_binding: LetBinding,
    projection_fields: HashMap<usize, usize>,
}

fn fuse_screma_groups(body: Term, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> (Term, bool) {
    let (params, inner) = extract_lambda_params(&body);
    let (bindings, tail) = flatten_let_chain(inner);

    let Some(group) = find_screma_group(&bindings, &tail) else {
        return (body, false);
    };

    let span = group.span;
    let mut result_fields = Vec::with_capacity(group.maps.len() + group.accumulators.len());
    result_fields.extend(group.maps.iter().map(|consumer| bindings[consumer.binding_idx].name_ty.clone()));
    result_fields
        .extend(group.accumulators.iter().map(|consumer| bindings[consumer.binding_idx].name_ty.clone()));
    let map_lams: Vec<super::SoacBody> = group
        .maps
        .iter()
        .map(|consumer| {
            if let Some(producer_lam) = &group.producer_lam {
                super::SoacBody {
                    lam: compose_lambdas(
                        producer_lam.lam.clone(),
                        consumer.lam.lam.clone(),
                        span,
                        symbols,
                        term_ids,
                    ),
                    captures: vec![],
                }
            } else {
                consumer.lam.clone()
            }
        })
        .collect();
    let accumulators: Vec<super::ScremaAccumulatorSpec> = group
        .accumulators
        .iter()
        .map(|consumer| {
            let producer_lam =
                group.producer_lam.as_ref().expect("accumulator Screma fusion requires a producer lambda");
            super::ScremaAccumulatorSpec {
                kind: consumer.accumulator,
                step_lam: super::SoacBody {
                    lam: compose_map_reduce(
                        producer_lam.lam.clone(),
                        consumer.op.lam.clone(),
                        span,
                        symbols,
                        term_ids,
                    ),
                    captures: vec![],
                },
                reduce_op: consumer.reduce_op.clone(),
                ne: consumer.ne.clone(),
            }
        })
        .collect();
    let mut projection_fields: HashMap<usize, usize> = HashMap::new();
    projection_fields.extend(
        group.maps.iter().enumerate().map(|(field_idx, consumer)| (consumer.binding_idx, field_idx)),
    );
    projection_fields.extend(
        group
            .accumulators
            .iter()
            .enumerate()
            .map(|(acc_idx, consumer)| (consumer.binding_idx, group.maps.len() + acc_idx)),
    );
    let rewrite = make_screma_rewrite(
        group.insert_at,
        group.skip_indices,
        group.symbol_name,
        result_fields,
        map_lams,
        accumulators,
        group.inputs,
        span,
        projection_fields,
        symbols,
        term_ids,
    );
    let new_bindings = apply_screma_rewrite(&bindings, rewrite, term_ids);

    let fused_inner = rebuild_let_chain(new_bindings, tail, term_ids);
    (rebuild_lambda_params(params, fused_inner, term_ids), true)
}

struct ScremaProducer {
    lam: super::SoacBody,
    inputs: Vec<ArrayExpr>,
}

struct ScremaReduceConsumer {
    binding_idx: usize,
    op: super::SoacBody,
    reduce_op: super::SoacBody,
    ne: Box<Term>,
    accumulator: super::ScremaAccumulator,
}

struct ScremaMapGroupConsumer {
    binding_idx: usize,
    lam: super::SoacBody,
}

struct ScremaGroup {
    insert_at: usize,
    skip_indices: Vec<usize>,
    symbol_name: &'static str,
    span: Span,
    inputs: Vec<ArrayExpr>,
    producer_lam: Option<super::SoacBody>,
    maps: Vec<ScremaMapGroupConsumer>,
    accumulators: Vec<ScremaReduceConsumer>,
}

fn find_screma_group(bindings: &[LetBinding], tail: &Term) -> Option<ScremaGroup> {
    find_producer_screma_group(bindings, tail).or_else(|| find_direct_map_screma_group(bindings))
}

fn find_producer_screma_group(bindings: &[LetBinding], tail: &Term) -> Option<ScremaGroup> {
    for idx in 0..bindings.len() {
        let Some(producer) = screma_producer_map(&bindings[idx].rhs) else {
            continue;
        };
        let producer_sym = bindings[idx].name;
        let mut maps = Vec::new();
        let mut accumulators = Vec::new();
        let mut selected_consumer_names = Vec::new();
        let mut cursor = idx + 1;
        let mut blocked = false;
        while cursor < bindings.len() {
            if let Some(lam) = screma_consumer_map(&bindings[cursor].rhs, producer_sym) {
                if term_mentions_any(&bindings[cursor].rhs, &selected_consumer_names) {
                    blocked = true;
                    break;
                }
                selected_consumer_names.push(bindings[cursor].name);
                maps.push(ScremaMapGroupConsumer {
                    binding_idx: cursor,
                    lam,
                });
                cursor += 1;
                continue;
            }
            if let Some(mut acc) = screma_consumer_accumulator(&bindings[cursor].rhs, producer_sym) {
                if term_mentions_any(&bindings[cursor].rhs, &selected_consumer_names) {
                    blocked = true;
                    break;
                }
                selected_consumer_names.push(bindings[cursor].name);
                acc.binding_idx = cursor;
                accumulators.push(acc);
                cursor += 1;
                continue;
            }
            if term_mentions_any(&bindings[cursor].rhs, &[producer_sym]) {
                blocked = true;
                break;
            }
            cursor += 1;
        }
        if !blocked
            && !maps.is_empty()
            && !accumulators.is_empty()
            && maps.len() + accumulators.len() >= 2
            && !term_mentions_any(tail, &[producer_sym])
        {
            return Some(ScremaGroup {
                insert_at: idx,
                skip_indices: vec![idx],
                symbol_name: "_screma",
                span: bindings[idx].span,
                inputs: producer.inputs,
                producer_lam: Some(producer.lam),
                maps,
                accumulators,
            });
        }
    }
    None
}

fn find_direct_map_screma_group(bindings: &[LetBinding]) -> Option<ScremaGroup> {
    let mut start = 0;
    while start < bindings.len() {
        let Some(first) = direct_screma_map(&bindings[start].rhs) else {
            start += 1;
            continue;
        };

        let Some(first_input_syms) = array_ref_symbols(&first.inputs) else {
            start += 1;
            continue;
        };
        let first_param_tys = param_types(&first.lam.lam.params);
        let mut maps = vec![ScremaMapGroupConsumer {
            binding_idx: start,
            lam: first.lam,
        }];
        let mut end = start + 1;
        while end < bindings.len() {
            let Some(next) = direct_screma_map(&bindings[end].rhs) else {
                break;
            };
            let Some(next_input_syms) = array_ref_symbols(&next.inputs) else {
                break;
            };
            if next_input_syms != first_input_syms || param_types(&next.lam.lam.params) != first_param_tys {
                break;
            }
            maps.push(ScremaMapGroupConsumer {
                binding_idx: end,
                lam: next.lam,
            });
            end += 1;
        }

        if maps.len() >= 2 {
            let candidate_names: Vec<SymbolId> = bindings[start..end].iter().map(|b| b.name).collect();
            if !bindings[start..end].iter().any(|binding| term_mentions_any(&binding.rhs, &candidate_names))
            {
                return Some(ScremaGroup {
                    insert_at: start,
                    skip_indices: vec![],
                    symbol_name: "_horizontal_map",
                    span: bindings[start].span,
                    inputs: first.inputs,
                    producer_lam: None,
                    maps,
                    accumulators: vec![],
                });
            }
        }
        start = end.max(start + 1);
    }
    None
}

fn screma_producer_map(term: &Term) -> Option<ScremaProducer> {
    map_screma_producer(term, MapProducerPolicy::Producer)
}

fn direct_screma_map(term: &Term) -> Option<ScremaProducer> {
    map_screma_producer(term, MapProducerPolicy::Direct)
}

enum MapProducerPolicy {
    Producer,
    Direct,
}

fn map_screma_producer(term: &Term, policy: MapProducerPolicy) -> Option<ScremaProducer> {
    let TermKind::Soac(SoacOp::Map {
        lam,
        inputs,
        destination: _,
    }) = &term.kind
    else {
        return None;
    };
    if !lam.captures.is_empty() {
        return None;
    }
    match policy {
        MapProducerPolicy::Producer if lam.lam.params.is_empty() => return None,
        MapProducerPolicy::Direct if inputs.is_empty() || inputs.len() != lam.lam.params.len() => {
            return None;
        }
        _ => {}
    }
    Some(ScremaProducer {
        lam: lam.clone(),
        inputs: inputs.clone(),
    })
}

fn screma_consumer_map(term: &Term, input_sym: SymbolId) -> Option<super::SoacBody> {
    let TermKind::Soac(SoacOp::Map { lam, inputs, .. }) = &term.kind else {
        return None;
    };
    if inputs.len() != 1 || inputs[0].as_named_ref() != Some(input_sym) || lam.lam.params.len() != 1 {
        return None;
    }
    Some(lam.clone())
}

fn screma_consumer_accumulator(term: &Term, input_sym: SymbolId) -> Option<ScremaReduceConsumer> {
    match &term.kind {
        TermKind::Soac(SoacOp::Reduce { op, ne, input }) => {
            if input.as_named_ref() != Some(input_sym) || op.lam.params.len() != 2 {
                return None;
            }
            Some(ScremaReduceConsumer {
                binding_idx: usize::MAX,
                op: op.clone(),
                reduce_op: op.clone(),
                ne: ne.clone(),
                accumulator: super::ScremaAccumulator::Reduce,
            })
        }
        TermKind::Soac(SoacOp::Scan {
            op,
            reduce_op,
            ne,
            input,
            ..
        }) => {
            if input.as_named_ref() != Some(input_sym) || op.lam.params.len() != 2 {
                return None;
            }
            Some(ScremaReduceConsumer {
                binding_idx: usize::MAX,
                op: op.clone(),
                reduce_op: reduce_op.clone(),
                ne: ne.clone(),
                accumulator: super::ScremaAccumulator::Scan,
            })
        }
        _ => None,
    }
}

fn fuse_map_into_screma_consumer(
    body: Term,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> (Term, bool) {
    let (params, inner) = extract_lambda_params(&body);
    let (bindings, tail) = flatten_let_chain(inner);

    let Some((producer_idx, consumer_idx, fused_rhs)) =
        find_map_into_screma_consumer(&bindings, &tail, symbols, term_ids)
    else {
        return (body, false);
    };

    let mut new_bindings = Vec::with_capacity(bindings.len() - 1);
    for (idx, binding) in bindings.into_iter().enumerate() {
        if idx == producer_idx {
            continue;
        }
        if idx == consumer_idx {
            new_bindings.push(LetBinding {
                rhs: fused_rhs.clone(),
                ..binding
            });
        } else {
            new_bindings.push(binding);
        }
    }

    let fused_inner = rebuild_let_chain(new_bindings, tail, term_ids);
    (rebuild_lambda_params(params, fused_inner, term_ids), true)
}

fn find_map_into_screma_consumer(
    bindings: &[LetBinding],
    tail: &Term,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<(usize, usize, Term)> {
    for producer_idx in 0..bindings.len() {
        let Some(producer) = screma_producer_map(&bindings[producer_idx].rhs) else {
            continue;
        };
        let producer_sym = bindings[producer_idx].name;
        for consumer_idx in producer_idx + 1..bindings.len() {
            let Some(fused_rhs) = map_into_screma_rhs(
                &producer,
                producer_sym,
                &bindings[consumer_idx].rhs,
                bindings[consumer_idx].span,
                symbols,
                term_ids,
            ) else {
                if term_mentions_any(&bindings[consumer_idx].rhs, &[producer_sym]) {
                    break;
                }
                continue;
            };
            let other_rhs_uses = bindings.iter().enumerate().any(|(idx, binding)| {
                idx != producer_idx
                    && idx != consumer_idx
                    && term_mentions_any(&binding.rhs, &[producer_sym])
            });
            if !other_rhs_uses && !term_mentions_any(tail, &[producer_sym]) {
                return Some((producer_idx, consumer_idx, fused_rhs));
            }
        }
    }
    None
}

fn map_into_screma_rhs(
    producer: &ScremaProducer,
    producer_sym: SymbolId,
    consumer: &Term,
    span: Span,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    let TermKind::Soac(SoacOp::Screma {
        map_lams,
        accumulators,
        inputs,
    }) = &consumer.kind
    else {
        return None;
    };
    if inputs.len() != 1 || inputs[0].as_named_ref() != Some(producer_sym) {
        return None;
    }
    if map_lams.iter().any(|map_lam| !map_lam.captures.is_empty())
        || accumulators.iter().any(|acc| !acc.step_lam.captures.is_empty())
    {
        return None;
    }
    if accumulators.iter().any(|acc| acc.step_lam.lam.params.len() != 2) {
        return None;
    }

    let map_lams = map_lams
        .iter()
        .map(|map_lam| super::SoacBody {
            lam: compose_lambdas(
                producer.lam.lam.clone(),
                map_lam.lam.clone(),
                span,
                symbols,
                term_ids,
            ),
            captures: vec![],
        })
        .collect();
    let accumulators = accumulators
        .iter()
        .map(|acc| super::ScremaAccumulatorSpec {
            kind: acc.kind,
            step_lam: super::SoacBody {
                lam: compose_map_reduce(
                    producer.lam.lam.clone(),
                    acc.step_lam.lam.clone(),
                    span,
                    symbols,
                    term_ids,
                ),
                captures: vec![],
            },
            reduce_op: acc.reduce_op.clone(),
            ne: acc.ne.clone(),
        })
        .collect();

    Some(Term {
        id: term_ids.next_id(),
        ty: consumer.ty.clone(),
        span: consumer.span,
        kind: TermKind::Soac(SoacOp::Screma {
            map_lams,
            accumulators,
            inputs: producer.inputs.clone(),
        }),
    })
}

fn make_screma_term(
    result_fields: Vec<Type<TypeName>>,
    map_lams: Vec<super::SoacBody>,
    accumulators: Vec<super::ScremaAccumulatorSpec>,
    inputs: Vec<ArrayExpr>,
    span: Span,
    term_ids: &mut TermIdSource,
) -> Term {
    let ty = Type::Constructed(TypeName::Tuple(result_fields.len()), result_fields);
    Term {
        id: term_ids.next_id(),
        ty,
        span,
        kind: TermKind::Soac(SoacOp::Screma {
            map_lams,
            accumulators,
            inputs,
        }),
    }
}

fn make_screma_rewrite(
    insert_at: usize,
    skip_indices: Vec<usize>,
    symbol_name: &str,
    result_fields: Vec<Type<TypeName>>,
    map_lams: Vec<super::SoacBody>,
    accumulators: Vec<super::ScremaAccumulatorSpec>,
    inputs: Vec<ArrayExpr>,
    span: Span,
    projection_fields: HashMap<usize, usize>,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> ScremaRewrite {
    let tuple_sym = symbols.alloc(symbol_name.to_string());
    let rhs = make_screma_term(result_fields, map_lams, accumulators, inputs, span, term_ids);
    let tuple_ty = rhs.ty.clone();
    let fused_binding = LetBinding {
        name: tuple_sym,
        name_ty: tuple_ty.clone(),
        rhs,
        span,
    };
    ScremaRewrite {
        insert_at,
        skip_indices,
        tuple_sym,
        tuple_ty,
        fused_binding,
        projection_fields,
    }
}

fn tuple_projection_binding(
    original: &LetBinding,
    tuple_sym: SymbolId,
    tuple_ty: &Type<TypeName>,
    proj_idx: usize,
    term_ids: &mut TermIdSource,
) -> LetBinding {
    let tuple_ref = Term {
        id: term_ids.next_id(),
        ty: tuple_ty.clone(),
        span: original.span,
        kind: TermKind::Var(VarRef::Symbol(tuple_sym)),
    };
    LetBinding {
        name: original.name,
        name_ty: original.name_ty.clone(),
        rhs: Term {
            id: term_ids.next_id(),
            ty: original.name_ty.clone(),
            span: original.span,
            kind: TermKind::TupleProj {
                tuple: Box::new(tuple_ref),
                idx: proj_idx,
            },
        },
        span: original.span,
    }
}

fn apply_screma_rewrite(
    bindings: &[LetBinding],
    rewrite: ScremaRewrite,
    term_ids: &mut TermIdSource,
) -> Vec<LetBinding> {
    let mut new_bindings = Vec::with_capacity(bindings.len() + 1 - rewrite.skip_indices.len());
    for (idx, binding) in bindings.iter().enumerate() {
        if idx == rewrite.insert_at {
            new_bindings.push(rewrite.fused_binding.clone());
        }
        if rewrite.skip_indices.contains(&idx) {
            continue;
        }
        if let Some(&proj_idx) = rewrite.projection_fields.get(&idx) {
            new_bindings.push(tuple_projection_binding(
                binding,
                rewrite.tuple_sym,
                &rewrite.tuple_ty,
                proj_idx,
                term_ids,
            ));
        } else {
            new_bindings.push(binding.clone());
        }
    }
    if rewrite.insert_at >= bindings.len() {
        new_bindings.push(rewrite.fused_binding);
    }
    new_bindings
}

fn flatten_let_chain(mut term: Term) -> (Vec<LetBinding>, Term) {
    let mut bindings = Vec::new();
    loop {
        match term.kind {
            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                bindings.push(LetBinding {
                    name,
                    name_ty,
                    rhs: *rhs,
                    span: term.span,
                });
                term = *body;
            }
            _ => return (bindings, term),
        }
    }
}

fn rebuild_let_chain(bindings: Vec<LetBinding>, mut body: Term, term_ids: &mut TermIdSource) -> Term {
    for binding in bindings.into_iter().rev() {
        let ty = body.ty.clone();
        body = Term {
            id: term_ids.next_id(),
            ty,
            span: binding.span,
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

fn rebuild_lambda_params(
    params: Vec<(SymbolId, Type<TypeName>)>,
    body: Term,
    term_ids: &mut TermIdSource,
) -> Term {
    if params.is_empty() {
        return body;
    }

    let ret_ty = body.ty.clone();
    let mut lam_ty = ret_ty.clone();
    for (_, param_ty) in params.iter().rev() {
        lam_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), lam_ty]);
    }
    Term {
        id: term_ids.next_id(),
        ty: lam_ty,
        span: body.span,
        kind: TermKind::Lambda(Lambda {
            params,
            body: Box::new(body),
            ret_ty,
        }),
    }
}

fn param_types(params: &[(SymbolId, Type<TypeName>)]) -> Vec<Type<TypeName>> {
    params.iter().map(|(_, ty)| ty.clone()).collect()
}

fn array_ref_symbols(inputs: &[ArrayExpr]) -> Option<Vec<SymbolId>> {
    inputs.iter().map(ArrayExpr::as_named_ref).collect()
}

fn term_mentions_any(term: &Term, syms: &[SymbolId]) -> bool {
    match &term.kind {
        TermKind::Var(VarRef::Symbol(sym)) => syms.contains(sym),
        _ => {
            let mut found = false;
            term.for_each_child(&mut |child| {
                if !found && term_mentions_any(child, syms) {
                    found = true;
                }
            });
            found
        }
    }
}

/// Bottom-up: recurse into children, applying graph-driven fusion at each
/// sub-expression that contains a Let chain with SOAC producers/consumers.
fn fuse_term(
    term: Term,
    ctx: &FusionContext<'_>,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Term {
    let term = term.map_children(&mut |child| fuse_term(child, ctx, symbols, term_ids));

    if matches!(term.kind, TermKind::Let { .. }) {
        let (fused, _) = fuse_def_body(term, ctx, symbols, term_ids);
        return fused;
    }

    term
}

/// Fuse within a body term. Returns the new body and whether any fusion happened.
fn fuse_def_body(
    body: Term,
    ctx: &FusionContext<'_>,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> (Term, bool) {
    let (params, inner) = extract_lambda_params(&body);

    let graph = producer_graph::build_producer_graph(&inner, ctx.summaries, &ctx.sym_to_def);

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
        edge.input_index,
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
    input_index: usize,
    consumer_ty: Type<TypeName>,
    span: Span,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    // Filter → Reduction: the producer is a `Filter` (not `Elementwise`), so
    // build the masked redomap here, before the Elementwise extraction below.
    // `reduce(op, ne, filter(p, xs))` ≡ `redomap(op∘mask, op, ne, xs)` with
    // `mask = λx. if p(x) then x else ne` — the dropped elements fold in as
    // `op(acc, ne) = acc` (ne is op's neutral element by reduce's contract).
    if let (
        FusionKind::FilterIntoReduce,
        ArraySemantics::Filter { input, pred },
        ArraySemantics::Reduction { op, init, .. },
    ) = (fusion_kind, producer, consumer)
    {
        if op.lam.params.len() != 2 || pred.lam.params.len() != 1 {
            return None;
        }
        let x_sym = pred.lam.params[0].0;
        let elem_ty = pred.lam.params[0].1.clone();
        // mask = λx. if p(x) then x else ne
        let mask_lam = Lambda {
            params: vec![(x_sym, elem_ty.clone())],
            body: Box::new(Term {
                id: term_ids.next_id(),
                ty: elem_ty.clone(),
                span,
                kind: TermKind::If {
                    cond: pred.lam.body.clone(),
                    then_branch: Box::new(Term {
                        id: term_ids.next_id(),
                        ty: elem_ty.clone(),
                        span,
                        kind: TermKind::Var(VarRef::Symbol(x_sym)),
                    }),
                    else_branch: init.clone(),
                },
            }),
            ret_ty: elem_ty,
        };
        // op∘mask: (acc, x) -> op(acc, mask(x)); the pure phase-2 combiner is
        // the original `op`.
        let composed_op = compose_map_reduce(mask_lam, op.lam.clone(), span, symbols, term_ids);
        return Some(Term {
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
                inputs: vec![input.clone()],
            }),
        });
    }

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
                    destination: SoacDestination::Fresh,
                }),
            })
        }

        (FusionKind::MapIntoReduce, ArraySemantics::Reduction { op, init, .. }) => {
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
                    // The pure combiner is the original scan op (without the
                    // producer `f`); phase 2 combines already-transformed block
                    // sums, so it must not re-apply `f`.
                    reduce_op: op.clone(),
                    ne: init.clone(),
                    input: input_exprs[0].clone(),
                    // Fusion runs before apply_ownership; ownership pass
                    // will (re-)decide destination on this fused Scan.
                    destination: SoacDestination::Fresh,
                }),
            })
        }

        (
            FusionKind::MapIntoScatter,
            ArraySemantics::ScatterOp {
                dest,
                lam: env,
                inputs: scatter_inputs,
            },
        ) => {
            // Compose the producer `f` into the scatter envelope at the fused
            // slot and splice the producer's inputs in place of the
            // materialized array (Futhark thesis §7.3.1:
            // `h = λx y → let y = f x in g y`).
            let composed = compose_map_into_envelope(
                prod_lam.clone(),
                env.lam.clone(),
                input_index,
                span,
                symbols,
                term_ids,
            );
            let mut new_inputs = scatter_inputs.clone();
            new_inputs.splice(input_index..=input_index, input_exprs.iter().cloned());
            // Horizontal-fusion dedup: when the index and value producers shared
            // a base (the particles shape), the splice leaves two slots reading
            // the same array. Collapse them to one read — the operational form of
            // Futhark's "compute the shared input once".
            let (composed, new_inputs) = dedup_envelope_inputs(composed, new_inputs, term_ids);
            Some(Term {
                id: term_ids.next_id(),
                ty: consumer_ty,
                span,
                kind: TermKind::Soac(SoacOp::Scatter {
                    dest: dest.clone(),
                    lam: super::SoacBody {
                        lam: composed,
                        captures: vec![],
                    },
                    inputs: new_inputs,
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
/// Fuse an inline SOAC producer that appears directly as another SOAC's
/// input — the inline counterpart of the graph-driven let-chain fusion.
/// Covers the same producer/consumer pairs as `build_fused_from_semantics`:
/// `map → map` (compose into one Map), `map → reduce` (Redomap), and
/// `map → scan` (fused Scan whose operator includes the map transform).
///
/// Inline nested SOACs (`scan(op, ne, map(g, xs))`) never get a let binding
/// for the summary-based path to recognize, so without this they reach EGIR
/// as a separate producer loop building a runtime-sized in-register array —
/// which the SPIR-V backend cannot represent (`Composite variant unsized
/// arrays not supported`).
fn fuse_inline_soac_inputs(term: Term, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> Term {
    // Recurse into children first (bottom-up).
    let term = term.map_children(&mut |child| fuse_inline_soac_inputs(child, symbols, term_ids));
    let span = term.span;

    match term.kind {
        // map(f, map(g, xs)) → map(f∘g, xs)
        TermKind::Soac(SoacOp::Map {
            lam,
            inputs,
            destination,
        }) => {
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
            Term {
                kind: TermKind::Soac(SoacOp::Map {
                    lam,
                    inputs,
                    destination,
                }),
                ..term
            }
        }

        // reduce(op, ne, map(g, xs)) → redomap(op∘g, op, ne, xs)
        TermKind::Soac(SoacOp::Reduce { op, ne, input }) => {
            if let Some(fused) = fuse_inline_accumulator_input(
                InlineAccumulatorKind::Reduce,
                op.clone(),
                op.clone(),
                ne.clone(),
                input.clone(),
                span,
                symbols,
                term_ids,
            ) {
                return Term {
                    kind: TermKind::Soac(fused),
                    ..term
                };
            }
            Term {
                kind: TermKind::Soac(SoacOp::Reduce { op, ne, input }),
                ..term
            }
        }

        // scan(op, ne, map(g, xs)) → scan(op∘g, ne, xs)   (single map input only)
        TermKind::Soac(SoacOp::Scan {
            op,
            reduce_op,
            ne,
            input,
            destination,
        }) => {
            if let Some(fused) = fuse_inline_accumulator_input(
                InlineAccumulatorKind::Scan,
                op.clone(),
                reduce_op.clone(),
                ne.clone(),
                input.clone(),
                span,
                symbols,
                term_ids,
            ) {
                return Term {
                    kind: TermKind::Soac(fused),
                    ..term
                };
            }
            Term {
                kind: TermKind::Soac(SoacOp::Scan {
                    op,
                    reduce_op,
                    ne,
                    input,
                    destination,
                }),
                ..term
            }
        }

        _ => term,
    }
}

enum InlineAccumulatorKind {
    Reduce,
    Scan,
}

fn fuse_inline_accumulator_input(
    kind: InlineAccumulatorKind,
    op: super::SoacBody,
    reduce_op: super::SoacBody,
    ne: Box<Term>,
    input: ArrayExpr,
    span: Span,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<SoacOp> {
    if op.lam.params.len() != 2 {
        return None;
    }
    let (map_sb, map_inputs) = inline_map_producer(&input)?;
    let composed_op = compose_map_reduce(map_sb.lam, op.lam, span, symbols, term_ids);
    let op = super::SoacBody {
        lam: composed_op,
        captures: vec![],
    };
    match kind {
        InlineAccumulatorKind::Reduce => Some(SoacOp::Redomap {
            op,
            reduce_op,
            ne,
            inputs: map_inputs,
        }),
        InlineAccumulatorKind::Scan => {
            let mut map_inputs = map_inputs.into_iter();
            let input = map_inputs.next()?;
            if map_inputs.next().is_some() {
                return None;
            }
            Some(SoacOp::Scan {
                op,
                reduce_op,
                ne,
                input,
                destination: SoacDestination::Fresh,
            })
        }
    }
}

/// If `input` is an inline `map(...)` producer (`ArrayExpr::Ref` wrapping a
/// `Map` SOAC), return its lambda body + parallel inputs; else `None`.
fn inline_map_producer(input: &ArrayExpr) -> Option<(super::SoacBody, Vec<ArrayExpr>)> {
    if let ArrayExpr::Ref(t) = input {
        if let TermKind::Soac(SoacOp::Map { lam, inputs, .. }) = &t.kind {
            return Some((lam.clone(), inputs.clone()));
        }
    }
    None
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
        destination: SoacDestination::Fresh,
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

/// Compose a map producer into a scatter envelope at one input slot.
/// `producer: (q…) → B` feeds envelope param `slot` (type `B`). The result
/// takes the producer's params in that slot's place and binds the original
/// slot param to the producer's body ahead of the envelope body:
/// `λ(…, q…, …) → let p_slot = producer_body in envelope_body`.
fn compose_map_into_envelope(
    producer: Lambda,
    envelope: Lambda,
    slot: usize,
    span: Span,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Lambda {
    let fresh_sym = symbols.alloc("_fused".to_string());
    let slot_param = envelope.params[slot].0;
    let slot_ty = envelope.params[slot].1.clone();
    let env_body_substituted = substitute_sym(*envelope.body, slot_param, fresh_sym, term_ids);

    let composed_body = Term {
        id: term_ids.next_id(),
        ty: envelope.ret_ty.clone(),
        span,
        kind: TermKind::Let {
            name: fresh_sym,
            name_ty: slot_ty,
            rhs: producer.body,
            body: Box::new(env_body_substituted),
        },
    };

    // Envelope params with the fused slot replaced by the producer's params.
    let mut params = envelope.params.clone();
    params.splice(slot..=slot, producer.params.iter().cloned());

    Lambda {
        params,
        body: Box::new(composed_body),
        ret_ty: envelope.ret_ty,
    }
}

/// Collapse envelope input slots that read the same named array into one.
/// `inputs[i]` lines up with `lam.params[i]`; when a later slot names an array
/// an earlier kept slot already reads, drop it and rewrite its param to the
/// kept one. This is horizontal fusion specialised to one consumer's slots:
/// the shared base is read once rather than per channel.
fn dedup_envelope_inputs(
    lam: Lambda,
    inputs: Vec<ArrayExpr>,
    term_ids: &mut TermIdSource,
) -> (Lambda, Vec<ArrayExpr>) {
    let mut kept_inputs: Vec<ArrayExpr> = Vec::new();
    let mut kept_params: Vec<(SymbolId, Type<TypeName>)> = Vec::new();
    let mut body = *lam.body;

    for (ae, param) in inputs.into_iter().zip(lam.params.into_iter()) {
        if let Some(sym) = ae.as_named_ref() {
            if let Some(pos) = kept_inputs.iter().position(|k| k.as_named_ref() == Some(sym)) {
                body = substitute_sym(body, param.0, kept_params[pos].0, term_ids);
                continue;
            }
        }
        kept_inputs.push(ae);
        kept_params.push(param);
    }

    (
        Lambda {
            params: kept_params,
            body: Box::new(body),
            ret_ty: lam.ret_ty,
        },
        kept_inputs,
    )
}

// =============================================================================
// Symbol substitution
// =============================================================================

/// Substitute all free occurrences of `old` with `Var(new)` in a term,
/// respecting shadowing by Let names, Lambda params, and Loop vars.
pub fn substitute_sym(term: Term, old: SymbolId, new: SymbolId, term_ids: &mut TermIdSource) -> Term {
    match term.kind {
        TermKind::Var(VarRef::Symbol(s)) if s == old => Term {
            id: term_ids.next_id(),
            kind: TermKind::Var(VarRef::Symbol(new)),
            ..term
        },

        TermKind::Var(VarRef::Symbol(_))
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
