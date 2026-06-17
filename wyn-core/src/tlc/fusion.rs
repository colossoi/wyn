//! Graph-driven SOAC fusion pass for TLC.
//!
//! Fuses consecutive SOAC operations to eliminate intermediate arrays.
//! Classifies every use of a let-bound producer before planning a rewrite,
//! then lowers each fusion group through one rewrite path.
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

use super::array_semantics::{
    ArraySemantics, FunctionSummary, FusionKind, ResultSemantics, can_fuse, classify_term,
    summarize_program,
};
use super::{
    ArrayExpr, Def, Lambda, Program, SoacDestination, SoacOp, Term, TermId, TermIdSource, TermKind,
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
/// 3. Classifies producer uses, plans fusion groups, and rewrites them
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
                // Bottom-up: fuse children first, then try one classified-use
                // rewrite for this definition. The outer fixpoint reruns the
                // analysis after each rewrite.
                let (new_body, did_child_fuse) = fuse_term(def.body, &ctx, &mut symbols, &mut term_ids);
                if did_child_fuse {
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

#[derive(Clone)]
struct ProducerInfo {
    binding_idx: usize,
    symbol: SymbolId,
    semantics: ArraySemantics,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum UseOwner {
    Binding(usize),
    Tail,
}

#[derive(Clone)]
struct UseSite {
    owner: UseOwner,
    kind: UseKind,
}

#[derive(Clone)]
enum UseKind {
    SoacInput {
        term_id: TermId,
        input_index: usize,
        semantics: ArraySemantics,
        ty: Type<TypeName>,
        span: Span,
    },
    ScremaInput,
    Length {
        term_id: TermId,
        ty: Type<TypeName>,
        span: Span,
    },
    Escape,
}

#[derive(Clone)]
struct ProjectionTemplate {
    tuple_sym: SymbolId,
    tuple_ty: Type<TypeName>,
    field_idx: usize,
    ty: Type<TypeName>,
    span: Span,
}

struct PlannedScremaRewrite {
    rewrite: ScremaRewrite,
    term_replacements: HashMap<TermId, ProjectionTemplate>,
    tail_projection: Option<ProjectionTemplate>,
}

enum FusionPlan {
    ReplaceConsumer {
        producer_idx: usize,
        consumer_idx: Option<usize>,
        fused_rhs: Term,
    },
    Screma(PlannedScremaRewrite),
}

struct ScremaRewrite {
    insert_at: usize,
    skip_indices: Vec<usize>,
    tuple_sym: SymbolId,
    tuple_ty: Type<TypeName>,
    fused_binding: LetBinding,
    projection_fields: HashMap<usize, usize>,
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
    maps: Vec<ScremaMapGroupConsumer>,
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
                    maps,
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

fn find_fusion_plan(
    bindings: &[LetBinding],
    tail: &Term,
    ctx: &FusionContext<'_>,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<FusionPlan> {
    if let Some(plan) = find_direct_horizontal_map_plan(bindings, symbols, term_ids) {
        return Some(plan);
    }

    let producers = producer_infos(bindings, ctx, term_ids);
    for producer in &producers {
        let uses = classify_uses_for_producer(producer, bindings, tail, ctx, term_ids);
        if uses.iter().any(|u| matches!(u.kind, UseKind::Escape)) || uses.is_empty() {
            continue;
        }
        if let Some(plan) = find_filter_plan(producer, &uses, bindings, tail, symbols, term_ids) {
            return Some(plan);
        }
        if let Some(plan) = find_map_group_plan(producer, &uses, bindings, tail, symbols, term_ids) {
            return Some(plan);
        }
        if let Some(plan) = find_pairwise_plan(producer, &uses, bindings, tail, symbols, term_ids) {
            return Some(plan);
        }
    }
    None
}

fn producer_infos(
    bindings: &[LetBinding],
    ctx: &FusionContext<'_>,
    term_ids: &mut TermIdSource,
) -> Vec<ProducerInfo> {
    bindings
        .iter()
        .enumerate()
        .filter_map(|(binding_idx, binding)| {
            let semantics = classify_with_context(&binding.rhs, ctx, term_ids);
            if matches!(semantics, ArraySemantics::Opaque) {
                None
            } else {
                Some(ProducerInfo {
                    binding_idx,
                    symbol: binding.name,
                    semantics,
                })
            }
        })
        .collect()
}

fn classify_uses_for_producer(
    producer: &ProducerInfo,
    bindings: &[LetBinding],
    tail: &Term,
    ctx: &FusionContext<'_>,
    term_ids: &mut TermIdSource,
) -> Vec<UseSite> {
    let mut uses = Vec::new();
    for (idx, binding) in bindings.iter().enumerate().skip(producer.binding_idx + 1) {
        uses.extend(classify_uses_in_owner(
            &binding.rhs,
            UseOwner::Binding(idx),
            producer.symbol,
            ctx,
            term_ids,
        ));
    }
    uses.extend(classify_uses_in_owner(
        tail,
        UseOwner::Tail,
        producer.symbol,
        ctx,
        term_ids,
    ));
    uses
}

fn classify_uses_in_owner(
    term: &Term,
    owner: UseOwner,
    producer_sym: SymbolId,
    ctx: &FusionContext<'_>,
    term_ids: &mut TermIdSource,
) -> Vec<UseSite> {
    let mut uses = Vec::new();
    collect_classified_uses(term, owner, producer_sym, ctx, term_ids, true, &mut uses);

    let recognized_refs = uses.len();
    let raw_refs = scoped_var_ref_count(term, producer_sym);
    if raw_refs > recognized_refs {
        uses.push(UseSite {
            owner,
            kind: UseKind::Escape,
        });
    }

    uses
}

fn collect_classified_uses(
    term: &Term,
    owner: UseOwner,
    producer_sym: SymbolId,
    ctx: &FusionContext<'_>,
    term_ids: &mut TermIdSource,
    allow_screma_input: bool,
    uses: &mut Vec<UseSite>,
) {
    if is_length_call_of(term, producer_sym) {
        uses.push(UseSite {
            owner,
            kind: UseKind::Length {
                term_id: term.id,
                ty: term.ty.clone(),
                span: term.span,
            },
        });
        return;
    }

    let semantics = classify_with_context(term, ctx, term_ids);
    let input_indices = semantic_input_indices(&semantics, producer_sym);
    if !input_indices.is_empty() {
        for input_index in input_indices {
            uses.push(UseSite {
                owner,
                kind: UseKind::SoacInput {
                    term_id: term.id,
                    input_index,
                    semantics: semantics.clone(),
                    ty: term.ty.clone(),
                    span: term.span,
                },
            });
        }
        return;
    }

    if allow_screma_input {
        let screma_inputs = top_screma_input_count(term, producer_sym);
        if screma_inputs > 0 {
            for _ in 0..screma_inputs {
                uses.push(UseSite {
                    owner,
                    kind: UseKind::ScremaInput,
                });
            }
            return;
        }
    }

    match &term.kind {
        TermKind::Lambda(lam) if lam.params.iter().any(|(sym, _)| *sym == producer_sym) => {}
        TermKind::Let { name, rhs, body, .. } => {
            collect_classified_uses(rhs, owner, producer_sym, ctx, term_ids, false, uses);
            if *name != producer_sym {
                collect_classified_uses(body, owner, producer_sym, ctx, term_ids, false, uses);
            }
        }
        TermKind::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body,
            ..
        } => {
            collect_classified_uses(init, owner, producer_sym, ctx, term_ids, false, uses);
            for (_, _, extraction) in init_bindings {
                collect_classified_uses(extraction, owner, producer_sym, ctx, term_ids, false, uses);
            }
            match kind {
                super::LoopKind::For { var, iter, .. } => {
                    collect_classified_uses(iter, owner, producer_sym, ctx, term_ids, false, uses);
                    if *loop_var != producer_sym && *var != producer_sym {
                        collect_classified_uses(body, owner, producer_sym, ctx, term_ids, false, uses);
                    }
                }
                super::LoopKind::ForRange { var, bound, .. } => {
                    collect_classified_uses(bound, owner, producer_sym, ctx, term_ids, false, uses);
                    if *loop_var != producer_sym && *var != producer_sym {
                        collect_classified_uses(body, owner, producer_sym, ctx, term_ids, false, uses);
                    }
                }
                super::LoopKind::While { cond } => {
                    collect_classified_uses(cond, owner, producer_sym, ctx, term_ids, false, uses);
                    if *loop_var != producer_sym {
                        collect_classified_uses(body, owner, producer_sym, ctx, term_ids, false, uses);
                    }
                }
            }
        }
        _ => term.for_each_child(&mut |child| {
            collect_classified_uses(child, owner, producer_sym, ctx, term_ids, false, uses);
        }),
    }
}

fn semantic_input_indices(semantics: &ArraySemantics, sym: SymbolId) -> Vec<usize> {
    semantics
        .input_exprs()
        .iter()
        .enumerate()
        .filter_map(|(idx, ae)| (ae.as_named_ref() == Some(sym)).then_some(idx))
        .collect()
}

fn top_screma_input_count(term: &Term, sym: SymbolId) -> usize {
    let TermKind::Soac(SoacOp::Screma { inputs, .. }) = &term.kind else {
        return 0;
    };
    inputs.iter().filter(|input| input.as_named_ref() == Some(sym)).count()
}

fn find_direct_horizontal_map_plan(
    bindings: &[LetBinding],
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<FusionPlan> {
    let group = find_direct_map_screma_group(bindings)?;
    let span = group.span;
    let result_fields: Vec<_> =
        group.maps.iter().map(|consumer| bindings[consumer.binding_idx].name_ty.clone()).collect();
    let map_lams: Vec<_> = group.maps.iter().map(|consumer| consumer.lam.clone()).collect();
    let projection_fields: HashMap<usize, usize> = group
        .maps
        .iter()
        .enumerate()
        .map(|(field_idx, consumer)| (consumer.binding_idx, field_idx))
        .collect();
    let rewrite = make_screma_rewrite(
        group.insert_at,
        group.skip_indices,
        group.symbol_name,
        result_fields,
        map_lams,
        vec![],
        group.inputs,
        span,
        projection_fields,
        symbols,
        term_ids,
    );
    Some(FusionPlan::Screma(PlannedScremaRewrite {
        rewrite,
        term_replacements: HashMap::new(),
        tail_projection: None,
    }))
}

fn find_map_group_plan(
    producer: &ProducerInfo,
    uses: &[UseSite],
    bindings: &[LetBinding],
    tail: &Term,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<FusionPlan> {
    let ArraySemantics::Elementwise {
        inputs: producer_inputs,
        body: producer_body,
    } = &producer.semantics
    else {
        return None;
    };
    if uses.len() < 2 || uses.iter().any(|u| !matches!(u.owner, UseOwner::Binding(_))) {
        return None;
    }
    if uses.iter().any(|u| !matches!(u.kind, UseKind::SoacInput { .. })) {
        return None;
    }

    let mut maps = Vec::new();
    let mut accumulators = Vec::new();
    let mut consumer_indices = Vec::new();
    for use_site in uses {
        let UseOwner::Binding(binding_idx) = use_site.owner else {
            return None;
        };
        let UseKind::SoacInput {
            term_id,
            input_index,
            semantics,
            ..
        } = &use_site.kind
        else {
            return None;
        };
        if *term_id != bindings[binding_idx].rhs.id {
            return None;
        }
        if *input_index != 0 {
            return None;
        }
        match semantics {
            ArraySemantics::Elementwise { inputs, body } if inputs.len() == 1 => {
                maps.push(ScremaMapGroupConsumer {
                    binding_idx,
                    lam: super::SoacBody {
                        lam: compose_lambdas(
                            producer_body.lam.clone(),
                            body.lam.clone(),
                            bindings[binding_idx].span,
                            symbols,
                            term_ids,
                        ),
                        captures: vec![],
                    },
                });
                consumer_indices.push(binding_idx);
            }
            ArraySemantics::Reduction { op, init, .. } if op.lam.params.len() == 2 => {
                accumulators.push(ScremaReduceConsumer {
                    binding_idx,
                    op: super::SoacBody {
                        lam: compose_map_reduce(
                            producer_body.lam.clone(),
                            op.lam.clone(),
                            bindings[binding_idx].span,
                            symbols,
                            term_ids,
                        ),
                        captures: vec![],
                    },
                    reduce_op: op.clone(),
                    ne: init.clone(),
                    accumulator: super::ScremaAccumulator::Reduce,
                });
                consumer_indices.push(binding_idx);
            }
            ArraySemantics::PrefixScan { .. } => {
                let scan_sym = bindings[binding_idx].name;
                if bindings
                    .iter()
                    .skip(binding_idx + 1)
                    .any(|binding| term_mentions_any(&binding.rhs, &[scan_sym]))
                    || !symbol_uses_are_direct_tail_values(tail, scan_sym)
                {
                    return None;
                }
                let mut accumulator =
                    screma_consumer_accumulator(&bindings[binding_idx].rhs, producer.symbol)?;
                accumulator.op = super::SoacBody {
                    lam: compose_map_reduce(
                        producer_body.lam.clone(),
                        accumulator.op.lam.clone(),
                        bindings[binding_idx].span,
                        symbols,
                        term_ids,
                    ),
                    captures: vec![],
                };
                accumulator.binding_idx = binding_idx;
                accumulators.push(accumulator);
                consumer_indices.push(binding_idx);
            }
            _ => return None,
        }
    }
    if consumer_indices_are_dependent(bindings, &consumer_indices) {
        return None;
    }

    let mut result_fields = Vec::with_capacity(maps.len() + accumulators.len());
    result_fields.extend(maps.iter().map(|consumer| bindings[consumer.binding_idx].name_ty.clone()));
    result_fields
        .extend(accumulators.iter().map(|consumer| bindings[consumer.binding_idx].name_ty.clone()));
    let accumulator_specs: Vec<_> = accumulators
        .iter()
        .map(|consumer| super::ScremaAccumulatorSpec {
            kind: consumer.accumulator,
            step_lam: consumer.op.clone(),
            reduce_op: consumer.reduce_op.clone(),
            ne: consumer.ne.clone(),
        })
        .collect();
    let mut projection_fields = HashMap::new();
    projection_fields
        .extend(maps.iter().enumerate().map(|(field_idx, consumer)| (consumer.binding_idx, field_idx)));
    projection_fields.extend(
        accumulators
            .iter()
            .enumerate()
            .map(|(acc_idx, consumer)| (consumer.binding_idx, maps.len() + acc_idx)),
    );
    let rewrite = make_screma_rewrite(
        producer.binding_idx,
        vec![producer.binding_idx],
        "_screma",
        result_fields,
        maps.into_iter().map(|m| m.lam).collect(),
        accumulator_specs,
        producer_inputs.clone(),
        bindings[producer.binding_idx].span,
        projection_fields,
        symbols,
        term_ids,
    );
    Some(FusionPlan::Screma(PlannedScremaRewrite {
        rewrite,
        term_replacements: HashMap::new(),
        tail_projection: None,
    }))
}

fn find_filter_plan(
    producer: &ProducerInfo,
    uses: &[UseSite],
    bindings: &[LetBinding],
    tail: &Term,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<FusionPlan> {
    let ArraySemantics::Filter { input, pred } = &producer.semantics else {
        return None;
    };
    if pred.lam.params.len() != 1 || uses.is_empty() {
        return None;
    }

    let mut reductions: Vec<(
        UseOwner,
        TermId,
        super::ScremaAccumulatorSpec,
        Type<TypeName>,
        Span,
    )> = Vec::new();
    let mut lengths: Vec<(UseOwner, TermId, Type<TypeName>, Span)> = Vec::new();
    for use_site in uses {
        match &use_site.kind {
            UseKind::Length { term_id, ty, span } => {
                lengths.push((use_site.owner, *term_id, ty.clone(), *span));
            }
            UseKind::SoacInput {
                term_id,
                semantics,
                ty,
                span,
                ..
            } => {
                let ArraySemantics::Reduction { op, init, .. } = semantics else {
                    return None;
                };
                if op.lam.params.len() != 2 {
                    return None;
                }
                let step_lam = filtered_reduce_step(pred, op, *span, symbols, term_ids)?;
                reductions.push((
                    use_site.owner,
                    *term_id,
                    super::ScremaAccumulatorSpec {
                        kind: super::ScremaAccumulator::Reduce,
                        step_lam: super::SoacBody {
                            lam: step_lam,
                            captures: vec![],
                        },
                        reduce_op: op.clone(),
                        ne: init.clone(),
                    },
                    ty.clone(),
                    *span,
                ));
            }
            UseKind::ScremaInput | UseKind::Escape => return None,
        }
    }
    if reductions.is_empty() && lengths.is_empty() {
        return None;
    }

    let mut result_fields = Vec::new();
    let mut accumulators = Vec::new();
    let mut projection_fields = HashMap::new();
    let mut tail_projection_field: Option<(usize, Type<TypeName>, Span)> = None;
    let mut nested_replacement_fields: Vec<(TermId, usize, Type<TypeName>, Span)> = Vec::new();
    let skip_indices = vec![producer.binding_idx];

    for (owner, term_id, acc, ty, span) in reductions {
        let field_idx = result_fields.len();
        result_fields.push(ty.clone());
        accumulators.push(acc);
        match owner {
            UseOwner::Binding(idx) if bindings[idx].rhs.id == term_id => {
                projection_fields.insert(idx, field_idx);
            }
            UseOwner::Binding(_) => {
                nested_replacement_fields.push((term_id, field_idx, ty, span));
            }
            UseOwner::Tail if tail.id == term_id => {
                tail_projection_field = Some((field_idx, ty, span));
            }
            UseOwner::Tail => {
                nested_replacement_fields.push((term_id, field_idx, ty, span));
            }
        }
    }

    let mut length_replacement_fields: Vec<(TermId, usize, Type<TypeName>, Span)> = Vec::new();
    if let Some((_, _, count_ty, count_span)) = lengths.first().cloned() {
        let count_field = result_fields.len();
        result_fields.push(count_ty.clone());
        accumulators.push(count_accumulator(
            pred,
            count_ty.clone(),
            count_span,
            symbols,
            term_ids,
        )?);
        for (owner, term_id, ty, span) in lengths {
            match owner {
                UseOwner::Binding(idx) if bindings[idx].rhs.id == term_id => {
                    projection_fields.insert(idx, count_field);
                }
                _ => length_replacement_fields.push((term_id, count_field, ty, span)),
            }
        }
    }

    let rewrite = make_screma_rewrite(
        producer.binding_idx,
        skip_indices,
        "_filtered_screma",
        result_fields,
        vec![],
        accumulators,
        vec![input.clone()],
        bindings[producer.binding_idx].span,
        projection_fields,
        symbols,
        term_ids,
    );
    let mut term_replacements = HashMap::new();
    for (term_id, field_idx, ty, span) in
        nested_replacement_fields.into_iter().chain(length_replacement_fields)
    {
        term_replacements.insert(
            term_id,
            ProjectionTemplate {
                tuple_sym: rewrite.tuple_sym,
                tuple_ty: rewrite.tuple_ty.clone(),
                field_idx,
                ty,
                span,
            },
        );
    }
    let tail_projection = tail_projection_field.map(|(field_idx, ty, span)| ProjectionTemplate {
        tuple_sym: rewrite.tuple_sym,
        tuple_ty: rewrite.tuple_ty.clone(),
        field_idx,
        ty,
        span,
    });
    Some(FusionPlan::Screma(PlannedScremaRewrite {
        rewrite,
        term_replacements,
        tail_projection,
    }))
}

fn find_pairwise_plan(
    producer: &ProducerInfo,
    uses: &[UseSite],
    bindings: &[LetBinding],
    tail: &Term,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<FusionPlan> {
    if uses.len() != 1 {
        return None;
    }
    let use_site = &uses[0];
    match &use_site.kind {
        UseKind::ScremaInput => {
            let UseOwner::Binding(consumer_idx) = use_site.owner else {
                return None;
            };
            let map_producer = screma_producer_map(&bindings[producer.binding_idx].rhs)?;
            let fused_rhs = map_into_screma_rhs(
                &map_producer,
                bindings[producer.binding_idx].name,
                &bindings[consumer_idx].rhs,
                bindings[consumer_idx].span,
                symbols,
                term_ids,
            )?;
            Some(FusionPlan::ReplaceConsumer {
                producer_idx: producer.binding_idx,
                consumer_idx: Some(consumer_idx),
                fused_rhs,
            })
        }
        UseKind::SoacInput {
            term_id,
            input_index,
            semantics,
            ty,
            span,
        } => {
            match use_site.owner {
                UseOwner::Binding(idx) if *term_id == bindings[idx].rhs.id => {}
                UseOwner::Tail if *term_id == tail.id => {}
                _ => return None,
            }
            let fk = can_fuse(&producer.semantics, semantics);
            if fk == FusionKind::NotFusible {
                return None;
            }
            if fk == FusionKind::ComposeElementwise {
                if let ArraySemantics::Elementwise { inputs, .. } = semantics {
                    if inputs.len() != 1 {
                        return None;
                    }
                }
            }
            let fused_rhs = build_fused_from_semantics(
                &producer.semantics,
                semantics,
                &fk,
                *input_index,
                ty.clone(),
                *span,
                symbols,
                term_ids,
            )?;
            Some(FusionPlan::ReplaceConsumer {
                producer_idx: producer.binding_idx,
                consumer_idx: match use_site.owner {
                    UseOwner::Binding(idx) => Some(idx),
                    UseOwner::Tail => None,
                },
                fused_rhs,
            })
        }
        UseKind::Length { .. } | UseKind::Escape => None,
    }
}

fn apply_fusion_plan(
    inner: &Term,
    bindings: &[LetBinding],
    tail: Term,
    plan: FusionPlan,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    match plan {
        FusionPlan::ReplaceConsumer {
            producer_idx,
            consumer_idx,
            fused_rhs,
        } => {
            let prod_sym = bindings[producer_idx].name;
            let cons_sym = consumer_idx.map(|idx| bindings[idx].name);
            rewrite_let_chain(inner, prod_sym, cons_sym, fused_rhs, term_ids)
        }
        FusionPlan::Screma(plan) => {
            let new_bindings = apply_planned_screma_rewrite(bindings, plan, term_ids);
            let tail = replace_projection_terms(tail, &new_bindings.1, term_ids);
            let tail = new_bindings.2.unwrap_or(tail);
            Some(rebuild_let_chain(new_bindings.0, tail, term_ids))
        }
    }
}

fn apply_planned_screma_rewrite(
    bindings: &[LetBinding],
    plan: PlannedScremaRewrite,
    term_ids: &mut TermIdSource,
) -> (Vec<LetBinding>, HashMap<TermId, ProjectionTemplate>, Option<Term>) {
    let mut new_bindings = Vec::with_capacity(bindings.len() + 1 - plan.rewrite.skip_indices.len());
    for (idx, binding) in bindings.iter().enumerate() {
        if idx == plan.rewrite.insert_at {
            new_bindings.push(plan.rewrite.fused_binding.clone());
        }
        if plan.rewrite.skip_indices.contains(&idx) {
            continue;
        }
        if let Some(&proj_idx) = plan.rewrite.projection_fields.get(&idx) {
            new_bindings.push(tuple_projection_binding(
                binding,
                plan.rewrite.tuple_sym,
                &plan.rewrite.tuple_ty,
                proj_idx,
                term_ids,
            ));
        } else {
            new_bindings.push(LetBinding {
                rhs: replace_projection_terms(binding.rhs.clone(), &plan.term_replacements, term_ids),
                ..binding.clone()
            });
        }
    }
    if plan.rewrite.insert_at >= bindings.len() {
        new_bindings.push(plan.rewrite.fused_binding);
    }
    let tail = plan.tail_projection.map(|projection| projection_term(&projection, term_ids));
    (new_bindings, plan.term_replacements, tail)
}

fn projection_term(projection: &ProjectionTemplate, term_ids: &mut TermIdSource) -> Term {
    let tuple_ref = Term {
        id: term_ids.next_id(),
        ty: projection.tuple_ty.clone(),
        span: projection.span,
        kind: TermKind::Var(VarRef::Symbol(projection.tuple_sym)),
    };
    Term {
        id: term_ids.next_id(),
        ty: projection.ty.clone(),
        span: projection.span,
        kind: TermKind::TupleProj {
            tuple: Box::new(tuple_ref),
            idx: projection.field_idx,
        },
    }
}

fn replace_projection_terms(
    term: Term,
    replacements: &HashMap<TermId, ProjectionTemplate>,
    term_ids: &mut TermIdSource,
) -> Term {
    if let Some(projection) = replacements.get(&term.id) {
        return projection_term(projection, term_ids);
    }
    let Term {
        id: _,
        ty,
        span,
        kind,
    } = term;
    match kind {
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => Term {
            id: term_ids.next_id(),
            ty,
            span,
            kind: TermKind::Let {
                name,
                name_ty,
                rhs: Box::new(replace_projection_terms(*rhs, replacements, term_ids)),
                body: Box::new(replace_projection_terms(*body, replacements, term_ids)),
            },
        },
        TermKind::Lambda(lam) => Term {
            id: term_ids.next_id(),
            ty,
            span,
            kind: TermKind::Lambda(Lambda {
                body: Box::new(replace_projection_terms(*lam.body, replacements, term_ids)),
                ..lam
            }),
        },
        other => Term {
            id: term_ids.next_id(),
            ty,
            span,
            kind: other,
        }
        .map_children(&mut |child| replace_projection_terms(child, replacements, term_ids)),
    }
}

fn classify_with_context(
    term: &Term,
    ctx: &FusionContext<'_>,
    term_ids: &mut TermIdSource,
) -> ArraySemantics {
    let direct = classify_term(term);
    if !matches!(direct, ArraySemantics::Opaque) {
        return direct;
    }
    let TermKind::App { func, args } = &term.kind else {
        return ArraySemantics::Opaque;
    };
    let TermKind::Var(VarRef::Symbol(callee_sym)) = &func.kind else {
        return ArraySemantics::Opaque;
    };
    let def_sym = ctx.sym_to_def.get(callee_sym).unwrap_or(callee_sym);
    let Some(summary) = ctx.summaries.get(def_sym) else {
        return ArraySemantics::Opaque;
    };
    let ResultSemantics::Produces(semantics) = &summary.result else {
        return ArraySemantics::Opaque;
    };
    substitute_summary_args(semantics, &summary.params, args, term_ids)
}

fn substitute_summary_args(
    semantics: &ArraySemantics,
    summary_params: &[(SymbolId, Type<TypeName>)],
    call_args: &[Term],
    term_ids: &mut TermIdSource,
) -> ArraySemantics {
    let param_to_arg: HashMap<SymbolId, Term> = summary_params
        .iter()
        .zip(call_args)
        .map(|((param_sym, _), arg)| (*param_sym, arg.clone()))
        .collect();

    let mut result = semantics.clone();
    for (&param_sym, arg) in &param_to_arg {
        result = subst_in_semantics(result, param_sym, arg, term_ids);
    }
    result
}

fn subst_in_semantics(
    sem: ArraySemantics,
    old: SymbolId,
    replacement: &Term,
    term_ids: &mut TermIdSource,
) -> ArraySemantics {
    fn sub_ae(ae: ArrayExpr, old: SymbolId, replacement: &Term, ids: &mut TermIdSource) -> ArrayExpr {
        substitute_term_in_array_expr(ae, old, replacement, ids)
    }

    fn sub_sb(
        sb: super::SoacBody,
        old: SymbolId,
        replacement: &Term,
        ids: &mut TermIdSource,
    ) -> super::SoacBody {
        super::SoacBody {
            lam: Lambda {
                body: if sb.lam.params.iter().any(|(param, _)| *param == old) {
                    sb.lam.body
                } else {
                    Box::new(substitute_term_expr(*sb.lam.body, old, replacement, ids))
                },
                ..sb.lam
            },
            captures: sb
                .captures
                .into_iter()
                .map(|(sym, ty, expr)| (sym, ty, substitute_term_expr(expr, old, replacement, ids)))
                .collect(),
        }
    }

    match sem {
        ArraySemantics::Elementwise { inputs, body } => ArraySemantics::Elementwise {
            inputs: inputs.into_iter().map(|ae| sub_ae(ae, old, replacement, term_ids)).collect(),
            body: sub_sb(body, old, replacement, term_ids),
        },
        ArraySemantics::Reduction { input, op, init } => ArraySemantics::Reduction {
            input: sub_ae(input, old, replacement, term_ids),
            op: sub_sb(op, old, replacement, term_ids),
            init: Box::new(substitute_term_expr(*init, old, replacement, term_ids)),
        },
        ArraySemantics::PrefixScan { input, op, init } => ArraySemantics::PrefixScan {
            input: sub_ae(input, old, replacement, term_ids),
            op: sub_sb(op, old, replacement, term_ids),
            init: Box::new(substitute_term_expr(*init, old, replacement, term_ids)),
        },
        ArraySemantics::Filter { input, pred } => ArraySemantics::Filter {
            input: sub_ae(input, old, replacement, term_ids),
            pred: sub_sb(pred, old, replacement, term_ids),
        },
        other => other,
    }
}

fn clone_term_with_fresh_ids(term: &Term, term_ids: &mut TermIdSource) -> Term {
    let mut cloned = term.clone().map_children(&mut |child| {
        let mut child =
            child.map_children(&mut |grandchild| clone_term_with_fresh_ids(&grandchild, term_ids));
        child.id = term_ids.next_id();
        child
    });
    cloned.id = term_ids.next_id();
    cloned
}

fn substitute_term_expr(
    term: Term,
    old: SymbolId,
    replacement: &Term,
    term_ids: &mut TermIdSource,
) -> Term {
    match term.kind {
        TermKind::Var(VarRef::Symbol(sym)) if sym == old => {
            clone_term_with_fresh_ids(replacement, term_ids)
        }
        TermKind::Lambda(lam) if lam.params.iter().any(|(param, _)| *param == old) => Term {
            kind: TermKind::Lambda(lam),
            ..term
        },
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            let rhs = substitute_term_expr(*rhs, old, replacement, term_ids);
            let body =
                if name == old { *body } else { substitute_term_expr(*body, old, replacement, term_ids) };
            Term {
                id: term_ids.next_id(),
                kind: TermKind::Let {
                    name,
                    name_ty,
                    rhs: Box::new(rhs),
                    body: Box::new(body),
                },
                ..term
            }
        }
        TermKind::Loop {
            loop_var,
            loop_var_ty,
            init,
            init_bindings,
            kind,
            body,
        } => {
            let init = substitute_term_expr(*init, old, replacement, term_ids);
            let init_binding_shadows = init_bindings.iter().any(|(sym, _, _)| *sym == old);
            let init_bindings = init_bindings
                .into_iter()
                .map(|(sym, ty, extraction)| {
                    if init_binding_shadows {
                        (sym, ty, extraction)
                    } else {
                        (
                            sym,
                            ty,
                            substitute_term_expr(extraction, old, replacement, term_ids),
                        )
                    }
                })
                .collect();
            let (kind, kind_shadows) = substitute_term_in_loop_kind(kind, old, replacement, term_ids);
            let body = if loop_var == old || init_binding_shadows || kind_shadows {
                *body
            } else {
                substitute_term_expr(*body, old, replacement, term_ids)
            };
            Term {
                id: term_ids.next_id(),
                kind: TermKind::Loop {
                    loop_var,
                    loop_var_ty,
                    init: Box::new(init),
                    init_bindings,
                    kind,
                    body: Box::new(body),
                },
                ..term
            }
        }
        other => Term { kind: other, ..term }
            .map_children(&mut |child| substitute_term_expr(child, old, replacement, term_ids)),
    }
}

fn substitute_term_in_loop_kind(
    kind: super::LoopKind,
    old: SymbolId,
    replacement: &Term,
    term_ids: &mut TermIdSource,
) -> (super::LoopKind, bool) {
    match kind {
        super::LoopKind::For { var, var_ty, iter } => (
            super::LoopKind::For {
                var,
                var_ty,
                iter: Box::new(substitute_term_expr(*iter, old, replacement, term_ids)),
            },
            var == old,
        ),
        super::LoopKind::ForRange { var, var_ty, bound } => (
            super::LoopKind::ForRange {
                var,
                var_ty,
                bound: Box::new(substitute_term_expr(*bound, old, replacement, term_ids)),
            },
            var == old,
        ),
        super::LoopKind::While { cond } => (
            super::LoopKind::While {
                cond: Box::new(substitute_term_expr(*cond, old, replacement, term_ids)),
            },
            false,
        ),
    }
}

fn substitute_term_in_array_expr(
    ae: ArrayExpr,
    old: SymbolId,
    replacement: &Term,
    term_ids: &mut TermIdSource,
) -> ArrayExpr {
    match ae {
        ArrayExpr::Ref(term) => {
            ArrayExpr::Ref(Box::new(substitute_term_expr(*term, old, replacement, term_ids)))
        }
        ArrayExpr::Zip(items) => ArrayExpr::Zip(
            items
                .into_iter()
                .map(|item| substitute_term_in_array_expr(item, old, replacement, term_ids))
                .collect(),
        ),
        ArrayExpr::Soac(op) => {
            let term = Term {
                id: term_ids.next_id(),
                ty: Type::Variable(0),
                span: replacement.span,
                kind: TermKind::Soac(*op),
            };
            match substitute_term_expr(term, old, replacement, term_ids).kind {
                TermKind::Soac(op) => ArrayExpr::Soac(Box::new(op)),
                other => ArrayExpr::Ref(Box::new(Term {
                    id: term_ids.next_id(),
                    ty: Type::Variable(0),
                    span: replacement.span,
                    kind: other,
                })),
            }
        }
        ArrayExpr::Literal(terms) => ArrayExpr::Literal(
            terms.into_iter().map(|term| substitute_term_expr(term, old, replacement, term_ids)).collect(),
        ),
        ArrayExpr::Range { start, len, step } => ArrayExpr::Range {
            start: Box::new(substitute_term_expr(*start, old, replacement, term_ids)),
            len: Box::new(substitute_term_expr(*len, old, replacement, term_ids)),
            step: step.map(|step| Box::new(substitute_term_expr(*step, old, replacement, term_ids))),
        },
        ArrayExpr::StorageView(view) => ArrayExpr::StorageView(super::StorageView {
            binding: view.binding,
            offset: Box::new(substitute_term_expr(*view.offset, old, replacement, term_ids)),
            len: Box::new(substitute_term_expr(*view.len, old, replacement, term_ids)),
            elem_ty: view.elem_ty,
        }),
    }
}

fn is_length_call_of(term: &Term, producer_sym: SymbolId) -> bool {
    let TermKind::App { func, args } = &term.kind else {
        return false;
    };
    if args.len() != 1 {
        return false;
    }
    let TermKind::Var(VarRef::Builtin { id, .. }) = &func.kind else {
        return false;
    };
    if *id != crate::builtins::catalog().known().length {
        return false;
    }
    matches!(
        &args[0].kind,
        TermKind::Var(VarRef::Symbol(sym)) if *sym == producer_sym
    )
}

fn scoped_var_ref_count(term: &Term, sym: SymbolId) -> usize {
    match &term.kind {
        TermKind::Var(VarRef::Symbol(s)) if *s == sym => 1,
        TermKind::Lambda(lam) if lam.params.iter().any(|(p, _)| *p == sym) => 0,
        TermKind::Let { name, rhs, body, .. } => {
            scoped_var_ref_count(rhs, sym) + if *name == sym { 0 } else { scoped_var_ref_count(body, sym) }
        }
        TermKind::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body,
            ..
        } => {
            let mut count = scoped_var_ref_count(init, sym)
                + init_bindings
                    .iter()
                    .map(|(_, _, extraction)| scoped_var_ref_count(extraction, sym))
                    .sum::<usize>();
            let body_shadowed = match kind {
                super::LoopKind::For { var, iter, .. } => {
                    count += scoped_var_ref_count(iter, sym);
                    *loop_var == sym || *var == sym
                }
                super::LoopKind::ForRange { var, bound, .. } => {
                    count += scoped_var_ref_count(bound, sym);
                    *loop_var == sym || *var == sym
                }
                super::LoopKind::While { cond } => {
                    count += scoped_var_ref_count(cond, sym);
                    *loop_var == sym
                }
            };
            if body_shadowed { count } else { count + scoped_var_ref_count(body, sym) }
        }
        _ => {
            let mut count = 0;
            term.for_each_child(&mut |child| {
                count += scoped_var_ref_count(child, sym);
            });
            count
        }
    }
}

fn consumer_indices_are_dependent(bindings: &[LetBinding], consumer_indices: &[usize]) -> bool {
    for (pos, idx) in consumer_indices.iter().enumerate() {
        let prior: Vec<_> = consumer_indices[..pos].iter().map(|i| bindings[*i].name).collect();
        if !prior.is_empty() && term_mentions_any(&bindings[*idx].rhs, &prior) {
            return true;
        }
    }
    false
}

fn symbol_uses_are_direct_tail_values(term: &Term, sym: SymbolId) -> bool {
    match &term.kind {
        TermKind::Var(VarRef::Symbol(_)) => true,
        TermKind::Tuple(parts) | TermKind::VecLit(parts) => {
            parts.iter().all(|part| symbol_uses_are_direct_tail_values(part, sym))
        }
        TermKind::TupleProj { tuple, .. } | TermKind::Coerce { inner: tuple, .. } => {
            symbol_uses_are_direct_tail_values(tuple, sym)
        }
        TermKind::IntLit(_)
        | TermKind::FloatLit(_)
        | TermKind::BoolLit(_)
        | TermKind::UnitLit
        | TermKind::Var(_)
        | TermKind::BinOp(_)
        | TermKind::UnOp(_)
        | TermKind::Extern(_) => true,
        _ => !term_mentions_any(term, &[sym]),
    }
}

fn filtered_reduce_step(
    pred: &super::SoacBody,
    op: &super::SoacBody,
    span: Span,
    _symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<Lambda> {
    if pred.lam.params.len() != 1 || op.lam.params.len() != 2 {
        return None;
    }
    let acc_param = op.lam.params[0].clone();
    let elem_param = pred.lam.params[0].clone();
    let op_elem = op.lam.params[1].0;
    let then_branch = substitute_sym(*op.lam.body.clone(), op_elem, elem_param.0, term_ids);
    let else_branch = Term {
        id: term_ids.next_id(),
        ty: op.lam.ret_ty.clone(),
        span,
        kind: TermKind::Var(VarRef::Symbol(acc_param.0)),
    };
    Some(Lambda {
        params: vec![acc_param, elem_param],
        body: Box::new(Term {
            id: term_ids.next_id(),
            ty: op.lam.ret_ty.clone(),
            span,
            kind: TermKind::If {
                cond: pred.lam.body.clone(),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            },
        }),
        ret_ty: op.lam.ret_ty.clone(),
    })
}

fn count_accumulator(
    pred: &super::SoacBody,
    count_ty: Type<TypeName>,
    span: Span,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<super::ScremaAccumulatorSpec> {
    if pred.lam.params.len() != 1 {
        return None;
    }
    let acc_sym = symbols.alloc("_count".to_string());
    let acc_param = (acc_sym, count_ty.clone());
    let elem_param = pred.lam.params[0].clone();
    let plus_one = add_terms(
        var_term(acc_sym, count_ty.clone(), span, term_ids),
        int_term("1", count_ty.clone(), span, term_ids),
        count_ty.clone(),
        span,
        term_ids,
    );
    let step_lam = Lambda {
        params: vec![acc_param.clone(), elem_param],
        body: Box::new(Term {
            id: term_ids.next_id(),
            ty: count_ty.clone(),
            span,
            kind: TermKind::If {
                cond: pred.lam.body.clone(),
                then_branch: Box::new(plus_one),
                else_branch: Box::new(var_term(acc_sym, count_ty.clone(), span, term_ids)),
            },
        }),
        ret_ty: count_ty.clone(),
    };

    let rhs_sym = symbols.alloc("_count_rhs".to_string());
    let reduce_op = Lambda {
        params: vec![acc_param, (rhs_sym, count_ty.clone())],
        body: Box::new(add_terms(
            var_term(acc_sym, count_ty.clone(), span, term_ids),
            var_term(rhs_sym, count_ty.clone(), span, term_ids),
            count_ty.clone(),
            span,
            term_ids,
        )),
        ret_ty: count_ty.clone(),
    };

    Some(super::ScremaAccumulatorSpec {
        kind: super::ScremaAccumulator::Reduce,
        step_lam: super::SoacBody {
            lam: step_lam,
            captures: vec![],
        },
        reduce_op: super::SoacBody {
            lam: reduce_op,
            captures: vec![],
        },
        ne: Box::new(int_term("0", count_ty, span, term_ids)),
    })
}

fn var_term(sym: SymbolId, ty: Type<TypeName>, span: Span, term_ids: &mut TermIdSource) -> Term {
    Term {
        id: term_ids.next_id(),
        ty,
        span,
        kind: TermKind::Var(VarRef::Symbol(sym)),
    }
}

fn int_term(value: &str, ty: Type<TypeName>, span: Span, term_ids: &mut TermIdSource) -> Term {
    Term {
        id: term_ids.next_id(),
        ty,
        span,
        kind: TermKind::IntLit(value.to_string()),
    }
}

fn add_terms(lhs: Term, rhs: Term, ty: Type<TypeName>, span: Span, term_ids: &mut TermIdSource) -> Term {
    let func_ty = Type::Constructed(
        TypeName::Arrow,
        vec![
            ty.clone(),
            Type::Constructed(TypeName::Arrow, vec![ty.clone(), ty.clone()]),
        ],
    );
    let func = Term {
        id: term_ids.next_id(),
        ty: func_ty,
        span,
        kind: TermKind::BinOp(crate::ast::BinaryOp { op: "+".to_string() }),
    };
    Term {
        id: term_ids.next_id(),
        ty,
        span,
        kind: TermKind::App {
            func: Box::new(func),
            args: vec![lhs, rhs],
        },
    }
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
) -> (Term, bool) {
    let mut changed = false;
    let term = term.map_children(&mut |child| {
        let (child, did_fuse) = fuse_term(child, ctx, symbols, term_ids);
        changed |= did_fuse;
        child
    });

    if matches!(term.kind, TermKind::Let { .. }) {
        let (fused, did_fuse) = fuse_def_body(term, ctx, symbols, term_ids);
        return (fused, changed || did_fuse);
    }

    (term, changed)
}

/// Fuse within a body term. Returns the new body and whether any fusion happened.
fn fuse_def_body(
    body: Term,
    ctx: &FusionContext<'_>,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> (Term, bool) {
    let (params, inner) = extract_lambda_params(&body);
    let (bindings, tail) = flatten_let_chain(inner.clone());
    if bindings.is_empty() {
        return (body, false);
    }

    let Some(plan) = find_fusion_plan(&bindings, &tail, ctx, symbols, term_ids) else {
        return (body, false);
    };

    let Some(fused_inner) = apply_fusion_plan(&inner, &bindings, tail, plan, term_ids) else {
        return (body, false);
    };

    (rebuild_lambda_params(params, fused_inner, term_ids), true)
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
