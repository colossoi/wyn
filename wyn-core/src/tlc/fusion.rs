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
use crate::{LookupSet, SymbolId, SymbolTable};
use polytype::Type;

use crate::LookupMap;

use super::array_semantics::{can_fuse, classify_term, ArraySemantics, FusionRecipe};
use super::parallelize::{collect_extra_slot_stores, let_term};
use super::subst::substitute_sym;
use super::{
    extract_lambda_params, mentions_any, ArrayExpr, Def, DefMeta, Lambda, LetBinding, LetChain, Program,
    SoacDestination, SoacOp, Term, TermId, TermIdSource, TermKind,
};

// =============================================================================
// Verifier: SOACs never hide behind a call boundary
// =============================================================================

/// A call site where a user function calls another user function whose body
/// still hides a SOAC (or a `length` intrinsic) behind the call boundary.
#[derive(Debug, PartialEq, Eq)]
pub struct CalledSoacHelper {
    /// The def whose body contains the offending call.
    pub caller: SymbolId,
    /// The SOAC-bearing def it calls.
    pub callee: SymbolId,
}

/// Verify that no def calls a SOAC-bearing user function — i.e. every SOAC
/// lives in the def fusion will optimize, never behind a call. This is the
/// precondition that lets fusion be purely *intra*procedural: a clean result
/// means interprocedural fusion needn't exist, because there is no
/// producer/consumer edge that crosses a call.
///
/// `inline::run_force_soac_helpers` is responsible for establishing this by
/// expanding every SOAC-bearing helper at its call sites before fusion runs.
/// It defers two categories — helpers with control flow, and (pre-monomorphize)
/// helpers with free type variables — so this validator only holds once those
/// have been resolved upstream (e.g. by running fusion after monomorphization).
///
/// Entry points are exempt by construction: they are roots, so no `App` names
/// them, and a SOAC in an entry body is never a *called* SOAC. Returns every
/// violation rather than the first, so the report is a complete picture of
/// where the invariant breaks.
pub fn verify_soac_helpers_inlined(program: &Program) -> Result<(), Vec<CalledSoacHelper>> {
    let soac_bearing: LookupSet<SymbolId> =
        program.defs.iter().filter(|d| super::inline::contains_soac(&d.body)).map(|d| d.name).collect();

    let mut violations = Vec::new();
    for def in &program.defs {
        collect_called_soac_helpers(&def.body, def.name, &soac_bearing, &mut violations);
    }
    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

fn collect_called_soac_helpers(
    term: &Term,
    caller: SymbolId,
    soac_bearing: &LookupSet<SymbolId>,
    out: &mut Vec<CalledSoacHelper>,
) {
    if let TermKind::App { func, .. } = &term.kind {
        if let TermKind::Var(VarRef::Symbol(callee)) = &func.kind {
            if soac_bearing.contains(callee) {
                out.push(CalledSoacHelper {
                    caller,
                    callee: *callee,
                });
            }
        }
    }
    term.for_each_child(&mut |c| collect_called_soac_helpers(c, caller, soac_bearing, out));
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
    debug_assert!(
        verify_soac_helpers_inlined(&program).is_ok(),
        "fusion entered with a SOAC helper behind a call boundary; \
         force-inline (or a pass between it and fusion) failed to expose it: {:?}",
        verify_soac_helpers_inlined(&program).err(),
    );

    // Normalize: lift SOACs out of nested positions into let bindings
    let mut program = super::normalize::normalize(program);

    let mut changed = true;
    while changed {
        changed = false;

        let mut symbols = program.symbols;
        let def_syms = program.def_syms;
        let mut term_ids = TermIdSource::new();

        let defs = program
            .defs
            .into_iter()
            .map(|def| {
                // Bottom-up: fuse children first, then try one classified-use
                // rewrite for this definition. The outer fixpoint reruns the
                // analysis after each rewrite.
                let (new_body, did_child_fuse) = fuse_term(def.body, &mut symbols, &mut term_ids);
                if did_child_fuse {
                    changed = true;
                }
                let (new_body, did_fuse) = fuse_def_body(new_body, &mut symbols, &mut term_ids);
                if did_fuse {
                    changed = true;
                }
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

    super::anf::debug_check(&program, "fusion");
    program
}

// =============================================================================
// Per-def fusion
// =============================================================================

#[derive(Clone)]
struct ProducerInfo {
    id: SoacNodeId,
    binding_idx: usize,
    symbol: SymbolId,
    semantics: ArraySemantics,
    inputs: Vec<SoacInputSlot>,
    outputs: Vec<SoacOutputSlot>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SoacNodeId(usize);

#[derive(Clone)]
struct SoacInputSlot {
    slot: usize,
    array: ArrayExpr,
    param: Option<(SymbolId, Type<TypeName>)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NamedInputSlot {
    slot: usize,
    array_sym: SymbolId,
    array_ty: Type<TypeName>,
    param_ty: Type<TypeName>,
}

#[derive(Clone)]
struct SoacOutputSlot {
    slot: usize,
    value: SymbolId,
    ty: Type<TypeName>,
    kind: SoacOutputKind,
}

/// What a producer's output *is*, named by the SOAC that yields it. Only
/// `ReduceScalar` is a scalar; every other kind is an array, which is what gates
/// a binding from being a fusion producer (`has_array_output`). The richer
/// vocabulary (vs a bare `Array`/`Scalar`) keeps the producer's shape legible at
/// the use sites and is the seam for slot-aware multi-output producers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SoacOutputKind {
    MapArray,
    ReduceScalar,
    ScanArray,
    FilterArray,
    /// Scatter / reduce-by-index / literal / range / storage-buffer — array
    /// producers that fusion treats uniformly (it never fuses *into* them here).
    OtherArray,
}

impl SoacOutputKind {
    fn is_array(self) -> bool {
        !matches!(self, SoacOutputKind::ReduceScalar)
    }
}

#[derive(Clone)]
struct FusionRegion {
    producers: Vec<ProducerInfo>,
    binding_nodes: Vec<Option<SoacNodeId>>,
    edges: Vec<UseEdge>,
}

#[derive(Clone)]
struct UseEdge {
    producer: SoacNodeId,
    output_slot: usize,
    site: UseSite,
}

impl FusionRegion {
    fn new(bindings: &[LetBinding], tail: &Term) -> Self {
        let mut producers = Vec::new();
        let mut binding_nodes = vec![None; bindings.len()];
        for (binding_idx, binding) in bindings.iter().enumerate() {
            let semantics = classify_term(&binding.rhs);
            if matches!(semantics, ArraySemantics::Opaque) {
                continue;
            }
            let id = SoacNodeId(producers.len());
            binding_nodes[binding_idx] = Some(id);
            let producer = ProducerInfo {
                id,
                binding_idx,
                symbol: binding.name,
                inputs: soac_input_slots(&semantics),
                outputs: soac_output_slots(binding, &semantics),
                semantics,
            };
            // The output slot mirrors the binding's declared type; a producer's
            // array output is exactly what a fused consumer projects.
            debug_assert!(
                producer.array_output_ty().map(|ty| ty == &binding.name_ty).unwrap_or(true),
                "producer output type must match its binding's declared type"
            );
            producers.push(producer);
        }
        let edges = producers
            .iter()
            .flat_map(|producer| {
                classify_uses_for_producer(producer, bindings, tail)
                    .into_iter()
                    .map(|site| UseEdge {
                        producer: producer.id,
                        output_slot: 0,
                        site,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        FusionRegion {
            producers,
            binding_nodes,
            edges,
        }
    }

    fn uses_for(&self, producer: &ProducerInfo) -> Vec<UseSite> {
        self.edges
            .iter()
            .filter(|edge| edge.producer == producer.id && edge.output_slot == 0)
            .map(|edge| edge.site.clone())
            .collect()
    }

    fn node_for_binding(&self, binding_idx: usize) -> Option<&ProducerInfo> {
        self.binding_nodes.get(binding_idx).and_then(|id| id.map(|SoacNodeId(idx)| &self.producers[idx]))
    }
}

impl ProducerInfo {
    fn output_symbol(&self) -> SymbolId {
        self.outputs
            .first()
            .map(|output| {
                debug_assert_eq!(output.slot, 0);
                output.value
            })
            .unwrap_or(self.symbol)
    }

    fn has_array_output(&self) -> bool {
        self.outputs.iter().any(|output| output.kind.is_array())
    }

    /// The element type of this producer's array output, if it has one — the
    /// field type a consumer projects when the producer is unioned into a
    /// fused Screma. (Currently every producer has exactly one output; this is
    /// the seam for slot-aware multi-output producers.)
    fn array_output_ty(&self) -> Option<&Type<TypeName>> {
        self.outputs.iter().find(|output| output.kind.is_array()).map(|output| &output.ty)
    }
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
struct BindingSoacUse {
    binding_idx: usize,
    input: SoacInputSlot,
    semantics: ArraySemantics,
}

#[derive(Clone)]
enum UseKind {
    SoacInput {
        term_id: TermId,
        input_index: usize,
        input: SoacInputSlot,
        semantics: ArraySemantics,
        ty: Type<TypeName>,
        span: Span,
    },
    ScremaInput {
        input_index: usize,
        input: ArrayExpr,
    },
    Length {
        term_id: TermId,
        ty: Type<TypeName>,
        span: Span,
    },
    Escape,
}

fn binding_soac_uses(uses: &[UseSite], bindings: &[LetBinding]) -> Option<Vec<BindingSoacUse>> {
    uses.iter()
        .map(|use_site| {
            let UseOwner::Binding(binding_idx) = use_site.owner else {
                return None;
            };
            let UseKind::SoacInput {
                term_id,
                input,
                semantics,
                ..
            } = &use_site.kind
            else {
                return None;
            };
            if *term_id != bindings[binding_idx].rhs.id {
                return None;
            }
            Some(BindingSoacUse {
                binding_idx,
                input: input.clone(),
                semantics: semantics.clone(),
            })
        })
        .collect()
}

#[derive(Clone)]
struct ProjectionTemplate {
    tuple_sym: SymbolId,
    tuple_ty: Type<TypeName>,
    field_idx: usize,
    ty: Type<TypeName>,
    span: Span,
    /// The fused Screma has a single output (one map lane xor one accumulator):
    /// its symbol IS that value, so reference it directly instead of projecting.
    /// Set from the rewrite's shape, never inferred from `tuple_ty` (which can
    /// be a tuple when the sole output is itself a tuple value).
    single_output: bool,
}

struct PlannedScremaRewrite {
    rewrite: ScremaRewrite,
    term_replacements: LookupMap<TermId, ProjectionTemplate>,
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
    projection_fields: LookupMap<usize, usize>,
    /// The fused Screma has a single output — consumers alias its symbol
    /// directly rather than projecting a field.
    single_output: bool,
}

/// One output field of a fused `Screma`: what it computes and every site that
/// reads it. A builder produces a `Vec<FusedOutput>`; `lower_fused_screma` owns
/// the field-ordering invariant (lanes first, then accumulators) and the
/// projection routing, so no builder hand-computes a field index.
struct FusedOutput {
    ty: Type<TypeName>,
    kind: FusedOutputKind,
    /// Every site that reads THIS field. A field can serve many consumers — a
    /// filter's `length` count feeds every `length` call.
    consumers: Vec<OutputConsumer>,
}

enum FusedOutputKind {
    /// An elementwise output lane (becomes a `ScremaLane`).
    Lane(super::SoacBody),
    /// A reduce/scan output (becomes a `ScremaAccumulatorSpec`).
    Accumulator(super::ScremaAccumulatorSpec),
}

/// Where a fused output's value flows. The three variants mirror the three sinks
/// consumed by `apply_planned_screma_rewrite`.
enum OutputConsumer {
    /// A top-level let binding aliasing/projecting this field.
    Binding(usize),
    /// A use nested inside another term (e.g. `length(filt)` inside a later
    /// binding's RHS), rewritten in place via `term_replacements`.
    NestedTerm {
        term_id: TermId,
        ty: Type<TypeName>,
        span: Span,
    },
    /// The whole entry tail value, replaced via `tail_projection`.
    Tail {
        ty: Type<TypeName>,
        span: Span,
    },
}

struct ScremaProducer {
    lam: super::SoacBody,
    inputs: Vec<ArrayExpr>,
}

struct ScremaReduceConsumer {
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

fn soac_input_slots(semantics: &ArraySemantics) -> Vec<SoacInputSlot> {
    let params: Vec<(SymbolId, Type<TypeName>)> = match semantics {
        ArraySemantics::Elementwise { body, .. } => body.lam.params.clone(),
        ArraySemantics::Reduction { op, .. } | ArraySemantics::PrefixScan { op, .. } => {
            op.lam.params.iter().skip(1).cloned().collect()
        }
        ArraySemantics::Filter { pred, .. } => pred.lam.params.clone(),
        ArraySemantics::ScatterOp { lam, .. } => lam.lam.params.clone(),
        ArraySemantics::IndexedReduction { op, .. } => op.lam.params.iter().skip(1).cloned().collect(),
        _ => vec![],
    };

    semantics
        .input_exprs()
        .into_iter()
        .enumerate()
        .map(|(slot, input)| SoacInputSlot {
            slot,
            array: input.clone(),
            param: params.get(slot).cloned(),
        })
        .collect()
}

fn soac_output_slots(binding: &LetBinding, semantics: &ArraySemantics) -> Vec<SoacOutputSlot> {
    let kind = match semantics {
        ArraySemantics::Elementwise { .. } => SoacOutputKind::MapArray,
        ArraySemantics::Reduction { .. } => SoacOutputKind::ReduceScalar,
        ArraySemantics::PrefixScan { .. } => SoacOutputKind::ScanArray,
        ArraySemantics::Filter { .. } => SoacOutputKind::FilterArray,
        _ => SoacOutputKind::OtherArray,
    };
    // The derived kind must agree with the independent `produces_array`
    // classification — a mismatch means a new semantics variant slipped the
    // match above.
    debug_assert_eq!(kind.is_array(), semantics.produces_array());
    vec![SoacOutputSlot {
        slot: 0,
        value: binding.name,
        ty: binding.name_ty.clone(),
        kind,
    }]
}

fn named_input_slot_domain(slots: &[SoacInputSlot]) -> Option<Vec<NamedInputSlot>> {
    slots
        .iter()
        .map(|slot| {
            let (_, param_ty) = slot.param.as_ref()?;
            match &slot.array {
                ArrayExpr::Var(VarRef::Symbol(array_sym), array_ty) => Some(NamedInputSlot {
                    slot: slot.slot,
                    array_sym: *array_sym,
                    array_ty: array_ty.clone(),
                    param_ty: param_ty.clone(),
                }),
                _ => None,
            }
        })
        .collect()
}

fn find_direct_map_screma_group(region: &FusionRegion, bindings: &[LetBinding]) -> Option<ScremaGroup> {
    let mut start = 0;
    while start < bindings.len() {
        let Some(first_node) = region.node_for_binding(start) else {
            start += 1;
            continue;
        };
        let Some(first) = map_screma_producer(&bindings[start].rhs, MapProducerPolicy::Direct) else {
            start += 1;
            continue;
        };

        let Some(first_domain) = named_input_slot_domain(&first_node.inputs) else {
            start += 1;
            continue;
        };
        let mut maps = vec![ScremaMapGroupConsumer {
            binding_idx: start,
            lam: first.lam,
        }];
        let mut end = start + 1;
        while end < bindings.len() {
            let Some(next_node) = region.node_for_binding(end) else {
                break;
            };
            let Some(next) = map_screma_producer(&bindings[end].rhs, MapProducerPolicy::Direct) else {
                break;
            };
            let Some(next_domain) = named_input_slot_domain(&next_node.inputs) else {
                break;
            };
            if next_domain != first_domain {
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
            if !bindings[start..end].iter().any(|binding| mentions_any(&binding.rhs, &candidate_names)) {
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
        lanes,
        accumulators,
        inputs,
    }) = &consumer.kind
    else {
        return None;
    };
    if inputs.len() != 1 || inputs[0].as_named_ref() != Some(producer_sym) {
        return None;
    }
    if accumulators.iter().any(|acc| acc.step_lam.lam.params.len() != 2) {
        return None;
    }

    let composed_lams: Vec<super::SoacBody> = lanes
        .iter()
        .map(|lane| compose_soac_bodies(&producer.lam, &lane.lam, span, symbols, term_ids))
        .collect();
    let accumulators = accumulators
        .iter()
        .map(|acc| super::ScremaAccumulatorSpec {
            kind: acc.kind,
            step_lam: compose_soac_map_reduce(&producer.lam, &acc.step_lam, span, symbols, term_ids),
            reduce_op: acc.reduce_op.clone(),
            ne: acc.ne.clone(),
        })
        .collect();

    // Folding the producer in makes every lane consume all of the producer's
    // inputs (the composed lambdas take them as their leading args).
    let lanes = super::screma_lanes_all_inputs(composed_lams, producer.inputs.len());
    Some(Term {
        id: term_ids.next_id(),
        ty: consumer.ty.clone(),
        span: consumer.span,
        kind: TermKind::Soac(SoacOp::Screma {
            lanes,
            accumulators,
            inputs: producer.inputs.clone(),
        }),
    })
}

fn make_screma_term(
    result_fields: Vec<Type<TypeName>>,
    lanes: Vec<super::ScremaLane>,
    accumulators: Vec<super::ScremaAccumulatorSpec>,
    inputs: Vec<ArrayExpr>,
    span: Span,
    term_ids: &mut TermIdSource,
) -> Term {
    // A single-output Screma (one map lane xor one accumulator) is typed as that
    // output directly, so consumers reference it by name. Only genuine
    // multi-output fusions carry a tuple.
    let ty = if super::is_single_output_screma(&lanes, &accumulators) {
        result_fields[0].clone()
    } else {
        Type::Constructed(TypeName::Tuple(result_fields.len()), result_fields)
    };
    Term {
        id: term_ids.next_id(),
        ty,
        span,
        kind: TermKind::Soac(SoacOp::Screma {
            lanes,
            accumulators,
            inputs,
        }),
    }
}

/// Union a set of self-describing outputs into one multi-output `Screma` and
/// route each output's consumers to the right rewrite sink. This is the single
/// place the lanes-first/accumulators-second field ordering (the contract
/// `convert_soac_screma` asserts) is materialized — builders never compute a
/// field index by hand.
fn lower_fused_screma(
    insert_at: usize,
    skip_indices: Vec<usize>,
    symbol_name: &str,
    inputs: Vec<ArrayExpr>,
    span: Span,
    outputs: Vec<FusedOutput>,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> PlannedScremaRewrite {
    // Lanes occupy field indices `0..num_lanes`, accumulators
    // `num_lanes..num_lanes+num_accs`, each in emit order. Assigning indices up
    // front (rather than partitioning then re-deriving) keeps every output's
    // consumer list attached to its final field — the off-by-one class the old
    // per-builder `maps.len() + acc_idx` arithmetic was prone to.
    let num_lanes = outputs.iter().filter(|o| matches!(o.kind, FusedOutputKind::Lane(_))).count();
    let mut result_fields: Vec<Option<Type<TypeName>>> = vec![None; outputs.len()];
    let mut map_lams: Vec<super::SoacBody> = Vec::with_capacity(num_lanes);
    let mut accumulators: Vec<super::ScremaAccumulatorSpec> = Vec::with_capacity(outputs.len() - num_lanes);
    let mut distribution: Vec<(usize, Vec<OutputConsumer>)> = Vec::with_capacity(outputs.len());
    for output in outputs {
        let FusedOutput { ty, kind, consumers } = output;
        let field_idx = match kind {
            FusedOutputKind::Lane(body) => {
                let idx = map_lams.len();
                map_lams.push(body);
                idx
            }
            FusedOutputKind::Accumulator(spec) => {
                let idx = num_lanes + accumulators.len();
                accumulators.push(spec);
                idx
            }
        };
        result_fields[field_idx] = Some(ty);
        distribution.push((field_idx, consumers));
    }
    let result_fields: Vec<Type<TypeName>> =
        result_fields.into_iter().map(|ty| ty.expect("every field index assigned exactly once")).collect();

    let tuple_sym = symbols.alloc(symbol_name.to_string());
    // Every lane consumes all inputs (the fusion paths read the same domain).
    let lanes = super::screma_lanes_all_inputs(map_lams, inputs.len());
    let single_output = super::is_single_output_screma(&lanes, &accumulators);
    debug_assert_eq!(result_fields.len(), lanes.len() + accumulators.len());
    let rhs = make_screma_term(result_fields, lanes, accumulators, inputs, span, term_ids);
    let tuple_ty = rhs.ty.clone();
    let fused_binding = LetBinding {
        name: tuple_sym,
        name_ty: tuple_ty.clone(),
        rhs,
        span,
    };

    let mut projection_fields = LookupMap::new();
    let mut term_replacements = LookupMap::new();
    let mut tail_projection = None;
    for (field_idx, consumers) in distribution {
        for consumer in consumers {
            match consumer {
                OutputConsumer::Binding(idx) => {
                    projection_fields.insert(idx, field_idx);
                }
                OutputConsumer::NestedTerm { term_id, ty, span } => {
                    term_replacements.insert(
                        term_id,
                        ProjectionTemplate {
                            tuple_sym,
                            tuple_ty: tuple_ty.clone(),
                            field_idx,
                            ty,
                            span,
                            single_output,
                        },
                    );
                }
                OutputConsumer::Tail { ty, span } => {
                    tail_projection = Some(ProjectionTemplate {
                        tuple_sym,
                        tuple_ty: tuple_ty.clone(),
                        field_idx,
                        ty,
                        span,
                        single_output,
                    });
                }
            }
        }
    }

    PlannedScremaRewrite {
        rewrite: ScremaRewrite {
            insert_at,
            skip_indices,
            tuple_sym,
            tuple_ty,
            fused_binding,
            projection_fields,
            single_output,
        },
        term_replacements,
        tail_projection,
    }
}

fn tuple_projection_binding(
    original: &LetBinding,
    tuple_sym: SymbolId,
    tuple_ty: &Type<TypeName>,
    proj_idx: usize,
    single_output: bool,
    term_ids: &mut TermIdSource,
) -> LetBinding {
    // Single-output Screma: its result IS the consumer's value, so alias its
    // symbol directly rather than projecting. (`single_output` comes from the
    // rewrite's shape — never from `tuple_ty`, which is a tuple when the sole
    // output is itself a tuple value.)
    let rhs_kind = if single_output {
        TermKind::Var(VarRef::Symbol(tuple_sym))
    } else {
        TermKind::TupleProj {
            tuple: Box::new(Term {
                id: term_ids.next_id(),
                ty: tuple_ty.clone(),
                span: original.span,
                kind: TermKind::Var(VarRef::Symbol(tuple_sym)),
            }),
            idx: proj_idx,
        }
    };
    LetBinding {
        name: original.name,
        name_ty: original.name_ty.clone(),
        rhs: Term {
            id: term_ids.next_id(),
            ty: original.name_ty.clone(),
            span: original.span,
            kind: rhs_kind,
        },
        span: original.span,
    }
}

fn find_fusion_plan(
    bindings: &[LetBinding],
    tail: &Term,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<FusionPlan> {
    let region = FusionRegion::new(bindings, tail);
    // TODO(fusion-union): each builder below independently emits a
    // `Vec<FusedOutput>` for a single producer/group and lowers it on its own.
    // Because `OutputConsumer` already supports one field feeding many sites and
    // a node mixing lane + accumulator outputs, the next step is to let several
    // builders contribute to a *shared* `Vec<FusedOutput>` (e.g. a filter group
    // and a map group over the same input) before one `lower_fused_screma` call
    // — turning "first matching builder wins" into a real multi-output merge.
    if let Some(plan) = find_direct_horizontal_map_plan(&region, bindings, symbols, term_ids) {
        return Some(plan);
    }

    for producer in &region.producers {
        if !producer.has_array_output() {
            continue;
        }
        let uses = region.uses_for(producer);
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

fn classify_uses_for_producer(
    producer: &ProducerInfo,
    bindings: &[LetBinding],
    tail: &Term,
) -> Vec<UseSite> {
    let mut uses = Vec::new();
    for (idx, binding) in bindings.iter().enumerate().skip(producer.binding_idx + 1) {
        uses.extend(classify_uses_in_owner(
            &binding.rhs,
            UseOwner::Binding(idx),
            producer.output_symbol(),
        ));
    }
    uses.extend(classify_uses_in_owner(
        tail,
        UseOwner::Tail,
        producer.output_symbol(),
    ));
    uses
}

fn classify_uses_in_owner(term: &Term, owner: UseOwner, producer_sym: SymbolId) -> Vec<UseSite> {
    // Alias-aware classification: when inlining wraps a call body in
    // `let X = Var(producer) in body`, the body uses `X` as an alias for
    // the producer. Track aliases as we descend so the recognizers see
    // through them.
    let mut producers: LookupSet<SymbolId> = LookupSet::new();
    producers.insert(producer_sym);
    let mut uses = Vec::new();
    collect_classified_uses(term, owner, &producers, true, &mut uses);

    let recognized_refs = uses.len();
    let raw_refs = aliased_var_ref_count(term, &producers);
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
    producers: &LookupSet<SymbolId>,
    allow_screma_input: bool,
    uses: &mut Vec<UseSite>,
) {
    if is_length_call_of(term, producers) {
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

    let semantics = classify_term(term);
    let input_slots = semantic_input_slots(&semantics, producers);
    if !input_slots.is_empty() {
        for input in input_slots {
            uses.push(UseSite {
                owner,
                kind: UseKind::SoacInput {
                    term_id: term.id,
                    input_index: input.slot,
                    input,
                    semantics: semantics.clone(),
                    ty: term.ty.clone(),
                    span: term.span,
                },
            });
        }
        return;
    }

    if allow_screma_input {
        let screma_inputs = top_screma_input_slots(term, producers);
        if !screma_inputs.is_empty() {
            for (input_index, input) in screma_inputs {
                uses.push(UseSite {
                    owner,
                    kind: UseKind::ScremaInput { input_index, input },
                });
            }
            return;
        }
    }

    match &term.kind {
        TermKind::Lambda(lam) if lam.params.iter().any(|(sym, _)| producers.contains(sym)) => {}
        TermKind::Let { name, rhs, body, .. } => {
            // Aliasing-let `let X = Var(producer-or-alias) in body`: the
            // rhs is structural — don't classify it as a consumer use.
            // Extend the alias set with `X` and recurse into body.
            let is_alias_binding = matches!(
                &rhs.kind,
                TermKind::Var(VarRef::Symbol(s)) if producers.contains(s)
            );
            if is_alias_binding {
                let mut extended = producers.clone();
                extended.insert(*name);
                collect_classified_uses(body, owner, &extended, false, uses);
            } else {
                collect_classified_uses(rhs, owner, producers, false, uses);
                if !producers.contains(name) {
                    collect_classified_uses(body, owner, producers, false, uses);
                }
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
            collect_classified_uses(init, owner, producers, false, uses);
            for (_, _, extraction) in init_bindings {
                collect_classified_uses(extraction, owner, producers, false, uses);
            }
            match kind {
                super::LoopKind::For { var, iter, .. } => {
                    collect_classified_uses(iter, owner, producers, false, uses);
                    if !producers.contains(loop_var) && !producers.contains(var) {
                        collect_classified_uses(body, owner, producers, false, uses);
                    }
                }
                super::LoopKind::ForRange { var, bound, .. } => {
                    collect_classified_uses(bound, owner, producers, false, uses);
                    if !producers.contains(loop_var) && !producers.contains(var) {
                        collect_classified_uses(body, owner, producers, false, uses);
                    }
                }
                super::LoopKind::While { cond } => {
                    collect_classified_uses(cond, owner, producers, false, uses);
                    if !producers.contains(loop_var) {
                        collect_classified_uses(body, owner, producers, false, uses);
                    }
                }
            }
        }
        _ => term.for_each_child(&mut |child| {
            collect_classified_uses(child, owner, producers, false, uses);
        }),
    }
}

fn semantic_input_slots(semantics: &ArraySemantics, producers: &LookupSet<SymbolId>) -> Vec<SoacInputSlot> {
    semantic_input_slots_from_slots(&soac_input_slots(semantics), producers)
}

fn semantic_input_slots_from_slots(
    slots: &[SoacInputSlot],
    producers: &LookupSet<SymbolId>,
) -> Vec<SoacInputSlot> {
    slots
        .iter()
        .filter_map(|slot| {
            slot.param.as_ref()?;
            slot.array.as_named_ref().filter(|s| producers.contains(s)).map(|_| slot.clone())
        })
        .collect()
}

fn top_screma_input_slots(term: &Term, producers: &LookupSet<SymbolId>) -> Vec<(usize, ArrayExpr)> {
    let TermKind::Soac(SoacOp::Screma { inputs, .. }) = &term.kind else {
        return vec![];
    };
    inputs
        .iter()
        .enumerate()
        .filter_map(|(idx, input)| {
            input.as_named_ref().filter(|s| producers.contains(s)).map(|_| (idx, input.clone()))
        })
        .collect()
}

fn find_direct_horizontal_map_plan(
    region: &FusionRegion,
    bindings: &[LetBinding],
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<FusionPlan> {
    let group = find_direct_map_screma_group(region, bindings)?;
    let outputs: Vec<FusedOutput> = group
        .maps
        .iter()
        .map(|consumer| FusedOutput {
            ty: bindings[consumer.binding_idx].name_ty.clone(),
            kind: FusedOutputKind::Lane(consumer.lam.clone()),
            consumers: vec![OutputConsumer::Binding(consumer.binding_idx)],
        })
        .collect();
    Some(FusionPlan::Screma(lower_fused_screma(
        group.insert_at,
        group.skip_indices,
        group.symbol_name,
        group.inputs,
        group.span,
        outputs,
        symbols,
        term_ids,
    )))
}

fn find_map_group_plan(
    producer: &ProducerInfo,
    uses: &[UseSite],
    bindings: &[LetBinding],
    tail: &Term,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<FusionPlan> {
    // Self-documenting declines: a `log::debug!` trace of *why* this builder
    // turns an edge down, so a "didn't fuse" can be diagnosed from the reason
    // rather than inferred from IR dumps. Enable with RUST_LOG=debug.
    macro_rules! reject {
        ($($arg:tt)*) => {{
            log::debug!("fuse map-group decline (producer {:?}): {}", producer.symbol, format!($($arg)*));
            return None;
        }};
    }

    let ArraySemantics::Elementwise {
        inputs: producer_inputs,
        body: producer_body,
    } = &producer.semantics
    else {
        reject!("producer is not an elementwise map");
    };
    // Horizontal grouping only: ≥2 consumers fused into one Screma exposed via
    // tail projections. Single-consumer *vertical* edges (`let m = map(..) in
    // scan(.., m)`) are handled by `find_vertical_accumulator_plan`, which has
    // none of this projection / tail-value machinery.
    if producer.inputs.len() != producer_inputs.len() {
        reject!("producer input slots are not aligned with its elementwise inputs");
    }
    if uses.len() < 2 {
        reject!("only {} consumer(s) — horizontal grouping needs ≥2", uses.len());
    }
    let Some(binding_uses) = binding_soac_uses(uses, bindings) else {
        reject!("not every use is a top-level binding SOAC input");
    };

    // Each consumer contributes one output reading THIS binding (field ordering
    // — lanes before accumulators — is `lower_fused_screma`'s job, so consumers
    // are emitted in encounter order regardless of kind).
    let mut outputs: Vec<FusedOutput> = Vec::new();
    let mut consumer_indices = Vec::new();
    for use_site in binding_uses {
        let binding_idx = use_site.binding_idx;
        // TODO(fusion-union): `SoacInputSlot.slot` is carried end-to-end, so
        // composing a producer into a consumer's input slot ≠ 0 is now a
        // localized change here (route the producer body to the right param),
        // not a representational one.
        if use_site.input.slot != 0 {
            reject!("producer feeds input index {}, not 0", use_site.input.slot);
        }
        let kind = match &use_site.semantics {
            ArraySemantics::Elementwise { inputs, body } if inputs.len() == 1 => FusedOutputKind::Lane(
                compose_soac_bodies(producer_body, body, bindings[binding_idx].span, symbols, term_ids),
            ),
            ArraySemantics::Reduction {
                op, reduce_op, init, ..
            } if op.lam.params.len() == 2 => FusedOutputKind::Accumulator(super::ScremaAccumulatorSpec {
                kind: super::ScremaAccumulator::Reduce,
                step_lam: compose_soac_map_reduce(
                    producer_body,
                    op,
                    bindings[binding_idx].span,
                    symbols,
                    term_ids,
                ),
                reduce_op: reduce_op.clone(),
                ne: init.clone(),
            }),
            ArraySemantics::PrefixScan { .. } => {
                let scan_sym = bindings[binding_idx].name;
                if bindings
                    .iter()
                    .skip(binding_idx + 1)
                    .any(|binding| mentions_any(&binding.rhs, &[scan_sym]))
                    || !symbol_uses_are_direct_tail_values(tail, scan_sym)
                {
                    reject!("scan result is re-read by a later binding or not a direct tail value");
                }
                let Some(accumulator) =
                    screma_consumer_accumulator(&bindings[binding_idx].rhs, producer.symbol)
                else {
                    reject!("scan accumulator unsupported (input not producer sym, or arity mismatch)");
                };
                FusedOutputKind::Accumulator(super::ScremaAccumulatorSpec {
                    kind: accumulator.accumulator,
                    step_lam: compose_soac_map_reduce(
                        producer_body,
                        &accumulator.op,
                        bindings[binding_idx].span,
                        symbols,
                        term_ids,
                    ),
                    reduce_op: accumulator.reduce_op,
                    ne: accumulator.ne,
                })
            }
            _ => reject!("consumer semantics unsupported for grouping (not map/reduce/scan)"),
        };
        outputs.push(FusedOutput {
            ty: bindings[binding_idx].name_ty.clone(),
            kind,
            consumers: vec![OutputConsumer::Binding(binding_idx)],
        });
        consumer_indices.push(binding_idx);
    }
    if consumer_indices_are_dependent(bindings, &consumer_indices) {
        reject!("grouped consumers are mutually dependent");
    }

    Some(FusionPlan::Screma(lower_fused_screma(
        producer.binding_idx,
        vec![producer.binding_idx],
        "_screma",
        producer_inputs.clone(),
        bindings[producer.binding_idx].span,
        outputs,
        symbols,
        term_ids,
    )))
}

fn find_filter_plan(
    producer: &ProducerInfo,
    uses: &[UseSite],
    bindings: &[LetBinding],
    tail: &Term,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<FusionPlan> {
    let ArraySemantics::Filter { map_lam, input, pred } = &producer.semantics else {
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
                let ArraySemantics::Reduction {
                    op, reduce_op, init, ..
                } = semantics
                else {
                    return None;
                };
                if op.lam.params.len() != 2 {
                    return None;
                }
                let step_lam = filtered_reduce_step(map_lam.as_ref(), pred, op, *span, symbols, term_ids)?;
                reductions.push((
                    use_site.owner,
                    *term_id,
                    super::ScremaAccumulatorSpec {
                        kind: super::ScremaAccumulator::Reduce,
                        step_lam,
                        reduce_op: reduce_op.clone(),
                        ne: init.clone(),
                    },
                    ty.clone(),
                    *span,
                ));
            }
            UseKind::ScremaInput { .. } | UseKind::Escape => return None,
        }
    }
    if reductions.is_empty() && lengths.is_empty() {
        return None;
    }

    // A reduction's whole-tail use projects via `tail_projection`; nested and
    // non-direct-binding uses rewrite in place. (Lengths never use the tail
    // sink — see below.)
    let reduction_consumer = |owner: UseOwner, term_id: TermId, ty: Type<TypeName>, span: Span| match owner
    {
        UseOwner::Binding(idx) if bindings[idx].rhs.id == term_id => OutputConsumer::Binding(idx),
        UseOwner::Tail if tail.id == term_id => OutputConsumer::Tail { ty, span },
        _ => OutputConsumer::NestedTerm { term_id, ty, span },
    };

    // Reductions first, then the (single) count — so the count is the last
    // accumulator field. `lower_fused_screma` partitions lanes before
    // accumulators but preserves order *within* accumulators, so this emit order
    // is what fixes the count's field index; keep it.
    let mut outputs: Vec<FusedOutput> = Vec::new();
    for (owner, term_id, acc, ty, span) in reductions {
        outputs.push(FusedOutput {
            ty: ty.clone(),
            kind: FusedOutputKind::Accumulator(acc),
            consumers: vec![reduction_consumer(owner, term_id, ty, span)],
        });
    }

    if let Some((_, _, count_ty, count_span)) = lengths.first().cloned() {
        // One shared count field feeds every `length` call. A whole-tail length
        // routes through `NestedTerm`, not `Tail` — `replace_projection_terms`
        // rewrites it inside the tail in place (matching prior behavior).
        let count_consumers: Vec<OutputConsumer> = lengths
            .into_iter()
            .map(|(owner, term_id, ty, span)| match owner {
                UseOwner::Binding(idx) if bindings[idx].rhs.id == term_id => OutputConsumer::Binding(idx),
                _ => OutputConsumer::NestedTerm { term_id, ty, span },
            })
            .collect();
        outputs.push(FusedOutput {
            ty: count_ty.clone(),
            kind: FusedOutputKind::Accumulator(count_accumulator(
                map_lam.as_ref(),
                pred,
                count_ty,
                count_span,
                symbols,
                term_ids,
            )?),
            consumers: count_consumers,
        });
    }

    Some(FusionPlan::Screma(lower_fused_screma(
        producer.binding_idx,
        vec![producer.binding_idx],
        "_filtered_screma",
        vec![input.clone()],
        bindings[producer.binding_idx].span,
        outputs,
        symbols,
        term_ids,
    )))
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
        UseKind::ScremaInput { input_index, input } => {
            let UseOwner::Binding(consumer_idx) = use_site.owner else {
                return None;
            };
            if *input_index != 0 || input.as_named_ref() != Some(producer.output_symbol()) {
                return None;
            }
            let map_producer =
                map_screma_producer(&bindings[producer.binding_idx].rhs, MapProducerPolicy::Producer)?;
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
            ..
        } => {
            // The raw consumer term (a top-level binding rhs, or the tail) and
            // where its fused value flows. Pairwise fusion only rewrites these two
            // sinks — never a nested occurrence.
            let consumer_term = match use_site.owner {
                UseOwner::Binding(idx) if *term_id == bindings[idx].rhs.id => &bindings[idx].rhs,
                UseOwner::Tail if *term_id == tail.id => tail,
                _ => return None,
            };
            let consumer_sink = match use_site.owner {
                UseOwner::Binding(idx) => OutputConsumer::Binding(idx),
                UseOwner::Tail => OutputConsumer::Tail {
                    ty: ty.clone(),
                    span: *span,
                },
            };
            let recipe = can_fuse(&producer.semantics, semantics)?;
            if recipe == FusionRecipe::ComposeElementwise {
                if let ArraySemantics::Elementwise { inputs, .. } = semantics {
                    if inputs.len() != 1 {
                        return None;
                    }
                }
            }

            // `map∘map`, `map→reduce`, and `map→scan` are all expressible as a
            // single-output `Screma`, so fold the elementwise producer into the
            // lone consumer and lower through the union path. Recipes whose result
            // is a non-`Screma` SOAC (`MapIntoFilter` → `Filter`, `MapIntoScatter`
            // → `Scatter`) fall through to direct composition below.
            if let ArraySemantics::Elementwise {
                inputs: producer_inputs,
                body: producer_body,
            } = &producer.semantics
            {
                if let Some(output) = pairwise_screma_output(
                    producer_body,
                    semantics,
                    consumer_term,
                    producer.output_symbol(),
                    consumer_sink,
                    ty.clone(),
                    *span,
                    symbols,
                    term_ids,
                ) {
                    return Some(FusionPlan::Screma(lower_fused_screma(
                        producer.binding_idx,
                        vec![producer.binding_idx],
                        "_screma",
                        producer_inputs.clone(),
                        bindings[producer.binding_idx].span,
                        vec![output],
                        symbols,
                        term_ids,
                    )));
                }
            }

            let fused_rhs = build_fused_from_semantics(
                &producer.semantics,
                semantics,
                &recipe,
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

/// Build the single fused output that folds an elementwise `producer_body` into
/// its lone `consumer` — for the recipes the union `Screma` representation can
/// express: `map∘map` (a `Lane`), `map→reduce` / `map→scan` (an `Accumulator`).
/// Returns `None` for a consumer whose fused result is a non-`Screma` SOAC
/// (`Filter`, `Scatter`), which [`build_fused_from_semantics`] lowers directly.
#[allow(clippy::too_many_arguments)]
fn pairwise_screma_output(
    producer_body: &super::SoacBody,
    consumer_semantics: &ArraySemantics,
    consumer_term: &Term,
    producer_sym: SymbolId,
    consumer: OutputConsumer,
    output_ty: Type<TypeName>,
    span: Span,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<FusedOutput> {
    let kind = match consumer_semantics {
        ArraySemantics::Elementwise { inputs, .. } if inputs.len() == 1 => {
            let ArraySemantics::Elementwise { body: cons_body, .. } = consumer_semantics else {
                unreachable!()
            };
            FusedOutputKind::Lane(compose_soac_bodies(
                producer_body,
                cons_body,
                span,
                symbols,
                term_ids,
            ))
        }
        ArraySemantics::Reduction { .. } | ArraySemantics::PrefixScan { .. } => {
            // Read the accumulator (reduce vs scan, op/reduce_op/ne) off the raw
            // consumer term, then compose the producer map into its per-element
            // step — the same construction the horizontal map-group builder uses.
            let acc = screma_consumer_accumulator(consumer_term, producer_sym)?;
            FusedOutputKind::Accumulator(super::ScremaAccumulatorSpec {
                kind: acc.accumulator,
                step_lam: compose_soac_map_reduce(producer_body, &acc.op, span, symbols, term_ids),
                reduce_op: acc.reduce_op,
                ne: acc.ne,
            })
        }
        _ => return None,
    };
    Some(FusedOutput {
        ty: output_ty,
        kind,
        consumers: vec![consumer],
    })
}

fn apply_fusion_plan(mut chain: LetChain, plan: FusionPlan, term_ids: &mut TermIdSource) -> Option<Term> {
    match plan {
        FusionPlan::ReplaceConsumer {
            producer_idx,
            consumer_idx,
            fused_rhs,
        } => {
            match consumer_idx {
                Some(idx) => {
                    let rhs_ty = chain.binding(idx)?.rhs.ty.clone();
                    chain.replace_binding_rhs(
                        idx,
                        Term {
                            ty: rhs_ty,
                            ..fused_rhs
                        },
                    );
                }
                None => {
                    let tail_ty = chain.tail().ty.clone();
                    chain.replace_tail(Term {
                        ty: tail_ty,
                        ..fused_rhs
                    });
                }
            }
            chain.remove_binding(producer_idx);
            Some(chain.into_term(term_ids))
        }
        FusionPlan::Screma(plan) => {
            let chain = apply_planned_screma_rewrite(chain, plan, term_ids);
            Some(chain.into_term(term_ids))
        }
    }
}

fn apply_planned_screma_rewrite(
    mut chain: LetChain,
    plan: PlannedScremaRewrite,
    term_ids: &mut TermIdSource,
) -> LetChain {
    let PlannedScremaRewrite {
        rewrite,
        term_replacements,
        tail_projection,
    } = plan;

    // Single-output Screma feeding exactly one `let` (no nested-term or tail
    // consumers): bind the Screma straight to that consumer's symbol — no
    // `tuple_sym`, no `Var(tuple_sym)` alias. Downstream passes (gather
    // residency) then see the producer as a bare `Soac(Screma)` under its own
    // name, not hidden behind an alias of an intermediate tuple symbol.
    if rewrite.single_output
        && term_replacements.is_empty()
        && tail_projection.is_none()
        && rewrite.projection_fields.len() == 1
    {
        let consumer_idx = *rewrite.projection_fields.keys().next().expect("one consumer");
        chain.replace_binding_rhs(consumer_idx, rewrite.fused_binding.rhs);
        chain.remove_bindings(&rewrite.skip_indices);
        return chain;
    }

    // The producer symbol(s) being replaced — their bindings disappear,
    // so any leftover `Var(producer)` references (e.g. structural alias
    // lets `let X = Var(filt) in body` that the classifier saw through)
    // need to be rebound to `Var(tuple_sym)` to keep the term well-formed.
    let producer_syms: Vec<SymbolId> =
        rewrite.skip_indices.iter().map(|&i| chain.bindings()[i].name).collect();

    for idx in 0..chain.bindings().len() {
        if rewrite.skip_indices.contains(&idx) {
            continue;
        }
        if let Some(&proj_idx) = rewrite.projection_fields.get(&idx) {
            let projection = tuple_projection_binding(
                chain.binding(idx).expect("projection binding index in range"),
                rewrite.tuple_sym,
                &rewrite.tuple_ty,
                proj_idx,
                rewrite.single_output,
                term_ids,
            );
            chain.replace_binding(idx, projection);
        } else {
            chain.rewrite_binding_rhs(idx, |rhs| {
                let mut rhs =
                    rewrite_terms_by_id(rhs, &term_replacements, term_ids, &mut |projection, ids| {
                        projection_term(projection, ids)
                    });
                for &p in &producer_syms {
                    rhs = substitute_sym(rhs, p, rewrite.tuple_sym, term_ids);
                }
                rhs
            });
        }
    }

    if let Some(projection) = tail_projection {
        chain.replace_tail(projection_term(&projection, term_ids));
    } else {
        chain.rewrite_tail(|tail| {
            rewrite_terms_by_id(tail, &term_replacements, term_ids, &mut |projection, ids| {
                projection_term(projection, ids)
            })
        });
    }

    chain.remove_bindings(&rewrite.skip_indices);
    chain.insert_binding_at_original_index(rewrite.insert_at, &rewrite.skip_indices, rewrite.fused_binding);
    chain
}

fn projection_term(projection: &ProjectionTemplate, term_ids: &mut TermIdSource) -> Term {
    // Single-output Screma: no tuple to project — reference the Screma symbol
    // directly (its result type equals `projection.ty`).
    if projection.single_output {
        return Term {
            id: term_ids.next_id(),
            ty: projection.ty.clone(),
            span: projection.span,
            kind: TermKind::Var(VarRef::Symbol(projection.tuple_sym)),
        };
    }
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

fn rewrite_terms_by_id<T, F>(
    term: Term,
    replacements: &LookupMap<TermId, T>,
    term_ids: &mut TermIdSource,
    make_replacement: &mut F,
) -> Term
where
    F: FnMut(&T, &mut TermIdSource) -> Term,
{
    if let Some(replacement) = replacements.get(&term.id) {
        return make_replacement(replacement, term_ids);
    }
    let mut rewritten = term
        .map_children(&mut |child| rewrite_terms_by_id(child, replacements, term_ids, make_replacement));
    rewritten.id = term_ids.next_id();
    rewritten
}

/// Convert a `replacement` term being substituted into a SOAC input position
/// into an ANF input atom: a bare name stays a name; an array expression is
/// itself an atom; a tuple-of-arrays (the SoA form of a `zip`) becomes a `Zip`
/// of atoms. A producer term has no atomic form for an input position.
fn is_length_call_of(term: &Term, producers: &LookupSet<SymbolId>) -> bool {
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
        TermKind::Var(VarRef::Symbol(sym)) if producers.contains(sym)
    )
}

/// Count raw `Var(p)` references for any `p` in `producers`, ignoring the
/// rhs of aliasing-let bindings `let X = Var(p) in body` (those are
/// structural pass-throughs, not consumer uses; `X` joins the alias set
/// when descending into `body`). Mirrors `collect_classified_uses`'s
/// scoping so `raw_refs` and `recognized_refs` agree.
fn aliased_var_ref_count(term: &Term, producers: &LookupSet<SymbolId>) -> usize {
    match &term.kind {
        TermKind::Var(VarRef::Symbol(s)) if producers.contains(s) => 1,
        TermKind::Lambda(lam) if lam.params.iter().any(|(p, _)| producers.contains(p)) => 0,
        TermKind::Let { name, rhs, body, .. } => {
            let is_alias_binding = matches!(
                &rhs.kind,
                TermKind::Var(VarRef::Symbol(s)) if producers.contains(s)
            );
            if is_alias_binding {
                let mut extended = producers.clone();
                extended.insert(*name);
                aliased_var_ref_count(body, &extended)
            } else {
                let rhs_count = aliased_var_ref_count(rhs, producers);
                let body_count =
                    if producers.contains(name) { 0 } else { aliased_var_ref_count(body, producers) };
                rhs_count + body_count
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
            let mut total = aliased_var_ref_count(init, producers);
            for (_, _, extraction) in init_bindings {
                total += aliased_var_ref_count(extraction, producers);
            }
            match kind {
                super::LoopKind::For { var, iter, .. } => {
                    total += aliased_var_ref_count(iter, producers);
                    if !producers.contains(loop_var) && !producers.contains(var) {
                        total += aliased_var_ref_count(body, producers);
                    }
                }
                super::LoopKind::ForRange { var, bound, .. } => {
                    total += aliased_var_ref_count(bound, producers);
                    if !producers.contains(loop_var) && !producers.contains(var) {
                        total += aliased_var_ref_count(body, producers);
                    }
                }
                super::LoopKind::While { cond } => {
                    total += aliased_var_ref_count(cond, producers);
                    if !producers.contains(loop_var) {
                        total += aliased_var_ref_count(body, producers);
                    }
                }
            }
            total
        }
        _ => {
            let mut total = 0;
            term.for_each_child(&mut |c| total += aliased_var_ref_count(c, producers));
            total
        }
    }
}

fn consumer_indices_are_dependent(bindings: &[LetBinding], consumer_indices: &[usize]) -> bool {
    for (pos, idx) in consumer_indices.iter().enumerate() {
        let prior: Vec<_> = consumer_indices[..pos].iter().map(|i| bindings[*i].name).collect();
        if !prior.is_empty() && mentions_any(&bindings[*idx].rhs, &prior) {
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
        _ => !mentions_any(term, &[sym]),
    }
}

fn filtered_reduce_step(
    map_lam: Option<&super::SoacBody>,
    pred: &super::SoacBody,
    op: &super::SoacBody,
    span: Span,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<super::SoacBody> {
    if pred.lam.params.len() != 1 || op.lam.params.len() != 2 {
        return None;
    }
    // A fused producer map folds in by composing `f` into the predicate and the
    // op, so the masked step tests/accumulates `f(x)`. `f` is pure, so applying
    // it in both is redundant but correct.
    let (pred_body, op_body) = match map_lam {
        Some(f) => (
            compose_soac_bodies(f, pred, span, symbols, term_ids),
            compose_soac_map_reduce(f, op, span, symbols, term_ids),
        ),
        None => (pred.clone(), op.clone()),
    };
    let pred_lam = pred_body.lam.clone();
    let op_lam = op_body.lam.clone();
    let acc_param = op_lam.params[0].clone();
    let elem_param = pred_lam.params[0].clone();
    let op_elem = op_lam.params[1].0;
    let then_branch = substitute_sym(*op_lam.body.clone(), op_elem, elem_param.0, term_ids);
    let else_branch = Term {
        id: term_ids.next_id(),
        ty: op_lam.ret_ty.clone(),
        span,
        kind: TermKind::Var(VarRef::Symbol(acc_param.0)),
    };
    Some(super::SoacBody {
        lam: Lambda {
            params: vec![acc_param, elem_param],
            body: Box::new(Term {
                id: term_ids.next_id(),
                ty: op_lam.ret_ty.clone(),
                span,
                kind: TermKind::If {
                    cond: pred_lam.body.clone(),
                    then_branch: Box::new(then_branch),
                    else_branch: Box::new(else_branch),
                },
            }),
            ret_ty: op_lam.ret_ty.clone(),
        },
        captures: merged_captures(&pred_body.captures, &op_body.captures),
    })
}

fn count_accumulator(
    map_lam: Option<&super::SoacBody>,
    pred: &super::SoacBody,
    count_ty: Type<TypeName>,
    span: Span,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<super::ScremaAccumulatorSpec> {
    if pred.lam.params.len() != 1 {
        return None;
    }
    // A fused producer map tests `pred(f(x))`, so count over the composed
    // predicate; the surviving count is unchanged by what value is kept.
    let pred_body = match map_lam {
        Some(f) => compose_soac_bodies(f, pred, span, symbols, term_ids),
        None => pred.clone(),
    };
    let pred_lam = pred_body.lam.clone();
    let acc_sym = symbols.alloc("_count".to_string());
    let acc_param = (acc_sym, count_ty.clone());
    let elem_param = pred_lam.params[0].clone();
    let plus_one = add_terms(
        Term::var(acc_sym, count_ty.clone(), span, term_ids.next_id()),
        Term::int_lit("1", count_ty.clone(), span, term_ids.next_id()),
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
                cond: pred_lam.body.clone(),
                then_branch: Box::new(plus_one),
                else_branch: Box::new(Term::var(acc_sym, count_ty.clone(), span, term_ids.next_id())),
            },
        }),
        ret_ty: count_ty.clone(),
    };

    let rhs_sym = symbols.alloc("_count_rhs".to_string());
    let reduce_op = Lambda {
        params: vec![acc_param, (rhs_sym, count_ty.clone())],
        body: Box::new(add_terms(
            Term::var(acc_sym, count_ty.clone(), span, term_ids.next_id()),
            Term::var(rhs_sym, count_ty.clone(), span, term_ids.next_id()),
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
            captures: pred_body.captures,
        },
        reduce_op: super::SoacBody {
            lam: reduce_op,
            captures: vec![],
        },
        ne: Box::new(Term::int_lit("0", count_ty, span, term_ids.next_id())),
    })
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

/// Bottom-up: recurse into children, applying graph-driven fusion at each
/// sub-expression that contains a Let chain with SOAC producers/consumers.
fn fuse_term(term: Term, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> (Term, bool) {
    let mut changed = false;
    let term = term.map_children(&mut |child| {
        let (child, did_fuse) = fuse_term(child, symbols, term_ids);
        changed |= did_fuse;
        child
    });

    if matches!(term.kind, TermKind::Let { .. }) {
        let (fused, did_fuse) = fuse_def_body(term, symbols, term_ids);
        return (fused, changed || did_fuse);
    }

    (term, changed)
}

/// Fuse within a body term. Returns the new body and whether any fusion happened.
fn fuse_def_body(body: Term, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> (Term, bool) {
    let (params, inner) = extract_lambda_params(&body);
    let chain = LetChain::from_term(inner);
    if chain.is_empty() {
        return (body, false);
    }

    let Some(plan) = find_fusion_plan(chain.bindings(), chain.tail(), symbols, term_ids) else {
        return (body, false);
    };

    let Some(fused_inner) = apply_fusion_plan(chain, plan, term_ids) else {
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
    recipe: &FusionRecipe,
    input_index: usize,
    consumer_ty: Type<TypeName>,
    span: Span,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    // Only the two recipes whose fused result is a non-`Screma` SOAC survive
    // here — `map→filter` (→ `Filter`) and `map→scatter` (→ `Scatter`). The
    // `Screma`-expressible recipes (`map∘map`, `map→reduce`, `map→scan`) are
    // lowered through `lower_fused_screma` by `pairwise_screma_output`.
    let (prod_body, input_exprs) = match producer {
        ArraySemantics::Elementwise { body, inputs } => (body, inputs.clone()),
        _ => return None,
    };

    match (recipe, consumer) {
        (
            FusionRecipe::MapIntoFilter,
            ArraySemantics::Filter {
                map_lam: None, pred, ..
            },
        ) => {
            // `filter(p, map(f, xs))` → a `Filter` carrying `f` as its `map_lam`,
            // reading the producer's own input `xs`. Per element the filter
            // computes `v = f(x)`, tests `p(v)`, keeps `v` — no intermediate array.
            if input_exprs.len() != 1 {
                return None;
            }
            Some(Term {
                id: term_ids.next_id(),
                ty: consumer_ty,
                span,
                kind: TermKind::Soac(SoacOp::Filter {
                    map_lam: Some(prod_body.clone()),
                    pred: pred.clone(),
                    input: input_exprs[0].clone(),
                    destination: SoacDestination::Fresh,
                }),
            })
        }

        (
            FusionRecipe::MapIntoScatter,
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
                prod_body.lam.clone(),
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
                        captures: merged_captures(&prod_body.captures, &env.captures),
                    },
                    inputs: new_inputs,
                }),
            })
        }

        _ => None,
    }
}

// =============================================================================
// Lambda composition
// =============================================================================

fn merged_captures(
    first: &[(SymbolId, Type<TypeName>, Term)],
    second: &[(SymbolId, Type<TypeName>, Term)],
) -> Vec<(SymbolId, Type<TypeName>, Term)> {
    let mut captures = Vec::with_capacity(first.len() + second.len());
    captures.extend(first.iter().cloned());
    for capture in second {
        if !captures.iter().any(|(sym, _, _)| *sym == capture.0) {
            captures.push(capture.clone());
        }
    }
    captures
}

fn compose_soac_bodies(
    f: &super::SoacBody,
    g: &super::SoacBody,
    span: Span,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> super::SoacBody {
    super::SoacBody {
        lam: compose_lambdas(f.lam.clone(), g.lam.clone(), span, symbols, term_ids),
        captures: merged_captures(&f.captures, &g.captures),
    }
}

fn compose_soac_map_reduce(
    map_lam: &super::SoacBody,
    reduce_op: &super::SoacBody,
    span: Span,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> super::SoacBody {
    super::SoacBody {
        lam: compose_map_reduce(
            map_lam.lam.clone(),
            reduce_op.lam.clone(),
            span,
            symbols,
            term_ids,
        ),
        captures: merged_captures(&map_lam.captures, &reduce_op.captures),
    }
}

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
pub(super) fn compose_map_into_envelope(
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
pub(super) fn dedup_envelope_inputs(
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
// Post-normalize horizontal fusion: equal-domain sibling output maps
// =============================================================================

/// The primary input array of a SOAC — the one whose length is its
/// iteration domain.
fn soac_primary_input(soac: &SoacOp) -> Option<&ArrayExpr> {
    match soac {
        SoacOp::Map { inputs, .. } | SoacOp::Screma { inputs, .. } | SoacOp::Scatter { inputs, .. } => {
            inputs.first()
        }
        SoacOp::Reduce { input, .. } | SoacOp::Scan { input, .. } | SoacOp::Filter { input, .. } => {
            Some(input)
        }
        SoacOp::ReduceByIndex { indices, .. } => Some(indices),
    }
}

/// The iteration domain of a SOAC, identified by the size component of its
/// primary input's array type. Two SOACs share a domain — and so may be
/// horizontally fused into one kernel — exactly when these compare equal:
/// the same size `Variable` / `SizeVar` / `Size`. `None` when the domain
/// can't be read off the input type; the caller treats an unknown domain
/// as its own singleton class rather than assuming equality.
fn soac_domain_key(soac: &SoacOp) -> Option<Type<TypeName>> {
    let ArrayExpr::Var(_, ty) = soac_primary_input(soac)? else {
        return None;
    };
    let stripped = crate::types::strip_unique(ty);
    crate::types::array_size(&stripped).cloned()
}

/// Partition slot positions by iteration domain. Slots whose key compares
/// equal land in one class (fusible); slots with a `None` key each form
/// their own singleton class — an unknown domain is never assumed equal to
/// another. Order is preserved: a class carries its members in slot order,
/// and classes appear in first-seen order.
fn partition_by_domain(keys: &[Option<Type<TypeName>>]) -> Vec<Vec<usize>> {
    let mut classes: Vec<(Option<Type<TypeName>>, Vec<usize>)> = Vec::new();
    for (i, key) in keys.iter().enumerate() {
        match key {
            Some(k) => {
                if let Some((_, members)) =
                    classes.iter_mut().find(|(class_key, _)| class_key.as_ref() == Some(k))
                {
                    members.push(i);
                } else {
                    classes.push((Some(k.clone()), vec![i]));
                }
            }
            None => classes.push((None, vec![i])),
        }
    }
    classes.into_iter().map(|(_, members)| members).collect()
}

/// One output slot recognised by the equal-domain fuser: a plain
/// captureless `map(lam, ref)` over a single named entry-param array.
struct FusibleMapSlot {
    slot_index: usize,
    input: ArrayExpr,
    input_sym: SymbolId,
    lam: Lambda,
    output_ty: Type<TypeName>,
    domain: Type<TypeName>,
}

fn fusible_map_slot(slot_index: usize, term: &Term) -> Option<FusibleMapSlot> {
    let TermKind::Soac(
        soac @ SoacOp::Map {
            lam,
            inputs,
            destination,
        },
    ) = &term.kind
    else {
        return None;
    };
    if *destination != SoacDestination::Fresh || !lam.captures.is_empty() {
        return None;
    }
    if inputs.len() != 1 || lam.lam.params.len() != 1 {
        return None;
    }
    let input = &inputs[0];
    let input_sym = input.as_named_ref()?;
    let ArrayExpr::Var(..) = input else {
        return None;
    };
    let domain = soac_domain_key(soac)?;
    Some(FusibleMapSlot {
        slot_index,
        input: input.clone(),
        input_sym,
        lam: lam.lam.clone(),
        output_ty: term.ty.clone(),
        domain,
    })
}

/// Fuse the sibling output maps of an all-pointwise multi-output compute entry
/// whose slots all share **one iteration domain** into a single multi-output
/// pointwise `Screma`. The domain is the slots' common outer size
/// (`soac_domain_key`); slots over distinct buffers (`<[n]>(xs, ys)`) fuse as
/// long as they share that size. Each lane reads its own input via the Screma's
/// `map_input_indices`; same-symbol lanes index one shared input.
///
/// Such an entry then has one tail SOAC (the Screma) and no sibling SOAC slots,
/// so the multi-output-Screma path (`make_map_plan` → EGIR `build_parallel_maps`)
/// lowers it as one guarded parallel kernel with one lane per output.
///
/// Entries left untouched (handled by the per-slot split in
/// `make_multidomain_map_plan`): those whose slots span more than one domain,
/// and those with any slot that isn't a captureless single-input `map` over a
/// named entry param. Per-domain fusion of a mixed-domain entry's matching
/// slots is not done here — each slot becomes its own stage.
pub(crate) fn fuse_equal_domain_sibling_maps(program: &mut Program, term_ids: &mut TermIdSource) {
    let indices: Vec<usize> = program
        .defs
        .iter()
        .enumerate()
        .filter_map(|(i, d)| match &d.meta {
            DefMeta::EntryPoint(decl) if decl.entry_type.is_compute() => Some(i),
            _ => None,
        })
        .collect();
    for idx in indices {
        let body = program.defs[idx].body.clone();
        if let Some(fused) = fuse_entry_body(body, &mut program.symbols, term_ids) {
            program.defs[idx].body = fused;
        }
    }
}

/// Descend through the entry's `Lambda` params to the `OutputSlotStore`
/// chain and try to fuse it. Returns the rewritten full body, or `None` if
/// nothing fused.
fn fuse_entry_body(term: Term, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> Option<Term> {
    match term.kind {
        TermKind::Lambda(lam) => {
            let new_body = fuse_entry_body(*lam.body, symbols, term_ids)?;
            Some(Term {
                kind: TermKind::Lambda(Lambda {
                    params: lam.params,
                    body: Box::new(new_body),
                    ret_ty: lam.ret_ty,
                }),
                ..term
            })
        }
        _ => fuse_output_slot_chain(&term, symbols, term_ids),
    }
}

fn fuse_output_slot_chain(
    chain: &Term,
    symbols: &mut SymbolTable,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    let mut slots: Vec<(usize, Term)> = Vec::new();
    collect_extra_slot_stores(chain, &mut slots);
    if slots.len() < 2 {
        return None;
    }
    let fusible: Vec<FusibleMapSlot> =
        slots.iter().map(|(i, t)| fusible_map_slot(*i, t)).collect::<Option<Vec<_>>>()?;

    // Fuse only when every lane shares one iteration domain. Mixed-domain
    // entries belong to the per-slot split path (`make_multidomain_map_plan`).
    let domains: Vec<Option<Type<TypeName>>> = fusible.iter().map(|s| Some(s.domain.clone())).collect();
    if partition_by_domain(&domains).len() != 1 {
        return None;
    }

    // The fused Screma's guard length is taken from `inputs[0]` downstream
    // (`build_parallel_maps`); that is sound only because every fused input
    // shares one domain. Restate the proven invariant at this construction
    // boundary so the lowering can simply trust the Screma it receives.
    debug_assert!(
        fusible.iter().all(|s| s.domain == fusible[0].domain),
        "equal-domain fuser built a Screma whose lanes span different domains"
    );

    // Deduplicated union of the lanes' input arrays, first-seen order. Each lane
    // reads exactly its own input via `map_input_indices`; same-symbol lanes
    // collapse to one union entry that several lanes index.
    let mut union: Vec<(SymbolId, ArrayExpr)> = Vec::new();
    for slot in &fusible {
        if !union.iter().any(|(sym, _)| *sym == slot.input_sym) {
            union.push((slot.input_sym, slot.input.clone()));
        }
    }
    let inputs: Vec<ArrayExpr> = union.iter().map(|(_, ae)| ae.clone()).collect();
    // Each lane keeps its own single-input map function, reading exactly its own
    // input via `input_indices`; same-symbol lanes index one shared union entry.
    let lanes: Vec<super::ScremaLane> = fusible
        .iter()
        .map(|slot| {
            let pos = union
                .iter()
                .position(|(sym, _)| *sym == slot.input_sym)
                .expect("union built from these lanes");
            super::ScremaLane {
                lam: super::SoacBody {
                    lam: slot.lam.clone(),
                    captures: vec![],
                },
                input_indices: vec![pos],
            }
        })
        .collect();

    let result_fields: Vec<Type<TypeName>> = fusible.iter().map(|s| s.output_ty.clone()).collect();
    let tuple_ty = Type::Constructed(TypeName::Tuple(result_fields.len()), result_fields);
    let screma = Term {
        id: term_ids.next_id(),
        ty: tuple_ty.clone(),
        span: chain.span,
        kind: TermKind::Soac(SoacOp::Screma {
            lanes,
            accumulators: vec![],
            inputs,
        }),
    };
    let tuple_sym = symbols.alloc("_fused_maps".to_string());

    // Per lane, a let-bound projection of the fused tuple. The original
    // `OutputSlotStore` chain is rewritten in place to store these projection
    // vars instead of the maps — preserving its `SideEffect`-typed terminator
    // (which `build_entry_outputs` reads as the entry's return type) and
    // mirroring the let-bound-projection shape `find_direct_map_screma_group`
    // produces, so EGIR's SOAC→OutputView retargeting recognises each output
    // and `build_parallel_maps` fires.
    let proj_syms: Vec<SymbolId> =
        fusible.iter().map(|_| symbols.alloc("_fused_proj".to_string())).collect();
    let mut slot_proj: LookupMap<usize, (SymbolId, Type<TypeName>)> = LookupMap::new();
    for (lane, slot) in fusible.iter().enumerate() {
        slot_proj.insert(slot.slot_index, (proj_syms[lane], slot.output_ty.clone()));
    }
    let mut acc = rewrite_output_slot_values(chain, &slot_proj, term_ids);
    for (lane, slot) in fusible.iter().enumerate().rev() {
        let proj = Term {
            id: term_ids.next_id(),
            ty: slot.output_ty.clone(),
            span: chain.span,
            kind: TermKind::TupleProj {
                tuple: Box::new(Term {
                    id: term_ids.next_id(),
                    ty: tuple_ty.clone(),
                    span: chain.span,
                    kind: TermKind::Var(VarRef::Symbol(tuple_sym)),
                }),
                idx: lane,
            },
        };
        acc = let_term(
            proj_syms[lane],
            slot.output_ty.clone(),
            proj,
            acc,
            chain.span,
            term_ids,
        );
    }

    Some(let_term(tuple_sym, tuple_ty, screma, acc, chain.span, term_ids))
}

/// Clone an `OutputSlotStore` let-chain, replacing each store's value with a
/// reference to the projection var assigned to that slot. The chain's let
/// structure and types (including its `SideEffect`-typed terminator) are
/// preserved.
fn rewrite_output_slot_values(
    term: &Term,
    slot_proj: &LookupMap<usize, (SymbolId, Type<TypeName>)>,
    term_ids: &mut TermIdSource,
) -> Term {
    match &term.kind {
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            let new_rhs = match &rhs.kind {
                TermKind::OutputSlotStore { slot_index, .. } if slot_proj.contains_key(slot_index) => {
                    let (psym, pty) = &slot_proj[slot_index];
                    Term {
                        id: term_ids.next_id(),
                        ty: rhs.ty.clone(),
                        span: rhs.span,
                        kind: TermKind::OutputSlotStore {
                            slot_index: *slot_index,
                            value: Box::new(Term {
                                id: term_ids.next_id(),
                                ty: pty.clone(),
                                span: rhs.span,
                                kind: TermKind::Var(VarRef::Symbol(*psym)),
                            }),
                        },
                    }
                }
                _ => (**rhs).clone(),
            };
            Term {
                id: term_ids.next_id(),
                ty: term.ty.clone(),
                span: term.span,
                kind: TermKind::Let {
                    name: *name,
                    name_ty: name_ty.clone(),
                    rhs: Box::new(new_rhs),
                    body: Box::new(rewrite_output_slot_values(body, slot_proj, term_ids)),
                },
            }
        }
        _ => term.clone(),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[path = "fusion_tests.rs"]
mod fusion_tests;
