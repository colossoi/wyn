//! Producer-consumer planning: one place that classifies, per entry, how each
//! array/scalar producer is *demanded* by its consumers and which lowering
//! *strategy* fits — built on the fusion substrate ([`super::producer_graph`] +
//! [`super::array_semantics`]) rather than re-deriving structure.
//!
//! ## What is load-bearing vs analysis
//!
//! Producer-residency decisions are made at the pipeline points where each
//! producer is visible, so for some strategies the planner is the *authority*
//! and for others a *classifier kept consistent with* the authority that runs
//! elsewhere:
//!
//! * `StoragePrepass(ScalarBroadcast)` — **execution-driving**:
//!   [`super::parallelize`]'s `lift_compute_scalar_reduces` consumes exactly this
//!   marking to hoist scalar reduces into prepass entries.
//! * `BoundedAggregate{capacity}` / `View` — the filter representation choice,
//!   decided by the shared authority [`filter_variant`]; `rep_specialize` calls
//!   the same function, so the planner's report and the executor agree by
//!   construction.
//! * `StoragePrepass(Gather)` — the *demand* that an array be materialized. The
//!   residency **authority** is `lift_gathers::gather_decision`, co-located with
//!   the rewrite machinery it needs and run inside `parallelize` once gather
//!   producers become visible (post-`normalize_for_gather`); see the boundary
//!   note below.
//! * `Fuse` — mirrors `array_semantics::can_fuse`, the seam shared with
//!   `fusion.rs`. Fusion runs much earlier in the pipeline, so the planner
//!   classifies fusability but does not *drive* it; `can_fuse` is the one
//!   decision both share.
//! * `LeaveAsIs` — no producer-derived decision.
//!
//! ## Boundary: producer residency (here) vs output-slot realization (EGIR)
//!
//! This planner owns *producer residency* in TLC, before EGIR. A separate,
//! deliberate boundary — `egir::realize_outputs::dispatch::compute_slot_source`
//! — owns *output-slot realization*, classifying on post-expansion EGraph shape
//! (consuming scan / retargetable map-scan / fixed aggregate / runtime-sized
//! reject / scalar store), which doesn't exist in TLC. The two are distinct
//! passes; neither subsumes the other.
//!
//! The producer graph sees only let-bound and tail producers, and its edges
//! capture producer→producer array-input links. The two demand signals that
//! distinguish the strategies are read directly off the IR here: a producer
//! captured into a SOAC operator lambda (`SoacBody::captures`) is consumed
//! *per element* — a gather if it's an array, a broadcast if it's a scalar —
//! while `ProducerNode::use_count` and the graph edges give the whole-array
//! consumer count.

use std::collections::HashSet;

use super::array_semantics::{ArraySemantics, FusionKind, can_fuse, summarize_program};
use super::fusion::build_sym_to_def;
use super::producer_graph::{ProducerGraph, ProducerId, build_producer_graph};
use super::{ArrayExpr, DefMeta, Program, SoacOp, Term, TermKind, VarRef};
use crate::SymbolId;
use crate::ast::TypeName;
use polytype::Type;

/// How a producer's result is demanded by its consumers. Recorded for the
/// report; the strategy is chosen from the combination of demands plus the
/// producer's own shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Demand {
    /// Consumed whole by a fusable SOAC (a single-use map→map/reduce/scan).
    Fuse,
    /// Read per element inside a sibling SOAC's lambda — a gather (array) or a
    /// broadcast (scalar), depending on the producer's shape.
    PerElement,
    /// Consumed whole by 2+ array-consuming SOACs (forces materialization).
    MultiConsumer,
    /// No recognized consumer in this entry (tail/output or unanalyzed).
    Loose,
}

/// The lowering strategy classified for one producer. See the module doc for
/// which variants drive execution (`StoragePrepass(ScalarBroadcast)`) versus
/// mirror an authority that runs elsewhere (`Bounded`/`View` ↔ [`filter_variant`],
/// `StoragePrepass(Gather)` ↔ `lift_gathers::gather_decision`, `Fuse` ↔
/// `can_fuse`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Compose the producer into its single consumer (no materialization).
    Fuse,
    /// Materialize to a scratch buffer in a prepass; consumers read it back.
    StoragePrepass(PrepassKind),
    /// A `filter` whose output is a fixed-capacity `{buffer,len}` aggregate;
    /// `capacity` is the static `Size(N)` of the filtered input.
    BoundedAggregate {
        capacity: usize,
    },
    /// A storage-buffer-backed `{offset,len}` view (runtime length OK).
    View,
    /// Nothing to do — fuses trivially, lowers as-is, or is a plain scalar.
    LeaveAsIs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrepassKind {
    /// An array materialized to a buffer; consumers read it via view loads.
    Gather,
    /// A scalar materialized to a buffer; consumers read slot 0.
    ScalarBroadcast,
}

/// A producer and the strategy planned for it.
#[derive(Debug, Clone)]
pub struct PlannedProducer {
    /// The let-bound name, if any (`None` for a tail producer).
    pub binding: Option<SymbolId>,
    /// `true` if this producer yields an array, `false` for a scalar result.
    pub produces_array: bool,
    pub demand: Demand,
    pub strategy: Strategy,
}

/// The plan for one entry: a strategy per producer in its body.
#[derive(Debug, Clone)]
pub struct EntryPlan {
    pub entry: SymbolId,
    pub producers: Vec<PlannedProducer>,
}

impl EntryPlan {
    /// The planned producer bound to `binding`, if present.
    pub fn by_binding(&self, binding: SymbolId) -> Option<&PlannedProducer> {
        self.producers.iter().find(|p| p.binding == Some(binding))
    }

    /// The let-binding symbols this entry's plan marked for gather
    /// materialization (`StoragePrepass(Gather)`) — the set the gather lift
    /// executes. `lift_gathers` is the executor of this report.
    pub fn gather_bindings(&self) -> HashSet<SymbolId> {
        self.producers
            .iter()
            .filter(|p| matches!(p.strategy, Strategy::StoragePrepass(PrepassKind::Gather)))
            .filter_map(|p| p.binding)
            .collect()
    }
}

/// Plan every entry in `program`. Pure analysis — `program` is not modified.
pub fn plan_program(program: &Program) -> Vec<EntryPlan> {
    let summaries = summarize_program(program);
    let sym_to_def = build_sym_to_def(&program.symbols, &program.def_syms);

    program
        .defs
        .iter()
        .filter(|d| matches!(d.meta, DefMeta::EntryPoint(_)))
        .map(|def| {
            let body = entry_body(&def.body);
            let graph = build_producer_graph(body, &summaries, &sym_to_def);
            let per_element = per_element_producers(body, &graph);
            let producers = (0..graph.node_count())
                .map(|i| plan_node(&graph, ProducerId(i as u32), &per_element))
                .collect();
            EntryPlan {
                entry: def.name,
                producers,
            }
        })
        .collect()
}

/// Report-only entry point: compute the plan and, in debug builds, log it.
/// The IR is untouched.
pub fn report(program: &Program) {
    if !cfg!(debug_assertions) {
        return;
    }
    let plans = plan_program(program);
    for plan in &plans {
        let name = program.symbols.get(plan.entry).cloned().unwrap_or_default();
        for p in &plan.producers {
            let bname = p
                .binding
                .and_then(|s| program.symbols.get(s).cloned())
                .unwrap_or_else(|| "<tail>".to_string());
            log::debug!(
                "producer_plan: entry `{name}` producer `{bname}` array={} demand={:?} strategy={:?}",
                p.produces_array,
                p.demand,
                p.strategy
            );
        }
    }
}

/// Peel an entry def's outer parameter `Lambda` to reach the let-chain + tail.
fn entry_body(body: &Term) -> &Term {
    match &body.kind {
        TermKind::Lambda(lam) => &lam.body,
        _ => body,
    }
}

/// Pick the strategy for one producer node from its shape, consumer count, and
/// whether it is consumed per element (captured into a SOAC lambda or runtime-
/// indexed).
fn plan_node(graph: &ProducerGraph, pid: ProducerId, per_element: &HashSet<SymbolId>) -> PlannedProducer {
    let node = graph.node(pid);
    let produces_array = node.semantics.produces_array();
    let is_per_element = node.binding.map(|s| per_element.contains(&s)).unwrap_or(false);

    let demand = classify_demand(graph, pid, is_per_element);
    let strategy = match (&node.semantics, produces_array, demand) {
        // `filter` is a representation choice, not a prepass: Bounded for a
        // statically-sized input, View for a runtime one — the same
        // `filter_variant` call `rep_specialize` executes.
        (ArraySemantics::Filter { input, .. }, _, _) => match filter_variant(input) {
            Some(FilterVariant::Bounded { capacity }) => Strategy::BoundedAggregate { capacity },
            Some(FilterVariant::View) => Strategy::View,
            // Input type unreadable (a fused-chain input): no producer-derived
            // variant. `rep_specialize` does nothing here (its `?` short-circuits
            // to "no specialization"), so the planner must report the same
            // "leave alone" — reporting `View` would disagree with the executor.
            None => Strategy::LeaveAsIs,
        },
        // A scalar SOAC result read per element inside a sibling lambda
        // broadcasts — hoist it to a one-element prepass buffer.
        (_, false, Demand::PerElement) => Strategy::StoragePrepass(PrepassKind::ScalarBroadcast),
        (_, false, _) => Strategy::LeaveAsIs,
        // An array read per element (a gather) or by 2+ whole-array consumers
        // must materialize.
        (_, true, Demand::PerElement | Demand::MultiConsumer) => {
            Strategy::StoragePrepass(PrepassKind::Gather)
        }
        // A single fusable consumer: compose.
        (_, true, Demand::Fuse) => Strategy::Fuse,
        (_, true, Demand::Loose) => Strategy::LeaveAsIs,
    };

    PlannedProducer {
        binding: node.binding,
        produces_array,
        demand,
        strategy,
    }
}

/// The concrete representation a `filter(pred, input)` producer must take.
/// The single source of truth for the filter variant choice: the planner
/// records it as a [`Strategy`] and `rep_specialize` executes it as a
/// `ConcreteVariant`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterVariant {
    /// Statically-sized input — a fixed-capacity `{buffer,len}` aggregate.
    Bounded {
        capacity: usize,
    },
    /// Runtime-length input — a `{offset,len}` view.
    View,
}

/// Decide a `filter` producer's variant from its input array's size: a static
/// `Size(N)` yields `Bounded{N}`; any runtime length yields `View`. `None` when
/// the input type can't be read — a fused-chain input (`Soac`/`Zip`/`Literal`/
/// `Range`) carries no bound array type, so there's no producer-derived variant.
pub fn filter_variant(input: &ArrayExpr) -> Option<FilterVariant> {
    let input_ty = filter_input_type(input)?;
    let size = crate::types::array_size(&input_ty)?;
    Some(match size {
        Type::Constructed(TypeName::Size(n), _) => FilterVariant::Bounded { capacity: *n },
        _ => FilterVariant::View,
    })
}

/// Best-effort array-type extraction for the shapes a `SoacOp::Filter` input
/// can take. Only `Ref` (a bound name with an Array-typed term) and
/// `StorageView` (an entry view-array) carry a usable array type; other
/// variants appear only in fused chains and yield `None`.
fn filter_input_type(ae: &ArrayExpr) -> Option<Type<TypeName>> {
    match ae {
        ArrayExpr::Ref(t) => Some(t.ty.clone()),
        ArrayExpr::StorageView(sv) => Some(crate::types::view_array_of(
            &sv.elem_ty,
            crate::types::region_tag(sv.binding),
        )),
        _ => None,
    }
}

/// Classify the dominant demand on a producer.
fn classify_demand(graph: &ProducerGraph, pid: ProducerId, is_per_element: bool) -> Demand {
    if is_per_element {
        // Captured into a SOAC operator lambda or read via runtime index →
        // consumed per element.
        return Demand::PerElement;
    }
    let node = graph.node(pid);
    let consumers = graph.consumers_of(pid);
    // `use_count` counts every reference (including non-SOAC); the graph edges
    // count whole-array SOAC consumers. More than one whole-array consumer (or
    // any extra non-SOAC use beyond a single consumer) forces materialization.
    if consumers.len() >= 2 || node.use_count >= 2 {
        return Demand::MultiConsumer;
    }
    if consumers.len() == 1 {
        let consumer = graph.node(consumers[0].consumer);
        if can_fuse(&node.semantics, &consumer.semantics) != FusionKind::NotFusible {
            return Demand::Fuse;
        }
        return Demand::MultiConsumer;
    }
    Demand::Loose
}

/// Collect the producer symbols consumed *per element* anywhere in `body` — a
/// producer captured into a SOAC operator lambda (`SoacBody::captures`) **or**
/// read through a runtime `Index`. Both are the gather shape for an array (and a
/// broadcast for a scalar): the consumer needs one element at a time, so a
/// computed producer must be materialized to a buffer rather than fused.
fn per_element_producers(body: &Term, graph: &ProducerGraph) -> HashSet<SymbolId> {
    let producers: HashSet<SymbolId> =
        (0..graph.node_count()).filter_map(|i| graph.node(ProducerId(i as u32)).binding).collect();
    let mut out = HashSet::new();
    walk_per_element(body, &producers, &mut out);
    out
}

fn walk_per_element(term: &Term, producers: &HashSet<SymbolId>, out: &mut HashSet<SymbolId>) {
    // Captured into a SOAC operator lambda.
    if let TermKind::Soac(soac) = &term.kind {
        for sb in soac_bodies(soac) {
            for (_, _, cap_term) in &sb.captures {
                if let TermKind::Var(VarRef::Symbol(s)) = &cap_term.kind {
                    if producers.contains(s) {
                        out.insert(*s);
                    }
                }
            }
        }
    }
    // Read through a runtime `Index` (a constant index fuses in
    // `static_index_fusion` and isn't a gather; only runtime indexing is).
    if let TermKind::Index { array, index } = &term.kind {
        if let TermKind::Var(VarRef::Symbol(s)) = &array.kind {
            if producers.contains(s) && !matches!(index.kind, TermKind::IntLit(_)) {
                out.insert(*s);
            }
        }
    }
    term.for_each_child(&mut |c| walk_per_element(c, producers, out));
}

/// The operator [`super::SoacBody`]s of a SOAC (every lambda it carries).
fn soac_bodies(soac: &SoacOp) -> Vec<&super::SoacBody> {
    match soac {
        SoacOp::Map { lam, .. } => vec![lam],
        SoacOp::Reduce { op, .. } => vec![op],
        SoacOp::Redomap { op, reduce_op, .. } => vec![op, reduce_op],
        SoacOp::Scan { op, reduce_op, .. } => vec![op, reduce_op],
        SoacOp::Filter { pred, .. } => vec![pred],
        _ => Vec::new(),
    }
}

#[cfg(test)]
#[path = "producer_plan_tests.rs"]
mod producer_plan_tests;
