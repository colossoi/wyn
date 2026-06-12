//! Producer-consumer planning (report-only in this stage).
//!
//! One place that classifies, per entry, how each array/scalar producer is
//! *demanded* by its consumers and which lowering *strategy* fits — built on
//! the fusion substrate ([`super::producer_graph`] + [`super::array_semantics`])
//! rather than re-deriving structure. The scattered deciders this is meant to
//! consolidate (fusion, `rep_specialize`'s variant choice, `lift_gathers`,
//! `parallelize`'s scalar-reduce hoist, output retargeting) each own a slice of
//! this same decision today.
//!
//! In this stage the planner only *reports* — it computes a [`Strategy`] per
//! producer and forwards the IR unchanged, so later stages can take over one
//! strategy at a time (asserting the planner agrees with the legacy pass before
//! replacing it).
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

/// The lowering strategy chosen for one producer. Payload-free in this stage
/// (report-only); later stages attach the concrete bindings/slots they need.
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
            let captured = soac_captured_producers(body, &graph);
            let producers = (0..graph.node_count())
                .map(|i| plan_node(&graph, ProducerId(i as u32), &captured))
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
/// whether it is captured per-element into a SOAC lambda.
fn plan_node(graph: &ProducerGraph, pid: ProducerId, captured: &HashSet<SymbolId>) -> PlannedProducer {
    let node = graph.node(pid);
    let produces_array = node.semantics.produces_array();
    let is_captured = node.binding.map(|s| captured.contains(&s)).unwrap_or(false);

    let demand = classify_demand(graph, pid, is_captured);
    let strategy = match (&node.semantics, produces_array, demand) {
        // `filter` is a representation choice, not a prepass: Bounded for a
        // statically-sized input, View otherwise (the same call rep_specialize
        // executes). An input whose type can't be read falls back to View.
        (ArraySemantics::Filter { input, .. }, _, _) => match filter_variant(input) {
            Some(FilterVariant::Bounded { capacity }) => Strategy::BoundedAggregate { capacity },
            Some(FilterVariant::View) | None => Strategy::View,
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
fn classify_demand(graph: &ProducerGraph, pid: ProducerId, is_captured: bool) -> Demand {
    if is_captured {
        // Captured into a SOAC operator lambda → consumed per element.
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

/// Collect the producer symbols that are captured into some SOAC operator
/// lambda anywhere in `body` — the per-element-consumed producers.
fn soac_captured_producers(body: &Term, graph: &ProducerGraph) -> HashSet<SymbolId> {
    let producers: HashSet<SymbolId> =
        (0..graph.node_count()).filter_map(|i| graph.node(ProducerId(i as u32)).binding).collect();
    let mut out = HashSet::new();
    walk_captures(body, &producers, &mut out);
    out
}

fn walk_captures(term: &Term, producers: &HashSet<SymbolId>, out: &mut HashSet<SymbolId>) {
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
    term.for_each_child(&mut |c| walk_captures(c, producers, out));
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
