//! The semantic layer of EGIR: its dependency graph, and the checks and
//! rendering that read it.
//!
//! Semantic EGIR uses a DAG over side-effectful semantic SOAC operations for
//! scheduling, fusion legality, multi-consumer materialization, and semantic
//! optimization. This module owns the edge builder ([`rebuild_dependencies`]),
//! the read-only query index over those edges ([`SemanticGraph`]), the
//! well-formedness check for the semantic boundary ([`verify`]), and the
//! human-readable dump of it ([`summary`]).

use std::collections::{HashMap, HashSet};

use super::graph_ops;
use super::program::{SemanticDependency, SemanticDependencyKind, SemanticOpId, SemanticProgram};
use super::soac::{filter, hist, screma};
use super::types::{
    EGraph, NodeId, SegResourceAccess, SegResourceAccessKind, SideEffect, SideEffectKind, SideEffectSite,
    Soac,
};

/// Rebuild the semantic dependency DAG stored on `inner`.
pub(crate) fn rebuild_dependencies(inner: &mut SemanticProgram) {
    inner.semantic_dependencies = dependencies(inner);
}

/// Build semantic value/effect/resource dependencies for every semantic SOAC in
/// the program.
pub(crate) fn dependencies(inner: &SemanticProgram) -> Vec<SemanticDependency> {
    let mut dependencies = Vec::new();
    for entry in &inner.entry_points {
        collect_graph_dependencies(&entry.name, &entry.graph, &mut dependencies);
    }
    for function in &inner.functions {
        collect_graph_dependencies(&function.name, &function.graph, &mut dependencies);
    }
    dependencies
}

/// Every edge runs between two ops of `graph`, so duplicates can only arise
/// within one scope and `seen` need not outlive this call.
fn collect_graph_dependencies(_scope: &str, graph: &EGraph, output: &mut Vec<SemanticDependency>) {
    struct Record<'a> {
        id: SemanticOpId,
        result: NodeId,
        effect: &'a SideEffect,
        resources: Vec<SegResourceAccess>,
    }
    let mut seen: HashSet<SemanticDependency> = HashSet::new();

    let mut records = Vec::new();
    for (_, block) in &graph.skeleton.blocks {
        for effect in &block.side_effects {
            let SideEffectKind::Soac(id, soac) = &effect.kind else {
                continue;
            };
            if let Some(result) = effect.result {
                let resources = match soac {
                    Soac::Screma(op) => match op.semantic_state() {
                        screma::SemanticState::Serial => read_resources(graph, effect),
                        screma::SemanticState::Segmented { resources, .. } => resources.clone(),
                    },
                    Soac::Filter(op) => {
                        let mut resources = read_resources(graph, effect);
                        let bindings: Vec<_> = match &op.state.storage {
                            filter::Output::Local { .. } => Vec::new(),
                            filter::Output::Runtime { scratch, length } => {
                                let mut bindings = vec![*scratch];
                                if let filter::RuntimeLength::Stored(length) = length {
                                    bindings.push(*length);
                                }
                                bindings
                            }
                        };
                        for binding in bindings {
                            resources.push(SegResourceAccess {
                                resource: binding,
                                access: SegResourceAccessKind::Write,
                            });
                        }
                        resources
                    }
                    Soac::Hist(_) => {
                        let mut resources = read_resources(graph, effect);
                        if let Some(destination) = effect
                            .operand_nodes
                            .first()
                            .and_then(|node| graph_ops::extract_storage_view_source(graph, *node))
                        {
                            if let Some(resource) =
                                resources.iter_mut().find(|resource| resource.resource == destination)
                            {
                                resource.access = SegResourceAccessKind::ReadWrite;
                            }
                        }
                        resources
                    }
                };
                records.push(Record {
                    id: *id,
                    result,
                    effect,
                    resources,
                });
            }
        }
    }

    for consumer_index in 0..records.len() {
        let consumer = &records[consumer_index];
        let reachable = graph_ops::value_producer_closure(graph, consumer.effect.referenced_nodes()).nodes;
        for producer in &records[..consumer_index] {
            if reachable.contains(&producer.result) {
                push_dependency(
                    output,
                    &mut seen,
                    &producer.id,
                    &consumer.id,
                    SemanticDependencyKind::Value,
                );
            }
            if matches!((producer.effect.effects, consumer.effect.effects),
                (Some((_, out)), Some((input, _))) if out == input)
            {
                push_dependency(
                    output,
                    &mut seen,
                    &producer.id,
                    &consumer.id,
                    SemanticDependencyKind::Effect,
                );
            }
            if producer.resources.iter().any(|left| {
                consumer.resources.iter().any(|right| {
                    left.resource == right.resource
                        && (left.access != SegResourceAccessKind::Read
                            || right.access != SegResourceAccessKind::Read)
                })
            }) {
                push_dependency(
                    output,
                    &mut seen,
                    &producer.id,
                    &consumer.id,
                    SemanticDependencyKind::Resource,
                );
            }
        }
    }
}

/// Append `edge` unless an identical one is already present. `output` keeps
/// discovery order; `seen` answers membership without rescanning it.
fn push_dependency(
    output: &mut Vec<SemanticDependency>,
    seen: &mut HashSet<SemanticDependency>,
    producer: &SemanticOpId,
    consumer: &SemanticOpId,
    kind: SemanticDependencyKind,
) {
    let edge = SemanticDependency {
        producer: producer.clone(),
        consumer: consumer.clone(),
        kind,
    };
    if seen.insert(edge.clone()) {
        output.push(edge);
    }
}

pub(crate) fn read_resources(graph: &EGraph, se: &SideEffect) -> Vec<SegResourceAccess> {
    let bindings = graph_ops::value_producer_closure(graph, se.referenced_nodes())
        .nodes
        .into_iter()
        .filter_map(|node| graph_ops::extract_storage_view_source(graph, node))
        .collect::<HashSet<_>>();
    let mut result: Vec<_> = bindings
        .into_iter()
        .map(|resource| SegResourceAccess {
            resource,
            access: SegResourceAccessKind::Read,
        })
        .collect();
    result.sort_by_key(|resource| resource.resource);
    result
}

/// Validate the semantic boundary before any target-aware scheduling occurs.
pub(crate) fn verify(inner: &SemanticProgram) -> Result<(), String> {
    let verify_effect = |scope: &str, effect: &SideEffect| -> Result<(), String> {
        let SideEffectKind::Soac(_, soac) = &effect.kind else {
            return Ok(());
        };
        let verify_body = |family: &str, body: &super::types::SegBody| {
            if inner.contains_region(body.region) {
                Ok(())
            } else {
                Err(format!(
                    "{scope}: {family} region `{}` is absent from the EGIR region arena",
                    body.region
                ))
            }
        };
        match soac {
            Soac::Screma(op) => {
                if let screma::SemanticState::Segmented { space, .. } = op.semantic_state() {
                    if space.dims.is_empty() {
                        return Err(format!("{scope}: semantic Screma has no concrete dimensions"));
                    }
                }
                for map in &op.lanes().maps {
                    verify_body("map", &map.body)?;
                }
                for operator in op.operators() {
                    verify_body("operator step", &operator.step)?;
                    verify_body("operator combine", &operator.combine)?;
                }
            }
            Soac::Filter(op) => {
                if op.state.space.dims.is_empty() {
                    return Err(format!("{scope}: semantic filter has no concrete dimensions"));
                }
                if let filter::Input::Mapped { body, .. } = &op.body.input {
                    verify_body("filter map", body)?;
                }
                verify_body("filter predicate", &op.body.predicate)?;
            }
            Soac::Hist(op) => {
                if let hist::SemanticState::Segmented(space) = &op.state {
                    if space.dims.is_empty() {
                        return Err(format!("{scope}: semantic histogram has no concrete dimensions"));
                    }
                }
                if !inner.contains_region(op.body.body.region) {
                    return Err(format!(
                        "{scope}: histogram region `{}` is absent",
                        op.body.body.region
                    ));
                }
            }
        }
        Ok(())
    };
    for entry in &inner.entry_points {
        for (_, block) in &entry.graph.skeleton.blocks {
            for effect in &block.side_effects {
                verify_effect(&format!("entry `{}`", entry.name), effect)?;
            }
        }
    }
    for function in &inner.functions {
        for (_, block) in &function.graph.skeleton.blocks {
            for effect in &block.side_effects {
                verify_effect(&format!("function `{}`", function.name), effect)?;
            }
        }
    }
    Ok(())
}

pub(crate) fn summary(inner: &SemanticProgram) -> String {
    use std::fmt::Write;

    let mut output = String::new();
    let mut print_graph = |scope: &str, graph: &EGraph| {
        for (_, block) in &graph.skeleton.blocks {
            for effect in &block.side_effects {
                match &effect.kind {
                    SideEffectKind::Soac(_, Soac::Screma(op)) => {
                        let kind = match op.flavor() {
                            screma::Flavor::Map => "SegMap",
                            screma::Flavor::Reduce => "SegRed",
                            screma::Flavor::Scan => "SegScan",
                            screma::Flavor::Composite => "SegComposite",
                        };
                        let _ = writeln!(
                            output,
                            "{scope}: {kind} state={:?} inputs={:?} maps={:?} kind={:?}",
                            op.semantic_state(),
                            op.lanes().inputs,
                            op.lanes().maps,
                            op.flavor(),
                        );
                    }
                    SideEffectKind::Soac(_, Soac::Filter(op)) => {
                        let _ = writeln!(
                            output,
                            "{scope}: Filter state={:?} input={:?} predicate={:?}",
                            op.state, op.body.input, op.body.predicate
                        );
                    }
                    SideEffectKind::Soac(_, Soac::Hist(op)) => {
                        let _ = writeln!(
                            output,
                            "{scope}: Hist state={:?} body={:?} update={:?}",
                            op.state, op.body.body, op.body.update_policy
                        );
                    }
                    SideEffectKind::Effect(_) => {}
                }
            }
        }
    };
    for entry in &inner.entry_points {
        print_graph(&format!("entry {}", entry.name), &entry.graph);
    }
    for function in &inner.functions {
        print_graph(&format!("function {}", function.name), &function.graph);
    }
    output
}

/// An index over `&[SemanticDependency]` answering the questions fusion asks:
/// do two ops conflict, who consumes a producer's value, and is one op
/// transitively downstream of another.
pub struct SemanticGraph {
    /// Dense interning of every op that appears in any edge.
    index: HashMap<SemanticOpId, usize>,
    /// Reverse lookup for dense operation indices.
    operations: Vec<SemanticOpId>,
    /// Value successors (consumers that read the producer's result).
    value_succ: Vec<Vec<usize>>,
    /// Unordered resource/effect reordering conflicts, stored both ways.
    conflict: HashSet<(usize, usize)>,
    /// Graph-local scalar/aggregate values captured by each semantic
    /// operation, stored in the same dense operation index as the ordinary
    /// semantic DAG.
    capture_pred: Vec<Vec<NodeId>>,
    /// Stable-for-this-snapshot operation locations. Fusion does not require
    /// these; scheduling policies use them to inspect and rewrite consumers.
    operation_sites: Vec<Option<SideEffectSite>>,
}

impl SemanticGraph {
    pub fn new(deps: &[SemanticDependency]) -> Self {
        let mut graph = Self {
            index: HashMap::new(),
            operations: Vec::new(),
            value_succ: Vec::new(),
            conflict: HashSet::new(),
            capture_pred: Vec::new(),
            operation_sites: Vec::new(),
        };
        for dep in deps {
            let p = graph.intern_operation(dep.producer);
            let c = graph.intern_operation(dep.consumer);
            match dep.kind {
                SemanticDependencyKind::Value => graph.value_succ[p].push(c),
                // Both explicit effect ordering and resource aliasing prohibit
                // moving another operation across this edge. A directly
                // adjacent pair may still be fused in source order; callers use
                // this relation for the operations *between* that pair.
                SemanticDependencyKind::Effect | SemanticDependencyKind::Resource => {
                    graph.conflict.insert((p, c));
                    graph.conflict.insert((c, p));
                }
            }
        }
        graph
    }

    /// Extend the semantic operation DAG with graph-local values captured by
    /// its SOACs. Capture sources are not assigned synthetic `SemanticOpId`s;
    /// their successors use the DAG's existing dense operation identities.
    pub(crate) fn with_operation_captures(deps: &[SemanticDependency], egir: &EGraph) -> Self {
        let mut graph = Self::new(deps);
        for (block, skeleton_block) in &egir.skeleton.blocks {
            for (effect_index, effect) in skeleton_block.side_effects.iter().enumerate() {
                let SideEffectKind::Soac(id, soac) = &effect.kind else {
                    continue;
                };
                let operation = graph.intern_operation(*id);
                graph.operation_sites[operation] = Some(SideEffectSite {
                    block,
                    index: effect_index,
                });
                let mut seen = HashSet::new();
                graph.capture_pred[operation]
                    .extend(soac.capture_nodes().filter(|source| seen.insert(*source)));
            }
        }
        graph
    }

    fn intern_operation(&mut self, operation: SemanticOpId) -> usize {
        if let Some(&index) = self.index.get(&operation) {
            return index;
        }
        let index = self.operations.len();
        self.index.insert(operation, index);
        self.operations.push(operation);
        self.value_succ.push(Vec::new());
        self.capture_pred.push(Vec::new());
        self.operation_sites.push(None);
        index
    }

    /// Semantic operations in deterministic dependency/source order.
    pub(crate) fn operations(&self) -> impl Iterator<Item = SemanticOpId> + '_ {
        self.operations.iter().copied()
    }

    /// Graph-local values directly captured by `operation`.
    pub(crate) fn operation_captures(&self, operation: &SemanticOpId) -> impl Iterator<Item = NodeId> + '_ {
        self.index.get(operation).into_iter().flat_map(|index| self.capture_pred[*index].iter().copied())
    }

    /// Locate an operation in the EGIR snapshot used to add capture sources.
    pub(crate) fn operation_site(&self, operation: &SemanticOpId) -> Option<SideEffectSite> {
        self.index.get(operation).and_then(|index| self.operation_sites[*index])
    }

    /// Resource and Effect edges are both reordering conflicts. A caller may
    /// still combine a directly adjacent pair while preserving source order,
    /// but cannot move either operation across such an edge.
    pub fn conflicts(&self, a: &SemanticOpId, b: &SemanticOpId) -> bool {
        match (self.index.get(a), self.index.get(b)) {
            (Some(&i), Some(&j)) => self.conflict.contains(&(i, j)),
            _ => false,
        }
    }

    /// True iff `b` is transitively reachable from `a` along *value* edges,
    /// i.e. `a`'s result flows (directly or indirectly) into `b`, making them a
    /// producer/consumer chain rather than fusable siblings.
    pub fn reachable_between(&self, a: &SemanticOpId, b: &SemanticOpId) -> bool {
        let (Some(&start), Some(&target)) = (self.index.get(a), self.index.get(b)) else {
            return false;
        };
        wyn_graph::reaches_ordered(start, target, wyn_graph::WalkOrder::DepthFirst, |node, out| {
            out.extend(self.value_succ[node].iter().copied());
        })
    }

    /// Number of semantic operations that directly consume `producer`'s
    /// result. Multiple uses inside one consumer count once because the DAG is
    /// operation-granular.
    pub fn value_consumer_count(&self, producer: &SemanticOpId) -> usize {
        self.index.get(producer).map(|&index| self.value_succ[index].len()).unwrap_or(0)
    }
}

#[cfg(test)]
#[path = "semantic_graph_tests.rs"]
mod semantic_graph_tests;
