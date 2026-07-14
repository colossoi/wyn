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
    EGraph, NodeId, SegResourceAccess, SegResourceAccessKind, SideEffect, SideEffectKind, Soac,
};

/// Rebuild the semantic dependency DAG stored on `inner`.
pub(crate) fn rebuild_dependencies(inner: &mut SemanticProgram) {
    super::program::assign_semantic_op_ids(inner);
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
            let SideEffectKind::Soac(soac) = &effect.kind else {
                continue;
            };
            if let Some(result) = effect.result {
                let resources = match soac {
                    Soac::Screma(op) => match &op.state {
                        screma::SemanticState::Serial => read_resources(graph, effect),
                        screma::SemanticState::Segmented { resources, .. } => resources.clone(),
                    },
                    Soac::Filter(op) => {
                        let mut resources = read_resources(graph, effect);
                        let bindings: Vec<_> = match &op.state.storage {
                            filter::Output::Local { .. } => Vec::new(),
                            filter::Output::Runtime { scratch, length } => {
                                let mut bindings = vec![*scratch];
                                if let filter::RuntimeLength::EntryOutput(length) = length {
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
                    id: effect.required_semantic_id(),
                    result,
                    effect,
                    resources,
                });
            }
        }
    }

    for consumer_index in 0..records.len() {
        let consumer = &records[consumer_index];
        let reachable = wyn_graph::reachable_set(
            consumer.effect.referenced_nodes(),
            wyn_graph::WalkOrder::DepthFirst,
            |node, dependencies| {
                dependencies.extend(graph.nodes[node].children());
            },
        );
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
    let mut bindings = HashSet::new();
    wyn_graph::for_each_reachable(
        se.referenced_nodes(),
        wyn_graph::WalkOrder::DepthFirst,
        |node, out| out.extend(graph.nodes[node].children()),
        |node| {
            if let Some(binding) = graph_ops::extract_storage_view_source(graph, node) {
                bindings.insert(binding);
            }
        },
    );
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
        let SideEffectKind::Soac(soac) = &effect.kind else {
            return Ok(());
        };
        let verify_body = |family: &str, body: &super::types::SegBody| {
            if inner.regions.contains_key(&body.region) {
                Ok(())
            } else {
                Err(format!(
                    "{scope}: {family} region `{}` is absent from the EGIR region arena",
                    body.region.index()
                ))
            }
        };
        match soac {
            Soac::Screma(op) => {
                if let screma::SemanticState::Segmented { space, .. } = &op.state {
                    if space.dims.is_empty() {
                        return Err(format!("{scope}: semantic Screma has no concrete dimensions"));
                    }
                }
                for map in &op.body.maps {
                    verify_body("map", &map.body)?;
                }
                for index in 0..op.body.kind.len() {
                    let operator = op.body.kind.operator(index).expect("operator index from kind length");
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
                if !inner.regions.contains_key(&op.body.body.region) {
                    return Err(format!(
                        "{scope}: histogram region `{}` is absent",
                        op.body.body.region.index()
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
                    SideEffectKind::Soac(Soac::Screma(op)) => {
                        let kind = match &op.body.kind {
                            screma::Kind::Map => "SegMap",
                            screma::Kind::Reduce(_) => "SegRed",
                            screma::Kind::Scan(_) => "SegScan",
                            screma::Kind::Composite(_) => "SegComposite",
                        };
                        let _ = writeln!(
                            output,
                            "{scope}: {kind} state={:?} inputs={:?} maps={:?} kind={:?}",
                            op.state, op.body.inputs, op.body.maps, op.body.kind,
                        );
                    }
                    SideEffectKind::Soac(Soac::Filter(op)) => {
                        let _ = writeln!(
                            output,
                            "{scope}: Filter state={:?} input={:?} predicate={:?}",
                            op.state, op.body.input, op.body.predicate
                        );
                    }
                    SideEffectKind::Soac(Soac::Hist(op)) => {
                        let _ = writeln!(
                            output,
                            "{scope}: Hist state={:?} body={:?} update={:?}",
                            op.state, op.body.body, op.body.update_policy
                        );
                    }
                    SideEffectKind::Inst(_) => {}
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
    /// Value successors (consumers that read the producer's result).
    value_succ: Vec<Vec<usize>>,
    /// Unordered resource/effect reordering conflicts, stored both ways.
    conflict: HashSet<(usize, usize)>,
}

impl SemanticGraph {
    pub fn new(deps: &[SemanticDependency]) -> Self {
        let mut index: HashMap<SemanticOpId, usize> = HashMap::new();
        let mut intern = |op: &SemanticOpId| -> usize {
            if let Some(&i) = index.get(op) {
                return i;
            }
            let i = index.len();
            index.insert(op.clone(), i);
            i
        };

        let mut value_pairs = Vec::new();
        let mut conflict = HashSet::new();
        for dep in deps {
            let p = intern(&dep.producer);
            let c = intern(&dep.consumer);
            match dep.kind {
                SemanticDependencyKind::Value => value_pairs.push((p, c)),
                // Both explicit effect ordering and resource aliasing prohibit
                // moving another operation across this edge. A directly
                // adjacent pair may still be fused in source order; callers use
                // this relation for the operations *between* that pair.
                SemanticDependencyKind::Effect | SemanticDependencyKind::Resource => {
                    conflict.insert((p, c));
                    conflict.insert((c, p));
                }
            }
        }

        let n = index.len();
        let mut value_succ = vec![Vec::new(); n];
        for (p, c) in value_pairs {
            value_succ[p].push(c);
        }

        Self {
            index,
            value_succ,
            conflict,
        }
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
