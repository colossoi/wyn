//! Shared semantic dependency facts, read-side analysis, and EGIR validation.

#![deny(clippy::let_underscore_must_use)]

mod facts;
pub(crate) use facts::{Facts, Incidence, ScopeKey, SourceValue};

use super::analysis::GraphAnalysis;
use super::ir::BodySite;
use crate::{LookupMap, SortedSet, StableMap};
use std::collections::HashSet;

use super::graph_ops;
use super::ir::{GraphResource, ProgramShape};
use super::program::{Program, SemanticOpId};
use super::soac::{hist, screma};
use super::types::{
    EGraph, ResourceAccess, SegResourceAccess, Semantic, SideEffect, SideEffectKind, SideEffectSite, Soac,
    SoacEffect, ValueId,
};

pub(crate) fn read_resources<R>(
    analysis: &GraphAnalysis<'_, Semantic<R>>,
    se: &SideEffect<Semantic<R>>,
) -> Vec<SegResourceAccess<R>>
where
    R: GraphResource + Copy + Ord,
{
    graph_ops::read_storage_resources(analysis, graph_ops::effect_value_inputs(analysis.graph(), se))
}

/// Validate the semantic boundary before any target-aware scheduling occurs.
pub(crate) fn verify<Tag, Shape, GlobalContext, R>(
    inner: &Program<Tag, Shape, GlobalContext>,
) -> Result<(), String>
where
    Shape: ProgramShape<Family = Semantic<R>>,
    R: GraphResource + Copy + Ord,
{
    let contains_region = |region| inner.functions.iter().any(|function| function.region == region);
    let verify_effect = |scope: &str,
                         graph: &EGraph<Semantic<R>>,
                         effect: &SideEffect<Semantic<R>>|
     -> Result<(), String> {
        let SideEffectKind::Soac(SoacEffect(_, soac)) = &effect.kind else {
            return Ok(());
        };
        let verify_body = |family: &str, body: &super::types::SegBody| {
            if contains_region(body.region) {
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
                op.validate_with_nodes(|node| graph.nodes.get(node).map(|node| node.ty.clone()))
                    .map_err(|error| format!("{scope}: {error}"))?;
                if let Some(body) = op.form.pre.seg_body() {
                    verify_body("Screma pre-lambda", body)?;
                }
                for scan in &op.form.scans {
                    verify_body(
                        "Screma scan operator",
                        scan.operator.seg_body().expect("validated scan operator has a region"),
                    )?;
                }
                for reduction in &op.form.reductions {
                    verify_body(
                        "Screma reduction operator",
                        reduction.operator.seg_body().expect("validated reduction operator has a region"),
                    )?;
                }
                if let Some(body) = op.form.post.seg_body() {
                    verify_body("Screma post-lambda", body)?;
                }
            }
            Soac::Filter(op) => {
                if let Some(body) = op.body.map.seg_body() {
                    verify_body("filter map", body)?;
                }
                verify_body(
                    "filter predicate",
                    op.body
                        .predicate
                        .seg_body()
                        .ok_or_else(|| format!("{scope}: filter predicate cannot be identity"))?,
                )?;
                op.body.validate().map_err(|error| format!("{scope}: {error}"))?;
            }
            Soac::Hist(op) => {
                if effect.operands.len() != op.inputs.len() {
                    return Err(format!(
                        "{scope}: Hist requires {} input operands, found {}",
                        op.inputs.len(),
                        effect.operands.len(),
                    ));
                }
                if let Some(body) = op.form.bucket.seg_body() {
                    verify_body("histogram bucket", body)?;
                }
                for (index, operation) in op.form.operations.iter().enumerate() {
                    if let hist::Update::Reduce { operator, .. } = &operation.update {
                        let body = operator.seg_body().ok_or_else(|| {
                            format!("{scope}: histogram reducer {index} cannot be identity")
                        })?;
                        verify_body("histogram reducer", body)?;
                    }
                }
                op.validate(|node| graph.nodes.get(node).map(|node| node.ty.clone()))
                    .map_err(|error| format!("{scope}: {error}"))?;
            }
        }
        Ok(())
    };
    for entry in &inner.entry_points {
        for (_, block) in &entry.graph.skeleton.blocks {
            for effect in &block.side_effects {
                verify_effect(&format!("entry `{}`", entry.name), &entry.graph, effect)?;
            }
        }
    }
    for function in &inner.functions {
        for (_, block) in &function.graph.skeleton.blocks {
            for effect in &block.side_effects {
                verify_effect(&format!("function `{}`", function.name), &function.graph, effect)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn summary<Tag, Shape, GlobalContext, R>(inner: &Program<Tag, Shape, GlobalContext>) -> String
where
    Shape: ProgramShape<Family = Semantic<R>>,
    R: GraphResource + Copy + Ord,
{
    let mut output = String::new();
    for entry in &inner.entry_points {
        write_graph_summary(&mut output, &format!("entry {}", entry.name), &entry.graph);
    }
    for function in &inner.functions {
        write_graph_summary(
            &mut output,
            &format!("function {}", function.name),
            &function.graph,
        );
    }
    output
}

#[cfg(test)]
pub(crate) fn write_graph_summary<R>(output: &mut String, scope: &str, graph: &EGraph<Semantic<R>>)
where
    R: GraphResource + Copy + Ord,
{
    for (_, block) in &graph.skeleton.blocks {
        for effect in &block.side_effects {
            match &effect.kind {
                SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => {
                    output.push_str(&format!(
                        "{scope}: Screma state={:?} inputs={:?} pre={:?} scans={:?} reductions={:?} post={:?}\n",
                        op.semantic_state(),
                        op.inputs,
                        op.form.pre,
                        op.form.scans,
                        op.form.reductions,
                        op.form.post,
                    ));
                }
                SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) => {
                    output.push_str(&format!(
                        "{scope}: Filter state={:?} inputs={:?} map={:?} predicate={:?}\n",
                        op.state, op.body.inputs, op.body.map, op.body.predicate
                    ));
                }
                SideEffectKind::Soac(SoacEffect(_, Soac::Hist(op))) => {
                    output.push_str(&format!(
                        "{scope}: Hist state={:?} inputs={:?} bucket={:?} operations={:?}\n",
                        op.state, op.inputs, op.form.bucket, op.form.operations
                    ));
                }
                SideEffectKind::Effect(_) => {}
            }
        }
    }
}

/// Read-side queries over the same incidences used by fusion.
#[derive(Default)]
pub struct SemanticGraph {
    consumers: LookupMap<SemanticOpId, SortedSet<SemanticOpId>>,
    captures: StableMap<SourceValue, SortedSet<SemanticOpId>>,
    sites: LookupMap<SemanticOpId, SideEffectSite>,
    pub(crate) array_residency_demands: HashSet<SemanticOpId>,
}

impl SemanticGraph {
    pub fn new<R: GraphResource + Copy + Ord>(graph: &EGraph<Semantic<R>>) -> Self {
        Self::for_bodies([(BodySite::Entry(0), &GraphAnalysis::new(graph))])
    }

    pub(crate) fn for_bodies<'a, R: GraphResource + Copy + Ord + 'a>(
        bodies: impl IntoIterator<Item = (BodySite, &'a GraphAnalysis<'a, Semantic<R>>)>,
    ) -> Self {
        let bodies = bodies.into_iter().collect::<Vec<_>>();
        // This index records SOAC dependencies. Without any SOACs it is empty,
        // but mixed inputs still need ordinary effects to connect their producers.
        if !bodies.iter().any(|(_, analysis)| {
            analysis
                .graph()
                .skeleton
                .blocks
                .values()
                .flat_map(|block| &block.side_effects)
                .any(|effect| effect.kind.soac_id().is_some())
        }) {
            return Self::default();
        }
        let mut facts = Facts::new();
        for (body, analysis) in bodies {
            facts.add_body(body, analysis, []).expect("valid EGIR incidences");
        }
        Self::from_facts(facts)
    }

    fn from_facts<R: GraphResource + Copy + Ord>(facts: Facts<R>) -> Self {
        let graph = facts.builder.finish(facts.operations).expect("one fact per operation");
        let producers = |roots: Vec<_>| {
            let mut pending = roots;
            let mut visited = HashSet::new();
            let mut producers = SortedSet::new();
            while let Some(port) = pending.pop() {
                if !visited.insert(port) {
                    continue;
                }
                pending.extend(facts.external.get(&port).into_iter().flatten());
                match facts.values.get(&port).map(|fact| &fact.incidence) {
                    Some(Incidence::Pure(inputs)) => pending.extend(inputs),
                    Some(Incidence::Project { base, .. }) => pending.push(*base),
                    _ => {
                        for producer in graph.producers(port).unwrap() {
                            producers.insert(producer);
                            pending.extend(graph.group(producer).unwrap().inputs());
                        }
                    }
                }
            }
            producers
        };
        let mut index = Self::default();
        for (_, group) in graph.groups() {
            let op = group.payload();
            let Some(id) = op.semantic_id else { continue };
            index.sites.insert(id, op.site);
            for &capture in &op.captures {
                index
                    .captures
                    .entry(SourceValue {
                        body: op.scope.0,
                        value: capture,
                    })
                    .or_default()
                    .insert(id);
            }
            for producer in producers(group.inputs().to_vec()) {
                if let Some(producer) = graph
                    .group(producer)
                    .unwrap()
                    .payload()
                    .semantic_id
                    .filter(|producer| *producer != id && index.sites.contains_key(producer))
                {
                    index.consumers.entry(producer).or_default().insert(id);
                }
            }
        }
        for producer in producers(facts.storage_uses) {
            let group = graph.group(producer).unwrap();
            if group.outputs().iter().any(|port| facts.runtime_arrays.contains(port)) {
                index.array_residency_demands.extend(group.payload().semantic_id);
            }
        }
        index
    }

    pub(crate) fn captured_values(&self, body: BodySite) -> impl Iterator<Item = ValueId> + '_ {
        self.captures.keys().filter(move |source| source.body == body).map(|source| source.value)
    }

    pub(crate) fn capture_consumers(&self, source: SourceValue) -> impl Iterator<Item = SemanticOpId> + '_ {
        self.captures.get(&source).into_iter().flatten().copied()
    }

    pub(crate) fn operation_site(&self, operation: &SemanticOpId) -> Option<SideEffectSite> {
        self.sites.get(operation).copied()
    }

    pub fn value_consumer_count(&self, producer: &SemanticOpId) -> usize {
        self.consumers.get(producer).map_or(0, SortedSet::len)
    }

    pub(crate) fn value_consumers(
        &self,
        producer: &SemanticOpId,
    ) -> impl Iterator<Item = SemanticOpId> + '_ {
        self.consumers.get(producer).into_iter().flatten().copied()
    }
}

#[cfg(test)]
#[path = "semantic_graph_tests.rs"]
mod semantic_graph_tests;
