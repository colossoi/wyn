//! Owned EGIR incidences shared by analysis and checked contraction.
use super::{
    graph_ops, read_resources, screma, EGraph, GraphResource, ResourceAccess, SegResourceAccess, Semantic,
    SemanticOpId, SideEffectKind, SideEffectSite, Soac, SoacEffect, ValueId,
};
use crate::egir::ir::{BodySite, ResultDestination};
use crate::egir::types::PureOp;
use crate::egir::types::{CallEffects, EffectOp, ResultBinding, ValueKind};
use crate::types::TypeExt;
use crate::{LookupMap, SortedSet, StableMap};
use wyn_fusion::{Builder, Error, GroupId, OrderingReason, PortId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SourceValue {
    pub body: BodySite,
    pub value: ValueId,
}

pub(crate) type ScopeKey = (BodySite, crate::flow::BlockId);

pub(crate) enum Incidence {
    Boundary,
    Pure(Vec<PortId>),
    Project {
        base: PortId,
        path: Vec<usize>,
    },
}

pub(crate) struct ValueFact {
    pub scope: ScopeKey,
    pub value: ValueId,
    pub incidence: Incidence,
}

pub(crate) struct OperationFact<R: GraphResource> {
    pub scope: ScopeKey,
    pub site: SideEffectSite,
    pub semantic_id: Option<SemanticOpId>,
    pub captures: Vec<ValueId>,
    pub resources: Vec<SegResourceAccess<R>>,
}

pub(crate) struct Facts<R: GraphResource> {
    pub builder: Builder<ScopeKey, R>,
    pub operations: StableMap<GroupId, OperationFact<R>>,
    pub ports: LookupMap<(ScopeKey, ValueId), PortId>,
    pub values: LookupMap<PortId, ValueFact>,
    pub external: LookupMap<PortId, Vec<PortId>>,
    pub runtime_arrays: SortedSet<PortId>,
    pub storage_uses: Vec<PortId>,
}

impl<R: GraphResource + Copy + Ord> Facts<R> {
    pub fn new() -> Self {
        Self {
            builder: Builder::new(),
            operations: StableMap::new(),
            ports: LookupMap::new(),
            values: LookupMap::new(),
            external: LookupMap::new(),
            runtime_arrays: SortedSet::new(),
            storage_uses: vec![],
        }
    }

    pub fn add_body(
        &mut self,
        body: BodySite,
        graph: &EGraph<Semantic<R>>,
        observers: impl IntoIterator<Item = (crate::flow::BlockId, ValueId)>,
    ) -> Result<(), Error> {
        let mut producers = LookupMap::new();
        let mut results = Vec::new();
        let mut groups = Vec::new();
        for (block, contents) in &graph.skeleton.blocks {
            for (index, effect) in contents.side_effects.iter().enumerate() {
                let scope = (body, block);
                let fields = graph
                    .effect_result_binding(effect)
                    .map(|result| result.top_level_fields())
                    .unwrap_or_default();
                let group = self.builder.operation(scope, vec![], fields.len())?;
                let ports = self.builder.outputs(group)?.to_vec();
                for (field, port) in fields.into_iter().zip(ports) {
                    for (path, leaf) in field.destination_leaves_with_paths() {
                        let ResultDestination::ReturnValue(value) = leaf.destination() else {
                            continue;
                        };
                        let (leaf_port, incidence) = if path.is_empty() {
                            (port, Incidence::Boundary)
                        } else {
                            (
                                self.builder.value(vec![port])?,
                                Incidence::Project {
                                    base: port,
                                    path: path.to_vec(),
                                },
                            )
                        };
                        if matches!(body, BodySite::Entry(_))
                            && graph.nodes[*value].ty.contains_runtime_sized_composite_array()
                        {
                            self.runtime_arrays.insert(port);
                        }
                        producers.insert(*value, (block, leaf_port));
                        self.values.insert(
                            leaf_port,
                            ValueFact {
                                scope,
                                value: *value,
                                incidence,
                            },
                        );
                    }
                    results.push((block, port, field));
                }
                let captures = match &effect.kind {
                    SideEffectKind::Soac(SoacEffect(_, soac)) => soac.capture_nodes().collect(),
                    _ => vec![],
                };
                self.operations.insert(
                    group,
                    OperationFact {
                        scope,
                        site: SideEffectSite { block, index },
                        semantic_id: effect.kind.soac_id().copied(),
                        captures,
                        resources: resources(graph, effect),
                    },
                );
                groups.push(group);
            }
        }
        for &group in &groups {
            let op = &self.operations[&group];
            let scope = op.scope;
            let effect = graph.skeleton.effect(op.site);
            let inputs = graph_ops::effect_value_inputs(graph, effect)
                .into_iter()
                .map(|value| self.port(scope, graph, value, &producers, &results))
                .collect::<Result<_, _>>()?;
            self.builder.set_inputs(group, inputs)?;
            let captures = self.operations[&group].captures.iter().copied();
            let histogram_inputs = effect
                .operands
                .iter()
                .filter_map(|operand| operand.value())
                .filter(|_| matches!(effect.kind, SideEffectKind::Soac(SoacEffect(_, Soac::Hist(_)))));
            self.storage_uses
                .extend(captures.chain(histogram_inputs).map(|value| self.ports[&(scope, value)]));
            for access in &self.operations[&group].resources {
                self.builder.access(group, access.resource, access.access != ResourceAccess::Read)?;
            }
        }
        for (position, &before) in groups.iter().enumerate() {
            let left = &self.operations[&before];
            let a = graph.skeleton.effect(left.site);
            for &after in &groups[position + 1..] {
                let right = &self.operations[&after];
                if left.scope != right.scope {
                    continue;
                }
                let b = graph.skeleton.effect(right.site);
                let opaque = |effect: &super::SideEffect<Semantic<R>>| match &effect.kind {
                    SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => {
                        matches!(op.semantic_state(), screma::SemanticState::Serial)
                    }
                    SideEffectKind::Soac(SoacEffect(_, Soac::Hist(op))) => {
                        matches!(op.state, super::hist::SemanticState::Serial)
                    }
                    SideEffectKind::Soac(_) => false,
                    SideEffectKind::Effect(EffectOp::Call { site }) => {
                        !matches!(graph.call(*site).effects(), CallEffects::Pure)
                    }
                    SideEffectKind::Effect(_) => true,
                };
                if opaque(a) || opaque(b) {
                    self.builder.order(before, after, OrderingReason::Opaque)?;
                }
                if matches!((a.effects, b.effects), (Some((_, output)), Some((input, _))) if output == input)
                {
                    self.builder.order(before, after, OrderingReason::Effect)?;
                }
            }
        }
        for (block, value) in graph
            .skeleton
            .blocks
            .iter()
            .flat_map(|(block, contents)| {
                contents.term.referenced_nodes().into_iter().map(move |value| (block, value))
            })
            .chain(observers)
        {
            let port = self.port((body, block), graph, value, &producers, &results)?;
            self.builder.observe(port)?;
        }
        let incoming = crate::egir::slice::incoming_arguments(graph);
        for (&(scope, value), &port) in &self.ports {
            if scope.0 == body {
                if let Some(inputs) = incoming.get(&value) {
                    let inputs = inputs.iter().map(|(block, value)| self.ports[&((body, *block), *value)]);
                    self.external.entry(port).or_default().extend(inputs);
                }
            }
        }
        Ok(())
    }

    fn port(
        &mut self,
        scope: ScopeKey,
        graph: &EGraph<Semantic<R>>,
        value: ValueId,
        producers: &LookupMap<ValueId, (crate::flow::BlockId, PortId)>,
        results: &[(
            crate::flow::BlockId,
            PortId,
            ResultBinding<polytype::Type<crate::ast::TypeName>>,
        )],
    ) -> Result<PortId, Error> {
        if let Some(port) = self.ports.get(&(scope, value)) {
            return Ok(*port);
        }
        let producer = producers.get(&value).copied().or_else(|| {
            results
                .iter()
                .find(|(_, _, result)| graph.value_has_result_origin(value, result))
                .map(|(block, port, _)| (*block, *port))
        });
        let (port, incidence) = if let Some((block, port)) = producer {
            if block == scope.1 {
                self.ports.insert((scope, value), port);
                return Ok(port);
            }
            self.builder.observe(port)?;
            let input = self.builder.input();
            self.external.insert(input, vec![port]);
            (input, Incidence::Boundary)
        } else {
            let node = graph.nodes.get(value).ok_or(Error::Port)?;
            if let Some(alias) = node.alias.or_else(|| graph_ops::projected_tuple_field(graph, value)) {
                let port = self.port(scope, graph, alias, producers, results)?;
                self.ports.insert((scope, value), port);
                return Ok(port);
            }
            let inputs = match node.kind() {
                ValueKind::BlockParam { .. } | ValueKind::FuncParam { .. } | ValueKind::Constant(_) => {
                    vec![]
                }
                _ => crate::egir::slice::value_inputs(graph, value),
            }
            .into_iter()
            .map(|value| self.port(scope, graph, value, producers, results))
            .collect::<Result<Vec<_>, _>>()?;
            if matches!(
                node.kind(),
                ValueKind::Pure {
                    op: PureOp::Index,
                    ..
                }
            ) {
                self.storage_uses.extend(inputs.first());
            }
            let port = if inputs.is_empty() {
                self.builder.input()
            } else {
                self.builder.value(inputs.clone())?
            };
            (port, Incidence::Pure(inputs))
        };
        self.ports.insert((scope, value), port);
        self.values.insert(
            port,
            ValueFact {
                scope,
                value,
                incidence,
            },
        );
        Ok(port)
    }
}

fn resources<R: GraphResource + Copy + Ord>(
    graph: &EGraph<Semantic<R>>,
    effect: &super::SideEffect<Semantic<R>>,
) -> Vec<SegResourceAccess<R>> {
    match &effect.kind {
        SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => match op.semantic_state() {
            screma::SemanticState::Segmented { resources, .. } => resources.clone(),
            screma::SemanticState::Serial => read_resources(graph, effect),
        },
        SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) => op.state.resources.clone(),
        SideEffectKind::Soac(SoacEffect(_, Soac::Hist(op))) => {
            let mut resources = read_resources(graph, effect);
            for destination in op
                .form
                .operations
                .iter()
                .flat_map(|op| &op.destinations)
                .filter_map(|view| graph_ops::extract_storage_view_source(graph, view.value()))
            {
                if let Some(access) = resources.iter_mut().find(|access| access.resource == destination) {
                    access.access = ResourceAccess::ReadWrite;
                } else {
                    resources.push(SegResourceAccess {
                        resource: destination,
                        access: ResourceAccess::ReadWrite,
                    });
                }
            }
            resources
        }
        SideEffectKind::Effect(_) => read_resources(graph, effect),
    }
}
