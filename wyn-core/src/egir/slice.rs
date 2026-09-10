//! Dependency incidences for an unchanged EGIR snapshot.
use super::analysis::GraphAnalysis;
use super::block_interface;
use super::soac::metadata::Metadata;
use super::types::{
    CallEffects, EGraph, EffectOp, Family, GraphResource, OperandRef, PureOp, Raw, SegBody, Semantic,
    SideEffect, SideEffectKind, SideEffectSite, SoacEffect, ValueId, ValueKind, WynSoacPhase,
};
use crate::flow::BlockId;
use std::collections::HashSet;
use wyn_slice::{Definition, Graph, Node, Observer};

pub(crate) type LiveSlice = wyn_slice::LiveSlice<ValueId, SideEffectSite>;

pub(crate) struct SliceFacts {
    pub graph: Graph<ValueId, SideEffectSite, BlockId>,
    inputs: HashSet<ValueId>,
}

pub(crate) struct ValueObservers {
    effects: HashSet<SideEffectSite>,
    terminators: HashSet<BlockId>,
}

impl ValueObservers {
    pub fn effect_sites(&self) -> impl Iterator<Item = SideEffectSite> + '_ {
        self.effects.iter().copied()
    }
    pub fn terminator_blocks(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.terminators.iter().copied()
    }
}

impl SliceFacts {
    pub(super) fn build<P: ValueProducerPhase>(analysis: &GraphAnalysis<'_, P>) -> Self {
        let graph = analysis.graph();
        let producers = analysis.producers();
        let interfaces = analysis.interfaces().expect("valid block interfaces for slicing");
        let mut inputs = HashSet::new();
        let values = graph
            .nodes
            .iter()
            .filter_map(|(value, node)| {
                let definition =
                    if let Some(alias) = node.alias.or_else(|| projected_tuple_field(graph, value)) {
                        Definition::Pure(vec![alias])
                    } else {
                        match node.kind() {
                            ValueKind::FuncParam { .. } => Definition::Input,
                            ValueKind::BlockParam { block, index } => {
                                let interface = interfaces.get(block)?;
                                interface
                                    .columns()
                                    .get(*index)
                                    .filter(|column| column.parameter().value() == value)?;
                                if interface.edges().is_empty() {
                                    Definition::Input
                                } else {
                                    Definition::Flow(
                                        block_interface::dependencies(interface, *index)
                                            .map(|(_, value)| value)
                                            .collect(),
                                    )
                                }
                            }
                            ValueKind::SideEffectResult => Definition::Produced(producers.site(value)?),
                            ValueKind::CallResult { call, .. } => {
                                Definition::Produced(producers.call_site(*call)?)
                            }
                            _ => Definition::Pure(value_inputs(graph, value)),
                        }
                    };
                if matches!(definition, Definition::Input) {
                    inputs.insert(value);
                }
                Some((value, definition))
            })
            .collect::<Vec<_>>();
        let operations = graph.skeleton.blocks.iter().flat_map(|(block, body)| {
            body.side_effects.iter().enumerate().map(move |(index, effect)| {
                (
                    SideEffectSite { block, index },
                    P::effect_value_inputs(graph, effect),
                )
            })
        });
        let observers = graph
            .skeleton
            .blocks
            .iter()
            .map(|(block, body)| (block, body.term.referenced_nodes().into_vec()));
        Self {
            graph: Graph::new(values, operations, observers),
            inputs,
        }
    }

    pub fn select(
        &self,
        values: impl IntoIterator<Item = ValueId>,
        operations: impl IntoIterator<Item = SideEffectSite>,
        supplied: impl IntoIterator<Item = ValueId>,
        allowed: impl Fn(Node<ValueId, SideEffectSite>) -> bool,
        retained: impl Fn(Observer<SideEffectSite, BlockId>) -> bool,
        external: &[ValueId],
    ) -> Result<LiveSlice, String> {
        self.select_with_demands(values, operations, [], supplied, allowed, retained, external)
    }

    pub fn select_with_demands(
        &self,
        values: impl IntoIterator<Item = ValueId>,
        operations: impl IntoIterator<Item = SideEffectSite>,
        demands: impl IntoIterator<Item = ValueId>,
        supplied: impl IntoIterator<Item = ValueId>,
        allowed: impl Fn(Node<ValueId, SideEffectSite>) -> bool,
        retained: impl Fn(Observer<SideEffectSite, BlockId>) -> bool,
        external: &[ValueId],
    ) -> Result<LiveSlice, String> {
        let supplied = self
            .inputs
            .iter()
            .copied()
            .filter(|value| allowed(Node::Value(*value)))
            .chain(supplied)
            .collect();
        self.graph
            .select(
                values,
                operations.into_iter().map(Node::Operation).chain(demands.into_iter().map(Node::Value)),
                &supplied,
                allowed,
                retained,
                external,
            )
            .map_err(|error| format!("invalid live slice: {error:?}"))
    }

    pub fn pure_observers(&self, source: ValueId) -> ValueObservers {
        self.observers(source, false)
    }
    pub fn value_observers(&self, source: ValueId) -> ValueObservers {
        self.observers(source, true)
    }
    fn observers(&self, source: ValueId, full: bool) -> ValueObservers {
        let (effects, terminators) = self.graph.observers(source, full);
        ValueObservers { effects, terminators }
    }
    pub fn pure_reaches(&self, source: ValueId, user: ValueId) -> bool {
        self.graph.pure_reaches(source, user)
    }
}

pub(crate) fn value_inputs<P: Family>(graph: &EGraph<P>, value: ValueId) -> Vec<ValueId> {
    if let Some(alias) = graph.nodes[value].alias.or_else(|| projected_tuple_field(graph, value)) {
        vec![alias]
    } else {
        graph.value_dependencies(value)
    }
}

/// Phase-specific SOAC metadata that contributes to a produced value.
///
/// Raw SOACs expose every form-owned reference, including histogram shapes,
/// destinations and operator seeds. Semantic SOACs additionally expose their
/// resolved iteration space.
pub(crate) trait ValueProducerPhase: WynSoacPhase {
    fn effect_metadata_inputs(effect: &SideEffect<Self>) -> Vec<ValueId>;

    fn effect_value_inputs(graph: &EGraph<Self>, effect: &SideEffect<Self>) -> Vec<ValueId> {
        let mut values = graph.effect_boundary_value_dependencies(effect);
        if let SideEffectKind::Effect(EffectOp::Call { site }) = effect.kind() {
            if graph.call(*site).effects() == CallEffects::Pure {
                values.extend(graph.call_value_dependencies(*site));
            }
        }
        if let SideEffectKind::Soac(SoacEffect(_, soac)) = effect.kind() {
            for capture in soac.seg_bodies().into_iter().flat_map(SegBody::captures) {
                if let OperandRef::Place(place) = capture {
                    values.extend(graph.place_value_dependencies(*place));
                }
            }
        }
        values.extend(Self::effect_metadata_inputs(effect));
        values
    }
}

impl<R: GraphResource> ValueProducerPhase for Raw<R> {
    fn effect_metadata_inputs(effect: &SideEffect<Self>) -> Vec<ValueId> {
        let SideEffectKind::Soac(SoacEffect(_, soac)) = &effect.kind else {
            return Vec::new();
        };
        soac.metadata_values()
    }
}

impl<R: GraphResource> ValueProducerPhase for Semantic<R> {
    fn effect_metadata_inputs(effect: &SideEffect<Self>) -> Vec<ValueId> {
        effect.semantic_metadata_inputs()
    }
}

pub(crate) fn effect_value_inputs<P: ValueProducerPhase>(
    graph: &EGraph<P>,
    effect: &SideEffect<P>,
) -> Vec<ValueId> {
    P::effect_value_inputs(graph, effect)
}

/// Return the selected field when a projection is applied directly to a
/// structural tuple. Pure value-flow consumers need only that field.
pub(crate) fn projected_tuple_field<P: Family>(graph: &EGraph<P>, node: ValueId) -> Option<ValueId> {
    let ValueKind::Pure {
        op: PureOp::Project { index },
        operands,
    } = graph.nodes.get(node)?.kind()
    else {
        return None;
    };
    let [tuple] = operands.as_slice() else {
        return None;
    };
    let mut tuple = graph.nodes.get(*tuple)?;
    while let Some(alias) = tuple.alias {
        tuple = graph.nodes.get(alias)?;
    }
    let ValueKind::Pure {
        op: PureOp::Tuple(arity),
        operands: fields,
    } = tuple.kind()
    else {
        return None;
    };
    (*arity == fields.len()).then(|| fields.get(*index as usize).copied()).flatten()
}
