//! A small invariant-preserving staged dataflow graph.
//!
//! Stages own executable bodies. Resident flows are the only values that can
//! cross between stages, and external inputs are structurally distinct from
//! those flows. [`StagedIrBuilder`] may temporarily contain destinationless
//! flows while lowering, but [`StagedIrBuilder::finish`] rejects them and any
//! cyclic stage graph.

#![forbid(unsafe_code)]

use thiserror::Error;
use wyn_base::IdArena;
use wyn_graph::{topo_sort_by_dependencies, WalkOrder};

/// Opaque identity of one executable stage, issued by [`StagedIrBuilder`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StageId(StageKey);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct StageKey(u32);

impl From<u32> for StageKey {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

/// Opaque identity of one typed resident value flow, issued by [`StagedIrBuilder`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FlowId(FlowKey);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct FlowKey(u32);

impl From<u32> for FlowKey {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

/// One executable body and its staged incidence.
#[derive(Clone, Debug)]
pub struct Stage<Body, Origin> {
    origin: Origin,
    body: Body,
    incoming_flows: Vec<FlowId>,
    outgoing_flows: Vec<FlowId>,
}

impl<Body, Origin> Stage<Body, Origin> {
    pub fn origin(&self) -> &Origin {
        &self.origin
    }

    pub fn body(&self) -> &Body {
        &self.body
    }

    pub fn incoming_flows(&self) -> &[FlowId] {
        &self.incoming_flows
    }

    pub fn outgoing_flows(&self) -> &[FlowId] {
        &self.outgoing_flows
    }
}

/// A typed value with complete resident storage crossing stage boundaries.
#[derive(Clone, Debug)]
pub struct ResidentFlow<ValueType, Storage> {
    value_type: ValueType,
    storage: Storage,
    producer: StageId,
    consumers: Vec<StageId>,
    published: bool,
}

impl<ValueType, Storage> ResidentFlow<ValueType, Storage> {
    pub fn value_type(&self) -> &ValueType {
        &self.value_type
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub const fn producer(&self) -> StageId {
        self.producer
    }

    pub fn consumers(&self) -> &[StageId] {
        &self.consumers
    }

    pub const fn is_published(&self) -> bool {
        self.published
    }
}

/// A source-level input, deliberately not representable as a resident flow.
#[derive(Clone, Debug)]
pub struct ExternalInput<ValueType, Storage> {
    value_type: ValueType,
    storage: Storage,
    consumers: Vec<StageId>,
}

impl<ValueType, Storage> ExternalInput<ValueType, Storage> {
    pub fn value_type(&self) -> &ValueType {
        &self.value_type
    }

    pub fn consumers(&self) -> &[StageId] {
        &self.consumers
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }
}

/// Finalized target-independent staged IR.
#[derive(Clone, Debug)]
pub struct StagedIr<Body, ValueType, Storage, Origin = ()> {
    stages: IdArena<StageKey, Stage<Body, Origin>>,
    flows: IdArena<FlowKey, ResidentFlow<ValueType, Storage>>,
    external_inputs: Vec<ExternalInput<ValueType, Storage>>,
}

impl<Body, ValueType, Storage, Origin> StagedIr<Body, ValueType, Storage, Origin> {
    pub fn stage(&self, id: StageId) -> Option<&Stage<Body, Origin>> {
        self.stages.get(id.0)
    }

    pub fn flow(&self, id: FlowId) -> Option<&ResidentFlow<ValueType, Storage>> {
        self.flows.get(id.0)
    }

    pub fn stages(&self) -> impl Iterator<Item = (StageId, &Stage<Body, Origin>)> {
        self.stages.iter().map(|(&id, stage)| (StageId(id), stage))
    }

    pub fn flows(&self) -> impl Iterator<Item = (FlowId, &ResidentFlow<ValueType, Storage>)> {
        self.flows.iter().map(|(&id, flow)| (FlowId(id), flow))
    }

    pub fn external_inputs(&self) -> impl ExactSizeIterator<Item = &ExternalInput<ValueType, Storage>> {
        self.external_inputs.iter()
    }

    pub fn stage_body_mut(&mut self, id: StageId) -> Option<&mut Body> {
        self.stages.get_mut(id.0).map(|stage| &mut stage.body)
    }

    /// Change the body representation without rebuilding validated topology.
    pub fn map_stage_bodies<NewBody>(
        self,
        mut map: impl FnMut(StageId, Body) -> NewBody,
    ) -> StagedIr<NewBody, ValueType, Storage, Origin> {
        let mut stages = IdArena::new();
        for (key, stage) in self.stages {
            let id = StageId(key);
            let mapped_key = stages.alloc(Stage {
                origin: stage.origin,
                body: map(id, stage.body),
                incoming_flows: stage.incoming_flows,
                outgoing_flows: stage.outgoing_flows,
            });
            debug_assert_eq!(mapped_key, key, "stage mapping must retain arena identities");
        }
        StagedIr {
            stages,
            flows: self.flows,
            external_inputs: self.external_inputs,
        }
    }

    /// Stages in producer-before-consumer order.
    pub fn topological_stages(&self) -> Vec<StageId> {
        topo_sort_by_dependencies(self.stages().map(|(id, _)| id), |stage, dependencies| {
            dependencies.extend(
                self.stage(stage)
                    .into_iter()
                    .flat_map(Stage::incoming_flows)
                    .filter_map(|flow| self.flow(*flow).map(ResidentFlow::producer)),
            );
        })
        .unwrap_or_else(|_| unreachable!("the builder rejects cyclic staged graphs"))
    }
}

/// Lowering-only construction state for staged IR.
#[derive(Clone, Debug)]
pub struct StagedIrBuilder<Body, ValueType, Storage, Origin = ()> {
    ir: StagedIr<Body, ValueType, Storage, Origin>,
}

impl<Body, ValueType, Storage, Origin> Default for StagedIrBuilder<Body, ValueType, Storage, Origin> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Body, ValueType, Storage, Origin> StagedIrBuilder<Body, ValueType, Storage, Origin> {
    pub fn new() -> Self {
        Self {
            ir: StagedIr {
                stages: IdArena::new(),
                flows: IdArena::new(),
                external_inputs: Vec::new(),
            },
        }
    }

    pub fn stage_count(&self) -> usize {
        self.ir.stages.len()
    }

    pub fn stages(&self) -> impl Iterator<Item = (StageId, &Origin, &Body)> {
        self.ir.stages().map(|(id, stage)| (id, stage.origin(), stage.body()))
    }

    pub fn stage_body_mut(&mut self, id: StageId) -> Option<&mut Body> {
        self.ir.stage_body_mut(id)
    }

    pub fn flow(&self, id: FlowId) -> Option<&ResidentFlow<ValueType, Storage>> {
        self.ir.flow(id)
    }

    pub fn add_stage(&mut self, origin: Origin, body: Body) -> Result<StageId, BuildError> {
        ensure_id_capacity(self.ir.stages.len(), "stages")?;
        Ok(StageId(self.ir.stages.alloc(Stage {
            origin,
            body,
            incoming_flows: Vec::new(),
            outgoing_flows: Vec::new(),
        })))
    }

    pub fn add_flow(
        &mut self,
        producer: StageId,
        value_type: ValueType,
        storage: Storage,
    ) -> Result<FlowId, BuildError> {
        self.stage(producer)?;
        ensure_id_capacity(self.ir.flows.len(), "resident flows")?;
        let id = FlowId(self.ir.flows.alloc(ResidentFlow {
            value_type,
            storage,
            producer,
            consumers: Vec::new(),
            published: false,
        }));
        self.ir.stages[producer.0].outgoing_flows.push(id);
        Ok(id)
    }

    pub fn add_consumer(&mut self, flow: FlowId, consumer: StageId) -> Result<(), BuildError> {
        let producer = self.require_flow(flow)?.producer;
        self.stage(consumer)?;
        if producer == consumer {
            return Err(BuildError::SelfDependency { stage: producer });
        }
        if self.stage_reaches(consumer, producer) {
            return Err(BuildError::Cycle { producer, consumer });
        }
        if self.ir.flows[flow.0].consumers.contains(&consumer) {
            return Err(BuildError::DuplicateConsumer { flow, consumer });
        }
        self.ir.flows[flow.0].consumers.push(consumer);
        self.ir.stages[consumer.0].incoming_flows.push(flow);
        Ok(())
    }

    pub fn publish(&mut self, flow: FlowId) -> Result<(), BuildError> {
        self.require_flow(flow)?;
        self.ir.flows[flow.0].published = true;
        Ok(())
    }

    /// Add one external input and all stages that consume it.
    pub fn add_external_input(
        &mut self,
        value_type: ValueType,
        storage: Storage,
        consumers: impl IntoIterator<Item = StageId>,
    ) -> Result<(), BuildError> {
        let mut checked = Vec::new();
        for consumer in consumers {
            self.stage(consumer)?;
            if !checked.contains(&consumer) {
                checked.push(consumer);
            }
        }
        self.ir.external_inputs.push(ExternalInput {
            value_type,
            storage,
            consumers: checked,
        });
        Ok(())
    }

    pub fn finish(self) -> Result<StagedIr<Body, ValueType, Storage, Origin>, BuildError> {
        for (flow, value) in self.ir.flows() {
            if value.consumers.is_empty() && !value.published {
                return Err(BuildError::FlowHasNoDestination { flow });
            }
        }
        if topo_sort_by_dependencies(self.ir.stages().map(|(id, _)| id), |stage, dependencies| {
            dependencies.extend(
                self.ir
                    .stage(stage)
                    .into_iter()
                    .flat_map(Stage::incoming_flows)
                    .filter_map(|flow| self.ir.flow(*flow).map(ResidentFlow::producer)),
            );
        })
        .is_err()
        {
            return Err(BuildError::CycleDetected);
        }
        Ok(self.ir)
    }

    fn stage(&self, id: StageId) -> Result<&Stage<Body, Origin>, BuildError> {
        self.ir.stage(id).ok_or(BuildError::UnknownStage(id))
    }

    fn require_flow(&self, id: FlowId) -> Result<&ResidentFlow<ValueType, Storage>, BuildError> {
        self.ir.flow(id).ok_or(BuildError::UnknownFlow(id))
    }

    fn stage_reaches(&self, start: StageId, target: StageId) -> bool {
        wyn_graph::reaches_ordered(start, target, WalkOrder::DepthFirst, |stage, successors| {
            let Some(stage) = self.ir.stage(stage) else {
                return;
            };
            successors.extend(
                stage
                    .outgoing_flows()
                    .iter()
                    .filter_map(|flow| self.ir.flow(*flow))
                    .flat_map(ResidentFlow::consumers)
                    .copied(),
            );
        })
    }
}

fn ensure_id_capacity(length: usize, collection: &'static str) -> Result<(), BuildError> {
    if length < u32::MAX as usize {
        Ok(())
    } else {
        Err(BuildError::CapacityExceeded { collection })
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum BuildError {
    #[error("unknown stage {0:?}")]
    UnknownStage(StageId),
    #[error("unknown resident flow {0:?}")]
    UnknownFlow(FlowId),
    #[error("stage {stage:?} cannot consume its own resident flow")]
    SelfDependency {
        stage: StageId,
    },
    #[error("connecting stage {producer:?} to {consumer:?} would create a cycle")]
    Cycle {
        producer: StageId,
        consumer: StageId,
    },
    #[error("stage {consumer:?} already consumes resident flow {flow:?}")]
    DuplicateConsumer {
        flow: FlowId,
        consumer: StageId,
    },
    #[error("resident flow {flow:?} has no stage or external destination")]
    FlowHasNoDestination {
        flow: FlowId,
    },
    #[error("stage graph contains a cycle")]
    CycleDetected,
    #[error("{collection} exceeded the 32-bit identity space")]
    CapacityExceeded {
        collection: &'static str,
    },
}

#[cfg(test)]
mod tests;
