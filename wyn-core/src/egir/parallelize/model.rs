//! Shared target-planning policy, decisions, errors, and immutable indexes.

use std::collections::BTreeMap;

use thiserror::Error;

use super::schedule::KernelMutationError;
use crate::egir::program::{
    CompilerFlowEndpoint, CompilerResourceFlow, LogicalResource, LogicalResourceArena, ResourceOrigin,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ParallelPolicy {
    pub(super) reduce_phase1_width: u32,
    pub(super) reduce_phase2_width: u32,
    pub(super) filter_scan_groups: u32,
}

impl Default for ParallelPolicy {
    fn default() -> Self {
        Self {
            reduce_phase1_width: REDUCE_PHASE1_WIDTH,
            reduce_phase2_width: REDUCE_PHASE2_WIDTH,
            filter_scan_groups: FILTER_SCAN_GROUPS,
        }
    }
}

pub(super) const REDUCE_PHASE1_WIDTH: u32 = 64;
const REDUCE_PHASE2_WIDTH: u32 = 256;
pub(super) const FILTER_SCAN_GROUPS: u32 = 4;

#[derive(Debug, Error)]
pub(super) enum ParallelizeError {
    #[error("{0}")]
    Invalid(String),
    #[error("kernel schedule mutation failed: {0}")]
    Schedule(#[from] KernelMutationError),
}

impl From<String> for ParallelizeError {
    fn from(value: String) -> Self {
        Self::Invalid(value)
    }
}

impl From<&str> for ParallelizeError {
    fn from(value: &str) -> Self {
        Self::Invalid(value.to_owned())
    }
}

pub(super) type Result<T> = std::result::Result<T, ParallelizeError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Why a valid semantic operation selected serial fallback.
pub(super) enum FallbackReason {
    UnsupportedPlacement,
    UnsupportedCaptures,
    UnsupportedViewShape,
    UnsupportedDestination,
    UnsupportedScratchLayout,
    UnsupportedOperationShape,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Immutable analysis outcome; only parallel variants may request scratch.
pub(super) enum RecipeSelection<T> {
    Parallel(T),
    Serial(FallbackReason),
}

/// Read-only resource lookup used while recognizing semantic views. Dense ids
/// and compiler ownership uniqueness are invariants of `LogicalResourceArena`.
pub(super) struct ResourceIndex<'a> {
    resources: &'a LogicalResourceArena,
}

impl<'a> ResourceIndex<'a> {
    pub(super) fn new(resources: &'a LogicalResourceArena) -> Self {
        Self { resources }
    }

    pub(super) fn get(&self, id: crate::ResourceId) -> Result<&'a LogicalResource> {
        self.resources
            .get(id)
            .ok_or_else(|| ParallelizeError::Invalid(format!("missing logical resource {id:?}")))
    }
}

/// Canonical compiler-flow edges shared by attachment and coalescing.
pub(super) struct ResourceFlowIndex {
    flows: Vec<(crate::ResourceId, CompilerResourceFlow)>,
    incoming: BTreeMap<CompilerFlowEndpoint, Vec<CompilerFlowEndpoint>>,
}

impl ResourceFlowIndex {
    pub(super) fn new(resources: &LogicalResourceArena) -> Self {
        let mut flows = resources
            .iter()
            .filter_map(|resource| match &resource.origin {
                ResourceOrigin::Compiler(compiler) => {
                    compiler.flow.clone().map(|flow| (resource.id(), flow))
                }
                ResourceOrigin::Host(_) => None,
            })
            .collect::<Vec<_>>();
        flows.sort_by_key(|(resource, _)| resource.0);
        let mut incoming = BTreeMap::<_, Vec<_>>::new();
        for (_, flow) in &flows {
            for consumer in &flow.consumers {
                incoming.entry(*consumer).or_default().push(flow.producer);
            }
        }
        for producers in incoming.values_mut() {
            producers.sort_unstable();
            producers.dedup();
        }
        Self { flows, incoming }
    }

    pub(super) fn flows(&self) -> &[(crate::ResourceId, CompilerResourceFlow)] {
        &self.flows
    }

    pub(super) fn incoming(&self, consumer: CompilerFlowEndpoint) -> &[CompilerFlowEndpoint] {
        self.incoming.get(&consumer).map(Vec::as_slice).unwrap_or(&[])
    }
}
