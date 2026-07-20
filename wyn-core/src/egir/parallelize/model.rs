//! Shared target-planning policy, decisions, errors, and immutable indexes.

use std::collections::BTreeMap;

use thiserror::Error;

use super::schedule::KernelMutationError;
use crate::egir::program::{
    CompilerFlowEndpoint, CompilerResourceFlow, LogicalResourceArena, ResourceOrigin,
};

pub(super) const REDUCE_PHASE1_WIDTH: u32 = 64;
pub(super) const REDUCE_PHASE2_WIDTH: u32 = 256;
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

/// Disjoint sets used to collect connected outputs and kernel pipelines.
/// Roots store the negative component size; other entries store a parent.
pub(super) struct DisjointSets {
    links: Vec<isize>,
}

impl DisjointSets {
    pub(super) fn new(len: usize) -> Self {
        Self { links: vec![-1; len] }
    }

    pub(super) fn representative(&mut self, index: usize) -> usize {
        let parent = self.links[index];
        if parent < 0 {
            return index;
        }
        let root = self.representative(parent as usize);
        self.links[index] = root as isize;
        root
    }

    pub(super) fn merge(&mut self, left: usize, right: usize) {
        let (left, right) = (self.representative(left), self.representative(right));
        if left == right {
            return;
        }
        let (larger, smaller) =
            if self.links[left] <= self.links[right] { (left, right) } else { (right, left) };
        self.links[larger] += self.links[smaller];
        self.links[smaller] = larger as isize;
    }
}

/// Candidate analysis either selects a target recipe or explains why the
/// operation must use fallback lowering.
pub(super) enum CandidateSelection<T> {
    Selected(T),
    Fallback,
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
        flows.sort_by_key(|(resource, _)| *resource);
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
