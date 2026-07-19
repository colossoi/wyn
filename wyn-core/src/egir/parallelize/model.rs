//! Shared target-planning policy, decisions, errors, and immutable indexes.

use std::collections::{BTreeMap, HashMap};

use thiserror::Error;

use super::schedule::KernelMutationError;
use crate::egir::program::{
    CompilerFlowEndpoint, CompilerResourceFlow, CompilerResourceKind, LogicalResource,
    LogicalResourceArena, ResourceOrigin, SemanticOpId,
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

/// Checked view of resources grouped by semantic owner, kind, and slot.
pub(super) struct ResourceIndex<'a> {
    resources: &'a LogicalResourceArena,
    owned: HashMap<(SemanticOpId, CompilerResourceKind), BTreeMap<usize, &'a LogicalResource>>,
}

impl<'a> ResourceIndex<'a> {
    pub(super) fn new(resources: &'a LogicalResourceArena) -> Result<Self> {
        let mut owned: HashMap<_, BTreeMap<_, _>> = HashMap::new();
        for resource in resources {
            let ResourceOrigin::Compiler(compiler) = &resource.origin else {
                continue;
            };
            if let Some(owner) = compiler.owner {
                let slots = owned.entry((owner, compiler.kind)).or_default();
                if slots.insert(compiler.slot, resource).is_some() {
                    return Err(ParallelizeError::Invalid(format!(
                        "compiler resource ownership bucket has duplicate slot {}",
                        compiler.slot
                    )));
                }
            }
        }
        Ok(Self { resources, owned })
    }

    pub(super) fn get(&self, id: crate::ResourceId) -> Result<&'a LogicalResource> {
        self.resources
            .get(id)
            .ok_or_else(|| ParallelizeError::Invalid(format!("missing logical resource {id:?}")))
    }

    #[cfg(test)]
    pub(super) fn owned(
        &self,
        owner: SemanticOpId,
        kind: CompilerResourceKind,
    ) -> Vec<&'a LogicalResource> {
        self.owned
            .get(&(owner, kind))
            .into_iter()
            .flat_map(|resources| resources.values().copied())
            .collect()
    }

    #[cfg(test)]
    pub(super) fn exactly_one(
        &self,
        owner: SemanticOpId,
        kind: CompilerResourceKind,
    ) -> Result<&'a LogicalResource> {
        let resources = self.owned.get(&(owner, kind));
        match resources.map(|resources| resources.values().copied().collect::<Vec<_>>()) {
            Some(resources) if resources.len() == 1 => Ok(resources[0]),
            _ => Err(ParallelizeError::Invalid(format!(
                "semantic operation {owner:?} requires exactly one {kind:?} resource, found {}",
                resources.map_or(0, BTreeMap::len)
            ))),
        }
    }

    pub(super) fn exactly_one_at(
        &self,
        owner: SemanticOpId,
        kind: CompilerResourceKind,
        slot: usize,
    ) -> Result<&'a LogicalResource> {
        let resources = self.owned.get(&(owner, kind));
        let resource = resources.and_then(|resources| resources.get(&slot)).copied();
        match (resources.map(BTreeMap::len), resource) {
            (Some(1), Some(resource)) => Ok(resource),
            (count, _) => Err(ParallelizeError::Invalid(format!(
                "semantic operation {owner:?} requires exactly one {kind:?} resource at slot {slot}, found {}",
                count.unwrap_or(0)
            ))),
        }
    }

    pub(super) fn ordered_slots(
        &self,
        owner: SemanticOpId,
        kind: CompilerResourceKind,
        first_slot: usize,
        count: usize,
    ) -> Result<Vec<&'a LogicalResource>> {
        let resources = self.owned.get(&(owner, kind));
        if resources.map_or(0, BTreeMap::len) != count {
            return Err(ParallelizeError::Invalid(format!(
                "semantic operation {owner:?} requires {count} {kind:?} resources, found {}",
                resources.map_or(0, BTreeMap::len)
            )));
        }
        let Some(resources) = resources else {
            return Ok(Vec::new());
        };
        let mut ordered = Vec::with_capacity(count);
        for offset in 0..count {
            let expected = first_slot + offset;
            let resource = resources.get(&expected).copied().ok_or_else(|| {
                ParallelizeError::Invalid(format!(
                    "semantic operation {owner:?} requires {kind:?} slot {expected}"
                ))
            })?;
            ordered.push(resource);
        }
        Ok(ordered)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn optional(
        &self,
        owner: SemanticOpId,
        kind: CompilerResourceKind,
    ) -> Result<Option<&'a LogicalResource>> {
        match self.owned.get(&(owner, kind)) {
            None => Ok(None),
            Some(resources) if resources.len() == 1 => Ok(resources.values().next().copied()),
            Some(resources) => Err(ParallelizeError::Invalid(format!(
                "semantic operation {owner:?} has {} optional {kind:?} resources",
                resources.len()
            ))),
        }
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
