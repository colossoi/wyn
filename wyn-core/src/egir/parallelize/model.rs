//! Shared target-planning policy, decisions, errors, and immutable indexes.

use std::collections::{BTreeMap, HashMap};

use thiserror::Error;

use super::schedule::KernelMutationError;
use crate::egir::program::{
    CompilerFlowEndpoint, CompilerResourceFlow, CompilerResourceKind, LogicalResource, ResourceOrigin,
    SemanticOpId,
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
    SequentialPolicy,
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

impl<T> RecipeSelection<T> {
    pub(super) fn without_payload(&self) -> RecipeSelection<()> {
        match self {
            Self::Parallel(_) => RecipeSelection::Parallel(()),
            Self::Serial(reason) => RecipeSelection::Serial(*reason),
        }
    }
}

/// Checked view of resources grouped by semantic owner, kind, and slot.
pub(super) struct ResourceIndex<'a> {
    resources: &'a [LogicalResource],
    owned: HashMap<(SemanticOpId, CompilerResourceKind), Vec<&'a LogicalResource>>,
}

impl<'a> ResourceIndex<'a> {
    pub(super) fn new(resources: &'a [LogicalResource]) -> Result<Self> {
        let mut owned: HashMap<_, Vec<_>> = HashMap::new();
        for (index, resource) in resources.iter().enumerate() {
            if resource.id.0 as usize != index {
                return Err(ParallelizeError::Invalid(format!(
                    "resource manifest is not dense at {:?}",
                    resource.id
                )));
            }
            let ResourceOrigin::Compiler(compiler) = &resource.origin else {
                continue;
            };
            if let Some(owner) = compiler.owner {
                owned.entry((owner, compiler.kind)).or_default().push(resource);
            }
        }
        for values in owned.values_mut() {
            values.sort_by_key(|resource| match &resource.origin {
                ResourceOrigin::Compiler(compiler) => compiler.slot,
                ResourceOrigin::Host(_) => usize::MAX,
            });
            for pair in values.windows(2) {
                let slots = pair.iter().map(|resource| match &resource.origin {
                    ResourceOrigin::Compiler(compiler) => Ok(compiler.slot),
                    ResourceOrigin::Host(_) => Err(ParallelizeError::Invalid(
                        "host resource appeared in compiler ownership index".into(),
                    )),
                });
                let slots = slots.collect::<Result<Vec<_>>>()?;
                if slots[0] == slots[1] {
                    return Err(ParallelizeError::Invalid(format!(
                        "compiler resource ownership bucket has duplicate slot {}",
                        slots[0]
                    )));
                }
            }
        }
        Ok(Self { resources, owned })
    }

    pub(super) fn get(&self, id: crate::ResourceId) -> Result<&'a LogicalResource> {
        self.resources
            .get(id.0 as usize)
            .filter(|resource| resource.id == id)
            .ok_or_else(|| ParallelizeError::Invalid(format!("missing logical resource {id:?}")))
    }

    pub(super) fn owned(&self, owner: SemanticOpId, kind: CompilerResourceKind) -> &[&'a LogicalResource] {
        self.owned.get(&(owner, kind)).map(Vec::as_slice).unwrap_or(&[])
    }

    pub(super) fn exactly_one(
        &self,
        owner: SemanticOpId,
        kind: CompilerResourceKind,
    ) -> Result<&'a LogicalResource> {
        let resources = self.owned(owner, kind);
        match resources {
            [resource] => Ok(*resource),
            _ => Err(ParallelizeError::Invalid(format!(
                "semantic operation {owner:?} requires exactly one {kind:?} resource, found {}",
                resources.len()
            ))),
        }
    }

    pub(super) fn exactly_one_at(
        &self,
        owner: SemanticOpId,
        kind: CompilerResourceKind,
        slot: usize,
    ) -> Result<&'a LogicalResource> {
        let resource = self.exactly_one(owner, kind)?;
        let ResourceOrigin::Compiler(compiler) = &resource.origin else {
            return Err(ParallelizeError::Invalid(
                "host resource appeared in compiler ownership index".into(),
            ));
        };
        if compiler.slot != slot {
            return Err(ParallelizeError::Invalid(format!(
                "semantic operation {owner:?} requires {kind:?} at slot {slot}, found slot {}",
                compiler.slot
            )));
        }
        Ok(resource)
    }

    pub(super) fn ordered_slots(
        &self,
        owner: SemanticOpId,
        kind: CompilerResourceKind,
        first_slot: usize,
        count: usize,
    ) -> Result<&[&'a LogicalResource]> {
        let resources = self.owned(owner, kind);
        if resources.len() != count {
            return Err(ParallelizeError::Invalid(format!(
                "semantic operation {owner:?} requires {count} {kind:?} resources, found {}",
                resources.len()
            )));
        }
        for (offset, resource) in resources.iter().enumerate() {
            let ResourceOrigin::Compiler(compiler) = &resource.origin else {
                return Err(ParallelizeError::Invalid(
                    "host resource appeared in compiler ownership index".into(),
                ));
            };
            let expected = first_slot + offset;
            if compiler.slot != expected {
                return Err(ParallelizeError::Invalid(format!(
                    "semantic operation {owner:?} requires {kind:?} slot {expected}, found {}",
                    compiler.slot
                )));
            }
        }
        Ok(resources)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn optional(
        &self,
        owner: SemanticOpId,
        kind: CompilerResourceKind,
    ) -> Result<Option<&'a LogicalResource>> {
        match self.owned(owner, kind) {
            [] => Ok(None),
            [resource] => Ok(Some(*resource)),
            resources => Err(ParallelizeError::Invalid(format!(
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
    pub(super) fn new(resources: &[LogicalResource]) -> Self {
        let mut flows = resources
            .iter()
            .filter_map(|resource| match &resource.origin {
                ResourceOrigin::Compiler(compiler) => compiler.flow.clone().map(|flow| (resource.id, flow)),
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
