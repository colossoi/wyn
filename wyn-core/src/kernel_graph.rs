//! Final physical kernel metadata shared by compiler routes and SSA.

use crate::interface::OutputSlotId;
use crate::pipeline_descriptor::DispatchLen;
use crate::{EntryId, ResourceId, ResourceUse};
use std::collections::HashSet;
pub use wyn_kernel_graph::KernelId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputRouteProjection {
    pub semantic_slot: OutputSlotId,
    pub physical_slot: OutputSlotId,
}

/// Persistent physical kernel graph.
///
/// Kernel bodies remain in the surrounding physical program's entry arena and
/// are named here by [`EntryId`]. Both compiler routes retain kernel identity,
/// dependencies, dispatch, resource access, provenance, and output routing here.
#[derive(Clone, Debug, Default)]
pub struct PhysicalKernelGraph {
    kernels: Vec<PhysicalKernel>,
}

impl PhysicalKernelGraph {
    /// Readout of an already ordered plan from another compiler IR.
    pub(crate) fn from_ordered(kernels: Vec<PhysicalKernel>) -> Result<Self, String> {
        let mut seen = HashSet::new();
        for kernel in &kernels {
            if kernel.dependencies.iter().any(|id| !seen.contains(id)) {
                return Err("physical kernels are not in dependency order".into());
            }
            if !seen.insert(kernel.id) {
                return Err("duplicate physical kernel identity".into());
            }
        }
        let graph = Self { kernels };
        graph.validate()?;
        Ok(graph)
    }

    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    pub fn kernels(&self) -> impl ExactSizeIterator<Item = &PhysicalKernel> {
        self.kernels.iter()
    }

    /// Compatibility name for callers that present kernels as schedule
    /// phases. New compiler code should prefer [`Self::kernels`].
    pub fn phases(&self) -> impl ExactSizeIterator<Item = &PhysicalKernel> {
        self.kernels()
    }

    pub fn kernel(&self, id: KernelId) -> Option<&PhysicalKernel> {
        self.kernels.iter().find(|kernel| kernel.id == id)
    }

    /// Kernel identities in the immutable finalized dependency order.
    pub fn topological_kernel_ids(&self) -> Vec<KernelId> {
        self.kernels.iter().map(|kernel| kernel.id).collect()
    }

    /// Check the adapter-owned entry identities. Topology is established by
    /// the finalized kernel plan and cannot be mutated through this graph.
    pub fn validate(&self) -> Result<(), String> {
        let mut entries = HashSet::new();
        for kernel in &self.kernels {
            if !entries.insert(kernel.entry) {
                return Err(format!(
                    "physical entry {:?} is owned by multiple kernels",
                    kernel.entry
                ));
            }
        }
        Ok(())
    }

    #[cfg(feature = "egir")]
    pub(crate) fn validate_entry_ids(
        &self,
        entry_ids: impl IntoIterator<Item = EntryId>,
    ) -> Result<(), String> {
        let expected = self.kernels.iter().map(|kernel| kernel.entry).collect::<HashSet<_>>();
        let actual_ids = entry_ids.into_iter().collect::<Vec<_>>();
        let actual = actual_ids.iter().copied().collect::<HashSet<_>>();
        if actual.len() != actual_ids.len() {
            return Err("physical body arena repeats an entry identity".into());
        }
        if expected != actual {
            let mut missing = expected.difference(&actual).copied().collect::<Vec<_>>();
            let mut unowned = actual.difference(&expected).copied().collect::<Vec<_>>();
            missing.sort_unstable();
            unowned.sort_unstable();
            return Err(format!(
                "physical kernel/body ownership mismatch; missing bodies: {missing:?}; unowned bodies: {unowned:?}"
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct PhysicalKernel {
    pub id: KernelId,
    pub entry: EntryId,
    pub entry_point: String,
    pub label: String,
    pub source_entry: Option<EntryId>,
    pub output_routes: Vec<OutputRouteProjection>,
    pub workgroup_size: (u32, u32, u32),
    pub domain: KernelDomain,
    pub resources: Vec<ResourceUse<ResourceId>>,
    pub dependencies: Vec<KernelId>,
}

/// Concrete host-visible launch domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelDomain {
    /// Exactly the workgroup count recorded here.
    Fixed {
        x: u32,
        y: u32,
        z: u32,
    },
    /// One logical invocation per element of a concrete length source.
    Elements(DispatchLen),
    /// One invocation per element of a logical storage resource. The
    /// descriptor binding is resolved only while publishing a validated plan.
    ResourceElements {
        resource: ResourceId,
        elem_bytes: u32,
    },
}
