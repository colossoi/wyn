//! Planning-session indexes and deterministic recipe-owned scratch allocation.

#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::collections::{BTreeMap, HashMap, HashSet};

use polytype::Type;

use crate::ast::TypeName;

use super::error::{ParallelizeError, Result};
use super::policy::ParallelPolicy;
use crate::egir::program::{
    AllocatedProgram, CompilerFlowEndpoint, CompilerResource, CompilerResourceFlow, CompilerResourceKind,
    LogicalResource, LogicalSize, ResourceOrigin, SemanticOpId,
};
use crate::egir::types::SegExtent;

/// Immutable lookup view over the final logical resource manifest used by one
/// planning session. Owner/kind buckets are ordered by compiler slot.
pub(crate) struct ResourceIndex<'a> {
    resources: &'a [LogicalResource],
    owned: HashMap<(SemanticOpId, CompilerResourceKind), Vec<&'a LogicalResource>>,
}

/// Canonical indexed view of compiler producer/consumer edges for one planning
/// session. Both materialization attachment and phase dependency coalescing
/// consume this same data.
pub(crate) struct ResourceFlowIndex {
    flows: Vec<(crate::ResourceId, CompilerResourceFlow)>,
    incoming: BTreeMap<CompilerFlowEndpoint, Vec<CompilerFlowEndpoint>>,
}

/// Authoritative record of the recipes selected during immutable preflight.
/// Emission may rebuild short-lived graph handles, but it must not reconsider
/// whether an operation is parallel or serial after scratch has been added.
#[derive(Default)]
pub(crate) struct CandidateIndex {
    filters: HashSet<SemanticOpId>,
    reduces: HashSet<SemanticOpId>,
    scans: HashSet<SemanticOpId>,
}

impl CandidateIndex {
    pub(crate) fn filter(&self, owner: SemanticOpId) -> bool {
        self.filters.contains(&owner)
    }

    pub(crate) fn reduce(&self, owner: SemanticOpId) -> bool {
        self.reduces.contains(&owner)
    }

    pub(crate) fn scan(&self, owner: SemanticOpId) -> bool {
        self.scans.contains(&owner)
    }

    fn record(&mut self, request: &ScratchRequest) {
        let Some(owner) = request.compiler.owner else {
            return;
        };
        match request.compiler.kind {
            CompilerResourceKind::FilterFlags => {
                self.filters.insert(owner);
            }
            CompilerResourceKind::ReducePartial => {
                self.reduces.insert(owner);
            }
            CompilerResourceKind::ScanBlockSums => {
                self.scans.insert(owner);
            }
            _ => {}
        }
    }
}

impl ResourceFlowIndex {
    pub(crate) fn new(resources: &[LogicalResource]) -> Self {
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

    pub(crate) fn flows(&self) -> &[(crate::ResourceId, CompilerResourceFlow)] {
        &self.flows
    }

    pub(crate) fn incoming(&self, consumer: CompilerFlowEndpoint) -> &[CompilerFlowEndpoint] {
        self.incoming.get(&consumer).map(Vec::as_slice).unwrap_or(&[])
    }
}

impl<'a> ResourceIndex<'a> {
    pub(crate) fn new(resources: &'a [LogicalResource]) -> Result<Self> {
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

    pub(crate) fn get(&self, id: crate::ResourceId) -> Result<&'a LogicalResource> {
        self.resources
            .get(id.0 as usize)
            .filter(|resource| resource.id == id)
            .ok_or_else(|| ParallelizeError::Invalid(format!("missing logical resource {id:?}")))
    }

    pub(crate) fn owned(&self, owner: SemanticOpId, kind: CompilerResourceKind) -> &[&'a LogicalResource] {
        self.owned.get(&(owner, kind)).map(Vec::as_slice).unwrap_or(&[])
    }

    pub(crate) fn exactly_one(
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

    pub(crate) fn exactly_one_at(
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

    pub(crate) fn ordered_slots(
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
    pub(crate) fn optional(
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

struct ScratchRequest {
    endpoint: CompilerFlowEndpoint,
    compiler: CompilerResource,
    elem_ty: Type<TypeName>,
    size: LogicalSize,
}

/// Append target-recipe work buffers immediately before kernel planning.
/// Semantic residency resources have already been established; these buffers
/// exist only when the parallel policy is selected.
pub(crate) fn allocate_parallel_scratch(
    inner: &mut AllocatedProgram,
    policy: ParallelPolicy,
) -> Result<CandidateIndex> {
    let resource_index = ResourceIndex::new(&inner.resources)?;
    let mut requests = filter_requests(inner, policy);
    requests.extend(segmented_requests(inner, &resource_index)?);
    requests.sort_by_key(|request| {
        (
            request.endpoint,
            request.compiler.owner,
            resource_kind_rank(request.compiler.kind),
            request.compiler.slot,
        )
    });
    let mut candidates = CandidateIndex::default();

    for request in requests {
        candidates.record(&request);
        let existing = inner.resources.iter().find(|resource| {
            matches!(
                &resource.origin,
                ResourceOrigin::Compiler(existing)
                    if existing.kind == request.compiler.kind
                        && existing.owner == request.compiler.owner
                        && existing.slot == request.compiler.slot
            )
        });
        if let Some(existing) = existing {
            if existing.elem_ty != request.elem_ty || existing.size != request.size {
                return Err(ParallelizeError::Invalid(format!(
                    "planned scratch {:?} for {:?} slot {} changed layout",
                    request.compiler.kind, request.compiler.owner, request.compiler.slot
                )));
            }
            continue;
        }
        inner.alloc_compiler_resource(request.compiler, request.elem_ty, request.size);
    }
    Ok(candidates)
}

fn filter_requests(inner: &AllocatedProgram, policy: ParallelPolicy) -> Vec<ScratchRequest> {
    let mut requests = Vec::new();
    for (endpoint, entry) in inner.entries_with_endpoints() {
        let Some(candidate) = super::analyze_filter_candidate(entry) else {
            continue;
        };
        let element_count_size = match candidate.space.dims.first() {
            Some(SegExtent::Fixed(count)) if candidate.space.dims.len() == 1 => {
                LogicalSize::FixedBytes(*count as u64 * 4)
            }
            Some(SegExtent::ResourceLength {
                resource, elem_bytes, ..
            }) if candidate.space.dims.len() == 1 => LogicalSize::LikeResource {
                resource: resource.0,
                elem_bytes: 4,
                src_elem_bytes: *elem_bytes,
            },
            _ => LogicalSize::SameAsDispatch { elem_bytes: 4 },
        };
        let worker_count_size =
            LogicalSize::FixedBytes((policy.filter_scan_groups * policy.reduce_phase1_width) as u64 * 4);
        for (slot, (kind, size)) in [
            (CompilerResourceKind::FilterFlags, element_count_size.clone()),
            (CompilerResourceKind::FilterOffsets, element_count_size.clone()),
            (
                CompilerResourceKind::FilterScanBlockSums,
                worker_count_size.clone(),
            ),
            (
                CompilerResourceKind::FilterScanBlockOffsets,
                worker_count_size.clone(),
            ),
        ]
        .into_iter()
        .enumerate()
        {
            requests.push(ScratchRequest {
                endpoint,
                compiler: CompilerResource::new(kind, Some(candidate.semantic_id), slot),
                elem_ty: Type::Constructed(TypeName::UInt(32), vec![]),
                size,
            });
        }
    }
    requests
}

fn segmented_requests(
    inner: &AllocatedProgram,
    resources: &ResourceIndex<'_>,
) -> Result<Vec<ScratchRequest>> {
    let mut requests = Vec::new();
    for (endpoint, entry) in inner.entries_with_endpoints() {
        let projected = crate::egir::program::PlannedEntry::project(entry)?;
        if let Some(candidate) = super::analyze_reduce_candidate(&projected, resources) {
            for (slot, elem_ty) in candidate.scratch_types.into_iter().enumerate() {
                requests.push(scratch_request(
                    endpoint,
                    candidate.owner,
                    slot,
                    CompilerResourceKind::ReducePartial,
                    elem_ty,
                )?);
            }
        } else if let Some(candidate) = super::analyze_scan_candidate(&projected) {
            for (slot, kind) in [
                CompilerResourceKind::ScanBlockSums,
                CompilerResourceKind::ScanBlockOffsets,
            ]
            .into_iter()
            .enumerate()
            {
                requests.push(scratch_request(
                    endpoint,
                    candidate.owner,
                    slot,
                    kind,
                    candidate.scratch_type.clone(),
                )?);
            }
        }
    }
    Ok(requests)
}

fn scratch_request(
    endpoint: CompilerFlowEndpoint,
    owner: SemanticOpId,
    slot: usize,
    kind: CompilerResourceKind,
    elem_ty: Type<TypeName>,
) -> Result<ScratchRequest> {
    let elem_bytes = crate::ssa::layout::type_byte_size(&elem_ty).ok_or_else(|| {
        ParallelizeError::Invalid(format!(
            "parallel scratch for {owner:?} has no static element size"
        ))
    })? as u32;
    Ok(ScratchRequest {
        endpoint,
        compiler: CompilerResource::new(kind, Some(owner), slot),
        elem_ty,
        size: LogicalSize::SameAsDispatch { elem_bytes },
    })
}

fn resource_kind_rank(kind: CompilerResourceKind) -> u8 {
    match kind {
        CompilerResourceKind::ReducePartial => 0,
        CompilerResourceKind::ScanBlockSums => 1,
        CompilerResourceKind::ScanBlockOffsets => 2,
        CompilerResourceKind::FilterFlags => 3,
        CompilerResourceKind::FilterOffsets => 4,
        CompilerResourceKind::FilterScanBlockSums => 5,
        CompilerResourceKind::FilterScanBlockOffsets => 6,
        _ => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resource(id: u32, owner: u32, kind: CompilerResourceKind, slot: usize) -> LogicalResource {
        LogicalResource {
            id: crate::ResourceId(id),
            origin: ResourceOrigin::Compiler(CompilerResource::new(kind, Some(SemanticOpId(owner)), slot)),
            elem_ty: Type::Constructed(TypeName::UInt(32), vec![]),
            size: LogicalSize::FixedBytes(4),
        }
    }

    #[test]
    fn resource_index_checks_density_and_exact_cardinality() {
        let sparse = [resource(1, 7, CompilerResourceKind::ReducePartial, 0)];
        assert!(ResourceIndex::new(&sparse).is_err());

        let missing_slot = [resource(0, 7, CompilerResourceKind::ReducePartial, 1)];
        let missing_slot = ResourceIndex::new(&missing_slot).expect("dense manifest");
        assert!(missing_slot
            .ordered_slots(SemanticOpId(7), CompilerResourceKind::ReducePartial, 0, 1)
            .is_err());

        let duplicate_slot = [
            resource(0, 7, CompilerResourceKind::ReducePartial, 0),
            resource(1, 7, CompilerResourceKind::ReducePartial, 0),
        ];
        assert!(ResourceIndex::new(&duplicate_slot).is_err());

        let resources = [
            resource(0, 7, CompilerResourceKind::ReducePartial, 1),
            resource(1, 7, CompilerResourceKind::ReducePartial, 0),
            resource(2, 8, CompilerResourceKind::ScanBlockSums, 0),
        ];
        let index = ResourceIndex::new(&resources).expect("dense test manifest");
        let ordered = index.owned(SemanticOpId(7), CompilerResourceKind::ReducePartial);
        assert_eq!(
            ordered.iter().map(|resource| resource.id.0).collect::<Vec<_>>(),
            [1, 0]
        );
        assert!(index.ordered_slots(SemanticOpId(7), CompilerResourceKind::ReducePartial, 0, 2).is_ok());
        assert!(index.exactly_one(SemanticOpId(7), CompilerResourceKind::ReducePartial).is_err());
        assert_eq!(
            index
                .exactly_one(SemanticOpId(8), CompilerResourceKind::ScanBlockSums)
                .expect("one scan-sum resource")
                .id,
            crate::ResourceId(2)
        );
        assert!(index.exactly_one_at(SemanticOpId(8), CompilerResourceKind::ScanBlockSums, 0).is_ok());
        assert!(index.exactly_one(SemanticOpId(9), CompilerResourceKind::ScanBlockSums).is_err());
        assert!(index.exactly_one(SemanticOpId(7), CompilerResourceKind::ScanBlockSums).is_err());
        assert!(index
            .optional(SemanticOpId(9), CompilerResourceKind::ScanBlockSums)
            .expect("missing optional resource is valid")
            .is_none());
        assert!(index.optional(SemanticOpId(7), CompilerResourceKind::ReducePartial).is_err());
    }
}
