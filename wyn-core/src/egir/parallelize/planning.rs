//! Target policy, checked planning failures, session indexes, and deterministic
//! recipe-owned scratch allocation.

use std::collections::{BTreeMap, HashMap};

use polytype::Type;
use thiserror::Error;

use crate::ast::TypeName;

use super::schedule::KernelMutationError;
use crate::egir::program::{
    AllocatedProgram, CompilerFlowEndpoint, CompilerResource, CompilerResourceFlow, CompilerResourceKind,
    LogicalResource, LogicalSize, ResourceOrigin, SemanticOpId,
};
use crate::egir::types::SegExtent;

/// Scheduling choices shared by candidate analysis, scratch sizing, and
/// physical kernel construction. Selected recipes retain any policy values
/// needed by later lowering stages instead of re-deriving them.
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
            reduce_phase2_width: PHASE2_WIDTH,
            filter_scan_groups: FILTER_SCAN_GROUPS,
        }
    }
}

/// Per-workgroup width of a synthesized phase-2 tree reduce.
const PHASE2_WIDTH: u32 = 256;
/// Per-workgroup width used to chunk a phase-1 partial reduce or scan.
pub(super) const REDUCE_PHASE1_WIDTH: u32 = 64;
/// Workgroup count for the runtime-filter chunk scan.
pub(super) const FILTER_SCAN_GROUPS: u32 = 4;

/// Checked failures produced while selecting and constructing kernel recipes.
#[derive(Debug, Error)]
pub(in crate::egir) enum ParallelizeError {
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

pub(in crate::egir) type Result<T> = std::result::Result<T, ParallelizeError>;

/// Immutable lookup view over the final logical resource manifest used by one
/// planning session. Owner/kind buckets are ordered by compiler slot.
pub(super) struct ResourceIndex<'a> {
    resources: &'a [LogicalResource],
    owned: HashMap<(SemanticOpId, CompilerResourceKind), Vec<&'a LogicalResource>>,
}

/// Canonical indexed view of compiler producer/consumer edges for one planning
/// session. Both materialization attachment and phase dependency coalescing
/// consume this same data.
pub(super) struct ResourceFlowIndex {
    flows: Vec<(crate::ResourceId, CompilerResourceFlow)>,
    incoming: BTreeMap<CompilerFlowEndpoint, Vec<CompilerFlowEndpoint>>,
}

/// Why a valid semantic operation cannot use a target-parallel recipe.
///
/// These reasons describe supported serial fallbacks. Missing graph facts,
/// resources, or routes remain checked internal errors instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FallbackReason {
    SequentialPolicy,
    UnsupportedPlacement,
    UnsupportedCaptures,
    UnsupportedViewShape,
    UnsupportedDestination,
    UnsupportedScratchLayout,
    UnsupportedOperationShape,
}

/// Result of immutable recipe analysis. A parallel payload is complete for
/// its immediate consumer; serial selection carries the reason no scratch or
/// graph mutation may occur.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum RecipeSelection<T> {
    Parallel(T),
    Serial(FallbackReason),
}

impl<T> RecipeSelection<T> {
    fn without_payload(&self) -> RecipeSelection<()> {
        match self {
            Self::Parallel(_) => RecipeSelection::Parallel(()),
            Self::Serial(reason) => RecipeSelection::Serial(*reason),
        }
    }
}

/// A graph-local segmented recipe paired with the projected body from which
/// its node handles were derived. Emission consumes the pair before mutating
/// the body, so those handles never cross a graph projection boundary.
pub(super) enum PreparedSegmented {
    Reduce {
        body: crate::egir::program::PlannedEntry,
        candidate: super::reduce::ReduceCandidate,
    },
    Scan {
        body: crate::egir::program::PlannedEntry,
        candidate: super::scan::ScanCandidate,
    },
}

/// Authoritative record of the recipes selected during immutable preflight.
/// Parallel reduce and scan recipes retain their owning projected bodies;
/// emission consumes those prepared units rather than repeating analysis.
pub(super) struct CandidateIndex {
    filters: HashMap<SemanticOpId, RecipeSelection<()>>,
    reduces: HashMap<SemanticOpId, RecipeSelection<()>>,
    scans: HashMap<SemanticOpId, RecipeSelection<()>>,
    prepared_segmented: HashMap<CompilerFlowEndpoint, PreparedSegmented>,
    default_fallback: Option<FallbackReason>,
}

impl CandidateIndex {
    fn preflight() -> Self {
        Self {
            filters: HashMap::new(),
            reduces: HashMap::new(),
            scans: HashMap::new(),
            prepared_segmented: HashMap::new(),
            default_fallback: None,
        }
    }

    pub(super) fn sequential() -> Self {
        Self {
            default_fallback: Some(FallbackReason::SequentialPolicy),
            ..Self::preflight()
        }
    }

    pub(super) fn filter(&self, owner: SemanticOpId) -> Result<RecipeSelection<()>> {
        self.selection(&self.filters, owner, "filter")
    }

    pub(super) fn reduce(&self, owner: SemanticOpId) -> Result<RecipeSelection<()>> {
        self.selection(&self.reduces, owner, "reduce")
    }

    pub(super) fn scan(&self, owner: SemanticOpId) -> Result<RecipeSelection<()>> {
        self.selection(&self.scans, owner, "scan")
    }

    pub(super) fn take_prepared(&mut self, endpoint: CompilerFlowEndpoint) -> Option<PreparedSegmented> {
        self.prepared_segmented.remove(&endpoint)
    }

    fn selection(
        &self,
        selections: &HashMap<SemanticOpId, RecipeSelection<()>>,
        owner: SemanticOpId,
        family: &str,
    ) -> Result<RecipeSelection<()>> {
        if let Some(selection) = selections.get(&owner) {
            return Ok(selection.clone());
        }
        if let Some(reason) = self.default_fallback {
            return Ok(RecipeSelection::Serial(reason));
        }
        Err(ParallelizeError::Invalid(format!(
            "semantic operation {owner:?} has no preflight {family} selection"
        )))
    }

    fn record_filter<T>(&mut self, owner: SemanticOpId, selection: &RecipeSelection<T>) -> Result<()> {
        Self::record(&mut self.filters, owner, selection, "filter")
    }

    fn record_reduce<T>(&mut self, owner: SemanticOpId, selection: &RecipeSelection<T>) -> Result<()> {
        Self::record(&mut self.reduces, owner, selection, "reduce")
    }

    fn record_scan<T>(&mut self, owner: SemanticOpId, selection: &RecipeSelection<T>) -> Result<()> {
        Self::record(&mut self.scans, owner, selection, "scan")
    }

    fn prepare_segmented(
        &mut self,
        endpoint: CompilerFlowEndpoint,
        prepared: PreparedSegmented,
    ) -> Result<()> {
        if self.prepared_segmented.insert(endpoint, prepared).is_some() {
            return Err(ParallelizeError::Invalid(format!(
                "flow endpoint {endpoint:?} has multiple prepared segmented recipes"
            )));
        }
        Ok(())
    }

    fn record<T>(
        selections: &mut HashMap<SemanticOpId, RecipeSelection<()>>,
        owner: SemanticOpId,
        selection: &RecipeSelection<T>,
        family: &str,
    ) -> Result<()> {
        if selections.insert(owner, selection.without_payload()).is_some() {
            return Err(ParallelizeError::Invalid(format!(
                "semantic operation {owner:?} has duplicate {family} preflight selections"
            )));
        }
        Ok(())
    }
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

struct ScratchRequest {
    endpoint: CompilerFlowEndpoint,
    compiler: CompilerResource,
    elem_ty: Type<TypeName>,
    size: LogicalSize,
}

struct ScratchPlan {
    candidates: CandidateIndex,
    requests: Vec<ScratchRequest>,
}

/// Append target-recipe work buffers immediately before kernel planning.
/// Semantic residency resources have already been established; these buffers
/// exist only when the parallel policy is selected.
pub(super) fn allocate_parallel_scratch(
    inner: &mut AllocatedProgram,
    policy: ParallelPolicy,
) -> Result<CandidateIndex> {
    let ScratchPlan {
        candidates,
        mut requests,
    } = preflight_parallel_recipes(inner, policy)?;
    requests.sort_by_key(|request| {
        (
            request.endpoint,
            request.compiler.owner,
            request.compiler.kind,
            request.compiler.slot,
        )
    });

    for request in requests {
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

/// Analyze all candidate operations and produce owned scratch requests without
/// mutating the program. Graph-local reduce and scan handles stay paired with
/// the projected body that owns them until emission consumes the pair.
fn preflight_parallel_recipes(inner: &AllocatedProgram, policy: ParallelPolicy) -> Result<ScratchPlan> {
    let resource_index = ResourceIndex::new(&inner.resources)?;
    let mut candidates = CandidateIndex::preflight();
    let mut requests = filter_requests(inner, policy, &mut candidates)?;
    requests.extend(segmented_requests(inner, &resource_index, &mut candidates)?);
    Ok(ScratchPlan { candidates, requests })
}

#[cfg(test)]
pub(super) fn preflight_fallback_reasons(inner: &AllocatedProgram) -> Result<Vec<FallbackReason>> {
    let plan = preflight_parallel_recipes(inner, ParallelPolicy::default())?;
    let mut reasons = plan
        .candidates
        .filters
        .values()
        .chain(plan.candidates.reduces.values())
        .chain(plan.candidates.scans.values())
        .filter_map(|selection| match selection {
            RecipeSelection::Parallel(()) => None,
            RecipeSelection::Serial(reason) => Some(*reason),
        })
        .collect::<Vec<_>>();
    reasons.sort_by_key(|reason| *reason as u8);
    Ok(reasons)
}

fn filter_requests(
    inner: &AllocatedProgram,
    policy: ParallelPolicy,
    candidates: &mut CandidateIndex,
) -> Result<Vec<ScratchRequest>> {
    let mut requests = Vec::new();
    for (endpoint, entry) in inner.entries_with_endpoints() {
        for (owner, selection) in super::analyze_filter_candidates(entry)? {
            candidates.record_filter(owner, &selection)?;
            let RecipeSelection::Parallel(candidate) = selection else {
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
            let worker_count_size = LogicalSize::FixedBytes(
                (policy.filter_scan_groups * policy.reduce_phase1_width) as u64 * 4,
            );
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
    }
    Ok(requests)
}

fn segmented_requests(
    inner: &AllocatedProgram,
    resources: &ResourceIndex<'_>,
    candidates: &mut CandidateIndex,
) -> Result<Vec<ScratchRequest>> {
    let mut requests = Vec::new();
    for (endpoint, entry) in inner.entries_with_endpoints() {
        let projected = crate::egir::program::PlannedEntry::project(entry)?;
        let Some(located) = super::segmented_recipe_effect(&projected) else {
            continue;
        };
        let Some(owner) = located.effect.kind.soac_id().copied() else {
            return Err(ParallelizeError::Invalid(
                "segmented recipe effect has no semantic operation id".into(),
            ));
        };
        let Ok(family) = super::seg_recipe_family(located.effect) else {
            continue;
        };
        match family {
            super::SegScratchFamily::Reduce => {
                let selection = super::analyze_reduce_candidate(&projected, resources)?;
                candidates.record_reduce(owner, &selection)?;
                if let RecipeSelection::Parallel(candidate) = selection {
                    for (slot, elem_ty) in candidate.scratch_types.iter().cloned().enumerate() {
                        requests.push(scratch_request(
                            endpoint,
                            candidate.owner,
                            slot,
                            CompilerResourceKind::ReducePartial,
                            elem_ty,
                        )?);
                    }
                    candidates.prepare_segmented(
                        endpoint,
                        PreparedSegmented::Reduce {
                            body: projected,
                            candidate,
                        },
                    )?;
                }
            }
            super::SegScratchFamily::Scan => {
                let selection = super::analyze_scan_candidate(&projected)?;
                candidates.record_scan(owner, &selection)?;
                if let RecipeSelection::Parallel(candidate) = selection {
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
                    candidates.prepare_segmented(
                        endpoint,
                        PreparedSegmented::Scan {
                            body: projected,
                            candidate,
                        },
                    )?;
                }
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
    })?;
    Ok(ScratchRequest {
        endpoint,
        compiler: CompilerResource::new(kind, Some(owner), slot),
        elem_ty,
        size: LogicalSize::SameAsDispatch { elem_bytes },
    })
}

#[cfg(test)]
#[path = "planning_tests.rs"]
mod tests;
