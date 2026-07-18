//! Immutable recipe analysis and deterministic recipe-owned scratch allocation.

use std::collections::HashMap;

use polytype::Type;

use crate::ast::TypeName;

use super::model::{
    FallbackReason, ParallelPolicy, ParallelizeError, RecipeSelection, ResourceIndex, Result,
};
use crate::egir::program::{
    AllocatedProgram, CompilerFlowEndpoint, CompilerResource, CompilerResourceKind, LogicalSize,
    ResourceOrigin, SemanticOpId,
};
use crate::egir::types::SegExtent;

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

struct ScratchRequest {
    endpoint: CompilerFlowEndpoint,
    compiler: CompilerResource,
    elem_ty: Type<TypeName>,
    size: LogicalSize,
}

pub(super) struct ParallelPlan {
    candidates: CandidateIndex,
    requests: Vec<ScratchRequest>,
}

impl ParallelPlan {
    /// Allocate exactly the scratch requested by successfully selected recipes.
    /// This is the first mutation after immutable analysis has completed.
    pub(super) fn allocate(mut self, inner: &mut AllocatedProgram) -> Result<CandidateIndex> {
        self.requests.sort_by_key(|request| {
            (
                request.endpoint,
                request.compiler.owner,
                request.compiler.kind,
                request.compiler.slot,
            )
        });

        for request in self.requests {
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
        Ok(self.candidates)
    }
}

/// Analyze all candidate operations and produce owned scratch requests without
/// mutating the program. Graph-local reduce and scan handles stay paired with
/// the projected body that owns them until emission consumes the pair.
pub(super) fn analyze(inner: &AllocatedProgram, policy: ParallelPolicy) -> Result<ParallelPlan> {
    let resource_index = ResourceIndex::new(&inner.resources)?;
    let mut candidates = CandidateIndex::preflight();
    let mut requests = filter_requests(inner, policy, &mut candidates)?;
    requests.extend(segmented_requests(inner, &resource_index, &mut candidates)?);
    Ok(ParallelPlan { candidates, requests })
}

#[cfg(test)]
pub(super) fn preflight_fallback_reasons(inner: &AllocatedProgram) -> Result<Vec<FallbackReason>> {
    let plan = analyze(inner, ParallelPolicy::default())?;
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
