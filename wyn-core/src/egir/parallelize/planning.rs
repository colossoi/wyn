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
use crate::egir::types::{SegExtent, SideEffectSite};

/// The target recipe selected for one projected physical kernel. Parallel
/// variants own the exact graph-local analysis payload consumed by emission;
/// lowering never re-runs candidate recognition.
enum AnalyzedRecipe {
    Filter(super::filter::FilterCandidate),
    Reduce(super::reduce::ReduceCandidate),
    Scan(super::scan::ScanCandidate),
    Map,
    Serial(SideEffectSite),
    Unchanged,
}

pub(super) enum PlannedRecipe {
    Filter(super::filter::BoundFilter),
    Reduce(super::reduce::BoundReduce),
    Scan(super::scan::BoundScan),
    Map,
    Serial(SideEffectSite),
    Unchanged,
}

/// One projected physical kernel and its ownership of semantic output slots.
pub(super) struct PlannedKernel<R = PlannedRecipe> {
    pub body: crate::egir::program::PlannedEntry,
    pub semantic_slots: Vec<usize>,
    pub recipe: R,
}

/// A non-empty endpoint plan. `primary` always reuses the seeded kernel;
/// siblings are installed beside it only when output-domain projection split
/// the source entry.
pub(super) struct EndpointPlan<R = PlannedRecipe> {
    pub split_outputs: bool,
    pub primary: PlannedKernel<R>,
    pub siblings: Vec<PlannedKernel<R>>,
}

/// Authoritative target recipes indexed by the endpoint that owns the seeded
/// kernel. Sequential policy deliberately carries no parallel recipe state.
pub(super) struct CandidateIndex<R = PlannedRecipe> {
    endpoints: HashMap<CompilerFlowEndpoint, EndpointPlan<R>>,
    fallbacks: Vec<FallbackReason>,
    sequential: bool,
}

impl CandidateIndex<AnalyzedRecipe> {
    fn parallel() -> Self {
        Self {
            endpoints: HashMap::new(),
            fallbacks: Vec::new(),
            sequential: false,
        }
    }

    fn insert(&mut self, endpoint: CompilerFlowEndpoint, plan: EndpointPlan<AnalyzedRecipe>) -> Result<()> {
        if self.endpoints.insert(endpoint, plan).is_some() {
            return Err(ParallelizeError::Invalid(format!(
                "flow endpoint {endpoint:?} has multiple target recipes"
            )));
        }
        Ok(())
    }

    fn bind(self, resources: &ResourceIndex<'_>) -> Result<CandidateIndex> {
        let endpoints = self
            .endpoints
            .into_iter()
            .map(|(endpoint, plan)| Ok((endpoint, bind_endpoint(plan, resources)?)))
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(CandidateIndex {
            endpoints,
            fallbacks: self.fallbacks,
            sequential: false,
        })
    }
}

impl CandidateIndex {
    pub(super) fn sequential() -> Self {
        Self {
            endpoints: HashMap::new(),
            fallbacks: Vec::new(),
            sequential: true,
        }
    }

    pub(super) fn take_endpoint(&mut self, endpoint: CompilerFlowEndpoint) -> Result<Option<EndpointPlan>> {
        if self.sequential {
            return Ok(None);
        }
        self.endpoints.remove(&endpoint).map(Some).ok_or_else(|| {
            ParallelizeError::Invalid(format!("flow endpoint {endpoint:?} has no target recipe"))
        })
    }
}

struct ScratchRequest {
    endpoint: CompilerFlowEndpoint,
    compiler: CompilerResource,
    elem_ty: Type<TypeName>,
    size: LogicalSize,
}

pub(super) struct ParallelPlan {
    candidates: CandidateIndex<AnalyzedRecipe>,
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
        let resources = ResourceIndex::new(&inner.resources)?;
        self.candidates.bind(&resources)
    }
}

fn bind_endpoint(
    plan: EndpointPlan<AnalyzedRecipe>,
    resources: &ResourceIndex<'_>,
) -> Result<EndpointPlan> {
    Ok(EndpointPlan {
        split_outputs: plan.split_outputs,
        primary: bind_kernel(plan.primary, resources)?,
        siblings: plan
            .siblings
            .into_iter()
            .map(|kernel| bind_kernel(kernel, resources))
            .collect::<Result<Vec<_>>>()?,
    })
}

fn bind_kernel(
    kernel: PlannedKernel<AnalyzedRecipe>,
    resources: &ResourceIndex<'_>,
) -> Result<PlannedKernel> {
    let recipe = match kernel.recipe {
        AnalyzedRecipe::Filter(candidate) => {
            PlannedRecipe::Filter(super::filter::BoundFilter::bind(candidate, resources)?)
        }
        AnalyzedRecipe::Reduce(candidate) => {
            PlannedRecipe::Reduce(super::reduce::BoundReduce::bind(candidate, resources)?)
        }
        AnalyzedRecipe::Scan(candidate) => {
            PlannedRecipe::Scan(super::scan::BoundScan::bind(candidate, resources)?)
        }
        AnalyzedRecipe::Map => PlannedRecipe::Map,
        AnalyzedRecipe::Serial(site) => PlannedRecipe::Serial(site),
        AnalyzedRecipe::Unchanged => PlannedRecipe::Unchanged,
    };
    Ok(PlannedKernel {
        body: kernel.body,
        semantic_slots: kernel.semantic_slots,
        recipe,
    })
}

/// Analyze every projected endpoint once. Recipes retain their projected body
/// and graph-local handles until emission consumes the endpoint plan.
pub(super) fn analyze(inner: &AllocatedProgram, policy: ParallelPolicy) -> Result<ParallelPlan> {
    let resources = ResourceIndex::new(&inner.resources)?;
    let mut candidates = CandidateIndex::parallel();
    let mut requests = Vec::new();
    for (endpoint, entry) in inner.entries_with_endpoints() {
        let plan = analyze_endpoint(
            entry,
            endpoint,
            policy,
            &resources,
            &mut requests,
            &mut candidates.fallbacks,
        )?;
        candidates.insert(endpoint, plan)?;
    }
    Ok(ParallelPlan { candidates, requests })
}

#[cfg(test)]
pub(super) fn preflight_fallback_reasons(inner: &AllocatedProgram) -> Result<Vec<FallbackReason>> {
    let plan = analyze(inner, ParallelPolicy::default())?;
    let mut reasons = plan.candidates.fallbacks;
    reasons.sort_by_key(|reason| *reason as u8);
    Ok(reasons)
}

fn analyze_endpoint(
    entry: &crate::egir::program::SemanticEntry,
    endpoint: CompilerFlowEndpoint,
    policy: ParallelPolicy,
    resources: &ResourceIndex<'_>,
    requests: &mut Vec<ScratchRequest>,
    fallbacks: &mut Vec<FallbackReason>,
) -> Result<EndpointPlan<AnalyzedRecipe>> {
    let projected = crate::egir::program::PlannedEntry::project(entry)?;
    if matches!(endpoint, CompilerFlowEndpoint::Entry(_)) {
        if let Some(split) = super::split_multidomain_seg_maps(&projected)? {
            let primary = analyze_projected_kernel(
                split.primary.entry,
                split.primary.semantic_slots,
                endpoint,
                policy,
                resources,
                requests,
                fallbacks,
            )?;
            let siblings = split
                .siblings
                .into_iter()
                .map(|split| {
                    analyze_projected_kernel(
                        split.entry,
                        split.semantic_slots,
                        endpoint,
                        policy,
                        resources,
                        requests,
                        fallbacks,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(EndpointPlan {
                split_outputs: true,
                primary,
                siblings,
            });
        }
    }

    let slots = (0..projected.outputs.len()).collect();
    let primary =
        analyze_projected_kernel(projected, slots, endpoint, policy, resources, requests, fallbacks)?;
    Ok(EndpointPlan {
        split_outputs: false,
        primary,
        siblings: Vec::new(),
    })
}

#[allow(clippy::too_many_arguments)]
fn analyze_projected_kernel(
    body: crate::egir::program::PlannedEntry,
    semantic_slots: Vec<usize>,
    endpoint: CompilerFlowEndpoint,
    policy: ParallelPolicy,
    resources: &ResourceIndex<'_>,
    requests: &mut Vec<ScratchRequest>,
    fallbacks: &mut Vec<FallbackReason>,
) -> Result<PlannedKernel<AnalyzedRecipe>> {
    let mut filters = super::analyze_filter_candidates(&body)?;
    if filters.len() == 1 {
        let (_, selection) = filters.remove(0);
        match selection {
            RecipeSelection::Parallel(candidate) => {
                requests.extend(filter_scratch_requests(endpoint, policy, &candidate));
                return Ok(PlannedKernel {
                    body,
                    semantic_slots,
                    recipe: AnalyzedRecipe::Filter(candidate),
                });
            }
            RecipeSelection::Serial(reason) => fallbacks.push(reason),
        }
    } else {
        fallbacks.extend(filters.into_iter().filter_map(|(_, selection)| match selection {
            RecipeSelection::Parallel(_) => None,
            RecipeSelection::Serial(reason) => Some(reason),
        }));
    }

    let recipe = match super::segmented_recipe(&body) {
        Some(super::SegmentedRecipe::Reduce(located)) => {
            let site = located.site;
            match super::analyze_reduce_candidate(&body, located, resources)? {
                RecipeSelection::Parallel(candidate) => {
                    for (slot, elem_ty) in candidate.scratch_types().cloned().enumerate() {
                        requests.push(scratch_request(
                            endpoint,
                            candidate.owner,
                            slot,
                            CompilerResourceKind::ReducePartial,
                            elem_ty,
                        )?);
                    }
                    AnalyzedRecipe::Reduce(candidate)
                }
                RecipeSelection::Serial(reason) => {
                    fallbacks.push(reason);
                    AnalyzedRecipe::Serial(site)
                }
            }
        }
        Some(super::SegmentedRecipe::Scan(located)) => {
            let site = located.site;
            match super::analyze_scan_candidate(&body, located)? {
                RecipeSelection::Parallel(candidate) => {
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
                    AnalyzedRecipe::Scan(candidate)
                }
                RecipeSelection::Serial(reason) => {
                    fallbacks.push(reason);
                    AnalyzedRecipe::Serial(site)
                }
            }
        }
        Some(super::SegmentedRecipe::Map) => AnalyzedRecipe::Map,
        Some(super::SegmentedRecipe::Composite(site)) => AnalyzedRecipe::Serial(site),
        _ => AnalyzedRecipe::Unchanged,
    };
    Ok(PlannedKernel {
        body,
        semantic_slots,
        recipe,
    })
}

fn filter_scratch_requests(
    endpoint: CompilerFlowEndpoint,
    policy: ParallelPolicy,
    candidate: &super::filter::FilterCandidate,
) -> Vec<ScratchRequest> {
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
    [
        (CompilerResourceKind::FilterFlags, element_count_size.clone()),
        (CompilerResourceKind::FilterOffsets, element_count_size),
        (
            CompilerResourceKind::FilterScanBlockSums,
            worker_count_size.clone(),
        ),
        (CompilerResourceKind::FilterScanBlockOffsets, worker_count_size),
    ]
    .into_iter()
    .enumerate()
    .map(|(slot, (kind, size))| ScratchRequest {
        endpoint,
        compiler: CompilerResource::new(kind, Some(candidate.semantic_id), slot),
        elem_ty: Type::Constructed(TypeName::UInt(32), vec![]),
        size,
    })
    .collect()
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
