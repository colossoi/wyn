//! Immutable recipe analysis and deterministic recipe-owned scratch allocation.

use std::collections::HashMap;

use polytype::Type;

use crate::ast::TypeName;
use crate::egir::soac::screma;
use crate::egir::types::{EGraph, SideEffect, SideEffectKind, SideEffectSite, Soac, SoacEffect};
use crate::flow::ExecutionModel;

use super::model::{FallbackReason, ParallelizeError, RecipeSelection, Result};
use crate::egir::program::{
    AllocatedProgram, CompilerFlowEndpoint, CompilerResource, CompilerResourceKey, CompilerResourceKind,
    LogicalResourceArena, LogicalSize, SemanticOpId,
};
use crate::egir::types::SegExtent;

#[derive(Clone, Copy)]
pub(super) struct LocatedScrema<'a> {
    pub site: SideEffectSite,
    pub effect: &'a SideEffect,
    pub owner: SemanticOpId,
    pub op: &'a screma::Op<crate::egir::types::Semantic>,
}

pub(super) struct ParallelReduce<'a> {
    located: LocatedScrema<'a>,
    lanes: &'a screma::Lanes,
    operators: &'a screma::NonEmpty<screma::Operator>,
}

pub(super) struct ParallelScan<'a> {
    located: LocatedScrema<'a>,
    lanes: &'a screma::Lanes,
    operators: &'a screma::NonEmpty<screma::Operator>,
}

#[derive(Clone)]
pub(super) struct SerialScremaRecipe {
    site: SideEffectSite,
    owner: SemanticOpId,
    op: screma::Op<crate::egir::types::Semantic>,
}

impl LocatedScrema<'_> {
    pub(super) fn serial_recipe(&self) -> SerialScremaRecipe {
        SerialScremaRecipe {
            site: self.site,
            owner: self.owner,
            op: self.op.clone(),
        }
    }
}

impl<'a> ParallelReduce<'a> {
    fn recognize(
        located: LocatedScrema<'a>,
        lanes: &'a screma::Lanes,
        operators: &'a screma::NonEmpty<screma::Operator>,
    ) -> RecipeSelection<Self> {
        if operators.iter().any(|operator| !operator.combine.captures.is_empty()) {
            return RecipeSelection::Serial(FallbackReason::UnsupportedCaptures);
        }
        if lanes.inputs.is_empty() {
            return RecipeSelection::Serial(FallbackReason::UnsupportedOperationShape);
        }
        if !lanes.maps.iter().all(|map| map.destination.is_output_view())
            || !operators.iter().all(|operator| operator.destination.is_unplaced_fresh())
        {
            return RecipeSelection::Serial(FallbackReason::UnsupportedDestination);
        }
        RecipeSelection::Parallel(Self {
            located,
            lanes,
            operators,
        })
    }

    pub(super) fn serial_recipe(&self) -> SerialScremaRecipe {
        self.located.serial_recipe()
    }

    pub(super) fn into_parts(
        self,
    ) -> (
        LocatedScrema<'a>,
        &'a screma::Lanes,
        &'a screma::NonEmpty<screma::Operator>,
    ) {
        (self.located, self.lanes, self.operators)
    }
}

impl<'a> ParallelScan<'a> {
    fn recognize(
        located: LocatedScrema<'a>,
        lanes: &'a screma::Lanes,
        operators: &'a screma::NonEmpty<screma::Operator>,
    ) -> RecipeSelection<Self> {
        if !operators.rest.is_empty() || lanes.inputs.len() != 1 {
            return RecipeSelection::Serial(FallbackReason::UnsupportedOperationShape);
        }
        if !operators.first.combine.captures.is_empty() {
            return RecipeSelection::Serial(FallbackReason::UnsupportedCaptures);
        }
        if !lanes.maps.iter().all(|map| map.destination.is_output_view())
            || !operators.iter().all(|operator| operator.destination.is_output_view())
        {
            return RecipeSelection::Serial(FallbackReason::UnsupportedDestination);
        }
        RecipeSelection::Parallel(Self {
            located,
            lanes,
            operators,
        })
    }

    pub(super) fn serial_recipe(&self) -> SerialScremaRecipe {
        self.located.serial_recipe()
    }

    pub(super) fn into_parts(
        self,
    ) -> (
        LocatedScrema<'a>,
        &'a screma::Lanes,
        &'a screma::NonEmpty<screma::Operator>,
    ) {
        (self.located, self.lanes, self.operators)
    }
}

enum SegmentedRecipe<'a> {
    Reduce(ParallelReduce<'a>),
    Scan(ParallelScan<'a>),
    Serial(SerialScremaRecipe),
    Map,
    Composite(LocatedScrema<'a>),
}

fn segmented_screma_effect(graph: &EGraph) -> Option<LocatedScrema<'_>> {
    graph.skeleton.blocks.iter().find_map(|(block, contents)| {
        contents.side_effects.iter().enumerate().find_map(|(index, effect)| {
            let SideEffectKind::Soac(SoacEffect(owner, Soac::Screma(op))) = &effect.kind else {
                return None;
            };
            matches!(
                op.semantic_state(),
                screma::SemanticState::Segmented {
                    placement: screma::Placement::Kernel,
                    ..
                }
            )
            .then_some(LocatedScrema {
                site: SideEffectSite { block, index },
                effect,
                owner: *owner,
                op,
            })
        })
    })
}

fn segmented_recipe_effect(entry: &crate::egir::program::PlannedEntry) -> Option<LocatedScrema<'_>> {
    if let Some(effect) = segmented_screma_effect(&entry.graph) {
        return Some(effect);
    }
    if !matches!(entry.execution_model, ExecutionModel::Compute { .. }) {
        return None;
    }
    let mut promoted = None;
    for (block, contents) in &entry.graph.skeleton.blocks {
        for (index, effect) in contents.side_effects.iter().enumerate() {
            let SideEffectKind::Soac(SoacEffect(owner, Soac::Screma(op))) = &effect.kind else {
                continue;
            };
            if matches!(
                op.semantic_state(),
                screma::SemanticState::Segmented {
                    placement: screma::Placement::LaneLocal,
                    output_slots,
                    ..
                } if !output_slots.is_empty()
            ) && matches!(op, screma::Op::Reduce { .. } | screma::Op::Scan { .. })
            {
                if promoted.is_some() {
                    return None;
                }
                promoted = Some(LocatedScrema {
                    site: SideEffectSite { block, index },
                    effect,
                    owner: *owner,
                    op,
                });
            }
        }
    }
    promoted
}

fn segmented_recipe(entry: &crate::egir::program::PlannedEntry) -> Option<SegmentedRecipe<'_>> {
    let located = segmented_recipe_effect(entry)?;
    Some(match located.op {
        screma::Op::Reduce { lanes, operators, .. } => {
            match ParallelReduce::recognize(located, lanes, operators) {
                RecipeSelection::Parallel(recipe) => SegmentedRecipe::Reduce(recipe),
                RecipeSelection::Serial(_) => SegmentedRecipe::Serial(located.serial_recipe()),
            }
        }
        screma::Op::Scan { lanes, operators, .. } => {
            match ParallelScan::recognize(located, lanes, operators) {
                RecipeSelection::Parallel(recipe) => SegmentedRecipe::Scan(recipe),
                RecipeSelection::Serial(_) => SegmentedRecipe::Serial(located.serial_recipe()),
            }
        }
        screma::Op::Map { .. } => SegmentedRecipe::Map,
        screma::Op::Composite { .. } => SegmentedRecipe::Composite(located),
    })
}

pub(super) fn parallel_recipe_effect(entry: &crate::egir::program::PlannedEntry) -> Option<&SideEffect> {
    segmented_recipe_effect(entry).map(|located| located.effect).or_else(|| {
        entry.graph.skeleton.blocks.values().find_map(|contents| {
            contents
                .side_effects
                .iter()
                .find(|effect| matches!(&effect.kind, SideEffectKind::Soac(SoacEffect(_, Soac::Filter(_)))))
        })
    })
}

pub(super) fn make_screma_serial(graph: &mut EGraph, recipe: SerialScremaRecipe) {
    let mut op = recipe.op;
    *op.semantic_state_mut() = screma::SemanticState::Serial;
    graph.skeleton.effect_mut(recipe.site).kind =
        SideEffectKind::Soac(SoacEffect(recipe.owner, Soac::Screma(op)));
}

/// The target recipe selected for one projected physical kernel. Parallel
/// variants own the exact graph-local analysis payload consumed by emission;
/// lowering never re-runs candidate recognition.
enum AnalyzedRecipe {
    Filter(super::filter::FilterCandidate),
    Reduce(super::reduce::ReduceCandidate),
    Scan(super::scan::ScanCandidate),
    Map,
    Serial(SerialScremaRecipe),
    Unchanged,
}

pub(super) enum PlannedRecipe {
    Filter(super::filter::BoundFilter),
    Reduce(super::reduce::BoundReduce),
    Scan(super::scan::BoundScan),
    Map,
    Serial(SerialScremaRecipe),
    Unchanged,
}

/// One projected physical kernel and its ownership of semantic output slots.
pub(super) struct PlannedKernel<R = PlannedRecipe> {
    body: crate::egir::program::PlannedEntry,
    semantic_slots: Vec<usize>,
    recipe: R,
}

impl<R> PlannedKernel<R> {
    fn new(body: crate::egir::program::PlannedEntry, semantic_slots: Vec<usize>, recipe: R) -> Self {
        Self {
            body,
            semantic_slots,
            recipe,
        }
    }

    /// The selected recipe and every graph-local handle it contains stay
    /// coupled to this body until lowering consumes the pair.
    pub(super) fn into_parts(self) -> (crate::egir::program::PlannedEntry, Vec<usize>, R) {
        (self.body, self.semantic_slots, self.recipe)
    }

    pub(super) fn seed_body(&self) -> crate::egir::program::PlannedEntry {
        self.body.clone()
    }
}

/// A non-empty endpoint plan. `primary` always reuses the seeded kernel;
/// siblings are installed beside it only when output-domain projection split
/// the source entry.
pub(super) struct EndpointPlan<R = PlannedRecipe> {
    split_outputs: bool,
    primary: PlannedKernel<R>,
    siblings: Vec<PlannedKernel<R>>,
}

impl<R> EndpointPlan<R> {
    fn new(split_outputs: bool, primary: PlannedKernel<R>, siblings: Vec<PlannedKernel<R>>) -> Self {
        Self {
            split_outputs,
            primary,
            siblings,
        }
    }

    pub(super) fn into_parts(self) -> (bool, PlannedKernel<R>, Vec<PlannedKernel<R>>) {
        (self.split_outputs, self.primary, self.siblings)
    }
}

/// Authoritative target recipes indexed by the endpoint that owns the seeded
/// kernel. Sequential policy deliberately carries no parallel recipe state.
pub(super) enum RecipeIndex<R = PlannedRecipe> {
    Sequential,
    Parallel(HashMap<CompilerFlowEndpoint, EndpointPlan<R>>),
}

impl RecipeIndex<AnalyzedRecipe> {
    fn parallel() -> Self {
        Self::Parallel(HashMap::new())
    }

    fn insert(&mut self, endpoint: CompilerFlowEndpoint, plan: EndpointPlan<AnalyzedRecipe>) -> Result<()> {
        let Self::Parallel(endpoints) = self else {
            return Err(ParallelizeError::Invalid(
                "cannot add a parallel recipe to a sequential recipe index".into(),
            ));
        };
        if endpoints.insert(endpoint, plan).is_some() {
            return Err(ParallelizeError::Invalid(format!(
                "flow endpoint {endpoint:?} has multiple target recipes"
            )));
        }
        Ok(())
    }

    fn bind_scratch(self, resources: &ScratchBindings) -> RecipeIndex {
        let Self::Parallel(endpoints) = self else {
            return RecipeIndex::sequential();
        };
        RecipeIndex::Parallel(
            endpoints
                .into_iter()
                .map(|(endpoint, plan)| {
                    let (split_outputs, primary, siblings) = plan.into_parts();
                    (
                        endpoint,
                        EndpointPlan::new(
                            split_outputs,
                            bind_kernel(primary, resources),
                            siblings.into_iter().map(|kernel| bind_kernel(kernel, resources)).collect(),
                        ),
                    )
                })
                .collect(),
        )
    }
}

impl RecipeIndex {
    pub(super) fn sequential() -> Self {
        Self::Sequential
    }

    pub(super) fn take_endpoint(&mut self, endpoint: CompilerFlowEndpoint) -> Result<Option<EndpointPlan>> {
        match self {
            Self::Sequential => Ok(None),
            Self::Parallel(endpoints) => endpoints.remove(&endpoint).map(Some).ok_or_else(|| {
                ParallelizeError::Invalid(format!("flow endpoint {endpoint:?} has no target recipe"))
            }),
        }
    }
}

struct ScratchRequest {
    endpoint: CompilerFlowEndpoint,
    key: CompilerResourceKey,
    elem_ty: Type<TypeName>,
    size: LogicalSize,
}

pub(super) struct ScratchBindings {
    ids: HashMap<CompilerResourceKey, crate::ResourceId>,
}

impl ScratchBindings {
    pub(super) fn id(
        &self,
        owner: SemanticOpId,
        kind: CompilerResourceKind,
        slot: usize,
    ) -> crate::ResourceId {
        self.ids[&CompilerResourceKey { owner, kind, slot }]
    }
}

pub(super) struct AnalyzedPlan {
    recipes: RecipeIndex<AnalyzedRecipe>,
    requests: Vec<ScratchRequest>,
}

impl AnalyzedPlan {
    /// Allocate exactly the scratch requested by successfully selected recipes.
    /// This is the first mutation after immutable analysis has completed.
    pub(super) fn allocate_scratch(mut self, inner: &mut AllocatedProgram) -> Result<RecipeIndex> {
        self.requests.sort_by_key(|request| {
            (
                request.endpoint,
                request.key.owner,
                request.key.kind,
                request.key.slot,
            )
        });

        let mut bindings = ScratchBindings { ids: HashMap::new() };
        for request in self.requests {
            let id = inner.alloc_compiler_resource(
                CompilerResource::new(request.key.kind, Some(request.key.owner), request.key.slot),
                request.elem_ty,
                request.size,
            );
            bindings.ids.insert(request.key, id);
        }
        Ok(self.recipes.bind_scratch(&bindings))
    }
}

fn bind_kernel(kernel: PlannedKernel<AnalyzedRecipe>, resources: &ScratchBindings) -> PlannedKernel {
    let (body, semantic_slots, recipe) = kernel.into_parts();
    let recipe = match recipe {
        AnalyzedRecipe::Filter(candidate) => {
            PlannedRecipe::Filter(super::filter::BoundFilter::bind(candidate, resources))
        }
        AnalyzedRecipe::Reduce(candidate) => {
            PlannedRecipe::Reduce(super::reduce::BoundReduce::bind(candidate, resources))
        }
        AnalyzedRecipe::Scan(candidate) => {
            PlannedRecipe::Scan(super::scan::BoundScan::bind(candidate, resources))
        }
        AnalyzedRecipe::Map => PlannedRecipe::Map,
        AnalyzedRecipe::Serial(site) => PlannedRecipe::Serial(site),
        AnalyzedRecipe::Unchanged => PlannedRecipe::Unchanged,
    };
    PlannedKernel::new(body, semantic_slots, recipe)
}

/// Analyze every projected endpoint once. Recipes retain their projected body
/// and graph-local handles until emission consumes the endpoint plan.
pub(super) fn analyze(inner: &AllocatedProgram) -> Result<AnalyzedPlan> {
    let mut recipes = RecipeIndex::parallel();
    let mut requests = Vec::new();
    for (endpoint, entry) in inner.entries_with_endpoints() {
        let plan = analyze_endpoint(entry, endpoint, &inner.resources, &mut requests)?;
        recipes.insert(endpoint, plan)?;
    }
    Ok(AnalyzedPlan { recipes, requests })
}

fn analyze_endpoint(
    entry: &crate::egir::program::SemanticEntry,
    endpoint: CompilerFlowEndpoint,
    resources: &LogicalResourceArena,
    requests: &mut Vec<ScratchRequest>,
) -> Result<EndpointPlan<AnalyzedRecipe>> {
    let projected = crate::egir::program::PlannedEntry::project(entry)?;
    if matches!(endpoint, CompilerFlowEndpoint::Entry(_)) {
        if let Some(split) = super::split_multidomain_seg_maps(&projected)? {
            let primary = analyze_projected_kernel(
                split.primary.entry,
                split.primary.semantic_slots,
                endpoint,
                resources,
                requests,
            )?;
            let siblings = split
                .siblings
                .into_iter()
                .map(|split| {
                    analyze_projected_kernel(
                        split.entry,
                        split.semantic_slots,
                        endpoint,
                        resources,
                        requests,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(EndpointPlan::new(true, primary, siblings));
        }
    }

    let slots = (0..projected.outputs.len()).collect();
    let primary = analyze_projected_kernel(projected, slots, endpoint, resources, requests)?;
    Ok(EndpointPlan::new(false, primary, Vec::new()))
}

#[allow(clippy::too_many_arguments)]
fn analyze_projected_kernel(
    body: crate::egir::program::PlannedEntry,
    semantic_slots: Vec<usize>,
    endpoint: CompilerFlowEndpoint,
    resources: &LogicalResourceArena,
    requests: &mut Vec<ScratchRequest>,
) -> Result<PlannedKernel<AnalyzedRecipe>> {
    let mut filters = super::analyze_filter_candidates(&body)?;
    if filters.len() == 1 {
        let (_, selection) = filters.remove(0);
        match selection {
            RecipeSelection::Parallel(candidate) => {
                requests.extend(filter_scratch_requests(endpoint, &candidate));
                return Ok(PlannedKernel::new(
                    body,
                    semantic_slots,
                    AnalyzedRecipe::Filter(candidate),
                ));
            }
            RecipeSelection::Serial(_) => {}
        }
    }

    let recipe = match segmented_recipe(&body) {
        Some(SegmentedRecipe::Reduce(located)) => {
            let serial = located.serial_recipe();
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
                RecipeSelection::Serial(_) => AnalyzedRecipe::Serial(serial),
            }
        }
        Some(SegmentedRecipe::Scan(located)) => {
            let serial = located.serial_recipe();
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
                RecipeSelection::Serial(_) => AnalyzedRecipe::Serial(serial),
            }
        }
        Some(SegmentedRecipe::Serial(serial)) => AnalyzedRecipe::Serial(serial),
        Some(SegmentedRecipe::Map) => AnalyzedRecipe::Map,
        Some(SegmentedRecipe::Composite(located)) => AnalyzedRecipe::Serial(located.serial_recipe()),
        _ => AnalyzedRecipe::Unchanged,
    };
    Ok(PlannedKernel::new(body, semantic_slots, recipe))
}

fn filter_scratch_requests(
    endpoint: CompilerFlowEndpoint,
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
    let worker_count_size = LogicalSize::FixedBytes(candidate.scan_worker_count() as u64 * 4);
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
        key: CompilerResourceKey {
            owner: candidate.semantic_id,
            kind,
            slot,
        },
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
        key: CompilerResourceKey { owner, kind, slot },
        elem_ty,
        size: LogicalSize::SameAsDispatch { elem_bytes },
    })
}

#[cfg(test)]
#[path = "planning_tests.rs"]
mod tests;
