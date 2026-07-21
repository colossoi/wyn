//! Immutable recipe analysis and deterministic recipe-owned scratch allocation.

use std::collections::HashMap;

use polytype::Type;

use crate::ast::TypeName;
use crate::egir::soac::screma;
use crate::egir::types::{EGraph, SideEffect, SideEffectKind, SideEffectSite, Soac, SoacEffect};

use super::model::{CandidateSelection, ParallelizeError, Result};
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

    pub(super) fn segmented(&self) -> Result<screma::Segmented<crate::egir::program::SemanticResourceRef>> {
        let screma::SemanticState::Segmented {
            space,
            output_slots,
            resources,
            ..
        } = self.op.semantic_state()
        else {
            return Err(ParallelizeError::Invalid(
                "selected parallel Screma lost its segmented semantic facts".into(),
            ));
        };
        Ok(screma::Segmented {
            space: space.clone(),
            output_slots: output_slots.clone(),
            resources: resources.clone(),
        })
    }
}

#[derive(Default)]
struct RecipeTargets {
    filters: Vec<SideEffectSite>,
    kernel_scremas: Vec<SideEffectSite>,
    promoted_folds: Vec<SideEffectSite>,
}

impl RecipeTargets {
    /// Target-relevant sites are classified once on the endpoint projection.
    /// Later output projection only remaps these handles.
    fn collect(entry: &crate::egir::program::PlannedEntry) -> Self {
        let mut targets = Self::default();
        for (block, contents) in &entry.graph.skeleton.blocks {
            for (index, effect) in contents.side_effects.iter().enumerate() {
                let site = SideEffectSite { block, index };
                match &effect.kind {
                    SideEffectKind::Soac(SoacEffect(_, Soac::Filter(_))) => targets.filters.push(site),
                    SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => match op.semantic_state() {
                        screma::SemanticState::Segmented {
                            placement: screma::Placement::Kernel,
                            ..
                        } => targets.kernel_scremas.push(site),
                        screma::SemanticState::Segmented {
                            placement: screma::Placement::LaneLocal,
                            output_slots,
                            ..
                        } if !output_slots.is_empty()
                            && matches!(op, screma::Op::Reduce { .. } | screma::Op::Scan { .. })
                            && entry.execution_model.is_compute() =>
                        {
                            targets.promoted_folds.push(site);
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
        targets
    }

    fn remap(&self, sites: &crate::LookupMap<SideEffectSite, SideEffectSite>) -> Self {
        let remap =
            |source: &[SideEffectSite]| source.iter().filter_map(|site| sites.get(site).copied()).collect();
        Self {
            filters: remap(&self.filters),
            kernel_scremas: remap(&self.kernel_scremas),
            promoted_folds: remap(&self.promoted_folds),
        }
    }

    fn screma_site(&self) -> Option<SideEffectSite> {
        self.kernel_scremas.first().copied().or_else(|| match self.promoted_folds.as_slice() {
            [site] => Some(*site),
            _ => None,
        })
    }
}

fn located_screma(
    entry: &crate::egir::program::PlannedEntry,
    site: SideEffectSite,
) -> Result<LocatedScrema<'_>> {
    let effect = entry.graph.skeleton.effect(site);
    let SideEffectKind::Soac(SoacEffect(owner, Soac::Screma(op))) = &effect.kind else {
        return Err(ParallelizeError::Invalid(format!(
            "selected Screma site {site:?} no longer contains a Screma operation"
        )));
    };
    Ok(LocatedScrema {
        site,
        effect,
        owner: *owner,
        op,
    })
}

pub(super) fn make_screma_serial(graph: &mut EGraph, recipe: SerialScremaRecipe) {
    let mut op = recipe.op;
    *op.semantic_state_mut() = screma::SemanticState::Serial;
    graph.skeleton.effect_mut(recipe.site).kind =
        SideEffectKind::Soac(SoacEffect(recipe.owner, Soac::Screma(op)));
}

/// The target recipe selected for one projected physical kernel. Algorithm
/// payloads change type when scratch ids are bound; the recipe shape does not.
pub(super) enum Recipe<Filter, Reduce, Scan> {
    Filter(Filter),
    Reduce(Reduce),
    Scan(Scan),
    Map(screma::Segmented<crate::egir::program::SemanticResourceRef>),
    Serial(SerialScremaRecipe),
    Unchanged,
}

type AnalyzedRecipe =
    Recipe<super::filter::FilterCandidate, super::reduce::ReduceCandidate, super::scan::ScanCandidate>;

pub(super) type PlannedRecipe =
    Recipe<super::filter::BoundFilter, super::reduce::BoundReduce, super::scan::BoundScan>;

/// One projected physical kernel and its ownership of semantic output slots.
pub(super) struct PlannedKernel<R = PlannedRecipe> {
    body: crate::egir::program::PlannedEntry,
    /// Maps this kernel's projected output slots back to the source entry.
    /// Unsplit kernels retain the source interface and need no mapping.
    output_projection: Option<Vec<usize>>,
    recipe: R,
}

impl<R> PlannedKernel<R> {
    fn new(
        body: crate::egir::program::PlannedEntry,
        output_projection: Option<Vec<usize>>,
        recipe: R,
    ) -> Self {
        Self {
            body,
            output_projection,
            recipe,
        }
    }

    /// The selected recipe and every graph-local handle it contains stay
    /// coupled to this body until lowering consumes the pair.
    pub(super) fn into_parts(self) -> (crate::egir::program::PlannedEntry, Option<Vec<usize>>, R) {
        (self.body, self.output_projection, self.recipe)
    }

    pub(super) fn seed_body(&self) -> crate::egir::program::PlannedEntry {
        self.body.clone()
    }
}

/// A non-empty endpoint plan. `primary` always reuses the seeded kernel;
/// siblings are installed beside it only when output-domain projection split
/// the source entry.
pub(super) struct EndpointPlan<R = PlannedRecipe> {
    primary: PlannedKernel<R>,
    siblings: Vec<PlannedKernel<R>>,
}

impl<R> EndpointPlan<R> {
    fn new(primary: PlannedKernel<R>, siblings: Vec<PlannedKernel<R>>) -> Self {
        Self { primary, siblings }
    }

    pub(super) fn into_parts(self) -> (PlannedKernel<R>, Vec<PlannedKernel<R>>) {
        (self.primary, self.siblings)
    }
}

/// Authoritative parallel recipes indexed by the endpoint that owns the
/// seeded kernel. Serial policy carries no recipe map.
pub(super) struct RecipeIndex<R = PlannedRecipe> {
    plans: Option<HashMap<CompilerFlowEndpoint, EndpointPlan<R>>>,
    required_elements: HashMap<CompilerFlowEndpoint, u32>,
}

impl RecipeIndex<AnalyzedRecipe> {
    fn parallel() -> Self {
        Self {
            plans: Some(HashMap::new()),
            required_elements: HashMap::new(),
        }
    }

    fn insert(&mut self, endpoint: CompilerFlowEndpoint, plan: EndpointPlan<AnalyzedRecipe>) -> Result<()> {
        let Some(endpoints) = &mut self.plans else {
            return Err(ParallelizeError::Invalid(
                "cannot add a parallel recipe to a serial recipe index".into(),
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
        let Self {
            plans,
            required_elements,
        } = self;
        let Some(endpoints) = plans else {
            return RecipeIndex::serial(required_elements);
        };
        RecipeIndex {
            plans: Some(
                endpoints
                    .into_iter()
                    .map(|(endpoint, plan)| {
                        let (primary, siblings) = plan.into_parts();
                        (
                            endpoint,
                            EndpointPlan::new(
                                bind_kernel(primary, resources),
                                siblings.into_iter().map(|kernel| bind_kernel(kernel, resources)).collect(),
                            ),
                        )
                    })
                    .collect(),
            ),
            required_elements,
        }
    }
}

impl RecipeIndex {
    fn serial(required_elements: HashMap<CompilerFlowEndpoint, u32>) -> Self {
        Self {
            plans: None,
            required_elements,
        }
    }

    pub(super) fn required_elements(&self, endpoint: CompilerFlowEndpoint) -> Option<u32> {
        self.required_elements.get(&endpoint).copied()
    }

    pub(super) fn take_endpoint(&mut self, endpoint: CompilerFlowEndpoint) -> Result<Option<EndpointPlan>> {
        let Some(endpoints) = &mut self.plans else {
            return Ok(None);
        };
        endpoints.remove(&endpoint).map(Some).ok_or_else(|| {
            ParallelizeError::Invalid(format!("flow endpoint {endpoint:?} has no target recipe"))
        })
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

    pub(super) fn serial_recipes(self) -> RecipeIndex {
        RecipeIndex::serial(self.recipes.required_elements)
    }
}

fn bind_kernel(kernel: PlannedKernel<AnalyzedRecipe>, resources: &ScratchBindings) -> PlannedKernel {
    let (body, output_projection, recipe) = kernel.into_parts();
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
        AnalyzedRecipe::Map(segment) => PlannedRecipe::Map(segment),
        AnalyzedRecipe::Serial(site) => PlannedRecipe::Serial(site),
        AnalyzedRecipe::Unchanged => PlannedRecipe::Unchanged,
    };
    PlannedKernel::new(body, output_projection, recipe)
}

/// Analyze every projected endpoint once. Recipes retain their projected body
/// and graph-local handles until emission consumes the endpoint plan.
pub(super) fn analyze(inner: &AllocatedProgram) -> Result<AnalyzedPlan> {
    let mut recipes = RecipeIndex::parallel();
    let mut requests = Vec::new();
    for (endpoint, entry) in inner.entries_with_endpoints() {
        let (plan, endpoint_requests, required_elements) =
            analyze_endpoint(entry, endpoint, &inner.resources)?;
        recipes.insert(endpoint, plan)?;
        if let Some(count) = required_elements {
            recipes.required_elements.insert(endpoint, count);
        }
        requests.extend(endpoint_requests);
    }
    Ok(AnalyzedPlan { recipes, requests })
}

fn analyze_endpoint(
    entry: &crate::egir::program::SemanticEntry,
    endpoint: CompilerFlowEndpoint,
    resources: &LogicalResourceArena,
) -> Result<(EndpointPlan<AnalyzedRecipe>, Vec<ScratchRequest>, Option<u32>)> {
    let projected = crate::egir::program::PlannedEntry::project(entry)?;
    let targets = RecipeTargets::collect(&projected);
    let required_elements = fixed_required_elements(&projected, &targets);
    let split = match endpoint {
        CompilerFlowEndpoint::Entry(_) => super::split_multidomain_seg_maps(&projected)?,
        CompilerFlowEndpoint::Materialization(_) => None,
    };
    let Some(split) = split else {
        let (primary, requests) = analyze_projected_kernel(projected, None, endpoint, resources, targets)?;
        return Ok((
            EndpointPlan::new(primary, Vec::new()),
            requests,
            required_elements,
        ));
    };
    let primary_targets = targets.remap(&split.primary.effect_sites);
    let (primary, mut requests) = analyze_projected_kernel(
        split.primary.entry,
        Some(split.primary.semantic_slots),
        endpoint,
        resources,
        primary_targets,
    )?;
    let mut siblings = Vec::with_capacity(split.siblings.len());
    for sibling in split.siblings {
        let sibling_targets = targets.remap(&sibling.effect_sites);
        let (sibling, sibling_requests) = analyze_projected_kernel(
            sibling.entry,
            Some(sibling.semantic_slots),
            endpoint,
            resources,
            sibling_targets,
        )?;
        siblings.push(sibling);
        requests.extend(sibling_requests);
    }
    Ok((EndpointPlan::new(primary, siblings), requests, required_elements))
}

fn fixed_required_elements(
    entry: &crate::egir::program::PlannedEntry,
    targets: &RecipeTargets,
) -> Option<u32> {
    let space = if targets.filters.len() == 1 {
        let SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) =
            &entry.graph.skeleton.effect(targets.filters[0]).kind
        else {
            return None;
        };
        &op.state.space
    } else {
        let site = targets.screma_site()?;
        let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &entry.graph.skeleton.effect(site).kind
        else {
            return None;
        };
        let screma::SemanticState::Segmented { space, .. } = op.semantic_state() else {
            return None;
        };
        space
    };
    space.dims.iter().try_fold(1u32, |count, extent| match extent {
        SegExtent::Fixed(size) => count.checked_mul(*size),
        _ => None,
    })
}

fn analyze_reduce_recipe(
    body: &crate::egir::program::PlannedEntry,
    endpoint: CompilerFlowEndpoint,
    resources: &LogicalResourceArena,
    located: LocatedScrema<'_>,
    lanes: &screma::Lanes,
    operators: &screma::NonEmpty<screma::Operator>,
) -> Result<(AnalyzedRecipe, Vec<ScratchRequest>)> {
    let serial = located.serial_recipe();
    let Some(candidate) = super::analyze_reduce_candidate(body, located, lanes, operators, resources)?
    else {
        return Ok((AnalyzedRecipe::Serial(serial), Vec::new()));
    };
    let requests = candidate
        .scratch_types()
        .cloned()
        .enumerate()
        .map(|(slot, elem_ty)| {
            scratch_request(
                endpoint,
                candidate.owner,
                slot,
                CompilerResourceKind::ReducePartial,
                elem_ty,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((AnalyzedRecipe::Reduce(candidate), requests))
}

fn analyze_scan_recipe(
    body: &crate::egir::program::PlannedEntry,
    endpoint: CompilerFlowEndpoint,
    located: LocatedScrema<'_>,
    lanes: &screma::Lanes,
    operators: &screma::NonEmpty<screma::Operator>,
) -> Result<(AnalyzedRecipe, Vec<ScratchRequest>)> {
    let serial = located.serial_recipe();
    let Some(candidate) = super::analyze_scan_candidate(body, located, lanes, operators)? else {
        return Ok((AnalyzedRecipe::Serial(serial), Vec::new()));
    };
    let requests = [
        CompilerResourceKind::ScanBlockSums,
        CompilerResourceKind::ScanBlockOffsets,
    ]
    .into_iter()
    .enumerate()
    .map(|(slot, kind)| {
        scratch_request(
            endpoint,
            candidate.owner,
            slot,
            kind,
            candidate.scratch_type.clone(),
        )
    })
    .collect::<Result<Vec<_>>>()?;
    Ok((AnalyzedRecipe::Scan(candidate), requests))
}

fn analyze_projected_kernel(
    body: crate::egir::program::PlannedEntry,
    output_projection: Option<Vec<usize>>,
    endpoint: CompilerFlowEndpoint,
    resources: &LogicalResourceArena,
    targets: RecipeTargets,
) -> Result<(PlannedKernel<AnalyzedRecipe>, Vec<ScratchRequest>)> {
    if targets.filters.len() == 1 {
        if let Some(CandidateSelection::Selected(candidate)) =
            super::analyze_filter_candidate(&body, targets.filters[0])
        {
            let requests = filter_scratch_requests(endpoint, &candidate);
            let kernel = PlannedKernel::new(body, output_projection, AnalyzedRecipe::Filter(candidate));
            return Ok((kernel, requests));
        }
    }

    let (recipe, requests) = match targets.screma_site() {
        Some(site) => {
            let located = located_screma(&body, site)?;
            match located.op {
                screma::Op::Reduce { lanes, operators, .. } => {
                    analyze_reduce_recipe(&body, endpoint, resources, located, lanes, operators)?
                }
                screma::Op::Scan { lanes, operators, .. } => {
                    analyze_scan_recipe(&body, endpoint, located, lanes, operators)?
                }
                screma::Op::Map { .. } => (AnalyzedRecipe::Map(located.segmented()?), Vec::new()),
                screma::Op::Composite { .. } => {
                    (AnalyzedRecipe::Serial(located.serial_recipe()), Vec::new())
                }
            }
        }
        None => (AnalyzedRecipe::Unchanged, Vec::new()),
    };
    Ok((PlannedKernel::new(body, output_projection, recipe), requests))
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
