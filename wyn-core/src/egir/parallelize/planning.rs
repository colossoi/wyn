//! Stage-owned kernel recipes and deterministic scratch allocation.

use crate::egir;
use crate::egir::soac::SegmentedMetadata;
use crate::ssa;
use crate::EntryId;
use crate::LookupMap;
use crate::ResourceId;
use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};

use polytype::Type;

use crate::ast::TypeName;
use crate::egir::soac::{hist, screma};
use crate::egir::types::{
    EGraph as FamilyGraph, Semantic as SemanticFamily, SideEffect as FamilySideEffect, SideEffectKind,
    SideEffectSite, Soac, SoacEffect,
};

use super::model::{ParallelizeError, Result};
use super::schedule::KernelDispatch;
use crate::egir::program::{
    CompilerResource, CompilerResourceKey, CompilerResourceKind, LogicalSize, SemanticOpId,
    SemanticResourceRef, StageOrigin,
};
use crate::egir::program::{
    KernelProgram, OutputSlotId, PlannedEntry, PlannedPublication, ResidentStorage,
};
use crate::egir::types::SegExtent;
use wyn_staged_ir::StagedIr;

use super::capabilities::{self, Strategy};

type Semantic = SemanticFamily<SemanticResourceRef>;
type EGraph = FamilyGraph<Semantic>;
type SideEffect = FamilySideEffect<Semantic>;

/// Semantic Scremas selected for parallel execution after optimization and
/// logical residency have finalized an endpoint graph.
pub(super) type ParallelScremas = HashSet<SemanticOpId>;

fn analyze_parallel_scremas(
    origin: &StageOrigin,
    entry: &egir::program::AllocatedEntry,
) -> ParallelScremas {
    let semantic_graph = OnceCell::new();
    let mut parallel = HashSet::new();
    let mut folds = Vec::new();
    for (_, block) in &entry.graph.skeleton.blocks {
        for effect in &block.side_effects {
            let SideEffectKind::Soac(SoacEffect(owner, Soac::Screma(op))) = &effect.kind else {
                continue;
            };
            let screma::SemanticState::Segmented(SegmentedMetadata { output_slots, .. }) =
                op.semantic_state()
            else {
                continue;
            };
            if entry.execution_model.is_compute()
                && (!output_slots.is_empty() || origin.generated_kind().is_some())
                && semantic_graph
                    .get_or_init(|| egir::semantic_graph::SemanticGraph::new(&entry.graph))
                    .value_consumer_count(owner)
                    == 0
            {
                parallel.insert(*owner);
                if op.form.layout().operator_input_count() != 0 {
                    folds.push(*owner);
                }
            }
        }
    }
    if folds.len() > 1 {
        parallel.retain(|operation| !folds.contains(operation));
    }
    parallel
}

#[derive(Clone, Copy)]
pub(super) struct LocatedHist<'a> {
    pub site: SideEffectSite,
    pub owner: SemanticOpId,
    pub op: &'a hist::Op<Semantic>,
}
#[derive(Clone, Copy)]
pub(super) struct LocatedScrema<'a> {
    pub site: SideEffectSite,
    pub effect: &'a SideEffect,
    pub owner: SemanticOpId,
    pub op: &'a screma::Op<Semantic>,
}

#[derive(Clone, Copy, Debug)]
pub struct OperationRef {
    pub site: SideEffectSite,
    pub owner: SemanticOpId,
}

impl OperationRef {
    pub(super) fn filter(self, entry: &PlannedEntry) -> Result<&egir::soac::filter::Op<Semantic>> {
        match &entry.graph.skeleton.effect(self.site).kind {
            SideEffectKind::Soac(SoacEffect(owner, Soac::Filter(op))) if *owner == self.owner => Ok(op),
            _ => Err("recipe filter identity differs from its body".into()),
        }
    }

    pub(super) fn screma(self, entry: &PlannedEntry) -> Result<LocatedScrema<'_>> {
        let located = located_screma(entry, self.site)?;
        if located.owner != self.owner {
            return Err("recipe operation identity differs from its body".into());
        }
        Ok(located)
    }
    pub(super) fn hist(self, entry: &PlannedEntry) -> Result<LocatedHist<'_>> {
        let located = located_hist(entry, self.site)?;
        if located.owner != self.owner {
            return Err("recipe operation identity differs from its body".into());
        }
        Ok(located)
    }
}

impl LocatedScrema<'_> {
    pub(super) fn reference(&self) -> OperationRef {
        OperationRef {
            site: self.site,
            owner: self.owner,
        }
    }

    pub(super) fn segmented(&self) -> Result<SegmentedMetadata<egir::program::SemanticResourceRef>> {
        let screma::SemanticState::Segmented(segment) = self.op.semantic_state() else {
            return Err(ParallelizeError::Invalid(
                "selected parallel Screma lost its segmented semantic facts".into(),
            ));
        };
        Ok(segment.clone())
    }
}

#[derive(Default)]
struct RecipeTargets {
    filters: Vec<SideEffectSite>,
    hists: Vec<SideEffectSite>,
    kernel_scremas: Vec<SideEffectSite>,
    promoted_folds: Vec<SideEffectSite>,
}

impl RecipeTargets {
    /// Target-relevant sites are classified once on the endpoint projection.
    /// Later output projection only remaps these handles.
    fn collect(entry: &egir::program::PlannedEntry, parallel: &ParallelScremas) -> Self {
        let mut targets = Self::default();
        for (block, contents) in &entry.graph.skeleton.blocks {
            for (index, effect) in contents.side_effects.iter().enumerate() {
                let site = SideEffectSite { block, index };
                match &effect.kind {
                    SideEffectKind::Soac(SoacEffect(_, Soac::Filter(_))) => targets.filters.push(site),
                    SideEffectKind::Soac(SoacEffect(_, Soac::Hist(op)))
                        if matches!(op.state, hist::SemanticState::Segmented(_)) =>
                    {
                        targets.hists.push(site);
                    }
                    SideEffectKind::Soac(SoacEffect(owner, Soac::Screma(op))) => {
                        match op.semantic_state() {
                            screma::SemanticState::Segmented(_) if parallel.contains(owner) => {
                                targets.kernel_scremas.push(site);
                            }
                            screma::SemanticState::Segmented(SegmentedMetadata {
                                output_slots, ..
                            }) if !output_slots.is_empty()
                                && (op.is_reduce() || !op.form.scans.is_empty())
                                && entry.execution_model.is_compute() =>
                            {
                                targets.promoted_folds.push(site);
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }
        targets
    }

    fn remap(&self, sites: &LookupMap<SideEffectSite, SideEffectSite>) -> Self {
        let remap =
            |source: &[SideEffectSite]| source.iter().filter_map(|site| sites.get(site).copied()).collect();
        Self {
            filters: remap(&self.filters),
            hists: remap(&self.hists),
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

fn located_hist(entry: &egir::program::PlannedEntry, site: SideEffectSite) -> Result<LocatedHist<'_>> {
    let effect = entry.graph.skeleton.effect(site);
    let SideEffectKind::Soac(SoacEffect(owner, Soac::Hist(op))) = &effect.kind else {
        return Err(ParallelizeError::Invalid(format!(
            "selected Hist site {site:?} no longer contains a Hist operation"
        )));
    };
    Ok(LocatedHist {
        site,
        owner: *owner,
        op,
    })
}
fn located_screma(entry: &egir::program::PlannedEntry, site: SideEffectSite) -> Result<LocatedScrema<'_>> {
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

pub(super) fn make_screma_serial(graph: &mut EGraph, operation: OperationRef) {
    let SideEffectKind::Soac(SoacEffect(owner, Soac::Screma(op))) =
        &mut graph.skeleton.effect_mut(operation.site).kind
    else {
        unreachable!("checked Screma recipe")
    };
    debug_assert_eq!(*owner, operation.owner);
    *op.semantic_state_mut() = screma::SemanticState::Serial;
}

#[derive(Debug)]
pub enum Recipe<R> {
    Filter(super::filter::FilterRecipe<R>),
    Hist(super::hist::HistRecipe<R>),
    Reduce(super::reduce::ReduceRecipe<R>),
    Scan(super::scan::ScanRecipe<R>),
    Map(OperationRef),
    Serial(OperationRef),
    Unchanged,
}

#[derive(Debug)]
pub struct RecipeKernel<R> {
    body: PlannedEntry,
    output_projection: Option<Vec<OutputSlotId>>,
    recipe: Recipe<R>,
}

impl<R> RecipeKernel<R> {
    pub fn body(&self) -> &PlannedEntry {
        &self.body
    }
    pub fn recipe(&self) -> &Recipe<R> {
        &self.recipe
    }
    pub fn output_projection(&self) -> Option<&[OutputSlotId]> {
        self.output_projection.as_deref()
    }
    pub(super) fn into_parts(self) -> (PlannedEntry, Option<Vec<usize>>, Recipe<R>) {
        (
            self.body,
            self.output_projection.map(|slots| slots.into_iter().map(|slot| slot.0).collect()),
            self.recipe,
        )
    }
    pub(super) fn entry_name(&self) -> &str {
        &self.body.name
    }
    pub(super) fn assign_entry_id(&mut self, id: EntryId) {
        self.body.id = id;
    }
    fn map_resources<T>(self, f: &mut impl FnMut(R) -> T) -> RecipeKernel<T> {
        RecipeKernel {
            body: self.body,
            output_projection: self.output_projection,
            recipe: self.recipe.map_resources(f),
        }
    }
}

#[derive(Debug)]
pub struct StagePlan<R> {
    pub(crate) publication: Option<PlannedPublication>,
    pub(crate) dispatch: KernelDispatch,
    pub(crate) required_elements: Option<u32>,
    primary: RecipeKernel<R>,
    siblings: Vec<RecipeKernel<R>>,
}

impl<R> StagePlan<R> {
    #[cfg(test)]
    pub(super) fn fixture(
        body: PlannedEntry,
        publication: Option<PlannedPublication>,
        dispatch: KernelDispatch,
    ) -> Self {
        Self {
            publication,
            required_elements: dispatch.required_elements(),
            dispatch,
            primary: RecipeKernel {
                body,
                output_projection: None,
                recipe: Recipe::Unchanged,
            },
            siblings: Vec::new(),
        }
    }

    pub fn kernels(&self) -> impl Iterator<Item = &RecipeKernel<R>> {
        std::iter::once(&self.primary).chain(&self.siblings)
    }
    pub fn publication(&self) -> Option<&PlannedPublication> {
        self.publication.as_ref()
    }
    pub fn dispatch(&self) -> &KernelDispatch {
        &self.dispatch
    }
    pub fn required_elements(&self) -> Option<u32> {
        self.required_elements
    }
    pub(super) fn entry(&self) -> &PlannedEntry {
        &self.primary.body
    }
    pub(super) fn into_parts(self) -> (RecipeKernel<R>, Vec<RecipeKernel<R>>) {
        (self.primary, self.siblings)
    }
    fn map_resources<T>(self, f: &mut impl FnMut(R) -> T) -> StagePlan<T> {
        StagePlan {
            publication: self.publication,
            dispatch: self.dispatch,
            required_elements: self.required_elements,
            primary: self.primary.map_resources(f),
            siblings: self.siblings.into_iter().map(|kernel| kernel.map_resources(f)).collect(),
        }
    }
}

pub type RecipeStages<R> = StagedIr<StagePlan<R>, Type<TypeName>, ResidentStorage, StageOrigin>;
pub type KernelRecipesPlanned = KernelProgram<RecipeStages<ScratchRef>>;
pub type RecipeScratchAllocated = KernelProgram<RecipeStages<ResourceId>>;

#[derive(Clone, Debug)]
pub enum ScratchRef {
    Existing(ResourceId),
    Allocate(ScratchRequirement),
}

#[derive(Clone, Debug)]
pub struct ScratchRequirement {
    pub key: CompilerResourceKey,
    pub elem_ty: Type<TypeName>,
    pub size: LogicalSize,
}

impl ScratchRef {
    pub(super) fn new(
        owner: SemanticOpId,
        kind: CompilerResourceKind,
        slot: usize,
        elem_ty: Type<TypeName>,
        size: LogicalSize,
    ) -> Self {
        Self::Allocate(ScratchRequirement {
            key: CompilerResourceKey { owner, kind, slot },
            elem_ty,
            size,
        })
    }
    pub(super) fn dispatch(
        owner: SemanticOpId,
        kind: CompilerResourceKind,
        slot: usize,
        elem_ty: Type<TypeName>,
    ) -> Result<Self> {
        let elem_bytes = ssa::layout::type_byte_size(&elem_ty).ok_or_else(|| {
            ParallelizeError::Invalid(format!(
                "parallel scratch for {owner:?} has no static element size"
            ))
        })?;
        Ok(Self::new(
            owner,
            kind,
            slot,
            elem_ty,
            LogicalSize::SameAsDispatch { elem_bytes },
        ))
    }
}

impl<R> Recipe<R> {
    pub fn resources(&self) -> Vec<&R> {
        use super::hist::HistRecipe;
        use super::scan::ScanPrefixes;
        match self {
            Self::Reduce(recipe) => recipe.accumulators.iter().map(|acc| &acc.partials).collect(),
            Self::Scan(recipe) => {
                let mut resources = vec![&recipe.block_sums, &recipe.block_offsets];
                if let ScanPrefixes::Scratch(prefixes) = &recipe.prefixes {
                    resources.push(prefixes);
                }
                resources
            }
            Self::Filter(recipe) => vec![
                &recipe.work.flags,
                &recipe.work.offsets,
                &recipe.work.block_sums,
                &recipe.work.block_offsets,
            ],
            Self::Hist(HistRecipe::Bucket { counts, overflow, .. }) => vec![counts, overflow],
            _ => Vec::new(),
        }
    }

    fn map_resources<T>(self, f: &mut impl FnMut(R) -> T) -> Recipe<T> {
        use super::hist::HistRecipe;
        use super::scan::ScanPrefixes;
        match self {
            Self::Reduce(recipe) => Recipe::Reduce(super::reduce::ReduceRecipe {
                operation: recipe.operation,
                routing: recipe.routing,
                accumulators: recipe
                    .accumulators
                    .into_iter()
                    .map(|acc| super::reduce::ReductionAccumulator {
                        capture_inputs: acc.capture_inputs,
                        partials: f(acc.partials),
                    })
                    .collect(),
            }),
            Self::Scan(recipe) => Recipe::Scan(super::scan::ScanRecipe {
                operation: recipe.operation,
                reduction_routing: recipe.reduction_routing,
                capture_inputs: recipe.capture_inputs,
                block_sums: f(recipe.block_sums),
                block_offsets: f(recipe.block_offsets),
                prefixes: match recipe.prefixes {
                    ScanPrefixes::DirectOutput => ScanPrefixes::DirectOutput,
                    ScanPrefixes::Scratch(r) => ScanPrefixes::Scratch(f(r)),
                },
            }),
            Self::Filter(recipe) => Recipe::Filter(super::filter::FilterRecipe {
                operation: recipe.operation,
                work: egir::soac::filter::WorkBuffers {
                    flags: f(recipe.work.flags),
                    offsets: f(recipe.work.offsets),
                    block_sums: f(recipe.work.block_sums),
                    block_offsets: f(recipe.work.block_offsets),
                },
            }),
            Self::Hist(HistRecipe::Atomic { operation, updates }) => {
                Recipe::Hist(HistRecipe::Atomic { operation, updates })
            }
            Self::Hist(HistRecipe::Bucket {
                operation,
                destination,
                input_resources,
                counts,
                overflow,
            }) => Recipe::Hist(HistRecipe::Bucket {
                operation,
                destination,
                input_resources,
                counts: f(counts),
                overflow: f(overflow),
            }),
            Self::Map(op) => Recipe::Map(op),
            Self::Serial(op) => Recipe::Serial(op),
            Self::Unchanged => Recipe::Unchanged,
        }
    }
}

pub(super) fn allocate_scratch(input: KernelRecipesPlanned) -> Result<RecipeScratchAllocated> {
    let (mut program, stages) = input.split_topology();
    let mut requests = Vec::new();
    for (stage, plan) in stages.stages() {
        for slot in plan.body().kernels().flat_map(|kernel| kernel.recipe.resources()) {
            match slot {
                ScratchRef::Allocate(request) => requests.push((stage, request)),
                ScratchRef::Existing(id) if !program.data.core.resources.contains(*id) => {
                    return Err(format!("recipe references missing resource {id:?}").into())
                }
                _ => {}
            }
        }
    }
    requests
        .sort_by_key(|(stage, request)| (*stage, request.key.owner, request.key.kind, request.key.slot));
    let arena = &mut program.data.core.resources;
    let mut bindings = HashMap::new();
    for (_, request) in requests {
        let key = request.key;
        let id = if let Some(id) = arena.compiler_resource(key.owner, key.kind, key.slot) {
            if arena[id].elem_ty() != &request.elem_ty || arena[id].size() != Some(&request.size) {
                return Err(format!("conflicting scratch requirements for {key:?}").into());
            }
            id
        } else {
            arena.allocate_compiler(
                CompilerResource::new(key.kind, Some(key.owner), key.slot),
                request.elem_ty.clone(),
                request.size.clone(),
            )
        };
        bindings.insert(key, id);
    }
    Ok(program.with_topology(stages.map_stage_bodies(|_, stage| {
        stage.map_resources(&mut |slot| match slot {
            ScratchRef::Existing(id) => id,
            ScratchRef::Allocate(request) => bindings[&request.key],
        })
    })))
}

pub(crate) fn construct_stage(
    program: &KernelProgram<()>,
    entry: egir::program::AllocatedEntry,
    origin: &StageOrigin,
    publication: Option<PlannedPublication>,
    dispatch: KernelDispatch,
) -> Result<StagePlan<ScratchRef>> {
    let parallel = analyze_parallel_scremas(origin, &entry);
    let projected = PlannedEntry::project(&entry)?.with_parallel_scremas(parallel.iter().copied());
    let targets = RecipeTargets::collect(&projected, &parallel);
    let required_elements = fixed_required_elements(&projected, &targets)
        .filter(|_| matches!(origin, StageOrigin::Authored) || origin.space().is_some())
        .or(dispatch.required_elements());
    let kernel = |body, slots, targets| construct_kernel(program, body, slots, targets);
    let (primary, siblings) = if program.data.profile.schedule == crate::SchedulePolicy::Serial {
        (
            RecipeKernel {
                body: projected,
                output_projection: None,
                recipe: Recipe::Unchanged,
            },
            Vec::new(),
        )
    } else if let Some(split) = if origin.generated_kind().is_none() {
        super::partition_entry_output_domains(&projected)?
    } else {
        None
    } {
        let primary_targets = targets.remap(&split.primary.effect_sites);
        let primary = kernel(
            split.primary.entry,
            Some(split.primary.semantic_slots),
            primary_targets,
        )?;
        let siblings = split
            .siblings
            .into_iter()
            .map(|sibling| {
                let targets = targets.remap(&sibling.effect_sites);
                kernel(sibling.entry, Some(sibling.semantic_slots), targets)
            })
            .collect::<Result<_>>()?;
        (primary, siblings)
    } else {
        (kernel(projected, None, targets)?, Vec::new())
    };
    Ok(StagePlan {
        publication,
        dispatch,
        required_elements,
        primary,
        siblings,
    })
}

fn fixed_required_elements(entry: &egir::program::PlannedEntry, targets: &RecipeTargets) -> Option<u32> {
    let space = if targets.filters.len() == 1 {
        let SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) =
            &entry.graph.skeleton.effect(targets.filters[0]).kind
        else {
            return None;
        };
        &op.state.segment.space
    } else if targets.hists.len() == 1 {
        let SideEffectKind::Soac(SoacEffect(_, Soac::Hist(op))) =
            &entry.graph.skeleton.effect(targets.hists[0]).kind
        else {
            return None;
        };
        let hist::SemanticState::Segmented(space) = &op.state else {
            return None;
        };
        space
    } else {
        let site = targets.screma_site()?;
        let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &entry.graph.skeleton.effect(site).kind
        else {
            return None;
        };
        let screma::SemanticState::Segmented(SegmentedMetadata { space, .. }) = op.semantic_state() else {
            return None;
        };
        space
    };
    space.dims().iter().try_fold(1u32, |count, extent| match extent {
        SegExtent::Fixed(size) => count.checked_mul(*size),
        _ => None,
    })
}

fn construct_kernel(
    program: &KernelProgram<()>,
    body: egir::program::PlannedEntry,
    output_projection: Option<Vec<usize>>,
    targets: RecipeTargets,
) -> Result<RecipeKernel<ScratchRef>> {
    let recipe = construct_recipe(program, &body, &targets)?;
    Ok(RecipeKernel {
        body,
        output_projection: output_projection.map(|slots| slots.into_iter().map(OutputSlotId).collect()),
        recipe,
    })
}

fn construct_recipe(
    program: &KernelProgram<()>,
    body: &PlannedEntry,
    targets: &RecipeTargets,
) -> Result<Recipe<ScratchRef>> {
    let mut bucket_histograms = 0usize;
    for site in &targets.hists {
        let located = located_hist(body, *site)?;
        if located
            .op
            .form
            .operations
            .iter()
            .any(|operation| matches!(operation.update, hist::Update::BucketInsert { .. }))
        {
            bucket_histograms += 1;
        }
    }
    if bucket_histograms != 0
        && (targets.hists.len() != 1
            || !targets.filters.is_empty()
            || !targets.kernel_scremas.is_empty()
            || !targets.promoted_folds.is_empty())
    {
        return Err(ParallelizeError::Invalid(
            "bucket_scatter cannot currently share one entry pipeline with another filter, histogram, scan, or reduction; split the operations into separate entries"
                .into(),
        ));
    }

    if targets.filters.len() == 1 {
        if let Some(candidate) = super::construct_filter_recipe(body, targets.filters[0]) {
            return Ok(Recipe::Filter(candidate));
        }
    }
    if let [site] = targets.hists.as_slice() {
        let located = located_hist(body, *site)?;
        if let Some(candidate) = super::hist::construct_hist_recipe(program, body, located) {
            return Ok(Recipe::Hist(candidate));
        }
        if bucket_histograms != 0 {
            return Err(ParallelizeError::Invalid(
                "bucket_scatter could not be lowered to its required init/insert/finish pipeline".into(),
            ));
        }
    }
    Ok(match targets.screma_site() {
        Some(site) => {
            let located = located_screma(body, site)?;
            match capabilities::classify(located.op) {
                Strategy::Reduce => {
                    super::construct_reduce_recipe(body, located, &program.data.core.resources)?
                        .map(Recipe::Reduce)
                        .unwrap_or(Recipe::Serial(located.reference()))
                }
                Strategy::Scan => {
                    super::construct_scan_recipe(body, located, &program.data.core.resources)?
                        .map(Recipe::Scan)
                        .unwrap_or(Recipe::Serial(located.reference()))
                }
                Strategy::Map => Recipe::Map(located.reference()),
                Strategy::Serial => Recipe::Serial(located.reference()),
            }
        }
        None => Recipe::Unchanged,
    })
}

#[cfg(test)]
#[path = "planning_tests.rs"]
mod tests;
