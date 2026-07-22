//! First-class compute schedule produced by EGIR parallelization.
//!
//! A schedule owns the relationship between generated entry points, dispatch
//! domains, resource accesses, and phase dependencies.  The pipeline
//! descriptor is a publication format derived from this graph after every
//! lowering has finished; it is not mutated while an individual lowering is
//! still speculative.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use crate::egir::program::{
    CompilerFlowEndpoint, CompilerResourceFlow, MaterializationId, MaterializationKind,
    MaterializationRequirement, OutputSlotId, PlannedEntry, PlannedPublication, SemanticEntry,
    SemanticEntryId, SemanticProgram, SemanticResourceRef,
};
use crate::egir::soac::filter;
use crate::egir::types::{Scheduled, SegExtent};
use crate::flow::ExecutionModel;
use crate::pipeline_descriptor::{
    Binding, ComputePipeline, ComputeStage, DispatchLen, DispatchSize, Pipeline, PipelineDescriptor,
};
use crate::{BindingRef, ResourceId};

use super::declared_resources;

mod finalize;
#[cfg(any(test, debug_assertions))]
mod validation;

#[cfg(test)]
mod tests;

/// A complete module-level compute schedule.
#[derive(Debug, Default)]
pub(in crate::egir) struct KernelPlan {
    phases: Vec<KernelPhase>,
    pipelines: Vec<ScheduledPipeline>,
    next_pipeline_order: usize,
    flow_sources: HashMap<CompilerFlowEndpoint, KernelId>,
    source_entries: Vec<SourceEntryPlan>,
}

fn record_seeded_kernel(
    seeded: &mut [Option<KernelId>],
    source: SemanticEntryId,
    kernel: KernelId,
    name: &str,
) -> Result<(), String> {
    let slot = &mut seeded[source.index()];
    if let Some(existing) = slot.replace(kernel) {
        return Err(format!(
            "semantic entry `{name}` is assigned to kernels {existing:?} and {kernel:?}"
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct SourceEntryPlan {
    publication: PlannedPublication,
    primary: KernelId,
    output_owners: HashMap<OutputSlotId, KernelId>,
}

impl KernelPlan {
    pub(in crate::egir) fn physical_entries(&self) -> impl Iterator<Item = &PlannedEntry<Scheduled>> {
        self.phases().map(|phase| phase.entry.as_ref())
    }
}

/// Stable identity of a physical kernel in a plan. Dependencies use this id
/// instead of vector positions so insertion cannot silently retarget an edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KernelId(u32);

impl KernelId {
    fn index(self) -> usize {
        self.0 as usize
    }

    #[cfg(test)]
    pub(super) const fn for_test(index: u32) -> Self {
        Self(index)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(super) enum KernelMutationError {
    #[error("{0}")]
    InvalidKernel(String),
    #[error("adding resource dependency {reader:?} -> {writer:?} would create a cycle")]
    DependencyCycle {
        reader: KernelId,
        writer: KernelId,
    },
}

impl From<String> for KernelMutationError {
    fn from(error: String) -> Self {
        Self::InvalidKernel(error)
    }
}

/// A body plus the target facts established by the code that created it.
/// Preparing the body only changes EGIR phase; it never re-inspects the graph
/// to infer dispatch, resources, or output ownership.
pub(super) struct PhaseSpec {
    body: PlannedEntry,
    label: &'static str,
    filter_plan: Option<super::prepare::ParallelFilterPlan>,
    expected_compute: bool,
    dispatch: KernelDispatch,
    resources: Vec<ScheduledResource>,
    serial_single_workgroup: bool,
    output_projection: Option<Vec<OutputSlotId>>,
}

impl PhaseSpec {
    pub(super) fn compute(body: PlannedEntry, dispatch: KernelDispatch, label: &'static str) -> Self {
        let resources = declared_resources(&body.resource_declarations);
        Self {
            body,
            label,
            filter_plan: None,
            expected_compute: true,
            dispatch,
            resources,
            serial_single_workgroup: false,
            output_projection: None,
        }
    }

    pub(super) fn graphics(body: PlannedEntry, dispatch: KernelDispatch) -> Self {
        let resources = declared_resources(&body.resource_declarations);
        Self {
            body,
            label: "graphics_passthrough",
            filter_plan: None,
            expected_compute: false,
            dispatch,
            resources,
            serial_single_workgroup: false,
            output_projection: None,
        }
    }

    pub(super) fn filter(
        body: PlannedEntry,
        dispatch: KernelDispatch,
        stage: filter::ParallelStage,
        config: filter::ParallelConfig<SemanticResourceRef>,
        storage: filter::RuntimeStorage<SemanticResourceRef>,
    ) -> Self {
        let label = match stage {
            filter::ParallelStage::Flags => "filter_flags",
            filter::ParallelStage::Scan => "filter_scan",
            filter::ParallelStage::Scatter => "filter_scatter",
        };
        let resources = declared_resources(&body.resource_declarations);
        Self {
            body,
            label,
            filter_plan: Some(super::prepare::ParallelFilterPlan::new(stage, config, storage)),
            expected_compute: true,
            dispatch,
            resources,
            serial_single_workgroup: false,
            output_projection: None,
        }
    }

    pub(super) fn with_resources(mut self, resources: Vec<ScheduledResource>) -> Self {
        self.resources = resources;
        self
    }

    pub(super) fn with_output_projection(mut self, outputs: Option<Vec<usize>>) -> Self {
        self.output_projection = outputs.map(|outputs| outputs.into_iter().map(OutputSlotId).collect());
        self
    }

    fn prepare(self) -> Result<PreparedPhase, String> {
        let entry = super::prepare::entry(self.body, self.filter_plan)?;
        if entry.execution_model.is_compute() != self.expected_compute {
            let expected = if self.expected_compute { "compute" } else { "graphics" };
            return Err(format!("entry `{}` cannot use a {expected} body", entry.name));
        }
        let required_elements = match &self.dispatch.domain {
            KernelDomain::Elements(DispatchLen::Fixed { count }) => Some(*count),
            _ => None,
        };
        Ok(PreparedPhase {
            label: self.label,
            entry: Arc::new(entry),
            dispatch: self.dispatch,
            resources: self.resources,
            serial_single_workgroup: self.serial_single_workgroup,
            required_elements,
            output_projection: self.output_projection,
        })
    }
}

struct PreparedPhase {
    label: &'static str,
    entry: Arc<PlannedEntry<Scheduled>>,
    dispatch: KernelDispatch,
    resources: Vec<ScheduledResource>,
    serial_single_workgroup: bool,
    required_elements: Option<u32>,
    output_projection: Option<Vec<OutputSlotId>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputRouteProjection {
    pub semantic_slot: OutputSlotId,
    pub physical_slot: OutputSlotId,
}

/// Ordered phases that share one host binding table.
#[derive(Clone, Debug)]
struct ScheduledPipeline {
    id: PipelineId,
    /// Original descriptor position, retained solely for stable publication.
    order: usize,
    /// Non-stage host metadata and already-published source bindings.
    template: ComputePipeline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PipelineId(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PhaseGroup {
    Pipeline(PipelineId),
    Graphics,
    Unpublished,
}

impl PhaseGroup {
    fn is_graphics(self) -> bool {
        matches!(self, Self::Graphics)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PhasePlacement {
    group: PhaseGroup,
    order: usize,
}

/// One executable kernel phase.
#[derive(Clone, Debug)]
struct KernelPhase {
    placement: PhasePlacement,
    /// Typed identity used by compiler-resource flow edges. It is separate
    /// from the projected source ABI because generated requirements have no
    /// semantic entry ABI of their own.
    flow_source: Option<CompilerFlowEndpoint>,
    label: &'static str,
    entry: Arc<PlannedEntry<Scheduled>>,
    source_entry: Option<SemanticEntryId>,
    output_routes: Vec<OutputRouteProjection>,
    dispatch: KernelDispatch,
    resources: Vec<ScheduledResource>,
    /// Materialization kernels collapse to one workgroup under serial policy.
    /// This is an execution constraint, unlike the diagnostic `label`.
    serial_single_workgroup: bool,
    /// Semantic work-item requirement retained for explicit-dispatch coverage
    /// checks without inspecting the scheduled graph.
    required_elements: Option<u32>,
    /// Stable kernel identities that must complete first.
    dependencies: Vec<KernelId>,
}

impl KernelPhase {
    fn entry_point(&self) -> &str {
        &self.entry.name
    }

    fn workgroup_size(&self) -> (u32, u32, u32) {
        execution_workgroup(&self.entry.execution_model)
    }

    fn resources(&self) -> &[ScheduledResource] {
        &self.resources
    }

    fn replace_body(&mut self, spec: PhaseSpec) -> Result<Option<Vec<OutputSlotId>>, String> {
        let prepared = spec.prepare()?;
        self.label = prepared.label;
        self.entry = prepared.entry;
        self.resources = prepared.resources;
        self.serial_single_workgroup = prepared.serial_single_workgroup;
        self.required_elements = prepared.required_elements;
        if !self.dispatch.explicit {
            self.dispatch.domain = prepared.dispatch.domain;
        }
        if let Some(outputs) = &prepared.output_projection {
            self.output_routes = outputs
                .iter()
                .copied()
                .enumerate()
                .map(|(physical, semantic_slot)| OutputRouteProjection {
                    semantic_slot,
                    physical_slot: OutputSlotId(physical),
                })
                .collect();
        }
        Ok(prepared.output_projection)
    }

    fn replace_scheduled_compute(
        &mut self,
        entry: PlannedEntry<Scheduled>,
        label: &'static str,
    ) -> Result<(), String> {
        if !entry.execution_model.is_compute() {
            return Err(format!("entry `{}` cannot use a compute body", entry.name));
        }
        self.label = label;
        self.entry = Arc::new(entry);
        Ok(())
    }
}

/// Graph-free schedule summary retained after physical EGIR has lowered to SSA.
#[derive(Clone, Debug)]
pub struct KernelPlanSummary {
    phases: Vec<KernelPhaseSummary>,
}

impl KernelPlanSummary {
    pub fn phases(&self) -> impl Iterator<Item = &KernelPhaseSummary> {
        self.phases.iter()
    }
}

#[derive(Clone, Debug)]
pub struct KernelPhaseSummary {
    pub id: KernelId,
    pub entry_point: String,
    pub label: String,
    pub source_entry: Option<SemanticEntryId>,
    pub output_routes: Vec<OutputRouteProjection>,
    pub workgroup_size: (u32, u32, u32),
    pub domain: KernelDomain,
    pub resources: Vec<ScheduledResource>,
    pub dependencies: Vec<KernelId>,
}

impl From<&KernelPlan> for KernelPlanSummary {
    fn from(plan: &KernelPlan) -> Self {
        let phases =
            plan.phases_with_ids().map(|(id, phase)| KernelPhaseSummary::from_phase(id, phase)).collect();
        Self { phases }
    }
}

impl KernelPhaseSummary {
    fn from_phase(id: KernelId, phase: &KernelPhase) -> Self {
        Self {
            id,
            entry_point: phase.entry_point().to_owned(),
            label: phase.label.to_owned(),
            source_entry: phase.source_entry,
            output_routes: phase.output_routes.clone(),
            workgroup_size: phase.workgroup_size(),
            domain: phase.dispatch.domain.clone(),
            resources: phase.resources().to_vec(),
            dependencies: phase.dependencies.clone(),
        }
    }
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

/// A launch domain and whether the host explicitly fixed it. Recipe-provided
/// domains replace inferred descriptor placeholders but never explicit host
/// dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct KernelDispatch {
    domain: KernelDomain,
    explicit: bool,
}

impl KernelDispatch {
    pub(super) fn inferred(baseline: KernelDomain) -> Self {
        Self {
            domain: baseline,
            explicit: false,
        }
    }

    pub(super) fn explicit(domain: KernelDomain) -> Self {
        Self {
            domain,
            explicit: true,
        }
    }
}

/// Conservative resource access for a phase.
pub type ScheduledResource = crate::egir::ir::SegResourceAccess<ResourceId>;

impl KernelPlan {
    fn phases(&self) -> impl Iterator<Item = &KernelPhase> {
        self.phases_with_ids().map(|(_, phase)| phase)
    }

    fn phases_with_ids(&self) -> impl Iterator<Item = (KernelId, &KernelPhase)> {
        self.ordered_phase_ids().into_iter().map(|id| (id, self.phase(id)))
    }

    fn ordered_phase_ids(&self) -> Vec<KernelId> {
        let mut pipelines = self.pipelines.iter().collect::<Vec<_>>();
        pipelines.sort_by_key(|pipeline| pipeline.order);
        let mut ids = pipelines
            .into_iter()
            .flat_map(|pipeline| self.phase_ids_in(PhaseGroup::Pipeline(pipeline.id)))
            .collect::<Vec<_>>();
        ids.extend(self.phase_ids_in(PhaseGroup::Graphics));
        ids.extend(self.phase_ids_in(PhaseGroup::Unpublished));
        ids
    }

    fn phase_ids_in(&self, group: PhaseGroup) -> Vec<KernelId> {
        let mut ids = self
            .phases
            .iter()
            .enumerate()
            .filter_map(|(index, phase)| {
                (phase.placement.group == group).then_some((phase.placement.order, KernelId(index as u32)))
            })
            .collect::<Vec<_>>();
        ids.sort_by_key(|(order, _)| *order);
        ids.into_iter().map(|(_, id)| id).collect()
    }

    pub(super) fn contains_flow_source(&self, source: CompilerFlowEndpoint) -> bool {
        self.flow_sources.contains_key(&source)
    }

    pub(super) fn kernel_for_flow_source(&self, source: CompilerFlowEndpoint) -> Option<KernelId> {
        self.flow_sources.get(&source).copied()
    }

    fn flow_resource_phases(
        &self,
        source: CompilerFlowEndpoint,
        resource: ResourceId,
        writes: bool,
    ) -> impl Iterator<Item = (KernelId, &KernelPhase)> {
        self.phases_with_ids().filter(move |(_, phase)| {
            phase.flow_source == Some(source)
                && phase.resources().iter().any(|item| {
                    item.resource == resource
                        && if writes { item.access.writes() } else { item.access.reads() }
                })
        })
    }

    pub(super) fn from_descriptor(
        descriptor: &PipelineDescriptor,
        semantic: &SemanticProgram,
    ) -> Result<Self, String> {
        let resources = &semantic.resources;
        let host_resources = crate::egir::program::host_resource_map(resources);
        let mut seeded = vec![None; semantic.entry_points.len()];
        let mut by_name = HashMap::new();
        for source in semantic.entry_ids() {
            let entry = &semantic[source];
            if by_name.insert(entry.name.as_str(), (source, entry)).is_some() {
                return Err(format!("duplicate semantic entry `{}`", entry.name));
            }
        }
        let mut phases = Vec::new();
        let mut pipelines = Vec::new();
        let mut flow_sources = HashMap::new();
        for (order, pipeline) in descriptor.pipelines.iter().enumerate() {
            let Pipeline::Compute(template) = pipeline else {
                continue;
            };
            let pipeline_id = PipelineId(pipelines.len() as u32);
            for (stage_order, stage) in template.stages.iter().enumerate() {
                let selection = domain_selection_from_stage(stage, &host_resources)?;
                let (source, entry) = by_name.get(stage.entry_point.as_str()).ok_or_else(|| {
                    format!("descriptor stage `{}` has no semantic entry", stage.entry_point)
                })?;
                let phase = phase_from_entry(
                    Some(*source),
                    entry,
                    selection,
                    "serial_compute",
                    PhasePlacement {
                        group: PhaseGroup::Pipeline(pipeline_id),
                        order: stage_order,
                    },
                )?;
                let id = KernelId(phases.len() as u32);
                record_seeded_kernel(&mut seeded, *source, id, &entry.name)?;
                flow_sources.insert(CompilerFlowEndpoint::Entry(*source), id);
                phases.push(phase);
            }
            pipelines.push(ScheduledPipeline {
                id: pipeline_id,
                order,
                template: template.clone(),
            });
        }
        let mut graphics_order = 0;
        for pipeline in &descriptor.pipelines {
            let Pipeline::Graphics(graphics) = pipeline else {
                continue;
            };
            for stage in &graphics.stages {
                let (source, entry) = by_name.get(stage.entry_point.as_str()).ok_or_else(|| {
                    format!("graphics stage `{}` has no semantic entry", stage.entry_point)
                })?;
                let phase = graphics_passthrough_phase(
                    *source,
                    entry,
                    PhasePlacement {
                        group: PhaseGroup::Graphics,
                        order: graphics_order,
                    },
                )?;
                graphics_order += 1;
                let id = KernelId(phases.len() as u32);
                record_seeded_kernel(&mut seeded, *source, id, &entry.name)?;
                flow_sources.insert(CompilerFlowEndpoint::Entry(*source), id);
                phases.push(phase);
            }
        }
        let published_names = descriptor
            .pipelines
            .iter()
            .flat_map(|pipeline| match pipeline {
                Pipeline::Compute(compute) => {
                    compute.stages.iter().map(|stage| stage.entry_point.as_str()).collect::<Vec<_>>()
                }
                Pipeline::Graphics(graphics) => {
                    graphics.stages.iter().map(|stage| stage.entry_point.as_str()).collect::<Vec<_>>()
                }
            })
            .collect::<HashSet<_>>();
        let mut unpublished_order = 0;
        for source in semantic.entry_ids() {
            let entry = &semantic[source];
            if published_names.contains(entry.name.as_str()) {
                continue;
            }
            let placement = PhasePlacement {
                group: PhaseGroup::Unpublished,
                order: unpublished_order,
            };
            unpublished_order += 1;
            let result = if entry.execution_model.is_compute() {
                phase_from_entry(
                    Some(source),
                    entry,
                    KernelDispatch::inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                    "serial_compute",
                    placement,
                )
            } else {
                graphics_passthrough_phase(source, entry, placement)
            };
            let phase = result?;
            let id = KernelId(phases.len() as u32);
            record_seeded_kernel(&mut seeded, source, id, &entry.name)?;
            flow_sources.insert(CompilerFlowEndpoint::Entry(source), id);
            phases.push(phase);
        }
        let seeded = seeded
            .into_iter()
            .enumerate()
            .map(|(index, kernel)| {
                kernel.ok_or_else(|| format!("semantic entry {} has no seeded kernel", index))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let source_entries = semantic
            .entry_ids()
            .map(|source| {
                let entry = &semantic[source];
                let primary = seeded[source.index()];
                SourceEntryPlan {
                    publication: PlannedPublication::from_semantic(entry),
                    primary,
                    output_owners: entry.output_routes.iter().map(|route| (route.slot, primary)).collect(),
                }
            })
            .collect();
        Ok(Self {
            phases,
            pipelines,
            next_pipeline_order: descriptor.pipelines.len(),
            flow_sources,
            source_entries,
        })
    }

    pub(super) fn primary_kernel(&self, source: SemanticEntryId) -> KernelId {
        self.source_entries[source.index()].primary
    }

    pub(super) fn set_required_elements(&mut self, endpoint: CompilerFlowEndpoint, count: u32) {
        if let Some(kernel) = self.flow_sources.get(&endpoint).copied() {
            self.phase_mut(kernel).required_elements = Some(count);
        }
    }

    fn push_phase(&mut self, phase: KernelPhase) -> KernelId {
        let id = KernelId(self.phases.len() as u32);
        self.phases.push(phase);
        id
    }

    fn next_order(&self, group: PhaseGroup) -> usize {
        self.phases
            .iter()
            .filter(|phase| phase.placement.group == group)
            .map(|phase| phase.placement.order + 1)
            .max()
            .unwrap_or(0)
    }

    fn shift_orders_from(&mut self, group: PhaseGroup, order: usize, amount: usize) {
        for phase in &mut self.phases {
            if phase.placement.group == group && phase.placement.order >= order {
                phase.placement.order += amount;
            }
        }
    }

    pub(super) fn add_materialization_before(
        &mut self,
        consumer: KernelId,
        requirement_id: MaterializationId,
        requirement: &MaterializationRequirement,
    ) -> Result<KernelId, KernelMutationError> {
        let consumer_placement = self.phase(consumer).placement;
        let generated_pipeline = consumer_placement.group.is_graphics().then(|| {
            let id = PipelineId(self.pipelines.len() as u32);
            let pipeline = ScheduledPipeline {
                id,
                order: self.next_pipeline_order,
                template: ComputePipeline {
                    bindings: Vec::new(),
                    stages: Vec::new(),
                    default_total_threads: None,
                    feedback: Vec::new(),
                },
            };
            (id, pipeline)
        });
        let placement = if let Some((pipeline, _)) = &generated_pipeline {
            PhasePlacement {
                group: PhaseGroup::Pipeline(*pipeline),
                order: 0,
            }
        } else {
            self.shift_orders_from(consumer_placement.group, consumer_placement.order, 1);
            PhasePlacement {
                group: consumer_placement.group,
                order: consumer_placement.order,
            }
        };
        let dependencies = self.phase(consumer).dependencies.clone();
        let phase = phase_from_materialization(requirement_id, requirement, dependencies, placement)?;
        if let Some((_, pipeline)) = generated_pipeline {
            self.next_pipeline_order += 1;
            self.pipelines.push(pipeline);
        }
        let id = self.push_phase(phase);
        self.flow_sources.insert(CompilerFlowEndpoint::Materialization(requirement_id), id);
        if !consumer_placement.group.is_graphics() {
            self.phase_mut(consumer).dependencies = vec![id];
        }
        Ok(id)
    }

    pub(super) fn add_sibling(
        &mut self,
        parent: KernelId,
        spec: PhaseSpec,
    ) -> Result<KernelId, KernelMutationError> {
        let placement = self.phase(parent).placement;
        if placement.group.is_graphics() {
            return Err(KernelMutationError::InvalidKernel(
                "graphics passthroughs cannot own compute siblings".into(),
            ));
        }
        let (source_entry, flow_source) = {
            let parent = self.phase(parent);
            (parent.source_entry, parent.flow_source)
        };
        let phase = phase_from_body(
            flow_source,
            source_entry,
            PhasePlacement {
                group: placement.group,
                order: self.next_order(placement.group),
            },
            spec,
        )?;
        Ok(self.push_phase(phase))
    }

    pub(super) fn make_serial(&mut self) -> Result<(), KernelMutationError> {
        let kernels = self
            .phases
            .iter()
            .enumerate()
            .filter_map(|(index, phase)| {
                (!phase.placement.group.is_graphics()).then_some(KernelId(index as u32))
            })
            .collect::<Vec<_>>();
        for kernel in kernels {
            let phase = self.phase_mut(kernel);
            let mut entry = phase.entry.as_ref().clone();
            super::prepare::force_serial(&mut entry.graph);
            let label = if phase.serial_single_workgroup { phase.label } else { "serial_compute" };
            phase.replace_scheduled_compute(entry, label)?;
            if phase.serial_single_workgroup {
                let domain = KernelDomain::Fixed { x: 1, y: 1, z: 1 };
                phase.dispatch = KernelDispatch::explicit(domain);
            }
        }
        Ok(())
    }

    fn record_output_owners(
        &mut self,
        kernel: KernelId,
        outputs: Vec<OutputSlotId>,
    ) -> Result<(), KernelMutationError> {
        let source_entry = self.phase(kernel).source_entry;
        if let Some(source) = source_entry {
            let entry = self.source_entries.get_mut(source.index()).ok_or_else(|| {
                KernelMutationError::InvalidKernel(format!(
                    "kernel {kernel:?} references missing semantic entry {source:?}"
                ))
            })?;
            let mut unique = HashSet::new();
            for output in &outputs {
                if output.0 >= entry.publication.outputs.len() || !unique.insert(*output) {
                    return Err(KernelMutationError::InvalidKernel(format!(
                        "kernel {kernel:?} has invalid output slot {output:?}"
                    )));
                }
                entry.output_owners.insert(*output, kernel);
            }
        }
        Ok(())
    }

    pub(super) fn commit_kernel(
        &mut self,
        kernel: KernelId,
        spec: PhaseSpec,
    ) -> Result<KernelId, KernelMutationError> {
        let output_projection = self.phase_mut(kernel).replace_body(spec)?;
        if let Some(outputs) = output_projection {
            self.record_output_owners(kernel, outputs)?;
        }
        Ok(kernel)
    }

    /// Install a complete phase chain transactionally. `kernel` remains the
    /// anchor identity; surrounding phases receive ids only after all bodies
    /// have prepared successfully.
    pub(super) fn replace_chain(
        &mut self,
        kernel: KernelId,
        before: Vec<PhaseSpec>,
        anchor: PhaseSpec,
        after: Vec<PhaseSpec>,
    ) -> Result<(), KernelMutationError> {
        let original = self.phase(kernel).clone();
        if original.placement.group.is_graphics() {
            return Err(KernelMutationError::InvalidKernel(
                "graphics passthroughs cannot be replaced by a compute chain".into(),
            ));
        }
        let source_entry = original.source_entry;
        let flow_source = original.flow_source;
        let group = original.placement.group;
        let anchor_order = original.placement.order;
        let mut additions = Vec::with_capacity(before.len() + after.len());
        let mut dependencies = original.dependencies.clone();
        let first_new_id = self.phases.len() as u32;
        for (index, spec) in before.into_iter().enumerate() {
            let id = KernelId(first_new_id + index as u32);
            let mut phase = phase_from_body(
                flow_source,
                source_entry,
                PhasePlacement {
                    group,
                    order: anchor_order + index,
                },
                spec,
            )?;
            phase.dependencies = dependencies;
            dependencies = vec![id];
            additions.push(phase);
        }
        let mut anchor_phase = original.clone();
        let output_projection = anchor_phase.replace_body(anchor)?;
        anchor_phase.placement.order = anchor_order + additions.len();
        anchor_phase.dependencies = dependencies;
        dependencies = vec![kernel];
        for (index, spec) in after.into_iter().enumerate() {
            let id = KernelId(first_new_id + additions.len() as u32);
            let mut phase = phase_from_body(
                flow_source,
                source_entry,
                PhasePlacement {
                    group,
                    order: anchor_phase.placement.order + index + 1,
                },
                spec,
            )?;
            phase.dependencies = dependencies;
            dependencies = vec![id];
            additions.push(phase);
        }

        let tail = dependencies[0];
        let dependents = self
            .phase_ids_in(group)
            .into_iter()
            .filter(|id| self.phase(*id).placement.order > anchor_order)
            .collect::<Vec<_>>();
        for dependent in dependents {
            for dependency in &mut self.phase_mut(dependent).dependencies {
                if *dependency == kernel {
                    *dependency = tail;
                }
            }
            self.phase_mut(dependent).dependencies.sort_unstable();
            self.phase_mut(dependent).dependencies.dedup();
        }
        self.shift_orders_from(group, anchor_order + 1, additions.len());
        *self.phase_mut(kernel) = anchor_phase;
        if let Some(outputs) = output_projection {
            self.record_output_owners(kernel, outputs)?;
        }
        for phase in additions {
            self.push_phase(phase);
        }
        Ok(())
    }

    pub(super) fn coalesce_resource_flows(
        &mut self,
        flows: &[(ResourceId, CompilerResourceFlow)],
    ) -> Result<(), KernelMutationError> {
        self.connect_resource_flows(flows)?;
        self.merge_connected_pipelines()
    }

    fn connect_resource_flows(
        &mut self,
        flows: &[(ResourceId, CompilerResourceFlow)],
    ) -> Result<(), KernelMutationError> {
        for (resource, flow) in flows {
            let writers = self
                .flow_resource_phases(flow.producer, *resource, true)
                .map(|(id, _)| id)
                .collect::<Vec<_>>();
            if writers.is_empty() {
                return Err(KernelMutationError::InvalidKernel(format!(
                    "flow producer {:?} does not declare a writer for {:?}",
                    flow.producer, resource
                )));
            }
            for consumer in &flow.consumers {
                let readers = self
                    .flow_resource_phases(*consumer, *resource, false)
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>();
                // Allocation records endpoint-level consumers before output-domain
                // projection. Projection may prove one result of a multi-result
                // materialization dead in every physical kernel for that endpoint.
                // The phase-owned resource facts are authoritative here: connect
                // every surviving reader, but do not resurrect a pruned edge.
                if readers.is_empty() {
                    continue;
                }
                for reader in readers {
                    for writer in writers.iter().copied().filter(|writer| *writer != reader) {
                        self.add_dependency(reader, writer)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn phase(&self, id: KernelId) -> &KernelPhase {
        &self.phases[id.index()]
    }

    fn phase_mut(&mut self, id: KernelId) -> &mut KernelPhase {
        &mut self.phases[id.index()]
    }

    fn add_dependency(&mut self, reader: KernelId, writer: KernelId) -> Result<(), KernelMutationError> {
        let reader_phase = self.phase(reader);
        if reader_phase.dependencies.contains(&writer) {
            return Ok(());
        }
        if self.depends_on(writer, reader) {
            return Err(KernelMutationError::DependencyCycle { reader, writer });
        }
        let phase = self.phase_mut(reader);
        phase.dependencies.push(writer);
        phase.dependencies.sort_unstable();
        Ok(())
    }

    fn depends_on(&self, start: KernelId, target: KernelId) -> bool {
        let mut pending = vec![start];
        let mut visited = HashSet::new();
        while let Some(kernel) = pending.pop() {
            if !visited.insert(kernel) {
                continue;
            }
            let phase = self.phase(kernel);
            for dependency in &phase.dependencies {
                if *dependency == target {
                    return true;
                }
                pending.push(*dependency);
            }
        }
        false
    }

    fn merge_connected_pipelines(&mut self) -> Result<(), KernelMutationError> {
        let pipeline_indices = self
            .pipelines
            .iter()
            .enumerate()
            .map(|(index, pipeline)| (pipeline.id, index))
            .collect::<HashMap<_, _>>();
        let owner = self
            .phases
            .iter()
            .enumerate()
            .filter_map(|(index, phase)| match phase.placement.group {
                PhaseGroup::Pipeline(pipeline) => {
                    Some((KernelId(index as u32), pipeline_indices[&pipeline]))
                }
                _ => None,
            })
            .collect::<HashMap<_, _>>();
        let mut pipeline_components = super::DisjointSets::new(self.pipelines.len());
        for (&phase, &pipeline) in &owner {
            for dependency in &self.phase(phase).dependencies {
                if let Some(&other) = owner.get(dependency) {
                    pipeline_components.merge(pipeline, other);
                }
            }
        }
        let mut components = BTreeMap::<usize, Vec<usize>>::new();
        for pipeline in 0..self.pipelines.len() {
            let root = pipeline_components.representative(pipeline);
            components.entry(root).or_default().push(pipeline);
        }
        let mut slots = std::mem::take(&mut self.pipelines).into_iter().map(Some).collect::<Vec<_>>();
        for mut group in components.into_values() {
            group.sort_by_key(|index| slots.get(*index).and_then(Option::as_ref).map(|slot| slot.order));
            let Some((first, rest)) = group.split_first() else {
                continue;
            };
            let mut merged = slots.get_mut(*first).and_then(Option::take).ok_or_else(|| {
                KernelMutationError::InvalidKernel("pipeline component lost its first member".into())
            })?;
            let mut member_ids = vec![merged.id];
            for index in rest {
                let pipeline = slots.get_mut(*index).and_then(Option::take).ok_or_else(|| {
                    KernelMutationError::InvalidKernel(
                        "pipeline component contains a duplicate member".into(),
                    )
                })?;
                merge_bindings(&mut merged.template.bindings, pipeline.template.bindings);
                merged.order = merged.order.min(pipeline.order);
                member_ids.push(pipeline.id);
            }
            let members = self
                .phases
                .iter()
                .enumerate()
                .filter_map(|(index, phase)| match phase.placement.group {
                    PhaseGroup::Pipeline(pipeline) if member_ids.contains(&pipeline) => {
                        Some(KernelId(index as u32))
                    }
                    _ => None,
                })
                .collect();
            let ordered = topologically_order_phases(&self.phases, members).map_err(|_| {
                KernelMutationError::InvalidKernel("kernel dependency graph contains a cycle".into())
            })?;
            for (order, phase) in ordered.into_iter().enumerate() {
                self.phase_mut(phase).placement = PhasePlacement {
                    group: PhaseGroup::Pipeline(merged.id),
                    order,
                };
            }
            self.pipelines.push(merged);
        }
        self.pipelines.sort_by_key(|pipeline| pipeline.order);
        Ok(())
    }
}

fn phase_from_entry(
    source_entry: Option<SemanticEntryId>,
    entry: &SemanticEntry,
    mut selection: KernelDispatch,
    label: &'static str,
    placement: PhasePlacement,
) -> Result<KernelPhase, String> {
    if !selection.explicit {
        if let Some(domain) = storage_image_domain_inputs(&entry.inputs, &selection.domain) {
            selection.domain = domain;
        }
    }
    let output_routes = output_projection(&entry.output_routes);
    let body = PlannedEntry::project(entry)?;
    let mut phase = phase_from_body(
        source_entry.map(CompilerFlowEndpoint::Entry),
        source_entry,
        placement,
        PhaseSpec::compute(body, selection, label),
    )?;
    phase.output_routes = output_routes;
    Ok(phase)
}

fn phase_from_body(
    flow_source: Option<CompilerFlowEndpoint>,
    source_entry: Option<SemanticEntryId>,
    placement: PhasePlacement,
    spec: PhaseSpec,
) -> Result<KernelPhase, String> {
    let prepared = spec.prepare()?;
    let phase = KernelPhase {
        placement,
        flow_source,
        label: prepared.label,
        entry: prepared.entry,
        source_entry,
        output_routes: Vec::new(),
        dispatch: prepared.dispatch,
        resources: prepared.resources,
        serial_single_workgroup: prepared.serial_single_workgroup,
        required_elements: prepared.required_elements,
        dependencies: Vec::new(),
    };
    Ok(phase)
}

fn phase_from_materialization(
    requirement_id: MaterializationId,
    requirement: &MaterializationRequirement,
    dependencies: Vec<KernelId>,
    placement: PhasePlacement,
) -> Result<KernelPhase, String> {
    let kind = requirement.kind();
    let domain =
        requirement.space().and_then(domain_from_space).unwrap_or(KernelDomain::Fixed { x: 1, y: 1, z: 1 });
    let spec = PhaseSpec::compute(
        PlannedEntry::project(requirement.entry())?,
        KernelDispatch::explicit(domain),
        match kind {
            MaterializationKind::SharedArray => "shared_array_materialization",
            MaterializationKind::Gather => "gather_prepass",
            MaterializationKind::Scalar => "scalar_prepass",
            MaterializationKind::RuntimeArray => "runtime_array_materialization",
        },
    );
    let mut phase = phase_from_body(
        Some(CompilerFlowEndpoint::Materialization(requirement_id)),
        None,
        placement,
        spec,
    )?;
    phase.serial_single_workgroup = true;
    phase.dependencies = dependencies;
    Ok(phase)
}

fn graphics_passthrough_phase(
    source_entry: SemanticEntryId,
    entry: &SemanticEntry,
    placement: PhasePlacement,
) -> Result<KernelPhase, String> {
    let domain = KernelDomain::Fixed { x: 1, y: 1, z: 1 };
    let spec = PhaseSpec::graphics(PlannedEntry::project(entry)?, KernelDispatch::inferred(domain));
    let mut phase = phase_from_body(
        Some(CompilerFlowEndpoint::Entry(source_entry)),
        Some(source_entry),
        placement,
        spec,
    )?;
    phase.output_routes = output_projection(&entry.output_routes);
    Ok(phase)
}

fn output_projection(routes: &[crate::egir::program::OutputRoute]) -> Vec<OutputRouteProjection> {
    let mut slots = routes.iter().map(|route| route.slot).collect::<Vec<_>>();
    slots.sort_unstable();
    slots.dedup();
    slots
        .into_iter()
        .enumerate()
        .map(|(physical, semantic_slot)| OutputRouteProjection {
            semantic_slot,
            physical_slot: OutputSlotId(physical),
        })
        .collect()
}

/// A compute entry with no SOAC-derived domain and a `#[storage_image]` param
/// runs one thread per texel of the image (the mountains / one_weekend
/// per-pixel pass shape) — the host resolves the size from the bound texture's
/// extent. Incidental storage-buffer inputs (e.g. mountains' keyboard buffer)
/// don't opt out; the image is the domain. Only upgrades the single-workgroup
/// placeholder domain; an explicit fixed grid stays as scheduled.
///
fn storage_image_domain_inputs(
    inputs: &[crate::egir::ir::EntryInput<SemanticResourceRef, crate::egir::types::WynLanguage>],
    baseline: &KernelDomain,
) -> Option<KernelDomain> {
    if !matches!(baseline, KernelDomain::Fixed { x: 1, y: 1, z: 1 }) {
        return None;
    }
    let (binding, ..) = inputs.iter().find_map(|input| input.storage_image_binding())?;
    Some(KernelDomain::Elements(DispatchLen::StorageImage {
        set: binding.set,
        binding: binding.binding,
    }))
}

fn execution_workgroup(model: &ExecutionModel) -> (u32, u32, u32) {
    match model {
        ExecutionModel::Compute { local_size } => *local_size,
        _ => (1, 1, 1),
    }
}

fn domain_from_dispatch(
    dispatch: &DispatchSize,
    host_resources: &HashMap<BindingRef, ResourceId>,
) -> Result<KernelDomain, String> {
    Ok(match dispatch {
        DispatchSize::Fixed { x, y, z, .. } => KernelDomain::Fixed { x: *x, y: *y, z: *z },
        DispatchSize::DerivedFrom {
            len:
                DispatchLen::InputBinding {
                    set,
                    binding,
                    elem_bytes,
                },
            ..
        } => KernelDomain::ResourceElements {
            resource: *host_resources.get(&BindingRef::new(*set, *binding)).ok_or_else(|| {
                format!("descriptor dispatch binding {set}:{binding} is absent from the resource manifest")
            })?,
            elem_bytes: *elem_bytes,
        },
        DispatchSize::DerivedFrom { len, .. } => KernelDomain::Elements(len.clone()),
    })
}

fn domain_selection_from_stage(
    stage: &ComputeStage,
    host_resources: &HashMap<BindingRef, ResourceId>,
) -> Result<KernelDispatch, String> {
    let domain = domain_from_dispatch(&stage.dispatch_size, host_resources)?;
    Ok(match stage.dispatch_size {
        // Honor the source's explicit intent: a user-pinned `#[dispatch]` grid
        // (including `1x1x1`) stays `Explicit` and is never re-inferred. Only
        // the unpinned default `1x1x1` placeholder is `Inferred`.
        DispatchSize::Fixed { explicit: true, .. } => KernelDispatch::explicit(domain),
        DispatchSize::Fixed { explicit: false, .. } => KernelDispatch::inferred(domain),
        DispatchSize::DerivedFrom { .. } => KernelDispatch::inferred(domain),
    })
}

pub(super) fn domain_from_space(space: &crate::egir::types::SegSpace) -> Option<KernelDomain> {
    if space.dims().iter().all(|extent| matches!(extent, SegExtent::Fixed(_))) {
        let count = space.dims().iter().try_fold(1u32, |product, extent| match extent {
            SegExtent::Fixed(n) => product.checked_mul(*n),
            _ => None,
        })?;
        return Some(KernelDomain::Elements(DispatchLen::Fixed { count }));
    }
    match space.dims() {
        [SegExtent::PushConstant { offset, .. }] => {
            Some(KernelDomain::Elements(DispatchLen::PushConstant {
                offset: *offset,
            }))
        }
        [SegExtent::ResourceLength {
            resource, elem_bytes, ..
        }] => Some(KernelDomain::ResourceElements {
            resource: resource.0,
            elem_bytes: *elem_bytes,
        }),
        _ => None,
    }
}

fn topologically_order_phases(
    arena: &[KernelPhase],
    phases: Vec<KernelId>,
) -> Result<Vec<KernelId>, Vec<KernelId>> {
    let phase_ids = phases.iter().copied().collect::<HashSet<_>>();
    let mut remaining = phases.into_iter().map(Some).collect::<Vec<_>>();
    let mut emitted = HashSet::new();
    let mut ordered = Vec::with_capacity(remaining.len());
    while ordered.len() < remaining.len() {
        let ready = remaining.iter().position(|candidate| {
            candidate.as_ref().is_some_and(|phase| {
                arena[phase.index()]
                    .dependencies
                    .iter()
                    .all(|dependency| !phase_ids.contains(dependency) || emitted.contains(dependency))
            })
        });
        let Some(index) = ready else {
            // Preserve stable order for the remainder. Validation reports the
            // actual cycle or unresolved dependency with kernel identities.
            ordered.extend(remaining.into_iter().flatten());
            return Err(ordered);
        };
        let Some(phase) = remaining[index].take() else {
            ordered.extend(remaining.into_iter().flatten());
            return Err(ordered);
        };
        emitted.insert(phase);
        ordered.push(phase);
    }
    Ok(ordered)
}

fn merge_bindings(target: &mut Vec<Binding>, source: Vec<Binding>) {
    for binding in source {
        if !target.iter().any(|existing| same_binding_slot(existing, &binding)) {
            target.push(binding);
        }
    }
}

fn same_binding_slot(left: &Binding, right: &Binding) -> bool {
    if std::mem::discriminant(left) == std::mem::discriminant(right)
        && left.slot().is_some_and(|slot| right.slot() == Some(slot))
    {
        return true;
    }
    match (left, right) {
        (
            Binding::PushConstant {
                offset: left_offset,
                size: left_size,
                ..
            },
            Binding::PushConstant {
                offset: right_offset,
                size: right_size,
                ..
            },
        ) => left_offset == right_offset && left_size == right_size,
        _ => false,
    }
}
