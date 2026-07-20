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
    CompilerFlowEndpoint, CompilerResourceFlow, EntryPublication, LogicalResourceArena, MaterializationId,
    MaterializationKind, MaterializationRequirement, OutputSlotId, PhysicalResourceTable, PlannedEntry,
    PlannedPublication, RegionInterner, ResourceOrigin, SemanticEntry, SemanticEntryId, SemanticFunc,
    SemanticProgram, SemanticResourceDecl, SemanticResourceRef,
};
use crate::egir::soac::{filter, screma};
use crate::egir::types::{EgirPhase, RegionId, Scheduled, SegExtent, SideEffectKind, Soac, SoacEffect};
use crate::flow::ExecutionModel;
use crate::pipeline_descriptor::{
    Binding, ComputePipeline, ComputeStage, DispatchLen, DispatchSize, Pipeline, PipelineDescriptor,
};
use crate::{BindingRef, ResourceAccess, ResourceId};

mod finalize;
#[cfg(any(test, debug_assertions))]
mod validation;

#[cfg(test)]
mod tests;

/// A complete module-level compute schedule.
#[derive(Clone, Debug, Default)]
pub(in crate::egir) struct KernelPlan {
    phases: crate::IdArena<KernelId, KernelPhase>,
    pipelines: Vec<ScheduledPipeline>,
    graphics_passthroughs: Vec<KernelId>,
    /// Executable entries used by descriptor-less probes and tests. They are
    /// fully planned and physicalized but never published as host stages.
    unpublished: Vec<KernelId>,
    source_entries: Vec<SourceEntryPlan>,
    /// Callable helpers synthesized by target planning. They are not admitted
    /// to semantic EGIR; physical construction publishes them only after the
    /// schedule has validated.
    generated_callables: Vec<SemanticFunc>,
    /// Source callable identities plus planner-generated additions. Planned
    /// Seg bodies use these stable ids before physical functions exist.
    region_interner: RegionInterner,
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
    /// Entry ABI records in deterministic descriptor-publication order. The
    /// kernel plan, rather than physical graphs, is the sole authority for
    /// backend-visible entry metadata.
    fn publications(&self, resources: &PhysicalResourceTable) -> Result<Vec<EntryPublication>, String> {
        let mut names = HashSet::new();
        let mut publications = Vec::new();
        for source in &self.source_entries {
            if names.insert(source.publication.name.as_str()) {
                publications.push(source.publication.publication(resources)?);
            }
        }
        for phase in self.phases() {
            let entry = &phase.entry;
            if names.insert(entry.name.as_str()) {
                publications.push(entry.publication(resources)?);
            }
        }
        Ok(publications)
    }

    pub(in crate::egir) fn physical_entries(&self) -> impl Iterator<Item = &PlannedEntry<Scheduled>> {
        self.phases().map(|phase| phase.entry.as_ref())
    }

    pub(in crate::egir) fn generated_callables(&self) -> impl Iterator<Item = &SemanticFunc> {
        self.generated_callables.iter()
    }

    pub(in crate::egir) fn region_interner(&self) -> &RegionInterner {
        &self.region_interner
    }

    fn install_phase_shells(&self, descriptor: &mut PipelineDescriptor) -> Result<(), String> {
        let mut rebuilt = descriptor
            .pipelines
            .iter()
            .enumerate()
            .filter_map(|(order, pipeline)| {
                matches!(pipeline, Pipeline::Graphics(_)).then(|| (order, pipeline.clone()))
            })
            .collect::<Vec<_>>();
        for scheduled in &self.pipelines {
            let mut compute = scheduled.template.clone();
            compute.stages = scheduled
                .phases
                .iter()
                .map(|id| self.phase(*id))
                .map(|phase| ComputeStage {
                    entry_point: phase.entry_point().to_owned(),
                    workgroup_size: phase.workgroup_size(),
                    dispatch_size: DispatchSize::Fixed {
                        x: 1,
                        y: 1,
                        z: 1,
                        explicit: false,
                    },
                    reads: Vec::new(),
                    writes: Vec::new(),
                })
                .collect();
            rebuilt.push((scheduled.order, Pipeline::Compute(compute)));
        }
        rebuilt.sort_by_key(|(order, _)| *order);
        descriptor.pipelines = rebuilt.into_iter().map(|(_, pipeline)| pipeline).collect();
        Ok(())
    }

    fn publish(
        &self,
        descriptor: &mut PipelineDescriptor,
        physical_resources: &PhysicalResourceTable,
    ) -> Result<(), String> {
        for scheduled in &self.pipelines {
            let Some(Pipeline::Compute(compute)) = descriptor.pipelines.iter_mut().find(|pipeline| {
                matches!(pipeline, Pipeline::Compute(candidate) if scheduled.phases.iter().any(|phase| {
                    candidate.stages.iter().any(|stage| stage.entry_point == self.phase(*phase).entry_point())
                }))
            }) else {
                return Err("scheduled compute pipeline was not installed".into());
            };
            let binding_index = compute
                .bindings
                .iter()
                .enumerate()
                .filter_map(|(index, binding)| binding_ref(binding).map(|binding| (binding, index)))
                .collect::<HashMap<_, _>>();
            let mut stages = Vec::with_capacity(scheduled.phases.len());
            for id in &scheduled.phases {
                let phase = self.phase(*id);
                let mut reads = Vec::new();
                let mut writes = Vec::new();
                for resource in phase.resources() {
                    let binding = physical_resources.binding(resource.resource);
                    let index = *binding_index.get(&binding).ok_or_else(|| {
                        format!(
                            "kernel `{}` references unpublished storage {binding}",
                            phase.entry_point()
                        )
                    })?;
                    if resource.access.reads() && !reads.contains(&index) {
                        reads.push(index);
                    }
                    if resource.access.writes() && !writes.contains(&index) {
                        writes.push(index);
                    }
                }
                let dispatch_size = match phase.domain.current() {
                    KernelDomain::Fixed { x, y, z } => DispatchSize::Fixed {
                        x: *x,
                        y: *y,
                        z: *z,
                        explicit: phase.domain.is_explicit(),
                    },
                    KernelDomain::Elements(len) => DispatchSize::DerivedFrom {
                        len: len.clone(),
                        workgroup_size: phase.workgroup_size().0,
                    },
                    KernelDomain::ResourceElements { resource, elem_bytes } => {
                        let binding = physical_resources.binding(*resource);
                        DispatchSize::DerivedFrom {
                            len: DispatchLen::InputBinding {
                                set: binding.set,
                                binding: binding.binding,
                                elem_bytes: *elem_bytes,
                            },
                            workgroup_size: phase.workgroup_size().0,
                        }
                    }
                };
                stages.push(ComputeStage {
                    entry_point: phase.entry_point().to_owned(),
                    workgroup_size: phase.workgroup_size(),
                    dispatch_size,
                    reads,
                    writes,
                });
            }
            compute.stages = stages;
        }
        Ok(())
    }
}

/// Stable identity of a physical kernel in a plan. Dependencies use this id
/// instead of vector positions so insertion cannot silently retarget an edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KernelId(pub u32);

impl From<u32> for KernelId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(super) enum KernelMutationError {
    #[error("no planned kernel with id {0:?}")]
    UnknownKernel(KernelId),
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

/// Body preparation requested while constructing or replacing a phase. The
/// variants encode the real graphics/compute/filter constraint; `label` is
/// retained only for diagnostics and summaries.
pub(super) enum KernelBodySpec {
    Graphics(PlannedEntry),
    Compute {
        body: PlannedEntry,
        label: &'static str,
    },
    Filter {
        body: PlannedEntry,
        plan: super::prepare::ParallelFilterPlan,
    },
}

impl KernelBodySpec {
    pub(super) fn graphics(body: PlannedEntry) -> Self {
        Self::Graphics(body)
    }

    pub(super) fn compute(body: PlannedEntry, label: &'static str) -> Self {
        Self::Compute { body, label }
    }

    pub(super) fn filter(
        body: PlannedEntry,
        stage: filter::ParallelStage,
        config: filter::ParallelConfig<SemanticResourceRef>,
        storage: filter::RuntimeStorage<SemanticResourceRef>,
    ) -> Self {
        Self::Filter {
            body,
            plan: super::prepare::ParallelFilterPlan::new(stage, config, storage),
        }
    }
}

pub(super) struct NewPhaseSpec {
    body: KernelBodySpec,
    domain: PhaseDomain,
}

impl NewPhaseSpec {
    pub(super) fn compute(body: PlannedEntry, domain: PhaseDomain, label: &'static str) -> Self {
        Self {
            body: KernelBodySpec::compute(body, label),
            domain,
        }
    }

    pub(super) fn graphics(body: PlannedEntry, domain: PhaseDomain) -> Self {
        Self {
            body: KernelBodySpec::graphics(body),
            domain,
        }
    }

    pub(super) fn filter(
        body: PlannedEntry,
        domain: PhaseDomain,
        stage: filter::ParallelStage,
        config: filter::ParallelConfig<SemanticResourceRef>,
        storage: filter::RuntimeStorage<SemanticResourceRef>,
    ) -> Self {
        Self {
            body: KernelBodySpec::filter(body, stage, config, storage),
            domain,
        }
    }
}

fn prepare_body(
    spec: KernelBodySpec,
) -> Result<(&'static str, Arc<PlannedEntry<Scheduled>>, Vec<ScheduledResource>), String> {
    let (body, label, filter_plan, expected_compute) = match spec {
        KernelBodySpec::Graphics(body) => (body, "graphics_passthrough", None, false),
        KernelBodySpec::Compute { body, label } => (body, label, None, true),
        KernelBodySpec::Filter { body, plan } => {
            let label = match plan.stage {
                filter::ParallelStage::Flags => "filter_flags",
                filter::ParallelStage::Scan => "filter_scan",
                filter::ParallelStage::Scatter => "filter_scatter",
            };
            (body, label, Some(plan), true)
        }
    };
    let entry = super::prepare::entry(body, filter_plan)?;
    let is_compute = matches!(entry.execution_model, ExecutionModel::Compute { .. });
    if is_compute != expected_compute {
        let expected = if expected_compute { "compute" } else { "graphics" };
        return Err(format!("entry `{}` cannot use a {expected} body", entry.name));
    }
    let resources = planned_resources(&entry);
    Ok((label, Arc::new(entry), resources))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputRouteProjection {
    pub semantic_slot: OutputSlotId,
    pub physical_slot: OutputSlotId,
}

/// Ordered phases that share one host binding table.
#[derive(Clone, Debug)]
struct ScheduledPipeline {
    /// Original descriptor position, retained solely for stable publication.
    order: usize,
    /// Non-stage host metadata and already-published source bindings.
    template: ComputePipeline,
    phases: Vec<KernelId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PhaseListId {
    Pipeline(usize),
    Graphics,
    Unpublished,
}

/// One executable kernel phase.
#[derive(Clone, Debug)]
struct KernelPhase {
    /// Typed identity used by compiler-resource flow edges. It is separate
    /// from the projected source ABI because generated requirements have no
    /// semantic entry ABI of their own.
    flow_source: Option<CompilerFlowEndpoint>,
    label: &'static str,
    entry: Arc<PlannedEntry<Scheduled>>,
    source_entry: Option<SemanticEntryId>,
    output_routes: Vec<OutputRouteProjection>,
    domain: PhaseDomain,
    resources: Vec<ScheduledResource>,
    /// Materialization kernels collapse to one workgroup under serial policy.
    /// This is an execution constraint, unlike the diagnostic `label`.
    serial_single_workgroup: bool,
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

    fn refresh_phase_facts(&mut self) {
        let entry = Arc::clone(&self.entry);
        self.domain.refresh(&entry);
        self.output_routes = output_projection(&entry.output_routes);
    }

    fn replace_body(&mut self, spec: KernelBodySpec) -> Result<(), String> {
        let (label, entry, resources) = prepare_body(spec)?;
        self.label = label;
        self.entry = entry;
        self.resources = resources;
        self.refresh_phase_facts();
        Ok(())
    }

    fn replace_scheduled_compute(
        &mut self,
        entry: PlannedEntry<Scheduled>,
        label: &'static str,
    ) -> Result<(), String> {
        if !matches!(entry.execution_model, ExecutionModel::Compute { .. }) {
            return Err(format!("entry `{}` cannot use a compute body", entry.name));
        }
        self.label = label;
        self.resources = planned_resources(&entry);
        self.entry = Arc::new(entry);
        self.refresh_phase_facts();
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
        let mut phases = Vec::new();
        for pipeline in &plan.pipelines {
            phases.extend(
                pipeline.phases.iter().map(|id| KernelPhaseSummary::from_phase(*id, plan.phase(*id))),
            );
        }
        phases.extend(
            plan.graphics_passthroughs
                .iter()
                .map(|id| KernelPhaseSummary::from_phase(*id, plan.phase(*id))),
        );
        phases
            .extend(plan.unpublished.iter().map(|id| KernelPhaseSummary::from_phase(*id, plan.phase(*id))));
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
            domain: phase.domain.current().clone(),
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

/// A launch domain together with the authority that may update it. Inferred
/// domains retain their descriptor baseline and their currently resolved
/// body-derived value in one state, so those facts cannot disagree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum PhaseDomain {
    Inferred {
        baseline: KernelDomain,
        current: KernelDomain,
    },
    Explicit(KernelDomain),
}

impl PhaseDomain {
    pub(super) fn inferred(baseline: KernelDomain) -> Self {
        Self::Inferred {
            current: baseline.clone(),
            baseline,
        }
    }

    pub(super) fn explicit(domain: KernelDomain) -> Self {
        Self::Explicit(domain)
    }

    fn current(&self) -> &KernelDomain {
        match self {
            Self::Inferred { current, .. } | Self::Explicit(current) => current,
        }
    }

    fn is_explicit(&self) -> bool {
        matches!(self, Self::Explicit(_))
    }

    fn refresh(&mut self, entry: &PlannedEntry<Scheduled>) {
        if let Self::Inferred { baseline, current } = self {
            *current = inferred_body_domain(entry, baseline.clone());
        }
    }
}

/// Conservative resource access for a phase.
pub type ScheduledResource = crate::egir::ir::SegResourceAccess<ResourceId>;

impl KernelPlan {
    pub(super) fn intern_callable(&mut self, name: impl AsRef<str>) -> RegionId {
        self.region_interner.intern(name.as_ref())
    }

    pub(super) fn callable_name(&self, id: RegionId) -> &str {
        self.region_interner.resolve(id)
    }

    pub(super) fn define_callable(&mut self, function: SemanticFunc) -> RegionId {
        assert!(
            self.region_interner.get(&function.name).is_none(),
            "planner-generated callable `{}` collides with a semantic callable",
            function.name
        );
        let id = self.region_interner.intern(&function.name);
        self.generated_callables.push(function);
        id
    }

    fn phases(&self) -> impl Iterator<Item = &KernelPhase> {
        self.phases_with_ids().map(|(_, phase)| phase)
    }

    fn phases_with_ids(&self) -> impl Iterator<Item = (KernelId, &KernelPhase)> {
        self.pipelines
            .iter()
            .flat_map(|pipeline| pipeline.phases.iter().map(|id| (*id, self.phase(*id))))
            .chain(self.graphics_passthroughs.iter().map(|id| (*id, self.phase(*id))))
            .chain(self.unpublished.iter().map(|id| (*id, self.phase(*id))))
    }

    fn contains_entry(&self, entry_point: &str) -> bool {
        self.phases().any(|phase| phase.entry_point() == entry_point)
    }

    pub(super) fn contains_flow_source(&self, source: CompilerFlowEndpoint) -> bool {
        self.phases().any(|phase| phase.flow_source == Some(source))
    }

    pub(super) fn kernel_for_flow_source(&self, source: CompilerFlowEndpoint) -> Option<KernelId> {
        self.phases_with_ids().find(|(_, phase)| phase.flow_source == Some(source)).map(|(id, _)| id)
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

    #[cfg(any(test, debug_assertions))]
    fn validate_for_finalization(
        &self,
        resources: &LogicalResourceArena,
        descriptor: &PipelineDescriptor,
    ) -> Result<(), String> {
        self.validate()?;
        self.validate_program(resources, descriptor)
    }

    pub(super) fn seed(
        descriptor: &PipelineDescriptor,
        semantic: &SemanticProgram,
    ) -> Result<Self, String> {
        let resources = &semantic.resources;
        let region_interner = &semantic.region_interner;
        let host_resources = crate::egir::program::host_resource_map(resources);
        let mut seeded = vec![None; semantic.entry_points.len()];
        let mut by_name = HashMap::new();
        for source in semantic.entry_ids() {
            let entry = &semantic[source];
            if by_name.insert(entry.name.as_str(), (source, entry)).is_some() {
                return Err(format!("duplicate semantic entry `{}`", entry.name));
            }
        }
        let mut phase_arena = crate::IdArena::new();
        let mut pipelines = Vec::new();
        for (order, pipeline) in descriptor.pipelines.iter().enumerate() {
            let Pipeline::Compute(template) = pipeline else {
                continue;
            };
            let mut phases = Vec::new();
            for stage in &template.stages {
                let id = phase_arena.alloc_id();
                let selection = domain_selection_from_stage(stage, &host_resources)?;
                let (source, entry) = by_name.get(stage.entry_point.as_str()).ok_or_else(|| {
                    format!("descriptor stage `{}` has no semantic entry", stage.entry_point)
                })?;
                let phase = phase_from_entry(Some(*source), entry, selection, source_label(entry))?;
                record_seeded_kernel(&mut seeded, *source, id, &entry.name)?;
                phase_arena.insert(id, phase);
                phases.push(id);
            }
            pipelines.push(ScheduledPipeline {
                order,
                template: template.clone(),
                phases,
            });
        }
        let mut graphics_passthroughs = Vec::new();
        for pipeline in &descriptor.pipelines {
            let Pipeline::Graphics(graphics) = pipeline else {
                continue;
            };
            for stage in &graphics.stages {
                let id = phase_arena.alloc_id();
                let (source, entry) = by_name.get(stage.entry_point.as_str()).ok_or_else(|| {
                    format!("graphics stage `{}` has no semantic entry", stage.entry_point)
                })?;
                let phase = graphics_passthrough_phase(*source, entry)?;
                record_seeded_kernel(&mut seeded, *source, id, &entry.name)?;
                phase_arena.insert(id, phase);
                graphics_passthroughs.push(id);
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
        let mut unpublished = Vec::new();
        for source in semantic.entry_ids() {
            let entry = &semantic[source];
            if published_names.contains(entry.name.as_str()) {
                continue;
            }
            let id = phase_arena.alloc_id();
            let result = if matches!(entry.execution_model, ExecutionModel::Compute { .. }) {
                phase_from_entry(
                    Some(source),
                    entry,
                    PhaseDomain::inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                    source_label(entry),
                )
            } else {
                graphics_passthrough_phase(source, entry)
            };
            let phase = result?;
            record_seeded_kernel(&mut seeded, source, id, &entry.name)?;
            phase_arena.insert(id, phase);
            unpublished.push(id);
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
            phases: phase_arena,
            pipelines,
            graphics_passthroughs,
            unpublished,
            source_entries,
            generated_callables: Vec::new(),
            region_interner: region_interner.clone(),
        })
    }

    pub(super) fn primary_kernel(&self, source: SemanticEntryId) -> KernelId {
        self.source_entries[source.index()].primary
    }

    pub(super) fn domain_of(&self, kernel: KernelId) -> Result<KernelDomain, KernelMutationError> {
        self.phases
            .get(kernel)
            .map(|phase| phase.domain.current().clone())
            .ok_or(KernelMutationError::UnknownKernel(kernel))
    }

    fn location(&self, kernel: KernelId) -> Result<(PhaseListId, usize), KernelMutationError> {
        if self.phases.get(kernel).is_none() {
            return Err(KernelMutationError::UnknownKernel(kernel));
        }
        self.pipelines
            .iter()
            .enumerate()
            .find_map(|(pipeline, planned)| {
                planned
                    .phases
                    .iter()
                    .position(|candidate| *candidate == kernel)
                    .map(|index| (PhaseListId::Pipeline(pipeline), index))
            })
            .or_else(|| {
                self.graphics_passthroughs
                    .iter()
                    .position(|candidate| *candidate == kernel)
                    .map(|index| (PhaseListId::Graphics, index))
            })
            .or_else(|| {
                self.unpublished
                    .iter()
                    .position(|candidate| *candidate == kernel)
                    .map(|index| (PhaseListId::Unpublished, index))
            })
            .ok_or_else(|| {
                KernelMutationError::InvalidKernel(format!("kernel {kernel:?} is absent from the schedule"))
            })
    }

    fn list(&self, id: PhaseListId) -> &Vec<KernelId> {
        match id {
            PhaseListId::Pipeline(index) => &self.pipelines[index].phases,
            PhaseListId::Graphics => &self.graphics_passthroughs,
            PhaseListId::Unpublished => &self.unpublished,
        }
    }

    fn list_mut(&mut self, id: PhaseListId) -> &mut Vec<KernelId> {
        match id {
            PhaseListId::Pipeline(index) => &mut self.pipelines[index].phases,
            PhaseListId::Graphics => &mut self.graphics_passthroughs,
            PhaseListId::Unpublished => &mut self.unpublished,
        }
    }

    fn allocate_kernel_id(&mut self) -> KernelId {
        self.phases.alloc_id()
    }

    fn insert_phase(&mut self, id: KernelId, phase: KernelPhase) {
        self.phases.insert(id, phase);
    }

    pub(super) fn add_materialization_before(
        &mut self,
        consumer: KernelId,
        requirement_id: MaterializationId,
        requirement: &MaterializationRequirement,
    ) -> Result<KernelId, KernelMutationError> {
        let (list_id, index) = self.location(consumer)?;
        if list_id == PhaseListId::Graphics {
            let id = self.allocate_kernel_id();
            let phase = phase_from_materialization(requirement_id, requirement, Vec::new())?;
            self.insert_phase(id, phase);
            self.unpublished.push(id);
            return Ok(id);
        }
        let dependencies = self.phase(consumer).dependencies.clone();
        let id = self.allocate_kernel_id();
        let phase = phase_from_materialization(requirement_id, requirement, dependencies)?;
        self.insert_phase(id, phase);
        let list = self.list_mut(list_id);
        list.insert(index, id);
        self.phase_mut(consumer).dependencies = vec![id];
        Ok(id)
    }

    pub(super) fn add_sibling(
        &mut self,
        parent: KernelId,
        spec: NewPhaseSpec,
    ) -> Result<KernelId, KernelMutationError> {
        let (list_id, _) = self.location(parent)?;
        if list_id == PhaseListId::Graphics {
            return Err(KernelMutationError::UnknownKernel(parent));
        }
        let (source_entry, flow_source) = {
            let parent = self.phase(parent);
            (parent.source_entry, parent.flow_source)
        };
        let id = self.allocate_kernel_id();
        let phase = phase_from_body(flow_source, source_entry, spec)?;
        self.insert_phase(id, phase);
        self.list_mut(list_id).push(id);
        Ok(id)
    }

    pub(super) fn make_serial(&mut self) -> Result<(), KernelMutationError> {
        let kernels = self
            .pipelines
            .iter()
            .flat_map(|pipeline| pipeline.phases.iter().copied())
            .chain(self.unpublished.iter().copied())
            .collect::<Vec<_>>();
        for kernel in kernels {
            let phase = self.phase_mut(kernel);
            let mut entry = phase.entry.as_ref().clone();
            super::prepare::force_serial(&mut entry.graph);
            let label = if phase.serial_single_workgroup { phase.label } else { "serial_compute" };
            phase.replace_scheduled_compute(entry, label)?;
            if phase.serial_single_workgroup {
                let domain = KernelDomain::Fixed { x: 1, y: 1, z: 1 };
                phase.domain = PhaseDomain::explicit(domain);
            }
        }
        Ok(())
    }

    pub(super) fn set_output_projection(
        &mut self,
        kernel: KernelId,
        outputs: Vec<OutputSlotId>,
    ) -> Result<(), KernelMutationError> {
        self.location(kernel)?;
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
        self.phase_mut(kernel).output_routes = outputs
            .into_iter()
            .enumerate()
            .map(|(physical, semantic_slot)| OutputRouteProjection {
                semantic_slot,
                physical_slot: OutputSlotId(physical),
            })
            .collect();
        Ok(())
    }

    pub(super) fn commit_kernel(
        &mut self,
        kernel: KernelId,
        spec: KernelBodySpec,
    ) -> Result<KernelId, KernelMutationError> {
        self.location(kernel)?;
        let phase = self.phase_mut(kernel);
        phase.replace_body(spec)?;
        Ok(kernel)
    }

    pub(super) fn install_reduce(
        &mut self,
        kernel: KernelId,
        partial: KernelBodySpec,
        combines: Vec<NewPhaseSpec>,
    ) -> Result<(), KernelMutationError> {
        self.install_chain(kernel, Vec::new(), partial, combines)
    }

    pub(super) fn install_scan(
        &mut self,
        kernel: KernelId,
        partial: KernelBodySpec,
        block_scan: NewPhaseSpec,
        apply_offsets: NewPhaseSpec,
    ) -> Result<(), KernelMutationError> {
        self.install_chain(kernel, Vec::new(), partial, vec![block_scan, apply_offsets])
    }

    pub(super) fn install_filter(
        &mut self,
        kernel: KernelId,
        flags: NewPhaseSpec,
        scan: NewPhaseSpec,
        combine: NewPhaseSpec,
        apply_offsets: NewPhaseSpec,
        scatter: KernelBodySpec,
    ) -> Result<(), KernelMutationError> {
        self.install_chain(
            kernel,
            vec![flags, scan, combine, apply_offsets],
            scatter,
            Vec::new(),
        )
    }

    /// Install a complete phase chain transactionally. `kernel` remains the
    /// anchor identity; surrounding phases receive ids only after all bodies
    /// have prepared successfully.
    fn install_chain(
        &mut self,
        kernel: KernelId,
        before: Vec<NewPhaseSpec>,
        anchor: KernelBodySpec,
        after: Vec<NewPhaseSpec>,
    ) -> Result<(), KernelMutationError> {
        let (list_id, index) = self.location(kernel)?;
        if list_id == PhaseListId::Graphics {
            return Err(KernelMutationError::UnknownKernel(kernel));
        }
        let original = self.phase(kernel).clone();
        let source_entry = original.source_entry;
        let flow_source = original.flow_source;
        let mut chain = Vec::with_capacity(before.len() + after.len() + 1);
        let mut additions = Vec::with_capacity(before.len() + after.len());
        let mut dependencies = original.dependencies.clone();
        for spec in before {
            let id = self.allocate_kernel_id();
            let mut phase = phase_from_body(flow_source, source_entry, spec)?;
            phase.dependencies = dependencies;
            dependencies = vec![id];
            chain.push(id);
            additions.push((id, phase));
        }
        let mut anchor_phase = original.clone();
        anchor_phase.replace_body(anchor)?;
        anchor_phase.dependencies = dependencies;
        dependencies = vec![kernel];
        chain.push(kernel);
        for spec in after {
            let id = self.allocate_kernel_id();
            let mut phase = phase_from_body(flow_source, source_entry, spec)?;
            phase.dependencies = dependencies;
            dependencies = vec![id];
            chain.push(id);
            additions.push((id, phase));
        }

        let tail = dependencies[0];
        let dependents = self.list(list_id).iter().skip(index + 1).copied().collect::<Vec<_>>();
        for dependent in dependents {
            for dependency in &mut self.phase_mut(dependent).dependencies {
                if *dependency == kernel {
                    *dependency = tail;
                }
            }
            self.phase_mut(dependent).dependencies.sort_unstable();
            self.phase_mut(dependent).dependencies.dedup();
        }
        *self.phase_mut(kernel) = anchor_phase;
        for (id, phase) in additions {
            self.insert_phase(id, phase);
        }
        self.list_mut(list_id).splice(index..=index, chain);
        Ok(())
    }

    pub(in crate::egir) fn check_explicit_dispatch_coverage(
        &self,
        entries: &[SemanticEntry],
    ) -> Result<(), String> {
        let by_name = entries.iter().map(|entry| (entry.name.as_str(), entry)).collect::<HashMap<_, _>>();
        for phase in
            self.pipelines.iter().flat_map(|pipeline| pipeline.phases.iter().map(|id| self.phase(*id)))
        {
            let PhaseDomain::Explicit(KernelDomain::Fixed { x, y, z }) = &phase.domain else {
                continue;
            };
            let Some(entry) = by_name.get(phase.entry_point()) else {
                continue;
            };
            let Some(KernelDomain::Elements(DispatchLen::Fixed { count })) = semantic_domain_entry(entry)
            else {
                continue;
            };
            let (wx, wy, wz) = execution_workgroup(&entry.execution_model);
            let total = x
                .saturating_mul(*y)
                .saturating_mul(*z)
                .saturating_mul(wx)
                .saturating_mul(wy)
                .saturating_mul(wz);
            if total < count {
                return Err(format!(
                    "#[dispatch({x}, {y}, {z})] launches {total} threads but entry `{}` requires {count}; {} elements would be dropped",
                    phase.entry_point(),
                    count - total,
                ));
            }
        }
        Ok(())
    }

    pub(super) fn coalesce_resource_flows(
        &mut self,
        flows: &[(ResourceId, CompilerResourceFlow)],
    ) -> Result<(), KernelMutationError> {
        let mut updated = self.clone();
        updated.complete_resource_flows(flows)?;
        updated.merge_connected_pipelines()?;
        *self = updated;
        Ok(())
    }

    fn complete_resource_flows(
        &mut self,
        flows: &[(ResourceId, CompilerResourceFlow)],
    ) -> Result<(), KernelMutationError> {
        for (resource, flow) in flows {
            let mut writers = self
                .flow_resource_phases(flow.producer, *resource, true)
                .map(|(id, _)| id)
                .collect::<Vec<_>>();
            if writers.is_empty() {
                if let Some(writer) = self
                    .phases_with_ids()
                    .filter(|(_, phase)| phase.flow_source == Some(flow.producer))
                    .map(|(id, _)| id)
                    .last()
                {
                    let phase = self.phase_mut(writer);
                    merge_scheduled_resource(&mut phase.resources, *resource, ResourceAccess::Write);
                    writers.push(writer);
                }
            }
            for consumer in &flow.consumers {
                let mut readers = self
                    .flow_resource_phases(*consumer, *resource, false)
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>();
                if readers.is_empty() {
                    let reader = self
                        .phases_with_ids()
                        .find(|(_, phase)| phase.flow_source == Some(*consumer))
                        .map(|(id, _)| id);
                    if let Some(reader) = reader {
                        let phase = self.phase_mut(reader);
                        merge_scheduled_resource(&mut phase.resources, *resource, ResourceAccess::Read);
                        readers.push(reader);
                    }
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
        &self.phases[id]
    }

    fn phase_mut(&mut self, id: KernelId) -> &mut KernelPhase {
        &mut self.phases[id]
    }

    fn add_dependency(&mut self, reader: KernelId, writer: KernelId) -> Result<(), KernelMutationError> {
        let reader_phase = self.phases.get(reader).ok_or(KernelMutationError::UnknownKernel(reader))?;
        if reader_phase.dependencies.contains(&writer) {
            return Ok(());
        }
        if self.phases.get(writer).is_none() {
            return Err(KernelMutationError::UnknownKernel(writer));
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
        let mut owner = HashMap::new();
        for (pipeline, planned) in self.pipelines.iter().enumerate() {
            for phase in &planned.phases {
                owner.insert(*phase, pipeline);
            }
        }
        let mut pipeline_components = super::DisjointSets::new(self.pipelines.len());
        for (pipeline, planned) in self.pipelines.iter().enumerate() {
            for dependency in planned.phases.iter().flat_map(|phase| &self.phase(*phase).dependencies) {
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
            for index in rest {
                let mut pipeline = slots.get_mut(*index).and_then(Option::take).ok_or_else(|| {
                    KernelMutationError::InvalidKernel(
                        "pipeline component contains a duplicate member".into(),
                    )
                })?;
                merge_bindings(&mut merged.template.bindings, pipeline.template.bindings);
                merged.phases.append(&mut pipeline.phases);
                merged.order = merged.order.min(pipeline.order);
            }
            merged.phases = topologically_order_phases(&self.phases, merged.phases).map_err(|_| {
                KernelMutationError::InvalidKernel("kernel dependency graph contains a cycle".into())
            })?;
            self.pipelines.push(merged);
        }
        self.pipelines.sort_by_key(|pipeline| pipeline.order);
        Ok(())
    }
}

fn phase_from_entry(
    source_entry: Option<SemanticEntryId>,
    entry: &SemanticEntry,
    selection: PhaseDomain,
    label: &'static str,
) -> Result<KernelPhase, String> {
    let body = PlannedEntry::project(entry)?;
    phase_from_body(
        source_entry.map(CompilerFlowEndpoint::Entry),
        source_entry,
        NewPhaseSpec::compute(body, selection, label),
    )
}

fn phase_from_body(
    flow_source: Option<CompilerFlowEndpoint>,
    source_entry: Option<SemanticEntryId>,
    spec: NewPhaseSpec,
) -> Result<KernelPhase, String> {
    let NewPhaseSpec { body, domain } = spec;
    let (label, entry, resources) = prepare_body(body)?;
    let mut phase = KernelPhase {
        flow_source,
        label,
        entry,
        source_entry,
        output_routes: Vec::new(),
        domain,
        resources,
        serial_single_workgroup: false,
        dependencies: Vec::new(),
    };
    phase.refresh_phase_facts();
    Ok(phase)
}

fn phase_from_materialization(
    requirement_id: MaterializationId,
    requirement: &MaterializationRequirement,
    dependencies: Vec<KernelId>,
) -> Result<KernelPhase, String> {
    let domain =
        semantic_domain_entry(&requirement.entry).unwrap_or(KernelDomain::Fixed { x: 1, y: 1, z: 1 });
    let spec = NewPhaseSpec::compute(
        PlannedEntry::project(&requirement.entry)?,
        PhaseDomain::explicit(domain),
        match requirement.kind {
            MaterializationKind::SharedArray => "shared_array_materialization",
            MaterializationKind::Gather => "gather_prepass",
            MaterializationKind::Scalar => "scalar_prepass",
            MaterializationKind::RuntimeArray => "runtime_array_materialization",
        },
    );
    let mut phase = phase_from_body(
        Some(CompilerFlowEndpoint::Materialization(requirement_id)),
        None,
        spec,
    )?;
    phase.serial_single_workgroup = true;
    phase.dependencies = dependencies;
    Ok(phase)
}

fn graphics_passthrough_phase(
    source_entry: SemanticEntryId,
    entry: &SemanticEntry,
) -> Result<KernelPhase, String> {
    let domain = KernelDomain::Fixed { x: 1, y: 1, z: 1 };
    let spec = NewPhaseSpec::graphics(PlannedEntry::project(entry)?, PhaseDomain::inferred(domain));
    phase_from_body(
        Some(CompilerFlowEndpoint::Entry(source_entry)),
        Some(source_entry),
        spec,
    )
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

fn source_label(entry: &SemanticEntry) -> &'static str {
    match super::parallel_recipe_effect(entry).map(|effect| &effect.kind) {
        Some(SideEffectKind::Soac(SoacEffect(_, Soac::Screma(screma::Op::Reduce { .. })))) => {
            "reduce_phase1"
        }
        Some(SideEffectKind::Soac(SoacEffect(_, Soac::Screma(screma::Op::Scan { .. })))) => "scan_phase1",
        _ => "serial_compute",
    }
}

fn inferred_body_domain(body: &PlannedEntry<Scheduled>, baseline: KernelDomain) -> KernelDomain {
    scheduled_domain_graph(&body.graph)
        .or_else(|| storage_image_domain_inputs(&body.inputs, &baseline))
        .unwrap_or(baseline)
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
) -> Result<PhaseDomain, String> {
    let domain = domain_from_dispatch(&stage.dispatch_size, host_resources)?;
    Ok(match stage.dispatch_size {
        // Honor the source's explicit intent: a user-pinned `#[dispatch]` grid
        // (including `1x1x1`) stays `Explicit` and is never re-inferred. Only
        // the unpinned default `1x1x1` placeholder is `Inferred`.
        DispatchSize::Fixed { explicit: true, .. } => PhaseDomain::explicit(domain),
        DispatchSize::Fixed { explicit: false, .. } => PhaseDomain::inferred(domain),
        DispatchSize::DerivedFrom { .. } => PhaseDomain::inferred(domain),
    })
}

fn semantic_domain_entry(entry: &SemanticEntry) -> Option<KernelDomain> {
    match &super::parallel_recipe_effect(entry)?.kind {
        SideEffectKind::Soac(SoacEffect(
            _,
            Soac::Screma(
                screma::Op::Map {
                    state: screma::SemanticState::Segmented { space, .. },
                    ..
                }
                | screma::Op::Reduce {
                    state: screma::SemanticState::Segmented { space, .. },
                    ..
                }
                | screma::Op::Scan {
                    state: screma::SemanticState::Segmented { space, .. },
                    ..
                }
                | screma::Op::Composite {
                    state: screma::SemanticState::Segmented { space, .. },
                    ..
                },
            ),
        )) => domain_from_space_in_graph(&entry.graph, space),
        SideEffectKind::Soac(SoacEffect(
            _,
            Soac::Filter(filter::Op {
                state: filter::SemanticState { space, .. },
                ..
            }),
        )) => domain_from_space_in_graph(&entry.graph, space),
        _ => None,
    }
}

fn scheduled_domain_graph(graph: &crate::egir::types::EGraph<Scheduled>) -> Option<KernelDomain> {
    match &super::prepare::parallel_effect(graph)?.2.kind {
        SideEffectKind::Soac(SoacEffect(
            _,
            Soac::Screma(
                screma::Op::Map {
                    state: screma::ScheduledState::Segmented(segment),
                    ..
                }
                | screma::Op::Reduce {
                    state: screma::ScheduledState::Segmented(segment),
                    ..
                }
                | screma::Op::Scan {
                    state: screma::ScheduledState::Segmented(segment),
                    ..
                }
                | screma::Op::Composite {
                    state: screma::ScheduledState::Segmented(segment),
                    ..
                },
            ),
        )) => domain_from_space_in_graph(graph, &segment.space),
        SideEffectKind::Soac(SoacEffect(
            _,
            Soac::Filter(filter::Op {
                state: filter::ScheduledState::Pipeline { space, plan, .. },
                ..
            }),
        )) if matches!(
            plan.stage,
            filter::ParallelStage::Flags | filter::ParallelStage::Scatter
        ) =>
        {
            domain_from_space_in_graph(graph, space)
        }
        _ => None,
    }
}

pub(super) fn domain_from_space(space: &crate::egir::types::SegSpace) -> Option<KernelDomain> {
    if space.dims.iter().all(|extent| matches!(extent, SegExtent::Fixed(_))) {
        let count = space.dims.iter().try_fold(1u32, |product, extent| match extent {
            SegExtent::Fixed(n) => product.checked_mul(*n),
            _ => None,
        })?;
        return Some(KernelDomain::Elements(DispatchLen::Fixed { count }));
    }
    match space.dims.as_slice() {
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

fn domain_from_space_in_graph<P>(
    graph: &crate::egir::types::EGraph<P>,
    space: &crate::egir::types::SegSpace,
) -> Option<KernelDomain>
where
    P: EgirPhase<Resource = SemanticResourceRef>,
{
    domain_from_space(space).or_else(|| {
        let [SegExtent::Value(value)] = space.dims.as_slice() else {
            return None;
        };
        let resource = crate::egir::graph_ops::extract_storage_view_source(graph, *value)?;
        let polytype::Type::Constructed(crate::ast::TypeName::Array, args) = graph.types.get(value)? else {
            return None;
        };
        let elem_ty = args.first()?;
        let elem_bytes = crate::ssa::layout::storage_elem_stride(elem_ty)?;
        Some(KernelDomain::ResourceElements {
            resource: resource.0,
            elem_bytes,
        })
    })
}

fn planned_resources(entry: &PlannedEntry<Scheduled>) -> Vec<ScheduledResource> {
    segmented_graph_resources(&entry.graph, &entry.resource_declarations)
        .unwrap_or_else(|| graph_resources(&entry.graph, &entry.resource_declarations))
}

fn segmented_graph_resources(
    graph: &crate::egir::types::EGraph<Scheduled>,
    declarations: &[SemanticResourceDecl],
) -> Option<Vec<ScheduledResource>> {
    let side_effect = super::prepare::parallel_effect(graph)?.2;
    if let SideEffectKind::Soac(SoacEffect(
        _,
        Soac::Filter(filter::Op {
            state: filter::ScheduledState::Pipeline { storage, plan, .. },
            ..
        }),
    )) = &side_effect.kind
    {
        let mut resources = graph_resources(graph, declarations);
        let mut push = |reference: SemanticResourceRef, access: ResourceAccess| {
            merge_scheduled_resource(&mut resources, reference.0, access);
        };
        let work = plan.buffers;
        match plan.stage {
            filter::ParallelStage::Flags => push(work.flags, ResourceAccess::Write),
            filter::ParallelStage::Scan => {
                push(work.flags, ResourceAccess::Read);
                push(work.offsets, ResourceAccess::Write);
                push(work.block_sums, ResourceAccess::Write);
            }
            filter::ParallelStage::Scatter => {
                push(work.flags, ResourceAccess::Read);
                push(work.offsets, ResourceAccess::Read);
                push(work.block_offsets, ResourceAccess::Read);
                if let filter::RuntimeLength::Stored(binding) = storage.length {
                    push(binding, ResourceAccess::Read);
                }
                push(storage.scratch, ResourceAccess::Write);
            }
        }
        resources.sort_by_key(|resource| resource.resource);
        return Some(resources);
    }
    let SideEffectKind::Soac(SoacEffect(
        _,
        Soac::Screma(
            screma::Op::Map {
                state: screma::ScheduledState::Segmented(segment),
                ..
            }
            | screma::Op::Reduce {
                state: screma::ScheduledState::Segmented(segment),
                ..
            }
            | screma::Op::Scan {
                state: screma::ScheduledState::Segmented(segment),
                ..
            }
            | screma::Op::Composite {
                state: screma::ScheduledState::Segmented(segment),
                ..
            },
        ),
    )) = &side_effect.kind
    else {
        return None;
    };
    Some(
        segment
            .resources
            .iter()
            .map(|resource| ScheduledResource {
                resource: resource.resource.0,
                access: resource.access,
            })
            .collect(),
    )
}

fn graph_resources<P: EgirPhase<Resource = SemanticResourceRef>>(
    graph: &crate::egir::types::EGraph<P>,
    declarations: &[crate::egir::program::SemanticResourceDecl],
) -> Vec<ScheduledResource> {
    let mut accesses: HashMap<ResourceId, ResourceAccess> = HashMap::new();
    let mut insert = |reference: SemanticResourceRef, access: ResourceAccess| {
        accesses.entry(reference.0).and_modify(|old| *old = old.merge(access)).or_insert(access);
    };

    for declaration in declarations {
        let access = match declaration.role {
            crate::interface::StorageRole::Input => ResourceAccess::Read,
            crate::interface::StorageRole::Output => ResourceAccess::Write,
            crate::interface::StorageRole::Intermediate => ResourceAccess::ReadWrite,
        };
        insert(declaration.resource, access);
    }

    // Planned graphs have already been projected to this kernel. Every
    // surviving storage view is therefore a conservative read; declarations
    // above upgrade outputs and intermediates to their stronger access.
    for node in graph.nodes.keys() {
        if let Some(resource) = crate::egir::graph_ops::extract_storage_view_source(graph, node) {
            insert(resource, ResourceAccess::Read);
        }
    }

    let mut resources: Vec<_> =
        accesses.into_iter().map(|(resource, access)| ScheduledResource { resource, access }).collect();
    resources.sort_by_key(|resource| resource.resource);
    resources
}

fn merge_scheduled_resource(
    resources: &mut Vec<ScheduledResource>,
    resource: ResourceId,
    access: ResourceAccess,
) {
    if let Some(existing) = resources.iter_mut().find(|candidate| candidate.resource == resource) {
        existing.access = existing.access.merge(access);
    } else {
        resources.push(ScheduledResource { resource, access });
        resources.sort_by_key(|item| item.resource);
    }
}

fn binding_ref(binding: &crate::pipeline_descriptor::Binding) -> Option<BindingRef> {
    use crate::pipeline_descriptor::Binding;
    match binding {
        Binding::StorageBuffer { set, binding, .. } | Binding::Uniform { set, binding, .. } => {
            Some(BindingRef::new(*set, *binding))
        }
        _ => None,
    }
}

fn topologically_order_phases(
    arena: &crate::IdArena<KernelId, KernelPhase>,
    phases: Vec<KernelId>,
) -> Result<Vec<KernelId>, Vec<KernelId>> {
    let phase_ids = phases.iter().copied().collect::<HashSet<_>>();
    let mut remaining = phases.into_iter().map(Some).collect::<Vec<_>>();
    let mut emitted = HashSet::new();
    let mut ordered = Vec::with_capacity(remaining.len());
    while ordered.len() < remaining.len() {
        let ready = remaining.iter().position(|candidate| {
            candidate.as_ref().is_some_and(|phase| {
                arena[*phase]
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
