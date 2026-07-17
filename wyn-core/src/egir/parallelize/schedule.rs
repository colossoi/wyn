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
    CompilerFlowEndpoint, CompilerResourceFlow, EntryPublication, InputSlotId, LogicalResource,
    MaterializationKind, MaterializationRequirement, OutputSlotId, PhysicalResourceTable, PlannedEntry,
    PlannedPublication, RegionInterner, ResourceOrigin, SemanticEntry, SemanticEntryId, SemanticFunc,
    SemanticResourceDecl, SemanticResourceRef,
};
use crate::egir::soac::{filter, screma};
use crate::egir::types::{EgirPhase, RegionId, Scheduled, SegExtent, Semantic, SideEffectKind, Soac};
use crate::flow::ExecutionModel;
use crate::interface::EntryInput;
use crate::pipeline_descriptor::{
    Binding, ComputePipeline, ComputeStage, DispatchLen, DispatchSize, Pipeline, PipelineDescriptor,
};
use crate::{BindingRef, ResourceAccess, ResourceId};

#[path = "schedule_validation.rs"]
mod validation;

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod schedule_tests;

/// A complete module-level compute schedule.
#[derive(Clone, Debug, Default)]
pub struct KernelPlan {
    pipelines: Vec<ScheduledPipeline>,
    graphics_passthroughs: Vec<KernelPhase>,
    /// Executable entries used by descriptor-less probes and tests. They are
    /// fully planned and physicalized but never published as host stages.
    unpublished: Vec<KernelPhase>,
    next_kernel_id: u32,
    semantic_abi: HashMap<SemanticEntryId, SemanticAbi>,
    source_publications: Vec<PlannedPublication>,
    unresolved_stages: Vec<String>,
    /// Callable helpers synthesized by target planning. They are not admitted
    /// to semantic EGIR; physical construction publishes them only after the
    /// plan has validated.
    generated_callables: Vec<SemanticFunc>,
    /// Source callable identities plus planner-generated additions. Planned
    /// Seg bodies use these stable ids before physical functions exist.
    region_interner: RegionInterner,
}

/// Typed handles for the physical kernels created while seeding a plan.
/// Entries whose initial projection failed have no handle; the corresponding
/// diagnostic remains on the plan and is reported during validation.
#[derive(Clone, Debug)]
pub struct SeededKernels {
    by_entry: Vec<Option<KernelId>>,
}

impl SeededKernels {
    pub fn entry(&self, source: SemanticEntryId) -> Option<KernelId> {
        self.by_entry.get(source.0 as usize).copied().flatten()
    }

    fn record(&mut self, source: SemanticEntryId, kernel: KernelId) {
        if let Some(slot) = self.by_entry.get_mut(source.0 as usize) {
            *slot = Some(kernel);
        }
    }
}

#[derive(Clone, Debug)]
struct SemanticAbi {
    name: String,
    input_count: usize,
    output_count: usize,
    routed_outputs: HashSet<OutputSlotId>,
}

/// Proof that all plan-local identities, dependency edges and ABI projections
/// have been checked. The inner plan is inaccessible to constructors outside
/// this module, so physicalization cannot accidentally accept an unchecked
/// plan.
#[derive(Clone, Debug)]
pub struct ValidatedKernelPlan(KernelPlan);

impl std::ops::Deref for ValidatedKernelPlan {
    type Target = KernelPlan;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ValidatedKernelPlan {
    pub fn published_plan(&self) -> PublishedKernelPlan {
        PublishedKernelPlan::from(&self.0)
    }

    /// Entry ABI records in deterministic descriptor-publication order. The
    /// validated plan, rather than physical graphs, is the sole authority for
    /// backend-visible entry metadata.
    pub fn publications(&self, resources: &PhysicalResourceTable) -> Result<Vec<EntryPublication>, String> {
        let mut names = HashSet::new();
        let mut publications = Vec::new();
        for source in &self.0.source_publications {
            if names.insert(source.name.as_str()) {
                publications.push(source.publication(resources)?);
            }
        }
        for phase in self.0.phases() {
            let entry = phase.recipe.entry();
            if names.insert(entry.name.as_str()) {
                publications.push(entry.publication(resources)?);
            }
        }
        Ok(publications)
    }

    pub(crate) fn physical_kernels(&self) -> impl Iterator<Item = &KernelPhase> {
        self.0.phases()
    }

    pub(crate) fn generated_callables(&self) -> impl Iterator<Item = &SemanticFunc> {
        self.0.generated_callables.iter()
    }

    pub(crate) fn region_interner(&self) -> &RegionInterner {
        &self.0.region_interner
    }

    pub fn install_phase_shells(&self, descriptor: &mut PipelineDescriptor) -> Result<(), String> {
        let mut rebuilt = descriptor
            .pipelines
            .iter()
            .enumerate()
            .filter_map(|(order, pipeline)| {
                matches!(pipeline, Pipeline::Graphics(_)).then(|| (order, pipeline.clone()))
            })
            .collect::<Vec<_>>();
        for scheduled in &self.0.pipelines {
            let mut compute = scheduled.template.clone();
            compute.stages = scheduled
                .phases
                .iter()
                .map(|phase| ComputeStage {
                    entry_point: phase.entry_point.clone(),
                    workgroup_size: phase.workgroup_size,
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

    pub fn publish(
        &self,
        descriptor: &mut PipelineDescriptor,
        physical_resources: &PhysicalResourceTable,
    ) -> Result<(), String> {
        for scheduled in &self.0.pipelines {
            let Some(Pipeline::Compute(compute)) = descriptor.pipelines.iter_mut().find(|pipeline| {
                matches!(pipeline, Pipeline::Compute(candidate) if scheduled.phases.iter().any(|phase| {
                    candidate.stages.iter().any(|stage| stage.entry_point == phase.entry_point)
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
            for phase in &scheduled.phases {
                let mut reads = Vec::new();
                let mut writes = Vec::new();
                for resource in &phase.resources {
                    let binding = physical_resources.binding(resource.resource).ok_or_else(|| {
                        format!(
                            "kernel `{}` references an unallocated resource",
                            phase.entry_point
                        )
                    })?;
                    let index = *binding_index.get(&binding).ok_or_else(|| {
                        format!(
                            "kernel `{}` references unpublished storage {binding}",
                            phase.entry_point
                        )
                    })?;
                    if resource.access.reads() && !reads.contains(&index) {
                        reads.push(index);
                    }
                    if resource.access.writes() && !writes.contains(&index) {
                        writes.push(index);
                    }
                }
                let dispatch_size = match &phase.domain {
                    KernelDomain::Fixed { x, y, z } => DispatchSize::Fixed {
                        x: *x,
                        y: *y,
                        z: *z,
                        explicit: matches!(phase.domain_selection, DomainSelection::Explicit(_)),
                    },
                    KernelDomain::Elements(len) => DispatchSize::DerivedFrom {
                        len: len.clone(),
                        workgroup_size: phase.workgroup_size.0,
                    },
                    KernelDomain::ResourceElements { resource, elem_bytes } => {
                        let binding = physical_resources.binding(*resource).ok_or_else(|| {
                            format!("kernel `{}` has unresolved dispatch resource", phase.entry_point)
                        })?;
                        DispatchSize::DerivedFrom {
                            len: DispatchLen::InputBinding {
                                set: binding.set,
                                binding: binding.binding,
                                elem_bytes: *elem_bytes,
                            },
                            workgroup_size: phase.workgroup_size.0,
                        }
                    }
                };
                stages.push(ComputeStage {
                    entry_point: phase.entry_point.clone(),
                    workgroup_size: phase.workgroup_size,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelMutationError {
    UnknownKernel(KernelId),
    InvalidKernel(String),
}

impl std::fmt::Display for KernelMutationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownKernel(kernel) => write!(formatter, "no planned kernel with id {kernel:?}"),
            Self::InvalidKernel(error) => formatter.write_str(error),
        }
    }
}

impl std::error::Error for KernelMutationError {}

/// Closed construction recipe retained by a planned kernel. The payload is a
/// graph-only projection; ABI, publication, dispatch and resources remain
/// first-class `KernelPhase` facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelKind {
    GraphicsPassthrough,
    SerialCompute,
    OutputDomainProjection,
    SharedArrayMaterialization,
    ScalarPrepass,
    GatherPrepass,
    FilterFlags,
    FilterScan,
    FilterCombine,
    FilterScatter,
    ReducePhase1,
    ReduceCombine,
    ScanPhase1,
    ScanBlock,
    ScanApplyOffsets,
}

/// Closed construction recipe: a classified reason paired with the only
/// retained planned-entry projection. There is no open or absent recipe state.
#[derive(Clone, Debug)]
pub struct KernelRecipe {
    kind: KernelKind,
    entry: Arc<PlannedEntry<Scheduled>>,
}

impl KernelRecipe {
    fn close(
        kind: KernelKind,
        entry: PlannedEntry<Semantic>,
        filter_plan: Option<filter::Plan<SemanticResourceRef>>,
    ) -> Self {
        let entry = super::prepare::entry(entry, filter_plan)
            .expect("kernel recipe must have valid scheduled SOAC states");
        Self {
            kind,
            entry: Arc::new(entry),
        }
    }

    pub fn kind(&self) -> KernelKind {
        self.kind
    }

    pub(crate) fn entry(&self) -> &PlannedEntry<Scheduled> {
        &self.entry
    }
}

/// Host ABI elements projected onto one physical entry. Slots are stable
/// semantic identities; physical binding allocation is tracked separately.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryAbiProjection {
    pub source_entry: Option<SemanticEntryId>,
    pub inputs: Vec<InputSlotId>,
    pub output_routes: Vec<OutputRouteProjection>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputRouteProjection {
    pub semantic_slot: OutputSlotId,
    pub physical_slot: OutputSlotId,
}

/// Ordered phases that share one host binding table.
#[derive(Clone, Debug)]
pub struct ScheduledPipeline {
    /// Original descriptor position, retained solely for stable publication.
    order: usize,
    /// Non-stage host metadata and already-published source bindings.
    template: ComputePipeline,
    phases: Vec<KernelPhase>,
}

#[derive(Clone, Copy)]
enum PhaseListId {
    Pipeline(usize),
    Unpublished,
}

/// One executable kernel phase.
#[derive(Clone, Debug)]
pub struct KernelPhase {
    pub id: KernelId,
    /// Typed identity used by compiler-resource flow edges. It is separate
    /// from the projected source ABI because generated requirements have no
    /// semantic entry ABI of their own.
    pub flow_source: Option<CompilerFlowEndpoint>,
    pub entry_point: String,
    pub recipe: KernelRecipe,
    pub abi: EntryAbiProjection,
    pub workgroup_size: (u32, u32, u32),
    pub domain: KernelDomain,
    /// The authoritative selection request. Inferred selections retain their
    /// descriptor baseline while the recipe establishes its final domain.
    pub domain_selection: DomainSelection,
    pub resources: Vec<ScheduledResource>,
    /// Stable kernel identities that must complete first.
    pub dependencies: Vec<KernelId>,
}

impl KernelPhase {
    fn refresh_phase_facts(&mut self) {
        let entry = Arc::clone(&self.recipe.entry);
        self.entry_point.clone_from(&entry.name);
        self.workgroup_size = execution_workgroup(&entry.execution_model);
        if let DomainSelection::Inferred(baseline) = &self.domain_selection {
            self.domain = inferred_body_domain(&entry, baseline.clone());
        }
        self.resources = planned_resources(&entry);
        self.abi.inputs = (0..entry.inputs.len()).map(InputSlotId).collect();
        self.abi.output_routes = output_projection(&entry.output_routes);
    }
}

/// Graph-free plan summary retained after physical EGIR has lowered to SSA.
#[derive(Clone, Debug)]
pub struct PublishedKernelPlan {
    phases: Vec<PublishedKernel>,
}

impl PublishedKernelPlan {
    pub fn phases(&self) -> impl Iterator<Item = &PublishedKernel> {
        self.phases.iter()
    }
}

#[derive(Clone, Debug)]
pub struct PublishedKernel {
    pub id: KernelId,
    pub placement: KernelPlacement,
    pub entry_point: String,
    pub kind: KernelKind,
    pub abi: EntryAbiProjection,
    pub workgroup_size: (u32, u32, u32),
    pub domain: KernelDomain,
    pub resources: Vec<ScheduledResource>,
    pub dependencies: Vec<KernelId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelPlacement {
    Compute {
        pipeline_order: usize,
        phase_index: usize,
    },
    Graphics,
    Unpublished,
}

impl From<&KernelPlan> for PublishedKernelPlan {
    fn from(plan: &KernelPlan) -> Self {
        let mut phases = Vec::new();
        for pipeline in &plan.pipelines {
            phases.extend(pipeline.phases.iter().enumerate().map(|(phase_index, phase)| {
                PublishedKernel::from_phase(
                    phase,
                    KernelPlacement::Compute {
                        pipeline_order: pipeline.order,
                        phase_index,
                    },
                )
            }));
        }
        phases.extend(
            plan.graphics_passthroughs
                .iter()
                .map(|phase| PublishedKernel::from_phase(phase, KernelPlacement::Graphics)),
        );
        phases.extend(
            plan.unpublished
                .iter()
                .map(|phase| PublishedKernel::from_phase(phase, KernelPlacement::Unpublished)),
        );
        Self { phases }
    }
}

impl PublishedKernel {
    fn from_phase(phase: &KernelPhase, placement: KernelPlacement) -> Self {
        Self {
            id: phase.id,
            placement,
            entry_point: phase.entry_point.clone(),
            kind: phase.recipe.kind(),
            abi: phase.abi.clone(),
            workgroup_size: phase.workgroup_size,
            domain: phase.domain.clone(),
            resources: phase.resources.clone(),
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

/// A domain together with the authority the caller assigns to it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DomainSelection {
    Inferred(KernelDomain),
    Explicit(KernelDomain),
}

/// Conservative resource access for a phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduledResource {
    pub resource: ResourceId,
    pub access: ResourceAccess,
}

impl KernelPlan {
    #[cfg(test)]
    pub(crate) fn generated_callables(&self) -> impl Iterator<Item = &SemanticFunc> {
        self.generated_callables.iter()
    }

    pub(crate) fn intern_callable(&mut self, name: impl AsRef<str>) -> RegionId {
        self.region_interner.intern(name.as_ref())
    }

    pub(crate) fn callable_name(&self, id: RegionId) -> &str {
        self.region_interner.resolve(id)
    }

    pub(crate) fn callable_names(&self, ids: impl IntoIterator<Item = RegionId>) -> Vec<String> {
        self.region_interner.resolve_cloned(ids)
    }

    pub(crate) fn define_callable(&mut self, function: SemanticFunc) -> RegionId {
        assert!(
            self.region_interner.get(&function.name).is_none(),
            "planner-generated callable `{}` collides with a semantic callable",
            function.name
        );
        let id = self.region_interner.intern(&function.name);
        self.generated_callables.push(function);
        id
    }

    pub fn phases(&self) -> impl Iterator<Item = &KernelPhase> {
        self.pipelines
            .iter()
            .flat_map(|pipeline| pipeline.phases.iter())
            .chain(self.graphics_passthroughs.iter())
            .chain(self.unpublished.iter())
    }

    pub fn contains_entry(&self, entry_point: &str) -> bool {
        self.phases().any(|phase| phase.entry_point == entry_point)
    }

    pub fn contains_flow_source(&self, source: CompilerFlowEndpoint) -> bool {
        self.phases().any(|phase| phase.flow_source == Some(source))
    }

    pub fn kernel_for_flow_source(&self, source: CompilerFlowEndpoint) -> Option<KernelId> {
        self.phases().find(|phase| phase.flow_source == Some(source)).map(|phase| phase.id)
    }

    pub(super) fn flow_resource_phases(
        &self,
        source: CompilerFlowEndpoint,
        resource: ResourceId,
        writes: bool,
    ) -> impl Iterator<Item = &KernelPhase> {
        self.phases().filter(move |phase| {
            phase.flow_source == Some(source)
                && phase.resources.iter().any(|item| {
                    item.resource == resource
                        && if writes { item.access.writes() } else { item.access.reads() }
                })
        })
    }

    pub fn into_validated(
        self,
        entries: &[SemanticEntry],
        resources: &[LogicalResource],
        descriptor: &PipelineDescriptor,
    ) -> Result<ValidatedKernelPlan, String> {
        self.validate()?;
        self.validate_program(entries, resources, descriptor)?;
        Ok(ValidatedKernelPlan(self))
    }

    pub fn seed(
        descriptor: &PipelineDescriptor,
        entries: &[SemanticEntry],
        resources: &[LogicalResource],
        region_interner: &RegionInterner,
    ) -> (Self, SeededKernels) {
        let host_resources = crate::egir::program::host_resource_map(resources);
        let semantic_abi = entries
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                (
                    SemanticEntryId(index as u32),
                    SemanticAbi {
                        name: entry.name.clone(),
                        input_count: entry.inputs.len(),
                        output_count: entry.outputs.len(),
                        routed_outputs: entry.output_routes.iter().map(|route| route.slot).collect(),
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        let mut unresolved_stages = Vec::new();
        let mut seeded = SeededKernels {
            by_entry: vec![None; entries.len()],
        };
        let source_publications = entries.iter().map(PlannedPublication::from_semantic).collect();
        let by_name = entries
            .iter()
            .enumerate()
            .map(|(index, entry)| (entry.name.as_str(), (SemanticEntryId(index as u32), entry)))
            .collect::<HashMap<_, _>>();
        let mut next_kernel_id = 0;
        let mut pipelines = Vec::new();
        for (order, pipeline) in descriptor.pipelines.iter().enumerate() {
            let Pipeline::Compute(template) = pipeline else {
                continue;
            };
            let mut phases = Vec::new();
            for stage in &template.stages {
                let id = KernelId(next_kernel_id);
                next_kernel_id += 1;
                let selection = domain_selection_from_stage(stage, &host_resources);
                let phase = if let Some((source, entry)) = by_name.get(stage.entry_point.as_str()) {
                    phase_from_entry(id, Some(*source), entry, selection, source_kind(entry))
                } else {
                    Err(format!(
                        "descriptor stage `{}` has no semantic entry",
                        stage.entry_point
                    ))
                };
                match phase {
                    Ok(phase) => {
                        if let Some(source) = phase.abi.source_entry {
                            seeded.record(source, phase.id);
                        }
                        phases.push(phase);
                    }
                    Err(error) => unresolved_stages.push(error),
                }
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
                let id = KernelId(next_kernel_id);
                next_kernel_id += 1;
                let phase = by_name
                    .get(stage.entry_point.as_str())
                    .ok_or_else(|| format!("graphics stage `{}` has no semantic entry", stage.entry_point))
                    .and_then(|(source, entry)| graphics_passthrough_phase(id, *source, entry));
                match phase {
                    Ok(phase) => {
                        if let Some(source) = phase.abi.source_entry {
                            seeded.record(source, phase.id);
                        }
                        graphics_passthroughs.push(phase);
                    }
                    Err(error) => unresolved_stages.push(error),
                }
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
        for (index, entry) in entries.iter().enumerate() {
            if published_names.contains(entry.name.as_str()) {
                continue;
            }
            let id = KernelId(next_kernel_id);
            next_kernel_id += 1;
            let result = if matches!(entry.execution_model, ExecutionModel::Compute { .. }) {
                phase_from_entry(
                    id,
                    Some(SemanticEntryId(index as u32)),
                    entry,
                    DomainSelection::Inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                    source_kind(entry),
                )
            } else {
                graphics_passthrough_phase(id, SemanticEntryId(index as u32), entry)
            };
            match result {
                Ok(phase) => {
                    seeded.record(SemanticEntryId(index as u32), phase.id);
                    unpublished.push(phase);
                }
                Err(error) => unresolved_stages.push(error),
            }
        }
        (
            Self {
                pipelines,
                graphics_passthroughs,
                unpublished,
                next_kernel_id,
                semantic_abi,
                source_publications,
                unresolved_stages,
                generated_callables: Vec::new(),
                region_interner: region_interner.clone(),
            },
            seeded,
        )
    }

    pub fn domain_of(&self, kernel: KernelId) -> Option<KernelDomain> {
        self.phases().find(|phase| phase.id == kernel).map(|phase| phase.domain.clone())
    }

    fn locate(&self, kernel: KernelId) -> Option<(PhaseListId, usize)> {
        self.pipelines
            .iter()
            .enumerate()
            .find_map(|(pipeline, planned)| {
                planned
                    .phases
                    .iter()
                    .position(|phase| phase.id == kernel)
                    .map(|index| (PhaseListId::Pipeline(pipeline), index))
            })
            .or_else(|| {
                self.unpublished
                    .iter()
                    .position(|phase| phase.id == kernel)
                    .map(|index| (PhaseListId::Unpublished, index))
            })
    }

    fn list(&self, id: PhaseListId) -> &Vec<KernelPhase> {
        match id {
            PhaseListId::Pipeline(index) => &self.pipelines[index].phases,
            PhaseListId::Unpublished => &self.unpublished,
        }
    }

    fn list_mut(&mut self, id: PhaseListId) -> &mut Vec<KernelPhase> {
        match id {
            PhaseListId::Pipeline(index) => &mut self.pipelines[index].phases,
            PhaseListId::Unpublished => &mut self.unpublished,
        }
    }

    fn allocate_kernel_id(&mut self) -> KernelId {
        let id = KernelId(self.next_kernel_id);
        self.next_kernel_id += 1;
        id
    }

    pub fn add_phase_after(
        &mut self,
        parent: KernelId,
        body: PlannedEntry,
        domain: DomainSelection,
        kind: KernelKind,
    ) -> Result<KernelId, KernelMutationError> {
        let (list_id, index) = self.locate(parent).ok_or(KernelMutationError::UnknownKernel(parent))?;
        let (parent_id, source_entry, flow_source) = {
            let parent = &self.list(list_id)[index];
            (parent.id, parent.abi.source_entry, parent.flow_source)
        };
        let id = self.allocate_kernel_id();
        let mut phase = phase_from_body(id, flow_source, source_entry, body, domain, kind)
            .map_err(KernelMutationError::InvalidKernel)?;
        phase.dependencies.push(parent_id);
        let list = self.list_mut(list_id);
        for dependent in list.iter_mut().skip(index + 1) {
            for dependency in &mut dependent.dependencies {
                if *dependency == parent_id {
                    *dependency = id;
                }
            }
            dependent.dependencies.sort_unstable();
            dependent.dependencies.dedup();
        }
        list.insert(index + 1, phase);
        Ok(id)
    }

    pub fn add_phase_before(
        &mut self,
        consumer: KernelId,
        body: PlannedEntry,
        domain: DomainSelection,
        kind: KernelKind,
    ) -> Result<KernelId, KernelMutationError> {
        let (list_id, index) = self.locate(consumer).ok_or(KernelMutationError::UnknownKernel(consumer))?;
        let (dependencies, source_entry, flow_source) = {
            let consumer = &self.list(list_id)[index];
            (
                consumer.dependencies.clone(),
                consumer.abi.source_entry,
                consumer.flow_source,
            )
        };
        let id = self.allocate_kernel_id();
        let mut phase = phase_from_body(id, flow_source, source_entry, body, domain, kind)
            .map_err(KernelMutationError::InvalidKernel)?;
        phase.dependencies = dependencies;
        let list = self.list_mut(list_id);
        list.insert(index, phase);
        list[index + 1].dependencies = vec![id];
        Ok(id)
    }

    pub(crate) fn add_filter_phase_before(
        &mut self,
        consumer: KernelId,
        body: PlannedEntry,
        domain: DomainSelection,
        kind: KernelKind,
        plan: filter::Plan<SemanticResourceRef>,
    ) -> Result<KernelId, KernelMutationError> {
        let (list_id, index) = self.locate(consumer).ok_or(KernelMutationError::UnknownKernel(consumer))?;
        let (dependencies, source_entry, flow_source) = {
            let consumer = &self.list(list_id)[index];
            (
                consumer.dependencies.clone(),
                consumer.abi.source_entry,
                consumer.flow_source,
            )
        };
        let id = self.allocate_kernel_id();
        let mut phase = phase_from_filter_body(id, flow_source, source_entry, body, domain, kind, plan)
            .map_err(KernelMutationError::InvalidKernel)?;
        phase.dependencies = dependencies;
        let list = self.list_mut(list_id);
        list.insert(index, phase);
        list[index + 1].dependencies = vec![id];
        Ok(id)
    }

    pub fn add_materialization_before(
        &mut self,
        consumer: KernelId,
        requirement: &MaterializationRequirement,
    ) -> Result<KernelId, KernelMutationError> {
        let Some((list_id, index)) = self.locate(consumer) else {
            if !self.graphics_passthroughs.iter().any(|phase| phase.id == consumer) {
                return Err(KernelMutationError::UnknownKernel(consumer));
            }
            let id = self.allocate_kernel_id();
            let phase = phase_from_materialization(id, requirement, Vec::new())
                .map_err(KernelMutationError::InvalidKernel)?;
            self.unpublished.push(phase);
            return Ok(id);
        };
        let dependencies = self.list(list_id)[index].dependencies.clone();
        let id = self.allocate_kernel_id();
        let phase = phase_from_materialization(id, requirement, dependencies)
            .map_err(KernelMutationError::InvalidKernel)?;
        let list = self.list_mut(list_id);
        list.insert(index, phase);
        list[index + 1].dependencies = vec![id];
        Ok(id)
    }

    pub fn add_sibling(
        &mut self,
        parent: KernelId,
        body: PlannedEntry,
        domain: DomainSelection,
        kind: KernelKind,
    ) -> Result<KernelId, KernelMutationError> {
        let (list_id, index) = self.locate(parent).ok_or(KernelMutationError::UnknownKernel(parent))?;
        let (source_entry, flow_source) = {
            let parent = &self.list(list_id)[index];
            (parent.abi.source_entry, parent.flow_source)
        };
        let id = self.allocate_kernel_id();
        let phase = phase_from_body(id, flow_source, source_entry, body, domain, kind)
            .map_err(KernelMutationError::InvalidKernel)?;
        self.list_mut(list_id).push(phase);
        Ok(id)
    }

    pub(crate) fn select_sequential_recipes(&mut self) {
        for phase in self
            .pipelines
            .iter_mut()
            .flat_map(|pipeline| pipeline.phases.iter_mut())
            .chain(self.unpublished.iter_mut())
        {
            let mut entry = phase.recipe.entry().clone();
            super::prepare::force_serial(&mut entry.graph);
            let kind = match phase.recipe.kind() {
                KernelKind::ReducePhase1 | KernelKind::ScanPhase1 => KernelKind::SerialCompute,
                kind => kind,
            };
            phase.recipe = KernelRecipe {
                kind,
                entry: Arc::new(entry),
            };
            phase.refresh_phase_facts();
            if matches!(
                kind,
                KernelKind::SharedArrayMaterialization
                    | KernelKind::GatherPrepass
                    | KernelKind::ScalarPrepass
                    | KernelKind::FilterScatter
            ) {
                let domain = KernelDomain::Fixed { x: 1, y: 1, z: 1 };
                phase.domain = domain.clone();
                phase.domain_selection = DomainSelection::Explicit(domain);
            }
        }
    }

    pub fn set_output_projection(
        &mut self,
        kernel: KernelId,
        outputs: Vec<OutputSlotId>,
    ) -> Result<(), KernelMutationError> {
        let (list, index) = self.locate(kernel).ok_or(KernelMutationError::UnknownKernel(kernel))?;
        self.list_mut(list)[index].abi.output_routes = outputs
            .into_iter()
            .enumerate()
            .map(|(physical, semantic_slot)| OutputRouteProjection {
                semantic_slot,
                physical_slot: OutputSlotId(physical),
            })
            .collect();
        Ok(())
    }

    pub fn commit_kernel(
        &mut self,
        kernel: KernelId,
        body: PlannedEntry,
        kind: KernelKind,
    ) -> Result<KernelId, KernelMutationError> {
        let (list, index) = self.locate(kernel).ok_or(KernelMutationError::UnknownKernel(kernel))?;
        let phase = &mut self.list_mut(list)[index];
        phase.recipe = KernelRecipe::close(kind, body, None);
        phase.refresh_phase_facts();
        Ok(kernel)
    }

    pub(crate) fn commit_filter_kernel(
        &mut self,
        kernel: KernelId,
        body: PlannedEntry,
        kind: KernelKind,
        plan: filter::Plan<SemanticResourceRef>,
    ) -> Result<KernelId, KernelMutationError> {
        let (list, index) = self.locate(kernel).ok_or(KernelMutationError::UnknownKernel(kernel))?;
        let phase = &mut self.list_mut(list)[index];
        phase.recipe = KernelRecipe::close(kind, body, Some(plan));
        phase.refresh_phase_facts();
        Ok(kernel)
    }

    pub fn check_explicit_dispatch_coverage(&self, entries: &[SemanticEntry]) -> Result<(), String> {
        let by_name = entries.iter().map(|entry| (entry.name.as_str(), entry)).collect::<HashMap<_, _>>();
        for phase in self.pipelines.iter().flat_map(|pipeline| &pipeline.phases) {
            let DomainSelection::Explicit(KernelDomain::Fixed { x, y, z }) = phase.domain_selection else {
                continue;
            };
            let Some(entry) = by_name.get(phase.entry_point.as_str()) else {
                continue;
            };
            let Some(KernelDomain::Elements(DispatchLen::Fixed { count })) = segmented_domain(entry) else {
                continue;
            };
            let (wx, wy, wz) = execution_workgroup(&entry.execution_model);
            let total = x
                .saturating_mul(y)
                .saturating_mul(z)
                .saturating_mul(wx)
                .saturating_mul(wy)
                .saturating_mul(wz);
            if total < count {
                return Err(format!(
                    "#[dispatch({x}, {y}, {z})] launches {total} threads but entry `{}` requires {count}; {} elements would be dropped",
                    phase.entry_point,
                    count - total,
                ));
            }
        }
        Ok(())
    }

    pub fn coalesce_resource_flows(&mut self, resources: &[LogicalResource]) {
        let mut flows = resources
            .iter()
            .filter_map(|resource| match &resource.origin {
                ResourceOrigin::Compiler(compiler) => compiler.flow.clone().map(|flow| (resource.id, flow)),
                ResourceOrigin::Host(_) => None,
            })
            .collect::<Vec<_>>();
        flows.sort_by_key(|(resource, _)| resource.0);
        self.complete_resource_flows(&flows);
        self.merge_connected_pipelines();
    }

    fn complete_resource_flows(&mut self, flows: &[(ResourceId, CompilerResourceFlow)]) {
        for (resource, flow) in flows {
            let mut writers = self
                .flow_resource_phases(flow.producer, *resource, true)
                .map(|phase| phase.id)
                .collect::<Vec<_>>();
            if writers.is_empty() {
                if let Some(writer) = self
                    .phases()
                    .filter(|phase| phase.flow_source == Some(flow.producer))
                    .map(|phase| phase.id)
                    .last()
                {
                    merge_scheduled_resource(
                        &mut self.phase_mut(writer).unwrap().resources,
                        *resource,
                        ResourceAccess::Write,
                    );
                    writers.push(writer);
                }
            }
            for consumer in &flow.consumers {
                let mut readers = self
                    .flow_resource_phases(*consumer, *resource, false)
                    .map(|phase| phase.id)
                    .collect::<Vec<_>>();
                if readers.is_empty() {
                    let reader = self
                        .phases()
                        .find(|phase| phase.flow_source == Some(*consumer))
                        .map(|phase| phase.id);
                    if let Some(reader) = reader {
                        merge_scheduled_resource(
                            &mut self.phase_mut(reader).unwrap().resources,
                            *resource,
                            ResourceAccess::Read,
                        );
                        readers.push(reader);
                    }
                }
                for reader in readers {
                    let phase = self.phase_mut(reader).unwrap();
                    phase.dependencies.extend(writers.iter().copied().filter(|writer| *writer != reader));
                    phase.dependencies.sort_unstable();
                    phase.dependencies.dedup();
                }
            }
        }
    }

    fn phase_mut(&mut self, id: KernelId) -> Option<&mut KernelPhase> {
        self.pipelines
            .iter_mut()
            .flat_map(|pipeline| pipeline.phases.iter_mut())
            .chain(self.graphics_passthroughs.iter_mut())
            .chain(self.unpublished.iter_mut())
            .find(|phase| phase.id == id)
    }

    fn merge_connected_pipelines(&mut self) {
        let mut owner = HashMap::new();
        for (pipeline, planned) in self.pipelines.iter().enumerate() {
            for phase in &planned.phases {
                owner.insert(phase.id, pipeline);
            }
        }
        let mut union = super::UnionFind::new(self.pipelines.len());
        for (pipeline, planned) in self.pipelines.iter().enumerate() {
            for dependency in planned.phases.iter().flat_map(|phase| &phase.dependencies) {
                if let Some(&other) = owner.get(dependency) {
                    union.union(pipeline, other);
                }
            }
        }
        let mut components = BTreeMap::<usize, Vec<usize>>::new();
        for pipeline in 0..self.pipelines.len() {
            let root = union.find(pipeline);
            components.entry(root).or_default().push(pipeline);
        }
        let mut slots = std::mem::take(&mut self.pipelines).into_iter().map(Some).collect::<Vec<_>>();
        for mut group in components.into_values() {
            group.sort_by_key(|index| slots[*index].as_ref().unwrap().order);
            let mut merged = slots[group[0]].take().unwrap();
            for index in group.into_iter().skip(1) {
                let mut pipeline = slots[index].take().unwrap();
                merge_bindings(&mut merged.template.bindings, pipeline.template.bindings);
                merged.phases.append(&mut pipeline.phases);
                merged.order = merged.order.min(pipeline.order);
            }
            merged.phases = topologically_order_phases(merged.phases)
                .unwrap_or_else(|stable_remainder| stable_remainder);
            self.pipelines.push(merged);
        }
        self.pipelines.sort_by_key(|pipeline| pipeline.order);
    }
}

fn phase_from_entry(
    id: KernelId,
    source_entry: Option<SemanticEntryId>,
    entry: &SemanticEntry,
    selection: DomainSelection,
    kind: KernelKind,
) -> Result<KernelPhase, String> {
    phase_from_body(
        id,
        source_entry.map(CompilerFlowEndpoint::Entry),
        source_entry,
        PlannedEntry::project(entry)?,
        selection,
        kind,
    )
}

fn phase_from_body(
    id: KernelId,
    flow_source: Option<CompilerFlowEndpoint>,
    source_entry: Option<SemanticEntryId>,
    body: PlannedEntry,
    selection: DomainSelection,
    kind: KernelKind,
) -> Result<KernelPhase, String> {
    phase_from_body_with_filter_plan(id, flow_source, source_entry, body, selection, kind, None)
}

fn phase_from_filter_body(
    id: KernelId,
    flow_source: Option<CompilerFlowEndpoint>,
    source_entry: Option<SemanticEntryId>,
    body: PlannedEntry,
    selection: DomainSelection,
    kind: KernelKind,
    plan: filter::Plan<SemanticResourceRef>,
) -> Result<KernelPhase, String> {
    phase_from_body_with_filter_plan(id, flow_source, source_entry, body, selection, kind, Some(plan))
}

fn phase_from_body_with_filter_plan(
    id: KernelId,
    flow_source: Option<CompilerFlowEndpoint>,
    source_entry: Option<SemanticEntryId>,
    body: PlannedEntry,
    selection: DomainSelection,
    kind: KernelKind,
    filter_plan: Option<filter::Plan<SemanticResourceRef>>,
) -> Result<KernelPhase, String> {
    let domain = match &selection {
        DomainSelection::Inferred(baseline) | DomainSelection::Explicit(baseline) => baseline.clone(),
    };
    let mut phase = KernelPhase {
        id,
        flow_source,
        entry_point: body.name.clone(),
        recipe: KernelRecipe::close(kind, body, filter_plan),
        abi: EntryAbiProjection {
            source_entry,
            inputs: Vec::new(),
            output_routes: Vec::new(),
        },
        workgroup_size: (1, 1, 1),
        domain,
        domain_selection: selection,
        resources: Vec::new(),
        dependencies: Vec::new(),
    };
    phase.refresh_phase_facts();
    Ok(phase)
}

fn phase_from_materialization(
    id: KernelId,
    requirement: &MaterializationRequirement,
    dependencies: Vec<KernelId>,
) -> Result<KernelPhase, String> {
    let domain = materialization_domain(requirement).unwrap_or(KernelDomain::Fixed { x: 1, y: 1, z: 1 });
    let mut phase = phase_from_body(
        id,
        Some(CompilerFlowEndpoint::Materialization(requirement.id)),
        None,
        PlannedEntry::project(&requirement.entry)?,
        DomainSelection::Explicit(domain),
        match requirement.kind {
            MaterializationKind::SharedArray => KernelKind::SharedArrayMaterialization,
            MaterializationKind::Gather => KernelKind::GatherPrepass,
            MaterializationKind::Scalar => KernelKind::ScalarPrepass,
            MaterializationKind::RuntimeArray => KernelKind::FilterScatter,
        },
    )?;
    phase.dependencies = dependencies;
    Ok(phase)
}

fn graphics_passthrough_phase(
    id: KernelId,
    source_entry: SemanticEntryId,
    entry: &SemanticEntry,
) -> Result<KernelPhase, String> {
    let domain = KernelDomain::Fixed { x: 1, y: 1, z: 1 };
    phase_from_body(
        id,
        Some(CompilerFlowEndpoint::Entry(source_entry)),
        Some(source_entry),
        PlannedEntry::project(entry)?,
        DomainSelection::Inferred(domain),
        KernelKind::GraphicsPassthrough,
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

fn source_kind(entry: &SemanticEntry) -> KernelKind {
    if !matches!(entry.execution_model, ExecutionModel::Compute { .. }) {
        return KernelKind::GraphicsPassthrough;
    }
    match super::parallel_effect(&entry.graph).map(|(_, _, effect)| &effect.kind) {
        Some(SideEffectKind::Soac(_, Soac::Screma(screma::Op::Reduce { .. }))) => KernelKind::ReducePhase1,
        Some(SideEffectKind::Soac(_, Soac::Screma(screma::Op::Scan { .. }))) => KernelKind::ScanPhase1,
        _ => KernelKind::SerialCompute,
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
fn storage_image_domain_inputs(inputs: &[EntryInput], baseline: &KernelDomain) -> Option<KernelDomain> {
    if !matches!(baseline, KernelDomain::Fixed { x: 1, y: 1, z: 1 }) {
        return None;
    }
    let (binding, ..) = inputs.iter().find_map(EntryInput::storage_image_binding)?;
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
) -> KernelDomain {
    match dispatch {
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
            resource: *host_resources
                .get(&BindingRef::new(*set, *binding))
                .expect("descriptor dispatch binding must be in the resource manifest"),
            elem_bytes: *elem_bytes,
        },
        DispatchSize::DerivedFrom { len, .. } => KernelDomain::Elements(len.clone()),
    }
}

fn domain_selection_from_stage(
    stage: &ComputeStage,
    host_resources: &HashMap<BindingRef, ResourceId>,
) -> DomainSelection {
    let domain = domain_from_dispatch(&stage.dispatch_size, host_resources);
    match stage.dispatch_size {
        // Honor the source's explicit intent: a user-pinned `#[dispatch]` grid
        // (including `1x1x1`) stays `Explicit` and is never re-inferred. Only
        // the unpinned default `1x1x1` placeholder is `Inferred`.
        DispatchSize::Fixed { explicit: true, .. } => DomainSelection::Explicit(domain),
        DispatchSize::Fixed { explicit: false, .. } => DomainSelection::Inferred(domain),
        DispatchSize::DerivedFrom { .. } => DomainSelection::Inferred(domain),
    }
}

pub(crate) fn segmented_domain(entry: &SemanticEntry) -> Option<KernelDomain> {
    semantic_domain_graph(&entry.graph)
}

fn materialization_domain(requirement: &MaterializationRequirement) -> Option<KernelDomain> {
    semantic_domain_graph(&requirement.entry.graph)
}

fn semantic_domain_graph(graph: &crate::egir::types::EGraph) -> Option<KernelDomain> {
    match &super::parallel_effect(graph)?.2.kind {
        SideEffectKind::Soac(
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
        ) => domain_from_space_in_graph(graph, space),
        SideEffectKind::Soac(
            _,
            Soac::Filter(filter::Op {
                state: filter::SemanticState { space, .. },
                ..
            }),
        ) => domain_from_space_in_graph(graph, space),
        _ => None,
    }
}

fn scheduled_domain_graph(graph: &crate::egir::types::EGraph<Scheduled>) -> Option<KernelDomain> {
    match &super::prepare::parallel_effect(graph)?.2.kind {
        SideEffectKind::Soac(
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
        ) => domain_from_space_in_graph(graph, &segment.space),
        SideEffectKind::Soac(
            _,
            Soac::Filter(filter::Op {
                state: filter::ScheduledState::Parallel { space, plan, .. },
                ..
            }),
        ) if matches!(
            plan.stage,
            filter::ParallelStage::Flags | filter::ParallelStage::Scatter
        ) =>
        {
            domain_from_space_in_graph(graph, space)
        }
        _ => None,
    }
}

pub(crate) fn domain_from_space(space: &crate::egir::types::SegSpace) -> Option<KernelDomain> {
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
    if let SideEffectKind::Soac(
        _,
        Soac::Filter(filter::Op {
            state: filter::ScheduledState::Parallel { storage, plan, .. },
            ..
        }),
    ) = &side_effect.kind
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
    let SideEffectKind::Soac(
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
    ) = &side_effect.kind
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

pub(super) fn topologically_order_phases(
    phases: Vec<KernelPhase>,
) -> Result<Vec<KernelPhase>, Vec<KernelPhase>> {
    let phase_ids = phases.iter().map(|phase| phase.id).collect::<HashSet<_>>();
    let mut remaining = phases.into_iter().map(Some).collect::<Vec<_>>();
    let mut emitted = HashSet::new();
    let mut ordered = Vec::with_capacity(remaining.len());
    while ordered.len() < remaining.len() {
        let ready = remaining.iter().position(|candidate| {
            candidate.as_ref().is_some_and(|phase| {
                phase
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
        let phase = remaining[index].take().expect("ready phase exists");
        emitted.insert(phase.id);
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
    match (left, right) {
        (
            Binding::StorageBuffer {
                set: left_set,
                binding: left_binding,
                ..
            },
            Binding::StorageBuffer {
                set: right_set,
                binding: right_binding,
                ..
            },
        )
        | (
            Binding::Uniform {
                set: left_set,
                binding: left_binding,
                ..
            },
            Binding::Uniform {
                set: right_set,
                binding: right_binding,
                ..
            },
        )
        | (
            Binding::Texture {
                set: left_set,
                binding: left_binding,
                ..
            },
            Binding::Texture {
                set: right_set,
                binding: right_binding,
                ..
            },
        )
        | (
            Binding::Sampler {
                set: left_set,
                binding: left_binding,
                ..
            },
            Binding::Sampler {
                set: right_set,
                binding: right_binding,
                ..
            },
        )
        | (
            Binding::StorageTexture {
                set: left_set,
                binding: left_binding,
                ..
            },
            Binding::StorageTexture {
                set: right_set,
                binding: right_binding,
                ..
            },
        ) => left_set == right_set && left_binding == right_binding,
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
