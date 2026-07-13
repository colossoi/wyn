//! First-class compute schedule produced by EGIR parallelization.
//!
//! A schedule owns the relationship between generated entry points, dispatch
//! domains, resource accesses, and phase dependencies.  The pipeline
//! descriptor is a publication format derived from this graph after every
//! lowering has finished; it is not mutated while an individual lowering is
//! still speculative.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use crate::egir::graph_ops;
use crate::egir::program::{
    CompilerFlowEndpoint, CompilerResourceFlow, EntryPublication, InputSlotId, LogicalResource,
    MaterializationRequirement, OutputSlotId, PhysicalResourceTable, PlannedEntryPublication,
    PlannedKernelBody, PrepassKind, PrepassRequirement, ResourceOrigin, SemanticEntry, SemanticEntryId,
    SemanticResourceDecl, SemanticResourceRef,
};
use crate::egir::types::{
    EgirSoac, FilterOutput, FilterPlan, FilterState, RuntimeFilterLength, SegExtent, SegPlacement,
    SegResourceAccessKind, SideEffectKind,
};
use crate::pipeline_descriptor::{
    Binding, ComputePipeline, ComputeStage, DispatchLen, DispatchSize, Pipeline, PipelineDescriptor,
};
use crate::ssa::types::{EntryInput, EntryOutput, ExecutionModel};
use crate::{BindingRef, ResourceId};

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
    publications: HashMap<String, PlannedEntryPublication>,
    publication_order: Vec<String>,
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
    pub fn published_plan(&self) -> KernelPlan {
        self.0.clone()
    }

    /// Entry ABI records in deterministic descriptor-publication order. The
    /// validated plan, rather than physical graphs, is the sole authority for
    /// backend-visible entry metadata.
    pub fn publications(&self, resources: &PhysicalResourceTable) -> Result<Vec<EntryPublication>, String> {
        self.0
            .publications_in_order()
            .into_iter()
            .map(|publication| publication.physicalize(resources))
            .collect()
    }

    pub(crate) fn physical_bodies(&self) -> impl Iterator<Item = &PlannedKernelBody> {
        self.0
            .phases()
            .map(|phase| phase.body.as_deref().expect("validated kernel plan contains a construction body"))
    }
}

/// Stable identity of a physical kernel in a plan. Dependencies use this id
/// instead of vector positions so insertion cannot silently retarget an edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KernelId(pub u32);

/// The only supported ways to construct a physical entry. Keeping this enum
/// closed prevents new lowering sites from falling back to arbitrary
/// clone-and-patch entry mutation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum KernelRecipe {
    GraphicsPassthrough,
    SerialCompute,
    OutputDomainProjection,
    MultiConsumerMaterialization,
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
    /// Closed construction payload consumed by physicalization. Metadata and
    /// the graph are captured when the recipe is committed, so physical EGIR
    /// never depends on a mutated semantic entry arena.
    pub(crate) body: Option<Arc<PlannedKernelBody>>,
    pub abi: EntryAbiProjection,
    pub workgroup_size: (u32, u32, u32),
    pub domain: KernelDomain,
    /// The authoritative selection request. Inferred selections retain their
    /// original fallback so reconciliation can recompute the complete domain
    /// after entry rewrites.
    pub domain_selection: DomainSelection,
    pub resources: Vec<ScheduledResource>,
    /// Stable kernel identities that must complete first.
    pub dependencies: Vec<KernelId>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceAccess {
    Read,
    Write,
    ReadWrite,
}

impl ResourceAccess {
    fn merge(self, other: Self) -> Self {
        if self == other {
            self
        } else {
            Self::ReadWrite
        }
    }

    pub fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    pub fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

impl KernelPlan {
    pub fn phases(&self) -> impl Iterator<Item = &KernelPhase> {
        self.pipelines
            .iter()
            .flat_map(|pipeline| pipeline.phases.iter())
            .chain(self.graphics_passthroughs.iter())
            .chain(self.unpublished.iter())
    }

    pub fn contains_entry(&self, entry_point: &str) -> bool {
        self.pipelines
            .iter()
            .any(|pipeline| pipeline.phases.iter().any(|phase| phase.entry_point == entry_point))
            || self.graphics_passthroughs.iter().any(|phase| phase.entry_point == entry_point)
    }

    pub fn contains_flow_source(&self, source: CompilerFlowEndpoint) -> bool {
        self.phases().any(|phase| phase.flow_source == Some(source))
    }

    pub fn entry_point_for_flow_source(&self, source: CompilerFlowEndpoint) -> Option<&str> {
        self.phases()
            .find(|phase| phase.flow_source == Some(source))
            .map(|phase| phase.entry_point.as_str())
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

    fn validate_program(
        &self,
        _entries: &[SemanticEntry],
        resources: &[LogicalResource],
        descriptor: &PipelineDescriptor,
    ) -> Result<(), String> {
        // The full-program check runs in `target_lowering` before validation;
        // this local check keeps direct plan-validation tests honest about the
        // resource arena passed to them.
        let resource_ids = resources.iter().map(|resource| resource.id).collect::<HashSet<_>>();
        let mut planned_names = HashSet::new();
        let mut projected_outputs: HashMap<SemanticEntryId, HashSet<OutputSlotId>> = HashMap::new();
        for phase in self.phases() {
            planned_names.insert(phase.entry_point.as_str());
            for required in &phase.resources {
                if !resource_ids.contains(&required.resource) {
                    return Err(format!(
                        "kernel `{}` references missing logical resource {:?}",
                        phase.entry_point, required.resource
                    ));
                }
            }
            let Some(source) = phase.abi.source_entry else {
                continue;
            };
            let Some(abi) = self.semantic_abi.get(&source) else {
                return Err(format!(
                    "kernel `{}` references missing semantic entry {:?}",
                    phase.entry_point, source
                ));
            };
            if phase.abi.inputs.iter().any(|slot| slot.0 >= abi.input_count) {
                return Err(format!(
                    "kernel `{}` has an invalid input projection",
                    phase.entry_point
                ));
            }
            let physical_output_count = phase.body.as_deref().map(|body| body.outputs.len()).unwrap_or(0);
            let mut physical_slots = HashSet::new();
            for route in &phase.abi.output_routes {
                let slot = route.semantic_slot;
                if slot.0 >= abi.output_count {
                    return Err(format!(
                        "kernel `{}` projects output {} from semantic entry `{}` with only {} outputs",
                        phase.entry_point, slot.0, abi.name, abi.output_count
                    ));
                }
                if route.physical_slot.0 >= physical_output_count
                    || !physical_slots.insert(route.physical_slot)
                {
                    return Err(format!(
                        "kernel `{}` has an invalid physical output-route projection",
                        phase.entry_point
                    ));
                }
                projected_outputs.entry(source).or_default().insert(slot);
            }
        }

        for resource in resources {
            let ResourceOrigin::Compiler(compiler) = &resource.origin else {
                continue;
            };
            let Some(flow) = &compiler.flow else {
                continue;
            };
            if !self
                .pipelines
                .iter()
                .flat_map(|pipeline| pipeline.phases.iter())
                .any(|phase| phase.flow_source.is_some_and(|source| flow.consumers.contains(&source)))
            {
                // Descriptor-less test/probe programs have executable EGIR
                // entries but no published stages. Allocation still verifies
                // their resource flow; there is no host schedule to check.
                continue;
            }
            let writers =
                self.phases()
                    .filter(|phase| {
                        phase.flow_source == Some(flow.producer)
                            && phase.resources.iter().any(|scheduled| {
                                scheduled.resource == resource.id && scheduled.access.writes()
                            })
                    })
                    .map(|phase| phase.id)
                    .collect::<HashSet<_>>();
            if writers.is_empty() {
                return Err(format!(
                    "resource {:?} ({:?}, flow {:?}) has no planned producer kernel",
                    resource.id, compiler.kind, flow
                ));
            }
            for consumer in &flow.consumers {
                let readers = self
                    .phases()
                    .filter(|phase| {
                        phase.flow_source == Some(*consumer)
                            && phase.resources.iter().any(|scheduled| {
                                scheduled.resource == resource.id && scheduled.access.reads()
                            })
                    })
                    .collect::<Vec<_>>();
                if readers.is_empty() {
                    return Err(format!(
                        "resource {:?} has no planned consumer kernel for {:?}",
                        resource.id, consumer
                    ));
                }
                for reader in readers {
                    if reader.recipe == KernelRecipe::GraphicsPassthrough {
                        continue;
                    }
                    if !reader.dependencies.iter().any(|dependency| writers.contains(dependency)) {
                        return Err(format!(
                            "kernel `{}` ({:?}) reads resource {:?} without depending on its producer {:?}; dependencies are {:?}",
                            reader.entry_point,
                            reader.id,
                            resource.id,
                            writers,
                            reader.dependencies
                        ));
                    }
                }
            }
        }

        for pipeline in &descriptor.pipelines {
            let stages = match pipeline {
                Pipeline::Compute(compute) => {
                    compute.stages.iter().map(|stage| stage.entry_point.as_str()).collect::<Vec<_>>()
                }
                Pipeline::Graphics(graphics) => {
                    graphics.stages.iter().map(|stage| stage.entry_point.as_str()).collect::<Vec<_>>()
                }
            };
            for stage in stages {
                if !planned_names.contains(stage) && matches!(pipeline, Pipeline::Compute(_)) {
                    return Err(format!("source compute stage `{stage}` has no planned kernel"));
                }
            }
        }

        let mut recipes_by_flow: HashMap<CompilerFlowEndpoint, HashSet<KernelRecipe>> = HashMap::new();
        for phase in self.phases() {
            if let Some(source) = phase.flow_source {
                recipes_by_flow.entry(source).or_default().insert(phase.recipe.clone());
            }
        }
        for (source, recipes) in &recipes_by_flow {
            let require = |recipe: KernelRecipe, context: &str| {
                if recipes.contains(&recipe) {
                    Ok(())
                } else {
                    Err(format!(
                        "compiler flow endpoint {source:?} has an incomplete {context} kernel family"
                    ))
                }
            };
            if recipes.contains(&KernelRecipe::FilterFlags) {
                require(KernelRecipe::FilterScan, "filter")?;
                require(KernelRecipe::FilterCombine, "filter")?;
                require(KernelRecipe::FilterScatter, "filter")?;
            }
            if recipes.contains(&KernelRecipe::ReducePhase1) {
                require(KernelRecipe::ReduceCombine, "reduce")?;
            }
            if recipes.contains(&KernelRecipe::ScanPhase1) {
                require(KernelRecipe::ScanBlock, "scan")?;
                require(KernelRecipe::ScanApplyOffsets, "scan")?;
            }
        }

        for (&source, abi) in &self.semantic_abi {
            if !recipes_by_flow.contains_key(&CompilerFlowEndpoint::Entry(source)) {
                continue;
            }
            let required = &abi.routed_outputs;
            if required.is_empty() {
                continue;
            }
            let projected = projected_outputs.get(&source).cloned().unwrap_or_default();
            if !required.is_subset(&projected) {
                let missing = required.difference(&projected).map(|slot| slot.0).collect::<Vec<_>>();
                return Err(format!(
                    "semantic entry `{}` has unplanned output slots {:?}",
                    abi.name, missing
                ));
            }
        }
        Ok(())
    }

    /// Seed a schedule from the source-level descriptor.  At this point every
    /// compute pipeline contains its original entry stage; later lowerings add
    /// phases to this graph without touching the descriptor.
    pub fn seed(
        descriptor: &PipelineDescriptor,
        entries: &[SemanticEntry],
        prepasses: &[PrepassRequirement],
        resources: &[LogicalResource],
    ) -> Self {
        let host_resources = resources
            .iter()
            .filter_map(|resource| match resource.origin {
                crate::egir::program::ResourceOrigin::Host(binding) => Some((binding, resource.id)),
                crate::egir::program::ResourceOrigin::Compiler(_) => None,
            })
            .collect::<HashMap<_, _>>();
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
        let mut publications = entries
            .iter()
            .map(|entry| (entry.name.clone(), entry.publication()))
            .collect::<HashMap<_, _>>();
        publications.extend(
            prepasses.iter().map(|prepass| (prepass.body.name.clone(), prepass.body.publication())),
        );
        let mut publication_order = entries.iter().map(|entry| entry.name.clone()).collect::<Vec<_>>();
        publication_order.extend(prepasses.iter().map(|prepass| prepass.body.name.clone()));
        let by_name: HashMap<&str, (SemanticEntryId, &SemanticEntry)> = entries
            .iter()
            .enumerate()
            .map(|(index, entry)| (entry.name.as_str(), (SemanticEntryId(index as u32), entry)))
            .collect();
        let prepass_by_name = prepasses
            .iter()
            .map(|prepass| (prepass.body.name.as_str(), prepass))
            .collect::<HashMap<_, _>>();
        let mut pipelines = Vec::new();
        let mut next_kernel_id = 0u32;
        for (pipeline_index, pipeline) in descriptor.pipelines.iter().enumerate() {
            let Pipeline::Compute(compute) = pipeline else {
                continue;
            };
            let phases = compute
                .stages
                .iter()
                .map(|stage| {
                    let id = KernelId(next_kernel_id);
                    next_kernel_id += 1;
                    let selection = domain_selection_from_stage(stage, &host_resources);
                    by_name
                        .get(stage.entry_point.as_str())
                        .map(|(source, entry)| {
                            phase_from_entry(
                                id,
                                Some(*source),
                                entry,
                                selection.clone(),
                                analyze_source_recipe(entry),
                            )
                        })
                        .or_else(|| {
                            prepass_by_name
                                .get(stage.entry_point.as_str())
                                .map(|prepass| phase_from_prepass(id, prepass, selection.clone()))
                        })
                        .unwrap_or_else(|| KernelPhase {
                            id,
                            flow_source: None,
                            entry_point: stage.entry_point.clone(),
                            recipe: KernelRecipe::SerialCompute,
                            body: None,
                            abi: EntryAbiProjection {
                                source_entry: None,
                                inputs: Vec::new(),
                                output_routes: Vec::new(),
                            },
                            workgroup_size: stage.workgroup_size,
                            domain: domain_from_dispatch(&stage.dispatch_size, &host_resources),
                            domain_selection: selection,
                            resources: Vec::new(),
                            dependencies: Vec::new(),
                        })
                })
                .collect();
            pipelines.push(ScheduledPipeline {
                order: pipeline_index,
                template: compute.clone(),
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
                    .map(|(source, entry)| graphics_passthrough_phase(id, *source, entry))
                    .unwrap_or_else(|| KernelPhase {
                        id,
                        flow_source: None,
                        entry_point: stage.entry_point.clone(),
                        recipe: KernelRecipe::GraphicsPassthrough,
                        body: None,
                        abi: EntryAbiProjection {
                            source_entry: None,
                            inputs: Vec::new(),
                            output_routes: Vec::new(),
                        },
                        workgroup_size: (1, 1, 1),
                        domain: KernelDomain::Fixed { x: 1, y: 1, z: 1 },
                        domain_selection: DomainSelection::Inferred(KernelDomain::Fixed {
                            x: 1,
                            y: 1,
                            z: 1,
                        }),
                        resources: Vec::new(),
                        dependencies: Vec::new(),
                    });
                graphics_passthroughs.push(phase);
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
            let source = SemanticEntryId(index as u32);
            let phase = if matches!(entry.execution_model, ExecutionModel::Compute { .. }) {
                phase_from_entry(
                    id,
                    Some(source),
                    entry,
                    DomainSelection::Inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                    analyze_source_recipe(entry),
                )
            } else {
                graphics_passthrough_phase(id, source, entry)
            };
            unpublished.push(phase);
        }
        for prepass in prepasses {
            if published_names.contains(prepass.body.name.as_str()) {
                continue;
            }
            let id = KernelId(next_kernel_id);
            next_kernel_id += 1;
            unpublished.push(phase_from_prepass(
                id,
                prepass,
                DomainSelection::Inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
            ));
        }
        Self {
            pipelines,
            graphics_passthroughs,
            unpublished,
            next_kernel_id,
            semantic_abi,
            publications,
            publication_order,
        }
    }

    fn record_planned_publication(&mut self, body: &PlannedKernelBody) {
        if !self.publications.contains_key(&body.name) {
            self.publication_order.push(body.name.clone());
            self.publications.insert(body.name.clone(), body.publication());
            return;
        }
        let publication = self.publications.get_mut(&body.name).expect("checked above");
        merge_publication(
            publication,
            &body.inputs,
            &body.outputs,
            &body.resource_declarations,
        );
    }

    fn publications_in_order(&self) -> Vec<&PlannedEntryPublication> {
        let mut ordered = Vec::with_capacity(self.publications.len());
        let mut seen = HashSet::new();
        for pipeline in &self.pipelines {
            for phase in &pipeline.phases {
                if seen.insert(phase.entry_point.as_str()) {
                    if let Some(publication) = self.publications.get(&phase.entry_point) {
                        ordered.push(publication);
                    }
                }
            }
        }
        for name in &self.publication_order {
            if seen.insert(name.as_str()) {
                if let Some(publication) = self.publications.get(name) {
                    ordered.push(publication);
                }
            }
        }
        ordered
    }

    /// Insert a generated phase immediately after `parent`. Any later phase
    /// that depended on `parent` is rewired to the new completion point, so a
    /// prepass family inserted before its consumer remains wholly before it.
    pub fn add_phase_after(
        &mut self,
        parent: &str,
        body: &PlannedKernelBody,
        domain: DomainSelection,
        recipe: KernelRecipe,
    ) {
        self.record_planned_publication(body);
        if let Some(parent_index) = self.unpublished.iter().position(|phase| phase.entry_point == parent) {
            let parent_id = self.unpublished[parent_index].id;
            let source_entry = self.unpublished[parent_index].abi.source_entry;
            let flow_source = self.unpublished[parent_index].flow_source;
            let id = KernelId(self.next_kernel_id);
            self.next_kernel_id += 1;
            let mut phase = phase_from_body(id, flow_source, source_entry, body, domain, recipe);
            phase.dependencies = vec![parent_id];
            self.unpublished.insert(parent_index + 1, phase);
            return;
        }
        let pipeline = self
            .pipelines
            .iter_mut()
            .find(|p| p.phases.iter().any(|phase| phase.entry_point == parent))
            .unwrap_or_else(|| panic!("no scheduled compute pipeline contains parent entry `{parent}`"));
        let parent_index = pipeline
            .phases
            .iter()
            .position(|phase| phase.entry_point == parent)
            .expect("parent was found above");
        let parent_id = pipeline.phases[parent_index].id;
        let source_entry = pipeline.phases[parent_index].abi.source_entry;
        let flow_source = pipeline.phases[parent_index].flow_source;
        let id = KernelId(self.next_kernel_id);
        self.next_kernel_id += 1;
        let mut phase = phase_from_body(id, flow_source, source_entry, body, domain, recipe);
        phase.dependencies = vec![parent_id];
        for dependent in pipeline.phases.iter_mut().skip(parent_index + 1) {
            for dependency in &mut dependent.dependencies {
                if *dependency == parent_id {
                    *dependency = id;
                }
            }
            dependent.dependencies.sort_unstable();
            dependent.dependencies.dedup();
        }
        pipeline.phases.insert(parent_index + 1, phase);
    }

    /// Insert a compiler-generated producer immediately before `consumer` and
    /// make the consumer depend on it. Existing dependency indices are shifted
    /// transactionally with the insertion.
    pub fn add_phase_before(
        &mut self,
        consumer: &str,
        body: &PlannedKernelBody,
        domain: DomainSelection,
        recipe: KernelRecipe,
    ) {
        self.record_planned_publication(body);
        if let Some(consumer_index) =
            self.unpublished.iter().position(|phase| phase.entry_point == consumer)
        {
            let inherited_dependencies = self.unpublished[consumer_index].dependencies.clone();
            let consumer_source = self.unpublished[consumer_index].abi.source_entry;
            let flow_source = self.unpublished[consumer_index].flow_source;
            let id = KernelId(self.next_kernel_id);
            self.next_kernel_id += 1;
            let mut producer = phase_from_body(id, flow_source, consumer_source, body, domain, recipe);
            producer.dependencies = inherited_dependencies;
            self.unpublished.insert(consumer_index, producer);
            self.unpublished[consumer_index + 1].dependencies = vec![id];
            return;
        }
        let pipeline = self
            .pipelines
            .iter_mut()
            .find(|pipeline| pipeline.phases.iter().any(|phase| phase.entry_point == consumer))
            .unwrap_or_else(|| {
                panic!("no scheduled compute pipeline contains consumer entry `{consumer}`")
            });
        let consumer_index = pipeline
            .phases
            .iter()
            .position(|phase| phase.entry_point == consumer)
            .expect("consumer was found above");
        let inherited_dependencies = pipeline.phases[consumer_index].dependencies.clone();
        let consumer_source = pipeline.phases[consumer_index].abi.source_entry;
        let flow_source = pipeline.phases[consumer_index].flow_source;
        let id = KernelId(self.next_kernel_id);
        self.next_kernel_id += 1;
        let mut producer = phase_from_body(id, flow_source, consumer_source, body, domain, recipe);
        producer.dependencies = inherited_dependencies;
        pipeline.phases.insert(consumer_index, producer);
        let consumer_phase = &mut pipeline.phases[consumer_index + 1];
        if !consumer_phase.dependencies.contains(&id) {
            consumer_phase.dependencies.push(id);
            consumer_phase.dependencies.sort_unstable();
        }
    }

    /// Insert a typed shared-materialization recipe before its first ready
    /// consumer. No semantic entry is synthesized or admitted to the entry
    /// arena.
    pub fn add_materialization_before(&mut self, consumer: &str, requirement: &MaterializationRequirement) {
        self.publication_order.push(requirement.name.clone());
        self.publications.insert(requirement.name.clone(), requirement.publication());
        if let Some(consumer_index) =
            self.unpublished.iter().position(|phase| phase.entry_point == consumer)
        {
            let dependencies = self.unpublished[consumer_index].dependencies.clone();
            let id = KernelId(self.next_kernel_id);
            self.next_kernel_id += 1;
            self.unpublished.insert(
                consumer_index,
                phase_from_materialization(id, requirement, dependencies),
            );
            self.unpublished[consumer_index + 1].dependencies = vec![id];
            return;
        }
        let pipeline = self
            .pipelines
            .iter_mut()
            .find(|pipeline| pipeline.phases.iter().any(|phase| phase.entry_point == consumer))
            .unwrap_or_else(|| {
                panic!("no scheduled compute pipeline contains consumer entry `{consumer}`")
            });
        let consumer_index = pipeline
            .phases
            .iter()
            .position(|phase| phase.entry_point == consumer)
            .expect("consumer was found above");
        let inherited_dependencies = pipeline.phases[consumer_index].dependencies.clone();
        let id = KernelId(self.next_kernel_id);
        self.next_kernel_id += 1;
        let producer = phase_from_materialization(id, requirement, inherited_dependencies);
        pipeline.phases.insert(consumer_index, producer);
        let consumer_phase = &mut pipeline.phases[consumer_index + 1];
        if !consumer_phase.dependencies.contains(&id) {
            consumer_phase.dependencies.push(id);
            consumer_phase.dependencies.sort_unstable();
        }
    }

    /// Add an independent sibling kernel to the same host pipeline. This is
    /// used for distinct output domains: source order is retained by the
    /// published phase list, but no data dependency is fabricated.
    pub fn add_sibling(
        &mut self,
        parent: &str,
        body: &PlannedKernelBody,
        domain: DomainSelection,
        recipe: KernelRecipe,
    ) {
        self.record_planned_publication(body);
        if let Some(parent) = self.unpublished.iter().find(|phase| phase.entry_point == parent) {
            let source_entry = parent.abi.source_entry;
            let flow_source = parent.flow_source;
            let id = KernelId(self.next_kernel_id);
            self.next_kernel_id += 1;
            self.unpublished.push(phase_from_body(
                id,
                flow_source,
                source_entry,
                body,
                domain,
                recipe,
            ));
            return;
        }
        let pipeline = self
            .pipelines
            .iter_mut()
            .find(|p| p.phases.iter().any(|phase| phase.entry_point == parent))
            .unwrap_or_else(|| panic!("no scheduled compute pipeline contains parent entry `{parent}`"));
        let parent_phase =
            pipeline.phases.iter().find(|phase| phase.entry_point == parent).expect("parent phase exists");
        let source_entry = parent_phase.abi.source_entry;
        let flow_source = parent_phase.flow_source;
        let id = KernelId(self.next_kernel_id);
        self.next_kernel_id += 1;
        let phase = phase_from_body(id, flow_source, source_entry, body, domain, recipe);
        pipeline.phases.push(phase);
    }

    pub fn domain_of(&self, entry_point: &str) -> Option<KernelDomain> {
        self.pipelines
            .iter()
            .flat_map(|pipeline| &pipeline.phases)
            .find(|phase| phase.entry_point == entry_point)
            .map(|phase| phase.domain.clone())
    }

    pub(crate) fn planned_body(&self, entry_point: &str) -> Option<&PlannedKernelBody> {
        self.phases().find(|phase| phase.entry_point == entry_point).and_then(|phase| phase.body.as_deref())
    }

    pub(crate) fn for_each_body_mut(&mut self, mut visit: impl FnMut(&mut PlannedKernelBody)) {
        for pipeline in &mut self.pipelines {
            for phase in &mut pipeline.phases {
                if let Some(body) = &mut phase.body {
                    visit(Arc::make_mut(body));
                }
            }
        }
        for phase in self.graphics_passthroughs.iter_mut().chain(self.unpublished.iter_mut()) {
            if let Some(body) = &mut phase.body {
                visit(Arc::make_mut(body));
            }
        }
    }

    pub fn set_output_projection(&mut self, entry_point: &str, outputs: Vec<OutputSlotId>) {
        let phase = self
            .pipelines
            .iter_mut()
            .flat_map(|pipeline| &mut pipeline.phases)
            .find(|phase| phase.entry_point == entry_point)
            .unwrap_or_else(|| panic!("no planned kernel named `{entry_point}`"));
        phase.abi.output_routes = outputs
            .into_iter()
            .enumerate()
            .map(|(physical, semantic_slot)| OutputRouteProjection {
                semantic_slot,
                physical_slot: OutputSlotId(physical),
            })
            .collect();
    }

    /// Commit the facts for one kernel at the same boundary that commits its
    /// physical recipe. This is deliberately per-entry: a later global
    /// reconciliation pass would make the entry list authoritative again.
    pub fn commit_kernel(&mut self, body: &PlannedKernelBody, recipe: KernelRecipe) {
        self.record_planned_publication(body);
        let phase = self
            .pipelines
            .iter_mut()
            .flat_map(|pipeline| pipeline.phases.iter_mut())
            .chain(self.unpublished.iter_mut())
            .find(|phase| phase.entry_point == body.name)
            .unwrap_or_else(|| panic!("no planned kernel named `{}`", body.name));
        phase.workgroup_size = body_workgroup(body);
        if let DomainSelection::Inferred(fallback) = &phase.domain_selection {
            phase.domain = inferred_body_domain(body, fallback.clone());
        }
        let filter_phase = body.graph.skeleton.blocks.iter().any(|(_, block)| {
            block.side_effects.iter().any(|effect| {
                matches!(
                    effect.kind,
                    SideEffectKind::Soac(EgirSoac::Filter {
                        state: FilterState::Scheduled { .. },
                        ..
                    })
                )
            })
        });
        phase.resources = if filter_phase {
            segmented_body_resources(body).unwrap_or_else(|| body_resources(body))
        } else {
            body_resources(body)
        };
        phase.recipe = recipe;
        phase.body = Some(Arc::new(body.clone()));
        phase.abi.inputs = (0..body.inputs.len()).map(InputSlotId).collect();
        phase.abi.output_routes = body_output_projection(body);
    }

    /// Reject an explicit `#[dispatch(...)]` grid that provably under-covers an
    /// entry's data-parallel domain. Only statically-sized domains
    /// (`DispatchLen::Fixed`) are decidable here; runtime-length domains keep
    /// the generated per-element `tid < len` guard (which is over-launch safe
    /// but cannot catch an under-launch at compile time). Compiler-pinned
    /// phases (filter scan, reduce/scan combines) have no static SOAC domain,
    /// so `segmented_domain` returns `None` for them and they are skipped.
    pub fn check_explicit_dispatch_coverage(&self, entries: &[SemanticEntry]) -> Result<(), String> {
        let by_name: HashMap<&str, &SemanticEntry> = entries.iter().map(|e| (e.name.as_str(), e)).collect();
        for pipeline in &self.pipelines {
            for phase in &pipeline.phases {
                let DomainSelection::Explicit(KernelDomain::Fixed { x, y, z }) = &phase.domain_selection
                else {
                    continue;
                };
                let Some(entry) = by_name.get(phase.entry_point.as_str()) else {
                    continue;
                };
                let Some(KernelDomain::Elements(DispatchLen::Fixed { count })) = segmented_domain(entry)
                else {
                    continue;
                };
                let (wx, wy, wz) = entry_workgroup(entry);
                let total = x
                    .saturating_mul(*y)
                    .saturating_mul(*z)
                    .saturating_mul(wx)
                    .saturating_mul(wy)
                    .saturating_mul(wz);
                if total < count {
                    return Err(format!(
                        "#[dispatch({x}, {y}, {z})] launches {total} threads but entry '{}' \
                         has a data-parallel domain of {count} elements, so {} would be \
                         dropped; enlarge the grid or drop #[dispatch] to let the compiler \
                         size the launch.",
                        phase.entry_point,
                        count - total
                    ));
                }
            }
        }
        Ok(())
    }

    /// Coalesce pipelines and add dependencies from the resource manifest's
    /// explicit producer/consumer edges. Entry provenance and declaration
    /// roles are deliberately not consulted here.
    pub fn coalesce_resource_flows(&mut self, resources: &[LogicalResource]) {
        let mut flows = resources
            .iter()
            .filter_map(|resource| match &resource.origin {
                ResourceOrigin::Compiler(compiler) => {
                    compiler.flow.as_ref().map(|flow| (resource.id, flow.clone()))
                }
                ResourceOrigin::Host(_) => None,
            })
            .collect::<Vec<_>>();
        flows.sort_by_key(|(resource, _)| resource.0);
        self.complete_resource_flow_accesses(&flows);
        self.add_resource_flow_dependencies(&flows);

        let mut kernel_pipeline = HashMap::new();
        for (pipeline_index, pipeline) in self.pipelines.iter().enumerate() {
            for phase in &pipeline.phases {
                kernel_pipeline.insert(phase.id, pipeline_index);
            }
        }
        let mut parents = (0..self.pipelines.len()).collect::<Vec<_>>();
        for (pipeline_index, pipeline) in self.pipelines.iter().enumerate() {
            for phase in &pipeline.phases {
                for dependency in &phase.dependencies {
                    if let Some(&dependency_pipeline) = kernel_pipeline.get(dependency) {
                        union_components(&mut parents, pipeline_index, dependency_pipeline);
                    }
                }
            }
        }
        let mut components: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for index in 0..self.pipelines.len() {
            let root = find_component(&mut parents, index);
            components.entry(root).or_default().push(index);
        }
        let mut slots = std::mem::take(&mut self.pipelines).into_iter().map(Some).collect::<Vec<_>>();
        let mut merged_pipelines = Vec::with_capacity(components.len());
        for mut indices in components.into_values() {
            indices.sort_by_key(|index| slots[*index].as_ref().expect("unclaimed pipeline").order);
            let first = indices.remove(0);
            let mut merged = slots[first].take().expect("component pipeline claimed once");
            for index in indices {
                let mut pipeline = slots[index].take().expect("component pipeline claimed once");
                merged.order = merged.order.min(pipeline.order);
                merge_bindings(&mut merged.template.bindings, pipeline.template.bindings);
                if merged.template.default_total_threads.is_none() {
                    merged.template.default_total_threads = pipeline.template.default_total_threads;
                }
                for feedback in pipeline.template.feedback {
                    if !merged.template.feedback.contains(&feedback) {
                        merged.template.feedback.push(feedback);
                    }
                }
                merged.phases.append(&mut pipeline.phases);
            }
            merged.phases = topologically_order_phases(merged.phases);
            merged_pipelines.push(merged);
        }
        merged_pipelines.sort_by_key(|pipeline| pipeline.order);
        self.pipelines = merged_pipelines;
    }

    fn complete_resource_flow_accesses(&mut self, flows: &[(ResourceId, CompilerResourceFlow)]) {
        for (resource, flow) in flows {
            let has_writer = self.phases().any(|phase| {
                phase.flow_source == Some(flow.producer)
                    && phase
                        .resources
                        .iter()
                        .any(|scheduled| scheduled.resource == *resource && scheduled.access.writes())
            });
            if !has_writer {
                let writer = self
                    .phases()
                    .filter(|phase| phase.flow_source == Some(flow.producer))
                    .map(|phase| phase.id)
                    .last();
                if let Some(writer) = writer {
                    let phase = self.phase_mut(writer).expect("planned producer phase disappeared");
                    merge_scheduled_resource(&mut phase.resources, *resource, ResourceAccess::Write);
                }
            }
            for consumer in &flow.consumers {
                let has_reader = self.phases().any(|phase| {
                    phase.flow_source == Some(*consumer)
                        && phase
                            .resources
                            .iter()
                            .any(|scheduled| scheduled.resource == *resource && scheduled.access.reads())
                });
                if !has_reader {
                    let reader = self
                        .phases()
                        .find(|phase| phase.flow_source == Some(*consumer))
                        .map(|phase| phase.id);
                    if let Some(reader) = reader {
                        let phase = self.phase_mut(reader).expect("planned consumer phase disappeared");
                        merge_scheduled_resource(&mut phase.resources, *resource, ResourceAccess::Read);
                    }
                }
            }
        }
    }

    fn phase_mut(&mut self, id: KernelId) -> Option<&mut KernelPhase> {
        self.pipelines
            .iter_mut()
            .flat_map(|pipeline| pipeline.phases.iter_mut())
            .chain(self.graphics_passthroughs.iter_mut())
            .find(|phase| phase.id == id)
    }

    fn add_resource_flow_dependencies(&mut self, flows: &[(ResourceId, CompilerResourceFlow)]) {
        for (resource, flow) in flows {
            let producer_ids =
                self.phases()
                    .filter(|phase| {
                        phase.flow_source == Some(flow.producer)
                            && phase.resources.iter().any(|scheduled| {
                                scheduled.resource == *resource && scheduled.access.writes()
                            })
                    })
                    .map(|phase| phase.id)
                    .collect::<Vec<_>>();
            for pipeline in &mut self.pipelines {
                for phase in &mut pipeline.phases {
                    if phase.flow_source.is_some_and(|source| flow.consumers.contains(&source))
                        && phase
                            .resources
                            .iter()
                            .any(|scheduled| scheduled.resource == *resource && scheduled.access.reads())
                    {
                        for producer in &producer_ids {
                            if !phase.dependencies.contains(producer) {
                                phase.dependencies.push(*producer);
                            }
                        }
                        phase.dependencies.sort_unstable();
                    }
                }
            }
        }
    }

    /// Validate graph-local invariants before publishing a host ABI.
    pub fn validate(&self) -> Result<(), String> {
        let mut names = HashSet::new();
        let mut ids = HashSet::new();
        for pipeline in &self.pipelines {
            let positions: HashMap<KernelId, usize> =
                pipeline.phases.iter().enumerate().map(|(index, phase)| (phase.id, index)).collect();
            for (index, phase) in pipeline.phases.iter().enumerate() {
                if phase.body.is_none() {
                    return Err(format!(
                        "kernel `{}` has no closed construction body",
                        phase.entry_point
                    ));
                }
                if !ids.insert(phase.id) {
                    return Err(format!("kernel id {:?} appears more than once", phase.id));
                }
                if !names.insert(phase.entry_point.as_str()) {
                    return Err(format!(
                        "entry `{}` appears in more than one scheduled phase",
                        phase.entry_point
                    ));
                }
                if phase.dependencies.iter().any(|dependency| {
                    positions.get(dependency).is_none_or(|&dependency_index| dependency_index >= index)
                }) {
                    return Err(format!(
                        "phase `{}` has a non-prior dependency: {:?}",
                        phase.entry_point, phase.dependencies
                    ));
                }
            }
        }
        for phase in &self.graphics_passthroughs {
            if phase.body.is_none() {
                return Err(format!(
                    "graphics entry `{}` has no closed construction body",
                    phase.entry_point
                ));
            }
            if !ids.insert(phase.id) {
                return Err(format!("kernel id {:?} appears more than once", phase.id));
            }
            if !names.insert(phase.entry_point.as_str()) {
                return Err(format!(
                    "entry `{}` appears in more than one scheduled phase",
                    phase.entry_point
                ));
            }
            if phase.recipe != KernelRecipe::GraphicsPassthrough
                || !phase.dependencies.is_empty()
                || phase.domain != (KernelDomain::Fixed { x: 1, y: 1, z: 1 })
            {
                return Err(format!(
                    "graphics entry `{}` is not an unchanged passthrough",
                    phase.entry_point
                ));
            }
        }
        let unpublished_positions = self
            .unpublished
            .iter()
            .enumerate()
            .map(|(index, phase)| (phase.id, index))
            .collect::<HashMap<_, _>>();
        for (index, phase) in self.unpublished.iter().enumerate() {
            if phase.body.is_none() {
                return Err(format!(
                    "unpublished entry `{}` has no closed construction body",
                    phase.entry_point
                ));
            }
            if !ids.insert(phase.id) {
                return Err(format!("kernel id {:?} appears more than once", phase.id));
            }
            if !names.insert(phase.entry_point.as_str()) {
                return Err(format!(
                    "entry `{}` appears in more than one planned kernel",
                    phase.entry_point
                ));
            }
            if phase.dependencies.iter().any(|dependency| {
                unpublished_positions
                    .get(dependency)
                    .is_none_or(|&dependency_index| dependency_index >= index)
            }) {
                return Err(format!(
                    "unpublished entry `{}` has a non-prior dependency: {:?}",
                    phase.entry_point, phase.dependencies
                ));
            }
        }
        Ok(())
    }

    /// Install entry-point shells before binding publication. This makes every
    /// generated entry discoverable by `publish_implicit_bindings`; the final
    /// `publish` call below fills dispatch and resource-index lists.
    pub fn install_phase_shells(&self, descriptor: &mut PipelineDescriptor) -> Result<(), String> {
        self.validate()?;
        let mut rebuilt: Vec<(usize, Pipeline)> = descriptor
            .pipelines
            .iter()
            .enumerate()
            .filter_map(|(index, pipeline)| {
                matches!(pipeline, Pipeline::Graphics(_)).then(|| (index, pipeline.clone()))
            })
            .collect();
        for scheduled in &self.pipelines {
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

    /// Materialise stages and their resource-index lists into the descriptor.
    /// Binding declarations must already have been published from the final
    /// entry list; this method never invents a binding.
    pub fn publish(
        &self,
        descriptor: &mut PipelineDescriptor,
        physical_resources: &PhysicalResourceTable,
    ) -> Result<(), String> {
        self.validate()?;
        for scheduled in &self.pipelines {
            let Some(Pipeline::Compute(compute)) =
                descriptor.pipelines.iter_mut().find(|pipeline| match pipeline {
                    Pipeline::Compute(compute) => scheduled.phases.iter().any(|phase| {
                        compute.stages.iter().any(|stage| stage.entry_point == phase.entry_point)
                    }),
                    _ => false,
                })
            else {
                return Err("scheduled compute pipeline was not installed".into());
            };
            let binding_index: HashMap<BindingRef, usize> = compute
                .bindings
                .iter()
                .enumerate()
                .filter_map(|(index, binding)| binding_ref(binding).map(|binding| (binding, index)))
                .collect();
            let mut stages = Vec::with_capacity(scheduled.phases.len());
            for phase in &scheduled.phases {
                let mut reads = Vec::new();
                let mut writes = Vec::new();
                for resource in &phase.resources {
                    let binding = physical_resources.binding(resource.resource).ok_or_else(|| {
                        format!(
                            "scheduled phase `{}` references unallocated resource {:?}",
                            phase.entry_point, resource.resource
                        )
                    })?;
                    let Some(&index) = binding_index.get(&binding) else {
                        return Err(format!(
                            "scheduled phase `{}` references unpublished storage {} for resource {:?}",
                            phase.entry_point, binding, resource.resource
                        ));
                    };
                    if resource.access.reads() && !reads.contains(&index) {
                        reads.push(index);
                    }
                    if resource.access.writes() && !writes.contains(&index) {
                        writes.push(index);
                    }
                }
                stages.push(ComputeStage {
                    entry_point: phase.entry_point.clone(),
                    workgroup_size: phase.workgroup_size,
                    dispatch_size: match &phase.domain {
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
                                format!(
                                    "scheduled phase `{}` dispatches from unallocated resource {:?}",
                                    phase.entry_point, resource
                                )
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
                    },
                    reads,
                    writes,
                });
            }
            compute.stages = stages;
        }
        Ok(())
    }
}

fn merge_publication(
    publication: &mut PlannedEntryPublication,
    inputs: &[EntryInput],
    outputs: &[EntryOutput],
    declarations: &[SemanticResourceDecl],
) {
    for input in inputs {
        if !publication.inputs.iter().any(|existing| {
            existing.storage_binding == input.storage_binding
                && existing.uniform_binding == input.uniform_binding
                && existing.texture_binding == input.texture_binding
                && existing.sampler_binding == input.sampler_binding
                && existing.storage_image_binding == input.storage_image_binding
                && existing.name == input.name
        }) {
            publication.inputs.push(input.clone());
        }
    }
    for output in outputs {
        if output.storage_binding.is_none() && output.target.is_none() {
            continue;
        }
        if !publication.outputs.iter().any(|existing| {
            existing.storage_binding == output.storage_binding && existing.target == output.target
        }) {
            publication.outputs.push(output.clone());
        }
    }
    for declaration in declarations {
        if !publication
            .resources
            .iter()
            .any(|existing| existing.resource == declaration.resource && existing.role == declaration.role)
        {
            publication.resources.push(declaration.clone());
        }
    }
}

fn phase_from_entry(
    id: KernelId,
    source_entry: Option<SemanticEntryId>,
    entry: &SemanticEntry,
    selection: DomainSelection,
    recipe: KernelRecipe,
) -> KernelPhase {
    let domain = match &selection {
        DomainSelection::Inferred(fallback) => inferred_domain(entry, fallback.clone()),
        DomainSelection::Explicit(domain) => domain.clone(),
    };
    KernelPhase {
        id,
        flow_source: source_entry.map(CompilerFlowEndpoint::Entry),
        entry_point: entry.name.clone(),
        recipe,
        body: Some(Arc::new(
            PlannedKernelBody::from_semantic(entry)
                .expect("validated semantic entry projection must be constructible"),
        )),
        abi: EntryAbiProjection {
            source_entry,
            inputs: (0..entry.inputs.len()).map(InputSlotId).collect(),
            output_routes: entry_output_projection(entry),
        },
        workgroup_size: entry_workgroup(entry),
        domain,
        domain_selection: selection,
        resources: segmented_resources(entry).unwrap_or_else(|| entry_resources(entry)),
        dependencies: Vec::new(),
    }
}

fn phase_from_body(
    id: KernelId,
    flow_source: Option<CompilerFlowEndpoint>,
    source_entry: Option<SemanticEntryId>,
    body: &PlannedKernelBody,
    selection: DomainSelection,
    recipe: KernelRecipe,
) -> KernelPhase {
    let domain = match &selection {
        DomainSelection::Inferred(fallback) => inferred_body_domain(body, fallback.clone()),
        DomainSelection::Explicit(domain) => domain.clone(),
    };
    KernelPhase {
        id,
        flow_source,
        entry_point: body.name.clone(),
        recipe,
        body: Some(Arc::new(body.clone())),
        abi: EntryAbiProjection {
            source_entry,
            inputs: (0..body.inputs.len()).map(InputSlotId).collect(),
            output_routes: body_output_projection(body),
        },
        workgroup_size: body_workgroup(body),
        domain,
        domain_selection: selection,
        resources: segmented_body_resources(body).unwrap_or_else(|| body_resources(body)),
        dependencies: Vec::new(),
    }
}

fn phase_from_prepass(
    id: KernelId,
    prepass: &PrepassRequirement,
    selection: DomainSelection,
) -> KernelPhase {
    let recipe = match prepass.kind {
        PrepassKind::Scalar => KernelRecipe::ScalarPrepass,
        PrepassKind::Gather => KernelRecipe::GatherPrepass,
    };
    phase_from_body(
        id,
        Some(CompilerFlowEndpoint::Prepass(prepass.id)),
        None,
        &prepass.body,
        selection,
        recipe,
    )
}

fn phase_from_materialization(
    id: KernelId,
    requirement: &MaterializationRequirement,
    dependencies: Vec<KernelId>,
) -> KernelPhase {
    let domain = materialization_domain(requirement).unwrap_or(KernelDomain::Fixed { x: 1, y: 1, z: 1 });
    let mut phase = KernelPhase {
        id,
        flow_source: Some(CompilerFlowEndpoint::Materialization(requirement.id)),
        entry_point: requirement.name.clone(),
        recipe: KernelRecipe::MultiConsumerMaterialization,
        body: Some(Arc::new(
            requirement.planned_body().expect("validated materialization projection must be constructible"),
        )),
        abi: EntryAbiProjection {
            source_entry: None,
            inputs: Vec::new(),
            output_routes: Vec::new(),
        },
        workgroup_size: match requirement.execution_model {
            ExecutionModel::Compute { local_size } => local_size,
            _ => (1, 1, 1),
        },
        domain: domain.clone(),
        domain_selection: DomainSelection::Explicit(domain),
        resources: materialization_resources(requirement),
        dependencies,
    };
    phase.resources.sort_by_key(|resource| resource.resource);
    phase
}

fn graphics_passthrough_phase(
    id: KernelId,
    source_entry: SemanticEntryId,
    entry: &SemanticEntry,
) -> KernelPhase {
    let domain = KernelDomain::Fixed { x: 1, y: 1, z: 1 };
    KernelPhase {
        id,
        flow_source: Some(CompilerFlowEndpoint::Entry(source_entry)),
        entry_point: entry.name.clone(),
        recipe: KernelRecipe::GraphicsPassthrough,
        body: Some(Arc::new(
            PlannedKernelBody::from_semantic(entry)
                .expect("validated graphics entry projection must be constructible"),
        )),
        abi: EntryAbiProjection {
            source_entry: Some(source_entry),
            inputs: (0..entry.inputs.len()).map(InputSlotId).collect(),
            output_routes: entry_output_projection(entry),
        },
        workgroup_size: (1, 1, 1),
        domain: domain.clone(),
        domain_selection: DomainSelection::Inferred(domain),
        resources: entry_resources(entry),
        dependencies: Vec::new(),
    }
}

fn entry_output_projection(entry: &SemanticEntry) -> Vec<OutputRouteProjection> {
    let mut slots = entry.output_routes.iter().map(|route| route.slot).collect::<Vec<_>>();
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

fn body_output_projection(body: &PlannedKernelBody) -> Vec<OutputRouteProjection> {
    let mut slots = body.output_routes.iter().map(|route| route.slot).collect::<Vec<_>>();
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

fn analyze_source_recipe(entry: &SemanticEntry) -> KernelRecipe {
    source_recipe(entry)
}

fn source_recipe(entry: &SemanticEntry) -> KernelRecipe {
    if !matches!(entry.execution_model, ExecutionModel::Compute { .. }) {
        return KernelRecipe::GraphicsPassthrough;
    }
    for (_, block) in &entry.graph.skeleton.blocks {
        for effect in &block.side_effects {
            if let SideEffectKind::Soac(EgirSoac::Seg {
                kind,
                placement: SegPlacement::Kernel,
                ..
            }) = &effect.kind
            {
                return match kind {
                    crate::egir::types::SegOpKind::SegRed { .. } => KernelRecipe::ReducePhase1,
                    crate::egir::types::SegOpKind::SegScan { .. } => KernelRecipe::ScanPhase1,
                    _ => KernelRecipe::SerialCompute,
                };
            }
            if let SideEffectKind::Soac(EgirSoac::Filter {
                state:
                    FilterState::Scheduled {
                        plan: FilterPlan::Scatter(_),
                        ..
                    },
                ..
            }) = &effect.kind
            {
                return KernelRecipe::FilterScatter;
            }
        }
    }
    KernelRecipe::SerialCompute
}

fn inferred_domain(entry: &SemanticEntry, fallback: KernelDomain) -> KernelDomain {
    segmented_domain(entry).or_else(|| storage_image_domain(entry, &fallback)).unwrap_or(fallback)
}

fn inferred_body_domain(body: &PlannedKernelBody, fallback: KernelDomain) -> KernelDomain {
    segmented_domain_graph(&body.graph)
        .or_else(|| storage_image_domain_inputs(&body.inputs, &fallback))
        .unwrap_or(fallback)
}

/// A compute entry with no SOAC-derived domain and a `#[storage_image]` param
/// runs one thread per texel of the image (the mountains / one_weekend
/// per-pixel pass shape) — the host resolves the size from the bound texture's
/// extent. Incidental storage-buffer inputs (e.g. mountains' keyboard buffer)
/// don't opt out; the image is the domain. Only upgrades the single-workgroup
/// placeholder domain; an explicit fixed grid stays as scheduled.
///
fn storage_image_domain(entry: &SemanticEntry, fallback: &KernelDomain) -> Option<KernelDomain> {
    storage_image_domain_inputs(&entry.inputs, fallback)
}

fn storage_image_domain_inputs(inputs: &[EntryInput], fallback: &KernelDomain) -> Option<KernelDomain> {
    if !matches!(fallback, KernelDomain::Fixed { x: 1, y: 1, z: 1 }) {
        return None;
    }
    let (binding, ..) = inputs.iter().find_map(|input| input.storage_image_binding)?;
    Some(KernelDomain::Elements(DispatchLen::StorageImage {
        set: binding.set,
        binding: binding.binding,
    }))
}

fn segmented_resources(entry: &SemanticEntry) -> Option<Vec<ScheduledResource>> {
    segmented_graph_resources(&entry.graph, &entry.resource_declarations)
}

fn segmented_body_resources(body: &PlannedKernelBody) -> Option<Vec<ScheduledResource>> {
    segmented_graph_resources(&body.graph, &body.resource_declarations)
}

fn segmented_graph_resources(
    graph: &crate::egir::types::EGraph,
    declarations: &[SemanticResourceDecl],
) -> Option<Vec<ScheduledResource>> {
    for (_, block) in &graph.skeleton.blocks {
        for side_effect in &block.side_effects {
            if let SideEffectKind::Soac(EgirSoac::Filter {
                state: FilterState::Scheduled { plan, .. },
                output,
                ..
            }) = &side_effect.kind
            {
                let mut resources = graph_resources(graph, declarations);
                let mut push = |reference: SemanticResourceRef, access: ResourceAccess| {
                    let resource =
                        reference.resource().expect("planner received a pending filter resource binding");
                    merge_scheduled_resource(&mut resources, resource, access);
                };
                match plan {
                    FilterPlan::Flags(work) => {
                        push(work.flags, ResourceAccess::Write);
                    }
                    FilterPlan::Scan(work) => {
                        push(work.flags, ResourceAccess::Read);
                        push(work.offsets, ResourceAccess::Write);
                        push(work.block_sums, ResourceAccess::Write);
                    }
                    FilterPlan::Scatter(work) => {
                        push(work.flags, ResourceAccess::Read);
                        push(work.offsets, ResourceAccess::Read);
                        push(work.block_offsets, ResourceAccess::Read);
                        if let FilterOutput::Runtime { scratch, length } = output {
                            if let RuntimeFilterLength::EntryOutput(binding) = length {
                                push(*binding, ResourceAccess::Read);
                            }
                            push(*scratch, ResourceAccess::Write);
                        }
                    }
                    FilterPlan::Serial => continue,
                }
                resources.sort_by_key(|resource| resource.resource);
                return Some(resources);
            }
            let SideEffectKind::Soac(EgirSoac::Seg {
                placement: SegPlacement::Kernel,
                resources,
                ..
            }) = &side_effect.kind
            else {
                continue;
            };
            return Some(
                resources
                    .iter()
                    .map(|resource| ScheduledResource {
                        resource: resource
                            .resource
                            .resource()
                            .expect("planner received a pending Seg resource binding"),
                        access: match resource.access {
                            SegResourceAccessKind::Read => ResourceAccess::Read,
                            SegResourceAccessKind::Write => ResourceAccess::Write,
                            SegResourceAccessKind::ReadWrite => ResourceAccess::ReadWrite,
                        },
                    })
                    .collect(),
            );
        }
    }
    None
}

fn entry_workgroup(entry: &SemanticEntry) -> (u32, u32, u32) {
    match entry.execution_model {
        ExecutionModel::Compute { local_size } => local_size,
        _ => (1, 1, 1),
    }
}

fn body_workgroup(body: &PlannedKernelBody) -> (u32, u32, u32) {
    match body.execution_model {
        ExecutionModel::Compute { local_size } => local_size,
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
    segmented_domain_graph(&entry.graph)
}

fn materialization_domain(requirement: &MaterializationRequirement) -> Option<KernelDomain> {
    segmented_domain_graph(&requirement.graph)
}

fn segmented_domain_graph(graph: &crate::egir::types::EGraph) -> Option<KernelDomain> {
    for (_, block) in &graph.skeleton.blocks {
        for side_effect in &block.side_effects {
            match &side_effect.kind {
                SideEffectKind::Soac(EgirSoac::Seg {
                    space,
                    placement: SegPlacement::Kernel,
                    ..
                }) => return domain_from_space(space),
                SideEffectKind::Soac(EgirSoac::Filter {
                    state:
                        FilterState::Scheduled {
                            space,
                            plan: FilterPlan::Flags(_) | FilterPlan::Scatter(_),
                        },
                    ..
                }) => return domain_from_space(space),
                _ => {}
            }
        }
    }
    None
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
            resource: resource.resource().expect("planner received a pending resource-length binding"),
            elem_bytes: *elem_bytes,
        }),
        _ => None,
    }
}

fn entry_resources(entry: &SemanticEntry) -> Vec<ScheduledResource> {
    graph_resources(&entry.graph, &entry.resource_declarations)
}

fn body_resources(body: &PlannedKernelBody) -> Vec<ScheduledResource> {
    graph_resources(&body.graph, &body.resource_declarations)
}

fn materialization_resources(requirement: &MaterializationRequirement) -> Vec<ScheduledResource> {
    kernel_seg_resources(&requirement.graph)
        .unwrap_or_else(|| graph_resources(&requirement.graph, &requirement.resource_declarations))
}

fn kernel_seg_resources(graph: &crate::egir::types::EGraph) -> Option<Vec<ScheduledResource>> {
    for (_, block) in &graph.skeleton.blocks {
        for side_effect in &block.side_effects {
            let SideEffectKind::Soac(EgirSoac::Seg {
                placement: SegPlacement::Kernel,
                resources,
                ..
            }) = &side_effect.kind
            else {
                continue;
            };
            return Some(
                resources
                    .iter()
                    .map(|resource| ScheduledResource {
                        resource: resource
                            .resource
                            .resource()
                            .expect("planner received a pending Seg resource binding"),
                        access: match resource.access {
                            SegResourceAccessKind::Read => ResourceAccess::Read,
                            SegResourceAccessKind::Write => ResourceAccess::Write,
                            SegResourceAccessKind::ReadWrite => ResourceAccess::ReadWrite,
                        },
                    })
                    .collect(),
            );
        }
    }
    None
}

fn graph_resources(
    graph: &crate::egir::types::EGraph,
    declarations: &[crate::egir::program::SemanticResourceDecl],
) -> Vec<ScheduledResource> {
    let mut accesses: HashMap<ResourceId, ResourceAccess> = HashMap::new();
    let mut insert = |reference: SemanticResourceRef, access: ResourceAccess| {
        let resource = reference.resource().expect("planner received a pending entry resource binding");
        accesses.entry(resource).and_modify(|old| *old = old.merge(access)).or_insert(access);
    };

    for declaration in declarations {
        let access = match declaration.role {
            crate::interface::StorageRole::Input => ResourceAccess::Read,
            crate::interface::StorageRole::Output => ResourceAccess::Write,
            crate::interface::StorageRole::Intermediate => ResourceAccess::ReadWrite,
        };
        insert(declaration.resource, access);
    }

    // A storage view reachable from an effect operand is conservatively a
    // read. Output/intermediate metadata above upgrades it when it is written.
    for (_, block) in &graph.skeleton.blocks {
        for side_effect in &block.side_effects {
            wyn_graph::for_each_reachable(
                side_effect.referenced_nodes(),
                wyn_graph::WalkOrder::DepthFirst,
                |node, out| out.extend(graph.nodes[node].children()),
                |node| {
                    if let Some(resource) = graph_ops::extract_storage_view_source(graph, node) {
                        insert(resource, ResourceAccess::Read);
                    }
                },
            );
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

fn find_component(parents: &mut [usize], index: usize) -> usize {
    if parents[index] != index {
        parents[index] = find_component(parents, parents[index]);
    }
    parents[index]
}

fn union_components(parents: &mut [usize], left: usize, right: usize) {
    let left = find_component(parents, left);
    let right = find_component(parents, right);
    if left != right {
        parents[right] = left;
    }
}

fn topologically_order_phases(phases: Vec<KernelPhase>) -> Vec<KernelPhase> {
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
            return ordered;
        };
        let phase = remaining[index].take().expect("ready phase exists");
        emitted.insert(phase.id);
        ordered.push(phase);
    }
    ordered
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

#[cfg(test)]
mod tests {
    use super::*;
    use polytype::Type;

    use crate::ast::{Span, TypeName};
    use crate::egir::types::EGraph;
    use crate::ssa::types::EntryInput;
    use crate::LookupMap;

    #[test]
    fn resource_access_merge_is_monotone() {
        assert_eq!(
            ResourceAccess::Read.merge(ResourceAccess::Write),
            ResourceAccess::ReadWrite
        );
        assert_eq!(
            ResourceAccess::Write.merge(ResourceAccess::Read),
            ResourceAccess::ReadWrite
        );
        assert_eq!(
            ResourceAccess::Read.merge(ResourceAccess::Read),
            ResourceAccess::Read
        );
    }

    #[test]
    fn resource_flow_topology_orders_chained_prepasses_before_consumer() {
        let phase = |id, dependencies| KernelPhase {
            id: KernelId(id),
            flow_source: Some(CompilerFlowEndpoint::Entry(SemanticEntryId(id))),
            entry_point: format!("phase_{id}"),
            recipe: KernelRecipe::SerialCompute,
            body: None,
            abi: EntryAbiProjection {
                source_entry: Some(SemanticEntryId(id)),
                inputs: Vec::new(),
                output_routes: Vec::new(),
            },
            workgroup_size: (1, 1, 1),
            domain: KernelDomain::Fixed { x: 1, y: 1, z: 1 },
            domain_selection: DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
            resources: Vec::new(),
            dependencies,
        };
        let ordered = topologically_order_phases(vec![
            phase(1, vec![]),
            phase(0, vec![KernelId(1), KernelId(2)]),
            phase(2, vec![KernelId(1)]),
        ]);
        assert_eq!(
            ordered.iter().map(|phase| phase.id).collect::<Vec<_>>(),
            vec![KernelId(1), KernelId(2), KernelId(0)]
        );
    }

    #[test]
    fn explicit_serial_domain_is_not_reinterpreted_from_storage_image() {
        let image = BindingRef::new(2, 7);
        let unit = Type::Constructed(TypeName::Unit, vec![]);
        let input = EntryInput {
            name: "image".into(),
            ty: unit.clone(),
            decoration: None,
            size_hint: None,
            storage_binding: None,
            storage_access: None,
            uniform_binding: None,
            push_constant: None,
            texture_binding: None,
            texture_backing: None,
            texture_resource: None,
            storage_image_resource: None,
            sampler_binding: None,
            storage_image_binding: Some((
                image,
                crate::pipeline_descriptor::StorageImageFormat::R32Float,
                crate::interface::StorageAccess::ReadOnly,
                crate::pipeline_descriptor::StorageTextureSize::SameAsWindow,
            )),
            length: None,
        };
        let mut entry = SemanticEntry::new(
            "serial".into(),
            Span::dummy(),
            ExecutionModel::Compute {
                local_size: (1, 1, 1),
            },
            vec![input],
            vec![],
            vec![],
            vec![],
            unit,
            EGraph::new(),
            LookupMap::new(),
        );
        let fixed = KernelDomain::Fixed { x: 1, y: 1, z: 1 };

        let explicit = phase_from_entry(
            KernelId(0),
            Some(SemanticEntryId(0)),
            &entry,
            DomainSelection::Explicit(fixed.clone()),
            KernelRecipe::SerialCompute,
        );
        assert_eq!(explicit.domain, fixed);
        assert_eq!(
            explicit.domain_selection,
            DomainSelection::Explicit(fixed.clone())
        );

        let inferred = phase_from_entry(
            KernelId(1),
            Some(SemanticEntryId(0)),
            &entry,
            DomainSelection::Inferred(fixed.clone()),
            KernelRecipe::SerialCompute,
        );
        assert_eq!(
            inferred.domain,
            KernelDomain::Elements(DispatchLen::StorageImage {
                set: image.set,
                binding: image.binding,
            })
        );
        assert_eq!(inferred.domain_selection, DomainSelection::Inferred(fixed));

        entry.inputs[0].storage_image_binding = None;
        let DomainSelection::Inferred(fallback) = &inferred.domain_selection else {
            unreachable!()
        };
        assert_eq!(
            inferred_domain(&entry, fallback.clone()),
            KernelDomain::Fixed { x: 1, y: 1, z: 1 },
            "reconciliation must recompute from the retained fallback, not the old resolved domain"
        );
    }
}
