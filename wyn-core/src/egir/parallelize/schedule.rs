//! First-class compute schedule produced by EGIR parallelization.
//!
//! A schedule owns the relationship between generated entry points, dispatch
//! domains, resource accesses, and phase dependencies.  The pipeline
//! descriptor is a publication format derived from this graph after every
//! lowering has finished; it is not mutated while an individual lowering is
//! still speculative.

use std::collections::{HashMap, HashSet};

use crate::egir::graph_ops;
use crate::egir::program::{
    EntryPublication, InputSlotId, LogicalResource, OutputSlotId, PhysicalResourceTable, SemanticEntry,
    SemanticEntryId,
};
use crate::egir::types::{
    EgirSoac, FilterOutput, FilterPlan, FilterState, RuntimeFilterLength, SegExtent, SegPlacement,
    SegResourceAccessKind, SideEffectKind,
};
use crate::pipeline_descriptor::{
    Binding, ComputePipeline, ComputeStage, DispatchLen, DispatchSize, Pipeline, PipelineDescriptor,
};
use crate::ssa::types::ExecutionModel;
use crate::{BindingRef, ResourceId};

/// A complete module-level compute schedule.
#[derive(Clone, Debug, Default)]
pub struct KernelPlan {
    pipelines: Vec<ScheduledPipeline>,
    next_kernel_id: u32,
    resource_ids: HashMap<BindingRef, ResourceId>,
    semantic_entries: HashMap<String, SemanticEntryId>,
    semantic_abi: HashMap<SemanticEntryId, SemanticAbi>,
    publications: HashMap<String, EntryPublication>,
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
    pub fn publications(&self) -> Vec<&EntryPublication> {
        self.0.publications_in_order()
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
    pub entry_point: String,
    pub recipe: KernelRecipe,
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
        self.pipelines.iter().flat_map(|pipeline| pipeline.phases.iter())
    }

    pub fn contains_entry(&self, entry_point: &str) -> bool {
        self.pipelines
            .iter()
            .any(|pipeline| pipeline.phases.iter().any(|phase| phase.entry_point == entry_point))
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
        entries: &[SemanticEntry],
        resources: &[LogicalResource],
        descriptor: &PipelineDescriptor,
    ) -> Result<(), String> {
        let resource_ids = resources.iter().map(|resource| resource.id).collect::<HashSet<_>>();
        let physical_entries =
            entries.iter().map(|entry| (entry.name.as_str(), entry)).collect::<HashMap<_, _>>();
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
            let physical_output_count = physical_entries
                .get(phase.entry_point.as_str())
                .map(|entry| entry.outputs.len())
                .unwrap_or(0);
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

        let mut recipes_by_source: HashMap<SemanticEntryId, HashSet<KernelRecipe>> = HashMap::new();
        for phase in self.phases() {
            if let Some(source) = phase.abi.source_entry {
                recipes_by_source.entry(source).or_default().insert(phase.recipe.clone());
            }
        }
        for (source, recipes) in &recipes_by_source {
            let require = |recipe: KernelRecipe, context: &str| {
                if recipes.contains(&recipe) {
                    Ok(())
                } else {
                    Err(format!(
                        "semantic entry {:?} has an incomplete {context} kernel family",
                        source
                    ))
                }
            };
            if recipes.contains(&KernelRecipe::FilterFlags) {
                require(KernelRecipe::FilterScan, "filter")?;
                require(KernelRecipe::ScanBlock, "filter")?;
                require(KernelRecipe::ScanApplyOffsets, "filter")?;
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
        resources: &[LogicalResource],
    ) -> Self {
        let resource_ids = resources
            .iter()
            .map(|resource| (resource.semantic_binding, resource.id))
            .collect::<HashMap<_, _>>();
        let semantic_entries = entries
            .iter()
            .enumerate()
            .map(|(index, entry)| (entry.name.clone(), SemanticEntryId(index as u32)))
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
        let publications = entries
            .iter()
            .map(|entry| (entry.name.clone(), entry.publication()))
            .collect::<HashMap<_, _>>();
        let publication_order = entries.iter().map(|entry| entry.name.clone()).collect();
        let by_name: HashMap<&str, (SemanticEntryId, &SemanticEntry)> = entries
            .iter()
            .enumerate()
            .map(|(index, entry)| (entry.name.as_str(), (SemanticEntryId(index as u32), entry)))
            .collect();
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
                    let selection = domain_selection_from_stage(stage);
                    by_name
                        .get(stage.entry_point.as_str())
                        .map(|(source, entry)| {
                            phase_from_entry(
                                id,
                                Some(*source),
                                entry,
                                selection.clone(),
                                &resource_ids,
                                analyze_source_recipe(entry),
                            )
                        })
                        .unwrap_or_else(|| KernelPhase {
                            id,
                            entry_point: stage.entry_point.clone(),
                            recipe: KernelRecipe::SerialCompute,
                            abi: EntryAbiProjection {
                                source_entry: None,
                                inputs: Vec::new(),
                                output_routes: Vec::new(),
                            },
                            workgroup_size: stage.workgroup_size,
                            domain: domain_from_dispatch(&stage.dispatch_size),
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
        Self {
            pipelines,
            next_kernel_id,
            resource_ids,
            semantic_entries,
            semantic_abi,
            publications,
            publication_order,
        }
    }

    fn record_publication(&mut self, entry: &SemanticEntry) {
        if !self.publications.contains_key(&entry.name) {
            self.publication_order.push(entry.name.clone());
            self.publications.insert(entry.name.clone(), entry.publication());
            return;
        }
        // A source phase may relinquish a host output to a generated combine
        // kernel. Descriptor publication still needs the original host ABI,
        // while compiler declarations added by the physical phase must also
        // be visible. Preserve the seed ABI and monotonically merge generated
        // resource declarations; stage reads/writes remain plan-owned.
        let publication = self.publications.get_mut(&entry.name).expect("checked above");
        for input in &entry.inputs {
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
        for output in &entry.outputs {
            if output.storage_binding.is_none() && output.target.is_none() {
                continue;
            }
            if !publication.outputs.iter().any(|existing| {
                existing.storage_binding == output.storage_binding && existing.target == output.target
            }) {
                publication.outputs.push(output.clone());
            }
        }
        for declaration in &entry.storage_bindings {
            if !publication.storage_bindings.iter().any(|existing| {
                existing.binding == declaration.binding && existing.role == declaration.role
            }) {
                publication.storage_bindings.push(declaration.clone());
            }
        }
    }

    fn publications_in_order(&self) -> Vec<&EntryPublication> {
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

    /// Add a generated phase after the current last phase of `parent`'s
    /// pipeline.  Dependencies are explicit even though host publication
    /// currently emits phases in topological order.
    pub fn add_phase_after(
        &mut self,
        parent: &str,
        entry: &SemanticEntry,
        domain: DomainSelection,
        recipe: KernelRecipe,
    ) {
        self.record_publication(entry);
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
        let source_entry = self
            .semantic_entries
            .get(&entry.name)
            .copied()
            .or(pipeline.phases[parent_index].abi.source_entry);
        let id = KernelId(self.next_kernel_id);
        self.next_kernel_id += 1;
        let mut phase = phase_from_entry(id, source_entry, entry, domain, &self.resource_ids, recipe);
        phase.dependencies = vec![parent_id];
        pipeline.phases.push(phase);
    }

    /// Insert a compiler-generated producer immediately before `consumer` and
    /// make the consumer depend on it. Existing dependency indices are shifted
    /// transactionally with the insertion.
    pub fn add_phase_before(
        &mut self,
        consumer: &str,
        entry: &SemanticEntry,
        domain: DomainSelection,
        recipe: KernelRecipe,
    ) {
        self.record_publication(entry);
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
        let consumer_source = self
            .semantic_entries
            .get(&entry.name)
            .copied()
            .or(pipeline.phases[consumer_index].abi.source_entry);
        let id = KernelId(self.next_kernel_id);
        self.next_kernel_id += 1;
        let mut producer = phase_from_entry(id, consumer_source, entry, domain, &self.resource_ids, recipe);
        producer.dependencies = inherited_dependencies;
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
        entry: &SemanticEntry,
        domain: DomainSelection,
        recipe: KernelRecipe,
    ) {
        self.record_publication(entry);
        let pipeline = self
            .pipelines
            .iter_mut()
            .find(|p| p.phases.iter().any(|phase| phase.entry_point == parent))
            .unwrap_or_else(|| panic!("no scheduled compute pipeline contains parent entry `{parent}`"));
        let source_entry = self.semantic_entries.get(&entry.name).copied().or_else(|| {
            pipeline
                .phases
                .iter()
                .find(|phase| phase.entry_point == parent)
                .and_then(|phase| phase.abi.source_entry)
        });
        let id = KernelId(self.next_kernel_id);
        self.next_kernel_id += 1;
        let phase = phase_from_entry(id, source_entry, entry, domain, &self.resource_ids, recipe);
        pipeline.phases.push(phase);
    }

    pub fn domain_of(&self, entry_point: &str) -> Option<KernelDomain> {
        self.pipelines
            .iter()
            .flat_map(|pipeline| &pipeline.phases)
            .find(|phase| phase.entry_point == entry_point)
            .map(|phase| phase.domain.clone())
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
    pub fn commit_kernel(&mut self, entry: &SemanticEntry, recipe: KernelRecipe) {
        self.record_publication(entry);
        for pipeline in &mut self.pipelines {
            for phase in &mut pipeline.phases {
                if phase.entry_point != entry.name {
                    continue;
                }
                phase.workgroup_size = entry_workgroup(entry);
                if let DomainSelection::Inferred(fallback) = &phase.domain_selection {
                    phase.domain = inferred_domain(entry, fallback.clone());
                }
                let filter_phase = entry.graph.skeleton.blocks.iter().any(|(_, block)| {
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
                let semantic_resources = filter_phase
                    || entry.origin == crate::interface::EntryOrigin::MultiConsumerMaterialization;
                phase.resources = if semantic_resources {
                    segmented_resources(entry, &self.resource_ids)
                        .unwrap_or_else(|| entry_resources(entry, &self.resource_ids))
                } else {
                    entry_resources(entry, &self.resource_ids)
                };
                phase.recipe = recipe;
                phase.abi.inputs = (0..entry.inputs.len()).map(InputSlotId).collect();
                phase.abi.output_routes = entry_output_projection(entry);
                return;
            }
        }
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

    /// Merge compiler-generated producer/consumer entries connected by a
    /// first-class `StorageRole::Output -> StorageRole::Input` edge. The
    /// resulting phases share one binding table and execute producer-first.
    /// User storage parameters do not create these declarations, so unrelated
    /// source entry points that happen to reuse a slot remain separate.
    pub fn coalesce_compiler_dependencies(&mut self, entries: &[SemanticEntry]) {
        let mut producers: HashMap<BindingRef, Vec<&str>> = HashMap::new();
        let mut consumers: HashMap<BindingRef, Vec<&str>> = HashMap::new();
        for entry in entries {
            for declaration in &entry.storage_bindings {
                match declaration.role {
                    crate::interface::StorageRole::Output => {
                        producers.entry(declaration.binding).or_default().push(&entry.name)
                    }
                    crate::interface::StorageRole::Input => {
                        consumers.entry(declaration.binding).or_default().push(&entry.name)
                    }
                    crate::interface::StorageRole::Intermediate => {}
                }
            }
        }

        loop {
            let mut merge = None;
            for (binding, producer_entries) in &producers {
                let Some(consumer_entries) = consumers.get(binding) else {
                    continue;
                };
                let producer_pipelines = schedule_indices_for_entries(&self.pipelines, producer_entries);
                let consumer_pipelines = schedule_indices_for_entries(&self.pipelines, consumer_entries);
                if producer_pipelines.len() == 1
                    && consumer_pipelines.len() == 1
                    && producer_pipelines[0] != consumer_pipelines[0]
                {
                    merge = Some((producer_pipelines[0], consumer_pipelines[0]));
                    break;
                }
            }
            let Some((producer, consumer)) = merge else {
                break;
            };
            self.merge_pipeline_into(producer, consumer);
        }
        self.add_compiler_resource_dependencies(&producers, &consumers);
    }

    fn merge_pipeline_into(&mut self, producer_index: usize, consumer_index: usize) {
        let producer = self.pipelines.remove(producer_index);
        let consumer_index =
            if producer_index < consumer_index { consumer_index - 1 } else { consumer_index };
        let consumer = &mut self.pipelines[consumer_index];
        let mut phases = producer.phases;
        phases.append(&mut consumer.phases);
        consumer.phases = phases;
        consumer.order = consumer.order.min(producer.order);
        merge_bindings(&mut consumer.template.bindings, producer.template.bindings);
        for feedback in producer.template.feedback {
            if !consumer.template.feedback.contains(&feedback) {
                consumer.template.feedback.push(feedback);
            }
        }
    }

    fn add_compiler_resource_dependencies(
        &mut self,
        producers: &HashMap<BindingRef, Vec<&str>>,
        consumers: &HashMap<BindingRef, Vec<&str>>,
    ) {
        for (binding, producer_entries) in producers {
            let Some(consumer_entries) = consumers.get(binding) else {
                continue;
            };
            for pipeline in &mut self.pipelines {
                let producer_phases: Vec<usize> = pipeline
                    .phases
                    .iter()
                    .enumerate()
                    .filter_map(|(index, phase)| {
                        producer_entries.iter().any(|entry| phase.entry_point == *entry).then_some(index)
                    })
                    .collect();
                let consumer_phases: Vec<usize> = pipeline
                    .phases
                    .iter()
                    .enumerate()
                    .filter_map(|(index, phase)| {
                        consumer_entries.iter().any(|entry| phase.entry_point == *entry).then_some(index)
                    })
                    .collect();
                for consumer_index in consumer_phases {
                    for &producer_index in &producer_phases {
                        // A merged producer pipeline is always placed before
                        // its consumer. If both entries were already in one
                        // pipeline, retain only a genuine forward dependency.
                        let producer_id = pipeline.phases[producer_index].id;
                        if producer_index < consumer_index
                            && !pipeline.phases[consumer_index].dependencies.contains(&producer_id)
                        {
                            pipeline.phases[consumer_index].dependencies.push(producer_id);
                        }
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

fn phase_from_entry(
    id: KernelId,
    source_entry: Option<SemanticEntryId>,
    entry: &SemanticEntry,
    selection: DomainSelection,
    resource_ids: &HashMap<BindingRef, ResourceId>,
    recipe: KernelRecipe,
) -> KernelPhase {
    let domain = match &selection {
        DomainSelection::Inferred(fallback) => inferred_domain(entry, fallback.clone()),
        DomainSelection::Explicit(domain) => domain.clone(),
    };
    KernelPhase {
        id,
        entry_point: entry.name.clone(),
        recipe,
        abi: EntryAbiProjection {
            source_entry,
            inputs: (0..entry.inputs.len()).map(InputSlotId).collect(),
            output_routes: entry_output_projection(entry),
        },
        workgroup_size: entry_workgroup(entry),
        domain,
        domain_selection: selection,
        resources: segmented_resources(entry, resource_ids)
            .unwrap_or_else(|| entry_resources(entry, resource_ids)),
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

fn analyze_source_recipe(entry: &SemanticEntry) -> KernelRecipe {
    use crate::interface::EntryOrigin;

    match entry.origin {
        EntryOrigin::ScalarPrepass => KernelRecipe::ScalarPrepass,
        EntryOrigin::GatherPrepass => KernelRecipe::GatherPrepass,
        EntryOrigin::MultiConsumerMaterialization => KernelRecipe::MultiConsumerMaterialization,
        EntryOrigin::ReducePhase2 => KernelRecipe::ReduceCombine,
        EntryOrigin::ScanPhase2 => KernelRecipe::ScanBlock,
        EntryOrigin::ScanPhase3 => KernelRecipe::ScanApplyOffsets,
        EntryOrigin::OutputDomainSplit => KernelRecipe::OutputDomainProjection,
        EntryOrigin::RuntimeFilter => filter_recipe(entry),
        EntryOrigin::Source => source_recipe(entry),
    }
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

fn filter_recipe(entry: &SemanticEntry) -> KernelRecipe {
    for (_, block) in &entry.graph.skeleton.blocks {
        for effect in &block.side_effects {
            if let SideEffectKind::Soac(EgirSoac::Filter {
                state: FilterState::Scheduled { plan, .. },
                ..
            }) = &effect.kind
            {
                return match plan {
                    FilterPlan::Flags(_) => KernelRecipe::FilterFlags,
                    FilterPlan::Scan(_) => KernelRecipe::FilterScan,
                    FilterPlan::Scatter(_) => KernelRecipe::FilterScatter,
                    FilterPlan::Serial => KernelRecipe::SerialCompute,
                };
            }
        }
    }
    // Filter phase-2 and phase-3 helpers are represented as scan recipes by
    // their dedicated origins when possible. A synthetic RuntimeFilter helper
    // with no Filter node is the combine phase.
    KernelRecipe::FilterCombine
}

fn inferred_domain(entry: &SemanticEntry, fallback: KernelDomain) -> KernelDomain {
    segmented_domain(entry).or_else(|| storage_image_domain(entry, &fallback)).unwrap_or(fallback)
}

/// A compute entry with no SOAC-derived domain and a `#[storage_image]` param
/// runs one thread per texel of the image (the mountains / one_weekend
/// per-pixel pass shape) — the host resolves the size from the bound texture's
/// extent. Incidental storage-buffer inputs (e.g. mountains' keyboard buffer)
/// don't opt out; the image is the domain. Only upgrades the single-workgroup
/// placeholder domain; an explicit fixed grid stays as scheduled.
///
fn storage_image_domain(entry: &SemanticEntry, fallback: &KernelDomain) -> Option<KernelDomain> {
    if !matches!(fallback, KernelDomain::Fixed { x: 1, y: 1, z: 1 }) {
        return None;
    }
    let (binding, ..) = entry.inputs.iter().find_map(|input| input.storage_image_binding)?;
    Some(KernelDomain::Elements(DispatchLen::StorageImage {
        set: binding.set,
        binding: binding.binding,
    }))
}

fn segmented_resources(
    entry: &SemanticEntry,
    resource_ids: &HashMap<BindingRef, ResourceId>,
) -> Option<Vec<ScheduledResource>> {
    for (_, block) in &entry.graph.skeleton.blocks {
        for side_effect in &block.side_effects {
            if let SideEffectKind::Soac(EgirSoac::Filter {
                state: FilterState::Scheduled { plan, .. },
                output,
                ..
            }) = &side_effect.kind
            {
                let mut resources = Vec::new();
                let mut push = |binding: BindingRef, access: ResourceAccess| {
                    let resource = resource_id_for(resource_ids, binding);
                    if let Some(existing) = resources
                        .iter_mut()
                        .find(|candidate: &&mut ScheduledResource| candidate.resource == resource)
                    {
                        existing.access = existing.access.merge(access);
                    } else {
                        resources.push(ScheduledResource { resource, access });
                    }
                };
                match plan {
                    FilterPlan::Flags(work) => {
                        for input in &entry.inputs {
                            if let Some(binding) = input.storage_binding.or(input.uniform_binding) {
                                push(binding, ResourceAccess::Read);
                            }
                        }
                        push(work.flags, ResourceAccess::Write);
                    }
                    FilterPlan::Scan(work) => {
                        push(work.flags, ResourceAccess::Read);
                        push(work.offsets, ResourceAccess::Write);
                        push(work.block_sums, ResourceAccess::Write);
                    }
                    FilterPlan::Scatter(work) => {
                        for input in &entry.inputs {
                            if let Some(binding) = input.storage_binding.or(input.uniform_binding) {
                                push(binding, ResourceAccess::Read);
                            }
                        }
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
                        resource: resource_id_for(resource_ids, resource.binding),
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

fn domain_from_dispatch(dispatch: &DispatchSize) -> KernelDomain {
    match dispatch {
        DispatchSize::Fixed { x, y, z, .. } => KernelDomain::Fixed { x: *x, y: *y, z: *z },
        DispatchSize::DerivedFrom { len, .. } => KernelDomain::Elements(len.clone()),
    }
}

fn domain_selection_from_stage(stage: &ComputeStage) -> DomainSelection {
    let domain = domain_from_dispatch(&stage.dispatch_size);
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
    for (_, block) in &entry.graph.skeleton.blocks {
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
            binding, elem_bytes, ..
        }] => Some(KernelDomain::Elements(DispatchLen::InputBinding {
            set: binding.set,
            binding: binding.binding,
            elem_bytes: *elem_bytes,
        })),
        _ => None,
    }
}

fn entry_resources(
    entry: &SemanticEntry,
    resource_ids: &HashMap<BindingRef, ResourceId>,
) -> Vec<ScheduledResource> {
    let mut accesses: HashMap<BindingRef, ResourceAccess> = HashMap::new();
    let mut insert = |binding: BindingRef, access: ResourceAccess| {
        accesses.entry(binding).and_modify(|old| *old = old.merge(access)).or_insert(access);
    };

    for input in &entry.inputs {
        if let Some(binding) = input.storage_binding {
            let access = match input.storage_access {
                Some(crate::interface::StorageAccess::WriteOnly) => ResourceAccess::Write,
                Some(crate::interface::StorageAccess::ReadWrite) => ResourceAccess::ReadWrite,
                _ => ResourceAccess::Read,
            };
            insert(binding, access);
        }
        if let Some(binding) = input.uniform_binding {
            insert(binding, ResourceAccess::Read);
        }
    }
    for output in &entry.outputs {
        if let Some(binding) = output.storage_binding {
            insert(binding, ResourceAccess::Write);
        }
    }
    for declaration in &entry.storage_bindings {
        let access = match declaration.role {
            crate::interface::StorageRole::Input => ResourceAccess::Read,
            crate::interface::StorageRole::Output => ResourceAccess::Write,
            crate::interface::StorageRole::Intermediate => ResourceAccess::ReadWrite,
        };
        insert(declaration.binding, access);
    }

    // A storage view reachable from an effect operand is conservatively a
    // read. Output/intermediate metadata above upgrades it when it is written.
    for (_, block) in &entry.graph.skeleton.blocks {
        for side_effect in &block.side_effects {
            wyn_graph::for_each_reachable(
                side_effect.referenced_nodes(),
                wyn_graph::WalkOrder::DepthFirst,
                |node, out| out.extend(entry.graph.nodes[node].children()),
                |node| {
                    if let Some(binding) = graph_ops::extract_storage_view_source(&entry.graph, node) {
                        insert(binding, ResourceAccess::Read);
                    }
                },
            );
        }
    }

    let mut resources: Vec<_> = accesses
        .into_iter()
        .map(|(binding, access)| ScheduledResource {
            resource: resource_id_for(resource_ids, binding),
            access,
        })
        .collect();
    resources.sort_by_key(|resource| resource.resource);
    resources
}

fn resource_id_for(resource_ids: &HashMap<BindingRef, ResourceId>, binding: BindingRef) -> ResourceId {
    *resource_ids
        .get(&binding)
        .unwrap_or_else(|| panic!("storage {binding} is absent from the logical-resource manifest"))
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

fn schedule_indices_for_entries(pipelines: &[ScheduledPipeline], entries: &[&str]) -> Vec<usize> {
    pipelines
        .iter()
        .enumerate()
        .filter_map(|(index, pipeline)| {
            pipeline
                .phases
                .iter()
                .any(|phase| entries.iter().any(|entry| phase.entry_point == *entry))
                .then_some(index)
        })
        .collect()
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
            crate::interface::EntryOrigin::Source,
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
            &HashMap::new(),
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
            &HashMap::new(),
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
