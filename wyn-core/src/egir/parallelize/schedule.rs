//! First-class compute schedule produced by EGIR parallelization.
//!
//! A schedule owns the relationship between generated entry points, dispatch
//! domains, resource accesses, and phase dependencies.  The pipeline
//! descriptor is a publication format derived from this graph after every
//! lowering has finished; it is not mutated while an individual lowering is
//! still speculative.

use std::collections::{HashMap, HashSet};

use crate::egir::graph_ops;
use crate::egir::program::EgirEntry;
use crate::egir::types::{EgirSoac, SegExtent, SegPlacement, SegResourceAccessKind, SideEffectKind};
use crate::pipeline_descriptor::{
    Binding, ComputePipeline, ComputeStage, DispatchLen, DispatchSize, Pipeline, PipelineDescriptor,
};
use crate::ssa::types::ExecutionModel;
use crate::BindingRef;

/// A complete module-level compute schedule.
#[derive(Clone, Debug, Default)]
pub struct KernelSchedule {
    pipelines: Vec<ScheduledPipeline>,
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
    pub entry_point: String,
    pub workgroup_size: (u32, u32, u32),
    pub domain: KernelDomain,
    pub resources: Vec<ScheduledResource>,
    /// Phase indices in this pipeline that must complete first.
    pub dependencies: Vec<usize>,
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

/// Conservative resource access for a phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduledResource {
    pub binding: BindingRef,
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

impl KernelSchedule {
    pub fn phases(&self) -> impl Iterator<Item = &KernelPhase> {
        self.pipelines.iter().flat_map(|pipeline| pipeline.phases.iter())
    }

    pub fn contains_entry(&self, entry_point: &str) -> bool {
        self.pipelines
            .iter()
            .any(|pipeline| pipeline.phases.iter().any(|phase| phase.entry_point == entry_point))
    }

    /// Seed a schedule from the source-level descriptor.  At this point every
    /// compute pipeline contains its original entry stage; later lowerings add
    /// phases to this graph without touching the descriptor.
    pub fn seed(descriptor: &PipelineDescriptor, entries: &[EgirEntry]) -> Self {
        let by_name: HashMap<&str, &EgirEntry> = entries.iter().map(|e| (e.name.as_str(), e)).collect();
        let mut pipelines = Vec::new();
        for (pipeline_index, pipeline) in descriptor.pipelines.iter().enumerate() {
            let Pipeline::Compute(compute) = pipeline else {
                continue;
            };
            let phases = compute
                .stages
                .iter()
                .map(|stage| {
                    by_name
                        .get(stage.entry_point.as_str())
                        .map(|entry| phase_from_entry(entry, domain_from_dispatch(&stage.dispatch_size)))
                        .unwrap_or_else(|| KernelPhase {
                            entry_point: stage.entry_point.clone(),
                            workgroup_size: stage.workgroup_size,
                            domain: domain_from_dispatch(&stage.dispatch_size),
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
        Self { pipelines }
    }

    /// Add a generated phase after the current last phase of `parent`'s
    /// pipeline.  Dependencies are explicit even though host publication
    /// currently emits phases in topological order.
    pub fn add_phase_after(&mut self, parent: &str, entry: &EgirEntry, domain: KernelDomain) {
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
        let mut phase = phase_from_entry(entry, domain);
        phase.dependencies = vec![parent_index];
        pipeline.phases.push(phase);
    }

    /// Insert a compiler-generated producer immediately before `consumer` and
    /// make the consumer depend on it. Existing dependency indices are shifted
    /// transactionally with the insertion.
    pub fn add_phase_before(&mut self, consumer: &str, entry: &EgirEntry, domain: KernelDomain) {
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
        for phase in &mut pipeline.phases {
            for dependency in &mut phase.dependencies {
                if *dependency >= consumer_index {
                    *dependency += 1;
                }
            }
        }
        let mut producer = phase_from_entry(entry, domain);
        producer.dependencies = inherited_dependencies;
        pipeline.phases.insert(consumer_index, producer);
        let consumer_phase = &mut pipeline.phases[consumer_index + 1];
        if !consumer_phase.dependencies.contains(&consumer_index) {
            consumer_phase.dependencies.push(consumer_index);
            consumer_phase.dependencies.sort_unstable();
        }
    }

    /// Add an independent sibling kernel to the same host pipeline. This is
    /// used for distinct output domains: source order is retained by the
    /// published phase list, but no data dependency is fabricated.
    pub fn add_sibling(&mut self, parent: &str, entry: &EgirEntry, domain: KernelDomain) {
        let pipeline = self
            .pipelines
            .iter_mut()
            .find(|p| p.phases.iter().any(|phase| phase.entry_point == parent))
            .unwrap_or_else(|| panic!("no scheduled compute pipeline contains parent entry `{parent}`"));
        let phase = phase_from_entry(entry, domain);
        pipeline.phases.push(phase);
    }

    pub fn domain_of(&self, entry_point: &str) -> Option<KernelDomain> {
        self.pipelines
            .iter()
            .flat_map(|pipeline| &pipeline.phases)
            .find(|phase| phase.entry_point == entry_point)
            .map(|phase| phase.domain.clone())
    }

    /// Refresh domains and resource sets after an entry has been rewritten or
    /// cloned. Explicit fixed domains chosen by a scheduler are preserved;
    /// pointwise SegMaps replace placeholders with their concrete domain.
    pub fn reconcile_entries(&mut self, entries: &[EgirEntry]) {
        let by_name: HashMap<&str, &EgirEntry> = entries.iter().map(|e| (e.name.as_str(), e)).collect();
        for pipeline in &mut self.pipelines {
            for phase in &mut pipeline.phases {
                let Some(entry) = by_name.get(phase.entry_point.as_str()) else {
                    continue;
                };
                phase.workgroup_size = entry_workgroup(entry);
                if let Some(domain) = segmented_domain(entry) {
                    phase.domain = domain;
                }
                let filter_phase = entry.graph.skeleton.blocks.iter().any(|(_, block)| {
                    block.side_effects.iter().any(|effect| {
                        matches!(
                            effect.kind,
                            SideEffectKind::Soac(EgirSoac::Filter {
                                phase: crate::egir::types::FilterPhase::Flags
                                    | crate::egir::types::FilterPhase::Scan
                                    | crate::egir::types::FilterPhase::Scatter,
                                ..
                            })
                        )
                    })
                });
                phase.resources = if filter_phase || entry.name.contains("_materialize_shared") {
                    segmented_resources(entry).unwrap_or_else(|| entry_resources(entry))
                } else {
                    entry_resources(entry)
                };
            }
        }
    }

    /// Merge compiler-generated producer/consumer entries connected by a
    /// first-class `StorageRole::Output -> StorageRole::Input` edge. The
    /// resulting phases share one binding table and execute producer-first.
    /// User storage parameters do not create these declarations, so unrelated
    /// source entry points that happen to reuse a slot remain separate.
    pub fn coalesce_compiler_dependencies(&mut self, entries: &[EgirEntry]) {
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
        let producer_len = producer.phases.len();
        for phase in &mut consumer.phases {
            for dependency in &mut phase.dependencies {
                *dependency += producer_len;
            }
        }
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
                        if producer_index < consumer_index
                            && !pipeline.phases[consumer_index].dependencies.contains(&producer_index)
                        {
                            pipeline.phases[consumer_index].dependencies.push(producer_index);
                        }
                    }
                }
            }
        }
    }

    /// Validate graph-local invariants before publishing a host ABI.
    pub fn validate(&self) -> Result<(), String> {
        let mut names = HashSet::new();
        for pipeline in &self.pipelines {
            for (index, phase) in pipeline.phases.iter().enumerate() {
                if !names.insert(phase.entry_point.as_str()) {
                    return Err(format!(
                        "entry `{}` appears in more than one scheduled phase",
                        phase.entry_point
                    ));
                }
                if phase.dependencies.iter().any(|&dependency| dependency >= index) {
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
                    dispatch_size: DispatchSize::Fixed { x: 1, y: 1, z: 1 },
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
    pub fn publish(&self, descriptor: &mut PipelineDescriptor) -> Result<(), String> {
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
                    let Some(&index) = binding_index.get(&resource.binding) else {
                        return Err(format!(
                            "scheduled phase `{}` references unpublished storage {}",
                            phase.entry_point, resource.binding
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
                        KernelDomain::Fixed { x, y, z } => DispatchSize::Fixed { x: *x, y: *y, z: *z },
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

fn phase_from_entry(entry: &EgirEntry, fallback: KernelDomain) -> KernelPhase {
    KernelPhase {
        entry_point: entry.name.clone(),
        workgroup_size: entry_workgroup(entry),
        domain: segmented_domain(entry)
            .or_else(|| storage_image_domain(entry, &fallback))
            .unwrap_or(fallback),
        resources: segmented_resources(entry).unwrap_or_else(|| entry_resources(entry)),
        dependencies: Vec::new(),
    }
}

/// A compute entry with no SOAC-derived domain, no storage-buffer input, and a
/// `#[storage_image]` param runs one thread per texel of the image (the
/// mountains / one_weekend per-pixel pass shape) — the host resolves the size
/// from the bound texture's extent. Only upgrades the single-workgroup
/// placeholder domain; an explicit fixed grid stays as scheduled.
fn storage_image_domain(entry: &EgirEntry, fallback: &KernelDomain) -> Option<KernelDomain> {
    if !matches!(fallback, KernelDomain::Fixed { x: 1, y: 1, z: 1 }) {
        return None;
    }
    if entry.inputs.iter().any(|input| input.storage_binding.is_some()) {
        return None;
    }
    let (binding, ..) = entry.inputs.iter().find_map(|input| input.storage_image_binding)?;
    Some(KernelDomain::Elements(DispatchLen::StorageImage {
        set: binding.set,
        binding: binding.binding,
    }))
}

fn segmented_resources(entry: &EgirEntry) -> Option<Vec<ScheduledResource>> {
    for (_, block) in &entry.graph.skeleton.blocks {
        for side_effect in &block.side_effects {
            if let SideEffectKind::Soac(EgirSoac::Filter {
                phase,
                scratch_out,
                len_out,
                work_buffers: Some(work),
                ..
            }) = &side_effect.kind
            {
                let mut resources = Vec::new();
                let mut push = |binding: BindingRef, access: ResourceAccess| {
                    if let Some(existing) = resources
                        .iter_mut()
                        .find(|resource: &&mut ScheduledResource| resource.binding == binding)
                    {
                        existing.access = existing.access.merge(access);
                    } else {
                        resources.push(ScheduledResource { binding, access });
                    }
                };
                match phase {
                    crate::egir::types::FilterPhase::Flags => {
                        for input in &entry.inputs {
                            if let Some(binding) = input.storage_binding.or(input.uniform_binding) {
                                push(binding, ResourceAccess::Read);
                            }
                        }
                        push(work.flags, ResourceAccess::Write);
                    }
                    crate::egir::types::FilterPhase::Scan => {
                        push(work.flags, ResourceAccess::Read);
                        push(work.offsets, ResourceAccess::Write);
                        if let Some(binding) = len_out {
                            push(*binding, ResourceAccess::Write);
                        }
                    }
                    crate::egir::types::FilterPhase::Scatter => {
                        for input in &entry.inputs {
                            if let Some(binding) = input.storage_binding.or(input.uniform_binding) {
                                push(binding, ResourceAccess::Read);
                            }
                        }
                        push(work.flags, ResourceAccess::Read);
                        push(work.offsets, ResourceAccess::Read);
                        if let Some(binding) = len_out {
                            push(*binding, ResourceAccess::Read);
                        }
                        if let Some(binding) = scratch_out {
                            push(*binding, ResourceAccess::Write);
                        }
                    }
                    crate::egir::types::FilterPhase::Semantic => continue,
                }
                resources.sort_by_key(|resource| (resource.binding.set, resource.binding.binding));
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
                        binding: resource.binding,
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

fn entry_workgroup(entry: &EgirEntry) -> (u32, u32, u32) {
    match entry.execution_model {
        ExecutionModel::Compute { local_size } => local_size,
        _ => (1, 1, 1),
    }
}

fn domain_from_dispatch(dispatch: &DispatchSize) -> KernelDomain {
    match dispatch {
        DispatchSize::Fixed { x, y, z } => KernelDomain::Fixed { x: *x, y: *y, z: *z },
        DispatchSize::DerivedFrom { len, .. } => KernelDomain::Elements(len.clone()),
    }
}

fn segmented_domain(entry: &EgirEntry) -> Option<KernelDomain> {
    for (_, block) in &entry.graph.skeleton.blocks {
        for side_effect in &block.side_effects {
            match &side_effect.kind {
                SideEffectKind::Soac(EgirSoac::Seg {
                    space,
                    placement: SegPlacement::Kernel,
                    ..
                }) => return domain_from_space(space),
                SideEffectKind::Soac(EgirSoac::Filter {
                    space: Some(space),
                    phase: crate::egir::types::FilterPhase::Flags | crate::egir::types::FilterPhase::Scatter,
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

fn entry_resources(entry: &EgirEntry) -> Vec<ScheduledResource> {
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
            let mut seen = HashSet::new();
            let mut stack = side_effect.referenced_nodes().collect::<Vec<_>>();
            while let Some(node) = stack.pop() {
                if !seen.insert(node) {
                    continue;
                }
                if let Some(binding) = graph_ops::extract_storage_view_source(&entry.graph, node) {
                    insert(binding, ResourceAccess::Read);
                }
                stack.extend(entry.graph.nodes[node].children());
            }
        }
    }

    let mut resources: Vec<_> =
        accesses.into_iter().map(|(binding, access)| ScheduledResource { binding, access }).collect();
    resources.sort_by_key(|resource| (resource.binding.set, resource.binding.binding));
    resources
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
}
