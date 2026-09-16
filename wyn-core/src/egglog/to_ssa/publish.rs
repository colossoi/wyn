//! Publish the finalized resource/dispatch readout through the shared ABI.

use super::*;
use crate::egglog::{DispatchId, EntryId as SourceEntryId};
use crate::egir::parallelize::{KernelDomain, KernelId, PhysicalKernel, PhysicalKernelGraph};
use crate::egir::program::EntryPublication;
use crate::egir::publish::PipelineDescriptorPublish;
use crate::interface::{EntryInputKind, StorageAccess};
use crate::pipeline_descriptor::{self as pd, ComputePipeline, ComputeStage, Pipeline, PipelineDescriptor};
use crate::ResourceAccess;

impl Compiler<'_> {
    pub(super) fn publish(
        &self,
        entries: &mut [ssa::types::EntryPoint],
        roots: &[(BlockId, SourceEntryId, [u32; 3], bool)],
    ) -> Result<(PipelineDescriptor, PhysicalKernelGraph), OptimizeError> {
        let _timing = super::super::timing::span("publish pipeline ABI");
        let mut pipeline = PipelineDescriptor::default();
        let mut associations = vec![];
        let entry_indices: BTreeMap<_, _> = entries.iter().enumerate().map(|(i, e)| (e.id, i)).collect();
        let dispatches: BTreeMap<_, _> =
            self.data.dispatches.iter().map(|(&id, d)| (d.kernel, (id, d))).collect();
        let order = wyn_graph::topo_sort_by_dependencies(self.data.dispatches.ids(), |id, out| {
            out.extend(self.data.dispatches[id].dependencies.iter().copied());
        })
        .map_err(|e| error(e.to_string()))?;
        let mut by_owner: BTreeMap<SourceEntryId, Vec<BlockId>> = BTreeMap::new();
        for &id in &order {
            let d = &self.data.dispatches[id];
            by_owner.entry(d.owner).or_default().push(d.kernel);
        }
        for &(root, owner, _, finish) in roots {
            if finish {
                by_owner.entry(owner).or_default().push(root);
            }
        }
        let by_binding: BTreeMap<_, _> = self.bindings.iter().map(|(&id, b)| (b.binding, id)).collect();
        let input_resources: BTreeMap<_, _> = self
            .data
            .buffers
            .iter()
            .filter_map(|(&id, b)| {
                if !matches!(b.storage, Storage::External(_)) {
                    return None;
                }
                self.value_binding(&Value::Buffer(id)).map(|(binding, _)| (binding, id))
            })
            .collect();
        let output_buffers: BTreeSet<_> = self.data.outputs.values().filter_map(|o| o.buffer).collect();
        let mut physical = vec![];
        let mut kernel_ids = BTreeMap::<DispatchId, KernelId>::new();
        for (owner, blocks) in by_owner {
            let pipeline_index = pipeline.pipelines.len();
            let source_name = &self.data.entries[owner].declaration.name;
            let mut stages = vec![];
            let mut ids = vec![];
            let mut previous = vec![];
            for root in blocks {
                let entry = &mut entries[entry_indices[&EntryId::from(root.as_u32())]];
                let size = match entry.execution_model {
                    crate::flow::ExecutionModel::Compute { local_size } => local_size,
                    _ => unreachable!(),
                };
                let dispatch = dispatches.get(&root).copied();
                let dispatch_size = dispatch.map_or(
                    pd::DispatchSize::Fixed {
                        x: 1,
                        y: 1,
                        z: 1,
                        explicit: true,
                    },
                    |(_, d)| self.dispatch_size(d),
                );
                let mut accesses = BTreeMap::<BindingRef, ResourceAccess>::new();
                let mut resources = BTreeMap::<BufferId, ResourceAccess>::new();
                if let Some((_, d)) = dispatch {
                    for (&id, access) in d
                        .reads
                        .iter()
                        .map(|i| (i, ResourceAccess::Read))
                        .chain(d.writes.iter().map(|i| (i, ResourceAccess::Write)))
                    {
                        resources.entry(id).and_modify(|a| *a = a.merge(access)).or_insert(access);
                        if let Some((binding, _)) = self.value_binding(&Value::Buffer(id)) {
                            accesses.entry(binding).and_modify(|a| *a = a.merge(access)).or_insert(access);
                        }
                    }
                }
                for declaration in &mut entry.storage_bindings {
                    let buffer = by_binding
                        .get(&declaration.binding)
                        .copied()
                        .ok_or_else(|| error("unplanned storage binding"))?;
                    let access = accesses.get(&declaration.binding).copied().unwrap_or_else(|| {
                        if output_buffers.contains(&buffer) && dispatch.is_none() {
                            ResourceAccess::Write
                        } else {
                            ResourceAccess::Read
                        }
                    });
                    accesses.insert(declaration.binding, access);
                    resources.entry(buffer).or_insert(access);
                    declaration.role = match access {
                        ResourceAccess::Read => StorageRole::Input,
                        ResourceAccess::Write => StorageRole::Output,
                        ResourceAccess::ReadWrite => StorageRole::InputOutput,
                    };
                }
                for input in &mut entry.inputs {
                    if let EntryInputKind::Storage {
                        exposure: crate::interface::BindingExposure::Host(binding),
                        access,
                        ..
                    } = &mut input.kind
                    {
                        if let Some(planned) = accesses.get(binding) {
                            *access = match planned {
                                ResourceAccess::Read => StorageAccess::ReadOnly,
                                ResourceAccess::Write => StorageAccess::WriteOnly,
                                ResourceAccess::ReadWrite => StorageAccess::ReadWrite,
                            };
                        }
                        if let Some(&buffer) = input_resources.get(binding) {
                            resources.entry(buffer).or_insert(ResourceAccess::from(*access));
                        }
                    }
                }
                entry.stage_descriptor_storage_accesses = accesses.iter().map(|(&b, &a)| (b, a)).collect();
                let id = KernelId::from(physical.len() as u32);
                let dependencies = if let Some((did, d)) = dispatch {
                    let deps = d
                        .dependencies
                        .iter()
                        .map(|dep| {
                            kernel_ids
                                .get(dep)
                                .copied()
                                .ok_or_else(|| error("dispatch dependency crosses entry ownership"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    kernel_ids.insert(did, id);
                    deps
                } else {
                    previous.clone()
                };
                let domain = match &dispatch_size {
                    pd::DispatchSize::Fixed { x, y, z, .. } => KernelDomain::Fixed { x: *x, y: *y, z: *z },
                    pd::DispatchSize::DerivedFrom { len, .. } => KernelDomain::Elements(len.clone()),
                };
                physical.push(PhysicalKernel {
                    id,
                    entry: entry.id,
                    entry_point: entry.name.clone(),
                    label: self.data.blocks[root].interface.as_ref().unwrap().name.clone(),
                    source_entry: Some(EntryId::from(owner.as_u32())),
                    output_routes: vec![],
                    workgroup_size: size,
                    domain,
                    resources: resources
                        .into_iter()
                        .map(|(id, access)| crate::egir::ir::SegResourceAccess {
                            resource: crate::ResourceId::from_egglog_buffer(id.as_u32()),
                            access,
                        })
                        .collect(),
                    dependencies,
                });
                previous.push(id);
                stages.push(ComputeStage {
                    entry_point: entry.name.clone(),
                    owner: source_name.clone(),
                    workgroup_size: size,
                    dispatch_size,
                    uses: Default::default(),
                });
                ids.push(entry.id);
            }
            pipeline.pipelines.push(Pipeline::Compute(ComputePipeline {
                bindings: vec![],
                stages,
                default_total_threads: None,
            }));
            associations.push(ids);
            for output in self.data.outputs.values().filter(|o| o.entry == owner) {
                // TODO: extend SourceResultBinding with view offset/live-length
                // metadata. It currently publishes backing/capacity only; a
                // compacted result's count remains explicit in the egglog plan.
                let id = output.buffer.ok_or_else(|| {
                    error(format!(
                        "output {} of {source_name} has no planned backing",
                        output.index
                    ))
                })?;
                let (binding, _) = self
                    .value_binding(&Value::Buffer(id))
                    .ok_or_else(|| error("output has no physical binding"))?;
                pipeline.source_results.push(pd::SourceResultBinding {
                    entry: source_name.clone(),
                    result: output.index,
                    pipeline_index,
                    set: binding.set,
                    binding: binding.binding,
                });
            }
        }
        let publications: Vec<_> = entries
            .iter()
            .map(|e| EntryPublication {
                id: e.id,
                name: e.name.clone(),
                execution_model: e.execution_model.clone(),
                inputs: e.inputs.clone(),
                outputs: vec![],
                storage_bindings: e.storage_bindings.clone(),
            })
            .collect();
        let publications: Vec<_> = publications.iter().collect();
        pipeline
            .publish_implicit_bindings(&publications, &associations)
            .map_err(|e| error(e.to_string()))?;
        // The shared publisher deliberately leaves storage traffic to the
        // finalized plan; it adds only non-storage interfaces itself.
        for (index, p) in pipeline.pipelines.iter_mut().enumerate() {
            let Pipeline::Compute(p) = p else { unreachable!() };
            let slots: BTreeMap<_, _> = p
                .bindings
                .iter()
                .enumerate()
                .filter_map(|(i, b)| match b {
                    pd::Binding::StorageBuffer { set, binding, .. } => {
                        Some((BindingRef::new(*set, *binding), i))
                    }
                    _ => None,
                })
                .collect();
            for (stage, id) in p.stages.iter_mut().zip(&associations[index]) {
                let entry = &entries[entry_indices[id]];
                let mut accesses = entry.stage_storage_accesses();
                accesses.extend(entry.stage_descriptor_storage_accesses.iter().map(|(&b, &a)| (b, a)));
                for (binding, access) in accesses {
                    if let Some(&slot) = slots.get(&binding) {
                        stage.uses.record(
                            slot,
                            match access {
                                ResourceAccess::Read => pd::Access::ReadOnly,
                                ResourceAccess::Write => pd::Access::WriteOnly,
                                ResourceAccess::ReadWrite => pd::Access::ReadWrite,
                            },
                        );
                    }
                }
            }
        }
        pipeline.publish_stage_binding_uses(&publications, &associations);
        for (index, p) in pipeline.pipelines.iter_mut().enumerate() {
            let Pipeline::Compute(p) = p else { unreachable!() };
            let outputs: BTreeSet<_> = pipeline
                .source_results
                .iter()
                .filter(|o| o.pipeline_index == index)
                .map(|o| (o.set, o.binding))
                .collect();
            let mut union = crate::LookupMap::new();
            for b in &mut p.bindings {
                if let pd::Binding::StorageBuffer {
                    set,
                    binding,
                    usage,
                    access,
                    ..
                } = b
                {
                    if outputs.contains(&(*set, *binding)) && *usage == pd::BufferUsage::Intermediate {
                        *usage = pd::BufferUsage::Output;
                    }
                    let a = match access {
                        pd::Access::ReadOnly => ResourceAccess::Read,
                        pd::Access::WriteOnly => ResourceAccess::Write,
                        pd::Access::ReadWrite => ResourceAccess::ReadWrite,
                    };
                    union.insert(BindingRef::new(*set, *binding), a);
                }
            }
            for id in &associations[index] {
                entries[entry_indices[id]].pipeline_storage_accesses = union.clone();
            }
        }
        pipeline.rebuild_frame_graph();
        Ok((
            pipeline,
            PhysicalKernelGraph::from_ordered(physical).map_err(error)?,
        ))
    }
}
