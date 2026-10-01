//! Source ABI declarations and publication for directly emitted entry bodies.
use super::{error, Compiler, OptimizeError};
use crate::egglog::to_ssa::host;
use crate::egglog::to_ssa::sizes;
use crate::flow::ExecutionModel;
use crate::host::Access;
use crate::host::Binding;
use crate::host::BufferUsage;
use crate::host::DispatchLen;
use crate::host::ResultKind;
use crate::host::ScalarSource;
use crate::host::SourceResultBinding;
use crate::host::{
    ComputePipeline, ComputeStage, DispatchSize, GraphicsPipeline, GraphicsStage, ModuleInterface,
    Pipeline, ShaderStage,
};
use crate::interface::publish::ModuleInterfacePublish;
use crate::interface::results::result_layout;
use crate::interface::EntryPublication;
use crate::interface::SourceResult;
use crate::kernel_graph::{KernelDomain, PhysicalKernel, PhysicalKernelGraph};
use crate::ssa::types::EntryPoint;
use crate::tlc::extract_lambda_params_ref;
use crate::tlc::DefMeta;
use crate::types::{strip_existentials, Type, TypeName};
use crate::BindingRef;
use crate::LookupSet;
use crate::ResourceAccess;
use crate::ResourceId;
use crate::ResourceUse;
use crate::{EntryId, LookupMap};
use wyn_base::IdSource;

pub(super) fn publish(
    compiler: &Compiler<'_, '_>,
    entries: &mut [EntryPoint],
) -> Result<(ModuleInterface, PhysicalKernelGraph), OptimizeError> {
    let source = compiler.program.source;
    let declarations: LookupMap<_, _> = source
        .defs
        .iter()
        .filter_map(|definition| match &definition.meta {
            DefMeta::EntryPoint(entry) => Some((definition.name, &entry.declaration)),
            _ => None,
        })
        .collect();
    let mut module = ModuleInterface::default();
    let mut associations: Vec<Vec<EntryId>> = Vec::new();
    let mut compute = LookupMap::default();
    let mut graphics = LookupMap::default();
    let mut kernels = Vec::new();
    let mut ids = IdSource::new();
    for entry in entries.iter() {
        let Some((owner, stage)) = compiler.entry_origins.get(&entry.id) else {
            return Err(error("entry provenance missing"));
        };
        let Some(&declaration) = declarations.get(owner) else {
            return Err(error("entry declaration missing"));
        };
        match entry.execution_model {
            ExecutionModel::Compute { local_size } => {
                let index = *compute.entry(*owner).or_insert_with(|| {
                    let index = module.pipelines.len();
                    module.pipelines.push(Pipeline::Compute(ComputePipeline {
                        bindings: vec![],
                        stages: vec![],
                        default_total_threads: entry.inputs.iter().find_map(|input| input.size_hint),
                    }));
                    associations.push(vec![]);
                    index
                });
                let dispatch = if let Some(stage) = stage {
                    sizes::dispatch(compiler, stage)?
                } else {
                    let grid = compiler
                        .plan
                        .entry_grid(
                            *owner,
                            None,
                            declaration.compute_dispatch.map(|grid| (grid.x, grid.y, grid.z)),
                        )
                        .unwrap_or((1, 1, 1));
                    DispatchSize::Fixed {
                        explicit: declaration.compute_dispatch.is_some(),
                        x: grid.0,
                        y: grid.1,
                        z: grid.2,
                    }
                };
                let Pipeline::Compute(pipeline) = &mut module.pipelines[index] else {
                    return Err(error("compute pipeline identity mismatch"));
                };
                pipeline.stages.push(ComputeStage {
                    dependencies: vec![],
                    entry_point: entry.name.clone(),
                    owner: declaration
                        .source_entry
                        .as_ref()
                        .map_or(&declaration.name, |entry| &entry.name)
                        .clone(),
                    workgroup_size: local_size,
                    dispatch_size: dispatch.clone(),
                    uses: Default::default(),
                });
                associations[index].push(entry.id);
                kernels.push(PhysicalKernel {
                    id: ids.next_id(),
                    entry: entry.id,
                    entry_point: entry.name.clone(),
                    label: entry.name.clone(),
                    source_entry: Some(entry.id),
                    output_routes: vec![],
                    workgroup_size: local_size,
                    domain: match dispatch {
                        DispatchSize::Fixed { x, y, z, .. } => KernelDomain::Fixed { x, y, z },
                        DispatchSize::DerivedFrom { len, workgroup_size }
                            if workgroup_size == local_size.0 =>
                        {
                            KernelDomain::Elements(len)
                        }
                        DispatchSize::DerivedFrom { len, workgroup_size }
                            if workgroup_size % local_size.0 == 0 =>
                        {
                            KernelDomain::ChunkedElements {
                                len,
                                chunk_size: workgroup_size / local_size.0,
                            }
                        }
                        _ => return Err(error("launch divisor must cover whole workgroups")),
                    },
                    resources: vec![],
                    dependencies: vec![],
                });
            }
            ExecutionModel::Vertex | ExecutionModel::Fragment => {
                let Some(group) = &declaration.graphics_group else {
                    return Err(error("graphics entry has no operation identity"));
                };
                let index = *graphics.entry((group.root, group.operation)).or_insert_with(|| {
                    let index = module.pipelines.len();
                    module.pipelines.push(Pipeline::Graphics(GraphicsPipeline {
                        source_operation: Some(group.operation),
                        invocation: group.invocation.clone(),
                        stages: vec![],
                        bindings: vec![],
                        vertex_inputs: vec![],
                        fragment_outputs: vec![],
                    }));
                    associations.push(vec![]);
                    index
                });
                let Pipeline::Graphics(pipeline) = &mut module.pipelines[index] else {
                    return Err(error("graphics pipeline identity mismatch"));
                };
                let Some(owner) = source.symbols.get(group.root) else {
                    return Err(error("graphics owner missing"));
                };
                pipeline.stages.push(GraphicsStage {
                    entry_point: entry.name.clone(),
                    owner: owner.clone(),
                    stage: if matches!(entry.execution_model, ExecutionModel::Vertex) {
                        ShaderStage::Vertex
                    } else {
                        ShaderStage::Fragment
                    },
                    uses: Default::default(),
                });
                associations[index].push(entry.id);
            }
        }
    }
    let publications: Vec<_> = entries
        .iter()
        .map(|entry| EntryPublication {
            id: entry.id,
            name: entry.name.clone(),
            execution_model: entry.execution_model.clone(),
            inputs: entry.inputs.clone(),
            outputs: entry.outputs.clone(),
            storage_bindings: entry.storage_bindings.clone(),
        })
        .collect();
    let publications: Vec<_> = publications.iter().collect();
    module
        .publish_implicit_bindings(&publications, &associations)
        .map_err(|err| error(err.to_string()))?;
    for (index, pipeline) in module.pipelines.iter_mut().enumerate() {
        let Pipeline::Compute(pipeline) = pipeline else {
            continue;
        };
        for (stage, id) in pipeline.stages.iter_mut().zip(&associations[index]) {
            let Some(entry) = entries.iter().find(|entry| entry.id == *id) else {
                return Err(error("stage entry missing"));
            };
            for (binding, access) in entry.stage_storage_accesses() {
                if let Some(slot) = pipeline.bindings.iter().position(|b|matches!(b,Binding::StorageBuffer {set,binding:b,..} if *set == binding.set && *b == binding.binding)) {
                    stage.uses.record(slot,match access {ResourceAccess::Read=>Access::ReadOnly,ResourceAccess::Write=>Access::WriteOnly,ResourceAccess::ReadWrite=>Access::ReadWrite});
                }
            }
        }
    }
    module.publish_stage_binding_uses(&publications, &associations);
    module.publish_graphics_io(&publications, &associations);
    for (&owner, &pipeline_index) in &compute {
        let declaration = declarations[&owner];
        let Some(definition) = source.defs.iter().find(|d| d.name == owner) else {
            return Err(error("output definition missing"));
        };
        let (body, _) = extract_lambda_params_ref(&definition.body);
        let result_type = strip_existentials(&body.ty);
        for (index, output) in compiler.plan.outputs.iter().filter(|o| o.owner == owner).enumerate() {
            let backing = compiler.plan.backing(output.resource).unwrap_or(output.resource);
            let binding = if let Some(buffer) = compiler.plan.buffers.get(&backing) {
                buffer.binding
            } else {
                let Some(source) = compiler.plan.external(backing) else {
                    return Err(error("output backing missing"));
                };
                let Some(DispatchLen::InputBinding { set, binding, .. }) =
                    compiler.host_lengths.get(&source)
                else {
                    return Err(error("output alias has no physical binding"));
                };
                BindingRef::new(*set, *binding)
            };
            let (name, kind) = match result_type {
                Type::Constructed(TypeName::Record(names), _) => {
                    (names.0[index].clone(), ResultKind::RecordField)
                }
                Type::Constructed(TypeName::Tuple(_), _) => {
                    (format!("result_{index}"), ResultKind::TupleField)
                }
                _ => (declaration.name.clone(), ResultKind::Value),
            };
            let results = declaration
                .source_entry
                .as_ref()
                .map(|s| s.outputs.get(index).cloned().unwrap_or_default())
                .unwrap_or_else(|| vec![SourceResult { index, name, kind }]);
            for result in results {
                module.source_results.push(SourceResultBinding {
                    entry: declaration.source_entry.as_ref().map_or(&declaration.name, |s| &s.name).clone(),
                    name: result.name,
                    kind: result.kind,
                    layout: result_layout(&output.ty),
                    result: result.index,
                    pipeline_index,
                    set: binding.set,
                    binding: binding.binding,
                });
            }
        }
    }
    for (index, pipeline) in module.pipelines.iter_mut().enumerate() {
        let bindings = match pipeline {
            Pipeline::Compute(p) => &mut p.bindings,
            Pipeline::Graphics(p) => &mut p.bindings,
        };
        let mut union = LookupMap::default();
        for binding in bindings {
            if let Binding::StorageBuffer {
                set,
                binding,
                usage,
                access,
                ..
            } = binding
            {
                if *usage == BufferUsage::Intermediate
                    && module
                        .source_results
                        .iter()
                        .any(|r| r.pipeline_index == index && r.set == *set && r.binding == *binding)
                {
                    *usage = BufferUsage::Output;
                }
                union.insert(
                    BindingRef::new(*set, *binding),
                    match access {
                        Access::ReadOnly => ResourceAccess::Read,
                        Access::WriteOnly => ResourceAccess::Write,
                        Access::ReadWrite => ResourceAccess::ReadWrite,
                    },
                );
            }
        }
        for entry in entries.iter_mut().filter(|entry| associations[index].contains(&entry.id)) {
            entry.pipeline_storage_accesses = union.clone();
        }
    }
    host::publish(compiler, entries, &mut module);
    for pipeline in &mut module.pipelines {
        let Pipeline::Compute(pipeline) = pipeline else {
            continue;
        };
        for binding in &mut pipeline.bindings {
            if let Binding::StorageBuffer {
                set, binding, usage, ..
            } = binding
            {
                if compiler.host_tasks.iter().any(|task|matches!(&task.destination,ScalarSource::Binding {set:s,binding:b} if s == set && b == binding)) { *usage = BufferUsage::Intermediate; }
            }
        }
    }
    // Generated compute and graphics captures share storage and capacity with
    // their producer in the same source entry. Only compiler-owned bindings
    // establish that identity; a consumer's input must not override it.
    // Descriptor slots may be reused by unrelated source entries.
    let compute_storage: std::collections::BTreeMap<_, _> = module
        .pipelines
        .iter()
        .filter_map(|item| {
            let Pipeline::Compute(compute) = item else {
                return None;
            };
            let owner = &compute.stages.first()?.owner;
            Some(compute.bindings.iter().map(move |binding| (owner.clone(), binding)))
        })
        .flatten()
        .filter_map(|(owner, item)| {
            let Binding::StorageBuffer {
                set,
                binding,
                resource,
                name,
                usage,
                length,
                ..
            } = item
            else {
                return None;
            };
            if *usage == BufferUsage::Input {
                return None;
            }
            Some((
                (owner, BindingRef::new(*set, *binding)),
                (
                    resource.as_ref().unwrap_or(name).clone(),
                    usage.clone(),
                    length.clone(),
                ),
            ))
        })
        .collect();
    for item in &mut module.pipelines {
        let (owner, bindings) = match item {
            Pipeline::Compute(compute) => (
                compute.stages.first().map(|stage| &stage.owner),
                &mut compute.bindings,
            ),
            Pipeline::Graphics(graphics) => (
                graphics.stages.first().map(|stage| &stage.owner),
                &mut graphics.bindings,
            ),
        };
        let Some(owner) = owner else { continue };
        for binding in bindings {
            let Binding::StorageBuffer {
                set,
                binding,
                resource,
                usage,
                length,
                ..
            } = binding
            else {
                continue;
            };
            let slot = BindingRef::new(*set, *binding);
            if let Some((name, _compute_usage, compute_length)) =
                compute_storage.get(&(owner.clone(), slot))
            {
                *resource = Some(name.clone());
                *length = compute_length.clone();
                if *usage == BufferUsage::Input {
                    *usage = BufferUsage::Intermediate;
                }
            }
        }
    }
    module.source_results.sort_by(|a, b| (&a.entry, a.result).cmp(&(&b.entry, b.result)));
    let mut stage_locations = LookupMap::default();
    for (pipeline, ids) in associations.iter().enumerate() {
        for (index, id) in ids.iter().enumerate() {
            if let Some((_, Some(stage))) = compiler.entry_origins.get(id) {
                stage_locations.insert(stage.key, (pipeline, index));
            }
        }
    }
    for (&key, &(pipeline, index)) in &stage_locations {
        if let Pipeline::Compute(compute) = &mut module.pipelines[pipeline] {
            compute.stages[index].dependencies = compiler
                .plan
                .dependencies(key)
                .into_iter()
                .map(|dependency| {
                    stage_locations
                        .get(&dependency)
                        .copied()
                        .ok_or_else(|| error("published stage dependency missing"))
                })
                .collect::<Result<_, _>>()?;
        }
    }
    module.rebuild_frame_graph();
    super::loops::publish(compiler, entries, &associations, &mut module)?;
    let mut logical_entries = LookupMap::default();
    let mut stage_ids = LookupMap::default();
    for kernel in &kernels {
        let (owner, stage) = &compiler.entry_origins[&kernel.entry];
        logical_entries.entry(*owner).or_insert(kernel.entry);
        if let Some(stage) = stage {
            stage_ids.insert(stage.key, kernel.id);
        }
    }
    for kernel in &mut kernels {
        let (owner, stage) = &compiler.entry_origins[&kernel.entry];
        kernel.source_entry = Some(logical_entries[owner]);
        if let Some(stage) = stage {
            kernel.dependencies = compiler
                .plan
                .dependencies(stage.key)
                .into_iter()
                .map(|key| {
                    let Some(id) = stage_ids.get(&key).copied() else {
                        return Err(error("kernel dependency missing"));
                    };
                    Ok(id)
                })
                .collect::<Result<_, _>>()?;
        }
        let Some(pass) = module.frame_graph.passes.iter().find(|pass| pass.name == kernel.entry_point)
        else {
            return Err(error("published compute pass missing"));
        };
        let mut resources = std::collections::BTreeMap::new();
        for (access, uses) in [
            (ResourceAccess::Read, &pass.reads),
            (ResourceAccess::Write, &pass.writes),
        ] {
            for resource in uses {
                let index =
                    u32::try_from(resource.resource).map_err(|_| error("too many physical resources"))?;
                let id = ResourceId::from_egglog_buffer(index);
                resources
                    .entry(id)
                    .and_modify(|old: &mut ResourceAccess| *old = old.merge(access))
                    .or_insert(access);
            }
        }
        kernel.resources =
            resources.into_iter().map(|(resource, access)| ResourceUse { resource, access }).collect();
    }
    // Publication also resolves dependencies introduced by output copies and
    // shared captures. Keep their physical identities in the same graph.
    let by_name: LookupMap<_, _> = kernels.iter().map(|k| (k.entry_point.clone(), k.id)).collect();
    for kernel in &mut kernels {
        if let Some(pass) = module.frame_graph.passes.iter().find(|p| p.name == kernel.entry_point) {
            for &before in &pass.depends_on {
                if let Some(id) = by_name.get(&module.frame_graph.passes[before].name) {
                    kernel.dependencies.push(*id);
                }
            }
        }
        kernel.dependencies.sort();
        kernel.dependencies.dedup();
    }
    let mut ordered = Vec::new();
    let mut ready = LookupSet::default();
    while !kernels.is_empty() {
        let Some(index) = kernels.iter().position(|k| k.dependencies.iter().all(|id| ready.contains(id)))
        else {
            return Err(error(format!(
                "cyclic physical kernel dependencies {:?}; passes {:?}",
                kernels.iter().map(|k| (&k.entry_point, k.id, &k.dependencies)).collect::<Vec<_>>(),
                module.frame_graph.passes.iter().map(|p| (&p.name, &p.depends_on)).collect::<Vec<_>>()
            )));
        };
        let kernel = kernels.remove(index);
        ready.insert(kernel.id);
        ordered.push(kernel);
    }
    Ok((module, PhysicalKernelGraph::from_ordered(ordered).map_err(error)?))
}
