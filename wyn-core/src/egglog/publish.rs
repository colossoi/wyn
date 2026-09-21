//! Publish the finalized resource/dispatch readout through the shared ABI.
use super::abi::error;
use super::{OptimizeError, Program, Scheduled};
use crate::flow::ExecutionModel;
use crate::host::{
    Access, Binding, BufferUsage, ComputePipeline, ComputeStage, GraphicsPipeline, GraphicsStage,
    ModuleInterface, Pipeline, ResultKind, ShaderStage, SourceResultBinding,
};
use crate::interface::publish::ModuleInterfacePublish;
use crate::interface::results::result_layout;
use crate::interface::StorageRole;
use crate::interface::{BindingExposure, EntryInputKind, EntryKind, EntryPublication, StorageAccess};
use crate::ssa::types::EntryPoint;
use crate::types::{strip_existentials, Type, TypeName};
use crate::BindingRef;
use crate::{EntryId, LookupMap, ResourceAccess};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn publish(
    data: &Program<Scheduled>,
    entries: &mut [EntryPoint],
) -> Result<ModuleInterface, OptimizeError> {
    let abi = &data.state.abi;
    let source_entries = &data.entries;
    let symbols = &data.symbols;
    let outputs = &data.state.outputs;
    let mut pipeline = ModuleInterface::default();
    let mut associations = vec![];
    let entry_indices: BTreeMap<_, _> = entries.iter().enumerate().map(|(i, e)| (e.id, i)).collect();
    let mut graphics_groups = BTreeMap::new();
    let symbol_names: BTreeMap<_, _> = symbols.values().map(|s| (s.source.0, &s.name)).collect();
    for (&owner, blocks) in &abi.entry_roots {
        let declaration = &source_entries[owner].declaration;
        if declaration.entry_kind != EntryKind::Compute {
            let Some(group) = declaration.graphics_group.as_ref() else {
                return Err(error("graphics entry is missing its operation identity"));
            };
            let key = (group.root.0, group.operation);
            let index = *graphics_groups.entry(key).or_insert_with(|| {
                let index = pipeline.pipelines.len();
                pipeline.pipelines.push(Pipeline::Graphics(GraphicsPipeline {
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
            let Pipeline::Graphics(graphics) = &mut pipeline.pipelines[index] else {
                unreachable!("graphics group points to non-graphics pipeline {index}")
            };
            for &root in blocks {
                let entry = &entries[entry_indices[&EntryId::from(root.as_u32())]];
                let Some(owner) = symbol_names.get(&group.root.0) else {
                    return Err(error("missing graphics owner"));
                };
                graphics.stages.push(GraphicsStage {
                    entry_point: entry.name.clone(),
                    owner: owner.to_string(),
                    stage: match entry.execution_model {
                        ExecutionModel::Vertex => ShaderStage::Vertex,
                        ExecutionModel::Fragment => ShaderStage::Fragment,
                        _ => return Err(error("compute dispatch in graphics entry")),
                    },
                    uses: Default::default(),
                });
                associations[index].push(entry.id);
            }
            continue;
        }
        let pipeline_index = pipeline.pipelines.len();
        let source_name = &source_entries[owner].declaration.name;
        let mut stages = vec![];
        let mut ids = vec![];
        for &root in blocks {
            let entry = &mut entries[entry_indices[&EntryId::from(root.as_u32())]];
            let size = match entry.execution_model {
                ExecutionModel::Compute { local_size } => local_size,
                _ => unreachable!(),
            };
            let dispatch_size = abi.dispatch_sizes[&root].clone();
            let accesses = &abi.root_accesses[&root];
            for declaration in &mut entry.storage_bindings {
                let Some(access) = accesses.get(&declaration.binding) else {
                    return Err(error(format!(
                        "unplanned storage access {:?} in {root:?} ({source_name})",
                        declaration.binding
                    )));
                };
                declaration.role = match access {
                    ResourceAccess::Read => StorageRole::Input,
                    ResourceAccess::Write => StorageRole::Output,
                    ResourceAccess::ReadWrite => StorageRole::InputOutput,
                };
            }
            for input in &mut entry.inputs {
                if let EntryInputKind::Storage {
                    exposure: BindingExposure::Host(binding),
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
                }
            }
            entry.stage_descriptor_storage_accesses = accesses.iter().map(|(&b, &a)| (b, a)).collect();
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
        for output in outputs.values().filter(|o| o.entry == owner) {
            let Some(id) = output.buffer else {
                return Err(error(format!(
                    "output {} of {source_name} has no planned backing",
                    output.index
                )));
            };
            let Some(binding) = abi.buffer_bindings.get(&id).copied() else {
                return Err(error("output has no physical binding"));
            };
            let region = &data.regions[data.definitions[source_entries[owner].definition].body];
            let result_type =
                region.results.first().map(|e| strip_existentials(&data.types[data.expressions[*e].ty].ty));
            let name = match result_type {
                Some(Type::Constructed(TypeName::Record(names), _)) => {
                    let Some(name) = names.0.get(output.index) else {
                        return Err(error("source result has no record field"));
                    };
                    name.clone()
                }
                Some(Type::Constructed(TypeName::Tuple(_), _)) => format!("result_{}", output.index),
                _ => source_name.clone(),
            };
            pipeline.source_results.push(SourceResultBinding {
                entry: source_name.clone(),
                name,
                kind: match result_type {
                    Some(Type::Constructed(TypeName::Record(_), _)) => ResultKind::RecordField,
                    Some(Type::Constructed(TypeName::Tuple(_), _)) => ResultKind::TupleField,
                    _ => ResultKind::Value,
                },
                layout: result_layout(&data.types[data.expressions[output.expression].ty].ty),
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
            outputs: e.outputs.clone(),
            storage_bindings: e.storage_bindings.clone(),
        })
        .collect();
    let publications: Vec<_> = publications.iter().collect();
    pipeline.publish_implicit_bindings(&publications, &associations).map_err(|e| error(e.to_string()))?;
    // The shared publisher deliberately leaves storage traffic to the
    // finalized plan; it adds only non-storage interfaces itself.
    for (index, p) in pipeline.pipelines.iter_mut().enumerate() {
        let Pipeline::Compute(p) = p else { continue };
        let slots: BTreeMap<_, _> = p
            .bindings
            .iter()
            .enumerate()
            .filter_map(|(i, b)| match b {
                Binding::StorageBuffer { set, binding, .. } => Some((BindingRef::new(*set, *binding), i)),
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
                            ResourceAccess::Read => Access::ReadOnly,
                            ResourceAccess::Write => Access::WriteOnly,
                            ResourceAccess::ReadWrite => Access::ReadWrite,
                        },
                    );
                }
            }
        }
    }
    pipeline.publish_stage_binding_uses(&publications, &associations);
    pipeline.publish_graphics_io(&publications, &associations);
    for (index, p) in pipeline.pipelines.iter_mut().enumerate() {
        let bindings = match p {
            Pipeline::Compute(p) => &mut p.bindings,
            Pipeline::Graphics(p) => &mut p.bindings,
        };
        let outputs: BTreeSet<_> = pipeline
            .source_results
            .iter()
            .filter(|o| o.pipeline_index == index)
            .map(|o| (o.set, o.binding))
            .collect();
        let mut union = LookupMap::new();
        for b in bindings {
            if let Binding::StorageBuffer {
                set,
                binding,
                usage,
                access,
                ..
            } = b
            {
                if outputs.contains(&(*set, *binding)) && *usage == BufferUsage::Intermediate {
                    *usage = BufferUsage::Output;
                }
                let a = match access {
                    Access::ReadOnly => ResourceAccess::Read,
                    Access::WriteOnly => ResourceAccess::Write,
                    Access::ReadWrite => ResourceAccess::ReadWrite,
                };
                union.insert(BindingRef::new(*set, *binding), a);
            }
        }
        for id in &associations[index] {
            entries[entry_indices[id]].pipeline_storage_accesses = union.clone();
        }
    }
    // Graphics captures share the physical storage and capacity of their compute producer.
    let compute_storage: BTreeMap<_, _> = pipeline
        .pipelines
        .iter()
        .filter_map(|item| {
            let Pipeline::Compute(compute) = item else {
                return None;
            };
            Some(compute.bindings.iter())
        })
        .flatten()
        .filter_map(|item| {
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
            Some((
                BindingRef::new(*set, *binding),
                (
                    resource.as_ref().unwrap_or(name).clone(),
                    usage.clone(),
                    length.clone(),
                ),
            ))
        })
        .collect();
    for item in &mut pipeline.pipelines {
        let Pipeline::Graphics(graphics) = item else {
            continue;
        };
        for binding in &mut graphics.bindings {
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
            if let Some((name, compute_usage, compute_length)) = compute_storage.get(&slot) {
                *resource = Some(name.clone());
                *length = compute_length.clone();
                *usage = if *compute_usage == BufferUsage::Input {
                    BufferUsage::Input
                } else {
                    BufferUsage::Intermediate
                };
            }
        }
    }
    pipeline.rebuild_frame_graph();
    Ok(pipeline)
}
