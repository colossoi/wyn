//! Source ABI declarations and publication for directly emitted entry bodies.
use super::sizes;
use crate::egglog::to_ssa::host::{self, scalar_type};
use crate::egglog::to_ssa::{error, Compiler};
use crate::egglog::OptimizeError;
use crate::flow::ExecutionModel;
use crate::host::ResultKind;
use crate::host::SourceResultBinding;
use crate::host::{Binding, BufferUsage};
use crate::host::{
    ComputePipeline, ComputeStage, DispatchSize, FrameGraph, GraphicsPipeline, GraphicsStage,
    ModuleInterface, Pipeline, ShaderStage,
};
use crate::host::{DispatchLoop, ScalarSource, ScalarTask};
use crate::interface::publish::{reconcile_storage_binding_access, ModuleInterfacePublish};
use crate::interface::results::result_layout;
use crate::interface::SourceResult;
use crate::ssa::types::EntryPoint;
use crate::tlc::extract_lambda_params_ref;
use crate::tlc::DefMeta;
use crate::types::{strip_existentials, Type, TypeName};
use crate::BindingRef;
use crate::ResourceAccess;
use crate::{EntryId, LookupMap};
use egglog_engine::Value;

pub(in crate::egglog) fn publish(
    compiler: &mut Compiler<'_, '_>,
    entries: &[EntryPoint],
) -> Result<ModuleInterface, OptimizeError> {
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
    for entry in entries.iter() {
        let Some((owner, stage)) = compiler.entry_origins.get(&entry.id) else {
            return Err(error("entry provenance missing"));
        };
        let Some(&declaration) = declarations.get(owner) else {
            return Err(error("entry declaration missing"));
        };
        let Some(token) = compiler.program.identities.symbols.get(owner) else {
            return Err(error("entry identity missing"));
        };
        let selected = super::required(&compiler.program.graph, "DefaultThreads", (token,))?;
        let threads = u32::try_from(compiler.facts.integer(selected))
            .map_err(|_| error("invalid selected thread count"))?;
        let Some(threads) = std::num::NonZeroU32::new(threads) else {
            return Err(error("selected thread count is zero"));
        };
        match entry.execution_model {
            ExecutionModel::Compute { local_size } => {
                let index = *compute.entry(*owner).or_insert_with(|| {
                    let index = module.pipelines.len();
                    module.pipelines.push(Pipeline::Compute(ComputePipeline {
                        stages: vec![],
                        bindings: vec![],
                        default_total_threads: Some(threads),
                    }));
                    associations.push(vec![]);
                    index
                });
                let dispatch = if let Some(stage) = stage {
                    sizes::dispatch(compiler, *stage)?
                } else {
                    let Some(symbol) = compiler.program.identities.symbols.get(owner) else {
                        return Err(error("entry symbol missing"));
                    };
                    let Some(grid) = compiler.facts.lookup("OriginalEntryGrid", (symbol,)) else {
                        return Err(error("entry grid missing"));
                    };
                    let (x, y, z) = compiler.facts.grid(grid)?;
                    DispatchSize::Fixed {
                        x,
                        y,
                        z,
                        explicit: declaration.compute_dispatch.is_some(),
                    }
                };
                let Pipeline::Compute(pipeline) = &mut module.pipelines[index] else {
                    return Err(error("compute pipeline identity mismatch"));
                };
                if compiler
                    .program
                    .identities
                    .symbols
                    .get(owner)
                    .is_some_and(|symbol| compiler.facts.contains("InterfaceOnlyEntry", (symbol,)))
                {
                    associations[index].push(entry.id);
                    continue;
                }
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
                if let DispatchSize::DerivedFrom { workgroup_size, .. } = dispatch {
                    if workgroup_size % local_size.0 != 0 {
                        return Err(error("launch divisor must cover whole workgroups"));
                    }
                }
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
    let publications: Vec<_> = entries.iter().collect();
    module
        .publish_implicit_bindings(&publications, &associations)
        .map_err(|err| error(err.to_string()))?;
    module.publish_graphics_io(&publications, &associations);
    for (index, pipeline) in module.pipelines.iter_mut().enumerate() {
        let (bindings, stages): (_, Vec<_>) = match pipeline {
            Pipeline::Compute(p) => (&p.bindings, p.stages.iter_mut().map(|s| &mut s.uses).collect()),
            Pipeline::Graphics(p) => (&p.bindings, p.stages.iter_mut().map(|s| &mut s.uses).collect()),
        };
        for (id, uses) in associations[index].iter().zip(stages) {
            let owner = compiler.entry_origins[id].0;
            let accesses = super::entry_accesses(compiler, owner, compiler.entry_origins[id].1)?;
            for (i, binding) in bindings.iter().enumerate() {
                let Some((set, binding)) = binding.slot() else {
                    let Binding::PushConstant { offset, size, .. } = binding else {
                        unreachable!("non-descriptor binding");
                    };
                    let root = match compiler.entry_origins[id].1 {
                        Some(stage) => compiler.facts.constructor("KernelRoot", (stage,)),
                        None => compiler
                            .program
                            .identities
                            .symbols
                            .get(&owner)
                            .and_then(|symbol| compiler.facts.constructor("EntryRoot", (symbol,))),
                    };
                    let Some(root) = root else {
                        return Err(error("stage resource root missing"));
                    };
                    if compiler
                        .facts
                        .contains("RootPushInput", (root, i64::from(*offset), i64::from(*size)))
                    {
                        uses.record(i, crate::host::Access::ReadOnly);
                    }
                    continue;
                };
                let binding = BindingRef::new(set, binding);
                let Some(&access) = accesses.get(&binding) else {
                    continue;
                };
                uses.record(
                    i,
                    match access {
                        ResourceAccess::Read => crate::host::Access::ReadOnly,
                        ResourceAccess::Write => crate::host::Access::WriteOnly,
                        ResourceAccess::ReadWrite => crate::host::Access::ReadWrite,
                    },
                );
            }
        }
    }
    for pipeline in &mut module.pipelines {
        match pipeline {
            Pipeline::Compute(p) => {
                reconcile_storage_binding_access(&mut p.bindings, p.stages.iter().map(|stage| &stage.uses))
            }
            Pipeline::Graphics(p) => {
                reconcile_storage_binding_access(&mut p.bindings, p.stages.iter().map(|stage| &stage.uses))
            }
        }
    }
    for (&owner, &pipeline_index) in &compute {
        let declaration = declarations[&owner];
        let Some(definition) = source.defs.iter().find(|d| d.name == owner) else {
            return Err(error("output definition missing"));
        };
        let (body, _) = extract_lambda_params_ref(&definition.body);
        let result_type = strip_existentials(&body.ty);
        for (index, id) in compiler.plan.outputs(owner)?.into_iter().enumerate() {
            let (_, ty, resource) = compiler.plan.output(id)?;
            let Some(backing) = compiler.plan.backing(resource) else {
                return Err(error("output backing missing"));
            };
            let binding = if let Some((binding, _, _)) = compiler.plan.buffer(backing)? {
                binding
            } else {
                let Some(source) = compiler.plan.external(backing) else {
                    return Err(error("output backing missing"));
                };
                let Some((binding, _)) = compiler.facts.input_storage(source)? else {
                    return Err(error("output alias has no physical binding"));
                };
                binding
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
            let results = if let Some(source) = &declaration.source_entry {
                let Some(outputs) = source.outputs.get(index) else {
                    return Err(error("source output route is missing"));
                };
                outputs.clone()
            } else {
                vec![SourceResult { index, name, kind }]
            };
            for result in results {
                module.source_results.push(SourceResultBinding {
                    entry: declaration.source_entry.as_ref().map_or(&declaration.name, |s| &s.name).clone(),
                    name: result.name,
                    kind: result.kind,
                    layout: result_layout(ty),
                    result: result.index,
                    pipeline_index,
                    set: binding.set,
                    binding: binding.binding,
                });
            }
        }
    }
    module.source_results.sort_by(|a, b| (&a.entry, a.result).cmp(&(&b.entry, b.result)));
    for result in &module.source_results {
        let Pipeline::Compute(pipeline) = &mut module.pipelines[result.pipeline_index] else {
            continue;
        };
        for binding in &mut pipeline.bindings {
            if let Binding::StorageBuffer {
                set, binding, usage, ..
            } = binding
            {
                if *set == result.set && *binding == result.binding && *usage == BufferUsage::Intermediate {
                    *usage = BufferUsage::Output;
                }
            }
        }
    }
    let mut resources = LookupMap::default();
    for entry in entries.iter() {
        let owner = compiler.entry_origins[&entry.id].0;
        let declaration = declarations[&owner];
        let source = declaration.source_entry.as_ref().map_or(&declaration.name, |entry| &entry.name);
        for storage in &entry.storage_bindings {
            let (Some(name), Some(length)) = (&storage.logical_resource, &storage.length) else {
                return Err(error("generated storage has no resource identity or capacity"));
            };
            let key = (source.clone(), storage.binding);
            if let Some((previous, capacity)) = resources.insert(key, (name.clone(), length.clone())) {
                if previous != *name || capacity != *length {
                    return Err(error(format!("generated resource identities disagree for {source}: {:?}: {previous} versus {name}", storage.binding)));
                }
            }
        }
    }
    for (index, pipeline) in module.pipelines.iter_mut().enumerate() {
        let Some(first) = associations[index].first() else {
            return Err(error("pipeline has no selected entry interface"));
        };
        let declaration = declarations[&compiler.entry_origins[first].0];
        let owner = declaration.graphics_group.as_ref().map(|group| group.root);
        let source = if let Some(owner) = owner {
            let Some(name) = compiler.program.source.symbols.get(owner) else {
                return Err(error("graphics source name missing"));
            };
            name
        } else {
            declaration.source_entry.as_ref().map_or(&declaration.name, |entry| &entry.name)
        };
        let bindings = match pipeline {
            Pipeline::Compute(pipeline) => &mut pipeline.bindings,
            Pipeline::Graphics(pipeline) => &mut pipeline.bindings,
        };
        for item in bindings {
            if let Binding::StorageBuffer {
                set,
                binding,
                resource,
                usage,
                length,
                ..
            } = item
            {
                if let Some((name, capacity)) =
                    resources.get(&(source.clone(), BindingRef::new(*set, *binding)))
                {
                    *resource = Some(name.clone());
                    *length = Some(capacity.clone());
                    if *usage == BufferUsage::Input {
                        *usage = BufferUsage::Intermediate;
                    }
                }
            }
        }
    }
    let mut locations = LookupMap::default();
    for (pipeline, entries) in associations.iter().enumerate() {
        for (index, entry) in entries.iter().enumerate() {
            let (owner, stage) = compiler.entry_origins[entry];
            let root = if let Some(stage) = stage {
                compiler.facts.constructor("KernelRoot", (stage,))
            } else {
                let Some(symbol) = compiler.program.identities.symbols.get(&owner) else {
                    return Err(error("published entry identity missing"));
                };
                if compiler.facts.contains("InterfaceOnlyEntry", (symbol,)) {
                    continue;
                }
                compiler.facts.constructor("EntryRoot", (symbol,))
            };
            let Some(root) = root else {
                return Err(error("published execution root missing"));
            };
            let index = if matches!(module.pipelines[pipeline], Pipeline::Graphics(_)) { 0 } else { index };
            locations.insert(root, (pipeline, index));
        }
    }
    let mut edges = Vec::new();
    crate::egglog::query::Query(&compiler.program.graph).for_each("PublicationDependency", |row| {
        let (Some(&before), Some(&after)) = (locations.get(&row[0]), locations.get(&row[1])) else {
            return Err(error("selected dependency has no published execution root"));
        };
        if before != after {
            edges.push((before, after));
        }
        Ok(())
    })?;
    edges.sort_unstable();
    edges.dedup();
    for &(before, (pipeline, index)) in &edges {
        if let Pipeline::Compute(compute) = &mut module.pipelines[pipeline] {
            compute.stages[index].dependencies.push(before);
        }
    }
    module.frame_graph = FrameGraph::from_selected_pipelines(&module.pipelines, &edges).map_err(error)?;
    publish_scalars(compiler, entries, &mut module)?;
    publish_loops(compiler, entries, &associations, &mut module)?;
    Ok(module)
}

fn publish_scalars(
    compiler: &Compiler<'_, '_>,
    entries: &[EntryPoint],
    module: &mut ModuleInterface,
) -> Result<(), OptimizeError> {
    let mut targets = Vec::new();
    compiler.program.graph.constructor_enodes("HostTarget", |row| targets.push(row.children.to_vec()))?;
    for entry in entries {
        let (owner, stage) = &compiler.entry_origins[&entry.id];
        if let Some(stage) = stage {
            for (&term, capture) in &compiler.program.stage.captures {
                if !capture.stages.contains(stage) {
                    continue;
                }
                let binding = compiler.plan.captures[&term];
                module.scalar_tasks.push(ScalarTask {
                    stage: entry.name.clone(),
                    destination: ScalarSource::Binding {
                        set: binding.set,
                        binding: binding.binding,
                    },
                    offset: 0,
                    ty: capture.ty,
                    value: capture.value.clone(),
                    replaces_dispatch: false,
                });
            }
        }
        let root = if let Some(stage) = stage {
            compiler.facts.constructor("KernelRoot", (*stage,))
        } else {
            let Some(token) = compiler.program.identities.symbols.get(owner) else {
                return Err(error("host entry identity missing"));
            };
            compiler.facts.constructor("EntryRoot", (token,))
        };
        let Some(root) = root else {
            return Err(error("host execution root missing"));
        };
        let preferred = super::required(&compiler.program.graph, "PreferredExecutor", (root,))?;
        if compiler.facts.enode("GpuExecutor", preferred).is_some() {
            continue;
        }
        if compiler.facts.enode("CpuExecutor", preferred).is_none() {
            return Err(error("unknown preferred executor"));
        }
        if !compiler.program.stage.host.keys().any(|(owner, _)| *owner == root) {
            continue;
        }
        for target in targets.iter().filter(|target| target[0] == root) {
            let Some(value) = compiler.program.stage.host.get(&(root, target[1])) else {
                return Err(error("selected host computation missing"));
            };
            let Some((binding, _, _)) = compiler.plan.buffer(target[3])? else {
                return Err(error("host destination allocation missing"));
            };
            let Some(ty) = compiler.facts.source_type(target[1]).and_then(scalar_type) else {
                return Err(error("selected host result type missing"));
            };
            module.scalar_tasks.push(ScalarTask {
                stage: entry.name.clone(),
                destination: ScalarSource::Binding {
                    set: binding.set,
                    binding: binding.binding,
                },
                offset: 0,
                ty,
                value: value.clone(),
                replaces_dispatch: true,
            });
        }
    }
    Ok(())
}

fn publish_loops(
    compiler: &mut Compiler<'_, '_>,
    entries: &[EntryPoint],
    associations: &[Vec<EntryId>],
    module: &mut ModuleInterface,
) -> Result<(), OptimizeError> {
    for entry in entries {
        let Some((_, Some(stage))) = compiler.entry_origins.get(&entry.id) else {
            continue;
        };
        let operation = compiler.plan.phase_operation(*stage)?;
        let phase = compiler.plan.phase_name(*stage)?;
        if phase != "loop_exit" {
            continue;
        }
        let Some(source) = compiler.plan.source(operation) else {
            return Err(error("loop source missing"));
        };
        let Some((header, iteration)) = compiler.facts.loops(source) else {
            return Err(error("loop regions missing"));
        };
        let Some(form) = compiler
            .facts
            .lookup("SourceLoopForm", (source,))
            .and_then(|form| compiler.facts.enode("ForCount", form))
        else {
            return Err(error("host loop has no count recipe"));
        };
        let context = super::required(&compiler.program.graph, "ScalarSourceContext", (form[0],))?;
        let count = host::expression(compiler.program, context, form[0])
            .map_err(|error| error.required("loop count"))?;
        let initial = compiler.facts.loop_initial(header)?;
        let initial_length = host::length(compiler.program, initial)
            .map_err(|error| error.required("loop initial length"))?;
        let slot = |value: Option<Value>| -> Result<ScalarSource, OptimizeError> {
            let Some(backing) =
                value.and_then(|v| compiler.plan.value_ref(v)).and_then(|v| compiler.plan.backing(v))
            else {
                return Err(error("loop state has no selected backing"));
            };
            let Some((binding, _, _)) = compiler.plan.buffer(backing)? else {
                return Err(error("loop state has no selected allocation"));
            };
            Ok(ScalarSource::Binding {
                set: binding.set,
                binding: binding.binding,
            })
        };
        let Some(pipeline) = associations.iter().position(|ids| ids.contains(&entry.id)) else {
            return Err(error("loop pipeline missing"));
        };
        let mut setup = None;
        let mut completion = None;
        let mut body = Vec::new();
        for (index, id) in associations[pipeline].iter().enumerate() {
            let Some((_, Some(candidate))) = compiler.entry_origins.get(id) else {
                continue;
            };
            let candidate_operation = compiler.plan.phase_operation(*candidate)?;
            let candidate_phase = compiler.plan.phase_name(*candidate)?;
            if candidate_operation == operation {
                if candidate_phase == "loop_enter" {
                    setup = Some(index);
                }
                if candidate_phase == "loop_exit" {
                    completion = Some(index);
                }
            }
            if compiler.facts.contains("LoopStage", (operation, *candidate)) {
                body.push(index);
            }
        }
        let (Some(setup), Some(completion)) = (setup, completion) else {
            return Err(error("loop stage boundary missing"));
        };
        module.dispatch_loops.push(DispatchLoop {
            initial_length,
            pipeline,
            setup,
            completion,
            body,
            count,
            index: slot(compiler.facts.iteration(iteration))?,
            current: slot(compiler.facts.loop_state(header))?,
            next: slot(compiler.facts.result(iteration))?,
        });
    }
    Ok(())
}
