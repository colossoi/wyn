//! Source ABI declarations and publication for directly emitted entry bodies.
use super::sizes;
use crate::binding_layout::extract_storage_binding;
use crate::egglog::host::lower::{self as host, scalar_type};
use crate::egglog::to_ssa::{error, Compiler};
use crate::egglog::OptimizeError;
use crate::flow::ExecutionModel;
use crate::host::ResultKind;
use crate::host::SourceResultBinding;
use crate::host::{Binding, BufferUsage, DrawBufferRef, DrawCall, DrawCount, GraphicsInvocation};
use crate::host::{
    ComputePipeline, ComputeStage, DispatchSize, FrameGraph, GraphicsPipeline, GraphicsStage,
    ModuleInterface, Pipeline, ShaderStage,
};
use crate::host::{DispatchLoop, ScalarExpr, ScalarSource, ScalarTask};
use crate::interface::publish::{reconcile_storage_binding_access, ModuleInterfacePublish};
use crate::interface::results::result_layout;
use crate::interface::{DrawBufferOperand, SourceResult};
use crate::ssa::types::EntryPoint;
use crate::tlc::extract_lambda_params_ref;
use crate::tlc::DefMeta;
use crate::types::{strip_existentials, Type, TypeName};
use crate::BindingRef;
use crate::ResourceAccess;
use crate::{EntryId, LookupMap, SymbolId};
use egglog_engine::Value;

/// Construct the physical draw operand only after the common storage planner
/// has selected backing for the expression. No descriptor is allocated here.
fn resolve_draw_buffer(
    compiler: &Compiler<'_, '_>,
    operand: &DrawBufferOperand,
) -> Result<DrawBufferRef, OptimizeError> {
    let (entry, slot) = match operand {
        DrawBufferOperand::Input(input) => return Ok(input.clone()),
        DrawBufferOperand::Result { entry, slot } => (entry, slot),
    };
    let outputs = compiler.facts.outputs(*entry)?;
    let Some(&output) = outputs.get(*slot) else {
        return Err(error("draw operand result missing"));
    };
    let (_, _, resource) = compiler.facts.output(output)?;
    let Some(backing) = compiler.facts.backing(resource) else {
        return Err(error("draw operand backing missing"));
    };
    let (binding, name) = if let Some((binding, _, _)) = compiler.bindings.buffer(backing)? {
        (binding, compiler.bindings.buffer_name(backing)?)
    } else {
        let Some(source) = compiler.facts.external(backing) else {
            return Err(error("draw operand source missing"));
        };
        let Some((binding, _)) = compiler.facts.input_storage(source)? else {
            return Err(error("draw operand binding missing"));
        };
        let Some(name) = compiler.program.source.defs.iter().find_map(|d| {
            let DefMeta::EntryPoint(e) = &d.meta else {
                return None;
            };
            e.declaration
                .params
                .iter()
                .find(|p| extract_storage_binding(p) == Some(binding))
                .map(|p| p.name.clone())
        }) else {
            return Err(error("draw operand input name missing"));
        };
        (binding, name)
    };
    Ok(DrawBufferRef {
        set: binding.set,
        binding: binding.binding,
        resource: Some(name.clone()),
        name,
    })
}

fn resolve_draw_count(
    compiler: &Compiler<'_, '_>,
    operand: &DrawBufferOperand,
    count: DrawCount,
) -> Result<DrawCount, OptimizeError> {
    if count != DrawCount::BufferLength {
        return Ok(count);
    }
    let DrawBufferOperand::Result { entry, slot } = operand else {
        return Ok(count);
    };
    let outputs = compiler.facts.outputs(*entry)?;
    let Some(&output) = outputs.get(*slot) else {
        return Err(error("draw operand result missing"));
    };
    let (source, _, _) = compiler.facts.output(output)?;
    match host::length(compiler.program, source) {
        Ok(ScalarExpr::I32(value)) => {
            return Ok(DrawCount::Fixed(
                u32::try_from(value).map_err(|_| error("draw count exceeds u32"))?,
            ))
        }
        Ok(ScalarExpr::U32(value)) => return Ok(DrawCount::Fixed(value)),
        Ok(_) | Err(host::Error::Unsupported) => {}
        Err(host::Error::Invalid(error)) => return Err(error),
    }
    Ok(count)
}

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
        let selected =
            crate::egglog::query::Query(&compiler.program.graph).required("DefaultThreads", (token,))?;
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
                    sizes::dispatch(compiler, *stage, local_size.0)?
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
            }
            ExecutionModel::Vertex | ExecutionModel::Fragment => {
                let Some(group) = &declaration.graphics_group else {
                    return Err(error("graphics entry has no operation identity"));
                };
                let mut draw = group
                    .invocation
                    .draw
                    .try_map_buffers(|operand| resolve_draw_buffer(compiler, operand))?;
                match (&group.invocation.draw, &mut draw) {
                    (DrawCall::Indexed { indices, .. }, DrawCall::Indexed { index_count, .. }) => {
                        *index_count = resolve_draw_count(compiler, indices, *index_count)?;
                    }
                    (DrawCall::Indirect { commands, .. }, DrawCall::Indirect { draw_count, .. })
                    | (
                        DrawCall::IndexedIndirect { commands, .. },
                        DrawCall::IndexedIndirect { draw_count, .. },
                    ) => {
                        *draw_count = resolve_draw_count(compiler, commands, *draw_count)?;
                    }
                    _ => {}
                }
                let invocation = GraphicsInvocation {
                    topology: group.invocation.topology,
                    raster_state: group.invocation.raster_state.clone(),
                    fragment_state: group.invocation.fragment_state.clone(),
                    draw,
                };
                let index = *graphics.entry((group.root, group.operation)).or_insert_with(|| {
                    let index = module.pipelines.len();
                    module.pipelines.push(Pipeline::Graphics(GraphicsPipeline {
                        source_operation: Some(group.operation),
                        invocation,
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
        .publish_implicit_bindings(&publications, &associations, |entry| {
            let declaration = declarations[&compiler.entry_origins[&entry.id].0];
            if let Some(group) = &declaration.graphics_group {
                source.symbols.get(group.root).expect("validated graphics owner")
            } else {
                declaration.source_entry.as_ref().map_or(&declaration.name, |entry| &entry.name)
            }
        })
        .map_err(|err| error(err.to_string()))?;
    module.publish_graphics_io(&publications, &associations);
    for (index, pipeline) in module.pipelines.iter_mut().enumerate() {
        let (bindings, stages): (_, Vec<_>) = match pipeline {
            Pipeline::Compute(p) => (&p.bindings, p.stages.iter_mut().map(|s| &mut s.uses).collect()),
            Pipeline::Graphics(p) => (&p.bindings, p.stages.iter_mut().map(|s| &mut s.uses).collect()),
        };
        for (id, uses) in associations[index].iter().zip(stages) {
            let owner = compiler.entry_origins[id].0;
            let entry = entries
                .iter()
                .find(|entry| entry.id == *id)
                .ok_or_else(|| error("published entry missing"))?;
            let accesses = &entry.stage_descriptor_storage_accesses;
            for (i, binding) in bindings.iter().enumerate() {
                let Some((set, binding)) = binding.slot() else {
                    let Binding::PushConstant { offset, size, .. } = binding else {
                        unreachable!("non-descriptor binding");
                    };
                    let root = compiler.facts.entry_root(owner, compiler.entry_origins[id].1)?;
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
    publish_results(compiler, &compute, &mut module)?;
    publish_frame_graph(compiler, &associations, &mut module)?;
    publish_scalars(compiler, entries, &mut module)?;
    publish_loops(compiler, entries, &associations, &mut module)?;
    Ok(module)
}

/// Map authored return values to their final buffers and mark returned storage.
fn publish_results(
    compiler: &Compiler<'_, '_>,
    compute: &LookupMap<SymbolId, usize>,
    module: &mut ModuleInterface,
) -> Result<(), OptimizeError> {
    let source = compiler.program.source;
    for (&owner, &pipeline_index) in compute {
        let Some(definition) = source.defs.iter().find(|d| d.name == owner) else {
            return Err(error("output definition missing"));
        };
        let DefMeta::EntryPoint(entry) = &definition.meta else {
            return Err(error("output definition is not an entry"));
        };
        let declaration = &entry.declaration;
        if declaration.buffer_demand {
            continue;
        }
        let (body, _) = extract_lambda_params_ref(&definition.body);
        let result_type = strip_existentials(&body.ty);
        for (index, id) in compiler.facts.outputs(owner)?.into_iter().enumerate() {
            let (_, ty, resource) = compiler.facts.output(id)?;
            let Some(backing) = compiler.facts.backing(resource) else {
                return Err(error("output backing missing"));
            };
            let binding = if let Some((binding, _, _)) = compiler.bindings.buffer(backing)? {
                binding
            } else {
                let Some(source) = compiler.facts.external(backing) else {
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
    Ok(())
}

/// Translate selected execution dependencies into host stage order.
fn publish_frame_graph(
    compiler: &Compiler<'_, '_>,
    associations: &[Vec<EntryId>],
    module: &mut ModuleInterface,
) -> Result<(), OptimizeError> {
    let mut locations = LookupMap::default();
    for (pipeline, entries) in associations.iter().enumerate() {
        for (index, entry) in entries.iter().enumerate() {
            let (owner, stage) = compiler.entry_origins[entry];
            if stage.is_none() {
                let symbol = compiler
                    .program
                    .identities
                    .symbols
                    .get(&owner)
                    .ok_or_else(|| error("published entry identity missing"))?;
                if compiler.facts.contains("InterfaceOnlyEntry", (symbol,)) {
                    continue;
                }
            }
            let root = compiler.facts.entry_root(owner, stage)?;
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
    Ok(())
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
                let binding = compiler.bindings.captures[&term];
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
        let root = compiler.facts.entry_root(*owner, *stage)?;
        if !compiler.program.stage.host.keys().any(|(owner, _)| *owner == root) {
            continue;
        }
        for target in targets.iter().filter(|target| target[0] == root) {
            let Some(value) = compiler.program.stage.host.get(&(root, target[1])) else {
                return Err(error("selected host computation missing"));
            };
            let Some((binding, _, _)) = compiler.bindings.buffer(target[3])? else {
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
        let operation = compiler.facts.phase_operation(*stage)?;
        let phase = compiler.facts.phase_name(*stage)?;
        if phase != "loop_exit" {
            continue;
        }
        let Some(source) = compiler.facts.source(operation) else {
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
        let context = crate::egglog::query::Query(&compiler.program.graph)
            .required("ScalarSourceContext", (form[0],))?;
        let count = host::expression(compiler.program, context, form[0])
            .map_err(|error| error.required("loop count"))?;
        let initial = compiler.facts.loop_initial(header)?;
        let initial_length = host::length(compiler.program, initial)
            .map_err(|error| error.required("loop initial length"))?;
        let slot = |value: Option<Value>| -> Result<ScalarSource, OptimizeError> {
            let Some(backing) =
                value.and_then(|v| compiler.facts.value_ref(v)).and_then(|v| compiler.facts.backing(v))
            else {
                return Err(error("loop state has no selected backing"));
            };
            let Some((binding, _, _)) = compiler.bindings.buffer(backing)? else {
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
            let candidate_operation = compiler.facts.phase_operation(*candidate)?;
            let candidate_phase = compiler.facts.phase_name(*candidate)?;
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
