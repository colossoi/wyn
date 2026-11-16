//! Lower public declarations and bind their values to the final memory signature.
use super::{builder_error, error, host, Body, Compiler, OptimizeError, Typed};
use crate::builtins::catalog;
use crate::egglog::allocation::{ArrayOutput, Output, Transfer};
use crate::egglog::execution::{Grouping, StepKind};
use crate::egglog::query::Query;
use crate::flow::ExecutionModel;
use crate::host::{
    BufferLen, BufferSource, ComputePipeline, ComputeStage, DispatchLoop, DispatchLoopControl,
    DispatchSize, Expr, FrameGraph, LoopCarry, ModuleInterface, Pipeline, ScalarExpr, ScalarSource,
    ScalarTask, ScalarType,
};
use crate::host::{
    DrawBufferRef, GraphicsInvocation, GraphicsPipeline, GraphicsStage, ResultKind, ResultShape,
    ShaderStage, SourceResultBinding,
};
use crate::interface::lowering::build_entry_outputs;
use crate::interface::publish::{
    reconcile_storage_binding_access, EntryInput, EntryOutput, EntryPoint, ModuleInterfacePublish,
    StorageBindingDecl, StorageHost,
};
use crate::interface::results::result_layout_with_storage;
use crate::interface::DrawBufferOperand;
use crate::interface::{Attribute, EntryInputKind, EntryKind, StorageLayout, StorageRole};
use crate::op::{BinaryOperator, OpTag, PureViewSource};
use crate::ssa::context::BackendGlobal;
use crate::ssa::layout::{block_layout, storage_elem_stride, storage_value_type, type_byte_size};
use crate::ssa::stage::Elaborated;
use crate::ssa::types::{InstKind, ValueRef};
use crate::tlc::{extract_lambda_params_ref, DefMeta};
use crate::types::{self, buffer_tag, strip_existentials, Type, TypeExt, TypeName};
use crate::LookupSet;
use crate::{BindingRef, CodegenTarget, LookupMap, ResourceAccess, StableMap};
use egglog_engine::Value;
use wyn_base::IdSource;

mod collectives;

pub(super) fn lower(
    compiler: &mut Compiler<'_, '_>,
    roots: Vec<Value>,
    target: CodegenTarget,
) -> Result<Elaborated, OptimizeError> {
    let mut entries = Vec::new();
    let mut source_names = LookupMap::default();
    let mut pipeline = ModuleInterface::default();
    let mut associations = Vec::new();
    let mut ids = IdSource::new();
    let mut origins = LookupMap::default();
    let mut graphics = LookupMap::default();
    let mut invocations = Vec::new();
    let mut next_generated = compiler.program.source.global_context.auto_storage_binding_ids.peek_id();
    for root in roots {
        let query = super::Query(&compiler.program.graph);
        let order = query.required("EntryOrder", root)?;
        let symbol = *compiler.program.symbols.resolve(query.0.value_to_base::<i64>(order));
        let Some(source) = compiler.program.source.defs.iter().find(|definition| definition.name == symbol)
        else {
            return Err(error("entry source metadata missing"));
        };
        let DefMeta::EntryPoint(metadata) = &source.meta else {
            return Err(error("entry declaration missing"));
        };
        let declaration = &metadata.declaration;
        let (source_body, _) = extract_lambda_params_ref(&source.body);
        let function = root;
        let Some(&region) = compiler.definitions.get(&function) else {
            return Err(error("entry graph function missing"));
        };
        let params = query.sequence("RegionParameter", region)?;
        let Some(inputs) = compiler.execution.storage.parameters.get(&function).cloned() else {
            return Err(error("entry has no parameter layout"));
        };
        let templates = inputs.templates;
        let parameter_inputs = inputs.components;

        let next = templates
            .iter()
            .filter_map(|(input, _)| input.descriptor_binding())
            .chain(
                declaration.outputs.iter().filter_map(|output| match &output.attribute {
                    Some(Attribute::Storage { set, binding, .. }) => Some(BindingRef::new(*set, *binding)),
                    _ => None,
                }),
            )
            .filter(|binding| binding.set == 0)
            .map(|binding| binding.binding + 1)
            .max()
            .unwrap_or(0)
            .max(next_generated);
        let lane_dependent = query
            .lookup("LaneDependentFunction", (function,))?
            .is_some_and(|value| query.0.value_to_base::<bool>(value));
        let invocation = if lane_dependent && declaration.entry_kind == EntryKind::Compute {
            let size = declaration.compute_dispatch.map_or(
                DispatchSize::Fixed {
                    x: 1,
                    y: 1,
                    z: 1,
                    explicit: false,
                },
                |grid| DispatchSize::Fixed {
                    x: grid.x,
                    y: grid.y,
                    z: grid.z,
                    explicit: true,
                },
            );
            ((128, 1, 1), size)
        } else {
            (
                (1, 1, 1),
                DispatchSize::Fixed {
                    x: 1,
                    y: 1,
                    z: 1,
                    explicit: true,
                },
            )
        };
        let mut body = Body::new(compiler, function, vec![], types::unit());
        body.generated = Some(GeneratedBuffers {
            owner: declaration.name.clone(),
            next,
            declarations: vec![],
            scalar_tasks: vec![],
        });
        let mut emission = EntryEmission {
            body,
            templates,
            params,
            parameter_inputs,
            inputs: vec![],
            parameters: LookupSet::default(),
            saved: StableMap::default(),
            name: declaration.name.clone(),
            owner: declaration
                .source_entry
                .as_ref()
                .map_or(&declaration.name, |source| &source.name)
                .clone(),
            kind: declaration.entry_kind,
            span: source_body.span,
            pipeline: pipeline.pipelines.len(),
            invocation,
            pending: false,
            pending_observable: false,
            restore_names: None,
            dispatch_override: None,
            stages: vec![],
            entries: vec![],
            loops: vec![],
        };
        emission.restart()?;
        let live = live_names(emission.body.query(), &[], region)?;
        emission
            .statements(region, &live)
            .map_err(|failure| error(format!("entry {}: {failure}", declaration.name)))?;
        let Some(results) = emission.body.compiler.execution.storage.results.get(&function).cloned() else {
            return Err(error("entry has no physical result plan"));
        };
        let output_arity = results.arity;
        let whole_result = results.whole;
        let output_types = results.types;
        let mut output_values = emission.body.results(region)?;
        if results.unpack {
            let [value]: [Typed; 1] =
                output_values.try_into().map_err(|_| error("entry result arity mismatch"))?;
            output_values = emission.body.unpack(value)?;
        }
        let (output_values, output_shapes) =
            emission.prepare_array_outputs(region, output_values, &results.layouts)?;
        let Some(generated) = emission.body.generated.as_ref() else {
            return Err(error("entry buffers missing"));
        };
        let mut bindings = IdSource::new();
        while bindings.peek_id() < generated.next {
            bindings.next_id();
        }
        let publish_direct = declaration.entry_kind == EntryKind::Compute
            && declaration.outputs.iter().all(|output| {
                output.attribute.is_none()
                    || (declaration.source_entry.is_some()
                        && matches!(output.attribute, Some(Attribute::Storage { set: 0, .. })))
            });
        let physical = if output_arity == 0 {
            source_body.ty.clone()
        } else if whole_result {
            let Some(value) = output_values.first() else {
                return Err(error("stage result missing"));
            };
            storage_value_type(&value.ty)
        } else {
            match strip_existentials(&source_body.ty) {
                Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), _) => {
                    Type::Constructed(
                        name.clone(),
                        output_values.iter().map(|value| storage_value_type(&value.ty)).collect(),
                    )
                }
                _ => output_values
                    .first()
                    .map(|value| storage_value_type(&value.ty))
                    .ok_or_else(|| error("entry output missing"))?,
            }
        };
        let destinations = if publish_direct {
            output_values
                .iter()
                .map(|value| {
                    result_buffer(&emission.body, value).filter(|binding| {
                        declaration.outputs.iter().all(|output| output.attribute.is_none())
                            || generated
                                .declarations
                                .iter()
                                .any(|declaration| declaration.binding == *binding)
                    })
                })
                .collect()
        } else {
            vec![]
        };
        let capacities = if output_arity == 0 {
            vec![]
        } else {
            output_capacities(&emission.body, &output_values, &destinations, &results.layouts)?
        };
        let mut outputs = build_entry_outputs(
            declaration,
            &physical,
            &capacities,
            &destinations,
            declaration.entry_kind == EntryKind::Compute,
            &mut bindings,
        )
        .map_err(|failure| error(failure.to_string()))?;
        if declaration.entry_kind == EntryKind::Compute {
            let arity = outputs.len();
            for (i, output) in outputs.iter_mut().enumerate() {
                if output_values[i].ty.is_array() {
                    let count = match &output_shapes[i] {
                        Some(ResultShape::Rectangular(dimensions)) => dimensions.first().cloned(),
                        Some(ResultShape::Segmented { .. }) => None,
                        None => Some(emission.body.host_array_length(&output_values[i])?),
                    };
                    output.host.elements = count.map(|count| (emission.pipeline, count));
                }
                output.host.target.get_or_insert_with(|| {
                    if arity == 1 {
                        format!("{}_output", declaration.name)
                    } else {
                        format!("{}_output_{i}", declaration.name)
                    }
                });
            }
        }
        if let Some(generated) = emission.body.generated.as_mut() {
            generated.next = bindings.peek_id();
            for (declared, output) in declaration.outputs.iter().zip(&outputs) {
                let Some(Attribute::Storage { set: 0, binding, .. }) = declared.attribute else {
                    continue;
                };
                let Some(selected) = output.storage_binding() else {
                    continue;
                };
                let prescribed = BindingRef::new(0, binding);
                if selected == prescribed {
                    continue;
                }
                let Some(mut alias) = generated
                    .declarations
                    .iter()
                    .find(|declaration| declaration.binding == selected)
                    .cloned()
                else {
                    continue;
                };
                alias.binding = prescribed;
                if generated
                    .declarations
                    .iter()
                    .any(|declaration| declaration.binding == prescribed && declaration.host != alias.host)
                {
                    return Err(error("selected output aliases an occupied descriptor slot"));
                }
                if !generated.declarations.iter().any(|declaration| declaration.binding == prescribed) {
                    generated.declarations.push(alias);
                }
            }
            if declaration.buffer_demand
                || (declaration.source_entry.is_some()
                    && declaration
                        .outputs
                        .iter()
                        .any(|output| matches!(output.attribute, Some(Attribute::Storage { set: 0, .. }))))
            {
                for output in &outputs {
                    let Some(binding) = output.storage_binding() else {
                        return Err(error("stage result is not storage"));
                    };
                    if generated.declarations.iter().any(|declaration| declaration.binding == binding) {
                        continue;
                    }
                    let Some(length) = output.host.length.clone() else {
                        return Err(error("stage result capacity missing"));
                    };
                    let Some(resource) = output.host.target.clone() else {
                        return Err(error("stage result resource missing"));
                    };
                    generated.declarations.push(StorageBindingDecl {
                        binding,
                        role: StorageRole::Intermediate,
                        elem_ty: storage_value_type(output.ty.elem_type().unwrap_or(&output.ty)),
                        host: StorageHost {
                            logical_resource: resource,
                            length,
                        },
                    });
                }
            }
        }
        if declaration.entry_kind == EntryKind::Compute && !declaration.buffer_demand {
            for (index, ((ty, value), output)) in
                output_types.iter().zip(&output_values).zip(&outputs).enumerate()
            {
                let Some(binding) = output.storage_binding() else {
                    return Err(error("source result has no output buffer"));
                };
                let (logical, shape) = if ty.is_array() {
                    let shape = if let Some(shape) = &output_shapes[index] {
                        shape.clone()
                    } else {
                        ResultShape::Rectangular(output_shape(&emission.body, value)?)
                    };
                    let mut element = ty;
                    let mut fixed_rows = true;
                    while element.is_array() {
                        let Some(next) = element.elem_type() else {
                            return Err(error("result element missing"));
                        };
                        let next = strip_existentials(next);
                        if next.is_array() {
                            fixed_rows &=
                                matches!(next.array_size(), Some(Type::Constructed(TypeName::Size(_), _)));
                        }
                        element = next;
                    }
                    if fixed_rows && matches!(shape, ResultShape::Rectangular(_)) {
                        let shape = if let Some(Type::Constructed(TypeName::Size(n), _)) = ty.array_size() {
                            let n = i64::try_from(*n).map_err(|_| error("result extent exceeds i64"))?;
                            ResultShape::Rectangular(vec![Expr::Integer(n)])
                        } else {
                            let Some(outer) = shape.expressions().first() else {
                                return Err(error("dynamic array result has no outer extent"));
                            };
                            ResultShape::Rectangular(vec![outer.clone()])
                        };
                        (ty.clone(), shape)
                    } else {
                        (
                            types::view_array_with_size(
                                element,
                                Type::Constructed(TypeName::SizePlaceholder, vec![]),
                                types::no_buffer(),
                            ),
                            shape,
                        )
                    }
                } else {
                    (ty.clone(), ResultShape::Rectangular(vec![]))
                };
                let (name, kind) = match strip_existentials(&source_body.ty) {
                    _ if whole_result => (declaration.name.clone(), ResultKind::Value),
                    Type::Constructed(TypeName::Record(names), _) => {
                        (names.0[index].clone(), ResultKind::RecordField)
                    }
                    Type::Constructed(TypeName::Tuple(_), _) => {
                        (format!("result_{index}"), ResultKind::TupleField)
                    }
                    _ => (declaration.name.clone(), ResultKind::Value),
                };
                let default = [crate::interface::SourceResult { index, name, kind }];
                let results = if let Some(source) = &declaration.source_entry {
                    let Some(results) = source.outputs.get(index) else {
                        return Err(error(format!(
                            "source output route {index} missing for {} ({} routes, {} outputs)",
                            declaration.name,
                            source.outputs.len(),
                            outputs.len()
                        )));
                    };
                    results.as_slice()
                } else {
                    &default
                };
                let physical_layout = storage_value_type(&output.ty);
                for result in results {
                    pipeline.source_results.push(SourceResultBinding {
                        entry: declaration
                            .source_entry
                            .as_ref()
                            .map_or(&declaration.name, |source| &source.name)
                            .clone(),
                        name: result.name.clone(),
                        kind: result.kind.clone(),
                        layout: result_layout_with_storage(&logical, &physical_layout),
                        shape: shape.clone(),
                        result: result.index,
                        pipeline_index: emission.pipeline,
                        set: binding.set,
                        binding: binding.binding,
                    });
                }
            }
        }
        let cpu_results = if declaration.entry_kind == EntryKind::Compute
            && emission.entries.is_empty()
            && output_values.iter().all(|value| host::scalar_type(&value.ty).is_some())
        {
            let parameters = emission
                .parameters
                .iter()
                .filter_map(|name| emission.body.host_values.get(name).map(|value| (*name, value.clone())))
                .collect::<StableMap<_, _>>()
                .into();
            host::cpu_body(emission.body.query(), function, &parameters, region)?
        } else {
            None
        };
        if declaration.entry_kind == EntryKind::Compute {
            emission.pending = emission.pending_observable || emission.body.active_dispatch.is_some();
        }
        emission.pending |= write_results(&mut emission.body, &output_values, &outputs)?;
        if !emission.pending && !emission.entries.is_empty() {
            let Some(generated) = emission.body.generated.as_ref() else {
                return Err(error("entry buffers missing"));
            };
            let Some(last) = emission.entries.last_mut() else {
                return Err(error("entry publication has no shader"));
            };
            last.storage_bindings = generated.declarations.clone();
            last.outputs.extend(outputs);
        } else {
            emission.flush(&LookupSet::default(), outputs, true)?;
        }
        if let Some(result) = cpu_results {
            let Some(stage) = emission.entries.last() else {
                return Err(error("CPU scalar result has no interface stage"));
            };
            let logical_results = emission.body.query().sequence("RegionYield", region)?.len();
            let result = if logical_results == 1 && output_values.len() > 1 {
                ScalarExpr::Field {
                    tuple: Box::new(result),
                    index: 0,
                }
            } else {
                result
            };
            for (i, (value, output)) in output_values.iter().zip(&stage.outputs).enumerate() {
                let Some(destination) = output.storage_binding() else {
                    return Err(error("CPU scalar output is not storage"));
                };
                let Some(ty) = host::scalar_type(&value.ty) else {
                    return Err(error("CPU scalar output has no scalar type"));
                };
                pipeline.scalar_tasks.push(ScalarTask {
                    stage: stage.name.clone(),
                    destination: scalar_source(destination),
                    offset: 0,
                    ty,
                    value: ScalarExpr::Field {
                        tuple: Box::new(result.clone()),
                        index: i,
                    },
                    replaces_dispatch: true,
                });
            }
        }
        let mut entry_ids = Vec::new();
        for mut entry in emission.entries {
            entry.id = ids.next_id();
            let owner = if let Some(source) = &declaration.source_entry {
                source.name.clone()
            } else if let Some(group) = &declaration.graphics_group {
                let Some(name) = emission.body.compiler.program.source.symbols.get(group.root) else {
                    return Err(error("graphics owner missing"));
                };
                name.clone()
            } else {
                declaration.name.clone()
            };
            source_names.insert(entry.id, owner);
            origins.insert(entry.id, symbol);
            entry_ids.push(entry.id);
            entries.push(entry);
        }
        pipeline.dispatch_loops.extend(emission.loops);
        if let Some(generated) = emission.body.generated.as_mut() {
            next_generated = next_generated.max(generated.next);
            pipeline.scalar_tasks.append(&mut generated.scalar_tasks);
        }
        if declaration.entry_kind == EntryKind::Compute {
            pipeline.pipelines.push(Pipeline::Compute(ComputePipeline {
                stages: emission.stages,
                bindings: vec![],
                default_total_threads: std::num::NonZeroU32::new(1),
            }));
            associations.push(entry_ids);
        } else {
            let Some(group) = &declaration.graphics_group else {
                return Err(error("graphics entry has no operation identity"));
            };
            let index = *graphics.entry((group.root, group.operation)).or_insert_with(|| {
                let index = pipeline.pipelines.len();
                pipeline.pipelines.push(Pipeline::Graphics(GraphicsPipeline {
                    source_operation: Some(group.operation),
                    invocation: GraphicsInvocation::default(),
                    stages: vec![],
                    bindings: vec![],
                    vertex_inputs: vec![],
                    fragment_outputs: vec![],
                }));
                associations.push(vec![]);
                invocations.push((index, &group.invocation));
                index
            });
            let Pipeline::Graphics(graphics) = &mut pipeline.pipelines[index] else {
                return Err(error("graphics pipeline identity mismatch"));
            };
            for id in entry_ids {
                let Some(entry) = entries.iter().find(|entry| entry.id == id) else {
                    return Err(error("graphics shader missing"));
                };
                graphics.stages.push(GraphicsStage {
                    entry_point: entry.name.clone(),
                    owner: source_names[&id].clone(),
                    stage: if declaration.entry_kind == EntryKind::Vertex {
                        ShaderStage::Vertex
                    } else {
                        ShaderStage::Fragment
                    },
                    uses: Default::default(),
                });
                associations[index].push(id);
            }
        }
    }

    for (index, invocation) in invocations {
        let draw = invocation.draw.try_map_buffers(|operand| {
            let (owner, slot) = match operand {
                DrawBufferOperand::Input(input) => return Ok(input.clone()),
                DrawBufferOperand::Result { entry, slot } => (*entry, *slot),
            };
            let Some(entry) = entries
                .iter()
                .find(|entry| origins.get(&entry.id) == Some(&owner) && !entry.outputs.is_empty())
            else {
                return Err(error("draw producer has no published output"));
            };
            let Some(output) = entry.outputs.get(slot) else {
                return Err(error("draw output slot missing"));
            };
            let Some(binding) = output.storage_binding() else {
                return Err(error("draw output is not storage"));
            };
            let name = output.host.target.clone().unwrap_or_else(|| {
                if entry.outputs.len() == 1 {
                    format!("{}_output", entry.name)
                } else {
                    format!("{}_output_{slot}", entry.name)
                }
            });
            Ok::<_, OptimizeError>(DrawBufferRef {
                elements: output.host.elements.clone(),
                set: binding.set,
                binding: binding.binding,
                resource: Some(
                    entry
                        .storage_bindings
                        .iter()
                        .find(|declaration| declaration.binding == binding)
                        .map_or_else(
                            || name.clone(),
                            |declaration| declaration.host.logical_resource.clone(),
                        ),
                ),
                name,
            })
        })?;
        let Pipeline::Graphics(graphics) = &mut pipeline.pipelines[index] else {
            return Err(error("graphics invocation has no pipeline"));
        };
        graphics.invocation = GraphicsInvocation {
            topology: invocation.topology,
            raster_state: invocation.raster_state.clone(),
            fragment_state: invocation.fragment_state.clone(),
            draw,
        };
    }
    for association in &associations {
        let mut accesses = LookupMap::default();
        for entry in entries.iter().filter(|entry| association.contains(&entry.id)) {
            for (&binding, &access) in &entry.stage_descriptor_storage_accesses {
                accesses
                    .entry(binding)
                    .and_modify(|previous: &mut ResourceAccess| *previous = previous.merge(access))
                    .or_insert(access);
            }
        }
        for entry in entries.iter_mut().filter(|entry| association.contains(&entry.id)) {
            entry.pipeline_storage_accesses = accesses.clone();
        }
    }
    let publications: Vec<_> = entries.iter().collect();
    pipeline
        .publish_implicit_bindings(&publications, &associations, |entry| {
            source_names[&entry.id].as_str()
        })
        .map_err(|failure| error(failure.to_string()))?;
    pipeline.publish_graphics_io(&publications, &associations);
    for pipeline in &mut pipeline.pipelines {
        if let Pipeline::Compute(pipeline) = pipeline {
            for stage in &mut pipeline.stages {
                let Some(entry) = entries.iter().find(|entry| entry.name == stage.entry_point) else {
                    return Err(error("published stage has no shader"));
                };
                for (i, binding) in pipeline.bindings.iter().enumerate() {
                    if let Some((set, slot)) = binding.slot() {
                        if let Some(&access) =
                            entry.stage_descriptor_storage_accesses.get(&BindingRef::new(set, slot))
                        {
                            stage.uses.record(
                                i,
                                match access {
                                    ResourceAccess::Read => crate::host::Access::ReadOnly,
                                    ResourceAccess::Write => crate::host::Access::WriteOnly,
                                    ResourceAccess::ReadWrite => crate::host::Access::ReadWrite,
                                },
                            );
                        }
                    } else {
                        stage.uses.record(i, crate::host::Access::ReadOnly);
                    }
                }
            }
            reconcile_storage_binding_access(
                &mut pipeline.bindings,
                pipeline.stages.iter().map(|stage| &stage.uses),
            );
        } else if let Pipeline::Graphics(pipeline) = pipeline {
            for stage in &mut pipeline.stages {
                let Some(entry) = entries.iter().find(|entry| entry.name == stage.entry_point) else {
                    return Err(error("graphics stage has no shader"));
                };
                for (i, binding) in pipeline.bindings.iter().enumerate() {
                    if let Some((set, slot)) = binding.slot() {
                        if let Some(&access) =
                            entry.stage_descriptor_storage_accesses.get(&BindingRef::new(set, slot))
                        {
                            stage.uses.record(
                                i,
                                match access {
                                    ResourceAccess::Read => crate::host::Access::ReadOnly,
                                    ResourceAccess::Write => crate::host::Access::WriteOnly,
                                    ResourceAccess::ReadWrite => crate::host::Access::ReadWrite,
                                },
                            );
                        } else if !matches!(binding, crate::host::Binding::StorageBuffer { .. }) {
                            stage.uses.record(i, crate::host::Access::ReadOnly);
                        }
                    }
                }
            }
            reconcile_storage_binding_access(
                &mut pipeline.bindings,
                pipeline.stages.iter().map(|stage| &stage.uses),
            );
        }
    }
    let mut dependencies = Vec::new();
    for (pipeline_index, declaration) in pipeline.pipelines.iter().enumerate() {
        if let Pipeline::Compute(compute) = declaration {
            for (stage_index, stage) in compute.stages.iter().enumerate() {
                dependencies.extend(
                    stage.dependencies.iter().map(|&before| (before, (pipeline_index, stage_index))),
                );
            }
        }
    }
    let mut frame_graph =
        FrameGraph::from_selected_pipelines(&pipeline.pipelines, &dependencies).map_err(error)?;
    // Concrete aliases are known here, including reads between writes to a render target.
    let mut writers = LookupMap::<(&str, usize), usize>::default();
    let mut readers = LookupMap::<(&str, usize), Vec<usize>>::default();
    for index in 0..frame_graph.passes.len() {
        let pass = &frame_graph.passes[index];
        let owner = match &pipeline.pipelines[pass.pipeline_index] {
            Pipeline::Compute(compute) => compute.stages[pass.stage_index].owner.as_str(),
            Pipeline::Graphics(graphics) => {
                graphics.stages.first().ok_or_else(|| error("draw has no shader stage"))?.owner.as_str()
            }
        };
        let mut before = Vec::new();
        for access in &pass.reads {
            let key = (owner, access.resource);
            if let Some(&writer) = writers.get(&key) {
                before.push(writer);
            }
            readers.entry(key).or_default().push(index);
        }
        for access in &pass.writes {
            let key = (owner, access.resource);
            if let Some(writer) = writers.insert(key, index).filter(|&writer| writer != index) {
                before.push(writer);
            }
            before.extend(
                readers.remove(&key).unwrap_or_default().into_iter().filter(|&reader| reader != index),
            );
        }
        let dependencies = &mut frame_graph.passes[index].depends_on;
        dependencies.extend(before);
        dependencies.sort_unstable();
        dependencies.dedup();
    }
    pipeline.frame_graph = frame_graph;
    Ok(Elaborated::from_parts(
        std::mem::take(&mut compiler.functions),
        entries.into_iter().map(EntryPoint::into_shader).collect(),
        BackendGlobal { pipeline, target },
    ))
}

struct EntryEmission<'a, 'p, 'source> {
    body: Body<'a, 'p, 'source>,
    templates: Vec<(EntryInput, Type)>,
    params: Vec<Value>,
    parameter_inputs: Vec<Vec<usize>>,
    inputs: Vec<EntryInput>,
    parameters: LookupSet<Value>,
    saved: StableMap<Value, Location>,
    name: String,
    owner: String,
    kind: EntryKind,
    span: crate::ast::Span,
    pipeline: usize,
    invocation: ((u32, u32, u32), DispatchSize),
    pending: bool,
    pending_observable: bool,
    restore_names: Option<LookupSet<Value>>,
    dispatch_override: Option<((u32, u32, u32), DispatchSize)>,
    stages: Vec<ComputeStage>,
    entries: Vec<EntryPoint>,
    loops: Vec<DispatchLoop>,
}

impl EntryEmission<'_, '_, '_> {
    fn restored_results(&mut self, region: Value, count: usize) -> Result<Vec<Typed>, OptimizeError> {
        let mut values = self.body.results(region)?;
        if values.len() == 1 && count != 1 {
            values = self.body.unpack(values.remove(0))?;
        }
        if values.len() != count {
            return Err(error("output transport changed value arity"));
        }
        Ok(values)
    }

    fn prepare_array_outputs(
        &mut self,
        results: Value,
        mut values: Vec<Typed>,
        plans: &[Output],
    ) -> Result<(Vec<Typed>, Vec<Option<ResultShape>>), OptimizeError> {
        if plans.len() != values.len() {
            return Err(error("output plan arity mismatch"));
        }
        let reads_results = plans.iter().any(|plan| {
            matches!(
                plan,
                Output::Array {
                    mode: ArrayOutput::Pack,
                    ..
                }
            )
        });
        if reads_results && self.body.active_dispatch.is_some() {
            let live = live_names(self.body.query(), &[], results)?;
            self.flush(&live, vec![], true)?;
            values = self.restored_results(results, values.len())?;
        }
        let mut counts = Vec::new();
        let mut shapes = vec![None; values.len()];
        for (index, value) in values.iter().enumerate() {
            let Output::Array { rank, leaf, mode, .. } = &plans[index] else {
                counts.push(None);
                continue;
            };
            if *mode == ArrayOutput::Direct {
                counts.push(None);
                continue;
            }
            let rank = *rank;
            let rank_u32 = u32::try_from(rank).map_err(|_| error("output rank exceeds u32"))?;
            let (binding, storage) = self.body.device_buffer(
                &types::i32(),
                Expr::Integer(i64::from(rank_u32) * 4),
                Body::number(rank_u32),
            )?;
            let zero = self.body.literal(0, &types::i32())?;
            let totals = self.body.array_totals(value.clone(), 0, vec![zero; rank])?;
            let publish = |body: &mut Body<'_, '_, '_>| {
                for (i, total) in totals.into_iter().enumerate() {
                    body.store(storage.clone(), Body::number(i as u32), total)?;
                }
                Ok(())
            };
            if let Some(leader) = publication_leader(&mut self.body)? {
                self.body.when(leader, publish)?;
            } else {
                publish(&mut self.body)?;
            }
            self.pending = true;
            self.pending_observable = true;
            if *mode == ArrayOutput::Count {
                let length = Expr::Scalar(ScalarExpr::Read {
                    source: scalar_source(binding),
                    offset: 0,
                    ty: ScalarType::I32,
                });
                self.body.set_shape(&value, vec![length.clone()])?;
                shapes[index] = Some(ResultShape::Rectangular(vec![length]));
                counts.push(None);
                continue;
            }
            counts.push(Some((
                binding,
                rank,
                leaf.clone(),
                output_shape(&self.body, value).ok(),
            )));
        }
        if counts.iter().all(Option::is_none) {
            return Ok((values, shapes));
        }
        self.pending = true;
        self.pending_observable = true;
        let live = live_names(self.body.query(), &[], results)?;
        self.flush(&live, vec![], true)?;
        values = self.restored_results(results, values.len())?;
        let mut materialized = false;
        for (i, plan) in counts.into_iter().enumerate() {
            let Some((binding, rank, leaf, known_shape)) = plan else {
                if let Some(ResultShape::Rectangular(shape)) = &shapes[i] {
                    self.body.set_shape(&values[i], shape.clone())?;
                }
                continue;
            };
            let totals = storage_view(&mut self.body, binding, types::i32(), Body::number(rank as u32))?;
            let lengths = (0..rank)
                .map(|depth| {
                    if let Some(extent) = known_shape.as_ref().and_then(|shape| shape.get(depth)) {
                        return extent.clone();
                    }
                    Expr::Scalar(ScalarExpr::Read {
                        source: scalar_source(binding),
                        offset: depth as u32 * 4,
                        ty: ScalarType::I32,
                    })
                })
                .collect::<Vec<_>>();
            materialized = true;
            let mut offsets = Vec::new();
            let mut offset_sources = Vec::new();
            for depth in 0..rank - 1 {
                let length = self.body.index(totals.clone(), Body::number(depth as u32))?;
                let length = self.body.binary(BinaryOperator::Add, length, Body::number(1))?;
                let bytes = Expr::Multiply(
                    Box::new(Expr::Add(
                        Box::new(lengths[depth].clone()),
                        Box::new(Expr::Integer(1)),
                    )),
                    Box::new(Expr::Integer(4)),
                );
                let (binding, array) = self.body.device_buffer(&types::i32(), bytes, length)?;
                self.body.store(array.clone(), Body::number(0), Body::number(0))?;
                offsets.push(array);
                offset_sources.push(scalar_source(binding));
            }
            let length = self.body.index(totals, Body::number((rank - 1) as u32))?;
            let Some(stride) = storage_elem_stride(&storage_value_type(&leaf)) else {
                return Err(error("output leaf has no storage stride"));
            };
            let bytes = Expr::Multiply(
                Box::new(lengths[rank - 1].clone()),
                Box::new(Expr::Integer(stride.into())),
            );
            let (_, payload) = self.body.device_buffer(&leaf, bytes, length)?;
            self.body.set_shape(&payload, vec![lengths[rank - 1].clone()])?;
            let zero = self.body.literal(0, &types::i32())?;
            self.body.flatten_array(values[i].clone(), &payload, &offsets, 0, vec![zero; rank])?;
            values[i] = payload;
            shapes[i] = Some(if rank == 1 {
                ResultShape::Rectangular(lengths)
            } else {
                ResultShape::Segmented {
                    offsets: offset_sources,
                    elements: lengths[rank - 1].clone(),
                }
            });
        }
        self.pending |= materialized;
        self.pending_observable |= materialized;
        Ok((values, shapes))
    }

    fn restart(&mut self) -> Result<(), OptimizeError> {
        self.body.reset_shader();
        self.inputs.clear();
        if self.params.len() != self.parameter_inputs.len() {
            return Err(error(
                "entry parameter components do not match its logical signature",
            ));
        }
        for (&parameter, components) in self.params.iter().zip(&self.parameter_inputs) {
            let mut values = Vec::new();
            let mut host = None;
            for &index in components {
                let (input, ty) = &self.templates[index];
                let (value, scalar) = bind_input(&mut self.body, input.clone(), ty, &mut self.inputs)?;
                values.push(value);
                host = scalar;
            }
            let value = if values.len() == 1 {
                values.remove(0)
            } else {
                host = None;
                self.body.tuple(values)?
            };
            self.body.bind(&[parameter], vec![value])?;
            if let Some(host) = host {
                if let Some((name, _)) = super::runtime_binding(self.body.query(), parameter)? {
                    self.body.host_values.insert(name, host);
                }
            }
        }
        self.parameters = self.body.values.keys().copied().collect();
        restore(&mut self.body, &self.saved, self.restore_names.take().as_ref())
    }

    fn flush(
        &mut self,
        live: &LookupSet<Value>,
        outputs: Vec<EntryOutput>,
        force: bool,
    ) -> Result<(), OptimizeError> {
        if !self.pending && !force {
            return Ok(());
        }
        if outputs.is_empty() {
            save(&mut self.body, live, &self.parameters, &mut self.saved)?;
        }
        let (local_size, mut dispatch_size) = if let Some(dispatch) = self.dispatch_override.take() {
            dispatch
        } else if let Some(statement) = self.body.active_dispatch {
            let dispatch = *self.body.compiler.execution.dispatch(statement)?;
            let size = if let Some((x, y, z)) = dispatch.grid {
                DispatchSize::Fixed {
                    x,
                    y,
                    z,
                    explicit: true,
                }
            } else if let Some(groups) = dispatch.groups {
                let [x, y, z] = groups.map(|index| {
                    host::guarded_index(
                        self.body.query(),
                        &self.body.host_values,
                        index,
                        self.body.guard.as_ref(),
                    )
                });
                let groups = [x?, y?, z?];
                let elements = Expr::Multiply(
                    Box::new(Expr::Multiply(
                        Box::new(groups[0].clone()),
                        Box::new(groups[1].clone()),
                    )),
                    Box::new(groups[2].clone()),
                );
                DispatchSize::Computed { elements, groups }
            } else {
                let query = self.body.query();
                let Some(launch) = self.body.compiler.execution.launches.get(&statement) else {
                    return Err(error("dispatch has no physical launch plan"));
                };
                let mut elements = Expr::Integer(1);
                for &extent in &launch.extents {
                    let length = host::guarded_index(
                        query,
                        &self.body.host_values,
                        extent,
                        self.body.guard.as_ref(),
                    )?;
                    elements = Expr::Multiply(
                        Box::new(elements),
                        Box::new(Expr::Max(Box::new(length), Box::new(Expr::Integer(0)))),
                    );
                }
                let groups = match launch.grouping {
                    Grouping::Single => Expr::Integer(1),
                    Grouping::PerElement => elements.clone(),
                    Grouping::PerWorkgroup => Expr::Ceiling(
                        Box::new(elements.clone()),
                        Box::new(Expr::Integer(i64::from(dispatch.width.get()))),
                    ),
                };
                let groups = Expr::Max(
                    Box::new(Expr::Integer(launch.minimum)),
                    Box::new(Expr::Min(Box::new(groups), Box::new(Expr::Integer(65_535)))),
                );
                DispatchSize::Computed {
                    elements,
                    groups: [groups, Expr::Integer(1), Expr::Integer(1)],
                }
            };
            (dispatch.local_size, size)
        } else {
            self.invocation.clone()
        };
        if let Some(condition) = &self.body.guard {
            let enabled = Expr::Scalar(ScalarExpr::If {
                condition: Box::new(condition.clone()),
                yes: Box::new(ScalarExpr::I32(1)),
                no: Box::new(ScalarExpr::I32(0)),
            });
            let (elements, mut groups) = match dispatch_size {
                DispatchSize::Fixed { x, y, z, .. } => (
                    Expr::Integer(i64::from(x)),
                    [
                        Expr::Integer(i64::from(x)),
                        Expr::Integer(i64::from(y)),
                        Expr::Integer(i64::from(z)),
                    ],
                ),
                DispatchSize::Computed { elements, groups } => (elements, groups),
            };
            groups[0] = Expr::Multiply(Box::new(groups[0].clone()), Box::new(enabled));
            dispatch_size = DispatchSize::Computed { elements, groups };
        }
        let Some(generated) = self.body.generated.as_ref() else {
            return Err(error("entry buffers missing"));
        };
        let inputs = self.inputs.clone();
        let storage_bindings = generated.declarations.clone();
        let mut accesses = std::mem::take(&mut self.body.storage_accesses);
        for input in &inputs {
            if matches!(
                input.kind,
                EntryInputKind::Texture { .. }
                    | EntryInputKind::Sampler { .. }
                    | EntryInputKind::Uniform { .. }
            ) {
                if let Some(binding) = input.descriptor_binding() {
                    accesses.entry(binding).or_insert(ResourceAccess::Read);
                }
            }
        }
        if self.kind == EntryKind::Compute && !self.pending && !outputs.is_empty() {
            dispatch_size = DispatchSize::Fixed {
                x: 0,
                y: 1,
                z: 1,
                explicit: true,
            };
        }
        let function_body = self.body.finish_shader()?;
        let phase = match self.kind {
            EntryKind::Vertex => "vertex",
            EntryKind::Fragment => "fragment",
            _ => "compute",
        };
        let name = if self.entries.is_empty() {
            format!("{}_{phase}", self.name)
        } else {
            format!("{}_{phase}_{}", self.name, self.entries.len())
        };
        if let Some(generated) = self.body.generated.as_mut() {
            for task in &mut generated.scalar_tasks {
                if task.stage.is_empty() {
                    task.stage = name.clone();
                }
            }
        }
        let execution_model = match self.kind {
            EntryKind::Compute => ExecutionModel::Compute { local_size },
            EntryKind::Vertex => ExecutionModel::Vertex,
            EntryKind::Fragment => ExecutionModel::Fragment,
            EntryKind::Root => return Err(error("unextracted graphics entry")),
        };
        if self.kind == EntryKind::Compute {
            self.stages.push(ComputeStage {
                dependencies: self
                    .stages
                    .last()
                    .map(|_| (self.pipeline, self.stages.len() - 1))
                    .into_iter()
                    .collect(),
                entry_point: name.clone(),
                owner: self.owner.clone(),
                workgroup_size: local_size,
                dispatch_size,
                uses: Default::default(),
            });
        }
        self.entries.push(EntryPoint {
            id: crate::EntryId(0),
            name,
            body: function_body,
            execution_model,
            inputs,
            parameter_inputs: self.parameter_inputs.clone(),
            outputs,
            storage_bindings,
            stage_descriptor_storage_accesses: accesses.clone(),
            pipeline_storage_accesses: accesses,
            span: self.span,
        });
        self.pending = false;
        self.pending_observable = false;
        self.restart()
    }

    fn statements(&mut self, region: Value, after: &LookupSet<Value>) -> Result<(), OptimizeError> {
        let Some(steps) = self.body.compiler.execution.regions.get(&region) else {
            return Err(error("host region has no physical schedule"));
        };
        let steps = steps.clone();
        for step in &steps {
            let statement = step.operation;
            let mut live = step.live.clone();
            live.extend(after.iter().copied());
            let pattern = self.body.query().sequence("Output", statement)?;
            if step.boundary {
                let mut before = statement_names(self.body.query(), &[statement])?;
                before.extend(live.iter().copied());
                self.flush(&before, vec![], false)?;
            }
            match &step.kind {
                StepKind::Dispatch { terminal, epilogue } => {
                    self.materialize_cpu_captures(&step.cpu_captures)?;
                    self.body.active_dispatch = Some(statement);
                    if let Some(plan) = self.body.compiler.execution.collectives.get(&statement).cloned() {
                        self.emit_collective(
                            statement,
                            &plan,
                            &live,
                            *terminal,
                            epilogue.as_ref().map(|epilogue| epilogue.operation),
                            epilogue.as_ref().map_or(&[], |epilogue| epilogue.preparation.as_slice()),
                        )?;
                    } else {
                        self.body.statement(statement)?;
                        if let Some(epilogue) = epilogue {
                            self.body.statements(&epilogue.preparation)?;
                            let domain =
                                self.body.query().required("OperationDomain", epilogue.operation)?;
                            let extent = self.body.query().required("DomainExtent", (domain, 0_i64))?;
                            let count = self.body.index_expression(extent)?;
                            self.scan_epilogue(
                                epilogue.operation,
                                (Body::number(0), count),
                                epilogue.cooperative,
                            )?;
                        }
                        self.pending = true;
                        if !terminal {
                            self.flush(&live, vec![], true)?;
                        }
                    }
                }
                StepKind::Branch => self.branch(statement, &live)?,
                StepKind::Loop => self.loop_body(statement, &live)?,
                StepKind::ScalarLoop => self.cpu_loop(statement)?,
                StepKind::Accumulator => {
                    let lane = self.body.query().required("ControlRegion", (statement, 0_i64))?;
                    let mut needed = live_names(self.body.query(), &[], lane)?;
                    needed.extend(names_in(self.body.query(), "OperationUse", statement)?);
                    needed.extend(live.iter().copied());
                    self.statements(lane, &needed)?;
                    let mut values = Vec::new();
                    for group in 0..crate::egglog::analysis::count(
                        self.body.query(),
                        "AccumulatorGroupCount",
                        statement,
                    )? {
                        for slot in 0..crate::egglog::analysis::count(
                            self.body.query(),
                            "AccumulatorArrayCount",
                            (statement, group),
                        )? {
                            let name =
                                self.body.query().required("AccumulatorArray", (statement, group, slot))?;
                            values.push(self.body.named(name)?);
                        }
                    }
                    values.extend(self.body.results(lane)?);
                    let values = self.body.scope_outputs(&pattern, values)?;
                    self.body.bind(&pattern, values)?;
                    self.pending = true;
                }
                StepKind::Shader => {
                    self.statement_effects(statement)?;
                    self.body.statement(statement)?;
                    self.pending = true;
                }
            }
        }
        Ok(())
    }

    fn statement_effects(&mut self, statement: Value) -> Result<(), OptimizeError> {
        host::remember(
            Query(&self.body.compiler.program.graph),
            self.body.function,
            &mut self.body.host_values,
            statement,
        )?;
        let query = self.body.query();
        let mut memory = false;
        for table in ["OperationRead", "OperationWrite"] {
            query.for_each(table, |row| {
                memory |= row[0] == statement;
                Ok(())
            })?;
        }
        let trap = query.contains("OperationTrap", statement)?;
        self.pending_observable |= memory || trap;
        Ok(())
    }

    fn cpu_loop(&mut self, loop_: Value) -> Result<(), OptimizeError> {
        let pattern = self.body.query().sequence("Output", loop_)?;
        let mut outputs = Vec::new();
        for &binding in &pattern {
            let Some((name, ty)) = super::runtime_binding(self.body.query(), binding)? else {
                return Err(error("planned host loop has no scalar representation"));
            };
            let Some(scalar) = host::scalar_type(&ty) else {
                return Err(error("planned host loop has no scalar representation"));
            };
            outputs.push((name, ty, scalar));
        }
        if outputs.is_empty() {
            return Err(error("planned host loop has no scalar representation"));
        }
        let Some(value) = host::cpu_loop(
            self.body.query(),
            self.body.function,
            &self.body.host_values,
            loop_,
        )?
        else {
            return Err(error("planned host loop has no scalar representation"));
        };
        for (i, (name, ty, scalar)) in outputs.into_iter().enumerate() {
            let expression = ScalarExpr::Field {
                tuple: Box::new(value.clone()),
                index: i,
            };
            self.publish_cpu_value(name, ty, scalar, expression)?;
        }
        Ok(())
    }

    fn publish_cpu_value(
        &mut self,
        name: Value,
        ty: Type,
        scalar: ScalarType,
        value: ScalarExpr,
    ) -> Result<(), OptimizeError> {
        let slot = self.scalar_slot(&ty)?;
        let location = Location::Buffer(slot, ty);
        let loaded = restore_value(&mut self.body, &location)?;
        self.body.values.insert(name, loaded);
        self.saved.insert(name, location);
        self.body.host_values.insert(name, value.clone());
        let Some(generated) = self.body.generated.as_mut() else {
            return Err(error("CPU value has no entry buffers"));
        };
        generated.scalar_tasks.push(ScalarTask {
            stage: String::new(),
            destination: scalar_source(slot),
            offset: 0,
            ty: scalar,
            value,
            replaces_dispatch: false,
        });
        Ok(())
    }

    fn scalar_slot(&mut self, ty: &Type) -> Result<BindingRef, OptimizeError> {
        let Some(bytes) = storage_elem_stride(&storage_value_type(ty)) else {
            return Err(error("dispatch witness has no storage size"));
        };
        let (slot, _) = self.body.device_buffer(ty, Expr::Integer(bytes.into()), Body::number(1))?;
        Ok(slot)
    }

    fn materialize_cpu_captures(&mut self, names: &[Value]) -> Result<(), OptimizeError> {
        for &name in names {
            if matches!(self.saved.get(&name), Some(Location::Buffer(..))) {
                continue;
            }
            let value = self.body.named(name)?;
            let Some(scalar) = host::scalar_type(&value.ty) else {
                return Err(error("planned CPU capture is not scalar"));
            };
            let Some(expression) = self.body.host_values.get(&name).cloned() else {
                return Err(error("planned CPU capture has no host expression"));
            };
            self.publish_cpu_value(name, value.ty, scalar, expression)?;
        }
        Ok(())
    }

    fn capture(&mut self, name: Value, value: Typed) -> Result<Location, OptimizeError> {
        let saved = save_named(&mut self.body, name, value, true)?;
        self.pending = true;
        self.pending_observable = true;
        Ok(saved)
    }

    fn loop_body(&mut self, expression: Value, after: &LookupSet<Value>) -> Result<(), OptimizeError> {
        let pattern = self.body.query().sequence("Output", expression)?;
        let region = self.body.query().required("ControlRegion", (expression, 0_i64))?;
        let merges = self.body.query().sequence("RegionParameter", region)?;
        let counted = self.body.query().contains("CountedLoopOperation", expression)?;
        let mut before = names_in(self.body.query(), "OperationUse", expression)?;
        before.extend(after.iter().copied());
        before.extend(live_names(
            self.body.query(),
            &self.body.query().operations(region)?,
            region,
        )?);
        self.flush(&before, vec![], false)?;
        let incoming = self.saved.clone();
        let incoming_host = self
            .body
            .host_values
            .iter()
            .map(|(&name, expression)| (name, expression.clone()))
            .collect::<StableMap<_, _>>();
        let mut carried_values = Vec::new();
        for (index, (merge, output)) in merges.into_iter().zip(&pattern).enumerate() {
            let parameter = merge;
            let argument = self.body.query().required("EdgeArgument", (expression, 0_i64, index as i64))?;
            if let Some((name, ty)) = super::runtime_binding(self.body.query(), parameter)? {
                let Some(value) = self.body.operand(argument)? else {
                    return Err(error("loop initial value missing"));
                };
                let Some((output, _)) = super::runtime_binding(self.body.query(), *output)? else {
                    return Err(error("loop output missing"));
                };
                let host_length = if self
                    .body
                    .query()
                    .contains("LoopPreservesOuterLength", (expression, index as i64))?
                {
                    self.body.host_array_length(&value).ok()
                } else {
                    None
                };
                let initial = self.capture(self.body.query().name(argument)?, value)?;
                let slots = saved_slots(&initial)?;
                let mut replacements = LookupMap::default();
                for (binding, element) in &slots {
                    let bytes = Expr::BufferLength {
                        source: scalar_source(*binding),
                        stride: 1,
                    };
                    let (slot, _) = self.body.device_buffer(element, bytes, Body::number(0))?;
                    replacements.insert(*binding, slot);
                }
                let mut current = remap_saved(&initial, &replacements);
                retain_fixed_shape(&mut current, &ty)?;
                if let (Some(length), Location::Array(array)) = (host_length, &mut current) {
                    if let Some((_, extent)) = array.dimensions.first_mut() {
                        extent.host = Some(length);
                    }
                }
                carried_values.push((name, output, initial, current, slots));
            }
        }
        let control = if counted {
            let Some(count) = host::scalar_index(
                self.body.query(),
                &self.body.host_values,
                self.body.query().required("LoopCount", expression)?,
            )?
            else {
                return Err(error("loop count has no host value"));
            };
            let slot = self.scalar_slot(&types::i32())?;
            let zero = self.body.literal(0, &types::i32())?;
            self.body.values.insert(self.body.query().required("LoopIndex", expression)?, zero);
            before.insert(self.body.query().required("LoopIndex", expression)?);
            DispatchLoopControl::Counted {
                count,
                index: scalar_source(slot),
            }
        } else {
            let condition = self.body.query().required("LoopCondition", expression)?;
            let Some((_, _, _, Location::Buffer(slot, _), _)) =
                carried_values.iter().find(|(name, ..)| *name == condition)
            else {
                return Err(error("loop predicate is not a scalar carry"));
            };
            DispatchLoopControl::While {
                condition: ScalarExpr::Read {
                    source: scalar_source(*slot),
                    offset: 0,
                    ty: ScalarType::Bool,
                },
            }
        };
        self.flush(&before, vec![], true)?;
        let setup = self.stages.len() - 1;
        for (name, _, _, current, _) in &carried_values {
            self.install_saved(*name, current.clone());
        }
        if counted {
            let DispatchLoopControl::Counted {
                index: ScalarSource::Binding { set, binding },
                ..
            } = &control
            else {
                return Err(error("loop ordinal binding missing"));
            };
            self.install_saved(
                self.body.query().required("LoopIndex", expression)?,
                Location::Buffer(BindingRef::new(*set, *binding), types::i32()),
            );
        }
        self.restart()?;
        if counted {
            let ordinal = self.body.named(self.body.query().required("LoopIndex", expression)?)?;
            let start = self.body.index_expression(self.body.query().required("LoopStart", expression)?)?;
            let step = self.body.index_expression(self.body.query().required("LoopStep", expression)?)?;
            let delta = self.body.binary(BinaryOperator::Multiply, ordinal, step)?;
            let value = self.body.binary(BinaryOperator::Add, start, delta)?;
            self.body.values.insert(self.body.query().required("LoopIndex", expression)?, value);
            let Some(ordinal) =
                self.body.host_values.get(&self.body.query().required("LoopIndex", expression)?).cloned()
            else {
                return Err(error("loop ordinal missing"));
            };
            let Some(start) = host::scalar_index(
                self.body.query(),
                &self.body.host_values,
                self.body.query().required("LoopStart", expression)?,
            )?
            else {
                return Err(error("loop start missing"));
            };
            let Some(step) = host::scalar_index(
                self.body.query(),
                &self.body.host_values,
                self.body.query().required("LoopStep", expression)?,
            )?
            else {
                return Err(error("loop step missing"));
            };
            self.body.host_values.insert(
                self.body.query().required("LoopIndex", expression)?,
                ScalarExpr::Apply {
                    op: "add".into(),
                    ty: ScalarType::I32,
                    args: vec![
                        start,
                        ScalarExpr::Apply {
                            op: "mul".into(),
                            ty: ScalarType::I32,
                            args: vec![ordinal, step],
                        },
                    ],
                },
            );
        }
        let mut live = live_names(self.body.query(), &[], region)?;
        live.extend(after.iter().copied());
        self.statements(region, &live)?;
        let values = self.body.results(region)?;
        if values.len() != carried_values.len() {
            return Err(error("loop carry arity mismatch"));
        }
        let mut carried = Vec::new();
        let results = self.body.query().sequence("RegionYield", region)?;
        for (((_, _, _, current, initial_slots), value), result) in
            carried_values.iter().zip(values).zip(results)
        {
            let next = self.capture(self.body.query().name(result)?, value)?;
            let current_slots = saved_slots(current)?;
            let next_slots = saved_slots(&next)?;
            if next_slots.len() != current_slots.len() {
                return Err(error("loop value changed its storage structure"));
            }
            for (((binding, ty), (initial, _)), (next, next_ty)) in
                current_slots.into_iter().zip(initial_slots).zip(next_slots)
            {
                if erase_buffers(&ty) != erase_buffers(&next_ty) {
                    return Err(error("loop value changed its physical element type"));
                }
                carried.push(LoopCarry {
                    binding: scalar_source(binding),
                    initial: BufferSource::Binding(scalar_source(*initial)),
                    next: BufferSource::Binding(scalar_source(next)),
                });
            }
        }
        self.flush(after, vec![], true)?;
        let completion = self.stages.len();
        self.saved = incoming;
        self.body.host_values = incoming_host.into();
        for (_, output, _, current, _) in carried_values {
            self.install_saved(output, current);
        }
        self.restart()?;
        self.flush(after, vec![], true)?;
        self.loops.push(DispatchLoop {
            pipeline: self.pipeline,
            setup,
            body: (setup + 1..completion).collect(),
            test: vec![],
            completion,
            control,
            carried,
        });
        Ok(())
    }

    fn install_saved(&mut self, name: Value, saved: Location) {
        if let Location::Buffer(binding, ty) = &saved {
            if let Some(ty) = host::scalar_type(ty) {
                self.body.host_values.insert(
                    name,
                    ScalarExpr::Read {
                        source: scalar_source(*binding),
                        offset: 0,
                        ty,
                    },
                );
            }
        }
        self.saved.insert(name, saved);
    }

    fn branch(&mut self, operation: Value, after: &LookupSet<Value>) -> Result<(), OptimizeError> {
        let pattern = self.body.query().sequence("Output", operation)?;
        let yes = self.body.query().required("ControlRegion", (operation, 0_i64))?;
        let no = self.body.query().required("ControlRegion", (operation, 1_i64))?;
        let condition_name =
            self.body.query().name(self.body.query().required("Input", (operation, 0_i64))?)?;
        let mut before = live_names(self.body.query(), &self.body.query().operations(yes)?, yes)?;
        before.extend(live_names(
            self.body.query(),
            &self.body.query().operations(no)?,
            no,
        )?);
        before.insert(condition_name);
        before.extend(after.iter().copied());
        self.flush(&before, vec![], false)?;
        let Some(condition) = host::atom(self.body.query(), &self.body.host_values, condition_name)? else {
            return Err(error("branch condition has no host value"));
        };
        let incoming = self.saved.clone();
        let incoming_host = self
            .body
            .host_values
            .iter()
            .map(|(&name, expression)| (name, expression.clone()))
            .collect::<StableMap<_, _>>();
        let enclosing = self.body.guard.clone();
        let mut arms = Vec::new();
        for (block, take_yes) in [(yes, true), (no, false)] {
            self.saved = incoming.clone();
            self.body.host_values = incoming_host.clone().into();
            self.restart()?;
            let enabled = if take_yes {
                condition.clone()
            } else {
                ScalarExpr::If {
                    condition: Box::new(condition.clone()),
                    yes: Box::new(ScalarExpr::Bool(false)),
                    no: Box::new(ScalarExpr::Bool(true)),
                }
            };
            self.body.guard = Some(if let Some(outer) = &enclosing {
                ScalarExpr::If {
                    condition: Box::new(outer.clone()),
                    yes: Box::new(enabled),
                    no: Box::new(ScalarExpr::Bool(false)),
                }
            } else {
                enabled
            });
            let mut live = live_names(self.body.query(), &[], block)?;
            live.extend(after.iter().copied());
            self.statements(block, &live)?;
            let values = self.body.results(block)?;
            let saved = values
                .into_iter()
                .zip(self.body.query().sequence("RegionYield", block)?)
                .map(|(value, name)| self.capture(self.body.query().name(name)?, value))
                .collect::<Result<Vec<_>, _>>()?;
            self.flush(after, vec![], true)?;
            arms.push(saved);
        }
        self.body.guard = enclosing;
        self.saved = incoming;
        self.body.host_values = incoming_host.into();
        self.restart()?;
        let [yes, no]: [Vec<Location>; 2] = arms.try_into().map_err(|_| error("branch arms missing"))?;
        if yes.len() != no.len() {
            return Err(error("branch arity mismatch"));
        }
        let mut selected = Vec::new();
        let test = self.body.atom(condition_name)?;
        for ((yes, no), &binding) in yes.iter().zip(&no).zip(&pattern) {
            let name = self.body.query().name(binding)?;
            let Some(plan) = self.body.compiler.execution.storage.transfers.get(&name).cloned() else {
                return Err(error("branch result has no storage plan"));
            };
            selected.push(self.merge_saved(&condition, test.clone(), yes, no, &plan)?);
        }
        self.body.bind(&pattern, selected)?;
        self.pending = true;
        self.pending_observable = true;
        Ok(())
    }

    fn merge_saved(
        &mut self,
        condition: &ScalarExpr,
        test: Typed,
        yes: &Location,
        no: &Location,
        plan: &Transfer,
    ) -> Result<Typed, OptimizeError> {
        if let Transfer::Product(plans) = plan {
            let (Location::Product(yes), Location::Product(no)) = (yes, no) else {
                return Err(error("branch product does not match its storage plan"));
            };
            if yes.len() != no.len() || yes.len() != plans.len() {
                return Err(error("branch product arity mismatch"));
            }
            let mut fields = Vec::new();
            for ((yes, no), plan) in yes.iter().zip(no).zip(plans) {
                fields.push(self.merge_saved(condition, test.clone(), yes, no, plan)?);
            }
            return self.body.tuple(fields);
        }
        if matches!(plan, Transfer::Array { .. } | Transfer::Virtual) {
            let left = saved_slots(yes)?;
            let right = saved_slots(no)?;
            if left.len() != right.len() {
                return Err(error("branch arrays have different storage structures"));
            }
            let mut yes_bindings = LookupMap::default();
            let mut no_bindings = LookupMap::default();
            let mut copies = Vec::new();
            for ((a, a_ty), (b, b_ty)) in left.into_iter().zip(right) {
                let element = storage_value_type(&erase_buffers(&a_ty));
                if element != storage_value_type(&erase_buffers(&b_ty)) {
                    return Err(error("branch arrays have different physical element types"));
                }
                let binding = if a == b {
                    a
                } else {
                    let bytes = Expr::Scalar(ScalarExpr::If {
                        condition: Box::new(condition.clone()),
                        yes: Box::new(ScalarExpr::BufferLength {
                            source: scalar_source(a),
                            stride: 1,
                        }),
                        no: Box::new(ScalarExpr::BufferLength {
                            source: scalar_source(b),
                            stride: 1,
                        }),
                    });
                    let (binding, _) = self.body.device_buffer(&element, bytes, Body::number(0))?;
                    copies.push((a, b, binding, element));
                    binding
                };
                yes_bindings.insert(a, binding);
                no_bindings.insert(b, binding);
            }
            let mut result = remap_saved(yes, &yes_bindings);
            if let (Location::Array(a), Location::Array(b)) = (&result, remap_saved(no, &no_bindings)) {
                if a.element != b.element
                    || a.dimensions.iter().map(|(ty, _)| ty).ne(b.dimensions.iter().map(|(ty, _)| ty))
                {
                    return Err(error("branch arrays have incompatible nested alias structures"));
                }
            }
            if let (Location::Array(result), Location::Array(a), Location::Array(b)) =
                (&mut result, yes, no)
            {
                for ((_, extent), ((_, a), (_, b))) in
                    result.dimensions.iter_mut().zip(a.dimensions.iter().zip(&b.dimensions))
                {
                    extent.host = match (a.host(), b.host()) {
                        (Some(a), Some(b)) if a == b => Some(a),
                        (Some(a), Some(b)) => Some(Expr::Scalar(ScalarExpr::If {
                            condition: Box::new(condition.clone()),
                            yes: Box::new(
                                host::scalar_extent(&a)
                                    .ok_or_else(|| error("branch extent is not scalar"))?,
                            ),
                            no: Box::new(
                                host::scalar_extent(&b)
                                    .ok_or_else(|| error("branch extent is not scalar"))?,
                            ),
                        })),
                        _ => None,
                    };
                }
            }
            // Descriptors keep their offsets: copy their entire backing buffers,
            // including nested payloads, before interpreting the selected descriptor.
            let copy = |body: &mut Body<'_, '_, '_>, take_yes: bool| {
                for (a, b, destination, element) in &copies {
                    let source = if take_yes { *a } else { *b };
                    let count = body.op(
                        OpTag::Intrinsic {
                            id: catalog().known().storage_len,
                            overload_idx: 0,
                        },
                        vec![Body::number(source.set), Body::number(source.binding)],
                        Body::number(0).ty,
                    )?;
                    let source = storage_view(body, source, element.clone(), count.clone())?;
                    let destination = storage_view(body, *destination, element.clone(), count.clone())?;
                    body.copy_elements(source, destination, count)?;
                }
                body.op(OpTag::Unit, vec![], types::unit())
            };
            self.body.branch(test, |body| copy(body, true), |body| copy(body, false))?;
            return restore_value(&mut self.body, &result);
        }
        self.body.branch(
            test,
            |body| restore_value(body, yes),
            |body| restore_value(body, no),
        )
    }
}

fn scalar_source(binding: BindingRef) -> ScalarSource {
    ScalarSource::Binding {
        set: binding.set,
        binding: binding.binding,
    }
}

fn statement_names(query: Query<'_>, statements: &[Value]) -> Result<LookupSet<Value>, OptimizeError> {
    let mut names = LookupSet::default();
    for &statement in statements {
        names.extend(names_in(query, "OperationUse", statement)?);
    }
    Ok(names)
}

fn names_in(query: Query<'_>, table: &str, value: Value) -> Result<LookupSet<Value>, OptimizeError> {
    let mut names = LookupSet::default();
    query.for_each(table, |row| {
        if row[0] == value {
            if let Some(name) = query.value_name(row[1])? {
                names.insert(name);
            }
        }
        Ok(())
    })?;
    Ok(names)
}

fn domain_names(query: Query<'_>, domain: Value) -> Result<LookupSet<Value>, OptimizeError> {
    let mut names = LookupSet::default();
    for axis in super::domain_axes(query, domain)? {
        query.for_each("IndexUse", |row| {
            if row[0] == axis.extent {
                names.insert(row[1]);
            }
            Ok(())
        })?;
    }
    Ok(names)
}

pub(super) struct GeneratedBuffers {
    owner: String,
    next: u32,
    declarations: Vec<StorageBindingDecl>,
    scalar_tasks: Vec<ScalarTask>,
}

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn device_buffer(
        &mut self,
        element: &Type,
        bytes: Expr,
        count: Typed,
    ) -> Result<(BindingRef, Typed), OptimizeError> {
        let block = self.current()?;
        self.device_buffer_at(element, bytes, count, block)
    }

    pub(in crate::egglog::to_ssa) fn device_buffer_at(
        &mut self,
        element: &Type,
        bytes: Expr,
        count: Typed,
        block: crate::ssa::types::BlockId,
    ) -> Result<(BindingRef, Typed), OptimizeError> {
        let element = storage_value_type(element);
        let Some(stride) = storage_elem_stride(&element) else {
            return Err(error("device element has no byte stride"));
        };
        let Some(generated) = self.generated.as_mut() else {
            return Err(error("device allocation is outside an entry"));
        };
        let binding = BindingRef::new(0, generated.next);
        generated.next =
            generated.next.checked_add(1).ok_or_else(|| error("descriptor slots exhausted"))?;
        let name = format!("{}_memory_{}", generated.owner, binding.binding);
        let bytes = Expr::Max(Box::new(bytes), Box::new(Expr::Integer(stride.into())));
        let length = buffer_capacity(bytes)?;
        generated.declarations.push(StorageBindingDecl {
            binding,
            role: StorageRole::Intermediate,
            elem_ty: element.clone(),
            host: StorageHost {
                logical_resource: name,
                length,
            },
        });
        let ty = types::view_array_with_size(
            &element,
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            buffer_tag(binding),
        );
        let value = self.builder.func_mut().append_inst(
            block,
            InstKind::Op {
                tag: OpTag::StorageView(PureViewSource::Storage(binding)),
                operands: vec![Body::number(0).value, count.value],
            },
            ty.clone(),
        );
        let value = Typed {
            value: value.into(),
            ty,
        };
        self.register_array(&value, Body::number(0), vec![count])?;
        Ok((binding, value))
    }
}

// Locations survive dispatch boundaries; SSA operands are only cached evaluations.
#[derive(Clone)]
pub(super) enum Location {
    Product(Vec<Location>),
    Parameter(Value),
    Constant(i64, Type),
    Buffer(BindingRef, Type),
    Array(ArrayLocation),
}

impl Location {
    fn persistent(&self) -> bool {
        match self {
            Self::Product(fields) => fields.iter().all(Self::persistent),
            Self::Array(array) => {
                array.binding.is_some()
                    && std::iter::once(&array.offset)
                        .chain(array.dimensions.iter().map(|(_, extent)| extent))
                        .all(|extent| !matches!(extent.value, ExtentValue::Local(_)))
            }
            Self::Parameter(_) | Self::Constant(..) | Self::Buffer(..) => true,
        }
    }
}

#[derive(Clone)]
pub(super) struct ArrayLocation {
    binding: Option<BindingRef>,
    element: Type,
    offset: Extent,
    dimensions: Vec<(Type, Extent)>,
}

#[derive(Clone)]
struct Extent {
    value: ExtentValue,
    host: Option<Expr>,
}

#[derive(Clone)]
enum ExtentValue {
    Constant(i64),
    Local(Typed),
    Stored(BindingRef, u32),
}

impl Extent {
    fn local(body: &Body<'_, '_, '_>, value: Typed) -> Self {
        Self {
            value: body.integer_literal(&value).map_or(ExtentValue::Local(value), ExtentValue::Constant),
            host: None,
        }
    }

    fn host(&self) -> Option<Expr> {
        self.host.clone().or_else(|| match self.value {
            ExtentValue::Constant(n) => Some(Expr::Integer(n)),
            ExtentValue::Stored(binding, index) => Some(Expr::Scalar(ScalarExpr::Read {
                source: scalar_source(binding),
                offset: index * 4,
                ty: ScalarType::I32,
            })),
            ExtentValue::Local(_) => None,
        })
    }

    fn emit(&self, body: &mut Body<'_, '_, '_>) -> Result<Typed, OptimizeError> {
        match &self.value {
            ExtentValue::Constant(n) => body.literal(*n, &types::i32()),
            ExtentValue::Local(value) => Ok(value.clone()),
            ExtentValue::Stored(binding, index) => {
                let storage = storage_view(body, *binding, Body::number(0).ty, Body::number(index + 1))?;
                let value = body.index(storage, Body::number(*index))?;
                body.cast(value, &types::i32())
            }
        }
    }

    fn remap(&mut self, bindings: &LookupMap<BindingRef, BindingRef>) {
        if let ExtentValue::Stored(binding, _) = &mut self.value {
            if let Some(next) = bindings.get(binding) {
                *binding = *next;
                self.host = None;
            }
        }
    }
}

impl Body<'_, '_, '_> {
    fn array_location(&mut self, value: &Typed) -> Result<ArrayLocation, OptimizeError> {
        if let Some(Location::Array(location)) = self.locations.get(&value.value) {
            return Ok(location.clone());
        }
        let binding = types::array_view_buffer(&value.ty);
        let (offset, dimensions) = if binding.is_some() {
            let offset = if let Some(n) = array_offset(self, value) {
                self.literal(n, &types::i32())?
            } else {
                self.op(
                    OpTag::Project { index: 0 },
                    vec![value.clone()],
                    Body::number(0).ty,
                )?
            };
            (offset, self.view_dimensions(value.clone())?)
        } else {
            (Body::number(0), vec![self.array_length(value.clone())?])
        };
        self.describe_array(value, offset, dimensions)
    }

    fn describe_array(
        &self,
        value: &Typed,
        offset: Typed,
        dimensions: Vec<Typed>,
    ) -> Result<ArrayLocation, OptimizeError> {
        let Some(element) = value.ty.elem_type().cloned() else {
            return Err(error("array location has no element type"));
        };
        let binding = types::array_view_buffer(&value.ty);
        let Some(dimension_types) = value.ty.array_dims() else {
            return Err(error("array dimensions missing"));
        };
        if dimensions.len() != dimension_types.len() {
            return Err(error("array location rank does not match its dimensions"));
        }
        let dimensions = dimension_types
            .iter()
            .cloned()
            .zip(dimensions.into_iter().map(|dimension| Extent::local(self, dimension)))
            .collect();
        Ok(ArrayLocation {
            binding,
            element,
            offset: Extent::local(self, offset),
            dimensions,
        })
    }

    pub(super) fn register_array(
        &mut self,
        value: &Typed,
        offset: Typed,
        dimensions: Vec<Typed>,
    ) -> Result<(), OptimizeError> {
        let location = self.describe_array(value, offset, dimensions)?;
        self.locations.insert(value.value, Location::Array(location));
        Ok(())
    }

    pub(super) fn set_shape(&mut self, value: &Typed, shape: Vec<Expr>) -> Result<(), OptimizeError> {
        let mut location = self.array_location(value)?;
        if shape.len() > location.dimensions.len() {
            return Err(error("host shape exceeds the array location rank"));
        }
        for ((_, dimension), extent) in location.dimensions.iter_mut().zip(shape) {
            dimension.host = Some(extent);
        }
        self.locations.insert(value.value, Location::Array(location));
        Ok(())
    }

    pub(super) fn host_shape(&self, value: &Typed) -> Option<Vec<Expr>> {
        match self.locations.get(&value.value)? {
            Location::Array(location) => {
                location.dimensions.iter().map(|(_, extent)| extent.host()).collect()
            }
            _ => None,
        }
    }

    pub(super) fn located_dimensions(
        &mut self,
        value: &Typed,
    ) -> Result<Option<Vec<Typed>>, OptimizeError> {
        let Some(Location::Array(location)) = self.locations.get(&value.value).cloned() else {
            return Ok(None);
        };
        location
            .dimensions
            .iter()
            .map(|(_, extent)| extent.emit(self))
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }

    pub(super) fn located_length(&self, value: &Typed) -> Option<Expr> {
        match self.locations.get(&value.value)? {
            Location::Array(location) => location.dimensions.first()?.1.host(),
            Location::Buffer(binding, ty)
                if ty.array_variant().is_some_and(types::is_array_variant_virtual) =>
            {
                Some(Expr::Scalar(ScalarExpr::Read {
                    source: scalar_source(*binding),
                    offset: 8,
                    ty: ScalarType::I32,
                }))
            }
            _ => None,
        }
    }
}

fn remap_saved(saved: &Location, bindings: &LookupMap<BindingRef, BindingRef>) -> Location {
    let get = |binding: &BindingRef| bindings.get(binding).copied().unwrap_or(*binding);
    match saved {
        Location::Product(fields) => {
            Location::Product(fields.iter().map(|field| remap_saved(field, bindings)).collect())
        }
        Location::Parameter(name) => Location::Parameter(*name),
        Location::Constant(n, ty) => Location::Constant(*n, remap_type(ty, bindings)),
        Location::Buffer(binding, ty) => Location::Buffer(get(binding), remap_type(ty, bindings)),
        Location::Array(location) => {
            let mut location = location.clone();
            location.binding = location.binding.as_ref().map(get);
            location.element = remap_type(&location.element, bindings);
            location.offset.remap(bindings);
            for (_, extent) in &mut location.dimensions {
                extent.remap(bindings);
            }
            Location::Array(location)
        }
    }
}

fn retain_fixed_shape(saved: &mut Location, logical: &Type) -> Result<(), OptimizeError> {
    match (saved, strip_existentials(logical)) {
        (
            Location::Product(values),
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields),
        ) => {
            if values.len() != fields.len() {
                return Err(error("loop shape product arity mismatch"));
            }
            for (value, field) in values.iter_mut().zip(fields) {
                retain_fixed_shape(value, field)?;
            }
        }
        (Location::Array(location), ty) => {
            if let Some(Type::Constructed(TypeName::Size(n), _)) = ty.array_size() {
                let n = i64::try_from(*n).map_err(|_| error("loop extent exceeds i64"))?;
                if let Some((_, extent)) = location.dimensions.first_mut() {
                    *extent = Extent {
                        value: ExtentValue::Constant(n),
                        host: None,
                    };
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn saved_slots(saved: &Location) -> Result<Vec<(BindingRef, Type)>, OptimizeError> {
    fn add(slots: &mut Vec<(BindingRef, Type)>, binding: BindingRef, ty: Type) {
        if !slots.iter().any(|(existing, _)| *existing == binding) {
            slots.push((binding, ty));
        }
    }
    fn nested(slots: &mut Vec<(BindingRef, Type)>, ty: &Type) -> Result<(), OptimizeError> {
        if let Some(binding) = types::array_view_buffer(ty) {
            let Some(element) = ty.elem_type() else {
                return Err(error("carried view element missing"));
            };
            add(slots, binding, element.clone());
        }
        if let Type::Constructed(_, fields) = ty {
            for field in fields {
                nested(slots, field)?;
            }
        }
        Ok(())
    }
    fn visit(slots: &mut Vec<(BindingRef, Type)>, saved: &Location) -> Result<(), OptimizeError> {
        match saved {
            Location::Product(fields) => {
                for field in fields {
                    visit(slots, field)?;
                }
            }
            Location::Constant(..) | Location::Parameter(..) => {
                return Err(error("carried value has not been materialized"))
            }
            Location::Buffer(binding, ty) => {
                add(slots, *binding, storage_value_type(ty));
                nested(slots, ty)?;
            }
            Location::Array(location) => {
                let Some(binding) = location.binding else {
                    return Err(error("carried array has no storage"));
                };
                add(slots, binding, location.element.clone());
                for extent in std::iter::once(&location.offset)
                    .chain(location.dimensions.iter().map(|(_, extent)| extent))
                {
                    match extent.value {
                        ExtentValue::Stored(binding, _) => add(slots, binding, Body::number(0).ty),
                        ExtentValue::Local(_) => {
                            return Err(error("carried extent has not been materialized"))
                        }
                        ExtentValue::Constant(_) => {}
                    }
                }
                nested(slots, &location.element)?;
            }
        }
        Ok(())
    }
    let mut slots = Vec::new();
    visit(&mut slots, saved)?;
    Ok(slots)
}

fn restore_value(body: &mut Body<'_, '_, '_>, saved: &Location) -> Result<Typed, OptimizeError> {
    let value = match saved {
        Location::Parameter(name) => body.named(*name)?,
        Location::Product(fields) => {
            let values =
                fields.iter().map(|field| restore_value(body, field)).collect::<Result<Vec<_>, _>>()?;
            body.tuple(values)?
        }
        Location::Array(location) => {
            let Some(binding) = location.binding else {
                return Err(error("restored array has no storage"));
            };
            let mut fields = vec![location.element.clone(), types::array_variant_view()];
            fields.extend(location.dimensions.iter().enumerate().map(|(axis, (ty, _))| {
                if axis == 0 {
                    Type::Constructed(TypeName::SizePlaceholder, vec![])
                } else {
                    ty.clone()
                }
            }));
            fields.push(buffer_tag(binding));
            let mut operands = vec![location.offset.emit(body)?];
            for (_, extent) in &location.dimensions {
                operands.push(extent.emit(body)?);
            }
            body.op(
                OpTag::StorageView(PureViewSource::Storage(binding)),
                operands,
                Type::Constructed(TypeName::Array, fields),
            )?
        }
        Location::Constant(n, ty) => body.literal(*n, ty)?,
        Location::Buffer(binding, ty) => {
            let storage = storage_view(body, *binding, storage_value_type(ty), Body::number(1))?;
            let value = body.index(storage, Body::number(0))?;
            body.cast(value, ty)?
        }
    };
    body.locations.insert(value.value, saved.clone());
    Ok(value)
}

fn save_named(
    body: &mut Body<'_, '_, '_>,
    name: Value,
    value: Typed,
    materialize: bool,
) -> Result<Location, OptimizeError> {
    let Some(plan) = body.compiler.execution.storage.transfers.get(&name).cloned() else {
        return Err(error("cross-dispatch value has no transport plan"));
    };
    save_component(body, value, &plan, materialize)
}

fn save_component(
    body: &mut Body<'_, '_, '_>,
    mut value: Typed,
    plan: &Transfer,
    materialize: bool,
) -> Result<Location, OptimizeError> {
    if !materialize {
        if let Some(location) = body.locations.get(&value.value).filter(|location| location.persistent()) {
            return Ok(location.clone());
        }
    }
    let original = value.value;
    if let Transfer::Product(fields) = plan {
        let mut saved = Vec::new();
        for (i, field_plan) in fields.iter().enumerate() {
            let field = body.field(value.clone(), i)?;
            saved.push(save_component(body, field, field_plan, materialize)?);
        }
        let location = Location::Product(saved);
        body.locations.insert(original, location.clone());
        return Ok(location);
    }
    if !materialize {
        if let Some(n) = body.literal_bits(&value) {
            return Ok(Location::Constant(n, value.ty));
        }
    }
    // A previous boundary may already have realized this value's storage plan.
    if (matches!(plan, Transfer::Array { copy: true }) && types::array_view_buffer(&value.ty).is_none())
        || (materialize && matches!(plan, Transfer::Virtual))
    {
        let Some(element) = value.ty.elem_type().cloned() else {
            return Err(error("saved array element missing"));
        };
        let Some(stride) = storage_elem_stride(&storage_value_type(&element)) else {
            return Err(error("saved array element has no stride"));
        };
        let length = match body.host_array_length(&value) {
            Ok(length) => length,
            Err(failure) => {
                let Some(Type::Constructed(TypeName::Size(capacity), _)) = value.ty.array_size() else {
                    return Err(failure);
                };
                Expr::Integer(i64::try_from(*capacity).map_err(|_| error("array capacity exceeds i64"))?)
            }
        };
        let shape = body.host_shape(&value);
        let count = body.array_length(value.clone())?;
        let bytes = Expr::Multiply(Box::new(length.clone()), Box::new(Expr::Integer(stride.into())));
        let (_, destination) = body.device_buffer(&element, bytes, count.clone())?;
        body.copy_elements(value, destination.clone(), count)?;
        if let Some(shape) = shape {
            body.set_shape(&destination, shape)?;
        }
        value = destination;
    }
    let location = if matches!(plan, Transfer::Array { .. })
        || (materialize && matches!(plan, Transfer::Virtual))
    {
        let mut location = body.array_location(&value)?;
        let extents = std::iter::once(&mut location.offset)
            .chain(location.dimensions.iter_mut().map(|(_, extent)| extent))
            .filter(|extent| materialize || matches!(extent.value, ExtentValue::Local(_)))
            .collect::<Vec<_>>();
        if !extents.is_empty() {
            let count = u32::try_from(extents.len()).map_err(|_| error("descriptor rank exceeds u32"))?;
            let (binding, storage) = body.device_buffer(
                &Body::number(0).ty,
                Expr::Integer(i64::from(count) * 4),
                Body::number(count),
            )?;
            for (index, extent) in extents.into_iter().enumerate() {
                let value = extent.emit(body)?;
                let value = body.cast(value, &Body::number(0).ty)?;
                body.store(storage.clone(), Body::number(index as u32), value)?;
                extent.value = ExtentValue::Stored(binding, index as u32);
            }
        }
        Location::Array(location)
    } else {
        let Some(bytes) = storage_elem_stride(&storage_value_type(&value.ty)) else {
            return Err(error("cross-dispatch value has no storage size"));
        };
        let (binding, slot) =
            body.device_buffer(&value.ty, Expr::Integer(bytes.into()), Body::number(1))?;
        body.store(slot, Body::number(0), value.clone())?;
        Location::Buffer(binding, value.ty)
    };
    body.locations.insert(original, location.clone());
    Ok(location)
}

fn remap_type(ty: &Type, bindings: &LookupMap<BindingRef, BindingRef>) -> Type {
    match ty {
        Type::Constructed(TypeName::Buffer(binding), _) => {
            buffer_tag(bindings.get(binding).copied().unwrap_or(*binding))
        }
        Type::Constructed(name, fields) => Type::Constructed(
            name.clone(),
            fields.iter().map(|field| remap_type(field, bindings)).collect(),
        ),
        _ => ty.clone(),
    }
}

fn erase_buffers(ty: &Type) -> Type {
    match ty {
        Type::Constructed(TypeName::Buffer(_), _) => types::no_buffer(),
        Type::Constructed(name, fields) => {
            Type::Constructed(name.clone(), fields.iter().map(erase_buffers).collect())
        }
        _ => ty.clone(),
    }
}

fn live_names(
    query: Query<'_>,
    statements: &[Value],
    results: Value,
) -> Result<LookupSet<Value>, OptimizeError> {
    let mut names = statement_names(query, statements)?;
    for name in query.sequence("RegionYield", results)? {
        if let Some(name) = query.value_name(name)? {
            names.insert(name);
        }
    }
    Ok(names)
}

fn restore(
    body: &mut Body<'_, '_, '_>,
    saved: &StableMap<Value, Location>,
    names: Option<&LookupSet<Value>>,
) -> Result<(), OptimizeError> {
    for (&name, saved) in saved {
        if names.is_some_and(|names| !names.contains(&name)) {
            continue;
        }
        let value = restore_value(body, saved)?;
        body.values.insert(name, value);
    }
    Ok(())
}

fn publication_leader(body: &mut Body<'_, '_, '_>) -> Result<Option<Typed>, OptimizeError> {
    Ok(if let Some(statement) = body.active_dispatch {
        let dispatch = *body.compiler.execution.dispatch(statement)?;
        let (group, _) = body.workgroups(dispatch.grid)?;
        let first = body.binary(BinaryOperator::Equal, group, Body::number(0))?;
        let lane = body.local_id()?;
        let first_lane = body.binary(BinaryOperator::Equal, lane, Body::number(0))?;
        Some(body.binary(BinaryOperator::LogicalAnd, first, first_lane)?)
    } else {
        None
    })
}

fn save(
    body: &mut Body<'_, '_, '_>,
    live: &LookupSet<Value>,
    parameters: &LookupSet<Value>,
    saved: &mut StableMap<Value, Location>,
) -> Result<(), OptimizeError> {
    saved.retain(|name, _| live.contains(name));
    let values = body
        .values
        .iter()
        .filter(|(name, _)| live.contains(name) && !parameters.contains(name))
        .map(|(&name, value)| (name, value.clone()))
        .collect::<Vec<_>>();
    let leader = publication_leader(body)?;
    let publish = |body: &mut Body<'_, '_, '_>| {
        for (name, value) in values {
            if value.ty.is_array() && types::array_view_buffer(&value.ty).is_some() {
                if let Some(&parameter) = parameters.iter().find(|parameter| {
                    body.values
                        .get(*parameter)
                        .is_some_and(|input| input.value == value.value && input.ty == value.ty)
                }) {
                    saved.insert(name, Location::Parameter(parameter));
                    continue;
                }
            }
            if value.ty == types::unit() {
                continue;
            }
            let location = save_named(body, name, value, false)?;
            if !body.host_values.contains_key(&name) {
                if let Location::Buffer(binding, ty) = &location {
                    if let Some(ty) = host::scalar_type(ty) {
                        body.host_values.insert(
                            name,
                            ScalarExpr::Read {
                                source: scalar_source(*binding),
                                offset: 0,
                                ty,
                            },
                        );
                    }
                }
            }
            saved.insert(name, location);
        }
        Ok(())
    };
    if let Some(leader) = leader {
        body.when(leader, publish)
    } else {
        publish(body)
    }
}

fn output_shape(body: &Body<'_, '_, '_>, value: &Typed) -> Result<Vec<Expr>, OptimizeError> {
    let mut shape = if let Some(shape) = body.host_shape(value) {
        shape.clone()
    } else {
        vec![body.host_array_length(value)?]
    };
    let mut ty = value.ty.elem_type();
    while let Some(element) = ty.filter(|ty| ty.is_array()) {
        if element.array_variant().is_some_and(types::is_array_variant_bounded) {
            return Err(error("a row capacity does not establish a uniform row length"));
        }
        let Some(Type::Constructed(TypeName::Size(size), _)) = element.array_size() else {
            return Err(error("nested output requires explicit row shape witnesses"));
        };
        shape.push(Expr::Integer(
            i64::try_from(*size).map_err(|_| error("output extent exceeds i64"))?,
        ));
        ty = element.elem_type();
    }
    Ok(shape)
}

fn buffer_capacity(bytes: Expr) -> Result<BufferLen, OptimizeError> {
    match bytes.constant().map_err(|failure| error(failure.to_string()))? {
        Some(bytes) => Ok(BufferLen::Fixed {
            bytes: u64::try_from(bytes).map_err(|_| error("negative buffer capacity"))?,
        }),
        None => Ok(BufferLen::Computed { bytes }),
    }
}

fn output_capacities(
    body: &Body<'_, '_, '_>,
    values: &[Typed],
    destinations: &[Option<BindingRef>],
    plans: &[Output],
) -> Result<Vec<BufferLen>, OptimizeError> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if let Some(binding) = destinations.get(index).copied().flatten() {
                if let Some(declaration) = body.generated.as_ref().and_then(|generated| {
                    generated.declarations.iter().find(|declaration| declaration.binding == binding)
                }) {
                    return Ok(declaration.host.length.clone());
                }
            }
            if let Some(element) = value.ty.elem_type().filter(|_| value.ty.is_array()) {
                let Some(stride) = storage_elem_stride(&storage_value_type(element)) else {
                    return Err(error("output array element has no storage stride"));
                };
                let bytes = Expr::Multiply(
                    Box::new(match &plans[index] {
                        Output::Array {
                            capacity: Some(capacity),
                            ..
                        } => Expr::Integer(*capacity),
                        _ => body.host_array_length(value)?,
                    }),
                    Box::new(Expr::Integer(stride.into())),
                );
                buffer_capacity(bytes)
            } else {
                let Some(bytes) = type_byte_size(&storage_value_type(&value.ty)) else {
                    return Err(error("output value has no static size"));
                };
                Ok(BufferLen::Fixed { bytes: bytes.into() })
            }
        })
        .collect()
}

fn array_offset(body: &Body<'_, '_, '_>, value: &Typed) -> Option<i64> {
    if let Some(Location::Array(location)) = body.locations.get(&value.value) {
        return match location.offset.value {
            ExtentValue::Constant(n) => Some(n),
            _ => None,
        };
    }
    let ValueRef::Ssa(id) = value.value else {
        return None;
    };
    let function = body.builder.func();
    let inst = function.inst_of_value(id)?;
    let InstKind::Op {
        tag: OpTag::StorageView(source),
        operands,
    } = &function.insts[inst].data
    else {
        return None;
    };
    let offset = body.integer_literal(&Typed {
        value: *operands.first()?,
        ty: types::i32(),
    })?;
    match source {
        PureViewSource::Storage(_) => Some(offset),
        PureViewSource::Inherited => {
            let parent = *operands.last()?;
            let ValueRef::Ssa(parent_id) = parent else {
                return None;
            };
            let parent = Typed {
                value: parent,
                ty: function.value_type(parent_id).clone(),
            };
            array_offset(body, &parent)?.checked_add(offset)
        }
        _ => None,
    }
}

fn result_buffer(body: &Body<'_, '_, '_>, value: &Typed) -> Option<BindingRef> {
    if value.ty.is_array() {
        return (array_offset(body, value) == Some(0))
            .then(|| types::array_view_buffer(&value.ty))
            .flatten();
    }
    if body.literal_bits(value).is_some() {
        return None;
    }
    match body.locations.get(&value.value)? {
        Location::Buffer(binding, ty) if *ty == value.ty => Some(*binding),
        _ => None,
    }
}

fn write_results(
    body: &mut Body<'_, '_, '_>,
    values: &[Typed],
    outputs: &[EntryOutput],
) -> Result<bool, OptimizeError> {
    if outputs.is_empty() {
        return Ok(false);
    }
    if values.len() != outputs.len() {
        return Err(error("public output arity differs from SSA results"));
    }
    let partition = if let Some(statement) = body.active_dispatch {
        let dispatch = *body.compiler.execution.dispatch(statement)?;
        let (group, groups) = body.workgroups(dispatch.grid)?;
        let lane = body.local_id()?;
        let base = body.binary(
            BinaryOperator::Multiply,
            group.clone(),
            Body::number(dispatch.width.get()),
        )?;
        let start = body.binary(BinaryOperator::Add, base, lane.clone())?;
        let step = body.binary(
            BinaryOperator::Multiply,
            groups,
            Body::number(dispatch.width.get()),
        )?;
        let first_group = body.binary(BinaryOperator::Equal, group, Body::number(0))?;
        let first_lane = body.binary(BinaryOperator::Equal, lane, Body::number(0))?;
        let leader = body.binary(BinaryOperator::LogicalAnd, first_group, first_lane)?;
        Some((start, step, leader))
    } else {
        None
    };
    let mut writes = false;
    for (index, (value, output)) in values.iter().zip(outputs).enumerate() {
        if value.ty.is_array() {
            let Some(binding) = output.storage_binding() else {
                return Err(error("array output has no buffer"));
            };
            if types::array_view_buffer(&value.ty) == Some(binding) {
                continue;
            }
            writes = true;
            let count = body.array_length(value.clone())?;
            let mut destination = body.array_location(value)?;
            destination.binding = Some(binding);
            destination.element = storage_value_type(&destination.element);
            destination.offset = Extent::local(body, Body::number(0));
            let destination = restore_value(body, &Location::Array(destination))?;
            body.counted(
                partition.as_ref().map_or_else(|| Body::number(0), |(start, ..)| start.clone()),
                count,
                partition.as_ref().map_or_else(|| Body::number(1), |(_, step, _)| step.clone()),
                vec![],
                |body, index, _| {
                    let item = body.index(value.clone(), index.clone())?;
                    body.store(destination, index, item)?;
                    Ok(vec![])
                },
            )?;
        } else {
            if output.storage_binding().is_some_and(|binding| result_buffer(body, value) == Some(binding)) {
                continue;
            }
            writes = true;
            if let Some(binding) = output.storage_binding() {
                let destination = storage_view(body, binding, output.ty.clone(), Body::number(1))?;
                if let Some((_, _, leader)) = &partition {
                    body.when(leader.clone(), |body| {
                        body.store(destination, Body::number(0), value.clone())
                    })?;
                } else {
                    body.store(destination, Body::number(0), value.clone())?;
                }
            } else {
                let value = body.cast(value.clone(), &output.ty)?;
                let place = body.builder.new_place(value.ty.clone());
                body.builder
                    .push_void_inst(InstKind::OutputSlot { index, result: place })
                    .map_err(builder_error)?;
                body.builder
                    .push_void_inst(InstKind::Store {
                        place,
                        value: value.value,
                    })
                    .map_err(builder_error)?;
            }
        }
    }
    Ok(writes)
}

pub(super) fn storage_view(
    body: &mut Body<'_, '_, '_>,
    binding: BindingRef,
    element: Type,
    count: Typed,
) -> Result<Typed, OptimizeError> {
    let zero = Body::number(0);
    let size = body
        .integer_literal(&count)
        .and_then(|count| usize::try_from(count).ok())
        .map(|count| Type::Constructed(TypeName::Size(count), vec![]))
        .unwrap_or_else(|| Type::Constructed(TypeName::SizePlaceholder, vec![]));
    let ty = types::view_array_with_size(&element, size, buffer_tag(binding));
    let value = body.op(
        OpTag::StorageView(PureViewSource::Storage(binding)),
        vec![zero.clone(), count.clone()],
        ty,
    )?;
    body.register_array(&value, zero, vec![count])?;
    Ok(value)
}

fn bind_input(
    body: &mut Body<'_, '_, '_>,
    mut input: EntryInput,
    ty: &Type,
    inputs: &mut Vec<EntryInput>,
) -> Result<(Typed, Option<ScalarExpr>), OptimizeError> {
    let scalar_storage = input.storage_binding().is_some() && !ty.is_array();
    if scalar_storage {
        input.ty = types::sized_array(1, input.ty.clone());
    }
    input.ty = storage_value_type(&input.ty);
    if let (Some(binding), Type::Constructed(TypeName::Array, fields)) =
        (input.storage_binding(), &mut input.ty)
    {
        let Some(variant) = fields.get_mut(1) else {
            return Err(error("storage input has no array representation"));
        };
        *variant = types::array_variant_view();
        if let Some(region) = fields.last_mut() {
            *region = buffer_tag(binding);
        }
    }
    let param_ty = if let Some(binding) = input.storage_binding() {
        let Some(element) = input.ty.elem_type() else {
            return Err(error("storage input has no element"));
        };
        types::view_array_with_size(
            element,
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            buffer_tag(binding),
        )
    } else {
        input.ty.clone()
    };
    let parameter = body.builder.func_mut().add_function_param(param_ty.clone());
    let mut value = Typed {
        value: parameter.into(),
        ty: param_ty,
    };
    if let Some(binding) = input.storage_binding() {
        let length = if let Some(Type::Constructed(TypeName::Size(n), _)) = input.ty.array_size() {
            Body::number(u32::try_from(*n).map_err(|_| error("input size exceeds u32"))?)
        } else {
            body.op(
                OpTag::Intrinsic {
                    id: catalog().known().storage_len,
                    overload_idx: 0,
                },
                vec![Body::number(binding.set), Body::number(binding.binding)],
                Type::Constructed(TypeName::UInt(32), vec![]),
            )?
        };
        let Some(element) = value.ty.elem_type().cloned() else {
            return Err(error("input view element missing"));
        };
        value = storage_view(body, binding, element, length.clone())?;
        if scalar_storage {
            value = body.index(value, Body::number(0))?;
        }
        if ty.is_array() {
            let host_length = if let Some(Type::Constructed(TypeName::Size(n), _)) = input.ty.array_size() {
                Expr::Integer(i64::try_from(*n).map_err(|_| error("input extent exceeds i64"))?)
            } else {
                Expr::Scalar(ScalarExpr::BufferLength {
                    source: ScalarSource::Binding {
                        set: binding.set,
                        binding: binding.binding,
                    },
                    stride: storage_elem_stride(
                        input.ty.elem_type().ok_or_else(|| error("input element missing"))?,
                    )
                    .ok_or_else(|| error("input element stride missing"))?,
                })
            };
            body.set_shape(&value, vec![host_length])?;
            inputs.push(input);
            return Ok((value, None));
        }
    }
    let host = if ty.is_array() {
        let Some(Type::Constructed(TypeName::Size(_), _)) = ty.array_size() else {
            return Err(error("value array input has no static extent"));
        };
        None
    } else {
        input_host_value(&input, ty, 0)?
    };
    inputs.push(input);
    Ok((value, host))
}

fn input_host_value(
    input: &EntryInput,
    ty: &Type,
    offset: u32,
) -> Result<Option<ScalarExpr>, OptimizeError> {
    let physical = storage_value_type(ty);
    if let Some(lanes) = ty.vec_size() {
        let Some(tensor) = ty.as_tensor() else {
            return Err(error("vector parameter has no element type"));
        };
        let Some(bytes) = type_byte_size(tensor.elem) else {
            return Ok(None);
        };
        return (0..lanes)
            .map(|i| input_host_value(input, tensor.elem, offset + i as u32 * bytes))
            .collect::<Result<Option<Vec<_>>, _>>()
            .map(|fields| fields.map(ScalarExpr::Tuple));
    }
    if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = ty {
        let rules =
            if input.uniform_binding().is_some() { StorageLayout::Std140 } else { StorageLayout::Std430 };
        let Some(layout) = block_layout(&physical, rules) else {
            return Err(error("host parameter has no aggregate layout"));
        };
        return fields
            .iter()
            .enumerate()
            .map(|(i, ty)| input_host_value(input, ty, offset + layout.member_offsets[i]))
            .collect::<Result<Option<Vec<_>>, _>>()
            .map(|fields| fields.map(ScalarExpr::Tuple));
    }
    let ty = match ty {
        Type::Constructed(TypeName::Bool, _) => ScalarType::Bool,
        Type::Constructed(TypeName::Int(32), _) => ScalarType::I32,
        Type::Constructed(TypeName::UInt(32), _) => ScalarType::U32,
        Type::Constructed(TypeName::Float(32), _) => ScalarType::F32,
        _ => return Ok(None),
    };
    let source = if let Some(slot) = input.push_constant() {
        ScalarSource::PushConstant {
            name: input.name.clone(),
            offset: slot.offset,
        }
    } else if let Some(binding) = input.storage_binding().or_else(|| input.uniform_binding()) {
        ScalarSource::Binding {
            set: binding.set,
            binding: binding.binding,
        }
    } else {
        return Ok(None);
    };
    Ok(Some(ScalarExpr::Parameter { source, offset, ty }))
}

#[cfg(test)]
#[path = "entry_tests.rs"]
mod tests;
