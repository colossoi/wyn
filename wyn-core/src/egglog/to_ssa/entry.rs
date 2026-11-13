//! Lower public declarations and bind their values to the final memory signature.
use super::{builder_error, error, host, Body, Compiler, OptimizeError, Typed};
use crate::binding_layout::{
    extract_io_decoration, extract_sampler_binding, extract_storage_binding, extract_storage_image_binding,
    extract_storage_image_resource, extract_texture_backing, extract_texture_binding,
    extract_texture_resource, extract_uniform_binding,
};
use crate::builtins::catalog;
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
    reconcile_storage_binding_access, EntryInput, EntryOutput, EntryPoint, InputHost,
    ModuleInterfacePublish, StorageBindingDecl, StorageHost,
};
use crate::interface::results::result_layout_with_storage;
use crate::interface::DrawBufferOperand;
use crate::interface::{
    Attribute, BindingExposure, EntryInputKind, EntryKind, EntryParamBindingKind, PushConstantSlot,
    StorageAccess, StorageLayout, StorageRole,
};
use crate::op::{BinaryOperator, OpTag, PureViewSource};
use crate::ssa::context::BackendGlobal;
use crate::ssa::layout::{
    block_layout, std430_type_layout, storage_elem_stride, storage_value_type, type_byte_size,
};
use crate::ssa::stage::Elaborated;
use crate::ssa::types::{InstKind, ValueRef};
use crate::tlc::{extract_lambda_params_ref, DefMeta};
use crate::types::{
    self, buffer_tag, canonical_storage_buffer_ty, strip_existentials, Type, TypeExt, TypeName,
};
use crate::LookupSet;
use crate::{BindingRef, CodegenTarget, LookupMap, ResourceAccess, StableMap};
use egglog_engine::sort::SetContainer;
use egglog_engine::{RawValues, Value};
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
        let entry = query.fields("EntryPoint", root)?;
        let symbol = *compiler.program.symbols.resolve(query.0.value_to_base::<i64>(entry[1]));
        let Some(source) = compiler.program.source.defs.iter().find(|definition| definition.name == symbol)
        else {
            return Err(error("entry source metadata missing"));
        };
        let DefMeta::EntryPoint(metadata) = &source.meta else {
            return Err(error("entry declaration missing"));
        };
        let declaration = &metadata.declaration;
        let (source_body, parameters) = extract_lambda_params_ref(&source.body);
        let Some(&function) = compiler.definitions.get(&entry[0]) else {
            return Err(error("entry memory function missing"));
        };
        let fields = query.fields("DefineFunction", function)?;
        let block = query.fields("Block", fields[3])?;
        let params = query
            .vector(fields[1])?
            .into_iter()
            .map(|param| Ok(query.fields("Parameter", param)?[0]))
            .collect::<Result<Vec<_>, OptimizeError>>()?;
        let mut pc_offset = 0u32;
        let mut parameter_inputs = Vec::new();
        let mut templates = Vec::new();
        for (index, (_, ty)) in parameters.iter().enumerate() {
            let Some(declared) = declaration.params.get(index) else {
                return Err(error("entry parameter declaration missing"));
            };
            let binding = metadata.data.param_bindings.get(index).and_then(Option::as_ref);
            let first = templates.len();
            if let Some(EntryParamBindingKind::TupleOfViews(layout)) = binding.map(|binding| &binding.kind)
            {
                let Type::Constructed(TypeName::Tuple(_), fields) = strip_existentials(ty) else {
                    return Err(error("split parameter is not a tuple"));
                };
                if fields.len() != layout.len() {
                    return Err(error("split parameter arity mismatch"));
                }
                for (i, (ty, field)) in fields.iter().zip(layout).enumerate() {
                    let input = EntryInput {
                        name: format!("{}_{}", declared.name, i),
                        ty: canonical_storage_buffer_ty(ty),
                        host: InputHost::default(),
                        kind: EntryInputKind::Storage {
                            exposure: BindingExposure::Host(field.binding),
                            access: StorageAccess::ReadWrite,
                        },
                    };
                    templates.push((input, ty.clone()));
                }
            } else {
                let mut host = InputHost::default();
                let kind = if let Some(binding) = binding
                    .map(|binding| binding.first_buffer().0)
                    .or_else(|| extract_storage_binding(declared))
                {
                    host.length = type_byte_size(&storage_value_type(ty))
                        .map(|bytes| BufferLen::Fixed { bytes: bytes.into() });
                    EntryInputKind::Storage {
                        exposure: BindingExposure::Host(binding),
                        access: StorageAccess::ReadWrite,
                    }
                } else if let Some(binding) = extract_uniform_binding(declared) {
                    EntryInputKind::Uniform { binding }
                } else if let Some(binding) = extract_texture_binding(declared) {
                    host.backing = extract_texture_backing(declared);
                    host.resource = extract_texture_resource(declared);
                    EntryInputKind::Texture { binding }
                } else if let Some(binding) = extract_sampler_binding(declared) {
                    EntryInputKind::Sampler { binding }
                } else if let Some((binding, format, access, size)) =
                    extract_storage_image_binding(declared)
                {
                    host.image_size = size;
                    host.resource = extract_storage_image_resource(declared);
                    EntryInputKind::StorageImage {
                        binding,
                        format,
                        access,
                    }
                } else if declaration.entry_kind != EntryKind::Compute
                    || extract_io_decoration(declared).is_some()
                {
                    EntryInputKind::Value {
                        decoration: extract_io_decoration(declared),
                    }
                } else {
                    let Some((size, align)) = std430_type_layout(&storage_value_type(ty)) else {
                        return Err(error("push constant has no static size"));
                    };
                    pc_offset = pc_offset.div_ceil(align) * align;
                    let slot = PushConstantSlot {
                        offset: pc_offset,
                        size,
                    };
                    pc_offset += size;
                    EntryInputKind::PushConstant { slot }
                };
                templates.push((
                    EntryInput {
                        name: declared.name.clone(),
                        ty: canonical_storage_buffer_ty(ty),
                        kind,
                        host,
                    },
                    ty.clone(),
                ));
            }
            parameter_inputs.push((first..templates.len()).collect());
        }

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
            inputs: vec![],
            declarations: vec![],
            scalar_slots: LookupMap::default(),
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
            pending_ordered: false,
            pending_host_ordered: Vec::new(),
            pending_observable: false,
            restore_names: None,
            dispatch_override: None,
            stages: vec![],
            entries: vec![],
            loops: vec![],
        };
        emission.restart()?;
        let live = live_names(emission.body.query(), &[], block[1])?;
        emission
            .statements(block[0], &live, true)
            .map_err(|failure| error(format!("entry {}: {failure}", declaration.name)))?;
        let output_arity = crate::interface::lowering::entry_output_arity(declaration, &source_body.ty);
        let whole_result =
            declaration.buffer_demand || (declaration.source_entry.is_some() && output_arity == 1);
        let mut output_values = emission.body.results(block[1])?;
        if !whole_result
            && matches!(
                strip_existentials(&source_body.ty),
                Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), _)
            )
            && output_values.len() == 1
        {
            output_values = emission.body.unpack(output_values.remove(0))?;
        }
        let (output_values, output_shapes) = emission.prepare_array_outputs(block[1], output_values)?;
        let capacities =
            if output_arity == 0 { vec![] } else { output_capacities(&emission.body, &output_values)? };
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
                    (array_offset(&emission.body, value) == Some(0))
                        .then(|| types::array_view_buffer(&value.ty))
                        .flatten()
                        .filter(|binding| {
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
            let types = match strip_existentials(&source_body.ty) {
                ty if whole_result => std::slice::from_ref(ty),
                Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => fields.as_slice(),
                ty => std::slice::from_ref(ty),
            };
            for (index, ((ty, value), output)) in types.iter().zip(&output_values).zip(&outputs).enumerate()
            {
                let array_product = types::as_soa_tuple(ty)
                    .and_then(|_| crate::interface::results::array_element(ty))
                    .map(|(element, size)| {
                        types::make_array1(
                            element,
                            types::array_variant_composite(),
                            size,
                            types::no_buffer(),
                        )
                    });
                let ty = array_product.as_ref().unwrap_or(ty);
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
            host::cpu_body(emission.body.query(), function, &parameters, fields[3])?
        } else {
            None
        };
        if declaration.entry_kind == EntryKind::Compute {
            emission.pending = emission.pending_observable || emission.body.active_dispatch.is_some();
        }
        emission.pending |= write_results(&mut emission.body, &output_values, &outputs)?;
        if !emission.pending && !emission.entries.is_empty() {
            let Some(last) = emission.entries.last_mut() else {
                return Err(error("entry publication has no shader"));
            };
            last.outputs.extend(outputs);
        } else {
            emission.flush(&LookupSet::default(), outputs, true)?;
        }
        if let Some(result) = cpu_results {
            let Some(stage) = emission.entries.last() else {
                return Err(error("CPU scalar result has no interface stage"));
            };
            let logical_results = emission.body.query().vector(block[1])?.len();
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
    saved: StableMap<Value, SavedValue>,
    name: String,
    owner: String,
    kind: EntryKind,
    span: crate::ast::Span,
    pipeline: usize,
    invocation: ((u32, u32, u32), DispatchSize),
    pending: bool,
    pending_ordered: bool,
    pending_host_ordered: Vec<ScalarExpr>,
    pending_observable: bool,
    restore_names: Option<LookupSet<Value>>,
    dispatch_override: Option<((u32, u32, u32), DispatchSize)>,
    stages: Vec<ComputeStage>,
    entries: Vec<EntryPoint>,
    loops: Vec<DispatchLoop>,
}

impl EntryEmission<'_, '_, '_> {
    fn prepare_array_outputs(
        &mut self,
        results: Value,
        mut values: Vec<Typed>,
    ) -> Result<(Vec<Typed>, Vec<Option<ResultShape>>), OptimizeError> {
        let mut counts = Vec::new();
        for value in &values {
            let product = types::as_soa_tuple(&value.ty)
                .and_then(|_| crate::interface::results::array_element(&value.ty));
            if product.is_none() && (!value.ty.is_array() || output_shape(&self.body, value).is_ok()) {
                counts.push(None);
                continue;
            }
            let mut rank = usize::from(product.is_some());
            let mut leaf = product.as_ref().map_or(&value.ty, |(element, _)| element);
            while leaf.is_array() {
                rank += leaf.array_rank().ok_or_else(|| error("output array rank missing"))?;
                let Some(element) = leaf.elem_type() else {
                    return Err(error("output array element missing"));
                };
                leaf = element;
            }
            let rank_u32 = u32::try_from(rank).map_err(|_| error("output rank exceeds u32"))?;
            let (binding, storage) = self.body.device_buffer(
                &types::i32(),
                Expr::Integer(i64::from(rank_u32) * 4),
                Body::number(rank_u32),
            )?;
            let zero = self.body.literal(0, &types::i32())?;
            let totals = self.body.array_totals(value.clone(), 0, vec![zero; rank])?;
            for (i, total) in totals.into_iter().enumerate() {
                self.body.store(storage.clone(), Body::number(i as u32), total)?;
            }
            counts.push(Some((binding, rank, leaf.clone())));
        }
        let mut shapes = vec![None; values.len()];
        if counts.iter().all(Option::is_none) {
            return Ok((values, shapes));
        }
        self.pending = true;
        self.pending_observable = true;
        let live = live_names(self.body.query(), &[], results)?;
        self.flush(&live, vec![], true)?;
        let mut restored = self.body.results(results)?;
        if restored.len() == 1 && values.len() != 1 {
            restored = self.body.unpack(restored.remove(0))?;
        }
        if restored.len() != values.len() {
            return Err(error("output transport changed value arity"));
        }
        values = restored;
        let mut materialized = false;
        for (i, plan) in counts.into_iter().enumerate() {
            let Some((binding, rank, leaf)) = plan else {
                continue;
            };
            let totals = storage_view(&mut self.body, binding, types::i32(), Body::number(rank as u32))?;
            let lengths = (0..rank)
                .map(|depth| {
                    Expr::Scalar(ScalarExpr::Read {
                        source: scalar_source(binding),
                        offset: depth as u32 * 4,
                        ty: ScalarType::I32,
                    })
                })
                .collect::<Vec<_>>();
            if rank == 1 && array_offset(&self.body, &values[i]) == Some(0) {
                self.body.host_lengths.insert(values[i].value, lengths[0].clone());
                shapes[i] = Some(ResultShape::Rectangular(lengths));
                continue;
            }
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
            self.body.host_lengths.insert(payload.value, lengths[rank - 1].clone());
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
        let Some(generated) = self.body.generated.as_ref() else {
            return Err(error("entry buffers missing"));
        };
        for input in &generated.inputs {
            self.body.builder.func_mut().add_function_param(input.ty.clone());
        }
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
                let statement = query.fields("Let", statement)?;
                let parallel = query.fields("Parallel", statement[3])?;
                let domain = query.fields("IterationSpace", parallel[1])?;
                let mut axes = query.vector(domain[1])?;
                let block = query.fields("ParallelBlock", parallel[2])?;
                let collective = query.vector(block[1])?.into_iter().try_fold(false, |found, clause| {
                    Ok::<_, OptimizeError>(found || query.enode("Fold", clause)?.is_some())
                })?;
                let atomic_updates = self.body.atomic_contributions(block[1])?;
                let compact = query.vector(block[1])?.into_iter().try_fold(false, |found, clause| {
                    Ok::<_, OptimizeError>(
                        found
                            || query.enode("Compact", clause)?.is_some()
                            || (query.enode("Contribute", clause)?.is_some() && !atomic_updates),
                    )
                })?;
                if collective {
                    axes.pop();
                }
                let mut elements = Expr::Integer(1);
                for axis in axes {
                    let axis = query.fields("DenseAxis", axis)?;
                    let length = host::guarded_index(
                        query,
                        &self.body.host_values,
                        axis[1],
                        self.body.guard.as_ref(),
                    )?;
                    elements = Expr::Multiply(
                        Box::new(elements),
                        Box::new(Expr::Max(Box::new(length), Box::new(Expr::Integer(0)))),
                    );
                }
                let groups = if compact {
                    Expr::Integer(1)
                } else if collective {
                    elements.clone()
                } else {
                    Expr::Ceiling(
                        Box::new(elements.clone()),
                        Box::new(Expr::Integer(i64::from(dispatch.width.get()))),
                    )
                };
                // Intermediate stages still publish descriptor witnesses for an empty domain.
                let groups = Expr::Max(
                    Box::new(Expr::Integer(i64::from(outputs.is_empty() || compact))),
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
        let mut inputs = self.inputs.clone();
        inputs.extend(generated.inputs.iter().cloned());
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
        self.pending_ordered = false;
        self.pending_host_ordered.clear();
        self.pending_observable = false;
        self.restart()
    }

    fn statements(
        &mut self,
        statements: Value,
        after: &LookupSet<Value>,
        terminal: bool,
    ) -> Result<(), OptimizeError> {
        let statements = self.body.query().vector(statements)?;
        let generated = self.body.query().required("GeneratedDispatches", RawValues(Vec::new()))?;
        let generated =
            self.body.query().0.value_to_base::<bool>(generated) && self.kind == EntryKind::Compute;
        let mut grouped = None;
        for (i, &statement) in statements.iter().enumerate() {
            if grouped == Some(statement) {
                continue;
            }
            let mut live = statement_names(self.body.query(), &statements[i + 1..])?;
            live.extend(after.iter().copied());
            let fields = self.body.query().fields("Let", statement)?;
            let expression = fields[3];
            let dispatch = self.body.compiler.execution.dispatch(statement).is_ok();
            if self.body.query().contains("RequiresSnapshot", (self.body.function, statement))? {
                let mut before = statement_names(self.body.query(), &[statement])?;
                before.extend(live.iter().copied());
                self.flush(&before, vec![], false)?;
            }
            if self.body.query().enode("Copy", expression)?.is_some() {
                let pattern = self.body.query().vector(fields[1])?;
                if let [binding] = pattern.as_slice() {
                    if let Some((output, ty)) = super::runtime_binding(self.body.query(), *binding)? {
                        if let Some(storage) = self.body.reuse_storage(output, &ty)? {
                            self.body.bind(&pattern, vec![storage])?;
                            continue;
                        }
                    }
                }
            }
            if dispatch {
                let mut before = statement_names(self.body.query(), &[statement])?;
                before.extend(live.iter().copied());
                if self.pending_ordered || !self.launch_evaluates_prefix(statement)? {
                    self.flush(&before, vec![], false)?;
                }
                let parallel = self.body.query().fields("Parallel", expression)?;
                let mut captures = names_in(self.body.query(), "NamesParallelBody", parallel[2])?;
                let bound = names_in(self.body.query(), "BindersParallelBody", parallel[2])?;
                captures.retain(|name| !bound.contains(name));
                let epilogue =
                    if generated && self.body.compiler.execution.dispatch(statement)?.groups.is_none() {
                        if let Some(&next) = statements.get(i + 1) {
                            self.body
                                .query()
                                .contains("ScanEpilogue", (self.body.function, statement, next))?
                                .then_some(next)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                if let Some(next) = epilogue {
                    let next = self.body.query().fields("Let", next)?;
                    let next = self.body.query().fields("Parallel", next[3])?;
                    let used = names_in(self.body.query(), "NamesParallelBody", next[2])?;
                    let bound = names_in(self.body.query(), "BindersParallelBody", next[2])?;
                    captures.extend(used.difference(&bound).copied());
                }
                self.materialize_cpu_captures(&captures)?;
                self.body.active_dispatch = Some(statement);
                let last = terminal && i + 1 + usize::from(epilogue.is_some()) == statements.len();
                grouped = epilogue;
                if self.chunked_collective(statement, &live, last, epilogue)? {
                    continue;
                }
                self.body.statement(statement)?;
                if let Some(next) = epilogue {
                    let domain = self.body.query().fields("IterationSpace", parallel[1])?;
                    let axis = self.body.query().vector(domain[1])?[0];
                    let axis = self.body.query().fields("DenseAxis", axis)?;
                    let count = self.body.index_expression(axis[1])?;
                    self.scan_epilogue(next, (Body::number(0), count))?;
                }
                self.pending = true;
                if !last {
                    self.flush(&live, vec![], true)?;
                }
            } else if let Some(branch) = self.body.query().enode("If", expression)? {
                let mut buffer_result = false;
                if generated {
                    for binding in self.body.query().vector(fields[1])? {
                        if let Some((_, ty)) = super::runtime_binding(self.body.query(), binding)? {
                            buffer_result |= contains_array_descriptor(&ty);
                        }
                    }
                }
                if buffer_result || self.has_host_effect(branch[2])? || self.has_host_effect(branch[3])? {
                    self.branch(fields[1], &branch, &live)?;
                } else {
                    self.statement_effects(statement)?;
                    self.body.statement(statement)?;
                    self.pending = true;
                }
            } else if let Some(loop_) = self.body.query().enode("Loop", expression)? {
                if generated && self.cpu_loop(fields[1], &loop_)? {
                    continue;
                }
                let mut buffer_carry = false;
                if generated {
                    for merge in self.body.query().vector(loop_[0])? {
                        let merge = self.body.query().fields("Carry", merge)?;
                        let parameter = self.body.query().fields("Parameter", merge[0])?;
                        if let Some((_, ty)) = super::runtime_binding(self.body.query(), parameter[0])? {
                            buffer_carry |= contains_array_descriptor(&ty);
                        }
                    }
                }
                if buffer_carry || self.has_host_effect(loop_[3])? {
                    self.loop_body(expression, fields[1], &loop_, &live)?;
                } else {
                    self.statement_effects(statement)?;
                    self.body.statement(statement)?;
                    self.pending = true;
                }
            } else if let Some(accumulator) = self.body.query().enode("WithAcc", expression)? {
                let lambda = self.body.query().fields("FunctionBody", accumulator[1])?;
                let block = self.body.query().fields("Block", lambda[2])?;
                let mut needed = live_names(self.body.query(), &[], block[1])?;
                let captures = names_in(self.body.query(), "NamesAccInputs", accumulator[0])?;
                let binders = names_in(self.body.query(), "BindersAccInputs", accumulator[0])?;
                needed.extend(captures.difference(&binders).copied());
                needed.extend(live.iter().copied());
                self.statements(block[0], &needed, terminal && i + 1 == statements.len())?;
                let mut values = Vec::new();
                for input in self.body.query().vector(accumulator[0])? {
                    let input = self.body.query().fields("AccumulatorInput", input)?;
                    for name in self.body.query().vector(input[0])? {
                        let value = self.body.named(name)?;
                        values.push(value);
                    }
                }
                values.extend(self.body.results(block[1])?);
                let pattern = self.body.query().vector(fields[1])?;
                if pattern.len() == 1 && values.len() > 1 {
                    values = vec![self.body.tuple(values)?];
                }
                self.body.bind(&pattern, values)?;
                self.pending = true;
            } else {
                self.statement_effects(statement)?;
                self.body.statement(statement)?;
                self.pending = true;
            }
        }
        Ok(())
    }

    fn statement_effects(&mut self, statement: Value) -> Result<(), OptimizeError> {
        let fields = self.body.query().fields("Let", statement)?;
        let expression = fields[3];
        host::remember(
            Query(&self.body.compiler.program.graph),
            self.body.function,
            &mut self.body.host_values,
            statement,
        )?;
        let effects = self.body.query().required("EffectsStm", (self.body.function, statement))?;
        let Some(effects) = self.body.query().0.value_to_container::<SetContainer>(effects) else {
            return Err(error("invalid statement effect set"));
        };
        self.pending_observable |= !effects.data.is_empty();
        // A prefix read still executes before every lane's work, including
        // empty domains. Ownership analysis forbids reusing its source.
        if !effects.data.is_empty() && self.body.query().enode("Index", expression)?.is_none() {
            let ordered = self.body.query().constructor("OrderedEffect", RawValues(Vec::new()))?;
            let pattern = self.body.query().vector(fields[1])?;
            let value = if let [binding] = pattern.as_slice() {
                super::runtime_binding(self.body.query(), *binding)?
                    .and_then(|(name, _)| self.body.host_values.get(&name).cloned())
                    .filter(host::cpu_available)
            } else {
                None
            };
            if effects.data.len() == 1 && ordered.is_some_and(|effect| effects.data.contains(&effect)) {
                if let Some(value) = value {
                    self.pending_host_ordered.push(value);
                } else {
                    self.pending_ordered = true;
                }
            } else {
                self.pending_ordered = true;
            }
        }
        Ok(())
    }

    fn has_host_effect(&self, block: Value) -> Result<bool, OptimizeError> {
        let block = self.body.query().fields("Block", block)?;
        for statement in self.body.query().vector(block[0])? {
            if self.body.compiler.execution.dispatch(statement).is_ok() {
                return Ok(true);
            }
            let fields = self.body.query().fields("Let", statement)?;
            if let Some(branch) = self.body.query().enode("If", fields[3])? {
                if self.has_host_effect(branch[2])? || self.has_host_effect(branch[3])? {
                    return Ok(true);
                }
            } else if let Some(loop_) = self.body.query().enode("Loop", fields[3])? {
                if self.has_host_effect(loop_[3])? {
                    return Ok(true);
                }
            } else if let Some(accumulator) = self.body.query().enode("WithAcc", fields[3])? {
                let lambda = self.body.query().fields("FunctionBody", accumulator[1])?;
                if self.has_host_effect(lambda[2])? {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn launch_evaluates_prefix(&self, statement: Value) -> Result<bool, OptimizeError> {
        if self.pending_host_ordered.is_empty() {
            return Ok(true);
        }
        let dispatch = self.body.compiler.execution.dispatch(statement)?;
        if dispatch.grid.is_some() || dispatch.groups.is_some() {
            return Ok(false);
        }
        let query = self.body.query();
        let statement = query.fields("Let", statement)?;
        let parallel = query.fields("Parallel", statement[3])?;
        let block = query.fields("ParallelBlock", parallel[2])?;
        for clause in query.vector(block[1])? {
            if query.enode("Collect", clause)?.is_none() {
                return Ok(false);
            }
        }
        let domain = query.fields("IterationSpace", parallel[1])?;
        let mut sizes = Vec::new();
        for axis in query.vector(domain[1])? {
            let axis = query.fields("DenseAxis", axis)?;
            if let Some(size) = host::scalar_index(query, &self.body.host_values, axis[1])? {
                sizes.push(size);
            }
        }
        Ok(self
            .pending_host_ordered
            .iter()
            .all(|value| sizes.iter().any(|size| host::evaluates(size, value))))
    }

    fn cpu_loop(&mut self, pattern: Value, loop_: &[Value]) -> Result<bool, OptimizeError> {
        let pattern = self.body.query().vector(pattern)?;
        let mut outputs = Vec::new();
        for &binding in &pattern {
            let Some((name, ty)) = super::runtime_binding(self.body.query(), binding)? else {
                return Ok(false);
            };
            let Some(scalar) = host::scalar_type(&ty) else {
                return Ok(false);
            };
            outputs.push((name, ty, scalar));
        }
        if outputs.is_empty() {
            return Ok(false);
        }
        let Some(value) = host::cpu_loop(
            self.body.query(),
            self.body.function,
            &self.body.host_values,
            loop_,
        )?
        else {
            return Ok(false);
        };
        for (i, (name, ty, scalar)) in outputs.into_iter().enumerate() {
            let expression = ScalarExpr::Field {
                tuple: Box::new(value.clone()),
                index: i,
            };
            let slot = self.scalar_slot(name, &ty)?;
            let storage = storage_view(&mut self.body, slot, storage_value_type(&ty), Body::number(1))?;
            let loaded = self.body.index(storage, Body::number(0))?;
            let loaded = self.body.cast(loaded, &ty)?;
            self.body.values.insert(name, loaded);
            self.saved.insert(name, SavedValue::Buffer(slot, ty));
            self.body.host_values.insert(name, expression.clone());
            let Some(generated) = self.body.generated.as_mut() else {
                return Err(error("CPU loop has no entry buffers"));
            };
            generated.scalar_tasks.push(ScalarTask {
                stage: String::new(),
                destination: scalar_source(slot),
                offset: 0,
                ty: scalar,
                value: expression,
                replaces_dispatch: false,
            });
        }
        Ok(true)
    }

    fn scalar_slot(&mut self, name: Value, ty: &Type) -> Result<BindingRef, OptimizeError> {
        let Some(bytes) = storage_elem_stride(&storage_value_type(ty)) else {
            return Err(error("dispatch witness has no storage size"));
        };
        let (slot, _) = self.body.device_buffer(ty, Expr::Integer(bytes.into()), Body::number(1))?;
        let Some(generated) = self.body.generated.as_mut() else {
            return Err(error("entry buffers missing"));
        };
        generated.scalar_slots.insert(name, (slot, ty.clone()));
        Ok(slot)
    }

    fn materialize_cpu_captures(&mut self, names: &LookupSet<Value>) -> Result<(), OptimizeError> {
        let captures = self
            .body
            .values
            .iter()
            .filter(|(name, value)| {
                names.contains(name)
                    && !self.parameters.contains(name)
                    && self.body.literal_bits(value).is_none()
            })
            .filter_map(|(&name, value)| {
                host::scalar_type(&value.ty).map(|ty| (name, value.ty.clone(), ty))
            })
            .collect::<Vec<_>>();
        for (name, ty, scalar) in captures {
            let Some(value) =
                self.body.host_values.get(&name).filter(|value| host::cpu_available(value)).cloned()
            else {
                continue;
            };
            if !matches!(&value, ScalarExpr::Apply { op, .. } if !op.starts_with("to-")) {
                continue;
            }
            let Some(generated) = self.body.generated.as_ref() else {
                return Err(error("CPU capture has no entry buffers"));
            };
            if generated.scalar_slots.contains_key(&name) {
                continue;
            }
            let slot = self.scalar_slot(name, &ty)?;
            let storage = storage_view(&mut self.body, slot, storage_value_type(&ty), Body::number(1))?;
            let loaded = self.body.index(storage, Body::number(0))?;
            let loaded = self.body.cast(loaded, &ty)?;
            self.body.values.insert(name, loaded);
            self.saved.insert(name, SavedValue::Buffer(slot, ty));
            let Some(generated) = self.body.generated.as_mut() else {
                return Err(error("CPU capture has no entry buffers"));
            };
            generated.scalar_tasks.push(ScalarTask {
                stage: String::new(),
                destination: scalar_source(slot),
                offset: 0,
                ty: scalar,
                value,
                replaces_dispatch: false,
            });
        }
        Ok(())
    }

    fn read_slot(&mut self, name: Value, slot: BindingRef, ty: &Type) {
        self.saved.insert(name, SavedValue::Buffer(slot, ty.clone()));
        if let Some(ty) = host::scalar_type(ty) {
            self.body.host_values.insert(
                name,
                ScalarExpr::Read {
                    source: scalar_source(slot),
                    offset: 0,
                    ty,
                },
            );
        }
    }

    fn capture(&mut self, value: Typed, materialize_arrays: bool) -> Result<SavedValue, OptimizeError> {
        let saved = save_component(&mut self.body, value, None, materialize_arrays)?;
        self.pending = true;
        self.pending_observable = true;
        self.materialize_constants(saved)
    }

    fn materialize_constants(&mut self, saved: SavedValue) -> Result<SavedValue, OptimizeError> {
        match saved {
            SavedValue::Constant(n, ty) => {
                let value = self.body.literal(n, &ty)?;
                let Some(bytes) = storage_elem_stride(&storage_value_type(&ty)) else {
                    return Err(error("carried scalar has no storage size"));
                };
                let (slot, storage) =
                    self.body.device_buffer(&ty, Expr::Integer(bytes.into()), Body::number(1))?;
                self.body.store(storage, Body::number(0), value)?;
                Ok(SavedValue::Buffer(slot, ty))
            }
            SavedValue::Product(fields) => Ok(SavedValue::Product(
                fields
                    .into_iter()
                    .map(|field| self.materialize_constants(field))
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            saved => Ok(saved),
        }
    }

    fn loop_body(
        &mut self,
        expression: Value,
        pattern: Value,
        loop_: &[Value],
        after: &LookupSet<Value>,
    ) -> Result<(), OptimizeError> {
        let pattern = self.body.query().vector(pattern)?;
        let merges = self.body.query().vector(loop_[0])?;
        let block = self.body.query().fields("Block", loop_[3])?;
        let mut before = names_in(self.body.query(), "NamesMerges", loop_[0])?;
        before.extend(after.iter().copied());
        before.extend(names_in(self.body.query(), "NamesLoopForm", loop_[1])?);
        before.extend(live_names(
            self.body.query(),
            &self.body.query().vector(block[0])?,
            block[1],
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
            let merge = self.body.query().fields("Carry", merge)?;
            let parameter = self.body.query().fields("Parameter", merge[0])?[0];
            if let Some((name, ty)) = super::runtime_binding(self.body.query(), parameter)? {
                let Some(value) = self.body.operand(merge[1])? else {
                    return Err(error("loop initial value missing"));
                };
                let Some((output, _)) = super::runtime_binding(self.body.query(), *output)? else {
                    return Err(error("loop output missing"));
                };
                let host_length = if self.body.query().contains(
                    "LoopPreservesOuterLength",
                    (self.body.function, expression, index as i64),
                )? {
                    self.body.host_array_length(&value).ok()
                } else {
                    None
                };
                let initial = self.capture(value, true)?;
                let slots = saved_slots(&self.body, &initial)?;
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
                if let (Some(length), SavedValue::Array { host_length, .. }) = (host_length, &mut current) {
                    *host_length = Some(length);
                }
                carried_values.push((name, output, initial, current, slots));
            }
        }
        let control = if let Some(form) = self.body.query().enode("Counted", loop_[1])? {
            let Some(count) = host::scalar_index(self.body.query(), &self.body.host_values, form[2])?
            else {
                return Err(error("loop count has no host value"));
            };
            let slot = self.scalar_slot(form[0], &types::i32())?;
            let zero = self.body.literal(0, &types::i32())?;
            self.body.values.insert(form[0], zero);
            before.insert(form[0]);
            DispatchLoopControl::Counted {
                count,
                index: scalar_source(slot),
            }
        } else {
            let condition = self.body.query().fields("While", loop_[1])?[0];
            let Some((_, _, _, SavedValue::Buffer(slot, _), _)) =
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
        if let Some(form) = self.body.query().enode("Counted", loop_[1])? {
            let DispatchLoopControl::Counted {
                index: ScalarSource::Binding { set, binding },
                ..
            } = &control
            else {
                return Err(error("loop ordinal binding missing"));
            };
            self.read_slot(form[0], BindingRef::new(*set, *binding), &types::i32());
        }
        self.restart()?;
        if let Some(form) = self.body.query().enode("Counted", loop_[1])? {
            let ordinal = self.body.named(form[0])?;
            let start = self.body.index_expression(form[1])?;
            let step = self.body.index_expression(form[3])?;
            let delta = self.body.binary(BinaryOperator::Multiply, ordinal, step)?;
            let value = self.body.binary(BinaryOperator::Add, start, delta)?;
            self.body.values.insert(form[0], value);
            let Some(ordinal) = self.body.host_values.get(&form[0]).cloned() else {
                return Err(error("loop ordinal missing"));
            };
            let Some(start) = host::scalar_index(self.body.query(), &self.body.host_values, form[1])?
            else {
                return Err(error("loop start missing"));
            };
            let Some(step) = host::scalar_index(self.body.query(), &self.body.host_values, form[3])? else {
                return Err(error("loop step missing"));
            };
            self.body.host_values.insert(
                form[0],
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
        let mut live = live_names(self.body.query(), &[], block[1])?;
        live.extend(after.iter().copied());
        self.statements(block[0], &live, false)?;
        let values = self.body.results(block[1])?;
        if values.len() != carried_values.len() {
            return Err(error("loop carry arity mismatch"));
        }
        let mut carried = Vec::new();
        for ((_, _, _, current, initial_slots), value) in carried_values.iter().zip(values) {
            let next = self.capture(value, true)?;
            let current_slots = saved_slots(&self.body, current)?;
            let next_slots = saved_slots(&self.body, &next)?;
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

    fn install_saved(&mut self, name: Value, saved: SavedValue) {
        if let SavedValue::Buffer(binding, ty) = &saved {
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

    fn branch(
        &mut self,
        pattern: Value,
        branch: &[Value],
        after: &LookupSet<Value>,
    ) -> Result<(), OptimizeError> {
        let yes = self.body.query().fields("Block", branch[2])?;
        let no = self.body.query().fields("Block", branch[3])?;
        let mut before = live_names(self.body.query(), &self.body.query().vector(yes[0])?, yes[1])?;
        before.extend(live_names(
            self.body.query(),
            &self.body.query().vector(no[0])?,
            no[1],
        )?);
        before.extend(names_in(self.body.query(), "NamesAtom", branch[0])?);
        before.extend(after.iter().copied());
        self.flush(&before, vec![], false)?;
        let Some(condition) = host::atom(self.body.query(), &self.body.host_values, branch[0])? else {
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
            let mut live = live_names(self.body.query(), &[], block[1])?;
            live.extend(after.iter().copied());
            self.statements(block[0], &live, false)?;
            let values = self.body.results(block[1])?;
            let saved =
                values.into_iter().map(|value| self.capture(value, true)).collect::<Result<Vec<_>, _>>()?;
            self.flush(after, vec![], true)?;
            arms.push(saved);
        }
        self.body.guard = enclosing;
        self.saved = incoming;
        self.body.host_values = incoming_host.into();
        self.restart()?;
        let pattern = self.body.query().vector(pattern)?;
        let [yes, no]: [Vec<SavedValue>; 2] = arms.try_into().map_err(|_| error("branch arms missing"))?;
        if yes.len() != no.len() {
            return Err(error("branch arity mismatch"));
        }
        let mut selected = Vec::new();
        let test = self.body.atom(branch[0])?;
        for (yes, no) in yes.iter().zip(&no) {
            selected.push(self.merge_saved(&condition, test.clone(), yes, no)?);
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
        yes: &SavedValue,
        no: &SavedValue,
    ) -> Result<Typed, OptimizeError> {
        if let (SavedValue::Product(yes), SavedValue::Product(no)) = (yes, no) {
            if yes.len() != no.len() {
                return Err(error("branch product arity mismatch"));
            }
            let mut fields = Vec::new();
            for (yes, no) in yes.iter().zip(no) {
                fields.push(self.merge_saved(condition, test.clone(), yes, no)?);
            }
            return self.body.tuple(fields);
        }
        if let (
            SavedValue::Array {
                element: yes_ty,
                descriptor: Some(yes_descriptor),
                tail: yes_tail,
                host_length: yes_length,
                ..
            },
            SavedValue::Array {
                element: no_ty,
                descriptor: Some(no_descriptor),
                tail: no_tail,
                host_length: no_length,
                ..
            },
        ) = (yes, no)
        {
            if yes_ty != no_ty || yes_tail.len() != no_tail.len() {
                return Err(error(
                    "branch arrays require a shared nested payload representation",
                ));
            }
            let rank = yes_tail.len() + 1;
            let dimensions = self.body.branch(
                test.clone(),
                |body| {
                    let array = restore_value(body, yes)?;
                    let dimensions = body.view_dimensions(array)?;
                    body.pack_results(dimensions)
                },
                |body| {
                    let array = restore_value(body, no)?;
                    let dimensions = body.view_dimensions(array)?;
                    body.pack_results(dimensions)
                },
            )?;
            let dimensions = self.body.unpack_results(dimensions, rank)?;
            let mut shape = Vec::new();
            for (axis, dimension) in dimensions.into_iter().enumerate() {
                let extent = |length: &Option<Expr>, tail: &[(Type, Option<Expr>)], descriptor| {
                    let known = if axis == 0 { length.clone() } else { tail[axis - 1].1.clone() };
                    known.unwrap_or_else(|| {
                        Expr::Scalar(ScalarExpr::Read {
                            source: scalar_source(descriptor),
                            offset: (axis as u32 + 1) * 4,
                            ty: ScalarType::I32,
                        })
                    })
                };
                let yes = extent(yes_length, yes_tail, *yes_descriptor);
                let no = extent(no_length, no_tail, *no_descriptor);
                let yes = host::scalar_extent(&yes)
                    .ok_or_else(|| error("branch extent has no scalar witness"))?;
                let no =
                    host::scalar_extent(&no).ok_or_else(|| error("branch extent has no scalar witness"))?;
                shape.push((
                    dimension,
                    Expr::Scalar(ScalarExpr::If {
                        condition: Box::new(condition.clone()),
                        yes: Box::new(yes),
                        no: Box::new(no),
                    }),
                ));
            }
            let block = self.body.current()?;
            let destination = self.body.ranked_buffer(yes_ty.clone(), &shape, block)?;
            self.body.branch(
                test,
                |body| {
                    let source = restore_value(body, yes)?;
                    body.copy_elements(source, destination.clone(), shape[0].0.clone())
                },
                |body| {
                    let source = restore_value(body, no)?;
                    body.copy_elements(source, destination.clone(), shape[0].0.clone())
                },
            )?;
            return Ok(destination);
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
        names.extend(names_in(query, "NamesStm", statement)?);
    }
    Ok(names)
}

fn contains_array_descriptor(ty: &Type) -> bool {
    (ty.is_array()
        && !(ty.array_variant().is_some_and(types::is_array_variant_composite)
            && matches!(ty.array_size(), Some(Type::Constructed(TypeName::Size(_), _)))))
        || matches!(ty, Type::Constructed(_, fields) if fields.iter().any(contains_array_descriptor))
}

fn names_in(query: Query<'_>, table: &str, value: Value) -> Result<LookupSet<Value>, OptimizeError> {
    let set = query.required(table, (value,))?;
    let Some(set) = query.0.value_to_container::<SetContainer>(set) else {
        return Err(error("invalid free-name set"));
    };
    set.data
        .iter()
        .map(|&name| Ok(query.application(name, &["ValueNameRef", "BlockNameRef", "AccNameRef"])?.1[0]))
        .collect()
}

pub(super) struct GeneratedBuffers {
    owner: String,
    next: u32,
    inputs: Vec<EntryInput>,
    declarations: Vec<StorageBindingDecl>,
    scalar_slots: LookupMap<Value, (BindingRef, Type)>,
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
        let length = BufferLen::Computed {
            bytes: Expr::Max(Box::new(bytes), Box::new(Expr::Integer(stride.into()))),
        };
        generated.declarations.push(StorageBindingDecl {
            binding,
            role: StorageRole::Intermediate,
            elem_ty: element.clone(),
            host: StorageHost {
                logical_resource: name.clone(),
                length: length.clone(),
            },
        });
        let input = EntryInput {
            name,
            ty: types::view_array_with_size(
                &element,
                Type::Constructed(TypeName::SizePlaceholder, vec![]),
                buffer_tag(binding),
            ),
            kind: EntryInputKind::Storage {
                exposure: BindingExposure::Host(binding),
                access: StorageAccess::ReadWrite,
            },
            host: InputHost {
                length: Some(length),
                ..InputHost::default()
            },
        };
        self.builder.func_mut().add_function_param(input.ty.clone());
        generated.inputs.push(input);
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
        Ok((
            binding,
            Typed {
                value: value.into(),
                ty,
            },
        ))
    }
}

// These slots are final cross-dispatch ABI transport, not a second memory layout.
#[derive(Clone)]
enum SavedValue {
    Product(Vec<SavedValue>),
    Parameter(Value),
    Constant(i64, Type),
    Buffer(BindingRef, Type),
    Array {
        binding: BindingRef,
        element: Type,
        descriptor: Option<BindingRef>,
        length: Option<i64>,
        offset: Option<i64>,
        host_length: Option<Expr>,
        tail: Vec<(Type, Option<Expr>)>,
    },
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

fn remap_saved(saved: &SavedValue, bindings: &LookupMap<BindingRef, BindingRef>) -> SavedValue {
    let get = |binding: &BindingRef| bindings.get(binding).copied().unwrap_or(*binding);
    match saved {
        SavedValue::Product(fields) => {
            SavedValue::Product(fields.iter().map(|field| remap_saved(field, bindings)).collect())
        }
        SavedValue::Parameter(name) => SavedValue::Parameter(*name),
        SavedValue::Constant(n, ty) => SavedValue::Constant(*n, remap_type(ty, bindings)),
        SavedValue::Buffer(binding, ty) => SavedValue::Buffer(get(binding), remap_type(ty, bindings)),
        SavedValue::Array {
            binding,
            element,
            descriptor,
            length,
            offset,
            host_length,
            tail,
        } => SavedValue::Array {
            binding: get(binding),
            element: remap_type(element, bindings),
            descriptor: descriptor.as_ref().map(get),
            // A source loop may change the extent on its next iteration.
            length: descriptor.is_none().then_some(*length).flatten(),
            offset: descriptor.is_none().then_some(*offset).flatten(),
            host_length: if descriptor.is_none() { host_length.clone() } else { None },
            tail: tail
                .iter()
                .map(|(ty, extent)| {
                    (
                        remap_type(ty, bindings),
                        if descriptor.is_none() { extent.clone() } else { None },
                    )
                })
                .collect(),
        },
    }
}

fn retain_fixed_shape(saved: &mut SavedValue, logical: &Type) -> Result<(), OptimizeError> {
    match (saved, strip_existentials(logical)) {
        (
            SavedValue::Product(values),
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields),
        ) => {
            if values.len() != fields.len() {
                return Err(error("loop shape product arity mismatch"));
            }
            for (value, field) in values.iter_mut().zip(fields) {
                retain_fixed_shape(value, field)?;
            }
        }
        (
            SavedValue::Array {
                length, host_length, ..
            },
            ty,
        ) => {
            if let Some(Type::Constructed(TypeName::Size(n), _)) = ty.array_size() {
                let n = i64::try_from(*n).map_err(|_| error("loop extent exceeds i64"))?;
                *length = Some(n);
                *host_length = Some(Expr::Integer(n));
            }
        }
        _ => {}
    }
    Ok(())
}

fn saved_slots(
    body: &Body<'_, '_, '_>,
    saved: &SavedValue,
) -> Result<Vec<(BindingRef, Type)>, OptimizeError> {
    fn add(slots: &mut Vec<(BindingRef, Type)>, binding: BindingRef, ty: Type) {
        if !slots.iter().any(|(existing, _)| *existing == binding) {
            slots.push((binding, ty));
        }
    }
    fn nested(
        body: &Body<'_, '_, '_>,
        slots: &mut Vec<(BindingRef, Type)>,
        ty: &Type,
    ) -> Result<(), OptimizeError> {
        if let Some(binding) = types::array_view_buffer(ty) {
            let Some(element) = ty.elem_type() else {
                return Err(error("carried view element missing"));
            };
            add(slots, binding, element.clone());
        }
        if let Type::Constructed(_, fields) = ty {
            for field in fields {
                nested(body, slots, field)?;
            }
        }
        Ok(())
    }
    fn visit(
        body: &Body<'_, '_, '_>,
        slots: &mut Vec<(BindingRef, Type)>,
        saved: &SavedValue,
    ) -> Result<(), OptimizeError> {
        match saved {
            SavedValue::Product(fields) => {
                for field in fields {
                    visit(body, slots, field)?;
                }
            }
            SavedValue::Constant(..) | SavedValue::Parameter(..) => {
                return Err(error("carried value has not been materialized"));
            }
            SavedValue::Buffer(binding, ty) => {
                add(slots, *binding, storage_value_type(ty));
                nested(body, slots, ty)?;
            }
            SavedValue::Array {
                binding,
                element,
                descriptor,
                tail,
                ..
            } => {
                let Some(descriptor) = descriptor else {
                    return Err(error("carried array descriptor has not been materialized"));
                };
                add(slots, *binding, element.clone());
                add(
                    slots,
                    *descriptor,
                    types::tuple(vec![Body::number(0).ty; 2 + tail.len()]),
                );
                nested(body, slots, element)?;
            }
        }
        Ok(())
    }
    let mut slots = Vec::new();
    visit(body, &mut slots, saved)?;
    Ok(slots)
}

fn live_names(
    query: Query<'_>,
    statements: &[Value],
    results: Value,
) -> Result<LookupSet<Value>, OptimizeError> {
    let mut names = statement_names(query, statements)?;
    names.extend(names_in(query, "NamesResults", results)?);
    Ok(names)
}

fn restore(
    body: &mut Body<'_, '_, '_>,
    saved: &StableMap<Value, SavedValue>,
    names: Option<&LookupSet<Value>>,
) -> Result<(), OptimizeError> {
    for (&name, saved) in saved {
        if names.is_some_and(|names| !names.contains(&name)) {
            continue;
        }
        let value = restore_value(body, saved)?;
        body.transported_values.insert(name, value.value);
        body.values.insert(name, value);
    }
    Ok(())
}

fn restore_value(body: &mut Body<'_, '_, '_>, saved: &SavedValue) -> Result<Typed, OptimizeError> {
    match saved {
        SavedValue::Parameter(name) => body.named(*name),
        SavedValue::Product(fields) => {
            let values =
                fields.iter().map(|field| restore_value(body, field)).collect::<Result<Vec<_>, _>>()?;
            body.tuple(values)
        }
        SavedValue::Array {
            binding,
            element,
            descriptor,
            length,
            offset: known_offset,
            host_length,
            tail,
        } => {
            let uint = Body::number(0).ty;
            let metadata = if let Some(descriptor) = descriptor {
                let slot = storage_view(
                    body,
                    *descriptor,
                    types::tuple(vec![uint; 2 + tail.len()]),
                    Body::number(1),
                )?;
                Some(body.index(slot, Body::number(0))?)
            } else {
                None
            };
            let offset = if let Some(offset) = known_offset {
                body.literal(*offset, &Body::number(0).ty)?
            } else {
                body.field(
                    metadata.clone().ok_or_else(|| error("saved array offset missing"))?,
                    0,
                )?
            };
            let count = if let Some(length) = length {
                body.literal(*length, &types::i32())?
            } else {
                body.field(
                    metadata.clone().ok_or_else(|| error("saved array length missing"))?,
                    1,
                )?
            };
            let mut fields = vec![
                element.clone(),
                types::array_variant_view(),
                Type::Constructed(TypeName::SizePlaceholder, vec![]),
            ];
            fields.extend(tail.iter().map(|(ty, _)| ty.clone()));
            fields.push(buffer_tag(*binding));
            let ty = Type::Constructed(TypeName::Array, fields);
            let mut operands = vec![offset, count];
            for axis in 0..tail.len() {
                operands.push(body.field(
                    metadata.clone().ok_or_else(|| error("saved array shape missing"))?,
                    axis + 2,
                )?);
            }
            let array = body.op(
                OpTag::StorageView(PureViewSource::Storage(*binding)),
                operands,
                ty,
            )?;
            let host_length = if let Some(length) = host_length {
                length.clone()
            } else if let Some(length) = length {
                Expr::Integer(*length)
            } else {
                let descriptor = descriptor.ok_or_else(|| error("saved host length missing"))?;
                Expr::Scalar(ScalarExpr::Read {
                    source: scalar_source(descriptor),
                    offset: 4,
                    ty: ScalarType::I32,
                })
            };
            body.host_lengths.insert(array.value, host_length.clone());
            if !tail.is_empty() {
                let descriptor = descriptor.ok_or_else(|| error("saved shape descriptor missing"))?;
                let mut shape = vec![host_length];
                shape.extend(tail.iter().enumerate().map(|(axis, (_, extent))| {
                    extent.clone().unwrap_or_else(|| {
                        Expr::Scalar(ScalarExpr::Read {
                            source: scalar_source(descriptor),
                            offset: (axis as u32 + 2) * 4,
                            ty: ScalarType::I32,
                        })
                    })
                }));
                body.host_shapes.insert(array.value, shape);
            }
            Ok(array)
        }
        SavedValue::Constant(n, ty) => body.literal(*n, ty),
        SavedValue::Buffer(binding, ty) => {
            let storage = storage_view(body, *binding, storage_value_type(ty), Body::number(1))?;
            let value = body.index(storage, Body::number(0))?;
            let value = body.cast(value, ty)?;
            if ty.array_variant().is_some_and(types::is_array_variant_virtual) {
                let scalar = match ty.elem_type() {
                    Some(Type::Constructed(TypeName::Int(32) | TypeName::UInt(32), _)) => {
                        Some(ScalarType::I32)
                    }
                    _ => None,
                };
                if let Some(ty) = scalar {
                    body.host_lengths.insert(
                        value.value,
                        Expr::Scalar(ScalarExpr::Read {
                            source: scalar_source(*binding),
                            offset: 8,
                            ty,
                        }),
                    );
                }
            }
            Ok(value)
        }
    }
}

fn save(
    body: &mut Body<'_, '_, '_>,
    live: &LookupSet<Value>,
    parameters: &LookupSet<Value>,
    saved: &mut StableMap<Value, SavedValue>,
) -> Result<(), OptimizeError> {
    // A view keeps its layout witnesses and backing handle live as well.
    let mut live = live.clone();
    let mut pending = live.iter().copied().collect::<Vec<_>>();
    while let Some(name) = pending.pop() {
        if let Some(info) = body.query().lookup("BindingInfo", (body.function, name))? {
            for dependency in names_in(body.query(), "NamesValueInfo", info)? {
                if live.insert(dependency) {
                    pending.push(dependency);
                }
            }
        }
    }
    saved.retain(|name, _| live.contains(name));
    let values = body
        .values
        .iter()
        .filter(|(name, _)| live.contains(name) && !parameters.contains(name))
        .map(|(&name, value)| (name, value.clone()))
        .collect::<Vec<_>>();
    let leader = if let Some(statement) = body.active_dispatch {
        let dispatch = *body.compiler.execution.dispatch(statement)?;
        let (group, _) = body.workgroups(dispatch.grid)?;
        let first = body.binary(BinaryOperator::Equal, group, Body::number(0))?;
        let lane = body.local_id()?;
        let first_lane = body.binary(BinaryOperator::Equal, lane, Body::number(0))?;
        Some(body.binary(BinaryOperator::LogicalAnd, first, first_lane)?)
    } else {
        None
    };
    for (name, value) in values {
        if saved.contains_key(&name) && body.transported_values.get(&name) == Some(&value.value) {
            continue;
        }
        if value.ty.is_array() && types::array_view_buffer(&value.ty).is_some() {
            if let Some(&parameter) = parameters.iter().find(|parameter| {
                body.values
                    .get(*parameter)
                    .is_some_and(|input| input.value == value.value && input.ty == value.ty)
            }) {
                saved.insert(name, SavedValue::Parameter(parameter));
                continue;
            }
        }
        if value.ty.is_array()
            || matches!(
                value.ty,
                Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), _)
            )
        {
            saved.insert(name, save_component(body, value, leader.as_ref(), false)?);
            continue;
        }
        let Some(generated) = body.generated.as_ref() else {
            return Err(error("dispatch buffers missing"));
        };
        let slot = generated.scalar_slots.get(&name).cloned();
        if slot.is_none() {
            if let Some(n) = body.literal_bits(&value) {
                saved.insert(name, SavedValue::Constant(n, value.ty));
                continue;
            }
        }
        if value.ty == types::unit() {
            continue;
        }
        let (binding, storage) = if let Some((binding, ty)) = slot {
            (
                binding,
                storage_view(body, binding, storage_value_type(&ty), Body::number(1))?,
            )
        } else {
            let Some(bytes) = storage_elem_stride(&storage_value_type(&value.ty)) else {
                return Err(error("cross-dispatch value has no storage size"));
            };
            body.device_buffer(&value.ty, Expr::Integer(bytes.into()), Body::number(1))?
        };
        if let Some(leader) = &leader {
            body.when(leader.clone(), |body| {
                body.store(storage, Body::number(0), value.clone())
            })?;
        } else {
            body.store(storage, Body::number(0), value.clone())?;
        }
        if !body.host_values.contains_key(&name) {
            if let Some(ty) = host::scalar_type(&value.ty) {
                body.host_values.insert(
                    name,
                    ScalarExpr::Read {
                        source: ScalarSource::Binding {
                            set: binding.set,
                            binding: binding.binding,
                        },
                        offset: 0,
                        ty,
                    },
                );
            }
        }
        let Some(generated) = body.generated.as_mut() else {
            return Err(error("dispatch buffers missing"));
        };
        generated.scalar_slots.insert(name, (binding, value.ty.clone()));
        saved.insert(
            name,
            match body.literal_bits(&value) {
                Some(n) => SavedValue::Constant(n, value.ty),
                None => SavedValue::Buffer(binding, value.ty),
            },
        );
    }
    Ok(())
}

fn save_component(
    body: &mut Body<'_, '_, '_>,
    mut value: Typed,
    leader: Option<&Typed>,
    materialize_arrays: bool,
) -> Result<SavedValue, OptimizeError> {
    if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = &value.ty {
        let mut saved = Vec::new();
        for i in 0..fields.len() {
            let field = body.field(value.clone(), i)?;
            saved.push(save_component(body, field, leader, materialize_arrays)?);
        }
        return Ok(SavedValue::Product(saved));
    }
    if let Some(n) = body.literal_bits(&value) {
        return Ok(SavedValue::Constant(n, value.ty));
    }
    if value.ty.is_array()
        && types::array_view_buffer(&value.ty).is_none()
        && (materialize_arrays || !value.ty.array_variant().is_some_and(types::is_array_variant_virtual))
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
        let count = body.array_length(value.clone())?;
        let bytes = Expr::Multiply(Box::new(length.clone()), Box::new(Expr::Integer(stride.into())));
        let (_, destination) = body.device_buffer(&element, bytes, count.clone())?;
        let copy = |body: &mut Body<'_, '_, '_>| {
            body.counted(
                Body::number(0),
                count,
                Body::number(1),
                vec![],
                |body, index, _| {
                    let item = body.index(value.clone(), index.clone())?;
                    body.store(destination.clone(), index, item)?;
                    Ok(vec![])
                },
            )?;
            Ok(())
        };
        if let Some(leader) = leader {
            body.when(leader.clone(), copy)?;
        } else {
            copy(body)?;
        }
        value = destination;
    }
    let (stored, ty, bytes, result) = if let Some(binding) = types::array_view_buffer(&value.ty) {
        let Some(element) = value.ty.elem_type().cloned() else {
            return Err(error("saved array has no element"));
        };
        let uint = Body::number(0).ty;
        let offset = body.op(OpTag::Project { index: 0 }, vec![value.clone()], uint.clone())?;
        let known_offset = array_offset(body, &value);
        // Host-known lengths survive dispatches; carried descriptors are refreshed at backedges.
        let host_length = body.host_array_length(&value).ok();
        let host_shape = body.host_shapes.get(&value.value);
        let tail = value.ty.array_dims().ok_or_else(|| error("saved array dimensions missing"))?[1..]
            .iter()
            .enumerate()
            .map(|(axis, ty)| {
                (
                    ty.clone(),
                    host_shape.and_then(|shape| shape.get(axis + 1)).cloned(),
                )
            })
            .collect::<Vec<_>>();
        let dimensions = body.view_dimensions(value.clone())?;
        let count = body.array_length(value)?;
        let length = body.integer_literal(&count);
        if !materialize_arrays && tail.is_empty() && length.is_some() && known_offset.is_some() {
            return Ok(SavedValue::Array {
                binding,
                element,
                descriptor: None,
                length,
                offset: known_offset,
                host_length,
                tail,
            });
        }
        let mut fields = vec![offset];
        for dimension in dimensions {
            fields.push(body.cast(dimension, &uint)?);
        }
        let bytes = u32::try_from(fields.len()).map_err(|_| error("descriptor rank exceeds u32"))? * 4;
        let stored = body.tuple(fields)?;
        let ty = stored.ty.clone();
        (
            stored,
            ty,
            bytes,
            Some((binding, element, length, known_offset, host_length, tail)),
        )
    } else {
        let Some(bytes) = storage_elem_stride(&storage_value_type(&value.ty)) else {
            return Err(error("cross-dispatch value has no storage size"));
        };
        (value.clone(), value.ty, bytes, None)
    };
    let (binding, slot) = body.device_buffer(&ty, Expr::Integer(bytes.into()), Body::number(1))?;
    if let Some(leader) = leader {
        body.when(leader.clone(), |body| body.store(slot, Body::number(0), stored))?;
    } else {
        body.store(slot, Body::number(0), stored)?;
    }
    Ok(match result {
        Some((storage, element, length, offset, host_length, tail)) => SavedValue::Array {
            binding: storage,
            element,
            descriptor: Some(binding),
            length,
            offset,
            host_length,
            tail,
        },
        None => SavedValue::Buffer(binding, ty),
    })
}

fn output_shape(body: &Body<'_, '_, '_>, value: &Typed) -> Result<Vec<Expr>, OptimizeError> {
    let mut shape = if let Some(shape) = body.host_shapes.get(&value.value) {
        shape.clone()
    } else {
        vec![body.host_array_length(value)?]
    };
    let mut ty = value.ty.elem_type();
    while let Some(element) = ty.filter(|ty| ty.is_array()) {
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

fn output_capacities(body: &Body<'_, '_, '_>, values: &[Typed]) -> Result<Vec<BufferLen>, OptimizeError> {
    values
        .iter()
        .map(|value| {
            if let Some(binding) = types::array_view_buffer(&value.ty) {
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
                Ok(BufferLen::Computed {
                    bytes: Expr::Multiply(
                        Box::new(body.host_array_length(value)?),
                        Box::new(Expr::Integer(stride.into())),
                    ),
                })
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
            let Some(element) = value.ty.elem_type() else {
                return Err(error("array output has no element"));
            };
            let destination = if value.ty.array_rank().is_some_and(|rank| rank > 1) {
                let dimensions = body.view_dimensions(value.clone())?;
                let mut fields = vec![storage_value_type(element), types::array_variant_view()];
                fields.extend(
                    value
                        .ty
                        .array_dims()
                        .ok_or_else(|| error("output dimensions missing"))?
                        .iter()
                        .cloned(),
                );
                fields.push(buffer_tag(binding));
                let mut operands = vec![Body::number(0)];
                operands.extend(dimensions);
                body.op(
                    OpTag::StorageView(PureViewSource::Storage(binding)),
                    operands,
                    Type::Constructed(TypeName::Array, fields),
                )?
            } else {
                storage_view(body, binding, storage_value_type(element), count.clone())?
            };
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
    body.op(
        OpTag::StorageView(PureViewSource::Storage(binding)),
        vec![zero, count],
        ty,
    )
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
            body.host_lengths.insert(value.value, host_length);
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
