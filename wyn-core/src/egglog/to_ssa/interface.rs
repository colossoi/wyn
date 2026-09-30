//! Source ABI declarations and publication for directly emitted entry bodies.
use super::kernels;
use super::plan::Stage;
use super::{builder_error, error, Body, Compiler, OptimizeError, Typed};
use crate::binding_layout::{
    extract_io_decoration, extract_sampler_binding, extract_storage_access, extract_storage_binding,
    extract_storage_image_binding, extract_storage_image_resource, extract_texture_backing,
    extract_texture_binding, extract_texture_resource, extract_uniform_binding,
};
use crate::builtins::catalog;
use crate::egglog::source::Term;
use crate::egglog::to_ssa::sizes;
use crate::flow::ExecutionModel;
use crate::host::BufferLen;
use crate::host::DispatchLen;
use crate::interface::lowering::{build_entry_outputs, extract_size_hint};
use crate::interface::StorageBindingDecl;
use crate::interface::StorageRole;
use crate::interface::{
    BindingExposure, EntryInput, EntryInputKind, EntryKind, EntryParamBindingKind, IoDecoration,
    PushConstantSlot, StorageAccess, TextureSource,
};
use crate::op::BinaryOperator;
use crate::op::{OpTag, PureViewSource};
use crate::ssa::layout::type_byte_size;
use crate::ssa::types::{EntryPoint, InstKind};
use crate::tlc::data::EntryInputBounds;
use crate::tlc::EntryPoint as SourceEntry;
use crate::types::{
    self, bool_type, buffer_tag, canonical_storage_buffer_ty, sized_array, strip_existentials, Type,
    TypeExt, TypeName,
};
use crate::{LookupMap, SymbolId};
use egglog_engine::Value;
use wyn_base::IdSource;

pub(super) fn entry<'source>(
    compiler: &mut Compiler<'_, 'source>,
    scope: Value,
    source: &'source Term,
    parameters: &[(SymbolId, Type)],
    entry: &SourceEntry<EntryInputBounds>,
    symbol: SymbolId,
    stage: Option<&Stage>,
) -> Result<EntryPoint, OptimizeError> {
    let decl = &entry.declaration;
    let mut lower = Body::new(compiler, scope, vec![], source.ty.clone())?;
    lower.grid = lower.compiler.plan.entry_grid(
        symbol,
        stage,
        decl.compute_dispatch.map(|grid| (grid.x, grid.y, grid.z)),
    );
    let mut inputs = Vec::new();
    let mut parameter_inputs = Vec::new();
    let mut offset = 0u32;
    for (index, (symbol, ty)) in parameters.iter().enumerate() {
        let Some(param) = decl.params.get(index) else {
            return Err(error("missing input declaration"));
        };
        let binding = entry.data.param_bindings.get(index).and_then(Option::as_ref);
        let mut declared = Vec::new();
        if let Some(EntryParamBindingKind::TupleOfViews(fields)) = binding.map(|binding| &binding.kind) {
            let Type::Constructed(TypeName::Tuple(_), types) = strip_existentials(ty) else {
                return Err(error("tuple input has no tuple type"));
            };
            for (i, (field, ty)) in fields.iter().zip(types).enumerate() {
                declared.push(EntryInput {
                    name: format!("{}_{}", param.name, i),
                    ty: canonical_storage_buffer_ty(ty),
                    size_hint: extract_size_hint(param),
                    kind: EntryInputKind::Storage {
                        exposure: BindingExposure::Host(field.binding),
                        access: StorageAccess::ReadOnly,
                        length: None,
                    },
                });
            }
        } else {
            let storage =
                binding.map(|binding| binding.first_buffer().0).or_else(|| extract_storage_binding(param));
            let decoration = extract_io_decoration(param);
            let kind = if let Some(binding) = storage {
                EntryInputKind::Storage {
                    exposure: BindingExposure::Host(binding),
                    access: extract_storage_access(param).unwrap_or(StorageAccess::ReadOnly),
                    length: entry.data.by_symbol.get(symbol).cloned().or_else(|| {
                        type_byte_size(ty).map(|bytes| BufferLen::Fixed { bytes: bytes.into() })
                    }),
                }
            } else if let Some(binding) = extract_uniform_binding(param) {
                EntryInputKind::Uniform { binding }
            } else if let Some(binding) = extract_texture_binding(param) {
                let source = match (extract_texture_backing(param), extract_texture_resource(param)) {
                    (backing, Some(name)) => TextureSource::Resource { name, backing },
                    (Some(binding), None) => TextureSource::Backing(binding),
                    _ => TextureSource::External,
                };
                EntryInputKind::Texture { binding, source }
            } else if let Some(binding) = extract_sampler_binding(param) {
                EntryInputKind::Sampler { binding }
            } else if let Some((binding, format, access, size)) = extract_storage_image_binding(param) {
                EntryInputKind::StorageImage {
                    binding,
                    format,
                    access,
                    size,
                    resource: extract_storage_image_resource(param),
                }
            } else if decl.entry_kind != EntryKind::Compute
                || matches!(decoration, Some(IoDecoration::BuiltIn(_)))
            {
                EntryInputKind::Value { decoration }
            } else {
                // Parameters are block members: aggregates include their tail padding.
                let Some((size, align)) = crate::ssa::layout::std430_type_layout(&storage_type(ty)?) else {
                    return Err(error("input has no byte layout"));
                };
                offset = offset.div_ceil(align) * align;
                let slot = PushConstantSlot { offset, size };
                offset += size;
                EntryInputKind::PushConstant { slot }
            };
            declared.push(EntryInput {
                name: param.name.clone(),
                ty: if *ty == bool_type() {
                    Type::Constructed(TypeName::UInt(32), vec![])
                } else {
                    canonical_storage_buffer_ty(ty)
                },
                size_hint: extract_size_hint(param),
                kind,
            });
        }
        let first = inputs.len();
        let mut values = Vec::new();
        for mut input in declared {
            let scalar_storage = input.storage_binding().is_some() && !input.ty.is_array();
            if scalar_storage {
                input.ty = sized_array(1, input.ty.clone());
            }
            let physical = if let Some(binding) = input.storage_binding() {
                let Some(element) = input.ty.elem_type() else {
                    return Err(error("storage input has no element type"));
                };
                view_type(&storage_type(element)?, buffer_tag(binding))
            } else {
                concrete(&input.ty)?
            };
            let parameter =
                lower.builder.func_mut().add_function_param(physical.clone(), input.name.clone());
            let mut value = Typed {
                value: parameter.into(),
                ty: physical,
            };
            if let Some(binding) = input.storage_binding() {
                let len = if let Some(Type::Constructed(TypeName::Size(n), _)) = input.ty.array_size() {
                    lower.literal(&n.to_string(), &Type::Constructed(TypeName::UInt(32), vec![]))?
                } else {
                    let set = lower.literal(
                        &binding.set.to_string(),
                        &Type::Constructed(TypeName::UInt(32), vec![]),
                    )?;
                    let slot = lower.literal(
                        &binding.binding.to_string(),
                        &Type::Constructed(TypeName::UInt(32), vec![]),
                    )?;
                    lower.op(
                        OpTag::Intrinsic {
                            id: catalog().known().storage_len,
                            overload_idx: 0,
                        },
                        vec![set, slot],
                        Type::Constructed(TypeName::UInt(32), vec![]),
                    )?
                };
                let zero = lower.literal("0", &Type::Constructed(TypeName::UInt(32), vec![]))?;
                value = lower.op(
                    OpTag::StorageView(PureViewSource::Storage(binding)),
                    vec![zero, len],
                    value.ty,
                )?;
            }

            if scalar_storage {
                let zero = lower.literal("0", &Type::Constructed(TypeName::UInt(32), vec![]))?;
                value = lower.index(value, zero)?;
            }
            values.push(value);
            inputs.push(input);
        }
        let value = if values.len() == 1 {
            values.remove(0)
        } else {
            lower.op(
                OpTag::Tuple(values.len()),
                values.clone(),
                types::tuple(values.iter().map(|value| value.ty.clone()).collect()),
            )?
        };
        let value = lower.cast(value, ty)?;
        let Some(formal) = lower.compiler.facts.parameter(scope, index as i64) else {
            return Err(error("missing entry parameter"));
        };
        if let Some(input) = inputs.get(first) {
            lower.compiler.input_interfaces.insert(formal, input.clone());
            let host_length = match &input.kind {
                EntryInputKind::Storage {
                    exposure: BindingExposure::Host(binding),
                    ..
                } => {
                    input.ty.elem_type().and_then(type_byte_size).map(|stride| DispatchLen::InputBinding {
                        set: binding.set,
                        binding: binding.binding,
                        elem_bytes: stride,
                    })
                }
                EntryInputKind::PushConstant { slot } => {
                    Some(DispatchLen::PushConstant { offset: slot.offset })
                }
                _ => None,
            };
            if let Some(length) = host_length {
                lower.compiler.host_lengths.insert(formal, length);
            }
        }
        lower.values.insert(formal, value);
        parameter_inputs.push((first..inputs.len()).collect());
    }
    let owner = decl
        .graphics_group
        .as_ref()
        .and_then(|group| lower.compiler.program.source.symbols.get(group.root))
        .unwrap_or(&decl.name);
    let phase = match stage.map(|stage| stage.phase.as_str()) {
        Some("elements" | "scalar") => "compute",
        Some("chunks") => "partials",
        Some(phase) => phase,
        None => match decl.entry_kind {
            EntryKind::Vertex => "vertex",
            EntryKind::Fragment => "fragment",
            _ if lower.compiler.plan.stages.iter().any(|s| s.owner == symbol) => "finish",
            _ => "compute",
        },
    };
    let name = super::plan::unique(format!("{owner}_{phase}"), &mut lower.compiler.entry_names);
    if stage.is_some_and(|s| s.phase != "scalar") && decl.compute_dispatch.is_none() {
        lower.host_stage = Some(name.clone());
    }
    let compute = decl.entry_kind == EntryKind::Compute;
    let result = if let Some(stage) = stage {
        kernels::emit(&mut lower, scope, stage)?;
        lower.op(OpTag::Unit, vec![], types::unit())?
    } else if compute {
        lower.op(OpTag::Unit, vec![], types::unit())?
    } else {
        lower.source(scope, source)?
    };
    let mut bindings = IdSource::<u32>::new();
    let maximum = inputs
        .iter()
        .filter_map(EntryInput::storage_binding)
        .filter(|binding| binding.set == 0)
        .map(|binding| binding.binding)
        .max();
    let maximum = Some(maximum.unwrap_or(0).max(lower.compiler.plan.next_binding.saturating_sub(1)));
    if let Some(maximum) = maximum {
        for _ in 0..=maximum {
            bindings.next_id();
        }
    }
    let outputs = if compute {
        vec![]
    } else {
        build_entry_outputs(
            decl,
            &result.ty,
            &[],
            &inputs,
            decl.entry_kind == EntryKind::Compute,
            &mut bindings,
        )
        .map_err(|err| error(err.to_string()))?
    };
    for (index, output) in outputs.iter().enumerate() {
        let value = if outputs.len() == 1 { result.clone() } else { lower.field(result.clone(), index)? };
        if let Some(binding) = output.storage_binding() {
            let element = value.ty.elem_type().filter(|_| value.ty.is_array()).unwrap_or(&value.ty);
            let element = storage_type(element)?;
            let len = if value.ty.is_array() {
                lower.length(value.clone())?
            } else {
                lower.literal("1", &types::i32())?
            };
            let zero = lower.literal("0", &types::i32())?;
            let view = lower.op(
                OpTag::StorageView(PureViewSource::Storage(binding)),
                vec![zero.clone(), len.clone()],
                view_type(&element, buffer_tag(binding)),
            )?;
            if value.ty.is_array() {
                lower.copy_array(view, value, len)?;
            } else {
                let value = lower.cast(value, &element)?;
                let (place, _) = lower.index_place(view, zero)?;
                lower
                    .builder
                    .push_void_inst(InstKind::Store {
                        place,
                        value: value.value,
                    })
                    .map_err(builder_error)?;
            }
        } else {
            let value = lower.cast(value, &output.ty)?;
            let place = lower.builder.new_place(value.ty.clone());
            lower
                .builder
                .push_void_inst(InstKind::OutputSlot { index, result: place })
                .map_err(builder_error)?;
            lower
                .builder
                .push_void_inst(InstKind::Store {
                    place,
                    value: value.value,
                })
                .map_err(builder_error)?;
        }
    }
    if compute {
        let planned = lower
            .compiler
            .plan
            .outputs
            .iter()
            .filter(|output| output.owner == symbol && output.copy && output.writer == stage.map(|s| s.key))
            .cloned()
            .collect::<Vec<_>>();
        let write = |lower: &mut Body<'_, '_, 'source>| -> Result<(), OptimizeError> {
            for output in planned {
                let value = lower.value(scope, output.source)?;
                let destination = lower.resource(scope, output.resource, 2)?;
                if value.ty.is_array() {
                    let length = lower.length(value.clone())?;
                    lower.copy_array(destination, value, length)?;
                } else {
                    let zero = lower.literal("0", &types::i32())?;
                    let (place, ty) = lower.index_place(destination, zero)?;
                    let value = lower.stored(value, &ty)?;
                    lower
                        .builder
                        .push_void_inst(InstKind::Store {
                            place,
                            value: value.value,
                        })
                        .map_err(builder_error)?;
                }
            }
            Ok(())
        };
        if stage.is_some_and(|stage| stage.width > 1) {
            let uint = Type::Constructed(TypeName::UInt(32), vec![]);
            let lane = lower.op(
                OpTag::Intrinsic {
                    id: catalog().known().local_id,
                    overload_idx: 0,
                },
                vec![],
                uint.clone(),
            )?;
            let zero = lower.literal("0", &uint)?;
            let first = lower.binary(BinaryOperator::Equal, lane, zero)?;
            lower.when(first, write)?;
        } else {
            write(&mut lower)?;
        }
    }
    let unit = lower.op(OpTag::Unit, vec![], types::unit())?;
    let resource_uses = lower.resource_uses.clone();
    let captures = std::mem::take(&mut lower.capture_bindings);
    let body = lower.finish(unit)?;
    for (&resource, &access) in &resource_uses {
        if let Some(source) = compiler.plan.external(resource) {
            if let Some(DispatchLen::InputBinding { set, binding, .. }) = compiler.host_lengths.get(&source)
            {
                for input in &mut inputs {
                    if let EntryInputKind::Storage {
                        exposure: BindingExposure::Host(slot),
                        access: mode,
                        ..
                    } = &mut input.kind
                    {
                        if slot.set == *set && slot.binding == *binding {
                            *mode = match access {
                                1 => StorageAccess::ReadOnly,
                                2 => StorageAccess::ReadWrite,
                                _ => StorageAccess::ReadWrite,
                            };
                        }
                    }
                }
            }
        }
    }
    let mut storage_bindings = captures;
    for (&resource, &access) in &resource_uses {
        if let Some(buffer) = compiler.plan.buffers.get(&resource) {
            storage_bindings.push(StorageBindingDecl {
                binding: buffer.binding,
                role: match access {
                    1 => StorageRole::Input,
                    2 => StorageRole::Output,
                    _ => StorageRole::InputOutput,
                },
                logical_resource: Some(buffer.name.clone()),
                elem_ty: storage_type(&buffer.element)?,
                length: Some(sizes::capacity(compiler, buffer)?),
            });
        }
    }
    storage_bindings.sort_by_key(|buffer| buffer.binding.binding);
    let id = compiler.entry_ids.next_id();
    compiler.entry_origins.insert(id, (symbol, stage.cloned()));
    let execution_model = match decl.entry_kind {
        EntryKind::Vertex => ExecutionModel::Vertex,
        EntryKind::Fragment => ExecutionModel::Fragment,
        EntryKind::Compute => ExecutionModel::Compute {
            local_size: (stage.map_or(1, |stage| stage.width), 1, 1),
        },
        EntryKind::Root => return Err(error("unextracted graphics entry")),
    };
    Ok(EntryPoint {
        id,
        name,
        body,
        execution_model,
        inputs,
        parameter_inputs,
        outputs,
        storage_bindings,
        stage_descriptor_storage_accesses: LookupMap::default(),
        pipeline_storage_accesses: LookupMap::default(),
        span: source.span,
    })
}

pub(super) fn concrete(ty: &Type) -> Result<Type, OptimizeError> {
    let ty = strip_existentials(ty);
    if ty.array_variant().is_some_and(types::is_array_variant_view) {
        return Ok(ty.clone());
    }
    if let Some(element) = ty.elem_type().filter(|_| ty.is_array()) {
        let Some(Type::Constructed(TypeName::Size(count), _)) = ty.array_size() else {
            return Err(error("runtime-sized array requires storage"));
        };
        return Ok(sized_array((*count).max(1), concrete(element)?));
    }
    match ty {
        Type::Constructed(name, fields) => Ok(Type::Constructed(
            name.clone(),
            fields.iter().map(concrete).collect::<Result<_, _>>()?,
        )),
        Type::Variable(_) => Err(error("unresolved scalar type")),
    }
}
pub(super) fn storage_type(ty: &Type) -> Result<Type, OptimizeError> {
    concrete(&crate::ssa::layout::storage_value_type(ty))
}

pub(super) fn view_type(element: &Type, region: Type) -> Type {
    types::view_array_with_size(
        element,
        Type::Constructed(TypeName::SizePlaceholder, vec![]),
        region,
    )
}
