//! Source ABI declarations and publication for directly emitted entry bodies.
use super::kernels;

use super::{builder_error, error, Body, Compiler, OptimizeError, Typed};
use crate::builtins::catalog;
use crate::flow::ExecutionModel;
use crate::interface::EntryKind;
use crate::op::BinaryOperator;
use crate::op::{OpTag, PureViewSource};
use crate::ssa::types::{EntryPoint, InstKind};
use crate::types::{self, buffer_tag, Type, TypeExt, TypeName};
use crate::SymbolId;
use egglog_engine::Value;

pub(super) fn entry<'source>(
    compiler: &mut Compiler<'_, 'source>,
    owner: SymbolId,
    stage: Option<Value>,
    published: &[EntryPoint],
) -> Result<EntryPoint, OptimizeError> {
    let definition = compiler
        .program
        .source
        .defs
        .iter()
        .find(|d| d.name == owner)
        .ok_or_else(|| error("entry definition missing"))?;
    let crate::tlc::DefMeta::EntryPoint(entry) = &definition.meta else {
        return Err(error("entry declaration missing"));
    };
    let scope = compiler.facts.definition(owner).ok_or_else(|| error("entry region missing"))?;
    let token =
        compiler.program.identities.symbols.get(&owner).ok_or_else(|| error("entry identity missing"))?;
    let root = match stage {
        Some(stage) => compiler.facts.constructor("KernelRoot", (stage,)),
        None => compiler.facts.constructor("EntryRoot", (token,)),
    }
    .ok_or_else(|| error("entry root missing"))?;
    let (source, parameters) = crate::tlc::extract_lambda_params_ref(&definition.body);
    let original = compiler.facts.contains("EmitOriginalEntry", (token,));
    let outputs = if stage.is_none() && original {
        let Some(outputs) = compiler.plan.entry_outputs.remove(&owner) else {
            return Err(OptimizeError::Output("entry output ABI missing".into()));
        };
        outputs
    } else {
        Vec::new()
    };
    let declaration = &entry.declaration;
    let name = if let Some(group) = &declaration.graphics_group {
        let Some(name) = compiler.program.source.symbols.get(group.root) else {
            return Err(OptimizeError::Output("graphics owner name missing".into()));
        };
        name
    } else {
        &declaration.name
    };
    let selected_phase = stage.map(|key| compiler.plan.phase_name(key)).transpose()?;
    let phase = match selected_phase.as_deref() {
        Some("elements" | "scalar") => "compute",
        Some("chunks") => "partials",
        Some(phase) => phase,
        None => match declaration.entry_kind {
            EntryKind::Vertex => "vertex",
            EntryKind::Fragment => "fragment",
            _ if compiler.facts.contains("FinishEntry", (token,)) => "finish",
            _ => "compute",
        },
    };
    let name = super::plan::unique(format!("{name}_{phase}"), &mut compiler.entry_names);
    let execution_model = match declaration.entry_kind {
        EntryKind::Vertex => ExecutionModel::Vertex,
        EntryKind::Fragment => ExecutionModel::Fragment,
        EntryKind::Compute => {
            let grid = crate::egglog::abi::required(&compiler.program.graph, "RootWorkgroup", (root,))?;
            ExecutionModel::Compute {
                local_size: compiler.facts.grid(grid)?,
            }
        }
        EntryKind::Root => return Err(OptimizeError::Output("unextracted graphics entry".into())),
    };
    let id = compiler.entry_ids.next_id();
    compiler.entry_origins.insert(id, (owner, stage));

    let decl = &entry.declaration;
    let symbol = owner;
    let mut lower = Body::new(compiler, scope, vec![], source.ty.clone())?;
    lower.grid = stage.map(|key| lower.compiler.plan.grid(key)).transpose()?.flatten();
    if stage.is_none() {
        let Some(token) = lower.compiler.program.identities.symbols.get(&symbol) else {
            return Err(error("entry symbol missing"));
        };
        let Some(grid) = lower.compiler.facts.lookup("OriginalEntryGrid", (token,)) else {
            return Err(error("original entry has no launch grid"));
        };
        lower.grid = Some(lower.compiler.facts.grid(grid)?);
    }
    let mut inputs = Vec::new();
    let mut parameter_inputs = Vec::new();
    for (index, (_, ty)) in parameters.iter().enumerate() {
        let declared = lower.compiler.facts.parameter_inputs(scope, index as i64)?;
        let first = inputs.len();
        let mut values = Vec::new();
        for selected in declared {
            let mut input = selected;
            let scalar_storage = input.storage_binding().is_some() && !input.ty.is_array();
            if scalar_storage {
                input.ty = types::sized_array(1, input.ty.clone());
            }
            let physical = if let Some(binding) = input.storage_binding() {
                let element = input.ty.elem_type().ok_or_else(|| error("storage input has no element"))?;
                view_type(
                    &lower.compiler.facts.physical_type(element, true)?,
                    buffer_tag(binding),
                )
            } else {
                lower.compiler.facts.physical_type(&input.ty, false)?
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
        lower.values.insert(formal, value);
        parameter_inputs.push((first..inputs.len()).collect());
    }
    let storage_bindings =
        crate::egglog::abi::storage_bindings(lower.compiler, owner, stage, published, &mut inputs)?;
    let compute = decl.entry_kind == EntryKind::Compute;
    let original = lower
        .compiler
        .program
        .identities
        .symbols
        .get(&symbol)
        .is_some_and(|token| lower.compiler.facts.contains("EmitOriginalEntry", (token,)));
    let result = if let Some(stage) = stage {
        let operation = lower.compiler.plan.phase_operation(stage)?;
        let phase = lower.compiler.plan.phase_name(stage)?;
        let extent = lower.compiler.plan.phase_extent(stage)?;
        let width = lower.compiler.plan.phase_width(stage)?;
        kernels::emit(&mut lower, scope, operation, &phase, extent, width)?;
        lower.op(OpTag::Unit, vec![], types::unit())?
    } else if compute && !original {
        lower.op(OpTag::Unit, vec![], types::unit())?
    } else {
        let Some(result) = lower.compiler.facts.result(scope) else {
            return Err(error("entry point has no selected result"));
        };
        lower.value(scope, result)?
    };
    for (index, output) in outputs.iter().enumerate() {
        let value = if outputs.len() == 1 { result.clone() } else { lower.field(result.clone(), index)? };
        if let Some(binding) = output.storage_binding() {
            let element = if value.ty.is_array() {
                let Some(element) = value.ty.elem_type() else {
                    return Err(error("array output has no element type"));
                };
                element
            } else {
                &value.ty
            };
            let element = lower.compiler.facts.physical_type(element, true)?;
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
        let planned = lower.compiler.plan.outputs(symbol)?;
        let write = |lower: &mut Body<'_, '_, 'source>| -> Result<(), OptimizeError> {
            for id in planned {
                if !lower.compiler.facts.contains("CopyOutput", (id,))
                    || lower.compiler.facts.lookup("SsaOutputWriter", (id,)) != stage
                {
                    continue;
                }
                let (source, ty, resource) = lower.compiler.plan.output(id)?;
                if ty.is_array() || types::as_soa_tuple(ty).is_some() {
                    let (array, fields) = lower.source_array(scope, source)?;
                    let length = lower.length(array.clone())?;
                    let destination = lower.resource(scope, resource, 2)?;
                    let zero = lower.literal("0", &types::i32())?;
                    let one = lower.literal("1", &types::i32())?;
                    lower.counted(zero, length, one, vec![], |lower, index, _| {
                        let mut value = lower.index(array, index.clone())?;
                        for field in fields {
                            value = lower.field(value, field)?;
                        }
                        kernels::store(lower, destination, index, value)?;
                        Ok(vec![])
                    })?;
                } else {
                    let value = lower.value(scope, source)?;
                    let destination = lower.resource(scope, resource, 2)?;
                    let zero = lower.literal("0", &types::i32())?;
                    let (place, ty) = lower.index_place(destination, zero)?;
                    let value = lower.cast(value, &ty)?;
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
        if stage
            .map(|key| lower.compiler.plan.phase_width(key).map(|width| width > 1))
            .transpose()?
            .unwrap_or(false)
        {
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
    let body = lower.finish(unit)?;
    Ok(EntryPoint {
        id,
        name,
        body,
        execution_model,
        inputs,
        parameter_inputs,
        outputs,
        storage_bindings,
        stage_descriptor_storage_accesses: crate::egglog::abi::entry_accesses(compiler, symbol, stage)?,
        pipeline_storage_accesses: crate::egglog::abi::pipeline_accesses(compiler, symbol)?,
        span: source.span,
    })
}

pub(super) fn view_type(element: &Type, region: Type) -> Type {
    types::view_array_with_size(
        element,
        Type::Constructed(TypeName::SizePlaceholder, vec![]),
        region,
    )
}
