//! Resolve source ABI declarations before emitting any shader instructions.
use super::query::Query;
use super::{OptimizeError, Program};
use crate::binding_layout::{
    extract_io_decoration, extract_sampler_binding, extract_storage_binding, extract_storage_image_binding,
    extract_storage_image_resource, extract_texture_backing, extract_texture_binding,
    extract_texture_resource, extract_uniform_binding,
};
use crate::egglog::to_ssa::{plan::unique, Compiler};
use crate::flow::ExecutionModel;
use crate::host::BufferLen;
use crate::interface::lowering::{build_entry_outputs, extract_size_hint};
use crate::interface::{
    BindingExposure, EntryInput, EntryInputKind, EntryParamBindingKind, PushConstantSlot, StorageAccess,
    TextureSource,
};
use crate::interface::{EntryKind, EntryPublication, StorageBindingDecl, StorageRole};
use crate::ssa::layout::{storage_value_type, type_byte_size};
use crate::tlc::{extract_lambda_params_ref, DefMeta};
use crate::types::{self, canonical_storage_buffer_ty, strip_existentials, Type, TypeExt, TypeName};
use crate::SymbolId;
use crate::{BindingRef, LookupMap, ResourceAccess};
use egglog_engine::{EGraph, IntoValues, Read, Value};
use std::collections::BTreeSet;
use wyn_base::IdSource;

pub(super) mod publication;
mod sizes;

pub(super) fn entry_accesses(
    compiler: &Compiler<'_, '_>,
    owner: SymbolId,
    stage: Option<Value>,
) -> Result<LookupMap<BindingRef, ResourceAccess>, OptimizeError> {
    let Some(token) = compiler.program.identities.symbols.get(&owner) else {
        return Err(OptimizeError::Output("entry identity missing".into()));
    };
    let root = match stage {
        Some(stage) => compiler.facts.constructor("KernelRoot", (stage,)),
        None => compiler.facts.constructor("EntryRoot", (token,)),
    };
    let Some(root) = root else {
        return Err(OptimizeError::Output("entry resource root missing".into()));
    };
    let mut result = root_accesses(compiler, root)?;
    if let Some(stage) = stage {
        for (&term, capture) in &compiler.program.stage.captures {
            if capture.stages.contains(&stage) {
                result.insert(compiler.plan.captures[&term], ResourceAccess::Read);
            }
        }
    }
    Ok(result)
}

pub(super) fn pipeline_accesses(
    compiler: &Compiler<'_, '_>,
    owner: SymbolId,
) -> Result<LookupMap<BindingRef, ResourceAccess>, OptimizeError> {
    let Some(token) = compiler.program.identities.symbols.get(&owner) else {
        return Err(OptimizeError::Output("pipeline identity missing".into()));
    };
    let mut result = LookupMap::default();
    Query(&compiler.program.graph).for_each("RootOwner", |row| {
        if compiler.facts.integer(row[1]) == token {
            for (binding, access) in root_accesses(compiler, row[0])? {
                result
                    .entry(binding)
                    .and_modify(|old: &mut ResourceAccess| *old = old.merge(access))
                    .or_insert(access);
            }
        }
        Ok(())
    })?;
    for (&term, capture) in &compiler.program.stage.captures {
        if capture.stages.iter().any(|&stage| compiler.facts.contains("PhaseOwner", (stage, token))) {
            result.insert(compiler.plan.captures[&term], ResourceAccess::Read);
        }
    }
    Ok(result)
}

fn root_accesses(
    compiler: &Compiler<'_, '_>,
    root: Value,
) -> Result<LookupMap<BindingRef, ResourceAccess>, OptimizeError> {
    let mut result = binding_accesses(compiler, "RootBindingAccess", |key| key == root)?;
    for value in compiler.facts.set("RootResourceSet", (root,)) {
        let flags = required(&compiler.program.graph, "RootResource", (root, value))?;
        let access = match compiler.facts.integer(flags) {
            0 => continue,
            1 => ResourceAccess::Read,
            2 => ResourceAccess::Write,
            3 => ResourceAccess::ReadWrite,
            _ => return Err(OptimizeError::Output("invalid root access".into())),
        };
        if let Some((binding, _, _)) = compiler.plan.buffer(value)? {
            result.entry(binding).and_modify(|old| *old = old.merge(access)).or_insert(access);
        }
    }
    Ok(result)
}

fn binding_accesses(
    compiler: &Compiler<'_, '_>,
    table: &str,
    matches: impl Fn(Value) -> bool,
) -> Result<LookupMap<BindingRef, ResourceAccess>, OptimizeError> {
    let query = Query(&compiler.program.graph);
    let mut result = LookupMap::default();
    query.for_function(table, |keys, flags| {
        if !matches(keys[0]) {
            return Ok(());
        }
        let Some(binding) = query.enode("InputBinding", keys[1])? else {
            return Err(OptimizeError::Output("selected access has no binding".into()));
        };
        let access = match compiler.facts.integer(flags) {
            1 => ResourceAccess::Read,
            2 => ResourceAccess::Write,
            3 => ResourceAccess::ReadWrite,
            _ => return Err(OptimizeError::Output("invalid selected binding access".into())),
        };
        result.insert(
            BindingRef::new(
                compiler.facts.unsigned(binding[0], "access set")?,
                compiler.facts.unsigned(binding[1], "access binding")?,
            ),
            access,
        );
        Ok(())
    })?;
    Ok(result)
}

pub(super) fn entry(
    compiler: &mut Compiler<'_, '_>,
    owner: SymbolId,
    stage: Option<Value>,
    published: &[EntryPublication],
) -> Result<EntryPublication, OptimizeError> {
    let Some(definition) = compiler.program.source.defs.iter().find(|d| d.name == owner) else {
        return Err(OptimizeError::Output("entry definition missing".into()));
    };
    let DefMeta::EntryPoint(entry) = &definition.meta else {
        return Err(OptimizeError::Output("entry declaration missing".into()));
    };
    let Some(scope) = compiler.facts.definition(owner) else {
        return Err(OptimizeError::Output("entry region missing".into()));
    };
    let Some(token) = compiler.program.identities.symbols.get(&owner) else {
        return Err(OptimizeError::Output("entry identity missing".into()));
    };
    let root = match stage {
        Some(stage) => compiler.facts.constructor("KernelRoot", (stage,)),
        None => compiler.facts.constructor("EntryRoot", (token,)),
    };
    let Some(root) = root else {
        return Err(OptimizeError::Output("entry root missing".into()));
    };
    let mut inputs = Vec::new();
    for i in 0..entry.declaration.params.len() {
        inputs.extend(
            compiler.facts.parameter_inputs(scope, i as i64)?.into_iter().map(|input| input.declaration),
        );
    }
    let mut storage_bindings = Vec::new();
    let mut resource_names: BTreeSet<_> = compiler
        .program
        .source
        .defs
        .iter()
        .filter_map(|d| {
            let DefMeta::EntryPoint(entry) = &d.meta else {
                return None;
            };
            Some(entry.declaration.params.iter().map(|p| p.name.clone()))
        })
        .flatten()
        .chain(
            published
                .iter()
                .flat_map(|entry| entry.storage_bindings.iter())
                .filter_map(|storage| storage.logical_resource.clone()),
        )
        .collect();
    for value in compiler.facts.set("RootResourceSet", (root,)) {
        let flags = required(&compiler.program.graph, "RootResource", (root, value))?;
        let access = match compiler.facts.integer(flags) {
            0 => None,
            1 => Some(StorageAccess::ReadOnly),
            2 => Some(StorageAccess::WriteOnly),
            3 => Some(StorageAccess::ReadWrite),
            _ => return Err(OptimizeError::Output("invalid planned access".into())),
        };
        if let Some((binding, element, _)) = compiler.plan.buffer(value)? {
            let name = if let Some(name) = published
                .iter()
                .flat_map(|entry| &entry.storage_bindings)
                .find(|storage| storage.binding == binding)
                .and_then(|storage| storage.logical_resource.clone())
            {
                name
            } else {
                let name = compiler.plan.buffer_name(value)?;
                if compiler.facts.lookup("PhysicalBinding", (value,)).is_some() {
                    resource_names.insert(name.clone());
                    name
                } else {
                    unique(name, &mut resource_names)
                }
            };
            storage_bindings.push(StorageBindingDecl {
                binding,
                role: match access {
                    Some(StorageAccess::ReadOnly) | None => StorageRole::Input,
                    Some(StorageAccess::WriteOnly) => StorageRole::Output,
                    Some(StorageAccess::ReadWrite) => StorageRole::InputOutput,
                },
                logical_resource: Some(name),
                elem_ty: compiler.facts.physical_type(element, true)?,
                length: Some(sizes::capacity(compiler, value)?),
            });
        } else {
            let Some(access) = access else {
                return Err(OptimizeError::Output(
                    "runtime loop storage has no allocation".into(),
                ));
            };
            let Some(source) = compiler.plan.external(value) else {
                return Err(OptimizeError::Output("external planned storage missing".into()));
            };
            let Some((binding, _)) = compiler.facts.input_storage(source)? else {
                return Err(OptimizeError::Output(format!(
                    "external storage {source:?} has no binding; type {:?}",
                    compiler.facts.source_type(source)
                )));
            };
            for input in &mut inputs {
                if input.storage_binding() == Some(binding) {
                    if let EntryInputKind::Storage { access: selected, .. } = &mut input.kind {
                        *selected = selected.merge(access);
                    }
                }
            }
        }
    }
    if let Some(stage) = stage {
        for (&term, capture) in &compiler.program.stage.captures {
            if !capture.stages.contains(&stage) {
                continue;
            }
            let binding = compiler.plan.captures[&term];
            let (_, fields) = compiler.program.stage.selected.app(term)?;
            let Some(ty) = compiler.facts.ty(compiler.program.stage.selected.values[fields[1]]) else {
                return Err(OptimizeError::Output("capture type missing".into()));
            };
            storage_bindings.push(StorageBindingDecl {
                binding,
                role: StorageRole::Input,
                logical_resource: Some(format!("host_capture_{}_{}", binding.set, binding.binding)),
                elem_ty: storage_value_type(ty),
                length: Some(BufferLen::Fixed { bytes: 4 }),
            });
        }
    }
    let original = compiler.facts.contains("EmitOriginalEntry", (token,));
    let outputs = if stage.is_none() && original {
        let Some(outputs) = compiler.plan.entry_outputs.get(&owner) else {
            return Err(OptimizeError::Output("entry output ABI missing".into()));
        };
        outputs.clone()
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
    let recipe = stage.map(|key| compiler.plan.recipe(key)).transpose()?;
    let phase = match recipe.as_ref().map(|stage| stage.phase.as_str()) {
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
    let name = unique(format!("{name}_{phase}"), &mut compiler.entry_names);
    let execution_model = match declaration.entry_kind {
        EntryKind::Vertex => ExecutionModel::Vertex,
        EntryKind::Fragment => ExecutionModel::Fragment,
        EntryKind::Compute => {
            let grid = required(&compiler.program.graph, "RootWorkgroup", (root,))?;
            ExecutionModel::Compute {
                local_size: compiler.facts.grid(grid)?,
            }
        }
        EntryKind::Root => return Err(OptimizeError::Output("unextracted graphics entry".into())),
    };
    let id = compiler.entry_ids.next_id();
    compiler.entry_origins.insert(id, (owner, stage));

    Ok(EntryPublication {
        id,
        name,
        execution_model,
        inputs,
        outputs,
        storage_bindings,
    })
}

#[derive(Clone)]
pub(super) struct Input {
    pub declaration: EntryInput,
    pub parameter_type: Type,
    pub scalar_storage: bool,
}

pub(super) fn required(graph: &EGraph, table: &str, keys: impl IntoValues) -> Result<Value, OptimizeError> {
    let Some(value) = graph.read(|r| r.lookup(table, keys))? else {
        return Err(OptimizeError::Output(format!("missing selected {table}")));
    };
    Ok(value)
}

pub(super) fn fields(
    graph: &EGraph,
    name: &str,
    value: Value,
) -> Result<Option<Vec<Value>>, OptimizeError> {
    let mut rows = Vec::new();
    graph.read(|r| r.enodes_for_eclass(name, value, |row| rows.push(row.children.to_vec())))?;
    if rows.len() > 1 {
        return Err(OptimizeError::Output(format!("ambiguous selected {name}")));
    }
    Ok(rows.pop())
}

pub(super) fn parameter_inputs(
    program: &Program<'_, super::Optimized>,
    scope: Value,
    index: i64,
) -> Result<Vec<Input>, OptimizeError> {
    let facts = super::to_ssa::read::Facts { program };
    let Some(symbol) = facts.definition_name(scope) else {
        return Err(OptimizeError::Output("parameter region is not an entry".into()));
    };
    let Some(definition) = program.source.defs.iter().find(|d| d.name == symbol) else {
        return Err(OptimizeError::Output("entry declaration missing".into()));
    };
    let DefMeta::EntryPoint(entry) = &definition.meta else {
        return Err(OptimizeError::Output("parameter region is not an entry".into()));
    };
    let (_, parameters) = extract_lambda_params_ref(&definition.body);
    let Some((symbol, ty)) = parameters.get(index as usize) else {
        return Err(OptimizeError::Output("entry parameter missing".into()));
    };
    let Some(param) = entry.declaration.params.get(index as usize) else {
        return Err(OptimizeError::Output("missing parameter declaration".into()));
    };
    let binding = entry.data.param_bindings.get(index as usize).and_then(Option::as_ref);
    let abi = required(&program.graph, "ParameterAbi", (scope, index))?;
    let access = required(&program.graph, "ParameterStorageAccess", (scope, index))?;
    let access = match program.graph.value_to_base::<i64>(access) {
        1 => StorageAccess::ReadOnly,
        2 => StorageAccess::WriteOnly,
        3 => StorageAccess::ReadWrite,
        _ => return Err(OptimizeError::Output("invalid parameter storage access".into())),
    };
    let mut inputs = Vec::new();
    if let Some(EntryParamBindingKind::TupleOfViews(fields)) = binding.map(|b| &b.kind) {
        let Type::Constructed(TypeName::Tuple(_), types) = strip_existentials(ty) else {
            return Err(OptimizeError::Output("tuple input has no tuple type".into()));
        };
        if fields.len() != types.len() {
            return Err(OptimizeError::Output("tuple input binding arity mismatch".into()));
        }
        for (i, (field, ty)) in fields.iter().zip(types).enumerate() {
            inputs.push(EntryInput {
                name: format!("{}_{}", param.name, i),
                ty: canonical_storage_buffer_ty(ty),
                size_hint: extract_size_hint(param),
                kind: EntryInputKind::Storage {
                    exposure: BindingExposure::Host(field.binding),
                    access,
                    length: None,
                },
            });
        }
    } else {
        let inferred = binding.map(|b| b.first_buffer().0);
        let declared = extract_storage_binding(param);
        if let (Some(inferred), Some(declared)) = (inferred, declared) {
            if inferred != declared {
                return Err(OptimizeError::Output(
                    "parameter storage bindings disagree".into(),
                ));
            }
        }
        let storage = inferred.or(declared);
        let kind =
            if let Some(binding) = storage {
                EntryInputKind::Storage {
                    exposure: BindingExposure::Host(binding),
                    access,
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
                    (None, None) => TextureSource::External,
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
            } else if fields(&program.graph, "ShaderInput", abi)?.is_some() {
                EntryInputKind::Value {
                    decoration: extract_io_decoration(param),
                }
            } else if let Some(fields) = fields(&program.graph, "PushInput", abi)? {
                EntryInputKind::PushConstant {
                    slot: PushConstantSlot {
                        offset: program.graph.value_to_base::<i64>(fields[0]) as u32,
                        size: program.graph.value_to_base::<i64>(fields[1]) as u32,
                    },
                }
            } else {
                return Err(OptimizeError::Output(
                    "selected input requires a declared binding".into(),
                ));
            };
        inputs.push(EntryInput {
            name: param.name.clone(),
            ty: storage_value_type(&canonical_storage_buffer_ty(ty)),
            size_hint: extract_size_hint(param),
            kind,
        });
    }
    inputs
        .into_iter()
        .map(|mut declaration| {
            let scalar_storage = declaration.storage_binding().is_some() && !declaration.ty.is_array();
            if scalar_storage {
                declaration.ty = types::sized_array(1, declaration.ty.clone());
            }
            let physical = if let Some(binding) = declaration.storage_binding() {
                let Some(element) = declaration.ty.elem_type() else {
                    return Err(OptimizeError::Output("storage input has no element".into()));
                };
                facts.physical_type(element, true).map(|element| {
                    types::view_array_with_size(
                        &element,
                        Type::Constructed(TypeName::SizePlaceholder, vec![]),
                        types::buffer_tag(binding),
                    )
                })
            } else {
                facts.physical_type(&declaration.ty, false)
            };
            let parameter_type = physical?;
            Ok(Input {
                declaration,
                parameter_type,
                scalar_storage,
            })
        })
        .collect()
}

pub(super) fn outputs(
    program: &Program<'_, super::Optimized>,
    symbol: SymbolId,
) -> Result<Vec<crate::interface::EntryOutput>, OptimizeError> {
    let facts = super::to_ssa::read::Facts { program };
    let Some(token) = program.identities.symbols.get(&symbol) else {
        return Err(OptimizeError::Output("entry identity missing".into()));
    };
    if !facts.contains("EmitOriginalEntry", (token,)) {
        return Ok(Vec::new());
    }
    let Some(definition) = program.source.defs.iter().find(|d| d.name == symbol) else {
        return Err(OptimizeError::Output("entry declaration missing".into()));
    };
    let DefMeta::EntryPoint(entry) = &definition.meta else {
        return Err(OptimizeError::Output("entry metadata missing".into()));
    };
    let Some(scope) = facts.definition(symbol) else {
        return Err(OptimizeError::Output("entry scope missing".into()));
    };
    let (body, parameters) = extract_lambda_params_ref(&definition.body);
    let mut bindings = IdSource::new();
    for index in 0..parameters.len() {
        for input in parameter_inputs(program, scope, index as i64)? {
            if let Some(binding) = input.declaration.descriptor_binding().filter(|b| b.set == 0) {
                while bindings.peek_id() <= binding.binding {
                    bindings.next_id();
                }
            }
        }
    }
    let mut capacities = Vec::new();
    if entry.declaration.entry_kind == EntryKind::Compute {
        let mut outputs = Vec::new();
        Query(&program.graph).for_each("SourceOutput", |row| {
            if facts.integer(row[1]) == token {
                outputs.push(facts.integer(row[0]));
            }
            Ok(())
        })?;
        outputs.sort_unstable();
        for output in outputs {
            let capacity = required(&program.graph, "SelectedOutputCapacity", (output,))?;
            capacities.push(sizes::output_capacity(&facts, capacity)?);
        }
    }
    build_entry_outputs(
        &entry.declaration,
        &storage_value_type(&body.ty),
        &capacities,
        entry.declaration.entry_kind == EntryKind::Compute,
        &mut bindings,
    )
    .map_err(|error| OptimizeError::Output(error.to_string()))
}
