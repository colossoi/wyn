//! Resolve source ABI declarations before emitting any shader instructions.
use super::query::Query;
use super::{OptimizeError, Program};
use crate::binding_layout::{
    extract_io_decoration, extract_sampler_binding, extract_storage_binding, extract_storage_image_binding,
    extract_storage_image_resource, extract_texture_backing, extract_texture_binding,
    extract_texture_resource, extract_uniform_binding,
};
use crate::egglog::to_ssa::{plan::unique, Compiler};
use crate::host::BufferLen;
use crate::interface::lowering::{build_entry_outputs, extract_size_hint};
use crate::interface::{
    BindingExposure, EntryInput, EntryInputKind, EntryParamBindingKind, PushConstantSlot, StorageAccess,
    TextureSource,
};
use crate::interface::{EntryKind, StorageBindingDecl, StorageRole};
use crate::ssa::layout::{storage_value_type, type_byte_size};
use crate::tlc::{extract_lambda_params_ref, DefMeta};
use crate::types::{canonical_storage_buffer_ty, strip_existentials, Type, TypeName};
use crate::SymbolId;
use crate::{BindingRef, LookupMap, ResourceAccess};
use egglog_engine::Value;
use std::collections::BTreeSet;
use wyn_base::IdSource;

pub(super) mod publication;
mod sizes;

pub(super) fn entry_accesses(
    compiler: &Compiler<'_, '_>,
    root: Value,
) -> Result<LookupMap<BindingRef, ResourceAccess>, OptimizeError> {
    let mut result = LookupMap::default();
    Query(&compiler.program.graph).for_function("RootBindingAccess", |keys, flags| {
        if keys[0] == root {
            let access = compiler
                .facts
                .storage_access(flags)?
                .ok_or_else(|| OptimizeError::Output("empty selected binding access".into()))?;
            result.insert(compiler.facts.binding(keys[1])?, ResourceAccess::from(access));
        }
        Ok(())
    })?;
    Ok(result)
}

pub(super) fn storage_bindings(
    compiler: &Compiler<'_, '_>,
    root: Value,
    stage: Option<Value>,
    published: &[crate::ssa::types::EntryPoint],
    inputs: &mut [EntryInput],
    accesses: &mut LookupMap<BindingRef, ResourceAccess>,
) -> Result<Vec<StorageBindingDecl>, OptimizeError> {
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
        let flags = Query(&compiler.program.graph).required("RootResource", (root, value))?;
        let access = compiler.facts.storage_access(flags)?;
        if let Some((binding, element, _)) = compiler.plan.buffer(value)? {
            if let Some(access) = access.map(ResourceAccess::from) {
                accesses.entry(binding).and_modify(|old| *old = old.merge(access)).or_insert(access);
            }
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
            for input in inputs.iter_mut() {
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
            accesses.insert(binding, ResourceAccess::Read);
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
    Ok(storage_bindings)
}

pub(super) fn parameter_inputs(
    program: &Program<'_, super::Optimized>,
    scope: Value,
    index: i64,
) -> Result<Vec<EntryInput>, OptimizeError> {
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
    let abi = Query(&program.graph).required("ParameterAbi", (scope, index))?;
    let access = Query(&program.graph).required("ParameterStorageAccess", (scope, index))?;
    let access = facts
        .storage_access(access)?
        .ok_or_else(|| OptimizeError::Output("empty parameter storage access".into()))?;
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
            } else if Query(&program.graph).enode("ShaderInput", abi)?.is_some() {
                EntryInputKind::Value {
                    decoration: extract_io_decoration(param),
                }
            } else if let Some(fields) = Query(&program.graph).enode("PushInput", abi)? {
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
    Ok(inputs)
}

pub(super) fn outputs(
    program: &Program<'_, super::Optimized>,
    symbol: SymbolId,
    bindings: &mut IdSource<u32>,
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
    let (body, _) = extract_lambda_params_ref(&definition.body);
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
            let capacity = Query(&program.graph).required("SelectedOutputCapacity", (output,))?;
            capacities.push(sizes::output_capacity(&facts, capacity)?);
        }
    }
    build_entry_outputs(
        &entry.declaration,
        &storage_value_type(&body.ty),
        &capacities,
        entry.declaration.entry_kind == EntryKind::Compute,
        bindings,
    )
    .map_err(|error| OptimizeError::Output(error.to_string()))
}
