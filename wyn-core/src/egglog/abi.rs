//! Final shader interfaces shared by SSA emission and publication.
use super::data::{BlockId, BufferId, EntryId, ParameterId};
use super::data::{
    DefinitionData, DefinitionId, EntryData, EntryParamData, EntryParamId, InputBoundData, InputBoundId,
    ParameterData, RegionData, RegionId, SymbolData, SymbolId, TypeData, TypeId,
};
use super::OptimizeError;
use crate::binding_layout::{
    extract_io_decoration, extract_sampler_binding, extract_storage_access, extract_storage_binding,
    extract_storage_image_binding, extract_storage_image_resource, extract_texture_backing,
    extract_texture_binding, extract_texture_resource, extract_uniform_binding,
};
use crate::interface::lowering::extract_size_hint;
use crate::interface::{
    BindingExposure, EntryInput, EntryInputKind, EntryKind, EntryParamBinding, EntryParamBindingKind,
    IoDecoration, PushConstantSlot, StorageAccess, StorageBindingDecl, TextureSource,
};
use crate::pipeline_descriptor::{BufferLen, DispatchSize};
use crate::ssa::layout::type_byte_size;
use crate::types::{
    bool_type, canonical_storage_buffer_ty, sized_array, strip_existentials, Diet, Type, TypeExt, TypeName,
};
use crate::BindingRef;
use std::collections::BTreeMap;
use wyn_base::IdArena;

#[derive(Clone, Debug, Default)]
pub(super) struct Abi {
    pub root_accesses: BTreeMap<BlockId, BTreeMap<BindingRef, crate::ResourceAccess>>,
    pub entry_roots: BTreeMap<EntryId, Vec<BlockId>>,
    pub inputs: BTreeMap<ParameterId, Vec<EntryInput>>,
    pub bindings: BTreeMap<BufferId, StorageBindingDecl>,
    pub buffer_bindings: BTreeMap<BufferId, BindingRef>,
    pub local_lengths: BTreeMap<BufferId, u64>,
    pub roots: Vec<(BlockId, EntryId, [u32; 3], bool)>,
    pub dispatch_sizes: BTreeMap<BlockId, DispatchSize>,
}

pub(super) fn error(message: impl Into<String>) -> OptimizeError {
    OptimizeError::Output(format!("shader interface: {}", message.into()))
}

pub(super) fn inputs(
    entries: &IdArena<EntryId, EntryData>,
    entry_params: &IdArena<EntryParamId, EntryParamData>,
    input_bounds: &IdArena<InputBoundId, InputBoundData>,
    symbols: &IdArena<SymbolId, SymbolData>,
    regions: &IdArena<RegionId, RegionData>,
    definitions: &IdArena<DefinitionId, DefinitionData>,
    types: &IdArena<TypeId, TypeData>,
    parameters: &IdArena<ParameterId, ParameterData>,
) -> Result<BTreeMap<ParameterId, Vec<EntryInput>>, OptimizeError> {
    let bindings: BTreeMap<_, _> =
        entry_params.values().map(|p| ((p.entry, p.position), p.binding.as_ref())).collect();
    let bounds: BTreeMap<_, _> =
        input_bounds.values().map(|b| ((b.entry, symbols[b.symbol].source.0), b.length.clone())).collect();
    let mut result = BTreeMap::new();
    for (&id, entry) in entries {
        let mut pc_offset = 0;
        let params = &regions[definitions[entry.definition].body].parameters;
        for (i, &param) in params.iter().enumerate() {
            let Some(source) = entry.declaration.params.get(i) else {
                return Err(error("entry parameter metadata missing"));
            };
            let ty = &types[parameters[param].ty].ty;
            let layout = bindings.get(&(id, i)).copied().flatten();
            let access = extract_storage_access(source).unwrap_or_else(|| {
                if entry.declaration.param_diets.get(i).is_some_and(Diet::is_consuming) {
                    StorageAccess::ReadWrite
                } else {
                    StorageAccess::ReadOnly
                }
            });
            if let Some(EntryParamBinding {
                kind: EntryParamBindingKind::TupleOfViews(fields),
                ..
            }) = layout
            {
                let Type::Constructed(TypeName::Tuple(_), tys) = ty else {
                    return Err(error("tuple input layout"));
                };
                result.insert(
                    param,
                    fields
                        .iter()
                        .zip(tys)
                        .enumerate()
                        .map(|(i, (f, ty))| EntryInput {
                            name: format!("{}_{}", source.name, i),
                            ty: canonical_storage_buffer_ty(ty),
                            size_hint: None,
                            kind: EntryInputKind::Storage {
                                exposure: BindingExposure::Host(f.binding),
                                access,
                                length: None,
                            },
                        })
                        .collect(),
                );
                continue;
            }
            let storage = layout.map(|p| p.first_buffer().0).or_else(|| extract_storage_binding(source));
            let decoration = extract_io_decoration(source);
            let kind = if let Some(binding) = storage {
                let length = layout
                    .and_then(|p| bounds.get(&(id, p.param_sym.0)))
                    .cloned()
                    .or_else(|| type_byte_size(ty).map(|bytes| BufferLen::Fixed { bytes: bytes.into() }));
                EntryInputKind::Storage {
                    exposure: BindingExposure::Host(binding),
                    access,
                    length,
                }
            } else if let Some(binding) = extract_uniform_binding(source) {
                EntryInputKind::Uniform { binding }
            } else if let Some(binding) = extract_texture_binding(source) {
                let backing = extract_texture_backing(source);
                let source = match (backing, extract_texture_resource(source)) {
                    (backing, Some(name)) => TextureSource::Resource { name, backing },
                    (Some(b), None) => TextureSource::Backing(b),
                    (None, None) => TextureSource::External,
                };
                EntryInputKind::Texture { binding, source }
            } else if let Some(binding) = extract_sampler_binding(source) {
                EntryInputKind::Sampler { binding }
            } else if let Some((binding, format, access, size)) = extract_storage_image_binding(source) {
                EntryInputKind::StorageImage {
                    binding,
                    format,
                    access,
                    size,
                    resource: extract_storage_image_resource(source),
                }
            } else if entry.declaration.entry_kind != EntryKind::Compute
                || matches!(decoration, Some(IoDecoration::BuiltIn(_)))
            {
                EntryInputKind::Value { decoration }
            } else {
                let Some(size) = type_byte_size(&storage_type(ty)?) else {
                    return Err(error(format!(
                        "entry parameter {} has no byte layout",
                        source.name
                    )));
                };
                let slot = PushConstantSlot {
                    offset: pc_offset,
                    size,
                };
                let Some(end) = pc_offset.checked_add(size) else {
                    return Err(error("parameter layout overflow"));
                };
                pc_offset = end;
                EntryInputKind::PushConstant { slot }
            };
            result.insert(
                param,
                vec![EntryInput {
                    name: source.name.clone(),
                    ty: if *ty == bool_type() { u32_type() } else { canonical_storage_buffer_ty(ty) },
                    size_hint: extract_size_hint(source),
                    kind,
                }],
            );
        }
    }
    Ok(result)
}

pub(super) fn u32_type() -> Type {
    Type::Constructed(TypeName::UInt(32), vec![])
}

pub(super) fn concrete(ty: &Type) -> Result<Type, OptimizeError> {
    let ty = strip_existentials(ty);
    if let Some(element) = ty.elem_type().filter(|_| ty.is_array()) {
        let count = match ty.array_size() {
            Some(Type::Constructed(TypeName::Size(n), _)) => *n as usize,
            _ => return Err(error("runtime-sized array requires a storage view")),
        };
        return Ok(sized_array(count.max(1), concrete(element)?));
    }
    match ty {
        Type::Constructed(name, args) => Ok(Type::Constructed(
            name.clone(),
            args.iter().map(concrete).collect::<Result<_, _>>()?,
        )),
        Type::Variable(_) => Err(error("unresolved scalar type")),
    }
}

pub(super) fn storage_type(ty: &Type) -> Result<Type, OptimizeError> {
    if *ty == bool_type() {
        Ok(u32_type())
    } else {
        concrete(ty)
    }
}
