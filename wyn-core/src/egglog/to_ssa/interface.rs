//! Translate retained source interfaces and planned resources into the shared
//! shader/runtime ABI. This module does not decide residency or stage ordering.

use super::*;
use crate::binding_layout::*;
use crate::egglog::ExprKind;
use crate::interface::{
    self, BindingExposure, EntryInput, EntryInputKind, EntryParamBindingKind, IoDecoration,
    PushConstantSlot, StorageAccess, TextureSource,
};
use crate::pipeline_descriptor::{BufferLen, DispatchSize};

pub(super) fn inputs(
    data: &AssociatedData,
) -> Result<BTreeMap<ParameterId, Vec<EntryInput>>, OptimizeError> {
    let bindings: BTreeMap<_, _> =
        data.entry_params.values().map(|p| ((p.entry, p.position), p.binding.as_ref())).collect();
    let bounds: BTreeMap<_, _> = data
        .input_bounds
        .values()
        .map(|b| ((b.entry, data.symbols[b.symbol].source.0), b.length.clone()))
        .collect();
    let mut result = BTreeMap::new();
    for (&id, entry) in &data.entries {
        let mut pc_offset = 0;
        let params = &data.regions[data.definitions[entry.definition].body].parameters;
        for (i, &param) in params.iter().enumerate() {
            let source =
                entry.declaration.params.get(i).ok_or_else(|| error("entry parameter metadata missing"))?;
            let ty = &data.types[data.parameters[param].ty].ty;
            let layout = bindings.get(&(id, i)).copied().flatten();
            let access = extract_storage_access(source).unwrap_or_else(|| {
                if entry.declaration.param_diets.get(i).is_some_and(types::Diet::is_consuming) {
                    StorageAccess::ReadWrite
                } else {
                    StorageAccess::ReadOnly
                }
            });
            if let Some(interface::EntryParamBinding {
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
                            ty: types::canonical_storage_buffer_ty(ty),
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
                let length = layout.and_then(|p| bounds.get(&(id, p.param_sym.0))).cloned().or_else(|| {
                    ssa::layout::type_byte_size(ty).map(|bytes| BufferLen::Fixed { bytes: bytes.into() })
                });
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
            } else if matches!(decoration, Some(IoDecoration::BuiltIn(_))) {
                EntryInputKind::Value { decoration }
            } else {
                let size = ssa::layout::type_byte_size(&storage_type(ty)?)
                    .ok_or_else(|| error(format!("entry parameter {} has no byte layout", source.name)))?;
                let slot = PushConstantSlot {
                    offset: pc_offset,
                    size,
                };
                pc_offset =
                    pc_offset.checked_add(size).ok_or_else(|| error("parameter layout overflow"))?;
                EntryInputKind::PushConstant { slot }
            };
            result.insert(
                param,
                vec![EntryInput {
                    name: source.name.clone(),
                    ty: if *ty == types::bool_type() {
                        u32_type()
                    } else {
                        types::canonical_storage_buffer_ty(ty)
                    },
                    size_hint: crate::egir::from_tlc::extract_size_hint(source),
                    kind,
                }],
            );
        }
    }
    Ok(result)
}

impl Compiler<'_> {
    pub(super) fn allocate_bindings(&mut self) -> Result<(), OptimizeError> {
        let reserved: BTreeSet<_> =
            self.inputs.values().flatten().filter_map(|i| i.descriptor_binding()).collect();
        let mut next = self.data.programs.values().map(|p| p.next_auto_storage_binding).max().unwrap_or(0);
        // Bindings are deterministic and assigned only to Allocation facts.
        for (&id, b) in &self.data.buffers {
            if b.storage != Storage::Device {
                continue;
            }
            while reserved.contains(&BindingRef::new(0, next)) {
                next += 1;
            }
            let binding = BindingRef::new(0, next);
            next = next.checked_add(1).ok_or_else(|| error("too many storage bindings"))?;
            self.bindings.insert(
                id,
                StorageBindingDecl {
                    binding,
                    elem_ty: storage_type(&b.element)?,
                    role: StorageRole::Intermediate,
                    logical_resource: Some(format!("egg_resource{}", id.as_u32())),
                    length: None,
                },
            );
        }
        let lengths = self
            .bindings
            .iter()
            .map(|(&id, b)| {
                let bytes = ssa::layout::type_byte_size(&b.elem_ty)
                    .ok_or_else(|| error("buffer element has no byte layout"))?;
                Ok((id, self.capacity(&self.data.buffers[id].length, bytes)))
            })
            .collect::<Result<Vec<_>, OptimizeError>>()?;
        for (id, length) in lengths {
            self.bindings.get_mut(&id).unwrap().length = Some(length);
        }
        Ok(())
    }

    pub(super) fn capacity(&self, length: &Value, elem_bytes: u32) -> BufferLen {
        if let Some(n) = self.constant(length) {
            return BufferLen::Fixed {
                bytes: n * u64::from(elem_bytes),
            };
        }
        if let Value::Primitive("length", a) = length {
            if let Some((binding, src_elem_bytes)) = self.value_binding(&a[0]) {
                return BufferLen::LikeInput {
                    set: binding.set,
                    binding: binding.binding,
                    elem_bytes,
                    src_elem_bytes,
                };
            }
        }
        // TODO: extend the shared runtime ABI with checked capacity formulas
        // (notably max(1, ceil(n/64)) scratch). Never substitute a made-up length.
        BufferLen::HostProvided {
            inputs: vec![],
            elem_bytes,
        }
    }

    pub(super) fn value_binding(&self, value: &Value) -> Option<(BindingRef, u32)> {
        match value {
            Value::Buffer(id) => match self.data.buffers[*id].storage {
                Storage::External(e) => self.value_binding(&Value::Source(e)),
                _ => {
                    let b = self.bindings.get(id)?;
                    Some((b.binding, ssa::layout::type_byte_size(&b.elem_ty)?))
                }
            },
            Value::Array(crate::egglog::Array::Value(e)) => self.value_binding(&Value::Source(*e)),
            Value::Source(e) => match &self.data.expressions[*e].kind {
                ExprKind::Parameter(p) => {
                    let (binding, ty) = self
                        .inputs
                        .get(p)?
                        .first()
                        .and_then(|i| Some((i.storage_binding()?, i.ty.elem_type()?)))?;
                    Some((binding, ssa::layout::type_byte_size(ty)?))
                }
                ExprKind::Coerce(e) | ExprKind::Array(crate::egglog::Array::Value(e)) => {
                    self.value_binding(&Value::Source(*e))
                }
                ExprKind::OperationResult(op) => self.value_binding(self.results.get(op)?),
                ExprKind::Project { tuple, index } => {
                    if let ExprKind::Parameter(p) = self.data.expressions[*tuple].kind {
                        let input = self.inputs.get(&p)?.get(*index)?;
                        return Some((
                            input.storage_binding()?,
                            ssa::layout::type_byte_size(input.ty.elem_type()?)?,
                        ));
                    }
                    if let ExprKind::OperationResult(op) = self.data.expressions[*tuple].kind {
                        if let Value::Tuple(fields) = self.results.get(&op)? {
                            return self.value_binding(fields.get(*index)?);
                        }
                    }
                    if let ExprKind::Tuple(fields) = &self.data.expressions[*tuple].kind {
                        return self.value_binding(&Value::Source(*fields.get(*index)?));
                    }
                    None
                }
                _ => None,
            },
            Value::Primitive("slice", args) => self.value_binding(&args[0]),
            _ => None,
        }
    }

    pub(super) fn constant(&self, value: &Value) -> Option<u64> {
        match value {
            Value::Int(n) => Some((*n).into()),
            Value::Source(e) => match &self.data.expressions[*e].kind {
                ExprKind::Int(n) => n.parse().ok(),
                ExprKind::Coerce(e) => self.constant(&Value::Source(*e)),
                _ => None,
            },
            Value::Primitive("length", args) => match &args[0] {
                Value::Array(crate::egglog::Array::Literal(xs)) => Some(xs.len() as u64),
                Value::Array(crate::egglog::Array::Range { len, .. }) => {
                    self.constant(&Value::Source(*len))
                }
                Value::Source(e) => match self.data.types[self.data.expressions[*e].ty].ty.array_size() {
                    Some(Type::Constructed(TypeName::Size(n), _)) => Some(*n as u64),
                    _ => None,
                },
                _ => None,
            },
            Value::Primitive(op, args) if args.len() == 2 => {
                let a = self.constant(&args[0])?;
                let b = self.constant(&args[1])?;
                match *op {
                    "add" => a.checked_add(b),
                    "mul" => a.checked_mul(b),
                    "sub" => a.checked_sub(b),
                    "ceil_div" if b != 0 => Some(a / b + u64::from(a % b != 0)),
                    "max" => Some(a.max(b)),
                    "min" => Some(a.min(b)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    pub(super) fn dispatch_size(&self, d: &super::super::DispatchData) -> DispatchSize {
        let grid = &self.data.grids[d.grid].groups;
        if let (Some(x), Some(y), Some(z)) = (
            self.constant(&grid[0]),
            self.constant(&grid[1]),
            self.constant(&grid[2]),
        ) {
            return DispatchSize::Fixed {
                x: x as u32,
                y: y as u32,
                z: z as u32,
                explicit: true,
            };
        }
        // Every generated parallel kernel uses a grid-stride loop. A single
        // workgroup is a valid dispatch for domains not expressible in the
        // current descriptor, including GPU-produced filter lengths. This is
        // execution policy only; allocation capacity and logical bounds remain exact.
        // TODO: publish checked multidimensional/capped grid formulas in the ABI.
        DispatchSize::Fixed {
            x: 1,
            y: 1,
            z: 1,
            explicit: true,
        }
    }
}

pub(super) fn storage_type(ty: &Type) -> Result<Type, OptimizeError> {
    if *ty == types::bool_type() {
        Ok(u32_type())
    } else {
        concrete(ty)
    }
}
