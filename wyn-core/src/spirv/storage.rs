//! `Constructor` methods that turn `#[storage]` / `#[uniform]`
//! bindings into Block-decorated SPIR-V variables with ArrayStride
//! and member-offset decorations, plus the `(set, binding) → buffer_id`
//! map view-indexing uses to recover a buffer var.

use super::*;
use crate::interface;
use crate::pipeline_descriptor;
use crate::ssa;
use crate::types;

impl Constructor {
    /// Get or assign a sequential buffer_id for a (set, binding) pair.
    /// Also registers the buffer_var in buffer_vars for later lookup.
    pub(super) fn get_or_assign_buffer_id(&mut self, set: u32, binding: u32) -> u32 {
        let use_key = self.storage_use(BindingRef::new(set, binding));
        if let Some(&id) = self.buffer_id_map.get(&use_key) {
            return id;
        }
        let id = self.buffer_vars.len() as u32;
        let buffer =
            self.storage_buffers.get(&use_key).expect("get_or_assign_buffer_id: storage buffer must exist");
        self.buffer_vars.push((buffer.variable, buffer.element_type));
        self.buffer_id_map.insert(use_key, id);
        id
    }

    /// Apply ArrayStride decorations for all nested fixed-size arrays in a type
    /// used inside a storage buffer. Uses layout::buffer_array_strides() for the
    /// stride values and walks nested arrays via the builder's
    /// array-element registry for SPIR-V IDs.
    /// Skips types that have already been decorated.
    pub(super) fn apply_buffer_array_strides(
        &mut self,
        spirv_type: spirv::Word,
        poly_type: &PolyType<TypeName>,
    ) {
        let strides = buffer_array_strides(poly_type);
        if strides.is_empty() {
            return;
        }
        let mut current = spirv_type;
        for stride in strides {
            if !self.builder.decorate_array_stride_once(builder::TypeId::new(current), stride) {
                break; // already decorated — nested types are too
            }
            if let Some(inner) = self.builder.array_element_type(builder::TypeId::new(current)) {
                current = *inner;
            } else {
                break;
            }
        }
    }

    /// Create a decorated interface block struct type.
    /// Atomically creates the OpTypeStruct AND all required decorations.
    /// Cached by kind + layout so identical blocks share one ID, but
    /// never share with plain tuple structs.
    pub(super) fn create_interface_block_type(
        &mut self,
        kind: InterfaceBlockKind,
        member_types: &[spirv::Word],
        member_offsets: &[u32],
        member_poly_types: &[&PolyType<TypeName>],
    ) -> spirv::Word {
        let key = InterfaceBlockKey {
            kind,
            members: member_types.iter().zip(member_offsets.iter()).map(|(&t, &o)| (t, o)).collect(),
        };
        if let Some(&ty) = self.interface_block_cache.get(&key) {
            return ty;
        }

        // Create a fresh struct — do NOT go through get_or_create_struct_type
        // to avoid sharing IDs with plain tuple structs.
        let ty = *self.builder.type_struct(member_types.iter().map(|&w| builder::TypeId::new(w)).collect());

        // Decorate as Block + member offsets (once per struct id).
        self.builder.decorate_block_once(builder::TypeId::new(ty), member_offsets);

        // Apply ArrayStride for array members
        for (i, poly_ty) in member_poly_types.iter().enumerate() {
            self.apply_buffer_array_strides(member_types[i], poly_ty);
        }

        self.interface_block_cache.insert(key, ty);
        ty
    }

    pub(super) fn get_or_create_buffer_block_type(
        &mut self,
        runtime_array_type: spirv::Word,
        matrix_stride: Option<u32>,
    ) -> spirv::Word {
        *self.builder.buffer_block_type(builder::TypeId::new(runtime_array_type), matrix_stride)
    }

    pub(super) fn get_or_create_uniform_block_type(&mut self, value_type: spirv::Word) -> spirv::Word {
        *self.builder.uniform_block_type(builder::TypeId::new(value_type))
    }

    /// Produce the explicit std430 representation of a logical Wyn value.
    /// Composite interface types must have identities distinct from ordinary
    /// function values because Vulkan restricts their layout decorations to
    /// interface storage classes.
    pub(super) fn storage_polytype_to_spirv(&mut self, ty: &PolyType<TypeName>) -> Result<spirv::Word> {
        if let Some(&cached) = self.storage_polytype_cache.get(ty) {
            return Ok(cached);
        }
        // rspirv structurally interns ordinary aggregate types. Materialize the
        // undecorated logical tree first, then mint explicit-layout composites
        // with forced fresh ids below.
        self.polytype_to_spirv(ty)?;

        let result = match ty {
            PolyType::Constructed(TypeName::Tuple(_) | TypeName::Record(_), members) => {
                let Some(layout) = ssa::layout::block_layout(ty, interface::StorageLayout::Std430) else {
                    return Err(err_spirv!(
                        "storage buffer element {:?} has no supported std430 struct layout",
                        ty
                    ));
                };
                let member_types = members
                    .iter()
                    .map(|member| self.storage_polytype_to_spirv(member))
                    .collect::<Result<Vec<_>>>()?;
                *self.builder.type_buffer_struct(
                    member_types.into_iter().map(builder::TypeId::new).collect(),
                    &layout.member_offsets,
                )
            }
            _ if ty.is_array()
                && ty.array_storage().is_some_and(|storage| {
                    matches!(
                        storage.variant,
                        PolyType::Constructed(TypeName::ArrayVariantComposite, _)
                    )
                }) =>
            {
                let tensor =
                    ty.as_tensor().ok_or_else(|| err_spirv!("malformed storage array type: {:?}", ty))?;
                let mut current = self.storage_polytype_to_spirv(tensor.elem)?;
                let mut stride = ssa::layout::storage_elem_stride(tensor.elem).ok_or_else(|| {
                    err_spirv!(
                        "storage array element has no known std430 size: {:?}",
                        tensor.elem
                    )
                })?;
                for dim in tensor.dims.iter().rev() {
                    let PolyType::Constructed(TypeName::Size(count), _) = dim else {
                        return Err(err_spirv!("storage array dimension is not concrete: {:?}", dim));
                    };
                    let count = u32::try_from(*count)
                        .map_err(|_| err_spirv!("storage array dimension exceeds SPIR-V limits"))?;
                    let count_id = self.const_u32(count);
                    current =
                        *self.builder.type_buffer_array(builder::TypeId::new(current), count_id, stride);
                    stride = stride
                        .checked_mul(count)
                        .ok_or_else(|| err_spirv!("storage array byte size exceeds SPIR-V limits"))?;
                }
                current
            }
            _ => self.polytype_to_spirv(ty)?,
        };

        self.storage_polytype_cache.insert(ty.clone(), result);
        Ok(result)
    }

    /// Convert between an ordinary function value and its explicit std430
    /// representation. When `add_decorations` is true, the result uses the
    /// decorated storage type; otherwise the result uses the undecorated
    /// logical type.
    pub(super) fn convert_storage_value(
        &mut self,
        value: spirv::Word,
        ty: &PolyType<TypeName>,
        add_decorations: bool,
    ) -> Result<spirv::Word> {
        let logical_ty = self.polytype_to_spirv(ty)?;
        let storage_ty = self.storage_polytype_to_spirv(ty)?;
        if logical_ty == storage_ty {
            return Ok(value);
        }
        let (source_ty, destination_ty) =
            if add_decorations { (logical_ty, storage_ty) } else { (storage_ty, logical_ty) };
        match ty {
            PolyType::Constructed(TypeName::Tuple(_) | TypeName::Record(_), members) => {
                let mut converted_members = Vec::with_capacity(members.len());
                for (index, member) in members.iter().enumerate() {
                    let source_member_ty = if add_decorations {
                        self.polytype_to_spirv(member)?
                    } else {
                        self.storage_polytype_to_spirv(member)?
                    };
                    let source_member =
                        self.builder.composite_extract(source_member_ty, None, value, [index as u32])?;
                    converted_members.push(self.convert_storage_value(
                        source_member,
                        member,
                        add_decorations,
                    )?);
                }
                Ok(self.builder.composite_construct(destination_ty, None, converted_members)?)
            }
            _ if ty.is_array() => {
                let (count, child_type) = {
                    let tensor = ty
                        .as_tensor()
                        .ok_or_else(|| err_spirv!("malformed storage array type: {:?}", ty))?;
                    let Some((dim, remaining)) = tensor.dims.split_first() else {
                        return Err(err_spirv!("storage array has no dimensions: {:?}", ty));
                    };
                    let PolyType::Constructed(TypeName::Size(count), _) = dim else {
                        return Err(err_spirv!("storage array dimension is not concrete: {:?}", dim));
                    };
                    let child_type = if remaining.is_empty() {
                        tensor.elem.clone()
                    } else {
                        let storage = ty
                            .array_storage()
                            .ok_or_else(|| err_spirv!("storage array metadata is missing: {:?}", ty))?;
                        let mut args = Vec::with_capacity(remaining.len() + 3);
                        args.push(tensor.elem.clone());
                        args.push(storage.variant.clone());
                        args.extend(remaining.iter().cloned());
                        args.push(storage.region.clone());
                        PolyType::Constructed(TypeName::Array, args)
                    };
                    (*count, child_type)
                };

                let source_child_ty = self.get_array_element_type(source_ty)?;
                let mut converted_elements = Vec::with_capacity(count);
                for index in 0..count {
                    let source_element =
                        self.builder.composite_extract(source_child_ty, None, value, [index as u32])?;
                    converted_elements.push(self.convert_storage_value(
                        source_element,
                        &child_type,
                        add_decorations,
                    )?);
                }
                Ok(self.builder.composite_construct(destination_ty, None, converted_elements)?)
            }
            _ => Err(err_spirv!(
                "cannot convert {:?} between logical and storage representations",
                ty
            )),
        }
    }

    /// Create a storage buffer variable for compute shaders.
    /// Returns the variable ID. Also registers it in storage_buffers for later lookup.
    /// Idempotent: returns existing variable if already created for this (set, binding).
    pub(super) fn create_storage_buffer(
        &mut self,
        array_ty: &PolyType<TypeName>,
        set: u32,
        binding: u32,
        writable: bool,
    ) -> Result<spirv::Word> {
        let use_key = StorageBufferUse {
            binding: BindingRef::new(set, binding),
            writable,
        };
        // Return existing if already created
        if let Some(buffer) = self.storage_buffers.get(&use_key) {
            return Ok(buffer.variable);
        }
        // Storage buffers can be either an array-shaped view (`[]T` → elem is
        // `T`) or a scalar / vec / struct output (e.g. a reduce result, which
        // the SOAC pass packs into a single-element `[]T` buffer at the
        // binding level even though the user-visible type is `T`). Use
        // `array_elem` rather than `elem_type` here so a vec-typed buffer
        // stays a vec instead of being unpacked into its component.
        let elem_ty = match types::array_elem(array_ty) {
            Some(elem) => elem.clone(),
            None => array_ty.clone(),
        };
        if types::contains_16_bit_scalar(&elem_ty) {
            self.builder.enable_capability(spirv::Capability::StorageBuffer16BitAccess);
        }
        let elem_spirv = self.storage_polytype_to_spirv(&elem_ty)?;

        // The std430 array stride is the element size rounded up to the
        // element's alignment — a `vec3<T>` is 12 bytes but aligns to 16, so
        // its runtime-array stride must be 16, not the packed 12 (Vulkan
        // rejects a stride not satisfying the element alignment). Struct
        // elements take their aligned size from `block_layout`, which also
        // supplies the member offsets below (a tight `type_byte_size` sum
        // under-strides structs whose members pad).
        let layout = ssa::layout::block_layout(&elem_ty, interface::StorageLayout::Std430);
        let stride = match &layout {
            Some(l) => l.size,
            None => {
                let Some(elem_size) = ssa::layout::storage_elem_stride(&elem_ty) else {
                    return Err(err_spirv!(
                        "storage buffer element type has no known std430 size: {:?}",
                        elem_ty
                    ));
                };
                let elem_align = std430_alignment(&elem_ty).unwrap_or(elem_size.max(1));
                elem_size.div_ceil(elem_align) * elem_align
            }
        };

        // Create runtime array type (cached to avoid duplicate decorations)
        let runtime_array = self.get_or_create_runtime_array_type(elem_spirv, stride);

        // Create block struct (cached)
        let matrix_stride = ssa::layout::std430_matrix_stride(&elem_ty);
        let block_struct = self.get_or_create_buffer_block_type(runtime_array, matrix_stride);

        let ptr_type = self.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, block_struct);
        let var_id = self.builder.variable(ptr_type, None, spirv::StorageClass::StorageBuffer, None);

        self.builder.decorate(
            var_id,
            spirv::Decoration::DescriptorSet,
            [Operand::LiteralBit32(set)],
        );
        self.builder.decorate(
            var_id,
            spirv::Decoration::Binding,
            [Operand::LiteralBit32(binding)],
        );
        if !writable {
            self.builder.decorate_nonwritable_once(builder::VarId::new(var_id));
        }

        // Store for later lookup (ptr_type used for StorageView struct construction)
        self.storage_buffers.insert(
            use_key,
            StorageBufferInfo {
                variable: var_id,
                element_type: elem_spirv,
            },
        );

        Ok(var_id)
    }

    /// Create (once) the `#[storage_image]` global for `br`: a format-aware
    /// `OpTypeImage` in `UniformConstant` storage, decorated with its
    /// descriptor set/binding and the source access qualifier. Registered in
    /// `storage_images` so both the entry (interface / `env`) and
    /// storage-image updates / `image_load` inside functions resolve the same
    /// module-scope variable. Idempotent — a binding shared across entries
    /// returns the existing var.
    pub(super) fn create_storage_image(
        &mut self,
        br: BindingRef,
        format: pipeline_descriptor::StorageImageFormat,
        access: interface::StorageAccess,
    ) -> spirv::Word {
        if let Some(&(var_id, _)) = self.storage_images.get(&br) {
            return var_id;
        }
        let img_type = *self.builder.type_image(
            builder::TypeId::new(self.f32_type),
            spirv::Dim::Dim2D,
            0,
            0,
            0,
            2,
            storage_image_format_to_spirv(format),
            None,
        );
        let ptr_type = self.get_or_create_ptr_type(spirv::StorageClass::UniformConstant, img_type);
        let var_id = self.builder.variable(ptr_type, None, spirv::StorageClass::UniformConstant, None);
        self.builder.decorate(
            var_id,
            spirv::Decoration::DescriptorSet,
            [Operand::LiteralBit32(br.set)],
        );
        self.builder.decorate(
            var_id,
            spirv::Decoration::Binding,
            [Operand::LiteralBit32(br.binding)],
        );
        // Encode the access qualifier as `NonReadable` / `NonWritable` so
        // naga/wgpu doesn't infer `ReadWrite` and reject a narrower host
        // descriptor.
        use crate::interface::StorageAccess;
        match access {
            StorageAccess::WriteOnly => self.builder.decorate(
                var_id,
                spirv::Decoration::NonReadable,
                std::iter::empty::<Operand>(),
            ),
            StorageAccess::ReadOnly => self.builder.decorate(
                var_id,
                spirv::Decoration::NonWritable,
                std::iter::empty::<Operand>(),
            ),
            StorageAccess::ReadWrite => {}
        }
        self.storage_images.insert(br, (var_id, img_type));
        var_id
    }
}
