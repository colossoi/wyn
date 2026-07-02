//! `Constructor` methods that turn `#[storage]` / `#[uniform]`
//! bindings into Block-decorated SPIR-V variables with ArrayStride
//! and member-offset decorations, plus the `(set, binding) → buffer_id`
//! map view-indexing uses to recover a buffer var.

use super::*;

impl Constructor {
    /// Get or assign a sequential buffer_id for a (set, binding) pair.
    /// Also registers the buffer_var in buffer_vars for later lookup.
    pub(super) fn get_or_assign_buffer_id(&mut self, set: u32, binding: u32) -> u32 {
        if let Some(&id) = self.buffer_id_map.get(&BindingRef::new(set, binding)) {
            return id;
        }
        let id = self.buffer_vars.len() as u32;
        let &(buffer_var, elem_ty, _) = self
            .storage_buffers
            .get(&BindingRef::new(set, binding))
            .expect("get_or_assign_buffer_id: storage buffer must exist");
        self.buffer_vars.push((buffer_var, elem_ty));
        self.buffer_id_map.insert(BindingRef::new(set, binding), id);
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
    ) -> spirv::Word {
        *self.builder.buffer_block_type(builder::TypeId::new(runtime_array_type))
    }

    pub(super) fn get_or_create_uniform_block_type(&mut self, value_type: spirv::Word) -> spirv::Word {
        *self.builder.uniform_block_type(builder::TypeId::new(value_type))
    }

    /// Create a storage buffer variable for compute shaders.
    /// Returns the variable ID. Also registers it in storage_buffers for later lookup.
    /// Idempotent: returns existing variable if already created for this (set, binding).
    pub(super) fn create_storage_buffer(
        &mut self,
        array_ty: &PolyType<TypeName>,
        set: u32,
        binding: u32,
    ) -> spirv::Word {
        // Return existing if already created
        if let Some(&(var_id, _, _)) = self.storage_buffers.get(&BindingRef::new(set, binding)) {
            return var_id;
        }
        // Storage buffers can be either an array-shaped view (`[]T` → elem is
        // `T`) or a scalar / vec / struct output (e.g. a reduce result, which
        // the SOAC pass packs into a single-element `[]T` buffer at the
        // binding level even though the user-visible type is `T`). Use
        // `array_elem` rather than `elem_type` here so a vec-typed buffer
        // stays a vec instead of being unpacked into its component.
        let elem_ty = match crate::types::array_elem(array_ty) {
            Some(elem) => elem.clone(),
            None => array_ty.clone(),
        };
        let elem_spirv = self.polytype_to_spirv(&elem_ty);

        // The std430 array stride is the element size rounded up to the
        // element's alignment — a `vec3<T>` is 12 bytes but aligns to 16, so
        // its runtime-array stride must be 16, not the packed 12 (Vulkan
        // rejects a stride not satisfying the element alignment). Struct
        // elements take their aligned size from `block_layout`, which also
        // supplies the member offsets below (a tight `type_byte_size` sum
        // under-strides structs whose members pad).
        let layout = crate::ssa::layout::block_layout(&elem_ty, crate::ssa::layout::LayoutRules::Std430);
        let stride = match &layout {
            Some(l) => l.size,
            None => {
                let elem_size =
                    type_byte_size(&elem_ty).expect("storage buffer element type must have known size");
                let elem_align = std430_alignment(&elem_ty).unwrap_or(elem_size.max(1));
                elem_size.div_ceil(elem_align) * elem_align
            }
        };

        // Ensure nested array types have ArrayStride for buffer layout
        self.apply_buffer_array_strides(elem_spirv, &elem_ty);

        // If the element type is a tuple/record/struct, add std430 member
        // offset decorations for the buffer layout. We add them to the elem
        // type directly since it will be used inside a runtime array in a
        // storage buffer.
        let is_struct = matches!(
            &elem_ty,
            PolyType::Constructed(TypeName::Tuple(_), _) | PolyType::Constructed(TypeName::Record(_), _)
        );
        if is_struct && self.builder.mark_buffer_layout_decorated_once(builder::TypeId::new(elem_spirv)) {
            let layout = layout.as_ref().unwrap_or_else(|| {
                panic!(
                    "storage buffer element {:?}: struct members must be 32-bit \
                     scalars or vectors of them (std430)",
                    elem_ty
                )
            });
            for (i, &offset) in layout.member_offsets.iter().enumerate() {
                self.builder.member_decorate(
                    elem_spirv,
                    i as u32,
                    spirv::Decoration::Offset,
                    [Operand::LiteralBit32(offset)],
                );
            }
        }

        // Create runtime array type (cached to avoid duplicate decorations)
        let runtime_array = self.get_or_create_runtime_array_type(elem_spirv, stride);

        // Create block struct (cached)
        let block_struct = self.get_or_create_buffer_block_type(runtime_array);

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

        // Store for later lookup (ptr_type used for StorageView struct construction)
        self.storage_buffers.insert(BindingRef::new(set, binding), (var_id, block_struct, ptr_type));

        var_id
    }

    /// Create (once) the `#[storage_image]` global for `br`: a format-aware
    /// `OpTypeImage` in `UniformConstant` storage, decorated with its
    /// descriptor set/binding and the source access qualifier. Registered in
    /// `storage_images` so both the entry (interface / `env`) and
    /// `image_store`/`image_load` inside functions resolve the same
    /// module-scope variable. Idempotent — a binding shared across entries
    /// returns the existing var.
    pub(super) fn create_storage_image(
        &mut self,
        br: BindingRef,
        format: crate::pipeline_descriptor::StorageImageFormat,
        access: crate::interface::StorageAccess,
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
