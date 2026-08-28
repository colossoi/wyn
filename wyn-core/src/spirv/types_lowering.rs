//! Wyn `PolyType` → SPIR-V type lowering: `Constructor::polytype_to_spirv`
//! and the structural-type delegators it dispatches to.

use super::*;

impl Constructor {
    /// Resolve a pointer address-space type to a SPIR-V StorageClass.
    pub(super) fn resolve_storage_class(addrspace: &PolyType<TypeName>) -> StorageClass {
        match addrspace {
            PolyType::Constructed(TypeName::PointerFunction, _) => StorageClass::Function,
            PolyType::Constructed(TypeName::PointerInput, _) => StorageClass::Input,
            PolyType::Constructed(TypeName::PointerOutput, _) => StorageClass::Output,
            PolyType::Constructed(TypeName::PointerStorage, _) => StorageClass::StorageBuffer,
            _ => StorageClass::Function,
        }
    }

    /// Get or create a pointer type
    pub(super) fn get_or_create_ptr_type(
        &mut self,
        storage_class: spirv::StorageClass,
        pointee_id: spirv::Word,
    ) -> spirv::Word {
        *self.builder.type_pointer(storage_class, builder::TypeId::new(pointee_id))
    }

    /// Convert a polytype Type to a SPIR-V type ID
    pub(super) fn polytype_to_spirv(&mut self, ty: &PolyType<TypeName>) -> Result<spirv::Word> {
        if let Some(&cached) = self.polytype_cache.get(ty) {
            return Ok(cached);
        }
        let result = self.polytype_to_spirv_uncached(ty)?;
        self.polytype_cache.insert(ty.clone(), result);
        Ok(result)
    }

    pub(super) fn polytype_to_spirv_uncached(&mut self, ty: &PolyType<TypeName>) -> Result<spirv::Word> {
        let result = match ty {
            PolyType::Variable(id) => {
                return Err(err_spirv!(
                    "unresolved type variable Variable({}) reached SPIR-V lowering",
                    id
                ));
            }
            PolyType::Constructed(name, args) => {
                // Assert that no UserVar or SizeVar reaches lowering
                match name {
                    TypeName::UserVar(v) => {
                        return Err(err_spirv!("user type variable '{}' reached SPIR-V lowering", v));
                    }
                    TypeName::SizeVar(v) => {
                        return Err(err_spirv!("size variable '{}' reached SPIR-V lowering", v));
                    }
                    _ => {}
                }

                match name {
                    TypeName::Int(32) => self.i32_type,
                    TypeName::Float(32) => self.f32_type,
                    TypeName::Int(bits) => *self.builder.type_int(*bits as u32, 1),
                    TypeName::UInt(bits) => *self.builder.type_int(*bits as u32, 0),
                    TypeName::Float(bits) => *self.builder.type_float(*bits as u32),
                    TypeName::Bool => self.bool_type,
                    TypeName::Unit => {
                        // Unit type - use void type
                        // Unit values are never actually constructed since they can only be assigned to _
                        self.void_type
                    }
                    TypeName::Tuple(_) => {
                        // Empty tuples should not reach lowering:
                        // - Unit values are bound to _ (not stored)
                        // - Empty closures are handled specially in map (dummy i32 passed directly)
                        if args.is_empty() {
                            return Err(err_spirv!(
                                "empty tuple reached SPIR-V lowering; unit values must be handled at call sites"
                            ));
                        }
                        // Non-empty tuple becomes struct
                        let field_types = args
                            .iter()
                            .map(|arg| self.polytype_to_spirv(arg))
                            .collect::<Result<Vec<_>>>()?;
                        self.get_or_create_struct_type(field_types)
                    }
                    TypeName::Array => {
                        let Some(tensor) = ty.as_tensor() else {
                            return Err(err_spirv!(
                                "malformed array type reached SPIR-V lowering: {:?}",
                                ty
                            ));
                        };
                        let Some(storage) = ty.array_storage() else {
                            return Err(err_spirv!("array storage metadata is missing: {:?}", ty));
                        };
                        let elem_type = self.polytype_to_spirv(tensor.elem)?;
                        let variant = storage.variant;

                        // Dispatch on variant first - View arrays are always {offset, len} structs
                        if let PolyType::Constructed(TypeName::ArrayVariantView, _) = variant {
                            // View variant: struct { offset: u32, len: u32 }. The
                            // backing storage buffer is identified by the concrete
                            // `Buffer(set, binding)` in the view's type, not a
                            // runtime field — so the descriptor survives phis and
                            // view-preserving intrinsics where reverse-mapping a
                            // runtime constant can't recover it.
                            self.get_or_create_struct_type(vec![self.u32_type, self.u32_type])
                        } else if let PolyType::Constructed(TypeName::ArrayVariantVirtual, _) = variant {
                            // Virtual variant: struct { start, step, len } for range representation
                            // Use the element type so u32 ranges get {u32, u32, u32}.
                            self.get_or_create_struct_type(vec![elem_type, elem_type, elem_type])
                        } else if let PolyType::Constructed(TypeName::ArrayVariantBounded, _) = variant {
                            // Bounded variant: struct { buffer: [N]T, len: i32 } —
                            // function-local fixed-capacity buffer plus a runtime count.
                            // The buffer member is a Composite [N]T (sized SPIR-V array).
                            // The len field is i32 to match the language's `length()`
                            // result type and the index type expected by `array_with`.
                            let buf_type = self.sized_tensor_to_spirv(elem_type, tensor.dims)?;
                            self.get_or_create_struct_type(vec![buf_type, self.i32_type])
                        } else {
                            // Composite variant (or placeholder): sized array value
                            self.sized_tensor_to_spirv(elem_type, tensor.dims)?
                        }
                    }
                    TypeName::Vec => {
                        let Some(tensor) = ty.as_tensor() else {
                            return Err(err_spirv!(
                                "malformed vector type reached SPIR-V lowering: {:?}",
                                ty
                            ));
                        };
                        let Some(size) = tensor.concrete_dim(0) else {
                            return Err(err_spirv!("vector size is not concrete: {:?}", ty));
                        };
                        let size = u32::try_from(size)
                            .map_err(|_| err_spirv!("vector size exceeds SPIR-V limits: {:?}", ty))?;
                        let elem_type = self.polytype_to_spirv(tensor.elem)?;
                        self.get_or_create_vec_type(elem_type, size)
                    }
                    TypeName::Mat => {
                        let Some(tensor) = ty.as_tensor() else {
                            return Err(err_spirv!(
                                "malformed matrix type reached SPIR-V lowering: {:?}",
                                ty
                            ));
                        };
                        let Some(cols) = tensor.concrete_dim(0) else {
                            return Err(err_spirv!("matrix column count is not concrete: {:?}", ty));
                        };
                        let Some(rows) = tensor.concrete_dim(1) else {
                            return Err(err_spirv!("matrix row count is not concrete: {:?}", ty));
                        };
                        let cols = u32::try_from(cols).map_err(|_| {
                            err_spirv!("matrix column count exceeds SPIR-V limits: {:?}", ty)
                        })?;
                        let rows = u32::try_from(rows)
                            .map_err(|_| err_spirv!("matrix row count exceeds SPIR-V limits: {:?}", ty))?;
                        let elem_type = self.polytype_to_spirv(tensor.elem)?;
                        let col_vec_type = self.get_or_create_vec_type(elem_type, rows);
                        *self.builder.type_matrix(builder::TypeId::new(col_vec_type), cols)
                    }
                    TypeName::Record(_fields) => {
                        let field_types = args
                            .iter()
                            .map(|arg| self.polytype_to_spirv(arg))
                            .collect::<Result<Vec<_>>>()?;
                        self.get_or_create_struct_type(field_types)
                    }
                    TypeName::Pointer => {
                        // Pointer type: args[0] is pointee type, args[1] is address space
                        let Some(pointee) = args.first() else {
                            return Err(err_spirv!("pointer type is missing its pointee: {:?}", ty));
                        };
                        let pointee_type = self.polytype_to_spirv(pointee)?;
                        let sc = args
                            .get(1)
                            .map(Constructor::resolve_storage_class)
                            .unwrap_or(StorageClass::Function);
                        self.get_or_create_ptr_type(sc, pointee_type)
                    }
                    TypeName::Existential(_) => {
                        // Existential type: unwrap and convert the inner type (in args[0])
                        // The size variable is runtime-determined, handled by Slice representation
                        let Some(inner) = args.first() else {
                            return Err(err_spirv!("existential type is missing its inner type: {:?}", ty));
                        };
                        self.polytype_to_spirv(inner)?
                    }
                    TypeName::Arrow => {
                        // Arrow types (function types) come from closures that have been defunctionalized.
                        // Since closures are represented as (captures_tuple, lambda_name), the actual
                        // runtime value is just the captures. The Arrow type is a phantom type used
                        // for type checking only. Map to unit type since it has no runtime representation.
                        self.void_type
                    }
                    TypeName::ArrayVariantComposite
                    | TypeName::ArrayVariantView
                    | TypeName::Buffer(_)
                    | TypeName::NoBuffer
                    | TypeName::PointerFunction
                    | TypeName::PointerInput
                    | TypeName::PointerOutput
                    | TypeName::PointerStorage => {
                        // Address space markers are used within Array/Pointer types but shouldn't appear
                        // as standalone types requiring SPIR-V representation.
                        return Err(err_spirv!(
                            "address-space marker {:?} reached SPIR-V lowering as a standalone type: {:?}",
                            name,
                            ty
                        ));
                    }
                    TypeName::AddressPlaceholder | TypeName::SizePlaceholder => {
                        return Err(err_spirv!(
                            "unresolved placeholder reached SPIR-V lowering: {:?}",
                            ty
                        ));
                    }
                    TypeName::Texture2D => {
                        // 2D float sampled image. sampled=1 (used with a
                        // sampler), Unknown format (sampled images don't
                        // carry a format). rspirv dedups type_image.
                        *self.builder.type_image(
                            builder::TypeId::new(self.f32_type),
                            spirv::Dim::Dim2D,
                            0, // depth: not a depth texture
                            0, // arrayed: single image
                            0, // ms: not multisampled
                            1, // sampled: sampled (vs storage) image
                            spirv::ImageFormat::Unknown,
                            None,
                        )
                    }
                    TypeName::Sampler => *self.builder.type_sampler(),
                    TypeName::Raster => {
                        return Err(err_spirv!(
                            "raster stage token reached runtime SPIR-V type lowering"
                        ));
                    }
                    TypeName::StorageTexture => {
                        return Err(err_spirv!("storage texture reached runtime SPIR-V type lowering"));
                    }
                    _ => {
                        return Err(err_spirv!("unsupported type reached SPIR-V lowering: {:?}", name));
                    }
                }
            }
        };
        Ok(result)
    }

    fn sized_tensor_to_spirv(
        &mut self,
        elem_type: spirv::Word,
        dims: &[PolyType<TypeName>],
    ) -> Result<spirv::Word> {
        let mut result = elem_type;
        for dim in dims.iter().rev() {
            let PolyType::Constructed(TypeName::Size(size), _) = dim else {
                return Err(err_spirv!(
                    "composite tensor dimension is not concrete at SPIR-V lowering: {:?}",
                    dim
                ));
            };
            let size = u32::try_from(*size)
                .map_err(|_| err_spirv!("tensor dimension exceeds SPIR-V limits: {:?}", dim))?;
            let size_const = self.const_u32(size);
            let array_type = *self.builder.type_array(builder::TypeId::new(result), size_const);
            self.builder
                .register_array_element(builder::TypeId::new(array_type), builder::TypeId::new(result));
            result = array_type;
        }
        Ok(result)
    }

    pub(super) fn get_or_create_vec_type(&mut self, elem_type: spirv::Word, size: u32) -> spirv::Word {
        *self.builder.type_vec(builder::TypeId::new(elem_type), size)
    }

    pub(super) fn get_or_create_struct_type(&mut self, field_types: Vec<spirv::Word>) -> spirv::Word {
        *self.builder.type_struct(field_types.into_iter().map(builder::TypeId::new).collect())
    }

    pub(super) fn get_or_create_runtime_array_type(
        &mut self,
        elem_type: spirv::Word,
        stride: u32,
    ) -> spirv::Word {
        *self.builder.type_runtime_array(builder::TypeId::new(elem_type), stride)
    }
}
