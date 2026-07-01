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
    pub(super) fn polytype_to_spirv(&mut self, ty: &PolyType<TypeName>) -> spirv::Word {
        if let Some(&cached) = self.polytype_cache.get(ty) {
            return cached;
        }
        let result = self.polytype_to_spirv_uncached(ty);
        self.polytype_cache.insert(ty.clone(), result);
        result
    }

    pub(super) fn polytype_to_spirv_uncached(&mut self, ty: &PolyType<TypeName>) -> spirv::Word {
        match ty {
            PolyType::Variable(id) => {
                panic!("BUG: Unresolved type variable Variable({}) reached lowering.", id);
            }
            PolyType::Constructed(name, args) => {
                // Assert that no UserVar or SizeVar reaches lowering
                match name {
                    TypeName::UserVar(v) => {
                        panic!("BUG: UserVar('{}') reached lowering.", v);
                    }
                    TypeName::SizeVar(v) => {
                        panic!("BUG: SizeVar('{}') reached lowering.", v);
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
                            panic!(
                                "BUG: Empty tuple type reached lowering. Empty tuples/unit values should be \
                                handled at call sites (let _ = ..., map with empty closures, etc.)"
                            );
                        }
                        // Non-empty tuple becomes struct
                        let field_types: Vec<spirv::Word> =
                            args.iter().map(|a| self.polytype_to_spirv(a)).collect();
                        self.get_or_create_struct_type(field_types)
                    }
                    TypeName::Array => {
                        // Array[elem, variant, dim_0, ...]
                        let elem = ty.elem_type().expect("Array has elem");
                        let elem_type = self.polytype_to_spirv(elem);
                        let size = ty.array_size().expect("Array has size");
                        let variant = ty.array_variant().expect("Array has variant");

                        // Dispatch on variant first - View arrays are always {offset, len} structs
                        if let PolyType::Constructed(TypeName::ArrayVariantView, _) = variant {
                            // View variant: struct { offset: u32, len: u32 }. The
                            // backing storage buffer is identified by the concrete
                            // `Region(set, binding)` in the view's type, not a
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
                            let n = match size {
                                PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
                                _ => panic!("BUG: Bounded array requires Size(N) capacity, got {:?}", size),
                            };
                            let size_const = self.const_u32(n);
                            let buf_type =
                                *self.builder.type_array(builder::TypeId::new(elem_type), size_const);
                            self.builder.register_array_element(
                                builder::TypeId::new(buf_type),
                                builder::TypeId::new(elem_type),
                            );
                            self.get_or_create_struct_type(vec![buf_type, self.i32_type])
                        } else {
                            // Composite variant (or placeholder): sized array value
                            match size {
                                PolyType::Constructed(TypeName::Size(n), _) => {
                                    // Fixed-size array (use unsigned int for array size per SPIR-V convention)
                                    let size_const = self.const_u32(*n as u32);
                                    let arr_type = *self
                                        .builder
                                        .type_array(builder::TypeId::new(elem_type), size_const);
                                    self.builder.register_array_element(
                                        builder::TypeId::new(arr_type),
                                        builder::TypeId::new(elem_type),
                                    );
                                    arr_type
                                }
                                PolyType::Constructed(TypeName::SizePlaceholder, _) => {
                                    panic!("SizePlaceholder should be resolved before SPIR-V lowering");
                                }
                                PolyType::Variable(_) => {
                                    // Unsized composite array - not supported
                                    panic!("BUG: Composite variant unsized arrays not supported: {:?}", ty);
                                }
                                _ => {
                                    panic!(
                                        "BUG: Array type has invalid size argument: {:?}. This should have been resolved during type checking.",
                                        size
                                    );
                                }
                            }
                        }
                    }
                    TypeName::Vec => {
                        // Vec[elem, Size(n)]
                        let elem = ty.elem_type().expect("Vec has elem");
                        let elem_type = self.polytype_to_spirv(elem);
                        let size = ty.vec_size().expect("Vec has concrete size") as u32;
                        self.get_or_create_vec_type(elem_type, size)
                    }
                    TypeName::Mat => {
                        // Mat[elem, Size(cols), Size(rows)]
                        let elem = ty.elem_type().expect("Mat has elem");
                        let elem_type = self.polytype_to_spirv(elem);
                        let cols = ty.mat_cols().expect("Mat has concrete cols") as u32;
                        let rows = ty.mat_rows().expect("Mat has concrete rows") as u32;
                        let col_vec_type = self.get_or_create_vec_type(elem_type, rows);
                        *self.builder.type_matrix(builder::TypeId::new(col_vec_type), cols)
                    }
                    TypeName::Record(_fields) => {
                        let field_types: Vec<spirv::Word> =
                            args.iter().map(|a| self.polytype_to_spirv(a)).collect();
                        self.get_or_create_struct_type(field_types)
                    }
                    TypeName::Pointer => {
                        // Pointer type: args[0] is pointee type, args[1] is address space
                        if args.is_empty() {
                            panic!("BUG: Pointer type requires a pointee type argument.");
                        }
                        let pointee_type = self.polytype_to_spirv(&args[0]);
                        let sc = args
                            .get(1)
                            .map(Constructor::resolve_storage_class)
                            .unwrap_or(StorageClass::Function);
                        self.get_or_create_ptr_type(sc, pointee_type)
                    }
                    TypeName::Unique => {
                        // Unique type wrapper: strip and convert underlying type
                        // Unique is only used for alias checking, has no runtime representation
                        if args.is_empty() {
                            panic!("BUG: Unique type requires an underlying type argument.");
                        }
                        self.polytype_to_spirv(&args[0])
                    }
                    TypeName::Existential(_) => {
                        // Existential type: unwrap and convert the inner type (in args[0])
                        // The size variable is runtime-determined, handled by Slice representation
                        let inner = &args[0];
                        self.polytype_to_spirv(inner)
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
                    | TypeName::Region(_)
                    | TypeName::NoRegion
                    | TypeName::PointerFunction
                    | TypeName::PointerInput
                    | TypeName::PointerOutput
                    | TypeName::PointerStorage => {
                        // Address space markers are used within Array/Pointer types but shouldn't appear
                        // as standalone types requiring SPIR-V representation.
                        panic!(
                            "BUG: Address space marker {:?} reached polytype_to_spirv as standalone type. \
                            This should only appear as part of Array[elem, addrspace, size] or Pointer[pointee, addrspace]. Full type: {:?}",
                            name, ty
                        );
                    }
                    TypeName::AddressPlaceholder | TypeName::SizePlaceholder => {
                        panic!("Placeholders should be resolved before SPIR-V lowering");
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
                    TypeName::StorageTexture => {
                        // Use the program-wide default format (set in
                        // `lower_ssa_program_impl`) so function signatures
                        // match the entry-point variable's format-aware type.
                        // None → fall back to Unknown; only happens when no
                        // entry uses a storage_image, in which case nothing
                        // reaches this type anyway.
                        let format = self
                            .storage_image_default_format
                            .map(storage_image_format_to_spirv)
                            .unwrap_or(spirv::ImageFormat::Unknown);
                        *self.builder.type_image(
                            builder::TypeId::new(self.f32_type),
                            spirv::Dim::Dim2D,
                            0,
                            0,
                            0,
                            2, // sampled=2 = storage image
                            format,
                            None,
                        )
                    }
                    _ => {
                        panic!(
                            "BUG: Unknown type reached lowering: {:?}. This should have been caught during type checking.",
                            name
                        )
                    }
                }
            }
        }
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
