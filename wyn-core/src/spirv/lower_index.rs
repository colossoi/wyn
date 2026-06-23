//! Array / slice / view indexing surface on `LowerCtx`.
//!
//! Holds the methods that turn an SSA `Index` / `Slice` /
//! `ViewIndex` / `VirtualIndex` / `CompositeIndex` instruction
//! into the right combination of `OpAccessChain` + `OpLoad` +
//! `OpCompositeExtract` — including the slice-view ↔ slice-composite
//! materialization routines that handle the boundary between
//! storage-backed and register-backed array data.
//!
//! Defined here as an `impl LowerCtx` block; sibling-file
//! dispatchers (`lower.rs::lower_inst`, `lower_builtin.rs`) call
//! into these methods as if they lived in `lower.rs`.

use super::lower::LowerCtx;
use super::*;

impl<'a, 'b> LowerCtx<'a, 'b> {
    /// Slice a storage view, materializing into a composite array.
    /// Loads each element from the buffer via AccessChain+Load.
    pub(super) fn slice_view_to_composite(
        &mut self,
        _view_id: spirv::Word,
        buffer_var: spirv::Word,
        base_offset: spirv::Word,
        start_id: spirv::Word,
        end_id: spirv::Word,
        elem_ty: &PolyType<TypeName>,
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        let start = self
            .constructor
            .get_const_i32_value(start_id)
            .ok_or_else(|| err_spirv!("slice_view_to_composite: start must be a constant"))?
            as u32;
        let end = self
            .constructor
            .get_const_i32_value(end_id)
            .ok_or_else(|| err_spirv!("slice_view_to_composite: end must be a constant"))?
            as u32;

        let elem_spirv = self.constructor.polytype_to_spirv(elem_ty);
        let elem_ptr_type =
            self.constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, elem_spirv);
        let zero = self.constructor.const_i32(0);
        let mut elements = Vec::with_capacity((end - start) as usize);
        for i in start..end {
            let idx_const = self.constructor.const_u32(i);
            let actual_index =
                self.constructor.builder.i_add(self.constructor.u32_type, None, base_offset, idx_const)?;
            let elem_ptr = self.constructor.builder.access_chain(
                elem_ptr_type,
                None,
                buffer_var,
                [zero, actual_index],
            )?;
            let elem = self.constructor.builder.load(elem_spirv, None, elem_ptr, None, [])?;
            elements.push(elem);
        }
        Ok(self.constructor.builder.composite_construct(result_ty, None, elements)?)
    }

    /// Slice a storage view, producing a new handle-based view with adjusted offset/len.
    pub(super) fn slice_view_to_view(
        &mut self,
        _view_id: spirv::Word,
        base_offset: spirv::Word,
        start_id: spirv::Word,
        end_id: spirv::Word,
    ) -> Result<spirv::Word> {
        let u32_ty = self.constructor.u32_type;
        let new_offset = self.constructor.builder.i_add(u32_ty, None, base_offset, start_id)?;
        let new_len = self.constructor.builder.i_sub(u32_ty, None, end_id, start_id)?;
        let view_struct_type = self.constructor.get_or_create_struct_type(vec![u32_ty, u32_ty]);
        Ok(self.constructor.builder.composite_construct(view_struct_type, None, [new_offset, new_len])?)
    }

    /// Slice a value (composite) array by extracting elements and constructing a new array.
    pub(super) fn slice_composite(
        &mut self,
        arr: spirv::Word,
        start_id: spirv::Word,
        end_id: spirv::Word,
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        let start =
            self.constructor
                .get_const_i32_value(start_id)
                .ok_or_else(|| err_spirv!("slice_composite: start must be a constant"))? as u32;
        let end = self
            .constructor
            .get_const_i32_value(end_id)
            .ok_or_else(|| err_spirv!("slice_composite: end must be a constant"))? as u32;
        if end <= start {
            bail_spirv!(
                "slice_composite: end ({}) must be greater than start ({})",
                end,
                start
            );
        }
        let elem_type = self.constructor.get_array_element_type(result_ty)?;
        let mut elements = Vec::with_capacity((end - start) as usize);
        for i in start..end {
            let elem = self.constructor.builder.composite_extract(elem_type, None, arr, [i])?;
            elements.push(elem);
        }
        Ok(self.constructor.builder.composite_construct(result_ty, None, elements)?)
    }

    /// Lower an index operation, dispatching based on the array variant.
    pub(super) fn lower_index(
        &mut self,
        base: ValueRef,
        index: ValueRef,
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        let base_ty = self.get_value_type_ref(base);
        let base_id = self.get_value_ref(base)?;
        let index_id = self.get_value_ref(index)?;

        // Dispatch based on the base type
        match base_ty {
            PolyType::Constructed(TypeName::Pointer, ptr_args) => {
                // Pointer indexing: access_chain + load
                let sc = ptr_args
                    .get(1)
                    .map(Constructor::resolve_storage_class)
                    .unwrap_or(StorageClass::Function);
                let elem_ptr_type = self.constructor.get_or_create_ptr_type(sc, result_ty);
                let elem_ptr =
                    self.constructor.builder.access_chain(elem_ptr_type, None, base_id, [index_id])?;
                Ok(self.constructor.builder.load(result_ty, None, elem_ptr, None, [])?)
            }

            _ if base_ty.is_array() => {
                let variant = base_ty.array_variant().expect("Array has variant");
                let elem = base_ty.elem_type().expect("Array has elem");

                if types::is_array_variant_view(variant) {
                    // View variant: {offset, len} struct; backing buffer
                    // recovered from the view type's region.
                    self.lower_view_index(
                        base.as_ssa().expect("view base must be SSA"),
                        base_id,
                        index_id,
                        result_ty,
                        elem,
                    )
                } else if types::is_array_variant_virtual(variant) {
                    // Virtual variant: {start, step, len} - computed array
                    self.lower_virtual_index(base_id, index_id, result_ty)
                } else if types::is_array_variant_bounded(variant) {
                    // Bounded variant: {buffer: [N]T, len: u32} struct.
                    // Extract the buffer (member 0), then index it as a Composite.
                    let n = match base_ty.array_size().expect("Array has size") {
                        PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
                        _ => bail_spirv!("Bounded array must have Size(N) capacity"),
                    };
                    let size_const = self.constructor.const_u32(n);
                    let buf_ty = self.constructor.builder.type_array(result_ty, size_const);
                    self.constructor.builder.register_array_element(
                        builder::TypeId::new(buf_ty),
                        builder::TypeId::new(result_ty),
                    );
                    let buf_id =
                        self.constructor.builder.composite_extract(buf_ty, None, base_id, [0u32])?;
                    if let Some(const_idx) = self.try_resolve_const_index(index) {
                        Ok(self.constructor.builder.composite_extract(
                            result_ty,
                            None,
                            buf_id,
                            [const_idx],
                        )?)
                    } else {
                        // Synthesize a composite [N]T base type for the helper.
                        let composite_ty = PolyType::Constructed(
                            TypeName::Array,
                            vec![
                                elem.clone(),
                                PolyType::Constructed(TypeName::ArrayVariantComposite, vec![]),
                                base_ty.array_size().expect("Array has size").clone(),
                                crate::types::no_region(),
                            ],
                        );
                        self.lower_composite_index(buf_id, index_id, result_ty, &composite_ty)
                    }
                } else {
                    // Composite variant: SPIR-V array value
                    // Check for compile-time constant index for OpCompositeExtract
                    if let Some(const_idx) = self.try_resolve_const_index(index) {
                        Ok(self.constructor.builder.composite_extract(
                            result_ty,
                            None,
                            base_id,
                            [const_idx],
                        )?)
                    } else {
                        self.lower_composite_index(base_id, index_id, result_ty, &base_ty)
                    }
                }
            }

            // Vec types - use vector_extract_dynamic
            PolyType::Constructed(TypeName::Vec, _) => {
                Ok(self.constructor.builder.vector_extract_dynamic(result_ty, None, base_id, index_id)?)
            }

            _ => bail_spirv!("Index called on non-array/non-pointer type: {:?}", base_ty),
        }
    }

    /// Lower indexing into a View array ({buffer_id, offset, len} handle-based struct).
    /// Try to resolve a ValueRef to a constant u32 index.
    /// Handles both inline ValueRef::Const and SSA instructions that produce constants.
    pub(super) fn try_resolve_const_index(&self, vr: ValueRef) -> Option<u32> {
        match vr {
            ValueRef::Const(ConstantValue::U32(i)) => Some(i),
            ValueRef::Const(ConstantValue::I32(i)) => Some(i as u32),
            ValueRef::Ssa(id) => {
                let inst_id = match self.body.inner.values.get(id)?.def {
                    crate::ssa::framework::ValueDef::Inst { inst } => inst,
                    _ => return None,
                };
                match &self.body.inner.insts.get(inst_id)?.data {
                    InstKind::Op {
                        tag: crate::op::OpTag::Int(s) | crate::op::OpTag::Uint(s),
                        ..
                    } => s.parse::<u32>().ok().or_else(|| s.parse::<i32>().ok().map(|i| i as u32)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// The SPIR-V storage-buffer variable backing a view value, recovered from
    /// the concrete `Region(set, binding)` in the value's type. A view's
    /// descriptor is a static property of its type — pinned at the entry,
    /// carried by unification through slices, block params, and calls — so it
    /// is read here rather than tracked in a side-map.
    pub(super) fn view_buffer_var(&mut self, view_ssa: ValueId) -> Result<spirv::Word> {
        let ty = self.body.get_value_type(view_ssa).clone();
        let br = crate::types::array_view_region(&ty).ok_or_else(|| {
            err_spirv_at!(
                self.blame_span(),
                "view value {:?} has no concrete buffer region in its type: {:?}",
                view_ssa,
                ty
            )
        })?;
        let buf_id = self.constructor.get_or_assign_buffer_id(br.set, br.binding);
        let (buffer_var, _) =
            self.constructor.buffer_vars.get(buf_id as usize).copied().ok_or_else(|| {
                err_spirv_at!(
                    self.blame_span(),
                    "view region (set={}, binding={}) → buffer_id {} not in buffer_vars",
                    br.set,
                    br.binding,
                    buf_id
                )
            })?;
        Ok(buffer_var)
    }

    pub(super) fn lower_view_index(
        &mut self,
        view_ssa: ValueId,
        view_id: spirv::Word,
        index_id: spirv::Word,
        result_ty: spirv::Word,
        elem_ty: &PolyType<TypeName>,
    ) -> Result<spirv::Word> {
        let u32_ty = self.constructor.u32_type;

        // Buffer-var lookup goes through the type's region, not runtime struct
        // extraction — field 0 of the view is redundant scaffolding.
        let buffer_var = self.view_buffer_var(view_ssa)?;
        let offset_val = self.constructor.builder.composite_extract(u32_ty, None, view_id, [0u32])?;

        // TODO: The view struct stores {u32, u32, u32} but the language uses i32 for
        // indices everywhere. This bitcast papers over the mismatch. The view struct
        // fields should be i32 to match the rest of the language, eliminating this cast
        // and the u32/i32 inconsistency throughout the SOAC lowering pipeline.
        let index_u32 = self.constructor.builder.bitcast(self.constructor.u32_type, None, index_id)?;

        // Compute final index = offset + index
        let final_index =
            self.constructor.builder.i_add(self.constructor.u32_type, None, offset_val, index_u32)?;

        // Access chain directly on the buffer variable
        let elem_spirv = self.constructor.polytype_to_spirv(elem_ty);
        let elem_ptr_type =
            self.constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, elem_spirv);
        let zero = self.constructor.const_u32(0);
        let elem_ptr =
            self.constructor.builder.access_chain(elem_ptr_type, None, buffer_var, [zero, final_index])?;
        Ok(self.constructor.builder.load(result_ty, None, elem_ptr, None, [])?)
    }

    /// Lower indexing into a Virtual array ({start, step, len} struct).
    pub(super) fn lower_virtual_index(
        &mut self,
        range_id: spirv::Word,
        index_id: spirv::Word,
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        // Virtual array is {start, step, len}
        // Result = start + index * step
        let i32_type = self.constructor.i32_type;
        let start = self.constructor.builder.composite_extract(i32_type, None, range_id, [0])?;
        let step = self.constructor.builder.composite_extract(i32_type, None, range_id, [1])?;
        let offset = self.constructor.builder.i_mul(i32_type, None, index_id, step)?;
        Ok(self.constructor.builder.i_add(result_ty, None, start, offset)?)
    }

    /// Lower indexing into a Composite array (SPIR-V array value).
    pub(super) fn lower_composite_index(
        &mut self,
        array_id: spirv::Word,
        index_id: spirv::Word,
        result_ty: spirv::Word,
        array_ty: &PolyType<TypeName>,
    ) -> Result<spirv::Word> {
        // If index is a compile-time constant, use OpCompositeExtract
        if let Some(literal_idx) = self.constructor.get_const_u32_value(index_id) {
            Ok(self.constructor.builder.composite_extract(result_ty, None, array_id, [literal_idx])?)
        } else {
            // Runtime index - must materialize to local variable
            let spirv_array_type = self.constructor.polytype_to_spirv(array_ty);
            let array_var = self.constructor.declare_variable("_w_index_tmp", spirv_array_type)?;
            self.constructor.builder.store(array_var, array_id, None, [])?;

            let elem_ptr_type = *self
                .constructor
                .builder
                .type_pointer(StorageClass::Function, builder::TypeId::new(result_ty));
            let elem_ptr =
                self.constructor.builder.access_chain(elem_ptr_type, None, array_var, [index_id])?;
            Ok(self.constructor.builder.load(result_ty, None, elem_ptr, None, [])?)
        }
    }

}
