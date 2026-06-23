//! `LowerCtx::lower_builtin_call` — SPIR-V dispatch for every
//! builtin (reduce / scatter / gather / texture / atomic / extern).

use super::lower::LowerCtx;
use super::*;

impl<'a, 'b> LowerCtx<'a, 'b> {
    pub(super) fn lower_builtin_call(
        &mut self,
        id: crate::builtins::BuiltinId,
        builtin: &BuiltinLowering,
        dispatch_name: &str,
        value_refs: &[ValueRef],
        arg_ids: &[spirv::Word],
        result_ty: spirv::Word,
        inst: &WynInstNode,
    ) -> Result<spirv::Word> {
        match builtin {
            BuiltinLowering::PrimOp(prim_op) => self.lower_primop(prim_op, arg_ids, result_ty),
            BuiltinLowering::LinkedSpirv(linkage_name) => {
                let func_id = self
                    .constructor
                    .linked_functions
                    .get(*linkage_name)
                    .copied()
                    .ok_or_else(|| err_spirv!("Unknown linked function: {}", linkage_name))?;
                Ok(self.constructor.builder.function_call(result_ty, None, func_id, arg_ids.to_vec())?)
            }
            BuiltinLowering::NotLowered => {
                bail_spirv!(
                    "NotLowered builtin '{}' reached backend dispatch — \
                     promote it to `BuiltinLowering::ByBuiltinId` (or another typed variant)",
                    dispatch_name
                )
            }
            BuiltinLowering::ExtInstSplat { ext, splat_args } => {
                // GLSL.std.450 ext-inst with operand splatting.
                // Splat each scalar at the named positions to vec
                // width before emitting `OpExtInst` — required
                // because the instruction expects every operand
                // to match the result type.
                let mut operands: Vec<Operand> = arg_ids.iter().map(|&id| Operand::IdRef(id)).collect();
                let result_ssa_ty = inst.result.map(|r| self.body.inner.value_type(r).clone());
                let result_is_vec = result_ssa_ty.as_ref().is_some_and(|t| t.is_vec());
                if result_is_vec {
                    let result_ssa_ty = result_ssa_ty.as_ref().unwrap();
                    for &pos in *splat_args {
                        if self.get_value_type_ref(value_refs[pos]).is_scalar() {
                            let splatted = self.splat_scalar(arg_ids[pos], result_ssa_ty, result_ty)?;
                            operands[pos] = Operand::IdRef(splatted);
                        }
                    }
                }
                let glsl = self.constructor.glsl_ext_inst_id;
                Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, *ext, operands)?)
            }
            BuiltinLowering::ByBuiltinId => {
                let known = catalog().known();
                if id == known.uninit {
                    // Zero-initialized value (OpConstantNull), cached by type.
                    Ok(*self.constructor.builder.const_null(builder::TypeId::new(result_ty)))
                } else if id == known.texture_load {
                    // texture_load(tex, coord, lod) → OpImageFetch. Raw texel
                    // fetch; referentially transparent (no derivatives).
                    // arg_ids = [image, coord, lod].
                    if arg_ids.len() != 3 {
                        bail_spirv!("texture_load requires 3 arguments");
                    }
                    Ok(self.constructor.builder.image_fetch(
                        result_ty,
                        None,
                        arg_ids[0],
                        arg_ids[1],
                        Some(spirv::ImageOperands::LOD),
                        [Operand::IdRef(arg_ids[2])],
                    )?)
                } else if id == known.image_store {
                    // image_store(image, ivec2, vec4) → OpImageWrite.
                    // arg_ids = [image, coord, texel]. SPIR-V's
                    // OpImageWrite has no result type / no result id;
                    // we return the zero word — the caller never reads
                    // the result of a unit-typed App.
                    if arg_ids.len() != 3 {
                        bail_spirv!("image_store requires 3 arguments");
                    }
                    self.constructor.builder.image_write(arg_ids[0], arg_ids[1], arg_ids[2], None, [])?;
                    Ok(0)
                } else if id == known.image_load {
                    // image_load(image, ivec2) → OpImageRead.
                    // arg_ids = [image, coord]. Result is vec4f32.
                    if arg_ids.len() != 2 {
                        bail_spirv!("image_load requires 2 arguments");
                    }
                    Ok(self.constructor.builder.image_read(
                        result_ty,
                        None,
                        arg_ids[0],
                        arg_ids[1],
                        None,
                        [],
                    )?)
                } else if id == known.texture_sample {
                    // texture_sample(tex, samp, uv, lod) → OpSampledImage +
                    // OpImageSampleExplicitLod. v1 uses EXPLICIT LOD (the
                    // trailing arg) rather than implicit/derivative LOD, so
                    // the result is a pure function of its arguments —
                    // referentially transparent and valid in any stage. See
                    // the texture plan's v2 note for gradient-based filtering
                    // (`texture_sample_grad`). arg_ids = [image, sampler, uv, lod].
                    if arg_ids.len() != 4 {
                        bail_spirv!("texture_sample requires 4 arguments");
                    }
                    let image_ty = self
                        .constructor
                        .polytype_to_spirv(&PolyType::Constructed(TypeName::Texture2D, vec![]));
                    let sampled_img_ty = self.constructor.builder.type_sampled_image(image_ty);
                    let sampled = self.constructor.builder.sampled_image(
                        sampled_img_ty,
                        None,
                        arg_ids[0],
                        arg_ids[1],
                    )?;
                    Ok(self.constructor.builder.image_sample_explicit_lod(
                        result_ty,
                        None,
                        sampled,
                        arg_ids[2],
                        spirv::ImageOperands::LOD,
                        [Operand::IdRef(arg_ids[3])],
                    )?)
                } else if id == known.array_with || id == known.array_with_in_place {
                    // _w_array_with(array, index, value) - array update.
                    // Same SPIR-V lowering for both flavors today — SPIR-V can
                    // already express OpCompositeInsert (literal idx) or a
                    // local-buffer round-trip (dynamic idx). An in-place
                    // optimization for the dynamic case is left as future work.
                    if arg_ids.len() != 3 {
                        bail_spirv!("ArrayWith requires 3 arguments");
                    }
                    let arr = arg_ids[0];
                    let idx = arg_ids[1];
                    let val = arg_ids[2];

                    // View-variant arrays write back into the backing storage
                    // buffer via OpAccessChain+OpStore. The "result" view is
                    // structurally the same as the input; carry the input's
                    // SPIR-V word. The result value's type carries the same
                    // region, so downstream `ViewIndex` consumers resolve the
                    // buffer from it.
                    let arr_ty = self.get_value_type_ref(value_refs[0]);
                    let is_view = arr_ty
                        .array_variant()
                        .map(|v| matches!(v, PolyType::Constructed(TypeName::ArrayVariantView, _)))
                        .unwrap_or(false);
                    if is_view {
                        let view_ssa = value_refs[0]
                            .as_ssa()
                            .ok_or_else(|| err_spirv!("array_with on view must take SSA view value"))?;
                        let buffer_var = self.view_buffer_var(view_ssa)?;
                        let u32_ty = self.constructor.u32_type;
                        let base_offset =
                            self.constructor.builder.composite_extract(u32_ty, None, arr, [0u32])?;
                        let idx_u32 = self.constructor.builder.bitcast(u32_ty, None, idx)?;
                        let final_index =
                            self.constructor.builder.i_add(u32_ty, None, base_offset, idx_u32)?;
                        let elem_ty = arr_ty.elem_type().expect("View has elem").clone();
                        let elem_spirv = self.constructor.polytype_to_spirv(&elem_ty);
                        let elem_ptr_type = self
                            .constructor
                            .get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, elem_spirv);
                        let zero = self.constructor.const_u32(0);
                        let elem_ptr = self.constructor.builder.access_chain(
                            elem_ptr_type,
                            None,
                            buffer_var,
                            [zero, final_index],
                        )?;
                        self.constructor.builder.store(elem_ptr, val, None, [])?;
                        return Ok(arr);
                    }

                    let literal_idx = match value_refs.get(1).and_then(|vr| vr.as_const()) {
                        Some(ConstantValue::I32(v)) => Some(v as i32),
                        Some(ConstantValue::U32(v)) => Some(v as i32),
                        _ => self.constructor.get_const_i32_value(idx),
                    };

                    // Bounded array: the value is a struct { buffer: [N]T, len:
                    // i32 }. Update buffer[idx] = val (len unchanged) and
                    // reassemble the struct. The generic path below would treat
                    // the struct itself as a plain array and fail to find its
                    // element type — this is the path the filter compaction loop
                    // hits when a Bounded filter result is materialized (e.g.
                    // filter→map / filter→scan).
                    let is_bounded = arr_ty
                        .array_variant()
                        .map(|v| matches!(v, PolyType::Constructed(TypeName::ArrayVariantBounded, _)))
                        .unwrap_or(false);
                    if is_bounded {
                        let elem_ty = arr_ty.elem_type().expect("Bounded has elem").clone();
                        let n = match arr_ty.array_size() {
                            Some(PolyType::Constructed(TypeName::Size(n), _)) => *n as u32,
                            other => {
                                bail_spirv!("Bounded array_with requires Size(N) capacity, got {:?}", other)
                            }
                        };
                        let elem_spirv = self.constructor.polytype_to_spirv(&elem_ty);
                        let size_const = self.constructor.const_u32(n);
                        let buf_type = self.constructor.builder.type_array(elem_spirv, size_const);
                        // The struct's [N]T member type must be element-resolvable
                        // for the dynamic-index access_chain below.
                        self.constructor.builder.register_array_element(
                            builder::TypeId::new(buf_type),
                            builder::TypeId::new(elem_spirv),
                        );
                        let buffer =
                            self.constructor.builder.composite_extract(buf_type, None, arr, [0u32])?;
                        let new_buffer = if let Some(literal_idx) = literal_idx {
                            self.constructor.builder.composite_insert(
                                buf_type,
                                None,
                                val,
                                buffer,
                                [literal_idx as u32],
                            )?
                        } else {
                            let buf_var =
                                self.constructor.declare_variable("_bounded_buf_tmp", buf_type)?;
                            self.constructor.builder.store(buf_var, buffer, None, [])?;
                            let elem_ptr_ty = *self.constructor.builder.type_pointer(
                                spirv::StorageClass::Function,
                                builder::TypeId::new(elem_spirv),
                            );
                            let elem_ptr =
                                self.constructor.builder.access_chain(elem_ptr_ty, None, buf_var, [idx])?;
                            self.constructor.builder.store(elem_ptr, val, None, [])?;
                            self.constructor.builder.load(buf_type, None, buf_var, None, [])?
                        };
                        // Reassemble { buffer: new_buffer, len: <unchanged> }.
                        return Ok(self.constructor.builder.composite_insert(
                            result_ty,
                            None,
                            new_buffer,
                            arr,
                            [0u32],
                        )?);
                    }

                    if let Some(literal_idx) = literal_idx {
                        Ok(self.constructor.builder.composite_insert(
                            result_ty,
                            None,
                            val,
                            arr,
                            [literal_idx as u32],
                        )?)
                    } else {
                        let arr_var = self.constructor.declare_variable("_array_with_tmp", result_ty)?;
                        self.constructor.builder.store(arr_var, arr, None, [])?;
                        let elem_ty = self.constructor.get_array_element_type(result_ty).map_err(|_| {
                            crate::err_spirv!(
                                "ArrayWith: element type not found for array type ID {}. \
                                 Unsized or view arrays may not support indexed writes.",
                                result_ty
                            )
                        })?;
                        let elem_ptr_ty = *self
                            .constructor
                            .builder
                            .type_pointer(spirv::StorageClass::Function, builder::TypeId::new(elem_ty));
                        let elem_ptr =
                            self.constructor.builder.access_chain(elem_ptr_ty, None, arr_var, [idx])?;
                        self.constructor.builder.store(elem_ptr, val, None, [])?;
                        Ok(self.constructor.builder.load(result_ty, None, arr_var, None, [])?)
                    }
                } else if id == known.length {
                    if arg_ids.len() != 1 {
                        bail_spirv!("length requires 1 argument");
                    }
                    let arr_ty = self.get_value_type_ref(value_refs[0]);
                    let variant = arr_ty
                        .array_variant()
                        .ok_or_else(|| err_spirv!("length: expected array type, got {:?}", arr_ty))?;
                    match variant {
                        // View: struct {buffer_ptr, offset, len} — len is u32 in
                        // the struct but the SSA result type is i32. Extract as
                        // u32 then bitcast.
                        // TODO: view struct should use i32 to match language conventions.
                        PolyType::Constructed(TypeName::ArrayVariantView, _) => {
                            let u32_ty = self.constructor.u32_type;
                            // View struct is {offset, len}; len is field 1.
                            let len_u32 = self.constructor.builder.composite_extract(
                                u32_ty,
                                None,
                                arg_ids[0],
                                [1u32],
                            )?;
                            Ok(self.constructor.builder.bitcast(result_ty, None, len_u32)?)
                        }
                        // Virtual (range): struct {start, step, len} — len field
                        // type matches element type (may be u32), but SSA result
                        // is i32. Extract with the actual field type, then
                        // bitcast if needed.
                        PolyType::Constructed(TypeName::ArrayVariantVirtual, _) => {
                            let elem_spirv = self
                                .constructor
                                .polytype_to_spirv(arr_ty.elem_type().expect("virtual array has elem"));
                            if elem_spirv == result_ty {
                                Ok(self.constructor.builder.composite_extract(
                                    result_ty,
                                    None,
                                    arg_ids[0],
                                    [2u32],
                                )?)
                            } else {
                                let len_raw = self.constructor.builder.composite_extract(
                                    elem_spirv,
                                    None,
                                    arg_ids[0],
                                    [2u32],
                                )?;
                                Ok(self.constructor.builder.bitcast(result_ty, None, len_raw)?)
                            }
                        }
                        // Composite: sized SPIR-V array — length is known from the type.
                        PolyType::Constructed(TypeName::ArrayVariantComposite, _) => {
                            match arr_ty.array_size().expect("Array has size") {
                                PolyType::Constructed(TypeName::Size(n), _) => {
                                    Ok(self.constructor.const_i32(*n as i32))
                                }
                                _ => bail_spirv!("length: composite array has unknown size"),
                            }
                        }
                        // Bounded: struct {buffer, len} — extract member 1 (the
                        // runtime count). The `len` field is already i32, matching
                        // SSA's length() result type.
                        PolyType::Constructed(TypeName::ArrayVariantBounded, _) => Ok(self
                            .constructor
                            .builder
                            .composite_extract(result_ty, None, arg_ids[0], [1u32])?),
                        _ => bail_spirv!("length: unknown array variant: {:?}", variant),
                    }
                } else if id == known.slice {
                    if arg_ids.len() != 3 {
                        bail_spirv!("_w_slice requires 3 arguments (arr, start, end)");
                    }
                    let arr = arg_ids[0];
                    let start_id = arg_ids[1];
                    let end_id = arg_ids[2];

                    let arr_ty = self.get_value_type_ref(value_refs[0]);
                    let is_view = arr_ty
                        .array_variant()
                        .map(|v| matches!(v, PolyType::Constructed(TypeName::ArrayVariantView, _)))
                        .unwrap_or(false);

                    if is_view {
                        let elem_ty = arr_ty.elem_type().expect("Array has elem").clone();
                        let u32_ty = self.constructor.u32_type;
                        let base_offset =
                            self.constructor.builder.composite_extract(u32_ty, None, arr, [0u32])?;
                        let result_is_composite = inst
                            .result
                            .map(|v| self.body.get_value_type(v))
                            .map(|t| {
                                t.array_variant()
                                    .map(|v| types::is_array_variant_composite(v))
                                    .unwrap_or(false)
                            })
                            .unwrap_or(false);

                        if result_is_composite {
                            let view_ssa = value_refs[0]
                                .as_ssa()
                                .ok_or_else(|| err_spirv!("slice_to_composite view operand must be SSA"))?;
                            let buffer_var = self.view_buffer_var(view_ssa)?;
                            self.slice_view_to_composite(
                                arr,
                                buffer_var,
                                base_offset,
                                start_id,
                                end_id,
                                &elem_ty,
                                result_ty,
                            )
                        } else {
                            // View-to-view slice: the result value's type carries
                            // the same region as the source, so downstream
                            // consumers recover the buffer from it — the struct
                            // holds only {offset, len}.
                            self.slice_view_to_view(arr, base_offset, start_id, end_id)
                        }
                    } else {
                        self.slice_composite(arr, start_id, end_id, result_ty)
                    }
                } else if id == known.storage_len {
                    if arg_ids.len() != 2 {
                        bail_spirv!("_w_storage_len requires 2 arguments (set, binding)");
                    }
                    let set = match value_refs[0].as_const() {
                        Some(ConstantValue::U32(v)) => v,
                        _ => self
                            .constructor
                            .get_const_u32_value(arg_ids[0])
                            .ok_or_else(|| err_spirv!("_w_storage_len: set must be a u32 constant"))?,
                    };
                    let binding = match value_refs[1].as_const() {
                        Some(ConstantValue::U32(v)) => v,
                        _ => self
                            .constructor
                            .get_const_u32_value(arg_ids[1])
                            .ok_or_else(|| err_spirv!("_w_storage_len: binding must be a u32 constant"))?,
                    };
                    let &(buffer_var, _, _) =
                        self.constructor.storage_buffers.get(&BindingRef::new(set, binding)).ok_or_else(
                            || err_spirv!("Storage buffer not found for set={}, binding={}", set, binding),
                        )?;
                    let len_u32 = self.constructor.builder.array_length(
                        self.constructor.u32_type,
                        None,
                        buffer_var,
                        0,
                    )?;
                    Ok(self.constructor.builder.bitcast(result_ty, None, len_u32)?)
                } else if id == known.thread_id {
                    let gid_var = self
                        .constructor
                        .global_invocation_id
                        .ok_or_else(|| err_spirv!("GlobalInvocationId not set for compute shader"))?;
                    let uvec3_type = self.constructor.get_or_create_vec_type(self.constructor.u32_type, 3);
                    let gid = self.constructor.builder.load(uvec3_type, None, gid_var, None, [])?;
                    Ok(self.constructor.builder.composite_extract(
                        self.constructor.u32_type,
                        None,
                        gid,
                        [0],
                    )?)
                } else if id == known.local_id {
                    let lid_var = self
                        .constructor
                        .local_invocation_id
                        .ok_or_else(|| err_spirv!("LocalInvocationId not set for compute shader"))?;
                    let uvec3_type = self.constructor.get_or_create_vec_type(self.constructor.u32_type, 3);
                    let lid = self.constructor.builder.load(uvec3_type, None, lid_var, None, [])?;
                    Ok(self.constructor.builder.composite_extract(
                        self.constructor.u32_type,
                        None,
                        lid,
                        [0],
                    )?)
                } else if id == known.num_workgroups {
                    let nwg_var = self
                        .constructor
                        .num_workgroups
                        .ok_or_else(|| err_spirv!("NumWorkgroups not set for compute shader"))?;
                    let uvec3_type = self.constructor.get_or_create_vec_type(self.constructor.u32_type, 3);
                    let nwg = self.constructor.builder.load(uvec3_type, None, nwg_var, None, [])?;
                    Ok(self.constructor.builder.composite_extract(
                        self.constructor.u32_type,
                        None,
                        nwg,
                        [0],
                    )?)
                } else if id == known.storage_index || id == known.storage_store {
                    bail_spirv!(
                        "{} reached backend dispatch — should be lowered to \
                         an InstKind::Load/Store side effect during EGIR conversion",
                        dispatch_name
                    )
                } else {
                    bail_spirv!(
                        "ByBuiltinId dispatch: unknown builtin id={:?} dispatch_name={:?}",
                        id,
                        dispatch_name
                    )
                }
            }
        }
    }
}
