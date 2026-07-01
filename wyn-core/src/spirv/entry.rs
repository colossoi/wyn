//! Entry-point I/O setup. The function body itself is lowered by
//! `lower_ssa_body_for_entry` in `mod.rs`.

use super::*;

/// Lower an SSA entry point to SPIR-V.
pub(super) fn lower_ssa_entry_point(
    constructor: &mut Constructor,
    entry: &EntryPoint,
    written_bindings: &LookupSet<BindingRef>,
) -> Result<()> {
    let body = &entry.body;
    let is_compute = matches!(entry.execution_model, ExecutionModel::Compute { .. });

    // Create I/O variables for entry point
    let mut interfaces = Vec::new();

    // For compute shaders, ensure GlobalInvocationId is created and listed as interface
    if is_compute {
        if let Some(gid_var) = constructor.global_invocation_id {
            // Already created by a previous entry point — just add to this entry's interface
            interfaces.push(gid_var);
        } else {
            let uvec3_type = constructor.get_or_create_vec_type(constructor.u32_type, 3);
            let ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Input, uvec3_type);
            let gid_var = constructor.builder.variable(ptr_type, None, spirv::StorageClass::Input, None);
            constructor.builder.decorate(
                gid_var,
                spirv::Decoration::BuiltIn,
                [Operand::BuiltIn(spirv::BuiltIn::GlobalInvocationId)],
            );
            constructor.global_invocation_id = Some(gid_var);
            interfaces.push(gid_var);
        }
        if let Some(lid_var) = constructor.local_invocation_id {
            interfaces.push(lid_var);
        } else {
            let uvec3_type = constructor.get_or_create_vec_type(constructor.u32_type, 3);
            let ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Input, uvec3_type);
            let lid_var = constructor.builder.variable(ptr_type, None, spirv::StorageClass::Input, None);
            constructor.builder.decorate(
                lid_var,
                spirv::Decoration::BuiltIn,
                [Operand::BuiltIn(spirv::BuiltIn::LocalInvocationId)],
            );
            constructor.local_invocation_id = Some(lid_var);
            interfaces.push(lid_var);
        }
        if let Some(nwg_var) = constructor.num_workgroups {
            interfaces.push(nwg_var);
        } else {
            let uvec3_type = constructor.get_or_create_vec_type(constructor.u32_type, 3);
            let ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Input, uvec3_type);
            let nwg_var = constructor.builder.variable(ptr_type, None, spirv::StorageClass::Input, None);
            constructor.builder.decorate(
                nwg_var,
                spirv::Decoration::BuiltIn,
                [Operand::BuiltIn(spirv::BuiltIn::NumWorkgroups)],
            );
            constructor.num_workgroups = Some(nwg_var);
            interfaces.push(nwg_var);
        }
    }

    // Create push constant block for compute shader broadcast inputs
    let pc_inputs: Vec<(usize, u32)> = entry
        .inputs
        .iter()
        .enumerate()
        .filter_map(|(i, inp)| inp.push_constant.map(|pc| (i, pc.offset)))
        .collect();
    let pc_var = if !pc_inputs.is_empty() {
        // Build member types for push constant block
        let member_types: Vec<spirv::Word> =
            pc_inputs.iter().map(|&(i, _)| constructor.polytype_to_spirv(&entry.inputs[i].ty)).collect();
        let member_offsets: Vec<u32> = pc_inputs.iter().map(|&(_, off)| off).collect();
        let member_poly_types: Vec<&PolyType<TypeName>> =
            pc_inputs.iter().map(|&(i, _)| &entry.inputs[i].ty).collect();

        let pc_struct = constructor.create_interface_block_type(
            InterfaceBlockKind::PushConstant,
            &member_types,
            &member_offsets,
            &member_poly_types,
        );

        let pc_ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::PushConstant, pc_struct);
        // Reuse the same push constant variable across entry points in the same module.
        // SPIR-V allows at most one PushConstant variable per module.
        let var_id = if let Some(existing) = constructor.push_constant_var {
            existing
        } else {
            let var_id =
                constructor.builder.variable(pc_ptr_type, None, spirv::StorageClass::PushConstant, None);
            constructor.push_constant_var = Some(var_id);
            var_id
        };
        interfaces.push(var_id);
        Some(var_id)
    } else {
        None
    };

    // Handle inputs
    let mut location = 0u32;
    // Uniform-bound inputs need their access-chain + load deferred until
    // after `begin_function`. Each entry here: (input.name, var_id, value_type).
    let mut uniform_loads: Vec<(String, spirv::Word, spirv::Word)> = Vec::new();
    for input in &entry.inputs {
        // Push constant inputs are handled separately above
        if input.push_constant.is_some() {
            continue;
        }

        let input_type = constructor.polytype_to_spirv(&input.ty);

        if let Some(IoDecoration::BuiltIn(builtin)) = &input.decoration {
            // WGSL's `@builtin(position)` is stage-aware (vertex-out vs
            // fragment-in), so the Wyn frontend lets either `position` or
            // `frag_coord` parse to `BuiltIn::Position`/`BuiltIn::FragCoord`
            // and trusts the backend to do the right thing for the stage.
            // SPIR-V's builtins are stage-specific: fragment inputs must
            // decorate as FragCoord, never Position (drivers silently
            // zero a Position-decorated fragment input).
            let stage_builtin = match (&entry.execution_model, builtin) {
                (ExecutionModel::Fragment, spirv::BuiltIn::Position) => spirv::BuiltIn::FragCoord,
                _ => *builtin,
            };
            // Reuse the module-level cached variable for the shared
            // compute builtins so the entry's interface doesn't end
            // up with two Input variables decorated with the same
            // BuiltIn — Vulkan rejects that (VUID-StandaloneSpirv-
            // OpEntryPoint-09658) even though earlier drivers
            // tolerated it.
            let cached = match stage_builtin {
                spirv::BuiltIn::GlobalInvocationId => constructor.global_invocation_id,
                spirv::BuiltIn::LocalInvocationId => constructor.local_invocation_id,
                spirv::BuiltIn::NumWorkgroups => constructor.num_workgroups,
                _ => None,
            };
            let var_id = if let Some(existing) = cached {
                existing
            } else {
                let ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Input, input_type);
                let new_var =
                    constructor.builder.variable(ptr_type, None, spirv::StorageClass::Input, None);
                constructor.builder.decorate(
                    new_var,
                    spirv::Decoration::BuiltIn,
                    [Operand::BuiltIn(stage_builtin)],
                );
                match stage_builtin {
                    spirv::BuiltIn::GlobalInvocationId => {
                        constructor.global_invocation_id = Some(new_var);
                    }
                    spirv::BuiltIn::LocalInvocationId => {
                        constructor.local_invocation_id = Some(new_var);
                    }
                    spirv::BuiltIn::NumWorkgroups => {
                        constructor.num_workgroups = Some(new_var);
                    }
                    _ => {}
                }
                new_var
            };
            constructor.env.insert(input.name.clone(), var_id);
            if !interfaces.contains(&var_id) {
                interfaces.push(var_id);
            }
        } else if let Some(br) = input.uniform_binding {
            // `#[uniform(set, binding)]` → Block-decorated `{value}` struct
            // in Uniform storage class; the helper caches the struct so
            // two params with the same value type don't double-decorate
            // (spirv-val rejects member-Offset applied twice).
            let block_struct = constructor.get_or_create_uniform_block_type(input_type);
            let ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Uniform, block_struct);
            let var_id = constructor.builder.variable(ptr_type, None, spirv::StorageClass::Uniform, None);
            constructor.builder.decorate(
                var_id,
                spirv::Decoration::DescriptorSet,
                [Operand::LiteralBit32(br.set)],
            );
            constructor.builder.decorate(
                var_id,
                spirv::Decoration::Binding,
                [Operand::LiteralBit32(br.binding)],
            );
            interfaces.push(var_id);
            uniform_loads.push((input.name.clone(), var_id, input_type));
        } else if let Some((br, format, access, _size)) = input.storage_image_binding {
            // `#[storage_image]` → opaque OpTypeImage in UniformConstant
            // storage. Pre-created before function bodies (see
            // `lower_ssa_program_impl`) so image ops inside SOAC-body functions
            // resolve the same global; the call here is idempotent.
            let var_id = constructor.create_storage_image(br, format, access);
            constructor.env.insert(input.name.clone(), var_id);
            interfaces.push(var_id);
        } else if let Some(br) = input.texture_binding.or(input.sampler_binding) {
            // `#[texture]` / `#[sampler]` → opaque handle in UniformConstant
            // storage, decorated DescriptorSet/Binding. Unlike a uniform
            // there is no Block struct: the var points straight at the
            // image/sampler type. The generic input-load loop below does a
            // plain `OpLoad` of this var (storage class is irrelevant to
            // OpLoad), yielding the image/sampler *object* the sample/fetch
            // ops consume — so we just stash the var in `env` by name here.
            let ptr_type =
                constructor.get_or_create_ptr_type(spirv::StorageClass::UniformConstant, input_type);
            let var_id =
                constructor.builder.variable(ptr_type, None, spirv::StorageClass::UniformConstant, None);
            constructor.builder.decorate(
                var_id,
                spirv::Decoration::DescriptorSet,
                [Operand::LiteralBit32(br.set)],
            );
            constructor.builder.decorate(
                var_id,
                spirv::Decoration::Binding,
                [Operand::LiteralBit32(br.binding)],
            );
            constructor.env.insert(input.name.clone(), var_id);
            interfaces.push(var_id);
        } else if let Some(br) = input.storage_binding {
            let var_id = constructor.create_storage_buffer(&input.ty, br.set, br.binding);
            // Mark input storage buffers as non-writable ONLY if no other
            // entry point writes to the same binding. In multi-entry modules
            // (e.g., reduce phase1 + phase2), the partials buffer is written
            // by phase1 and read by phase2 — it must stay writable.
            //
            // `create_storage_buffer` returns the same var for a shared
            // binding, so guard the decoration to fire once per var — two
            // entries reading the same never-written input would otherwise
            // decorate it `NonWritable` twice (spirv-val rejects).
            if !written_bindings.contains(&br) {
                constructor.builder.decorate_nonwritable_once(builder::VarId::new(var_id));
            }
            interfaces.push(var_id);
        } else {
            // Regular input with location
            let loc = input
                .decoration
                .as_ref()
                .and_then(|d| match d {
                    IoDecoration::Location(l) => Some(*l),
                    _ => None,
                })
                .unwrap_or_else(|| {
                    let l = location;
                    location += 1;
                    l
                });

            let ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Input, input_type);
            let var_id = constructor.builder.variable(ptr_type, None, spirv::StorageClass::Input, None);
            constructor.builder.decorate(var_id, spirv::Decoration::Location, [Operand::LiteralBit32(loc)]);
            constructor.env.insert(input.name.clone(), var_id);
            interfaces.push(var_id);
        }
    }

    // Handle outputs
    let mut output_vars = Vec::new();
    let mut output_location = 0u32;
    for output in &entry.outputs {
        if let Some(br) = output.storage_binding {
            let var_id = constructor.create_storage_buffer(&output.ty, br.set, br.binding);
            interfaces.push(var_id);
            // Don't add to output_vars - storage buffers are accessed differently
        } else if let Some(IoDecoration::BuiltIn(builtin)) = &output.decoration {
            let output_type = constructor.polytype_to_spirv(&output.ty);
            let ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Output, output_type);
            let var_id = constructor.builder.variable(ptr_type, None, spirv::StorageClass::Output, None);
            constructor.builder.decorate(var_id, spirv::Decoration::BuiltIn, [Operand::BuiltIn(*builtin)]);
            output_vars.push(var_id);
            interfaces.push(var_id);
        } else {
            let output_type = constructor.polytype_to_spirv(&output.ty);
            let loc = output
                .decoration
                .as_ref()
                .and_then(|d| match d {
                    IoDecoration::Location(l) => Some(*l),
                    _ => None,
                })
                .unwrap_or_else(|| {
                    let l = output_location;
                    output_location += 1;
                    l
                });

            let ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Output, output_type);
            let var_id = constructor.builder.variable(ptr_type, None, spirv::StorageClass::Output, None);
            constructor.builder.decorate(var_id, spirv::Decoration::Location, [Operand::LiteralBit32(loc)]);
            output_vars.push(var_id);
            interfaces.push(var_id);
        }
    }

    // Workgroup-shared arrays (phase2 tree reduce): create one module-scope
    // `OpVariable` in Workgroup storage per distinct id referenced by a
    // `StorageView(Workgroup{id, count})` in this body, and list it in the
    // entry interface (required by SPIR-V >= 1.4). Must precede `entry_point()`
    // consuming `interfaces`, hence the pre-scan here rather than lazily during
    // `ViewIndex` lowering.
    if is_compute {
        for (_, inst) in body.inner.insts.iter() {
            let InstKind::Op {
                tag: crate::op::OpTag::StorageView(crate::op::PureViewSource::Workgroup { id, count }),
                ..
            } = &inst.data
            else {
                continue;
            };
            if let Some(&(var, _)) = constructor.workgroup_vars.get(id) {
                if !interfaces.contains(&var) {
                    interfaces.push(var);
                }
                continue;
            }
            let result = inst.result.expect("StorageView(Workgroup) must have a result");
            let view_ty = body.get_value_type(result);
            // Workgroup-shared partial buffer: array-shaped for vector reduces
            // (`[]T` → elem is `T`), or scalar/vec/struct-shaped for single-
            // element reductions (the view itself IS the elem).
            let elem_ty = match crate::types::array_elem(view_ty) {
                Some(elem) => elem.clone(),
                None => view_ty.clone(),
            };
            // KNOWN LIMITATION (struct/array reduce accumulators): `elem_spirv`
            // is the *same* SPIR-V type id `polytype_to_spirv` hands the
            // partials storage buffer — and rspirv dedups type ids while SPIR-V
            // decorations are per-id. So when the accumulator is a struct/array
            // (e.g. the miner's `(u32, [8]u32)`), this Workgroup array's element
            // inherits the buffer's `Offset`/`ArrayStride` (explicit layout),
            // which Vulkan forbids on the Workgroup storage class. `spirv-val`
            // (and validation layers) flag it; AMD/typical drivers tolerate it
            // and run correctly. Scalar accumulators (no layout decorations) are
            // clean. A real fix is either SPV_KHR_workgroup_memory_explicit_layout
            // (Block-wrap this array, keep the decorated type) or an undecorated
            // shared type with component-wise value reconciliation at the
            // buffer<->shared boundary; deferred.
            let elem_spirv = constructor.polytype_to_spirv(&elem_ty);
            let count_const = constructor.const_u32(*count);
            let arr_ty = *constructor.builder.type_array(wspirv::TypeId::new(elem_spirv), count_const);
            let ptr_ty = constructor.get_or_create_ptr_type(spirv::StorageClass::Workgroup, arr_ty);
            let var = constructor.builder.variable(ptr_ty, None, spirv::StorageClass::Workgroup, None);
            constructor.workgroup_vars.insert(*id, (var, elem_spirv));
            interfaces.push(var);
        }
    }

    // Store interfaces for entry point declaration
    constructor.entry_point_interfaces.insert(entry.name.clone(), interfaces);

    // Set output variables for OutputPtr lowering
    constructor.current_entry_outputs = output_vars;

    // Begin void function for entry point — I/O is via variables, not params.
    let void_type = constructor.void_type;
    let param_names: Vec<&str> = Vec::new();
    let param_types: Vec<spirv::Word> = Vec::new();
    let (_, _, first_code_block) =
        constructor.begin_function(&entry.name, &param_names, &param_types, void_type)?;

    // Load push constant members via AccessChain from the push constant variable.
    if let Some(pc_var_id) = pc_var {
        for (member_idx, &(input_idx, _offset)) in pc_inputs.iter().enumerate() {
            let input = &entry.inputs[input_idx];
            let member_type = constructor.polytype_to_spirv(&input.ty);
            let member_ptr_type =
                constructor.get_or_create_ptr_type(spirv::StorageClass::PushConstant, member_type);
            let idx_const = constructor.const_u32(member_idx as u32);
            let access_chain =
                constructor.builder.access_chain(member_ptr_type, None, pc_var_id, [idx_const])?;
            let loaded = constructor.builder.load(member_type, None, access_chain, None, [])?;
            constructor.env.insert(input.name.clone(), loaded);
        }
    }

    // Load uniform members: each `#[uniform]` param is an OpVariable
    // pointing at a Block-decorated `{value_type}` struct in Uniform
    // storage. The body references the value, so AccessChain to
    // member 0 + Load and put the loaded value in env.
    for (name, var_id, value_type) in &uniform_loads {
        let member_ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Uniform, *value_type);
        let zero = constructor.const_i32(0);
        let access_chain = constructor.builder.access_chain(member_ptr_type, None, *var_id, [zero])?;
        let loaded = constructor.builder.load(*value_type, None, access_chain, None, [])?;
        constructor.env.insert(name.clone(), loaded);
    }

    // Load input values from their pointer variables.
    // Entry point inputs are SPIR-V Input variables (pointers), but the SSA body
    // expects loaded values. Load them now and update env with the loaded values.
    for input in &entry.inputs {
        // Skip storage buffers, push constants, and uniforms — each
        // uses a different access pattern handled above.
        if input.storage_binding.is_some()
            || input.push_constant.is_some()
            || input.uniform_binding.is_some()
        {
            continue;
        }
        // Storage-image variables have a format-aware `OpTypeImage`
        // built at the binding site (`storage_image_binding` branch
        // above) — the polytype-derived placeholder has Format=Unknown
        // and doesn't match the variable's pointee type. Rebuild with
        // the binding's format for the load result type.
        let input_type = if let Some((_, format, _, _)) = input.storage_image_binding {
            *constructor.builder.type_image(
                wspirv::TypeId::new(constructor.f32_type),
                spirv::Dim::Dim2D,
                0,
                0,
                0,
                2,
                storage_image_format_to_spirv(format),
                None,
            )
        } else {
            constructor.polytype_to_spirv(&input.ty)
        };
        if let Some(&var_id) = constructor.env.get(&input.name) {
            let loaded = constructor.builder.load(input_type, None, var_id, None, [])?;
            constructor.env.insert(input.name.clone(), loaded);
        }
    }

    // Lower the body — entry points are void functions, so SSA must
    // use OutputPtr+Store then ReturnUnit; a Return(value) terminator
    // will produce an error.
    lower::LowerCtx::new(constructor, body, true, entry.span, Vec::new(), first_code_block).lower()?;

    constructor.end_function()?;

    // Clear output variables
    constructor.current_entry_outputs.clear();

    Ok(())
}
