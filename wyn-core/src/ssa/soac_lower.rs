//! SOAC lowering pass: expands first-class SOAC instructions into explicit loops.
//!
//! This pass runs after SSA optimization and before backend lowering (SPIR-V/GLSL).
//! It transforms `InstKind::Soac(Soac::Map { .. })` into for-range loops,
//! `InstKind::Soac(Soac::Reduce { .. })` into accumulation loops, etc.

use std::collections::HashMap;

use crate::ast::TypeName;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::soa_helpers::{
    extract_array_size, is_soa_tuple, soa_array_with, soa_elem_type, soa_index, soa_length, soa_uninit,
};
use crate::types::TypeExt;

/// Maximum fixed-size array length for which map will be unrolled
/// instead of lowered to a for-range loop.
const MAP_UNROLL_THRESHOLD: usize = 16;
use crate::ssa::types::Program;
use crate::ssa::types::{
    BlockId, EffectToken, FuncBody, InstKind, Soac, Terminator, TerminatorExt, ValueId,
};
use polytype::Type;

/// Lower all SOAC instructions in the program to explicit loops.
pub fn lower_soacs(mut program: Program) -> Program {
    for func in &mut program.functions {
        if has_soac_instructions(&func.body) {
            func.body = lower_func_body(&func.body);
        }
    }
    for entry in &mut program.entry_points {
        if has_soac_instructions(&entry.body) {
            entry.body = lower_func_body(&entry.body);
        }
    }
    program
}

/// Check if a function body contains any SOAC instructions.
fn has_soac_instructions(body: &FuncBody) -> bool {
    body.inner.insts.values().any(|inst| matches!(inst.data, InstKind::Soac(_)))
}

/// Rebuild a function body, expanding SOAC instructions into loops.
fn lower_func_body(old_body: &FuncBody) -> FuncBody {
    let params: Vec<(Type<TypeName>, String)> =
        old_body.params.iter().map(|(_, ty, name)| (ty.clone(), name.clone())).collect();

    let mut builder = FuncBuilder::new(params, old_body.return_ty.clone());

    let mut value_map: HashMap<ValueId, ValueId> = HashMap::new();
    let mut effect_map: HashMap<EffectToken, EffectToken> = HashMap::new();
    effect_map.insert(old_body.entry_effect(), builder.entry_effect());

    // Map function params
    for (i, (old_val, _, _)) in old_body.params.iter().enumerate() {
        value_map.insert(*old_val, builder.get_param(i));
    }

    let mut block_map: HashMap<BlockId, BlockId> = HashMap::new();
    let old_entry = old_body.entry_block();
    block_map.insert(old_entry, builder.entry());

    // Pre-create all non-entry blocks with their parameters.
    // Skip dead blocks (empty insts + Unreachable terminator) — they can't be
    // branched to, and Unreachable doubles as the builder's "unterminated" sentinel
    // so finish() would reject them.
    for (bid, block) in &old_body.inner.blocks {
        if bid == old_entry {
            // Entry block params (if any)
            for &param in &block.params {
                let ty = old_body.inner.value_type(param).clone();
                let new_val = builder.add_block_param(builder.entry(), ty);
                value_map.insert(param, new_val);
            }
            continue;
        }

        // Skip dead blocks entirely
        if block.insts.is_empty()
            && block.params.is_empty()
            && matches!(block.term, Terminator::Unreachable)
        {
            continue;
        }

        let param_types: Vec<Type<TypeName>> =
            block.params.iter().map(|&p| old_body.inner.value_type(p).clone()).collect();
        let (new_block_id, new_param_vals) = builder.create_block_with_params(param_types);
        block_map.insert(bid, new_block_id);

        for (&old_param, &new_val) in block.params.iter().zip(new_param_vals.iter()) {
            value_map.insert(old_param, new_val);
        }
    }

    // Process each block
    for (bid, block) in &old_body.inner.blocks {
        // Skip dead blocks (not pre-created, nothing branches to them)
        if block.insts.is_empty()
            && block.params.is_empty()
            && matches!(block.term, Terminator::Unreachable)
        {
            continue;
        }

        let new_block_id = block_map[&bid];
        if bid != old_entry {
            builder.switch_to_block_unchecked(new_block_id);
        }

        // Defer control header
        let deferred_control =
            old_body.control_headers.get(&bid).map(|ctrl| ctrl.remap(&|b| block_map[&b]));

        // Process instructions
        for &inst_id in &block.insts {
            let inst = &old_body.inner.insts[inst_id];

            match &inst.data {
                InstKind::Soac(soac) => {
                    let result_ty = inst
                        .result
                        .map(|r| old_body.inner.value_type(r).clone())
                        .unwrap_or_else(|| Type::Constructed(TypeName::Unit, vec![]));
                    let new_val = expand_soac(&mut builder, soac, result_ty, &value_map);
                    if let (Some(old_result), Some(new_result)) = (inst.result, new_val) {
                        value_map.insert(old_result, new_result);
                    }
                }
                _ => {
                    let new_kind = inst.data.remap(&|v| value_map[&v]);

                    // Remap effects from InstNode.effects
                    let new_effects = inst.effects.map(|(ein, _eout)| {
                        let mapped_in = *effect_map.get(&ein).unwrap_or(&ein);
                        let mapped_out = builder.alloc_effect();
                        (mapped_in, mapped_out)
                    });
                    if let (Some((_, old_out)), Some((_, new_out))) = (inst.effects, new_effects) {
                        effect_map.insert(old_out, new_out);
                    }

                    if let Some(result) = inst.result {
                        let result_ty = old_body.inner.value_type(result).clone();
                        let new_val = builder
                            .push_inst_with_effects(new_kind, result_ty, new_effects)
                            .expect("BUG: failed to push remapped instruction");
                        value_map.insert(result, new_val);
                    } else {
                        builder
                            .push_void_inst_with_effects(new_kind, new_effects)
                            .expect("BUG: failed to push remapped void instruction");
                    }
                }
            }
        }

        // Set control header on the block that will carry the terminator
        if let Some(ctrl) = deferred_control {
            if let Some(current) = builder.current_block() {
                builder.set_control_header(current, ctrl);
            }
        }

        // Remap and set terminator
        let new_term = block.term.remap(&|v| value_map[&v], &|b| block_map[&b]);
        builder.terminate(new_term).ok();
    }

    builder.finish().expect("BUG: failed to finish rebuilt function body")
}

/// Expand a SOAC instruction into explicit loop(s) using the builder.
fn expand_soac(
    builder: &mut FuncBuilder,
    soac: &Soac,
    result_ty: Type<TypeName>,
    value_map: &HashMap<ValueId, ValueId>,
) -> Option<ValueId> {
    match soac {
        Soac::Map {
            func,
            inputs,
            captures,
            zipped,
            input_array_types,
            input_elem_types,
            output_elem_type,
            zipped_param_type,
        } => expand_map(
            builder,
            func,
            &remap_values(inputs, value_map),
            &remap_values(captures, value_map),
            *zipped,
            input_array_types,
            input_elem_types,
            output_elem_type,
            zipped_param_type.as_ref(),
            result_ty,
        ),
        Soac::Reduce {
            func,
            input,
            init,
            captures,
            input_array_type,
            input_elem_type,
        } => expand_reduce(
            builder,
            func,
            value_map[input],
            value_map[init],
            &remap_values(captures, value_map),
            input_array_type,
            input_elem_type,
            result_ty,
        ),
        Soac::Scan { .. } => {
            panic!("internal compiler error: Scan SOAC lowering not yet implemented")
        }
    }
}

fn remap_values(values: &[ValueId], map: &HashMap<ValueId, ValueId>) -> Vec<ValueId> {
    values.iter().map(|v| map[v]).collect()
}

/// Expand a Map SOAC into a for-range loop.
fn expand_map(
    builder: &mut FuncBuilder,
    func: &str,
    inputs: &[ValueId],
    captures: &[ValueId],
    zipped: bool,
    input_array_types: &[Type<TypeName>],
    input_elem_types: &[Type<TypeName>],
    output_elem_type: &Type<TypeName>,
    zipped_param_type: Option<&Type<TypeName>>,
    result_ty: Type<TypeName>,
) -> Option<ValueId> {
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

    let first_input_ty = &input_array_types[0];
    let array_size = extract_array_size(first_input_ty);

    // For small fixed-size arrays with non-SoA results, unroll instead of a loop.
    // SoA results need a transpose which is worse than the loop accumulator.
    if let Some(n) = array_size {
        if n <= MAP_UNROLL_THRESHOLD && is_soa_tuple(&result_ty).is_none() {
            return expand_map_unrolled(
                builder,
                n,
                func,
                inputs,
                captures,
                zipped,
                input_array_types,
                input_elem_types,
                output_elem_type,
                zipped_param_type,
                result_ty,
            );
        }
    }

    let len = match array_size {
        Some(n) => builder.push_int(&n.to_string(), i32_ty.clone()).ok()?,
        None => soa_length(builder, inputs[0], first_input_ty).ok()?,
    };

    let init_array = soa_uninit(builder, &result_ty).ok()?;

    let loop_blocks = builder.create_for_range_loop(result_ty.clone());

    let zero = builder.push_int("0", i32_ty.clone()).ok()?;
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![init_array, zero],
        })
        .ok()?;

    // Header
    builder.switch_to_block(loop_blocks.header).ok()?;
    let cond = builder.push_binop("<", loop_blocks.index, len, bool_ty).ok()?;
    builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: loop_blocks.body,
            then_args: vec![],
            else_target: loop_blocks.exit,
            else_args: vec![loop_blocks.acc],
        })
        .ok()?;

    // Body
    builder.switch_to_block(loop_blocks.body).ok()?;

    let mut input_elems: Vec<ValueId> = Vec::with_capacity(inputs.len());
    for (i, &arr) in inputs.iter().enumerate() {
        let elem = soa_index(
            builder,
            arr,
            loop_blocks.index,
            &input_array_types[i],
            &input_elem_types[i],
        )
        .ok()?;
        input_elems.push(elem);
    }

    let mut call_args: Vec<ValueId> = if zipped {
        let zipped_ty = zipped_param_type.expect("BUG: zipped map without param type").clone();
        let zipped_val = builder.push_tuple(input_elems, zipped_ty).ok()?;
        vec![zipped_val]
    } else {
        input_elems
    };
    call_args.extend(captures.iter().copied());

    let output_elem = builder.push_call(func, call_args, output_elem_type.clone()).ok()?;

    let new_arr = soa_array_with(
        builder,
        loop_blocks.acc,
        loop_blocks.index,
        output_elem,
        &result_ty,
    )
    .ok()?;

    let one = builder.push_int("1", i32_ty.clone()).ok()?;
    let next_i = builder.push_binop("+", loop_blocks.index, one, i32_ty).ok()?;
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![new_arr, next_i],
        })
        .ok()?;

    // Exit
    builder.switch_to_block(loop_blocks.exit).ok()?;

    Some(loop_blocks.result)
}

/// Expand a Map SOAC by unrolling: emit N individual calls and construct
/// the result directly, avoiding the loop/accumulator/array_with overhead.
fn expand_map_unrolled(
    builder: &mut FuncBuilder,
    n: usize,
    func: &str,
    inputs: &[ValueId],
    captures: &[ValueId],
    zipped: bool,
    input_array_types: &[Type<TypeName>],
    input_elem_types: &[Type<TypeName>],
    output_elem_type: &Type<TypeName>,
    zipped_param_type: Option<&Type<TypeName>>,
    result_ty: Type<TypeName>,
) -> Option<ValueId> {
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

    // Phase 1: Emit N calls to the map function.
    let mut call_results: Vec<ValueId> = Vec::with_capacity(n);

    for i in 0..n {
        let const_i = builder.push_int(&i.to_string(), i32_ty.clone()).ok()?;

        // Extract input elements for this iteration
        let mut input_elems: Vec<ValueId> = Vec::with_capacity(inputs.len());
        for (j, &arr) in inputs.iter().enumerate() {
            let elem =
                soa_index(builder, arr, const_i, &input_array_types[j], &input_elem_types[j]).ok()?;
            input_elems.push(elem);
        }

        // Build call arguments: zip if needed, then append captures
        let mut call_args: Vec<ValueId> = if zipped {
            let zipped_ty = zipped_param_type.expect("BUG: zipped map without param type").clone();
            let zipped_val = builder.push_tuple(input_elems, zipped_ty).ok()?;
            vec![zipped_val]
        } else {
            input_elems
        };
        call_args.extend(captures.iter().copied());

        let result = builder.push_call(func, call_args, output_elem_type.clone()).ok()?;
        call_results.push(result);
    }

    // Phase 2: Construct the result.
    if is_soa_tuple(&result_ty).is_some() {
        // SoA result: transpose N element-tuples into M component arrays
        soa_pack_unrolled(builder, &call_results, &result_ty)
    } else {
        // Plain array: pack N results directly
        builder.push_array_lit(call_results, result_ty).ok()
    }
}

/// Transpose N element values into an SoA tuple.
///
/// Given N values each of element type (A, B, C) and SoA result type
/// ([N]A, [N]B, [N]C): project field j from each value, pack into [N]Tj
/// array, then combine arrays into the SoA tuple.
fn soa_pack_unrolled(
    builder: &mut FuncBuilder,
    values: &[ValueId],
    soa_ty: &Type<TypeName>,
) -> Option<ValueId> {
    let components = is_soa_tuple(soa_ty)?;
    let mut packed: Vec<ValueId> = Vec::with_capacity(components.len());

    for (field_idx, comp_ty) in components.iter().enumerate() {
        // Determine the element type for this component
        let field_elem_ty = match comp_ty {
            ty if ty.is_array() => ty.elem_type().expect("Array has elem").clone(),
            ty if is_soa_tuple(ty).is_some() => soa_elem_type(ty),
            _ => comp_ty.clone(),
        };

        // Project this field from each of the N values
        let mut field_values: Vec<ValueId> = Vec::with_capacity(values.len());
        for &val in values {
            let projected = builder.push_project(val, field_idx as u32, field_elem_ty.clone()).ok()?;
            field_values.push(projected);
        }

        if is_soa_tuple(comp_ty).is_some() {
            // Nested SoA: recurse
            let nested = soa_pack_unrolled(builder, &field_values, comp_ty)?;
            packed.push(nested);
        } else {
            // Plain array component
            let arr = builder.push_array_lit(field_values, comp_ty.clone()).ok()?;
            packed.push(arr);
        }
    }

    builder.push_tuple(packed, soa_ty.clone()).ok()
}

/// Expand a Reduce SOAC into a for-range loop.
fn expand_reduce(
    builder: &mut FuncBuilder,
    func: &str,
    arr_value: ValueId,
    init_value: ValueId,
    captures: &[ValueId],
    input_array_type: &Type<TypeName>,
    input_elem_type: &Type<TypeName>,
    result_ty: Type<TypeName>,
) -> Option<ValueId> {
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let acc_ty = result_ty.clone();

    let len = soa_length(builder, arr_value, input_array_type).ok()?;

    let loop_blocks = builder.create_for_range_loop(acc_ty.clone());

    let zero = builder.push_int("0", i32_ty.clone()).ok()?;
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![init_value, zero],
        })
        .ok()?;

    // Header
    builder.switch_to_block(loop_blocks.header).ok()?;
    let cond = builder.push_binop("<", loop_blocks.index, len, bool_ty).ok()?;
    builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: loop_blocks.body,
            then_args: vec![],
            else_target: loop_blocks.exit,
            else_args: vec![loop_blocks.acc],
        })
        .ok()?;

    // Body
    builder.switch_to_block(loop_blocks.body).ok()?;

    let elem = soa_index(
        builder,
        arr_value,
        loop_blocks.index,
        input_array_type,
        input_elem_type,
    )
    .ok()?;

    let mut call_args = vec![loop_blocks.acc, elem];
    call_args.extend(captures.iter().copied());
    let new_acc = builder.push_call(func, call_args, acc_ty).ok()?;

    let one = builder.push_int("1", i32_ty.clone()).ok()?;
    let next_i = builder.push_binop("+", loop_blocks.index, one, i32_ty).ok()?;
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![new_acc, next_i],
        })
        .ok()?;

    // Exit
    builder.switch_to_block(loop_blocks.exit).ok()?;

    Some(loop_blocks.result)
}
