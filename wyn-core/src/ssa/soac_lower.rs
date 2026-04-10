//! SOAC lowering pass: expands first-class SOAC instructions into explicit loops.
//!
//! This pass runs after SSA optimization and before backend lowering (SPIR-V/GLSL).
//! It transforms `InstKind::Soac(Soac::Map { .. })` into for-range loops,
//! `InstKind::Soac(Soac::Reduce { .. })` into accumulation loops, etc.

use std::collections::{HashMap, HashSet};

use crate::ast::TypeName;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::soa_helpers::{
    extract_array_size, is_soa_tuple, soa_array_with, soa_elem_type, soa_index, soa_length, soa_uninit,
};
use crate::types::TypeExt;

/// Maximum fixed-size array length for which map will be unrolled
/// instead of lowered to a for-range loop.
const MAP_UNROLL_THRESHOLD: usize = 16;
use crate::ssa::soac_analysis::ArrayProvenance;
use crate::ssa::types::Program;
use crate::ssa::types::{
    BlockId, EffectToken, EntryInput, EntryOutput, EntryPoint, ExecutionModel, FuncBody, InstKind, Soac,
    Terminator, TerminatorExt, ValueId, ValueRef,
};
use crate::types::is_virtual_array;
use polytype::Type;

// =============================================================================
// SOAC lowering context (used by storage-aware loop builders + strategies)
// =============================================================================

/// Context for SOAC lowering. Wraps a FuncBuilder with cached types
/// and a reference to the entry point (for capture remapping).
pub struct LowerCtx<'a> {
    pub builder: FuncBuilder,
    pub entry: &'a EntryPoint,
    pub u32_ty: Type<TypeName>,
    pub i32_ty: Type<TypeName>,
    pub bool_ty: Type<TypeName>,
    pub unit_ty: Type<TypeName>,
}

impl<'a> LowerCtx<'a> {
    pub fn new(builder: FuncBuilder, entry: &'a EntryPoint) -> Self {
        Self {
            builder,
            entry,
            u32_ty: Type::Constructed(TypeName::UInt(32), vec![]),
            i32_ty: Type::Constructed(TypeName::Int(32), vec![]),
            bool_ty: Type::Constructed(TypeName::Bool, vec![]),
            unit_ty: Type::Constructed(TypeName::Unit, vec![]),
        }
    }

    pub fn push_int(&mut self, value: &str) -> Option<ValueId> {
        self.builder.push_int(value, self.u32_ty.clone()).ok()
    }

    pub fn push_i32(&mut self, value: &str) -> Option<ValueId> {
        self.builder.push_int(value, self.i32_ty.clone()).ok()
    }

    pub fn push_binop(
        &mut self,
        op: &str,
        lhs: ValueId,
        rhs: ValueId,
        ty: Type<TypeName>,
    ) -> Option<ValueId> {
        self.builder.push_binop(op, lhs, rhs, ty).ok()
    }

    pub fn push_intrinsic(
        &mut self,
        name: &str,
        args: Vec<ValueId>,
        ty: Type<TypeName>,
    ) -> Option<ValueId> {
        self.builder.push_intrinsic(name, args, ty).ok()
    }

    pub fn push_call(&mut self, func: &str, args: Vec<ValueId>, ty: Type<TypeName>) -> Option<ValueId> {
        self.builder.push_call(func, args, ty).ok()
    }

    pub fn push_inst(&mut self, kind: InstKind, ty: Type<TypeName>) -> Option<ValueId> {
        self.builder.push_inst(kind, ty).ok()
    }

    pub fn entry_effect(&self) -> EffectToken {
        self.builder.entry_effect()
    }

    pub fn finish(self) -> Option<FuncBody> {
        self.builder.finish().ok()
    }

    pub fn compute_thread_chunk(
        &mut self,
        input_len: ValueId,
        total_threads: u32,
    ) -> Option<(ValueId, ValueId, ValueId)> {
        let thread_id = self.push_intrinsic("_w_intrinsic_thread_id", vec![], self.u32_ty.clone())?;
        let total_threads_val = self.push_int(&total_threads.to_string())?;
        let threads_minus_1 = self.push_int(&(total_threads - 1).to_string())?;

        let len_plus = self.push_binop("+", input_len, threads_minus_1, self.u32_ty.clone())?;
        let chunk_size = self.push_binop("/", len_plus, total_threads_val, self.u32_ty.clone())?;
        let chunk_start = self.push_binop("*", thread_id, chunk_size, self.u32_ty.clone())?;
        let start_plus_size = self.push_binop("+", chunk_start, chunk_size, self.u32_ty.clone())?;
        let chunk_end = self.push_call("u32.min", vec![start_plus_size, input_len], self.u32_ty.clone())?;

        Some((thread_id, chunk_start, chunk_end))
    }

    pub fn remap_captures(&mut self, captures: &[ValueId]) -> Option<Vec<ValueId>> {
        let body = self.entry.body.clone();
        let mut memo: HashMap<ValueId, ValueId> = body
            .params
            .iter()
            .enumerate()
            .map(|(i, (src, _, _))| (*src, self.builder.get_param(i)))
            .collect();

        let mut result = Vec::new();
        for &capture in captures {
            let new_val = remap_value(&body, capture, &mut self.builder, &mut memo)?;
            result.push(new_val);
        }
        Some(result)
    }
}

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

    // Process blocks in RPO so that values are defined before use.
    let rpo = compute_rpo(old_body);
    for bid in &rpo {
        let bid = *bid;
        let block = &old_body.inner.blocks[bid];
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
            input_array_types,
            input_elem_types,
            output_elem_type,
        } => expand_map(
            builder,
            func,
            &remap_values(inputs, value_map),
            &remap_values(captures, value_map),
            input_array_types,
            input_elem_types,
            output_elem_type,
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
        Soac::Scan {
            func,
            input,
            init,
            captures,
            input_array_type,
            input_elem_type,
        } => expand_scan(
            builder,
            func,
            value_map[input],
            value_map[init],
            &remap_values(captures, value_map),
            input_array_type,
            input_elem_type,
            result_ty,
        ),
        Soac::Redomap {
            func,
            inputs,
            init,
            captures,
            input_array_types,
            input_elem_types,
            ..
        } => expand_redomap(
            builder,
            func,
            &remap_values(inputs, value_map),
            value_map[init],
            &remap_values(captures, value_map),
            input_array_types,
            input_elem_types,
            result_ty,
        ),
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
    input_array_types: &[Type<TypeName>],
    input_elem_types: &[Type<TypeName>],
    output_elem_type: &Type<TypeName>,
    result_ty: Type<TypeName>,
) -> Option<ValueId> {
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

    let first_input_ty = &input_array_types[0];
    let array_size = extract_array_size(first_input_ty);

    // For small fixed-size arrays with non-SoA results, unroll instead of a loop.
    if let Some(n) = array_size {
        if n <= MAP_UNROLL_THRESHOLD && is_soa_tuple(&result_ty).is_none() {
            return expand_map_unrolled(
                builder,
                n,
                func,
                inputs,
                captures,
                input_array_types,
                input_elem_types,
                output_elem_type,
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

    let mut call_args = input_elems;
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
    input_array_types: &[Type<TypeName>],
    input_elem_types: &[Type<TypeName>],
    output_elem_type: &Type<TypeName>,
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

        let mut call_args = input_elems;
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

/// Expand a Scan SOAC into a sequential loop that accumulates and stores each
/// intermediate result. Inclusive scan: out[i] = op(out[i-1], input[i]).
fn expand_scan(
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
    let acc_ty = input_elem_type.clone();

    let len = soa_length(builder, arr_value, input_array_type).ok()?;
    let init_array = soa_uninit(builder, &result_ty).ok()?;

    // Loop with output_array as the loop accumulator (like map),
    // plus a scalar accumulator as an extra block param.
    let loop_blocks = builder.create_for_range_loop(result_ty.clone());
    let scalar_acc = builder.add_block_param(loop_blocks.header, acc_ty.clone());

    let zero = builder.push_int("0", i32_ty.clone()).ok()?;
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![init_array, zero, init_value],
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

    // Body: new_acc = op(scalar_acc, elem); out[i] = new_acc
    builder.switch_to_block(loop_blocks.body).ok()?;

    let elem = soa_index(
        builder,
        arr_value,
        loop_blocks.index,
        input_array_type,
        input_elem_type,
    )
    .ok()?;

    let mut call_args = vec![scalar_acc, elem];
    call_args.extend(captures.iter().copied());
    let new_acc = builder.push_call(func, call_args, acc_ty).ok()?;

    let new_arr = soa_array_with(builder, loop_blocks.acc, loop_blocks.index, new_acc, &result_ty).ok()?;

    let one = builder.push_int("1", i32_ty.clone()).ok()?;
    let next_i = builder.push_binop("+", loop_blocks.index, one, i32_ty).ok()?;
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![new_arr, next_i, new_acc],
        })
        .ok()?;

    // Exit
    builder.switch_to_block(loop_blocks.exit).ok()?;

    Some(loop_blocks.result)
}

/// Expand a Redomap SOAC into a single for-range reduction loop over
/// multiple input arrays, without materializing intermediate arrays.
///
/// The combined operator takes `(acc, x1, ..., xn, captures...)` and returns
/// the new accumulator.
fn expand_redomap(
    builder: &mut FuncBuilder,
    func: &str,
    inputs: &[ValueId],
    init_value: ValueId,
    captures: &[ValueId],
    input_array_types: &[Type<TypeName>],
    input_elem_types: &[Type<TypeName>],
    result_ty: Type<TypeName>,
) -> Option<ValueId> {
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let acc_ty = result_ty.clone();

    // Compute length from first input array
    let first_input_ty = &input_array_types[0];
    let len = match extract_array_size(first_input_ty) {
        Some(n) => builder.push_int(&n.to_string(), i32_ty.clone()).ok()?,
        None => soa_length(builder, inputs[0], first_input_ty).ok()?,
    };

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

    // Body: index all input arrays, call func(acc, elem0, ..., elemN, captures...)
    builder.switch_to_block(loop_blocks.body).ok()?;

    let mut call_args = vec![loop_blocks.acc];
    for (i, &arr) in inputs.iter().enumerate() {
        let elem = soa_index(
            builder,
            arr,
            loop_blocks.index,
            &input_array_types[i],
            &input_elem_types[i],
        )
        .ok()?;
        call_args.push(elem);
    }
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

/// Compute reverse post-order of blocks reachable from the entry.
fn compute_rpo(body: &FuncBody) -> Vec<BlockId> {
    let entry = body.entry_block();
    let mut visited = HashSet::new();
    let mut post_order = Vec::new();
    rpo_visit(body, entry, &mut visited, &mut post_order);
    post_order.reverse();
    post_order
}

fn rpo_visit(body: &FuncBody, bid: BlockId, visited: &mut HashSet<BlockId>, post_order: &mut Vec<BlockId>) {
    if !visited.insert(bid) {
        return;
    }
    let block = &body.inner.blocks[bid];
    for succ in block.term.successors() {
        rpo_visit(body, succ, visited, post_order);
    }
    post_order.push(bid);
}
/// Build the parallel map loop body: setup strategies, chunk work, loop, call map function, store.
fn build_map_body<I: InputStrategy, O: OutputStrategy>(
    ctx: &mut LowerCtx,
    input_strategy: &mut I,
    output_strategy: &O,
    map_function: &str,
    captures: &[ValueId],
    output_elem_type: &Type<TypeName>,
    total_threads: u32,
) -> Option<()> {
    // 1. Setup input and output strategies (resources created once)
    let (input_handle, input_len, _input_elem_ty) = input_strategy.setup(ctx)?;
    let output_handle = output_strategy.setup(ctx, output_elem_type)?;

    // 2. Get thread ID and calculate chunk bounds
    let (_thread_id, chunk_start, chunk_end) = ctx.compute_thread_chunk(input_len, total_threads)?;

    // 3. Create loop structure
    let u32_ty = ctx.u32_ty.clone();
    let bool_ty = ctx.bool_ty.clone();

    let (header, header_params) = ctx.builder.create_block_with_params(vec![u32_ty.clone()]);
    let loop_index = header_params[0];
    let body_block = ctx.builder.create_block();
    let exit_block = ctx.builder.create_block();
    ctx.builder.mark_loop_header(header, exit_block, body_block).ok()?;

    // Branch to header with initial index
    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![chunk_start],
        })
        .ok()?;

    // Header: check i < chunk_end
    ctx.builder.switch_to_block(header).ok()?;
    let cond = ctx.builder.push_binop("<", loop_index, chunk_end, bool_ty.clone()).ok()?;
    ctx.builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: body_block,
            then_args: vec![],
            else_target: exit_block,
            else_args: vec![],
        })
        .ok()?;

    // 4. Loop body: get element, apply function, store result
    ctx.builder.switch_to_block(body_block).ok()?;

    let input_elem = input_strategy.get_element(ctx, input_handle, loop_index)?;

    // Build call args: element first, then remapped captures
    let mut call_args = vec![input_elem];
    let remapped_captures = ctx.remap_captures(captures)?;
    call_args.extend(remapped_captures);

    // Apply map function with the correct return type
    let result_elem = ctx.push_call(map_function, call_args, output_elem_type.clone())?;

    // Store result
    output_strategy.store_result(ctx, output_handle, loop_index, result_elem, output_elem_type)?;

    // Increment index and branch back to header
    let one = ctx.push_int("1")?;
    let next_i = ctx.push_binop("+", loop_index, one, u32_ty.clone())?;
    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![next_i],
        })
        .ok()?;

    // 5. Exit: return unit
    ctx.builder.switch_to_block(exit_block).ok()?;
    ctx.builder.terminate(Terminator::Return(None)).ok()?;

    Some(())
}

// =============================================================================
// Reduce Parallelization (2 dispatches)
// =============================================================================

/// Parallelize a reduce SOAC into two entry points:
/// 1. `{name}_phase1_chunks` — each thread reduces its chunk → partials[thread_id]
/// 2. `{name}_phase2_combine` — single thread combines partials → result[0]
///
/// Returns the two new entry points (replacing the original).
fn build_reduce_phase1(
    entry: &EntryPoint,
    source: &ArrayProvenance,
    reduce_function: &str,
    init: ValueId,
    captures: &[ValueId],
    elem_type: &Type<TypeName>,
    total_threads: u32,
    partials_binding: (u32, u32),
    local_size: (u32, u32, u32),
) -> Option<EntryPoint> {
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    let params: Vec<(Type<TypeName>, String)> =
        entry.inputs.iter().map(|i| (i.ty.clone(), i.name.clone())).collect();
    let builder = FuncBuilder::new(params, unit_ty.clone());
    let mut ctx = LowerCtx::new(builder, entry);

    // Setup input based on provenance
    let (input_len, input_strategy_data) = match source {
        ArrayProvenance::EntryStorage {
            param_index,
            storage_binding,
            ..
        } => {
            let mut input = StorageInput::new(*param_index, *storage_binding);
            let (handle, len, _elem_ty) = input.setup(&mut ctx)?;
            (len, ReduceInputData::Storage { input, handle })
        }
        ArrayProvenance::Range { value } => {
            let mut input = RangeInput::new(*value, &entry.body)?;
            let (handle, len, _elem_ty) = input.setup(&mut ctx)?;
            (len, ReduceInputData::Range { input, handle })
        }
        ArrayProvenance::Unknown => return None,
    };

    // Setup partials output buffer
    let partials_output = StorageOutput::new(partials_binding.0, partials_binding.1);
    let partials_view = partials_output.setup(&mut ctx, elem_type)?;

    // Get thread ID and chunk bounds
    let (thread_id, chunk_start, chunk_end) = ctx.compute_thread_chunk(input_len, total_threads)?;

    // Remap init value and captures from original entry body
    let remapped_init = remap_entry_value(&mut ctx, init)?;
    let remapped_captures = ctx.remap_captures(captures)?;

    // Build reduction loop: acc = init; for i in chunk_start..chunk_end: acc = f(acc, elem)
    let u32_ty = ctx.u32_ty.clone();
    let bool_ty = ctx.bool_ty.clone();

    let (header, header_params) =
        ctx.builder.create_block_with_params(vec![elem_type.clone(), u32_ty.clone()]);
    let acc = header_params[0];
    let loop_index = header_params[1];
    let body_block = ctx.builder.create_block();
    let (exit_block, exit_params) = ctx.builder.create_block_with_params(vec![elem_type.clone()]);
    let final_acc = exit_params[0];
    ctx.builder.mark_loop_header(header, exit_block, body_block).ok()?;

    // Branch to header with (init, chunk_start)
    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![remapped_init, chunk_start],
        })
        .ok()?;

    // Header: check i < chunk_end
    ctx.builder.switch_to_block(header).ok()?;
    let cond = ctx.builder.push_binop("<", loop_index, chunk_end, bool_ty).ok()?;
    ctx.builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: body_block,
            then_args: vec![],
            else_target: exit_block,
            else_args: vec![acc],
        })
        .ok()?;

    // Body: get element, call reduce function
    ctx.builder.switch_to_block(body_block).ok()?;

    let input_elem = match &input_strategy_data {
        ReduceInputData::Storage { input, handle } => input.get_element(&mut ctx, *handle, loop_index)?,
        ReduceInputData::Range { input, handle } => input.get_element(&mut ctx, *handle, loop_index)?,
    };

    // call_args: (acc, elem, captures...)
    let mut call_args = vec![acc, input_elem];
    call_args.extend(remapped_captures.iter().copied());
    let new_acc = ctx.push_call(reduce_function, call_args, elem_type.clone())?;

    let one = ctx.push_int("1")?;
    let next_i = ctx.push_binop("+", loop_index, one, u32_ty.clone())?;
    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![new_acc, next_i],
        })
        .ok()?;

    // Exit: store final_acc to partials[thread_id]
    ctx.builder.switch_to_block(exit_block).ok()?;
    partials_output.store_result(&mut ctx, partials_view, thread_id, final_acc, elem_type)?;
    ctx.builder.terminate(Terminator::Return(None)).ok()?;

    let body = ctx.finish()?;

    // Build the partials output entry output
    let array_view_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_type.clone(),
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    );

    Some(EntryPoint {
        name: format!("{}_phase1_chunks", entry.name),
        body,
        execution_model: ExecutionModel::Compute { local_size },
        inputs: entry.inputs.clone(),
        outputs: vec![EntryOutput {
            ty: array_view_ty,
            decoration: None,
            storage_binding: Some(partials_binding),
        }],
        span: entry.span,
    })
}

/// Build Phase 2 of reduce: single thread combines partial results.
fn build_reduce_phase2(
    entry: &EntryPoint,
    reduce_function: &str,
    init: ValueId,
    captures: &[ValueId],
    elem_type: &Type<TypeName>,
    total_threads: u32,
    partials_binding: (u32, u32),
    result_binding: (u32, u32),
) -> Option<EntryPoint> {
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

    // Phase 2 has no entry-level params (it reads from partials buffer directly).
    // But we need the function referenced by reduce_function to exist in the program,
    // and we need to remap captures. Captures may reference entry params, so
    // Phase 2 needs the same input params as the original entry.
    let params: Vec<(Type<TypeName>, String)> =
        entry.inputs.iter().map(|i| (i.ty.clone(), i.name.clone())).collect();
    let builder = FuncBuilder::new(params, unit_ty.clone());
    let mut ctx = LowerCtx::new(builder, entry);

    // Remap init and captures
    let remapped_init = remap_entry_value(&mut ctx, init)?;
    let remapped_captures = ctx.remap_captures(captures)?;

    // Setup partials input buffer (read from) — emit storage view directly
    let partials_view_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_type.clone(),
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    );
    let partials_view = ctx
        .builder
        .emit_storage_view(partials_binding.0, partials_binding.1, partials_view_ty.clone())
        .ok()?;

    // Setup result output buffer
    let result_output = StorageOutput::new(result_binding.0, result_binding.1);
    let result_view = result_output.setup(&mut ctx, elem_type)?;

    // Loop: acc = init; for t in 0..total_threads: acc = f(acc, partials[t])
    let total = ctx.push_int(&total_threads.to_string())?;
    let zero = ctx.push_int("0")?;

    let (header, header_params) =
        ctx.builder.create_block_with_params(vec![elem_type.clone(), u32_ty.clone()]);
    let acc = header_params[0];
    let loop_index = header_params[1];
    let body_block = ctx.builder.create_block();
    let (exit_block, exit_params) = ctx.builder.create_block_with_params(vec![elem_type.clone()]);
    let final_acc = exit_params[0];
    ctx.builder.mark_loop_header(header, exit_block, body_block).ok()?;

    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![remapped_init, zero],
        })
        .ok()?;

    // Header: check t < total_threads
    ctx.builder.switch_to_block(header).ok()?;
    let cond = ctx.builder.push_binop("<", loop_index, total, bool_ty).ok()?;
    ctx.builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: body_block,
            then_args: vec![],
            else_target: exit_block,
            else_args: vec![acc],
        })
        .ok()?;

    // Body: load partials[t], acc = f(acc, elem)
    ctx.builder.switch_to_block(body_block).ok()?;

    let ptr = ctx.push_inst(
        InstKind::StorageViewIndex {
            view: crate::ssa::types::ValueRef::from(partials_view),
            index: crate::ssa::types::ValueRef::from(loop_index),
        },
        elem_type.clone(),
    )?;
    let effect_in = ctx.entry_effect();
    let partial_elem = ctx.builder.push_load(ptr, elem_type.clone(), effect_in).ok()?;

    let mut call_args = vec![acc, partial_elem];
    call_args.extend(remapped_captures.iter().copied());
    let new_acc = ctx.push_call(reduce_function, call_args, elem_type.clone())?;

    let one = ctx.push_int("1")?;
    let next_t = ctx.push_binop("+", loop_index, one, u32_ty.clone())?;
    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![new_acc, next_t],
        })
        .ok()?;

    // Exit: store final acc to result[0]
    ctx.builder.switch_to_block(exit_block).ok()?;
    let zero_idx = ctx.push_int("0")?;
    result_output.store_result(&mut ctx, result_view, zero_idx, final_acc, elem_type)?;
    ctx.builder.terminate(Terminator::Return(None)).ok()?;

    let body = ctx.finish()?;

    let result_array_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_type.clone(),
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    );

    // Phase 2 inputs: partials buffer + any push-constant inputs from the
    // original entry that captures/init might reference.
    let mut phase2_inputs = Vec::new();
    // Add the partials buffer as a storage input
    phase2_inputs.push(EntryInput {
        name: format!("{}_partials", entry.name),
        ty: partials_view_ty.clone(),
        decoration: None,
        size_hint: None,
        storage_binding: Some(partials_binding),
        push_constant_offset: None,
    });
    // Carry forward push-constant inputs (captures may reference them)
    for input in &entry.inputs {
        if input.push_constant_offset.is_some() {
            phase2_inputs.push(input.clone());
        }
    }

    Some(EntryPoint {
        name: format!("{}_phase2_combine", entry.name),
        body,
        execution_model: ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        inputs: phase2_inputs,
        outputs: vec![EntryOutput {
            ty: result_array_ty,
            decoration: None,
            storage_binding: Some(result_binding),
        }],
        span: entry.span,
    })
}

/// Helper enum to hold input strategy data after setup (avoids trait object boxing).
enum ReduceInputData {
    Storage {
        input: StorageInput,
        handle: ValueId,
    },
    Range {
        input: RangeInput,
        handle: RangeHandle,
    },
}

// =============================================================================
// Redomap parallelization — 2 phases (like reduce but with fused map+reduce)
// =============================================================================

/// Parallelize a redomap (fused map+reduce) into two entry points.
/// Phase 1: each thread runs the fused op over its chunk → partials[thread_id]
/// Phase 2: single thread combines partials with the pure reduce combiner → result[0]
fn build_redomap_phase1(
    entry: &EntryPoint,
    source: &ArrayProvenance,
    redomap_function: &str,
    init: ValueId,
    captures: &[ValueId],
    acc_type: &Type<TypeName>,
    total_threads: u32,
    partials_binding: (u32, u32),
    local_size: (u32, u32, u32),
) -> Option<EntryPoint> {
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    let params: Vec<(Type<TypeName>, String)> =
        entry.inputs.iter().map(|i| (i.ty.clone(), i.name.clone())).collect();
    let builder = FuncBuilder::new(params, unit_ty.clone());
    let mut ctx = LowerCtx::new(builder, entry);

    let (input_len, input_strategy_data) = match source {
        ArrayProvenance::EntryStorage {
            param_index,
            storage_binding,
            ..
        } => {
            let mut input = StorageInput::new(*param_index, *storage_binding);
            let (handle, len, _elem_ty) = input.setup(&mut ctx)?;
            (len, ReduceInputData::Storage { input, handle })
        }
        ArrayProvenance::Range { value } => {
            let mut input = RangeInput::new(*value, &entry.body)?;
            let (handle, len, _elem_ty) = input.setup(&mut ctx)?;
            (len, ReduceInputData::Range { input, handle })
        }
        _ => return None,
    };

    // Thread chunking (same as reduce)
    let (thread_id, chunk_start, chunk_end) = ctx.compute_thread_chunk(input_len, total_threads)?;

    // Setup partials output buffer
    let partials_output = StorageOutput::new(partials_binding.0, partials_binding.1);
    let partials_view = partials_output.setup(&mut ctx, acc_type)?;

    let remapped_init = remap_entry_value(&mut ctx, init)?;
    let remapped_captures = ctx.remap_captures(captures)?;

    // Build reduction loop: acc = init; for i in chunk_start..chunk_end: acc = redomap_f(acc, elem, captures...)
    let u32_ty = ctx.u32_ty.clone();
    let bool_ty = ctx.bool_ty.clone();

    let (header, header_params) =
        ctx.builder.create_block_with_params(vec![acc_type.clone(), u32_ty.clone()]);
    let acc = header_params[0];
    let loop_index = header_params[1];
    let body_block = ctx.builder.create_block();
    let (exit_block, exit_params) = ctx.builder.create_block_with_params(vec![acc_type.clone()]);
    let final_acc = exit_params[0];
    ctx.builder.mark_loop_header(header, exit_block, body_block).ok()?;

    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![remapped_init, chunk_start],
        })
        .ok()?;

    ctx.builder.switch_to_block(header).ok()?;
    let cond = ctx.builder.push_binop("<", loop_index, chunk_end, bool_ty).ok()?;
    ctx.builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: body_block,
            then_args: vec![],
            else_target: exit_block,
            else_args: vec![acc],
        })
        .ok()?;

    ctx.builder.switch_to_block(body_block).ok()?;

    let input_elem = match &input_strategy_data {
        ReduceInputData::Storage { input, handle } => input.get_element(&mut ctx, *handle, loop_index)?,
        ReduceInputData::Range { input, handle } => input.get_element(&mut ctx, *handle, loop_index)?,
    };

    // call: redomap_func(acc, elem, captures...)
    let mut call_args = vec![acc, input_elem];
    call_args.extend(remapped_captures.iter().copied());
    let new_acc = ctx.push_call(redomap_function, call_args, acc_type.clone())?;

    let one = ctx.push_int("1")?;
    let next_i = ctx.push_binop("+", loop_index, one, u32_ty.clone())?;
    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![new_acc, next_i],
        })
        .ok()?;

    ctx.builder.switch_to_block(exit_block).ok()?;
    partials_output.store_result(&mut ctx, partials_view, thread_id, final_acc, acc_type)?;
    ctx.builder.terminate(Terminator::Return(None)).ok()?;

    let body = ctx.finish()?;

    let array_view_ty = Type::Constructed(
        TypeName::Array,
        vec![
            acc_type.clone(),
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    );

    Some(EntryPoint {
        name: format!("{}_phase1_chunks", entry.name),
        body,
        execution_model: ExecutionModel::Compute { local_size },
        inputs: entry.inputs.clone(),
        outputs: vec![EntryOutput {
            ty: array_view_ty,
            decoration: None,
            storage_binding: Some(partials_binding),
        }],
        span: entry.span,
    })
}

// =============================================================================
// Scan parallelization — 3 phases
// =============================================================================

/// Parallelize a scan entry point into 3 compute dispatches.
///
/// Phase 1: Each thread scans its chunk, writes results to output, last value to block_sums
/// Phase 2: Single thread scans block_sums → block_offsets (exclusive scan)
/// Phase 3: Each thread adds block_offsets[tid] to its output chunk
fn build_scan_phase1(
    entry: &EntryPoint,
    source: &ArrayProvenance,
    scan_function: &str,
    init: ValueId,
    captures: &[ValueId],
    elem_type: &Type<TypeName>,
    total_threads: u32,
    output_binding: (u32, u32),
    block_sums_binding: (u32, u32),
    local_size: (u32, u32, u32),
) -> Option<EntryPoint> {
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    let params: Vec<(Type<TypeName>, String)> =
        entry.inputs.iter().map(|i| (i.ty.clone(), i.name.clone())).collect();
    let builder = FuncBuilder::new(params, unit_ty.clone());
    let mut ctx = LowerCtx::new(builder, entry);

    // Setup input
    let (input_len, input_data) = match source {
        ArrayProvenance::EntryStorage {
            param_index,
            storage_binding,
            ..
        } => {
            let mut input = StorageInput::new(*param_index, *storage_binding);
            let (handle, len, _) = input.setup(&mut ctx)?;
            (len, ReduceInputData::Storage { input, handle })
        }
        ArrayProvenance::Range { value } => {
            let mut input = RangeInput::new(*value, &entry.body)?;
            let (handle, len, _) = input.setup(&mut ctx)?;
            (len, ReduceInputData::Range { input, handle })
        }
        ArrayProvenance::Unknown => return None,
    };

    // Setup output buffer and block_sums buffer
    let output = StorageOutput::new(output_binding.0, output_binding.1);
    let output_view = output.setup(&mut ctx, elem_type)?;
    let block_sums_out = StorageOutput::new(block_sums_binding.0, block_sums_binding.1);
    let block_sums_view = block_sums_out.setup(&mut ctx, elem_type)?;

    // Thread chunk bounds
    let (thread_id, chunk_start, chunk_end) = ctx.compute_thread_chunk(input_len, total_threads)?;

    // Remap init and captures
    let remapped_init = remap_entry_value(&mut ctx, init)?;
    let remapped_captures = ctx.remap_captures(captures)?;

    let u32_ty = ctx.u32_ty.clone();
    let bool_ty = ctx.bool_ty.clone();

    // Loop: acc = init; for i in chunk_start..chunk_end: acc = op(acc, elem); out[i] = acc
    let (header, header_params) =
        ctx.builder.create_block_with_params(vec![elem_type.clone(), u32_ty.clone()]);
    let acc = header_params[0];
    let loop_index = header_params[1];
    let body_block = ctx.builder.create_block();
    let (exit_block, exit_params) = ctx.builder.create_block_with_params(vec![elem_type.clone()]);
    let final_acc = exit_params[0];
    ctx.builder.mark_loop_header(header, exit_block, body_block).ok()?;

    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![remapped_init, chunk_start],
        })
        .ok()?;

    // Header
    ctx.builder.switch_to_block(header).ok()?;
    let cond = ctx.builder.push_binop("<", loop_index, chunk_end, bool_ty).ok()?;
    ctx.builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: body_block,
            then_args: vec![],
            else_target: exit_block,
            else_args: vec![acc],
        })
        .ok()?;

    // Body: get elem, acc = op(acc, elem), store acc to output[i]
    ctx.builder.switch_to_block(body_block).ok()?;

    let input_elem = match &input_data {
        ReduceInputData::Storage { input, handle } => input.get_element(&mut ctx, *handle, loop_index)?,
        ReduceInputData::Range { input, handle } => input.get_element(&mut ctx, *handle, loop_index)?,
    };

    let mut call_args = vec![acc, input_elem];
    call_args.extend(remapped_captures.iter().copied());
    let new_acc = ctx.push_call(scan_function, call_args, elem_type.clone())?;

    // Store to output[i] (inclusive scan)
    output.store_result(&mut ctx, output_view, loop_index, new_acc, elem_type)?;

    let one = ctx.push_int("1")?;
    let next_i = ctx.push_binop("+", loop_index, one, u32_ty.clone())?;
    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![new_acc, next_i],
        })
        .ok()?;

    // Exit: store final_acc to block_sums[thread_id]
    ctx.builder.switch_to_block(exit_block).ok()?;
    block_sums_out.store_result(&mut ctx, block_sums_view, thread_id, final_acc, elem_type)?;
    ctx.builder.terminate(Terminator::Return(None)).ok()?;

    let body = ctx.finish()?;

    let array_view_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_type.clone(),
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    );

    Some(EntryPoint {
        name: format!("{}_phase1_local_scans", entry.name),
        body,
        execution_model: ExecutionModel::Compute { local_size },
        inputs: entry.inputs.clone(),
        outputs: vec![
            EntryOutput {
                ty: array_view_ty.clone(),
                decoration: None,
                storage_binding: Some(output_binding),
            },
            EntryOutput {
                ty: array_view_ty,
                decoration: None,
                storage_binding: Some(block_sums_binding),
            },
        ],
        span: entry.span,
    })
}

/// Phase 2: Single thread scans block_sums → block_offsets (exclusive scan).
/// block_offsets[t] = sum of block_sums[0..t] (not including t).
fn build_scan_phase2(
    entry: &EntryPoint,
    scan_function: &str,
    init: ValueId,
    captures: &[ValueId],
    elem_type: &Type<TypeName>,
    total_threads: u32,
    block_sums_binding: (u32, u32),
    block_offsets_binding: (u32, u32),
) -> Option<EntryPoint> {
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

    let params: Vec<(Type<TypeName>, String)> =
        entry.inputs.iter().map(|i| (i.ty.clone(), i.name.clone())).collect();
    let builder = FuncBuilder::new(params, unit_ty.clone());
    let mut ctx = LowerCtx::new(builder, entry);

    let remapped_init = remap_entry_value(&mut ctx, init)?;
    let remapped_captures = ctx.remap_captures(captures)?;

    // Setup block_sums input
    let view_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_type.clone(),
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    );
    let block_sums_view =
        ctx.builder.emit_storage_view(block_sums_binding.0, block_sums_binding.1, view_ty.clone()).ok()?;

    // Setup block_offsets output
    let offsets_output = StorageOutput::new(block_offsets_binding.0, block_offsets_binding.1);
    let offsets_view = offsets_output.setup(&mut ctx, elem_type)?;

    // Exclusive scan: for t in 0..total_threads: offsets[t] = acc; acc = op(acc, sums[t])
    let total = ctx.push_int(&total_threads.to_string())?;
    let zero = ctx.push_int("0")?;

    let (header, header_params) =
        ctx.builder.create_block_with_params(vec![elem_type.clone(), u32_ty.clone()]);
    let acc = header_params[0];
    let loop_index = header_params[1];
    let body_block = ctx.builder.create_block();
    let exit_block = ctx.builder.create_block();
    ctx.builder.mark_loop_header(header, exit_block, body_block).ok()?;

    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![remapped_init, zero],
        })
        .ok()?;

    // Header
    ctx.builder.switch_to_block(header).ok()?;
    let cond = ctx.builder.push_binop("<", loop_index, total, bool_ty).ok()?;
    ctx.builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: body_block,
            then_args: vec![],
            else_target: exit_block,
            else_args: vec![],
        })
        .ok()?;

    // Body: offsets[t] = acc (exclusive: store BEFORE updating)
    ctx.builder.switch_to_block(body_block).ok()?;

    offsets_output.store_result(&mut ctx, offsets_view, loop_index, acc, elem_type)?;

    // Load block_sums[t]
    let ptr = ctx.push_inst(
        InstKind::StorageViewIndex {
            view: crate::ssa::types::ValueRef::from(block_sums_view),
            index: crate::ssa::types::ValueRef::from(loop_index),
        },
        elem_type.clone(),
    )?;
    let effect_in = ctx.entry_effect();
    let block_sum = ctx.builder.push_load(ptr, elem_type.clone(), effect_in).ok()?;

    // acc = op(acc, block_sum)
    let mut call_args = vec![acc, block_sum];
    call_args.extend(remapped_captures.iter().copied());
    let new_acc = ctx.push_call(scan_function, call_args, elem_type.clone())?;

    let one = ctx.push_int("1")?;
    let next_t = ctx.push_binop("+", loop_index, one, u32_ty.clone())?;
    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![new_acc, next_t],
        })
        .ok()?;

    // Exit
    ctx.builder.switch_to_block(exit_block).ok()?;
    ctx.builder.terminate(Terminator::Return(None)).ok()?;

    let body = ctx.finish()?;

    let mut phase2_inputs = Vec::new();
    phase2_inputs.push(EntryInput {
        name: format!("{}_block_sums", entry.name),
        ty: view_ty.clone(),
        decoration: None,
        size_hint: None,
        storage_binding: Some(block_sums_binding),
        push_constant_offset: None,
    });
    for input in &entry.inputs {
        if input.push_constant_offset.is_some() {
            phase2_inputs.push(input.clone());
        }
    }

    Some(EntryPoint {
        name: format!("{}_phase2_scan_sums", entry.name),
        body,
        execution_model: ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        inputs: phase2_inputs,
        outputs: vec![EntryOutput {
            ty: view_ty,
            decoration: None,
            storage_binding: Some(block_offsets_binding),
        }],
        span: entry.span,
    })
}

/// Phase 3: Each thread adds block_offsets[thread_id] to its output chunk.
/// output[i] = op(block_offsets[tid], output[i])
fn build_scan_phase3(
    entry: &EntryPoint,
    scan_function: &str,
    elem_type: &Type<TypeName>,
    total_threads: u32,
    output_binding: (u32, u32),
    block_offsets_binding: (u32, u32),
    local_size: (u32, u32, u32),
) -> Option<EntryPoint> {
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

    // Phase 3 needs entry params for GlobalInvocationId setup and function references
    let params: Vec<(Type<TypeName>, String)> =
        entry.inputs.iter().map(|i| (i.ty.clone(), i.name.clone())).collect();
    let builder = FuncBuilder::new(params, unit_ty.clone());
    let mut ctx = LowerCtx::new(builder, entry);

    let view_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_type.clone(),
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    );

    // Setup output (read-write) and offsets (read) buffers
    let output_view =
        ctx.builder.emit_storage_view(output_binding.0, output_binding.1, view_ty.clone()).ok()?;
    let offsets_view = ctx
        .builder
        .emit_storage_view(block_offsets_binding.0, block_offsets_binding.1, view_ty.clone())
        .ok()?;

    // Get output length and thread chunk bounds
    let set_val = ctx.push_int(&output_binding.0.to_string())?;
    let binding_val = ctx.push_int(&output_binding.1.to_string())?;
    let output_len = ctx.push_intrinsic(
        "_w_intrinsic_storage_len",
        vec![set_val, binding_val],
        u32_ty.clone(),
    )?;
    let (thread_id, chunk_start, chunk_end) = ctx.compute_thread_chunk(output_len, total_threads)?;

    // Load this thread's offset: block_offsets[thread_id]
    let offset_ptr = ctx.push_inst(
        InstKind::StorageViewIndex {
            view: crate::ssa::types::ValueRef::from(offsets_view),
            index: crate::ssa::types::ValueRef::from(thread_id),
        },
        elem_type.clone(),
    )?;
    let effect_in = ctx.entry_effect();
    let offset = ctx.builder.push_load(offset_ptr, elem_type.clone(), effect_in).ok()?;

    // Loop: for i in chunk_start..chunk_end: output[i] = op(offset, output[i])
    let (header, header_params) = ctx.builder.create_block_with_params(vec![u32_ty.clone()]);
    let loop_index = header_params[0];
    let body_block = ctx.builder.create_block();
    let exit_block = ctx.builder.create_block();
    ctx.builder.mark_loop_header(header, exit_block, body_block).ok()?;

    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![chunk_start],
        })
        .ok()?;

    // Header
    ctx.builder.switch_to_block(header).ok()?;
    let cond = ctx.builder.push_binop("<", loop_index, chunk_end, bool_ty).ok()?;
    ctx.builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: body_block,
            then_args: vec![],
            else_target: exit_block,
            else_args: vec![],
        })
        .ok()?;

    // Body: load output[i], compute op(offset, output[i]), store back
    ctx.builder.switch_to_block(body_block).ok()?;

    let elem_ptr = ctx.push_inst(
        InstKind::StorageViewIndex {
            view: crate::ssa::types::ValueRef::from(output_view),
            index: crate::ssa::types::ValueRef::from(loop_index),
        },
        elem_type.clone(),
    )?;
    let effect_in2 = ctx.entry_effect();
    let current_val = ctx.builder.push_load(elem_ptr, elem_type.clone(), effect_in2).ok()?;

    // new_val = op(offset, current_val)
    let new_val = ctx.push_call(scan_function, vec![offset, current_val], elem_type.clone())?;

    // Store back to output[i]
    let store_ptr = ctx.push_inst(
        InstKind::StorageViewIndex {
            view: crate::ssa::types::ValueRef::from(output_view),
            index: crate::ssa::types::ValueRef::from(loop_index),
        },
        elem_type.clone(),
    )?;
    let store_effect = ctx.entry_effect();
    ctx.builder.push_store(store_ptr, new_val, store_effect).ok()?;

    let one = ctx.push_int("1")?;
    let next_i = ctx.push_binop("+", loop_index, one, u32_ty.clone())?;
    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![next_i],
        })
        .ok()?;

    // Exit
    ctx.builder.switch_to_block(exit_block).ok()?;
    ctx.builder.terminate(Terminator::Return(None)).ok()?;

    let body = ctx.finish()?;

    // Phase 3 inputs: original entry inputs + output + offsets buffers
    let mut phase3_inputs: Vec<EntryInput> = entry.inputs.clone();
    phase3_inputs.push(EntryInput {
        name: format!("{}_output", entry.name),
        ty: view_ty.clone(),
        decoration: None,
        size_hint: None,
        storage_binding: Some(output_binding),
        push_constant_offset: None,
    });
    phase3_inputs.push(EntryInput {
        name: format!("{}_block_offsets", entry.name),
        ty: view_ty,
        decoration: None,
        size_hint: None,
        storage_binding: Some(block_offsets_binding),
        push_constant_offset: None,
    });

    Some(EntryPoint {
        name: format!("{}_phase3_add_offsets", entry.name),
        body,
        execution_model: ExecutionModel::Compute { local_size },
        inputs: phase3_inputs,
        outputs: vec![],
        span: entry.span,
    })
}

/// Strategy for reading input elements in a parallel loop.
pub trait InputStrategy {
    type Handle: Copy;

    /// Setup resources and return (handle, length, element_type).
    /// Called once before the loop.
    fn setup(&mut self, ctx: &mut LowerCtx) -> Option<(Self::Handle, ValueId, Type<TypeName>)>;

    /// Get the element at the given index using the handle from setup.
    /// Called inside the loop body.
    fn get_element(&self, ctx: &mut LowerCtx, handle: Self::Handle, index: ValueId) -> Option<ValueId>;
}

/// Strategy for writing output elements in a parallel loop.
pub trait OutputStrategy {
    type Handle: Copy;

    /// Setup resources and return a handle for storing results.
    /// Called once before the loop.
    fn setup(&self, ctx: &mut LowerCtx, elem_ty: &Type<TypeName>) -> Option<Self::Handle>;

    /// Store a result at the given index using the handle from setup.
    /// Called inside the loop body.
    fn store_result(
        &self,
        ctx: &mut LowerCtx,
        handle: Self::Handle,
        index: ValueId,
        value: ValueId,
        elem_ty: &Type<TypeName>,
    ) -> Option<()>;
}

// =============================================================================
// Storage Buffer Input
// =============================================================================

/// Input strategy for reading from a storage buffer.
pub struct StorageInput {
    param_index: usize,
    set: u32,
    binding: u32,
    /// Element type, populated by setup.
    elem_ty: Option<Type<TypeName>>,
}

impl StorageInput {
    pub fn new(param_index: usize, (set, binding): (u32, u32)) -> Self {
        Self {
            param_index,
            set,
            binding,
            elem_ty: None,
        }
    }
}

impl InputStrategy for StorageInput {
    /// The storage buffer view created during setup.
    type Handle = ValueId;

    fn setup(&mut self, ctx: &mut LowerCtx) -> Option<(ValueId, ValueId, Type<TypeName>)> {
        let array_ty = ctx.entry.inputs.get(self.param_index)?.ty.clone();
        let elem_ty = array_ty.elem_type()?.clone();

        let view = ctx.builder.emit_storage_view(self.set, self.binding, array_ty).ok()?;

        let set_val = ctx.push_int(&self.set.to_string())?;
        let binding_val = ctx.push_int(&self.binding.to_string())?;
        let storage_len = ctx.push_intrinsic(
            "_w_intrinsic_storage_len",
            vec![set_val, binding_val],
            ctx.u32_ty.clone(),
        )?;

        self.elem_ty = Some(elem_ty.clone());
        Some((view, storage_len, elem_ty))
    }

    fn get_element(&self, ctx: &mut LowerCtx, view: ValueId, index: ValueId) -> Option<ValueId> {
        let elem_ty = self.elem_ty.as_ref()?.clone();

        let ptr = ctx.push_inst(
            InstKind::StorageViewIndex {
                view: ValueRef::from(view),
                index: ValueRef::from(index),
            },
            elem_ty.clone(),
        )?;

        // Effect tokens are unordered markers (SPIR-V backend ignores them for ordering).
        // We use entry_effect() for all parallel iterations since they're independent.
        let effect_in = ctx.entry_effect();
        let elem = ctx.builder.push_load(ptr, elem_ty, effect_in).ok()?;

        Some(elem)
    }
}

// =============================================================================
// Range Input
// =============================================================================

/// Handle for range input, holding remapped start/step values.
#[derive(Clone, Copy)]
pub struct RangeHandle {
    /// Remapped start value. None for iota (starts at 0).
    start: Option<ValueId>,
    /// Remapped step value. None when step is 1.
    step: Option<ValueId>,
}

/// Input strategy for computing elements from a range (iota).
pub struct RangeInput {
    /// Kind of range: either explicit ArrayRange or simple iota
    kind: RangeKind,
    /// Length of the range (in the original body, before remapping)
    len: ValueId,
    /// Element type (usually i32)
    elem_ty: Type<TypeName>,
}

/// The kind of range input.
enum RangeKind {
    /// Full range with explicit start/step
    Explicit {
        start: ValueId,
        step: Option<ValueId>,
    },
    /// Simple iota(n): range from 0 to n with step 1
    Iota,
}

impl RangeInput {
    /// Create a RangeInput by extracting range parameters from the original body.
    ///
    /// Only handles ranges produced by instructions (ArrayRange, iota calls).
    /// Values defined by block parameters (loop-carried ranges, phi-like merges)
    /// are not recognized and return `None`, conservatively skipping parallelization.
    pub fn new(range_value: ValueId, original_body: &FuncBody) -> Option<Self> {
        // Find the instruction that produces the range value.
        // Note: block-param-defined values won't be found here — that's intentional.
        for (_iid, inst) in &original_body.inner.insts {
            if inst.result == Some(range_value) {
                let result_ty = inst.result.map(|r| original_body.inner.value_type(r));
                match &inst.data {
                    // Direct ArrayRange instruction
                    InstKind::ArrayRange { start, len, step } => {
                        let elem_ty = result_ty.and_then(extract_array_elem_type)?;
                        return Some(Self {
                            kind: RangeKind::Explicit {
                                start: start.as_ssa()?,
                                step: step.and_then(|s| s.as_ssa()),
                            },
                            len: len.as_ssa()?,
                            elem_ty,
                        });
                    }
                    // Call that produces a virtual array (iota) - range from 0 to len with step 1
                    InstKind::Call { args, .. }
                        if result_ty.is_some_and(|t| is_virtual_array(t)) && args.len() == 1 =>
                    {
                        let elem_ty = result_ty.and_then(extract_array_elem_type)?;
                        return Some(Self {
                            kind: RangeKind::Iota,
                            len: args[0].as_ssa()?,
                            elem_ty,
                        });
                    }
                    _ => {}
                }
            }
        }
        None
    }
}

/// Extract element type from an array type.
fn extract_array_elem_type(ty: &Type<TypeName>) -> Option<Type<TypeName>> {
    ty.elem_type().filter(|_| ty.is_array()).cloned()
}

/// Return the scalar type name string (e.g. "i32", "u32", "f32") for a primitive type.
fn scalar_name(ty: &Type<TypeName>) -> Option<&'static str> {
    match ty {
        Type::Constructed(TypeName::Int(32), _) => Some("i32"),
        Type::Constructed(TypeName::Int(64), _) => Some("i64"),
        Type::Constructed(TypeName::UInt(32), _) => Some("u32"),
        Type::Constructed(TypeName::UInt(64), _) => Some("u64"),
        Type::Constructed(TypeName::Float(32), _) => Some("f32"),
        Type::Constructed(TypeName::Float(64), _) => Some("f64"),
        _ => None,
    }
}

impl InputStrategy for RangeInput {
    type Handle = RangeHandle;

    fn setup(&mut self, ctx: &mut LowerCtx) -> Option<(RangeHandle, ValueId, Type<TypeName>)> {
        // Remap the length value (may be a computed expression) to the new builder
        let new_len = remap_entry_value(ctx, self.len)?;

        // Convert length to u32 for indexing (GPU invocation indices are u32)
        let elem_name = scalar_name(&self.elem_ty)?;
        let len_u32 = if elem_name == "u32" {
            new_len
        } else {
            ctx.push_call(&format!("u32.{elem_name}"), vec![new_len], ctx.u32_ty.clone())?
        };

        // Remap start/step once during setup instead of per-element
        let handle = match &self.kind {
            RangeKind::Iota => RangeHandle {
                start: None,
                step: None,
            },
            RangeKind::Explicit { start, step } => {
                let new_start = remap_entry_value(ctx, *start)?;
                let new_step = match step {
                    Some(s) => Some(remap_entry_value(ctx, *s)?),
                    None => None,
                };
                RangeHandle {
                    start: Some(new_start),
                    step: new_step,
                }
            }
        };

        Some((handle, len_u32, self.elem_ty.clone()))
    }

    fn get_element(&self, ctx: &mut LowerCtx, handle: RangeHandle, index: ValueId) -> Option<ValueId> {
        // Convert index (u32) to element type for arithmetic
        let elem_name = scalar_name(&self.elem_ty)?;
        let index_elem = if elem_name == "u32" {
            index
        } else {
            ctx.push_call(&format!("{elem_name}.u32"), vec![index], self.elem_ty.clone())?
        };

        match handle.start {
            None => {
                // iota(n): element at index i is just i
                Some(index_elem)
            }
            Some(start) => {
                // Compute: start + index * step (or start + index if step is 1)
                let elem = if let Some(step) = handle.step {
                    let idx_times_step = ctx.push_binop("*", index_elem, step, self.elem_ty.clone())?;
                    ctx.push_binop("+", start, idx_times_step, self.elem_ty.clone())?
                } else {
                    ctx.push_binop("+", start, index_elem, self.elem_ty.clone())?
                };

                Some(elem)
            }
        }
    }
}

// =============================================================================
// Storage Buffer Output
// =============================================================================

/// Output strategy for writing to a storage buffer.
pub struct StorageOutput {
    set: u32,
    binding: u32,
}

impl StorageOutput {
    pub fn new(set: u32, binding: u32) -> Self {
        Self { set, binding }
    }
}

impl OutputStrategy for StorageOutput {
    /// The output storage buffer view.
    type Handle = ValueId;

    fn setup(&self, ctx: &mut LowerCtx, elem_ty: &Type<TypeName>) -> Option<ValueId> {
        let array_ty = Type::Constructed(
            TypeName::Array,
            vec![
                elem_ty.clone(),
                Type::Constructed(TypeName::SizePlaceholder, vec![]),
                Type::Constructed(TypeName::ArrayVariantView, vec![]),
            ],
        );
        ctx.builder.emit_storage_view(self.set, self.binding, array_ty).ok()
    }

    fn store_result(
        &self,
        ctx: &mut LowerCtx,
        output_view: ValueId,
        index: ValueId,
        value: ValueId,
        elem_ty: &Type<TypeName>,
    ) -> Option<()> {
        // Effect tokens are unordered markers (SPIR-V backend ignores them for ordering).
        // We use entry_effect() for all parallel iterations since they're independent.
        let effect_in = ctx.entry_effect();
        ctx.builder.emit_storage_store(output_view, index, value, elem_ty.clone(), effect_in).ok()?;
        Some(())
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Recursively copy a value and its dependency cone from a source `FuncBody`
/// into a target `FuncBuilder`. Uses a memo cache to avoid duplicating shared
/// dependencies. Only pure (non-effectful) instructions are supported; returns
/// `None` if an effectful instruction is encountered.
pub fn remap_value(
    source: &FuncBody,
    value: ValueId,
    builder: &mut crate::ssa::builder::FuncBuilder,
    memo: &mut HashMap<ValueId, ValueId>,
) -> Option<ValueId> {
    if let Some(&mapped) = memo.get(&value) {
        return Some(mapped);
    }

    // Find the instruction that produces this value.
    let inst = source.inner.insts.values().find(|i| i.result == Some(value))?;
    let rty = source.inner.value_type(value).clone();

    let new_val = match &inst.data {
        InstKind::Int(v) => builder.push_int(v, rty).ok()?,
        InstKind::Float(v) => builder.push_float(v, rty).ok()?,
        InstKind::Bool(v) => builder.push_bool(*v).ok()?,
        InstKind::Unit => builder.push_inst(InstKind::Unit, rty).ok()?,
        InstKind::String(v) => builder.push_inst(InstKind::String(v.clone()), rty).ok()?,

        InstKind::BinOp { op, lhs, rhs } => {
            let new_lhs = remap_ref(source, *lhs, builder, memo)?;
            let new_rhs = remap_ref(source, *rhs, builder, memo)?;
            builder.push_binop(op, new_lhs, new_rhs, rty).ok()?
        }
        InstKind::UnaryOp { op, operand } => {
            let new_op = remap_ref(source, *operand, builder, memo)?;
            builder.push_unary(op, new_op, rty).ok()?
        }

        InstKind::Tuple(elems) => {
            let new_elems = remap_refs(source, elems, builder, memo)?;
            builder.push_tuple(new_elems, rty).ok()?
        }
        InstKind::Vector(elems) => {
            let new_elems: Vec<ValueRef> =
                remap_refs(source, elems, builder, memo)?.into_iter().map(ValueRef::from).collect();
            builder.push_inst(InstKind::Vector(new_elems), rty).ok()?
        }
        InstKind::Matrix(rows) => {
            let new_rows: Option<Vec<Vec<ValueRef>>> = rows
                .iter()
                .map(|row| {
                    remap_refs(source, row, builder, memo)
                        .map(|vs| vs.into_iter().map(ValueRef::from).collect())
                })
                .collect();
            builder.push_inst(InstKind::Matrix(new_rows?), rty).ok()?
        }
        InstKind::ArrayLit { elements } => {
            let new_elems = remap_refs(source, elements, builder, memo)?;
            builder.push_array_lit(new_elems, rty).ok()?
        }
        InstKind::ArrayRange { start, len, step } => {
            let new_start = remap_ref(source, *start, builder, memo)?;
            let new_len = remap_ref(source, *len, builder, memo)?;
            let new_step = match step {
                Some(s) => Some(remap_ref(source, *s, builder, memo)?),
                None => None,
            };
            builder
                .push_inst(
                    InstKind::ArrayRange {
                        start: ValueRef::from(new_start),
                        len: ValueRef::from(new_len),
                        step: new_step.map(ValueRef::from),
                    },
                    rty,
                )
                .ok()?
        }

        InstKind::Project { base, index } => {
            let new_base = remap_ref(source, *base, builder, memo)?;
            builder.push_project(new_base, *index, rty).ok()?
        }
        InstKind::Index { base, index } => {
            let new_base = remap_ref(source, *base, builder, memo)?;
            let new_index = remap_ref(source, *index, builder, memo)?;
            builder.push_index(new_base, new_index, rty).ok()?
        }

        InstKind::Call { func, args } => {
            let new_args = remap_refs(source, args, builder, memo)?;
            builder.push_call(func, new_args, rty).ok()?
        }
        InstKind::Intrinsic { name, args } => {
            let new_args = remap_refs(source, args, builder, memo)?;
            builder.push_intrinsic(name, new_args, rty).ok()?
        }

        InstKind::Global(name) => builder.push_global(name, rty).ok()?,
        InstKind::Extern(name) => builder.push_inst(InstKind::Extern(name.clone()), rty).ok()?,

        // Effectful / opaque — cannot remap
        InstKind::Alloca { .. }
        | InstKind::Load { .. }
        | InstKind::Store { .. }
        | InstKind::StorageView { .. }
        | InstKind::StorageViewIndex { .. }
        | InstKind::StorageViewLen { .. }
        | InstKind::OutputPtr { .. }
        | InstKind::Soac(_)
        | InstKind::Materialize { .. }
        | InstKind::DynamicExtract { .. } => return None,
    };

    memo.insert(value, new_val);
    Some(new_val)
}

/// Remap a ValueRef operand: for SSA refs, recursively remap; for Const, this
/// is a no-op (constants have no dependencies). Returns the remapped SSA ValueId.
fn remap_ref(
    source: &FuncBody,
    vr: ValueRef,
    builder: &mut crate::ssa::builder::FuncBuilder,
    memo: &mut HashMap<ValueId, ValueId>,
) -> Option<ValueId> {
    match vr {
        ValueRef::Ssa(id) => remap_value(source, id, builder, memo),
        ValueRef::Const(c) => {
            // Materialize the constant as an SSA instruction
            use crate::ssa::types::ConstantValue;
            match c {
                ConstantValue::I32(v) => {
                    builder.push_int(&v.to_string(), Type::Constructed(TypeName::Int(32), vec![])).ok()
                }
                ConstantValue::U32(v) => {
                    builder.push_int(&v.to_string(), Type::Constructed(TypeName::UInt(32), vec![])).ok()
                }
                ConstantValue::F32(bits) => builder
                    .push_float(
                        &f32::from_bits(bits).to_string(),
                        Type::Constructed(TypeName::Float(32), vec![]),
                    )
                    .ok(),
                ConstantValue::Bool(b) => builder.push_bool(b).ok(),
            }
        }
    }
}

fn remap_refs(
    source: &FuncBody,
    values: &[ValueRef],
    builder: &mut crate::ssa::builder::FuncBuilder,
    memo: &mut HashMap<ValueId, ValueId>,
) -> Option<Vec<ValueId>> {
    values.iter().map(|v| remap_ref(source, *v, builder, memo)).collect()
}

/// Convenience wrapper: remap a single value from the entry body using a fresh memo
/// pre-seeded with entry param mappings.
pub fn remap_entry_value(ctx: &mut LowerCtx, value: ValueId) -> Option<ValueId> {
    let body = ctx.entry.body.clone();
    let mut memo: HashMap<ValueId, ValueId> =
        body.params.iter().enumerate().map(|(i, (src, _, _))| (*src, ctx.builder.get_param(i))).collect();

    remap_value(&body, value, &mut ctx.builder, &mut memo)
}
