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
use crate::ssa::types::Program;
use crate::ssa::types::{
    BlockId, EffectToken, EntryPoint, FuncBody, InstKind, Soac, Terminator, TerminatorExt, ValueId,
    ValueRef,
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
        Soac::MapInto {
            func,
            inputs,
            captures,
            output_view,
            input_array_types,
            input_elem_types,
            output_elem_type,
        } => expand_map_into(
            builder,
            func,
            &remap_values(inputs, value_map),
            &remap_values(captures, value_map),
            value_map[output_view],
            input_array_types,
            input_elem_types,
            output_elem_type,
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
        Soac::ScanInto {
            func,
            input,
            init,
            captures,
            output_view,
            input_array_type,
            input_elem_type,
        } => expand_scan_into(
            builder,
            func,
            value_map[input],
            value_map[init],
            &remap_values(captures, value_map),
            value_map[output_view],
            input_array_type,
            input_elem_type,
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

/// Check if an array type is a view (storage-buffer-backed) array.
fn is_view_array(ty: &Type<TypeName>) -> bool {
    match ty {
        Type::Constructed(TypeName::Array, args) if args.len() == 3 => {
            crate::types::is_array_variant_view(&args[2])
        }
        _ => false,
    }
}

/// Read an element from an array.
/// - View arrays: StorageViewIndex + Load
/// - Virtual arrays: compute start + index * step (like the old RangeInput strategy)
/// - Everything else: soa_index (which uses push_index)
fn read_element(
    builder: &mut FuncBuilder,
    arr: ValueId,
    index: ValueId,
    arr_ty: &Type<TypeName>,
    elem_ty: &Type<TypeName>,
) -> Option<ValueId> {
    if is_view_array(arr_ty) {
        let ptr = builder
            .push_inst(
                InstKind::StorageViewIndex {
                    view: ValueRef::Ssa(arr),
                    index: ValueRef::Ssa(index),
                },
                elem_ty.clone(),
            )
            .ok()?;
        let effect = builder.entry_effect();
        builder.push_load(ptr, elem_ty.clone(), effect).ok()
    } else if is_virtual_array(arr_ty) {
        // Virtual array {start, step, len}: element = start + index * step
        let start = builder.push_project(arr, 0, elem_ty.clone()).ok()?;
        let step = builder.push_project(arr, 1, elem_ty.clone()).ok()?;
        let idx_times_step = builder.push_binop("*", index, step, elem_ty.clone()).ok()?;
        builder.push_binop("+", start, idx_times_step, elem_ty.clone()).ok()
    } else {
        soa_index(builder, arr, index, arr_ty, elem_ty).ok()
    }
}

/// Expand a Map SOAC using storage-aware strategies (for compute entry points).
///
/// Expand a MapInto SOAC: loop over inputs, call function, write each result
/// to the output storage view via StorageViewIndex + Store.
fn expand_map_into(
    builder: &mut FuncBuilder,
    func: &str,
    inputs: &[ValueId],
    captures: &[ValueId],
    output_view: ValueId,
    input_array_types: &[Type<TypeName>],
    input_elem_types: &[Type<TypeName>],
    output_elem_type: &Type<TypeName>,
) -> Option<ValueId> {
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);

    let first_input_ty = &input_array_types[0];
    let len = soa_length(builder, inputs[0], first_input_ty).ok()?;

    // Loop with index only — no accumulator. Writes go to storage.
    let header = builder.create_block();
    let body_block = builder.create_block();
    let exit = builder.create_block();
    let index = builder.add_block_param(header, i32_ty.clone());

    let zero = builder.push_int("0", i32_ty.clone()).ok()?;
    builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![zero],
        })
        .ok()?;

    builder.switch_to_block_unchecked(header);
    builder.mark_loop_header(header, exit, body_block);
    let cond = builder.push_binop("<", index, len, bool_ty.clone()).ok()?;
    builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: body_block,
            then_args: vec![],
            else_target: exit,
            else_args: vec![],
        })
        .ok()?;

    builder.switch_to_block_unchecked(body_block);

    // Read input elements — dispatches to StorageViewIndex+Load for view arrays,
    // soa_index for everything else (composite arrays, virtual/range arrays).
    let mut input_elems: Vec<ValueId> = Vec::with_capacity(inputs.len());
    for (j, &arr) in inputs.iter().enumerate() {
        let elem = read_element(builder, arr, index, &input_array_types[j], &input_elem_types[j])?;
        input_elems.push(elem);
    }

    let mut call_args = input_elems;
    call_args.extend(captures.iter().copied());

    let output_elem = builder.push_call(func, call_args, output_elem_type.clone()).ok()?;

    // Write to output storage view
    let out_ptr = builder
        .push_inst(
            InstKind::StorageViewIndex {
                view: ValueRef::Ssa(output_view),
                index: ValueRef::Ssa(index),
            },
            output_elem_type.clone(),
        )
        .ok()?;
    let effect = builder.entry_effect();
    builder.push_store(out_ptr, output_elem, effect).ok()?;

    let one = builder.push_int("1", i32_ty.clone()).ok()?;
    let next_i = builder.push_binop("+", index, one, i32_ty).ok()?;
    builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![next_i],
        })
        .ok()?;

    builder.switch_to_block_unchecked(exit);

    // MapInto doesn't produce a value — the output is in storage.
    // Return a dummy.
    let dummy_ty = Type::Constructed(TypeName::Bool, vec![]);
    builder.push_inst(InstKind::Bool(false), dummy_ty).ok()
}

/// Expand a ScanInto SOAC: loop with accumulator, write each result to output storage view.
fn expand_scan_into(
    builder: &mut FuncBuilder,
    func: &str,
    arr_value: ValueId,
    init_value: ValueId,
    captures: &[ValueId],
    output_view: ValueId,
    input_array_type: &Type<TypeName>,
    input_elem_type: &Type<TypeName>,
) -> Option<ValueId> {
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let acc_ty = input_elem_type.clone();

    let len = soa_length(builder, arr_value, input_array_type).ok()?;

    // Loop with (acc, index) — no array accumulator needed.
    let header = builder.create_block();
    let body_block = builder.create_block();
    let exit = builder.create_block();
    let acc = builder.add_block_param(header, acc_ty.clone());
    let index = builder.add_block_param(header, i32_ty.clone());

    let zero = builder.push_int("0", i32_ty.clone()).ok()?;
    builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![init_value, zero],
        })
        .ok()?;

    builder.switch_to_block_unchecked(header);
    let _ = builder.mark_loop_header(header, exit, body_block);
    let cond = builder.push_binop("<", index, len, bool_ty).ok()?;
    builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: body_block,
            then_args: vec![],
            else_target: exit,
            else_args: vec![],
        })
        .ok()?;

    builder.switch_to_block_unchecked(body_block);

    let elem = read_element(builder, arr_value, index, input_array_type, input_elem_type)?;

    let mut call_args = vec![acc, elem];
    call_args.extend(captures.iter().copied());
    let new_acc = builder.push_call(func, call_args, acc_ty.clone()).ok()?;

    // Write new_acc to output view at index
    let out_ptr = builder
        .push_inst(
            InstKind::StorageViewIndex {
                view: ValueRef::Ssa(output_view),
                index: ValueRef::Ssa(index),
            },
            input_elem_type.clone(),
        )
        .ok()?;
    let effect = builder.entry_effect();
    builder.push_store(out_ptr, new_acc, effect).ok()?;

    let one = builder.push_int("1", i32_ty.clone()).ok()?;
    let next_i = builder.push_binop("+", index, one, i32_ty).ok()?;
    builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![new_acc, next_i],
        })
        .ok()?;

    builder.switch_to_block_unchecked(exit);

    let dummy_ty = Type::Constructed(TypeName::Bool, vec![]);
    builder.push_inst(InstKind::Bool(false), dummy_ty).ok()
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
        let elem = read_element(
            builder,
            arr,
            loop_blocks.index,
            &input_array_types[i],
            &input_elem_types[i],
        )?;
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

/// Expand a Map over storage-backed (view) arrays.
///
/// Instead of accumulating into an array with array_with, this reads each
/// element via Index (which the SPIR-V backend lowers to StorageViewIndex+Load
/// for view arrays) and writes results via StorageViewIndex+Store.
///
/// Returns the input array as the "result" (the output was written in-place
/// to the output storage buffer by to_ssa's entry point output handling).
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

    let elem = read_element(
        builder,
        arr_value,
        loop_blocks.index,
        input_array_type,
        input_elem_type,
    )?;

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

    let elem = read_element(
        builder,
        arr_value,
        loop_blocks.index,
        input_array_type,
        input_elem_type,
    )?;

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
        let elem = read_element(
            builder,
            arr,
            loop_blocks.index,
            &input_array_types[i],
            &input_elem_types[i],
        )?;
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

/// Recursively remap a ValueId from one FuncBody into a new builder,
/// re-creating pure instructions as needed.
pub fn remap_value(
    source: &FuncBody,
    value: ValueId,
    builder: &mut crate::ssa::builder::FuncBuilder,
    memo: &mut HashMap<ValueId, ValueId>,
) -> Option<ValueId> {
    if let Some(&mapped) = memo.get(&value) {
        return Some(mapped);
    }

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

fn remap_ref(
    source: &FuncBody,
    vr: ValueRef,
    builder: &mut crate::ssa::builder::FuncBuilder,
    memo: &mut HashMap<ValueId, ValueId>,
) -> Option<ValueId> {
    match vr {
        ValueRef::Ssa(id) => remap_value(source, id, builder, memo),
        ValueRef::Const(c) => {
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

/// Remap a single value from the entry body, pre-seeded with entry param mappings.
pub fn remap_entry_value(ctx: &mut LowerCtx, value: ValueId) -> Option<ValueId> {
    let body = ctx.entry.body.clone();
    let mut memo: HashMap<ValueId, ValueId> =
        body.params.iter().enumerate().map(|(i, (src, _, _))| (*src, ctx.builder.get_param(i))).collect();
    remap_value(&body, value, &mut ctx.builder, &mut memo)
}
