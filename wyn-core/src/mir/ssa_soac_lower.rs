//! SOAC lowering pass: expands first-class SOAC instructions into explicit loops.
//!
//! This pass runs after SSA optimization and before backend lowering (SPIR-V/GLSL).
//! It transforms `InstKind::Soac(SsaSoac::Map { .. })` into for-range loops,
//! `InstKind::Soac(SsaSoac::Reduce { .. })` into accumulation loops, etc.
//!
//! Functions with no SOAC instructions pass through unchanged.

use std::collections::HashMap;

use crate::ast::{NodeId, Span, TypeName};
use crate::mir::soa_helpers::{extract_array_size, soa_array_with, soa_index, soa_length, soa_uninit};
use crate::mir::ssa::{
    BlockId, ControlHeader, EffectToken, FuncBody, InstKind,
    SsaSoac, Terminator, ValueId, ViewSource,
};
use crate::mir::ssa_builder::FuncBuilder;
use crate::tlc::to_ssa::SsaProgram;
use polytype::Type;

/// Lower all SOAC instructions in the program to explicit loops.
pub fn lower_soacs(mut program: SsaProgram) -> SsaProgram {
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
    body.insts.iter().any(|inst| matches!(inst.kind, InstKind::Soac(_)))
}

/// Rebuild a function body, expanding SOAC instructions into loops.
fn lower_func_body(old_body: &FuncBody) -> FuncBody {
    let params: Vec<(Type<TypeName>, String)> = old_body
        .params
        .iter()
        .map(|(_, ty, name)| (ty.clone(), name.clone()))
        .collect();

    let mut builder = FuncBuilder::new(params, old_body.return_ty.clone());

    // Map old ValueId -> new ValueId
    let mut value_map: HashMap<ValueId, ValueId> = HashMap::new();

    // Map old EffectToken -> new EffectToken
    let mut effect_map: HashMap<EffectToken, EffectToken> = HashMap::new();
    effect_map.insert(old_body.entry_effect, builder.entry_effect());

    // Map function params
    for (i, (old_val, _, _)) in old_body.params.iter().enumerate() {
        value_map.insert(*old_val, builder.get_param(i));
    }

    // Map old BlockId -> new BlockId (entry block is already created)
    let mut block_map: HashMap<BlockId, BlockId> = HashMap::new();
    block_map.insert(BlockId::ENTRY, BlockId::ENTRY);

    // Pre-create all non-entry blocks with their parameters
    for (idx, block) in old_body.blocks.iter().enumerate() {
        if idx == 0 {
            // Entry block params (if any) — entry block has no params normally
            for param in &block.params {
                let new_val = builder.alloc_value(param.ty.clone());
                value_map.insert(param.value, new_val);
            }
            continue;
        }

        let param_types: Vec<Type<TypeName>> = block.params.iter().map(|p| p.ty.clone()).collect();
        let (new_block_id, new_param_vals) = builder.create_block_with_params(param_types);
        block_map.insert(BlockId(idx as u32), new_block_id);

        for (old_param, &new_val) in block.params.iter().zip(new_param_vals.iter()) {
            value_map.insert(old_param.value, new_val);
        }
    }

    // Process each block
    for (idx, block) in old_body.blocks.iter().enumerate() {
        let old_block_id = BlockId(idx as u32);

        // Skip dead blocks
        if block.is_dead() {
            let new_block_id = block_map[&old_block_id];
            if new_block_id != BlockId::ENTRY {
                builder.switch_to_block_unchecked(new_block_id);
            }
            builder.terminate(Terminator::Unreachable).ok();
            continue;
        }

        let new_block_id = block_map[&old_block_id];
        if idx == 0 {
            // Entry block is already current
        } else {
            builder.switch_to_block_unchecked(new_block_id);
        }

        // Copy control header
        if let Some(ref ctrl) = block.control {
            let new_ctrl = remap_control_header(ctrl, &block_map);
            builder.set_control_header(new_block_id, new_ctrl);
        }

        // Process instructions
        for &inst_id in &block.insts {
            let inst = &old_body.insts[inst_id.index()];

            match &inst.kind {
                InstKind::Soac(soac) => {
                    // Expand the SOAC into loops
                    let result_ty = inst.result_ty.clone();
                    let span = inst.span;
                    let node_id = inst.node_id;
                    let new_val = expand_soac(
                        &mut builder,
                        soac,
                        result_ty,
                        span,
                        node_id,
                        &value_map,
                    );
                    if let (Some(old_result), Some(new_result)) = (inst.result, new_val) {
                        value_map.insert(old_result, new_result);
                    }
                }
                _ => {
                    // Re-emit the instruction with remapped values
                    let new_kind = remap_inst_kind(&inst.kind, &value_map, &effect_map, &mut builder);
                    // Track new effect tokens
                    register_new_effects(&inst.kind, &new_kind, &mut effect_map);

                    if inst.result.is_some() {
                        let new_val = builder
                            .push_inst(new_kind, inst.result_ty.clone(), inst.span, inst.node_id)
                            .expect("BUG: failed to push remapped instruction");
                        value_map.insert(inst.result.unwrap(), new_val);
                    } else {
                        builder
                            .push_void_inst(new_kind, inst.span, inst.node_id)
                            .expect("BUG: failed to push remapped void instruction");
                    }
                }
            }
        }

        // Remap and set terminator
        if let Some(ref term) = block.terminator {
            let new_term = remap_terminator(term, &value_map, &block_map);
            builder.terminate(new_term).ok();
        }
    }

    builder.finish().expect("BUG: failed to finish rebuilt function body")
}

/// Expand a SOAC instruction into explicit loop(s) using the builder.
/// Returns the result ValueId of the expansion.
fn expand_soac(
    builder: &mut FuncBuilder,
    soac: &SsaSoac,
    result_ty: Type<TypeName>,
    span: Span,
    node_id: NodeId,
    value_map: &HashMap<ValueId, ValueId>,
) -> Option<ValueId> {
    match soac {
        SsaSoac::Map {
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
            span,
            node_id,
        ),
        SsaSoac::Reduce {
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
            span,
            node_id,
        ),
        SsaSoac::Scan { .. } => {
            panic!("internal compiler error: Scan SOAC lowering not yet implemented")
        }
    }
}

fn remap_values(values: &[ValueId], map: &HashMap<ValueId, ValueId>) -> Vec<ValueId> {
    values.iter().map(|v| map[v]).collect()
}

/// Expand a Map SOAC into a for-range loop.
/// This reproduces the loop pattern previously emitted by `convert_soac_map` in `to_ssa.rs`.
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
    span: Span,
    node_id: NodeId,
) -> Option<ValueId> {
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);

    // Get length from first input
    let first_input_ty = &input_array_types[0];
    let array_size = extract_array_size(first_input_ty);

    let len = match array_size {
        Some(n) => builder
            .push_int(&n.to_string(), i32_ty.clone(), span, node_id)
            .ok()?,
        None => soa_length(builder, inputs[0], first_input_ty, span, node_id).ok()?,
    };

    // Create uninitialized result array (SoA-aware)
    let init_array = soa_uninit(builder, &result_ty, span, node_id).ok()?;

    // Create loop
    let loop_blocks = builder.create_for_range_loop(result_ty.clone());

    // Branch to header
    let zero = builder.push_int("0", i32_ty.clone(), span, node_id).ok()?;
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![init_array, zero],
        })
        .ok()?;

    // Header
    builder.switch_to_block(loop_blocks.header).ok()?;
    let cond = builder
        .push_binop("<", loop_blocks.index, len, bool_ty, span, node_id)
        .ok()?;
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

    // Index each input array at loop position (SoA-aware)
    let mut input_elems: Vec<ValueId> = Vec::with_capacity(inputs.len());
    for (i, &arr) in inputs.iter().enumerate() {
        let elem = soa_index(
            builder,
            arr,
            loop_blocks.index,
            &input_array_types[i],
            &input_elem_types[i],
            span,
            node_id,
        )
        .ok()?;
        input_elems.push(elem);
    }

    // Build function call args
    let mut call_args: Vec<ValueId> = if zipped {
        let zipped_ty = zipped_param_type.expect("BUG: zipped map without param type").clone();
        let zipped_val = builder.push_tuple(input_elems, zipped_ty, span, node_id).ok()?;
        vec![zipped_val]
    } else {
        input_elems
    };
    call_args.extend(captures.iter().copied());

    let output_elem = builder
        .push_call(func, call_args, output_elem_type.clone(), span, node_id)
        .ok()?;

    // Update accumulator array (SoA-aware)
    let new_arr = soa_array_with(
        builder,
        loop_blocks.acc,
        loop_blocks.index,
        output_elem,
        &result_ty,
        span,
        node_id,
    )
    .ok()?;

    let one = builder.push_int("1", i32_ty.clone(), span, node_id).ok()?;
    let next_i = builder
        .push_binop("+", loop_blocks.index, one, i32_ty, span, node_id)
        .ok()?;
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
    span: Span,
    node_id: NodeId,
) -> Option<ValueId> {
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);
    let acc_ty = result_ty.clone();

    // Get length (SoA-aware)
    let len = soa_length(builder, arr_value, input_array_type, span, node_id).ok()?;

    // Create loop
    let loop_blocks = builder.create_for_range_loop(acc_ty.clone());

    // Branch to header with (init, 0)
    let zero = builder.push_int("0", i32_ty.clone(), span, node_id).ok()?;
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![init_value, zero],
        })
        .ok()?;

    // Header
    builder.switch_to_block(loop_blocks.header).ok()?;
    let cond = builder
        .push_binop("<", loop_blocks.index, len, bool_ty, span, node_id)
        .ok()?;
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

    // Index array element (SoA-aware)
    let elem = soa_index(
        builder,
        arr_value,
        loop_blocks.index,
        input_array_type,
        input_elem_type,
        span,
        node_id,
    )
    .ok()?;

    let mut call_args = vec![loop_blocks.acc, elem];
    call_args.extend(captures.iter().copied());
    let new_acc = builder.push_call(func, call_args, acc_ty, span, node_id).ok()?;

    let one = builder.push_int("1", i32_ty.clone(), span, node_id).ok()?;
    let next_i = builder
        .push_binop("+", loop_blocks.index, one, i32_ty, span, node_id)
        .ok()?;
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

// =============================================================================
// Value / effect / block remapping helpers
// =============================================================================

fn remap_effect(
    e: &EffectToken,
    em: &HashMap<EffectToken, EffectToken>,
    builder: &mut FuncBuilder,
) -> EffectToken {
    em.get(e).copied().unwrap_or_else(|| builder.alloc_effect())
}

fn remap_inst_kind(
    kind: &InstKind,
    vm: &HashMap<ValueId, ValueId>,
    em: &HashMap<EffectToken, EffectToken>,
    builder: &mut FuncBuilder,
) -> InstKind {
    let rv = |v: &ValueId| -> ValueId { vm[v] };

    match kind {
        InstKind::Int(s) => InstKind::Int(s.clone()),
        InstKind::Float(s) => InstKind::Float(s.clone()),
        InstKind::Bool(b) => InstKind::Bool(*b),
        InstKind::Unit => InstKind::Unit,
        InstKind::String(s) => InstKind::String(s.clone()),
        InstKind::BinOp { op, lhs, rhs } => InstKind::BinOp {
            op: op.clone(),
            lhs: rv(lhs),
            rhs: rv(rhs),
        },
        InstKind::UnaryOp { op, operand } => InstKind::UnaryOp {
            op: op.clone(),
            operand: rv(operand),
        },
        InstKind::Tuple(elems) => InstKind::Tuple(elems.iter().map(rv).collect()),
        InstKind::ArrayLit { elements } => InstKind::ArrayLit {
            elements: elements.iter().map(rv).collect(),
        },
        InstKind::ArrayRange { start, len, step } => InstKind::ArrayRange {
            start: rv(start),
            len: rv(len),
            step: step.as_ref().map(rv),
        },
        InstKind::Vector(elems) => InstKind::Vector(elems.iter().map(rv).collect()),
        InstKind::Matrix(rows) => {
            InstKind::Matrix(rows.iter().map(|row| row.iter().map(rv).collect()).collect())
        }
        InstKind::Project { base, index } => InstKind::Project {
            base: rv(base),
            index: *index,
        },
        InstKind::Index { base, index } => InstKind::Index {
            base: rv(base),
            index: rv(index),
        },
        InstKind::Call { func, args } => InstKind::Call {
            func: func.clone(),
            args: args.iter().map(rv).collect(),
        },
        InstKind::Global(name) => InstKind::Global(name.clone()),
        InstKind::Extern(name) => InstKind::Extern(name.clone()),
        InstKind::Intrinsic { name, args } => InstKind::Intrinsic {
            name: name.clone(),
            args: args.iter().map(rv).collect(),
        },
        InstKind::Alloca {
            elem_ty,
            effect_in,
            ..
        } => {
            let new_in = remap_effect(effect_in, em, builder);
            let new_out = builder.alloc_effect();
            InstKind::Alloca {
                elem_ty: elem_ty.clone(),
                effect_in: new_in,
                effect_out: new_out,
            }
        }
        InstKind::Load {
            ptr,
            effect_in,
            ..
        } => {
            let new_in = remap_effect(effect_in, em, builder);
            let new_out = builder.alloc_effect();
            InstKind::Load {
                ptr: rv(ptr),
                effect_in: new_in,
                effect_out: new_out,
            }
        }
        InstKind::Store {
            ptr,
            value,
            effect_in,
            ..
        } => {
            let new_in = remap_effect(effect_in, em, builder);
            let new_out = builder.alloc_effect();
            InstKind::Store {
                ptr: rv(ptr),
                value: rv(value),
                effect_in: new_in,
                effect_out: new_out,
            }
        }
        InstKind::StorageView { source, offset, len } => InstKind::StorageView {
            source: match source {
                ViewSource::Storage { set, binding } => ViewSource::Storage {
                    set: *set,
                    binding: *binding,
                },
                ViewSource::Inherited { parent } => ViewSource::Inherited { parent: rv(parent) },
            },
            offset: rv(offset),
            len: rv(len),
        },
        InstKind::StorageViewIndex { view, index } => InstKind::StorageViewIndex {
            view: rv(view),
            index: rv(index),
        },
        InstKind::StorageViewLen { view } => InstKind::StorageViewLen { view: rv(view) },
        InstKind::OutputPtr { index } => InstKind::OutputPtr { index: *index },
        InstKind::Soac(_) => {
            panic!("internal compiler error: nested Soac in remap_inst_kind")
        }
    }
}

/// After remapping an effectful instruction, register the new effect_out in the map.
fn register_new_effects(
    old_kind: &InstKind,
    new_kind: &InstKind,
    effect_map: &mut HashMap<EffectToken, EffectToken>,
) {
    let old_out = match old_kind {
        InstKind::Alloca { effect_out, .. }
        | InstKind::Load { effect_out, .. }
        | InstKind::Store { effect_out, .. } => Some(*effect_out),
        _ => None,
    };
    let new_out = match new_kind {
        InstKind::Alloca { effect_out, .. }
        | InstKind::Load { effect_out, .. }
        | InstKind::Store { effect_out, .. } => Some(*effect_out),
        _ => None,
    };
    if let (Some(old), Some(new)) = (old_out, new_out) {
        effect_map.insert(old, new);
    }
}

fn remap_terminator(
    term: &Terminator,
    vm: &HashMap<ValueId, ValueId>,
    bm: &HashMap<BlockId, BlockId>,
) -> Terminator {
    let rv = |v: &ValueId| -> ValueId { vm[v] };
    let rb = |b: &BlockId| -> BlockId { bm[b] };

    match term {
        Terminator::Branch { target, args } => Terminator::Branch {
            target: rb(target),
            args: args.iter().map(rv).collect(),
        },
        Terminator::CondBranch {
            cond,
            then_target,
            then_args,
            else_target,
            else_args,
        } => Terminator::CondBranch {
            cond: rv(cond),
            then_target: rb(then_target),
            then_args: then_args.iter().map(rv).collect(),
            else_target: rb(else_target),
            else_args: else_args.iter().map(rv).collect(),
        },
        Terminator::Return(v) => Terminator::Return(rv(v)),
        Terminator::ReturnUnit => Terminator::ReturnUnit,
        Terminator::Unreachable => Terminator::Unreachable,
    }
}

fn remap_control_header(ctrl: &ControlHeader, bm: &HashMap<BlockId, BlockId>) -> ControlHeader {
    match ctrl {
        ControlHeader::Loop { merge, continue_block } => ControlHeader::Loop {
            merge: bm[merge],
            continue_block: bm[continue_block],
        },
        ControlHeader::Selection { merge } => ControlHeader::Selection { merge: bm[merge] },
    }
}
