//! SSA-level function inlining pass.
//!
//! Runs after SSA conversion. Inlines small functions at their call sites
//! by copying callee blocks into the caller with fresh IDs.
//!
//! Constants passed as arguments stay as `ValueRef::Const` in the inlined
//! instructions — they are never materialized as SSA instructions unless
//! they appear in a terminator (which requires a `ValueId`).

use super::types::{FuncBody, InstKind, Program, Soac, Terminator, ValueId, ValueRef};
use std::collections::{HashMap, HashSet};
use wyn_ssa::{self, InstId, ValueDef};

/// Maximum number of SSA instructions for a function to be inlined.
const INLINE_INST_THRESHOLD: usize = 30;

/// Inline small functions in the program.
pub fn inline_small_functions(mut program: Program) -> Program {
    let mut candidates: HashMap<String, FuncBody> = program
        .functions
        .iter()
        .filter(|f| {
            let size = f.body.num_insts();
            if size == 0 || size > INLINE_INST_THRESHOLD {
                return false;
            }
            // Skip SOAC lambda bodies — they're called from SOAC loop expansion.
            if f.name.starts_with("_w_lambda_") {
                return false;
            }
            // Only inline single-block functions (no control flow).
            // Multi-block callees break SPIR-V structured control flow.
            if f.body.inner.blocks.len() != 1 {
                return false;
            }
            // Skip functions containing SOACs.
            let has_soac = f.body.inner.insts.values().any(|inst| matches!(inst.data, InstKind::Soac(_)));
            !has_soac
        })
        .map(|f| (f.name.clone(), f.body.clone()))
        .collect();

    // Also include program-level constants (zero-arg defs with constant bodies).
    for constant in &program.constants {
        candidates.insert(constant.name.clone(), constant.body.clone());
    }

    if candidates.is_empty() {
        return program;
    }

    // Inline Global references (constants) into ALL functions, including lambdas.
    let constant_names: HashSet<String> = program.constants.iter().map(|c| c.name.clone()).collect();
    for func in &mut program.functions {
        inline_globals_in_body(&mut func.body, &candidates, &constant_names);
    }
    for entry in &mut program.entry_points {
        inline_globals_in_body(&mut entry.body, &candidates, &constant_names);
    }

    // Inline small function calls (but not into lambda bodies).
    for func in &mut program.functions {
        if func.name.starts_with("_w_lambda_") {
            continue;
        }
        inline_calls_in_body(&mut func.body, &candidates);
    }
    for entry in &mut program.entry_points {
        inline_calls_in_body(&mut entry.body, &candidates);
    }

    // Clean up block structure introduced by splicing.
    for func in &mut program.functions {
        wyn_ssa::forward_single_pred_params(&mut func.body.inner);
        wyn_ssa::eliminate_empty_blocks(&mut func.body.inner);
    }
    for entry in &mut program.entry_points {
        wyn_ssa::forward_single_pred_params(&mut entry.body.inner);
        wyn_ssa::eliminate_empty_blocks(&mut entry.body.inner);
    }

    remove_dead_functions(&mut program);
    program
}

/// Inline Global references to constants by replacing them with the constant's body.
fn inline_globals_in_body(
    body: &mut FuncBody,
    candidates: &HashMap<String, FuncBody>,
    constant_names: &HashSet<String>,
) {
    loop {
        let site = body.inner.insts.iter().find_map(|(inst_id, inst)| {
            if let InstKind::Global(name) = &inst.data {
                if constant_names.contains(name) && candidates.contains_key(name) {
                    return Some((inst_id, name.clone()));
                }
            }
            None
        });
        let Some((inst_id, name)) = site else { break };
        let callee_body = &candidates[&name];
        inline_call_site(body, inst_id, callee_body);
    }
}

fn inline_calls_in_body(body: &mut FuncBody, candidates: &HashMap<String, FuncBody>) {
    loop {
        let Some((inst_id, callee_name)) = find_inline_call(body, candidates) else {
            break;
        };
        let callee_body = &candidates[&callee_name];
        inline_call_site(body, inst_id, callee_body);
    }
}

fn find_inline_call(body: &FuncBody, candidates: &HashMap<String, FuncBody>) -> Option<(InstId, String)> {
    for (inst_id, inst) in &body.inner.insts {
        match &inst.data {
            InstKind::Call { func, .. } if inst.result.is_some() && candidates.contains_key(func) => {
                return Some((inst_id, func.clone()));
            }
            InstKind::Global(name) if inst.result.is_some() && candidates.contains_key(name) => {
                return Some((inst_id, name.clone()));
            }
            _ => {}
        }
    }
    None
}

/// The mapping from callee ValueId to caller value. A callee function param
/// may map to either an SSA value or an inline constant.
#[derive(Clone)]
enum ValueMapping {
    Ssa(ValueId),
    Const(super::types::ConstantValue),
}

/// Inline a single-block callee by inserting its instructions directly
/// into the caller block at the call site. No block splitting needed.
fn inline_call_site(caller: &mut FuncBody, call_inst: InstId, callee: &FuncBody) {
    assert_eq!(
        callee.inner.blocks.len(),
        1,
        "inline_call_site requires single-block callee"
    );

    let call_block = caller.inner.insts[call_inst].parent;
    let call_result = caller.inner.insts[call_inst].result;

    let call_args: Vec<ValueRef> = match &caller.inner.insts[call_inst].data {
        InstKind::Call { args, .. } => args.clone(),
        InstKind::Global(_) => vec![], // zero-arg constant
        _ => unreachable!(),
    };

    // Map callee function params → call args.
    let mut value_map: HashMap<ValueId, ValueMapping> = HashMap::new();
    let callee_func_params: Vec<(ValueId, usize)> = callee
        .inner
        .values
        .iter()
        .filter_map(|(vid, vinfo)| match vinfo.def {
            ValueDef::FunctionParam { index } => Some((vid, index)),
            _ => None,
        })
        .collect();

    for (callee_param_vid, param_index) in &callee_func_params {
        if *param_index >= call_args.len() {
            continue;
        }
        let mapping = match &call_args[*param_index] {
            ValueRef::Ssa(vid) => ValueMapping::Ssa(*vid),
            ValueRef::Const(c) => ValueMapping::Const(*c),
        };
        value_map.insert(*callee_param_vid, mapping);
    }

    // Find the call's position in the block so we can insert before it.
    let call_pos = caller.inner.blocks[call_block]
        .insts
        .iter()
        .position(|&id| id == call_inst)
        .expect("call inst not found in its block");

    // Copy callee instructions into the caller block at the call position.
    let callee_entry = callee.inner.entry;
    let callee_block = &callee.inner.blocks[callee_entry];
    let mut return_value: Option<ValueMapping> = None;
    let mut insert_offset = 0;

    for &callee_inst_id in &callee_block.insts {
        let callee_inst = &callee.inner.insts[callee_inst_id];
        let mut new_data = callee_inst.data.clone();

        new_data.substitute_values(&mut |vr: &mut ValueRef| {
            if let ValueRef::Ssa(vid) = vr {
                match value_map.get(vid) {
                    Some(ValueMapping::Ssa(new_vid)) => *vr = ValueRef::Ssa(*new_vid),
                    Some(ValueMapping::Const(c)) => *vr = ValueRef::Const(*c),
                    None => {
                        panic!(
                            "ssa_inline: unmapped callee value {:?} in {:?}",
                            vid, callee_inst.data
                        );
                    }
                }
            }
        });

        if let Some(callee_result) = callee_inst.result {
            let ty = callee.inner.values[callee_result].ty.clone();
            let new_val =
                caller.inner.insert_inst_at_index(call_block, call_pos + insert_offset, new_data, ty, None);
            value_map.insert(callee_result, ValueMapping::Ssa(new_val));
        } else {
            // Insert void inst at position
            let inst_id = caller.inner.append_void_inst(call_block, new_data, None);
            // Move from end to the right position
            let insts = &mut caller.inner.blocks[call_block].insts;
            let popped = insts.pop().unwrap();
            insts.insert(call_pos + insert_offset, popped);
            let _ = inst_id;
        }
        insert_offset += 1;
    }

    // Find the callee's return value.
    if let Terminator::Return(Some(ret_val)) = &callee_block.term {
        return_value = Some(value_map.get(ret_val).cloned().unwrap_or(ValueMapping::Ssa(*ret_val)));
    }

    // Replace the call result with the return value.
    if let (Some(call_result_vid), Some(ret_mapping)) = (call_result, return_value) {
        match ret_mapping {
            ValueMapping::Ssa(ret_vid) => {
                caller.inner.replace_all_uses(call_result_vid, ret_vid);
            }
            ValueMapping::Const(c) => {
                // Must materialize the constant as an instruction.
                let ty = caller.inner.values[call_result_vid].ty.clone();
                let const_inst = materialize_const(c);
                let materialized = caller.inner.append_inst(call_block, const_inst, ty, None);
                caller.inner.replace_all_uses(call_result_vid, materialized);
            }
        }
        caller.inner.values.remove(call_result_vid);
    }

    // Remove the call instruction from the block's inst list and the arena.
    caller.inner.blocks[call_block].insts.retain(|&id| id != call_inst);
    caller.inner.insts.remove(call_inst);
}

fn materialize_const(c: super::types::ConstantValue) -> InstKind {
    use super::types::ConstantValue;
    match c {
        ConstantValue::I32(v) => InstKind::Int(v.to_string()),
        ConstantValue::U32(v) => InstKind::Int(v.to_string()),
        ConstantValue::F32(bits) => InstKind::Float(f32::from_bits(bits).to_string()),
        ConstantValue::Bool(v) => InstKind::Bool(v),
    }
}

fn remove_dead_functions(program: &mut Program) {
    let mut referenced: HashSet<String> = HashSet::new();
    let mut worklist: Vec<String> = Vec::new();

    for entry in &program.entry_points {
        collect_called_functions(&entry.body, &mut worklist);
    }
    for constant in &program.constants {
        collect_called_functions(&constant.body, &mut worklist);
    }

    while let Some(name) = worklist.pop() {
        if !referenced.insert(name.clone()) {
            continue;
        }
        if let Some(func) = program.functions.iter().find(|f| f.name == name) {
            collect_called_functions(&func.body, &mut worklist);
        }
    }

    program.functions.retain(|f| referenced.contains(&f.name));
}

fn collect_called_functions(body: &FuncBody, out: &mut Vec<String>) {
    for (_, inst) in &body.inner.insts {
        match &inst.data {
            InstKind::Call { func, .. } => out.push(func.clone()),
            InstKind::Soac(soac) => match soac {
                Soac::Map { func, .. } => out.push(func.clone()),
                Soac::Reduce { func, .. } => out.push(func.clone()),
                Soac::Scan { func, .. } => out.push(func.clone()),
                Soac::Redomap { func, .. } => out.push(func.clone()),
            },
            _ => {}
        }
    }
}
