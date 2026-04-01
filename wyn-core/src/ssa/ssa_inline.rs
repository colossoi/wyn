//! SSA-level function inlining pass.
//!
//! Runs after SSA conversion. Inlines small functions at their call sites
//! by copying callee blocks into the caller with fresh IDs.
//!
//! Constants passed as arguments stay as `ValueRef::Const` in the inlined
//! instructions — they are never materialized as SSA instructions unless
//! they appear in a terminator (which requires a `ValueId`).

use super::types::{BlockId, FuncBody, InstKind, Program, Soac, Terminator, ValueId, ValueRef};
use std::collections::{HashMap, HashSet};
use wyn_ssa::{self, InstId, ValueDef};

/// Maximum number of SSA instructions for a function to be inlined.
const INLINE_INST_THRESHOLD: usize = 30;

/// Inline small functions in the program.
pub fn inline_small_functions(mut program: Program) -> Program {
    let candidates: HashMap<String, FuncBody> = program
        .functions
        .iter()
        .filter(|f| {
            let size = f.body.num_insts();
            if size == 0 || size > INLINE_INST_THRESHOLD {
                return false;
            }
            // Skip SOAC lambda bodies — they're called from SOAC loop expansion,
            // not from normal Call instructions.
            if f.name.starts_with("_w_lambda_") {
                return false;
            }
            // Skip functions containing SOACs (they have complex lowering requirements).
            let has_soac = f.body.inner.insts.values().any(|inst| {
                matches!(inst.data, InstKind::Soac(_))
            });
            !has_soac
        })
        .map(|f| (f.name.clone(), f.body.clone()))
        .collect();

    if candidates.is_empty() {
        return program;
    }

    for func in &mut program.functions {
        // Don't inline into SOAC lambda bodies — they have special calling conventions.
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

fn inline_calls_in_body(body: &mut FuncBody, candidates: &HashMap<String, FuncBody>) {
    // Don't inline into lambda bodies — they're SOAC loop bodies.
    // (This pass runs on program.functions, which includes lambdas.)
    loop {
        let Some((inst_id, callee_name)) = find_inline_call(body, candidates) else {
            break;
        };
        let callee_body = &candidates[&callee_name];
        inline_call_site(body, inst_id, callee_body);
    }
}

fn find_inline_call(
    body: &FuncBody,
    candidates: &HashMap<String, FuncBody>,
) -> Option<(InstId, String)> {
    for (inst_id, inst) in &body.inner.insts {
        if let InstKind::Call { func, .. } = &inst.data {
            if inst.result.is_some() && candidates.contains_key(func) {
                return Some((inst_id, func.clone()));
            }
        }
    }
    None
}

/// The mapping from callee ValueId to caller value. A callee function param
/// may map to either an SSA value or an inline constant.
enum ValueMapping {
    Ssa(ValueId),
    Const(super::types::ConstantValue),
}

fn inline_call_site(caller: &mut FuncBody, call_inst: InstId, callee: &FuncBody) {
    let call_block = caller.inner.insts[call_inst].parent;

    let call_args: Vec<ValueRef> = match &caller.inner.insts[call_inst].data {
        InstKind::Call { args, .. } => args.clone(),
        _ => unreachable!(),
    };

    // Split the block at the call. Removes the call instruction, creates
    // a continuation block with a param for the call result.
    let split = wyn_ssa::split_block_at(&mut caller.inner, call_inst);
    let cont_block = split.cont_block;

    // Build mappings.
    let mut value_map: HashMap<ValueId, ValueMapping> = HashMap::new();
    let mut block_map: HashMap<BlockId, BlockId> = HashMap::new();

    // Map callee function params → call args.
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

    // Create fresh blocks.
    let callee_entry_block = caller.inner.create_block();
    block_map.insert(callee.inner.entry, callee_entry_block);

    for (callee_bid, _) in &callee.inner.blocks {
        if callee_bid == callee.inner.entry {
            continue;
        }
        let new_bid = caller.inner.create_block();
        block_map.insert(callee_bid, new_bid);
    }

    // Create fresh block params.
    for (callee_bid, callee_block) in &callee.inner.blocks {
        let new_bid = block_map[&callee_bid];
        for &callee_param in &callee_block.params {
            let ty = callee.inner.values[callee_param].ty.clone();
            let new_param = caller.inner.add_block_param(new_bid, ty);
            value_map.insert(callee_param, ValueMapping::Ssa(new_param));
        }
    }

    // Copy instructions, using substitute_values to keep constants as ValueRef::Const.
    for (callee_bid, callee_block) in &callee.inner.blocks {
        let new_bid = block_map[&callee_bid];
        for &callee_inst_id in &callee_block.insts {
            let callee_inst = &callee.inner.insts[callee_inst_id];
            let mut new_data = callee_inst.data.clone();

            // Remap ValueRefs: Ssa values get remapped through value_map,
            // and const-mapped params become ValueRef::Const directly.
            new_data.substitute_values(&mut |vr: &mut ValueRef| {
                if let ValueRef::Ssa(vid) = vr {
                    match value_map.get(vid) {
                        Some(ValueMapping::Ssa(new_vid)) => *vr = ValueRef::Ssa(*new_vid),
                        Some(ValueMapping::Const(c)) => *vr = ValueRef::Const(*c),
                        None => {} // shouldn't happen
                    }
                }
            });

            if let Some(callee_result) = callee_inst.result {
                let ty = callee.inner.values[callee_result].ty.clone();
                let new_val = caller.inner.append_inst(new_bid, new_data, ty, None);
                value_map.insert(callee_result, ValueMapping::Ssa(new_val));
            } else {
                caller.inner.append_void_inst(new_bid, new_data, None);
            }
        }
    }

    // Set terminators on copied blocks.
    // Terminators use raw ValueId, so const-mapped values must be materialized here.
    for (callee_bid, callee_block) in &callee.inner.blocks {
        let new_bid = block_map[&callee_bid];
        let new_term = remap_terminator(
            caller,
            new_bid,
            &callee_block.term,
            &value_map,
            &block_map,
            cont_block,
            callee,
        );
        caller.inner.blocks[new_bid].term = new_term;
    }

    // Branch from the original call block to the callee entry.
    caller.inner.blocks[call_block].term = Terminator::Branch {
        target: callee_entry_block,
        args: vec![],
    };
}

/// Resolve a callee ValueId through the mapping, materializing constants
/// as instructions only when needed (for terminators).
fn resolve_value_id(
    caller: &mut FuncBody,
    block: BlockId,
    vid: &ValueId,
    value_map: &HashMap<ValueId, ValueMapping>,
    callee: &FuncBody,
) -> ValueId {
    match value_map.get(vid) {
        Some(ValueMapping::Ssa(new_vid)) => *new_vid,
        Some(ValueMapping::Const(c)) => {
            // Must materialize: terminators need real ValueIds.
            let ty = callee.inner.values[*vid].ty.clone();
            let inst = materialize_const(*c);
            caller.inner.append_inst(block, inst, ty, None)
        }
        None => panic!("resolve_value_id: unmapped callee value {:?}", vid),
    }
}

fn remap_terminator(
    caller: &mut FuncBody,
    new_bid: BlockId,
    term: &Terminator,
    value_map: &HashMap<ValueId, ValueMapping>,
    block_map: &HashMap<BlockId, BlockId>,
    cont_block: BlockId,
    callee: &FuncBody,
) -> Terminator {
    // Collect all ValueIds referenced by the terminator, then resolve them.
    // This avoids multiple mutable borrows of caller.
    let mut vid_refs: Vec<ValueId> = Vec::new();
    match term {
        Terminator::Return(Some(val)) => vid_refs.push(*val),
        Terminator::Branch { args, .. } => vid_refs.extend(args),
        Terminator::CondBranch {
            cond,
            then_args,
            else_args,
            ..
        } => {
            vid_refs.push(*cond);
            vid_refs.extend(then_args);
            vid_refs.extend(else_args);
        }
        _ => {}
    }

    // Resolve all referenced values (may materialize constants).
    let resolved: Vec<ValueId> = vid_refs
        .iter()
        .map(|vid| resolve_value_id(caller, new_bid, vid, value_map, callee))
        .collect();

    // Rebuild terminator with resolved values.
    let mut ri = resolved.into_iter();
    match term {
        Terminator::Return(Some(_)) => Terminator::Branch {
            target: cont_block,
            args: vec![ri.next().unwrap()],
        },
        Terminator::Return(None) => Terminator::Branch {
            target: cont_block,
            args: vec![],
        },
        Terminator::Branch { target, args } => Terminator::Branch {
            target: block_map[target],
            args: (0..args.len()).map(|_| ri.next().unwrap()).collect(),
        },
        Terminator::CondBranch {
            then_target,
            then_args,
            else_target,
            else_args,
            ..
        } => {
            let cond = ri.next().unwrap();
            let new_then_args: Vec<_> = (0..then_args.len()).map(|_| ri.next().unwrap()).collect();
            let new_else_args: Vec<_> = (0..else_args.len()).map(|_| ri.next().unwrap()).collect();
            Terminator::CondBranch {
                cond,
                then_target: block_map[then_target],
                then_args: new_then_args,
                else_target: block_map[else_target],
                else_args: new_else_args,
            }
        }
        Terminator::Unreachable => Terminator::Unreachable,
    }
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
            },
            _ => {}
        }
    }
}
