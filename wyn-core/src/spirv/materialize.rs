//! SPIR-V preparation pass: rewrite dynamic composite indexing into
//! Materialize + DynamicExtract form.
//!
//! SPIR-V's OpCompositeExtract requires literal indices. When an Index
//! instruction uses a runtime index (e.g., a loop variable), the backend
//! must spill the composite to a local variable and use OpAccessChain.
//!
//! This pass makes that spill explicit in the SSA IR by rewriting:
//!     %result = Index { base: %arr, index: %i }
//! into:
//!     %handle = Materialize { value: %arr }
//!     %result = DynamicExtract { base: %handle, index: %i }
//!
//! After this rewrite, `lift_and_merge` can hoist Materialize out of loops
//! since it only depends on the (loop-invariant) array value.

use crate::ssa::types::{FuncBody, InstKind, Program, ValueRef, WynFunction};
use std::collections::HashSet;
use wyn_ssa::ValueLike;

/// Run the materialize pass on the entire program.
pub fn materialize_dynamic_indices(mut program: Program) -> Program {
    for func in &mut program.functions {
        materialize_func(&mut func.body);
    }
    for entry in &mut program.entry_points {
        materialize_func(&mut entry.body);
    }
    program
}

fn materialize_func(body: &mut FuncBody) {
    rewrite_dynamic_indices(body);
    hoist_materialize_past_loops(body);
    wyn_ssa::lift_and_merge(&mut body.inner);
}

fn rewrite_dynamic_indices(body: &mut FuncBody) {
    let mut rewrites: Vec<(wyn_ssa::InstId, ValueRef, ValueRef)> = Vec::new();

    for (iid, inst) in &body.inner.insts {
        if let InstKind::Index { base, index } = &inst.data {
            // Check if the index is produced by a constant instruction or is an inline constant
            let is_const = match index {
                ValueRef::Const(_) => true,
                ValueRef::Ssa(id) => is_constant_int(&body.inner, *id),
            };
            if !is_const {
                if let Some(result) = inst.result {
                    rewrites.push((iid, *base, *index));
                    let _ = result; // used for identification only
                }
            }
        }
    }

    if rewrites.is_empty() {
        return;
    }

    // Apply rewrites: for each Index to rewrite, replace it with Materialize + DynamicExtract.
    for (inst_id, base, index) in rewrites {
        let result = body.inner.insts[inst_id].result;
        let parent = body.inner.insts[inst_id].parent;

        // Get the type of the original result (element type)
        let result_ty = result.map(|r| body.inner.value_type(r).clone());

        // Get the type of the base (array type) — used for Materialize's result type
        let base_id = base.as_ssa().expect("Materialize base must be SSA");
        let base_ty = body.inner.value_type(base_id).clone();

        // Find the position of this instruction in its block
        let pos = body.inner.blocks[parent]
            .insts
            .iter()
            .position(|&id| id == inst_id)
            .expect("instruction not found in parent block");

        // Remove the original Index instruction
        body.inner.insts.remove(inst_id);
        body.inner.blocks[parent].insts.remove(pos);
        if let Some(r) = result {
            body.inner.values.remove(r);
        }

        // Insert Materialize at the same position
        let mat_value = body.inner.insert_inst_at_index(
            parent,
            pos,
            InstKind::Materialize {
                value: ValueRef::from(base_id),
            },
            base_ty,
            None,
        );

        // Insert DynamicExtract right after Materialize
        if let Some(elem_ty) = result_ty {
            let extract_value = body.inner.insert_inst_at_index(
                parent,
                pos + 1,
                InstKind::DynamicExtract {
                    base: ValueRef::from(mat_value),
                    index,
                },
                elem_ty,
                None,
            );

            // Rewrite all uses of the old result to the new DynamicExtract result
            if let Some(old_result) = result {
                body.inner.replace_all_uses(old_result, extract_value);
            }
        }
    }
}

/// Check if a value is produced by a constant integer instruction.
fn is_constant_int(
    func: &wyn_ssa::Function<
        InstKind,
        crate::ssa::types::EffectToken,
        polytype::Type<crate::ast::TypeName>,
    >,
    value: wyn_ssa::ValueId,
) -> bool {
    let inst_id = match func.inst_of_value(value) {
        Some(id) => id,
        None => return false, // block param or function param — not constant
    };
    matches!(func.insts[inst_id].data, InstKind::Int(_))
}

/// Move Materialize instructions (and their loop-invariant operand chains) to the
/// function entry block. The generic lift_and_merge can't cross CondBranch blocks
/// (which include loop headers), so we handle that here with control_headers knowledge.
fn hoist_materialize_past_loops(body: &mut FuncBody) {
    let entry = body.inner.entry;

    // Find all Materialize instructions not in the entry block
    let to_hoist: Vec<wyn_ssa::InstId> = body
        .inner
        .insts
        .iter()
        .filter(|(_, inst)| matches!(inst.data, InstKind::Materialize { .. }) && inst.parent != entry)
        .map(|(id, _)| id)
        .collect();

    for mat_id in to_hoist {
        // Also hoist the operand chain (the ArrayLit that produces the array value)
        let operand_ids = collect_hoistable_chain(&body.inner, mat_id);

        for id in operand_ids {
            if body.inner.insts[id].parent == entry {
                continue;
            }
            let old_block = body.inner.insts[id].parent;
            if let Some(pos) = body.inner.blocks[old_block].insts.iter().position(|&i| i == id) {
                body.inner.blocks[old_block].insts.remove(pos);
            }
            body.inner.blocks[entry].insts.push(id);
            body.inner.insts[id].parent = entry;
        }
    }
}

/// Collect an instruction and all its hoistable SSA operands (transitively).
/// Returns them in dependency order (operands first).
fn collect_hoistable_chain(func: &WynFunction, root: wyn_ssa::InstId) -> Vec<wyn_ssa::InstId> {
    let mut result = Vec::new();
    let mut visited = HashSet::new();
    collect_chain_rec(func, root, &mut result, &mut visited);
    result
}

fn collect_chain_rec(
    func: &WynFunction,
    inst_id: wyn_ssa::InstId,
    result: &mut Vec<wyn_ssa::InstId>,
    visited: &mut HashSet<wyn_ssa::InstId>,
) {
    if !visited.insert(inst_id) {
        return;
    }
    if !func.insts.contains_key(inst_id) {
        return;
    }
    // Recurse into SSA operands first (dependency order)
    let uses: Vec<wyn_ssa::ValueId> = func.insts[inst_id].data.ssa_uses();
    for val_id in uses {
        let def_inst = func.values.get(val_id).and_then(|v| match v.def {
            wyn_ssa::ValueDef::Inst { inst } => Some(inst),
            _ => None,
        });
        if let Some(def_inst) = def_inst {
            if func.insts[def_inst].data.is_hoistable() {
                collect_chain_rec(func, def_inst, result, visited);
            }
        }
    }
    result.push(inst_id);
}
