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

use crate::ssa::types::{FuncBody, InstKind, Program};

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
    wyn_ssa::lift_and_merge(&mut body.inner);
}

fn rewrite_dynamic_indices(body: &mut FuncBody) {
    let mut rewrites: Vec<(wyn_ssa::InstId, wyn_ssa::ValueId, wyn_ssa::ValueId)> = Vec::new();

    for (iid, inst) in &body.inner.insts {
        if let InstKind::Index { base, index } = &inst.data {
            // Check if the index is produced by a constant instruction
            if !is_constant_int(&body.inner, *index) {
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
        let base_ty = body.inner.value_type(base).clone();

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
            InstKind::Materialize { value: base },
            base_ty,
            None,
        );

        // Insert DynamicExtract right after Materialize
        if let Some(elem_ty) = result_ty {
            let extract_value = body.inner.insert_inst_at_index(
                parent,
                pos + 1,
                InstKind::DynamicExtract {
                    base: mat_value,
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
