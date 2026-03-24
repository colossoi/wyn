//! SSA instruction merging: transplant instructions from one FuncBody into a FuncBuilder.
//!
//! Two entry points:
//! - `merge_instructions`: single-block source, splices into current block (constant hoisting)
//! - `merge_body`: multi-block source, creates new blocks in target (inlining / body rebuild)

use std::collections::HashMap;

use crate::mir::ssa::{BlockId, EffectToken, FuncBody, Terminator, ValueId};
use crate::mir::ssa_builder::FuncBuilder;
use crate::mir::ssa_soac_lower::register_new_effects;

/// Result of merging a source FuncBody into a target FuncBuilder.
pub struct MergeResult {
    /// Maps source ValueIds to target ValueIds.
    pub value_map: HashMap<ValueId, ValueId>,
    /// Maps source BlockIds to target BlockIds.
    pub block_map: HashMap<BlockId, BlockId>,
    /// Maps source EffectTokens to target EffectTokens.
    pub effect_map: HashMap<EffectToken, EffectToken>,
}

/// Merge a single-block body's instructions into the current block of the target.
///
/// Does NOT copy the terminator — the caller provides continuation context.
/// Returns the final value/effect maps and the remapped result ValueId of the
/// source body's `Return` terminator (if any).
///
/// This is the constant-hoisting primitive: splice a constant definition's
/// instructions into the caller at the current insertion point.
///
/// Panics if the source has more than one block or contains `Soac` instructions.
pub fn merge_instructions(
    source: &FuncBody,
    builder: &mut FuncBuilder,
    seed_values: HashMap<ValueId, ValueId>,
    seed_effects: HashMap<EffectToken, EffectToken>,
) -> (MergeResult, Option<ValueId>) {
    assert_eq!(
        source.blocks.len(),
        1,
        "merge_instructions requires single-block source"
    );

    let block = &source.blocks[0];
    let mut value_map = seed_values;
    let mut effect_map = seed_effects;
    let mut last_result = None;

    for &inst_id in &block.insts {
        let inst = &source.insts[inst_id.index()];
        let new_kind = inst.kind.remap(
            &|v| value_map[v],
            &|e| *effect_map.get(e).unwrap_or(e),
            &mut || builder.alloc_effect(),
        );
        register_new_effects(&inst.kind, &new_kind, &mut effect_map);

        if inst.result.is_some() {
            let new_val = builder
                .push_inst(new_kind, inst.result_ty.clone(), inst.span, inst.node_id)
                .expect("BUG: failed to push merged instruction");
            value_map.insert(inst.result.unwrap(), new_val);
            last_result = Some(new_val);
        } else {
            builder
                .push_void_inst(new_kind, inst.span, inst.node_id)
                .expect("BUG: failed to push merged void instruction");
        }
    }

    // Extract the return value from the terminator, if present
    let return_val = match &block.terminator {
        Some(Terminator::Return(v)) => Some(value_map[v]),
        _ => last_result,
    };

    let result = MergeResult {
        value_map,
        block_map: HashMap::new(),
        effect_map,
    };
    (result, return_val)
}

/// Merge a multi-block FuncBody into a target FuncBuilder.
///
/// Creates new blocks for all non-entry source blocks. The entry block's
/// instructions are spliced into the builder's current block.
/// Copies terminators and control headers with remapping.
///
/// `seed_values` provides initial ValueId mappings (e.g., source params → target values).
/// `seed_effects` provides initial EffectToken mappings.
///
/// Panics if a source instruction is `Soac`.
pub fn merge_body(
    source: &FuncBody,
    builder: &mut FuncBuilder,
    seed_values: HashMap<ValueId, ValueId>,
    seed_effects: HashMap<EffectToken, EffectToken>,
) -> MergeResult {
    let mut value_map = seed_values;
    let mut effect_map = seed_effects;
    let mut block_map: HashMap<BlockId, BlockId> = HashMap::new();

    // Entry block merges into the builder's current block.
    let current = builder.current_block().expect("no current block");
    block_map.insert(BlockId::ENTRY, current);

    // Register entry block params
    for param in &source.blocks[0].params {
        let new_val = builder.alloc_value(param.ty.clone());
        value_map.insert(param.value, new_val);
    }

    // Pre-create all non-entry blocks with their parameters
    for (idx, block) in source.blocks.iter().enumerate() {
        if idx == 0 {
            continue;
        }
        let param_types: Vec<_> = block.params.iter().map(|p| p.ty.clone()).collect();
        let (new_block_id, new_param_vals) = builder.create_block_with_params(param_types);
        block_map.insert(BlockId(idx as u32), new_block_id);
        for (old_param, &new_val) in block.params.iter().zip(new_param_vals.iter()) {
            value_map.insert(old_param.value, new_val);
        }
    }

    // Process each block
    for (idx, block) in source.blocks.iter().enumerate() {
        if block.is_dead() {
            continue;
        }

        // Switch to the target block (entry block is already current)
        if idx != 0 {
            let new_block_id = block_map[&BlockId(idx as u32)];
            builder.switch_to_block(new_block_id).expect("BUG: failed to switch block during merge");
        }

        // Set control header
        if let Some(ref ctrl) = block.control {
            let target_block = block_map[&BlockId(idx as u32)];
            let new_ctrl = ctrl.remap(&|b| block_map[b]);
            builder.set_control_header(target_block, new_ctrl);
        }

        // Re-emit instructions
        for &inst_id in &block.insts {
            let inst = &source.insts[inst_id.index()];
            let new_kind = inst.kind.remap(
                &|v| value_map[v],
                &|e| *effect_map.get(e).unwrap_or(e),
                &mut || builder.alloc_effect(),
            );
            register_new_effects(&inst.kind, &new_kind, &mut effect_map);

            if inst.result.is_some() {
                let new_val = builder
                    .push_inst(new_kind, inst.result_ty.clone(), inst.span, inst.node_id)
                    .expect("BUG: failed to push merged instruction");
                value_map.insert(inst.result.unwrap(), new_val);
            } else {
                builder
                    .push_void_inst(new_kind, inst.span, inst.node_id)
                    .expect("BUG: failed to push merged void instruction");
            }
        }

        // Remap and set terminator
        if let Some(ref term) = block.terminator {
            let new_term = term.remap(&|v| value_map[v], &|b| block_map[b]);
            builder.terminate(new_term).ok();
        }
    }

    MergeResult {
        value_map,
        block_map,
        effect_map,
    }
}
