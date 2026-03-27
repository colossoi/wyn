//! SSA instruction merging: transplant instructions from one FuncBody into a FuncBuilder.
//!
//! Two entry points:
//! - `merge_instructions`: single-block source, splices into current block (constant hoisting)
//! - `merge_body`: multi-block source, creates new blocks in target (inlining / body rebuild)

use std::collections::HashMap;

use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::{BlockId, EffectToken, FuncBody, Terminator, TerminatorExt, ValueId};

/// Result of merging a source FuncBody into a target FuncBuilder.
pub struct MergeResult {
    pub value_map: HashMap<ValueId, ValueId>,
    pub block_map: HashMap<BlockId, BlockId>,
    pub effect_map: HashMap<EffectToken, EffectToken>,
}

/// Merge a single-block body's instructions into the current block of the target.
///
/// Does NOT copy the terminator — the caller provides continuation context.
/// Returns the final value/effect maps and the remapped result ValueId of the
/// source body's `Return` terminator (if any).
pub fn merge_instructions(
    source: &FuncBody,
    builder: &mut FuncBuilder,
    seed_values: HashMap<ValueId, ValueId>,
    seed_effects: HashMap<EffectToken, EffectToken>,
) -> (MergeResult, Option<ValueId>) {
    assert_eq!(
        source.inner.blocks.len(),
        1,
        "merge_instructions requires single-block source"
    );

    let entry = source.entry_block();
    let block = &source.inner.blocks[entry];
    let mut value_map = seed_values;
    let mut effect_map = seed_effects;
    let mut last_result = None;

    for &inst_id in &block.insts {
        let inst = &source.inner.insts[inst_id];
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
            let result_ty = source.inner.value_type(result).clone();
            let new_val = builder
                .push_inst_with_effects(new_kind, result_ty, new_effects)
                .expect("BUG: failed to push merged instruction");
            value_map.insert(result, new_val);
            last_result = Some(new_val);
        } else {
            builder
                .push_void_inst_with_effects(new_kind, new_effects)
                .expect("BUG: failed to push merged void instruction");
        }
    }

    // Extract the return value from the terminator, if present
    let return_val = match &block.term {
        Terminator::Return(Some(v)) => Some(value_map[v]),
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
pub fn merge_body(
    source: &FuncBody,
    builder: &mut FuncBuilder,
    seed_values: HashMap<ValueId, ValueId>,
    seed_effects: HashMap<EffectToken, EffectToken>,
) -> MergeResult {
    let mut value_map = seed_values;
    let mut effect_map = seed_effects;
    let mut block_map: HashMap<BlockId, BlockId> = HashMap::new();

    let source_entry = source.entry_block();

    // Entry block merges into the builder's current block.
    let current = builder.current_block().expect("no current block");
    block_map.insert(source_entry, current);

    // Register entry block params
    let entry_block = &source.inner.blocks[source_entry];
    for &param in &entry_block.params {
        let ty = source.inner.value_type(param).clone();
        let new_val = builder.add_block_param(current, ty);
        value_map.insert(param, new_val);
    }

    // Pre-create all non-entry blocks with their parameters
    for (bid, block) in &source.inner.blocks {
        if bid == source_entry {
            continue;
        }
        let param_types: Vec<_> =
            block.params.iter().map(|&p| source.inner.value_type(p).clone()).collect();
        let (new_block_id, new_param_vals) = builder.create_block_with_params(param_types);
        block_map.insert(bid, new_block_id);
        for (&old_param, &new_val) in block.params.iter().zip(new_param_vals.iter()) {
            value_map.insert(old_param, new_val);
        }
    }

    // Process each block
    for (bid, block) in &source.inner.blocks {
        // Skip dead blocks
        if block.insts.is_empty() && matches!(block.term, Terminator::Unreachable) {
            continue;
        }

        // Switch to the target block (entry block is already current)
        if bid != source_entry {
            let new_block_id = block_map[&bid];
            builder.switch_to_block(new_block_id).expect("BUG: failed to switch block during merge");
        }

        // Set control header
        if let Some(ctrl) = source.control_headers.get(&bid) {
            let target_block = block_map[&bid];
            let new_ctrl = ctrl.remap(&|b| block_map[&b]);
            builder.set_control_header(target_block, new_ctrl);
        }

        // Re-emit instructions
        for &inst_id in &block.insts {
            let inst = &source.inner.insts[inst_id];
            let new_kind = inst.data.remap(&|v| value_map[&v]);

            let new_effects = inst.effects.map(|(ein, _eout)| {
                let mapped_in = *effect_map.get(&ein).unwrap_or(&ein);
                let mapped_out = builder.alloc_effect();
                (mapped_in, mapped_out)
            });
            if let (Some((_, old_out)), Some((_, new_out))) = (inst.effects, new_effects) {
                effect_map.insert(old_out, new_out);
            }

            if let Some(result) = inst.result {
                let result_ty = source.inner.value_type(result).clone();
                let new_val = builder
                    .push_inst_with_effects(new_kind, result_ty, new_effects)
                    .expect("BUG: failed to push merged instruction");
                value_map.insert(result, new_val);
            } else {
                builder
                    .push_void_inst_with_effects(new_kind, new_effects)
                    .expect("BUG: failed to push merged void instruction");
            }
        }

        // Remap and set terminator
        let new_term = block.term.remap(&|v| value_map[&v], &|b| block_map[&b]);
        builder.terminate(new_term).ok();
    }

    MergeResult {
        value_map,
        block_map,
        effect_map,
    }
}
