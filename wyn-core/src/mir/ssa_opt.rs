//! SSA peephole optimizations: param forwarding and empty block elimination.
//!
//! Pass 1 — Single-predecessor param forwarding:
//!   When a block has exactly one predecessor that branches unconditionally to it,
//!   replace all uses of the block's params with the branch args, then clear both.
//!
//! Pass 2 — Empty block elimination:
//!   Blocks with no instructions that unconditionally branch to another block
//!   can be bypassed. All predecessors are redirected to the target.
//!   The entry block (BlockId(0)) is never eliminated.
//!   Iterates to fixpoint for chained empty blocks.

use crate::mir::ssa::{BlockId, ControlHeader, FuncBody, InstKind, Terminator, ValueId};
use crate::tlc::to_ssa::SsaProgram;
use std::collections::{HashMap, HashSet};

/// Run SSA peephole optimizations on the entire program.
pub fn optimize(mut program: SsaProgram) -> SsaProgram {
    for func in &mut program.functions {
        optimize_func(&mut func.body);
    }
    for entry in &mut program.entry_points {
        optimize_func(&mut entry.body);
    }
    program
}

fn optimize_func(body: &mut FuncBody) {
    forward_params(body);
    eliminate_empty_blocks(body);
}

// =============================================================================
// Pass 1: Single-predecessor param forwarding
// =============================================================================

fn forward_params(body: &mut FuncBody) {
    // Build predecessor map: target_block -> list of (source_block, branch_args).
    // We count *all* incoming edges (unconditional and conditional) so we can
    // identify blocks with exactly one predecessor edge.
    let mut pred_map: HashMap<BlockId, Vec<(BlockId, Vec<ValueId>)>> = HashMap::new();

    for (idx, block) in body.blocks.iter().enumerate() {
        let src = BlockId(idx as u32);
        if let Some(ref term) = block.terminator {
            match term {
                Terminator::Branch { target, args } => {
                    pred_map.entry(*target).or_default().push((src, args.clone()));
                }
                Terminator::CondBranch {
                    then_target,
                    then_args,
                    else_target,
                    else_args,
                    ..
                } => {
                    pred_map.entry(*then_target).or_default().push((src, then_args.clone()));
                    pred_map.entry(*else_target).or_default().push((src, else_args.clone()));
                }
                _ => {}
            }
        }
    }

    // Collect substitutions from blocks with exactly one predecessor
    // where that predecessor branches unconditionally.
    let mut substitutions: HashMap<ValueId, ValueId> = HashMap::new();
    let mut blocks_to_clear_params: Vec<BlockId> = Vec::new();
    let mut branches_to_clear_args: Vec<BlockId> = Vec::new();

    for (target, preds) in &pred_map {
        if preds.len() != 1 {
            continue;
        }
        let (pred_id, pred_args) = &preds[0];

        // The predecessor must use an unconditional Branch (not CondBranch)
        let is_unconditional = matches!(
            body.blocks[pred_id.index()].terminator,
            Some(Terminator::Branch { .. })
        );
        if !is_unconditional {
            continue;
        }

        let target_block = &body.blocks[target.index()];
        if target_block.params.is_empty() {
            continue;
        }

        // param_value → branch_arg
        for (param, arg) in target_block.params.iter().zip(pred_args.iter()) {
            substitutions.insert(param.value, *arg);
        }

        blocks_to_clear_params.push(*target);
        branches_to_clear_args.push(*pred_id);
    }

    if substitutions.is_empty() {
        return;
    }

    // Clear params and branch args
    for bid in blocks_to_clear_params {
        body.blocks[bid.index()].params.clear();
    }
    for bid in branches_to_clear_args {
        if let Some(Terminator::Branch { args, .. }) = &mut body.blocks[bid.index()].terminator {
            args.clear();
        }
    }

    // Resolve transitive chains (a→b, b→c  ⇒  a→c)
    let resolved = resolve_substitutions(&substitutions);

    apply_substitutions(body, &resolved);
}

/// Chase substitution chains to a fixpoint.
fn resolve_substitutions(subs: &HashMap<ValueId, ValueId>) -> HashMap<ValueId, ValueId> {
    let mut resolved = HashMap::new();
    for (&from, &to) in subs {
        let mut current = to;
        while let Some(&next) = subs.get(&current) {
            current = next;
        }
        resolved.insert(from, current);
    }
    resolved
}

/// Rewrite every ValueId reference in instructions and terminators.
fn apply_substitutions(body: &mut FuncBody, subs: &HashMap<ValueId, ValueId>) {
    let sub = |v: &mut ValueId| {
        if let Some(&r) = subs.get(v) {
            *v = r;
        }
    };

    for inst in &mut body.insts {
        match &mut inst.kind {
            InstKind::BinOp { lhs, rhs, .. } => {
                sub(lhs);
                sub(rhs);
            }
            InstKind::UnaryOp { operand, .. } => sub(operand),
            InstKind::Tuple(elems) | InstKind::Vector(elems) | InstKind::ArrayLit { elements: elems } => {
                for e in elems {
                    sub(e);
                }
            }
            InstKind::Matrix(rows) => {
                for row in rows {
                    for e in row {
                        sub(e);
                    }
                }
            }
            InstKind::Project { base, .. } => sub(base),
            InstKind::Index { base, index } => {
                sub(base);
                sub(index);
            }
            InstKind::Call { args, .. } | InstKind::Intrinsic { args, .. } => {
                for a in args {
                    sub(a);
                }
            }
            InstKind::Load { ptr, .. } => sub(ptr),
            InstKind::Store { ptr, value, .. } => {
                sub(ptr);
                sub(value);
            }
            InstKind::ArrayRange { start, len, step } => {
                sub(start);
                sub(len);
                if let Some(s) = step {
                    sub(s);
                }
            }
            InstKind::StorageView { offset, len, .. } => {
                sub(offset);
                sub(len);
            }
            InstKind::StorageViewIndex { view, index } => {
                sub(view);
                sub(index);
            }
            InstKind::StorageViewLen { view } => sub(view),
            // No ValueId references in these variants
            InstKind::Int(_)
            | InstKind::Float(_)
            | InstKind::Bool(_)
            | InstKind::Unit
            | InstKind::String(_)
            | InstKind::Global(_)
            | InstKind::Extern(_)
            | InstKind::Alloca { .. }
            | InstKind::OutputPtr { .. } => {}
        }
    }

    for block in &mut body.blocks {
        if let Some(ref mut term) = block.terminator {
            match term {
                Terminator::Branch { args, .. } => {
                    for a in args {
                        sub(a);
                    }
                }
                Terminator::CondBranch {
                    cond,
                    then_args,
                    else_args,
                    ..
                } => {
                    sub(cond);
                    for a in then_args {
                        sub(a);
                    }
                    for a in else_args {
                        sub(a);
                    }
                }
                Terminator::Return(v) => sub(v),
                Terminator::ReturnUnit | Terminator::Unreachable => {}
            }
        }
    }
}

// =============================================================================
// Pass 2: Empty block elimination
// =============================================================================

fn eliminate_empty_blocks(body: &mut FuncBody) {
    // Collect all blocks that are merge or continue targets — these are
    // structurally required by SPIR-V and must not be eliminated.
    let mut protected: HashSet<BlockId> = HashSet::new();
    for block in body.blocks.iter() {
        if let Some(ref ctrl) = block.control {
            match ctrl {
                ControlHeader::Loop {
                    merge,
                    continue_block,
                } => {
                    protected.insert(*merge);
                    protected.insert(*continue_block);
                }
                ControlHeader::Selection { merge } => {
                    protected.insert(*merge);
                }
            }
        }
    }

    loop {
        let mut changed = false;

        // Collect redirect info for empty blocks:
        // (empty_block, target, branch_args, param_values)
        let mut redirects: Vec<(BlockId, BlockId, Vec<ValueId>, Vec<ValueId>)> = Vec::new();

        for (idx, block) in body.blocks.iter().enumerate() {
            let bid = BlockId(idx as u32);
            if bid == BlockId::ENTRY {
                continue;
            }
            if !block.insts.is_empty() {
                continue;
            }
            if block.control.is_some() {
                continue; // structured control-flow headers must stay
            }
            if protected.contains(&bid) {
                continue; // merge/continue targets must stay
            }
            if let Some(Terminator::Branch { target, args }) = &block.terminator {
                let param_values: Vec<ValueId> = block.params.iter().map(|p| p.value).collect();
                redirects.push((bid, *target, args.clone(), param_values));
            }
        }

        for (empty_id, target, branch_args, param_values) in &redirects {
            // Check that redirecting won't create a degenerate CondBranch
            // (both arms targeting the same block), which produces invalid
            // SPIR-V phi nodes with duplicate parent blocks.
            let safe = body.blocks.iter().all(|block| {
                if let Some(ref term) = block.terminator {
                    can_safely_redirect(term, *empty_id, *target)
                } else {
                    true
                }
            });
            if !safe {
                continue;
            }

            // Rewrite every predecessor that references the empty block
            for block in body.blocks.iter_mut() {
                if let Some(ref mut term) = block.terminator {
                    if rewrite_target(term, *empty_id, *target, branch_args, param_values) {
                        changed = true;
                    }
                }
            }

            // Mark eliminated block as unreachable (not None — SPIR-V lowering
            // iterates all blocks so they need valid terminators)
            body.blocks[empty_id.index()].terminator = Some(Terminator::Unreachable);
            body.blocks[empty_id.index()].params.clear();
        }

        if !changed {
            break;
        }
    }
}

/// Check whether redirecting `from` → `to` would cause a CondBranch
/// to have both arms targeting the same block (invalid in SPIR-V).
fn can_safely_redirect(term: &Terminator, from: BlockId, to: BlockId) -> bool {
    match term {
        Terminator::CondBranch {
            then_target,
            else_target,
            ..
        } => {
            let then_new = if *then_target == from { to } else { *then_target };
            let else_new = if *else_target == from { to } else { *else_target };
            then_new != else_new
        }
        _ => true,
    }
}

/// Rewrite occurrences of `from` → `to` inside a terminator,
/// threading block params through as needed. Returns true if rewritten.
fn rewrite_target(
    term: &mut Terminator,
    from: BlockId,
    to: BlockId,
    to_args: &[ValueId],
    from_params: &[ValueId],
) -> bool {
    let mut changed = false;

    match term {
        Terminator::Branch { target, args } if *target == from => {
            *args = substitute_through(to_args, from_params, args);
            *target = to;
            changed = true;
        }
        Terminator::CondBranch {
            then_target,
            then_args,
            else_target,
            else_args,
            ..
        } => {
            if *then_target == from {
                *then_args = substitute_through(to_args, from_params, then_args);
                *then_target = to;
                changed = true;
            }
            if *else_target == from {
                *else_args = substitute_through(to_args, from_params, else_args);
                *else_target = to;
                changed = true;
            }
        }
        _ => {}
    }

    changed
}

/// Map the empty block's outgoing args through its params.
///
/// `to_args`     — what the empty block passes to its target
/// `from_params` — the empty block's own param values
/// `pred_args`   — what the predecessor passes to the empty block
///
/// Each value in `to_args` that matches a `from_params[i]` is replaced
/// with `pred_args[i]`; other values pass through unchanged.
fn substitute_through(to_args: &[ValueId], from_params: &[ValueId], pred_args: &[ValueId]) -> Vec<ValueId> {
    if from_params.is_empty() {
        return to_args.to_vec();
    }

    let param_map: HashMap<ValueId, ValueId> =
        from_params.iter().zip(pred_args.iter()).map(|(&p, &a)| (p, a)).collect();

    to_args.iter().map(|v| param_map.get(v).copied().unwrap_or(*v)).collect()
}
