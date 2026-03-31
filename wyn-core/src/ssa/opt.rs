//! SSA peephole optimizations: param forwarding, project folding, CSE, DCE,
//! and empty block elimination.

use crate::ast::TypeName;
use crate::ssa::types::Program;
use crate::ssa::types::{BlockId, ControlHeader, FuncBody, InstKind, Terminator, ValueId, ValueRef};
use polytype::Type;
use std::collections::{HashMap, HashSet};

/// Run SSA peephole optimizations on the entire program.
pub fn optimize(mut program: Program) -> Program {
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
    project_fold(body);
    local_cse(body);
    dce(body);
    eliminate_empty_blocks(body);
}

// =============================================================================
// Pass 1: Single-predecessor param forwarding
// =============================================================================

fn forward_params(body: &mut FuncBody) {
    let mut pred_map: HashMap<BlockId, Vec<(BlockId, Vec<ValueId>)>> = HashMap::new();

    for (bid, block) in &body.inner.blocks {
        match &block.term {
            Terminator::Branch { target, args } => {
                pred_map.entry(*target).or_default().push((bid, args.clone()));
            }
            Terminator::CondBranch {
                then_target,
                then_args,
                else_target,
                else_args,
                ..
            } => {
                pred_map.entry(*then_target).or_default().push((bid, then_args.clone()));
                pred_map.entry(*else_target).or_default().push((bid, else_args.clone()));
            }
            _ => {}
        }
    }

    let mut substitutions: HashMap<ValueId, ValueId> = HashMap::new();
    let mut blocks_to_clear_params: Vec<BlockId> = Vec::new();
    let mut branches_to_clear_args: Vec<BlockId> = Vec::new();

    for (target, preds) in &pred_map {
        if preds.len() != 1 {
            continue;
        }
        let (pred_id, pred_args) = &preds[0];

        let is_unconditional = matches!(body.inner.blocks[*pred_id].term, Terminator::Branch { .. });
        if !is_unconditional {
            continue;
        }

        let target_block = &body.inner.blocks[*target];
        if target_block.params.is_empty() {
            continue;
        }

        // params are Vec<ValueId> — each param IS the value
        for (param, arg) in target_block.params.iter().zip(pred_args.iter()) {
            substitutions.insert(*param, *arg);
        }

        blocks_to_clear_params.push(*target);
        branches_to_clear_args.push(*pred_id);
    }

    if substitutions.is_empty() {
        return;
    }

    for bid in blocks_to_clear_params {
        body.inner.blocks[bid].params.clear();
    }
    for bid in branches_to_clear_args {
        if let Terminator::Branch { args, .. } = &mut body.inner.blocks[bid].term {
            args.clear();
        }
    }

    let resolved = resolve_substitutions(&substitutions);
    apply_substitutions(body, &resolved);
}

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

fn apply_substitutions(body: &mut FuncBody, subs: &HashMap<ValueId, ValueId>) {
    let mut sub = |vr: &mut ValueRef| {
        if let ValueRef::Ssa(id) = vr {
            if let Some(&r) = subs.get(id) {
                *vr = ValueRef::Ssa(r);
            }
        }
    };

    for (_iid, inst) in &mut body.inner.insts {
        inst.data.substitute_values(&mut sub);
    }

    for (_bid, block) in &mut body.inner.blocks {
        let rewritten = block.term.map_values(|v| if let Some(&r) = subs.get(&v) { r } else { v });
        block.term = rewritten;
    }
}

// =============================================================================
// Pass 2: Project folding — copy propagation through composites
// =============================================================================

fn project_fold(body: &mut FuncBody) {
    let mut def_map: HashMap<ValueId, wyn_ssa::InstId> = HashMap::new();
    for (iid, inst) in &body.inner.insts {
        if let Some(result) = inst.result {
            def_map.insert(result, iid);
        }
    }

    let mut substitutions: HashMap<ValueId, ValueId> = HashMap::new();

    for (_iid, inst) in &body.inner.insts {
        if let InstKind::Project { base, index } = &inst.data {
            if let Some(base_id) = base.as_ssa() {
                let mut resolved_base = base_id;
                while let Some(&next) = substitutions.get(&resolved_base) {
                    resolved_base = next;
                }

                if let Some(&def_iid) = def_map.get(&resolved_base) {
                    let elems = match &body.inner.insts[def_iid].data {
                        InstKind::Tuple(elems)
                        | InstKind::Vector(elems)
                        | InstKind::ArrayLit { elements: elems } => Some(elems),
                        _ => None,
                    };
                    if let Some(elems) = elems {
                        if let Some(elem) = elems.get(*index as usize) {
                            if let Some(elem_id) = elem.as_ssa() {
                                if let Some(result) = inst.result {
                                    substitutions.insert(result, elem_id);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if substitutions.is_empty() {
        return;
    }

    let resolved = resolve_substitutions(&substitutions);
    apply_substitutions(body, &resolved);
}

// =============================================================================
// Pass 3: Local CSE — per-block common subexpression elimination
// =============================================================================

fn local_cse(body: &mut FuncBody) {
    #[derive(Hash, Eq, PartialEq)]
    enum CseKey {
        Int(String, Type<TypeName>),
        Float(String, Type<TypeName>),
        Bool(bool),
        Global(String),
        Extern(String),
    }

    let mut substitutions: HashMap<ValueId, ValueId> = HashMap::new();

    for (_bid, block) in &body.inner.blocks {
        let mut seen: HashMap<CseKey, ValueId> = HashMap::new();

        for &inst_id in &block.insts {
            let inst = &body.inner.insts[inst_id];
            let result_ty = inst.result.map(|r| body.inner.value_type(r).clone());
            let key = match &inst.data {
                InstKind::Int(v) => result_ty.map(|ty| CseKey::Int(v.clone(), ty)),
                InstKind::Float(v) => result_ty.map(|ty| CseKey::Float(v.clone(), ty)),
                InstKind::Bool(v) => Some(CseKey::Bool(*v)),
                InstKind::Global(name) => Some(CseKey::Global(name.clone())),
                InstKind::Extern(name) => Some(CseKey::Extern(name.clone())),
                _ => None,
            };

            if let (Some(key), Some(result)) = (key, inst.result) {
                match seen.entry(key) {
                    std::collections::hash_map::Entry::Occupied(e) => {
                        substitutions.insert(result, *e.get());
                    }
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(result);
                    }
                }
            }
        }
    }

    if substitutions.is_empty() {
        return;
    }

    let resolved = resolve_substitutions(&substitutions);
    apply_substitutions(body, &resolved);
}

// =============================================================================
// Pass 4: Dead code elimination
// =============================================================================

fn dce(body: &mut FuncBody) {
    let mut used: HashSet<ValueId> = HashSet::new();

    for (_iid, inst) in &body.inner.insts {
        for v in inst.data.ssa_uses() {
            used.insert(v);
        }
    }

    for (_bid, block) in &body.inner.blocks {
        block.term.for_each_value(|v| {
            used.insert(v);
        });
    }

    // Can't borrow body.inner.blocks mutably while also reading body.inner.insts,
    // so collect inst ids to remove first.
    let mut to_remove: Vec<(BlockId, wyn_ssa::InstId)> = Vec::new();
    for (bid, block) in &body.inner.blocks {
        for &inst_id in &block.insts {
            let inst = &body.inner.insts[inst_id];
            if is_side_effecting(&inst.data) {
                continue;
            }
            match inst.result {
                Some(result) if !used.contains(&result) => {
                    to_remove.push((bid, inst_id));
                }
                None => {}
                _ => {}
            }
        }
    }

    for (_bid, inst_id) in to_remove {
        body.inner.remove_inst(inst_id);
    }
}

fn is_side_effecting(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::Call { .. }
            | InstKind::Intrinsic { .. }
            | InstKind::Alloca { .. }
            | InstKind::Load { .. }
            | InstKind::Store { .. }
            | InstKind::OutputPtr { .. }
            | InstKind::Soac(_)
    )
}

// =============================================================================
// Pass 5: Empty block elimination
// =============================================================================

fn eliminate_empty_blocks(body: &mut FuncBody) {
    let mut protected: HashSet<BlockId> = HashSet::new();
    for (&bid, ctrl) in &body.control_headers {
        let _ = bid;
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

    loop {
        let mut changed = false;

        let mut redirects: Vec<(BlockId, BlockId, Vec<ValueId>, Vec<ValueId>)> = Vec::new();

        for (bid, block) in &body.inner.blocks {
            if bid == body.entry_block() {
                continue;
            }
            if !block.insts.is_empty() {
                continue;
            }
            if body.control_headers.contains_key(&bid) {
                continue;
            }
            if protected.contains(&bid) {
                continue;
            }
            if let Terminator::Branch { target, args } = &block.term {
                let param_values: Vec<ValueId> = block.params.iter().copied().collect();
                redirects.push((bid, *target, args.clone(), param_values));
            }
        }

        for (empty_id, target, branch_args, param_values) in &redirects {
            let safe = body
                .inner
                .blocks
                .values()
                .all(|block| can_safely_redirect(&block.term, *empty_id, *target));
            if !safe {
                continue;
            }

            for (_bid, block) in &mut body.inner.blocks {
                if rewrite_target(&mut block.term, *empty_id, *target, branch_args, param_values) {
                    changed = true;
                }
            }

            body.inner.blocks[*empty_id].term = Terminator::Unreachable;
            body.inner.blocks[*empty_id].params.clear();
        }

        if !changed {
            break;
        }
    }
}

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

fn substitute_through(to_args: &[ValueId], from_params: &[ValueId], pred_args: &[ValueId]) -> Vec<ValueId> {
    if from_params.is_empty() {
        return to_args.to_vec();
    }

    let param_map: HashMap<ValueId, ValueId> =
        from_params.iter().zip(pred_args.iter()).map(|(&p, &a)| (p, a)).collect();

    to_args.iter().map(|v| param_map.get(v).copied().unwrap_or(*v)).collect()
}
