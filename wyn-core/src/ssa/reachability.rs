//! Whole-module reachability for SSA definitions.
//!
//! Per-body elaboration is demand-driven, so it already omits unused
//! instructions. Late egglog inlining can nevertheless leave complete function
//! definitions with no path from an entry point. This phase walks
//! the final SSA call graph and removes those orphan definitions before
//! backend validation and emission.

use crate::ssa;
use std::collections::VecDeque;

use crate::op::OpTag;
use crate::ssa::types::{FuncBody, InstKind, Program};
use crate::{FunctionId, LookupMap, LookupSet};

#[cfg(test)]
#[path = "reachability_tests.rs"]
mod reachability_tests;

/// Remove SSA function definitions that cannot be reached from
/// any entry point.
///
/// Definition order is preserved. Only instructions in CFG-reachable blocks
/// contribute graph edges, so a stale call in a disconnected block does not
/// keep its target alive.
pub fn filter_reachable(program: ssa::stage::Placed) -> ssa::stage::Reachable {
    retain_reachable(program).retag()
}

fn retain_reachable<Tag, GlobalContext>(
    mut program: Program<Tag, GlobalContext>,
) -> Program<Tag, GlobalContext> {
    let function_indices = program
        .functions
        .iter()
        .enumerate()
        .map(|(index, function)| (function.id, index))
        .collect::<LookupMap<_, _>>();
    let mut live_functions = LookupSet::new();
    let mut pending: VecDeque<_> =
        program.entry_points.iter().flat_map(|entry| body_references(&entry.body)).collect();
    while let Some(id) = pending.pop_front() {
        if live_functions.insert(id) {
            if let Some(&index) = function_indices.get(&id) {
                pending.extend(body_references(&program.functions[index].body));
            }
        }
    }
    program.functions.retain(|function| live_functions.contains(&function.id));
    program
}

fn body_references(body: &FuncBody) -> Vec<FunctionId> {
    let mut references = Vec::new();
    let mut visited = LookupSet::new();
    let mut pending = VecDeque::from([body.inner.entry]);

    while let Some(block_id) = pending.pop_front() {
        if !visited.insert(block_id) {
            continue;
        }
        let Some(block) = body.inner.blocks.get(block_id) else {
            continue;
        };
        for &inst_id in &block.insts {
            let Some(instruction) = body.inner.insts.get(inst_id) else {
                continue;
            };
            match &instruction.data {
                InstKind::Op {
                    tag: OpTag::Call(function),
                    ..
                } => references.push(*function),
                _ => {}
            }
        }
        pending.extend(block.term.successors());
    }

    references
}
