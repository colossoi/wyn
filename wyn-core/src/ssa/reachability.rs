//! Whole-module reachability for SSA definitions.
//!
//! Per-body elaboration is demand-driven, so it already omits unused
//! instructions. Late EGIR inlining can nevertheless leave complete function
//! or constant definitions with no path from an entry point. This phase walks
//! the final SSA call/global graph and removes those orphan definitions before
//! backend validation and emission.

use crate::ssa;
use std::collections::VecDeque;

use crate::op::OpTag;
use crate::ssa::types::{FuncBody, InstKind, Program};
use crate::{FunctionId, GlobalId, LookupMap, LookupSet};

#[cfg(test)]
#[path = "reachability_tests.rs"]
mod reachability_tests;

#[derive(Clone, Copy, Debug)]
enum Definition {
    Function(FunctionId),
    Constant(GlobalId),
}

/// Remove SSA function and constant definitions that cannot be reached from
/// any entry point.
///
/// Definition order is preserved. Only instructions in CFG-reachable blocks
/// contribute graph edges, so a stale call in a disconnected block does not
/// keep its target alive.
pub fn filter_reachable(program: ssa::stage::Elaborated) -> ssa::stage::Reachable {
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
    let constant_indices = program
        .constants
        .iter()
        .enumerate()
        .map(|(index, constant)| (constant.id, index))
        .collect::<LookupMap<_, _>>();

    let mut live_functions = LookupSet::new();
    let mut live_constants = LookupSet::new();
    let mut pending = VecDeque::new();

    for entry in &program.entry_points {
        enqueue_references(
            &entry.body,
            &mut live_functions,
            &mut live_constants,
            &mut pending,
        );
    }

    while let Some(definition) = pending.pop_front() {
        let body = match definition {
            Definition::Function(id) => {
                function_indices.get(&id).map(|&index| &program.functions[index].body)
            }
            Definition::Constant(id) => {
                constant_indices.get(&id).map(|&index| &program.constants[index].body)
            }
        };
        if let Some(body) = body {
            enqueue_references(body, &mut live_functions, &mut live_constants, &mut pending);
        }
    }

    program.functions.retain(|function| live_functions.contains(&function.id));
    program.constants.retain(|constant| live_constants.contains(&constant.id));
    program
}

fn enqueue_references(
    body: &FuncBody,
    live_functions: &mut LookupSet<FunctionId>,
    live_constants: &mut LookupSet<GlobalId>,
    pending: &mut VecDeque<Definition>,
) {
    for reference in body_references(body) {
        let newly_live = match reference {
            Definition::Function(id) => live_functions.insert(id),
            Definition::Constant(id) => live_constants.insert(id),
        };
        if newly_live {
            pending.push_back(reference);
        }
    }
}

fn body_references(body: &FuncBody) -> Vec<Definition> {
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
                } => references.push(Definition::Function(*function)),
                InstKind::Op {
                    tag: OpTag::Global(constant),
                    ..
                } => references.push(Definition::Constant(*constant)),
                _ => {}
            }
        }
        pending.extend(block.term.successors());
    }

    references
}
