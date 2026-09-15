//! Skeleton-level CFG rewrites run between canonicalize and elaborate.
//!
//! Rewrites preserve block-interface arity and compose eliminated parameters'
//! replacements into a forest whose export points directly to surviving values.
//!
//! - `fold_constant_branches`: `CondBranch { cond: literal_bool, ... }` →
//!   `Branch` to the chosen arm.
//! - `remove_unreachable_blocks`: after branch folding, drop skeleton
//!   blocks no longer reachable from the entry block.
//! - `eliminate_redundant_params`: block params whose every incoming arg
//!   is the same ValueId are stripped from the block's param list and
//!   from every predecessor's branch args; the stripped param is aliased
//!   to the common incoming ValueId.

/// Physical EGIR after skeleton control-flow simplification.
#[derive(Debug, Clone, Copy)]
pub enum SkeletonOptimizedTag {}
pub type SkeletonOptimized = super::program::PhysicalProgram<SkeletonOptimizedTag>;

use crate::LookupMap;
use wyn_graph::ReplacementForest;

use crate::ssa::types::ConstantValue;

use super::types::{EGraph, Family, PureOp, SkeletonTerminator, ValueId, ValueKind};

/// Run skeleton rewrites on every body and attach each eliminated block
/// parameter's canonical replacement directly to that node.
pub fn optimize_skeleton(program: super::rewrite::Rewritten) -> SkeletonOptimized {
    program
        .map_graphs(|_, mut graph| {
            let aliases = run_one_body(&mut graph);
            graph.install_aliases(aliases);
            super::flow_liveness::prune_unused_allocations(&mut graph);
            graph
        })
        .retag_physical()
}

/// Run all enabled skeleton rewrites to fixpoint. Returns an alias map
/// mapping stripped block-param ValueNodeIds to their replacement ValueNodeIds.
pub fn run_one_body<P: Family>(graph: &mut EGraph<P>) -> LookupMap<ValueId, ValueId> {
    let mut replacements = ReplacementForest::new();
    loop {
        // Phase order: fold first, prune dead CFG second, phi-elim third.
        // Folding can expose unreachable arms and shrink a
        // block's predecessor set (CondBranch → Branch), newly exposing an
        // "all incoming args are the same ValueId" situation.
        let folded = fold_constant_branches(graph);
        let pruned = remove_unreachable_blocks(graph);
        let phi_elided = eliminate_redundant_params(graph, &mut replacements);
        if !folded && !pruned && !phi_elided {
            break;
        }
    }
    replacements.into_map()
}

/// Rewrite every `CondBranch` whose condition is a literal bool into a
/// direct `Branch` to the chosen arm. Returns true if any block's
/// terminator was rewritten.
fn fold_constant_branches<P: Family>(graph: &mut EGraph<P>) -> bool {
    let mut changed = false;
    for block in graph.skeleton.blocks.values_mut() {
        if let SkeletonTerminator::CondBranch {
            cond,
            then_target,
            then_args,
            else_target,
            else_args,
        } = &block.term
        {
            if let Some(b) = is_const_bool(&graph.nodes[*cond].kind) {
                let (target, args) = if b { (then_target, then_args) } else { (else_target, else_args) };
                block.term = SkeletonTerminator::Branch {
                    target: *target,
                    args: args.clone(),
                };
                changed = true;
            }
        }
    }
    changed
}

/// Recognize literal booleans in constant and pure-operation nodes.
fn is_const_bool<R>(kind: &ValueKind<R>) -> Option<bool> {
    match kind {
        ValueKind::Constant(ConstantValue::Bool(b)) => Some(*b),
        ValueKind::Pure {
            op: PureOp::Bool(b),
            operands,
        } if operands.is_empty() => Some(*b),
        _ => None,
    }
}

/// Remove skeleton blocks that cannot be reached from the entry block.
/// The pure nodes and block-param nodes owned by dead blocks stay in the
/// sea; with no reachable demand path to them, later stages ignore them.
fn remove_unreachable_blocks<P: Family>(graph: &mut EGraph<P>) -> bool {
    let reachable = wyn_graph::reachable_set(
        [graph.skeleton.entry],
        wyn_graph::WalkOrder::DepthFirst,
        |bid, out| {
            if let Some(block) = graph.skeleton.blocks.get(bid) {
                out.extend(block.term.successors());
            }
        },
    );

    let before = graph.skeleton.blocks.len();
    graph.skeleton.blocks.retain(|bid, _| reachable.contains(&bid));
    graph.skeleton.blocks.len() != before
}

/// Eliminate every block param whose incoming arg is the same ValueId on
/// every predecessor branch. Admit replacements before removing parameter slots.
///
/// Candidates require exact ValueId equality across incoming arguments.
fn eliminate_redundant_params<P: Family>(
    graph: &mut EGraph<P>,
    replacements: &mut ReplacementForest<ValueId>,
) -> bool {
    let entry = graph.skeleton.entry;
    let removed = super::block_interface::select_columns(graph, |block, interface| {
        let mut retained = Vec::new();
        for (slot, column) in interface.columns().iter().enumerate() {
            let parameter = column.parameter().value();
            let common = column.common_argument().map(|argument| argument.value());
            if let Some(value) = common.filter(|&value| block != entry && value != parameter) {
                replacements
                    .replace(parameter, value)
                    .unwrap_or_else(|error| panic!("invalid skeleton parameter replacement: {error}"));
            } else {
                retained.push(slot);
            }
        }
        Ok(retained)
    })
    .expect("valid skeleton interfaces");
    !removed.is_empty()
}

#[cfg(test)]
#[path = "skel_opt_tests.rs"]
mod skel_opt_tests;
