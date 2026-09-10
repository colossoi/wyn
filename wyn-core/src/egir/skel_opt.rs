//! Skeleton-level CFG rewrites run between canonicalize and elaborate.
//!
//! Rewrites operate in "ValueId land": terminator args and block params are
//! `ValueId`s, not `ValueId`s, which makes the rewrites cleanly composable
//! with the hash-consed sea of nodes.
//!
//! Rewrites today:
//! - `fold_constant_branches`: `CondBranch { cond: literal_bool, ... }` →
//!   `Branch` to the chosen arm.
//! - `remove_unreachable_blocks`: after branch folding, drop skeleton
//!   blocks no longer reachable from the entry block.
//! - `eliminate_redundant_params`: block params whose every incoming arg
//!   is the same ValueId are stripped from the block's param list and
//!   from every predecessor's branch args; the stripped param is aliased
//!   to the common incoming ValueId and returned for the elaborator to
//!   merge into its `best` map.
//!
//! ## Invariants preserved
//!
//! 1. Terminator arg counts always equal target block param counts.
//! 2. `aliases` keys are exclusively block-param ValueNodeIds that have been
//!    removed from their owning block's `params` list.
//! 3. `aliases` values are never themselves aliases (closure is walked
//!    before the map is returned).

/// Physical EGIR after skeleton control-flow simplification.
#[derive(Debug, Clone, Copy)]
pub enum SkeletonOptimizedTag {}
pub type SkeletonOptimized = super::program::PhysicalProgram<SkeletonOptimizedTag>;

use crate::flow::BlockId;
use crate::{LookupMap, LookupSet};

use crate::ssa::types::ConstantValue;

use super::types::{EGraph, Family, PureOp, SkeletonTerminator, ValueId, ValueKind};

/// Run skeleton rewrites on every body and attach each eliminated block
/// parameter's canonical replacement directly to that node.
pub fn optimize_skeleton(program: super::rewrite::Rewritten) -> SkeletonOptimized {
    program
        .map_graphs(|_, mut graph| {
            let aliases = run_one_body(&mut graph);
            graph.install_aliases(aliases);
            graph
        })
        .retag_physical()
}

/// Run all enabled skeleton rewrites to fixpoint. Returns an alias map
/// mapping stripped block-param ValueNodeIds to their replacement ValueNodeIds.
pub fn run_one_body<P: Family>(graph: &mut EGraph<P>) -> LookupMap<ValueId, ValueId> {
    let mut aliases: LookupMap<ValueId, ValueId> = LookupMap::new();
    loop {
        // Phase order: fold first, prune dead CFG second, phi-elim third.
        // Folding can expose unreachable arms and shrink a
        // block's predecessor set (CondBranch → Branch), newly exposing an
        // "all incoming args are the same ValueId" situation.
        let folded = fold_constant_branches(graph);
        let pruned = remove_unreachable_blocks(graph);
        let new_aliases = eliminate_redundant_params(graph);
        let phi_elided = !new_aliases.is_empty();
        merge_aliases(&mut aliases, new_aliases);
        if !folded && !pruned && !phi_elided {
            break;
        }
    }
    close_aliases(&mut aliases);
    debug_assert!(
        aliases.values().all(|v| !aliases.contains_key(v)),
        "alias closure violated: {:?}",
        aliases,
    );
    aliases
}

/// Rewrite every `CondBranch` whose condition is a literal bool into a
/// direct `Branch` to the chosen arm. Returns true if any block's
/// terminator was rewritten.
fn fold_constant_branches<P: Family>(graph: &mut EGraph<P>) -> bool {
    // Collect rewrites first so we don't hold a borrow of graph.nodes
    // while mutating skeleton.blocks.
    let mut rewrites: Vec<(BlockId, SkeletonTerminator)> = Vec::new();
    for (bid, block) in &graph.skeleton.blocks {
        if let SkeletonTerminator::CondBranch {
            cond,
            then_target,
            then_args,
            else_target,
            else_args,
        } = &block.term
        {
            if let Some(b) = is_const_bool(*cond, graph) {
                let new_term = if b {
                    SkeletonTerminator::Branch {
                        target: *then_target,
                        args: then_args.clone(),
                    }
                } else {
                    SkeletonTerminator::Branch {
                        target: *else_target,
                        args: else_args.clone(),
                    }
                };
                rewrites.push((bid, new_term));
            }
        }
    }
    let changed = !rewrites.is_empty();
    for (bid, term) in rewrites {
        graph.skeleton.blocks[bid].term = term;
    }
    changed
}

/// Does `nid` resolve to a literal boolean? Bools appear in two forms:
/// - `ValueKind::Constant(ConstantValue::Bool(_))` — from canonicalize of
///   `ValueRef::Const(Bool _)`.
/// - `ValueKind::Pure { op: PureOp::Bool(_), operands: [] }` — from TLC's
///   `BoolLit` or canonicalize of `InstKind::Bool`.
///
/// We do not consult the `best` map (doesn't exist yet at this stage).
/// Only literal constants are recognized; union-extract winners are out
/// of scope.
fn is_const_bool<P: Family>(nid: ValueId, graph: &EGraph<P>) -> Option<bool> {
    match &graph.nodes[nid].kind {
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
    let reachable: LookupSet<BlockId> = wyn_graph::reachable_set(
        [graph.skeleton.entry],
        wyn_graph::WalkOrder::DepthFirst,
        |bid, out| {
            let Some(block) = graph.skeleton.blocks.get(bid) else {
                return;
            };
            match &block.term {
                SkeletonTerminator::Branch { target, .. } => out.push(*target),
                SkeletonTerminator::CondBranch {
                    then_target,
                    else_target,
                    ..
                } => {
                    out.push(*then_target);
                    out.push(*else_target);
                }
                SkeletonTerminator::Return(_) | SkeletonTerminator::Unreachable => {}
            }
        },
    );

    let dead: Vec<BlockId> = graph.skeleton.blocks.keys().filter(|bid| !reachable.contains(bid)).collect();
    if dead.is_empty() {
        return false;
    }
    for bid in dead {
        graph.skeleton.blocks.remove(bid);
    }
    true
}

/// Eliminate every block param whose incoming arg is the same ValueId on
/// every predecessor branch. Returns a map from stripped param ValueId
/// to its replacement ValueId.
///
/// Equality is strict ValueId equality — we deliberately do not consult
/// the `best` map. Hash-consing at intern time should have already
/// dedup'd structurally-equal subtrees; mixing CFG rewriting with
/// e-graph equivalence reasoning is where subtle bugs live.
fn eliminate_redundant_params<P: Family>(graph: &mut EGraph<P>) -> LookupMap<ValueId, ValueId> {
    let entry = graph.skeleton.entry;
    let mut aliases = LookupMap::new();
    super::block_interface::select_columns(graph, |block, interface| {
        Ok(interface
            .columns()
            .iter()
            .enumerate()
            .filter_map(|(slot, column)| {
                let parameter = column.parameter().value();
                if block != entry {
                    if let Some(value) = column.common_argument().map(|argument| argument.value()) {
                        if value != parameter {
                            aliases.insert(parameter, value);
                            return None;
                        }
                    }
                }
                Some(slot)
            })
            .collect())
    })
    .expect("valid skeleton interfaces");
    aliases
}
/// Insert every entry from `new_aliases` into `aliases`, and partially
/// forward existing alias targets through one hop of the new map.
///
/// This is intentionally a *partial* forward: the final single-source-
/// of-truth closure is done by `close_aliases` after the last iteration
/// of the fixpoint loop. Re-running closure on every iteration would be
/// wasted work.
fn merge_aliases(aliases: &mut LookupMap<ValueId, ValueId>, new_aliases: LookupMap<ValueId, ValueId>) {
    for (_k, v) in aliases.iter_mut() {
        if let Some(&forwarded) = new_aliases.get(v) {
            *v = forwarded;
        }
    }
    for (k, v) in new_aliases {
        aliases.insert(k, v);
    }
}

/// Walk the alias map to closure: for every key `k`, follow
/// `aliases[aliases[k]]...` until a non-aliased value is reached.
/// Union-find-style path compression.
///
/// Alias cycles are a logic bug (every alias points to a value defined
/// earlier in SSA order; cycles imply we tried to alias a
/// live value to something that depends on itself). On detection we
/// panic so the upstream bug surfaces loudly.
fn close_aliases(aliases: &mut LookupMap<ValueId, ValueId>) {
    let keys: Vec<ValueId> = aliases.keys().copied().collect();
    for k in keys {
        let mut visited: LookupSet<ValueId> = LookupSet::new();
        visited.insert(k);
        let mut cur = aliases[&k];
        while let Some(&next) = aliases.get(&cur) {
            if !visited.insert(cur) {
                panic!(
                    "alias cycle detected: node {:?} reached twice while closing aliases starting at {:?}",
                    cur, k
                );
            }
            cur = next;
        }
        aliases.insert(k, cur);
    }
}

#[cfg(test)]
#[path = "skel_opt_tests.rs"]
mod skel_opt_tests;
