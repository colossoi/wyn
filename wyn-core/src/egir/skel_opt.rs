//! Skeleton-level CFG rewrites run between canonicalize and elaborate.
//!
//! Rewrites operate in "NodeId land": terminator args and block params are
//! `NodeId`s, not `ValueId`s, which makes the rewrites cleanly composable
//! with the hash-consed sea of nodes.
//!
//! Rewrites today:
//! - `fold_constant_branches`: `CondBranch { cond: literal_bool, ... }` →
//!   `Branch` to the chosen arm.
//! - `remove_unreachable_blocks`: after branch folding, drop skeleton
//!   blocks no longer reachable from the entry block.
//! - `eliminate_redundant_params`: block params whose every incoming arg
//!   is the same NodeId are stripped from the block's param list and
//!   from every predecessor's branch args; the stripped param is aliased
//!   to the common incoming NodeId and returned for the elaborator to
//!   merge into its `best` map.
//!
//! ## Invariants preserved
//!
//! 1. Terminator arg counts always equal target block param counts.
//! 2. `aliases` keys are exclusively block-param NodeIds that have been
//!    removed from their owning block's `params` list.
//! 3. `aliases` values are never themselves aliases (closure is walked
//!    before the map is returned).

use crate::flow::BlockId;
use crate::{LookupMap, LookupSet, SortedSet};

use crate::ssa::types::ConstantValue;

use super::program::PhysicalProgram;
use super::types::{EGraph, ENode, EgirPhase, NodeId, PureOp, SkeletonTerminator};

/// Run skeleton rewrites on every function and entry point in the program,
/// extending each body's alias map with the freshly stripped block params.
pub fn run(inner: &mut PhysicalProgram) {
    for f in &mut inner.functions {
        let new_aliases = run_one_body(&mut f.graph);
        f.aliases.extend(new_aliases);
    }
    for e in &mut inner.entry_points {
        let new_aliases = run_one_body(&mut e.graph);
        e.aliases.extend(new_aliases);
    }
}

/// Run all enabled skeleton rewrites to fixpoint. Returns an alias map
/// mapping stripped block-param NodeIds to their replacement NodeIds.
pub fn run_one_body<P: EgirPhase>(graph: &mut EGraph<P>) -> LookupMap<NodeId, NodeId> {
    let mut aliases: LookupMap<NodeId, NodeId> = LookupMap::new();
    loop {
        // Phase order: fold first, prune dead CFG second, phi-elim third.
        // Folding can expose unreachable arms and shrink a
        // block's predecessor set (CondBranch → Branch), newly exposing an
        // "all incoming args are the same NodeId" situation.
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
fn fold_constant_branches<P: EgirPhase>(graph: &mut EGraph<P>) -> bool {
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
            if let Some(b) = is_const_bool(*cond, &graph.nodes) {
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
/// - `ENode::Constant(ConstantValue::Bool(_))` — from canonicalize of
///   `ValueRef::Const(Bool _)`.
/// - `ENode::Pure { op: PureOp::Bool(_), operands: [] }` — from TLC's
///   `BoolLit` or canonicalize of `InstKind::Bool`.
///
/// We do not consult the `best` map (doesn't exist yet at this stage).
/// Only literal constants are recognized; union-extract winners are out
/// of scope.
fn is_const_bool<R>(nid: NodeId, nodes: &slotmap::SlotMap<NodeId, ENode<R>>) -> Option<bool> {
    match &nodes[nid] {
        ENode::Constant(ConstantValue::Bool(b)) => Some(*b),
        ENode::Pure {
            op: PureOp::Bool(b),
            operands,
        } if operands.is_empty() => Some(*b),
        _ => None,
    }
}

/// Remove skeleton blocks that cannot be reached from the entry block.
/// The pure nodes and block-param nodes owned by dead blocks stay in the
/// sea; with no reachable demand path to them, later stages ignore them.
fn remove_unreachable_blocks<P: EgirPhase>(graph: &mut EGraph<P>) -> bool {
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

/// Eliminate every block param whose incoming arg is the same NodeId on
/// every predecessor branch. Returns a map from stripped param NodeId
/// to its replacement NodeId.
///
/// Equality is strict NodeId equality — we deliberately do not consult
/// the `best` map. Hash-consing at intern time should have already
/// dedup'd structurally-equal subtrees; mixing CFG rewriting with
/// e-graph equivalence reasoning is where subtle bugs live.
fn eliminate_redundant_params<P: EgirPhase>(graph: &mut EGraph<P>) -> LookupMap<NodeId, NodeId> {
    use smallvec::SmallVec;

    // incoming[B][i] = every distinct NodeId passed into B.params[i] by
    // some predecessor branch terminator. We only need to know "is the
    // set of size 1?", so track up to two distinct values.
    let mut incoming: LookupMap<BlockId, Vec<SmallVec<[NodeId; 2]>>> = LookupMap::new();
    for (bid, block) in &graph.skeleton.blocks {
        let mut per_param = Vec::with_capacity(block.params.len());
        per_param.resize(block.params.len(), SmallVec::<[NodeId; 2]>::new());
        incoming.insert(bid, per_param);
    }

    let collect = |target: BlockId,
                   args: &[NodeId],
                   incoming: &mut LookupMap<BlockId, Vec<SmallVec<[NodeId; 2]>>>| {
        let slots = incoming.get_mut(&target).expect("target in skeleton");
        debug_assert_eq!(
            slots.len(),
            args.len(),
            "arity mismatch at branch to {:?}",
            target
        );
        for (i, &arg) in args.iter().enumerate() {
            if !slots[i].contains(&arg) && slots[i].len() < 2 {
                slots[i].push(arg);
            }
        }
    };

    for (_bid, block) in &graph.skeleton.blocks {
        match &block.term {
            SkeletonTerminator::Branch { target, args } => collect(*target, args, &mut incoming),
            SkeletonTerminator::CondBranch {
                then_target,
                then_args,
                else_target,
                else_args,
                ..
            } => {
                collect(*then_target, then_args, &mut incoming);
                collect(*else_target, else_args, &mut incoming);
            }
            SkeletonTerminator::Return(_) | SkeletonTerminator::Unreachable => {}
        }
    }

    // Determine which params are redundant. Skip the entry block: it has
    // no predecessors and its params (when present) come from the source
    // function params, not from a branch.
    let entry = graph.skeleton.entry;
    let mut redundant: LookupMap<BlockId, Vec<(usize, NodeId)>> = LookupMap::new();
    let mut aliases: LookupMap<NodeId, NodeId> = LookupMap::new();
    for (bid, block) in &graph.skeleton.blocks {
        if bid == entry {
            continue;
        }
        let slots = incoming.get(&bid).expect("slots initialized");
        for (i, param) in block.params.iter().enumerate() {
            // slots[i].len() meanings:
            //   0 = this block is unreachable (no predecessor terminator
            //       branches here). Branch-folding in the same fixpoint
            //       loop can newly make blocks unreachable, so this is
            //       a legitimate state, not an invariant violation. Leave
            //       the param alone; elaborate's domtree walk won't visit
            //       unreachable blocks anyway.
            //   1 = every predecessor passes the same value → redundancy
            //       candidate.
            //   2 = multiple distinct incoming values → real phi.
            if slots[i].len() != 1 {
                continue;
            }
            let x = slots[i][0];
            if x == *param {
                // Self-referential (e.g., a block that branches to
                // itself as its only predecessor and feeds its own param
                // back). Not a valid redundancy.
                continue;
            }
            redundant.entry(bid).or_default().push((i, *param));
            aliases.insert(*param, x);
        }
    }

    // Apply removals while keeping each block's parameter list and every
    // incoming branch's argument list in sync.
    if redundant.is_empty() {
        return aliases;
    }

    for (bid, slots) in &redundant {
        let slots = slots.iter().map(|(index, _)| *index).collect::<SortedSet<_>>();
        graph.remove_block_param_slots(*bid, &slots);
    }

    // Invariant 1 from the plan: every branch's arg count must equal its
    // target block's param count. This is the exact class of bug that
    // the descending-index sweep is designed to preserve; assert it.
    debug_assert!(
        check_branch_arity(graph),
        "branch/param arity mismatch after eliminate_redundant_params"
    );

    aliases
}

/// Debug check: for every terminator branching to a block, the arg list
/// must have the same length as the target block's param list.
fn check_branch_arity<P: EgirPhase>(graph: &EGraph<P>) -> bool {
    for (_bid, block) in &graph.skeleton.blocks {
        match &block.term {
            SkeletonTerminator::Branch { target, args } => {
                if args.len() != graph.skeleton.blocks[*target].params.len() {
                    return false;
                }
            }
            SkeletonTerminator::CondBranch {
                then_target,
                then_args,
                else_target,
                else_args,
                ..
            } => {
                if then_args.len() != graph.skeleton.blocks[*then_target].params.len() {
                    return false;
                }
                if else_args.len() != graph.skeleton.blocks[*else_target].params.len() {
                    return false;
                }
            }
            SkeletonTerminator::Return(_) | SkeletonTerminator::Unreachable => {}
        }
    }
    true
}

/// Insert every entry from `new_aliases` into `aliases`, and partially
/// forward existing alias targets through one hop of the new map.
///
/// This is intentionally a *partial* forward: the final single-source-
/// of-truth closure is done by `close_aliases` after the last iteration
/// of the fixpoint loop. Re-running closure on every iteration would be
/// wasted work.
fn merge_aliases(aliases: &mut LookupMap<NodeId, NodeId>, new_aliases: LookupMap<NodeId, NodeId>) {
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
fn close_aliases(aliases: &mut LookupMap<NodeId, NodeId>) {
    let keys: Vec<NodeId> = aliases.keys().copied().collect();
    for k in keys {
        let mut visited: LookupSet<NodeId> = LookupSet::new();
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
