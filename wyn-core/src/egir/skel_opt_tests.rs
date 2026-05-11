//! Unit tests for the skel_opt module.

use super::*;
use crate::ast::TypeName;
use crate::egir::types::{EGraph, NodeId, PureOp, SkeletonTerminator};
use crate::ssa::framework::BlockId;
use crate::ssa::types::ConstantValue;
use polytype::Type;
use smallvec::smallvec;
use std::collections::HashMap;

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}
fn bool_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Bool, vec![])
}

// -- is_const_bool -------------------------------------------------------

#[test]
fn is_const_bool_recognizes_constant_true() {
    let mut graph = EGraph::new();
    let b = graph.intern_constant(ConstantValue::Bool(true), bool_ty());
    assert_eq!(is_const_bool(b, &graph.nodes), Some(true));
}

#[test]
fn is_const_bool_recognizes_constant_false() {
    let mut graph = EGraph::new();
    let b = graph.intern_constant(ConstantValue::Bool(false), bool_ty());
    assert_eq!(is_const_bool(b, &graph.nodes), Some(false));
}

#[test]
fn is_const_bool_recognizes_pure_bool() {
    let mut graph = EGraph::new();
    let b = graph.intern_pure(PureOp::Bool(true), smallvec![], bool_ty());
    assert_eq!(is_const_bool(b, &graph.nodes), Some(true));
}

#[test]
fn is_const_bool_rejects_non_bool() {
    let mut graph = EGraph::new();
    let n = graph.intern_pure(PureOp::Int("42".into()), smallvec![], i32_ty());
    assert_eq!(is_const_bool(n, &graph.nodes), None);
}

// -- fold_constant_branches ---------------------------------------------

/// Build a `CondBranch { cond, ... }` skeleton into `graph`'s entry
/// block where then/else are fresh sibling blocks each terminating
/// in `Return(None)`. Returns (entry, then_bid, else_bid).
fn build_condbranch_skel(
    graph: &mut EGraph,
    cond: NodeId,
    then_arg: Option<NodeId>,
    else_arg: Option<NodeId>,
) -> (BlockId, BlockId, BlockId) {
    let entry = graph.skeleton.entry;
    let then_bid = graph.skeleton.create_block();
    let else_bid = graph.skeleton.create_block();
    graph.skeleton.blocks[then_bid].term = SkeletonTerminator::Return(None);
    graph.skeleton.blocks[else_bid].term = SkeletonTerminator::Return(None);
    let then_args = then_arg.map_or_else(Vec::new, |a| vec![a]);
    let else_args = else_arg.map_or_else(Vec::new, |a| vec![a]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: then_bid,
        then_args,
        else_target: else_bid,
        else_args,
    };
    (entry, then_bid, else_bid)
}

#[test]
fn fold_constant_branch_true_selects_then() {
    let mut graph = EGraph::new();
    let t = graph.intern_constant(ConstantValue::Bool(true), bool_ty());
    let (entry, then_bid, _else_bid) = build_condbranch_skel(&mut graph, t, None, None);
    let changed = fold_constant_branches(&mut graph);
    assert!(changed);
    match &graph.skeleton.blocks[entry].term {
        SkeletonTerminator::Branch { target, .. } => assert_eq!(*target, then_bid),
        other => panic!("expected Branch, got {:?}", other),
    }
}

#[test]
fn fold_constant_branch_false_selects_else() {
    let mut graph = EGraph::new();
    let f = graph.intern_constant(ConstantValue::Bool(false), bool_ty());
    let (entry, _then_bid, else_bid) = build_condbranch_skel(&mut graph, f, None, None);
    let changed = fold_constant_branches(&mut graph);
    assert!(changed);
    match &graph.skeleton.blocks[entry].term {
        SkeletonTerminator::Branch { target, .. } => assert_eq!(*target, else_bid),
        other => panic!("expected Branch, got {:?}", other),
    }
}

#[test]
fn fold_constant_branch_nonconst_left_alone() {
    let mut graph = EGraph::new();
    let cond = graph.add_func_param(0, bool_ty());
    let (entry, _then_bid, _else_bid) = build_condbranch_skel(&mut graph, cond, None, None);
    let changed = fold_constant_branches(&mut graph);
    assert!(!changed);
    assert!(matches!(
        &graph.skeleton.blocks[entry].term,
        SkeletonTerminator::CondBranch { .. }
    ));
}

#[test]
fn fold_constant_branch_preserves_chosen_args() {
    let mut graph = EGraph::new();
    let t = graph.intern_constant(ConstantValue::Bool(true), bool_ty());
    let x = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty());
    let y = graph.intern_pure(PureOp::Int("2".into()), smallvec![], i32_ty());
    let (entry, _then_bid, _else_bid) = build_condbranch_skel(&mut graph, t, Some(x), Some(y));
    fold_constant_branches(&mut graph);
    match &graph.skeleton.blocks[entry].term {
        SkeletonTerminator::Branch { args, .. } => {
            assert_eq!(args.as_slice(), &[x], "should carry then_args only");
        }
        other => panic!("{:?}", other),
    }
}

// -- eliminate_redundant_params -----------------------------------------

/// entry -CondBranch-> b1|b2; b1 -> merge(left); b2 -> merge(right); merge
/// returns its single param. If left == right, phi-elim should fire.
fn build_merge_skel(
    graph: &mut EGraph,
    merge_arg_left: NodeId,
    merge_arg_right: NodeId,
) -> (BlockId, BlockId, NodeId) {
    let entry = graph.skeleton.entry;
    let b1 = graph.skeleton.create_block();
    let b2 = graph.skeleton.create_block();
    let merge = graph.skeleton.create_block();

    let param = graph.add_block_param(merge, 0, i32_ty());
    graph.skeleton.blocks[merge].params.push(param);
    graph.skeleton.blocks[merge].term = SkeletonTerminator::Return(Some(param));

    let cond = graph.add_func_param(0, bool_ty());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: b1,
        then_args: vec![],
        else_target: b2,
        else_args: vec![],
    };
    graph.skeleton.blocks[b1].term = SkeletonTerminator::Branch {
        target: merge,
        args: vec![merge_arg_left],
    };
    graph.skeleton.blocks[b2].term = SkeletonTerminator::Branch {
        target: merge,
        args: vec![merge_arg_right],
    };
    (merge, b2, param)
}

#[test]
fn phi_elim_strips_param_with_matching_incoming() {
    let mut graph = EGraph::new();
    let x = graph.intern_pure(PureOp::Int("7".into()), smallvec![], i32_ty());
    let (merge, _b2, param) = build_merge_skel(&mut graph, x, x);

    let aliases = eliminate_redundant_params(&mut graph);

    assert_eq!(aliases.get(&param), Some(&x));
    assert!(
        graph.skeleton.blocks[merge].params.is_empty(),
        "param should be stripped"
    );
    for (_bid, block) in &graph.skeleton.blocks {
        if let SkeletonTerminator::Branch { target, args } = &block.term {
            if *target == merge {
                assert!(args.is_empty(), "branch arg slot should be stripped");
            }
        }
    }
}

#[test]
fn phi_elim_preserves_param_with_differing_incoming() {
    // Loop-accumulator shape: preheader X, backedge Y.
    let mut graph = EGraph::new();
    let x = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty());
    let y = graph.intern_pure(PureOp::Int("2".into()), smallvec![], i32_ty());
    let (merge, _b2, param) = build_merge_skel(&mut graph, x, y);

    let aliases = eliminate_redundant_params(&mut graph);

    assert!(aliases.is_empty(), "differing incoming must not alias");
    assert_eq!(graph.skeleton.blocks[merge].params.len(), 1);
    assert!(graph.nodes.contains_key(param), "param node still present");
}

#[test]
fn phi_elim_rejects_self_referential_param() {
    // Block B takes param p, and B's only incoming arg IS p (self-loop).
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let b = graph.skeleton.create_block();
    let param = graph.add_block_param(b, 0, i32_ty());
    graph.skeleton.blocks[b].params.push(param);
    let x = graph.intern_pure(PureOp::Int("5".into()), smallvec![], i32_ty());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: b,
        args: vec![x],
    };
    graph.skeleton.blocks[b].term = SkeletonTerminator::Branch {
        target: b,
        args: vec![param],
    };

    let aliases = eliminate_redundant_params(&mut graph);

    // Incoming set is {x, param}: two distinct values → not redundant.
    // Even if it were {param} alone, self-alias is rejected.
    assert!(aliases.is_empty());
}

// -- merge_aliases / close_aliases --------------------------------------

fn mk_nid(graph: &mut EGraph) -> NodeId {
    graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty())
}

#[test]
fn merge_aliases_forwards_existing_through_new() {
    let mut graph = EGraph::new();
    let a = mk_nid(&mut graph);
    let b = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty());
    let c = graph.intern_pure(PureOp::Int("2".into()), smallvec![], i32_ty());
    let mut aliases = HashMap::new();
    aliases.insert(a, b);
    let mut new_aliases = HashMap::new();
    new_aliases.insert(b, c);
    merge_aliases(&mut aliases, new_aliases);
    assert_eq!(aliases.get(&a), Some(&c));
    assert_eq!(aliases.get(&b), Some(&c));
}

#[test]
fn close_aliases_compresses_chain() {
    let mut graph = EGraph::new();
    let a = mk_nid(&mut graph);
    let b = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty());
    let c = graph.intern_pure(PureOp::Int("2".into()), smallvec![], i32_ty());
    let d = graph.intern_pure(PureOp::Int("3".into()), smallvec![], i32_ty());
    let mut aliases = HashMap::new();
    aliases.insert(a, b);
    aliases.insert(b, c);
    aliases.insert(c, d);
    close_aliases(&mut aliases);
    assert_eq!(aliases[&a], d);
    assert_eq!(aliases[&b], d);
    assert_eq!(aliases[&c], d);
}

#[test]
#[should_panic(expected = "alias cycle detected")]
fn close_aliases_panics_on_cycle() {
    let mut graph = EGraph::new();
    let a = mk_nid(&mut graph);
    let b = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty());
    let mut aliases = HashMap::new();
    aliases.insert(a, b);
    aliases.insert(b, a); // cycle — logic bug upstream
    close_aliases(&mut aliases);
}

#[test]
fn close_aliases_compresses_two_step_chain() {
    // p1 → p2 → x, all three distinct. Closure must collapse p1 and p2
    // to x directly. (Covered end-to-end by
    // optimize_skeleton_alias_closure_invariant, but exercising the
    // helper in isolation is still worth a dedicated test.)
    let mut graph = EGraph::new();
    let x = graph.intern_pure(PureOp::Int("42".into()), smallvec![], i32_ty());
    let p1 = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty());
    let p2 = graph.intern_pure(PureOp::Int("2".into()), smallvec![], i32_ty());
    let mut aliases = HashMap::new();
    aliases.insert(p1, p2);
    aliases.insert(p2, x);
    close_aliases(&mut aliases);
    assert_eq!(aliases[&p1], x);
    assert_eq!(aliases[&p2], x);
    assert!(aliases.values().all(|v| !aliases.contains_key(v)));
}

// -- optimize_skeleton fixpoint -----------------------------------------

#[test]
fn optimize_skeleton_cascades_fold_into_phi_elim() {
    // entry CondBranch{true} → A/B, both A and B → merge(x). Fold fires
    // once, phi-elim fires once, second iteration finds nothing.
    let mut graph = EGraph::new();
    let t = graph.intern_constant(ConstantValue::Bool(true), bool_ty());
    let x = graph.intern_pure(PureOp::Int("9".into()), smallvec![], i32_ty());

    let entry = graph.skeleton.entry;
    let a = graph.skeleton.create_block();
    let b = graph.skeleton.create_block();
    let merge = graph.skeleton.create_block();

    let param = graph.add_block_param(merge, 0, i32_ty());
    graph.skeleton.blocks[merge].params.push(param);
    graph.skeleton.blocks[merge].term = SkeletonTerminator::Return(Some(param));

    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond: t,
        then_target: a,
        then_args: vec![],
        else_target: b,
        else_args: vec![],
    };
    graph.skeleton.blocks[a].term = SkeletonTerminator::Branch {
        target: merge,
        args: vec![x],
    };
    graph.skeleton.blocks[b].term = SkeletonTerminator::Branch {
        target: merge,
        args: vec![x],
    };

    let aliases = run_one_body(&mut graph);

    assert!(matches!(
        &graph.skeleton.blocks[entry].term,
        SkeletonTerminator::Branch { .. }
    ));
    assert!(graph.skeleton.blocks[merge].params.is_empty());
    assert_eq!(aliases.get(&param), Some(&x));
}

#[test]
fn optimize_skeleton_alias_closure_invariant() {
    // entry → A(p1); A → B(p2); B returns p2. Both params get stripped
    // across two iterations; closure forwards p2 past p1 to x.
    let mut graph = EGraph::new();
    let x = graph.intern_pure(PureOp::Int("42".into()), smallvec![], i32_ty());

    let entry = graph.skeleton.entry;
    let a = graph.skeleton.create_block();
    let b = graph.skeleton.create_block();

    let p1 = graph.add_block_param(a, 0, i32_ty());
    graph.skeleton.blocks[a].params.push(p1);
    let p2 = graph.add_block_param(b, 0, i32_ty());
    graph.skeleton.blocks[b].params.push(p2);

    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: a,
        args: vec![x],
    };
    graph.skeleton.blocks[a].term = SkeletonTerminator::Branch {
        target: b,
        args: vec![p1],
    };
    graph.skeleton.blocks[b].term = SkeletonTerminator::Return(Some(p2));

    let aliases = run_one_body(&mut graph);

    assert!(graph.skeleton.blocks[a].params.is_empty());
    assert!(graph.skeleton.blocks[b].params.is_empty());
    assert_eq!(aliases[&p2], x);
    assert_eq!(aliases[&p1], x);
    assert!(aliases.values().all(|v| !aliases.contains_key(v)));
}

// -- additional defensive tests -----------------------------------------

#[test]
fn phi_elim_leaves_block_alone_when_unreachable_post_fold() {
    // entry CondBranch{true} → A/B. After fold, B is unreachable. B has
    // a block param; because `incoming[B][0]` is empty (no remaining
    // predecessor branches into B), phi-elim must not touch it. The
    // param stays in place; elaborate's domtree walk won't emit B.
    //
    // BUT: in the current one-pass fold+collect model, the fold and the
    // phi-elim run inside the same fixpoint iteration. The incoming
    // collection walks *every* block's terminator, INCLUDING B. If B
    // still branches somewhere (it does — to a return), that doesn't
    // add to any block's incoming list for B itself. So B's slots are
    // indeed empty and we leave its param alone. Verify that.
    let mut graph = EGraph::new();
    let t = graph.intern_constant(ConstantValue::Bool(true), bool_ty());
    let entry = graph.skeleton.entry;
    let a = graph.skeleton.create_block();
    let b = graph.skeleton.create_block();
    // B has a block param (pretending it expects an arg from a predecessor
    // that will disappear post-fold).
    let b_param = graph.add_block_param(b, 0, i32_ty());
    graph.skeleton.blocks[b].params.push(b_param);
    graph.skeleton.blocks[b].term = SkeletonTerminator::Return(Some(b_param));
    graph.skeleton.blocks[a].term = SkeletonTerminator::Return(None);
    // Entry CondBranch: true takes A (no args), false takes B (arg = some
    // literal). After folding, entry becomes Branch(A) and the edge into B
    // vanishes.
    let lit = graph.intern_pure(PureOp::Int("99".into()), smallvec![], i32_ty());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond: t,
        then_target: a,
        then_args: vec![],
        else_target: b,
        else_args: vec![lit],
    };

    let aliases = run_one_body(&mut graph);

    // Fold fires: entry is now Branch(a).
    assert!(matches!(
        &graph.skeleton.blocks[entry].term,
        SkeletonTerminator::Branch { .. }
    ));
    // Phi-elim does NOT touch B's param (no surviving incoming edges).
    assert_eq!(graph.skeleton.blocks[b].params.len(), 1);
    assert!(!aliases.contains_key(&b_param));
}

#[test]
fn phi_elim_preserves_loop_header_param() {
    // Classic loop header shape: preheader → header(init); header
    // cond-branches to body or exit; body → header(body_val). The two
    // incoming args (init and body_val) are distinct → phi must stay.
    let mut graph = EGraph::new();
    let init = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty());
    let one = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty());

    let entry = graph.skeleton.entry; // preheader
    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let exit = graph.skeleton.create_block();

    let acc = graph.add_block_param(header, 0, i32_ty());
    graph.skeleton.blocks[header].params.push(acc);

    // body_val = acc + 1 (a NodeId distinct from init)
    let body_val = graph.intern_pure(PureOp::BinOp("+".into()), smallvec![acc, one], i32_ty());

    // Entry unconditionally to header with init.
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![init],
    };
    // Header: condition (non-const function param) → body or exit.
    let cond = graph.add_func_param(0, bool_ty());
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: body,
        then_args: vec![],
        else_target: exit,
        else_args: vec![],
    };
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![body_val],
    };
    graph.skeleton.blocks[exit].term = SkeletonTerminator::Return(Some(acc));

    let aliases = run_one_body(&mut graph);

    assert!(
        aliases.is_empty(),
        "loop header accumulator must not be aliased: {:?}",
        aliases
    );
    assert_eq!(graph.skeleton.blocks[header].params.len(), 1);
}

#[test]
fn phi_elim_handles_condbranch_with_same_target_both_arms() {
    // Unusual but legal shape: a CondBranch where then_target ==
    // else_target, with the same arg passed on both arms. Phi-elim
    // should treat this as a single-incoming-value situation.
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let target = graph.skeleton.create_block();

    let param = graph.add_block_param(target, 0, i32_ty());
    graph.skeleton.blocks[target].params.push(param);
    graph.skeleton.blocks[target].term = SkeletonTerminator::Return(Some(param));

    let cond = graph.add_func_param(0, bool_ty());
    let x = graph.intern_pure(PureOp::Int("5".into()), smallvec![], i32_ty());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: target,
        then_args: vec![x],
        else_target: target,
        else_args: vec![x],
    };

    let aliases = eliminate_redundant_params(&mut graph);

    // Both arms pass x; only one distinct value → param is redundant.
    assert_eq!(aliases.get(&param), Some(&x));
    assert!(graph.skeleton.blocks[target].params.is_empty());
}
