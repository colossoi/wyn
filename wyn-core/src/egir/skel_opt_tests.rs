//! Unit tests for the skel_opt module.

use super::*;
use crate::ast::TypeName;
use crate::egir::types::{EGraph as GenericEGraph, PureOp, SkeletonTerminator, ValueId};
use crate::flow::BlockId;
use crate::op;
use crate::ssa::types::ConstantValue;
use polytype::Type;
use smallvec::smallvec;

type EGraph = GenericEGraph;

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
    assert_eq!(is_const_bool(&graph.nodes[b].kind), Some(true));
}

#[test]
fn is_const_bool_recognizes_constant_false() {
    let mut graph = EGraph::new();
    let b = graph.intern_constant(ConstantValue::Bool(false), bool_ty());
    assert_eq!(is_const_bool(&graph.nodes[b].kind), Some(false));
}

#[test]
fn is_const_bool_recognizes_pure_bool() {
    let mut graph = EGraph::new();
    let b = graph.intern_pure(PureOp::Bool(true), smallvec![], bool_ty(), None);
    assert_eq!(is_const_bool(&graph.nodes[b].kind), Some(true));
}

#[test]
fn is_const_bool_rejects_non_bool() {
    let mut graph = EGraph::new();
    let n = graph.intern_pure(PureOp::Int("42".into()), smallvec![], i32_ty(), None);
    assert_eq!(is_const_bool(&graph.nodes[n].kind), None);
}

// -- fold_constant_branches ---------------------------------------------

/// Build a `CondBranch { cond, ... }` skeleton into `graph`'s entry
/// block where then/else are fresh sibling blocks each terminating
/// in `Return(None)`. Returns (entry, then_bid, else_bid).
fn build_condbranch_skel(
    graph: &mut EGraph,
    cond: ValueId,
    then_arg: Option<ValueId>,
    else_arg: Option<ValueId>,
) -> (BlockId, BlockId, BlockId) {
    let entry = graph.skeleton.entry;
    let then_bid = graph.skeleton.create_block();
    let else_bid = graph.skeleton.create_block();
    graph.skeleton.blocks[then_bid].term = SkeletonTerminator::Return(None);
    graph.skeleton.blocks[else_bid].term = SkeletonTerminator::Return(None);
    let then_args = graph.admit_flow_values(then_arg);
    let else_args = graph.admit_flow_values(else_arg);
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
    let cond = graph.add_block_param(graph.skeleton.entry, bool_ty());
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
    let x = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty(), None);
    let y = graph.intern_pure(PureOp::Int("2".into()), smallvec![], i32_ty(), None);
    let (entry, _then_bid, _else_bid) = build_condbranch_skel(&mut graph, t, Some(x), Some(y));
    fold_constant_branches(&mut graph);
    match &graph.skeleton.blocks[entry].term {
        SkeletonTerminator::Branch { args, .. } => {
            assert_eq!(
                args.iter().map(|arg| arg.value()).collect::<Vec<_>>(),
                [x],
                "should carry then_args only"
            );
        }
        other => panic!("{:?}", other),
    }
}

// -- eliminate_redundant_params -----------------------------------------

/// entry -CondBranch-> b1|b2; b1 -> merge(left); b2 -> merge(right); merge
/// returns its single param. If left == right, phi-elim should fire.
fn build_merge_skel(
    graph: &mut EGraph,
    merge_arg_left: ValueId,
    merge_arg_right: ValueId,
) -> (BlockId, BlockId, ValueId) {
    let entry = graph.skeleton.entry;
    let b1 = graph.skeleton.create_block();
    let b2 = graph.skeleton.create_block();
    let merge = graph.skeleton.create_block();

    let param = graph.add_block_param(merge, i32_ty());
    graph.skeleton.blocks[merge].term = SkeletonTerminator::Return(Some(graph.value_result(param)));

    let cond = graph.add_block_param(graph.skeleton.entry, bool_ty());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: b1,
        then_args: vec![],
        else_target: b2,
        else_args: vec![],
    };
    let left_args = graph.admit_flow_values([merge_arg_left]);
    graph.skeleton.blocks[b1].term = SkeletonTerminator::Branch {
        target: merge,
        args: left_args,
    };
    let right_args = graph.admit_flow_values([merge_arg_right]);
    graph.skeleton.blocks[b2].term = SkeletonTerminator::Branch {
        target: merge,
        args: right_args,
    };
    (merge, b2, param)
}

#[test]
fn phi_elim_strips_param_with_matching_incoming() {
    let mut graph = EGraph::new();
    let x = graph.intern_pure(PureOp::Int("7".into()), smallvec![], i32_ty(), None);
    let (merge, _b2, param) = build_merge_skel(&mut graph, x, x);

    let mut replacements = ReplacementForest::new();
    assert!(eliminate_redundant_params(&mut graph, &mut replacements));
    let aliases = replacements.into_map();

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
fn invalid_parameter_replacements_panic_before_removing_slots() {
    for conflict in [false, true] {
        let mut graph = EGraph::new();
        let x = graph.intern_constant(ConstantValue::I32(7), i32_ty());
        let y = graph.intern_constant(ConstantValue::I32(8), i32_ty());
        let (merge, _, param) = build_merge_skel(&mut graph, x, x);
        let mut replacements = ReplacementForest::new();
        let (old, survivor) = if conflict { (param, y) } else { (x, param) };
        assert_eq!(replacements.replace(old, survivor), Ok(true));

        let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            eliminate_redundant_params(&mut graph, &mut replacements);
        }))
        .expect_err("invalid replacement must fail");
        let message = error.downcast_ref::<String>().expect("replacement diagnostic");
        assert!(message.contains("invalid skeleton parameter replacement"));
        assert!(message.contains(if conflict { "conflict" } else { "cycle" }));
        assert!(message.contains(&format!("{param:?}")));
        assert!(message.contains(&format!("{x:?}")));
        assert_eq!(graph.skeleton.blocks[merge].params.len(), 1);
        assert_eq!(graph.skeleton.blocks[merge].params[0].value(), param);
        for (_, block) in &graph.skeleton.blocks {
            if let SkeletonTerminator::Branch { target, args } = &block.term {
                if *target == merge {
                    assert_eq!(args.iter().map(|arg| arg.value()).collect::<Vec<_>>(), [x]);
                }
            }
        }
        assert_eq!(replacements.into_map(), LookupMap::from([(old, survivor)]));
    }
}

#[test]
fn phi_elim_preserves_param_with_differing_incoming() {
    // Loop-accumulator shape: preheader X, backedge Y.
    let mut graph = EGraph::new();
    let x = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty(), None);
    let y = graph.intern_pure(PureOp::Int("2".into()), smallvec![], i32_ty(), None);
    let (merge, _b2, param) = build_merge_skel(&mut graph, x, y);

    let mut replacements = ReplacementForest::new();
    assert!(!eliminate_redundant_params(&mut graph, &mut replacements));
    let aliases = replacements.into_map();

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
    let param = graph.add_block_param(b, i32_ty());
    let x = graph.intern_pure(PureOp::Int("5".into()), smallvec![], i32_ty(), None);
    let entry_args = graph.admit_flow_values([x]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: b,
        args: entry_args,
    };
    let backedge_args = graph.admit_flow_values([param]);
    graph.skeleton.blocks[b].term = SkeletonTerminator::Branch {
        target: b,
        args: backedge_args,
    };

    let mut replacements = ReplacementForest::new();
    assert!(!eliminate_redundant_params(&mut graph, &mut replacements));
    let aliases = replacements.into_map();

    // Incoming set is {x, param}: two distinct values → not redundant.
    // Even if it were {param} alone, self-alias is rejected.
    assert!(aliases.is_empty());
}

// -- optimize_skeleton fixpoint -----------------------------------------

#[test]
fn optimize_skeleton_cascades_fold_into_phi_elim() {
    // entry CondBranch{true} → A/B, both A and B → merge(x). Fold fires
    // once, phi-elim fires once, second iteration finds nothing.
    let mut graph = EGraph::new();
    let t = graph.intern_constant(ConstantValue::Bool(true), bool_ty());
    let x = graph.intern_pure(PureOp::Int("9".into()), smallvec![], i32_ty(), None);

    let entry = graph.skeleton.entry;
    let a = graph.skeleton.create_block();
    let b = graph.skeleton.create_block();
    let merge = graph.skeleton.create_block();

    let param = graph.add_block_param(merge, i32_ty());
    graph.skeleton.blocks[merge].term = SkeletonTerminator::Return(Some(graph.value_result(param)));

    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond: t,
        then_target: a,
        then_args: vec![],
        else_target: b,
        else_args: vec![],
    };
    let a_args = graph.admit_flow_values([x]);
    graph.skeleton.blocks[a].term = SkeletonTerminator::Branch {
        target: merge,
        args: a_args,
    };
    let b_args = graph.admit_flow_values([x]);
    graph.skeleton.blocks[b].term = SkeletonTerminator::Branch {
        target: merge,
        args: b_args,
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
    // in one pass; both replacements must resolve to x.
    let mut graph = EGraph::new();
    let x = graph.intern_pure(PureOp::Int("42".into()), smallvec![], i32_ty(), None);

    let entry = graph.skeleton.entry;
    let a = graph.skeleton.create_block();
    let b = graph.skeleton.create_block();

    let p1 = graph.add_block_param(a, i32_ty());
    let p2 = graph.add_block_param(b, i32_ty());

    let entry_args = graph.admit_flow_values([x]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: a,
        args: entry_args,
    };
    let a_args = graph.admit_flow_values([p1]);
    graph.skeleton.blocks[a].term = SkeletonTerminator::Branch {
        target: b,
        args: a_args,
    };
    graph.skeleton.blocks[b].term = SkeletonTerminator::Return(Some(graph.value_result(p2)));

    let aliases = run_one_body(&mut graph);

    assert!(graph.skeleton.blocks[a].params.is_empty());
    assert!(graph.skeleton.blocks[b].params.is_empty());
    assert_eq!(aliases[&p2], x);
    assert_eq!(aliases[&p1], x);
    assert!(aliases.values().all(|v| !aliases.contains_key(v)));
}

// -- additional defensive tests -----------------------------------------

#[test]
fn optimize_skeleton_removes_block_unreachable_post_fold() {
    // entry CondBranch{true} → A/B. After folding, B and its block parameter
    // are unreachable. Pruning must remove B before phi elimination considers
    // that dead edge.
    let mut graph = EGraph::new();
    let t = graph.intern_constant(ConstantValue::Bool(true), bool_ty());
    let entry = graph.skeleton.entry;
    let a = graph.skeleton.create_block();
    let b = graph.skeleton.create_block();
    // B has a block param (pretending it expects an arg from a predecessor
    // that will disappear post-fold).
    let b_param = graph.add_block_param(b, i32_ty());
    graph.skeleton.blocks[b].term = SkeletonTerminator::Return(Some(graph.value_result(b_param)));
    graph.skeleton.blocks[a].term = SkeletonTerminator::Return(None);
    // Entry CondBranch: true takes A (no args), false takes B (arg = some
    // literal). After folding, entry becomes Branch(A) and the edge into B
    // vanishes.
    let lit = graph.intern_pure(PureOp::Int("99".into()), smallvec![], i32_ty(), None);
    let else_args = graph.admit_flow_values([lit]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond: t,
        then_target: a,
        then_args: vec![],
        else_target: b,
        else_args,
    };

    let aliases = run_one_body(&mut graph);

    // Fold fires: entry is now Branch(a).
    assert!(matches!(
        &graph.skeleton.blocks[entry].term,
        SkeletonTerminator::Branch { .. }
    ));
    // Dead blocks are pruned before phi-elim considers their empty
    // incoming sets.
    assert!(!graph.skeleton.blocks.contains_key(b));
    assert!(!aliases.contains_key(&b_param));
}

#[test]
fn phi_elim_preserves_loop_header_param() {
    // Classic loop header shape: preheader → header(init); header
    // cond-branches to body or exit; body → header(body_val). The two
    // incoming args (init and body_val) are distinct → phi must stay.
    let mut graph = EGraph::new();
    let init = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty(), None);
    let one = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty(), None);

    let entry = graph.skeleton.entry; // preheader
    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let exit = graph.skeleton.create_block();

    let acc = graph.add_block_param(header, i32_ty());

    // body_val = acc + 1 (a ValueId distinct from init)
    let body_val = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![acc, one],
        i32_ty(),
        None,
    );

    // Entry unconditionally to header with init.
    let init_args = graph.admit_flow_values([init]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: header,
        args: init_args,
    };
    // Header: condition (non-const function param) → body or exit.
    let cond = graph.add_block_param(graph.skeleton.entry, bool_ty());
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: body,
        then_args: vec![],
        else_target: exit,
        else_args: vec![],
    };
    let body_args = graph.admit_flow_values([body_val]);
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: header,
        args: body_args,
    };
    graph.skeleton.blocks[exit].term = SkeletonTerminator::Return(Some(graph.value_result(acc)));

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

    let param = graph.add_block_param(target, i32_ty());
    graph.skeleton.blocks[target].term = SkeletonTerminator::Return(Some(graph.value_result(param)));

    let cond = graph.add_block_param(graph.skeleton.entry, bool_ty());
    let x = graph.intern_pure(PureOp::Int("5".into()), smallvec![], i32_ty(), None);
    let then_args = graph.admit_flow_values([x]);
    let else_args = graph.admit_flow_values([x]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: target,
        then_args,
        else_target: target,
        else_args,
    };

    let mut replacements = ReplacementForest::new();
    assert!(eliminate_redundant_params(&mut graph, &mut replacements));
    let aliases = replacements.into_map();

    // Both arms pass x; only one distinct value → param is redundant.
    assert_eq!(aliases.get(&param), Some(&x));
    assert!(graph.skeleton.blocks[target].params.is_empty());
}
