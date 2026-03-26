use crate::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TestEffect(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TestType;

#[derive(Clone, Debug, PartialEq, Eq)]
enum TestInst {
    Int(i64),
    Add(ValueId, ValueId),
    Pair(ValueId, ValueId),
    SideEffect(ValueId),
}

impl Instr for TestInst {
    fn for_each_operand(&self, mut f: impl FnMut(ValueId)) {
        match self {
            TestInst::Int(_) => {}
            TestInst::Add(a, b) | TestInst::Pair(a, b) => {
                f(*a);
                f(*b);
            }
            TestInst::SideEffect(v) => f(*v),
        }
    }

    fn map_operands(&self, mut f: impl FnMut(ValueId) -> ValueId) -> Self {
        match self {
            TestInst::Int(n) => TestInst::Int(*n),
            TestInst::Add(a, b) => TestInst::Add(f(*a), f(*b)),
            TestInst::Pair(a, b) => TestInst::Pair(f(*a), f(*b)),
            TestInst::SideEffect(v) => TestInst::SideEffect(f(*v)),
        }
    }
}

impl ValueLike for TestInst {
    fn is_hoistable(&self) -> bool {
        true
    }

    fn is_closed(&self) -> bool {
        matches!(self, TestInst::Int(_))
    }

    fn equivalent_to(&self, other: &Self) -> bool {
        self == other
    }
}

type TestFunc = Function<TestInst, TestEffect, TestType>;

const TY: TestType = TestType;

#[test]
fn inline_entry_param_replaces_uses_and_removes_param() {
    let mut f = TestFunc::new();
    let p0 = f.add_block_param(f.entry, TY);

    let pair = f.append_inst(f.entry, TestInst::Pair(p0, p0), TY, None);
    f.blocks[f.entry].term = Terminator::Return(Some(pair));

    inline_entry_param(&mut f, 0, TestInst::Int(42), TY).unwrap();
    assert!(f.blocks[f.entry].params.is_empty());

    let mut saw_pair = false;
    for (_id, node) in &f.insts {
        if let TestInst::Pair(a, b) = node.data {
            saw_pair = true;
            let ia = f.inst_of_value(a).unwrap();
            let ib = f.inst_of_value(b).unwrap();
            assert_eq!(f.insts[ia].data, TestInst::Int(42));
            assert_eq!(f.insts[ib].data, TestInst::Int(42));
        }
    }
    assert!(saw_pair);
}

#[test]
fn lift_and_merge_merges_duplicate_closed_values() {
    let mut f = TestFunc::new();

    let c1 = f.append_inst(f.entry, TestInst::Int(7), TY, None);
    let c2 = f.append_inst(f.entry, TestInst::Int(7), TY, None);
    let pair = f.append_inst(f.entry, TestInst::Pair(c1, c2), TY, None);
    f.blocks[f.entry].term = Terminator::Return(Some(pair));

    let before = f.insts.len();
    lift_and_merge(&mut f);
    let after = f.insts.len();

    assert!(after < before);

    let mut pair_operands = None;
    for (_id, node) in &f.insts {
        if let TestInst::Pair(a, b) = node.data {
            pair_operands = Some((a, b));
        }
    }

    let (a, b) = pair_operands.unwrap();
    assert_eq!(a, b);
}

#[test]
fn lift_and_merge_merges_duplicate_non_closed_values() {
    let mut f = TestFunc::new();
    let p0 = f.add_block_param(f.entry, TY);
    let p1 = f.add_block_param(f.entry, TY);

    let x1 = f.append_inst(f.entry, TestInst::Add(p0, p1), TY, None);
    let x2 = f.append_inst(f.entry, TestInst::Add(p0, p1), TY, None);
    let pair = f.append_inst(f.entry, TestInst::Pair(x1, x2), TY, None);
    f.blocks[f.entry].term = Terminator::Return(Some(pair));

    let before = f.insts.len();
    lift_and_merge(&mut f);
    assert!(f.insts.len() < before);

    let mut pair_operands = None;
    for (_id, node) in &f.insts {
        if let TestInst::Pair(a, b) = node.data {
            pair_operands = Some((a, b));
        }
    }

    let (a, b) = pair_operands.unwrap();
    assert_eq!(a, b);
}

#[test]
fn inline_non_entry_block_param_updates_predecessor_edge_args() {
    let mut f = TestFunc::new();
    let cond = f.add_block_param(f.entry, TY);

    let join = f.create_block();
    let jp = f.add_block_param(join, TY);

    let t = f.create_block();
    let e = f.create_block();

    let c1 = f.append_inst(t, TestInst::Int(1), TY, None);
    f.blocks[t].term = Terminator::Jump {
        target: join,
        args: vec![c1],
    };

    let c2 = f.append_inst(e, TestInst::Int(1), TY, None);
    f.blocks[e].term = Terminator::Jump {
        target: join,
        args: vec![c2],
    };

    f.blocks[f.entry].term = Terminator::Branch {
        cond,
        then_block: t,
        then_args: vec![],
        else_block: e,
        else_args: vec![],
    };

    let out = f.append_inst(join, TestInst::Pair(jp, jp), TY, None);
    f.blocks[join].term = Terminator::Return(Some(out));

    inline_block_param(&mut f, join, 0, TestInst::Int(9), TY).unwrap();

    match &f.blocks[t].term {
        Terminator::Jump { target, args } => {
            assert_eq!(*target, join);
            assert!(args.is_empty());
        }
        _ => panic!("expected jump"),
    }

    match &f.blocks[e].term {
        Terminator::Jump { target, args } => {
            assert_eq!(*target, join);
            assert!(args.is_empty());
        }
        _ => panic!("expected jump"),
    }
}

#[test]
fn effectful_inst_creation_and_query() {
    let mut f = TestFunc::new();
    let p0 = f.add_block_param(f.entry, TY);

    let pure_val = f.append_inst(f.entry, TestInst::Int(1), TY, None);
    let eff_val = f.append_inst(
        f.entry,
        TestInst::SideEffect(p0),
        TY,
        Some((TestEffect(0), TestEffect(1))),
    );

    assert!(!f.is_effectful(f.inst_of_value(pure_val).unwrap()));
    assert!(f.is_effectful(f.inst_of_value(eff_val).unwrap()));

    let eff_inst = f.inst_of_value(eff_val).unwrap();
    assert_eq!(f.insts[eff_inst].effects, Some((TestEffect(0), TestEffect(1))));
}

#[test]
fn remove_effectful_inst_splices_chain() {
    let mut f = TestFunc::new();
    let p0 = f.add_block_param(f.entry, TY);

    // Chain: e0 -> inst_a -> e1 -> inst_b -> e2
    f.append_inst(
        f.entry,
        TestInst::SideEffect(p0),
        TY,
        Some((TestEffect(0), TestEffect(1))),
    );
    let b = f.append_inst(
        f.entry,
        TestInst::SideEffect(p0),
        TY,
        Some((TestEffect(1), TestEffect(2))),
    );

    // Remove inst_a (consumes e0, produces e1)
    // inst_b should now consume e0 instead of e1
    let a_inst = f.blocks[f.entry].insts[0];
    f.remove_inst(a_inst);

    let b_inst = f.inst_of_value(b).unwrap();
    assert_eq!(f.insts[b_inst].effects, Some((TestEffect(0), TestEffect(2))));
}

#[test]
fn replace_effect_works() {
    let mut f = TestFunc::new();
    let p0 = f.add_block_param(f.entry, TY);

    let v = f.append_inst(
        f.entry,
        TestInst::SideEffect(p0),
        TY,
        Some((TestEffect(5), TestEffect(6))),
    );

    assert!(f.replace_effect(TestEffect(5), TestEffect(99)));

    let inst = f.inst_of_value(v).unwrap();
    assert_eq!(f.insts[inst].effects, Some((TestEffect(99), TestEffect(6))));
}

#[test]
fn verify_effects_ok_for_valid_chain() {
    let mut f = TestFunc::new();
    let p0 = f.add_block_param(f.entry, TY);

    f.append_inst(
        f.entry,
        TestInst::SideEffect(p0),
        TY,
        Some((TestEffect(0), TestEffect(1))),
    );
    f.append_inst(
        f.entry,
        TestInst::SideEffect(p0),
        TY,
        Some((TestEffect(1), TestEffect(2))),
    );
    f.append_inst(f.entry, TestInst::Int(42), TY, None);

    assert!(f.verify_effects().is_ok());
}

#[test]
fn verify_effects_catches_double_consume() {
    let mut f = TestFunc::new();
    let p0 = f.add_block_param(f.entry, TY);

    // Both instructions consume TestEffect(0) — invalid
    f.append_inst(
        f.entry,
        TestInst::SideEffect(p0),
        TY,
        Some((TestEffect(0), TestEffect(1))),
    );
    f.append_inst(
        f.entry,
        TestInst::SideEffect(p0),
        TY,
        Some((TestEffect(0), TestEffect(2))),
    );

    let errs = f.verify_effects().unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e,
        EffectVerifyError::DoubleConsume {
            effect: TestEffect(0),
            ..
        }
    )));
}

#[test]
fn lift_and_merge_skips_effectful_instructions() {
    let mut f = TestFunc::new();
    let p0 = f.add_block_param(f.entry, TY);

    // Two identical effectful instructions — should NOT be merged
    let a = f.append_inst(
        f.entry,
        TestInst::SideEffect(p0),
        TY,
        Some((TestEffect(0), TestEffect(1))),
    );
    let b = f.append_inst(
        f.entry,
        TestInst::SideEffect(p0),
        TY,
        Some((TestEffect(1), TestEffect(2))),
    );
    let pair = f.append_inst(f.entry, TestInst::Pair(a, b), TY, None);
    f.blocks[f.entry].term = Terminator::Return(Some(pair));

    lift_and_merge(&mut f);

    // Both instructions should still exist
    assert!(f.inst_of_value(a).is_some());
    assert!(f.inst_of_value(b).is_some());
}

#[test]
fn value_type_tracks_types() {
    let mut f = Function::<TestInst, TestEffect, &str>::new();
    let p0 = f.add_block_param(f.entry, "int");
    let v = f.append_inst(f.entry, TestInst::Int(42), "int", None);
    let pair = f.append_inst(f.entry, TestInst::Pair(p0, v), "pair", None);

    assert_eq!(*f.value_type(p0), "int");
    assert_eq!(*f.value_type(v), "int");
    assert_eq!(*f.value_type(pair), "pair");
}

#[test]
fn forward_single_pred_params_substitutes_and_clears() {
    // entry -> block0(%x, %y) -> block1
    // block0 has one predecessor (entry), so %a=%x, %b=%y should be forwarded.
    let mut f = TestFunc::new();
    let x = f.add_block_param(f.entry, TY);
    let y = f.add_block_param(f.entry, TY);

    let block0 = f.create_block();
    let a = f.add_block_param(block0, TY);
    let b = f.add_block_param(block0, TY);

    // entry jumps to block0 with (x, y)
    f.blocks[f.entry].term = Terminator::Jump {
        target: block0,
        args: vec![x, y],
    };

    // block0 uses %a and %b
    let sum = f.append_inst(block0, TestInst::Add(a, b), TY, None);
    f.blocks[block0].term = Terminator::Return(Some(sum));

    forward_single_pred_params(&mut f);

    // block0 params should be cleared
    assert!(f.blocks[block0].params.is_empty());

    // entry's jump args should be cleared
    match &f.blocks[f.entry].term {
        Terminator::Jump { args, .. } => assert!(args.is_empty()),
        _ => panic!("expected jump"),
    }

    // The Add instruction should now reference x and y directly
    let add_inst = f.inst_of_value(sum).unwrap();
    match &f.insts[add_inst].data {
        TestInst::Add(lhs, rhs) => {
            assert_eq!(*lhs, x);
            assert_eq!(*rhs, y);
        }
        _ => panic!("expected Add"),
    }
}

#[test]
fn forward_single_pred_params_skips_multi_predecessor_blocks() {
    // Two predecessors jump to the same block — should NOT forward.
    let mut f = TestFunc::new();
    let cond = f.add_block_param(f.entry, TY);

    let join = f.create_block();
    let jp = f.add_block_param(join, TY);

    let t = f.create_block();
    let e = f.create_block();

    let c1 = f.append_inst(t, TestInst::Int(1), TY, None);
    f.blocks[t].term = Terminator::Jump {
        target: join,
        args: vec![c1],
    };

    let c2 = f.append_inst(e, TestInst::Int(2), TY, None);
    f.blocks[e].term = Terminator::Jump {
        target: join,
        args: vec![c2],
    };

    f.blocks[f.entry].term = Terminator::Branch {
        cond,
        then_block: t,
        then_args: vec![],
        else_block: e,
        else_args: vec![],
    };

    let out = f.append_inst(join, TestInst::Pair(jp, jp), TY, None);
    f.blocks[join].term = Terminator::Return(Some(out));

    forward_single_pred_params(&mut f);

    // join still has its param — two predecessors, can't forward
    assert_eq!(f.blocks[join].params.len(), 1);
}
