use super::*;
use crate::flow::ControlHeader;
use crate::ssa::ir::ConstantValue;

#[derive(Clone)]
struct Instruction(Vec<ValueRef>);

impl VisitValues for Instruction {
    fn values(&self) -> Vec<ValueRef> {
        self.0.clone()
    }

    fn visit_values_mut(&mut self, visit: &mut dyn FnMut(&mut ValueRef)) {
        self.0.iter_mut().for_each(visit);
    }
}

fn number(value: i32) -> ValueRef {
    ValueRef::Const(ConstantValue::I32(value))
}

#[test]
fn forwarding_chains_rewrite_instructions_returns_and_edges_in_dominator_order() {
    let mut f = Function::new();
    let input = f.add_function_param((), "input".into());
    // Deliberately create the consumer before its predecessor.
    let last = f.create_block();
    let first = f.create_block();
    let a = f.add_block_param(first, ());
    let b = f.add_block_param(last, ());
    let bits = f.add_block_param(last, ());
    let negative_zero = ValueRef::Const(ConstantValue::F32((-0.0_f32).to_bits()));
    f.blocks[f.entry].term = Terminator::Branch {
        target: first,
        args: vec![input.into()],
    };
    f.blocks[first].term = Terminator::Branch {
        target: last,
        args: vec![a.into(), negative_zero],
    };
    let used = f.append_inst(last, Instruction(vec![b.into(), bits.into()]), ());
    f.blocks[last].term = Terminator::Return(Some(b.into()));

    eliminate_single_input_params(&mut f);

    assert_eq!(f.blocks.len(), 3);
    assert_eq!(f.params, vec![input]);
    for param in [a, b, bits] {
        assert!(!f.values.contains_key(param));
    }
    for block in [first, last] {
        assert!(f.blocks[block].params.is_empty());
    }
    for block in [f.entry, first] {
        assert!(matches!(&f.blocks[block].term, Terminator::Branch { args, .. } if args.is_empty()));
    }
    assert_eq!(
        f.insts[f.inst_of_value(used).unwrap()].data.0,
        vec![input.into(), negative_zero]
    );
    assert!(matches!(f.blocks[last].term, Terminator::Return(Some(value)) if value == input.into()));
}

#[test]
fn single_input_continue_and_exit_params_disappear_but_loop_carried_values_remain() {
    let mut f = Function::new();
    let condition = f.add_function_param((), "condition".into());
    let header = f.create_block();
    let body = f.create_block();
    let continuing = f.create_block();
    let merge = f.create_block();
    let carried = f.add_block_param(header, ());
    let next = f.add_block_param(continuing, ());
    let result = f.add_block_param(merge, ());
    f.blocks[f.entry].term = Terminator::Branch {
        target: header,
        args: vec![number(0)],
    };
    f.blocks[header].control_header = Some(ControlHeader::Loop {
        merge,
        continue_block: continuing,
    });
    f.blocks[header].term = Terminator::CondBranch {
        cond: condition.into(),
        then_target: body,
        then_args: vec![],
        else_target: merge,
        else_args: vec![carried.into()],
    };
    let increment = f.append_inst(body, Instruction(vec![carried.into(), number(1)]), ());
    f.blocks[body].term = Terminator::Branch {
        target: continuing,
        args: vec![increment.into()],
    };
    f.blocks[continuing].term = Terminator::Branch {
        target: header,
        args: vec![next.into()],
    };
    f.blocks[merge].term = Terminator::Return(Some(result.into()));

    eliminate_single_input_params(&mut f);

    assert_eq!(f.blocks.len(), 5);
    assert_eq!(f.blocks[header].params, vec![carried]);
    assert!(matches!(f.values[carried].def, ValueDef::Param { block, index: 0 } if block == header));
    assert!(f.blocks[continuing].params.is_empty());
    assert!(f.blocks[merge].params.is_empty());
    assert!(!f.values.contains_key(next));
    assert!(!f.values.contains_key(result));
    assert!(
        matches!(&f.blocks[continuing].term, Terminator::Branch { target, args }
        if *target == header && args == &[increment.into()])
    );
    assert!(
        matches!(&f.blocks[header].term, Terminator::CondBranch { else_args, .. } if else_args.is_empty())
    );
    assert!(matches!(f.blocks[merge].term, Terminator::Return(Some(value)) if value == carried.into()));
    assert!(matches!(f.blocks[header].control_header,
        Some(ControlHeader::Loop { merge: m, continue_block: c }) if m == merge && c == continuing));
}

#[test]
fn distinct_merge_inputs_remain_while_branch_parameters_are_substituted() {
    let mut f = Function::<Instruction, ()>::new();
    let condition = f.add_function_param((), "condition".into());
    let left = f.create_block();
    let right = f.create_block();
    let merge = f.create_block();
    let left_value = f.add_block_param(left, ());
    let right_value = f.add_block_param(right, ());
    let result = f.add_block_param(merge, ());
    f.blocks[f.entry].control_header = Some(ControlHeader::Selection { merge });
    f.blocks[f.entry].term = Terminator::CondBranch {
        cond: condition.into(),
        then_target: left,
        then_args: vec![number(1)],
        else_target: right,
        else_args: vec![number(2)],
    };
    for (block, value) in [(left, left_value), (right, right_value)] {
        f.blocks[block].term = Terminator::Branch {
            target: merge,
            args: vec![value.into()],
        };
    }
    f.blocks[merge].term = Terminator::Return(Some(result.into()));

    eliminate_single_input_params(&mut f);

    assert!(
        matches!(&f.blocks[f.entry].term, Terminator::CondBranch { then_args, else_args, .. }
        if then_args.is_empty() && else_args.is_empty())
    );
    for (block, expected) in [(left, number(1)), (right, number(2))] {
        assert!(f.blocks[block].params.is_empty());
        assert!(matches!(&f.blocks[block].term, Terminator::Branch { args, .. } if args == &[expected]));
    }
    assert_eq!(f.blocks[merge].params, vec![result]);
    assert!(matches!(f.blocks[merge].term, Terminator::Return(Some(value)) if value == result.into()));
    assert!(matches!(
        f.blocks[f.entry].control_header,
        Some(ControlHeader::Selection { .. })
    ));
}

#[test]
fn two_edges_from_one_predecessor_do_not_count_as_one_input() {
    let mut f = Function::<Instruction, ()>::new();
    let condition = f.add_function_param((), "condition".into());
    let merge = f.create_block();
    let result = f.add_block_param(merge, ());
    f.blocks[f.entry].term = Terminator::CondBranch {
        cond: condition.into(),
        then_target: merge,
        then_args: vec![number(1)],
        else_target: merge,
        else_args: vec![number(2)],
    };
    f.blocks[merge].term = Terminator::Return(Some(result.into()));

    eliminate_single_input_params(&mut f);

    assert_eq!(f.blocks[merge].params, vec![result]);
    assert!(
        matches!(&f.blocks[f.entry].term, Terminator::CondBranch { then_args, else_args, .. }
        if then_args == &[number(1)] && else_args == &[number(2)])
    );
}

#[test]
fn unreachable_forwarding_cycles_and_zero_input_params_are_left_alone() {
    let mut f = Function::<Instruction, ()>::new();
    let a = f.create_block();
    let b = f.create_block();
    let no_predecessor = f.create_block();
    let av = f.add_block_param(a, ());
    let bv = f.add_block_param(b, ());
    let unused = f.add_block_param(no_predecessor, ());
    f.blocks[a].term = Terminator::Branch {
        target: b,
        args: vec![av.into()],
    };
    f.blocks[b].term = Terminator::Branch {
        target: a,
        args: vec![bv.into()],
    };

    eliminate_single_input_params(&mut f);

    for (block, param) in [(a, av), (b, bv), (no_predecessor, unused)] {
        assert_eq!(f.blocks[block].params, vec![param]);
        assert!(f.values.contains_key(param));
    }
}
