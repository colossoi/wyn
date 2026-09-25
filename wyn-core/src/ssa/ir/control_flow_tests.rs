use super::*;

#[test]
fn constant_selections_preserve_chosen_arguments_and_remove_only_unreachable_definitions() {
    for condition in [Some(false), Some(true), None] {
        let mut f = Function::new();
        let input = f.add_function_param((), "condition".into());
        let yes = f.create_block();
        let no = f.create_block();
        let merge = f.create_block();
        let result = f.add_block_param(merge, ());
        f.blocks[f.entry].control_header = Some(ControlHeader::Selection { merge });
        let yes_arg = ValueRef::Const(ConstantValue::I32(11));
        let no_arg = ValueRef::Const(ConstantValue::I32(22));
        f.blocks[f.entry].term = Terminator::CondBranch {
            cond: condition.map_or(input.into(), |v| ValueRef::Const(ConstantValue::Bool(v))),
            then_target: yes,
            then_args: vec![yes_arg],
            else_target: no,
            else_args: vec![no_arg],
        };
        let arms = [yes, no].map(|block| {
            let param = f.add_block_param(block, ());
            let value = f.append_inst(block, (), ());
            // Resultless instructions model effects: cleanup must remove dead
            // arm effects, but cannot speculate or drop live arm effects.
            let effect = f.append_void_inst(block, ());
            f.blocks[block].term = Terminator::Branch {
                target: merge,
                args: vec![value.into()],
            };
            (block, param, value, f.inst_of_value(value).unwrap(), effect)
        });
        f.blocks[merge].term = Terminator::Return(Some(result.into()));

        fold_constant_selections(&mut f);

        if let Some(condition) = condition {
            let (target, arg) = if condition { (yes, yes_arg) } else { (no, no_arg) };
            assert!(matches!(&f.blocks[f.entry].term,
                Terminator::Branch { target: actual, args } if *actual == target && args == &[arg]));
            assert!(f.blocks[f.entry].control_header.is_none());
            assert_eq!(f.blocks.len(), 3);
        } else {
            assert!(matches!(f.blocks[f.entry].term, Terminator::CondBranch { .. }));
            assert!(f.blocks[f.entry].control_header.is_some());
            assert_eq!(f.blocks.len(), 4);
        }
        for (block, param, value, inst, effect) in arms {
            let kept = condition.is_none_or(|condition| (block == yes) == condition);
            assert_eq!(f.blocks.contains_key(block), kept);
            assert_eq!(f.values.contains_key(param), kept);
            assert_eq!(f.values.contains_key(value), kept);
            assert_eq!(f.insts.contains_key(inst), kept);
            assert_eq!(f.insts.contains_key(effect), kept);
            if kept {
                assert!(matches!(&f.blocks[block].term,
                    Terminator::Branch { target, args } if *target == merge && args == &[value.into()]));
            }
        }
        assert_eq!(f.params, vec![input]);
        assert_eq!(f.blocks[merge].params, vec![result]);
    }
}

#[test]
fn constant_selection_cleanup_preserves_enclosing_loop_structural_targets() {
    let mut f = Function::<(), ()>::new();
    let selection = f.create_block();
    let yes = f.create_block();
    let no = f.create_block();
    let merge = f.create_block();
    let continue_block = f.create_block();
    f.blocks[f.entry].control_header = Some(ControlHeader::Loop {
        merge,
        continue_block,
    });
    f.blocks[f.entry].term = Terminator::Branch {
        target: selection,
        args: vec![],
    };
    f.blocks[selection].control_header = Some(ControlHeader::Selection { merge });
    f.blocks[selection].term = Terminator::CondBranch {
        cond: ValueRef::Const(ConstantValue::Bool(false)),
        then_target: yes,
        then_args: vec![],
        else_target: no,
        else_args: vec![],
    };
    f.blocks[yes].term = Terminator::Branch {
        target: continue_block,
        args: vec![],
    };
    f.blocks[no].term = Terminator::Return(None);
    f.blocks[continue_block].term = Terminator::Branch {
        target: f.entry,
        args: vec![],
    };
    f.blocks[merge].term = Terminator::Return(None);

    fold_constant_selections(&mut f);

    assert!(!f.blocks.contains_key(yes));
    assert!(f.blocks.contains_key(merge));
    assert!(f.blocks.contains_key(continue_block));
    assert!(matches!(
        f.blocks[f.entry].control_header,
        Some(ControlHeader::Loop { .. })
    ));
}
