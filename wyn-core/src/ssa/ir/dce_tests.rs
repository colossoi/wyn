use super::*;
use crate::ssa::ir::ConstantValue;

#[derive(Clone)]
struct Instruction {
    args: Vec<ValueRef>,
    pure: bool,
}

impl VisitValues for Instruction {
    fn values(&self) -> Vec<ValueRef> {
        self.args.clone()
    }
    fn visit_values_mut(&mut self, visit: &mut dyn FnMut(&mut ValueRef)) {
        self.args.iter_mut().for_each(visit);
    }
}

fn pure(args: Vec<ValueRef>) -> Instruction {
    Instruction { args, pure: true }
}
fn number(value: i32) -> ValueRef {
    ValueRef::Const(ConstantValue::I32(value))
}
fn clean(function: &mut Function<Instruction, ()>) {
    eliminate_dead_values(function, |instruction| instruction.pure);
}

#[test]
fn removes_dead_loop_carried_cycles_without_removing_loop_control() {
    let mut f = Function::new();
    let input = f.add_function_param((), "input".into());
    let header = f.create_block();
    let body = f.create_block();
    let merge = f.create_block();
    let dead = f.add_block_param(header, ());
    let live = f.add_block_param(header, ());
    let result = f.add_block_param(merge, ());
    let seed = f.append_inst(f.entry, pure(vec![input.into()]), ());
    f.blocks[f.entry].term = Terminator::Branch {
        target: header,
        args: vec![seed.into(), number(0)],
    };
    let condition = f.append_inst(header, pure(vec![live.into(), input.into()]), ());
    f.blocks[header].control_header = Some(ControlHeader::Loop {
        merge,
        continue_block: body,
    });
    f.blocks[header].term = Terminator::CondBranch {
        cond: condition.into(),
        then_target: body,
        then_args: vec![],
        else_target: merge,
        else_args: vec![live.into()],
    };
    let dead_next = f.append_inst(body, pure(vec![dead.into()]), ());
    let live_next = f.append_inst(body, pure(vec![live.into(), number(1)]), ());
    f.blocks[body].term = Terminator::Branch {
        target: header,
        args: vec![dead_next.into(), live_next.into()],
    };
    f.blocks[merge].term = Terminator::Return(Some(result.into()));
    clean(&mut f);
    for value in [seed, dead, dead_next] {
        assert!(!f.values.contains_key(value));
    }
    assert_eq!(f.blocks[header].params, vec![live]);
    assert!(matches!(f.values[live].def, ValueDef::Param { index: 0, .. }));
    assert!(matches!(&f.blocks[f.entry].term, Terminator::Branch { args, .. } if args == &[number(0)]));
    assert!(matches!(&f.blocks[body].term, Terminator::Branch { args, .. } if args == &[live_next.into()]));
    assert!(matches!(f.blocks[header].term, Terminator::CondBranch { .. }));
    assert_eq!(f.params, vec![input]);
    // Even when the loop has no live result, its termination condition and
    // the counter feeding it must remain.
    f.blocks[merge].term = Terminator::Return(Some(number(7)));
    clean(&mut f);
    assert!(f.blocks[merge].params.is_empty());
    assert_eq!(f.blocks[header].params, vec![live]);
    assert!(matches!(f.blocks[header].term, Terminator::CondBranch { .. }));
}

fn selection(effect: bool, returned: bool) -> Function<Instruction, ()> {
    let mut f = Function::new();
    let input = f.add_function_param((), "input".into());
    let yes = f.create_block();
    let no = f.create_block();
    let merge = f.create_block();
    let dead = f.add_block_param(merge, ());
    let result = f.add_block_param(merge, ());
    let cond = f.append_inst(f.entry, pure(vec![input.into()]), ());
    f.blocks[f.entry].control_header = Some(ControlHeader::Selection { merge });
    f.blocks[f.entry].term = Terminator::CondBranch {
        cond: cond.into(),
        then_target: yes,
        then_args: vec![],
        else_target: no,
        else_args: vec![],
    };
    for (block, value) in [(yes, 1), (no, 2)] {
        let unused = f.append_inst(block, pure(vec![input.into()]), ());
        f.blocks[block].term = Terminator::Branch {
            target: merge,
            args: vec![unused.into(), number(value)],
        };
    }
    if effect {
        // An opaque call with an unused result must remain, just like a store.
        let opaque = f.append_inst(
            yes,
            Instruction {
                args: vec![input.into()],
                pure: false,
            },
            (),
        );
        f.append_void_inst(
            no,
            Instruction {
                args: vec![input.into()],
                pure: false,
            },
        );
        assert!(f.values.contains_key(opaque));
    }
    f.blocks[merge].term = Terminator::Return(Some(if returned { result.into() } else { number(7) }));
    assert_ne!(dead, result);
    f
}

#[test]
fn removes_unused_phi_dependencies_and_the_empty_selection() {
    let mut f = selection(false, false);
    clean(&mut f);
    assert_eq!(f.blocks.len(), 2);
    assert!(f.insts.is_empty());
    assert!(f.blocks.values().all(|block| block.params.is_empty()));
    assert!(f.blocks[f.entry].control_header.is_none());
    assert!(matches!(f.blocks[f.entry].term, Terminator::Branch { .. }));
}

#[test]
fn keeps_effectful_arms_and_live_merge_arguments() {
    for (effect, returned) in [(true, false), (false, true)] {
        let mut f = selection(effect, returned);
        clean(&mut f);
        assert_eq!(f.blocks.len(), 4);
        assert!(matches!(f.blocks[f.entry].term, Terminator::CondBranch { .. }));
        if effect {
            assert_eq!(f.insts.values().filter(|node| !node.data.pure).count(), 2);
        }
        for (id, block) in &f.blocks {
            assert_eq!(
                block.params.len(),
                usize::from(returned && matches!(block.term, Terminator::Return(_)))
            );
            for (index, &value) in block.params.iter().enumerate() {
                assert!(
                    matches!(f.values[value].def, ValueDef::Param { block, index: i } if block == id && i == index)
                );
            }
        }
    }
}
