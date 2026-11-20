use super::*;
use crate::flow::Terminator;

#[derive(Clone, Debug)]
struct TestInstruction(Vec<ValueRef>);

impl VisitValues for TestInstruction {
    fn values(&self) -> Vec<ValueRef> {
        self.0.clone()
    }

    fn visit_values_mut(&mut self, visit: &mut dyn FnMut(&mut ValueRef)) {
        self.0.iter_mut().for_each(visit);
    }
}

#[test]
fn schedules_reachable_dependencies_once_and_discards_unreachable_values() {
    let mut function = Function::<TestInstruction, ()>::new();
    let entry = function.entry;
    let parameter = function.add_function_param(());
    let loop_body = function.create_block();
    let loop_parameter = function.add_block_param(loop_body, ());
    function.blocks[entry].term = Terminator::Branch {
        target: loop_body,
        args: vec![parameter.into()],
    };

    let invariant = function.append_floating_inst(entry, TestInstruction(vec![parameter.into()]), ());
    let result = function.append_floating_inst(entry, TestInstruction(vec![invariant.into()]), ());
    let dead = function.append_floating_inst(entry, TestInstruction(vec![parameter.into()]), ());
    let consumer = function.append_void_inst(
        loop_body,
        TestInstruction(vec![result.into(), loop_parameter.into()]),
    );
    function.blocks[loop_body].term = Terminator::Return(None);

    schedule_floating(&mut function).unwrap();

    assert_eq!(function.blocks[entry].insts.len(), 2);
    assert_eq!(function.blocks[loop_body].insts, vec![consumer]);
    assert_eq!(function.block_of_value(invariant), Some(entry));
    assert_eq!(function.block_of_value(result), Some(entry));
    assert!(!function.values.contains_key(dead));
}

#[test]
fn scoped_values_wait_for_use_without_crossing_their_branch() {
    let mut function = Function::<TestInstruction, ()>::new();
    let entry = function.entry;
    let parameter = function.add_function_param(());
    let arm = function.create_block();
    let other = function.create_block();
    function.blocks[entry].term = Terminator::CondBranch {
        cond: parameter.into(),
        then_target: arm,
        then_args: vec![],
        else_target: other,
        else_args: vec![],
    };
    let value = function.append_floating_inst(arm, TestInstruction(vec![parameter.into()]), ());
    let read = function.append_void_inst(arm, TestInstruction(vec![parameter.into()]));
    let consumer = function.append_void_inst(arm, TestInstruction(vec![value.into()]));
    function.blocks[arm].term = Terminator::Return(None);
    function.blocks[other].term = Terminator::Return(Some(parameter.into()));

    schedule_floating(&mut function).unwrap();

    assert!(function.blocks[entry].insts.is_empty());
    assert_eq!(function.block_of_value(value), Some(arm));
    assert_eq!(
        function.blocks[arm].insts,
        vec![read, function.inst_of_value(value).unwrap(), consumer]
    );
}
