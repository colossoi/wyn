use super::*;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::eliminate_dead_pure_instructions;
use crate::ssa::types::{ConstantValue, Terminator};
use crate::types::{bool_type, i32, sized_array};

fn binary(op: BinaryOperator, a: ValueRef, b: ValueRef) -> InstKind {
    InstKind::Op {
        tag: OpTag::BinOp(op),
        operands: vec![a, b],
    }
}

#[test]
fn common_branch_work_moves_to_its_operand_dominator_but_division_stays_guarded() {
    let mut body = FuncBuilder::new(
        vec![(i32(), "x".into()), (bool_type(), "condition".into())],
        i32(),
    )
    .finish_unchecked();
    let entry = body.inner.entry;
    let x = body.inner.params[0].into();
    let condition = body.inner.params[1].into();
    let left = body.inner.create_block();
    let right = body.inner.create_block();
    body.inner.blocks[entry].term = Terminator::CondBranch {
        cond: condition,
        then_target: left,
        then_args: vec![],
        else_target: right,
        else_args: vec![],
    };
    for block in [left, right] {
        let doubled = body.inner.append_inst(block, binary(BinaryOperator::Add, x, x), i32());
        let divided =
            body.inner.append_inst(block, binary(BinaryOperator::Divide, doubled.into(), x), i32());
        body.inner.blocks[block].term = Terminator::Return(Some(divided.into()));
    }
    float_pure_values(&mut body);
    crate::ssa::ir::schedule_floating(&mut body.inner).unwrap();
    assert_eq!(body.inner.blocks[entry].insts.len(), 1);
    for block in [left, right] {
        assert_eq!(body.inner.blocks[block].insts.len(), 1);
        assert!(matches!(
            body.inner.insts[body.inner.blocks[block].insts[0]].data,
            InstKind::Op {
                tag: OpTag::BinOp(BinaryOperator::Divide),
                ..
            }
        ));
    }
}

#[test]
fn dynamic_array_materialization_is_shared_outside_the_loop() {
    let mut body = FuncBuilder::new(vec![(sized_array(128, i32()), "xs".into())], i32()).finish_unchecked();
    let entry = body.inner.entry;
    let array = body.inner.params[0];
    let loop_body = body.inner.create_block();
    let i = body.inner.add_block_param(loop_body, i32());
    body.inner.blocks[entry].term = Terminator::Branch {
        target: loop_body,
        args: vec![ValueRef::Const(ConstantValue::I32(0))],
    };
    let mut results = vec![];
    for _ in 0..2 {
        results.push(body.inner.append_inst(
            loop_body,
            InstKind::Op {
                tag: OpTag::Index,
                operands: vec![array.into(), i.into()],
            },
            i32(),
        ));
    }
    let sum = body.inner.append_inst(
        loop_body,
        binary(BinaryOperator::Add, results[0].into(), results[1].into()),
        i32(),
    );
    body.inner.blocks[loop_body].term = Terminator::Branch {
        target: loop_body,
        args: vec![sum.into()],
    };
    float_pure_values(&mut body);
    crate::ssa::ir::schedule_floating(&mut body.inner).unwrap();
    let materialized: Vec<_> = body
        .inner
        .insts
        .values()
        .filter(|n| {
            matches!(
                n.data,
                InstKind::Op {
                    tag: OpTag::Materialize,
                    ..
                }
            )
        })
        .collect();
    assert_eq!(materialized.len(), 1);
    assert_eq!(materialized[0].placement.block(), Some(entry));
    assert_eq!(
        body.inner
            .insts
            .values()
            .filter(|n| matches!(
                n.data,
                InstKind::Op {
                    tag: OpTag::DynamicExtract,
                    ..
                }
            ))
            .count(),
        2
    );
}

#[test]
fn helper_substitution_does_not_confuse_distinct_value_arenas() {
    let mut helper =
        FuncBuilder::new(vec![(i32(), "a".into()), (i32(), "b".into())], i32()).finish_unchecked();
    let sum = helper.inner.append_inst(
        helper.inner.entry,
        binary(
            BinaryOperator::Subtract,
            helper.inner.params[0].into(),
            helper.inner.params[1].into(),
        ),
        i32(),
    );
    helper.inner.blocks[helper.inner.entry].term = Terminator::Return(Some(sum.into()));
    let mut caller =
        FuncBuilder::new(vec![(i32(), "x".into()), (i32(), "y".into())], i32()).finish_unchecked();
    let result = caller.inner.append_inst(
        caller.inner.entry,
        InstKind::Op {
            tag: OpTag::Call(FunctionId::from(0)),
            operands: vec![caller.inner.params[1].into(), caller.inner.params[0].into()],
        },
        i32(),
    );
    caller.inner.blocks[caller.inner.entry].term = Terminator::Return(Some(result.into()));
    inline_small_helpers(&mut caller, |_| Some(&helper));
    let [id] = caller.inner.blocks[caller.inner.entry].insts.as_slice() else {
        panic!("single inlined instruction")
    };
    let InstKind::Op {
        tag: OpTag::BinOp(BinaryOperator::Subtract),
        operands,
    } = &caller.inner.insts[*id].data
    else {
        panic!("subtraction")
    };
    assert_eq!(
        *operands,
        vec![caller.inner.params[1].into(), caller.inner.params[0].into()]
    );
}

#[test]
fn dead_instruction_worklist_removes_long_chains() {
    let mut body = FuncBuilder::new(vec![(i32(), "x".into())], i32()).finish_unchecked();
    let x = body.inner.params[0].into();
    let mut value = x;
    for _ in 0..4096 {
        value =
            body.inner.append_inst(body.inner.entry, binary(BinaryOperator::Add, value, x), i32()).into();
    }
    body.inner.blocks[body.inner.entry].term = Terminator::Return(Some(x));
    eliminate_dead_pure_instructions(&mut body);
    assert_eq!(body.num_insts(), 0);
}

#[test]
fn newly_exposed_integer_constants_wrap_at_the_declared_width() {
    use crate::types::Type;
    let ty = Type::Constructed(crate::types::TypeName::UInt(32), vec![]);
    let mut body = FuncBuilder::new(vec![], ty.clone()).finish_unchecked();
    let value = body.inner.append_inst(
        body.inner.entry,
        binary(
            BinaryOperator::Multiply,
            ValueRef::Const(ConstantValue::U32(2654435769)),
            ValueRef::Const(ConstantValue::U32(747796405)),
        ),
        ty,
    );
    body.inner.blocks[body.inner.entry].term = Terminator::Return(Some(value.into()));
    float_pure_values(&mut body);
    crate::ssa::ir::schedule_floating(&mut body.inner).unwrap();
    assert!(
        matches!(body.inner.blocks[body.inner.entry].term, Terminator::Return(Some(ValueRef::Const(ConstantValue::U32(n)))) if n == 2654435769u32.wrapping_mul(747796405))
    );
}
