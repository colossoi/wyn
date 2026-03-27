#![cfg(test)]

use crate::ast::TypeName;
use crate::ssa::builder::*;
use crate::ssa::types::Terminator;
use polytype::Type;

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn bool_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Bool, vec![])
}

#[test]
fn test_simple_function() {
    let mut builder = FuncBuilder::new(
        vec![(i32_ty(), "x".to_string()), (i32_ty(), "y".to_string())],
        i32_ty(),
    );

    let x = builder.get_param(0);
    let y = builder.get_param(1);

    let result = builder.push_binop("+", x, y, i32_ty()).unwrap();

    builder.terminate(Terminator::Return(Some(result))).unwrap();

    let body = builder.finish().unwrap();

    assert_eq!(body.params.len(), 2);
    assert_eq!(body.num_blocks(), 1);
    assert_eq!(body.num_insts(), 1);
    assert_eq!(body.num_values(), 3); // x, y, result
}

#[test]
fn test_conditional() {
    let mut builder = FuncBuilder::new(
        vec![(i32_ty(), "x".to_string()), (i32_ty(), "y".to_string())],
        i32_ty(),
    );

    let x = builder.get_param(0);
    let y = builder.get_param(1);

    let then_block = builder.create_block();
    let else_block = builder.create_block();
    let (merge_block, merge_params) = builder.create_block_with_params(vec![i32_ty()]);
    let result = merge_params[0];

    let cond = builder.push_binop(">", x, y, bool_ty()).unwrap();
    builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: then_block,
            then_args: vec![],
            else_target: else_block,
            else_args: vec![],
        })
        .unwrap();

    builder.switch_to_block(then_block).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: merge_block,
            args: vec![x],
        })
        .unwrap();

    builder.switch_to_block(else_block).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: merge_block,
            args: vec![y],
        })
        .unwrap();

    builder.switch_to_block(merge_block).unwrap();
    builder.terminate(Terminator::Return(Some(result))).unwrap();

    let body = builder.finish().unwrap();

    assert_eq!(body.num_blocks(), 4);
    assert_eq!(body.get_block(merge_block).params.len(), 1);
}

#[test]
fn test_unterminated_block_error() {
    let mut builder = FuncBuilder::new(vec![], Type::Constructed(TypeName::Unit, vec![]));
    let block2 = builder.create_block();
    let result = builder.switch_to_block(block2);
    assert!(matches!(result, Err(BuilderError::UnterminatedBlock(_))));
}

#[test]
fn test_finish_unterminated_error() {
    let mut builder = FuncBuilder::new(vec![], Type::Constructed(TypeName::Unit, vec![]));
    // Create a user block but don't terminate it
    let _block = builder.create_block();
    builder.terminate(Terminator::Return(None)).unwrap();
    builder.switch_to_block(_block).unwrap();
    // _block is now current but unterminated
    let result = builder.finish();
    assert!(matches!(result, Err(BuilderError::UnterminatedBlock(_))));
}

#[test]
fn test_if_then_else_pattern() {
    let mut builder = FuncBuilder::new(vec![(i32_ty(), "x".to_string())], i32_ty());

    let x = builder.get_param(0);
    let if_blocks = builder.create_if_then_else(i32_ty());

    let zero = builder.push_int("0", i32_ty()).unwrap();
    let cond = builder.push_binop("<", x, zero, bool_ty()).unwrap();
    builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: if_blocks.then_block,
            then_args: vec![],
            else_target: if_blocks.else_block,
            else_args: vec![],
        })
        .unwrap();

    builder.switch_to_block(if_blocks.then_block).unwrap();
    let neg_x = builder.push_unary("-", x, i32_ty()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: if_blocks.merge_block,
            args: vec![neg_x],
        })
        .unwrap();

    builder.switch_to_block(if_blocks.else_block).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: if_blocks.merge_block,
            args: vec![x],
        })
        .unwrap();

    builder.switch_to_block(if_blocks.merge_block).unwrap();
    builder.terminate(Terminator::Return(Some(if_blocks.result))).unwrap();

    let body = builder.finish().unwrap();

    assert_eq!(body.num_blocks(), 4);
    assert_eq!(body.get_block(if_blocks.merge_block).params.len(), 1);

    let verify_result = crate::ssa::verify::verify_func(&body);
    assert!(
        verify_result.is_ok(),
        "Verification failed: {:?}",
        verify_result.err()
    );
}

#[test]
fn test_while_loop_pattern() {
    let mut builder = FuncBuilder::new(vec![(i32_ty(), "n".to_string())], i32_ty());

    let n = builder.get_param(0);
    let loop_blocks = builder.create_while_loop(i32_ty());

    let zero = builder.push_int("0", i32_ty()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![zero],
        })
        .unwrap();

    builder.switch_to_block(loop_blocks.header).unwrap();
    let cond = builder.push_binop("<", loop_blocks.acc, n, bool_ty()).unwrap();
    builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: loop_blocks.body,
            then_args: vec![],
            else_target: loop_blocks.exit,
            else_args: vec![loop_blocks.acc],
        })
        .unwrap();

    builder.switch_to_block(loop_blocks.body).unwrap();
    let one = builder.push_int("1", i32_ty()).unwrap();
    let new_acc = builder.push_binop("+", loop_blocks.acc, one, i32_ty()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![new_acc],
        })
        .unwrap();

    builder.switch_to_block(loop_blocks.exit).unwrap();
    builder.terminate(Terminator::Return(Some(loop_blocks.result))).unwrap();

    let body = builder.finish().unwrap();

    assert_eq!(body.num_blocks(), 4);
    assert_eq!(body.get_block(loop_blocks.header).params.len(), 1);
    assert_eq!(body.get_block(loop_blocks.exit).params.len(), 1);

    let verify_result = crate::ssa::verify::verify_func(&body);
    assert!(
        verify_result.is_ok(),
        "Verification failed: {:?}",
        verify_result.err()
    );
}

#[test]
fn test_for_range_loop_pattern() {
    let mut builder = FuncBuilder::new(vec![(i32_ty(), "n".to_string())], i32_ty());

    let n = builder.get_param(0);
    let loop_blocks = builder.create_for_range_loop(i32_ty());

    let zero = builder.push_int("0", i32_ty()).unwrap();
    let zero2 = builder.push_int("0", i32_ty()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![zero, zero2],
        })
        .unwrap();

    builder.switch_to_block(loop_blocks.header).unwrap();
    let cond = builder.push_binop("<", loop_blocks.index, n, bool_ty()).unwrap();
    builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: loop_blocks.body,
            then_args: vec![],
            else_target: loop_blocks.exit,
            else_args: vec![loop_blocks.acc],
        })
        .unwrap();

    builder.switch_to_block(loop_blocks.body).unwrap();
    let new_acc = builder.push_binop("+", loop_blocks.acc, loop_blocks.index, i32_ty()).unwrap();
    let one = builder.push_int("1", i32_ty()).unwrap();
    let next_i = builder.push_binop("+", loop_blocks.index, one, i32_ty()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![new_acc, next_i],
        })
        .unwrap();

    builder.switch_to_block(loop_blocks.exit).unwrap();
    builder.terminate(Terminator::Return(Some(loop_blocks.result))).unwrap();

    let body = builder.finish().unwrap();

    assert_eq!(body.num_blocks(), 4);
    assert_eq!(body.get_block(loop_blocks.header).params.len(), 2);
    assert_eq!(body.get_block(loop_blocks.exit).params.len(), 1);

    let verify_result = crate::ssa::verify::verify_func(&body);
    assert!(
        verify_result.is_ok(),
        "Verification failed: {:?}",
        verify_result.err()
    );
}

#[test]
fn test_nested_if_in_loop() {
    let mut builder = FuncBuilder::new(
        vec![
            (i32_ty(), "arr_len".to_string()),
            (i32_ty(), "threshold".to_string()),
        ],
        i32_ty(),
    );

    let arr_len = builder.get_param(0);
    let threshold = builder.get_param(1);

    let loop_blocks = builder.create_for_range_loop(i32_ty());

    let zero = builder.push_int("0", i32_ty()).unwrap();
    let zero2 = builder.push_int("0", i32_ty()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![zero, zero2],
        })
        .unwrap();

    builder.switch_to_block(loop_blocks.header).unwrap();
    let loop_cond = builder.push_binop("<", loop_blocks.index, arr_len, bool_ty()).unwrap();

    let if_blocks = builder.create_if_then_else(i32_ty());

    builder
        .terminate(Terminator::CondBranch {
            cond: loop_cond,
            then_target: loop_blocks.body,
            then_args: vec![],
            else_target: loop_blocks.exit,
            else_args: vec![loop_blocks.acc],
        })
        .unwrap();

    builder.switch_to_block(loop_blocks.body).unwrap();
    let if_cond = builder.push_binop(">", loop_blocks.index, threshold, bool_ty()).unwrap();
    builder
        .terminate(Terminator::CondBranch {
            cond: if_cond,
            then_target: if_blocks.then_block,
            then_args: vec![],
            else_target: if_blocks.else_block,
            else_args: vec![],
        })
        .unwrap();

    builder.switch_to_block(if_blocks.then_block).unwrap();
    let one = builder.push_int("1", i32_ty()).unwrap();
    let count_plus_one = builder.push_binop("+", loop_blocks.acc, one, i32_ty()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: if_blocks.merge_block,
            args: vec![count_plus_one],
        })
        .unwrap();

    builder.switch_to_block(if_blocks.else_block).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: if_blocks.merge_block,
            args: vec![loop_blocks.acc],
        })
        .unwrap();

    builder.switch_to_block(if_blocks.merge_block).unwrap();
    let one2 = builder.push_int("1", i32_ty()).unwrap();
    let next_i = builder.push_binop("+", loop_blocks.index, one2, i32_ty()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![if_blocks.result, next_i],
        })
        .unwrap();

    builder.switch_to_block(loop_blocks.exit).unwrap();
    builder.terminate(Terminator::Return(Some(loop_blocks.result))).unwrap();

    let body = builder.finish().unwrap();

    assert_eq!(body.num_blocks(), 7);

    let verify_result = crate::ssa::verify::verify_func(&body);
    assert!(
        verify_result.is_ok(),
        "Verification failed: {:?}",
        verify_result.err()
    );
}
