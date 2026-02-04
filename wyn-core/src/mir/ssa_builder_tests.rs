#![cfg(test)]

use crate::ast::{NodeId, Span, TypeName};
use crate::mir::ssa::Terminator;
use crate::mir::ssa_builder::*;
use polytype::Type;

fn dummy_span() -> Span {
    Span {
        start_line: 1,
        start_col: 1,
        end_line: 1,
        end_col: 1,
    }
}

fn dummy_node() -> NodeId {
    NodeId(0)
}

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

#[test]
fn test_simple_function() {
    // Build: fn add(x: i32, y: i32) -> i32 { x + y }
    let mut builder = FuncBuilder::new(
        vec![(i32_ty(), "x".to_string()), (i32_ty(), "y".to_string())],
        i32_ty(),
    );

    let x = builder.get_param(0);
    let y = builder.get_param(1);

    let result = builder.push_binop("+", x, y, i32_ty(), dummy_span(), dummy_node()).unwrap();

    builder.terminate(Terminator::Return(result)).unwrap();

    let body = builder.finish().unwrap();

    assert_eq!(body.params.len(), 2);
    assert_eq!(body.num_blocks(), 1);
    assert_eq!(body.num_insts(), 1);
    assert_eq!(body.num_values(), 3); // x, y, result
}

#[test]
fn test_conditional() {
    // Build: fn max(x: i32, y: i32) -> i32 { if x > y then x else y }
    let mut builder = FuncBuilder::new(
        vec![(i32_ty(), "x".to_string()), (i32_ty(), "y".to_string())],
        i32_ty(),
    );

    let x = builder.get_param(0);
    let y = builder.get_param(1);

    // Create blocks
    let then_block = builder.create_block();
    let else_block = builder.create_block();
    let (merge_block, merge_params) = builder.create_block_with_params(vec![i32_ty()]);
    let result = merge_params[0];

    // Entry: compare and branch
    let cond = builder
        .push_binop(
            ">",
            x,
            y,
            Type::Constructed(TypeName::Str("bool"), vec![]),
            dummy_span(),
            dummy_node(),
        )
        .unwrap();
    builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: then_block,
            then_args: vec![],
            else_target: else_block,
            else_args: vec![],
        })
        .unwrap();

    // Then block: return x
    builder.switch_to_block(then_block).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: merge_block,
            args: vec![x],
        })
        .unwrap();

    // Else block: return y
    builder.switch_to_block(else_block).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: merge_block,
            args: vec![y],
        })
        .unwrap();

    // Merge block: return result
    builder.switch_to_block(merge_block).unwrap();
    builder.terminate(Terminator::Return(result)).unwrap();

    let body = builder.finish().unwrap();

    assert_eq!(body.num_blocks(), 4); // entry, then, else, merge
    assert_eq!(body.get_block(merge_block).params.len(), 1);
}

#[test]
fn test_unterminated_block_error() {
    let mut builder = FuncBuilder::new(vec![], Type::Constructed(TypeName::Unit, vec![]));

    let block2 = builder.create_block();

    // Try to switch without terminating entry block
    let result = builder.switch_to_block(block2);
    assert!(matches!(result, Err(BuilderError::UnterminatedBlock(_))));
}

#[test]
fn test_finish_unterminated_error() {
    let builder = FuncBuilder::new(vec![], Type::Constructed(TypeName::Unit, vec![]));

    // Entry block not terminated
    let result = builder.finish();
    assert!(matches!(result, Err(BuilderError::UnterminatedBlock(_))));
}

// =========================================================================
// Control Flow Pattern Tests
// =========================================================================

fn bool_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Str("bool"), vec![])
}

#[test]
fn test_if_then_else_pattern() {
    // Build: fn abs(x: i32) -> i32 { if x < 0 then -x else x }
    let mut builder = FuncBuilder::new(vec![(i32_ty(), "x".to_string())], i32_ty());

    let x = builder.get_param(0);

    // Create the if-then-else structure
    let if_blocks = builder.create_if_then_else(i32_ty());

    // Entry: compare x < 0 and branch
    let zero = builder.push_int("0", i32_ty(), dummy_span(), dummy_node()).unwrap();
    let cond = builder.push_binop("<", x, zero, bool_ty(), dummy_span(), dummy_node()).unwrap();
    builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: if_blocks.then_block,
            then_args: vec![],
            else_target: if_blocks.else_block,
            else_args: vec![],
        })
        .unwrap();

    // Then block: compute -x and branch to merge
    builder.switch_to_block(if_blocks.then_block).unwrap();
    let neg_x = builder.push_unary("-", x, i32_ty(), dummy_span(), dummy_node()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: if_blocks.merge_block,
            args: vec![neg_x],
        })
        .unwrap();

    // Else block: use x and branch to merge
    builder.switch_to_block(if_blocks.else_block).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: if_blocks.merge_block,
            args: vec![x],
        })
        .unwrap();

    // Merge block: return result
    builder.switch_to_block(if_blocks.merge_block).unwrap();
    builder.terminate(Terminator::Return(if_blocks.result)).unwrap();

    let body = builder.finish().unwrap();

    // Verify structure
    assert_eq!(body.num_blocks(), 4); // entry, then, else, merge
    assert_eq!(body.get_block(if_blocks.merge_block).params.len(), 1);

    // Verify with verifier
    let verify_result = crate::mir::ssa_verify::verify_func(&body);
    assert!(
        verify_result.is_ok(),
        "Verification failed: {:?}",
        verify_result.err()
    );
}

#[test]
fn test_while_loop_pattern() {
    // Build: fn sum_to_n(n: i32) -> i32 { loop acc = 0 while acc < n do acc + 1 }
    // (This is a silly example - it's really "count to n" - but demonstrates the pattern)
    let mut builder = FuncBuilder::new(vec![(i32_ty(), "n".to_string())], i32_ty());

    let n = builder.get_param(0);

    // Create the while loop structure
    let loop_blocks = builder.create_while_loop(i32_ty());

    // Entry: branch to header with initial value 0
    let zero = builder.push_int("0", i32_ty(), dummy_span(), dummy_node()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![zero],
        })
        .unwrap();

    // Header: check acc < n
    builder.switch_to_block(loop_blocks.header).unwrap();
    let cond = builder.push_binop("<", loop_blocks.acc, n, bool_ty(), dummy_span(), dummy_node()).unwrap();
    builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: loop_blocks.body,
            then_args: vec![],
            else_target: loop_blocks.exit,
            else_args: vec![loop_blocks.acc],
        })
        .unwrap();

    // Body: compute acc + 1
    builder.switch_to_block(loop_blocks.body).unwrap();
    let one = builder.push_int("1", i32_ty(), dummy_span(), dummy_node()).unwrap();
    let new_acc =
        builder.push_binop("+", loop_blocks.acc, one, i32_ty(), dummy_span(), dummy_node()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![new_acc],
        })
        .unwrap();

    // Exit: return result
    builder.switch_to_block(loop_blocks.exit).unwrap();
    builder.terminate(Terminator::Return(loop_blocks.result)).unwrap();

    let body = builder.finish().unwrap();

    // Verify structure
    assert_eq!(body.num_blocks(), 4); // entry, header, body, exit
    assert_eq!(body.get_block(loop_blocks.header).params.len(), 1); // acc
    assert_eq!(body.get_block(loop_blocks.exit).params.len(), 1); // result

    // Verify with verifier
    let verify_result = crate::mir::ssa_verify::verify_func(&body);
    assert!(
        verify_result.is_ok(),
        "Verification failed: {:?}",
        verify_result.err()
    );
}

#[test]
fn test_for_range_loop_pattern() {
    // Build: fn sum_range(n: i32) -> i32 { loop acc = 0 for i < n do acc + i }
    let mut builder = FuncBuilder::new(vec![(i32_ty(), "n".to_string())], i32_ty());

    let n = builder.get_param(0);

    // Create the for-range loop structure
    let loop_blocks = builder.create_for_range_loop(i32_ty());

    // Entry: branch to header with (0, 0)
    let zero = builder.push_int("0", i32_ty(), dummy_span(), dummy_node()).unwrap();
    let zero2 = builder.push_int("0", i32_ty(), dummy_span(), dummy_node()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![zero, zero2],
        })
        .unwrap();

    // Header: check i < n
    builder.switch_to_block(loop_blocks.header).unwrap();
    let cond =
        builder.push_binop("<", loop_blocks.index, n, bool_ty(), dummy_span(), dummy_node()).unwrap();
    builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: loop_blocks.body,
            then_args: vec![],
            else_target: loop_blocks.exit,
            else_args: vec![loop_blocks.acc],
        })
        .unwrap();

    // Body: compute acc + i, increment i
    builder.switch_to_block(loop_blocks.body).unwrap();
    let new_acc = builder
        .push_binop(
            "+",
            loop_blocks.acc,
            loop_blocks.index,
            i32_ty(),
            dummy_span(),
            dummy_node(),
        )
        .unwrap();
    let one = builder.push_int("1", i32_ty(), dummy_span(), dummy_node()).unwrap();
    let next_i =
        builder.push_binop("+", loop_blocks.index, one, i32_ty(), dummy_span(), dummy_node()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![new_acc, next_i],
        })
        .unwrap();

    // Exit: return result
    builder.switch_to_block(loop_blocks.exit).unwrap();
    builder.terminate(Terminator::Return(loop_blocks.result)).unwrap();

    let body = builder.finish().unwrap();

    // Verify structure
    assert_eq!(body.num_blocks(), 4); // entry, header, body, exit
    assert_eq!(body.get_block(loop_blocks.header).params.len(), 2); // acc, i
    assert_eq!(body.get_block(loop_blocks.exit).params.len(), 1); // result

    // Verify with verifier
    let verify_result = crate::mir::ssa_verify::verify_func(&body);
    assert!(
        verify_result.is_ok(),
        "Verification failed: {:?}",
        verify_result.err()
    );
}

#[test]
fn test_nested_if_in_loop() {
    // Build a more complex example: loop with if inside
    // fn count_positive(arr_len: i32, threshold: i32) -> i32 {
    //     loop count = 0 for i < arr_len do
    //         if i > threshold then count + 1 else count
    // }
    let mut builder = FuncBuilder::new(
        vec![
            (i32_ty(), "arr_len".to_string()),
            (i32_ty(), "threshold".to_string()),
        ],
        i32_ty(),
    );

    let arr_len = builder.get_param(0);
    let threshold = builder.get_param(1);

    // Create for-range loop
    let loop_blocks = builder.create_for_range_loop(i32_ty());

    // Entry: branch to header with (0, 0)
    let zero = builder.push_int("0", i32_ty(), dummy_span(), dummy_node()).unwrap();
    let zero2 = builder.push_int("0", i32_ty(), dummy_span(), dummy_node()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![zero, zero2],
        })
        .unwrap();

    // Header: check i < arr_len
    builder.switch_to_block(loop_blocks.header).unwrap();
    let loop_cond = builder
        .push_binop(
            "<",
            loop_blocks.index,
            arr_len,
            bool_ty(),
            dummy_span(),
            dummy_node(),
        )
        .unwrap();

    // Create if-then-else for the body (we need to create it while in header to get proper block IDs)
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

    // Body: if i > threshold then count + 1 else count
    builder.switch_to_block(loop_blocks.body).unwrap();
    let if_cond = builder
        .push_binop(
            ">",
            loop_blocks.index,
            threshold,
            bool_ty(),
            dummy_span(),
            dummy_node(),
        )
        .unwrap();
    builder
        .terminate(Terminator::CondBranch {
            cond: if_cond,
            then_target: if_blocks.then_block,
            then_args: vec![],
            else_target: if_blocks.else_block,
            else_args: vec![],
        })
        .unwrap();

    // If-then: count + 1
    builder.switch_to_block(if_blocks.then_block).unwrap();
    let one = builder.push_int("1", i32_ty(), dummy_span(), dummy_node()).unwrap();
    let count_plus_one =
        builder.push_binop("+", loop_blocks.acc, one, i32_ty(), dummy_span(), dummy_node()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: if_blocks.merge_block,
            args: vec![count_plus_one],
        })
        .unwrap();

    // If-else: count (unchanged)
    builder.switch_to_block(if_blocks.else_block).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: if_blocks.merge_block,
            args: vec![loop_blocks.acc],
        })
        .unwrap();

    // If-merge: branch back to loop header with new_acc and next_i
    builder.switch_to_block(if_blocks.merge_block).unwrap();
    let one2 = builder.push_int("1", i32_ty(), dummy_span(), dummy_node()).unwrap();
    let next_i =
        builder.push_binop("+", loop_blocks.index, one2, i32_ty(), dummy_span(), dummy_node()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: loop_blocks.header,
            args: vec![if_blocks.result, next_i],
        })
        .unwrap();

    // Loop exit: return result
    builder.switch_to_block(loop_blocks.exit).unwrap();
    builder.terminate(Terminator::Return(loop_blocks.result)).unwrap();

    let body = builder.finish().unwrap();

    // Verify structure: entry, header, body, if_then, if_else, if_merge, exit = 7 blocks
    assert_eq!(body.num_blocks(), 7);

    // Verify with verifier
    let verify_result = crate::mir::ssa_verify::verify_func(&body);
    assert!(
        verify_result.is_ok(),
        "Verification failed: {:?}",
        verify_result.err()
    );
}
