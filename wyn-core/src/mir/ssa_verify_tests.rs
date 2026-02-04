#![cfg(test)]

use crate::ast::{NodeId, Span, TypeName};
use crate::mir::ssa::{Terminator, ValueId};
use crate::mir::ssa_builder::FuncBuilder;
use crate::mir::ssa_verify::*;
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
fn test_valid_simple_function() {
    let mut builder = FuncBuilder::new(
        vec![(i32_ty(), "x".to_string()), (i32_ty(), "y".to_string())],
        i32_ty(),
    );

    let x = builder.get_param(0);
    let y = builder.get_param(1);
    let sum = builder.push_binop("+", x, y, i32_ty(), dummy_span(), dummy_node()).unwrap();
    builder.terminate(Terminator::Return(sum)).unwrap();

    let body = builder.finish().unwrap();
    let result = verify_func(&body);
    assert!(result.is_ok());
}

#[test]
fn test_use_before_def() {
    let mut builder = FuncBuilder::new(vec![], i32_ty());

    // Use an undefined value
    let undefined_value = ValueId(999);
    builder.terminate(Terminator::Return(undefined_value)).unwrap();

    let body = builder.finish().unwrap();
    let result = verify_func(&body);
    assert!(result.is_err());

    let errors = result.unwrap_err();
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, VerifyError::UseBeforeDef { value, .. } if *value == undefined_value))
    );
}

#[test]
fn test_block_arg_mismatch() {
    let mut builder = FuncBuilder::new(vec![], i32_ty());

    // Create a block expecting 2 params
    let (target, _) = builder.create_block_with_params(vec![i32_ty(), i32_ty()]);

    // Branch with only 1 arg
    let one = builder.push_int("1", i32_ty(), dummy_span(), dummy_node()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target,
            args: vec![one], // Should be 2 args
        })
        .unwrap();

    // Terminate target block
    builder.switch_to_block_unchecked(target);
    let result = builder.push_int("42", i32_ty(), dummy_span(), dummy_node()).unwrap();
    builder.terminate(Terminator::Return(result)).unwrap();

    let body = builder.finish().unwrap();
    let result = verify_func(&body);
    assert!(result.is_err());

    let errors = result.unwrap_err();
    assert!(errors.iter().any(|e| matches!(
        e,
        VerifyError::BlockArgCountMismatch {
            expected: 2,
            got: 1,
            ..
        }
    )));
}

#[test]
fn test_valid_conditional() {
    let mut builder = FuncBuilder::new(vec![(i32_ty(), "x".to_string())], i32_ty());

    let x = builder.get_param(0);
    let zero = builder.push_int("0", i32_ty(), dummy_span(), dummy_node()).unwrap();
    let cond = builder
        .push_binop(
            ">",
            x,
            zero,
            Type::Constructed(TypeName::Str("bool"), vec![]),
            dummy_span(),
            dummy_node(),
        )
        .unwrap();

    let then_block = builder.create_block();
    let else_block = builder.create_block();
    let (merge, merge_params) = builder.create_block_with_params(vec![i32_ty()]);

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
            target: merge,
            args: vec![x],
        })
        .unwrap();

    builder.switch_to_block(else_block).unwrap();
    builder
        .terminate(Terminator::Branch {
            target: merge,
            args: vec![zero],
        })
        .unwrap();

    builder.switch_to_block(merge).unwrap();
    builder.terminate(Terminator::Return(merge_params[0])).unwrap();

    let body = builder.finish().unwrap();
    let result = verify_func(&body);
    assert!(result.is_ok(), "Errors: {:?}", result.err());
}
