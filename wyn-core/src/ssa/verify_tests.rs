#![cfg(test)]

use crate::ast::TypeName;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::Terminator;
use crate::ssa::verify::*;
use polytype::Type;

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
    let sum = builder.push_binop("+", x, y, i32_ty()).unwrap();
    builder.terminate(Terminator::Return(Some(sum))).unwrap();

    let body = builder.finish().unwrap();
    assert!(verify_func(&body).is_ok());
}

#[test]
fn test_use_before_def() {
    // Create a function, get a valid value, then use it where it shouldn't be reachable.
    // We can't construct a ValueId from thin air anymore (slotmap keys).
    // Instead, test that verify catches a real structural error: wrong arg count.
    // (The old test used ValueId(999) which isn't possible with slotmap.)
    let mut builder = FuncBuilder::new(vec![], i32_ty());
    let (target, _) = builder.create_block_with_params(vec![i32_ty()]);
    // Branch to target with 0 args but it expects 1
    builder.terminate(Terminator::Branch { target, args: vec![] }).unwrap();
    builder.switch_to_block_unchecked(target);
    builder.terminate(Terminator::Return(None)).unwrap();

    let body = builder.finish().unwrap();
    let result = verify_func(&body);
    assert!(result.is_err());
}

#[test]
fn test_block_arg_mismatch() {
    let mut builder = FuncBuilder::new(vec![], i32_ty());

    let (target, _) = builder.create_block_with_params(vec![i32_ty(), i32_ty()]);

    let one = builder.push_int("1", i32_ty()).unwrap();
    builder
        .terminate(Terminator::Branch {
            target,
            args: vec![one],
        })
        .unwrap();

    builder.switch_to_block_unchecked(target);
    let result = builder.push_int("42", i32_ty()).unwrap();
    builder.terminate(Terminator::Return(Some(result))).unwrap();

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
    let zero = builder.push_int("0", i32_ty()).unwrap();
    let cond = builder.push_binop(">", x, zero, Type::Constructed(TypeName::Bool, vec![])).unwrap();

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
    builder.terminate(Terminator::Return(Some(merge_params[0]))).unwrap();

    let body = builder.finish().unwrap();
    assert!(
        verify_func(&body).is_ok(),
        "Errors: {:?}",
        verify_func(&body).err()
    );
}
