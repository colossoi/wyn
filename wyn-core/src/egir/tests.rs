//! Integration tests for the aegraph pipeline.

use crate::ast::TypeName;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::{InstKind, ValueRef};
use polytype::Type;

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn bool_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Bool, vec![])
}

/// Build a simple function and roundtrip through the aegraph.
#[test]
fn roundtrip_simple_add() {
    // Build: fn f(a: i32, b: i32) -> i32 { a + b }
    let mut builder = FuncBuilder::new(vec![(i32_ty(), "a".into()), (i32_ty(), "b".into())], i32_ty());

    let a = builder.get_param(0);
    let b = builder.get_param(1);
    let sum = builder
        .push_inst(
            InstKind::BinOp {
                op: "+".into(),
                lhs: ValueRef::Ssa(a),
                rhs: ValueRef::Ssa(b),
            },
            i32_ty(),
        )
        .unwrap();
    builder.terminate(crate::ssa::framework::Terminator::Return(Some(sum))).unwrap();

    let body = builder.finish().unwrap();

    // Roundtrip through aegraph.
    let result = super::optimize_func(&body);

    // Verify the output has the right structure.
    assert_eq!(result.params.len(), 2);
    assert_eq!(result.params[0].2, "a");
    assert_eq!(result.params[1].2, "b");

    let entry = result.get_block(result.entry_block());
    // Should have at least one instruction (the add).
    assert!(!entry.insts.is_empty(), "Entry block should have instructions");

    // The last instruction before the terminator should be a BinOp.
    let last_inst = result.get_inst(*entry.insts.last().unwrap());
    assert!(
        matches!(&last_inst.data, InstKind::BinOp { op, .. } if op == "+"),
        "Expected BinOp(+), got {:?}",
        last_inst.data
    );

    // Terminator should be a return.
    assert!(matches!(
        &entry.term,
        crate::ssa::framework::Terminator::Return(Some(_))
    ));
}

/// Test that GVN deduplicates identical pure computations.
#[test]
fn gvn_deduplicates_constants() {
    // Build: fn f() -> (i32, i32) { let x = 42; let y = 42; (x, y) }
    // Using a Tuple instead of `x + y` so the constants aren't folded away.
    use crate::ast::TypeName;
    use polytype::Type;
    let pair_ty = Type::Constructed(TypeName::Tuple(2), vec![i32_ty(), i32_ty()]);
    let mut builder = FuncBuilder::new(vec![], pair_ty.clone());

    let x = builder.push_inst(InstKind::Int("42".into()), i32_ty()).unwrap();
    let y = builder.push_inst(InstKind::Int("42".into()), i32_ty()).unwrap();
    let pair =
        builder.push_inst(InstKind::Tuple(vec![ValueRef::Ssa(x), ValueRef::Ssa(y)]), pair_ty).unwrap();
    builder.terminate(crate::ssa::framework::Terminator::Return(Some(pair))).unwrap();

    let body = builder.finish().unwrap();
    let result = super::optimize_func(&body);

    let entry = result.get_block(result.entry_block());
    // GVN should deduplicate the two `42` constants into one.
    let const_count = entry
        .insts
        .iter()
        .filter(|&&iid| matches!(&result.get_inst(iid).data, InstKind::Int(s) if s == "42"))
        .count();
    assert_eq!(
        const_count, 1,
        "GVN should deduplicate identical constants; found {} copies of Int(42)",
        const_count
    );
}

/// Test that DCE removes unused pure computations.
#[test]
fn dce_removes_dead_code() {
    // Build: fn f(a: i32) -> i32 { let dead = a + a; a }
    let mut builder = FuncBuilder::new(vec![(i32_ty(), "a".into())], i32_ty());

    let a = builder.get_param(0);
    let _dead = builder
        .push_inst(
            InstKind::BinOp {
                op: "+".into(),
                lhs: ValueRef::Ssa(a),
                rhs: ValueRef::Ssa(a),
            },
            i32_ty(),
        )
        .unwrap();
    builder.terminate(crate::ssa::framework::Terminator::Return(Some(a))).unwrap();

    let body = builder.finish().unwrap();
    let result = super::optimize_func(&body);

    let entry = result.get_block(result.entry_block());
    // The dead `a + a` should not appear — DCE via demand-driven elaboration.
    let add_count = entry
        .insts
        .iter()
        .filter(|&&iid| matches!(&result.get_inst(iid).data, InstKind::BinOp { .. }))
        .count();
    assert_eq!(
        add_count, 0,
        "DCE should remove unused add; found {} BinOp instructions",
        add_count
    );
}

/// Test if/else control flow roundtrip.
#[test]
fn roundtrip_if_else() {
    // fn f(c: bool, a: i32, b: i32) -> i32 { if c then a else b }
    let mut builder = FuncBuilder::new(
        vec![
            (bool_ty(), "c".into()),
            (i32_ty(), "a".into()),
            (i32_ty(), "b".into()),
        ],
        i32_ty(),
    );

    let c = builder.get_param(0);
    let a = builder.get_param(1);
    let b = builder.get_param(2);

    let ite = builder.create_if_then_else(i32_ty());

    builder
        .terminate(crate::ssa::framework::Terminator::CondBranch {
            cond: c,
            then_target: ite.then_block,
            then_args: vec![],
            else_target: ite.else_block,
            else_args: vec![],
        })
        .unwrap();

    // Then block: return a
    builder.switch_to_block(ite.then_block).unwrap();
    builder
        .terminate(crate::ssa::framework::Terminator::Branch {
            target: ite.merge_block,
            args: vec![a],
        })
        .unwrap();

    // Else block: return b
    builder.switch_to_block(ite.else_block).unwrap();
    builder
        .terminate(crate::ssa::framework::Terminator::Branch {
            target: ite.merge_block,
            args: vec![b],
        })
        .unwrap();

    // Merge block: return result
    builder.switch_to_block(ite.merge_block).unwrap();
    builder.terminate(crate::ssa::framework::Terminator::Return(Some(ite.result))).unwrap();

    let body = builder.finish().unwrap();
    let result = super::optimize_func(&body);

    // Should have 4 blocks: entry, then, else, merge.
    assert_eq!(result.inner.blocks.len(), 4, "Expected 4 blocks for if/else");

    // Entry should have a CondBranch terminator.
    let entry = result.get_block(result.entry_block());
    assert!(
        matches!(&entry.term, crate::ssa::framework::Terminator::CondBranch { .. }),
        "Entry should end with CondBranch"
    );
}
