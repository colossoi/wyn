use super::*;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::eliminate_dead_pure_instructions;

#[test]
fn promotion_precedes_dce_and_detaches_the_constructor_tree() {
    let f32_ty = Type::Constructed(TypeName::Float(32), vec![]);
    let vec2_ty = Type::Constructed(
        TypeName::Vec,
        vec![f32_ty, Type::Constructed(TypeName::Size(2), vec![])],
    );
    let mut builder = FuncBuilder::new(Vec::new(), vec2_ty.clone());
    let vector = builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::Vector,
                operands: vec![
                    ValueRef::Const(ConstantValue::from_f32(1.0)),
                    ValueRef::Const(ConstantValue::from_f32(2.0)),
                ],
            },
            vec2_ty.clone(),
        )
        .unwrap();
    let materialized = builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::Materialize,
                operands: vec![vector.into()],
            },
            vec2_ty,
        )
        .unwrap();
    builder.terminate(crate::ssa::types::Terminator::Return(Some(materialized.into()))).unwrap();
    let mut body = builder.finish().unwrap();
    let mut table = Vec::new();

    promote_body(&mut body, &mut table);
    eliminate_dead_pure_instructions(&mut body);

    assert_eq!(table.len(), 1);
    assert_eq!(
        body.num_insts(),
        1,
        "the detached vector constructor should be dead"
    );
    assert!(body.inner.insts.values().any(|node| matches!(
        node.data,
        InstKind::Op {
            tag: OpTag::AddressableConstant(_),
            ref operands,
        } if operands.is_empty()
    )));
}
