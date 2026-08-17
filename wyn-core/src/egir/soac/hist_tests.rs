use super::*;
use crate::egir::types::{Raw, SegBody};
use crate::FunctionId;
use std::collections::HashMap;

fn node(index: u64) -> ValueId {
    ValueId::from(slotmap::KeyData::from_ffi(index))
}

fn scalar(name: TypeName) -> Type<TypeName> {
    Type::Constructed(name, vec![])
}

fn array(element: Type<TypeName>) -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            element,
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            Type::Constructed(TypeName::Size(8), vec![]),
            crate::types::no_buffer(),
        ],
    )
}

fn general_histogram() -> (Op<Raw>, HashMap<ValueId, Type<TypeName>>) {
    let i32_type = scalar(TypeName::Int(32));
    let u32_type = scalar(TypeName::UInt(32));
    let f32_type = scalar(TypeName::Float(32));
    let bool_type = scalar(TypeName::Bool);
    let nodes = HashMap::from([
        (node(1), i32_type.clone()),
        (node(2), i32_type.clone()),
        (node(3), i32_type.clone()),
        (node(4), array(f32_type.clone())),
        (node(5), array(u32_type.clone())),
        (node(6), f32_type.clone()),
        (node(7), u32_type.clone()),
        (node(8), i32_type.clone()),
        (node(9), i32_type.clone()),
        (node(10), array(bool_type.clone())),
    ]);
    let op = Op::<Raw> {
        inputs: vec![SoacInputType::array(array(i32_type.clone()))],
        form: HistForm {
            bucket: screma::Lambda::region(
                SegBody {
                    region: FunctionId::from_index(0),
                    captures: vec![],
                },
                vec![i32_type.clone()],
                vec![
                    i32_type.clone(),
                    i32_type.clone(),
                    i32_type,
                    f32_type.clone(),
                    u32_type.clone(),
                    bool_type.clone(),
                ],
            ),
            operations: vec![
                HistOp {
                    emission: Emission::Always,
                    shape: vec![node(1), node(2)],
                    race_factor: node(3),
                    destinations: vec![ViewId::test(node(4)), ViewId::test(node(5))],
                    update: Update::Reduce {
                        operator: screma::Lambda::region(
                            SegBody {
                                region: FunctionId::from_index(1),
                                captures: vec![],
                            },
                            vec![
                                f32_type.clone(),
                                u32_type.clone(),
                                f32_type.clone(),
                                u32_type.clone(),
                            ],
                            vec![f32_type, u32_type],
                        ),
                        neutral: vec![node(6), node(7)],
                    },
                },
                HistOp {
                    emission: Emission::Always,
                    shape: vec![node(8)],
                    race_factor: node(9),
                    destinations: vec![ViewId::test(node(10))],
                    update: Update::OrderedOverwrite {
                        value_types: vec![bool_type],
                    },
                },
            ],
        },
        state: RawState,
    };
    (op, nodes)
}

#[test]
fn accepts_multiple_multidimensional_component_operations() {
    let (op, nodes) = general_histogram();
    op.validate(|node| nodes.get(&node).cloned())
        .expect("general Futhark-shaped histogram should validate");
    assert_eq!(op.form.index_count(), 3);
    assert_eq!(op.form.value_count(), 3);
}

#[test]
fn bucket_results_put_all_indices_before_all_values() {
    let (mut op, nodes) = general_histogram();
    op.form.bucket.result_types.swap(2, 3);
    let error = op
        .validate(|node| nodes.get(&node).cloned())
        .expect_err("interleaving an operation value with indices must be rejected");
    assert!(error.contains("bucket lambda"), "unexpected error: {error}");
}
