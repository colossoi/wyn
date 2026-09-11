use super::*;
use crate::egir::soac::Lambda;
use crate::egir::types::{Raw, SegBody};
use crate::types;
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
            types::no_buffer(),
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
            bucket: Lambda::region(
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
                        operator: Lambda::region(
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

#[test]
fn decoder_routes_mixed_guards_ranks_and_components() {
    let (mut op, _) = general_histogram();
    op.form.operations[1].emission = Emission::Guarded;
    let mut third = op.form.operations[0].clone();
    third.emission = Emission::Guarded;
    op.form.operations.push(third);
    let results = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let decoded = op.form.decode_results(&results).unwrap().collect::<Vec<_>>();
    assert_eq!(decoded.len(), 3);
    assert_eq!(decoded[0].guard, None);
    assert_eq!(decoded[1].guard, Some(&results[0]));
    assert_eq!(decoded[2].guard, Some(&results[1]));
    assert_eq!(decoded[0].indices, &results[2..4]);
    assert_eq!(decoded[1].indices, &results[4..5]);
    assert_eq!(decoded[2].indices, &results[5..7]);
    assert_eq!(decoded[0].values, &results[7..9]);
    assert_eq!(decoded[1].values, &results[9..10]);
    assert_eq!(decoded[2].values, &results[10..12]);
    for (decoded, operation) in decoded.iter().zip(&op.form.operations) {
        assert!(std::ptr::eq(decoded.operation, operation));
    }
}

#[test]
fn decoder_rejects_missing_and_extra_results() {
    let (op, _) = general_histogram();
    for count in [0, 5, 7] {
        let results = vec![0; count];
        let error = op.form.decode_results(&results).err().expect("invalid result arity");
        assert!(error.contains("expected 6"), "{error}");
    }
}

#[test]
fn shared_lambda_validation_rejects_invalid_bucket_and_reducer() {
    let (mut op, nodes) = general_histogram();
    op.form.bucket.body = crate::egir::soac::LambdaBody::Identity;
    let error = op.validate(|node| nodes.get(&node).cloned()).unwrap_err();
    assert!(error.contains("histogram bucket identity lambda"), "{error}");

    let (mut op, nodes) = general_histogram();
    let Update::Reduce { operator, .. } = &mut op.form.operations[0].update else {
        unreachable!()
    };
    operator.parameter_types.swap(0, 1);
    let error = op.validate(|node| nodes.get(&node).cloned()).unwrap_err();
    assert!(error.contains("operator must have type (a, a) -> a"), "{error}");
}
