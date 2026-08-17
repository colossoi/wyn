use super::*;
use crate::egir::types::{Raw, RegionId, Semantic};

fn scalar(name: TypeName) -> Type<TypeName> {
    Type::Constructed(name, vec![])
}

fn array(element: Type<TypeName>) -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            element,
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            Type::Constructed(TypeName::Size(4), vec![]),
            crate::types::no_buffer(),
        ],
    )
}

fn region(index: usize, parameters: Vec<Type<TypeName>>, results: Vec<Type<TypeName>>) -> Lambda {
    Lambda::region(
        SegBody {
            region: RegionId::from_index(index as u32),
            captures: vec![],
        },
        parameters,
        results,
    )
}

fn node(index: u64) -> ValueId {
    ValueId::from(slotmap::KeyData::from_ffi(index))
}

fn valid_scan_op() -> Op<Raw> {
    let i32_type = scalar(TypeName::Int(32));
    Op {
        inputs: vec![SoacInputType::array(array(i32_type.clone()))],
        form: ScremaForm {
            pre: region(0, vec![i32_type.clone()], vec![i32_type.clone()]),
            scans: vec![Scan {
                operator: region(
                    1,
                    vec![i32_type.clone(), i32_type.clone()],
                    vec![i32_type.clone()],
                ),
                neutral: vec![node(1)],
            }],
            reductions: vec![],
            post: Lambda::identity(vec![i32_type]),
        },
        result_state: vec![ResultState {
            ownership: SoacOwnership::Fresh,
        }],
        state: RawState,
    }
}
#[test]
fn mixed_form_uses_futhark_result_and_post_parameter_order() {
    let i32_type = scalar(TypeName::Int(32));
    let u32_type = scalar(TypeName::UInt(32));
    let bool_type = scalar(TypeName::Bool);
    let array = array(i32_type.clone());
    let form = ScremaForm {
        pre: region(
            0,
            vec![i32_type.clone()],
            vec![
                i32_type.clone(),
                i32_type.clone(),
                u32_type.clone(),
                bool_type.clone(),
            ],
        ),
        scans: vec![Scan {
            operator: region(
                1,
                vec![
                    i32_type.clone(),
                    i32_type.clone(),
                    i32_type.clone(),
                    i32_type.clone(),
                ],
                vec![i32_type.clone(), i32_type.clone()],
            ),
            neutral: vec![
                ValueId::from(slotmap::KeyData::from_ffi(1)),
                ValueId::from(slotmap::KeyData::from_ffi(2)),
            ],
        }],
        reductions: vec![Reduce {
            operator: region(
                2,
                vec![u32_type.clone(), u32_type.clone()],
                vec![u32_type.clone()],
            ),
            neutral: vec![ValueId::from(slotmap::KeyData::from_ffi(3))],
            commutative: false,
        }],
        post: region(
            3,
            vec![i32_type.clone(), i32_type.clone(), bool_type.clone()],
            vec![bool_type.clone(), i32_type.clone()],
        ),
    };
    let op = Op::<Raw> {
        inputs: vec![SoacInputType::array(array)],
        form,
        result_state: vec![
            ResultState {
                ownership: SoacOwnership::Fresh,
            },
            ResultState {
                ownership: SoacOwnership::Fresh,
            },
            ResultState {
                ownership: SoacOwnership::Fresh,
            },
        ],
        state: RawState,
    };

    op.validate().unwrap();
    assert_eq!(
        op.form.post_input_types().unwrap(),
        vec![i32_type.clone(), i32_type, bool_type]
    );
    assert_eq!(
        op.form.result_id(0),
        Some(ResultId::Reduction {
            reduction: 0,
            component: 0,
        })
    );
    assert_eq!(op.form.result_id(1), Some(ResultId::Post(0)));
    assert_eq!(op.form.result_id(2), Some(ResultId::Post(1)));
}

#[test]
fn post_lambda_without_scan_is_not_canonical() {
    let i32_type = scalar(TypeName::Int(32));
    let bool_type = scalar(TypeName::Bool);
    let array = array(i32_type.clone());
    let op = Op::<Raw> {
        inputs: vec![SoacInputType::array(array)],
        form: ScremaForm {
            pre: region(0, vec![i32_type.clone()], vec![i32_type.clone()]),
            scans: vec![],
            reductions: vec![],
            post: region(1, vec![i32_type], vec![bool_type]),
        },
        result_state: vec![ResultState {
            ownership: SoacOwnership::Fresh,
        }],
        state: RawState,
    };

    assert!(op.validate().unwrap_err().contains("post-lambda but no scans"));
}
#[test]
fn validation_covers_input_operator_and_phase_result_contracts() {
    let mut wrong_input = valid_scan_op();
    wrong_input.form.pre.parameter_types = vec![scalar(TypeName::UInt(32))];

    let mut wrong_operator = valid_scan_op();
    wrong_operator.form.scans[0].operator.parameter_types.pop();

    let mut wrong_result_state = valid_scan_op();
    wrong_result_state.result_state.clear();

    for (case, op, expected) in [
        ("input", wrong_input, "pre-lambda parameters"),
        ("operator", wrong_operator, "operator must have 2 parameters"),
        ("phase result", wrong_result_state, "phase state describes 0"),
    ] {
        let error = op.validate().unwrap_err();
        assert!(
            error.contains(expected),
            "{case} validation produced {error:?}, expected it to contain {expected:?}"
        );
    }
}

#[test]
fn node_traversal_covers_every_lambda_and_neutral() {
    let unit = scalar(TypeName::Unit);
    let mut op = Op::<Semantic> {
        inputs: vec![SoacInputType::array(array(unit.clone()))],
        form: ScremaForm {
            pre: Lambda::region(
                SegBody {
                    region: RegionId::from_index(0),
                    captures: vec![OperandRef::Value(node(1))],
                },
                vec![unit.clone()],
                vec![unit.clone(), unit.clone()],
            ),
            scans: vec![Scan {
                operator: Lambda::region(
                    SegBody {
                        region: RegionId::from_index(1),
                        captures: vec![OperandRef::Value(node(2))],
                    },
                    vec![unit.clone(), unit.clone()],
                    vec![unit.clone()],
                ),
                neutral: vec![node(3)],
            }],
            reductions: vec![Reduce {
                operator: Lambda::region(
                    SegBody {
                        region: RegionId::from_index(2),
                        captures: vec![OperandRef::Value(node(4))],
                    },
                    vec![unit.clone(), unit.clone()],
                    vec![unit.clone()],
                ),
                neutral: vec![node(5)],
                commutative: false,
            }],
            post: Lambda::region(
                SegBody {
                    region: RegionId::from_index(3),
                    captures: vec![OperandRef::Value(node(6))],
                },
                vec![unit],
                vec![],
            ),
        },
        result_state: vec![ResultState {
            ownership: SoacOwnership::Fresh,
        }],
        state: SemanticState::Serial,
    };

    assert_eq!(
        op.referenced_nodes(),
        vec![node(1), node(2), node(4), node(6), node(3), node(5)]
    );

    let mut index = 0;
    op.remap_referenced_values(|_| {
        let replacement = node(10 + index);
        index += 1;
        replacement
    });
    assert_eq!(
        op.referenced_nodes(),
        vec![node(10), node(11), node(13), node(15), node(12), node(14)]
    );
}
