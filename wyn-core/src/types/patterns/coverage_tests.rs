//! Tests for the Maranget usefulness algorithm and `check_match`.
//!
//! The matrices and queries are built directly in `CovPat` form so
//! the algorithm can be tested in isolation from the AST → CovPat
//! lowering. End-to-end coverage of the lowering happens via
//! `types::checker_tests` once Phase 2 wires `check_match` into the
//! checker.

use super::*;
use crate::ast::Span;
use crate::types::{Type, TypeName, sum, tuple};

fn dummy_span() -> Span {
    Span::dummy()
}

fn i32_ty() -> Type {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn bool_ty() -> Type {
    Type::Constructed(TypeName::Bool, vec![])
}

fn unit_ty() -> Type {
    Type::Constructed(TypeName::Unit, vec![])
}

fn sum_ab() -> Type {
    sum(vec![("a".to_string(), vec![]), ("b".to_string(), vec![])])
}

fn sum_ab_payload() -> Type {
    sum(vec![
        ("left".to_string(), vec![i32_ty()]),
        ("right".to_string(), vec![bool_ty()]),
    ])
}

#[test]
fn empty_matrix_makes_any_query_useful() {
    let m: Vec<Vec<CovPat>> = vec![];
    let q = vec![CovPat::Wild];
    let result = useful(&m, &q, &[i32_ty()]);
    assert!(result.is_some(), "any query is useful against an empty matrix");
}

#[test]
fn wild_covers_wild() {
    let m = vec![vec![CovPat::Wild]];
    let q = vec![CovPat::Wild];
    assert!(useful(&m, &q, &[i32_ty()]).is_none(), "wild is covered by wild");
}

#[test]
fn sum_two_variants_both_present_is_exhaustive() {
    let arms = vec![
        (
            ast::Pattern {
                h: ast::Header {
                    id: ast::NodeId(0),
                    span: dummy_span(),
                },
                kind: ast::PatternKind::Constructor("a".to_string(), vec![]),
            },
            dummy_span(),
        ),
        (
            ast::Pattern {
                h: ast::Header {
                    id: ast::NodeId(1),
                    span: dummy_span(),
                },
                kind: ast::PatternKind::Constructor("b".to_string(), vec![]),
            },
            dummy_span(),
        ),
    ];
    let result = check_match(&sum_ab(), &arms, dummy_span());
    assert!(
        result.is_ok(),
        "match on #a | #b with both arms is exhaustive: got {:?}",
        result
    );
}

#[test]
fn sum_missing_variant_is_non_exhaustive() {
    let arms = vec![(
        ast::Pattern {
            h: ast::Header {
                id: ast::NodeId(0),
                span: dummy_span(),
            },
            kind: ast::PatternKind::Constructor("a".to_string(), vec![]),
        },
        dummy_span(),
    )];
    let result = check_match(&sum_ab(), &arms, dummy_span());
    match result {
        Err(CoverageError::NonExhaustive { missing, .. }) => {
            // Witness should name the missing constructor #b.
            match missing {
                CovPat::Ctor(name, _) => assert_eq!(name, "b", "witness should be #b"),
                other => panic!("expected Ctor witness, got {:?}", other),
            }
        }
        other => panic!("expected NonExhaustive, got {:?}", other),
    }
}

#[test]
fn wildcard_arm_makes_sum_exhaustive() {
    let arms = vec![
        (
            ast::Pattern {
                h: ast::Header {
                    id: ast::NodeId(0),
                    span: dummy_span(),
                },
                kind: ast::PatternKind::Constructor("a".to_string(), vec![]),
            },
            dummy_span(),
        ),
        (
            ast::Pattern {
                h: ast::Header {
                    id: ast::NodeId(1),
                    span: dummy_span(),
                },
                kind: ast::PatternKind::Wildcard,
            },
            dummy_span(),
        ),
    ];
    let result = check_match(&sum_ab(), &arms, dummy_span());
    assert!(result.is_ok(), "wildcard after #a covers #b: got {:?}", result);
}

#[test]
fn redundant_arm_after_wildcard_is_flagged() {
    let arms = vec![
        (
            ast::Pattern {
                h: ast::Header {
                    id: ast::NodeId(0),
                    span: dummy_span(),
                },
                kind: ast::PatternKind::Wildcard,
            },
            dummy_span(),
        ),
        (
            ast::Pattern {
                h: ast::Header {
                    id: ast::NodeId(1),
                    span: dummy_span(),
                },
                kind: ast::PatternKind::Constructor("a".to_string(), vec![]),
            },
            dummy_span(),
        ),
    ];
    let result = check_match(&sum_ab(), &arms, dummy_span());
    match result {
        Err(CoverageError::Redundant { arm_index, .. }) => {
            assert_eq!(arm_index, 1, "arm 1 (#a after wildcard) should be redundant");
        }
        other => panic!("expected Redundant, got {:?}", other),
    }
}

#[test]
fn bool_requires_both_or_wildcard() {
    // Only `true` arm — non-exhaustive.
    let arms = vec![(
        ast::Pattern {
            h: ast::Header {
                id: ast::NodeId(0),
                span: dummy_span(),
            },
            kind: ast::PatternKind::Literal(ast::PatternLiteral::Bool(true)),
        },
        dummy_span(),
    )];
    let result = check_match(&bool_ty(), &arms, dummy_span());
    match result {
        Err(CoverageError::NonExhaustive { missing, .. }) => match missing {
            CovPat::Lit(CovLit::Bool(false)) => {}
            other => panic!("expected witness `false`, got {:?}", other),
        },
        other => panic!(
            "expected NonExhaustive for bool with only `true`, got {:?}",
            other
        ),
    }

    // Both arms — exhaustive.
    let arms = vec![
        (
            ast::Pattern {
                h: ast::Header {
                    id: ast::NodeId(0),
                    span: dummy_span(),
                },
                kind: ast::PatternKind::Literal(ast::PatternLiteral::Bool(true)),
            },
            dummy_span(),
        ),
        (
            ast::Pattern {
                h: ast::Header {
                    id: ast::NodeId(1),
                    span: dummy_span(),
                },
                kind: ast::PatternKind::Literal(ast::PatternLiteral::Bool(false)),
            },
            dummy_span(),
        ),
    ];
    assert!(check_match(&bool_ty(), &arms, dummy_span()).is_ok());
}

#[test]
fn int_requires_wildcard_for_exhaustive() {
    // Only `0` and `1` arms — still infinite missing values.
    let arms = vec![
        (
            ast::Pattern {
                h: ast::Header {
                    id: ast::NodeId(0),
                    span: dummy_span(),
                },
                kind: ast::PatternKind::Literal(ast::PatternLiteral::Int(crate::lexer::IntString(
                    "0".to_string(),
                ))),
            },
            dummy_span(),
        ),
        (
            ast::Pattern {
                h: ast::Header {
                    id: ast::NodeId(1),
                    span: dummy_span(),
                },
                kind: ast::PatternKind::Literal(ast::PatternLiteral::Int(crate::lexer::IntString(
                    "1".to_string(),
                ))),
            },
            dummy_span(),
        ),
    ];
    let result = check_match(&i32_ty(), &arms, dummy_span());
    assert!(matches!(result, Err(CoverageError::NonExhaustive { .. })));

    // With a wildcard, exhaustive.
    let arms = vec![
        (
            ast::Pattern {
                h: ast::Header {
                    id: ast::NodeId(0),
                    span: dummy_span(),
                },
                kind: ast::PatternKind::Literal(ast::PatternLiteral::Int(crate::lexer::IntString(
                    "0".to_string(),
                ))),
            },
            dummy_span(),
        ),
        (
            ast::Pattern {
                h: ast::Header {
                    id: ast::NodeId(1),
                    span: dummy_span(),
                },
                kind: ast::PatternKind::Wildcard,
            },
            dummy_span(),
        ),
    ];
    assert!(check_match(&i32_ty(), &arms, dummy_span()).is_ok());
}

#[test]
fn nested_constructor_with_missing_inner_is_non_exhaustive() {
    // Outer sum `#left(i32) | #right(bool)`. Arms:
    //   #left(_) -> ...    -- covers all of #left
    //   #right(true) -> ...
    // `#right(false)` is missing.
    let arms = vec![
        (
            ast::Pattern {
                h: ast::Header {
                    id: ast::NodeId(0),
                    span: dummy_span(),
                },
                kind: ast::PatternKind::Constructor(
                    "left".to_string(),
                    vec![ast::Pattern {
                        h: ast::Header {
                            id: ast::NodeId(1),
                            span: dummy_span(),
                        },
                        kind: ast::PatternKind::Wildcard,
                    }],
                ),
            },
            dummy_span(),
        ),
        (
            ast::Pattern {
                h: ast::Header {
                    id: ast::NodeId(2),
                    span: dummy_span(),
                },
                kind: ast::PatternKind::Constructor(
                    "right".to_string(),
                    vec![ast::Pattern {
                        h: ast::Header {
                            id: ast::NodeId(3),
                            span: dummy_span(),
                        },
                        kind: ast::PatternKind::Literal(ast::PatternLiteral::Bool(true)),
                    }],
                ),
            },
            dummy_span(),
        ),
    ];
    let result = check_match(&sum_ab_payload(), &arms, dummy_span());
    assert!(
        matches!(result, Err(CoverageError::NonExhaustive { .. })),
        "expected non-exhaustive (missing #right(false)), got {:?}",
        result
    );
}

#[test]
fn tuple_pattern_with_all_wildcards_is_exhaustive() {
    let scrut_ty = tuple(vec![i32_ty(), bool_ty()]);
    let arms = vec![(
        ast::Pattern {
            h: ast::Header {
                id: ast::NodeId(0),
                span: dummy_span(),
            },
            kind: ast::PatternKind::Tuple(vec![
                ast::Pattern {
                    h: ast::Header {
                        id: ast::NodeId(1),
                        span: dummy_span(),
                    },
                    kind: ast::PatternKind::Wildcard,
                },
                ast::Pattern {
                    h: ast::Header {
                        id: ast::NodeId(2),
                        span: dummy_span(),
                    },
                    kind: ast::PatternKind::Wildcard,
                },
            ]),
        },
        dummy_span(),
    )];
    assert!(check_match(&scrut_ty, &arms, dummy_span()).is_ok());
}

#[test]
fn unit_match_with_unit_pattern_is_exhaustive() {
    let arms = vec![(
        ast::Pattern {
            h: ast::Header {
                id: ast::NodeId(0),
                span: dummy_span(),
            },
            kind: ast::PatternKind::Unit,
        },
        dummy_span(),
    )];
    assert!(check_match(&unit_ty(), &arms, dummy_span()).is_ok());
}
