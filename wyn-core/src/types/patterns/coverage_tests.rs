//! Tests for the Maranget usefulness algorithm and `check_match`.
//!
//! The matrices and queries are built directly in `CovPat` form so
//! the algorithm can be tested in isolation from the AST → CovPat
//! lowering. `types::checker_tests` covers the checker integration.

use super::*;
use crate::ast::Span;
use crate::types::{sum, tuple, Type, TypeName};

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

// =========================================================================
// Compact helpers + additional tests
// =========================================================================

fn pat(kind: ast::PatternKind) -> ast::Pattern {
    ast::Pattern {
        h: ast::Header {
            id: ast::NodeId(0),
            span: dummy_span(),
        },
        kind,
    }
}

fn name(n: &str) -> ast::Pattern {
    pat(ast::PatternKind::Name(n.to_string()))
}
fn wild() -> ast::Pattern {
    pat(ast::PatternKind::Wildcard)
}
fn ctor(n: &str, sub: Vec<ast::Pattern>) -> ast::Pattern {
    pat(ast::PatternKind::Constructor(n.to_string(), sub))
}
fn lit_int(s: &str) -> ast::Pattern {
    pat(ast::PatternKind::Literal(ast::PatternLiteral::Int(
        crate::lexer::IntString(s.to_string()),
    )))
}
fn lit_bool(b: bool) -> ast::Pattern {
    pat(ast::PatternKind::Literal(ast::PatternLiteral::Bool(b)))
}
fn tup_pat(sub: Vec<ast::Pattern>) -> ast::Pattern {
    pat(ast::PatternKind::Tuple(sub))
}
fn arms(pats: Vec<ast::Pattern>) -> Vec<(ast::Pattern, ast::Span)> {
    pats.into_iter().map(|p| (p, dummy_span())).collect()
}

fn sum_abc() -> Type {
    sum(vec![
        ("a".to_string(), vec![]),
        ("b".to_string(), vec![]),
        ("c".to_string(), vec![]),
    ])
}

fn f32_ty() -> Type {
    Type::Constructed(TypeName::Float(32), vec![])
}

// ----- Three-variant sum coverage -----

#[test]
fn three_variant_sum_all_arms_is_exhaustive() {
    let result = check_match(
        &sum_abc(),
        &arms(vec![ctor("a", vec![]), ctor("b", vec![]), ctor("c", vec![])]),
        dummy_span(),
    );
    assert!(result.is_ok(), "got {:?}", result);
}

#[test]
fn three_variant_sum_missing_c_witness_names_c() {
    let result = check_match(
        &sum_abc(),
        &arms(vec![ctor("a", vec![]), ctor("b", vec![])]),
        dummy_span(),
    );
    match result {
        Err(CoverageError::NonExhaustive {
            missing: CovPat::Ctor(n, _),
            ..
        }) => {
            assert_eq!(n, "c", "witness should name #c");
        }
        other => panic!("expected NonExhaustive(#c), got {:?}", other),
    }
}

#[test]
fn three_variant_sum_only_one_arm_is_non_exhaustive() {
    let result = check_match(&sum_abc(), &arms(vec![ctor("a", vec![])]), dummy_span());
    assert!(matches!(result, Err(CoverageError::NonExhaustive { .. })));
}

#[test]
fn three_variant_sum_with_wildcard_covers_all() {
    let result = check_match(&sum_abc(), &arms(vec![ctor("a", vec![]), wild()]), dummy_span());
    assert!(result.is_ok(), "got {:?}", result);
}

// ----- Nested constructor coverage -----

#[test]
fn nested_ctor_with_inner_wildcard_covers_inner_sum() {
    // outer: #left(i32 | bool) | #right(bool). Inner is a sum we
    // approximate with #right(bool) on the outer here. We test:
    //   case #left(_)  -> ...   -- covers all of #left's payload
    //   case #right(_) -> ...   -- covers all of #right's payload
    let result = check_match(
        &sum_ab_payload(),
        &arms(vec![ctor("left", vec![wild()]), ctor("right", vec![wild()])]),
        dummy_span(),
    );
    assert!(result.is_ok(), "got {:?}", result);
}

#[test]
fn nested_ctor_with_inner_literal_is_non_exhaustive_for_other_lit() {
    // #left(0) covers only #left(0); #left(other-int) is missing.
    let result = check_match(
        &sum_ab_payload(),
        &arms(vec![
            ctor("left", vec![lit_int("0")]),
            ctor("right", vec![wild()]),
        ]),
        dummy_span(),
    );
    assert!(matches!(result, Err(CoverageError::NonExhaustive { .. })));
}

#[test]
fn nested_ctor_with_name_binds_covers_payload() {
    // #left(x) treats x as a name binding (effective wildcard for
    // coverage). Covers all of #left.
    let result = check_match(
        &sum_ab_payload(),
        &arms(vec![ctor("left", vec![name("x")]), ctor("right", vec![wild()])]),
        dummy_span(),
    );
    assert!(result.is_ok(), "got {:?}", result);
}

// ----- Tuple coverage -----

#[test]
fn tuple_bool_bool_cartesian_is_exhaustive() {
    let scrut = tuple(vec![bool_ty(), bool_ty()]);
    let cases = arms(vec![
        tup_pat(vec![lit_bool(true), lit_bool(true)]),
        tup_pat(vec![lit_bool(true), lit_bool(false)]),
        tup_pat(vec![lit_bool(false), lit_bool(true)]),
        tup_pat(vec![lit_bool(false), lit_bool(false)]),
    ]);
    let result = check_match(&scrut, &cases, dummy_span());
    assert!(result.is_ok(), "got {:?}", result);
}

#[test]
fn tuple_bool_bool_missing_corner_is_non_exhaustive() {
    let scrut = tuple(vec![bool_ty(), bool_ty()]);
    let cases = arms(vec![
        tup_pat(vec![lit_bool(true), lit_bool(true)]),
        tup_pat(vec![lit_bool(true), lit_bool(false)]),
        tup_pat(vec![lit_bool(false), lit_bool(true)]),
    ]);
    let result = check_match(&scrut, &cases, dummy_span());
    assert!(matches!(result, Err(CoverageError::NonExhaustive { .. })));
}

#[test]
fn tuple_with_wildcard_second_covers_everything_for_first() {
    let scrut = tuple(vec![bool_ty(), i32_ty()]);
    let cases = arms(vec![
        tup_pat(vec![lit_bool(true), wild()]),
        tup_pat(vec![lit_bool(false), wild()]),
    ]);
    let result = check_match(&scrut, &cases, dummy_span());
    assert!(result.is_ok(), "got {:?}", result);
}

#[test]
fn tuple_with_wildcard_first_covers_everything() {
    let scrut = tuple(vec![bool_ty(), i32_ty()]);
    let cases = arms(vec![tup_pat(vec![wild(), wild()])]);
    assert!(check_match(&scrut, &cases, dummy_span()).is_ok());
}

// ----- Redundancy -----

#[test]
fn exact_duplicate_arm_is_redundant() {
    let cases = arms(vec![ctor("a", vec![]), ctor("a", vec![]), ctor("b", vec![])]);
    let result = check_match(&sum_ab(), &cases, dummy_span());
    match result {
        Err(CoverageError::Redundant { arm_index, .. }) => {
            assert_eq!(arm_index, 1, "second #a should be redundant");
        }
        other => panic!("expected Redundant, got {:?}", other),
    }
}

#[test]
fn strictly_narrower_arm_after_wildcard_payload_is_redundant() {
    let cases = arms(vec![
        ctor("left", vec![wild()]),
        ctor("left", vec![lit_int("0")]),
        ctor("right", vec![wild()]),
    ]);
    let result = check_match(&sum_ab_payload(), &cases, dummy_span());
    assert!(matches!(
        result,
        Err(CoverageError::Redundant { arm_index: 1, .. })
    ));
}

#[test]
fn wildcard_first_makes_all_later_arms_redundant() {
    let cases = arms(vec![wild(), ctor("a", vec![])]);
    let result = check_match(&sum_ab(), &cases, dummy_span());
    assert!(matches!(
        result,
        Err(CoverageError::Redundant { arm_index: 1, .. })
    ));
}

#[test]
fn redundant_literal_after_same_literal() {
    let cases = arms(vec![lit_int("0"), lit_int("0"), wild()]);
    let result = check_match(&i32_ty(), &cases, dummy_span());
    assert!(matches!(
        result,
        Err(CoverageError::Redundant { arm_index: 1, .. })
    ));
}

// ----- Float infinite universe -----

#[test]
fn float_only_literals_is_non_exhaustive() {
    let cases = arms(vec![pat(ast::PatternKind::Literal(ast::PatternLiteral::Float(
        0.0,
    )))]);
    let result = check_match(&f32_ty(), &cases, dummy_span());
    assert!(matches!(result, Err(CoverageError::NonExhaustive { .. })));
}

#[test]
fn float_literal_then_wildcard_is_exhaustive() {
    let cases = arms(vec![
        pat(ast::PatternKind::Literal(ast::PatternLiteral::Float(0.0))),
        wild(),
    ]);
    let result = check_match(&f32_ty(), &cases, dummy_span());
    assert!(result.is_ok(), "got {:?}", result);
}

// ----- lower() function -----

#[test]
fn lower_strips_typed_wrapper() {
    use crate::ast::Type as AstType;
    let inner = name("x");
    let typed = pat(ast::PatternKind::Typed(
        Box::new(inner),
        AstType::Constructed(crate::ast::TypeName::Bool, vec![]),
    ));
    let cp = lower(&typed);
    assert!(matches!(cp, CovPat::Wild));
}

#[test]
fn lower_collapses_name_to_wild() {
    assert!(matches!(lower(&name("x")), CovPat::Wild));
}

#[test]
fn lower_constructor_keeps_name_and_sub_arity() {
    let cp = lower(&ctor("foo", vec![wild(), name("y")]));
    match cp {
        CovPat::Ctor(n, sub) => {
            assert_eq!(n, "foo");
            assert_eq!(sub.len(), 2);
            assert!(matches!(sub[0], CovPat::Wild));
            assert!(matches!(sub[1], CovPat::Wild));
        }
        other => panic!("expected Ctor, got {:?}", other),
    }
}

#[test]
fn lower_literal_int_carries_text() {
    let cp = lower(&lit_int("42"));
    match cp {
        CovPat::Lit(CovLit::Int(s)) => assert_eq!(s, "42"),
        other => panic!("expected Int lit, got {:?}", other),
    }
}

#[test]
fn lower_literal_bool() {
    let cp = lower(&lit_bool(true));
    assert!(matches!(cp, CovPat::Lit(CovLit::Bool(true))));
}

#[test]
fn lower_unit_pattern() {
    assert!(matches!(lower(&pat(ast::PatternKind::Unit)), CovPat::UnitP));
}

// ----- Witness shape sanity -----

#[test]
fn missing_payload_witness_is_wild() {
    // #left missing entirely; witness is #left(_).
    let cases = arms(vec![ctor("right", vec![wild()])]);
    let result = check_match(&sum_ab_payload(), &cases, dummy_span());
    match result {
        Err(CoverageError::NonExhaustive {
            missing: CovPat::Ctor(n, sub),
            ..
        }) => {
            assert_eq!(n, "left");
            assert_eq!(sub.len(), 1, "missing #left witness has one payload slot");
            assert!(
                matches!(sub[0], CovPat::Wild),
                "payload slot witness should be wild"
            );
        }
        other => panic!("expected NonExhaustive #left(_), got {:?}", other),
    }
}

// ----- Integer-literal canonicalization -----

#[test]
fn int_literal_with_leading_zero_canonicalizes() {
    // Two arms `case 01 -> ...` and `case 1 -> ...` denote the same
    // numeric value; the second must be flagged redundant.
    let cases = arms(vec![lit_int("01"), lit_int("1"), wild()]);
    let result = check_match(&i32_ty(), &cases, dummy_span());
    assert!(
        matches!(result, Err(CoverageError::Redundant { arm_index: 1, .. })),
        "leading-zero literal should compare equal to plain literal, got {:?}",
        result
    );
}

#[test]
fn int_literal_negative_zero_canonicalizes() {
    // `-0` and `0` denote the same numeric value.
    let cases = arms(vec![lit_int("-0"), lit_int("0"), wild()]);
    let result = check_match(&i32_ty(), &cases, dummy_span());
    assert!(
        matches!(result, Err(CoverageError::Redundant { arm_index: 1, .. })),
        "negative-zero literal should compare equal to zero, got {:?}",
        result
    );
}
