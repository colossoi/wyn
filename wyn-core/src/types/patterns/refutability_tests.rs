//! Tests for `is_irrefutable`.

use super::*;
use crate::ast;
use crate::types::{Type, TypeName, sum, tuple};

fn dummy_span() -> ast::Span {
    ast::Span::dummy()
}

fn pat(kind: ast::PatternKind, id: u32) -> ast::Pattern {
    ast::Pattern {
        h: ast::Header {
            id: ast::NodeId(id),
            span: dummy_span(),
        },
        kind,
    }
}

fn i32_ty() -> Type {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn bool_ty() -> Type {
    Type::Constructed(TypeName::Bool, vec![])
}

#[test]
fn name_is_irrefutable() {
    let p = pat(ast::PatternKind::Name("x".to_string()), 0);
    assert!(is_irrefutable(&p, &i32_ty()).is_ok());
}

#[test]
fn wildcard_is_irrefutable() {
    let p = pat(ast::PatternKind::Wildcard, 0);
    assert!(is_irrefutable(&p, &i32_ty()).is_ok());
}

#[test]
fn literal_is_refutable() {
    let p = pat(
        ast::PatternKind::Literal(ast::PatternLiteral::Int(crate::lexer::IntString("0".to_string()))),
        0,
    );
    assert!(is_irrefutable(&p, &i32_ty()).is_err());
}

#[test]
fn tuple_of_names_is_irrefutable() {
    let ty = tuple(vec![i32_ty(), bool_ty()]);
    let p = pat(
        ast::PatternKind::Tuple(vec![
            pat(ast::PatternKind::Name("x".to_string()), 1),
            pat(ast::PatternKind::Name("y".to_string()), 2),
        ]),
        0,
    );
    assert!(is_irrefutable(&p, &ty).is_ok());
}

#[test]
fn tuple_with_inner_literal_is_refutable() {
    let ty = tuple(vec![i32_ty(), bool_ty()]);
    let p = pat(
        ast::PatternKind::Tuple(vec![
            pat(ast::PatternKind::Name("x".to_string()), 1),
            pat(ast::PatternKind::Literal(ast::PatternLiteral::Bool(true)), 2),
        ]),
        0,
    );
    assert!(is_irrefutable(&p, &ty).is_err());
}

#[test]
fn single_variant_constructor_is_irrefutable() {
    let ty = sum(vec![("only".to_string(), vec![i32_ty()])]);
    let p = pat(
        ast::PatternKind::Constructor(
            "only".to_string(),
            vec![pat(ast::PatternKind::Name("x".to_string()), 1)],
        ),
        0,
    );
    assert!(is_irrefutable(&p, &ty).is_ok());
}

#[test]
fn multi_variant_constructor_is_refutable() {
    let ty = sum(vec![("a".to_string(), vec![]), ("b".to_string(), vec![])]);
    let p = pat(ast::PatternKind::Constructor("a".to_string(), vec![]), 0);
    let err = is_irrefutable(&p, &ty).expect_err("expected refutable error");
    assert!(
        err.reason.contains("#b"),
        "reason should mention missing variant #b: {}",
        err.reason
    );
}

#[test]
fn single_variant_with_refutable_payload_is_refutable() {
    // Single variant carrying multi-variant sub-sum: refutable because
    // the sub-pattern fails to cover.
    let inner = sum(vec![("p".to_string(), vec![]), ("q".to_string(), vec![])]);
    let outer = sum(vec![("only".to_string(), vec![inner])]);
    let p = pat(
        ast::PatternKind::Constructor(
            "only".to_string(),
            vec![pat(ast::PatternKind::Constructor("p".to_string(), vec![]), 1)],
        ),
        0,
    );
    assert!(is_irrefutable(&p, &outer).is_err());
}

#[test]
fn unit_against_unit_is_irrefutable() {
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    let p = pat(ast::PatternKind::Unit, 0);
    assert!(is_irrefutable(&p, &unit_ty).is_ok());
}
