//! Tests for MIR transformation operations.

use crate::ast::{NodeId, Span, TypeName};
use crate::mir::transform::sort_stmts_by_deps;
use crate::mir::{Body, Expr, LocalDecl, LocalKind, Stmt};

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

fn i32_ty() -> polytype::Type<TypeName> {
    polytype::Type::Constructed(TypeName::Int(32), vec![])
}

/// Helper to create a simple body with locals and expressions for testing.
fn create_test_body() -> Body {
    Body::new()
}

#[test]
fn test_sort_stmts_no_deps() {
    // Two independent stmts: a = 1, b = 2
    // Should preserve original order (or any order is fine)
    let mut body = create_test_body();

    let local_a = body.alloc_local(LocalDecl {
        name: "a".to_string(),
        ty: i32_ty(),
        span: dummy_span(),
        kind: LocalKind::Let,
    });
    let local_b = body.alloc_local(LocalDecl {
        name: "b".to_string(),
        ty: i32_ty(),
        span: dummy_span(),
        kind: LocalKind::Let,
    });

    let one = body.alloc_expr(Expr::Int("1".to_string()), i32_ty(), dummy_span(), dummy_node());
    let two = body.alloc_expr(Expr::Int("2".to_string()), i32_ty(), dummy_span(), dummy_node());

    let stmts = vec![
        Stmt {
            local: local_a,
            rhs: one,
        },
        Stmt {
            local: local_b,
            rhs: two,
        },
    ];

    let sorted = sort_stmts_by_deps(&body, &stmts);

    assert_eq!(sorted.len(), 2);
    // Both orders are valid since there are no deps
}

#[test]
fn test_sort_stmts_simple_chain() {
    // Chain: a = 1, b = a + 2
    // b depends on a, so a must come first
    let mut body = create_test_body();

    let local_a = body.alloc_local(LocalDecl {
        name: "a".to_string(),
        ty: i32_ty(),
        span: dummy_span(),
        kind: LocalKind::Let,
    });
    let local_b = body.alloc_local(LocalDecl {
        name: "b".to_string(),
        ty: i32_ty(),
        span: dummy_span(),
        kind: LocalKind::Let,
    });

    let one = body.alloc_expr(Expr::Int("1".to_string()), i32_ty(), dummy_span(), dummy_node());
    let two = body.alloc_expr(Expr::Int("2".to_string()), i32_ty(), dummy_span(), dummy_node());
    let local_a_ref = body.alloc_expr(Expr::Local(local_a), i32_ty(), dummy_span(), dummy_node());
    let add = body.alloc_expr(
        Expr::BinOp {
            op: "+".to_string(),
            lhs: local_a_ref,
            rhs: two,
        },
        i32_ty(),
        dummy_span(),
        dummy_node(),
    );

    // Add stmts in WRONG order: b = a + 2, then a = 1
    let stmts = vec![
        Stmt {
            local: local_b,
            rhs: add,
        },
        Stmt {
            local: local_a,
            rhs: one,
        },
    ];

    let sorted = sort_stmts_by_deps(&body, &stmts);

    assert_eq!(sorted.len(), 2);
    // a = 1 must come before b = a + 2
    assert_eq!(sorted[0].local, local_a, "a should be first");
    assert_eq!(sorted[1].local, local_b, "b should be second");
}

#[test]
fn test_sort_stmts_three_chain() {
    // Chain: a = 1, b = a + 2, c = b + 3
    // c depends on b, b depends on a
    let mut body = create_test_body();

    let local_a = body.alloc_local(LocalDecl {
        name: "a".to_string(),
        ty: i32_ty(),
        span: dummy_span(),
        kind: LocalKind::Let,
    });
    let local_b = body.alloc_local(LocalDecl {
        name: "b".to_string(),
        ty: i32_ty(),
        span: dummy_span(),
        kind: LocalKind::Let,
    });
    let local_c = body.alloc_local(LocalDecl {
        name: "c".to_string(),
        ty: i32_ty(),
        span: dummy_span(),
        kind: LocalKind::Let,
    });

    let one = body.alloc_expr(Expr::Int("1".to_string()), i32_ty(), dummy_span(), dummy_node());
    let two = body.alloc_expr(Expr::Int("2".to_string()), i32_ty(), dummy_span(), dummy_node());
    let three = body.alloc_expr(Expr::Int("3".to_string()), i32_ty(), dummy_span(), dummy_node());

    let local_a_ref = body.alloc_expr(Expr::Local(local_a), i32_ty(), dummy_span(), dummy_node());
    let add_ab = body.alloc_expr(
        Expr::BinOp {
            op: "+".to_string(),
            lhs: local_a_ref,
            rhs: two,
        },
        i32_ty(),
        dummy_span(),
        dummy_node(),
    );

    let local_b_ref = body.alloc_expr(Expr::Local(local_b), i32_ty(), dummy_span(), dummy_node());
    let add_bc = body.alloc_expr(
        Expr::BinOp {
            op: "+".to_string(),
            lhs: local_b_ref,
            rhs: three,
        },
        i32_ty(),
        dummy_span(),
        dummy_node(),
    );

    // Add stmts in reverse order: c, b, a
    let stmts = vec![
        Stmt {
            local: local_c,
            rhs: add_bc,
        },
        Stmt {
            local: local_b,
            rhs: add_ab,
        },
        Stmt {
            local: local_a,
            rhs: one,
        },
    ];

    let sorted = sort_stmts_by_deps(&body, &stmts);

    assert_eq!(sorted.len(), 3);
    assert_eq!(sorted[0].local, local_a, "a should be first");
    assert_eq!(sorted[1].local, local_b, "b should be second");
    assert_eq!(sorted[2].local, local_c, "c should be third");
}

#[test]
fn test_sort_stmts_diamond() {
    // Diamond: a = 1, b = a + 2, c = a + 3, d = b + c
    // d depends on b and c, both depend on a
    let mut body = create_test_body();

    let local_a = body.alloc_local(LocalDecl {
        name: "a".to_string(),
        ty: i32_ty(),
        span: dummy_span(),
        kind: LocalKind::Let,
    });
    let local_b = body.alloc_local(LocalDecl {
        name: "b".to_string(),
        ty: i32_ty(),
        span: dummy_span(),
        kind: LocalKind::Let,
    });
    let local_c = body.alloc_local(LocalDecl {
        name: "c".to_string(),
        ty: i32_ty(),
        span: dummy_span(),
        kind: LocalKind::Let,
    });
    let local_d = body.alloc_local(LocalDecl {
        name: "d".to_string(),
        ty: i32_ty(),
        span: dummy_span(),
        kind: LocalKind::Let,
    });

    let one = body.alloc_expr(Expr::Int("1".to_string()), i32_ty(), dummy_span(), dummy_node());
    let two = body.alloc_expr(Expr::Int("2".to_string()), i32_ty(), dummy_span(), dummy_node());
    let three = body.alloc_expr(Expr::Int("3".to_string()), i32_ty(), dummy_span(), dummy_node());

    let local_a_ref1 = body.alloc_expr(Expr::Local(local_a), i32_ty(), dummy_span(), dummy_node());
    let local_a_ref2 = body.alloc_expr(Expr::Local(local_a), i32_ty(), dummy_span(), dummy_node());
    let local_b_ref = body.alloc_expr(Expr::Local(local_b), i32_ty(), dummy_span(), dummy_node());
    let local_c_ref = body.alloc_expr(Expr::Local(local_c), i32_ty(), dummy_span(), dummy_node());

    let add_ab = body.alloc_expr(
        Expr::BinOp {
            op: "+".to_string(),
            lhs: local_a_ref1,
            rhs: two,
        },
        i32_ty(),
        dummy_span(),
        dummy_node(),
    );
    let add_ac = body.alloc_expr(
        Expr::BinOp {
            op: "+".to_string(),
            lhs: local_a_ref2,
            rhs: three,
        },
        i32_ty(),
        dummy_span(),
        dummy_node(),
    );
    let add_bc = body.alloc_expr(
        Expr::BinOp {
            op: "+".to_string(),
            lhs: local_b_ref,
            rhs: local_c_ref,
        },
        i32_ty(),
        dummy_span(),
        dummy_node(),
    );

    // Add stmts in worst order: d, c, b, a
    let stmts = vec![
        Stmt {
            local: local_d,
            rhs: add_bc,
        },
        Stmt {
            local: local_c,
            rhs: add_ac,
        },
        Stmt {
            local: local_b,
            rhs: add_ab,
        },
        Stmt {
            local: local_a,
            rhs: one,
        },
    ];

    let sorted = sort_stmts_by_deps(&body, &stmts);

    assert_eq!(sorted.len(), 4);

    // Find positions
    let pos_a = sorted.iter().position(|s| s.local == local_a).unwrap();
    let pos_b = sorted.iter().position(|s| s.local == local_b).unwrap();
    let pos_c = sorted.iter().position(|s| s.local == local_c).unwrap();
    let pos_d = sorted.iter().position(|s| s.local == local_d).unwrap();

    // a must come before b and c
    assert!(pos_a < pos_b, "a must come before b");
    assert!(pos_a < pos_c, "a must come before c");
    // b and c must come before d
    assert!(pos_b < pos_d, "b must come before d");
    assert!(pos_c < pos_d, "c must come before d");
}

#[test]
fn test_sort_stmts_external_dep() {
    // Stmt depends on a local that's NOT in the stmt list (e.g., a parameter)
    // This should not cause issues
    let mut body = create_test_body();

    // local_p is a "parameter" - not defined by any stmt
    let local_p = body.alloc_local(LocalDecl {
        name: "p".to_string(),
        ty: i32_ty(),
        span: dummy_span(),
        kind: LocalKind::Param,
    });
    let local_a = body.alloc_local(LocalDecl {
        name: "a".to_string(),
        ty: i32_ty(),
        span: dummy_span(),
        kind: LocalKind::Let,
    });

    let one = body.alloc_expr(Expr::Int("1".to_string()), i32_ty(), dummy_span(), dummy_node());
    let local_p_ref = body.alloc_expr(Expr::Local(local_p), i32_ty(), dummy_span(), dummy_node());
    let add = body.alloc_expr(
        Expr::BinOp {
            op: "+".to_string(),
            lhs: local_p_ref,
            rhs: one,
        },
        i32_ty(),
        dummy_span(),
        dummy_node(),
    );

    // a = p + 1 (depends on parameter p, not on another stmt)
    let stmts = vec![Stmt {
        local: local_a,
        rhs: add,
    }];

    let sorted = sort_stmts_by_deps(&body, &stmts);

    assert_eq!(sorted.len(), 1);
    assert_eq!(sorted[0].local, local_a);
}

#[test]
fn test_sort_stmts_empty() {
    let body = create_test_body();
    let stmts: Vec<Stmt> = vec![];

    let sorted = sort_stmts_by_deps(&body, &stmts);

    assert_eq!(sorted.len(), 0);
}
