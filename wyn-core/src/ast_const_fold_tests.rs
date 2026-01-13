use crate::ast::{BinaryOp, ExprKind, Expression, Header, NodeId, Span};
use crate::ast_const_fold::AstConstFolder;

fn test_header() -> Header {
    Header {
        id: NodeId(0),
        span: Span::new(0, 0, 0, 0),
    }
}

fn make_int(n: i32) -> Expression {
    Expression {
        h: test_header(),
        kind: ExprKind::IntLiteral(n.to_string().into()),
    }
}

fn make_ident(name: &str) -> Expression {
    Expression {
        h: test_header(),
        kind: ExprKind::Identifier(vec![], name.to_string()),
    }
}

fn make_binop(op: &str, lhs: Expression, rhs: Expression) -> Expression {
    Expression {
        h: test_header(),
        kind: ExprKind::BinaryOp(BinaryOp { op: op.to_string() }, Box::new(lhs), Box::new(rhs)),
    }
}

fn make_float(n: f32) -> Expression {
    Expression {
        h: test_header(),
        kind: ExprKind::FloatLiteral(n),
    }
}

#[test]
fn test_fold_simple_addition() {
    let mut folder = AstConstFolder::new();
    let mut expr = make_binop("+", make_int(2), make_int(7));
    folder.fold_expr(&mut expr);
    assert_eq!(expr.kind, ExprKind::IntLiteral("9".into()));
}

#[test]
fn test_fold_nested_arithmetic() {
    let mut folder = AstConstFolder::new();
    // (2 + 3) * 4 = 20
    let inner = make_binop("+", make_int(2), make_int(3));
    let mut expr = make_binop("*", inner, make_int(4));
    folder.fold_expr(&mut expr);
    assert_eq!(expr.kind, ExprKind::IntLiteral("20".into()));
}

#[test]
fn test_inline_constant() {
    let mut folder = AstConstFolder::new();
    folder.add_constant("SIZE", 16);

    let mut expr = make_ident("SIZE");
    folder.fold_expr(&mut expr);
    assert_eq!(expr.kind, ExprKind::IntLiteral("16".into()));
}

#[test]
fn test_fold_with_constant_reference() {
    let mut folder = AstConstFolder::new();
    folder.add_constant("X", 5);

    // X + 3 = 8
    let mut expr = make_binop("+", make_ident("X"), make_int(3));
    folder.fold_expr(&mut expr);
    assert_eq!(expr.kind, ExprKind::IntLiteral("8".into()));
}

#[test]
fn test_no_fold_non_constant() {
    let mut folder = AstConstFolder::new();
    // y + 3 where y is not a constant - should not fold
    let mut expr = make_binop("+", make_ident("y"), make_int(3));
    folder.fold_expr(&mut expr);
    // Should still be a binop (just with folded children)
    assert!(matches!(expr.kind, ExprKind::BinaryOp(..)));
}

#[test]
fn test_division_by_zero_not_folded() {
    let mut folder = AstConstFolder::new();
    let mut expr = make_binop("/", make_int(10), make_int(0));
    folder.fold_expr(&mut expr);
    // Should not fold division by zero
    assert!(matches!(expr.kind, ExprKind::BinaryOp(..)));
}

// =============================================================================
// Slice Constant Folding Tests
// =============================================================================

fn make_array_ident(name: &str) -> Expression {
    Expression {
        h: test_header(),
        kind: ExprKind::Identifier(vec![], name.to_string()),
    }
}

fn make_slice(array: Expression, start: Option<Expression>, end: Option<Expression>) -> Expression {
    Expression {
        h: test_header(),
        kind: ExprKind::Slice(crate::ast::SliceExpr {
            array: Box::new(array),
            start: start.map(Box::new),
            end: end.map(Box::new),
        }),
    }
}

#[test]
fn test_fold_slice_start_constant() {
    let mut folder = AstConstFolder::new();
    // arr[1+2:10] should fold to arr[3:10]
    let start = make_binop("+", make_int(1), make_int(2));
    let mut expr = make_slice(make_array_ident("arr"), Some(start), Some(make_int(10)));
    folder.fold_expr(&mut expr);

    if let ExprKind::Slice(slice) = &expr.kind {
        let start = slice.start.as_ref().expect("start should exist");
        assert_eq!(
            start.kind,
            ExprKind::IntLiteral("3".into()),
            "Slice start should be folded to 3"
        );
    } else {
        panic!("Expected Slice, got {:?}", expr.kind);
    }
}

#[test]
fn test_fold_slice_end_constant() {
    let mut folder = AstConstFolder::new();
    // arr[0:5+5] should fold to arr[0:10]
    let end = make_binop("+", make_int(5), make_int(5));
    let mut expr = make_slice(make_array_ident("arr"), Some(make_int(0)), Some(end));
    folder.fold_expr(&mut expr);

    if let ExprKind::Slice(slice) = &expr.kind {
        let end = slice.end.as_ref().expect("end should exist");
        assert_eq!(
            end.kind,
            ExprKind::IntLiteral("10".into()),
            "Slice end should be folded to 10"
        );
    } else {
        panic!("Expected Slice, got {:?}", expr.kind);
    }
}

#[test]
fn test_fold_slice_both_components() {
    let mut folder = AstConstFolder::new();
    // arr[1+1:4*2] should fold to arr[2:8]
    let start = make_binop("+", make_int(1), make_int(1));
    let end = make_binop("*", make_int(4), make_int(2));
    let mut expr = make_slice(make_array_ident("arr"), Some(start), Some(end));
    folder.fold_expr(&mut expr);

    if let ExprKind::Slice(slice) = &expr.kind {
        let start = slice.start.as_ref().expect("start should exist");
        let end = slice.end.as_ref().expect("end should exist");
        assert_eq!(start.kind, ExprKind::IntLiteral("2".into()), "start should fold to 2");
        assert_eq!(end.kind, ExprKind::IntLiteral("8".into()), "end should fold to 8");
    } else {
        panic!("Expected Slice, got {:?}", expr.kind);
    }
}

#[test]
fn test_fold_slice_with_constant_inlining() {
    let mut folder = AstConstFolder::new();
    folder.add_constant("SIZE", 10);
    folder.add_constant("OFFSET", 2);

    // arr[OFFSET:SIZE] should inline to arr[2:10]
    let mut expr = make_slice(
        make_array_ident("arr"),
        Some(make_ident("OFFSET")),
        Some(make_ident("SIZE")),
    );
    folder.fold_expr(&mut expr);

    if let ExprKind::Slice(slice) = &expr.kind {
        let start = slice.start.as_ref().expect("start should exist");
        let end = slice.end.as_ref().expect("end should exist");
        assert_eq!(start.kind, ExprKind::IntLiteral("2".into()), "start should inline to 2");
        assert_eq!(end.kind, ExprKind::IntLiteral("10".into()), "end should inline to 10");
    } else {
        panic!("Expected Slice, got {:?}", expr.kind);
    }
}

#[test]
fn test_fold_slice_no_fold_with_variable() {
    let mut folder = AstConstFolder::new();
    // arr[n:10] where n is not a constant - start should not fold
    let mut expr = make_slice(make_array_ident("arr"), Some(make_ident("n")), Some(make_int(10)));
    folder.fold_expr(&mut expr);

    if let ExprKind::Slice(slice) = &expr.kind {
        let start = slice.start.as_ref().expect("start should exist");
        // Start should still be an identifier since n is not known
        assert!(
            matches!(start.kind, ExprKind::Identifier(_, _)),
            "Start should remain identifier when not a constant"
        );
    } else {
        panic!("Expected Slice, got {:?}", expr.kind);
    }
}

// =============================================================================
// Zero Subtraction to Negation Optimization Tests
// =============================================================================

#[test]
fn test_zero_minus_float_becomes_negation() {
    let mut folder = AstConstFolder::new();
    // 0.0 - x should become -x
    let mut expr = make_binop("-", make_float(0.0), make_ident("x"));
    folder.fold_expr(&mut expr);

    if let ExprKind::UnaryOp(op, operand) = &expr.kind {
        assert_eq!(op.op, "-", "Should be negation operator");
        assert!(
            matches!(operand.kind, ExprKind::Identifier(_, ref name) if name == "x"),
            "Operand should be x"
        );
    } else {
        panic!("Expected UnaryOp, got {:?}", expr.kind);
    }
}

#[test]
fn test_zero_minus_int_becomes_negation() {
    let mut folder = AstConstFolder::new();
    // 0 - y should become -y
    let mut expr = make_binop("-", make_int(0), make_ident("y"));
    folder.fold_expr(&mut expr);

    if let ExprKind::UnaryOp(op, operand) = &expr.kind {
        assert_eq!(op.op, "-", "Should be negation operator");
        assert!(
            matches!(operand.kind, ExprKind::Identifier(_, ref name) if name == "y"),
            "Operand should be y"
        );
    } else {
        panic!("Expected UnaryOp, got {:?}", expr.kind);
    }
}

#[test]
fn test_nonzero_minus_does_not_become_negation() {
    let mut folder = AstConstFolder::new();
    // 1.0 - x should NOT become negation
    let mut expr = make_binop("-", make_float(1.0), make_ident("x"));
    folder.fold_expr(&mut expr);

    assert!(
        matches!(expr.kind, ExprKind::BinaryOp(..)),
        "Non-zero subtraction should remain BinaryOp"
    );
}
