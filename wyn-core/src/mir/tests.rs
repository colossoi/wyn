//! Tests for MIR data structures.

use crate::IdArena;
use crate::ast::{NodeCounter, Span, TypeName};
use crate::mir::{Body, Def, Expr, LocalDecl, LocalKind, Program};
use polytype::Type;

fn i32_type() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn f32_type() -> Type<TypeName> {
    Type::Constructed(TypeName::Float(32), vec![])
}

fn test_span() -> Span {
    Span::new(1, 1, 1, 1)
}

#[test]
fn test_simple_function() {
    let mut nc = NodeCounter::new();
    let span = test_span();

    // Build body for: x + y
    let mut body = Body::new();

    // Allocate locals for parameters (params are now LocalIds into body.locals)
    let x_local = body.alloc_local(LocalDecl {
        name: "x".to_string(),
        ty: i32_type(),
        kind: LocalKind::Param,
        span,
    });
    let y_local = body.alloc_local(LocalDecl {
        name: "y".to_string(),
        ty: i32_type(),
        kind: LocalKind::Param,
        span,
    });

    // Build expression: x + y
    let x_ref = body.alloc_expr(Expr::Local(x_local), i32_type(), span, nc.next_id());
    let y_ref = body.alloc_expr(Expr::Local(y_local), i32_type(), span, nc.next_id());
    let add_expr = body.alloc_expr(
        Expr::BinOp {
            op: "+".to_string(),
            lhs: x_ref,
            rhs: y_ref,
        },
        i32_type(),
        span,
        nc.next_id(),
    );
    body.set_root(add_expr);

    // Represents: def add(x, y) = x + y
    // params are now LocalIds pointing to the locals table
    let add_fn = Def::Function {
        id: nc.next_id(),
        name: "add".to_string(),
        params: vec![x_local, y_local],
        ret_type: i32_type(),
        scheme: None,
        attributes: vec![],
        body,
        span,
    };

    let program = Program {
        defs: vec![add_fn],
        lambda_registry: IdArena::new(),
    };

    assert_eq!(program.defs.len(), 1);
    match &program.defs[0] {
        Def::Function {
            name, body, params, ..
        } => {
            assert_eq!(name, "add");
            assert_eq!(body.locals.len(), 2);
            assert_eq!(params.len(), 2);
            // Verify params point to correct locals
            assert_eq!(body.get_local(params[0]).name, "x");
            assert_eq!(body.get_local(params[1]).name, "y");
            // Root should be a BinOp
            match body.get_expr(body.root) {
                Expr::BinOp { op, .. } => assert_eq!(op, "+"),
                other => panic!("Expected BinOp, got {:?}", other),
            }
        }
        _ => panic!("Expected Function"),
    }
}

#[test]
fn test_constant() {
    let mut nc = NodeCounter::new();
    let span = test_span();

    // Build body for: 3.14159
    let mut body = Body::new();
    let float_expr = body.alloc_expr(Expr::Float("3.14159".to_string()), f32_type(), span, nc.next_id());
    body.set_root(float_expr);

    // Represents: def pi = 3.14159
    let pi_const = Def::Constant {
        id: nc.next_id(),
        name: "pi".to_string(),
        ty: f32_type(),
        attributes: vec![],
        body,
        span,
    };

    match &pi_const {
        Def::Constant { name, body, .. } => {
            assert_eq!(name, "pi");
            match body.get_expr(body.root) {
                Expr::Float(s) => assert_eq!(s, "3.14159"),
                other => panic!("Expected Float, got {:?}", other),
            }
        }
        _ => panic!("Expected Constant"),
    }
}

#[test]
fn test_let_binding() {
    let mut nc = NodeCounter::new();
    let span = test_span();

    // Build body for: let x = 42 in x + 1
    let mut body = Body::new();

    // Allocate local for x
    let x_local = body.alloc_local(LocalDecl {
        name: "x".to_string(),
        ty: i32_type(),
        kind: LocalKind::Let,
        span,
    });

    // Build expressions
    let lit_42 = body.alloc_expr(Expr::Int("42".to_string()), i32_type(), span, nc.next_id());
    let x_ref = body.alloc_expr(Expr::Local(x_local), i32_type(), span, nc.next_id());
    let lit_1 = body.alloc_expr(Expr::Int("1".to_string()), i32_type(), span, nc.next_id());
    let add_expr = body.alloc_expr(
        Expr::BinOp {
            op: "+".to_string(),
            lhs: x_ref,
            rhs: lit_1,
        },
        i32_type(),
        span,
        nc.next_id(),
    );
    let let_expr = body.alloc_expr(
        Expr::Let {
            local: x_local,
            rhs: lit_42,
            body: add_expr,
        },
        i32_type(),
        span,
        nc.next_id(),
    );
    body.set_root(let_expr);

    // Verify structure
    match body.get_expr(body.root) {
        Expr::Let {
            local,
            rhs,
            body: let_body,
        } => {
            assert_eq!(body.get_local(*local).name, "x");
            match body.get_expr(*rhs) {
                Expr::Int(s) => assert_eq!(s, "42"),
                other => panic!("Expected Int, got {:?}", other),
            }
            match body.get_expr(*let_body) {
                Expr::BinOp { op, .. } => assert_eq!(op, "+"),
                other => panic!("Expected BinOp, got {:?}", other),
            }
        }
        other => panic!("Expected Let, got {:?}", other),
    }
}

#[test]
fn test_if_expression() {
    let mut nc = NodeCounter::new();
    let span = test_span();

    let bool_type = Type::Constructed(TypeName::Str("bool"), vec![]);

    // Build body for: if true then 1 else 2
    let mut body = Body::new();

    let cond = body.alloc_expr(Expr::Bool(true), bool_type.clone(), span, nc.next_id());
    let then_ = body.alloc_expr(Expr::Int("1".to_string()), i32_type(), span, nc.next_id());
    let else_ = body.alloc_expr(Expr::Int("2".to_string()), i32_type(), span, nc.next_id());
    let if_expr = body.alloc_expr(Expr::If { cond, then_, else_ }, i32_type(), span, nc.next_id());
    body.set_root(if_expr);

    // Verify structure
    match body.get_expr(body.root) {
        Expr::If { cond, then_, else_ } => {
            match body.get_expr(*cond) {
                Expr::Bool(b) => assert!(*b),
                other => panic!("Expected Bool, got {:?}", other),
            }
            match body.get_expr(*then_) {
                Expr::Int(s) => assert_eq!(s, "1"),
                other => panic!("Expected Int, got {:?}", other),
            }
            match body.get_expr(*else_) {
                Expr::Int(s) => assert_eq!(s, "2"),
                other => panic!("Expected Int, got {:?}", other),
            }
        }
        other => panic!("Expected If, got {:?}", other),
    }
}
