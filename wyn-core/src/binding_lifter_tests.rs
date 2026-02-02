//! Tests for the binding lifting (expression remapping) pass.

use crate::IdArena;
use crate::ast::{NodeId, Span, TypeName};
use crate::binding_lifter::lift_bindings;
use crate::mir::{Block, Body, Def, Expr, ExprId, LocalDecl, LocalId, LocalKind, LoopKind, Program};
use polytype::Type;
use std::sync::atomic::{AtomicU32, Ordering};

// =============================================================================
// Test Helpers - Type Constructors
// =============================================================================

fn i32_type() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn test_span() -> Span {
    Span::new(1, 1, 1, 1)
}

// Global counter for test node IDs (tests don't care about the actual values)
static TEST_NODE_ID: AtomicU32 = AtomicU32::new(0);

fn next_id() -> NodeId {
    NodeId(TEST_NODE_ID.fetch_add(1, Ordering::Relaxed))
}

// =============================================================================
// Test Helpers - Expression Building in Body
// =============================================================================

/// Builder for creating test expressions in a Body.
struct TestBodyBuilder {
    body: Body,
}

impl TestBodyBuilder {
    fn new() -> Self {
        TestBodyBuilder { body: Body::new() }
    }

    /// Allocate a local variable.
    fn alloc_local(&mut self, name: &str, ty: Type<TypeName>, kind: LocalKind) -> LocalId {
        self.body.alloc_local(LocalDecl {
            name: name.to_string(),
            span: test_span(),
            ty,
            kind,
        })
    }

    /// Create a local reference expression.
    fn local(&mut self, local_id: LocalId) -> ExprId {
        let ty = self.body.get_local(local_id).ty.clone();
        self.body.alloc_expr(Expr::Local(local_id), ty, test_span(), next_id())
    }

    /// Create an integer literal expression.
    fn int_lit(&mut self, n: i32) -> ExprId {
        self.body.alloc_expr(Expr::Int(n.to_string()), i32_type(), test_span(), next_id())
    }

    /// Create a binary operation expression.
    fn binop(&mut self, op: &str, lhs: ExprId, rhs: ExprId) -> ExprId {
        let ty = self.body.get_type(lhs).clone();
        self.body.alloc_expr(
            Expr::BinOp {
                op: op.to_string(),
                lhs,
                rhs,
            },
            ty,
            test_span(),
            next_id(),
        )
    }

    /// Add a statement binding.
    fn stmt(&mut self, local: LocalId, rhs: ExprId) {
        self.body.push_stmt(local, rhs);
    }

    /// Create a for-range loop expression.
    fn for_range_loop(
        &mut self,
        loop_var: LocalId,
        init: ExprId,
        iter_var: LocalId,
        bound: ExprId,
        body: ExprId,
    ) -> ExprId {
        let ty = self.body.get_type(body).clone();
        // Create a reference to loop_var for init_bindings
        let loop_var_ref = self.local(loop_var);
        self.body.alloc_expr(
            Expr::Loop {
                loop_var,
                init,
                init_bindings: vec![(loop_var, loop_var_ref)],
                kind: LoopKind::ForRange { var: iter_var, bound },
                body: Block::new(body),
            },
            ty,
            test_span(),
            next_id(),
        )
    }

    /// Finalize the body with the given root expression.
    fn build(mut self, root: ExprId) -> Body {
        self.body.set_root(root);
        self.body
    }
}

/// Create a simple test program with one function.
fn make_test_program(body: Body) -> Program {
    Program {
        defs: vec![Def::Function {
            id: next_id(),
            name: "test_fn".to_string(),
            params: vec![],
            ret_type: i32_type(),
            attributes: vec![],
            body,
            span: test_span(),
            dps_output: None,
        }],
        lambda_registry: IdArena::new(),
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[test]
fn test_simple_expression_remapping() {
    // Test that lift_bindings correctly remaps a simple expression
    let mut b = TestBodyBuilder::new();

    let x = b.alloc_local("x", i32_type(), LocalKind::Let);
    let lit_42 = b.int_lit(42);
    let x_ref = b.local(x);
    let lit_1 = b.int_lit(1);
    let add_expr = b.binop("+", x_ref, lit_1);

    // x = 42; root = x + 1
    b.stmt(x, lit_42);
    let body = b.build(add_expr);
    let program = make_test_program(body);
    let result = lift_bindings(program);

    // Verify the structure is preserved
    match &result.defs[0] {
        Def::Function { body, .. } => {
            // Should have 1 local
            assert_eq!(body.locals.len(), 1);
            assert_eq!(body.get_local(LocalId(0)).name, "x");

            // Should have 1 statement
            let stmts: Vec<_> = body.iter_stmts().collect();
            assert_eq!(stmts.len(), 1);

            // Root should be a BinOp
            match body.get_expr(body.root) {
                Expr::BinOp { op, .. } => assert_eq!(op, "+"),
                other => panic!("Expected BinOp, got {:?}", other),
            }
        }
        _ => panic!("Expected Function def"),
    }
}

#[test]
fn test_loop_with_statements() {
    // Test that statements and loops are correctly handled
    let mut b = TestBodyBuilder::new();

    let acc = b.alloc_local("acc", i32_type(), LocalKind::LoopVar);
    let i = b.alloc_local("i", i32_type(), LocalKind::LoopVar);
    let x = b.alloc_local("x", i32_type(), LocalKind::Let);

    // x = 42 (statement before loop)
    let lit_42 = b.int_lit(42);
    b.stmt(x, lit_42);

    // Loop body: acc + x
    let acc_ref = b.local(acc);
    let x_ref = b.local(x);
    let add_expr = b.binop("+", acc_ref, x_ref);

    // loop acc = 0 for i < 10 do acc + x
    let init = b.int_lit(0);
    let bound = b.int_lit(10);
    let loop_expr = b.for_range_loop(acc, init, i, bound, add_expr);

    let body = b.build(loop_expr);
    let program = make_test_program(body);
    let result = lift_bindings(program);

    match &result.defs[0] {
        Def::Function { body, .. } => {
            // Should have 3 locals (acc, i, x)
            assert_eq!(body.locals.len(), 3);

            // Should have 1 statement (x = 42)
            let stmts: Vec<_> = body.iter_stmts().collect();
            assert_eq!(stmts.len(), 1);

            // Root should be a Loop
            match body.get_expr(body.root) {
                Expr::Loop { loop_var, .. } => {
                    assert_eq!(body.get_local(*loop_var).name, "acc");
                }
                other => panic!("Expected Loop, got {:?}", other),
            }
        }
        _ => panic!("Expected Function def"),
    }
}

#[test]
fn test_chain_of_statements() {
    // Test multiple chained statements
    let mut b = TestBodyBuilder::new();

    let x = b.alloc_local("x", i32_type(), LocalKind::Let);
    let y = b.alloc_local("y", i32_type(), LocalKind::Let);

    // x = 1
    let lit_1 = b.int_lit(1);
    b.stmt(x, lit_1);

    // y = x + 2
    let x_ref = b.local(x);
    let lit_2 = b.int_lit(2);
    let x_plus_2 = b.binop("+", x_ref, lit_2);
    b.stmt(y, x_plus_2);

    // root = y
    let y_ref = b.local(y);
    let body = b.build(y_ref);
    let program = make_test_program(body);
    let result = lift_bindings(program);

    match &result.defs[0] {
        Def::Function { body, .. } => {
            // Should have 2 locals
            assert_eq!(body.locals.len(), 2);

            // Should have 2 statements in order
            let stmts: Vec<_> = body.iter_stmts().collect();
            assert_eq!(stmts.len(), 2);
            assert_eq!(body.get_local(stmts[0].local).name, "x");
            assert_eq!(body.get_local(stmts[1].local).name, "y");

            // Root should be Local(y)
            match body.get_expr(body.root) {
                Expr::Local(local_id) => {
                    assert_eq!(body.get_local(*local_id).name, "y");
                }
                other => panic!("Expected Local, got {:?}", other),
            }
        }
        _ => panic!("Expected Function def"),
    }
}

#[test]
fn test_no_statements() {
    // Test body with no statements (just a root expression)
    let mut b = TestBodyBuilder::new();

    let lit_42 = b.int_lit(42);
    let body = b.build(lit_42);
    let program = make_test_program(body);
    let result = lift_bindings(program);

    match &result.defs[0] {
        Def::Function { body, .. } => {
            // No locals, no statements
            assert_eq!(body.locals.len(), 0);
            let stmts: Vec<_> = body.iter_stmts().collect();
            assert_eq!(stmts.len(), 0);

            // Root should be Int(42)
            match body.get_expr(body.root) {
                Expr::Int(s) => assert_eq!(s, "42"),
                other => panic!("Expected Int, got {:?}", other),
            }
        }
        _ => panic!("Expected Function def"),
    }
}
