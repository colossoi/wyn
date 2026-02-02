//! Tests for the constant folding optimization pass.

use crate::IdArena;
use crate::ast::{NodeId, Span, TypeName};
use crate::constant_folding::fold_constants;
use crate::mir::{ArrayBacking, Block, Body, Def, Expr, ExprId, LocalDecl, LocalId, LocalKind, Program};
use polytype::Type;
use std::sync::atomic::{AtomicU32, Ordering};

// =============================================================================
// Test Helpers - Type Constructors
// =============================================================================

fn i32_type() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn f32_type() -> Type<TypeName> {
    Type::Constructed(TypeName::Float(32), vec![])
}

fn bool_type() -> Type<TypeName> {
    Type::Constructed(TypeName::Str("bool".into()), vec![])
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
    fn alloc_local(&mut self, name: &str, ty: Type<TypeName>) -> LocalId {
        self.body.alloc_local(LocalDecl {
            name: name.to_string(),
            span: test_span(),
            ty,
            kind: LocalKind::Let,
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

    /// Create a float literal expression.
    fn float_lit(&mut self, n: f64) -> ExprId {
        self.body.alloc_expr(Expr::Float(n.to_string()), f32_type(), test_span(), next_id())
    }

    /// Create a boolean literal expression.
    fn bool_lit(&mut self, b: bool) -> ExprId {
        self.body.alloc_expr(Expr::Bool(b), bool_type(), test_span(), next_id())
    }

    /// Create a binary operation expression.
    fn binop(&mut self, op: &str, lhs: ExprId, rhs: ExprId, ty: Type<TypeName>) -> ExprId {
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

    /// Create a unary operation expression.
    fn unop(&mut self, op: &str, operand: ExprId, ty: Type<TypeName>) -> ExprId {
        self.body.alloc_expr(
            Expr::UnaryOp {
                op: op.to_string(),
                operand,
            },
            ty,
            test_span(),
            next_id(),
        )
    }

    /// Create an if expression.
    fn if_expr(&mut self, cond: ExprId, then_: ExprId, else_: ExprId) -> ExprId {
        let ty = self.body.get_type(then_).clone();
        self.body.alloc_expr(
            Expr::If {
                cond,
                then_: Block::new(then_),
                else_: Block::new(else_),
            },
            ty,
            test_span(),
            next_id(),
        )
    }

    /// Create an array literal expression.
    fn array(&mut self, elements: Vec<ExprId>, ty: Type<TypeName>) -> ExprId {
        let len = elements.len();
        let size = self.body.alloc_expr(
            Expr::Int(len.to_string()),
            Type::Constructed(TypeName::Int(32), vec![]),
            test_span(),
            next_id(),
        );
        self.body.alloc_expr(
            Expr::Array {
                backing: ArrayBacking::Literal(elements),
                size,
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

/// Create a simple test program with one constant def.
fn make_test_program(body: Body) -> Program {
    Program {
        defs: vec![Def::Constant {
            id: next_id(),
            name: "test_const".to_string(),
            ty: i32_type(),
            attributes: vec![],
            body,
            span: test_span(),
        }],
        lambda_registry: IdArena::new(),
    }
}

// =============================================================================
// Test Helpers - Result Inspection
// =============================================================================

/// Get the root expression from the first def in the program.
fn get_root_expr(program: &Program) -> &Expr {
    match &program.defs[0] {
        Def::Constant { body, .. } | Def::Function { body, .. } | Def::EntryPoint { body, .. } => {
            body.get_expr(body.root)
        }
        _ => panic!("Expected def with body"),
    }
}

/// Get the body from the first def in the program.
fn get_body(program: &Program) -> &Body {
    match &program.defs[0] {
        Def::Constant { body, .. } | Def::Function { body, .. } | Def::EntryPoint { body, .. } => body,
        _ => panic!("Expected def with body"),
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[test]
fn test_fold_float_division() {
    // 135.0 / 255.0
    let mut b = TestBodyBuilder::new();
    let lhs = b.float_lit(135.0);
    let rhs = b.float_lit(255.0);
    let div = b.binop("/", lhs, rhs, f32_type());
    let body = b.build(div);

    let program = make_test_program(body);
    let result = fold_constants(program).unwrap();

    match get_root_expr(&result) {
        Expr::Float(val) => {
            let v: f64 = val.parse().unwrap();
            assert!((v - 0.529411765).abs() < 0.000001);
        }
        other => panic!("Expected folded float literal, got {:?}", other),
    }
}

#[test]
fn test_fold_integer_addition() {
    // 10 + 32
    let mut b = TestBodyBuilder::new();
    let lhs = b.int_lit(10);
    let rhs = b.int_lit(32);
    let add = b.binop("+", lhs, rhs, i32_type());
    let body = b.build(add);

    let program = make_test_program(body);
    let result = fold_constants(program).unwrap();

    match get_root_expr(&result) {
        Expr::Int(val) => {
            assert_eq!(val, "42");
        }
        other => panic!("Expected folded int literal, got {:?}", other),
    }
}

#[test]
fn test_fold_constant_if_true() {
    // if true then 1 else 2
    let mut b = TestBodyBuilder::new();
    let cond = b.bool_lit(true);
    let then_ = b.int_lit(1);
    let else_ = b.int_lit(2);
    let if_expr = b.if_expr(cond, then_, else_);
    let body = b.build(if_expr);

    let program = make_test_program(body);
    let result = fold_constants(program).unwrap();

    match get_root_expr(&result) {
        Expr::Int(val) => {
            assert_eq!(val, "1");
        }
        other => panic!("Expected folded to then branch, got {:?}", other),
    }
}

#[test]
fn test_fold_constant_if_false() {
    // if false then 1 else 2
    let mut b = TestBodyBuilder::new();
    let cond = b.bool_lit(false);
    let then_ = b.int_lit(1);
    let else_ = b.int_lit(2);
    let if_expr = b.if_expr(cond, then_, else_);
    let body = b.build(if_expr);

    let program = make_test_program(body);
    let result = fold_constants(program).unwrap();

    match get_root_expr(&result) {
        Expr::Int(val) => {
            assert_eq!(val, "2");
        }
        other => panic!("Expected folded to else branch, got {:?}", other),
    }
}

#[test]
fn test_fold_array_literal() {
    // [1 + 2, 3 * 4]
    let mut b = TestBodyBuilder::new();
    let one = b.int_lit(1);
    let two = b.int_lit(2);
    let add = b.binop("+", one, two, i32_type());
    let three = b.int_lit(3);
    let four = b.int_lit(4);
    let mul = b.binop("*", three, four, i32_type());
    let array = b.array(vec![add, mul], i32_type());
    let body = b.build(array);

    let program = make_test_program(body);
    let result = fold_constants(program).unwrap();

    let result_body = get_body(&result);
    match get_root_expr(&result) {
        Expr::Array {
            backing: ArrayBacking::Literal(elements),
            ..
        } => {
            assert_eq!(elements.len(), 2);
            match result_body.get_expr(elements[0]) {
                Expr::Int(v) => assert_eq!(v, "3"),
                other => panic!("Expected int literal, got {:?}", other),
            }
            match result_body.get_expr(elements[1]) {
                Expr::Int(v) => assert_eq!(v, "12"),
                other => panic!("Expected int literal, got {:?}", other),
            }
        }
        other => panic!("Expected array literal, got {:?}", other),
    }
}

#[test]
fn test_fold_negation() {
    // -42
    let mut b = TestBodyBuilder::new();
    let lit = b.int_lit(42);
    let neg = b.unop("-", lit, i32_type());
    let body = b.build(neg);

    let program = make_test_program(body);
    let result = fold_constants(program).unwrap();

    match get_root_expr(&result) {
        Expr::Int(val) => {
            assert_eq!(val, "-42");
        }
        other => panic!("Expected folded negation, got {:?}", other),
    }
}

#[test]
fn test_fold_boolean_not() {
    // !true
    let mut b = TestBodyBuilder::new();
    let lit = b.bool_lit(true);
    let not = b.unop("!", lit, bool_type());
    let body = b.build(not);

    let program = make_test_program(body);
    let result = fold_constants(program).unwrap();

    match get_root_expr(&result) {
        Expr::Bool(val) => {
            assert!(!*val);
        }
        other => panic!("Expected folded boolean, got {:?}", other),
    }
}

#[test]
fn test_fold_comparison() {
    // 10 < 20
    let mut b = TestBodyBuilder::new();
    let lhs = b.int_lit(10);
    let rhs = b.int_lit(20);
    let cmp = b.binop("<", lhs, rhs, bool_type());
    let body = b.build(cmp);

    let program = make_test_program(body);
    let result = fold_constants(program).unwrap();

    match get_root_expr(&result) {
        Expr::Bool(val) => {
            assert!(*val);
        }
        other => panic!("Expected folded comparison, got {:?}", other),
    }
}

#[test]
fn test_no_fold_with_variable() {
    // x + 1 (should not fold since x is a variable)
    let mut b = TestBodyBuilder::new();
    let x = b.alloc_local("x", i32_type());
    let x_ref = b.local(x);
    let one = b.int_lit(1);
    let add = b.binop("+", x_ref, one, i32_type());
    let body = b.build(add);

    let program = make_test_program(body);
    let result = fold_constants(program).unwrap();

    // Should remain a BinOp since we can't fold with a variable
    match get_root_expr(&result) {
        Expr::BinOp { op, .. } => {
            assert_eq!(op, "+");
        }
        other => panic!("Expected unfoldable binop, got {:?}", other),
    }
}
