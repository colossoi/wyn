//! Tests for the binding lifting (code motion) optimization pass.

use crate::IdArena;
use crate::ast::{NodeId, Span, TypeName};
use crate::binding_lifter::lift_bindings;
use crate::mir::{Body, Def, Expr, ExprId, LocalDecl, LocalId, LocalKind, LoopKind, Program};
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

    /// Create a let binding expression.
    fn let_bind(&mut self, local: LocalId, rhs: ExprId, body: ExprId) -> ExprId {
        let ty = self.body.get_type(body).clone();
        self.body.alloc_expr(Expr::Let { local, rhs, body }, ty, test_span(), next_id())
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
                body,
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
        }],
        lambda_registry: IdArena::new(),
    }
}

// =============================================================================
// Test Helpers - Structure Inspection
// =============================================================================

/// Get a simplified structural representation of an expression.
/// Returns a vector of tags like ["let", "loop", "let", "expr"].
fn get_structure(body: &Body, expr_id: ExprId) -> Vec<&'static str> {
    let mut result = vec![];
    collect_structure(body, expr_id, &mut result);
    result
}

fn collect_structure(body: &Body, expr_id: ExprId, result: &mut Vec<&'static str>) {
    match body.get_expr(expr_id) {
        Expr::Let { body: let_body, .. } => {
            result.push("let");
            collect_structure(body, *let_body, result);
        }
        Expr::Loop { body: loop_body, .. } => {
            result.push("loop");
            collect_structure(body, *loop_body, result);
        }
        _ => {
            result.push("expr");
        }
    }
}

/// Get the structure of a program's first function.
fn get_program_structure(program: &Program) -> Vec<&'static str> {
    match &program.defs[0] {
        Def::Function { body, .. } => get_structure(body, body.root),
        _ => panic!("Expected Function def"),
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[test]
fn test_hoist_constant_from_loop() {
    // loop acc = 0 for i < 10 do
    //   let x = 42 in acc + x
    // =>
    // let x = 42 in loop acc = 0 for i < 10 do acc + x

    let mut b = TestBodyBuilder::new();

    // Create locals
    let acc = b.alloc_local("acc", i32_type(), LocalKind::LoopVar);
    let i = b.alloc_local("i", i32_type(), LocalKind::LoopVar);
    let x = b.alloc_local("x", i32_type(), LocalKind::Let);

    // Build: let x = 42 in acc + x
    let lit_42 = b.int_lit(42);
    let acc_ref = b.local(acc);
    let x_ref = b.local(x);
    let add_expr = b.binop("+", acc_ref, x_ref);
    let let_x = b.let_bind(x, lit_42, add_expr);

    // Build: loop acc = 0 for i < 10 do <body>
    let init = b.int_lit(0);
    let bound = b.int_lit(10);
    let loop_expr = b.for_range_loop(acc, init, i, bound, let_x);

    let body = b.build(loop_expr);
    let program = make_test_program(body);
    let result = lift_bindings(program);

    let structure = get_program_structure(&result);
    assert_eq!(structure, vec!["let", "loop", "expr"]); // x hoisted before loop
}

#[test]
fn test_no_hoist_loop_dependent() {
    // loop acc = 0 for i < 10 do
    //   let x = acc + 1 in x * 2
    // => unchanged (x depends on acc)

    let mut b = TestBodyBuilder::new();

    let acc = b.alloc_local("acc", i32_type(), LocalKind::LoopVar);
    let i = b.alloc_local("i", i32_type(), LocalKind::LoopVar);
    let x = b.alloc_local("x", i32_type(), LocalKind::Let);

    // Build: let x = acc + 1 in x * 2
    let acc_ref = b.local(acc);
    let lit_1 = b.int_lit(1);
    let acc_plus_1 = b.binop("+", acc_ref, lit_1);
    let x_ref = b.local(x);
    let lit_2 = b.int_lit(2);
    let x_times_2 = b.binop("*", x_ref, lit_2);
    let let_x = b.let_bind(x, acc_plus_1, x_times_2);

    let init = b.int_lit(0);
    let bound = b.int_lit(10);
    let loop_expr = b.for_range_loop(acc, init, i, bound, let_x);

    let body = b.build(loop_expr);
    let program = make_test_program(body);
    let result = lift_bindings(program);

    let structure = get_program_structure(&result);
    assert_eq!(structure, vec!["loop", "let", "expr"]); // x stays in loop
}

#[test]
fn test_hoist_chain_of_invariants() {
    // loop acc = 0 for i < 10 do
    //   let x = 1 in
    //   let y = x + 2 in
    //   acc + y
    // =>
    // let x = 1 in let y = x + 2 in loop acc = 0 for i < 10 do acc + y

    let mut b = TestBodyBuilder::new();

    let acc = b.alloc_local("acc", i32_type(), LocalKind::LoopVar);
    let i = b.alloc_local("i", i32_type(), LocalKind::LoopVar);
    let x = b.alloc_local("x", i32_type(), LocalKind::Let);
    let y = b.alloc_local("y", i32_type(), LocalKind::Let);

    // Build innermost: acc + y
    let acc_ref = b.local(acc);
    let y_ref = b.local(y);
    let acc_plus_y = b.binop("+", acc_ref, y_ref);

    // Build: let y = x + 2 in acc + y
    let x_ref = b.local(x);
    let lit_2 = b.int_lit(2);
    let x_plus_2 = b.binop("+", x_ref, lit_2);
    let let_y = b.let_bind(y, x_plus_2, acc_plus_y);

    // Build: let x = 1 in <let_y>
    let lit_1 = b.int_lit(1);
    let let_x = b.let_bind(x, lit_1, let_y);

    let init = b.int_lit(0);
    let bound = b.int_lit(10);
    let loop_expr = b.for_range_loop(acc, init, i, bound, let_x);

    let body = b.build(loop_expr);
    let program = make_test_program(body);
    let result = lift_bindings(program);

    let structure = get_program_structure(&result);
    assert_eq!(structure, vec!["let", "let", "loop", "expr"]);
}

#[test]
fn test_partial_hoist_transitive_dependency() {
    // loop acc = 0 for i < 10 do
    //   let x = 42 in          -- hoistable (no deps)
    //   let y = acc + x in     -- NOT hoistable (depends on acc)
    //   let z = y * 2 in       -- NOT hoistable (depends on y which depends on acc)
    //   z

    let mut b = TestBodyBuilder::new();

    let acc = b.alloc_local("acc", i32_type(), LocalKind::LoopVar);
    let i = b.alloc_local("i", i32_type(), LocalKind::LoopVar);
    let x = b.alloc_local("x", i32_type(), LocalKind::Let);
    let y = b.alloc_local("y", i32_type(), LocalKind::Let);
    let z = b.alloc_local("z", i32_type(), LocalKind::Let);

    // Build innermost: z
    let z_ref = b.local(z);

    // Build: let z = y * 2 in z
    let y_ref = b.local(y);
    let lit_2 = b.int_lit(2);
    let y_times_2 = b.binop("*", y_ref, lit_2);
    let let_z = b.let_bind(z, y_times_2, z_ref);

    // Build: let y = acc + x in <let_z>
    let acc_ref = b.local(acc);
    let x_ref = b.local(x);
    let acc_plus_x = b.binop("+", acc_ref, x_ref);
    let let_y = b.let_bind(y, acc_plus_x, let_z);

    // Build: let x = 42 in <let_y>
    let lit_42 = b.int_lit(42);
    let let_x = b.let_bind(x, lit_42, let_y);

    let init = b.int_lit(0);
    let bound = b.int_lit(10);
    let loop_expr = b.for_range_loop(acc, init, i, bound, let_x);

    let body = b.build(loop_expr);
    let program = make_test_program(body);
    let result = lift_bindings(program);

    let structure = get_program_structure(&result);
    // x hoisted, y and z stay (transitive dependency through y)
    assert_eq!(structure, vec!["let", "loop", "let", "let", "expr"]);
}

#[test]
fn test_iteration_var_dependency() {
    // loop acc = 0 for i < 10 do
    //   let x = i * 2 in       -- NOT hoistable (depends on i)
    //   acc + x

    let mut b = TestBodyBuilder::new();

    let acc = b.alloc_local("acc", i32_type(), LocalKind::LoopVar);
    let i = b.alloc_local("i", i32_type(), LocalKind::LoopVar);
    let x = b.alloc_local("x", i32_type(), LocalKind::Let);

    // Build: let x = i * 2 in acc + x
    let i_ref = b.local(i);
    let lit_2 = b.int_lit(2);
    let i_times_2 = b.binop("*", i_ref, lit_2);
    let acc_ref = b.local(acc);
    let x_ref = b.local(x);
    let acc_plus_x = b.binop("+", acc_ref, x_ref);
    let let_x = b.let_bind(x, i_times_2, acc_plus_x);

    let init = b.int_lit(0);
    let bound = b.int_lit(10);
    let loop_expr = b.for_range_loop(acc, init, i, bound, let_x);

    let body = b.build(loop_expr);
    let program = make_test_program(body);
    let result = lift_bindings(program);

    let structure = get_program_structure(&result);
    assert_eq!(structure, vec!["loop", "let", "expr"]); // x stays in loop
}

#[test]
fn test_no_bindings_in_loop() {
    // loop acc = 0 for i < 10 do acc + 1
    // => unchanged (nothing to hoist)

    let mut b = TestBodyBuilder::new();

    let acc = b.alloc_local("acc", i32_type(), LocalKind::LoopVar);
    let i = b.alloc_local("i", i32_type(), LocalKind::LoopVar);

    // Build: acc + 1
    let acc_ref = b.local(acc);
    let lit_1 = b.int_lit(1);
    let acc_plus_1 = b.binop("+", acc_ref, lit_1);

    let init = b.int_lit(0);
    let bound = b.int_lit(10);
    let loop_expr = b.for_range_loop(acc, init, i, bound, acc_plus_1);

    let body = b.build(loop_expr);
    let program = make_test_program(body);
    let result = lift_bindings(program);

    let structure = get_program_structure(&result);
    assert_eq!(structure, vec!["loop", "expr"]);
}

#[test]
fn test_multiple_independent_hoistable() {
    // loop acc = 0 for i < 10 do
    //   let x = 1 in
    //   let y = 2 in
    //   acc + x + y
    // =>
    // let x = 1 in let y = 2 in loop ... acc + x + y

    let mut b = TestBodyBuilder::new();

    let acc = b.alloc_local("acc", i32_type(), LocalKind::LoopVar);
    let i = b.alloc_local("i", i32_type(), LocalKind::LoopVar);
    let x = b.alloc_local("x", i32_type(), LocalKind::Let);
    let y = b.alloc_local("y", i32_type(), LocalKind::Let);

    // Build: acc + x + y
    let acc_ref = b.local(acc);
    let x_ref = b.local(x);
    let y_ref = b.local(y);
    let acc_plus_x = b.binop("+", acc_ref, x_ref);
    let add_y = b.binop("+", acc_plus_x, y_ref);

    // Build: let y = 2 in <add_y>
    let lit_2 = b.int_lit(2);
    let let_y = b.let_bind(y, lit_2, add_y);

    // Build: let x = 1 in <let_y>
    let lit_1 = b.int_lit(1);
    let let_x = b.let_bind(x, lit_1, let_y);

    let init = b.int_lit(0);
    let bound = b.int_lit(10);
    let loop_expr = b.for_range_loop(acc, init, i, bound, let_x);

    let body = b.build(loop_expr);
    let program = make_test_program(body);
    let result = lift_bindings(program);

    let structure = get_program_structure(&result);
    assert_eq!(structure, vec!["let", "let", "loop", "expr"]);
}

#[test]
fn test_mixed_hoistable_and_dependent() {
    // loop acc = 0 for i < 10 do
    //   let a = 100 in         -- hoistable
    //   let b = acc in         -- NOT hoistable
    //   let c = 200 in         -- hoistable (doesn't depend on b)
    //   b + a + c

    let mut b = TestBodyBuilder::new();

    let acc = b.alloc_local("acc", i32_type(), LocalKind::LoopVar);
    let i = b.alloc_local("i", i32_type(), LocalKind::LoopVar);
    let a = b.alloc_local("a", i32_type(), LocalKind::Let);
    let b_local = b.alloc_local("b", i32_type(), LocalKind::Let);
    let c = b.alloc_local("c", i32_type(), LocalKind::Let);

    // Build: b + a + c
    let b_ref = b.local(b_local);
    let a_ref = b.local(a);
    let c_ref = b.local(c);
    let b_plus_a = b.binop("+", b_ref, a_ref);
    let add_c = b.binop("+", b_plus_a, c_ref);

    // Build: let c = 200 in <add_c>
    let lit_200 = b.int_lit(200);
    let let_c = b.let_bind(c, lit_200, add_c);

    // Build: let b = acc in <let_c>
    let acc_ref = b.local(acc);
    let let_b = b.let_bind(b_local, acc_ref, let_c);

    // Build: let a = 100 in <let_b>
    let lit_100 = b.int_lit(100);
    let let_a = b.let_bind(a, lit_100, let_b);

    let init = b.int_lit(0);
    let bound = b.int_lit(10);
    let loop_expr = b.for_range_loop(acc, init, i, bound, let_a);

    let body = b.build(loop_expr);
    let program = make_test_program(body);
    let result = lift_bindings(program);

    let structure = get_program_structure(&result);
    // a and c hoisted, b stays
    assert_eq!(structure, vec!["let", "let", "loop", "let", "expr"]);
}
