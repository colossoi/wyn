//! Tests for the ANF normalization pass.

use crate::ast::{NodeCounter, NodeId, Span, TypeName};
use crate::error::CompilerError;
use crate::mir::{Expr, ExprKind, Literal};
use crate::normalize::{Normalizer, is_atomic};
use polytype::Type;
use std::sync::atomic::{AtomicU32, Ordering};

/// Helper to run full pipeline through lowering with normalization
fn compile_through_lowering(input: &str) -> Result<(), CompilerError> {
    let (module_manager, mut node_counter) = crate::cached_module_manager();
    let parsed = crate::Compiler::parse(input, &mut node_counter)?;
    let (flattened, mut backend) = parsed
        .desugar(&mut node_counter)?
        .resolve(&module_manager)?
        .type_check(&module_manager)?
        .alias_check()?
        .fold_ast_constants()
        .flatten(&module_manager)?;
    flattened
        .hoist_materializations()
        .normalize(&mut backend.node_counter)
        .monomorphize()?
        .filter_reachable()
        .fold_constants()?
        .lift_bindings()?
        .lower()?;
    Ok(())
}

fn test_span() -> Span {
    Span::new(1, 1, 1, 1)
}

fn i32_type() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

// Global counter for test node IDs (tests don't care about the actual values)
static TEST_NODE_ID: AtomicU32 = AtomicU32::new(0);

fn next_id() -> NodeId {
    NodeId(TEST_NODE_ID.fetch_add(1, Ordering::Relaxed))
}

fn var(name: &str) -> Expr {
    Expr::new(
        next_id(),
        i32_type(),
        ExprKind::Var(name.to_string()),
        test_span(),
    )
}

fn int_lit(n: i32) -> Expr {
    Expr::new(
        next_id(),
        i32_type(),
        ExprKind::Literal(Literal::Int(n.to_string())),
        test_span(),
    )
}

fn binop(op: &str, lhs: Expr, rhs: Expr) -> Expr {
    Expr::new(
        next_id(),
        i32_type(),
        ExprKind::BinOp {
            op: op.to_string(),
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
        test_span(),
    )
}

fn call(func: &str, args: Vec<Expr>) -> Expr {
    Expr::new(
        next_id(),
        i32_type(),
        ExprKind::Call {
            func: func.to_string(),
            args,
        },
        test_span(),
    )
}

#[test]
fn test_atomic_var() {
    let expr = var("x");
    assert!(is_atomic(&expr));
}

#[test]
fn test_atomic_int_literal() {
    let expr = int_lit(42);
    assert!(is_atomic(&expr));
}

#[test]
fn test_not_atomic_binop() {
    let expr = binop("+", var("a"), var("b"));
    assert!(!is_atomic(&expr));
}

#[test]
fn test_normalize_binop_with_atomic_operands() {
    // a + b should stay as a + b (no new bindings)
    let expr = binop("+", var("a"), var("b"));
    let mut bindings = Vec::new();
    let mut normalizer = Normalizer::new(0, NodeCounter::new());
    let result = normalizer.normalize_expr(expr, &mut bindings);

    assert!(bindings.is_empty());
    matches!(result.kind, ExprKind::BinOp { .. });
}

#[test]
fn test_normalize_nested_binop() {
    // (a + b) + c should become:
    // let _w_norm_0 = a + b in _w_norm_0 + c
    let inner = binop("+", var("a"), var("b"));
    let expr = binop("+", inner, var("c"));
    let mut bindings = Vec::new();
    let mut normalizer = Normalizer::new(0, NodeCounter::new());
    let result = normalizer.normalize_expr(expr, &mut bindings);

    // Should have one binding for the inner binop
    assert_eq!(bindings.len(), 1);
    assert_eq!(bindings[0].0, "_w_norm_0");

    // Result should be _w_norm_0 + c
    if let ExprKind::BinOp { lhs, rhs, .. } = &result.kind {
        assert!(matches!(lhs.kind, ExprKind::Var(ref n) if n == "_w_norm_0"));
        assert!(matches!(rhs.kind, ExprKind::Var(ref n) if n == "c"));
    } else {
        panic!("Expected BinOp");
    }
}

#[test]
fn test_normalize_call_with_binop_arg() {
    // foo(a + b) should become:
    // let _w_norm_0 = a + b in foo(_w_norm_0)
    let arg = binop("+", var("a"), var("b"));
    let expr = call("foo", vec![arg]);
    let mut bindings = Vec::new();
    let mut normalizer = Normalizer::new(0, NodeCounter::new());
    let result = normalizer.normalize_expr(expr, &mut bindings);

    // Should have one binding for the binop
    assert_eq!(bindings.len(), 1);

    // Result should be foo(_w_norm_0)
    if let ExprKind::Call { args, .. } = &result.kind {
        assert_eq!(args.len(), 1);
        assert!(matches!(args[0].kind, ExprKind::Var(ref n) if n == "_w_norm_0"));
    } else {
        panic!("Expected Call");
    }
}

#[test]
fn test_normalize_loop_with_tuple_state() {
    // This tests that loops with tuple state work correctly after normalization.
    // The loop produces a tuple (acc, i) on each iteration.
    let source = r#"
def sum<[n]>(arr:[n]f32) -> f32 =
  let (result, _) = loop (acc, i) = (0.0f32, 0) while i < length(arr) do
    (acc + arr[i], i + 1)
  in result

#[vertex]
def vertex_main(vertex_id:i32) -> #[builtin(position)] vec4f32 =
  let result = sum([1.0f32, 1.0f32, 1.0f32]) in
  @[result, result, 0.0f32, 1.0f32]
"#;
    compile_through_lowering(source).expect("Should compile with normalized loop");
}
