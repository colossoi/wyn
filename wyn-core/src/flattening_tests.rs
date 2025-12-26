#![cfg(test)]

use crate::lexer::tokenize;
use crate::mir;
use crate::parser::Parser;
use crate::type_checker::TypeChecker;

fn flatten_program(input: &str) -> mir::Program {
    // Use the typestate API to ensure proper compilation pipeline
    let mut frontend = crate::cached_frontend();
    let parsed = crate::Compiler::parse(input, &mut frontend.node_counter).expect("Parsing failed");
    let (flattened, _backend) = parsed
        .desugar(&mut frontend.node_counter)
        .expect("Desugaring failed")
        .resolve(&frontend.module_manager)
        .expect("Name resolution failed")
        .fold_ast_constants()
        .type_check(&frontend.module_manager, &mut frontend.schemes)
        .expect("Type checking failed")
        .alias_check()
        .expect("Borrow checking failed")
        .flatten(&frontend.module_manager, &frontend.schemes)
        .expect("Flattening failed");
    // Run hoisting pass to optimize materializations
    flattened.hoist_materializations().mir
}

fn flatten_to_string(input: &str) -> String {
    format!("{}", flatten_program(input))
}

/// Extract just one function's MIR from the full program output
fn extract_function_mir(mir_str: &str, fn_name: &str) -> String {
    let prefix = format!("def {} ", fn_name);
    let mut result = String::new();
    let mut capturing = false;

    for line in mir_str.lines() {
        if line.starts_with(&prefix) {
            capturing = true;
        } else if capturing && line.starts_with("def ") {
            break;
        }
        if capturing {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

/// Find a definition by name (can be Function or Constant)
fn find_def<'a>(mir: &'a mir::Program, name: &str) -> &'a mir::Def {
    mir.defs
        .iter()
        .find(|d| match d {
            mir::Def::Function { name: n, .. } => n == name,
            mir::Def::Constant { name: n, .. } => n == name,
            _ => false,
        })
        .unwrap_or_else(|| panic!("Definition '{}' not found", name))
}

/// Get the body from a definition (Function or Constant)
fn get_body<'a>(def: &'a mir::Def) -> &'a mir::Body {
    match def {
        mir::Def::Function { body, .. } => body,
        mir::Def::Constant { body, .. } => body,
        other => panic!("Expected Function or Constant, got {:?}", other),
    }
}

/// Helper to check that code fails type checking (for testing error cases)
fn should_fail_type_check(input: &str) -> bool {
    let (module_manager, mut node_counter) = crate::cached_module_manager();
    let tokens = tokenize(input).expect("Tokenization failed");
    let mut parser = Parser::new(tokens, &mut node_counter);
    let ast = parser.parse().expect("Parsing failed");

    let mut type_checker = TypeChecker::new(&module_manager);
    type_checker.load_builtins().expect("Failed to load builtins");
    type_checker.check_program(&ast).is_err()
}

#[test]
fn test_simple_constant() {
    let mir = flatten_program("def x = 42");
    // Nullary function becomes a Constant def in MIR
    let x_def = find_def(&mir, "x");
    let body = get_body(x_def);
    // Root should be an integer literal
    match body.get_expr(body.root) {
        mir::Expr::Int(s) => assert_eq!(s, "42"),
        other => panic!("Expected Int, got {:?}", other),
    }
}

#[test]
fn test_simple_function() {
    let mir = flatten_program("def add(x, y) = x + y");
    let add_def = mir.defs.iter().find(|d| matches!(d, mir::Def::Function { name, .. } if name == "add"));
    let add_def = add_def.expect("Expected function 'add'");
    match add_def {
        mir::Def::Function { params, body, .. } => {
            assert_eq!(params.len(), 2);
            // Root should be a BinOp
            match body.get_expr(body.root) {
                mir::Expr::BinOp { op, .. } => assert_eq!(op, "+"),
                other => panic!("Expected BinOp, got {:?}", other),
            }
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_let_binding() {
    // Use variables in the expression to prevent constant folding
    let mir = flatten_program("def f(a, b) = let x = a in x + b");
    let f_def = find_def(&mir, "f");
    let body = get_body(f_def);
    // Root should be a Let
    match body.get_expr(body.root) {
        mir::Expr::Let {
            rhs, body: let_body, ..
        } => {
            // RHS should be a local variable reference
            match body.get_expr(*rhs) {
                mir::Expr::Local(_) => {}
                other => panic!("Expected Local for rhs, got {:?}", other),
            }
            // Body should be BinOp
            match body.get_expr(*let_body) {
                mir::Expr::BinOp { op, .. } => assert_eq!(op, "+"),
                other => panic!("Expected BinOp for body, got {:?}", other),
            }
        }
        other => panic!("Expected Let, got {:?}", other),
    }
}

#[test]
fn test_tuple_pattern() {
    let mir = flatten_program("def f = let (a, b) = (1, 2) in a + b");
    // Tuple pattern desugars to multiple lets with intrinsic tuple access
    let f_def = find_def(&mir, "f");
    let body = get_body(f_def);
    // Should have Let bindings for tuple destructuring
    // The structure may vary, but there should be intrinsic calls for tuple_access
    let has_tuple_access = body
        .exprs
        .iter()
        .any(|expr| matches!(expr, mir::Expr::Intrinsic { name, .. } if name == "tuple_access"));
    assert!(
        has_tuple_access,
        "Expected tuple_access intrinsic for tuple pattern"
    );
}

#[test]
fn test_lambda_tuple_pattern_param() {
    // Test lambda with tuple pattern parameter: |(x, y)| x + y
    let mir = flatten_program("def f = let add = |(x, y)| x + y in add((1, 2))");

    // Check that lambda registry has the lambda
    assert!(
        !mir.lambda_registry.is_empty(),
        "Lambda registry should contain the generated lambda"
    );

    // Find the generated lambda function and verify it destructures the tuple param
    let add_fn = mir.defs.iter().find(|d| {
        if let mir::Def::Function { name, .. } = d { name.contains("_w_lam_f_") } else { false }
    });
    assert!(add_fn.is_some(), "Generated lambda function should exist");

    if let Some(mir::Def::Function { body, .. }) = add_fn {
        // The body should contain tuple_access intrinsics for destructuring the param
        let has_tuple_access = body
            .exprs
            .iter()
            .any(|expr| matches!(expr, mir::Expr::Intrinsic { name, .. } if name == "tuple_access"));
        assert!(
            has_tuple_access,
            "Lambda with tuple param should have tuple_access for destructuring"
        );
    }
}

#[test]
fn test_lambda_nested_tuple_pattern_param() {
    // Test lambda with nested tuple pattern: |((a, b), c)| a + b + c
    let mir = flatten_program("def f = let add = |((a, b), c)| a + b + c in add(((1, 2), 3))");

    // Should compile successfully with tuple destructuring
    assert!(
        !mir.lambda_registry.is_empty(),
        "Lambda registry should contain the generated lambda"
    );
}

#[test]
fn test_lambda_defunctionalization() {
    let mir = flatten_program("def f = |x| x + 1");

    // Check that lambda registry has the lambda
    assert!(
        !mir.lambda_registry.is_empty(),
        "Lambda registry should contain the generated lambda"
    );

    // Check that closure is created in the body
    let f_def = find_def(&mir, "f");
    let body = get_body(f_def);
    let has_closure = body.exprs.iter().any(
        |expr| matches!(expr, mir::Expr::Closure { lambda_name, .. } if lambda_name.contains("_w_lam_f_")),
    );
    assert!(has_closure, "Expected @closure expression with _w_lam_f_ prefix");
}

#[test]
fn test_lambda_with_capture() {
    let mir = flatten_program("def f(y) = let g = |x| x + y in g(1)");

    // Lambda should capture y - check for closure with captures
    let f_def = find_def(&mir, "f");
    let body = get_body(f_def);
    let has_closure_with_capture = body.exprs.iter().any(|expr| {
        if let mir::Expr::Closure { captures, .. } = expr {
            // Check that captures points to a non-Unit expression (i.e., has actual captures)
            !matches!(body.exprs[captures.index()], mir::Expr::Unit)
        } else {
            false
        }
    });
    assert!(
        has_closure_with_capture,
        "Expected closure with captured variables"
    );
}

#[test]
fn test_nested_let() {
    // Use parameters to prevent constant folding
    let mir = flatten_program("def f(a, b) = let x = a in let y = b in x + y");
    let f_def = find_def(&mir, "f");
    let body = get_body(f_def);
    // Root should be a Let (outer)
    match body.get_expr(body.root) {
        mir::Expr::Let { body: outer_body, .. } => {
            // Inner should also be a Let
            match body.get_expr(*outer_body) {
                mir::Expr::Let { body: inner_body, .. } => {
                    // Innermost should be BinOp
                    match body.get_expr(*inner_body) {
                        mir::Expr::BinOp { op, .. } => assert_eq!(op, "+"),
                        other => panic!("Expected BinOp, got {:?}", other),
                    }
                }
                other => panic!("Expected inner Let, got {:?}", other),
            }
        }
        other => panic!("Expected outer Let, got {:?}", other),
    }
}

#[test]
fn test_if_expression() {
    let mir = flatten_program("def f(x) = if x then 1 else 0");
    let f_def = find_def(&mir, "f");
    let body = get_body(f_def);
    match body.get_expr(body.root) {
        mir::Expr::If { then_, else_, .. } => {
            match body.get_expr(*then_) {
                mir::Expr::Int(s) => assert_eq!(s, "1"),
                other => panic!("Expected Int 1, got {:?}", other),
            }
            match body.get_expr(*else_) {
                mir::Expr::Int(s) => assert_eq!(s, "0"),
                other => panic!("Expected Int 0, got {:?}", other),
            }
        }
        other => panic!("Expected If, got {:?}", other),
    }
}

#[test]
fn test_function_call() {
    let mir = flatten_program("def g(y: i32) -> i32 = y + 1\ndef f(x) = g(x)");
    assert!(mir.defs.len() >= 2, "Expected at least 2 functions");

    let names: Vec<_> = mir
        .defs
        .iter()
        .filter_map(|d| match d {
            mir::Def::Function { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"g"), "Expected function g");
    assert!(names.contains(&"f"), "Expected function f");
}

#[test]
fn test_array_literal() {
    let mir = flatten_program("def arr = [1, 2, 3]");
    let arr_def = find_def(&mir, "arr");
    let body = get_body(arr_def);
    match body.get_expr(body.root) {
        mir::Expr::Array(elems) => {
            assert_eq!(elems.len(), 3, "Expected 3 elements");
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

#[test]
fn test_record_literal() {
    // Records are now represented as tuples in MIR, with fields in source order
    let mir = flatten_program("def r = {x: 1, y: 2}");
    let r_def = find_def(&mir, "r");
    let body = get_body(r_def);
    match body.get_expr(body.root) {
        mir::Expr::Tuple(elems) => {
            assert_eq!(elems.len(), 2, "Expected 2 element tuple for record");
        }
        other => panic!("Expected Tuple (record representation), got {:?}", other),
    }
}

#[test]
fn test_while_loop() {
    let mir = flatten_program("def f = loop x = 0 while x < 10 do x + 1");
    let f_def = find_def(&mir, "f");
    let body = get_body(f_def);
    match body.get_expr(body.root) {
        mir::Expr::Loop { kind, .. } => match kind {
            mir::LoopKind::While { .. } => {}
            other => panic!("Expected While loop kind, got {:?}", other),
        },
        other => panic!("Expected Loop, got {:?}", other),
    }
}

#[test]
fn test_for_range_loop() {
    let mir = flatten_program("def f = loop acc = 0 for i < 10 do acc + i");
    let f_def = find_def(&mir, "f");
    let body = get_body(f_def);
    match body.get_expr(body.root) {
        mir::Expr::Loop { kind, .. } => match kind {
            mir::LoopKind::ForRange { .. } => {}
            other => panic!("Expected ForRange loop kind, got {:?}", other),
        },
        other => panic!("Expected Loop, got {:?}", other),
    }
}

#[test]
fn test_binary_ops() {
    let mir = flatten_program("def f(x, y) = x * y + x / y");
    let f_def = find_def(&mir, "f");
    let body = get_body(f_def);
    // Check that we have +, *, and / operations in the body
    let ops: std::collections::HashSet<_> = body
        .exprs
        .iter()
        .filter_map(
            |e| {
                if let mir::Expr::BinOp { op, .. } = e { Some(op.as_str()) } else { None }
            },
        )
        .collect();
    assert!(ops.contains("+"), "Expected + operator");
    assert!(ops.contains("*"), "Expected * operator");
    assert!(ops.contains("/"), "Expected / operator");
}

#[test]
fn test_unary_op() {
    let mir = flatten_program("def f(x) = -x");
    let f_def = find_def(&mir, "f");
    let body = get_body(f_def);
    match body.get_expr(body.root) {
        mir::Expr::UnaryOp { op, .. } => assert_eq!(op, "-"),
        other => panic!("Expected UnaryOp, got {:?}", other),
    }
}

#[test]
fn test_array_index() {
    let mir = flatten_program("def f(arr, i) = arr[i]");
    let f_def = find_def(&mir, "f");
    let body = get_body(f_def);
    // Array index becomes intrinsic call
    let has_index =
        body.exprs.iter().any(|e| matches!(e, mir::Expr::Intrinsic { name, .. } if name == "index"));
    assert!(has_index, "Expected index intrinsic for array indexing");
}

#[test]
fn test_multiple_lambdas() {
    let mir = flatten_program(
        r#"
def f =
    let a = |x| x + 1 in
    let b = |y| y * 2 in
    (a, b)
"#,
    );
    // Should have original + 2 lambdas
    assert!(mir.defs.len() >= 3);
}

#[test]
fn test_map_with_lambda() {
    // Test map with inline lambda - just verify it compiles and produces a def
    let mir = flatten_program(
        r#"
def test: [4]i32 =
    map((|x: i32| x + 1), [0, 1, 2, 3])
"#,
    );
    // Should have at least the test function and a lambda
    assert!(mir.defs.len() >= 2, "Expected at least test function and lambda");
    // Should have lambda in registry
    assert!(
        !mir.lambda_registry.is_empty(),
        "Lambda registry should have the inline lambda"
    );
}

#[test]
fn test_lambda_captures_typed_variable() {
    // This test reproduces an issue where a lambda captures a typed variable (like an array),
    // and the free variable rewriting creates _w_closure.mat, which then fails when trying
    // to resolve 'mat' as a field access on the closure.
    let mir = flatten_program(
        r#"
def test_capture(arr: [4]i32) -> i32 =
    let result = map((|i: i32| arr[i]), [0, 1, 2, 3]) in
    result[0]
"#,
    );
    // Lambda should capture arr - check for closure with captures in body
    let mut has_closure_with_capture = false;
    for def in &mir.defs {
        if let mir::Def::Function { body, .. } = def {
            for expr in &body.exprs {
                if let mir::Expr::Closure { captures, .. } = expr {
                    // Check that captures points to a non-Unit expression
                    if !matches!(body.exprs[captures.index()], mir::Expr::Unit) {
                        has_closure_with_capture = true;
                        break;
                    }
                }
            }
        }
    }
    assert!(has_closure_with_capture, "Lambda should capture arr");
}

#[test]
fn test_qualified_name_f32_sqrt() {
    // This test reproduces an issue where f32.sqrt is treated as field access
    // instead of a qualified builtin name. The identifier 'f32' has no type in
    // the type_table because it's a type name, not a variable.
    let mir = flatten_program(
        r#"
def length2(v: vec2f32) -> f32 =
    f32.sqrt((v.x * v.x + v.y * v.y))
"#,
    );
    // Check for f32.sqrt call (should be a Call with qualified name)
    let mut has_sqrt_call = false;
    for def in &mir.defs {
        if let mir::Def::Function { body, .. } = def {
            for expr in &body.exprs {
                if let mir::Expr::Call { func, .. } = expr {
                    if func.contains("sqrt") {
                        has_sqrt_call = true;
                        break;
                    }
                }
            }
        }
    }
    assert!(has_sqrt_call, "Expected f32.sqrt call");
}

#[test]
fn test_map_with_closure_application() {
    // This test checks that map with a lambda generates a closure record
    // and registers the lambda in the registry for dispatch.
    let mir = flatten_program(
        r#"
def test_map(arr: [4]i32) -> [4]i32 =
    map((|x: i32| x + 1), arr)
"#,
    );

    // Should have lambda function generated
    assert!(mir.defs.len() >= 2, "Expected test function + lambda");

    // Lambda registry should have the lambda function
    assert_eq!(mir.lambda_registry.len(), 1, "Expected 1 lambda in registry");
    let (_, info) = mir.lambda_registry.iter().next().unwrap();
    assert!(
        info.name.contains("_w_lam_test_map_"),
        "Lambda name should contain _w_lam_test_map_"
    );
    assert_eq!(info.arity, 1, "Lambda should have arity 1");

    // Check for closure expression
    let mut has_closure = false;
    for def in &mir.defs {
        if let mir::Def::Function { body, name, .. } = def {
            if name == "test_map" {
                for expr in &body.exprs {
                    if let mir::Expr::Closure { lambda_name, .. } = expr {
                        if lambda_name.contains("_w_lam_test_map_") {
                            has_closure = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    assert!(
        has_closure,
        "Expected closure expression with _w_lam_test_map_ prefix"
    );
}

#[test]
fn test_direct_closure_call() {
    // This test checks that directly calling a closure generates a direct lambda call
    let mir = flatten_program(
        r#"
def test_apply(x: i32) -> i32 =
    let f = |y: i32| y + x in
    f(10)
"#,
    );

    // Should have lambda function generated
    assert!(mir.defs.len() >= 2, "Expected test function + lambda");

    // Lambda registry should have the lambda function
    assert!(
        !mir.lambda_registry.is_empty(),
        "Lambda registry should have the generated lambda"
    );

    // Check for direct call to lambda (should NOT use apply1 intrinsic)
    let mut has_apply_intrinsic = false;
    for def in &mir.defs {
        if let mir::Def::Function { body, name, .. } = def {
            if name == "test_apply" {
                for expr in &body.exprs {
                    if let mir::Expr::Intrinsic { name, .. } = expr {
                        if name == "apply1" {
                            has_apply_intrinsic = true;
                        }
                    }
                }
            }
        }
    }
    assert!(
        !has_apply_intrinsic,
        "Direct closure call should NOT use apply1 intrinsic"
    );
}

// Tests for function value restrictions (Futhark-style defunctionalization constraints)

#[test]
fn test_error_array_of_functions() {
    // Arrays of functions are not permitted
    assert!(
        should_fail_type_check(
            r#"
def test: [2](i32 -> i32) =
    [|x: i32| x + 1, |x: i32| x * 2]
"#
        ),
        "Should reject arrays of functions"
    );
}

#[test]
fn test_error_function_from_if() {
    // A function cannot be returned from an if expression
    assert!(
        should_fail_type_check(
            r#"
def choose(b: bool) -> (i32 -> i32) =
    if b then |x: i32| x + 1 else |x: i32| x * 2
"#
        ),
        "Should reject function returned from if expression"
    );
}

#[test]
fn test_error_loop_parameter_function() {
    // A loop parameter cannot be a function
    assert!(
        should_fail_type_check(
            r#"
def test: (i32 -> i32) =
    loop f = |x: i32| x while false do f
"#
        ),
        "Should reject function as loop parameter"
    );
}

#[test]
fn test_lambda_with_vector_literal() {
    // Test that vector literals work inside lambdas
    let mir = flatten_program(
        r#"
def test(v: vec3f32) -> vec4f32 =
    let f = |x: vec3f32| @[x.x, x.y, x.z, 1.0f32] in
    f(v)
"#,
    );

    // Check that we have a Vector expression somewhere in the MIR
    let mut has_vector = false;
    for def in &mir.defs {
        if let mir::Def::Function { body, .. } = def {
            for expr in &body.exprs {
                if matches!(expr, mir::Expr::Vector(_)) {
                    has_vector = true;
                    break;
                }
            }
        }
    }
    assert!(has_vector, "Expected Vector literal in MIR");
}

#[test]
fn test_f32_sum_inline_definition() {
    // Test sum with inline definition (simpler than using prelude)
    let mir = flatten_program(
        r#"
def mysum<[n]>(arr: [n]f32) -> f32 =
  let (result, _) = loop (acc, i) = (0.0f32, 0) while i < length(arr) do
    (acc + arr[i], i + 1)
  in result

def test: f32 =
  let arr = [1.0f32, 2.0f32, 3.0f32] in
  mysum(arr)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("mysum"));
}

#[test]
fn test_f32_sum_simple() {
    // Test that f32.sum from prelude works through full compilation
    let source = r#"
def test: f32 =
  let arr = [1.0f32, 2.0f32, 3.0f32] in
  f32.sum(arr)
"#;

    // This should compile successfully
    let mut frontend = crate::cached_frontend();
    let result = crate::Compiler::parse(source, &mut frontend.node_counter)
        .and_then(|p| p.desugar(&mut frontend.node_counter))
        .and_then(|d| d.resolve(&frontend.module_manager))
        .map(|r| r.fold_ast_constants())
        .and_then(|f| f.type_check(&frontend.module_manager, &mut frontend.schemes))
        .and_then(|t| t.alias_check())
        .and_then(|a| a.flatten(&frontend.module_manager, &frontend.schemes))
        .map(|(f, _backend)| f.hoist_materializations().normalize())
        .and_then(|n| n.monomorphize())
        .map(|m| m.skip_folding())
        .map(|f| f.filter_reachable())
        .map(|r| r.lift_bindings())
        .and_then(|l| l.lower());
    assert!(result.is_ok(), "Compilation failed: {:?}", result.err());
}

#[test]
fn test_f32_conversions() {
    // Test f32 type conversion via module functions
    let mir = flatten_program(
        r#"
def test_conversions(x: i32) -> f32 =
  let f1 = f32.i32(x) in
  let i1 = f32.to_i64(f1) in
  let f2 = f32.i64(i1) in
  f2
"#,
    );
    // Check for conversion calls in the MIR
    let f_def = find_def(&mir, "test_conversions");
    let body = get_body(f_def);
    let has_conversion = body.exprs.iter().any(|e| {
        matches!(e, mir::Expr::Call { func, .. } if func.contains("f32") && (func.contains("i32") || func.contains("i64")))
    });
    assert!(has_conversion, "Expected f32 conversion calls in MIR");
}

#[test]
fn test_f32_math_operations() {
    // Test f32 math operations including GLSL extended ops
    let mir = flatten_program(
        r#"
def test_math(x: f32) -> f32 =
  let a = f32.sin(x) in
  let b = f32.cos(x) in
  let c = f32.sqrt(a) in
  let d = f32.exp(b) in
  let e = f32.log(c) in
  let f = d ** 2.0f32 in
  let g = f32.sinh(x) in
  let h = f32.asinh(g) in
  let i = f32.atan2(x, a) in
  f32.fma(f, e, i)
"#,
    );
    // Check for f32 math function calls in the MIR
    let f_def = find_def(&mir, "test_math");
    let body = get_body(f_def);
    let call_names: Vec<_> = body
        .exprs
        .iter()
        .filter_map(
            |e| {
                if let mir::Expr::Call { func, .. } = e { Some(func.as_str()) } else { None }
            },
        )
        .collect();
    // Check for key math operations
    assert!(call_names.iter().any(|n| n.contains("sin")), "Expected sin call");
    assert!(call_names.iter().any(|n| n.contains("cos")), "Expected cos call");
    assert!(
        call_names.iter().any(|n| n.contains("sqrt")),
        "Expected sqrt call"
    );
    assert!(
        call_names.iter().any(|n| n.contains("sinh")),
        "Expected sinh call"
    );
    assert!(
        call_names.iter().any(|n| n.contains("asinh")),
        "Expected asinh call"
    );
    assert!(
        call_names.iter().any(|n| n.contains("atan2")),
        "Expected atan2 call"
    );
    assert!(call_names.iter().any(|n| n.contains("fma")), "Expected fma call");
}

#[test]
fn test_operator_section_direct_application() {
    // Test operator section applied directly to arguments
    let mir = flatten_program(
        r#"
def test_add(x: i32, y: i32) -> i32 = (+)(x, y)
"#,
    );
    let mir_str = format!("{}", mir);
    // Operator section (+) applied to arguments should flatten correctly
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_add"));
}

#[test]
fn test_operator_section_with_map() {
    // Test operator section passed to map (special higher-order function)
    let mir = flatten_program(
        r#"
def test_map(arr: [3]i32) -> [3]i32 = map((|x: i32| (+)(x, 1)), arr)
"#,
    );
    let mir_str = format!("{}", mir);
    // Should successfully flatten with lambda wrapping operator section
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_map"));
}

#[test]
fn test_mul_overloads() {
    // Test all three overloaded versions of mul:
    // - mul_mat_mat: mat * mat -> mat
    // - mul_mat_vec: mat * vec -> vec
    // - mul_vec_mat: vec * mat -> vec
    // Note: Square matrices use mat4f32 syntax (not mat4x4f32)
    let mir = flatten_program(
        r#"
def test_mul_overloads(m1: mat4f32, m2: mat4f32, v: vec4f32) -> vec4f32 =
    let mat_result = mul(m1, m2) in          -- mul_mat_mat
    let vec_result1 = mul(mat_result, v) in  -- mul_mat_vec
    let vec_result2 = mul(v, m1) in          -- mul_vec_mat
    vec_result1
"#,
    );
    // Check for mul variant calls in the MIR
    let f_def = find_def(&mir, "test_mul_overloads");
    let body = get_body(f_def);
    let call_names: Vec<_> = body
        .exprs
        .iter()
        .filter_map(
            |e| {
                if let mir::Expr::Call { func, .. } = e { Some(func.as_str()) } else { None }
            },
        )
        .collect();
    // All three should be desugared to their specific variants
    assert!(
        call_names.iter().any(|n| n.contains("mul_mat_mat")),
        "Expected mul_mat_mat in MIR"
    );
    assert!(
        call_names.iter().any(|n| n.contains("mul_mat_vec")),
        "Expected mul_mat_vec in MIR"
    );
    assert!(
        call_names.iter().any(|n| n.contains("mul_vec_mat")),
        "Expected mul_vec_mat in MIR"
    );
}

#[test]
fn test_no_redundant_materializations_simple_var() {
    // Test that accessing the same array with dynamic index in multiple branches.
    // NOTE: Full CSE across let-bindings is not implemented, so we allow 2 materializations
    // here (one for the outer let, one for the if-then branch). The hoisting pass only
    // deduplicates materializations that appear in BOTH branches of an if.
    let mir = flatten_program(
        r#"
def test(arr: [3]i32, i: i32) -> i32 =
  let x = arr[i] in
  if true then arr[i] else x
"#,
    );
    let mir_str = format!("{}", mir);
    let test_fn = extract_function_mir(&mir_str, "test");
    println!("MIR output for test:\n{}", test_fn);

    // Allow up to 2 materializations (CSE across let-bindings not yet implemented)
    let materialize_count = test_fn.matches("@materialize").count();
    assert!(
        materialize_count <= 2,
        "Expected at most 2 materializations, found {}. MIR:\n{}",
        materialize_count,
        test_fn
    );
}

#[test]
fn test_no_redundant_materializations_complex_expr() {
    // Test that indexing a complex expression (not a simple variable) in both
    // branches of an if does not create TWO separate materializations.
    // The materialize_hoisting pass should hoist the common materialization.
    let mir = flatten_program(
        r#"
def identity(arr: [3]i32) -> [3]i32 = arr
def test(arr: [3]i32, i: i32) -> i32 =
  if true then (identity(arr))[i] else (identity(arr))[i]
"#,
    );

    let mir_str = format!("{}", mir);
    let test_fn = extract_function_mir(&mir_str, "test");
    println!("MIR output for test:\n{}", test_fn);

    // After hoisting, there should be at most 1 materialize
    let materialize_count = test_fn.matches("@materialize").count();
    assert!(
        materialize_count <= 1,
        "Expected at most 1 materialize, found {}. MIR:\n{}",
        materialize_count,
        test_fn
    );
}

#[test]
fn test_no_materialize_for_loop_tuple_destructuring() {
    // Loop tuple destructuring should use tuple_access directly, not materialize
    let mir = flatten_program(
        r#"
def test(arr: [10]i32) -> i32 =
  let (sum, _) = loop (acc, i) = (0, 0) while i < 10 do
    (acc + arr[i], i + 1)
  in sum
"#,
    );

    let mir_str = format!("{}", mir);
    let test_fn = extract_function_mir(&mir_str, "test");
    println!("MIR for test:\n{}", test_fn);

    // The loop tuple pattern (acc, i) should NOT require materialize
    // Only the array index arr[i] might need it
    let materialize_count = test_fn.matches("@materialize").count();
    assert!(
        materialize_count <= 1,
        "Loop tuple destructuring should not use materialize. Found {} materializations. MIR:\n{}",
        materialize_count,
        test_fn
    );
}

#[test]
fn test_no_materialize_for_let_tuple_destructuring() {
    // Let tuple destructuring should use tuple_access directly, not materialize
    let mir = flatten_program(
        r#"
def test(x: i32) -> i32 =
  let pair = (x, x + 1) in
  let (a, b) = pair in
  a + b
"#,
    );

    let mir_str = format!("{}", mir);
    let test_fn = extract_function_mir(&mir_str, "test");
    println!("MIR for test:\n{}", test_fn);

    // Tuple destructuring should NOT require materialize at all
    let materialize_count = test_fn.matches("@materialize").count();
    assert_eq!(
        materialize_count, 0,
        "Let tuple destructuring should not use materialize. Found {} materializations. MIR:\n{}",
        materialize_count, test_fn
    );
}

// =============================================================================
// Currying and partial application tests
// =============================================================================

#[test]
fn test_curried_function_definition() {
    // Curried function with parenthesized typed parameters
    let mir = flatten_program(
        r#"
def add(x: f32, y: f32) -> f32 = x + y
def test: f32 = add(1.0f32, 2.0f32)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR:\n{}", mir_str);
    assert!(mir_str.contains("def add"));
    assert!(mir_str.contains("def test"));
}

#[test]
fn test_curried_function_application() {
    // Curried function application with multiple arguments
    let mir = flatten_program(
        r#"
def add(x: f32, y: f32) -> f32 = x + y
def test: f32 =
    let a = 1.0f32 in
    let b = 2.0f32 in
    add(a, b)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR:\n{}", mir_str);
    assert!(mir_str.contains("add"));
}

#[test]
fn test_higher_order_function_parameter() {
    // Function parameter with arrow type
    let mir = flatten_program(
        r#"
def apply(f: f32 -> f32, x: f32) -> f32 = f(x)
def double(x: f32) -> f32 = x + x
def test: f32 = apply(double, 3.0f32)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR:\n{}", mir_str);
    assert!(mir_str.contains("def apply"));
    assert!(mir_str.contains("def double"));
}

#[test]
fn test_binary_function_parameter() {
    // Binary function parameter (f32 -> f32 -> f32)
    let mir = flatten_program(
        r#"
def fold2(op: f32 -> f32 -> f32, x: f32, y: f32) -> f32 = op(x, y)
def add(a: f32, b: f32) -> f32 = a + b
def test: f32 = fold2(add, 1.0f32, 2.0f32)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR:\n{}", mir_str);
    assert!(mir_str.contains("def fold2"));
}

#[test]
fn test_reduce_pattern() {
    // The reduce pattern with curried binary operator
    let mir = flatten_program(
        r#"
def reduce_f32(op: f32 -> f32 -> f32, init: f32, arr: [4]f32) -> f32 =
    let x0 = op(init, arr[0]) in
    let x1 = op(x0, arr[1]) in
    let x2 = op(x1, arr[2]) in
    op(x2, arr[3])

def test: f32 =
    let arr = [1.0f32, 2.0f32, 3.0f32, 4.0f32] in
    reduce_f32((|x, y| x + y), 0.0f32, arr)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR:\n{}", mir_str);
    assert!(mir_str.contains("def reduce_f32"));
}

#[test]
fn test_map_lambda_without_extra_parens() {
    // Reproduce issue: map with lambda without extra parentheses around the lambda
    // This matches the pattern: map(|v:vec3f32| ..., arr)
    // vs the working pattern: map((|v:vec3f32| ...), arr)
    let mir = flatten_program(
        r#"
def test_map_no_parens(arr: [4]i32) -> [4]i32 =
    map(|x: i32| x + 1, arr)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_map_no_parens"));
}

#[test]
fn test_map_lambda_with_capture() {
    // Reproduce issue: map with lambda that captures variable (like da_rasterizer)
    // Pattern: map(|v:type| func(v, captured_var), arr)
    let mir = flatten_program(
        r#"
def test_map_capture(arr: [4]i32, offset: i32) -> [4]i32 =
    map(|x: i32| x + offset, arr)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_map_capture"));
}

#[test]
fn test_generic_function_dot2() {
    // Reproduce issue from primitives.wyn: generic function dot2<E, T>
    let mir = flatten_program(
        r#"
def dot2<E, T>(v: T) -> E = dot(v, v)

def test_dot2(p: vec3f32) -> f32 =
    dot2(p)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_dot2"));
}

#[test]
fn test_map_with_mul_capture() {
    // Reproduce da_rasterizer issue: map with lambda capturing a mat4f32 and calling mul
    let mir = flatten_program(
        r#"
def cube_corners: [8]vec3f32 =
    [ @[-1.0, -1.0, 1.0],
      @[-1.0, 1.0, 1.0],
      @[1.0, 1.0, 1.0],
      @[1.0, -1.0, 1.0],
      @[-1.0, -1.0, -1.0],
      @[-1.0, 1.0, -1.0],
      @[1.0, 1.0, -1.0],
      @[1.0, -1.0, -1.0] ]

def test_map_mul(mat: mat4f32) -> [8]vec4f32 =
    map(|v:vec3f32| mul(@[v.x, v.y, v.z, 1.0], mat), cube_corners)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_map_mul"));
}

#[test]
fn test_multiline_map_lambda() {
    // Reproduce da_rasterizer pattern: multiline lambda with let-in inside map
    let mir = flatten_program(
        r#"
def arr: [4]f32 = [1.0, 2.0, 3.0, 4.0]

def test_multiline: [4]f32 =
    map(|q:f32|
              let zinv = 1.0 / q in
              zinv,
           arr)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_multiline"));
}

#[test]
fn test_loop_with_map() {
    // Reproduce da_rasterizer pattern: map inside a loop
    let mir = flatten_program(
        r#"
def cube: [4]f32 = [1.0, 2.0, 3.0, 4.0]

def test_loop_map: f32 =
    let (_, acc) = loop (idx, acc) = (0i32, 0.0f32) while idx < 4 do
      let mat = 2.0f32 in
      let v4s = map(|v:f32| v * mat, cube) in
      (idx + 1, acc + v4s[0])
    in acc
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_loop_map"));
}

#[test]
fn test_da_rasterizer_minimal() {
    // Minimal reproduction of da_rasterizer.wyn structure
    let mir = flatten_program(
        r#"
#[uniform(set=1, binding=0)] def iResolution: vec2f32
#[uniform(set=1, binding=1)] def iTime: f32

def verts: [3]vec4f32 =
  [@[-1.0, -1.0, 0.0, 1.0],
   @[3.0, -1.0, 0.0, 1.0],
   @[-1.0, 3.0, 0.0, 1.0]]

#[vertex]
def vertex_main(#[builtin(vertex_index)] vertex_id:i32) -> #[builtin(position)] vec4f32 = verts[vertex_id]

def translation(p: vec3f32) -> mat4f32 =
  @[
    [1.0f32, 0.0f32, 0.0f32, p.x],
    [0.0f32, 1.0f32, 0.0f32, p.y],
    [0.0f32, 0.0f32, 1.0f32, p.z],
    [0.0f32, 0.0f32, 0.0f32, 1.0f32]
  ]

def rotation_euler(a: vec3f32) -> mat4f32 =
  let s = [f32.sin(a.x), f32.sin(a.y), f32.sin(a.z)] in
  let c = [f32.cos(a.x), f32.cos(a.y), f32.cos(a.z)] in
  @[
    [c[1]*c[2], c[1]*s[2], 0.0f32 - s[1], 0.0f32],
    [s[0]*s[1]*c[2] - c[0]*s[2], s[0]*s[1]*s[2] + c[0]*c[2], s[0]*c[1], 0.0f32],
    [c[0]*s[1]*c[2] + s[0]*s[2], c[0]*s[1]*s[2] - s[0]*c[2], c[0]*c[1], 0.0f32],
    [0.0f32, 0.0f32, 0.0f32, 1.0f32]
  ]

def cube_corners: [8]vec3f32 =
    [ @[-1.0, -1.0, 1.0],
      @[-1.0, 1.0, 1.0],
      @[1.0, 1.0, 1.0],
      @[1.0, -1.0, 1.0],
      @[-1.0, -1.0, -1.0],
      @[-1.0, 1.0, -1.0],
      @[1.0, 1.0, -1.0],
      @[1.0, -1.0, -1.0] ]

def main_image(iResolution:vec2f32, iTime:f32, fragCoord:vec2f32) -> vec4f32 =
  let cam = translation(@[0.0, 0.0, 10.0]) in
  let rot = rotation_euler(@[iTime, iTime*0.86, iTime*0.473]) in
  let rot_cam : mat4f32 = mul(rot, cam) in
  let mat  : mat4f32 = mul(translation(@[4.0, 4.0, -4.0]), rot_cam) in
  let v4s : [8]vec4f32 = map(|v:vec3f32| mul(@[v.x, v.y, v.z, 1.0], mat), cube_corners) in
  v4s[0]

#[fragment]
def fragment_main(#[builtin(frag_coord)] pos:vec4f32) -> #[location(0)] vec4f32 =
  main_image(@[iResolution.x, iResolution.y], iTime, @[pos.x, pos.y])
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("fragment_main"));
}

#[test]
fn test_reduce_with_lambda() {
    // Test that reduce with a lambda generates correct MIR
    let mir = flatten_program(
        r#"
def sum_array(arr: [4]f32) -> f32 =
    reduce((|acc: f32, x: f32| acc + x), 0.0, arr)
"#,
    );

    // Should have the sum_array function and a lambda function
    assert!(mir.defs.len() >= 2, "Expected sum_array function + lambda");

    // Lambda registry should have the lambda function for the binary operator
    assert_eq!(
        mir.lambda_registry.len(),
        1,
        "Expected 1 lambda in registry for reduce operator"
    );
    let (_, info) = mir.lambda_registry.iter().next().unwrap();
    assert_eq!(info.arity, 2, "reduce operator lambda should have arity 2");

    // Check that reduce call exists with closure
    let mut has_reduce_call = false;
    for def in &mir.defs {
        if let mir::Def::Function { body, name, .. } = def {
            if name == "sum_array" {
                for expr in &body.exprs {
                    if let mir::Expr::Call { func, args } = expr {
                        if func == "reduce" && args.len() == 3 {
                            has_reduce_call = true;
                        }
                    }
                }
            }
        }
    }
    assert!(has_reduce_call, "Expected reduce call with 3 arguments");
}

#[test]
fn test_reduce_product() {
    // Test reduce for computing product
    let mir = flatten_program(
        r#"
def product_array(arr: [4]f32) -> f32 =
    reduce((|acc: f32, x: f32| acc * x), 1.0, arr)
"#,
    );

    // Should compile successfully
    let sum_def = find_def(&mir, "product_array");
    assert!(matches!(sum_def, mir::Def::Function { .. }));
}

#[test]
fn test_filter_basic() {
    // Test basic filter with a predicate
    let mir = flatten_program(
        r#"
def is_positive(x: i32) -> bool = x > 0

def filter_positive(arr: [5]i32) -> ?k. [k]i32 =
    filter(is_positive, arr)
"#,
    );

    // Should have the is_positive and filter_positive functions
    assert!(mir.defs.len() >= 2, "Expected at least 2 functions");
    let filter_def = find_def(&mir, "filter_positive");
    assert!(matches!(filter_def, mir::Def::Function { .. }));
}

#[test]
fn test_filter_with_lambda() {
    // Test filter with an inline lambda predicate
    let mir = flatten_program(
        r#"
def filter_evens(arr: [4]i32) -> ?k. [k]i32 =
    filter((|x: i32| x % 2 == 0), arr)
"#,
    );

    // Should have the filter_evens function and a lambda
    let filter_def = find_def(&mir, "filter_evens");
    assert!(matches!(filter_def, mir::Def::Function { .. }));
    assert!(
        !mir.lambda_registry.is_empty(),
        "Expected lambda in registry for filter predicate"
    );
}

#[test]
fn test_filter_length() {
    // Test that length() can be called on filter result
    let mir = flatten_program(
        r#"
def is_positive(x: i32) -> bool = x > 0

def count_positive(arr: [5]i32) -> i32 =
    let filtered = filter(is_positive, arr) in
    length(filtered)
"#,
    );

    let count_def = find_def(&mir, "count_positive");
    assert!(matches!(count_def, mir::Def::Function { .. }));
}

#[test]
fn test_for_in_loop_array() {
    // Test for-in loop over a static array
    let mir = flatten_program(
        r#"
def sum_arr(arr: [5]i32) -> i32 =
    loop acc = 0 for x in arr do acc + x
"#,
    );

    let sum_def = find_def(&mir, "sum_arr");
    assert!(matches!(sum_def, mir::Def::Function { .. }));
}

#[test]
fn test_for_in_loop_with_filter() {
    // Test for-in loop over a filter result (slice)
    let mir = flatten_program(
        r#"
def is_positive(x: i32) -> bool = x > 0

def sum_positive(arr: [5]i32) -> i32 =
    let filtered = filter(is_positive, arr) in
    loop acc = 0 for x in filtered do acc + x
"#,
    );

    let sum_def = find_def(&mir, "sum_positive");
    assert!(matches!(sum_def, mir::Def::Function { .. }));
}

// =============================================================================
// GLSL output tests - verify types are preserved through the full pipeline
// =============================================================================

/// Compile source to Shadertoy GLSL through the full pipeline
fn compile_to_glsl(input: &str) -> String {
    let mut frontend = crate::cached_frontend();
    let parsed = crate::Compiler::parse(input, &mut frontend.node_counter).expect("Parsing failed");
    let glsl = parsed
        .desugar(&mut frontend.node_counter)
        .expect("Desugaring failed")
        .resolve(&frontend.module_manager)
        .expect("Name resolution failed")
        .fold_ast_constants()
        .type_check(&frontend.module_manager, &mut frontend.schemes)
        .expect("Type checking failed")
        .alias_check()
        .expect("Alias checking failed")
        .flatten(&frontend.module_manager, &frontend.schemes)
        .expect("Flattening failed")
        .0
        .hoist_materializations()
        .normalize()
        .monomorphize()
        .expect("Monomorphization failed")
        .partial_eval()
        .expect("Partial eval failed")
        .filter_reachable()
        .lift_bindings()
        .lower_shadertoy()
        .expect("GLSL lowering failed");
    glsl
}

#[test]
fn test_glsl_vector_constructor_types() {
    // Verify that vector literals use vec2/vec3/vec4 constructors, not float()
    let glsl = compile_to_glsl(
        r#"
#[uniform(set=0, binding=0)] def iResolution: vec2f32
#[uniform(set=0, binding=1)] def iTime: f32

#[fragment]
def fragment_main(#[builtin(position)] pos: vec4f32) -> #[location(0)] vec4f32 =
    let v2 = @[pos.x, pos.y] in
    let v3 = @[pos.x, pos.y, pos.z] in
    @[v2.x, v2.y, v3.z, 1.0]
"#,
    );

    // Should use vec2() not float() for 2-element vectors
    assert!(glsl.contains("vec2("), "Expected vec2() constructor in GLSL");
    assert!(!glsl.contains("float("), "Should not have float() constructor for vectors");

    // Should use vec4() for output
    assert!(glsl.contains("vec4("), "Expected vec4() constructor in GLSL");
}

#[test]
fn test_glsl_let_bindings_preserved() {
    // Verify that let bindings are preserved (not inlined everywhere)
    let glsl = compile_to_glsl(
        r#"
#[uniform(set=0, binding=0)] def iResolution: vec2f32
#[uniform(set=0, binding=1)] def iTime: f32

#[fragment]
def fragment_main(#[builtin(position)] pos: vec4f32) -> #[location(0)] vec4f32 =
    let coord = @[pos.x, pos.y] in
    let uv = @[coord.x / iResolution.x, coord.y / iResolution.y] in
    @[uv.x, uv.y, 0.0, 1.0]
"#,
    );

    // Let bindings should create local variables
    assert!(glsl.contains("vec2 coord"), "Expected 'coord' local variable");
    assert!(glsl.contains("vec2 uv"), "Expected 'uv' local variable");

    // coord should only appear a few times (declaration + uses), not duplicated everywhere
    let coord_count = glsl.matches("coord").count();
    assert!(
        coord_count <= 6,
        "Expected coord to appear <= 6 times (not duplicated), found {} times",
        coord_count
    );
}

#[test]
fn test_glsl_normalization_variables() {
    // Verify that normalization creates _w_norm_ intermediate variables
    let glsl = compile_to_glsl(
        r#"
#[uniform(set=0, binding=0)] def iTime: f32

#[fragment]
def fragment_main(#[builtin(position)] pos: vec4f32) -> #[location(0)] vec4f32 =
    let x = pos.x + pos.y * iTime in
    @[x, x, x, 1.0]
"#,
    );

    // Normalization should create _w_norm_ variables for subexpressions
    assert!(
        glsl.contains("_w_norm_"),
        "Expected _w_norm_ normalization variables in GLSL"
    );
}
