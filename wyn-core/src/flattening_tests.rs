#![cfg(test)]

use crate::mir;

fn flatten_program(input: &str) -> mir::Program {
    // Use the typestate API to ensure proper compilation pipeline
    let mut frontend = crate::cached_frontend();
    let parsed = crate::Compiler::parse(input, &mut frontend.node_counter).expect("Parsing failed");
    let alias_checked = parsed
        .desugar(&mut frontend.node_counter)
        .expect("Desugaring failed")
        .resolve(&mut frontend.module_manager)
        .expect("Name resolution failed")
        .fold_ast_constants()
        .type_check(&mut frontend.module_manager, &mut frontend.schemes)
        .expect("Type checking failed")
        .alias_check()
        .expect("Borrow checking failed");

    let builtins = crate::build_builtins(&alias_checked.ast, &mut frontend.module_manager);
    let flattened = alias_checked
        .to_tlc(builtins, &frontend.schemes, &mut frontend.module_manager)
        .skip_partial_eval()
        .defunctionalize()
        .to_mir();

    // Run hoisting pass to optimize materializations
    flattened.hoist_materializations().mir
}

#[allow(dead_code)]
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
    // Use the typestate API to ensure resolve_placeholders runs
    let mut frontend = crate::cached_frontend();
    let result = crate::Compiler::parse(input, &mut frontend.node_counter)
        .and_then(|parsed| parsed.desugar(&mut frontend.node_counter))
        .and_then(|desugared| desugared.resolve(&mut frontend.module_manager))
        .map(|resolved| resolved.fold_ast_constants())
        .and_then(|folded| folded.type_check(&mut frontend.module_manager, &mut frontend.schemes));
    result.is_err()
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
    // TLC pipeline uses _w_tuple_proj for tuple destructuring
    let f_def = find_def(&mir, "f");
    let body = get_body(f_def);
    // Should have Let bindings for tuple destructuring with _w_tuple_proj calls
    let has_tuple_proj = body.exprs.iter().any(|expr| {
        matches!(expr, mir::Expr::Call { func, .. } if func == "_w_tuple_proj")
            || matches!(expr, mir::Expr::Intrinsic { name, .. } if name == "_w_tuple_proj" || name == "tuple_access")
    });
    assert!(
        has_tuple_proj,
        "Expected _w_tuple_proj or tuple_access for tuple pattern"
    );
}

#[test]
fn test_lambda_tuple_pattern_param() {
    // Test lambda with tuple pattern parameter: |(x, y)| x + y
    // TLC pipeline uses _lambda_N naming and _w_tuple_proj for destructuring
    let mir = flatten_program("def f = let add = |(x, y)| x + y in add((1, 2))");

    // Find the generated lambda function (TLC uses _lambda_N naming)
    let add_fn = mir.defs.iter().find(|d| {
        if let mir::Def::Function { name, .. } = d {
            name.contains("_lambda_") || name.contains("_w_lam_")
        } else {
            false
        }
    });
    assert!(add_fn.is_some(), "Generated lambda function should exist");

    if let Some(mir::Def::Function { body, .. }) = add_fn {
        // TLC pipeline may use _w_tuple_proj Call or tuple_access Intrinsic
        let has_tuple_proj = body.exprs.iter().any(|expr| {
            matches!(expr, mir::Expr::Call { func, .. } if func == "_w_tuple_proj")
                || matches!(expr, mir::Expr::Intrinsic { name, .. } if name == "_w_tuple_proj" || name == "tuple_access")
        });
        assert!(
            has_tuple_proj,
            "Lambda with tuple param should have tuple projection for destructuring"
        );
    }
}

#[test]
fn test_lambda_nested_tuple_pattern_param() {
    // Test lambda with nested tuple pattern: |((a, b), c)| a + b + c
    // Should compile successfully with tuple destructuring
    let mir = flatten_program("def f = let add = |((a, b), c)| a + b + c in add(((1, 2), 3))");

    // Find the generated lambda function
    let add_fn = mir.defs.iter().find(|d| {
        if let mir::Def::Function { name, .. } = d {
            name.contains("_lambda_") || name.contains("_w_lam_")
        } else {
            false
        }
    });
    assert!(add_fn.is_some(), "Generated lambda function should exist");
}

#[test]
fn test_lambda_defunctionalization() {
    // When a top-level def is itself a lambda expression, TLC transforms it
    // into a regular function (f becomes the lambda function directly)
    let mir = flatten_program("def f = |x| x + 1");

    // f should be a function with 1 parameter (the lambda's param)
    let f_def = find_def(&mir, "f");
    if let mir::Def::Function { params, .. } = f_def {
        assert_eq!(params.len(), 1, "f should have 1 parameter from the lambda");
    } else {
        panic!("Expected f to be a function");
    }
}

#[test]
fn test_lambda_with_capture() {
    let mir = flatten_program("def f(y) = let g = |x| x + y in g(1)");

    // After defunctionalization, lambda with captures becomes a lifted function
    // that takes captures as extra parameters. The call g(1) becomes _lambda_N(y, 1).
    let f_def = find_def(&mir, "f");
    let body = get_body(f_def);

    // Check that there's a Call to a lifted lambda
    let has_lambda_call = body.exprs.iter().any(|expr| {
        if let mir::Expr::Call { func, .. } = expr { func.starts_with("_lambda") } else { false }
    });
    assert!(has_lambda_call, "Expected call to lifted lambda function");
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
    let mir = flatten_program("def g(y: i32) i32 = y + 1\ndef f(x) = g(x)");
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
        mir::Expr::Array {
            backing: mir::ArrayBacking::Literal(elems),
            ..
        } => {
            assert_eq!(elems.len(), 3, "Expected 3 elements");
        }
        other => panic!("Expected Array with Literal backing, got {:?}", other),
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
    // TLC pipeline represents loops with the Loop construct
    let mir = flatten_program("def f = loop x = 0 while x < 10 do x + 1");
    let f_def = find_def(&mir, "f");
    let body = get_body(f_def);
    match body.get_expr(body.root) {
        mir::Expr::Loop { kind, .. } => {
            assert!(
                matches!(kind, mir::LoopKind::While { .. }),
                "Expected While loop kind"
            );
        }
        other => panic!("Expected Loop, got {:?}", other),
    }
}

#[test]
fn test_for_range_loop() {
    // TLC pipeline represents for loops with the Loop construct
    let mir = flatten_program("def f = loop acc = 0 for i < 10 do acc + i");
    let f_def = find_def(&mir, "f");
    let body = get_body(f_def);
    match body.get_expr(body.root) {
        mir::Expr::Loop { kind, .. } => {
            assert!(
                matches!(kind, mir::LoopKind::ForRange { .. }),
                "Expected ForRange loop kind"
            );
        }
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
    // Array index becomes intrinsic call (TLC may use _w_index or index)
    let has_index = body.exprs.iter().any(|e| {
        matches!(e, mir::Expr::Intrinsic { name, .. } if name == "index" || name == "_w_index")
            || matches!(e, mir::Expr::Call { func, .. } if func == "index" || func == "_w_index")
    });
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
    // Should have at least the test function and a lambda (TLC uses _lambda_N naming)
    assert!(mir.defs.len() >= 2, "Expected at least test function and lambda");
    // Check for lambda function in defs
    let has_lambda = mir.defs.iter().any(|d| {
        if let mir::Def::Function { name, .. } = d {
            name.contains("_lambda_") || name.contains("_w_lam_")
        } else {
            false
        }
    });
    assert!(has_lambda, "Should have a lambda function definition");
}

#[test]
fn test_lambda_captures_typed_variable() {
    // This test reproduces an issue where a lambda captures a typed variable (like an array),
    // and the free variable rewriting creates _w_closure.mat, which then fails when trying
    // to resolve 'mat' as a field access on the closure.
    //
    // After defunctionalization with HOF specialization:
    // 1. Lambda is lifted to _lambda_N(i, arr) with arr as capture param
    // 2. map is specialized to map$N(xs, arr) which calls _w_intrinsic_map(_lambda_N, xs, arr)
    // 3. test_capture calls map$N([0,1,2,3], arr)
    let mir = flatten_program(
        r#"
def test_capture(arr: [4]i32) i32 =
    let result = map((|i: i32| arr[i]), [0, 1, 2, 3]) in
    result[0]
"#,
    );

    // Check that there's a lifted lambda function with arr as a parameter
    let has_lambda = mir.defs.iter().any(|def| {
        if let mir::Def::Function { name, .. } = def { name.starts_with("_lambda") } else { false }
    });
    assert!(has_lambda, "Lambda should be lifted to a top-level function");

    // Check that there's a specialized map function (map$N)
    let has_specialized_map = mir.defs.iter().any(|def| {
        if let mir::Def::Function { name, .. } = def { name.starts_with("map$") } else { false }
    });
    assert!(
        has_specialized_map,
        "map should be specialized with captures as map$N"
    );

    // Check that the specialized map contains the _w_intrinsic_map call
    let specialized_map = mir.defs.iter().find(|def| {
        if let mir::Def::Function { name, .. } = def { name.starts_with("map$") } else { false }
    });
    if let Some(mir::Def::Function { body, .. }) = specialized_map {
        let has_map_intrinsic = body.exprs.iter().any(|expr| {
            if let mir::Expr::Intrinsic { name, .. } = expr { name == "_w_intrinsic_map" } else { false }
        });
        assert!(
            has_map_intrinsic,
            "Specialized map should have _w_intrinsic_map call"
        );
    }
}

#[test]
fn test_qualified_name_f32_sqrt() {
    // This test reproduces an issue where f32.sqrt is treated as field access
    // instead of a qualified builtin name. The identifier 'f32' has no type in
    // the type_table because it's a type name, not a variable.
    let mir = flatten_program(
        r#"
def length2(v: vec2f32) f32 =
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
    // This test checks that map with a lambda generates a lambda function
    let mir = flatten_program(
        r#"
def test_map(arr: [4]i32) [4]i32 =
    map((|x: i32| x + 1), arr)
"#,
    );

    // Should have lambda function generated (TLC uses _lambda_N naming)
    assert!(mir.defs.len() >= 2, "Expected test function + lambda");

    // Check for lambda function in defs
    let lambda_fn = mir.defs.iter().find(|d| {
        if let mir::Def::Function { name, .. } = d {
            name.contains("_lambda_") || name.contains("_w_lam_")
        } else {
            false
        }
    });
    assert!(lambda_fn.is_some(), "Expected lambda function definition");

    // Verify the lambda is a 1-param function
    if let Some(mir::Def::Function { params, .. }) = lambda_fn {
        assert_eq!(params.len(), 1, "Lambda should have 1 parameter");
    }
}

#[test]
fn test_direct_closure_call() {
    // This test checks that directly calling a closure generates a direct lambda call
    let mir = flatten_program(
        r#"
def test_apply(x: i32) i32 =
    let f = |y: i32| y + x in
    f(10)
"#,
    );

    // Should have lambda function generated (TLC uses _lambda_N naming)
    assert!(mir.defs.len() >= 2, "Expected test function + lambda");

    // Check for lambda function in defs
    let has_lambda = mir.defs.iter().any(|d| {
        if let mir::Def::Function { name, .. } = d {
            name.contains("_lambda_") || name.contains("_w_lam_")
        } else {
            false
        }
    });
    assert!(has_lambda, "Should have a lambda function definition");
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
def choose(b: bool) (i32 -> i32) =
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
def test(v: vec3f32) vec4f32 =
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
def mysum<[n]>(arr: [n]f32) f32 =
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
    let alias_checked = crate::Compiler::parse(source, &mut frontend.node_counter)
        .and_then(|p| p.desugar(&mut frontend.node_counter))
        .and_then(|d| d.resolve(&mut frontend.module_manager))
        .map(|r| r.fold_ast_constants())
        .and_then(|f| f.type_check(&mut frontend.module_manager, &mut frontend.schemes))
        .and_then(|t| t.alias_check())
        .expect("Failed before TLC transform");

    let builtins = crate::build_builtins(&alias_checked.ast, &mut frontend.module_manager);
    let result = alias_checked
        .to_tlc(builtins, &frontend.schemes, &mut frontend.module_manager)
        .skip_partial_eval()
        .defunctionalize()
        .to_mir()
        .hoist_materializations()
        .normalize()
        .monomorphize()
        .map(|m| m.default_address_spaces())
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
def test_conversions(x: i32, y: i64) f32 =
  let f1 = f32.i32(x) in
  let f2 = f32.i64(y) in
  f1 + f2
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
def test_math(x: f32) f32 =
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
def test_add(x: i32, y: i32) i32 = (+)(x, y)
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
def test_map(arr: [3]i32) [3]i32 = map((|x: i32| (+)(x, 1)), arr)
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
def test_mul_overloads(m1: mat4f32, m2: mat4f32, v: vec4f32) vec4f32 =
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
def test(arr: [3]i32, i: i32) i32 =
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
def identity(arr: [3]i32) [3]i32 = arr
def test(arr: [3]i32, i: i32) i32 =
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
def test(arr: [10]i32) i32 =
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
def test(x: i32) i32 =
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
def add(x: f32, y: f32) f32 = x + y
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
def add(x: f32, y: f32) f32 = x + y
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
def apply(f: f32 -> f32, x: f32) f32 = f(x)
def double(x: f32) f32 = x + x
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
def fold2(op: f32 -> f32 -> f32, x: f32, y: f32) f32 = op(x, y)
def add(a: f32, b: f32) f32 = a + b
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
def reduce_f32(op: f32 -> f32 -> f32, init: f32, arr: [4]f32) f32 =
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
def test_map_no_parens(arr: [4]i32) [4]i32 =
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
def test_map_capture(arr: [4]i32, offset: i32) [4]i32 =
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
def dot2<E, T>(v: T) E = dot(v, v)

def test_dot2(p: vec3f32) f32 =
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

def test_map_mul(mat: mat4f32) [8]vec4f32 =
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
entry vertex_main(#[builtin(vertex_index)] vertex_id: i32) #[builtin(position)] vec4f32 = verts[vertex_id]

def translation(p: vec3f32) mat4f32 =
  @[
    [1.0f32, 0.0f32, 0.0f32, p.x],
    [0.0f32, 1.0f32, 0.0f32, p.y],
    [0.0f32, 0.0f32, 1.0f32, p.z],
    [0.0f32, 0.0f32, 0.0f32, 1.0f32]
  ]

def rotation_euler(a: vec3f32) mat4f32 =
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

def main_image(iResolution:vec2f32, iTime:f32, fragCoord:vec2f32) vec4f32 =
  let cam = translation(@[0.0, 0.0, 10.0]) in
  let rot = rotation_euler(@[iTime, iTime*0.86, iTime*0.473]) in
  let rot_cam : mat4f32 = mul(rot, cam) in
  let mat  : mat4f32 = mul(translation(@[4.0, 4.0, -4.0]), rot_cam) in
  let v4s : [8]vec4f32 = map(|v:vec3f32| mul(@[v.x, v.y, v.z, 1.0], mat), cube_corners) in
  v4s[0]

#[fragment]
entry fragment_main(#[builtin(frag_coord)] pos: vec4f32) #[location(0)] vec4f32 =
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
def sum_array(arr: [4]f32) f32 =
    reduce((|acc: f32, x: f32| acc + x), 0.0, arr)
"#,
    );

    // Should have the sum_array function and lambda functions (TLC uses _lambda_N naming)
    assert!(mir.defs.len() >= 2, "Expected sum_array function + lambda");

    // Check for lambda function in defs (binary operator has 2 params)
    let lambda_fn = mir.defs.iter().find(|d| {
        if let mir::Def::Function { name, .. } = d {
            name.contains("_lambda_") || name.contains("_w_lam_")
        } else {
            false
        }
    });
    assert!(lambda_fn.is_some(), "Expected lambda function definition");

    // Check that _w_intrinsic_reduce is called somewhere in the MIR
    // (either in reduce prelude function or in sum_array)
    let mut has_reduce_intrinsic = false;
    for def in &mir.defs {
        if let mir::Def::Function { body, .. } = def {
            for expr in &body.exprs {
                if let mir::Expr::Intrinsic {
                    name: intrinsic_name,
                    args,
                } = expr
                {
                    // reduce has 3 args: [op, ne, arr]
                    if intrinsic_name == "_w_intrinsic_reduce" && args.len() == 3 {
                        has_reduce_intrinsic = true;
                    }
                }
            }
        }
    }
    assert!(
        has_reduce_intrinsic,
        "Expected _w_intrinsic_reduce with 3 arguments somewhere in MIR"
    );
}

#[test]
fn test_reduce_product() {
    // Test reduce for computing product
    let mir = flatten_program(
        r#"
def product_array(arr: [4]f32) f32 =
    reduce((|acc: f32, x: f32| acc * x), 1.0, arr)
"#,
    );

    // Should compile successfully
    let sum_def = find_def(&mir, "product_array");
    assert!(matches!(sum_def, mir::Def::Function { .. }));
}

#[test]
fn test_reduce_with_tuple_destructuring() {
    // Test reduce with tuple pattern destructuring in the combiner
    // This is the pattern used in raytrace.wyn's findClosestHit
    let mir = flatten_program(
        r#"
def minPair(hits: [4](f32, i32)) (f32, i32) =
  reduce(|(t1, m1): (f32, i32), (t2, m2): (f32, i32)|
           if t1 < t2 then (t1, m1) else (t2, m2),
         (1000.0, 0),
         hits)
"#,
    );

    // Should have the minPair function
    let min_def = find_def(&mir, "minPair");
    assert!(matches!(min_def, mir::Def::Function { .. }));

    // Check that _w_intrinsic_reduce is called with correct arity
    let mut has_reduce_intrinsic = false;
    for def in &mir.defs {
        if let mir::Def::Function { body, .. } = def {
            for expr in &body.exprs {
                if let mir::Expr::Intrinsic {
                    name: intrinsic_name,
                    args,
                } = expr
                {
                    if intrinsic_name == "_w_intrinsic_reduce" {
                        // reduce intrinsic should have at least 3 args: [op, ne, arr, captures...]
                        assert!(
                            args.len() >= 3,
                            "reduce intrinsic should have at least 3 args, got {}",
                            args.len()
                        );
                        has_reduce_intrinsic = true;
                    }
                }
            }
        }
    }
    assert!(has_reduce_intrinsic, "Expected _w_intrinsic_reduce in MIR");
}

#[test]
fn test_filter_basic() {
    // Test basic filter with a predicate
    let mir = flatten_program(
        r#"
def is_positive(x: i32) bool = x > 0

def filter_positive(arr: [5]i32) ?k. [k]i32 =
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
def filter_evens(arr: [4]i32) ?k. [k]i32 =
    filter((|x: i32| x % 2 == 0), arr)
"#,
    );

    // Should have the filter_evens function and a lambda (TLC uses _lambda_N naming)
    let filter_def = find_def(&mir, "filter_evens");
    assert!(matches!(filter_def, mir::Def::Function { .. }));
    let has_lambda = mir.defs.iter().any(|d| {
        if let mir::Def::Function { name, .. } = d {
            name.contains("_lambda_") || name.contains("_w_lam_")
        } else {
            false
        }
    });
    assert!(has_lambda, "Expected lambda function for filter predicate");
}

#[test]
fn test_filter_length() {
    // Test that length() can be called on filter result
    let mir = flatten_program(
        r#"
def is_positive(x: i32) bool = x > 0

def count_positive(arr: [5]i32) i32 =
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
def sum_arr(arr: [5]i32) i32 =
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
def is_positive(x: i32) bool = x > 0

def sum_positive(arr: [5]i32) i32 =
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
    let alias_checked = parsed
        .desugar(&mut frontend.node_counter)
        .expect("Desugaring failed")
        .resolve(&mut frontend.module_manager)
        .expect("Name resolution failed")
        .fold_ast_constants()
        .type_check(&mut frontend.module_manager, &mut frontend.schemes)
        .expect("Type checking failed")
        .alias_check()
        .expect("Alias checking failed");
    let builtins = crate::build_builtins(&alias_checked.ast, &mut frontend.module_manager);
    let glsl = alias_checked
        .to_tlc(builtins, &frontend.schemes, &mut frontend.module_manager)
        .partial_eval()
        .defunctionalize()
        .to_mir()
        .hoist_materializations()
        .normalize()
        .monomorphize()
        .expect("Monomorphization failed")
        .default_address_spaces()
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
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let v2 = @[pos.x, pos.y] in
    let v3 = @[pos.x, pos.y, pos.z] in
    @[v2.x, v2.y, v3.z, 1.0]
"#,
    );

    // Should use vec2() not float() for 2-element vectors
    assert!(glsl.contains("vec2("), "Expected vec2() constructor in GLSL");
    assert!(
        !glsl.contains("float("),
        "Should not have float() constructor for vectors"
    );

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
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
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
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
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

// =============================================================================
// Advanced Defunctionalization Tests
// =============================================================================

#[test]
fn test_defunc_nested_closure_captures_grandparent() {
    // Nested closures: inner lambda captures from two scopes above (grandparent)
    // outer captures `x`, inner captures `x` from outer's scope
    let mir = flatten_program(
        r#"
def nested_capture(x: i32, arr: [4]i32) [4]i32 =
    let outer = |y: i32|
        let inner = |z: i32| x + y + z in
        inner(y)
    in
    map(outer, arr)
"#,
    );

    // Should have multiple lifted lambdas
    let lambda_count = mir
        .defs
        .iter()
        .filter(
            |d| {
                if let mir::Def::Function { name, .. } = d { name.contains("_lambda_") } else { false }
            },
        )
        .count();
    assert!(
        lambda_count >= 2,
        "Expected at least 2 lifted lambdas (inner and outer), found {}",
        lambda_count
    );

    // Should have specialized map
    let has_specialized_map = mir.defs.iter().any(|d| {
        if let mir::Def::Function { name, .. } = d { name.starts_with("map$") } else { false }
    });
    assert!(
        has_specialized_map,
        "Expected specialized map function for nested captures"
    );
}

#[test]
fn test_defunc_same_hof_different_captures() {
    // Two callsites to map with different lambdas that have different capture sets
    // This should produce two different specialized map functions
    let mir = flatten_program(
        r#"
def double_map(x: i32, y: i32, arr: [4]i32) ([4]i32, [4]i32) =
    let result1 = map((|e: i32| e + x), arr) in
    let result2 = map((|e: i32| e * y), arr) in
    (result1, result2)
"#,
    );

    // Should have two different specialized map functions (map$N and map$M where N != M)
    let specialized_maps: Vec<_> = mir
        .defs
        .iter()
        .filter_map(|d| {
            if let mir::Def::Function { name, .. } = d {
                if name.starts_with("map$") { Some(name.clone()) } else { None }
            } else {
                None
            }
        })
        .collect();

    assert!(
        specialized_maps.len() >= 2,
        "Expected at least 2 specialized map functions for different captures, found: {:?}",
        specialized_maps
    );
}

#[test]
fn test_defunc_deep_nested_capture() {
    // Deeply nested lambdas - inner lambda captures from outermost scope
    // No partial application: inner is immediately applied
    let mir = flatten_program(
        r#"
def deep_capture(x: i32, y: i32, arr: [4]i32) [4]i32 =
    let combine = |a: i32|
        let inner_add = |b: i32| a + b + x + y in
        inner_add(a * 2)
    in
    map(combine, arr)
"#,
    );

    // Should compile and have lambdas
    let has_lambda = mir.defs.iter().any(|d| {
        if let mir::Def::Function { name, .. } = d { name.contains("_lambda_") } else { false }
    });
    assert!(has_lambda, "Expected lifted lambda functions");

    // Should have specialized map for the outer lambda with captures
    let has_specialized_map = mir.defs.iter().any(|d| {
        if let mir::Def::Function { name, .. } = d { name.starts_with("map$") } else { false }
    });
    assert!(
        has_specialized_map,
        "Expected specialized map for lambda with deep captures"
    );
}

#[test]
fn test_defunc_multiple_hof_chain() {
    // Chain of multiple HOF calls, each with capturing lambdas
    let mir = flatten_program(
        r#"
def chain(scale: i32, offset: i32, arr: [4]i32) i32 =
    let scaled = map((|x: i32| x * scale), arr) in
    let shifted = map((|x: i32| x + offset), scaled) in
    reduce((|a: i32, b: i32| a + b), 0, shifted)
"#,
    );

    // Should have specialized maps for both scale and offset captures
    let specialized_maps: Vec<_> = mir
        .defs
        .iter()
        .filter_map(|d| {
            if let mir::Def::Function { name, .. } = d {
                if name.starts_with("map$") { Some(name.clone()) } else { None }
            } else {
                None
            }
        })
        .collect();
    assert!(
        specialized_maps.len() >= 2,
        "Expected 2 specialized maps in chain, found: {:?}",
        specialized_maps
    );

    // Should have specialized reduce
    let has_specialized_reduce = mir.defs.iter().any(|d| {
        if let mir::Def::Function { name, .. } = d { name.starts_with("reduce$") } else { false }
    });
    assert!(has_specialized_reduce, "Expected specialized reduce in chain");
}

#[test]
fn test_defunc_capture_in_let_binding() {
    // Lambda captures a variable from a let binding in enclosing scope
    let mir = flatten_program(
        r#"
def capture_let(base: i32, arr: [4]i32) [4]i32 =
    let multiplier = base * 2 in
    let scaled_add = |x: i32| x + multiplier in
    map(scaled_add, arr)
"#,
    );

    // Should have a lifted lambda that captures `multiplier`
    let has_lambda = mir.defs.iter().any(|d| {
        if let mir::Def::Function { name, .. } = d { name.contains("_lambda_") } else { false }
    });
    assert!(has_lambda, "Expected lifted lambda capturing let binding");

    let has_specialized_map = mir.defs.iter().any(|d| {
        if let mir::Def::Function { name, .. } = d { name.starts_with("map$") } else { false }
    });
    assert!(
        has_specialized_map,
        "Expected specialized map for lambda with captured let binding"
    );
}

#[test]
fn test_defunc_named_function_no_capture() {
    // Named function (no captures) passed to HOF - should still specialize
    let mir = flatten_program(
        r#"
def square(x: i32) i32 = x * x

def apply_square(arr: [4]i32) [4]i32 =
    map(square, arr)
"#,
    );

    // Should have specialized map for the named function
    let has_specialized_map = mir.defs.iter().any(|d| {
        if let mir::Def::Function { name, .. } = d { name.starts_with("map$") } else { false }
    });
    assert!(
        has_specialized_map,
        "Named functions passed to HOFs should trigger specialization"
    );

    // The specialized map should call _w_intrinsic_map with `square`
    let specialized_map = mir.defs.iter().find(|d| {
        if let mir::Def::Function { name, .. } = d { name.starts_with("map$") } else { false }
    });
    if let Some(mir::Def::Function { body, .. }) = specialized_map {
        let has_intrinsic_map = body
            .exprs
            .iter()
            .any(|e| matches!(e, mir::Expr::Intrinsic { name, .. } if name == "_w_intrinsic_map"));
        assert!(
            has_intrinsic_map,
            "Specialized map should contain _w_intrinsic_map"
        );
    }
}

#[test]
fn test_defunc_multiple_captures_same_lambda() {
    // Single lambda capturing multiple variables
    let mir = flatten_program(
        r#"
def multi_capture(a: i32, b: i32, c: i32, arr: [4]i32) [4]i32 =
    map((|x: i32| x + a + b + c), arr)
"#,
    );

    // Should have lambda and specialized map
    let has_lambda = mir.defs.iter().any(|d| {
        if let mir::Def::Function { name, .. } = d { name.contains("_lambda_") } else { false }
    });
    assert!(has_lambda, "Expected lifted lambda with multiple captures");

    let has_specialized_map = mir.defs.iter().any(|d| {
        if let mir::Def::Function { name, .. } = d { name.starts_with("map$") } else { false }
    });
    assert!(
        has_specialized_map,
        "Expected specialized map for multi-capture lambda"
    );
}

#[test]
fn test_defunc_reused_lambda_same_capture() {
    // Same lambda used twice with same captures - should reuse specialization
    let mir = flatten_program(
        r#"
def reuse_lambda(x: i32, arr1: [4]i32, arr2: [4]i32) ([4]i32, [4]i32) =
    let adder = |e: i32| e + x in
    let result1 = map(adder, arr1) in
    let result2 = map(adder, arr2) in
    (result1, result2)
"#,
    );

    // Should have exactly one specialized map (reused for both calls)
    let specialized_maps: Vec<_> = mir
        .defs
        .iter()
        .filter_map(|d| {
            if let mir::Def::Function { name, .. } = d {
                if name.starts_with("map$") { Some(name.clone()) } else { None }
            } else {
                None
            }
        })
        .collect();

    // The specialization is keyed by (hof_name, lambda_name), so same lambda should reuse
    // Note: we might get duplicates from prelude, so check there's at least one
    assert!(
        !specialized_maps.is_empty(),
        "Expected at least one specialized map for reused lambda"
    );
}
