// =============================================================================
// Tests
// =============================================================================

use super::soac_analysis::*;
use crate::types::TypeName;
use polytype::Type;
use std::collections::HashMap;

#[test]
fn test_soac_kind_from_name() {
    assert_eq!(
        SoacKind::from_intrinsic_name("_w_intrinsic_map"),
        Some(SoacKind::Map)
    );
    assert_eq!(
        SoacKind::from_intrinsic_name("_w_intrinsic_inplace_map"),
        Some(SoacKind::InplaceMap)
    );
    assert_eq!(
        SoacKind::from_intrinsic_name("_w_intrinsic_reduce"),
        Some(SoacKind::Reduce)
    );
    assert_eq!(
        SoacKind::from_intrinsic_name("_w_intrinsic_scan"),
        Some(SoacKind::Scan)
    );
    assert_eq!(
        SoacKind::from_intrinsic_name("_w_intrinsic_filter"),
        Some(SoacKind::Filter)
    );
    assert_eq!(
        SoacKind::from_intrinsic_name("_w_intrinsic_scatter"),
        Some(SoacKind::Scatter)
    );
    assert_eq!(
        SoacKind::from_intrinsic_name("_w_intrinsic_hist_1d"),
        Some(SoacKind::Hist1D)
    );
    assert_eq!(
        SoacKind::from_intrinsic_name("_w_intrinsic_zip"),
        Some(SoacKind::Zip)
    );
    assert_eq!(SoacKind::from_intrinsic_name("some_other_fn"), None);
}

#[test]
fn test_soac_kind_is_independent() {
    assert!(SoacKind::Map.is_independent());
    assert!(SoacKind::InplaceMap.is_independent());
    assert!(SoacKind::Zip.is_independent());

    assert!(!SoacKind::Reduce.is_independent());
    assert!(!SoacKind::Scan.is_independent());
    assert!(!SoacKind::Filter.is_independent());
    assert!(!SoacKind::Scatter.is_independent());
    assert!(!SoacKind::Hist1D.is_independent());
}

#[test]
fn test_size_expr_display() {
    assert_eq!(SizeExpr::Const(42).to_string(), "42");
    assert_eq!(SizeExpr::Var("n".to_string()).to_string(), "n");
    assert_eq!(SizeExpr::Hint(1024).to_string(), "hint(1024)");
    assert_eq!(SizeExpr::Unknown.to_string(), "?");
    assert_eq!(
        SizeExpr::Product(
            Box::new(SizeExpr::Const(2)),
            Box::new(SizeExpr::Var("n".to_string()))
        )
        .to_string(),
        "(2 * n)"
    );
    assert_eq!(
        SizeExpr::BoundedBy(Box::new(SizeExpr::Const(100))).to_string(),
        "≤100"
    );
}

#[test]
fn test_extract_size_from_type() {
    // Array[f32, Function, Size(10)]
    let sized_array = Type::Constructed(
        TypeName::Array,
        vec![
            Type::Constructed(TypeName::Float(32), vec![]),
            Type::Constructed(TypeName::AddressFunction, vec![]),
            Type::Constructed(TypeName::Size(10), vec![]),
        ],
    );
    assert_eq!(extract_size_from_type(&sized_array), SizeExpr::Const(10));

    // Array[f32, Storage, ?size] - size is a type variable (determined by parallelization)
    let variable_size_array = Type::Constructed(
        TypeName::Array,
        vec![
            Type::Constructed(TypeName::Float(32), vec![]),
            Type::Constructed(TypeName::AddressStorage, vec![]),
            Type::Variable(42), // Type variable for size
        ],
    );
    assert_eq!(extract_size_from_type(&variable_size_array), SizeExpr::Unknown);

    // Array[f32, Storage, SizeVar("n")]
    let var_sized_array = Type::Constructed(
        TypeName::Array,
        vec![
            Type::Constructed(TypeName::Float(32), vec![]),
            Type::Constructed(TypeName::AddressStorage, vec![]),
            Type::Constructed(TypeName::SizeVar("n".to_string()), vec![]),
        ],
    );
    assert_eq!(
        extract_size_from_type(&var_sized_array),
        SizeExpr::Var("n".to_string())
    );

    // Non-array type
    let scalar = Type::Constructed(TypeName::Float(32), vec![]);
    assert_eq!(extract_size_from_type(&scalar), SizeExpr::Unknown);
}

#[test]
fn test_output_size_for_soac() {
    let analyzer = SoacAnalyzer::new(HashMap::new());
    let input = SizeExpr::Hint(1024);

    // Map preserves size
    assert_eq!(
        analyzer.output_size_for_soac(SoacKind::Map, &input),
        SizeExpr::Hint(1024)
    );

    // Reduce produces scalar
    assert_eq!(
        analyzer.output_size_for_soac(SoacKind::Reduce, &input),
        SizeExpr::Const(1)
    );

    // Filter is bounded
    assert_eq!(
        analyzer.output_size_for_soac(SoacKind::Filter, &input),
        SizeExpr::BoundedBy(Box::new(SizeExpr::Hint(1024)))
    );

    // Scan preserves size
    assert_eq!(
        analyzer.output_size_for_soac(SoacKind::Scan, &input),
        SizeExpr::Hint(1024)
    );
}

#[test]
fn test_soac_expected_args() {
    assert_eq!(SoacKind::Map.expected_args(), 2);
    assert_eq!(SoacKind::InplaceMap.expected_args(), 2);
    assert_eq!(SoacKind::Reduce.expected_args(), 3);
    assert_eq!(SoacKind::Scan.expected_args(), 3);
    assert_eq!(SoacKind::Filter.expected_args(), 2);
    assert_eq!(SoacKind::Scatter.expected_args(), 3);
    assert_eq!(SoacKind::Hist1D.expected_args(), 5);
    assert_eq!(SoacKind::Zip.expected_args(), 2);
}

// =============================================================================
// Helper functions for building test terms
// =============================================================================

use super::{Def, DefMeta, Term, TermId, TermKind};
use crate::ast::Span;

fn dummy_span() -> Span {
    Span {
        start_line: 1,
        start_col: 1,
        end_line: 1,
        end_col: 1,
    }
}

fn next_id(counter: &mut u32) -> TermId {
    let id = TermId(*counter);
    *counter += 1;
    id
}

fn mk_var(counter: &mut u32, name: &str, ty: Type<TypeName>) -> Term {
    Term {
        id: next_id(counter),
        ty,
        span: dummy_span(),
        kind: TermKind::Var(name.to_string()),
    }
}

fn mk_lam(counter: &mut u32, param: &str, param_ty: Type<TypeName>, body: Term) -> Term {
    let body_ty = body.ty.clone();
    let lam_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), body_ty]);
    Term {
        id: next_id(counter),
        ty: lam_ty,
        span: dummy_span(),
        kind: TermKind::Lam {
            param: param.to_string(),
            param_ty,
            body: Box::new(body),
        },
    }
}

fn mk_app(counter: &mut u32, func: Term, arg: Term, result_ty: Type<TypeName>) -> Term {
    Term {
        id: next_id(counter),
        ty: result_ty,
        span: dummy_span(),
        kind: TermKind::App {
            func: Box::new(func),
            arg: Box::new(arg),
        },
    }
}

fn mk_let(counter: &mut u32, name: &str, rhs: Term, body: Term) -> Term {
    let body_ty = body.ty.clone();
    let rhs_ty = rhs.ty.clone();
    Term {
        id: next_id(counter),
        ty: body_ty,
        span: dummy_span(),
        kind: TermKind::Let {
            name: name.to_string(),
            name_ty: rhs_ty,
            rhs: Box::new(rhs),
            body: Box::new(body),
        },
    }
}

fn mk_float_lit(counter: &mut u32, val: f32) -> Term {
    Term {
        id: next_id(counter),
        ty: Type::Constructed(TypeName::Float(32), vec![]),
        span: dummy_span(),
        kind: TermKind::FloatLit(val),
    }
}

fn f32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Float(32), vec![])
}

fn array_ty(elem: Type<TypeName>) -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            elem,
            Type::Constructed(TypeName::AddressStorage, vec![]),
            Type::Variable(99), // Unknown size
        ],
    )
}

fn arrow_ty(from: Type<TypeName>, to: Type<TypeName>) -> Type<TypeName> {
    Type::Constructed(TypeName::Arrow, vec![from, to])
}

/// Build a curried call: f(a1)(a2)...(an)
fn mk_curried_call(
    counter: &mut u32,
    func_name: &str,
    args: Vec<Term>,
    result_ty: Type<TypeName>,
) -> Term {
    if args.is_empty() {
        return mk_var(counter, func_name, result_ty);
    }

    // Build type for the function
    let mut ty = result_ty.clone();
    for arg in args.iter().rev() {
        ty = arrow_ty(arg.ty.clone(), ty);
    }

    let mut current = mk_var(counter, func_name, ty);
    let mut remaining_ty = result_ty.clone();

    // Build intermediate types
    let mut intermediate_tys: Vec<Type<TypeName>> = vec![result_ty.clone()];
    for arg in args.iter().rev().skip(1) {
        remaining_ty = arrow_ty(arg.ty.clone(), remaining_ty.clone());
        intermediate_tys.push(remaining_ty.clone());
    }
    intermediate_tys.reverse();

    for (i, arg) in args.into_iter().enumerate() {
        let app_ty = intermediate_tys.get(i).cloned().unwrap_or_else(|| result_ty.clone());
        current = mk_app(counter, current, arg, app_ty);
    }

    current
}

// =============================================================================
// Integration tests for hint propagation through call stacks
// =============================================================================

#[test]
fn test_hint_propagation_through_single_function_call() {
    // Scenario:
    // def process(arr: []f32) = map(|x| x * 2.0, arr)
    // entry main(#[size_hint(512)] data: []f32) = process(data)
    //
    // The map inside `process` should see hint(512) from the entry point.

    let mut counter = 0u32;

    // Build `process` function body: map(|x| x * 2.0, arr)
    // As TLC: Lam("arr", App(App(Var("map"), Lam("x", ...)), Var("arr")))
    let lambda_body = mk_var(&mut counter, "x", f32_ty()); // simplified: just return x
    let lambda = mk_lam(&mut counter, "x", f32_ty(), lambda_body);
    let arr_ref = mk_var(&mut counter, "arr", array_ty(f32_ty()));
    let map_call = mk_curried_call(
        &mut counter,
        "map",
        vec![lambda, arr_ref],
        array_ty(f32_ty()),
    );
    let process_body = mk_lam(&mut counter, "arr", array_ty(f32_ty()), map_call);

    let process_def = Def {
        name: "process".to_string(),
        ty: arrow_ty(array_ty(f32_ty()), array_ty(f32_ty())),
        body: process_body,
        meta: DefMeta::Function,
        arity: 1,
    };

    // Set up analyzer with process definition
    let all_defs: HashMap<&str, &Def> = [("process", &process_def)].into_iter().collect();
    let mut analyzer = SoacAnalyzer::new(all_defs);

    // Simulate entry point environment: data has hint(512)
    analyzer.env.insert("data".to_string(), SizeExpr::Hint(512));

    // Build the call: process(data)
    let data_ref = mk_var(&mut counter, "data", array_ty(f32_ty()));
    let call_to_process = mk_curried_call(
        &mut counter,
        "process",
        vec![data_ref],
        array_ty(f32_ty()),
    );

    // Analyze the call
    analyzer.analyze_term(&call_to_process);

    // Should find one map SOAC with hint(512)
    assert_eq!(analyzer.results.len(), 1);
    assert_eq!(analyzer.results[0].kind, SoacKind::Map);
    assert_eq!(analyzer.results[0].input_size, SizeExpr::Hint(512));
}

#[test]
fn test_hint_propagation_through_nested_call_stack() {
    // Scenario:
    // def inner(arr) = map(|x| x, arr)
    // def outer(arr) = inner(arr)
    // entry main(#[size_hint(256)] data) = outer(data)
    //
    // The map inside `inner` should see hint(256).

    let mut counter = 0u32;

    // Build inner: Lam("arr", map(|x| x, arr))
    let x_ref = mk_var(&mut counter, "x", f32_ty());
    let lambda = mk_lam(&mut counter, "x", f32_ty(), x_ref);
    let arr_ref = mk_var(&mut counter, "arr", array_ty(f32_ty()));
    let map_call = mk_curried_call(&mut counter, "map", vec![lambda, arr_ref], array_ty(f32_ty()));
    let inner_body = mk_lam(&mut counter, "arr", array_ty(f32_ty()), map_call);

    let inner_def = Def {
        name: "inner".to_string(),
        ty: arrow_ty(array_ty(f32_ty()), array_ty(f32_ty())),
        body: inner_body,
        meta: DefMeta::Function,
        arity: 1,
    };

    // Build outer: Lam("arr", inner(arr))
    let arr_ref2 = mk_var(&mut counter, "arr", array_ty(f32_ty()));
    let inner_call = mk_curried_call(&mut counter, "inner", vec![arr_ref2], array_ty(f32_ty()));
    let outer_body = mk_lam(&mut counter, "arr", array_ty(f32_ty()), inner_call);

    let outer_def = Def {
        name: "outer".to_string(),
        ty: arrow_ty(array_ty(f32_ty()), array_ty(f32_ty())),
        body: outer_body,
        meta: DefMeta::Function,
        arity: 1,
    };

    // Set up analyzer
    let all_defs: HashMap<&str, &Def> =
        [("inner", &inner_def), ("outer", &outer_def)].into_iter().collect();
    let mut analyzer = SoacAnalyzer::new(all_defs);

    // Entry environment: data has hint(256)
    analyzer.env.insert("data".to_string(), SizeExpr::Hint(256));

    // Build call: outer(data)
    let data_ref = mk_var(&mut counter, "data", array_ty(f32_ty()));
    let call_to_outer = mk_curried_call(&mut counter, "outer", vec![data_ref], array_ty(f32_ty()));

    analyzer.analyze_term(&call_to_outer);

    // Should find one map SOAC with hint(256) propagated through both calls
    assert_eq!(analyzer.results.len(), 1);
    assert_eq!(analyzer.results[0].kind, SoacKind::Map);
    assert_eq!(analyzer.results[0].input_size, SizeExpr::Hint(256));
}

#[test]
fn test_hint_propagation_through_let_binding() {
    // Scenario:
    // entry main(#[size_hint(100)] arr) =
    //   let processed = map(|x| x, arr) in
    //   reduce(|a, b| a + b, 0.0, processed)
    //
    // Both SOACs should see hint(100) - the reduce sees it from the map output.

    let mut counter = 0u32;

    // Build: let processed = map(|x| x, arr) in reduce(|a, b| a + b, 0.0, processed)
    // Build map lambda step by step to avoid borrow issues
    let x_ref1 = mk_var(&mut counter, "x", f32_ty());
    let map_lambda = mk_lam(&mut counter, "x", f32_ty(), x_ref1);
    let arr_ref = mk_var(&mut counter, "arr", array_ty(f32_ty()));
    let map_call = mk_curried_call(&mut counter, "map", vec![map_lambda, arr_ref], array_ty(f32_ty()));

    // Build reduce lambda step by step
    let a_ref = mk_var(&mut counter, "a", f32_ty());
    let inner_lam = mk_lam(&mut counter, "b", f32_ty(), a_ref);
    let reduce_lambda = mk_lam(&mut counter, "a", f32_ty(), inner_lam);
    let neutral = mk_float_lit(&mut counter, 0.0);
    let processed_ref = mk_var(&mut counter, "processed", array_ty(f32_ty()));
    let reduce_call = mk_curried_call(
        &mut counter,
        "reduce",
        vec![reduce_lambda, neutral, processed_ref],
        f32_ty(),
    );

    let let_expr = mk_let(&mut counter, "processed", map_call, reduce_call);

    // Set up analyzer
    let all_defs: HashMap<&str, &Def> = HashMap::new();
    let mut analyzer = SoacAnalyzer::new(all_defs);
    analyzer.env.insert("arr".to_string(), SizeExpr::Hint(100));

    analyzer.analyze_term(&let_expr);

    // Should find both map and reduce SOACs
    assert_eq!(analyzer.results.len(), 2);

    let map_soac = analyzer.results.iter().find(|s| s.kind == SoacKind::Map).unwrap();
    let reduce_soac = analyzer.results.iter().find(|s| s.kind == SoacKind::Reduce).unwrap();

    // Map input should be hint(100)
    assert_eq!(map_soac.input_size, SizeExpr::Hint(100));
    // Reduce input should also be hint(100) (propagated from map output via let binding)
    assert_eq!(reduce_soac.input_size, SizeExpr::Hint(100));
}

// =============================================================================
// Tests for lambda captures with size hints
// =============================================================================

#[test]
fn test_lambda_captures_hinted_variable() {
    // Scenario:
    // entry main(#[size_hint(64)] data: []f32, #[size_hint(32)] weights: []f32) =
    //   map(|x| reduce(|a, b| a + b, 0.0, weights), data)
    //
    // The outer map should see hint(64) from data.
    // The inner reduce inside the lambda captures `weights` with hint(32).

    let mut counter = 0u32;

    // Build reduce lambda step by step to avoid borrow issues
    let a_ref = mk_var(&mut counter, "a", f32_ty());
    let inner_lam = mk_lam(&mut counter, "b", f32_ty(), a_ref);
    let reduce_lambda = mk_lam(&mut counter, "a", f32_ty(), inner_lam);
    let neutral = mk_float_lit(&mut counter, 0.0);
    let weights_ref = mk_var(&mut counter, "weights", array_ty(f32_ty()));
    let reduce_call = mk_curried_call(
        &mut counter,
        "reduce",
        vec![reduce_lambda, neutral, weights_ref],
        f32_ty(),
    );

    // Outer map: map(|x| <reduce_call>, data)
    let map_lambda = mk_lam(&mut counter, "x", f32_ty(), reduce_call);
    let data_ref = mk_var(&mut counter, "data", array_ty(f32_ty()));
    let map_call = mk_curried_call(&mut counter, "map", vec![map_lambda, data_ref], array_ty(f32_ty()));

    // Set up analyzer with both hints
    let all_defs: HashMap<&str, &Def> = HashMap::new();
    let mut analyzer = SoacAnalyzer::new(all_defs);
    analyzer.env.insert("data".to_string(), SizeExpr::Hint(64));
    analyzer.env.insert("weights".to_string(), SizeExpr::Hint(32));

    analyzer.analyze_term(&map_call);

    // Should find both map and reduce SOACs
    assert_eq!(analyzer.results.len(), 2);

    let map_soac = analyzer.results.iter().find(|s| s.kind == SoacKind::Map).unwrap();
    let reduce_soac = analyzer.results.iter().find(|s| s.kind == SoacKind::Reduce).unwrap();

    // Map sees hint(64) from data
    assert_eq!(map_soac.input_size, SizeExpr::Hint(64));
    assert_eq!(map_soac.nesting_depth, 0);

    // Reduce sees hint(32) from captured weights
    assert_eq!(reduce_soac.input_size, SizeExpr::Hint(32));
    assert_eq!(reduce_soac.nesting_depth, 1); // Nested inside map
}

#[test]
fn test_nested_soacs_with_different_hints() {
    // Scenario: map inside map with different source arrays
    // map(|row| map(|x| x, row), matrix)
    // where matrix has hint(10) rows and each row has hint(20) elements

    let mut counter = 0u32;

    // For this test, we'll use a simplified model where matrix is [][]f32
    // and we track the inner array size separately
    let inner_array_ty = array_ty(f32_ty());
    let outer_array_ty = array_ty(inner_array_ty.clone());

    // Inner map: map(|x| x, row)
    let x_ref = mk_var(&mut counter, "x", f32_ty());
    let inner_lambda = mk_lam(&mut counter, "x", f32_ty(), x_ref);
    let row_ref = mk_var(&mut counter, "row", inner_array_ty.clone());
    let inner_map = mk_curried_call(&mut counter, "map", vec![inner_lambda, row_ref], inner_array_ty.clone());

    // Outer map: map(|row| inner_map, matrix)
    let outer_lambda = mk_lam(&mut counter, "row", inner_array_ty.clone(), inner_map);
    let matrix_ref = mk_var(&mut counter, "matrix", outer_array_ty.clone());
    let outer_map = mk_curried_call(&mut counter, "map", vec![outer_lambda, matrix_ref], outer_array_ty.clone());

    // Set up analyzer
    let all_defs: HashMap<&str, &Def> = HashMap::new();
    let mut analyzer = SoacAnalyzer::new(all_defs);
    analyzer.env.insert("matrix".to_string(), SizeExpr::Hint(10));

    analyzer.analyze_term(&outer_map);

    // Should find two map SOACs
    assert_eq!(analyzer.results.len(), 2);

    let outer_soac = analyzer.results.iter().find(|s| s.nesting_depth == 0).unwrap();
    let inner_soac = analyzer.results.iter().find(|s| s.nesting_depth == 1).unwrap();

    // Outer map sees hint(10)
    assert_eq!(outer_soac.input_size, SizeExpr::Hint(10));

    // Inner map sees Unknown (row comes from lambda param, not tracked)
    assert_eq!(inner_soac.input_size, SizeExpr::Unknown);
}
