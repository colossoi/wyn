#![cfg(test)]
//! Integration tests for the full compilation pipeline.
//!
//! These tests verify that source code compiles correctly through all stages:
//! parse → desugar → resolve → type_check → alias_check → TLC → monomorphize → SSA
//!
//! All tests include entry points to ensure monomorphization can find reachable code.

use crate::ssa::types::Program;

/// Run source through the pipeline up to SSA.
fn compile_to_ssa(input: &str) -> Program {
    crate::compile_thru_ssa(input).expect("compile to SSA").ssa
}

/// Helper to check that code fails type checking (for testing error cases).
fn should_fail_type_check(input: &str) -> bool {
    crate::compile_thru_frontend(input).is_err()
}

/// Check that a function with the given name exists in the SSA program.
fn has_function(ssa: &Program, name: &str) -> bool {
    ssa.functions.iter().any(|f| f.name == name)
}

/// Helper to compile up through TLC fusion (stops before defunctionalization).
/// Off-milestone stop — drives the typestate API directly so the same
/// `module_manager` covers both `type_check` and `to_tlc`.
fn compile_to_fused_tlc(input: &str) -> crate::tlc::Program {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let type_checked = crate::Compiler::parse(input, &mut node_counter)
        .expect("parse")
        .desugar(&mut node_counter)
        .expect("desugar")
        .resolve(&module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    let tlc = type_checked.to_tlc(&module_manager, false);
    let fused =
        tlc.partial_eval().normalize_soacs().fuse_maps().apply_ownership().expect("apply_ownership");
    fused.0.tlc
}

// =============================================================================
// Phase 2 regression: unbound `Var(Symbol(sym))` references through TLC passes
// =============================================================================

/// Walk a term and assert that every `TermKind::Var(VarRef::Symbol(sym))`
/// references a sym that is either:
/// - bound by an enclosing Let / Lambda param / Loop var / SOAC element
///   parameter, or
/// - a top-level def name in `top_level`.
///
/// On violation, panics with the offending sym, its symbol-table name,
/// and the pipeline stage name.
fn assert_no_unbound_var_refs(program: &crate::tlc::Program, stage: &str) {
    use crate::SymbolId;
    use crate::tlc::{ArrayExpr, Lambda, LoopKind, SoacOp, Term, TermKind};
    use std::collections::HashSet;

    fn walk(
        term: &Term,
        bound: &HashSet<SymbolId>,
        symbols: &crate::SymbolTable,
        stage: &str,
        def_name: &str,
    ) {
        match &term.kind {
            TermKind::Var(crate::tlc::VarRef::Symbol(sym)) => {
                assert!(
                    bound.contains(sym),
                    "[{stage}] def `{def_name}`: unbound Var(sym{:?}) name={:?}",
                    sym.0,
                    symbols.get(*sym)
                );
            }
            TermKind::Var(crate::tlc::VarRef::Builtin { .. })
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::Extern(_) => {}
            TermKind::App { func, args } => {
                walk(func, bound, symbols, stage, def_name);
                for a in args {
                    walk(a, bound, symbols, stage, def_name);
                }
            }
            TermKind::Lambda(Lambda { params, body, .. }) => {
                let mut inner = bound.clone();
                for (p, _) in params {
                    inner.insert(*p);
                }
                walk(body, &inner, symbols, stage, def_name);
            }
            TermKind::Let { name, rhs, body, .. } => {
                walk(rhs, bound, symbols, stage, def_name);
                let mut inner = bound.clone();
                inner.insert(*name);
                walk(body, &inner, symbols, stage, def_name);
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                walk(cond, bound, symbols, stage, def_name);
                walk(then_branch, bound, symbols, stage, def_name);
                walk(else_branch, bound, symbols, stage, def_name);
            }
            TermKind::Loop {
                loop_var,
                init,
                init_bindings,
                kind,
                body,
                ..
            } => {
                walk(init, bound, symbols, stage, def_name);
                for (_, _, e) in init_bindings {
                    walk(e, bound, symbols, stage, def_name);
                }
                match kind {
                    LoopKind::For { iter, .. } => walk(iter, bound, symbols, stage, def_name),
                    LoopKind::ForRange { bound: bnd, .. } => {
                        walk(bnd, bound, symbols, stage, def_name);
                    }
                    LoopKind::While { cond } => walk(cond, bound, symbols, stage, def_name),
                }
                let mut inner = bound.clone();
                inner.insert(*loop_var);
                if let LoopKind::For { var, .. } | LoopKind::ForRange { var, .. } = kind {
                    inner.insert(*var);
                }
                for (n, _, _) in init_bindings {
                    inner.insert(*n);
                }
                walk(body, &inner, symbols, stage, def_name);
            }
            TermKind::Soac(soac) => walk_soac(soac, bound, symbols, stage, def_name),
            TermKind::ArrayExpr(ae) => walk_array_expr(ae, bound, symbols, stage, def_name),
            TermKind::Force(inner) => walk(inner, bound, symbols, stage, def_name),
        }
    }

    fn walk_lambda(
        lam: &crate::tlc::Lambda,
        bound: &HashSet<SymbolId>,
        symbols: &crate::SymbolTable,
        stage: &str,
        def_name: &str,
    ) {
        let mut inner = bound.clone();
        for (p, _) in &lam.params {
            inner.insert(*p);
        }
        walk(&lam.body, &inner, symbols, stage, def_name);
    }

    fn walk_soac(
        soac: &SoacOp,
        bound: &HashSet<SymbolId>,
        symbols: &crate::SymbolTable,
        stage: &str,
        def_name: &str,
    ) {
        match soac {
            SoacOp::Map { lam, inputs, .. } => {
                for i in inputs {
                    walk_array_expr(i, bound, symbols, stage, def_name);
                }
                walk_lambda(&lam.lam, bound, symbols, stage, def_name);
            }
            SoacOp::Reduce { op, ne, input, .. } => {
                walk(ne, bound, symbols, stage, def_name);
                walk_array_expr(input, bound, symbols, stage, def_name);
                walk_lambda(&op.lam, bound, symbols, stage, def_name);
            }
            SoacOp::Scan { op, ne, input } => {
                walk(ne, bound, symbols, stage, def_name);
                walk_array_expr(input, bound, symbols, stage, def_name);
                walk_lambda(&op.lam, bound, symbols, stage, def_name);
            }
            SoacOp::Redomap {
                op,
                reduce_op,
                ne,
                inputs,
                ..
            } => {
                walk(ne, bound, symbols, stage, def_name);
                for i in inputs {
                    walk_array_expr(i, bound, symbols, stage, def_name);
                }
                walk_lambda(&op.lam, bound, symbols, stage, def_name);
                walk_lambda(&reduce_op.lam, bound, symbols, stage, def_name);
            }
            SoacOp::Filter { pred, input } => {
                walk_array_expr(input, bound, symbols, stage, def_name);
                walk_lambda(&pred.lam, bound, symbols, stage, def_name);
            }
            SoacOp::Scatter { indices, values, .. } => {
                walk_array_expr(indices, bound, symbols, stage, def_name);
                walk_array_expr(values, bound, symbols, stage, def_name);
            }
            SoacOp::ReduceByIndex {
                op,
                ne,
                indices,
                values,
                ..
            } => {
                walk(ne, bound, symbols, stage, def_name);
                walk_array_expr(indices, bound, symbols, stage, def_name);
                walk_array_expr(values, bound, symbols, stage, def_name);
                walk_lambda(&op.lam, bound, symbols, stage, def_name);
            }
        }
    }

    fn walk_array_expr(
        ae: &ArrayExpr,
        bound: &HashSet<SymbolId>,
        symbols: &crate::SymbolTable,
        stage: &str,
        def_name: &str,
    ) {
        match ae {
            ArrayExpr::Ref(t) => walk(t, bound, symbols, stage, def_name),
            ArrayExpr::Zip(arrs) => {
                for a in arrs {
                    walk_array_expr(a, bound, symbols, stage, def_name);
                }
            }
            ArrayExpr::Soac(soac) => walk_soac(soac, bound, symbols, stage, def_name),
            ArrayExpr::Generate { index_fn, .. } => {
                walk_lambda(&index_fn.lam, bound, symbols, stage, def_name)
            }
            ArrayExpr::Literal(elems) => {
                for e in elems {
                    walk(e, bound, symbols, stage, def_name);
                }
            }
            ArrayExpr::Range { start, len } => {
                walk(start, bound, symbols, stage, def_name);
                walk(len, bound, symbols, stage, def_name);
            }
            ArrayExpr::StorageBuffer { offset, len, .. } => {
                walk(offset, bound, symbols, stage, def_name);
                walk(len, bound, symbols, stage, def_name);
            }
        }
    }

    // `bound` = everything the TLC name-resolver considers a top-level
    // symbol, not just things with bodies. `def_syms` holds the
    // pre-allocated SymbolId for every prelude/user/sig top-level name.
    let mut top_level: std::collections::HashSet<SymbolId> = std::collections::HashSet::new();
    for sym in program.def_syms.values() {
        top_level.insert(*sym);
    }
    // Catch any defs whose own SymbolId isn't already in `def_syms` —
    // shouldn't happen by construction, but assert via union.
    for d in &program.defs {
        top_level.insert(d.name);
    }
    for def in &program.defs {
        let def_name = program.symbols.get(def.name).cloned().unwrap_or_default();
        walk(&def.body, &top_level, &program.symbols, stage, &def_name);
    }
}

/// Regression: under Phase 2, `let x = arr in body[x]` was leaving a
/// free `Var(Symbol)` in the post-`partial_eval` TLC because `apply`'s
/// `Var(Builtin)` and catch-all arms cloned the original term instead
/// of residualizing through the eval'd args. The fix lives in
/// `tlc::partial_eval::residualize_call`. This test compiles the
/// minimal repro through the full canonical TLC pipeline and asserts
/// no `Var(Symbol(sym))` references an unbound symbol.
#[test]
fn let_binding_substitution_survives_partial_eval() {
    let source = r#"
#[fragment]
entry frag() #[location(0)] vec4f32 =
    let range = [1, 2, 3, 4] in
    @[f32.i32(range[0]), 0.0, 0.0, 1.0]
"#;
    let tlc = crate::compile_thru_tlc(source).expect("compile_thru_tlc");
    assert_no_unbound_var_refs(&tlc.tlc, "compile_thru_tlc");
}

// =============================================================================
// SOAC Fusion Integration Tests
// =============================================================================

#[test]
fn consuming_map_compiles_end_to_end() {
    // `*[N]T` map whose input is dead-after: the ownership pass
    // sets `consumes_input = true`, EGIR conversion produces
    // `SoacDestination::InputBuffer`, and `soac_expand` emits the
    // in-place loop. Compiling end-to-end through SSA exercises
    // every layer.
    let _ssa = compile_to_ssa(
        r#"
def f(a: *[8]i32) [8]i32 = map(|x: i32| x + 1, a)
"#,
    );
}

/// Count `_w_intrinsic_uninit` calls across the entire SSA program
/// (functions + entry points). `Fresh` Map destinations introduce
/// one per allocation; the `InputBuffer` destination should
/// introduce zero. Aggregating across all bodies sidesteps
/// inlining choices that move the map's body between functions.
fn count_uninit_in_program(ssa: &Program) -> usize {
    let mut count = 0;
    let bodies = ssa
        .functions
        .iter()
        .map(|f| &f.body.inner.insts)
        .chain(ssa.entry_points.iter().map(|e| &e.body.inner.insts));
    for insts in bodies {
        for (_id, inst) in insts {
            match &inst.data {
                crate::ssa::types::InstKind::Call { func: f, .. } => {
                    if f == "_w_intrinsic_uninit" {
                        count += 1;
                    }
                }
                crate::ssa::types::InstKind::Intrinsic { id, .. } => {
                    let name = crate::builtins::by_id(*id).raw.surface_name;
                    if name == "_w_intrinsic_uninit" {
                        count += 1;
                    }
                }
                _ => {}
            }
        }
    }
    count
}

#[test]
fn consuming_map_skips_fresh_allocation() {
    // For an input-side DPS Map, the loop's carried buffer starts
    // as the input parameter — no `_w_intrinsic_uninit` call
    // should be emitted. Compare against a non-consuming Map
    // (caller-borrowed input) which falls back to Fresh and
    // emits at least one uninit allocation.
    let consuming_ssa = compile_to_ssa(
        r#"
def bump(a: *[8]i32) [8]i32 = map(|x: i32| x + 1, a)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let r = bump([1, 2, 3, 4, 5, 6, 7, 8]) in
    @[f32.i32(r[0]), f32.i32(r[1]), 0.0, 0.0]
"#,
    );
    assert_eq!(
        count_uninit_in_program(&consuming_ssa),
        0,
        "consuming map (`*[N]T` input, dead-after) should not allocate a fresh buffer",
    );

    let borrowing_ssa = compile_to_ssa(
        r#"
def bump(a: [8]i32) [8]i32 = map(|x: i32| x + 1, a)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let r = bump([1, 2, 3, 4, 5, 6, 7, 8]) in
    @[f32.i32(r[0]), f32.i32(r[1]), 0.0, 0.0]
"#,
    );
    assert!(
        count_uninit_in_program(&borrowing_ssa) >= 1,
        "non-consuming map (caller-borrowed input) should allocate a fresh buffer",
    );
}

#[test]
fn test_map_reduce_fusion_end_to_end() {
    let source = r#"
def globalArr: [4]f32 = [10.0, 20.0, 30.0, 40.0]

def myMap(ro: f32, rd: f32) [4]f32 =
  map(|x: f32| x + ro + rd, globalArr)

def myReduce(hits: [4]f32) f32 =
  reduce(|acc: f32, x: f32| if acc < x then acc else x, 999.0, hits)

#[fragment]
entry fragment_main() #[location(0)] vec4f32 =
  let hits = myMap(1.0, 2.0) in
  let closest = myReduce(hits) in
  @[closest, 0.0, 0.0, 1.0]
"#;

    let tlc = compile_to_fused_tlc(source);

    // After fusion, check that myMap's body is no longer a standalone map
    // or that some def contains a fused reduce
    let my_map_has_map = tlc.defs.iter().any(|def| {
        let name = tlc.symbols.get(def.name).cloned().unwrap_or_default();
        if name != "myMap" {
            return false;
        }
        let (_, body) = crate::tlc::extract_lambda_params(&def.body);
        has_soac_kind(&body, "Map")
    });

    let any_has_reduce = tlc.defs.iter().any(|def| {
        let (_, body) = crate::tlc::extract_lambda_params(&def.body);
        has_soac_kind(&body, "Reduce")
    });

    // Check fragment_main: does it contain a fused Reduce?
    let fragment_main = tlc
        .defs
        .iter()
        .find(|def| tlc.symbols.get(def.name).map(|s| s.as_str()) == Some("fragment_main"))
        .expect("fragment_main not found");

    let (_, frag_body) = crate::tlc::extract_lambda_params(&fragment_main.body);
    let frag_has_redomap = has_soac_kind(&frag_body, "Redomap");
    let frag_has_reduce = has_soac_kind(&frag_body, "Reduce");
    let frag_has_map = has_soac_kind(&frag_body, "Map");

    eprintln!("fragment_main has Redomap: {}", frag_has_redomap);
    eprintln!("fragment_main has Reduce: {}", frag_has_reduce);
    eprintln!("fragment_main has Map: {}", frag_has_map);
    eprintln!(
        "fragment_main body: {:?}",
        std::mem::discriminant(&frag_body.kind)
    );

    // Print the Let chain structure
    fn print_term(term: &crate::tlc::Term, syms: &crate::SymbolTable, depth: usize) {
        let indent = "  ".repeat(depth);
        match &term.kind {
            crate::tlc::TermKind::Let { name, rhs, body, .. } => {
                let n = syms.get(*name).cloned().unwrap_or_else(|| format!("{:?}", name));
                eprintln!("{indent}let {n} = ...");
                print_term(rhs, syms, depth + 1);
                print_term(body, syms, depth);
            }
            crate::tlc::TermKind::Soac(soac) => {
                eprintln!("{indent}SOAC {:?}", std::mem::discriminant(soac));
            }
            crate::tlc::TermKind::App { func, args } => {
                eprintln!("{indent}App:");
                print_term(func, syms, depth + 1);
                for a in args {
                    print_term(a, syms, depth + 1);
                }
            }
            crate::tlc::TermKind::Var(crate::tlc::VarRef::Symbol(s)) => {
                let n = syms.get(*s).cloned().unwrap_or_else(|| format!("{:?}", s));
                eprintln!("{indent}Var({n})");
            }
            other => {
                eprintln!("{indent}{:?}", std::mem::discriminant(other));
            }
        }
    }
    print_term(&frag_body, &tlc.symbols, 0);

    // The fusion should have replaced the let chain with a fused SOAC
    // or at minimum the fragment_main should contain a Reduce
    assert!(
        frag_has_redomap || frag_has_reduce,
        "Expected fragment_main to contain a fused Redomap or Reduce after interprocedural fusion"
    );
}

fn has_soac_kind(term: &crate::tlc::Term, kind: &str) -> bool {
    use crate::tlc::{SoacOp, TermKind};
    match &term.kind {
        TermKind::Soac(SoacOp::Map { .. }) if kind == "Map" => true,
        TermKind::Soac(SoacOp::Reduce { .. }) if kind == "Reduce" => true,
        TermKind::Soac(SoacOp::Redomap { .. }) if kind == "Redomap" => true,
        TermKind::Let { rhs, body, .. } => has_soac_kind(rhs, kind) || has_soac_kind(body, kind),
        TermKind::Lambda(lam) => has_soac_kind(&lam.body, kind),
        TermKind::App { func, args } => {
            has_soac_kind(func, kind) || args.iter().any(|a| has_soac_kind(a, kind))
        }
        _ => false,
    }
}

// =============================================================================
// Basic Expressions
// =============================================================================

#[test]
fn test_basic_expressions() {
    // Tests: functions, let bindings, if expressions, binary/unary ops
    let ssa = compile_to_ssa(
        r#"
def add(x: i32, y: i32) i32 = x + y

def with_let(a: i32, b: i32) i32 =
    let x = a in
    let y = b in
    x + y

def with_if(x: bool) i32 = if x then 1 else 0

def with_ops(x: i32, y: i32) i32 = x * y + x / y - (-x)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let b = add(1, 2) in
    let c = with_let(3, 4) in
    let d = with_if(true) in
    let e = with_ops(5, 6) in
    @[f32.i32(b + c + d + e), 0.0, 0.0, 1.0]
"#,
    );

    // Compilation success is the test (partial eval may inline simple functions)
}

// =============================================================================
// Data Structures
// =============================================================================

#[test]
fn test_data_structures() {
    // Tests: arrays, tuples, records, tuple patterns
    let _ssa = compile_to_ssa(
        r#"
def arr = [1, 2, 3]

def record = {x: 1, y: 2}

def tuple_destruct: i32 =
    let (a, b) = (1, 2) in a + b

def nested_tuple: i32 =
    let ((a, b), c) = ((1, 2), 3) in a + b + c

def array_index(arr: [4]i32, i: i32) i32 = arr[i]

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = arr[0] in
    let b = record.x in
    let c = tuple_destruct in
    let d = nested_tuple in
    let e = array_index([1, 2, 3, 4], 0) in
    @[f32.i32(a + b + c + d + e), 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Tuple Positional Access
// =============================================================================

#[test]
fn test_tuple_positional_access() {
    // Tests: .0, .1 on tuples, chained access, in expressions
    let _ssa = compile_to_ssa(
        r#"
def first(t: (i32, f32)) i32 = t.0

def second(t: (i32, f32)) f32 = t.1

def sum_pair(t: (i32, i32)) i32 = t.0 + t.1

def nested(t: ((i32, i32), f32)) i32 =
    let inner = t.0 in inner.0 + inner.1

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let t = (42, 3.14) in
    let a = first(t) in
    let b = sum_pair((1, 2)) in
    let c = nested(((10, 20), 1.0)) in
    @[f32.i32(a + b + c), 0.0, 0.0, 1.0]
"#,
    );
}

// =============================================================================
// Loops
// =============================================================================

#[test]
fn test_loops() {
    // Tests: while loops, for-range loops, for-in loops
    let _ssa = compile_to_ssa(
        r#"
def while_loop: i32 =
    loop x = 0 while x < 10 do x + 1

def for_range_loop: i32 =
    loop acc = 0 for i < 10 do acc + i

def for_in_loop(arr: [5]i32) i32 =
    loop acc = 0 for x in arr do acc + x

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = while_loop in
    let b = for_range_loop in
    let c = for_in_loop([1, 2, 3, 4, 5]) in
    @[f32.i32(a + b + c), 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Lambdas and Closures
// =============================================================================

#[test]
fn test_lambdas_and_closures() {
    // Tests: lambdas with captures, nested lambdas, direct calls, tuple params
    let _ssa = compile_to_ssa(
        r#"
def with_capture(y: i32) i32 =
    let f = |x: i32| x + y in
    f(10)

def nested_lambda(x: i32) i32 =
    let outer = |a: i32|
        let inner = |b: i32| a + b + x in
        inner(a)
    in
    outer(5)

def tuple_param_lambda: i32 =
    let add = |(x, y): (i32, i32)| x + y in
    add((1, 2))

def direct_call: i32 =
    let inc = |x: i32| x + 1 in
    inc(5)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let b = with_capture(10) in
    let c = nested_lambda(100) in
    let d = tuple_param_lambda in
    let e = direct_call in
    @[f32.i32(b + c + d + e), 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Higher-Order Functions (map, reduce, filter)
// =============================================================================

#[test]
fn test_higher_order_functions() {
    // Tests: map, reduce, filter with lambdas and named functions
    let _ssa = compile_to_ssa(
        r#"
def double(x: i32) i32 = x * 2

def map_named(arr: [4]i32) [4]i32 = map(double, arr)

def map_lambda(arr: [4]i32) [4]i32 = map(|x: i32| x + 1, arr)

def map_with_capture(arr: [4]i32, offset: i32) [4]i32 =
    map(|x: i32| x + offset, arr)

def reduce_sum(arr: [4]f32) f32 =
    reduce(|acc: f32, x: f32| acc + x, 0.0, arr)

def reduce_tuple(hits: [4](f32, i32)) (f32, i32) =
    reduce(|(t1, m1): (f32, i32), (t2, m2): (f32, i32)|
             if t1 < t2 then (t1, m1) else (t2, m2),
           (1000.0, 0),
           hits)

def is_positive(x: i32) bool = x > 0

def filter_positive(arr: [5]i32) ?k. [k]i32 =
    filter(is_positive, arr)

def filter_lambda(arr: [4]i32) ?k. [k]i32 =
    filter(|x: i32| x % 2 == 0, arr)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = map_named([1, 2, 3, 4]) in
    let b = map_lambda([1, 2, 3, 4]) in
    let c = map_with_capture([1, 2, 3, 4], 10) in
    let d = reduce_sum([1.0, 2.0, 3.0, 4.0]) in
    let (t, _) = reduce_tuple([(1.0, 0), (2.0, 1), (0.5, 2), (3.0, 3)]) in
    let e = filter_positive([1, -2, 3, -4, 5]) in
    let f = filter_lambda([1, 2, 3, 4]) in
    @[d + t, f32.i32(a[0] + b[0] + c[0] + length(e) + length(f)), 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Defunctionalization Scenarios
// =============================================================================

#[test]
fn test_defunctionalization() {
    // Tests various defunctionalization scenarios
    let _ssa = compile_to_ssa(
        r#"
def different_captures(x: i32, y: i32, arr: [4]i32) ([4]i32, [4]i32) =
    let result1 = map(|e: i32| e + x, arr) in
    let result2 = map(|e: i32| e * y, arr) in
    (result1, result2)

def nested_capture(x: i32, arr: [4]i32) [4]i32 =
    let outer = |y: i32|
        let inner = |z: i32| x + y + z in
        inner(y)
    in
    map(outer, arr)

def reused_lambda(x: i32, arr1: [4]i32, arr2: [4]i32) ([4]i32, [4]i32) =
    let adder = |e: i32| e + x in
    let result1 = map(adder, arr1) in
    let result2 = map(adder, arr2) in
    (result1, result2)

def hof_chain(scale: i32, offset: i32, arr: [4]i32) i32 =
    let scaled = map(|x: i32| x * scale, arr) in
    let shifted = map(|x: i32| x + offset, scaled) in
    reduce(|a: i32, b: i32| a + b, 0, shifted)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let (a, b) = different_captures(1, 2, [1, 2, 3, 4]) in
    let c = nested_capture(10, [1, 2, 3, 4]) in
    let (d, e) = reused_lambda(5, [1, 2, 3, 4], [5, 6, 7, 8]) in
    let f = hof_chain(2, 10, [1, 2, 3, 4]) in
    @[f32.i32(a[0] + b[0] + c[0] + d[0] + e[0] + f), 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Type Checking Errors
// =============================================================================

#[test]
fn test_type_errors() {
    // These fail during type checking, before monomorphization, so no entry points needed

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

    // Function from if expression
    assert!(
        should_fail_type_check(
            r#"
def choose(b: bool) (i32 -> i32) =
    if b then |x: i32| x + 1 else |x: i32| x * 2
"#
        ),
        "Should reject function returned from if expression"
    );

    // Loop parameter cannot be a function
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

/// Companion to `test_spirv_loop_carrying_map_over_iota`, with the
/// loop initialized from a composite ARRAY LITERAL rather than a
/// `map(…, iota(…))` call. Before the fix these had asymmetric
/// outcomes: literal-init produced a Composite variant, map-init
/// produced a Virtual variant, and the loop back-edge unification
/// failed with "Loop body type must match loop variable type:
/// Failure(Virtual, Composite)". Now that `map`'s output variant is
/// pinned to Composite, both styles compile.
#[test]
fn test_spirv_loop_carrying_literal_init() {
    let source = r#"
def f(seed: f32) [4]f32 =
    let init: [4]f32 = [seed, seed, seed, seed] in
    let (_, out) =
        loop (i, arr) = (0, init) while i < 2 do
            let arr' = map(|j: i32| arr[j] + 1.0, iota(4))
            in (i + 1, arr')
    in out

#[compute]
entry main(x: []f32) [4]f32 = f(x[0])
"#;
    compile_to_spirv(source).expect(
        "loop back-edge carrying a literal-init array across a map(…, iota(…)) body \
         should compile; both init and body variants are Composite",
    );
}

// =============================================================================
// Materialization Optimization
// =============================================================================

#[test]
fn test_materialization_optimization() {
    // Tests that materialization hoisting works correctly
    let _ssa = compile_to_ssa(
        r#"
def identity(arr: [3]i32) [3]i32 = arr

def no_redundant_complex(arr: [3]i32, i: i32) i32 =
    if true then (identity(arr))[i] else (identity(arr))[i]

def no_materialize_tuple(x: i32) i32 =
    let pair = (x, x + 1) in
    let (a, b) = pair in
    a + b

def no_materialize_loop_tuple(arr: [10]i32) i32 =
    let (sum, _) = loop (acc, i) = (0, 0) while i < 10 do
        (acc + arr[i], i + 1)
    in sum

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = no_redundant_complex([1, 2, 3], 0) in
    let b = no_materialize_tuple(5) in
    let c = no_materialize_loop_tuple([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) in
    @[f32.i32(a + b + c), 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Math Functions and Conversions
// =============================================================================

#[test]
fn test_math_and_conversions() {
    // Tests: f32 conversions, math operations, qualified names
    let _ssa = compile_to_ssa(
        r#"
def conversions(x: i32, y: i64) f32 =
    let f1 = f32.i32(x) in
    let f2 = f32.i64(y) in
    f1 + f2

def math_ops(x: f32) f32 =
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

def vector_length(v: vec2f32) f32 =
    f32.sqrt(v.x * v.x + v.y * v.y)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = conversions(1, 2i64) in
    let b = math_ops(1.0) in
    let c = vector_length(@[3.0, 4.0]) in
    @[a + b + c, 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Matrix Operations
// =============================================================================

#[test]
fn test_matrix_operations() {
    // Tests: mul overloads (mat*mat, mat*vec, vec*mat)
    let _ssa = compile_to_ssa(
        r#"
def test_mul(m1: mat4f32, m2: mat4f32, v: vec4f32) vec4f32 =
    let mat_result = mul(m1, m2) in
    let vec_result1 = mul(mat_result, v) in
    let vec_result2 = mul(v, m1) in
    vec_result1

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let m = @[[1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]] in
    let v = @[1.0, 2.0, 3.0, 1.0] in
    test_mul(m, m, v)
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Complex Shader Integration
// =============================================================================

#[test]
fn test_complex_shader() {
    // Full shader with uniforms, matrices, map, multiple functions
    let _ssa = compile_to_ssa(
        r#"
#[uniform(set=1, binding=0)] def iResolution: vec2f32
#[uniform(set=1, binding=1)] def iTime: f32

def verts: [3]vec4f32 =
    [@[-1.0, -1.0, 0.0, 1.0],
     @[3.0, -1.0, 0.0, 1.0],
     @[-1.0, 3.0, 0.0, 1.0]]

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vertex_id: i32) #[builtin(position)] vec4f32 =
    verts[vertex_id]

def translation(p: vec3f32) mat4f32 =
    @[[1.0f32, 0.0f32, 0.0f32, p.x],
      [0.0f32, 1.0f32, 0.0f32, p.y],
      [0.0f32, 0.0f32, 1.0f32, p.z],
      [0.0f32, 0.0f32, 0.0f32, 1.0f32]]

def rotation_y(angle: f32) mat4f32 =
    let s = f32.sin(angle) in
    let c = f32.cos(angle) in
    @[[c, 0.0f32, s, 0.0f32],
      [0.0f32, 1.0f32, 0.0f32, 0.0f32],
      [0.0 - s, 0.0f32, c, 0.0f32],
      [0.0f32, 0.0f32, 0.0f32, 1.0f32]]

def cube_corners: [8]vec3f32 =
    [@[-1.0, -1.0, 1.0], @[-1.0, 1.0, 1.0],
     @[1.0, 1.0, 1.0], @[1.0, -1.0, 1.0],
     @[-1.0, -1.0, -1.0], @[-1.0, 1.0, -1.0],
     @[1.0, 1.0, -1.0], @[1.0, -1.0, -1.0]]

def main_image(res: vec2f32, time: f32, fragCoord: vec2f32) vec4f32 =
    let cam = translation(@[0.0, 0.0, 10.0]) in
    let rot = rotation_y(time) in
    let mat = rot * cam in
    let v4s = map(|v: vec3f32| @[v.x, v.y, v.z, 1.0] * mat, cube_corners) in
    v4s[0]

#[fragment]
entry fragment_main(#[builtin(frag_coord)] pos: vec4f32) #[location(0)] vec4f32 =
    main_image(@[iResolution.x, iResolution.y], iTime, @[pos.x, pos.y])
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Full Pipeline to SPIR-V
// =============================================================================

#[test]
fn test_function_call_with_array_arg() {
    // Test calling a function with an array literal argument
    let source = r#"
def sum_first_two(arr: [4]i32) i32 =
    arr[0] + arr[1]

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let result = sum_first_two([1, 2, 3, 4]) in
    @[f32.i32(result), 0.0, 0.0, 1.0]
"#;

    let result = crate::compile_thru_spirv(source);

    assert!(result.is_ok(), "SPIR-V compilation failed: {:?}", result.err());
}

#[test]
fn test_compute_shader_with_storage_slice() {
    // Test compute shader with storage buffer slice
    let source = r#"
def sum_first_two(arr: [4]i32) i32 =
    arr[0] + arr[1]

#[compute]
entry compute_main(data: []i32) i32 =
    let from_storage = sum_first_two(data[0..4]) in
    let from_literal = sum_first_two([1, 2, 3, 4]) in
    from_storage + from_literal
"#;

    let result = crate::compile_thru_spirv(source);

    assert!(result.is_ok(), "SPIR-V compilation failed: {:?}", result.err());
}

#[test]
fn test_full_pipeline_to_spirv() {
    // Verify the full pipeline compiles successfully to SPIR-V
    let source = r#"
#[uniform(set=1, binding=0)] def iTime: f32

def compute(x: f32, y: f32) f32 =
    let a = f32.sin(x) in
    let b = f32.cos(y) in
    a + b

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let s = compute(pos.x, pos.y) in
    @[s + iTime, 0.0, 0.0, 1.0]
"#;

    let result = crate::compile_thru_spirv(source);

    assert!(result.is_ok(), "SPIR-V compilation failed: {:?}", result.err());
}

// =============================================================================
// Array Variant Monomorphization
// =============================================================================

#[test]
fn test_array_variant_monomorphization() {
    // Slicing a view with constant bounds produces a Composite array (materialized at
    // the call site), so both call sites use the same Composite variant of sum_first_two.
    let ssa = compile_to_ssa(
        r#"
def sum_first_two(arr: [4]i32) i32 =
    arr[0] + arr[1]

#[compute]
entry compute_main(data: []i32) i32 =
    let from_storage = sum_first_two(data[0..4]) in
    let from_literal = sum_first_two([1, 2, 3, 4]) in
    from_storage + from_literal
"#,
    );

    // Collect all sum_first_two variants (including buffer-specialized)
    let sum_versions: Vec<_> =
        ssa.functions.iter().filter(|f| f.name.starts_with("sum_first_two")).collect();

    eprintln!("sum_first_two SSA functions:");
    for f in &sum_versions {
        eprintln!("  {}", f.name);
        // Show param types
        for (val, ty, name) in &f.body.params {
            eprintln!("    param {} ({:?}) :: {:?}", name, val, ty);
        }
        // Show all instructions that involve indexing or storage views
        for inst in f.body.inner.insts.values() {
            match &inst.data {
                crate::ssa::types::InstKind::Index { .. } => {
                    eprintln!("    inst {:?}: Index", inst.result);
                }
                crate::ssa::types::InstKind::StorageView { .. } => {
                    eprintln!("    inst {:?}: StorageView", inst.result);
                }
                crate::ssa::types::InstKind::ViewIndex { .. } => {
                    eprintln!("    inst {:?}: ViewIndex", inst.result);
                }
                _ => {}
            }
        }
    }

    // After TLC-level inlining and DCE, sum_first_two may be fully inlined
    // at all call sites and eliminated. The important thing is that the program
    // compiles successfully to SSA — buffer specialization is tested implicitly.
    // (The function may or may not survive depending on inlining thresholds.)
}

// =============================================================================
// SPIR-V Block Param / Phi Node Tests
// =============================================================================

/// Compile source all the way through SPIR-V and return Ok/Err.
fn compile_to_spirv(input: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    Ok(crate::compile_thru_spirv(input)?.spirv)
}

/// Helper: compile source through SSA. Same as `compile_to_ssa`; kept as
/// a separate name because some legacy tests distinguished module-bearing
/// programs from non-module ones — `compile_thru_frontend` handles both.
fn compile_to_ssa_with_modules(input: &str) -> Program {
    crate::compile_thru_ssa(input).expect("compile to SSA").ssa
}

/// Verify that nested if/else chains compile to SPIR-V.
#[test]
fn test_spirv_nested_if_else_block_params() {
    let source = r#"
def choose(a: f32, b: f32, c: f32, sel1: i32, sel2: i32) f32 =
    let x = if sel1 == 0 then a
            else if sel1 == 1 then b
            else c in
    let y = if sel2 == 0 then a
            else if sel2 == 1 then c
            else b in
    x + y

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let r = choose(pos.x, pos.y, pos.z, 1, 2) in
    @[r, 0.0, 0.0, 1.0]
"#;
    compile_to_spirv(source).expect("Nested if/else should compile to SPIR-V");
}

/// Verify many conditional branches producing block params compile to SPIR-V.
#[test]
fn test_spirv_many_conditional_block_params() {
    let source = r#"
def process(a: f32, b: f32, c: f32, d: f32, flag: i32) (f32, f32, f32, f32) =
    let x = if flag == 0 then a + b else a - b in
    let y = if flag == 1 then b + c else b * c in
    let z = if flag == 2 then c + d else c - d in
    let w = if flag == 0 then d * a else d + a in
    (x, y, z, w)

def combine(t: (f32, f32, f32, f32)) f32 =
    let (a, b, c, d) = t in
    a + b + c + d

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let result = process(pos.x, pos.y, pos.z, pos.w, 1) in
    let s = combine(result) in
    @[s, 0.0, 0.0, 1.0]
"#;
    compile_to_spirv(source).expect("Many conditionals should compile to SPIR-V");
}

/// Verify maps over small arrays with nested conditionals compile to SPIR-V.
#[test]
fn test_spirv_map_with_nested_conditionals() {
    let source = r#"
def selectValue(x: f32, flag: i32) f32 =
    if flag == 0 then x * 2.0
    else if flag == 1 then x + 1.0
    else x - 1.0

#[compute]
entry compute_main(data: [8]f32) [8]f32 =
    map(|x| selectValue(x, 1), data)
"#;
    compile_to_spirv(source).expect("Map with nested conditionals should compile to SPIR-V");
}

/// Verify multiple maps followed by a reduce compile to SPIR-V.
#[test]
fn test_spirv_multiple_maps_and_reduce() {
    let source = r#"
#[compute]
entry compute_main(data: [8](f32, f32)) [8]f32 =
    let first = map(|t| let (a, _) = t in a, data) in
    let second = map(|t| let (_, b) = t in b, data) in
    let combined = map(|(a, b): (f32, f32)| a + b, zip(first, second)) in
    let total = reduce(|a: f32, b: f32| a + b, 0.0, combined) in
    map(|x| x + total, combined)
"#;
    compile_to_spirv(source).expect("Multiple maps + reduce should compile to SPIR-V");
}

/// Verify conditional array element selection compiles to SPIR-V
/// (the finalOrigins/finalDirs pattern from raytrace.wyn).
#[test]
fn test_spirv_conditional_array_construction() {
    let source = r#"
def build(a: [4]f32, b: [4]f32, flags: [4]i32) [4]f32 =
    [
        if flags[0] == 1 then b[0] else a[0],
        if flags[1] == 1 then b[1] else a[1],
        if flags[2] == 1 then b[2] else a[2],
        if flags[3] == 1 then b[3] else a[3]
    ]

#[compute]
entry compute_main(data: [4]f32) [4]f32 =
    let doubled = map(|x| x * 2.0, data) in
    let flags = [1, 0, 1, 0] in
    build(data, doubled, flags)
"#;
    compile_to_spirv(source).expect("Conditional array construction should compile to SPIR-V");
}

/// Regression: mapping a lambda whose return type is a mixed
/// scalar/vector tuple. The SoA transform rewrites the output
/// `[N](f32, i32, vec3f32)` into a tuple-of-arrays before EGIR
/// conversion; `egir::soac_expand` must split the per-iteration
/// ArrayWith into per-component ArrayWith calls + a Tuple repack.
/// Without the split, SPIR-V lowering's runtime-index path hit a
/// cache miss and failed with "element type not found".
#[test]
fn test_spirv_map_array_of_mixed_tuple() {
    let source = r#"
def build(xs: [8]f32) [8](f32, i32, vec3f32) =
    map(|x: f32| (x + 1.0, 0, @[x, x, x]), xs)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let arr = build([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) in
    let (a, _, v) = arr[3] in
    @[a, v.x, v.y, v.z]
"#;
    compile_to_spirv(source).expect("map over [N](f32, i32, vec3f32) should compile to SPIR-V");
}

/// Regression: nested SoA. The element type itself contains a
/// composite array, so the SoA transform produces a tuple of
/// arrays whose components are themselves arrays — exercising
/// `emit_write_element`'s recursion through `soa_element_type`.
#[test]
fn test_spirv_map_array_of_nested_tuple() {
    let source = r#"
def build(xs: [4]f32) [4](f32, [3]f32) =
    map(|x: f32| (x + 1.0, [x, x, x]), xs)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let arr = build([0.0, 1.0, 2.0, 3.0]) in
    let (a, inner) = arr[2] in
    @[a, inner[0], inner[1], inner[2]]
"#;
    compile_to_spirv(source).expect("map over [N](f32, [M]f32) should compile to SPIR-V");
}

/// Regression: a loop that carries an array whose next-iteration
/// value comes from `map(…, iota(N))`.
///
/// Before the fix, `map`'s type scheme claimed the output variant
/// was the same as the input's. Since `iota` returns a Virtual
/// array, `map`'s output was typed Virtual — but `egir::soac_expand`
/// actually materializes the result via `_w_intrinsic_uninit` +
/// `_w_intrinsic_array_with_inplace`, which is a Composite
/// representation. That type/representation mismatch surfaced as
/// a SPIR-V "ArrayWith: element type not found" cache miss when
/// the loop back-edge carried what SSA thought was a Virtual
/// array. Fix: pin `map` / `scan` / `filter` output variant to
/// Composite in the type scheme, matching the runtime
/// representation.
#[test]
fn test_spirv_loop_carrying_map_over_iota() {
    let source = r#"
def f(seed: f32) [4]f32 =
  let init: [4]f32 = map(|j: i32| seed + f32.i32(j), iota(4)) in
  let (_, out) =
    loop (i, arr) = (0, init) while i < 2 do
      let arr' = map(|j: i32| arr[j] + 1.0, iota(4))
      in (i + 1, arr')
  in out

#[compute]
entry main(x: []f32) [4]f32 = f(x[0])
"#;
    compile_to_spirv(source).expect(
        "loop carrying `map(..., iota(N))` should compile; currently fails with \
         ArrayWith cache miss because the back-edge array is Virtual variant",
    );
}

/// Test the specific raytrace.wyn file compiles to SPIR-V.
#[test]
fn test_spirv_raytrace() {
    let source = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../testfiles/raytrace.wyn"))
        .expect("Could not read testfiles/raytrace.wyn");
    compile_to_spirv(&source).expect("raytrace.wyn should compile to SPIR-V");
}

/// Regression: if/else before interprocedural map+reduce fusion caused
/// UnterminatedBlock in soac_lower. The if/else creates a dead Unreachable
/// block in SSA, which soac_lower's rebuild would pre-create but finish()
/// rejected because Unreachable doubles as the "unterminated" sentinel.
#[test]
fn test_interproc_fusion_if_before_fused_reduce() {
    let source = r#"
def maxDist: f32 = 100.0
def globalData: [12]f32 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

def producer(x: f32) [12]f32 =
  map(|a: f32| a * x, globalData)

def consumer(arr: [12]f32) f32 =
  reduce(|acc: f32, x: f32| if acc < x then acc else x, maxDist, arr)

def scene(x: f32, y: f32) f32 =
  let ground = if y > 0.0 then y else maxDist in
  let hits = producer(x) in
  let closest = consumer(hits) in
  closest + ground

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vid: i32) #[builtin(position)] vec4f32 =
  let r = scene(1.0, 0.5) in
  @[r, 0.0, 0.0, 1.0]
"#;
    compile_to_spirv(source).expect("if-before-interproc-fusion should compile");
}

/// Verify raytrace.wyn compiles through SSA to SPIR-V without errors.
/// This exercises the RPO block emission and incremental array literal
/// lowering that were needed for complex cross-block value references.
/// (test_spirv_raytrace covers this; this test verifies the SSA is well-formed
/// enough that compile_to_ssa_with_modules succeeds.)
#[test]
fn test_ssa_raytrace_well_formed() {
    let source = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../testfiles/raytrace.wyn"))
        .expect("Could not read testfiles/raytrace.wyn");

    let ssa = compile_to_ssa_with_modules(&source);

    // Verify key functions exist
    assert!(
        ssa.functions.iter().any(|f| f.name == "trace"),
        "trace should be in SSA output"
    );
}

// =============================================================================
// Constant Inlining
// =============================================================================

/// Constants that reference other constants should be fully inlined.
#[test]
fn test_constant_referencing_constant() {
    let source = r#"
def PI: f32 = 3.14159265
def TAU: f32 = PI * 2.0
def QUARTER_TAU: f32 = TAU / 4.0

#[fragment]
entry frag(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
  @[QUARTER_TAU, PI, TAU, 1.0]
"#;
    compile_to_spirv(source).expect("constants referencing constants should compile");
}

// `test_soa_eliminates_extraction_maps` was deleted: it inspected post-SSA for
// `InstKind::Soac(Soac::Map { .. })`, but SOACs no longer exist as SSA
// instructions — they're expanded inside EGIR (`egir::soac_expand`) before
// elaboration produces any SSA. The underlying optimization concern (extraction
// maps after a zip-map should fold to tuple projections) is now an EGIR-level
// question and would need a different shape of test.

/// Pipeline that includes TLC inline_small (the new pass).
fn compile_to_ssa_with_inline_small(input: &str) -> Program {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let parsed = crate::Compiler::parse(input, &mut node_counter).expect("Parsing failed");
    let type_checked = parsed
        .desugar(&mut node_counter)
        .expect("Desugaring failed")
        .resolve(&mut module_manager)
        .expect("Name resolution failed")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("Type checking failed");

    type_checked
        .to_tlc(&module_manager, false)
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .apply_ownership()
        .expect("apply_ownership")
        .defunctionalize()
        .monomorphize()
        .buffer_specialize()
        .fold_generated_lambdas()
        .inline_small()
        .to_egraph()
        .expect("SSA conversion failed")
        .expand_soacs(true)
        .materialize()
        .optimize_skeleton()
        .elaborate()
        .ssa
}

#[test]
fn test_constant_inlining_global_ref() {
    // Minimal repro: a constant def used by a function, going through inline_small.
    // This should NOT produce an unresolved Global("PI") in SSA.
    let ssa = compile_to_ssa_with_inline_small(
        r#"
def PI: f32 = 3.141592

def use_pi(x: f32) f32 = x * PI

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let r = use_pi(pos.x) in
    @[r, 0.0, 0.0, 1.0]
"#,
    );

    // Dump what we got.
    eprintln!("{}", crate::ssa::print::format_program(&ssa));

    // Check that no Global("PI") instruction exists — it should have been inlined.
    for func in &ssa.functions {
        for (_id, inst) in &func.body.inner.insts {
            if let crate::ssa::types::InstKind::Global(name) = &inst.data {
                assert_ne!(
                    name, "PI",
                    "Global @PI should have been inlined, but survived in function '{}'",
                    func.name
                );
            }
        }
    }
    for ep in &ssa.entry_points {
        for (_id, inst) in &ep.body.inner.insts {
            if let crate::ssa::types::InstKind::Global(name) = &inst.data {
                assert_ne!(
                    name, "PI",
                    "Global @PI should have been inlined, but survived in entry '{}'",
                    ep.name
                );
            }
        }
    }
}

// ============================================================================
// `--fill-holes`: type-hole default fill
// ============================================================================

/// Compile through TLC with `fill_holes = true`; return the resulting
/// `TlcTransformed` so tests can inspect `fill_hole_errors` and the
/// program shape.
fn compile_tlc_with_fill_holes(input: &str) -> crate::TlcTransformed {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let parsed = crate::Compiler::parse(input, &mut node_counter).expect("parse");
    let type_checked = parsed
        .desugar(&mut node_counter)
        .expect("desugar")
        .resolve(&mut module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    type_checked.to_tlc(&module_manager, true)
}

#[test]
fn fill_holes_numeric_scalars_compile_clean() {
    // Scalar holes (i32 / f32 / bool) default to 0 / 0.0 / false and
    // compile through with no fill-hole errors.
    for src in ["def x: i32 = ???", "def y: f32 = ???", "def z: bool = ???"] {
        let tlc = compile_tlc_with_fill_holes(src);
        assert!(
            tlc.0.fill_hole_errors.is_empty(),
            "scalar hole in `{}` should fill cleanly: {:?}",
            src,
            tlc.0.fill_hole_errors
        );
    }
}

#[test]
fn fill_holes_vec_compiles_clean() {
    let tlc = compile_tlc_with_fill_holes("def v: vec3f32 = ???");
    assert!(
        tlc.0.fill_hole_errors.is_empty(),
        "vec3 hole should fill cleanly: {:?}",
        tlc.0.fill_hole_errors
    );
}

#[test]
fn fill_holes_rejects_function_type() {
    let tlc = compile_tlc_with_fill_holes("def f: i32 -> i32 = ???");
    assert!(
        !tlc.0.fill_hole_errors.is_empty(),
        "function-typed hole should surface a fill-hole error"
    );
    let msg = format!("{:?}", tlc.0.fill_hole_errors[0]);
    assert!(
        msg.contains("function value") || msg.contains("Arrow"),
        "error should mention function type: {}",
        msg
    );
}

#[test]
fn fill_holes_respects_inferred_type_from_context() {
    // Hole's type is inferred from the enclosing context (array
    // element type here). Default-fill fires at the inferred type.
    let tlc = compile_tlc_with_fill_holes("def arr: [3]i32 = [1i32, ???, 3i32]");
    assert!(
        tlc.0.fill_hole_errors.is_empty(),
        "hole in i32 array should fill as i32 cleanly: {:?}",
        tlc.0.fill_hole_errors
    );
}

// =============================================================================
// Known-failing tests for higher-order-function defunctionalization gaps
// =============================================================================
//
// Both currently fail. Keep them as guards: when the underlying bugs in
// defunctionalization / SPIR-V lowering are fixed, these tests start
// passing and the cleanup will be obvious.

/// Bug 1: A closure with captured free variables, passed to a user-
/// defined HOF, produces SPIR-V that fails `spirv-val` with
/// `OpFunctionCall Function <id>'s parameter count does not match the
/// argument count`. Defunc/mono specializes both the HOF and the
/// closure body correctly (each gets the captured-env params it
/// needs), but the call-sites for the function-typed parameter inside
/// the HOF body still pass only the original arguments — the captured
/// env never gets threaded through.
///
/// Minimal repro: a 2-arg HOF that calls its f-arg twice, with a
/// closure capturing two scalars from the entry's params.
#[test]
fn hof_closure_with_captures_lowers_to_valid_spirv() {
    let src = r#"
def apply2(f: f32 -> f32, x0: f32, x1: f32) f32 = f(x0) + f(x1)

#[compute]
entry test(a: f32, b: f32) f32 =
  let g = |y: f32| y * y + a + b in
  apply2(g, 1.0f32, 2.0f32)
"#;
    let spirv = compile_to_spirv(src).expect("compile");
    assert_spirv_call_arities_match(&spirv);
}

/// Bug 2: An inline lambda with no captures, called from inside a
/// `map` body, panics in SPIR-V lowering with "BUG: Unknown type
/// reached lowering: Ignored". The HOF passed the lambda survives
/// type checking but the elaborated function's signature carries an
/// `Ignored` placeholder where the lambda's argument type should be —
/// presumably defunctionalization or the elaborator is leaving the
/// inner call's type unresolved.
///
/// Minimal repro: a HOF taking `f32 -> f32`, called inside a `map`
/// over `[]f32`, with a closure-free inline lambda.
#[test]
fn hof_no_capture_lambda_in_map_body_lowers_without_panic() {
    let src = r#"
def apply2(f: f32 -> f32, x0: f32, x1: f32) f32 = f(x0) + f(x1)

#[compute]
entry test(in_arr: []f32) []f32 =
  map(|x: f32| apply2(|y: f32| y * y, x, x + 1.0f32), in_arr)
"#;
    // catch_unwind because the bug surfaces as a panic in
    // spirv/mod.rs (not a Result::Err).
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compile_to_spirv(src)));
    let bytes = result.expect("compilation panicked — see Bug 2 docstring").expect("compile returned Err");
    assert_spirv_call_arities_match(&bytes);
}

// =============================================================================
// Sum-type lowering (Phase C) integration tests
// =============================================================================

/// Build a sum value with one constructor and select on it. Exercises
/// constructor-expression → flattened-tuple lowering and match →
/// tag-checked if-chain lowering, end-to-end through SPIR-V.
#[test]
fn sum_type_lowering_compiles_to_spirv() {
    let src = r#"
def pick(v: #left(f32) | #right(f32)) f32 =
    match v
    case #left(x) -> x + 1.0f32
    case #right(y) -> y * 2.0f32

#[fragment]
entry main() #[location(0)] vec4f32 =
    let a = pick(#left(0.5f32)) in
    let b = pick(#right(0.25f32)) in
    @[a, b, 0.0f32, 1.0f32]
"#;
    let spirv = compile_to_spirv(src).expect("sum-type program should compile to SPIR-V");
    assert_spirv_call_arities_match(&spirv);
}

/// Multi-payload constructor with mixed arities: `#point(f32, f32)`
/// and a nullary `#origin`. Verifies that the flattened-no-sharing
/// layout zero-fills dead slots in the nullary case.
#[test]
fn sum_type_multi_payload_compiles_to_spirv() {
    let src = r#"
def length_sq(p: #point(f32, f32) | #origin) f32 =
    match p
    case #point(x, y) -> x * x + y * y
    case #origin -> 0.0f32

#[fragment]
entry main() #[location(0)] vec4f32 =
    let a = length_sq(#point(3.0f32, 4.0f32)) in
    let b = length_sq(#origin) in
    @[a, b, 0.0f32, 1.0f32]
"#;
    let spirv = compile_to_spirv(src).expect("multi-payload sum should compile to SPIR-V");
    assert_spirv_call_arities_match(&spirv);
}

// =============================================================================
// Swizzle-with lowering (Phase C) integration tests
// =============================================================================

/// Plain `=` swizzle update: write `e` into v.yz, leaving v.x intact.
#[test]
fn swizzle_with_plain_assign_compiles_to_spirv() {
    let src = r#"
def update(v: vec3f32, e: vec2f32) vec3f32 = v with .yz = e

#[fragment]
entry main() #[location(0)] vec4f32 =
    let v = update(@[1.0f32, 2.0f32, 3.0f32], @[20.0f32, 30.0f32]) in
    @[v.x, v.y, v.z, 1.0f32]
"#;
    let spirv = compile_to_spirv(src).expect("plain swizzle-with should compile to SPIR-V");
    assert_spirv_call_arities_match(&spirv);
}

/// Compound `*=` swizzle update: vec2 × mat2 multiply, written into
/// v.yz. Exercises the binary-op path inside transform_vec_with.
#[test]
fn swizzle_with_compound_mul_compiles_to_spirv() {
    let src = r#"
def update(v: vec3f32, m: mat2f32) vec3f32 = v with .yz *= m

#[fragment]
entry main() #[location(0)] vec4f32 =
    let m: mat2f32 = @[[1.0f32, 0.0f32], [0.0f32, 1.0f32]] in
    let v = update(@[1.0f32, 2.0f32, 3.0f32], m) in
    @[v.x, v.y, v.z, 1.0f32]
"#;
    let spirv = compile_to_spirv(src).expect("compound swizzle-with should compile to SPIR-V");
    assert_spirv_call_arities_match(&spirv);
}

/// The full GLSL pattern from the original request: four chained
/// `with .swizzle *= mat2` rotations on a direction vector.
#[test]
fn swizzle_with_chained_rotations_compiles_to_spirv() {
    let src = r#"
def rot(a: f32) mat2f32 =
    let c = f32.cos(a) in
    let s = f32.sin(a) in
    @[[c, s], [0.0f32 - s, c]]

def transform(dir0: vec3f32, mx: f32, my: f32) vec3f32 =
    let d1 = dir0 with .yz *= rot(my) in
    let d2 = d1 with .xz *= rot(mx) in
    let d3 = d2 with .yz *= rot(my) in
    d3 with .xz *= rot(mx)

#[fragment]
entry main() #[location(0)] vec4f32 =
    let d = transform(@[0.0f32, 0.0f32, 1.0f32], 0.5f32, 0.3f32) in
    @[d.x, d.y, d.z, 1.0f32]
"#;
    let spirv = compile_to_spirv(src).expect("chained swizzle-with rotations should compile");
    assert_spirv_call_arities_match(&spirv);
}

/// Walk every `OpFunctionCall` in a SPIR-V module and assert each
/// call's argument count matches the called function's declared
/// parameter count. The arity-mismatch class of bug above produces
/// SPIR-V that round-trips through rspirv but fails this invariant.
fn assert_spirv_call_arities_match(spirv_words: &[u32]) {
    use rspirv::binary::parse_words;
    use rspirv::dr::{Loader, Operand};
    use rspirv::spirv::Op;

    let mut loader = Loader::new();
    parse_words(spirv_words, &mut loader).expect("parse spirv");
    let module = loader.module();

    // Map function id → declared parameter count.
    let mut arities: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for func in &module.functions {
        if let Some(def) = &func.def {
            if let Some(Operand::IdRef(_)) = def.result_id.map(Operand::IdRef) {}
            if let Some(id) = def.result_id {
                arities.insert(id, func.parameters.len());
            }
        }
    }

    // Walk every block's instructions for OpFunctionCall.
    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                if inst.class.opcode == Op::FunctionCall {
                    // Operands: [0] callee IdRef, [1..] argument IdRefs.
                    let callee = match inst.operands.first() {
                        Some(Operand::IdRef(id)) => *id,
                        _ => continue,
                    };
                    let arg_count = inst.operands.len() - 1;
                    let expected = match arities.get(&callee) {
                        Some(n) => *n,
                        None => continue, // external call (e.g. GlslExt)
                    };
                    assert_eq!(
                        arg_count, expected,
                        "OpFunctionCall to function %{} passes {} args but the function declares {} parameters",
                        callee, arg_count, expected
                    );
                }
            }
        }
    }
}
