//! Source-level integration-style tests for the `rep_specialize` pass.
//!
//! Every case is a small program that previously tripped the
//! backend-boundary verifier `egir::verify_no_abstract` (when
//! `filter(...)` flowed into a non-inlined user helper) and now must
//! compile end-to-end to SPIR-V cleanly.

use crate::compile_thru_spirv;

#[test]
fn filter_then_user_helper_static_capacity() {
    // Static-capacity filter input → producer-side Bounded(N).
    // The consumer `center` carries Abstract at its arr param and
    // would trip the verifier without rep_specialize.
    let src = r#"
def sum<[n]>(arr: [n]i32) i32 = reduce(|a: i32, b: i32| a + b, 0, arr)
def center(arr: []i32) i32 = sum(arr)
#[compute]
entry tick() i32 =
  let kept = filter(|x: i32| x > 0, [1, -2, 3, -4, 5]) in
  center(kept)
"#;
    compile_thru_spirv(src)
        .expect("filter(static-capacity) → user helper must compile after rep_specialize");
}

#[test]
fn filter_then_user_helper_runtime_size() {
    // Runtime-sized filter input → producer-side View. The
    // specialized consumer's variant becomes View; the size slot
    // stays Skolem (View has runtime length, no static capacity).
    let src = r#"
def sum<[n]>(arr: [n]i32) i32 = reduce(|a: i32, b: i32| a + b, 0, arr)
def center(arr: []i32) i32 = sum(arr)
#[compute]
entry tick(#[storage(set=2, binding=0, access=read)] xs: []i32) i32 =
  let kept = filter(|x: i32| x > 0, xs) in
  center(kept)
"#;
    compile_thru_spirv(src).expect("filter(runtime) → user helper must compile after rep_specialize");
}

#[test]
fn filter_into_nested_helper_specializes_recursively() {
    // `center` calls `sum`. Both have Abstract-typed params; both
    // must specialize. (The `center` body's nested call to `sum`
    // re-triggers rep_specialize via the recursive `rewrite_def_body`
    // on the cloned spec body.)
    let src = r#"
def sum<[n]>(arr: [n]i32) i32 = reduce(|a: i32, b: i32| a + b, 0, arr)
def center(arr: []i32) i32 =
  let s = sum(arr) in
  s + sum(arr)
#[compute]
entry tick() i32 =
  let kept = filter(|x: i32| x > 0, [1, -2, 3, -4, 5, -6]) in
  center(kept)
"#;
    compile_thru_spirv(src).expect("nested user helpers must specialize transitively");
}

/// Negative test: run the same program through the
/// `parallelize_soacs` shortcut on `TlcEntrySoacsMaterialized` (which
/// bypasses rep_specialize). The verifier in `egir::verify_no_abstract`
/// must fire — this both proves the shortcut path *is* lethal without
/// rep_specialize and pins that the verifier catches the leak.
#[test]
fn verifier_fires_when_rep_specialize_is_bypassed() {
    use crate::Compiler;

    let src = r#"
def sum<[n]>(arr: [n]i32) i32 = reduce(|a: i32, b: i32| a + b, 0, arr)
def center(arr: []i32) i32 =
  let s = sum(arr) in
  s + sum(arr)
#[compute]
entry tick() i32 =
  let kept = filter(|x: i32| x > 0, [1, -2, 3, -4, 5, -6]) in
  center(kept)
"#;

    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    // Run the full TLC pipeline except `rep_specialize` — go directly
    // through `parallelize_soacs` on `TlcEntrySoacsMaterialized`. The
    // resulting program still has `Abstract` on `sum`/`center`'s param,
    // so the verifier in `lower(...)` must reject it.
    let parsed = Compiler::parse(src, &mut node_counter).expect("parse");
    let elaborated = parsed.elaborate_modules(&mut module_manager, &mut node_counter).expect("elaborate");
    let resolved = elaborated.resolve(&module_manager).expect("resolve").fold_ast_constants();
    let type_checked = resolved.type_check(&mut module_manager).expect("typecheck");
    let tlc = type_checked
        .to_tlc(&module_manager, false)
        .pin_entry_regions()
        .expect("pin_entry_regions")
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .apply_ownership()
        .expect("apply_ownership")
        .normalize_outputs()
        .expect("normalize_outputs")
        .lift_gathers()
        .defunctionalize()
        .monomorphize()
        .fold_generated_lambdas()
        .inline_small()
        .materialize_entry_soacs()
        // ← shortcut skipping rep_specialize:
        .parallelize_soacs(false)
        .expect("parallelize_soacs")
        .filter_reachable();

    let ssa = tlc
        .to_egraph()
        .expect("to_egraph")
        .expand_soacs(true)
        .materialize()
        .optimize_skeleton()
        .elaborate();

    let result = ssa.lower();
    let err = result.err().expect("verifier must reject Abstract reaching the backend");
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("abstract") && msg.contains("reached backend"),
        "expected verifier error naming Abstract reaching the backend, got: {msg}"
    );
}

#[test]
fn non_filter_let_binding_does_not_trigger_spec() {
    // Control: a let-bound value that isn't a filter result has no
    // producer-derived variant, so rep_specialize never fires. The
    // verifier passes anyway because no Abstract surfaces.
    let src = r#"
def double<[n]>(arr: [n]i32) i32 = reduce(|a: i32, b: i32| a + b, 0, arr)
#[compute]
entry tick() i32 =
  let xs = [1, 2, 3, 4] in
  double(xs)
"#;
    compile_thru_spirv(src).expect("non-filter let binding compiles unchanged");
}
