//! Source-level integration-style tests for the `rep_specialize` pass.
//!
//! These programs carry `filter(...)` results through user helpers and assert
//! that representation specialization eliminates every abstract array before
//! the EGIR backend boundary.

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
