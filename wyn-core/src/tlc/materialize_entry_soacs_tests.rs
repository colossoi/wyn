//! Pass-level tests for `materialize_entry_soacs` — does it inline the *right*
//! producer calls into entries, and leave per-element-nested ones alone?
//!
//! We run the pass on a program compiled to the pre-defunctionalize stage
//! (`lift_gathers`): there the producer calls are still present (no inlining
//! yet) and SOAC operator lambdas are still inline (so the only `Var`-callee
//! `App`s are user helper calls). Region concreteness — which the pass needs at
//! its real post-monomorphize position — is exercised by the end-to-end tests,
//! not here; this fixes the *materialization decision*.

use crate::tlc::{DefMeta, Program, Term, TermKind, VarRef};

/// Compile `src` to the `lift_gathers` stage and run the pass on it.
fn materialize(src: &str) -> Program {
    crate::dyn_pipeline::compile_to_tlc_program(src)
}

fn entry_body<'a>(program: &'a Program, name: &str) -> &'a Term {
    program
        .defs
        .iter()
        .find(|d| {
            matches!(d.meta, DefMeta::EntryPoint(_))
                && program.symbols.get(d.name).map(|s| s.as_str()) == Some(name)
        })
        .map(|d| &d.body)
        .unwrap_or_else(|| panic!("no entry named {name}"))
}

/// Count user function-call applications (`App` whose callee is a
/// `Var(Symbol)`) anywhere in `term`. Pre-defunctionalize, SOAC operator
/// lambdas are inline (they contribute none), so this counts exactly the
/// surviving helper calls — a materialized producer leaves zero, an
/// un-inlined per-element helper call leaves it standing.
fn helper_calls(term: &Term) -> usize {
    let mut n = 0;
    if let TermKind::App { func, .. } = &term.kind {
        if matches!(func.kind, TermKind::Var(VarRef::Symbol(_))) {
            n += 1;
        }
    }
    term.for_each_child(&mut |c| n += helper_calls(c));
    n
}

/// True iff a SOAC is directly bound in the entry's top-level let-chain / tail /
/// output-slot value — i.e. exposed at the entry boundary (not inside a lambda).
fn top_level_soac(term: &Term) -> bool {
    match &term.kind {
        TermKind::Lambda(lam) => top_level_soac(&lam.body),
        TermKind::Let { rhs, body, .. } => {
            matches!(rhs.kind, TermKind::Soac(_)) || top_level_soac(rhs) || top_level_soac(body)
        }
        TermKind::OutputSlotStore { value, .. } => {
            matches!(value.kind, TermKind::Soac(_)) || top_level_soac(value)
        }
        TermKind::Soac(_) => true,
        _ => false,
    }
}

/// 1. `entry e(xs) = helper(xs)` — tail call → materializes.
#[test]
fn tail_call_materializes() {
    let p = materialize(
        "\
def stencil(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0i32, xs)
#[compute]
entry e(xs: []i32) []i32 = stencil(xs)
",
    );
    let body = entry_body(&p, "e");
    assert_eq!(helper_calls(body), 0, "stencil call must be inlined: {body:?}");
    assert!(
        top_level_soac(body),
        "the scan must be exposed at the entry boundary"
    );
}

/// 2. `entry e(xs) = let y = helper(xs) in y` — let-bound, Var-follow tail.
#[test]
fn let_bound_call_materializes() {
    let p = materialize(
        "\
def stencil(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0i32, xs)
#[compute]
entry e(xs: []i32) []i32 = let y = stencil(xs) in y
",
    );
    let body = entry_body(&p, "e");
    assert_eq!(helper_calls(body), 0, "stencil call must be inlined: {body:?}");
    assert!(
        top_level_soac(body),
        "the scan must be exposed at the entry boundary"
    );
}

/// 3. `entry e(xs) = helper2(helper1(xs))` — fixpoint materializes both (the
/// arg-position producer is reached via the let `build_inline_lets` introduces).
#[test]
fn chained_calls_materialize() {
    let p = materialize(
        "\
def s1(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0i32, xs)
def s2(xs: []i32) []i32 = scan(|a: i32, b: i32| a * b, 1i32, xs)
#[compute]
entry e(xs: []i32) []i32 = s2(s1(xs))
",
    );
    let body = entry_body(&p, "e");
    assert_eq!(
        helper_calls(body),
        0,
        "both scan helpers must be inlined: {body:?}"
    );
    assert!(
        top_level_soac(body),
        "a scan must be exposed at the entry boundary"
    );
}

// Removed: `per_element_call_is_not_materialized` tested that
// `materialize_entry_soacs` keeps a per-element helper call inside its
// map lambda (instead of hoisting the helper's inner reduce to the entry
// boundary). With `force_inline_soac_helpers` running before
// `materialize_entry_soacs`, any SOAC-bodied helper is inlined first,
// so the fixture this test depended on (a surviving helper call with
// `reduce` in its body) is unreachable from any Wyn source. The
// protection logic itself still lives in `materialize_entry_soacs` and
// is correct; it just has no Wyn-source way to be exercised.

/// 5. Helper with a local `let` captured by its SOAC → materializes with the
/// captured binding carried along (no dangling var); the case the surgical
/// summary-rebuild alternative would have broken.
#[test]
fn helper_local_capture_materializes() {
    let p = materialize(
        "\
def stencil(xs: []i32) []i32 =
  let w = 5i32 in
  scan(|a: i32, b: i32| a + b + w, 0i32, xs)
#[compute]
entry e(xs: []i32) []i32 = stencil(xs)
",
    );
    let body = entry_body(&p, "e");
    assert_eq!(helper_calls(body), 0, "stencil call must be inlined: {body:?}");
    assert!(
        top_level_soac(body),
        "the scan must be exposed at the entry boundary"
    );
    // The captured `w = 5` must have come along as an entry-level binding; if it
    // hadn't, the scan's operator would reference an unbound symbol.
    assert!(
        contains_int_lit(body, "5"),
        "the helper-local `let w = 5` must be materialized into the entry: {body:?}",
    );
}

fn contains_int_lit(term: &Term, lit: &str) -> bool {
    if matches!(&term.kind, TermKind::IntLit(s) if s == lit) {
        return true;
    }
    let mut found = false;
    term.for_each_child(&mut |c| found |= contains_int_lit(c, lit));
    found
}
