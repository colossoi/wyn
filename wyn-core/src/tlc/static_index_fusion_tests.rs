//! Tests for static-index fusion: `map(f, src)[k]` (literal `k`) over a
//! directly-nested elementwise producer collapses to a scalar element, while a
//! `Var`-reached (let-bound) producer is left untouched.

use super::*;
use crate::tlc::DefMeta;

/// Drive a source string to the exposed-producer slot (after
/// `expose_entry_producer_helpers`, before `fuse_static_indices`) where
/// `static_index_fusion::run` fires, and return the program — so the unfused
/// `map(..)[k]` is still present for the pass under test to remove.
fn exposed(source: &str) -> crate::tlc::Program {
    crate::test_pipeline::compile_thru_expose_producers(source)
}

/// The entry body, after peeling outer parameter `Lambda`s.
fn entry_body(program: &crate::tlc::Program) -> Term {
    let def =
        program.defs.iter().find(|d| matches!(d.meta, DefMeta::EntryPoint(_))).expect("an entry point");
    let mut body = &def.body;
    while let TermKind::Lambda(l) = &body.kind {
        body = &l.body;
    }
    body.clone()
}

/// True if any subterm is an `Index` whose `array` is (through enclosing `let`s)
/// a `Soac(Map)` — i.e. an unfused `map(..)[k]`.
fn has_index_into_map(term: &Term) -> bool {
    let mut found = false;
    fn array_is_map(mut t: &Term) -> bool {
        loop {
            match &t.kind {
                TermKind::Let { body, .. } => t = body,
                TermKind::Soac(SoacOp::Map { .. }) => return true,
                _ => return false,
            }
        }
    }
    fn walk(t: &Term, found: &mut bool) {
        if let TermKind::Index { array, .. } = &t.kind {
            if array_is_map(array) {
                *found = true;
            }
        }
        t.clone().map_children(&mut |c| {
            walk(&c, found);
            c
        });
    }
    walk(term, &mut found);
    found
}

#[test]
fn constant_index_into_inlined_map_is_fused_away() {
    // `g` inlines to `map(|i| f32.i32(i), 0..<256)`; `g(256)[3]` is a directly
    // nested `Index(.. Map .., 3)` that fusion must remove.
    let program = exposed(
        r#"
def g(n: i32) []f32 = map(|i: i32| f32.i32(i), 0i32 ..< n)
#[compute]
entry e() [1]f32 = [g(256)[3]]
"#,
    );
    let before = entry_body(&program);
    assert!(
        has_index_into_map(&before),
        "precondition: an unfused map(..)[3] is present"
    );

    let mut fused = program;
    super::run(&mut fused);
    let after = entry_body(&fused);
    assert!(
        !has_index_into_map(&after),
        "constant index into a nested map must be fused away"
    );
}

#[test]
fn constant_index_into_let_bound_producer_is_left_for_gather() {
    // `ys` is let-bound and read via `Var(ys)`, so the index is `Index(Var, 3)`,
    // not `Index(Map, 3)`. Fusion must NOT fire (multi-use producers belong to
    // the gather path); the body is unchanged.
    let program = exposed(
        r#"
#[compute]
entry e(xs: []i32) i32 =
    let ys = map(|x: i32| x + 1, xs) in
    ys[3] + ys[5]
"#,
    );
    let before = entry_body(&program);
    let mut fused = program.clone();
    super::run(&mut fused);
    let after = entry_body(&fused);
    assert_eq!(
        format!("{before:?}"),
        format!("{after:?}"),
        "a let-bound (Var-reached) producer must be left untouched by static-index fusion"
    );
}
