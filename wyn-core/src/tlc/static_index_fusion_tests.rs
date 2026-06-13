//! Tests for static-index fusion: `map(f, src)[k]` (literal `k`) over a
//! directly-nested elementwise producer collapses to a scalar element, while a
//! `Var`-reached (let-bound) producer is left untouched.

use super::*;
use crate::Compiler;
use crate::tlc::DefMeta;

/// Drive a source string to the early exposed-producer TLC slot where
/// `static_index_fusion::run` fires and return the program.
fn exposed(source: &str) -> crate::tlc::Program {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let parsed = Compiler::parse(source, &mut node_counter).expect("parse");
    let tc = parsed
        .resolve(&mut module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    let early = tc
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
        .expose_entry_producer_helpers();
    early.0.tlc.clone()
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

    let fused = super::run(program);
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
    let fused = super::run(program.clone());
    let after = entry_body(&fused);
    assert_eq!(
        format!("{before:?}"),
        format!("{after:?}"),
        "a let-bound (Var-reached) producer must be left untouched by static-index fusion"
    );
}
