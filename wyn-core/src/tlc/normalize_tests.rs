//! Tests for `tlc::normalize` — SOAC lifting and let-flattening.
//!
//! `tlc::normalize::normalize` (the SOAC-lift + let-flatten) runs at the start
//! of `tlc::fusion::run`, i.e. inside `fuse_maps` — NOT in `normalize_soacs`
//! (that is `tlc::soa::run`). So we compile through `fuse_maps` and then assert
//! the flattening invariant: no `let` binding has a `let`-expression as its rhs.
//! Every binding therefore lives on a single chain, which is what the
//! (chain-walking) fusion driver needs to see producer/consumer edges. The
//! structural check is independent of downstream lowering, which has unrelated
//! gaps for some of these shapes.

use crate::tlc::{Program, Term, TermKind};
use crate::Compiler;

/// Compile a Wyn source string up to (and including) `fuse_maps` — where
/// `tlc::normalize::normalize` (SOAC-lift + let-flatten) actually runs —
/// returning the TLC `Program` for structural inspection.
fn normalized_program(source: &str) -> Program {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    Compiler::parse(source, &mut node_counter)
        .expect("parse")
        .resolve(&mut module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check")
        .to_tlc(&module_manager, false)
        .pin_entry_regions()
        .expect("pin_entry_regions")
        .partial_eval()
        .normalize_soacs()
        .force_inline_soac_helpers()
        .fuse_maps()
        .0
        .tlc
}

/// Assert no `let` in the program has a `let` as its rhs.
fn assert_no_nested_let_rhs(program: &Program) {
    fn walk(t: &Term) {
        if let TermKind::Let { rhs, .. } = &t.kind {
            assert!(
                !matches!(rhs.kind, TermKind::Let { .. }),
                "a let binding has a let-expression as its rhs — normalize did not flatten it"
            );
        }
        t.for_each_child(&mut |c| walk(c));
    }
    for def in &program.defs {
        walk(&def.body);
    }
}

#[test]
fn flattens_soac_lifted_out_of_a_binop() {
    // `reduce(..) * x` makes `normalize` lift the reduce out of the binop. Before
    // flattening that left `let r = (let _anf = reduce(..) in _anf * x)` — a let
    // whose rhs is a let, invisible to the top-level-only fusion driver. The lift
    // happens inside `normalize` (after `partial_eval`), so this exercises the
    // flattening specifically.
    let src = r#"
def g(x: f32) f32 =
  let r = reduce(|p: f32, q: f32| p + q, 0.0, [1.0, 2.0, 3.0, 4.0]) * x in r

#[compute]
entry e(#[storage(set=2, binding=0, access=read)] a: []f32) []f32 =
  map(|x: f32| g(x), a[0..8])
"#;
    assert_no_nested_let_rhs(&normalized_program(src));
}

#[test]
fn flattens_source_level_nested_let() {
    // A `let` nested directly in source (its rhs is a `let`) must also be
    // un-nested onto the top-level chain.
    let src = r#"
#[compute]
entry e(#[storage(set=2, binding=0, access=read)] a: []f32) []f32 =
  map(|x: f32| (let y = x * 2.0 in let z = (let w = y + 1.0 in w * 3.0) in z), a[0..8])
"#;
    assert_no_nested_let_rhs(&normalized_program(src));
}
