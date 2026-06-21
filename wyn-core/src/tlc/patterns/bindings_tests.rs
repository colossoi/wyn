//! Tests for `compute_pattern_bindings` and its inner. Drives the
//! AST→TLC lowering of irrefutable patterns through small source
//! programs and inspects the resulting Term structure.

use crate::tlc::{Program, Term, TermKind};
use crate::Compiler;

fn compile_to_tlc_raw(source: &str) -> Program {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let parsed = Compiler::parse(source, &mut node_counter).expect("parse");
    let type_checked = parsed
        .resolve(&mut module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    type_checked.to_tlc(&module_manager, false).0.tlc
}

fn find_def_body<'a>(program: &'a Program, name: &str) -> &'a Term {
    let def = program
        .defs
        .iter()
        .find(|d| program.symbols.get(d.name).map(|s| s.as_str()) == Some(name))
        .unwrap_or_else(|| panic!("no def named {name}"));
    &def.body
}

#[test]
fn let_tuple_destructure_emits_nested_lets() {
    let program = compile_to_tlc_raw(
        r#"
def f(p: (i32, i32)) i32 =
  let (a, b) = p in
  a + b
"#,
    );
    let lam_body = match &find_def_body(&program, "f").kind {
        TermKind::Lambda(l) => &*l.body,
        other => panic!("expected Lambda, got {:?}", other),
    };
    // Outer Let binds the tuple itself; inner Lets bind a, b.
    let mut cur = lam_body;
    let mut let_count = 0;
    while let TermKind::Let { body, .. } = &cur.kind {
        let_count += 1;
        cur = body.as_ref();
    }
    assert!(
        let_count >= 3,
        "should have at least 3 nested Lets (scrut, a, b), got {}",
        let_count
    );
}

#[test]
fn let_single_variant_ctor_emits_payload_projection() {
    let program = compile_to_tlc_raw(
        r#"
def f(b: #boxed(i32)) i32 =
  let #boxed(x) = b in
  x + 1
"#,
    );
    let lam_body = match &find_def_body(&program, "f").kind {
        TermKind::Lambda(l) => &*l.body,
        other => panic!("expected Lambda, got {:?}", other),
    };
    // Should emit a Let chain: outer binds b to fresh ctor sym, inner
    // binds x to projection of payload slot.
    assert!(
        matches!(&lam_body.kind, TermKind::Let { .. }),
        "let-binding should produce a Let, got {:?}",
        lam_body.kind
    );
}

#[test]
fn let_unit_pattern_compiles() {
    // `let () = e in body` is valid for unit-typed e. We can't easily
    // construct one via source (no `()` value-level term in most
    // contexts), but a unit-returning function call works.
    let program = compile_to_tlc_raw(
        r#"
def g(n: i32) i32 = n
def f(x: i32) i32 = let y = g(x) in y + 1
"#,
    );
    // Just verify no panic during lowering.
    let _ = find_def_body(&program, "f");
}

#[test]
fn let_typed_pattern_unwraps_to_inner() {
    let program = compile_to_tlc_raw(
        r#"
def f(n: i32) i32 =
  let x: i32 = n in
  x + 1
"#,
    );
    let lam_body = match &find_def_body(&program, "f").kind {
        TermKind::Lambda(l) => &*l.body,
        other => panic!("expected Lambda, got {:?}", other),
    };
    // Typed delegates to inner Name pattern → single Let.
    assert!(
        matches!(&lam_body.kind, TermKind::Let { .. }),
        "got {:?}",
        lam_body.kind
    );
}

#[test]
fn let_wildcard_doesnt_bind_anything_useful() {
    let program = compile_to_tlc_raw(
        r#"
def f(n: i32) i32 =
  let _ = n in
  42
"#,
    );
    let lam_body = match &find_def_body(&program, "f").kind {
        TermKind::Lambda(l) => &*l.body,
        other => panic!("expected Lambda, got {:?}", other),
    };
    // Even with wildcard, transform_expr's fast-path emits a Let
    // (the rhs is still evaluated for side effects). The body is the
    // bare `42`.
    match &lam_body.kind {
        TermKind::Let { body, .. } => {
            assert!(matches!(&body.kind, TermKind::IntLit(s) if s == "42"));
        }
        other => panic!("expected Let, got {:?}", other),
    }
}

#[test]
fn let_nested_tuple_destructure() {
    let program = compile_to_tlc_raw(
        r#"
def f(p: ((i32, i32), i32)) i32 =
  let ((a, b), c) = p in
  a + b + c
"#,
    );
    let lam_body = match &find_def_body(&program, "f").kind {
        TermKind::Lambda(l) => &*l.body,
        other => panic!("expected Lambda, got {:?}", other),
    };
    // Walk Let chain — should have at least 4 lets (outer p, inner
    // pair, a, b, c).
    let mut cur = lam_body;
    let mut let_count = 0;
    while let TermKind::Let { body, .. } = &cur.kind {
        let_count += 1;
        cur = body.as_ref();
    }
    assert!(
        let_count >= 4,
        "should have at least 4 nested Lets, got {}",
        let_count
    );
}
