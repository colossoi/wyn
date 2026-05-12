//! Tests for general match lowering. The shape of the produced TLC
//! is pinned down: a `Let` binding the scrutinee, then an `If`-chain
//! (or bare body when there's a single arm or the last arm is
//! reached). End-to-end behavioral coverage happens via the
//! testfile suite.

use crate::Compiler;
use crate::tlc::{Program, Term, TermKind};

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

/// Match expressions lower to `Let scrut = <expr> in <chain>`. Return
/// the chain body.
fn unwrap_match_let(body: &Term) -> &Term {
    match &body.kind {
        TermKind::Let { body, .. } => body,
        other => panic!("expected outer Let from match lowering, got {:?}", other),
    }
}

#[test]
fn match_on_bool_two_arms() {
    // Bool scrutinee, two literal arms — exhaustive on bool.
    let program = compile_to_tlc_raw(
        r#"
def f(b: bool) i32 =
  match b
  case true  -> 1
  case false -> 2
"#,
    );
    let body = find_def_body(&program, "f");
    // Body is a Lambda whose body is the match.
    let lam_body = match &body.kind {
        TermKind::Lambda(l) => &*l.body,
        other => panic!("expected Lambda, got {:?}", other),
    };
    let chain = unwrap_match_let(lam_body);
    // Chain shape: If(cond, then_branch=arm1, else_branch=arm2-body).
    let (then_b, else_b) = match &chain.kind {
        TermKind::If {
            then_branch,
            else_branch,
            ..
        } => (then_branch.as_ref(), else_branch.as_ref()),
        other => panic!("expected If chain, got {:?}", other),
    };
    assert!(matches!(&then_b.kind, TermKind::IntLit(s) if s == "1"));
    assert!(matches!(&else_b.kind, TermKind::IntLit(s) if s == "2"));
}

#[test]
fn match_with_wildcard_catchall() {
    // Wildcard last arm: should be the bare else body, no condition.
    let program = compile_to_tlc_raw(
        r#"
def f(n: i32) i32 =
  match n
  case 0 -> 100
  case _ -> 200
"#,
    );
    let body = find_def_body(&program, "f");
    let lam_body = match &body.kind {
        TermKind::Lambda(l) => &*l.body,
        other => panic!("expected Lambda, got {:?}", other),
    };
    let chain = unwrap_match_let(lam_body);
    // Outer If: cond = (n == 0), then = 100, else = 200.
    let (cond, then_b, else_b) = match &chain.kind {
        TermKind::If {
            cond,
            then_branch,
            else_branch,
        } => (cond.as_ref(), then_branch.as_ref(), else_branch.as_ref()),
        other => panic!("expected If, got {:?}", other),
    };
    // cond is `scrut == 0`: an App over BinOp(==).
    assert!(
        matches!(&cond.kind, TermKind::App { .. }),
        "cond shape: {:?}",
        cond.kind
    );
    assert!(matches!(&then_b.kind, TermKind::IntLit(s) if s == "100"));
    assert!(matches!(&else_b.kind, TermKind::IntLit(s) if s == "200"));
}

#[test]
fn match_with_name_binding_arm() {
    // Name-pattern catch-all binds the scrutinee. Body uses the binding.
    let program = compile_to_tlc_raw(
        r#"
def f(n: i32) i32 =
  match n
  case 0 -> 0
  case x -> x + 1
"#,
    );
    let body = find_def_body(&program, "f");
    let lam_body = match &body.kind {
        TermKind::Lambda(l) => &*l.body,
        other => panic!("expected Lambda, got {:?}", other),
    };
    let chain = unwrap_match_let(lam_body);
    // else_b must be a Let binding `x` (the name pattern).
    let else_b = match &chain.kind {
        TermKind::If { else_branch, .. } => else_branch.as_ref(),
        other => panic!("expected If, got {:?}", other),
    };
    assert!(
        matches!(&else_b.kind, TermKind::Let { .. }),
        "else_b should be Let binding x: {:?}",
        else_b.kind
    );
}

#[test]
fn match_constructor_two_arms_no_payload() {
    // Two-variant sum, exhaustive coverage.
    let program = compile_to_tlc_raw(
        r#"
def f(t: #yes | #no) i32 =
  match t
  case #yes -> 1
  case #no  -> 0
"#,
    );
    let body = find_def_body(&program, "f");
    let lam_body = match &body.kind {
        TermKind::Lambda(l) => &*l.body,
        other => panic!("expected Lambda, got {:?}", other),
    };
    let chain = unwrap_match_let(lam_body);
    // Outer If: cond tests tag against #yes (tag=0).
    let (then_b, else_b) = match &chain.kind {
        TermKind::If {
            then_branch,
            else_branch,
            ..
        } => (then_branch.as_ref(), else_branch.as_ref()),
        other => panic!("expected If, got {:?}", other),
    };
    assert!(matches!(&then_b.kind, TermKind::IntLit(s) if s == "1"));
    assert!(matches!(&else_b.kind, TermKind::IntLit(s) if s == "0"));
}

#[test]
fn match_single_arm_wildcard_elides_test() {
    // Single wildcard arm: no If, just the body bound by Let scrut.
    let program = compile_to_tlc_raw(
        r#"
def f(n: i32) i32 =
  match n
  case _ -> 42
"#,
    );
    let body = find_def_body(&program, "f");
    let lam_body = match &body.kind {
        TermKind::Lambda(l) => &*l.body,
        other => panic!("expected Lambda, got {:?}", other),
    };
    let chain = unwrap_match_let(lam_body);
    // No If at all — bare body.
    assert!(
        matches!(&chain.kind, TermKind::IntLit(s) if s == "42"),
        "got {:?}",
        chain.kind
    );
}
