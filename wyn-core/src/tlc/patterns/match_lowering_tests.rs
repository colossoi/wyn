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

// =========================================================================
// Additional coverage
// =========================================================================

#[test]
fn match_three_arm_chain_has_two_ifs_then_bare_last() {
    let program = compile_to_tlc_raw(
        r#"
def f(t: #a | #b | #c) i32 =
  match t
  case #a -> 1
  case #b -> 2
  case #c -> 3
"#,
    );
    let lam_body = match &find_def_body(&program, "f").kind {
        TermKind::Lambda(l) => &*l.body,
        other => panic!("expected Lambda, got {:?}", other),
    };
    let outer_if = unwrap_match_let(lam_body);
    let (then1, else1) = match &outer_if.kind {
        TermKind::If {
            then_branch,
            else_branch,
            ..
        } => (then_branch.as_ref(), else_branch.as_ref()),
        other => panic!("expected outer If, got {:?}", other),
    };
    assert!(matches!(&then1.kind, TermKind::IntLit(s) if s == "1"));
    let (then2, else2) = match &else1.kind {
        TermKind::If {
            then_branch,
            else_branch,
            ..
        } => (then_branch.as_ref(), else_branch.as_ref()),
        other => panic!("expected nested If, got {:?}", other),
    };
    assert!(matches!(&then2.kind, TermKind::IntLit(s) if s == "2"));
    assert!(matches!(&else2.kind, TermKind::IntLit(s) if s == "3"));
}

#[test]
fn match_on_tuple_scrutinee() {
    let program = compile_to_tlc_raw(
        r#"
def f(p: (i32, bool)) i32 =
  match p
  case (0, _)    -> 1
  case (_, true) -> 2
  case _         -> 3
"#,
    );
    let lam_body = match &find_def_body(&program, "f").kind {
        TermKind::Lambda(l) => &*l.body,
        other => panic!("expected Lambda, got {:?}", other),
    };
    let outer_if = unwrap_match_let(lam_body);
    assert!(
        matches!(&outer_if.kind, TermKind::If { .. }),
        "outer should be If"
    );
}

#[test]
fn match_with_name_arm_binds_scrutinee_name() {
    let program = compile_to_tlc_raw(
        r#"
def f(n: i32) i32 =
  match n
  case 0 -> 0
  case x -> x + 1
"#,
    );
    let lam_body = match &find_def_body(&program, "f").kind {
        TermKind::Lambda(l) => &*l.body,
        other => panic!("expected Lambda, got {:?}", other),
    };
    let outer_if = unwrap_match_let(lam_body);
    let else_b = match &outer_if.kind {
        TermKind::If { else_branch, .. } => else_branch.as_ref(),
        other => panic!("expected If, got {:?}", other),
    };
    match &else_b.kind {
        TermKind::Let { body, .. } => {
            assert!(
                matches!(&body.kind, TermKind::App { .. }),
                "Let body should be App, got {:?}",
                body.kind
            );
        }
        other => panic!("expected Let for name binding, got {:?}", other),
    }
}

#[test]
fn match_scrutinee_bound_to_fresh_symbol() {
    let program = compile_to_tlc_raw(
        r#"
def f(n: i32) i32 =
  match n
  case 0 -> 0
  case _ -> 1
"#,
    );
    let lam_body = match &find_def_body(&program, "f").kind {
        TermKind::Lambda(l) => &*l.body,
        other => panic!("expected Lambda, got {:?}", other),
    };
    let outer_let_name = match &lam_body.kind {
        TermKind::Let { name, .. } => *name,
        other => panic!("expected outer Let, got {:?}", other),
    };
    let resolved = program.symbols.get(outer_let_name).map(|s| s.as_str()).unwrap_or("?");
    assert!(
        resolved.starts_with("_w_match_scrut_"),
        "scrutinee should be bound to a fresh _w_match_scrut_* symbol, got '{}'",
        resolved
    );
}

#[test]
fn match_with_constructor_payload_binding_emits_let() {
    let program = compile_to_tlc_raw(
        r#"
def f(t: #boxed(i32) | #empty) i32 =
  match t
  case #boxed(x) -> x + 1
  case #empty    -> 0
"#,
    );
    let lam_body = match &find_def_body(&program, "f").kind {
        TermKind::Lambda(l) => &*l.body,
        other => panic!("expected Lambda, got {:?}", other),
    };
    let outer_if = unwrap_match_let(lam_body);
    let then_b = match &outer_if.kind {
        TermKind::If { then_branch, .. } => then_branch.as_ref(),
        other => panic!("expected If, got {:?}", other),
    };
    assert!(
        matches!(&then_b.kind, TermKind::Let { .. }),
        "then should be Let for payload x: {:?}",
        then_b.kind
    );
}

#[test]
fn match_constructor_arm_cond_is_tag_equality() {
    let program = compile_to_tlc_raw(
        r#"
def f(t: #yes | #no) i32 =
  match t
  case #yes -> 1
  case #no  -> 0
"#,
    );
    let lam_body = match &find_def_body(&program, "f").kind {
        TermKind::Lambda(l) => &*l.body,
        other => panic!("expected Lambda, got {:?}", other),
    };
    let outer_if = unwrap_match_let(lam_body);
    let cond = match &outer_if.kind {
        TermKind::If { cond, .. } => cond.as_ref(),
        other => panic!("expected If, got {:?}", other),
    };
    match &cond.kind {
        TermKind::App { func, args } => {
            assert!(
                matches!(&func.kind, TermKind::BinOp(_)),
                "func should be BinOp, got {:?}",
                func.kind
            );
            assert_eq!(args.len(), 2, "== takes two args");
            assert!(
                matches!(&args[1].kind, TermKind::IntLit(_)),
                "rhs should be tag literal"
            );
        }
        other => panic!("expected App for cond, got {:?}", other),
    }
}

/// Regression test for the eager-AND lowering of nested constructor
/// patterns. With `match outer case #pair(#left(_), ...) -> ...`, the
/// lowering eagerly evaluates the outer tag check AND the inner
/// payload tag check; on a value whose outer tag mismatches, the
/// inner projection reads the blank-filled dead slot. Correctness
/// depends on blank-fill (commits f8131a5/f3d2fc7) guaranteeing dead
/// payload slots carry well-typed zero values, so an inner test
/// against a non-zero tag is `false` and the AND structurally yields
/// `false` — the arm correctly skips and the chain falls through.
///
/// This test pins the IR shape end-to-end: the first-arm condition
/// must be an AND of the outer-tag test and the inner-tag test,
/// confirming the eager-AND lowering is in place and not being
/// silently regressed to short-circuit form (which would only paper
/// over the blank-fill dependency).
#[test]
fn nested_ctor_with_outer_mismatch_emits_eager_and_chain() {
    let program = compile_to_tlc_raw(
        r#"
def go(t: #pair(#left(i32) | #right(i32)) | #solo(i32)) i32 =
  match t
  case #pair(#left(x))  -> x + 100
  case #pair(#right(y)) -> y + 200
  case #solo(z)         -> z + 300
"#,
    );
    let lam_body = match &find_def_body(&program, "go").kind {
        TermKind::Lambda(l) => &*l.body,
        other => panic!("expected Lambda, got {:?}", other),
    };
    let outer_if = unwrap_match_let(lam_body);
    assert!(
        matches!(&outer_if.kind, TermKind::If { .. }),
        "outer should be If for #pair(#left) test, got {:?}",
        outer_if.kind
    );
    // First-arm cond: AND of outer-tag test and inner-tag test.
    let first_cond = match &outer_if.kind {
        TermKind::If { cond, .. } => cond.as_ref(),
        _ => unreachable!(),
    };
    match &first_cond.kind {
        TermKind::App { func, .. } => {
            assert!(
                matches!(&func.kind, TermKind::BinOp(op) if op.op == "&&"),
                "first-arm cond should be `outer_tag == ... && inner_tag == ...`, got {:?}",
                func.kind
            );
        }
        other => panic!("expected App for AND cond, got {:?}", other),
    }
}
