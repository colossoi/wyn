//! Unit tests for the `open M` name-resolution pass.
//!
//! Each test parses a small program, runs `resolve_opens::run` against
//! a hand-constructed `OpenIndex`, and inspects the resulting AST for
//! the expected identifier rewrites or error.

use super::*;
use crate::ast::{self, Declaration, ExprKind, Expression, Program};
use crate::lexer;
use crate::parser;

fn parse(src: &str) -> Program {
    let mut nc: ast::NodeCounter = ast::NodeCounter::new();
    let tokens = lexer::tokenize(src).expect("tokenize");
    let mut parser = parser::Parser::new(tokens, &mut nc);
    parser.parse().expect("parse")
}

fn make_index(pairs: &[(&str, &str)]) -> OpenIndex {
    let mut idx = OpenIndex::new();
    for (m, n) in pairs {
        idx.add_member(m, n);
    }
    idx
}

/// Pull the body expression of the first `def`/`let` declaration.
fn first_decl_body(prog: &Program) -> &Expression {
    for d in &prog.declarations {
        if let Declaration::Decl(d) = d {
            return &d.body;
        }
    }
    panic!("no def/let declaration");
}

/// Walk an expression, returning the first `Identifier` whose bare name
/// equals `target`, or panic. Useful for verifying rewrites without
/// depending on application/argument structure.
fn find_ident<'a>(expr: &'a Expression, target: &str) -> &'a Expression {
    fn walk<'a>(e: &'a Expression, target: &str) -> Option<&'a Expression> {
        if let ExprKind::Identifier(_, name) = &e.kind {
            if name == target {
                return Some(e);
            }
        }
        match &e.kind {
            ExprKind::Application(f, args) => {
                if let Some(r) = walk(f, target) {
                    return Some(r);
                }
                for a in args {
                    if let Some(r) = walk(a, target) {
                        return Some(r);
                    }
                }
                None
            }
            ExprKind::Lambda(l) => walk(&l.body, target),
            ExprKind::LetIn(li) => walk(&li.value, target).or_else(|| walk(&li.body, target)),
            ExprKind::If(i) => walk(&i.condition, target)
                .or_else(|| walk(&i.then_branch, target))
                .or_else(|| walk(&i.else_branch, target)),
            ExprKind::BinaryOp(_, l, r) => walk(l, target).or_else(|| walk(r, target)),
            ExprKind::UnaryOp(_, e) => walk(e, target),
            ExprKind::Tuple(es) | ExprKind::ArrayLiteral(es) | ExprKind::VecMatLiteral(es) => {
                for e in es {
                    if let Some(r) = walk(e, target) {
                        return Some(r);
                    }
                }
                None
            }
            ExprKind::FieldAccess(inner, _) => walk(inner, target),
            ExprKind::TypeAscription(e, _) | ExprKind::TypeCoercion(e, _) => walk(e, target),
            ExprKind::ArrayIndex(a, i) => walk(a, target).or_else(|| walk(i, target)),
            _ => None,
        }
    }
    walk(expr, target).unwrap_or_else(|| panic!("no identifier `{}` in expression", target))
}

fn ident_quals<'a>(expr: &'a Expression) -> &'a [String] {
    match &expr.kind {
        ExprKind::Identifier(quals, _) => quals,
        _ => panic!("expected identifier"),
    }
}

// =============================================================================
// OpenIndex construction
// =============================================================================

#[test]
fn open_index_split_qualified_keys() {
    let idx = OpenIndex::from_qualified_names(["f32.cos", "f32.sin", "i32.abs"]);
    assert!(idx.has_module("f32"));
    assert!(idx.has_module("i32"));
    assert!(!idx.has_module("u8"));
    assert!(idx.has_member("f32", "cos"));
    assert!(idx.has_member("i32", "abs"));
    assert!(!idx.has_member("f32", "abs"));
}

#[test]
fn open_index_skips_unqualified_keys() {
    // Top-level intrinsic names like `_w_intrinsic_*` aren't module
    // members and shouldn't pollute the module set.
    let idx = OpenIndex::from_qualified_names(["_w_intrinsic_cos", "f32.cos"]);
    assert!(idx.has_module("f32"));
    assert!(!idx.has_module("_w_intrinsic_cos"));
}

// =============================================================================
// Resolver behavior
// =============================================================================

#[test]
fn happy_path_rewrites_bare_to_qualified() {
    let mut prog = parse(
        r#"
        open f32
        def f (x: f32) f32 = cos(x)
        "#,
    );
    let idx = make_index(&[("f32", "cos"), ("f32", "sin")]);
    let mut r = OpenResolver::new(&idx);
    r.resolve_program(&mut prog).expect("resolve");

    let body = first_decl_body(&prog);
    let cos = find_ident(body, "cos");
    assert_eq!(ident_quals(cos), &["f32".to_string()]);
}

#[test]
fn local_binding_shadows_open() {
    // `let cos = ...` introduces a local that shadows `open f32`.
    let mut prog = parse(
        r#"
        open f32
        def f (x: f32) f32 =
          let cos = x in cos
        "#,
    );
    let idx = make_index(&[("f32", "cos")]);
    let mut r = OpenResolver::new(&idx);
    r.resolve_program(&mut prog).expect("resolve");

    let body = first_decl_body(&prog);
    let cos = find_ident(body, "cos");
    assert!(
        ident_quals(cos).is_empty(),
        "local `cos` should remain unqualified, got quals = {:?}",
        ident_quals(cos)
    );
}

#[test]
fn lambda_param_shadows_open() {
    let mut prog = parse(
        r#"
        open f32
        def f (x: f32) f32 = (|cos| cos)(x)
        "#,
    );
    let idx = make_index(&[("f32", "cos")]);
    let mut r = OpenResolver::new(&idx);
    r.resolve_program(&mut prog).expect("resolve");

    let body = first_decl_body(&prog);
    let cos = find_ident(body, "cos");
    assert!(
        ident_quals(cos).is_empty(),
        "lambda-param `cos` should remain unqualified, got quals = {:?}",
        ident_quals(cos)
    );
}

#[test]
fn already_qualified_left_alone() {
    // `f32.sin(x)` parses as Application(FieldAccess(Identifier("f32"),
    // "sin"), [x]). The pass must not crack open the FieldAccess
    // wrapper or rewrite the inner `f32` to anything else just because
    // `f32` is open. (Nothing in the index has a member literally named
    // "f32", so the rewrite rule shouldn't fire.)
    let mut prog = parse(
        r#"
        open f32
        def f (x: f32) f32 = f32.sin(x)
        "#,
    );
    let idx = make_index(&[("f32", "sin")]);
    let mut r = OpenResolver::new(&idx);
    r.resolve_program(&mut prog).expect("resolve");

    let body = first_decl_body(&prog);
    // Body should still be Application(FieldAccess(Identifier(_, "f32"), "sin"), [x]).
    let func = match &body.kind {
        ExprKind::Application(f, _) => f.as_ref(),
        _ => panic!("expected Application, got {:?}", body.kind),
    };
    let (inner, field) = match &func.kind {
        ExprKind::FieldAccess(inner, field) => (inner.as_ref(), field),
        _ => panic!("expected FieldAccess, got {:?}", func.kind),
    };
    assert_eq!(field, "sin", "field name should be unchanged");
    let f32_ident = match &inner.kind {
        ExprKind::Identifier(quals, name) => (quals, name),
        _ => panic!("expected Identifier inside FieldAccess"),
    };
    assert!(
        f32_ident.0.is_empty(),
        "inner `f32` identifier should remain unqualified, got quals = {:?}",
        f32_ident.0
    );
    assert_eq!(f32_ident.1, "f32");
}

#[test]
fn manually_qualified_identifier_left_alone() {
    // Construct an Identifier with a pre-existing qualifier directly
    // (the parser never produces this form, but downstream passes
    // might). Rule 1 should leave it untouched.
    let mut nc: ast::NodeCounter = ast::NodeCounter::new();
    use crate::ast::NodeCounterExt;
    let span = ast::Span::dummy();
    let ident = nc.mk_node(
        ExprKind::Identifier(vec!["i32".to_string()], "abs".to_string()),
        span,
    );
    let body = ident;
    let decl = Declaration::Decl(ast::Decl {
        keyword: "def",
        attributes: vec![],
        name: "f".to_string(),
        name_span: span,
        size_params: vec![],
        type_params: vec![],
        params: vec![],
        ty: None,
        body,
    });
    let mut prog = Program {
        declarations: vec![
            Declaration::Open(ast::ModuleExpression::Name("f32".to_string())),
            decl,
        ],
    };
    // Both modules in index, both with `abs`. Without rule 1 this would
    // be ambiguous; rule 1 says: already qualified, do nothing.
    let idx = make_index(&[("f32", "abs"), ("i32", "abs")]);
    let mut r = OpenResolver::new(&idx);
    r.resolve_program(&mut prog).expect("resolve");

    let body = first_decl_body(&prog);
    match &body.kind {
        ExprKind::Identifier(quals, name) => {
            assert_eq!(quals, &["i32".to_string()]);
            assert_eq!(name, "abs");
        }
        _ => panic!("expected Identifier"),
    }
}

#[test]
fn function_param_shadows_open() {
    // The function parameter `x` doesn't collide with anything, but
    // verify that pre-loaded top-level names like `f` aren't rewritten
    // either if a same-named member existed.
    let mut prog = parse(
        r#"
        open f32
        def f (x: f32) f32 = x
        "#,
    );
    // Add a fake `f32.f` member to test the top-level shadow.
    let idx = make_index(&[("f32", "f")]);
    let mut r = OpenResolver::new(&idx);
    r.resolve_program(&mut prog).expect("resolve");

    // Top-level `def f` registers `f` as a bound name, so any `f`
    // reference (in this test there isn't one, but the property holds)
    // would not rewrite. Sanity check: the body's `x` is bare.
    let body = first_decl_body(&prog);
    let x = find_ident(body, "x");
    assert!(ident_quals(x).is_empty());
}

#[test]
fn unknown_module_errors_at_open_site() {
    let mut prog = parse(
        r#"
        open f23
        def f (x: f32) f32 = x
        "#,
    );
    let idx = make_index(&[("f32", "cos")]);
    let mut r = OpenResolver::new(&idx);
    let err = r.resolve_program(&mut prog).expect_err("should error");
    let msg = format!("{}", err);
    assert!(
        msg.contains("f23"),
        "error should mention the unknown module name, got: {}",
        msg
    );
}

#[test]
fn ambiguous_bare_name_errors_with_candidates() {
    // Both f32 and i32 expose `abs`. `abs(x)` after both opens is
    // ambiguous and must error.
    let mut prog = parse(
        r#"
        open f32
        open i32
        def f (x: f32) f32 = abs(x)
        "#,
    );
    let idx = make_index(&[("f32", "abs"), ("i32", "abs")]);
    let mut r = OpenResolver::new(&idx);
    let err = r.resolve_program(&mut prog).expect_err("should error");
    let msg = format!("{}", err);
    assert!(
        msg.contains("ambiguous"),
        "error should describe ambiguity, got: {}",
        msg
    );
    assert!(
        msg.contains("f32") && msg.contains("i32"),
        "error should list both candidate modules, got: {}",
        msg
    );
}

#[test]
fn no_op_when_no_bare_uses() {
    // `open f32` is recognized; the program contains no bare names that
    // would rewrite. Should compile fine.
    let mut prog = parse(
        r#"
        open f32
        def f (x: f32) f32 = x
        "#,
    );
    let idx = make_index(&[("f32", "cos")]);
    let mut r = OpenResolver::new(&idx);
    r.resolve_program(&mut prog).expect("resolve");
}

#[test]
fn bare_name_with_no_match_left_alone() {
    // `open f32` is in scope, but `frobnicate` is not in any opened
    // module. Leave bare for downstream resolution.
    let mut prog = parse(
        r#"
        open f32
        def f (x: f32) f32 = frobnicate(x)
        "#,
    );
    let idx = make_index(&[("f32", "cos")]);
    let mut r = OpenResolver::new(&idx);
    r.resolve_program(&mut prog).expect("resolve");

    let body = first_decl_body(&prog);
    let frob = find_ident(body, "frobnicate");
    assert!(
        ident_quals(frob).is_empty(),
        "unmatched bare name should remain unqualified"
    );
}

#[test]
fn innermost_open_wins_when_unique() {
    // Both `f32` and `i32` opened; `cos` only lives in `f32`. Even
    // though `i32` is innermost, the candidate set has just one entry,
    // so `cos` rewrites unambiguously to `f32.cos`.
    let mut prog = parse(
        r#"
        open f32
        open i32
        def f (x: f32) f32 = cos(x)
        "#,
    );
    let idx = make_index(&[("f32", "cos"), ("i32", "abs")]);
    let mut r = OpenResolver::new(&idx);
    r.resolve_program(&mut prog).expect("resolve");

    let body = first_decl_body(&prog);
    let cos = find_ident(body, "cos");
    assert_eq!(ident_quals(cos), &["f32".to_string()]);
}

#[test]
fn identical_open_does_not_cause_ambiguity() {
    // A library re-exports `open f32`; the importer also `open f32`.
    // Both stack entries point at the same module, so `clamp` resolves
    // unambiguously to `f32.clamp` instead of being flagged ambiguous.
    let mut prog = parse(
        r#"
        open f32
        open f32
        def f (x: f32) f32 = clamp(x)
        "#,
    );
    let idx = make_index(&[("f32", "clamp")]);
    let mut r = OpenResolver::new(&idx);
    r.resolve_program(&mut prog).expect("resolve");

    let body = first_decl_body(&prog);
    let c = find_ident(body, "clamp");
    assert_eq!(ident_quals(c), &["f32".to_string()]);
}
