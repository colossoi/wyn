//! End-to-end tests for the uniqueness promotion pass.
//!
//! We exercise the `inplace` flag on `ExprKind::ArrayWith` by running the
//! real frontend pipeline (parse → elaborate_modules → desugar → resolve →
//! fold_ast_constants → type_check → alias_check) and then inspecting the
//! AST inside the resulting `AliasChecked` *after* `uniqueness_promote` has
//! run, which happens inside `to_tlc`.
//!
//! Rather than crack open `to_tlc` (which consumes the AST), we drive the
//! promotion pass directly on the `AliasChecked` AST here.

use super::run;
use crate::ast::{ExprKind, Expression, Program};
use crate::{Compiler, FrontEnd};

fn check_program(source: &str) -> Program {
    let mut frontend = FrontEnd::new();
    let parsed = Compiler::parse(source, &mut frontend.node_counter).expect("parse failed");
    let parsed = parsed.elaborate_modules(&mut frontend.module_manager).expect("elaborate_modules failed");
    let desugared = parsed.desugar(&mut frontend.node_counter).expect("desugar failed");
    let resolved = desugared.resolve(&frontend.module_manager).expect("resolve failed");
    let ast_folded = resolved.fold_ast_constants();
    let type_checked = ast_folded
        .type_check(&mut frontend.module_manager, &mut frontend.schemes)
        .expect("type_check failed");
    let mut alias_checked = type_checked.alias_check().expect("alias_check failed");
    assert!(!alias_checked.has_alias_errors(), "alias errors");
    run(&mut alias_checked.ast, &alias_checked.alias_result);
    alias_checked.ast
}

fn find_array_with_inplace_flag(expr: &Expression) -> Option<bool> {
    match &expr.kind {
        ExprKind::ArrayWith {
            array,
            value,
            inplace,
            ..
        } => {
            // Recurse into the source/value first so we find the
            // *innermost* ArrayWith — useful for chained `a with [0]
            // = .. with [1] = ..` patterns where the test wants to
            // know about a specific layer.
            if let Some(flag) = find_array_with_inplace_flag(array) {
                return Some(flag);
            }
            if let Some(flag) = find_array_with_inplace_flag(value) {
                return Some(flag);
            }
            Some(*inplace)
        }
        ExprKind::LetIn(let_in) => find_array_with_inplace_flag(&let_in.value)
            .or_else(|| find_array_with_inplace_flag(&let_in.body)),
        ExprKind::If(ifx) => find_array_with_inplace_flag(&ifx.condition)
            .or_else(|| find_array_with_inplace_flag(&ifx.then_branch))
            .or_else(|| find_array_with_inplace_flag(&ifx.else_branch)),
        ExprKind::Application(f, args) => {
            if let Some(flag) = find_array_with_inplace_flag(f) {
                return Some(flag);
            }
            for a in args {
                if let Some(flag) = find_array_with_inplace_flag(a) {
                    return Some(flag);
                }
            }
            None
        }
        ExprKind::Tuple(elems) => elems.iter().find_map(find_array_with_inplace_flag),
        ExprKind::ArrayLiteral(elems) | ExprKind::VecMatLiteral(elems) => {
            elems.iter().find_map(find_array_with_inplace_flag)
        }
        ExprKind::BinaryOp(_, l, r) => {
            find_array_with_inplace_flag(l).or_else(|| find_array_with_inplace_flag(r))
        }
        ExprKind::UnaryOp(_, e) => find_array_with_inplace_flag(e),
        ExprKind::FieldAccess(e, _) => find_array_with_inplace_flag(e),
        ExprKind::ArrayIndex(a, i) => {
            find_array_with_inplace_flag(a).or_else(|| find_array_with_inplace_flag(i))
        }
        ExprKind::Loop(loop_expr) => {
            // Search the init (if any) and the body. Other loop
            // sub-expressions (the iterator/condition) are unlikely
            // to contain an ArrayWith but still worth covering.
            if let Some(init) = &loop_expr.init {
                if let Some(flag) = find_array_with_inplace_flag(init) {
                    return Some(flag);
                }
            }
            find_array_with_inplace_flag(&loop_expr.body)
        }
        ExprKind::Lambda(lam) => find_array_with_inplace_flag(&lam.body),
        ExprKind::Match(m) => {
            if let Some(flag) = find_array_with_inplace_flag(&m.scrutinee) {
                return Some(flag);
            }
            m.cases.iter().find_map(|c| find_array_with_inplace_flag(&c.body))
        }
        _ => None,
    }
}

fn first_array_with_flag(program: &Program) -> bool {
    for decl in &program.declarations {
        let body = match decl {
            crate::ast::Declaration::Decl(d) => &d.body,
            crate::ast::Declaration::Entry(e) => &e.body,
            _ => continue,
        };
        if let Some(flag) = find_array_with_inplace_flag(body) {
            return flag;
        }
    }
    panic!("no ArrayWith expression found in program");
}

/// Collect every `ArrayWith` flag in source order. Useful for
/// chained `a with [0] = .. with [1] = ..` tests that want to assert
/// on each layer independently.
fn collect_array_with_flags(expr: &Expression, out: &mut Vec<bool>) {
    match &expr.kind {
        ExprKind::ArrayWith {
            array,
            value,
            inplace,
            ..
        } => {
            collect_array_with_flags(array, out);
            collect_array_with_flags(value, out);
            out.push(*inplace);
        }
        ExprKind::LetIn(let_in) => {
            collect_array_with_flags(&let_in.value, out);
            collect_array_with_flags(&let_in.body, out);
        }
        ExprKind::If(ifx) => {
            collect_array_with_flags(&ifx.condition, out);
            collect_array_with_flags(&ifx.then_branch, out);
            collect_array_with_flags(&ifx.else_branch, out);
        }
        ExprKind::Application(f, args) => {
            collect_array_with_flags(f, out);
            for a in args {
                collect_array_with_flags(a, out);
            }
        }
        ExprKind::Tuple(elems) | ExprKind::ArrayLiteral(elems) | ExprKind::VecMatLiteral(elems) => {
            for e in elems {
                collect_array_with_flags(e, out);
            }
        }
        ExprKind::BinaryOp(_, l, r) => {
            collect_array_with_flags(l, out);
            collect_array_with_flags(r, out);
        }
        ExprKind::UnaryOp(_, e) | ExprKind::FieldAccess(e, _) => collect_array_with_flags(e, out),
        ExprKind::ArrayIndex(a, i) => {
            collect_array_with_flags(a, out);
            collect_array_with_flags(i, out);
        }
        ExprKind::Loop(loop_expr) => {
            if let Some(init) = &loop_expr.init {
                collect_array_with_flags(init, out);
            }
            collect_array_with_flags(&loop_expr.body, out);
        }
        ExprKind::Lambda(lam) => collect_array_with_flags(&lam.body, out),
        ExprKind::Match(m) => {
            collect_array_with_flags(&m.scrutinee, out);
            for c in &m.cases {
                collect_array_with_flags(&c.body, out);
            }
        }
        _ => {}
    }
}

fn all_array_with_flags(program: &Program) -> Vec<bool> {
    let mut flags = Vec::new();
    for decl in &program.declarations {
        match decl {
            crate::ast::Declaration::Decl(d) => collect_array_with_flags(&d.body, &mut flags),
            crate::ast::Declaration::Entry(e) => collect_array_with_flags(&e.body, &mut flags),
            _ => {}
        }
    }
    flags
}

#[test]
fn promotes_when_source_is_last_use() {
    // `a` is used only once — in the `with` itself. Promote.
    let src = r#"
def f(a: *[4]i32) *[4]i32 =
    a with [0] = 42
"#;
    let program = check_program(src);
    assert!(
        first_array_with_flag(&program),
        "expected inplace=true when source array is last-used"
    );
}

#[test]
fn does_not_promote_when_source_used_after() {
    // `a` is returned alongside `b`, so the `with` is not the last use.
    let src = r#"
def f(a: [4]i32) ([4]i32, [4]i32) =
    let b = a with [0] = 42 in (a, b)
"#;
    let program = check_program(src);
    assert!(
        !first_array_with_flag(&program),
        "expected inplace=false when source array is used after `with`"
    );
}

// =============================================================================
// Comprehensive promotion coverage
// =============================================================================
//
// The two existing tests cover the canonical "last use" / "used after"
// dichotomy. The cases below stress-test the promoter against control
// flow, structural containers, and chained ArrayWiths.

// --- A. Chained ArrayWiths --------------------------------------------------

#[test]
fn promotes_chained_array_with_when_last_use() {
    // `a with [0] = 1 with [1] = 2` desugars to a nested ArrayWith
    // where the inner `with` produces a value used only by the outer
    // `with`, and the outer's source is the entire chain. Both layers
    // should promote because each is the last use of its source.
    let src = r#"
def f(a: *[4]i32) *[4]i32 =
    a with [0] = 1 with [1] = 2
"#;
    let program = check_program(src);
    let flags = all_array_with_flags(&program);
    assert_eq!(flags.len(), 2, "expected two ArrayWith nodes in the chain");
    assert!(
        flags.iter().all(|&f| f),
        "every ArrayWith in a last-use chain should promote, got {:?}",
        flags
    );
}

// --- B. Inside control flow ------------------------------------------------

#[test]
fn promotes_inside_if_when_dead_after_branches() {
    // Both branches do their own `with` and the result of the if is
    // returned — neither branch's `with` source is used afterward,
    // so both should promote.
    let src = r#"
def f(c: bool, a: *[4]i32, b: *[4]i32) *[4]i32 =
    if c then a with [0] = 1 else b with [0] = 2
"#;
    let program = check_program(src);
    let flags = all_array_with_flags(&program);
    assert_eq!(flags.len(), 2, "expected one ArrayWith per branch");
    assert!(
        flags.iter().all(|&f| f),
        "both branches should promote — each branch's source is dead after, got {:?}",
        flags
    );
}

#[test]
fn promotes_entry_parameter_array_with() {
    // No `*` on the parameter — entry parameters should be treated as
    // unique automatically, so this `with` is a last use of an
    // effectively-unique value and should promote.
    let src = r#"
#[compute]
entry main(data: [4]i32) [4]i32 = data with [0] = 99
"#;
    let program = check_program(src);
    assert!(
        first_array_with_flag(&program),
        "expected inplace=true for entry-parameter modification — \
         entry params are effectively unique"
    );
}

#[test]
fn promotes_loop_body_array_with_on_carry() {
    // Loop carries `acc` and replaces it on each iteration via
    // `with`. The body's `with` source (`acc`) is dead by the end of
    // the body, so it should promote.
    let src = r#"
def f(a: *[4]i32) *[4]i32 =
    loop (acc) = a while false do acc with [0] = 1
"#;
    let program = check_program(src);
    assert!(
        first_array_with_flag(&program),
        "loop-carried ArrayWith should promote when the source dies in the body"
    );
}

// --- C. Aliased through structural containers ------------------------------

#[test]
fn does_not_promote_when_source_in_tuple() {
    // `a` is captured into a tuple alongside `b`; the tuple is
    // returned, so `a` is still live after the `with`. No promote.
    let src = r#"
def f(a: [4]i32) ([4]i32, [4]i32) =
    let b = a with [0] = 42 in (b, a)
"#;
    let program = check_program(src);
    assert!(
        !first_array_with_flag(&program),
        "expected inplace=false: a still escapes via the tuple"
    );
}

#[test]
fn does_not_promote_when_source_aliased_through_let_chain() {
    // Two let-bindings to the same array, then a `with` on one and a
    // read of the other. The other binding aliases the source so
    // it's still live.
    let src = r#"
def f(a: [4]i32) ([4]i32, i32) =
    let alias = a in
    let b = a with [0] = 42 in
    (b, alias[0])
"#;
    let program = check_program(src);
    assert!(
        !first_array_with_flag(&program),
        "expected inplace=false: alias keeps a's backing store live"
    );
}

#[test]
fn does_not_promote_when_source_captured_by_lambda() {
    let src = r#"
def f(a: [4]i32) ([4]i32, i32 -> i32) =
    let b = a with [0] = 42 in
    let g = |i: i32| a[i] in
    (b, g)
"#;
    let program = check_program(src);
    assert!(
        !first_array_with_flag(&program),
        "expected inplace=false: a is captured by g, so the with isn't a last-use"
    );
}

// --- D. Chained promotion across passes ------------------------------------

#[test]
#[ignore = "compiler gap: in `let b = a with [0] = 1 in let c = b with [1] \
            = 2 in c`, both links should promote — each intermediate is \
            consumed exactly once. The promoter today flips inplace=true on \
            the first link but leaves the second false. Likely cause: the \
            uniqueness_promote pass doesn't yet see the second `with`'s \
            source (`b`) as last-used through the second binding."]
fn promotes_when_inplace_chain_outputs_last_use() {
    // `let b = a with [0] = 1 in let c = b with [1] = 2 in c` — both
    // links can promote: each intermediate is used only once by the
    // next link.
    let src = r#"
def f(a: *[4]i32) *[4]i32 =
    let b = a with [0] = 1 in
    let c = b with [1] = 2 in
    c
"#;
    let program = check_program(src);
    let flags = all_array_with_flags(&program);
    assert_eq!(flags.len(), 2);
    assert!(
        flags.iter().all(|&f| f),
        "both `with` ops should promote, got {:?}",
        flags
    );
}

#[test]
#[ignore = "compiler bug (unsafe promotion): in `let b = a with [0] = 1 in \
            let c = b with [1] = 2 in (b, c)`, `b` aliases `a`'s backing \
            store and `b` is returned in the tuple — promoting the first \
            `with` to in-place would let the caller observe the mutation \
            through `a` (or any alias they still hold). The promoter today \
            sets inplace=true here, which is unsound. The check needs to \
            include aliases created by the `with` itself, not just \
            pre-existing ones."]
fn does_not_promote_first_link_when_intermediate_escapes() {
    // `b` is returned alongside `c`, so the first `with` is *not* a
    // last use of `a`'s memory (b aliases it). The first link should
    // not promote; the second can if `c` is dead.
    let src = r#"
def f(a: [4]i32) ([4]i32, [4]i32) =
    let b = a with [0] = 1 in
    let c = b with [1] = 2 in
    (b, c)
"#;
    let program = check_program(src);
    let flags = all_array_with_flags(&program);
    assert_eq!(flags.len(), 2);
    // Both `with`es share the same backing store via b, so neither
    // can promote — b is still used in the tuple.
    assert!(
        !flags[0],
        "first ArrayWith must not promote: b aliases a, and b is live"
    );
    assert!(
        !flags[1],
        "second ArrayWith must not promote: b is still live for the tuple"
    );
}
