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
        ExprKind::ArrayWith { inplace, .. } => Some(*inplace),
        ExprKind::LetIn(let_in) => find_array_with_inplace_flag(&let_in.value)
            .or_else(|| find_array_with_inplace_flag(&let_in.body)),
        ExprKind::If(ifx) => find_array_with_inplace_flag(&ifx.then_branch)
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
        _ => None,
    }
}

fn first_array_with_flag(program: &Program) -> bool {
    for decl in &program.declarations {
        if let crate::ast::Declaration::Decl(d) = decl {
            if let Some(flag) = find_array_with_inplace_flag(&d.body) {
                return flag;
            }
        }
    }
    panic!("no ArrayWith expression found in program");
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
