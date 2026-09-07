use super::{FrontendWarning, UnusedBindingKind, UnusedDeclarationKind};
use crate::compile_thru_frontend;

fn frontend_warnings(source: &str) -> Vec<FrontendWarning> {
    compile_thru_frontend(source).expect("source should type check").global_context.warnings
}

fn unused_names(warnings: &[FrontendWarning]) -> Vec<&str> {
    warnings
        .iter()
        .filter_map(|warning| match warning {
            FrontendWarning::UnusedBinding { name, .. }
            | FrontendWarning::UnusedDeclaration { name, .. } => Some(name.as_str()),
            FrontendWarning::TypeHoleFilled { .. } => None,
        })
        .collect()
}

#[test]
fn reports_each_unused_parameter_and_let_binding() {
    let warnings = frontend_warnings(
        r#"
entry main(used: i32, unused: i32, another: i32) i32 =
  let dead = 1 in
  used
"#,
    );

    assert_eq!(unused_names(&warnings), ["unused", "another", "dead"]);
    assert!(matches!(
        warnings[0],
        FrontendWarning::UnusedBinding {
            kind: UnusedBindingKind::Parameter,
            ..
        }
    ));
    assert!(matches!(
        warnings[2],
        FrontendWarning::UnusedBinding {
            kind: UnusedBindingKind::LetBinding,
            ..
        }
    ));
}

#[test]
fn symbol_identity_distinguishes_shadowed_bindings() {
    let warnings = frontend_warnings(
        r#"
entry main(value: i32, condition: bool) i32 =
  let value = if condition then 1 else 2 in
  value
"#,
    );

    assert_eq!(unused_names(&warnings), ["value"]);
    assert!(matches!(
        warnings[0],
        FrontendWarning::UnusedBinding {
            kind: UnusedBindingKind::Parameter,
            ..
        }
    ));
}

#[test]
fn constant_folded_references_still_count_as_uses() {
    let warnings = frontend_warnings(
        r#"
def constant: i32 = 1
entry main() i32 =
  let local = constant in
  local
"#,
    );

    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
}

#[test]
fn constant_uses_inside_unreachable_code_do_not_create_roots() {
    let warnings = frontend_warnings(
        r#"
def constant: i32 = 1
def dead: i32 = constant
entry main() i32 = 0
"#,
    );

    assert_eq!(unused_names(&warnings), ["constant", "dead"]);
}

#[test]
fn leading_underscores_and_wildcards_suppress_binding_warnings() {
    let warnings = frontend_warnings(
        r#"
def public(pair: (i32, i32, i32), _parameter: i32) i32 =
  let (used, _ignored, _) = pair in
  used
"#,
    );

    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
}

#[test]
fn reports_unreachable_definitions_in_executable_mode() {
    let warnings = frontend_warnings(
        r#"
def used(value: i32) i32 = value
def dead(value: i32) i32 = value
def cycle_a(value: i32) i32 = cycle_b(value)
def cycle_b(value: i32) i32 = cycle_a(value)
entry main(value: i32) i32 = used(value)
"#,
    );

    assert_eq!(unused_names(&warnings), ["dead", "cycle_a", "cycle_b"]);
    assert!(warnings.iter().all(|warning| matches!(
        warning,
        FrontendWarning::UnusedDeclaration {
            kind: UnusedDeclarationKind::Definition,
            ..
        }
    )));
}

#[test]
fn treats_top_level_definitions_as_public_without_entries() {
    let warnings = frontend_warnings("def public(unused: i32) i32 = 0\n");

    assert_eq!(unused_names(&warnings), ["unused"]);
    assert!(matches!(
        warnings[0],
        FrontendWarning::UnusedBinding {
            kind: UnusedBindingKind::Parameter,
            ..
        }
    ));
}

#[test]
fn reports_nested_lambda_loop_and_match_bindings() {
    let warnings = frontend_warnings(
        r#"
def nested(value: #some(i32) | #none) i32 =
  let from_lambda = (|unused_lambda: i32| 1)(0) in
  let from_loop = loop acc = 0 for unused_index < 1 do acc in
  match value
  case #some(unused_payload) -> from_lambda + from_loop
  case #none -> 0
"#,
    );

    assert_eq!(
        unused_names(&warnings),
        ["unused_lambda", "unused_index", "unused_payload"]
    );
    assert!(matches!(
        warnings[0],
        FrontendWarning::UnusedBinding {
            kind: UnusedBindingKind::Parameter,
            ..
        }
    ));
    assert!(matches!(
        warnings[1],
        FrontendWarning::UnusedBinding {
            kind: UnusedBindingKind::LoopVariable,
            ..
        }
    ));
    assert!(matches!(
        warnings[2],
        FrontendWarning::UnusedBinding {
            kind: UnusedBindingKind::MatchBinding,
            ..
        }
    ));
}

#[test]
fn unused_warnings_do_not_count_as_type_holes() {
    let typed =
        compile_thru_frontend("def public(unused: i32) i32 = 0\n").expect("source should type check");

    crate::ast_type_holes::reject_type_holes(typed)
        .expect("an unused-code warning must not make type-hole rejection fail");
}
