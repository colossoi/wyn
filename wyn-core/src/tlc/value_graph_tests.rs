use super::{Source, Value, ValueGraph};
use crate::tlc::{self, DefMeta};
use crate::{ast_type_holes, compile_thru_frontend, SymbolId, SymbolTable};

fn graph(source: &str) -> (ValueGraph, Value, Vec<SymbolId>, SymbolTable) {
    let program = compile_thru_frontend(source).expect("typed source");
    let program = ast_type_holes::reject_type_holes(program).expect("resolved types");
    let program = tlc::lower_from_ast(program).expect("TLC");
    let entry = program.defs.iter().find(|def| matches!(def.meta, DefMeta::EntryPoint(_))).expect("entry");
    let (body, parameters) = tlc::extract_lambda_params_ref(&entry.body);
    let (values, result) = ValueGraph::build(body);
    let parameters = parameters.iter().map(|(symbol, _)| *symbol).collect();
    (values, result, parameters, program.symbols)
}

#[test]
fn array_components_keep_parameter_and_constant_origins_through_aliases() {
    let (values, result, parameters, symbols) = graph(
        "entry main(x: i32) [2]i32 =
           let y = 7 in let alias = x in let copy = alias in [copy, y]",
    );
    assert_eq!(values.input_origin(result, &[0]), Some(parameters[0]));
    assert_eq!(values.input_origin(result, &[1]), None);
    let dependent = values.dependent_bindings(|symbol| symbol == parameters[0]);
    for symbol in values.bindings.keys() {
        match symbols.get(*symbol).map(String::as_str) {
            Some("alias" | "copy" | "x") => {
                assert!(dependent.contains(symbol));
                assert!(matches!(
                    values.projected_source(Source::Input(*symbol), &[]),
                    Some(Source::Input(input)) if input == parameters[0]
                ));
            }
            Some("y") => assert!(!dependent.contains(symbol)),
            _ => {}
        }
    }
}

#[test]
fn projecting_a_component_does_not_inherit_other_components_dependencies() {
    let (values, result, parameters, symbols) = graph(
        "entry main(x: i32, y: i32) i32 =
           let pair = (x, y) in let selected = pair.1 in selected",
    );
    assert_eq!(values.input_origin(result, &[]), Some(parameters[1]));
    let dependent = values.dependent_bindings(|symbol| symbol == parameters[0]);
    let selected = values
        .bindings
        .keys()
        .find(|symbol| symbols.get(**symbol).map(String::as_str) == Some("selected"))
        .expect("selected binding");
    assert!(!dependent.contains(selected));
}

#[test]
fn branch_projections_require_the_same_forwarded_input() {
    let (values, result, parameters, _) = graph(
        "entry main(x: i32, y: i32, choose: bool) (i32, i32) =
           if choose then (x, x) else (x, y)",
    );
    assert_eq!(values.input_origin(result, &[0]), Some(parameters[0]));
    assert_eq!(values.input_origin(result, &[1]), None);
}

#[test]
fn a_computed_dependency_is_not_an_unchanged_input() {
    let (values, result, parameters, symbols) =
        graph("entry main(x: i32) i32 = let computed = x + 1 in computed");
    assert_eq!(values.input_origin(result, &[]), None);
    let dependent = values.dependent_bindings(|symbol| symbol == parameters[0]);
    let computed = values
        .bindings
        .keys()
        .find(|symbol| symbols.get(**symbol).map(String::as_str) == Some("computed"))
        .expect("computed binding");
    assert!(dependent.contains(computed));
}

fn binding(values: &ValueGraph, symbols: &SymbolTable, name: &str) -> SymbolId {
    *values
        .bindings
        .keys()
        .find(|symbol| symbols.get(**symbol).map(String::as_str) == Some(name))
        .expect(name)
}

#[test]
fn hoisting_keeps_dependencies_of_discarded_evaluations() {
    for expression in ["(x, 7).1", "(let ignored = x in 7)"] {
        let (values, _, parameters, symbols) = graph(&format!(
            "entry main(x: i32) i32 = let selected = {expression} in selected"
        ));
        let selected = binding(&values, &symbols, "selected");
        assert!(!values.dependent_bindings(|symbol| symbol == parameters[0]).contains(&selected));
        let depends = values.dependencies(|symbol| symbol == parameters[0]);
        assert!(depends.0[values.bindings[&selected].evaluation.0]);
    }
}

#[test]
fn producer_paths_survive_aliases_and_nested_projections() {
    let (mut values, _, _, symbols) = graph(
        "entry main(x: i32) i32 =
           let produced = (x + 1, (x + 2, x + 3)) in
           let alias = produced in let field = alias.1 in let selected = field.0 in selected",
    );
    let produced = binding(&values, &symbols, "produced");
    for name in ["produced", "alias", "field", "selected"] {
        values.name_producer(binding(&values, &symbols, name));
    }
    let selected = binding(&values, &symbols, "selected");
    let origin = values.binding_origin(selected, &[]).expect("named producer");
    assert_eq!(origin.producer, produced);
    assert_eq!(origin.path, [1, 0]);
    let field = values.project(values.bindings[&selected].value, &[]).expect("selected field");
    assert!(
        matches!(values.sources[field.0], Source::Term(_)),
        "computed field retains its occurrence"
    );
    assert!(matches!(
        values.projected_source(Source::Input(selected), &[]),
        Some(Source::Term(_))
    ));
}

#[test]
fn naming_a_forwarded_input_does_not_rename_the_input() {
    let (mut values, _, parameters, symbols) = graph(
        "entry main(xs: (i32, i32)) i32 =
           let copy = xs in let selected = xs.0 in selected",
    );
    let copy = binding(&values, &symbols, "copy");
    let selected = binding(&values, &symbols, "selected");
    values.name_producer(copy);
    values.name_producer(selected);
    assert!(values.binding_origin(parameters[0], &[]).is_none());
    assert_eq!(
        values.binding_origin(copy, &[0]).expect("copy producer").producer,
        copy
    );
    assert_eq!(
        values.binding_origin(selected, &[]).expect("separate producer").producer,
        selected
    );
}

#[test]
fn structural_forwarding_does_not_select_a_value_inside_a_branch() {
    let (values, result, _, _) = graph(
        "entry main(x: i32, choose: bool) (i32, i32) =
           if choose then (x, 7) else (x, 8)",
    );
    assert!(values.project(result, &[0]).is_none());
}

#[test]
fn captures_include_discarded_evaluations_and_exclude_nested_binders() {
    let (values, _, parameters, symbols) = graph(
        "entry main(xs: []i32, bias: i32, n: i32) []i32 =
           let ignored = bias in
           map(|x| loop acc = x for i < n do acc + i, xs)",
    );
    assert_eq!(
        values.free_symbols().collect::<Vec<_>>(),
        [parameters[1], parameters[2], parameters[0]]
    );
    let captures: Vec<tlc::Term> = values.free_variables(|symbol| symbol == parameters[0]);
    assert_eq!(captures.len(), 2);
    for (capture, name) in captures.iter().zip(["bias", "n"]) {
        let tlc::TermKind::Var(tlc::VarRef::Symbol(symbol)) = capture.kind else {
            panic!("variable capture")
        };
        assert_eq!(symbols.get(symbol).map(String::as_str), Some(name));
        assert_ne!(capture.id, tlc::TermId::SYNTHETIC);
    }
}

#[test]
fn capture_identity_distinguishes_shadowed_parameters() {
    let (values, _, parameters, _) = graph(
        "entry main(xs: []i32, x: i32) []i32 =
           let saved = x in map(|x| x + saved, xs)",
    );
    assert_eq!(
        values.free_symbols().collect::<Vec<_>>(),
        [parameters[1], parameters[0]]
    );
}

#[test]
fn evaluation_dependencies_include_scatter_destinations() {
    let program = crate::test_pipeline::compile_thru_static_index(
        "entry main(dest: *[4]i32) [4]i32 = scatter(dest, [0, 1], [2, 3])",
    );
    let entry = program.defs.iter().find(|def| matches!(def.meta, DefMeta::EntryPoint(_))).unwrap();
    let (body, parameters) = tlc::extract_lambda_params_ref(&entry.body);
    assert!(ValueGraph::evaluation_uses(body, |symbol| symbol == parameters[0].0));
}
