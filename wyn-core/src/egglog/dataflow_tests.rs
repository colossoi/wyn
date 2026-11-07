use super::{reference_paths, Dataflow, Value};
use crate::egglog::query::Query;
use crate::tlc::{self, DefMeta};
use crate::{ast_type_holes, compile_thru_frontend, SymbolId, SymbolTable};
use egglog_engine::Write;

#[test]
fn import_returns_engine_errors_instead_of_panicking() {
    let program = compile_thru_frontend("entry main(x:i32) i32=x+1").unwrap();
    let program = ast_type_holes::reject_type_holes(program).unwrap();
    let program = tlc::lower_from_ast(program).unwrap();
    let entry = program.defs.iter().find(|def| matches!(def.meta, DefMeta::EntryPoint(_))).unwrap();
    let (body, _) = tlc::extract_lambda_params_ref(&entry.body);
    let mut graph = egglog_engine::EGraph::default();
    let mut flow = Dataflow::default();
    // Missing schema makes the engine reject the first imported fact.
    let result = graph.update(|mut sink| flow.import(body, &mut sink));
    assert!(result.is_err());
}

#[test]
fn imported_reference_occurrences_match_authored_paths_in_every_subtree() {
    let source = "entry main(x: (i32, (i32, i32)), choose: bool) i32 =
        let alias = x in let ignored = (alias.1).0 in
        if choose then (alias.1).1 else let pair = (x.0, (alias.1).0) in pair.0";
    let program = compile_thru_frontend(source).unwrap();
    let program = ast_type_holes::reject_type_holes(program).unwrap();
    let program = tlc::lower_from_ast(program).unwrap();
    let entry = program.defs.iter().find(|def| matches!(def.meta, DefMeta::EntryPoint(_))).unwrap();
    let (body, _) = tlc::extract_lambda_params_ref(&entry.body);
    let (flow, _) = Dataflow::build(body);
    body.walk(&mut |term: &tlc::Term| {
        let expected: crate::LookupSet<_> = reference_paths(term).into_iter().collect();
        let actual: crate::LookupSet<_> =
            flow.capture_sources(&flow, term.id).unwrap_or_default().into_values().flatten().collect();
        assert_eq!(actual, expected, "reference paths for {:?}", term.id);
        tlc::WalkDecision::Recurse
    });
}

#[test]
fn demand_stops_at_references_and_branch_boundaries_but_dependencies_cross_them() {
    let (mut values, _, parameters, symbols) = graph(
        "entry main(x: i32, choose: bool) i32 =
           let produced = x + 1 in
           let selected = if choose then produced else x in selected",
    );
    let produced = values.bindings[&binding(&values, &symbols, "produced")].evaluation;
    let selected = values.bindings[&binding(&values, &symbols, "selected")].evaluation;
    let borrowed = Query(&values.graph).row("FlowReference", |row| row[1] == produced).unwrap().unwrap()[0];
    values
        .graph
        .update(|mut sink| {
            sink.add("FlowDemand", (borrowed, borrowed))?;
            sink.add("FlowDemand", (selected, selected))?;
            Ok(())
        })
        .unwrap();
    values.graph.parse_and_run_program(None, "(run-schedule (saturate flow-evaluation))").unwrap();
    let query = Query(&values.graph);
    assert!(!query.contains("FlowDemand", (borrowed, produced)).unwrap());
    assert!(!query.contains("FlowDemand", (selected, borrowed)).unwrap());
    assert!(!query.contains("FlowDemand", (selected, produced)).unwrap());
    let depends = values.dependencies(|symbol| symbol == parameters[0]);
    assert!(depends.0.contains(&borrowed));
    assert!(depends.0.contains(&selected));
}

fn graph(source: &str) -> (Dataflow, Value, Vec<SymbolId>, SymbolTable) {
    let program = compile_thru_frontend(source).expect("typed source");
    let program = ast_type_holes::reject_type_holes(program).expect("resolved types");
    let program = tlc::lower_from_ast(program).expect("TLC");
    let entry = program.defs.iter().find(|def| matches!(def.meta, DefMeta::EntryPoint(_))).expect("entry");
    let (body, parameters) = tlc::extract_lambda_params_ref(&entry.body);
    let (values, result) = Dataflow::try_build(body).unwrap();
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
                assert_eq!(values.binding_input(*symbol), Some(parameters[0]));
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
fn record_update_forwards_fields_but_keeps_operand_evaluation() {
    let (values, result, parameters, symbols) = graph(
        "entry main(x: i32, y: i32, z: i32) i32 =
          let old = { a = x, b = y } in
          let updated = old with a = z in
          let selected = updated.b in selected",
    );
    assert_eq!(values.input_origin(result, &[]), Some(parameters[1]));
    let updated = binding(&values, &symbols, "updated");
    assert_eq!(
        values.input_origin(values.bindings[&updated].value, &[0]),
        Some(parameters[2])
    );
    let selected = binding(&values, &symbols, "selected");
    for parameter in [parameters[0], parameters[2]] {
        assert!(!values.dependent_bindings(|symbol| symbol == parameter).contains(&selected));
        let depends = values.dependencies(|symbol| symbol == parameter);
        assert!(depends.0.contains(&values.bindings[&selected].evaluation));
    }
}

#[test]
fn branch_projections_require_the_same_forwarded_input() {
    let (values, result, parameters, _) = graph(
        "entry main(x: i32, y: i32, choose: bool) (i32, i32) =
           if choose then (x, x) else (x, y)",
    );
    assert_eq!(values.input_origin(result, &[0]), Some(parameters[0]));
    assert_eq!(values.input_origin(result, &[1]), None);
    let (values, result, parameters, _) = graph(
        "entry main(x: i32, y: i32, choose: bool) i32 =
           let pair = if choose then (x, x) else (x, y) in pair.0",
    );
    assert_eq!(values.input_origin(result, &[]), Some(parameters[0]));
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

fn binding(values: &Dataflow, symbols: &SymbolTable, name: &str) -> SymbolId {
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
        assert!(depends.0.contains(&values.bindings[&selected].evaluation));
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

    assert_eq!(values.binding_input(selected), None);
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
    let (values, result, parameters, symbols) = graph(
        "entry main(x: i32, choose: bool) i32 =
           let selected = if choose then x else x in selected",
    );
    assert_eq!(values.input_origin(result, &[]), Some(parameters[0]));
    assert_eq!(values.binding_input(binding(&values, &symbols, "selected")), None);
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
    assert!(Dataflow::evaluation_uses(body, |symbol| symbol == parameters[0].0));
}
