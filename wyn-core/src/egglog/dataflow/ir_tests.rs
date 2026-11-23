use super::{operation, outputs, parameters, region, yields, Code, Name, Parameter};
use crate::egglog::graph::Graph;
use crate::egglog::memory::load;
use crate::egglog::payload::Payload;
use crate::egglog::query::Query;
use crate::egglog::test_graph;
use crate::egglog::OptimizeError;
use crate::op::BinaryOperator;
use egglog_engine::{RawValues, Write};

#[test]
fn cloned_parameters_preserve_the_callers_value_type() {
    let mut graph = test_graph::fixture(
        &[],
        r#"
        (let context (OperationId 0)) (let body (RegionId 0))
        (let parameter (ValueId 1)) (let argument (ValueId 2))
        (let scalar (ElementValue (ScalarElement (I32))))
        (let declared-array (ArrayType (ValueArray) (vec-of (UnspecifiedExtent)) scalar))
        (let actual-array (ArrayType (ValueArray) (vec-of (FixedExtent 10)) scalar))
        (set (ValueType parameter) (LogicalValue declared-array))
        (set (ValueType argument) (LogicalValue actual-array))
        (set (FixtureParameter body 0) (ValueNameRef parameter))
        (set (ParameterDiet body 0) (Observe))
        (set (FixtureYield body 0) (ValueNameRef parameter))
        (set (RegionReturnType body 0) (LogicalReturn declared-array))
        (let argument-ref (ValueNameRef argument))
        "#,
    )
    .unwrap();
    let query = Query(&graph);
    let body = query.constructor("RegionId", 0_i64).unwrap().unwrap();
    let context = query.constructor("OperationId", 0_i64).unwrap().unwrap();
    let argument = query.constructor("ValueId", 2_i64).unwrap().unwrap();
    let argument_ref = query.constructor("ValueNameRef", argument).unwrap().unwrap();
    let cloned = super::super::edit::clone_region(&mut graph, context, body, &[argument_ref]).unwrap();
    assert_eq!(graph.yield_at(cloned, 0), Some(argument_ref));
    graph
        .parse_and_run_program(
            None,
            "(check (= (ValueType argument) (LogicalValue actual-array)))",
        )
        .unwrap();
}

#[test]
fn operations_share_connectivity_without_sharing_occurrences_or_payloads() {
    let mut graph = Graph::default();
    load(&mut graph).unwrap();
    let (add, sub) = graph
        .update(|mut sink| {
            Ok((|| {
                let scalar = sink.add("I32", RawValues(Vec::new()))?;
                let scalar = sink.add("ScalarElement", scalar)?;
                let scalar = sink.add("ScalarInfo", scalar)?;
                let observe = sink.add("Observe", RawValues(Vec::new()))?;
                let a = sink.add("ValueId", 0_i64)?;
                let b = sink.add("ValueId", 1_i64)?;
                let x = sink.add("ValueId", 2_i64)?;
                let y = sink.add("ValueId", 3_i64)?;
                let inputs = [Name::Value(a), Name::Value(b)];
                let add = operation(
                    &mut sink,
                    0,
                    Code::Scalar(Payload::Binary(BinaryOperator::Add)),
                    &inputs,
                )?;
                let sub = operation(
                    &mut sink,
                    1,
                    Code::Scalar(Payload::Binary(BinaryOperator::Subtract)),
                    &inputs,
                )?;
                outputs(&mut sink, add, &[Name::Value(x)])?;
                outputs(&mut sink, sub, &[Name::Value(y)])?;
                sink.set("ValueType", x, scalar)?;
                sink.set("ValueType", y, scalar)?;
                let body = region(&mut sink, 0, &[add, sub])?;
                parameters(
                    &mut sink,
                    body,
                    &[
                        Parameter {
                            name: Name::Value(a),
                            info: scalar,
                            diet: observe,
                        },
                        Parameter {
                            name: Name::Value(b),
                            info: scalar,
                            diet: observe,
                        },
                    ],
                )?;
                yields(&mut sink, body, &[Name::Value(x), Name::Value(y)])?;
                let element = sink.add("I32", RawValues(Vec::new()))?;
                let element = sink.add("ScalarElement", element)?;
                let returns = sink.add("ReturnScalar", element)?;
                sink.set("RegionReturnType", (body.0, 0_i64), returns)?;
                sink.set("RegionReturnType", (body.0, 1_i64), returns)?;
                let function = sink.add("FunctionId", (0_i64, "example"))?;
                sink.set_function_root(function, body.0);
                sink.set("FunctionResultCount", function, 2_i64)?;
                sink.set("FunctionResultType", (function, 0_i64), returns)?;
                sink.set("FunctionResultType", (function, 1_i64), returns)?;
                sink.set_entry(function, 0_i64);
                Ok::<_, OptimizeError>((add, sub))
            })())
        })
        .unwrap()
        .unwrap();
    assert_ne!(add, sub);
    let query = Query(&graph);
    for slot in 0_i64..2 {
        assert_eq!(
            query.0.input(add.0, slot).unwrap(),
            query.0.input(sub.0, slot).unwrap()
        );
    }
    let add_code = graph.code(add.0).unwrap();
    let sub_code = graph.code(sub.0).unwrap();
    assert_eq!(add_code, Code::Scalar(Payload::Binary(BinaryOperator::Add)));
    assert_eq!(sub_code, Code::Scalar(Payload::Binary(BinaryOperator::Subtract)));
    crate::egglog::analysis::run(&mut graph).unwrap();
    let root = test_graph::root(&graph);
    let live = crate::egglog::analysis::liveness(&mut graph, &[root]).unwrap();
    let query = Query(&graph);
    let input = query.0.input(add.0, 0_i64).unwrap();
    assert!(crate::egglog::analysis::available_before(query, add.0, input).unwrap());
    let name = query.name(input).unwrap();
    assert!(live.after(root, 0).contains(&name));
    assert!(!live.after(root, 1).contains(&name));
    let output = query.0.output(add.0, 0_i64).unwrap();
    graph.update(|mut sink| sink.set_input(sub.0, 0_i64, output)).unwrap();
    crate::egglog::analysis::run(&mut graph).unwrap();
    assert!(crate::egglog::analysis::uses_output_of(Query(&graph), sub.0, add.0).unwrap());
    graph.update(|mut sink| sink.set_input(sub.0, 0_i64, input)).unwrap();
    crate::egglog::analysis::run(&mut graph).unwrap();
    assert!(!crate::egglog::analysis::uses_output_of(Query(&graph), sub.0, add.0).unwrap());
    let functions = graph.get_function_names();
    for obsolete in ["Block", "Let", "MemoryProgram", "RenameProgram"] {
        assert!(
            !functions.iter().any(|name| name == obsolete),
            "flat graph registered {obsolete}"
        );
    }
}
