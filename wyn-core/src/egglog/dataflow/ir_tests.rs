use super::{code, operation, outputs, parameters, region, yields, Code, Name, Parameter};
use crate::egglog::memory::load;
use crate::egglog::payload::Payload;
use crate::egglog::query::Query;
use crate::egglog::OptimizeError;
use crate::op::BinaryOperator;
use egglog_engine::{EGraph, RawValues, Write};

#[test]
fn operations_share_connectivity_without_sharing_occurrences_or_payloads() {
    let mut graph = EGraph::default();
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
                sink.add("Function", function)?;
                sink.set("FunctionRoot", function, body.0)?;
                sink.set("FunctionResultCount", function, 2_i64)?;
                sink.set("FunctionResultType", (function, 0_i64), returns)?;
                sink.set("FunctionResultType", (function, 1_i64), returns)?;
                sink.set("EntryOrder", function, 0_i64)?;
                Ok::<_, OptimizeError>((add, sub))
            })())
        })
        .unwrap()
        .unwrap();
    assert_ne!(add, sub);
    let query = Query(&graph);
    for slot in 0_i64..2 {
        assert_eq!(
            query.required("Input", (add.0, slot)).unwrap(),
            query.required("Input", (sub.0, slot)).unwrap()
        );
    }
    let add_code = code(&graph, query.required("OperationPayload", add.0).unwrap());
    let sub_code = code(&graph, query.required("OperationPayload", sub.0).unwrap());
    assert_eq!(add_code, Code::Scalar(Payload::Binary(BinaryOperator::Add)));
    assert_eq!(sub_code, Code::Scalar(Payload::Binary(BinaryOperator::Subtract)));
    crate::egglog::analysis::run(&mut graph).unwrap();
    let query = Query(&graph);
    let input = query.required("Input", (add.0, 0_i64)).unwrap();
    assert!(query.contains("AvailableAt", (add.0, input)).unwrap());
    assert!(query.contains("NameLiveAfter", (add.0, input)).unwrap());
    assert!(!query.contains("NameLiveAfter", (sub.0, input)).unwrap());
    let output = query.required("Output", (add.0, 0_i64)).unwrap();
    graph
        .update(|mut sink| {
            sink.remove("Input", (sub.0, 0_i64))?;
            sink.set("Input", (sub.0, 0_i64), output)
        })
        .unwrap();
    crate::egglog::analysis::run(&mut graph).unwrap();
    assert!(Query(&graph).contains("OperationDependency", (sub.0, add.0)).unwrap());
    graph
        .update(|mut sink| {
            sink.remove("Input", (sub.0, 0_i64))?;
            sink.set("Input", (sub.0, 0_i64), input)
        })
        .unwrap();
    crate::egglog::analysis::run(&mut graph).unwrap();
    assert!(!Query(&graph).contains("OperationDependency", (sub.0, add.0)).unwrap());
    let functions = graph.get_function_names();
    for obsolete in ["Block", "Let", "MemoryProgram", "RenameProgram"] {
        assert!(
            !functions.iter().any(|name| name == obsolete),
            "flat graph registered {obsolete}"
        );
    }
}
