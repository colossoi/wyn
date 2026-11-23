use super::{read, register, Payload};
use crate::egglog::query::Query;
use crate::egglog::{graph::Graph, memory};
use crate::op::BinaryOperator;
use egglog_engine::{sort::Boxed, Core, Write};

#[test]
fn instruction_payloads_have_no_operands_and_preserve_operator_identity() {
    let mut graph = Graph::default();
    register(&mut graph).unwrap();
    graph
        .parse_and_run_program(None, "(function TestPayload (i64) InstructionPayload :no-merge)")
        .unwrap();
    graph
        .update(|mut sink| {
            let add = sink.base_to_value(Boxed(Payload::Binary(BinaryOperator::Add)));
            let sub = sink.base_to_value(Boxed(Payload::Binary(BinaryOperator::Subtract)));
            sink.set("TestPayload", 0_i64, add)?;
            sink.set("TestPayload", 1_i64, sub)
        })
        .unwrap();
    let add = Query(&graph).required("TestPayload", (0_i64,)).unwrap();
    let sub = Query(&graph).required("TestPayload", (1_i64,)).unwrap();
    assert_ne!(add, sub);
    assert_eq!(read(&graph, add), Payload::Binary(BinaryOperator::Add));
    assert_eq!(read(&graph, sub), Payload::Binary(BinaryOperator::Subtract));
    graph
        .parse_and_run_program(
            None,
            r#"
        (check (= (TestPayload 0) (wyn-payload "+" 2)))
        (check (= (TestPayload 1) (wyn-payload "-" 2)))
        (check (!= (wyn-payload "-" 1) (wyn-payload "-" 2)))
    "#,
        )
        .unwrap();
}

#[test]
fn arithmetic_payloads_share_generic_operand_structure() {
    let mut graph = Graph::default();
    memory::load(&mut graph).unwrap();
    use crate::egglog::dataflow::ir::{operation, Code, Name};
    let (add, sub, again) = graph
        .update(|mut sink| {
            Ok((|| -> Result<_, crate::egglog::OptimizeError> {
                let a = sink.add("ValueId", 0_i64)?;
                let b = sink.add("ValueId", 1_i64)?;
                let inputs = [Name::Value(a), Name::Value(b)];
                Ok((
                    operation(
                        &mut sink,
                        0,
                        Code::Scalar(Payload::Binary(BinaryOperator::Add)),
                        &inputs,
                    )?,
                    operation(
                        &mut sink,
                        1,
                        Code::Scalar(Payload::Binary(BinaryOperator::Subtract)),
                        &inputs,
                    )?,
                    operation(
                        &mut sink,
                        2,
                        Code::Scalar(Payload::Binary(BinaryOperator::Add)),
                        &inputs,
                    )?,
                ))
            })())
        })
        .unwrap()
        .unwrap();
    assert_ne!(add, again);
    let query = Query(&graph);
    for slot in 0_i64..2 {
        assert_eq!(
            query.required("Input", (add.0, slot)).unwrap(),
            query.required("Input", (sub.0, slot)).unwrap()
        );
    }
    assert_ne!(graph.code(add.0).unwrap(), graph.code(sub.0).unwrap());
    assert_eq!(graph.code(add.0).unwrap(), graph.code(again.0).unwrap());
}
