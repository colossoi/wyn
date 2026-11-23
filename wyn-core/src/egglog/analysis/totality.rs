//! Prove literal shift bounds from the definitions of the two actual operands.
use super::{Graph, OptimizeError, Origin, Query};
use crate::egglog::dataflow::ir::Code;
use crate::egglog::payload::Payload;
use crate::op::BinaryOperator;
use egglog_engine::{RawValues, Write};

pub(super) fn publish_literal_shifts(graph: &mut Graph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut supported = Vec::new();
    for kind in ["I32", "U32"] {
        let Some(primitive) = query.constructor(kind, RawValues(Vec::new()))? else {
            continue;
        };
        let Some(element) = query.constructor("ScalarElement", primitive)? else {
            continue;
        };
        supported.extend(query.constructor("ScalarInfo", element)?);
    }
    let mut total = Vec::new();
    for (operation, code) in graph.codes() {
        if !matches!(
            code,
            Code::Scalar(Payload::Binary(
                BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight
            ))
        ) || !query.contains("LiveOperation", operation)?
        {
            continue;
        }
        let inputs = graph.inputs(operation);
        let [value, amount] = inputs.as_slice() else {
            continue;
        };
        let (Some(value_name), Some(amount_name)) = (query.value_name(*value)?, query.value_name(*amount)?)
        else {
            continue;
        };
        let Some(ty) = query.lookup("ValueType", value_name)? else {
            continue;
        };
        if !supported.contains(&ty) || query.lookup("ValueType", amount_name)? != Some(ty) {
            continue;
        }
        let Some(Origin::Result(literal, 0)) = query.live_origin(*amount)? else {
            continue;
        };
        if let Some(bits) = query.lookup("LiteralBits", literal)? {
            if (0..32).contains(&graph.value_to_base::<i64>(bits)) {
                total.push(operation);
            }
        }
    }
    graph.update(|mut sink| {
        for operation in total {
            sink.set("OperationTotal", operation, true)?;
        }
        Ok(())
    })?;
    Ok(())
}
