//! Flat executable connectivity. Rust payloads contain attributes, never bodies or operands.
use crate::egglog::graph::{Graph, State};
use crate::egglog::payload::{publish_properties, register, Payload};
use crate::egglog::OptimizeError;
use crate::types::Type;
use egglog_engine::sort::Boxed;
use egglog_engine::{Core, Value, Write};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::egglog) struct OperationId(pub Value);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::egglog) struct RegionId(pub Value);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::egglog) enum Name {
    Value(Value),
    Accumulator(Value),
}

impl Name {
    pub fn fact(self, sink: &mut State<'_, '_, '_>) -> Result<Value, OptimizeError> {
        Ok(match self {
            Self::Value(value) => sink.add("ValueNameRef", value)?,
            Self::Accumulator(value) => sink.add("AccNameRef", value)?,
        })
    }
}

#[derive(Clone, Copy)]
pub(in crate::egglog) struct Parameter {
    pub name: Name,
    pub info: Value,
    pub diet: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::egglog) enum Bounds {
    Proven,
    Ignore,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(in crate::egglog) enum Code {
    Scalar(Payload),
    Constant {
        ty: Type,
        bits: i64,
    },
    Product,
    Project {
        index: usize,
    },
    Length {
        axis: usize,
    },
    Index,
    /// Reads a declared traversal input at the lane coordinate.
    TraversalInput,
    Scratch,
    IndexValue,
    Call,
    Branch,
    CountedLoop,
    WhileLoop,
    Parallel,
    WithAcc,
    BucketScatter {
        coordinates: Vec<Vec<i64>>,
        depth: usize,
    },
    View,
    ArrayLiteral,
    Range,
    Copy,
    Update {
        bounds: Bounds,
    },
    Accumulate {
        bounds: Bounds,
    },
}

impl Code {
    pub fn has_expression_identity(&self) -> bool {
        matches!(
            self,
            Self::Scalar(_)
                | Self::Constant { .. }
                | Self::Product
                | Self::Project { .. }
                | Self::Length { .. }
                | Self::IndexValue
        )
    }
}

pub(in crate::egglog) fn load(graph: &mut Graph) -> Result<(), OptimizeError> {
    register(graph)?;
    graph.parse_and_run_program(Some("shared dataflow".into()), include_str!("../flow.egg"))?;
    graph.parse_and_run_program(Some("flat dataflow".into()), include_str!("../memory/schema.egg"))?;
    graph.register_fields()?;
    publish_properties(graph)?;
    crate::egglog::analysis::load(graph)?;
    super::edit::load(graph)?;
    crate::egglog::fusion::load(graph)?;
    Ok(())
}

pub(in crate::egglog) fn operation(
    sink: &mut State<'_, '_, '_>,
    id: i64,
    code: Code,
    inputs: &[Name],
) -> Result<OperationId, OptimizeError> {
    let operation = sink.add("OperationId", id)?;
    operation_at(sink, OperationId(operation), code, inputs)
}

pub(in crate::egglog) fn operation_at(
    sink: &mut State<'_, '_, '_>,
    operation: OperationId,
    code: Code,
    inputs: &[Name],
) -> Result<OperationId, OptimizeError> {
    let operation = operation.0;

    if matches!(code, Code::ArrayLiteral) {
        sink.set("InstructionWork", operation, inputs.len() as i64)?;
    }
    properties(sink, operation, code)?;
    let inputs = inputs.iter().map(|input| input.fact(sink)).collect::<Result<Vec<_>, _>>()?;
    sink.set_inputs(operation, inputs)?;
    Ok(OperationId(operation))
}

pub(in crate::egglog) fn properties(
    sink: &mut State<'_, '_, '_>,
    operation: Value,
    code: Code,
) -> Result<(), OptimizeError> {
    if matches!(
        code,
        Code::Scalar(_)
            | Code::Constant { .. }
            | Code::Product
            | Code::Project { .. }
            | Code::Length { .. }
            | Code::Index
            | Code::TraversalInput
            | Code::IndexValue
            | Code::View
            | Code::Range
            | Code::Update { .. }
            | Code::Accumulate { .. }
    ) {
        sink.set("InstructionWork", operation, 1_i64)?;
    }
    match &code {
        Code::Scalar(payload) => {
            let payload = sink.base_to_value(Boxed(*payload));
            sink.set("ScalarPayload", operation, payload)?;
        }
        Code::Constant { bits, .. } => {
            sink.set("LiteralBits", operation, *bits)?;
            sink.add("ConstantOperation", operation)?;
            sink.add("DiscardableOperation", operation)?;
            sink.add("RepeatableOperation", operation)?;
        }
        Code::Product => {
            sink.add("ProductOperation", operation)?;
            sink.add("DiscardableOperation", operation)?;
            sink.add("RepeatableOperation", operation)?;
        }
        Code::Project { index } => {
            sink.add("ProjectionOperation", (operation, *index as i64))?;
            sink.add("DiscardableOperation", operation)?;
            sink.add("RepeatableOperation", operation)?;
        }
        Code::Length { axis } => {
            sink.add("LengthOperation", (operation, *axis as i64))?;
            sink.add("DiscardableOperation", operation)?;
            sink.add("RepeatableOperation", operation)?;
        }
        Code::Index | Code::TraversalInput => {
            sink.add("IndexOperation", operation)?;
            sink.add("ReadsInput", (operation, 0_i64))?;
            sink.add("OrderedOperation", operation)?;
        }
        Code::Scratch | Code::ArrayLiteral | Code::Range => {
            sink.add("FreshOutput", (operation, 0_i64))?;
            match code {
                Code::Scratch => {
                    sink.add("ScratchAllocation", operation)?;
                }
                Code::ArrayLiteral => {
                    sink.add("ArrayLiteralOperation", operation)?;
                }
                Code::Range => {
                    sink.add("RangeOperation", operation)?;
                    sink.add("DiscardableOperation", operation)?;
                    sink.add("RepeatableOperation", operation)?;
                }
                _ => {}
            }
        }
        Code::IndexValue => {
            sink.add("IndexCalculationOperation", operation)?;
            sink.add("DiscardableOperation", operation)?;
            sink.add("RepeatableOperation", operation)?;
        }
        Code::Call => {
            sink.add("CallOperation", operation)?;
        }
        Code::Branch => {
            sink.add("BranchOperation", operation)?;
        }
        Code::CountedLoop => {
            sink.add("CountedLoopOperation", operation)?;
        }
        Code::WhileLoop => {
            sink.add("WhileLoopOperation", operation)?;
            sink.add("OrderedOperation", operation)?;
        }
        Code::Parallel => {
            sink.add("ParallelOperation", operation)?;
        }
        Code::WithAcc => {
            sink.add("AccumulatorOperation", operation)?;
        }
        Code::BucketScatter { .. } => {
            sink.add("OrderedOperation", operation)?;
        }
        Code::View => {
            sink.add("ViewOperation", operation)?;
            sink.add("AliasesInput", (operation, 0_i64, 0_i64))?;
            sink.add("DiscardableOperation", operation)?;
        }
        Code::Copy => {
            sink.add("MaterializedCopyOperation", operation)?;
            sink.add("PreservesInputShape", (operation, 0_i64, 0_i64))?;
            sink.add("ReadsInput", (operation, 0_i64))?;
            sink.add("FreshOutput", (operation, 0_i64))?;
        }
        Code::Update { .. } | Code::Accumulate { .. } => {
            sink.add("PreservesInputShape", (operation, 0_i64, 0_i64))?;
            if matches!(code, Code::Accumulate { .. }) {
                sink.add("ConsumesInput", (operation, 0_i64))?;
            }
            sink.add("WritesInput", (operation, 0_i64))?;
            sink.add("AliasesInput", (operation, 0_i64, 0_i64))?;
        }
    }
    sink.set_code(operation, code)?;
    Ok(())
}

pub(in crate::egglog) fn outputs(
    sink: &mut State<'_, '_, '_>,
    operation: OperationId,
    outputs: &[Name],
) -> Result<(), OptimizeError> {
    let names = outputs.iter().map(|name| name.fact(sink)).collect::<Result<Vec<_>, _>>()?;
    sink.set_outputs(operation.0, names)?;
    Ok(())
}

pub(in crate::egglog) fn region(
    sink: &mut State<'_, '_, '_>,
    id: i64,
    operations: &[OperationId],
) -> Result<RegionId, OptimizeError> {
    let region = sink.add("RegionId", id)?;

    sink.add("FlowScope", region)?;
    for (position, operation) in operations.iter().enumerate() {
        sink.set_region(operation.0, region);
        sink.set_order(operation.0, position as i64);
    }
    Ok(RegionId(region))
}

pub(in crate::egglog) fn parameters(
    sink: &mut State<'_, '_, '_>,
    region: RegionId,
    parameters: &[Parameter],
) -> Result<(), OptimizeError> {
    let mut names = Vec::with_capacity(parameters.len());
    for (index, parameter) in parameters.iter().enumerate() {
        match parameter.name {
            Name::Value(value) => sink.set_value_type(value, parameter.info)?,
            Name::Accumulator(value) => sink.set("AccumulatorType", value, parameter.info)?,
        }
        let name = parameter.name.fact(sink)?;
        names.push(name);
        sink.set("ParameterDiet", (region.0, index as i64), parameter.diet)?;
    }
    sink.set_parameters(region.0, names)?;
    Ok(())
}

pub(in crate::egglog) fn yields(
    sink: &mut State<'_, '_, '_>,
    region: RegionId,
    values: &[Name],
) -> Result<(), OptimizeError> {
    let values = values.iter().map(|value| value.fact(sink)).collect::<Result<Vec<_>, _>>()?;
    sink.set_yields(region.0, values)?;
    Ok(())
}

#[cfg(test)]
#[path = "ir_tests.rs"]
mod tests;
