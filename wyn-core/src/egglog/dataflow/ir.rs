//! Flat executable connectivity. Rust payloads contain attributes, never bodies or operands.
use crate::ast::Span as SourceSpan;
use crate::egglog::payload::{publish_properties, register, Payload};
use crate::egglog::{output_error, OptimizeError};
use crate::types::Type;
use egglog_engine::ast::{Literal, Span};
use egglog_engine::prelude::{add_base_sort, BaseSort};
use egglog_engine::sort::{BaseValues, Boxed};
use egglog_engine::{Core, EGraph, FullState, TermDag, TermId, Value, Write};

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
    pub fn fact(self, sink: &mut FullState<'_, '_>) -> Result<Value, OptimizeError> {
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
    Scratch,
    Use,
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
    Uninitialized,
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

#[derive(Debug)]
struct CodeSort;

#[derive(Debug)]
struct SourceSpanSort;

impl BaseSort for SourceSpanSort {
    type Base = Boxed<SourceSpan>;

    fn name(&self) -> &str {
        "SourceSpan"
    }

    fn reconstruct_termdag(&self, values: &BaseValues, value: Value, dag: &mut TermDag) -> TermId {
        dag.lit(Literal::String(format!(
            "{:?}",
            values.unwrap::<Boxed<SourceSpan>>(value).0
        )))
    }
}

impl BaseSort for CodeSort {
    type Base = Boxed<Code>;

    fn name(&self) -> &str {
        "OperationCode"
    }

    fn reconstruct_termdag(&self, values: &BaseValues, value: Value, dag: &mut TermDag) -> TermId {
        // Payload dumps describe Rust attributes; they are not executable program syntax.
        dag.lit(Literal::String(format!(
            "{:?}",
            values.unwrap::<Boxed<Code>>(value).0
        )))
    }
}

pub(in crate::egglog) fn load(graph: &mut EGraph) -> Result<(), OptimizeError> {
    register(graph)?;
    add_base_sort(graph, SourceSpanSort, Span::Panic).map_err(|error| output_error(error.to_string()))?;
    add_base_sort(graph, CodeSort, Span::Panic).map_err(|error| output_error(error.to_string()))?;
    graph.parse_and_run_program(Some("shared dataflow".into()), include_str!("../flow.egg"))?;
    graph.parse_and_run_program(Some("flat dataflow".into()), include_str!("../memory/schema.egg"))?;
    publish_properties(graph)?;
    crate::egglog::analysis::load(graph)?;
    super::edit::load(graph)?;
    crate::egglog::fusion::load(graph)?;
    Ok(())
}

pub(in crate::egglog) fn code(graph: &EGraph, value: Value) -> Code {
    graph.value_to_base::<Boxed<Code>>(value).0
}

pub(in crate::egglog) fn operation(
    sink: &mut FullState<'_, '_>,
    id: i64,
    code: Code,
    inputs: &[Name],
) -> Result<OperationId, OptimizeError> {
    let operation = sink.add("OperationId", id)?;
    operation_at(sink, OperationId(operation), code, inputs)
}

pub(in crate::egglog) fn operation_at(
    sink: &mut FullState<'_, '_>,
    operation: OperationId,
    code: Code,
    inputs: &[Name],
) -> Result<OperationId, OptimizeError> {
    let operation = operation.0;
    sink.add("Operation", operation)?;
    if matches!(code, Code::ArrayLiteral) {
        sink.set("InstructionWork", operation, inputs.len() as i64)?;
    }
    properties(sink, operation, code)?;
    sink.set("InputCount", operation, inputs.len() as i64)?;
    for (index, input) in inputs.iter().enumerate() {
        let input = input.fact(sink)?;
        sink.set("Input", (operation, index as i64), input)?;
    }
    Ok(OperationId(operation))
}

pub(in crate::egglog) fn properties(
    sink: &mut FullState<'_, '_>,
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
            | Code::Use
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
        Code::Use => {
            sink.add("ConsumesInput", (operation, 0_i64))?;
            sink.add("CopyOperation", operation)?;
            sink.add("DiscardableOperation", operation)?;
            sink.add("RepeatableOperation", operation)?;
        }
        Code::Index => {
            sink.add("IndexOperation", operation)?;
            sink.add("ReadsInput", (operation, 0_i64))?;
            sink.add("OrderedOperation", operation)?;
        }
        Code::Scratch | Code::ArrayLiteral | Code::Range | Code::Uninitialized => {
            sink.add("FreshOutput", (operation, 0_i64))?;
            match code {
                Code::Scratch | Code::Uninitialized => {
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
    let code = sink.base_to_value(Boxed(code));
    sink.set("OperationPayload", operation, code)?;
    Ok(())
}

pub(in crate::egglog) fn outputs(
    sink: &mut FullState<'_, '_>,
    operation: OperationId,
    outputs: &[Name],
) -> Result<(), OptimizeError> {
    sink.set("OutputCount", operation.0, outputs.len() as i64)?;
    for (index, output) in outputs.iter().enumerate() {
        let output = output.fact(sink)?;
        sink.set("Output", (operation.0, index as i64), output)?;
    }
    Ok(())
}

pub(in crate::egglog) fn region(
    sink: &mut FullState<'_, '_>,
    id: i64,
    operations: &[OperationId],
) -> Result<RegionId, OptimizeError> {
    let region = sink.add("RegionId", id)?;
    sink.add("Region", region)?;
    sink.add("FlowScope", region)?;
    for (position, operation) in operations.iter().enumerate() {
        sink.set("OperationRegion", operation.0, region)?;
        sink.set("OperationOrder", operation.0, position as i64)?;
    }
    Ok(RegionId(region))
}

pub(in crate::egglog) fn parameters(
    sink: &mut FullState<'_, '_>,
    region: RegionId,
    parameters: &[Parameter],
) -> Result<(), OptimizeError> {
    sink.set("RegionParameterCount", region.0, parameters.len() as i64)?;
    for (index, parameter) in parameters.iter().enumerate() {
        match parameter.name {
            Name::Value(value) => sink.set("ValueType", value, parameter.info)?,
            Name::Accumulator(value) => sink.set("AccumulatorType", value, parameter.info)?,
        }
        let name = parameter.name.fact(sink)?;
        sink.set("RegionParameter", (region.0, index as i64), name)?;
        sink.set("ParameterDiet", (region.0, index as i64), parameter.diet)?;
    }
    Ok(())
}

pub(in crate::egglog) fn yields(
    sink: &mut FullState<'_, '_>,
    region: RegionId,
    values: &[Name],
) -> Result<(), OptimizeError> {
    sink.set("RegionYieldCount", region.0, values.len() as i64)?;
    for (index, value) in values.iter().enumerate() {
        let name = value.fact(sink)?;
        sink.set("RegionYield", (region.0, index as i64), name)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "ir_tests.rs"]
mod tests;
