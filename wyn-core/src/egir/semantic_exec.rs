//! Deterministic CPU oracle for semantic segmented operations.
//!
//! This deliberately executes semantic values rather than scheduled kernels;
//! optional adapter tests compare backend readback against the same oracle.

use crate::egir::program::SemanticRegion;
use crate::egir::types::{ENode, NodeId, PureOp, RegionId, SkeletonTerminator};
use crate::LookupMap;

#[cfg(test)]
#[path = "semantic_exec_tests.rs"]
mod semantic_exec_tests;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Tuple(Vec<Value>),
}

/// Test-only interpreter for pure EGIR regions. It deliberately evaluates the
/// typed region arena used by SegBody/SegBinOp, so semantic tests exercise the
/// representation rather than parallel Rust closures alone.
pub struct RegionExecutor<'a> {
    regions: &'a LookupMap<RegionId, SemanticRegion>,
}

impl<'a> RegionExecutor<'a> {
    pub fn new(regions: &'a LookupMap<RegionId, SemanticRegion>) -> Self {
        Self { regions }
    }

    pub fn call(&self, region: &RegionId, arguments: &[Value]) -> Result<Value, String> {
        let body =
            self.regions.get(region).ok_or_else(|| format!("unknown EGIR region #{}", region.index()))?;
        let SkeletonTerminator::Return(Some(result)) =
            body.graph.skeleton.blocks[body.graph.skeleton.entry].term
        else {
            return Err(format!("region `{}` is not a pure return region", body.name));
        };
        let mut memo = LookupMap::new();
        self.eval_node(body, result, arguments, &mut memo)
    }

    fn eval_node(
        &self,
        region: &SemanticRegion,
        node: NodeId,
        arguments: &[Value],
        memo: &mut LookupMap<NodeId, Value>,
    ) -> Result<Value, String> {
        if let Some(value) = memo.get(&node) {
            return Ok(value.clone());
        }
        let value = match &region.graph.nodes[node] {
            ENode::FuncParam { index } => {
                arguments.get(*index).cloned().ok_or_else(|| format!("missing region argument {index}"))?
            }
            ENode::Constant(crate::ssa::types::ConstantValue::I32(value)) => Value::Int(*value as i64),
            ENode::Constant(crate::ssa::types::ConstantValue::U32(value)) => Value::Int(*value as i64),
            ENode::Constant(crate::ssa::types::ConstantValue::Bool(value)) => Value::Bool(*value),
            ENode::Constant(crate::ssa::types::ConstantValue::F32(_)) => {
                return Err("floating-point region execution is not needed by semantic tests".into())
            }
            ENode::Union { left, .. } => self.eval_node(region, *left, arguments, memo)?,
            ENode::Pure { op, operands } => {
                let values: Result<Vec<_>, _> = operands
                    .iter()
                    .map(|operand| self.eval_node(region, *operand, arguments, memo))
                    .collect();
                self.eval_pure(op, &values?)?
            }
            ENode::BlockParam { .. } | ENode::SideEffectResult => {
                return Err("effectful/CFG values are outside the pure region executor".into())
            }
        };
        memo.insert(node, value.clone());
        Ok(value)
    }

    fn eval_pure(&self, op: &PureOp, values: &[Value]) -> Result<Value, String> {
        let ints = || -> Result<Vec<i64>, String> {
            values
                .iter()
                .map(|value| match value {
                    Value::Int(value) => Ok(*value),
                    _ => Err("integer operator received a non-integer".into()),
                })
                .collect()
        };
        match op {
            PureOp::Int(value) | PureOp::Uint(value) => {
                value.parse().map(Value::Int).map_err(|_| format!("invalid integer literal `{value}`"))
            }
            PureOp::Bool(value) => Ok(Value::Bool(*value)),
            PureOp::Tuple(_) => Ok(Value::Tuple(values.to_vec())),
            PureOp::Project { index } => match values.first() {
                Some(Value::Tuple(fields)) => fields
                    .get(*index as usize)
                    .cloned()
                    .ok_or_else(|| format!("tuple projection {index} is out of bounds")),
                _ => Err("tuple projection received a non-tuple".into()),
            },
            PureOp::BinOp(operator) => {
                let values = ints()?;
                match (operator.as_str(), values.as_slice()) {
                    ("+", [left, right]) => Ok(Value::Int(left.wrapping_add(*right))),
                    ("-", [left, right]) => Ok(Value::Int(left.wrapping_sub(*right))),
                    ("*", [left, right]) => Ok(Value::Int(left.wrapping_mul(*right))),
                    ("<", [left, right]) => Ok(Value::Bool(left < right)),
                    ("==", [left, right]) => Ok(Value::Bool(left == right)),
                    _ => Err(format!("unsupported integer operator `{operator}`")),
                }
            }
            PureOp::Call(callee) => {
                let region = self
                    .regions
                    .iter()
                    .find(|(_, body)| body.name == *callee)
                    .map(|(id, _)| *id)
                    .ok_or_else(|| format!("unknown EGIR region `{callee}`"))?;
                self.call(&region, values)
            }
            _ => Err(format!("unsupported pure region operation {op:?}")),
        }
    }
}

pub fn map<T, U>(input: &[T], f: impl Fn(&T) -> U) -> Vec<U> {
    input.iter().map(f).collect()
}

pub fn reduce<T: Clone>(input: &[T], neutral: T, op: impl Fn(&T, &T) -> T) -> T {
    input.iter().fold(neutral, |acc, value| op(&acc, value))
}

pub fn inclusive_scan<T: Clone>(input: &[T], neutral: T, op: impl Fn(&T, &T) -> T) -> Vec<T> {
    let mut acc = neutral;
    input
        .iter()
        .map(|value| {
            acc = op(&acc, value);
            acc.clone()
        })
        .collect()
}

pub fn scanomap<T, U: Clone>(
    input: &[T],
    map_body: impl Fn(&T) -> U,
    neutral: U,
    op: impl Fn(&U, &U) -> U,
) -> (Vec<U>, Vec<U>) {
    let mapped = map(input, map_body);
    let scanned = inclusive_scan(&mapped, neutral, op);
    (mapped, scanned)
}

pub fn filter<T: Clone>(input: &[T], pred: impl Fn(&T) -> bool) -> Vec<T> {
    input.iter().filter(|value| pred(value)).cloned().collect()
}

pub fn scatter<T: Clone>(initial: &[T], updates: &[(usize, T)]) -> Vec<T> {
    let mut output = initial.to_vec();
    for (index, value) in updates {
        if let Some(slot) = output.get_mut(*index) {
            *slot = value.clone();
        }
    }
    output
}
