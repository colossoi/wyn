//! Deterministic CPU oracle for semantic segmented operations.
//!
//! This deliberately executes semantic values rather than scheduled kernels;
//! optional adapter tests compare backend readback against the same oracle.

use crate::egir::program::SemanticFunc;
use crate::egir::reify::Segmented;
use crate::egir::soac::screma;
use crate::egir::types::{ENode, NodeId, PureOp, RegionId, Semantic, SkeletonTerminator};
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
    program: &'a Segmented,
}

impl<'a> RegionExecutor<'a> {
    pub fn new(program: &'a Segmented) -> Self {
        Self { program }
    }

    pub fn call(&self, region: &RegionId, arguments: &[Value]) -> Result<Value, String> {
        let body = self.program.region(*region).ok_or_else(|| format!("unknown EGIR region {region}"))?;
        let SkeletonTerminator::Return(Some(result)) =
            body.graph.skeleton.blocks[body.graph.skeleton.entry].term
        else {
            return Err(format!("region `{}` is not a pure return region", body.name));
        };
        let mut memo = LookupMap::new();
        self.eval_node(body, result, arguments, &mut memo)
    }

    /// Invoke a canonical Screma lambda and unpack its logical result fields.
    fn call_lambda(
        &self,
        lambda: &screma::Lambda,
        parameters: &[Value],
        captures: &[Value],
    ) -> Result<Vec<Value>, String> {
        if parameters.len() != lambda.parameter_types.len() {
            return Err(format!(
                "lambda expects {} parameters, found {}",
                lambda.parameter_types.len(),
                parameters.len()
            ));
        }
        if captures.len() != lambda.capture_count() {
            return Err(format!(
                "lambda expects {} captures, found {}",
                lambda.capture_count(),
                captures.len()
            ));
        }
        if lambda.is_identity() {
            return Ok(parameters.to_vec());
        }
        let mut arguments = Vec::with_capacity(parameters.len() + captures.len());
        arguments.extend_from_slice(parameters);
        arguments.extend_from_slice(captures);
        let body = lambda.seg_body().expect("non-identity lambda has no region");
        let packed = self.call(&body.region, &arguments)?;
        match lambda.result_types.len() {
            1 => Ok(vec![packed]),
            arity => match packed {
                Value::Tuple(fields) if fields.len() == arity => Ok(fields),
                value => Err(format!(
                    "{arity}-result lambda returned an incompatible value {value:?}"
                )),
            },
        }
    }

    fn eval_node(
        &self,
        region: &SemanticFunc,
        node: NodeId,
        arguments: &[Value],
        memo: &mut LookupMap<NodeId, Value>,
    ) -> Result<Value, String> {
        if let Some(value) = memo.get(&node) {
            return Ok(value.clone());
        }
        let value = match &region.graph.nodes[node].kind {
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
                match (operator, values.as_slice()) {
                    (crate::op::BinaryOperator::Add, [left, right]) => {
                        Ok(Value::Int(left.wrapping_add(*right)))
                    }
                    (crate::op::BinaryOperator::Subtract, [left, right]) => {
                        Ok(Value::Int(left.wrapping_sub(*right)))
                    }
                    (crate::op::BinaryOperator::Multiply, [left, right]) => {
                        Ok(Value::Int(left.wrapping_mul(*right)))
                    }
                    (crate::op::BinaryOperator::Less, [left, right]) => Ok(Value::Bool(left < right)),
                    (crate::op::BinaryOperator::Equal, [left, right]) => Ok(Value::Bool(left == right)),
                    _ => Err(format!("unsupported integer operator `{operator}`")),
                }
            }
            PureOp::Call(callee) => {
                if !self.program.contains_region(*callee) {
                    return Err(format!("unknown EGIR region {callee:?}"));
                }
                self.call(callee, values)
            }
            _ => Err(format!("unsupported pure region operation {op:?}")),
        }
    }
}

/// Execute a capture-free, pointwise canonical Screma on concrete lanes.
pub(crate) fn execute_map_screma(
    program: &Segmented,
    op: &screma::Op<Semantic>,
    inputs: &[Vec<Value>],
) -> Result<Vec<Vec<Value>>, String> {
    op.validate()?;
    if !op.is_map() {
        return Err("semantic map executor received a collective Screma".into());
    }
    if inputs.len() != op.inputs.len() {
        return Err(format!(
            "map expects {} input arrays, found {}",
            op.inputs.len(),
            inputs.len()
        ));
    }
    let len = inputs.first().map_or(0, Vec::len);
    if inputs.iter().any(|input| input.len() != len) {
        return Err("map input arrays have different lengths".into());
    }
    if op.form.pre.capture_count() != 0 || op.form.post.capture_count() != 0 {
        return Err("map executor requires capture-free lambdas".into());
    }

    let executor = RegionExecutor::new(program);
    let mut outputs = vec![Vec::with_capacity(len); op.form.result_count()];
    for index in 0..len {
        let parameters = inputs.iter().map(|input| input[index].clone()).collect::<Vec<_>>();
        let mapped = executor.call_lambda(&op.form.pre, &parameters, &[])?;
        let results = executor.call_lambda(&op.form.post, &mapped, &[])?;
        if results.len() != outputs.len() {
            return Err(format!(
                "map produced {} fields, expected {}",
                results.len(),
                outputs.len()
            ));
        }
        for (output, result) in outputs.iter_mut().zip(results) {
            output.push(result);
        }
    }
    Ok(outputs)
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
