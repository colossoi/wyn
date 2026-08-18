//! Deterministic CPU oracle for semantic segmented operations.
//!
//! This deliberately executes semantic values rather than scheduled kernels;
//! optional adapter tests compare backend readback against the same oracle.

use crate::ast;
use crate::egir::program::Func;
use crate::egir::reify::Segmented;
use crate::egir::soac::{hist, screma};
use crate::egir::types::{
    PureOp, ResultBinding, ResultDestination, Semantic, SkeletonTerminator, ValueId, ValueKind,
};
use crate::op;
use crate::ssa;
use crate::{FunctionId, LookupMap};

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

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BucketExecution {
    pub buckets: Vec<Vec<Value>>,
    pub counts: Vec<u32>,
    pub overflow: bool,
}

impl<'a> RegionExecutor<'a> {
    pub fn new(program: &'a Segmented) -> Self {
        Self { program }
    }

    pub fn call(&self, region: &FunctionId, arguments: &[Value]) -> Result<Value, String> {
        let body = self.program.region(*region).ok_or_else(|| format!("unknown EGIR region {region}"))?;
        let SkeletonTerminator::Return(Some(result)) =
            &body.graph.skeleton.blocks[body.graph.skeleton.entry].term
        else {
            return Err(format!("region `{}` is not a pure return region", body.name));
        };
        let mut memo = LookupMap::new();
        self.eval_result(body, result, arguments, &mut memo)
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
        region: &Func<Semantic>,
        node: ValueId,
        arguments: &[Value],
        memo: &mut LookupMap<ValueId, Value>,
    ) -> Result<Value, String> {
        if let Some(value) = memo.get(&node) {
            return Ok(value.clone());
        }
        let value = match &region.graph.nodes[node].kind {
            ValueKind::FuncParam { parameter } => {
                let index = parameter.index();
                arguments.get(index).cloned().ok_or_else(|| format!("missing region argument {index}"))?
            }
            ValueKind::Constant(ssa::types::ConstantValue::I32(value)) => Value::Int(*value as i64),
            ValueKind::Constant(ssa::types::ConstantValue::U32(value)) => Value::Int(*value as i64),
            ValueKind::Constant(ssa::types::ConstantValue::Bool(value)) => Value::Bool(*value),
            ValueKind::Constant(ssa::types::ConstantValue::F32(_)) => {
                return Err("floating-point region execution is not needed by semantic tests".into())
            }
            ValueKind::Union { left, .. } => self.eval_node(region, *left, arguments, memo)?,
            ValueKind::Pure { op, operands } => {
                let values: Result<Vec<_>, _> = operands
                    .iter()
                    .map(|operand| self.eval_node(region, *operand, arguments, memo))
                    .collect();
                self.eval_pure(op, &values?)?
            }
            ValueKind::CallResult { call, slot } => {
                let call = region.graph.call(*call);
                let values = call
                    .arguments()
                    .iter()
                    .map(|argument| {
                        let argument = argument
                            .value()
                            .ok_or_else(|| "pure region executor cannot pass a place".to_owned())?;
                        self.eval_node(region, argument, arguments, memo)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let returned = self.call(&call.callee(), &values)?;
                flattened_value(&returned)
                    .get(slot.index())
                    .copied()
                    .cloned()
                    .ok_or_else(|| format!("call result slot {} is absent", slot.index()))?
            }
            ValueKind::BlockParam { .. }
            | ValueKind::PlaceLength { .. }
            | ValueKind::PlaceView { .. }
            | ValueKind::SideEffectResult => {
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
                    (op::BinaryOperator::Add, [left, right]) => Ok(Value::Int(left.wrapping_add(*right))),
                    (op::BinaryOperator::Subtract, [left, right]) => {
                        Ok(Value::Int(left.wrapping_sub(*right)))
                    }
                    (op::BinaryOperator::Multiply, [left, right]) => {
                        Ok(Value::Int(left.wrapping_mul(*right)))
                    }
                    (op::BinaryOperator::Remainder, [left, right]) => {
                        Ok(Value::Int(left.wrapping_rem(*right)))
                    }
                    (op::BinaryOperator::Less, [left, right]) => Ok(Value::Bool(left < right)),
                    (op::BinaryOperator::GreaterEqual, [left, right]) => Ok(Value::Bool(left >= right)),
                    (op::BinaryOperator::Equal, [left, right]) => Ok(Value::Bool(left == right)),
                    _ => Err(format!("unsupported integer operator `{operator}`")),
                }
            }
            _ => Err(format!("unsupported pure region operation {op:?}")),
        }
    }

    fn eval_result(
        &self,
        region: &Func<Semantic>,
        result: &ResultBinding<polytype::Type<ast::TypeName>>,
        arguments: &[Value],
        memo: &mut LookupMap<ValueId, Value>,
    ) -> Result<Value, String> {
        if result.is_product() {
            return (0..result.field_count())
                .map(|index| {
                    self.eval_result(
                        region,
                        &result.field(index).expect("result field disappeared"),
                        arguments,
                        memo,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Value::Tuple);
        }
        let mut value = None;
        result.for_each_destination(|_, destination| {
            if let ResultDestination::ReturnValue(result) = destination {
                value = Some(*result);
            }
        });
        self.eval_node(
            region,
            value.ok_or_else(|| "pure region result is destination-passed".to_owned())?,
            arguments,
            memo,
        )
    }
}

fn flattened_value(value: &Value) -> Vec<&Value> {
    match value {
        Value::Tuple(fields) => fields.iter().flat_map(flattened_value).collect(),
        value => vec![value],
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

/// Execute one canonical guarded bucket-insert histogram over a fixed ranked
/// domain. Inputs are flattened in their own row-major coordinate spaces;
/// `SoacInputType::dimensions` selects the logical coordinates visible to
/// each input. This is a deterministic semantic oracle, not an ordering model
/// for the parallel atomic implementation.
pub(crate) fn execute_bucket_hist(
    program: &Segmented,
    op: &hist::Op<Semantic>,
    domain: &[usize],
    inputs: &[Vec<Value>],
    mut buckets: Vec<Vec<Value>>,
) -> Result<BucketExecution, String> {
    let [operation] = op.form.operations.as_slice() else {
        return Err("bucket executor requires exactly one histogram operation".into());
    };
    if !matches!(operation.update, hist::Update::BucketInsert { .. })
        || !matches!(operation.emission, hist::Emission::Guarded)
    {
        return Err("bucket executor requires one guarded bucket insertion".into());
    }
    if inputs.len() != op.inputs.len() {
        return Err(format!(
            "bucket executor expects {} inputs, found {}",
            op.inputs.len(),
            inputs.len()
        ));
    }
    if op.form.bucket.capture_count() != 0 {
        return Err("bucket executor currently requires a capture-free envelope".into());
    }
    let capacity = buckets.first().map_or(0, Vec::len);
    if buckets.iter().any(|bucket| bucket.len() != capacity) {
        return Err("bucket destination rows have different capacities".into());
    }
    for (input, values) in op.inputs.iter().zip(inputs) {
        let expected = input.dimensions.iter().try_fold(1usize, |product, dimension| {
            let extent = domain
                .get(usize::from(*dimension))
                .copied()
                .ok_or_else(|| format!("input dimension {} is outside rank {}", dimension, domain.len()))?;
            product.checked_mul(extent).ok_or_else(|| "ranked input size overflows usize".to_string())
        })?;
        if values.len() != expected {
            return Err(format!(
                "ranked input {:?} has {} values, expected {expected}",
                input.dimensions,
                values.len()
            ));
        }
    }

    let lane_count = domain.iter().try_fold(1usize, |product, extent| {
        product.checked_mul(*extent).ok_or_else(|| "ranked domain size overflows usize".to_string())
    })?;
    let executor = RegionExecutor::new(program);
    let mut counts = vec![0u32; buckets.len()];
    let mut overflow = false;
    for lane in 0..lane_count {
        let mut remaining = lane;
        let mut coordinates = vec![0usize; domain.len()];
        for dimension in (0..domain.len()).rev() {
            let extent = domain[dimension];
            if extent == 0 {
                return Ok(BucketExecution {
                    buckets,
                    counts,
                    overflow,
                });
            }
            coordinates[dimension] = remaining % extent;
            remaining /= extent;
        }
        let parameters = op
            .inputs
            .iter()
            .zip(inputs)
            .map(|(input, values)| {
                let index = input.dimensions.iter().try_fold(0usize, |index, dimension| {
                    let dimension = usize::from(*dimension);
                    index
                        .checked_mul(domain[dimension])
                        .and_then(|index| index.checked_add(coordinates[dimension]))
                        .ok_or_else(|| "ranked input index overflows usize".to_string())
                })?;
                values
                    .get(index)
                    .cloned()
                    .ok_or_else(|| format!("ranked input index {index} is out of bounds"))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let results = executor.call_lambda(&op.form.bucket, &parameters, &[])?;
        let [Value::Bool(active), Value::Int(key), value] = results.as_slice() else {
            return Err(format!(
                "guarded bucket envelope returned incompatible values {results:?}"
            ));
        };
        if !*active {
            continue;
        }
        let Ok(bucket) = usize::try_from(*key) else {
            overflow = true;
            continue;
        };
        let Some(count) = counts.get_mut(bucket) else {
            overflow = true;
            continue;
        };
        let slot = *count;
        *count = count.wrapping_add(1);
        if let Some(destination) = buckets.get_mut(bucket).and_then(|bucket| bucket.get_mut(slot as usize))
        {
            *destination = value.clone();
        } else {
            overflow = true;
        }
    }
    Ok(BucketExecution {
        buckets,
        counts,
        overflow,
    })
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
