//! Deterministic CPU oracle for semantic segmented operations.
//!
//! This deliberately executes semantic values rather than scheduled kernels;
//! optional adapter tests compare backend readback against the same oracle.

use crate::egir::program::EgirRegion;
use crate::egir::types::{ENode, NodeId, PureOp, RegionId, SkeletonTerminator};
use crate::LookupMap;

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
    regions: &'a LookupMap<RegionId, EgirRegion>,
}

impl<'a> RegionExecutor<'a> {
    pub fn new(regions: &'a LookupMap<RegionId, EgirRegion>) -> Self {
        Self { regions }
    }

    pub fn call(&self, region: &RegionId, arguments: &[Value]) -> Result<Value, String> {
        let body = self
            .regions
            .get(region)
            .ok_or_else(|| format!("unknown EGIR region #{}", region.index()))?;
        let SkeletonTerminator::Return(Some(result)) = body.graph.skeleton.blocks[body.graph.skeleton.entry].term
        else {
            return Err(format!("region `{}` is not a pure return region", body.name));
        };
        let mut memo = LookupMap::new();
        self.eval_node(body, result, arguments, &mut memo)
    }

    fn eval_node(
        &self,
        region: &EgirRegion,
        node: NodeId,
        arguments: &[Value],
        memo: &mut LookupMap<NodeId, Value>,
    ) -> Result<Value, String> {
        if let Some(value) = memo.get(&node) {
            return Ok(value.clone());
        }
        let value = match &region.graph.nodes[node] {
            ENode::FuncParam { index } => arguments
                .get(*index)
                .cloned()
                .ok_or_else(|| format!("missing region argument {index}"))?,
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
            PureOp::Int(value) | PureOp::Uint(value) => value
                .parse()
                .map(Value::Int)
                .map_err(|_| format!("invalid integer literal `{value}`")),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TypeName;
    use crate::egir::program::EgirRegion;
    use crate::egir::types::{EGraph, PureOp, SkeletonTerminator};
    use polytype::Type;
    use smallvec::smallvec;

    /// Affine composition `(a,b) o (c,d) = (a*c, a*d+b)` is associative
    /// but noncommutative, making operand-order bugs visible.
    fn compose(left: &(i64, i64), right: &(i64, i64)) -> (i64, i64) {
        (
            left.0.wrapping_mul(right.0),
            left.0.wrapping_mul(right.1).wrapping_add(left.1),
        )
    }

    fn affine_region() -> (RegionId, LookupMap<RegionId, EgirRegion>) {
        let int = Type::Constructed(TypeName::Int(64), vec![]);
        let pair = Type::Constructed(TypeName::Tuple(2), vec![int.clone(), int.clone()]);
        let mut graph = EGraph::new();
        let left = graph.add_func_param(0, pair.clone());
        let right = graph.add_func_param(1, pair.clone());
        let la = graph.intern_pure(PureOp::Project { index: 0 }, smallvec![left], int.clone());
        let lb = graph.intern_pure(PureOp::Project { index: 1 }, smallvec![left], int.clone());
        let ra = graph.intern_pure(PureOp::Project { index: 0 }, smallvec![right], int.clone());
        let rb = graph.intern_pure(PureOp::Project { index: 1 }, smallvec![right], int.clone());
        let out_a = graph.intern_pure(PureOp::BinOp("*".into()), smallvec![la, ra], int.clone());
        let scaled = graph.intern_pure(PureOp::BinOp("*".into()), smallvec![la, rb], int.clone());
        let out_b = graph.intern_pure(PureOp::BinOp("+".into()), smallvec![scaled, lb], int.clone());
        let result = graph.intern_pure(PureOp::Tuple(2), smallvec![out_a, out_b], pair.clone());
        graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(result));
        let id = RegionId::from_index(0);
        let mut regions = LookupMap::new();
        regions.insert(
            id,
            EgirRegion {
                name: "affine_compose".to_string(),
                params: vec![(pair.clone(), "left".into()), (pair.clone(), "right".into())],
                return_ty: pair,
                graph,
                control_headers: LookupMap::new(),
            },
        );
        (id, regions)
    }

    #[test]
    fn region_executor_runs_noncommutative_reduce_and_scan_values() {
        let (region, regions) = affine_region();
        let executor = RegionExecutor::new(&regions);
        for len in [0usize, 1, 63, 64, 65, 513] {
            let inputs: Vec<_> = (0..len)
                .map(|index| Value::Tuple(vec![Value::Int(2), Value::Int(index as i64 + 1)]))
                .collect();
            let neutral = Value::Tuple(vec![Value::Int(1), Value::Int(0)]);
            let mut reduction = neutral.clone();
            let mut scan = Vec::new();
            for value in &inputs {
                reduction = executor.call(&region, &[reduction, value.clone()]).unwrap();
                scan.push(reduction.clone());
            }
            assert_eq!(scan.last().cloned().unwrap_or(neutral), reduction, "len={len}");
        }
    }

    #[test]
    fn semantic_reduce_and_scan_cover_dispatch_boundaries() {
        for len in [0usize, 1, 63, 64, 65, 513] {
            let values: Vec<_> = (0..len).map(|i| (2, i as i64 + 1)).collect();
            let reduced = reduce(&values, (1, 0), compose);
            let scanned = inclusive_scan(&values, (1, 0), compose);
            assert_eq!(scanned.last().copied().unwrap_or((1, 0)), reduced, "len={len}");

            let reversed = reduce(&values.iter().cloned().rev().collect::<Vec<_>>(), (1, 0), compose);
            if len > 1 {
                assert_ne!(reduced, reversed, "oracle must detect noncommutative reordering");
            }
        }
    }

    #[test]
    fn semantic_scanomap_routes_both_outputs() {
        let (mapped, scanned) = scanomap(&[1, 2, 3, 4], |x| x * 3, 0, |a, b| a + b);
        assert_eq!(mapped, [3, 6, 9, 12]);
        assert_eq!(scanned, [3, 9, 18, 30]);
    }

    #[test]
    fn semantic_filter_and_ordered_scatter_match_source_contract() {
        assert_eq!(filter(&[1, 2, 3, 4, 5], |x| x % 2 == 1), [1, 3, 5]);
        assert_eq!(scatter(&[0, 0, 0], &[(1, 4), (1, 9), (8, 7)]), [0, 9, 0]);
    }

    #[test]
    fn stepped_range_semantics_are_value_level() {
        let range: Vec<i32> = (0..6).map(|i| 3 + i * 4).collect();
        assert_eq!(map(&range, |x| x * 2), [6, 14, 22, 30, 38, 46]);
    }
}
