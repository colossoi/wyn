use super::screma::{workgroup_scan, Accumulator};
// Expand scheduled phases straight into SSA instructions and structured loops.
use super::super::{error, Body, OptimizeError};
use super::{element, write_arrays};
use crate::builtins::catalog;
use crate::op::{BinaryOperator, OpTag};
use crate::types::{self, Type, TypeName};
use crate::LookupMap;
use egglog_engine::Value;

pub(super) fn compact(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    phase_owner: Value,
    phase_width: u32,
    plan: Value,
    results: &[(String, i64, Value)],
) -> Result<(), OptimizeError> {
    let operation = body.compiler.facts.filter(plan)?;
    let Some(source) = body.compiler.facts.source(operation) else {
        return Err(error("filter source missing"));
    };
    let Some(input) = body.compiler.facts.input(operation, 0) else {
        return Err(error("filter input missing"));
    };
    let Some(domain) = body.compiler.facts.domain(phase_owner) else {
        return Err(error("filter domain missing"));
    };
    let n = body.extent(scope, domain)?;
    let uint = Type::Constructed(TypeName::UInt(32), vec![]);
    let n = body.cast(n, &uint)?;
    let zero = Body::number(0);
    let accumulator = Accumulator::new(body, plan, source, operation, zero.clone())?;
    let one = Body::number(1);
    let width = Body::number(phase_width);
    let sum = body.binary(BinaryOperator::Add, n.clone(), Body::number(phase_width - 1))?;
    let chunks = body.binary(BinaryOperator::Divide, sum, width.clone())?;
    let lane = body.local_id()?;
    let counts = body.counted(
        zero.clone(),
        chunks,
        one.clone(),
        vec![zero.clone()],
        |body, tile, state| {
            let base = body.binary(BinaryOperator::Multiply, tile, width)?;
            let index = body.binary(BinaryOperator::Add, base, lane.clone())?;
            let valid = body.binary(BinaryOperator::Less, index.clone(), n)?;
            // Carry the evaluated input with its flag. Only valid lanes evaluate
            // the producer; only selected lanes consume the retained element.
            let retained = body.branch_with_type(
                scope,
                valid,
                |body| {
                    let value =
                        element(body, scope, plan, input, index.clone(), &mut LookupMap::default())?;
                    let predicate = body.callback(scope, operation, vec![value.clone()])?;
                    let flag = body.cast(predicate, &uint)?;
                    body.tuple(vec![flag, value])
                },
                |body, ty| {
                    let Type::Constructed(TypeName::Tuple(_), fields) = ty else {
                        return Err(error("filter input pair has no tuple representation"));
                    };
                    let unused = body.op(
                        OpTag::Intrinsic {
                            id: catalog().known().uninit,
                            overload_idx: 0,
                        },
                        vec![],
                        fields[1].clone(),
                    )?;
                    body.tuple(vec![zero.clone(), unused])
                },
                None,
            )?;
            let flag = body.field(retained.clone(), 0)?;
            let (prefixes, _, totals) = workgroup_scan(
                body,
                scope,
                std::slice::from_ref(&accumulator),
                vec![flag.clone()],
                lane.clone(),
                phase_width,
            )?;
            let prefix = prefixes[0].clone();
            let total = totals[0].clone();
            let selected = body.binary(BinaryOperator::NotEqual, flag, zero.clone())?;
            body.when(selected, |body| {
                let mut cache = LookupMap::default();
                let value = body.field(retained, 1)?;
                cache.insert(input, value.clone());
                cache.insert(source, value);
                let offset = body.binary(BinaryOperator::Subtract, prefix, one.clone())?;
                let output_index = body.binary(BinaryOperator::Add, state[0].clone(), offset)?;
                for (role, _, output_source) in results {
                    if role != "array" {
                        continue;
                    }
                    if let Some(output) = body.slot(scope, phase_owner, "output", 0, 2)? {
                        let value = element(body, scope, plan, *output_source, index.clone(), &mut cache)?;
                        body.store(output, output_index.clone(), value)?;
                    }
                }
                Ok(())
            })?;
            Ok(vec![body.binary(BinaryOperator::Add, state[0].clone(), total)?])
        },
    )?;
    let first = body.binary(BinaryOperator::Equal, lane, zero.clone())?;
    body.when(first, |body| {
        if let Some(output) = body.slot(scope, phase_owner, "length", 0, 2)? {
            body.store(output, zero, counts[0].clone())?;
        }
        Ok(())
    })
}

pub(super) fn serial_filter(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    phase_owner: Value,
    plan: Value,
    results: &[(String, i64, Value)],
) -> Result<(), OptimizeError> {
    let Some(source) = body.compiler.facts.source(phase_owner) else {
        return Err(error("missing source"));
    };
    let Some(input) = body.compiler.facts.input(phase_owner, 0) else {
        return Err(error("operation input missing"));
    };
    let Some(domain) = body.compiler.facts.domain(phase_owner) else {
        return Err(error("filter domain missing"));
    };
    let n = body.extent(scope, domain)?;
    let zero = body.literal("0", &types::i32())?;
    let one = body.literal("1", &types::i32())?;
    let count = body.counted(
        zero.clone(),
        n,
        one.clone(),
        vec![zero.clone()],
        |body, index, state| {
            let mut cache = LookupMap::default();
            let value = element(body, scope, plan, input, index.clone(), &mut cache)?;
            let keep = body.callback(scope, phase_owner, vec![value.clone()])?;
            let next = body.branch(
                scope,
                keep,
                |body| {
                    cache.insert(source, value);
                    write_arrays(
                        body,
                        scope,
                        phase_owner,
                        plan,
                        results,
                        state[0].clone(),
                        &mut cache,
                    )?;
                    body.binary(BinaryOperator::Add, state[0].clone(), one.clone())
                },
                |_| Ok(state[0].clone()),
                None,
            )?;
            Ok(vec![next])
        },
    )?;
    if let Some(output) = body.slot(scope, phase_owner, "length", 0, 2)? {
        body.store(output, zero, count[0].clone())?;
    }
    Ok(())
}
