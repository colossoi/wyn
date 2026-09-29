use super::super::interface;
use super::screma::write_arrays;
// Expand scheduled phases straight into SSA instructions and structured loops.
use super::super::plan::Stage;
use super::super::{builder_error, error, Body, OptimizeError};
use super::{element, store};
use crate::builtins::catalog;
use crate::op::{BinaryOperator, OpTag, PureViewSource};
use crate::ssa::types::InstKind;
use crate::tlc::{SoacOp, TermKind};
use crate::types::{self, Type, TypeName};
use crate::LookupMap;
use egglog_engine::Value;

pub(super) fn compact(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    stage: &Stage,
    plan: Value,
    results: &[(String, i64, Value)],
) -> Result<(), OptimizeError> {
    let members = body.compiler.plan.members(plan);
    let Some((operation, source)) = members.iter().find_map(|&operation| {
        let source = body.compiler.plan.source(operation)?;
        let (term, _) = body.compiler.program.identities.origins.get(&source)?;
        matches!(&term.kind, TermKind::Soac(SoacOp::Filter { .. })).then_some((operation, source))
    }) else {
        return Err(error("filter callback missing"));
    };
    let Some(input) = body.compiler.facts.input(operation, 0) else {
        return Err(error("filter input missing"));
    };
    let Some(domain) = body.compiler.plan.domain(stage.operation) else {
        return Err(error("filter domain missing"));
    };
    let n = body.extent(scope, domain)?;
    let uint = Type::Constructed(TypeName::UInt(32), vec![]);
    let n = body.cast(n, &uint)?;
    let zero = body.literal("0", &uint)?;
    let one = body.literal("1", &uint)?;
    let width = body.literal(&stage.width.to_string(), &uint)?;
    let last = body.literal(&(stage.width - 1).to_string(), &uint)?;
    let sum = body.binary(BinaryOperator::Add, n.clone(), last.clone())?;
    let chunks = body.binary(BinaryOperator::Divide, sum, width.clone())?;
    let lane = body.op(
        OpTag::Intrinsic {
            id: catalog().known().local_id,
            overload_idx: 0,
        },
        vec![],
        uint.clone(),
    )?;
    let mut shared = Vec::new();
    for id in 0..2 {
        shared.push(body.op(
            OpTag::StorageView(PureViewSource::Workgroup {
                id,
                count: stage.width,
            }),
            vec![zero.clone(), width.clone()],
            interface::view_type(&uint, types::no_buffer()),
        )?);
    }
    let counts = body.counted(
        zero.clone(),
        chunks,
        one.clone(),
        vec![zero.clone()],
        |body, tile, state| {
            let base = body.binary(BinaryOperator::Multiply, tile, width)?;
            let index = body.binary(BinaryOperator::Add, base, lane.clone())?;
            let valid = body.binary(BinaryOperator::Less, index.clone(), n)?;
            let flag = body.branch(
                scope,
                valid,
                |body| {
                    let value =
                        element(body, scope, plan, input, index.clone(), &mut LookupMap::default())?;
                    let predicate = body.callback(scope, operation, vec![value])?;
                    body.cast(predicate, &uint)
                },
                |_| Ok(zero.clone()),
                None,
            )?;
            let mut prefix = flag.clone();
            store(body, shared[0].clone(), lane.clone(), prefix.clone())?;
            body.builder.push_void_inst(InstKind::ControlBarrier).map_err(builder_error)?;
            let mut bank = 0;
            for bit in 0..stage.width.trailing_zeros() {
                let distance = body.literal(&(1u32 << bit).to_string(), &uint)?;
                let valid = body.binary(BinaryOperator::GreaterEqual, lane.clone(), distance.clone())?;
                prefix = body.branch(
                    scope,
                    valid,
                    |body| {
                        let index = body.binary(BinaryOperator::Subtract, lane.clone(), distance)?;
                        let peer = body.index(shared[bank].clone(), index)?;
                        body.binary(BinaryOperator::Add, prefix.clone(), peer)
                    },
                    |_| Ok(prefix.clone()),
                    None,
                )?;
                bank = 1 - bank;
                store(body, shared[bank].clone(), lane.clone(), prefix.clone())?;
                body.builder.push_void_inst(InstKind::ControlBarrier).map_err(builder_error)?;
            }
            let total = body.index(shared[bank].clone(), last)?;
            body.builder.push_void_inst(InstKind::ControlBarrier).map_err(builder_error)?;
            let selected = body.binary(BinaryOperator::NotEqual, flag, zero.clone())?;
            body.when(selected, |body| {
                let mut cache = LookupMap::default();
                let value = element(body, scope, plan, input, index.clone(), &mut cache)?;
                cache.insert(source, value);
                let offset = body.binary(BinaryOperator::Subtract, prefix, one.clone())?;
                let output_index = body.binary(BinaryOperator::Add, state[0].clone(), offset)?;
                for (role, _, output_source) in results {
                    if role != "array" {
                        continue;
                    }
                    if let Some(output) = body.slot(scope, stage.operation, "output", 0, 2)? {
                        let value = element(body, scope, plan, *output_source, index.clone(), &mut cache)?;
                        store(body, output, output_index.clone(), value)?;
                    }
                }
                Ok(())
            })?;
            Ok(vec![body.binary(BinaryOperator::Add, state[0].clone(), total)?])
        },
    )?;
    let first = body.binary(BinaryOperator::Equal, lane, zero.clone())?;
    body.when(first, |body| {
        if let Some(output) = body.slot(scope, stage.operation, "length", 0, 2)? {
            store(body, output, zero, counts[0].clone())?;
        }
        Ok(())
    })
}

pub(super) fn serial_filter(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    stage: &Stage,
    plan: Value,
    results: &[(String, i64, Value)],
) -> Result<(), OptimizeError> {
    let Some(source) = body.compiler.plan.source(stage.operation) else {
        return Err(error("missing source"));
    };
    let Some(input) = body.compiler.facts.input(stage.operation, 0) else {
        return Err(error("operation input missing"));
    };
    let Some(domain) = body.compiler.plan.domain(stage.operation) else {
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
            let keep = body.callback(scope, stage.operation, vec![value.clone()])?;
            let next = body.branch(
                scope,
                keep,
                |body| {
                    cache.insert(source, value);
                    write_arrays(
                        body,
                        scope,
                        stage.operation,
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
    if let Some(output) = body.slot(scope, stage.operation, "length", 0, 2)? {
        store(body, output, zero, count[0].clone())?;
    }
    Ok(())
}
