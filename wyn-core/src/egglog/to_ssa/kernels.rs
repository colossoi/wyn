//! Expand scheduled phases straight into SSA instructions and structured loops.

use super::plan::Stage;
use super::{builder_error, error, Body, OptimizeError, Typed};
use crate::builtins::catalog;
use crate::op::{BinaryOperator, OpTag};
use crate::ssa::types::InstKind;
use crate::tlc::{SoacOp, TermKind};
use crate::types::{self, Type, TypeName};
use crate::LookupMap;
use egglog_engine::Value;

mod filter;
mod indexed;
mod screma;
use filter::{compact, serial_filter};
pub(super) use indexed::bucket_updates;
use indexed::{buckets, indexed};
use screma::screma;

pub(super) fn emit(body: &mut Body<'_, '_, '_>, scope: Value, stage: &Stage) -> Result<(), OptimizeError> {
    if stage.phase == "scalar" {
        let operations = body.compiler.plan.scalar_group(stage.operation);
        if let Some(context) = body.compiler.facts.dispatch_context(stage.operation) {
            body.context = context;
        }
        body.active_operations.extend(operations.iter().copied());
        for operation in operations {
            let Some(source) = body.compiler.plan.source(operation) else {
                return Err(error("scalar dispatch source missing"));
            };
            let value = body.value(scope, source)?;
            if let Some(output) = body.slot(scope, operation, "scalar", 0, 2)? {
                let zero = body.literal("0", &types::i32())?;
                store(body, output, zero, value)?;
            }
        }
        return Ok(());
    }
    let Some(plan) = body.compiler.plan.group(stage.operation) else {
        return Err(error("kernel has no fusion plan"));
    };
    let results = body.compiler.plan.results(plan);
    match stage.phase.as_str() {
        "elements" => {
            let n = body.extent(scope, stage.extent)?;
            let (start, step) = invocation(body, stage.width)?;
            body.counted(start, n, step, vec![], |body, index, _| {
                let mut cache = LookupMap::default();
                let mut output_index = 0;
                for (role, _, source) in &results {
                    if role != "array" {
                        continue;
                    }
                    if let Some(output) = body.slot(scope, stage.operation, "output", output_index, 2)? {
                        let value = element(body, scope, plan, *source, index.clone(), &mut cache)?;
                        store(body, output, index.clone(), value)?;
                    }
                    output_index += 1;
                }
                Ok(vec![])
            })?;
            Ok(())
        }
        "scatter" | "initialize" | "atomic" => indexed(body, scope, stage, plan),
        "clear" | "buckets" => buckets(body, scope, stage, plan),
        "ordered" => {
            let Some(source) = body.compiler.plan.source(stage.operation) else {
                return Err(error("missing source"));
            };
            let Some(&(term, _)) = body.compiler.program.identities.origins.get(&source) else {
                return Err(error("ordered operation source missing"));
            };
            match term.kind {
                TermKind::Soac(SoacOp::Scatter { .. } | SoacOp::ReduceByIndex { .. }) => {
                    indexed(body, scope, stage, plan)
                }
                TermKind::Soac(SoacOp::BucketScatter { .. }) => buckets(body, scope, stage, plan),
                TermKind::Soac(SoacOp::Filter { .. }) => serial_filter(body, scope, stage, plan, &results),
                _ => screma(body, scope, stage, plan, &results),
            }
        }
        "compact" => compact(body, scope, stage, plan, &results),
        "chunks" | "combine" | "offsets" => screma(body, scope, stage, plan, &results),
        _ => Err(error(format!(
            "scheduled {} kernel is not implemented",
            stage.phase
        ))),
    }
}

fn invocation(body: &mut Body<'_, '_, '_>, width: u32) -> Result<(Typed, Typed), OptimizeError> {
    let uint = Type::Constructed(TypeName::UInt(32), vec![]);
    let start = body.op(
        OpTag::Intrinsic {
            id: catalog().known().thread_id,
            overload_idx: 0,
        },
        vec![],
        uint.clone(),
    )?;
    let groups_ty = uint.clone();
    let groups = body.op(
        OpTag::Intrinsic {
            id: catalog().known().num_workgroups,
            overload_idx: 0,
        },
        vec![],
        groups_ty,
    )?;
    let width = body.literal(&width.to_string(), &uint)?;
    if let Some((x, y, z)) = body.grid {
        let mut start = start;
        for (id, stride) in [
            (catalog().known().thread_id_y, x),
            (catalog().known().thread_id_z, x * y),
        ] {
            let axis = body.op(OpTag::Intrinsic { id, overload_idx: 0 }, vec![], uint.clone())?;
            let stride = body.literal(&stride.to_string(), &uint)?;
            let stride = body.binary(BinaryOperator::Multiply, stride, width.clone())?;
            let offset = body.binary(BinaryOperator::Multiply, axis, stride)?;
            start = body.binary(BinaryOperator::Add, start, offset)?;
        }
        let groups = body.literal(&(u64::from(x) * u64::from(y) * u64::from(z)).to_string(), &uint)?;
        return Ok((start, body.binary(BinaryOperator::Multiply, groups, width)?));
    }
    let step = body.binary(BinaryOperator::Multiply, groups, width)?;
    Ok((start, step))
}

pub(super) fn element(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    plan: Value,
    source: Value,
    index: Typed,
    cache: &mut LookupMap<Value, Typed>,
) -> Result<Typed, OptimizeError> {
    if let Some(value) = cache.get(&source) {
        return Ok(value.clone());
    }
    if let Some(actual) = body.compiler.facts.alias(source) {
        return element(body, scope, plan, actual, index, cache);
    }
    if let Some((parent, field)) = body.compiler.facts.projection(source) {
        let value = element(body, scope, plan, parent, index, cache)?;
        return body.field(value, field);
    }
    if let Some((array, start, _)) = body.compiler.facts.slice(source) {
        let start = body.value(scope, start)?;
        let start = body.cast(start, &index.ty)?;
        let index = body.binary(BinaryOperator::Add, index, start)?;
        // The base is evaluated at a different index from this stream's cache.
        let value = element(body, scope, plan, array, index, &mut LookupMap::default())?;
        cache.insert(source, value.clone());
        return Ok(value);
    }
    if let Some(operation) = body.compiler.facts.operation(source) {
        if body.compiler.plan.member(plan, operation) {
            let Some(&(term, owner)) = body.compiler.program.identities.origins.get(&source) else {
                return Err(error("fused source is missing"));
            };
            if let TermKind::Soac(SoacOp::Map { .. }) = &term.kind {
                let inputs = body.compiler.facts.inputs(operation);
                let mut arguments = Vec::new();
                for (_, input) in inputs {
                    arguments.push(element(body, scope, plan, input, index.clone(), cache)?);
                }
                let value = body.callback(owner, operation, arguments)?;
                cache.insert(source, value.clone());
                return Ok(value);
            }
        }
    }
    let array = body.value(scope, source)?;
    let value = body.index(array, index)?;
    cache.insert(source, value.clone());
    Ok(value)
}

fn store(
    body: &mut Body<'_, '_, '_>,
    array: Typed,
    index: Typed,
    value: Typed,
) -> Result<(), OptimizeError> {
    let (place, ty) = body.index_place(array, index)?;
    let value = body.stored(value, &ty)?;
    body.builder
        .push_void_inst(InstKind::Store {
            place,
            value: value.value,
        })
        .map_err(builder_error)?;
    Ok(())
}
