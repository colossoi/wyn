//! Lower the selected map traversal, retaining nested output storage.
use super::super::{error, Body, OptimizeError};
use super::{coordinates, emit_traversal, iteration, updates, workgroup};
use crate::egglog::facts::OutputAction;
use crate::egglog::query::Query;
use crate::op::BinaryOperator;
use crate::types::TypeExt;
use crate::LookupMap;
use egglog_engine::{RawValues, Value};

pub(super) fn emit(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    owner: Value,
    phase: &str,
    width: u32,
    plan: Value,
    results: &[(i64, Value)],
) -> Result<(), OptimizeError> {
    let group = body.has_workgroup_scope("MapExecution", owner);
    if group && Query(&body.compiler.program.graph).flag("SubgroupsEnabled", RawValues(vec![]))? {
        body.prepare_subgroups(width)?;
    }
    let extents = body.compiler.facts.iteration_dimensions(owner)?;
    let mut dimensions = Vec::new();
    if extents.len() > 1 {
        for extent in extents {
            dimensions.push(body.extent(scope, extent)?);
        }
    }
    let Some(extent) = body.compiler.facts.lookup("IterationExtent", (owner,)) else {
        return Err(error("map launch domain missing"));
    };
    let n = body.extent(scope, extent)?;
    let (start, step) = if group && !matches!(phase, "ordered" | "local") {
        workgroup(body)?
    } else {
        iteration(body, phase, width)?
    };
    body.counted(start, n, step, vec![], |body, index, _| {
        let mut cache = LookupMap::default();
        let indices = if dimensions.is_empty() {
            vec![index]
        } else {
            coordinates::decode(body, &dimensions, index)?
        };
        for (slot, _) in results {
            let output = body.slot(scope, owner, "output", *slot, 2)?;
            let mappings = body.compiler.facts.set("MappedOutputs", (owner, *slot));
            if output.is_none() && mappings.is_empty() {
                continue;
            }
            let Some(traversal) = body.compiler.facts.lookup("MapOutputTraversal", (owner, *slot)) else {
                return Err(error("map output has no selected callback traversal"));
            };
            emit_traversal(
                body,
                scope,
                traversal,
                &indices,
                &mut cache,
                &mut |body, scope, leaf| {
                    if let Some(fields) = body.compiler.facts.enode("TraversalResult", leaf) {
                        let Some(output) = output.clone() else {
                            return Err(error("parallel output missing"));
                        };
                        body.with_output(fields[1], fields[3], output, &indices, |body| {
                            super::emit(body, scope, fields[1], "local", width)
                        })
                    } else {
                        let Some(fields) = body.compiler.facts.enode("TraversalValue", leaf) else {
                            return Err(error("invalid traversal leaf"));
                        };
                        let source = fields[0];
                        let previous = body.workgroup_width;
                        let previous_execution = body.invocation_execution;
                        if group {
                            body.workgroup_width = Some(width);
                            body.invocation_execution = false;
                        }
                        let value = body.value(scope, source);
                        body.workgroup_width = previous;
                        body.invocation_execution = previous_execution;
                        let value = value?;
                        for destination in &mappings {
                            body.store_mapped(scope, *destination, indices[0].clone(), value.clone())?;
                        }
                        let Some(output) = &output else {
                            return Ok(());
                        };
                        if group {
                            let lane = body.local_id()?;
                            if value.ty.is_array() {
                                let length = body.length(value.clone())?;
                                body.copy_array_strided(
                                    (output.clone(), &indices),
                                    value,
                                    length,
                                    &[],
                                    lane,
                                    Body::number(width),
                                )?;
                            } else {
                                let first = body.binary(BinaryOperator::Equal, lane, Body::number(0))?;
                                body.when(first, |body| body.store_at(output.clone(), &indices, value))?;
                            }
                            body.workgroup_barrier()
                        } else {
                            body.store_at(output.clone(), &indices, value)
                        }
                    }
                },
            )?;
        }
        if matches!(body.compiler.facts.action(plan)?, OutputAction::Updates) {
            updates::update(body, scope, owner, phase, plan, &indices, &mut cache)?;
        }
        Ok(vec![])
    })?;
    Ok(())
}
