//! Lower the selected map traversal, retaining nested output storage.
use super::super::{error, Body, OptimizeError};
use super::{coordinates, emit_bindings, invocation, screma, workgroup};
use crate::op::BinaryOperator;
use crate::LookupMap;
use egglog_engine::sort::S;
use egglog_engine::{RawValues, Value};

pub(super) fn emit(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    owner: Value,
    segmented: bool,
    width: u32,
    results: &[(String, i64, Value)],
) -> Result<(), OptimizeError> {
    let group = body.has_workgroup_scope("MapExecution", owner);
    if group && body.compiler.facts.contains("NativeSubgroups", RawValues(vec![])) {
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
    let (start, step) = if group { workgroup(body)? } else { invocation(body, width)? };
    body.counted(start, n, step, vec![], |body, index, _| {
        let mut cache = LookupMap::default();
        let indices = if dimensions.is_empty() {
            vec![index]
        } else {
            coordinates::decode(body, &dimensions, index)?
        };
        for (_, slot, _) in results.iter().filter(|r| r.0 == "array") {
            let output = body.slot(scope, owner, "output", *slot, 2)?;
            let tiled = body.compiler.facts.set("TiledOutputs", (owner, *slot));
            if output.is_none() && tiled.is_empty() {
                continue;
            }
            let Some(bindings) = body.compiler.facts.lookup("MapOutputBindings", (owner, *slot)) else {
                return Err(error("map output has no selected callback bindings"));
            };
            let Some(axes) = body.compiler.facts.lookup("MapOutputAxes", (owner, *slot)) else {
                return Err(error("map output has no selected coordinate mapping"));
            };
            emit_bindings(
                body,
                scope,
                bindings,
                &indices,
                &mut cache,
                &mut |body, scope, source| {
                    if segmented {
                        let Some(result) = body.compiler.facts.lookup("SegmentOutput", (owner, *slot))
                        else {
                            return Err(error("segmented output has no selected result"));
                        };
                        let Some(result) = body.compiler.facts.enode("SegmentResult", result) else {
                            return Err(error("invalid segmented result"));
                        };
                        let [_, owner, plan, role, slot] = result[..] else {
                            return Err(error("invalid segmented result"));
                        };
                        let role = body.compiler.program.graph.value_to_base::<S>(role);
                        let slot = body.compiler.facts.unsigned(slot, "segment result slot")? as usize;
                        let results = body.compiler.facts.results(plan)?;
                        let Some(output) = output.clone() else {
                            return Err(error("segment output missing"));
                        };
                        let target = screma::SegmentOutput {
                            source,
                            output,
                            indices: &indices,
                            axes,
                            role: role.as_ref(),
                            slot,
                        };
                        screma::screma(
                            body,
                            scope,
                            owner,
                            "segment",
                            width,
                            plan,
                            &results,
                            Some(&target),
                        )
                    } else {
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
                        let indices = coordinates::project(body, axes, &indices, None)?;
                        for destination in &tiled {
                            body.store_tiled(scope, *destination, indices[0].clone(), value.clone())?;
                        }
                        let Some(output) = &output else {
                            return Ok(());
                        };
                        if group {
                            let lane = body.local_id()?;
                            let first = body.binary(BinaryOperator::Equal, lane, Body::number(0))?;
                            body.when(first, |body| body.store_at(output.clone(), &indices, value))?;
                            body.workgroup_barrier()
                        } else {
                            body.store_at(output.clone(), &indices, value)
                        }
                    }
                },
            )?;
        }
        Ok(vec![])
    })?;
    Ok(())
}
