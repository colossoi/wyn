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
    let query = Query(&body.compiler.program.graph);
    let mut domains = Vec::<(Value, Vec<i64>)>::new();
    for &(slot, _) in results {
        let domain = query.required("MapOutputDomain", (owner, slot))?;
        if let Some((_, slots)) = domains.iter_mut().find(|(shape, _)| *shape == domain) {
            slots.push(slot);
        } else {
            domains.push((domain, vec![slot]));
        }
    }
    let updates = matches!(body.compiler.facts.action(plan)?, OutputAction::Updates);
    let iteration_domain = query.required("IterationShape", (owner,))?;
    if updates && !domains.iter().any(|(shape, _)| *shape == iteration_domain) {
        domains.push((iteration_domain, vec![]));
    }
    let domains = domains
        .into_iter()
        .map(|(domain, slots)| Ok((domain, query.required("ShapeExtent", (domain,))?, slots)))
        .collect::<Result<Vec<_>, OptimizeError>>()?;
    let group = body.has_workgroup_scope("MapExecution", owner);
    if group && Query(&body.compiler.program.graph).flag("SubgroupsEnabled", RawValues(vec![]))? {
        body.prepare_subgroups(width)?;
    }
    for (domain, extent, slots) in domains {
        let n = body.extent(scope, extent)?;
        let (start, step) = if group && !matches!(phase, "ordered" | "local") {
            workgroup(body)?
        } else {
            iteration(body, phase, width)?
        };
        body.counted(start, n, step, vec![], |body, index, _| {
            let mut cache = LookupMap::default();
            let indices = coordinates::domain(body, scope, domain, index)?;
            for slot in &slots {
                let output = body.output_target(scope, owner, "output", *slot)?;
                let mappings = body.compiler.facts.set("MappedOutputs", (owner, *slot));
                if output.is_none() && mappings.is_empty() {
                    continue;
                }
                let Some(traversal) = body.compiler.facts.lookup("MapOutputTraversal", (owner, *slot))
                else {
                    return Err(error("map output has no selected callback traversal"));
                };
                emit_traversal(
                    body,
                    scope,
                    Some(plan),
                    traversal,
                    &indices,
                    &mut cache,
                    &|body, scope, leaf| {
                        if let super::TraversalLeaf::Result { owner, port, .. } = leaf {
                            let Some(output) = output.clone() else {
                                return Err(error("parallel output missing"));
                            };
                            body.with_output(owner, port, output, &indices, |body| {
                                super::emit(body, scope, owner, "local", width)
                            })?;
                        } else {
                            let previous = body.workgroup_width;
                            let previous_execution = body.invocation_execution;
                            if group {
                                body.workgroup_width = Some(width);
                                body.invocation_execution = false;
                            }
                            let value = leaf.value(body, scope);
                            body.workgroup_width = previous;
                            body.invocation_execution = previous_execution;
                            let value = value?;
                            for destination in &mappings {
                                body.store_mapped(scope, *destination, &indices, value.clone())?;
                            }
                            let Some(output) = &output else {
                                return body.tuple(vec![]);
                            };
                            if group {
                                let lane = body.local_id()?;
                                if value.ty.is_array() {
                                    let length = body.length(value.clone())?;
                                    let (array, coordinates) = output.destination(&indices);
                                    body.copy_array_strided(
                                        (array, &coordinates),
                                        value,
                                        length,
                                        &[],
                                        lane,
                                        Body::number(width),
                                    )?;
                                } else {
                                    let first =
                                        body.binary(BinaryOperator::Equal, lane, Body::number(0))?;
                                    body.when(first, |body| output.store(body, &indices, value))?;
                                }
                                body.workgroup_barrier()?;
                            } else {
                                output.store(body, &indices, value)?;
                            }
                        }
                        body.tuple(vec![])
                    },
                )?;
            }
            if updates && domain == iteration_domain {
                updates::update(body, scope, owner, phase, plan, &indices, &mut cache)?;
            }
            Ok(vec![])
        })?;
    }
    Ok(())
}
