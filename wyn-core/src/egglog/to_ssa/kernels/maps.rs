//! Lower the selected map traversal, retaining nested output storage.
use super::super::{error, Body, OptimizeError};
use super::{coordinates, emit_bindings, invocation, screma, workgroup};
use crate::LookupMap;
use egglog_engine::sort::S;
use egglog_engine::Value;

pub(super) fn emit(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    owner: Value,
    cooperative: bool,
    width: u32,
    results: &[(String, i64, Value)],
) -> Result<(), OptimizeError> {
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
    let (start, step) = if cooperative { workgroup(body)? } else { invocation(body, width)? };
    body.counted(start, n, step, vec![], |body, index, _| {
        let mut cache = LookupMap::default();
        let indices = if dimensions.is_empty() {
            vec![index]
        } else {
            coordinates::decode(body, &dimensions, index)?
        };
        for (_, slot, _) in results.iter().filter(|r| r.0 == "array") {
            let Some(output) = body.slot(scope, owner, "output", *slot, 2)? else {
                continue;
            };
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
                    if cooperative {
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
                        let target = screma::SegmentOutput {
                            source,
                            output: output.clone(),
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
                        let value = body.value(scope, source)?;
                        let indices = coordinates::project(body, axes, &indices, None)?;
                        body.store_at(output.clone(), &indices, value)
                    }
                },
            )?;
        }
        Ok(vec![])
    })?;
    Ok(())
}
