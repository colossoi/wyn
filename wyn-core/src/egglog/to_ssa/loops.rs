use super::host;
use super::kernels::{invocation, store};
use super::plan::Stage;
use super::{error, Body, Compiler, OptimizeError};
use crate::host::{DispatchLoop, ModuleInterface, ScalarSource};
use crate::ssa::types::EntryPoint;
use crate::tlc::{LoopKind, TermKind};
use crate::LookupMap;
use egglog_engine::Value;

pub(super) fn emit(body: &mut Body<'_, '_, '_>, scope: Value, stage: &Stage) -> Result<(), OptimizeError> {
    let Some(source) = body.compiler.plan.source(stage.operation) else {
        return Err(error("loop source missing"));
    };
    let Some((header, iteration)) = body.compiler.facts.loops(source) else {
        return Err(error("loop regions missing"));
    };
    let Some(state) = body.compiler.facts.loop_state(header) else {
        return Err(error("loop state missing"));
    };
    let Some(resource) = body.compiler.plan.value_ref(state) else {
        return Err(error("loop carry storage missing"));
    };
    let output = body.resource(scope, resource, if stage.phase == "loop_enter" { 2 } else { 1 })?;
    if stage.phase == "loop_enter" {
        let Some(&(term, owner)) = body.compiler.program.identities.origins.get(&source) else {
            return Err(error("loop source term missing"));
        };
        let TermKind::Loop { init, .. } = &term.kind else {
            return Err(error("loop stage must name a loop"));
        };
        let initial = body.source(owner, init)?;
        let n = body.length(initial.clone())?;
        let (start, step) = invocation(body, stage.width)?;
        body.counted(start, n, step, vec![], |body, index, _| {
            let value = body.index(initial, index.clone())?;
            store(body, output, index, value)?;
            Ok(vec![])
        })?;
    } else {
        for value in [
            body.compiler.facts.result(iteration),
            body.compiler.facts.iteration(iteration),
        ] {
            let Some(resource) = value.and_then(|v| body.compiler.plan.value_ref(v)) else {
                return Err(error("loop backedge must have materialized storage"));
            };
            body.resource(scope, resource, 1)?;
        }
    }
    Ok(())
}

pub(super) fn publish(
    compiler: &Compiler<'_, '_>,
    entries: &[EntryPoint],
    associations: &[Vec<crate::EntryId>],
    module: &mut ModuleInterface,
) -> Result<(), OptimizeError> {
    let order = module
        .frame_graph
        .topological_order()
        .map_err(|cycle| error(format!("cyclic loop dispatch dependencies: {cycle:?}")))?;
    let positions: std::collections::BTreeMap<_, _> = order
        .iter()
        .enumerate()
        .map(|(position, &index)| {
            let pass = &module.frame_graph.passes[index];
            ((pass.pipeline_index, pass.stage_index), position)
        })
        .collect();
    for entry in entries {
        let Some((_, Some(stage))) = compiler.entry_origins.get(&entry.id) else {
            continue;
        };
        if stage.phase != "loop_exit" {
            continue;
        }
        let Some(source) = compiler.plan.source(stage.operation) else {
            return Err(error("loop source missing"));
        };
        let Some(&(term, owner)) = compiler.program.identities.origins.get(&source) else {
            return Err(error("loop term missing"));
        };
        let TermKind::Loop {
            kind: LoopKind::ForRange { bound, .. },
            init,
            ..
        } = &term.kind
        else {
            return Err(error("host loop must be counted"));
        };
        let Some(&bound) = compiler.program.identities.occurrences.get(&(owner, bound.id)) else {
            return Err(error("loop bound missing"));
        };
        let Some(count) = host::expression(compiler, bound, &LookupMap::default()) else {
            return Err(error("loop bound cannot be evaluated on the host"));
        };
        let Some((header, iteration)) = compiler.facts.loops(source) else {
            return Err(error("loop regions missing"));
        };
        let slot = |source: Option<Value>| {
            let Some(resource) =
                source.and_then(|v| compiler.plan.value_ref(v)).and_then(|v| compiler.plan.backing(v))
            else {
                return Err(error("loop state must be materialized"));
            };
            let Some(buffer) = compiler.plan.buffers.get(&resource) else {
                return Err(error("loop requires owned storage"));
            };
            Ok(ScalarSource::Binding {
                set: buffer.binding.set,
                binding: buffer.binding.binding,
            })
        };
        let current = slot(compiler.facts.loop_state(header))?;
        let next = slot(compiler.facts.result(iteration))?;
        if current == next {
            return Err(error("loop input and destination must be distinct"));
        }
        let Some(begin) = entries.iter().find(|e| {
            compiler.entry_origins.get(&e.id).is_some_and(|(_, s)| {
                s.as_ref().is_some_and(|s| s.operation == stage.operation && s.phase == "loop_enter")
            })
        }) else {
            return Err(error("loop entry missing"));
        };
        let pipeline = associations
            .iter()
            .position(|ids| ids.contains(&entry.id))
            .ok_or_else(|| error("loop pipeline missing"))?;
        let stage_index = |id| {
            associations[pipeline]
                .iter()
                .position(|&candidate| candidate == id)
                .ok_or_else(|| error("loop stage belongs to a different pipeline"))
        };
        let mut body = entries
            .iter()
            .filter(|entry| {
                let Some((_, Some(stage))) = compiler.entry_origins.get(&entry.id) else {
                    return false;
                };
                compiler
                    .plan
                    .source(stage.operation)
                    .and_then(|source| compiler.program.identities.origins.get(&source))
                    .is_some_and(|(_, scope)| *scope == iteration)
            })
            .map(|entry| stage_index(entry.id))
            .collect::<Result<Vec<_>, _>>()?;
        body.sort_by_key(|stage| positions[&(pipeline, *stage)]);
        let initial = compiler
            .program
            .identities
            .occurrences
            .get(&(owner, init.id))
            .ok_or_else(|| error("loop initializer missing"))?;
        let initial_length = host::array_length(compiler, *initial)
            .ok_or_else(|| error("loop initializer length cannot be evaluated on the host"))?;
        module.dispatch_loops.push(DispatchLoop {
            initial_length,
            pipeline,
            setup: stage_index(begin.id)?,
            completion: stage_index(entry.id)?,
            body,
            count,
            index: slot(compiler.facts.iteration(iteration))?,
            current,
            next,
        });
    }
    Ok(())
}
