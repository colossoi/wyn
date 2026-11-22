//! Lift total invariant operations through their current region owners.
use super::{domains, effects_conflict, Body, Code, Effects, Query, Site};
use crate::egglog::{timing, OptimizeError};
use crate::memory_ir::{Definition, OperationId, UseSite, ValueRef};
use crate::{LookupMap, LookupSet, SortedSet};
use egglog_engine::sort::Z;
use egglog_engine::{EGraph, RawValues, Value, Write};
use std::collections::VecDeque;

pub(in crate::egglog::fusion) fn hoist(graph: &mut EGraph) -> Result<bool, OptimizeError> {
    let _timing = timing::span("Rust cross-region hoisting");
    let query = Query(graph);
    let mut parents = LookupMap::new();
    query.for_each("RegionOwnedBy", |row| {
        parents.insert(row[0], row[1]);
        Ok(())
    })?;
    if parents.is_empty() {
        return Ok(false);
    }
    let mut movable = LookupSet::new();
    let mut initial = LookupMap::new();
    query.for_function("MovableOperation", |key, flag| {
        let op = key[0];
        if query.0.value_to_base::<bool>(flag) && !query.contains("InvocationDependent", op)? {
            movable.insert(op);
            let source = query.required("OperationRegion", op)?;
            if let Some(&owner) = parents.get(&source) {
                if !query.contains("AccumulatorOperation", owner)? {
                    initial.insert(op, source);
                }
            }
        }
        Ok(())
    })?;
    if initial.is_empty() {
        return Ok(false);
    }
    // Skip the import when every possible mover still needs a local definition.
    query.for_each("OperationUse", |row| {
        if let Some(&source) = initial.get(&row[0]) {
            if query.contains("DefinitionRegion", (row[1], source))? {
                initial.remove(&row[0]);
            }
        }
        Ok(())
    })?;
    if initial.is_empty() {
        return Ok(false);
    }

    let mut repeated = LookupSet::new();
    let roles = ["LoopRegion", "LaneRegion"]
        .into_iter()
        .map(|role| query.constructor(role, RawValues(vec![])))
        .collect::<Result<Vec<_>, _>>()?;
    query.for_function("ControlRole", |key, role| {
        if roles.contains(&Some(role)) {
            repeated.insert(query.required("ControlRegion", (key[0], key[1]))?);
        }
        Ok(())
    })?;

    let mut body = Body::read(query, &[], domains(query)?, LookupSet::new())?;
    let mut queued: LookupSet<_> = initial.values().map(|region| body.regions[region]).collect();
    let ordered: SortedSet<_> = queued.iter().copied().collect();
    let mut pending: VecDeque<_> = ordered.into_iter().collect();
    let mut changed = false;
    let no_effects = Effects::default();
    while let Some(region) = pending.pop_front() {
        queued.remove(&region);
        let Some(owner) = body.parent(region) else {
            continue;
        };
        let target = body.graph.operations[owner];
        if Query(graph).contains("AccumulatorOperation", target)? {
            continue;
        }
        let branch = Query(graph).contains("BranchOperation", target)?;
        let repeated = repeated.contains(&body.graph.regions[region]);
        let mut index = 0;
        while let Some(&op) = body.graph.ir.regions[region].operations.get(index) {
            let key = body.graph.operations[op];
            if !movable.contains(&key)
                || !body.captures_available_before(op, owner)
                || (repeated && !arrays_stay_read_only(Query(graph), &body, op)?)
                || (branch && !cheap(graph, key)?)
                || effects_conflict(
                    graph,
                    body.effects.get(op).unwrap_or(&no_effects),
                    body.effects.get(owner).unwrap_or(&no_effects),
                )?
            {
                index += 1;
                continue;
            }
            body.lift(op, owner);
            changed = true;
        }
        for dirty in body.take_dirty() {
            // The source was visited in occurrence order, so its remaining operations
            // have already seen every change to their earlier definitions.
            if dirty != region && body.parent(dirty).is_some() && queued.insert(dirty) {
                pending.push_back(dirty);
            }
        }
    }
    if changed {
        body.publish(graph)?;
    }
    Ok(changed)
}

fn cheap(graph: &mut EGraph, operation: Value) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    if let Some(work) = query.lookup("InstructionWork", operation)? {
        return Ok(graph.value_to_base::<i64>(work) <= 4);
    }
    // Compound work is the estimate for the input to this pass. Seed only
    // candidates that have passed scope and repeated-storage checks.
    if !query.contains("OperationWorkDemand", operation)? {
        graph.update(|mut sink| sink.add("OperationWorkDemand", operation))?;
        graph.parse_and_run_program(None, "(run-schedule (saturate graph-work))")?;
    }
    Ok(Query(graph)
        .lookup("OperationWork", operation)?
        .is_some_and(|work| u64::try_from(&*graph.value_to_base::<Z>(work)).is_ok_and(|work| work <= 4)))
}

fn arrays_stay_read_only(query: Query<'_>, body: &Body, op: OperationId) -> Result<bool, OptimizeError> {
    let key = body.graph.operations[op];
    for slot in 0..body.graph.ir.operations[op].result_types.len() {
        let name = body.graph.names[&ValueRef::Result(op, slot)];
        let Some(value) = query.value_name(name)? else {
            continue;
        };
        if !query.contains("ArrayValue", value)? {
            continue;
        }
        if query.contains("ScratchAllocation", key)? {
            return Ok(false);
        }
        for site in body.users(name) {
            let Site::Operand(UseSite::Input(user, 0)) = site else {
                return Ok(false);
            };
            let user = &body.graph.ir.operations[user];
            if matches!(user.definition, Definition::Other(Code::Length { .. })) {
                continue;
            }
            if !matches!(user.definition, Definition::Other(Code::Index)) {
                return Ok(false);
            }
            if query.enode("ScalarInfo", user.result_types[0])?.is_none() {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
