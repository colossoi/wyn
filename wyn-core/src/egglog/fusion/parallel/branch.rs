//! Join matching guards and move producers dominated by a single arm.
use super::hoist::cheap;
use super::placement::{child_on_path, demands, movable_group, pinned_operations, read_only};
use super::{barriers, domains, fuse_region, Body, Code, Query};
use crate::egglog::dataflow::edit;
use crate::egglog::{output_error, timing, OptimizeError};
use crate::memory_ir::{OperationId, RegionId, ValueRef};
use crate::{LookupMap, LookupSet, SortedSet, StableMap};
use egglog_engine::{EGraph, Value, Write};

pub(in crate::egglog::fusion) fn run(graph: &mut EGraph) -> Result<bool, OptimizeError> {
    let _timing = timing::span("Rust branch fusion and sinking");
    let query = Query(graph);
    let mut branches = LookupSet::new();
    query.for_each("BranchOperation", |row| {
        branches.insert(row[0]);
        Ok(())
    })?;
    if branches.is_empty() {
        return Ok(false);
    }
    let pinned = pinned_operations(query)?;
    let mut regions = Vec::new();
    query.for_function("RegionOperations", |key, order| {
        let mut producer = false;
        let mut previous_guard = None;
        for op in query.vector(order)? {
            if branches.contains(&op) {
                let guard = query.required("Input", (op, 0_i64))?;
                if producer || previous_guard == Some(guard) {
                    regions.push(key[0]);
                    break;
                }
                previous_guard = Some(guard);
            } else {
                previous_guard = None;
            }
            if pinned.contains(&op) {
                producer = false;
            } else {
                producer |= query.lookup("ControlRegion", (op, 0_i64))?.is_some();
            }
        }
        Ok(())
    })?;
    if regions.is_empty() {
        return Ok(false);
    }
    let mut body = Body::read(query, domains(query)?, barriers(query)?)?;
    body.pinned.extend(pinned.iter().filter_map(|key| body.graph.operation_ids.get(key).copied()));
    let mut pending: SortedSet<_> = regions.into_iter().map(|region| body.regions[&region]).collect();
    let mut visited = SortedSet::new();
    let mut changed = false;
    while let Some(region) = pending.pop_first() {
        visited.insert(region);
        let affected = if let Some((first, second)) = adjacent(&body, region) {
            body.join_branches(first, second)?.to_vec()
        } else if let Some(destination) = sink_one(graph, &mut body, region)? {
            vec![destination]
        } else {
            continue;
        };
        changed = true;
        // A joined or prepended arm can expose an ordinary fusion without changing its domain.
        for affected in affected.into_iter().chain([region]) {
            fuse_region(graph, &mut body, affected)?;
        }
        pending.extend(body.take_dirty());
    }
    let copies = plan_copies(graph, &body, visited)?;
    if changed {
        body.publish(graph)?;
    }
    if copies.is_empty() {
        return Ok(changed);
    }
    publish_copies(graph, copies)?;
    Ok(true)
}

fn adjacent(body: &Body, region: RegionId) -> Option<(OperationId, OperationId)> {
    for pair in body.graph.ir.regions[region].operations.windows(2) {
        let (first, second) = (pair[0], pair[1]);
        if !body.is_code(first, &Code::Branch) || !body.is_code(second, &Code::Branch) {
            continue;
        }
        let a = &body.graph.ir.operations[first];
        let b = &body.graph.ir.operations[second];
        if a.inputs != b.inputs
            || [a, b].into_iter().any(|operation| operation.bodies.iter().any(|edge| {
                !edge.arguments.is_empty() || !body.graph.ir.regions[edge.region].parameters.is_empty()
            }))
            || body.tree_uses(second).iter().any(|&(_, name)| {
                matches!(body.graph.refs.get(&name), Some(ValueRef::Result(op, _)) if *op == first)
            })
        {
            continue;
        }
        return Some((first, second));
    }
    None
}

fn sink_one(
    graph: &mut EGraph,
    body: &mut Body,
    region: RegionId,
) -> Result<Option<RegionId>, OptimizeError> {
    let mut selected = None;
    for &producer in &body.graph.ir.regions[region].operations {
        if body.graph.ir.operations[producer].bodies.is_empty() || !read_only(body, producer) {
            continue;
        }
        let (members, uses) = demands(Query(graph), body, producer)?;
        let Some((scope, destination)) = uses.iter().find_map(|&use_| child_on_path(body, use_, region))
        else {
            continue;
        };
        // Shared producers still execute once when several arms can demand the result.
        if !body.is_code(scope, &Code::Branch) || uses.iter().any(|&use_| !body.inside(use_, destination)) {
            continue;
        }
        let Some(moved) = movable_group(body, producer, scope, &members) else {
            continue;
        };
        if !cheap(graph, body.graph.operations[producer])? {
            selected = Some((scope, destination, moved));
            break;
        }
    }
    let Some((scope, destination, moved)) = selected else {
        return Ok(None);
    };
    body.sink_producers(scope, destination, &moved);
    Ok(Some(destination))
}

struct ArmCopy {
    scope: Value,
    arm: Value,
    members: Vec<Value>,
    operations: Vec<Value>,
    depth: usize,
}

fn plan_copies(
    graph: &mut EGraph,
    body: &Body,
    regions: SortedSet<RegionId>,
) -> Result<Vec<ArmCopy>, OptimizeError> {
    let mut arms = StableMap::<RegionId, Vec<OperationId>>::new();
    for region in regions {
        for &producer in &body.graph.ir.regions[region].operations {
            if body.graph.ir.operations[producer].bodies.is_empty() || !read_only(body, producer) {
                continue;
            }
            let (members, uses) = demands(Query(graph), body, producer)?;
            let Some(targets) = uses
                .iter()
                .map(|&use_| child_on_path(body, use_, region))
                .collect::<Option<SortedSet<_>>>()
            else {
                continue;
            };
            if targets.len() < 2 {
                continue;
            }
            let mut guards = LookupSet::new();
            if !targets.iter().all(|&(scope, arm)| {
                body.is_code(scope, &Code::Branch)
                    && body.graph.ir.regions[arm].parameters.is_empty()
                    && guards.insert(body.graph.ir.operations[scope].inputs[0])
            }) {
                continue;
            }
            let Some(moves) = targets
                .into_iter()
                .map(|(scope, arm)| {
                    movable_group(body, producer, scope, &members).map(|members| (arm, members))
                })
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            if cheap(graph, body.graph.operations[producer])? {
                continue;
            }
            for (arm, members) in moves {
                arms.entry(arm).or_default().extend(members);
            }
        }
    }
    let mut copies = Vec::new();
    for (arm, mut members) in arms {
        let Some(scope) = body.parent(arm) else {
            return Err(output_error("copy destination has no branch owner"));
        };
        members.sort_unstable_by_key(|&op| body.positions[op].1);
        members.dedup();
        let mut depth = 0;
        let mut region = arm;
        while let Some(owner) = body.parent(region) {
            depth += 1;
            region = body.positions[owner].0;
        }
        copies.push(ArmCopy {
            scope: body.graph.operations[scope],
            arm: body.graph.regions[arm],
            members: members.into_iter().map(|op| body.graph.operations[op]).collect(),
            operations: body.graph.ir.regions[arm]
                .operations
                .iter()
                .map(|&op| body.graph.operations[op])
                .collect(),
            depth,
        });
    }
    // Finish inner rewrites before a containing arm is copied.
    copies.sort_by_key(|copy| std::cmp::Reverse(copy.depth));
    Ok(copies)
}

fn publish_copies(graph: &mut EGraph, copies: Vec<ArmCopy>) -> Result<(), OptimizeError> {
    let _timing = timing::span("branch producer copy publication");
    let mut remaining = LookupMap::<_, usize>::new();
    for copy in &copies {
        for &member in &copy.members {
            *remaining.entry(member).or_default() += 1;
        }
    }
    for copy in copies {
        let query = Query(graph);
        let mut operations = Vec::new();
        for op in copy.operations {
            if query.lookup("OperationRegion", op)? == Some(copy.arm) {
                operations.push(op);
            }
        }
        let edge = i64::from(query.required("ControlRegion", (copy.scope, 1_i64))? == copy.arm);
        let cloned = edit::clone_region_with_prefix(graph, copy.scope, copy.arm, &[], &copy.members)?;
        let query = Query(graph);
        let cloned_operations = copy
            .members
            .iter()
            .copied()
            .chain(operations)
            .map(|operation| {
                let Some(key) = query.constructor("ClonedOperation", (copy.scope, operation))? else {
                    return Err(output_error("branch producer copy lost a scheduled operation"));
                };
                Ok(key)
            })
            .collect::<Result<Vec<_>, OptimizeError>>()?;
        let retired = copy
            .members
            .into_iter()
            .filter(|op| {
                let count = remaining.entry(*op).or_default();
                *count -= 1;
                *count == 0
            })
            .collect::<Vec<_>>();
        graph.update(|mut sink| {
            for (order, operation) in cloned_operations.into_iter().enumerate() {
                sink.remove("OperationOrder", operation)?;
                sink.set("OperationOrder", operation, order as i64)?;
            }
            sink.remove("ControlRegion", (copy.scope, edge))?;
            sink.set("ControlRegion", (copy.scope, edge), cloned)?;
            for op in retired {
                sink.remove("OperationRegion", op)?;
            }
            Ok(())
        })?;
    }
    edit::remove(graph, &remaining.into_keys().collect::<Vec<_>>())
}
