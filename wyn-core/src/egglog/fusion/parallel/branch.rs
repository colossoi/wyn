//! Join matching guards and execute each sunk producer at most once.
use super::hoist::cheap;
use super::placement::{child_on_path, demands, movable_group, pinned_operations, read_only};
use super::{barriers, domains, fuse_region, Body, Code, Query};
use crate::egglog::graph::Graph;
use crate::egglog::{timing, OptimizeError};
use crate::memory_ir::{Definition, OperationId, RegionId, ValueRef};
use crate::{LookupSet, SortedSet, StableMap};

pub(in crate::egglog::fusion) fn run(graph: &mut Graph) -> Result<bool, OptimizeError> {
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
    for (&region, operations) in &graph.region_orders {
        let mut producer = false;
        let mut previous_guard = None;
        for &op in operations {
            if branches.contains(&op) {
                let guard = query.required("Input", (op, 0_i64))?;
                if producer || previous_guard == Some(guard) {
                    regions.push(region);
                    break;
                }
                previous_guard = Some(guard);
            } else {
                previous_guard = None;
            }
            if pinned.contains(&op) {
                producer = false;
            } else {
                producer |= query.0.control_at(op, 0_i64).is_some();
            }
        }
    }
    if regions.is_empty() {
        return Ok(false);
    }
    let mut body = Body::read(query, domains(query)?, barriers(query)?)?;
    body.pinned.extend(pinned.iter().filter_map(|key| body.graph.operation_ids.get(key).copied()));
    let mut pending: SortedSet<_> = regions.into_iter().map(|region| body.regions[&region]).collect();
    let mut changed = false;
    while let Some(region) = pending.pop_first() {
        let affected = if let Some((first, second)) = adjacent(&body, region) {
            body.join_branches(first, second)?.to_vec()
        } else if let Some(destination) = sink_one(graph, &mut body, region)? {
            vec![destination]
        } else if let Some(destination) = sink_shared_loop(graph, &mut body, region)? {
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
    if changed {
        body.publish(graph)?;
    }
    Ok(changed)
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
    graph: &mut Graph,
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

fn sink_shared_loop(
    graph: &mut Graph,
    body: &mut Body,
    region: RegionId,
) -> Result<Option<RegionId>, OptimizeError> {
    let mut selected = None;
    for &producer in &body.graph.ir.regions[region].operations {
        let query = Query(graph);
        let op = &body.graph.ir.operations[producer];
        if !matches!(
            op.definition,
            Definition::Other(Code::CountedLoop | Code::WhileLoop)
        ) || !read_only(body, producer)
        {
            continue;
        }
        let [ty] = op.result_types[..] else { continue };
        let [edge] = &op.bodies[..] else { continue };
        let [fallback] = edge.arguments[..] else { continue };
        if query.enode("ScalarInfo", ty)?.is_none() {
            continue;
        }
        let (members, uses) = demands(query, body, producer)?;
        if members.len() != 1 {
            continue;
        }
        let Some(targets) =
            uses.iter().map(|&use_| child_on_path(body, use_, region)).collect::<Option<SortedSet<_>>>()
        else {
            continue;
        };
        if targets.len() < 2 {
            continue;
        }
        let Some(before) =
            targets.iter().map(|&(scope, _)| scope).min_by_key(|&scope| body.positions[scope].1)
        else {
            continue;
        };
        let mut conditions = StableMap::new();
        let mut legal = true;
        for (scope, arm) in targets {
            if !body.is_code(scope, &Code::Branch) {
                legal = false;
                break;
            }
            let condition = body.graph.ir.operations[scope].inputs[0];
            let taken = body.control_region(scope, 0)? == arm;
            if !body.available_before(condition, before)
                || conditions.insert(condition, taken).is_some_and(|previous| previous != taken)
            {
                legal = false;
                break;
            }
        }
        if legal
            && movable_group(body, producer, before, &members).is_some()
            && !cheap(graph, body.graph.operations[producer])?
        {
            selected = Some((producer, before, conditions, fallback));
            break;
        }
    }
    let Some((producer, before, conditions, fallback)) = selected else {
        return Ok(None);
    };
    body.guard_shared_loop(
        graph,
        producer,
        before,
        conditions.into_iter().collect(),
        fallback,
    )
    .map(Some)
}
