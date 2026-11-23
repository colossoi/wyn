//! Resolve direct calls through attached control, reducer and combiner regions.
use crate::egglog::dataflow::ir::Code;
use crate::egglog::graph::Graph;
use crate::egglog::query::Query;
use crate::egglog::OptimizeError;
use crate::memory_ir::call_graph::{CallGraph, CallSite};
use crate::{LookupMap, LookupSet, StableMap};
use egglog_engine::{RawValues, Value, Write};
use wyn_graph::{for_each_reachable, WalkOrder};

#[derive(Default)]
struct RegionCalls {
    children: Vec<Value>,
    sites: Vec<CallSite<Value, Value>>,
    operations: Vec<Value>,
}

pub(super) fn publish(graph: &mut Graph, for_inlining: bool) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut regions = LookupMap::<Value, RegionCalls>::new();
    let mut owners = LookupMap::new();
    let mut multiple_owners = false;
    let mut loop_carries = Vec::new();
    let positions = graph.structure().read();
    let mut record = |operation, child| {
        multiple_owners |= owners.insert(child, operation).is_some_and(|old| old != operation);
        if graph.has_operation(operation) {
            if let Some(parent) = positions.operations.get(&operation).and_then(|position| position.region)
            {
                regions.entry(parent).or_default().children.push(child);
            }
        }
    };
    for (&(operation, slot), edge) in &positions.controls {
        record(operation, edge.region);
        if slot != 0 {
            continue;
        }
        for (slot, &initial) in edge.arguments.iter().enumerate() {
            let slot = slot as i64;
            let Some(parameter) = query.0.parameter_at(edge.region, slot) else {
                continue;
            };
            let Some(next) = query.0.yield_at(edge.region, slot) else {
                continue;
            };
            if let (Some(parameter), Some(initial), Some(next)) = (
                query.value_name(parameter)?,
                query.value_name(initial)?,
                query.value_name(next)?,
            ) {
                loop_carries.push((operation, slot, parameter, initial, next));
            }
        }
    }
    for callbacks in [&positions.reducers, &positions.combiners] {
        for (&operation, bodies) in callbacks {
            for &(_, child) in bodies {
                record(operation, child);
            }
        }
    }
    for (&operation, &callee) in &positions.targets {
        if let Some(region) = positions.operations.get(&operation).and_then(|position| position.region) {
            regions.entry(region).or_default().sites.push(CallSite { operation, callee });
        }
    }
    for (&region, operations) in &positions.by_region {
        let body = regions.entry(region).or_default();
        body.operations
            .extend(operations.iter().copied().filter(|&operation| graph.has_operation(operation)));
    }
    let mut reached = LookupMap::<Value, Vec<Value>>::new();
    let mut calls = StableMap::new();
    let mut parallel_regions = LookupSet::new();
    if for_inlining {
        parallel_regions.extend(
            graph
                .codes()
                .filter(|(_, code)| matches!(code, Code::Parallel))
                .filter_map(|(op, _)| positions.operations.get(&op)?.region),
        );
    }
    let mut parallel_functions = Vec::new();
    let mut memory_signatures = Vec::new();
    for (&function, &root) in &positions.roots {
        if for_inlining {
            for &parameter in positions.parameters.get(&root).into_iter().flatten() {
                let requires_inline = match query.value_name(parameter)? {
                    Some(name) => match query.lookup("ValueType", name)? {
                        Some(info) => query.enode("LogicalValue", info)?.is_some(),
                        None => false,
                    },
                    None => query.enode("AccNameRef", parameter)?.is_some(),
                };
                if requires_inline {
                    memory_signatures.push(function);
                    break;
                }
            }
        }
        let bodies = reached.entry(function).or_default();
        let sites = calls.entry(function).or_insert_with(Vec::new);
        for_each_reachable(
            [root],
            WalkOrder::DepthFirst,
            |region, children| {
                if let Some(body) = regions.get(&region) {
                    children.extend_from_slice(&body.children);
                }
            },
            |region| {
                bodies.push(region);
                if let Some(body) = regions.get(&region) {
                    sites.extend_from_slice(&body.sites);
                }
            },
        );
        if bodies.iter().any(|region| parallel_regions.contains(region)) {
            parallel_functions.push(function);
        }
    }
    let function_roots: Vec<_> =
        positions.roots.iter().map(|(&function, &root)| (function, root)).collect();
    let functions: Vec<_> = positions.roots.keys().copied().collect();
    let mut roots: Vec<_> = positions.entries.keys().copied().collect();
    drop(positions);
    let call_graph = CallGraph { calls };
    if query.flag("RetainAllFunctions", RawValues(Vec::new()))? {
        roots.extend_from_slice(&functions);
    }
    let (live, recursive) = call_graph.analyze(roots);
    graph.update(|mut sink| {
        for carry in loop_carries {
            sink.add("LoopValueCarry", carry)?;
        }
        for function in parallel_functions {
            sink.add("InlineParallel", function)?;
        }
        for function in memory_signatures {
            sink.set("InlineRequired", function, true)?;
        }
        for function in functions {
            sink.set("RecursiveFunction", function, recursive.contains(&function))?;
        }
        for &(function, region) in &function_roots {
            sink.add("RootFunction", (region, function))?;
        }
        if multiple_owners {
            sink.add("InvalidDataflow", "executable region has more than one owner")?;
        }
        for (child, operation) in owners {
            sink.set("RegionOwnedBy", child, operation)?;
        }
        for caller in live {
            sink.add("LiveFunction", caller)?;
            for &region in reached.get(&caller).into_iter().flatten() {
                sink.add("LiveRegion", region)?;
                if let Some(body) = regions.get(&region) {
                    for &operation in &body.operations {
                        sink.add("LiveOperation", operation)?;
                    }
                }
            }
            if let Some(sites) = call_graph.calls.get(&caller) {
                for site in sites {
                    sink.add("InlineCall", (caller, site.operation, site.callee))?;
                }
            }
        }
        Ok(())
    })?;
    Ok(())
}
