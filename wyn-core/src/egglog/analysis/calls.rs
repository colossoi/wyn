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

pub(in crate::egglog) struct Reachability {
    pub calls: CallGraph<Value, Value>,
    live: Vec<Value>,
    recursive: LookupSet<Value>,
    bodies: LookupMap<Value, Vec<Value>>,
    regions: LookupMap<Value, RegionCalls>,
    owners: LookupMap<Value, Value>,
    multiple_owners: bool,
}

impl Reachability {
    pub fn read(graph: &Graph) -> Result<Self, OptimizeError> {
        let query = Query(graph);
        let mut regions = LookupMap::<Value, RegionCalls>::new();
        let mut owners = LookupMap::new();
        let mut multiple_owners = false;
        let positions = graph.structure().read();
        let mut record = |operation, child| {
            multiple_owners |= owners.insert(child, operation).is_some_and(|old| old != operation);
            if graph.has_operation(operation) {
                if let Some(parent) =
                    positions.operations.get(&operation).and_then(|position| position.region)
                {
                    regions.entry(parent).or_default().children.push(child);
                }
            }
        };
        for (&(operation, _), edge) in &positions.controls {
            record(operation, edge.region);
        }
        for callbacks in [&positions.reducers, &positions.combiners] {
            for (&operation, bodies) in callbacks {
                for &(_, child) in bodies {
                    record(operation, child);
                }
            }
        }
        for (&operation, &callee) in &positions.targets {
            if let Some(region) = positions.operations.get(&operation).and_then(|position| position.region)
            {
                regions.entry(region).or_default().sites.push(CallSite { operation, callee });
            }
        }
        for (&region, operations) in &positions.by_region {
            regions
                .entry(region)
                .or_default()
                .operations
                .extend(operations.iter().copied().filter(|&operation| graph.has_operation(operation)));
        }
        let mut bodies = LookupMap::<Value, Vec<Value>>::new();
        let mut calls = StableMap::new();
        for (&function, &root) in &positions.roots {
            let sites = calls.entry(function).or_insert_with(Vec::new);
            let reached = bodies.entry(function).or_default();
            for_each_reachable(
                [root],
                WalkOrder::DepthFirst,
                |region, children| {
                    if let Some(body) = regions.get(&region) {
                        children.extend_from_slice(&body.children);
                    }
                },
                |region| {
                    reached.push(region);
                    if let Some(body) = regions.get(&region) {
                        sites.extend_from_slice(&body.sites);
                    }
                },
            );
        }
        let calls = CallGraph { calls };
        let mut roots: Vec<_> = positions.entries.keys().copied().collect();
        if query.flag("RetainAllFunctions", RawValues(Vec::new()))? {
            roots.extend(positions.roots.keys().copied());
        }
        let (live, recursive) = calls.analyze(roots);
        Ok(Self {
            calls,
            live,
            recursive,
            bodies,
            regions,
            owners,
            multiple_owners,
        })
    }

    /// Select calls in callee-first order so a clone contains all selected callee edits.
    pub fn inlines(&self, graph: &Graph, all: bool) -> Result<Vec<(Value, Value)>, OptimizeError> {
        let query = Query(graph);
        let mut parallel = LookupSet::new();
        let mut required = LookupMap::new();
        let mut selected = Vec::new();
        for &caller in &self.live {
            let Some(root) = graph.root_of(caller) else {
                continue;
            };
            let Some(sites) = self.calls.calls.get(&caller) else {
                continue;
            };
            let mut exposes_parallel = false;
            for site in sites {
                if let Some(&callee) = required.get(&site.callee) {
                    selected.push((site.operation, callee));
                }
                exposes_parallel |= parallel.contains(&site.callee);
            }
            if self.recursive.contains(&caller) {
                continue;
            }
            for &region in self.bodies.get(&caller).into_iter().flatten() {
                if all || exposes_parallel {
                    break;
                }
                for &operation in self.regions.get(&region).into_iter().flat_map(|body| &body.operations) {
                    if graph.code(operation)? == Code::Parallel {
                        exposes_parallel = true;
                        break;
                    }
                }
            }
            if exposes_parallel {
                parallel.insert(caller);
            }
            if all || exposes_parallel || has_memory_signature(query, caller, root)? {
                required.insert(caller, root);
            }
        }
        Ok(selected)
    }
}

fn has_memory_signature(query: Query<'_>, function: Value, root: Value) -> Result<bool, OptimizeError> {
    for parameter in query.0.parameters(root) {
        if let Some(name) = query.value_name(parameter)? {
            if let Some(info) = query.lookup("ValueType", name)? {
                if query.enode("LogicalValue", info)?.is_some() {
                    return Ok(true);
                }
            }
        } else if query.enode("AccNameRef", parameter)?.is_some() {
            return Ok(true);
        }
    }
    let Some(count) = query.lookup("FunctionResultCount", function)? else {
        return Ok(false);
    };
    for slot in 0..query.0.value_to_base::<i64>(count) {
        let result = query.required("FunctionResultType", (function, slot))?;
        if query.enode("LogicalReturn", result)?.is_some()
            || query.enode("ReturnAccumulator", result)?.is_some()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(super) fn publish(graph: &mut Graph) -> Result<(), OptimizeError> {
    let reach = Reachability::read(graph)?;
    let query = Query(graph);
    let mut loop_carries = Vec::new();
    let positions = graph.structure().read();
    for (&(operation, edge_index), edge) in &positions.controls {
        if edge_index != 0 {
            continue;
        }
        for (slot, &initial) in edge.arguments.iter().enumerate() {
            let slot = slot as i64;
            let Some(parameter) = graph.parameter_at(edge.region, slot) else {
                continue;
            };
            let Some(next) = graph.yield_at(edge.region, slot) else {
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
    let roots: Vec<_> = positions.roots.iter().map(|(&function, &root)| (function, root)).collect();
    drop(positions);
    graph.update(|mut sink| {
        for carry in loop_carries {
            sink.add("LoopValueCarry", carry)?;
        }
        for (function, region) in roots {
            sink.set("RecursiveFunction", function, reach.recursive.contains(&function))?;
            sink.add("RootFunction", (region, function))?;
        }
        if reach.multiple_owners {
            sink.add("InvalidDataflow", "executable region has more than one owner")?;
        }
        for (child, operation) in reach.owners {
            sink.set("RegionOwnedBy", child, operation)?;
        }
        for caller in reach.live {
            sink.add("LiveFunction", caller)?;
            for &region in reach.bodies.get(&caller).into_iter().flatten() {
                sink.add("LiveRegion", region)?;
                if let Some(body) = reach.regions.get(&region) {
                    for &operation in &body.operations {
                        sink.add("LiveOperation", operation)?;
                    }
                }
            }
        }
        Ok(())
    })?;
    Ok(())
}
