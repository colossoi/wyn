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
}

pub(in crate::egglog) struct Reachability {
    pub calls: CallGraph<Value, Value>,
    pub(super) live: Vec<Value>,
    recursive: LookupSet<Value>,
    pub(super) bodies: LookupMap<Value, Vec<Value>>,
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
        for (&child, sites) in &positions.region_uses {
            for &site in sites {
                let operation = site.operation();
                multiple_owners |= owners.insert(child, operation).is_some_and(|old| old != operation);
                if graph.has_operation(operation) {
                    if let Some(parent) =
                        positions.operations.get(&operation).and_then(|position| position.region)
                    {
                        regions.entry(parent).or_default().children.push(child);
                    }
                }
            }
        }
        for (&operation, &callee) in &positions.targets {
            if let Some(region) = positions.operations.get(&operation).and_then(|position| position.region)
            {
                regions.entry(region).or_default().sites.push(CallSite { operation, callee });
            }
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
            owners,
            multiple_owners,
        })
    }

    pub fn roots(&self, graph: &Graph) -> Vec<Value> {
        self.live.iter().filter_map(|&function| graph.root_of(function)).collect()
    }

    /// Select calls in callee-first order so a clone contains all selected callee edits.
    pub fn inlines(&self, graph: &Graph, all: bool) -> Result<Vec<(Value, Value)>, OptimizeError> {
        let query = Query(graph);
        let positions = graph.structure().read();
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
                for &operation in positions.by_region.get(&region).into_iter().flatten() {
                    if graph.has_operation(operation) && graph.code(operation)? == Code::Parallel {
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

pub(super) fn publish(
    graph: &mut Graph,
    reach: &Reachability,
    scopes: &[super::scopes::Subjects],
) -> Result<(), OptimizeError> {
    let affected: LookupSet<_> = scopes.iter().filter_map(|scope| scope.function).collect();
    let query = Query(graph);
    let mut loop_carries = Vec::new();
    let positions = graph.structure().read();
    for &operation in scopes.iter().flat_map(|scope| &scope.operations) {
        let Some(edge) = positions.controls.get(&(operation, 0)) else {
            continue;
        };
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
    let roots: Vec<_> = affected
        .iter()
        .filter_map(|&function| positions.roots.get(&function).map(|&root| (function, root)))
        .collect();
    let mut callees = Vec::new();
    for caller in &affected {
        for site in reach.calls.calls.get(caller).into_iter().flatten() {
            if let Some(&region) = positions.roots.get(&site.callee) {
                callees.push((site.operation, region));
            }
        }
    }
    drop(positions);
    let changed = std::mem::take(&mut graph.dirty_owners);
    graph.update(|mut sink| {
        for &region in &changed {
            sink.remove("RegionOwnedBy", region)?;
        }
        for call in callees {
            sink.add("CallBody", call)?;
        }
        for carry in loop_carries {
            sink.add("LoopValueCarry", carry)?;
        }
        for (function, region) in roots {
            sink.set("RecursiveFunction", function, reach.recursive.contains(&function))?;
            sink.add("RootFunction", (region, function))?;
        }
        let site = sink.add("InvalidRegionOwners", RawValues(vec![]))?;
        let invalid_owners = (site, "executable region has more than one owner");
        if reach.multiple_owners {
            sink.add("InvalidDataflow", invalid_owners)?;
        } else {
            sink.remove("InvalidDataflow", invalid_owners)?;
        }
        for region in changed {
            if let Some(&operation) = reach.owners.get(&region) {
                sink.set("RegionOwnedBy", region, operation)?;
            }
        }
        for scope in scopes {
            if let Some(function) = scope.function {
                sink.add("LiveFunction", function)?;
            }
            for &region in &scope.regions {
                sink.add("LiveRegion", region)?;
            }
            for &operation in &scope.operations {
                sink.add("LiveOperation", operation)?;
            }
        }
        Ok(())
    })?;
    Ok(())
}
