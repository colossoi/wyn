//! Resolve direct calls through attached control, reducer and combiner regions.
use crate::egglog::graph::Graph;
use crate::egglog::query::Query;
use crate::egglog::OptimizeError;
use crate::memory_ir::call_graph::{CallGraph, CallSite};
use crate::{LookupMap, StableMap};
use egglog_engine::{RawValues, Value, Write};
use wyn_graph::{for_each_reachable, WalkOrder};

#[derive(Default)]
struct RegionCalls {
    children: Vec<Value>,
    sites: Vec<CallSite<Value, Value>>,
}

fn attached_region(query: Query<'_>, operation: Value) -> Result<Option<Value>, OptimizeError> {
    if query.contains("Operation", operation)? {
        return query.lookup("OperationRegion", operation);
    }
    Ok(None)
}

pub(super) fn publish(graph: &mut Graph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut regions = LookupMap::<Value, RegionCalls>::new();
    for table in ["ControlRegion", "ResultReducer", "AccumulatorCombiner"] {
        query.for_function(table, |keys, child| {
            if let Some(parent) = attached_region(query, keys[0])? {
                regions.entry(parent).or_default().children.push(child);
            }
            Ok(())
        })?;
    }
    query.for_function("CallTarget", |keys, callee| {
        let operation = keys[0];
        if let Some(region) = attached_region(query, operation)? {
            regions.entry(region).or_default().sites.push(CallSite { operation, callee });
        }
        Ok(())
    })?;
    let mut calls = StableMap::new();
    query.for_function("FunctionRoot", |keys, root| {
        let sites = calls.entry(keys[0]).or_insert_with(Vec::new);
        for_each_reachable(
            [root],
            WalkOrder::DepthFirst,
            |region, children| {
                if let Some(body) = regions.get(&region) {
                    children.extend_from_slice(&body.children);
                }
            },
            |region| {
                if let Some(body) = regions.get(&region) {
                    sites.extend_from_slice(&body.sites);
                }
            },
        );
        Ok(())
    })?;
    let call_graph = CallGraph { calls };
    let mut functions = Vec::new();
    query.for_each("Function", |row| {
        functions.push(row[0]);
        Ok(())
    })?;
    let mut roots = Vec::new();
    query.for_function("EntryOrder", |keys, _| {
        roots.push(keys[0]);
        Ok(())
    })?;
    if query.flag("RetainAllFunctions", RawValues(Vec::new()))? {
        roots.extend_from_slice(&functions);
    }
    let (live, recursive) = call_graph.analyze(roots);
    graph.update(|mut sink| {
        for function in functions {
            sink.set("RecursiveFunction", function, recursive.contains(&function))?;
        }
        for caller in live {
            sink.add("LiveFunction", caller)?;
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
