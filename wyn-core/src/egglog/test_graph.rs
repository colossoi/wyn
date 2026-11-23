//! Flat graph fixtures for phase tests; no executable syntax is reconstructed.
use super::dataflow::ir::{self, Code};
use super::query::Query;
use super::{from_tlc, memory, OptimizeError};
use crate::egglog::graph::Graph;
use egglog_engine::{Value, Write};

pub(super) fn fixture(codes: &[(i64, Code)], facts: &str) -> Result<Graph, OptimizeError> {
    let mut graph = Graph::default();
    memory::load(&mut graph)?;
    graph.update(|mut sink| {
        Ok((|| -> Result<(), OptimizeError> {
            for (id, code) in codes {
                let operation = sink.add("OperationId", *id)?;
                sink.add("Operation", operation)?;
                ir::properties(&mut sink, operation, code.clone())?;
            }
            Ok(())
        })())
    })??;
    facts_with_positions(&mut graph, facts)?;
    Ok(graph)
}

pub(super) fn import(source: &str) -> Result<Graph, OptimizeError> {
    let source = crate::compile_thru_tlc(source).unwrap();
    Ok(from_tlc::import(&source)?.0)
}

pub(super) fn entry(graph: &Graph) -> Value {
    Query(graph).first("EntryOrder").unwrap().unwrap()[0]
}

pub(super) fn root(graph: &Graph) -> Value {
    Query(graph).0.function_root(entry(graph)).unwrap()
}

pub(super) fn facts_with_positions(graph: &mut Graph, facts: &str) -> Result<(), OptimizeError> {
    let fields = [
        "FixtureRegion",
        "FixtureOrder",
        "FixtureFunctionRoot",
        "FixtureCallTarget",
        "FixtureControlRegion",
        "FixtureInput",
        "FixtureYield",
        "FixtureOutput",
        "FixtureParameter",
        "FixtureResultReducer",
        "FixtureAccumulatorCombiner",
    ];
    if !fields.iter().any(|field| facts.contains(field)) {
        graph.parse_and_run_program(None, facts)?;
        return Ok(());
    }
    if !graph.get_function_names().iter().any(|name| name == "FixtureRegion") {
        graph.parse_and_run_program(
            None,
            "
            (function FixtureRegion (OperationKey) RegionKey :no-merge)
            (function FixtureOrder (OperationKey) i64 :no-merge)
            (function FixtureFunctionRoot (FunctionName) RegionKey :no-merge)
            (function FixtureCallTarget (OperationKey) FunctionName :no-merge)
            (function FixtureControlRegion (OperationKey i64) RegionKey :no-merge)
            (function FixtureInput (OperationKey i64) Name :no-merge)
            (function FixtureYield (RegionKey i64) Name :no-merge)
            (function FixtureOutput (OperationKey i64) Name :no-merge)
            (function FixtureParameter (RegionKey i64) Name :no-merge)
            (function FixtureResultReducer (OperationKey i64) RegionKey :no-merge)
            (function FixtureAccumulatorCombiner (OperationKey i64) RegionKey :no-merge)
        ",
        )?;
    }
    let (setup, checks) = facts.split_once("(run-schedule").unwrap_or((facts, ""));
    graph.parse_and_run_program(None, setup)?;
    let mut regions = Vec::new();
    let mut orders = Vec::new();
    let query = Query(graph);
    query.for_function("FixtureRegion", |key, region| {
        regions.push((key[0], region));
        Ok(())
    })?;
    query.for_function("FixtureOrder", |key, order| {
        orders.push((key[0], graph.value_to_base::<i64>(order)));
        Ok(())
    })?;
    let mut roots = Vec::new();
    let mut targets = Vec::new();
    query.for_function("FixtureFunctionRoot", |key, region| {
        roots.push((key[0], region));
        Ok(())
    })?;
    query.for_function("FixtureCallTarget", |key, function| {
        targets.push((key[0], function));
        Ok(())
    })?;
    let mut controls = Vec::new();
    query.for_function("FixtureControlRegion", |keys, child| {
        controls.push((keys[0], graph.value_to_base::<i64>(keys[1]), child));
        Ok(())
    })?;
    let mut inputs = crate::LookupMap::<Value, Vec<(i64, Value)>>::new();
    query.for_function("FixtureInput", |keys, value| {
        inputs.entry(keys[0]).or_default().push((graph.value_to_base::<i64>(keys[1]), value));
        Ok(())
    })?;
    let mut yields = crate::LookupMap::<Value, Vec<(i64, Value)>>::new();
    query.for_function("FixtureYield", |keys, value| {
        yields.entry(keys[0]).or_default().push((graph.value_to_base::<i64>(keys[1]), value));
        Ok(())
    })?;
    let mut outputs = crate::LookupMap::<Value, Vec<(i64, Value)>>::new();
    query.for_function("FixtureOutput", |keys, value| {
        outputs.entry(keys[0]).or_default().push((graph.value_to_base::<i64>(keys[1]), value));
        Ok(())
    })?;
    let mut parameters = crate::LookupMap::<Value, Vec<(i64, Value)>>::new();
    query.for_function("FixtureParameter", |keys, value| {
        parameters.entry(keys[0]).or_default().push((graph.value_to_base::<i64>(keys[1]), value));
        Ok(())
    })?;
    let mut reducers = Vec::new();
    let mut combiners = Vec::new();
    for (table, output) in [
        ("FixtureResultReducer", &mut reducers),
        ("FixtureAccumulatorCombiner", &mut combiners),
    ] {
        query.for_function(table, |keys, region| {
            output.push((keys[0], graph.value_to_base::<i64>(keys[1]), region));
            Ok(())
        })?;
    }
    graph.update(|mut sink| {
        for (op, slot, region) in reducers {
            sink.set_reducer(op, slot, region);
            sink.set("RegionOwnedBy", region, op)?;
            sink.add("ReducerBody", (op, slot, region))?;
        }
        for (op, slot, region) in combiners {
            sink.set_combiner(op, slot, region);
            sink.set("RegionOwnedBy", region, op)?;
        }
        for (region, mut names) in parameters {
            sink.add("Region", region)?;
            names.sort_unstable_by_key(|&(slot, _)| slot);
            assert!(names.iter().enumerate().all(|(i, &(slot, _))| i as i64 == slot));
            for &(slot, name) in &names {
                let site = sink.add("ParameterSite", (region, slot))?;
                sink.set("Defines", name, site)?;
                sink.add("FormalOrigin", (name, region, slot))?;
            }
            sink.set_parameters(region, names.into_iter().map(|(_, name)| name).collect())?;
        }
        for (operation, mut values) in outputs {
            values.sort_unstable_by_key(|&(slot, _)| slot);
            assert!(values.iter().enumerate().all(|(i, &(slot, _))| i as i64 == slot));
            for &(slot, value) in &values {
                let site = sink.add("OperationSite", (operation, slot))?;
                sink.set("Defines", value, site)?;
                sink.add("ResultOrigin", (value, operation, slot))?;
            }
            sink.set_outputs(operation, values.into_iter().map(|(_, value)| value).collect())?;
        }
        for (region, mut values) in yields {
            values.sort_unstable_by_key(|&(slot, _)| slot);
            assert!(values.iter().enumerate().all(|(i, &(slot, _))| i as i64 == slot));
            for &(_, value) in &values {
                sink.add("EndDemand", (region, value))?;
            }
            sink.set_yields(region, values.into_iter().map(|(_, value)| value).collect())?;
        }
        for (operation, mut values) in inputs {
            values.sort_unstable_by_key(|&(slot, _)| slot);
            assert!(values.iter().enumerate().all(|(i, &(slot, _))| i as i64 == slot));
            for &(_, value) in &values {
                sink.add("DirectOperationUse", (operation, value))?;
            }
            sink.set_inputs(operation, values.into_iter().map(|(_, value)| value).collect())?;
        }
        for (operation, slot, child) in controls {
            sink.set_control(operation, slot, child);
            sink.set("RegionOwnedBy", child, operation)?;
        }
        for (function, region) in roots {
            sink.set_function_root(function, region);
            sink.add("RootFunction", (region, function))?;
        }
        for (operation, function) in targets {
            sink.set_call_target(operation, function);
        }
        for (operation, region) in regions {
            sink.set_region(operation, region);
        }
        for (operation, order) in orders {
            sink.set_order(operation, order);
        }
        Ok(())
    })?;
    for field in fields {
        graph.clear_function(field)?;
    }
    if !checks.is_empty() {
        super::analysis::publish_proof_operands(graph)?;
        graph.parse_and_run_program(None, &format!("(run-schedule{checks}"))?;
    }
    Ok(())
}
