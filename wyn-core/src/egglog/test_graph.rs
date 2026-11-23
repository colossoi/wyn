//! Flat graph fixtures for phase tests; no executable syntax is reconstructed.
use super::dataflow::ir::{self, Code};
use super::query::Query;
use super::{from_tlc, memory, OptimizeError};
use crate::egglog::graph::{
    CollectiveInput, Dimension, Formula, Graph, IterationDomain, OutputBinding, ParallelResults,
};
use egglog_engine::{Value, Write};

pub(super) fn fixture(codes: &[(i64, Code)], facts: &str) -> Result<Graph, OptimizeError> {
    let mut graph = Graph::default();
    memory::load(&mut graph)?;
    graph.update(|mut sink| {
        Ok((|| -> Result<(), OptimizeError> {
            for (id, code) in codes {
                let operation = sink.add("OperationId", *id)?;

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
    graph.entries()[0].0
}

pub(super) fn for_entries(
    query: Query<'_>,
    mut visit: impl FnMut(&[Value], Value) -> Result<(), OptimizeError>,
) -> Result<(), OptimizeError> {
    for (function, order) in query.0.entries() {
        visit(&[function], query.0.base_to_value(order))?;
    }
    Ok(())
}

pub(super) fn root(graph: &Graph) -> Value {
    Query(graph).0.function_root(entry(graph)).unwrap()
}

pub(super) fn facts_with_positions(graph: &mut Graph, facts: &str) -> Result<(), OptimizeError> {
    let fields = [
        "FixtureRegion",
        "FixtureOrder",
        "FixtureFunctionRoot",
        "FixtureEntryOrder",
        "FixtureExternal",
        "FixtureCallTarget",
        "FixtureControlRegion",
        "FixtureInput",
        "FixtureYield",
        "FixtureOutput",
        "FixtureParameter",
        "FixtureResultReducer",
        "FixtureAccumulatorCombiner",
        "FixtureOperationIndex",
        "FixtureOperationShape",
        "FixtureOperationSlices",
        "FixtureResultValue",
        "FixtureResultNeutral",
        "FixtureResultPredicate",
        "FixtureAccumulatorArray",
        "FixtureOperationDomain",
        "FixtureDomainRank",
        "FixtureDomainAxis",
        "FixtureDomainOrdinal",
        "FixtureDomainExtent",
        "FixtureResultKindOf",
        "FixtureResultCount",
        "FixtureOutputClause",
        "FixtureOutputField",
        "FixtureOutputRole",
        "FixtureAccumulatorPolicy",
        "FixtureAccumulatorGroupCount",
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
            (function FixtureEntryOrder (FunctionName) i64 :no-merge)
            (relation FixtureExternal (FunctionName))
            (function FixtureCallTarget (OperationKey) FunctionName :no-merge)
            (function FixtureControlRegion (OperationKey i64) RegionKey :no-merge)
            (function FixtureInput (OperationKey i64) Name :no-merge)
            (function FixtureYield (RegionKey i64) Name :no-merge)
            (function FixtureOutput (OperationKey i64) Name :no-merge)
            (function FixtureParameter (RegionKey i64) Name :no-merge)
            (function FixtureResultReducer (OperationKey i64) RegionKey :no-merge)
            (function FixtureAccumulatorCombiner (OperationKey i64) RegionKey :no-merge)
            (function FixtureOperationIndex (OperationKey i64) IndexExp :no-merge)
            (function FixtureOperationShape (OperationKey) Shape :no-merge)
            (function FixtureOperationSlices (OperationKey) Slices :no-merge)
            (function FixtureResultValue (OperationKey i64 i64) Name :no-merge)
            (function FixtureResultNeutral (OperationKey i64 i64) Name :no-merge)
            (function FixtureResultPredicate (OperationKey i64) ValueName :no-merge)
            (function FixtureAccumulatorArray (OperationKey i64 i64) ValueName :no-merge)
            (function FixtureOperationDomain (OperationKey) Domain :no-merge)
            (function FixtureDomainRank (Domain) i64 :no-merge)
            (function FixtureDomainAxis (Domain i64) ValueName :no-merge)
            (function FixtureDomainOrdinal (Domain) ValueName :no-merge)
            (function FixtureDomainExtent (Domain i64) IndexExp :no-merge)
            (function FixtureResultKindOf (OperationKey i64) ResultKind :no-merge)
            (function FixtureResultCount (OperationKey) i64 :no-merge)
            (function FixtureOutputClause (OperationKey i64) i64 :no-merge)
            (function FixtureOutputField (OperationKey i64) i64 :no-merge)
            (function FixtureOutputRole (OperationKey i64) ResultRole :no-merge)
            (function FixtureAccumulatorPolicy (OperationKey i64) AccPolicy :no-merge)
            (function FixtureAccumulatorGroupCount (OperationKey) i64 :no-merge)
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
    let mut entries = Vec::new();
    let mut externals = Vec::new();
    query.for_function("FixtureEntryOrder", |key, order| {
        entries.push((key[0], graph.value_to_base::<i64>(order)));
        Ok(())
    })?;
    query.for_each("FixtureExternal", |row| {
        externals.push(row[0]);
        Ok(())
    })?;
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
    let mut collective = crate::LookupMap::<(Value, i64, CollectiveInput), Vec<(i64, Value)>>::new();
    for (table, kind) in [
        ("FixtureResultValue", CollectiveInput::Contribution),
        ("FixtureResultNeutral", CollectiveInput::Neutral),
        ("FixtureResultPredicate", CollectiveInput::Predicate),
        ("FixtureAccumulatorArray", CollectiveInput::Backing),
    ] {
        query.for_function(table, |keys, value| {
            let field = keys.get(2).map_or(0, |&key| graph.value_to_base::<i64>(key));
            collective
                .entry((keys[0], graph.value_to_base(keys[1]), kind))
                .or_default()
                .push((field, value));
            Ok(())
        })?;
    }
    let mut formulas = Vec::new();
    query.for_function("FixtureOperationIndex", |keys, term| {
        assert_eq!(graph.value_to_base::<i64>(keys[1]), 0);
        formulas.push((keys[0], Formula::Index, term));
        Ok(())
    })?;
    for (field, kind) in [
        ("FixtureOperationShape", Formula::Extents),
        ("FixtureOperationSlices", Formula::Slices),
    ] {
        query.for_function(field, |keys, term| {
            formulas.push((keys[0], kind, term));
            Ok(())
        })?;
    }
    let mut descriptors = crate::LookupMap::<Value, ParallelResults>::new();
    let mut policies = crate::LookupMap::<Value, Vec<Option<Value>>>::new();
    query.for_function("FixtureResultCount", |keys, count| {
        descriptors
            .entry(keys[0])
            .or_default()
            .kinds
            .resize(graph.value_to_base::<i64>(count) as usize, None);
        Ok(())
    })?;
    query.for_function("FixtureResultKindOf", |keys, kind| {
        let clause = graph.value_to_base::<i64>(keys[1]) as usize;
        let kinds = &mut descriptors.entry(keys[0]).or_default().kinds;
        kinds.resize(kinds.len().max(clause + 1), None);
        kinds[clause] = Some(kind);
        Ok(())
    })?;
    for table in ["FixtureOutputClause", "FixtureOutputField", "FixtureOutputRole"] {
        query.for_function(table, |keys, value| {
            let slot = graph.value_to_base::<i64>(keys[1]) as usize;
            let bindings = &mut descriptors.entry(keys[0]).or_default().bindings;
            bindings.resize(bindings.len().max(slot + 1), OutputBinding::default());
            let binding = &mut bindings[slot];
            match table {
                "FixtureOutputClause" => binding.clause = Some(graph.value_to_base(value)),
                "FixtureOutputField" => binding.field = Some(graph.value_to_base(value)),
                _ => binding.role = Some(value),
            }
            Ok(())
        })?;
    }
    query.for_function("FixtureAccumulatorGroupCount", |keys, count| {
        policies.entry(keys[0]).or_default().resize(graph.value_to_base::<i64>(count) as usize, None);
        Ok(())
    })?;
    query.for_function("FixtureAccumulatorPolicy", |keys, policy| {
        let group = graph.value_to_base::<i64>(keys[1]) as usize;
        let groups = policies.entry(keys[0]).or_default();
        groups.resize(groups.len().max(group + 1), None);
        groups[group] = Some(policy);
        Ok(())
    })?;
    let mut op_domains = Vec::new();
    let mut domains = crate::LookupMap::<Value, IterationDomain>::new();
    query.for_function("FixtureOperationDomain", |keys, domain| {
        op_domains.push((keys[0], domain));
        Ok(())
    })?;
    query.for_function("FixtureDomainRank", |keys, rank| {
        domains
            .entry(keys[0])
            .or_default()
            .dimensions
            .resize(graph.value_to_base::<i64>(rank) as usize, Dimension::default());
        Ok(())
    })?;
    query.for_function("FixtureDomainOrdinal", |keys, ordinal| {
        domains.entry(keys[0]).or_default().ordinal = Some(ordinal);
        Ok(())
    })?;
    for table in ["FixtureDomainAxis", "FixtureDomainExtent"] {
        query.for_function(table, |keys, value| {
            let slot = graph.value_to_base::<i64>(keys[1]) as usize;
            let dimensions = &mut domains.entry(keys[0]).or_default().dimensions;
            dimensions.resize(dimensions.len().max(slot + 1), Dimension::default());
            if table == "FixtureDomainAxis" {
                dimensions[slot].coordinate = Some(value);
            } else {
                dimensions[slot].extent = Some(value);
            }
            Ok(())
        })?;
    }
    graph.update(|mut sink| {
        for (op, domain) in op_domains {
            sink.set_operation_domain(op, domain);
        }
        for (id, domain) in domains {
            sink.set_domain(id, domain);
        }
        for (op, results) in descriptors {
            sink.set_parallel_results(op, results);
        }
        for (op, policies) in policies {
            sink.set_accumulator_policies(op, policies);
        }
        for (op, kind, formula) in formulas {
            sink.set_formula(op, kind, formula)?;
        }
        for ((op, group, kind), mut values) in collective {
            values.sort_unstable_by_key(|&(field, _)| field);
            assert!(values.iter().enumerate().all(|(index, &(field, _))| index as i64 == field));
            sink.set_collective_inputs(
                op,
                group,
                kind,
                values.into_iter().map(|(_, value)| value).collect(),
            )?;
        }
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
        for (function, order) in entries {
            sink.set_entry(function, order);
        }
        for function in externals {
            sink.set_external(function);
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
    } else {
        super::analysis::publish_collective_uses(graph)?;
    }
    Ok(())
}
