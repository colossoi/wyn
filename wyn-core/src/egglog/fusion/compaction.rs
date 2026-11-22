//! Compose compacted consumers and preserve the predicate around partial callbacks.
use crate::egglog::dataflow::ir::{self, Code, Name, OperationId, Parameter, RegionId};
use crate::egglog::query::Query;
use crate::egglog::{analysis, OptimizeError};
use egglog_engine::{EGraph, RawValues, Value, Write};

mod count;

pub(super) fn run(graph: &mut EGraph) -> Result<bool, OptimizeError> {
    graph.parse_and_run_program(
        None,
        "(run-schedule
        (saturate compact-index epilogue-index)
        (saturate transform-availability dataflow-scopes)
        (saturate compact-legality epilogue-legality)
        (saturate compact-select))",
    )?;
    if let Some(pair) = Query(graph).first("CompactCandidate")? {
        consumer(graph, pair[0], pair[1], graph.value_to_base::<i64>(pair[2]))?;
        return Ok(true);
    }
    count::run(graph)
}

fn consumer(graph: &mut EGraph, first: Value, second: Value, clause: i64) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let predicate = query.required("ResultPredicate", (first, clause))?;
    let mut loads = Vec::new();
    query.for_each("PointwiseLoad", |row| {
        if row[0] == first && row[1] == second {
            let source = query.required("Input", (row[2], 0_i64))?;
            query.for_function("Output", |keys, output| {
                if keys[0] == first
                    && output == source
                    && analysis::count(query, "OutputClause", (first, keys[1]))? == clause
                {
                    let field = query.required("OutputField", (first, keys[1]))?;
                    loads.push((
                        row[2],
                        query.required("ResultValue", (first, query.0.base_to_value(clause), field))?,
                    ));
                }
                Ok(())
            })?;
        }
        Ok(())
    })?;
    let collected = query.enode("Collected", query.required("ResultKindOf", (second, 0_i64))?)?.is_some();
    let outputs = analysis::count(query, "OutputCount", second)?;
    let mut prefixes = Vec::new();
    for slot in 0..outputs {
        if query.enode("PrefixOutput", query.required("OutputRole", (second, slot))?)?.is_some() {
            prefixes.push(slot);
        }
    }
    let packed = if collected { (0..outputs).collect() } else { prefixes };
    if !packed.is_empty() {
        graph.update(|mut sink| {
            if collected {
                let kind = sink.add("Compacted", RawValues(Vec::new()))?;
                sink.remove("ResultKindOf", (second, 0_i64))?;
                sink.set("ResultKindOf", (second, 0_i64), kind)?;
            }
            sink.set("ResultPredicate", (second, 0_i64), predicate)?;
            let role = sink.add("CompactedOutput", RawValues(Vec::new()))?;
            for slot in packed {
                sink.remove("OutputRole", (second, slot))?;
                sink.set("OutputRole", (second, slot), role)?;
            }
            Ok(())
        })?;
    }
    let new_regions = if collected { Vec::new() } else { mask(graph, second, predicate)? };
    super::parallel::apply(graph, first, second, &loads, &new_regions)
}

fn mask(
    graph: &mut EGraph,
    operation: Value,
    predicate: Value,
) -> Result<Vec<(Value, Vec<Value>)>, OptimizeError> {
    let query = Query(graph);
    let yes = query.required("ControlRegion", (operation, 0_i64))?;
    let mut parameters = Vec::new();
    for slot in 0..analysis::count(query, "RegionParameterCount", yes)? {
        let name = query.name(query.required("RegionParameter", (yes, slot))?)?;
        parameters.push(Parameter {
            name: Name::Value(name),
            info: query.required("ValueType", name)?,
            diet: query.required("ParameterDiet", (yes, slot))?,
        });
    }
    let count = analysis::count(query, "ResultValueCount", (operation, 0_i64))?;
    let mut contributions = Vec::new();
    let mut neutral = Vec::new();
    let mut types = Vec::new();
    for slot in 0..count {
        let result = query.required("ResultValue", (operation, 0_i64, slot))?;
        if let Some(fields) = query.enode("AccNameRef", result)? {
            let token = fields[0];
            let capability = query.required("TokenCapability", token)?;
            let mut incoming = None;
            query.for_each("RegionCapture", |row| {
                if row[0] == yes {
                    if let Some(fields) = query.enode("AccNameRef", row[1])? {
                        if query.required("TokenCapability", fields[0])? == capability {
                            incoming = Some(fields[0]);
                        }
                    }
                }
                Ok(())
            })?;
            let Some(incoming) = incoming else {
                return Err(OptimizeError::Output(
                    "guarded contribution has no incoming capability".into(),
                ));
            };
            contributions.push(Name::Accumulator(token));
            neutral.push(Name::Accumulator(incoming));
            let info = query.required("AccumulatorType", token)?;
            types.push((Name::Accumulator(token), info, info));
        } else {
            let value = query.name(result)?;
            contributions.push(Name::Value(value));
            neutral.push(Name::Value(
                query.name(query.required("ResultNeutral", (operation, 0_i64, slot))?)?,
            ));
            let info = query.required("ValueType", value)?;
            types.push((Name::Value(value), info, query.fields("ScalarInfo", info)?[0]));
        }
    }
    let (branch, lane, otherwise) = graph.update(|mut sink| {
        Ok((|| -> Result<_, OptimizeError> {
            let no = sink.add("DerivedRegion", (operation, 91_i64, 0_i64))?;
            let lane = sink.add("DerivedRegion", (operation, 91_i64, 1_i64))?;
            for region in [no, lane] {
                sink.add("Region", region)?;
                sink.add("FlowScope", region)?;
            }
            ir::parameters(&mut sink, RegionId(no), &[])?;
            ir::parameters(&mut sink, RegionId(lane), &parameters)?;
            for slot in 0..parameters.len() as i64 {
                sink.remove("RegionParameter", (yes, slot))?;
                sink.remove("ParameterDiet", (yes, slot))?;
            }
            sink.remove("RegionParameterCount", yes)?;
            sink.set("RegionParameterCount", yes, 0_i64)?;
            sink.remove("RegionYieldCount", yes)?;
            ir::yields(&mut sink, RegionId(yes), &contributions)?;
            ir::yields(&mut sink, RegionId(no), &neutral)?;
            ir::yields(&mut sink, RegionId(lane), &[])?;
            let id = sink.add("DerivedOperation", (operation, 91_i64, 0_i64))?;
            let branch = ir::operation_at(
                &mut sink,
                OperationId(id),
                Code::Branch,
                &[Name::Value(predicate)],
            )?;
            sink.set("OperationRegion", id, lane)?;
            sink.set("OperationOrder", id, 0_i64)?;
            for (slot, region, role) in [(0_i64, yes, "ThenRegion"), (1, no, "ElseRegion")] {
                let role = sink.add(role, RawValues(Vec::new()))?;
                sink.set("ControlRegion", (id, slot), region)?;
                sink.set("ControlRole", (id, slot), role)?;
            }
            let mut outputs = Vec::new();
            for (slot, (original, info, returns)) in types.into_iter().enumerate() {
                let (name, returns) = match original {
                    Name::Value(_) => {
                        let value = sink.add("DerivedValue", (operation, 91_i64, slot as i64))?;
                        sink.set("ValueType", value, info)?;
                        (Name::Value(value), sink.add("ReturnScalar", returns)?)
                    }
                    Name::Accumulator(token) => {
                        let value = sink.add("ClonedAccumulator", (id, token))?;
                        sink.set("AccumulatorType", value, info)?;
                        (Name::Accumulator(value), sink.add("ReturnAccumulator", returns)?)
                    }
                };
                outputs.push(name);
                for region in [yes, no] {
                    sink.set("RegionReturnType", (region, slot as i64), returns)?;
                }
                let name = name.fact(&mut sink)?;
                sink.remove("ResultValue", (operation, 0_i64, slot as i64))?;
                sink.set("ResultValue", (operation, 0_i64, slot as i64), name)?;
            }
            ir::outputs(&mut sink, branch, &outputs)?;
            sink.remove("ControlRegion", (operation, 0_i64))?;
            sink.set("ControlRegion", (operation, 0_i64), lane)?;
            Ok((id, lane, no))
        })())
    })??;
    Ok(vec![(lane, vec![branch]), (otherwise, vec![])])
}
