//! Compose compacted consumers and preserve the predicate around partial callbacks.
use crate::egglog::dataflow::edit;
use crate::egglog::dataflow::ir::{self, Code, Name, OperationId, Parameter, RegionId};
use crate::egglog::payload::Payload;
use crate::egglog::query::Query;
use crate::egglog::{analysis, OptimizeError};
use crate::op::BinaryOperator;
use crate::types;
use egglog_engine::{EGraph, FullState, RawValues, Value, Write};

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
    if let Some(candidate) = Query(graph).first("CompactCountCandidate")? {
        count(
            graph,
            candidate[0],
            graph.value_to_base::<i64>(candidate[1]),
            graph.value_to_base::<i64>(candidate[2]),
        )?;
        return Ok(true);
    }
    Ok(false)
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
    if !collected {
        mask(graph, second, predicate)?;
    }
    graph.update(|mut sink| {
        for (load, value) in loads {
            sink.add("FusionForwardLoad", (first, second, load, value))?;
        }
        Ok(())
    })?;
    super::fuse(graph, first, second)
}

fn mask(graph: &mut EGraph, operation: Value, predicate: Value) -> Result<(), OptimizeError> {
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
    graph.update(|mut sink| {
        Ok((|| -> Result<(), OptimizeError> {
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
            Ok(())
        })())
    })??;
    Ok(())
}

fn count(graph: &mut EGraph, operation: Value, clause: i64, output_slot: i64) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let source = query.required("Output", (operation, output_slot))?;
    let lane = query.required("ControlRegion", (operation, 0_i64))?;
    let predicate = query.required("ResultPredicate", (operation, clause))?;
    let mut lengths = Vec::new();
    query.for_each("LengthOperation", |row| {
        if query.0.value_to_base::<i64>(row[1]) == 0 && query.required("Input", (row[0], 0_i64))? == source
        {
            lengths.push((row[0], query.required("Output", (row[0], 0_i64))?));
        }
        Ok(())
    })?;
    let mut packed = None;
    query.for_function("OutputRole", |keys, role| {
        if keys[0] == operation && query.enode("CompactedOutput", role)?.is_some() {
            let other = query.required("OutputClause", (operation, keys[1]))?;
            if query.enode("Folded", query.required("ResultKindOf", (operation, other))?)?.is_some()
                && query.lookup("ResultPredicate", (operation, other))? == Some(predicate)
            {
                packed = Some(query.required("Output", (operation, keys[1]))?);
            }
        }
        Ok(())
    })?;
    if let Some(packed) = packed {
        graph.update(|mut sink| {
            for &(length, _) in &lengths {
                sink.remove("Input", (length, 0_i64))?;
                sink.set("Input", (length, 0_i64), packed)?;
            }
            Ok(())
        })?;
        return Ok(());
    }
    lengths.sort_by_key(|&(operation, _)| operation);
    let Some(&(_, output)) = lengths.first() else {
        return Ok(());
    };
    edit::replace(
        graph,
        &lengths.iter().skip(1).map(|&(_, name)| (name, output)).collect::<Vec<_>>(),
    )?;
    let mut order = 0;
    for operation in Query(graph).operations(lane)? {
        order = order.max(analysis::count(Query(graph), "OperationOrder", operation)? + 1);
    }
    let (zero, one) = graph.update(|mut sink| {
        Ok((|| -> Result<(Value, Value), OptimizeError> {
            let context = sink.add("DerivedOperation", (operation, 92_i64, clause))?;
            let element = sink.add("I32", RawValues(Vec::new()))?;
            let element = sink.add("ScalarElement", element)?;
            let info = sink.add("ScalarInfo", element)?;
            let returns = sink.add("ReturnScalar", element)?;
            let observe = sink.add("Observe", RawValues(Vec::new()))?;
            let zero = constant(&mut sink, context, 0, 0, info)?;
            let one = constant(&mut sink, context, 1, 1, info)?;
            let flag = sink.add("DerivedValue", (context, 92_i64, 2_i64))?;
            sink.set("ValueType", flag, info)?;
            let id = sink.add("DerivedOperation", (context, 92_i64, 2_i64))?;
            let choose = ir::operation_at(
                &mut sink,
                OperationId(id),
                Code::Scalar(Payload::Select),
                &[Name::Value(zero.1), Name::Value(one.1), Name::Value(predicate)],
            )?;
            sink.add("DiscardableOperation", id)?;
            sink.add("RepeatableOperation", id)?;
            ir::outputs(&mut sink, choose, &[Name::Value(flag)])?;
            sink.set("OperationRegion", id, lane)?;
            sink.set("OperationOrder", id, order)?;
            let left = sink.add("DerivedValue", (context, 92_i64, 3_i64))?;
            let right = sink.add("DerivedValue", (context, 92_i64, 4_i64))?;
            let sum = sink.add("DerivedValue", (context, 92_i64, 5_i64))?;
            sink.set("ValueType", sum, info)?;
            let id = sink.add("DerivedOperation", (context, 92_i64, 3_i64))?;
            let add = ir::operation_at(
                &mut sink,
                OperationId(id),
                Code::Scalar(Payload::Binary(BinaryOperator::Add)),
                &[Name::Value(left), Name::Value(right)],
            )?;
            sink.add("DiscardableOperation", id)?;
            sink.add("RepeatableOperation", id)?;
            ir::outputs(&mut sink, add, &[Name::Value(sum)])?;
            let reducer = sink.add("DerivedRegion", (context, 92_i64, 0_i64))?;
            sink.add("Region", reducer)?;
            sink.add("FlowScope", reducer)?;
            ir::parameters(
                &mut sink,
                RegionId(reducer),
                &[
                    Parameter {
                        name: Name::Value(left),
                        info,
                        diet: observe,
                    },
                    Parameter {
                        name: Name::Value(right),
                        info,
                        diet: observe,
                    },
                ],
            )?;
            ir::yields(&mut sink, RegionId(reducer), &[Name::Value(sum)])?;
            sink.set("RegionReturnType", (reducer, 0_i64), returns)?;
            sink.set("OperationRegion", id, reducer)?;
            sink.set("OperationOrder", id, 0_i64)?;
            let kind = sink.add("Total", RawValues(Vec::new()))?;
            let kind = sink.add("Folded", kind)?;
            sink.remove("ResultKindOf", (operation, clause))?;
            sink.set("ResultKindOf", (operation, clause), kind)?;
            sink.remove("ResultPredicate", (operation, clause))?;
            sink.remove("ResultValue", (operation, clause, 0_i64))?;
            let flag = Name::Value(flag).fact(&mut sink)?;
            sink.set("ResultValue", (operation, clause, 0_i64), flag)?;
            sink.set("ResultReducer", (operation, clause), reducer)?;
            sink.set("ResultNeutralCount", (operation, clause), 1_i64)?;
            let neutral = Name::Value(zero.1).fact(&mut sink)?;
            sink.set("ResultNeutral", (operation, clause, 0_i64), neutral)?;
            sink.set("ResultCommutative", (operation, clause), true)?;
            sink.remove("Output", (operation, output_slot))?;
            sink.set("Output", (operation, output_slot), output)?;
            let role = sink.add("TotalOutput", RawValues(Vec::new()))?;
            sink.remove("OutputRole", (operation, output_slot))?;
            sink.set("OutputRole", (operation, output_slot), role)?;
            Ok((zero.0, one.0))
        })())
    })??;
    edit::move_before(graph, zero, operation)?;
    edit::move_before(graph, one, operation)?;
    edit::remove(
        graph,
        &lengths.into_iter().map(|(operation, _)| operation).collect::<Vec<_>>(),
    )
}

fn constant(
    sink: &mut FullState<'_, '_>,
    owner: Value,
    slot: i64,
    bits: i64,
    info: Value,
) -> Result<(Value, Value), OptimizeError> {
    let value = sink.add("DerivedValue", (owner, 92_i64, slot))?;
    sink.set("ValueType", value, info)?;
    let id = sink.add("DerivedOperation", (owner, 92_i64, slot))?;
    let operation = ir::operation_at(
        sink,
        OperationId(id),
        Code::Constant {
            ty: types::i32(),
            bits,
        },
        &[],
    )?;
    ir::outputs(sink, operation, &[Name::Value(value)])?;
    Ok((id, value))
}
