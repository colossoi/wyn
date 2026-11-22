//! Replace compacted arrays that are observed only through their outer length.
use crate::egglog::analysis::count;
use crate::egglog::dataflow::{
    edit,
    ir::{self, Code, Name, OperationId, Parameter, RegionId},
};
use crate::egglog::payload::Payload;
use crate::egglog::query::Query;
use crate::egglog::{timing, OptimizeError};
use crate::op::BinaryOperator;
use crate::types;
use crate::{LookupMap, LookupSet, StableMap};
use egglog_engine::{EGraph, FullState, RawValues, Value, Write};

struct Candidate {
    operation: Value,
    clause: i64,
    slot: i64,
    lengths: Vec<(Value, Value)>,
    observed: bool,
}

pub(in crate::egglog::fusion) fn run(graph: &mut EGraph) -> Result<bool, OptimizeError> {
    let _timing = timing::span("Rust compact length");
    let query = Query(graph);
    let Some(compacted) = query.constructor("Compacted", RawValues(vec![]))? else {
        return Ok(false);
    };
    let mut clauses = LookupSet::new();
    query.for_function("ResultKindOf", |key, kind| {
        if kind == compacted
            && query.contains("LiveOperation", key[0])?
            && count(query, "ResultValueCount", (key[0], key[1]))? == 1
            && count(query, "DomainRank", query.required("OperationDomain", key[0])?)? == 1
        {
            clauses.insert((key[0], key[1]));
        }
        Ok(())
    })?;
    if clauses.is_empty() {
        return Ok(false);
    }
    let mut candidates = StableMap::new();
    query.for_function("OutputClause", |key, clause| {
        if clauses.contains(&(key[0], clause)) {
            candidates.insert(
                query.required("Output", (key[0], key[1]))?,
                Candidate {
                    operation: key[0],
                    clause: graph.value_to_base(clause),
                    slot: graph.value_to_base(key[1]),
                    lengths: Vec::new(),
                    observed: false,
                },
            );
        }
        Ok(())
    })?;
    let mut length_uses = LookupSet::new();
    query.for_each("LengthOperation", |row| {
        if graph.value_to_base::<i64>(row[1]) == 0 {
            let input = query.required("Input", (row[0], 0_i64))?;
            if let Some(candidate) = candidates.get_mut(&input) {
                let output = query.required("Output", (row[0], 0_i64))?;
                candidate.lengths.push((row[0], output));
                length_uses.insert((row[0], input));
            }
        }
        Ok(())
    })?;
    candidates.retain(|_, candidate| !candidate.lengths.is_empty());
    if candidates.is_empty() {
        return Ok(false);
    }
    query.for_each("DirectOperationUse", |row| {
        if let Some(candidate) = candidates.get_mut(&row[1]) {
            candidate.observed |= !length_uses.contains(&(row[0], row[1]));
        }
        Ok(())
    })?;
    query.for_each("EndDemand", |row| {
        if let Some(candidate) = candidates.get_mut(&row[1]) {
            candidate.observed = true;
        }
        Ok(())
    })?;
    candidates.retain(|_, candidate| !candidate.observed);
    if candidates.is_empty() {
        return Ok(false);
    }

    let mut reused = Vec::new();
    let mut reductions = Vec::new();
    let mut orders = StableMap::new();
    let mut removed = Vec::new();
    let mut replacements = Vec::new();
    for (_, mut candidate) in candidates {
        let operation = candidate.operation;
        let predicate = query.required("ResultPredicate", (operation, candidate.clause))?;
        if let Some(packed) = packed_output(query, operation, predicate)? {
            reused.extend(candidate.lengths.into_iter().map(|(length, _)| (length, packed)));
            continue;
        }
        candidate.lengths.sort_unstable();
        let Some(&(_, output)) = candidate.lengths.first() else {
            continue;
        };
        let region = query.required("OperationRegion", operation)?;
        let lane = query.required("ControlRegion", (operation, 0_i64))?;
        for region in [region, lane] {
            if !orders.contains_key(&region) {
                orders.insert(region, query.vector(query.required("RegionOperations", region)?)?);
            }
        }
        reductions.push((
            operation,
            candidate.clause,
            candidate.slot,
            predicate,
            output,
            lane,
        ));
        replacements.extend(candidate.lengths.iter().skip(1).map(|&(_, name)| (name, output)));
        removed.extend(candidate.lengths.into_iter().map(|(length, _)| length));
    }

    graph.update(|mut sink| {
        Ok((|| -> Result<(), OptimizeError> {
            for (length, packed) in reused {
                sink.remove("Input", (length, 0_i64))?;
                sink.remove("InputDiet", (length, 0_i64))?;
                sink.set("Input", (length, 0_i64), packed)?;
            }
            let mut before = LookupMap::<Value, Vec<Value>>::new();
            let mut generated = LookupSet::new();
            for (operation, clause, slot, predicate, output, lane) in reductions {
                let (constants, flag) = make_count(&mut sink, operation, clause, slot, predicate, output)?;
                before.entry(operation).or_default().extend(constants);
                orders.entry(lane).or_default().push(flag);
                generated.extend(constants);
                generated.insert(flag);
            }
            // Keep each affected body in Rust while inserting all counts into its order.
            for (region, operations) in orders {
                let mut order = 0_i64;
                for operation in operations {
                    for item in before.remove(&operation).into_iter().flatten().chain([operation]) {
                        if generated.contains(&item) {
                            sink.set("OperationRegion", item, region)?;
                        }
                        sink.remove("OperationOrder", item)?;
                        sink.set("OperationOrder", item, order)?;
                        order += 1;
                    }
                }
            }
            Ok(())
        })())
    })??;
    if !replacements.is_empty() {
        edit::replace(graph, &replacements)?;
    }
    if removed.is_empty() {
        edit::changed(graph)?;
    } else {
        edit::remove(graph, &removed)?;
    }
    Ok(true)
}

fn packed_output(
    query: Query<'_>,
    operation: Value,
    predicate: Value,
) -> Result<Option<Value>, OptimizeError> {
    for slot in 0..count(query, "OutputCount", operation)? {
        let role = query.required("OutputRole", (operation, slot))?;
        if query.enode("CompactedOutput", role)?.is_none() {
            continue;
        }
        let other = query.required("OutputClause", (operation, slot))?;
        if query.enode("Folded", query.required("ResultKindOf", (operation, other))?)?.is_some()
            && query.lookup("ResultPredicate", (operation, other))? == Some(predicate)
        {
            return Ok(Some(query.required("Output", (operation, slot))?));
        }
    }
    Ok(None)
}

fn make_count(
    sink: &mut FullState<'_, '_>,
    operation: Value,
    clause: i64,
    output_slot: i64,
    predicate: Value,
    output: Value,
) -> Result<([Value; 2], Value), OptimizeError> {
    let context = sink.add("DerivedOperation", (operation, 92_i64, clause))?;
    let element = sink.add("I32", RawValues(Vec::new()))?;
    let element = sink.add("ScalarElement", element)?;
    let info = sink.add("ScalarInfo", element)?;
    let returns = sink.add("ReturnScalar", element)?;
    let observe = sink.add("Observe", RawValues(Vec::new()))?;
    let zero = constant(sink, context, 0, 0, info)?;
    let one = constant(sink, context, 1, 1, info)?;
    let flag = sink.add("DerivedValue", (context, 92_i64, 2_i64))?;
    sink.set("ValueType", flag, info)?;
    let id = sink.add("DerivedOperation", (context, 92_i64, 2_i64))?;
    let flag_operation = ir::operation_at(
        sink,
        OperationId(id),
        Code::Scalar(Payload::Select),
        &[Name::Value(zero.1), Name::Value(one.1), Name::Value(predicate)],
    )?;
    sink.add("DiscardableOperation", id)?;
    sink.add("RepeatableOperation", id)?;
    ir::outputs(sink, flag_operation, &[Name::Value(flag)])?;
    let left = sink.add("DerivedValue", (context, 92_i64, 3_i64))?;
    let right = sink.add("DerivedValue", (context, 92_i64, 4_i64))?;
    let sum = sink.add("DerivedValue", (context, 92_i64, 5_i64))?;
    sink.set("ValueType", sum, info)?;
    let id = sink.add("DerivedOperation", (context, 92_i64, 3_i64))?;
    let add = ir::operation_at(
        sink,
        OperationId(id),
        Code::Scalar(Payload::Binary(BinaryOperator::Add)),
        &[Name::Value(left), Name::Value(right)],
    )?;
    sink.add("DiscardableOperation", id)?;
    sink.add("RepeatableOperation", id)?;
    ir::outputs(sink, add, &[Name::Value(sum)])?;
    let reducer = sink.add("DerivedRegion", (context, 92_i64, 0_i64))?;
    sink.add("Region", reducer)?;
    sink.add("FlowScope", reducer)?;
    ir::parameters(
        sink,
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
    ir::yields(sink, RegionId(reducer), &[Name::Value(sum)])?;
    sink.set("RegionReturnType", (reducer, 0_i64), returns)?;
    sink.set("OperationRegion", id, reducer)?;
    sink.set("OperationOrder", id, 0_i64)?;
    let kind = sink.add("Total", RawValues(Vec::new()))?;
    let kind = sink.add("Folded", kind)?;
    sink.remove("ResultKindOf", (operation, clause))?;
    sink.set("ResultKindOf", (operation, clause), kind)?;
    sink.remove("ResultPredicate", (operation, clause))?;
    sink.remove("ResultValue", (operation, clause, 0_i64))?;
    let flag = Name::Value(flag).fact(sink)?;
    sink.set("ResultValue", (operation, clause, 0_i64), flag)?;
    sink.set("ResultReducer", (operation, clause), reducer)?;
    sink.set("ResultNeutralCount", (operation, clause), 1_i64)?;
    let neutral = Name::Value(zero.1).fact(sink)?;
    sink.set("ResultNeutral", (operation, clause, 0_i64), neutral)?;
    sink.set("ResultCommutative", (operation, clause), true)?;
    sink.remove("Output", (operation, output_slot))?;
    sink.set("Output", (operation, output_slot), output)?;
    let role = sink.add("TotalOutput", RawValues(Vec::new()))?;
    sink.remove("OutputRole", (operation, output_slot))?;
    sink.set("OutputRole", (operation, output_slot), role)?;
    Ok(([zero.0, one.0], flag_operation.0))
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
