//! Apply an admitted domain composition to keyed cells and coordinate proofs.
use crate::egglog::dataflow::edit::{self, OPERATION_CELLS};
use crate::egglog::dataflow::ir::{self, Code, Name, OperationId, Parameter, RegionId};
use crate::egglog::graph::Graph;
use crate::egglog::query::Query;
use crate::egglog::{analysis, OptimizeError};
use egglog_engine::{RawValues, Value, Write};

pub(super) fn lift_row_length(graph: &mut Graph) -> Result<bool, OptimizeError> {
    let Some(candidate) = Query(graph).first("LiftUniformRowLength")? else {
        return Ok(false);
    };
    let (length, outer, array) = (candidate[0], candidate[1], candidate[2]);
    edit::set_code(graph, length, Code::Length { axis: 1 })?;
    edit::inputs(graph, length, &[array])?;
    edit::move_before(graph, length, outer)?;
    Ok(true)
}

pub(super) fn canonicalize(graph: &mut Graph) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    let mut replacements = Vec::new();
    query.for_function("DomainExtent", |keys, index| {
        if query.enode("IndexConstant", index)?.is_none() {
            if let Some(n) = query.lookup("StaticIndex", index)? {
                replacements.push((keys[0], keys[1], query.0.value_to_base::<i64>(n)));
            }
        }
        Ok(())
    })?;
    if replacements.is_empty() {
        return Ok(false);
    }
    graph.update(|mut sink| {
        for (domain, slot, n) in replacements {
            let value = sink.add("IndexConstant", n)?;
            sink.remove("DomainExtent", (domain, slot))?;
            sink.set("DomainExtent", (domain, slot), value)?;
        }
        Ok(())
    })?;
    edit::changed(graph)?;
    Ok(true)
}

pub(super) fn apply(graph: &mut Graph, outer: Value, inner: Value) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let domain = query.required("OperationDomain", outer)?;
    let inner_domain = query.required("OperationDomain", inner)?;
    let body = query.required("ControlRegion", (outer, 0_i64))?;
    let inner_body = query.required("ControlRegion", (inner, 0_i64))?;
    let rank = analysis::count(query, "DomainRank", domain)?;
    let inner_rank = analysis::count(query, "DomainRank", inner_domain)?;
    let flat = query.required("DomainOrdinal", domain)?;
    let inner_flat = query.required("DomainOrdinal", inner_domain)?;
    let mut axes = Vec::new();
    let mut inner_axes = Vec::new();
    for slot in 0..rank {
        axes.push((
            query.required("DomainAxis", (domain, slot))?,
            query.required("DomainExtent", (domain, slot))?,
        ));
    }
    for slot in 0..inner_rank {
        inner_axes.push((
            query.required("DomainAxis", (inner_domain, slot))?,
            query.required("DomainExtent", (inner_domain, slot))?,
        ));
    }
    let mut parameters = Vec::new();
    for (region, rank) in [(body, rank), (inner_body, inner_rank)] {
        for slot in 1..=rank {
            let name = query.required("RegionParameter", (region, slot))?;
            let value = query.fields("ValueNameRef", name)?[0];
            parameters.push(Parameter {
                name: Name::Value(value),
                info: query.required("ValueType", value)?,
                diet: query.required("ParameterDiet", (region, slot))?,
            });
        }
    }
    let info = query.required("ValueType", flat)?;
    let diet = query.required("ParameterDiet", (body, 0_i64))?;
    let prefix = query
        .operations(body)?
        .into_iter()
        .filter(|&operation| operation != inner)
        .map(|operation| Ok((operation, query.0.operation_order(operation)?)))
        .collect::<Result<Vec<_>, OptimizeError>>()?;
    let nested = query
        .operations(inner_body)?
        .into_iter()
        .map(|operation| Ok((operation, query.0.operation_order(operation)?)))
        .collect::<Result<Vec<_>, OptimizeError>>()?;
    let base = query.0.operation_order(inner)? + 2;
    let result_count = analysis::count(query, "ResultCount", inner)?;
    let mut removed = Vec::new();
    let mut copied = Vec::new();
    for &table in OPERATION_CELLS {
        if (table.starts_with("Result") && table != "ResultCount")
            || ["OutputClause", "OutputField", "OutputRole"].contains(&table)
        {
            query.for_function(table, |keys, value| {
                if keys[0] == outer {
                    removed.push((table, keys.to_vec()));
                }
                if keys[0] == inner {
                    let mut keys = keys.to_vec();
                    keys[0] = outer;
                    copied.push((table, keys, value));
                }
                Ok(())
            })?;
        }
    }
    let mut certificates = Vec::new();
    query.for_each("OperationCertificate", |row| {
        if row[0] == inner {
            certificates.push(row[1]);
        }
        Ok(())
    })?;
    // Only identities and small immutable coordinate metadata cross this edit boundary.
    graph.update(|mut sink| {
        Ok((|| -> Result<(), OptimizeError> {
            for (table, keys) in removed {
                sink.remove(table, RawValues(keys))?;
            }
            for (table, keys, value) in copied {
                sink.set(table, RawValues(keys), value)?;
            }
            sink.remove("ResultCount", outer)?;
            sink.set("ResultCount", outer, result_count)?;
            let ordinal = sink.add("DerivedValue", (inner, 40_i64, 0_i64))?;
            sink.set("ValueType", ordinal, info)?;
            sink.remove("DomainOrdinal", domain)?;
            sink.set("DomainOrdinal", domain, ordinal)?;
            sink.remove("DomainRank", domain)?;
            sink.set("DomainRank", domain, rank + inner_rank)?;
            for (slot, &(axis, extent)) in inner_axes.iter().enumerate() {
                sink.set("DomainAxis", (domain, rank + slot as i64), axis)?;
                sink.set("DomainExtent", (domain, rank + slot as i64), extent)?;
            }
            for slot in 0..=rank {
                sink.remove("RegionParameter", (body, slot))?;
                sink.remove("ParameterDiet", (body, slot))?;
            }
            sink.remove("RegionParameterCount", body)?;
            parameters.insert(
                0,
                Parameter {
                    name: Name::Value(ordinal),
                    info,
                    diet,
                },
            );
            ir::parameters(&mut sink, RegionId(body), &parameters)?;
            for (slot, (name, axes)) in [(flat, &axes), (inner_flat, &inner_axes)].into_iter().enumerate() {
                let operation = sink.add("DerivedOperation", (inner, 40_i64, slot as i64))?;
                let operation = ir::operation_at(&mut sink, OperationId(operation), Code::IndexValue, &[])?;
                ir::outputs(&mut sink, operation, &[Name::Value(name)])?;
                let mut index = sink.add("IndexConstant", 0_i64)?;
                for &(axis, extent) in axes {
                    let scaled = sink.add("IndexMul", (index, extent))?;
                    let coordinate = sink.add("IndexValue", axis)?;
                    index = sink.add("IndexAdd", (scaled, coordinate))?;
                }
                sink.set("OperationIndex", (operation.0, 0_i64), index)?;
                sink.set_region(operation.0, body);
                sink.set_order(operation.0, slot as i64);
            }
            for (offset, operations) in [(2_i64, prefix), (base, nested)] {
                for (operation, order) in operations {
                    let order = order + offset;
                    sink.set_region(operation, body);
                    sink.set_order(operation, order);
                }
            }
            for value in certificates {
                sink.add("OperationCertificate", (outer, value))?;
            }
            Ok(())
        })())
    })??;
    edit::remove(graph, &[inner])
}
