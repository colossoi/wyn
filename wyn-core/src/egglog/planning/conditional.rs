//! Factor admitted map arms by reparenting their regions around one branch occurrence.
use crate::egglog::dataflow::edit;
use crate::egglog::dataflow::ir::{self, Code, Name, OperationId, Parameter, RegionId};
use crate::egglog::query::Query;
use crate::egglog::{analysis, OptimizeError};
use egglog_engine::{EGraph, RawValues, Value, Write};

pub(super) fn apply(graph: &mut EGraph, branch: Value, yes: Value, no: Value) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let a = query.required("ControlRegion", (yes, 0_i64))?;
    let b = query.required("ControlRegion", (no, 0_i64))?;
    let yes_params = query.sequence("RegionParameter", a)?;
    let no_params = query.sequence("RegionParameter", b)?;
    edit::replace(graph, &no_params.into_iter().zip(yes_params).collect::<Vec<_>>())?;
    let query = Query(graph);
    let mut parameters = Vec::new();
    for slot in 0..analysis::count(query, "RegionParameterCount", a)? {
        let name = query.name(query.required("RegionParameter", (a, slot))?)?;
        parameters.push(Parameter {
            name: Name::Value(name),
            info: query.required("ValueType", name)?,
            diet: query.required("ParameterDiet", (a, slot))?,
        });
    }
    let parameter_count = parameters.len();
    let outputs = query.sequence("Output", branch)?;
    let count = outputs.len();
    let mut yes_values = Vec::new();
    let mut no_values = Vec::new();
    let mut returns = Vec::new();
    for slot in 0..count as i64 {
        let y = query.required("ResultValue", (yes, 0_i64, slot))?;
        let n = query.required("ResultValue", (no, 0_i64, slot))?;
        yes_values.push(Name::Value(query.name(y)?));
        no_values.push(Name::Value(query.name(n)?));
        let info = query.required("ValueType", query.name(y)?)?;
        let (kind, fields) = query.application(info, &["ScalarInfo", "LogicalValue", "ResourceInfo"])?;
        let kind = match kind {
            "ScalarInfo" => "ReturnScalar",
            "LogicalValue" => "LogicalReturn",
            "ResourceInfo" => "ReturnResource",
            _ => unreachable!(),
        };
        returns.push((info, kind, fields[0]));
    }
    let parent = query.required("OperationRegion", branch)?;
    let order = query.required("OperationOrder", branch)?;
    let domain = query.required("OperationDomain", yes)?;
    let execution = query.required("OperationExecution", yes)?;
    let width = query.lookup("OperationWidth", yes)?;
    let mut certificates = Vec::new();
    query.for_each("OperationCertificate", |row| {
        if [branch, yes, no].contains(&row[0]) {
            certificates.push(row[1]);
        }
        Ok(())
    })?;
    let outputs =
        outputs.into_iter().map(|name| query.name(name).map(Name::Value)).collect::<Result<Vec<_>, _>>()?;
    graph.update(|mut sink| {
        Ok((|| -> Result<(), OptimizeError> {
            let lane = sink.add("DerivedRegion", (branch, 61_i64, 0_i64))?;
            sink.add("Region", lane)?;
            sink.add("FlowScope", lane)?;
            ir::parameters(&mut sink, RegionId(lane), &parameters)?;
            ir::yields(&mut sink, RegionId(lane), &[])?;
            for arm in [a, b] {
                for slot in 0..parameter_count as i64 {
                    sink.remove("RegionParameter", (arm, slot))?;
                    sink.remove("ParameterDiet", (arm, slot))?;
                }
                sink.remove("RegionParameterCount", arm)?;
                sink.set("RegionParameterCount", arm, 0_i64)?;
                sink.remove("RegionYieldCount", arm)?;
            }
            ir::yields(&mut sink, RegionId(a), &yes_values)?;
            ir::yields(&mut sink, RegionId(b), &no_values)?;
            let mut bindings = Vec::new();
            for (slot, (info, kind, ty)) in returns.into_iter().enumerate() {
                let value = sink.add("DerivedValue", (branch, 61_i64, slot as i64))?;
                sink.set("ValueType", value, info)?;
                bindings.push(Name::Value(value));
                let info = sink.add(kind, ty)?;
                for arm in [a, b] {
                    sink.remove("RegionReturnType", (arm, slot as i64))?;
                    sink.set("RegionReturnType", (arm, slot as i64), info)?;
                }
                sink.remove("Output", (branch, slot as i64))?;
            }
            ir::outputs(&mut sink, OperationId(branch), &bindings)?;
            sink.remove("ControlRegion", (branch, 0_i64))?;
            sink.set("ControlRegion", (branch, 0_i64), a)?;
            sink.remove("ControlRegion", (branch, 1_i64))?;
            sink.set("ControlRegion", (branch, 1_i64), b)?;
            sink.remove("OperationRegion", branch)?;
            sink.set("OperationRegion", branch, lane)?;
            sink.remove("OperationOrder", branch)?;
            sink.set("OperationOrder", branch, 0_i64)?;
            let parallel = sink.add("DerivedOperation", (branch, 61_i64, 0_i64))?;
            let parallel = ir::operation_at(&mut sink, OperationId(parallel), Code::Parallel, &[])?;
            ir::outputs(&mut sink, parallel, &outputs)?;
            sink.set("OperationRegion", parallel.0, parent)?;
            sink.set("OperationOrder", parallel.0, order)?;
            sink.set("ControlRegion", (parallel.0, 0_i64), lane)?;
            let role = sink.add("LaneRegion", RawValues(Vec::new()))?;
            sink.set("ControlRole", (parallel.0, 0_i64), role)?;
            sink.set("OperationDomain", parallel.0, domain)?;
            sink.set("OperationExecution", parallel.0, execution)?;
            if let Some(width) = width {
                sink.set("OperationWidth", parallel.0, width)?;
            }
            sink.set("ResultCount", parallel.0, 1_i64)?;
            let kind = sink.add("Collected", RawValues(Vec::new()))?;
            sink.set("ResultKindOf", (parallel.0, 0_i64), kind)?;
            sink.set("ResultValueCount", (parallel.0, 0_i64), count as i64)?;
            let role = sink.add("CollectedOutput", RawValues(Vec::new()))?;
            for (slot, value) in bindings.into_iter().enumerate() {
                let name = value.fact(&mut sink)?;
                sink.set("ResultValue", (parallel.0, 0_i64, slot as i64), name)?;
                sink.set("OutputClause", (parallel.0, slot as i64), 0_i64)?;
                sink.set("OutputField", (parallel.0, slot as i64), slot as i64)?;
                sink.set("OutputRole", (parallel.0, slot as i64), role)?;
            }
            for value in certificates {
                sink.add("OperationCertificate", (parallel.0, value))?;
            }
            Ok(())
        })())
    })??;
    edit::remove(graph, &[yes, no])
}
