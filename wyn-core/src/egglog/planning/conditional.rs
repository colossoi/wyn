//! Factor admitted map arms by reparenting their regions around one branch occurrence.
use crate::egglog::dataflow::edit;
use crate::egglog::dataflow::ir::{self, Code, Name, OperationId, Parameter, RegionId};
use crate::egglog::graph::CertificateSite;
use crate::egglog::graph::Graph;
use crate::egglog::graph::{CollectiveInput, OutputBinding, ParallelResults};
use crate::egglog::query::Query;
use crate::egglog::OptimizeError;
use egglog_engine::{RawValues, Value, Write};

pub(super) fn apply(graph: &mut Graph, branch: Value, yes: Value, no: Value) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let a = query.0.control_region(yes, 0_i64)?;
    let b = query.0.control_region(no, 0_i64)?;
    let yes_params = query.0.parameters(a);
    let no_params = query.0.parameters(b);
    edit::replace(graph, &no_params.into_iter().zip(yes_params).collect::<Vec<_>>())?;
    let query = Query(graph);
    let mut parameters = Vec::new();
    for slot in 0..query.0.parameter_count(a) {
        let name = query.name(query.0.parameter(a, slot)?)?;
        parameters.push(Parameter {
            name: Name::Value(name),
            info: query.required("ValueType", name)?,
            diet: query.required("ParameterDiet", (a, slot))?,
        });
    }
    let parameter_count = parameters.len();
    let outputs = query.0.outputs(branch);
    let count = outputs.len();
    let mut yes_values = Vec::new();
    let mut no_values = Vec::new();
    let mut returns = Vec::new();
    for slot in 0..count as i64 {
        let y = query.0.collective_input(yes, 0_i64, CollectiveInput::Contribution, slot)?;
        let n = query.0.collective_input(no, 0_i64, CollectiveInput::Contribution, slot)?;
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
    let parent = query.0.operation_region(branch)?;
    let order = query.0.operation_order(branch)?;
    let domain = query.0.operation_domain(yes)?;
    let execution = query.required("OperationExecution", yes)?;
    let width = query.lookup("OperationWidth", yes)?;
    let mut arm_operations = Vec::new();
    for (slot, (map, lane)) in [(yes, a), (no, b)].into_iter().enumerate() {
        let arm = query.0.control_region(branch, slot as i64)?;
        let mut operations = query.operations(arm)?;
        operations.retain(|&operation| operation != map);
        if !operations.is_empty() {
            operations.extend(query.operations(lane)?);
            arm_operations.push((lane, operations));
        }
    }
    let mut certificates = graph.operation_certificates(yes);
    certificates.extend(graph.operation_certificates(no));
    graph.update(|mut sink| {
        Ok((|| -> Result<(), OptimizeError> {
            let lane = sink.add("DerivedRegion", (branch, 61_i64, 0_i64))?;

            sink.add("FlowScope", lane)?;
            ir::parameters(&mut sink, RegionId(lane), &parameters)?;
            ir::yields(&mut sink, RegionId(lane), &[])?;
            for (arm, operations) in arm_operations {
                for (slot, operation) in operations.into_iter().enumerate() {
                    sink.set_region(operation, arm);
                    sink.set_order(operation, slot as i64);
                }
            }
            for arm in [a, b] {
                for slot in 0..parameter_count as i64 {
                    sink.remove("ParameterDiet", (arm, slot))?;
                }
                sink.set_parameters(arm, Vec::new())?;
            }
            ir::yields(&mut sink, RegionId(a), &yes_values)?;
            ir::yields(&mut sink, RegionId(b), &no_values)?;
            let mut bindings = Vec::new();
            for (slot, (info, kind, ty)) in returns.into_iter().enumerate() {
                let value = sink.add("DerivedValue", (branch, 61_i64, slot as i64))?;
                sink.set_value_type(value, info)?;
                bindings.push(Name::Value(value));
                let info = sink.add(kind, ty)?;
                for arm in [a, b] {
                    sink.remove("RegionReturnType", (arm, slot as i64))?;
                    sink.set("RegionReturnType", (arm, slot as i64), info)?;
                }
            }
            ir::outputs(&mut sink, OperationId(branch), &bindings)?;
            sink.set_control(branch, 0_i64, a);
            sink.set_control(branch, 1_i64, b);
            sink.set_region(branch, lane);
            sink.set_order(branch, 0_i64);
            let parallel = sink.add("DerivedOperation", (branch, 61_i64, 0_i64))?;
            let parallel = ir::operation_at(&mut sink, OperationId(parallel), Code::Parallel, &[])?;
            sink.set_outputs(parallel.0, outputs)?;
            sink.set_region(parallel.0, parent);
            sink.set_order(parallel.0, order);
            sink.set_control(parallel.0, 0_i64, lane);
            sink.set_operation_domain(parallel.0, domain);
            sink.set("OperationExecution", parallel.0, execution)?;
            if let Some(width) = width {
                sink.set("OperationWidth", parallel.0, width)?;
            }
            let kind = sink.add("Collected", RawValues(Vec::new()))?;
            let values =
                bindings.into_iter().map(|value| value.fact(&mut sink)).collect::<Result<Vec<_>, _>>()?;
            let role = sink.add("CollectedOutput", RawValues(Vec::new()))?;
            sink.set_parallel_results(
                parallel.0,
                ParallelResults {
                    kinds: vec![Some(kind)],
                    bindings: (0..values.len())
                        .map(|slot| OutputBinding::new(0, slot as i64, role))
                        .collect(),
                },
            );
            sink.set_collective_inputs(parallel.0, 0, CollectiveInput::Contribution, values)?;
            for value in certificates {
                sink.add_certificate(CertificateSite::Operation(parallel.0), value);
            }
            Ok(())
        })())
    })??;
    edit::remove(graph, &[yes, no])
}
