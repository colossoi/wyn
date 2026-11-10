use super::Execution;
use crate::egglog::query::Query;
use crate::egglog::{from_tlc, fuse, place, schedule};
use crate::{compile_thru_tlc, PipelineTopologyPolicy};
use egglog_engine::Write;

#[test]
fn incomplete_local_storage_is_rejected_at_the_execution_boundary() {
    let source = compile_thru_tlc(include_str!(
        "../../../testfiles/regressions/local_array_capacity.wyn"
    ))
    .unwrap();
    let program = schedule(
        place(
            fuse(from_tlc(&source).unwrap()).unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap(),
    )
    .unwrap();
    for (table, expected) in [
        ("LocalExecution", "local buffer has no execution plan"),
        ("PhysicalLocalCapacity", "PhysicalLocalCapacity"),
        ("LocalElementLayout", "LocalElementLayout"),
    ] {
        let mut graph = program.graph.clone();
        graph.clear_function(table).unwrap();
        let Err(error) = Execution::select(&graph) else {
            panic!("accepted local storage without {table}");
        };
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn incomplete_dispatches_are_rejected_before_scalar_lowering() {
    let source = compile_thru_tlc("entry main(xs:[]i32) []i32=map(|x|x+1,xs)").unwrap();
    let program = schedule(
        place(
            fuse(from_tlc(&source).unwrap()).unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap(),
    )
    .unwrap();
    let domain = Query(&program.graph).row("PhaseDomain", |_| true).unwrap().unwrap();
    for (mode, expected) in [
        ("missing", "stage domain missing"),
        ("ambiguous", "ambiguous PhaseDomain"),
        ("zero-width", "stage width must be positive"),
    ] {
        let mut graph = program.graph.clone();
        if mode != "ambiguous" {
            graph.clear_function("PhaseDomain").unwrap();
        }
        if mode != "missing" {
            graph.update(|mut sink| sink.add("PhaseDomain", (domain[0], domain[1], 0i64))).unwrap();
        }
        let Err(error) = Execution::select(&graph) else {
            panic!("accepted {mode} dispatch domain");
        };
        assert!(error.to_string().contains(expected), "{mode}: {error}");
    }
}
