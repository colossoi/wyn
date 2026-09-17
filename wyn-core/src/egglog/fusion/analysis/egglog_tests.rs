use super::{Egglog, Sink};
use crate::egglog::data::OperationId;
use egglog_engine::EGraph;

#[test]
fn egglog_propagates_only_selected_components_and_keeps_condition_dependencies() {
    let mut facts = Egglog::new();
    let independent = facts.independent();
    let scan = facts.scan(OperationId::from(0));
    let tuple = facts.tuple(&[independent, scan]);
    let independent_field = facts.field(tuple, 0);
    let scan_field = facts.field(tuple, 1);
    let conditional = facts.choice(scan, independent, independent);
    facts.collective_dependency(OperationId::from(1), independent_field);
    facts.collective_dependency(OperationId::from(2), scan_field);
    facts.collective_dependency(OperationId::from(3), conditional);
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, &facts.text).unwrap();
    graph.parse_and_run_program(None, include_str!("../fusion.egg")).unwrap();
    graph
        .parse_and_run_program(
            None,
            "
        (run-schedule (saturate (run fusion-dependencies)))
        (fail (check (ScanFeeds (OperationId 0) (OperationId 1))))
        (check (ScanFeeds (OperationId 0) (OperationId 2)))
        (check (ScanFeeds (OperationId 0) (OperationId 3)))
    ",
        )
        .unwrap();
}
