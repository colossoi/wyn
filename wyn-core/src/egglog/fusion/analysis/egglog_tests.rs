use super::{Egglog, Sink};
use crate::egglog::data::{Array, ExprId, OperationId, RegionId, TypeId};
use crate::egglog::SCHEMA;
use egglog_engine::{EGraph, Write};

#[test]
fn egglog_propagates_only_selected_components_and_keeps_condition_dependencies() {
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, SCHEMA).unwrap();
    graph
        .update(|state| {
            let mut facts = Egglog::new(state)?;
            let independent = facts.independent();
            let scan = facts.scan(OperationId::from(0))?;
            let tuple = facts.tuple(&[independent, scan])?;
            let independent_field = facts.field(tuple, 0)?;
            let scan_field = facts.field(tuple, 1)?;
            let conditional = facts.choice(scan, independent, independent)?;
            facts.collective_dependency(OperationId::from(1), independent_field)?;
            facts.collective_dependency(OperationId::from(2), scan_field)?;
            facts.collective_dependency(OperationId::from(3), conditional)
        })
        .unwrap();
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

#[test]
fn batched_sets_merge_and_rebuild_after_unions_without_changing_numeric_members() {
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, SCHEMA).unwrap();
    graph
        .update(|state| {
            let mut facts = Egglog::new(state)?;
            let a = OperationId::from(0);
            let b = OperationId::from(1);
            let c = OperationId::from(2);
            facts.scalar_region(RegionId::from(0), &[a, b])?;
            facts.length(a, b, TypeId::from(17), true)?;
            facts.length(a, c, TypeId::from(23), true)?;
            facts.stream(a, b, &[])?;
            facts.stream(a, b, &[(ExprId::from(4), ExprId::from(9))])?;
            facts.domain(a, &Array::Zip(vec![Array::Value(ExprId::from(17))]), None)?;
            facts.domain(b, &Array::Zip(vec![Array::Value(ExprId::from(23))]), None)?;
            facts.domain(c, &Array::Literal(vec![ExprId::from(17), ExprId::from(23)]), None)?;
            let left = facts.domain_key(&Array::Value(ExprId::from(17)))?;
            let right = facts.domain_key(&Array::Value(ExprId::from(23)))?;
            facts.state.union(left, right)?;
            let left = facts.operation_key(a)?;
            let right = facts.operation_key(b)?;
            facts.state.union(left, right)
        })
        .unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (check (= (set-length (ScalarRegionMembers (RegionId 0))) 1))
        (check (= (set-length (ValidLengths (Group (OperationId 0)))) 2))
        (check (= (LengthTypes (Group (OperationId 0))) (set-of 17 23)))
        (check (= (set-length (StreamSlices (Group (OperationId 0)) (Group (OperationId 1)))) 2))
        (check (= (SourceDomain (OperationId 0)) (SourceDomain (OperationId 1))))
        (check (= (SourceDomain (OperationId 2)) (DomainLiteral (vec-of 17 23))))
    "#,
        )
        .unwrap();
}
