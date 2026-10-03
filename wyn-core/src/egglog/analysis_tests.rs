use egglog_engine::EGraph;

#[test]
fn work_proofs_distinguish_shared_regions_from_repeated_inline_work() {
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(
            None,
            concat!(
                include_str!("ids.egg"),
                "\n(datatype SourceValue (SourceGlobal i64))
        (function SourceWork (SourceValue) i64 :no-merge)
        (function SourceRegionWork (RegionKey) i64 :no-merge)
        (function SourceRegionDuplicable (RegionKey) bool :no-merge)
        (function SourceOperationValue (OperationKey) SourceValue :no-merge)
        (function SourceOperationCheap (OperationKey) bool :merge (or old new))
        (relation SourceSummaryEnters (SourceValue RegionKey))
        (relation SourceRegionEnters (RegionKey RegionKey))
        (relation SourceCallable (SourceValue RegionKey))\n",
            ),
        )
        .unwrap();
    super::work::register(&mut graph).unwrap();
    graph.parse_and_run_program(None, include_str!("analysis/work.egg")).unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (set (SourceOperationValue (OperationId 0)) (SourceGlobal 0))
        (set (SourceWork (SourceGlobal 0)) 1)
        (set (SourceRegionWork (RegionId 0)) 1)
        (set (SourceRegionWork (RegionId 1)) 1)
        (set (SourceRegionWork (RegionId 2)) 1)
        (set (SourceRegionWork (RegionId 3)) 40)
        (SourceSummaryEnters (SourceGlobal 0) (RegionId 0))
        (SourceRegionEnters (RegionId 0) (RegionId 1))
        (SourceRegionEnters (RegionId 0) (RegionId 2))
        (SourceRegionEnters (RegionId 1) (RegionId 3))
        (SourceRegionEnters (RegionId 2) (RegionId 3))
        (SourceCallable (SourceGlobal 0) (RegionId 0))
        (set (SourceRegionDuplicable (RegionId 0)) true)
        (set (SourceOperationValue (OperationId 1)) (SourceGlobal 1))
        (set (SourceWork (SourceGlobal 1)) 1)
        (SourceSummaryEnters (SourceGlobal 1) (RegionId 4))
        (set (SourceRegionWork (RegionId 4)) 1)
        (set (SourceRegionDuplicable (RegionId 4)) true)
        (SourceRegionEnters (RegionId 4) (RegionId 4))
        (SourceCallable (SourceGlobal 1) (RegionId 4))
        (run-schedule (saturate work-summaries) (saturate work-costs) (saturate work-select))
        (check (= (SourceOperationCheap (OperationId 0)) true))
        (check (= (WorkCost (RegionId 0)) 65))
        (fail (check (SourceInlineEligible (RegionId 0))))
        (check (= (SourceOperationCheap (OperationId 1)) false))
        (fail (check (SourceInlineEligible (RegionId 4))))
    "#,
        )
        .unwrap();
}
