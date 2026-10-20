use egglog_engine::EGraph;

#[test]
fn expression_schedules_preserve_request_boundaries_and_settled_priorities() {
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, include_str!("evaluation.egg")).unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
            (set (ExprScheduleWeight 0) 0)
            (set (ExprScheduleWeight 1) 1)
            (set (ExprScheduleWeight 2) 1)
            (set (ExprScheduleWeight 3) 1)
            (ExprScheduleOperand 1 0)
            (ExprScheduleOperand 2 0)
            (ExprScheduleOperand 3 1)
            (set (ExprScheduleCondition 1) 0)
            (set (ExprScheduleCondition 2) 0)
            (set (ExprScheduleCondition 3) 1)
            (run-schedule (saturate schedule-rank))

            (set (ExprScheduleReorderable 100) true)
            (set (ExprScheduleReorderable 101) false)
            (set (ExprSchedulePriority 100 1) 0)
            (set (ExprSchedulePriority 100 2) 0)
            (set (ExprSchedulePriority 100 3) 0)
            (set (ExprSchedulePriority 101 1) 0)
            (set (ExprSchedulePriority 101 2) 0)
            (run-schedule (saturate schedule-anchors))
            (check (= (ExprSchedulePriority 100 1) 1))
            (check (= (ExprSchedulePriority 100 3) 2))
            (check (= (ExprScheduleGroup 100 1) (ExprScheduleGroup 100 2)))
            (check (= (ExprScheduleGroup 100 3) 1))
            (check (= (ExprSchedulePriority 101 1) 0))
            (check (= (ExprSchedulePriority 101 2) 0))
            (check (= (ExprScheduleGroup 101 1) 1))
            (check (= (ExprScheduleGroup 101 2) 2))
            (fail (check (= priority (ExprSchedulePriority 101 3))))

            (set (ExprScheduleReorderable 102) true)
            (set (ExprSchedulePriority 102 3) 0)
            (set (ExprSchedulePriority 100 3) 0)
            (run-schedule (saturate schedule-anchors))
            (check (= (ExprSchedulePriority 102 3) 2))
            (check (= (ExprSchedulePriority 100 3) 2))
            (fail (check (= group (ExprScheduleGroup 102 1))))
            "#,
        )
        .unwrap();
}

fn graph() -> EGraph {
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(
            None,
            concat!(
                include_str!("../ids.egg"),
                r#"
                (datatype TypeKey (TypeId i64))
                (datatype FusionPlan (FusionSource OperationKey))
                (function SourceReadOnly (SourceValue) bool :merge (and old new))
                (function SourceRegionReadOnly (RegionKey) bool :merge (and old new))
                "#,
                include_str!("schema.egg"),
                include_str!("analysis.egg"),
                r#"
                (let $ctx (ExprContextFunction (RegionId 0)))
                (let $ty (TypeId 0))
                (let $x (ExprParameter $ctx $ty (RegionId 0) 0))
                (let $zero (ExprLiteral $ctx $ty "0"))
                (let $nil (ExprNil $ctx))
                (ExprContextActive $ctx)
                (set (SourceReadOnly (SourceGlobal 0)) true)
                (set (SourceReadOnly (SourceGlobal 1)) false)
                (set (SourceRegionReadOnly (RegionId 1)) false)
                "#,
            ),
        )
        .unwrap();
    graph
}

#[test]
fn readonly_proofs_do_not_authorize_speculation_or_ignore_argument_effects() {
    graph()
        .parse_and_run_program(
            None,
            r#"
            (let $partial (ExprBinary $ctx $ty "/" $x $zero))
            (let $loop (ExprExecute $ctx $ty (SourceGlobal 0)))
            (let $write (ExprExecute $ctx $ty (SourceGlobal 1)))
            (let $leaf (ExprLeaf $ctx $ty (SourceGlobal 1)))
            (let $args (ExprCons $ctx $write $nil))
            (let $call (ExprCall $ctx $ty (SourceGlobal 0) (RegionId 0) $args))
            (let $invoke (ExprInvoke $ctx $ty (RegionId 1) $nil))
            (let $choice (ExprChoice $ctx $ty $x $loop $write (SourceTerm (RegionId 0) 0)))
            (let $unknown (ExprExecute $ctx $ty (SourceGlobal 99)))
            (let $state (ExprLeaf $ctx $ty (SourceFormal (RegionId 0) 1)))
            (ExprRoot $ctx (RegionId 0) (SourceFormal (RegionId 0) 1) $state)
            (run-schedule (saturate scalar-analysis) (saturate scalar-effects))
            (check (= (ExprReadOnly $partial) true))
            (check (= (ExprReadOnly $loop) true))
            (check (= (ExprReadOnly $state) true))
            (fail (check (ExprTotal $partial)))
            (fail (check (ExprTotal $loop)))
            (check (ExprTotal $leaf))
            (check (= (ExprReadOnly $leaf) false))
            (check (= (ExprArgsReadOnly $args) false))
            (check (= (ExprReadOnly $call) false))
            (check (= (ExprReadOnly $invoke) false))
            (check (= (ExprReadOnly $choice) false))
            (fail (check (= (ExprReadOnly $unknown) true)))
            "#,
        )
        .unwrap();
}

#[test]
fn leaf_materialization_includes_effects_of_other_selected_roots() {
    graph()
        .parse_and_run_program(
            None,
            r#"
            (set (SourceReadOnly (SourceGlobal 2)) true)
            (let $leaf (ExprLeaf $ctx $ty (SourceGlobal 0)))
            (let $other (ExprLeaf $ctx $ty (SourceGlobal 2)))
            (let $write (ExprInstruction $ctx $ty (SourceGlobal 1) "write" $nil))
            (ExprRoot $ctx (RegionId 0) (SourceGlobal 0) $other)
            (ExprRoot $ctx (RegionId 0) (SourceGlobal 2) $write)
            (run-schedule (saturate scalar-analysis) (saturate scalar-effects))
            (check (= (ExprReadOnly $leaf) false))
            "#,
        )
        .unwrap();
}

#[test]
fn eclass_merging_cannot_hide_an_effectful_alternative() {
    graph()
        .parse_and_run_program(
            None,
            r#"
            (let $write (ExprExecute $ctx $ty (SourceGlobal 1)))
            (run-schedule (saturate scalar-analysis) (saturate scalar-effects))
            (union $write $zero)
            (run-schedule (saturate scalar-analysis) (saturate scalar-effects))
            (check (= (ExprReadOnly $zero) false))
            "#,
        )
        .unwrap();
}
