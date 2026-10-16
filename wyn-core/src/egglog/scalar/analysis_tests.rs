use egglog_engine::EGraph;

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
            (let $choice (ExprChoice $ctx $ty $x $loop $write))
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
