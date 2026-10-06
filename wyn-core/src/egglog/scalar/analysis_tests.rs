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
                (let $ctx (ScalarFunction (RegionId 0)))
                (let $ty (TypeId 0))
                (let $x (ScalarParameter $ctx $ty (RegionId 0) 0))
                (let $zero (ScalarLiteral $ctx $ty "0"))
                (let $nil (ScalarNil $ctx))
                (ScalarActive $ctx)
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
            (let $partial (ScalarBinary $ctx $ty "/" $x $zero))
            (let $loop (ScalarExecute $ctx $ty (SourceGlobal 0)))
            (let $write (ScalarExecute $ctx $ty (SourceGlobal 1)))
            (let $leaf (ScalarLeaf $ctx $ty (SourceGlobal 1)))
            (let $args (ScalarCons $ctx $write $nil))
            (let $call (ScalarCall $ctx $ty (SourceGlobal 0) (RegionId 0) $args))
            (let $invoke (ScalarInvoke $ctx $ty (RegionId 1) $nil))
            (let $choice (ScalarChoice $ctx $ty $x $loop $write))
            (let $unknown (ScalarExecute $ctx $ty (SourceGlobal 99)))
            (let $state (ScalarLeaf $ctx $ty (SourceFormal (RegionId 0) 1)))
            (ScalarRoot $ctx (RegionId 0) (SourceFormal (RegionId 0) 1) $state)
            (run-schedule (saturate scalar-analysis) (saturate scalar-effects))
            (check (= (ScalarReadOnly $partial) true))
            (check (= (ScalarReadOnly $loop) true))
            (check (= (ScalarReadOnly $state) true))
            (fail (check (ScalarTotal $partial)))
            (fail (check (ScalarTotal $loop)))
            (check (ScalarTotal $leaf))
            (check (= (ScalarReadOnly $leaf) false))
            (check (= (ScalarArgsReadOnly $args) false))
            (check (= (ScalarReadOnly $call) false))
            (check (= (ScalarReadOnly $invoke) false))
            (check (= (ScalarReadOnly $choice) false))
            (fail (check (= (ScalarReadOnly $unknown) true)))
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
            (let $leaf (ScalarLeaf $ctx $ty (SourceGlobal 0)))
            (let $other (ScalarLeaf $ctx $ty (SourceGlobal 2)))
            (let $write (ScalarInstruction $ctx $ty (SourceGlobal 1) "write" $nil))
            (ScalarRoot $ctx (RegionId 0) (SourceGlobal 0) $other)
            (ScalarRoot $ctx (RegionId 0) (SourceGlobal 2) $write)
            (run-schedule (saturate scalar-analysis) (saturate scalar-effects))
            (check (= (ScalarReadOnly $leaf) false))
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
            (let $write (ScalarExecute $ctx $ty (SourceGlobal 1)))
            (run-schedule (saturate scalar-analysis) (saturate scalar-effects))
            (union $write $zero)
            (run-schedule (saturate scalar-analysis) (saturate scalar-effects))
            (check (= (ScalarReadOnly $zero) false))
            "#,
        )
        .unwrap();
}
