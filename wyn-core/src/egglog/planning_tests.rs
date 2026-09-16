use super::*;
use crate::{compile_thru_tlc, tlc};

fn graph(facts: &str) -> EGraph {
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, include_str!("ids.egg")).unwrap();
    graph.parse_and_run_program(None, KEYS).unwrap();
    graph.parse_and_run_program(None, RULES).unwrap();
    graph.parse_and_run_program(None, "(CounterType (TypeId 999))").unwrap();
    graph.parse_and_run_program(None, facts).unwrap();
    graph.parse_and_run_program(None, RUN).unwrap();
    graph
}

fn check(graph: &mut EGraph, facts: &str) {
    graph.parse_and_run_program(None, &format!("(check {facts})")).unwrap();
}

fn count(graph: &EGraph, relation: &str) -> usize {
    graph.function_to_dag(relation, usize::MAX, false).unwrap().0.len()
}

const MAP: &str = r#"
    (HostRoot 0 (RegionId 0))
    (Site (OperationId 0) (RegionId 0))
    (CollectiveShape (OperationId 0) 0 0 true)
    (InputDomain (OperationId 0) (Length (ExprId 0)))
    (ArrayResult (OperationId 0) 0 (TypeId 0))
    (ParameterValue (ExprId 0) (RegionId 0))
    (Operand (OperationId 0) "input" (ExprId 0))
    (ResultTuple (ExprId 1) (OperationId 0))
    (Projection (ExprId 2) (ExprId 1) 0)
"#;

#[test]
fn map_to_reduce_derives_materialization_scratch_and_order() {
    let mut g = graph(&format!(
        r#"{MAP}
        (Site (OperationId 1) (RegionId 0))
        (CollectiveShape (OperationId 1) 0 1 true)
        (TotalResult (OperationId 1) 0 (TypeId 0))
        (Accumulator (OperationId 1) 0 (TypeId 0))
        (InputDomain (OperationId 1) (Length (ExprId 2)))
        (Operand (OperationId 1) "input" (ExprId 2))
        (DirectResult (ExprId 3) (OperationId 1) 0)
        (ExitValue (RegionId 0) 0 (ExprId 3))
    "#
    ));
    check(
        &mut g,
        r#"
        (Allocation (Result (OperationId 0) 0) (Length (ExprId 0)))
        (Allocation (Temporary (OperationId 1) "partial" 0) (ChunkCount (Length (ExprId 2)) 64))
        (Allocation (Result (OperationId 1) 0) (Fixed 1))
        (Before (Stage (OperationId 0) "elements") (Stage (OperationId 1) "chunks"))
        (Before (Stage (OperationId 1) "chunks") (Stage (OperationId 1) "combine"))
    "#,
    );
    assert_eq!(count(&g, "Before"), 2, "do not store transitive ordering edges");
    assert_eq!(count(&g, "Allocation"), 3);
}

#[test]
fn shared_producer_has_one_backing_and_independent_consumers() {
    let mut facts = MAP.to_string();
    for op in [1, 2] {
        writeln!(
            facts,
            r#"
            (Site (OperationId {op}) (RegionId 0))
            (CollectiveShape (OperationId {op}) 0 0 true)
            (InputDomain (OperationId {op}) (Length (ExprId 2)))
            (ArrayResult (OperationId {op}) 0 (TypeId 0))
            (Operand (OperationId {op}) "input" (ExprId 2))
        "#
        )
        .unwrap();
    }
    let mut g = graph(&facts);
    check(
        &mut g,
        r#"
        (Backing (Result (OperationId 0) 0) (Result (OperationId 0) 0))
        (Access (Stage (OperationId 1) "elements") (Result (OperationId 0) 0) "read")
        (Access (Stage (OperationId 2) "elements") (Result (OperationId 0) 0) "read")
    "#,
    );
    assert_eq!(
        count(&g, "Allocation"),
        1,
        "unused consumer results require no storage"
    );
    assert_eq!(count(&g, "Before"), 2);
}

#[test]
fn compacted_capacity_and_live_count_are_distinct() {
    let mut g = graph(
        r#"
        (HostRoot 0 (RegionId 0)) (Site (OperationId 0) (RegionId 0))
        (FilterShape (OperationId 0) true) (FilterResult (OperationId 0) (TypeId 0))
        (InputDomain (OperationId 0) (Length (ExprId 0)))
        (DirectResult (ExprId 1) (OperationId 0) 0)
        (ExitValue (RegionId 0) 0 (ExprId 1))
    "#,
    );
    check(
        &mut g,
        r#"
        (Capacity (Result (OperationId 0) 0) (Length (ExprId 0)))
        (LiveLength (Result (OperationId 0) 0) (Stored (Result (OperationId 0) 1)))
        (Allocation (Result (OperationId 0) 1) (Fixed 1))
        (Produces (Stage (OperationId 0) "offsets") (Result (OperationId 0) 1))
        (Produces (Stage (OperationId 0) "compact") (Result (OperationId 0) 0))
    "#,
    );
    assert_eq!(count(&g, "Phase"), 4);
    assert_eq!(count(&g, "Allocation"), 6);
}

#[test]
fn sliced_in_place_update_keeps_value_versions_and_shared_backing() {
    let mut g = graph(&format!(
        r#"{MAP}
        (SliceView (ExprId 3) (ExprId 2) (ExprId 10) (ExprId 11))
        (Site (OperationId 1) (RegionId 0)) (IndexedWrite (OperationId 1))
        (UpdatedResult (OperationId 1) 0 (ExprId 3))
        (Operand (OperationId 1) "environment" (ExprId 3))
        (DirectResult (ExprId 4) (OperationId 1) 0)
        (ExitValue (RegionId 0) 0 (ExprId 4))
        (Site (OperationId 2) (RegionId 0))
        (CollectiveShape (OperationId 2) 0 0 true)
        (InputDomain (OperationId 2) (Length (ExprId 3)))
        (Operand (OperationId 2) "input" (ExprId 3))
        (EffectInput 0 (OperationId 2)) (EffectWait (OperationId 1) 0)
    "#
    ));
    check(
        &mut g,
        r#"
        (Backing (Result (OperationId 1) 0) (Result (OperationId 0) 0))
        (Access (Stage (OperationId 1) "ordered") (Result (OperationId 0) 0) "write")
        (Access (Stage (OperationId 2) "elements") (Result (OperationId 0) 0) "read")
        (Before (Stage (OperationId 2) "elements") (Stage (OperationId 1) "ordered"))
        (OrderEdge (End (OperationId 2)) (Gate 0))
        (OrderEdge (Gate 0) (Start (OperationId 1)))
    "#,
    );
    assert_eq!(count(&g, "Allocation"), 1);
}

#[test]
fn independent_output_domains_stay_independent() {
    let mut g = graph(&format!(
        r#"{MAP}
        (ExitValue (RegionId 0) 0 (ExprId 2))
        (Site (OperationId 1) (RegionId 0))
        (CollectiveShape (OperationId 1) 0 0 true)
        (InputDomain (OperationId 1) (Fixed 137))
        (ArrayResult (OperationId 1) 0 (TypeId 0))
        (DirectResult (ExprId 3) (OperationId 1) 0)
        (ExitValue (RegionId 0) 1 (ExprId 3))
    "#
    ));
    check(
        &mut g,
        r#"
        (PhaseDomain (Stage (OperationId 0) "elements") (Length (ExprId 0)) 64)
        (PhaseDomain (Stage (OperationId 1) "elements") (Fixed 137) 64)
    "#,
    );
    assert_eq!(count(&g, "Before"), 0);
    assert_eq!(count(&g, "Allocation"), 2);
}

#[test]
fn repeated_launch_sites_retain_scope_and_cross_region_handoffs() {
    let mut g = graph(&format!(
        r#"{MAP}
        (Site (OperationId 1) (RegionId 0)) (ScalarSite (OperationId 1))
        (Enters (OperationId 1) (RegionId 1)) (Enters (OperationId 1) (RegionId 2))
        (Repeated (OperationId 1) (RegionId 1) (RegionId 2))
        (Site (OperationId 2) (RegionId 2))
        (CollectiveShape (OperationId 2) 0 0 true)
        (InputDomain (OperationId 2) (Length (ExprId 2)))
        (Operand (OperationId 2) "input" (ExprId 2))
        (Site (OperationId 3) (RegionId 3))
        (CollectiveShape (OperationId 3) 0 0 true)
        (InputDomain (OperationId 3) (Fixed 4))
    "#
    ));
    check(
        &mut g,
        r#"
        (Phase (Stage (OperationId 2) "elements") (RegionId 2))
        (HostHandoff (RegionId 0) (RegionId 2) (Result (OperationId 0) 0))
    "#,
    );
    assert_eq!(count(&g, "Phase"), 2, "a device callback is not a host launch");
    assert_eq!(
        count(&g, "Before"),
        0,
        "host control owns cross-region sequencing"
    );
}

#[test]
fn imported_source_plans_before_block_generation() {
    for source in [
        "entry main(xs: []i32) i32 = reduce(|a: i32, b: i32| a + b, 0, map(|x: i32| x * 2, xs))",
        "entry main(xs: []i32) ?k. [k]i32 = filter(|x: i32| x > 0, xs)",
        "entry main(xs: [4]i32, n: i32) [4]i32 = loop acc = xs for k < n do map(|x: i32| x + k, acc)",
    ] {
        let tlc = tlc::infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
        let mut converted = super::super::convert_program(&tlc).unwrap();
        let summary = snapshot::analyze(&converted.data);
        let (_, g) = analyze(&mut converted.data, &summary).unwrap();
        assert!(count(&g, "Phase") > 0);
        assert!(count(&g, "Allocation") > 0);
        assert!(converted.data.blocks.is_empty());
        assert!(converted.data.buffers.is_empty());
        read(&g, &mut converted.data).unwrap();
        assert_eq!(
            converted.data.buffers.values().filter(|b| b.storage == super::super::Storage::Device).count(),
            count(&g, "Allocation"),
            "every physical allocation must have exactly one logical allocation fact"
        );
    }
}

#[test]
fn consumers_wait_for_their_component_writer_not_the_last_recipe_phase() {
    let mut g = graph(
        r#"
        (HostRoot 0 (RegionId 0)) (Site (OperationId 0) (RegionId 0))
        (CollectiveShape (OperationId 0) 0 1 true)
        (InputDomain (OperationId 0) (Fixed 256))
        (TotalResult (OperationId 0) 0 (TypeId 0))
        (Accumulator (OperationId 0) 0 (TypeId 0))
        (ArrayResult (OperationId 0) 1 (TypeId 0))
        (ResultTuple (ExprId 0) (OperationId 0))
        (Projection (ExprId 1) (ExprId 0) 1)
        (Site (OperationId 1) (RegionId 0))
        (CollectiveShape (OperationId 1) 0 0 true)
        (InputDomain (OperationId 1) (Fixed 256))
        (Operand (OperationId 1) "input" (ExprId 1))
        (SourceDependency (OperationId 1) (OperationId 0))
        (Emitted (Stage (OperationId 0) "chunks") 0)
        (Emitted (Stage (OperationId 0) "combine") 1)
        (Emitted (Stage (OperationId 1) "elements") 2)
    "#,
    );
    check(&mut g, "(DispatchDependency 0 1) (DispatchDependency 0 2)");
    assert_eq!(count(&g, "DispatchDependency"), 2);
}

#[test]
fn effects_cross_scalar_sites_and_stop_at_the_next_launch() {
    let mut g = graph(&format!(
        r#"{MAP}
        (Site (OperationId 1) (RegionId 0)) (ScalarSite (OperationId 1))
        (Site (OperationId 2) (RegionId 0))
        (CollectiveShape (OperationId 2) 0 0 true)
        (InputDomain (OperationId 2) (Fixed 4))
        (EffectInput 0 (OperationId 0)) (EffectWait (OperationId 1) 0)
        (EffectInput 1 (OperationId 1)) (EffectWait (OperationId 2) 1)
        (Emitted (Stage (OperationId 0) "elements") 0)
        (Emitted (Stage (OperationId 2) "elements") 1)
    "#
    ));
    check(&mut g, "(DispatchDependency 0 1)");
    assert_eq!(count(&g, "DispatchDependency"), 1);
}

#[test]
fn chain_planning_keeps_linear_fact_counts() {
    for n in [32, 128] {
        let mut facts = String::from("(HostRoot 0 (RegionId 0))\n");
        for i in 0..n {
            writeln!(
                facts,
                r#"
                (Site (OperationId {i}) (RegionId 0))
                (CollectiveShape (OperationId {i}) 0 0 true)
                (InputDomain (OperationId {i}) (Fixed 64))
                (ArrayResult (OperationId {i}) 0 (TypeId 0))
                (DirectResult (ExprId {i}) (OperationId {i}) 0)
                (Emitted (Stage (OperationId {i}) "elements") {i})
            "#
            )
            .unwrap();
            if i > 0 {
                writeln!(facts, "(Operand (OperationId {i}) \"input\" (ExprId {}))", i - 1).unwrap();
            }
        }
        let g = graph(&facts);
        assert_eq!(count(&g, "Before"), n - 1);
        assert_eq!(count(&g, "DispatchDependency"), n - 1);
        assert_eq!(count(&g, "Allocation"), n - 1);
        assert!(count(&g, "OrderEdge") < 4 * n);
    }
}

#[test]
fn scalar_observers_also_require_materialization() {
    let mut g = graph(&format!(
        r#"{MAP}
        (Site (OperationId 1) (RegionId 0)) (ScalarSite (OperationId 1))
        (Operand (OperationId 1) "environment" (ExprId 2))
        (DirectResult (ExprId 3) (OperationId 1) 0)
        (ExitValue (RegionId 0) 0 (ExprId 3))
    "#
    ));
    check(
        &mut g,
        r#"
        (Requires (InOperation (OperationId 1)) (Result (OperationId 0) 0))
        (Allocation (Result (OperationId 0) 0) (Length (ExprId 0)))
        (OrderEdge (At (Stage (OperationId 0) "elements")) (Start (OperationId 1)))
    "#,
    );
    assert_eq!(count(&g, "Allocation"), 1);
}

#[test]
fn old_value_readers_precede_updates_without_a_precomputed_effect_edge() {
    let mut g = graph(&format!(
        r#"{MAP}
        (Site (OperationId 1) (RegionId 0)) (IndexedWrite (OperationId 1))
        (UpdatedResult (OperationId 1) 0 (ExprId 2))
        (Operand (OperationId 1) "environment" (ExprId 2))
        (Site (OperationId 2) (RegionId 0))
        (CollectiveShape (OperationId 2) 0 0 true)
        (InputDomain (OperationId 2) (Fixed 4))
        (Operand (OperationId 2) "input" (ExprId 2))
        (DirectResult (ExprId 4) (OperationId 1) 0)
        (Site (OperationId 3) (RegionId 0))
        (CollectiveShape (OperationId 3) 0 0 true)
        (InputDomain (OperationId 3) (Fixed 4))
        (Operand (OperationId 3) "input" (ExprId 4))
    "#
    ));
    check(
        &mut g,
        r#"
        (Before (Stage (OperationId 2) "elements") (Stage (OperationId 1) "ordered"))
        (Before (Stage (OperationId 1) "ordered") (Stage (OperationId 3) "elements"))
        (Backing (Result (OperationId 1) 0) (Result (OperationId 0) 0))
    "#,
    );
    // Consumers of the new version do not acquire an anti-dependency back to it.
    assert_eq!(count(&g, "Before"), 4);
}

#[test]
fn bucket_counts_and_overflow_are_typed_fresh_resources() {
    let mut g = graph(
        r#"
        (HostRoot 0 (RegionId 0)) (Site (OperationId 0) (RegionId 0))
        (IndexedWrite (OperationId 0))
        (ParameterValue (ExprId 0) (RegionId 0))
        (UpdatedResult (OperationId 0) 0 (ExprId 0))
        (BucketResult (OperationId 0) (ExprId 0))
        (ResultTuple (ExprId 1) (OperationId 0))
        (ExitValue (RegionId 0) 0 (ExprId 1))
    "#,
    );
    check(
        &mut g,
        r#"
        (Backing (Result (OperationId 0) 0) (Source (ExprId 0)))
        (Allocation (Result (OperationId 0) 1) (Length (ExprId 0)))
        (Allocation (Result (OperationId 0) 2) (Fixed 1))
        (ElementType (Result (OperationId 0) 1) (TypeId 999))
        (ElementType (Result (OperationId 0) 2) (TypeId 999))
    "#,
    );
    assert_eq!(count(&g, "Allocation"), 2);
}

#[test]
fn effect_gates_do_not_expand_to_all_operation_pairs_during_planning() {
    let n = 128;
    let mut facts = String::from("(HostRoot 0 (RegionId 0))\n");
    for i in 0..2 * n {
        writeln!(
            facts,
            r#"
            (Site (OperationId {i}) (RegionId 0))
            (CollectiveShape (OperationId {i}) 0 0 true)
            (InputDomain (OperationId {i}) (Fixed 64))
        "#
        )
        .unwrap();
        if i < n {
            writeln!(facts, "(EffectInput 0 (OperationId {i}))").unwrap();
        } else {
            writeln!(facts, "(EffectWait (OperationId {i}) 0)").unwrap();
        }
    }
    let g = graph(&facts);
    assert_eq!(count(&g, "Before"), 0);
    assert_eq!(count(&g, "OrderEdge"), 6 * n);
}

#[test]
fn shared_scalar_dag_is_summarized_once_across_many_stages() {
    let n = 128;
    let mut facts = String::from("(HostRoot 0 (RegionId 0))\n(ParameterValue (ExprId 0) (RegionId 0))\n");
    for i in 1..=n {
        writeln!(facts, "(ChildValue (ExprId {i}) (ExprId {}))", i - 1).unwrap();
        writeln!(
            facts,
            r#"
            (Site (OperationId {i}) (RegionId 0))
            (CollectiveShape (OperationId {i}) 0 0 true)
            (InputDomain (OperationId {i}) (Fixed 64))
            (Operand (OperationId {i}) "environment" (ExprId {n}))
        "#
        )
        .unwrap();
    }
    let g = graph(&facts);
    assert_eq!(count(&g, "References"), n + 1);
    assert_eq!(count(&g, "Demand"), n);
    assert_eq!(count(&g, "Requires"), n);
}

#[test]
fn selecting_a_tuple_field_does_not_materialize_its_siblings() {
    let mut g = graph(&format!(
        r#"{MAP}
        (Site (OperationId 1) (RegionId 0))
        (CollectiveShape (OperationId 1) 0 0 true)
        (InputDomain (OperationId 1) (Fixed 8))
        (ArrayResult (OperationId 1) 0 (TypeId 0))
        (DirectResult (ExprId 3) (OperationId 1) 0)
        (FieldValue (ExprId 4) 0 (ExprId 2)) (ChildValue (ExprId 4) (ExprId 2))
        (FieldValue (ExprId 4) 1 (ExprId 3)) (ChildValue (ExprId 4) (ExprId 3))
        (Projection (ExprId 5) (ExprId 4) 0)
        (ExitValue (RegionId 0) 0 (ExprId 5))
    "#
    ));
    check(
        &mut g,
        "(Allocation (Result (OperationId 0) 0) (Length (ExprId 0)))",
    );
    assert_eq!(count(&g, "Allocation"), 1);
}
