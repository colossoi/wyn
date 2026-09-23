use super::super::data::intern_type;
use super::super::{from_tlc, Storage};
use super::{abi, facts, outputs, read, KEYS, RULES, RUN};
use crate::compile_thru_tlc;
use crate::egglog::abi::inputs;
use crate::egglog::dependencies::analyze;
use crate::egglog::{Program, Scheduled};
use crate::tlc::infer_input_slice_bounds;
use crate::types::{Type, TypeName};
use egglog_engine::EGraph;
use std::fmt::Write;

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
    let mut count = 0;
    graph.constructor_enodes(relation, |_| count += 1).unwrap();
    count
}

const MAP: &str = r#"
    (HostRoot 0 (RegionId 0))
    (Site (OperationId 0) (RegionId 0))
    (CollectiveShape (OperationId 0) false false)
    (InputDomain (OperationId 0) (Length (ExprId 0)))
    (ArrayResult (OperationId 0) 0 (TypeId 0))
    (ParameterValue (ExprId 0))
    (Operand (OperationId 0) "input" (ExprId 0))
    (ResultTuple (ExprId 1) (OperationId 0))
    (Projection (ExprId 2) (ExprId 1) 0)
"#;

#[test]
fn scalar_duplication_requires_both_legality_and_affordable_work() {
    for (legal, cheap) in [(false, false), (false, true), (true, false), (true, true)] {
        let mut g = graph(&format!(
            "(Site (OperationId 0) (RegionId 0))
             (ScalarCandidate (OperationId 0))
             (ExecutionSummary (OperationId 0) true {legal} {cheap})"
        ));
        check(
            &mut g,
            &format!("(= (Rematerialize (OperationId 0)) {})", legal && cheap),
        );
        check(
            &mut g,
            &format!(
                "(= (IsScalarGroupCandidate (OperationId 0)) {})",
                !(legal && cheap)
            ),
        );
    }
}

#[test]
fn map_reuse_is_selected_only_after_all_old_value_uses_are_known() {
    for (extra, expected) in [
        ("", "(Reuse (ExprId 0))"),
        ("(Site (OperationId 1) (RegionId 0)) (CollectiveShape (OperationId 1) false false) (InputDomain (OperationId 1) (Length (ExprId 0)))", "(Reuse (ExprId 0))"),
        ("(Operand (OperationId 0) \"environment\" (ExprId 0))", "(Allocate)"),
        ("(Operand (OperationId 0) \"input\" (ExprId 3)) (ParameterValue (ExprId 3))", "(Reuse (ExprId 0))"),
        ("(Operand (OperationId 0) \"input\" (ExprId 3)) (SliceView (ExprId 3) (ExprId 0) (ExprId 4) (ExprId 5))", "(Allocate)"),
        ("(Operand (OperationId 0) \"input\" (ExprId 3)) (DirectResult (ExprId 3) (OperationId 1) 0) (UpdatedResult (OperationId 1) 0 (ExprId 0))", "(Allocate)"),
        ("(ExitValue (RegionId 0) (ExprId 0))", "(Allocate)"),
        ("(AbiOutputBinding 0 0 7) (ReturnArray 0 (ExprId 2))", "(Allocate)"),
        ("(Site (OperationId 1) (RegionId 0)) (CollectiveShape (OperationId 1) false false) (InputDomain (OperationId 1) (Fixed 4)) (Operand (OperationId 1) \"input\" (ExprId 0))", "(Allocate)"),
    ] {
        let mut g = graph(&format!("{MAP} {extra} (ReusePermission (OperationId 0) 0 (ExprId 0)) (ExitValue (RegionId 0) (ExprId 2))"));
        check(&mut g, &format!("(= (StorageFor (Result (OperationId 0) 0)) {expected})"));
        assert_eq!(count(&g, "Before"), 0, "reuse must not serialize independent readers");
        assert_eq!(count(&g, "Allocation"), usize::from(expected == "(Allocate)"));
    }
}

#[test]
fn fused_outputs_choose_one_eligible_owner_of_the_input() {
    for (extra, first, second) in [
        ("", "(Reuse (ExprId 0))", "(Allocate)"),
        (
            "(AbiOutputBinding 0 0 7) (ReturnArray 0 (ExprId 2))",
            "(Allocate)",
            "(Reuse (ExprId 0))",
        ),
    ] {
        let mut g = graph(&format!(
            r#"{MAP} {extra}
        (ArrayResult (OperationId 0) 1 (TypeId 0))
        (ReusePermission (OperationId 0) 1 (ExprId 0))
        (ReusePermission (OperationId 0) 0 (ExprId 0))
        (Materialize (Result (OperationId 0) 0))
        (Materialize (Result (OperationId 0) 1))"#
        ));
        check(&mut g, &format!("(= (StorageFor (Result (OperationId 0) 0)) {first}) (= (StorageFor (Result (OperationId 0) 1)) {second})"));
        assert_eq!(count(&g, "Allocation"), 1);
    }
}

#[test]
fn reuse_accepts_only_existing_proofs_that_other_readers_have_finished() {
    let reader = r#"
        (Site (OperationId 1) (RegionId 0))
        (CollectiveShape (OperationId 1) false false)
        (InputDomain (OperationId 1) (Length (ExprId 0)))
        (Operand (OperationId 1) "input" (ExprId 0))"#;
    let collective = r#"
        (Site (OperationId 1) (RegionId 0))
        (CollectiveShape (OperationId 1) true false)
        (InputDomain (OperationId 1) (Length (ExprId 0)))
        (Operand (OperationId 1) "input" (ExprId 0))"#;
    let scalar = r#"
        (Site (OperationId 1) (RegionId 0)) (ScalarSite (OperationId 1))
        (Operand (OperationId 1) "environment" (ExprId 0))"#;
    for (reads, order, allowed) in [
        (reader, "", false),
        (
            reader,
            "(Before (Stage (OperationId 0) \"elements\") (Stage (OperationId 1) \"elements\"))",
            false,
        ),
        (
            reader,
            "(Before (Stage (OperationId 1) \"elements\") (Stage (OperationId 0) \"elements\"))",
            true,
        ),
        (
            reader,
            "(EffectInput 1 (OperationId 1)) (EffectWait (OperationId 0) 1)",
            true,
        ),
        (
            collective,
            "(Before (Stage (OperationId 1) \"offsets\") (Stage (OperationId 0) \"elements\"))",
            true,
        ),
        (scalar, "", false),
        (
            scalar,
            "(EffectInput 1 (OperationId 1)) (EffectWait (OperationId 0) 1)",
            true,
        ),
    ] {
        let facts = format!("{MAP} {reads} {order} (ExitValue (RegionId 0) (ExprId 2))");
        let baseline = graph(&facts);
        let mut g = graph(&format!("{facts} (ReusePermission (OperationId 0) 0 (ExprId 0))"));
        let expected = if allowed { "(Reuse (ExprId 0))" } else { "(Allocate)" };
        check(
            &mut g,
            &format!("(= (StorageFor (Result (OperationId 0) 0)) {expected})"),
        );
        assert_eq!(count(&g, "Before"), count(&baseline, "Before"));
        assert_eq!(
            count(&g, "DispatchDependency"),
            count(&baseline, "DispatchDependency")
        );
    }
}

#[test]
fn owned_intermediate_reuse_preserves_live_values_layout_and_invocation_scope() {
    for (extra, ty, region, allowed) in [
        ("", 0, 0, true),
        ("(ExitValue (RegionId 0) (ExprId 2))", 0, 0, false),
        (
            "(Operand (OperationId 1) \"environment\" (ExprId 2))",
            0,
            0,
            false,
        ),
        ("", 1, 0, false),
        ("(HostRoot 1 (RegionId 1))", 0, 1, false),
    ] {
        let mut g = graph(&format!(
            r#"{MAP} {extra}
            (Site (OperationId 1) (RegionId {region}))
            (CollectiveShape (OperationId 1) false false)
            (InputDomain (OperationId 1) (Length (ExprId 2)))
            (Operand (OperationId 1) "input" (ExprId 2))
            (ArrayResult (OperationId 1) 0 (TypeId {ty}))
            (DirectResult (ExprId 3) (OperationId 1) 0)
            (ExitValue (RegionId {region}) (ExprId 3))"#
        ));
        let expected = if allowed { "(Reuse (ExprId 2))" } else { "(Allocate)" };
        check(
            &mut g,
            &format!("(= (StorageFor (Result (OperationId 1) 0)) {expected})"),
        );
        assert_eq!(count(&g, "Allocation"), if allowed { 1 } else { 2 });
    }
}

#[test]
fn owned_reuse_selects_an_eligible_input_after_checking_each_candidate() {
    for (extra, chosen) in [("", 2), ("(ExitValue (RegionId 0) (ExprId 2))", 3)] {
        let mut g = graph(&format!(
            r#"{MAP} {extra}
            (Site (OperationId 1) (RegionId 0)) (CollectiveShape (OperationId 1) false false)
            (InputDomain (OperationId 1) (Length (ExprId 0)))
            (Operand (OperationId 1) "input" (ExprId 0))
            (ArrayResult (OperationId 1) 0 (TypeId 0)) (DirectResult (ExprId 3) (OperationId 1) 0)
            (Site (OperationId 2) (RegionId 0)) (CollectiveShape (OperationId 2) false false)
            (InputDomain (OperationId 2) (Length (ExprId 2)))
            (Operand (OperationId 2) "input" (ExprId 2)) (Operand (OperationId 2) "input" (ExprId 3))
            (ArrayResult (OperationId 2) 0 (TypeId 0)) (DirectResult (ExprId 4) (OperationId 2) 0)
            (ArrayResult (OperationId 2) 1 (TypeId 0)) (Materialize (Result (OperationId 2) 1))
            (ExitValue (RegionId 0) (ExprId 4))"#
        ));
        check(
            &mut g,
            &format!("(= (StorageFor (Result (OperationId 2) 0)) (Reuse (ExprId {chosen})))"),
        );
        assert_eq!(count(&g, "Allocation"), 3);
        assert_eq!(
            count(&g, "ReuseCandidate"),
            2,
            "inspect inputs once across output slots"
        );
        assert_eq!(count(&g, "DispatchDependency"), 2);
    }
}

#[test]
fn scan_reuse_uses_recipe_boundaries_without_ordering_independent_readers() {
    let scan = MAP.replace(
        "(CollectiveShape (OperationId 0) false false)",
        "(CollectiveShape (OperationId 0) true false)",
    );
    let reader = r#"
        (Site (OperationId 1) (RegionId 0)) (CollectiveShape (OperationId 1) false false)
        (InputDomain (OperationId 1) (Length (ExprId 0)))
        (Operand (OperationId 1) "input" (ExprId 0))"#;
    for (reads, order, allowed) in [
        ("", "", true),
        ("(ExitValue (RegionId 0) (ExprId 0))", "", false),
        ("(Operand (OperationId 0) \"environment\" (ExprId 0))", "", false),
        (reader, "", false),
        (
            reader,
            "(Before (Stage (OperationId 1) \"elements\") (Stage (OperationId 0) \"chunks\"))",
            true,
        ),
        (
            reader,
            "(Before (Stage (OperationId 1) \"elements\") (Stage (OperationId 0) \"combine\"))",
            true,
        ),
        (
            reader,
            "(Before (Stage (OperationId 0) \"offsets\") (Stage (OperationId 1) \"elements\"))",
            false,
        ),
        (
            reader,
            "(EffectInput 1 (OperationId 1)) (EffectWait (OperationId 0) 1)",
            true,
        ),
    ] {
        let facts = format!(
            r#"{scan} {reads} {order}
            (Accumulator (OperationId 0) 0 (TypeId 0))
            (ScanComponent (OperationId 0) 0 (TypeId 0))
            (ExitValue (RegionId 0) (ExprId 2))"#
        );
        let baseline = graph(&facts);
        let mut g = graph(&format!("{facts} (ReusePermission (OperationId 0) 0 (ExprId 0))"));
        let expected = if allowed { "(Reuse (ExprId 0))" } else { "(Allocate)" };
        check(
            &mut g,
            &format!("(= (StorageFor (Result (OperationId 0) 0)) {expected})"),
        );
        check(
            &mut g,
            r#"(= (ReadFinished (Stage (OperationId 0) "offsets") (InStage (Stage (OperationId 0) "chunks"))) true)"#,
        );
        assert_eq!(count(&g, "Allocation"), if allowed { 3 } else { 4 });
        assert_eq!(count(&g, "Before"), count(&baseline, "Before"));
        assert_eq!(
            count(&g, "DispatchDependency"),
            count(&baseline, "DispatchDependency")
        );
        assert_eq!(
            count(&g, "ReuseBoundary"),
            3,
            "only this recipe's phases are visited"
        );
    }
}

#[test]
fn scan_can_reuse_an_owned_intermediate_without_a_permission_hint() {
    let mut g = graph(&format!(
        r#"{MAP}
        (Site (OperationId 1) (RegionId 0)) (CollectiveShape (OperationId 1) true false)
        (InputDomain (OperationId 1) (Length (ExprId 2)))
        (Operand (OperationId 1) "input" (ExprId 2))
        (Accumulator (OperationId 1) 0 (TypeId 0)) (ScanComponent (OperationId 1) 0 (TypeId 0))
        (ArrayResult (OperationId 1) 0 (TypeId 0)) (DirectResult (ExprId 3) (OperationId 1) 0)
        (ExitValue (RegionId 0) (ExprId 3))"#
    ));
    check(
        &mut g,
        r#"
        (= (StorageFor (Result (OperationId 1) 0)) (Reuse (ExprId 2)))
        (= (Backing (Result (OperationId 1) 0)) (Result (OperationId 0) 0))
        (= (& (Access (Stage (OperationId 1) "chunks") (Result (OperationId 0) 0)) 1) 1)
        (= (& (Access (Stage (OperationId 1) "offsets") (Result (OperationId 0) 0)) 2) 2)"#,
    );
    assert_eq!(count(&g, "Allocation"), 4);
    // The prefix scratch also connects chunks directly to offsets.
    assert_eq!(count(&g, "DispatchDependency"), 4);
}

#[test]
fn map_to_reduce_derives_materialization_scratch_and_order() {
    let mut g = graph(&format!(
        r#"{MAP}
        (Site (OperationId 1) (RegionId 0))
        (CollectiveShape (OperationId 1) false true)
        (TotalResult (OperationId 1) 0 (TypeId 0))
        (Accumulator (OperationId 1) 0 (TypeId 0))
        (InputDomain (OperationId 1) (Length (ExprId 2)))
        (Operand (OperationId 1) "input" (ExprId 2))
        (DirectResult (ExprId 3) (OperationId 1) 0)
        (ExitValue (RegionId 0) (ExprId 3))
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
            (CollectiveShape (OperationId {op}) false false)
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
        (= (Backing (Result (OperationId 0) 0)) (Result (OperationId 0) 0))
        (= (& (Access (Stage (OperationId 1) "elements") (Result (OperationId 0) 0)) 1) 1)
        (= (& (Access (Stage (OperationId 2) "elements") (Result (OperationId 0) 0)) 1) 1)
    "#,
    );
    assert_eq!(
        count(&g, "Allocation"),
        1,
        "unused consumer results require no storage"
    );
    assert_eq!(count(&g, "Before"), 2);
    // Live sibling results must stay independent; neither reader may clobber
    // the shared producer while the other dispatch is still running.
    for op in [1, 2] {
        writeln!(
            facts,
            "(DirectResult (ExprId {}) (OperationId {op}) 0) (ExitValue (RegionId 0) (ExprId {}))",
            op + 2,
            op + 2
        )
        .unwrap();
    }
    let mut g = graph(&facts);
    for op in [1, 2] {
        check(
            &mut g,
            &format!("(= (StorageFor (Result (OperationId {op}) 0)) (Allocate))"),
        );
    }
    assert_eq!(count(&g, "Allocation"), 3);
    assert_eq!(count(&g, "DispatchDependency"), 2);
}

#[test]
fn compacted_capacity_and_live_count_are_distinct() {
    let mut g = graph(
        r#"
        (HostRoot 0 (RegionId 0)) (Site (OperationId 0) (RegionId 0))
        (FilterShape (OperationId 0)) (FilterResult (OperationId 0) (TypeId 0))
        (InputDomain (OperationId 0) (Length (ExprId 0)))
        (DirectResult (ExprId 1) (OperationId 0) 0)
        (ExitValue (RegionId 0) (ExprId 1))
    "#,
    );
    check(
        &mut g,
        r#"
        (CapacityExtent (Result (OperationId 0) 0) (Length (ExprId 0)))
        (LiveLength (Result (OperationId 0) 0) (Stored (Result (OperationId 0) 1)))
        (Allocation (Result (OperationId 0) 1) (Fixed 1))
        (Produces (Stage (OperationId 0) "compact") (Result (OperationId 0) 1))
        (Produces (Stage (OperationId 0) "compact") (Result (OperationId 0) 0))
        (PhaseDomain (Stage (OperationId 0) "compact") (Fixed 64) 64)
    "#,
    );
    assert_eq!(count(&g, "Phase"), 1);
    assert_eq!(count(&g, "Allocation"), 2);
    assert_eq!(count(&g, "Scratch"), 0);
    g.parse_and_run_program(
        None,
        r#"
        (TypeStride (TypeId 0) 4)
        (AbiArrayLength (AbiExpr 0) (AbiNumber 64))
        (AbiExtent (Length (ExprId 1)))
        (run-schedule (seq (saturate (run abi)) (run abi-final)))
    "#,
    )
    .unwrap();
    check(
        &mut g,
        r#"
        (= (BufferCapacity (Result (OperationId 0) 0)) (FixedCapacity 256))
        (AbiArrayLength (AbiExpr 1) (AbiExtent (Stored (Result (OperationId 0) 1))))
        (= (AbiBoundKnown (AbiExtent (Length (ExprId 1)))) true)
    "#,
    );
    g.parse_and_run_program(
        None,
        "(fail (check (= (AbiConstant (AbiExtent (Length (ExprId 1)))) 64)))",
    )
    .unwrap();
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
        (ExitValue (RegionId 0) (ExprId 4))
        (Site (OperationId 2) (RegionId 0))
        (CollectiveShape (OperationId 2) false false)
        (InputDomain (OperationId 2) (Length (ExprId 3)))
        (Operand (OperationId 2) "input" (ExprId 3))
        (EffectInput 0 (OperationId 2)) (EffectWait (OperationId 1) 0)
    "#
    ));
    check(
        &mut g,
        r#"
        (= (Backing (Result (OperationId 1) 0)) (Result (OperationId 0) 0))
        (= (Access (Stage (OperationId 1) "ordered") (Result (OperationId 0) 0)) 3)
        (= (& (Access (Stage (OperationId 2) "elements") (Result (OperationId 0) 0)) 1) 1)
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
        (ExitValue (RegionId 0) (ExprId 2))
        (Site (OperationId 1) (RegionId 0))
        (CollectiveShape (OperationId 1) false false)
        (InputDomain (OperationId 1) (Fixed 137))
        (ArrayResult (OperationId 1) 0 (TypeId 0))
        (DirectResult (ExprId 3) (OperationId 1) 0)
        (ExitValue (RegionId 0) (ExprId 3))
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
fn repeated_launch_sites_keep_cross_region_ordering_in_host_control() {
    let mut g = graph(&format!(
        r#"{MAP}
        (Site (OperationId 1) (RegionId 0)) (ScalarSite (OperationId 1))
        (Enters (OperationId 1) (RegionId 1)) (Enters (OperationId 1) (RegionId 2))
        (Site (OperationId 2) (RegionId 2))
        (CollectiveShape (OperationId 2) false false)
        (InputDomain (OperationId 2) (Length (ExprId 2)))
        (Operand (OperationId 2) "input" (ExprId 2))
        (Site (OperationId 3) (RegionId 3))
        (CollectiveShape (OperationId 3) false false)
        (InputDomain (OperationId 3) (Fixed 4))
    "#
    ));
    check(
        &mut g,
        r#"
        (Phase (Stage (OperationId 2) "elements") (RegionId 2))
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
        let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
        let converted = from_tlc(&tlc).unwrap();
        let mut converted = Program {
            ir: converted.ir,
            state: Scheduled::default(),
        };
        let summary = analyze(&converted.ir);
        let count_type = intern_type(&mut converted.ir, Type::Constructed(TypeName::UInt(32), vec![]));
        converted.state.abi.inputs = inputs(
            &converted.entries,
            &converted.entry_params,
            &converted.input_bounds,
            &converted.symbols,
            &converted.regions,
            &converted.definitions,
            &converted.types,
            &converted.parameters,
        )
        .unwrap();
        let mut uniforms = vec![];
        let mut g = EGraph::default();
        g.parse_and_run_program(None, include_str!("ids.egg")).unwrap();
        g.parse_and_run_program(None, KEYS).unwrap();
        g.parse_and_run_program(None, RULES).unwrap();
        g.update(|mut sink| {
            let mut imported = outputs(&mut converted, &mut sink)?;
            facts(
                &converted,
                &summary,
                count_type,
                crate::PipelineTopologyPolicy::AllowGenerated,
                &mut imported,
                &mut sink,
            )?;
            super::super::execution::order_facts(&summary.schedules(&converted).unwrap(), &mut sink)?;
            abi::facts(
                &converted.state.abi.inputs,
                &converted.state.outputs,
                &converted.entries,
                &converted.types,
                &imported.types,
                &mut sink,
                &mut uniforms,
            )
        })
        .unwrap();
        g.parse_and_run_program(None, RUN).unwrap();
        assert!(count(&g, "Phase") > 0);
        assert!(count(&g, "Allocation") > 0);
        assert!(count(&g, "AbiRoot") > 0);
        assert!(count(&g, "AbiBufferBinding") > 0);
        let mut capacities = 0;
        g.function_entries("BufferCapacity", |_| capacities += 1).unwrap();
        assert!(capacities > 0);
        assert_eq!(count(&g, "AbiInvalidHost"), 0);
        assert!(converted.state.blocks.is_empty());
        assert!(converted.state.buffers.is_empty());
        let plan = read(&g, &mut converted).unwrap();
        assert_eq!(plan.stages.len(), count(&g, "Phase"));
        assert_eq!(
            converted.state.dispatches.values().map(|s| s.dependencies.len()).sum::<usize>(),
            count(&g, "DispatchDependency")
        );
        let ids: std::collections::BTreeSet<_> = plan.stages.values().copied().collect();
        assert!(converted
            .state
            .dispatches
            .values()
            .all(|s| s.dependencies.iter().all(|id| ids.contains(id))));
        assert_eq!(converted.state.dispatches.len(), plan.stages.len());
        assert!(converted
            .state
            .dispatches
            .values()
            .all(|s| converted.state.blocks[s.kernel].interface.is_some()));
        assert_eq!(
            plan.buffers.len() + plan.local_slots.len(),
            converted.state.buffers.len()
        );
        assert_eq!(plan.launches.len(), converted.state.dispatches.len());
        assert_eq!(
            converted.state.buffers.values().filter(|b| b.storage == Storage::Device).count(),
            count(&g, "Allocation"),
            "every physical allocation must have exactly one logical allocation fact"
        );
    }
}

#[test]
fn local_allocations_and_binding_aliases_are_resolved_without_backend_ids() {
    let mut g = graph(&format!(
        r#"{MAP}
        (SourceEntry 0 (RegionId 0) true)
        (ExitValue (RegionId 0) (ExprId 2))
        (TypeStride (TypeId 0) 4)
        (AbiArrayLength (AbiExpr 0) (AbiNumber 128))
        (OutputBacking 0 (Result (OperationId 0) 0))
        (AbiOutputBinding 0 0 5)
        (AbiRootNeed (KernelRoot (Stage (OperationId 0) "elements")) (AbiExpr 10))
        (AbiStorage (AbiExpr 10) (InputBinding 0 5) 4)
        (DeviceRegion (RegionId 1))
        (Site (OperationId 1) (RegionId 1))
        (CollectiveShape (OperationId 1) false false)
        (ArrayResult (OperationId 1) 0 (TypeId 0)) (TotalCount (OperationId 1) 0)
        (InputDomain (OperationId 1) (Fixed 0))
    "#
    ));
    check(
        &mut g,
        r#"
        (AbiRoot (KernelRoot (Stage (OperationId 0) "elements")) 0 64 1 1 false)
        (= (RootLaunch (KernelRoot (Stage (OperationId 0) "elements"))) (FixedLaunch 2 1 1))
        (= (BufferCapacity (Result (OperationId 0) 0)) (FixedCapacity 512))
        (= (AbiAccess (KernelRoot (Stage (OperationId 0) "elements")) (InputBinding 0 5)) 3)
        (LocalBuffer (OperationId 1) "output" 0 (TypeId 0) (Fixed 0))
        (AbiLocalLength (OperationId 1) "output" 0 1)
    "#,
    );
    assert_eq!(count(&g, "Phase"), 1);
    assert_eq!(count(&g, "Allocation"), 1);
}

#[test]
fn launch_domains_share_extent_handling_after_chunk_normalization() {
    for (domain, normalized, width, groups) in [
        ("(Fixed 129)", "(Fixed 129)", 64, 3),
        ("(Length (ExprId 0))", "(Length (ExprId 0))", 64, 3),
        ("(Scalar (ExprId 1))", "(Scalar (ExprId 1))", 64, 3),
        (
            "(Stored (Source (ExprId 2)))",
            "(Stored (Source (ExprId 2)))",
            64,
            3,
        ),
        (
            "(Product (Fixed 3) (Fixed 43))",
            "(Product (Fixed 3) (Fixed 43))",
            64,
            3,
        ),
        (
            "(ChunkCount (ChunkCount (Fixed 8193) 4) 8)",
            "(Fixed 8193)",
            2048,
            5,
        ),
    ] {
        let mut g = graph(&format!(
            r#"
            (SourceEntry 0 (RegionId 0) true) (HostRoot 0 (RegionId 0))
            (Site (OperationId 0) (RegionId 0)) (CollectiveShape (OperationId 0) false false)
            (InputDomain (OperationId 0) {domain})
            (AbiArrayLength (AbiExpr 0) (AbiNumber 129))
            (AbiAlias (AbiExpr 1) (AbiNumber 129))
            (AbiBound (AbiExtent (Stored (Source (ExprId 2)))) (AbiNumber 129))
        "#
        ));
        check(
            &mut g,
            &format!(
                r#"
                (AbiGridDomain (Stage (OperationId 0) "elements") (AbiExtent {normalized}) {width})
                (= (RootLaunch (KernelRoot (Stage (OperationId 0) "elements"))) (FixedLaunch {groups} 1 1))
            "#
            ),
        );
        assert_eq!(count(&g, "AbiGridDomain"), 1, "{domain}");
    }

    // A chunk factor that cannot be folded into the launch width must not fall
    // through to the generic extent rule with an unnormalized domain.
    let g = graph(
        r#"
        (SourceEntry 0 (RegionId 0) true) (HostRoot 0 (RegionId 0))
        (Site (OperationId 0) (RegionId 0)) (CollectiveShape (OperationId 0) false false)
        (InputDomain (OperationId 0) (ChunkCount (Fixed 1) 4294967296))
    "#,
    );
    assert_eq!(count(&g, "AbiGridDomain"), 0);
}

#[test]
fn consumers_wait_for_their_component_writer_not_the_last_recipe_phase() {
    let mut g = graph(
        r#"
        (HostRoot 0 (RegionId 0)) (Site (OperationId 0) (RegionId 0))
        (CollectiveShape (OperationId 0) false true)
        (InputDomain (OperationId 0) (Fixed 256))
        (TotalResult (OperationId 0) 0 (TypeId 0))
        (Accumulator (OperationId 0) 0 (TypeId 0))
        (ArrayResult (OperationId 0) 1 (TypeId 0))
        (ResultTuple (ExprId 0) (OperationId 0))
        (Projection (ExprId 1) (ExprId 0) 1)
        (Site (OperationId 1) (RegionId 0))
        (CollectiveShape (OperationId 1) false false)
        (InputDomain (OperationId 1) (Fixed 256))
        (Operand (OperationId 1) "input" (ExprId 1))
        (SourceDependency (OperationId 1) (OperationId 0))
    "#,
    );
    check(
        &mut g,
        r#"(DispatchDependency (Stage (OperationId 0) "chunks") (Stage (OperationId 0) "combine"))
        (DispatchDependency (Stage (OperationId 0) "chunks") (Stage (OperationId 1) "elements"))"#,
    );
    assert_eq!(count(&g, "DispatchDependency"), 2);
}

#[test]
fn effects_cross_scalar_sites_and_stop_at_the_next_launch() {
    let mut g = graph(&format!(
        r#"{MAP}
        (Site (OperationId 1) (RegionId 0)) (ScalarSite (OperationId 1))
        (Site (OperationId 2) (RegionId 0))
        (CollectiveShape (OperationId 2) false false)
        (InputDomain (OperationId 2) (Fixed 4))
        (EffectInput 0 (OperationId 0)) (EffectWait (OperationId 1) 0)
        (EffectInput 1 (OperationId 1)) (EffectWait (OperationId 2) 1)
    "#
    ));
    check(
        &mut g,
        r#"(DispatchDependency (Stage (OperationId 0) "elements") (Stage (OperationId 2) "elements"))"#,
    );
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
                (CollectiveShape (OperationId {i}) false false)
                (InputDomain (OperationId {i}) (Fixed 64))
                (ArrayResult (OperationId {i}) 0 (TypeId 0))
                (DirectResult (ExprId {i}) (OperationId {i}) 0)
            "#
            )
            .unwrap();
            if i > 0 {
                writeln!(facts, "(Operand (OperationId {i}) \"input\" (ExprId {}))", i - 1).unwrap();
            }
        }
        let mut g = graph(&facts);
        assert_eq!(count(&g, "Before"), n - 1);
        assert_eq!(count(&g, "DispatchDependency"), n - 1);
        assert_eq!(count(&g, "Allocation"), 1);
        assert!(count(&g, "ReuseCandidate") < n);
        assert!(count(&g, "ReuseAlias") < n);
        check(
            &mut g,
            &format!(
                "(= (Backing (Result (OperationId {}) 0)) (Result (OperationId 0) 0))",
                n - 2
            ),
        );
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
        (ExitValue (RegionId 0) (ExprId 3))
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
        (CollectiveShape (OperationId 2) false false)
        (InputDomain (OperationId 2) (Fixed 4))
        (Operand (OperationId 2) "input" (ExprId 2))
        (DirectResult (ExprId 4) (OperationId 1) 0)
        (Site (OperationId 3) (RegionId 0))
        (CollectiveShape (OperationId 3) false false)
        (InputDomain (OperationId 3) (Fixed 4))
        (Operand (OperationId 3) "input" (ExprId 4))
    "#
    ));
    check(
        &mut g,
        r#"
        (Before (Stage (OperationId 2) "elements") (Stage (OperationId 1) "ordered"))
        (Before (Stage (OperationId 1) "ordered") (Stage (OperationId 3) "elements"))
        (= (Backing (Result (OperationId 1) 0)) (Result (OperationId 0) 0))
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
        (ParameterValue (ExprId 0))
        (UpdatedResult (OperationId 0) 0 (ExprId 0))
        (BucketResult (OperationId 0) (ExprId 0))
        (ResultTuple (ExprId 1) (OperationId 0))
        (ExitValue (RegionId 0) (ExprId 1))
    "#,
    );
    check(
        &mut g,
        r#"
        (= (Backing (Result (OperationId 0) 0)) (Source (ExprId 0)))
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
            (CollectiveShape (OperationId {i}) false false)
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
    let mut facts = String::from("(HostRoot 0 (RegionId 0))\n(ParameterValue (ExprId 0))\n");
    for i in 1..=n {
        writeln!(facts, "(ChildValue (ExprId {i}) (ExprId {}))", i - 1).unwrap();
        writeln!(
            facts,
            r#"
            (Site (OperationId {i}) (RegionId 0))
            (CollectiveShape (OperationId {i}) false false)
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
        (CollectiveShape (OperationId 1) false false)
        (InputDomain (OperationId 1) (Fixed 8))
        (ArrayResult (OperationId 1) 0 (TypeId 0))
        (DirectResult (ExprId 3) (OperationId 1) 0)
        (FieldValue (ExprId 4) 0 (ExprId 2)) (ChildValue (ExprId 4) (ExprId 2))
        (FieldValue (ExprId 4) 1 (ExprId 3)) (ChildValue (ExprId 4) (ExprId 3))
        (Projection (ExprId 5) (ExprId 4) 0)
        (ExitValue (RegionId 0) (ExprId 5))
    "#
    ));
    check(
        &mut g,
        "(Allocation (Result (OperationId 0) 0) (Length (ExprId 0)))",
    );
    assert_eq!(count(&g, "Allocation"), 1);
}

#[test]
fn tuple_input_import_exposes_fields_without_a_parent_buffer() {
    let tlc = infer_input_slice_bounds(
        compile_thru_tlc(
            "entry main(pair: ([]i32, []i32)) []i32 = let (xs, ys) = pair in map(|x:i32|x+length(ys),xs)",
        )
        .unwrap(),
    );
    let mut converted = Program {
        ir: from_tlc(&tlc).unwrap().ir,
        state: Scheduled::default(),
    };
    let summary = analyze(&converted);
    let count_type = intern_type(&mut converted.ir, Type::Constructed(TypeName::UInt(32), vec![]));
    let mut g = EGraph::default();
    g.parse_and_run_program(None, include_str!("ids.egg")).unwrap();
    g.parse_and_run_program(None, KEYS).unwrap();
    g.parse_and_run_program(None, RULES).unwrap();
    let mut parents = vec![];
    g.update(|mut sink| {
        let mut imported = outputs(&mut converted, &mut sink)?;
        parents.extend(imported.input_tuples.iter().copied());
        facts(
            &converted,
            &summary,
            count_type,
            crate::PipelineTopologyPolicy::AllowGenerated,
            &mut imported,
            &mut sink,
        )
    })
    .unwrap();
    assert!(!parents.is_empty());
    for parent in parents {
        g.parse_and_run_program(
            None,
            &format!("(fail (check (ParameterValue (ExprId {}))))", parent.as_u32()),
        )
        .unwrap();
        for index in 0..2 {
            let field = converted
                .expressions
                .iter()
                .find_map(|(&id, expression)| {
                    matches!(expression.kind, super::ExprKind::Project { tuple, index: field }
                    if tuple == parent && field == index)
                    .then_some(id)
                })
                .unwrap();
            check(&mut g, &format!("(ParameterValue (ExprId {}))", field.as_u32()));
        }
    }
}

#[test]
fn whole_tuple_captures_publish_all_materialized_fields() {
    let mut g = graph(
        r#"
        (ResultTuple (ExprId 0) (OperationId 0))
        (ResultSlot (OperationId 0) 0) (ResultSlot (OperationId 0) 1)
        (Projection (ExprId 1) (ExprId 0) 0)
        (AbiStorage (AbiResource (Result (OperationId 0) 0)) (InputBinding 0 5) 4)
        (AbiStorage (AbiResource (Result (OperationId 0) 1)) (InputBinding 0 6) 4)
        (AbiRootNeed (EntryRoot 100) (AbiExpr 0)) (AbiRootNeed (EntryRoot 101) (AbiExpr 1))
    "#,
    );
    g.parse_and_run_program(None, "(run-schedule (seq (saturate (run abi)) (run abi-final)))").unwrap();
    check(
        &mut g,
        r#"
        (AbiRootStorage (EntryRoot 100) (InputBinding 0 5)) (AbiRootStorage (EntryRoot 100) (InputBinding 0 6))
        (AbiRootStorage (EntryRoot 101) (InputBinding 0 5)) (AbiRootStorage (EntryRoot 101) (InputBinding 0 6))
    "#,
    );
}

#[test]
fn conflicting_capacity_policies_fail_inside_egglog() {
    let mut g = graph("");
    let result = g.parse_and_run_program(
        None,
        r#"
        (set (BufferCapacity (Output 0)) (FixedCapacity 64))
        (set (BufferCapacity (Output 0)) (HostCapacity 4))
    "#,
    );
    assert!(result.is_err());
}

#[test]
fn conflicting_launch_policies_fail_inside_egglog() {
    let mut g = graph("");
    let result = g.parse_and_run_program(
        None,
        r#"
        (set (RootLaunch (EntryRoot 0)) (FixedLaunch 2 3 4))
        (set (RootLaunch (EntryRoot 0)) (ParameterLaunch 0 64))
    "#,
    );
    assert!(result.is_err());
}

#[test]
fn conflicting_scheduling_recipes_fail_inside_egglog() {
    let mut g = graph("(Site (OperationId 0) (RegionId 0))");
    let result = g.parse_and_run_program(
        None,
        r#"
        (CollectiveShape (OperationId 0) false false)
        (FilterShape (OperationId 0))
        (run schedule)
    "#,
    );
    assert!(result.is_err());
}

#[test]
fn callback_effects_block_parallelism_through_nested_calls() {
    let mut g = graph(&format!(
        r#"{MAP}
        (Callback (OperationId 0) (RegionId 1))
        (Site (OperationId 1) (RegionId 1))
        (Enters (OperationId 1) (RegionId 2))
        (Site (OperationId 2) (RegionId 2))
        (ParallelEffect (OperationId 2))
    "#
    ));
    check(&mut g, "(= (Plan (OperationId 0)) (Serial))");
}
