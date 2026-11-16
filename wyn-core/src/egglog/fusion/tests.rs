use crate::compile_thru_tlc;
use crate::egglog::dataflow::{
    edit,
    ir::{self, Code},
};
use crate::egglog::payload::Payload;
use crate::egglog::{analysis, from_tlc, fuse, query::Query, test_graph};
use crate::op::BinaryOperator;

#[test]
fn placement_uses_complete_demand_and_effect_proofs() {
    let facts = r#"
        (LiveOperation (OperationId 1)) (LiveOperation (OperationId 2))
        (LiveOperation (OperationId 3)) (LiveOperation (OperationId 4))
        (LiveRegion (RegionId 0)) (LiveRegion (RegionId 20)) (LiveRegion (RegionId 21))
        (set (ControlRegion (OperationId 1) 0) (RegionId 10))
        (set (ControlRegion (OperationId 2) 0) (RegionId 20))
        (set (ControlRegion (OperationId 2) 1) (RegionId 21))
        (set (OperationRegion (OperationId 1)) (RegionId 0))
        (set (OperationOrder (OperationId 1)) 0)
        (set (OperationRegion (OperationId 2)) (RegionId 0))
        (set (OperationOrder (OperationId 2)) 2)
        (set (OperationRegion (OperationId 3)) (RegionId 21))
        (set (OperationOrder (OperationId 3)) 0)
        (set (OperationRegion (OperationId 4)) (RegionId 0))
        (set (OperationOrder (OperationId 4)) 1)
        (set (Output (OperationId 1) 0) (ValueNameRef (ValueId 1)))
        (DirectOperationUse (OperationId 3) (ValueNameRef (ValueId 1)))
        (set (PureOperation (OperationId 3)) true)
        (NextOperation (OperationId 1) (OperationId 4))
        (NextOperation (OperationId 4) (OperationId 2))
        (FlowScopeAncestor (RegionId 20) (RegionId 20))
        (FlowScopeAncestor (RegionId 21) (RegionId 21))
    "#;
    let template = test_graph::fixture(
        &[
            (1, Code::CountedLoop),
            (2, Code::Branch),
            (3, Code::Use),
            (4, Code::Use),
        ],
        facts,
    )
    .unwrap();
    for (extra, sinks) in [
        ("", true),
        ("(OperationUse (OperationId 2) (ValueNameRef (ValueId 1)))", true),
        ("(DirectOperationUse (OperationId 99) (ValueNameRef (ValueId 1))) (set (OperationRegion (OperationId 99)) (RegionId 99))", true),
        ("(OperationWrite (OperationId 1) (ExternalMemory))", false),
        ("(OperationWrite (OperationId 4) (ExternalMemory))", false),
        ("(ConsumedToken (OperationId 1) (AccumulatorId 1))", false),
        ("(InvocationDependent (OperationId 1))", false),
        ("(EndDemand (RegionId 0) (ValueNameRef (ValueId 1)))", false),
        ("(EndDemand (RegionId 20) (ValueNameRef (ValueId 1)))", false),
        ("(DirectOperationUse (OperationId 2) (ValueNameRef (ValueId 1)))", false),
        ("(OperationUse (OperationId 1) (ValueNameRef (ValueId 9)))", false),
        ("(OperationUse (OperationId 1) (ValueNameRef (ValueId 9))) (AvailableAt (OperationId 2) (ValueNameRef (ValueId 9)))", true),
    ] {
        let mut graph = template.clone();
        graph.parse_and_run_program(None, extra).unwrap();
        graph.parse_and_run_program(None, "(run-schedule (saturate placement-index) (saturate placement-demand dataflow-scopes) (saturate placement-legality) (saturate placement-select))").unwrap();
        assert_eq!(Query(&graph).first("SinkProducer").unwrap().is_some(), sinks, "{extra}");
    }
}

#[test]
fn moving_into_empty_regions_preserves_operation_order() {
    let mut graph = test_graph::fixture(
        &[(1, Code::Use), (2, Code::Use)],
        r#"
        (set (OperationRegion (OperationId 1)) (RegionId 0))
        (set (OperationOrder (OperationId 1)) 0)
        (set (OperationRegion (OperationId 2)) (RegionId 0))
        (set (OperationOrder (OperationId 2)) 1)
        (Region (RegionId 1))
    "#,
    )
    .unwrap();
    let query = Query(&graph);
    let first = query.constructor("OperationId", 1_i64).unwrap().unwrap();
    let second = query.constructor("OperationId", 2_i64).unwrap().unwrap();
    let destination = query.constructor("RegionId", 1_i64).unwrap().unwrap();
    let source = query.constructor("RegionId", 0_i64).unwrap().unwrap();
    edit::move_into(&mut graph, first, destination, None).unwrap();
    edit::move_into(&mut graph, second, destination, None).unwrap();
    assert_eq!(Query(&graph).operations(destination).unwrap(), [first, second]);
    assert!(Query(&graph).operations(source).unwrap().is_empty());
    edit::move_before(&mut graph, second, first).unwrap();
    assert_eq!(Query(&graph).operations(destination).unwrap(), [second, first]);
}

#[test]
fn merging_matching_branches_preserves_certificates_and_path_order() {
    let mut graph = test_graph::fixture(
        &[
            (1, Code::Branch),
            (2, Code::Branch),
            (3, Code::Use),
            (4, Code::Use),
            (5, Code::Use),
            (6, Code::Use),
        ],
        r#"
        (set (ControlRegion (OperationId 1) 0) (RegionId 10))
        (set (ControlRegion (OperationId 1) 1) (RegionId 11))
        (set (ControlRegion (OperationId 2) 0) (RegionId 20))
        (set (ControlRegion (OperationId 2) 1) (RegionId 21))
        (set (OutputCount (OperationId 1)) 1) (set (OutputCount (OperationId 2)) 1)
        (set (Output (OperationId 1) 0) (ValueNameRef (ValueId 1)))
        (set (Output (OperationId 2) 0) (ValueNameRef (ValueId 2)))
        (set (RegionYieldCount (RegionId 10)) 1) (set (RegionYieldCount (RegionId 11)) 1)
        (set (RegionYieldCount (RegionId 20)) 1) (set (RegionYieldCount (RegionId 21)) 1)
        (set (RegionYield (RegionId 10) 0) (ValueNameRef (ValueId 3)))
        (set (RegionYield (RegionId 11) 0) (ValueNameRef (ValueId 4)))
        (set (RegionYield (RegionId 20) 0) (ValueNameRef (ValueId 5)))
        (set (RegionYield (RegionId 21) 0) (ValueNameRef (ValueId 6)))
        (set (RegionReturnType (RegionId 20) 0) (ReturnScalar (ScalarElement (I32))))
        (set (RegionReturnType (RegionId 21) 0) (ReturnScalar (ScalarElement (I32))))
        (set (OperationRegion (OperationId 3)) (RegionId 10)) (set (OperationOrder (OperationId 3)) 4)
        (set (OperationRegion (OperationId 4)) (RegionId 11)) (set (OperationOrder (OperationId 4)) 9)
        (set (OperationRegion (OperationId 5)) (RegionId 20)) (set (OperationOrder (OperationId 5)) 0)
        (set (OperationRegion (OperationId 6)) (RegionId 21)) (set (OperationOrder (OperationId 6)) 0)
        (YieldCertificate (RegionId 20) 0 (ValueId 8))
        (YieldCertificate (RegionId 21) 0 (ValueId 9))
        (OperationCertificate (OperationId 2) (ValueId 7))
        (BranchPair (OperationId 1) (OperationId 2))
        (set (FusionOutputBase (OperationId 1) (OperationId 2)) 1)
    "#,
    )
    .unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (run-schedule (saturate graph-fusion))
        (BranchFuseSelected (OperationId 1) (OperationId 2))
        (run-schedule (run graph-fusion-apply))
        (check (= (OutputCount (OperationId 1)) 2))
        (check (= (Output (OperationId 1) 1) (ValueNameRef (ValueId 2))))
        (check (= (RegionYieldCount (RegionId 10)) 2) (= (RegionYieldCount (RegionId 11)) 2))
        (check (= (RegionYield (RegionId 10) 1) (ValueNameRef (ValueId 5))))
        (check (= (RegionYield (RegionId 11) 1) (ValueNameRef (ValueId 6))))
        (check (= (RegionReturnType (RegionId 10) 1) (ReturnScalar (ScalarElement (I32)))))
        (check (= (RegionReturnType (RegionId 11) 1) (ReturnScalar (ScalarElement (I32)))))
        (check (YieldCertificate (RegionId 10) 1 (ValueId 8)))
        (check (YieldCertificate (RegionId 11) 1 (ValueId 9)))
        (check (OperationCertificate (OperationId 1) (ValueId 7)))
        (check (= (OperationRegion (OperationId 5)) (RegionId 10)) (= (OperationOrder (OperationId 5)) 5))
        (check (= (OperationRegion (OperationId 6)) (RegionId 11)) (= (OperationOrder (OperationId 6)) 10))
    "#,
        )
        .unwrap();
}

#[test]
fn compact_length_uses_a_count_without_a_collection() {
    for (source, has_prefix) in [
        ("entry main(xs:[]i32) i32=length(filter(|x|x>0,xs))", false),
        (
            "entry main(xs:[]i32) ([]i32,i32)=(scan((+),0,xs),length(filter(|x|x>0,xs)))",
            true,
        ),
    ] {
        let source = compile_thru_tlc(source).unwrap();
        let program = fuse(from_tlc(&source).unwrap()).unwrap();
        let query = Query(&program.graph);
        let mut totals = 0;
        let mut packed = 0;
        let mut prefixes = 0;
        query
            .for_function("OutputRole", |_, role| {
                totals += usize::from(query.enode("TotalOutput", role)?.is_some());
                packed += usize::from(query.enode("CompactedOutput", role)?.is_some());
                prefixes += usize::from(query.enode("PrefixOutput", role)?.is_some());
                Ok(())
            })
            .unwrap();
        assert_eq!((totals, packed, prefixes), (1, 0, usize::from(has_prefix)));
        if has_prefix {
            let mut mixed = false;
            query
                .for_each("LiveOperation", |row| {
                    if query.contains("ParallelOperation", row[0])?
                        && analysis::count(query, "OutputCount", row[0])? == 2
                    {
                        mixed = true;
                    }
                    Ok(())
                })
                .unwrap();
            assert!(mixed, "count conversion must preserve the retained scan output");
        }
    }
}

#[test]
fn compact_consumers_preserve_an_unrelated_scan_and_guard_partial_callbacks() {
    for (source, reduction) in [
        ("entry main(xs:[]i32) ([]i32,[]i32)=(scan((+),0,xs),map(|x|x+1,filter(|x|x>0,xs)))", false),
        ("entry main(xs:[]i32) ([]i32,i32)=(scan((+),0,xs),reduce((+),0,map(|x|100/x,filter(|x|x>0,xs))))", true),
    ] {
        let source = compile_thru_tlc(source).unwrap();
        let program = fuse(from_tlc(&source).unwrap()).unwrap();
        let query = Query(&program.graph);
        let root = test_graph::root(&program.graph);
        let parallels: Vec<_> = query.operations(root).unwrap().into_iter()
            .filter(|&operation| query.contains("ParallelOperation", operation).unwrap()).collect();
        assert_eq!(parallels.len(), 1);
        let mut roles = (0, 0, 0);
        query.for_function("OutputRole", |keys, role| {
            if keys[0] == parallels[0] {
                roles.0 += usize::from(query.enode("PrefixOutput", role)?.is_some());
                roles.1 += usize::from(query.enode("CompactedOutput", role)?.is_some());
                roles.2 += usize::from(query.enode("TotalOutput", role)?.is_some());
            }
            Ok(())
        }).unwrap();
        assert_eq!(roles, (1, usize::from(!reduction), usize::from(reduction)));
        if reduction {
            let mut guarded = 0;
            query.for_each("LiveOperation", |row| {
                if ir::code(&program.graph, query.required("OperationPayload", row[0])?)
                    == Code::Scalar(Payload::Binary(BinaryOperator::Divide)) {
                    let region = query.required("OperationRegion", row[0])?;
                    let owner = query.row("RegionOwnedBy", |row| row[0] == region)?.unwrap()[1];
                    assert!(query.contains("BranchOperation", owner)?);
                    guarded += 1;
                }
                Ok(())
            }).unwrap();
            assert_eq!(guarded, 1);
        }
    }
}

#[test]
fn scan_barriers_survive_intermediate_map_contraction() {
    let source =
        compile_thru_tlc("entry main(xs:[64]i32) i32 = reduce((+),0i32,map(|x|x+1,scan((+),0i32,xs)))")
            .unwrap();
    let program = fuse(from_tlc(&source).unwrap()).unwrap();
    let query = Query(&program.graph);
    let mut scans = 0;
    let mut totals = 0;
    query
        .for_each("LiveOperation", |row| {
            if query.contains("ParallelOperation", row[0])? {
                let mut scan = false;
                let mut total = false;
                for slot in 0..analysis::count(query, "ResultCount", row[0])? {
                    let kind = query.required("ResultKindOf", (row[0], slot))?;
                    if let Some(fold) = query.enode("Folded", kind)? {
                        total |= query.enode("Total", fold[0])?.is_some();
                        scan |= query.enode("InclusiveScan", fold[0])?.is_some();
                    }
                }
                assert!(
                    !(scan && total),
                    "a dependent total must wait for completed scan prefixes"
                );
                scans += usize::from(scan);
                totals += usize::from(total);
            }
            Ok(())
        })
        .unwrap();
    assert_eq!((scans, totals), (1, 1));
}

#[test]
fn free_names_preserve_large_shared_fanout() {
    let mut source = String::from("entry main(x:i32) (");
    source.push_str(&vec!["i32"; 256].join(","));
    source.push_str(")=(");
    source.push_str(&(0..256).map(|i| format!("x+{i}")).collect::<Vec<_>>().join(","));
    source.push(')');
    let tlc = compile_thru_tlc(&source).unwrap();
    let mut program = from_tlc(&tlc).unwrap();
    analysis::run(&mut program.graph).unwrap();
    let query = Query(&program.graph);
    let entry = query.first("EntryOrder").unwrap().unwrap()[0];
    let root = query.required("FunctionRoot", entry).unwrap();
    let input = query.required("RegionParameter", (root, 0_i64)).unwrap();
    let mut uses = 0;
    query
        .for_each("DirectOperationUse", |row| {
            uses += usize::from(row[1] == input);
            Ok(())
        })
        .unwrap();
    assert!(
        uses >= 256,
        "shared input must remain a dependency of every operation"
    );
}

#[test]
fn fusion_across_scalar_work_preserves_dependencies_and_traps() {
    for source in [
        "entry main(xs:[]i32) (i32,[]i32)=let ys=filter(|x|x>0,xs) let n=length(ys) in (n,map(|x|x+n,xs))",
        "entry main(xs:[]i32,n:i32) ([]i32,i32,[]i32)=let a=map(|x|x+1,xs) let guard=100/n let b=map(|x|x*2,xs) in (a,guard,b)",
    ] {
        let source = compile_thru_tlc(source).unwrap();
        let program = fuse(from_tlc(&source).unwrap()).unwrap();
        let query = Query(&program.graph);
        let parallels = query.operations(test_graph::root(&program.graph)).unwrap().into_iter()
            .filter(|&operation| query.contains("ParallelOperation", operation).unwrap()).count();
        assert_eq!(parallels, 2, "an unavailable input or intervening trap prevents fusion");
    }
}

#[test]
fn scatter_fuses_across_copy_and_repeated_shape_calculations() {
    let source = compile_thru_tlc(include_str!("../../../../tests/compiler/library/computed_dispatch_extents_preserve_parameters_wrapping_and_grid_limits.wyn")).unwrap();
    let program = fuse(from_tlc(&source).unwrap()).unwrap();
    let query = Query(&program.graph);
    let mut parallels = 0;
    query
        .for_each("LiveOperation", |row| {
            parallels += usize::from(query.contains("ParallelOperation", row[0])?);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        parallels, 1,
        "the mapped contribution is computed in the scatter lane"
    );
}

#[test]
fn accumulator_sinking_stabilizes_when_domains_differ() {
    let source = compile_thru_tlc("entry main(dest:*[]i32,n:i32,m:i32) []i32 = let values=map(|i|i+1,iota(n)) in scatter(dest,iota(m),values)").unwrap();
    let program = fuse(from_tlc(&source).unwrap()).unwrap();
    let query = Query(&program.graph);
    let mut parallels = 0;
    query
        .for_each("LiveOperation", |row| {
            parallels += usize::from(query.contains("ParallelOperation", row[0])?);
            Ok(())
        })
        .unwrap();
    assert_eq!(parallels, 2, "unequal domains must retain separate traversals");
}
