use super::super::query::Query;
use super::super::{
    finish_for_test, from_tlc, fuse, optimize_for_test, place, schedule, schedule_with_subgroups,
};
use crate::egglog::dataflow::ir::{Bounds, Code};
use crate::egglog::graph::Formula;
use crate::egglog::graph::Graph;
use crate::egglog::payload::read;
use crate::egglog::{analysis, test_graph};
use crate::{compile_thru_tlc, CodegenTarget, PipelineTopologyPolicy};
use egglog_engine::sort::{S, Z};
use egglog_engine::Value;
use egglog_engine::Write;

#[test]
fn whole_row_fusion_preserves_conflicting_callback_effects() {
    for (producer, consumer, fused) in [
        ("ReadsInput", "ReadsInput", true),
        ("WritesInput", "ReadsInput", false),
        ("ReadsInput", "WritesInput", false),
    ] {
        let facts = include_str!("tests/array_reduce_effects.egg")
            .replace("PRODUCER_EFFECTS", &format!("({producer} (OperationId 2) 0)"))
            .replace("CONSUMER_EFFECTS", &format!("({consumer} (OperationId 4) 0)"));
        let mut graph = test_graph::fixture(
            &[
                (
                    90,
                    Code::Constant {
                        ty: crate::types::i32(),
                        bits: 0,
                    },
                ),
                (0, Code::ArrayLiteral),
                (1, Code::Parallel),
                (2, Code::Scalar(crate::egglog::payload::Payload::Unit)),
                (3, Code::Parallel),
                (4, Code::Scalar(crate::egglog::payload::Payload::Vector)),
                (5, Code::Index),
            ],
            &facts,
        )
        .unwrap();
        super::super::memory::validate(&mut graph).unwrap();
        super::super::fusion::run(&mut graph).unwrap();
        super::super::memory::validate(&mut graph).unwrap();
        assert_eq!(
            top_level(&graph, test_graph::entry(&graph), "Parallel").len(),
            if fused { 1 } else { 2 }
        );
    }
}

#[test]
fn every_collective_callback_must_be_available_before_distribution() {
    use crate::egglog::payload::Payload;
    use crate::op::BinaryOperator;
    for blocked in [None, Some(20), Some(30)] {
        let mut facts = include_str!("tests/collective_distribution.egg").to_owned();
        for (prefix, base) in [("FIRST", 20), ("SECOND", 30)] {
            let capture = if blocked == Some(base) {
                format!("(set (FixtureInput (OperationId {}) 0) (ValueNameRef (ValueId {}))) (set (FixtureInput (OperationId {}) 1) (ValueNameRef (ValueId 10))) (set (FixtureOutput (OperationId {}) 0) (ValueNameRef (ValueId {}))) (set (ValueType (ValueId {})) scalar) (set (FixtureRegion (OperationId {})) (RegionId {})) (set (FixtureOrder (OperationId {})) 1) (DiscardableOperation (OperationId {})) (RepeatableOperation (OperationId {}))",base+1,base+2,base+1,base+1,base+3,base+3,base+1,base,base+1,base+1,base+1)
            } else {
                String::new()
            };
            let result = base + if blocked == Some(base) { 3 } else { 2 };
            facts = facts.replace(&format!("{prefix}_CAPTURE"), &capture).replace(
                &format!("{prefix}_RESULT"),
                &format!("(ValueNameRef (ValueId {result}))"),
            );
        }
        let mut codes = vec![
            (0, Code::Parallel),
            (1, Code::Parallel),
            (
                90,
                Code::Constant {
                    ty: crate::types::i32(),
                    bits: 0,
                },
            ),
        ];
        for id in [20, 21, 30, 31] {
            codes.push((id, Code::Scalar(Payload::Binary(BinaryOperator::Add))));
        }
        let mut graph = test_graph::fixture(&codes, &facts).unwrap();
        analysis::run(&mut graph).unwrap();
        super::super::fusion::run(&mut graph).unwrap();
        super::place(
            &mut graph,
            PipelineTopologyPolicy::AllowGenerated,
            super::CostModel::default(),
        )
        .unwrap();
        let query = Query(&graph);
        let outer = top_level(&graph, test_graph::entry(&graph), "Parallel");
        assert_eq!(outer.len(), 1);
        assert_eq!(rank(query, outer[0]), if blocked.is_some() { 1 } else { 2 });
        assert_eq!(
            query.0.output_count(outer[0]),
            2,
            "both source results must survive the legality decision"
        );
    }
}

#[test]
fn product_loops_track_array_component_length_proofs() {
    for (source, capacity) in [
        ("entry main(steps:i32) []i32=let (ys,k)=loop (ys,k)=([0,0,0,0],0) while k<steps do (ys with [k%4]=k,k+1) in ys", Some(4)),
        ("entry main(steps:i32, choose:bool) []i32=let (ys,k)=loop (ys,k)=([0,0,0,0],0) while k<steps do if choose then (ys with [0]=k,k+1) else (ys with [1]=k,k+1) in ys", Some(4)),
        ("entry main(steps:i32, choose:bool) []i32=let (ys,k)=loop (ys,k)=(replicate(4,0i32),0) while k<steps do if choose then (ys with [0]=k,k+1) else (replicate(length(ys)+1,0i32),k+1) in ys", None),
        ("entry main(steps:i32) []i32=let (ys,k)=loop (ys,k)=(replicate(4,0i32),0) while k<steps do (map(|i|if i<length(ys) then ys[i] else k,iota(length(ys)+1)),k+1) in ys", None),
    ] {
        let source = compile_thru_tlc(source).unwrap();
        let mut program = from_tlc(&source).unwrap();
        super::super::analysis::run(&mut program.graph).unwrap();
        super::analyze(&mut program.graph).unwrap();let query=Query(&program.graph);let mut components=0;
        query.for_each("LoopArrayComponent",|row| {
            let bound=query.lookup("OuterCapacity",row[3])?.map(|value|program.graph.value_to_base::<i64>(value));
            assert_eq!(query.contains("LoopPreservesComponentLength",(row[0],row[1],row[2]))?,capacity.is_some());assert_eq!(bound,capacity);components+=1;Ok(())
        }).unwrap();
        assert!(components > 0, "exercise a projected array in a product carry");
    }
}

#[test]
fn sliced_local_collections_retain_their_source_capacity() {
    let source = compile_thru_tlc("entry main(xs:[][8]f32) []f32 = map(|row| let values=map(|x|x+1.0,row[1..i32(row[0])]) in reduce(|a,b|min(a,b),f32.highest,values)+reduce(|a,b|max(a,b),f32.lowest,values),xs)").unwrap();
    let mut program = selected(&source);
    super::super::analysis::run(&mut program.graph).unwrap();
    let query = Query(&program.graph);
    let mut collections = 0;
    query
        .for_each("LiveOperation", |row| {
            if query.contains("ParallelOperation", row[0])?
                && query.enode("InvocationLocal", query.required("OperationExecution", row[0])?)?.is_some()
            {
                for name in query.0.outputs(row[0]) {
                    let name = query.name(name)?;
                    let info = query.required("ValueType", name)?;
                    if array_extents(query, info).is_some() {
                        assert_eq!(
                            program.graph.value_to_base::<i64>(query.required("OuterCapacity", name)?),
                            8
                        );
                        collections += 1;
                    }
                }
            }
            Ok(())
        })
        .unwrap();
    assert!(collections > 0, "exercise a sliced local collection");
}

fn selected(
    source: &crate::tlc::stage::OwnershipApplied,
) -> super::super::Program<'_, super::super::InterfacesPrepared> {
    let imported = from_tlc(source).unwrap();
    let fused = fuse(imported).unwrap();
    let optimized = optimize_for_test(fused).unwrap();
    let placed = place(optimized, PipelineTopologyPolicy::AllowGenerated).unwrap();
    let scheduled = schedule(placed).unwrap();
    super::super::prepare_interfaces(scheduled).unwrap()
}

fn planned(source: &str, check: impl FnOnce(&Graph)) {
    let tlc = compile_thru_tlc(source).unwrap();
    let program = schedule(
        place(
            optimize_for_test(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap(),
    )
    .unwrap();
    check(&program.graph);
}

// Compile related cases once; keep every assertion tied to its source entry.
fn cases_source(sources: &[impl AsRef<str>]) -> String {
    sources
        .iter()
        .enumerate()
        .map(|(i, source)| source.as_ref().replacen("entry main(", &format!("entry case_{i}("), 1))
        .collect::<Vec<_>>()
        .join("\n")
}

fn check_entries<Stage>(
    program: &super::super::Program<'_, Stage>,
    count: usize,
    check: impl Fn(usize, &Graph, Value),
) {
    let query = Query(&program.graph);
    let mut seen = vec![false; count];
    crate::egglog::test_graph::for_entries(query, |keys, _| {
        let fields = query.fields("FunctionId", keys[0])?;
        let name = program.graph.value_to_base::<S>(fields[1]);
        if let Some(index) = name.as_str().strip_prefix("case_") {
            let index: usize = index.parse().unwrap();
            seen[index] = true;
            check(index, &program.graph, keys[0]);
        }
        Ok(())
    })
    .unwrap();
    assert!(seen.into_iter().all(|present| present));
}

fn planned_cases(sources: &[impl AsRef<str>], check: impl Fn(usize, &Graph, Value)) {
    for (index, source) in sources.iter().enumerate() {
        let tlc = compile_thru_tlc(&cases_source(std::slice::from_ref(source))).unwrap();
        let program = schedule(
            place(
                optimize_for_test(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap(),
                PipelineTopologyPolicy::AllowGenerated,
            )
            .unwrap(),
        )
        .unwrap();
        check_entries(&program, 1, |_, graph, function| check(index, graph, function));
    }
}

fn placed_cases(sources: &[impl AsRef<str>], check: impl Fn(usize, &Graph, Value)) {
    for (index, source) in sources.iter().enumerate() {
        let tlc = compile_thru_tlc(&cases_source(std::slice::from_ref(source))).unwrap();
        let program = place(
            optimize_for_test(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap();
        check_entries(&program, 1, |_, graph, function| check(index, graph, function));
    }
}

fn distributed_cases(sources: &[impl AsRef<str>], check: impl Fn(usize, &Graph, Value)) {
    for (index, source) in sources.iter().enumerate() {
        let tlc = compile_thru_tlc(&cases_source(std::slice::from_ref(source))).unwrap();
        let mut program = place(
            optimize_for_test(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap();
        super::schedule(&mut program.graph).unwrap();
        check_entries(&program, 1, |_, graph, function| check(index, graph, function));
    }
}

fn fused_cases(sources: &[impl AsRef<str>], check: impl Fn(usize, &Graph, Value)) {
    let tlc = compile_thru_tlc(&cases_source(sources)).unwrap();
    let program = fuse(from_tlc(&tlc).unwrap()).unwrap();
    check_entries(&program, sources.len(), check);
}

fn top_level(graph: &Graph, function: Value, constructor: &str) -> Vec<Value> {
    let query = Query(graph);
    let root = query.0.function_root(function).unwrap();
    query
        .operations(root)
        .unwrap()
        .into_iter()
        .filter(|&operation| code_is(query, operation, constructor))
        .collect()
}

fn dispatches(graph: &Graph, function: Value) -> Vec<super::super::execution::Dispatch> {
    let mut graph = graph.proof_snapshot();
    let execution = super::super::execution::Execution::select(&mut graph).unwrap();
    let query = Query(&graph);
    fn collect(
        query: Query<'_>,
        execution: &super::super::execution::Execution,
        region: Value,
        stages: &mut Vec<super::super::execution::Dispatch>,
    ) {
        for operation in query.operations(region).unwrap() {
            if let Ok(dispatch) = execution.dispatch(operation) {
                stages.push(*dispatch);
            } else {
                for child in super::super::execution::control_regions(query, operation).unwrap() {
                    collect(query, execution, child, stages);
                }
            }
        }
    }
    let mut stages = Vec::new();
    collect(
        query,
        &execution,
        query.0.function_root(function).unwrap(),
        &mut stages,
    );
    stages
}

fn array_extents(query: Query<'_>, info: Value) -> Option<Vec<Option<i64>>> {
    let info = query.enode("LogicalValue", info).unwrap()?;
    fn dimensions(query: Query<'_>, ty: Value) -> Option<Vec<Option<i64>>> {
        if let Some(array) = query.enode("ArrayType", ty).unwrap() {
            let mut extents = query
                .vector(array[1])
                .unwrap()
                .into_iter()
                .map(|extent| {
                    query
                        .enode("FixedExtent", extent)
                        .unwrap()
                        .map(|fixed| query.0.value_to_base::<i64>(fixed[0]))
                })
                .collect::<Vec<_>>();
            extents.extend(dimensions(query, array[2]).unwrap_or_default());
            return Some(extents);
        }
        let product = query.enode("ProductType", ty).unwrap()?;
        let mut shapes =
            query.vector(product[0]).unwrap().into_iter().filter_map(|field| dimensions(query, field));
        let shape = shapes.next()?;
        assert!(
            shapes.all(|other| other == shape),
            "a coordinate-mapped product requires aligned array fields"
        );
        Some(shape)
    }
    dimensions(query, info[0])
}

#[test]
fn invariant_array_regions_are_evaluated_outside_parallel_callbacks() {
    fused_cases(&["def producer(n:i32) []f32=map(|i|f32.i32(i),0i32..<n) entry main(n:i32,j:i32) [4]f32=map(|p|producer(n)[j],iota(4))"], |_, graph, function| {
        assert_eq!(top_level(graph, function, "Parallel").len(), 2,
            "the producer and its consumer must be separate enclosing statements");
    });
}

#[test]
fn invariant_reductions_and_scans_move_with_their_memory_results() {
    fused_cases(
        &[
            "entry main(xs:[]i32) [4]i32=map(|p|reduce((+),0i32,xs),iota(4))",
            "entry main(xs:[]i32,j:i32) [4]i32=map(|p|scan((+),0i32,xs)[j],iota(4))",
        ],
        |_, graph, function| {
            let query = Query(graph);
            let collectives = top_level(graph, function, "Parallel")
                .into_iter()
                .filter(|&operation| !folds(query, operation).is_empty())
                .count();
            assert_eq!(collectives, 1, "the invariant collective must leave the callback");
        },
    );
}

#[test]
fn invariant_memory_computations_move_out_of_loops_and_stay_guarded_by_branches() {
    fused_cases(
        &[
            "entry main(xs:[]i32,b:bool) i32=if b then reduce((+),0i32,xs) else 0i32",
            "entry main(xs:[]i32,n:i32) i32=loop x=0i32 for i<n do x+reduce((+),0i32,xs)",
        ],
        |index, graph, function| {
            assert_eq!(
                top_level(graph, function, "Parallel").len(),
                usize::from(index == 1),
                "a closed reduction may leave a loop without speculating expensive branch work"
            );
            if index == 0 {
                assert_eq!(top_level(graph, function, "Branch").len(), 1);
            }
        },
    );
}

#[test]
fn invariant_fusion_moves_all_members_together() {
    fused_cases(&["entry main() [4]i32=map(|p|let a=map(|i|i+2,iota(8)) in let b=map(|i|i+1,iota(8)) in a[0]+b[0],iota(4))"], |_, graph, function| {
            let query=Query(graph);
            assert!(top_level(graph,function,"Parallel").into_iter().any(|operation| infos(query,operation).into_iter().filter(|&info|array_extents(query,info).is_some()).count()==2),"the complete fused computation must leave the callback");
        });
}

#[test]
fn retained_fusion_outputs_block_motion_when_any_member_depends_on_the_iteration() {
    fused_cases(&["entry main() [4]i32=map(|p|let a=map(|i|i+p,iota(8)) in let b=map(|i|i+1,iota(8)) in a[0]+b[0],iota(4))"], |_, graph, function| {
            let query=Query(graph);let top=top_level(graph,function,"Parallel");
            let dependent=within(graph,function).into_iter().filter(|&operation|code_is(query,operation,"Parallel")&&!top.contains(&operation)&&sizes(query,operation)==[8]).count();
            assert_eq!(dependent,1,"the iteration-dependent row must stay in the callback; its independent sibling may move out");
        });
}

#[test]
fn dependent_or_partial_regions_stay_inside_parallel_callbacks() {
    fused_cases(
        &[
            "entry main(j:i32) [4]i32=map(|p|map(|i|i+p,iota(8))[j],iota(4))",
            "entry main(j:i32) [4]i32=map(|p|map(|i|i/j,iota(8))[0],iota(4))",
            "entry main(j:i32) [4]i32=map(|p|map(|i|loop x=i while x<j do x+1,iota(8))[0],iota(4))",
        ],
        |index, graph, function| {
            let query = Query(graph);
            let root = query.0.function_root(function).unwrap();
            let result = query.0.yield_value(root, 0_i64).unwrap();
            let mut checked = false;
            for outer in top_level(graph, function, "Parallel") {
                if !query.0.outputs(outer).contains(&result) {
                    assert!(
                        !query.contains("OperationTrap", outer).unwrap(),
                        "case {index}: only total independent producers may leave the callback"
                    );
                    continue;
                }
                let body = query.0.control_region(outer, 0_i64).unwrap();
                assert!(
                    within(graph, function).into_iter().any(|operation| code_is(
                        query, operation, "Parallel"
                    ) && query
                        .0
                        .operation_region(operation)
                        .unwrap()
                        == body),
                    "case {index}: the dependent or partial computation stays nested"
                );
                checked = true;
            }
            assert!(checked, "case {index}: result-producing callback missing");
        },
    );
}

#[test]
fn exported_views_have_their_own_logical_domain() {
    planned_cases(
        &["entry main() [][]i32=let rows=map(|n|map(|i|i+n,iota(3)),[0,1,3,7]) in rows[1..3]"],
        |_, graph, function| {
            let query = Query(graph);
            let views = top_level(graph, function, "View");
            assert_eq!(views.len(), 1);
            let operation = views[0];
            let slices = query.vector(query.0.formula(operation, Formula::Slices).unwrap()).unwrap();
            let slice = query.fields("Slice", slices[0]).unwrap();
            assert_eq!(
                graph.value_to_base::<i64>(query.required("StaticIndex", slice[1]).unwrap()),
                2
            );
            let source = query.name(query.0.input(operation, 0_i64).unwrap()).unwrap();
            let count = query.required("LogicalOuterCount", source).unwrap();
            assert_eq!(
                graph.value_to_base::<i64>(query.required("StaticIndex", count).unwrap()),
                4
            );
        },
    );
}

#[test]
fn collective_width_tracks_materialized_accumulator_size() {
    let cases = [(8, 256), (9, 128), (32, 64), (65, 32)];
    let sources = cases.map(|(columns, _)| {
        format!(
            "entry main(xs:[][{columns}]i32) [][{columns}]i32 =
         scan(|a,b|map2((+),a,b),replicate({columns},0i32),xs)"
        )
    });
    planned_cases(&sources, |index, graph, function| {
        let stages = dispatches(graph, function);
        assert!(!stages.is_empty(), "case {index}: missing scan dispatch");
        assert_eq!(stages[0].width.get(), cases[index].1, "case {index}");
    });
}

#[test]
fn scan_input_cannot_capture_a_fused_total() {
    fused_cases(
        &[
            "entry main(xs:[]i32) []i32 = let t=reduce((+),0,xs) in scan((+),0,map(|x|x+t,xs))",
            "entry main(xs:[]i32) []i32 = let t=reduce((+),0,xs) in scan((+),t,xs)",
            "entry main(xs:[]i32) []i32 = let t=reduce((+),0,xs) in scan(|a,b|a+b+t,0,xs)",
        ],
        |index, graph, function| {
            let query = Query(graph);
            let mut scans = Vec::new();
            let mut totals = Vec::new();
            for operation in top_level(graph, function, "Parallel") {
                for kind in folds(query, operation) {
                    if query.enode("Total", kind).unwrap().is_some() {
                        totals.push(operation);
                    } else {
                        scans.push(operation);
                    }
                }
            }
            assert_eq!((scans.len(), totals.len()), (1, 1), "case {index}");
            assert_ne!(
                scans[0], totals[0],
                "case {index}: scan inputs depend on a completed total"
            );
        },
    );
}

#[test]
fn updated_slice_preserves_its_descriptor() {
    planned(
        include_str!("../../../testfiles/regressions/runtime_array_updated_slice.wyn"),
        |graph| {
            let query = Query(graph);
            let mut updates = 0;
            query
                .for_each("LiveOperation", |row| {
                    if let Code::Update { .. } = query.0.code(row[0])? {
                        let output = query.name(query.0.output(row[0], 0_i64)?)?;
                        let source = query.name(query.0.input(row[0], 0_i64)?)?;
                        assert_eq!(
                            query.required("ValueType", output)?,
                            query.required("ValueType", source)?,
                            "an update preserves the complete typed memory view"
                        );
                        updates += 1;
                    }
                    Ok(())
                })
                .unwrap();
            assert_eq!(updates, 1);
        },
    );
}

#[test]
fn shrinking_loop_does_not_inherit_its_initial_extent() {
    planned_cases(
        &["entry main(xs:[]i32) []i32=loop out=xs for i<2 do out[0..length(out)-1]"],
        |_, graph, function| {
            let query = Query(graph);
            let loops = top_level(graph, function, "Loop");
            assert_eq!(loops.len(), 1);
            let operation = loops[0];
            assert!(!query.contains("LoopPreservesOuterLength", (operation, 0_i64)).unwrap());
            let initial = query.name(query.0.argument(operation, 0_i64, 0_i64).unwrap()).unwrap();
            let output = query.name(query.0.output(operation, 0_i64).unwrap()).unwrap();
            assert!(array_extents(query, query.required("ValueType", output).unwrap()).is_some());
            assert!(!query.contains("OuterShapeSource", (output, initial)).unwrap());
        },
    );
}

#[test]
fn loop_capacity_obligations_use_the_length_preservation_proof() {
    planned_cases(
        &[
            include_str!("../../../testfiles/regressions/triangulation_allocated_array_loop.wyn"),
            "entry main(n:i32) []i32=loop out=iota(n) for i<2 do out[1..length(out)]",
        ],
        |index, graph, function| {
            let query = Query(graph);
            let loops = top_level(graph, function, "Loop");
            assert_eq!(loops.len(), 1);
            assert_eq!(
                query.contains("LoopPreservesOuterLength", (loops[0], 0_i64)).unwrap(),
                index == 0
            );
        },
    );
}

fn local_capacities(graph: &Graph) -> Vec<i64> {
    let mut graph = graph.proof_snapshot();
    super::analyze(&mut graph).unwrap();
    let query = Query(&graph);
    let mut capacities = Vec::new();
    query
        .for_each("LiveOperation", |row| {
            if query.contains("ParallelOperation", row[0])?
                && query.enode("InvocationLocal", query.required("OperationExecution", row[0])?)?.is_some()
            {
                for name in query.0.outputs(row[0]) {
                    let name = query.name(name)?;
                    if let Some(capacity) = query.lookup("OuterCapacity", name)? {
                        capacities.push(graph.value_to_base::<i64>(capacity));
                    }
                }
            }
            Ok(())
        })
        .unwrap();
    capacities
}

#[test]
fn compiler_repro_local_capacity_plan() {
    planned(
        include_str!("../../../testfiles/regressions/local_array_capacity.wyn"),
        |graph| {
            let capacities = local_capacities(graph);
            assert!(!capacities.is_empty());
            assert!(
                capacities.iter().all(|&elements| elements == 3),
                "three local elements: {capacities:?}"
            );
        },
    );
}

#[test]
fn compiler_repro_nested_capacity_plan() {
    let source =
        "entry main(xs:[]i32) []i32=map(|x| let a=loop a=[x,x+1,x+2] for k<2 do map(|y|y+1,a) in a[0],xs)";
    planned(source, |graph| {
        let capacities = local_capacities(graph);
        assert!(capacities.contains(&3));
        assert!(
            capacities.iter().all(|&elements| elements == 1 || elements == 3),
            "{capacities:?}"
        );
    });
}

#[test]
fn result_tree_counts_are_available_to_fusion_choices() {
    // Scan consumers share a final dispatch rather than a pre-reduction body;
    // planning_contracts covers the resulting public execution plan.
    let cases = [
        (
            "entry main(xs:[]i32) ([]i32,[]i32)=(map(|x|x+1,xs),map(|x|x*2,xs))",
            vec![(2, false, false)],
        ),
        (
            "entry main(xs:[]i32) i32=reduce((+),0,map(|x|x+1,xs))",
            vec![(0, false, true)],
        ),
        (
            "entry main(xs:[]i32) ([]i32,i32)=let ys=map(|x|x+1,xs) in (ys,reduce((+),0,ys))",
            vec![(1, false, true)],
        ),
        (
            "entry main(xs:[]i32) []i32=map(|x|x+1,filter(|x|x>0,xs))",
            vec![(1, false, false)],
        ),
        (
            "entry main(xs:[]i32) (i32,i32)=let ys=filter(|x|x>0,xs) in (length(ys),reduce((+),0,ys))",
            vec![(0, false, true)],
        ),
        (
            "entry main(xs:[]i32) i32=length(filter(|x|x>0,xs))",
            vec![(0, false, true)],
        ),
        (
            "entry main(xs:[]i32) (i32,i32)=let a=filter(|x|x>0,xs) let b=filter(|x|x<20,map(|x|x+1,a)) in (length(a),length(b))",
            vec![(0, false, true)],
        ),
        (
            "entry main(xs:[]i32) ([]i32,i32)=(scan((+),0,xs),reduce((+),0,xs))",
            vec![(1, true, true)],
        ),
    ];
    let sources: Vec<_> = cases.iter().map(|(source, _)| *source).collect();
    let tlc = compile_thru_tlc(&cases_source(&sources)).unwrap();
    let program = fuse(from_tlc(&tlc).unwrap()).unwrap();
    check_entries(&program, cases.len(), |index, graph, region| {
        let query = Query(graph);
        let mut counts = Vec::new();
        for operation in top_level(graph, region, "Parallel") {
            let arrays = infos(query, operation)
                .into_iter()
                .filter(|&info| array_extents(query, info).is_some())
                .count() as i64;
            let mut scans = false;
            let mut totals = false;
            for kind in folds(query, operation) {
                scans |= query.enode("Total", kind).unwrap().is_none();
                totals |= query.enode("Total", kind).unwrap().is_some()
                    || query.enode("InclusiveScanAndTotal", kind).unwrap().is_some()
                    || query.enode("ExclusiveScanAndTotal", kind).unwrap().is_some();
            }
            counts.push((arrays, scans, totals));
        }
        counts.sort();
        assert_eq!(counts, cases[index].1, "case {index}");
    });
}

#[test]
fn accumulator_state_order_is_read_from_the_plan() {
    let tlc = compile_thru_tlc(
        "entry main(xs:[]i32) ([]i32,bool)=
         (scan((+),0,xs),reduce(|a,b|a&&b,true,map(|x|x>0,xs)))",
    )
    .unwrap();
    let mut program = place(
        optimize_for_test(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    // Reorder complete clauses and their bindings, instead of mutating a
    // separate accumulator table. Different element types expose slot mistakes.
    let query = Query(&program.graph);
    let operation = top_level(&program.graph, test_graph::entry(&program.graph), "Parallel")
        .into_iter()
        .find(|&operation| folds(query, operation).len() == 2)
        .expect("two independent accumulator clauses");
    let mut results = query.0.parallel_results(operation).unwrap();
    results.kinds.reverse();
    results.bindings.reverse();
    for binding in &mut results.bindings {
        binding.clause = binding.clause.map(|clause| 1 - clause);
    }
    let mut outputs = query.0.outputs(operation);
    outputs.reverse();
    let reducers =
        query.0.reducers(operation).into_iter().map(|(slot, region)| (1 - slot, region)).collect();
    let edges = query.0.collective_groups(operation);
    program
        .graph
        .update(|mut sink| {
            sink.set_outputs(operation, outputs)?;
            sink.set_reducers(operation, reducers);
            sink.clear_collective_inputs(operation);
            for ((clause, kind), values) in edges {
                sink.set_collective_inputs(operation, 1 - clause, kind, values)?;
            }
            sink.set_parallel_results(operation, results);
            Ok(())
        })
        .unwrap();
    super::super::dataflow::edit::changed(&mut program.graph).unwrap();
    let ssa = finish_for_test(
        super::super::prepare_interfaces(schedule(program).unwrap()).unwrap(),
        CodegenTarget::Wgsl,
    )
    .unwrap();
    assert!(!ssa.entry_points.is_empty());
    let wgsl = crate::lower_ssa_to_wgsl(ssa).unwrap();
    let module = naga::front::wgsl::parse_str(&wgsl).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn flat_and_ranked_collectives_share_dimension_metadata() {
    let cases = [
        ("entry main(xs:[6]i32) [6]i32=map(|x|x+1,xs)", "map", 1, 6),
        ("entry main(xs:[6]i32) i32=reduce(|a,b|a+b,0,xs)", "reduce", 1, 6),
        ("entry main(xs:[6]i32) [6]i32=scan(|a,b|a+b,0,xs)", "scan", 1, 6),
        ("entry main(xs:[6]i32) []i32=filter(|x|x>0,xs)", "filter", 1, 6),
        (
            "entry main(dest:*[2][3]i32) ([2][3]i32,[2]u32,u32)=
             bucket_scatter_2d(dest,[[(0,1),(0,2),(0,3)],[(1,4),(1,5),(1,6)]])",
            "bucket-scatter",
            2,
            6,
        ),
        (
            "entry main(dest:*[2][4]i32) ([2][4]i32,[2]u32,u32)=
             bucket_scatter_3d(dest,[[[(0,1),(0,2)],[(0,3),(0,4)]],
                                      [[(1,5),(1,6)],[(1,7),(1,8)]]])",
            "bucket-scatter",
            3,
            8,
        ),
    ];
    let sources = cases.iter().map(|case| case.0).collect::<Vec<_>>();
    distributed_cases(&sources, |index, graph, function| {
        let (_source, kind, expected_rank, count) = cases[index];
        let query = Query(graph);
        let mut matching = 0;
        for operation in within(graph, function) {
            let code = query.0.code(operation).unwrap();
            if kind == "bucket-scatter" {
                if let Code::BucketScatter { coordinates, depth } = code {
                    assert_eq!(depth, expected_rank as usize);
                    let inputs = query.0.inputs(operation);
                    let mut dimensions = vec![None; expected_rank as usize];
                    for (input, mapping) in inputs.into_iter().skip(1).zip(coordinates) {
                        let name = query.name(input).unwrap();
                        let extents = array_extents(query, query.required("ValueType", name).unwrap())
                            .expect("bucket input array");
                        assert_eq!(extents.len(), mapping.len());
                        for (extent, axis) in extents.into_iter().zip(mapping) {
                            let axis = axis as usize;
                            let extent = extent.expect("fixed input extent");
                            if let Some(previous) = dimensions[axis] {
                                assert_eq!(previous, extent);
                            }
                            dimensions[axis] = Some(extent);
                        }
                    }
                    assert_eq!(
                        dimensions
                            .into_iter()
                            .map(|extent| extent.expect("covered domain axis"))
                            .product::<i64>(),
                        count
                    );
                    matching += 1;
                }
                continue;
            }
            if !matches!(code, Code::Parallel) {
                continue;
            }
            let matches = (0..query.0.result_count(operation)).any(|clause| {
                let value = query.0.result_kind(operation, clause).unwrap();
                query
                    .enode(
                        match kind {
                            "map" => "Collected",
                            "filter" => "Compacted",
                            _ => "Folded",
                        },
                        value,
                    )
                    .unwrap()
                    .is_some()
            });
            if !matches {
                continue;
            }
            assert_eq!(rank(query, operation), expected_rank as usize, "case {index}");
            assert_eq!(
                sizes(query, operation).into_iter().product::<i64>(),
                count,
                "case {index}"
            );
            matching += 1;
        }
        assert!(matching > 0, "case {index}: no {kind} traversal");
    });
}

#[test]
fn nested_domains_retain_distinct_axes_and_capture_origins() {
    fused_cases(
        &["entry main(xs:[4]i32) [4][4]i32=map(|x|map(|y|x+y,xs),xs)"],
        |_, graph, function| {
            let query = Query(graph);
            let [outer] = top_level(graph, function, "Parallel")[..] else {
                panic!("one outer map");
            };
            let domain = query.0.operation_domain(outer).unwrap();
            let outer_index = query.0.domain_axis(domain, 0_i64).unwrap();
            let body = query.0.control_region(outer, 0_i64).unwrap();
            let inner = query
                .operations(body)
                .unwrap()
                .into_iter()
                .find(|&operation| code_is(query, operation, "Parallel"))
                .expect("the inner map retains its captured outer value");
            let domain = query.0.operation_domain(inner).unwrap();
            let inner_index = query.0.domain_axis(domain, 0_i64).unwrap();
            assert_ne!(outer_index, inner_index, "nested axes need distinct binders");
            let lane = query.0.control_region(inner, 0_i64).unwrap();
            let mut captures = false;
            query
                .for_each("RegionCapture", |row| {
                    captures |= row[0] == lane && query.contains("DefinitionRegion", (row[1], body))?;
                    Ok(())
                })
                .unwrap();
            assert!(
                captures,
                "the inner body must capture a value bound by the outer body"
            );
        },
    );
}

#[test]
fn ranked_inputs_keep_their_independent_coordinate_mappings() {
    planned(
        "entry main(dest:*[4][8]i32) ([4][8]i32,[4]u32,u32)=
         let items=map(|i|map(|j|((i+j)%4,i*6+j),iota(6)),iota(4)) in
         bucket_scatter_2d(dest,items)",
        |graph| {
            let query = Query(graph);
            let function = test_graph::entry(graph);
            let [operation] = top_level(graph, function, "BucketScatter")[..] else {
                panic!("one bucket scatter");
            };
            let Code::BucketScatter { coordinates, depth } = query.0.code(operation).unwrap() else {
                panic!("bucket payload");
            };
            assert_eq!(coordinates, [vec![0], vec![1]]);
            assert_eq!(depth, 2);
            assert_eq!(query.0.input_count(operation), 3);
        },
    );
}

#[test]
fn rectangular_maps_expand_launch_without_expanding_row_capacity() {
    let source = "entry main(xs:[4]i32) [4][3]i32=map(|x|map(|y|x+y,iota(3)),xs)";
    distributed_cases(&[source], |_, graph, function| {
        let query = Query(graph);
        let parallel = top_level(graph, function, "Parallel");
        assert_eq!(parallel.len(), 1);
        assert_eq!(
            sizes(query, parallel[0]),
            [4, 3],
            "flattening must keep the logical row and column extents"
        );
        assert_eq!(
            array_extents(query, infos(query, parallel[0])[0]).unwrap(),
            [Some(4), Some(3)],
            "the output retains its logical row capacity"
        );
    });
}

#[test]
fn rectangular_planning_and_emission_use_fused_structure() {
    let tlc = compile_thru_tlc(
        "entry main(xs:[4]i32) [4][3]i32=
         map(|x|map(|y|x+y,map(|y|y+1,iota(3))),map(|x|x*2,xs))",
    )
    .unwrap();
    let mut program = place(
        optimize_for_test(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    super::schedule(&mut program.graph).unwrap();
    let query = Query(&program.graph);
    let function = test_graph::entry(&program.graph);
    let ranks = top_level(&program.graph, function, "Parallel")
        .into_iter()
        .map(|operation| rank(query, operation))
        .collect::<Vec<_>>();
    assert_eq!(ranks, [2]);
    let program = schedule(program).unwrap();
    let program = super::super::prepare_interfaces(program).unwrap();
    finish_for_test(program, CodegenTarget::Wgsl).unwrap();
}

#[test]
fn rectangular_candidates_retain_expensive_but_legal_recomputation() {
    use crate::egglog::payload::Payload;
    use crate::op::BinaryOperator;
    let mut prefix = String::new();
    let mut codes = vec![
        (400, Code::Parallel),
        (200, Code::Parallel),
        (201, Code::Scalar(Payload::Binary(BinaryOperator::Add))),
        (
            900,
            Code::Constant {
                ty: crate::types::i32(),
                bits: 2,
            },
        ),
    ];
    for i in 1_i64..=80 {
        codes.push((i, Code::Scalar(Payload::Binary(BinaryOperator::Multiply))));
        prefix.push_str(&format!("(set (FixtureInput (OperationId {i}) 0) (ValueNameRef (ValueId {}))) (set (FixtureInput (OperationId {i}) 1) (ValueNameRef (ValueId 1000))) (set (FixtureOutput (OperationId {i}) 0) (ValueNameRef (ValueId {i}))) (set (ValueType (ValueId {i})) scalar) (set (FixtureRegion (OperationId {i})) (RegionId 1)) (set (FixtureOrder (OperationId {i})) {}) (RepeatableOperation (OperationId {i})) (DiscardableOperation (OperationId {i}))",i-1,i-1));
    }
    let mut graph = test_graph::fixture(
        &codes,
        &include_str!("tests/distribution_cost.egg").replace("PREFIX", &prefix),
    )
    .unwrap();
    analysis::run(&mut graph).unwrap();
    super::super::fusion::run(&mut graph).unwrap();
    super::place(
        &mut graph,
        PipelineTopologyPolicy::AllowGenerated,
        super::CostModel::default(),
    )
    .unwrap();
    let query = Query(&graph);
    let mut legal = 0;
    query
        .for_each("DistributionPair", |row| {
            assert!(query.flag("DistributionSafe", (row[0], row[1]))?);
            let body = query.0.control_region(row[0], 0_i64)?;
            assert_eq!(
                analysis::count(query, "RegionOperationCount", body)? - 1,
                80,
                "expensive repeated prefix"
            );
            assert!(!query.contains("DistributeNest", (row[0], row[1]))?);
            legal += 1;
            Ok(())
        })
        .unwrap();
    assert!(legal > 0, "cost must not remove a legal distribution candidate");
    let statements = top_level(&graph, test_graph::entry(&graph), "Parallel");
    assert_eq!(statements.len(), 1);
    assert_eq!(
        rank(query, statements[0]),
        1,
        "ample outer parallelism should avoid repeating the expensive prefix"
    );
}

#[test]
fn rectangular_maps_select_arbitrary_rank_and_preserve_sequential_prefixes() {
    let cases = [
        ("entry main(xs:[]i32) [][3]i32=map(|x|map(|y|x+y,iota(3)),map(|x|x+1,xs))", 2),
        ("entry main(xs:[]i32) [][3]i32=map(|x|map(|y|x+y,map(|y|y+1,iota(3))),xs)", 2),
        ("entry main(xs:[]i32) [][3]i32=map(|x|map(|y|x+y,iota(3)),map(|x|loop a=x for i<16 do a+i,xs))", 1),
        ("entry main(xs:[]i32) [][3]i32=map(|x|let t=x*x in map(|y|t+y,iota(3)),xs)", 2),
        ("entry main(xs:[][3]i32) [][3]i32=map(|row|map(|x|x+1,row),xs)", 2),
        ("entry main(xs:[2]i32) [2][3][4]i32=map(|x|map(|y|map(|z|x+y+z,iota(4)),iota(3)),xs)", 3),
        ("entry main(xs:[2]i32) [2][3][4][5]i32=map(|x|map(|y|map(|z|map(|w|x+y+z+w,iota(5)),iota(4)),iota(3)),xs)", 4),
        ("entry main(xs:[2]i32) [2][0]i32=map(|x|map(|y|x+y,iota(0)),xs)", 2),
        ("entry main(xs:[]i32) [][3]i32=map(|x|let t=loop a=x for i<16 do a+i in map(|y|t+y,iota(3)),xs)", 1),
        ("entry main(xs:[]i32) [][3]i32=map(|x|let t=reduce(|a,b|a+b,0,[x,x,x]) in map(|y|t+y,iota(3)),xs)", 1),
    ];
    check_ranks(&cases);
}

#[test]
fn nested_collectives_are_costed_as_part_of_the_lane_body() {
    check_ranks(&[(
        "entry main(xs:[]i32) [][3]i32=map(|x|map(|y|reduce(|a,b|a+b,0,[x,y]),iota(3)),xs)",
        2,
    )]);
}

#[test]
fn conditional_distribution_preserves_empty_and_partial_evaluation() {
    let cases = [
        ("entry main(xs:[3]i32) [3][4]i32=map(|x|if x>0 then map(|y|100/(x+y),iota(4)) else map(|y|x+y,iota(4)),xs)", 2),
        ("entry main(xs:[3]i32) [3][4]i32=map(|x|if 1/x>0 then map(|y|x+y,iota(4)) else map(|y|x-y,iota(4)),xs)", 1),
        ("entry main(xs:[3]i32) [3][0]i32=map(|x|if x>0 then map(|y|1/x+y,iota(0)) else map(|y|x-y,iota(0)),xs)", 2),
        ("entry main(xs:[3]i32) [3][0]i32=map(|x|let offset=1/x in map(|y|offset+y,iota(0)),xs)", 1),
    ];
    check_ranks(&cases);
}

fn check_ranks(cases: &[(&str, usize)]) {
    let sources = cases.iter().map(|case| case.0).collect::<Vec<_>>();
    distributed_cases(&sources, |index, graph, function| {
        let (source, expected_rank) = cases[index];
        let query = Query(graph);
        let ranks = top_level(graph, function, "Parallel")
            .into_iter()
            .map(|operation| rank(query, operation))
            .collect::<Vec<_>>();
        assert_eq!(ranks, [expected_rank], "{source}");
    });
}

#[test]
fn fused_outputs_traverse_independent_rectangular_domains() {
    let sources = [3, 5].map(|inner| {
        format!(
            "entry main(xs:[4]i32) ([4][3]i32,[4][{inner}]i32)=
        (map(|x|map(|y|x+y,iota(3)),xs), map(|x|map(|y|x*y,iota({inner})),xs))"
        )
    });
    fused_cases(&sources, |index, graph, function| {
        let query = Query(graph);
        let parallels = top_level(graph, function, "Parallel");
        assert_eq!(
            parallels.len(),
            1,
            "case {index}: both outputs share their outer traversal"
        );
        let mut shapes = infos(query, parallels[0])
            .into_iter()
            .filter_map(|info| array_extents(query, info))
            .collect::<Vec<_>>();
        shapes.sort();
        assert_eq!(
            shapes,
            [vec![Some(4), Some(3)], vec![Some(4), Some([3, 5][index])]]
        );
    });
}

#[test]
fn rectangular_runtime_dimensions_must_be_uniform_across_outer_coordinates() {
    let sources = [
        "entry main(xs:[]i32,n:i32) [][]i32=map(|x|map(|y|x+y,iota(n)),xs)",
        "entry main(xs:[]i32,ys:[]i32) [][]i32=map(|x|map(|y|x+y,ys),xs)",
        "entry main(xs:[]i32,n:i32,m:i32) [][][]i32=map(|x|map(|y|map(|z|x+y+z,iota(m)),iota(n)),xs)",
        "entry main(xs:[]i32,n:i32) [][]i32=map(|x|map(|y|x+y,iota(n+1)),xs)",
    ];
    planned_cases(&sources, |index, graph, function| {
        let query = Query(graph);
        let root = query.0.function_root(function).unwrap();
        let output = query.0.yield_value(root, 0_i64).unwrap();
        let ranks = top_level(graph, function, "Parallel")
            .into_iter()
            .filter(|&operation| query.0.outputs(operation).contains(&output))
            .map(|operation| rank(query, operation))
            .collect::<Vec<_>>();
        assert_eq!(
            ranks,
            [[2].as_slice(), &[2], &[3], &[2]][index],
            "{}",
            sources[index]
        );
        assert!(
            top_level(graph, function, "Loop").is_empty(),
            "planning must not manufacture a source loop to select array storage"
        );
    });
}

#[test]
fn segmented_collectives_select_workgroups_for_each_outer_coordinate() {
    planned_cases(
        &[
            "entry main(xs:[4][513]i32) []i32=map(|row|reduce((+),0,row),xs)",
            "entry main(xs:[4][513]i32) [][513]i32=map(|row|scan((+),0,row),xs)",
            "entry main(xs:[4][3][17]i32) [][3]i32=map(|rows|map(|row|reduce((+),0,row),rows),xs)",
        ],
        |index, graph, function| {
            let query = Query(graph);
            let kernels = top_level(graph, function, "Parallel");
            assert_eq!(kernels.len(), 1, "case {index}: one segmented kernel");
            let operation = kernels[0];
            assert_eq!(rank(query, operation), [2, 2, 3][index]);
            assert!(!folds(query, operation).is_empty());
            let execution = query
                .fields(
                    "DeviceKernel",
                    query.required("OperationExecution", operation).unwrap(),
                )
                .unwrap();
            assert!(query.enode("AutomaticGrid", execution[0]).unwrap().is_some());
            let stages = dispatches(graph, function);
            assert_eq!(stages.len(), 1);
            assert!(stages[0].width.get() > 1);
        },
    );
}

#[test]
fn indexed_reducers_share_native_recognition_and_keep_cas_fallback() {
    for (reducer, expected) in [
        ("|a:i32,b:i32|a+b", "AtomicAdd"),
        ("|a:i32,b:i32|b+a", "AtomicAdd"),
        ("|a:i32,b:i32|a&b", "AtomicAnd"),
        ("|a:i32,b:i32|a|b", "AtomicOr"),
        ("|a:i32,b:i32|a^b", "AtomicXor"),
        ("|a:i32,b:i32|a+a", "AtomicCas"),
        ("|a:i32,b:i32|a-b", "AtomicCas"),
        ("|a:i32,b:i32|a+b+k", "AtomicCas"),
        ("|a:i32,b:i32|let unused=1/k in a+b", "AtomicCas"),
        ("(+)", "AtomicAdd"),
        ("combine", "AtomicAdd"),
    ] {
        let source = format!("def combine(a:i32,b:i32) i32 = a+b\nentry main(dest:*[3]i32,xs:[5]i32,k:i32) [3]i32 = reduce_by_index(dest,{reducer},0,map(|x:i32|x%3,xs),xs)");
        planned(&source, |graph| {
            let query = Query(graph);
            let mut checked = 0;
            for &(_, combine) in graph.structure().read().combiners.values().flatten() {
                let native = query
                    .lookup("NativeBinaryReducer", combine)
                    .unwrap()
                    .map(|value| read(graph, value).spelling().into_owned());
                let opcode = match expected {
                    "AtomicAdd" => Some("+"),
                    "AtomicAnd" => Some("&"),
                    "AtomicOr" => Some("|"),
                    "AtomicXor" => Some("^"),
                    _ => None,
                };
                assert_eq!(native.as_deref(), opcode, "{reducer}");
                checked += 1;
            }
            assert_eq!(checked, 1, "one logical indexed combiner: {reducer}");
        });
    }
}

#[test]
fn subgroup_addition_is_opt_in_and_uses_small_segments() {
    let tlc =
        compile_thru_tlc("entry main(xs:[4][32]u32) [4][32]u32 = map(|row|scan((+),0u32,row),xs)").unwrap();
    let mut portable_width = None;
    for enabled in [false, true] {
        let placed = place(
            optimize_for_test(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap();
        let program = schedule_with_subgroups(placed, enabled).unwrap();
        let function = test_graph::entry(&program.graph);
        let stages = dispatches(&program.graph, function);
        assert_eq!(stages.len(), 1);
        let width = stages[0].width.get();
        if enabled {
            assert_eq!(width, 32, "subgroup launch follows the fixed row width");
            assert!(width <= portable_width.unwrap());
        } else {
            portable_width = Some(width);
        }
        let program = super::super::prepare_interfaces(program).unwrap();
        let ssa = finish_for_test(program, CodegenTarget::Wgsl).unwrap();
        let wgsl = crate::lower_ssa_to_wgsl(ssa).unwrap();
        assert_eq!(wgsl.contains("subgroupInclusiveAdd("), enabled);
    }
}

#[test]
fn loop_placement_accounts_for_iteration_work_before_storage() {
    for (source, dispatched, hoisted_reductions) in [
        ("entry main(xs:[4]i32) [4]i32=loop ys=xs for i<128 do map(|x|x+1,ys)", false, 0),
        ("entry main(xs:[4]i32) ([4]i32,i32)=loop (ys,i)=(xs,0) while i<128 do (map(|x|x+1,ys),i+1)", false, 0),
        ("entry main(xs:[16]i32) ([16]i32,i32)=loop (ys,i)=(xs,0) while i<128 do (scan((+),0,ys),i+1)", false, 0),
        ("entry main(xs:[4097]i32) ([4097]i32,i32)=loop (ys,i)=(xs,0) while i<128 do (map(|x|x+1,ys),i+1)", true, 0),
        ("entry main(xs:[]i32) ([]i32,i32)=loop (ys,i)=(map(|x|x,xs),0) while i<128 do (map(|x|x+1,ys),i+1)", true, 0),
        ("entry main(xs:[4]i32,big:[65536]i32) ([4]i32,i32)=loop (ys,i)=(xs,0) while i<128 do (map(|x|x+reduce((+),0,big),ys),i+1)", false, 1),
    ] {
        placed_cases(&[source], |_, graph, function| {
            let query=Query(graph);let loops=within(graph,function).into_iter().filter(|&operation|code_is(query,operation,"Loop")).collect::<Vec<_>>();
            assert_eq!(loops.len(),1,"{source}");let operation=loops[0];let body=query.0.control_region(operation, 0_i64).unwrap();
            assert_eq!(query.flag("HostRegion",body).unwrap(),dispatched,"{source}");assert_eq!(query.flag("LocalLoop",operation).unwrap(),!dispatched,"{source}");
            let reductions=top_level(graph,function,"Parallel").into_iter().map(|operation|folds(query,operation).len()).sum::<usize>();
            assert_eq!(reductions,hoisted_reductions,"the invariant reduction must execute once before the local loop: {source}");
        });
    }
}

#[test]
fn nested_loop_work_uses_constant_trip_counts() {
    for (bound, dispatched) in [("0", false), ("2", false), ("65536", true), ("limit", true)] {
        let source = format!(
            "entry main(xs:[4]i32,limit:i32) ([4]i32,i32) =
             loop (ys,i)=(xs,0) while i<128 do
               let next=loop zs=ys for j<{bound} do map(|x|x+1,zs) in (next,i+1)"
        );
        placed_cases(&[&source], |_, graph, function| {
            let query = Query(graph);
            let loops = within(graph, function)
                .into_iter()
                .filter(|&operation| query.contains("WhileLoopOperation", operation).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(loops.len(), 1);
            let body = query.0.control_region(loops[0], 0_i64).unwrap();
            assert_eq!(query.flag("HostRegion", body).unwrap(), dispatched, "{source}");
        });
    }
}

// Compile related cases once; keep every assertion tied to its source entry.

#[test]
fn fusion_compares_logical_shapes_of_independent_inputs() {
    let sources = [17, 19].map(|size| {
        format!("entry main(xs:[17]i32,ys:[{size}]i32) (i32,i32)=(reduce((+),0,xs),reduce((+),0,ys))")
    });
    fused_cases(&sources, |index, graph, function| {
        let query = Query(graph);
        let parallels = top_level(graph, function, "Parallel");
        assert_eq!(parallels.len(), if index == 0 { 1 } else { 2 });
        let count = parallels.into_iter().map(|operation| folds(query, operation).len()).sum::<usize>();
        assert_eq!(count, 2, "fusion must retain both reductions");
    });
}

#[test]
fn update_handles_and_reservation_results_cannot_escape_their_scope() {
    let tlc = compile_thru_tlc("entry main(dest:*[2]i32) [2]i32=scatter(dest,[0,1],[7,9])").unwrap();
    for slot in [0_i64, 1] {
        let mut graph = fuse(from_tlc(&tlc).unwrap()).unwrap().graph;
        let query = Query(&graph);
        let mut operation = None;
        query
            .for_each("LiveOperation", |row| {
                if matches!(query.0.code(row[0])?, Code::Accumulate { .. }) {
                    operation = Some(row[0]);
                }
                Ok(())
            })
            .unwrap();
        let operation = operation.expect("indexed update occurrence");
        graph
            .update(|mut sink| {
                let missing = sink.add(if slot == 0 { "AccumulatorId" } else { "ValueId" }, 999999_i64)?;
                let name = sink.add(if slot == 0 { "AccNameRef" } else { "ValueNameRef" }, missing)?;
                sink.set_input(operation, slot, name)
            })
            .unwrap();
        let error = super::super::memory::validate(&mut graph).unwrap_err();
        assert!(error.to_string().contains("outside its scope"), "{error}");
    }
}

#[test]
fn calibrated_costs_select_among_the_same_legal_map_executions() {
    let tlc = compile_thru_tlc("entry main(xs:[4][513]i32) [4]i32=map(|row|reduce((+),0,row),xs)").unwrap();
    for (barrier_ns, expected_rank) in [(0.0, 2), (1e9, 1)] {
        let costs = super::super::cost::CostModel {
            barrier_ns,
            ..Default::default()
        };
        let placed = super::super::place_with_cost_model(
            optimize_for_test(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
            costs,
        )
        .unwrap();
        let program = super::super::schedule_with_cost_model(placed, false, costs).unwrap();
        let query = Query(&program.graph);
        let function = test_graph::entry(&program.graph);
        let [kernel] = top_level(&program.graph, function, "Parallel")[..] else {
            panic!("one map kernel");
        };
        assert_eq!(rank(query, kernel), expected_rank, "barrier_ns={barrier_ns}");
        let program = super::super::prepare_interfaces(program).unwrap();
        finish_for_test(program, CodegenTarget::Wgsl).unwrap();
    }
}

#[test]
fn scan_epilogues_require_pointwise_reads_and_completed_captures() {
    use crate::egglog::payload::Payload;
    use crate::op::BinaryOperator;
    for (index, bias, opcode, expected) in [
        (11, 1001, BinaryOperator::Add, true),
        (1000, 1001, BinaryOperator::Add, false),
        (11, 3, BinaryOperator::Add, true),
        (11, 1000, BinaryOperator::Divide, false),
    ] {
        let facts = include_str!("tests/scan_epilogue.egg")
            .replace("INDEX", &format!("(ValueNameRef (ValueId {index}))"))
            .replace("BIAS", &format!("(ValueNameRef (ValueId {bias}))"))
            .replace(
                "OPCODE_EFFECTS",
                if opcode == BinaryOperator::Divide {
                    "(OrderedOperation (OperationId 5))"
                } else {
                    "(DiscardableOperation (OperationId 5)) (RepeatableOperation (OperationId 5))"
                },
            );
        let mut graph = test_graph::fixture(
            &[
                (
                    90,
                    Code::Constant {
                        ty: crate::types::i32(),
                        bits: 0,
                    },
                ),
                (
                    91,
                    Code::Constant {
                        ty: crate::types::i32(),
                        bits: 1,
                    },
                ),
                (0, Code::Parallel),
                (1, Code::Index),
                (3, Code::Parallel),
                (4, Code::Index),
                (5, Code::Scalar(Payload::Binary(opcode))),
                (20, Code::Scalar(Payload::Binary(BinaryOperator::Add))),
            ],
            &facts,
        )
        .unwrap();
        analysis::run(&mut graph).unwrap();
        super::epilogues(&mut graph).unwrap();
        assert_eq!(
            Query(&graph).first("ScanEpilogue").unwrap().is_some(),
            expected,
            "index={index} bias={bias} opcode={opcode:?}"
        );
    }
}

fn code_is(query: Query<'_>, operation: Value, kind: &str) -> bool {
    let code = query.0.code(operation).unwrap();
    matches!(
        (kind, code),
        ("Parallel", Code::Parallel)
            | ("Branch", Code::Branch)
            | ("Loop", Code::CountedLoop | Code::WhileLoop)
            | ("View", Code::View)
            | ("BucketScatter", Code::BucketScatter { .. })
    )
}
fn rank(query: Query<'_>, operation: Value) -> usize {
    query.0.domain_rank(query.0.operation_domain(operation).unwrap()).unwrap() as usize
}
fn sizes(query: Query<'_>, operation: Value) -> Vec<i64> {
    let domain = query.0.operation_domain(operation).unwrap();
    (0..rank(query, operation) as i64)
        .map(|slot| {
            query.0.value_to_base::<i64>(
                query.required("StaticIndex", query.0.domain_extent(domain, slot).unwrap()).unwrap(),
            )
        })
        .collect()
}
fn infos(query: Query<'_>, operation: Value) -> Vec<Value> {
    query
        .0
        .outputs(operation)
        .into_iter()
        .map(|name| query.required("ValueType", query.name(name).unwrap()).unwrap())
        .collect()
}
fn folds(query: Query<'_>, operation: Value) -> Vec<Value> {
    (0..query.0.result_count(operation))
        .filter_map(|slot| {
            query
                .enode("Folded", query.0.result_kind(operation, slot).unwrap())
                .unwrap()
                .map(|fields| fields[0])
        })
        .collect()
}
fn within(graph: &Graph, function: Value) -> Vec<Value> {
    let query = Query(graph);
    let root = query.0.function_root(function).unwrap();
    let mut operations = Vec::new();
    query
        .for_each("LiveOperation", |row| {
            if query.contains("FlowScopeAncestor", (root, query.0.operation_region(row[0])?))? {
                operations.push(row[0]);
            }
            Ok(())
        })
        .unwrap();
    operations
}

#[test]
fn indexed_literal_bounds_require_every_element() {
    for (elements, expected) in [("[-3,7,0]", Some((-3, 7))), ("[-3,n,7]", None)] {
        let source = compile_thru_tlc(&format!("entry main(i:i32,n:i32) i32={elements}[i]")).unwrap();
        let mut program = from_tlc(&source).unwrap();
        analysis::run(&mut program.graph).unwrap();
        super::analyze(&mut program.graph).unwrap();
        let query = Query(&program.graph);
        let mut indexed = 0;
        query
            .for_each("IndexOperation", |row| {
                let value = query.0.output(row[0], 0_i64)?;
                let low = query.lookup("IntegerLower", value)?.map(|v| query.0.value_to_base::<i64>(v));
                let high = query.lookup("IntegerUpper", value)?.map(|v| query.0.value_to_base::<i64>(v));
                assert_eq!(low.zip(high), expected);
                indexed += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(indexed, 1);
    }
}

#[test]
fn collection_shape_proofs_distinguish_extents_from_capacity() {
    let mut graph = test_graph::fixture(
        &[(0, Code::Parallel), (1, Code::Uninitialized)],
        r#"
        (set (FixtureOperationShape (OperationId 1)) (vec-of (IndexConstant 2)))
        (set (FixtureOutput (OperationId 1) 0) (ValueNameRef (ValueId 30)))
        (LiveOperation (OperationId 0))
        (set (FixtureOperationDomain (OperationId 0)) (DomainId 0))
        (set (FixtureDomainRank (DomainId 0)) 2)
        (set (FixtureDomainExtent (DomainId 0) 0) (IndexConstant 8))
        (set (FixtureDomainExtent (DomainId 0) 1) (IndexConstant 4))
        (set (FixtureOutput (OperationId 0) 0) (ValueNameRef (ValueId 0)))
        (set (FixtureOutputRole (OperationId 0) 0) (CollectedOutput))
        (set (FixtureOutputClause (OperationId 0) 0) 0)
        (set (FixtureOutputField (OperationId 0) 0) 0)
        (set (FixtureResultKindOf (OperationId 0) 0) (Collected))
        (set (FixtureResultValue (OperationId 0) 0 0) (ValueNameRef (ValueId 10)))
        (set (FixtureOutput (OperationId 0) 1) (ValueNameRef (ValueId 1)))
        (set (FixtureOutputRole (OperationId 0) 1) (CompactedOutput))
        (set (FixtureOutput (OperationId 0) 2) (ValueNameRef (ValueId 2)))
        (set (FixtureOutputRole (OperationId 0) 2) (TotalOutput))
        (set (FixtureOutputClause (OperationId 0) 2) 2)
        (set (FixtureOutputField (OperationId 0) 2) 0)
        (set (FixtureResultReducer (OperationId 0) 2) (RegionId 2))
        (ReducerPreservesOuterLength (RegionId 2) 0)
        (set (FixtureResultNeutral (OperationId 0) 2 0) (ValueNameRef (ValueId 20)))
        (set (FixtureOutput (OperationId 0) 3) (ValueNameRef (ValueId 3)))
        (set (FixtureOutputRole (OperationId 0) 3) (UpdatedOutput))
        (run-schedule (saturate graph-sizes))
        (check (= (CollectionRank (ValueId 0)) 2))
        (check (= (CollectionExtent (ValueId 0) 1) (IndexConstant 4)))
        (check (= (CollectionElement (ValueId 0)) (ValueId 10)))
        (check (= (CollectionRank (ValueId 1)) 2))
        (check (= (CollectionRank (ValueId 2)) 1))
        (check (= (CollectionElement (ValueId 2)) (ValueId 20)))
    "#,
    )
    .unwrap();
    let query = Query(&graph);
    let scratch = query.constructor("ValueId", 30_i64).unwrap().unwrap();
    let extent = query.required("LogicalOuterCount", scratch).unwrap();
    assert_eq!(
        graph.value_to_base::<i64>(query.required("StaticIndex", extent).unwrap()),
        2
    );

    let compact = query.constructor("ValueId", 1_i64).unwrap().unwrap();
    let updated = query.constructor("ValueId", 3_i64).unwrap().unwrap();
    assert!(query.lookup("LogicalOuterCount", compact).unwrap().is_none());
    assert!(query.lookup("CollectionRank", updated).unwrap().is_none());
    let previous = crate::egglog::analysis::scopes::seed_fixture(&mut graph).unwrap();
    crate::egglog::analysis::scopes::retire(&mut graph, &[previous]).unwrap();
    super::analyze(&mut graph).unwrap();
    assert!(
        Query(&graph).lookup("CollectionRank", compact).unwrap().is_none(),
        "invalidated proofs must not survive a dead producer"
    );
}

#[test]
fn collected_shape_does_not_require_unrelated_result_constructors() {
    test_graph::fixture(
        &[(0, Code::Parallel)],
        r#"
        (LiveOperation (OperationId 0))
        (set (FixtureOperationDomain (OperationId 0)) (DomainId 0))
        (set (FixtureDomainRank (DomainId 0)) 2)
        (set (FixtureDomainExtent (DomainId 0) 1) (IndexConstant 3))
        (set (FixtureOutput (OperationId 0) 0) (ValueNameRef (ValueId 0)))
        (set (FixtureOutputRole (OperationId 0) 0) (CollectedOutput))
        (run-schedule (saturate graph-sizes))
        (check (= (CollectionRank (ValueId 0)) 2))
        (check (= (CollectionExtent (ValueId 0) 1) (IndexConstant 3)))
    "#,
    )
    .unwrap();
}

#[test]
fn element_update_snapshots_require_no_escaping_copy() {
    for (escape, safe) in [
        ("", true),
        (
            "(set (FixtureYield (RegionId 0) 0) (ValueNameRef (ValueId 1)))",
            false,
        ),
        (
            "(LiveOperation (OperationId 3)) (OperationUse (OperationId 3) (ValueNameRef (ValueId 1)))",
            false,
        ),
    ] {
        let facts = format!(
            r#"
            (LiveOperation (OperationId 0)) (LiveOperation (OperationId 1)) (LiveOperation (OperationId 2))
            (set (FixtureOutput (OperationId 0) 0) (ValueNameRef (ValueId 1)))
            (set (FixtureInput (OperationId 1) 0) (ValueNameRef (ValueId 1)))
            (set (FixtureOperationSlices (OperationId 1)) (vec-of (At (IndexConstant 0))))
            (OperationUse (OperationId 1) (ValueNameRef (ValueId 1)))
            (OperationUse (OperationId 2) (ValueNameRef (ValueId 1)))
            {escape}
            (run-schedule (saturate reuse-reads) (saturate reuse-legality) (saturate reuse-select))
        "#
        );
        let graph = test_graph::fixture(
            &[
                (0, Code::Copy),
                (
                    1,
                    Code::Update {
                        bounds: Bounds::Proven,
                    },
                ),
                (2, Code::Index),
                (3, Code::Product),
            ],
            &facts,
        )
        .unwrap();
        let query = Query(&graph);
        let value = query.constructor("ValueId", 1_i64).unwrap().unwrap();
        assert_eq!(query.contains("CanSnapshotCopy", value).unwrap(), safe);
    }
}

#[test]
fn slice_lengths_do_not_require_a_source_capacity_bound() {
    let graph = test_graph::fixture(&[(0, Code::View)], r#"
        (set (FixtureInput (OperationId 0) 0) (ValueNameRef (ValueId 0)))
        (set (FixtureOutput (OperationId 0) 0) (ValueNameRef (ValueId 1)))
        (set (FixtureOperationSlices (OperationId 0)) (vec-of (Slice (IndexConstant 2) (IndexConstant 5) (IndexConstant 1))))
        (run-schedule (saturate graph-sizes))
        (check (= (LogicalOuterCount (ValueId 1)) (IndexConstant 5)))
    "#).unwrap();
    let query = Query(&graph);
    let source = query.constructor("ValueId", 0_i64).unwrap().unwrap();
    assert!(query.lookup("OuterCapacity", source).unwrap().is_none());
}

#[test]
fn speculation_requires_native_outputs_purity_and_bounded_work() {
    for (element, pure, work, expected) in [
        ("(ScalarElement (I32))", true, 4, true),
        ("(VectorElement (F16) 4)", true, 4, true),
        ("(ScalarElement (I64))", true, 1, false),
        ("(VectorElement (F32) 8)", true, 1, false),
        ("(ScalarElement (I32))", false, 1, false),
        ("(ScalarElement (I32))", true, 5, false),
    ] {
        let graph = test_graph::fixture(
            &[(0, Code::Branch)],
            &format!(
                r#"
            (LiveOperation (OperationId 0))
            (set (PureOperation (OperationId 0)) {pure})

            (set (FixtureOutput (OperationId 0) 0) (ValueNameRef (ValueId 0)))
            (set (ValueType (ValueId 0)) (ScalarInfo {element}))
            (set (FixtureControlRegion (OperationId 0) 0) (RegionId 0))
            (set (FixtureControlRegion (OperationId 0) 1) (RegionId 1))
            (set (RegionWork (RegionId 0)) (bigint {work}))
            (set (RegionWork (RegionId 1)) (bigint 0))
            (run-schedule (saturate physical-legality))
        "#
            ),
        )
        .unwrap();
        let query = Query(&graph);
        let operation = query.constructor("OperationId", 0_i64).unwrap().unwrap();
        let can_speculate = super::speculation_arms(query, operation)
            .unwrap()
            .is_some_and(|arms| super::small_arms(query, arms).unwrap());
        assert_eq!(can_speculate, expected, "{element}, pure={pure}, work={work}");
    }
}

#[test]
fn retry_safety_allows_reads_but_rejects_writes_and_traps() {
    let mut graph = test_graph::fixture(
        &[],
        r#"
        (LiveRegion (RegionId 0)) (LiveRegion (RegionId 1)) (LiveRegion (RegionId 2))
        (RegionRead (RegionId 0) (ExternalMemory))
        (RegionWrite (RegionId 1) (ExternalMemory))
        (RegionTrap (RegionId 2))
        (run-schedule (saturate physical-legality))
        (check (= (RetrySafe (RegionId 0)) true))
        (check (= (RetrySafe (RegionId 1)) false))
        (check (= (RetrySafe (RegionId 2)) false))
    "#,
    )
    .unwrap();
    let previous = analysis::scopes::seed_fixture(&mut graph).unwrap();
    analysis::scopes::retire(&mut graph, &[previous]).unwrap();
    let query = Query(&graph);
    let region = query.constructor("RegionId", 0_i64).unwrap().unwrap();
    assert!(
        query.lookup("RetrySafe", region).unwrap().is_none(),
        "invalidated eligibility cannot survive analysis refresh"
    );
}

#[test]
fn atomic_candidates_require_one_fresh_32_bit_scalar_array() {
    for (element, count, fresh, expected) in [
        ("(ScalarElement (I32))", 1, true, true),
        ("(ScalarElement (U32))", 1, true, true),
        ("(ScalarElement (F32))", 1, true, false),
        ("(ScalarElement (I64))", 1, true, false),
        ("(ScalarElement (I32))", 2, true, false),
        ("(ScalarElement (I32))", 1, false, false),
    ] {
        let fact = if fresh { "(FreshArray (ValueId 0))" } else { "" };
        let extra =
            if count == 2 { "(set (FixtureAccumulatorArray (OperationId 0) 0 1) (ValueId 1))" } else { "" };
        let graph = test_graph::fixture(&[], &format!(r#"
            (set (FixtureAccumulatorArray (OperationId 0) 0 0) (ValueId 0))
            {extra}
            (set (ValueType (ValueId 0)) (LogicalValue (ArrayType (ValueArray) (vec-of (FixedExtent 4)) (ElementValue {element}))))
            {fact}
            (run-schedule (saturate physical-legality))
        "#)).unwrap();
        assert_eq!(
            Query(&graph).first("AtomicAccumulatorArray").unwrap().is_some(),
            expected,
            "{element}, count={count}, fresh={fresh}"
        );
    }
}

#[test]
fn kernel_policies_distinguish_compaction_atomics_and_native_scans() {
    for (kind, device, rank, atomic, predicated, algorithm, grouping, native) in [
        (
            "(Collected)",
            true,
            1,
            false,
            false,
            "ParallelKernel",
            "WorkgroupGroups",
            false,
        ),
        (
            "(Compacted)",
            true,
            1,
            false,
            false,
            "CompactionKernel",
            "SingleGroup",
            false,
        ),
        ("(Compacted)", false, 1, false, false, "OrderedKernel", "", false),
        (
            "(Compacted)",
            true,
            2,
            false,
            false,
            "OrderedKernel",
            "SingleGroup",
            false,
        ),
        (
            "(Contributed)",
            true,
            1,
            false,
            false,
            "OrderedKernel",
            "SingleGroup",
            false,
        ),
        (
            "(Contributed)",
            true,
            1,
            true,
            false,
            "ParallelKernel",
            "WorkgroupGroups",
            false,
        ),
        (
            "(Folded (InclusiveScan))",
            true,
            1,
            false,
            false,
            "ParallelKernel",
            "ElementGroups",
            true,
        ),
        (
            "(Folded (InclusiveScan))",
            true,
            1,
            false,
            true,
            "OrderedKernel",
            "ElementGroups",
            true,
        ),
    ] {
        let mut graph = test_graph::fixture(
            &[(0, Code::Parallel)],
            &format!(
                r#"
            (LiveOperation (OperationId 0))
            (set (FixtureOperationDomain (OperationId 0)) (DomainId 0)) (set (FixtureDomainRank (DomainId 0)) {rank})
            (set (FixtureResultCount (OperationId 0)) 1) (set (FixtureResultKindOf (OperationId 0) 0) {kind})
            (set (FixtureResultValue (OperationId 0) 0 0) (AccNameRef (AccumulatorId 0)))
            (set (TokenCapability (AccumulatorId 0)) (TokenBindingAt (OperationId 1) 0))
            (set (FixtureResultReducer (OperationId 0) 0) (RegionId 1)) (LiveRegion (RegionId 1))
            (set (FixtureResultNeutral (OperationId 0) 0 0) (ValueNameRef (ValueId 10)))
            (set (ConstantValue (ValueNameRef (ValueId 10))) 0) (NativeAddition (RegionId 1))
            (set (SubgroupsEnabled) true)
        "#
            ),
        )
        .unwrap();
        super::load_execution(&mut graph).unwrap();
        let facts = format!(
            "{} {} {}",
            if device { "(EmittedDispatch (OperationId 0))" } else { "" },
            "",
            if predicated { "(set (FixtureResultPredicate (OperationId 0) 0) (ValueId 99))" } else { "" }
        );
        test_graph::facts_with_positions(
            &mut graph,
            &format!(
                r#"{facts}
            (run-schedule (saturate physical-legality) (saturate execution-features)
                (saturate execution-atomics) (saturate execution-policy) (saturate execution-select))"#
            ),
        )
        .unwrap();
        let query = Query(&graph);
        let operation = query.constructor("OperationId", 0_i64).unwrap().unwrap();
        let selected = query.required("SelectedKernelAlgorithm", (operation, atomic)).unwrap();
        assert!(
            query.enode(algorithm, selected).unwrap().is_some(),
            "{kind}, device={device}, rank={rank}, atomic={atomic}, predicated={predicated}"
        );
        assert_eq!(query.contains("UseNativeSubgroups", operation).unwrap(), native);
        assert_eq!(
            query.contains("CanChunkCollective", operation).unwrap(),
            kind == "(Collected)" || (kind.starts_with("(Folded") && !predicated)
        );
        if device {
            let selected = query.required("SelectedLaunchGrouping", (operation, atomic)).unwrap();
            assert!(query.enode(grouping, selected).unwrap().is_some());
            assert_eq!(
                graph.value_to_base::<i64>(query.required("LaunchRank", operation).unwrap()),
                rank - i64::from(native)
            );
        }
    }
}

#[test]
fn fixed_loop_states_require_every_array_component_to_preserve_length() {
    for preserved in [false, true] {
        let second = if preserved {
            "(LoopPreservesComponentLength (OperationId 0) 0 (FieldPath (RootPath) 1))"
        } else {
            ""
        };
        let mut graph = test_graph::fixture(
            &[],
            &format!(
                r#"
            (LoopValueCarry (OperationId 0) 0 (ValueId 10) (ValueId 20) (ValueId 30))
            (LoopArrayComponent (OperationId 0) 0 (FieldPath (RootPath) 0) (ValueId 11) (ValueId 21))
            (LoopArrayComponent (OperationId 0) 0 (FieldPath (RootPath) 1) (ValueId 12) (ValueId 22))
            (LoopPreservesComponentLength (OperationId 0) 0 (FieldPath (RootPath) 0))
            {second}
            (run-schedule (saturate physical-legality))
        "#
            ),
        )
        .unwrap();
        super::load_execution(&mut graph).unwrap();
        graph
            .parse_and_run_program(
                None,
                "(run-schedule (saturate execution-features) (saturate execution-atomics))",
            )
            .unwrap();
        assert_eq!(
            Query(&graph).first("FixedLoopState").unwrap().is_some(),
            preserved
        );
    }
}

#[test]
fn retained_call_work_composes_and_unknown_work_stays_unknown() {
    for (callee, expected) in [
        (
            "(set (FixtureFunctionRoot leaf) (RegionId 2)) (set (RecursiveFunction leaf) false)",
            Some(3),
        ),
        (
            "(set (FixtureFunctionRoot leaf) (RegionId 2)) (set (RecursiveFunction leaf) true)",
            None,
        ),
        ("(FixtureExternal leaf)", None),
    ] {
        let facts = format!(
            r#"
            (let middle (FunctionId 1 "middle")) (let leaf (FunctionId 2 "leaf"))
            (set (FixtureFunctionRoot middle) (RegionId 1)) (set (RecursiveFunction middle) false)
            {callee}
            (set (FixtureCallTarget (OperationId 0)) middle)
            (set (FixtureCallTarget (OperationId 1)) leaf)
            (set (FixtureRegion (OperationId 1)) (RegionId 1))
            (set (FixtureRegion (OperationId 2)) (RegionId 2))
            (set (FixtureOrder (OperationId 1)) 0)
            (set (FixtureOrder (OperationId 2)) 0)
            (LiveOperation (OperationId 0)) (LiveOperation (OperationId 1)) (LiveOperation (OperationId 2))
            (OperationWorkDemand (OperationId 0))
            "#,
        );
        let mut graph = test_graph::fixture(
            &[
                (0, Code::Call),
                (1, Code::Call),
                (
                    2,
                    Code::Constant {
                        ty: crate::types::i32(),
                        bits: 7,
                    },
                ),
            ],
            &facts,
        )
        .unwrap();
        let operation = Query(&graph).first("OperationWorkDemand").unwrap().unwrap()[0];
        super::work::solve(&mut graph).unwrap();
        let query = Query(&graph);
        let work = query
            .lookup("OperationWork", operation)
            .unwrap()
            .map(|value| u64::try_from(&*graph.value_to_base::<Z>(value)).unwrap());
        assert_eq!(work, expected, "{callee}");
    }
}
