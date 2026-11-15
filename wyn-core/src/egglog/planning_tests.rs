use super::super::query::Query;
use super::super::{finish_for_test, from_tlc, fuse, place, schedule, schedule_with_subgroups};
use crate::egglog::dataflow::ir::{self, Code};
use crate::egglog::payload::read;
use crate::egglog::{analysis, test_graph};
use crate::{compile_thru_tlc, CodegenTarget, PipelineTopologyPolicy};
use egglog_engine::{sort::S, EGraph, Value};
use egglog_engine::{Core, Write};

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
                format!("(set (InputCount (OperationId {})) 2) (set (Input (OperationId {}) 0) (ValueNameRef (ValueId {}))) (set (Input (OperationId {}) 1) (ValueNameRef (ValueId 10))) (set (OutputCount (OperationId {})) 1) (set (Output (OperationId {}) 0) (ValueNameRef (ValueId {}))) (set (ValueType (ValueId {})) scalar) (set (OperationRegion (OperationId {})) (RegionId {})) (set (OperationOrder (OperationId {})) 1) (DiscardableOperation (OperationId {})) (RepeatableOperation (OperationId {}))",base+1,base+1,base+2,base+1,base+1,base+1,base+3,base+3,base+1,base,base+1,base+1,base+1)
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
            analysis::count(query, "OutputCount", outer[0]).unwrap(),
            2,
            "both source results must survive the legality decision"
        );
    }
}

#[test]
fn product_loops_track_array_component_length_proofs() {
    for (source, capacity) in [
        ("entry main(steps:i32) []i32=let (ys,k)=loop (ys,k)=([0,0,0,0],0) while k<steps do (ys with [k%4]=k,k+1) in ys", Some(4)),
        ("entry main(steps:i32) []i32=let (ys,k)=loop (ys,k)=(replicate(4,0i32),0) while k<steps do (map(|i|if i<length(ys) then ys[i] else k,iota(length(ys)+1)),k+1) in ys", None),
    ] {
        let source = compile_thru_tlc(source).unwrap();
        let mut program = from_tlc(&source).unwrap();
        super::super::analysis::run(&mut program.graph).unwrap();
        super::analyze(&mut program.graph).unwrap();let query=Query(&program.graph);let mut components=0;
        query.for_each("LoopArrayComponent",|row| {
            let bound=query.lookup("OuterCapacity",row[3])?.map(|value|program.graph.value_to_base::<i64>(value));
            assert_eq!(bound,capacity);assert_eq!(query.contains("LoopPreservesComponentLength",(row[0],row[1],row[2]))?,capacity.is_some());components+=1;Ok(())
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
                for name in query.sequence("Output", row[0])? {
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
) -> super::super::Program<'_, super::super::Optimized> {
    let imported = from_tlc(source).unwrap();
    let fused = fuse(imported).unwrap();
    let placed = place(fused, PipelineTopologyPolicy::AllowGenerated).unwrap();
    let scheduled = schedule(placed).unwrap();
    let interfaces = super::super::prepare_interfaces(scheduled).unwrap();
    super::super::optimize_with_policy(interfaces, super::super::ScalarOptimization::Basic).unwrap()
}

fn planned(source: &str, check: impl FnOnce(&EGraph)) {
    let tlc = compile_thru_tlc(source).unwrap();
    let program = schedule(
        place(
            fuse(from_tlc(&tlc).unwrap()).unwrap(),
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
    check: impl Fn(usize, &EGraph, Value),
) {
    let query = Query(&program.graph);
    let mut seen = vec![false; count];
    query
        .for_function("EntryOrder", |keys, _| {
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

fn planned_cases(sources: &[impl AsRef<str>], check: impl Fn(usize, &EGraph, Value)) {
    for (index, source) in sources.iter().enumerate() {
        let tlc = compile_thru_tlc(&cases_source(std::slice::from_ref(source))).unwrap();
        let program = schedule(
            place(
                fuse(from_tlc(&tlc).unwrap()).unwrap(),
                PipelineTopologyPolicy::AllowGenerated,
            )
            .unwrap(),
        )
        .unwrap();
        check_entries(&program, 1, |_, graph, function| check(index, graph, function));
    }
}

fn placed_cases(sources: &[impl AsRef<str>], check: impl Fn(usize, &EGraph, Value)) {
    for (index, source) in sources.iter().enumerate() {
        let tlc = compile_thru_tlc(&cases_source(std::slice::from_ref(source))).unwrap();
        let program = place(
            fuse(from_tlc(&tlc).unwrap()).unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap();
        check_entries(&program, 1, |_, graph, function| check(index, graph, function));
    }
}

fn distributed_cases(sources: &[impl AsRef<str>], check: impl Fn(usize, &EGraph, Value)) {
    for (index, source) in sources.iter().enumerate() {
        let tlc = compile_thru_tlc(&cases_source(std::slice::from_ref(source))).unwrap();
        let mut program = place(
            fuse(from_tlc(&tlc).unwrap()).unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap();
        super::schedule(&mut program.graph, super::CostModel::default()).unwrap();
        check_entries(&program, 1, |_, graph, function| check(index, graph, function));
    }
}

fn fused_cases(sources: &[impl AsRef<str>], check: impl Fn(usize, &EGraph, Value)) {
    let tlc = compile_thru_tlc(&cases_source(sources)).unwrap();
    let program = fuse(from_tlc(&tlc).unwrap()).unwrap();
    check_entries(&program, sources.len(), check);
}

fn top_level(graph: &EGraph, function: Value, constructor: &str) -> Vec<Value> {
    let query = Query(graph);
    let root = query.required("FunctionRoot", function).unwrap();
    query
        .operations(root)
        .unwrap()
        .into_iter()
        .filter(|&operation| code_is(query, operation, constructor))
        .collect()
}

fn dispatches(graph: &EGraph, function: Value) -> Vec<super::super::execution::Dispatch> {
    let query = Query(graph);
    let entry = graph.value_to_base::<i64>(query.required("EntryOrder", function).unwrap());
    let execution = super::super::execution::Execution::select(graph).unwrap();
    execution.entries[&entry].iter().map(|stage| *execution.dispatch(*stage).unwrap()).collect()
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
            let root = query.required("FunctionRoot", function).unwrap();
            let result = query.required("RegionYield", (root, 0_i64)).unwrap();
            let mut checked = false;
            for outer in top_level(graph, function, "Parallel") {
                if !query.sequence("Output", outer).unwrap().contains(&result) {
                    assert!(
                        !query.contains("OperationTrap", outer).unwrap(),
                        "case {index}: only total independent producers may leave the callback"
                    );
                    continue;
                }
                let body = query.required("ControlRegion", (outer, 0_i64)).unwrap();
                assert!(
                    within(graph, function).into_iter().any(|operation| code_is(
                        query, operation, "Parallel"
                    ) && query
                        .required("OperationRegion", operation)
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
        &["entry main() [][]i32=let rows=map(|n|map(|i|i+n,iota(n)),[0,1,3,7]) in rows[1..3]"],
        |_, graph, function| {
            let query = Query(graph);
            let views = top_level(graph, function, "View");
            assert_eq!(views.len(), 1);
            let operation = views[0];
            let slices = query.vector(query.required("OperationSlices", operation).unwrap()).unwrap();
            let slice = query.fields("Slice", slices[0]).unwrap();
            assert_eq!(
                graph.value_to_base::<i64>(query.required("StaticIndex", slice[1]).unwrap()),
                2
            );
            let source = query.name(query.required("Input", (operation, 0_i64)).unwrap()).unwrap();
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
                    if let Code::Update { .. } =
                        ir::code(graph, query.required("OperationPayload", row[0])?)
                    {
                        let output = query.name(query.required("Output", (row[0], 0_i64))?)?;
                        let source = query.name(query.required("Input", (row[0], 0_i64))?)?;
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
            let initial =
                query.name(query.required("EdgeArgument", (operation, 0_i64, 0_i64)).unwrap()).unwrap();
            let output = query.name(query.required("Output", (operation, 0_i64)).unwrap()).unwrap();
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

fn local_capacities(graph: &EGraph) -> Vec<i64> {
    let mut graph = graph.clone();
    super::analyze(&mut graph).unwrap();
    let query = Query(&graph);
    let mut capacities = Vec::new();
    query
        .for_each("LiveOperation", |row| {
            if query.contains("ParallelOperation", row[0])?
                && query.enode("InvocationLocal", query.required("OperationExecution", row[0])?)?.is_some()
            {
                for name in query.sequence("Output", row[0])? {
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
        fuse(from_tlc(&tlc).unwrap()).unwrap(),
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
    let mut rows = Vec::new();
    for &table in super::super::dataflow::edit::OPERATION_CELLS {
        if (table.starts_with("Result") && table != "ResultCount")
            || ["Output", "OutputRole", "OutputClause", "OutputField"].contains(&table)
        {
            query
                .for_function(table, |keys, value| {
                    if keys[0] == operation {
                        rows.push((table, keys.to_vec(), value));
                    }
                    Ok(())
                })
                .unwrap();
        }
    }
    program
        .graph
        .update(|mut sink| {
            for (table, keys, _) in &rows {
                sink.remove(table, egglog_engine::RawValues(keys.clone()))?;
            }
            for (table, mut keys, mut value) in rows {
                let slot = sink.value_to_base::<i64>(keys[1]);
                keys[1] = sink.base_to_value(1_i64 - slot);
                if table == "OutputClause" {
                    value = sink.base_to_value(1_i64 - sink.value_to_base::<i64>(value));
                }
                sink.set(table, egglog_engine::RawValues(keys), value)?;
            }
            Ok(())
        })
        .unwrap();
    let ssa = finish_for_test(
        super::super::optimize_with_policy(
            super::super::prepare_interfaces(schedule(program).unwrap()).unwrap(),
            super::super::ScalarOptimization::Basic,
        )
        .unwrap(),
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
            let code = ir::code(graph, query.required("OperationPayload", operation).unwrap());
            if kind == "bucket-scatter" {
                if let Code::BucketScatter { coordinates, depth } = code {
                    assert_eq!(depth, expected_rank as usize);
                    let inputs = query.sequence("Input", operation).unwrap();
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
            let matches = (0..analysis::count(query, "ResultCount", operation).unwrap()).any(|clause| {
                let value = query.required("ResultKindOf", (operation, clause)).unwrap();
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
            let domain = query.required("OperationDomain", outer).unwrap();
            let outer_index = query.required("DomainAxis", (domain, 0_i64)).unwrap();
            let body = query.required("ControlRegion", (outer, 0_i64)).unwrap();
            let inner = query
                .operations(body)
                .unwrap()
                .into_iter()
                .find(|&operation| code_is(query, operation, "Parallel"))
                .expect("the inner map retains its captured outer value");
            let domain = query.required("OperationDomain", inner).unwrap();
            let inner_index = query.required("DomainAxis", (domain, 0_i64)).unwrap();
            assert_ne!(outer_index, inner_index, "nested axes need distinct binders");
            let lane = query.required("ControlRegion", (inner, 0_i64)).unwrap();
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
            let Code::BucketScatter { coordinates, depth } =
                ir::code(graph, query.required("OperationPayload", operation).unwrap())
            else {
                panic!("bucket payload");
            };
            assert_eq!(coordinates[0], [0, 1]);
            assert_eq!(depth, 2);
            let input = query.required("Input", (operation, 1_i64)).unwrap();
            let producer = top_level(graph, function, "Parallel")
                .into_iter()
                .find(|&operation| query.sequence("Output", operation).unwrap().contains(&input))
                .expect("bucket input producer");
            assert_eq!(sizes(query, producer), [4, 6]);
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
        fuse(from_tlc(&tlc).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    super::schedule(&mut program.graph, super::CostModel::default()).unwrap();
    let query = Query(&program.graph);
    let function = test_graph::entry(&program.graph);
    let ranks = top_level(&program.graph, function, "Parallel")
        .into_iter()
        .map(|operation| rank(query, operation))
        .collect::<Vec<_>>();
    assert_eq!(ranks, [2]);
    let program = schedule(program).unwrap();
    let program = super::super::optimize_with_policy(
        super::super::prepare_interfaces(program).unwrap(),
        super::super::ScalarOptimization::Basic,
    )
    .unwrap();
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
        prefix.push_str(&format!("(set (InputCount (OperationId {i})) 2) (set (Input (OperationId {i}) 0) (ValueNameRef (ValueId {}))) (set (Input (OperationId {i}) 1) (ValueNameRef (ValueId 1000))) (set (OutputCount (OperationId {i})) 1) (set (Output (OperationId {i}) 0) (ValueNameRef (ValueId {i}))) (set (ValueType (ValueId {i})) scalar) (set (OperationRegion (OperationId {i})) (RegionId 1)) (set (OperationOrder (OperationId {i})) {}) (RepeatableOperation (OperationId {i})) (DiscardableOperation (OperationId {i}))",i-1,i-1));
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
            let body = query.required("ControlRegion", (row[0], 0_i64))?;
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
        "entry main(xs:[]i32) [][]i32=map(|x|map(|y|x+y,iota(x)),xs)",
        "entry main(xs:[]i32,n:i32) [][]i32=map(|x|map(|y|x+y,iota(n+1)),xs)",
    ];
    planned_cases(&sources, |index, graph, function| {
        let query = Query(graph);
        let root = query.required("FunctionRoot", function).unwrap();
        let output = query.required("RegionYield", (root, 0_i64)).unwrap();
        let ranks = top_level(graph, function, "Parallel")
            .into_iter()
            .filter(|&operation| query.sequence("Output", operation).unwrap().contains(&output))
            .map(|operation| rank(query, operation))
            .collect::<Vec<_>>();
        assert_eq!(
            ranks,
            [[2].as_slice(), &[2], &[3], &[1], &[2]][index],
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
            query
                .for_function("AccumulatorCombiner", |_, combine| {
                    let native = query
                        .lookup("NativeBinaryReducer", combine)?
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
                    Ok(())
                })
                .unwrap();
            assert_eq!(checked, 1, "one logical indexed combiner: {reducer}");
        });
    }
}

#[test]
fn subgroup_addition_is_opt_in_and_uses_small_segments() {
    let tlc =
        compile_thru_tlc("entry main(xs:[4][32]u32) [4][32]u32 = map(|row|scan((+),0u32,row),xs)").unwrap();
    for enabled in [false, true] {
        let placed = place(
            fuse(from_tlc(&tlc).unwrap()).unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap();
        let program = schedule_with_subgroups(placed, enabled).unwrap();
        let function = test_graph::entry(&program.graph);
        let stages = dispatches(&program.graph, function);
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].width.get(), 32);
        let program = super::super::optimize_for_test(program).unwrap();
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
            assert_eq!(loops.len(),1,"{source}");let operation=loops[0];let body=query.required("ControlRegion",(operation,0_i64)).unwrap();
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
            let body = query.required("ControlRegion", (loops[0], 0_i64)).unwrap();
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
    let program = fuse(from_tlc(&tlc).unwrap()).unwrap();
    for slot in [0_i64, 1] {
        let mut graph = program.graph.clone();
        let query = Query(&graph);
        let mut operation = None;
        query
            .for_each("LiveOperation", |row| {
                if matches!(
                    ir::code(&graph, query.required("OperationPayload", row[0])?),
                    Code::Accumulate { .. }
                ) {
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
                sink.remove("Input", (operation, slot))?;
                sink.set("Input", (operation, slot), name)
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
            fuse(from_tlc(&tlc).unwrap()).unwrap(),
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
        let program =
            super::super::optimize_with_policy(program, super::super::ScalarOptimization::Basic).unwrap();
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
        (11, 3, BinaryOperator::Add, false),
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
        assert_eq!(Query(&graph).first("ScanEpilogue").unwrap().is_some(), expected);
    }
}

fn code_is(query: Query<'_>, operation: Value, kind: &str) -> bool {
    let code = ir::code(query.0, query.required("OperationPayload", operation).unwrap());
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
    analysis::count(
        query,
        "DomainRank",
        query.required("OperationDomain", operation).unwrap(),
    )
    .unwrap() as usize
}
fn sizes(query: Query<'_>, operation: Value) -> Vec<i64> {
    let domain = query.required("OperationDomain", operation).unwrap();
    (0..rank(query, operation) as i64)
        .map(|slot| {
            query.0.value_to_base::<i64>(
                query
                    .required(
                        "StaticIndex",
                        query.required("DomainExtent", (domain, slot)).unwrap(),
                    )
                    .unwrap(),
            )
        })
        .collect()
}
fn infos(query: Query<'_>, operation: Value) -> Vec<Value> {
    query
        .sequence("Output", operation)
        .unwrap()
        .into_iter()
        .map(|name| query.required("ValueType", query.name(name).unwrap()).unwrap())
        .collect()
}
fn folds(query: Query<'_>, operation: Value) -> Vec<Value> {
    (0..analysis::count(query, "ResultCount", operation).unwrap())
        .filter_map(|slot| {
            query
                .enode(
                    "Folded",
                    query.required("ResultKindOf", (operation, slot)).unwrap(),
                )
                .unwrap()
                .map(|fields| fields[0])
        })
        .collect()
}
fn within(graph: &EGraph, function: Value) -> Vec<Value> {
    let query = Query(graph);
    let root = query.required("FunctionRoot", function).unwrap();
    let mut operations = Vec::new();
    query
        .for_each("LiveOperation", |row| {
            if query.contains(
                "FlowScopeAncestor",
                (root, query.required("OperationRegion", row[0])?),
            )? {
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
                let value = query.required("Output", (row[0], 0_i64))?;
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
