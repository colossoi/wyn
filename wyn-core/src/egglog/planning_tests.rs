use super::super::query::Query;
use super::super::{finish_for_test, from_tlc, fuse, place, schedule, schedule_with_subgroups};
use crate::{compile_thru_tlc, CodegenTarget, PipelineTopologyPolicy};
use egglog_engine::{
    sort::{SetContainer, VecContainer, S},
    EGraph, Value,
};

#[test]
fn whole_row_fusion_preserves_conflicting_callback_effects() {
    let template = super::super::fusion::new_graph().unwrap();
    for (producer, consumer, fused) in [
        ("(set-of (SharedRead))", "(set-of (SharedRead))", true),
        ("(set-of (SharedWrite))", "(set-of (SharedRead))", false),
        ("(set-of (SharedRead))", "(set-of (SharedWrite))", false),
    ] {
        let mut graph = template.clone();
        let fixture = include_str!("tests/array_reduce_effects.egg")
            .replace("PRODUCER_EFFECTS", producer)
            .replace("CONSUMER_EFFECTS", consumer);
        graph.parse_and_run_program(None, &fixture).unwrap();
        super::super::memory::validate(&mut graph).unwrap();
        super::super::fusion::run(&mut graph).unwrap();
        super::super::memory::validate(&mut graph).unwrap();
        let query = Query(&graph);
        let program = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
        let function = query.vector(query.fields("MemoryProgram", program).unwrap()[0]).unwrap()[0];
        assert_eq!(
            top_level(&graph, function, "Parallel").len(),
            if fused { 1 } else { 2 }
        );
    }
}

#[test]
fn every_collective_callback_must_be_available_before_distribution() {
    let template = super::super::fusion::new_graph().unwrap();
    for blocked in [None, Some(20), Some(30)] {
        let mut fixture = include_str!("tests/collective_distribution.egg").to_owned();
        for (prefix, base) in [("FIRST", 20), ("SECOND", 30)] {
            let capture = if blocked == Some(base) {
                format!("(Let (StatementTag {}) (vec-of (BindValue (ValueId {}) (ScalarInfo (ScalarElement (I32))))) (vec-of) (PrimOp \"+\" (vec-of (Var (ValueId {})) (Var (ValueId 10)))))", base + 1, base + 3, base + 2)
            } else {
                String::new()
            };
            let result = base + if blocked == Some(base) { 3 } else { 2 };
            fixture = fixture
                .replace(&format!("{prefix}_CAPTURE"), &capture)
                .replace(&format!("{prefix}_RESULT"), &format!("(Var (ValueId {result}))"));
        }
        let mut graph = template.clone();
        graph.parse_and_run_program(None, &fixture).unwrap();
        super::super::analysis::run(&mut graph).unwrap();
        super::super::fusion::run(&mut graph).unwrap();
        super::place(
            &mut graph,
            PipelineTopologyPolicy::AllowGenerated,
            super::CostModel::default(),
        )
        .unwrap();
        let query = Query(&graph);
        let program = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
        let function = query.vector(query.fields("MemoryProgram", program).unwrap()[0]).unwrap()[0];
        let outer = top_level(&graph, function, "Parallel");
        assert_eq!(outer.len(), 1);
        let parallel = query.fields("Parallel", query.fields("Let", outer[0]).unwrap()[3]).unwrap();
        let domain = query.fields("IterationSpace", parallel[1]).unwrap();
        assert_eq!(
            query.vector(domain[1]).unwrap().len(),
            if blocked.is_some() { 1 } else { 2 }
        );
        assert_eq!(
            query.vector(query.fields("Let", outer[0]).unwrap()[1]).unwrap().len(),
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
        let query = Query(&program.graph);
        let current = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
        let mut components = 0;
        for function in query.vector(query.fields("MemoryProgram", current).unwrap()[0]).unwrap() {
            query.for_each("LoopArrayComponent", |row| {
                if row[0] != function { return Ok(()); }
                let bound = query.lookup("OuterCapacity", (function, row[4]))?
                    .map(|value| program.graph.value_to_base::<i64>(value));
                assert_eq!(bound, capacity);
                assert_eq!(query.contains("LoopPreservesComponentLength", (function, row[1], row[2], row[3]))?, capacity.is_some());
                components += 1;
                Ok(())
            }).unwrap();

        }
        assert!(components > 0, "exercise a projected array in a product carry");
    }
}

#[test]
fn sliced_local_collections_retain_their_source_capacity() {
    let source = compile_thru_tlc("entry main(xs:[][8]f32) []f32 = map(|row| let values=map(|x|x+1.0,row[1..i32(row[0])]) in reduce(|a,b|min(a,b),f32.highest,values)+reduce(|a,b|max(a,b),f32.lowest,values),xs)").unwrap();
    let mut program = selected(&source);
    super::super::analysis::run(&mut program.graph).unwrap();
    let query = Query(&program.graph);
    let current = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
    let function = query.vector(query.fields("MemoryProgram", current).unwrap()[0]).unwrap()[0];
    let root = query.constructor("FunctionRoot", (function,)).unwrap().unwrap();
    let mut collections = 0;
    query
        .for_each("ContainsStm", |row| {
            if row[0] != root {
                return Ok(());
            }
            let statement = query.fields("Let", row[1])?;
            let Some(parallel) = query.enode("Parallel", statement[3])? else {
                return Ok(());
            };
            if query.enode("InvocationLocal", parallel[0])?.is_none() {
                return Ok(());
            }
            for binding in query.vector(statement[1])? {
                let Some(binding) = query.enode("BindValue", binding)? else {
                    continue;
                };
                if query.enode("LogicalValue", binding[1])?.is_some() {
                    let capacity = query.required("OuterCapacity", (function, binding[0]))?;
                    assert_eq!(program.graph.value_to_base::<i64>(capacity), 8);
                    collections += 1;
                }
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(collections, 1);
    finish_for_test(program, CodegenTarget::Wgsl).unwrap();
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
    let root = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
    let fields = query.fields("MemoryProgram", root).unwrap();
    let mut seen = vec![false; count];
    for function in query.vector(fields[0]).unwrap() {
        let definition = query.fields("DefineFunction", function).unwrap();
        let name = query.fields("FunctionId", definition[0]).unwrap();
        let name = program.graph.value_to_base::<S>(name[1]);
        let Some(index) = name.as_str().strip_prefix("case_") else {
            continue;
        };
        let index: usize = index.parse().unwrap();
        seen[index] = true;
        check(index, &program.graph, function);
    }
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
    let function = query.fields("DefineFunction", function).unwrap();
    let body = query.fields("Block", function[3]).unwrap();
    query
        .vector(body[0])
        .unwrap()
        .into_iter()
        .filter(|statement| {
            let statement = query.fields("Let", *statement).unwrap();
            query.enode(constructor, statement[3]).unwrap().is_some()
        })
        .collect()
}

fn dispatches(graph: &EGraph, function: Value) -> Vec<super::super::execution::Dispatch> {
    let query = Query(graph);
    let name = query.fields("DefineFunction", function).unwrap()[0];
    let program = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
    let fields = query.fields("MemoryProgram", program).unwrap();
    let execution = super::super::execution::Execution::select(graph).unwrap();
    let entry = query
        .vector(fields[2])
        .unwrap()
        .into_iter()
        .find_map(|entry| {
            let entry = query.fields("EntryPoint", entry).unwrap();
            (entry[0] == name).then(|| graph.value_to_base::<i64>(entry[1]))
        })
        .unwrap();
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
                .filter(|statement| {
                    let statement = query.fields("Let", *statement).unwrap();
                    let parallel = query.fields("Parallel", statement[3]).unwrap();
                    let body = query.fields("ParallelBlock", parallel[2]).unwrap();
                    query
                        .vector(body[1])
                        .unwrap()
                        .into_iter()
                        .any(|clause| query.enode("Fold", clause).unwrap().is_some())
                })
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
                assert_eq!(top_level(graph, function, "If").len(), 1);
            }
        },
    );
}

#[test]
fn invariant_fusion_moves_all_members_together() {
    fused_cases(&["entry main() [4]i32=map(|p|let a=map(|i|i+2,iota(8)) in let b=map(|i|i+1,iota(8)) in a[0]+b[0],iota(4))"], |_, graph, function| {
        let query = Query(graph);
        assert!(top_level(graph, function, "Parallel").into_iter().any(|statement| {
            let statement = query.fields("Let", statement).unwrap();
            query.vector(statement[1]).unwrap().into_iter().filter(|binding| {
                    query.enode("BindValue", *binding).unwrap().is_some_and(|binding| array_extents(query, binding[1]).is_some())
            }).count() == 2
        }), "the complete fused computation must leave the callback");
    });
}

#[test]
fn retained_fusion_outputs_block_motion_when_any_member_depends_on_the_iteration() {
    fused_cases(&["entry main() [4]i32=map(|p|let a=map(|i|i+p,iota(8)) in let b=map(|i|i+1,iota(8)) in a[0]+b[0],iota(4))"], |_, graph, function| {
        let query = Query(graph);
        let root = query.constructor("FunctionRoot", (function,)).unwrap().unwrap();
        let top = top_level(graph, function, "Parallel");
        let mut dependent = 0;
        query.for_each("ContainsStm", |row| {
            if row[0] != root { return Ok(()); }
            let statement = query.fields("Let", row[1])?;
            if query.enode("Parallel", statement[3])?.is_none() { return Ok(()); }
            let parallel = query.fields("Parallel", statement[3])?;
            let domain = query.fields("IterationSpace", parallel[1])?;
            let axis = query.fields("DenseAxis", query.vector(domain[1])?[0])?;
            let extent = query.required("StaticIndex", (function, axis[1]))?;
            if graph.value_to_base::<i64>(extent) == 8 && !top.contains(&row[1]) {
                dependent += 1;
            }
            Ok(())
        }).unwrap();
        assert_eq!(dependent, 1, "the iteration-dependent row must stay in the callback; its independent sibling may move out");
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
            let definition = query.fields("DefineFunction", function).unwrap();
            let body = query.fields("Block", definition[3]).unwrap();
            let results = query.vector(body[1]).unwrap();
            let result = query.fields("Yield", *results.last().unwrap()).unwrap();
            let operand = query.fields("ValueOperand", result[1]).unwrap();
            let name = query.fields("Var", operand[0]).unwrap()[0];
            let mut checked = false;
            for outer in top_level(graph, function, "Parallel") {
                let statement = query.fields("Let", outer).unwrap();
                let exports_result = query.vector(statement[1]).unwrap().iter().any(|binding| {
                    query.enode("BindValue", *binding).unwrap().is_some_and(|binding| binding[0] == name)
                });
                if !exports_result {
                    let effects = query.required("EffectsStm", (function, outer)).unwrap();
                    assert!(
                        graph.value_to_container::<SetContainer>(effects).unwrap().data.is_empty(),
                        "case {index}: only total independent producers may leave the callback"
                    );
                    continue;
                }
                let parallel = query.fields("Parallel", statement[3]).unwrap();
                let root = query.constructor("ParallelRoot", (parallel[2],)).unwrap().unwrap();
                let mut nested = false;
                query
                    .for_each("ContainsExp", |row| {
                        nested |= row[0] == root && query.enode("Parallel", row[1])?.is_some();
                        Ok(())
                    })
                    .unwrap();
                assert!(
                    nested,
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
            let statement = query.fields("Let", views[0]).unwrap();
            let view = query.fields("View", statement[3]).unwrap();
            let slices = query.vector(view[1]).unwrap();
            let slice = query.fields("Slice", slices[0]).unwrap();
            assert_eq!(
                graph.value_to_base::<i64>(query.required("StaticIndex", (function, slice[1])).unwrap()),
                2
            );
            let source_count = query.required("LogicalOuterCount", (function, view[0])).unwrap();
            assert_eq!(
                graph
                    .value_to_base::<i64>(query.required("StaticIndex", (function, source_count)).unwrap()),
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
            for statement in top_level(graph, function, "Parallel") {
                let fields = query.fields("Let", statement).unwrap();
                let parallel = query.fields("Parallel", fields[3]).unwrap();
                let body = query.fields("ParallelBlock", parallel[2]).unwrap();
                for clause in query.vector(body[1]).unwrap() {
                    if let Some(fold) = query.enode("Fold", clause).unwrap() {
                        if query.enode("Total", fold[0]).unwrap().is_some() {
                            totals.push(statement);
                        } else {
                            scans.push(statement);
                        }
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
            let current = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
            let root = query.constructor("ProgramRoot", (current,)).unwrap().unwrap();
            let mut updates = 0;
            query
                .for_each("ContainsStm", |row| {
                    if row[0] != root {
                        return Ok(());
                    }
                    let statement = query.fields("Let", row[1])?;
                    let Some(update) = query.enode("Update", statement[3])? else {
                        return Ok(());
                    };
                    let pattern = query.vector(statement[1])?;
                    assert_eq!(pattern.len(), 1);
                    let destination = query.fields("BindValue", pattern[0])?;
                    let mut checked = false;
                    query.for_each("ProgramFunction", |member| {
                        if member[0] != current {
                            return Ok(());
                        }
                        let Some(source) = query.lookup("BindingInfo", (member[1], update[1]))? else {
                            return Ok(());
                        };
                        assert_eq!(
                            destination[1], source,
                            "an update preserves the complete typed memory view"
                        );
                        checked = true;
                        Ok(())
                    })?;
                    assert!(checked);
                    updates += 1;
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
            let statement = query.fields("Let", loops[0]).unwrap();
            assert!(!query.contains("LoopPreservesOuterLength", (function, statement[3], 0i64)).unwrap());
            let loop_ = query.fields("Loop", statement[3]).unwrap();
            let carry = query.fields("Carry", query.vector(loop_[0]).unwrap()[0]).unwrap();
            let initial =
                query.fields("Var", query.fields("ValueOperand", carry[1]).unwrap()[0]).unwrap()[0];
            let result = query.fields("BindValue", query.vector(statement[1]).unwrap()[0]).unwrap();
            assert!(array_extents(query, result[1]).is_some());
            assert!(!query.contains("LengthSource", (function, result[0], initial)).unwrap());
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
            let statement = query.fields("Let", loops[0]).unwrap();
            assert_eq!(
                query.contains("LoopPreservesOuterLength", (function, statement[3], 0i64)).unwrap(),
                index == 0
            );
        },
    );
}

fn local_capacities(graph: &EGraph) -> Vec<i64> {
    let mut graph = graph.clone();
    graph.parse_and_run_program(None, "(run-schedule (saturate schedule-cost))").unwrap();
    let query = Query(&graph);
    let current = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
    let root = query.constructor("ProgramRoot", (current,)).unwrap().unwrap();
    let mut capacities = Vec::new();
    query
        .for_each("ContainsStm", |row| {
            if row[0] != root {
                return Ok(());
            }
            let statement = query.fields("Let", row[1])?;
            let Some(parallel) = query.enode("Parallel", statement[3])? else {
                return Ok(());
            };
            if query.enode("InvocationLocal", parallel[0])?.is_none() {
                return Ok(());
            }
            query.for_each("ProgramFunction", |member| {
                if member[0] != current {
                    return Ok(());
                }
                let function_root = query.constructor("FunctionRoot", (member[1],))?.unwrap();
                if !query.contains("ContainsStm", (function_root, row[1]))? {
                    return Ok(());
                }
                for binding in query.vector(statement[1])? {
                    let Some(binding) = query.enode("BindValue", binding)? else {
                        continue;
                    };
                    if let Some(capacity) = query.lookup("OuterCapacity", (member[1], binding[0]))? {
                        capacities.push(graph.value_to_base::<i64>(capacity));
                    }
                }
                Ok(())
            })?;
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
            "entry main(xs:[]i32) []i32=map(|x|x+1,scan((+),0,xs))",
            vec![(1, true, false)],
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
        for statement in top_level(graph, region, "Parallel") {
            let statement = query.fields("Let", statement).unwrap();
            let parallel = query.fields("Parallel", statement[3]).unwrap();
            let body = query.fields("ParallelBlock", parallel[2]).unwrap();
            let arrays = query
                .vector(statement[1])
                .unwrap()
                .into_iter()
                .filter(|binding| {
                    query
                        .enode("BindValue", *binding)
                        .unwrap()
                        .is_some_and(|binding| array_extents(query, binding[1]).is_some())
                })
                .count() as i64;
            let mut scans = false;
            let mut totals = false;
            for clause in query.vector(body[1]).unwrap() {
                if let Some(fold) = query.enode("Fold", clause).unwrap() {
                    scans |= query.enode("Total", fold[0]).unwrap().is_none();
                    totals |= query.enode("Total", fold[0]).unwrap().is_some()
                        || query.enode("InclusiveScanAndTotal", fold[0]).unwrap().is_some()
                        || query.enode("ExclusiveScanAndTotal", fold[0]).unwrap().is_some();
                }
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
    program
        .graph
        .parse_and_run_program(
            None,
            r#"
        (relation ReorderedAccumulatorTuple ())
        (ruleset test-reorder-accumulators)
        (rule ((CurrentProgram program) (ProgramFunction program function)
            (ContainsStm (FunctionRoot function) (Let id pattern certs
                (Parallel execution domain (ParallelBlock statements clauses))))
            (= (vec-length pattern) 2) (= (vec-length clauses) 2)
            (= (vec-get clauses 0) (Fold first-kind first-reducer first-values))
            (= (vec-get clauses 1) (Fold second-kind second-reducer second-values)))
          ((set (PendingRewrite "test-reorder" program) (set-of
              (Rename (map-empty) (map-empty) (map-empty) (map-empty)
                  (map-insert (map-empty) id (vec-of (Let id
                      (vec-of (vec-get pattern 1) (vec-get pattern 0)) certs
                      (Parallel execution domain (ParallelBlock statements
                          (vec-of (vec-get clauses 1) (vec-get clauses 0))))))))))
           (RewriteDemand "test-reorder" program) (ReorderedAccumulatorTuple))
          :ruleset test-reorder-accumulators)
        (run-schedule test-reorder-accumulators (saturate memory-rewrite memory-rename))
        (check (ReorderedAccumulatorTuple))
    "#,
        )
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
        let (_source, kind, rank, count) = cases[index];
        let query = Query(graph);
        let root = query.constructor("FunctionRoot", (function,)).unwrap().unwrap();
        let mut matching = 0;
        query
            .for_each("ContainsExp", |row| {
                if row[0] != root {
                    return Ok(());
                }
                if kind == "bucket-scatter" {
                    let Some(bucket) = query.enode("BucketScatter", row[1])? else {
                        return Ok(());
                    };
                    assert_eq!(graph.value_to_base::<i64>(bucket[4]), rank);
                    let mut dimensions = vec![None; rank as usize];
                    for (source, mapping) in
                        query.vector(bucket[2])?.into_iter().zip(query.vector(bucket[3])?)
                    {
                        let info = query.required("BindingInfo", (function, source))?;
                        let extents = array_extents(query, info).expect("bucket input array");
                        let mapping = query.vector(mapping)?;
                        assert_eq!(extents.len(), mapping.len());
                        for (extent, axis) in extents.into_iter().zip(mapping) {
                            let axis = graph.value_to_base::<i64>(axis) as usize;
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
                    return Ok(());
                }
                let Some(parallel) = query.enode("Parallel", row[1])? else {
                    return Ok(());
                };
                let body = query.fields("ParallelBlock", parallel[2])?;
                let clauses = query.vector(body[1])?;
                let constructor = match kind {
                    "map" => "Collect",
                    "filter" => "Compact",
                    "bucket-scatter" => "Contribute",
                    _ => "Fold",
                };
                if !clauses.iter().any(|clause| query.enode(constructor, *clause).unwrap().is_some()) {
                    return Ok(());
                }
                let domain = query.fields("IterationSpace", parallel[1])?;
                let axes = query.vector(domain[1])?;
                assert_eq!(axes.len(), rank as usize, "case {index}");
                let mut elements = 1;
                for axis in axes {
                    let axis = query.fields("DenseAxis", axis)?;
                    elements *=
                        graph.value_to_base::<i64>(query.required("StaticIndex", (function, axis[1]))?);
                }
                assert_eq!(elements, count, "case {index}");
                matching += 1;
                Ok(())
            })
            .unwrap();
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
                panic!("one outer map")
            };
            let outer = query.fields("Let", outer).unwrap();
            let outer = query.fields("Parallel", outer[3]).unwrap();
            let domain = query.fields("IterationSpace", outer[1]).unwrap();
            let axis = query.vector(domain[1]).unwrap()[0];
            let outer_index = query.fields("DenseAxis", axis).unwrap()[0];
            let body = query.fields("ParallelBlock", outer[2]).unwrap();
            let inner = query
                .vector(body[0])
                .unwrap()
                .into_iter()
                .find_map(|statement| {
                    let statement = query.fields("Let", statement).unwrap();
                    query.enode("Parallel", statement[3]).unwrap()
                })
                .expect("the inner map retains its captured outer value");
            let inner_domain = query.fields("IterationSpace", inner[1]).unwrap();
            let inner_axis = query.vector(inner_domain[1]).unwrap()[0];
            let inner_index = query.fields("DenseAxis", inner_axis).unwrap()[0];
            assert_ne!(outer_index, inner_index, "nested axes need distinct binders");
            let names = query.required("NamesParallelBody", (inner[2],)).unwrap();
            let names = graph.value_to_container::<SetContainer>(names).unwrap();
            let bound = query.required("BindersParallelBody", (inner[2],)).unwrap();
            let bound = graph.value_to_container::<SetContainer>(bound).unwrap();
            let bindings = query.required("BindersStms", (body[0],)).unwrap();
            let bindings = graph.value_to_container::<SetContainer>(bindings).unwrap();
            assert!(
                names.data.difference(&bound.data).any(|name| bindings.data.contains(name)),
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
            let current = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
            let function = query.vector(query.fields("MemoryProgram", current).unwrap()[0]).unwrap()[0];
            let [operation] = top_level(graph, function, "BucketScatter")[..] else {
                panic!("one bucket scatter")
            };
            let statement = query.fields("Let", operation).unwrap();
            let bucket = query.fields("BucketScatter", statement[3]).unwrap();
            let mapping = query.vector(bucket[3]).unwrap()[0];
            let axes = graph.value_to_container::<VecContainer>(mapping).unwrap();
            assert_eq!(
                axes.data.iter().map(|&axis| graph.value_to_base::<i64>(axis)).collect::<Vec<_>>(),
                [0, 1]
            );
            let input = query.vector(bucket[2]).unwrap()[0];
            let producer = top_level(graph, function, "Parallel")
                .into_iter()
                .find(|statement| {
                    let statement = query.fields("Let", *statement).unwrap();
                    query.vector(statement[1]).unwrap().iter().any(|binding| {
                        query
                            .enode("BindValue", *binding)
                            .unwrap()
                            .is_some_and(|binding| binding[0] == input)
                    })
                })
                .expect("bucket input producer");
            let producer = query.fields("Let", producer).unwrap();
            let parallel = query.fields("Parallel", producer[3]).unwrap();
            let domain = query.fields("IterationSpace", parallel[1]).unwrap();
            let sizes = query
                .vector(domain[1])
                .unwrap()
                .into_iter()
                .map(|axis| {
                    let axis = query.fields("DenseAxis", axis).unwrap();
                    graph.value_to_base::<i64>(query.required("StaticIndex", (function, axis[1])).unwrap())
                })
                .collect::<Vec<_>>();
            assert_eq!(sizes, [4, 6]);
            assert_eq!(graph.value_to_base::<i64>(bucket[4]), 2);
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
        let statement = query.fields("Let", parallel[0]).unwrap();
        let expression = query.fields("Parallel", statement[3]).unwrap();
        let domain = query.fields("IterationSpace", expression[1]).unwrap();
        let axes = query.vector(domain[1]).unwrap();
        let sizes = axes
            .into_iter()
            .map(|axis| {
                let axis = query.fields("DenseAxis", axis).unwrap();
                graph.value_to_base::<i64>(query.required("StaticIndex", (function, axis[1])).unwrap())
            })
            .collect::<Vec<_>>();
        assert_eq!(
            sizes,
            [4, 3],
            "flattening must keep the logical row and column extents"
        );
        let binding = query.vector(statement[1]).unwrap()[0];
        let binding = query.fields("BindValue", binding).unwrap();
        let extents = array_extents(query, binding[1]).unwrap();
        assert_eq!(
            extents,
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
    let root = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
    let fields = query.fields("MemoryProgram", root).unwrap();
    let mut ranks = Vec::new();
    for function in query.vector(fields[0]).unwrap() {
        for statement in top_level(&program.graph, function, "Parallel") {
            let statement = query.fields("Let", statement).unwrap();
            let parallel = query.fields("Parallel", statement[3]).unwrap();
            let domain = query.fields("IterationSpace", parallel[1]).unwrap();
            ranks.push(query.vector(domain[1]).unwrap().len());
        }
    }
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
    let mut prefix = String::new();
    for i in 1..=80 {
        prefix.push_str(&format!("(Let (StatementTag {i}) (vec-of (BindValue (ValueId {i}) scalar)) (vec-of) (PrimOp \"*\" (vec-of (Var (ValueId {})) (LiteralBits (I32) 2))))\n", i-1));
    }
    let mut graph = super::super::fusion::new_graph().unwrap();
    graph
        .parse_and_run_program(
            None,
            &include_str!("tests/distribution_cost.egg").replace("PREFIX", &prefix),
        )
        .unwrap();
    super::super::analysis::run(&mut graph).unwrap();
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
        .for_each("DistributionCandidate", |row| {
            assert!(query.flag("DistributionSafe", (row[0], row[2]))?);
            assert_eq!(query.vector(row[4])?.len(), 80, "expensive repeated prefix");
            assert!(!query.contains("DistributeNest", (row[0], row[1], row[2], row[3], row[4]))?);
            legal += 1;
            Ok(())
        })
        .unwrap();
    assert!(legal > 0, "cost must not remove a legal distribution candidate");
    let current = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
    let function = query.vector(query.fields("MemoryProgram", current).unwrap()[0]).unwrap()[0];
    let statements = top_level(&graph, function, "Parallel");
    assert_eq!(statements.len(), 1);
    let statement = query.fields("Let", statements[0]).unwrap();
    let parallel = query.fields("Parallel", statement[3]).unwrap();
    let domain = query.fields("IterationSpace", parallel[1]).unwrap();
    assert_eq!(
        query.vector(domain[1]).unwrap().len(),
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
            .map(|statement| {
                let statement = query.fields("Let", statement).unwrap();
                let parallel = query.fields("Parallel", statement[3]).unwrap();
                let domain = query.fields("IterationSpace", parallel[1]).unwrap();
                query.vector(domain[1]).unwrap().len()
            })
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
        let statement = query.fields("Let", parallels[0]).unwrap();
        let mut shapes = query
            .vector(statement[1])
            .unwrap()
            .into_iter()
            .filter_map(|binding| {
                let binding = query.enode("BindValue", binding).unwrap()?;
                array_extents(query, binding[1])
            })
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
    ];
    planned_cases(&sources, |index, graph, function| {
        let query = Query(graph);
        let definition = query.fields("DefineFunction", function).unwrap();
        let body = query.fields("Block", definition[3]).unwrap();
        let results = query.vector(body[1]).unwrap();
        let result = query.fields("Yield", *results.last().unwrap()).unwrap();
        let operand = query.fields("ValueOperand", result[1]).unwrap();
        let payload = query.fields("Var", operand[0]).unwrap()[0];
        let ranks = top_level(graph, function, "Parallel")
            .into_iter()
            .filter_map(|statement| {
                let statement = query.fields("Let", statement).unwrap();
                if !query.vector(statement[1]).unwrap().into_iter().any(|binding| {
                    query.enode("BindValue", binding).unwrap().is_some_and(|binding| binding[0] == payload)
                }) {
                    return None;
                }
                let parallel = query.fields("Parallel", statement[3]).unwrap();
                let domain = query.fields("IterationSpace", parallel[1]).unwrap();
                Some(query.vector(domain[1]).unwrap().len())
            })
            .collect::<Vec<_>>();
        if index < 3 {
            assert_eq!(ranks, [[2].as_slice(), &[2], &[3]][index], "{}", sources[index]);
            assert!(
                top_level(graph, function, "Loop").is_empty(),
                "uniform rows need no growable loop"
            );
        } else {
            assert_eq!(
                ranks,
                [1],
                "a varying row stays nested beneath its outer traversal"
            );
            assert!(
                top_level(graph, function, "Loop").is_empty(),
                "planning must not manufacture a source loop to select array storage"
            );
        }
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
            let statement = query.fields("Let", kernels[0]).unwrap();
            let parallel = query.fields("Parallel", statement[3]).unwrap();
            let domain = query.fields("IterationSpace", parallel[1]).unwrap();
            assert_eq!(query.vector(domain[1]).unwrap().len(), [2, 2, 3][index]);
            let body = query.fields("ParallelBlock", parallel[2]).unwrap();
            assert!(query
                .vector(body[1])
                .unwrap()
                .iter()
                .any(|clause| query.enode("Fold", *clause).unwrap().is_some()));
            let execution = query.fields("DeviceKernel", parallel[0]).unwrap();
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
            let current = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
            let root = query.constructor("ProgramRoot", (current,)).unwrap().unwrap();
            let mut checked = 0;
            query
                .for_each("ContainsAccInput", |row| {
                    if row[0] != root {
                        return Ok(());
                    }
                    let input = query.fields("AccumulatorInput", row[1])?;
                    let Some(combine) = query.enode("CombineWrites", input[1])? else {
                        return Ok(());
                    };
                    let native = query
                        .lookup("NativeBinaryReducer", (combine[0],))?
                        .map(|value| graph.value_to_base::<S>(value).to_string());
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
        let query = Query(&program.graph);
        let current = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
        let function = query.vector(query.fields("MemoryProgram", current).unwrap()[0]).unwrap()[0];
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
            let query = Query(graph);
            let mut count = 0;
            query.for_each("CheckStatement", |row| {
                if query.lookup("ScopeRoot", (row[0],))? != Some(function)
                    || query.enode("Loop", row[3])?.is_none() { return Ok(()); }
                let scope = query.constructor("NestedScope", (row[0], row[1], 0i64))?.unwrap();
                assert_eq!(query.flag("HostScope", (scope,))?, dispatched, "{source}");
                assert_eq!(query.flag("LocalLoop", (function, row[3]))?, !dispatched, "{source}");
                count += 1;
                Ok(())
            }).unwrap();
            assert_eq!(count, 1, "{source}");
            let mut reductions = 0;
            for statement in top_level(graph, function, "Parallel") {
                let statement = query.fields("Let", statement).unwrap();
                let parallel = query.fields("Parallel", statement[3]).unwrap();
                let body = query.fields("ParallelBlock", parallel[2]).unwrap();
                reductions += query.vector(body[1]).unwrap().into_iter()
                    .filter(|clause| query.enode("Fold", *clause).unwrap().is_some()).count();
            }
            assert_eq!(reductions, hoisted_reductions,
                "the invariant reduction must execute once before the local loop: {source}");
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
            let mut checked = 0;
            query
                .for_each("CheckStatement", |row| {
                    if query.lookup("ScopeRoot", (row[0],))? != Some(function) {
                        return Ok(());
                    }
                    let Some(loop_) = query.enode("Loop", row[3])? else {
                        return Ok(());
                    };
                    if query.enode("While", loop_[1])?.is_none() {
                        return Ok(());
                    }
                    let scope = query.constructor("NestedScope", (row[0], row[1], 0i64))?.unwrap();
                    assert_eq!(query.flag("HostScope", (scope,))?, dispatched, "{source}");
                    checked += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(checked, 1);
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
        let folds = parallels
            .into_iter()
            .map(|statement| {
                let statement = query.fields("Let", statement).unwrap();
                let parallel = query.fields("Parallel", statement[3]).unwrap();
                let body = query.fields("ParallelBlock", parallel[2]).unwrap();
                query
                    .vector(body[1])
                    .unwrap()
                    .into_iter()
                    .filter(|clause| query.enode("Fold", *clause).unwrap().is_some())
                    .count()
            })
            .sum::<usize>();
        assert_eq!(folds, 2, "fusion must retain both reductions");
    });
}

#[test]
fn update_handles_and_reservation_results_cannot_escape_their_scope() {
    let tlc = compile_thru_tlc("entry main(dest:*[2]i32) [2]i32=scatter(dest,[0,1],[7,9])").unwrap();
    let program = fuse(from_tlc(&tlc).unwrap()).unwrap();
    for (token, values) in [
        ("(AccumulatorId 999999)", "values"),
        ("token", "(vec-of (Var (ValueId 999999)))"),
    ] {
        let mut graph = program.graph.clone();
        graph
            .parse_and_run_program(
                None,
                &format!(
                    r#"
            (ruleset escape-update)
            (rule ((CurrentProgram program) (ContainsStm (ProgramRoot program)
                (Let id pattern certs (Accumulate bounds token indices values))))
              ((set (PendingRewrite "escape" program) (set-of
                (Rename (map-empty) (map-empty) (map-empty) (map-empty)
                  (map-insert (map-empty) id (vec-of
                    (Let id pattern certs (Accumulate bounds {token} indices {values})))))))
               (RewriteDemand "escape" program)) :ruleset escape-update)
            (run-schedule escape-update (saturate memory-rewrite memory-rename))
        "#
                ),
            )
            .unwrap();
        let error = super::super::memory::validate(&mut graph).unwrap_err();
        assert!(error.to_string().contains("outside its scope"), "{error}");
    }
}

#[test]
fn calibrated_costs_select_among_the_same_legal_map_executions() {
    let tlc = compile_thru_tlc("entry main(xs:[4][513]i32) [4]i32=map(|row|reduce((+),0,row),xs)").unwrap();
    for (barrier_ns, rank) in [(0.0, 2), (1e9, 1)] {
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
        let current = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
        let function = query.vector(query.fields("MemoryProgram", current).unwrap()[0]).unwrap()[0];
        let [kernel] = top_level(&program.graph, function, "Parallel")[..] else {
            panic!("one map kernel");
        };
        let kernel = query.fields("Let", kernel).unwrap();
        let parallel = query.fields("Parallel", kernel[3]).unwrap();
        let domain = query.fields("IterationSpace", parallel[1]).unwrap();
        assert_eq!(
            query.vector(domain[1]).unwrap().len(),
            rank,
            "barrier_ns={barrier_ns}"
        );
        let program = super::super::prepare_interfaces(program).unwrap();
        let program =
            super::super::optimize_with_policy(program, super::super::ScalarOptimization::Basic).unwrap();
        finish_for_test(program, CodegenTarget::Wgsl).unwrap();
    }
}
