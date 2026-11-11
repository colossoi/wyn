use super::super::query::Query;
use super::super::{finish_for_test, from_tlc, fuse, place, schedule, schedule_with_subgroups};
use crate::{compile_thru_tlc, CodegenTarget, PipelineTopologyPolicy};
use egglog_engine::{
    sort::{SetContainer, VecContainer, S},
    EGraph, Value, Write,
};

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
        let program = place(fuse(from_tlc(&tlc).unwrap()).unwrap(), PipelineTopologyPolicy::AllowGenerated).unwrap();
        check_entries(&program, 1, |_, graph, function| check(index, graph, function));
    }
}

fn distributed_cases(sources: &[impl AsRef<str>], check: impl Fn(usize, &EGraph, Value)) {
    for (index, source) in sources.iter().enumerate() {
        let tlc = compile_thru_tlc(&cases_source(std::slice::from_ref(source))).unwrap();
        let mut program = place(fuse(from_tlc(&tlc).unwrap()).unwrap(), PipelineTopologyPolicy::AllowGenerated).unwrap();
        super::distribute(&mut program.graph, super::CostModel::default()).unwrap();
        check_entries(&program, 1, |_, graph, function| check(index, graph, function));
    }
}

fn operations(graph: &EGraph, kind: &str) -> Vec<Value> {
    let mut operations = Vec::new();
    Query(graph)
        .for_function("OperatorDefinition", |row, info| {
            if Query(graph).operation_kind(info)? == kind {
                operations.push(row[0]);
            }
            Ok(())
        })
        .unwrap();
    operations
}

fn operations_in(graph: &EGraph, kind: &str, region: Value) -> Vec<Value> {
    operations(graph, kind)
        .into_iter()
        .filter(|&op| Query(graph).required("RegionOperationOwner", (op,)).unwrap() == region)
        .collect()
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
fn invariant_memory_computations_move_across_control_scopes() {
    fused_cases(
        &[
            "entry main(xs:[]i32,b:bool) i32=if b then reduce((+),0i32,xs) else 0i32",
            "entry main(xs:[]i32,n:i32) i32=loop x=0i32 for i<n do x+reduce((+),0i32,xs)",
        ],
        |_, graph, function| {
            assert_eq!(
                top_level(graph, function, "Parallel").len(),
                1,
                "the closed reduction must be in the enclosing function body"
            );
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
                query.enode("BindValue", *binding).unwrap().is_some_and(|binding| query.enode("ArrayValue", binding[1]).unwrap().is_some())
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
        let mut fused = false;
        query.for_each("ContainsStm", |row| {
            if row[0] != root { return Ok(()); }
            let statement = query.fields("Let", row[1])?;
            if query.enode("Parallel", statement[3])?.is_none() { return Ok(()); }
            let arrays = query.vector(statement[1])?.into_iter().filter(|binding| {
                query.enode("BindValue", *binding).unwrap().is_some_and(|binding|
                    query.enode("ArrayValue", binding[1]).unwrap().is_some())
            }).count();
            if arrays == 2 {
                fused = true;
                assert!(!top.contains(&row[1]), "a captured iteration prevents hoisting the fused body");
            }
            Ok(())
        }).unwrap();
        assert!(fused, "exercise motion of a retained multi-output fusion");
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
    planned(
        "entry main() [][]i32=let rows=map(|n|map(|i|i+n,iota(n)),[0,1,3,7]) in rows[1..3]",
        |graph| {
            let query = Query(graph);
            query
                .for_function("ArrayView", |keys, _| {
                    assert!(query.flag("HasDirectResult", (keys[0],))?);
                    let owner = query.required("SourceValueOperation", (keys[0],))?;
                    let domain = query.required("InputDomain", (owner,))?;
                    let length = query.required("ConstantLogicalExtent", (domain,))?;
                    assert_eq!(graph.value_to_base::<i64>(length), 2);
                    Ok(())
                })
                .unwrap();
        },
    );
}

#[test]
fn outputs_that_capture_a_total_wait_for_the_collective_barrier() {
    planned_cases(&[
        "entry main(xs:[257]i32) [257]i32=let prefix=scan((+),0,xs) let total=reduce((+),0,xs) in map(|x|x+total,prefix)",
        "entry main(xs:[257]i32) [257]i32=let total=reduce((+),0,xs) in map(|x|x+total,xs)",
        "entry main(xs:[257]i32) [3]i32=let total=reduce((+),0,xs) let dest=[0,0,0] in reduce_by_index(dest,(+),0,map(|x|x%3,xs),map(|x|x+total,xs))",
    ], |index, graph, region| {
            let query = Query(graph);
            let reduce = operations_in(graph, "reduce", region)[0];
            let plan = query.required("SelectedPlan", (reduce,)).unwrap();
            for op in operations_in(graph, "map", region) {
                assert_eq!(plan, query.required("SelectedPlan", (op,)).unwrap(), "case {index}");
            }
            let owner = query.required("Owner", (plan,)).unwrap();
            let total = query.constructor("Result", (owner, 0i64)).unwrap().unwrap();
            for phase in ["chunks", "combine", "offsets"] {
                let stage = query.constructor("Stage", (owner, phase)).unwrap().unwrap();
                assert_eq!(
                    query.contains("Needs", (stage, total)).unwrap(),
                    phase == "offsets"
                );
                assert_eq!(
                    query.contains("Produces", (stage, total)).unwrap(),
                    phase == "combine"
                );
            }
            let mut phases = 0;
            query
                .for_each("Phase", |row| {
                    if row[1] == region { phases += 1; }
                    Ok(())
                })
                .unwrap();
            assert_eq!(phases, if index == 2 { 4 } else { 3 });
        },
    );
}

#[test]
fn radix_step_has_three_phases() {
    let source = format!(
        "{}\n entry main(xs: []i32) []i32 = radix_sort_step(xs, |i:i32,x:i32| (x >> i) & 1, 0)",
        include_str!("../../../pkg/sort/src/radix_sort.wyn")
    );
    planned_cases(&[source], |_, graph, function| {
        assert_eq!(dispatches(graph, function).len(), 3);
    });
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
fn four_bit_radix_step_has_three_phases() {
    let source = format!(
        "{}\n entry main(xs: []vec2u32) []vec2u32 = radix_sort_step(xs, |i,p| u32.get_bit(i,p.x), 0)",
        include_str!("../../../pkg/sort/src/four_bit.wyn")
    );
    planned_cases(&[source], |_, graph, function| {
        let stages = dispatches(graph, function);
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[0].width.get(), 64);
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
            assert_ne!(scans[0], totals[0], "case {index}: scan inputs depend on a completed total");
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
fn returned_array_lengths_share_storage() {
    planned(
        include_str!("../../../testfiles/regressions/runtime_array_three_outputs.wyn"),
        |graph| {
            let query = Query(graph);
            let mut slots = Vec::new();
            query
                .for_function("MetadataIndex", |keys, _| {
                    slots.push(query.required("Backing", (keys[0],))?);
                    Ok(())
                })
                .unwrap();
            assert!(!slots.is_empty());
            assert!(slots.iter().all(|slot| *slot == slots[0]));
            query
                .for_function("InputShapeBuffer", |_, storage| {
                    assert_eq!(storage, slots[0]);
                    Ok(())
                })
                .unwrap();
        },
    );
}

#[test]
fn copied_runtime_slices_do_not_overwrite_other_outputs() {
    planned(
        include_str!("../../../testfiles/regressions/triangulation_runtime_array_return_alias.wyn"),
        |graph| {
            let query = Query(graph);
            let mut backings = Vec::new();
            query
                .for_each("ReturnArray", |row| {
                    let backing = query.required("OutputBacking", (row[0],))?;
                    let view = query.record("MemoryLocation", "ArrayIn", (row[1],))?.unwrap();
                    let source = query.required("Backing", (view[0],))?;
                    if query.lookup("SourceSlice", (row[1],))?.is_some() {
                        assert_ne!(backing, source, "copying a slice must not overwrite its source");
                    }
                    assert!(
                        !backings.contains(&backing),
                        "returned arrays need separate destinations"
                    );
                    backings.push(backing);
                    Ok(())
                })
                .unwrap();
            assert_eq!(backings.len(), 2);
        },
    );
}

#[test]
fn shrinking_loop_does_not_inherit_its_initial_extent() {
    planned_cases(&["entry main(xs:[]i32) []i32 =
         loop out=xs for i<2 do out[0..length(out)-1]"], |_, graph, function| {
        let query = Query(graph);
        let definition = query.fields("DefineFunction", function).unwrap();
        let initial = query.vector(definition[1]).unwrap().into_iter().find_map(|parameter| {
            let parameter = query.fields("Parameter", parameter).unwrap();
            let binding = query.enode("BindValue", parameter[0]).unwrap()?;
            let info = query.enode("ArrayInfo", binding[1]).unwrap()?;
            let location = query.fields("ArrayIn", info[1]).unwrap();
            Some(query.fields("Layout", location[1]).unwrap()[1])
        }).unwrap();
        let mut arrays = 0;
        for statement in top_level(graph, function, "Loop") {
            let statement = query.fields("Let", statement).unwrap();
            let bound = query.required("BindersPattern", (statement[1],)).unwrap();
            let bound = graph.value_to_container::<SetContainer>(bound).unwrap();
            for binding in query.vector(statement[1]).unwrap() {
                let Some(binding) = query.enode("BindValue", binding).unwrap() else { continue; };
                let Some(info) = query.enode("ArrayInfo", binding[1]).unwrap() else { continue; };
                let location = query.fields("ArrayIn", info[1]).unwrap();
                let layout = query.fields("Layout", location[1]).unwrap();
                assert_ne!(layout[1], initial, "the shrinking result must not inherit input dimensions");
                let dim = query.fields("Dim", query.vector(layout[1]).unwrap()[0]).unwrap();
                let names = query.required("NamesIndexExp", (dim[1],)).unwrap();
                let names = graph.value_to_container::<SetContainer>(names).unwrap();
                assert!(names.data.intersection(&bound.data).next().is_some(),
                    "the result length comes from the completed loop");
                arrays += 1;
            }
        }
        assert_eq!(arrays, 1);
    });
}

#[test]
fn loop_capacity_obligations_use_the_length_preservation_proof() {
    planned_cases(
        &[
            include_str!("../../../testfiles/regressions/triangulation_allocated_array_loop.wyn"),
            "entry main(n:i32) []i32 = loop out=iota(n) for i<2 do out[1..length(out)]",
        ],
        |index, graph, scope| {
            let preserves = index == 0;
            let query = Query(graph);
            let mut loops = 0;
            query
                .for_function("SourceLoop", |row, info| {
                    let value = row[0];
                    if query.required("SourceResult", (scope,))? != value {
                        return Ok(());
                    }
                    assert_eq!(query.flag("LoopPreservesLength", (value,))?, preserves);
                    let regions = query.enode("LoopInfo", info)?.unwrap();
                    let initial =
                        query.record("SourceLoopInitial", "LoopInitialInfo", (regions[0],))?.unwrap();
                    let next = query.required("SourceResult", (regions[1],))?;
                    let storage = query.constructor("LoopStorage", (value,))?.unwrap();
                    let bounds = graph
                        .value_to_container::<SetContainer>(query.required("AllocationBounds", (storage,))?)
                        .unwrap();
                    assert!(bounds.data.contains(&query.constructor("Length", (initial[1],))?.unwrap()));
                    assert_eq!(
                        bounds.data.contains(&query.constructor("Length", (next,))?.unwrap()),
                        !preserves
                    );
                    loops += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(loops, 1);
        },
    );
}

fn private_capacities(graph: &EGraph) -> Vec<i64> {
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
            let Some(allocation) = query.enode("Alloc", statement[3])? else {
                return Ok(());
            };
            if query.enode("PrivateSpace", allocation[0])?.is_none() {
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
                let bytes = query.required("StaticIndex", (member[1], allocation[1]))?;
                capacities.push(graph.value_to_base::<i64>(bytes));
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
            let capacities = private_capacities(graph);
            assert!(!capacities.is_empty());
            assert!(
                capacities.iter().all(|&bytes| bytes == 12),
                "three i32 elements: {capacities:?}"
            );
        },
    );
}

#[test]
fn compiler_repro_nested_capacity_plan() {
    let source =
        "entry main(xs:[]i32) []i32=map(|x| let a=loop a=[x,x+1,x+2] for k<2 do map(|y|y+1,a) in a[0],xs)";
    planned(source, |graph| {
        let capacities = private_capacities(graph);
        assert!(capacities.contains(&12));
        assert!(
            capacities.iter().all(|&bytes| bytes == 4 || bytes == 12),
            "{capacities:?}"
        );
    });
}

#[test]
fn compiler_repro_array_reduce_preserves_effect_barrier() {
    for source in ["entry main(xs:[][2]i32) [2]i32 =
        reduce(|a,b|map(|i|a[i]+b[i],iota(2)),[0,0],map(|row|map(|x|x+1,row),xs))"]
    {
        let tlc = compile_thru_tlc(source).unwrap();
        let mut imported = from_tlc(&tlc).unwrap();
        imported
            .graph
            .parse_and_run_program(
                None,
                "(run-schedule (saturate source-summary) (saturate source-dependencies))",
            )
            .unwrap();
        for writer in [None, Some("producer"), Some("consumer")] {
            let mut graph = imported.graph.clone();
            let query = Query(&graph);
            let reducer = operations(&graph, "reduce")[0];
            let producer = query
                .row("SourceDepends", |row| {
                    row[0] == reducer && query.enode("Input", row[2]).unwrap().is_some()
                })
                .unwrap()
                .unwrap()[1];
            let value = query.required("SourceOperationValue", (reducer,)).unwrap();
            if let Some(writer) = writer {
                let operation = if writer == "producer" { producer } else { reducer };
                let written = query.required("SourceOperationValue", (operation,)).unwrap();
                graph
                    .update(|mut sink| {
                        sink.set("SourceReadOnly", (written,), false)?;
                        sink.set("SourceSpeculatable", (written,), false)
                    })
                    .unwrap();
            }
            super::super::analysis::run(&mut graph).unwrap();
            super::super::fusion::run(&mut graph).unwrap();
            let query = Query(&graph);
            assert!(!query.flag("SourceSpeculatable", (value,)).unwrap());
            assert_eq!(
                query.required("SelectedPlan", (producer,)).unwrap()
                    == query.required("SelectedPlan", (reducer,)).unwrap(),
                writer.is_none()
            );
        }
    }
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
        query
            .for_each("Current", |row| {
                let mut belongs = false;
                query.for_each("Member", |member| {
                    if member[0] == row[0] {
                        belongs |= query.required("RegionOperationOwner", (member[1],))? == region;
                    }
                    Ok(())
                })?;
                if !belongs {
                    return Ok(());
                }
                let plan = row[1];
                if query.contains("ScremaPlan", (plan,))? || query.contains("FilterPlan", (plan,))? {
                    counts.push((
                        graph.value_to_base::<i64>(query.required("ArrayOutputs", (plan,))?),
                        graph.value_to_base::<bool>(query.required("HasScans", (plan,))?),
                        graph.value_to_base::<bool>(query.required("HasReductions", (plan,))?),
                    ));
                }
                Ok(())
            })
            .unwrap();
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
fn fused_scan_and_filter_count_have_distinct_accumulator_slots() {
    planned(
        "entry main(xs:[]i32) ([]i32,i32,i32)=let ys=filter(|x|x>0,xs) in
         (scan((+),0,xs),length(ys),reduce((+),0,ys))",
        |graph| {
            let query = Query(graph);
            let mut rows = Vec::new();
            query
                .for_each("Accumulator", |row| {
                    rows.push(row.to_vec());
                    Ok(())
                })
                .unwrap();
            assert_eq!(rows.len(), 3);
            assert!(
                rows.iter().all(|row| row[0] == rows[0][0]),
                "all components must be fused"
            );
            let mut slots: Vec<_> = rows.iter().map(|row| graph.value_to_base::<i64>(row[1])).collect();
            slots.sort();
            assert_eq!(slots, [0, 1, 2]);
            for row in rows {
                let slot = graph.value_to_base::<i64>(row[1]);
                let partial = query.constructor("Temporary", (row[0], "partial", slot)).unwrap().unwrap();
                assert!(query.contains("ElementType", (partial, row[2])).unwrap());
                if let Some(output) = query.enode("TotalOutput", row[3]).unwrap() {
                    assert_eq!(slot, graph.value_to_base::<i64>(output[0]) + 1);
                }
            }
        },
    );
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
        query.for_each("ContainsExp", |row| {
            if row[0] != root { return Ok(()); }
            let Some(parallel) = query.enode("Parallel", row[1])? else { return Ok(()); };
            let body = query.fields("ParallelBlock", parallel[2])?;
            let clauses = query.vector(body[1])?;
            let constructor = match kind { "map" => "Collect", "bucket-scatter" => "Contribute", _ => "Fold" };
            if !clauses.iter().any(|clause| query.enode(constructor, *clause).unwrap().is_some()) {
                return Ok(());
            }
            let domain = query.fields("IterationSpace", parallel[1])?;
            let axes = query.vector(domain[1])?;
            assert_eq!(axes.len(), rank as usize, "case {index}");
            let mut elements = 1;
            for axis in axes {
                let axis = query.fields("DenseAxis", axis)?;
                elements *= graph.value_to_base::<i64>(query.required("StaticIndex", (function, axis[1]))?);
            }
            assert_eq!(elements, count, "case {index}");
            matching += 1;
            Ok(())
        }).unwrap();
        assert!(matching > 0, "case {index}: no {kind} traversal");
    });
}

#[test]
fn fused_filter_postmap_retains_the_traversed_input_domain() {
    planned(
        "entry main(xs:[]i32) []i32=map(|x|x+3,filter(|x|x>0,xs))",
        |graph| {
            let query = Query(graph);
            let [filter] = operations(graph, "filter")[..] else {
                panic!("one filter")
            };
            let [map] = operations(graph, "map")[..] else {
                panic!("one map")
            };
            let mut plans = Vec::new();
            query
                .for_each("Current", |row| {
                    if query.contains("PlanMember", (row[1], filter))? {
                        plans.push(row[1]);
                    }
                    Ok(())
                })
                .unwrap();
            let [plan] = plans[..] else {
                panic!("one selected filter plan")
            };
            assert!(query.contains("PlanMember", (plan, map)).unwrap());
            assert_eq!(
                query.required("PlanDomain", (plan,)).unwrap(),
                query.required("SourceDomainShape", (filter,)).unwrap()
            );
            let input = query.required("SourceInput", (filter, 0i64)).unwrap();
            let extent = query.required("SourceExtent", (input,)).unwrap();
            let owner = query.required("Owner", (plan,)).unwrap();
            assert_eq!(query.required("InputDomain", (owner,)).unwrap(), extent);
        },
    );
}

#[test]
fn nested_domains_retain_distinct_axes_and_capture_origins() {
    fused_cases(&["entry main(xs:[4]i32) [4][4]i32=map(|x|map(|y|x+y,xs),xs)"], |_, graph, function| {
        let query = Query(graph);
        let [outer] = top_level(graph, function, "Parallel")[..] else { panic!("one outer map") };
        let outer = query.fields("Let", outer).unwrap();
        let outer = query.fields("Parallel", outer[3]).unwrap();
        let domain = query.fields("IterationSpace", outer[1]).unwrap();
        let axis = query.vector(domain[1]).unwrap()[0];
        let outer_index = query.fields("DenseAxis", axis).unwrap()[0];
        let body = query.fields("ParallelBlock", outer[2]).unwrap();
        let inner = query.vector(body[0]).unwrap().into_iter().find_map(|statement| {
            let statement = query.fields("Let", statement).unwrap();
            query.enode("Parallel", statement[3]).unwrap()
        }).expect("the inner map retains its captured outer value");
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
        assert!(names.data.difference(&bound.data).any(|name| bindings.data.contains(name)),
            "the inner body must capture a value bound by the outer body");
    });
}

#[test]
fn ranked_inputs_keep_their_independent_coordinate_mappings() {
    planned(
        "entry main(dest:*[4][8]i32) ([4][8]i32,[4]u32,u32)=
         let items=map(|i|map(|j|((i+j)%4,i*6+j),iota(6)),iota(4)) in
         bucket_scatter_2d(dest,items)",
        |graph| {
            let query = Query(graph);
            let [operation] = operations(graph, "bucket-scatter")[..] else {
                panic!("one bucket scatter")
            };
            let mapping = query.required("SourceInputMap", (operation, 0i64)).unwrap();
            let axes = graph.value_to_container::<VecContainer>(mapping).unwrap();
            assert_eq!(
                axes.data.iter().map(|&axis| graph.value_to_base::<i64>(axis)).collect::<Vec<_>>(),
                [0, 1]
            );
            let input = query.required("SourceInput", (operation, 0i64)).unwrap();
            assert!(query.lookup("ElementTraversal", (input, 2i64)).unwrap().is_some());
            let plan = query.required("SelectedPlan", (operation,)).unwrap();
            let owner = query.required("Owner", (plan,)).unwrap();
            let shape = query.required("IterationShape", (owner,)).unwrap();
            let axes = query.required("ShapeDimensions", (shape,)).unwrap();
            for (&extent, count) in
                graph.value_to_container::<VecContainer>(axes).unwrap().data.iter().zip([4, 6])
            {
                let actual = query.required("ConstantLogicalExtent", (extent,)).unwrap();
                assert_eq!(graph.value_to_base::<i64>(actual), count);
            }
        },
    );
}

#[test]
fn map_and_bucket_emission_read_layouts_and_require_resolved_grids() {
    let tlc = compile_thru_tlc(
        r#"
entry case_0(xs:[2]i32) [2][3]i32=map(|x|map(|y|x+y,iota(3)),xs)
entry case_1(dest:*[2][2]i32) ([2][2]i32,[2]u32,u32)=
         bucket_scatter_2d(dest,[[(0,1),(0,2)],[(1,3),(1,4)]])
"#,
    )
    .unwrap();
    let program = selected(&tlc);
    let mut prepared = super::super::prepare_host(program).unwrap();
    for table in ["ArrayStorage", "SelectedWidth", "DistributionCandidate", "DomainRenaming"] {
        prepared.program.graph.clear_function(table).unwrap();
    }
    super::super::to_ssa::lower(&prepared.program, CodegenTarget::Wgsl).unwrap();
    prepared.program.graph.parse_and_run_program(None, r#"
        (ruleset invalidate-grid)
        (rule ((CurrentProgram program) (ContainsStm (ProgramRoot program)
            (Let id pattern certs (Launch grid body))))
          ((set (PendingRewrite "invalid-grid" program) (set-of
            (Rename (map-empty) (map-empty) (map-empty) (map-empty)
              (map-insert (map-empty) id (vec-of
                (Let id pattern certs (Launch (AutomaticGrid) body)))))))
           (RewriteDemand "invalid-grid" program)) :ruleset invalidate-grid)
        (run-schedule invalidate-grid (saturate memory-rewrite memory-rename))
    "#).unwrap();
    let error = super::super::to_ssa::lower(&prepared.program, CodegenTarget::Wgsl).unwrap_err();
    assert!(error.to_string().contains("unresolved dispatch grid"), "{error}");
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
        let info = query.fields("ArrayValue", binding[1]).unwrap();
        let extents = query.vector(info[1]).unwrap().into_iter().map(|size|
            graph.value_to_base::<i64>(query.required("StaticIndex", (function, size)).unwrap())
        ).collect::<Vec<_>>();
        assert_eq!(extents, [4, 3], "the output retains its logical row capacity");
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
        fuse(from_tlc(&tlc).unwrap()).unwrap(), PipelineTopologyPolicy::AllowGenerated,
    ).unwrap();
    super::distribute(&mut program.graph, super::CostModel::default()).unwrap();
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
    let mut expression = String::from("x");
    for i in 1..=40 {
        expression = format!("({expression} * x + {i})");
    }
    let source = format!(
        "def costly(x:i32) i32={expression}
         entry main(xs:[8192]i32) [8192][3]i32=
         map(|x|let t=costly(x) in map(|y|t+y,iota(3)),xs)"
    );
    planned(&source, |graph| {
        let query = Query(graph);
        let mut expensive = false;
        query
            .for_function("WorkInlineCost", |_, cost| {
                expensive |= graph.value_to_base::<i64>(cost) >= 65;
                Ok(())
            })
            .unwrap();
        assert!(expensive, "fixture must exceed the old repetition budget");
        let mut legal_flattening = false;
        query
            .for_each("OutputTraversal", |row| {
                let rank = query.required("ShapeRank", (row[2],))?;
                legal_flattening |= graph.value_to_base::<i64>(rank) == 2;
                Ok(())
            })
            .unwrap();
        assert!(legal_flattening, "cost must not remove a legal schedule");
        let mut flattened = false;
        query
            .for_function("IterationRank", |_, rank| {
                flattened |= graph.value_to_base::<i64>(rank) == 2;
                Ok(())
            })
            .unwrap();
        assert!(
            !flattened,
            "ample outer parallelism should avoid repeating the expensive prefix"
        );
    });
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
        assert_eq!(parallels.len(), 1, "case {index}: both outputs share their outer traversal");
        let statement = query.fields("Let", parallels[0]).unwrap();
        let mut shapes = query.vector(statement[1]).unwrap().into_iter().filter_map(|binding| {
            let binding = query.enode("BindValue", binding).unwrap()?;
            let array = query.enode("ArrayValue", binding[1]).unwrap()?;
            Some(query.vector(array[1]).unwrap().into_iter().map(|size|
                graph.value_to_base::<i64>(query.required("StaticIndex", (function, size)).unwrap())
            ).collect::<Vec<_>>())
        }).collect::<Vec<_>>();
        shapes.sort();
        assert_eq!(shapes, [vec![4, 3], vec![4, [3, 5][index]]]);
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
        let root = query.constructor("FunctionRoot", (function,)).unwrap().unwrap();
        let block_of = |binding| {
            let binding = query.enode("BindValue", binding).unwrap()?;
            let info = query.enode("ArrayInfo", binding[1]).unwrap()?;
            Some((binding[0], query.fields("ArrayIn", info[1]).unwrap()[0]))
        };
        let mut payload_block = None;
        query
            .for_each("ContainsBinding", |row| {
                if row[0] == root {
                    if let Some((name, block)) = block_of(row[1]) {
                        if name == payload {
                            payload_block = Some(block);
                        }
                    }
                }
                Ok(())
            })
            .unwrap();
        let payload_block = payload_block.expect("returned payload has an explicit memory binding");
        let ranks = top_level(graph, function, "Parallel")
            .into_iter()
            .filter_map(|statement| {
                let statement = query.fields("Let", statement).unwrap();
                if !query
                    .vector(statement[1])
                    .unwrap()
                    .into_iter()
                    .any(|binding| block_of(binding).is_some_and(|(_, block)| block == payload_block))
                {
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
            assert!(ranks.is_empty(), "varying rows cannot use a dense payload domain");
            assert!(
                !top_level(graph, function, "Loop").is_empty(),
                "varying rows retain their sequential collection"
            );
        }
    });
}

#[test]
fn segment_callback_blockers_cover_every_output_of_a_fused_plan() {
    let tlc =
        compile_thru_tlc("entry main(xs:[17]i32) (i32,i32)=(reduce((+),0,xs),reduce((*),1,xs))").unwrap();
    let program = place(
        fuse(from_tlc(&tlc).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    for blocked in [false, true] {
        let mut graph = program.graph.clone();
        let reducers = operations(&graph, "reduce");
        assert_eq!(reducers.len(), 2);
        let query = Query(&graph);
        let plan = query.required("SelectedPlan", (reducers[0],)).unwrap();
        assert_eq!(plan, query.required("SelectedPlan", (reducers[1],)).unwrap());
        let callback = query.required("SourceOperatorBody", (reducers[0],)).unwrap();
        let values: Vec<_> = reducers
            .iter()
            .map(|&reducer| query.required("SourceOperationValue", (reducer,)).unwrap())
            .collect();
        graph
            .update(|mut sink| {
                // Request both results independently, as outer map traversal would.
                for &value in &values {
                    sink.add("RectangularDemand", (value,))?;
                    sink.add("TraversalDemand", (value, 0i64))?;
                }
                if blocked {
                    let region = sink.add("FlowRegion", callback)?;
                    sink.set("SourceDuplicable", region, false)?;
                }
                Ok(())
            })
            .unwrap();
        super::schedule(&mut graph, super::super::cost::CostModel::default()).unwrap();
        let query = Query(&graph);
        assert_eq!(query.flag("TraversalPlanBlocked", (plan,)).unwrap(), blocked);
        for value in values {
            assert!(query.row("TraversalPlan", |row| row == [value, plan]).unwrap().is_some());
            assert_eq!(
                query.lookup("DirectOutput", (value,)).unwrap().is_some(),
                !blocked
            );
        }
    }
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
            let grid = query.fields("Grid3D", execution[0]).unwrap();
            let groups = query.vector(grid[0]).unwrap()[0];
            let count = query.required("StaticIndex", (function, groups)).unwrap();
            assert_eq!(graph.value_to_base::<i64>(count), [4, 4, 12][index]);
            let width = query.vector(grid[1]).unwrap()[0];
            let width = query.fields("IndexConstant", width).unwrap();
            assert!(graph.value_to_base::<i64>(width[0]) > 1);
        },
    );
}

#[test]
fn workgroup_candidates_cover_uniform_control_flow_and_varying_inner_domains() {
    let sources = [
        "entry main(xs:[][17]i32) []i32=map(|row|if row[0]>0 then reduce((+),0,row) else 0,xs)",
        "entry main(xs:[][17]i32) []i32=map(|row|loop a=0 for i<2 do a+reduce((+),0,row),xs)",
        "entry main(xs:[]i32) []i32=map(|n|reduce((+),0,iota(n)),xs)",
        "entry main(xs:[][2][17]i32) [][2]i32=map(|rows|map(|row|if row[0]>0 then reduce((+),0,row) else 0,rows),xs)",
        "entry main(xs:[][2][17]i32) [][2]i32=map(|rows|map(|row|loop a=0 for i<2 do a+reduce((+),0,row),rows),xs)",
    ];
    planned_cases(&sources, |_, graph, region| {
        let query = Query(graph);
        for map in operations_in(graph, "map", region) {
            let traversal = query.required("MapOutputTraversal", (map, 0i64)).unwrap();
            assert!(query.contains("GroupSafeTraversal", (traversal,)).unwrap());
            assert!(query.contains("ParallelTraversalWork", (traversal,)).unwrap());
        }
    });
}

#[test]
fn selected_traversal_uses_shared_region_declarations() {
    let source = r#"
def row(x:i32) [3]i32=map(|y|x+y,iota(3))
          entry case_0(xs:[]i32) [][3]i32=map(row,xs)
entry case_1(xs:[][3][17]i32) [][3]i32=
          map(|rows|map(|row|reduce((+),0,map(|x|x+row[0],row)),rows),xs)
entry case_2(xs:[][3][17]i32) [][3][17]i32=
          map(|rows|map(|row|map(|x|x*2,scan((+),0,map(|x|x+1,row))),rows),xs)
def sum_row(row:[17]i32) i32=reduce((+),0,row)
entry case_3(xs:[][17]i32) []i32=map(sum_row,xs)
def repeat_sum(n:i32,x:i32) i32=reduce((+),0,map(|y|x+y,iota(n)))
entry case_4(xs:[]i32,n:i32) []i32=map(|x|repeat_sum(n,x),xs)
entry case_5(xs:[][17]i32) []i32=map(|row|if row[0]>0 then reduce((+),0,row) else reduce((*),1,row),xs)
"#;
    let tlc = compile_thru_tlc(source).unwrap();
    let mut program = selected(&tlc);
    let query = Query(&program.graph);
    let mut expected = crate::LookupMap::default();
    query
        .for_each("SourceOriginalEntry", |row| {
            let symbol = program.symbols.resolve(program.graph.value_to_base::<i64>(row[0]));
            let name = program.source.symbols.get(*symbol).unwrap();
            let index: usize = name.strip_prefix("case_").unwrap().parse().unwrap();
            expected.insert(
                row[1],
                [
                    (2, "TraversalValue"),
                    (2, "TraversalResult"),
                    (2, "TraversalResult"),
                    (1, "TraversalResult"),
                    (1, "TraversalResult"),
                    (1, "TraversalChoice"),
                ][index],
            );
            Ok(())
        })
        .unwrap();
    let mut outputs = 0;
    query
        .for_each("MapPlan", |row| {
            if !query.flag("MapHost", (row[0],))? {
                return Ok(());
            }
            let region = query.required("RegionOperationOwner", (row[1],))?;
            let (rank, leaf) = expected[&region];
            let mut traversal = query.required("MapOutputTraversal", (row[1], 0i64))?;
            let mut depth = 0;
            loop {
                if let Some(fields) = query.enode("TraversalFor", traversal)? {
                    depth += 1;
                    traversal = fields[2];
                } else if let Some(fields) = query.enode("TraversalCall", traversal)? {
                    traversal = fields[1];
                } else {
                    assert!(query.enode(leaf, traversal)?.is_some(), "{leaf}");
                    break;
                }
            }
            assert_eq!(depth, rank, "{source}");
            outputs += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(outputs, 6, "{source}");
    for table in [
        "SourceOperatorBody",
        "SourceCallArgument",
        "Traversal",
        "TraversalExtent",
    ] {
        program.graph.clear_function(table).unwrap();
    }
    finish_for_test(program, CodegenTarget::Wgsl).unwrap();
}

#[test]
fn segment_lengths_are_bound_inside_each_workgroup() {
    planned_cases(
        &[
            "entry main(xs:[4]i32,n:i32) []i32=map(|x|reduce((+),0,map(|y|x+y,iota(n))),xs)",
            "entry main(xs:[4]i32,n:i32) [][]i32=map(|x|scan((+),0,map(|y|x+y,iota(n))),xs)",
            "entry main(xs:[4]i32) []i32=map(|x|reduce((+),0,iota(x)),xs)",
        ],
        |index, graph, region| {
            let query = Query(graph);
            let maps = operations_in(graph, "map", region);
            assert_eq!(maps.len(), 1);
            let plan = query.required("SelectedPlan", (maps[0],)).unwrap();
            assert!(query.flag("MapHost", (plan,)).unwrap());
            assert_eq!(
                query
                    .enode(
                        "WorkgroupScope",
                        query.required("MapExecution", (maps[0],)).unwrap()
                    )
                    .unwrap()
                    .is_some(),
                true,
                "case {index}"
            );
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
            let operation = operations(graph, "reduce-by-index")[0];
            let body = query.required("SourceOperatorBody", (operation,)).unwrap();
            let update = query.required("AtomicReducer", (body,)).unwrap();
            assert!(
                query.enode(expected, update).unwrap().is_some(),
                "{reducer}: {expected}"
            );
            let plan = query.required("SelectedPlan", (operation,)).unwrap();
            let owner = query.required("Owner", (plan,)).unwrap();
            assert!(query.enode("Elements", query.required("Plan", (owner,)).unwrap()).unwrap().is_some());
        });
    }
}

#[test]
fn subgroup_addition_is_opt_in_and_uses_small_segments() {
    let tlc =
        compile_thru_tlc("entry main(xs:[4][32]u32) [4][32]u32 = map(|row|scan((+),0u32,row),xs)").unwrap();
    let mut placed = place(
        fuse(from_tlc(&tlc).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    let baseline = placed.graph.clone();
    for enabled in [false, true] {
        placed.graph = baseline.clone();
        let program = schedule_with_subgroups(placed, enabled).unwrap();
        let query = Query(&program.graph);
        let scan = operations(&program.graph, "scan")[0];
        let body = query.required("SourceOperatorBody", (scan,)).unwrap();
        assert_eq!(query.contains("SubgroupAddition", (body,)).unwrap(), enabled);
        let mut segments = 0;
        query
            .for_each("MapPlan", |row| {
                if query.flag("MapHost", (row[0],))?
                    && query.enode("WorkgroupScope", query.required("MapExecution", (row[1],))?)?.is_some()
                {
                    let width = query.required("MapWorkgroupWidth", (row[1],))?;
                    assert_eq!(program.graph.value_to_base::<i64>(width), 32);
                    segments += 1;
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(segments, 1);
        placed = program.advance(super::super::Placed);
    }
}

#[test]
fn tiled_outputs_require_complete_tiles_and_preserve_other_consumers() {
    for (count, fused) in [(8, true), (7, false)] {
        for reduced in [false, true] {
            for retained in [false, true] {
                let metadata =
                    if reduced { "reduce((+),0,map(|r|r[2],rows))" } else { "map(|r|r[2],rows)" };
                let metadata_type = if reduced { "i32".to_string() } else { format!("[{count}]i32") };
                let row_type = if retained { format!(",[{count}][3]i32") } else { String::new() };
                let row_result = if retained { ",rows" } else { "" };
                let source = format!("def row(i:i32) [3]i32 = [i,i+1,i+2]
                entry main() ([16]i32,{metadata_type}{row_type}) =
                let rows = map(row,iota({count})) in
                (map(|a|let i=(a/8)*4+a%4 in if i<length(rows) then rows[i][(a%8)/4] else -1,iota(16)), {metadata}{row_result})");
                planned(&source, |graph| {
                    let query = Query(graph);
                    let mut destinations = 0;
                    query
                        .for_function("MappedOutputs", |_, _| {
                            destinations += 1;
                            Ok(())
                        })
                        .unwrap();
                    assert_eq!(destinations, usize::from(fused), "{source}");
                    query
                        .for_each("MappedOutput", |row| {
                            let producer = row[0];
                            let slot = row[1];
                            let gather = row[2];
                            query.for_each("KernelPhase", |phase| {
                                assert_ne!(phase[0], gather, "tiled gather must not launch: {source}");
                                Ok(())
                            })?;
                            let result = query.constructor("Result", (producer, slot))?.unwrap();
                            let mut allocated = false;
                            query.for_each("Allocation", |allocation| {
                                allocated |= allocation[0] == result;
                                Ok(())
                            })?;
                            assert_eq!(allocated, retained, "only observed rows allocate: {source}");
                            Ok(())
                        })
                        .unwrap();
                });
            }
        }
    }
}
#[test]
fn primitive_dispatch_recipes_publish_their_phase_widths_and_order() {
    let source = compile_thru_tlc("").unwrap();
    let mut program = place(
        fuse(from_tlc(&source).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    program
        .graph
        .parse_and_run_program(
            None,
            r#"
        (set (Plan (OperationId 0)) (Elements))
        (set (ElideElements (OperationId 0)) false)
        (set (MapExecution (OperationId 0)) (InvocationScope))
        (set (IterationExtent (OperationId 0)) (Fixed 1024))
        (set (OperationGrid (OperationId 0)) (AutomaticGrid))
        (set (Plan (OperationId 1)) (Totals))
        (set (Plan (OperationId 2)) (Prefixes))
        (set (SelectedCollectiveGroups (OperationId 1)) 256)
        (set (CollectiveCombineWidth (OperationId 1)) 256)
        (set (SelectedCollectiveGroups (OperationId 2)) 256)
        (set (CollectiveCombineWidth (OperationId 2)) 256)
        (set (CollectiveChunkWidth (OperationId 1)) 256)
        (set (CollectiveChunkWidth (OperationId 2)) 256)
        (set (OperationGrid (OperationId 1)) (AutomaticGrid))
        (set (OperationGrid (OperationId 2)) (AutomaticGrid))
        (set (InputDomain (OperationId 2)) (Fixed 1024))
        (Accumulator (OperationId 3) 0 (TypeId 0) (LocalPrefix (Result (OperationId 3) 1)) (SourceTerm (RegionId 0) 0))
        (set (ParallelBodySafe (OperationId 3)) true)
        (set (OperationGrid (OperationId 3)) (FixedGrid 2 3 4))
        (set (Plan (OperationId 4)) (Elements))
        (set (ParallelBodySafe (OperationId 4)) true)
        (set (IterationExtent (OperationId 4)) (Fixed 5))
        (set (ElideElements (OperationId 4)) false)
        (set (MapExecution (OperationId 4)) (InvocationScope))
        (set (OperationGrid (OperationId 4)) (AutomaticGrid))
        (set (Plan (OperationId 5)) (LocalFold 1))
        (run-schedule (saturate schedule))
        (check (KernelPhase (OperationId 0) "elements" (Fixed 1024) 64 (AutomaticGrid)))
        (check (KernelPhase (OperationId 1) "chunks" (Fixed 65536) 256 (FixedGrid 256 1 1)))
        (check (KernelPhase (OperationId 1) "combine" (Fixed 256) 256 (FixedGrid 1 1 1)))
        (check (PhaseLink (OperationId 1) "chunks" "combine"))
        (check (KernelPhase (OperationId 2) "offsets" (Fixed 1024) 64 (AutomaticGrid)))
        (check (PhaseLink (OperationId 2) "combine" "offsets"))
        (check (= (Plan (OperationId 3)) (LocalFold 64)))
        (check (KernelPhase (OperationId 3) "local" (Fixed 64) 64 (FixedGrid 1 1 1)))
        (check (= (Plan (OperationId 4)) (Elements)))
        (check (KernelPhase (OperationId 4) "elements" (Fixed 5) 64 (AutomaticGrid)))
        (check (= (Plan (OperationId 5)) (LocalFold 1)))
        (check (KernelPhase (OperationId 5) "ordered" (Fixed 1) 1 (FixedGrid 1 1 1)))
    "#,
        )
        .unwrap();
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
            query.for_each("CheckStatement", |row| {
                if query.lookup("ScopeRoot", (row[0],))? != Some(function) { return Ok(()); }
                let Some(loop_) = query.enode("Loop", row[3])? else { return Ok(()); };
                if query.enode("While", loop_[1])?.is_none() { return Ok(()); }
                let scope = query.constructor("NestedScope", (row[0], row[1], 0i64))?.unwrap();
                assert_eq!(query.flag("HostScope", (scope,))?, dispatched, "{source}");
                checked += 1;
                Ok(())
            }).unwrap();
            assert_eq!(checked, 1);
        });
    }
}

// Compile related cases once; keep every assertion tied to its source entry.

#[test]
fn fixed_collective_launches_share_geometry_and_storage() {
    for (size, groups, combine) in [
        (0, 1, 32),
        (1, 1, 32),
        (257, 2, 32),
        (8193, 33, 64),
        (65537, 256, 256),
    ] {
        planned(
            &format!("entry main(xs:[{size}]i32) [{size}]i32=scan((+),0,xs)"),
            |graph| {
                let query = Query(graph);
                let scan = operations(graph, "scan")[0];
                let plan = query.required("SelectedPlan", (scan,)).unwrap();
                let owner = query.required("Owner", (plan,)).unwrap();
                assert_eq!(
                    graph.value_to_base::<i64>(query.required("CollectiveGroups", (owner,)).unwrap()),
                    groups
                );
                assert_eq!(
                    graph.value_to_base::<i64>(query.required("CollectiveCombineWidth", (owner,)).unwrap()),
                    combine
                );
                let stage = query.constructor("Stage", (owner, "chunks")).unwrap().unwrap();
                let grid = query.record("LaunchGrid", "FixedGrid", (stage,)).unwrap().unwrap();
                assert_eq!(graph.value_to_base::<i64>(grid[0]), groups);
                for role in ["partial", "offset"] {
                    let temporary = query.constructor("Temporary", (owner, role, 0i64)).unwrap().unwrap();
                    let extent = query.record("CapacityExtent", "Fixed", (temporary,)).unwrap().unwrap();
                    assert_eq!(graph.value_to_base::<i64>(extent[0]), groups, "{role}");
                }
            },
        );
    }
}

#[test]
fn fluid_radix_scan_avoids_empty_groups_and_oversized_carries() {
    let source = format!(
        "{}\nentry main(xs:[4320]vec2u32) [4320]vec2u32 = radix_sort_step(xs,|i,p|u32.get_bit(i,p.x),0)",
        include_str!("../../../pkg/sort/src/four_bit.wyn")
    );
    planned(&source, |graph| {
        let query = Query(graph);
        let scan = operations(graph, "scan")[0];
        let plan = query.required("SelectedPlan", (scan,)).unwrap();
        let owner = query.required("Owner", (plan,)).unwrap();
        let width = graph.value_to_base::<i64>(query.required("CollectiveChunkWidth", (owner,)).unwrap());
        let groups = graph.value_to_base::<i64>(query.required("CollectiveGroups", (owner,)).unwrap());
        let combine =
            graph.value_to_base::<i64>(query.required("CollectiveCombineWidth", (owner,)).unwrap());
        assert_eq!(width, 64);
        assert!(
            (1..=(4320 + width - 1) / width).contains(&groups),
            "no empty groups"
        );
        assert_eq!(combine, (groups as u64).next_power_of_two().max(32) as i64);
    });
}

#[test]
fn observing_an_intermediate_filter_length_preserves_its_cardinality() {
    planned("entry main(xs:[]i32) (i32,i32)=let a=filter(|x|x>0,xs) let b=filter(|x|x<20,map(|x|x+1,a)) in (length(a),length(b))", |graph| {
        let mut totals = Vec::new();
        Query(graph).for_each("TotalResult", |row| { totals.push((row[0], row[1])); Ok(()) }).unwrap();
        assert_eq!(totals.len(), 2, "each observed cardinality needs its own count");
    });
}

#[test]
fn guarded_streams_feed_updates_and_prefixes_without_a_compaction_buffer() {
    planned_cases(
        &[
            "entry main(xs:[]i32) [3]i32=let ys=filter(|x|x!=0,xs)
             let dest=[-1,-1,-1] in scatter(dest,map(|x|x-1,ys),map(|x|100/x,ys))",
            "entry main(xs:[]i32) []i32=map(|x|x*2,scan((+),0,map(|x|100/x,filter(|x|x!=0,xs))))",
        ],
        |case, graph, region| {
            let query = Query(graph);
            let filter = operations_in(graph, "filter", region)[0];
            let consumer = operations_in(graph, ["scatter", "scan"][case], region)[0];
            assert_eq!(
                query.required("SelectedPlan", (filter,)).unwrap(),
                query.required("SelectedPlan", (consumer,)).unwrap()
            );
        },
    );
}

#[test]
fn fusion_compares_logical_shapes_of_independent_inputs() {
    let sources = [17, 19].map(|size| format!(
        "entry main(xs:[17]i32,ys:[{size}]i32) (i32,i32)=(reduce((+),0,xs),reduce((+),0,ys))"
    ));
    fused_cases(&sources, |index, graph, function| {
        let query = Query(graph);
        let parallels = top_level(graph, function, "Parallel");
        assert_eq!(parallels.len(), if index == 0 { 1 } else { 2 });
        let folds = parallels.into_iter().map(|statement| {
            let statement = query.fields("Let", statement).unwrap();
            let parallel = query.fields("Parallel", statement[3]).unwrap();
            let body = query.fields("ParallelBlock", parallel[2]).unwrap();
            query.vector(body[1]).unwrap().into_iter()
                .filter(|clause| query.enode("Fold", *clause).unwrap().is_some()).count()
        }).sum::<usize>();
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
        graph.parse_and_run_program(None, &format!(r#"
            (ruleset escape-update)
            (rule ((CurrentProgram program) (ContainsStm (ProgramRoot program)
                (Let id pattern certs (Accumulate bounds token indices values))))
              ((set (PendingRewrite "escape" program) (set-of
                (Rename (map-empty) (map-empty) (map-empty) (map-empty)
                  (map-insert (map-empty) id (vec-of
                    (Let id pattern certs (Accumulate bounds {token} indices {values})))))))
               (RewriteDemand "escape" program)) :ruleset escape-update)
            (run-schedule escape-update (saturate memory-rewrite memory-rename))
        "#)).unwrap();
        let error = super::super::memory::validate(&mut graph).unwrap_err();
        assert!(error.to_string().contains("outside its scope"), "{error}");
    }
}

#[test]
fn output_coordinate_maps_accept_permutations_and_reject_overlapping_digits() {
    let coordinates = [
        ("a/2", "a%2", 8, true),
        ("a%8", "a/8", 8, true),
        ("4*(a/8)+a%4", "(a/4)%2", 8, true),
        ("4*(a%2)+2*((a/4)%2)+a/8", "(a/2)%2", 8, true),
        ("a%8", "a%2", 8, false),
        ("a%8", "a/8", 4, false),
        ("(a/2)%4+4*(a/3)", "a%2", 8, false),
    ];
    let sources = coordinates.map(|(row, column, bound, _)| {
        format!(
            "entry main() [16]i32=let rows=map(|i|[10*i,10*i+1],iota(8)) in
         map(|a|let r={row} in if r<{bound} then rows[r][{column}] else -1,iota(16))"
        )
    });
    planned_cases(&sources, |index, graph, function| {
        let launches = top_level(graph, function, "Launch");
        assert_eq!(
            launches.len(),
            if coordinates[index].3 { 1 } else { 2 },
            "a complete coordinate permutation removes the intermediate dispatch: {}",
            sources[index]
        );
    });
}

#[test]
fn independent_update_scopes_share_a_loop_and_keep_distinct_exports() {
    for (source, count) in [
        ("entry main(xs:[4]i32) ([3]i32,[5]i32)=let da=[0,0,0] let db=[1,1,1,1,1] let a=reduce_by_index(da,(+),0,map(|x|x%3,xs),xs) let b=reduce_by_index(db,|a,b|a+b+1,0,map(|x|x%5,xs),xs) in (a,b)", 2),
        ("entry main(xs:[4]i32) ([4]i32,[3]i32,[3]i32)=let p=scan((+),0,xs) let da=[1,1,1] let db=[2,2,2] let a=reduce_by_index(da,(+),0,map(|x|x%3,p),p) let b=reduce_by_index(db,|a,b|a+b+1,0,map(|x|x%3,p),p) in (p,map(|x|2*x,a),b)", 2),
        ("entry main() ([2]u32,u32,[2]u32,u32)=let da=[[-1,-1],[-1,-1]] let db=[[-2],[-2]] let (a,ac,ao)=bucket_scatter_2d(da,[[(0,10),(0,11),(1,20)]]) let (b,bc,bo)=bucket_scatter_2d(db,[[(0,30),(0,31),(1,40)]]) in (ac,ao,bc,bo)", 6),
    ] {
        let tlc = compile_thru_tlc(source).unwrap();
        let program = selected(&tlc);
        let graph = &program.graph;
        let query = Query(graph);
        let mut scopes = Vec::new();
        query.for_function("PlanUpdates", |keys, scope| {
            let owner = query.required("Owner", (keys[0],))?;
            if query.lookup("SelectedPlan", (owner,))? == Some(keys[0]) {
                scopes.push((owner, scope));
            }
            Ok(())
        }).unwrap();
        assert_eq!(scopes.len(), 1, "{source}");
        for scan in operations(graph, "scan") {
            assert_eq!(query.required("SelectedPlan", (scan,)).unwrap(), query.required("SelectedPlan", (scopes[0].0,)).unwrap());
        }
        let scope = query.enode("WithUpdates", scopes[0].1).unwrap().unwrap();
        assert_eq!(graph.value_to_container::<VecContainer>(scope[0]).unwrap().data.len(), count);
        assert_eq!(graph.value_to_container::<VecContainer>(scope[1]).unwrap().data.len(), 2);
        let ssa = finish_for_test(program, CodegenTarget::Wgsl).unwrap();
        crate::lower_ssa_to_wgsl(ssa).unwrap();
    }
}

#[test]
fn closed_array_regions_distribute_before_physical_placement() {
    let cases = [
        ("entry main(xs:[4]i32,ys:[4]i32,flag:bool) [4]i32=if flag then map(|x|x+1,xs) else map(|y|y+2,ys)", true),
        ("entry main(xs:[4]i32,ys:[4]i32,flag:bool) [4](i32,i32)=if flag then map(|x|(x,x+1),xs) else map(|y|(y,y+2),ys)", true),
        ("entry main(xs:[]i32,ys:[]i32,flag:bool) []i32=if flag then map(|x|x+1,xs) else map(|y|y+2,ys)", false),
        ("entry main(xs:[0]i32,k:i32) [0]i32=if 1/k>0 then map(|x|x+1,xs) else map(|x|x+2,xs)", false),
    ];
    planned_cases(&cases.map(|(source, _)| source), |index, graph, scope| {
        let query = Query(graph);
        let operations = operations_in(graph, "control", scope);
        assert_eq!(operations.len(), 1);
        let owner = operations[0];
        let plan = query.required("SelectedPlan", (owner,)).unwrap();
        assert_eq!(
            query.enode("Distributed", plan).unwrap().is_some(),
            cases[index].1
        );
        if cases[index].1 {
            let stage = query.constructor("Stage", (owner, "elements")).unwrap().unwrap();
            assert!(query.contains("Phase", (stage, scope)).unwrap());
            let region = query.required("SourceOperationValue", (owner,)).unwrap();
            let branch = query.record("SourceBranch", "BranchInfo", (region,)).unwrap().unwrap();
            for arm in &branch[1..] {
                assert!(!query.contains("HostRegion", (*arm,)).unwrap());
                let body = query.record("RegionDefinition", "RegionBody", (*arm,)).unwrap().unwrap();
                assert!(!graph.value_to_container::<VecContainer>(body[1]).unwrap().data.is_empty());
            }
        }
    });
}

#[test]
fn ranked_fusion_preserves_prefix_evaluation_and_retained_rows() {
    let cases = [
        ("entry main(dest:*[4][8]i32) ([4][8]i32,[4]u32,u32)=let items:[4][6](i32,i32)=map(|bucket|let offset=bucket*2 let shifted=offset+1 in map(|pair|((shifted+pair)%4,shifted+pair),iota(6)),iota(4)) in bucket_scatter_2d(dest,items)", true),
        ("entry main(dest:*[4][8]i32,k:i32) ([4][8]i32,[4]u32,u32)=let items:[4][0](i32,i32)=map(|bucket|let offset=1/k in map(|pair|(bucket,offset+pair),iota(0)),iota(4)) in bucket_scatter_2d(dest,items)", false),
        ("entry main(dest:*[4][8]i32) ([4][6](i32,i32),([4][8]i32,[4]u32,u32))=let items:[4][6](i32,i32)=map(|bucket|map(|pair|((bucket+pair)%4,bucket+pair),iota(6)),iota(4)) in (items,bucket_scatter_2d(dest,items))", false),
    ];
    planned_cases(&cases.map(|(source, _)| source), |index, graph, scope| {
        let query = Query(graph);
        let producer = operations_in(graph, "map", scope)[0];
        let consumer = operations_in(graph, "bucket-scatter", scope)[0];
        let a = query.required("SelectedPlan", (producer,)).unwrap();
        let b = query.required("SelectedPlan", (consumer,)).unwrap();
        assert_eq!(a == b, cases[index].1, "case {index}");
        let shape = query.required("PlanDomain", (b,)).unwrap();
        let axes = query.required("ShapeDimensions", (shape,)).unwrap();
        assert_eq!(
            graph.value_to_container::<VecContainer>(axes).unwrap().data.len(),
            2
        );
        if index == 1 {
            let owner = query.required("Owner", (a,)).unwrap();
            assert_eq!(
                query.required("IterationRank", (owner,)).map(|v| graph.value_to_base::<i64>(v)).unwrap(),
                1
            );
        }
    });
}

#[test]
fn output_coordinate_maps_cover_every_axis_of_ranked_producers() {
    let sources = [
        "entry main() [12]i32=let rows=map(|i|[[10*i,10*i+1],[10*i+2,10*i+3]],iota(3)) in map(|a|rows[a%3][(a/3)%2][a/6],iota(12))",
        "entry main() [24]i32=let rows=map(|i|map(|j|map(|k|100*i+10*j+k,iota(4)),iota(3)),iota(2)) in map(|a|rows[a%2][(a/2)%3][a/6],iota(24))",
    ];
    let tlc = compile_thru_tlc(&cases_source(&sources)).unwrap();
    let program = selected(&tlc);
    check_entries(&program, sources.len(), |index, graph, function| {
        assert_eq!(
            top_level(graph, function, "Launch").len(),
            1,
            "{}",
            sources[index]
        );
    });
    let prepared = super::super::prepare_host(program).unwrap();
    let ssa = super::super::to_ssa(prepared, CodegenTarget::Portable).unwrap();
    crate::lower_ssa_to_spirv(ssa.clone()).unwrap();
    crate::lower_ssa_to_wgsl_with_program(ssa).unwrap();
}

#[test]
fn calibrated_costs_select_among_the_same_legal_map_executions() {
    let tlc = compile_thru_tlc("entry main(xs:[4][513]i32) [4]i32=map(|row|reduce((+),0,row),xs)").unwrap();
    for (barrier_ns, scope) in [(0.0, "WorkgroupScope"), (1e9, "InvocationScope")] {
        let placed = place(
            fuse(from_tlc(&tlc).unwrap()).unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap();
        let program = super::super::schedule_with_cost_model(
            placed,
            false,
            super::super::cost::CostModel {
                barrier_ns,
                ..Default::default()
            },
        )
        .unwrap();
        let query = Query(&program.graph);
        let mut checked = 0;
        query
            .for_each("MapPlan", |row| {
                if query.flag("MapHost", (row[0],))? {
                    let traversal = query.required("MapOutputTraversal", (row[1], 0i64))?;
                    assert!(query.contains("GroupSafeTraversal", (traversal,))?);
                    assert!(query.contains("ParallelTraversalWork", (traversal,))?);
                    let execution = query.required("MapExecution", (row[1],))?;
                    assert!(query.enode(scope, execution)?.is_some());
                    checked += 1;
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(checked, 1);
        let program = super::super::prepare_interfaces(program).unwrap();
        let program =
            super::super::optimize_with_policy(program, super::super::ScalarOptimization::Basic).unwrap();
        finish_for_test(program, CodegenTarget::Wgsl).unwrap();
    }
}

#[test]
fn memory_layout_slices_compose_across_axes_and_negative_strides() {
    let mut graph = EGraph::default();
    super::super::memory::load(&mut graph).unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (let base (Layout (IndexConstant 7)
            (vec-of (Dim (IndexConstant 9) (IndexConstant 4))
                    (Dim (IndexConstant 1) (IndexConstant 9)))))
        (let slices (vec-of
            (Slice (IndexConstant 1) (IndexConstant 2) (IndexConstant 2))
            (Slice (IndexConstant 4) (IndexConstant 5) (IndexConstant -1))))
        (SliceLayoutAt base slices 0 (IndexConstant 0) (vec-of))
        (run-schedule (saturate memory-allocation))
        (check (= (SliceLayout base slices) (Layout offset dimensions))
            (= (NormalizedIndex offset) (IndexConstant 20))
            (= (vec-get dimensions 0) (Dim outer-stride (IndexConstant 2)))
            (= (NormalizedIndex outer-stride) (IndexConstant 18))
            (= (vec-get dimensions 1) (Dim inner-stride (IndexConstant 5)))
            (= (NormalizedIndex inner-stride) (IndexConstant -1)))
    "#,
        )
        .unwrap();
}
