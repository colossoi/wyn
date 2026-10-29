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
    let mut seen = vec![false; count];
    Query(&program.graph)
        .for_each("SourceOriginalEntry", |row| {
            let symbol = program.identities.symbols.resolve(program.graph.value_to_base::<i64>(row[0]));
            let name = program.source.symbols.get(*symbol).unwrap();
            let index: usize = name.strip_prefix("case_").unwrap().parse().unwrap();
            seen[index] = true;
            check(index, &program.graph, row[1]);
            Ok(())
        })
        .unwrap();
    assert!(seen.into_iter().all(|present| present));
}

fn planned_cases(sources: &[impl AsRef<str>], check: impl Fn(usize, &EGraph, Value)) {
    let tlc = compile_thru_tlc(&cases_source(sources)).unwrap();
    let program = schedule(
        place(
            fuse(from_tlc(&tlc).unwrap()).unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap(),
    )
    .unwrap();
    check_entries(&program, sources.len(), check);
}

fn operations(graph: &EGraph, kind: &str) -> Vec<Value> {
    let mut operations = Vec::new();
    Query(graph)
        .for_function("SourceOperation", |row, info| {
            let fields = Query(graph).enode("OperationInfo", info)?.unwrap();
            if graph.value_to_base::<S>(fields[1]).as_str() == kind {
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
        .filter(|&op| {
            Query(graph).record("SourceOperation", "OperationInfo", (op,)).unwrap().unwrap()[0] == region
        })
        .collect()
}

#[test]
fn scan_output_can_capture_a_fused_total() {
    planned(
        "entry main(xs: [257]i32) [257]i32 =
         let prefix = scan((+), 0, xs) in
         let total = reduce((+), 0, xs) in
         map(|x| x + total, prefix)",
        |graph| {
            let query = Query(graph);
            let scan = operations(graph, "scan")[0];
            let reduce = operations(graph, "reduce")[0];
            let plan = query.required("SelectedPlan", (scan,)).unwrap();
            assert_eq!(plan, query.required("SelectedPlan", (reduce,)).unwrap());
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
                .for_each("Phase", |_| {
                    phases += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(phases, 3);
        },
    );
}

#[test]
fn radix_step_has_three_phases() {
    let source = format!(
        "{}\n entry main(xs: []i32) []i32 = radix_sort_step(xs, |i:i32,x:i32| (x >> i) & 1, 0)",
        include_str!("../../../pkg/sort/src/radix_sort.wyn")
    );
    planned(&source, |graph| {
        let query = Query(graph);
        let mut phases = Vec::new();
        query
            .for_each("Phase", |row| {
                let stage = query.enode("Stage", row[0])?.unwrap();
                let info = query.record("SourceOperation", "OperationInfo", (stage[0],))?.unwrap();
                phases.push((
                    graph.value_to_base::<S>(stage[1]).to_string(),
                    graph.value_to_base::<S>(info[1]).to_string(),
                ));
                Ok(())
            })
            .unwrap();
        phases.sort();
        assert_eq!(
            phases,
            vec![
                ("chunks".into(), "scatter".into()),
                ("combine".into(), "scatter".into()),
                ("offsets".into(), "scatter".into())
            ]
        );
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
    planned_cases(&sources, |index, graph, region| {
        let query = Query(graph);
        let scans = operations_in(graph, "scan", region);
        assert_eq!(scans.len(), 1);
        let plan = query.required("SelectedPlan", (scans[0],)).unwrap();
        let owner = query.required("Owner", (plan,)).unwrap();
        let width = query.required("CollectiveChunkWidth", (owner,)).unwrap();
        assert_eq!(graph.value_to_base::<i64>(width), cases[index].1, "case {index}");
    });
}

#[test]
fn four_bit_radix_step_has_three_phases() {
    let source = format!(
        "{}\n entry main(xs: []vec2u32) []vec2u32 = radix_sort_step(xs, |i,p| u32.get_bit(i,p.x), 0)",
        include_str!("../../../pkg/sort/src/four_bit.wyn")
    );
    planned(&source, |graph| {
        let query = Query(graph);
        let scan = operations(graph, "scan")[0];
        let plan = query.required("SelectedPlan", (scan,)).unwrap();
        let owner = query.required("Owner", (plan,)).unwrap();
        let width = query.required("CollectiveChunkWidth", (owner,)).unwrap();
        assert_eq!(graph.value_to_base::<i64>(width), 64);
        let mut phases = Vec::new();
        query
            .for_each("Phase", |row| {
                let stage = query.enode("Stage", row[0])?.unwrap();
                let info = query.record("SourceOperation", "OperationInfo", (stage[0],))?.unwrap();
                phases.push((
                    graph.value_to_base::<S>(stage[1]).to_string(),
                    graph.value_to_base::<S>(info[1]).to_string(),
                ));
                Ok(())
            })
            .unwrap();
        phases.sort();
        assert_eq!(
            phases,
            vec![
                ("chunks".into(), "scatter".into()),
                ("combine".into(), "scatter".into()),
                ("offsets".into(), "scatter".into())
            ]
        );
    });
}

#[test]
fn scan_input_cannot_capture_a_fused_total() {
    planned_cases(
        &[
            "entry main(xs:[]i32) []i32 = let t=reduce((+),0,xs) in scan((+),0,map(|x|x+t,xs))",
            "entry main(xs:[]i32) []i32 = let t=reduce((+),0,xs) in scan((+),t,xs)",
            "entry main(xs:[]i32) []i32 = let t=reduce((+),0,xs) in scan(|a,b|a+b+t,0,xs)",
        ],
        |index, graph, region| {
            let query = Query(graph);
            let scans = operations_in(graph, "scan", region);
            let reductions = operations_in(graph, "reduce", region);
            assert_eq!((scans.len(), reductions.len()), (1, 1));
            assert_ne!(
                query.required("SelectedPlan", (scans[0],)).unwrap(),
                query.required("SelectedPlan", (reductions[0],)).unwrap(),
                "case {index}"
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
                .for_each("SourceResultOperand", |row| {
                    let view = query.required("ArrayView", (row[0],))?;
                    assert_eq!(view, query.required("ArrayView", (row[2],))?);
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
                    let view = query.record("ArrayView", "ViewInfo", (row[1],))?.unwrap();
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
    planned(
        "entry main(xs:[]i32) []i32 =
         loop out=xs for i<2 do out[0..length(out)-1]",
        |graph| {
            let query = Query(graph);
            let mut loops = 0;
            query
                .for_function("SourceLoop", |row, _| {
                    let shape = query.record("LogicalExtent", "Scalar", (row[0],))?.unwrap();
                    assert_eq!(query.enode("SourceShape", shape[0])?, Some(vec![row[0]]));
                    loops += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(loops, 1);
        },
    );
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

#[test]
fn compiler_repro_local_capacity_plan() {
    planned(
        include_str!("../../../testfiles/regressions/local_array_capacity.wyn"),
        |graph| {
            let query = Query(graph);
            let mut buffers = 0;
            query
                .for_each("LocalBuffer", |row| {
                    let capacity = query.required("PhysicalLocalCapacity", (row[4],))?;
                    assert_eq!(graph.value_to_base::<i64>(capacity), 3);
                    buffers += 1;
                    Ok(())
                })
                .unwrap();
            assert!(buffers > 0);
        },
    );
}

#[test]
fn compiler_repro_nested_capacity_plan() {
    let source =
        "entry main(xs:[]i32) []i32=map(|x| let a=loop a=[x,x+1,x+2] for k<2 do map(|y|y+1,a) in a[0],xs)";
    planned(source, |graph| {
        let query = Query(graph);
        let mut capacities = Vec::new();
        query
            .for_each("LocalBuffer", |row| {
                let capacity = query.required("PhysicalLocalCapacity", (row[4],))?;
                capacities.push(graph.value_to_base::<i64>(capacity));
                let slot = query.required("BufferSlot", (row[0], row[1], row[2]))?;
                query.required("LocalElementLayout", (slot,))?;
                Ok(())
            })
            .unwrap();
        assert!(capacities.contains(&3));
        assert!(capacities.iter().all(|&capacity| capacity == 1 || capacity == 3));
    });
}

#[test]
fn compiler_repro_array_reduce_preserves_effect_barrier() {
    for source in ["entry main(xs:[][2]i32) [2]i32 =
        reduce(|a,b|map(|i|a[i]+b[i],iota(2)),[0,0],map(|row|map(|x|x+1,row),xs))"]
    {
        let tlc = compile_thru_tlc(source).unwrap();
        let imported = from_tlc(&tlc).unwrap();
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
                        belongs |= query.record("SourceOperation", "OperationInfo", (member[1],))?.unwrap()
                            [0]
                            == region;
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
fn indexed_phase_domains_preserve_validation() {
    let tlc = compile_thru_tlc("entry main(xs:[]i32) []i32=map(|x|x+1,xs)").unwrap();
    let program = selected(&tlc);
    let mut prepared = super::super::prepare_host(program).unwrap();
    let baseline = prepared.program.graph.clone();
    for (mode, expected) in [
        ("missing", "stage domain missing"),
        ("ambiguous", "ambiguous PhaseDomain"),
        ("zero-width", "stage width must be positive"),
    ] {
        prepared.program.graph = baseline.clone();
        let program = &mut prepared.program;
        let domain = Query(&program.graph).row("PhaseDomain", |_| true).unwrap().unwrap();
        if mode != "ambiguous" {
            program.graph.clear_function("PhaseDomain").unwrap();
        }
        if mode != "missing" {
            program
                .graph
                .update(|mut sink| {
                    sink.add("PhaseDomain", (domain[0], domain[1], 0i64))?;
                    Ok(())
                })
                .unwrap();
        }
        let Err(error) = super::super::to_ssa::lower(program, CodegenTarget::Wgsl) else {
            panic!("invalid {mode} phase domain was accepted");
        };
        assert!(error.to_string().contains(expected), "{mode}: {error}");
    }
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
    let mut rows = Vec::new();
    Query(&program.graph)
        .for_each("Accumulator", |row| {
            rows.push(row.to_vec());
            Ok(())
        })
        .unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][0], rows[1][0], "the scan and reduction must be fused");
    assert_ne!(
        rows[0][2], rows[1][2],
        "different types expose incorrect slot ordering"
    );
    let slot = rows[0][1];
    rows[0][1] = rows[1][1];
    rows[1][1] = slot;
    program.graph.clear_function("Accumulator").unwrap();
    program
        .graph
        .update(|mut sink| {
            for row in rows {
                sink.add("Accumulator", egglog_engine::RawValues(row))?;
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
                if graph.value_to_base::<S>(row[3]).as_str() == "count" {
                    assert_eq!(slot, graph.value_to_base::<i64>(row[4]) + 1);
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
    planned_cases(&sources, |index, graph, scope| {
        let (_source, kind, rank, count) = cases[index];
        let query = Query(graph);
        let matching = operations(graph, kind)
            .into_iter()
            .filter(|&operation| query.required("OperationRegion", (operation,)).unwrap() == scope)
            .collect::<Vec<_>>();
        let [operation] = matching[..] else {
            panic!("one {kind}")
        };
        assert_eq!(
            graph.value_to_base::<i64>(query.required("SourceDomainRank", (operation,)).unwrap()),
            rank
        );
        let extent = query.required("SourceDomainExtent", (operation,)).unwrap();
        assert_eq!(
            graph.value_to_base::<i64>(query.required("StaticCapacityBound", (extent,)).unwrap()),
            count
        );
        for axis in 0..rank {
            query.required("SourceDomainAxis", (operation, axis)).unwrap();
        }
        query
            .for_function("SourceInputRank", |keys, value| {
                if keys[0] == operation {
                    for dimension in 0..graph.value_to_base::<i64>(value) {
                        let axis = graph.value_to_base::<i64>(
                            query.required("SourceInputDimension", (operation, keys[1], dimension))?,
                        );
                        assert!((0..rank).contains(&axis));
                    }
                }
                Ok(())
            })
            .unwrap();
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
            assert_eq!(query.required("PlanDomain", (plan,)).unwrap(), filter);
            let input = query.required("SourceInput", (filter, 0i64)).unwrap();
            let extent = query.required("SourceExtent", (input,)).unwrap();
            let owner = query.required("Owner", (plan,)).unwrap();
            assert_eq!(query.required("InputDomain", (owner,)).unwrap(), extent);
        },
    );
}

#[test]
fn nested_domains_retain_distinct_axes_and_capture_origins() {
    planned(
        "entry main(xs:[4]i32) [4][4]i32=map(|x|map(|y|x+y,xs),xs)",
        |graph| {
            let query = Query(graph);
            let maps = operations(graph, "map");
            assert_eq!(maps.len(), 2);
            let mut bodies = Vec::new();
            for operation in maps {
                let body = query.required("SourceOperatorBody", (operation,)).unwrap();
                query.required("SourceParameter", (body, 0i64)).unwrap();
                query.required("SourceDomainAxis", (operation, 0i64)).unwrap();
                bodies.push(body);
            }
            assert_ne!(bodies[0], bodies[1]);
            let mut capture_origins = Vec::new();
            query
                .for_function("SourceCapture", |row, actual| {
                    if bodies.contains(&row[0]) {
                        if let Some(origin) = query.enode("SourceFormal", actual)? {
                            capture_origins.push((row[0], origin[0]));
                        }
                    }
                    Ok(())
                })
                .unwrap();
            assert!(
                !capture_origins.is_empty(),
                "inner map must retain its outer element capture"
            );
            assert!(capture_origins.iter().all(|(capture, origin)| capture != origin));
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
            let [operation] = operations(graph, "bucket-scatter")[..] else {
                panic!("one bucket scatter")
            };
            for input in 0..2i64 {
                let rank = query.required("SourceInputRank", (operation, input)).unwrap();
                assert_eq!(graph.value_to_base::<i64>(rank), 1);
                let axis = query.required("SourceInputDimension", (operation, input, 0i64)).unwrap();
                assert_eq!(graph.value_to_base::<i64>(axis), input);
            }
            let extent = query.required("SourceDomainExtent", (operation,)).unwrap();
            let product = query.enode("Product", extent).unwrap().unwrap();
            for (axis, factor) in product.into_iter().enumerate() {
                let input = query.required("SourceInput", (operation, axis as i64)).unwrap();
                assert_eq!(factor, query.required("SourceExtent", (input,)).unwrap());
            }
        },
    );
}

#[test]
fn map_and_bucket_emission_require_selected_axes_not_source_domains() {
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
    for table in ["SourceDomainAxis", "SourceDomainRank", "PlanDomain"] {
        prepared.program.graph.clear_function(table).unwrap();
    }
    super::super::to_ssa::lower(&prepared.program, CodegenTarget::Wgsl).unwrap();
    let mut owners = Vec::new();
    Query(&prepared.program.graph)
        .for_function("IterationRank", |keys, _| {
            owners.push(keys[0]);
            Ok(())
        })
        .unwrap();
    assert!(owners.len() >= 2);
    prepared.program.graph.clear_function("IterationAxis").unwrap();
    let facts = super::super::facts::Facts {
        program: &prepared.program,
    };
    for owner in owners {
        let error = facts.iteration_dimensions(owner).unwrap_err();
        assert!(error.to_string().contains("domain axis has no planned extent"));
    }
}

#[test]
fn rectangular_maps_expand_launch_without_expanding_row_capacity() {
    let source = "entry main(xs:[4]i32) [4][3]i32=map(|x|map(|y|x+y,iota(3)),xs)";
    planned(source, |graph| {
        let query = Query(graph);
        let mut flattened = 0;
        query
            .for_function("IterationRank", |keys, rank| {
                if graph.value_to_base::<i64>(rank) == 2 {
                    flattened += 1;
                    let launch = query.required("IterationExtent", (keys[0],))?;
                    assert_eq!(
                        graph.value_to_base::<i64>(query.required("StaticCapacityBound", (launch,))?),
                        12
                    );
                    let outer = query.required("IterationAxis", (keys[0], 0i64))?;
                    assert_eq!(query.required("InputDomain", (keys[0],))?, outer);
                    assert_eq!(
                        graph.value_to_base::<i64>(query.required("StaticCapacityBound", (outer,))?),
                        4
                    );
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(flattened, 1);
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
    // Remove source operation classification; immutable inputs and callbacks remain
    // shared by source analysis and the fused plan.
    program.graph.clear_function("SourceOperation").unwrap();
    let program = schedule(program).unwrap();
    let graph = &program.graph;
    let query = Query(graph);
    let mut ranks = Vec::new();
    query
        .for_function("IterationRank", |_, rank| {
            ranks.push(graph.value_to_base::<i64>(rank));
            Ok(())
        })
        .unwrap();
    assert!(ranks.contains(&2));

    let mut program = super::super::optimize_with_policy(
        super::super::prepare_interfaces(program).unwrap(),
        super::super::ScalarOptimization::Basic,
    )
    .unwrap();
    program.graph.clear_function("SourceOperation").unwrap();
    finish_for_test(program, CodegenTarget::Wgsl).unwrap();
}

#[test]
fn rectangular_maps_ignore_scalar_recomputation_cost() {
    let mut expression = String::from("x");
    for i in 1..=40 {
        expression = format!("({expression} * x + {i})");
    }
    let source = format!(
        "def costly(x:i32) i32={expression}
         entry main(xs:[4]i32) [4][3]i32=
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
        let mut flattened = false;
        query
            .for_function("IterationRank", |_, rank| {
                flattened |= graph.value_to_base::<i64>(rank) == 2;
                Ok(())
            })
            .unwrap();
        assert!(
            flattened,
            "legal scalar recomputation must not block nested parallelism"
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
        ("entry main(xs:[]i32) [][3]i32=map(|x|map(|y|reduce(|a,b|a+b,0,[x,y]),iota(3)),xs)", 2),
        ("entry main(xs:[2]i32) [2][0]i32=map(|x|map(|y|x+y,iota(0)),xs)", 2),
        ("entry main(xs:[]i32) [][3]i32=map(|x|let t=loop a=x for i<16 do a+i in map(|y|t+y,iota(3)),xs)", 1),
        ("entry main(xs:[]i32) [][3]i32=map(|x|let t=reduce(|a,b|a+b,0,[x,x,x]) in map(|y|t+y,iota(3)),xs)", 1),
    ];
    let sources = cases.iter().map(|case| case.0).collect::<Vec<_>>();
    planned_cases(&sources, |index, graph, scope| {
        let (source, expected_rank) = cases[index];
        let query = Query(graph);
        let mut ranks = Vec::new();
        query
            .for_function("IterationRank", |keys, rank| {
                let region = query.required("OperationRegion", (keys[0],))?;
                let host = query.contains("HostRegion", (region,))?;
                if host && region == scope {
                    ranks.push(graph.value_to_base::<i64>(rank));
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(ranks, vec![expected_rank], "{source}");
    });
}

#[test]
fn fused_outputs_only_share_a_rectangular_launch_when_inner_shapes_agree() {
    let sources = [3, 5].map(|inner| {
        format!(
            "entry main(xs:[4]i32) ([4][3]i32,[4][{inner}]i32)=
        (map(|x|map(|y|x+y,iota(3)),xs), map(|x|map(|y|x*y,iota({inner})),xs))"
        )
    });
    planned_cases(&sources, |index, graph, region| {
        let expected_rank = [2, 1][index];
        let query = Query(graph);
        let mut fused = 0;
        query
            .for_each("MapPlan", |row| {
                if query.required("OperationRegion", (row[1],))? != region
                    || !query.flag("MapHost", (row[0],))?
                {
                    return Ok(());
                }
                let mut outputs = 0;
                query.for_each("PlanResult", |result| {
                    outputs += usize::from(result[0] == row[0]);
                    Ok(())
                })?;
                if outputs == 2 {
                    fused += 1;
                    let rank = query.required("IterationRank", (row[1],))?;
                    assert_eq!(graph.value_to_base::<i64>(rank), expected_rank);
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(fused, 1, "case {index}");
    });
}

#[test]
fn rectangular_runtime_dimensions_must_be_uniform_across_outer_coordinates() {
    planned_cases(
        &[
            "entry main(xs:[]i32,n:i32) [][]i32=map(|x|map(|y|x+y,iota(n)),xs)",
            "entry main(xs:[]i32,ys:[]i32) [][]i32=map(|x|map(|y|x+y,ys),xs)",
            "entry main(xs:[]i32,n:i32,m:i32) [][][]i32=map(|x|map(|y|map(|z|x+y+z,iota(m)),iota(n)),xs)",
            "entry main(xs:[]i32) [][]i32=map(|x|map(|y|x+y,iota(x)),xs)",
        ],
        |index, graph, region| {
            let query = Query(graph);
            let mut ranks = Vec::new();
            query
                .for_each("MapPlan", |row| {
                    let info = query.required("SourceOperation", (row[1],))?;
                    let owner = query.enode("OperationInfo", info)?.unwrap()[0];
                    if owner == region && query.flag("MapHost", (row[0],))? {
                        ranks.push(graph.value_to_base::<i64>(query.required("IterationRank", (row[1],))?));
                    }
                    Ok(())
                })
                .unwrap();
            assert_eq!(ranks, vec![[2, 2, 3, 1][index]], "case {index}");
        },
    );
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
                    sink.add("SegmentDemand", (value, 0i64))?;
                }
                if blocked {
                    sink.set("SourceRegionDuplicable", (callback,), false)?;
                }
                Ok(())
            })
            .unwrap();
        super::schedule(&mut graph).unwrap();
        let query = Query(&graph);
        assert_eq!(query.flag("SegmentPlanBlocked", (plan,)).unwrap(), blocked);
        for value in values {
            assert_eq!(query.required("SegmentCandidatePlan", (value,)).unwrap(), plan);
            assert_eq!(
                query.lookup("SegmentTerminal", (value, 0i64)).unwrap().is_some(),
                !blocked
            );
        }
    }
}

#[test]
fn segmented_collectives_select_workgroups_for_each_outer_coordinate() {
    planned_cases(
        &[
            "entry main(xs:[][513]i32) []i32=map(|row|reduce((+),0,row),xs)",
            "entry main(xs:[][513]i32) [][513]i32=map(|row|scan((+),0,row),xs)",
            "entry main(xs:[][3][17]i32) [][3]i32=map(|rows|map(|row|reduce((+),0,row),rows),xs)",
        ],
        |index, graph, region| {
            let query = Query(graph);
            let maps = operations_in(graph, "map", region);
            let mut groups = 0;
            query
                .for_function("MapSegmented", |keys, enabled| {
                    groups += usize::from(maps.contains(&keys[0]) && graph.value_to_base::<bool>(enabled));
                    Ok(())
                })
                .unwrap();
            assert_eq!(groups, 1, "case {index}");
        },
    );
}

#[test]
fn segmented_collectives_do_not_cross_callback_control_flow() {
    let sources = [
        "entry main(xs:[][17]i32) []i32=map(|row|if row[0]>0 then reduce((+),0,row) else 0,xs)",
        "entry main(xs:[][17]i32) []i32=map(|row|loop a=0 for i<2 do a+reduce((+),0,row),xs)",
        "entry main(xs:[]i32) []i32=map(|n|reduce((+),0,iota(n)),xs)",
    ];
    planned(&cases_source(&sources), |graph| {
        Query(graph)
            .for_function("MapSegmented", |_, enabled| {
                assert!(
                    !graph.value_to_base::<bool>(enabled),
                    "all three control-flow cases"
                );
                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn selected_map_bindings_replace_callback_discovery() {
    let source = r#"
def row(x:i32) [3]i32=map(|y|x+y,iota(3))
          entry case_0(xs:[]i32) [][3]i32=map(row,xs)
entry case_1(xs:[][3][17]i32) [][3]i32=
          map(|rows|map(|row|reduce((+),0,map(|x|x+row[0],row)),rows),xs)
entry case_2(xs:[][3][17]i32) [][3][17]i32=
          map(|rows|map(|row|map(|x|x*2,scan((+),0,map(|x|x+1,row))),rows),xs)
"#;
    let tlc = compile_thru_tlc(source).unwrap();
    let mut program = selected(&tlc);
    let query = Query(&program.graph);
    let mut expected = crate::LookupMap::default();
    query
        .for_each("SourceOriginalEntry", |row| {
            let symbol = program.identities.symbols.resolve(program.graph.value_to_base::<i64>(row[0]));
            let name = program.source.symbols.get(*symbol).unwrap();
            let index: usize = name.strip_prefix("case_").unwrap().parse().unwrap();
            expected.insert(row[1], [(2, 2), (2, 2), (2, 3)][index]);
            Ok(())
        })
        .unwrap();
    let mut outputs = 0;
    query
        .for_each("MapPlan", |row| {
            if !query.flag("MapHost", (row[0],))? {
                return Ok(());
            }
            let info = query.required("SourceOperation", (row[1],))?;
            let region = query.enode("OperationInfo", info)?.unwrap()[0];
            let (rank, output_rank) = expected[&region];
            let mut bindings = query.required("MapOutputBindings", (row[1], 0i64))?;
            let mut axes = Vec::new();
            loop {
                if let Some(fields) = query.enode("MapIndexBinding", bindings)? {
                    axes.push(program.graph.value_to_base::<i64>(fields[4]));
                    bindings = fields[5];
                } else if let Some(fields) = query.enode("MapCallBinding", bindings)? {
                    bindings = fields[3];
                } else {
                    assert!(query.enode("MapValueBinding", bindings)?.is_some());
                    break;
                }
            }
            assert_eq!(axes, (0..rank).collect::<Vec<_>>(), "{source}");
            let axes = query.required("MapOutputAxes", (row[1], 0i64))?;
            let axes = program.graph.value_to_container::<VecContainer>(axes).unwrap();
            assert_eq!(
                axes.data.iter().map(|&axis| program.graph.value_to_base::<i64>(axis)).collect::<Vec<_>>(),
                (0..output_rank).collect::<Vec<_>>(),
                "{source}"
            );
            outputs += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(outputs, 3, "{source}");
    let mut collectives = Vec::new();
    query
        .for_function("SourceOperatorBody", |keys, callback| {
            if !query.flag("PlanMapping", (keys[0],))? {
                collectives.push((keys[0], callback));
            }
            Ok(())
        })
        .unwrap();
    for table in [
        "SourceOperatorBody",
        "PlanCallee",
        "PlanCallArgument",
        "SegmentTerminal",
    ] {
        program.graph.clear_function(table).unwrap();
    }
    program
        .graph
        .update(|mut sink| {
            for (op, callback) in collectives {
                sink.set("SourceOperatorBody", (op,), callback)?;
            }
            Ok(())
        })
        .unwrap();
    finish_for_test(program, CodegenTarget::Wgsl).unwrap();
}

#[test]
fn runtime_segmented_collectives_require_uniform_dimensions() {
    planned_cases(
        &[
            "entry main(xs:[]i32,n:i32) []i32=map(|x|reduce((+),0,map(|y|x+y,iota(n))),xs)",
            "entry main(xs:[]i32,n:i32) [][]i32=map(|x|scan((+),0,map(|y|x+y,iota(n))),xs)",
            "entry main(xs:[]i32) []i32=map(|x|reduce((+),0,iota(x)),xs)",
        ],
        |index, graph, region| {
            let query = Query(graph);
            let maps = operations_in(graph, "map", region);
            assert_eq!(maps.len(), 1);
            let plan = query.required("SelectedPlan", (maps[0],)).unwrap();
            assert!(query.flag("MapHost", (plan,)).unwrap());
            assert_eq!(
                query.flag("MapSegmented", (maps[0],)).unwrap(),
                index < 2,
                "case {index}"
            );
        },
    );
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
        assert_eq!(query.contains("SubgroupAddition", (scan,)).unwrap(), enabled);
        let mut segments = 0;
        query
            .for_each("MapPlan", |row| {
                if query.flag("MapHost", (row[0],))? && query.flag("MapSegmented", (row[1],))? {
                    let width = query.required("SegmentWidth", (row[1],))?;
                    assert_eq!(
                        program.graph.value_to_base::<i64>(width),
                        if enabled { 32 } else { 256 }
                    );
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
    let cases = [(8, false), (8, true), (7, false), (7, true)];
    let sources: Vec<_> = cases.iter().enumerate().map(|(index, (count, reduced))| {
        let metadata = if *reduced { "reduce((+),0,map(|r|r[2],rows))" } else { "map(|r|r[2],rows)" };
        let metadata_type = if *reduced { "i32".to_string() } else { format!("[{count}]i32") };
        format!("def row_{index}(i:i32) [3]i32 = [i,i+1,i+2]
            entry main() ([16]i32,{metadata_type}) =
            let rows = map(row_{index},iota({count})) in
            (map(|a|let i=(a/8)*4+a%4 in if i<length(rows) then rows[i][(a%8)/4] else -1,iota(16)), {metadata})")
    }).collect();
    planned_cases(&sources, |index, graph, region| {
        let query = Query(graph);
        let mut destinations = 0;
        query
            .for_function("TiledOutputs", |keys, _| {
                let owner = query.record("SourceOperation", "OperationInfo", (keys[0],))?.unwrap()[0];
                destinations += usize::from(owner == region);
                Ok(())
            })
            .unwrap();
        assert_eq!(destinations, usize::from(cases[index].0 == 8), "case {index}");
    });
}

#[test]
fn primitive_dispatch_recipes_publish_their_phase_widths_and_order() {
    let mut graph = super::super::fusion::new_graph().unwrap();
    super::load(&mut graph).unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (set (Plan (OperationId 0)) (Elements))
        (set (ElideElements (OperationId 0)) false)
        (set (MapSegmented (OperationId 0)) false)
        (set (MapExecution (OperationId 0)) (InvocationScope))
        (set (IterationExtent (OperationId 0)) (Fixed 1024))
        (set (OperationGrid (OperationId 0)) (AutomaticGrid))
        (set (Plan (OperationId 1)) (Totals))
        (set (Plan (OperationId 2)) (Prefixes))
        (set (CollectiveChunkWidth (OperationId 1)) 256)
        (set (CollectiveChunkWidth (OperationId 2)) 256)
        (set (OperationGrid (OperationId 1)) (AutomaticGrid))
        (set (OperationGrid (OperationId 2)) (AutomaticGrid))
        (set (InputDomain (OperationId 2)) (Fixed 1024))
        (FilterShape (OperationId 3))
        (set (ParallelBodySafe (OperationId 3)) true)
        (set (OperationGrid (OperationId 3)) (FixedGrid 2 3 4))
        (IndexedReducer (OperationId 4) true (AtomicAdd))
        (set (ParallelBodySafe (OperationId 4)) true)
        (set (InputDomain (OperationId 4)) (Fixed 5))
        (set (OperationGrid (OperationId 4)) (AutomaticGrid))
        (IndexedReducer (OperationId 5) false (AtomicAdd))
        (run-schedule (saturate schedule))
        (check (KernelPhase (OperationId 0) "elements" (Fixed 1024) 64 (AutomaticGrid)))
        (check (KernelPhase (OperationId 1) "chunks" (Fixed 65536) 256 (FixedGrid 256 1 1)))
        (check (KernelPhase (OperationId 1) "combine" (Fixed 256) 256 (FixedGrid 1 1 1)))
        (check (PhaseLink (OperationId 1) "chunks" "combine"))
        (check (KernelPhase (OperationId 2) "offsets" (Fixed 1024) 64 (AutomaticGrid)))
        (check (PhaseLink (OperationId 2) "combine" "offsets"))
        (check (= (Plan (OperationId 3)) (Compact)))
        (check (KernelPhase (OperationId 3) "compact" (Fixed 64) 64 (FixedGrid 1 1 1)))
        (check (= (Plan (OperationId 4)) (Atomic (AtomicAdd))))
        (check (KernelPhase (OperationId 4) "atomic" (Fixed 5) 64 (AutomaticGrid)))
        (check (= (Plan (OperationId 5)) (Serial)))
        (check (KernelPhase (OperationId 5) "ordered" (Fixed 1) 1 (FixedGrid 1 1 1)))
    "#,
        )
        .unwrap();
}
