use super::super::query::Query;
use super::super::{from_tlc, fuse, optimize, place, schedule, to_ssa};
use crate::{compile_thru_tlc, CodegenTarget, PipelineTopologyPolicy};
use egglog_engine::{
    sort::{VecContainer, S},
    EGraph, Value, Write,
};

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
    for source in [
        include_str!("../../../testfiles/regressions/array_loop.wyn"),
        include_str!("../../../testfiles/regressions/array_reduce.wyn"),
        include_str!("../../../testfiles/regressions/matrix_map.wyn"),
        include_str!("../../../testfiles/regressions/matrix_reduce_loop.wyn"),
    ] {
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
}

#[test]
fn compiler_repro_array_reduce_preserves_effect_barrier() {
    for source in [
        include_str!("../../../testfiles/regressions/array_reduce.wyn"),
        include_str!("../../../testfiles/regressions/matrix_reduce_loop.wyn"),
    ] {
        let tlc = compile_thru_tlc(source).unwrap();
        for writer in [None, Some("producer"), Some("consumer")] {
            let mut program = from_tlc(&tlc).unwrap();
            let query = Query(&program.graph);
            let reducer = operations(&program.graph, "reduce")[0];
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
                program
                    .graph
                    .update(|mut sink| {
                        sink.set("SourceReadOnly", (written,), false)?;
                        sink.set("SourceSpeculatable", (written,), false)
                    })
                    .unwrap();
            }
            let program = fuse(program).unwrap();
            let query = Query(&program.graph);
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
    for (source, expected) in [
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
    ] {
        let tlc = compile_thru_tlc(source).unwrap();
        let program = fuse(from_tlc(&tlc).unwrap()).unwrap();
        let query = Query(&program.graph);
        let mut counts = Vec::new();
        query
            .for_each("Current", |row| {
                let plan = row[1];
                if query.contains("ScremaPlan", (plan,))? || query.contains("FilterPlan", (plan,))? {
                    counts.push((
                        program.graph.value_to_base::<i64>(query.required("ArrayOutputs", (plan,))?),
                        program.graph.value_to_base::<bool>(query.required("HasScans", (plan,))?),
                        program.graph.value_to_base::<bool>(query.required("HasReductions", (plan,))?),
                    ));
                }
                Ok(())
            })
            .unwrap();
        counts.sort();
        assert_eq!(counts, expected, "{source}");
    }
}

#[test]
fn indexed_phase_domains_preserve_validation() {
    let tlc = compile_thru_tlc("entry main(xs:[]i32) []i32=map(|x|x+1,xs)").unwrap();
    for (mode, expected) in [
        ("missing", "stage domain missing"),
        ("ambiguous", "ambiguous PhaseDomain"),
        ("zero-width", "stage width must be positive"),
    ] {
        let mut program = optimize(
            schedule(
                place(
                    fuse(from_tlc(&tlc).unwrap()).unwrap(),
                    PipelineTopologyPolicy::AllowGenerated,
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
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
        let Err(error) = to_ssa(program, CodegenTarget::Wgsl) else {
            panic!("invalid {mode} phase domain was accepted");
        };
        assert!(error.to_string().contains(expected), "{mode}: {error}");
    }
}

#[test]
fn collective_emission_requires_the_planned_group_count() {
    let tlc = compile_thru_tlc("entry main(xs:[]i32) i32=reduce((+),0,xs)").unwrap();
    let mut program = optimize(
        schedule(
            place(
                fuse(from_tlc(&tlc).unwrap()).unwrap(),
                PipelineTopologyPolicy::AllowGenerated,
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    program.graph.clear_function("CollectiveGroups").unwrap();
    let Err(error) = to_ssa(program, CodegenTarget::Wgsl) else {
        panic!("emission must not reconstruct a missing group count from the grid");
    };
    assert!(
        error.to_string().contains("collective group count missing"),
        "{error}"
    );
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
    let ssa = to_ssa(optimize(schedule(program).unwrap()).unwrap(), CodegenTarget::Wgsl).unwrap();
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
    for (source, kind, rank, count) in [
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
    ] {
        planned(source, |graph| {
            let query = Query(graph);
            let [operation] = operations(graph, kind)[..] else {
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
    for source in [
        "entry main(xs:[2]i32) [2][3]i32=map(|x|map(|y|x+y,iota(3)),xs)",
        "entry main(dest:*[2][2]i32) ([2][2]i32,[2]u32,u32)=
         bucket_scatter_2d(dest,[[(0,1),(0,2)],[(1,3),(1,4)]])",
    ] {
        let tlc = compile_thru_tlc(source).unwrap();
        let planned = || {
            optimize(
                schedule(
                    place(
                        fuse(from_tlc(&tlc).unwrap()).unwrap(),
                        PipelineTopologyPolicy::AllowGenerated,
                    )
                    .unwrap(),
                )
                .unwrap(),
            )
            .unwrap()
        };
        let mut selected_only = planned();
        for table in ["SourceDomainAxis", "SourceDomainRank", "PlanDomain"] {
            selected_only.graph.clear_function(table).unwrap();
        }
        to_ssa(selected_only, CodegenTarget::Wgsl).unwrap();
        let mut program = planned();
        program.graph.clear_function("IterationAxis").unwrap();
        let Err(error) = to_ssa(program, CodegenTarget::Wgsl) else {
            panic!("a missing dimension was silently accepted");
        };
        assert!(
            error.to_string().contains("domain axis has no planned extent"),
            "{error}"
        );
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
    let tlc = compile_thru_tlc(source).unwrap();
    let program = optimize(
        schedule(
            place(
                fuse(from_tlc(&tlc).unwrap()).unwrap(),
                PipelineTopologyPolicy::AllowGenerated,
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    to_ssa(program, CodegenTarget::Wgsl).unwrap();
}

#[test]
fn rectangular_planning_and_emission_use_fused_structure() {
    let tlc = compile_thru_tlc(
        "entry main(xs:[4]i32) [4][3]i32=
         map(|x|map(|y|x+y,map(|y|y+1,iota(3))),map(|x|x*2,xs))",
    )
    .unwrap();
    let program = place(
        fuse(from_tlc(&tlc).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    let mut graph = program.graph.clone();
    // Remove source operation classification; immutable inputs and callbacks remain
    // shared by source analysis and the fused plan.
    graph.clear_function("SourceOperation").unwrap();
    super::schedule(&mut graph).unwrap();
    let query = Query(&graph);
    let mut ranks = Vec::new();
    query
        .for_function("IterationRank", |_, rank| {
            ranks.push(graph.value_to_base::<i64>(rank));
            Ok(())
        })
        .unwrap();
    assert!(ranks.contains(&2));

    let mut program = optimize(schedule(program).unwrap()).unwrap();
    program.graph.clear_function("SourceOperation").unwrap();
    to_ssa(program, CodegenTarget::Wgsl).unwrap();
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
    for (source, expected_rank) in [
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
    ] {
        planned(source, |graph| {
            let query = Query(graph);
            let mut ranks = Vec::new();
            query.for_function("IterationRank", |keys, rank| {
                let region = query.required("OperationRegion", (keys[0],))?;
                let host = query.contains("HostRegion", (region,))?;
                if host { ranks.push(graph.value_to_base::<i64>(rank)); }
                Ok(())
            }).unwrap();
            assert_eq!(ranks, vec![expected_rank], "{source}");
        });
    }
}

#[test]
fn fused_outputs_only_share_a_rectangular_launch_when_inner_shapes_agree() {
    for (inner, expected_rank) in [(3, 2), (5, 1)] {
        let source = format!(
            "entry main(xs:[4]i32) ([4][3]i32,[4][{inner}]i32)=
             (map(|x|map(|y|x+y,iota(3)),xs),
              map(|x|map(|y|x*y,iota({inner})),xs))"
        );
        planned(&source, |graph| {
            let query = Query(graph);
            let mut fused = 0;
            query
                .for_each("MapPlan", |row| {
                    if !query.flag("MapHost", (row[0],))? {
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
            assert_eq!(fused, 1, "{source}");
        });
    }
}

#[test]
fn rectangular_runtime_dimensions_must_be_uniform_across_outer_coordinates() {
    for (source, expected_rank) in [
        (
            "entry main(xs:[]i32,n:i32) [][]i32=map(|x|map(|y|x+y,iota(n)),xs)",
            2,
        ),
        (
            "entry main(xs:[]i32,ys:[]i32) [][]i32=map(|x|map(|y|x+y,ys),xs)",
            2,
        ),
        (
            "entry main(xs:[]i32,n:i32,m:i32) [][][]i32=map(|x|map(|y|map(|z|x+y+z,iota(m)),iota(n)),xs)",
            3,
        ),
        ("entry main(xs:[]i32) [][]i32=map(|x|map(|y|x+y,iota(x)),xs)", 1),
    ] {
        planned(source, |graph| {
            let query = Query(graph);
            let mut ranks = Vec::new();
            query
                .for_each("MapPlan", |row| {
                    if query.flag("MapHost", (row[0],))? {
                        ranks.push(graph.value_to_base::<i64>(query.required("IterationRank", (row[1],))?));
                    }
                    Ok(())
                })
                .unwrap();
            assert_eq!(ranks, vec![expected_rank], "{source}");
        });
    }
}

#[test]
fn segment_callback_blockers_cover_every_output_of_a_fused_plan() {
    let tlc =
        compile_thru_tlc("entry main(xs:[17]i32) (i32,i32)=(reduce((+),0,xs),reduce((*),1,xs))").unwrap();
    for blocked in [false, true] {
        let mut program = place(
            fuse(from_tlc(&tlc).unwrap()).unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap();
        let reducers = operations(&program.graph, "reduce");
        assert_eq!(reducers.len(), 2);
        let query = Query(&program.graph);
        let plan = query.required("SelectedPlan", (reducers[0],)).unwrap();
        assert_eq!(plan, query.required("SelectedPlan", (reducers[1],)).unwrap());
        let callback = query.required("SourceOperatorBody", (reducers[0],)).unwrap();
        let values: Vec<_> = reducers
            .iter()
            .map(|&reducer| query.required("SourceOperationValue", (reducer,)).unwrap())
            .collect();
        program
            .graph
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
        let program = schedule(program).unwrap();
        let query = Query(&program.graph);
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
    for source in [
        "entry main(xs:[][513]i32) []i32=map(|row|reduce((+),0,row),xs)",
        "entry main(xs:[][513]i32) [][513]i32=map(|row|scan((+),0,row),xs)",
        "entry main(xs:[][3][17]i32) [][3]i32=map(|rows|map(|row|reduce((+),0,row),rows),xs)",
    ] {
        planned(source, |graph| {
            let query = Query(graph);
            let mut groups = 0;
            query
                .for_function("MapCooperative", |_, enabled| {
                    groups += usize::from(graph.value_to_base::<bool>(enabled));
                    Ok(())
                })
                .unwrap();
            assert_eq!(groups, 1, "{source}");
        });
    }
}

#[test]
fn segmented_collectives_do_not_cross_callback_control_flow() {
    for source in [
        "entry main(xs:[][17]i32) []i32=map(|row|if row[0]>0 then reduce((+),0,row) else 0,xs)",
        "entry main(xs:[][17]i32) []i32=map(|row|loop a=0 for i<2 do a+reduce((+),0,row),xs)",
        "entry main(xs:[]i32) []i32=map(|n|reduce((+),0,iota(n)),xs)",
    ] {
        planned(source, |graph| {
            Query(graph)
                .for_function("MapCooperative", |_, enabled| {
                    assert!(!graph.value_to_base::<bool>(enabled), "{source}");
                    Ok(())
                })
                .unwrap();
        });
    }
}

#[test]
fn selected_map_bindings_replace_callback_discovery() {
    for (source, rank, output_rank) in [
        (
            "def row(x:i32) [3]i32=map(|y|x+y,iota(3))
          entry main(xs:[]i32) [][3]i32=map(row,xs)",
            2,
            2,
        ),
        (
            "entry main(xs:[][3][17]i32) [][3]i32=
          map(|rows|map(|row|reduce((+),0,map(|x|x+row[0],row)),rows),xs)",
            2,
            2,
        ),
        (
            "entry main(xs:[][3][17]i32) [][3][17]i32=
          map(|rows|map(|row|map(|x|x*2,scan((+),0,map(|x|x+1,row))),rows),xs)",
            2,
            3,
        ),
    ] {
        let tlc = compile_thru_tlc(source).unwrap();
        let mut program = optimize(
            schedule(
                place(
                    fuse(from_tlc(&tlc).unwrap()).unwrap(),
                    PipelineTopologyPolicy::AllowGenerated,
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let query = Query(&program.graph);
        let mut outputs = 0;
        query
            .for_each("MapPlan", |row| {
                if !query.flag("MapHost", (row[0],))? {
                    return Ok(());
                }
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
                    axes.data
                        .iter()
                        .map(|&axis| program.graph.value_to_base::<i64>(axis))
                        .collect::<Vec<_>>(),
                    (0..output_rank).collect::<Vec<_>>(),
                    "{source}"
                );
                outputs += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(outputs, 1, "{source}");
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
        to_ssa(program, CodegenTarget::Wgsl).unwrap();
    }
}

#[test]
fn runtime_segmented_collectives_require_uniform_dimensions() {
    for (source, cooperative) in [
        (
            "entry main(xs:[]i32,n:i32) []i32=map(|x|reduce((+),0,map(|y|x+y,iota(n))),xs)",
            true,
        ),
        (
            "entry main(xs:[]i32,n:i32) [][]i32=map(|x|scan((+),0,map(|y|x+y,iota(n))),xs)",
            true,
        ),
        (
            "entry main(xs:[]i32) []i32=map(|x|reduce((+),0,iota(x)),xs)",
            false,
        ),
    ] {
        planned(source, |graph| {
            let query = Query(graph);
            query
                .for_each("MapPlan", |row| {
                    if query.flag("MapHost", (row[0],))? {
                        let actual = query.flag("MapCooperative", (row[1],))?;
                        assert_eq!(actual, cooperative, "{source}");
                    }
                    Ok(())
                })
                .unwrap();
        });
    }
}
