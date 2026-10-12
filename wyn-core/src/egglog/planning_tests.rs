use super::super::query::Query;
use super::super::{from_tlc, fuse, optimize, place, schedule, to_ssa};
use crate::{compile_thru_tlc, CodegenTarget, PipelineTopologyPolicy};
use egglog_engine::{sort::S, EGraph, Value};

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
        .for_each("SourceOperation", |row| {
            if graph.value_to_base::<S>(row[2]).as_str() == kind {
                operations.push(row[0]);
            }
            Ok(())
        })
        .unwrap();
    operations
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
                graph.value_to_base::<i64>(query.required("LocalCapacity", (extent,)).unwrap()),
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
            assert!(query.contains("InputDomain", (owner, extent)).unwrap());
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
                .for_each("SourceCapture", |row| {
                    if bodies.contains(&row[0]) {
                        if let Some(origin) = query.enode("SourceFormal", row[2])? {
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
fn selected_phases_declare_their_cooperation_scope() {
    for source in [
        "entry main(xs:[]i32) [][5]i32=map(|x|scan(|a,b|a+b,0,[x,x,x,x,x]),xs)",
        "entry main(xs:[]i32) []i32=scan(|a,b|a+b,0,xs)",
        "entry main(xs:[]i32) []i32=filter(|x|x>0,xs)",
    ] {
        planned(source, |graph| {
            let query = Query(graph);
            query
                .for_each("Phase", |row| {
                    let stage = query.enode("Stage", row[0])?.unwrap();
                    let phase = graph.value_to_base::<S>(stage[1]);
                    let expected = if matches!(phase.as_ref(), "chunks" | "combine" | "compact") {
                        "WorkgroupScope"
                    } else {
                        "InvocationScope"
                    };
                    let execution = query.required("PhaseExecution", (row[0],))?;
                    assert!(query.enode(expected, execution)?.is_some(), "{phase:?}");
                    Ok(())
                })
                .unwrap();
        });
    }
}

#[test]
fn missing_ranked_axis_is_rejected_before_kernel_emission() {
    let tlc = compile_thru_tlc(
        "entry main(dest:*[2][2]i32) ([2][2]i32,[2]u32,u32)=
         bucket_scatter_2d(dest,[[(0,1),(0,2)],[(1,3),(1,4)]])",
    )
    .unwrap();
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
    program.graph.clear_function("SourceDomainAxis").unwrap();
    let Err(error) = to_ssa(program, CodegenTarget::Wgsl) else {
        panic!("a missing dimension was silently accepted");
    };
    assert!(
        error.to_string().contains("domain axis has no planned extent"),
        "{error}"
    );
}

#[test]
fn rectangular_maps_expand_launch_without_expanding_row_capacity() {
    let source = "entry main(xs:[4]i32) [4][3]i32=map(|x|map(|y|x+y,iota(3)),xs)";
    planned(source, |graph| {
        let query = Query(graph);
        let mut flattened = 0;
        query
            .for_function("MapLaunchRank", |keys, rank| {
                if graph.value_to_base::<i64>(rank) == 2 {
                    flattened += 1;
                    let launch = query.required("MapLaunch", (keys[0],))?;
                    assert_eq!(
                        graph.value_to_base::<i64>(query.required("LocalCapacity", (launch,))?),
                        12
                    );
                    let outer = query.required("MapLaunchAxis", (keys[0], 0i64))?;
                    assert!(query.contains("InputDomain", (keys[0], outer))?);
                    assert_eq!(
                        graph.value_to_base::<i64>(query.required("LocalCapacity", (outer,))?),
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
    // Leave scalar bodies and identities intact, but remove original SOAC structure.
    for table in ["SourceOperation", "SourceInput", "SourceOperatorBody"] {
        graph.clear_function(table).unwrap();
    }
    super::schedule(&mut graph).unwrap();
    let query = Query(&graph);
    let mut ranks = Vec::new();
    query
        .for_function("MapLaunchRank", |_, rank| {
            ranks.push(graph.value_to_base::<i64>(rank));
            Ok(())
        })
        .unwrap();
    assert!(ranks.contains(&2));

    let mut program = optimize(schedule(program).unwrap()).unwrap();
    for table in ["SourceOperation", "SourceInput", "SourceOperatorBody"] {
        program.graph.clear_function(table).unwrap();
    }
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
            .for_function("WorkCost", |_, cost| {
                expensive |= graph.value_to_base::<i64>(cost) >= 65;
                Ok(())
            })
            .unwrap();
        assert!(expensive, "fixture must exceed the old repetition budget");
        let mut flattened = false;
        query
            .for_function("MapLaunchRank", |_, rank| {
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
            query.for_function("MapLaunchRank", |keys, rank| {
                let mut host = false;
                query.for_each("Site", |row| {
                    if row[0] == keys[0] && query.contains("HostRegion", (row[1],))? {
                        host = true;
                    }
                    Ok(())
                })?;
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
                        let rank = query.required("MapLaunchRank", (row[1],))?;
                        assert_eq!(graph.value_to_base::<i64>(rank), expected_rank);
                    }
                    Ok(())
                })
                .unwrap();
            assert_eq!(fused, 1, "{source}");
        });
    }
}
