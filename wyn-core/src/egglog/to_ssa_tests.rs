use crate::egglog::{finish_for_test, from_tlc, fuse, place, schedule};
use crate::host::Pipeline;
use crate::PipelineTopologyPolicy;
use crate::{compile_thru_tlc, host, CodegenTarget};

#[test]
fn logical_array_bindings_preserve_physical_boolean_storage() {
    use crate::types::{self, Type, TypeName};
    let logical = types::sized_array(4, types::bool_type());
    let storage = types::view_array_with_size(
        &Type::Constructed(TypeName::UInt(32), vec![]),
        Type::Constructed(TypeName::SizePlaceholder, vec![]),
        types::buffer_tag(crate::BindingRef::new(0, 2)),
    );
    assert!(super::same_value_structure(&storage, &logical));
    assert!(super::same_value_structure(
        &types::tuple(vec![storage.clone()]),
        &types::tuple(vec![logical]),
    ));
    assert!(!super::same_value_structure(
        &storage,
        &types::sized_array(4, types::i32())
    ));
    assert!(!super::same_value_structure(
        &Type::Constructed(TypeName::UInt(32), vec![]),
        &types::bool_type(),
    ));
}

#[test]
fn fused_map_chain_keeps_permission_to_reuse_its_external_input() {
    let output = pipeline("entry main(xs:*[]i32) []i32 = let ys=map(|x:i32|x+1,xs) in map(|x:i32|x*2,ys)");
    let Pipeline::Compute(p) = &output.pipelines[0] else {
        panic!("compute")
    };
    assert_eq!(p.stages.len(), 1);
    assert_eq!(p.bindings.len(), 2, "the input length has its own binding");
    let host::ResultShape::Rectangular(dimensions) = &output.source_results[0].shape else {
        panic!("rectangular result");
    };
    assert!(matches!(
        dimensions.as_slice(),
        [host::Expr::Scalar(host::ScalarExpr::BufferLength { .. })]
    ));
    assert_eq!(p.stages[0].uses.access(0), Some(host::Access::ReadWrite));
    assert_eq!(output.source_results[0].binding, 0);
}

#[test]
fn length_observers_do_not_block_content_reuse() {
    let output =
        pipeline("entry main(xs:*[]i32) ([]i32,i32) = let ys=map(|x:i32|x+1,xs) in (ys,length(xs))");
    assert_eq!(output.source_results[0].binding, 0);
    let Pipeline::Compute(p) = &output.pipelines[0] else {
        panic!("compute")
    };
    assert_eq!(
        p.bindings.len(),
        3,
        "shared input/result array, its length, and the published scalar result"
    );
    assert_ne!(output.source_results[0].binding, output.source_results[1].binding);
}

#[test]
fn multi_input_map_can_reuse_its_nonprimary_unique_input() {
    let output = pipeline("entry main(xs:[4]i32,ys:*[4]i32) [4]i32 = map(|(x,y)|x+y,zip(xs,ys))");
    let Pipeline::Compute(p) = &output.pipelines[0] else {
        panic!("compute")
    };
    let result = &output.source_results[0];
    assert!(p.bindings.iter().any(|b| matches!(
        b,
        host::Binding::StorageBuffer { name, set, binding, .. }
            if name == "ys" && (*set, *binding) == (result.set, result.binding)
    )));
    assert_eq!(p.bindings.len(), 2);
}

#[test]
fn fused_intermediate_element_types_do_not_restrict_final_reuse() {
    let output = pipeline(
        "entry main(xs:*[]i32) []i32 = let bs=map(|x:i32|x>0,xs) in map(|b:bool|if b then 1 else 0,bs)",
    );
    assert_eq!(output.source_results[0].binding, 0);
    let Pipeline::Compute(p) = &output.pipelines[0] else {
        panic!("compute")
    };
    assert_eq!(p.stages.len(), 1);
    assert_eq!(p.bindings.len(), 2, "the input length has its own binding");
    let host::ResultShape::Rectangular(dimensions) = &output.source_results[0].shape else {
        panic!("rectangular result");
    };
    assert!(matches!(
        dimensions.as_slice(),
        [host::Expr::Scalar(host::ScalarExpr::BufferLength { .. })]
    ));
}

#[test]
fn independent_writers_cannot_both_reuse_one_input() {
    let output = pipeline("entry main(xs:*[]i32) ([]i32,[]i32) = (map(|x:i32|x+1,xs),map(|x:i32|x*2,xs))");
    let results = &output.source_results;
    assert_eq!(results.len(), 2);
    assert_ne!(results[0].binding, results[1].binding);
}

#[test]
fn reuse_requires_ownership_pointwise_access_and_no_live_alias() {
    for source in [
        "entry main(xs:[]i32) []i32=map(|x|x+1,xs)",
        "entry main(xs:*[4]i32) [4]i32=map(|i|xs[3-i],iota(4))",
        "entry main(xs:*[4]i32) ([4]i32,[4]i32)=(xs,map(|x|x+1,xs))",
    ] {
        let output = pipeline(source);
        let result = output.source_results.last().unwrap();
        assert_ne!(result.binding, 0, "{source}");
        let Pipeline::Compute(pipeline) = &output.pipelines[0] else {
            panic!("compute pipeline");
        };
        assert!(
            pipeline.stages.iter().all(|stage| {
                !matches!(
                    stage.uses.access(0),
                    Some(host::Access::ReadWrite | host::Access::WriteOnly)
                )
            }),
            "{source}"
        );
    }
}

#[test]
fn selected_grid_dimensions_are_positive() {
    use crate::egglog::execution::Execution;
    for x in [0, 1, 32] {
        let mut graph=crate::egglog::test_graph::fixture(&[(0,crate::egglog::dataflow::ir::Code::Parallel)],&format!(r#"
            (let name (FunctionId 0 "grid"))
            (let grid (Grid3D (vec-of (IndexConstant 2) (IndexConstant 3) (IndexConstant 4))
                (vec-of (IndexConstant {x}) (IndexConstant 1) (IndexConstant 1))))
            (set (FixtureFunctionRoot name) (RegionId 0)) (set (EntryOrder name) 0)
            (set (FixtureRegion (OperationId 0)) (RegionId 0)) (set (FixtureOrder (OperationId 0)) 0)
            (set (OperationExecution (OperationId 0)) (DeviceKernel grid)) (set (OperationWidth (OperationId 0)) {x})
        "#)).unwrap();
        let result = Execution::select(&mut graph);
        if x == 0 {
            assert!(result.is_err());
        } else {
            let selected = result.unwrap();
            let statement =
                crate::egglog::query::Query(&graph).constructor("OperationId", 0_i64).unwrap().unwrap();
            assert_eq!(selected.dispatch(statement).unwrap().grid, Some((2, 3, 4)));
            assert_eq!(selected.dispatch(statement).unwrap().width.get(), x);
        }
    }
}

#[test]
fn fused_map_reduce_normalizes() {
    compile_thru_tlc(
        "entry main(xs: []f32) f32 = reduce(|a: f32, b: f32| a + b, 0.0, map(|x: f32| x * 2.0, xs))",
    )
    .unwrap();
}

#[test]
fn external_functions_retain_their_declared_signatures() {
    let tlc = compile_thru_tlc(
        "#[linked(\"foreign_add\")] extern add(a:i32,b:i32) i32\nentry main(x:i32) i32 = add(x, 3)",
    )
    .unwrap();
    let program = crate::egglog::optimize_with_policy(
        fuse(from_tlc(&tlc).unwrap()).unwrap(),
        crate::egglog::ScalarOptimization::Basic,
    )
    .unwrap();
    let program = place(program, PipelineTopologyPolicy::AllowGenerated).unwrap();
    let program = crate::egglog::prepare_interfaces(schedule(program).unwrap()).unwrap();
    let ssa = finish_for_test(program, CodegenTarget::Spirv).unwrap();
    let external = ssa.functions.iter().find(|f| f.linkage_name.as_deref() == Some("foreign_add")).unwrap();
    assert!(external.span.is_generated());
    assert_eq!(external.body.params().len(), 2);
    assert_eq!(external.body.return_ty, crate::types::i32());
    let host = crate::host::Program::new(ssa.global_context.pipeline).unwrap();
    assert!(host.entries[0]
        .operations
        .iter()
        .any(|op| matches!(op, crate::host::Operation::Dispatch { .. })));
}

#[test]
fn sliced_fused_map_normalizes() {
    pipeline(
        "entry main(xs: []i32) [4]i32 =
        let a=map(|x:i32|x+1,xs) in map(|x:i32|x*2,a[2..6])",
    );
}

fn pipeline(source: &str) -> host::ModuleInterface {
    pipeline_with_policy(source, crate::egglog::ScalarOptimization::Basic)
}

fn pipeline_with_policy(source: &str, policy: crate::egglog::ScalarOptimization) -> host::ModuleInterface {
    lowered_with_policy(source, policy).global_context.pipeline
}

fn lowered_with_policy(
    source: &str,
    policy: crate::egglog::ScalarOptimization,
) -> crate::ssa::stage::Elaborated {
    lowered_for_target(source, policy, CodegenTarget::Wgsl)
}

fn lowered_for_target(
    source: &str,
    policy: crate::egglog::ScalarOptimization,
    target: CodegenTarget,
) -> crate::ssa::stage::Elaborated {
    let tlc = compile_thru_tlc(source).unwrap();
    let program = schedule(
        place(
            crate::egglog::optimize_with_policy(fuse(from_tlc(&tlc).unwrap()).unwrap(), policy).unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap(),
    )
    .unwrap();
    let program = crate::egglog::prepare_interfaces(program).unwrap();
    finish_for_test(program, target).unwrap()
}

#[test]
fn input_length_uses_do_not_require_a_separate_dispatch() {
    let output = pipeline(
        "entry main(xs: []i32, index:i32) []i32 =
         let a=xs[index] in map(|i:i32|i+a, iota(length(xs)))",
    );
    let [Pipeline::Compute(p)] = output.pipelines.as_slice() else {
        panic!("compute pipeline")
    };
    assert_eq!(p.stages.len(), 1, "immutable read and length stay inside the map");
}

#[test]
fn scalar_projection_setup_is_evaluated_inside_its_map() {
    let output = pipeline(include_str!("../../../testfiles/scalar_setup.wyn"));
    let [Pipeline::Compute(p)] = output.pipelines.as_slice() else {
        panic!("compute pipeline")
    };
    assert_eq!(p.stages.len(), 1);
}

#[test]
fn reads_before_consuming_updates_remain_materialized() {
    let output = pipeline(
        "entry main(xs:*[]i32, index:i32) []i32 =
         let old=xs[index] in map(|x:i32|x+old, xs)",
    );
    let [Pipeline::Compute(p)] = output.pipelines.as_slice() else {
        panic!("compute pipeline")
    };
    assert_eq!(
        p.stages.len(),
        2,
        "save the old value before overwriting its buffer"
    );
}

#[test]
fn independent_entries_keep_their_own_dispatches_and_results() {
    let output = pipeline(
        "entry first(xs:[5]i32) [5]i32 = map(|x|x*3,map(|x:i32|x+1,xs))\n\
         entry second(ys:[9]i32) [9]i32 = map(|x|x-1,map(|x:i32|x*2,ys))",
    );
    assert_eq!(output.pipelines.len(), 2);
    assert_eq!(output.source_results.len(), 2);
    assert_ne!(
        output.source_results[0].pipeline_index,
        output.source_results[1].pipeline_index
    );
    assert_ne!(
        (
            output.source_results[0].pipeline_index,
            output.source_results[0].set,
            output.source_results[0].binding
        ),
        (
            output.source_results[1].pipeline_index,
            output.source_results[1].set,
            output.source_results[1].binding
        ),
    );
    for (pipeline, other_input) in output.pipelines.iter().zip(["ys", "xs"]) {
        let Pipeline::Compute(pipeline) = pipeline else {
            panic!("compute")
        };
        assert_eq!(
            pipeline.stages.len(),
            1,
            "each map chain should fuse independently"
        );
        assert!(
            !pipeline.bindings.iter().any(|binding| matches!(
                binding, host::Binding::StorageBuffer { name, .. } if name == other_input
            )),
            "a plan must not import another entry's operand"
        );
    }
}

#[test]
fn in_place_results_keep_the_host_input_binding_and_upload_role() {
    use host::{Access, Binding, BufferUsage, Pipeline};
    let output = pipeline("entry main(dest:*[3]i32,xs:[5]i32) [3]i32 = reduce_by_index(dest,|a:i32,b:i32|a+b,0,map(|x:i32|x%3,xs),xs)");
    let result = &output.source_results[0];
    assert_eq!((result.set, result.binding), (0, 0));
    let Pipeline::Compute(p) = &output.pipelines[0] else {
        panic!("compute pipeline")
    };
    assert!(p.bindings.iter().any(|b| matches!(
        b,
        Binding::StorageBuffer {
            set: 0,
            binding: 0,
            usage: BufferUsage::Input,
            access: Access::ReadWrite,
            ..
        }
    )));
}

#[test]
fn fused_outputs_reuse_distinct_consumed_inputs() {
    let output = pipeline(
        "entry main(xs:*[4]i32,ys:*[4]i32) ([4]i32,[4]i32) = (map(|x:i32|x+1,ys),map(|x:i32|x*2,xs))",
    );
    let mut bindings = output.source_results.iter().map(|result| result.binding).collect::<Vec<_>>();
    bindings.sort();
    assert_eq!(bindings, [0, 1], "each consumed buffer serves exactly one output");
    let Pipeline::Compute(p) = &output.pipelines[0] else {
        panic!("compute")
    };
    assert_eq!(p.bindings.len(), 2);
    assert_eq!(p.stages.len(), 1);
    assert_eq!(p.stages[0].uses.access(0), Some(host::Access::ReadWrite));
    assert_eq!(p.stages[0].uses.access(1), Some(host::Access::ReadWrite));
    assert!(p.stages.iter().all(|stage| {
        let (x, y, z) = stage.workgroup_size;
        x.is_power_of_two() && x <= 1024 && (y, z) == (1, 1)
    }));
}

#[test]
fn tuple_of_views_uses_the_tlc_component_bindings() {
    let source = "entry main(xs: ([]i32,[]i32)) ([]i32,[]i32) = xs";
    let expected = vec![0, 1];
    let output = pipeline(source);
    assert_eq!(output.source_results.len(), 2);
    assert_eq!(
        output.source_results.iter().map(|o| o.binding).collect::<Vec<_>>(),
        expected
    );
}

// Evaluate published sizing expressions at concrete input lengths; representation
// choices such as Fixed versus Computed must not change the allocation contract.
fn allocation_index(expression: &host::Expr, input_elements: i64) -> i64 {
    use host::Expr;
    match expression {
        Expr::Integer(n) => *n,
        Expr::Scalar(value) => scalar_index(value, input_elements),
        Expr::BufferLength { source, stride } => {
            assert_eq!(*source, host::ScalarSource::Binding { set: 0, binding: 0 });
            input_elements * 4 / i64::from(*stride)
        }
        Expr::Add(a, b) => allocation_index(a, input_elements) + allocation_index(b, input_elements),
        Expr::Subtract(a, b) => allocation_index(a, input_elements) - allocation_index(b, input_elements),
        Expr::Multiply(a, b) => allocation_index(a, input_elements) * allocation_index(b, input_elements),
        Expr::Floor(a, b) => {
            allocation_index(a, input_elements).div_euclid(allocation_index(b, input_elements))
        }
        Expr::Ceiling(a, b) => {
            let (a, b) = (
                allocation_index(a, input_elements),
                allocation_index(b, input_elements),
            );
            a.div_euclid(b) + i64::from(a.rem_euclid(b) != 0)
        }
        Expr::Min(a, b) => allocation_index(a, input_elements).min(allocation_index(b, input_elements)),
        Expr::Max(a, b) => allocation_index(a, input_elements).max(allocation_index(b, input_elements)),
        other => panic!("sizing unexpectedly requires device results: {other:?}"),
    }
}

fn scalar_index(value: &host::ScalarExpr, input_elements: i64) -> i64 {
    use host::ScalarExpr;
    match value {
        ScalarExpr::I32(n) => i64::from(*n),
        ScalarExpr::U32(n) => i64::from(*n),
        ScalarExpr::Bool(b) => i64::from(*b),
        ScalarExpr::BufferLength { source, stride } => allocation_index(
            &host::Expr::BufferLength {
                source: source.clone(),
                stride: *stride,
            },
            input_elements,
        ),
        ScalarExpr::If { condition, yes, no } => scalar_index(
            if scalar_index(condition, input_elements) != 0 { yes } else { no },
            input_elements,
        ),
        ScalarExpr::Apply { op, args, .. } => {
            let a = scalar_index(&args[0], input_elements);
            if op == "to-i32" {
                return a;
            }
            let b = scalar_index(&args[1], input_elements);
            match op.as_str() {
                "add" => a + b,
                "sub" => a - b,
                "mul" => a * b,
                "div" => a / b,
                "rem" => a % b,
                "lt" => i64::from(a < b),
                "gt" => i64::from(a > b),
                other => panic!("unexpected capacity operation {other}"),
            }
        }
        other => panic!("sizing unexpectedly reads a device scalar: {other:?}"),
    }
}

fn allocation_bytes(length: &host::BufferLen, input_elements: i64) -> i64 {
    match length {
        host::BufferLen::Fixed { bytes } => i64::try_from(*bytes).unwrap(),
        host::BufferLen::Computed { bytes } => allocation_index(bytes, input_elements),
        other => panic!("unexpected allocation policy {other:?}"),
    }
}

#[test]
fn runtime_collective_scratch_matches_its_bounded_chunk_grid() {
    use host::{Binding, DispatchSize};
    let output = pipeline("entry main(xs: []i32) []i32 = scan(|a:i32,b:i32|a+b,0,xs)");
    let [Pipeline::Compute(p)] = output.pipelines.as_slice() else {
        panic!("compute")
    };
    let stages = p.stages.iter().filter(|stage| stage.workgroup_size.0 > 1).collect::<Vec<_>>();
    assert_eq!(stages.len(), 3, "chunks, carry scan, and carry application");
    let scratch = [(0, 1), (1, 2)].map(|(producer, consumer)| {
        let buffers = p
            .bindings
            .iter()
            .enumerate()
            .filter_map(|(index, binding)| {
                let Binding::StorageBuffer {
                    length: Some(length),
                    usage: host::BufferUsage::Intermediate,
                    ..
                } = binding
                else {
                    return None;
                };
                let written = matches!(
                    stages[producer].uses.access(index),
                    Some(host::Access::WriteOnly | host::Access::ReadWrite)
                );
                let read = matches!(
                    stages[consumer].uses.access(index),
                    Some(host::Access::ReadOnly | host::Access::ReadWrite)
                );
                (written && read).then_some(length)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            buffers.len(),
            1,
            "one buffer carries each scalar collective dependency"
        );
        buffers[0]
    });
    for elements in [0, 1, 17, 513, 65536, 1000000] {
        let groups = match &stages[0].dispatch_size {
            DispatchSize::Fixed { x, y, z, .. } => i64::from(*x) * i64::from(*y) * i64::from(*z),
            DispatchSize::Computed { groups, .. } => {
                groups.iter().map(|g| allocation_index(g, elements)).product()
            }
        };
        assert!((1..=256).contains(&groups));
        for length in scratch {
            assert_eq!(
                allocation_bytes(length, elements),
                groups * 4,
                "one partial/carry per chunk for input {elements}"
            );
        }
    }
}

#[test]
fn bounded_filter_output_does_not_use_its_packed_backing_as_a_length() {
    let output = pipeline(include_str!("../../../testfiles/filter_then_map.wyn"));
    let result = &output.source_results[0];
    let Pipeline::Compute(p) = &output.pipelines[result.pipeline_index] else {
        panic!("compute")
    };
    let length = p
        .bindings
        .iter()
        .find_map(|binding| match binding {
            host::Binding::StorageBuffer {
                set, binding, length, ..
            } if (*set, *binding) == (result.set, result.binding) => length.as_ref(),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        allocation_bytes(length, 0),
        16384,
        "capacity must cover every retained input"
    );
    let host::ResultShape::Rectangular(dimensions) = &result.shape else {
        panic!("rectangular output")
    };
    assert!(
        !matches!(dimensions.as_slice(), [host::Expr::Scalar(host::ScalarExpr::BufferLength { source: host::ScalarSource::Binding { set, binding }, .. })]
        if (*set, *binding) == (result.set, result.binding)),
        "physical capacity is not the filtered length"
    );
}

#[test]
fn runtime_index_into_shared_array_helper_normalizes() {
    compile_thru_tlc(
        "def g(n: i32) []f32 = map(|i: i32| f32.i32(i),0i32..<n)\nentry e(j: i32) [1]f32 = [g(256)[j]]",
    )
    .unwrap();
}

#[test]
fn collective_phase_accesses_exclude_unused_inputs_and_producer_captures() {
    let module = lowered_with_policy(
        "entry main(xs:[]i32,unused:[]i32,bias:i32) i32=reduce((+),0,map(|x|x+bias*bias,xs))",
        crate::egglog::ScalarOptimization::Full,
    );
    let [Pipeline::Compute(p)] = module.global_context.pipeline.pipelines.as_slice() else {
        panic!("compute pipeline");
    };
    for (index, binding) in p.bindings.iter().enumerate() {
        if matches!(binding, host::Binding::StorageBuffer { name, .. } if name == "unused") {
            assert!(p.stages.iter().all(|stage| stage.uses.access(index).is_none()));
        }
    }
    let stages =
        p.stages
            .iter()
            .filter(|stage| {
                let entry =
                    module.entry_points.iter().find(|entry| entry.name == stage.entry_point).unwrap();
                entry.body.inner.insts.values().any(|instruction| {
                    matches!(instruction.data, crate::ssa::types::InstKind::ControlBarrier)
                })
            })
            .collect::<Vec<_>>();
    assert_eq!(stages.len(), 2, "partial reduction followed by combination");
    let producer = module.entry_points.iter().find(|entry| entry.name == stages[0].entry_point).unwrap();
    let bias = p
        .bindings
        .iter()
        .position(|binding| {
            let Some((set, slot)) = binding.slot() else {
                return false;
            };
            matches!(
                binding,
                host::Binding::StorageBuffer {
                    access: host::Access::ReadOnly,
                    usage: host::BufferUsage::Intermediate,
                    ..
                }
            ) && producer.storage_bindings.iter().any(|declaration| {
                declaration.binding == crate::BindingRef::new(set, slot)
                    && declaration.elem_ty == crate::types::i32()
            })
        })
        .unwrap();
    assert!(
        p.stages
            .iter()
            .any(|stage| stage.entry_point != stages[1].entry_point && stage.uses.access(bias).is_some()),
        "producer computation must retain its scalar capture"
    );
    assert!(
        stages[1].uses.access(bias).is_none(),
        "the combiner must not evaluate producer captures"
    );
}
