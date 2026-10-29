use crate::egglog::{finish_for_test, from_tlc, fuse, place, schedule};
use crate::host::Pipeline;
use crate::PipelineTopologyPolicy;
use crate::{compile_thru_tlc, host, CodegenTarget};

#[test]
fn fused_map_chain_keeps_permission_to_reuse_its_external_input() {
    let output = pipeline("entry main(xs:*[]i32) []i32 = let ys=map(|x:i32|x+1,xs) in map(|x:i32|x*2,ys)");
    let Pipeline::Compute(p) = &output.pipelines[0] else {
        panic!("compute")
    };
    assert_eq!(p.stages.len(), 1);
    assert_eq!(p.bindings.len(), 2, "reused array and explicit input length");
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
        "input/result array, scalar result, and input length"
    );
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
    assert_eq!(p.bindings.len(), 2, "reused array and explicit input length");
}

#[test]
fn independent_writers_cannot_both_reuse_one_input() {
    let output = pipeline("entry main(xs:*[]i32) ([]i32,[]i32) = (map(|x:i32|x+1,xs),map(|x:i32|x*2,xs))");
    let results = &output.source_results;
    assert_eq!(results.len(), 2);
    assert_ne!(results[0].binding, results[1].binding);
}

#[test]
fn selected_grid_dimensions_are_positive() {
    use egglog_engine::Write;
    let tlc = compile_thru_tlc("").unwrap();
    let imported = from_tlc(&tlc).unwrap();
    let mut program = imported.advance(crate::egglog::Optimized {
        captures: Default::default(),
        host: Default::default(),
        selected: crate::egglog::scalar::Selected {
            dag: Default::default(),
            values: vec![],
            roots: Default::default(),
            extents: Default::default(),
        },
        scalars: crate::egglog::fusion::new_graph().unwrap(),
    });
    for (dims, expected) in [
        ((0i64, 1i64, 1i64), "grid x"),
        ((1, 0, 1), "grid y"),
        ((1, 1, 0), "grid z"),
    ] {
        let grid = program.graph.update(|mut sink| sink.add("FixedGrid", dims)).unwrap();
        let error = crate::egglog::facts::Facts { program: &program }.grid(grid).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
    let grid = program.graph.update(|mut sink| sink.add("FixedGrid", (2i64, 3i64, 4i64))).unwrap();
    assert_eq!(
        crate::egglog::facts::Facts { program: &program }.grid(grid).unwrap(),
        (2, 3, 4)
    );
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
        crate::egglog::prepare_interfaces(
            schedule(
                place(
                    fuse(from_tlc(&tlc).unwrap()).unwrap(),
                    PipelineTopologyPolicy::AllowGenerated,
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap(),
        crate::egglog::ScalarOptimization::Basic,
    )
    .unwrap();
    let ssa = finish_for_test(program, CodegenTarget::Spirv).unwrap();
    let external = ssa.functions.iter().find(|f| f.linkage_name.as_deref() == Some("foreign_add")).unwrap();
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
    compile_thru_tlc(
        "entry main(xs: []i32) [4]i32 =
        let a=map(|x:i32|x+1,xs) in map(|x:i32|x*2,a[2..6])",
    )
    .unwrap();
}

fn pipeline(source: &str) -> host::ModuleInterface {
    pipeline_with_policy(source, crate::egglog::ScalarOptimization::Basic)
}

fn pipeline_with_policy(source: &str, policy: crate::egglog::ScalarOptimization) -> host::ModuleInterface {
    let tlc = compile_thru_tlc(source).unwrap();
    let program = schedule(
        place(
            fuse(from_tlc(&tlc).unwrap()).unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap(),
    )
    .unwrap();
    let program = crate::egglog::prepare_interfaces(program).unwrap();
    let program = crate::egglog::optimize_with_policy(program, policy).unwrap();
    finish_for_test(program, CodegenTarget::Wgsl).unwrap().global_context.pipeline
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
    assert_ne!(output.source_results[0].binding, output.source_results[1].binding);
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
fn fused_maps_publish_reused_nonprimary_inputs() {
    let output = pipeline(
        "entry main(xs:*[4]i32,ys:*[4]i32) ([4]i32,[4]i32) = (map(|x:i32|x+1,ys),map(|x:i32|x*2,xs))",
    );
    assert_eq!(
        output.source_results.iter().map(|r| r.binding).collect::<Vec<_>>(),
        [1, 0]
    );
    let Pipeline::Compute(p) = &output.pipelines[0] else {
        panic!("compute")
    };
    assert_eq!(p.bindings.len(), 2);
    assert!(p.stages.iter().all(|stage| stage.workgroup_size == (64, 1, 1)));
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

#[test]
fn runtime_collective_scratch_matches_its_bounded_chunk_grid() {
    use host::{Binding, BufferLen, DispatchSize};
    let output = pipeline("entry main(xs: []i32) []i32 = scan(|a:i32,b:i32|a+b,0,xs)");
    let [Pipeline::Compute(p)] = output.pipelines.as_slice() else {
        panic!("compute")
    };
    let DispatchSize::Fixed { x, y, z, .. } = p.stages[0].dispatch_size else {
        panic!("collective chunk grid must be bounded");
    };
    let bytes = u64::from(x) * u64::from(y) * u64::from(z) * 4;
    assert_eq!(
        p.bindings
            .iter()
            .filter(|binding| matches!(binding,
                Binding::StorageBuffer { length: Some(BufferLen::Fixed { bytes: capacity }), .. }
                    if *capacity == bytes
            ))
            .count(),
        2,
        "one partial and one carry per chunk"
    );
}

#[test]
fn bounded_filter_output_does_not_use_its_packed_backing_as_a_length() {
    use host::{Binding, BufferLen};
    let output = pipeline(include_str!("../../../testfiles/filter_then_map.wyn"));
    let result = &output.source_results[0];
    let Pipeline::Compute(p) = &output.pipelines[result.pipeline_index] else {
        panic!("compute")
    };
    assert!(p.bindings.iter().any(|b| matches!(b, Binding::StorageBuffer {
        set, binding, length: Some(BufferLen::Fixed { bytes: 16384 }), ..
    } if (*set, *binding) == (result.set, result.binding))));
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
    let output = pipeline_with_policy(
        "entry main(xs:[]i32,unused:[]i32,bias:i32) i32=reduce((+),0,map(|x|x+bias*bias,xs))",
        crate::egglog::ScalarOptimization::Full,
    );
    let [Pipeline::Compute(p)] = output.pipelines.as_slice() else {
        panic!("compute pipeline");
    };
    let unused = p
        .bindings
        .iter()
        .position(
            |binding| matches!(binding, host::Binding::StorageBuffer { name, .. } if name == "unused"),
        )
        .unwrap();
    assert!(p.stages.iter().all(|stage| stage.uses.access(unused).is_none()));
    let combine = p.stages.iter().find(|stage| stage.entry_point.ends_with("combine")).unwrap();
    // Combining needs the live input length, but not the producer's bias computation.
    assert!(output
        .scalar_tasks
        .iter()
        .filter(|task| task.stage == combine.entry_point)
        .all(|task| matches!(task.value, host::ScalarExpr::BufferLength { .. })));
    assert!(
        output.scalar_tasks.iter().any(|task| task.stage != combine.entry_point
            && !matches!(task.value, host::ScalarExpr::BufferLength { .. }))
    );
}
