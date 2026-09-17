use crate::egglog::{from_tlc, fuse, insert_expressions, schedule, simplify_and_place, to_ssa};
use crate::interface::EntryParamBindingKind;
use crate::pipeline_descriptor::Pipeline;
use crate::tlc::infer_input_slice_bounds;
use crate::PipelineTopologyPolicy;
use crate::{
    compile_thru_tlc, lower_ssa_to_spirv, lower_ssa_to_wgsl, lower_ssa_to_wgsl_with_pipeline,
    pipeline_descriptor, CodegenTarget, LoweredWgsl,
};

fn compile(source: &str) -> naga::Module {
    let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    let program = schedule(
        simplify_and_place(insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    let ssa = to_ssa(&program, CodegenTarget::Wgsl).unwrap();
    let source = lower_ssa_to_wgsl(ssa).unwrap();
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}\n{source}", e.emit_to_string(&source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{e:?}\n{source}"));
    module
}

#[test]
fn fused_map_reduce_reaches_the_existing_wgsl_backend() {
    compile("entry main(xs: []f32) f32 = reduce(|a: f32, b: f32| a + b, 0.0, map(|x: f32| x * 2.0, xs))");
}

#[test]
fn fused_indexed_updates_reach_wgsl() {
    compile(
        "entry main(dest: *[3]i32, xs: [5]i32) [3]i32 =
        reduce_by_index(dest, |a:i32,b:i32|a+b, 0,
            map(|x:i32|x-1,xs), map(|x:i32|x*3,xs))",
    );
}

#[test]
fn filtered_reduction_and_shared_count_reach_wgsl() {
    compile(
        "entry main(xs: []i32) (i32,i32) =
        let kept=filter(|x:i32|x>0,xs) in
        (length(kept),reduce(|a:i32,b:i32|a+b,0,kept))",
    );
}

#[test]
fn mapped_filter_result_keeps_its_runtime_sized_view() {
    compile(include_str!("../../../testfiles/filter_then_map.wyn"));
}

#[test]
fn signed_and_unsigned_integer_power_reach_wgsl() {
    compile(include_str!("../../../testfiles/int_pow.wyn"));
}

#[test]
fn invocation_local_filter_keeps_a_bounded_array_length() {
    compile(include_str!("../../../testfiles/filter_demo.wyn"));
}

#[test]
fn computed_array_outputs_have_storage_and_a_writer() {
    use pipeline_descriptor::{Binding, BufferLen};
    for source in [
        include_str!("../../../testfiles/array_param_view_multi.wyn"),
        include_str!("../../../testfiles/array_param_view_slice.wyn"),
    ] {
        compile(source);
        let output = pipeline(source);
        let [result] = output.pipeline.source_results.as_slice() else {
            panic!("one array result");
        };
        let Pipeline::Compute(p) = &output.pipeline.pipelines[result.pipeline_index] else {
            panic!("compute result");
        };
        assert!(p.bindings.iter().any(|b| matches!(b, Binding::StorageBuffer {
            set, binding, length: Some(BufferLen::Fixed { bytes: 4 }), ..
        } if (*set, *binding) == (result.set, result.binding))));
    }
}

#[test]
fn sliced_fused_map_reaches_wgsl() {
    compile(
        "entry main(xs: []i32) [4]i32 =
        let a=map(|x:i32|x+1,xs) in map(|x:i32|x*2,a[2..6])",
    );
}

#[test]
fn scalar_loop_reaches_the_existing_wgsl_backend() {
    compile("entry main(n: i32) i32 = loop acc = 0 for i < n do acc + i");
}

#[test]
fn scalar_loop_result_is_stored_before_parallel_consumers() {
    use Pipeline;
    let source = "entry main(xs: []i32, n: i32) []i32 =
        let bias = loop acc = 0 for i < n do acc + i in
        map(|x: i32| x + bias, xs)";
    compile(source);
    let output = pipeline(source);
    let [Pipeline::Compute(p)] = output.pipeline.pipelines.as_slice() else {
        panic!("compute pipeline");
    };
    assert_eq!(p.stages.len(), 2);
    assert_eq!(p.stages[0].workgroup_size, (1, 1, 1));
    assert_eq!(p.stages[1].workgroup_size, (64, 1, 1));
}

#[test]
fn scan_and_filter_emit_all_scheduled_kernels() {
    let scan = compile("entry main(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)");
    let filter = compile("entry main(xs: []i32) ?k. [k]i32 = filter(|x: i32| x % 3 == 1, xs)");
    assert_eq!(scan.entry_points.len(), 3);
    assert_eq!(filter.entry_points.len(), 4);
}

#[test]
fn tuple_collectives_and_runtime_captures_reach_wgsl() {
    compile("entry main(xs: []i32, bias: i32) (i32, i32) = reduce(|a: (i32, i32), b: (i32, i32)| (a.0 + b.0, a.1 + b.1), (0, 0), map(|x: i32| (x + bias, 1), xs))");
}

#[test]
fn nested_device_collectives_reach_wgsl() {
    compile("entry main(xs: []i32) []i32 = map(|x: i32| reduce(|a: i32, b: i32| a + b, 0, map(|y: i32| y + x, iota(5))), xs)");
}

#[test]
fn array_and_while_loops_reach_wgsl() {
    compile(
        "entry main(xs: [4]i32, n: i32) (i32, i32) =
        let a = loop acc = 0 for x in xs do acc + x in
        let (b, _) = loop (acc, i) = (0, 0) while i < n do (acc + i, i + 1) in (a, b)",
    );
}

#[test]
fn conditional_expressions_inside_device_loops_reach_wgsl() {
    compile("entry main(xs: []i32) []i32 = map(|x: i32| loop acc = 0 for i < x do if i % 2 == 0 then acc + i else acc, xs)");
}

#[test]
fn branching_loop_tests_and_header_array_reads_reach_wgsl() {
    compile("entry main(xs: [4]i32, limit: i32) i32 = loop acc = 0 while (if acc < limit then xs[acc % 4] > 0 else false) do acc + 1");
    compile("entry main(xs: [4]i32) i32 = loop acc = 0 for i < xs[0] do acc + i");
}

#[test]
fn runtime_control_keeps_collectives_inside_device_execution() {
    for source in [
        "entry main(xs: [4]i32, n: i32) [4]i32 = loop acc = xs for k < n do map(|x: i32| x + k, acc)",
        "entry main(xs: [4]i32, n: i32) i32 = loop acc = 0 for k < n do acc + reduce(|a:i32,b:i32|a+b, 0, map(|x:i32|x+k,xs))",
        "entry main(xs: [4]i32, yes: bool) [4]i32 = if yes then map(|x:i32|x+1,xs) else map(|x:i32|x-1,xs)",
    ] {
        compile(source);
    }
}

#[test]
fn existing_spirv_backend_also_accepts_the_handoff() {
    for source in [
        "entry main(xs: []i32) []i32 = map(|x: i32| x * 2, xs)",
        "entry main(xs: []i32) [2]i32 = [xs[0], xs[1]]",
        include_str!("../../../testfiles/filter_captures_runtime_array.wyn"),
        include_str!("../../../testfiles/reduce_over_map.wyn"),
        include_str!("../../../testfiles/scan_compute.wyn"),
        include_str!("../../../testfiles/filter_then_map.wyn"),
        "entry main(xs: [4]i32, n: i32) [4]i32 = loop acc = xs for k < n do map(|x: i32| x + k, acc)",
    ] {
        let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
        let program = schedule(
            simplify_and_place(insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap())
                .unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap();
        let ssa = to_ssa(&program, CodegenTarget::Spirv).unwrap();
        let output = lower_ssa_to_spirv(ssa).unwrap();
        let bytes: Vec<_> = output.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
        let module = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[test]
fn vector_input_and_runtime_gather_after_scan_reach_wgsl() {
    compile(include_str!("../../../testfiles/gather_scan_chain.wyn"));
}

#[test]
fn graphics_stages_preserve_shader_interfaces_and_draw_metadata() {
    use pipeline_descriptor::{Pipeline, ShaderStage};
    let source = include_str!("../../../testfiles/unified_triangle.wyn");
    let module = compile(source);
    assert_eq!(module.entry_points.len(), 2);
    assert!(module.entry_points.iter().any(|e| e.stage == naga::ShaderStage::Vertex));
    assert!(module.entry_points.iter().any(|e| e.stage == naga::ShaderStage::Fragment));
    let output = pipeline(source);
    let [Pipeline::Graphics(graphics)] = output.pipeline.pipelines.as_slice() else {
        panic!("one graphics pipeline");
    };
    assert_eq!(graphics.stages.len(), 2);
    assert_eq!(graphics.source_operation, Some(0));
    assert!(graphics.stages.iter().any(|s| matches!(s.stage, ShaderStage::Vertex)));
    assert!(graphics.stages.iter().any(|s| matches!(s.stage, ShaderStage::Fragment)));
    assert!(!graphics.fragment_outputs.is_empty());
    assert!(output.pipeline.source_results.is_empty());
}

fn pipeline(source: &str) -> LoweredWgsl {
    let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    let program = schedule(
        simplify_and_place(insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    lower_ssa_to_wgsl_with_pipeline(to_ssa(&program, CodegenTarget::Wgsl).unwrap()).unwrap()
}

#[test]
fn runtime_input_lengths_and_output_allocations_share_the_published_bindings() {
    use pipeline_descriptor::{Binding, BufferLen, Pipeline};
    let output = pipeline("entry main(xs: []i32, bias: i32) []i32 = map(|x:i32|x+bias, xs)");
    let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(module.entry_points.iter().any(|e| e
        .function
        .expressions
        .iter()
        .any(|(_, x)| matches!(x, naga::Expression::ArrayLength(_)))));
    let [Pipeline::Compute(p)] = output.pipeline.pipelines.as_slice() else {
        panic!("compute pipeline")
    };
    assert_eq!(p.stages.len(), 1);
    assert_eq!(p.stages[0].owner, "main");
    let result = &output.pipeline.source_results[0];
    assert!(p.bindings.iter().any(|b| matches!(b, Binding::StorageBuffer { set, binding, length: Some(BufferLen::LikeInput { set: 0, binding: 0, elem_bytes: 4, src_elem_bytes: 4 }), .. } if (*set, *binding) == (result.set, result.binding))));
    assert!(p.bindings.iter().any(
        |b| matches!(b, Binding::StorageBuffer { members, .. } if members.iter().any(|m| m.name == "bias"))
    ));
    assert!(p.bindings.iter().all(|b| !matches!(b, Binding::PushConstant { .. })));
}

#[test]
fn reduction_publication_has_scratch_writers_readers_and_a_source_result() {
    use pipeline_descriptor::{Binding, BufferLen, Pipeline};
    let output = pipeline("entry main(xs: [137]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs)");
    let [Pipeline::Compute(p)] = output.pipeline.pipelines.as_slice() else {
        panic!("compute pipeline")
    };
    assert_eq!(p.stages.len(), 3, "chunks, combine, scalar result publication");
    let scratch = p
        .bindings
        .iter()
        .position(|b| {
            matches!(
                b,
                Binding::StorageBuffer {
                    length: Some(BufferLen::Fixed { bytes: 12 }),
                    ..
                }
            )
        })
        .unwrap();
    assert!(p.stages[0].writes.contains(&scratch));
    assert!(p.stages[1].reads.contains(&scratch));
    assert!(!p.stages[1].writes.contains(&scratch));
    assert_eq!(output.pipeline.source_results.len(), 1);
    assert_eq!(output.pipeline.frame_graph.passes.len(), 3);
    assert!(output.pipeline.frame_graph.topological_order().is_ok());
}

#[test]
fn independent_entries_keep_their_own_dispatches_and_results() {
    let output = pipeline("entry first(xs:[5]i32) [5]i32 = map(|x:i32|x+1,xs)\nentry second(xs:[9]i32) [9]i32 = map(|x:i32|x*2,xs)");
    assert_eq!(output.pipeline.pipelines.len(), 2);
    assert_eq!(output.pipeline.source_results.len(), 2);
    assert_ne!(
        output.pipeline.source_results[0].pipeline_index,
        output.pipeline.source_results[1].pipeline_index
    );
    assert_ne!(
        output.pipeline.source_results[0].binding,
        output.pipeline.source_results[1].binding
    );
}

#[test]
fn scalar_results_after_collectives_are_executed_and_published() {
    let output = pipeline("entry main(xs:[]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs) * 3 + 1");
    assert_eq!(output.pipeline.source_results.len(), 1);
    let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(module.entry_points.len(), 3);
}

#[test]
fn in_place_results_keep_the_host_input_binding_and_upload_role() {
    use pipeline_descriptor::{Access, Binding, BufferUsage, Pipeline};
    let output = pipeline("entry main(dest:*[3]i32,xs:[5]i32) [3]i32 = reduce_by_index(dest,|a:i32,b:i32|a+b,0,map(|x:i32|x%3,xs),xs)");
    let result = &output.pipeline.source_results[0];
    assert_eq!((result.set, result.binding), (0, 0));
    let Pipeline::Compute(p) = &output.pipeline.pipelines[0] else {
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
fn consuming_maps_publish_the_input_as_their_read_write_result() {
    use pipeline_descriptor::{Access, Binding, BufferUsage};
    let output = pipeline("entry main(xs:*[]i32) []i32 = map(|x:i32|x+7,xs)");
    assert_eq!(output.pipeline.source_results[0].binding, 0);
    let Pipeline::Compute(p) = &output.pipeline.pipelines[0] else {
        panic!("compute")
    };
    assert_eq!(p.bindings.len(), 1);
    assert!(matches!(
        p.bindings[0],
        Binding::StorageBuffer {
            usage: BufferUsage::Input,
            access: Access::ReadWrite,
            ..
        }
    ));
    let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn fused_maps_publish_reused_nonprimary_inputs() {
    let output = pipeline(
        "entry main(xs:*[4]i32,ys:*[4]i32) ([4]i32,[4]i32) = (map(|x:i32|x+1,ys),map(|x:i32|x*2,xs))",
    );
    assert_eq!(
        output.pipeline.source_results.iter().map(|r| r.binding).collect::<Vec<_>>(),
        [1, 0]
    );
    let Pipeline::Compute(p) = &output.pipeline.pipelines[0] else {
        panic!("compute")
    };
    assert_eq!(p.bindings.len(), 2);
    assert_eq!(p.stages.len(), 1);
    let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn tuple_of_views_uses_the_tlc_component_bindings() {
    let source = "entry main(xs: ([]i32,[]i32)) ([]i32,[]i32) = xs";
    let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    let imported = from_tlc(&tlc).unwrap();
    let expected: Vec<_> = imported
        .ir
        .entry_params
        .values()
        .filter_map(|p| p.binding.as_ref())
        .flat_map(|p| match &p.kind {
            EntryParamBindingKind::Single { binding, .. } => vec![binding.binding],
            EntryParamBindingKind::TupleOfViews(fields) => {
                fields.iter().map(|f| f.binding.binding).collect()
            }
        })
        .collect();
    let output = pipeline(source);
    assert_eq!(output.pipeline.source_results.len(), 2);
    assert_eq!(
        output.pipeline.source_results.iter().map(|o| o.binding).collect::<Vec<_>>(),
        expected
    );
    let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn computed_graphics_captures_share_the_producer_binding() {
    use pipeline_descriptor::Binding;
    let output = pipeline(include_str!("../../../testfiles/playground/conway.wyn"));
    let [result] = output.pipeline.source_results.as_slice() else {
        panic!("one computed board");
    };
    assert!(output.pipeline.pipelines.iter().any(|p| {
        let Pipeline::Graphics(p) = p else { return false };
        p.bindings.iter().any(|b| {
            matches!(b, Binding::StorageBuffer { set, binding, .. }
            if (*set, *binding) == (result.set, result.binding))
        })
    }));
}

#[test]
fn host_sized_outputs_publish_uniform_dependencies_and_storage_stride() {
    use pipeline_descriptor::{Binding, BufferLen, HostSizeScalar};
    let source = include_str!("../../../testfiles/regressions/uniform_output_size.wyn")
        .replace("{ resolution: vec3f32 }", "{ padding: f32, resolution: vec3f32 }")
        .replace("([]f32,", "([]vec3f32,")
        .replace(
            "target_load(rendered, @[i % width, i / width], 0u32)",
            "@[f32(i), 0.0, 0.0]",
        );
    let output = pipeline(&source);
    let lengths: Vec<_> = output
        .pipeline
        .pipelines
        .iter()
        .filter_map(|p| {
            let Pipeline::Compute(p) = p else { return None };
            p.bindings.iter().find_map(|b| match b {
                Binding::StorageBuffer {
                    length: Some(BufferLen::HostProvided { inputs, elem_bytes }),
                    ..
                } => Some((inputs, *elem_bytes)),
                _ => None,
            })
        })
        .collect();
    assert_eq!(lengths.len(), 1);
    let (inputs, stride) = lengths[0];
    assert_eq!(stride, 16);
    assert_eq!(
        inputs.iter().map(|i| (i.name.as_str(), i.offset, i.scalar)).collect::<Vec<_>>(),
        [
            ("frame_resolution_x", 16, HostSizeScalar::F32),
            ("frame_resolution_y", 20, HostSizeScalar::F32)
        ]
    );
}

#[test]
fn shared_helper_reuses_its_emitted_body_and_storage_requirements() {
    use crate::egglog::blocks::{BufferData, Function, FunctionKind, Storage, Value};
    use crate::egglog::{Program, Scheduled};
    use crate::interface::{StorageBindingDecl, StorageRole};
    let mut data = Program {
        ir: Default::default(),
        state: Scheduled::default(),
    };
    let buffer = data.state.buffers.alloc(BufferData {
        name: "shared".into(),
        length: Value::Int(1),
        element: crate::types::i32(),
        storage: Storage::Device,
    });
    data.state.abi.bindings.insert(
        buffer,
        StorageBindingDecl {
            binding: crate::BindingRef::new(0, 0),
            elem_ty: crate::types::i32(),
            role: StorageRole::Input,
            logical_resource: None,
            length: None,
        },
    );
    let root = Function {
        name: "load".into(),
        kind: FunctionKind::Device,
        results: 1,
        blocks: vec![],
    }
    .insert(vec![], &mut data.state.blocks, &mut data.state.bodies);
    let crate::egglog::Exit::Return(returns) = data.state.blocks[root].exit else {
        panic!("return")
    };
    data.state.bodies[returns].results.push(Value::op("index", [Value::Buffer(buffer), Value::Int(0)]));
    let mut compiler = super::Compiler {
        origins: Default::default(),
        placements: Default::default(),
        data: &data,
        functions: vec![],
        specializations: Default::default(),
        active: Default::default(),
        used: Default::default(),
    };
    let first = compiler.function(root, vec![]).unwrap();
    assert_eq!(
        compiler.functions[first.0 as usize].body.return_ty,
        crate::types::i32()
    );
    assert!(compiler.used.contains(&buffer));
    compiler.used.clear();
    assert_eq!(compiler.function(root, vec![]).unwrap(), first);
    assert_eq!(compiler.functions.len(), 1);
    assert!(
        compiler.used.contains(&buffer),
        "cached helper still needs its storage in each shader"
    );
}

#[test]
fn runtime_launches_use_buffer_and_scalar_domains() {
    use pipeline_descriptor::{DispatchLen, DispatchSize};
    for source in [
        "entry main(xs: []i32) []i32 = map(|x:i32|x+1,xs)",
        "entry main(n: i32) []i32 = map(|i|i+1,iota(n))",
    ] {
        let output = pipeline(source);
        let [Pipeline::Compute(p)] = output.pipeline.pipelines.as_slice() else {
            panic!("compute")
        };
        assert!(matches!(
            p.stages[0].dispatch_size,
            DispatchSize::DerivedFrom {
                len: DispatchLen::InputBinding { .. } | DispatchLen::StorageBuffer { .. },
                workgroup_size: 64
            }
        ));
    }
}

#[test]
fn runtime_collective_scratch_has_an_input_capacity_and_chunk_grid() {
    use pipeline_descriptor::{Binding, BufferLen, DispatchLen, DispatchSize};
    let output = pipeline("entry main(xs: []i32) []i32 = scan(|a:i32,b:i32|a+b,0,xs)");
    let [Pipeline::Compute(p)] = output.pipeline.pipelines.as_slice() else {
        panic!("compute")
    };
    assert!(matches!(
        p.stages[0].dispatch_size,
        DispatchSize::DerivedFrom {
            len: DispatchLen::InputBinding { elem_bytes: 4, .. },
            workgroup_size: 4096
        }
    ));
    assert!(p.bindings.iter().all(|b| !matches!(
        b,
        Binding::StorageBuffer {
            length: Some(BufferLen::HostProvided { .. }),
            ..
        }
    )));
}

#[test]
fn bounded_filter_output_does_not_use_its_packed_backing_as_a_length() {
    use pipeline_descriptor::{Binding, BufferLen};
    let output = pipeline(include_str!("../../../testfiles/filter_then_map.wyn"));
    let result = &output.pipeline.source_results[0];
    let Pipeline::Compute(p) = &output.pipeline.pipelines[result.pipeline_index] else {
        panic!("compute")
    };
    assert!(p.bindings.iter().any(|b| matches!(b, Binding::StorageBuffer {
        set, binding, length: Some(BufferLen::Fixed { bytes: 16384 }), ..
    } if (*set, *binding) == (result.set, result.binding))));
}

#[test]
fn explicit_grids_preserve_all_axes_in_the_shader_and_descriptor() {
    use crate::interface::ComputeDispatchGrid;
    use pipeline_descriptor::DispatchSize;
    for source in [
        "entry main(xs:[4096]i32) [4096]i32 = map(|x:i32|x+1,xs)",
        "entry main(xs:[4096]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs)",
    ] {
        for (x, y, z) in [(1, 1, 1), (2, 3, 4)] {
            let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
            let mut imported = from_tlc(&tlc).unwrap();
            for (_, entry) in &mut imported.ir.entries {
                entry.declaration.compute_dispatch = Some(ComputeDispatchGrid { x, y, z });
            }
            let scheduled = schedule(
                simplify_and_place(insert_expressions(fuse(imported).unwrap()).unwrap()).unwrap(),
                PipelineTopologyPolicy::AllowGenerated,
            )
            .unwrap();
            let output =
                lower_ssa_to_wgsl_with_pipeline(to_ssa(&scheduled, CodegenTarget::Wgsl).unwrap()).unwrap();
            let [Pipeline::Compute(p)] = output.pipeline.pipelines.as_slice() else {
                panic!("compute")
            };
            assert_eq!(
                p.stages[0].dispatch_size,
                DispatchSize::Fixed {
                    x,
                    y,
                    z,
                    explicit: true
                }
            );
            for stage in &p.stages[1..] {
                assert_eq!(
                    stage.dispatch_size,
                    DispatchSize::Fixed {
                        x: 1,
                        y: 1,
                        z: 1,
                        explicit: true
                    }
                );
            }
            let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            if y > 1 {
                assert!(output.wgsl.contains(".y") && output.wgsl.contains(".z"));
            }
        }
    }
}

#[test]
fn direct_mode_keeps_collectives_in_the_authored_entry() {
    for source in [
        "entry main(xs:[137]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs)",
        "entry main(xs:[137]i32) [137]i32 = map(|x:i32|x+1,xs)",
        include_str!("../../../testfiles/unified_triangle.wyn"),
    ] {
        let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
        let scheduled = schedule(
            simplify_and_place(insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap())
                .unwrap(),
            PipelineTopologyPolicy::AuthoredOnly,
        )
        .unwrap();
        assert!(scheduled.state.dispatches.is_empty());
        let output =
            lower_ssa_to_wgsl_with_pipeline(to_ssa(&scheduled, CodegenTarget::Wgsl).unwrap()).unwrap();
        for pipeline in &output.pipeline.pipelines {
            if let Pipeline::Compute(p) = pipeline {
                assert_eq!(p.stages.len(), 1);
            }
        }
        let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}
