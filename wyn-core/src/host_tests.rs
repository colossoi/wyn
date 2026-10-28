use crate::host::arithmetic::{add, ceiling, dimension, floor, multiply, signed_size, size, subtract};
use crate::host::{
    Allocation, Binding, Expr, FrameGraph, Operation, Pipeline, Program, ResultLayout, ShaderFormat,
};
use crate::{compile_thru_ssa, lower_ssa_to_spirv, lower_ssa_to_wgsl_with_program};
use std::collections::BTreeSet;
use wyn_host_interp::Program as WhlProgram;

fn compile(source: &str) -> Program {
    lower_ssa_to_wgsl_with_program(compile_thru_ssa(source).unwrap()).unwrap().program
}

fn assert_dependency(graph: &FrameGraph, before: usize, after: usize) {
    let mut pending = graph.passes[after].depends_on.clone();
    let mut visited = BTreeSet::new();
    while let Some(index) = pending.pop() {
        if index == before {
            return;
        }
        if visited.insert(index) {
            pending.extend(&graph.passes[index].depends_on);
        }
    }
    panic!(
        "{} must depend on {}",
        graph.passes[after].name, graph.passes[before].name
    );
}

#[test]
fn runtime_array_joins_wait_for_producers_in_branch_arms() {
    for source in [
        "entry main(xs:[]i32) []i32 = if xs[0]>0 then replicate(2,7i32) else xs",
        "entry main(xs:[]i32) []i32 = if length(xs)>10 then replicate(0,0i32)
         else if length(xs)>5 then replicate(2,7i32) else xs",
    ] {
        let program = compile(source);
        let graph = &program.interface.frame_graph;
        let join = graph.passes.iter().position(|pass| pass.name == "main_view").unwrap();
        let producers: Vec<_> = graph
            .passes
            .iter()
            .enumerate()
            .filter(|(_, pass)| pass.name.starts_with("main_compute"))
            .map(|(i, _)| i)
            .collect();
        assert!(!producers.is_empty());
        for producer in producers {
            assert_dependency(graph, producer, join);
        }
    }
}

#[test]
fn device_computed_trip_counts_are_published_before_the_loop() {
    let program = compile(
        "entry main(xs:[]i32) []i32 =
        let n=reduce((+),0,xs) in
        loop values=xs for i<n do map(|x|x+1,values)",
    );
    let whl = program.to_whl("device_count", ShaderFormat::Wgsl).unwrap();
    assert!(whl.contains("gpu-read-scalar"), "{whl}");
    assert_eq!(
        program.entries[0].operations.iter().filter(|op| matches!(op, Operation::Loop { .. })).count(),
        1
    );
}

#[test]
fn rematerialized_helpers_and_dispatch_extents_publish_their_inputs() {
    for source in [
        include_str!("../../testfiles/scalar_setup.wyn"),
        include_str!("../../testfiles/rust_host_filter.wyn"),
    ] {
        let ssa = compile_thru_ssa(source).unwrap();
        let program = lower_ssa_to_spirv(ssa).unwrap().program;
        let Pipeline::Compute(pipeline) = &program.interface.pipelines[0] else {
            panic!("compute pipeline");
        };
        let push =
            pipeline.bindings.iter().position(|b| matches!(b, Binding::PushConstant { .. })).unwrap();
        assert!(pipeline.stages[0].uses.reads.contains(&push), "{source}");
    }
}

#[test]
fn finish_dispatch_waits_for_all_reduction_results() {
    let program = compile(
        "entry main(xs:[]i32) (i32,i32) =
        let ys=map(|x|x*x+17,xs) in
        (reduce((+),0,ys),reduce(|x,y|if x>y then x else y,0,ys))",
    );
    let dispatches: Vec<_> = program.entries[0]
        .operations
        .iter()
        .filter_map(|operation| {
            let Operation::Dispatch { pipeline, stage, .. } = operation else {
                return None;
            };
            let Pipeline::Compute(compute) = &program.interface.pipelines[*pipeline] else {
                panic!("compute pipeline");
            };
            Some(compute.stages[*stage].entry_point.as_str())
        })
        .collect();
    let combine = dispatches.iter().position(|name| name.ends_with("_combine")).unwrap();
    let finish = dispatches.iter().position(|name| name.ends_with("_finish")).unwrap();
    assert!(combine < finish, "{dispatches:?}");
}

#[test]
fn graphics_compute_helpers_capture_computed_records_and_return_both_arrays() {
    let source = include_str!("../../testfiles/graphics_computed_record_capture.wyn");
    for format in [ShaderFormat::Spirv, ShaderFormat::Wgsl] {
        let ssa = compile_thru_ssa(source).unwrap();
        let program = match format {
            ShaderFormat::Spirv => lower_ssa_to_spirv(ssa).unwrap().program,
            ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa).unwrap().program,
        };
        let [entry] = program.entries.as_slice() else {
            panic!("one source entry")
        };
        let resources = &program.interface.frame_graph.resources;
        let inputs: BTreeSet<_> = entry.inputs.iter().map(|id| resources[id.0].name.as_str()).collect();
        assert_eq!(inputs, BTreeSet::from(["ui", "points", "target"]), "{format:?}");
        assert_eq!(entry.results.len(), 3);
        assert_ne!(entry.results[0], entry.results[1]);
        for result in &entry.results[..2] {
            assert!(
                !entry.inputs.contains(result),
                "computed result must not alias an input"
            );
            assert!(entry.allocations.iter().any(
                |allocation| matches!(allocation, Allocation::Buffer { resource, .. } if resource == result)
            ));
        }
        assert!(entry.operations.iter().any(|op| matches!(op, Operation::Dispatch { .. })));
        assert!(entry.operations.iter().any(|op| matches!(op, Operation::Draw { .. })));
        program.to_rust_wgpu("record_capture", format).unwrap();
    }
}

#[test]
fn generated_compute_captures_are_allocated_by_the_authored_host_entry() {
    let source = include_str!("../../testfiles/regressions/graphics_computed_stage_inputs.wyn");
    for format in [ShaderFormat::Spirv, ShaderFormat::Wgsl] {
        let ssa = compile_thru_ssa(source).unwrap();
        let program = match format {
            ShaderFormat::Spirv => lower_ssa_to_spirv(ssa).unwrap().program,
            ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa).unwrap().program,
        };
        let [entry] = program.entries.as_slice() else {
            panic!("one authored entry")
        };
        let resources = &program.interface.frame_graph.resources;
        let inputs: BTreeSet<_> = entry.inputs.iter().map(|id| resources[id.0].name.as_str()).collect();
        assert_eq!(inputs, BTreeSet::from(["xs", "target"]), "{format:?}");
        assert!(!entry.inputs.contains(&entry.results[0]));
        assert!(entry.allocations.iter().any(|a| a.resource() == entry.results[0]));
        WhlProgram::parse(&program.to_whl("staged", format).unwrap()).unwrap();
    }
}

fn check_whl(source: &str) {
    let mut depth = 0i32;
    let mut string = false;
    let mut escaped = false;
    let mut comment = false;
    for c in source.chars() {
        if comment {
            if c == '\n' {
                comment = false;
            }
            continue;
        }
        if string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                string = false;
            }
            continue;
        }
        match c {
            ';' => comment = true,
            '"' => string = true,
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                assert!(depth >= 0, "unexpected closing parenthesis in {source}");
            }
            _ => {}
        }
    }
    assert!(!string && depth == 0, "unbalanced WHL: {source}");
}

#[test]
fn host_program_preserves_compute_phases_and_returns_source_results() {
    let program = compile("entry main(xs: []i32) []i32 = scan(|a:i32,b:i32|a+b,0,xs)");
    let [entry] = program.entries.as_slice() else {
        panic!("one host entry");
    };
    let stages = program
        .interface
        .pipelines
        .iter()
        .map(|p| match p {
            Pipeline::Compute(c) => c.stages.len(),
            Pipeline::Graphics(_) => 0,
        })
        .sum::<usize>();
    assert_eq!(
        entry.operations.iter().filter(|op| matches!(op, Operation::Dispatch { .. })).count(),
        stages
    );
    assert_eq!(entry.results.len(), 1);
    assert!(entry
        .operations
        .iter()
        .all(|o| matches!(o, Operation::Dispatch { .. } | Operation::Scalar { .. })));
    let whl = program.to_whl("scan.wgsl", ShaderFormat::Wgsl).unwrap();
    check_whl(&whl);
    assert_eq!(whl.matches("(gpu-dispatch ").count(), stages);
    assert!(whl.contains("(gpu-alloc "));
    assert!(whl.contains("(gpu-free "));
    assert!(whl.contains(":source-name \"main\""));
}

#[test]
fn rust_recording_api_batches_scan_passes_and_scratch_clears() {
    let program = compile("entry main(xs: []i32) []i32 = scan(|a:i32,b:i32|a+b,0,xs)");
    let rust = program.to_rust_wgpu("scan.wgsl", ShaderFormat::Wgsl).unwrap();
    let (wrapper, recording) = rust.split_once("pub fn encode_main(").unwrap();
    assert_eq!(wrapper.matches("queue.submit(").count(), 1);
    assert_eq!(wrapper.matches("create_command_encoder(").count(), 1);
    assert!(!recording.contains("queue.submit("));
    assert!(!recording.contains("create_command_encoder("));
    assert!(recording.contains("encoder.clear_buffer("));
    assert_eq!(recording.matches("pass.dispatch_workgroups(").count(), 3);
}

#[test]
fn both_host_outputs_include_buffer_capacity_arithmetic() {
    let program = compile("entry main(xs: []i32) []i32 = map(|x:i32|x+1,xs)");
    let whl = program.to_whl("map.wgsl", ShaderFormat::Wgsl).unwrap();
    check_whl(&whl);
    assert!(whl.contains("(gpu-buffer-size "));
    assert!(whl.contains("(floor "));
    let rust = program.to_rust_wgpu("map.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("create_compute_pipeline"));
    assert!(rust.contains("dispatch_workgroups"));
    assert!(rust.contains("floor("));
    assert!(rust.contains("i64"));
    assert!(rust.contains("u64"));
    for dependency in ["BigInt", "num_bigint", "num_integer", "num_traits"] {
        assert!(!rust.contains(dependency), "unexpected dependency: {dependency}");
    }
    assert!(!rust.contains("todo!"));
}

#[test]
fn integer_capacity_expressions_reach_both_hosts() {
    let program = compile("entry main(n:i32) []i32 = iota(n*2+3)");
    let [entry] = program.entries.as_slice() else {
        panic!("one host entry");
    };
    assert!(entry.allocations.iter().any(
        |a| matches!(a,Allocation::Buffer{bytes,..} if bytes.to_whl().unwrap().contains("host-read-scalar"))
    ));
    let whl = program.to_whl("sizes.wgsl", ShaderFormat::Wgsl).unwrap();
    check_whl(&whl);
    assert!(whl.contains("(wyn-i32-add "), "{whl}");
    assert!(whl.contains("(wyn-i32-mul "), "{whl}");
    assert!(!whl.contains("4294967296"));
    let rust = program.to_rust_wgpu("sizes.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains(".wrapping_add("));
    assert!(rust.contains(".wrapping_mul("));
    assert!(rust.contains("n: &ParameterBuffer"));
    let compact: String = rust.split_whitespace().collect();
    assert!(!rust.contains("n_host: &[u8]"));
    assert!(compact.contains("n.upload(device,encoder)"));
    assert!(compact.contains("support::scalar_bytes(resource_"));
    assert!(!rust.contains("read_host_scalar"));
    assert!(!rust.contains("read_gpu_word"));
    assert!(rust.contains("pub fn encode_main("));
}

#[test]
fn resolution_allocation_expressions_keep_cpu_parameters() {
    let source = include_str!("../../testfiles/rust_host_resolution_sizes.wyn");
    for format in [ShaderFormat::Spirv, ShaderFormat::Wgsl] {
        let ssa = compile_thru_ssa(source).unwrap();
        let program = match format {
            ShaderFormat::Spirv => lower_ssa_to_spirv(ssa).unwrap().program,
            ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa).unwrap().program,
        };
        let entry = &program.entries[0];
        assert_eq!(entry.results.len(), 4);
        assert_eq!(entry.host_inputs.len(), 1);
        let (&frame, &bytes) = entry.host_inputs.first_key_value().unwrap();
        assert_eq!(program.interface.frame_graph.resources[frame.0].name, "frame");
        assert_eq!(
            bytes, 24,
            "only resolution.x/y are needed at std140 offsets 16/20"
        );
        let whl = program.to_whl("resolution_sizes", format).unwrap();
        assert!(whl.contains(":host-bytes 24"));
        assert!(whl.contains("host-read-scalar"));
        assert!(!whl.contains("gpu-read-scalar"));
        for id in entry.results.iter().take(3) {
            let Allocation::Buffer { bytes, .. } =
                entry.allocations.iter().find(|a| a.resource() == *id).unwrap()
            else {
                panic!("computed output allocation");
            };
            let mut reads = Vec::new();
            bytes.clone().reads_mut(&mut |_, offset| reads.push(*offset));
            assert_eq!(reads, [16, 20]);
        }
        let rust = program.to_rust_wgpu("resolution_sizes", format).unwrap();
        assert!(rust.contains("frame: &ParameterBuffer"), "{rust}");
        assert!(!rust.contains("frame_host: &[u8]"), "{rust}");
        assert!(rust.contains("frame.upload(device, encoder)"), "{rust}");
        assert!(rust.contains("pub fn encode_sizes("), "{rust}");
        for readback in [
            "read_gpu_word",
            "map_async",
            "wait_indefinitely",
            "support::read_f32",
        ] {
            assert!(!rust.contains(readback), "unexpected {readback}: {rust}");
        }
    }
}

#[test]
fn record_output_descriptors_preserve_authored_field_names() {
    let program = compile("entry main() {count:i32, gain:f32} = {count=7, gain=2.5}");
    let results = &program.interface.source_results;
    assert_eq!(
        results.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
        ["count", "gain"]
    );
    let rust = program.to_rust_wgpu("record.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("name: \"count\""));
    assert!(rust.contains("name: \"gain\""));
    assert_eq!(rust.matches("kind: ResultKind::RecordField").count(), 2);
    assert!(!rust.contains("pub fn read_"));
    let whl = program.to_whl("record.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(whl.contains(":source-name \"count\""));
    assert!(whl.contains(":value-layout :i32"));
    check_whl(&whl);
}

#[test]
fn array_output_descriptors_publish_padding_and_logical_view_ranges() {
    let program =
        compile("entry positions(xs: []vec3f32) []vec3f32 = map(|x:vec3f32| x + @[1.0,2.0,3.0],xs)");
    let ResultLayout::Array { element, stride, .. } = &program.interface.source_results[0].layout else {
        panic!("array layout");
    };
    assert_eq!(*stride, 16);
    assert!(matches!(
        element.as_ref(),
        ResultLayout::Sequence {
            count: 3,
            stride: 4,
            ..
        }
    ));
    let rust = program.to_rust_wgpu("vectors.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("range: BufferRange::Bytes"), "{rust}");
    assert!(rust.contains("ResultLayout::Array {"));
    assert!(rust.contains("stride: 16u32"));
    assert!(rust.contains("length: None"));
    assert!(!rust.contains("map_async"));
}

#[test]
fn tuple_output_descriptors_require_no_readback_or_decoder() {
    let program = compile("entry statistics() (i32,u32,f32) = (-7,4294967295u32,2.5)");
    let rust = program.to_rust_wgpu("tuple.wgsl", ShaderFormat::Wgsl).unwrap();
    assert_eq!(rust.matches("kind: ResultKind::TupleField").count(), 3);
    for (index, scalar) in ["I32", "U32", "F32"].iter().enumerate() {
        assert!(rust.contains(&format!("name: \"result_{index}\"")));
        assert!(rust.contains(&format!("ResultLayout::Scalar(ResultScalar::{scalar})")));
    }
    for operation in ["pub fn read_", "map_async", "from_le_bytes", ".poll("] {
        assert!(
            !rust.contains(operation),
            "output descriptor contains {operation}"
        );
    }
}

#[test]
fn single_field_record_output_descriptor_preserves_record_shape() {
    let program = compile("entry main() {value:i32} = {value=7}");
    let rust = program.to_rust_wgpu("single.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("kind: ResultKind::RecordField"), "{rust}");
    assert!(rust.contains("name: \"value\""));
}

#[test]
fn output_descriptors_preserve_nested_record_offsets() {
    let program = compile("entry particles(xs: []{position:vec3f32, weight:f32}) []{position:vec3f32, weight:f32} = map(|x| {position=x.position,weight=x.weight+1.0},xs)");
    let ResultLayout::Array { element, stride, .. } = &program.interface.source_results[0].layout else {
        panic!("array layout");
    };
    let ResultLayout::Record { fields, size } = element.as_ref() else {
        panic!("record layout");
    };
    assert_eq!((*stride, *size), (16, 16));
    assert_eq!(
        fields.iter().map(|f| (f.name.as_str(), f.offset)).collect::<Vec<_>>(),
        [("position", 0), ("weight", 12)]
    );
    let rust = program.to_rust_wgpu("particles.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("offset: 12u32"));
    assert!(rust.contains("ResultLayout::Record"));
    assert!(!rust.contains("from_le_bytes"));
}

#[test]
fn output_descriptors_do_not_require_a_rust_decoder_for_every_type() {
    let program = compile("entry main() f16 = 1.0f16");
    assert!(matches!(
        program.interface.source_results[0].layout,
        ResultLayout::Unsupported(_)
    ));
    let rust = program.to_rust_wgpu("half.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("ResultLayout::Unsupported("));
    assert!(rust.contains("range: BufferRange::CallerProvided"));
    assert!(!rust.contains("from_le_bytes"));
}

#[test]
fn published_shader_names_identify_source_phases_and_stay_consistent() {
    let source = "entry totals(xs:[137]i32,ys:[67]i32) (i32,i32) = (reduce(|a:i32,b:i32|a+b,0,xs),reduce(|a:i32,b:i32|a*b,1,ys))";
    let ssa = compile_thru_ssa(source).unwrap();
    let names: BTreeSet<_> = ssa.entry_points.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names.len(), ssa.entry_points.len());
    assert!(names.iter().all(|n| n.starts_with("totals_")), "{names:?}");
    assert!(names.contains("totals_partials"));
    assert!(names.contains("totals_combine"));
    let compiled = lower_ssa_to_wgsl_with_program(ssa).unwrap();
    let module = naga::front::wgsl::parse_str(&compiled.wgsl).unwrap();
    let shader_names: BTreeSet<_> = module.entry_points.iter().map(|e| e.name.as_str()).collect();
    for pipeline in &compiled.program.interface.pipelines {
        let Pipeline::Compute(pipeline) = pipeline else {
            panic!("compute pipeline")
        };
        for stage in &pipeline.stages {
            assert!(shader_names.contains(stage.entry_point.as_str()));
        }
    }
    let resource_names: Vec<_> =
        compiled.program.interface.frame_graph.resources.iter().map(|r| r.name.as_str()).collect();
    assert!(resource_names.contains(&"totals_result_0"), "{resource_names:?}");
    assert!(resource_names.contains(&"totals_result_1"));
    assert!(resource_names.iter().any(|n| n.starts_with("totals_scratch")));
    let whl = compiled.program.to_whl("totals.wgsl", ShaderFormat::Wgsl).unwrap();
    let rust = compiled.program.to_rust_wgpu("totals.wgsl", ShaderFormat::Wgsl).unwrap();
    for artifact in [&compiled.wgsl, &whl, &rust] {
        for prefix in ["egg_kernel", "egg_resource", "egg_helper"] {
            assert!(!artifact.contains(prefix), "unexpected generated name {prefix}");
        }
    }
}

#[test]
fn generated_buffer_names_do_not_alias_source_inputs_with_the_same_name() {
    let program = compile("entry main(main_output:[4]i32) [4]i32 = map(|x:i32|x+1,main_output)");
    let entry = &program.entries[0];
    assert!(!entry.inputs.contains(&entry.results[0]));
    let resource = &program.interface.frame_graph.resources[entry.results[0].0];
    assert_eq!(resource.name, "main_output_2");
}

#[test]
fn rust_buffer_arguments_keep_source_names_in_bindings_sizes_and_results() {
    let program = compile("entry main(buffer:[]i32) []i32 = map(|x:i32|x+1,buffer)");
    let rust = program.to_rust_wgpu("map.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("buffer: &Buffer"), "{rust}");
    assert!(rust.contains("buffer.size()"));
    assert!(rust.contains("buffer: &buffer,"));
    assert!(rust.contains("size: wgpu::BufferSize::new(buffer.size())"));

    let program = compile("entry echo(samples:[]i32) []i32 = samples");
    let rust = program.to_rust_wgpu("echo.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("samples: &Buffer"));
    assert!(rust.contains("Buffer::clone(&samples)"));
}

#[test]
fn rust_generated_result_preserves_an_aliased_source_input() {
    let program = compile(
        "entry generated(result_field_1:[]i32, count:f32) ([]i32, []i32) = (result_field_1, iota(i32(count)))"
    );
    let entry = &program.entries[0];
    assert!(entry.inputs.contains(&entry.results[0]));
    assert!(!entry.inputs.contains(&entry.results[1]));
    let rust = program.to_rust_wgpu("collision.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("result_field_1: &Buffer"));
    assert!(rust.contains("Buffer::clone(&result_field_1)"));
    assert!(!rust.contains("result_field_1_2: &Buffer"));
}

#[test]
fn rust_allocates_graphics_intermediates_from_host_scalar_expressions() {
    let program = compile(
        r#"
entry scene(count:f32, target:render_target<vec4f32>) render_target<vec4f32> =
  let vertices = map(|i:i32| @[f32(i), 0.0, 0.0, 1.0], iota(i32(count)))
  let triangles = rasterize_triangles(direct_draw(3u32, 1u32),
    |vertex_index:u32, _:u32, _:u32|
      vertex_output(vertices[i32(vertex_index)], @[1.0, 0.0, 0.0, 1.0])) in
  shade(target, triangles, |value, _, _, _, _| value)
"#,
    );
    let entry = &program.entries[0];
    assert!(program.interface.source_results.is_empty());
    assert!(entry.allocations.iter().any(|a| matches!(a, Allocation::Buffer { .. })));
    let rust = program.to_rust_wgpu("scratch.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("count: &ParameterBuffer"));
    assert!(entry.inputs.iter().all(|id| {
        program.interface.frame_graph.resources[id.0].bindings.iter().all(|binding| {
            !matches!(
                program.bindings(binding.pipeline_index)[binding.binding_index],
                Binding::StorageBuffer {
                    usage: crate::host::BufferUsage::Output | crate::host::BufferUsage::Intermediate,
                    ..
                }
            )
        })
    }));
    assert!(!rust.contains("count_host: &[u8]"));
    assert!(!rust.contains("read_gpu_word"));
}

#[test]
fn rust_graphics_arguments_keep_buffer_texture_and_sampler_source_names() {
    let program = compile(include_str!("../../testfiles/texture_sample.wyn"));
    let rust = program.to_rust_wgpu("texture.wgsl", ShaderFormat::Wgsl).unwrap();
    for parameter in [
        "positions: &Buffer",
        "tex: &Texture",
        "samp: &Sampler",
        "screen: &Texture",
    ] {
        assert!(rust.contains(parameter), "missing {parameter} in {rust}");
    }
    assert!(rust.contains("buffer: &positions,"));
    assert!(rust.contains("size: wgpu::BufferSize::new(positions.size())"));
    let compact: String = rust.split_whitespace().collect();
    assert!(compact.contains("tex.create_view("));
    assert!(rust.contains("BindingResource::Sampler(&samp)"));
    assert!(compact.contains("screen.create_view("));
    assert!(rust.contains("Texture::clone(&screen)"));
}

#[test]
fn rust_argument_names_handle_keywords_and_generated_binding_collisions() {
    let mut program = compile("entry main(input:[]i32) []i32 = map(|x:i32|x+1,input)");
    let input = *program.entries[0].inputs.iter().next().unwrap();

    for (source, parameter) in [
        ("async", "r#async"),
        ("crate", "crate_2"),
        ("device", "device_2"),
        ("queue", "queue_2"),
        ("shader", "shader"),
        ("context", "context_2"),
        ("groups", "groups_2"),
        ("pipeline", "pipeline_2"),
        ("group_0", "group_0_2"),
        ("resource_1", "resource_1_2"),
        ("size", "size_2"),
    ] {
        // Only the emitted Rust argument spelling changes; the shader and resource graph do not.
        let resource = &mut program.interface.frame_graph.resources[input.0];
        resource.name = source.into();
        for binding in &mut resource.bindings {
            binding.name = source.into();
        }
        let rust = program.to_rust_wgpu("names.wgsl", ShaderFormat::Wgsl).unwrap();
        assert!(rust.contains(&format!("{parameter}: &Buffer")), "{rust}");
        assert!(rust.contains(&format!("{parameter}.size()")), "{rust}");
        assert!(rust.contains(&format!("buffer: &{parameter},")), "{rust}");
    }
}

#[test]
fn fixed_width_host_arithmetic_checks_overflow_and_keeps_large_byte_sizes() {
    let bytes = multiply(i64::from(u32::MAX), 16).unwrap();
    assert_eq!(size(bytes).unwrap(), 68_719_476_720);
    assert_eq!(signed_size(size(bytes).unwrap()).unwrap(), bytes);
    assert!(dimension(bytes).is_err());
    assert!(size(-1).is_err());
    assert!(signed_size(u64::MAX).is_err());
    assert!(add(i64::MAX, 1).is_err());
    assert!(subtract(i64::MIN, 1).is_err());
    assert!(multiply(i64::MAX, 2).is_err());
    assert_eq!(dimension(i64::from(u32::MAX)).unwrap(), u32::MAX);
}

#[test]
fn fixed_width_division_rounds_without_overflowing_intermediates() {
    for (a, b, down, up) in [(5, 2, 2, 3), (-5, 2, -3, -2), (5, -2, -3, -2), (-5, -2, 2, 3)] {
        assert_eq!(floor(a, b).unwrap(), down);
        assert_eq!(ceiling(a, b).unwrap(), up);
    }
    assert_eq!(ceiling(i64::MAX, 2).unwrap(), (i64::MAX / 2) + 1);
    for (a, b) in [(1, 0), (i64::MIN, -1)] {
        assert!(floor(a, b).is_err());
        assert!(ceiling(a, b).is_err());
    }
}

#[test]
fn graphics_capture_uses_the_produced_buffer_after_its_writer() {
    let program = compile(include_str!("../../testfiles/playground/conway.wyn"));
    let [entry] = program.entries.as_slice() else {
        panic!("one graphics host entry");
    };
    let result = entry.results[0];
    assert!(
        !entry.inputs.contains(&result),
        "computed capture is allocated by the host function"
    );
    let draw = entry.operations.iter().position(|op| matches!(op, Operation::Draw { .. })).unwrap();
    let producer = entry
        .operations
        .iter()
        .position(|op| {
            let Operation::Dispatch { pipeline, stage, .. } = op else {
                return false;
            };
            let Pipeline::Compute(compute) = &program.interface.pipelines[*pipeline] else {
                return false;
            };
            compute.stages[*stage]
                .uses
                .writes
                .iter()
                .any(|&binding| program.binding_resource(*pipeline, binding).unwrap() == result)
        })
        .unwrap();
    assert!(producer < draw, "graphics waits for its captured buffer");
    let whl = program.to_whl("conway.wgsl", ShaderFormat::Wgsl).unwrap();
    check_whl(&whl);
    assert!(whl.contains("define-gpu-graphics"));
    assert!(whl.contains("gpu-draw"));
    let rust = program.to_rust_wgpu("conway.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("create_render_pipeline"));
    assert!(rust.contains("begin_render_pass"));
    assert!(rust.contains("OutputResource::Texture(Texture::clone("));
    assert!(!rust.contains("read_buffers"));
    for pipeline in &program.interface.pipelines {
        if let Pipeline::Graphics(graphics) = pipeline {
            for stage in &graphics.stages {
                assert!(
                    stage.entry_point.starts_with(&format!("{}_", stage.owner)),
                    "{}",
                    stage.entry_point
                );
            }
        }
    }
}

#[test]
fn graphics_aliases_preserve_host_buffer_identity() {
    // The exhaustive spellings stop at stage extraction. These two in-process
    // compilations check the final allocation and scheduling contract as well.
    for (prefix, input, computed) in [
        ("", "values", false),
        (
            "let produced = map(|v| v + @[1.0, 0.0, 0.0, 0.0], values)",
            "produced",
            true,
        ),
    ] {
        let program = compile(&format!(
            "entry reproduce(values: []vec4f32, target: render_target<vec4f32>) render_target<vec4f32> =
               {prefix}
               let first = ({input}, target) let (unpacked, _) = first
               let second = {{ data = (unpacked, values), tag = 2i32 }}
               let alias = second.data.0
               let fragments = rasterize_triangles(direct_draw(3u32, 1u32),
                 |_, _, _| vertex_output(alias[0], ())) in
               shade(target, fragments, |_, _, _, _, _| alias[0])"
        ));
        let [entry] = program.entries.as_slice() else {
            panic!("one source entry")
        };
        assert_eq!(entry.inputs.len(), 2, "generated buffers must remain internal");
        assert_eq!(entry.results.len(), 1, "only the render target is returned");
        assert!(entry.inputs.contains(&entry.results[0]), "borrowed render target");
        // Computed captures allocate element storage and an explicit input length.
        assert_eq!(entry.allocations.len(), 2 * usize::from(computed));
        let captured = if computed {
            let Allocation::Buffer { resource, .. } = entry.allocations[0] else {
                panic!("computed capture needs one buffer")
            };
            assert!(!entry.inputs.contains(&resource));
            resource
        } else {
            assert!(entry.operations.iter().all(|op| matches!(op, Operation::Draw { .. })));
            *entry.inputs.iter().find(|id| **id != entry.results[0]).unwrap()
        };
        let draws = entry
            .operations
            .iter()
            .enumerate()
            .filter_map(|(index, op)| match op {
                Operation::Draw { pipeline } => Some((index, *pipeline)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let [(draw, pipeline)] = draws.as_slice() else {
            panic!("one draw")
        };
        let Pipeline::Graphics(graphics) = &program.interface.pipelines[*pipeline] else {
            panic!("graphics pipeline")
        };
        assert_eq!(graphics.stages.len(), 2);
        for stage in &graphics.stages {
            let buffers = stage
                .uses
                .reads
                .iter()
                .filter(|&&binding| matches!(graphics.bindings[binding], Binding::StorageBuffer { .. }))
                .map(|&binding| program.binding_resource(*pipeline, binding).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                buffers,
                [captured],
                "{:?} capture, computed={computed}",
                stage.stage
            );
        }
        if computed {
            assert!(
                entry.operations[..*draw].iter().any(|op| {
                    let Operation::Dispatch { pipeline, stage, .. } = op else {
                        return false;
                    };
                    let Pipeline::Compute(compute) = &program.interface.pipelines[*pipeline] else {
                        return false;
                    };
                    compute.stages[*stage]
                        .uses
                        .writes
                        .iter()
                        .any(|&binding| program.binding_resource(*pipeline, binding).unwrap() == captured)
                }),
                "the captured buffer must be produced before the draw"
            );
        }
    }
}

#[test]
fn whl_paths_escape_lisp_strings() {
    let program = compile("entry main() i32 = 1");
    let whl = program.to_whl("a\\b\"c.wgsl", ShaderFormat::Wgsl).unwrap();
    check_whl(&whl);
    assert!(whl.contains("a\\\\b\\\"c.wgsl"));
}

#[test]
fn rust_spirv_embeds_binary_and_binds_compute_push_constants() {
    let program = lower_ssa_to_spirv(
        compile_thru_ssa("entry main(xs: []i32, bias: i32) []i32 = map(|x:i32|x+bias,xs)").unwrap(),
    )
    .unwrap()
    .program;
    let rust = program.to_rust_wgpu("shader.spv", ShaderFormat::Spirv).unwrap();
    assert!(rust.contains("ShaderSource::SpirV"));
    assert!(rust.contains("include_bytes!(\"shader.spv\")"));
    assert!(rust.contains("PushConstantRange"));
    assert!(rust.contains("pass.set_push_constants("));
    assert!(rust.contains("bias: &[u8]"));
    assert!(!rust.contains("include_str!"));
}

#[test]
fn rust_spirv_emits_explicit_graphics_layouts() {
    let program = lower_ssa_to_spirv(
        compile_thru_ssa(include_str!("../../testfiles/playground/conway.wyn")).unwrap(),
    )
    .unwrap()
    .program;
    let rust = program.to_rust_wgpu("conway.spv", ShaderFormat::Spirv).unwrap();
    assert!(rust.contains("ShaderStages::VERTEX | ShaderStages::FRAGMENT"));
    assert!(rust.contains("create_render_pipeline"));
    assert!(rust.contains("create_pipeline_layout"));
}

#[test]
fn filter_phase_layout_matches_the_selected_shader_storage_access() {
    let source = include_str!("../../testfiles/rust_host_storage_access_mismatch.wyn");
    for (format, read_only) in [(ShaderFormat::Spirv, true), (ShaderFormat::Wgsl, false)] {
        let ssa = compile_thru_ssa(source).unwrap();
        let program = match format {
            ShaderFormat::Spirv => lower_ssa_to_spirv(ssa).unwrap().program,
            ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa).unwrap().program,
        };
        let rust = program.to_rust_wgpu("storage_access", format).unwrap();
        let [Pipeline::Compute(pipeline)] = program.interface.pipelines.as_slice() else {
            panic!("compute pipeline")
        };
        let (consumer, producers) = pipeline.stages.split_last().unwrap();
        let shared = consumer
            .reads
            .iter()
            .find(|binding| {
                !consumer.writes.contains(binding)
                    && producers.iter().any(|stage| stage.writes.contains(binding))
            })
            .unwrap();
        let Binding::StorageBuffer {
            set, binding: slot, ..
        } = pipeline.bindings[*shared]
        else {
            panic!("shared storage buffer")
        };
        let phase = rust
            .split("let compute_")
            .find(|phase| phase.contains(&format!("entry_point: Some({:?})", consumer.entry_point)))
            .unwrap();
        let layout = phase.split("let pipeline =").next().unwrap();
        let binding = layout
            .split("BindGroupLayoutEntry {")
            .find(|entry| entry.trim_start().starts_with(&format!("binding: {slot}u32,")))
            .unwrap();
        assert!(
            binding.contains(&format!("read_only: {read_only}")),
            "{format:?}: {binding}"
        );

        let whl = WhlProgram::parse(&program.to_whl("storage_access", format).unwrap()).unwrap();
        let phase = whl
            .kernels
            .values()
            .find(|kernel| kernel.options.text(":entry").unwrap() == consumer.entry_point)
            .unwrap();
        let abi = phase.options.get(":abi").unwrap().list().unwrap();
        let binding = abi
            .iter()
            .map(|value| value.list().unwrap())
            .find(|binding| {
                binding[1].text().unwrap() == ":storage"
                    && binding[2].u32().unwrap() == set
                    && binding[3].u32().unwrap() == slot
            })
            .unwrap();
        let parameter =
            phase.parameters.iter().find(|parameter| parameter.name == binding[0].text().unwrap()).unwrap();
        assert_eq!(
            parameter.access.as_deref(),
            Some(if read_only { ":read" } else { ":read-write" })
        );
    }
}

#[test]
fn uniform_sized_maps_launch_from_their_logical_domain() {
    let source = include_str!("../../testfiles/rust_host_runtime_dispatch.wyn");
    for format in [ShaderFormat::Spirv, ShaderFormat::Wgsl] {
        let ssa = compile_thru_ssa(source).unwrap();
        let program = match format {
            ShaderFormat::Spirv => lower_ssa_to_spirv(ssa).unwrap().program,
            ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa).unwrap().program,
        };
        let [entry] = program.entries.as_slice() else {
            panic!("one host entry");
        };
        let pixels = entry.results[0];
        assert!(
            entry.allocations.iter().any(|a| a.resource() == pixels),
            "host computes and allocates the output capacity"
        );
        let [Operation::Dispatch { groups, .. }, Operation::Draw { .. }] = entry.operations.as_slice()
        else {
            panic!("one map followed by its consuming draw");
        };
        let mut domain = groups[0].clone();
        let mut sources = Vec::new();
        domain.reads_mut(&mut |source, _| sources.push(source.clone()));
        assert!(!sources.is_empty(), "launch reads the source dimensions");
        assert!(
            !sources.contains(&crate::host::ScalarSource::Resource(pixels)),
            "allocation capacity must not determine the logical launch domain"
        );
        assert_eq!(&groups[1..], &[Expr::Integer(1), Expr::Integer(1)]);
        let rust = program.to_rust_wgpu("runtime_dispatch", format).unwrap();
        let call = rust.split_once("pub fn host_reproduce(").unwrap().1;
        assert!(call.contains("_bytes"), "{call}");
        assert!(call.contains("ceiling("), "{call}");
        let whl = program.to_whl("runtime_dispatch", format).unwrap();
        assert!(whl.contains(&groups[0].to_whl().unwrap()), "{whl}");
    }
}

#[test]
fn indirect_draw_waits_for_both_count_epilogue_and_compacted_vertices() {
    let source = include_str!("../../testfiles/rust_host_indirect_epilogue.wyn");
    for format in [ShaderFormat::Spirv, ShaderFormat::Wgsl] {
        let ssa = compile_thru_ssa(source).unwrap();
        let program = match format {
            ShaderFormat::Spirv => lower_ssa_to_spirv(ssa).unwrap().program,
            ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa).unwrap().program,
        };
        let graph = &program.interface.frame_graph;
        assert_eq!(graph.passes.len(), 2, "one compaction dispatch and one draw");
        let [indirect] = graph.indirect_draws.as_slice() else {
            panic!("one indirect draw")
        };
        let writer = graph
            .passes
            .iter()
            .position(|pass| pass.writes.iter().any(|w| w.resource == indirect.buffer_resource))
            .unwrap();
        let compact = graph.passes.iter().position(|pass| pass.name.ends_with("compact")).unwrap();
        assert_eq!(
            writer, compact,
            "compaction also publishes the indirect draw command"
        );
        let draw = &graph.passes[indirect.draw_pass];
        assert!(draw.depends_on.contains(&writer));
        assert!(draw.depends_on.contains(&compact));
        assert!(graph.passes[compact]
            .writes
            .iter()
            .any(|w| draw.reads.iter().any(|r| r.resource == w.resource)));
    }
}

#[test]
fn texture_readers_finish_before_the_target_is_overwritten() {
    let program = compile(
        "def replace_state: fragment_state = {
             depth_test = #disabled, depth_write = false, blend = #replace, color_write = true
         }
         entry main(vertices:[3]vec4f32, coords:[137]vec2i32, target:render_target<f32>)
         ([137]f32,render_target<f32>) =
         let first = rasterize_triangles(direct_draw(3u32,1u32),
             |i,_,_|vertex_output(vertices[i32(i)],()))
         let image = shade_with(replace_state,target,first,|_,_,_,_,_|0.25)
         let old = map(|coord|target_load(image,coord,0u32),coords)
         let second = rasterize_triangles(direct_draw(3u32,1u32),
             |i,_,_|vertex_output(vertices[i32(i)],()))
         let updated = shade_with(replace_state,image,second,|_,_,_,_,_|0.75)
         in (old,updated)",
    );
    let entry = &program.entries[0];
    let [sampled, target] = entry.results.as_slice() else {
        panic!("sampled values and updated target")
    };
    let graph = &program.interface.frame_graph;
    let draw = |operation| {
        graph
            .passes
            .iter()
            .position(|pass| {
                matches!(
                    &program.interface.pipelines[pass.pipeline_index],
                    Pipeline::Graphics(graphics) if graphics.source_operation == Some(operation)
                )
            })
            .unwrap()
    };
    let reader = graph
        .passes
        .iter()
        .position(|pass| pass.writes.iter().any(|access| access.resource == sampled.0))
        .unwrap();
    assert!(graph.passes[reader].reads.iter().any(|access| access.resource == target.0));
    for writer in [draw(0), draw(1)] {
        assert!(graph.passes[writer].writes.iter().any(|access| access.resource == target.0));
    }
    assert_dependency(graph, draw(0), reader);
    assert_dependency(graph, reader, draw(1));
}

#[test]
fn graphics_root_composes_multiple_computations_and_source_results() {
    let source = include_str!("../../testfiles/rust_host_frame_composition.wyn");
    for format in [ShaderFormat::Spirv, ShaderFormat::Wgsl] {
        let ssa = compile_thru_ssa(source).unwrap();
        let program = match format {
            ShaderFormat::Spirv => lower_ssa_to_spirv(ssa).unwrap().program,
            ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa).unwrap().program,
        };
        let [entry] = program.entries.as_slice() else {
            panic!(
                "one source root, got {:?}",
                program.entries.iter().map(|e| &e.name).collect::<Vec<_>>()
            );
        };
        assert_eq!(entry.name, "reproduce");
        assert_eq!(entry.inputs.len(), 2);
        assert_eq!(entry.results.len(), 3);
        assert_eq!(
            entry.operations.iter().filter(|op| matches!(op, Operation::Dispatch { .. })).count(),
            2
        );
        let draw = entry.operations.iter().position(|op| matches!(op, Operation::Draw { .. })).unwrap();
        for &id in &entry.results[..2] {
            assert!(!entry.inputs.contains(&id));
            assert!(entry
                .allocations
                .iter()
                .any(|a| matches!(a, Allocation::Buffer { resource, .. } if *resource == id)));
            assert!(entry.operations.iter().any(|op| {
                let Operation::Dispatch { pipeline, stage, .. } = op else {
                    return false;
                };
                let Pipeline::Compute(compute) = &program.interface.pipelines[*pipeline] else {
                    return false;
                };
                compute.stages[*stage]
                    .uses
                    .writes
                    .iter()
                    .any(|&b| program.binding_resource(*pipeline, b).unwrap() == id)
            }));
        }
        let Operation::Draw { pipeline } = entry.operations[draw] else {
            unreachable!()
        };
        assert!(program
            .parameter_indices(pipeline, None)
            .iter()
            .any(|&b| program.binding_resource(pipeline, b).unwrap() == entry.results[0]));
        assert!(entry.operations[..draw].iter().any(|op| {
            let Operation::Dispatch { pipeline, stage, .. } = op else {
                return false;
            };
            let Pipeline::Compute(compute) = &program.interface.pipelines[*pipeline] else {
                return false;
            };
            compute.stages[*stage]
                .uses
                .writes
                .iter()
                .any(|&b| program.binding_resource(*pipeline, b).unwrap() == entry.results[0])
        }));
        let results = &program.interface.source_results;
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.entry == "reproduce"));
        assert_eq!(
            results.iter().map(|r| (r.result, r.name.as_str())).collect::<Vec<_>>(),
            [(0, "result_0"), (1, "result_1")]
        );
        let rust = program.to_rust_wgpu("composition", format).unwrap();
        assert_eq!(rust.matches("pub fn host_").count(), 1);
        assert_eq!(rust.matches("pass.dispatch_workgroups(").count(), 2);
        assert_eq!(rust.matches("OutputResource::Buffer {").count(), 2);
        assert_eq!(rust.matches("OutputResource::Texture(Texture::clone(").count(), 1);
    }
}
