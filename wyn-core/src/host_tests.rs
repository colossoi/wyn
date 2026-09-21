use crate::host::arithmetic::{
    add, ceiling, dimension, floor, modulo, multiply, signed_size, size, subtract,
};
use crate::host::{Allocation, Operation, Pipeline, Program, ResultLayout, ResultScalar, ShaderFormat};
use crate::{compile_thru_ssa, lower_ssa_to_wgsl_with_program};
use std::collections::BTreeSet;

fn compile(source: &str) -> Program {
    lower_ssa_to_wgsl_with_program(compile_thru_ssa(source).unwrap()).unwrap().program
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
    assert_eq!(entry.operations.len(), stages);
    assert_eq!(entry.results.len(), 1);
    assert!(entry.operations.iter().all(|o| matches!(o, Operation::Dispatch { .. })));
    let whl = program.to_whl("scan.wgsl", ShaderFormat::Wgsl).unwrap();
    check_whl(&whl);
    assert_eq!(whl.matches("(gpu-dispatch ").count(), stages);
    assert!(whl.contains("(gpu-alloc "));
    assert!(whl.contains("(gpu-free "));
    assert!(whl.contains(":source-name \"main\""));
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
    assert!(entry
        .allocations
        .iter()
        .any(|a| matches!(a,Allocation::Buffer{bytes,..} if bytes.to_whl().contains("gpu-read-scalar"))));
    let whl = program.to_whl("sizes.wgsl", ShaderFormat::Wgsl).unwrap();
    check_whl(&whl);
    assert!(whl.contains("(i32-add "), "{whl}");
    assert!(whl.contains("(i32-mul "), "{whl}");
    assert!(!whl.contains("4294967296"));
    let rust = program.to_rust_wgpu("sizes.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains(".wrapping_add("));
    assert!(rust.contains(".wrapping_mul("));
    assert!(rust.contains("n: &Buffer"));
    let compact: String = rust.split_whitespace().collect();
    assert!(compact.contains("support::read_i32(device,queue,&n,"));
    assert!(!rust.contains("read_host_scalar"));
}

#[test]
fn scalar_output_descriptors_preserve_names_types_and_byte_ranges() {
    for (ty, literal, scalar) in [
        ("i32", "-7", ResultScalar::I32),
        ("u32", "4294967295u32", ResultScalar::U32),
        ("f32", "2.5", ResultScalar::F32),
    ] {
        let program = compile(&format!("entry result() {ty} = {literal}"));
        assert_eq!(program.interface.source_results[0].name, "result");
        assert_eq!(
            program.interface.source_results[0].layout,
            ResultLayout::Scalar(scalar)
        );
        let rust = program.to_rust_wgpu("result.wgsl", ShaderFormat::Wgsl).unwrap();
        assert!(rust.contains("-> Result<OutputDescriptor, HostError>"), "{rust}");
        assert!(rust.contains("name: \"result\""));
        assert!(rust.contains(&format!("ResultLayout::Scalar(ResultScalar::{scalar:?})")));
        assert!(rust.contains("size: 4u64"));
        assert!(rust.contains("Buffer::clone("));
        assert!(rust.contains("pub mod output"));
        for operation in [
            "pub fn read_",
            "map_async",
            "copy_buffer_to_buffer",
            "from_le_bytes",
        ] {
            assert!(
                !rust.contains(operation),
                "output descriptor contains {operation}"
            );
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
fn array_output_descriptors_publish_padding_and_unknown_view_ranges() {
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
    assert!(rust.contains("range: BufferRange::CallerProvided"), "{rust}");
    assert!(rust.contains("layout: ResultLayout::Array"));
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
    for operation in [
        "pub fn read_",
        "map_async",
        "copy_buffer_to_buffer",
        "from_le_bytes",
        ".poll(",
    ] {
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
    for kernel in ssa.global_context.physical_kernels.kernels() {
        let entry = ssa.entry_points.iter().find(|e| e.id == kernel.entry).unwrap();
        assert_eq!(kernel.entry_point, entry.name);
    }
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
    assert!(rust.contains("resource: buffer.as_entire_binding()"));

    let program = compile("entry echo(samples:[]i32) []i32 = samples");
    let rust = program.to_rust_wgpu("echo.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("samples: &Buffer"));
    assert!(rust.contains("Buffer::clone(&samples)"));
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
    assert!(rust.contains("positions.as_entire_binding()"));
    let compact: String = rust.split_whitespace().collect();
    assert!(compact.contains("tex.create_view("));
    assert!(rust.contains("BindingResource::Sampler(&samp)"));
    assert!(compact.contains("screen.create_view("));
    assert!(rust.contains("Texture::clone(&screen)"));
}

#[test]
fn rust_argument_names_handle_keywords_and_generated_binding_collisions() {
    for (source, parameter) in [
        ("async", "r#async"),
        ("crate", "crate_2"),
        ("device", "device_2"),
        ("queue", "queue_2"),
        ("shader", "shader_2"),
        ("groups", "groups_2"),
        ("pipeline", "pipeline_2"),
        ("group_0", "group_0_2"),
        ("resource_1", "resource_1_2"),
        ("size", "size_2"),
    ] {
        let program = compile(&format!(
            "entry main({source}:[]i32) []i32 = map(|x:i32|x+1,{source})"
        ));
        let rust = program.to_rust_wgpu("names.wgsl", ShaderFormat::Wgsl).unwrap();
        assert!(rust.contains(&format!("{parameter}: &Buffer")), "{rust}");
        assert!(rust.contains(&format!("{parameter}.size()")), "{rust}");
        assert!(
            rust.contains(&format!("resource: {parameter}.as_entire_binding()")),
            "{rust}"
        );
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
    for (a, b, down, up, remainder) in [
        (5, 2, 2, 3, 1),
        (-5, 2, -3, -2, 1),
        (5, -2, -3, -2, -1),
        (-5, -2, 2, 3, -1),
    ] {
        assert_eq!(floor(a, b).unwrap(), down);
        assert_eq!(ceiling(a, b).unwrap(), up);
        assert_eq!(modulo(a, b).unwrap(), remainder);
    }
    assert_eq!(ceiling(i64::MAX, 2).unwrap(), (i64::MAX / 2) + 1);
    assert_eq!(modulo(i64::MIN, i64::MAX).unwrap(), i64::MAX - 1);
    for (a, b) in [(1, 0), (i64::MIN, -1)] {
        assert!(floor(a, b).is_err());
        assert!(ceiling(a, b).is_err());
        assert!(modulo(a, b).is_err());
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
fn whl_paths_escape_lisp_strings_and_rust_requires_wgsl() {
    let program = compile("entry main() i32 = 1");
    let whl = program.to_whl("a\\b\"c.wgsl", ShaderFormat::Wgsl).unwrap();
    check_whl(&whl);
    assert!(whl.contains("a\\\\b\\\"c.wgsl"));
    assert!(program.to_rust_wgpu("shader.spv", ShaderFormat::Spirv).is_err());
}
