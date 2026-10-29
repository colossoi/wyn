use crate::host::arithmetic::{add, ceiling, dimension, floor, multiply, signed_size, size, subtract};
use crate::host::{Binding, FrameGraph, Pipeline, Program, ResultLayout, ShaderFormat};

fn result_program(fields: Vec<(&str, crate::types::Type)>, kind: crate::host::ResultKind) -> Program {
    use crate::host::{
        Access, BufferLen, BufferUsage, ComputePipeline, ComputeStage, DispatchSize, ModuleInterface,
        SourceResultBinding, StageBindingUses,
    };
    let bindings = fields
        .iter()
        .enumerate()
        .map(|(index, (name, _))| Binding::StorageBuffer {
            set: 0,
            binding: index as u32,
            access: Access::WriteOnly,
            usage: BufferUsage::Output,
            name: (*name).into(),
            resource: None,
            length: Some(BufferLen::Fixed { bytes: 64 }),
            members: vec![],
        })
        .collect();
    let pipelines = vec![Pipeline::Compute(ComputePipeline {
        bindings,
        default_total_threads: None,
        stages: vec![ComputeStage {
            dependencies: vec![],
            entry_point: "main".into(),
            owner: "main".into(),
            workgroup_size: (1, 1, 1),
            dispatch_size: DispatchSize::Fixed {
                x: 1,
                y: 1,
                z: 1,
                explicit: false,
            },
            uses: StageBindingUses {
                reads: vec![],
                writes: (0..fields.len()).collect(),
            },
        }],
    })];
    let frame_graph = FrameGraph::from_selected_pipelines(&pipelines, &[]).unwrap();
    Program::new(ModuleInterface {
        source_results: fields
            .into_iter()
            .enumerate()
            .map(|(index, (name, ty))| SourceResultBinding {
                entry: "main".into(),
                name: name.into(),
                kind,
                layout: crate::interface::results::result_layout(&ty),
                shape: if matches!(
                    crate::interface::results::result_layout(&ty),
                    ResultLayout::Array { .. }
                ) {
                    vec![crate::host::Expr::Integer(4)]
                } else {
                    vec![]
                },
                result: index,
                pipeline_index: 0,
                set: 0,
                binding: index as u32,
            })
            .collect(),
        pipelines,
        frame_graph,
        ..Default::default()
    })
    .unwrap()
}

fn runtime_array(element: crate::types::Type) -> crate::types::Type {
    crate::types::make_array1(
        element,
        crate::types::array_variant_composite(),
        crate::types::Type::Variable(0),
        crate::types::no_buffer(),
    )
}

fn graphics_program() -> Program {
    use crate::host::*;
    let bindings = vec![
        Binding::StorageBuffer {
            set: 0,
            binding: 0,
            access: Access::ReadOnly,
            usage: BufferUsage::Input,
            name: "positions".into(),
            resource: None,
            length: None,
            members: vec![],
        },
        Binding::Texture {
            set: 0,
            binding: 1,
            name: "tex".into(),
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
            backing: None,
            resource: None,
        },
        Binding::Sampler {
            set: 0,
            binding: 2,
            name: "samp".into(),
            binding_type: SamplerBindingType::Filtering,
        },
    ];
    let pipelines = vec![Pipeline::Graphics(GraphicsPipeline {
        source_operation: None,
        invocation: Default::default(),
        bindings,
        stages: [ShaderStage::Vertex, ShaderStage::Fragment]
            .into_iter()
            .enumerate()
            .map(|(index, stage)| GraphicsStage {
                entry_point: format!("stage_{index}"),
                owner: "main".into(),
                stage,
                uses: StageBindingUses {
                    reads: vec![0, 1, 2],
                    writes: vec![],
                },
            })
            .collect(),
        vertex_inputs: vec![],
        fragment_outputs: vec![FragmentOutput {
            location: 0,
            name: "screen".into(),
        }],
    })];
    let frame_graph = FrameGraph::from_selected_pipelines(&pipelines, &[]).unwrap();
    Program::new(ModuleInterface {
        pipelines,
        frame_graph,
        ..Default::default()
    })
    .unwrap()
}

fn map_program(input: &str, alias: bool, push_constant: bool) -> Program {
    use crate::host::*;
    let mut bindings = vec![Binding::StorageBuffer {
        set: 0,
        binding: 0,
        access: Access::ReadOnly,
        usage: BufferUsage::Input,
        name: input.into(),
        resource: None,
        length: None,
        members: vec![],
    }];
    if !alias {
        bindings.push(Binding::StorageBuffer {
            set: 0,
            binding: 1,
            access: Access::WriteOnly,
            usage: BufferUsage::Output,
            name: "main_output".into(),
            resource: None,
            members: vec![],
            length: Some(BufferLen::LikeInput {
                set: 0,
                binding: 0,
                elem_bytes: 4,
                src_elem_bytes: 4,
            }),
        });
    }
    if push_constant {
        bindings.push(Binding::PushConstant {
            offset: 0,
            size: 4,
            name: "bias".into(),
        });
    }
    let output = u32::from(!alias);
    let elements = Expr::BufferLength {
        source: ScalarSource::Binding { set: 0, binding: 0 },
        stride: 4,
    };
    let pipelines = vec![Pipeline::Compute(ComputePipeline {
        stages: vec![ComputeStage {
            dependencies: vec![],
            entry_point: "main".into(),
            owner: "main".into(),
            workgroup_size: (64, 1, 1),
            dispatch_size: DispatchSize::linear(elements.clone(), 64).unwrap(),
            uses: StageBindingUses {
                reads: if push_constant { vec![0, bindings.len() - 1] } else { vec![0] },
                writes: if alias { vec![] } else { vec![1] },
            },
        }],
        bindings,
        default_total_threads: None,
    })];
    let frame_graph = FrameGraph::from_selected_pipelines(&pipelines, &[]).unwrap();
    Program::new(ModuleInterface {
        pipelines,
        frame_graph,
        source_results: vec![SourceResultBinding {
            entry: "main".into(),
            name: "main".into(),
            kind: ResultKind::Value,
            layout: crate::interface::results::result_layout(&runtime_array(crate::types::i32())),
            shape: vec![elements],
            result: 0,
            pipeline_index: 0,
            set: 0,
            binding: output,
        }],
        ..Default::default()
    })
    .unwrap()
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
fn both_host_outputs_include_buffer_capacity_arithmetic() {
    let program = map_program("xs", false, false);
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
fn record_output_descriptors_preserve_authored_field_names() {
    let program = result_program(
        vec![("count", crate::types::i32()), ("gain", crate::types::f32())],
        crate::host::ResultKind::RecordField,
    );
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
    let program = result_program(
        vec![(
            "positions",
            runtime_array(crate::types::vec(3, crate::types::f32())),
        )],
        crate::host::ResultKind::Value,
    );
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
    let program = result_program(
        vec![
            ("result_0", crate::types::i32()),
            (
                "result_1",
                crate::types::Type::Constructed(crate::types::TypeName::UInt(32), vec![]),
            ),
            ("result_2", crate::types::f32()),
        ],
        crate::host::ResultKind::TupleField,
    );
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
    let program = result_program(
        vec![("value", crate::types::i32())],
        crate::host::ResultKind::RecordField,
    );
    let rust = program.to_rust_wgpu("single.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("kind: ResultKind::RecordField"), "{rust}");
    assert!(rust.contains("name: \"value\""));
}

#[test]
fn output_descriptors_preserve_nested_record_offsets() {
    let program = result_program(
        vec![(
            "particles",
            runtime_array(crate::types::record(vec![
                ("position".into(), crate::types::vec(3, crate::types::f32())),
                ("weight".into(), crate::types::f32()),
            ])),
        )],
        crate::host::ResultKind::Value,
    );
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
    let program = result_program(
        vec![(
            "main",
            crate::types::Type::Constructed(crate::types::TypeName::Float(16), vec![]),
        )],
        crate::host::ResultKind::Value,
    );
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
fn rust_buffer_arguments_keep_source_names_in_bindings_sizes_and_results() {
    let program = map_program("buffer", false, false);
    let rust = program.to_rust_wgpu("map.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("buffer: &Buffer"), "{rust}");
    assert!(rust.contains("buffer.size()"));
    assert!(rust.contains("buffer: &buffer,"));
    assert!(rust.contains("size: wgpu::BufferSize::new(buffer.size())"));

    let program = map_program("samples", true, false);
    let rust = program.to_rust_wgpu("echo.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("samples: &Buffer"));
    assert!(rust.contains("Buffer::clone(&samples)"));
}

#[test]
fn rust_generated_result_preserves_an_aliased_source_input() {
    let mut interface = map_program("result_field_1", false, false).interface;
    let mut alias = interface.source_results[0].clone();
    alias.name = "result_0".into();
    alias.binding = 0;
    alias.kind = crate::host::ResultKind::TupleField;
    interface.source_results[0].result = 1;
    interface.source_results[0].name = "result_1".into();
    interface.source_results[0].kind = crate::host::ResultKind::TupleField;
    interface.source_results.insert(0, alias);
    let program = Program::new(interface).unwrap();
    let entry = &program.entries[0];
    assert!(entry.inputs.contains(&entry.results[0]));
    assert!(!entry.inputs.contains(&entry.results[1]));
    let rust = program.to_rust_wgpu("collision.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("result_field_1: &Buffer"));
    assert!(rust.contains("Buffer::clone(&result_field_1)"));
    assert!(!rust.contains("result_field_1_2: &Buffer"));
}

#[test]
fn rust_graphics_arguments_keep_buffer_texture_and_sampler_source_names() {
    let program = graphics_program();
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
    let mut program = map_program("input", false, false);
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
fn whl_paths_escape_lisp_strings() {
    let program = result_program(
        vec![("main", crate::types::i32())],
        crate::host::ResultKind::Value,
    );
    let whl = program.to_whl("a\\b\"c.wgsl", ShaderFormat::Wgsl).unwrap();
    check_whl(&whl);
    assert!(whl.contains("a\\\\b\\\"c.wgsl"));
}

#[test]
fn rust_spirv_embeds_binary_and_binds_compute_push_constants() {
    let program = map_program("xs", false, true);
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
    let program = graphics_program();
    let rust = program.to_rust_wgpu("conway.spv", ShaderFormat::Spirv).unwrap();
    assert!(rust.contains("ShaderStages::VERTEX | ShaderStages::FRAGMENT"));
    assert!(rust.contains("create_render_pipeline"));
    assert!(rust.contains("create_pipeline_layout"));
}
