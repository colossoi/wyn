use super::*;
use wyn_pipeline_descriptor::{BufferLen, UniformMember};

#[test]
fn spirv_passthrough_is_only_enabled_for_vulkan() {
    assert!(supports_spirv_passthrough(Backend::Vulkan));
    assert!(!supports_spirv_passthrough(Backend::Metal));
    assert!(!supports_spirv_passthrough(Backend::Dx12));
    assert!(!supports_spirv_passthrough(Backend::Gl));
    assert!(!supports_spirv_passthrough(Backend::BrowserWebGpu));
}

#[test]
fn packs_wgsl_parameter_blocks_at_descriptor_offsets() {
    let bindings = vec![Binding::StorageBuffer {
        set: 1,
        binding: 0,
        access: Access::ReadOnly,
        usage: BufferUsage::Input,
        name: "params".to_string(),
        resource: None,
        length: Some(BufferLen::Fixed { bytes: 32 }),
        members: vec![
            UniformMember {
                name: "count".to_string(),
                offset: 0,
                size: 4,
            },
            UniformMember {
                name: "direction".to_string(),
                offset: 16,
                size: 12,
            },
        ],
    }];
    let values = vec![
        PushConstantSpec {
            name: "count".to_string(),
            offset: 0,
            data: 130u32.to_le_bytes().to_vec(),
        },
        PushConstantSpec {
            name: "direction".to_string(),
            offset: 0,
            data: [1.0f32, 2.0, 3.0].into_iter().flat_map(f32::to_le_bytes).collect(),
        },
    ];

    let blocks = build_parameter_block_bytes(&bindings, &values, false).unwrap();
    let block = &blocks[&(1, 0)];
    assert_eq!(&block[0..4], &130u32.to_le_bytes());
    assert!(block[4..16].iter().all(|byte| *byte == 0));
    assert_eq!(&block[16..28], values[1].data.as_slice());
    assert!(block[28..32].iter().all(|byte| *byte == 0));

    let dispatch = DispatchSize::DerivedFrom {
        len: DispatchLen::StorageBuffer {
            set: 1,
            binding: 0,
            offset: 0,
        },
        workgroup_size: 64,
    };
    assert_eq!(
        resolve_dispatch_size_with_parameters(&dispatch, &StorageBuffers::new(), &[], &blocks).unwrap(),
        (3, 1, 1)
    );
}

#[test]
fn host_expression_dispatch_uses_uniform_bytes_and_checks_errors() {
    use wyn_pipeline_descriptor::{HostBinary, HostExpression, HostScalar};
    let dim = |offset| HostExpression::Convert {
        to: HostScalar::I32,
        value: Box::new(HostExpression::Uniform {
            name: format!("dim_{offset}"),
            set: 2,
            binding: 3,
            offset,
            scalar: HostScalar::F32,
        }),
    };
    let count = HostExpression::Binary {
        op: HostBinary::Multiply,
        left: Box::new(dim(16)),
        right: Box::new(dim(20)),
    };
    let dispatch = DispatchSize::DerivedFrom {
        len: DispatchLen::HostExpression { count },
        workgroup_size: 64,
    };
    let dispatch: DispatchSize = serde_json::from_str(&serde_json::to_string(&dispatch).unwrap()).unwrap();
    let buffers = StorageBuffers::new();
    let mut snapshot = ParameterBlockBytes::new();
    assert!(resolve_dispatch_size_with_parameters(&dispatch, &buffers, &[], &snapshot).is_err());
    for (w, h, expected) in [
        (1280.0f32, 800.0f32, 16000),
        (160.0, 100.0, 250),
        (17.9, 9.8, 3),
        (0.0, 32.0, 0),
    ] {
        let mut bytes = vec![0; 24];
        bytes[16..20].copy_from_slice(&w.to_le_bytes());
        bytes[20..24].copy_from_slice(&h.to_le_bytes());
        snapshot.insert((2, 3), bytes);
        assert_eq!(
            resolve_dispatch_size_with_parameters(&dispatch, &buffers, &[], &snapshot).unwrap(),
            (expected, 1, 1)
        );
    }
    for (w, h) in [
        (-1.0f32, 2.0f32),
        (f32::NAN, 2.0),
        (f32::INFINITY, 2.0),
        (65536.0, 65536.0),
    ] {
        let bytes = snapshot.get_mut(&(2, 3)).unwrap();
        bytes[16..20].copy_from_slice(&w.to_le_bytes());
        bytes[20..24].copy_from_slice(&h.to_le_bytes());
        assert!(resolve_dispatch_size_with_parameters(&dispatch, &buffers, &[], &snapshot).is_err());
    }
    snapshot.insert((2, 3), vec![0; 19]);
    assert!(resolve_dispatch_size_with_parameters(&dispatch, &buffers, &[], &snapshot).is_err());
}

// Run with `cargo build -p wyn`, then `cargo test --manifest-path
// extra/viz/Cargo.toml uniform_map_gpu_coverage -- --ignored --nocapture`.
#[test]
#[ignore = "requires a GPU adapter and the current target/debug/wyn compiler"]
fn uniform_map_gpu_coverage() {
    use wyn_pipeline_descriptor::BufferLen;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let compiler = root.join("target/debug/wyn");
    let temp = std::env::temp_dir().join(format!("wyn-uniform-dispatch-{}", std::process::id()));
    std::fs::create_dir_all(&temp).unwrap();
    let source =
        std::fs::read_to_string(root.join("testfiles/regressions/uniform_output_size.wyn")).unwrap();
    let ctx = pollster::block_on(GpuContext::request(DeviceRequest::default())).unwrap();
    let (device, queue) = ctx.into_device_queue();
    for coarse in [false, true] {
        let source = if coarse {
            source.replace(
                "width * height",
                "((width + 7i32) / 8i32) * ((height + 7i32) / 8i32)",
            )
        } else {
            source.clone()
        };
        let input = temp.join("source.wyn");
        std::fs::write(&input, source).unwrap();
        for target in ["wgsl", "spirv"] {
            let shader = temp.join(if target == "wgsl" { "shader.wgsl" } else { "shader.spv" });
            let result = std::process::Command::new(&compiler)
                .args(["build", "--graphics", "--max-warnings", "0", "--target", target])
                .arg(&input)
                .arg("-o")
                .arg(&shader)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            let descriptor: PipelineDescriptor =
                serde_json::from_slice(&std::fs::read(shader.with_extension("json")).unwrap()).unwrap();
            let cp = descriptor
                .pipelines
                .iter()
                .find_map(|p| match p {
                    Pipeline::Compute(cp) => Some(cp),
                    _ => None,
                })
                .unwrap();
            let module = crate::spirv::load_shader_module(&device, &shader).unwrap();
            for (width, height) in [(1280u32, 800u32), (17, 9), (1, 1), (0, 32)] {
                let expected = if coarse { width.div_ceil(8) * height.div_ceil(8) } else { width * height };
                let mut snapshot = ParameterBlockBytes::new();
                let mut buffers = StorageBuffers::new();
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("rendered depth"),
                    size: wgpu::Extent3d {
                        width: width.max(1),
                        height: height.max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let view = texture.create_view(&Default::default());
                for binding in &cp.bindings {
                    let (set, slot) = binding.slot().unwrap();
                    assert_eq!(set, 0);
                    if let Binding::Uniform { size, .. } = binding {
                        let mut bytes = vec![0; (*size).max(16) as usize];
                        bytes[..4].copy_from_slice(&(width as f32).to_le_bytes());
                        bytes[4..8].copy_from_slice(&(height as f32).to_le_bytes());
                        let buffer = device.create_buffer(&BufferDescriptor {
                            label: None,
                            size: bytes.len() as u64,
                            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        queue.write_buffer(&buffer, 0, &bytes);
                        buffers.insert((set, slot), (buffer, bytes.len() as u64));
                        snapshot.insert((set, slot), bytes);
                    }
                }
                buffers.extend(
                    create_binding_buffers(
                        &device,
                        &queue,
                        &cp.bindings,
                        &HashMap::new(),
                        Some(&cp.stages.last().unwrap().dispatch_size),
                        &[],
                        &snapshot,
                        false,
                    )
                    .unwrap(),
                );
                let mut output = None;
                for binding in &cp.bindings {
                    if let Binding::StorageBuffer {
                        set,
                        binding,
                        length: Some(length),
                        usage: BufferUsage::Output,
                        ..
                    } = binding
                    {
                        assert!(matches!(length, BufferLen::HostExpression { .. }));
                        let (buffer, bytes) = &buffers[&(*set, *binding)];
                        // Every logical element must overwrite poison; an empty domain leaves it untouched.
                        queue.write_buffer(buffer, 0, &vec![0xff; *bytes as usize]);
                        output = Some((*set, *binding));
                    }
                }
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.5,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                }
                for stage in &cp.stages {
                    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: None,
                        layout: None,
                        module: &module,
                        entry_point: Some(&stage.entry_point),
                        compilation_options: Default::default(),
                        cache: None,
                    });
                    let entries: Vec<_> = cp
                        .bindings
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| stage.reads.contains(index) || stage.writes.contains(index))
                        .map(|(_, binding)| {
                            let (set, slot) = binding.slot().unwrap();
                            wgpu::BindGroupEntry {
                                binding: slot,
                                resource: match binding {
                                    Binding::Texture { .. } => wgpu::BindingResource::TextureView(&view),
                                    _ => buffers[&(set, slot)].0.as_entire_binding(),
                                },
                            }
                        })
                        .collect();
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &pipeline.get_bind_group_layout(0),
                        entries: &entries,
                    });
                    let (x, y, z) = resolve_dispatch_size_with_parameters(
                        &stage.dispatch_size,
                        &buffers,
                        &[],
                        &snapshot,
                    )
                    .unwrap();
                    if matches!(stage.dispatch_size, DispatchSize::DerivedFrom { .. }) {
                        assert_eq!(x, expected.div_ceil(64));
                    }
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(x, y, z);
                }
                queue.submit(Some(encoder.finish()));
                let (buffer, size) = &buffers[&output.unwrap()];
                let values = readback_buffer(&device, &queue, buffer, *size).unwrap();
                if expected == 0 {
                    assert!(values[0].is_nan());
                } else {
                    assert_eq!(values.len(), expected as usize);
                    assert!(
                        values.iter().all(|v| *v == 0.5),
                        "incomplete coverage: {target}, coarse={coarse}, {width}x{height}"
                    );
                }
                eprintln!(
                    "GPU coverage passed: {target}, coarse={coarse}, {width}x{height}, {expected} elements"
                );
            }
        }
    }
    std::fs::remove_dir_all(&temp).unwrap();
}
