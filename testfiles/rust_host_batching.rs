// Copied beside compiler-generated modules by scripts/test_rust_host_gpu.ps1.
#[allow(dead_code, non_snake_case, unused_imports, unused_variables)]
#[path = "capture_spv.rs"]
mod capture_spv;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables)]
#[path = "capture_wgsl.rs"]
mod capture_wgsl;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables)]
#[path = "epilogues_spv.rs"]
mod epilogues_spv;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables)]
#[path = "epilogues_wgsl.rs"]
mod epilogues_wgsl;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables)]
#[path = "filter_command_spv.rs"]
mod filter_command_spv;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables)]
#[path = "filter_command_wgsl.rs"]
mod filter_command_wgsl;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables)]
#[path = "filter_post_spv.rs"]
mod filter_post_spv;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables)]
#[path = "filter_post_wgsl.rs"]
mod filter_post_wgsl;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables)]
#[path = "filter_spv.rs"]
mod filter_spv;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables)]
#[path = "filter_wgsl.rs"]
mod filter_wgsl;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables)]
#[path = "setup_spv.rs"]
mod setup_spv;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables)]
#[path = "setup_wgsl.rs"]
mod setup_wgsl;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables, unused_mut)]
#[path = "batching_spv.rs"]
mod spv;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables, unused_mut)]
#[path = "batching_wgsl.rs"]
mod wgsl;

#[cfg(test)]
mod tests {
    use super::{
        capture_spv, capture_wgsl, epilogues_spv, epilogues_wgsl, filter_command_spv, filter_command_wgsl,
        filter_post_spv, filter_post_wgsl, filter_spv, filter_wgsl, setup_spv, setup_wgsl, spv, wgsl,
    };
    use wgpu::util::DeviceExt;

    fn input(device: &wgpu::Device, values: &[i32]) -> wgpu::Buffer {
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
        })
    }

    fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<i32> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        queue.submit(Some(encoder.finish()));
        let (sender, receiver) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| sender.send(result).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let bytes = staging.slice(..).get_mapped_range();
        bytes.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect()
    }

    macro_rules! buffer {
        ($module:ident, $value:expr) => {
            buffer!($module, $value, 0)
        };
        ($module:ident, $value:expr, $index:expr) => {
            match &$value.values[$index].resource {
                $module::output::OutputResource::Buffer { buffer, .. } => buffer,
                other => panic!("expected buffer, got {other:?}"),
            }
        };
    }

    #[test]
    fn recorded_uploads_scratch_reuse_gpu_captures_and_readback_boundaries() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::PUSH_CONSTANTS,
            required_limits: wgpu::Limits {
                max_push_constant_size: 128,
                ..Default::default()
            },
            ..Default::default()
        }))
        .unwrap();
        eprintln!("GPU: {:?}", adapter.get_info());
        let mut setup_spv_context = setup_spv::HostContext::new(&device).unwrap();
        let mut setup_wgsl_context = setup_wgsl::HostContext::new(&device).unwrap();
        for [w, h] in [[1.0f32, 1.0f32], [1920.0, 1080.0], [1080.0, 1920.0]] {
            let viewport = input(&device, &[w.to_bits() as i32, h.to_bits() as i32]);
            let viewport_bytes: Vec<_> = [w, h].into_iter().flat_map(f32::to_le_bytes).collect();
            let pixel = [1.0 / w, 1.0 / h];
            let tangent = [0.17632698 * (w / h), 0.17632698];
            let multiplier = [tangent[0] * 2.0, tangent[1] * -2.0];
            let offset = [-tangent[0], tangent[1]];
            let depth = [(1000.0 * 0.1) / (1000.0 - 0.1), 1000.0 / (1000.0 - 0.1)];
            for n in [1, 63, 64, 65, 257] {
                let values: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 - 7.0).collect();
                let xs = input(
                    &device,
                    &values.iter().map(|v| v.to_bits() as i32).collect::<Vec<_>>(),
                );
                let expected: Vec<f32> = values
                    .iter()
                    .flat_map(|&x| {
                        (0..2).map(move |i| {
                            pixel[i] * x
                                + depth[i]
                                + tangent[i]
                                + multiplier[i]
                                + offset[i]
                                + multiplier[i] * pixel[i]
                        })
                    })
                    .collect();
                let spv_output =
                    setup_spv::host_setup(&mut setup_spv_context, &queue, &xs, &viewport_bytes).unwrap();
                let wgsl_output =
                    setup_wgsl::host_setup(&mut setup_wgsl_context, &queue, &xs, &viewport).unwrap();
                for (backend, actual) in [
                    ("SPIR-V", read(&device, &queue, buffer!(setup_spv, spv_output))),
                    ("WGSL", read(&device, &queue, buffer!(setup_wgsl, wgsl_output))),
                ] {
                    assert_eq!(actual.len(), expected.len());
                    for (bits, &expected) in actual.into_iter().zip(&expected) {
                        let actual = f32::from_bits(bits as u32);
                        assert!(
                            (actual - expected).abs() <= 2.0e-6 * expected.abs().max(1.0),
                            "{backend} setup {w}x{h} n={n}: {actual} != {expected}"
                        );
                    }
                }
            }
        }
        let xs = input(&device, &[5, 2, 9]);
        let mut spv = spv::HostContext::new(&device).unwrap();
        // Every call reuses the same scratch slot. Uploads must appear between
        // calls, after the recorded clear and before that call's dispatch.
        for _ in 0..2 {
            let mut encoder = device.create_command_encoder(&Default::default());
            let outputs: Vec<_> = [1i32, 2, 3]
                .into_iter()
                .map(|bias| spv::encode_affine(&mut spv, &mut encoder, &bias.to_le_bytes(), &xs).unwrap())
                .collect();
            queue.submit(Some(encoder.finish()));
            for (bias, output) in [1, 2, 3].into_iter().zip(outputs) {
                assert_eq!(
                    read(&device, &queue, buffer!(spv, output)),
                    vec![5 + bias * bias, 2 + bias * bias, 9 + bias * bias]
                );
            }
        }
        let ones = input(&device, &[1; 137]);
        let output = spv::host_captured(&mut spv, &queue, &ones).unwrap();
        assert_eq!(read(&device, &queue, buffer!(spv, output)), vec![18770; 137]);
        let output = spv::host_indexed(&mut spv, &queue, &xs).unwrap();
        assert_eq!(read(&device, &queue, buffer!(spv, output)), vec![10, 7, 14]);
        let mut wgsl = wgsl::HostContext::new(&device).unwrap();
        let output = wgsl::host_captured(&mut wgsl, &queue, &ones).unwrap();
        assert_eq!(read(&device, &queue, buffer!(wgsl, output)), vec![18770; 137]);
        let index = input(&device, &[1]);
        let output = spv::host_dynamic_index(&mut spv, &queue, &xs, &1i32.to_le_bytes()).unwrap();
        assert_eq!(read(&device, &queue, buffer!(spv, output)), vec![10, 7, 14]);
        let output = wgsl::host_dynamic_index(&mut wgsl, &queue, &xs, &index).unwrap();
        assert_eq!(read(&device, &queue, buffer!(wgsl, output)), vec![10, 7, 14]);
        for n in [0i32, 4] {
            let scalar = input(&device, &[n]);
            let expected: Vec<_> = [5, 2, 9].into_iter().map(|x| x + 5 + 3 * (0..n).sum::<i32>()).collect();
            let output = spv::host_grouped(&mut spv, &queue, &xs, &n.to_le_bytes()).unwrap();
            assert_eq!(read(&device, &queue, buffer!(spv, output)), expected);
            let output = wgsl::host_grouped(&mut wgsl, &queue, &xs, &scalar).unwrap();
            assert_eq!(read(&device, &queue, buffer!(wgsl, output)), expected);
        }
        // WGSL scalar inputs are GPU uniforms. These entries exercise the
        // submitting API's readback boundary and subsequent recorded upload.
        let n = input(&device, &[4]);
        let output = wgsl::host_dynamic(&mut wgsl, &queue, &n).unwrap();
        assert_eq!(
            read(&device, &queue, buffer!(wgsl, output)),
            (0..11).collect::<Vec<_>>()
        );
        for bias in [2, 3] {
            let scalar = input(&device, &[bias]);
            let output = wgsl::host_affine(&mut wgsl, &queue, &xs, &scalar).unwrap();
            assert_eq!(
                read(&device, &queue, buffer!(wgsl, output)),
                vec![5 + bias * bias, 2 + bias * bias, 9 + bias * bias]
            );
        }
        let mut attached_spv = epilogues_spv::HostContext::new(&device).unwrap();
        let mut attached_wgsl = epilogues_wgsl::HostContext::new(&device).unwrap();
        for seed in [-7i32, 0, 5] {
            for n in [0i32, 1, 7] {
                for size in [1, 3, 65] {
                    let values: Vec<_> = (0..size).map(|i| i - 2).collect();
                    let xs = input(&device, &values);
                    let parameters = input(&device, &[seed, n]);
                    let a = seed + n * (n - 1) / 2;
                    let b = a + n * (n - 1);
                    let expected = vec![
                        values.iter().map(|x| x + a).collect::<Vec<_>>(),
                        vec![0, a, a * 2],
                        vec![a, b],
                    ];
                    let output = epilogues_spv::host_attached(
                        &mut attached_spv,
                        &queue,
                        &seed.to_le_bytes(),
                        &n.to_le_bytes(),
                        &xs,
                    )
                    .unwrap();
                    let actual: Vec<_> =
                        (0..3).map(|i| read(&device, &queue, buffer!(epilogues_spv, output, i))).collect();
                    assert_eq!(actual, expected);
                    let output =
                        epilogues_wgsl::host_attached(&mut attached_wgsl, &queue, &xs, &parameters)
                            .unwrap();
                    let actual: Vec<_> =
                        (0..3).map(|i| read(&device, &queue, buffer!(epilogues_wgsl, output, i))).collect();
                    assert_eq!(actual, expected);
                }
            }
        }
        let mut capture_spv = capture_spv::HostContext::new(&device).unwrap();
        let mut capture_wgsl = capture_wgsl::HostContext::new(&device).unwrap();
        for pattern in 0..4 {
            let events: Vec<_> = (0..32)
                .map(|i| {
                    if pattern == 0 || (pattern == 2 && i % 2 == 0) || (pattern == 3 && i % 7 == 0) {
                        i + 1
                    } else {
                        0
                    }
                })
                .collect();
            let mut state = 0;
            let mut last = [0; 4];
            for (i, &event) in events.iter().enumerate() {
                if event > 0 {
                    state += event;
                    last[i % 4] = event;
                }
            }
            let expected = vec![
                (0..64).map(|i| last[i % 4]).collect::<Vec<_>>(),
                (0..128).map(|i| i + state).collect(),
                vec![state],
            ];
            let events = input(&device, &events);
            let output = capture_spv::host_repro(&mut capture_spv, &queue, &events).unwrap();
            let actual: Vec<_> =
                (0..3).map(|i| read(&device, &queue, buffer!(capture_spv, output, i))).collect();
            assert_eq!(actual, expected);
            let output = capture_wgsl::host_repro(&mut capture_wgsl, &queue, &events).unwrap();
            let actual: Vec<_> =
                (0..3).map(|i| read(&device, &queue, buffer!(capture_wgsl, output, i))).collect();
            assert_eq!(actual, expected);
        }
        // Exercise tile boundaries, empty input, and tinyporto-sized domains.
        // Verify stable compaction, post-maps, and lane-zero count publication.
        let mut spv = filter_spv::HostContext::new(&device).unwrap();
        let mut wgsl = filter_wgsl::HostContext::new(&device).unwrap();
        let mut post_spv = filter_post_spv::HostContext::new(&device).unwrap();
        let mut post_wgsl = filter_post_wgsl::HostContext::new(&device).unwrap();
        let mut command_spv = filter_command_spv::HostContext::new(&device).unwrap();
        let mut command_wgsl = filter_command_wgsl::HostContext::new(&device).unwrap();
        for n in [0i32, 1, 63, 64, 65, 255, 256, 257, 1600, 4096, 63592] {
            for pattern in 0..4 {
                let values: Vec<_> = (0..n.max(1))
                    .map(|i| {
                        let keep = match pattern {
                            0 => true,
                            1 => false,
                            2 => i % 2 == 0,
                            _ => (i * 17 + 11) % 101 < 7,
                        };
                        if keep {
                            i * 7 + 3
                        } else {
                            -(i + 1)
                        }
                    })
                    .collect();
                let expected: Vec<_> = values[..n as usize].iter().copied().filter(|x| *x > 0).collect();
                let xs = input(&device, &values);
                let scalar = input(&device, &[n]);
                let output = filter_spv::host_filtered(&mut spv, &queue, &n.to_le_bytes(), &xs).unwrap();
                assert_eq!(
                    read(&device, &queue, buffer!(filter_spv, output, 0)),
                    vec![expected.len() as i32]
                );
                assert_eq!(
                    &read(&device, &queue, buffer!(filter_spv, output, 1))[..expected.len()],
                    expected
                );
                // WGSL publishes a uniform parameter block for each filter stage.
                let output = filter_wgsl::host_filtered(&mut wgsl, &queue, &xs, &scalar, &scalar).unwrap();
                assert_eq!(
                    read(&device, &queue, buffer!(filter_wgsl, output, 0)),
                    vec![expected.len() as i32]
                );
                assert_eq!(
                    &read(&device, &queue, buffer!(filter_wgsl, output, 1))[..expected.len()],
                    expected
                );
                let records: Vec<_> = expected.iter().flat_map(|&value| [value, value * 3 + n]).collect();
                let output =
                    filter_post_spv::host_post_mapped(&mut post_spv, &queue, &n.to_le_bytes(), &xs)
                        .unwrap();
                assert_eq!(
                    read(&device, &queue, buffer!(filter_post_spv, output, 0)),
                    vec![expected.len() as i32]
                );
                assert_eq!(
                    &read(&device, &queue, buffer!(filter_post_spv, output, 1))[..records.len()],
                    records,
                    "SPIR-V n={n}, pattern={pattern}"
                );
                let output =
                    filter_post_wgsl::host_post_mapped(&mut post_wgsl, &queue, &xs, &scalar).unwrap();
                assert_eq!(
                    read(&device, &queue, buffer!(filter_post_wgsl, output, 0)),
                    vec![expected.len() as i32]
                );
                assert_eq!(
                    &read(&device, &queue, buffer!(filter_post_wgsl, output, 1))[..records.len()],
                    records,
                    "WGSL n={n}, pattern={pattern}"
                );
                let output =
                    filter_command_spv::host_command(&mut command_spv, &queue, &n.to_le_bytes(), &xs)
                        .unwrap();
                assert_eq!(
                    read(&device, &queue, buffer!(filter_command_spv, output, 0)),
                    [36, expected.len() as i32, 0, 0]
                );
                assert_eq!(
                    &read(&device, &queue, buffer!(filter_command_spv, output, 1))[..expected.len()],
                    expected
                );
                let output =
                    filter_command_wgsl::host_command(&mut command_wgsl, &queue, &xs, &scalar).unwrap();
                assert_eq!(
                    read(&device, &queue, buffer!(filter_command_wgsl, output, 0)),
                    [36, expected.len() as i32, 0, 0]
                );
                assert_eq!(
                    &read(&device, &queue, buffer!(filter_command_wgsl, output, 1))[..expected.len()],
                    expected
                );
            }
        }
    }
}
