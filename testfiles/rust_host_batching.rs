// Copied beside compiler-generated modules by scripts/test_rust_host_gpu.ps1.
#[allow(dead_code, non_snake_case, unused_imports, unused_variables)]
#[path = "filter_spv.rs"]
mod filter_spv;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables)]
#[path = "filter_wgsl.rs"]
mod filter_wgsl;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables, unused_mut)]
#[path = "batching_spv.rs"]
mod spv;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables, unused_mut)]
#[path = "batching_wgsl.rs"]
mod wgsl;

#[cfg(test)]
mod tests {
    use super::{filter_spv, filter_wgsl, spv, wgsl};
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
        // Exercise every scan boundary, including padding lanes, empty input,
        // and changes in scratch allocation size between calls.
        let mut spv = filter_spv::HostContext::new(&device).unwrap();
        let mut wgsl = filter_wgsl::HostContext::new(&device).unwrap();
        for n in [0i32, 1, 63, 64, 65, 255, 256, 257, 4096, 39592] {
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
                            i + 1
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
                let output = filter_wgsl::host_filtered(
                    &mut wgsl, &queue, &xs, &scalar, &scalar, &scalar, &scalar, &scalar, &scalar,
                )
                .unwrap();
                assert_eq!(
                    read(&device, &queue, buffer!(filter_wgsl, output, 0)),
                    vec![expected.len() as i32]
                );
                assert_eq!(
                    &read(&device, &queue, buffer!(filter_wgsl, output, 1))[..expected.len()],
                    expected
                );
            }
        }
    }
}
