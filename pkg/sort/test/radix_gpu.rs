// Copied beside the generated Rust/WGPU module by scripts/test_radix_gpu.ps1.
#[allow(dead_code, non_snake_case, unused_imports, unused_variables, unused_mut)]
mod nested;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables, unused_mut)]
mod radix;
#[allow(dead_code, non_snake_case, unused_imports, unused_variables, unused_mut)]
mod tuples;

#[cfg(test)]
mod tests {
    use super::radix::{self, output::OutputResource, HostContext};
    use super::{nested, tuples};
    use wgpu::util::DeviceExt;

    fn input(device: &wgpu::Device, values: &[u32]) -> wgpu::Buffer {
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("radix input"),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    }

    fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, n: usize) -> Vec<u32> {
        if n == 0 {
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            return vec![];
        }
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix readback"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (n * 4) as u64);
        queue.submit(Some(encoder.finish()));
        let (sender, receiver) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| sender.send(result).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let bytes = staging.slice(..).get_mapped_range();
        bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
    }

    fn float_order(bits: u32) -> u32 {
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits ^ 0x8000_0000
        }
    }

    #[test]
    fn tuple_loop_outputs_follow_zero_odd_and_even_iterations() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .unwrap();
        let mut tuples = tuples::HostContext::new(&device).unwrap();
        let mut nested = nested::HostContext::new(&device).unwrap();
        macro_rules! check {
            ($module:ident, $result:expr, $expected:expr) => {{
                let result = $result.unwrap();
                let expected: Vec<Vec<u32>> = $expected;
                assert_eq!(result.values.len(), expected.len());
                for (output, expected) in result.values.iter().zip(expected) {
                    let $module::output::OutputResource::Buffer { buffer, .. } = &output.resource else {
                        panic!("array result");
                    };
                    assert_eq!(
                        read(&device, &queue, buffer, expected.len()),
                        expected,
                        "{}",
                        result.entry
                    );
                }
            }};
        }
        for k in [0, 1, 2, 3, 5] {
            eprintln!("checking tuple loops k={k}");
            let bound = input(&device, &[k]);
            let tuple_k = tuples::ParameterBuffer::new(&device, &k.to_le_bytes()).unwrap();
            let nested_k = nested::ParameterBuffer::new(&device, &k.to_le_bytes()).unwrap();
            check!(
                tuples,
                tuples::host_mapped(&mut tuples, &queue, &tuple_k, &bound, &bound),
                vec![vec![1 + k, 3 + k], vec![2 + 2 * k, 4 + 2 * k]]
            );
            check!(
                tuples,
                tuples::host_scanned(&mut tuples, &queue, &tuple_k, &bound, &bound, &bound, &bound),
                vec![vec![1, 3 + k], vec![2, 4 + 2 * k]]
            );
            check!(
                tuples,
                tuples::host_flags(&mut tuples, &queue, &tuple_k, &bound, &bound),
                vec![vec![1, 2]]
            );
            check!(
                tuples,
                tuples::host_nested(&mut tuples, &queue, &tuple_k, &bound, &bound),
                vec![vec![(3 << k) - 2, (4 << k) - 2]]
            );
            check!(
                nested,
                nested::host_main(&mut nested, &queue, &bound),
                vec![vec![1, 3], vec![2, 1, 4, 0]]
            );
            check!(
                nested,
                nested::host_updated(&mut nested, &queue, &nested_k, &bound, &bound),
                vec![vec![1 + k, 3 + k], vec![2 + 2 * k, 1 - k % 2, 4 + 2 * k, k % 2]]
            );
        }
        eprintln!("verified 30 tuple-loop cases");
    }

    #[test]
    fn radix_variants_preserve_values_and_stable_payload_order() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        eprintln!("GPU: {:?}", adapter.get_info());
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .unwrap();
        let mut context = HostContext::new(&device).unwrap();
        type Sort = fn(
            &mut HostContext,
            &wgpu::Queue,
            &wgpu::Buffer,
        ) -> Result<radix::output::OutputDescriptor, radix::HostError>;
        let sorts: [(&str, Sort, fn(u32) -> u32); 8] = [
            ("unsigned", radix::host_unsigned, |x| x),
            ("signed", radix::host_signed, |x| x ^ 0x8000_0000),
            ("floats", radix::host_floats, float_order),
            ("unsigned_key", radix::host_unsigned_key, |x| {
                ((x as i32) >> 8) as u32
            }),
            ("signed_key", radix::host_signed_key, |x| {
                (((x as i32) >> 8) as u32) ^ 0x8000_0000
            }),
            ("float_key", radix::host_float_key, |x| {
                float_order(x & 0xffff_ff00)
            }),
            ("odd_bits", radix::host_odd_bits, |x| x & 63),
            ("zero_bits", radix::host_zero_bits, |_| 0),
        ];
        // Includes signed limits, infinities, both zeros, subnormals, and NaN payloads.
        let edge = [
            0,
            0x8000_0000,
            0x7fff_ffff,
            0xffff_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0100,
            0xffc0_0100,
            0x7f80_0100,
            0xff80_0100,
            1,
            0x8000_0001,
            0x3f80_0000,
            0xbf80_0000,
        ];
        let mut cases = 0;
        // Repeated smaller lengths exercise context/scratch reuse after larger calls.
        for n in [1, 0, 2, 63, 64, 65, 255, 256, 257, 4097, 65537, 100003, 8, 0, 1] {
            let mut seed = 0x9e37_79b9u32;
            let random: Vec<_> = (0..n)
                .map(|i| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    if i % 3 == 0 {
                        edge[i % edge.len()]
                    } else {
                        seed
                    }
                })
                .collect();
            let duplicates: Vec<_> = (0..n)
                .map(|i| (edge[(i * 7 + i / 13) % edge.len()] & 0xffff_ff00) | (i as u32 & 255))
                .collect();
            for (pattern, values) in [("random", random), ("duplicates", duplicates)] {
                let xs = input(&device, &values);
                for (name, sort, key) in sorts {
                    eprintln!("checking {name} {pattern} n={n}");
                    let result = sort(&mut context, &queue, &xs).unwrap();
                    let OutputResource::Buffer { buffer, .. } = &result.values[0].resource else {
                        panic!("array result");
                    };
                    let mut expected = values.clone();
                    expected.sort_by_key(|&x| key(x));
                    assert_eq!(
                        read(&device, &queue, buffer, n),
                        expected,
                        "{name} {pattern} n={n}"
                    );
                    assert_eq!(read(&device, &queue, &xs, n), values, "input changed: {name}");
                    cases += 1;
                }
            }
        }
        for n in [0, 7, 8, 9] {
            let values: Vec<_> = (0..n).rev().map(|x| x as u32).collect();
            let xs = input(&device, &values);
            let result = radix::host_fixed(&mut context, &queue, &xs, &xs, &xs, &xs, &xs, &xs, &xs, &xs);
            if n == 8 {
                let result = result.unwrap();
                let OutputResource::Buffer { buffer, .. } = &result.values[0].resource else {
                    panic!("array result");
                };
                assert_eq!(read(&device, &queue, buffer, n), (0..8).collect::<Vec<_>>());
            } else {
                assert!(
                    matches!(result, Err(radix::HostError::Invalid(_))),
                    "fixed length {n}"
                );
            }
            cases += 1;
        }
        eprintln!("verified {cases} radix cases");
    }
}
