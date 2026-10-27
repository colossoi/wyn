// Copied beside generated radix.rs by benchmark_radix.py. Timed runs exclude
// pipeline setup and verification; resident timings include submission and GPU completion.
mod radix;

use radix::{output::OutputResource, HostContext};
use std::{error::Error, sync::mpsc, time::Instant};
use wgpu::util::DeviceExt;

fn read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    bytes: u64,
) -> Result<Vec<i32>, Box<dyn Error>> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("benchmark readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
    queue.submit(Some(encoder.finish()));
    let (send, receive) = mpsc::sync_channel(1);
    staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        if send.send(result).is_err() { /* Caller has already returned a polling error. */ }
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receive.recv()??;
    let mapped = staging.slice(..).get_mapped_range();
    let values = mapped.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    drop(mapped);
    staging.unmap();
    Ok(values)
}

fn main() -> Result<(), Box<dyn Error>> {
    let sizes: Vec<usize> = (0..=6).map(|power| (64usize * 8usize.pow(power)) / 4).collect();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
    eprintln!("GPU: {:?}", adapter.get_info());
    let limits = adapter.limits();
    eprintln!(
        "Limits: max buffer {}, max storage binding {}",
        limits.max_buffer_size, limits.max_storage_buffer_binding_size
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))?;
    let mut context = HostContext::new(&device)?;
    type Sort = fn(
        &mut HostContext,
        &wgpu::Queue,
        &wgpu::Buffer,
    ) -> Result<radix::output::OutputDescriptor, radix::HostError>;
    let sorts: [(u32, Sort); 2] = [(2, radix::host_two), (4, radix::host_four)];
    println!("bits,n,run,resident_ms,end_to_end_ms");
    for n in sizes {
        let mut seed = 0x12345678u32;
        let input: Vec<i32> = (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed as i32
            })
            .collect();
        let bytes: Vec<_> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut expected = input.clone();
        expected.sort_unstable();
        for (bits, sort) in sorts {
            let prefix_bytes = (n as u64) * if bits == 2 { 16 } else { 64 };
            let estimated_bytes = prefix_bytes + (n as u64) * 12 + 1024 * 1024;
            if prefix_bytes > limits.max_buffer_size
                || prefix_bytes > u64::from(limits.max_storage_buffer_binding_size)
                || estimated_bytes > 5 * 1024 * 1024 * 1024
            {
                eprintln!("SKIP bits={bits} n={n}: prefix={prefix_bytes} bytes, estimated working set={estimated_bytes}; binding limit or 5 GiB memory budget exceeded");
                continue;
            }
            for run in 0..6 {
                let total = Instant::now();
                let xs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("benchmark input"),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });
                queue.submit([]);
                device.poll(wgpu::PollType::wait_indefinitely())?;
                let start = Instant::now();
                let result = sort(&mut context, &queue, &xs)?;
                device.poll(wgpu::PollType::wait_indefinitely())?;
                let resident = start.elapsed().as_secs_f64() * 1000.0;
                let OutputResource::Buffer { buffer, .. } = &result.values[0].resource else {
                    return Err("sort did not return a buffer".into());
                };
                let output = read(&device, &queue, buffer, bytes.len() as u64)?;
                let elapsed = total.elapsed().as_secs_f64() * 1000.0;
                if output != expected {
                    return Err(format!("incorrect {bits}-bit sort: n={n}, run={run}").into());
                }
                if run > 0 {
                    println!("{bits},{n},{run},{resident:.6},{elapsed:.6}");
                }
            }
        }
    }
    Ok(())
}
