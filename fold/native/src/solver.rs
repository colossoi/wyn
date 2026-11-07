use crate::simulation::{Simulation, N};
use wgpu::util::DeviceExt;

extern "C" {
    fn fold_cpu(
        p: *const f32,
        old: *const f32,
        edges: *const f32,
        parameters: *const f32,
        grabs: *const f32,
        output: *mut f32,
    );
}

pub struct Solver {
    gpu: Option<GpuSolver>,
}
impl Solver {
    pub fn new(device: &wgpu::Device, sim: &Simulation) -> Self {
        let use_gpu = std::env::args().any(|a| a == "--gpu-solver");
        eprintln!(
            "Fold solver: {}",
            if use_gpu { "serial GPU (diagnostic)" } else { "Wyn-generated native CPU" }
        );
        Self {
            gpu: use_gpu.then(|| GpuSolver::new(device, sim)),
        }
    }
    pub fn upload_edges(&self, queue: &wgpu::Queue, sim: &Simulation) {
        if let Some(gpu) = &self.gpu {
            gpu.upload_edges(queue, sim)
        }
    }
    pub fn solve(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sim: &mut Simulation,
        thickness: f64,
        stiffness: f64,
    ) -> Result<(), String> {
        if let Some(gpu) = &self.gpu {
            return gpu.solve(device, queue, sim, thickness, stiffness);
        }
        let (p, old, parameters, grabs) = sim.transport(thickness, stiffness);
        let mut output = vec![[0.0f32; 4]; N];
        // The C ABI consumes fixed-size arrays copied into aligned Clang
        // vectors. All lengths/layouts are fixed by the compiled Wyn entry.
        unsafe {
            fold_cpu(
                p.as_ptr().cast(),
                old.as_ptr().cast(),
                sim.edges.as_ptr().cast(),
                parameters.as_ptr(),
                grabs.as_ptr().cast(),
                output.as_mut_ptr().cast(),
            );
        }
        let count = sim.active * 13;
        if output[0][3] < 0.0 || output[..count].iter().any(|p| p[..3].iter().any(|x| !x.is_finite())) {
            return Err("Invalid Wyn CPU solver result".into());
        }
        for (dst, src) in sim.p[..count].iter_mut().zip(&output) {
            *dst = [src[0] as f64, src[1] as f64, src[2] as f64];
        }
        Ok(())
    }
    pub fn fixture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &std::path::Path,
    ) -> Result<(), String> {
        if let Some(gpu) = &self.gpu {
            return gpu.fixture(device, queue, path);
        }
        let load = |name: &str, size: usize| -> Result<Vec<f32>, String> {
            let bytes = std::fs::read(path.join(format!("{name}.bin"))).map_err(|e| e.to_string())?;
            if bytes.len() != size {
                return Err(format!("Unexpected {name} length"));
            }
            Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
        };
        let p = load("input_p", N * 16)?;
        let old = load("old", N * 16)?;
        let edges = load("edges", 13880 * 16)?;
        let parameters = load("parameters", 176)?;
        let grabs = load("grabs", 144)?;
        let mut output = vec![0.0f32; N * 4];
        unsafe {
            fold_cpu(
                p.as_ptr(),
                old.as_ptr(),
                edges.as_ptr(),
                parameters.as_ptr(),
                grabs.as_ptr(),
                output.as_mut_ptr(),
            );
        }
        std::fs::write(path.join("actual.bin"), bytemuck::cast_slice(&output)).map_err(|e| e.to_string())
    }
}

struct GpuSolver {
    pipeline: wgpu::ComputePipeline,
    groups: Vec<wgpu::BindGroup>,
    buffers: Vec<wgpu::Buffer>,
    readback: wgpu::Buffer,
    timestamps: Option<(wgpu::QuerySet, wgpu::Buffer)>,
}
impl GpuSolver {
    pub fn new(device: &wgpu::Device, sim: &Simulation) -> Self {
        let path = std::env::var("FOLD_SHADER").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/../generated/constraints.wgsl").to_string()
        });
        let code =
            std::fs::read_to_string(path).expect("Build the Wyn solver first: python3 fold/build.py");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Wyn XPBD"),
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Wyn ordered XPBD"),
            layout: None,
            module: &module,
            entry_point: Some("constraints_compute"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sizes = [39728, 39728, 222080, 144, 39728, 176];
        let buffers = sizes
            .iter()
            .map(|&size| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Wyn solver transport"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect::<Vec<_>>();
        let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &(0..5)
                .map(|i| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: buffers[i].as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers[5].as_entire_binding(),
            }],
        });
        let scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fold serial scratch"),
            size: 177824,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(2),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scratch.as_entire_binding(),
            }],
        });
        let readback = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Wyn positions readback"),
            contents: &vec![0; N * 16 + 16],
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        });
        let _ = sim;
        let timestamps = device.features().contains(wgpu::Features::TIMESTAMP_QUERY).then(|| {
            (
                device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: None,
                    ty: wgpu::QueryType::Timestamp,
                    count: 2,
                }),
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 16,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
            )
        });
        Self {
            pipeline,
            groups: vec![group0, group1, group2],
            buffers,
            readback,
            timestamps,
        }
    }
    pub fn upload_edges(&self, queue: &wgpu::Queue, sim: &Simulation) {
        queue.write_buffer(&self.buffers[2], 0, bytemuck::cast_slice(&sim.edges));
    }
    pub fn solve(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sim: &mut Simulation,
        thickness: f64,
        stiffness: f64,
    ) -> Result<(), String> {
        let (p, old, parameters, grabs) = sim.transport(thickness, stiffness);
        for (i, bytes) in [
            (0, bytemuck::cast_slice(&p)),
            (1, bytemuck::cast_slice(&old)),
            (3, bytemuck::cast_slice(&grabs)),
            (5, bytemuck::cast_slice(&parameters)),
        ] {
            queue.write_buffer(&self.buffers[i], 0, bytes)
        }
        let floats = self.dispatch(device, queue)?;
        let valid =
            floats[3] >= 0.0 && (0..sim.active * 13).all(|i| (0..3).all(|a| floats[i * 4 + a].is_finite()));
        if valid {
            for i in 0..sim.active * 13 {
                for a in 0..3 {
                    sim.p[i][a] = floats[i * 4 + a] as f64;
                }
            }
        }
        if valid {
            Ok(())
        } else {
            Err("Invalid Wyn solver result/contact chain".into())
        }
    }
    fn dispatch(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<f32>, String> {
        let mut encoder = device.create_command_encoder(&Default::default());
        let copy_only = std::env::args().any(|a| a == "--profile-copy");
        if copy_only {
            encoder.copy_buffer_to_buffer(&self.buffers[0], 0, &self.buffers[4], 0, N as u64 * 16);
        } else {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: self.timestamps.as_ref().map(|(query_set, _)| {
                    wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }
                }),
            });
            pass.set_pipeline(&self.pipeline);
            for (i, group) in self.groups.iter().enumerate() {
                pass.set_bind_group(i as u32, group, &[])
            }
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.buffers[4], 0, &self.readback, 0, N as u64 * 16);
        if let Some((query, resolve)) = self.timestamps.as_ref().filter(|_| !copy_only) {
            encoder.resolve_query_set(query, 0..2, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, &self.readback, N as u64 * 16, 16);
        }
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        self.readback.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
        rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        let result = self.readback.slice(..).get_mapped_range();
        let floats: Vec<f32> = bytemuck::cast_slice(&result[..N * 16]).to_vec();
        if self.timestamps.is_some() && !copy_only {
            let begin = u64::from_le_bytes(result[N * 16..N * 16 + 8].try_into().unwrap());
            let end = u64::from_le_bytes(result[N * 16 + 8..N * 16 + 16].try_into().unwrap());
            eprintln!(
                "PROFILE solver_gpu_ms={:.3}",
                end.saturating_sub(begin) as f64 * queue.get_timestamp_period() as f64 / 1e6
            );
        }
        drop(result);
        self.readback.unmap();
        Ok(floats)
    }
    pub fn fixture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &std::path::Path,
    ) -> Result<(), String> {
        for (i, name) in [
            (0, "input_p"),
            (1, "old"),
            (2, "edges"),
            (3, "grabs"),
            (5, "parameters"),
        ] {
            let data = std::fs::read(path.join(format!("{name}.bin"))).map_err(|e| e.to_string())?;
            if data.len() as u64 != self.buffers[i].size() {
                return Err(format!("Wrong {name} size"));
            }
            queue.write_buffer(&self.buffers[i], 0, &data);
        }
        let result = self.dispatch(device, queue)?;
        std::fs::write(path.join("actual.bin"), bytemuck::cast_slice(&result)).map_err(|e| e.to_string())
    }
}
