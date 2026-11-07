use crate::simulation::{Simulation, N};
use wgpu::util::DeviceExt;
pub struct Solver {
    pipeline: wgpu::ComputePipeline,
    groups: Vec<wgpu::BindGroup>,
    buffers: Vec<wgpu::Buffer>,
    readback: wgpu::Buffer,
}
impl Solver {
    pub fn new(device: &wgpu::Device, sim: &Simulation) -> Self {
        let code = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../generated/constraints.wgsl"
        ))
        .expect("Build the Wyn solver first: python3 fold/build.py");
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
            contents: &vec![0; N * 16],
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        });
        let _ = sim;
        Self {
            pipeline,
            groups: vec![group0, group1, group2],
            buffers,
            readback,
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
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            for (i, group) in self.groups.iter().enumerate() {
                pass.set_bind_group(i as u32, group, &[])
            }
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.buffers[4], 0, &self.readback, 0, N as u64 * 16);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        self.readback.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
        rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        let result = self.readback.slice(..).get_mapped_range();
        let floats: Vec<f32> = bytemuck::cast_slice(&result).to_vec();
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
