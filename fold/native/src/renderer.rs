use crate::{
    math::*,
    mesh::{Mesh, Vertex},
};
use wgpu::util::DeviceExt;
pub struct Target {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}
impl Target {
    fn new(device: &wgpu::Device, w: u32, h: u32, format: wgpu::TextureFormat) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Fold render target"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        Self { texture, view }
    }
}
pub struct Renderer {
    pub width: u32,
    pub height: u32,
    pub output: Target,
    front: Target,
    front_depth: Target,
    behind: Target,
    behind_depth: Target,
    beauty: Target,
    beauty_depth: Target,
    ambient: Target,
    ambient_pipeline: wgpu::RenderPipeline,
    graded: Target,
    antialias: wgpu::RenderPipeline,
    geometry: wgpu::RenderPipeline,
    peel: wgpu::RenderPipeline,
    surface: wgpu::RenderPipeline,
    post: wgpu::RenderPipeline,
    groups: Vec<wgpu::BindGroup>,
    uniform: wgpu::Buffer,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    floor: wgpu::Buffer,
    index_count: u32,
}
pub struct Settings {
    pub speed: f64,
    pub thickness: f64,
    pub stiffness: f64,
    pub transmission: f32,
    pub ior: f32,
    pub tint: f32,
    pub roughness: f32,
    pub exposure: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub temperature: f32,
    pub ao: bool,
    pub pause: bool,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            speed: 2.0,
            thickness: 0.35,
            stiffness: 0.5,
            transmission: 1.0,
            ior: 1.46,
            tint: 0.95,
            roughness: 0.0,
            exposure: 0.0,
            contrast: 1.2,
            saturation: 2.0,
            temperature: 0.0,
            ao: true,
            pause: false,
        }
    }
}
fn pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    fragment: &str,
    format: wgpu::TextureFormat,
    quad: bool,
) -> wgpu::RenderPipeline {
    let attributes = wgpu::vertex_attr_array![0=>Float32x3,1=>Float32x3,2=>Float32,3=>Float32];
    let layout = wgpu::VertexBufferLayout {
        array_stride: 32,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &attributes,
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(fragment),
        layout: None,
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(if quad { "quad" } else { "vertex" }),
            compilation_options: Default::default(),
            buffers: if quad { &[] } else { std::slice::from_ref(&layout) },
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fragment),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: if quad { None } else { Some(wgpu::Face::Back) },
            ..Default::default()
        },
        depth_stencil: if quad {
            None
        } else {
            Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            })
        },
        multisample: Default::default(),
        multiview: None,
        cache: None,
    })
}
impl Renderer {
    pub fn new(device: &wgpu::Device, width: u32, height: u32, mesh: &Mesh) -> Self {
        let hdr = wgpu::TextureFormat::Rgba16Float;
        let depth = wgpu::TextureFormat::Depth32Float;
        let front = Target::new(device, width, height, hdr);
        let front_depth = Target::new(device, width, height, depth);
        let behind = Target::new(device, width, height, hdr);
        let behind_depth = Target::new(device, width, height, depth);
        let beauty = Target::new(device, width, height, hdr);
        let beauty_depth = Target::new(device, width, height, depth);
        let ambient = Target::new(device, width, height, hdr);
        let graded = Target::new(device, width, height, hdr);
        // Post produces display-referred values, so the output is deliberately UNORM.
        let output = Target::new(device, width, height, wgpu::TextureFormat::Rgba8Unorm);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fold native material"),
            source: wgpu::ShaderSource::Wgsl(include_str!("render.wgsl").into()),
        });
        let geometry = pipeline(device, &module, "geometry", hdr, false);
        let peel = pipeline(device, &module, "peel", hdr, false);
        let surface = pipeline(device, &module, "beauty", hdr, false);
        let post = pipeline(device, &module, "post", hdr, true);
        let antialias = pipeline(
            device,
            &module,
            "antialias",
            wgpu::TextureFormat::Rgba8Unorm,
            true,
        );
        let ambient_pipeline = pipeline(device, &module, "ambient", hdr, true);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 128,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let u = || wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        };
        let t = |binding, view| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(view),
        };
        let s = || wgpu::BindGroupEntry {
            binding: 6,
            resource: wgpu::BindingResource::Sampler(&sampler),
        };
        let groups = vec![
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &geometry.get_bind_group_layout(0),
                entries: &[u()],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &peel.get_bind_group_layout(0),
                entries: &[u(), t(1, &front_depth.view)],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &surface.get_bind_group_layout(0),
                entries: &[
                    u(),
                    t(1, &front_depth.view),
                    t(2, &behind.view),
                    t(3, &behind_depth.view),
                    s(),
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &post.get_bind_group_layout(0),
                entries: &[
                    u(),
                    t(1, &front_depth.view),
                    t(4, &beauty.view),
                    t(5, &front.view),
                    t(7, &ambient.view),
                    s(),
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &ambient_pipeline.get_bind_group_layout(0),
                entries: &[u(), t(1, &front_depth.view), t(5, &front.view)],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &antialias.get_bind_group_layout(0),
                entries: &[u(), t(8, &graded.view), s()],
            }),
        ];
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fold shell"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let floor_points = [
            [-100.0, 0.0, -100.0],
            [-100.0, 0.0, 100.0],
            [100.0, 0.0, -100.0],
            [100.0, 0.0, -100.0],
            [-100.0, 0.0, 100.0],
            [100.0, 0.0, 100.0],
        ];
        let floor_vertices = floor_points.map(|p| Vertex {
            p,
            n: [0.0, 1.0, 0.0],
            birth: -1000000.0,
            fade: 1.0,
        });
        let floor = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&floor_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Self {
            width,
            height,
            output,
            front,
            front_depth,
            behind,
            behind_depth,
            beauty,
            beauty_depth,
            ambient,
            ambient_pipeline,
            graded,
            antialias,
            geometry,
            peel,
            surface,
            post,
            groups,
            uniform,
            vertices,
            indices,
            floor,
            index_count: mesh.indices.len() as u32,
        }
    }
    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &Mesh,
        settings: &Settings,
        visible: bool,
    ) {
        let (right, up, forward) = basis();
        let f = (1.0 / (55.0f64.to_radians() * 0.5).tan()) as f32;
        let aspect = self.width as f32 / self.height as f32;
        let mut frame = Vec::new();
        for v in [right, up, forward, EYE] {
            frame.extend(v.map(|x| x as f32));
            frame.push(0.0);
        }
        frame.extend([
            self.width as f32,
            self.height as f32,
            f / aspect,
            f,
            settings.thickness as f32,
            settings.ior,
            settings.tint,
            settings.roughness,
            settings.exposure,
            settings.contrast,
            settings.saturation,
            settings.temperature,
            settings.transmission,
            if settings.ao { 1.0 } else { 0.0 },
            0.0,
            0.0,
        ]);
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&frame));
        queue.write_buffer(&self.vertices, 0, bytemuck::cast_slice(&mesh.vertices));
        let mut encoder = device.create_command_encoder(&Default::default());
        for (i, (pipeline, color, depth)) in [
            (&self.geometry, &self.front, &self.front_depth),
            (&self.peel, &self.behind, &self.behind_depth),
            (&self.surface, &self.beauty, &self.beauty_depth),
        ]
        .into_iter()
        .enumerate()
        {
            let clear = if i == 0 {
                wgpu::Color::TRANSPARENT
            } else {
                wgpu::Color {
                    r: 0.815,
                    g: 0.791,
                    b: 0.723,
                    a: 1.0,
                }
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fold layered shell"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.groups[i], &[]);
            pass.set_vertex_buffer(0, self.floor.slice(..));
            pass.draw(0..6, 0..1);
            if visible {
                pass.set_vertex_buffer(0, self.vertices.slice(..));
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.index_count, 0, 0..1);
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fold ambient occlusion"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ambient.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.ambient_pipeline);
            pass.set_bind_group(0, &self.groups[4], &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fold AO / ACES / grading / antialias"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.graded.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.post);
            pass.set_bind_group(0, &self.groups[3], &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fold display antialias"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.output.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.antialias);
            pass.set_bind_group(0, &self.groups[5], &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
    }
    pub fn snapshot(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &std::path::Path,
    ) -> Result<(), String> {
        let stride = (self.width * 4 + 255) & !255;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: stride as u64 * self.height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            self.output.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
        rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        let mapped = buffer.slice(..).get_mapped_range();
        let mut pixels = Vec::new();
        for row in mapped.chunks_exact(stride as usize) {
            pixels.extend_from_slice(&row[..self.width as usize * 4]);
        }
        image::save_buffer(path, &pixels, self.width, self.height, image::ColorType::Rgba8)
            .map_err(|e| e.to_string())?;
        drop(mapped);
        buffer.unmap();
        Ok(())
    }
}
