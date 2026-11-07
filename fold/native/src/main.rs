mod math;
mod mesh;
mod renderer;
mod simulation;
mod solver;
use math::*;
use renderer::*;
use simulation::*;
use std::{path::PathBuf, sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

struct State {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
    sim: Simulation,
    mesh: mesh::Mesh,
    solver: solver::Solver,
    renderer: Renderer,
    settings: Settings,
    blit: wgpu::RenderPipeline,
    blit_group: wgpu::BindGroup,
    last: Instant,
    frames: u32,
    cursor: [f64; 2],
    drag: Option<(V, V)>,
    window: Option<Arc<Window>>,
}
fn blit(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    view: &wgpu::TextureView,
) -> (wgpu::RenderPipeline, wgpu::BindGroup) {
    let module=device.create_shader_module(wgpu::ShaderModuleDescriptor {label:None,source:wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var image:texture_2d<f32>; @group(0) @binding(1) var linear_sampler:sampler; struct Quad {@builtin(position) p:vec4f,@location(0) uv:vec2f}; @vertex fn vertex(@builtin(vertex_index) i:u32)->Quad {var q:Quad;let p=vec2f(f32((i<<1u)&2u),f32(i&2u));q.p=vec4f(p*2.0-1.0,0,1);q.uv=vec2f(p.x,1.0-p.y);return q;} @fragment fn fragment(q:Quad)->@location(0) vec4f {return textureSampleLevel(image,linear_sampler,q.uv,0.0);}".into())});
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fragment"),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview: None,
        cache: None,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    (pipeline, group)
}
impl State {
    async fn new(window: Option<Arc<Window>>, width: u32, height: u32) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = window.clone().map(|window| instance.create_surface(window).expect("native surface"));
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: surface.as_ref(),
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
            })
            .await
            .expect("GPU adapter");
        eprintln!("Fold adapter: {}", adapter.get_info().name);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Wyn Fold native"),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("GPU device");
        let config = surface.as_ref().map(|surface| {
            let caps = surface.get_capabilities(&adapter);
            let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width,
                height,
                present_mode: wgpu::PresentMode::Fifo,
                desired_maximum_frame_latency: 2,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
            };
            surface.configure(&device, &config);
            config
        });
        let settings = Settings::default();
        let mut sim = Simulation::new();
        sim.viewport(width as f64 / height as f64, settings.thickness);
        sim.reset(settings.thickness, settings.speed);
        let arguments = std::env::args().collect::<Vec<_>>();
        if let Some(index) = arguments.iter().position(|s| s == "--load-state") {
            sim = serde_json::from_reader(std::fs::File::open(&arguments[index + 1]).expect("state file"))
                .expect("native state");
            sim.viewport(width as f64 / height as f64, settings.thickness);
        }
        let mut mesh = mesh::Mesh::new();
        mesh.update(&sim, settings.thickness);
        eprintln!("Creating native render passes");
        let scale = window.as_ref().map(|w| w.scale_factor()).unwrap_or(1.0);
        let renderer = Renderer::new(
            &device,
            (width as f64 / scale).round().max(1.0) as u32,
            (height as f64 / scale).round().max(1.0) as u32,
            &mesh,
        );
        eprintln!("Creating compiled Wyn solver");
        let solver = solver::Solver::new(&device, &sim);
        solver.upload_edges(&queue, &sim);
        let (blit, blit_group) = blit(
            &device,
            config.as_ref().map(|c| c.format).unwrap_or(wgpu::TextureFormat::Rgba8Unorm),
            &renderer.output.view,
        );
        eprintln!("FOLD_READY drag to grab; Space pause; R reset; Esc quit; +/- feed; [/] stiffness; ,/. thickness; T transmission; A AO; H roughness");
        Self {
            device,
            queue,
            surface,
            config,
            sim,
            mesh,
            solver,
            renderer,
            settings,
            blit,
            blit_group,
            last: Instant::now(),
            frames: 0,
            cursor: [0.0; 2],
            drag: None,
            window,
        }
    }
    fn step(&mut self) -> Result<(), String> {
        self.sim.prepare(self.settings.speed, self.settings.thickness);
        if self.sim.active > 0 {
            self.solver.solve(
                &self.device,
                &self.queue,
                &mut self.sim,
                self.settings.thickness,
                self.settings.stiffness,
            )?;
            self.sim.finish();
        }
        Ok(())
    }
    fn draw(&mut self, advance: bool) -> Result<(), String> {
        let now = Instant::now();
        let delta = (now - self.last).as_secs_f64().min(0.05);
        self.last = now;
        if advance && !self.settings.pause {
            self.sim.accumulator += delta;
            while self.sim.accumulator + 1e-10 >= DT {
                self.step()?;
                self.sim.accumulator -= DT;
            }
        }
        self.mesh.update(&self.sim, self.settings.thickness);
        if self.frames == 0 {
            eprintln!("Submitting native render passes");
        }
        self.renderer.draw(
            &self.device,
            &self.queue,
            &self.mesh,
            &self.settings,
            self.sim.active >= 2,
        );
        if self.frames == 0 {
            eprintln!("Native render passes submitted");
        }
        if let Some(surface) = &self.surface {
            let output = surface.get_current_texture().map_err(|e| e.to_string())?;
            let view = output.texture.create_view(&Default::default());
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&self.blit);
                pass.set_bind_group(0, &self.blit_group, &[]);
                pass.draw(0..3, 0..1);
            }
            self.queue.submit([encoder.finish()]);
            output.present();
        }
        self.frames += 1;
        Ok(())
    }
    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if let Some(config) = &mut self.config {
            config.width = width;
            config.height = height;
            self.surface.as_ref().unwrap().configure(&self.device, config);
        }
        self.sim.viewport(width as f64 / height as f64, self.settings.thickness);
        let scale = self.window.as_ref().map(|w| w.scale_factor()).unwrap_or(1.0);
        self.renderer = Renderer::new(
            &self.device,
            (width as f64 / scale).round().max(1.0) as u32,
            (height as f64 / scale).round().max(1.0) as u32,
            &self.mesh,
        );
        let (p, g) = blit(
            &self.device,
            self.config.as_ref().unwrap().format,
            &self.renderer.output.view,
        );
        self.blit = p;
        self.blit_group = g;
    }
    fn pointer(&mut self, pressed: bool) {
        if !pressed {
            self.sim.grab = None;
            self.drag = None;
            return;
        }
        let x = self.cursor[0]
            / self.config.as_ref().map(|c| c.width).unwrap_or(self.renderer.width) as f64
            * 2.0
            - 1.0;
        let y = 1.0
            - self.cursor[1]
                / self.config.as_ref().map(|c| c.height).unwrap_or(self.renderer.height) as f64
                * 2.0;
        let direction = ray(x, y, self.renderer.width as f64 / self.renderer.height as f64);
        if let Some((point, row, col)) = self.mesh.pick(EYE, direction) {
            let r = (row as f64 / 1000000.0 * (self.sim.active - 1) as f64).round() as usize;
            let index = r * 13 + col;
            if self.sim.begin_grab(index) {
                self.drag = Some((point, sub(self.sim.p[index], point)));
            }
        }
    }
    fn move_pointer(&mut self) {
        if let Some((point, offset)) = self.drag {
            let x = (self.cursor[0]
                / self.config.as_ref().map(|c| c.width).unwrap_or(self.renderer.width) as f64
                * 2.0
                - 1.0)
                .clamp(-0.99, 0.99);
            let y = (1.0
                - self.cursor[1]
                    / self.config.as_ref().map(|c| c.height).unwrap_or(self.renderer.height) as f64
                    * 2.0)
                .clamp(-0.99, 0.99);
            let direction = ray(x, y, self.renderer.width as f64 / self.renderer.height as f64);
            let (_, _, normal) = basis();
            let distance = dot(sub(point, EYE), normal) / dot(direction, normal);
            if let Some(g) = &mut self.sim.grab {
                g.target = add(add(EYE, mul(direction, distance)), offset);
            }
        }
    }
    fn key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Space => self.settings.pause = !self.settings.pause,
            KeyCode::KeyR => {
                self.sim.reset(self.settings.thickness, self.settings.speed);
                self.drag = None
            }
            KeyCode::Equal => self.settings.speed = (self.settings.speed + 0.1).min(2.0),
            KeyCode::Minus => self.settings.speed = (self.settings.speed - 0.1).max(0.0),
            KeyCode::BracketLeft => self.settings.stiffness = (self.settings.stiffness - 0.05).max(0.0),
            KeyCode::BracketRight => self.settings.stiffness = (self.settings.stiffness + 0.05).min(1.0),
            KeyCode::Comma => self.settings.thickness = (self.settings.thickness - 0.025).max(0.025),
            KeyCode::Period => self.settings.thickness = (self.settings.thickness + 0.025).min(1.0),
            KeyCode::KeyT => self.settings.transmission = 1.0 - self.settings.transmission,
            KeyCode::KeyA => self.settings.ao = !self.settings.ao,
            KeyCode::KeyH => self.settings.roughness = (self.settings.roughness + 0.1) % 0.6,
            KeyCode::KeyI => self.settings.ior = (self.settings.ior - 0.02).max(1.0),
            KeyCode::KeyO => self.settings.ior = (self.settings.ior + 0.02).min(2.5),
            KeyCode::KeyG => self.settings.tint = (self.settings.tint + 0.1) % 1.1,
            KeyCode::KeyD => self.settings.exposure = (self.settings.exposure - 0.1).max(-3.0),
            KeyCode::KeyE => self.settings.exposure = (self.settings.exposure + 0.1).min(3.0),
            KeyCode::KeyC => self.settings.contrast = (self.settings.contrast - 0.1).max(0.0),
            KeyCode::KeyV => self.settings.contrast = (self.settings.contrast + 0.1).min(3.0),
            KeyCode::KeyS => self.settings.saturation = (self.settings.saturation - 0.1).max(0.0),
            KeyCode::KeyW => self.settings.saturation = (self.settings.saturation + 0.1).min(3.0),
            KeyCode::KeyJ => self.settings.temperature = (self.settings.temperature - 0.1).max(-2.0),
            KeyCode::KeyK => self.settings.temperature = (self.settings.temperature + 0.1).min(2.0),
            _ => {}
        }
        self.sim.viewport(
            self.renderer.width as f64 / self.renderer.height as f64,
            self.settings.thickness,
        );
        if let Some(w) = &self.window {
            w.set_title(&format!("Wyn Fold | feed {:.1} stiffness {:.2} thickness {:.3} | Space pause · R reset · drag cloth",self.settings.speed,self.settings.stiffness,self.settings.thickness));
        }
    }
}
#[derive(Default)]
struct App {
    state: Option<State>,
    snapshot: Option<PathBuf>,
    max_frames: Option<u32>,
}
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Wyn Fold")
                        .with_inner_size(winit::dpi::LogicalSize::new(900, 900)),
                )
                .expect("native window"),
        );
        let size = window.inner_size();
        self.state = Some(pollster::block_on(State::new(
            Some(window.clone()),
            size.width,
            size.height,
        )));
        window.request_redraw();
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => s.resize(size.width, size.height),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed && !event.repeat {
                    if let PhysicalKey::Code(key) = event.physical_key {
                        if key == KeyCode::Escape {
                            event_loop.exit()
                        } else {
                            s.key(key)
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                s.cursor = [position.x, position.y];
                s.move_pointer();
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => s.pointer(state == ElementState::Pressed),
            WindowEvent::Focused(false) => s.pointer(false),
            WindowEvent::RedrawRequested => {
                if let Err(error) = s.draw(true) {
                    eprintln!("Fold failed: {error}");
                    event_loop.exit();
                    return;
                }
                if let Some(path) = self.snapshot.take() {
                    if let Err(error) = s.renderer.snapshot(&s.device, &s.queue, &path) {
                        eprintln!("Snapshot failed: {error}");
                    } else {
                        eprintln!("Snapshot: {}", path.display());
                    }
                }
                if self.max_frames.is_some_and(|n| s.frames >= n) {
                    event_loop.exit()
                } else {
                    s.window.as_ref().unwrap().request_redraw()
                }
            }
            _ => {}
        }
    }
}
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let value = |key: &str| args.iter().position(|s| s == key).and_then(|i| args.get(i + 1));
    let snapshot = value("--snapshot").map(PathBuf::from);
    let max_frames = value("--frames").map(|s| s.parse::<u32>().expect("frame count"));
    if args.iter().any(|s| s == "--headless") {
        let size = value("--size").map(|s| s.parse::<u32>().expect("image size")).unwrap_or(900);
        let mut state = pollster::block_on(State::new(None, size, size));
        if args.iter().any(|s| s == "--no-ao") {
            state.settings.ao = false;
        }
        if let Some(path) = value("--fixture") {
            state
                .solver
                .fixture(&state.device, &state.queue, &PathBuf::from(path))
                .expect("native numerical fixture");
            return;
        }
        let steps = value("--steps").map(|s| s.parse::<usize>().expect("step count")).unwrap_or(1);
        for _ in 0..steps {
            state.step().expect("Wyn simulation step");
        }
        if let Some(path) = value("--save-state") {
            serde_json::to_writer(std::fs::File::create(path).expect("state output"), &state.sim)
                .expect("save native state");
        }
        if args.iter().any(|s| s == "--no-render") {
            println!("Stepped to {:.6}", state.sim.time);
            return;
        }
        state.draw(false).expect("native rendering");
        if let Some(path) = snapshot {
            state.renderer.snapshot(&state.device, &state.queue, &path).expect("native snapshot");
        }
        println!(
            "Rendered {} particles at simulation time {:.6}",
            state.sim.active * 13,
            state.sim.time
        );
    } else {
        let event_loop = EventLoop::new().expect("native event loop");
        event_loop
            .run_app(&mut App {
                state: None,
                snapshot,
                max_frames,
            })
            .expect("native app");
    }
}
