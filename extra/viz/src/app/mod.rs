//! Interactive winit + wgpu application shell shared by the
//! `vertex-fragment` and `testpattern` modes. `PipelineSpec` describes
//! what to render; `State` owns the surface + pipeline; `App` is the
//! `winit::ApplicationHandler` wrapper.

mod render;
mod uniforms;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, Color, CommandEncoderDescriptor,
    InstanceFlags, LoadOp, Operations, PresentMode, RenderPipeline, StoreOp, SurfaceConfiguration,
};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

use crate::gpu::{DeviceRequest, GpuContext};
use uniforms::{MouseUniform, ResolutionUniform, TimeUniform};

// --- Pipeline spec passed to the app -----------------------------------------

/// Configuration for the interactive viewer. Both `vf` and `testpattern`
/// modes flow through this struct — they only differ in `shader` and a
/// few fixed-default fields (testpattern uses Fifo, validate=true, etc.).
pub struct PipelineSpec {
    pub shader: Shader,
    pub vertex_entry: String,
    pub fragment_entry: String,
    /// Bind shadertoy-style uniforms (iResolution, iTime, iMouse,
    /// difficulty) discovered from the SPIR-V's JSON sidecar. Only
    /// meaningful for `Shader::Spirv`; ignored for `Shader::Wgsl`,
    /// which uses its own hardcoded resolution uniform.
    pub shadertoy: bool,
    pub max_frames: Option<u32>,
    pub verbose: bool,
    pub validate: bool,
    pub present_mode: PresentMode,
    pub difficulty: i32,
    pub size: Option<(u32, u32)>,
}

/// Where the WGSL/SPIR-V module comes from.
pub enum Shader {
    /// Load SPIR-V from disk; entry-point names come from
    /// `PipelineSpec.{vertex_entry, fragment_entry}`.
    Spirv(PathBuf),
    /// Embedded WGSL source compiled in-place. Used by the built-in
    /// test pattern; `vertex_entry` / `fragment_entry` must name
    /// functions inside `source`.
    Wgsl(&'static str),
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: SurfaceConfiguration,
    pipeline: RenderPipeline,
    // Uniform support - separate buffers for iResolution, iTime, iMouse, difficulty (optional, enabled with --shadertoy)
    resolution_buffer: Option<wgpu::Buffer>,
    time_buffer: Option<wgpu::Buffer>,
    mouse_buffer: Option<wgpu::Buffer>,
    _difficulty_buffer: Option<wgpu::Buffer>,
    uniform_bind_group: Option<BindGroup>,
    /// Descriptor-set index at which `uniform_bind_group` is bound. Defaults
    /// to 0; overridden from the sidecar pipeline descriptor when the shader
    /// declares its uniforms at a different set.
    uniform_bind_group_set: u32,
    /// Empty bind group, bound at every set index below
    /// `uniform_bind_group_set` to satisfy wgpu's contiguous-sets requirement.
    empty_bind_group: Option<BindGroup>,
    start_time: std::time::Instant,
    // Mouse tracking
    mouse_pos: [f32; 2],
    mouse_click_pos: [f32; 2],
    mouse_pressed: bool,
    // Frame limiting (optional, for debugging)
    frame_count: u32,
    max_frames: Option<u32>,
    // Frame timing (includes GPU wait)
    frame_times: [f64; 60], // Recent frame times in ms (ring buffer)
    frame_time_idx: usize,
    verbose: bool,
}

impl State {
    /// Print VRAM memory statistics from the allocator
    fn print_memory_stats(&self, label: &str) {
        if let Some(report) = self.device.generate_allocator_report() {
            let allocated_mb = report.total_allocated_bytes as f64 / (1024.0 * 1024.0);
            let reserved_mb = report.total_reserved_bytes as f64 / (1024.0 * 1024.0);
            eprintln!(
                "[Memory {}] Allocated: {:.2} MB, Reserved: {:.2} MB, Blocks: {}, Allocations: {}",
                label,
                allocated_mb,
                reserved_mb,
                report.blocks.len(),
                report.allocations.len()
            );
        } else {
            eprintln!(
                "[Memory {}] Allocator report not available on this backend",
                label
            );
        }
    }
}

impl State {
    async fn new(window: Arc<Window>, spec: &PipelineSpec) -> Result<Self> {
        let validate = spec.validate;
        if validate {
            eprintln!("[viz] Validation layers ENABLED");
        } else {
            eprintln!("[viz] Validation layers DISABLED");
        }

        let win = window.clone();
        let ctx = GpuContext::request(DeviceRequest {
            instance_flags: if validate {
                InstanceFlags::VALIDATION | InstanceFlags::DEBUG
            } else {
                InstanceFlags::empty()
            },
            desired_features: wgpu::Features::SPIRV_SHADER_PASSTHROUGH,
            surface_target: Some(Box::new(move |inst| {
                inst.create_surface(win).context("failed to create wgpu surface")
            })),
            ..Default::default()
        })
        .await?;

        let surface = ctx.surface.expect("surface_target was Some, surface must be present");
        let adapter = ctx.adapter;
        let device = ctx.device;
        let queue = ctx.queue;

        let verbose = spec.verbose;
        if verbose {
            let info = adapter.get_info();
            eprintln!("[viz] Adapter: {} ({:?})", info.name, info.backend);
            eprintln!("[viz] Driver: {}", info.driver);
            eprintln!("[viz] Driver info: {}", info.driver_info);
            eprintln!(
                "[viz] SPIRV_SHADER_PASSTHROUGH supported: {}",
                device.features().contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH)
            );
        }

        // Set up uncaptured error handler to catch GPU errors at runtime
        device.on_uncaptured_error(Box::new(|error| {
            eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
            eprintln!("║                     GPU VALIDATION ERROR                     ║");
            eprintln!("╚══════════════════════════════════════════════════════════════╝");
            eprintln!("{:?}", error);
            eprintln!();
        }));

        // Set up device lost handler
        device.set_device_lost_callback(|reason, message| {
            eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
            eprintln!("║                      GPU DEVICE LOST                         ║");
            eprintln!("╚══════════════════════════════════════════════════════════════╝");
            eprintln!("Reason: {:?}", reason);
            eprintln!("Message: {}", message);
            eprintln!();
            std::process::exit(1);
        });

        let size = window.inner_size();
        let present_mode = spec.present_mode;
        if verbose {
            eprintln!("[viz] Present mode: {:?}", present_mode);
        }
        let config =
            render::configure_surface(&surface, &device, &adapter, size.width, size.height, present_mode)?;

        // Shadertoy uniforms: discovered from the SPIR-V's JSON sidecar.
        // Only `--shadertoy` on a SPIR-V shader uses this path; the WGSL
        // test pattern builds its own resolution uniform inside
        // `build_wgsl_render_pipeline`.
        let shadertoy = if let (true, Shader::Spirv(path)) = (spec.shadertoy, &spec.shader) {
            Some(uniforms::build_shadertoy(&device, &queue, path, spec.difficulty)?)
        } else {
            None
        };
        let uniform_bind_group_set = shadertoy.as_ref().map(|s| s.bind_group_set).unwrap_or(0);

        // wgpu requires set indices in the pipeline layout to be
        // contiguous from 0; if the shadertoy uniforms live at set > 0,
        // we need an empty bind group at every lower set.
        let (empty_bind_group_layout, empty_bind_group) = if uniform_bind_group_set > 0 {
            let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("empty_bind_group_layout"),
                entries: &[],
            });
            let group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("empty_bind_group"),
                layout: &layout,
                entries: &[],
            });
            (Some(layout), Some(group))
        } else {
            (None, None)
        };

        let max_frames = spec.max_frames;

        let (pipeline, resolution_buffer, time_buffer, mouse_buffer, difficulty_buffer, uniform_bind_group) =
            match &spec.shader {
                Shader::Spirv(path) => {
                    let pipeline = render::build_spirv_render_pipeline(
                        &device,
                        config.format,
                        path,
                        &spec.vertex_entry,
                        &spec.fragment_entry,
                        shadertoy.as_ref().map(|s| {
                            (
                                &s.bind_group_layout,
                                empty_bind_group_layout.as_ref(),
                                s.bind_group_set,
                            )
                        }),
                    )?;
                    let (rb, tb, mb, db, bg) = match shadertoy {
                        Some(s) => (
                            s.resolution_buffer,
                            s.time_buffer,
                            s.mouse_buffer,
                            s.difficulty_buffer,
                            Some(s.bind_group),
                        ),
                        None => (None, None, None, None, None),
                    };
                    (pipeline, rb, tb, mb, db, bg)
                }
                Shader::Wgsl(source) => {
                    let (pipeline, res_buffer, bind_group) = render::build_wgsl_render_pipeline(
                        &device,
                        &queue,
                        config.format,
                        config.width,
                        config.height,
                        source,
                    );
                    (pipeline, Some(res_buffer), None, None, None, Some(bind_group))
                }
            };

        let now = std::time::Instant::now();

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            resolution_buffer,
            time_buffer,
            mouse_buffer,
            _difficulty_buffer: difficulty_buffer,
            uniform_bind_group,
            uniform_bind_group_set,
            empty_bind_group,
            start_time: now,
            mouse_pos: [0.0, 0.0],
            mouse_click_pos: [0.0, 0.0],
            mouse_pressed: false,
            frame_count: 0,
            max_frames,
            frame_times: [0.0; 60],
            frame_time_idx: 0,
            verbose,
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn render(&mut self) {
        // Start frame timing (only when verbose)
        let frame_start = if self.verbose { Some(std::time::Instant::now()) } else { None };

        self.frame_count += 1;

        // Update uniform data for this frame if Shadertoy mode is enabled
        if let (Some(ref resolution_buffer), Some(ref time_buffer)) =
            (&self.resolution_buffer, &self.time_buffer)
        {
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(self.start_time).as_secs_f32();

            // Update iResolution
            let resolution = ResolutionUniform {
                resolution: [self.config.width as f32, self.config.height as f32, 1.0],
                _pad: 0.0,
            };
            self.queue.write_buffer(resolution_buffer, 0, bytemuck::cast_slice(&[resolution]));

            // Update iTime
            let time = TimeUniform { time: elapsed };
            self.queue.write_buffer(time_buffer, 0, bytemuck::cast_slice(&[time]));

            // Update iMouse
            if let Some(mouse_buffer) = &self.mouse_buffer {
                // Shadertoy convention: (x, y, click_x, click_y)
                // click_x/click_y are negative when not pressed
                let mouse = MouseUniform {
                    mouse: if self.mouse_pressed {
                        [
                            self.mouse_pos[0],
                            self.mouse_pos[1],
                            self.mouse_click_pos[0],
                            self.mouse_click_pos[1],
                        ]
                    } else {
                        [self.mouse_pos[0], self.mouse_pos[1], -1.0, -1.0]
                    },
                };
                self.queue.write_buffer(mouse_buffer, 0, bytemuck::cast_slice(&[mouse]));
            }
        }

        match self.surface.get_current_texture() {
            Ok(frame) => {
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: Operations {
                                load: LoadOp::Clear(Color {
                                    r: 0.02,
                                    g: 0.02,
                                    b: 0.02,
                                    a: 1.0,
                                }),
                                store: StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });

                    rpass.set_pipeline(&self.pipeline);
                    // Set uniform bind group if available, at the set number
                    // declared by the shader (via sidecar descriptor). Fill
                    // any lower sets with the shared empty bind group so the
                    // pipeline layout is fully satisfied.
                    if let Some(ref bind_group) = self.uniform_bind_group {
                        if let Some(ref empty) = self.empty_bind_group {
                            for i in 0..self.uniform_bind_group_set {
                                rpass.set_bind_group(i, empty, &[]);
                            }
                        }
                        rpass.set_bind_group(self.uniform_bind_group_set, bind_group, &[]);
                    }
                    rpass.draw(0..3, 0..1);
                }

                self.queue.submit(Some(encoder.finish()));
                frame.present();

                // Wait for GPU to finish to get accurate frame timing (only when verbose)
                if let Some(frame_start) = frame_start {
                    let _ = self.device.poll(wgpu::PollType::Wait);

                    let frame_time_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
                    self.frame_times[self.frame_time_idx] = frame_time_ms;
                    self.frame_time_idx += 1;
                    if self.frame_time_idx >= 60 {
                        let avg = self.frame_times.iter().sum::<f64>() / 60.0;
                        let min = self.frame_times.iter().cloned().fold(f64::INFINITY, f64::min);
                        let max = self.frame_times.iter().cloned().fold(0.0, f64::max);
                        let fps = 1000.0 / avg;
                        eprintln!(
                            "[Frame {}] avg: {:.2}ms, min: {:.2}ms, max: {:.2}ms, fps: {:.1}",
                            self.frame_count, avg, min, max, fps
                        );
                        self.frame_time_idx = 0;
                    }
                }

                // Check frame limit if set
                if let Some(max) = self.max_frames {
                    if self.frame_count >= max {
                        eprintln!("Reached {} frames, exiting.", max);
                        self.print_memory_stats("exit");
                        std::process::exit(0);
                    }
                }
            }
            Err(e @ wgpu::SurfaceError::Lost) | Err(e @ wgpu::SurfaceError::Outdated) => {
                eprintln!("surface {e}; reconfiguring");
                let size = self.window.inner_size();
                if size.width > 0 && size.height > 0 {
                    self.config.width = size.width;
                    self.config.height = size.height;
                }
                self.surface.configure(&self.device, &self.config);
            }
            Err(wgpu::SurfaceError::Timeout) => {
                eprintln!("surface timeout; skipping frame");
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                eprintln!("GPU out of memory; exiting");
                std::process::exit(1);
            }
            Err(wgpu::SurfaceError::Other) => {
                // Non-fatal miscellaneous error; skip this frame.
                eprintln!("surface error: Other; skipping frame");
            }
        }
    }
}

pub struct App {
    state: Option<State>,
    spec: PipelineSpec,
}

impl App {
    pub fn new(spec: PipelineSpec) -> Self {
        Self { state: None, spec }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_size = self.spec.size;
        let mut attrs = WindowAttributes::default().with_title("wgpu + SPIR-V");
        if let Some((w, h)) = window_size {
            attrs = attrs.with_inner_size(PhysicalSize::new(w, h));
        }
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("failed to create window: {e}");
                std::process::exit(1);
            }
        };

        match pollster::block_on(State::new(window, &self.spec)) {
            Ok(state) => {
                state.print_memory_stats("startup");
                self.state = Some(state);
            }
            Err(e) => {
                eprintln!("failed to initialize GPU state: {e:#}");
                std::process::exit(1);
            }
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let Some(state) = &mut self.state {
            if state.window.id() == window_id {
                match event {
                    WindowEvent::CloseRequested => {
                        state.print_memory_stats("exit");
                        std::process::exit(0);
                    }
                    WindowEvent::Resized(size) => state.resize(size),
                    WindowEvent::RedrawRequested => state.render(),
                    WindowEvent::ScaleFactorChanged {
                        scale_factor: _,
                        mut inner_size_writer,
                    } => {
                        // Request a size; a `Resized` will follow.
                        let _ = inner_size_writer.request_inner_size(state.window.inner_size());
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        state.mouse_pos = [position.x as f32, position.y as f32];
                    }
                    WindowEvent::MouseInput {
                        state: button_state,
                        button,
                        ..
                    } => {
                        use winit::event::{ElementState, MouseButton};
                        if button == MouseButton::Left {
                            match button_state {
                                ElementState::Pressed => {
                                    state.mouse_pressed = true;
                                    state.mouse_click_pos = state.mouse_pos;
                                }
                                ElementState::Released => {
                                    state.mouse_pressed = false;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &mut self.state {
            state.window.request_redraw();
        }
    }
}
