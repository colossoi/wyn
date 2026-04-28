//! Interactive winit + wgpu application shell shared by the
//! `vertex-fragment` and `testpattern` modes. `PipelineSpec` describes
//! what to render; `State` owns the surface + pipeline; `App` is the
//! `winit::ApplicationHandler` wrapper.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages, Color,
    ColorTargetState, CommandEncoderDescriptor, DeviceDescriptor, FragmentState, Instance,
    InstanceDescriptor, InstanceFlags, LoadOp, MultisampleState, Operations,
    PipelineLayoutDescriptor, PowerPreference, PresentMode, PrimitiveState, RenderPipeline,
    RequestAdapterOptions, ShaderStages, StoreOp, SurfaceConfiguration, TextureUsages, Trace,
    VertexState,
};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

use crate::json::{Binding, Pipeline, PipelineDescriptor};
use crate::spirv::load_spirv_module;


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

/// Single shadertoy-style uniform binding declared in a wyn pipeline
/// JSON sidecar (next to the .spv module). The interactive viewer reads
/// the sidecar to discover where the SPIR-V expects iResolution / iTime
/// / iMouse / etc. to be bound.
#[derive(Debug, Clone)]
struct UniformDecl {
    set: u32,
    binding: u32,
    name: String,
}

/// Try to read `<spv_path>.json` (or `<spv_path_without_ext>.json`) and
/// extract the uniforms declared by the first graphics pipeline. Returns
/// an empty vec if no sidecar is present.
fn load_sidecar_uniforms(spv_path: &Path) -> Vec<UniformDecl> {
    let json_path = spv_path.with_extension("json");
    let Ok(content) = fs::read_to_string(&json_path) else {
        return Vec::new();
    };
    let Ok(desc) = serde_json::from_str::<PipelineDescriptor>(&content) else {
        return Vec::new();
    };
    for p in &desc.pipelines {
        if let Pipeline::Graphics(g) = p {
            return g
                .bindings
                .iter()
                .filter_map(|b| {
                    if let Binding::Uniform { set, binding, name } = b {
                        Some(UniformDecl {
                            set: *set,
                            binding: *binding,
                            name: name.clone(),
                        })
                    } else {
                        None
                    }
                })
                .collect();
        }
    }
    Vec::new()
}

// Uniform buffers - one per shader uniform
// iResolution: [3]f32 (x = width, y = height, z = device pixel ratio, default 1.0)
// + 1 f32 padding for 16-byte std140 alignment of vec3.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ResolutionUniform {
    resolution: [f32; 3],
    _pad: f32,
}

// iTime: f32
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TimeUniform {
    time: f32,
}

// iMouse: vec4f32 (x, y, click_x, click_y)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MouseUniform {
    mouse: [f32; 4],
}

// difficulty: i32 (binding 2)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DifficultyUniform {
    difficulty: i32,
    _pad: [i32; 3], // Pad to 16 bytes for alignment
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
        // Extract validation flag from spec (validation is ON by default)
        let validate = spec.validate;

        // Create instance with validation layers (enabled by default)
        let instance_flags = if validate {
            eprintln!("[viz] Validation layers ENABLED");
            InstanceFlags::VALIDATION | InstanceFlags::DEBUG
        } else {
            eprintln!("[viz] Validation layers DISABLED");
            InstanceFlags::empty()
        };

        let instance = Instance::new(&InstanceDescriptor {
            flags: instance_flags,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).context("failed to create wgpu surface")?;

        // v26: returns Result<Adapter, RequestAdapterError>
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("request_adapter failed")?;

        // Extract verbose flag from spec
        let verbose = spec.verbose;

        // Print adapter info when verbose
        if verbose {
            let info = adapter.get_info();
            eprintln!("[viz] Adapter: {} ({:?})", info.name, info.backend);
            eprintln!("[viz] Driver: {}", info.driver);
            eprintln!("[viz] Driver info: {}", info.driver_info);
        }

        // Check if SPIRV_SHADER_PASSTHROUGH is supported
        let adapter_features = adapter.features();
        let spirv_passthrough_supported =
            adapter_features.contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH);

        if verbose {
            eprintln!(
                "[viz] SPIRV_SHADER_PASSTHROUGH supported: {}",
                spirv_passthrough_supported
            );
        }

        // Build required features
        let mut required_features = wgpu::Features::empty();
        if spirv_passthrough_supported {
            required_features |= wgpu::Features::SPIRV_SHADER_PASSTHROUGH;
        }

        // v26: request_device takes a single descriptor; trace is in the descriptor
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: None,
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: Trace::Off,
            })
            .await
            .context("failed to create logical device")?;

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

        let caps = surface.get_capabilities(&adapter);
        // Prefer a non-sRGB 8-bit format so the GPU doesn't re-encode on
        // write — shadertoy-style shaders apply their own gamma via
        // pow(col, 0.45) and expect the framebuffer to treat their output
        // as final pixels. Higher-precision formats (Rgba16Float) would
        // halve or quarter throughput without fixing the main issue
        // (contour banding in the shader math, not quantization).
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .or_else(|| caps.formats.get(0).copied())
            .ok_or_else(|| anyhow!("surface reports no supported formats"))?;
        let size = window.inner_size();

        // Extract present mode from spec (default to Fifo for TestPattern)
        let present_mode = spec.present_mode;

        if verbose {
            eprintln!("[viz] Present mode: {:?}", present_mode);
        }

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: caps
                .alpha_modes
                .get(0)
                .copied()
                .ok_or_else(|| anyhow!("surface reports no alpha modes"))?,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // === Conditionally create uniform buffers based on spec ======
        // With --shadertoy we provide iResolution/iTime/iMouse/difficulty to
        // the shader. The (set, binding) for each uniform comes from the
        // sidecar pipeline descriptor (`<spv>.json`) when present, so the
        // layout matches what the shader actually declares. We fall back to
        // the canonical Shadertoy positions (set=0, bindings=[0,1,2,5]) when
        // no sidecar is available.
        let (
            resolution_buffer,
            time_buffer,
            mouse_buffer,
            difficulty_buffer,
            uniform_bind_group,
            uniform_bind_group_layout,
            uniform_bind_group_set,
        ) = if let (true, Shader::Spirv(path)) = (spec.shadertoy, &spec.shader) {
            let difficulty = &spec.difficulty;
            // Map each well-known Shadertoy name to a slot on its preferred
            // (legacy) binding. The sidecar, if present, overrides the
            // binding and set number to match the shader's declarations.
            struct Slot {
                default_binding: u32,
                actual_set: u32,
                actual_binding: u32,
                present: bool,
            }
            let mut resolution = Slot {
                default_binding: 0,
                actual_set: 0,
                actual_binding: 0,
                present: false,
            };
            let mut time = Slot {
                default_binding: 1,
                actual_set: 0,
                actual_binding: 1,
                present: false,
            };
            let mut difficulty_slot = Slot {
                default_binding: 2,
                actual_set: 0,
                actual_binding: 2,
                present: false,
            };
            let mut mouse = Slot {
                default_binding: 5,
                actual_set: 0,
                actual_binding: 5,
                present: false,
            };

            let sidecar = load_sidecar_uniforms(path);
            if sidecar.is_empty() {
                return Err(anyhow!(
                    "viz vf --shadertoy: no sidecar pipeline descriptor found next to {:?}. \
                     Recompile the shader with `wyn compile` to emit the `.json` alongside the `.spv`.",
                    path
                ));
            }
            // Descriptor-driven: only allocate slots the shader declares.
            // All uniforms are assumed to live in the same set; mixed sets
            // would need per-set bind groups (not supported yet).
            for u in &sidecar {
                let slot = match u.name.as_str() {
                    "iResolution" => Some(&mut resolution),
                    "iTime" => Some(&mut time),
                    "difficulty" | "iDifficulty" => Some(&mut difficulty_slot),
                    "iMouse" => Some(&mut mouse),
                    _ => None,
                };
                if let Some(s) = slot {
                    s.actual_set = u.set;
                    s.actual_binding = u.binding;
                    s.present = true;
                }
            }

            // All declared uniforms must share one set.
            let present_sets: Vec<u32> = [&resolution, &time, &difficulty_slot, &mouse]
                .iter()
                .filter(|s| s.present)
                .map(|s| s.actual_set)
                .collect();
            let bind_group_set = present_sets.first().copied().unwrap_or(0);
            if present_sets.iter().any(|s| *s != bind_group_set) {
                return Err(anyhow!(
                    "viz vf: shadertoy uniforms are split across multiple descriptor sets; \
                     only a single set is currently supported"
                ));
            }

            let make_buffer = |label: &'static str, size: u64| {
                device.create_buffer(&BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };

            let resolution_buffer = resolution.present.then(|| {
                let buf = make_buffer(
                    "resolution_buffer",
                    std::mem::size_of::<ResolutionUniform>() as u64,
                );
                let initial = ResolutionUniform {
                    resolution: [800.0, 600.0, 1.0],
                    _pad: 0.0,
                };
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[initial]));
                buf
            });
            let time_buffer = time.present.then(|| {
                let buf = make_buffer("time_buffer", std::mem::size_of::<TimeUniform>() as u64);
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[TimeUniform { time: 0.0 }]));
                buf
            });
            let difficulty_buffer = difficulty_slot.present.then(|| {
                let buf = make_buffer(
                    "difficulty_buffer",
                    std::mem::size_of::<DifficultyUniform>() as u64,
                );
                let initial = DifficultyUniform {
                    difficulty: *difficulty,
                    _pad: [0, 0, 0],
                };
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[initial]));
                buf
            });
            let mouse_buffer = mouse.present.then(|| {
                let buf = make_buffer("mouse_buffer", std::mem::size_of::<MouseUniform>() as u64);
                queue.write_buffer(
                    &buf,
                    0,
                    bytemuck::cast_slice(&[MouseUniform {
                        mouse: [0.0, 0.0, 0.0, 0.0],
                    }]),
                );
                buf
            });

            let buf_ty = BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            };
            let buf_layout = |binding: u32| BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: buf_ty,
                count: None,
            };
            let slots: Vec<(u32, &wgpu::Buffer)> = [
                resolution_buffer.as_ref().map(|b| (resolution.actual_binding, b)),
                time_buffer.as_ref().map(|b| (time.actual_binding, b)),
                difficulty_buffer.as_ref().map(|b| (difficulty_slot.actual_binding, b)),
                mouse_buffer.as_ref().map(|b| (mouse.actual_binding, b)),
            ]
            .into_iter()
            .flatten()
            .collect();
            let layout_entries: Vec<BindGroupLayoutEntry> =
                slots.iter().map(|(binding, _)| buf_layout(*binding)).collect();
            let group_entries: Vec<BindGroupEntry> = slots
                .iter()
                .map(|(binding, buffer)| BindGroupEntry {
                    binding: *binding,
                    resource: BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: 0,
                        size: None,
                    }),
                })
                .collect();

            let uniform_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("uniform_bind_group_layout"),
                entries: &layout_entries,
            });
            let uniform_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("uniform_bind_group"),
                layout: &uniform_bind_group_layout,
                entries: &group_entries,
            });

            (
                resolution_buffer,
                time_buffer,
                mouse_buffer,
                difficulty_buffer,
                Some(uniform_bind_group),
                Some(uniform_bind_group_layout),
                bind_group_set,
            )
        } else {
            (None, None, None, None, None, None, 0)
        };

        // If the uniform bind group lives at set > 0, we also need an empty
        // bind group to satisfy wgpu's contiguous-set layout rule.
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

        // Make these mutable so TestPattern can set them
        let mut resolution_buffer = resolution_buffer;
        let mut uniform_bind_group = uniform_bind_group;

        let max_frames = spec.max_frames;

        // === Build pipeline from the chosen shader source ====================
        let pipeline = match &spec.shader {
            Shader::Spirv(path) => {
                let vertex = &spec.vertex_entry;
                let fragment = &spec.fragment_entry;
                let shadertoy = &spec.shadertoy;
                let module = load_spirv_module(&device, path)
                    .with_context(|| format!("load SPIR-V module {:?}", path))?;

                // Build the bind group layout list. The uniform layout (if
                // any) lives at `uniform_bind_group_set`; wgpu requires set
                // numbers to be contiguous, so we prefix the shared empty
                // layout for any lower sets.
                let mut bind_group_layouts_vec: Vec<&wgpu::BindGroupLayout> = vec![];
                if *shadertoy {
                    if let Some(ref ubl) = uniform_bind_group_layout {
                        if let Some(ref el) = empty_bind_group_layout {
                            for _ in 0..uniform_bind_group_set {
                                bind_group_layouts_vec.push(el);
                            }
                        }
                        bind_group_layouts_vec.push(ubl);
                    }
                }

                let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("layout"),
                    bind_group_layouts: &bind_group_layouts_vec,
                    push_constant_ranges: &[],
                });

                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("pipeline"),
                    layout: Some(&layout),
                    vertex: VertexState {
                        module: &module,
                        entry_point: Some(vertex.as_str()),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(FragmentState {
                        module: &module,
                        entry_point: Some(fragment.as_str()),
                        targets: &[Some(ColorTargetState {
                            format: config.format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: MultisampleState::default(),
                    multiview: None,
                    cache: None,
                })
            }
            Shader::Wgsl(shader_source) => {
                eprintln!("[viz] Loading built-in test pattern shader (WGSL)");

                // Create resolution uniform buffer for test pattern
                let res_buffer = device.create_buffer(&BufferDescriptor {
                    label: Some("test_pattern_resolution"),
                    size: std::mem::size_of::<ResolutionUniform>() as u64,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let initial_res = ResolutionUniform {
                    resolution: [config.width as f32, config.height as f32, 1.0],
                    _pad: 0.0,
                };
                queue.write_buffer(&res_buffer, 0, bytemuck::cast_slice(&[initial_res]));

                let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("test_pattern_bind_group_layout"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

                let bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("test_pattern_bind_group"),
                    layout: &bind_group_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &res_buffer,
                            offset: 0,
                            size: None,
                        }),
                    }],
                });

                // Store these for later use - reuse the existing fields
                resolution_buffer = Some(res_buffer);
                uniform_bind_group = Some(bind_group);

                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("test_pattern_shader"),
                    source: wgpu::ShaderSource::Wgsl((*shader_source).into()),
                });

                let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("test_pattern_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("test_pattern_pipeline"),
                    layout: Some(&layout),
                    vertex: VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        targets: &[Some(ColorTargetState {
                            format: config.format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: MultisampleState::default(),
                    multiview: None,
                    cache: None,
                })
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
