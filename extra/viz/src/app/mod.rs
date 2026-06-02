//! Interactive winit + wgpu application shell shared by the
//! `vertex-fragment` and `testpattern` modes. `PipelineSpec` describes
//! what to render; `State` owns the surface + pipeline; `App` is the
//! `winit::ApplicationHandler` wrapper.

mod render;
mod uniforms;
mod vertex_buffers;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, Color, CommandEncoderDescriptor, Extent3d,
    InstanceFlags, LoadOp, Operations, PresentMode, RenderPipeline, StoreOp, SurfaceConfiguration,
    TextureDescriptor, TextureDimension, TextureUsages, TextureView, TextureViewDescriptor,
};

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

use crate::gpu::{self, DeviceRequest, GpuContext};
use crate::modes::simulate::BindingLoc;
use render::DEPTH_FORMAT;
use uniforms::{MouseUniform, ResolutionUniform, TimeUniform};

use wyn_pipeline_descriptor::{ComputePipeline, GraphicsPipeline, PipelineDescriptor};

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
    /// Number of vertex-shader invocations per draw (`rpass.draw(0..vertex_count, 0..1)`).
    pub vertex_count: u32,
    /// Primitive topology for the draw call.
    pub topology: wgpu::PrimitiveTopology,
    /// Optional directory of per-binding `.bin` files uploaded into the
    /// shader's `storage_buffer` bindings. Requires `--shadertoy`
    /// (the storage buffers share its bind group).
    pub storage_dir: Option<PathBuf>,
    /// Optional flat little-endian u32 index buffer. When present, viz
    /// uses `draw_indexed(0..(file_size/4))`; when absent, falls back
    /// to non-indexed `draw(0..vertex_count)`.
    pub index_buffer: Option<PathBuf>,
}

/// Configuration for the `simulate` viewer (compute + ping-pong + render).
/// Built by `modes::simulate::run_simulate` after parsing the SPIR-V's
/// JSON sidecar and resolving the ping-pong / display bindings;
/// consumed by `State::new_simulate` to construct GPU state.
pub struct SimulateSpec {
    pub shader_path: PathBuf,
    pub compute_entry: String,
    pub vertex_entry: String,
    pub fragment_entry: String,
    /// Grid width × height. Drives compute push-constants
    /// (`width`/`height`) and the fragment's `grid_width`/`grid_height`
    /// uniforms.
    pub grid: (u32, u32),
    /// Initial board state, row-major, `grid.0 * grid.1` `i32`s.
    pub initial_board: Vec<i32>,
    /// Resolved compute pipeline metadata (workgroup size, bindings).
    pub compute_pipeline: ComputePipeline,
    /// Resolved graphics pipeline metadata (bindings).
    pub graphics_pipeline: GraphicsPipeline,
    /// Compute pipeline's read-only storage input (the ping-pong
    /// "input" slot on the compute side).
    pub input_binding: BindingLoc,
    /// Compute pipeline's write-only storage output (the ping-pong
    /// "output" slot on the compute side).
    pub output_binding: BindingLoc,
    /// Graphics pipeline's read-only storage input — the same physical
    /// buffer the compute just wrote.
    pub display_binding: BindingLoc,
    // Generic viewer knobs, mirroring `PipelineSpec`'s shared fields.
    pub max_frames: Option<u32>,
    pub verbose: bool,
    pub validate: bool,
    pub present_mode: PresentMode,
    pub size: Option<(u32, u32)>,
    pub vertex_count: u32,
}

/// Configuration for the interactive shape of `pipeline` mode: an
/// arbitrary chain of compute pipelines plus a graphics pipeline that
/// renders to a window, all driven from the descriptor JSON. Built by
/// `modes::pipeline::run_pipeline_interactive` after detecting a
/// graphics pipeline in the descriptor.
///
/// This is `pipeline` mode's interactive variant — same descriptor
/// schema as the headless path, plus the windowing knobs `vf` and
/// `simulate` carry.
pub struct InteractivePipelineSpec {
    pub shader_path: PathBuf,
    pub descriptor: PipelineDescriptor,
    /// Resolved vertex stage entry-point name.
    pub vertex_entry: String,
    /// Resolved fragment stage entry-point name.
    pub fragment_entry: String,
    pub max_frames: Option<u32>,
    pub verbose: bool,
    pub validate: bool,
    pub present_mode: PresentMode,
    pub size: Option<(u32, u32)>,
    pub vertex_count: u32,
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
    start_time: std::time::Instant,
    // Mouse tracking (consumed by vf's shadertoy iMouse uniform; harmless
    // for other modes that don't read it).
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
    /// Per-frame depth attachment, recreated on resize. Sized to
    /// `config.width × config.height`, format `DEPTH_FORMAT`.
    depth_view: TextureView,
    /// Mode-specific GPU state. The `App` constructor picks one
    /// variant; the rest of the per-frame path dispatches on this.
    mode: AppMode,
}

/// Mode-specific GPU state held by `State`. Each variant owns the
/// pipelines, bind groups, and buffers that variant's `render` path
/// consumes; common surface / device / window state stays in `State`
/// itself.
enum AppMode {
    VertexFragment(VfState),
    Simulate(SimulateState),
    /// Interactive `pipeline` mode: descriptor-driven N compute stages
    /// then one graphics pipeline per frame. Cross-stage data flows
    /// through storage buffers, storage textures (compute-writes
    /// readable as fragment-sampled textures via the same wgpu::
    /// Texture allocated once at startup), and uniforms.
    Pipeline(PipelineState),
}

/// One compute stage in a `PipelineState` chain. Mirrors a single
/// `ComputePipeline` in the descriptor.
struct PipelineComputeStage {
    pipeline: wgpu::ComputePipeline,
    /// Bind groups indexed by descriptor-set number. Empty sets below
    /// the lowest used one get a no-op group attached at render time
    /// to satisfy wgpu's contiguous-sets requirement.
    bind_groups_by_set: Vec<Option<BindGroup>>,
    workgroups: (u32, u32, u32),
    push_constants: Vec<u8>,
    label: String,
}

/// GPU state for `AppMode::Pipeline`. Holds every pipeline + bind
/// group needed to drive the descriptor each frame.
struct PipelineState {
    compute_stages: Vec<PipelineComputeStage>,
    render_pipeline: RenderPipeline,
    /// Render-side bind groups indexed by descriptor-set number.
    render_bind_groups_by_set: Vec<Option<BindGroup>>,
    vertex_count: u32,
    // Shadertoy-style uniform buffers, populated per frame when the
    // graphics pipeline declares the matching uniform binding.
    resolution_buffer: Option<wgpu::Buffer>,
    time_buffer: Option<wgpu::Buffer>,
    mouse_buffer: Option<wgpu::Buffer>,
    frame_buffer: Option<wgpu::Buffer>,
    // Storage buffers + textures + samplers — held to keep the GPU
    // resources alive for the bind groups' lifetime.
    _storage_buffers: HashMap<u32, (wgpu::Buffer, u64)>,
    _storage_textures: HashMap<(u32, u32), gpu::StorageTextureResource>,
    _samplers: HashMap<(u32, u32), wgpu::Sampler>,
}

/// GPU state for `simulate`: a compute pipeline that ping-pongs
/// between two storage buffers + a render pipeline that reads
/// whichever buffer was most-recently-written. Each frame selects a
/// bind group by `frame_count % 2`.
struct SimulateState {
    compute_pipeline: wgpu::ComputePipeline,
    /// `[direction_for_even_frames, direction_for_odd_frames]`. Frame
    /// N's compute pass binds `compute_bind_groups[N % 2]`, which is
    /// pre-wired with (input=buf_X, output=buf_Y).
    compute_bind_groups: [BindGroup; 2],
    /// Packed push-constant block (e.g. `[width:i32, height:i32]` for
    /// Conway). Empty if the compute pipeline declares no push
    /// constants.
    compute_push_constants: Vec<u8>,
    /// `(x, y, z)` workgroup count for the compute dispatch.
    compute_workgroups: [u32; 3],
    render_pipeline: RenderPipeline,
    /// `[group_for_even_frames, group_for_odd_frames]`, mirroring
    /// `compute_bind_groups`: at frame N, the fragment reads the
    /// buffer the compute just wrote, so `render_bind_groups[N % 2]`
    /// is wired with `display=output_of_compute_N`.
    render_bind_groups: [BindGroup; 2],
    /// Descriptor-set index at which `render_bind_groups[..]` is
    /// bound (the graphics pipeline's bindings set — set 1 for Conway).
    render_bind_group_set: u32,
    /// Empty bind group, bound at every set index below
    /// `render_bind_group_set` to satisfy wgpu's contiguous-sets
    /// requirement.
    empty_bind_group: Option<BindGroup>,
    /// `iResolution` buffer (rewritten per frame from
    /// `config.width`/`config.height`). `None` when the graphics
    /// pipeline doesn't declare `iResolution`.
    resolution_buffer: Option<wgpu::Buffer>,
    vertex_count: u32,
    // Held to keep alive; not directly accessed in `render_simulate`.
    _ping_pong_buffers: [wgpu::Buffer; 2],
    _grid_width_buffer: Option<wgpu::Buffer>,
    _grid_height_buffer: Option<wgpu::Buffer>,
}

/// GPU state for the `vf` / `testpattern` interactive viewers. Holds
/// exactly one render pipeline + (optionally) shadertoy uniforms +
/// vertex / index / storage buffers.
struct VfState {
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
    /// Number of vertex-shader invocations per draw call.
    vertex_count: u32,
    /// Host-uploaded storage buffers; held only to keep them alive for
    /// the bind group's lifetime.
    _storage_buffers: Vec<wgpu::Buffer>,
    /// Per-attribute vertex buffers; bound in order via
    /// `set_vertex_buffer` each frame.
    vertex_buffers: Vec<wgpu::Buffer>,
    /// Optional u32 index buffer + its element count; when `Some`,
    /// render() uses `draw_indexed` and `vertex_count` is ignored.
    index_buffer: Option<(wgpu::Buffer, u32)>,
}

/// Allocate a depth texture matching `config.width × config.height` and
/// return a default view. Called at startup and on every resize.
fn create_depth_view(device: &wgpu::Device, config: &SurfaceConfiguration) -> TextureView {
    let tex = device.create_texture(&TextureDescriptor {
        label: Some("depth"),
        size: Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&TextureViewDescriptor::default())
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
            Some(uniforms::build_shadertoy(
                &device,
                &queue,
                path,
                spec.difficulty,
                spec.storage_dir.as_deref(),
            )?)
        } else {
            None
        };
        let uniform_bind_group_set = shadertoy.as_ref().map(|s| s.bind_group_set).unwrap_or(0);

        // Vertex buffers from the sidecar's `vertex_inputs`. Loaded
        // from the same `--storage-dir` as the storage buffers, one
        // `.bin` per attribute. Empty for shaders without
        // `#[location(n)]` vertex inputs.
        let vertex_buffers = match &spec.shader {
            Shader::Spirv(path) => {
                vertex_buffers::build_vertex_buffers(&device, &queue, path, spec.storage_dir.as_deref())?
            }
            Shader::Wgsl(_) => vertex_buffers::VertexBuffers::empty(),
        };

        // Optional index buffer — flat little-endian u32. Indexed draws
        // when present; non-indexed `draw(0..vertex_count)` otherwise.
        let index_buffer = if let Some(path) = spec.index_buffer.as_deref() {
            let data = std::fs::read(path)
                .with_context(|| format!("viz vf --index-buffer: reading {:?}", path))?;
            if data.len() % 4 != 0 {
                return Err(anyhow::anyhow!(
                    "viz vf --index-buffer: {:?} is {} bytes, not a multiple of 4 (u32 indices)",
                    path,
                    data.len(),
                ));
            }
            let count = (data.len() / 4) as u32;
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("index_buffer"),
                size: data.len() as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, &data);
            Some((buf, count))
        } else {
            None
        };

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

        let (
            pipeline,
            resolution_buffer,
            time_buffer,
            mouse_buffer,
            difficulty_buffer,
            storage_buffers,
            uniform_bind_group,
        ) = match &spec.shader {
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
                    spec.topology,
                    &vertex_buffers.attribs,
                )?;
                let (rb, tb, mb, db, sb, bg) = match shadertoy {
                    Some(s) => (
                        s.resolution_buffer,
                        s.time_buffer,
                        s.mouse_buffer,
                        s.difficulty_buffer,
                        s.storage_buffers,
                        Some(s.bind_group),
                    ),
                    None => (None, None, None, None, Vec::new(), None),
                };
                (pipeline, rb, tb, mb, db, sb, bg)
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
                (
                    pipeline,
                    Some(res_buffer),
                    None,
                    None,
                    None,
                    Vec::new(),
                    Some(bind_group),
                )
            }
        };

        let now = std::time::Instant::now();
        let depth_view = create_depth_view(&device, &config);

        let vf = VfState {
            pipeline,
            resolution_buffer,
            time_buffer,
            mouse_buffer,
            _difficulty_buffer: difficulty_buffer,
            uniform_bind_group,
            uniform_bind_group_set,
            empty_bind_group,
            vertex_count: spec.vertex_count,
            _storage_buffers: storage_buffers,
            vertex_buffers: vertex_buffers.buffers,
            index_buffer,
        };

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            start_time: now,
            mouse_pos: [0.0, 0.0],
            mouse_click_pos: [0.0, 0.0],
            mouse_pressed: false,
            frame_count: 0,
            max_frames,
            frame_times: [0.0; 60],
            frame_time_idx: 0,
            verbose,
            depth_view,
            mode: AppMode::VertexFragment(vf),
        })
    }

    /// Construct GPU state for the interactive `pipeline` mode: a
    /// descriptor-driven compute chain + one graphics pipeline. See
    /// `InteractivePipelineSpec` for the descriptor source. Filled in
    /// incrementally; the current implementation handles only the
    /// minimal `storage_image_roundtrip` shape (one compute writing a
    /// storage texture + one graphics pipeline sampling it). Storage
    /// buffers, push constants, and Shadertoy-style uniforms land in
    /// follow-up commits.
    async fn new_pipeline(window: Arc<Window>, spec: &InteractivePipelineSpec) -> Result<Self> {
        let _ = window;
        let _ = spec;
        anyhow::bail!(
            "interactive pipeline mode: GPU state construction not yet implemented \
             (skeleton committed; coming in the next commit)"
        )
    }

    async fn new_simulate(window: Arc<Window>, spec: &SimulateSpec) -> Result<Self> {
        use wgpu::util::DeviceExt;
        use wyn_pipeline_descriptor::Binding;

        let validate = spec.validate;
        let verbose = spec.verbose;
        let present_mode = spec.present_mode;

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
            // SPIRV passthrough so the compute + graphics SPIR-V loads
            // raw, and PUSH_CONSTANTS for the compute's width/height
            // (and any future scalar push consts).
            desired_features: wgpu::Features::SPIRV_SHADER_PASSTHROUGH | wgpu::Features::PUSH_CONSTANTS,
            // Conway pushes 8 bytes; 128 is well below the typical
            // adapter max but covers most simulate shaders.
            limits_overlay: Some(Box::new(|limits, _adapter| {
                limits.max_push_constant_size = 128;
            })),
            surface_target: Some(Box::new(move |inst| {
                inst.create_surface(win).context("failed to create wgpu surface")
            })),
        })
        .await?;

        let surface = ctx.surface.expect("surface_target was Some, surface must be present");
        let adapter = ctx.adapter;
        let device = ctx.device;
        let queue = ctx.queue;

        if verbose {
            let info = adapter.get_info();
            eprintln!("[viz] Adapter: {} ({:?})", info.name, info.backend);
        }
        device.on_uncaptured_error(Box::new(|error| {
            eprintln!("\n[GPU validation error]\n{:?}\n", error);
        }));
        device.set_device_lost_callback(|reason, message| {
            eprintln!("\n[GPU device lost]\nReason: {:?}\nMessage: {}", reason, message);
            std::process::exit(1);
        });

        let size = window.inner_size();
        let config =
            render::configure_surface(&surface, &device, &adapter, size.width, size.height, present_mode)?;

        // --- Ping-pong storage buffers -------------------------------------
        let cells = spec.grid.0 as u64 * spec.grid.1 as u64;
        let buf_size = cells * 4;
        let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("simulate.board_a"),
            contents: bytemuck::cast_slice(&spec.initial_board),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulate.board_b"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Graphics uniforms (iResolution + grid_width + grid_height) ----
        // Walk the graphics pipeline's declared uniforms; for each known
        // name, allocate the right buffer. Unknown names error loudly.
        let (resolution_buffer, grid_width_buffer, grid_height_buffer, time_buffer, uniform_entries) =
            uniforms::build_simulate_uniforms(&device, &queue, &spec.graphics_pipeline, spec.grid)?;

        // --- Compute bind-group layout (set 0 in conway): input + output ---
        let compute_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("simulate.compute.bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: spec.input_binding.binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: spec.output_binding.binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let make_compute_bg = |label: &str, input: &wgpu::Buffer, output: &wgpu::Buffer| {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some(label),
                layout: &compute_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: spec.input_binding.binding,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: spec.output_binding.binding,
                        resource: output.as_entire_binding(),
                    },
                ],
            })
        };
        let compute_bind_groups = [
            // frame_count % 2 == 0: read buf_b, write buf_a
            make_compute_bg("simulate.compute.bg.b_to_a", &buf_b, &buf_a),
            // frame_count % 2 == 1: read buf_a, write buf_b (initial state lives in buf_a)
            make_compute_bg("simulate.compute.bg.a_to_b", &buf_a, &buf_b),
        ];

        // --- Compute push-constant block (e.g. Conway's width/height) ------
        let mut compute_pc = Vec::new();
        for b in &spec.compute_pipeline.bindings {
            if let Binding::PushConstant { name, size, .. } = b {
                let value = match name.as_str() {
                    "width" => spec.grid.0 as i32,
                    "height" => spec.grid.1 as i32,
                    other => {
                        return Err(anyhow!(
                            "simulate: compute pipeline declares unknown push constant {:?}; \
                             known: width, height",
                            other
                        ));
                    }
                };
                if *size != 4 {
                    return Err(anyhow!(
                        "simulate: push constant {:?} has unexpected size {} (expected 4)",
                        name,
                        size
                    ));
                }
                compute_pc.extend_from_slice(&value.to_le_bytes());
            }
        }

        // --- Compute pipeline ----------------------------------------------
        let compute_module = crate::spirv::load_spirv_module(&device, &spec.shader_path)
            .with_context(|| format!("load SPIR-V {:?}", spec.shader_path))?;

        let push_ranges = if compute_pc.is_empty() {
            Vec::new()
        } else {
            vec![wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..compute_pc.len() as u32,
            }]
        };
        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("simulate.compute.layout"),
            bind_group_layouts: &[&compute_bgl],
            push_constant_ranges: &push_ranges,
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("simulate.compute.pipeline"),
            layout: Some(&compute_layout),
            module: &compute_module,
            entry_point: Some(&spec.compute_entry),
            compilation_options: Default::default(),
            cache: None,
        });

        let wg_x = spec.compute_pipeline.workgroup_size.0.max(1);
        let total_threads = (spec.grid.0 * spec.grid.1) as u32;
        let dispatch_x = total_threads.div_ceil(wg_x);
        let compute_workgroups = [dispatch_x, 1, 1];

        // --- Render bind-group layout (set 1 in conway): uniforms + storage
        let mut render_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = uniform_entries.clone();
        render_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: spec.display_binding.binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let render_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("simulate.render.bgl"),
            entries: &render_layout_entries,
        });

        let make_render_bg = |label: &str, display: &wgpu::Buffer| {
            let mut entries: Vec<wgpu::BindGroupEntry> = Vec::new();
            for b in &spec.graphics_pipeline.bindings {
                if let Binding::Uniform { binding, name, .. } = b {
                    let resource = match name.as_str() {
                        "iResolution" => resolution_buffer.as_ref().unwrap().as_entire_binding(),
                        "iTime" => time_buffer.as_ref().unwrap().as_entire_binding(),
                        "grid_width" => grid_width_buffer.as_ref().unwrap().as_entire_binding(),
                        "grid_height" => grid_height_buffer.as_ref().unwrap().as_entire_binding(),
                        _ => continue, // already errored in build_simulate_uniforms
                    };
                    entries.push(wgpu::BindGroupEntry {
                        binding: *binding,
                        resource,
                    });
                }
            }
            entries.push(wgpu::BindGroupEntry {
                binding: spec.display_binding.binding,
                resource: display.as_entire_binding(),
            });
            device.create_bind_group(&BindGroupDescriptor {
                label: Some(label),
                layout: &render_bgl,
                entries: &entries,
            })
        };
        let render_bind_groups = [
            // frame_count % 2 == 0: compute wrote buf_a → fragment reads buf_a
            make_render_bg("simulate.render.bg.reads_a", &buf_a),
            // frame_count % 2 == 1: compute wrote buf_b → fragment reads buf_b
            make_render_bg("simulate.render.bg.reads_b", &buf_b),
        ];

        // --- Render pipeline -----------------------------------------------
        let (empty_bgl, empty_bg) = if spec.display_binding.set > 0 {
            let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("simulate.empty.bgl"),
                entries: &[],
            });
            let group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("simulate.empty.bg"),
                layout: &layout,
                entries: &[],
            });
            (Some(layout), Some(group))
        } else {
            (None, None)
        };

        let render_pipeline = render::build_spirv_render_pipeline(
            &device,
            config.format,
            &spec.shader_path,
            &spec.vertex_entry,
            &spec.fragment_entry,
            Some((&render_bgl, empty_bgl.as_ref(), spec.display_binding.set)),
            wgpu::PrimitiveTopology::TriangleList,
            &[],
        )?;

        let depth_view = create_depth_view(&device, &config);
        let now = std::time::Instant::now();

        let sim = SimulateState {
            compute_pipeline,
            compute_bind_groups,
            compute_push_constants: compute_pc,
            compute_workgroups,
            render_pipeline,
            render_bind_groups,
            render_bind_group_set: spec.display_binding.set,
            empty_bind_group: empty_bg,
            resolution_buffer,
            vertex_count: spec.vertex_count,
            _ping_pong_buffers: [buf_a, buf_b],
            _grid_width_buffer: grid_width_buffer,
            _grid_height_buffer: grid_height_buffer,
        };

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            start_time: now,
            mouse_pos: [0.0, 0.0],
            mouse_click_pos: [0.0, 0.0],
            mouse_pressed: false,
            frame_count: 0,
            max_frames: spec.max_frames,
            frame_times: [0.0; 60],
            frame_time_idx: 0,
            verbose,
            depth_view,
            mode: AppMode::Simulate(sim),
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_view = create_depth_view(&self.device, &self.config);
        }
    }

    fn render(&mut self) {
        // Start frame timing (only when verbose)
        let frame_start = if self.verbose { Some(std::time::Instant::now()) } else { None };

        self.frame_count += 1;

        match self.surface.get_current_texture() {
            Ok(frame) => {
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

                match &self.mode {
                    AppMode::VertexFragment(vf) => render_vf(
                        vf,
                        &view,
                        &self.depth_view,
                        &self.queue,
                        &self.config,
                        self.start_time,
                        self.mouse_pos,
                        self.mouse_click_pos,
                        self.mouse_pressed,
                        &mut encoder,
                    ),
                    AppMode::Simulate(sim) => render_simulate(
                        sim,
                        &view,
                        &self.depth_view,
                        &self.queue,
                        &self.config,
                        self.frame_count,
                        &mut encoder,
                    ),
                    AppMode::Pipeline(state) => render_pipeline(
                        state,
                        &view,
                        &self.depth_view,
                        &self.queue,
                        &self.config,
                        self.start_time,
                        self.frame_count,
                        self.mouse_pos,
                        self.mouse_click_pos,
                        self.mouse_pressed,
                        &mut encoder,
                    ),
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

/// Per-frame VfState path: shadertoy uniform writes + single render pass.
/// Extracted from `State::render` so that the dispatch in `State::render`
/// stays small and other modes (e.g. `Simulate`) can slot in alongside.
#[allow(clippy::too_many_arguments)]
fn render_vf(
    vf: &VfState,
    view: &TextureView,
    depth_view: &TextureView,
    queue: &wgpu::Queue,
    config: &SurfaceConfiguration,
    start_time: std::time::Instant,
    mouse_pos: [f32; 2],
    mouse_click_pos: [f32; 2],
    mouse_pressed: bool,
    encoder: &mut wgpu::CommandEncoder,
) {
    // Update uniform data for this frame if Shadertoy mode is enabled
    if let (Some(ref resolution_buffer), Some(ref time_buffer)) = (&vf.resolution_buffer, &vf.time_buffer) {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(start_time).as_secs_f32();

        // Update iResolution
        let resolution = ResolutionUniform {
            resolution: [config.width as f32, config.height as f32, 1.0],
            _pad: 0.0,
        };
        queue.write_buffer(resolution_buffer, 0, bytemuck::cast_slice(&[resolution]));

        // Update iTime
        let time = TimeUniform { time: elapsed };
        queue.write_buffer(time_buffer, 0, bytemuck::cast_slice(&[time]));

        // Update iMouse
        if let Some(mouse_buffer) = &vf.mouse_buffer {
            // Shadertoy convention: (x, y, click_x, click_y)
            // click_x/click_y are negative when not pressed
            let mouse = MouseUniform {
                mouse: if mouse_pressed {
                    [mouse_pos[0], mouse_pos[1], mouse_click_pos[0], mouse_click_pos[1]]
                } else {
                    [mouse_pos[0], mouse_pos[1], -1.0, -1.0]
                },
            };
            queue.write_buffer(mouse_buffer, 0, bytemuck::cast_slice(&[mouse]));
        }
    }

    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color {
                    // Linear white — surface is non-sRGB, so clear
                    // values display directly. Matches the original
                    // masthead's background (and lets its fog math,
                    // ported below, fade distant geometry into it).
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                    a: 1.0,
                }),
                store: StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(Operations {
                load: LoadOp::Clear(1.0),
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        ..Default::default()
    });

    rpass.set_pipeline(&vf.pipeline);
    // Set uniform bind group if available, at the set number
    // declared by the shader (via sidecar descriptor). Fill
    // any lower sets with the shared empty bind group so the
    // pipeline layout is fully satisfied.
    if let Some(ref bind_group) = vf.uniform_bind_group {
        if let Some(ref empty) = vf.empty_bind_group {
            for i in 0..vf.uniform_bind_group_set {
                rpass.set_bind_group(i, empty, &[]);
            }
        }
        rpass.set_bind_group(vf.uniform_bind_group_set, bind_group, &[]);
    }
    for (slot, buf) in vf.vertex_buffers.iter().enumerate() {
        rpass.set_vertex_buffer(slot as u32, buf.slice(..));
    }
    match &vf.index_buffer {
        Some((buf, count)) => {
            rpass.set_index_buffer(buf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..*count, 0, 0..1);
        }
        None => {
            rpass.draw(0..vf.vertex_count, 0..1);
        }
    }
}

/// Per-frame simulate path: update iResolution, run the compute pass
/// (writing into one of the two ping-pong buffers), then the render
/// pass (reading the just-written buffer). Bind groups are pre-built
/// in pairs; selection by `frame_count % 2` matches each frame's
/// compute write to the same frame's fragment read.
fn render_simulate(
    sim: &SimulateState,
    view: &TextureView,
    depth_view: &TextureView,
    queue: &wgpu::Queue,
    config: &SurfaceConfiguration,
    frame_count: u32,
    encoder: &mut wgpu::CommandEncoder,
) {
    // Update iResolution from the current surface size if the fragment
    // shader declared it.
    if let Some(ref resolution_buffer) = sim.resolution_buffer {
        let resolution = ResolutionUniform {
            resolution: [config.width as f32, config.height as f32, 1.0],
            _pad: 0.0,
        };
        queue.write_buffer(resolution_buffer, 0, bytemuck::cast_slice(&[resolution]));
    }

    let bg_index = (frame_count as usize) % 2;

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("simulate.compute_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&sim.compute_pipeline);
        cpass.set_bind_group(0, &sim.compute_bind_groups[bg_index], &[]);
        if !sim.compute_push_constants.is_empty() {
            cpass.set_push_constants(0, &sim.compute_push_constants);
        }
        cpass.dispatch_workgroups(
            sim.compute_workgroups[0],
            sim.compute_workgroups[1],
            sim.compute_workgroups[2],
        );
    }

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("simulate.render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        rpass.set_pipeline(&sim.render_pipeline);
        if let Some(ref empty) = sim.empty_bind_group {
            for i in 0..sim.render_bind_group_set {
                rpass.set_bind_group(i, empty, &[]);
            }
        }
        rpass.set_bind_group(sim.render_bind_group_set, &sim.render_bind_groups[bg_index], &[]);
        rpass.draw(0..sim.vertex_count, 0..1);
    }
}

#[allow(clippy::too_many_arguments)]
fn render_pipeline(
    state: &PipelineState,
    view: &TextureView,
    depth_view: &TextureView,
    queue: &wgpu::Queue,
    config: &SurfaceConfiguration,
    start_time: std::time::Instant,
    frame_count: u32,
    mouse_pos: [f32; 2],
    mouse_click_pos: [f32; 2],
    mouse_pressed: bool,
    encoder: &mut wgpu::CommandEncoder,
) {
    // Update Shadertoy-style uniforms when the graphics pipeline asked
    // for them. Each is independently optional; the constructor sets
    // the `Option` based on the descriptor.
    if let Some(ref buf) = state.resolution_buffer {
        let u = ResolutionUniform {
            resolution: [config.width as f32, config.height as f32, 1.0],
            _pad: 0.0,
        };
        queue.write_buffer(buf, 0, bytemuck::cast_slice(&[u]));
    }
    if let Some(ref buf) = state.time_buffer {
        let t = start_time.elapsed().as_secs_f32();
        queue.write_buffer(buf, 0, bytemuck::cast_slice(&[t]));
    }
    if let Some(ref buf) = state.mouse_buffer {
        let m = [
            mouse_pos[0],
            mouse_pos[1],
            if mouse_pressed { mouse_click_pos[0] } else { 0.0 },
            if mouse_pressed { mouse_click_pos[1] } else { 0.0 },
        ];
        queue.write_buffer(buf, 0, bytemuck::cast_slice(&m));
    }
    if let Some(ref buf) = state.frame_buffer {
        queue.write_buffer(buf, 0, bytemuck::cast_slice(&[frame_count]));
    }

    // Dispatch each compute stage in descriptor order.
    for stage in &state.compute_stages {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&stage.label),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&stage.pipeline);
        for (set, bg_opt) in stage.bind_groups_by_set.iter().enumerate() {
            if let Some(bg) = bg_opt {
                cpass.set_bind_group(set as u32, bg, &[]);
            }
        }
        if !stage.push_constants.is_empty() {
            cpass.set_push_constants(0, &stage.push_constants);
        }
        let (x, y, z) = stage.workgroups;
        cpass.dispatch_workgroups(x, y, z);
    }

    // Render the single graphics pipeline.
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("pipeline.render_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::BLACK),
                store: StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(Operations {
                load: LoadOp::Clear(1.0),
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        ..Default::default()
    });
    rpass.set_pipeline(&state.render_pipeline);
    for (set, bg_opt) in state.render_bind_groups_by_set.iter().enumerate() {
        if let Some(bg) = bg_opt {
            rpass.set_bind_group(set as u32, bg, &[]);
        }
    }
    rpass.draw(0..state.vertex_count, 0..1);
}

pub struct App {
    state: Option<State>,
    spec: AppSpec,
}

/// Internal spec dispatcher. `App::new` wraps a `PipelineSpec` for the
/// `vf`/`testpattern` path; `App::new_simulate` wraps a `SimulateSpec`
/// for the simulate path. `State::resumed` matches and calls the
/// appropriate `State::new_*` constructor.
enum AppSpec {
    Vf(PipelineSpec),
    Simulate(SimulateSpec),
    Pipeline(InteractivePipelineSpec),
}

impl AppSpec {
    fn size(&self) -> Option<(u32, u32)> {
        match self {
            AppSpec::Vf(s) => s.size,
            AppSpec::Simulate(s) => s.size,
            AppSpec::Pipeline(s) => s.size,
        }
    }
}

impl App {
    pub fn new(spec: PipelineSpec) -> Self {
        Self {
            state: None,
            spec: AppSpec::Vf(spec),
        }
    }

    pub fn new_simulate(spec: SimulateSpec) -> Self {
        Self {
            state: None,
            spec: AppSpec::Simulate(spec),
        }
    }

    pub fn new_pipeline(spec: InteractivePipelineSpec) -> Self {
        Self {
            state: None,
            spec: AppSpec::Pipeline(spec),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_size = self.spec.size();
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

        let result = match &self.spec {
            AppSpec::Vf(s) => pollster::block_on(State::new(window, s)),
            AppSpec::Simulate(s) => pollster::block_on(State::new_simulate(window, s)),
            AppSpec::Pipeline(s) => pollster::block_on(State::new_pipeline(window, s)),
        };
        match result {
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
