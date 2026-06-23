//! Interactive winit + wgpu application shell shared by the
//! `pipeline` and `testpattern` modes. `PipelineSpec` configures the
//! testpattern path; `InteractivePipelineSpec` configures the
//! descriptor-driven pipeline path. `State` owns the surface +
//! pipeline; `App` is the `winit::ApplicationHandler` wrapper.

mod render;
mod uniforms;
mod vertex_buffers;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, Color, CommandEncoderDescriptor, Extent3d,
    InstanceFlags, LoadOp, Operations, PipelineLayoutDescriptor, PresentMode, RenderPipeline, StoreOp,
    SurfaceConfiguration, TextureDescriptor, TextureDimension, TextureUsages, TextureView,
    TextureViewDescriptor,
};

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

use crate::gpu::{self, DeviceRequest, GpuContext};
use render::DEPTH_FORMAT;
use uniforms::{MouseUniform, ResolutionUniform};

use wyn_pipeline_descriptor::PipelineDescriptor;

// --- Pipeline spec passed to the app -----------------------------------------

/// Configuration for the `testpattern` viewer: a built-in WGSL shader
/// with a single resolution uniform at (group 0, binding 0).
pub struct PipelineSpec {
    pub shader: Shader,
    pub max_frames: Option<u32>,
    pub verbose: bool,
    pub validate: bool,
    pub present_mode: PresentMode,
    pub size: Option<(u32, u32)>,
}

/// Configuration for the interactive shape of `pipeline` mode: an
/// arbitrary chain of compute pipelines plus a graphics pipeline that
/// renders to a window, all driven from the descriptor JSON. Built by
/// `modes::pipeline::run_pipeline_interactive` after detecting a
/// graphics pipeline in the descriptor.
pub struct InteractivePipelineSpec {
    pub shader_path: PathBuf,
    pub descriptor: PipelineDescriptor,
    /// Resolved vertex stage entry-point name.
    pub vertex_entry: String,
    /// Resolved fragment stage entry-point name.
    pub fragment_entry: String,
    /// Per-compute-entry dispatch overrides (`--dispatch ENTRY:WxH[xD]`
    /// on the CLI). Total thread counts on each axis; viz divides by
    /// the descriptor's workgroup_size at dispatch time. Empty when
    /// the compiler's default dispatch is correct for every stage.
    pub dispatch_overrides: HashMap<String, (u32, u32, u32)>,
    /// Ping-pong feedback pairs, one per `--feedback ENTRY:READ=WRITE`
    /// on the CLI. Each entry is `(entry_point, read_binding_name,
    /// write_binding_name)`. The host resolves names to `(set,
    /// binding)` at state construction, allocates two physical
    /// textures per write binding, and swaps which one is bound each
    /// frame. Empty for shaders with no self-feedback.
    pub feedback_specs: Vec<(String, String, String)>,
    pub max_frames: Option<u32>,
    pub verbose: bool,
    pub validate: bool,
    pub present_mode: PresentMode,
    pub size: Option<(u32, u32)>,
    pub vertex_count: u32,
    /// Primitive topology for the graphics pipeline.
    pub topology: wgpu::PrimitiveTopology,
    /// Directory of per-binding `.bin` files. The host loads each
    /// `vertex_inputs[i]` declared on the graphics pipeline into a
    /// vertex buffer, and each `storage_buffer` binding (other than
    /// recognized host-uploaded names like `keyboard`) into a storage
    /// buffer, by reading `<storage_dir>/<binding_name>.bin`.
    pub storage_dir: Option<PathBuf>,
    /// Host storage buffers to allocate and seed once, by binding name →
    /// init spec (from `--buffer-init NAME:BYTES:SPEC`). For a buffer the
    /// shader writes but no `--input`/`--feedback` supplies (a scratch
    /// framebuffer, `0`) or reads as random initial state (`rng`).
    pub buffer_inits: HashMap<String, gpu::BufferInit>,
    /// Flat little-endian `u32` index buffer file. When present, the
    /// host binds it and dispatches `draw_indexed` with
    /// `file_size / 4` indices in place of the non-indexed
    /// `draw(0..vertex_count)`.
    pub index_buffer: Option<PathBuf>,
    /// Storage buffers to read back and dump as f32-array JSON when the
    /// run ends (`--output NAME:FILE`). Keyed by binding name → output
    /// path. Read back at the `--max-frames` exit, so pair with
    /// `--max-frames` to get a deterministic snapshot.
    pub outputs: HashMap<String, PathBuf>,
}

/// Embedded WGSL source compiled in-place. Used by the built-in test
/// pattern; `vertex_entry` / `fragment_entry` must name functions
/// inside `source`.
pub struct Shader(pub &'static str);

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: SurfaceConfiguration,
    start_time: std::time::Instant,
    // Mouse tracking (consumed by pipeline mode's iMouse uniform;
    // harmless for testpattern, which doesn't read it).
    mouse_pos: [f32; 2],
    mouse_click_pos: [f32; 2],
    mouse_pressed: bool,
    // Keyboard state for the Shadertoy 256×3 convention. Rows in
    // declaration order are `currently_down`, `pressed_this_frame`,
    // `toggled`. Each row has 256 columns indexed by Shadertoy's
    // keycode (ASCII for printable keys; named keys map per a small
    // table near `apply_keyboard_event`). `pressed_this_frame` clears
    // at the end of every frame.
    keyboard: [u8; 256 * 3],
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
    /// `--output NAME:FILE` targets to dump at the `--max-frames` exit.
    /// Resolved to concrete GPU buffers at construction so the exit path
    /// is a pure readback. Empty for the testpattern/vf path.
    output_targets: Vec<OutputTarget>,
}

/// One resolved `--output` request: where to read the bytes from and
/// where to write the resulting f32 JSON.
struct OutputTarget {
    name: String,
    path: PathBuf,
    /// Backing buffer(s). For a plain host buffer this is length 1; for a
    /// ping-pong feedback write-side it holds both slots and the dump
    /// reads `frame_count % 2` (the slot last written).
    buffers: Vec<wgpu::Buffer>,
    byte_size: u64,
    ping_pong: bool,
}

/// Mode-specific GPU state held by `State`. Each variant owns the
/// pipelines, bind groups, and buffers that variant's `render` path
/// consumes; common surface / device / window state stays in `State`
/// itself.
enum AppMode {
    VertexFragment(VfState),
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
    /// Bind groups indexed by (parity, descriptor-set number). Parity
    /// 0 and 1 hold logically-distinct ping-pong texture bindings; for
    /// pipelines untouched by feedback, both parities point at the
    /// same physical resources but are built as separate `BindGroup`s
    /// to keep the indexing path uniform. Empty sets below the lowest
    /// used one get a no-op group attached at render time to satisfy
    /// wgpu's contiguous-sets requirement.
    bind_groups_by_set: [Vec<Option<BindGroup>>; 2],
    workgroups: (u32, u32, u32),
    push_constants: Vec<u8>,
    label: String,
}

/// GPU state for `AppMode::Pipeline`. Holds every pipeline + bind
/// group needed to drive the descriptor each frame.
struct PipelineState {
    compute_stages: Vec<PipelineComputeStage>,
    render_pipeline: RenderPipeline,
    /// Render-side bind groups indexed by (parity, descriptor-set
    /// number). See `PipelineComputeStage.bind_groups_by_set` for the
    /// parity convention.
    render_bind_groups_by_set: [Vec<Option<BindGroup>>; 2],
    vertex_count: u32,
    /// One vertex buffer per declared `#[location(n)]` attribute, in
    /// shader_location order. Empty for full-screen-triangle shaders.
    vertex_buffers: Vec<wgpu::Buffer>,
    /// Optional `(buffer, index_count)` for indexed draws.
    index_buffer: Option<(wgpu::Buffer, u32)>,
    // Shadertoy-style uniform buffers, populated per frame when the
    // graphics pipeline declares the matching uniform binding.
    resolution_buffer: Option<wgpu::Buffer>,
    time_buffer: Option<wgpu::Buffer>,
    mouse_buffer: Option<wgpu::Buffer>,
    frame_buffer: Option<wgpu::Buffer>,
    /// Host-uploaded textures, written each frame from CPU state. Today
    /// the only entry is the Shadertoy keyboard texture (256×3 R8Unorm),
    /// allocated when the descriptor declares a Texture binding named
    /// "keyboard" / "iKeyboard".
    host_textures: HashMap<(u32, u32), gpu::HostTextureResource>,
    host_buffers: HashMap<(u32, u32), gpu::HostBufferResource>,
    // Storage buffers + textures + samplers — held to keep the GPU
    // resources alive for the bind groups' lifetime.
    _storage_buffers: HashMap<u32, (wgpu::Buffer, u64)>,
    _storage_textures: HashMap<(u32, u32), gpu::StorageTextureResource>,
    _samplers: HashMap<(u32, u32), wgpu::Sampler>,
}

/// GPU state for the `testpattern` interactive viewer. One render
/// pipeline + a resolution uniform at (group 0, binding 0).
struct VfState {
    pipeline: RenderPipeline,
    resolution_buffer: Option<wgpu::Buffer>,
    uniform_bind_group: Option<BindGroup>,
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
            desired_features: wgpu::Features::EXPERIMENTAL_PASSTHROUGH_SHADERS,
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
                device.features().contains(wgpu::Features::EXPERIMENTAL_PASSTHROUGH_SHADERS)
            );
        }

        // Set up uncaptured error handler to catch GPU errors at runtime
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
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

        let max_frames = spec.max_frames;

        let (pipeline, resolution_buffer, uniform_bind_group) = {
            let Shader(source) = spec.shader;
            let (pipeline, res_buffer, bind_group) = render::build_wgsl_render_pipeline(
                &device,
                &queue,
                config.format,
                config.width,
                config.height,
                source,
            );
            (pipeline, Some(res_buffer), Some(bind_group))
        };

        let now = std::time::Instant::now();
        let depth_view = create_depth_view(&device, &config);

        let vf = VfState {
            pipeline,
            resolution_buffer,
            uniform_bind_group,
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
            keyboard: [0u8; 256 * 3],
            frame_count: 0,
            max_frames,
            frame_times: [0.0; 60],
            frame_time_idx: 0,
            verbose,
            depth_view,
            mode: AppMode::VertexFragment(vf),
            output_targets: Vec::new(),
        })
    }

    /// Construct GPU state for the interactive `pipeline` mode: a
    /// descriptor-driven compute chain + one graphics pipeline. See
    /// `InteractivePipelineSpec` for the descriptor source.
    ///
    /// v1 supports: storage textures (cross-stage write+sample),
    /// samplers, and uniform/push-constant-less compute + graphics
    /// pipelines. Storage buffers, push constants, and Shadertoy
    /// uniforms (iResolution / iTime / iMouse / iFrame) are scaffolded
    /// — the `PipelineState`'s slots are `None` for now and follow-up
    /// commits wire the descriptor → buffer mapping for each.
    async fn new_pipeline(window: Arc<Window>, spec: &InteractivePipelineSpec) -> Result<Self> {
        use wyn_pipeline_descriptor::Pipeline as DescPipeline;

        let validate = spec.validate;
        let verbose = spec.verbose;
        let present_mode = spec.present_mode;

        let win = window.clone();
        let ctx = GpuContext::request(DeviceRequest {
            instance_flags: if validate {
                InstanceFlags::VALIDATION | InstanceFlags::DEBUG
            } else {
                InstanceFlags::empty()
            },
            // FLOAT32_FILTERABLE lets us linear-sample rgba32float
            // storage textures, which feedback-shaders (e.g. mountains
            // painting the heightmap) need so per-frame brush
            // increments at the edges of the brush don't round to
            // zero under f16 precision.
            desired_features: wgpu::Features::EXPERIMENTAL_PASSTHROUGH_SHADERS
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::FLOAT32_FILTERABLE,
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
            eprintln!(
                "[viz pipeline-interactive] Adapter: {} ({:?})",
                info.name, info.backend
            );
        }
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            eprintln!("\n[GPU validation error]\n{:?}\n", error);
        }));

        let size = window.inner_size();
        let config =
            render::configure_surface(&surface, &device, &adapter, size.width, size.height, present_mode)?;
        let depth_view = create_depth_view(&device, &config);

        // Phase 6: resolve `--feedback ENTRY:READ=WRITE` specs against
        // the descriptor. Each spec names a compute entry plus two
        // binding names (read + write) inside it; we look them up to
        // get `(set, binding)` tuples the storage-texture allocator and
        // bind-group builder consume directly.
        // A feedback spec resolves to either a texture pair (READ is a
        // `Binding::Texture`, WRITE is a `Binding::StorageTexture`) or a
        // buffer pair (both sides are `Binding::StorageBuffer`). Same-name
        // matches across kinds within one spec are a mixed-kind error.
        let mut texture_feedback_pairs: Vec<gpu::FeedbackPair> = Vec::new();
        let mut buffer_feedback_pairs: Vec<gpu::FeedbackPair> = Vec::new();
        for (entry, read, write) in &spec.feedback_specs {
            let cp = spec
                .descriptor
                .pipelines
                .iter()
                .find_map(|p| match p {
                    DescPipeline::Compute(cp) if cp.stages.iter().any(|s| s.entry_point == *entry) => {
                        Some(cp)
                    }
                    _ => None,
                })
                .ok_or_else(|| {
                    anyhow!(
                        "--feedback '{}:{}={}' — no compute pipeline with that entry point",
                        entry,
                        read,
                        write
                    )
                })?;
            let mut tex_read: Option<(u32, u32)> = None;
            let mut tex_write: Option<(u32, u32)> = None;
            let mut buf_read: Option<(u32, u32)> = None;
            let mut buf_write: Option<(u32, u32)> = None;
            for b in &cp.bindings {
                match b {
                    wyn_pipeline_descriptor::Binding::Texture {
                        set, binding, name, ..
                    } if name == read => tex_read = Some((*set, *binding)),
                    wyn_pipeline_descriptor::Binding::StorageTexture {
                        set, binding, name, ..
                    } if name == write => tex_write = Some((*set, *binding)),
                    wyn_pipeline_descriptor::Binding::StorageBuffer {
                        set, binding, name, ..
                    } if name == read => buf_read = Some((*set, *binding)),
                    wyn_pipeline_descriptor::Binding::StorageBuffer {
                        set, binding, name, ..
                    } if name == write => buf_write = Some((*set, *binding)),
                    _ => {}
                }
            }
            match (tex_read, tex_write, buf_read, buf_write) {
                (Some((rs, rb)), Some((ws, wb)), None, None) => {
                    texture_feedback_pairs.push(gpu::FeedbackPair {
                        read_set: rs,
                        read_binding: rb,
                        write_set: ws,
                        write_binding: wb,
                    });
                }
                (None, None, Some((rs, rb)), Some((ws, wb))) => {
                    buffer_feedback_pairs.push(gpu::FeedbackPair {
                        read_set: rs,
                        read_binding: rb,
                        write_set: ws,
                        write_binding: wb,
                    });
                }
                (Some(_), _, Some(_), _) | (_, Some(_), _, Some(_)) => {
                    return Err(anyhow!(
                        "--feedback '{}:{}={}' — names match both texture and storage-buffer \
                         bindings; feedback pairs must be all-texture or all-buffer",
                        entry,
                        read,
                        write
                    ));
                }
                _ => {
                    // Enumerate what the entry actually exposes so the user
                    // can see the real binding names — auto-allocated entries
                    // name their feedback buffers generically (e.g. `input_N`
                    // for reads, `tail_output` for the result), which rarely
                    // match a hand-written `--feedback` spec.
                    let mut textures: Vec<String> = Vec::new();
                    let mut storage_images: Vec<String> = Vec::new();
                    let mut storage_buffers: Vec<String> = Vec::new();
                    for b in &cp.bindings {
                        match b {
                            wyn_pipeline_descriptor::Binding::Texture {
                                set, binding, name, ..
                            } => textures.push(format!("'{name}' (set={set}, binding={binding})")),
                            wyn_pipeline_descriptor::Binding::StorageTexture {
                                set, binding, name, ..
                            } => storage_images
                                .push(format!("'{name}' (set={set}, binding={binding})")),
                            wyn_pipeline_descriptor::Binding::StorageBuffer {
                                set, binding, name, ..
                            } => storage_buffers
                                .push(format!("'{name}' (set={set}, binding={binding})")),
                            _ => {}
                        }
                    }
                    let list = |v: &[String]| {
                        if v.is_empty() {
                            "(none)".to_string()
                        } else {
                            v.join(", ")
                        }
                    };
                    let status = |found: bool| if found { "found" } else { "NOT FOUND" };
                    return Err(anyhow!(
                        "--feedback '{entry}:{read}={write}' could not resolve a feedback pair.\n\
                         \x20 read binding '{read}': {}\n\
                         \x20 write binding '{write}': {}\n\
                         A pair must be a texture2d (read) + storage_image (write), or a \
                         storage_buffer (read) + storage_buffer (write) — both on entry '{entry}'.\n\
                         Bindings actually declared on '{entry}':\n\
                         \x20 storage_buffers: {}\n\
                         \x20 textures: {}\n\
                         \x20 storage_images: {}\n\
                         Tip: auto-allocated entries name buffers generically (reads `input_N`, \
                         result `tail_output`); check the names in the descriptor JSON next to the .spv.",
                        status(tex_read.is_some() || buf_read.is_some()),
                        status(tex_write.is_some() || buf_write.is_some()),
                        list(&storage_buffers),
                        list(&textures),
                        list(&storage_images),
                    ));
                }
            }
        }
        let feedback_reads: HashMap<(u32, u32), (u32, u32)> = texture_feedback_pairs
            .iter()
            .map(|p| ((p.read_set, p.read_binding), (p.write_set, p.write_binding)))
            .collect();
        let buffer_feedback_reads: HashMap<(u32, u32), (u32, u32)> = buffer_feedback_pairs
            .iter()
            .map(|p| ((p.read_set, p.read_binding), (p.write_set, p.write_binding)))
            .collect();

        // Allocate cross-pipeline resources up front (Phase 1b helpers).
        let storage_textures = gpu::create_storage_textures(
            &device,
            &spec.descriptor,
            Some((config.width, config.height)),
            &texture_feedback_pairs,
        );
        let feedback_buffers =
            gpu::create_feedback_buffers(&device, &spec.descriptor, &buffer_feedback_pairs)?;
        let host_textures = gpu::create_host_textures(&device, &spec.descriptor, &storage_textures);
        let host_buffers = gpu::create_host_buffers(
            &device,
            &queue,
            &spec.descriptor,
            spec.storage_dir.as_deref(),
            &spec.buffer_inits,
        )?;

        // Resolve `--output NAME:FILE` requests to concrete buffers now,
        // while the feedback and host pools are in scope. The actual
        // readback happens at the `--max-frames` exit. We map each output
        // name → its `(set, binding)` via the descriptor, then prefer the
        // feedback write-side pool (ping-pong) and fall back to the
        // host-buffer pool. The buffers stay alive via the bind groups;
        // these clones are cheap Arc handles.
        let output_targets: Vec<OutputTarget> = {
            let mut name_to_slot: HashMap<&str, (u32, u32)> = HashMap::new();
            for pipeline in &spec.descriptor.pipelines {
                let bindings: &[wyn_pipeline_descriptor::Binding] = match pipeline {
                    DescPipeline::Compute(cp) => &cp.bindings,
                    DescPipeline::Graphics(gp) => &gp.bindings,
                };
                for b in bindings {
                    if let wyn_pipeline_descriptor::Binding::StorageBuffer {
                        set, binding, name, ..
                    } = b
                    {
                        name_to_slot.entry(name.as_str()).or_insert((*set, *binding));
                    }
                }
            }
            let mut targets = Vec::new();
            for (name, path) in &spec.outputs {
                let Some(&key) = name_to_slot.get(name.as_str()) else {
                    eprintln!(
                        "[viz pipeline] --output '{name}': no storage_buffer binding by that \
                         name in the descriptor; skipping"
                    );
                    continue;
                };
                if let Some(res) = feedback_buffers.get(&key) {
                    targets.push(OutputTarget {
                        name: name.clone(),
                        path: path.clone(),
                        byte_size: res.buffers[0].size(),
                        buffers: res.buffers.clone(),
                        ping_pong: true,
                    });
                } else if let Some(res) = host_buffers.get(&key) {
                    targets.push(OutputTarget {
                        name: name.clone(),
                        path: path.clone(),
                        byte_size: res.buffer.size(),
                        buffers: vec![res.buffer.clone()],
                        ping_pong: false,
                    });
                } else {
                    eprintln!(
                        "[viz pipeline] --output '{name}': binding ({}, {}) is neither a \
                         feedback write-side nor a host buffer; can't read it back",
                        key.0, key.1
                    );
                }
            }
            targets
        };

        // Vertex attributes declared on a graphics pipeline (one buffer
        // per `#[location(n)]` attribute, file-loaded from
        // `<storage_dir>/<name>.bin`).
        let vertex_buffer_pack = vertex_buffers::build_vertex_buffers(
            &device,
            &queue,
            &spec.shader_path,
            spec.storage_dir.as_deref(),
        )?;

        // Optional index buffer — flat little-endian u32. Indexed draws
        // when present; non-indexed `draw(0..vertex_count)` otherwise.
        let index_buffer: Option<(wgpu::Buffer, u32)> = if let Some(path) = spec.index_buffer.as_deref() {
            let data = std::fs::read(path)
                .with_context(|| format!("viz pipeline --index-buffer: reading {:?}", path))?;
            if data.len() % 4 != 0 {
                return Err(anyhow::anyhow!(
                    "viz pipeline --index-buffer: {:?} is {} bytes, not a multiple of 4 (u32 indices)",
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
        let samplers = gpu::create_samplers(&device, &spec.descriptor);
        // Phase 4: Shadertoy-style uniforms (iResolution / iTime /
        // iMouse / iFrame). One buffer per `(set, binding)` declared by
        // any pipeline (graphics OR compute); same-slot declarations
        // across pipelines reuse the same physical buffer. The
        // per-frame render path writes their values.
        let all_uniform_bindings: Vec<wyn_pipeline_descriptor::Binding> = spec
            .descriptor
            .pipelines
            .iter()
            .flat_map(|p| match p {
                DescPipeline::Compute(cp) => cp.bindings.iter().cloned().collect::<Vec<_>>(),
                DescPipeline::Graphics(gp) => gp.bindings.iter().cloned().collect(),
            })
            .collect();
        let uniforms = uniforms::build_pipeline_uniforms(&device, &all_uniform_bindings)
            .context("build_pipeline_uniforms")?;

        // Load the SPIR-V module once; both compute and graphics
        // pipelines reuse the same shader binary.
        let module = crate::spirv::load_spirv_module(&device, &spec.shader_path)
            .with_context(|| format!("load SPIR-V module {:?}", spec.shader_path))?;

        // Buffer-size map for dispatch resolution. `DispatchLen::InputBinding`
        // queries this by binding number to compute element count from the
        // source buffer's byte size. Sources we publish:
        //   * Feedback ping-pong slots — the read-side binding stands in
        //     for the dispatch source. Both slots are sized identically,
        //     so slot 0's byte size is authoritative.
        //   * Host-allocated buffers — `--framebuffer` / `--buffer-init` /
        //     `<storage_dir>/<name>.bin` allocations. A compute stage
        //     dispatching `DerivedFrom(InputBinding(host_buf))` (e.g. a
        //     framebuffer-clear lifted out as its own dispatch) gets the
        //     right per-element count.
        let mut dispatch_buffer_sizes: HashMap<u32, (wgpu::Buffer, u64)> = HashMap::new();
        for pair in &buffer_feedback_pairs {
            if let Some(res) = feedback_buffers.get(&(pair.write_set, pair.write_binding)) {
                let buf = &res.buffers[0];
                dispatch_buffer_sizes.insert(pair.read_binding, (buf.clone(), buf.size()));
            }
        }
        for ((_set, binding), res) in &host_buffers {
            dispatch_buffer_sizes
                .entry(*binding)
                .or_insert_with(|| (res.buffer.clone(), res.buffer.size()));
        }

        // ----- Compute stages -----
        let mut compute_stages: Vec<PipelineComputeStage> = Vec::new();
        for pipeline in &spec.descriptor.pipelines {
            let DescPipeline::Compute(cp) = pipeline else {
                continue;
            };

            // Group bindings by `set` to build one bind-group layout +
            // per-parity bind groups per set. The layout is parity-
            // invariant (it describes the binding *types*, not the
            // resources), so we build it once at parity 0 and reuse it
            // for the parity 1 build.
            let max_set = cp.bindings.iter().filter_map(binding_set).max().unwrap_or(0);
            let mut bgls: Vec<wgpu::BindGroupLayout> = Vec::with_capacity((max_set + 1) as usize);
            let mut bgs: [Vec<Option<BindGroup>>; 2] = [Vec::new(), Vec::new()];

            for set in 0..=max_set {
                let any_in_set = cp.bindings.iter().any(|b| binding_set(b) == Some(set));
                if !any_in_set {
                    let layout = empty_bind_group_layout(&device, &format!("compute_empty_set{set}"));
                    for parity in 0..2 {
                        let empty_bg = device.create_bind_group(&BindGroupDescriptor {
                            label: Some(&format!("compute_empty_bg_set{set}_p{parity}")),
                            layout: &layout,
                            entries: &[],
                        });
                        bgs[parity].push(Some(empty_bg));
                    }
                    bgls.push(layout);
                    continue;
                }
                let mut set_layout: Option<wgpu::BindGroupLayout> = None;
                for parity in 0..2 {
                    let (layout, bg) = gpu::build_resource_bind_group_for_set(
                        &device,
                        &cp.bindings,
                        set,
                        wgpu::ShaderStages::COMPUTE,
                        &storage_textures,
                        &host_textures,
                        &host_buffers,
                        &feedback_buffers,
                        &samplers,
                        &uniforms.by_set_binding,
                        &feedback_reads,
                        &buffer_feedback_reads,
                        parity,
                    )
                    .with_context(|| {
                        let entry_points: Vec<&str> =
                            cp.stages.iter().map(|s| s.entry_point.as_str()).collect();
                        format!(
                            "compute {:?}: build bind group for set {} (parity {})",
                            entry_points, set, parity
                        )
                    })?;
                    if set_layout.is_none() {
                        set_layout = Some(layout);
                    }
                    bgs[parity].push(Some(bg));
                }
                bgls.push(set_layout.expect("layout built for parity 0"));
            }

            let bgl_borrows: Vec<&wgpu::BindGroupLayout> = bgls.iter().collect();
            let layout_label =
                format!("compute_layout_{}", cp.stages.first().map(|s| s.entry_point.as_str()).unwrap_or("?"));
            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(&layout_label),
                bind_group_layouts: &bgl_borrows,
                push_constant_ranges: &[],
            });

            // One `PipelineComputeStage` per descriptor stage; bind
            // groups + layout are shared across stages of the same
            // `Compute` pipeline (so we clone the bgs into each
            // per-stage record).
            let n_stages = cp.stages.len();
            for (si, stage) in cp.stages.iter().enumerate() {
                let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&stage.entry_point),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(&stage.entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                });

                // Resolve dispatch. `DispatchLen::InputBinding` reads the source
                // buffer's byte size from `dispatch_buffer_sizes` (populated above
                // from the feedback pool). `DispatchLen::StorageImage` reads from
                // `storage_textures`. CLI `--dispatch ENTRY:WxH[xD]` overrides
                // everything per-entry — the override is in TOTAL threads on each
                // axis, so divide by the matching workgroup-size dim to get
                // workgroup counts.
                let workgroups =
                    if let Some(&(w, h, d)) = spec.dispatch_overrides.get(&stage.entry_point) {
                        let (wgx, wgy, wgz) = (
                            stage.workgroup_size.0.max(1),
                            stage.workgroup_size.1.max(1),
                            stage.workgroup_size.2.max(1),
                        );
                        (w.div_ceil(wgx), h.div_ceil(wgy), d.div_ceil(wgz))
                    } else {
                        gpu::resolve_dispatch_size_with_textures(
                            &stage.dispatch_size,
                            stage.workgroup_size,
                            &dispatch_buffer_sizes,
                            &[],
                            &storage_textures,
                        )
                    };
                if verbose {
                    eprintln!(
                        "[viz pipeline-interactive] compute '{}': dispatch = {} × {} × {}",
                        stage.entry_point, workgroups.0, workgroups.1, workgroups.2
                    );
                }

                // Move bgs on the last stage; clone for earlier ones.
                let stage_bgs = if si + 1 == n_stages { std::mem::take(&mut bgs) } else { bgs.clone() };
                compute_stages.push(PipelineComputeStage {
                    pipeline: compute_pipeline,
                    bind_groups_by_set: stage_bgs,
                    workgroups,
                    push_constants: Vec::new(),
                    label: format!("compute.{}", stage.entry_point),
                });
            }
        }

        // ----- Graphics pipeline -----
        let graphics_bindings = collect_graphics_bindings(&spec.descriptor);
        let g_max_set = graphics_bindings.iter().filter_map(binding_set).max().unwrap_or(0);
        let mut g_bgls: Vec<wgpu::BindGroupLayout> = Vec::with_capacity((g_max_set + 1) as usize);
        let mut g_bgs: [Vec<Option<BindGroup>>; 2] = [Vec::new(), Vec::new()];
        for set in 0..=g_max_set {
            let any_in_set = graphics_bindings.iter().any(|b| binding_set(b) == Some(set));
            if !any_in_set {
                let layout = empty_bind_group_layout(&device, &format!("graphics_empty_set{set}"));
                for parity in 0..2 {
                    let empty_bg = device.create_bind_group(&BindGroupDescriptor {
                        label: Some(&format!("graphics_empty_bg_set{set}_p{parity}")),
                        layout: &layout,
                        entries: &[],
                    });
                    g_bgs[parity].push(Some(empty_bg));
                }
                g_bgls.push(layout);
                continue;
            }
            let mut set_layout: Option<wgpu::BindGroupLayout> = None;
            for parity in 0..2 {
                let (layout, bg) = gpu::build_resource_bind_group_for_set(
                    &device,
                    &graphics_bindings,
                    set,
                    wgpu::ShaderStages::FRAGMENT,
                    &storage_textures,
                    &host_textures,
                    &host_buffers,
                    &feedback_buffers,
                    &samplers,
                    &uniforms.by_set_binding,
                    &feedback_reads,
                    &buffer_feedback_reads,
                    parity,
                )
                .with_context(|| {
                    format!("graphics: build bind group for set {} (parity {})", set, parity)
                })?;
                if set_layout.is_none() {
                    set_layout = Some(layout);
                }
                g_bgs[parity].push(Some(bg));
            }
            g_bgls.push(set_layout.expect("layout built for parity 0"));
        }

        let g_bgl_borrows: Vec<&wgpu::BindGroupLayout> = g_bgls.iter().collect();
        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("graphics_layout"),
            bind_group_layouts: &g_bgl_borrows,
            push_constant_ranges: &[],
        });
        // One wgpu::VertexAttribute + matching VertexBufferLayout per
        // declared `#[location(n)]` attribute. Owned locally so both
        // arrays live until `create_render_pipeline` returns.
        let wgpu_attribs: Vec<wgpu::VertexAttribute> = vertex_buffer_pack
            .attribs
            .iter()
            .map(|(fmt, location)| wgpu::VertexAttribute {
                format: vertex_buffers::wgpu_vertex_format(*fmt),
                offset: 0,
                shader_location: *location,
            })
            .collect();
        let vertex_buffer_layouts: Vec<wgpu::VertexBufferLayout> = wgpu_attribs
            .iter()
            .zip(vertex_buffer_pack.attribs.iter())
            .map(|(attr, (fmt, _))| wgpu::VertexBufferLayout {
                array_stride: fmt.byte_size() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: std::slice::from_ref(attr),
            })
            .collect();
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("graphics_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some(&spec.vertex_entry),
                buffers: &vertex_buffer_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(&spec.fragment_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: spec.topology,
                ..Default::default()
            },
            depth_stencil: Some(render::default_depth_state()),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let state = PipelineState {
            compute_stages,
            render_pipeline,
            render_bind_groups_by_set: g_bgs,
            vertex_count: spec.vertex_count,
            vertex_buffers: vertex_buffer_pack.buffers,
            index_buffer,
            resolution_buffer: uniforms.resolution,
            time_buffer: uniforms.time,
            mouse_buffer: uniforms.mouse,
            frame_buffer: uniforms.frame,
            host_textures,
            host_buffers,
            _storage_buffers: HashMap::new(),
            _storage_textures: storage_textures,
            _samplers: samplers,
        };

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            start_time: std::time::Instant::now(),
            mouse_pos: [0.0, 0.0],
            mouse_click_pos: [0.0, 0.0],
            mouse_pressed: false,
            keyboard: [0u8; 256 * 3],
            frame_count: 0,
            max_frames: spec.max_frames,
            frame_times: [0.0; 60],
            frame_time_idx: 0,
            verbose,
            depth_view,
            mode: AppMode::Pipeline(state),
            output_targets,
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

    /// Read back every `--output` target and write it as f32-array JSON.
    /// Called once at the `--max-frames` exit. For a ping-pong feedback
    /// buffer the freshest data is in the slot just written this frame
    /// (`frame_count % 2`, matching the bind-group parity convention).
    fn dump_outputs(&self) {
        for t in &self.output_targets {
            let buf = if t.ping_pong { &t.buffers[(self.frame_count as usize) % 2] } else { &t.buffers[0] };
            match gpu::readback_buffer(&self.device, &self.queue, buf, t.byte_size) {
                Ok(data) => match crate::json::write_f32_json(&t.path, &data) {
                    Ok(()) => eprintln!(
                        "[viz pipeline] wrote {} f32 from '{}' to {}",
                        data.len(),
                        t.name,
                        t.path.display()
                    ),
                    Err(e) => eprintln!("[viz pipeline] --output '{}': write failed: {e:#}", t.name),
                },
                Err(e) => eprintln!("[viz pipeline] --output '{}': readback failed: {e:#}", t.name),
            }
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
                        &self.keyboard,
                        &mut encoder,
                    ),
                }
                // Clear the "pressed this frame" row of the keyboard
                // state — every press lives for exactly one frame in
                // the Shadertoy convention. Cheap (256 bytes) and
                // harmless when no keyboard texture is bound.
                clear_keyboard_pressed_this_frame(&mut self.keyboard);

                self.queue.submit(Some(encoder.finish()));
                frame.present();

                // Wait for GPU to finish to get accurate frame timing (only when verbose)
                if let Some(frame_start) = frame_start {
                    let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

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
                        self.dump_outputs();
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

/// Per-frame `testpattern` path: write the resolution uniform then
/// draw the full-screen triangle.
fn render_vf(
    vf: &VfState,
    view: &TextureView,
    depth_view: &TextureView,
    queue: &wgpu::Queue,
    config: &SurfaceConfiguration,
    encoder: &mut wgpu::CommandEncoder,
) {
    if let Some(ref resolution_buffer) = vf.resolution_buffer {
        let resolution = ResolutionUniform {
            resolution: [config.width as f32, config.height as f32, 1.0],
            _pad: 0.0,
        };
        queue.write_buffer(resolution_buffer, 0, bytemuck::cast_slice(&[resolution]));
    }

    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("pass"),
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

    rpass.set_pipeline(&vf.pipeline);
    if let Some(ref bind_group) = vf.uniform_bind_group {
        rpass.set_bind_group(0, bind_group, &[]);
    }
    rpass.draw(0..3, 0..1);
}

/// Map a winit `KeyCode` to Shadertoy's row index. Shadertoy uses
/// JavaScript keyCodes — printable keys = ASCII, named keys per a
/// fixed table. We cover the keys the Mountains shader actually
/// touches (Enter, Backspace, Shift) plus a generous set so basic
/// shaders work without further tweaking. Unknown keys return None.
fn shadertoy_keycode(key: &winit::keyboard::KeyCode) -> Option<u8> {
    use winit::keyboard::KeyCode::*;
    Some(match key {
        Backspace => 8,
        Tab => 9,
        Enter | NumpadEnter => 13,
        ShiftLeft | ShiftRight => 16,
        ControlLeft | ControlRight => 17,
        AltLeft | AltRight => 18,
        Pause => 19,
        CapsLock => 20,
        Escape => 27,
        Space => 32,
        PageUp => 33,
        PageDown => 34,
        End => 35,
        Home => 36,
        ArrowLeft => 37,
        ArrowUp => 38,
        ArrowRight => 39,
        ArrowDown => 40,
        Delete => 46,
        Digit0 => 48,
        Digit1 => 49,
        Digit2 => 50,
        Digit3 => 51,
        Digit4 => 52,
        Digit5 => 53,
        Digit6 => 54,
        Digit7 => 55,
        Digit8 => 56,
        Digit9 => 57,
        KeyA => 65,
        KeyB => 66,
        KeyC => 67,
        KeyD => 68,
        KeyE => 69,
        KeyF => 70,
        KeyG => 71,
        KeyH => 72,
        KeyI => 73,
        KeyJ => 74,
        KeyK => 75,
        KeyL => 76,
        KeyM => 77,
        KeyN => 78,
        KeyO => 79,
        KeyP => 80,
        KeyQ => 81,
        KeyR => 82,
        KeyS => 83,
        KeyT => 84,
        KeyU => 85,
        KeyV => 86,
        KeyW => 87,
        KeyX => 88,
        KeyY => 89,
        KeyZ => 90,
        _ => return None,
    })
}

/// Apply a winit keyboard event to the 256×3 keyboard state.
/// Row 0 (offset 0..256) = currently down. Row 1 (offset 256..512) =
/// pressed this frame (cleared by `clear_keyboard_pressed_this_frame`
/// at end of frame). Row 2 (offset 512..768) = toggled (flipped on
/// each press).
fn apply_keyboard_event(state: &mut [u8; 256 * 3], event: &winit::event::KeyEvent) {
    let winit::keyboard::PhysicalKey::Code(code) = event.physical_key else {
        return;
    };
    let Some(idx) = shadertoy_keycode(&code) else {
        return;
    };
    let i = idx as usize;
    match event.state {
        winit::event::ElementState::Pressed => {
            // Mark currently down + this-frame press, only on the
            // first press event (winit fires KeyboardInput on repeat
            // too; filter on `event.repeat`).
            state[i] = 0xff;
            if !event.repeat {
                state[256 + i] = 0xff;
                state[512 + i] = if state[512 + i] != 0 { 0 } else { 0xff };
            }
        }
        winit::event::ElementState::Released => {
            state[i] = 0;
        }
    }
}

/// Clear the pressed-this-frame row at the end of each frame.
fn clear_keyboard_pressed_this_frame(state: &mut [u8; 256 * 3]) {
    for byte in &mut state[256..512] {
        *byte = 0;
    }
}

/// `Some(set)` for any binding that lives in a descriptor set; `None`
/// for `PushConstant` (which has no set).
fn binding_set(b: &wyn_pipeline_descriptor::Binding) -> Option<u32> {
    use wyn_pipeline_descriptor::Binding;
    match b {
        Binding::StorageBuffer { set, .. } => Some(*set),
        Binding::Uniform { set, .. } => Some(*set),
        Binding::Texture { set, .. } => Some(*set),
        Binding::Sampler { set, .. } => Some(*set),
        Binding::StorageTexture { set, .. } => Some(*set),
        Binding::PushConstant { .. } => None,
    }
}

/// Empty bind group layout used to satisfy wgpu's contiguous-set rule
/// at any set index a pipeline doesn't otherwise populate.
fn empty_bind_group_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[],
    })
}

/// Collect every `Binding` declared on any Graphics pipeline in the
/// descriptor. Wyn emits one graphics pipeline per entry point, so a
/// shader's vertex bindings + fragment bindings can be split across
/// two `Pipeline::Graphics` entries; we re-merge them for the unified
/// render bind-group construction.
fn collect_graphics_bindings(desc: &PipelineDescriptor) -> Vec<wyn_pipeline_descriptor::Binding> {
    desc.pipelines
        .iter()
        .filter_map(|p| {
            if let wyn_pipeline_descriptor::Pipeline::Graphics(g) = p {
                Some(g.bindings.clone())
            } else {
                None
            }
        })
        .flatten()
        .collect()
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
    keyboard: &[u8; 256 * 3],
    encoder: &mut wgpu::CommandEncoder,
) {
    // Push host-uploaded textures (currently: keyboard) up to the
    // GPU before the frame's compute + render passes consume them.
    for ((_, _), res) in &state.host_textures {
        match res.kind {
            gpu::HostTextureKind::Keyboard => {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &res.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    keyboard,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        // R8Unorm = 1 byte per texel; 256 texels per row.
                        bytes_per_row: Some(res.extent.0),
                        rows_per_image: Some(res.extent.1),
                    },
                    wgpu::Extent3d {
                        width: res.extent.0,
                        height: res.extent.1,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
    }
    // Push host-uploaded storage buffers (keyboard as 768 u32 entries,
    // matching the same row * 256 + keycode layout the texture path
    // uses, just laid out flat instead of as a 2D image).
    for ((_, _), res) in &state.host_buffers {
        match res.kind {
            gpu::HostBufferKind::Keyboard => {
                let mut as_u32 = [0u32; 768];
                for (dst, src) in as_u32.iter_mut().zip(keyboard.iter()) {
                    *dst = *src as u32;
                }
                queue.write_buffer(&res.buffer, 0, bytemuck::cast_slice(&as_u32));
            }
            // Loaded once at startup; the file contents are already on the GPU.
            gpu::HostBufferKind::FileLoaded => {}
        }
    }
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
        // iTime is padded to 16 bytes (`min_binding_size` on most
        // adapters); write a vec4 with `[t, 0, 0, 0]`.
        let t = start_time.elapsed().as_secs_f32();
        queue.write_buffer(buf, 0, bytemuck::cast_slice(&[t, 0.0, 0.0, 0.0]));
    }
    if let Some(ref buf) = state.mouse_buffer {
        let u = MouseUniform {
            mouse: [
                mouse_pos[0],
                mouse_pos[1],
                if mouse_pressed { mouse_click_pos[0] } else { 0.0 },
                if mouse_pressed { mouse_click_pos[1] } else { 0.0 },
            ],
        };
        queue.write_buffer(buf, 0, bytemuck::cast_slice(&[u]));
    }
    if let Some(ref buf) = state.frame_buffer {
        let u = uniforms::FrameUniform {
            frame: frame_count,
            _pad: [0; 3],
        };
        queue.write_buffer(buf, 0, bytemuck::cast_slice(&[u]));
    }

    // Pick the bind-group parity for this frame. Phase 6: compute and
    // graphics pipelines both hold two bind-group sets; ping-pong
    // textures alternate which physical slot is "current" each frame.
    let parity = (frame_count as usize) % 2;

    // Dispatch each compute stage in descriptor order.
    for stage in &state.compute_stages {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&stage.label),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&stage.pipeline);
        for (set, bg_opt) in stage.bind_groups_by_set[parity].iter().enumerate() {
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
    for (set, bg_opt) in state.render_bind_groups_by_set[parity].iter().enumerate() {
        if let Some(bg) = bg_opt {
            rpass.set_bind_group(set as u32, bg, &[]);
        }
    }
    for (slot, buf) in state.vertex_buffers.iter().enumerate() {
        rpass.set_vertex_buffer(slot as u32, buf.slice(..));
    }
    match &state.index_buffer {
        Some((buf, count)) => {
            rpass.set_index_buffer(buf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..*count, 0, 0..1);
        }
        None => {
            rpass.draw(0..state.vertex_count, 0..1);
        }
    }
}

pub struct App {
    state: Option<State>,
    spec: AppSpec,
}

/// Internal spec dispatcher. `App::new` wraps a `PipelineSpec` for the
/// `testpattern` path; `App::new_pipeline` wraps an
/// `InteractivePipelineSpec` for the descriptor-driven `pipeline` path.
/// `State::resumed` matches and calls the appropriate
/// `State::new_*` constructor.
enum AppSpec {
    Vf(PipelineSpec),
    Pipeline(InteractivePipelineSpec),
}

impl AppSpec {
    fn size(&self) -> Option<(u32, u32)> {
        match self {
            AppSpec::Vf(s) => s.size,
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
                    WindowEvent::KeyboardInput { event, .. } => {
                        apply_keyboard_event(&mut state.keyboard, &event);
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
