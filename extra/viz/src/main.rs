// src/main.rs
//#![windows_subsystem = "windows"]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages, Color,
    ColorTargetState, CommandEncoderDescriptor, DeviceDescriptor, FragmentState, Instance,
    InstanceDescriptor, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor, PowerPreference,
    PresentMode, PrimitiveState, RenderPipeline, RequestAdapterOptions, ShaderModuleDescriptorPassthrough,
    ShaderStages, StoreOp, SurfaceConfiguration, TextureUsages, Trace, VertexState,
};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes};

use rspirv::binary::parse_words;
use rspirv::dr::{Loader, Operand};
use rspirv::spirv::ExecutionModel;

// --- CLI ---------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "wgpu-spv", about = "Tiny wgpu demo that builds a pipeline from SPIR-V", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Use a single SPIR-V module with separate vertex & fragment entry points
    #[command(name = "vf", visible_alias = "vertex-fragment")]
    VertexFragment {
        /// Path to the SPIR-V module containing both entry points
        path: PathBuf,
        /// Vertex shader entry point name (auto-detected if omitted)
        #[arg(long)]
        vertex: Option<String>,
        /// Fragment shader entry point name (auto-detected if omitted)
        #[arg(long)]
        fragment: Option<String>,
        /// Enable Shadertoy-compatible uniforms (iResolution, iTime)
        #[arg(long)]
        shadertoy: bool,
        /// Maximum number of frames to render before exiting (for debugging)
        #[arg(long)]
        max_frames: Option<u32>,
        /// Print frame timing statistics
        #[arg(short, long)]
        verbose: bool,
    },
    /// Run a compute shader (headless)
    #[command(name = "compute")]
    Compute {
        /// Path to the SPIR-V module containing the compute entry point
        path: PathBuf,
        /// Compute shader entry point name (auto-detected if omitted)
        #[arg(long)]
        entry: Option<String>,
        /// Number of workgroups to dispatch (X dimension)
        #[arg(long, default_value = "1")]
        workgroups_x: u32,
        /// Number of workgroups to dispatch (Y dimension)
        #[arg(long, default_value = "1")]
        workgroups_y: u32,
        /// Number of workgroups to dispatch (Z dimension)
        #[arg(long, default_value = "1")]
        workgroups_z: u32,
        /// Print verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    /// Show device and driver information
    #[command(name = "info")]
    Info,
}

// --- Pipeline spec passed to the app -----------------------------------------

enum PipelineSpec {
    VertexFragment {
        path: PathBuf,
        vertex: String,
        fragment: String,
        shadertoy: bool,
        max_frames: Option<u32>,
        verbose: bool,
    },
}

// --- Entry point detection ---------------------------------------------------

/// Parse SPIR-V words and extract all entry points with their execution models.
fn detect_entry_points(spirv_words: &[u32]) -> Result<Vec<(String, ExecutionModel)>> {
    let mut loader = Loader::new();
    parse_words(spirv_words, &mut loader).map_err(|e| anyhow!("Failed to parse SPIR-V: {:?}", e))?;
    let module = loader.module();

    let mut entry_points = Vec::new();

    // module.entry_points contains OpEntryPoint instructions
    // Operands: [0] ExecutionModel, [1] IdRef (function), [2] LiteralString (name), [3..] interfaces
    for instruction in &module.entry_points {
        if instruction.operands.len() < 3 {
            continue;
        }

        let execution_model = match &instruction.operands[0] {
            Operand::ExecutionModel(model) => *model,
            _ => continue,
        };

        let name = match &instruction.operands[2] {
            Operand::LiteralString(s) => s.clone(),
            _ => continue,
        };

        entry_points.push((name, execution_model));
    }

    Ok(entry_points)
}

/// Resolve vertex and fragment entry point names.
/// If both provided, use them. If neither, auto-detect. If only one, error.
fn resolve_entry_points(
    path: &Path,
    vertex_arg: Option<String>,
    fragment_arg: Option<String>,
) -> Result<(String, String)> {
    match (vertex_arg, fragment_arg) {
        (Some(v), Some(f)) => Ok((v, f)),
        (None, None) => auto_detect_entry_points(path),
        (Some(_), None) => Err(anyhow!(
            "--vertex was provided but --fragment was not. Provide both or neither for auto-detection."
        )),
        (None, Some(_)) => Err(anyhow!(
            "--fragment was provided but --vertex was not. Provide both or neither for auto-detection."
        )),
    }
}

/// Auto-detect entry points from a SPIR-V file.
/// Succeeds if exactly one Vertex and one Fragment entry point exist (and no others).
fn auto_detect_entry_points(path: &Path) -> Result<(String, String)> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;

    if bytes.len() % 4 != 0 {
        return Err(anyhow!("SPIR-V file size is not aligned to 4-byte words"));
    }

    let spirv_words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let entry_points = detect_entry_points(&spirv_words)?;

    let vertex_entries: Vec<_> =
        entry_points.iter().filter(|(_, m)| *m == ExecutionModel::Vertex).collect();

    let fragment_entries: Vec<_> =
        entry_points.iter().filter(|(_, m)| *m == ExecutionModel::Fragment).collect();

    let other_entries: Vec<_> = entry_points
        .iter()
        .filter(|(_, m)| *m != ExecutionModel::Vertex && *m != ExecutionModel::Fragment)
        .collect();

    match (vertex_entries.len(), fragment_entries.len(), other_entries.len()) {
        (1, 1, 0) => {
            let vertex_name = vertex_entries[0].0.clone();
            let fragment_name = fragment_entries[0].0.clone();
            eprintln!(
                "Auto-detected entry points: vertex='{}', fragment='{}'",
                vertex_name, fragment_name
            );
            Ok((vertex_name, fragment_name))
        }
        _ => {
            let mut msg = String::from("Cannot auto-detect entry points.\n\nFound entry points:\n");

            if !vertex_entries.is_empty() {
                msg.push_str("  Vertex:\n");
                for (name, _) in &vertex_entries {
                    msg.push_str(&format!("    - {}\n", name));
                }
            }

            if !fragment_entries.is_empty() {
                msg.push_str("  Fragment:\n");
                for (name, _) in &fragment_entries {
                    msg.push_str(&format!("    - {}\n", name));
                }
            }

            if !other_entries.is_empty() {
                msg.push_str("  Other:\n");
                for (name, model) in &other_entries {
                    msg.push_str(&format!("    - {} ({:?})\n", name, model));
                }
            }

            if entry_points.is_empty() {
                msg.push_str("  (none found)\n");
            }

            msg.push_str("\nExpected exactly 1 Vertex and 1 Fragment entry point.\n");
            msg.push_str("Use --vertex and --fragment to specify entry points manually.");

            Err(anyhow!(msg))
        }
    }
}

/// Auto-detect a compute entry point from a SPIR-V file.
/// Succeeds if exactly one GLCompute entry point exists.
fn auto_detect_compute_entry_point(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;

    if bytes.len() % 4 != 0 {
        return Err(anyhow!("SPIR-V file size is not aligned to 4-byte words"));
    }

    let spirv_words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let entry_points = detect_entry_points(&spirv_words)?;

    let compute_entries: Vec<_> =
        entry_points.iter().filter(|(_, m)| *m == ExecutionModel::GLCompute).collect();

    match compute_entries.len() {
        1 => {
            let name = compute_entries[0].0.clone();
            eprintln!("Auto-detected compute entry point: '{}'", name);
            Ok(name)
        }
        0 => Err(anyhow!("No GLCompute entry point found in SPIR-V module")),
        _ => {
            let mut msg = String::from("Multiple compute entry points found:\n");
            for (name, _) in &compute_entries {
                msg.push_str(&format!("  - {}\n", name));
            }
            msg.push_str("Use --entry to specify which one to run.");
            Err(anyhow!(msg))
        }
    }
}

/// Create a headless device (no window/surface required)
async fn create_headless_device(verbose: bool) -> Result<(wgpu::Device, wgpu::Queue)> {
    let instance = Instance::new(&InstanceDescriptor::default());

    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .context("request_adapter failed")?;

    let adapter_features = adapter.features();
    let spirv_passthrough_supported =
        adapter_features.contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH);

    if verbose {
        println!("SPIRV_SHADER_PASSTHROUGH supported: {}", spirv_passthrough_supported);
    }

    let mut required_features = wgpu::Features::empty();
    if spirv_passthrough_supported {
        required_features |= wgpu::Features::SPIRV_SHADER_PASSTHROUGH;
    }

    adapter
        .request_device(&DeviceDescriptor {
            label: None,
            required_features,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: Trace::Off,
        })
        .await
        .context("failed to create logical device")
}

/// Create debug buffer and staging buffer for compute shaders
fn create_compute_debug_buffers(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::Buffer, wgpu::Buffer) {
    let debug_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("debug_buffer"),
        size: DEBUG_BUFFER_SIZE,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut init_data = vec![0u32; (DEBUG_BUFFER_SIZE / 4) as usize];
    init_data[2] = DEFAULT_MAX_LOOPS;
    queue.write_buffer(&debug_buffer, 0, bytemuck::cast_slice(&init_data));

    let staging_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("debug_staging_buffer"),
        size: DEBUG_BUFFER_SIZE,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    (debug_buffer, staging_buffer)
}

/// Print decoded GDP debug output from a mapped buffer
fn print_debug_buffer(u32_data: &[u32], verbose: bool) {
    let write_head_global = u32_data[0] as usize;
    let max_loops = u32_data[2] as usize;
    let data_start = 3;
    let data_len = 4093;

    if write_head_global == 0 {
        println!("[DEBUG] No output");
        return;
    }

    let loop_count = write_head_global / data_len;
    let words_to_read = write_head_global.min(data_len);

    println!(
        "\n=== Debug Output ({} words written, loops={}/{}) ===",
        write_head_global, loop_count, max_loops
    );

    if verbose {
        println!("Raw buffer hex (first 100 u32s):");
        for i in 0..100.min(u32_data.len()) {
            if i % 16 == 0 {
                print!("\n{:04x}: ", i);
            }
            print!("{:08x} ", u32_data[i]);
        }
        println!("\n");
    }

    let data_slice = &u32_data[data_start..data_start + data_len];
    let mut pos = 0;
    let mut count = 0;

    while pos < words_to_read {
        let header = data_slice[pos];
        let type_tag = header & 0xFF;
        let size = (header >> 8) as usize;

        if size == 0 || pos + 1 + size > data_len {
            break;
        }

        match type_tag {
            0x00 => println!("U: {}", data_slice[pos + 1]),
            0x01 => println!("I: {}", data_slice[pos + 1] as i32),
            0x02 => {
                let mut bytes = Vec::new();
                for i in 0..size {
                    let word = data_slice[pos + 1 + i];
                    bytes.extend_from_slice(&word.to_le_bytes());
                }
                while bytes.last() == Some(&0) {
                    bytes.pop();
                }
                println!("S: {}", String::from_utf8_lossy(&bytes));
            }
            0x03 => println!("F: {}", f32::from_bits(data_slice[pos + 1])),
            _ => {
                println!("Unknown type: 0x{:02x}", type_tag);
                break;
            }
        }
        pos += 1 + size;
        count += 1;
    }

    println!("({} values decoded)", count);
    println!("=================================\n");
}

/// Run a compute shader headlessly (no window)
async fn run_compute_shader(
    path: PathBuf,
    entry: String,
    workgroups: (u32, u32, u32),
    verbose: bool,
) -> Result<()> {
    let (device, queue) = create_headless_device(verbose).await?;

    let (debug_buffer, staging_buffer) = create_compute_debug_buffers(&device, &queue);

    let debug_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("debug_bind_group_layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let debug_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("debug_bind_group"),
        layout: &debug_bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &debug_buffer,
                offset: 0,
                size: None,
            }),
        }],
    });

    let module = load_spirv_module(&device, &path)?;

    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("compute_layout"),
        bind_group_layouts: &[&debug_bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute_pipeline"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(&entry),
        compilation_options: Default::default(),
        cache: None,
    });

    if verbose {
        println!("Dispatching {} x {} x {} workgroups...", workgroups.0, workgroups.1, workgroups.2);
    }

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("compute_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &debug_bind_group, &[]);
        cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    encoder.copy_buffer_to_buffer(&debug_buffer, 0, &staging_buffer, 0, DEBUG_BUFFER_SIZE);
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::Wait);

    // Read back debug buffer
    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::Wait);

    if rx.recv().unwrap().is_ok() {
        let data = buffer_slice.get_mapped_range();
        print_debug_buffer(bytemuck::cast_slice(&data), verbose);
        drop(data);
        staging_buffer.unmap();
    }

    Ok(())
}

// --- App state ---------------------------------------------------------------

/// Debug buffer size: 16KB = 4096 u32s
/// Layout: { write_head: u32, read_head: u32, max_loops: u32, data: [4093]u32 }
const DEBUG_BUFFER_SIZE: u64 = 16384;
const DEFAULT_MAX_LOOPS: u32 = 100; // Limit debug output to prevent GPU lockup

// Uniform buffers - one per shader uniform
// iResolution: [2]f32
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ResolutionUniform {
    resolution: [f32; 2],
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

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: SurfaceConfiguration,
    pipeline: RenderPipeline,
    // Debug buffer support (optional - only present when shader uses debug intrinsics)
    debug_buffer: Option<wgpu::Buffer>,
    debug_staging_buffer: Option<wgpu::Buffer>,
    debug_bind_group: Option<BindGroup>,
    // Uniform support - separate buffers for iResolution, iTime, iMouse (optional, enabled with --shadertoy)
    resolution_buffer: Option<wgpu::Buffer>,
    time_buffer: Option<wgpu::Buffer>,
    mouse_buffer: Option<wgpu::Buffer>,
    uniform_bind_group: Option<BindGroup>,
    start_time: std::time::Instant,
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
        let instance = Instance::new(&InstanceDescriptor::default());

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

        // Check if SPIRV_SHADER_PASSTHROUGH is supported
        let adapter_features = adapter.features();
        let spirv_passthrough_supported =
            adapter_features.contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH);

        println!(
            "SPIRV_SHADER_PASSTHROUGH supported: {}",
            spirv_passthrough_supported
        );

        // Build required features
        let mut required_features = wgpu::Features::empty();
        if spirv_passthrough_supported {
            required_features |= wgpu::Features::SPIRV_SHADER_PASSTHROUGH;
        }
        // Need writable storage from vertex shader for debug buffer
        required_features |= wgpu::Features::VERTEX_WRITABLE_STORAGE;

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

        let caps = surface.get_capabilities(&adapter);
        let format =
            caps.formats.get(0).copied().ok_or_else(|| anyhow!("surface reports no supported formats"))?;
        let size = window.inner_size();

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: PresentMode::Fifo,
            alpha_mode: caps
                .alpha_modes
                .get(0)
                .copied()
                .ok_or_else(|| anyhow!("surface reports no alpha modes"))?,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // === Create debug buffer (always, shader may or may not use it) =========
        let debug_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("debug_buffer"),
            size: DEBUG_BUFFER_SIZE,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize debug buffer: [write_head=0, read_head=0, max_loops=DEFAULT_MAX_LOOPS, data=zeros]
        let mut init_data = vec![0u32; (DEBUG_BUFFER_SIZE / 4) as usize];
        init_data[2] = DEFAULT_MAX_LOOPS; // Set max_loops
        queue.write_buffer(&debug_buffer, 0, bytemuck::cast_slice(&init_data));

        let debug_staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("debug_staging_buffer"),
            size: DEBUG_BUFFER_SIZE,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create init guard buffer for _init function atomic guard (binding 1)
        // Layout: { guard: u32 } initialized to 0
        let init_guard_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("init_guard_buffer"),
            size: 4, // Single u32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Initialize to 0
        queue.write_buffer(&init_guard_buffer, 0, bytemuck::cast_slice(&[0u32]));

        // Create bind group layout for debug buffer at set=0, binding=0 and init guard at binding=1
        let debug_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("debug_bind_group_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let debug_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("debug_bind_group"),
            layout: &debug_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &debug_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &init_guard_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // === Conditionally create uniform buffers (set=1) based on spec ======
        // Shadertoy canonical uniform ordering:
        //   binding 0: iResolution (vec3, we use vec2)
        //   binding 1: iTime (f32)
        //   binding 5: iMouse (vec4)
        let (resolution_buffer, time_buffer, mouse_buffer, uniform_bind_group, uniform_bind_group_layout) =
            if let PipelineSpec::VertexFragment { shadertoy: true, .. } = spec {
                // Binding 0: iResolution ([2]f32)
                let resolution_buffer = device.create_buffer(&BufferDescriptor {
                    label: Some("resolution_buffer"),
                    size: std::mem::size_of::<ResolutionUniform>() as u64,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let initial_resolution = ResolutionUniform {
                    resolution: [800.0, 600.0],
                };
                queue.write_buffer(&resolution_buffer, 0, bytemuck::cast_slice(&[initial_resolution]));

                // Binding 1: iTime (f32)
                let time_buffer = device.create_buffer(&BufferDescriptor {
                    label: Some("time_buffer"),
                    size: std::mem::size_of::<TimeUniform>() as u64,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let initial_time = TimeUniform { time: 0.0 };
                queue.write_buffer(&time_buffer, 0, bytemuck::cast_slice(&[initial_time]));

                // Binding 5: iMouse (vec4f32)
                let mouse_buffer = device.create_buffer(&BufferDescriptor {
                    label: Some("mouse_buffer"),
                    size: std::mem::size_of::<MouseUniform>() as u64,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let initial_mouse = MouseUniform {
                    mouse: [0.0, 0.0, 0.0, 0.0],
                };
                queue.write_buffer(&mouse_buffer, 0, bytemuck::cast_slice(&[initial_mouse]));

                let uniform_bind_group_layout =
                    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                        label: Some("uniform_bind_group_layout"),
                        entries: &[
                            BindGroupLayoutEntry {
                                binding: 0,
                                visibility: ShaderStages::VERTEX_FRAGMENT,
                                ty: BindingType::Buffer {
                                    ty: BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            BindGroupLayoutEntry {
                                binding: 1,
                                visibility: ShaderStages::VERTEX_FRAGMENT,
                                ty: BindingType::Buffer {
                                    ty: BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            BindGroupLayoutEntry {
                                binding: 5,
                                visibility: ShaderStages::VERTEX_FRAGMENT,
                                ty: BindingType::Buffer {
                                    ty: BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });

                let uniform_bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("uniform_bind_group"),
                    layout: &uniform_bind_group_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &resolution_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &time_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                        BindGroupEntry {
                            binding: 5,
                            resource: BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &mouse_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                    ],
                });

                (
                    Some(resolution_buffer),
                    Some(time_buffer),
                    Some(mouse_buffer),
                    Some(uniform_bind_group),
                    Some(uniform_bind_group_layout),
                )
            } else {
                (None, None, None, None, None)
            };

        // Extract max_frames and verbose from spec
        let (max_frames, verbose) = match spec {
            PipelineSpec::VertexFragment {
                max_frames, verbose, ..
            } => (*max_frames, *verbose),
        };

        // === Build pipeline from the chosen mode ==============================
        let pipeline = match spec {
            PipelineSpec::VertexFragment {
                path,
                vertex,
                fragment,
                shadertoy,
                ..
            } => {
                let module = load_spirv_module(&device, path)
                    .with_context(|| format!("load SPIR-V module {:?}", path))?;

                // Build bind group layout list conditionally
                let mut bind_group_layouts_vec = vec![&debug_bind_group_layout];
                if *shadertoy {
                    if let Some(ref ubl) = uniform_bind_group_layout {
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
        };

        let now = std::time::Instant::now();

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            debug_buffer: Some(debug_buffer),
            debug_staging_buffer: Some(debug_staging_buffer),
            debug_bind_group: Some(debug_bind_group),
            resolution_buffer,
            time_buffer,
            mouse_buffer,
            uniform_bind_group,
            start_time: now,
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
                resolution: [self.config.width as f32, self.config.height as f32],
            };
            self.queue.write_buffer(resolution_buffer, 0, bytemuck::cast_slice(&[resolution]));

            // Update iTime
            let time = TimeUniform { time: elapsed };
            self.queue.write_buffer(time_buffer, 0, bytemuck::cast_slice(&[time]));
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
                    // Set debug bind group if available (set=0)
                    if let Some(ref bind_group) = self.debug_bind_group {
                        rpass.set_bind_group(0, bind_group, &[]);
                    }
                    // Set uniform bind group if available (set=1)
                    if let Some(ref bind_group) = self.uniform_bind_group {
                        rpass.set_bind_group(1, bind_group, &[]);
                    }
                    rpass.draw(0..3, 0..1);
                }

                // Copy debug buffer to staging buffer for readback
                if let (Some(ref debug_buffer), Some(ref staging_buffer)) =
                    (&self.debug_buffer, &self.debug_staging_buffer)
                {
                    encoder.copy_buffer_to_buffer(debug_buffer, 0, staging_buffer, 0, DEBUG_BUFFER_SIZE);
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
                        self.print_debug_output();
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

    /// Read and print all debug output from the buffer (call on exit)
    fn print_debug_output(&mut self) {
        let staging_buffer = match &self.debug_staging_buffer {
            Some(b) => b,
            None => return,
        };

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        // Wait for the GPU to finish
        let _ = self.device.poll(wgpu::PollType::Wait);

        if rx.recv().unwrap().is_ok() {
            let data = buffer_slice.get_mapped_range();
            let u32_data: &[u32] = bytemuck::cast_slice(&data);

            // Buffer layout: [write_head, read_head, max_loops, data[4093]]
            let write_head_global = u32_data[0] as usize; // Unbounded counter
            let _read_head = u32_data[1] as usize;
            let max_loops = u32_data[2] as usize;
            let data_start = 3;
            let data_len = 4093;

            if write_head_global == 0 {
                eprintln!("[DEBUG] No output");
                return;
            }

            // Calculate actual position in ring buffer and number of loops
            let loop_count = write_head_global / data_len;
            let words_to_read = write_head_global.min(data_len);

            eprintln!(
                "\n=== Debug Output ({} words written, loops={}/{}) ===",
                write_head_global, loop_count, max_loops
            );

            // Print raw hex data (first 100 words)
            eprintln!("Raw buffer hex (first 100 u32s):");
            for i in 0..100.min(u32_data.len()) {
                if i % 16 == 0 {
                    eprint!("\n{:04x}: ", i);
                }
                eprint!("{:08x} ", u32_data[i]);
            }
            eprintln!("\n");

            // Simple GDP decoder - inline
            // Format: header word + data words
            // Header: type (bits 0-7) | size (bits 8-31)
            // Type: 0x00=u32, 0x01=i32, 0x02=string, 0x03=f32
            let data_slice = &u32_data[data_start..data_start + data_len];
            let mut pos = 0;
            let mut count = 0;

            while pos < words_to_read {
                let header = data_slice[pos];
                let type_tag = header & 0xFF;
                let size = (header >> 8) as usize;

                if size == 0 || pos + 1 + size > data_len {
                    // Invalid or incomplete - stop
                    break;
                }

                match type_tag {
                    0x00 => {
                        // u32
                        let value = data_slice[pos + 1];
                        eprintln!("U: {}", value);
                    }
                    0x01 => {
                        // i32
                        let bits = data_slice[pos + 1];
                        let value = bits as i32;
                        eprintln!("I: {}", value);
                    }
                    0x02 => {
                        // string
                        let mut bytes = Vec::new();
                        for i in 0..size {
                            let word = data_slice[pos + 1 + i];
                            bytes.push((word & 0xFF) as u8);
                            bytes.push(((word >> 8) & 0xFF) as u8);
                            bytes.push(((word >> 16) & 0xFF) as u8);
                            bytes.push(((word >> 24) & 0xFF) as u8);
                        }
                        // Strip trailing zeros
                        while bytes.last() == Some(&0) {
                            bytes.pop();
                        }
                        let s = String::from_utf8_lossy(&bytes);
                        eprintln!("S: {}", s);
                    }
                    0x03 => {
                        // f32
                        let bits = data_slice[pos + 1];
                        let value = f32::from_bits(bits);
                        eprintln!("F: {}", value);
                    }
                    _ => {
                        eprintln!("Unknown type: 0x{:02x}", type_tag);
                        break;
                    }
                }

                pos += 1 + size;
                count += 1;
            }

            eprintln!("({} values decoded)", count);
            eprintln!("=================================\n");

            drop(data);
            staging_buffer.unmap();
        }
    }
}

// Load a .spv file and create a ShaderModule using the SPIR-V helper
fn load_spirv_module(device: &wgpu::Device, path: &Path) -> Result<wgpu::ShaderModule> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;

    // Check if SPIRV_SHADER_PASSTHROUGH is supported
    if device.features().contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH) {
        // Convert bytes to u32 words for SPIR-V passthrough
        let mut spirv_data = Vec::new();
        for chunk in bytes.chunks_exact(4) {
            let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            spirv_data.push(word);
        }

        // Use create_shader_module_passthrough to bypass wgpu's SPIR-V validation
        // This allows loading SPIR-V with unsupported capabilities like Linkage
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        let shader_module = unsafe {
            device.create_shader_module_passthrough(ShaderModuleDescriptorPassthrough::SpirV(
                wgpu::ShaderModuleDescriptorSpirV {
                    label: Some(&format!("{}", path.display())),
                    source: std::borrow::Cow::Borrowed(&spirv_data),
                },
            ))
        };

        // Check for validation errors even with passthrough
        let error_option = pollster::block_on(device.pop_error_scope());
        if let Some(error) = error_option {
            return Err(anyhow::Error::msg(format!(
                "Shader validation failed (passthrough): {}",
                error
            )));
        }

        Ok(shader_module)
    } else {
        // Fall back to regular shader module creation with validation
        let source = wgpu::util::make_spirv(&bytes);

        // Push an error scope to catch shader validation errors
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{}", path.display())),
            source,
        });

        // Check for validation errors
        let error_option = pollster::block_on(device.pop_error_scope());
        if let Some(error) = error_option {
            return Err(anyhow::Error::msg(format!("Shader validation failed: {}", error)));
        }

        Ok(shader_module)
    }
}

// --- Winit app shell ---------------------------------------------------------

struct App {
    state: Option<State>,
    spec: PipelineSpec,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = match event_loop.create_window(WindowAttributes::default().with_title("wgpu + SPIR-V"))
        {
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
                        state.print_debug_output();
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

async fn show_device_info() -> Result<()> {
    let instance = Instance::new(&InstanceDescriptor::default());

    // Try to get an adapter
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .context("No suitable adapter found")?;

    let info = adapter.get_info();
    let features = adapter.features();
    let limits = adapter.limits();

    println!("GPU Device Information:");
    println!("  Name: {}", info.name);
    println!("  Vendor: {:?}", info.vendor);
    println!("  Device: {}", info.device);
    println!("  Device Type: {:?}", info.device_type);
    println!("  Driver: {}", info.driver);
    println!("  Driver Info: {}", info.driver_info);
    println!("  Backend: {:?}", info.backend);

    println!("\nSupported Features:");
    println!("  {:#?}", features);

    println!("\nDevice Limits:");
    println!("  Max Texture Dimension 1D: {}", limits.max_texture_dimension_1d);
    println!("  Max Texture Dimension 2D: {}", limits.max_texture_dimension_2d);
    println!("  Max Texture Dimension 3D: {}", limits.max_texture_dimension_3d);
    println!("  Max Bind Groups: {}", limits.max_bind_groups);
    println!(
        "  Max Uniform Buffer Binding Size: {}",
        limits.max_uniform_buffer_binding_size
    );
    println!(
        "  Max Storage Buffer Binding Size: {}",
        limits.max_storage_buffer_binding_size
    );

    Ok(())
}

fn main() -> Result<()> {
    // Parse CLI and map to our pipeline spec
    let cli = Cli::parse();

    match cli.command {
        Command::Info => {
            pollster::block_on(show_device_info())?;
            return Ok(());
        }
        Command::VertexFragment {
            path,
            vertex,
            fragment,
            shadertoy,
            max_frames,
            verbose,
        } => {
            // Resolve entry points: use provided names or auto-detect
            let (vertex_name, fragment_name) = resolve_entry_points(&path, vertex, fragment)?;

            let spec = PipelineSpec::VertexFragment {
                path,
                vertex: vertex_name,
                fragment: fragment_name,
                shadertoy,
                max_frames,
                verbose,
            };

            let event_loop = EventLoop::new().context("failed to create event loop")?;
            let mut app = App { state: None, spec };

            if let Err(e) = event_loop.run_app(&mut app) {
                return Err(anyhow!(e)).context("winit event loop errored");
            }
        }
        Command::Compute {
            path,
            entry,
            workgroups_x,
            workgroups_y,
            workgroups_z,
            verbose,
        } => {
            let entry_name = entry.unwrap_or_else(|| {
                auto_detect_compute_entry_point(&path).unwrap_or_else(|e| {
                    eprintln!("{}", e);
                    std::process::exit(1);
                })
            });

            pollster::block_on(run_compute_shader(
                path,
                entry_name,
                (workgroups_x, workgroups_y, workgroups_z),
                verbose,
            ))?;
        }
    }

    Ok(())
}
