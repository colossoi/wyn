//! `pipeline` subcommand — execute a pipeline described by a JSON
//! pipeline-descriptor sidecar (the same format `wyn-core` emits),
//! reading inputs from `--input name:file.json` and writing outputs
//! to `--output name:file.json` (or stdout if no output path).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use wgpu::{CommandEncoderDescriptor, PipelineLayoutDescriptor};
use winit::event_loop::EventLoop;

use crate::app::{App, InteractivePipelineSpec};
use crate::gpu::{
    ComputeExecutor, build_bind_group, build_push_constant_bytes, create_binding_buffers,
    create_headless_device, readback_buffer, resolve_dispatch_size,
};
use crate::json::{
    Binding, BufferUsage, ComputePipeline, Pipeline, PipelineDescriptor, write_f32_json,
};
use crate::specs::PushConstantSpec;
use crate::spirv::load_spirv_module;
use wyn_pipeline_descriptor::ShaderStage;

/// Knobs that only mean anything on the interactive path. Bundled so
/// `run_pipeline`'s signature doesn't grow N positional args every
/// time the interactive mode learns another flag.
pub struct InteractiveOpts {
    pub storage_dir: Option<PathBuf>,
    /// Host storage buffers to allocate and seed once, by binding name →
    /// init spec (from `--buffer-init NAME:SPEC`). The byte size is
    /// resolved at runtime per binding: explicit `--storage-bytes`
    /// override first, otherwise the descriptor's `length: Fixed
    /// { bytes }`, otherwise an error.
    pub buffer_inits: HashMap<String, crate::gpu::BufferInitSpec>,
    /// Explicit byte size for a binding (from `--storage-bytes
    /// NAME:BYTES`). Backs any binding the descriptor publishes with
    /// `length: null` (e.g. a framebuffer the shader iterates via
    /// `length(fb)`).
    pub storage_bytes: HashMap<String, u64>,
    /// Bindings declared as framebuffers (from `--framebuffer
    /// NAME[:FORMAT]`). Sized by `--size W×H × format.bytes_per_texel()`
    /// and always zero-initialized. Shorthand for the common
    /// `--storage-bytes NAME:W*H*B --buffer-init NAME:0` pair.
    pub framebuffers: HashMap<String, FramebufferFormat>,
    pub index_buffer: Option<PathBuf>,
    pub present_mode: wgpu::PresentMode,
    pub validate: bool,
    pub size: Option<(u32, u32)>,
    pub max_frames: Option<u32>,
    pub vertex_count: u32,
    pub topology: wgpu::PrimitiveTopology,
    /// Image files to upload as host textures, by binding name (from
    /// `--image NAME:FILE`). Decoded eagerly on the interactive path.
    pub images: HashMap<String, PathBuf>,
    /// Storage textures to dump as PNG at the `--max-frames` exit, by
    /// binding name (from `--dump-texture NAME:FILE`).
    pub dump_textures: HashMap<String, PathBuf>,
}

/// Per-texel format for a `--framebuffer` binding. v1 supports
/// `vec4f32` only — the format the wyn-emitted graphics pipelines
/// scatter into today.
#[derive(Debug, Clone, Copy)]
pub enum FramebufferFormat {
    Vec4F32,
}

impl Default for FramebufferFormat {
    fn default() -> Self {
        Self::Vec4F32
    }
}

impl FramebufferFormat {
    pub fn bytes_per_texel(self) -> u64 {
        match self {
            Self::Vec4F32 => 16,
        }
    }
}

impl std::str::FromStr for FramebufferFormat {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "vec4f32" => Ok(Self::Vec4F32),
            other => Err(anyhow!(
                "--framebuffer: unsupported format '{other}'. Supported: vec4f32"
            )),
        }
    }
}

impl std::fmt::Display for FramebufferFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vec4F32 => f.write_str("vec4f32"),
        }
    }
}

/// Resolve every host-allocated storage buffer the run needs into a
/// concrete `BufferInit` keyed by binding name. Two flag families feed
/// in:
///
///   * `--framebuffer NAME[:FORMAT]` — sized as `W × H ×
///     format.bytes_per_texel()` from `--size`, zero-initialized. Hard
///     errors if the same name also appears in `--buffer-init`,
///     `--storage-bytes`, or the descriptor's `length: Fixed { bytes }`
///     with a disagreeing byte count.
///   * `--buffer-init NAME:SPEC` — size precedence:
///       1. `--storage-bytes NAME:BYTES` if supplied;
///       2. otherwise the descriptor's `length: Fixed { bytes }`.
///     Both present → must agree. Neither present → error pointing at
///     `--storage-bytes`.
fn resolve_buffer_inits(
    desc: &PipelineDescriptor,
    inits: &HashMap<String, crate::gpu::BufferInitSpec>,
    storage_bytes: &HashMap<String, u64>,
    framebuffers: &HashMap<String, FramebufferFormat>,
    size: Option<(u32, u32)>,
) -> Result<HashMap<String, crate::gpu::BufferInit>> {
    use wyn_pipeline_descriptor::{Binding, BufferLen};
    let descriptor_bytes: HashMap<&str, u64> = desc
        .pipelines
        .iter()
        .flat_map(|p| match p {
            Pipeline::Compute(cp) => cp.bindings.as_slice(),
            Pipeline::Graphics(gp) => gp.bindings.as_slice(),
        })
        .filter_map(|b| match b {
            Binding::StorageBuffer {
                name,
                length: Some(BufferLen::Fixed { bytes }),
                ..
            } => Some((name.as_str(), *bytes)),
            _ => None,
        })
        .collect();

    let mut out = HashMap::new();

    for (name, &format) in framebuffers {
        if inits.contains_key(name) {
            return Err(anyhow!(
                "--framebuffer {name} conflicts with --buffer-init {name}:<spec>; \
                 framebuffers are always zero-initialized"
            ));
        }
        if storage_bytes.contains_key(name) {
            return Err(anyhow!(
                "--framebuffer {name} conflicts with --storage-bytes {name}:<bytes>; \
                 framebuffer size is computed from --size and format"
            ));
        }
        let (w, h) = size.ok_or_else(|| {
            anyhow!("--framebuffer {name} requires --size W×H to compute byte count")
        })?;
        let bytes = w as u64 * h as u64 * format.bytes_per_texel();
        if let Some(&desc_bytes) = descriptor_bytes.get(name.as_str()) {
            if desc_bytes != bytes {
                return Err(anyhow!(
                    "--framebuffer {name} (--size {w}×{h} × {format} = {bytes}) \
                     disagrees with the descriptor's length:{{fixed,bytes:{desc_bytes}}} \
                     on binding `{name}`"
                ));
            }
        }
        out.insert(name.clone(), crate::gpu::BufferInit {
            bytes,
            spec: crate::gpu::BufferInitSpec::Zero,
        });
    }

    for (name, &spec) in inits {
        let from_flag = storage_bytes.get(name).copied();
        let from_desc = descriptor_bytes.get(name.as_str()).copied();
        let bytes = match (from_flag, from_desc) {
            (Some(a), Some(b)) if a != b => {
                return Err(anyhow!(
                    "--storage-bytes {name}:{a} disagrees with the descriptor's \
                     length:{{fixed,bytes:{b}}} on binding `{name}`; the two must \
                     match (or drop one of them)"
                ));
            }
            (Some(a), _) => a,
            (None, Some(b)) => b,
            (None, None) => {
                return Err(anyhow!(
                    "--buffer-init {name}:<spec>: descriptor doesn't publish a fixed \
                     length for binding `{name}`; declare its byte size with \
                     `--storage-bytes {name}:BYTES`"
                ));
            }
        };
        out.insert(name.clone(), crate::gpu::BufferInit { bytes, spec });
    }
    Ok(out)
}

pub async fn run_pipeline(
    spv_path: PathBuf,
    pipeline_path: PathBuf,
    inputs: HashMap<String, PathBuf>,
    outputs: HashMap<String, PathBuf>,
    push_constants: &[PushConstantSpec],
    dispatch_overrides: &HashMap<String, (u32, u32, u32)>,
    feedback_specs: &[(String, String, String)],
    interactive_opts: InteractiveOpts,
    verbose: bool,
) -> Result<()> {
    let desc_json = fs::read_to_string(&pipeline_path)
        .with_context(|| format!("Failed to read pipeline descriptor: {}", pipeline_path.display()))?;
    let desc: PipelineDescriptor =
        serde_json::from_str(&desc_json).with_context(|| "Failed to parse pipeline descriptor JSON")?;

    if desc.pipelines.is_empty() {
        return Err(anyhow!("Pipeline descriptor has no pipelines"));
    }

    // Auto-detect interactive vs headless. A descriptor with a
    // `Graphics` pipeline asks for a window: switch to the interactive
    // path. Otherwise stick with the original headless compute runner.
    let has_graphics = desc.pipelines.iter().any(|p| matches!(p, Pipeline::Graphics(_)));
    if has_graphics {
        if !inputs.is_empty() {
            eprintln!(
                "[viz pipeline] --input is ignored in interactive mode \
                 (descriptor has a graphics pipeline; inputs come from --feedback / \
                 --buffer-init / --storage-dir)"
            );
        }
        if !outputs.is_empty() && interactive_opts.max_frames.is_none() {
            eprintln!(
                "[viz pipeline] --output in interactive mode dumps buffers at the \
                 --max-frames exit; without --max-frames the run never reaches the \
                 dump. Pass --max-frames N to snapshot after N frames."
            );
        }
        return run_pipeline_interactive(
            spv_path,
            desc,
            dispatch_overrides.clone(),
            feedback_specs.to_vec(),
            outputs,
            interactive_opts,
            verbose,
        );
    }
    if !feedback_specs.is_empty() {
        eprintln!(
            "[viz pipeline] --feedback is ignored in headless mode \
             (no frames, no previous-state notion)"
        );
    }
    if !interactive_opts.images.is_empty() {
        eprintln!(
            "[viz pipeline] --image is ignored in headless mode \
             (texture bindings are only wired on the interactive path)"
        );
    }

    let (device, queue) = create_headless_device(verbose).await?;
    let module = load_spirv_module(&device, &spv_path)?;

    for (pi, pipeline) in desc.pipelines.iter().enumerate() {
        match pipeline {
            Pipeline::Compute(cp) => {
                run_compute(&device, &queue, &module, cp, &inputs, &outputs, push_constants, verbose)
                    .with_context(|| format!("Pipeline {} (compute) failed", pi))?;
            }
            Pipeline::Graphics(_) => {
                // Unreachable now (caught above), kept for completeness
                // if a future descriptor shape allows mixed headless +
                // graphics dispatches.
            }
        }
    }

    Ok(())
}

/// Interactive path. Opens a window, runs every compute pipeline in
/// the descriptor each frame, then renders the one graphics pipeline.
/// Activated automatically when the descriptor contains a graphics
/// pipeline (the headless `--input` / `--output` flags are ignored).
fn run_pipeline_interactive(
    spv_path: PathBuf,
    desc: PipelineDescriptor,
    dispatch_overrides: HashMap<String, (u32, u32, u32)>,
    feedback_specs: Vec<(String, String, String)>,
    outputs: HashMap<String, PathBuf>,
    opts: InteractiveOpts,
    verbose: bool,
) -> Result<()> {
    // Resolve the vertex and fragment entry-point names from the
    // descriptor. Wyn emits one Graphics pipeline per entry point, so
    // a vertex-only and a fragment-only pipeline coexist; we collect
    // their stage names and re-merge here.
    let stages: Vec<&wyn_pipeline_descriptor::GraphicsStage> = desc
        .pipelines
        .iter()
        .filter_map(|p| if let Pipeline::Graphics(g) = p { Some(g) } else { None })
        .flat_map(|g| g.stages.iter())
        .collect();
    let vertex_entry = stages
        .iter()
        .find_map(|s| matches!(s.stage, ShaderStage::Vertex).then(|| s.entry_point.clone()))
        .ok_or_else(|| anyhow!("descriptor lacks a vertex stage"))?;
    let fragment_entry = stages
        .iter()
        .find_map(|s| matches!(s.stage, ShaderStage::Fragment).then(|| s.entry_point.clone()))
        .ok_or_else(|| anyhow!("descriptor lacks a fragment stage"))?;

    let resolved_buffer_inits = resolve_buffer_inits(
        &desc,
        &opts.buffer_inits,
        &opts.storage_bytes,
        &opts.framebuffers,
        opts.size,
    )?;
    // Decode `--image` files eagerly so a bad path or unsupported
    // format fails before the window opens.
    let images: HashMap<String, crate::gpu::LoadedImage> = opts
        .images
        .iter()
        .map(|(name, path)| {
            let img = image::open(path)
                .with_context(|| format!("--image {}: failed to load {}", name, path.display()))?
                .to_rgba8();
            let (width, height) = img.dimensions();
            if verbose {
                println!("[viz pipeline] --image {}: {} ({}x{})", name, path.display(), width, height);
            }
            Ok((
                name.clone(),
                crate::gpu::LoadedImage {
                    rgba8: img.into_raw(),
                    width,
                    height,
                },
            ))
        })
        .collect::<Result<_>>()?;
    let spec = InteractivePipelineSpec {
        shader_path: spv_path,
        descriptor: desc,
        vertex_entry,
        fragment_entry,
        dispatch_overrides,
        feedback_specs,
        max_frames: opts.max_frames,
        verbose,
        validate: opts.validate,
        present_mode: opts.present_mode,
        size: opts.size,
        vertex_count: opts.vertex_count,
        topology: opts.topology,
        storage_dir: opts.storage_dir,
        buffer_inits: resolved_buffer_inits,
        index_buffer: opts.index_buffer,
        outputs,
        images,
        dump_textures: opts.dump_textures,
    };

    let event_loop = EventLoop::new().context("failed to create event loop")?;
    let mut app = App::new_pipeline(spec);
    event_loop.run_app(&mut app).map_err(|e| anyhow!(e)).context("winit event loop errored")?;
    Ok(())
}

/// Create wgpu buffers for a set of bindings. Returns a map from binding number

/// Run a compute pipeline as a sequence of dispatches over a shared
/// binding table. Single-stage and multi-stage pipelines flow through
/// the same path; the `mp.stages.len() == 1` case covers what used to
/// be `run_single_compute`.
fn run_compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    module: &wgpu::ShaderModule,
    mp: &ComputePipeline,
    inputs: &HashMap<String, PathBuf>,
    outputs: &HashMap<String, PathBuf>,
    push_constants: &[PushConstantSpec],
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("Running compute pipeline ({} stages)", mp.stages.len());
        for (i, stage) in mp.stages.iter().enumerate() {
            println!(
                "  Stage {}: {} (reads {:?}, writes {:?})",
                i, stage.entry_point, stage.reads, stage.writes
            );
        }
    }

    // Build push constant data from CLI args matched against descriptor bindings
    let pc_bytes = build_push_constant_bytes(&mp.bindings, push_constants, verbose)?;
    let total_pc_size = pc_bytes.len() as u32;

    // Stage-0's dispatch sizes any `SameAsDispatch` output bindings —
    // the single-stage case carries the only-stage's dispatch, the
    // multi-stage case carries phase 1's, which is the size primary
    // outputs are tied to in current reduce/scan/scheduler layouts.
    let dispatch_hint = mp.stages.first().map(|s| &s.dispatch_size);
    let buffers =
        create_binding_buffers(device, queue, &mp.bindings, inputs, dispatch_hint, &pc_bytes, verbose)?;
    let (layout, bind_group) = build_bind_group(device, &mp.bindings, &buffers)?;

    let pc_ranges: Vec<wgpu::PushConstantRange> = if total_pc_size > 0 {
        vec![wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..total_pc_size,
        }]
    } else {
        vec![]
    };

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("multi_compute_layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &pc_ranges,
    });

    // Execute stages in order
    for (si, stage) in mp.stages.iter().enumerate() {
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("stage_{}", stage.entry_point)),
            layout: Some(&pipeline_layout),
            module,
            entry_point: Some(&stage.entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        let dispatch = resolve_dispatch_size(&stage.dispatch_size, &buffers, &pc_bytes);
        if verbose {
            println!(
                "Stage {} ({}): dispatch {} x {} x {}",
                si, stage.entry_point, dispatch.0, dispatch.1, dispatch.2
            );
        }

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some(&format!("stage_{}_encoder", si)),
        });
        let stage_label = format!("stage_{}", si);
        ComputeExecutor {
            label: &stage_label,
            pipeline: &pipeline,
            bind_groups: &[&bind_group],
            push_constant_bytes: &pc_bytes,
            dispatch,
            timestamps: None,
        }
        .record(&mut encoder);
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    // Read back and output results
    output_results(device, queue, &mp.bindings, &buffers, outputs)?;

    Ok(())
}

/// Print f32 data to stdout — fallback for outputs without an
/// explicit `--output name:file.json` redirection.
fn print_f32_data(name: &str, data: &[f32]) {
    println!("\n=== {} ({} elements) ===", name, data.len());
    let show = data.len().min(64);
    for (i, chunk) in data[..show].chunks(8).enumerate() {
        print!("  [{:3}]: ", i * 8);
        for val in chunk {
            print!("{:8.3} ", val);
        }
        println!();
    }
    if data.len() > show {
        println!("  ... ({} more elements)", data.len() - show);
    }
    println!();
}

/// Read back output/intermediate buffers and write/print results.
fn output_results(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bindings: &[Binding],
    buffers: &HashMap<u32, (wgpu::Buffer, u64)>,
    outputs: &HashMap<String, PathBuf>,
) -> Result<()> {
    for b in bindings {
        if let Binding::StorageBuffer {
            binding, name, usage, ..
        } = b
        {
            // Only read back output and intermediate buffers (skip inputs unless
            // explicitly requested via --output)
            let should_output = *usage != BufferUsage::Input || outputs.contains_key(name.as_str());

            if !should_output {
                continue;
            }

            if let Some((buf, size)) = buffers.get(binding) {
                let data = readback_buffer(device, queue, buf, *size)?;

                if let Some(path) = outputs.get(name.as_str()) {
                    write_f32_json(path, &data)?;
                    println!("Wrote {} elements to {}", data.len(), path.display());
                } else {
                    print_f32_data(name, &data);
                }
            }
        }
    }
    Ok(())
}
