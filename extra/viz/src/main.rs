// src/main.rs — argument parsing and command dispatch only. Each
// subcommand's implementation lives in its own `modes::*` submodule;
// shared helpers live in `gpu`, `spirv`, `specs`, `json`, and `app`.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use wgpu::PresentMode;

mod app;
mod gpu;
mod json;
mod modes;
mod specs;
mod spirv;

use crate::specs::PushConstantSpec;

#[derive(Parser, Debug)]
#[command(
    name = "viz",
    about = "Headless and interactive runner for SPIR-V / WGSL shaders.",
    long_about = "Headless and interactive runner for SPIR-V / WGSL shaders.\n\
                  \n\
                  Drives the wyn compiler's output (and hand-rolled modules):\n\
                  interactive vertex+fragment viewing, headless compute,\n\
                  descriptor-driven pipelines, naga validation, and adapter\n\
                  inspection.",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, Default)]
enum PresentModeArg {
    #[default]
    Fifo,
    Mailbox,
    Immediate,
}

impl From<PresentModeArg> for PresentMode {
    fn from(arg: PresentModeArg) -> Self {
        match arg {
            PresentModeArg::Fifo => PresentMode::Fifo,
            PresentModeArg::Mailbox => PresentMode::Mailbox,
            PresentModeArg::Immediate => PresentMode::Immediate,
        }
    }
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, Default)]
enum TopologyArg {
    #[default]
    TriangleList,
    TriangleStrip,
    LineList,
    LineStrip,
    PointList,
}

impl From<TopologyArg> for wgpu::PrimitiveTopology {
    fn from(arg: TopologyArg) -> Self {
        match arg {
            TopologyArg::TriangleList => wgpu::PrimitiveTopology::TriangleList,
            TopologyArg::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
            TopologyArg::LineList => wgpu::PrimitiveTopology::LineList,
            TopologyArg::LineStrip => wgpu::PrimitiveTopology::LineStrip,
            TopologyArg::PointList => wgpu::PrimitiveTopology::PointList,
        }
    }
}

/// Parse 76-byte raw header hex into 19 big-endian u32 words for SHA256.
fn parse_size(s: &str) -> std::result::Result<(u32, u32), String> {
    let sep = if s.contains('x') { 'x' } else { ',' };
    let parts: Vec<&str> = s.splitn(2, sep).collect();
    if parts.len() != 2 {
        return Err(format!("expected WxH or W,H, got '{s}'"));
    }
    let w = parts[0].parse::<u32>().map_err(|e| format!("bad width: {e}"))?;
    let h = parts[1].parse::<u32>().map_err(|e| format!("bad height: {e}"))?;
    Ok((w, h))
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run a pipeline described by a JSON pipeline descriptor
    #[command(name = "pipeline", visible_alias = "run")]
    Pipeline {
        /// Path to the SPIR-V module
        path: PathBuf,
        /// Path to the pipeline descriptor JSON. Defaults to
        /// `<spv-path>.json` (the file `wyn compile` writes next to
        /// its output by default), so this only needs to be supplied
        /// when the descriptor lives somewhere else.
        #[arg(long, short)]
        pipeline: Option<PathBuf>,
        /// Input data: name:file.json (repeatable)
        #[arg(long = "input", value_name = "NAME:FILE")]
        inputs: Vec<String>,
        /// Upload an image file (PNG/JPEG) as the texture binding named
        /// NAME (repeatable). The shader side is a plain
        /// `#[texture(...)] NAME: texture2d` entry parameter; the
        /// texture is Rgba8Unorm at the image's native size, uploaded
        /// once at startup. Interactive mode only.
        ///
        /// Format: `NAME:FILE`. Example: `input_image:photo.png`.
        #[arg(long = "image", value_name = "NAME:FILE", verbatim_doc_comment)]
        images: Vec<String>,
        /// Read back the storage texture behind the named resource and
        /// write it as a PNG when the run ends (repeatable). Float
        /// formats are clamped to [0,1]; r32float dumps as grayscale.
        /// Pair with --max-frames for a deterministic snapshot.
        ///
        /// Format: `NAME:FILE`. Example: `ao_final:ao.png` (NAME is the
        /// storage-write view's parameter name).
        #[arg(long = "dump-texture", value_name = "NAME:FILE", verbatim_doc_comment)]
        dump_textures: Vec<String>,
        /// Seed a host storage buffer once at startup, by binding name
        /// (repeatable). Use for a host-provided buffer the shader
        /// writes but no `--input`/`--feedback` supplies — e.g. a
        /// scatter framebuffer (`0`) or a random initial particle state
        /// (`rng`).
        ///
        /// `SPEC` is `0` (zero-filled) or `rng` (uniform-random `f32`
        /// in `[0, 1)`, one per 4 bytes).
        ///
        /// The byte count is taken from the descriptor's
        /// `length: { fixed, bytes }` for that binding when the
        /// compiler could infer it (typically inputs the shader slices
        /// as `param[0..K]`). For bindings the compiler can't size
        /// (e.g. a framebuffer the shader iterates via `length(fb)`),
        /// declare it explicitly with `--storage-bytes NAME:BYTES`.
        ///
        /// Format: `NAME:SPEC`. Examples: `fb:0`, `state:rng`.
        #[arg(long = "buffer-init", value_name = "NAME:SPEC", verbatim_doc_comment)]
        buffer_inits: Vec<String>,
        /// Declare the byte size for a storage buffer whose descriptor
        /// `length` is `null` (the compiler couldn't infer it).
        /// Repeatable. Pairs with `--buffer-init NAME:SPEC` (or any
        /// other consumer that needs to know the allocation size).
        ///
        /// Format: `NAME:BYTES`. Example: `fb:4194304` (a 512x512x16
        /// `vec4f32` framebuffer).
        #[arg(long = "storage-bytes", value_name = "NAME:BYTES", verbatim_doc_comment)]
        storage_bytes: Vec<String>,
        /// Declare a storage binding as a framebuffer (repeatable):
        /// allocate `W × H × format.bytes_per_texel()` bytes from
        /// `--size W×H`, zero-initialize once, treat it as the
        /// scatter target for the graphics pipeline. Shorthand for
        /// `--buffer-init NAME:0 --storage-bytes NAME:W*H*B`.
        ///
        /// `FORMAT` defaults to `vec4f32` when omitted. Supported
        /// formats today: `vec4f32`.
        ///
        /// Format: `NAME[:FORMAT]`. Examples: `fb`, `fb:vec4f32`.
        #[arg(long = "framebuffer", value_name = "NAME[:FORMAT]", verbatim_doc_comment)]
        framebuffers: Vec<String>,
        /// Output file: name:file.json (repeatable, omit to print to stdout)
        #[arg(long = "output", value_name = "NAME:FILE")]
        outputs: Vec<String>,
        /// Push constant (repeatable).
        ///
        /// Format: `name:type=value`. Type is one of i32, u32, f32,
        /// or i32xN / u32xN / f32xN for fixed-size arrays.
        ///
        /// Examples:
        ///   "n:i32=64"
        ///   "header_base:u32x19=0,0,0,..."
        #[arg(long = "push-constant", value_name = "SPEC", verbatim_doc_comment, value_parser = PushConstantSpec::parse)]
        push_constants: Vec<PushConstantSpec>,
        /// Set a uniform block member once at startup (repeatable).
        /// Placement comes from the descriptor's published member
        /// layout; value syntax matches --push-constant.
        ///
        /// Format: `name.member:type=value` (or `name:type=value` for
        /// a bare scalar/vector uniform).
        ///
        /// Examples:
        ///   "c.radius:f32=0.35"
        ///   "c.tint:f32x2=0.9,0.2"
        #[arg(long = "uniform", value_name = "SPEC", verbatim_doc_comment, value_parser = specs::UniformSpec::parse)]
        uniform_values: Vec<specs::UniformSpec>,
        /// Override the per-frame compute dispatch for one compute
        /// entry (repeatable). Format: `ENTRY:WxH[xD]`.
        ///
        /// `ENTRY` matches the compute pipeline's `entry_point`.
        /// `W` / `H` / `D` are total thread counts on each axis (viz
        /// divides by the descriptor's `workgroup_size` to compute
        /// workgroup counts). `D` defaults to 1 when omitted.
        ///
        /// Only meaningful in interactive mode (when the descriptor
        /// has a graphics pipeline); ignored in headless mode. Useful
        /// when the compiler's default dispatch doesn't match the
        /// resource you want to fill — e.g. a storage-image-writing
        /// compute whose default is 1 thread.
        ///
        /// Examples:
        ///   "paint:1024x1024"
        ///   "erode_b:512x512x1"
        #[arg(long = "dispatch", value_name = "ENTRY:WxH[xD]", verbatim_doc_comment)]
        dispatch: Vec<String>,
        /// Wire a ping-pong feedback pair: the binding named `READ` in
        /// the compute entry `ENTRY` reads the *previous frame*'s value
        /// of the storage_image named `WRITE` (Shadertoy-style
        /// self-feedback). The host allocates two physical textures for
        /// the pair and swaps which is bound each frame.
        ///
        /// Format: `ENTRY:READ=WRITE`. Repeatable.
        ///
        /// Example:
        ///   "buffer_a:prev_a=out_a"
        #[arg(long = "feedback", value_name = "ENTRY:READ=WRITE", verbatim_doc_comment)]
        feedback: Vec<String>,
        /// Directory of per-binding files. For each `storage_buffer`
        /// declared in the descriptor (other than recognized
        /// host-uploaded names like `keyboard`), and for each
        /// `vertex_inputs[i]` declared on a graphics pipeline, viz
        /// loads `<dir>/<name>.bin` — a flat little-endian byte file —
        /// and binds it as a storage buffer or a vertex buffer.
        #[arg(long)]
        storage_dir: Option<PathBuf>,
        /// Flat little-endian u32 index buffer file. When present, viz
        /// sets it via `set_index_buffer` and dispatches `draw_indexed`
        /// with `file_size / 4` indices (`--vertex-count` is ignored).
        /// Without this, viz does a non-indexed `draw(0..vertex_count)`.
        #[arg(long)]
        index_buffer: Option<PathBuf>,
        /// Present mode: fifo (vsync), mailbox (triple-buffer), immediate (no sync)
        #[arg(long, value_enum, default_value = "fifo")]
        present_mode: PresentModeArg,
        /// Disable GPU validation layers (validation is ON by default)
        #[arg(long)]
        no_validate: bool,
        /// Window size as WxH (e.g. --size 256x256)
        #[arg(long, value_parser = parse_size)]
        size: Option<(u32, u32)>,
        /// Maximum number of frames to render before exiting (for debugging)
        #[arg(long)]
        max_frames: Option<u32>,
        /// Vertex count for the draw call. Default 3 matches a
        /// fullscreen-triangle vertex shader; bump to the number of
        /// invocations a `vertex_index`-driven vertex shader expects.
        /// Ignored when `--index-buffer` is supplied (the index file's
        /// length drives `draw_indexed` instead).
        #[arg(long, default_value = "3")]
        vertex_count: u32,
        /// Primitive topology for the draw call.
        #[arg(long, value_enum, default_value = "triangle-list")]
        topology: TopologyArg,
        /// Print verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    /// Show device and driver information
    #[command(name = "info")]
    Info,
    /// Validate a SPIR-V module without rendering (headless naga validation)
    #[command(name = "validate")]
    Validate {
        /// Path to the SPIR-V module
        path: PathBuf,
        /// Print verbose output (show adapter info, entry points)
        #[arg(short, long)]
        verbose: bool,
    },
    /// Render a built-in test pattern (no shader file needed, always validates)
    #[command(name = "testpattern")]
    TestPattern {
        /// Maximum number of frames to render before exiting
        #[arg(long)]
        max_frames: Option<u32>,
        /// Print frame timing statistics
        #[arg(short, long)]
        verbose: bool,
    },
}

fn main() -> Result<()> {
    // Install a logger backend so wgpu's `log::warn!` / validation
    // messages actually go somewhere. Off until RUST_LOG asks for it,
    // so the default invocation stays quiet.
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Command::Info => {
            pollster::block_on(modes::info::show_device_info())?;
        }
        Command::Pipeline {
            path,
            pipeline,
            inputs,
            images,
            dump_textures,
            buffer_inits,
            storage_bytes,
            framebuffers,
            outputs,
            push_constants,
            uniform_values,
            dispatch,
            feedback,
            storage_dir,
            index_buffer,
            present_mode,
            no_validate,
            size,
            max_frames,
            vertex_count,
            topology,
            verbose,
        } => {
            let parse_pairs = |pairs: &[String]| -> Result<HashMap<String, PathBuf>> {
                pairs
                    .iter()
                    .map(|s| {
                        let (name, file) = s
                            .split_once(':')
                            .ok_or_else(|| anyhow!("Invalid format '{}'. Expected name:file.json", s))?;
                        Ok((name.to_string(), PathBuf::from(file)))
                    })
                    .collect()
            };
            let input_map = parse_pairs(&inputs)?;
            let image_map = parse_pairs(&images)?;
            let dump_texture_map = parse_pairs(&dump_textures)?;
            let output_map = parse_pairs(&outputs)?;

            // Parse `--buffer-init NAME:SPEC` into a name → spec map.
            // The byte count is resolved later from
            // `--storage-bytes` or the descriptor's `length`.
            let buffer_init_specs: HashMap<String, gpu::BufferInitSpec> = buffer_inits
                .iter()
                .map(|s| {
                    let (name, spec) = s
                        .split_once(':')
                        .ok_or_else(|| anyhow!("Invalid --buffer-init '{}'. Expected NAME:SPEC", s))?;
                    let spec = match spec {
                        "0" => gpu::BufferInitSpec::Zero,
                        "rng" => gpu::BufferInitSpec::Rng,
                        other => {
                            return Err(anyhow!(
                                "Invalid --buffer-init '{}': unknown spec '{}'. Expected '0' or 'rng'",
                                s,
                                other
                            ));
                        }
                    };
                    Ok((name.to_string(), spec))
                })
                .collect::<Result<HashMap<_, _>>>()?;

            // Parse `--storage-bytes NAME:BYTES` into a name → bytes map.
            let storage_bytes_map: HashMap<String, u64> = storage_bytes
                .iter()
                .map(|s| {
                    let (name, bytes) = s
                        .split_once(':')
                        .ok_or_else(|| anyhow!("Invalid --storage-bytes '{}'. Expected NAME:BYTES", s))?;
                    let bytes: u64 = bytes
                        .parse()
                        .with_context(|| format!("--storage-bytes '{}': cannot parse byte size", s))?;
                    Ok((name.to_string(), bytes))
                })
                .collect::<Result<HashMap<_, _>>>()?;

            // Parse `--framebuffer NAME[:FORMAT]` into a name → format map.
            // Empty FORMAT defaults to `vec4f32`.
            let framebuffer_map: HashMap<String, modes::pipeline::FramebufferFormat> = framebuffers
                .iter()
                .map(|s| {
                    let (name, format) = match s.split_once(':') {
                        Some((n, f)) => (n, f.parse::<modes::pipeline::FramebufferFormat>()?),
                        None => (s.as_str(), modes::pipeline::FramebufferFormat::default()),
                    };
                    Ok((name.to_string(), format))
                })
                .collect::<Result<HashMap<_, _>>>()?;

            // Parse `--dispatch ENTRY:WxH[xD]` into a HashMap keyed by
            // compute entry-point name. Total thread counts; viz
            // divides by the descriptor's workgroup_size at use.
            let dispatch_overrides: HashMap<String, (u32, u32, u32)> = dispatch
                .iter()
                .map(|s| {
                    let (entry, dims) = s
                        .split_once(':')
                        .ok_or_else(|| anyhow!("Invalid --dispatch '{}'. Expected ENTRY:WxH[xD]", s))?;
                    let parts: Vec<&str> = dims.split('x').collect();
                    let parse = |p: &str| -> Result<u32> {
                        p.parse().with_context(|| format!("--dispatch '{}': cannot parse '{}'", s, p))
                    };
                    let (w, h, d) = match parts.as_slice() {
                        [w, h] => (parse(w)?, parse(h)?, 1u32),
                        [w, h, d] => (parse(w)?, parse(h)?, parse(d)?),
                        _ => {
                            return Err(anyhow!(
                                "Invalid --dispatch '{}'. Expected WxH or WxHxD after ':'",
                                s
                            ));
                        }
                    };
                    Ok((entry.to_string(), (w, h, d)))
                })
                .collect::<Result<HashMap<_, _>>>()?;

            // Parse `--feedback ENTRY:READ=WRITE` into (entry, read,
            // write) triples. Names are resolved to (set, binding) when
            // we have the descriptor in hand.
            let feedback_specs: Vec<(String, String, String)> = feedback
                .iter()
                .map(|s| {
                    let (entry, rw) = s
                        .split_once(':')
                        .ok_or_else(|| anyhow!("Invalid --feedback '{}'. Expected ENTRY:READ=WRITE", s))?;
                    let (read, write) = rw
                        .split_once('=')
                        .ok_or_else(|| anyhow!("Invalid --feedback '{}'. Expected ENTRY:READ=WRITE", s))?;
                    Ok((entry.to_string(), read.to_string(), write.to_string()))
                })
                .collect::<Result<Vec<_>>>()?;

            let pipeline_path = pipeline.unwrap_or_else(|| path.with_extension("json"));
            pollster::block_on(modes::pipeline::run_pipeline(
                path,
                pipeline_path,
                input_map,
                output_map,
                &push_constants,
                &dispatch_overrides,
                &feedback_specs,
                modes::pipeline::InteractiveOpts {
                    storage_dir,
                    buffer_inits: buffer_init_specs,
                    storage_bytes: storage_bytes_map,
                    framebuffers: framebuffer_map,
                    index_buffer,
                    present_mode: present_mode.into(),
                    validate: !no_validate,
                    size,
                    max_frames,
                    vertex_count,
                    topology: topology.into(),
                    images: image_map,
                    dump_textures: dump_texture_map,
                    uniform_values,
                },
                verbose,
            ))?;
        }
        Command::Validate { path, verbose } => {
            let is_wgsl = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("wgsl"))
                .unwrap_or(false);
            if is_wgsl {
                modes::validate::validate_wgsl_file(&path, verbose)?;
            } else {
                pollster::block_on(modes::validate::validate_spirv(&path, verbose))?;
            }
        }
        Command::TestPattern { max_frames, verbose } => {
            modes::testpattern::run_test_pattern(max_frames, verbose)?;
        }
    }

    Ok(())
}
