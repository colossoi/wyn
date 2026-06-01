// src/main.rs — argument parsing and command dispatch only. Each
// subcommand's implementation lives in its own `modes::*` submodule;
// shared helpers live in `gpu`, `spirv`, `specs`, `json`, and `app`.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Result, anyhow};
use clap::{Parser, Subcommand};
use wgpu::PresentMode;

mod app;
mod gpu;
mod json;
mod modes;
mod specs;
mod spirv;

use crate::specs::{PushConstantSpec, StorageBufferSpec, UniformSpec};
use crate::spirv::{auto_detect_compute_entry_point, resolve_entry_points};

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
        /// Disable GPU validation layers (validation is ON by default)
        #[arg(long)]
        no_validate: bool,
        /// Present mode: fifo (vsync), mailbox (triple-buffer), immediate (no sync)
        #[arg(long, value_enum, default_value = "fifo")]
        present_mode: PresentModeArg,
        /// Difficulty level for shaders that use it (binding 2)
        #[arg(long, default_value = "3")]
        difficulty: i32,
        /// Window size as WxH (e.g. --size 256x256)
        #[arg(long, value_parser = parse_size)]
        size: Option<(u32, u32)>,
        /// Vertex count for the draw call. Default 3 matches a fullscreen-triangle
        /// or basic-triangle shader; bump to the number of indices a
        /// `vertex_index`-driven vertex shader expects to dispatch.
        #[arg(long, default_value = "3")]
        vertex_count: u32,
        /// Primitive topology for the draw call.
        #[arg(long, value_enum, default_value = "triangle-list")]
        topology: TopologyArg,
        /// Directory of per-binding storage-buffer files. For each
        /// `storage_buffer` binding the sidecar declares, viz loads
        /// `<dir>/<name>.bin` — a flat little-endian byte file — and
        /// uploads it. Requires `--shadertoy` (the storage buffers
        /// share its bind group).
        #[arg(long)]
        storage_dir: Option<PathBuf>,
        /// Flat little-endian u32 index buffer file. When present, viz
        /// sets it via `set_index_buffer` and dispatches `draw_indexed`
        /// with `file_size / 4` indices (`--vertex-count` is ignored).
        /// Without this, viz does a non-indexed `draw(0..vertex_count)`.
        #[arg(long)]
        index_buffer: Option<PathBuf>,
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
        /// Storage buffer binding (repeatable).
        ///
        /// Format: one of
        ///   `binding:size:type`                  - set 0 (back-compat short form)
        ///   `set:binding:size:type`              - explicit set
        ///   `set:binding:size:type:input.json`   - explicit set + initial data
        ///
        /// Type: i32, u32, f32.
        ///
        /// Examples:
        ///   "1:64:i32"             - set 0 binding 1, 64 zero i32s
        ///   "0:1:64:i32"           - explicit set 0
        ///   "1:0:64:f32"           - set 1, binding 0
        ///   "0:0:8:f32:data.json"  - set 0 binding 0, 8 f32s from data.json
        #[arg(long = "storage", value_name = "SPEC", verbatim_doc_comment, value_parser = StorageBufferSpec::parse)]
        storage_buffers: Vec<StorageBufferSpec>,
        /// Uniform buffer binding (repeatable).
        ///
        /// Format: `set:binding:type=value[,value...]`. Type is one of
        /// i32, u32, f32, or vec2/vec3/vec4 of those.
        ///
        /// Examples:
        ///   "1:0:f32=0.5"
        ///   "1:1:vec3f32=1920,1080,1"
        ///   "0:2:i32=42"
        #[arg(long = "uniform", value_name = "SPEC", verbatim_doc_comment, value_parser = UniformSpec::parse)]
        uniforms: Vec<UniformSpec>,
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
        /// Print verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    /// Run a pipeline described by a JSON pipeline descriptor
    #[command(name = "pipeline", visible_alias = "run")]
    Pipeline {
        /// Path to the SPIR-V module
        path: PathBuf,
        /// Path to the pipeline descriptor JSON
        #[arg(long, short)]
        pipeline: PathBuf,
        /// Input data: name:file.json (repeatable)
        #[arg(long = "input", value_name = "NAME:FILE")]
        inputs: Vec<String>,
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
    /// Interactive simulation viewer: drives a SPIR-V module with one
    /// compute entry + one graphics entry. Each frame dispatches the
    /// compute entry to write the next generation into one of two
    /// ping-pong storage buffers, then renders the fragment reading
    /// that buffer. Conway's Game of Life is the motivating example.
    #[command(name = "simulate")]
    Simulate {
        /// Path to the SPIR-V module
        path: PathBuf,
        /// Compute entry name (auto-detected if there's exactly one
        /// `Compute` pipeline in the descriptor)
        #[arg(long)]
        compute: Option<String>,
        /// Vertex entry name (auto-detected)
        #[arg(long)]
        vertex: Option<String>,
        /// Fragment entry name (auto-detected)
        #[arg(long)]
        fragment: Option<String>,
        /// Grid dimensions WxH. Drives compute push constants
        /// (width/height) and the fragment's grid_width/grid_height
        /// uniforms.
        #[arg(long, value_parser = parse_size, default_value = "64x64")]
        grid: (u32, u32),
        /// Initial board state: a built-in name (`glider`, `blinker`,
        /// `beacon`, `pulsar`, `random`) or a path to a `.rle`,
        /// `.cells`, or `.bin` file. Default: `random` with a
        /// time-based seed.
        #[arg(long, default_value = "random")]
        pattern: String,
        /// RNG seed for `--pattern random`. Defaults to a time-based
        /// seed; pass an explicit value for reproducible runs.
        #[arg(long)]
        seed: Option<u64>,
        /// Maximum number of frames to render before exiting
        #[arg(long)]
        max_frames: Option<u32>,
        /// Print frame timing statistics
        #[arg(short, long)]
        verbose: bool,
        /// Disable GPU validation layers (validation is ON by default)
        #[arg(long)]
        no_validate: bool,
        /// Present mode: fifo (vsync), mailbox (triple-buffer), immediate (no sync)
        #[arg(long, value_enum, default_value = "fifo")]
        present_mode: PresentModeArg,
        /// Window size as WxH (e.g. --size 512x512)
        #[arg(long, value_parser = parse_size)]
        size: Option<(u32, u32)>,
        /// Vertex count for the draw call. Default 3 matches a
        /// fullscreen-triangle vertex shader.
        #[arg(long, default_value = "3")]
        vertex_count: u32,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Info => {
            pollster::block_on(modes::info::show_device_info())?;
        }
        Command::VertexFragment {
            path,
            vertex,
            fragment,
            shadertoy,
            max_frames,
            verbose,
            no_validate,
            present_mode,
            difficulty,
            size,
            vertex_count,
            topology,
            storage_dir,
            index_buffer,
        } => {
            let (vertex_name, fragment_name) = resolve_entry_points(&path, vertex, fragment)?;
            modes::vf::run_vertex_fragment(
                path,
                vertex_name,
                fragment_name,
                shadertoy,
                max_frames,
                verbose,
                !no_validate,
                present_mode.into(),
                difficulty,
                size,
                vertex_count,
                topology.into(),
                storage_dir,
                index_buffer,
            )?;
        }
        Command::Compute {
            path,
            entry,
            workgroups_x,
            workgroups_y,
            workgroups_z,
            storage_buffers,
            uniforms,
            push_constants,
            verbose,
        } => {
            let entry_name = entry.unwrap_or_else(|| {
                auto_detect_compute_entry_point(&path).unwrap_or_else(|e| {
                    eprintln!("{}", e);
                    std::process::exit(1);
                })
            });

            pollster::block_on(modes::compute::run_compute_shader(
                path,
                entry_name,
                (workgroups_x, workgroups_y, workgroups_z),
                storage_buffers,
                uniforms,
                push_constants,
                verbose,
            ))?;
        }
        Command::Pipeline {
            path,
            pipeline,
            inputs,
            outputs,
            push_constants,
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
            let output_map = parse_pairs(&outputs)?;

            pollster::block_on(modes::pipeline::run_pipeline(
                path,
                pipeline,
                input_map,
                output_map,
                &push_constants,
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
        Command::Simulate {
            path,
            compute,
            vertex,
            fragment,
            grid,
            pattern,
            seed,
            max_frames,
            verbose,
            no_validate,
            present_mode,
            size,
            vertex_count,
        } => {
            let parsed_pattern = modes::simulate::parse_pattern(&pattern);
            modes::simulate::run_simulate(
                path,
                compute,
                vertex,
                fragment,
                grid,
                parsed_pattern,
                seed,
                max_frames,
                verbose,
                !no_validate,
                present_mode.into(),
                size,
                vertex_count,
            )?;
        }
    }

    Ok(())
}
