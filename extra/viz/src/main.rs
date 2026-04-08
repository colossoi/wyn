// src/main.rs
//#![windows_subsystem = "windows"]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages, Color,
    ColorTargetState, CommandEncoderDescriptor, DeviceDescriptor, FragmentState, Instance,
    InstanceDescriptor, InstanceFlags, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor,
    PowerPreference, PresentMode, PrimitiveState, RenderPipeline, RequestAdapterOptions,
    ShaderModuleDescriptorPassthrough, ShaderStages, StoreOp, SurfaceConfiguration, TextureUsages, Trace,
    VertexState,
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
        /// Storage buffer: binding:size:type[:input.json]
        /// Examples: "1:64:i32" (zeros), "1:64:i32:data.json" (from file)
        /// Type: i32, u32, f32. Repeat for multiple buffers.
        #[arg(long = "storage", value_name = "SPEC")]
        storage_buffers: Vec<String>,
        /// Push constant: name:type=value
        /// Examples: "n:i32=64", "header_base:u32x19=0,0,0,..."
        /// Type: i32, u32, f32, i32xN, u32xN, f32xN
        #[arg(long = "push-constant", value_name = "SPEC")]
        push_constants: Vec<String>,
        /// Print verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    /// Run a pipeline described by a JSON pipeline descriptor
    #[command(name = "run")]
    Run {
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
        /// Push constant: name:type=value
        /// Examples: "n:i32=64", "header_base:u32x19=0,0,0,..."
        /// Type: i32, u32, f32, i32xN, u32xN, f32xN
        #[arg(long = "push-constant", value_name = "SPEC")]
        push_constants: Vec<String>,
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
    /// Run the Bitcoin miner shader and report hash hits
    #[command(name = "miner")]
    Miner {
        /// Path to the linked miner SPIR-V module
        #[arg(default_value = "testfiles/miner.spv")]
        path: PathBuf,
        /// Block header base (19 hex u32 words, comma-separated)
        /// Example: "6a09e667,bb67ae85,...,5be0cd19"
        /// If omitted, uses a test header of all zeros.
        #[arg(long)]
        header: Option<String>,
        /// Number of nonces to try
        #[arg(long, short, default_value = "1024")]
        nonces: u32,
        /// Starting nonce offset
        #[arg(long, default_value = "0")]
        nonce_offset: u32,
        /// Difficulty: number of leading zero bytes required in the hash
        #[arg(long, short, default_value = "1")]
        difficulty: u32,
        /// Number of workgroups (each has 64 threads)
        #[arg(long)]
        workgroups: Option<u32>,
        /// Print verbose output
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

// --- Test pattern shaders (embedded WGSL) ------------------------------------

const TEST_PATTERN_SHADER: &str = r#"
// Resolution uniform (16-byte aligned)
struct Globals {
    resolution: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> globals: Globals;

// Vertex shader - fullscreen big triangle
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    return vec4<f32>(pos[vertex_index], 0.0, 1.0);
}

// Fragment shader - colored test pattern
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let res = globals.resolution;
    let fragCoord = pos.xy;
    let uv = fragCoord / res;

    // Checkerboard using integer math (fragCoord >= 0, so truncation == floor)
    let grid_size = 64.0;
    let gx = i32(fragCoord.x / grid_size);
    let gy = i32(fragCoord.y / grid_size);
    let checker = f32((gx + gy) & 1);

    // Color gradient + diagonal stripes
    let r = uv.x;
    let g = uv.y;
    let b = checker * 0.5 + 0.5;
    let stripe = sin((uv.x + uv.y) * 20.0) * 0.5 + 0.5;

    return vec4<f32>(r * stripe, g, b * (1.0 - stripe * 0.3), 1.0);
}
"#;

// --- Pipeline spec passed to the app -----------------------------------------

enum PipelineSpec {
    VertexFragment {
        path: PathBuf,
        vertex: String,
        fragment: String,
        shadertoy: bool,
        max_frames: Option<u32>,
        verbose: bool,
        validate: bool,
        present_mode: PresentMode,
        difficulty: i32,
        size: Option<(u32, u32)>,
    },
    TestPattern {
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
    let spirv_passthrough_supported = adapter_features.contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH);

    if verbose {
        println!(
            "SPIRV_SHADER_PASSTHROUGH supported: {}",
            spirv_passthrough_supported
        );
    }

    let mut required_features = wgpu::Features::empty();
    if spirv_passthrough_supported {
        required_features |= wgpu::Features::SPIRV_SHADER_PASSTHROUGH;
    }
    if adapter_features.contains(wgpu::Features::PUSH_CONSTANTS) {
        required_features |= wgpu::Features::PUSH_CONSTANTS;
    }

    let mut limits = wgpu::Limits::default();
    limits.max_push_constant_size = adapter.limits().max_push_constant_size;

    adapter
        .request_device(&DeviceDescriptor {
            label: None,
            required_features,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: Trace::Off,
        })
        .await
        .context("failed to create logical device")
}

/// Storage buffer specification parsed from CLI
#[derive(Debug, Clone)]
struct StorageBufferSpec {
    binding: u32,
    size_elements: u32,
    element_type: StorageElementType,
    /// Optional path to JSON file with initial data
    input_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy)]
enum StorageElementType {
    I32,
    U32,
    F32,
}

impl StorageBufferSpec {
    /// Parse from "binding:size:type[:input.json]" format
    /// Examples:
    ///   "1:64:i32"           - 64 i32s initialized to zero
    ///   "1:64:i32:data.json" - 64 i32s loaded from data.json
    fn parse(spec: &str) -> Result<Self> {
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() < 3 || parts.len() > 4 {
            return Err(anyhow!(
                "Invalid storage buffer spec '{}'. Expected format: binding:size:type[:input.json]",
                spec
            ));
        }

        let binding =
            parts[0].parse::<u32>().map_err(|_| anyhow!("Invalid binding number: {}", parts[0]))?;
        let size_elements = parts[1].parse::<u32>().map_err(|_| anyhow!("Invalid size: {}", parts[1]))?;
        let element_type = match parts[2].to_lowercase().as_str() {
            "i32" => StorageElementType::I32,
            "u32" => StorageElementType::U32,
            "f32" => StorageElementType::F32,
            other => {
                return Err(anyhow!(
                    "Unknown element type: {}. Expected i32, u32, or f32",
                    other
                ));
            }
        };

        let input_file = if parts.len() == 4 { Some(PathBuf::from(parts[3])) } else { None };

        Ok(Self {
            binding,
            size_elements,
            element_type,
            input_file,
        })
    }

    fn byte_size(&self) -> u64 {
        self.size_elements as u64 * 4
    }

    /// Load initial data from JSON file or return zeros
    fn load_initial_data(&self) -> Result<Vec<u8>> {
        match &self.input_file {
            Some(path) => {
                let content = fs::read_to_string(path)
                    .with_context(|| format!("Failed to read input file: {}", path.display()))?;
                let json: serde_json::Value = serde_json::from_str(&content)
                    .with_context(|| format!("Failed to parse JSON from: {}", path.display()))?;

                let array = json.as_array().ok_or_else(|| anyhow!("JSON input must be an array"))?;

                if array.len() != self.size_elements as usize {
                    return Err(anyhow!(
                        "JSON array has {} elements but buffer expects {}",
                        array.len(),
                        self.size_elements
                    ));
                }

                let mut bytes = Vec::with_capacity(self.byte_size() as usize);
                match self.element_type {
                    StorageElementType::I32 => {
                        for (i, val) in array.iter().enumerate() {
                            let n =
                                val.as_i64().ok_or_else(|| anyhow!("Element {} is not an integer", i))?
                                    as i32;
                            bytes.extend_from_slice(&n.to_le_bytes());
                        }
                    }
                    StorageElementType::U32 => {
                        for (i, val) in array.iter().enumerate() {
                            let n = val
                                .as_u64()
                                .ok_or_else(|| anyhow!("Element {} is not a positive integer", i))?
                                as u32;
                            bytes.extend_from_slice(&n.to_le_bytes());
                        }
                    }
                    StorageElementType::F32 => {
                        for (i, val) in array.iter().enumerate() {
                            let n = val.as_f64().ok_or_else(|| anyhow!("Element {} is not a number", i))?
                                as f32;
                            bytes.extend_from_slice(&n.to_le_bytes());
                        }
                    }
                }
                Ok(bytes)
            }
            None => Ok(vec![0u8; self.byte_size() as usize]),
        }
    }
}

#[derive(Debug)]
struct PushConstantSpec {
    name: String,
    offset: u32,
    data: Vec<u8>,
}

impl PushConstantSpec {
    /// Parse from "name:type=value" format
    /// Examples: "n:i32=64", "header_base:u32x19=0,0,0,..."
    fn parse(spec: &str) -> Result<Self> {
        let (name_type, value) =
            spec.split_once('=').ok_or_else(|| anyhow!("Push constant spec must contain '=': {}", spec))?;
        let (name, ty) = name_type
            .split_once(':')
            .ok_or_else(|| anyhow!("Push constant spec must have format name:type=value: {}", spec))?;

        let data = parse_push_constant_value(ty, value)?;

        Ok(Self {
            name: name.to_string(),
            offset: 0, // filled in later
            data,
        })
    }

    fn byte_size(&self) -> u32 {
        self.data.len() as u32
    }
}

fn parse_push_constant_value(ty: &str, value: &str) -> Result<Vec<u8>> {
    // Check for array types like u32x19, i32x4, f32x3
    if let Some(rest) = ty.strip_prefix("u32x") {
        let count: usize = rest.parse().map_err(|_| anyhow!("Invalid array size: {}", rest))?;
        let values: Vec<u32> = value
            .split(',')
            .map(|v| {
                let v = v.trim();
                if let Some(hex) = v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
                    u32::from_str_radix(hex, 16).map_err(|e| anyhow!("Invalid hex u32: {}: {}", v, e))
                } else {
                    v.parse::<u32>().map_err(|e| anyhow!("Invalid u32: {}: {}", v, e))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        if values.len() != count {
            return Err(anyhow!("Expected {} values for {}, got {}", count, ty, values.len()));
        }
        Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
    } else if let Some(rest) = ty.strip_prefix("i32x") {
        let count: usize = rest.parse().map_err(|_| anyhow!("Invalid array size: {}", rest))?;
        let values: Vec<i32> = value
            .split(',')
            .map(|v| v.trim().parse::<i32>().map_err(|e| anyhow!("Invalid i32: {}: {}", v, e)))
            .collect::<Result<Vec<_>>>()?;
        if values.len() != count {
            return Err(anyhow!("Expected {} values for {}, got {}", count, ty, values.len()));
        }
        Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
    } else if let Some(rest) = ty.strip_prefix("f32x") {
        let count: usize = rest.parse().map_err(|_| anyhow!("Invalid array size: {}", rest))?;
        let values: Vec<f32> = value
            .split(',')
            .map(|v| v.trim().parse::<f32>().map_err(|e| anyhow!("Invalid f32: {}: {}", v, e)))
            .collect::<Result<Vec<_>>>()?;
        if values.len() != count {
            return Err(anyhow!("Expected {} values for {}, got {}", count, ty, values.len()));
        }
        Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
    } else {
        match ty {
            "i32" => {
                let v: i32 = value.parse().map_err(|e| anyhow!("Invalid i32 '{}': {}", value, e))?;
                Ok(v.to_le_bytes().to_vec())
            }
            "u32" => {
                let v = if let Some(hex) = value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
                    u32::from_str_radix(hex, 16).map_err(|e| anyhow!("Invalid hex u32: {}: {}", value, e))?
                } else {
                    value.parse::<u32>().map_err(|e| anyhow!("Invalid u32: {}: {}", value, e))?
                };
                Ok(v.to_le_bytes().to_vec())
            }
            "f32" => {
                let v: f32 = value.parse().map_err(|e| anyhow!("Invalid f32 '{}': {}", value, e))?;
                Ok(v.to_le_bytes().to_vec())
            }
            other => Err(anyhow!(
                "Unknown push constant type: {}. Use i32, u32, f32, u32xN, i32xN, f32xN",
                other
            )),
        }
    }
}

/// Print storage buffer contents
fn print_storage_buffer(spec: &StorageBufferSpec, data: &[u8]) {
    println!(
        "\n=== Storage Buffer (binding {}, {} elements) ===",
        spec.binding, spec.size_elements
    );

    let u32_data: &[u32] = bytemuck::cast_slice(data);
    let elements_to_show = (spec.size_elements as usize).min(64); // Show at most 64 elements

    match spec.element_type {
        StorageElementType::I32 => {
            let i32_data: Vec<i32> = u32_data.iter().map(|&x| x as i32).collect();
            for (i, chunk) in i32_data[..elements_to_show].chunks(8).enumerate() {
                print!("  [{:3}]: ", i * 8);
                for val in chunk {
                    print!("{:8} ", val);
                }
                println!();
            }
        }
        StorageElementType::U32 => {
            for (i, chunk) in u32_data[..elements_to_show].chunks(8).enumerate() {
                print!("  [{:3}]: ", i * 8);
                for val in chunk {
                    print!("{:8} ", val);
                }
                println!();
            }
        }
        StorageElementType::F32 => {
            let f32_data: Vec<f32> = u32_data.iter().map(|&x| f32::from_bits(x)).collect();
            for (i, chunk) in f32_data[..elements_to_show].chunks(8).enumerate() {
                print!("  [{:3}]: ", i * 8);
                for val in chunk {
                    print!("{:8.3} ", val);
                }
                println!();
            }
        }
    }

    if spec.size_elements as usize > elements_to_show {
        println!(
            "  ... ({} more elements)",
            spec.size_elements as usize - elements_to_show
        );
    }
    println!();
}

// --- Pipeline descriptor types (mirrors wyn-core/src/pipeline_descriptor.rs) --

mod pipeline_desc {
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    pub struct PipelineDescriptor {
        pub pipelines: Vec<Pipeline>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    pub enum Pipeline {
        Compute(ComputePipeline),
        MultiCompute(MultiComputePipeline),
        Graphics(GraphicsPipeline),
    }

    #[derive(Debug, Deserialize)]
    pub struct ComputePipeline {
        pub entry_point: String,
        pub workgroup_size: (u32, u32, u32),
        pub dispatch_size: DispatchSize,
        pub bindings: Vec<Binding>,
    }

    #[derive(Debug, Deserialize)]
    pub struct MultiComputePipeline {
        pub bindings: Vec<Binding>,
        pub stages: Vec<ComputeStage>,
    }

    #[derive(Debug, Deserialize)]
    pub struct ComputeStage {
        pub entry_point: String,
        pub workgroup_size: (u32, u32, u32),
        pub dispatch_size: DispatchSize,
        pub reads: Vec<usize>,
        pub writes: Vec<usize>,
    }

    #[derive(Debug, Deserialize)]
    pub struct GraphicsPipeline {
        pub stages: Vec<GraphicsStage>,
        pub bindings: Vec<Binding>,
        pub vertex_inputs: Vec<VertexAttribute>,
        pub fragment_outputs: Vec<FragmentOutput>,
    }

    #[derive(Debug, Deserialize)]
    pub struct GraphicsStage {
        pub entry_point: String,
        pub stage: ShaderStage,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ShaderStage {
        Vertex,
        Fragment,
    }

    #[derive(Debug, Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    pub enum DispatchSize {
        Fixed {
            x: u32,
            y: u32,
            z: u32,
        },
        DerivedFromInputLength {
            workgroup_size: u32,
        },
    }

    #[derive(Debug, Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum Binding {
        StorageBuffer {
            set: u32,
            binding: u32,
            access: Access,
            usage: BufferUsage,
            name: String,
        },
        Uniform {
            set: u32,
            binding: u32,
            name: String,
        },
        PushConstant {
            offset: u32,
            size: u32,
            name: String,
        },
    }

    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(rename_all = "snake_case")]
    pub enum Access {
        ReadOnly,
        WriteOnly,
        ReadWrite,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(rename_all = "snake_case")]
    pub enum BufferUsage {
        Input,
        Output,
        Intermediate,
    }

    #[derive(Debug, Deserialize)]
    pub struct VertexAttribute {
        pub location: u32,
        pub name: String,
    }

    #[derive(Debug, Deserialize)]
    pub struct FragmentOutput {
        pub location: u32,
        pub name: String,
    }

    impl Binding {
        pub fn name(&self) -> &str {
            match self {
                Binding::StorageBuffer { name, .. } => name,
                Binding::Uniform { name, .. } => name,
                Binding::PushConstant { name, .. } => name,
            }
        }

        pub fn wgpu_binding(&self) -> u32 {
            match self {
                Binding::StorageBuffer { binding, .. } => *binding,
                Binding::Uniform { binding, .. } => *binding,
                Binding::PushConstant { .. } => panic!("PushConstant has no binding number"),
            }
        }

        pub fn is_storage(&self) -> bool {
            matches!(self, Binding::StorageBuffer { .. })
        }

        pub fn is_input(&self) -> bool {
            matches!(
                self,
                Binding::StorageBuffer {
                    usage: BufferUsage::Input,
                    ..
                }
            )
        }

        pub fn is_output(&self) -> bool {
            matches!(
                self,
                Binding::StorageBuffer {
                    usage: BufferUsage::Output,
                    ..
                }
            )
        }

        pub fn is_read_only(&self) -> bool {
            matches!(
                self,
                Binding::StorageBuffer {
                    access: Access::ReadOnly,
                    ..
                }
            )
        }
    }
}

// --- Pipeline-descriptor-driven execution ------------------------------------

/// Load a JSON array of f32 values from a file.
fn load_f32_json(path: &Path) -> Result<Vec<f32>> {
    let content =
        fs::read_to_string(path).with_context(|| format!("Failed to read: {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse JSON: {}", path.display()))?;
    let array = json.as_array().ok_or_else(|| anyhow!("JSON input must be an array"))?;
    array
        .iter()
        .enumerate()
        .map(|(i, v)| v.as_f64().map(|f| f as f32).ok_or_else(|| anyhow!("Element {} is not a number", i)))
        .collect()
}

/// Write f32 data as a JSON array to a file.
fn write_f32_json(path: &Path, data: &[f32]) -> Result<()> {
    let json =
        serde_json::to_string_pretty(&data.iter().map(|&f| serde_json::json!(f)).collect::<Vec<_>>())?;
    fs::write(path, json).with_context(|| format!("Failed to write: {}", path.display()))
}

/// Print f32 data to stdout.
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

/// Run a pipeline from a descriptor + SPIR-V module.
async fn run_pipeline(
    spv_path: PathBuf,
    pipeline_path: PathBuf,
    inputs: HashMap<String, PathBuf>,
    outputs: HashMap<String, PathBuf>,
    push_constants: &[String],
    verbose: bool,
) -> Result<()> {
    let desc_json = fs::read_to_string(&pipeline_path)
        .with_context(|| format!("Failed to read pipeline descriptor: {}", pipeline_path.display()))?;
    let desc: pipeline_desc::PipelineDescriptor =
        serde_json::from_str(&desc_json).with_context(|| "Failed to parse pipeline descriptor JSON")?;

    if desc.pipelines.is_empty() {
        return Err(anyhow!("Pipeline descriptor has no pipelines"));
    }

    let (device, queue) = create_headless_device(verbose).await?;
    let module = load_spirv_module(&device, &spv_path)?;

    for (pi, pipeline) in desc.pipelines.iter().enumerate() {
        match pipeline {
            pipeline_desc::Pipeline::Compute(cp) => {
                run_single_compute(&device, &queue, &module, cp, &inputs, &outputs, push_constants, verbose)
                    .with_context(|| format!("Pipeline {} (compute) failed", pi))?;
            }
            pipeline_desc::Pipeline::MultiCompute(mp) => {
                run_multi_compute(&device, &queue, &module, mp, &inputs, &outputs, push_constants, verbose)
                    .with_context(|| format!("Pipeline {} (multi_compute) failed", pi))?;
            }
            pipeline_desc::Pipeline::Graphics(_) => {
                eprintln!(
                    "Pipeline {} is a graphics pipeline (not yet supported by `run`)",
                    pi
                );
            }
        }
    }

    Ok(())
}

/// Create wgpu buffers for a set of bindings. Returns a map from binding number
/// to (gpu_buffer, byte_size).
fn create_binding_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bindings: &[pipeline_desc::Binding],
    inputs: &HashMap<String, PathBuf>,
    verbose: bool,
) -> Result<HashMap<u32, (wgpu::Buffer, u64)>> {
    let mut buffers = HashMap::new();

    for b in bindings {
        if let pipeline_desc::Binding::StorageBuffer {
            binding, name, usage, ..
        } = b
        {
            let (data_bytes, element_count) = if let Some(path) = inputs.get(name.as_str()) {
                let data = load_f32_json(path)?;
                let count = data.len();
                let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                if verbose {
                    println!("Loaded {} elements for '{}' from {}", count, name, path.display());
                }
                (bytes, count)
            } else if *usage == pipeline_desc::BufferUsage::Input {
                return Err(anyhow!(
                    "No input file provided for '{}'. Use --input {}:<file.json>",
                    name,
                    name
                ));
            } else {
                // Intermediate/output buffers: allocate a reasonable size.
                // For intermediate buffers in multi-compute, the runtime needs to
                // allocate based on the pipeline's needs. Use 1024 elements as default.
                let count = 1024usize;
                if verbose {
                    println!("Allocated {} zero elements for '{}'", count, name);
                }
                (vec![0u8; count * 4], count)
            };

            let byte_size = data_bytes.len() as u64;
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(name),
                size: byte_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, &data_bytes);

            buffers.insert(*binding, (buffer, byte_size));
            let _ = element_count; // used in verbose output above
        }
    }

    Ok(buffers)
}

/// Build a bind group layout + bind group from binding descriptors.
fn build_bind_group(
    device: &wgpu::Device,
    bindings: &[pipeline_desc::Binding],
    buffers: &HashMap<u32, (wgpu::Buffer, u64)>,
) -> Result<(wgpu::BindGroupLayout, BindGroup)> {
    let mut layout_entries = Vec::new();
    let mut group_entries = Vec::new();

    for b in bindings {
        if let pipeline_desc::Binding::StorageBuffer { binding, access, .. } = b {
            let read_only = *access == pipeline_desc::Access::ReadOnly;
            layout_entries.push(BindGroupLayoutEntry {
                binding: *binding,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });

            let (buf, _) =
                buffers.get(binding).ok_or_else(|| anyhow!("No buffer for binding {}", binding))?;
            group_entries.push(BindGroupEntry {
                binding: *binding,
                resource: BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buf,
                    offset: 0,
                    size: None,
                }),
            });
        }
    }

    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("pipeline_bind_group_layout"),
        entries: &layout_entries,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("pipeline_bind_group"),
        layout: &layout,
        entries: &group_entries,
    });

    Ok((layout, bind_group))
}

/// Compute dispatch dimensions from a DispatchSize spec.
fn resolve_dispatch_size(
    dispatch: &pipeline_desc::DispatchSize,
    buffers: &HashMap<u32, (wgpu::Buffer, u64)>,
    bindings: &[pipeline_desc::Binding],
    reads: Option<&[usize]>,
) -> (u32, u32, u32) {
    match dispatch {
        pipeline_desc::DispatchSize::Fixed { x, y, z } => (*x, *y, *z),
        pipeline_desc::DispatchSize::DerivedFromInputLength { workgroup_size } => {
            // Find the first input binding to derive length from
            let input_binding = reads
                .and_then(|r| r.first())
                .and_then(|&idx| bindings.get(idx))
                .or_else(|| bindings.iter().find(|b| b.is_input()));

            let elements = input_binding
                .map(|b| {
                    let binding_num = b.wgpu_binding();
                    buffers.get(&binding_num).map(|(_, size)| (*size / 4) as u32).unwrap_or(0)
                })
                .unwrap_or(0);

            let wg = *workgroup_size;
            let groups = (elements + wg - 1) / wg;
            (groups, 1, 1)
        }
    }
}

/// Read back a GPU buffer to CPU as f32 data.
fn readback_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    byte_size: u64,
) -> Result<Vec<f32>> {
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("readback_staging"),
        size: byte_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_size);
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::Wait);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::Wait);
    rx.recv().unwrap()?;

    let data = slice.get_mapped_range();
    let u32_data: &[u32] = bytemuck::cast_slice(&data);
    let f32_data: Vec<f32> = u32_data.iter().map(|&x| f32::from_bits(x)).collect();
    drop(data);
    staging.unmap();

    Ok(f32_data)
}

/// Build push constant bytes from descriptor PushConstant bindings + CLI --push-constant args.
/// The descriptor provides the layout (offsets and sizes); the CLI provides values by name.
fn build_push_constant_bytes(
    bindings: &[pipeline_desc::Binding],
    push_constants: &[String],
    verbose: bool,
) -> Result<Vec<u8>> {
    // Collect PushConstant bindings from the descriptor
    let pc_bindings: Vec<_> = bindings
        .iter()
        .filter_map(|b| {
            if let pipeline_desc::Binding::PushConstant { offset, size, name } = b {
                Some((name.as_str(), *offset, *size))
            } else {
                None
            }
        })
        .collect();

    if pc_bindings.is_empty() && push_constants.is_empty() {
        return Ok(vec![]);
    }

    // If we have CLI push constants but no descriptor bindings, use sequential layout
    if pc_bindings.is_empty() && !push_constants.is_empty() {
        let mut pc_specs: Vec<PushConstantSpec> = push_constants
            .iter()
            .map(|s| PushConstantSpec::parse(s))
            .collect::<Result<Vec<_>>>()?;

        let mut offset = 0u32;
        for spec in &mut pc_specs {
            spec.offset = offset;
            offset += spec.byte_size();
        }
        let total = offset as usize;
        let mut bytes = vec![0u8; total];
        for spec in &pc_specs {
            let start = spec.offset as usize;
            let end = start + spec.data.len();
            bytes[start..end].copy_from_slice(&spec.data);
        }
        if verbose {
            println!("Push constants ({} bytes, sequential layout):", total);
            for spec in &pc_specs {
                println!("  {} @ offset {}: {} bytes", spec.name, spec.offset, spec.byte_size());
            }
        }
        return Ok(bytes);
    }

    // Parse CLI push constants into a map by name
    let cli_specs: HashMap<String, PushConstantSpec> = push_constants
        .iter()
        .map(|s| {
            let spec = PushConstantSpec::parse(s)?;
            Ok((spec.name.clone(), spec))
        })
        .collect::<Result<HashMap<_, _>>>()?;

    // Compute total size from descriptor
    let total_size = pc_bindings
        .iter()
        .map(|(_, offset, size)| offset + size)
        .max()
        .unwrap_or(0) as usize;

    let mut bytes = vec![0u8; total_size];

    for (name, offset, size) in &pc_bindings {
        if let Some(spec) = cli_specs.get(*name) {
            if spec.data.len() != *size as usize {
                return Err(anyhow!(
                    "Push constant '{}' expects {} bytes but got {} bytes from CLI",
                    name,
                    size,
                    spec.data.len()
                ));
            }
            let start = *offset as usize;
            let end = start + spec.data.len();
            bytes[start..end].copy_from_slice(&spec.data);
            if verbose {
                println!("Push constant '{}' @ offset {}: {} bytes", name, offset, size);
            }
        }
        // If not provided via CLI, leave as zeros
    }

    Ok(bytes)
}

/// Run a single-dispatch compute pipeline.
fn run_single_compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    module: &wgpu::ShaderModule,
    cp: &pipeline_desc::ComputePipeline,
    inputs: &HashMap<String, PathBuf>,
    outputs: &HashMap<String, PathBuf>,
    push_constants: &[String],
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("Running compute pipeline: {}", cp.entry_point);
    }

    let buffers = create_binding_buffers(device, queue, &cp.bindings, inputs, verbose)?;
    let (layout, bind_group) = build_bind_group(device, &cp.bindings, &buffers)?;

    // Build push constant data from CLI args matched against descriptor bindings
    let pc_bytes = build_push_constant_bytes(&cp.bindings, push_constants, verbose)?;
    let total_pc_size = pc_bytes.len() as u32;

    let pc_ranges: Vec<wgpu::PushConstantRange> = if total_pc_size > 0 {
        vec![wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..total_pc_size,
        }]
    } else {
        vec![]
    };

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("compute_layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &pc_ranges,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute_pipeline"),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(&cp.entry_point),
        compilation_options: Default::default(),
        cache: None,
    });

    let dispatch = resolve_dispatch_size(&cp.dispatch_size, &buffers, &cp.bindings, None);
    if verbose {
        println!("Dispatch: {} x {} x {}", dispatch.0, dispatch.1, dispatch.2);
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
        cpass.set_bind_group(0, &bind_group, &[]);
        if !pc_bytes.is_empty() {
            cpass.set_push_constants(0, &pc_bytes);
        }
        cpass.dispatch_workgroups(dispatch.0, dispatch.1, dispatch.2);
    }
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::Wait);

    // Read back and output results
    output_results(device, queue, &cp.bindings, &buffers, outputs)?;

    Ok(())
}

/// Run a multi-dispatch compute pipeline (e.g. reduce with phase1 + phase2).
fn run_multi_compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    module: &wgpu::ShaderModule,
    mp: &pipeline_desc::MultiComputePipeline,
    inputs: &HashMap<String, PathBuf>,
    outputs: &HashMap<String, PathBuf>,
    push_constants: &[String],
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("Running multi-compute pipeline ({} stages)", mp.stages.len());
        for (i, stage) in mp.stages.iter().enumerate() {
            println!(
                "  Stage {}: {} (reads {:?}, writes {:?})",
                i, stage.entry_point, stage.reads, stage.writes
            );
        }
    }

    let buffers = create_binding_buffers(device, queue, &mp.bindings, inputs, verbose)?;
    let (layout, bind_group) = build_bind_group(device, &mp.bindings, &buffers)?;

    // Build push constant data from CLI args matched against descriptor bindings
    let pc_bytes = build_push_constant_bytes(&mp.bindings, push_constants, verbose)?;
    let total_pc_size = pc_bytes.len() as u32;

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

        let dispatch =
            resolve_dispatch_size(&stage.dispatch_size, &buffers, &mp.bindings, Some(&stage.reads));
        if verbose {
            println!(
                "Stage {} ({}): dispatch {} x {} x {}",
                si, stage.entry_point, dispatch.0, dispatch.1, dispatch.2
            );
        }

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some(&format!("stage_{}_encoder", si)),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("stage_{}", si)),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            if !pc_bytes.is_empty() {
                cpass.set_push_constants(0, &pc_bytes);
            }
            cpass.dispatch_workgroups(dispatch.0, dispatch.1, dispatch.2);
        }
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::Wait);
    }

    // Read back and output results
    output_results(device, queue, &mp.bindings, &buffers, outputs)?;

    Ok(())
}

/// Read back output/intermediate buffers and write/print results.
fn output_results(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bindings: &[pipeline_desc::Binding],
    buffers: &HashMap<u32, (wgpu::Buffer, u64)>,
    outputs: &HashMap<String, PathBuf>,
) -> Result<()> {
    for b in bindings {
        if let pipeline_desc::Binding::StorageBuffer {
            binding, name, usage, ..
        } = b
        {
            // Only read back output and intermediate buffers (skip inputs unless
            // explicitly requested via --output)
            let should_output =
                *usage != pipeline_desc::BufferUsage::Input || outputs.contains_key(name.as_str());

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

/// Run the Bitcoin miner shader and check for hash hits
async fn run_miner(
    path: PathBuf,
    header_arg: Option<String>,
    nonces: u32,
    nonce_offset: u32,
    difficulty: u32,
    workgroups_override: Option<u32>,
    verbose: bool,
) -> Result<()> {
    // Parse header: 19 hex u32 words, or default to zeros
    let header_base: Vec<u32> = if let Some(ref h) = header_arg {
        let words: Vec<u32> = h
            .split(',')
            .map(|w| {
                let w = w.trim();
                if let Some(hex) = w.strip_prefix("0x").or_else(|| w.strip_prefix("0X")) {
                    u32::from_str_radix(hex, 16).map_err(|e| anyhow!("Invalid hex u32 '{}': {}", w, e))
                } else {
                    u32::from_str_radix(w, 16).map_err(|e| anyhow!("Invalid hex u32 '{}': {}", w, e))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        if words.len() != 19 {
            return Err(anyhow!("Header must be exactly 19 u32 words, got {}", words.len()));
        }
        words
    } else {
        vec![0u32; 19]
    };

    let workgroup_size = 64u32;
    let num_workgroups = workgroups_override.unwrap_or_else(|| (nonces + workgroup_size - 1) / workgroup_size);

    // Output buffer: n hashes, each [8]u32 = 32 bytes
    let output_elements = nonces as u64 * 8;
    let output_byte_size = output_elements * 4;

    if verbose {
        println!("Miner configuration:");
        println!("  Header: {:08x?}", header_base);
        println!("  Nonces: {} (offset {})", nonces, nonce_offset);
        println!("  Difficulty: {} leading zero bytes", difficulty);
        println!("  Workgroups: {} x 1 x 1 (workgroup size {})", num_workgroups, workgroup_size);
        println!("  Output buffer: {} bytes ({} hashes)", output_byte_size, nonces);
    }

    let (device, queue) = create_headless_device(verbose).await?;

    // Storage buffer for output hashes
    let output_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("miner_output"),
        size: output_byte_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("miner_staging"),
        size: output_byte_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Bind group
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("miner_bind_group_layout"),
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

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("miner_bind_group"),
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &output_buffer,
                offset: 0,
                size: None,
            }),
        }],
    });

    // Push constants: header_base (76 bytes) + n (4 bytes) + nonce_offset (4 bytes) = 84 bytes
    let mut pc_bytes = Vec::with_capacity(84);
    for word in &header_base {
        pc_bytes.extend_from_slice(&word.to_le_bytes());
    }
    pc_bytes.extend_from_slice(&nonces.to_le_bytes()); // n as i32, but same bits
    pc_bytes.extend_from_slice(&nonce_offset.to_le_bytes());

    let module = load_spirv_module(&device, &path)?;

    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("miner_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..84,
        }],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("miner_pipeline"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("mine"),
        compilation_options: Default::default(),
        cache: None,
    });

    let start_time = std::time::Instant::now();

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("miner_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("miner_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_push_constants(0, &pc_bytes);
        cpass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_byte_size);
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::Wait);

    // Read back results
    let buffer_slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::Wait);
    rx.recv().unwrap()?;

    let elapsed = start_time.elapsed();
    let data = buffer_slice.get_mapped_range();
    let hash_words: &[u32] = bytemuck::cast_slice(&data);

    // Check each hash for difficulty hits.
    // SHA256 state words h0..h7: h0 is the most significant 32 bits of the digest.
    // Bitcoin compares hashes as big-endian 256-bit numbers, so leading zeros
    // in the display hash correspond to leading zero bytes starting from h0.
    let mut hits = Vec::new();
    for i in 0..nonces as usize {
        let hash = &hash_words[i * 8..(i + 1) * 8];
        let nonce = nonce_offset + i as u32;
        if verbose {
            print!("  nonce {:>10} -> ", nonce);
            for word in hash {
                print!("{:08x}", word);
            }
            let lz = leading_zero_bytes(hash);
            println!("  ({} leading zero bytes)", lz);
        }
        if meets_difficulty(hash, difficulty) {
            hits.push((nonce, hash.to_vec()));
        }
    }

    let hash_rate = nonces as f64 / elapsed.as_secs_f64();

    println!("Mined {} nonces in {:.2?} ({:.0} H/s)", nonces, elapsed, hash_rate);

    if hits.is_empty() {
        println!("No hits found (difficulty: {} leading zero bytes)", difficulty);
    } else {
        println!("{} hit(s) found:", hits.len());
        for (nonce, hash) in &hits {
            print!("  nonce {:>10} -> ", nonce);
            for word in hash {
                print!("{:08x}", word);
            }
            println!();
        }
    }

    drop(data);
    staging.unmap();

    Ok(())
}

/// Count leading zero bytes in a hash (8 x u32, big-endian word order: h0 is MSB).
fn leading_zero_bytes(hash: &[u32]) -> u32 {
    let mut zero_bytes = 0u32;
    for &word in hash {
        let lz = word.leading_zeros() / 8; // full zero bytes in this word
        zero_bytes += lz;
        if lz < 4 {
            break; // this word has a non-zero byte, stop counting
        }
    }
    zero_bytes
}

/// Check if a hash has at least `n` leading zero bytes.
fn meets_difficulty(hash: &[u32], n: u32) -> bool {
    leading_zero_bytes(hash) >= n
}

/// Run a compute shader headlessly (no window)
async fn run_compute_shader(
    path: PathBuf,
    entry: String,
    workgroups: (u32, u32, u32),
    storage_specs: Vec<StorageBufferSpec>,
    push_constants: &[String],
    verbose: bool,
) -> Result<()> {
    let (device, queue) = create_headless_device(verbose).await?;

    // Create storage buffers for each spec
    let mut storage_buffers: Vec<(StorageBufferSpec, wgpu::Buffer, wgpu::Buffer)> = Vec::new();
    for spec in &storage_specs {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("storage_buffer_{}", spec.binding)),
            size: spec.byte_size(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize from JSON file or zeros
        let init_data = spec.load_initial_data()?;
        if verbose && spec.input_file.is_some() {
            println!(
                "Loaded {} bytes for binding {} from {:?}",
                init_data.len(),
                spec.binding,
                spec.input_file
            );
        }
        queue.write_buffer(&buffer, 0, &init_data);

        let staging = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("staging_buffer_{}", spec.binding)),
            size: spec.byte_size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        storage_buffers.push((spec.clone(), buffer, staging));
    }

    // Build bind group layout entries for storage buffers
    let layout_entries: Vec<_> = storage_buffers
        .iter()
        .map(|(spec, _, _)| BindGroupLayoutEntry {
            binding: spec.binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("compute_bind_group_layout"),
        entries: &layout_entries,
    });

    // Build bind group entries
    let bind_group_entries: Vec<_> = storage_buffers
        .iter()
        .map(|(spec, buffer, _)| BindGroupEntry {
            binding: spec.binding,
            resource: BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: None,
            }),
        })
        .collect();

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("compute_bind_group"),
        layout: &bind_group_layout,
        entries: &bind_group_entries,
    });

    let module = load_spirv_module(&device, &path)?;

    // Parse and layout push constants
    let mut pc_specs: Vec<PushConstantSpec> = push_constants
        .iter()
        .map(|s| PushConstantSpec::parse(s))
        .collect::<Result<Vec<_>>>()?;

    let mut offset = 0u32;
    for spec in &mut pc_specs {
        spec.offset = offset;
        offset += spec.byte_size();
    }
    let total_pc_size = offset;

    let mut pc_bytes = vec![0u8; total_pc_size as usize];
    for spec in &pc_specs {
        let start = spec.offset as usize;
        let end = start + spec.data.len();
        pc_bytes[start..end].copy_from_slice(&spec.data);
    }

    if verbose && !pc_specs.is_empty() {
        println!("Push constants ({} bytes):", total_pc_size);
        for spec in &pc_specs {
            println!("  {} @ offset {}: {} bytes", spec.name, spec.offset, spec.byte_size());
        }
    }

    let pc_ranges: Vec<wgpu::PushConstantRange> = if total_pc_size > 0 {
        vec![wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..total_pc_size,
        }]
    } else {
        vec![]
    };

    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("compute_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &pc_ranges,
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
        println!(
            "Dispatching {} x {} x {} workgroups...",
            workgroups.0, workgroups.1, workgroups.2
        );
        println!(
            "Storage buffers: {:?}",
            storage_specs.iter().map(|s| s.binding).collect::<Vec<_>>()
        );
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
        cpass.set_bind_group(0, &bind_group, &[]);
        if !pc_bytes.is_empty() {
            cpass.set_push_constants(0, &pc_bytes);
        }
        cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    // Copy all buffers to staging
    for (spec, buffer, staging) in &storage_buffers {
        encoder.copy_buffer_to_buffer(buffer, 0, staging, 0, spec.byte_size());
    }

    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::Wait);

    // Read back storage buffers
    for (spec, _, staging) in &storage_buffers {
        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait);

        if rx.recv().unwrap().is_ok() {
            let data = buffer_slice.get_mapped_range();
            print_storage_buffer(spec, &data);
            drop(data);
            staging.unmap();
        }
    }

    Ok(())
}

// --- App state ---------------------------------------------------------------

// Uniform buffers - one per shader uniform
// iResolution: [2]f32 + [2]f32 padding for 16-byte alignment
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ResolutionUniform {
    resolution: [f32; 2],
    _pad: [f32; 2],
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
        let validate = match spec {
            PipelineSpec::VertexFragment { validate, .. } => *validate,
            PipelineSpec::TestPattern { .. } => true, // always validate for test pattern
        };

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
        let verbose = match spec {
            PipelineSpec::VertexFragment { verbose, .. } => *verbose,
            PipelineSpec::TestPattern { verbose, .. } => *verbose,
        };

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
        let format =
            caps.formats.get(0).copied().ok_or_else(|| anyhow!("surface reports no supported formats"))?;
        let size = window.inner_size();

        // Extract present mode from spec (default to Fifo for TestPattern)
        let present_mode = match spec {
            PipelineSpec::VertexFragment { present_mode, .. } => *present_mode,
            PipelineSpec::TestPattern { .. } => PresentMode::Fifo,
        };

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
        // Shadertoy canonical uniform ordering:
        //   binding 0: iResolution (vec3, we use vec2)
        //   binding 1: iTime (f32)
        //   binding 2: difficulty (i32)
        //   binding 5: iMouse (vec4)
        let (
            resolution_buffer,
            time_buffer,
            mouse_buffer,
            difficulty_buffer,
            uniform_bind_group,
            uniform_bind_group_layout,
        ) = if let PipelineSpec::VertexFragment {
            shadertoy: true,
            difficulty,
            ..
        } = spec
        {
            // Binding 0: iResolution ([2]f32)
            let resolution_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("resolution_buffer"),
                size: std::mem::size_of::<ResolutionUniform>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let initial_resolution = ResolutionUniform {
                resolution: [800.0, 600.0],
                _pad: [0.0, 0.0],
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

            // Binding 2: difficulty (i32)
            let difficulty_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("difficulty_buffer"),
                size: std::mem::size_of::<DifficultyUniform>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let initial_difficulty = DifficultyUniform {
                difficulty: *difficulty,
                _pad: [0, 0, 0],
            };
            queue.write_buffer(&difficulty_buffer, 0, bytemuck::cast_slice(&[initial_difficulty]));

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

            let uniform_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
                        binding: 2,
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
                        binding: 2,
                        resource: BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &difficulty_buffer,
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
                Some(difficulty_buffer),
                Some(uniform_bind_group),
                Some(uniform_bind_group_layout),
            )
        } else {
            (None, None, None, None, None, None)
        };

        // Make these mutable so TestPattern can set them
        let mut resolution_buffer = resolution_buffer;
        let mut uniform_bind_group = uniform_bind_group;

        // Extract max_frames and verbose from spec
        let (max_frames, verbose) = match spec {
            PipelineSpec::VertexFragment {
                max_frames, verbose, ..
            } => (*max_frames, *verbose),
            PipelineSpec::TestPattern {
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
                let mut bind_group_layouts_vec = vec![];
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
            PipelineSpec::TestPattern { .. } => {
                eprintln!("[viz] Loading built-in test pattern shader (WGSL)");

                // Create resolution uniform buffer for test pattern
                let res_buffer = device.create_buffer(&BufferDescriptor {
                    label: Some("test_pattern_resolution"),
                    size: std::mem::size_of::<ResolutionUniform>() as u64,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let initial_res = ResolutionUniform {
                    resolution: [config.width as f32, config.height as f32],
                    _pad: [0.0, 0.0],
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
                    source: wgpu::ShaderSource::Wgsl(TEST_PATTERN_SHADER.into()),
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
                resolution: [self.config.width as f32, self.config.height as f32],
                _pad: [0.0, 0.0],
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
                    // Set uniform bind group if available (set=0)
                    if let Some(ref bind_group) = self.uniform_bind_group {
                        rpass.set_bind_group(0, bind_group, &[]);
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

// Load a .spv file and create a ShaderModule using the SPIR-V helper
fn load_spirv_module(device: &wgpu::Device, path: &Path) -> Result<wgpu::ShaderModule> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;

    // Basic SPIR-V validation
    if bytes.len() < 20 {
        return Err(anyhow!("SPIR-V file too small ({} bytes)", bytes.len()));
    }
    if bytes.len() % 4 != 0 {
        return Err(anyhow!(
            "SPIR-V file size {} is not aligned to 4-byte words",
            bytes.len()
        ));
    }

    // Check magic number
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != 0x07230203 {
        return Err(anyhow!(
            "Invalid SPIR-V magic number: 0x{:08x} (expected 0x07230203)",
            magic
        ));
    }

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
            eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
            eprintln!("║                  SHADER COMPILATION ERROR                    ║");
            eprintln!("╚══════════════════════════════════════════════════════════════╝");
            eprintln!("File: {}", path.display());
            eprintln!("Mode: SPIR-V passthrough");
            eprintln!();
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
            eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
            eprintln!("║                  SHADER COMPILATION ERROR                    ║");
            eprintln!("╚══════════════════════════════════════════════════════════════╝");
            eprintln!("File: {}", path.display());
            eprintln!("Mode: naga validation");
            eprintln!();
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
        let window_size = match &self.spec {
            PipelineSpec::VertexFragment { size, .. } => *size,
            PipelineSpec::TestPattern { .. } => None,
        };
        let mut attrs = WindowAttributes::default().with_title("wgpu + SPIR-V");
        if let Some((w, h)) = window_size {
            attrs = attrs.with_inner_size(PhysicalSize::new(w, h));
        }
        let window = match event_loop.create_window(attrs)
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

async fn validate_spirv(path: &Path, verbose: bool) -> Result<()> {
    let instance = Instance::new(&InstanceDescriptor {
        flags: InstanceFlags::VALIDATION,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .context("No suitable GPU adapter found")?;

    if verbose {
        let info = adapter.get_info();
        eprintln!("[validate] Adapter: {} ({:?})", info.name, info.backend);
    }

    let (device, _queue) = adapter
        .request_device(&DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: Trace::Off,
        })
        .await
        .context("Failed to create GPU device")?;

    if verbose {
        // Show detected entry points
        let bytes = fs::read(path)?;
        if bytes.len() >= 20 && bytes.len() % 4 == 0 {
            let words: Vec<u32> =
                bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let mut loader = Loader::new();
            if parse_words(&words, &mut loader).is_ok() {
                let module = loader.module();
                for ep in &module.entry_points {
                    let name = ep.operands.iter().find_map(|op| {
                        if let Operand::LiteralString(s) = op { Some(s.as_str()) } else { None }
                    });
                    let model = ep.operands.iter().find_map(|op| {
                        if let Operand::ExecutionModel(m) = op { Some(format!("{:?}", m)) } else { None }
                    });
                    eprintln!(
                        "[validate] Entry point: {} ({})",
                        name.unwrap_or("?"),
                        model.as_deref().unwrap_or("?")
                    );
                }
            }
        }
    }

    eprintln!("[validate] Loading {}", path.display());
    match load_spirv_module(&device, path) {
        Ok(_) => {
            eprintln!("[validate] OK — shader module created successfully");
            Ok(())
        }
        Err(e) => Err(e),
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
            no_validate,
            present_mode,
            difficulty,
            size,
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
                validate: !no_validate, // validation is ON by default
                present_mode: present_mode.into(),
                difficulty,
                size,
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
            storage_buffers,
            push_constants,
            verbose,
        } => {
            let entry_name = entry.unwrap_or_else(|| {
                auto_detect_compute_entry_point(&path).unwrap_or_else(|e| {
                    eprintln!("{}", e);
                    std::process::exit(1);
                })
            });

            // Parse storage buffer specs
            let storage_specs: Vec<StorageBufferSpec> =
                storage_buffers.iter().map(|s| StorageBufferSpec::parse(s)).collect::<Result<Vec<_>>>()?;

            pollster::block_on(run_compute_shader(
                path,
                entry_name,
                (workgroups_x, workgroups_y, workgroups_z),
                storage_specs,
                &push_constants,
                verbose,
            ))?;
        }
        Command::Run {
            path,
            pipeline,
            inputs,
            outputs,
            push_constants,
            verbose,
        } => {
            // Parse name:file pairs
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

            pollster::block_on(run_pipeline(path, pipeline, input_map, output_map, &push_constants, verbose))?;
        }
        Command::Miner {
            path,
            header,
            nonces,
            nonce_offset,
            difficulty,
            workgroups,
            verbose,
        } => {
            pollster::block_on(run_miner(path, header, nonces, nonce_offset, difficulty, workgroups, verbose))?;
        }
        Command::Validate { path, verbose } => {
            pollster::block_on(validate_spirv(&path, verbose))?;
            return Ok(());
        }
        Command::TestPattern { max_frames, verbose } => {
            eprintln!("[viz] Test pattern mode - built-in WGSL shader");

            let spec = PipelineSpec::TestPattern { max_frames, verbose };

            let event_loop = EventLoop::new().context("failed to create event loop")?;
            let mut app = App { state: None, spec };

            if let Err(e) = event_loop.run_app(&mut app) {
                return Err(anyhow!(e)).context("winit event loop errored");
            }
        }
    }

    Ok(())
}
