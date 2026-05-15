//! Vertex-buffer support for the interactive `vf` mode.
//!
//! Mirrors the storage-buffer path in `uniforms.rs`: read the SPIR-V's
//! JSON sidecar, find every `vertex_inputs[i]` (a Wyn `#[location(n)]`
//! `#[vertex]` entry parameter), load `<dir>/<name>.bin` from the
//! `--storage-dir` argument as a `VERTEX`-usage buffer, and return the
//! info the render pipeline needs to wire each one up.
//!
//! One vertex buffer per attribute (matches the descriptor's
//! one-buffer-per-attribute convention — offset 0, stride = format
//! byte size). Interleaved buffers would need explicit offset/stride
//! on the descriptor side too; for the masthead this is fine.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use wgpu::{Buffer, BufferDescriptor, BufferUsages};
use wyn_pipeline_descriptor::VertexFormat;

use crate::json::{Pipeline, PipelineDescriptor};

/// One vertex-input attribute declared in the sidecar.
#[derive(Debug, Clone)]
struct VertexInputDecl {
    location: u32,
    name: String,
    format: VertexFormat,
}

/// Read `<spv_path>.json` and pull the `vertex_inputs` of the first
/// graphics pipeline. Empty when no sidecar or no vertex pipeline.
fn load_sidecar_vertex_inputs(spv_path: &Path) -> Vec<VertexInputDecl> {
    let json_path = spv_path.with_extension("json");
    let Ok(content) = fs::read_to_string(&json_path) else {
        return Vec::new();
    };
    let Ok(desc) = serde_json::from_str::<PipelineDescriptor>(&content) else {
        return Vec::new();
    };
    for p in &desc.pipelines {
        if let Pipeline::Graphics(g) = p {
            if !g.vertex_inputs.is_empty() {
                return g
                    .vertex_inputs
                    .iter()
                    .map(|v| VertexInputDecl {
                        location: v.location,
                        name: v.name.clone(),
                        format: v.format,
                    })
                    .collect();
            }
        }
    }
    Vec::new()
}

/// What the render pipeline needs to wire vertex buffers: the buffers
/// themselves (kept alive so they outlive the bind/draw calls), and
/// per-buffer `(format, shader_location)` so the
/// `VertexBufferLayout`s can be built at pipeline-creation time.
pub struct VertexBuffers {
    pub buffers: Vec<Buffer>,
    pub attribs: Vec<(VertexFormat, u32)>,
}

impl VertexBuffers {
    pub fn empty() -> Self {
        Self {
            buffers: Vec::new(),
            attribs: Vec::new(),
        }
    }
}

/// Build vertex buffers from a sidecar's `vertex_inputs` + a directory
/// of per-attribute `.bin` files. Returns empty when the shader
/// declares no vertex inputs (so the caller can use the same code
/// path for vertex-attribute and non-vertex-attribute shaders).
pub fn build_vertex_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    spv_path: &Path,
    storage_dir: Option<&Path>,
) -> Result<VertexBuffers> {
    let decls = load_sidecar_vertex_inputs(spv_path);
    if decls.is_empty() {
        return Ok(VertexBuffers::empty());
    }
    let Some(dir) = storage_dir else {
        return Err(anyhow!(
            "viz vf: shader declares {} vertex_input attribute(s) but no \
             --storage-dir was supplied; each attribute needs `<dir>/<name>.bin`",
            decls.len()
        ));
    };

    let mut buffers = Vec::with_capacity(decls.len());
    let mut attribs = Vec::with_capacity(decls.len());
    for decl in &decls {
        let path = dir.join(format!("{}.bin", decl.name));
        let data = fs::read(&path).with_context(|| {
            format!(
                "viz vf --storage-dir: vertex attribute '{}' (location {}) expects {:?}",
                decl.name, decl.location, path,
            )
        })?;
        // Sanity: the file size must be a multiple of the format stride;
        // mismatch usually means the .bin was packed for a different format.
        let stride = decl.format.byte_size() as usize;
        if stride == 0 || data.len() % stride != 0 {
            return Err(anyhow!(
                "viz vf --storage-dir: '{}' is {} bytes, not a multiple of the \
                 declared format's {}-byte stride",
                path.display(),
                data.len(),
                stride,
            ));
        }
        let buf = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("vertex_{}", decl.name)),
            size: data.len() as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, &data);
        buffers.push(buf);
        attribs.push((decl.format, decl.location));
    }

    Ok(VertexBuffers { buffers, attribs })
}

/// Map our `VertexFormat` to wgpu's.
pub fn wgpu_vertex_format(f: VertexFormat) -> wgpu::VertexFormat {
    use VertexFormat::*;
    match f {
        Float32 => wgpu::VertexFormat::Float32,
        Float32x2 => wgpu::VertexFormat::Float32x2,
        Float32x3 => wgpu::VertexFormat::Float32x3,
        Float32x4 => wgpu::VertexFormat::Float32x4,
        Sint32 => wgpu::VertexFormat::Sint32,
        Sint32x2 => wgpu::VertexFormat::Sint32x2,
        Sint32x3 => wgpu::VertexFormat::Sint32x3,
        Sint32x4 => wgpu::VertexFormat::Sint32x4,
        Uint32 => wgpu::VertexFormat::Uint32,
        Uint32x2 => wgpu::VertexFormat::Uint32x2,
        Uint32x3 => wgpu::VertexFormat::Uint32x3,
        Uint32x4 => wgpu::VertexFormat::Uint32x4,
    }
}
