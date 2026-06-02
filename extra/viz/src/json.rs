//! Re-export of the shared `wyn_pipeline_descriptor` types plus the
//! f32-array I/O used by the descriptor-driven `pipeline` mode.
//!
//! The descriptor schema lives in its own crate so the compiler
//! (which serializes it) and host runtimes (which deserialize it)
//! share a single source of truth.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};

pub use wyn_pipeline_descriptor::{
    Access, Binding, BufferUsage, ComputePipeline, DispatchLen, DispatchSize, MultiComputePipeline,
    Pipeline, PipelineDescriptor, SamplerBindingType, StorageImageFormat, TextureSampleType,
    TextureViewDimension,
};

pub fn load_f32_json(path: &Path) -> Result<Vec<f32>> {
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
pub fn write_f32_json(path: &Path, data: &[f32]) -> Result<()> {
    let json =
        serde_json::to_string_pretty(&data.iter().map(|&f| serde_json::json!(f)).collect::<Vec<_>>())?;
    fs::write(path, json).with_context(|| format!("Failed to write: {}", path.display()))
}
