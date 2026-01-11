//! JSON pipeline configuration and execution.
//!
//! Allows running compute shaders from JSON configuration files, supporting
//! multi-buffer setups and push constants for iterative dispatch patterns.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Runtime configuration for a compute pipeline.
#[derive(Debug, Deserialize)]
pub struct PipelineConfig {
    /// Path to the interface JSON (from Wyn compiler).
    pub interface: PathBuf,
    /// Path to the SPIR-V shader.
    pub spirv: PathBuf,
    /// Dispatch dimensions [x, y, z].
    pub dispatch: [u32; 3],
    /// Buffer data sources (name -> file path).
    #[serde(default)]
    pub buffer_data: HashMap<String, PathBuf>,
    /// Push constants (name -> value).
    #[serde(default)]
    pub push_constants: HashMap<String, u32>,
    /// Entry point name (defaults to "main").
    #[serde(default = "default_entry")]
    pub entry: String,
}

fn default_entry() -> String {
    "main".to_string()
}

/// Shader interface (matches Wyn's --output-interface format).
#[derive(Debug, Deserialize)]
pub struct ShaderInterface {
    pub version: String,
    pub entry_points: Vec<EntryPointInfo>,
    pub buffers: Vec<BufferBindingInfo>,
}

#[derive(Debug, Deserialize)]
pub struct EntryPointInfo {
    pub name: String,
    pub execution_model: String,
    pub workgroup_size: Option<[u32; 3]>,
}

#[derive(Debug, Deserialize)]
pub struct BufferBindingInfo {
    pub name: String,
    pub set: u32,
    pub binding: u32,
    pub buffer_type: String,
    pub layout: String,
    pub access: String,
    pub fields: Vec<BufferFieldInfo>,
}

#[derive(Debug, Deserialize)]
pub struct BufferFieldInfo {
    pub name: String,
    pub type_name: String,
    pub offset: u32,
}

impl PipelineConfig {
    /// Load pipeline configuration from a JSON file.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read pipeline config: {:?}", path))?;
        let config: Self = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse pipeline config: {:?}", path))?;
        Ok(config)
    }

    /// Resolve relative paths in the config relative to the config file's directory.
    pub fn resolve_paths(&mut self, base_dir: &Path) {
        if self.interface.is_relative() {
            self.interface = base_dir.join(&self.interface);
        }
        if self.spirv.is_relative() {
            self.spirv = base_dir.join(&self.spirv);
        }
        for path in self.buffer_data.values_mut() {
            if path.is_relative() {
                *path = base_dir.join(&*path);
            }
        }
    }
}

impl ShaderInterface {
    /// Load shader interface from a JSON file.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read interface: {:?}", path))?;
        let interface: Self = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse interface: {:?}", path))?;
        Ok(interface)
    }

    /// Find entry point by name.
    pub fn find_entry_point(&self, name: &str) -> Option<&EntryPointInfo> {
        self.entry_points.iter().find(|ep| ep.name == name)
    }

    /// Get buffers sorted by binding index.
    pub fn buffers_by_binding(&self) -> Vec<&BufferBindingInfo> {
        let mut buffers: Vec<_> = self.buffers.iter().collect();
        buffers.sort_by_key(|b| (b.set, b.binding));
        buffers
    }
}

/// Load SPIR-V from a file path.
pub fn load_spirv(path: &Path) -> Result<Vec<u32>> {
    let bytes = std::fs::read(path).with_context(|| format!("Failed to read SPIR-V: {:?}", path))?;

    if bytes.len() % 4 != 0 {
        bail!("SPIR-V file size must be a multiple of 4 bytes");
    }

    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(words)
}

/// Load binary buffer data from a file (as f32 values).
pub fn load_buffer_data(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("Failed to read buffer data: {:?}", path))?;

    if bytes.len() % 4 != 0 {
        bail!("Buffer data file size must be a multiple of 4 bytes");
    }

    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(floats)
}

/// Load binary buffer data from a file (as u32 values).
pub fn load_buffer_data_u32(path: &Path) -> Result<Vec<u32>> {
    let bytes = std::fs::read(path).with_context(|| format!("Failed to read buffer data: {:?}", path))?;

    if bytes.len() % 4 != 0 {
        bail!("Buffer data file size must be a multiple of 4 bytes");
    }

    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(words)
}
