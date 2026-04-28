//! JSON I/O: pipeline descriptor parsing (mirrors
//! `wyn-core/src/pipeline_descriptor.rs`) and f32 array load/save used
//! by the descriptor-driven `pipeline` mode.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
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


