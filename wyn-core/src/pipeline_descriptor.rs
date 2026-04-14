//! Pipeline descriptor for compiled Wyn programs.
//!
//! The compiler emits a JSON pipeline descriptor alongside the SPIR-V module
//! describing how to execute the program: which entry points to invoke, in
//! what order, and what GPU resources (buffers, uniforms, push constants) each
//! stage uses.
//!
//! A generic host runtime reads this descriptor and sets up the Vulkan/WebGPU
//! pipeline accordingly. All algorithm knowledge lives in the compiler.

use serde::Serialize;

/// Top-level pipeline descriptor. One per compiled program.
#[derive(Debug, Default, Serialize)]
pub struct PipelineDescriptor {
    /// Individual pipelines in this program (one per top-level entry or multi-dispatch SOAC).
    pub pipelines: Vec<Pipeline>,
}

/// A single pipeline within the program.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Pipeline {
    /// Single compute dispatch (Map, Scatter, simple compute).
    Compute(ComputePipeline),
    /// Multi-dispatch compute (Reduce, Scan, Filter).
    MultiCompute(MultiComputePipeline),
    /// Graphics pipeline (Vertex → Fragment).
    Graphics(GraphicsPipeline),
}

/// Single-dispatch compute pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct ComputePipeline {
    pub entry_point: String,
    pub workgroup_size: (u32, u32, u32),
    pub dispatch_size: DispatchSize,
    pub bindings: Vec<Binding>,
}

/// Multi-dispatch compute pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct MultiComputePipeline {
    /// All bindings used across all stages.
    pub bindings: Vec<Binding>,
    /// Stages to execute in order.
    pub stages: Vec<ComputeStage>,
}

/// A single stage in a multi-dispatch compute pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct ComputeStage {
    pub entry_point: String,
    pub workgroup_size: (u32, u32, u32),
    pub dispatch_size: DispatchSize,
    /// Indices into the parent pipeline's `bindings` that this stage reads.
    pub reads: Vec<usize>,
    /// Indices into the parent pipeline's `bindings` that this stage writes.
    pub writes: Vec<usize>,
}

/// Graphics pipeline (vertex + fragment stages).
#[derive(Debug, Clone, Serialize)]
pub struct GraphicsPipeline {
    pub stages: Vec<GraphicsStage>,
    pub bindings: Vec<Binding>,
    pub vertex_inputs: Vec<VertexAttribute>,
    pub fragment_outputs: Vec<FragmentOutput>,
}

/// A stage in a graphics pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct GraphicsStage {
    pub entry_point: String,
    pub stage: ShaderStage,
}

/// Shader stage type.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ShaderStage {
    Vertex,
    Fragment,
}

/// How to determine the compute dispatch grid size.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DispatchSize {
    /// Fixed dispatch grid.
    Fixed {
        x: u32,
        y: u32,
        z: u32,
    },
    /// Derive from input array length: ceil(input_length / workgroup_size).
    DerivedFromInputLength {
        workgroup_size: u32,
    },
}

/// A GPU resource binding used by the pipeline.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Binding {
    /// Storage buffer (descriptor set binding).
    StorageBuffer {
        set: u32,
        binding: u32,
        access: Access,
        usage: BufferUsage,
        name: String,
    },
    /// Uniform buffer (descriptor set binding).
    Uniform {
        set: u32,
        binding: u32,
        name: String,
    },
    /// Push constant range.
    PushConstant {
        offset: u32,
        size: u32,
        name: String,
    },
}

/// Access mode for a storage buffer.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

/// How a buffer is used in the pipeline.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BufferUsage {
    /// Read-only input from the host.
    Input,
    /// Written by the pipeline, read back by the host.
    Output,
    /// Internal to the pipeline (written by one stage, read by another).
    Intermediate,
}

/// Vertex input attribute.
#[derive(Debug, Clone, Serialize)]
pub struct VertexAttribute {
    pub location: u32,
    pub name: String,
}

/// Fragment output.
#[derive(Debug, Clone, Serialize)]
pub struct FragmentOutput {
    pub location: u32,
    pub name: String,
}
