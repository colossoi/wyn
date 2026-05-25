//! Pipeline descriptor for compiled Wyn programs.
//!
//! The compiler emits a JSON pipeline descriptor alongside the SPIR-V module
//! describing how to execute the program: which entry points to invoke, in
//! what order, and what GPU resources (buffers, uniforms, push constants) each
//! stage uses.
//!
//! A generic host runtime (e.g. `viz`) reads this descriptor and sets up the
//! Vulkan/WebGPU pipeline accordingly. All algorithm knowledge lives in the
//! compiler.

use serde::{Deserialize, Serialize};

/// Top-level pipeline descriptor. One per compiled program.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PipelineDescriptor {
    /// Individual pipelines in this program (one per top-level entry or multi-dispatch SOAC).
    pub pipelines: Vec<Pipeline>,
}

/// A single pipeline within the program.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputePipeline {
    pub entry_point: String,
    pub workgroup_size: (u32, u32, u32),
    pub dispatch_size: DispatchSize,
    pub bindings: Vec<Binding>,
    /// Host-runtime default for the total work size, sourced from
    /// `#[size_hint(N)]` on an input parameter. When the application
    /// doesn't supply an explicit dispatch count, a thin host can
    /// dispatch `ceil(default_total_threads / workgroup_size.0)`
    /// workgroups without inspecting buffer length. The compiled
    /// shader does not assume the actual length equals this hint —
    /// it remains dynamic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_total_threads: Option<std::num::NonZeroU32>,
}

/// Multi-dispatch compute pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiComputePipeline {
    /// All bindings used across all stages.
    pub bindings: Vec<Binding>,
    /// Stages to execute in order.
    pub stages: Vec<ComputeStage>,
    /// Host-runtime default for the total work size; same semantics as
    /// `ComputePipeline::default_total_threads`, applied to whichever
    /// stages dispatch `DerivedFromInputLength`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_total_threads: Option<std::num::NonZeroU32>,
}

/// A single stage in a multi-dispatch compute pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphicsPipeline {
    pub stages: Vec<GraphicsStage>,
    pub bindings: Vec<Binding>,
    pub vertex_inputs: Vec<VertexAttribute>,
    pub fragment_outputs: Vec<FragmentOutput>,
}

/// A stage in a graphics pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphicsStage {
    pub entry_point: String,
    pub stage: ShaderStage,
}

/// Shader stage type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShaderStage {
    Vertex,
    Fragment,
}

/// How to determine the compute dispatch grid size.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Sampled texture (descriptor set binding). Bound from a
    /// `#[texture(set, binding)]` entry-point param of type `texture2d`.
    Texture {
        set: u32,
        binding: u32,
        name: String,
        sample_type: TextureSampleType,
        view_dimension: TextureViewDimension,
        multisampled: bool,
    },
    /// Sampler (descriptor set binding). Bound from a
    /// `#[sampler(set, binding)]` entry-point param of type `sampler`.
    Sampler {
        set: u32,
        binding: u32,
        name: String,
        binding_type: SamplerBindingType,
    },
}

impl Binding {
    /// Descriptor-set binding number for storage / uniform / texture /
    /// sampler bindings. Panics on `PushConstant`, which has no binding
    /// number — push constants live in their own range and are addressed
    /// by offset.
    pub fn wgpu_binding(&self) -> u32 {
        match self {
            Binding::StorageBuffer { binding, .. } => *binding,
            Binding::Uniform { binding, .. } => *binding,
            Binding::Texture { binding, .. } => *binding,
            Binding::Sampler { binding, .. } => *binding,
            Binding::PushConstant { .. } => panic!("PushConstant has no binding number"),
        }
    }

    /// True iff this is a storage buffer marked as a host-supplied input.
    pub fn is_input(&self) -> bool {
        matches!(
            self,
            Binding::StorageBuffer {
                usage: BufferUsage::Input,
                ..
            }
        )
    }
}

/// Access mode for a storage buffer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

/// How a buffer is used in the pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BufferUsage {
    /// Read-only input from the host.
    Input,
    /// Written by the pipeline, read back by the host.
    Output,
    /// Internal to the pipeline (written by one stage, read by another).
    Intermediate,
}

/// Sampled type of a texture binding. Mirrors the wgpu
/// `TextureSampleType` subset Wyn produces. v1 always emits
/// `Float { filterable: true }` (the only `texture2d` sampled type).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureSampleType {
    Float {
        filterable: bool,
    },
    Sint,
    Uint,
    Depth,
}

/// View dimension of a texture binding. v1 always emits `D2`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureViewDimension {
    D1,
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
}

/// Sampler binding mode. v1 always emits `Filtering`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerBindingType {
    Filtering,
    NonFiltering,
    Comparison,
}

/// Scalar/vector format of a vertex-buffer attribute. Mirrors the
/// wgpu `VertexFormat` subset Wyn can currently produce — 32-bit
/// float / signed-int / unsigned-int scalars and 2-4 wide vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
}

impl VertexFormat {
    /// Byte size of one attribute element: 4 bytes per 32-bit component.
    pub fn byte_size(self) -> u32 {
        use VertexFormat::*;
        match self {
            Float32 | Sint32 | Uint32 => 4,
            Float32x2 | Sint32x2 | Uint32x2 => 8,
            Float32x3 | Sint32x3 | Uint32x3 => 12,
            Float32x4 | Sint32x4 | Uint32x4 => 16,
        }
    }
}

/// Vertex input attribute. One attribute == one vertex buffer: the
/// host uploads a tightly-packed buffer per attribute (offset 0,
/// stride = `format.byte_size()`), mirroring viz's one-`.bin`-per-
/// binding `--storage-dir` convention. Interleaved buffers (explicit
/// offset/stride) are a later extension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexAttribute {
    pub location: u32,
    pub name: String,
    pub format: VertexFormat,
}

/// Fragment output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentOutput {
    pub location: u32,
    pub name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_format_byte_size() {
        assert_eq!(VertexFormat::Float32.byte_size(), 4);
        assert_eq!(VertexFormat::Sint32.byte_size(), 4);
        assert_eq!(VertexFormat::Uint32.byte_size(), 4);
        assert_eq!(VertexFormat::Float32x2.byte_size(), 8);
        assert_eq!(VertexFormat::Float32x3.byte_size(), 12);
        assert_eq!(VertexFormat::Float32x4.byte_size(), 16);
        assert_eq!(VertexFormat::Uint32x4.byte_size(), 16);
    }

    #[test]
    fn vertex_attribute_serde_round_trip() {
        let attr = VertexAttribute {
            location: 1,
            name: "color".to_string(),
            format: VertexFormat::Float32x3,
        };
        let json = serde_json::to_string(&attr).unwrap();
        // Format serializes snake_case.
        assert!(json.contains("\"float32x3\""), "got: {json}");
        let back: VertexAttribute = serde_json::from_str(&json).unwrap();
        assert_eq!(back.location, 1);
        assert_eq!(back.name, "color");
        assert_eq!(back.format, VertexFormat::Float32x3);
    }
}
