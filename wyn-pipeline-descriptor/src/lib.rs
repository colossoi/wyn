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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineDescriptor {
    /// Individual pipelines in this program (one per top-level entry or multi-dispatch SOAC).
    pub pipelines: Vec<Pipeline>,
}

/// A single pipeline within the program.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Pipeline {
    /// One or more compute dispatches sharing a binding table. A
    /// single-dispatch SOAC (Map, Scatter, simple compute) is the
    /// `stages.len() == 1` case; multi-dispatch SOACs (Reduce, Scan,
    /// Filter, ordered-prefix scheduling) populate multiple stages
    /// run in order by the host runtime.
    Compute(ComputePipeline),
    /// Graphics pipeline (Vertex → Fragment).
    Graphics(GraphicsPipeline),
}

/// Compute pipeline: a binding table plus N≥1 dispatch stages run in
/// order, sharing the same bindings. The `stages.len() == 1` case
/// covers single-dispatch SOACs; multi-stage covers Reduce/Scan/
/// Filter phase chains and the ordered-prefix scheduler's lifted
/// stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputePipeline {
    /// All bindings used across all stages.
    pub bindings: Vec<Binding>,
    /// Stages to execute in order. Length ≥ 1.
    pub stages: Vec<ComputeStage>,
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

/// A single dispatch stage within a `ComputePipeline`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeStage {
    pub entry_point: String,
    pub workgroup_size: (u32, u32, u32),
    pub dispatch_size: DispatchSize,
    /// Indices into the parent pipeline's `bindings` that this stage
    /// reads. Empty when the host derives access from `Binding.access`
    /// (the single-stage case has historically left this empty).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reads: Vec<usize>,
    /// Indices into the parent pipeline's `bindings` that this stage
    /// writes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DispatchSize {
    /// Fixed dispatch grid (absolute workgroup counts).
    Fixed {
        x: u32,
        y: u32,
        z: u32,
    },
    /// Dispatch `ceil(len / workgroup_size)` workgroups, where `len` is the
    /// number of iterations resolved from `DispatchLen`. Replaces the old
    /// `DerivedFromInputLength`, which only said "derive from *an* input
    /// buffer" and silently guessed the wrong one for range-driven maps.
    DerivedFrom {
        len: DispatchLen,
        workgroup_size: u32,
    },
}

/// The source of truth for a `DerivedFrom` dispatch's iteration count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DispatchLen {
    /// One iteration per element of the buffer at (`set`, `binding`) — e.g.
    /// `map(f, arr)` over a storage-buffer input. The host reads the buffer's
    /// element count.
    InputBinding {
        set: u32,
        binding: u32,
        /// Bytes per element of that buffer, so the host recovers the element
        /// count from its byte size.
        elem_bytes: u32,
    },
    /// A compile-time-known iteration count — e.g. `map(f, iota(6144))`.
    /// (Struct variant, not `Fixed(u32)`, so it serializes under the internal
    /// `kind` tag.)
    Fixed {
        count: u32,
    },
    /// A runtime count read from a scalar push-constant — e.g. `map(f,
    /// iota(n))` where `n` is an entry parameter. The host reads the u32 at
    /// `offset` in the push-constant block.
    PushConstant {
        offset: u32,
    },
    /// One iteration per texel of the storage texture at (`set`,
    /// `binding`) — used for compute entries whose primary output is a
    /// storage image (`image_store`). The host reads the allocated
    /// `wgpu::Texture`'s `width × height` (the storage texture's
    /// resolution is set by the descriptor's `StorageTextureSize`
    /// policy at allocation time). 2D dispatch: the host divides by
    /// the workgroup_size's x/y dims to produce workgroup counts.
    StorageImage {
        set: u32,
        binding: u32,
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
        /// Sizing policy for compiler-managed buffers whose length isn't a
        /// host-supplied input (e.g. a gather intermediate). `None` for
        /// host inputs (sized from the supplied data) and ordinary outputs.
        #[serde(default)]
        length: Option<BufferLen>,
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
    /// Storage image (descriptor set binding). Bound from a
    /// `#[storage_image(set, binding, format, access)]` entry-point
    /// param of type `storage_image`. The same `(set, binding)` slot
    /// may also be declared as a `Texture` in another pipeline — viz
    /// allocates one wgpu texture and binds it through two views.
    ///
    /// `size` is the resolution policy the host uses to allocate the
    /// backing `wgpu::Texture`. Defaults to `SameAsWindow` so a
    /// compute shader writing per-pixel naturally tracks the swapchain
    /// size; producers that want a fixed grid (e.g. the Mountains
    /// shader's BUFFER_SIZE-capped erosion textures) opt in to
    /// `Fixed`.
    StorageTexture {
        set: u32,
        binding: u32,
        name: String,
        format: StorageImageFormat,
        access: Access,
        #[serde(default)]
        size: StorageTextureSize,
    },
}

/// Resolution policy for a storage texture's backing `wgpu::Texture`.
/// Resolved by the host at allocation time (and on window resize for
/// `SameAsWindow`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StorageTextureSize {
    /// Track the swapchain surface size. The default — a fragment
    /// shader sampling this texture covers each output pixel exactly
    /// once.
    #[default]
    SameAsWindow,
    /// Fixed `(width, height)` in pixels. Used when the producer's
    /// dispatch is sized to a constant grid (e.g. the Mountains
    /// shader's `BUFFER_SIZE` cap that decouples compute resolution
    /// from window resolution).
    Fixed {
        width: u32,
        height: u32,
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
            Binding::StorageTexture { binding, .. } => *binding,
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

/// Compile-time sizing policy for a compiler-managed storage buffer whose
/// length isn't a host-supplied input. The host runtime resolves this to a
/// byte size when allocating the buffer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BufferLen {
    /// Minimum bytes the host must allocate. For an *output* binding this
    /// is the bytes the shader will write (so it's also the maximum
    /// useful size). For an *input* binding it's the bytes the shader
    /// will read (e.g. inferred from a `param[0..K]` slice); the host is
    /// free to over-allocate.
    Fixed {
        bytes: u64,
    },
    /// Same element *count* as the buffer at (`set`, `binding`), whose
    /// elements are `src_elem_bytes` each; this buffer's elements are
    /// `elem_bytes` each. Byte size = `src_bytes / src_elem_bytes *
    /// elem_bytes`. A `map` output keeps its input's element count but its
    /// element size may differ (e.g. `[]vec4f32` → `[]i32`).
    LikeInput {
        set: u32,
        binding: u32,
        elem_bytes: u32,
        src_elem_bytes: u32,
    },
    /// One `elem_bytes`-sized element per dispatched thread. A parallel
    /// `map`/`scan` writes exactly one output element per thread, so its
    /// output length equals the resolved dispatch thread count — which the
    /// host computes anyway (it covers buffer inputs, static and dynamic
    /// ranges uniformly). Byte size = `dispatch_threads * elem_bytes`. The
    /// thread count isn't a `src_bytes` lookup, so this resolves via
    /// `dispatch_elem_bytes`, not `resolve_bytes`.
    SameAsDispatch {
        elem_bytes: u32,
    },
}

impl BufferLen {
    /// Resolve to a byte size given a lookup of already-allocated buffers'
    /// byte sizes by (set, binding). Returns `None` if a referenced source
    /// buffer hasn't been sized yet, or for `SameAsDispatch` (which needs the
    /// resolved dispatch thread count — see `dispatch_elem_bytes`).
    pub fn resolve_bytes(&self, src_bytes: impl Fn(u32, u32) -> Option<u64>) -> Option<u64> {
        match self {
            BufferLen::Fixed { bytes } => Some(*bytes),
            BufferLen::LikeInput {
                set,
                binding,
                elem_bytes,
                src_elem_bytes,
            } => {
                let bytes = src_bytes(*set, *binding)?;
                Some(bytes / *src_elem_bytes as u64 * *elem_bytes as u64)
            }
            BufferLen::SameAsDispatch { .. } => None,
        }
    }

    /// Element byte size if this buffer is sized by the dispatch thread count
    /// (`SameAsDispatch`); the host multiplies it by the resolved threads.
    pub fn dispatch_elem_bytes(&self) -> Option<u32> {
        match self {
            BufferLen::SameAsDispatch { elem_bytes } => Some(*elem_bytes),
            _ => None,
        }
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

/// Pixel format for a storage-image binding. Bound at shader-compile
/// time via the `#[storage_image(..., format=FMT, ...)]` attribute;
/// the host allocates the wgpu texture with the matching format.
/// The whitelist starts narrow — formats are added as shaders demand
/// them. Names match the lowercase wgpu/WGSL spelling for round-trip
/// clarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageImageFormat {
    Rgba8Unorm,
    Rgba16Float,
    Rgba32Float,
    R32Float,
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
    fn buffer_len_resolve_like_input() {
        // A `[]vec4f32` (16 B/elem) input of 64 elements → 1024 bytes; a gather
        // of `[]i32` (4 B/elem) keeping its element count is 64 * 4 = 256 bytes.
        let len = BufferLen::LikeInput {
            set: 0,
            binding: 0,
            elem_bytes: 4,
            src_elem_bytes: 16,
        };
        assert_eq!(
            len.resolve_bytes(|s, b| (s == 0 && b == 0).then_some(1024)),
            Some(256)
        );
        // Source not yet allocated → unresolved.
        assert_eq!(len.resolve_bytes(|_, _| None), None);
        // Fixed is independent of any source.
        assert_eq!(
            BufferLen::Fixed { bytes: 40 }.resolve_bytes(|_, _| None),
            Some(40)
        );
    }

    #[test]
    fn buffer_len_same_as_dispatch() {
        // A dispatch-sized output isn't a `src_bytes` lookup — it resolves via
        // `dispatch_elem_bytes` and the host scales by the dispatch thread count.
        let len = BufferLen::SameAsDispatch { elem_bytes: 4 };
        assert_eq!(len.resolve_bytes(|_, _| Some(1024)), None);
        assert_eq!(len.dispatch_elem_bytes(), Some(4));
        assert_eq!(BufferLen::Fixed { bytes: 40 }.dispatch_elem_bytes(), None);
        let json = serde_json::to_string(&len).unwrap();
        assert!(json.contains("\"same_as_dispatch\""), "got: {json}");
        assert_eq!(serde_json::from_str::<BufferLen>(&json).unwrap(), len);
    }

    #[test]
    fn dispatch_len_serde_round_trip() {
        // `DerivedFrom` wraps an internally-tagged `DispatchLen`, so each source
        // variant must round-trip with its `kind` tag.
        for len in [
            DispatchLen::InputBinding {
                set: 0,
                binding: 1,
                elem_bytes: 4,
            },
            DispatchLen::Fixed { count: 6144 },
            DispatchLen::PushConstant { offset: 8 },
        ] {
            let size = DispatchSize::DerivedFrom {
                len: len.clone(),
                workgroup_size: 64,
            };
            let json = serde_json::to_string(&size).unwrap();
            assert!(json.contains("\"derived_from\""), "got: {json}");
            assert_eq!(serde_json::from_str::<DispatchSize>(&json).unwrap(), size);
        }
    }

    #[test]
    fn buffer_len_serde_round_trip() {
        let len = BufferLen::LikeInput {
            set: 0,
            binding: 2,
            elem_bytes: 4,
            src_elem_bytes: 16,
        };
        let json = serde_json::to_string(&len).unwrap();
        assert!(json.contains("\"like_input\""), "got: {json}");
        assert_eq!(serde_json::from_str::<BufferLen>(&json).unwrap(), len);
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
