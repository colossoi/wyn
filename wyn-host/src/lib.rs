//! Typed host programs with WHL and Rust/WGPU emitters.

pub mod arithmetic;
pub mod interface;
mod program;
pub mod readback;
mod results;
mod rust_results;
mod rust_wgpu;
mod whl;

pub use interface::{
    Access, BackingRef, Binding, BlendMode, BufferLen, BufferUsage, ComputePipeline, ComputeStage,
    CullMode, DepthTest, DispatchLen, DispatchSize, DrawBufferRef, DrawCall, DrawCount, FillMode,
    FragmentOutput, FragmentState, FrameAccess, FrameBindingRef, FrameGraph, FramePass, FramePassKind,
    FrameResource, FrameResourceExtent, FrameResourceKind, FrontFace, GraphicsInvocation, GraphicsPipeline,
    GraphicsStage, HostSizeInput, HostSizeScalar, IndexFormat, IntegerOp, ModuleInterface, Pipeline,
    PrimitiveTopology, RasterState, SamplerBindingType, Scissor, ShaderStage, SizeExpr, SizeOp,
    SourceResultBinding, StageBindingUses, StorageImageFormat, StorageTextureSize, TextureSampleType,
    TextureViewDimension, UniformMember, VertexAttribute, VertexFormat, Viewport,
};
pub use program::{Allocation, Entry, Expr, HostError, Operation, Program, ResourceId, ShaderFormat};
pub use results::{ResultField, ResultKind, ResultLayout, ResultScalar};
