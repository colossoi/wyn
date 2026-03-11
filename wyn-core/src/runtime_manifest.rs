//! Runtime manifest for multi-dispatch compute shader execution.
//!
//! The compiler emits a JSON manifest alongside the SPIR-V module describing:
//! - Buffer declarations (inputs, outputs, intermediates)
//! - Dispatch sequence (which entry points to call, in what order)
//!
//! A generic host runtime reads this manifest and executes the dispatches
//! sequentially. All algorithm knowledge lives in the compiler — the runtime
//! is SOAC-agnostic.

use serde::Serialize;

/// Runtime manifest describing the dispatch plan for a compiled program.
#[derive(Debug, Default, Serialize)]
pub struct RuntimeManifest {
    /// Buffer declarations (inputs, outputs, intermediates).
    pub buffers: Vec<BufferDecl>,
    /// Dispatches to execute in order.
    pub dispatches: Vec<Dispatch>,
}

/// A buffer used by the compute pipeline.
#[derive(Debug, Serialize)]
pub struct BufferDecl {
    /// Descriptor set index.
    pub set: u32,
    /// Binding index within the descriptor set.
    pub binding: u32,
    /// How this buffer is used.
    pub usage: BufferUsage,
    /// Human-readable name.
    pub name: String,
}

/// How a buffer is used in the dispatch pipeline.
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BufferUsage {
    /// Read-only input from the host.
    Input,
    /// Written by the final dispatch, read back by the host.
    Output,
    /// Written by one dispatch, read by a later dispatch. Not visible to the host.
    Intermediate,
}

/// A single compute shader dispatch.
#[derive(Debug, Serialize)]
pub struct Dispatch {
    /// Name of the SPIR-V entry point to invoke.
    pub entry_point: String,
    /// Workgroup size (local_size_x, local_size_y, local_size_z).
    pub workgroup_size: (u32, u32, u32),
    /// How to determine the number of workgroups to dispatch.
    pub dispatch_size: DispatchSize,
}

/// How to determine the dispatch grid size.
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DispatchSize {
    /// Fixed dispatch grid.
    Fixed { x: u32, y: u32, z: u32 },
    /// Derive from input array length: ceil(input_length / workgroup_size).
    DerivedFromInputLength { workgroup_size: u32 },
}
