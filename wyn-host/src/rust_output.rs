//! Output data returned by generated host functions. Clients own readback and decoding.
use wgpu::{Buffer, Texture};

#[derive(Debug, Clone)]
pub struct OutputDescriptor {
    pub entry: &'static str,
    /// Source result order, followed by any additional render targets.
    pub values: Vec<OutputValue>,
}

#[derive(Debug, Clone)]
pub struct OutputValue {
    pub name: &'static str,
    pub kind: ResultKind,
    /// Identity in RESOURCE_NAMES; repeated identities denote shared backing.
    pub resource_id: usize,
    pub resource: OutputResource,
}

#[derive(Debug, Clone)]
pub enum OutputResource {
    Buffer {
        /// Retains the GPU allocation. Reading it requires suitable COPY_SRC usage.
        buffer: Buffer,
        layout: ResultLayout,
        range: BufferRange,
    },
    /// Retains the texture for presentation, further GPU work, or client readback.
    Texture(Texture),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferRange {
    Bytes {
        offset: u64,
        size: u64,
    },
    /// The compiler has not published the logical view's byte range.
    /// The client must supply it; buffer capacity is not the live result size.
    CallerProvided,
}
