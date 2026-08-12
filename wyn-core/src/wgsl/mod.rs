//! WGSL code generation backend.
//!
//! WGSL is the W3C shading language for WebGPU. The backend aims for
//! full SPIR-V parity: compute shaders with `@workgroup_size`, storage
//! buffers via `@group/@binding`, strict types, structs, and dynamic
//! indexing through `var<function>` locals.
//!
//! The `structurize` pass (in `crate::structured`) reshapes the SSA CFG
//! into a target-agnostic Node tree of sequential statements, if-else,
//! and while loops — the constructs WGSL exposes textually. Everything
//! else (type mapping, identifier mangling, instruction dispatch,
//! entry-point emission) is WGSL-specific and written against the WGSL
//! spec.

mod int64_emulation;
pub mod ssa_lowering;

pub use ssa_lowering::{lower, lower_with_options};

/// Policy for 64-bit integer values that reach the WGSL backend.
///
/// WGSL has no concrete 64-bit integer type. Keeping this policy explicit
/// prevents backend-specific emulation from silently changing the cost model
/// or ABI of every WGSL compilation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WgslInt64Mode {
    /// Reject runtime 64-bit integer values, preserving the historical
    /// behavior of the WGSL backend.
    #[default]
    Reject,
    /// Emulate unsigned 64-bit integers with pairs of `u32` values.
    EmulateU64,
}

/// Backend-specific options for WGSL source generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WgslOptions {
    pub int64_mode: WgslInt64Mode,
}

impl WgslOptions {
    pub const U64_EMULATION: Self = Self {
        int64_mode: WgslInt64Mode::EmulateU64,
    };
}
