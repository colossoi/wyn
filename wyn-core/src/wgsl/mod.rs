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

pub mod ssa_lowering;

pub use ssa_lowering::lower;
