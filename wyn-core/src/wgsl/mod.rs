//! WGSL code generation backend.
//!
//! WGSL is the W3C shading language for WebGPU. Unlike the GLSL backend
//! (which targets WebGL2's reduced feature surface), this backend aims
//! for full SPIR-V parity: compute shaders with `@workgroup_size`,
//! storage buffers via `@group/@binding`, strict types, structs, and
//! dynamic indexing through `var<function>` locals.
//!
//! The `structurize` pass (in `crate::structured`) is shared with the
//! GLSL backend and provides a target-agnostic Node tree over the SSA
//! CFG. Everything else — type mapping, identifier mangling, instruction
//! dispatch, entry-point emission — is WGSL-specific and written against
//! the WGSL spec, not adapted from GLSL.

pub mod ssa_lowering;

pub use ssa_lowering::lower;
