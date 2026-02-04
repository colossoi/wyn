//! GLSL code generation backend.
//!
//! This module contains the SSA to GLSL lowering pass.

pub mod ssa_lowering;

pub use ssa_lowering::{GlslOutput, lower, lower_shadertoy};
