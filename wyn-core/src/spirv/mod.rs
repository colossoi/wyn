//! SPIR-V code generation backend.
//!
//! This module contains the lowering pass from SSA to SPIR-V.

#[cfg(test)]
mod lowering_tests;
mod ssa_lowering;

pub use ssa_lowering::lower_ssa_program;
