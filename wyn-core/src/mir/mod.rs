//! SSA-based intermediate representation for the Wyn compiler.
//!
//! This module provides the SSA (Static Single Assignment) representation used
//! after TLC lowering. The pipeline is: TLC → SSA → SPIR-V/GLSL.
//!
//! ## Submodules
//!
//! - `ssa`: Core SSA types (FuncBody, Inst, InstKind, Terminator, etc.)
//! - `ssa_builder`: Builder for constructing SSA functions
//! - `ssa_soac_analysis`: Analysis for identifying SOAC patterns
//! - `ssa_verify`: Verification of SSA invariants
//! - `layout`: Type layout calculations for SPIR-V memory operations

pub mod layout;
pub mod ssa;
pub mod ssa_builder;
pub mod ssa_soac_analysis;
pub mod ssa_verify;

#[cfg(test)]
mod ssa_builder_tests;
#[cfg(test)]
mod ssa_soac_analysis_tests;
#[cfg(test)]
mod ssa_tests;
#[cfg(test)]
mod ssa_verify_tests;
