//! SSA-based intermediate representation for the Wyn compiler.
//!
//! This module provides the SSA (Static Single Assignment) representation used
//! after TLC lowering. The pipeline is: TLC → SSA → SPIR-V/GLSL.
//!
//! ## Submodules
//!
//! - `types`: Core SSA types (FuncBody, Inst, InstKind, Terminator, etc.)
//! - `builder`: Builder for constructing SSA functions
//! - `soac_analysis`: Analysis for identifying SOAC patterns
//! - `verify`: Verification of SSA invariants
//! - `layout`: Type layout calculations for SPIR-V memory operations

pub mod builder;
pub mod layout;
pub mod merge;
pub mod opt;
pub mod print;
pub mod reachability;
pub mod soa_helpers;
pub mod soac_analysis;
pub mod soac_lower;
pub mod types;
pub mod verify;

#[cfg(test)]
mod builder_tests;
#[cfg(test)]
mod soac_analysis_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod verify_tests;
