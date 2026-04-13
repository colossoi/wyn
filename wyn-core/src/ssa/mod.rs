//! SSA-based intermediate representation for the Wyn compiler.
//!
//! This module provides the SSA (Static Single Assignment) representation used
//! after TLC lowering. The pipeline is: TLC → SSA → SPIR-V/GLSL.
//!
//! ## Submodules
//!
//! - `types`: Core SSA types (FuncBody, Inst, InstKind, Terminator, etc.)
//! - `builder`: Builder for constructing SSA functions
//! - `verify`: Verification of SSA invariants
//! - `layout`: Type layout calculations for SPIR-V memory operations

pub mod builder;
pub mod framework;
pub mod layout;
pub mod merge;
pub mod print;
pub mod types;
pub mod verify;

#[cfg(test)]
mod builder_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod verify_tests;
