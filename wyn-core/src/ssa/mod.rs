//! SSA-based intermediate representation for the Wyn compiler.
//!
//! With EGIR as the mid-end, this layer is strictly "the IR the codegens
//! consume". No optimization passes live here — the types are defined, the
//! builder emits them from EGIR's `elaborate`, and the SPIR-V / WGSL backends
//! read them.
//!
//! ## Submodules
//!
//! - `framework`: Generic `Function` / `BasicBlock` / `InstNode` / `Terminator`
//!   types parameterized over instruction + value-type kind.
//! - `types`: Wyn-specific `InstKind`, `Program`, and the
//!   concrete `FuncBody = Function<InstKind, Type>` instantiation.
//! - `builder`: `FuncBuilder` that EGIR's `elaborate` uses to materialize SSA.
//! - `layout`: Type byte-size helpers for SPIR-V memory operations.
//! - `print`: Debug formatter for SSA bodies.

pub mod builder;
pub mod framework;
pub mod layout;
pub mod print;
pub mod types;

#[cfg(test)]
mod tests;
