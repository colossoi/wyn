//! SSA-based intermediate representation for the Wyn compiler.
//!
//! With EGIR as the mid-end, this layer is strictly "the IR the codegens
//! consume". The builder emits it from EGIR's `elaborate`; SSA then prunes
//! unreachable module definitions and records target validation before the
//! SPIR-V / WGSL backends read it.
//!
//! ## Submodules
//!
//! - `framework`: Generic `Function` / `BasicBlock` / `InstNode` / `Terminator`
//!   types parameterized over instruction + value-type kind.
//! - `types`: Wyn-specific `InstKind`, `Program`, and the
//!   concrete `FuncBody = Function<InstKind, Type>` instantiation.
//! - `builder`: `FuncBuilder` that EGIR's `elaborate` uses to materialize SSA.
//! - `reachability`: whole-module function and constant definition pruning.
//! - `layout`: Type byte-size helpers for SPIR-V memory operations.
//! - `print`: Debug formatter for SSA bodies.

pub mod builder;
pub mod framework;
pub mod layout;
pub mod print;
pub mod reachability;
pub(crate) mod storage_function_variants;
pub mod types;

pub use reachability::filter_reachable;
pub use types::{context, stage, Program};

/// Validate reachable SSA for SPIR-V and record that proof in its
/// top-level type.
pub fn prepare_spirv(program: stage::Reachable) -> crate::error::Result<stage::SpirvReady> {
    if program.global_context.profile.target == crate::CodegenTarget::Wgsl {
        return Err(crate::err_spirv!(
            "SSA was scheduled for WGSL and cannot be lowered as SPIR-V"
        ));
    }
    crate::egir::verify_no_abstract::verify_no_abstract_types(&program)?;
    crate::spirv::verify_buffer_layouts::verify_buffer_layouts(&program)?;
    Ok(program.retag())
}

/// Validate reachable SSA for WGSL and record that proof in its
/// top-level type.
pub fn prepare_wgsl(program: stage::Reachable) -> crate::error::Result<stage::WgslReady> {
    if program.global_context.profile.target == crate::CodegenTarget::Spirv {
        return Err(crate::err_spirv!(
            "SSA was scheduled for SPIR-V and cannot be lowered as WGSL"
        ));
    }
    crate::egir::verify_no_abstract::verify_no_abstract_types(&program)?;
    Ok(program.retag())
}

#[cfg(test)]
mod tests;
