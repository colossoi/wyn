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
pub(crate) mod storage_function_variants;
pub mod types;

pub use types::{context, stage, Program, Stage};

/// Validate an elaborated SSA program for SPIR-V and record that proof in its
/// top-level type.
pub fn prepare_spirv(
    program: Program<stage::Elaborated>,
) -> crate::error::Result<Program<stage::SpirvReady>> {
    if program.global_context.profile.target == crate::CodegenTarget::Wgsl {
        return Err(crate::err_spirv!(
            "SSA was scheduled for WGSL and cannot be lowered as SPIR-V"
        ));
    }
    crate::egir::verify_no_abstract::run(&program)?;
    crate::spirv::verify_buffer_layouts::run(&program)?;
    Ok(program.into_stage())
}

/// Validate an elaborated SSA program for WGSL and record that proof in its
/// top-level type.
pub fn prepare_wgsl(
    program: Program<stage::Elaborated>,
) -> crate::error::Result<Program<stage::WgslReady>> {
    if program.global_context.profile.target == crate::CodegenTarget::Spirv {
        return Err(crate::err_spirv!(
            "SSA was scheduled for SPIR-V and cannot be lowered as WGSL"
        ));
    }
    crate::egir::verify_no_abstract::run(&program)?;
    Ok(program.into_stage())
}

#[cfg(test)]
mod tests;
