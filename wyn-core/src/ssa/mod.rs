//! SSA-based intermediate representation for the Wyn compiler.
//!
//! With EGIR as the mid-end, this layer is strictly "the IR the codegens
//! consume". The builder emits it from EGIR's `elaborate`; SSA then prunes
//! unreachable module definitions and records target validation before the
//! SPIR-V / WGSL backends read it.
//!
//! ## Submodules
//!
//! - `ir`: Generic SSA representation and structural transformations.
//! - `types`: Wyn-specific `InstKind`, `Program`, and the
//!   concrete `FuncBody = Function<InstKind, Type>` instantiation.
//! - `builder`: `FuncBuilder` that EGIR's `elaborate` uses to materialize SSA.
//! - `reachability`: whole-module function and constant definition pruning.
//! - `backend_validation`: Concrete type representation checks for backend emission.
//! - `layout`: Type byte-size helpers for SPIR-V memory operations.
//! - `print`: Debug formatter for SSA bodies.

pub mod addressable_constants;
pub mod backend_validation;
pub mod builder;
pub mod ir;
pub mod layout;
mod optimize;
pub mod print;
pub mod reachability;
pub(crate) mod storage_function_variants;
pub mod types;
pub mod uses;

use crate::err_spirv;
use crate::error;
use crate::spirv;
use crate::CodegenTarget;
pub use addressable_constants::promote_addressable_constants;
pub use optimize::optimize;
pub use reachability::filter_reachable;
pub use types::{context, stage, Program};
pub use uses::{eliminate_dead_pure_instructions, UseSite, ValueUses};

/// Assign every reachable floating expression to a concrete SSA block.
pub fn place_floating(mut program: stage::Optimized) -> error::Result<stage::Placed> {
    for body in program
        .functions
        .iter_mut()
        .map(|function| &mut function.body)
        .chain(program.entry_points.iter_mut().map(|entry| &mut entry.body))
        .chain(program.constants.iter_mut().map(|constant| &mut constant.body))
    {
        ir::schedule_floating(&mut body.inner)
            .map_err(|error| error::CompilerError::Internal(error.to_string()))?;
    }
    Ok(program.retag())
}

fn eliminate_dead_values(program: &mut stage::Reachable) {
    for function in &mut program.functions {
        eliminate_dead_pure_instructions(&mut function.body);
    }
    for entry in &mut program.entry_points {
        eliminate_dead_pure_instructions(&mut entry.body);
    }
    for constant in &mut program.constants {
        eliminate_dead_pure_instructions(&mut constant.body);
    }
}

/// Validate reachable SSA for SPIR-V and record that proof in its
/// top-level type.
pub fn prepare_spirv(mut program: stage::Reachable) -> error::Result<stage::SpirvReady> {
    if program.global_context.profile.target == CodegenTarget::Wgsl {
        return Err(err_spirv!(
            "SSA was scheduled for WGSL and cannot be lowered as SPIR-V"
        ));
    }
    eliminate_dead_values(&mut program);
    backend_validation::verify_no_abstract_types(&program)?;
    spirv::verify_buffer_layouts::verify_buffer_layouts(&program)?;
    Ok(program.retag())
}

/// Validate reachable SSA for WGSL and record that proof in its
/// top-level type.
pub fn prepare_wgsl(mut program: stage::Reachable) -> error::Result<stage::WgslReady> {
    if program.global_context.profile.target == CodegenTarget::Spirv {
        return Err(err_spirv!(
            "SSA was scheduled for SPIR-V and cannot be lowered as WGSL"
        ));
    }
    promote_addressable_constants(&mut program);
    eliminate_dead_values(&mut program);
    backend_validation::verify_no_abstract_types(&program)?;
    Ok(program.retag())
}

#[cfg(test)]
mod tests;
