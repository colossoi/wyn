//! WGSL SSA-to-text lowering.
//!
//! Scaffold only for now. Subsequent commits will fill in:
//! 1. Type mapping (`type_to_wgsl`)
//! 2. Identifier mangling with WGSL reserved-word handling
//! 3. Per-`InstKind` emission (literals, arithmetic, tuple/array
//!    operations, memory, storage views, dynamic indexing)
//! 4. Entry-point wrapping (`@vertex`, `@fragment`, `@compute`)
//! 5. Structured control flow via `crate::structured::structurize`

use crate::error::Result;
use crate::ssa::types::Program;

/// Lower an SSA program to a WGSL module. Returns a single `String`
/// containing the full module (all entry points, types, bindings, and
/// functions) — WGSL doesn't split into vertex/fragment/compute files
/// the way GLSL sometimes does, it's one module with `@vertex` /
/// `@fragment` / `@compute` attributes distinguishing entry points.
pub fn lower(_program: &Program) -> Result<String> {
    Err(crate::err_wgsl!("WGSL backend is not yet implemented"))
}

#[cfg(test)]
#[path = "ssa_lowering_tests.rs"]
mod ssa_lowering_tests;
