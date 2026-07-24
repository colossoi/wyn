//! Shared test entry point for the TLC pipeline.
//!
//! One front-end + canonical TLC chain that every per-pass test module
//! calls, so a pipeline reorder only has to touch one helper rather
//! than a dozen bespoke front-end-plus-chain copies.

use crate::tlc::{self, Program};
use crate::{cached_compiler_init, Compiler};

/// Front-end (parse → resolve → type-check → to_tlc → pin_entry_buffers →
/// validate_ownership) shared by every `compile_*` helper, so they differ only
/// in how far down the canonical chain they run.
fn front_end(src: &str) -> Program<tlc::stage::OwnershipValidated> {
    let (mut node_counter, mut module_manager) = cached_compiler_init();
    let type_checked = Compiler::parse(src, &mut node_counter)
        .expect("parse")
        .resolve(&module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    let program = type_checked.to_tlc(&module_manager, false);
    let program = tlc::pin_entry_buffers(program).expect("pin_entry_buffers");
    tlc::validate_ownership(program).expect("validate_ownership")
}

/// Run the front-end + the canonical TLC pipeline to `Program<Reachable>`.
pub(crate) fn compile_to_reachable(src: &str) -> Program<tlc::stage::Reachable> {
    crate::optimize_tlc_for_test(front_end(src))
}

// Stage-boundary helpers for source-normalization pass tests. Each
// returns the program at the input boundary of the next pass, so a test can run
// that pass itself and observe its effect (rather than re-running the whole
// pipeline, which already ran the pass — and everything after it).

/// Through source-level SOAC ANF normalization, immediately before nested
/// runtime-index producers are floated.
pub(crate) fn compile_thru_expose_producers(src: &str) -> Program<tlc::stage::SoacsAnfNormalized> {
    crate::optimize_tlc_for_test_thru_soac_normalization(front_end(src))
}

/// Compatibility name for tests whose subject starts at the same SOAC ANF
/// boundary and then runs runtime-index producer exposure directly.
pub(crate) fn compile_thru_static_index(src: &str) -> Program<tlc::stage::SoacsAnfNormalized> {
    crate::optimize_tlc_for_test_thru_soac_normalization(front_end(src))
}
