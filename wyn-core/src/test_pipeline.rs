//! Shared test entry point for the TLC pipeline.
//!
//! One front-end + canonical TLC chain that every per-pass test module
//! calls, so a pipeline reorder only has to touch one helper rather
//! than a dozen bespoke front-end-plus-chain copies.

use crate::tlc;

/// Front-end (parse → resolve → type-check → to_tlc → pin_entry_buffers →
/// validate_ownership) shared by every `compile_*` helper, so they differ only
/// in how far down the canonical chain they run.
fn front_end(src: &str) -> tlc::stage::OwnershipValidated {
    let type_checked = crate::compile_thru_frontend(src).expect("type_check");
    let program = crate::ast_type_holes::reject_type_holes(type_checked).expect("type holes");
    let program = tlc::lower_from_ast(program);
    let program = tlc::pin_entry_buffers(program).expect("pin_entry_buffers");
    tlc::validate_ownership(program).expect("validate_ownership")
}

/// Run the front-end + the canonical TLC pipeline to `tlc::stage::Reachable`.
pub(crate) fn compile_to_reachable(src: &str) -> tlc::stage::Reachable {
    crate::optimize_tlc_for_test(front_end(src))
}

// Stage-boundary helpers for source-normalization pass tests. Each
// returns the program at the input boundary of the next pass, so a test can run
// that pass itself and observe its effect (rather than re-running the whole
// pipeline, which already ran the pass — and everything after it).

/// Through source-level SOAC ANF normalization, immediately before nested
/// runtime-index producers are floated.
pub(crate) fn compile_thru_expose_producers(src: &str) -> tlc::stage::SoacsAnfNormalized {
    crate::optimize_tlc_for_test_thru_soac_normalization(front_end(src))
}

/// Compatibility name for tests whose subject starts at the same SOAC ANF
/// boundary and then runs runtime-index producer exposure directly.
pub(crate) fn compile_thru_static_index(src: &str) -> tlc::stage::SoacsAnfNormalized {
    crate::optimize_tlc_for_test_thru_soac_normalization(front_end(src))
}
