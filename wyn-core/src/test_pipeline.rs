//! Shared test entry point for the TLC pipeline.
//!
//! One front-end + `optimize_for_test` chain that every per-pass test module
//! calls, so a pipeline reorder only has to touch `optimize_for_test` rather
//! than a dozen bespoke front-end-plus-chain copies.

use crate::tlc::Program;
use crate::{cached_compiler_init, Compiler};

/// Front-end (parse → resolve → type-check → to_tlc → pin_entry_regions →
/// validate_ownership) shared by every `compile_*` helper, so they differ only
/// in how far down the `optimize_for_test` chain they run.
fn front_end(src: &str) -> crate::TlcOwnershipValidated {
    let (mut node_counter, mut module_manager) = cached_compiler_init();
    let type_checked = Compiler::parse(src, &mut node_counter)
        .expect("parse")
        .resolve(&module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    type_checked
        .to_tlc(&module_manager, false)
        .pin_entry_regions()
        .expect("pin_entry_regions")
        .validate_ownership()
        .expect("validate_ownership")
}

/// Run the front-end + the canonical `optimize_for_test` pipeline to
/// `TlcReachable`. `disable_parallelize = true` is the common case;
/// parallelizer / egir tests pass `false`.
pub(crate) fn compile_to_reachable(src: &str, disable_parallelize: bool) -> crate::TlcReachable {
    front_end(src).optimize_for_test(disable_parallelize)
}

/// As [`compile_to_reachable`], returning just the final TLC program.
pub(crate) fn compile_to_tlc_program(src: &str) -> Program {
    compile_to_reachable(src, true).0.tlc
}

// Stage-boundary helpers for per-pass tests in the residency cluster. Each
// returns the program at the input boundary of the next pass, so a test can run
// that pass itself and observe its effect (rather than re-running the whole
// pipeline, which already ran the pass — and everything after it).

/// Through `expose_entry_producer_helpers` — the input stage for
/// `static_index_fusion::run`.
pub(crate) fn compile_thru_expose_producers(src: &str) -> Program {
    front_end(src).optimize_for_test_thru_expose_producers().0.tlc
}

/// Through `fuse_static_indices` — the input stage for
/// `runtime_index_producers::run`.
pub(crate) fn compile_thru_static_index(src: &str) -> Program {
    front_end(src).optimize_for_test_thru_expose_producers().fuse_static_indices().0.tlc
}

/// Through `float_runtime_index_nested_producers` — the input stage for
/// `lift_gathers::run` (gather residency).
pub(crate) fn compile_thru_runtime_index(src: &str) -> Program {
    front_end(src)
        .optimize_for_test_thru_expose_producers()
        .fuse_static_indices()
        .float_runtime_index_nested_producers()
        .0
        .tlc
}
