//! Shared test entry point for the TLC pipeline.
//!
//! One front-end + `optimize_for_test` chain that every per-pass test module
//! calls, so a pipeline reorder only has to touch `optimize_for_test` rather
//! than a dozen bespoke front-end-plus-chain copies.

use crate::tlc::Program;
use crate::{cached_compiler_init, Compiler};

/// Run the front-end + the canonical `optimize_for_test` pipeline to
/// `TlcReachable`. `disable_parallelize = true` is the common case;
/// parallelizer / egir tests pass `false`.
pub(crate) fn compile_to_reachable(src: &str, disable_parallelize: bool) -> crate::TlcReachable {
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
        .optimize_for_test(disable_parallelize)
}

/// As [`compile_to_reachable`], returning just the final TLC program.
pub(crate) fn compile_to_tlc_program(src: &str) -> Program {
    compile_to_reachable(src, true).0.tlc
}
