//! Final TLC definition reachability boundary.
//!
//! The graph analysis and in-place elimination live in [`super::dce`] because
//! inlining also uses them for local cleanup. This module owns only the
//! externally visible phase transition.

use super::context::BackendGlobal;
use super::ownership::OwnershipApplied;

/// TLC containing only definitions reachable from an entry point.
#[derive(Debug, Clone, Copy)]
pub enum ReachableTag {}
pub type Reachable = super::Program<ReachableTag, super::defunctionalize::ClosureConverted, BackendGlobal>;

/// Eliminate unreachable definitions before semantic EGIR conversion.
///
/// Definition bodies are not rewritten: live definitions and their term IDs
/// move intact into the resulting program.
pub fn filter_reachable(program: OwnershipApplied) -> Reachable {
    let mut program = program.map_global_context::<ReachableTag, _>(|global| BackendGlobal {
        auto_storage_binding_ids: global.auto_storage_binding_ids,
    });

    super::dce::eliminate_unreachable_defs(&mut program.defs);
    super::defunctionalize::verify_hof_specialized(&program).unwrap_or_else(|error| {
        panic!(
            "hof-specialization verifier failed after filter_reachable: {}",
            error
        )
    });
    program
}
