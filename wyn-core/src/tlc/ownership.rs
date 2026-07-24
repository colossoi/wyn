//! TLC ownership validation, analysis, and ownership-driven rewriting.

mod analysis;
mod apply;
mod liveness;
mod validate;

pub use apply::{apply_ownership, OwnershipApplied};
pub use validate::{check, validate, OwnershipValidated};

#[cfg(test)]
use super::VarRef;
#[cfg(test)]
use analysis::{analyze, build, AnalysisState, Origin, OwnerId};
#[cfg(test)]
use apply::eligible_unique_input_soacs;

#[cfg(test)]
#[path = "ownership_tests.rs"]
mod ownership_tests;

#[cfg(test)]
#[path = "ownership_rebuild_tests.rs"]
mod ownership_rebuild_tests;
