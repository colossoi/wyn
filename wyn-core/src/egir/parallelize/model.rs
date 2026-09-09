//! Shared target-planning policy, decisions, errors, and immutable indexes.

use thiserror::Error;

use super::schedule::KernelMutationError;

pub(super) const REDUCE_PHASE1_WIDTH: u32 = 64;
pub(super) const REDUCE_PHASE2_WIDTH: u32 = 256;
pub(super) const FILTER_SCAN_GROUPS: u32 = 4;

#[derive(Debug, Error)]
pub(super) enum ParallelizeError {
    #[error("{0}")]
    Invalid(String),
    #[error("kernel schedule mutation failed: {0}")]
    Schedule(#[from] KernelMutationError),
}

impl From<String> for ParallelizeError {
    fn from(value: String) -> Self {
        Self::Invalid(value)
    }
}

impl From<&str> for ParallelizeError {
    fn from(value: &str) -> Self {
        Self::Invalid(value.to_owned())
    }
}

pub(super) type Result<T> = std::result::Result<T, ParallelizeError>;

/// Candidate analysis either selects a target recipe or explains why the
/// operation must use fallback lowering.
pub(super) enum CandidateSelection<T> {
    Selected(T),
    Fallback,
}
