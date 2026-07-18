//! Checked failures produced while selecting and constructing kernel recipes.

#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use thiserror::Error;

use super::schedule::KernelMutationError;

#[derive(Debug, Error)]
pub(crate) enum ParallelizeError {
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

pub(crate) type Result<T> = std::result::Result<T, ParallelizeError>;
