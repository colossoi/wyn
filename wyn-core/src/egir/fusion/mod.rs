//! Compositional semantic fusion: snapshot, checked planning, then application.
use crate::egir::reify::Segmented;
use crate::egir::semantic_opt::SemanticOptimizationTrace;
use thiserror::Error;
mod algebra;
mod emit;
mod planner;
mod projection;
mod recipe;
mod snapshot;
mod space;

#[derive(Debug, Error)]
pub(crate) enum FusionError {
    #[error(transparent)]
    Graph(#[from] wyn_fusion::Error),
    #[error("semantic fusion: {0}")]
    InvalidCandidate(String),
}
impl FusionError {
    fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidCandidate(message.into())
    }
}
type FusionResult<T> = Result<T, FusionError>;

pub(super) fn run(
    program: Segmented,
    limit: Option<usize>,
) -> FusionResult<(Segmented, bool, SemanticOptimizationTrace)> {
    let (snapshot, catalog) = snapshot::Snapshot::build(&program)?;
    let planned = planner::plan(snapshot, limit)?;
    if planned.plan.actions().is_empty() {
        return Ok((program, false, SemanticOptimizationTrace::default()));
    }
    let (program, trace) = emit::apply(program, planned, catalog)?;
    Ok((program, true, trace))
}
#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
