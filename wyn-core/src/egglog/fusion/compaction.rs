//! Run compaction rewrites in their consumer-before-count order.
use super::parallel;
use crate::egglog::OptimizeError;
use egglog_engine::EGraph;

mod count;

pub(super) fn run(graph: &mut EGraph) -> Result<bool, OptimizeError> {
    if parallel::compact(graph)? {
        return Ok(true);
    }
    count::run(graph)
}
