//! Run compaction rewrites in their consumer-before-count order.
use super::parallel;
use crate::egglog::graph::Graph;
use crate::egglog::OptimizeError;

mod count;

pub(super) fn run(graph: &mut Graph) -> Result<bool, OptimizeError> {
    if parallel::compact(graph)? {
        return Ok(true);
    }
    count::run(graph)
}
