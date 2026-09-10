//! Shared structural facts for one immutable EGIR snapshot.
//!
//! Keep this context in the read phase of a pass and drop it before rewriting
//! its source. The graph borrow prevents querying these facts after a mutation.
//! Only requested facts are built. Parameter/capture seeds, solved dependence
//! results, and selection policy belong to individual consumers, not this owner.

use std::cell::OnceCell;

use super::block_interface::{self, BlockInterfaces};
use super::loop_analysis::LoopAnalysis;
use super::slice::{SliceFacts, ValueProducerPhase};
use super::types::{EGraph, Family, SideEffectIndex};

pub struct GraphAnalysis<'g, P: Family> {
    graph: &'g EGraph<P>,
    producers: OnceCell<SideEffectIndex>,
    interfaces: OnceCell<Result<BlockInterfaces, String>>,
    loops: OnceCell<LoopAnalysis>,
    slice: OnceCell<SliceFacts>,
}

impl<'g, P: Family> GraphAnalysis<'g, P> {
    pub fn new(graph: &'g EGraph<P>) -> Self {
        Self {
            graph,
            producers: OnceCell::new(),
            interfaces: OnceCell::new(),
            loops: OnceCell::new(),
            slice: OnceCell::new(),
        }
    }

    pub fn graph(&self) -> &'g EGraph<P> {
        self.graph
    }

    pub(crate) fn producers(&self) -> &SideEffectIndex {
        self.producers.get_or_init(|| self.graph.side_effect_index())
    }

    pub(crate) fn interfaces(&self) -> Result<&BlockInterfaces, String> {
        self.interfaces.get_or_init(|| block_interface::extract(self.graph)).as_ref().map_err(Clone::clone)
    }

    pub(crate) fn loops(&self) -> &LoopAnalysis {
        self.loops.get_or_init(|| LoopAnalysis::build(&self.graph.skeleton))
    }

    pub(crate) fn slice(&self) -> &SliceFacts
    where
        P: ValueProducerPhase,
    {
        self.slice.get_or_init(|| SliceFacts::build(self))
    }
}

#[cfg(test)]
#[path = "analysis_tests.rs"]
mod tests;
