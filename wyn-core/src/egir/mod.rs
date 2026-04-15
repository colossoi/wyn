//! Acyclic e-graph (aegraph) mid-end optimizer for Wyn.
//!
//! The EGraph is a sea-of-nodes-with-CFG representation where:
//! - Pure operators float in a hash-consed acyclic graph (GVN for free)
//! - Side-effectful operators remain anchored in a CFG skeleton
//! - Scoped elaboration converts back to sequential SSA (DCE for free)
//! - Rewrite rules are applied eagerly during construction (Phase 2+)

mod domtree;
mod elaborate;
mod extract;
mod fold;
mod loop_analysis;
mod materialize;
pub mod pipeline;
pub mod rewrite;
mod scoped_map;
mod skel_opt;
mod soac_expand;
pub mod types;

pub mod from_tlc;

#[cfg(test)]
mod from_tlc_tests;
