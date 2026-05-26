//! Acyclic e-graph (aegraph) mid-end optimizer for Wyn.
//!
//! The EGraph is a sea-of-nodes-with-CFG representation where:
//! - Pure operators float in a hash-consed acyclic graph (GVN for free)
//! - Side-effectful operators remain anchored in a CFG skeleton
//! - Scoped elaboration converts back to sequential SSA (DCE for free)
//! - Rewrite rules are applied eagerly during construction (Phase 2+)

pub(crate) mod domtree;
pub(crate) mod elaborate;
mod extract;
mod fold;
mod loop_analysis;
pub(crate) mod materialize;
pub mod program;
pub mod rewrite;
mod scoped_map;
pub(crate) mod skel_opt;
pub(crate) mod soac_expand;
pub mod types;

pub mod assign_outputs;
pub mod builder;
pub mod from_tlc;
pub mod graph_ops;
pub mod parallelize;
