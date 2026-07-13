//! Acyclic e-graph (aegraph) mid-end optimizer for Wyn.
//!
//! The EGraph is a sea-of-nodes-with-CFG representation where:
//! - Pure operators float in a hash-consed acyclic graph (GVN for free)
//! - Side-effectful operators remain anchored in a CFG skeleton
//! - Scoped elaboration converts back to sequential SSA (DCE for free)
//! - Rewrite rules are applied eagerly during construction (Phase 2+)
//!
//! Shape inspired by Chris Fallin's aegraph writeup
//! (acyclic, GVN'd, side-effect skeleton, scoped elaboration as
//! extraction): <https://cfallin.org/blog/2026/04/09/aegraph/>.

pub(crate) mod elaborate;
mod extract;
mod fold;
mod loop_analysis;
pub(crate) mod materialize;
pub(crate) mod multi_consumer;
pub mod program;
pub mod publish;
pub(crate) mod resource_erasure;
pub mod rewrite;
mod scoped_map;
pub(crate) mod skel_opt;
pub(crate) mod soac_expand;
pub mod types;

pub mod builder;
pub mod from_tlc;
pub(crate) mod fusion;
pub mod graph_ops;
pub mod graph_projector;
pub mod parallelize;
pub mod realize_outputs;
pub(crate) mod semantic_graph;
pub(crate) mod semantic_opt;
pub(crate) mod target_lowering;
pub mod verify_no_abstract;
pub(crate) mod verify_physical;

#[cfg(test)]
pub(crate) mod semantic_exec;
