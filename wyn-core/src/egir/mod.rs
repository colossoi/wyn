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

pub mod allocation;
pub mod elaborate;
mod extract;
mod fold;
pub(crate) mod inlining;
pub mod ir;
mod loop_analysis;
pub mod materialize;
pub mod partial_inline;
pub(crate) mod physical_call_abi;
pub(crate) mod physical_flow;
pub mod program;
pub mod publish;
pub mod resource_erasure;
pub mod rewrite;
mod scoped_map;
pub mod skel_opt;
pub mod soac;
pub mod soac_expand;
pub mod types;

pub mod builder;
pub mod from_tlc;
pub(crate) mod fusion;
pub mod graph_ops;
pub mod graph_projector;
pub mod parallelize;
pub(crate) mod pipeline_seed;
pub mod reify;
pub(crate) mod semantic_graph;
pub mod semantic_opt;
pub mod stage_lift;
pub(crate) mod structured_cfg;
// Keep the complete query surface available to later scheduling consumers;
// lifting and residency currently use only a subset of the recorded facts.
#[allow(dead_code)]
pub(crate) mod stage_variance;
pub mod verify_no_abstract;
pub(crate) mod verify_physical;

pub use allocation::{
    allocate_semantic_resources, finalize_staged_ir, plan_logical_resources, resolve_residency,
    resolve_scratch_sizes, verify_allocated_resources, ResidencyDraft, ResourcesAllocated,
};
pub use elaborate::elaborate;
pub use materialize::{materialize_dynamic_extracts, Materialized};
pub use parallelize::plan;
pub use partial_inline::{partially_inline_calls, PartiallyInlined};
pub use reify::reify_soacs;
pub use resource_erasure::{erase_resources, ResourcesErased};
pub use rewrite::rewrite;
pub use semantic_opt::{
    eliminate_dead_semantic_operations, fuse_semantic_operations, lift_stage_uniform_values,
    optimize_semantic_operations, optimize_semantic_operations_with_trace, Optimized,
    SemanticOperationsOptimized, SemanticOptimizationRelation, SemanticOptimizationTrace,
};
pub use skel_opt::{optimize_skeleton, SkeletonOptimized};
pub use soac_expand::{expand_soacs, SoacsExpanded};

#[cfg(test)]
pub(crate) mod semantic_exec;
