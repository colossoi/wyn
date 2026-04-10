//! Rewrite rule infrastructure for the aegraph.
//!
//! Phase 1: trait definition only, no rules.
//! Phase 2 will add ConstantFold, IdentityElim, ProjectFold, etc.

use super::types::{EGraph, NodeId};

/// Result of attempting a rewrite on a node.
pub enum RewriteResult {
    /// No rewrite applies.
    NoMatch,
    /// The rewrite produces a strictly better node. The original is discarded
    /// (no union node created). Use for clear wins like `x + 0 → x`.
    Subsume(NodeId),
    /// The rewrite produces an alternative. A union node is created joining
    /// the original and the replacement, and cost-based extraction picks the best.
    Replace(NodeId),
}

/// A rewrite rule applied eagerly during canonicalization.
pub trait RewriteRule {
    /// Try to rewrite a node. Called immediately after the node is interned.
    /// The graph is mutable so rules can intern new nodes as part of the RHS.
    fn try_rewrite(&self, graph: &mut EGraph, node: NodeId) -> RewriteResult;
}

/// A collection of rewrite rules.
pub struct RewriteSet {
    rules: Vec<Box<dyn RewriteRule>>,
}

impl RewriteSet {
    pub fn new() -> Self {
        RewriteSet { rules: Vec::new() }
    }

    pub fn add(&mut self, rule: Box<dyn RewriteRule>) {
        self.rules.push(rule);
    }

    /// Apply all rules to a node. Returns the final NodeId (possibly a union
    /// tree if Replace results occurred, or a subsumption).
    pub fn apply_all(&self, graph: &mut EGraph, mut current: NodeId) -> NodeId {
        for rule in &self.rules {
            match rule.try_rewrite(graph, current) {
                RewriteResult::NoMatch => continue,
                RewriteResult::Subsume(better) => return better,
                RewriteResult::Replace(alt) => {
                    current = graph.add_union(current, alt);
                }
            }
        }
        current
    }
}

/// The default (empty) rewrite set for Phase 1.
pub fn default_rewrites() -> RewriteSet {
    RewriteSet::new()
}
