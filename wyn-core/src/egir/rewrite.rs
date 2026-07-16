//! Rewrite rules over the pure sea, arbitrated by cost extraction.
//!
//! A rule proposes an alternative form for a single pure node. A `Replace`
//! result does not commit: the node becomes a `Union` of both forms in
//! place, and `extract` picks the cheaper side during elaboration, so the
//! crossover point lives in `extract::op_cost`, not in the rule.
//!
//! Rules run once, over the physical program (`run`, called from
//! `EgirPlanned::lower_to_ssa`): after fusion and planning — whose subgraph
//! clones reject unions — and right before the only consumers that resolve
//! them (`skel_opt` scopes unions out; `elaborate` extracts).

use polytype::Type;
use smallvec::smallvec;

use crate::ast::TypeName;

use super::program::PhysicalProgram;
use super::types::{EGraph, ENode, EgirPhase, NodeId, PureOp};

/// Result of attempting a rewrite on a node.
pub enum RewriteResult {
    /// No rewrite applies.
    NoMatch,
    /// The rewrite produces a strictly better node. The original is
    /// discarded; extraction has no choice to make. Use for clear wins
    /// like `x + 0 → x`.
    Subsume(NodeId),
    /// The rewrite produces an alternative. The original and the
    /// replacement are joined under a union and cost-based extraction
    /// picks the best.
    Replace(NodeId),
}

/// A rewrite rule applied to each pure node of the graph.
pub trait RewriteRule<P: EgirPhase> {
    /// Try to rewrite `node`. The graph is mutable so rules can intern new
    /// nodes for the RHS. The returned node must not contain `node` in its
    /// operand cone.
    fn try_rewrite(&self, graph: &mut EGraph<P>, node: NodeId) -> RewriteResult;
}

/// A collection of rewrite rules.
pub struct RewriteSet<P: EgirPhase> {
    rules: Vec<Box<dyn RewriteRule<P>>>,
}

impl<P: EgirPhase> RewriteSet<P> {
    pub fn new() -> Self {
        RewriteSet { rules: Vec::new() }
    }

    pub fn add(&mut self, rule: Box<dyn RewriteRule<P>>) {
        self.rules.push(rule);
    }

    /// Apply every rule to every pure node currently in the graph. Nodes
    /// interned by the rules themselves are not revisited. Returns true if
    /// any rewrite fired.
    pub fn apply_to_graph(&self, graph: &mut EGraph<P>) -> bool {
        let ids: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(node, ENode::Pure { .. }))
            .map(|(id, _)| id)
            .collect();
        let mut changed = false;
        for id in ids {
            changed |= self.apply_to_node(graph, id);
        }
        changed
    }

    /// Apply rules to one pure node, in place: references to `node` keep
    /// their id and see the rewrite through it. The first rule that fires
    /// wins (the node is no longer pure afterwards). Returns true if a
    /// rewrite fired.
    pub fn apply_to_node(&self, graph: &mut EGraph<P>, node: NodeId) -> bool {
        for rule in &self.rules {
            match rule.try_rewrite(graph, node) {
                RewriteResult::NoMatch => continue,
                RewriteResult::Subsume(better) => {
                    graph.subsume_pure_in_place(node, better);
                    return true;
                }
                RewriteResult::Replace(alt) => {
                    graph.union_pure_in_place(node, alt);
                    return true;
                }
            }
        }
        false
    }
}

/// Apply the default rewrite set to every function and entry graph of the
/// physical program. Runs between materialization and skeleton
/// optimization, so elaboration's cost extraction is what resolves the
/// unions it creates.
pub fn run(program: &mut PhysicalProgram) {
    let rules = default_rewrites();
    for f in &mut program.functions {
        rules.apply_to_graph(&mut f.graph);
    }
    for e in &mut program.entry_points {
        rules.apply_to_graph(&mut e.graph);
    }
}

/// Every rule whose outcome is arbitrated by cost extraction.
pub fn default_rewrites<P: EgirPhase>() -> RewriteSet<P> {
    let mut set = RewriteSet::new();
    set.add(Box::new(PowToMulChain));
    set
}

/// Longest multiply chain worth proposing; a longer chain loses to the
/// modeled `**` cost under any plausible tuning, so interning it would be
/// pure waste.
const MAX_CHAIN: u32 = 16;

/// `x ** k` (integral constant `k ≥ 2`) proposes a left-to-right multiply
/// chain as an alternative to the backend's `**` lowering (GLSL.std.450
/// `Pow` for floats, an exponentiation-by-squaring helper for ints). The
/// chain is exact and has no `x ≤ 0` edge cases; extraction weighs its
/// `k - 1` multiplies against `extract`'s modeled `Pow` cost.
///
/// The exponent literal can be any numeric flavor (`i32` / `u32` / `f32`);
/// float `**` requires same-typed operands, so `x ** 3` arrives with the
/// exponent as `f32`, and only whole floats qualify. Non-integral or
/// out-of-range exponents (e.g. `2.5`, `-1`, `0`) leave the node alone.
///
/// Vector bases are skipped: there is no correct scalar rewrite for
/// `vec ** k` — a componentwise `v * v` looks like squared-magnitude but
/// isn't. With no alternative here and no backend lowering for `**` on a
/// vec, the call falls through to a clean compile-time error so users
/// write the intended `dot(v, v)` / `magnitude(v) ** k` form.
pub struct PowToMulChain;

impl<P: EgirPhase> RewriteRule<P> for PowToMulChain {
    fn try_rewrite(&self, graph: &mut EGraph<P>, node: NodeId) -> RewriteResult {
        let ENode::Pure {
            op: PureOp::BinOp(name),
            operands,
        } = &graph.nodes[node]
        else {
            return RewriteResult::NoMatch;
        };
        if name != "**" || operands.len() != 2 {
            return RewriteResult::NoMatch;
        }
        let (base, exp) = (operands[0], operands[1]);
        let result_ty = graph.types[&node].clone();
        if matches!(result_ty, Type::Constructed(TypeName::Vec, _)) {
            return RewriteResult::NoMatch;
        }
        let k: u32 = if let Some(v) = graph.as_i32(exp) {
            match u32::try_from(v) {
                Ok(v) => v,
                Err(_) => return RewriteResult::NoMatch,
            }
        } else if let Some(v) = graph.as_u32(exp) {
            v
        } else if let Some(f) = graph.as_f32(exp) {
            if f.fract() != 0.0 || !(2.0..=MAX_CHAIN as f32).contains(&f) {
                return RewriteResult::NoMatch;
            }
            f as u32
        } else {
            return RewriteResult::NoMatch;
        };
        if !(2..=MAX_CHAIN).contains(&k) {
            return RewriteResult::NoMatch;
        }
        let mut chain = base;
        for _ in 1..k {
            chain = graph.intern_pure(
                PureOp::BinOp("*".into()),
                smallvec![chain, base],
                result_ty.clone(),
                None,
            );
        }
        RewriteResult::Replace(chain)
    }
}

#[cfg(test)]
#[path = "rewrite_tests.rs"]
mod rewrite_tests;
