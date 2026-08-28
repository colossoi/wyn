//! Target-independent optimization of semantic EGIR: dead-SegOp elimination,
//! indexed-demand scalarization, and graph-rewriting fusion (same-space
//! horizontal, producer/consumer, envelope, and filter consumers). Every
//! rewrite is gated by the semantic dependency DAG so two ops are never fused
//! or reordered across a conflicting resource or effect.

/// Semantic EGIR after dead-operation elimination and fusion reach a fixpoint.
#[derive(Debug, Clone, Copy)]
pub enum SemanticOperationsOptimizedTag {}
pub type SemanticOperationsOptimized = super::program::Program<
    SemanticOperationsOptimizedTag,
    super::ir::ProgramFamily<
        super::types::Semantic,
        super::program::NoStorageDeclaration,
        super::ir::RealizedOutputRoute,
        super::program::SemanticProgramData,
    >,
    super::program::RewriteGlobal,
>;

/// Semantic EGIR after target-independent graph optimization and stage lifting.
#[derive(Debug, Clone, Copy)]
pub enum OptimizedTag {}
pub type Optimized = super::program::Program<
    OptimizedTag,
    super::ir::ProgramFamily<
        super::types::Semantic,
        super::program::NoStorageDeclaration,
        super::ir::RealizedOutputRoute,
        super::program::SemanticProgramData,
    >,
    super::program::RewriteGlobal,
>;

use super::ir::BodySite;
use super::program::SemanticOpId;
use super::reify::Segmented;
use super::semantic_graph::SemanticGraph;
use super::soac::screma;
use super::types::{
    EGraph, GraphResource, ResourceAccess, Semantic, SideEffectKind, Soac, SoacEffect, ValueId,
};
use crate::flow::BlockId;
use crate::LookupMap;
use std::collections::BTreeMap;

#[cfg(test)]
#[path = "semantic_opt_tests.rs"]
mod semantic_opt_tests;

/// One structural relationship observed while semantic optimization rewrote a
/// single operation or eliminated dead work. A fusion typically has multiple
/// `before` identities and one `after` identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticOptimizationRelation {
    pub before: Vec<SemanticOpId>,
    pub after: Vec<SemanticOpId>,
}

/// Compiler-authored provenance for consumers that need to relate semantic
/// operations across the optimization boundary.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SemanticOptimizationTrace {
    pub relations: Vec<SemanticOptimizationRelation>,
}

/// Eliminate dead segmented operations and fuse legal operations to a single
/// shared fixpoint.
pub fn optimize_semantic_operations(program: Segmented) -> SemanticOperationsOptimized {
    optimize_semantic_operations_with_trace(program).0
}

/// The fixpoint transition with compiler-authored rewrite provenance.
pub fn optimize_semantic_operations_with_trace(
    program: Segmented,
) -> (SemanticOperationsOptimized, SemanticOptimizationTrace) {
    let mut trace = SemanticOptimizationTrace::default();
    let mut program = program;

    // Fixpoint: rebuild the DAG, take one legal rewrite, repeat. Rebuilding
    // between rewrites keeps the legality oracle sound — a stale DAG is the
    // top correctness risk. Dead elimination runs first to shrink the graph.
    loop {
        let (rewritten, changed, step_trace) = eliminate_dead_semantic_operations(program);
        program = rewritten;
        trace.extend(step_trace);
        if changed {
            continue;
        }

        let (rewritten, changed, step_trace) = fuse_semantic_operations(program);
        program = rewritten;
        trace.extend(step_trace);
        if changed {
            continue;
        }
        break;
    }

    (program.retag(), trace)
}

/// Apply one whole-program dead semantic-operation elimination step.
///
/// The returned boolean reports whether the program changed. Callers driving
/// the production fixpoint must rebuild all semantic analyses before invoking
/// another semantic optimization sub-pass.
pub fn eliminate_dead_semantic_operations(
    program: Segmented,
) -> (Segmented, bool, SemanticOptimizationTrace) {
    let Some(patch) = analyze_dead_seg_ops(&program) else {
        return (program, false, SemanticOptimizationTrace::default());
    };
    let before = semantic_operation_fingerprints(&program);
    let program = apply_dead_seg_ops(program, patch);
    let mut trace = SemanticOptimizationTrace::default();
    trace.record(before, semantic_operation_fingerprints(&program));
    (program, true, trace)
}

/// Apply at most one legal semantic fusion rewrite.
///
/// Candidate analysis and its dependency oracle are intentionally rebuilt for
/// every call. This keeps the public sub-pass boundary safe for inspection
/// clients while preserving the production optimizer's legality invariant.
pub fn fuse_semantic_operations(program: Segmented) -> (Segmented, bool, SemanticOptimizationTrace) {
    let dependencies = super::semantic_graph::dependencies(&program);
    let oracle = SemanticGraph::new(&dependencies);
    let before = semantic_operation_fingerprints(&program);
    let (program, changed) = super::fusion::rewrite_once(program, &oracle);
    let mut trace = SemanticOptimizationTrace::default();
    if changed {
        trace.record(before, semantic_operation_fingerprints(&program));
    }
    (program, changed, trace)
}

/// Lift values that are uniform at their execution stage, then validate the
/// final semantic dependency graph in debug builds.
pub fn lift_stage_uniform_values(program: SemanticOperationsOptimized) -> Optimized {
    let program: Segmented = program.retag();
    let program = super::stage_lift::lift_stage_uniform_values(program)
        .expect("stage-uniform region lifting must preserve semantic EGIR");

    if cfg!(debug_assertions) {
        if let Err(error) = super::semantic_graph::verify(&program) {
            panic!("semantic optimization produced invalid EGIR: {error}");
        }
    }
    program.retag()
}

fn semantic_operation_fingerprints(program: &Segmented) -> BTreeMap<SemanticOpId, String> {
    program
        .entry_points
        .iter()
        .map(|entry| &entry.graph)
        .chain(program.functions.iter().map(|function| &function.graph))
        .chain(program.constants.iter().map(|constant| &constant.graph))
        .flat_map(|graph| graph.skeleton.blocks.iter().flat_map(|(_, block)| block.side_effects.iter()))
        .filter_map(|effect| {
            let SideEffectKind::Soac(SoacEffect(id, soac)) = effect.kind() else {
                return None;
            };
            Some((*id, format!("{soac:#?}")))
        })
        .collect()
}

impl SemanticOptimizationTrace {
    fn extend(&mut self, mut other: Self) {
        self.relations.append(&mut other.relations);
    }

    fn record(&mut self, before: BTreeMap<SemanticOpId, String>, after: BTreeMap<SemanticOpId, String>) {
        let removed = before.keys().filter(|id| !after.contains_key(id)).copied();
        let added = after.keys().filter(|id| !before.contains_key(id)).copied();
        let changed = before.iter().filter_map(|(id, fingerprint)| {
            after.get(id).filter(|after| *after != fingerprint).map(|_| *id)
        });

        let mut before_ids = removed.chain(changed.clone()).collect::<Vec<_>>();
        let mut after_ids = added.chain(changed).collect::<Vec<_>>();
        before_ids.sort_unstable();
        before_ids.dedup();
        after_ids.sort_unstable();
        after_ids.dedup();

        if !before_ids.is_empty() || !after_ids.is_empty() {
            self.relations.push(SemanticOptimizationRelation {
                before: before_ids,
                after: after_ids,
            });
        }
    }
}

/// Remove SegOps (of any placement) that write no observable resource and whose
/// result is unused. The outer fixpoint re-runs it so producer chains collapse.
type DeadGraphPatch = LookupMap<BlockId, Vec<usize>>;

struct DeadSegOpsPatch {
    bodies: LookupMap<BodySite, DeadGraphPatch>,
}

fn analyze_dead_seg_ops(inner: &Segmented) -> Option<DeadSegOpsPatch> {
    let mut bodies = LookupMap::new();
    for (index, entry) in inner.entry_points.iter().enumerate() {
        let patch = dead_seg_ops_in_graph(
            &entry.graph,
            entry.routes().flat_map(|route| route.referenced_values()),
        );
        if !patch.is_empty() {
            bodies.insert(BodySite::Entry(index), patch);
        }
    }
    for function in &inner.functions {
        let patch = dead_seg_ops_in_graph(&function.graph, []);
        if !patch.is_empty() {
            bodies.insert(BodySite::Function(function.region), patch);
        }
    }
    (!bodies.is_empty()).then_some(DeadSegOpsPatch { bodies })
}

fn apply_dead_seg_ops(inner: Segmented, mut patch: DeadSegOpsPatch) -> Segmented {
    let rebuilt = inner.map_graphs(|site, mut graph| {
        let Some(blocks) = patch.bodies.remove(&site) else {
            return graph;
        };
        for (block, mut effects) in blocks {
            effects.sort_unstable();
            for effect in effects.into_iter().rev() {
                graph.skeleton.blocks[block].side_effects.remove(effect);
            }
        }
        graph
    });
    assert!(
        patch.bodies.is_empty(),
        "dead-SegOp patches targeted bodies absent from the rebuilt program"
    );
    rebuilt
}

fn dead_seg_ops_in_graph<R: GraphResource>(
    graph: &EGraph<Semantic<R>>,
    external_roots: impl IntoIterator<Item = ValueId>,
) -> DeadGraphPatch {
    // Live values are those reachable from an observable root.  Looking at
    // children of every interned node is too conservative: dead Project nodes
    // remain in an e-graph and would otherwise keep their producer alive.
    let mut roots = external_roots.into_iter().collect::<Vec<_>>();
    for (_, block) in &graph.skeleton.blocks {
        for effect in &block.side_effects {
            roots.extend(super::graph_ops::effect_value_inputs(graph, effect));
        }
        roots.extend(block.term.referenced_nodes());
    }

    let used = wyn_graph::reachable_set(roots, wyn_graph::WalkOrder::DepthFirst, |node, out| {
        if let Some(definition) = graph.nodes.get(node) {
            out.extend(definition.kind.children());
        }
    });
    let mut patch = LookupMap::new();
    for (block_id, block) in &graph.skeleton.blocks {
        let dead = block
            .side_effects
            .iter()
            .enumerate()
            .filter_map(|(index, effect)| {
                let SideEffectKind::Soac(SoacEffect(_, soac)) = &effect.kind else {
                    return None;
                };
                // A Seg with no resource write and no output routing is observable
                // only through its result. Filter/Hist/Screma may write in ways not
                // summarized here, so keep them conservatively.
                let observable = match soac {
                    Soac::Screma(op) => match op.semantic_state() {
                        screma::SemanticState::Segmented {
                            resources,
                            output_slots,
                            ..
                        } => {
                            !output_slots.is_empty()
                                || resources.iter().any(|r| r.access != ResourceAccess::Read)
                        }
                        screma::SemanticState::Serial => true,
                    },
                    _ => true,
                };
                (!observable
                    && effect
                        .result
                        .as_ref()
                        .is_none_or(|result| result.values().iter().all(|value| !used.contains(value))))
                .then_some(index)
            })
            .collect::<Vec<_>>();
        if !dead.is_empty() {
            patch.insert(block_id, dead);
        }
    }
    patch
}

pub(super) fn eliminate_dead_seg_ops_in_graph<R: GraphResource>(
    graph: &mut EGraph<Semantic<R>>,
    external_roots: impl IntoIterator<Item = ValueId>,
) -> bool {
    let mut patch = dead_seg_ops_in_graph(graph, external_roots);
    let changed = !patch.is_empty();
    for (block, mut effects) in patch.drain() {
        effects.sort_unstable();
        for effect in effects.into_iter().rev() {
            graph.skeleton.blocks[block].side_effects.remove(effect);
        }
    }
    changed
}
