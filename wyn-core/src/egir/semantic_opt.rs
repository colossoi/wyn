//! Target-independent optimization of semantic EGIR: resource-access
//! canonicalization, dead-SegOp elimination, indexed-demand scalarization, and
//! graph-rewriting fusion (same-space horizontal, producer/consumer, envelope,
//! and filter consumers). Every rewrite is gated by the semantic dependency DAG
//! so two ops are never fused or reordered across a conflicting resource or
//! effect.

/// Semantic EGIR after target-independent graph optimization.
#[derive(Debug, Clone, Copy)]
pub enum OptimizedTag {}
pub type Optimized = super::program::Program<
    OptimizedTag,
    super::ir::ProgramFamily<
        super::types::Semantic,
        super::program::SemanticResourceDecl,
        super::ir::RealizedOutputRoute,
        super::program::CoreProgramData,
    >,
    super::program::RewriteGlobal,
>;

use super::ir::BodySite;
use super::program::SemanticOpId;
use super::reify::Segmented;
use super::semantic_graph::SemanticGraph;
use super::soac::screma;
use super::types::{EGraph, ResourceAccess, SegResourceAccess, SideEffectKind, Soac, SoacEffect, ValueId};
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

pub fn optimize_semantics(program: Segmented) -> Optimized {
    optimize_semantics_with_trace(program).0
}

pub fn optimize_semantics_with_trace(program: Segmented) -> (Optimized, SemanticOptimizationTrace) {
    let mut trace = SemanticOptimizationTrace::default();
    let mut program = program.map_graphs(|_, mut graph| {
        canonicalize_resource_accesses(&mut graph);
        graph
    });

    // Fixpoint: rebuild the DAG, take one legal rewrite, repeat. Rebuilding
    // between rewrites keeps the legality oracle sound — a stale DAG is the
    // top correctness risk. Dead elimination runs first to shrink the graph.
    loop {
        let deps = super::semantic_graph::dependencies(&program);
        let oracle = SemanticGraph::new(&deps);

        if let Some(patch) = analyze_dead_seg_ops(&program) {
            let before = semantic_operation_fingerprints(&program);
            program = apply_dead_seg_ops(program, patch);
            trace.record(before, semantic_operation_fingerprints(&program));
            continue;
        }
        let before = semantic_operation_fingerprints(&program);
        let (rewritten, changed) = super::fusion::rewrite_once(program, &oracle);
        program = rewritten;
        if changed {
            trace.record(before, semantic_operation_fingerprints(&program));
            continue;
        }
        break;
    }

    program = super::stage_lift::lift_stage_uniform_values(program)
        .expect("stage-uniform region lifting must preserve semantic EGIR");

    if cfg!(debug_assertions) {
        if let Err(error) = super::semantic_graph::verify(&program) {
            panic!("semantic optimization produced invalid EGIR: {error}");
        }
    }
    (program.retag(), trace)
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

fn canonicalize_resource_accesses(graph: &mut EGraph) {
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &mut effect.kind else {
                continue;
            };
            if let screma::SemanticState::Segmented { resources, .. } = op.semantic_state_mut() {
                *resources = SegResourceAccess::merge(resources, &[]);
            }
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

fn dead_seg_ops_in_graph(
    graph: &EGraph,
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

pub(super) fn eliminate_dead_seg_ops_in_graph(
    graph: &mut EGraph,
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
