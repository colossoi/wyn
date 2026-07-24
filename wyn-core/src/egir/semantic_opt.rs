//! Target-independent optimization of semantic EGIR: resource-access
//! canonicalization, dead-SegOp elimination, indexed-demand scalarization, and
//! graph-rewriting fusion (same-space horizontal, producer/consumer, envelope,
//! and filter consumers). Every rewrite is gated by the semantic dependency DAG
//! so two ops are never fused or reordered across a conflicting resource or
//! effect.

/// Semantic EGIR after target-independent graph optimization.
#[derive(Debug, Clone, Copy, Default)]
pub struct Optimized;

impl super::ir::Stage for Optimized {
    type Family = super::types::Semantic;
    type ResourceDecl = super::program::SemanticResourceDecl;
    type OutputRoute = super::ir::RealizedOutputRoute;
    type ProgramData = super::program::CoreProgramData;
    type GlobalContext = super::program::RewriteGlobal;
}

use super::ir::BodySite;
use super::program::Program;
use super::reify::Segmented;
use super::semantic_graph::SemanticGraph;
use super::soac::screma;
use super::types::{EGraph, NodeId, ResourceAccess, SegResourceAccess, SideEffectKind, Soac, SoacEffect};
use crate::flow::BlockId;
use crate::LookupMap;

#[cfg(test)]
#[path = "semantic_opt_tests.rs"]
mod semantic_opt_tests;

pub fn run(program: Program<Segmented>) -> Program<Optimized> {
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
            program = apply_dead_seg_ops(program, patch);
            continue;
        }
        if let Some(patch) = super::fusion::analyze(&program, &oracle) {
            program = super::fusion::apply(program, patch);
            continue;
        }
        break;
    }

    program =
        super::stage_lift::run(program).expect("stage-uniform region lifting must preserve semantic EGIR");

    if cfg!(debug_assertions) {
        if let Err(error) = super::semantic_graph::verify(&program) {
            panic!("semantic optimization produced invalid EGIR: {error}");
        }
    }
    program.into_stage()
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

fn analyze_dead_seg_ops(inner: &Program<Segmented>) -> Option<DeadSegOpsPatch> {
    let mut bodies = LookupMap::new();
    for (index, entry) in inner.entry_points.iter().enumerate() {
        let patch = dead_seg_ops_in_graph(&entry.graph);
        if !patch.is_empty() {
            bodies.insert(BodySite::Entry(index), patch);
        }
    }
    for function in &inner.functions {
        let patch = dead_seg_ops_in_graph(&function.graph);
        if !patch.is_empty() {
            bodies.insert(BodySite::Function(function.region), patch);
        }
    }
    (!bodies.is_empty()).then_some(DeadSegOpsPatch { bodies })
}

fn apply_dead_seg_ops(inner: Program<Segmented>, mut patch: DeadSegOpsPatch) -> Program<Segmented> {
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

fn dead_seg_ops_in_graph(graph: &EGraph) -> DeadGraphPatch {
    // Live values are those reachable from an observable root.  Looking at
    // children of every interned node is too conservative: dead Project nodes
    // remain in an e-graph and would otherwise keep their producer alive.
    let mut roots = Vec::<NodeId>::new();
    for (_, block) in &graph.skeleton.blocks {
        for effect in &block.side_effects {
            roots.extend(effect.referenced_nodes());
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
                (!observable && effect.result.is_none_or(|result| !used.contains(&result))).then_some(index)
            })
            .collect::<Vec<_>>();
        if !dead.is_empty() {
            patch.insert(block_id, dead);
        }
    }
    patch
}

pub(super) fn eliminate_dead_seg_ops_in_graph(graph: &mut EGraph) -> bool {
    let mut patch = dead_seg_ops_in_graph(graph);
    let changed = !patch.is_empty();
    for (block, mut effects) in patch.drain() {
        effects.sort_unstable();
        for effect in effects.into_iter().rev() {
            graph.skeleton.blocks[block].side_effects.remove(effect);
        }
    }
    changed
}
