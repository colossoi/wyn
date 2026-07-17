//! Target-independent optimization of semantic EGIR: resource-access
//! canonicalization, dead-SegOp elimination, indexed-demand scalarization, and
//! graph-rewriting fusion (same-space horizontal, producer/consumer, envelope,
//! and filter consumers). Every rewrite is gated by the semantic dependency DAG
//! so two ops are never fused or reordered across a conflicting resource or
//! effect.

use super::program::SemanticProgram;
use super::semantic_graph::SemanticGraph;
use super::soac::screma;
use super::types::{EGraph, NodeId, ResourceAccess, SegResourceAccess, SideEffectKind, Soac, SoacEffect};

#[cfg(test)]
#[path = "semantic_opt_tests.rs"]
mod semantic_opt_tests;

pub fn run(inner: &mut SemanticProgram) {
    for entry in &mut inner.entry_points {
        canonicalize_resource_accesses(&mut entry.graph);
    }
    for function in &mut inner.functions {
        canonicalize_resource_accesses(&mut function.graph);
    }

    // Fixpoint: rebuild the DAG, take one legal rewrite, repeat. Rebuilding
    // between rewrites keeps the legality oracle sound — a stale DAG is the
    // top correctness risk. Dead elimination runs first to shrink the graph.
    loop {
        super::semantic_graph::rebuild_dependencies(inner);
        let deps = inner.semantic_dependencies.clone();
        let oracle = SemanticGraph::new(&deps);

        let mut changed = eliminate_dead_seg_ops(inner);
        if !changed {
            changed = super::fusion::indexed::scalarize_indexed_segmap(inner);
        }
        if !changed {
            changed = super::fusion::vertical::fuse_producer_into_consumer(inner, &oracle);
        }
        if !changed {
            changed = super::fusion::envelope::fuse_producer_into_envelope(inner, &oracle);
        }
        if !changed {
            changed = super::fusion::filter::fuse_filter_consumers(inner, &oracle);
        }
        if !changed {
            changed = super::fusion::horizontal::fuse_sibling_seg_ops(inner, &oracle);
        }
        if !changed {
            break;
        }
    }

    super::semantic_graph::rebuild_dependencies(inner);
    if cfg!(debug_assertions) {
        if let Err(error) = super::semantic_graph::verify(inner) {
            panic!("semantic optimization produced invalid EGIR: {error}");
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
/// result is unused. Generalizes the former lane-local-only elimination; the
/// outer fixpoint re-runs it so producer chains collapse. Returns whether any
/// graph changed.
fn eliminate_dead_seg_ops(inner: &mut SemanticProgram) -> bool {
    let mut changed = false;
    for entry in &mut inner.entry_points {
        changed |= eliminate_dead_seg_ops_in_graph(&mut entry.graph);
    }
    for function in &mut inner.functions {
        changed |= eliminate_dead_seg_ops_in_graph(&mut function.graph);
    }
    changed
}

pub(super) fn eliminate_dead_seg_ops_in_graph(graph: &mut EGraph) -> bool {
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
            out.extend(definition.children());
        }
    });
    let mut changed = false;
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        let before = block.side_effects.len();
        block.side_effects.retain(|effect| {
            let SideEffectKind::Soac(SoacEffect(_, soac)) = &effect.kind else {
                return true;
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
            observable || effect.result.is_some_and(|result| used.contains(&result))
        });
        changed |= block.side_effects.len() != before;
    }
    changed
}
