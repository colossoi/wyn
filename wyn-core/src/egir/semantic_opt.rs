//! Target-independent optimization of semantic EGIR: resource-access
//! canonicalization, dead-SegOp elimination, and graph-rewriting fusion
//! (same-space horizontal, single-consumer producer/consumer). Every rewrite is
//! gated by the semantic dependency DAG so two ops are never fused or reordered
//! across a conflicting resource or effect.

use std::collections::{HashMap, HashSet};

use super::fusion::legality::SemanticGraph;
use super::program::EgirInner;
use super::types::{
    EGraph, EgirSoac, NodeId, SegResourceAccess, SegResourceAccessKind, SideEffectKind, SkeletonTerminator,
};

pub fn run(inner: &mut EgirInner) {
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
        super::parallelize::rebuild_semantic_dependencies(inner);
        let deps = inner.semantic_dependencies.clone();
        let oracle = SemanticGraph::new(&deps);

        let mut changed = eliminate_dead_seg_ops(inner);
        if !changed {
            changed = super::fusion::horizontal::fuse_sibling_seg_ops(inner, &oracle);
        }
        if !changed {
            break;
        }
    }

    super::parallelize::rebuild_semantic_dependencies(inner);
    let mut seen = HashSet::new();
    inner.semantic_dependencies.retain(|dependency| seen.insert(dependency.clone()));
}

fn canonicalize_resource_accesses(graph: &mut EGraph) {
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            let SideEffectKind::Soac(EgirSoac::Seg { resources, .. }) = &mut effect.kind else {
                continue;
            };
            let mut merged = HashMap::new();
            for resource in resources.drain(..) {
                merged
                    .entry(resource.binding)
                    .and_modify(|access| {
                        if *access != resource.access {
                            *access = SegResourceAccessKind::ReadWrite;
                        }
                    })
                    .or_insert(resource.access);
            }
            let mut normalized: Vec<_> =
                merged.into_iter().map(|(binding, access)| SegResourceAccess { binding, access }).collect();
            normalized.sort_by_key(|resource| (resource.binding.set, resource.binding.binding));
            *resources = normalized;
        }
    }
}

/// Remove SegOps (of any placement) that write no observable resource and whose
/// result is unused. Generalizes the former lane-local-only elimination; the
/// outer fixpoint re-runs it so producer chains collapse. Returns whether any
/// graph changed.
fn eliminate_dead_seg_ops(inner: &mut EgirInner) -> bool {
    let mut changed = false;
    for entry in &mut inner.entry_points {
        changed |= eliminate_dead_seg_ops_in_graph(&mut entry.graph);
    }
    for function in &mut inner.functions {
        changed |= eliminate_dead_seg_ops_in_graph(&mut function.graph);
    }
    changed
}

fn eliminate_dead_seg_ops_in_graph(graph: &mut EGraph) -> bool {
    let mut used = HashSet::<NodeId>::new();
    for (_, node) in &graph.nodes {
        used.extend(node.children());
    }
    for (_, block) in &graph.skeleton.blocks {
        for effect in &block.side_effects {
            used.extend(effect.referenced_nodes());
        }
        match &block.term {
            SkeletonTerminator::Return(value) => used.extend(value.iter().copied()),
            SkeletonTerminator::Branch { args, .. } => used.extend(args.iter().copied()),
            SkeletonTerminator::CondBranch {
                cond,
                then_args,
                else_args,
                ..
            } => {
                used.insert(*cond);
                used.extend(then_args.iter().copied());
                used.extend(else_args.iter().copied());
            }
            SkeletonTerminator::Unreachable => {}
        }
    }
    let mut changed = false;
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        let before = block.side_effects.len();
        block.side_effects.retain(|effect| {
            let SideEffectKind::Soac(soac) = &effect.kind else {
                return true;
            };
            // A Seg with no resource write and no output routing is observable
            // only through its result. Filter/Hist/Screma may write in ways not
            // summarized here, so keep them conservatively.
            let observable = match soac {
                EgirSoac::Seg {
                    placement: _,
                    resources,
                    output_slots,
                    ..
                } => {
                    !output_slots.is_empty()
                        || resources.iter().any(|r| r.access != SegResourceAccessKind::Read)
                }
                _ => true,
            };
            observable || effect.result.is_some_and(|result| used.contains(&result))
        });
        changed |= block.side_effects.len() != before;
    }
    changed
}
