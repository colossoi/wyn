//! Target-independent optimization of semantic EGIR.

use std::collections::{HashMap, HashSet};

use super::program::EgirInner;
use super::types::{
    EGraph, EgirSoac, NodeId, SegResourceAccess, SegResourceAccessKind, SideEffectKind,
    SkeletonTerminator,
};

pub fn run(inner: &mut EgirInner) {
    for entry in &mut inner.entry_points {
        optimize_graph(&mut entry.graph);
    }
    for function in &mut inner.functions {
        optimize_graph(&mut function.graph);
    }
    super::parallelize::rebuild_semantic_dependencies(inner);
    let mut seen = HashSet::new();
    inner.semantic_dependencies.retain(|dependency| seen.insert(dependency.clone()));
}

fn optimize_graph(graph: &mut EGraph) {
    canonicalize_resource_accesses(graph);
    eliminate_dead_lane_local_ops(graph);
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
            let mut normalized: Vec<_> = merged
                .into_iter()
                .map(|(binding, access)| SegResourceAccess { binding, access })
                .collect();
            normalized.sort_by_key(|resource| (resource.binding.set, resource.binding.binding));
            *resources = normalized;
        }
    }
}

fn eliminate_dead_lane_local_ops(graph: &mut EGraph) {
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
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        block.side_effects.retain(|effect| {
            let SideEffectKind::Soac(EgirSoac::Seg {
                placement: super::types::SegPlacement::LaneLocal,
                resources,
                ..
            }) = &effect.kind
            else {
                return true;
            };
            let observable_resource = resources
                .iter()
                .any(|resource| resource.access != SegResourceAccessKind::Read);
            observable_resource || effect.result.is_some_and(|result| used.contains(&result))
        });
    }
}
