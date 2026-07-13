//! Target-independent optimization of semantic EGIR: resource-access
//! canonicalization, dead-SegOp elimination, and graph-rewriting fusion
//! (same-space horizontal, single-consumer producer/consumer). Every rewrite is
//! gated by the semantic dependency DAG so two ops are never fused or reordered
//! across a conflicting resource or effect.

use std::collections::HashMap;

use super::program::SemanticProgram;
use super::semantic_graph::SemanticGraph;
use super::types::{
    EGraph, EgirSoac, NodeId, SegResourceAccess, SegResourceAccessKind, SideEffectKind, SkeletonTerminator,
};

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
            changed = super::fusion::horizontal::fuse_sibling_seg_ops(inner, &oracle);
        }
        if !changed {
            changed = super::fusion::vertical::fuse_producer_into_consumer(inner, &oracle);
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
        match &block.term {
            SkeletonTerminator::Return(value) => roots.extend(value.iter().copied()),
            SkeletonTerminator::Branch { args, .. } => roots.extend(args.iter().copied()),
            SkeletonTerminator::CondBranch {
                cond,
                then_args,
                else_args,
                ..
            } => {
                roots.push(*cond);
                roots.extend(then_args.iter().copied());
                roots.extend(else_args.iter().copied());
            }
            SkeletonTerminator::Unreachable => {}
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TypeName;
    use crate::egir::types::{
        EgirSoac, PureOp, SegLevel, SegOpKind, SegPlacement, SegSpace, SideEffect, SoacDestination,
    };
    use polytype::Type;
    use smallvec::smallvec;

    #[test]
    fn unreachable_project_does_not_keep_dead_segop_alive() {
        let mut graph = EGraph::new();
        let int = Type::Constructed(TypeName::Int(32), vec![]);
        let tuple = Type::Constructed(TypeName::Tuple(1), vec![int.clone()]);
        let result = graph.alloc_side_effect_result(tuple);
        let _dead_project = graph.intern_pure(PureOp::Project { index: 0 }, smallvec![result], int.clone());
        graph.skeleton.blocks[graph.skeleton.entry].side_effects.push(SideEffect {
            semantic_id: None,
            kind: SideEffectKind::Soac(EgirSoac::Seg {
                space: SegSpace {
                    level: SegLevel::Thread,
                    dims: vec![crate::egir::types::SegExtent::Fixed(1)],
                },
                placement: SegPlacement::LaneLocal,
                kind: SegOpKind::SegMap,
                map_bodies: vec![],
                input_array_types: vec![],
                input_elem_types: vec![],
                map_output_elem_types: vec![int],
                map_input_indices: vec![],
                map_destinations: vec![SoacDestination::Fresh],
                acc_destinations: vec![],
                result_types: vec![],
                output_slots: vec![],
                resources: vec![],
            }),
            operand_nodes: smallvec![],
            result: Some(result),
            effects: None,
            span: None,
        });
        assert!(eliminate_dead_seg_ops_in_graph(&mut graph));
        assert!(graph.skeleton.blocks[graph.skeleton.entry].side_effects.is_empty());
    }
}
