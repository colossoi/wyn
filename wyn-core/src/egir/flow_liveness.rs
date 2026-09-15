//! Remove unobserved loop and branch result fields before array materialization.

use super::block_interface;
use super::ir::PlaceOp;
use super::physical_flow::type_contains_materialized_flow;
use super::slice::{projected_tuple_field, value_inputs};
use super::soac::metadata::Metadata;
use super::types::{
    EGraph, EffectOp, OperandRef, Physical, Scheduled, SideEffectKind, SideEffectSite, SkeletonTerminator,
    SoacEffect, ValueKind,
};

pub(crate) fn prune_unused_state(graph: &mut EGraph<Scheduled>) -> Result<(), String> {
    if !graph.skeleton.blocks.values().any(|block| {
        block
            .params
            .iter()
            .any(|parameter| type_contains_materialized_flow(graph.nodes[parameter.value()].ty()))
    }) {
        return Ok(());
    }
    block_interface::split_product_columns(graph)?;
    // Make field-sensitive dependencies explicit for physical lowering too.
    loop {
        let aliases: Vec<_> = graph
            .nodes
            .iter()
            .filter_map(|(value, node)| {
                let field = projected_tuple_field(graph, value)?;
                (node.alias().is_none() && node.ty() == graph.nodes[field].ty()).then_some((value, field))
            })
            .collect();
        if aliases.is_empty() {
            break;
        }
        for &(value, field) in &aliases {
            graph.replace_value_references(value, field);
        }
        graph.install_aliases(aliases);
    }
    graph.canonicalize_boundary_operands();
    let interfaces = block_interface::extract(graph)?;
    let mut roots = Vec::new();
    for block in graph.skeleton.blocks.values() {
        for effect in &block.side_effects {
            roots.extend(graph.effect_boundary_value_dependencies(effect));
            match effect.kind() {
                SideEffectKind::Effect(EffectOp::Call { site }) => {
                    roots.extend(graph.call_value_dependencies(*site));
                }
                SideEffectKind::Soac(SoacEffect(_, soac)) => {
                    roots.extend(soac.metadata_values());
                    for capture in soac.metadata_bodies().into_iter().flat_map(|body| body.captures()) {
                        if let OperandRef::Place(place) = capture {
                            roots.extend(graph.place_value_dependencies(*place));
                        }
                    }
                }
                _ => {}
            }
        }
        match &block.term {
            SkeletonTerminator::CondBranch { cond, .. } => roots.push(*cond),
            SkeletonTerminator::Return(_) => roots.extend(block.term.referenced_nodes()),
            SkeletonTerminator::Branch { .. } | SkeletonTerminator::Unreachable => {}
        }
    }
    // Edge arguments only become live when their destination field is live.
    // Following every edge eagerly would retain self-sustaining dead state.
    let live = wyn_graph::reachable_set(roots, wyn_graph::WalkOrder::DepthFirst, |value, out| {
        let canonical = graph.canonical_value(value);
        if value != canonical {
            out.push(canonical);
        } else if let ValueKind::BlockParam { block, index } = graph.nodes[value].kind() {
            out.extend(block_interface::dependencies(&interfaces[block], *index).map(|(_, value)| value));
        } else {
            out.extend(value_inputs(graph, value));
        }
    });
    block_interface::select_columns(graph, |_, interface| {
        Ok(interface
            .columns()
            .iter()
            .enumerate()
            .filter_map(|(index, column)| live.contains(&column.parameter().value()).then_some(index))
            .collect())
    })?;
    Ok(())
}

/// Required call expansion can leave a copied argument whose callee never
/// reads it. Remove direct stores to unobserved local allocations. Derived
/// addresses stay conservative; loop-carried arrays are handled before lowering.
pub(crate) fn prune_unused_allocations(graph: &mut EGraph<Physical>) {
    let mut unused = graph
        .places()
        .iter()
        .filter_map(|(id, place)| matches!(place.op(), PlaceOp::AllocaResult).then_some(id))
        .collect::<crate::LookupSet<_>>();
    let mut roots = Vec::new();
    for block in graph.skeleton.blocks.values() {
        roots.extend(block.term.referenced_nodes());
        if let SkeletonTerminator::Return(Some(result)) = &block.term {
            result.for_each_place(|place| {
                unused.remove(&place);
            });
        }
        for effect in &block.side_effects {
            roots.extend(graph.effect_boundary_value_dependencies(effect));
            for operand in effect.operands() {
                if let OperandRef::Place(place) = operand {
                    unused.remove(place);
                }
            }
            if let Some(result) = graph.effect_result_binding(effect) {
                result.for_each_place(|place| {
                    unused.remove(&place);
                });
            }
            match effect.kind() {
                SideEffectKind::Effect(EffectOp::Load { place } | EffectOp::Atomic { place, .. }) => {
                    unused.remove(place);
                }
                SideEffectKind::Effect(EffectOp::Call { site }) => {
                    roots.extend(graph.call_value_dependencies(*site));
                    for operand in graph.call(*site).arguments() {
                        if let OperandRef::Place(place) = operand {
                            unused.remove(&place);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    for place in graph.places().values() {
        if let PlaceOp::Index { base, .. } | PlaceOp::Slice { base, .. } = place.op() {
            unused.remove(base);
        }
    }
    let live = wyn_graph::reachable_set(roots, wyn_graph::WalkOrder::DepthFirst, |value, out| {
        out.extend(value_inputs(graph, value));
    });
    for value in live {
        if let ValueKind::PlaceView { place } | ValueKind::PlaceLength { place } = graph.nodes[value].kind()
        {
            unused.remove(place);
        }
    }
    let mut dead = Vec::new();
    for (block, contents) in &graph.skeleton.blocks {
        for (index, effect) in contents.side_effects.iter().enumerate() {
            if matches!(effect.kind(),
            SideEffectKind::Effect(EffectOp::Alloca { result: place } | EffectOp::Store { place })
                if unused.contains(place))
            {
                dead.push(SideEffectSite { block, index });
            }
        }
    }
    for site in dead.into_iter().rev() {
        graph.skeleton.remove_effect_splicing_dependencies(site);
    }
}
