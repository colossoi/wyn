//! Erase compile-time-only GPU resource handles before SSA construction.
//!
//! Buffer monomorphization makes a `StorageTexture` parameter's binding part
//! of its type. Binding-qualified image operations therefore need no runtime
//! image value. This pass removes those zero-footprint parameters and the
//! matching call operands after all SegOps have expanded, when the complete
//! call graph is concrete.

/// Physical EGIR with compile-time-only resource handles erased.
#[derive(Debug, Clone, Copy)]
pub enum ResourcesErasedTag {}
pub type ResourcesErased = crate::egir::program::Program<
    ResourcesErasedTag,
    crate::egir::ir::ProgramFamily<
        crate::egir::types::Physical,
        crate::interface::StorageBindingDecl,
        crate::egir::ir::RealizedOutputRoute,
        crate::egir::program::CoreProgramData,
    >,
    crate::egir::program::PlannedGlobal,
>;

#[cfg(test)]
#[path = "resource_erasure_tests.rs"]
mod resource_erasure_tests;

use crate::ast::TypeName;
use crate::egir::from_tlc::ConvertError;
use crate::egir::program::{PhysicalFunc, Program};
use crate::egir::types::{EGraph, ENode, EffectOp, Family, PureOp, SideEffectKind, SkeletonTerminator};
use crate::{LookupMap, LookupSet};
use polytype::Type;
use smallvec::SmallVec;

pub fn erase_resources(
    program: super::skel_opt::SkeletonOptimized,
) -> Result<ResourcesErased, ConvertError> {
    let erasures: LookupMap<crate::FunctionId, Vec<bool>> = program
        .functions
        .iter()
        .map(|function| {
            (
                function.region,
                function.params.iter().map(|(ty, _)| is_storage_image(ty)).collect(),
            )
        })
        .collect();

    let Program {
        functions,
        externs,
        entry_points,
        constants,
        data,
        global_context,
        state: _,
    } = program;
    let functions = functions
        .into_iter()
        .map(|function| erase_function_resources(function, &erasures))
        .collect::<Result<_, ConvertError>>()?;
    let entry_points = entry_points
        .into_iter()
        .map(|entry| entry.try_map_graph(|graph| rewrite_graph(graph, &erasures)))
        .collect::<Result<_, ConvertError>>()?;
    let constants = constants
        .into_iter()
        .map(|constant| constant.try_map_graph(|graph| rewrite_graph(graph, &erasures)))
        .collect::<Result<_, ConvertError>>()?;
    Ok(Program::from_parts(
        functions,
        externs,
        entry_points,
        constants,
        data,
        global_context,
    ))
}

fn is_storage_image(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::StorageTexture, _))
}

fn rewrite_graph<P: Family>(
    mut graph: EGraph<P>,
    erasures: &LookupMap<crate::FunctionId, Vec<bool>>,
) -> Result<EGraph<P>, ConvertError> {
    // Calls can be pure nodes or effect-anchored instructions. Rewrite both;
    // filtering by the callee's original signature keeps every positional ABI
    // change in one table.
    let pure_calls: Vec<_> = graph
        .nodes
        .iter()
        .filter_map(|(nid, node)| match &node.kind {
            ENode::Pure {
                op: PureOp::Call(callee),
                ..
            } if erasures.contains_key(callee) => Some((nid, *callee)),
            _ => None,
        })
        .collect();
    for (nid, callee) in pure_calls {
        let mask = &erasures[&callee];
        let mut error = None;
        graph.update_pure_node(nid, |_, operands| {
            error = filter_smallvec(operands, mask, &callee).err();
        });
        if let Some(error) = error {
            return Err(error);
        }
    }

    for (_, block) in &mut graph.skeleton.blocks {
        for effect in &mut block.side_effects {
            let SideEffectKind::Effect(EffectOp::Op { tag }) = &mut effect.kind else {
                continue;
            };
            if let PureOp::Call(callee) = tag {
                if let Some(mask) = erasures.get(callee) {
                    filter_smallvec(&mut effect.operand_nodes, mask, callee)?;
                }
            }
        }
    }
    Ok(graph)
}

fn filter_smallvec(
    operands: &mut SmallVec<[crate::egir::types::NodeId; 4]>,
    mask: &[bool],
    callee: &crate::FunctionId,
) -> Result<(), ConvertError> {
    if operands.len() != mask.len() {
        return Err(ConvertError::Internal(format!(
            "call to {callee:?} has {} EGIR operands but its concrete signature has {} parameters",
            operands.len(),
            mask.len()
        )));
    }
    *operands = operands
        .iter()
        .copied()
        .zip(mask)
        .filter_map(|(operand, erase)| (!erase).then_some(operand))
        .collect();
    Ok(())
}

fn erase_function_resources(
    function: PhysicalFunc,
    erasures: &LookupMap<crate::FunctionId, Vec<bool>>,
) -> Result<PhysicalFunc, ConvertError> {
    let crate::egir::ir::Func {
        region,
        name,
        span,
        linkage_name,
        params,
        return_ty,
        graph,
    } = function;
    let mut graph = rewrite_graph(graph, erasures)?;
    let erase: Vec<bool> = params.iter().map(|(ty, _)| is_storage_image(ty)).collect();
    if !erase.iter().any(|erase| *erase) {
        return Ok(PhysicalFunc::new(
            region,
            name,
            span,
            linkage_name,
            params,
            return_ty,
            graph,
        ));
    }

    let mut new_indices = vec![None; erase.len()];
    let mut next = 0;
    for (index, should_erase) in erase.iter().copied().enumerate() {
        if !should_erase {
            new_indices[index] = Some(next);
            next += 1;
        }
    }

    let mut erased_nodes = Vec::new();
    for (node_id, node) in &mut graph.nodes {
        let ENode::FuncParam { index } = &mut node.kind else {
            continue;
        };
        match new_indices.get(*index).copied().flatten() {
            Some(new_index) => *index = new_index,
            None => erased_nodes.push(node_id),
        }
    }

    // A remaining use means a new storage-image value operation was added
    // without being reified to a binding-qualified operation. Fail here rather
    // than letting a backend recreate an opaque runtime handle.
    let live = live_nodes(&graph);
    for erased in erased_nodes {
        if live.contains(&erased) {
            return Err(ConvertError::Internal(format!(
                "storage-image parameter in `{}` still has a runtime EGIR use after resource erasure",
                name
            )));
        }
        // Drop the dead param node: its index is stale after the
        // renumbering above and can collide with a surviving param's new
        // index, making elaboration's index-keyed param registration pick
        // the corpse over the real param.
        graph.remove_func_param(erased);
    }

    let params =
        params.into_iter().zip(erase).filter_map(|(param, erase)| (!erase).then_some(param)).collect();
    Ok(PhysicalFunc::new(
        region,
        name,
        span,
        linkage_name,
        params,
        return_ty,
        graph,
    ))
}

fn live_nodes<P: Family>(graph: &EGraph<P>) -> LookupSet<crate::egir::types::NodeId> {
    let mut roots = Vec::new();
    for (_, block) in &graph.skeleton.blocks {
        for effect in &block.side_effects {
            roots.extend(effect.operand_nodes.iter().copied());
        }
        match &block.term {
            SkeletonTerminator::Return(Some(value)) => roots.push(*value),
            SkeletonTerminator::Return(None) | SkeletonTerminator::Unreachable => {}
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
        }
    }

    wyn_graph::reachable_set(roots, wyn_graph::WalkOrder::DepthFirst, |node, out| {
        out.extend(graph.nodes[node].children());
    })
}
