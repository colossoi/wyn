//! Erase compile-time-only GPU resource handles before SSA construction.
//!
//! Region monomorphization makes a `StorageTexture` parameter's binding part
//! of its type. Binding-qualified image operations therefore need no runtime
//! image value. This pass removes those zero-footprint parameters and the
//! matching call operands after all SegOps have expanded, when the complete
//! call graph is concrete.

#[cfg(test)]
#[path = "resource_erasure_tests.rs"]
mod resource_erasure_tests;

use crate::ast::TypeName;
use crate::egir::from_tlc::ConvertError;
use crate::egir::program::{EgirFunc, EgirInner};
use crate::egir::types::{EGraph, ENode, PureOp, SideEffectKind, SkeletonTerminator};
use crate::ssa::types::{InstKind, ValueRef};
use crate::types::TypeExt;
use crate::{LookupMap, LookupSet};
use polytype::Type;
use smallvec::SmallVec;

pub fn run(inner: &mut EgirInner) -> Result<(), ConvertError> {
    let erasures: LookupMap<String, Vec<bool>> = inner
        .functions
        .iter()
        .map(|function| {
            (
                function.name.clone(),
                function.params.iter().map(|(ty, _)| is_storage_image(ty)).collect(),
            )
        })
        .collect();

    for function in &mut inner.functions {
        rewrite_graph(&mut function.graph, &erasures)?;
        erase_function_params(function)?;
    }
    for entry in &mut inner.entry_points {
        rewrite_graph(&mut entry.graph, &erasures)?;
    }
    Ok(())
}

fn is_storage_image(ty: &Type<TypeName>) -> bool {
    matches!(
        TypeExt::strip_unique(ty),
        Type::Constructed(TypeName::StorageTexture, _)
    )
}

fn rewrite_graph(graph: &mut EGraph, erasures: &LookupMap<String, Vec<bool>>) -> Result<(), ConvertError> {
    // Calls can be pure nodes or effect-anchored instructions. Rewrite both;
    // filtering by the callee's original signature keeps every positional ABI
    // change in one table.
    for (_, node) in &mut graph.nodes {
        if let ENode::Pure {
            op: PureOp::Call(callee),
            operands,
        } = node
        {
            if let Some(mask) = erasures.get(callee) {
                filter_smallvec(operands, mask, callee)?;
            }
        }
    }

    for (_, block) in &mut graph.skeleton.blocks {
        for effect in &mut block.side_effects {
            let SideEffectKind::Inst(InstKind::Op { tag, operands }) = &mut effect.kind else {
                continue;
            };
            if let PureOp::Call(callee) = tag {
                if let Some(mask) = erasures.get(callee) {
                    filter_smallvec(&mut effect.operand_nodes, mask, callee)?;
                    filter_vec(operands, mask, callee)?;
                }
            }
        }
    }
    Ok(())
}

fn filter_smallvec(
    operands: &mut SmallVec<[crate::egir::types::NodeId; 4]>,
    mask: &[bool],
    callee: &str,
) -> Result<(), ConvertError> {
    if operands.len() != mask.len() {
        return Err(ConvertError::Internal(format!(
            "call to `{callee}` has {} EGIR operands but its concrete signature has {} parameters",
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

fn filter_vec(operands: &mut Vec<ValueRef>, mask: &[bool], callee: &str) -> Result<(), ConvertError> {
    if operands.len() != mask.len() {
        return Err(ConvertError::Internal(format!(
            "call to `{callee}` has {} SSA operand placeholders but its concrete signature has {} parameters",
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

fn erase_function_params(function: &mut EgirFunc) -> Result<(), ConvertError> {
    let erase: Vec<bool> = function.params.iter().map(|(ty, _)| is_storage_image(ty)).collect();
    if !erase.iter().any(|erase| *erase) {
        return Ok(());
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
    for (node_id, node) in &mut function.graph.nodes {
        let ENode::FuncParam { index } = node else {
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
    let live = live_nodes(&function.graph);
    for erased in erased_nodes {
        if live.contains(&erased) {
            return Err(ConvertError::Internal(format!(
                "storage-image parameter in `{}` still has a runtime EGIR use after resource erasure",
                function.name
            )));
        }
    }

    function.params = function
        .params
        .drain(..)
        .zip(erase)
        .filter_map(|(param, erase)| (!erase).then_some(param))
        .collect();
    Ok(())
}

fn live_nodes(graph: &EGraph) -> LookupSet<crate::egir::types::NodeId> {
    let mut stack = Vec::new();
    for (_, block) in &graph.skeleton.blocks {
        for effect in &block.side_effects {
            stack.extend(effect.operand_nodes.iter().copied());
        }
        match &block.term {
            SkeletonTerminator::Return(Some(value)) => stack.push(*value),
            SkeletonTerminator::Return(None) | SkeletonTerminator::Unreachable => {}
            SkeletonTerminator::Branch { args, .. } => stack.extend(args.iter().copied()),
            SkeletonTerminator::CondBranch {
                cond,
                then_args,
                else_args,
                ..
            } => {
                stack.push(*cond);
                stack.extend(then_args.iter().copied());
                stack.extend(else_args.iter().copied());
            }
        }
    }

    let mut live = LookupSet::new();
    while let Some(node) = stack.pop() {
        if live.insert(node) {
            stack.extend(graph.nodes[node].children());
        }
    }
    live
}
