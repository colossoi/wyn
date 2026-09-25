//! Concrete array storage preparation shared by both compiler routes.
use super::ir::{LoopScopes, Substitutions, ValueDef};
use super::stage::{Elaborated, Optimized};
use super::types::{BlockId, FuncBody, InstId, InstKind, ValueId, ValueRef, WynFunction};
use crate::builtins::{by_id, Purity};
use crate::op::{BinaryOperator, OpTag};
use crate::types::{is_array_variant_view, is_virtual_array, Type, TypeExt, TypeName};
use std::collections::HashMap;
use wyn_graph::DominatorTree;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ExpressionScope {
    Floating,
    Block(BlockId),
}

/// Prepare concrete storage introduced during SSA lowering.
/// Source helper expansion, folding, expression sharing and hoisting belong to Egglog.
pub fn optimize(mut program: Elaborated) -> Optimized {
    for body in program
        .functions
        .iter_mut()
        .map(|f| &mut f.body)
        .chain(program.entry_points.iter_mut().map(|e| &mut e.body))
        .chain(program.constants.iter_mut().map(|c| &mut c.body))
    {
        prepare_values(body);
    }
    program.retag()
}

pub(crate) fn is_speculatable(data: &InstKind) -> bool {
    match data {
        InstKind::Op {
            tag: OpTag::Intrinsic { id, overload_idx },
            ..
        } => {
            let builtin = by_id(*id);
            builtin.raw.purity == Purity::Pure
                && builtin
                    .overloads()
                    .get(*overload_idx)
                    .is_some_and(|overload| overload.lowering.is_speculatable())
        }
        InstKind::Op {
            tag: OpTag::BinOp(op),
            ..
        } => !matches!(
            op,
            BinaryOperator::Divide
                | BinaryOperator::Remainder
                | BinaryOperator::FloorDivide
                | BinaryOperator::FloorRemainder
                | BinaryOperator::Power
                | BinaryOperator::ShiftLeft
                | BinaryOperator::ShiftRight
                | BinaryOperator::ShiftRightLogical
        ),
        InstKind::Op {
            tag:
                OpTag::Int(_)
                | OpTag::Uint(_)
                | OpTag::Float(_)
                | OpTag::Bool(_)
                | OpTag::Unit
                | OpTag::UnaryOp(_)
                | OpTag::Tuple(_)
                | OpTag::Vector(_)
                | OpTag::Matrix { .. }
                | OpTag::ArrayLit(_)
                | OpTag::Project { .. }
                | OpTag::Materialize,
            ..
        } => true,
        _ => false,
    }
}

fn reusable(data: &InstKind) -> bool {
    // Partial arithmetic is deterministic for the same SSA operands. Reusing
    // an already evaluated result needs no proof that it is safe to speculate.
    // Keep memory, opaque calls, and context-dependent intrinsics excluded.
    match data {
        InstKind::Op {
            tag: OpTag::Intrinsic { id, overload_idx },
            ..
        } => {
            let builtin = by_id(*id);
            builtin.raw.purity == Purity::Pure
                && builtin
                    .overloads()
                    .get(*overload_idx)
                    .is_some_and(|overload| overload.lowering.is_reusable())
        }
        InstKind::Op {
            tag: OpTag::BinOp(_), ..
        } => true,
        _ => is_speculatable(data),
    }
}

#[allow(dead_code)] // Kept for comparison; disabled during the pure Egglog port.
fn reuse_dominating_expressions(body: &mut FuncBody) {
    let function = &mut body.inner;
    let dominators = DominatorTree::build(function.entry, |block, successors| {
        successors.extend(function.blocks[block].term.successors());
    });
    let mut expressions: HashMap<_, (BlockId, ValueId)> = HashMap::new();
    let mut replacements = Substitutions::default();
    for &block in dominators.preorder() {
        for instruction in std::mem::take(&mut function.blocks[block].insts) {
            function.insts[instruction].data.substitute_values(&mut |value| replacements.resolve(value));
            let node = &function.insts[instruction];
            if let (Some(result), InstKind::Op { tag, operands }) = (node.result, &node.data) {
                if reusable(&node.data) {
                    // SSA dominance decides reuse. Textual backends own their
                    // lexical declarations, including values used after loops.
                    let key = (function.values[result].ty.clone(), tag.clone(), operands.clone());
                    let previous = expressions.get(&key).and_then(|&(producer, previous)| {
                        dominators.dominates(producer, block).then_some(previous)
                    });
                    if let Some(previous) = previous {
                        replacements.insert(result, previous.into());
                        function.insts.remove(instruction);
                        continue;
                    }
                    // In dominator preorder a candidate outside the current
                    // subtree is no longer needed. Never move the retained
                    // instruction, including when it is guarded or in a loop.
                    expressions.insert(key, (block, result));
                }
            }
            function.blocks[block].insts.push(instruction);
        }
    }
    replacements.finish(function);
}

fn materialize_dynamic_index(
    function: &mut WynFunction,
    loop_scopes: &LoopScopes,
    original_block: BlockId,
    instruction: InstId,
    expressions: &mut HashMap<(ExpressionScope, ValueId), ValueId>,
) {
    let InstKind::Op {
        tag: OpTag::Index,
        operands,
    } = &function.insts[instruction].data
    else {
        return;
    };
    let [ValueRef::Ssa(array), ValueRef::Ssa(index)] = operands.as_slice() else {
        return;
    };
    if let ValueDef::Inst { inst } = function.values[*index].def {
        if matches!(
            function.insts[inst].data,
            InstKind::Op {
                tag: OpTag::Int(_) | OpTag::Uint(_),
                ..
            }
        ) {
            return;
        }
    }
    let ty = &function.values[*array].ty;
    let is_scalar_array = ty.is_array()
        && ty.elem_type().is_some_and(|element| {
            matches!(
                element,
                Type::Constructed(
                    TypeName::Int(_) | TypeName::UInt(_) | TypeName::Float(_) | TypeName::Bool,
                    _
                )
            )
        });
    if !is_scalar_array || ty.array_variant().is_some_and(is_array_variant_view) || is_virtual_array(ty) {
        return;
    }
    let array = *array;
    let ty = ty.clone();
    let tag = OpTag::Materialize;
    let materialize_operands = vec![array.into()];
    let scope = if loop_scopes.value_varies(function, array.into()) {
        ExpressionScope::Block(original_block)
    } else {
        ExpressionScope::Floating
    };
    let key = (scope, array);
    let materialized = *expressions.entry(key).or_insert_with(|| {
        let data = InstKind::Op {
            tag,
            operands: materialize_operands,
        };
        match scope {
            ExpressionScope::Floating => function.append_floating_inst(data, ty),
            ExpressionScope::Block(block) => function.append_inst(block, data, ty),
        }
    });
    if let InstKind::Op { tag, operands } = &mut function.insts[instruction].data {
        *tag = OpTag::DynamicExtract;
        operands[0] = materialized.into();
    }
}

fn prepare_values(body: &mut FuncBody) {
    // SSA reuse and early folding are disabled for the pure Egglog port.
    // Backend preparation still folds generated control flow.
    let loop_scopes = LoopScopes::analyze(&body.inner);
    let mut expressions = HashMap::new();
    let blocks = body.inner.blocks.keys().collect::<Vec<_>>();
    for block in blocks {
        for instruction in std::mem::take(&mut body.inner.blocks[block].insts) {
            materialize_dynamic_index(
                &mut body.inner,
                &loop_scopes,
                block,
                instruction,
                &mut expressions,
            );
            body.inner.blocks[block].insts.push(instruction);
        }
    }
}

#[cfg(test)]
#[path = "optimize_tests.rs"]
mod tests;
