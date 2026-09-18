//! Late scalar cleanup shared by both compiler routes. Dominating immutable
//! computations can be reused; only total ones may move across control boundaries.
use super::ir::{inline_single_block, LoopScopes, Substitutions};
use super::stage::{Elaborated, Optimized};
use super::types::{BlockId, ConstantValue, FuncBody, InstId, InstKind, ValueId, ValueRef, WynFunction};
use crate::builtins::{by_id, Purity};
use crate::op::{BinaryOperator, OpTag};
use crate::scalar_eval::{binary, Scalar};
use crate::types::{is_array_variant_view, is_virtual_array, Type, TypeExt, TypeName};
use crate::{BindingRef, FunctionId};
use std::collections::HashMap;
use wyn_base::split_one_mut;
use wyn_graph::{topo_sort_by_dependencies, DominatorTree};

const SMALL_HELPER_INSTRUCTION_LIMIT: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ExpressionScope {
    Floating,
    Block(BlockId),
}

type ExpressionKey = (
    ExpressionScope,
    Type,
    OpTag<BindingRef, FunctionId>,
    Vec<ValueRef>,
);

/// Inline small helpers, reuse dominating expressions, fold constants, and
/// intern movable expressions.
/// A separate pass assigns floating expressions to blocks before lowering.
pub fn optimize(mut program: Elaborated) -> Optimized {
    let indices: HashMap<_, _> = program.functions.iter().enumerate().map(|(i, f)| (f.id, i)).collect();
    // Callees are simplified first, so a small forwarding helper can disappear
    // in the same traversal. Recursive helpers are left for backend validation.
    if let Ok(order) = topo_sort_by_dependencies(0..program.functions.len(), |i, out| {
        for node in program.functions[i].body.inner.insts.values() {
            if let InstKind::Op {
                tag: OpTag::Call(id), ..
            } = node.data
            {
                out.extend(indices.get(&id).copied());
            }
        }
    }) {
        for i in order {
            let (current, other_functions) = split_one_mut(&mut program.functions, i);
            inline_small_helpers(&mut current.body, |id| {
                let helper = other_functions(*indices.get(&id)?)?;
                helper.linkage_name.is_none().then_some(&helper.body)
            });
            float_pure_values(&mut current.body);
        }
    }
    for body in program
        .entry_points
        .iter_mut()
        .map(|e| &mut e.body)
        .chain(program.constants.iter_mut().map(|c| &mut c.body))
    {
        inline_small_helpers(body, |id| {
            let f = &program.functions[*indices.get(&id)?];
            f.linkage_name.is_none().then_some(&f.body)
        });
        float_pure_values(body);
    }
    program.retag()
}

fn inline_small_helpers<'a>(body: &mut FuncBody, lookup: impl Fn(FunctionId) -> Option<&'a FuncBody>) {
    let mut replacements = Substitutions::default();
    let blocks: Vec<_> = body.inner.blocks.keys().collect();
    for block in blocks {
        for id in std::mem::take(&mut body.inner.blocks[block].insts) {
            body.inner.insts[id].data.substitute_values(&mut |value| replacements.resolve(value));
            let candidate = match &body.inner.insts[id].data {
                InstKind::Op {
                    tag: OpTag::Call(function),
                    operands,
                } => body.inner.insts[id].result.and_then(|result| {
                    lookup(*function)
                        .filter(|helper| is_small_inline_candidate(helper, operands.len()))
                        .map(|helper| (result, operands.clone(), helper))
                }),
                _ => None,
            };
            let Some((result, operands, helper)) = candidate else {
                body.inner.blocks[block].insts.push(id);
                continue;
            };
            let Some(returned) = inline_single_block(&mut body.inner, block, &helper.inner, &operands)
            else {
                body.inner.blocks[block].insts.push(id);
                continue;
            };
            replacements.insert(result, returned);
            body.inner.insts.remove(id);
        }
    }
    replacements.finish(&mut body.inner);
}

fn is_small_inline_candidate(helper: &FuncBody, argument_count: usize) -> bool {
    helper.num_blocks() == 1
        && helper.num_insts() <= SMALL_HELPER_INSTRUCTION_LIMIT
        && helper.inner.params.len() == argument_count
        // Cloning at the call site preserves execution and instruction order;
        // it does not require permission to speculate. Op instructions contain
        // only value operands. Place-bearing instructions need a place remapper
        // and are deliberately excluded from this single-block inliner.
        && helper.inner.insts.values().all(|node| {
            node.result.is_some() && matches!(node.data, InstKind::Op { .. })
        })
}

fn movable(data: &InstKind) -> bool {
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
    matches!(
        data,
        InstKind::Op {
            tag: OpTag::BinOp(_),
            ..
        }
    ) || movable(data)
}

fn reuse_dominating_expressions(body: &mut FuncBody, loop_scopes: &LoopScopes) {
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
                    // Enclosing values are visible inside a loop. WGSL keeps
                    // loop-local results inside it, so never search child or
                    // sibling scopes, even if their blocks dominate this use.
                    let mut key = (
                        loop_scopes.scope(block),
                        function.values[result].ty.clone(),
                        tag.clone(),
                        operands.clone(),
                    );
                    let previous = loop_scopes.enclosing_scopes(block).find_map(|scope| {
                        key.0 = scope;
                        let &(producer, previous) = expressions.get(&key)?;
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
                    key.0 = loop_scopes.scope(block);
                    expressions.insert(key, (block, result));
                }
            }
            function.blocks[block].insts.push(instruction);
        }
    }
    replacements.finish(function);
}

fn integer(function: &WynFunction, value: ValueRef) -> Option<i64> {
    match value {
        ValueRef::Const(ConstantValue::I32(n)) => Some(i64::from(n)),
        ValueRef::Const(ConstantValue::U32(n)) => Some(i64::from(n)),
        ValueRef::Ssa(id) => match &function.insts[function.inst_of_value(id)?].data {
            InstKind::Op {
                tag: OpTag::Int(s) | OpTag::Uint(s),
                ..
            } => s.parse().ok(),
            _ => None,
        },
        _ => None,
    }
}

fn fold_integer_binary(
    function: &mut WynFunction,
    instruction: InstId,
    replacements: &mut Substitutions,
) -> bool {
    let (
        Some(result),
        InstKind::Op {
            tag: OpTag::BinOp(operator),
            operands,
        },
    ) = (
        function.insts[instruction].result,
        &function.insts[instruction].data,
    )
    else {
        return false;
    };
    let [left, right] = operands.as_slice() else {
        return false;
    };
    let ty = &function.values[result].ty;
    if !matches!(ty, Type::Constructed(TypeName::Int(32) | TypeName::UInt(32), _)) {
        return false;
    }
    let Some(Scalar::Int(value)) = integer(function, *left)
        .zip(integer(function, *right))
        .and_then(|(left, right)| binary(*operator, Scalar::Int(left), Scalar::Int(right), ty))
    else {
        return false;
    };
    let constant = if matches!(ty, Type::Constructed(TypeName::UInt(32), _)) {
        ConstantValue::U32(value as u32)
    } else {
        ConstantValue::I32(value as i32)
    };
    replacements.insert(result, ValueRef::Const(constant));
    function.insts.remove(instruction);
    true
}

fn materialize_dynamic_index(
    function: &mut WynFunction,
    loop_scopes: &LoopScopes,
    original_block: BlockId,
    instruction: InstId,
    expressions: &mut HashMap<ExpressionKey, ValueId>,
) {
    let InstKind::Op {
        tag: OpTag::Index,
        operands,
    } = &function.insts[instruction].data
    else {
        return;
    };
    let [ValueRef::Ssa(array), ValueRef::Ssa(_)] = operands.as_slice() else {
        return;
    };
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
    let key = (scope, ty.clone(), tag.clone(), materialize_operands.clone());
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

fn float_or_share_instruction(
    function: &mut WynFunction,
    loop_scopes: &LoopScopes,
    original_block: BlockId,
    instruction: InstId,
    expressions: &mut HashMap<ExpressionKey, ValueId>,
    replacements: &mut Substitutions,
) -> bool {
    let node = &function.insts[instruction];
    let (Some(result), InstKind::Op { tag, operands }) = (node.result, &node.data) else {
        return false;
    };
    if !movable(&node.data) {
        return false;
    }
    let scope = if operands.iter().any(|operand| loop_scopes.value_varies(function, *operand)) {
        ExpressionScope::Block(original_block)
    } else {
        ExpressionScope::Floating
    };
    let key = (
        scope,
        function.values[result].ty.clone(),
        tag.clone(),
        operands.clone(),
    );
    if let Some(&previous) = expressions.get(&key) {
        replacements.insert(result, previous.into());
        function.insts.remove(instruction);
        return true;
    }
    expressions.insert(key, result);
    if matches!(scope, ExpressionScope::Floating) {
        function.float_inst(instruction);
        return true;
    }
    false
}

fn float_pure_values(body: &mut FuncBody) {
    let loop_scopes = LoopScopes::analyze(&body.inner);
    reuse_dominating_expressions(body, &loop_scopes);
    let mut replacements = Substitutions::default();
    let mut expressions = HashMap::new();
    let blocks = body.inner.blocks.keys().collect::<Vec<_>>();
    for block in blocks {
        for instruction in std::mem::take(&mut body.inner.blocks[block].insts) {
            body.inner.insts[instruction].data.substitute_values(&mut |value| replacements.resolve(value));
            // Inlining can expose integer constants after egglog has finished.
            // Evaluate with Wyn's width/wrapping semantics before WGSL's stricter
            // constant-expression checker sees an overflowing literal operation.
            if fold_integer_binary(&mut body.inner, instruction, &mut replacements) {
                continue;
            }
            // A materialization is itself interned, so every immutable source
            // has one addressable representation in the final legal scope.
            materialize_dynamic_index(
                &mut body.inner,
                &loop_scopes,
                block,
                instruction,
                &mut expressions,
            );
            if !float_or_share_instruction(
                &mut body.inner,
                &loop_scopes,
                block,
                instruction,
                &mut expressions,
                &mut replacements,
            ) {
                body.inner.blocks[block].insts.push(instruction);
            }
        }
    }
    replacements.finish(&mut body.inner);
}

#[cfg(test)]
#[path = "optimize_tests.rs"]
mod tests;
