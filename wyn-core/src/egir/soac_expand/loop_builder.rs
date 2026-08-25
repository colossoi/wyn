//! Generic loop construction for physical SOAC expansion.

use super::array_io::emit_length;
use super::*;
use crate::egir::graph_ops::{bind_by_value_result, rebind_physical_result};
use crate::egir::structured_cfg::{
    detach_effect_for_inline_replacement, finish_counted_loop_iteration, replace_effect_with_counted_loop,
    restore_inline_effect_continuation,
};
use crate::egir::types::{by_value_function_result, SideEffectSite};
use crate::op;
use crate::ssa;
use wyn_base::IdSource;

/// One expanded-loop iteration. A body can finish in a different CFG block
/// when its effectful work is conditionally executed.
pub(super) struct LoopBody {
    pub(super) tail: BlockId,
    pub(super) carried: Vec<ValueId>,
}

/// Emit a real loop via `build_loop_skeleton`, invoking `emit_body` in the
/// body block to produce the new carried values, then wire the back-edge.
fn build_loop<F>(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx_in_block: usize,
    len_input: &(ValueId, Type<TypeName>),
    carried: &[(Type<TypeName>, ValueId)],
    results: &[LoopResultBinding],
    next_effect: &mut IdSource<EffectToken>,
    mut emit_body: F,
) -> Result<BlockId, String>
where
    F: FnMut(
        &mut EGraph<Physical>,
        &mut IdSource<EffectToken>,
        BlockId,
        ValueId,
        &[ValueId],
    ) -> Result<LoopBody, String>,
{
    let mut exit_bindings = Vec::new();
    for binding in results {
        match binding.source {
            LoopResultSource::Carried(index) => exit_bindings.push((binding, index)),
            LoopResultSource::ConstantFalse => {
                let result = binding
                    .result
                    .single_value()
                    .ok_or_else(|| "a boolean loop result must have one by-value leaf".to_owned())?;
                graph.replace_node_preserving_type(
                    result,
                    ValueKind::Constant(ssa::types::ConstantValue::Bool(false)),
                );
            }
        }
    }
    let exit_carried = exit_bindings.iter().map(|(_, index)| *index).collect::<Vec<_>>();
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let handles = replace_effect_with_counted_loop(
        graph,
        SideEffectSite {
            block: bid,
            index: idx_in_block,
        },
        carried,
        &exit_carried,
        |graph| emit_length(graph, len_input.0, &len_input.1, &i32_ty),
        |graph, exit, value| {
            let binding = exit_bindings[exit].0;
            let abi = by_value_function_result::<WynLanguage>(binding.result.ty().clone());
            let replacement = bind_by_value_result(graph, &abi, value);
            rebind_physical_result(graph, &binding.result, &replacement)
        },
    )?;
    let body = emit_body(graph, next_effect, handles.body, handles.index, &handles.carried)?;
    debug_assert_eq!(body.carried.len(), carried.len());
    let next_i_nid = increment(graph, handles.index);
    let mut args = body.carried;
    args.push(next_i_nid);
    finish_counted_loop_iteration(graph, body.tail, handles.header, args);
    let _replaced_effect = handles.effect;
    Ok(handles.continuation)
}

/// Try to unroll a small loop; if the trip count isn't statically small (or
/// `allow_unroll` is false), fall back to a real loop. Both paths share the
/// same `emit_body` closure — write iteration logic once.
pub(super) fn expand_loop<F>(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx_in_block: usize,
    len_input: &(ValueId, Type<TypeName>),
    carried: &[(Type<TypeName>, ValueId)],
    results: &[LoopResultBinding],
    next_effect: &mut IdSource<EffectToken>,
    allow_unroll: bool,
    mut emit_body: F,
) -> Result<BlockId, String>
where
    F: FnMut(
        &mut EGraph<Physical>,
        &mut IdSource<EffectToken>,
        BlockId,
        ValueId,
        &[ValueId],
    ) -> Result<LoopBody, String>,
{
    if allow_unroll {
        if let Some(continuation) = try_unroll(
            graph,
            bid,
            idx_in_block,
            len_input,
            carried,
            results,
            next_effect,
            &mut emit_body,
        )? {
            return Ok(continuation);
        }
    }
    build_loop(
        graph,
        bid,
        idx_in_block,
        len_input,
        carried,
        results,
        next_effect,
        emit_body,
    )
}

/// Generic small-loop unroller. Returns the continuation block when the loop
/// was unrolled, or `None` when the caller should emit a real loop.
///
/// `emit_body(graph, next_effect, block, idx_const_nid, carried_in)` produces
/// the `carried_out` ValueNodeIds and the block that continues the iteration.
fn try_unroll<F>(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx_in_block: usize,
    len_input: &(ValueId, Type<TypeName>),
    carried: &[(Type<TypeName>, ValueId)],
    results: &[LoopResultBinding],
    next_effect: &mut IdSource<EffectToken>,
    mut emit_body: F,
) -> Result<Option<BlockId>, String>
where
    F: FnMut(
        &mut EGraph<Physical>,
        &mut IdSource<EffectToken>,
        BlockId,
        ValueId,
        &[ValueId],
    ) -> Result<LoopBody, String>,
{
    const UNROLL_THRESHOLD: usize = 16;

    // SoA-tuple driving inputs don't have a direct `array_size`; skip.
    if as_soa_tuple(&len_input.1).is_some() {
        return Ok(None);
    }
    let Some(size_ty) = len_input.1.array_size() else {
        return Ok(None);
    };
    let n = match size_ty {
        Type::Constructed(TypeName::Size(n), _) => *n,
        _ => return Ok(None),
    };
    if n > UNROLL_THRESHOLD {
        return Ok(None);
    }

    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

    // A body may introduce selections, so the original continuation belongs
    // to its final tail rather than necessarily the source block.
    let continuation = detach_effect_for_inline_replacement(
        graph,
        SideEffectSite {
            block: bid,
            index: idx_in_block,
        },
    )?;

    let mut carried_nids: Vec<ValueId> = carried.iter().map(|(_, init)| *init).collect();
    let mut current = bid;
    for i in 0..n {
        let idx_nid = graph.intern_pure(PureOp::Int(i.to_string()), smallvec![], i32_ty.clone(), None);
        let body = emit_body(graph, next_effect, current, idx_nid, &carried_nids)?;
        debug_assert_eq!(body.carried.len(), carried.len());
        carried_nids = body.carried;
        current = body.tail;
    }

    for binding in results {
        bind_unrolled_result(graph, binding, &carried_nids)?;
    }

    restore_inline_effect_continuation(graph, current, continuation);
    Ok(Some(current))
}

#[derive(Clone)]
pub(super) struct LoopResultBinding {
    pub(super) result: ResultBinding<Type<TypeName>>,
    pub(super) source: LoopResultSource,
}

#[derive(Clone)]
pub(super) enum LoopResultSource {
    Carried(usize),
    ConstantFalse,
}

fn bind_unrolled_result(
    graph: &mut EGraph<Physical>,
    binding: &LoopResultBinding,
    carried: &[ValueId],
) -> Result<(), String> {
    match &binding.source {
        LoopResultSource::Carried(index) => {
            let abi = by_value_function_result::<WynLanguage>(binding.result.ty().clone());
            let replacement = bind_by_value_result(graph, &abi, carried[*index]);
            rebind_physical_result(graph, &binding.result, &replacement)?;
        }
        LoopResultSource::ConstantFalse => {
            let result = binding
                .result
                .single_value()
                .ok_or_else(|| "a boolean loop result must have one by-value leaf".to_owned())?;
            graph.replace_node_preserving_type(
                result,
                ValueKind::Constant(ssa::types::ConstantValue::Bool(false)),
            );
        }
    }
    Ok(())
}

/// Emit `idx + 1` as a pure op.
fn increment(graph: &mut EGraph<Physical>, idx_nid: ValueId) -> ValueId {
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let one_nid = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone(), None);
    graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![idx_nid, one_nid],
        i32_ty,
        None,
    )
}
