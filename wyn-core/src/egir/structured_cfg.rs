//! Invariant-preserving builders for structured EGIR control flow.
//!
//! Passes provide policy values and iteration bodies; this layer owns effect
//! replacement, continuation preservation, and the paired installation of a
//! terminator with its `ControlHeader`.

use crate::ast::TypeName;
use crate::flow::{BlockId, ControlHeader};
use crate::op;
use polytype::Type;
use smallvec::smallvec;
use wyn_base::IdSource;

use super::ir::Family;
use super::physical_flow::PhysicalMerge;
use super::types::{
    EGraph, EffectToken, FlowValueId, Physical, PureOp, ResultBinding, SideEffect, SideEffectSite,
    SkeletonTerminator, ValueId,
};

/// Install a guarded selection and its structured merge annotation as one
/// operation.
pub(crate) fn install_selection<P: Family>(
    graph: &mut EGraph<P>,
    header: BlockId,
    condition: ValueId,
    then_target: BlockId,
    else_target: BlockId,
    merge: BlockId,
) {
    let block = &mut graph.skeleton.blocks[header];
    block.term = SkeletonTerminator::CondBranch {
        cond: condition,
        then_target,
        then_args: vec![],
        else_target,
        else_args: vec![],
    };
    block.control_header = Some(ControlHeader::Selection { merge });
}

/// Install a counted-loop branch and its structured loop annotation as one
/// operation.
pub(crate) fn install_loop<P: Family>(
    graph: &mut EGraph<P>,
    header: BlockId,
    condition: ValueId,
    body: BlockId,
    merge: BlockId,
    merge_args: Vec<FlowValueId>,
    continue_block: BlockId,
) {
    let block = &mut graph.skeleton.blocks[header];
    block.term = SkeletonTerminator::CondBranch {
        cond: condition,
        then_target: body,
        then_args: vec![],
        else_target: merge,
        else_args: merge_args,
    };
    block.control_header = Some(ControlHeader::Loop {
        merge,
        continue_block,
    });
}

/// Effect removed from a block together with the continuation that retains its
/// suffix, terminator, and structured-control ownership.
pub(crate) struct EffectContinuation<P: Family> {
    pub(crate) effect: SideEffect<P>,
    pub(crate) continuation: BlockId,
}

/// Replace an effect with an empty branchable gap while preserving everything
/// that originally followed it in a continuation block.
pub(crate) fn replace_effect_with_continuation<P: Family>(
    graph: &mut EGraph<P>,
    site: SideEffectSite,
) -> Result<EffectContinuation<P>, String> {
    let source = graph
        .skeleton
        .blocks
        .get(site.block)
        .ok_or_else(|| format!("effect replacement references missing block {:?}", site.block))?;
    if site.index >= source.side_effects.len() {
        return Err(format!(
            "effect replacement index {} is outside block {:?}",
            site.index, site.block
        ));
    }
    let continuation = graph.skeleton.split_block_before_effect(site.block, site.index);
    let effect = graph.skeleton.blocks[continuation].side_effects.remove(0);
    Ok(EffectContinuation { effect, continuation })
}

/// Handles for an effect replaced by a guarded body and continuation.
pub(crate) struct GuardedEffect<P: Family> {
    pub(crate) effect: SideEffect<P>,
    pub(crate) body: BlockId,
    pub(crate) continuation: BlockId,
}

/// Replace an effect with `if condition { body }`, preserving its suffix,
/// terminator, and structured-control metadata in the returned continuation.
/// The condition closure runs after the continuation and body blocks have been
/// allocated, matching the construction order used by existing lowerers.
pub(crate) fn replace_effect_with_guarded_selection<P: Family>(
    graph: &mut EGraph<P>,
    site: SideEffectSite,
    emit_condition: impl FnOnce(&mut EGraph<P>) -> Result<ValueId, String>,
) -> Result<GuardedEffect<P>, String> {
    let replacement = replace_effect_with_continuation(graph, site)?;
    let effect = replacement.effect;
    let continuation = replacement.continuation;
    let body = graph.skeleton.create_block();
    let condition = emit_condition(graph)?;
    install_selection(graph, site.block, condition, body, continuation, continuation);
    Ok(GuardedEffect {
        effect,
        body,
        continuation,
    })
}

/// Complete a guarded replacement by joining its emitted body to the
/// preserved continuation.
pub(crate) fn finish_guarded_selection<P: Family>(
    graph: &mut EGraph<P>,
    tail: BlockId,
    continuation: BlockId,
) {
    graph.skeleton.blocks[tail].term = SkeletonTerminator::Branch {
        target: continuation,
        args: vec![],
    };
}

/// Handles exposed to a pass emitting one counted-loop iteration.
pub(crate) struct CountedLoop {
    pub(crate) effect: SideEffect<Physical>,
    pub(crate) body: BlockId,
    pub(crate) continuation: BlockId,
    pub(crate) carried: PhysicalMerge,
    pub(crate) index: ValueId,
}

/// Replace an effect with an i32 counted loop carrying structured physical
/// result bindings.
///
/// `exit_carried` selects which carried values become continuation block
/// parameters or fixed places. `bind_exit` lets the pass bind those physical
/// results without taking ownership of any CFG surgery. `emit_trip_count`
/// remains pass policy.
pub(crate) fn replace_effect_with_counted_loop(
    graph: &mut EGraph<Physical>,
    site: SideEffectSite,
    carried: &[ResultBinding<Type<TypeName>>],
    exit_carried: &[usize],
    effect_ids: &mut IdSource<EffectToken>,
    emit_trip_count: impl FnOnce(&mut EGraph<Physical>) -> ValueId,
    mut bind_exit: impl FnMut(
        &mut EGraph<Physical>,
        usize,
        &ResultBinding<Type<TypeName>>,
    ) -> Result<(), String>,
) -> Result<CountedLoop, String> {
    for index in exit_carried {
        if *index >= carried.len() {
            return Err(format!(
                "loop exit carried index {index} is outside {} carried values",
                carried.len()
            ));
        }
    }

    let replacement = replace_effect_with_continuation(graph, site)?;
    let effect = replacement.effect;
    let continuation = replacement.continuation;
    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let carried_state = PhysicalMerge::new(
        graph,
        header,
        carried.iter().map(|binding| {
            super::types::by_value_function_result::<super::types::WynLanguage>(binding.ty().clone())
        }),
        effect_ids,
    )?;
    let exit_sources =
        exit_carried.iter().map(|index| &carried_state.bindings()[*index]).collect::<Vec<_>>();
    let exit_state = PhysicalMerge::reusing_places(graph, continuation, exit_sources, effect_ids)?;
    for (exit, binding) in exit_state.bindings().iter().enumerate() {
        bind_exit(graph, exit, binding.result())?;
    }
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let index = graph.add_block_param(header, i32_ty.clone());

    let zero = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone(), None);
    let (initial_tail, mut initial_args) =
        carried_state.connect_results(graph, site.block, carried, effect_ids)?;
    initial_args.push(graph.admit_flow_value(zero));
    graph.skeleton.blocks[initial_tail].term = SkeletonTerminator::Branch {
        target: header,
        args: initial_args,
    };

    let trip_count = emit_trip_count(graph);
    let condition = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Less),
        smallvec![index, trip_count],
        Type::Constructed(TypeName::Bool, vec![]),
        None,
    );
    let exit_values = exit_carried
        .iter()
        .map(|index| carried_state.bindings()[*index].result().clone())
        .collect::<Vec<_>>();
    let (exit_tail, merge_args) = exit_state.connect_results(graph, header, &exit_values, effect_ids)?;
    if exit_tail != header {
        return Err("counted-loop exit unexpectedly required a place transfer".into());
    }
    install_loop(graph, header, condition, body, continuation, merge_args, body);

    Ok(CountedLoop {
        effect,
        body,
        continuation,
        carried: carried_state,
        index,
    })
}

/// Wire the emitted iteration tail back to a counted-loop header.
pub(crate) fn finish_counted_loop_iteration(
    graph: &mut EGraph<Physical>,
    tail: BlockId,
    state: &PhysicalMerge,
    carried: &[ResultBinding<Type<TypeName>>],
    index: ValueId,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<(), String> {
    let (tail, mut arguments) = state.connect_results(graph, tail, carried, effect_ids)?;
    arguments.push(graph.admit_flow_value(index));
    graph.skeleton.blocks[tail].term = SkeletonTerminator::Branch {
        target: state.block(),
        args: arguments,
    };
    Ok(())
}

/// Original executable continuation detached while an effect is replaced by
/// inline/unrolled code in its source block.
pub(crate) struct InlineEffectContinuation<P: Family> {
    suffix: Vec<SideEffect<P>>,
    terminator: SkeletonTerminator,
    control_header: Option<ControlHeader>,
}

/// Remove an effect and detach everything that must follow its inline
/// replacement, including structured-control ownership.
pub(crate) fn detach_effect_for_inline_replacement<P: Family>(
    graph: &mut EGraph<P>,
    site: SideEffectSite,
) -> Result<InlineEffectContinuation<P>, String> {
    let block = graph
        .skeleton
        .blocks
        .get_mut(site.block)
        .ok_or_else(|| format!("effect replacement references missing block {:?}", site.block))?;
    if site.index >= block.side_effects.len() {
        return Err(format!(
            "effect replacement index {} is outside block {:?}",
            site.index, site.block
        ));
    }
    block.side_effects.remove(site.index);
    let suffix = block.side_effects.drain(site.index..).collect();
    let terminator = std::mem::replace(&mut block.term, SkeletonTerminator::Unreachable);
    let control_header = block.control_header.take();
    Ok(InlineEffectContinuation {
        suffix,
        terminator,
        control_header,
    })
}

/// Restore the exact suffix and structured continuation after inline code.
pub(crate) fn restore_inline_effect_continuation<P: Family>(
    graph: &mut EGraph<P>,
    tail: BlockId,
    continuation: InlineEffectContinuation<P>,
) {
    let block = &mut graph.skeleton.blocks[tail];
    block.side_effects.extend(continuation.suffix);
    block.term = continuation.terminator;
    block.control_header = continuation.control_header;
}

#[cfg(test)]
#[path = "structured_cfg_tests.rs"]
mod structured_cfg_tests;
