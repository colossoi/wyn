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

use super::ir::Family;
use super::types::{EGraph, FlowValueId, PureOp, SideEffect, SideEffectSite, SkeletonTerminator, ValueId};

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

fn split_out_effect<P: Family>(
    graph: &mut EGraph<P>,
    site: SideEffectSite,
) -> Result<(SideEffect<P>, BlockId), String> {
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
    Ok((effect, continuation))
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
    let (effect, continuation) = split_out_effect(graph, site)?;
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
pub(crate) struct CountedLoop<P: Family> {
    pub(crate) effect: SideEffect<P>,
    pub(crate) header: BlockId,
    pub(crate) body: BlockId,
    pub(crate) continuation: BlockId,
    pub(crate) carried: Vec<ValueId>,
    pub(crate) index: ValueId,
}

/// Replace an effect with an i32 counted loop carrying typed values.
///
/// `exit_carried` selects which carried values become continuation block
/// parameters. `bind_exit` lets the pass bind those values to its logical
/// results without taking ownership of any CFG surgery. `emit_trip_count`
/// remains pass policy and is invoked at the same point as the prior local
/// loop builder.
pub(crate) fn replace_effect_with_counted_loop<P: Family>(
    graph: &mut EGraph<P>,
    site: SideEffectSite,
    carried: &[(Type<TypeName>, ValueId)],
    exit_carried: &[usize],
    emit_trip_count: impl FnOnce(&mut EGraph<P>) -> ValueId,
    mut bind_exit: impl FnMut(&mut EGraph<P>, usize, ValueId) -> Result<(), String>,
) -> Result<CountedLoop<P>, String> {
    for index in exit_carried {
        if *index >= carried.len() {
            return Err(format!(
                "loop exit carried index {index} is outside {} carried values",
                carried.len()
            ));
        }
    }

    let (effect, continuation) = split_out_effect(graph, site)?;
    let mut continuation_args = Vec::with_capacity(exit_carried.len());
    for (exit, index) in exit_carried.iter().copied().enumerate() {
        let value = graph.add_block_param(continuation, carried[index].0.clone());
        bind_exit(graph, exit, value)?;
        continuation_args.push(index);
    }

    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let mut carried_values = Vec::with_capacity(carried.len());
    for (ty, _) in carried {
        carried_values.push(graph.add_block_param(header, ty.clone()));
    }
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let index = graph.add_block_param(header, i32_ty.clone());

    let zero = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone(), None);
    let mut initial = carried.iter().map(|(_, value)| *value).collect::<Vec<_>>();
    initial.push(zero);
    graph.skeleton.blocks[site.block].term = SkeletonTerminator::Branch {
        target: header,
        args: graph.admit_flow_values(initial),
    };

    let trip_count = emit_trip_count(graph);
    let condition = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Less),
        smallvec![index, trip_count],
        Type::Constructed(TypeName::Bool, vec![]),
        None,
    );
    let merge_args = graph
        .admit_flow_values(continuation_args.iter().map(|carried_index| carried_values[*carried_index]));
    install_loop(graph, header, condition, body, continuation, merge_args, body);

    Ok(CountedLoop {
        effect,
        header,
        body,
        continuation,
        carried: carried_values,
        index,
    })
}

/// Wire the emitted iteration tail back to a counted-loop header.
pub(crate) fn finish_counted_loop_iteration<P: Family>(
    graph: &mut EGraph<P>,
    tail: BlockId,
    header: BlockId,
    carried: impl IntoIterator<Item = ValueId>,
) {
    graph.skeleton.blocks[tail].term = SkeletonTerminator::Branch {
        target: header,
        args: graph.admit_flow_values(carried),
    };
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
