//! Generic loop construction for physical SOAC expansion.

use super::array_io::emit_length;
use super::*;

/// One expanded-loop iteration. A body can finish in a different CFG block
/// when its effectful work is conditionally executed.
pub(super) struct LoopBody {
    pub(super) tail: BlockId,
    pub(super) carried: Vec<ValueId>,
}

/// Emit a real loop via `build_loop_skeleton`, invoking `emit_body` in the
/// body block to produce the new carried values, then wire the back-edge.
fn build_loop<F>(
    graph: &mut EGraph,
    bid: BlockId,
    idx_in_block: usize,
    len_input: &(ValueId, Type<TypeName>),
    carried: &[(Type<TypeName>, ValueId)],
    result: &ResultBinding,
    next_effect: &mut crate::IdSource<EffectToken>,
    mut emit_body: F,
) where
    F: FnMut(&mut EGraph, &mut crate::IdSource<EffectToken>, BlockId, ValueId, &[ValueId]) -> LoopBody,
{
    let handles = build_loop_skeleton(
        graph,
        bid,
        idx_in_block,
        LoopSkeletonSpec {
            carried: carried.to_vec(),
            result: result.clone(),
            len_input: len_input.clone(),
        },
    );
    let body = emit_body(
        graph,
        next_effect,
        handles.body,
        handles.idx_nid,
        &handles.carried,
    );
    debug_assert_eq!(body.carried.len(), carried.len());
    let next_i_nid = increment(graph, handles.idx_nid);
    let mut args = body.carried;
    args.push(next_i_nid);
    let args = graph.admit_flow_values(args);
    graph.skeleton.blocks[body.tail].term = SkeletonTerminator::Branch {
        target: handles.header,
        args,
    };
}

/// Try to unroll a small loop; if the trip count isn't statically small (or
/// `allow_unroll` is false), fall back to a real loop. Both paths share the
/// same `emit_body` closure — write iteration logic once.
pub(super) fn expand_loop<F>(
    graph: &mut EGraph,
    bid: BlockId,
    idx_in_block: usize,
    len_input: &(ValueId, Type<TypeName>),
    carried: &[(Type<TypeName>, ValueId)],
    result: &ResultBinding,
    next_effect: &mut crate::IdSource<EffectToken>,
    allow_unroll: bool,
    mut emit_body: F,
) where
    F: FnMut(&mut EGraph, &mut crate::IdSource<EffectToken>, BlockId, ValueId, &[ValueId]) -> LoopBody,
{
    if allow_unroll
        && try_unroll(
            graph,
            bid,
            idx_in_block,
            len_input,
            carried,
            result,
            next_effect,
            &mut emit_body,
        )
    {
        return;
    }
    build_loop(
        graph,
        bid,
        idx_in_block,
        len_input,
        carried,
        result,
        next_effect,
        emit_body,
    );
}

/// Generic small-loop unroller. Returns `true` if the loop was unrolled into
/// a short CFG chain rooted at `bid`; `false` if the trip count isn't
/// statically known to be small, and the caller should fall back to emitting a
/// real loop via `build_loop_skeleton`.
///
/// `emit_body(graph, next_effect, block, idx_const_nid, carried_in)` produces
/// the `carried_out` ValueNodeIds and the block that continues the iteration.
fn try_unroll<F>(
    graph: &mut EGraph,
    bid: BlockId,
    idx_in_block: usize,
    len_input: &(ValueId, Type<TypeName>),
    carried: &[(Type<TypeName>, ValueId)],
    result: &ResultBinding,
    next_effect: &mut crate::IdSource<EffectToken>,
    mut emit_body: F,
) -> bool
where
    F: FnMut(&mut EGraph, &mut crate::IdSource<EffectToken>, BlockId, ValueId, &[ValueId]) -> LoopBody,
{
    const UNROLL_THRESHOLD: usize = 16;

    // SoA-tuple driving inputs don't have a direct `array_size`; skip.
    if as_soa_tuple(&len_input.1).is_some() {
        return false;
    }
    let Some(size_ty) = len_input.1.array_size() else {
        return false;
    };
    let n = match size_ty {
        Type::Constructed(TypeName::Size(n), _) => *n,
        _ => return false,
    };
    if n > UNROLL_THRESHOLD {
        return false;
    }

    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

    // Stash side-effects and the original continuation. A body may introduce
    // selections, so the suffix belongs to its final continuation block, not
    // necessarily the original block.
    let suffix: Vec<SideEffect> = graph.skeleton.blocks[bid].side_effects.drain(idx_in_block..).collect();
    let original_term = std::mem::replace(
        &mut graph.skeleton.blocks[bid].term,
        SkeletonTerminator::Unreachable,
    );

    let mut carried_nids: Vec<ValueId> = carried.iter().map(|(_, init)| *init).collect();
    let mut current = bid;
    for i in 0..n {
        let idx_nid = graph.intern_pure(PureOp::Int(i.to_string()), smallvec![], i32_ty.clone(), None);
        let body = emit_body(graph, next_effect, current, idx_nid, &carried_nids);
        debug_assert_eq!(body.carried.len(), carried.len());
        carried_nids = body.carried;
        current = body.tail;
    }

    // Rebind the original SOAC result ValueId from the carried tuple.
    match result {
        ResultBinding::TupleFromCarried {
            result_node,
            tuple_ty,
            indices,
        } => {
            let tuple_parts: smallvec::SmallVec<[ValueId; 4]> =
                indices.iter().map(|idx| carried_nids[*idx]).collect();
            graph.replace_pure_node(*result_node, PureOp::Tuple(tuple_parts.len()), tuple_parts);
            graph.retype_node(*result_node, tuple_ty.clone());
        }
        ResultBinding::DummyBool { result_node } => {
            graph.replace_node_preserving_type(
                *result_node,
                ValueKind::Constant(crate::ssa::types::ConstantValue::Bool(false)),
            );
        }
    }

    graph.skeleton.blocks[current].side_effects.extend(suffix);
    graph.skeleton.blocks[current].term = original_term;
    true
}

/// Description of an accumulator-only SOAC (Reduce, reducing Screma): loop over one or
/// more input arrays, thread a scalar accumulator through a per-iteration call,
/// and yield the final accumulator as the result. No output array.

/// Common skeleton shared by every SOAC expansion: split the enclosing block
/// at the SOAC's index, create header/body/after blocks, wire the preheader
/// branch, and install the condbr on the header.
struct LoopSkeletonSpec {
    /// Per loop-carried value: (type, initial value in preheader).
    /// These become `header`'s block params, in order, followed by the index.
    carried: Vec<(Type<TypeName>, ValueId)>,
    /// How the original SOAC result ValueId should be rebound after expansion.
    result: ResultBinding,
    /// Input array for length calculation: (arr_nid, arr_ty).
    len_input: (ValueId, Type<TypeName>),
}

#[derive(Clone)]
pub(super) enum ResultBinding {
    /// Rebind `result_node` as a tuple of carried values. Used by
    /// Screma, which produces N maps + N accumulators into one tuple.
    TupleFromCarried {
        result_node: ValueId,
        tuple_ty: Type<TypeName>,
        indices: Vec<usize>,
    },
    /// Rebind `result_node` as a constant `Bool(false)` (dummy) — the SOAC
    /// produces no consumed value (the OutputView destination's writes
    /// are effectful and the "result" is discarded by the entry-point
    /// finalize step).
    DummyBool {
        result_node: ValueId,
    },
}

struct LoopHandles {
    header: BlockId,
    body: BlockId,
    /// One ValueId per loop-carried, matching the order in `spec.carried`.
    /// These are the header block-param ValueNodeIds, available inside body and
    /// on the else branch into `after`.
    pub(super) carried: Vec<ValueId>,
    /// The header's index block param.
    idx_nid: ValueId,
}

fn build_loop_skeleton(
    graph: &mut EGraph,
    bid: BlockId,
    idx_in_block: usize,
    spec: LoopSkeletonSpec,
) -> LoopHandles {
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

    // Split `bid` into preheader (bid) + after (holding suffix side-effects + old term).
    let after = graph.skeleton.split_block_before_effect(bid, idx_in_block);

    // The split moves the branching terminator to `after`: if `bid`
    // carries structured-control-flow header metadata (e.g. a Selection
    // whose CondBranch is in `old_term`), that metadata follows to
    // `after`, since `bid`'s new terminator is an unconditional branch to
    // the loop header — `after` is the selection/loop header now.
    // Rebind the SOAC's original result ValueId:
    //   - Carried: becomes the `after` block's param, populated from
    //     `carried[idx]` via the header's else branch below.
    //   - DummyBool: becomes an inline `Bool(false)` constant node in place.
    //     Consumers (if any) see a scalar false, matching the SSA pass's
    //     dummy-result convention for effect-only variants.
    match &spec.result {
        ResultBinding::TupleFromCarried {
            result_node,
            tuple_ty,
            indices,
        } => {
            let mut operands = smallvec::SmallVec::new();
            for carried_idx in indices {
                let Some((part_ty, _)) = spec.carried.get(*carried_idx) else {
                    continue;
                };
                let part_nid = graph.add_block_param(after, part_ty.clone());
                operands.push(part_nid);
            }
            graph.replace_pure_node(*result_node, PureOp::Tuple(operands.len()), operands);
            graph.retype_node(*result_node, tuple_ty.clone());
        }
        ResultBinding::DummyBool { result_node } => {
            graph.replace_node_preserving_type(
                *result_node,
                ValueKind::Constant(crate::ssa::types::ConstantValue::Bool(false)),
            );
        }
    }

    // Build header with one block-param per carried plus the index.
    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let mut carried_nids = Vec::with_capacity(spec.carried.len());
    for (ty, _) in &spec.carried {
        let nid = graph.add_block_param(header, ty.clone());
        carried_nids.push(nid);
    }
    let idx_nid = graph.add_block_param(header, i32_ty.clone());

    // Preheader terminator: br header(init_carried..., 0).
    let zero_nid = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone(), None);
    let mut preheader_args: Vec<ValueId> = spec.carried.iter().map(|(_, init)| *init).collect();
    preheader_args.push(zero_nid);
    let preheader_args = graph.admit_flow_values(preheader_args);
    graph.skeleton.blocks[bid].term = SkeletonTerminator::Branch {
        target: header,
        args: preheader_args,
    };

    // Header terminator: condbr i<len -> body / after(result_carried).
    let len_nid = emit_length(graph, spec.len_input.0, &spec.len_input.1, &i32_ty);
    let cond_nid = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Less),
        smallvec![idx_nid, len_nid],
        bool_ty,
        None,
    );
    let else_args = match &spec.result {
        ResultBinding::TupleFromCarried { indices, .. } => {
            graph.admit_flow_values(indices.iter().map(|idx| carried_nids[*idx]))
        }
        // No `after` block param in the dummy case — branch with empty args.
        ResultBinding::DummyBool { .. } => vec![],
    };
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond: cond_nid,
        then_target: body,
        then_args: vec![],
        else_target: after,
        else_args,
    };

    graph.skeleton.blocks[header].control_header = Some(ControlHeader::Loop {
        merge: after,
        continue_block: body,
    });

    LoopHandles {
        header,
        body,
        carried: carried_nids,
        idx_nid,
    }
}

/// Emit `idx + 1` as a pure op.
fn increment(graph: &mut EGraph, idx_nid: ValueId) -> ValueId {
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let one_nid = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone(), None);
    graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![idx_nid, one_nid],
        i32_ty,
        None,
    )
}
