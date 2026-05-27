//! Shared EGraph emission primitives. The three EGIR-construction
//! contexts — `from_tlc::Converter`, `egir::builder::EntryBuilder`, and
//! the in-place rewrite helpers in `egir::parallelize` — all need to
//! intern the same set of pure ops (literals, intrinsics, BinOps,
//! StorageViews) and push the same shapes of side-effects (`Store`,
//! `Pending(soac)`). This module owns those primitives so the three
//! contexts don't drift in their representation.
//!
//! The functions take `Option<Span>` for span attachment; pass
//! `None` when no source span is available, otherwise the caller's
//! current span. Bigger stateful helpers (`emit_store_through_view`,
//! `emit_pending_soac`) also take the target `BlockId` and a mutable
//! effect-token counter.

use polytype::Type;
use smallvec::{SmallVec, smallvec};

use crate::ast::{Span, TypeName};
use crate::builtins::BuiltinId;
use crate::builtins::catalog;
use crate::ssa::framework::BlockId;
use crate::ssa::types::{ConstantValue, InstKind, ValueRef};

use super::types::{
    EGraph, ENode, EffectToken, NodeId, PendingSoac, PureOp, PureViewSource, SideEffect, SideEffectKind,
};

// ---------------------------------------------------------------------------
// Pure ops
// ---------------------------------------------------------------------------

/// `u32` literal — the helper most code reaches for. Same canonical
/// shape (`PureOp::Uint(n.to_string())`) as `from_tlc` produces from
/// `TermKind::IntLit` so hash-consing deduplicates across the two
/// emission paths.
pub fn intern_u32(graph: &mut EGraph, n: u32, span: Option<Span>) -> NodeId {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    graph.intern_pure_with_span(PureOp::Uint(n.to_string()), smallvec![], u32_ty, span)
}

/// Constant via `EGraph::intern_constant` (canonical `ENode::Constant`
/// form). Use this when the value comes through a `ConstantValue`
/// already (e.g. carrying a reduce's neutral element across passes).
/// For freshly-typed-out integer/float literals from terms, prefer the
/// `PureOp::Uint`/`Int`/`Float` form via the other helpers.
pub fn intern_constant(graph: &mut EGraph, value: ConstantValue, ty: Type<TypeName>) -> NodeId {
    graph.intern_constant(value, ty)
}

/// Generic intrinsic call (`PureOp::Intrinsic` with `overload_idx: 0`).
pub fn intern_intrinsic(
    graph: &mut EGraph,
    id: BuiltinId,
    operands: SmallVec<[NodeId; 4]>,
    ty: Type<TypeName>,
    span: Option<Span>,
) -> NodeId {
    graph.intern_pure_with_span(PureOp::Intrinsic { id, overload_idx: 0 }, operands, ty, span)
}

/// Binary op (`PureOp::BinOp`). `op` is the operator string (`"+"`,
/// `"-"`, etc.) — matches the convention `from_tlc` uses.
pub fn intern_binop(
    graph: &mut EGraph,
    op: &str,
    lhs: NodeId,
    rhs: NodeId,
    ty: Type<TypeName>,
    span: Option<Span>,
) -> NodeId {
    graph.intern_pure_with_span(PureOp::BinOp(op.into()), smallvec![lhs, rhs], ty, span)
}

/// `StorageView(Storage{set, binding})` with the default
/// `[0, _w_intrinsic_storage_len(set, binding)]` operand pair.
pub fn intern_storage_view(
    graph: &mut EGraph,
    set: u32,
    binding: u32,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> NodeId {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let set_nid = intern_u32(graph, set, span);
    let binding_nid = intern_u32(graph, binding, span);
    let storage_len_id = catalog().known().storage_len;
    let len_nid = intern_intrinsic(
        graph,
        storage_len_id,
        smallvec![set_nid, binding_nid],
        u32_ty,
        span,
    );
    let zero_nid = intern_u32(graph, 0, span);
    graph.intern_pure_with_span(
        PureOp::StorageView(PureViewSource::Storage { set, binding }),
        smallvec![zero_nid, len_nid],
        view_ty,
        span,
    )
}

/// `StorageView(Storage{set, binding})` with caller-supplied `offset`
/// and `len`. Used to build a chunked sub-view of a larger storage
/// buffer (phase1 of parallel reduce/scan).
pub fn intern_chunked_storage_view(
    graph: &mut EGraph,
    set: u32,
    binding: u32,
    offset: NodeId,
    len: NodeId,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> NodeId {
    graph.intern_pure_with_span(
        PureOp::StorageView(PureViewSource::Storage { set, binding }),
        smallvec![offset, len],
        view_ty,
        span,
    )
}

// ---------------------------------------------------------------------------
// Side effects
// ---------------------------------------------------------------------------

/// Find the next unused `EffectToken` by scanning all skeleton blocks.
/// Mirrors (and supersedes) `soac_expand::next_effect_token` and the
/// `egir::parallelize::max_effect` helper.
pub fn next_effect_token(graph: &EGraph) -> u32 {
    let mut max = 0u32;
    for (_, block) in &graph.skeleton.blocks {
        for se in &block.side_effects {
            if let Some((a, b)) = se.effects {
                max = max.max(a.0).max(b.0);
            }
        }
    }
    max + 1
}

pub fn alloc_effect(next_effect: &mut u32) -> EffectToken {
    let t = EffectToken(*next_effect);
    *next_effect += 1;
    t
}

/// Emit a `Store` side-effect in `block`. `place_nid` must be a place-
/// producing pure op (`ViewIndex`, `OutputSlot`). Returns the produced
/// effect-out token.
pub fn emit_store(
    graph: &mut EGraph,
    block: BlockId,
    place_nid: NodeId,
    value_nid: NodeId,
    next_effect: &mut u32,
    span: Option<Span>,
) -> EffectToken {
    let effect_in = EffectToken(0); // placeholder; real chain is built by elaborate
    let effect_out = alloc_effect(next_effect);
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Inst(InstKind::Store {
            place: Default::default(),
            value: ValueRef::Ssa(Default::default()),
        }),
        operand_nodes: smallvec![place_nid, value_nid],
        result: None,
        effects: Some((effect_in, effect_out)),
        span,
    });
    effect_out
}

/// Emit a workgroup execution+memory barrier (`InstKind::ControlBarrier`)
/// in `block`. No operands or result; the effect token keeps it ordered
/// against the workgroup-shared loads/stores it synchronizes. Returns the
/// produced effect-out token.
pub fn emit_workgroup_barrier(graph: &mut EGraph, block: BlockId, next_effect: &mut u32) -> EffectToken {
    let effect_in = EffectToken(0); // placeholder; real chain is built by elaborate
    let effect_out = alloc_effect(next_effect);
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Inst(InstKind::ControlBarrier),
        operand_nodes: smallvec![],
        result: None,
        effects: Some((effect_in, effect_out)),
        span: None,
    });
    effect_out
}

/// Emit a store through a `StorageView` at `index_nid`. Builds the
/// `ViewIndex` pure node and the `Store` side-effect.
pub fn emit_storage_store(
    graph: &mut EGraph,
    block: BlockId,
    view_nid: NodeId,
    index_nid: NodeId,
    value_nid: NodeId,
    elem_ty: Type<TypeName>,
    next_effect: &mut u32,
    span: Option<Span>,
) {
    let place_nid =
        graph.intern_pure_with_span(PureOp::ViewIndex, smallvec![view_nid, index_nid], elem_ty, span);
    let _ = emit_store(graph, block, place_nid, value_nid, next_effect, span);
}

/// Push a `SideEffectKind::Pending(soac)` side-effect into `block` with
/// the given operands; returns the allocated `result_nid` (typed as
/// `result_ty`, which the SOAC's lowering recovers from
/// `graph.types[result_nid]`).
pub fn emit_pending_soac(
    graph: &mut EGraph,
    block: BlockId,
    soac: PendingSoac,
    operands: SmallVec<[NodeId; 4]>,
    result_ty: Type<TypeName>,
    next_effect: &mut u32,
    span: Option<Span>,
) -> NodeId {
    let result_nid = graph.alloc_side_effect_result(result_ty);
    let effect_in = EffectToken(0);
    let effect_out = alloc_effect(next_effect);
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Pending(soac),
        operand_nodes: operands,
        result: Some(result_nid),
        effects: Some((effect_in, effect_out)),
        span,
    });
    result_nid
}

// ---------------------------------------------------------------------------
// Read-side inspection
// ---------------------------------------------------------------------------

/// If `view_nid` is a `PureOp::StorageView(Storage{set, binding})`,
/// return the `(set, binding)` pair. Otherwise `None`.
pub fn extract_storage_view_source(graph: &EGraph, view_nid: NodeId) -> Option<(u32, u32)> {
    match &graph.nodes[view_nid] {
        ENode::Pure {
            op: PureOp::StorageView(PureViewSource::Storage { set, binding }),
            ..
        } => Some((*set, *binding)),
        _ => None,
    }
}

/// If `nid` is a `PureOp::ArrayRange`, return `(start, len, step?)`
/// NodeIds. Otherwise `None`.
pub fn extract_array_range_operands(
    graph: &EGraph,
    nid: NodeId,
) -> Option<(NodeId, NodeId, Option<NodeId>)> {
    match &graph.nodes[nid] {
        ENode::Pure {
            op: PureOp::ArrayRange { has_step },
            operands,
            ..
        } => {
            let step = if *has_step { Some(operands[2]) } else { None };
            Some((operands[0], operands[1], step))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Cross-graph cloning
// ---------------------------------------------------------------------------

/// Recursively clone a pure subgraph rooted at `root` from `src` into
/// `dst`, returning the new root `NodeId`. Used to copy a reduce's
/// neutral element (or any pure value) from one entry's EGraph into
/// another's — phase2 needs a fresh copy of phase1's NE since EGraph
/// NodeIds don't cross entries.
///
/// Only pure nodes and constants are cloned; encountering a
/// `SideEffectResult` or a `BlockParam` returns `Err` because those
/// reference cross-block / cross-effect data that doesn't translate.
pub fn clone_pure_subgraph(src: &EGraph, dst: &mut EGraph, root: NodeId) -> Result<NodeId, String> {
    let mut memo: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();
    clone_inner(src, dst, root, &mut memo)
}

fn clone_inner(
    src: &EGraph,
    dst: &mut EGraph,
    nid: NodeId,
    memo: &mut std::collections::HashMap<NodeId, NodeId>,
) -> Result<NodeId, String> {
    if let Some(&existing) = memo.get(&nid) {
        return Ok(existing);
    }
    let ty = src.types[&nid].clone();
    let new_nid = match &src.nodes[nid] {
        ENode::Constant(c) => dst.intern_constant(*c, ty),
        ENode::Pure { op, operands, .. } => {
            let new_ops: SmallVec<[NodeId; 4]> = operands
                .iter()
                .map(|&op_nid| clone_inner(src, dst, op_nid, memo))
                .collect::<Result<_, _>>()?;
            dst.intern_pure(op.clone(), new_ops, ty)
        }
        other => {
            return Err(format!(
                "clone_pure_subgraph: non-pure node {:?}",
                std::mem::discriminant(other)
            ));
        }
    };
    memo.insert(nid, new_nid);
    Ok(new_nid)
}
