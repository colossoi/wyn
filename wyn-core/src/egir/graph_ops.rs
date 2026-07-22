//! Shared EGraph emission primitives. The three EGIR-construction
//! contexts — `from_tlc::Converter`, `egir::builder::EntryBuilder`, and
//! the in-place rewrite helpers in `egir::parallelize` — all need to
//! intern the same set of pure ops (literals, intrinsics, BinOps,
//! StorageViews) and push the same shapes of side-effects (`Store`,
//! semantic `Soac` effects). This module owns those primitives so the three
//! contexts don't drift in their representation.
//!
//! The functions take `Option<Span>` for span attachment; pass
//! `None` when no source span is available, otherwise the caller's
//! current span. Bigger stateful helpers (`emit_store_through_view`,
//! `emit_pending_soac`) also take the target `BlockId` and a mutable
//! effect-token counter.

use crate::LookupMap;
use polytype::Type;
use smallvec::{smallvec, SmallVec};
use std::collections::{HashMap, HashSet};

use crate::ast::{Span, TypeName};
use crate::builtins::{catalog, BuiltinId};
use crate::flow::BlockId;
use crate::ssa::types::ConstantValue;
use crate::BindingRef;

use super::types::{
    EGraph, ENode, EffectOp, EffectToken, EgirPhase, GraphResource, NodeId, Physical, PureOp,
    PureViewSource, Raw, ResourceAccess, SegResourceAccess, Semantic, SideEffect, SideEffectKind,
    SideEffectSite, SkeletonTerminator, Soac, SoacEffect, WynSoacPhase,
};

#[cfg(test)]
#[path = "graph_ops_tests.rs"]
mod graph_ops_tests;

/// Phase-specific SOAC metadata that contributes to a produced value.
///
/// Raw SOACs have captures and operator seeds but no resolved segmented
/// iteration space.  Semantic SOACs additionally expose their resolved space
/// through `SideEffect::referenced_nodes`.
pub(crate) trait ValueProducerPhase: EgirPhase {
    fn effect_value_inputs(effect: &SideEffect<Self>) -> Vec<NodeId>;
}

impl<R: GraphResource> ValueProducerPhase for Raw<R> {
    fn effect_value_inputs(effect: &SideEffect<Self>) -> Vec<NodeId> {
        let mut nodes = effect.operand_nodes.to_vec();
        let SideEffectKind::Soac(SoacEffect(_, soac)) = &effect.kind else {
            return nodes;
        };
        nodes.extend(soac.seg_bodies().into_iter().flat_map(|body| body.captures.iter().copied()));
        if let Soac::Screma(op) = soac {
            for operator in op.operators() {
                nodes.push(operator.neutral);
                nodes.extend(operator.shape.iter().copied());
            }
        }
        nodes
    }
}

impl<R: GraphResource> ValueProducerPhase for Semantic<R> {
    fn effect_value_inputs(effect: &SideEffect<Self>) -> Vec<NodeId> {
        effect.referenced_nodes().collect()
    }
}

/// The complete value-producing closure behind one or more EGIR values.
///
/// `ENode::children` covers floating pure expressions, but intentionally has
/// no edges for effect results or block parameters.  Analyses that need the
/// actual producer must also follow an effect result to its anchored effect and
/// a block parameter to every incoming CFG argument.  Keeping both visited
/// sets makes loop-carried values finite even though those additional edges can
/// form cycles.
#[derive(Debug, Default)]
pub(crate) struct ValueProducerClosure {
    pub(crate) nodes: HashSet<NodeId>,
    pub(crate) effects: HashSet<SideEffectSite>,
}

/// Follow pure tails, value-producing effects, and CFG block arguments to the
/// values that can contribute to `roots`.
pub(crate) fn value_producer_closure<P: ValueProducerPhase>(
    graph: &EGraph<P>,
    roots: impl IntoIterator<Item = NodeId>,
) -> ValueProducerClosure {
    let producer_index = graph.side_effect_index();
    let mut closure = ValueProducerClosure::default();
    let mut pending = roots.into_iter().collect::<Vec<_>>();

    while let Some(node) = pending.pop() {
        if !closure.nodes.insert(node) {
            continue;
        }
        let Some(definition) = graph.nodes.get(node) else {
            continue;
        };
        match definition {
            ENode::Pure { operands, .. } => pending.extend(operands.iter().copied()),
            ENode::Union { left, right } => pending.extend([*left, *right]),
            ENode::BlockParam { block, index } => {
                extend_incoming_block_args(graph, *block, *index, &mut pending);
            }
            ENode::SideEffectResult => {
                let Some(site) = producer_index.site(node) else {
                    continue;
                };
                if closure.effects.insert(site) {
                    pending.extend(P::effect_value_inputs(graph.skeleton.effect(site)));
                }
            }
            ENode::FuncParam { .. } | ENode::Constant(_) => {}
        }
    }

    closure
}

/// Follow every value used by executable graph structure, together with
/// caller-supplied result roots. This is the common reachability boundary for
/// analyses of a projected recipe: block effects and terminators are executed,
/// while projection-preserved but unused metadata is not.
pub(crate) fn execution_value_producer_closure<P: ValueProducerPhase>(
    graph: &EGraph<P>,
    result_roots: impl IntoIterator<Item = NodeId>,
) -> ValueProducerClosure {
    value_producer_closure(
        graph,
        execution_value_roots(graph).into_iter().chain(result_roots),
    )
}

/// Values referenced directly by executable effects and terminators.
///
/// The phase adapter includes SOAC captures and other producer metadata that
/// the phase-agnostic IR cannot see through `P::Soac`.
pub(crate) fn execution_value_roots<P: ValueProducerPhase>(graph: &EGraph<P>) -> Vec<NodeId> {
    graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, block)| {
            block.side_effects.iter().flat_map(P::effect_value_inputs).chain(block.term.referenced_nodes())
        })
        .collect()
}

/// Pure-graph reachability from every executable effect and terminator.
pub(crate) fn reachable_execution_values<P: ValueProducerPhase>(graph: &EGraph<P>) -> Vec<NodeId> {
    wyn_graph::reachable_from_ordered(
        execution_value_roots(graph),
        wyn_graph::WalkOrder::DepthFirst,
        |node, out| {
            if let Some(definition) = graph.nodes.get(node) {
                out.extend(definition.children());
            }
        },
    )
}

/// Maximal movable values at the boundary of executable graph structure.
///
/// A value belongs to the frontier when it is movable and is either used
/// directly by an effect/terminator or consumed by a non-movable value. The
/// predicate owns the meaning of "movable" (loop invariant, stage invariant,
/// cloneable, and so on), while this helper owns the shared graph boundary
/// calculation.
pub(crate) fn maximal_execution_frontier<P: ValueProducerPhase>(
    graph: &EGraph<P>,
    mut movable: impl FnMut(NodeId) -> bool,
) -> Vec<NodeId> {
    let reachable = reachable_execution_values(graph);
    let reachable_set = reachable.iter().copied().collect::<HashSet<_>>();
    let movable = reachable.iter().map(|node| (*node, movable(*node))).collect::<HashMap<_, _>>();
    let mut boundary = execution_value_roots(graph).into_iter().collect::<HashSet<_>>();
    for node in &reachable {
        if movable[node] {
            continue;
        }
        if let Some(definition) = graph.nodes.get(*node) {
            boundary
                .extend(definition.children().into_iter().filter(|child| reachable_set.contains(child)));
        }
    }
    let mut frontier =
        reachable.into_iter().filter(|node| boundary.contains(node) && movable[node]).collect::<Vec<_>>();
    frontier.sort_unstable();
    frontier.dedup();
    frontier
}

/// Storage resources read by the complete producer closure behind `roots`.
pub(crate) fn read_storage_resources<P>(
    graph: &EGraph<P>,
    roots: impl IntoIterator<Item = NodeId>,
) -> Vec<SegResourceAccess<super::program::SemanticResourceRef>>
where
    P: ValueProducerPhase + EgirPhase<Resource = super::program::SemanticResourceRef>,
{
    let resources = value_producer_closure(graph, roots)
        .nodes
        .into_iter()
        .filter_map(|node| extract_storage_view_source(graph, node))
        .collect::<HashSet<_>>();
    let mut resources = resources
        .into_iter()
        .map(|resource| SegResourceAccess {
            resource,
            access: ResourceAccess::Read,
        })
        .collect::<Vec<_>>();
    resources.sort_by_key(|resource| resource.resource);
    resources
}

/// Return the output selected by a direct projection of `root`.
pub(crate) fn projection_index<P: EgirPhase>(
    graph: &EGraph<P>,
    node: NodeId,
    root: NodeId,
) -> Option<usize> {
    match graph.nodes.get(node)? {
        ENode::Pure {
            op: PureOp::Project { index },
            operands,
        } if operands.first() == Some(&root) => Some(*index as usize),
        _ => None,
    }
}

/// Follow nested projections back to `root` and return its selected output.
/// For `Project(Project(root, outer), inner)`, this is `outer`.
pub(crate) fn root_projection_index<P: EgirPhase>(
    graph: &EGraph<P>,
    node: NodeId,
    root: NodeId,
) -> Option<usize> {
    let mut current = node;
    let mut root_index = None;
    loop {
        if current == root {
            return root_index;
        }
        let ENode::Pure {
            op: PureOp::Project { index },
            operands,
        } = graph.nodes.get(current)?
        else {
            return None;
        };
        root_index = Some(*index as usize);
        current = *operands.first()?;
    }
}

fn extend_incoming_block_args<P: EgirPhase>(
    graph: &EGraph<P>,
    target: BlockId,
    index: usize,
    pending: &mut Vec<NodeId>,
) {
    for (_, predecessor) in &graph.skeleton.blocks {
        match &predecessor.term {
            SkeletonTerminator::Branch {
                target: branch_target,
                args,
            } if *branch_target == target => {
                pending.extend(args.get(index).copied());
            }
            SkeletonTerminator::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
                ..
            } => {
                let mut reaches_target = false;
                if *then_target == target {
                    pending.extend(then_args.get(index).copied());
                    reaches_target = true;
                }
                if *else_target == target {
                    pending.extend(else_args.get(index).copied());
                    reaches_target = true;
                }
                if reaches_target {
                    pending.push(*cond);
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Pure ops
// ---------------------------------------------------------------------------

/// `u32` literal — the helper most code reaches for. Same canonical
/// shape (`PureOp::Uint(n.to_string())`) as `from_tlc` produces from
/// `TermKind::IntLit` so hash-consing deduplicates across the two
/// emission paths.
pub fn intern_u32<P: EgirPhase>(graph: &mut EGraph<P>, n: u32, span: Option<Span>) -> NodeId {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    graph.intern_pure(PureOp::Uint(n.to_string()), smallvec![], u32_ty, span)
}

/// Constant via `EGraph::intern_constant` (canonical `ENode::Constant`
/// form). Use this when the value comes through a `ConstantValue`
/// already (e.g. carrying a reduce's neutral element across passes).
/// For freshly-typed-out integer/float literals from terms, prefer the
/// `PureOp::Uint`/`Int`/`Float` form via the other helpers.
pub fn intern_constant<P: EgirPhase>(
    graph: &mut EGraph<P>,
    value: ConstantValue,
    ty: Type<TypeName>,
) -> NodeId {
    graph.intern_constant(value, ty)
}

/// Generic intrinsic call (`PureOp::Intrinsic` with `overload_idx: 0`).
pub fn intern_intrinsic<P: EgirPhase>(
    graph: &mut EGraph<P>,
    id: BuiltinId,
    operands: SmallVec<[NodeId; 4]>,
    ty: Type<TypeName>,
    span: Option<Span>,
) -> NodeId {
    graph.intern_pure(PureOp::Intrinsic { id, overload_idx: 0 }, operands, ty, span)
}

/// Binary op (`PureOp::BinOp`). `op` is the operator string (`"+"`,
/// `"-"`, etc.) — matches the convention `from_tlc` uses.
pub fn intern_binop<P: EgirPhase>(
    graph: &mut EGraph<P>,
    op: &str,
    lhs: NodeId,
    rhs: NodeId,
    ty: Type<TypeName>,
    span: Option<Span>,
) -> NodeId {
    graph.intern_pure(PureOp::BinOp(op.into()), smallvec![lhs, rhs], ty, span)
}

/// `StorageView(Storage(br))` with the default
/// `[0, _w_intrinsic_storage_len(set, binding)]` operand pair.
pub fn intern_storage_view(
    graph: &mut EGraph<Physical>,
    br: BindingRef,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> NodeId {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let set_nid = intern_u32(graph, br.set, span);
    let binding_nid = intern_u32(graph, br.binding, span);
    let storage_len_id = catalog().known().storage_len;
    let len_nid = intern_intrinsic(
        graph,
        storage_len_id,
        smallvec![set_nid, binding_nid],
        u32_ty,
        span,
    );
    let zero_nid = intern_u32(graph, 0, span);
    let view_ty = crate::types::view_array_of(&view_ty, crate::types::buffer_tag(br));
    graph.intern_pure(
        PureOp::StorageView(PureViewSource::Storage(br)),
        smallvec![zero_nid, len_nid],
        view_ty,
        span,
    )
}

/// Target-independent storage view used after logical-resource allocation.
pub fn intern_resource_view<P: EgirPhase<Resource = super::program::SemanticResourceRef>>(
    graph: &mut EGraph<P>,
    resource: crate::ResourceId,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> NodeId {
    let len = intern_resource_len(graph, resource, span);
    let zero = intern_u32(graph, 0, span);
    intern_chunked_resource_view(graph, resource, zero, len, view_ty, span)
}

/// Target-independent logical-resource length.
pub fn intern_resource_len<P: EgirPhase<Resource = super::program::SemanticResourceRef>>(
    graph: &mut EGraph<P>,
    resource: crate::ResourceId,
    span: Option<Span>,
) -> NodeId {
    graph.intern_pure(
        PureOp::ResourceLen(super::program::SemanticResourceRef(resource)),
        smallvec![],
        Type::Constructed(TypeName::UInt(32), vec![]),
        span,
    )
}

pub fn intern_chunked_resource_view<P: EgirPhase<Resource = super::program::SemanticResourceRef>>(
    graph: &mut EGraph<P>,
    resource: crate::ResourceId,
    offset: NodeId,
    len: NodeId,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> NodeId {
    let view_ty =
        crate::types::view_array_of(&view_ty, Type::Constructed(TypeName::Resource(resource), vec![]));
    graph.intern_pure(
        PureOp::StorageView(PureViewSource::Storage(super::program::SemanticResourceRef(
            resource,
        ))),
        smallvec![offset, len],
        view_ty,
        span,
    )
}

/// A workgroup-shared array view: `StorageView(Workgroup{id, count})` with
/// `[offset=0, len=count]`. `view_ty` is the array type `[count]elem`; the
/// backends recover the element type from it to declare a module-scope
/// `array<elem, count>` in workgroup storage. Indexed with the same
/// `ViewIndex` + `Load`/`Store` machinery as storage views.
pub fn emit_workgroup_view<P: EgirPhase>(
    graph: &mut EGraph<P>,
    id: u32,
    count: u32,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> NodeId {
    let zero_nid = intern_u32(graph, 0, span);
    let count_nid = intern_u32(graph, count, span);
    // Workgroup-shared memory is not descriptor-bound: no (set, binding) region.
    let view_ty = crate::types::view_array_of(&view_ty, crate::types::no_buffer());
    graph.intern_pure(
        PureOp::StorageView(PureViewSource::Workgroup { id, count }),
        smallvec![zero_nid, count_nid],
        view_ty,
        span,
    )
}

/// `StorageView(Storage(br))` with caller-supplied `offset` and `len`.
/// Builds a chunked sub-view of a larger storage buffer (phase1 of
/// parallel reduce/scan).
pub fn intern_chunked_storage_view(
    graph: &mut EGraph<Physical>,
    br: BindingRef,
    offset: NodeId,
    len: NodeId,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> NodeId {
    let view_ty = crate::types::view_array_of(&view_ty, crate::types::buffer_tag(br));
    graph.intern_pure(
        PureOp::StorageView(PureViewSource::Storage(br)),
        smallvec![offset, len],
        view_ty,
        span,
    )
}

// ---------------------------------------------------------------------------
// Side effects
// ---------------------------------------------------------------------------

pub fn alloc_effect(effect_ids: &mut crate::IdSource<EffectToken>) -> EffectToken {
    effect_ids.next_id()
}

/// Emit a `Store` side-effect in `block`. `place_nid` must be a place-
/// producing pure op (`ViewIndex`, `OutputSlot`). Returns the produced
/// effect-out token.
pub fn emit_store<P: EgirPhase>(
    graph: &mut EGraph<P>,
    block: BlockId,
    place_nid: NodeId,
    value_nid: NodeId,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> EffectToken {
    let effect_in = alloc_effect(effect_ids);
    let effect_out = alloc_effect(effect_ids);
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Store),
        operand_nodes: smallvec![place_nid, value_nid],
        result: None,
        effects: Some((effect_in, effect_out)),
        span,
    });
    effect_out
}

/// Emit a workgroup execution+memory barrier
/// in `block`. No operands or result; the effect token keeps it ordered
/// against the workgroup-shared loads/stores it synchronizes. Returns the
/// produced effect-out token.
pub fn emit_workgroup_barrier<P: EgirPhase>(
    graph: &mut EGraph<P>,
    block: BlockId,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> EffectToken {
    let effect_in = alloc_effect(effect_ids);
    let effect_out = alloc_effect(effect_ids);
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::ControlBarrier),
        operand_nodes: smallvec![],
        result: None,
        effects: Some((effect_in, effect_out)),
        span: None,
    });
    effect_out
}

/// Emit a store through a `StorageView` at `index_nid`. Builds the
/// `ViewIndex` pure node and the `Store` side-effect.
pub fn emit_storage_store<P: EgirPhase>(
    graph: &mut EGraph<P>,
    block: BlockId,
    view_nid: NodeId,
    index_nid: NodeId,
    value_nid: NodeId,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> EffectToken {
    let place_nid = graph.intern_pure(PureOp::ViewIndex, smallvec![view_nid, index_nid], elem_ty, span);
    emit_store(graph, block, place_nid, value_nid, effect_ids, span)
}

/// Emit a `Load` of `place_nid` (a place-producing pure op like `ViewIndex`)
/// in `block`; returns the loaded-value node (typed `elem_ty`).
pub fn emit_load<P: EgirPhase>(
    graph: &mut EGraph<P>,
    block: BlockId,
    place_nid: NodeId,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> NodeId {
    let (result, effect) = detached_load(graph, place_nid, elem_ty, effect_ids, span);
    graph.skeleton.blocks[block].side_effects.push(effect);
    result
}

/// Construct a `Load` and its result without choosing its position in a
/// block. Rewriters use this when a synthesized load must be inserted before
/// an existing scheduled operation instead of appended to the block tail.
pub fn detached_load<P: EgirPhase>(
    graph: &mut EGraph<P>,
    place_nid: NodeId,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> (NodeId, SideEffect<P>) {
    let effect_in = alloc_effect(effect_ids);
    let effect_out = alloc_effect(effect_ids);
    let result = graph.alloc_side_effect_result(elem_ty);
    let effect = SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Load),
        operand_nodes: smallvec![place_nid],
        result: Some(result),
        effects: Some((effect_in, effect_out)),
        span,
    };
    (result, effect)
}

/// Emit a function-local `Alloca` side-effect in `block`. The returned NodeId
/// represents the allocated place — pass it to `intern_place_index` for
/// element-level addressing, or to `emit_load` / `emit_store` for whole-value
/// access. The place's element type is `elem_ty`; for an `[T;N]` allocation
/// `Load` returns the whole array and `PlaceIndex` produces `T`-typed sub-places.
pub fn emit_alloca<P: EgirPhase>(
    graph: &mut EGraph<P>,
    block: BlockId,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> NodeId {
    let effect_in = alloc_effect(effect_ids);
    let effect_out = alloc_effect(effect_ids);
    let place_nid = graph.alloc_side_effect_result(elem_ty.clone());
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Alloca { elem_ty }),
        operand_nodes: smallvec![],
        result: Some(place_nid),
        effects: Some((effect_in, effect_out)),
        span,
    });
    place_nid
}

/// Intern a `PlaceIndex` pure node: index into an existing place to produce a
/// sub-place addressing one element. The parent place can be an `Alloca`'d
/// array or any other place-producing node; the result has element type
/// `elem_ty` (e.g. `T` for an `[T;N]` parent).
pub fn intern_place_index<P: EgirPhase>(
    graph: &mut EGraph<P>,
    parent_place_nid: NodeId,
    index_nid: NodeId,
    elem_ty: Type<TypeName>,
    span: Option<Span>,
) -> NodeId {
    graph.intern_pure(
        PureOp::PlaceIndex,
        smallvec![parent_place_nid, index_nid],
        elem_ty,
        span,
    )
}

/// Emit `place[index] = value` as a `PlaceIndex` sub-place + `Store` in
/// `block`. Companion to `emit_storage_store` for function-local Alloca'd
/// arrays — no whole-array `Load`/`Store` round-trip.
pub fn emit_place_index_store<P: EgirPhase>(
    graph: &mut EGraph<P>,
    block: BlockId,
    parent_place_nid: NodeId,
    index_nid: NodeId,
    value_nid: NodeId,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) {
    let elem_place_nid = intern_place_index(graph, parent_place_nid, index_nid, elem_ty, span);
    let _ = emit_store(graph, block, elem_place_nid, value_nid, effect_ids, span);
}

/// Emit `view[index]` as a `ViewIndex` place + `Load` in `block`; returns the
/// loaded value. Companion to `emit_storage_store`.
pub fn emit_view_load<P: EgirPhase>(
    graph: &mut EGraph<P>,
    block: BlockId,
    view_nid: NodeId,
    index_nid: NodeId,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> NodeId {
    let place_nid = graph.intern_pure(
        PureOp::ViewIndex,
        smallvec![view_nid, index_nid],
        elem_ty.clone(),
        span,
    );
    emit_load(graph, block, place_nid, elem_ty, effect_ids, span)
}

/// Push a `SideEffectKind::Soac(SoacEffect(id, soac))` side-effect into `block` with
/// the given operands; returns the allocated `result_nid` (typed as
/// `result_ty`, which the SOAC's lowering recovers from
/// `graph.types[result_nid]`).
pub fn emit_pending_soac<P: WynSoacPhase>(
    graph: &mut EGraph<P>,
    block: BlockId,
    id: P::SoacId,
    soac: Soac<P>,
    operands: SmallVec<[NodeId; 4]>,
    result_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> NodeId {
    let result_nid = graph.alloc_side_effect_result(result_ty);
    let effect_in = alloc_effect(effect_ids);
    let effect_out = alloc_effect(effect_ids);
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Soac(SoacEffect(id, soac)),
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

/// Return the semantic identity carried by a storage-view node.
pub fn extract_storage_view_source<P: EgirPhase<Resource = super::program::SemanticResourceRef>>(
    graph: &EGraph<P>,
    view_nid: NodeId,
) -> Option<super::program::SemanticResourceRef> {
    match &graph.nodes[view_nid] {
        ENode::Pure {
            op: PureOp::StorageView(PureViewSource::Storage(resource)),
            ..
        } => Some(*resource),
        _ => None,
    }
}

/// Find the storage resource beneath a semantic place expression.
pub(crate) fn storage_resource_under(
    graph: &EGraph,
    root: NodeId,
) -> Option<super::program::SemanticResourceRef> {
    wyn_graph::find_map_reachable(
        [root],
        wyn_graph::WalkOrder::DepthFirst,
        |node, out| {
            if let Some(value) = graph.nodes.get(node) {
                out.extend(value.children());
            }
        },
        |node| extract_storage_view_source(graph, node),
    )
}

/// If `nid` is a `PureOp::ArrayRange`, return `(start, len, step?)`
/// NodeIds. Otherwise `None`.
pub fn extract_array_range_operands<P: EgirPhase>(
    graph: &EGraph<P>,
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
/// `dst`, returning the new root `NodeId`. Copies a reduce's neutral
/// element (or any pure value) from one entry's EGraph into another's —
/// phase2 needs a fresh copy of phase1's NE since EGraph NodeIds don't
/// cross entries.
///
/// Only pure nodes and constants are cloned; encountering a
/// `SideEffectResult` or a `BlockParam` returns `Err` because those
/// reference cross-block / cross-effect data that doesn't translate.
pub fn clone_pure_subgraph<P: EgirPhase>(
    src: &EGraph<P>,
    dst: &mut EGraph<P>,
    root: NodeId,
) -> Result<NodeId, String> {
    let mut memo: LookupMap<NodeId, NodeId> = LookupMap::new();
    clone_value_subgraph(src, dst, root, &mut memo, ConstantCopy::Intern, false)
}

/// Clone a pure subgraph of `src` into `dst`, but substitute the given `src`
/// nodes for already-existing `dst` nodes: any `(from, to)` pre-seeds the clone
/// memo, so a reference to `from` in `src` becomes `to` in `dst`. Lets a value
/// rooted at a non-pure node (e.g. a SOAC result) be re-expressed over a
/// replacement `dst` value without rebuilding its projection structure by hand.
pub fn clone_pure_subgraph_substituting<P: EgirPhase>(
    src: &EGraph<P>,
    dst: &mut EGraph<P>,
    root: NodeId,
    subs: &[(NodeId, NodeId)],
) -> Result<NodeId, String> {
    let mut memo: LookupMap<NodeId, NodeId> = subs.iter().copied().collect();
    clone_value_subgraph(src, dst, root, &mut memo, ConstantCopy::Intern, false)
}

#[derive(Clone, Copy)]
pub(crate) enum ConstantCopy {
    Intern,
    PreserveIdentity,
}

pub(crate) fn clone_value_subgraph<P: EgirPhase>(
    src: &EGraph<P>,
    dst: &mut EGraph<P>,
    nid: NodeId,
    memo: &mut LookupMap<NodeId, NodeId>,
    constants: ConstantCopy,
    allow_unions: bool,
) -> Result<NodeId, String> {
    if let Some(&existing) = memo.get(&nid) {
        return Ok(existing);
    }
    let ty = src
        .types
        .get(&nid)
        .cloned()
        .ok_or_else(|| format!("clone_value_subgraph: node {nid:?} has no type"))?;
    let new_nid =
        match src.nodes.get(nid).ok_or_else(|| format!("clone_value_subgraph: missing node {nid:?}"))? {
            ENode::Constant(c) => match constants {
                ConstantCopy::Intern => dst.intern_constant(*c, ty),
                ConstantCopy::PreserveIdentity => {
                    let target = dst.nodes.insert(ENode::Constant(*c));
                    dst.types.insert(target, ty);
                    target
                }
            },
            ENode::Pure { op, operands, .. } => {
                let new_ops: SmallVec<[NodeId; 4]> = operands
                    .iter()
                    .map(|&operand| clone_value_subgraph(src, dst, operand, memo, constants, allow_unions))
                    .collect::<Result<_, _>>()?;
                dst.intern_pure(op.clone(), new_ops, ty, src.node_spans.get(&nid).copied())
            }
            ENode::Union { left, right } if allow_unions => {
                let left = clone_value_subgraph(src, dst, *left, memo, constants, allow_unions)?;
                let right = clone_value_subgraph(src, dst, *right, memo, constants, allow_unions)?;
                dst.add_union(left, right)
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

/// Replace every *reference* to `old` with `new` across a whole body — pure
/// node operands, side-effect operands, SOAC captures, and terminator args. The
/// `old` node's definition is left intact (now unreferenced). Fusion uses this
/// to rewire the results of a producer/sibling op onto the fused op's result.
pub fn replace_all_references(graph: &mut EGraph<Semantic>, old: NodeId, new: NodeId) {
    if old == new {
        return;
    }
    let swap = |slot: &mut NodeId| {
        if *slot == old {
            *slot = new;
        }
    };
    graph.replace_node_references(old, new);
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            for slot in effect.referenced_node_slots() {
                swap(slot);
            }
        }
        block.term.visit_nodes_mut(swap);
    }
}
