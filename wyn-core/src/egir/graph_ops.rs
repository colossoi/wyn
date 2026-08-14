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

use super::ir::{Family, PlaceOp, Value};
use super::types::{
    EGraph, EffectOp, EffectToken, GraphResource, LoadMode, OperandRef, Physical, PlaceAccess, PlaceId,
    PlaceRegion, PlaceType, PureOp, PureViewSource, Raw, ResourceAccess, ResultBinding,
    SegResourceAccess, Semantic, SegBody, SideEffect, SideEffectKind, SideEffectSite,
    SkeletonTerminator, Soac, SoacEffect, ValueId, ValueKind, WynSoacPhase,
};

#[cfg(test)]
#[path = "graph_ops_tests.rs"]
mod graph_ops_tests;

pub fn decompose_value(graph: &mut EGraph<impl Family>, value: ValueId) -> Vec<ValueId> {
    fn walk<P: Family>(graph: &mut EGraph<P>, value: ValueId, ty: Type<TypeName>, out: &mut Vec<ValueId>) {
        let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = ty else {
            out.push(value);
            return;
        };
        for (index, field) in fields.into_iter().enumerate() {
            let field_value = graph.intern_pure(
                PureOp::Project { index: index as u32 },
                smallvec![value],
                field.clone(),
                None,
            );
            walk(graph, field_value, field, out);
        }
    }

    let mut values = Vec::new();
    let ty = graph.value(value).ty().clone();
    walk(graph, value, ty, &mut values);
    values
}

pub fn bind_by_value_result<P: Family>(
    graph: &mut EGraph<P>,
    abi: &super::types::FunctionResult<Type<TypeName>>,
    value: ValueId,
) -> ResultBinding<Type<TypeName>> {
    let values = decompose_value(graph, value);
    abi.bind(
        |slot, _| values[slot.index()],
        |_| panic!("a by-value result cannot bind a destination parameter"),
    )
}

pub fn pack_result_values<P: Family>(
    graph: &mut EGraph<P>,
    binding: &ResultBinding<Type<TypeName>>,
) -> Result<ValueId, String> {
    fn walk<P: Family>(
        graph: &mut EGraph<P>,
        ty: &Type<TypeName>,
        values: &mut impl Iterator<Item = ValueId>,
    ) -> Result<ValueId, String> {
        let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = ty else {
            return values.next().ok_or_else(|| "result binding has too few by-value leaves".into());
        };
        let fields = fields
            .iter()
            .map(|field| walk(graph, field, values))
            .collect::<Result<SmallVec<[ValueId; 4]>, _>>()?;
        Ok(graph.intern_pure(PureOp::Tuple(fields.len()), fields, ty.clone(), None))
    }

    if binding.destination_count() != binding.values().len() {
        return Err("place-backed result requires an explicit load before value materialization".into());
    }
    let mut values = binding.values().into_iter();
    let result = walk(graph, binding.ty(), &mut values)?;
    if values.next().is_some() {
        return Err("result binding has too many by-value leaves".into());
    }
    Ok(result)
}

/// Phase-specific SOAC metadata that contributes to a produced value.
///
/// Raw SOACs have captures and operator seeds but no resolved segmented
/// iteration space.  Semantic SOACs additionally expose their resolved space
/// through `SideEffect::referenced_nodes`.
pub(crate) trait ValueProducerPhase: Family {
    fn effect_metadata_inputs(effect: &SideEffect<Self>) -> Vec<ValueId>;

    fn effect_value_inputs(graph: &EGraph<Self>, effect: &SideEffect<Self>) -> Vec<ValueId> {
        let mut values = graph.effect_boundary_value_dependencies(effect);
        values.extend(Self::effect_metadata_inputs(effect));
        values
    }
}

impl<R: GraphResource> ValueProducerPhase for Raw<R> {
    fn effect_metadata_inputs(effect: &SideEffect<Self>) -> Vec<ValueId> {
        let mut nodes = Vec::new();
        let SideEffectKind::Soac(SoacEffect(_, soac)) = &effect.kind else {
            return nodes;
        };
        nodes.extend(soac.seg_bodies().into_iter().flat_map(SegBody::capture_values));
        if let Soac::Screma(op) = soac {
            nodes.extend(op.form.scans.iter().flat_map(|scan| scan.neutral.iter().copied()));
            nodes.extend(op.form.reductions.iter().flat_map(|reduction| reduction.neutral.iter().copied()));
        }
        nodes
    }
}

impl<R: GraphResource> ValueProducerPhase for Semantic<R> {
    fn effect_metadata_inputs(effect: &SideEffect<Self>) -> Vec<ValueId> {
        effect.referenced_nodes().collect()
    }
}

pub(crate) fn effect_value_inputs<P: ValueProducerPhase>(
    graph: &EGraph<P>,
    effect: &SideEffect<P>,
) -> Vec<ValueId> {
    P::effect_value_inputs(graph, effect)
}

/// The complete value-producing closure behind one or more EGIR values.
///
/// `ValueKind::children` covers floating pure expressions, but intentionally has
/// no edges for effect results or block parameters.  Analyses that need the
/// actual producer must also follow an effect result to its anchored effect and
/// a block parameter to every incoming CFG argument.  Keeping both visited
/// sets makes loop-carried values finite even though those additional edges can
/// form cycles.
#[derive(Debug, Default)]
pub(crate) struct ValueProducerClosure {
    pub(crate) nodes: HashSet<ValueId>,
    pub(crate) effects: HashSet<SideEffectSite>,
}

impl ValueProducerClosure {
    pub(crate) fn contains_node(&self, node: ValueId) -> bool {
        self.nodes.contains(&node)
    }
}

/// Executable graph locations whose values depend on a source value.
///
/// Locations are stable only for the graph snapshot used to build the
/// corresponding [`ValueUseIndex`].
#[derive(Debug, Default)]
pub(crate) struct ValueObservers {
    effects: HashSet<SideEffectSite>,
    terminators: HashSet<BlockId>,
}

impl ValueObservers {
    pub(crate) fn effect_sites(&self) -> impl Iterator<Item = SideEffectSite> + '_ {
        self.effects.iter().copied()
    }

    pub(crate) fn terminator_blocks(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.terminators.iter().copied()
    }
}

/// Reverse value-flow and executable-use index for one immutable graph
/// snapshot.
///
/// Pure successors follow only floating pure/union operands. Value successors
/// additionally cross side-effect results and CFG block arguments, mirroring
/// [`value_producer_closure`] in the opposite direction. This lets passes ask
/// centralized observer and liveness questions instead of repeatedly scanning
/// every effect and terminator with a producer-reachability query.
///
/// Rebuild the index after inserting, removing, reordering, or rewriting graph
/// structure. In particular, the [`SideEffectSite`] values it returns must not
/// survive a skeleton mutation.
pub(crate) struct ValueUseIndex {
    pure_successors: LookupMap<ValueId, Vec<ValueId>>,
    value_successors: LookupMap<ValueId, Vec<ValueId>>,
    effect_observers: LookupMap<ValueId, Vec<SideEffectSite>>,
    terminator_observers: LookupMap<ValueId, Vec<BlockId>>,
}

impl ValueUseIndex {
    pub(crate) fn build<P: ValueProducerPhase>(graph: &EGraph<P>) -> Self {
        let mut index = Self {
            pure_successors: LookupMap::new(),
            value_successors: LookupMap::new(),
            effect_observers: LookupMap::new(),
            terminator_observers: LookupMap::new(),
        };

        for (user, definition) in &graph.nodes {
            for source in definition.kind.children() {
                index.pure_successors.entry(source).or_default().push(user);
                index.value_successors.entry(source).or_default().push(user);
            }
        }

        for (block, body) in &graph.skeleton.blocks {
            for (effect_index, effect) in body.side_effects.iter().enumerate() {
                let site = SideEffectSite {
                    block,
                    index: effect_index,
                };
                for source in P::effect_value_inputs(graph, effect) {
                    index.effect_observers.entry(source).or_default().push(site);
                    if let Some(result) = &effect.result {
                        for result in result.values() {
                            index.value_successors.entry(source).or_default().push(result);
                        }
                    }
                }
            }
            for source in body.term.referenced_nodes() {
                index.terminator_observers.entry(source).or_default().push(block);
            }
            index_block_argument_successors(graph, &mut index.value_successors, &body.term);
        }

        index
    }

    /// Effects and terminators reached through floating pure/union users.
    pub(crate) fn pure_observers(&self, source: ValueId) -> ValueObservers {
        self.observers(source, &self.pure_successors)
    }

    /// Effects and terminators reached through complete value flow, including
    /// effect results and incoming CFG block arguments.
    pub(crate) fn value_observers(&self, source: ValueId) -> ValueObservers {
        self.observers(source, &self.value_successors)
    }

    /// Whether `user` consumes `source` through floating pure/union nodes.
    pub(crate) fn pure_reaches(&self, source: ValueId, user: ValueId) -> bool {
        self.reaches(source, user, &self.pure_successors)
    }

    fn observers(&self, source: ValueId, successors: &LookupMap<ValueId, Vec<ValueId>>) -> ValueObservers {
        let mut observers = ValueObservers::default();
        self.walk_users(source, successors, |user| {
            observers.effects.extend(self.effect_observers.get(&user).into_iter().flatten().copied());
            observers
                .terminators
                .extend(self.terminator_observers.get(&user).into_iter().flatten().copied());
            false
        });
        observers
    }

    fn reaches(
        &self,
        source: ValueId,
        target: ValueId,
        successors: &LookupMap<ValueId, Vec<ValueId>>,
    ) -> bool {
        self.walk_users(source, successors, |user| user == target)
    }

    fn walk_users(
        &self,
        source: ValueId,
        successors: &LookupMap<ValueId, Vec<ValueId>>,
        mut visit: impl FnMut(ValueId) -> bool,
    ) -> bool {
        let mut seen = HashSet::new();
        let mut pending = vec![source];
        while let Some(user) = pending.pop() {
            if !seen.insert(user) {
                continue;
            }
            if visit(user) {
                return true;
            }
            pending.extend(successors.get(&user).into_iter().flatten().copied());
        }
        false
    }
}

fn index_block_argument_successors<P: Family>(
    graph: &EGraph<P>,
    successors: &mut LookupMap<ValueId, Vec<ValueId>>,
    term: &SkeletonTerminator,
) {
    let mut add_edge = |target: BlockId, args: &[super::types::FlowValueId], condition: Option<ValueId>| {
        let Some(target_block) = graph.skeleton.blocks.get(target) else {
            return;
        };
        for (&argument, &parameter) in args.iter().zip(&target_block.params) {
            successors.entry(argument.value()).or_default().push(parameter.value());
            if let Some(condition) = condition {
                successors.entry(condition).or_default().push(parameter.value());
            }
        }
    };
    match term {
        SkeletonTerminator::Branch { target, args } => add_edge(*target, args, None),
        SkeletonTerminator::CondBranch {
            cond,
            then_target,
            then_args,
            else_target,
            else_args,
        } => {
            add_edge(*then_target, then_args, Some(*cond));
            add_edge(*else_target, else_args, Some(*cond));
        }
        SkeletonTerminator::Return(_) | SkeletonTerminator::Unreachable => {}
    }
}

/// Follow pure tails, value-producing effects, and CFG block arguments to the
/// values that can contribute to `roots`.
pub(crate) fn value_producer_closure<P: ValueProducerPhase>(
    graph: &EGraph<P>,
    roots: impl IntoIterator<Item = ValueId>,
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
        match &definition.kind {
            ValueKind::Pure { operands, .. } => pending.extend(operands.iter().copied()),
            ValueKind::Union { left, right } => pending.extend([*left, *right]),
            ValueKind::BlockParam { block, index } => {
                extend_incoming_block_args(graph, *block, *index, &mut pending);
            }
            ValueKind::SideEffectResult => {
                let Some(site) = producer_index.site(node) else {
                    continue;
                };
                if closure.effects.insert(site) {
                    pending.extend(P::effect_value_inputs(graph, graph.skeleton.effect(site)));
                }
            }
            ValueKind::CallResult { call, .. } => {
                pending.extend(graph.call(*call).arguments().iter().filter_map(|argument| argument.value()));
            }
            ValueKind::PlaceLength { place } => {
                pending.extend(graph.place_value_dependencies(*place));
            }
            ValueKind::FuncParam { .. } | ValueKind::Constant(_) => {}
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
    result_roots: impl IntoIterator<Item = ValueId>,
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
pub(crate) fn execution_value_roots<P: ValueProducerPhase>(graph: &EGraph<P>) -> Vec<ValueId> {
    graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, block)| {
            block
                .side_effects
                .iter()
                .flat_map(|effect| P::effect_value_inputs(graph, effect))
                .chain(block.term.referenced_nodes())
        })
        .collect()
}

/// Pure-graph reachability from every executable effect and terminator.
pub(crate) fn reachable_execution_values<P: ValueProducerPhase>(graph: &EGraph<P>) -> Vec<ValueId> {
    wyn_graph::reachable_from_ordered(
        execution_value_roots(graph),
        wyn_graph::WalkOrder::DepthFirst,
        |node, out| {
            if let Some(definition) = graph.nodes.get(node) {
                out.extend(definition.kind.children());
            }
        },
    )
}

/// Whether `target` is reachable from `root` through floating pure/union
/// operands only. This is the common dependency predicate for use-site and
/// fusion analyses that deliberately stop at effect results and CFG params.
pub(crate) fn pure_depends_on<P: Family>(graph: &EGraph<P>, root: ValueId, target: ValueId) -> bool {
    wyn_graph::reaches_ordered(root, target, wyn_graph::WalkOrder::DepthFirst, |node, out| {
        if let Some(definition) = graph.nodes.get(node) {
            out.extend(definition.kind.children());
        }
    })
}

/// Whether the complete value-producing closure behind `root` contains
/// `target`, crossing effect results and incoming block arguments as needed.
pub(crate) fn value_depends_on<P: ValueProducerPhase>(
    graph: &EGraph<P>,
    root: ValueId,
    target: ValueId,
) -> bool {
    value_producer_closure(graph, [root]).contains_node(target)
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
    mut movable: impl FnMut(ValueId) -> bool,
) -> Vec<ValueId> {
    let reachable = reachable_execution_values(graph);
    let reachable_set = reachable.iter().copied().collect::<HashSet<_>>();
    let movable = reachable.iter().map(|node| (*node, movable(*node))).collect::<HashMap<_, _>>();
    let mut boundary = execution_value_roots(graph).into_iter().collect::<HashSet<_>>();
    for node in &reachable {
        if movable[node] {
            continue;
        }
        if let Some(definition) = graph.nodes.get(*node) {
            boundary.extend(
                definition.kind.children().into_iter().filter(|child| reachable_set.contains(child)),
            );
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
    roots: impl IntoIterator<Item = ValueId>,
) -> Vec<SegResourceAccess<super::program::SemanticResourceRef>>
where
    P: ValueProducerPhase + Family<Resource = super::program::SemanticResourceRef>,
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
pub(crate) fn projection_index<P: Family>(
    graph: &EGraph<P>,
    node: ValueId,
    root: ValueId,
) -> Option<usize> {
    match &graph.nodes.get(node)?.kind {
        ValueKind::Pure {
            op: PureOp::Project { index },
            operands,
        } if operands.first() == Some(&root) => Some(*index as usize),
        _ => None,
    }
}

/// Follow nested projections back to `root` and return its selected output.
/// For `Project(Project(root, outer), inner)`, this is `outer`.
pub(crate) fn root_projection_index<P: Family>(
    graph: &EGraph<P>,
    node: ValueId,
    root: ValueId,
) -> Option<usize> {
    let mut current = node;
    let mut root_index = None;
    loop {
        if current == root {
            return root_index;
        }
        let ValueKind::Pure {
            op: PureOp::Project { index },
            operands,
        } = &graph.nodes.get(current)?.kind
        else {
            return None;
        };
        root_index = Some(*index as usize);
        current = *operands.first()?;
    }
}

fn extend_incoming_block_args<P: Family>(
    graph: &EGraph<P>,
    target: BlockId,
    index: usize,
    pending: &mut Vec<ValueId>,
) {
    for (_, predecessor) in &graph.skeleton.blocks {
        match &predecessor.term {
            SkeletonTerminator::Branch {
                target: branch_target,
                args,
            } if *branch_target == target => {
                pending.extend(args.get(index).map(|argument| argument.value()));
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
                    pending.extend(then_args.get(index).map(|argument| argument.value()));
                    reaches_target = true;
                }
                if *else_target == target {
                    pending.extend(else_args.get(index).map(|argument| argument.value()));
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
pub fn intern_u32<P: Family>(graph: &mut EGraph<P>, n: u32, span: Option<Span>) -> ValueId {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    graph.intern_pure(PureOp::Uint(n.to_string()), smallvec![], u32_ty, span)
}

/// Constant via `EGraph::intern_constant` (canonical `ValueKind::Constant`
/// form). Use this when the value comes through a `ConstantValue`
/// already (e.g. carrying a reduce's neutral element across passes).
/// For freshly-typed-out integer/float literals from terms, prefer the
/// `PureOp::Uint`/`Int`/`Float` form via the other helpers.
pub fn intern_constant<P: Family>(
    graph: &mut EGraph<P>,
    value: ConstantValue,
    ty: Type<TypeName>,
) -> ValueId {
    graph.intern_constant(value, ty)
}

/// Generic intrinsic call (`PureOp::Intrinsic` with `overload_idx: 0`).
pub fn intern_intrinsic<P: Family>(
    graph: &mut EGraph<P>,
    id: BuiltinId,
    operands: SmallVec<[ValueId; 4]>,
    ty: Type<TypeName>,
    span: Option<Span>,
) -> ValueId {
    graph.intern_pure(PureOp::Intrinsic { id, overload_idx: 0 }, operands, ty, span)
}

/// Binary op (`PureOp::BinOp`). `op` is the operator string (`"+"`,
/// `"-"`, etc.) — matches the convention `from_tlc` uses.
pub fn intern_binop<P: Family>(
    graph: &mut EGraph<P>,
    op: crate::op::BinaryOperator,
    lhs: ValueId,
    rhs: ValueId,
    ty: Type<TypeName>,
    span: Option<Span>,
) -> ValueId {
    graph.intern_pure(PureOp::BinOp(op), smallvec![lhs, rhs], ty, span)
}

/// `StorageView(Storage(br))` with the default
/// `[0, _w_intrinsic_storage_len(set, binding)]` operand pair.
pub fn intern_storage_view(
    graph: &mut EGraph<Physical>,
    br: BindingRef,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> ValueId {
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
pub fn intern_resource_view<P: Family<Resource = super::program::SemanticResourceRef>>(
    graph: &mut EGraph<P>,
    resource: crate::ResourceId,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> ValueId {
    let len = intern_resource_len(graph, resource, span);
    let zero = intern_u32(graph, 0, span);
    intern_chunked_resource_view(graph, resource, zero, len, view_ty, span)
}

/// Target-independent logical-resource length.
pub fn intern_resource_len<P: Family<Resource = super::program::SemanticResourceRef>>(
    graph: &mut EGraph<P>,
    resource: crate::ResourceId,
    span: Option<Span>,
) -> ValueId {
    graph.intern_pure(
        PureOp::ResourceLen(super::program::SemanticResourceRef(resource)),
        smallvec![],
        Type::Constructed(TypeName::UInt(32), vec![]),
        span,
    )
}

pub fn intern_chunked_resource_view<P: Family<Resource = super::program::SemanticResourceRef>>(
    graph: &mut EGraph<P>,
    resource: crate::ResourceId,
    offset: ValueId,
    len: ValueId,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> ValueId {
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
pub fn emit_workgroup_view<P: Family>(
    graph: &mut EGraph<P>,
    id: u32,
    count: u32,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> ValueId {
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
    offset: ValueId,
    len: ValueId,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> ValueId {
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
pub fn emit_store<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    place: PlaceId,
    value_nid: ValueId,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> EffectToken {
    let effect_in = alloc_effect(effect_ids);
    let effect_out = alloc_effect(effect_ids);
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Store { place }),
        operands: smallvec![OperandRef::Value(value_nid)],
        result: None,
        effects: Some((effect_in, effect_out)),
        span,
    });
    effect_out
}

/// Emit an atomic integer update through an addressable place. The returned
/// node is the value observed before the update.
pub fn emit_atomic<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    place: PlaceId,
    op: crate::ssa::types::AtomicOp,
    values: &[ValueId],
    result_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> ValueId {
    assert_eq!(values.len(), op.value_arity());
    let effect_in = alloc_effect(effect_ids);
    let effect_out = alloc_effect(effect_ids);
    let result = graph.alloc_side_effect_result(result_ty);
    let operands = values.iter().copied().map(OperandRef::Value).collect();
    let result_binding = graph.value_result(result);
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Atomic { place, op }),
        operands,
        result: Some(result_binding),
        effects: Some((effect_in, effect_out)),
        span,
    });
    result
}
/// Emit a workgroup execution+memory barrier
/// in `block`. No operands or result; the effect token keeps it ordered
/// against the workgroup-shared loads/stores it synchronizes. Returns the
/// produced effect-out token.
pub fn emit_workgroup_barrier<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> EffectToken {
    let effect_in = alloc_effect(effect_ids);
    let effect_out = alloc_effect(effect_ids);
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::ControlBarrier),
        operands: smallvec![],
        result: None,
        effects: Some((effect_in, effect_out)),
        span: None,
    });
    effect_out
}

/// Emit a store through a `StorageView` at `index_nid`. Builds the
/// `ViewIndex` pure node and the `Store` side-effect.
pub fn emit_storage_store<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    view_nid: ValueId,
    index_nid: ValueId,
    value_nid: ValueId,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> EffectToken {
    let view = graph.view_id(view_nid);
    let place = graph.add_view_index_place(view, index_nid, elem_ty, span);
    emit_store(graph, block, place, value_nid, effect_ids, span)
}

/// Emit a `Load` of `place_nid` (a place-producing pure op like `ViewIndex`)
/// in `block`; returns the loaded-value node (typed `elem_ty`).
pub fn emit_load<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    place: PlaceId,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> ValueId {
    let (result, effect) = detached_load(graph, place, elem_ty, LoadMode::Element, effect_ids, span);
    graph.skeleton.blocks[block].side_effects.push(effect);
    result
}

/// Construct a `Load` and its result without choosing its position in a
/// block. Rewriters use this when a synthesized load must be inserted before
/// an existing scheduled operation instead of appended to the block tail.
pub fn detached_load<P: Family>(
    graph: &mut EGraph<P>,
    place: PlaceId,
    elem_ty: Type<TypeName>,
    mode: LoadMode,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> (ValueId, SideEffect<P>) {
    let effect_in = alloc_effect(effect_ids);
    let effect_out = alloc_effect(effect_ids);
    let result = graph.alloc_side_effect_result(elem_ty);
    let effect = SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Load { place, mode }),
        operands: smallvec![],
        result: Some(graph.value_result(result)),
        effects: Some((effect_in, effect_out)),
        span,
    };
    (result, effect)
}

/// Emit a function-local `Alloca` side-effect in `block`. The returned ValueId
/// represents the allocated place — pass it to `intern_place_index` for
/// element-level addressing, or to `emit_load` / `emit_store` for whole-value
/// access. The place's element type is `elem_ty`; for an `[T;N]` allocation
/// `Load` returns the whole array and `PlaceIndex` produces `T`-typed sub-places.
pub fn emit_alloca<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> PlaceId {
    let effect_in = alloc_effect(effect_ids);
    let effect_out = alloc_effect(effect_ids);
    let place = graph.add_alloca_place(
        PlaceType {
            pointee: elem_ty,
            region: PlaceRegion::Function,
            access: PlaceAccess::ReadWrite,
        },
        span,
    );
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Alloca { result: place }),
        operands: smallvec![],
        result: None,
        effects: Some((effect_in, effect_out)),
        span,
    });
    place
}

/// Intern a `PlaceIndex` pure node: index into an existing place to produce a
/// sub-place addressing one element. The parent place can be an `Alloca`'d
/// array or any other place-producing node; the result has element type
/// `elem_ty` (e.g. `T` for an `[T;N]` parent).
pub fn intern_place_index<P: Family>(
    graph: &mut EGraph<P>,
    parent_place: PlaceId,
    index_nid: ValueId,
    elem_ty: Type<TypeName>,
    span: Option<Span>,
) -> PlaceId {
    graph.add_index_place(parent_place, index_nid, elem_ty, span)
}

/// Emit `place[index] = value` as a `PlaceIndex` sub-place + `Store` in
/// `block`. Companion to `emit_storage_store` for function-local Alloca'd
/// arrays — no whole-array `Load`/`Store` round-trip.
pub fn emit_place_index_store<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    parent_place: PlaceId,
    index_nid: ValueId,
    value_nid: ValueId,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) {
    let elem_place = intern_place_index(graph, parent_place, index_nid, elem_ty, span);
    let _ = emit_store(graph, block, elem_place, value_nid, effect_ids, span);
}

/// Emit `view[index]` as a `ViewIndex` place + `Load` in `block`; returns the
/// loaded value. Companion to `emit_storage_store`.
pub fn emit_view_load<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    view_nid: ValueId,
    index_nid: ValueId,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> ValueId {
    let view = graph.view_id(view_nid);
    let place = graph.add_view_index_place(view, index_nid, elem_ty.clone(), span);
    emit_load(graph, block, place, elem_ty, effect_ids, span)
}

/// Push a SOAC side effect into `block`. The effect owns a result tree whose
/// by-value leaves are allocated independently; the returned value is the
/// product assembled from those leaves for expression-level consumers.
pub fn emit_pending_soac<P: WynSoacPhase>(
    graph: &mut EGraph<P>,
    block: BlockId,
    id: P::SoacId,
    soac: Soac<P>,
    operands: SmallVec<[OperandRef; 4]>,
    result_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> ValueId {
    let result_abi = super::ir::by_value_function_result::<super::types::WynLanguage>(result_ty);
    let result_binding = result_abi.bind(
        |_, ty| graph.alloc_side_effect_result(ty.clone()),
        |_| unreachable!("a pending SOAC has no destination parameters"),
    );
    let result_value = pack_result_values(graph, &result_binding)
        .expect("a newly allocated by-value SOAC result can be assembled");
    let effect_in = alloc_effect(effect_ids);
    let effect_out = alloc_effect(effect_ids);
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Soac(SoacEffect(id, soac)),
        operands: operands,
        result: Some(result_binding),
        effects: Some((effect_in, effect_out)),
        span,
    });
    result_value
}

// ---------------------------------------------------------------------------
// Read-side inspection
// ---------------------------------------------------------------------------

/// Return the semantic identity carried by a storage-view node.
pub fn extract_storage_view_source<P: Family<Resource = super::program::SemanticResourceRef>>(
    graph: &EGraph<P>,
    view_nid: ValueId,
) -> Option<super::program::SemanticResourceRef> {
    match &graph.nodes[view_nid].kind {
        ValueKind::Pure {
            op: PureOp::StorageView(PureViewSource::Storage(resource)),
            ..
        } => Some(*resource),
        _ => None,
    }
}

/// Find the storage resource beneath a semantic place expression.
pub(crate) fn storage_resource_under<P: Family<Resource = super::program::SemanticResourceRef>>(
    graph: &EGraph<P>,
    root: ValueId,
) -> Option<super::program::SemanticResourceRef> {
    wyn_graph::find_map_reachable(
        [root],
        wyn_graph::WalkOrder::DepthFirst,
        |node, out| {
            if let Some(value) = graph.nodes.get(node) {
                out.extend(value.kind.children());
            }
        },
        |node| extract_storage_view_source(graph, node),
    )
}

/// If `nid` is a `PureOp::ArrayRange`, return `(start, len, step?)`
/// ValueNodeIds. Otherwise `None`.
pub fn extract_array_range_operands<P: Family>(
    graph: &EGraph<P>,
    nid: ValueId,
) -> Option<(ValueId, ValueId, Option<ValueId>)> {
    match &graph.nodes[nid].kind {
        ValueKind::Pure {
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
/// `dst`, returning the new root `ValueId`. Copies a reduce's neutral
/// element (or any pure value) from one entry's EGraph into another's —
/// phase2 needs a fresh copy of phase1's NE since EGraph ValueNodeIds don't
/// cross entries.
///
/// Only pure nodes and constants are cloned; encountering a
/// `SideEffectResult` or a `BlockParam` returns `Err` because those
/// reference cross-block / cross-effect data that doesn't translate.
pub fn clone_pure_subgraph<P: Family>(
    src: &EGraph<P>,
    dst: &mut EGraph<P>,
    root: ValueId,
) -> Result<ValueId, String> {
    let mut memo: LookupMap<ValueId, ValueId> = LookupMap::new();
    clone_value_subgraph(
        src,
        dst,
        root,
        &mut memo,
        ConstantCopy::Intern,
        false,
        PureCopy::Preserve,
    )
}

/// Clone an addressable place and the pure value/place dependencies that
/// define its address into another graph.
pub fn clone_place_subgraph<P: Family>(
    src: &EGraph<P>,
    dst: &mut EGraph<P>,
    root: PlaceId,
) -> Result<PlaceId, String> {
    fn clone_place<P: Family>(
        src: &EGraph<P>,
        dst: &mut EGraph<P>,
        source: PlaceId,
        values: &mut LookupMap<ValueId, ValueId>,
        places: &mut LookupMap<PlaceId, PlaceId>,
    ) -> Result<PlaceId, String> {
        if let Some(&target) = places.get(&source) {
            return Ok(target);
        }
        let place = src
            .places
            .get(source)
            .ok_or_else(|| format!("clone_place_subgraph: missing place {source:?}"))?
            .clone();
        let ty = place.ty().clone();
        let span = place.span();
        let clone_value = |value, dst: &mut EGraph<P>, values: &mut LookupMap<ValueId, ValueId>| {
            clone_value_subgraph(
                src,
                dst,
                value,
                values,
                ConstantCopy::Intern,
                false,
                PureCopy::Preserve,
            )
        };
        let target = match place.op() {
            PlaceOp::Parameter { parameter } => dst.add_place_parameter(*parameter, ty),
            PlaceOp::AllocaResult => dst.add_alloca_place(ty, span),
            PlaceOp::Index { base, index } => {
                let base = clone_place(src, dst, *base, values, places)?;
                let index = clone_value(*index, dst, values)?;
                dst.add_index_place(base, index, ty.pointee, span)
            }
            PlaceOp::ViewIndex { view, index } => {
                let view = clone_value(view.value(), dst, values)?;
                let index = clone_value(*index, dst, values)?;
                let view = dst.view_id(view);
                dst.add_view_index_place(view, index, ty.pointee, span)
            }
            PlaceOp::OutputSlot { index } => dst.add_output_place(*index, ty),
        };
        places.insert(source, target);
        Ok(target)
    }

    clone_place(src, dst, root, &mut LookupMap::new(), &mut LookupMap::new())
}

/// Clone one typed boundary operand without collapsing its value, view, or
/// place representation.
pub fn clone_operand_subgraph<P: Family>(
    src: &EGraph<P>,
    dst: &mut EGraph<P>,
    operand: OperandRef,
) -> Result<OperandRef, String> {
    Ok(match operand {
        OperandRef::Value(value) => OperandRef::Value(clone_pure_subgraph(src, dst, value)?),
        OperandRef::View(view) => {
            let value = clone_pure_subgraph(src, dst, view.value())?;
            OperandRef::View(dst.view_id(value))
        }
        OperandRef::Place(place) => OperandRef::Place(clone_place_subgraph(src, dst, place)?),
    })
}

/// Clone a pure subgraph of `src` into `dst`, but substitute the given `src`
/// nodes for already-existing `dst` nodes: any `(from, to)` pre-seeds the clone
/// memo, so a reference to `from` in `src` becomes `to` in `dst`. Lets a value
/// rooted at a non-pure node (e.g. a SOAC result) be re-expressed over a
/// replacement `dst` value without rebuilding its projection structure by hand.
pub fn clone_pure_subgraph_substituting<P: Family>(
    src: &EGraph<P>,
    dst: &mut EGraph<P>,
    root: ValueId,
    subs: &[(ValueId, ValueId)],
) -> Result<ValueId, String> {
    let mut memo: LookupMap<ValueId, ValueId> = subs.iter().copied().collect();
    clone_value_subgraph(
        src,
        dst,
        root,
        &mut memo,
        ConstantCopy::Intern,
        false,
        PureCopy::Preserve,
    )
}

#[derive(Clone, Copy)]
pub(crate) enum ConstantCopy {
    Intern,
    PreserveIdentity,
}

#[derive(Clone, Copy)]
pub(crate) enum PureCopy {
    /// Reproduce the source DAG exactly apart from hash-consing.
    Preserve,
    /// Re-run algebraic folds after operands have been substituted.
    Fold,
}

pub(crate) fn clone_value_subgraph<P: Family>(
    src: &EGraph<P>,
    dst: &mut EGraph<P>,
    nid: ValueId,
    memo: &mut LookupMap<ValueId, ValueId>,
    constants: ConstantCopy,
    allow_unions: bool,
    pure: PureCopy,
) -> Result<ValueId, String> {
    if let Some(&existing) = memo.get(&nid) {
        return Ok(existing);
    }
    let source = src.nodes.get(nid).ok_or_else(|| format!("clone_value_subgraph: missing node {nid:?}"))?;
    let ty = source.ty.clone();
    let new_nid = match &source.kind {
        ValueKind::Constant(c) => match constants {
            ConstantCopy::Intern => dst.intern_constant(*c, ty),
            ConstantCopy::PreserveIdentity => {
                let target = dst.nodes.insert(Value {
                    kind: ValueKind::Constant(*c),
                    ty,
                    span: source.span,
                    alias: None,
                });
                target
            }
        },
        ValueKind::Pure { op, operands, .. } => {
            let new_ops: SmallVec<[ValueId; 4]> = operands
                .iter()
                .map(|&operand| {
                    clone_value_subgraph(src, dst, operand, memo, constants, allow_unions, pure)
                })
                .collect::<Result<_, _>>()?;
            if matches!(pure, PureCopy::Fold) {
                if let Some(folded) = dst.try_algebraic_fold(op, &new_ops, &ty) {
                    folded
                } else {
                    dst.intern_pure(op.clone(), new_ops, ty, source.span)
                }
            } else {
                dst.intern_pure(op.clone(), new_ops, ty, source.span)
            }
        }
        ValueKind::Union { left, right } if allow_unions => {
            let left = clone_value_subgraph(src, dst, *left, memo, constants, allow_unions, pure)?;
            let right = clone_value_subgraph(src, dst, *right, memo, constants, allow_unions, pure)?;
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
pub fn replace_all_references(graph: &mut EGraph<Semantic>, old: ValueId, new: ValueId) {
    if old == new {
        return;
    }
    let swap = |value: ValueId| if value == old { new } else { value };
    graph.replace_node_references(old, new);
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            effect.remap_referenced_values(swap);
        }
        block.term.visit_values_mut(|slot| *slot = swap(*slot));
    }
}
