//! Phase-agnostic, low-level data structures for EGIR.

use slotmap::{new_key_type, SlotMap};
use smallvec::SmallVec;

use crate::ast::Span;
use crate::flow::{BlockId, ControlHeader, ExecutionModel};
use crate::interface::{EntryInput as InterfaceEntryInput, EntryOutput as InterfaceEntryOutput};
use crate::op::OpTag;
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::types::ExternDecl;
use crate::{LookupMap, SortedSet};

pub use crate::op::PureViewSource;
pub use crate::types::SoacOwnership;

use super::soac::{filter, hist, screma};

/// Effect token for ordering effectful ops during EGIR passes.
///
/// These are purely an EGIR-internal concept — they never reach the SSA
/// backend. `elaborate` emits instructions in skeleton block order and
/// doesn't pass the tokens through. The token chain only exists to support
/// rewriting passes (e.g. `soac_expand` allocating fresh tokens for new
/// Load/Store side-effects so they don't collide with existing ones).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectToken(u32);

impl From<u32> for EffectToken {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl std::fmt::Display for EffectToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "!{}", self.0)
    }
}

new_key_type! {
    /// Identity of a node in the e-graph. Every pure node, union node,
    /// block param, function param, and constant gets one.
    pub struct NodeId;

}

/// Opaque handle into the program-level region arena (`SemanticProgram::regions`).
///
/// Region *identity* is a checked arena index, never a re-derived string. A
/// region still lowers to a named SSA function — that name is the call ABI and
/// lives on the callable `Func`/the `RegionInterner`, recovered via the arena when a
/// `PureOp::Call` is emitted. Semantic SegOps carry only this index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegionId(u32);

impl RegionId {
    #[cfg(test)]
    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }

    #[cfg(test)]
    pub const fn index(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for RegionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Name ↔ arena-index interner for callable regions.
///
/// Region identity is the assigned `RegionId` (a dense index). The textual
/// name is retained because it is the SSA `Call` ABI. Interning the same name
/// twice returns the same index, so segmented-body construction and the
/// function arena agree without a separate resolution pass.
impl From<u32> for RegionId {
    fn from(index: u32) -> Self {
        Self(index)
    }
}

pub type RegionInterner = crate::Interner<RegionId, String>;

// ---------------------------------------------------------------------------
// PureOp — operator identity for hash-consing
// ---------------------------------------------------------------------------

/// The type and literal payloads stored by a core IR graph.
pub trait Language: Clone + std::fmt::Debug + Eq + std::hash::Hash {
    type Const: Clone + std::fmt::Debug + Eq + std::hash::Hash;
    type Ty: Clone + std::fmt::Debug + Eq + std::hash::Hash;
}

/// Phase-typed operator identity without operands.
pub type PureOp<R> = OpTag<R>;

// ---------------------------------------------------------------------------
// NodeKey — hash-cons key = operator + operands + result type
// ---------------------------------------------------------------------------

/// The full identity of a pure node for hash-consing: operator, operands
/// (already-canonical `NodeId`s), and result type. `ty` is part of the
/// key because two otherwise-equal pure ops with different result types
/// are semantically different values — e.g.
/// `_w_intrinsic_storage_len(0, 0)` can be retyped at a rewrite site
/// from its registered `u32` to a caller-required `i32`, and collapsing
/// those two interns into one node would silently let the first-inserted
/// type win at the merged site. Mirrors the 3b8cb24 fix that split
/// `PureOp::Int` / `PureOp::Uint` tags for literals, but uniformly for
/// every pure op.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeKey<R, Lang: Language> {
    pub op: PureOp<R>,
    pub operands: SmallVec<[NodeId; 4]>,
    pub ty: Lang::Ty,
}

// ---------------------------------------------------------------------------
// ENode — what lives in the sea of nodes
// ---------------------------------------------------------------------------

/// A node in the e-graph.
#[derive(Clone, Debug)]
pub enum ENode<R, Lang: Language> {
    /// A pure instruction, hash-consed and floating.
    Pure {
        op: PureOp<R>,
        operands: SmallVec<[NodeId; 4]>,
    },
    /// Union of two equivalent representations (binary tree of eclasses).
    Union {
        left: NodeId,
        right: NodeId,
    },
    /// Function parameter.
    FuncParam {
        index: usize,
    },
    /// Block parameter (merge point in CFG skeleton).
    BlockParam {
        block: BlockId,
        index: usize,
    },
    /// Inline constant value.
    Constant(Lang::Const),
    /// Side-effect result — a value produced by an effectful instruction
    /// in the skeleton. Not hash-consed; each is unique.
    SideEffectResult,
}

impl<R, Lang: Language> ENode<R, Lang> {
    /// Return all child NodeIds referenced by this node.
    pub fn children(&self) -> SmallVec<[NodeId; 4]> {
        match self {
            ENode::Pure { operands, .. } => operands.clone(),
            ENode::Union { left, right } => smallvec::smallvec![*left, *right],
            ENode::FuncParam { .. }
            | ENode::BlockParam { .. }
            | ENode::Constant(_)
            | ENode::SideEffectResult => SmallVec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Skeleton — the CFG of side-effectful instructions
// ---------------------------------------------------------------------------

/// A side effect anchored in the skeleton CFG.
#[derive(Clone, Debug)]
pub struct SideEffect<P: EgirPhase, Lang: Language> {
    pub kind: SideEffectKind<P, Lang>,
    /// Canonical EGIR value operands for this effect.
    pub operand_nodes: SmallVec<[NodeId; 4]>,
    /// Result value, if this effect produces one. Addressable-place results
    /// are carried by the corresponding `EffectOp` variant instead.
    pub result: Option<NodeId>,
    /// Effect token chain.
    pub effects: Option<(EffectToken, EffectToken)>,
    /// Source span of the user expression that produced this side-effect,
    /// or `None` for synthesized side-effects (e.g. SOAC expansion).
    pub span: Option<Span>,
}

/// EGIR-native effect operation. Value and place operands are represented by
/// the enclosing side effect's `NodeId` operands; SSA identities are
/// introduced only when the graph is elaborated.
#[derive(Clone, Debug)]
pub enum EffectOp<R, Ty> {
    Op {
        tag: OpTag<R>,
    },
    Alloca {
        elem_ty: Ty,
    },
    Load,
    Store,
    ControlBarrier,
}

impl<R, Ty> EffectOp<R, Ty> {
    pub fn try_map_resource<S, E>(
        self,
        map: &mut impl FnMut(R) -> Result<S, E>,
    ) -> Result<EffectOp<S, Ty>, E> {
        Ok(match self {
            Self::Op { tag } => EffectOp::Op {
                tag: tag.try_map_resource(map)?,
            },
            Self::Alloca { elem_ty } => EffectOp::Alloca { elem_ty },
            Self::Load => EffectOp::Load,
            Self::Store => EffectOp::Store,
            Self::ControlBarrier => EffectOp::ControlBarrier,
        })
    }
}

/// A skeleton side effect's concrete kind.
#[derive(Clone, Debug)]
pub enum SideEffectKind<P: EgirPhase, Lang: Language> {
    Effect(EffectOp<P::Resource, Lang::Ty>),
    /// A placeholder for an unexpanded SOAC. Produced by `from_tlc` and
    /// consumed by `soac_expand`. Never reaches elaborate.
    Soac(P::SoacId, Soac<P>),
}

impl<P: EgirPhase, Lang: Language> SideEffectKind<P, Lang> {
    pub fn soac_id(&self) -> Option<&P::SoacId> {
        match self {
            Self::Effect(_) => None,
            Self::Soac(id, _) => Some(id),
        }
    }
}

/// External storage selected for a SOAC result by EGIR lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoacPlacement {
    InputBuffer,
    OutputView,
}

/// Logical ownership plus any external placement selected during lowering.
/// An unplaced result is function-local; `InputBuffer` aliases an input and
/// `OutputView` consumes an explicit output-view operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoacDestination {
    pub ownership: SoacOwnership,
    pub placement: Option<SoacPlacement>,
}

impl SoacDestination {
    pub const fn fresh() -> Self {
        Self {
            ownership: SoacOwnership::Fresh,
            placement: None,
        }
    }

    pub const fn unique_input() -> Self {
        Self {
            ownership: SoacOwnership::UniqueInput,
            placement: None,
        }
    }

    pub const fn placed(self, placement: SoacPlacement) -> Self {
        Self {
            placement: Some(placement),
            ..self
        }
    }

    pub fn place(&mut self, placement: SoacPlacement) {
        self.placement = Some(placement);
    }

    pub fn make_fresh(&mut self) {
        *self = Self::fresh();
    }

    pub const fn is_unplaced_fresh(self) -> bool {
        matches!(self.ownership, SoacOwnership::Fresh) && self.placement.is_none()
    }

    pub const fn is_unplaced(self) -> bool {
        self.placement.is_none()
    }

    pub const fn is_unplaced_unique_input(self) -> bool {
        matches!(self.ownership, SoacOwnership::UniqueInput) && self.placement.is_none()
    }

    pub const fn is_input_buffer(self) -> bool {
        matches!(self.placement, Some(SoacPlacement::InputBuffer))
    }

    pub const fn is_output_view(self) -> bool {
        matches!(self.placement, Some(SoacPlacement::OutputView))
    }
}

/// One concrete dimension of a segmented iteration space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SegExtent<R> {
    Fixed(u32),
    PushConstant {
        node: NodeId,
        offset: u32,
    },
    ResourceLength {
        node: NodeId,
        resource: R,
        elem_bytes: u32,
    },
    /// A concrete EGIR value whose provenance is not host-dispatchable. Such
    /// spaces remain valid for lane-local/serial lowering.
    Value(NodeId),
}

/// The parallel iteration space of a `Seg` op. wyn is 1-D: a flat global
/// thread index ranging over `len` elements. The thread index node itself is
/// bound during expansion (`build_parallel_maps`/`chunk_soac_inputs`), not at
/// node-construction time.
#[derive(Clone, Debug)]
pub struct SegSpace<R> {
    pub dims: Vec<SegExtent<R>>,
}

impl<R> SegSpace<R> {
    pub(crate) fn referenced_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.dims.iter().filter_map(|extent| match extent {
            SegExtent::PushConstant { node, .. }
            | SegExtent::ResourceLength { node, .. }
            | SegExtent::Value(node) => Some(*node),
            SegExtent::Fixed(_) => None,
        })
    }

    pub(crate) fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        self.dims
            .iter_mut()
            .filter_map(|extent| match extent {
                SegExtent::PushConstant { node, .. }
                | SegExtent::ResourceLength { node, .. }
                | SegExtent::Value(node) => Some(node),
                SegExtent::Fixed(_) => None,
            })
            .collect()
    }
}

/// A complete callable body and the values captured from its surrounding
/// graph. Captures are explicit values, never an operand-count convention.
#[derive(Clone, Debug)]
pub struct SegBody {
    pub region: RegionId,
    pub captures: Vec<NodeId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegResourceAccess<R> {
    pub resource: R,
    pub access: crate::ResourceAccess,
}

impl<R: Copy + Ord> SegResourceAccess<R> {
    pub fn merge(a: &[Self], b: &[Self]) -> Vec<Self> {
        let mut merged: std::collections::BTreeMap<R, crate::ResourceAccess> =
            std::collections::BTreeMap::new();
        for resource in a.iter().chain(b) {
            merged
                .entry(resource.resource)
                .and_modify(|access| *access = access.merge(resource.access))
                .or_insert(resource.access);
        }
        merged.into_iter().map(|(resource, access)| Self { resource, access }).collect()
    }
}

pub trait EgirPhase: Clone + std::fmt::Debug {
    type Resource: GraphResource;
    type ResourceDecl: Clone + std::fmt::Debug;
    type SoacId: Clone + std::fmt::Debug;
    type MapState: Clone + std::fmt::Debug;
    type ReduceState: Clone + std::fmt::Debug;
    type ScanState: Clone + std::fmt::Debug;
    type CompositeState: Clone + std::fmt::Debug;
    type FilterState: Clone + std::fmt::Debug;
    type HistState: Clone + std::fmt::Debug;
}

#[derive(Clone, Debug)]
pub struct SoacInputType<Ty> {
    pub array: Ty,
}

#[derive(Clone, Debug)]
pub enum Soac<P: EgirPhase> {
    Screma(screma::Op<P>),
    Filter(filter::Op<P>),
    Hist(hist::Op<P>),
}

impl<P: EgirPhase> Soac<P> {
    pub(crate) fn seg_bodies(&self) -> Vec<&SegBody> {
        match self {
            Self::Screma(op) => {
                let mut bodies = op.lanes().maps.iter().map(|map| &map.body).collect::<Vec<_>>();
                for operator in op.operators() {
                    bodies.extend([&operator.step, &operator.combine]);
                }
                bodies
            }
            Self::Filter(op) => {
                let mut bodies = Vec::with_capacity(2);
                if let filter::Input::Mapped { body, .. } = &op.body.input {
                    bodies.push(body);
                }
                bodies.push(&op.body.predicate);
                bodies
            }
            Self::Hist(op) => vec![&op.body.body],
        }
    }
}

/// Terminator using NodeIds for value references.
pub type SkeletonTerminator = crate::flow::Terminator<NodeId>;

/// A block in the skeleton CFG.
#[derive(Clone, Debug)]
pub struct SkeletonBlock<P: EgirPhase, Lang: Language> {
    /// Block parameters as NodeIds.
    pub params: Vec<NodeId>,
    /// Effectful instructions, in order.
    pub side_effects: Vec<SideEffect<P, Lang>>,
    /// Block terminator.
    pub term: SkeletonTerminator,
}

impl<P: EgirPhase, Lang: Language> SkeletonBlock<P, Lang> {
    pub fn new() -> Self {
        SkeletonBlock {
            params: Vec::new(),
            side_effects: Vec::new(),
            term: SkeletonTerminator::Unreachable,
        }
    }
}

/// The skeleton CFG (blocks + effectful instructions).
#[derive(Clone, Debug)]
pub struct Skeleton<P: EgirPhase, Lang: Language> {
    pub entry: BlockId,
    pub blocks: SlotMap<BlockId, SkeletonBlock<P, Lang>>,
}

impl<P: EgirPhase, Lang: Language> Skeleton<P, Lang> {
    pub fn new() -> Self {
        let mut blocks = SlotMap::with_key();
        let entry = blocks.insert(SkeletonBlock::new());
        Skeleton { entry, blocks }
    }

    pub fn create_block(&mut self) -> BlockId {
        self.blocks.insert(SkeletonBlock::new())
    }

    /// Split a block immediately before one of its side effects.
    ///
    /// The returned continuation receives the selected effect and every
    /// following effect, plus the original terminator. The original block is
    /// terminated by an unconditional branch to the continuation.
    pub fn split_block_before_effect(&mut self, block: BlockId, effect_index: usize) -> BlockId {
        let continuation = self.create_block();
        let source = &mut self.blocks[block];
        let suffix = source.side_effects.split_off(effect_index);
        let old_term = std::mem::replace(
            &mut source.term,
            SkeletonTerminator::Branch {
                target: continuation,
                args: Vec::new(),
            },
        );
        self.blocks[continuation].side_effects = suffix;
        self.blocks[continuation].term = old_term;
        continuation
    }
}

/// Stable-for-a-snapshot location of a side effect in the skeleton.
///
/// Side effects are still stored in ordered per-block vectors, so a site must
/// not outlive an insertion/removal/reorder in those vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SideEffectSite {
    pub block: BlockId,
    pub index: usize,
}

/// Read-side index from every `SideEffectResult` node to its producer.
///
/// Build this once for a graph snapshot and share it across related queries.
/// Rebuild it after any structural skeleton mutation.
pub struct SideEffectIndex {
    by_result: LookupMap<NodeId, SideEffectSite>,
}

impl SideEffectIndex {
    pub fn build<P: EgirPhase, Lang: Language>(graph: &EGraph<P, Lang>) -> Self {
        let mut by_result = LookupMap::new();
        for (block, skeleton_block) in &graph.skeleton.blocks {
            for (index, effect) in skeleton_block.side_effects.iter().enumerate() {
                let Some(result) = effect.result else {
                    continue;
                };
                let previous = by_result.insert(result, SideEffectSite { block, index });
                assert!(
                    previous.is_none(),
                    "side-effect result has more than one producer: {result:?}"
                );
            }
        }
        Self { by_result }
    }

    pub fn site(&self, result: NodeId) -> Option<SideEffectSite> {
        self.by_result.get(&result).copied()
    }

    pub fn effect<'a, P: EgirPhase, Lang: Language>(
        &self,
        graph: &'a EGraph<P, Lang>,
        result: NodeId,
    ) -> Option<&'a SideEffect<P, Lang>> {
        let site = self.site(result)?;
        let effect = graph.skeleton.blocks.get(site.block)?.side_effects.get(site.index)?;
        (effect.result == Some(result)).then_some(effect)
    }

    pub fn effect_mut<'a, P: EgirPhase, Lang: Language>(
        &self,
        graph: &'a mut EGraph<P, Lang>,
        result: NodeId,
    ) -> Option<&'a mut SideEffect<P, Lang>> {
        let site = self.site(result)?;
        let effect = graph.skeleton.blocks.get_mut(site.block)?.side_effects.get_mut(site.index)?;
        (effect.result == Some(result)).then_some(effect)
    }
}

// ---------------------------------------------------------------------------
// EGraph — the main container
// ---------------------------------------------------------------------------

/// The acyclic e-graph: a sea of pure nodes + a CFG skeleton of side effects.
#[derive(Clone, Debug)]
pub struct EGraph<P: EgirPhase, Lang: Language> {
    /// All nodes (pure, union, params, constants, side-effect results).
    pub nodes: SlotMap<NodeId, ENode<P::Resource, Lang>>,
    /// Type of each node's result.
    pub types: LookupMap<NodeId, Lang::Ty>,
    /// Hash-cons table: NodeKey → existing NodeId.
    hash_cons: LookupMap<NodeKey<P::Resource, Lang>, NodeId>,
    /// Constant dedup cache.
    const_cache: LookupMap<Lang::Const, NodeId>,
    /// The CFG skeleton.
    pub skeleton: Skeleton<P, Lang>,
    /// Source span associated with each pure node (first-writer-wins —
    /// later interns of the same hash-consed node keep the original span).
    pub node_spans: LookupMap<NodeId, Span>,
}

/// Graph state excluding indexes derived from that state.
///
/// Transformations may consume and rebuild an `EGraph` through this boundary
/// without gaining direct access to its hash-consing internals.
pub(super) struct EGraphParts<P: EgirPhase, Lang: Language> {
    pub(super) nodes: SlotMap<NodeId, ENode<P::Resource, Lang>>,
    pub(super) types: LookupMap<NodeId, Lang::Ty>,
    pub(super) skeleton: Skeleton<P, Lang>,
    pub(super) node_spans: LookupMap<NodeId, Span>,
}

pub trait GraphResource: Clone + std::fmt::Debug + Eq + std::hash::Hash {}

impl<T> GraphResource for T where T: Clone + std::fmt::Debug + Eq + std::hash::Hash {}

impl<P: EgirPhase, Lang: Language> EGraph<P, Lang> {
    pub fn new() -> Self {
        EGraph {
            nodes: SlotMap::with_key(),
            types: LookupMap::new(),
            hash_cons: LookupMap::new(),
            const_cache: LookupMap::new(),
            skeleton: Skeleton::new(),
            node_spans: LookupMap::new(),
        }
    }

    pub(super) fn into_parts(self) -> EGraphParts<P, Lang> {
        let Self {
            nodes,
            types,
            hash_cons: _,
            const_cache: _,
            skeleton,
            node_spans,
        } = self;
        EGraphParts {
            nodes,
            types,
            skeleton,
            node_spans,
        }
    }

    pub(super) fn from_parts(parts: EGraphParts<P, Lang>) -> Self {
        let EGraphParts {
            nodes,
            types,
            skeleton,
            node_spans,
        } = parts;
        let mut graph = Self {
            nodes,
            types,
            hash_cons: LookupMap::new(),
            const_cache: LookupMap::new(),
            skeleton,
            node_spans,
        };
        graph.rebuild_hash_cons();
        graph.rebuild_const_cache();
        graph
    }

    pub fn side_effect_index(&self) -> SideEffectIndex {
        SideEffectIndex::build(self)
    }

    fn pure_node_key(&self, id: NodeId) -> Option<NodeKey<P::Resource, Lang>> {
        let ENode::Pure { op, operands } = self.nodes.get(id)? else {
            return None;
        };
        Some(NodeKey {
            op: op.clone(),
            operands: operands.clone(),
            ty: self.types.get(&id)?.clone(),
        })
    }

    fn unindex_current_pure(&mut self, id: NodeId) {
        let Some(key) = self.pure_node_key(id) else {
            return;
        };
        if self.hash_cons.get(&key) == Some(&id) {
            self.hash_cons.remove(&key);
        }
    }

    fn index_current_pure(&mut self, id: NodeId) {
        let Some(key) = self.pure_node_key(id) else {
            return;
        };
        self.hash_cons.entry(key).or_insert(id);
    }

    /// Replace a node in place without changing its result type, keeping the
    /// pure-node hash-cons table consistent across the mutation.
    pub fn replace_node_preserving_type(&mut self, id: NodeId, node: ENode<P::Resource, Lang>) {
        self.unindex_current_pure(id);
        self.nodes[id] = node;
        self.index_current_pure(id);
    }

    /// Replace a pure node's operator and operands without changing its result
    /// type, keeping the hash-cons table consistent across the mutation.
    pub fn replace_pure_node(
        &mut self,
        id: NodeId,
        op: PureOp<P::Resource>,
        operands: SmallVec<[NodeId; 4]>,
    ) {
        self.replace_node_preserving_type(id, ENode::Pure { op, operands });
    }

    /// Mutate a pure node's operator and operands in place while maintaining
    /// the hash-cons table. Returns false if `id` is not a pure node.
    pub fn update_pure_node<F>(&mut self, id: NodeId, update: F) -> bool
    where
        F: FnOnce(&mut PureOp<P::Resource>, &mut SmallVec<[NodeId; 4]>),
    {
        if !matches!(self.nodes.get(id), Some(ENode::Pure { .. })) {
            return false;
        }
        self.unindex_current_pure(id);
        if let ENode::Pure { op, operands } = &mut self.nodes[id] {
            update(op, operands);
        }
        self.index_current_pure(id);
        true
    }

    /// Change a node's result type while maintaining the pure-node hash-cons
    /// key when the node is hash-consed.
    pub fn retype_node(&mut self, id: NodeId, ty: Lang::Ty) {
        self.unindex_current_pure(id);
        self.types.insert(id, ty);
        self.index_current_pure(id);
    }

    /// Remove a function-parameter node and its graph-owned metadata.
    pub fn remove_func_param(&mut self, id: NodeId) -> bool {
        if !matches!(self.nodes.get(id), Some(ENode::FuncParam { .. })) {
            return false;
        }
        self.types.remove(&id);
        self.node_spans.remove(&id);
        self.nodes.remove(id).is_some()
    }

    /// Replace references inside graph-owned nodes. Skeleton side-effect
    /// operands are handled by higher-level graph rewriting helpers.
    pub fn replace_node_references(&mut self, old: NodeId, new: NodeId) {
        if old == new {
            return;
        }

        let ids: Vec<NodeId> = self.nodes.keys().collect();
        for id in ids {
            match self.nodes.get(id) {
                Some(ENode::Pure { operands, .. }) if operands.contains(&old) => {
                    self.update_pure_node(id, |_, operands| {
                        for operand in operands {
                            if *operand == old {
                                *operand = new;
                            }
                        }
                    });
                }
                Some(ENode::Union { .. }) => {
                    if let ENode::Union { left, right } = &mut self.nodes[id] {
                        if *left == old {
                            *left = new;
                        }
                        if *right == old {
                            *right = new;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Rebuild the pure-node hash-cons table after a bulk rewrite that may
    /// have changed pure node operands, operators, or result types in place.
    pub fn rebuild_hash_cons(&mut self) {
        let mut rebuilt = LookupMap::new();
        for (id, node) in self.nodes.iter() {
            if matches!(node, ENode::Pure { .. }) {
                if let Some(key) = self.pure_node_key(id) {
                    rebuilt.entry(key).or_insert(id);
                }
            }
        }
        self.hash_cons = rebuilt;
    }

    fn rebuild_const_cache(&mut self) {
        self.const_cache = self
            .nodes
            .iter()
            .filter_map(|(id, node)| match node {
                ENode::Constant(value) => Some((value.clone(), id)),
                _ => None,
            })
            .collect();
    }

    /// Check that every hash-cons entry points to a pure node matching its key
    /// and that every current pure-node key is represented in the table.
    pub fn verify_hash_cons(&self) -> Result<(), String> {
        for (key, &id) in &self.hash_cons {
            let Some(current) = self.pure_node_key(id) else {
                return Err(format!(
                    "hash_cons key {:?} points to non-pure node {:?}",
                    key, id
                ));
            };
            if &current != key {
                return Err(format!(
                    "hash_cons key {:?} points to node {:?} with current key {:?}",
                    key, id, current
                ));
            }
        }

        for (id, node) in self.nodes.iter() {
            if matches!(node, ENode::Pure { .. }) {
                let Some(key) = self.pure_node_key(id) else {
                    return Err(format!("pure node {:?} has no type", id));
                };
                match self.hash_cons.get(&key) {
                    Some(&indexed) if self.pure_node_key(indexed).as_ref() == Some(&key) => {}
                    Some(&indexed) => {
                        return Err(format!(
                            "pure node {:?} key {:?} is represented by stale node {:?}",
                            id, key, indexed
                        ));
                    }
                    None => {
                        return Err(format!(
                            "pure node {:?} key {:?} is missing from hash_cons",
                            id, key
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Allocate a function parameter node.
    pub fn add_func_param(&mut self, index: usize, ty: Lang::Ty) -> NodeId {
        let id = self.nodes.insert(ENode::FuncParam { index });
        self.types.insert(id, ty);
        id
    }

    /// Append a parameter to a block and allocate its corresponding node.
    pub fn add_block_param(&mut self, block: BlockId, ty: Lang::Ty) -> NodeId {
        let index = self.skeleton.blocks[block].params.len();
        let id = self.nodes.insert(ENode::BlockParam { block, index });
        self.types.insert(id, ty);
        self.skeleton.blocks[block].params.push(id);
        id
    }

    /// Remove parameter slots from a block and from every incoming branch.
    ///
    /// Removed parameter nodes remain in the node sea so a caller can alias
    /// their uses before a later cleanup. Surviving parameter nodes are
    /// renumbered to match their new positions in the block parameter list.
    /// Returns the removed nodes in ascending order of their former slots.
    pub fn remove_block_param_slots(&mut self, block: BlockId, slots: &SortedSet<usize>) -> Vec<NodeId> {
        let param_count = self.skeleton.blocks[block].params.len();
        assert!(
            slots.iter().all(|&slot| slot < param_count),
            "block parameter slot out of bounds"
        );

        let removed = slots.iter().map(|&slot| self.skeleton.blocks[block].params[slot]).collect();

        for &slot in slots.iter().rev() {
            self.skeleton.blocks[block].params.remove(slot);
        }

        for (_, predecessor) in self.skeleton.blocks.iter_mut() {
            match &mut predecessor.term {
                SkeletonTerminator::Branch { target, args } if *target == block => {
                    for &slot in slots.iter().rev() {
                        args.remove(slot);
                    }
                }
                SkeletonTerminator::CondBranch {
                    then_target,
                    then_args,
                    else_target,
                    else_args,
                    ..
                } => {
                    if *then_target == block {
                        for &slot in slots.iter().rev() {
                            then_args.remove(slot);
                        }
                    }
                    if *else_target == block {
                        for &slot in slots.iter().rev() {
                            else_args.remove(slot);
                        }
                    }
                }
                _ => {}
            }
        }

        let surviving_params = self.skeleton.blocks[block].params.clone();
        for (index, param) in surviving_params.into_iter().enumerate() {
            match &mut self.nodes[param] {
                ENode::BlockParam {
                    block: owner,
                    index: old_index,
                } if *owner == block => *old_index = index,
                _ => panic!("block parameter list contains a mismatched node"),
            }
        }

        removed
    }

    /// Intern a constant, deduplicating.
    pub fn intern_constant(&mut self, c: Lang::Const, ty: Lang::Ty) -> NodeId {
        if let Some(&existing) = self.const_cache.get(&c) {
            return existing;
        }
        let id = self.nodes.insert(ENode::Constant(c.clone()));
        self.types.insert(id, ty);
        self.const_cache.insert(c, id);
        id
    }

    /// Intern a pure node with an attached source span. The span is recorded
    /// on first intern; subsequent interns of an equivalent hash-consed node
    /// keep the original span.
    pub fn intern_pure(
        &mut self,
        op: PureOp<P::Resource>,
        operands: SmallVec<[NodeId; 4]>,
        ty: Lang::Ty,
        span: Option<Span>,
    ) -> NodeId {
        let key = NodeKey {
            op: op.clone(),
            operands: operands.clone(),
            ty: ty.clone(),
        };
        if let Some(&existing) = self.hash_cons.get(&key) {
            return existing;
        }
        let id = self.nodes.insert(ENode::Pure { op, operands });
        self.types.insert(id, ty);
        self.hash_cons.insert(key, id);
        if let Some(s) = span {
            self.node_spans.insert(id, s);
        }
        id
    }

    /// Allocate a node for a side-effect result (not hash-consed).
    pub fn alloc_side_effect_result(&mut self, ty: Lang::Ty) -> NodeId {
        let id = self.nodes.insert(ENode::SideEffectResult);
        self.types.insert(id, ty);
        id
    }

    /// Create a union node joining two alternatives.
    pub fn add_union(&mut self, left: NodeId, right: NodeId) -> NodeId {
        // Use the type of the left (they should be equivalent).
        let ty = self.types[&left].clone();
        let id = self.nodes.insert(ENode::Union { left, right });
        self.types.insert(id, ty);
        id
    }

    /// Turn a pure node into a union of itself and `alt`, in place: the
    /// original node is re-inserted under a fresh id (returned) and `id`
    /// becomes `Union { fresh, alt }`. Every existing reference to `id` —
    /// pure operands, side-effect slots, terminator args — sees both
    /// alternatives with no rewiring; extraction picks the cheaper side.
    pub fn union_pure_in_place(&mut self, id: NodeId, alt: NodeId) -> NodeId {
        assert_ne!(
            id, alt,
            "union_pure_in_place: alternative must differ from the node"
        );
        debug_assert!(matches!(self.nodes[id], ENode::Pure { .. }));
        let ty = self.types[&id].clone();
        let original = std::mem::replace(
            &mut self.nodes[id],
            ENode::Union {
                left: alt,
                right: alt,
            },
        );
        let fresh = self.nodes.insert(original);
        self.types.insert(fresh, ty);
        if let Some(span) = self.node_spans.get(&id).copied() {
            self.node_spans.insert(fresh, span);
        }
        // The hash-cons key for the original node now belongs to its fresh id.
        if let Some(key) = self.pure_node_key(fresh) {
            self.hash_cons.insert(key, fresh);
        }
        self.nodes[id] = ENode::Union {
            left: fresh,
            right: alt,
        };
        fresh
    }

    /// Discard a pure node in favor of `better`, in place: `id` becomes a
    /// degenerate union both of whose sides are `better`, so extraction can
    /// only pick `better` and existing references follow it. The discarded
    /// node's hash-cons key is retired.
    pub fn subsume_pure_in_place(&mut self, id: NodeId, better: NodeId) {
        assert_ne!(
            id, better,
            "subsume_pure_in_place: replacement must differ from the node"
        );
        debug_assert!(matches!(self.nodes[id], ENode::Pure { .. }));
        if let Some(key) = self.pure_node_key(id) {
            self.hash_cons.remove(&key);
        }
        self.nodes[id] = ENode::Union {
            left: better,
            right: better,
        };
    }
}

// ---------------------------------------------------------------------------
// Program and body containers
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Func<P: EgirPhase, Lang: Language> {
    pub name: String,
    pub span: Span,
    pub linkage_name: Option<String>,
    pub params: Vec<(Lang::Ty, String)>,
    pub return_ty: Lang::Ty,
    pub graph: EGraph<P, Lang>,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
    pub aliases: LookupMap<NodeId, NodeId>,
}

impl<P: EgirPhase, Lang: Language> Func<P, Lang> {
    pub fn new(
        name: String,
        span: Span,
        linkage_name: Option<String>,
        params: Vec<(Lang::Ty, String)>,
        return_ty: Lang::Ty,
        graph: EGraph<P, Lang>,
        control_headers: LookupMap<BlockId, ControlHeader>,
    ) -> Self {
        Self {
            name,
            span,
            linkage_name,
            params,
            return_ty,
            graph,
            control_headers,
            aliases: LookupMap::new(),
        }
    }
}

/// A body-backed compile-time constant retained in EGIR until final
/// elaboration. Constant bodies have no parameters and must be proven pure.
#[derive(Clone, Debug)]
pub struct ConstantDef<P: EgirPhase, Lang: Language> {
    pub name: String,
    pub span: Span,
    pub return_ty: Lang::Ty,
    pub graph: EGraph<P, Lang>,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
    pub aliases: LookupMap<NodeId, NodeId>,
}

/// One write site for an entry output slot.
#[derive(Debug, Clone, Copy)]
pub struct SlotSource {
    pub block: BlockId,
    pub value: NodeId,
}

/// Stable identity of a declared entry-output position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OutputSlotId(pub usize);

/// The concrete side effect that fulfils an output route after realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputWriter {
    Value(NodeId),
    Effect(EffectToken),
}

/// Declared output ownership carried through EGIR physicalization.
#[derive(Debug, Clone)]
pub struct OutputRoute {
    pub source: SlotSource,
    pub slot: OutputSlotId,
    pub writers: Vec<OutputWriter>,
}

/// One entry input together with its phase-typed resource identity, when the
/// slot is backed by a logical or physical resource.
#[derive(Debug, Clone)]
pub struct EntryInput<R, Lang: Language> {
    pub inner: InterfaceEntryInput<Lang::Ty>,
    pub resource: Option<R>,
}

impl<R, Lang: Language> std::ops::Deref for EntryInput<R, Lang> {
    type Target = InterfaceEntryInput<Lang::Ty>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<R, Lang: Language> std::ops::DerefMut for EntryInput<R, Lang> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// One entry output together with its phase-typed resource identity, when the
/// slot is backed by a logical or physical resource.
#[derive(Debug, Clone)]
pub struct EntryOutput<R, Lang: Language> {
    pub inner: InterfaceEntryOutput<Lang::Ty>,
    pub resource: Option<R>,
}

impl<R, Lang: Language> std::ops::Deref for EntryOutput<R, Lang> {
    type Target = InterfaceEntryOutput<Lang::Ty>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<R, Lang: Language> std::ops::DerefMut for EntryOutput<R, Lang> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Clone, Debug)]
pub struct Entry<P: EgirPhase, Lang: Language> {
    pub name: String,
    pub span: Span,
    pub execution_model: ExecutionModel,
    pub inputs: Vec<EntryInput<P::Resource, Lang>>,
    pub outputs: Vec<EntryOutput<P::Resource, Lang>>,
    pub resource_declarations: Vec<P::ResourceDecl>,
    pub params: Vec<(Lang::Ty, String)>,
    pub return_ty: Lang::Ty,
    pub graph: EGraph<P, Lang>,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
    pub aliases: LookupMap<NodeId, NodeId>,
    pub output_routes: Vec<OutputRoute>,
}

impl<P: EgirPhase, Lang: Language> Entry<P, Lang> {
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_resources(
        name: String,
        span: Span,
        execution_model: ExecutionModel,
        inputs: Vec<InterfaceEntryInput<Lang::Ty>>,
        outputs: Vec<InterfaceEntryOutput<Lang::Ty>>,
        resource_declarations: Vec<P::ResourceDecl>,
        params: Vec<(Lang::Ty, String)>,
        return_ty: Lang::Ty,
        graph: EGraph<P, Lang>,
        control_headers: LookupMap<BlockId, ControlHeader>,
    ) -> Self {
        Self {
            name,
            span,
            execution_model,
            inputs: inputs
                .into_iter()
                .map(|inner| EntryInput {
                    inner,
                    resource: None,
                })
                .collect(),
            outputs: outputs
                .into_iter()
                .map(|inner| EntryOutput {
                    inner,
                    resource: None,
                })
                .collect(),
            resource_declarations,
            params,
            return_ty,
            graph,
            control_headers,
            aliases: LookupMap::new(),
            output_routes: Vec::new(),
        }
    }
}

/// Whole-program EGIR container. Concrete compiler checkpoints wrap this
/// generic substrate and determine the phase-specific graph payload.
pub struct Program<P: EgirPhase, Lang: Language> {
    pub functions: Vec<Func<P, Lang>>,
    pub externs: Vec<ExternDecl<Lang::Ty>>,
    pub entry_points: Vec<Entry<P, Lang>>,
    pub constants: Vec<ConstantDef<P, Lang>>,
    pub pipeline: PipelineDescriptor,
    pub input_names: LookupMap<(u32, u32), String>,
    /// Region identity to the corresponding entry in `functions`.
    pub regions: LookupMap<RegionId, usize>,
    pub region_interner: RegionInterner,
}

fn record_region(
    interner: &mut RegionInterner,
    regions: &mut LookupMap<RegionId, usize>,
    function_index: usize,
    function_name: &str,
) -> RegionId {
    let id = interner.intern(function_name);
    regions.insert(id, function_index);
    id
}

impl<P: EgirPhase, Lang: Language> Program<P, Lang> {
    pub fn new(
        functions: Vec<Func<P, Lang>>,
        externs: Vec<ExternDecl<Lang::Ty>>,
        entry_points: Vec<Entry<P, Lang>>,
        constants: Vec<ConstantDef<P, Lang>>,
        pipeline: PipelineDescriptor,
        mut region_interner: RegionInterner,
    ) -> Self {
        let mut regions = LookupMap::new();
        for (index, function) in functions.iter().enumerate() {
            record_region(&mut region_interner, &mut regions, index, &function.name);
        }
        Self {
            functions,
            externs,
            entry_points,
            constants,
            pipeline,
            input_names: LookupMap::new(),
            regions,
            region_interner,
        }
    }

    /// Convenience constructor for a single function body.
    pub fn single_function(func: Func<P, Lang>) -> Self {
        Self::new(
            vec![func],
            vec![],
            vec![],
            vec![],
            PipelineDescriptor::default(),
            RegionInterner::default(),
        )
    }

    pub fn intern_region(&mut self, name: impl AsRef<str>) -> RegionId {
        self.region_interner.intern(name.as_ref())
    }

    pub fn define_region(&mut self, function: Func<P, Lang>) -> RegionId {
        let index = self.functions.len();
        let id = record_region(
            &mut self.region_interner,
            &mut self.regions,
            index,
            &function.name,
        );
        self.functions.push(function);
        id
    }

    pub fn contains_region(&self, id: RegionId) -> bool {
        self.regions.contains_key(&id)
    }

    pub fn region(&self, id: RegionId) -> Option<&Func<P, Lang>> {
        self.regions.get(&id).and_then(|&index| self.functions.get(index))
    }

    pub fn region_mut(&mut self, id: RegionId) -> Option<&mut Func<P, Lang>> {
        let index = *self.regions.get(&id)?;
        self.functions.get_mut(index)
    }

    pub fn iter_regions(&self) -> impl Iterator<Item = (RegionId, &Func<P, Lang>)> {
        self.regions
            .iter()
            .filter_map(|(&id, &index)| self.functions.get(index).map(|function| (id, function)))
    }

    pub fn region_name(&self, id: RegionId) -> &str {
        self.region_interner.resolve(id)
    }
}
