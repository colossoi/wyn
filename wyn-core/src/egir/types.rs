//! Core data structures for the acyclic e-graph (aegraph).

use crate::ast::{Span, TypeName};
use crate::ssa::framework::BlockId;
use crate::ssa::types::{ConstantValue, InstKind};

/// Effect token for ordering effectful ops during EGIR passes.
///
/// These are purely an EGIR-internal concept — they never reach the SSA
/// backend. `elaborate` emits instructions in skeleton block order and
/// doesn't pass the tokens through. The token chain only exists to support
/// rewriting passes (e.g. `soac_expand` allocating fresh tokens for new
/// Load/Store side-effects so they don't collide with existing ones).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectToken(pub u32);

impl std::fmt::Display for EffectToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "!{}", self.0)
    }
}
use crate::LookupMap;
use polytype::Type;
use slotmap::{new_key_type, SlotMap};
use smallvec::SmallVec;

use super::soac::{filter, hist, screma};

new_key_type! {
    /// Identity of a node in the e-graph. Every pure node, union node,
    /// block param, function param, and constant gets one.
pub struct NodeId;
}

/// Opaque handle into the program-level region arena (`SemanticProgram::regions`).
///
/// Region *identity* is a checked arena index, never a re-derived string. A
/// region still lowers to a named SSA function — that name is the call ABI and
/// lives on `SemanticRegion`/the `RegionInterner`, recovered via the arena when a
/// `PureOp::Call` is emitted. Semantic SegOps carry only this index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegionId(u32);

impl RegionId {
    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }

    pub const fn index(self) -> u32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// PureOp — operator identity for hash-consing (re-exported from `crate::op`)
// ---------------------------------------------------------------------------

/// Phase-typed operator identity without operands.
pub type PureOp<R = super::program::SemanticResourceRef> = crate::op::OpTag<R>;
pub type PureViewSource<R = super::program::SemanticResourceRef> = crate::op::PureViewSource<R>;

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
pub struct NodeKey<R = super::program::SemanticResourceRef> {
    pub op: PureOp<R>,
    pub operands: SmallVec<[NodeId; 4]>,
    pub ty: Type<TypeName>,
}

// ---------------------------------------------------------------------------
// ENode — what lives in the sea of nodes
// ---------------------------------------------------------------------------

/// A node in the e-graph.
#[derive(Clone, Debug)]
pub enum ENode<R = super::program::SemanticResourceRef> {
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
    /// Inline constant (lifted from `ValueRef::Const`).
    Constant(ConstantValue),
    /// Side-effect result — a value produced by an effectful instruction
    /// in the skeleton. Not hash-consed; each is unique.
    SideEffectResult,
}

impl<R> ENode<R> {
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

/// A side-effectful instruction anchored in the skeleton CFG.
#[derive(Clone, Debug)]
pub struct SideEffect<P: EgirPhase = Semantic> {
    /// Stable semantic identity assigned after segmentation and preserved by
    /// every projection. Synthesized physical-only effects use `None`.
    pub semantic_id: Option<super::program::SemanticOpId>,
    /// What this side-effect is. Either an SSA `InstKind` that survives into
    /// the final `FuncBody`, or an intermediate `Soac` that must be
    /// rewritten by `soac_expand` before `elaborate` runs.
    pub kind: SideEffectKind<P>,
    /// Operands resolved to NodeIds.
    pub operand_nodes: SmallVec<[NodeId; 4]>,
    /// Result value, if this instruction produces one.
    pub result: Option<NodeId>,
    /// Effect token chain.
    pub effects: Option<(EffectToken, EffectToken)>,
    /// Source span of the user expression that produced this side-effect,
    /// or `None` for synthesized side-effects (e.g. SOAC expansion).
    pub span: Option<Span>,
}

/// A skeleton side-effect's concrete kind.
#[derive(Clone, Debug)]
pub enum SideEffectKind<P: EgirPhase = Semantic> {
    /// An SSA-level effectful instruction (`Alloca` / `Load` / `Store` /
    /// `Call` / `Intrinsic` / `StorageView*` / `OutputPtr` with effects).
    /// This is what lands in the final `FuncBody` after elaboration.
    Inst(InstKind<P::Resource>),
    /// A placeholder for an unexpanded SOAC. Produced by `from_tlc` and
    /// consumed by `soac_expand`. Never reaches elaborate.
    Soac(Soac<P>),
}

/// Where an array-producing SOAC's per-iteration result is written. TLC only
/// supplies the logical `Fresh`/`UniqueInput` ownership fact; the physical
/// `InputBuffer`/`OutputView` choices exist exclusively in EGIR.
///
/// A Screma side effect stores only `[inputs..., output_views...]` in its
/// operand vector. Callable captures and accumulator neutrals are explicit in
/// `screma::Body`, so they cannot drift out of sync with the operation.
/// `InputBuffer` has no output-view operand; its result aliases an input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoacDestination {
    Fresh,
    UniqueInput,
    InputBuffer,
    OutputView,
}

impl From<crate::tlc::SoacDestination> for SoacDestination {
    fn from(destination: crate::tlc::SoacDestination) -> Self {
        match destination {
            crate::tlc::SoacDestination::Fresh => Self::Fresh,
            crate::tlc::SoacDestination::UniqueInput => Self::UniqueInput,
        }
    }
}

/// Execution level of a `Seg` op. wyn currently only emits thread-level
/// kernels (one invocation per lane); block-level would be added here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegLevel {
    Thread,
}

/// One concrete dimension of a segmented iteration space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SegExtent<R = super::program::SemanticResourceRef> {
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
pub struct SegSpace<R = super::program::SemanticResourceRef> {
    pub level: SegLevel,
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
pub enum SegResourceAccessKind {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegResourceAccess<R = super::program::SemanticResourceRef> {
    pub resource: R,
    pub access: SegResourceAccessKind,
}

impl<R: Copy + Ord> SegResourceAccess<R> {
    pub fn merge(a: &[Self], b: &[Self]) -> Vec<Self> {
        let mut merged = std::collections::BTreeMap::new();
        for resource in a.iter().chain(b) {
            merged
                .entry(resource.resource)
                .and_modify(|access| {
                    if *access != resource.access {
                        *access = SegResourceAccessKind::ReadWrite;
                    }
                })
                .or_insert(resource.access);
        }
        merged.into_iter().map(|(resource, access)| Self { resource, access }).collect()
    }
}

pub trait EgirPhase: Clone + std::fmt::Debug {
    type Resource: GraphResource;
    type MapState: Clone + std::fmt::Debug;
    type ReduceState: Clone + std::fmt::Debug;
    type ScanState: Clone + std::fmt::Debug;
    type CompositeState: Clone + std::fmt::Debug;
    type FilterState: Clone + std::fmt::Debug;
    type HistState: Clone + std::fmt::Debug;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Raw<R = super::program::SemanticResourceRef>(std::marker::PhantomData<fn() -> R>);

#[derive(Clone, Copy, Debug, Default)]
pub struct Semantic<R = super::program::SemanticResourceRef>(std::marker::PhantomData<fn() -> R>);

#[derive(Clone, Copy, Debug, Default)]
pub struct Scheduled<R = super::program::SemanticResourceRef>(std::marker::PhantomData<fn() -> R>);

#[derive(Clone, Copy, Debug, Default)]
pub struct Physical;

#[derive(Clone, Debug)]
pub struct SoacInputType {
    pub array: Type<TypeName>,
    pub element: Type<TypeName>,
}

impl<R: GraphResource> EgirPhase for Raw<R> {
    type Resource = R;
    type MapState = screma::RawState;
    type ReduceState = screma::RawState;
    type ScanState = screma::RawState;
    type CompositeState = screma::RawState;
    type FilterState = filter::RawState<R>;
    type HistState = hist::RawState;
}

impl<R: GraphResource> EgirPhase for Semantic<R> {
    type Resource = R;
    type MapState = screma::SemanticState<R>;
    type ReduceState = screma::SemanticState<R>;
    type ScanState = screma::SemanticState<R>;
    type CompositeState = screma::SemanticState<R>;
    type FilterState = filter::SemanticState<R>;
    type HistState = hist::SemanticState<R>;
}

impl<R: GraphResource> EgirPhase for Scheduled<R> {
    type Resource = R;
    type MapState = screma::ScheduledState<R>;
    type ReduceState = screma::ScheduledState<R>;
    type ScanState = screma::ScheduledState<R>;
    type CompositeState = screma::ScheduledState<R>;
    type FilterState = filter::ScheduledState<R>;
    type HistState = hist::ScheduledState<R>;
}

impl EgirPhase for Physical {
    type Resource = super::program::PhysicalResourceRef;
    type MapState = screma::PhysicalMapState;
    type ReduceState = screma::PhysicalSerialState;
    type ScanState = screma::PhysicalSerialState;
    type CompositeState = screma::PhysicalSerialState;
    type FilterState = filter::PhysicalState;
    type HistState = hist::PhysicalState;
}

#[derive(Clone, Debug)]
pub enum Soac<P: EgirPhase> {
    Screma(screma::Op<P>),
    Filter(filter::Op<P>),
    Hist(hist::Op<P>),
}

impl<P: EgirPhase> Soac<P> {
    fn for_each_body_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        match self {
            Self::Screma(op) => op.for_each_type_mut(visit),
            Self::Filter(op) => op.body.for_each_type_mut(visit),
            Self::Hist(op) => op.body.for_each_type_mut(visit),
        }
    }

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

impl<R: GraphResource> Soac<Raw<R>> {
    pub(crate) fn for_each_type_mut(&mut self, mut visit: impl FnMut(&mut Type<TypeName>)) {
        self.for_each_body_type_mut(&mut visit);
        if let Self::Filter(op) = self {
            op.state.for_each_type_mut(&mut visit);
        }
    }
}

impl Soac<Physical> {
    pub(crate) fn for_each_type_mut(&mut self, mut visit: impl FnMut(&mut Type<TypeName>)) {
        self.for_each_body_type_mut(&mut visit);
        if let Self::Filter(op) = self {
            op.state.for_each_type_mut(&mut visit);
        }
    }
}

impl<R: GraphResource> Soac<Semantic<R>> {
    pub fn capture_nodes(&self) -> impl Iterator<Item = NodeId> {
        let nodes = match self {
            Self::Screma(op) => op.capture_nodes(),
            Self::Filter(op) => op.capture_nodes(),
            Self::Hist(op) => op.capture_nodes(),
        };
        nodes.into_iter()
    }

    /// Concrete iteration space seen by scheduling, independent of SOAC
    /// family. Serial Scremas and histograms have no dispatched space.
    pub(crate) fn scheduling_space(&self) -> Option<&SegSpace<R>> {
        match self {
            Self::Screma(op) => match op.semantic_state() {
                screma::SemanticState::Serial => None,
                screma::SemanticState::Segmented { space, .. } => Some(space),
            },
            Self::Filter(op) => Some(&op.state.space),
            Self::Hist(op) => match &op.state {
                hist::SemanticState::Serial => None,
                hist::SemanticState::Segmented(space) => Some(space),
            },
        }
    }

    fn referenced_nodes(&self) -> Vec<NodeId> {
        match self {
            Self::Screma(op) => op.referenced_nodes(),
            Self::Filter(op) => op.referenced_nodes(),
            Self::Hist(op) => op.referenced_nodes(),
        }
    }

    fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        match self {
            Self::Screma(op) => op.referenced_node_slots(),
            Self::Filter(op) => op.referenced_node_slots(),
            Self::Hist(op) => op.referenced_node_slots(),
        }
    }
}

impl<P: EgirPhase> SideEffect<P> {
    pub fn required_semantic_id(&self) -> super::program::SemanticOpId {
        self.semantic_id.expect("semantic operation id assigned after segmentation")
    }
}

impl<R: GraphResource> SideEffect<Semantic<R>> {
    /// Every graph value used by the effect, including SOAC captures,
    /// operator metadata, and semantic iteration-space extents.
    pub fn referenced_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        let metadata = match &self.kind {
            SideEffectKind::Soac(soac) => soac.referenced_nodes(),
            SideEffectKind::Inst(_) => Vec::new(),
        };
        self.operand_nodes.iter().copied().chain(metadata)
    }

    pub fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        let Self {
            kind, operand_nodes, ..
        } = self;
        let mut slots = operand_nodes.iter_mut().collect::<Vec<_>>();
        if let SideEffectKind::Soac(soac) = kind {
            slots.extend(soac.referenced_node_slots());
        }
        slots
    }
}

/// Terminator using NodeIds for value references.
#[derive(Clone, Debug)]
pub enum SkeletonTerminator {
    Return(Option<NodeId>),
    Branch {
        target: BlockId,
        args: Vec<NodeId>,
    },
    CondBranch {
        cond: NodeId,
        then_target: BlockId,
        then_args: Vec<NodeId>,
        else_target: BlockId,
        else_args: Vec<NodeId>,
    },
    Unreachable,
}

impl SkeletonTerminator {
    pub fn referenced_nodes(&self) -> SmallVec<[NodeId; 8]> {
        match self {
            Self::Return(value) => value.iter().copied().collect(),
            Self::Branch { args, .. } => args.iter().copied().collect(),
            Self::CondBranch {
                cond,
                then_args,
                else_args,
                ..
            } => std::iter::once(*cond)
                .chain(then_args.iter().copied())
                .chain(else_args.iter().copied())
                .collect(),
            Self::Unreachable => SmallVec::new(),
        }
    }

    pub fn visit_nodes_mut(&mut self, mut visit: impl FnMut(&mut NodeId)) {
        match self {
            Self::Return(value) => value.iter_mut().for_each(visit),
            Self::Branch { args, .. } => args.iter_mut().for_each(visit),
            Self::CondBranch {
                cond,
                then_args,
                else_args,
                ..
            } => {
                visit(cond);
                then_args.iter_mut().for_each(&mut visit);
                else_args.iter_mut().for_each(visit);
            }
            Self::Unreachable => {}
        }
    }

    pub fn try_map<E>(
        &self,
        mut map_node: impl FnMut(NodeId) -> Result<NodeId, E>,
        mut map_block: impl FnMut(BlockId) -> Result<BlockId, E>,
    ) -> Result<Self, E> {
        Ok(match self {
            Self::Return(value) => Self::Return(value.map(&mut map_node).transpose()?),
            Self::Branch { target, args } => Self::Branch {
                target: map_block(*target)?,
                args: args.iter().copied().map(map_node).collect::<Result<_, _>>()?,
            },
            Self::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
            } => Self::CondBranch {
                cond: map_node(*cond)?,
                then_target: map_block(*then_target)?,
                then_args: then_args.iter().copied().map(&mut map_node).collect::<Result<_, _>>()?,
                else_target: map_block(*else_target)?,
                else_args: else_args.iter().copied().map(map_node).collect::<Result<_, _>>()?,
            },
            Self::Unreachable => Self::Unreachable,
        })
    }
}

/// A block in the skeleton CFG.
#[derive(Clone, Debug)]
pub struct SkeletonBlock<P: EgirPhase = Semantic> {
    /// Block parameters as NodeIds.
    pub params: Vec<NodeId>,
    /// Effectful instructions, in order.
    pub side_effects: Vec<SideEffect<P>>,
    /// Block terminator.
    pub term: SkeletonTerminator,
}

impl<P: EgirPhase> SkeletonBlock<P> {
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
pub struct Skeleton<P: EgirPhase = Semantic> {
    pub entry: BlockId,
    pub blocks: SlotMap<BlockId, SkeletonBlock<P>>,
}

impl<P: EgirPhase> Skeleton<P> {
    pub fn new() -> Self {
        let mut blocks = SlotMap::with_key();
        let entry = blocks.insert(SkeletonBlock::new());
        Skeleton { entry, blocks }
    }

    pub fn create_block(&mut self) -> BlockId {
        self.blocks.insert(SkeletonBlock::new())
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
    pub fn build<P: EgirPhase>(graph: &EGraph<P>) -> Self {
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

    pub fn effect<'a, P: EgirPhase>(
        &self,
        graph: &'a EGraph<P>,
        result: NodeId,
    ) -> Option<&'a SideEffect<P>> {
        let site = self.site(result)?;
        let effect = graph.skeleton.blocks.get(site.block)?.side_effects.get(site.index)?;
        (effect.result == Some(result)).then_some(effect)
    }

    pub fn effect_mut<'a, P: EgirPhase>(
        &self,
        graph: &'a mut EGraph<P>,
        result: NodeId,
    ) -> Option<&'a mut SideEffect<P>> {
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
pub struct EGraph<P: EgirPhase = Semantic> {
    /// All nodes (pure, union, params, constants, side-effect results).
    pub nodes: SlotMap<NodeId, ENode<P::Resource>>,
    /// Type of each node's result.
    pub types: LookupMap<NodeId, Type<TypeName>>,
    /// Hash-cons table: NodeKey → existing NodeId.
    hash_cons: LookupMap<NodeKey<P::Resource>, NodeId>,
    /// Constant dedup cache.
    pub const_cache: LookupMap<ConstantValue, NodeId>,
    /// The CFG skeleton.
    pub skeleton: Skeleton<P>,
    /// Source span associated with each pure node (first-writer-wins —
    /// later interns of the same hash-consed node keep the original span).
    pub node_spans: LookupMap<NodeId, Span>,
}

pub trait GraphResource: Clone + std::fmt::Debug + Eq + std::hash::Hash {}

impl<T> GraphResource for T where T: Clone + std::fmt::Debug + Eq + std::hash::Hash {}

impl<P: EgirPhase> EGraph<P> {
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

    pub(crate) fn try_map_phase<Q, E>(
        self,
        mut map_soac: impl FnMut(BlockId, usize, Soac<P>) -> Result<Soac<Q>, E>,
    ) -> Result<(EGraph<Q>, LookupMap<BlockId, BlockId>), E>
    where
        Q: EgirPhase<Resource = P::Resource>,
    {
        let EGraph {
            mut nodes,
            types,
            hash_cons,
            const_cache,
            skeleton,
            node_spans,
        } = self;
        let source_entry = skeleton.entry;
        let source_blocks = skeleton.blocks.into_iter().collect::<Vec<_>>();
        let mut blocks = SlotMap::with_key();
        let mut block_map = LookupMap::new();
        for (source, _) in &source_blocks {
            block_map.insert(*source, blocks.insert(SkeletonBlock::<Q>::new()));
        }

        for node in nodes.values_mut() {
            if let ENode::BlockParam { block, .. } = node {
                *block = block_map[block];
            }
        }

        for (source, block) in source_blocks {
            let side_effects = block
                .side_effects
                .into_iter()
                .enumerate()
                .map(|(index, effect)| {
                    let kind = match effect.kind {
                        SideEffectKind::Inst(inst) => SideEffectKind::Inst(inst),
                        SideEffectKind::Soac(soac) => SideEffectKind::Soac(map_soac(source, index, soac)?),
                    };
                    Ok(SideEffect {
                        semantic_id: effect.semantic_id,
                        kind,
                        operand_nodes: effect.operand_nodes,
                        result: effect.result,
                        effects: effect.effects,
                        span: effect.span,
                    })
                })
                .collect::<Result<_, E>>()?;
            let term =
                block.term.try_map(|node| Ok::<_, E>(node), |target| Ok::<_, E>(block_map[&target]))?;
            blocks[block_map[&source]] = SkeletonBlock {
                params: block.params,
                side_effects,
                term,
            };
        }

        Ok((
            EGraph {
                nodes,
                types,
                hash_cons,
                const_cache,
                skeleton: Skeleton {
                    entry: block_map[&source_entry],
                    blocks,
                },
                node_spans,
            },
            block_map,
        ))
    }

    /// Rebuild a graph when both its compiler phase and resource identity
    /// change. Graph structure is mapped here; the caller owns the direct
    /// business-logic conversion for each SOAC.
    pub(crate) fn try_map_resources_and_phase<Q, E>(
        self,
        mut map_resource: impl FnMut(P::Resource) -> Result<Q::Resource, E>,
        mut map_soac: impl FnMut(Soac<P>, &LookupMap<NodeId, NodeId>) -> Result<Soac<Q>, E>,
    ) -> Result<(EGraph<Q>, LookupMap<NodeId, NodeId>, LookupMap<BlockId, BlockId>), E>
    where
        Q: EgirPhase,
    {
        let EGraph {
            nodes,
            types,
            hash_cons: _,
            const_cache,
            skeleton,
            node_spans,
        } = self;
        let source_entry = skeleton.entry;
        let source_blocks = skeleton.blocks.into_iter().collect::<Vec<_>>();
        let mut blocks = SlotMap::with_key();
        let mut block_map = LookupMap::new();
        for (source, _) in &source_blocks {
            block_map.insert(*source, blocks.insert(SkeletonBlock::<Q>::new()));
        }

        let source_nodes = nodes.into_iter().collect::<Vec<_>>();
        let mut nodes = SlotMap::with_key();
        let mut node_map = LookupMap::new();
        for (source, _) in &source_nodes {
            node_map.insert(
                *source,
                nodes.insert(ENode::<Q::Resource>::Constant(ConstantValue::Bool(false))),
            );
        }
        for (source, node) in source_nodes {
            let node = match node {
                ENode::Pure { op, operands } => ENode::Pure {
                    op: op.try_map_resource(&mut map_resource)?,
                    operands: operands.into_iter().map(|node| node_map[&node]).collect(),
                },
                ENode::Union { left, right } => ENode::Union {
                    left: node_map[&left],
                    right: node_map[&right],
                },
                ENode::FuncParam { index } => ENode::FuncParam { index },
                ENode::BlockParam { block, index } => ENode::BlockParam {
                    block: block_map[&block],
                    index,
                },
                ENode::Constant(value) => ENode::Constant(value),
                ENode::SideEffectResult => ENode::SideEffectResult,
            };
            nodes[node_map[&source]] = node;
        }

        for (source, block) in source_blocks {
            let side_effects = block
                .side_effects
                .into_iter()
                .map(|effect| {
                    let kind = match effect.kind {
                        SideEffectKind::Inst(inst) => {
                            SideEffectKind::Inst(inst.try_map_resource(&mut map_resource)?)
                        }
                        SideEffectKind::Soac(soac) => SideEffectKind::Soac(map_soac(soac, &node_map)?),
                    };
                    Ok(SideEffect::<Q> {
                        semantic_id: effect.semantic_id,
                        kind,
                        operand_nodes: effect
                            .operand_nodes
                            .into_iter()
                            .map(|node| node_map[&node])
                            .collect(),
                        result: effect.result.map(|node| node_map[&node]),
                        effects: effect.effects,
                        span: effect.span,
                    })
                })
                .collect::<Result<Vec<_>, E>>()?;
            let term = block.term.try_map(
                |node| Ok::<_, E>(node_map[&node]),
                |target| Ok::<_, E>(block_map[&target]),
            )?;
            blocks[block_map[&source]] = SkeletonBlock {
                params: block.params.into_iter().map(|node| node_map[&node]).collect(),
                side_effects,
                term,
            };
        }

        let mut graph = EGraph {
            nodes,
            types: types.into_iter().map(|(node, ty)| (node_map[&node], ty)).collect(),
            hash_cons: LookupMap::new(),
            const_cache: const_cache.into_iter().map(|(value, node)| (value, node_map[&node])).collect(),
            skeleton: Skeleton {
                entry: block_map[&source_entry],
                blocks,
            },
            node_spans: node_spans.into_iter().map(|(node, span)| (node_map[&node], span)).collect(),
        };
        graph.rebuild_hash_cons();
        Ok((graph, node_map, block_map))
    }

    pub fn side_effect_index(&self) -> SideEffectIndex {
        SideEffectIndex::build(self)
    }

    fn pure_node_key(&self, id: NodeId) -> Option<NodeKey<P::Resource>> {
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
    pub fn replace_node_preserving_type(&mut self, id: NodeId, node: ENode<P::Resource>) {
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
    pub fn retype_node(&mut self, id: NodeId, ty: Type<TypeName>) {
        self.unindex_current_pure(id);
        self.types.insert(id, ty);
        self.index_current_pure(id);
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
    pub fn add_func_param(&mut self, index: usize, ty: Type<TypeName>) -> NodeId {
        let id = self.nodes.insert(ENode::FuncParam { index });
        self.types.insert(id, ty);
        id
    }

    /// Allocate a block parameter node.
    pub fn add_block_param(&mut self, block: BlockId, index: usize, ty: Type<TypeName>) -> NodeId {
        let id = self.nodes.insert(ENode::BlockParam { block, index });
        self.types.insert(id, ty);
        id
    }

    /// Intern a constant, deduplicating.
    pub fn intern_constant(&mut self, c: ConstantValue, ty: Type<TypeName>) -> NodeId {
        if let Some(&existing) = self.const_cache.get(&c) {
            return existing;
        }
        let id = self.nodes.insert(ENode::Constant(c));
        self.types.insert(id, ty);
        self.const_cache.insert(c, id);
        id
    }

    /// Intern a pure node, hash-consing to deduplicate.
    /// Returns the existing NodeId if an equivalent node already exists (GVN).
    pub fn intern_pure(
        &mut self,
        op: PureOp<P::Resource>,
        operands: SmallVec<[NodeId; 4]>,
        ty: Type<TypeName>,
    ) -> NodeId {
        self.intern_pure_with_span(op, operands, ty, None)
    }

    /// Intern a pure node with an attached source span. The span is recorded
    /// on first intern; subsequent interns of an equivalent hash-consed node
    /// keep the original span.
    pub fn intern_pure_with_span(
        &mut self,
        op: PureOp<P::Resource>,
        operands: SmallVec<[NodeId; 4]>,
        ty: Type<TypeName>,
        span: Option<Span>,
    ) -> NodeId {
        if let Some(folded) = self.try_algebraic_fold(&op, &operands, &ty) {
            return folded;
        }
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
    pub fn alloc_side_effect_result(&mut self, ty: Type<TypeName>) -> NodeId {
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

#[cfg(test)]
#[path = "types_tests.rs"]
mod types_tests;

// ---------------------------------------------------------------------------
// Conversion helpers: InstKind ↔ PureOp
// ---------------------------------------------------------------------------

/// Build an `InstKind::Op` from a `PureOp` + operands as `ValueRef::Ssa(ValueId)`.
/// Panics on the place-producing tags (`ViewIndex`, `PlaceIndex`, `OutputSlot`) —
/// those need `elaborate`'s place-aware path which allocates a fresh `PlaceId`.
pub fn rebuild_inst_kind(
    op: &PureOp<crate::BindingRef>,
    operands: &[crate::ssa::framework::ValueId],
) -> InstKind {
    use crate::ssa::types::ValueRef;
    if matches!(
        op,
        PureOp::ViewIndex | PureOp::PlaceIndex | PureOp::OutputSlot { .. }
    ) {
        panic!(
            "rebuild_inst_kind: place-producing op {:?} must be built via elaborate's \
             place-aware path (allocates a fresh PlaceId from FuncBody.places)",
            op
        );
    }
    InstKind::Op {
        tag: op.clone(),
        operands: operands.iter().map(|&id| ValueRef::Ssa(id)).collect(),
    }
}

/// Rebuild an effectful `InstKind` from the original kind and new operands
/// (as `ValueId`s, same order as `value_uses()`).
pub fn rebuild_effectful_inst_kind(
    original: &InstKind,
    operands: &[crate::ssa::framework::ValueId],
) -> InstKind {
    use crate::ssa::types::ValueRef;
    let mut result = original.clone();
    let mut idx = 0;
    result.substitute_values(&mut |vr| {
        *vr = ValueRef::Ssa(operands[idx]);
        idx += 1;
    });
    result
}
