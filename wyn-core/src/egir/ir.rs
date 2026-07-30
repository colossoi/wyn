//! Phase-agnostic, low-level data structures for EGIR.

use slotmap::{new_key_type, SlotMap};
use smallvec::SmallVec;

use crate::ast::Span;
use crate::flow::{BlockId, ControlHeader, ExecutionModel};
use crate::interface::{EntryInput as InterfaceEntryInput, EntryOutput as InterfaceEntryOutput};
use crate::op::OpTag;
use crate::ssa::types::AtomicOp;
use crate::types::ExternDecl;
use crate::{LookupMap, LookupSet, SortedSet};

pub use crate::op::PureViewSource;
pub use crate::types::SoacOwnership;

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

/// Splice a producer and consumer's effect-token chains around a fused operation.
pub fn splice_effect_tokens(
    producer: Option<(EffectToken, EffectToken)>,
    consumer: Option<(EffectToken, EffectToken)>,
) -> Option<(EffectToken, EffectToken)> {
    match (producer, consumer) {
        (Some((input, _)), Some((_, output))) => Some((input, output)),
        (Some(effects), None) | (None, Some(effects)) => Some(effects),
        (None, None) => None,
    }
}

new_key_type! {
    /// Identity of a node in the e-graph. Every pure node, union node,
    /// block param, function param, and constant gets one.
    pub struct NodeId;

}

/// A callable used as a structured SOAC region.
///
/// Regions and ordinary functions deliberately share one identity realm:
/// every call target is a [`crate::FunctionId`]. The alias preserves the
/// domain-specific vocabulary used by segmented operators without opening a
/// second allocator.
pub type RegionId = crate::FunctionId;

/// Program-owned identity arenas with names as one-way metadata.
///
/// There is intentionally no name-to-ID lookup. All resolution happens before
/// these arenas are built; their strings exist only for diagnostics, emitted
/// symbols, and host-facing entry metadata.
pub type RegionArena = crate::IdArena<RegionId, String>;
pub type GlobalArena = crate::IdArena<crate::GlobalId, String>;
pub type EntryArena = crate::IdArena<crate::EntryId, String>;

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
/// type win at the merged site. The type distinction applies uniformly to
/// literals and every other pure operation.
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

/// One graph node together with all metadata intrinsically owned by that
/// identity.
#[derive(Clone, Debug)]
pub struct Node<R, Lang: Language> {
    pub kind: ENode<R, Lang>,
    pub ty: Lang::Ty,
    /// First source span attached to this hash-consed value.
    pub span: Option<Span>,
    /// Canonical replacement selected by CFG simplification, if any.
    pub alias: Option<NodeId>,
}

impl<R, Lang: Language> Node<R, Lang> {
    /// Return the graph dependencies referenced by this node.
    pub fn children(&self) -> SmallVec<[NodeId; 4]> {
        self.kind.children()
    }
}

// ---------------------------------------------------------------------------
// Skeleton — the CFG of side-effectful instructions
// ---------------------------------------------------------------------------

/// A side effect anchored in the skeleton CFG.
#[derive(Clone, Debug)]
pub struct SideEffect<P: Family, Lang: Language> {
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
    Atomic(AtomicOp),
    ControlBarrier,
}

impl<R, Ty> EffectOp<R, Ty> {
    /// Resource identity carried directly by the operation tag, if any.
    pub fn referenced_resource(&self) -> Option<&R> {
        match self {
            Self::Op { tag } => tag.referenced_resource(),
            Self::Alloca { .. } | Self::Load | Self::Store | Self::Atomic(_) | Self::ControlBarrier => None,
        }
    }

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
            Self::Atomic(op) => EffectOp::Atomic(op),
            Self::ControlBarrier => EffectOp::ControlBarrier,
        })
    }
}

/// A skeleton side effect's concrete kind.
#[derive(Clone, Debug)]
pub enum SideEffectKind<P: Family, Lang: Language> {
    Effect(EffectOp<P::Resource, Lang::Ty>),
    /// A placeholder for an unexpanded SOAC. Produced by `from_tlc` and
    /// consumed by `soac_expand`. Never reaches elaborate.
    Soac(P::Soac),
}

/// External storage selected for a SOAC result by EGIR lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoacPlacement {
    InputBuffer,
    OutputView,
}

/// Complete lowering state for a SOAC result destination. Keeping candidate
/// ownership and resolved placement in one enum prevents combinations whose
/// ownership applies only before a concrete destination is selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoacDestination {
    Fresh,
    UniqueInput,
    InputBuffer,
    OutputView,
}

impl SoacDestination {
    pub const fn fresh() -> Self {
        Self::Fresh
    }

    pub const fn unique_input() -> Self {
        Self::UniqueInput
    }

    pub const fn placed(self, placement: SoacPlacement) -> Self {
        match placement {
            SoacPlacement::InputBuffer => Self::InputBuffer,
            SoacPlacement::OutputView => Self::OutputView,
        }
    }

    pub fn place(&mut self, placement: SoacPlacement) {
        *self = self.placed(placement);
    }

    pub fn make_fresh(&mut self) {
        *self = Self::fresh();
    }

    pub const fn is_unplaced_fresh(self) -> bool {
        matches!(self, Self::Fresh)
    }

    pub const fn is_unplaced(self) -> bool {
        matches!(self, Self::Fresh | Self::UniqueInput)
    }

    pub const fn is_unplaced_unique_input(self) -> bool {
        matches!(self, Self::UniqueInput)
    }

    pub const fn is_input_buffer(self) -> bool {
        matches!(self, Self::InputBuffer)
    }

    pub const fn is_output_view(self) -> bool {
        matches!(self, Self::OutputView)
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
    dims: Vec<SegExtent<R>>,
}

impl<R> SegSpace<R> {
    pub(crate) fn new(extent: SegExtent<R>) -> Self {
        Self { dims: vec![extent] }
    }

    pub(crate) fn from_dims(dims: Vec<SegExtent<R>>) -> Option<Self> {
        (!dims.is_empty()).then_some(Self { dims })
    }

    pub(crate) fn dims(&self) -> &[SegExtent<R>] {
        &self.dims
    }

    pub(crate) fn into_dims(self) -> Vec<SegExtent<R>> {
        self.dims
    }

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

impl<R: Copy> SegSpace<R> {
    /// Rewrite one referenced graph value and, for a resource-length extent,
    /// keep its resource identity synchronized with the replacement view.
    pub(crate) fn replace_reference(&mut self, old: NodeId, new: NodeId, resource: R) {
        for extent in &mut self.dims {
            match extent {
                SegExtent::PushConstant { node, .. } | SegExtent::Value(node) if *node == old => {
                    *node = new;
                }
                SegExtent::ResourceLength {
                    node,
                    resource: extent_resource,
                    ..
                } if *node == old => {
                    *node = new;
                    *extent_resource = resource;
                }
                _ => {}
            }
        }
    }

    /// Retarget a one-dimensional dynamic domain to the length of a concrete
    /// resource view. Returns false when the current space is not the supported
    /// one-dimensional dynamic shape.
    pub(crate) fn retarget_single_resource_length(
        &mut self,
        view: NodeId,
        resource: R,
        elem_bytes: u32,
    ) -> bool {
        let [extent] = self.dims.as_mut_slice() else {
            return false;
        };
        if !matches!(extent, SegExtent::Value(_) | SegExtent::ResourceLength { .. }) {
            return false;
        }
        *extent = SegExtent::ResourceLength {
            node: view,
            resource,
            elem_bytes,
        };
        true
    }
}

/// A complete callable body and the values captured from its surrounding
/// graph. Captures are explicit values, never an operand-count convention.
#[derive(Clone, Debug)]
pub struct SegBody {
    pub region: RegionId,
    pub captures: Vec<NodeId>,
}

impl SegBody {
    /// Number of non-capture parameters in this body's region ABI.
    ///
    /// Segment bodies bind their lane/element parameters first and append one
    /// function parameter for each capture.
    pub(crate) fn leading_parameter_count<P: Family, Lang: Language>(
        &self,
        function: &Func<P, Lang>,
    ) -> Result<usize, String> {
        function.params.len().checked_sub(self.captures.len()).ok_or_else(|| {
            format!(
                "region `{}` has {} parameters but {} captures",
                function.name,
                function.params.len(),
                self.captures.len()
            )
        })
    }

    /// Map this body's capture parameters to their enclosing graph values.
    pub(crate) fn capture_bindings<P: Family, Lang: Language>(
        &self,
        function: &Func<P, Lang>,
    ) -> Result<LookupMap<NodeId, NodeId>, String> {
        let leading = self.leading_parameter_count(function)?;
        let mut bindings = LookupMap::new();
        for (node, definition) in &function.graph.nodes {
            let ENode::FuncParam { index } = &definition.kind else {
                continue;
            };
            if *index < leading || *index >= function.params.len() {
                continue;
            }
            let capture = self.captures.get(*index - leading).copied().ok_or_else(|| {
                format!(
                    "region `{}` has out-of-range capture parameter {index}",
                    function.name
                )
            })?;
            bindings.insert(node, capture);
        }
        Ok(bindings)
    }
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

/// Phase-varying data embedded recursively in EGIR nodes.
///
/// Every associated type corresponds to an actual field stored in the graph.
/// Proof-only typestates and program-wide context stay at [`Program`].
pub trait Family: Clone + std::fmt::Debug {
    type Resource: GraphResource;
    type Soac: Clone + std::fmt::Debug;
}

/// The complete collection of types physically stored by an EGIR program.
///
/// [`ProgramFamily`] makes this collection explicit at each top-level
/// typestate alias. Descendants receive only the individual types they store.
pub trait ProgramShape {
    type Family: Family;
    type ResourceDecl: Clone + std::fmt::Debug;
    type OutputRoute: Clone + std::fmt::Debug;
    type ProgramData: std::fmt::Debug;
}

/// A transparent description of one EGIR tree representation.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProgramFamily<GraphFamily, ResourceDecl, OutputRoute, ProgramData>(
    std::marker::PhantomData<fn() -> (GraphFamily, ResourceDecl, OutputRoute, ProgramData)>,
);

impl<GraphFamily, ResourceDecl, OutputRoute, ProgramData> ProgramShape
    for ProgramFamily<GraphFamily, ResourceDecl, OutputRoute, ProgramData>
where
    GraphFamily: Family,
    ResourceDecl: Clone + std::fmt::Debug,
    OutputRoute: Clone + std::fmt::Debug,
    ProgramData: std::fmt::Debug,
{
    type Family = GraphFamily;
    type ResourceDecl = ResourceDecl;
    type OutputRoute = OutputRoute;
    type ProgramData = ProgramData;
}

#[derive(Clone, Debug)]
pub struct SoacInputType<Ty> {
    pub array: Ty,
}

/// Terminator using NodeIds for value references.
pub type SkeletonTerminator = crate::flow::Terminator<NodeId>;

/// A block in the skeleton CFG.
#[derive(Clone, Debug)]
pub struct SkeletonBlock<P: Family, Lang: Language> {
    /// Block parameters as NodeIds.
    pub params: Vec<NodeId>,
    /// Effectful instructions, in order.
    pub side_effects: Vec<SideEffect<P, Lang>>,
    /// Block terminator.
    pub term: SkeletonTerminator,
    /// Structured-control metadata intrinsically owned by this block.
    pub control_header: Option<ControlHeader>,
}

impl<P: Family, Lang: Language> SkeletonBlock<P, Lang> {
    pub fn new() -> Self {
        SkeletonBlock {
            params: Vec::new(),
            side_effects: Vec::new(),
            term: SkeletonTerminator::Unreachable,
            control_header: None,
        }
    }
}

/// The skeleton CFG (blocks + effectful instructions).
#[derive(Clone, Debug)]
pub struct Skeleton<P: Family, Lang: Language> {
    pub entry: BlockId,
    pub blocks: SlotMap<BlockId, SkeletonBlock<P, Lang>>,
}

impl<P: Family, Lang: Language> Skeleton<P, Lang> {
    pub fn new() -> Self {
        let mut blocks = SlotMap::with_key();
        let entry = blocks.insert(SkeletonBlock::new());
        Skeleton { entry, blocks }
    }

    pub fn create_block(&mut self) -> BlockId {
        self.blocks.insert(SkeletonBlock::new())
    }

    pub fn effect(&self, site: SideEffectSite) -> &SideEffect<P, Lang> {
        &self.blocks[site.block].side_effects[site.index]
    }

    pub fn effect_mut(&mut self, site: SideEffectSite) -> &mut SideEffect<P, Lang> {
        &mut self.blocks[site.block].side_effects[site.index]
    }

    pub fn get_effect(&self, site: SideEffectSite) -> Option<&SideEffect<P, Lang>> {
        self.blocks.get(site.block)?.side_effects.get(site.index)
    }

    pub fn get_effect_mut(&mut self, site: SideEffectSite) -> Option<&mut SideEffect<P, Lang>> {
        self.blocks.get_mut(site.block)?.side_effects.get_mut(site.index)
    }

    /// Remove a snapshot-stable set of side-effect sites without invalidating
    /// any site before it has been consumed. Duplicate sites are harmless.
    pub fn remove_effect_sites(&mut self, effects: impl IntoIterator<Item = SideEffectSite>) {
        let mut by_block = std::collections::HashMap::<BlockId, Vec<usize>>::new();
        for site in effects {
            by_block.entry(site.block).or_default().push(site.index);
        }
        for (block, mut indices) in by_block {
            indices.sort_unstable();
            indices.dedup();
            for index in indices.into_iter().rev() {
                self.blocks[block].side_effects.remove(index);
            }
        }
    }

    /// Split a block immediately before one of its side effects.
    ///
    /// The returned continuation receives the selected effect and every
    /// following effect, plus the original terminator and any structured
    /// control metadata. The original block is terminated by an unconditional
    /// branch to the continuation.
    pub fn split_block_before_effect(&mut self, block: BlockId, effect_index: usize) -> BlockId {
        let continuation = self.create_block();
        let source = &mut self.blocks[block];
        let suffix = source.side_effects.split_off(effect_index);
        let control_header = source.control_header.take();
        let old_term = std::mem::replace(
            &mut source.term,
            SkeletonTerminator::Branch {
                target: continuation,
                args: Vec::new(),
            },
        );
        self.blocks[continuation].side_effects = suffix;
        self.blocks[continuation].term = old_term;
        self.blocks[continuation].control_header = control_header;
        continuation
    }

    /// Verify that every CFG edge supplies one argument per target block parameter.
    pub fn verify_branch_arities(&self) -> Result<(), String> {
        for (source, block) in &self.blocks {
            let check = |target: BlockId, args: &[NodeId]| {
                let target_block = self
                    .blocks
                    .get(target)
                    .ok_or_else(|| format!("branch from {source:?} targets an absent block {target:?}"))?;
                if args.len() != target_block.params.len() {
                    return Err(format!(
                        "branch from {source:?} to {target:?} supplies {} arguments for {} parameters",
                        args.len(),
                        target_block.params.len()
                    ));
                }
                Ok(())
            };
            match &block.term {
                SkeletonTerminator::Branch { target, args } => check(*target, args)?,
                SkeletonTerminator::CondBranch {
                    then_target,
                    then_args,
                    else_target,
                    else_args,
                    ..
                } => {
                    check(*then_target, then_args)?;
                    check(*else_target, else_args)?;
                }
                SkeletonTerminator::Return(_) | SkeletonTerminator::Unreachable => {}
            }
        }
        Ok(())
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
    pub fn build<P: Family, Lang: Language>(graph: &EGraph<P, Lang>) -> Self {
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

    pub fn effect<'a, P: Family, Lang: Language>(
        &self,
        graph: &'a EGraph<P, Lang>,
        result: NodeId,
    ) -> Option<&'a SideEffect<P, Lang>> {
        let site = self.site(result)?;
        let effect = graph.skeleton.blocks.get(site.block)?.side_effects.get(site.index)?;
        (effect.result == Some(result)).then_some(effect)
    }

    pub fn effect_mut<'a, P: Family, Lang: Language>(
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
pub struct EGraph<P: Family, Lang: Language> {
    /// All nodes (pure, union, params, constants, side-effect results),
    /// including their type, span, and canonical alias.
    pub nodes: SlotMap<NodeId, Node<P::Resource, Lang>>,
    /// Hash-cons table: NodeKey → existing NodeId.
    hash_cons: LookupMap<NodeKey<P::Resource, Lang>, NodeId>,
    /// Constant dedup cache.
    const_cache: LookupMap<Lang::Const, NodeId>,
    /// The CFG skeleton.
    pub skeleton: Skeleton<P, Lang>,
}

/// Graph state excluding indexes derived from that state.
///
/// Transformations may consume and rebuild an `EGraph` through this boundary
/// without gaining direct access to its hash-consing internals.
pub(super) struct EGraphParts<P: Family, Lang: Language> {
    pub(super) nodes: SlotMap<NodeId, Node<P::Resource, Lang>>,
    pub(super) skeleton: Skeleton<P, Lang>,
}

pub trait GraphResource: Clone + std::fmt::Debug + Eq + std::hash::Hash {}

impl<T> GraphResource for T where T: Clone + std::fmt::Debug + Eq + std::hash::Hash {}

impl<P: Family, Lang: Language> EGraph<P, Lang> {
    pub fn new() -> Self {
        EGraph {
            nodes: SlotMap::with_key(),
            hash_cons: LookupMap::new(),
            const_cache: LookupMap::new(),
            skeleton: Skeleton::new(),
        }
    }

    pub(super) fn into_parts(self) -> EGraphParts<P, Lang> {
        let Self {
            nodes,
            hash_cons: _,
            const_cache: _,
            skeleton,
        } = self;
        EGraphParts { nodes, skeleton }
    }

    pub(super) fn from_parts(parts: EGraphParts<P, Lang>) -> Self {
        let EGraphParts { nodes, skeleton } = parts;
        let mut graph = Self {
            nodes,
            hash_cons: LookupMap::new(),
            const_cache: LookupMap::new(),
            skeleton,
        };
        graph.rebuild_hash_cons();
        graph.rebuild_const_cache();
        graph
    }

    pub fn side_effect_index(&self) -> SideEffectIndex {
        SideEffectIndex::build(self)
    }

    fn pure_node_key(&self, id: NodeId) -> Option<NodeKey<P::Resource, Lang>> {
        let node = self.nodes.get(id)?;
        let ENode::Pure { op, operands } = &node.kind else {
            return None;
        };
        Some(NodeKey {
            op: op.clone(),
            operands: operands.clone(),
            ty: node.ty.clone(),
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
        self.nodes[id].kind = node;
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
        if !matches!(
            self.nodes.get(id).map(|node| &node.kind),
            Some(ENode::Pure { .. })
        ) {
            return false;
        }
        self.unindex_current_pure(id);
        if let ENode::Pure { op, operands } = &mut self.nodes[id].kind {
            update(op, operands);
        }
        self.index_current_pure(id);
        true
    }

    /// Change a node's result type while maintaining the pure-node hash-cons
    /// key when the node is hash-consed.
    pub fn retype_node(&mut self, id: NodeId, ty: Lang::Ty) {
        self.unindex_current_pure(id);
        self.nodes[id].ty = ty;
        self.index_current_pure(id);
    }

    /// Remove a function-parameter node and its graph-owned metadata.
    pub fn remove_func_param(&mut self, id: NodeId) -> bool {
        if !matches!(
            self.nodes.get(id).map(|node| &node.kind),
            Some(ENode::FuncParam { .. })
        ) {
            return false;
        }
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
            if self.nodes[id].alias == Some(old) {
                self.nodes[id].alias = Some(new);
            }
            match self.nodes.get(id).map(|node| &node.kind) {
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
                    if let ENode::Union { left, right } = &mut self.nodes[id].kind {
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

    /// Install canonical aliases produced by a graph rewrite. Alias ownership
    /// follows the source node, so later graph copies and removals cannot leave
    /// a detached side table behind.
    pub fn install_aliases(&mut self, aliases: impl IntoIterator<Item = (NodeId, NodeId)>) {
        for (source, target) in aliases {
            self.nodes[source].alias = Some(target);
        }
    }

    /// Rebuild the pure-node hash-cons table after a bulk rewrite that may
    /// have changed pure node operands, operators, or result types in place.
    pub fn rebuild_hash_cons(&mut self) {
        let mut rebuilt = LookupMap::new();
        for (id, node) in self.nodes.iter() {
            if matches!(&node.kind, ENode::Pure { .. }) {
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
            .filter_map(|(id, node)| match &node.kind {
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
            if matches!(&node.kind, ENode::Pure { .. }) {
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

    fn insert_node(&mut self, kind: ENode<P::Resource, Lang>, ty: Lang::Ty, span: Option<Span>) -> NodeId {
        self.nodes.insert(Node {
            kind,
            ty,
            span,
            alias: None,
        })
    }

    /// Allocate a function parameter node.
    pub fn add_func_param(&mut self, index: usize, ty: Lang::Ty) -> NodeId {
        self.insert_node(ENode::FuncParam { index }, ty, None)
    }

    /// Append a parameter to a block and allocate its corresponding node.
    pub fn add_block_param(&mut self, block: BlockId, ty: Lang::Ty) -> NodeId {
        let index = self.skeleton.blocks[block].params.len();
        let id = self.insert_node(ENode::BlockParam { block, index }, ty, None);
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
            match &mut self.nodes[param].kind {
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
        let id = self.insert_node(ENode::Constant(c.clone()), ty, None);
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
        let id = self.insert_node(ENode::Pure { op, operands }, ty, span);
        self.hash_cons.insert(key, id);
        id
    }

    /// Allocate a node for a side-effect result (not hash-consed).
    pub fn alloc_side_effect_result(&mut self, ty: Lang::Ty) -> NodeId {
        self.insert_node(ENode::SideEffectResult, ty, None)
    }

    /// Create a union node joining two alternatives.
    pub fn add_union(&mut self, left: NodeId, right: NodeId) -> NodeId {
        // Use the type of the left (they should be equivalent).
        let ty = self.nodes[left].ty.clone();
        self.insert_node(ENode::Union { left, right }, ty, None)
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
        debug_assert!(matches!(&self.nodes[id].kind, ENode::Pure { .. }));
        let original_kind = self.nodes[id].kind.clone();
        let original_ty = self.nodes[id].ty.clone();
        let original_span = self.nodes[id].span;
        let fresh = self.nodes.insert(Node {
            kind: original_kind,
            ty: original_ty,
            span: original_span,
            alias: None,
        });
        // The hash-cons key for the original node now belongs to its fresh id.
        if let Some(key) = self.pure_node_key(fresh) {
            self.hash_cons.insert(key, fresh);
        }
        self.nodes[id].kind = ENode::Union {
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
        debug_assert!(matches!(&self.nodes[id].kind, ENode::Pure { .. }));
        if let Some(key) = self.pure_node_key(id) {
            self.hash_cons.remove(&key);
        }
        self.nodes[id].kind = ENode::Union {
            left: better,
            right: better,
        };
    }

    /// Drop structured-control metadata whose header or required target block
    /// was removed by CFG simplification.
    pub fn retain_live_control_headers(&mut self) {
        let invalid = self
            .skeleton
            .blocks
            .iter()
            .filter_map(|(header, block)| {
                let control = block.control_header.as_ref()?;
                let valid = matches!(block.term, SkeletonTerminator::CondBranch { .. })
                    && match control {
                        ControlHeader::Loop {
                            merge,
                            continue_block,
                        } => {
                            self.skeleton.blocks.contains_key(*merge)
                                && self.skeleton.blocks.contains_key(*continue_block)
                        }
                        ControlHeader::Selection { merge } => self.skeleton.blocks.contains_key(*merge),
                    };
                (!valid).then_some(header)
            })
            .collect::<Vec<_>>();
        for header in invalid {
            self.skeleton.blocks[header].control_header = None;
        }
    }
}

// ---------------------------------------------------------------------------
// Program and body containers
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Func<P: Family, Lang: Language> {
    /// Stable identity used by segmented bodies that call this region.
    pub region: RegionId,
    /// Diagnostic and emitted-symbol metadata; never used to resolve a call.
    pub name: String,
    pub span: Span,
    pub linkage_name: Option<String>,
    pub params: Vec<(Lang::Ty, String)>,
    pub return_ty: Lang::Ty,
    pub graph: EGraph<P, Lang>,
}

impl<P: Family, Lang: Language> Func<P, Lang> {
    pub fn map_graph(mut self, map: impl FnOnce(EGraph<P, Lang>) -> EGraph<P, Lang>) -> Self {
        self.graph = map(self.graph);
        self
    }

    pub fn try_map_graph<E>(
        mut self,
        map: impl FnOnce(EGraph<P, Lang>) -> Result<EGraph<P, Lang>, E>,
    ) -> Result<Self, E> {
        self.graph = map(self.graph)?;
        Ok(self)
    }

    pub fn new(
        region: RegionId,
        name: String,
        span: Span,
        linkage_name: Option<String>,
        params: Vec<(Lang::Ty, String)>,
        return_ty: Lang::Ty,
        graph: EGraph<P, Lang>,
    ) -> Self {
        Self {
            region,
            name,
            span,
            linkage_name,
            params,
            return_ty,
            graph,
        }
    }

    /// Append one value to both sides of a segmented-body capture ABI.
    pub(crate) fn push_seg_body_capture(
        &mut self,
        body: &mut SegBody,
        capture: NodeId,
        ty: Lang::Ty,
        name: String,
    ) -> NodeId {
        let index = self.params.len();
        let parameter = self.graph.add_func_param(index, ty.clone());
        self.params.push((ty, name));
        body.captures.push(capture);
        parameter
    }

    /// Retain selected capture slots and compact both sides of the ABI.
    ///
    /// Leading lane/element parameters are always retained. Removed parameter
    /// nodes remain as unique out-of-ABI tombstones because dead pure arena
    /// nodes can still refer to them until demand-driven extraction.
    pub(crate) fn retain_seg_body_captures(
        &mut self,
        body: &mut SegBody,
        retained_captures: &SortedSet<usize>,
    ) -> Result<(), String> {
        let leading = body.leading_parameter_count(self)?;
        let parameter_count = self.params.len();
        if let Some(index) = retained_captures.iter().find(|index| **index >= body.captures.len()) {
            return Err(format!(
                "region `{}` cannot retain missing capture {index}",
                self.name
            ));
        }

        let mut retained = vec![false; parameter_count];
        retained[..leading].fill(true);
        for &capture in retained_captures {
            retained[leading + capture] = true;
        }

        let mut next_index = 0;
        let remapped = retained
            .iter()
            .map(|retain| {
                retain.then(|| {
                    let index = next_index;
                    next_index += 1;
                    index
                })
            })
            .collect::<Vec<_>>();
        let parameter_nodes = self
            .graph
            .nodes
            .iter()
            .filter_map(|(node, definition)| match &definition.kind {
                ENode::FuncParam { index } => Some((node, *index)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut tombstone_index = next_index;
        for (node, old_index) in parameter_nodes {
            if let Some(new_index) = remapped.get(old_index).copied().flatten() {
                self.graph.nodes[node].kind = ENode::FuncParam { index: new_index };
            } else {
                self.graph.nodes[node].kind = ENode::FuncParam {
                    index: tombstone_index,
                };
                tombstone_index += 1;
            }
        }

        self.params = std::mem::take(&mut self.params)
            .into_iter()
            .enumerate()
            .filter_map(|(index, parameter)| retained[index].then_some(parameter))
            .collect();
        body.captures = std::mem::take(&mut body.captures)
            .into_iter()
            .enumerate()
            .filter_map(|(index, capture)| retained[leading + index].then_some(capture))
            .collect();
        Ok(())
    }
}

/// A body-backed compile-time constant retained in EGIR until final
/// elaboration. Constant bodies have no parameters and must be proven pure.
#[derive(Clone, Debug)]
pub struct ConstantDef<P: Family, Lang: Language> {
    pub id: crate::GlobalId,
    /// Diagnostic and emitted-symbol metadata; never used to resolve a global.
    pub name: String,
    pub span: Span,
    pub return_ty: Lang::Ty,
    pub graph: EGraph<P, Lang>,
}

impl<P: Family, Lang: Language> ConstantDef<P, Lang> {
    pub fn map_graph(mut self, map: impl FnOnce(EGraph<P, Lang>) -> EGraph<P, Lang>) -> Self {
        self.graph = map(self.graph);
        self
    }

    pub fn try_map_graph<E>(
        mut self,
        map: impl FnOnce(EGraph<P, Lang>) -> Result<EGraph<P, Lang>, E>,
    ) -> Result<Self, E> {
        self.graph = map(self.graph)?;
        Ok(self)
    }
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
pub struct UnrealizedOutputRoute {
    pub source: SlotSource,
}

/// An output route whose concrete writers have been installed in the graph.
#[derive(Debug, Clone)]
pub struct RealizedOutputRoute {
    pub source: SlotSource,
    pub writers: Vec<OutputWriter>,
}

pub trait RemapBlockIds {
    fn remap_block_ids(&mut self, blocks: &LookupMap<BlockId, BlockId>);
}

impl RemapBlockIds for UnrealizedOutputRoute {
    fn remap_block_ids(&mut self, blocks: &LookupMap<BlockId, BlockId>) {
        self.source.block = blocks[&self.source.block];
    }
}

impl RemapBlockIds for RealizedOutputRoute {
    fn remap_block_ids(&mut self, blocks: &LookupMap<BlockId, BlockId>) {
        self.source.block = blocks[&self.source.block];
    }
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
pub struct EntryOutput<R, Route, Lang: Language> {
    pub inner: InterfaceEntryOutput<Lang::Ty>,
    pub resource: Option<R>,
    /// Every control-flow source capable of producing this declared output.
    pub routes: Vec<Route>,
}

impl<R, Route, Lang: Language> std::ops::Deref for EntryOutput<R, Route, Lang> {
    type Target = InterfaceEntryOutput<Lang::Ty>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<R, Route, Lang: Language> std::ops::DerefMut for EntryOutput<R, Route, Lang> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Clone, Debug)]
pub struct Entry<P: Family, ResourceDecl, Route, Lang: Language> {
    pub id: crate::EntryId,
    /// Host-facing entry symbol and diagnostic metadata.
    pub name: String,
    pub span: Span,
    pub execution_model: ExecutionModel,
    pub inputs: Vec<EntryInput<P::Resource, Lang>>,
    /// Structural association from source parameter to its ABI input slots.
    /// A parameter may expand into multiple slots (for example tuple views).
    pub parameter_inputs: Vec<Vec<super::program::InputSlotId>>,
    pub outputs: Vec<EntryOutput<P::Resource, Route, Lang>>,
    pub resource_declarations: Vec<ResourceDecl>,
    pub params: Vec<(Lang::Ty, String)>,
    pub return_ty: Lang::Ty,
    pub graph: EGraph<P, Lang>,
}

impl<P: Family, ResourceDecl: Clone, Route: Clone, Lang: Language> Entry<P, ResourceDecl, Route, Lang> {
    pub fn routes(&self) -> impl Iterator<Item = &Route> {
        self.outputs.iter().flat_map(|output| &output.routes)
    }

    pub fn routes_mut(&mut self) -> impl Iterator<Item = &mut Route> {
        self.outputs.iter_mut().flat_map(|output| &mut output.routes)
    }

    pub fn map_graph(mut self, map: impl FnOnce(EGraph<P, Lang>) -> EGraph<P, Lang>) -> Self {
        self.graph = map(self.graph);
        self
    }

    pub fn try_map_graph<E>(
        mut self,
        map: impl FnOnce(EGraph<P, Lang>) -> Result<EGraph<P, Lang>, E>,
    ) -> Result<Self, E> {
        self.graph = map(self.graph)?;
        Ok(self)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_resources(
        name: String,
        id: crate::EntryId,
        span: Span,
        execution_model: ExecutionModel,
        inputs: Vec<InterfaceEntryInput<Lang::Ty>>,
        outputs: Vec<InterfaceEntryOutput<Lang::Ty>>,
        resource_declarations: Vec<ResourceDecl>,
        params: Vec<(Lang::Ty, String)>,
        return_ty: Lang::Ty,
        graph: EGraph<P, Lang>,
    ) -> Self {
        let parameter_inputs = (0..params.len())
            .map(|index| {
                (index < inputs.len()).then(|| vec![super::program::InputSlotId(index)]).unwrap_or_default()
            })
            .collect();
        Self {
            id,
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
            parameter_inputs,
            outputs: outputs
                .into_iter()
                .map(|inner| EntryOutput {
                    inner,
                    resource: None,
                    routes: Vec::new(),
                })
                .collect(),
            resource_declarations,
            params,
            return_ty,
            graph,
        }
    }

    /// Retain selected original parameter indices and compact the entry
    /// interface and corresponding function-parameter nodes together.
    pub fn retain_parameter_indices(&mut self, retained: &SortedSet<usize>) {
        let mut kept = self
            .graph
            .nodes
            .iter()
            .filter_map(|(node, definition)| match &definition.kind {
                ENode::FuncParam { index } if retained.contains(index) => Some((*index, node)),
                _ => None,
            })
            .collect::<Vec<_>>();
        kept.sort_by_key(|(index, _)| *index);
        kept.dedup_by_key(|(index, _)| *index);

        let kept_slots = kept
            .iter()
            .flat_map(|(index, _)| self.parameter_inputs[*index].iter().copied())
            .collect::<LookupSet<_>>();
        let mut remapped_slots = LookupMap::new();
        self.inputs = std::mem::take(&mut self.inputs)
            .into_iter()
            .enumerate()
            .filter_map(|(old_index, input)| {
                let old_slot = super::program::InputSlotId(old_index);
                kept_slots.contains(&old_slot).then(|| {
                    let new_slot = super::program::InputSlotId(remapped_slots.len());
                    remapped_slots.insert(old_slot, new_slot);
                    input
                })
            })
            .collect();
        self.parameter_inputs = kept
            .iter()
            .map(|(index, _)| {
                self.parameter_inputs[*index].iter().map(|slot| remapped_slots[slot]).collect()
            })
            .collect();
        self.params = kept.iter().map(|(index, _)| self.params[*index].clone()).collect();

        let retained_nodes = kept.iter().map(|(_, node)| *node).collect::<LookupSet<_>>();
        let removed = self
            .graph
            .nodes
            .iter()
            .filter_map(|(node, definition)| {
                (matches!(&definition.kind, ENode::FuncParam { .. }) && !retained_nodes.contains(&node))
                    .then_some(node)
            })
            .collect::<Vec<_>>();
        for node in removed {
            self.graph.remove_func_param(node);
        }
        for (new_index, (_, node)) in kept.into_iter().enumerate() {
            if let Some(ENode::FuncParam { index }) =
                self.graph.nodes.get_mut(node).map(|node| &mut node.kind)
            {
                *index = new_index;
            }
        }
    }

    /// Drop structured-control metadata whose header or required target block
    /// was removed by CFG simplification.
    pub fn retain_live_control_headers(&mut self) {
        self.graph.retain_live_control_headers();
    }
}

impl<P: Family, ResourceDecl, Route, Lang: Language> Entry<P, ResourceDecl, Route, Lang> {
    /// Consume an entry while changing only the representation of each
    /// output route.
    pub fn map_output_routes<T>(self, mut map: impl FnMut(Route) -> T) -> Entry<P, ResourceDecl, T, Lang> {
        let Self {
            name,
            id,
            span,
            execution_model,
            inputs,
            parameter_inputs,
            outputs,
            resource_declarations,
            params,
            return_ty,
            graph,
        } = self;
        Entry {
            id,
            name,
            span,
            execution_model,
            inputs,
            parameter_inputs,
            outputs: outputs
                .into_iter()
                .map(|output| EntryOutput {
                    inner: output.inner,
                    resource: output.resource,
                    routes: output.routes.into_iter().map(&mut map).collect(),
                })
                .collect(),
            resource_declarations,
            params,
            return_ty,
            graph,
        }
    }
}

/// Whole-program EGIR container at one externally visible checkpoint.
#[derive(Debug)]
pub struct Program<Tag, Shape: ProgramShape, GlobalContext, Lang: Language> {
    pub functions: Vec<Func<Shape::Family, Lang>>,
    pub externs: Vec<ExternDecl<Lang::Ty>>,
    pub entry_points: Vec<Entry<Shape::Family, Shape::ResourceDecl, Shape::OutputRoute, Lang>>,
    pub constants: Vec<ConstantDef<Shape::Family, Lang>>,
    /// Program-owned IR data selected by this checkpoint.
    pub data: Shape::ProgramData,
    /// Program-wide state available at this exact pipeline checkpoint.
    pub global_context: GlobalContext,
    pub(crate) state: std::marker::PhantomData<fn() -> Tag>,
}

/// Stable address of one graph-bearing body within a program snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BodySite {
    Function(RegionId),
    Entry(usize),
    Constant(usize),
}

/// One graph-bearing program member removed from a program for a consuming
/// rewrite.
pub enum Body<P: Family, ResourceDecl, OutputRoute, Lang: Language> {
    Function(Func<P, Lang>),
    Entry(Entry<P, ResourceDecl, OutputRoute, Lang>),
    Constant(ConstantDef<P, Lang>),
}

impl<Tag, Shape: ProgramShape, GlobalContext, Lang: Language> Program<Tag, Shape, GlobalContext, Lang> {
    pub fn from_parts(
        functions: Vec<Func<Shape::Family, Lang>>,
        externs: Vec<ExternDecl<Lang::Ty>>,
        entry_points: Vec<Entry<Shape::Family, Shape::ResourceDecl, Shape::OutputRoute, Lang>>,
        constants: Vec<ConstantDef<Shape::Family, Lang>>,
        data: Shape::ProgramData,
        global_context: GlobalContext,
    ) -> Self {
        Self {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state: std::marker::PhantomData,
        }
    }

    pub fn body_graph(&self, site: BodySite) -> Option<&EGraph<Shape::Family, Lang>> {
        match site {
            BodySite::Function(region) => {
                self.functions.iter().find(|function| function.region == region).map(|body| &body.graph)
            }
            BodySite::Entry(index) => self.entry_points.get(index).map(|body| &body.graph),
            BodySite::Constant(index) => self.constants.get(index).map(|body| &body.graph),
        }
    }

    /// Change only the top-level proof tag while preserving all stored types.
    pub fn retag<NewTag>(self) -> Program<NewTag, Shape, GlobalContext, Lang> {
        let Self {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state: _,
        } = self;
        Program {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state: std::marker::PhantomData,
        }
    }

    /// Consume and rebuild every graph-bearing body while moving all
    /// non-graph fields directly into the resulting program.
    pub fn map_graphs(
        self,
        mut map: impl FnMut(BodySite, EGraph<Shape::Family, Lang>) -> EGraph<Shape::Family, Lang>,
    ) -> Self {
        match self.try_map_graphs(|site, graph| Ok::<_, std::convert::Infallible>(map(site, graph))) {
            Ok(program) => program,
            Err(error) => match error {},
        }
    }

    /// Fallible counterpart to [`Self::map_graphs`].
    pub fn try_map_graphs<E>(
        self,
        mut map: impl FnMut(BodySite, EGraph<Shape::Family, Lang>) -> Result<EGraph<Shape::Family, Lang>, E>,
    ) -> Result<Self, E> {
        let Self {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state,
        } = self;
        let functions = functions
            .into_iter()
            .map(|function| {
                let site = BodySite::Function(function.region);
                function.try_map_graph(|graph| map(site, graph))
            })
            .collect::<Result<_, E>>()?;
        let entry_points = entry_points
            .into_iter()
            .enumerate()
            .map(|(index, entry)| entry.try_map_graph(|graph| map(BodySite::Entry(index), graph)))
            .collect::<Result<_, E>>()?;
        let constants = constants
            .into_iter()
            .enumerate()
            .map(|(index, constant)| constant.try_map_graph(|graph| map(BodySite::Constant(index), graph)))
            .collect::<Result<_, E>>()?;
        Ok(Self {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state,
        })
    }

    /// Fallibly rebuild every graph-bearing body while allowing the rewrite
    /// to update both program-owned data and carried global state.
    pub fn try_map_graphs_with_state<E>(
        self,
        mut map: impl FnMut(
            BodySite,
            EGraph<Shape::Family, Lang>,
            &mut Shape::ProgramData,
            &mut GlobalContext,
        ) -> Result<EGraph<Shape::Family, Lang>, E>,
    ) -> Result<Self, E> {
        let Self {
            functions,
            externs,
            entry_points,
            constants,
            mut data,
            mut global_context,
            state,
        } = self;
        let functions = functions
            .into_iter()
            .map(|function| {
                let site = BodySite::Function(function.region);
                function.try_map_graph(|graph| map(site, graph, &mut data, &mut global_context))
            })
            .collect::<Result<_, E>>()?;
        let entry_points = entry_points
            .into_iter()
            .enumerate()
            .map(|(index, entry)| {
                entry.try_map_graph(|graph| {
                    map(BodySite::Entry(index), graph, &mut data, &mut global_context)
                })
            })
            .collect::<Result<_, E>>()?;
        let constants = constants
            .into_iter()
            .enumerate()
            .map(|(index, constant)| {
                constant.try_map_graph(|graph| {
                    map(BodySite::Constant(index), graph, &mut data, &mut global_context)
                })
            })
            .collect::<Result<_, E>>()?;
        Ok(Self {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state,
        })
    }

    /// Consume the program and append synthesized callable regions while
    /// moving every existing member unchanged.
    pub fn extend_functions(self, additional: impl IntoIterator<Item = Func<Shape::Family, Lang>>) -> Self {
        let Self {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state,
        } = self;
        Self {
            functions: functions.into_iter().chain(additional).collect(),
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state,
        }
    }

    /// Consume the program and rebuild only its program-owned data.
    pub fn map_data(self, map: impl FnOnce(Shape::ProgramData) -> Shape::ProgramData) -> Self {
        let Self {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state,
        } = self;
        Self {
            functions,
            externs,
            entry_points,
            constants,
            data: map(data),
            global_context,
            state,
        }
    }

    /// Consume and replace exactly one graph-bearing body. All other bodies
    /// move directly into the rebuilt program.
    pub fn rewrite_body(
        self,
        site: BodySite,
        rewrite: impl FnOnce(
            Body<Shape::Family, Shape::ResourceDecl, Shape::OutputRoute, Lang>,
        ) -> Body<Shape::Family, Shape::ResourceDecl, Shape::OutputRoute, Lang>,
    ) -> Self {
        match self.try_rewrite_body(site, |body| Ok::<_, std::convert::Infallible>(rewrite(body))) {
            Ok(program) => program,
            Err(error) => match error {},
        }
    }

    /// Fallible counterpart to [`Self::rewrite_body`].
    pub fn try_rewrite_body<E>(
        self,
        site: BodySite,
        rewrite: impl FnOnce(
            Body<Shape::Family, Shape::ResourceDecl, Shape::OutputRoute, Lang>,
        )
            -> Result<Body<Shape::Family, Shape::ResourceDecl, Shape::OutputRoute, Lang>, E>,
    ) -> Result<Self, E> {
        let Self {
            functions,
            externs,
            entry_points,
            constants,
            data,
            global_context,
            state,
        } = self;
        let mut rewrite = Some(rewrite);
        let mut rebuilt_functions = Vec::with_capacity(functions.len());
        for function in functions {
            if site != BodySite::Function(function.region) {
                rebuilt_functions.push(function);
                continue;
            }
            match rewrite.take().expect("body patch applied more than once")(Body::Function(function))? {
                Body::Function(function) => rebuilt_functions.push(function),
                _ => panic!("function body patch returned a different body kind"),
            }
        }
        let mut rebuilt_entries = Vec::with_capacity(entry_points.len());
        for (index, entry) in entry_points.into_iter().enumerate() {
            if site != BodySite::Entry(index) {
                rebuilt_entries.push(entry);
                continue;
            }
            match rewrite.take().expect("body patch applied more than once")(Body::Entry(entry))? {
                Body::Entry(entry) => rebuilt_entries.push(entry),
                _ => panic!("entry body patch returned a different body kind"),
            }
        }
        let mut rebuilt_constants = Vec::with_capacity(constants.len());
        for (index, constant) in constants.into_iter().enumerate() {
            if site != BodySite::Constant(index) {
                rebuilt_constants.push(constant);
                continue;
            }
            match rewrite.take().expect("body patch applied more than once")(Body::Constant(constant))? {
                Body::Constant(constant) => rebuilt_constants.push(constant),
                _ => panic!("constant body patch returned a different body kind"),
            }
        }
        assert!(
            rewrite.is_none(),
            "body patch targeted a body absent from the program"
        );
        Ok(Self {
            functions: rebuilt_functions,
            externs,
            entry_points: rebuilt_entries,
            constants: rebuilt_constants,
            data,
            global_context,
            state,
        })
    }
}
