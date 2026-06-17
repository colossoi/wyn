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
use polytype::Type;
use slotmap::{SlotMap, new_key_type};
use smallvec::SmallVec;
use std::collections::HashMap;

new_key_type! {
    /// Identity of a node in the e-graph. Every pure node, union node,
    /// block param, function param, and constant gets one.
    pub struct NodeId;
}

// ---------------------------------------------------------------------------
// PureOp — operator identity for hash-consing (re-exported from `crate::op`)
// ---------------------------------------------------------------------------

/// Alias for the shared `OpTag` enum (operator identity without operands).
/// Operands live in `ENode::Pure { operands }`.
pub use crate::op::{OpTag as PureOp, PureViewSource};

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
pub struct NodeKey {
    pub op: PureOp,
    pub operands: SmallVec<[NodeId; 4]>,
    pub ty: Type<TypeName>,
}

// ---------------------------------------------------------------------------
// ENode — what lives in the sea of nodes
// ---------------------------------------------------------------------------

/// A node in the e-graph.
#[derive(Clone, Debug)]
pub enum ENode {
    /// A pure instruction, hash-consed and floating.
    Pure {
        op: PureOp,
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

impl ENode {
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
pub struct SideEffect {
    /// What this side-effect is. Either an SSA `InstKind` that survives into
    /// the final `FuncBody`, or an intermediate `PendingSoac` that must be
    /// rewritten by `soac_expand` before `elaborate` runs.
    pub kind: SideEffectKind,
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
pub enum SideEffectKind {
    /// An SSA-level effectful instruction (`Alloca` / `Load` / `Store` /
    /// `Call` / `Intrinsic` / `StorageView*` / `OutputPtr` with effects).
    /// This is what lands in the final `FuncBody` after elaboration.
    Inst(InstKind),
    /// A placeholder for an unexpanded SOAC. Produced by `from_tlc` and
    /// consumed by `soac_expand`. Never reaches elaborate.
    Pending(PendingSoac),
}

/// Where an array-producing SOAC's per-iteration result is written.
/// Defined in `crate::tlc` and re-exported here so EGIR consumers see
/// the same type that TLC's `SoacOp::{Map, Scan, Filter}` carry.
///
/// Operand layouts in `PendingSoac` are variant-dependent and follow
/// each destination:
/// - `Fresh` (Map): `[input_0, ..., input_{n-1}, ...captures]`
/// - `Fresh` (Scan): `[input, init, ...captures]`
/// - `OutputView` (Map): `[input_0, ..., ...captures, output_view]`
/// - `OutputView` (Scan): `[input, init, ...captures, output_view]`
/// - `OutputView` (Screma): `[inputs..., init_accs..., map_captures...,
///   acc_step_captures..., acc_reduce_op_captures..., output_views...]`
/// - `InputBuffer`: operand layout matches `Fresh`; the difference is
///   that the result aliases `inputs[0]` instead of a fresh allocation.
pub use crate::tlc::{ScremaAccumulator, SoacDestination};

#[derive(Clone, Debug)]
pub struct PendingScremaAccumulator {
    pub kind: ScremaAccumulator,
    pub step_func: String,
    pub reduce_op_func: String,
    pub step_capture_count: usize,
    pub reduce_op_capture_count: usize,
}

/// An unexpanded SOAC operation held in the skeleton until `soac_expand` rewrites
/// it into an explicit loop. All operand NodeIds live in `SideEffect.operand_nodes`
/// with a variant-specific layout (documented per-variant in `soac_expand`).
#[derive(Clone, Debug)]
pub enum PendingSoac {
    /// Multi-result map+accumulator: one pass writes mapped outputs and
    /// threads accumulator outputs. Operand layout:
    /// `[inputs..., init_accs..., map_captures..., acc_step_captures...,
    /// acc_reduce_op_captures..., output_views...]`.
    Screma {
        map_funcs: Vec<String>,
        accumulators: Vec<PendingScremaAccumulator>,
        input_array_types: Vec<Type<TypeName>>,
        input_elem_types: Vec<Type<TypeName>>,
        map_output_elem_types: Vec<Type<TypeName>>,
        map_capture_counts: Vec<usize>,
        map_destinations: Vec<SoacDestination>,
        acc_destinations: Vec<SoacDestination>,
    },
    /// `filter pred input` carrying the elements of `input` that
    /// satisfied the predicate, plus a runtime count.
    /// Operands: `[input, ...pred_captures]`.
    ///
    /// Two lowerings, keyed by `scratch_out`:
    /// - `None` (static-capacity input): the result is a function-local
    ///   `Array[T, Size(N), Bounded]` `{buffer, len}` struct where N is the
    ///   input's static size (`output_capacity_size`). The runtime `len`
    ///   field is the count of accepted elements.
    /// - `Some(br)` (runtime-sized input): the result is a runtime-length
    ///   storage view over the reserved scratch buffer at `br`, typed
    ///   `Array[T, View, _, Region(br)]`. The surviving count is the view's
    ///   `len` operand (an SSA value). A runtime-sized input cannot back a
    ///   function-local array, so it compacts into a descriptor-bound buffer.
    Filter {
        pred_func: String,
        input_array_type: Type<TypeName>,
        input_elem_type: Type<TypeName>,
        /// `Size(N)` — the input's static capacity (static lowering only;
        /// for the runtime lowering this is unused, the capacity is the
        /// scratch buffer's host-sized length).
        output_capacity_size: Type<TypeName>,
        /// `Fresh` (allocate a new `[N]T` buffer) or `InputBuffer`
        /// (reuse the input array's backing slot as the output —
        /// the result `View` aliases the input). `OutputView` is
        /// not yet supported for Filter. Ignored when `scratch_out` is set.
        destination: SoacDestination,
        /// `Some(br)` selects the runtime scratch-view lowering, compacting
        /// kept elements into the storage buffer at `br`. `None` is the
        /// static function-local Bounded lowering.
        scratch_out: Option<crate::BindingRef>,
        /// `Some(br)` (runtime lowering only) makes the loop also store the
        /// surviving count into `br[0]` — a host-readable length cell paired
        /// with the output buffer when the filter result is a compute-entry
        /// output. `None` when the count flows only as the result view's `len`
        /// operand (in-kernel consumers like `reduce` / `length`).
        len_out: Option<crate::BindingRef>,
    },
    /// `scatter`: over the parallel `input` arrays, the lifted `func` yields an
    /// `(index, value)` pair per element, written as `dest[index] = value`;
    /// out-of-bounds indices are ignored (Futhark semantics). Lowered serially
    /// (a `tid==0`-guarded loop) in this cut; the result is dummy (the in-place
    /// writes are the effect). Operands: `[dest_view, inputs.., captures..]` —
    /// `dest_view` is the destination's lowered `StorageView`, and the trailing
    /// `capture_count` operands are the envelope captures.
    Scatter {
        func: String,
        input_array_types: Vec<Type<TypeName>>,
        input_elem_types: Vec<Type<TypeName>>,
        capture_count: usize,
        index_type: Type<TypeName>,
        value_type: Type<TypeName>,
        dest_elem_type: Type<TypeName>,
    },
    /// Wrapper marking a pointwise Screma as parallel at the entry
    /// boundary. The `egir::parallelize` pass tags a planned compute
    /// entry's tail SOAC with this; `soac_expand` dispatches to
    /// `build_parallel_screma_maps` instead of the serial-loop builder.
    /// Operand layout is identical to the inner Screma's — soac_expand
    /// peels the wrapper before consuming operands.
    Parallel {
        serial: Box<PendingSoac>,
    },
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

/// A block in the skeleton CFG.
#[derive(Clone, Debug)]
pub struct SkeletonBlock {
    /// Block parameters as NodeIds.
    pub params: Vec<NodeId>,
    /// Effectful instructions, in order.
    pub side_effects: Vec<SideEffect>,
    /// Block terminator.
    pub term: SkeletonTerminator,
}

impl SkeletonBlock {
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
pub struct Skeleton {
    pub entry: BlockId,
    pub blocks: SlotMap<BlockId, SkeletonBlock>,
}

impl Skeleton {
    pub fn new() -> Self {
        let mut blocks = SlotMap::with_key();
        let entry = blocks.insert(SkeletonBlock::new());
        Skeleton { entry, blocks }
    }

    pub fn create_block(&mut self) -> BlockId {
        self.blocks.insert(SkeletonBlock::new())
    }
}

// ---------------------------------------------------------------------------
// EGraph — the main container
// ---------------------------------------------------------------------------

/// The acyclic e-graph: a sea of pure nodes + a CFG skeleton of side effects.
#[derive(Clone, Debug)]
pub struct EGraph {
    /// All nodes (pure, union, params, constants, side-effect results).
    pub nodes: SlotMap<NodeId, ENode>,
    /// Type of each node's result.
    pub types: HashMap<NodeId, Type<TypeName>>,
    /// Hash-cons table: NodeKey → existing NodeId.
    pub hash_cons: HashMap<NodeKey, NodeId>,
    /// Constant dedup cache.
    pub const_cache: HashMap<ConstantValue, NodeId>,
    /// The CFG skeleton.
    pub skeleton: Skeleton,
    /// Source span associated with each pure node (first-writer-wins —
    /// later interns of the same hash-consed node keep the original span).
    pub node_spans: HashMap<NodeId, Span>,
}

impl EGraph {
    pub fn new() -> Self {
        EGraph {
            nodes: SlotMap::with_key(),
            types: HashMap::new(),
            hash_cons: HashMap::new(),
            const_cache: HashMap::new(),
            skeleton: Skeleton::new(),
            node_spans: HashMap::new(),
        }
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
        op: PureOp,
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
        op: PureOp,
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
}

// ---------------------------------------------------------------------------
// Conversion helpers: InstKind ↔ PureOp
// ---------------------------------------------------------------------------

/// Build an `InstKind::Op` from a `PureOp` + operands as `ValueRef::Ssa(ValueId)`.
/// Panics on the place-producing tags (`ViewIndex`, `OutputSlot`) — those need
/// `elaborate`'s place-aware path which allocates a fresh `PlaceId`.
pub fn rebuild_inst_kind(op: &PureOp, operands: &[crate::ssa::framework::ValueId]) -> InstKind {
    use crate::ssa::types::ValueRef;
    if matches!(op, PureOp::ViewIndex | PureOp::OutputSlot { .. }) {
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
