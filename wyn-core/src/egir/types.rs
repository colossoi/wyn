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
pub struct SideEffect<R = super::program::SemanticResourceRef> {
    /// Stable semantic identity assigned after segmentation and preserved by
    /// every projection. Synthesized physical-only effects use `None`.
    pub semantic_id: Option<super::program::SemanticOpId>,
    /// What this side-effect is. Either an SSA `InstKind` that survives into
    /// the final `FuncBody`, or an intermediate `EgirSoac` that must be
    /// rewritten by `soac_expand` before `elaborate` runs.
    pub kind: SideEffectKind<R>,
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
pub enum SideEffectKind<R = super::program::SemanticResourceRef> {
    /// An SSA-level effectful instruction (`Alloca` / `Load` / `Store` /
    /// `Call` / `Intrinsic` / `StorageView*` / `OutputPtr` with effects).
    /// This is what lands in the final `FuncBody` after elaboration.
    Inst(InstKind<R>),
    /// A placeholder for an unexpanded SOAC. Produced by `from_tlc` and
    /// consumed by `soac_expand`. Never reaches elaborate.
    Soac(EgirSoac<R>),
}

/// Where an array-producing SOAC's per-iteration result is written. TLC only
/// supplies the logical `Fresh`/`UniqueInput` ownership fact; the physical
/// `InputBuffer`/`OutputView` choices exist exclusively in EGIR.
///
/// Operand layouts in `EgirSoac` are variant-dependent and follow
/// each destination:
/// - `Fresh` (Map): `[input_0, ..., input_{n-1}, ...captures]`
/// - `Fresh` (Scan): `[input, init, ...captures]`
/// - `OutputView` (Map): `[input_0, ..., ...captures, output_view]`
/// - `OutputView` (Scan): `[input, init, ...captures, output_view]`
/// - `OutputView` (Screma): `[inputs..., init_accs..., map_captures...,
///   acc_step_captures..., acc_reduce_op_captures..., output_views...]`
/// - `InputBuffer`: operand layout matches `Fresh`; the difference is
///   that the result aliases `inputs[0]` instead of a fresh allocation.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScremaAccumulator {
    Reduce,
    Scan,
}

#[derive(Clone, Debug)]
pub struct ScremaOperator {
    pub kind: ScremaAccumulator,
    pub step: SegBody,
    pub combine: SegBody,
    /// Input-array positions consumed by `step`, after its accumulator.
    /// Empty on legacy construction means every Screma input.
    pub input_indices: Vec<usize>,
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

/// Placement selected independently of the backend scheduling strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegPlacement {
    Kernel,
    LaneLocal,
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

/// Conflict semantics for histogram/scatter updates. Ordered overwrite is the
/// source-language scatter behavior and therefore remains serial unless a
/// later proof replaces it with a parallel-safe policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistUpdatePolicy {
    OrderedOverwrite,
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

/// A reduce/scan operator of a `Seg` op: the (fused map+)step body, the bare
/// combiner, and whether it commutes (lets the tree-reduce reorder operands).
/// The neutral element is supplied as an `init_acc` operand, matching the
/// `Screma` layout. This is the reified `SegBinOp`; cross-lane combination is
/// explicit in the operator rather than implicit in the loop body.
#[derive(Clone, Debug)]
pub struct SegBinOp {
    pub kind: ScremaAccumulator,
    pub step: SegBody,
    pub combine: SegBody,
    /// Input-array positions consumed by the per-element step body.
    pub input_indices: Vec<usize>,
    /// Neutral value for this operator, interned in the surrounding EGraph.
    pub neutral: NodeId,
    /// Logical vectorized-operator dimensions. Empty for today's scalar and
    /// aggregate Wyn reductions; retained explicitly for Futhark-style
    /// vectorized operators rather than baking that assumption into lowering.
    pub shape: Vec<NodeId>,
    pub commutative: bool,
}

/// The semantic operation performed over a segmented iteration space.
///
/// This mirrors Futhark's `SegMap` / `SegRed` / `SegScan` distinction: the
/// operator kind is explicit in EGIR instead of being rediscovered from a
/// generic Screma accumulator list during lowering. `Scatter { space: Some }`
/// remains the current `SegHist` representation until scatter scheduling is
/// migrated into this same boundary.
#[derive(Clone, Debug)]
pub enum SegOpKind {
    SegMap,
    SegRed {
        operators: Vec<SegBinOp>,
    },
    SegScan {
        operators: Vec<SegBinOp>,
    },
    /// A same-space Screma containing more than one accumulator class. It is
    /// retained semantically and scheduled conservatively until the target
    /// lowering can split the shared pure mapping region safely.
    SegComposite {
        operators: Vec<SegBinOp>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilterWorkBuffers<R = super::program::SemanticResourceRef> {
    pub flags: R,
    pub offsets: R,
    pub block_sums: R,
    pub block_offsets: R,
}

#[derive(Clone, Debug)]
/// Storage contract for a filter result; local and runtime-only fields cannot
/// coexist.
pub enum FilterOutput<R = super::program::SemanticResourceRef> {
    Local {
        capacity: Type<TypeName>,
        destination: SoacDestination,
    },
    Runtime {
        scratch: R,
        length: RuntimeFilterLength<R>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Where a runtime filter's surviving count is observable.
pub enum RuntimeFilterLength<R = super::program::SemanticResourceRef> {
    ViewOnly,
    EntryOutput(R),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// One executable filter lowering selected by terminal scheduling.
pub enum FilterPlan<R = super::program::SemanticResourceRef> {
    Serial,
    Flags(FilterWorkBuffers<R>),
    Scan(FilterWorkBuffers<R>),
    Scatter(FilterWorkBuffers<R>),
}

#[derive(Clone, Debug)]
/// Facts known about a filter at each compiler boundary.
pub enum FilterState<R = super::program::SemanticResourceRef> {
    Raw,
    Semantic {
        space: SegSpace<R>,
    },
    Scheduled {
        space: SegSpace<R>,
        plan: FilterPlan<R>,
    },
}

#[derive(Clone, Debug)]
/// Execution classification for scatter/histogram operations.
pub enum HistExecution<R = super::program::SemanticResourceRef> {
    Raw,
    Serial,
    Segmented(SegSpace<R>),
}

/// An unexpanded SOAC operation held in the skeleton until `soac_expand` rewrites
/// it into an explicit loop. All operand NodeIds live in `SideEffect.operand_nodes`
/// with a variant-specific layout (documented per-variant in `soac_expand`).
#[derive(Clone, Debug)]
pub enum EgirSoac<R = super::program::SemanticResourceRef> {
    /// Multi-result map+accumulator: one pass writes mapped outputs and
    /// threads accumulator outputs. Operand layout:
    /// `[inputs..., init_accs..., map_captures..., acc_step_captures...,
    /// acc_reduce_op_captures..., output_views...]`.
    Screma {
        map_bodies: Vec<SegBody>,
        accumulators: Vec<ScremaOperator>,
        input_array_types: Vec<Type<TypeName>>,
        input_elem_types: Vec<Type<TypeName>>,
        map_output_elem_types: Vec<Type<TypeName>>,
        /// Which inputs each map func consumes: `map_input_indices[k]` lists the
        /// positions into the input arrays whose elements feed `map_funcs[k]`,
        /// in order, before its captures. Invariant: one entry per map func.
        /// Default (every func reads every input) is `(0..n_inputs)` per func.
        map_input_indices: Vec<Vec<usize>>,
        map_destinations: Vec<SoacDestination>,
        acc_destinations: Vec<SoacDestination>,
    },
    /// `filter pred input` carrying the elements of `input` that
    /// satisfied the predicate, plus a runtime count.
    /// Operands: `[input, ...pred_captures]`.
    ///
    /// `output` distinguishes bounded local storage from runtime scratch, and
    /// `state` carries only the facts established at the current compiler
    /// phase.
    Filter {
        state: FilterState<R>,
        /// `Some(body)` folds an elementwise producer `map(f, …)` in: per element
        /// the loop computes `v = f(x)` with the body's explicit captures,
        /// tests `pred(v)`, and keeps `v`. `None` is a plain
        /// filter (the surviving input element is kept and `pred` tests it).
        map_body: Option<SegBody>,
        /// The filter's output element type — `f`'s return type when a map is
        /// fused, else `input_elem_type`. Sizes the result buffer/view.
        output_elem_type: Type<TypeName>,
        pred_body: SegBody,
        input_array_type: Type<TypeName>,
        input_elem_type: Type<TypeName>,
        /// Complete local or runtime output-storage contract.
        output: FilterOutput<R>,
    },
    /// `scatter`: over the parallel `input` arrays, the lifted `func` yields an
    /// `(index, value)` pair per element, written as `dest[index] = value`;
    /// out-of-bounds indices are ignored (Futhark semantics). Lowered serially
    /// (a `tid==0`-guarded loop) in this cut; the result is dummy (the in-place
    /// writes are the effect). Operands: `[dest_view, inputs.., captures..]` —
    /// `dest_view` is the destination's lowered `StorageView`, and the trailing
    /// `capture_count` operands are the envelope captures.
    Hist {
        body: SegBody,
        input_array_types: Vec<Type<TypeName>>,
        input_elem_types: Vec<Type<TypeName>>,
        index_type: Type<TypeName>,
        value_type: Type<TypeName>,
        dest_elem_type: Type<TypeName>,
        update_policy: HistUpdatePolicy,
        /// Raw, serial, or segmented execution classification.
        execution: HistExecution<R>,
    },
    /// A reified parallel SOAC — the `SegOp`. Semantically a (1-D) map nest
    /// over `space`, with pointwise `map_funcs` lanes and optional reduce/scan
    /// `accumulators` (`SegBinOp`s) combining across lanes. Replaces the old
    /// `Parallel { serial }` wrapper: `egir::parallelize` builds this from a
    /// semantic compute operation, and terminal lowering drives
    /// lowering (lane-indexed map kernel; chunked two-phase reduce; three-phase
    /// scan) from its fields. Operand layout matches the serial `Screma`'s.
    Seg {
        space: SegSpace<R>,
        placement: SegPlacement,
        kind: SegOpKind,
        map_bodies: Vec<SegBody>,
        input_array_types: Vec<Type<TypeName>>,
        input_elem_types: Vec<Type<TypeName>>,
        map_output_elem_types: Vec<Type<TypeName>>,
        map_input_indices: Vec<Vec<usize>>,
        map_destinations: Vec<SoacDestination>,
        acc_destinations: Vec<SoacDestination>,
        result_types: Vec<Type<TypeName>>,
        /// Entry-output slots written by this operation, fixed at semantic
        /// construction time rather than recovered while scheduling.
        output_slots: Vec<usize>,
        /// Conservative logical resource summary owned by the semantic op.
        resources: Vec<SegResourceAccess<R>>,
    },
}

impl SegOpKind {
    /// The reduce/scan operators, empty for a plain `SegMap`.
    pub fn operators(&self) -> &[SegBinOp] {
        match self {
            SegOpKind::SegMap => &[],
            SegOpKind::SegRed { operators }
            | SegOpKind::SegScan { operators }
            | SegOpKind::SegComposite { operators } => operators,
        }
    }
}

/// Classify an already-built operator list into the matching `SegOpKind`:
/// empty is a map, all-reduce a `SegRed`, all-scan a `SegScan`, otherwise a
/// mixed `SegComposite`. Shared by segmentation and horizontal fusion.
pub fn reify_seg_kind_operators(operators: Vec<SegBinOp>) -> SegOpKind {
    if operators.is_empty() {
        SegOpKind::SegMap
    } else if operators.iter().all(|op| matches!(op.kind, ScremaAccumulator::Reduce)) {
        SegOpKind::SegRed { operators }
    } else if operators.iter().all(|op| matches!(op.kind, ScremaAccumulator::Scan)) {
        SegOpKind::SegScan { operators }
    } else {
        SegOpKind::SegComposite { operators }
    }
}

impl<R> EgirSoac<R> {
    pub fn try_map_resources<S, E>(self, mut map: impl FnMut(R) -> Result<S, E>) -> Result<EgirSoac<S>, E> {
        fn map_space<R, S, E>(
            space: SegSpace<R>,
            map: &mut impl FnMut(R) -> Result<S, E>,
        ) -> Result<SegSpace<S>, E> {
            Ok(SegSpace {
                level: space.level,
                dims: space
                    .dims
                    .into_iter()
                    .map(|extent| match extent {
                        SegExtent::Fixed(value) => Ok(SegExtent::Fixed(value)),
                        SegExtent::PushConstant { node, offset } => {
                            Ok(SegExtent::PushConstant { node, offset })
                        }
                        SegExtent::ResourceLength {
                            node,
                            resource,
                            elem_bytes,
                        } => Ok(SegExtent::ResourceLength {
                            node,
                            resource: map(resource)?,
                            elem_bytes,
                        }),
                        SegExtent::Value(node) => Ok(SegExtent::Value(node)),
                    })
                    .collect::<Result<Vec<_>, E>>()?,
            })
        }

        fn map_work<R, S, E>(
            work: FilterWorkBuffers<R>,
            map: &mut impl FnMut(R) -> Result<S, E>,
        ) -> Result<FilterWorkBuffers<S>, E> {
            Ok(FilterWorkBuffers {
                flags: map(work.flags)?,
                offsets: map(work.offsets)?,
                block_sums: map(work.block_sums)?,
                block_offsets: map(work.block_offsets)?,
            })
        }

        fn map_filter_state<R, S, E>(
            state: FilterState<R>,
            map: &mut impl FnMut(R) -> Result<S, E>,
        ) -> Result<FilterState<S>, E> {
            Ok(match state {
                FilterState::Raw => FilterState::Raw,
                FilterState::Semantic { space } => FilterState::Semantic {
                    space: map_space(space, map)?,
                },
                FilterState::Scheduled { space, plan } => FilterState::Scheduled {
                    space: map_space(space, map)?,
                    plan: match plan {
                        FilterPlan::Serial => FilterPlan::Serial,
                        FilterPlan::Flags(work) => FilterPlan::Flags(map_work(work, map)?),
                        FilterPlan::Scan(work) => FilterPlan::Scan(map_work(work, map)?),
                        FilterPlan::Scatter(work) => FilterPlan::Scatter(map_work(work, map)?),
                    },
                },
            })
        }

        fn map_filter_output<R, S, E>(
            output: FilterOutput<R>,
            map: &mut impl FnMut(R) -> Result<S, E>,
        ) -> Result<FilterOutput<S>, E> {
            Ok(match output {
                FilterOutput::Local {
                    capacity,
                    destination,
                } => FilterOutput::Local {
                    capacity,
                    destination,
                },
                FilterOutput::Runtime { scratch, length } => FilterOutput::Runtime {
                    scratch: map(scratch)?,
                    length: match length {
                        RuntimeFilterLength::ViewOnly => RuntimeFilterLength::ViewOnly,
                        RuntimeFilterLength::EntryOutput(resource) => {
                            RuntimeFilterLength::EntryOutput(map(resource)?)
                        }
                    },
                },
            })
        }

        Ok(match self {
            EgirSoac::Screma {
                map_bodies,
                accumulators,
                input_array_types,
                input_elem_types,
                map_output_elem_types,
                map_input_indices,
                map_destinations,
                acc_destinations,
            } => EgirSoac::Screma {
                map_bodies,
                accumulators,
                input_array_types,
                input_elem_types,
                map_output_elem_types,
                map_input_indices,
                map_destinations,
                acc_destinations,
            },
            EgirSoac::Filter {
                state,
                map_body,
                output_elem_type,
                pred_body,
                input_array_type,
                input_elem_type,
                output,
            } => EgirSoac::Filter {
                state: map_filter_state(state, &mut map)?,
                map_body,
                output_elem_type,
                pred_body,
                input_array_type,
                input_elem_type,
                output: map_filter_output(output, &mut map)?,
            },
            EgirSoac::Hist {
                body,
                input_array_types,
                input_elem_types,
                index_type,
                value_type,
                dest_elem_type,
                update_policy,
                execution,
            } => EgirSoac::Hist {
                body,
                input_array_types,
                input_elem_types,
                index_type,
                value_type,
                dest_elem_type,
                update_policy,
                execution: match execution {
                    HistExecution::Raw => HistExecution::Raw,
                    HistExecution::Serial => HistExecution::Serial,
                    HistExecution::Segmented(space) => {
                        HistExecution::Segmented(map_space(space, &mut map)?)
                    }
                },
            },
            EgirSoac::Seg {
                space,
                placement,
                kind,
                map_bodies,
                input_array_types,
                input_elem_types,
                map_output_elem_types,
                map_input_indices,
                map_destinations,
                acc_destinations,
                result_types,
                output_slots,
                resources,
            } => EgirSoac::Seg {
                space: map_space(space, &mut map)?,
                placement,
                kind,
                map_bodies,
                input_array_types,
                input_elem_types,
                map_output_elem_types,
                map_input_indices,
                map_destinations,
                acc_destinations,
                result_types,
                output_slots,
                resources: resources
                    .into_iter()
                    .map(|access| {
                        Ok(SegResourceAccess {
                            resource: map(access.resource)?,
                            access: access.access,
                        })
                    })
                    .collect::<Result<Vec<_>, E>>()?,
            },
        })
    }

    pub fn visit_types_mut(&mut self, mut visit: impl FnMut(&mut Type<TypeName>)) {
        let mut visit_many = |types: &mut Vec<Type<TypeName>>| {
            for ty in types {
                visit(ty);
            }
        };
        match self {
            EgirSoac::Screma {
                input_array_types,
                input_elem_types,
                map_output_elem_types,
                ..
            } => {
                visit_many(input_array_types);
                visit_many(input_elem_types);
                visit_many(map_output_elem_types);
            }
            EgirSoac::Filter {
                output_elem_type,
                input_array_type,
                input_elem_type,
                output,
                ..
            } => {
                visit(output_elem_type);
                visit(input_array_type);
                visit(input_elem_type);
                if let FilterOutput::Local { capacity, .. } = output {
                    visit(capacity);
                }
            }
            EgirSoac::Hist {
                input_array_types,
                input_elem_types,
                index_type,
                value_type,
                dest_elem_type,
                ..
            } => {
                visit_many(input_array_types);
                visit_many(input_elem_types);
                visit(index_type);
                visit(value_type);
                visit(dest_elem_type);
            }
            EgirSoac::Seg {
                input_array_types,
                input_elem_types,
                map_output_elem_types,
                result_types,
                ..
            } => {
                visit_many(input_array_types);
                visit_many(input_elem_types);
                visit_many(map_output_elem_types);
                visit_many(result_types);
            }
        }
    }

    /// Every `SegBody` this SOAC carries, in a stable order: map lanes first,
    /// then each accumulator's step and combine. Captures and the callee region
    /// live here, never inline in `SideEffect::operand_nodes`.
    pub(crate) fn seg_bodies(&self) -> Vec<&SegBody> {
        match self {
            EgirSoac::Screma {
                map_bodies,
                accumulators,
                ..
            } => map_bodies
                .iter()
                .chain(accumulators.iter().flat_map(|op| [&op.step, &op.combine]))
                .collect(),
            EgirSoac::Seg { map_bodies, kind, .. } => map_bodies
                .iter()
                .chain(kind.operators().iter().flat_map(|op| [&op.step, &op.combine]))
                .collect(),
            EgirSoac::Filter {
                map_body, pred_body, ..
            } => map_body.iter().chain(std::iter::once(pred_body)).collect(),
            EgirSoac::Hist { body, .. } => vec![body],
        }
    }

    /// All capture nodes across every body. Captures are graph references that
    /// generic reachability must treat exactly like operands.
    pub fn capture_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.seg_bodies().into_iter().flat_map(|body| body.captures.iter().copied())
    }

    pub fn visit_capture_nodes_mut(&mut self, mut visit: impl FnMut(&mut NodeId)) {
        let mut visit_body = |body: &mut SegBody| {
            for capture in &mut body.captures {
                visit(capture);
            }
        };
        match self {
            EgirSoac::Screma {
                map_bodies,
                accumulators,
                ..
            } => {
                for body in map_bodies {
                    visit_body(body);
                }
                for operator in accumulators {
                    visit_body(&mut operator.step);
                    visit_body(&mut operator.combine);
                }
            }
            EgirSoac::Seg { map_bodies, kind, .. } => {
                for body in map_bodies {
                    visit_body(body);
                }
                for operator in match kind {
                    SegOpKind::SegMap => &mut [][..],
                    SegOpKind::SegRed { operators }
                    | SegOpKind::SegScan { operators }
                    | SegOpKind::SegComposite { operators } => operators.as_mut_slice(),
                } {
                    visit_body(&mut operator.step);
                    visit_body(&mut operator.combine);
                }
            }
            EgirSoac::Filter {
                map_body, pred_body, ..
            } => {
                if let Some(body) = map_body {
                    visit_body(body);
                }
                visit_body(pred_body);
            }
            EgirSoac::Hist { body, .. } => visit_body(body),
        }
    }

    /// Every graph value stored outside `SideEffect::operand_nodes`.
    /// Segmented domains and operator neutrals/shapes are first-class value
    /// edges just like callable-body captures; graph projection and liveness
    /// must never rediscover them variant by variant.
    pub fn metadata_nodes(&self) -> Vec<NodeId> {
        let mut nodes = self.capture_nodes().collect::<Vec<_>>();
        let mut add_space = |space: &SegSpace<R>| {
            nodes.extend(space.dims.iter().filter_map(|extent| match extent {
                SegExtent::PushConstant { node, .. }
                | SegExtent::ResourceLength { node, .. }
                | SegExtent::Value(node) => Some(*node),
                SegExtent::Fixed(_) => None,
            }));
        };
        match self {
            EgirSoac::Seg { space, kind, .. } => {
                add_space(space);
                for operator in kind.operators() {
                    nodes.push(operator.neutral);
                    nodes.extend(operator.shape.iter().copied());
                }
            }
            EgirSoac::Filter { state, .. } => match state {
                FilterState::Semantic { space } | FilterState::Scheduled { space, .. } => add_space(space),
                FilterState::Raw => {}
            },
            EgirSoac::Hist {
                execution: HistExecution::Segmented(space),
                ..
            } => add_space(space),
            EgirSoac::Screma { .. } | EgirSoac::Hist { .. } => {}
        }
        nodes
    }

    pub fn visit_metadata_nodes_mut(&mut self, mut visit: impl FnMut(&mut NodeId)) {
        self.visit_capture_nodes_mut(&mut visit);
        let mut visit_space = |space: &mut SegSpace<R>| {
            for extent in &mut space.dims {
                match extent {
                    SegExtent::PushConstant { node, .. }
                    | SegExtent::ResourceLength { node, .. }
                    | SegExtent::Value(node) => visit(node),
                    SegExtent::Fixed(_) => {}
                }
            }
        };
        match self {
            EgirSoac::Seg { space, kind, .. } => {
                visit_space(space);
                for operator in match kind {
                    SegOpKind::SegMap => &mut [][..],
                    SegOpKind::SegRed { operators }
                    | SegOpKind::SegScan { operators }
                    | SegOpKind::SegComposite { operators } => operators.as_mut_slice(),
                } {
                    visit(&mut operator.neutral);
                    for node in &mut operator.shape {
                        visit(node);
                    }
                }
            }
            EgirSoac::Filter { state, .. } => match state {
                FilterState::Semantic { space } | FilterState::Scheduled { space, .. } => {
                    visit_space(space)
                }
                FilterState::Raw => {}
            },
            EgirSoac::Hist {
                execution: HistExecution::Segmented(space),
                ..
            } => visit_space(space),
            EgirSoac::Screma { .. } | EgirSoac::Hist { .. } => {}
        }
    }
}

impl<R> SideEffect<R> {
    pub fn required_semantic_id(&self) -> super::program::SemanticOpId {
        self.semantic_id.expect("semantic operation id assigned after segmentation")
    }

    /// Every graph node this side-effect references: its operands plus, for a
    /// SOAC, its bodies' captures. This is the authoritative use-set for
    /// liveness and reachability now that captures are not inlined into
    /// `operand_nodes`.
    pub fn referenced_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        let metadata = match &self.kind {
            SideEffectKind::Soac(soac) => soac.metadata_nodes(),
            SideEffectKind::Inst(_) => Vec::new(),
        };
        self.operand_nodes.iter().copied().chain(metadata)
    }

    /// Mutate every value edge, including explicit captures. Rewriters must
    /// use this rather than editing `operand_nodes` alone when substituting a
    /// graph value globally.
    pub fn visit_referenced_nodes_mut(&mut self, mut visit: impl FnMut(&mut NodeId)) {
        for operand in &mut self.operand_nodes {
            visit(operand);
        }
        if let SideEffectKind::Soac(soac) = &mut self.kind {
            soac.visit_metadata_nodes_mut(visit);
        }
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
pub struct SkeletonBlock<R = super::program::SemanticResourceRef> {
    /// Block parameters as NodeIds.
    pub params: Vec<NodeId>,
    /// Effectful instructions, in order.
    pub side_effects: Vec<SideEffect<R>>,
    /// Block terminator.
    pub term: SkeletonTerminator,
}

impl<R> SkeletonBlock<R> {
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
pub struct Skeleton<R = super::program::SemanticResourceRef> {
    pub entry: BlockId,
    pub blocks: SlotMap<BlockId, SkeletonBlock<R>>,
}

impl<R> Skeleton<R> {
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
    pub fn build<R>(graph: &EGraph<R>) -> Self {
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

    pub fn effect<'a, R>(&self, graph: &'a EGraph<R>, result: NodeId) -> Option<&'a SideEffect<R>> {
        let site = self.site(result)?;
        let effect = graph.skeleton.blocks.get(site.block)?.side_effects.get(site.index)?;
        (effect.result == Some(result)).then_some(effect)
    }

    pub fn effect_mut<'a, R>(
        &self,
        graph: &'a mut EGraph<R>,
        result: NodeId,
    ) -> Option<&'a mut SideEffect<R>> {
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
pub struct EGraph<R = super::program::SemanticResourceRef> {
    /// All nodes (pure, union, params, constants, side-effect results).
    pub nodes: SlotMap<NodeId, ENode<R>>,
    /// Type of each node's result.
    pub types: LookupMap<NodeId, Type<TypeName>>,
    /// Hash-cons table: NodeKey → existing NodeId.
    hash_cons: LookupMap<NodeKey<R>, NodeId>,
    /// Constant dedup cache.
    pub const_cache: LookupMap<ConstantValue, NodeId>,
    /// The CFG skeleton.
    pub skeleton: Skeleton<R>,
    /// Source span associated with each pure node (first-writer-wins —
    /// later interns of the same hash-consed node keep the original span).
    pub node_spans: LookupMap<NodeId, Span>,
}

pub trait GraphResource: Clone + std::fmt::Debug + Eq + std::hash::Hash {}

impl<T> GraphResource for T where T: Clone + std::fmt::Debug + Eq + std::hash::Hash {}

impl<R: GraphResource> EGraph<R> {
    fn new_typed() -> Self {
        EGraph {
            nodes: SlotMap::with_key(),
            types: LookupMap::new(),
            hash_cons: LookupMap::new(),
            const_cache: LookupMap::new(),
            skeleton: Skeleton::new(),
            node_spans: LookupMap::new(),
        }
    }

    /// Consume a graph at one compiler phase and rebuild every resource-bearing
    /// node and effect for the next phase. Both arena identity maps are
    /// returned so graph-adjacent metadata can be projected atomically.
    pub(crate) fn try_map_resources<S: GraphResource, E>(
        self,
        mut map_resource: impl FnMut(R) -> Result<S, E>,
    ) -> Result<(EGraph<S>, LookupMap<NodeId, NodeId>, LookupMap<BlockId, BlockId>), E> {
        let EGraph {
            nodes,
            types,
            hash_cons: _,
            const_cache,
            skeleton,
            node_spans,
        } = self;

        let old_blocks = skeleton.blocks.into_iter().collect::<Vec<_>>();
        let mut blocks = SlotMap::with_key();
        let mut block_map = LookupMap::new();
        for (old_id, _) in &old_blocks {
            block_map.insert(*old_id, blocks.insert(SkeletonBlock::new()));
        }

        let old_nodes = nodes.into_iter().collect::<Vec<_>>();
        let mut new_nodes = SlotMap::with_key();
        let mut node_map = LookupMap::new();
        for (old_id, _) in &old_nodes {
            node_map.insert(
                *old_id,
                new_nodes.insert(ENode::Constant(ConstantValue::Bool(false))),
            );
        }
        for (old_id, node) in old_nodes {
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
            new_nodes[node_map[&old_id]] = node;
        }

        for (old_id, block) in old_blocks {
            let side_effects = block
                .side_effects
                .into_iter()
                .map(|effect| {
                    let kind = match effect.kind {
                        SideEffectKind::Inst(inst) => {
                            SideEffectKind::Inst(inst.try_map_resource(&mut map_resource)?)
                        }
                        SideEffectKind::Soac(soac) => {
                            let mut soac = soac.try_map_resources(&mut map_resource)?;
                            soac.visit_metadata_nodes_mut(|node| *node = node_map[node]);
                            SideEffectKind::Soac(soac)
                        }
                    };
                    Ok(SideEffect {
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
            blocks[block_map[&old_id]] = SkeletonBlock {
                params: block.params.into_iter().map(|node| node_map[&node]).collect(),
                side_effects,
                term,
            };
        }

        let mut graph = EGraph {
            nodes: new_nodes,
            types: types.into_iter().map(|(node, ty)| (node_map[&node], ty)).collect(),
            hash_cons: LookupMap::new(),
            const_cache: const_cache.into_iter().map(|(value, node)| (value, node_map[&node])).collect(),
            skeleton: Skeleton {
                entry: block_map[&skeleton.entry],
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

    fn pure_node_key(&self, id: NodeId) -> Option<NodeKey<R>> {
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
    pub fn replace_node_preserving_type(&mut self, id: NodeId, node: ENode<R>) {
        self.unindex_current_pure(id);
        self.nodes[id] = node;
        self.index_current_pure(id);
    }

    /// Replace a pure node's operator and operands without changing its result
    /// type, keeping the hash-cons table consistent across the mutation.
    pub fn replace_pure_node(&mut self, id: NodeId, op: PureOp<R>, operands: SmallVec<[NodeId; 4]>) {
        self.replace_node_preserving_type(id, ENode::Pure { op, operands });
    }

    /// Mutate a pure node's operator and operands in place while maintaining
    /// the hash-cons table. Returns false if `id` is not a pure node.
    pub fn update_pure_node<F>(&mut self, id: NodeId, update: F) -> bool
    where
        F: FnOnce(&mut PureOp<R>, &mut SmallVec<[NodeId; 4]>),
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
        op: PureOp<R>,
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
        op: PureOp<R>,
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

impl EGraph<super::program::SemanticResourceRef> {
    pub fn new() -> Self {
        Self::new_typed()
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
