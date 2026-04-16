//! Core data structures for the acyclic e-graph (aegraph).

use crate::ast::{Span, TypeName};
use crate::ssa::framework::BlockId;
use crate::ssa::types::{ConstantValue, InstKind, ViewSource};

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
// PureOp — the operator identity for hash-consing (no operands, Hash+Eq)
// ---------------------------------------------------------------------------

/// The operator part of a pure instruction, without its operands.
/// Operands are stored separately in `ENode::Pure { operands }`.
///
/// This mirrors the pure subset of `InstKind` but:
/// - Strips out `ValueRef` operands (they become `NodeId` in `ENode`)
/// - Derives `Hash + Eq` for hash-consing
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PureOp {
    /// Signed integer literal (i8, i16, i32, i64).
    Int(String),
    /// Unsigned integer literal (u8, u16, u32, u64).
    Uint(String),
    Float(String),
    Bool(bool),
    Unit,
    StringLit(String),
    Global(String),
    Extern(String),
    BinOp(String),
    UnaryOp(String),
    Tuple(usize),
    Vector(usize),
    Matrix {
        rows: usize,
        cols: usize,
    },
    ArrayLit(usize),
    ArrayRange {
        has_step: bool,
    },
    Project {
        index: u32,
    },
    Index,
    Materialize,
    DynamicExtract,
    Call(String),
    Intrinsic(String),
    /// Storage buffer view creation. The `Inherited` parent (if any) is
    /// carried in the operands tail, not in this tag, so equivalent views
    /// with the same backing source hash-cons together.
    StorageView(PureViewSource),
    StorageViewIndex,
    StorageViewLen,
    OutputPtr {
        index: usize,
    },
}

/// Hashable variant of `ViewSource` for use inside a `PureOp`.
/// Drops the `ValueId` from `Inherited` — that parent is stored as an
/// operand in the `ENode::Pure`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PureViewSource {
    Storage {
        set: u32,
        binding: u32,
    },
    Inherited,
}

// ---------------------------------------------------------------------------
// NodeKey — hash-cons key = operator + operands
// ---------------------------------------------------------------------------

/// The full identity of a pure node for hash-consing: the operator plus
/// its operands (which are already-canonical `NodeId`s).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeKey {
    pub op: PureOp,
    pub operands: SmallVec<[NodeId; 4]>,
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

/// An unexpanded SOAC operation held in the skeleton until `soac_expand` rewrites
/// it into an explicit loop. All operand NodeIds live in `SideEffect.operand_nodes`
/// with a variant-specific layout (documented per-variant in `soac_expand`).
#[derive(Clone, Debug)]
pub enum PendingSoac {
    /// `map f inputs` → composite output array.
    /// Operands: `[input_0, ..., input_{n-1}, ...captures]`.
    Map {
        func: String,
        input_array_types: Vec<Type<TypeName>>,
        input_elem_types: Vec<Type<TypeName>>,
        output_elem_type: Type<TypeName>,
    },
    /// `reduce f init input` → scalar accumulator.
    /// Operands: `[input, init, ...captures]`.
    Reduce {
        func: String,
        input_array_type: Type<TypeName>,
        input_elem_type: Type<TypeName>,
    },
    /// `scan f init input` → composite output array.
    /// Operands: `[input, init, ...captures]`.
    Scan {
        func: String,
        input_array_type: Type<TypeName>,
        input_elem_type: Type<TypeName>,
    },
    /// `map_into f inputs view` → writes to storage view (unit-valued).
    /// Operands: `[input_0, ..., input_{n-1}, ...captures, output_view]`.
    MapInto {
        func: String,
        input_array_types: Vec<Type<TypeName>>,
        input_elem_types: Vec<Type<TypeName>>,
        output_elem_type: Type<TypeName>,
    },
    /// `scan_into f init input view` → writes to storage view (unit-valued).
    /// Operands: `[input, init, ...captures, output_view]`.
    ScanInto {
        func: String,
        input_array_type: Type<TypeName>,
        input_elem_type: Type<TypeName>,
    },
    /// Fused map+reduce: per iteration `acc = func(acc, x1, ..., xn, ...caps)`.
    /// Operands: `[input_0, ..., input_{n-1}, init, ...captures, ...reduce_captures]`.
    Redomap {
        func: String,
        reduce_func: String,
        input_array_types: Vec<Type<TypeName>>,
        input_elem_types: Vec<Type<TypeName>>,
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

/// Extract a `PureOp` from a pure `InstKind`, returning `None` if the
/// instruction is not pure/hoistable.
pub fn extract_pure_op(kind: &InstKind, ty: &Type<TypeName>) -> Option<PureOp> {
    match kind {
        InstKind::Int(s) => {
            let is_unsigned = matches!(ty, Type::Constructed(TypeName::UInt(_), _));
            if is_unsigned { Some(PureOp::Uint(s.clone())) } else { Some(PureOp::Int(s.clone())) }
        }
        InstKind::Float(s) => Some(PureOp::Float(s.clone())),
        InstKind::Bool(b) => Some(PureOp::Bool(*b)),
        InstKind::Unit => Some(PureOp::Unit),
        InstKind::String(s) => Some(PureOp::StringLit(s.clone())),
        InstKind::Global(s) => Some(PureOp::Global(s.clone())),
        InstKind::Extern(s) => Some(PureOp::Extern(s.clone())),
        InstKind::BinOp { op, .. } => Some(PureOp::BinOp(op.clone())),
        InstKind::UnaryOp { op, .. } => Some(PureOp::UnaryOp(op.clone())),
        InstKind::Tuple(elems) => Some(PureOp::Tuple(elems.len())),
        InstKind::Vector(elems) => Some(PureOp::Vector(elems.len())),
        InstKind::Matrix(rows) => {
            let cols = rows.first().map_or(0, |r| r.len());
            Some(PureOp::Matrix {
                rows: rows.len(),
                cols,
            })
        }
        InstKind::ArrayLit { elements } => Some(PureOp::ArrayLit(elements.len())),
        InstKind::ArrayRange { step, .. } => Some(PureOp::ArrayRange {
            has_step: step.is_some(),
        }),
        InstKind::Project { index, .. } => Some(PureOp::Project { index: *index }),
        InstKind::Index { .. } => Some(PureOp::Index),
        InstKind::Materialize { .. } => Some(PureOp::Materialize),
        InstKind::DynamicExtract { .. } => Some(PureOp::DynamicExtract),
        InstKind::Call { func, .. } => Some(PureOp::Call(func.clone())),
        InstKind::Intrinsic { name, .. } => Some(PureOp::Intrinsic(name.clone())),
        InstKind::StorageView { source, .. } => {
            let src = match source {
                ViewSource::Storage { set, binding } => PureViewSource::Storage {
                    set: *set,
                    binding: *binding,
                },
                ViewSource::Inherited { .. } => PureViewSource::Inherited,
            };
            Some(PureOp::StorageView(src))
        }
        InstKind::StorageViewIndex { .. } => Some(PureOp::StorageViewIndex),
        InstKind::StorageViewLen { .. } => Some(PureOp::StorageViewLen),
        InstKind::OutputPtr { index } => Some(PureOp::OutputPtr { index: *index }),
        // Blacklist: the only truly effectful InstKinds. Everything else above
        // is treated as pure (hash-consable, hoistable).
        InstKind::Alloca { .. } | InstKind::Load { .. } | InstKind::Store { .. } => None,
    }
}

/// Rebuild an `InstKind` from a `PureOp` + operands as `ValueRef::Ssa(ValueId)`.
///
/// The `operands` must be in the same order as `InstKind::value_uses()` returns
/// for the original instruction.
pub fn rebuild_inst_kind(op: &PureOp, operands: &[crate::ssa::framework::ValueId]) -> InstKind {
    use crate::ssa::types::ValueRef;
    let vr = |i: usize| -> ValueRef { ValueRef::Ssa(operands[i]) };
    match op {
        PureOp::Int(s) | PureOp::Uint(s) => InstKind::Int(s.clone()),
        PureOp::Float(s) => InstKind::Float(s.clone()),
        PureOp::Bool(b) => InstKind::Bool(*b),
        PureOp::Unit => InstKind::Unit,
        PureOp::StringLit(s) => InstKind::String(s.clone()),
        PureOp::Global(s) => InstKind::Global(s.clone()),
        PureOp::Extern(s) => InstKind::Extern(s.clone()),
        PureOp::BinOp(op_name) => InstKind::BinOp {
            op: op_name.clone(),
            lhs: vr(0),
            rhs: vr(1),
        },
        PureOp::UnaryOp(op_name) => InstKind::UnaryOp {
            op: op_name.clone(),
            operand: vr(0),
        },
        PureOp::Tuple(n) => InstKind::Tuple((0..*n).map(|i| vr(i)).collect()),
        PureOp::Vector(n) => InstKind::Vector((0..*n).map(|i| vr(i)).collect()),
        PureOp::Matrix { rows, cols } => {
            let mut mat = Vec::with_capacity(*rows);
            let mut idx = 0;
            for _ in 0..*rows {
                let row: Vec<ValueRef> = (0..*cols)
                    .map(|_| {
                        let v = vr(idx);
                        idx += 1;
                        v
                    })
                    .collect();
                mat.push(row);
            }
            InstKind::Matrix(mat)
        }
        PureOp::ArrayLit(n) => InstKind::ArrayLit {
            elements: (0..*n).map(|i| vr(i)).collect(),
        },
        PureOp::ArrayRange { has_step } => InstKind::ArrayRange {
            start: vr(0),
            len: vr(1),
            step: if *has_step { Some(vr(2)) } else { None },
        },
        PureOp::Project { index } => InstKind::Project {
            base: vr(0),
            index: *index,
        },
        PureOp::Index => InstKind::Index {
            base: vr(0),
            index: vr(1),
        },
        PureOp::Materialize => InstKind::Materialize { value: vr(0) },
        PureOp::DynamicExtract => InstKind::DynamicExtract {
            base: vr(0),
            index: vr(1),
        },
        PureOp::Call(func) => InstKind::Call {
            func: func.clone(),
            args: (0..operands.len()).map(|i| vr(i)).collect(),
        },
        PureOp::Intrinsic(name) => InstKind::Intrinsic {
            name: name.clone(),
            args: (0..operands.len()).map(|i| vr(i)).collect(),
        },
        PureOp::StorageView(src) => {
            let source = match src {
                PureViewSource::Storage { set, binding } => ViewSource::Storage {
                    set: *set,
                    binding: *binding,
                },
                PureViewSource::Inherited => ViewSource::Inherited { parent: operands[2] },
            };
            InstKind::StorageView {
                source,
                offset: vr(0),
                len: vr(1),
            }
        }
        PureOp::StorageViewIndex => InstKind::StorageViewIndex {
            view: vr(0),
            index: vr(1),
        },
        PureOp::StorageViewLen => InstKind::StorageViewLen { view: vr(0) },
        PureOp::OutputPtr { index } => InstKind::OutputPtr { index: *index },
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
