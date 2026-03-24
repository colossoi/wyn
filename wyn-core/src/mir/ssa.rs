//! SSA (Static Single Assignment) form for MIR.
//!
//! This module provides a proper SSA representation with:
//! - **SSA values** (ValueId) produced exactly once
//! - **Explicit CFG** with block parameters (MLIR/Cranelift style, not phi nodes)
//! - **Effect tokens** for ordering side effects
//!
//! ## Block Parameters vs Phi Nodes
//!
//! Instead of phi nodes at block entry, we use block parameters:
//! ```text
//! // Phi-style (not used):
//! merge:
//!     %result = phi [%then_val, then_block], [%else_val, else_block]
//!
//! // Block-parameter style (used here):
//! merge(%result: T):
//!     // %result is defined here, predecessors pass the value
//!
//! then_block:
//!     br merge(%then_val)
//!
//! else_block:
//!     br merge(%else_val)
//! ```
//!
//! This makes the data flow explicit at branch sites rather than requiring
//! inspection of phi nodes to understand where values come from.

use crate::ast::{NodeId, Span, TypeName};
use polytype::Type;

// =============================================================================
// ID Types
// =============================================================================

/// SSA value - defined exactly once.
///
/// Values are produced by instructions or block parameters.
/// Each value has a single definition point and can be used multiple times.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

impl ValueId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for ValueId {
    fn from(id: u32) -> Self {
        ValueId(id)
    }
}

/// Instruction within a function.
///
/// Instructions are stored in a flat arena and referenced by InstId.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstId(pub u32);

impl InstId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for InstId {
    fn from(id: u32) -> Self {
        InstId(id)
    }
}

/// Basic block within a function.
///
/// BlockId(0) is always the entry block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl BlockId {
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// The entry block is always BlockId(0).
    pub const ENTRY: BlockId = BlockId(0);
}

impl From<u32> for BlockId {
    fn from(id: u32) -> Self {
        BlockId(id)
    }
}

/// Effect token for side effect ordering.
///
/// Effect tokens form a chain that ensures effectful operations
/// (loads, stores, etc.) execute in the correct order.
/// Pure operations don't need effect tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectToken(pub u32);

impl EffectToken {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for EffectToken {
    fn from(id: u32) -> Self {
        EffectToken(id)
    }
}

// =============================================================================
// Basic Blocks
// =============================================================================

/// Structured control flow header information for SPIR-V lowering.
///
/// SPIR-V requires explicit merge/continue annotations for loops and selections.
/// This metadata is attached to header blocks by the builder and consumed during lowering.
#[derive(Debug, Clone)]
pub enum ControlHeader {
    /// This block is a loop header.
    /// Requires `OpLoopMerge merge continue None` before the conditional branch.
    Loop {
        /// The block where the loop exits to (post-loop code).
        merge: BlockId,
        /// The block that branches back to the header (typically end of loop body).
        continue_block: BlockId,
    },
    /// This block is a selection header (if-then-else).
    /// Requires `OpSelectionMerge merge None` before the conditional branch.
    Selection {
        /// The block where both branches reconverge.
        merge: BlockId,
    },
}

/// A basic block in the CFG.
///
/// Blocks have parameters (replacing phi nodes), a sequence of instructions,
/// and a terminator that transfers control to other blocks.
#[derive(Debug, Clone)]
pub struct Block {
    /// Block parameters (replaces phi nodes).
    /// Each predecessor passes arguments at its branch site.
    pub params: Vec<BlockParam>,

    /// Instructions in execution order.
    pub insts: Vec<InstId>,

    /// How control leaves this block.
    pub terminator: Option<Terminator>,

    /// Structured control flow header info (for SPIR-V lowering).
    /// Set by the builder when creating loops/selections.
    pub control: Option<ControlHeader>,
}

impl Block {
    /// Create a new empty block with no parameters.
    pub fn new() -> Self {
        Block {
            params: Vec::new(),
            insts: Vec::new(),
            terminator: None,
            control: None,
        }
    }

    /// Create a block with the given parameters.
    pub fn with_params(params: Vec<BlockParam>) -> Self {
        Block {
            params,
            insts: Vec::new(),
            terminator: None,
            control: None,
        }
    }

    /// A dead block was eliminated by the optimizer but remains in the Vec
    /// to preserve BlockId indices.
    pub fn is_dead(&self) -> bool {
        self.insts.is_empty() && matches!(self.terminator, Some(Terminator::Unreachable))
    }
}

impl Default for Block {
    fn default() -> Self {
        Self::new()
    }
}

/// A block parameter (replaces phi nodes).
#[derive(Debug, Clone)]
pub struct BlockParam {
    /// The value this parameter defines.
    pub value: ValueId,
    /// Type of this parameter.
    pub ty: Type<TypeName>,
    /// Optional name for debugging.
    pub name: Option<String>,
}

/// How control leaves a basic block.
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Unconditional branch: `br target(args...)`
    Branch {
        target: BlockId,
        args: Vec<ValueId>,
    },

    /// Conditional branch: `br_if cond, then(then_args...), else(else_args...)`
    CondBranch {
        cond: ValueId,
        then_target: BlockId,
        then_args: Vec<ValueId>,
        else_target: BlockId,
        else_args: Vec<ValueId>,
    },

    /// Return a value from the function.
    Return(ValueId),

    /// Return unit (void return).
    ReturnUnit,

    /// Unreachable code (e.g., after a guaranteed panic).
    Unreachable,
}

impl Terminator {
    /// Return all ValueIds referenced by this terminator.
    pub fn value_uses(&self) -> Vec<ValueId> {
        match self {
            Terminator::Branch { args, .. } => args.clone(),
            Terminator::CondBranch {
                cond,
                then_args,
                else_args,
                ..
            } => {
                let mut u = vec![*cond];
                u.extend_from_slice(then_args);
                u.extend_from_slice(else_args);
                u
            }
            Terminator::Return(v) => vec![*v],
            Terminator::ReturnUnit | Terminator::Unreachable => vec![],
        }
    }

    /// Apply a substitution function to all ValueId references in place.
    pub fn substitute_values(&mut self, sub: &mut impl FnMut(&mut ValueId)) {
        match self {
            Terminator::Branch { args, .. } => {
                for a in args {
                    sub(a);
                }
            }
            Terminator::CondBranch {
                cond,
                then_args,
                else_args,
                ..
            } => {
                sub(cond);
                for a in then_args {
                    sub(a);
                }
                for a in else_args {
                    sub(a);
                }
            }
            Terminator::Return(v) => sub(v),
            Terminator::ReturnUnit | Terminator::Unreachable => {}
        }
    }

    /// Create a new Terminator with all ValueIds and BlockIds remapped.
    pub fn remap(
        &self,
        rv: &impl Fn(&ValueId) -> ValueId,
        rb: &impl Fn(&BlockId) -> BlockId,
    ) -> Terminator {
        match self {
            Terminator::Branch { target, args } => Terminator::Branch {
                target: rb(target),
                args: args.iter().map(rv).collect(),
            },
            Terminator::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
            } => Terminator::CondBranch {
                cond: rv(cond),
                then_target: rb(then_target),
                then_args: then_args.iter().map(rv).collect(),
                else_target: rb(else_target),
                else_args: else_args.iter().map(rv).collect(),
            },
            Terminator::Return(v) => Terminator::Return(rv(v)),
            Terminator::ReturnUnit => Terminator::ReturnUnit,
            Terminator::Unreachable => Terminator::Unreachable,
        }
    }
}

impl ControlHeader {
    /// Create a new ControlHeader with all BlockIds remapped.
    pub fn remap(&self, rb: &impl Fn(&BlockId) -> BlockId) -> ControlHeader {
        match self {
            ControlHeader::Loop {
                merge,
                continue_block,
            } => ControlHeader::Loop {
                merge: rb(merge),
                continue_block: rb(continue_block),
            },
            ControlHeader::Selection { merge } => ControlHeader::Selection { merge: rb(merge) },
        }
    }
}

// =============================================================================
// Instructions
// =============================================================================

/// An SSA instruction.
///
/// Each instruction optionally produces a result value and has a kind
/// that describes the operation.
#[derive(Debug, Clone)]
pub struct Inst {
    /// The value this instruction produces, if any.
    pub result: Option<ValueId>,
    /// Type of the result (or Unit if no result).
    pub result_ty: Type<TypeName>,
    /// The operation this instruction performs.
    pub kind: InstKind,
    /// Source location for error messages.
    pub span: Span,
    /// Original AST node ID for diagnostics.
    pub node_id: NodeId,
}

/// The kind of operation an instruction performs.
#[derive(Debug, Clone)]
pub enum InstKind {
    // =========================================================================
    // Pure Operations (no effect tokens needed)
    // =========================================================================
    /// Integer literal.
    Int(String),
    /// Float literal.
    Float(String),
    /// Boolean literal.
    Bool(bool),
    /// Unit value.
    Unit,
    /// String literal.
    String(String),

    /// Binary operation.
    BinOp {
        op: String,
        lhs: ValueId,
        rhs: ValueId,
    },

    /// Unary operation.
    UnaryOp {
        op: String,
        operand: ValueId,
    },

    /// Tuple construction.
    Tuple(Vec<ValueId>),

    /// Array construction with literal elements.
    ArrayLit {
        elements: Vec<ValueId>,
    },

    /// Array from range (virtual, computed on demand).
    ArrayRange {
        start: ValueId,
        /// Length of the range.
        len: ValueId,
        /// Step (None means 1).
        step: Option<ValueId>,
    },

    /// Vector construction (@[x, y, z]).
    Vector(Vec<ValueId>),

    /// Matrix construction (@[[a, b], [c, d]]).
    Matrix(Vec<Vec<ValueId>>),

    /// Tuple/struct field projection.
    Project {
        base: ValueId,
        index: u32,
    },

    /// Array/vector indexing (for fixed-size arrays).
    Index {
        base: ValueId,
        index: ValueId,
    },

    /// Function call.
    Call {
        func: String,
        args: Vec<ValueId>,
    },

    /// Reference to a global constant or function.
    Global(String),

    /// External function reference (linked SPIR-V).
    Extern(String),

    /// Compiler intrinsic call.
    Intrinsic {
        name: String,
        args: Vec<ValueId>,
    },

    // =========================================================================
    // Effectful Operations (consume/produce effect tokens)
    // =========================================================================
    /// Allocate local storage (returns a pointer).
    Alloca {
        /// Type of the value to store.
        elem_ty: Type<TypeName>,
        /// Input effect token.
        effect_in: EffectToken,
        /// Output effect token.
        effect_out: EffectToken,
    },

    /// Load a value from a pointer.
    Load {
        ptr: ValueId,
        effect_in: EffectToken,
        effect_out: EffectToken,
    },

    /// Store a value to a pointer.
    Store {
        ptr: ValueId,
        value: ValueId,
        effect_in: EffectToken,
        effect_out: EffectToken,
    },

    /// Create a storage buffer view.
    StorageView {
        source: ViewSource,
        offset: ValueId,
        len: ValueId,
    },

    /// Index into a storage view. SSA result type is the element type;
    /// SPIR-V lowering wraps it in a StorageBuffer pointer internally.
    StorageViewIndex {
        view: ValueId,
        index: ValueId,
    },

    /// Get the length of a storage view.
    StorageViewLen {
        view: ValueId,
    },

    // =========================================================================
    // Entry Point I/O
    // =========================================================================
    /// Get a pointer to an entry point output variable.
    /// Used in entry points to explicitly store results before returning.
    OutputPtr {
        /// Index of the output (0 for single output, 0..n for tuple outputs).
        index: usize,
    },

    // =========================================================================
    // First-class SOAC operations (lowered to loops by ssa_soac_lower)
    // =========================================================================
    /// A first-class SOAC (Second-Order Array Combinator) operation.
    /// These are preserved through optimization passes and lowered to explicit
    /// loops by the `ssa_soac_lower` pass right before backend lowering.
    Soac(SsaSoac),
}

impl InstKind {
    /// Return all ValueIds referenced by this instruction (read-only).
    pub fn value_uses(&self) -> Vec<ValueId> {
        match self {
            InstKind::Int(_)
            | InstKind::Float(_)
            | InstKind::Bool(_)
            | InstKind::Unit
            | InstKind::String(_)
            | InstKind::Global(_)
            | InstKind::Extern(_)
            | InstKind::Alloca { .. }
            | InstKind::OutputPtr { .. } => vec![],

            InstKind::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],
            InstKind::UnaryOp { operand, .. } => vec![*operand],
            InstKind::Tuple(elems) | InstKind::Vector(elems) | InstKind::ArrayLit { elements: elems } => {
                elems.clone()
            }
            InstKind::ArrayRange { start, len, step } => {
                let mut u = vec![*start, *len];
                if let Some(s) = step {
                    u.push(*s);
                }
                u
            }
            InstKind::Matrix(rows) => rows.iter().flatten().copied().collect(),
            InstKind::Project { base, .. } => vec![*base],
            InstKind::Index { base, index } => vec![*base, *index],
            InstKind::Call { args, .. } | InstKind::Intrinsic { args, .. } => args.clone(),
            InstKind::Load { ptr, .. } => vec![*ptr],
            InstKind::Store { ptr, value, .. } => vec![*ptr, *value],
            InstKind::StorageView { source, offset, len } => {
                let mut u = vec![*offset, *len];
                if let ViewSource::Inherited { parent } = source {
                    u.push(*parent);
                }
                u
            }
            InstKind::StorageViewIndex { view, index } => vec![*view, *index],
            InstKind::StorageViewLen { view } => vec![*view],
            InstKind::Soac(soac) => soac.uses(),
        }
    }

    /// Apply a substitution function to all ValueId references in place.
    pub fn substitute_values(&mut self, sub: &mut impl FnMut(&mut ValueId)) {
        match self {
            InstKind::BinOp { lhs, rhs, .. } => {
                sub(lhs);
                sub(rhs);
            }
            InstKind::UnaryOp { operand, .. } => sub(operand),
            InstKind::Tuple(elems) | InstKind::Vector(elems) | InstKind::ArrayLit { elements: elems } => {
                for e in elems {
                    sub(e);
                }
            }
            InstKind::Matrix(rows) => {
                for row in rows {
                    for e in row {
                        sub(e);
                    }
                }
            }
            InstKind::Project { base, .. } => sub(base),
            InstKind::Index { base, index } => {
                sub(base);
                sub(index);
            }
            InstKind::Call { args, .. } | InstKind::Intrinsic { args, .. } => {
                for a in args {
                    sub(a);
                }
            }
            InstKind::Load { ptr, .. } => sub(ptr),
            InstKind::Store { ptr, value, .. } => {
                sub(ptr);
                sub(value);
            }
            InstKind::ArrayRange { start, len, step } => {
                sub(start);
                sub(len);
                if let Some(s) = step {
                    sub(s);
                }
            }
            InstKind::StorageView { source, offset, len } => {
                if let ViewSource::Inherited { parent } = source {
                    sub(parent);
                }
                sub(offset);
                sub(len);
            }
            InstKind::StorageViewIndex { view, index } => {
                sub(view);
                sub(index);
            }
            InstKind::StorageViewLen { view } => sub(view),
            InstKind::Soac(soac) => soac.substitute_uses(sub),
            InstKind::Int(_)
            | InstKind::Float(_)
            | InstKind::Bool(_)
            | InstKind::Unit
            | InstKind::String(_)
            | InstKind::Global(_)
            | InstKind::Extern(_)
            | InstKind::Alloca { .. }
            | InstKind::OutputPtr { .. } => {}
        }
    }

    /// Create a new InstKind with all ValueIds and EffectTokens remapped.
    ///
    /// `rv`: maps source ValueId → target ValueId
    /// `re_in`: maps source effect_in → target effect_in
    /// `alloc_effect`: allocates fresh effect_out tokens for Alloca/Load/Store
    ///
    /// Panics on `Soac` — SOAC lowering expands those rather than remapping.
    pub fn remap(
        &self,
        rv: &impl Fn(&ValueId) -> ValueId,
        re_in: &impl Fn(&EffectToken) -> EffectToken,
        alloc_effect: &mut impl FnMut() -> EffectToken,
    ) -> InstKind {
        match self {
            InstKind::Int(s) => InstKind::Int(s.clone()),
            InstKind::Float(s) => InstKind::Float(s.clone()),
            InstKind::Bool(b) => InstKind::Bool(*b),
            InstKind::Unit => InstKind::Unit,
            InstKind::String(s) => InstKind::String(s.clone()),
            InstKind::BinOp { op, lhs, rhs } => InstKind::BinOp {
                op: op.clone(),
                lhs: rv(lhs),
                rhs: rv(rhs),
            },
            InstKind::UnaryOp { op, operand } => InstKind::UnaryOp {
                op: op.clone(),
                operand: rv(operand),
            },
            InstKind::Tuple(elems) => InstKind::Tuple(elems.iter().map(rv).collect()),
            InstKind::ArrayLit { elements } => InstKind::ArrayLit {
                elements: elements.iter().map(rv).collect(),
            },
            InstKind::ArrayRange { start, len, step } => InstKind::ArrayRange {
                start: rv(start),
                len: rv(len),
                step: step.as_ref().map(rv),
            },
            InstKind::Vector(elems) => InstKind::Vector(elems.iter().map(rv).collect()),
            InstKind::Matrix(rows) => {
                InstKind::Matrix(rows.iter().map(|row| row.iter().map(rv).collect()).collect())
            }
            InstKind::Project { base, index } => InstKind::Project {
                base: rv(base),
                index: *index,
            },
            InstKind::Index { base, index } => InstKind::Index {
                base: rv(base),
                index: rv(index),
            },
            InstKind::Call { func, args } => InstKind::Call {
                func: func.clone(),
                args: args.iter().map(rv).collect(),
            },
            InstKind::Global(name) => InstKind::Global(name.clone()),
            InstKind::Extern(name) => InstKind::Extern(name.clone()),
            InstKind::Intrinsic { name, args } => InstKind::Intrinsic {
                name: name.clone(),
                args: args.iter().map(rv).collect(),
            },
            InstKind::Alloca {
                elem_ty, effect_in, ..
            } => InstKind::Alloca {
                elem_ty: elem_ty.clone(),
                effect_in: re_in(effect_in),
                effect_out: alloc_effect(),
            },
            InstKind::Load { ptr, effect_in, .. } => InstKind::Load {
                ptr: rv(ptr),
                effect_in: re_in(effect_in),
                effect_out: alloc_effect(),
            },
            InstKind::Store {
                ptr,
                value,
                effect_in,
                ..
            } => InstKind::Store {
                ptr: rv(ptr),
                value: rv(value),
                effect_in: re_in(effect_in),
                effect_out: alloc_effect(),
            },
            InstKind::StorageView { source, offset, len } => InstKind::StorageView {
                source: match source {
                    ViewSource::Storage { set, binding } => ViewSource::Storage {
                        set: *set,
                        binding: *binding,
                    },
                    ViewSource::Inherited { parent } => ViewSource::Inherited { parent: rv(parent) },
                },
                offset: rv(offset),
                len: rv(len),
            },
            InstKind::StorageViewIndex { view, index } => InstKind::StorageViewIndex {
                view: rv(view),
                index: rv(index),
            },
            InstKind::StorageViewLen { view } => InstKind::StorageViewLen { view: rv(view) },
            InstKind::OutputPtr { index } => InstKind::OutputPtr { index: *index },
            InstKind::Soac(_) => {
                panic!("ICE: Soac in InstKind::remap — lower SOACs before remapping")
            }
        }
    }
}

/// First-class SOAC (Second-Order Array Combinator) operations in SSA.
///
/// After defunctionalization, lambda bodies are just function references,
/// so we store the function name (like `Call`) rather than nested regions.
#[derive(Debug, Clone)]
pub enum SsaSoac {
    /// `map f inputs` — apply `f` to each element, producing an output array.
    Map {
        /// Name of the map function (post-defunctionalization).
        func: String,
        /// Input arrays (one for single-input map, multiple for zip-fused map).
        inputs: Vec<ValueId>,
        /// Captured variables passed as extra arguments to `func`.
        captures: Vec<ValueId>,
        /// Whether inputs were zip-fused (multiple inputs packed into tuple arg).
        zipped: bool,
        /// Types of each input array (for SoA-aware length/indexing).
        input_array_types: Vec<Type<TypeName>>,
        /// Element types of each input array (for SoA-aware indexing).
        input_elem_types: Vec<Type<TypeName>>,
        /// Element type of the output array (for SoA-aware array_with).
        output_elem_type: Type<TypeName>,
        /// Type of the zipped parameter (when `zipped` is true).
        zipped_param_type: Option<Type<TypeName>>,
    },
    /// `reduce f init input` — fold `f` over elements with initial value `init`.
    Reduce {
        /// Name of the reduce operator function.
        func: String,
        /// The input array to reduce over.
        input: ValueId,
        /// The initial accumulator value.
        init: ValueId,
        /// Captured variables passed as extra arguments to `func`.
        captures: Vec<ValueId>,
        /// Type of the input array (for SoA-aware operations).
        input_array_type: Type<TypeName>,
        /// Element type of the input array (for SoA-aware indexing).
        input_elem_type: Type<TypeName>,
    },
    /// `scan f init input` — like reduce but produces array of intermediate results.
    Scan {
        /// Name of the scan operator function.
        func: String,
        /// The input array to scan over.
        input: ValueId,
        /// The initial accumulator value.
        init: ValueId,
        /// Captured variables passed as extra arguments to `func`.
        captures: Vec<ValueId>,
        /// Type of the input array (for SoA-aware operations).
        input_array_type: Type<TypeName>,
        /// Element type of the input array (for SoA-aware indexing).
        input_elem_type: Type<TypeName>,
    },
}

impl SsaSoac {
    /// Return all ValueIds referenced by this SOAC operation.
    pub fn uses(&self) -> Vec<ValueId> {
        match self {
            SsaSoac::Map { inputs, captures, .. } => {
                let mut uses = inputs.clone();
                uses.extend(captures.iter().copied());
                uses
            }
            SsaSoac::Reduce {
                input,
                init,
                captures,
                ..
            } => {
                let mut uses = vec![*input, *init];
                uses.extend(captures.iter().copied());
                uses
            }
            SsaSoac::Scan {
                input,
                init,
                captures,
                ..
            } => {
                let mut uses = vec![*input, *init];
                uses.extend(captures.iter().copied());
                uses
            }
        }
    }

    /// Apply a substitution function to all ValueId references in this SOAC.
    pub fn substitute_uses(&mut self, sub: &mut impl FnMut(&mut ValueId)) {
        match self {
            SsaSoac::Map { inputs, captures, .. } => {
                for v in inputs.iter_mut() {
                    sub(v);
                }
                for v in captures.iter_mut() {
                    sub(v);
                }
            }
            SsaSoac::Reduce {
                input,
                init,
                captures,
                ..
            } => {
                sub(input);
                sub(init);
                for v in captures.iter_mut() {
                    sub(v);
                }
            }
            SsaSoac::Scan {
                input,
                init,
                captures,
                ..
            } => {
                sub(input);
                sub(init);
                for v in captures.iter_mut() {
                    sub(v);
                }
            }
        }
    }

    /// Return the name of the function applied by this SOAC.
    pub fn func_name(&self) -> &str {
        match self {
            SsaSoac::Map { func, .. } => func,
            SsaSoac::Reduce { func, .. } => func,
            SsaSoac::Scan { func, .. } => func,
        }
    }
}

/// Where a view array gets its data from.
#[derive(Debug, Clone, PartialEq)]
pub enum ViewSource {
    /// Backed by a storage buffer at (set, binding).
    Storage {
        set: u32,
        binding: u32,
    },
    /// Inherited from another view value (e.g. a function parameter or a slice).
    /// The SPIR-V backend follows this chain to find the underlying Storage source.
    Inherited {
        parent: ValueId,
    },
}

// =============================================================================
// Function Body
// =============================================================================

/// An SSA function body.
///
/// Contains blocks forming a CFG, with BlockId(0) as the entry block.
#[derive(Debug, Clone)]
pub struct FuncBody {
    /// Function parameters (value, type, name).
    pub params: Vec<(ValueId, Type<TypeName>, String)>,

    /// Return type of the function.
    pub return_ty: Type<TypeName>,

    /// Basic blocks. BlockId(0) is the entry block.
    pub blocks: Vec<Block>,

    /// Instruction arena. Indexed by InstId.
    pub insts: Vec<Inst>,

    /// Type of each value. Indexed by ValueId.
    pub value_types: Vec<Type<TypeName>>,

    /// Initial effect token for the function entry.
    pub entry_effect: EffectToken,

    /// Next effect token ID (for allocation).
    next_effect: u32,

    /// Next value ID (for allocation).
    next_value: u32,

    /// DPS output parameter (if using destination-passing style).
    pub dps_output: Option<ValueId>,
}

impl FuncBody {
    /// Create a new function body with the given parameters and return type.
    pub fn new(params: Vec<(Type<TypeName>, String)>, return_ty: Type<TypeName>) -> Self {
        let mut value_types = Vec::new();
        let mut func_params = Vec::new();

        // Allocate values for parameters
        for (i, (ty, name)) in params.into_iter().enumerate() {
            let value_id = ValueId(i as u32);
            value_types.push(ty.clone());
            func_params.push((value_id, ty, name));
        }

        // Entry block (no parameters)
        let entry_block = Block::new();

        let next_value = value_types.len() as u32;
        FuncBody {
            params: func_params,
            return_ty,
            blocks: vec![entry_block],
            insts: Vec::new(),
            value_types,
            entry_effect: EffectToken(0),
            next_effect: 1,
            next_value,
            dps_output: None,
        }
    }

    /// Allocate a new value ID with the given type.
    pub fn alloc_value(&mut self, ty: Type<TypeName>) -> ValueId {
        let id = ValueId(self.next_value);
        self.next_value += 1;
        self.value_types.push(ty);
        id
    }

    /// Allocate a new effect token.
    pub fn alloc_effect(&mut self) -> EffectToken {
        let token = EffectToken(self.next_effect);
        self.next_effect += 1;
        token
    }

    /// Get the type of a value.
    pub fn get_value_type(&self, value: ValueId) -> &Type<TypeName> {
        &self.value_types[value.index()]
    }

    /// Get a block by ID.
    pub fn get_block(&self, id: BlockId) -> &Block {
        &self.blocks[id.index()]
    }

    /// Get a mutable reference to a block.
    pub fn get_block_mut(&mut self, id: BlockId) -> &mut Block {
        &mut self.blocks[id.index()]
    }

    /// Get an instruction by ID.
    pub fn get_inst(&self, id: InstId) -> &Inst {
        &self.insts[id.index()]
    }

    /// Number of blocks in this function.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Number of instructions in this function.
    pub fn num_insts(&self) -> usize {
        self.insts.len()
    }

    /// Number of values in this function.
    pub fn num_values(&self) -> usize {
        self.value_types.len()
    }
}

// =============================================================================
// Display Implementations
// =============================================================================

impl std::fmt::Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

impl std::fmt::Display for InstId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i{}", self.0)
    }
}

impl std::fmt::Display for EffectToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "!{}", self.0)
    }
}
