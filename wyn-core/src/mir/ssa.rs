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
        set: u32,
        binding: u32,
        offset: ValueId,
        len: ValueId,
    },

    /// Index into a storage view (returns a pointer).
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
