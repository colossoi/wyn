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

use crate::BindingRef;
use crate::ast::{Span, TypeName};
use crate::interface;
use crate::op::OpTag;
use polytype::Type;
use rspirv::spirv;
use slotmap::SlotMap;
use std::collections::HashMap;

// Re-export ID types from wyn-ssa.
pub use crate::ssa::framework::{BlockId, InstId, PlaceId, ValueId};
// Re-export Terminator from wyn-ssa.
pub use crate::ssa::framework::Terminator;
// Re-export BasicBlock from wyn-ssa.
pub use crate::ssa::framework::BasicBlock;

/// A compile-time constant value that can be carried inline in a `ValueRef`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstantValue {
    I32(i32),
    U32(u32),
    F32(u32), // stored as bits for Eq/Hash
    Bool(bool),
}

impl ConstantValue {
    pub fn from_f32(v: f32) -> Self {
        ConstantValue::F32(v.to_bits())
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            ConstantValue::F32(bits) => Some(f32::from_bits(*bits)),
            _ => None,
        }
    }
}

/// A reference to a value: either an SSA instruction result or an inline constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueRef {
    Ssa(ValueId),
    Const(ConstantValue),
}

impl ValueRef {
    pub fn as_ssa(&self) -> Option<ValueId> {
        match self {
            ValueRef::Ssa(id) => Some(*id),
            _ => None,
        }
    }
    pub fn as_const(&self) -> Option<ConstantValue> {
        match self {
            ValueRef::Const(c) => Some(*c),
            _ => None,
        }
    }
    pub fn map_ssa(&self, f: impl Fn(ValueId) -> ValueId) -> ValueRef {
        match self {
            ValueRef::Ssa(id) => ValueRef::Ssa(f(*id)),
            ValueRef::Const(c) => ValueRef::Const(*c),
        }
    }
    pub fn substitute(&mut self, f: &mut impl FnMut(&mut ValueId)) {
        if let ValueRef::Ssa(id) = self {
            f(id);
        }
    }
}

impl From<ValueId> for ValueRef {
    fn from(id: ValueId) -> Self {
        ValueRef::Ssa(id)
    }
}

// =============================================================================
// ControlHeader (side-map metadata, not part of BasicBlock)
// =============================================================================

/// Structured control flow header information for SPIR-V lowering.
///
/// SPIR-V requires explicit merge/continue annotations for loops and selections.
/// Stored in a side-map on FuncBody, keyed by BlockId.
#[derive(Debug, Clone)]
pub enum ControlHeader {
    Loop {
        merge: BlockId,
        continue_block: BlockId,
    },
    Selection {
        merge: BlockId,
    },
}

impl ControlHeader {
    pub fn remap(&self, rb: &impl Fn(BlockId) -> BlockId) -> ControlHeader {
        match self {
            ControlHeader::Loop {
                merge,
                continue_block,
            } => ControlHeader::Loop {
                merge: rb(*merge),
                continue_block: rb(*continue_block),
            },
            ControlHeader::Selection { merge } => ControlHeader::Selection { merge: rb(*merge) },
        }
    }
}

// =============================================================================
// Terminator extension: remap (block target remapping not in wyn-ssa)
// =============================================================================

/// Extension methods for Terminator that wyn-core needs but aren't in wyn-ssa.
pub trait TerminatorExt {
    fn remap(&self, rv: &impl Fn(ValueId) -> ValueId, rb: &impl Fn(BlockId) -> BlockId) -> Terminator;
}

impl TerminatorExt for Terminator {
    fn remap(&self, rv: &impl Fn(ValueId) -> ValueId, rb: &impl Fn(BlockId) -> BlockId) -> Terminator {
        match self {
            Terminator::Branch { target, args } => Terminator::Branch {
                target: rb(*target),
                args: args.iter().copied().map(rv).collect(),
            },
            Terminator::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
            } => Terminator::CondBranch {
                cond: rv(*cond),
                then_target: rb(*then_target),
                then_args: then_args.iter().copied().map(rv).collect(),
                else_target: rb(*else_target),
                else_args: else_args.iter().copied().map(rv).collect(),
            },
            Terminator::Return(Some(v)) => Terminator::Return(Some(rv(*v))),
            Terminator::Return(None) => Terminator::Return(None),
            Terminator::Unreachable => Terminator::Unreachable,
        }
    }
}

// =============================================================================
// Instructions
// =============================================================================

/// The kind of operation an instruction performs.
///
/// Pure operations (literals, arithmetic, calls, intrinsics, view ops,
/// materialize/extract) all collapse into a single `Op { tag, operands }`
/// variant carrying an `OpTag` from `crate::op`. The five effectful
/// variants stay separate because they carry a `PlaceId`, which can't
/// fold into an operand list.
#[derive(Debug, Clone)]
pub enum InstKind {
    /// A pure operation. Operand layout per tag is documented on
    /// `OpTag` in `crate::op`.
    Op {
        tag: OpTag,
        operands: Vec<ValueRef>,
    },

    // =========================================================================
    // Effectful Operations (effects tracked on InstNode, not here)
    // =========================================================================
    /// Allocate a local place (function-scope variable).
    Alloca {
        elem_ty: Type<TypeName>,
        result: PlaceId,
    },

    /// Load a value from a place.
    Load {
        place: PlaceId,
    },

    /// Store a value to a place.
    Store {
        place: PlaceId,
        value: ValueRef,
    },

    /// Index into a storage view. Produces an addressable place
    /// (see `PlaceId` — registered in `FuncBody.places` at emit time).
    ViewIndex {
        view: ValueRef,
        index: ValueRef,
        result: PlaceId,
    },

    /// Index into a place whose element type is an array — produces a sub-place
    /// addressing one element. Mirrors `ViewIndex` but takes a `PlaceId` instead
    /// of a value, avoiding the whole-array `Load` that the value-form would
    /// require. The natural pair for `Alloca { elem_ty: [T;N] }`: backends
    /// lower to `OpAccessChain` (SPIR-V) / `var[idx]` (WGSL) directly.
    PlaceIndex {
        place: PlaceId,
        index: ValueRef,
        result: PlaceId,
    },

    /// An entry-point output slot — a Place that `Store` writes into before
    /// returning. Registered in `FuncBody.places` at emit time.
    OutputSlot {
        /// Index of the output (0 for single output, 0..n for tuple outputs).
        index: usize,
        result: PlaceId,
    },

    /// Workgroup execution + memory barrier (`workgroupBarrier()` /
    /// `OpControlBarrier`). No operands, no result; ordered by the effect
    /// chain so it isn't reordered around the workgroup-shared loads/stores
    /// it synchronizes. Emitted by the workgroup-parallel reduce phase 2.
    ControlBarrier,
}

impl InstKind {
    /// Return all ValueRefs referenced by this instruction (read-only).
    /// Place operands (see `place_uses`) are traversed separately.
    pub fn value_uses(&self) -> Vec<ValueRef> {
        match self {
            InstKind::Op { operands, .. } => operands.clone(),
            InstKind::Alloca { .. }
            | InstKind::OutputSlot { .. }
            | InstKind::Load { .. }
            | InstKind::ControlBarrier => vec![],
            InstKind::Store { value, .. } => vec![*value],
            InstKind::ViewIndex { view, index, .. } => vec![*view, *index],
            InstKind::PlaceIndex { index, .. } => vec![*index],
        }
    }

    /// Return all places *consumed* by this instruction — i.e. places that
    /// appear as operands (the `place` field of `Load`/`Store`), not
    /// the place this instruction *produces* (that's its result). Use
    /// `place_result()` for the latter.
    pub fn place_uses(&self) -> Vec<PlaceId> {
        match self {
            InstKind::Load { place } => vec![*place],
            InstKind::Store { place, .. } => vec![*place],
            InstKind::PlaceIndex { place, .. } => vec![*place],
            _ => vec![],
        }
    }

    /// The `PlaceId` this instruction *produces*, if any.
    pub fn place_result(&self) -> Option<PlaceId> {
        match self {
            InstKind::Alloca { result, .. }
            | InstKind::OutputSlot { result, .. }
            | InstKind::ViewIndex { result, .. }
            | InstKind::PlaceIndex { result, .. } => Some(*result),
            _ => None,
        }
    }

    /// Return only the SSA ValueIds referenced by this instruction.
    pub fn ssa_uses(&self) -> Vec<ValueId> {
        self.value_uses().into_iter().filter_map(|r| r.as_ssa()).collect()
    }

    /// Apply a substitution function to all ValueRef references in place.
    pub fn substitute_values(&mut self, sub: &mut impl FnMut(&mut ValueRef)) {
        match self {
            InstKind::Op { operands, .. } => {
                for o in operands {
                    sub(o);
                }
            }
            InstKind::Store { value, .. } => sub(value),
            InstKind::ViewIndex { view, index, .. } => {
                sub(view);
                sub(index);
            }
            InstKind::PlaceIndex { index, .. } => sub(index),
            InstKind::Alloca { .. }
            | InstKind::OutputSlot { .. }
            | InstKind::Load { .. }
            | InstKind::ControlBarrier => {}
        }
    }

    /// Apply a substitution function to every `PlaceId` referenced by this
    /// instruction (both operand places and the `result` place, if any).
    pub fn substitute_places(&mut self, sub: &mut impl FnMut(&mut PlaceId)) {
        match self {
            InstKind::Load { place } => sub(place),
            InstKind::Store { place, .. } => sub(place),
            InstKind::Alloca { result, .. }
            | InstKind::OutputSlot { result, .. }
            | InstKind::ViewIndex { result, .. } => sub(result),
            InstKind::PlaceIndex { place, result, .. } => {
                sub(place);
                sub(result);
            }
            InstKind::Op { .. } | InstKind::ControlBarrier => {}
        }
    }

    /// Create a new InstKind with all ValueIds remapped.
    pub fn remap(&self, rv: &impl Fn(ValueId) -> ValueId) -> InstKind {
        let mut result = self.clone();
        result.substitute_values(&mut |vr| {
            *vr = vr.map_ssa(rv);
        });
        result
    }
}

/// Metadata for a `PlaceId` — addressable locations that `Load` reads
/// from and `Store` writes to. Places are identity-based (distinct
/// places never alias); they never flow through block params. The
/// defining instruction is found by scanning the IR; see
/// `FuncBody::place_of_inst`.
#[derive(Debug, Clone)]
pub struct PlaceInfo {
    pub elem_ty: Type<TypeName>,
}

/// Where a view array gets its data from.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ViewSource {
    /// Backed by a storage buffer at `binding.set` / `binding.binding`.
    Storage(BindingRef),
    /// Inherited from another view value (e.g. a function parameter or a slice).
    /// The SPIR-V backend follows this chain to find the underlying Storage source.
    Inherited {
        parent: ValueId,
    },
}

// =============================================================================
// Function Body
// =============================================================================

/// The concrete wyn-ssa Function type used throughout wyn-core.
pub type WynFunction = crate::ssa::framework::Function<InstKind, Type<TypeName>>;
/// The concrete wyn-ssa InstNode type.
pub type WynInstNode = crate::ssa::framework::InstNode<InstKind>;

/// An SSA function body.
#[derive(Debug, Clone)]
pub struct FuncBody {
    /// The underlying generic SSA function.
    pub inner: WynFunction,

    /// Structured control flow headers (for SPIR-V lowering).
    pub control_headers: HashMap<BlockId, ControlHeader>,

    /// Function parameters (value, type, name).
    pub params: Vec<(ValueId, Type<TypeName>, String)>,

    /// Return type of the function.
    pub return_ty: Type<TypeName>,

    /// DPS output parameter (if using destination-passing style).
    pub dps_output: Option<ValueId>,

    /// Addressable places (`OutputSlot`, `ViewIndex`, `Alloca` results).
    pub places: SlotMap<PlaceId, PlaceInfo>,
}

impl FuncBody {
    /// Get the type of a value.
    pub fn get_value_type(&self, value: ValueId) -> &Type<TypeName> {
        self.inner.value_type(value)
    }

    /// Get a block by ID.
    pub fn get_block(&self, id: BlockId) -> &BasicBlock {
        &self.inner.blocks[id]
    }

    /// Get a mutable reference to a block.
    pub fn get_block_mut(&mut self, id: BlockId) -> &mut BasicBlock {
        &mut self.inner.blocks[id]
    }

    /// Get an instruction by ID.
    pub fn get_inst(&self, id: InstId) -> &WynInstNode {
        &self.inner.insts[id]
    }

    /// Entry block ID.
    pub fn entry_block(&self) -> BlockId {
        self.inner.entry
    }

    /// Number of blocks in this function.
    pub fn num_blocks(&self) -> usize {
        self.inner.blocks.len()
    }

    /// Number of instructions in this function.
    pub fn num_insts(&self) -> usize {
        self.inner.insts.len()
    }

    /// Number of values in this function.
    pub fn num_values(&self) -> usize {
        self.inner.values.len()
    }

    /// Element type of the place (what `Load` returns / `Store` writes).
    pub fn place_elem_ty(&self, place: PlaceId) -> &Type<TypeName> {
        &self.places[place].elem_ty
    }

    /// `PlaceId` defined by the given instruction, if any.
    pub fn place_of_inst(&self, inst: InstId) -> Option<PlaceId> {
        match &self.inner.insts[inst].data {
            InstKind::Alloca { result, .. }
            | InstKind::OutputSlot { result, .. }
            | InstKind::ViewIndex { result, .. } => Some(*result),
            _ => None,
        }
    }
}

// =============================================================================
// Instr trait implementation for InstKind
// =============================================================================

// =============================================================================
// Display Implementations
// =============================================================================

// =============================================================================
// Program-Level Types
// =============================================================================

/// An SSA program — the result of converting TLC to SSA.
#[derive(Debug, Clone)]
pub struct Program {
    /// Function definitions with their SSA bodies.
    pub functions: Vec<Function>,
    /// Entry point definitions.
    pub entry_points: Vec<EntryPoint>,
    /// Program-level constant definitions (zero-arg defs with purely constant bodies).
    /// Emitted once at module scope; functions reference them via `InstKind::Global`.
    pub constants: Vec<Constant>,
}

/// A program-level constant definition.
#[derive(Debug, Clone)]
pub struct Constant {
    pub name: String,
    pub body: FuncBody,
}

/// A function definition.
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub body: FuncBody,
    pub span: Span,
    /// For extern functions, the linkage name.
    pub linkage_name: Option<String>,
}

/// An entry point definition.
#[derive(Debug, Clone)]
pub struct EntryPoint {
    pub name: String,
    pub body: FuncBody,
    pub execution_model: ExecutionModel,
    pub inputs: Vec<EntryInput>,
    pub outputs: Vec<EntryOutput>,
    /// Compiler-introduced storage bindings the entry touches beyond its
    /// declared inputs/outputs (e.g. partials/intermediate buffers emitted
    /// by `parallelize`). Carried end-to-end so SPIR-V generation has a
    /// single source of truth for each entry's buffer interface.
    pub storage_bindings: Vec<interface::StorageBindingDecl>,
    pub span: Span,
}

/// Execution model for entry points.
#[derive(Debug, Clone)]
pub enum ExecutionModel {
    Vertex,
    Fragment,
    Compute {
        local_size: (u32, u32, u32),
    },
}

/// A compute-broadcast input's placement in the push-constant block.
/// `offset` is the byte offset within the block; `size` is the byte
/// width of the param's static layout. Both are decided at binding
/// allocation time, so downstream backends never have to re-derive
/// either from the param type.
#[derive(Debug, Clone, Copy)]
pub struct PushConstantSlot {
    pub offset: u32,
    pub size: u32,
}

/// Input to an entry point.
#[derive(Debug, Clone)]
pub struct EntryInput {
    pub name: String,
    pub ty: Type<TypeName>,
    pub decoration: Option<IoDecoration>,
    pub size_hint: Option<std::num::NonZeroU32>,
    pub storage_binding: Option<BindingRef>,
    /// Declared `access` of a `#[storage(...)]` param. `WriteOnly`/`ReadWrite`
    /// mark the buffer as written in place (e.g. a `scatter` destination), so
    /// the descriptor and WGSL emit a writable binding rather than read-only.
    pub storage_access: Option<crate::interface::StorageAccess>,
    /// Programmer-attributed `#[uniform(set, binding)]` on this param.
    pub uniform_binding: Option<BindingRef>,
    /// Placement in the per-entry push-constant block for compute-broadcast
    /// inputs. `Some` carries both byte offset and byte size, decided at
    /// the moment the offset was assigned.
    pub push_constant: Option<PushConstantSlot>,
    /// Programmer-attributed `#[texture(set, binding)]` on a `texture2d` param.
    pub texture_binding: Option<BindingRef>,
    /// Programmer-attributed `#[sampler(set, binding)]` on a `sampler` param.
    pub sampler_binding: Option<BindingRef>,
    /// Programmer-attributed `#[storage_image(set, binding, format, access,
    /// size)]` on a `storage_image` param. Carries the binding ref plus
    /// the format / access / sizing-policy attributes from the source.
    pub storage_image_binding: Option<(
        BindingRef,
        crate::pipeline_descriptor::StorageImageFormat,
        crate::interface::StorageAccess,
        crate::pipeline_descriptor::StorageTextureSize,
    )>,
}

/// Output from an entry point.
#[derive(Debug, Clone)]
pub struct EntryOutput {
    pub ty: Type<TypeName>,
    pub decoration: Option<IoDecoration>,
    /// For compute shaders with unsized array outputs.
    pub storage_binding: Option<BindingRef>,
    /// Sizing policy for a runtime-sized storage output (a parallel map/scan
    /// writes one element per thread → `SameAsDispatch`). `None` for
    /// fixed/graphics outputs and entries whose output size the host already
    /// knows another way.
    pub length: Option<crate::pipeline_descriptor::BufferLen>,
}

/// I/O decoration for entry point parameters.
#[derive(Debug, Clone)]
pub enum IoDecoration {
    BuiltIn(spirv::BuiltIn),
    Location(u32),
}
