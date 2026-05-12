//! Operator tags shared between EGIR's pure node identity (`PureOp`) and
//! SSA's `InstKind::Op` form.
//!
//! Each variant identifies a kind of operation; operands are carried
//! separately:
//! - In EGIR: as `SmallVec<[NodeId; 4]>` inside `ENode::Pure`.
//! - In SSA:  as `Vec<ValueRef>` inside `InstKind::Op { tag, operands }`.
//!
//! ## Operand layout per tag
//!
//! - `Int` / `Uint` / `Float` / `Bool` / `Unit` / `Global` / `Extern`: 0
//! - `BinOp(_)`: `[lhs, rhs]`
//! - `UnaryOp(_)`: `[operand]`
//! - `Tuple(n)` / `Vector(n)` / `ArrayLit(n)`: `n` operands
//! - `Matrix { rows, cols }`: `rows * cols` operands, row-major
//! - `ArrayRange { has_step }`: `[start, len]` or `[start, len, step]`
//! - `Project { index }`: `[base]`
//! - `Index`: `[base, index]`
//! - `Materialize`: `[value]`
//! - `DynamicExtract`: `[base, index]`
//! - `Call(name)` / `Intrinsic { .. }`: variable-arity arg list
//! - `StorageView(Storage)`: `[offset, len]`
//! - `StorageView(Inherited)`: `[offset, len, parent]`
//! - `StorageViewLen`: `[view]`
//! - `ViewIndex`: `[view, index]` — EGIR-only (place-producing; SSA uses
//!   `InstKind::ViewIndex` which carries a fresh `PlaceId`).
//! - `OutputSlot { index }`: 0 — EGIR-only (place-producing; SSA uses
//!   `InstKind::OutputSlot`).

/// The operator identity shared by EGIR's pure nodes and SSA's `InstKind::Op`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OpTag {
    /// Signed integer literal (i8, i16, i32, i64).
    Int(String),
    /// Unsigned integer literal (u8, u16, u32, u64).
    Uint(String),
    Float(String),
    Bool(bool),
    Unit,
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
    Intrinsic {
        id: crate::builtins::BuiltinId,
        overload_idx: usize,
    },
    /// Storage buffer view creation. The `Inherited` parent (if any) is
    /// carried in the operands tail, not in this tag, so equivalent views
    /// with the same backing source hash-cons together.
    StorageView(PureViewSource),
    StorageViewLen,
    /// EGIR-only: place-producing view-index. Never appears in
    /// `InstKind::Op` — SSA uses `InstKind::ViewIndex` directly (carries
    /// a fresh `PlaceId`).
    ViewIndex,
    /// EGIR-only: place-producing entry-output slot. Never appears in
    /// `InstKind::Op` — SSA uses `InstKind::OutputSlot` directly.
    OutputSlot {
        index: usize,
    },
}

/// Hashable variant of `ViewSource` for use inside an `OpTag`. Drops the
/// `ValueId` from `Inherited` — that parent is stored as an operand in the
/// containing `ENode::Pure` or `InstKind::Op`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PureViewSource {
    Storage {
        set: u32,
        binding: u32,
    },
    Inherited,
}
