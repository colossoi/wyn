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
//! - `StorageImageLoad(resource)`: `[coord]`
//! - `StorageImageStore(resource)`: `[coord, texel]`
//! - `StorageView(Storage)`: `[offset, len]`
//! - `StorageView(Inherited)`: `[offset, len, parent]`
//! - `StorageViewLen`: `[view]`
//! - `ViewIndex`: `[view, index]` — EGIR-only (place-producing; SSA uses
//!   `InstKind::ViewIndex` which carries a fresh `PlaceId`).
//! - `OutputSlot { index }`: 0 — EGIR-only (place-producing; SSA uses
//!   `InstKind::OutputSlot`).

use crate::BindingRef;

/// The operator identity shared by EGIR's pure nodes and SSA's `InstKind::Op`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OpTag<R = BindingRef> {
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
    /// Read a storage image whose resource is fixed by region
    /// monomorphization. The image handle is absent from the runtime operands.
    StorageImageLoad(R),
    /// Write a storage image whose resource is fixed by region
    /// monomorphization. Runtime operands are coordinate and texel only.
    StorageImageStore(R),
    /// Storage buffer view creation. The `Inherited` parent (if any) is
    /// carried in the operands tail, not in this tag, so equivalent views
    /// with the same backing source hash-cons together.
    StorageView(PureViewSource<R>),
    /// EGIR-only logical storage length. Physicalization resolves the
    /// `ResourceId` to a descriptor binding and rewrites this to the ordinary
    /// storage-length intrinsic before SSA elaboration.
    ResourceLen(R),
    StorageViewLen,
    /// EGIR-only: place-producing view-index. Never appears in
    /// `InstKind::Op` — SSA uses `InstKind::ViewIndex` directly (carries
    /// a fresh `PlaceId`).
    ViewIndex,
    /// EGIR-only: index into another place to produce a sub-place. Operands
    /// `[parent_place, index]`. Mirrors `ViewIndex` but the parent is itself
    /// a place (e.g. an `Alloca`'d `[T;N]`) rather than a value-typed view —
    /// elaborate maps it to `InstKind::PlaceIndex` carrying a fresh `PlaceId`.
    /// Used by `soac_expand` to write directly into a function-local array
    /// place without going through a whole-array `Load`/`Store` round-trip.
    PlaceIndex,
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
pub enum PureViewSource<R = BindingRef> {
    Storage(R),
    Inherited,
    /// Workgroup-shared array, `id`-th in the entry, of `count` elements.
    /// Unlike Storage there is no descriptor binding — the backend declares
    /// a module-scope `array<T, count>` in workgroup storage. The element
    /// type comes from the view's result type. Emitted by the
    /// workgroup-parallel reduce phase 2.
    Workgroup {
        id: u32,
        count: u32,
    },
}

impl<R> PureViewSource<R> {
    pub fn try_map_resource<S, E>(
        self,
        map: &mut impl FnMut(R) -> Result<S, E>,
    ) -> Result<PureViewSource<S>, E> {
        Ok(match self {
            PureViewSource::Storage(resource) => PureViewSource::Storage(map(resource)?),
            PureViewSource::Inherited => PureViewSource::Inherited,
            PureViewSource::Workgroup { id, count } => PureViewSource::Workgroup { id, count },
        })
    }
}

impl<R> OpTag<R> {
    /// Resource identity carried directly by this operator, excluding
    /// resource handles represented by ordinary operands.
    pub fn referenced_resource(&self) -> Option<&R> {
        match self {
            OpTag::StorageImageLoad(resource)
            | OpTag::StorageImageStore(resource)
            | OpTag::ResourceLen(resource)
            | OpTag::StorageView(PureViewSource::Storage(resource)) => Some(resource),
            _ => None,
        }
    }

    pub fn try_map_resource<S, E>(self, map: &mut impl FnMut(R) -> Result<S, E>) -> Result<OpTag<S>, E> {
        Ok(match self {
            OpTag::Int(value) => OpTag::Int(value),
            OpTag::Uint(value) => OpTag::Uint(value),
            OpTag::Float(value) => OpTag::Float(value),
            OpTag::Bool(value) => OpTag::Bool(value),
            OpTag::Unit => OpTag::Unit,
            OpTag::Global(value) => OpTag::Global(value),
            OpTag::Extern(value) => OpTag::Extern(value),
            OpTag::BinOp(value) => OpTag::BinOp(value),
            OpTag::UnaryOp(value) => OpTag::UnaryOp(value),
            OpTag::Tuple(value) => OpTag::Tuple(value),
            OpTag::Vector(value) => OpTag::Vector(value),
            OpTag::Matrix { rows, cols } => OpTag::Matrix { rows, cols },
            OpTag::ArrayLit(value) => OpTag::ArrayLit(value),
            OpTag::ArrayRange { has_step } => OpTag::ArrayRange { has_step },
            OpTag::Project { index } => OpTag::Project { index },
            OpTag::Index => OpTag::Index,
            OpTag::Materialize => OpTag::Materialize,
            OpTag::DynamicExtract => OpTag::DynamicExtract,
            OpTag::Call(value) => OpTag::Call(value),
            OpTag::Intrinsic { id, overload_idx } => OpTag::Intrinsic { id, overload_idx },
            OpTag::StorageImageLoad(resource) => OpTag::StorageImageLoad(map(resource)?),
            OpTag::StorageImageStore(resource) => OpTag::StorageImageStore(map(resource)?),
            OpTag::StorageView(source) => OpTag::StorageView(source.try_map_resource(map)?),
            OpTag::ResourceLen(resource) => OpTag::ResourceLen(map(resource)?),
            OpTag::StorageViewLen => OpTag::StorageViewLen,
            OpTag::ViewIndex => OpTag::ViewIndex,
            OpTag::PlaceIndex => OpTag::PlaceIndex,
            OpTag::OutputSlot { index } => OpTag::OutputSlot { index },
        })
    }
}
