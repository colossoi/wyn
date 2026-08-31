//! Operator tags shared between EGIR's pure node identity (`PureOp`) and
//! SSA's `InstKind::Op` form.
//!
//! Each variant identifies a kind of operation; operands are carried
//! separately:
//! - In EGIR: as `SmallVec<[ValueId; 4]>` inside `ValueKind::Pure`.
//! - In SSA:  as `Vec<ValueRef>` inside `InstKind::Op { tag, operands }`.
//!
//! ## Operand layout per tag
//!
//! - `Int` / `Uint` / `Float` / `Bool` / `Unit` / `Global`: 0
//! - `BinOp(_)`: `[lhs, rhs]`
//! - `UnaryOp(_)`: `[operand]`
//! - `Tuple(n)` / `Vector(n)` / `ArrayLit(n)`: `n` operands
//! - `Matrix { rows, cols }`: `rows * cols` operands, row-major
//! - `ArrayRange { has_step }`: `[start, len]` or `[start, len, step]`
//! - `Project { index }`: `[base]`
//! - `Index`: `[base, index]`
//! - `Materialize`: `[value]`
//! - `DynamicExtract`: `[base, index]`
//! - `Call(function)` / `Intrinsic { .. }`: variable-arity arg list
//! - `StorageImageLoad(resource)`: `[coord]`
//! - `StorageImageStore(resource)`: `[coord, texel]`
//! - `StorageView(Storage)`: `[offset, len]`
//! - `StorageView(Inherited)`: `[offset, len, parent]`
//! - `StorageViewLen`: `[view]`

use crate::builtins;
use crate::BindingRef;
use crate::GlobalId;

/// Index into the SSA program's table of promoted addressable constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AddressableConstantId(pub u32);

/// Structurally resolved binary operator.
///
/// The parser may temporarily carry an operator token as text, but TLC and all
/// later IRs use this closed representation. Backends dispatch on variants and
/// use [`BinaryOperator::symbol`] only when rendering output or diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    FloorDivide,
    FloorRemainder,
    Power,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    LogicalAnd,
    LogicalOr,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftLeft,
    ShiftRight,
    ShiftRightLogical,
}

impl BinaryOperator {
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Subtract => "-",
            Self::Multiply => "*",
            Self::Divide => "/",
            Self::Remainder => "%",
            Self::FloorDivide => "//",
            Self::FloorRemainder => "%%",
            Self::Power => "**",
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
            Self::LogicalAnd => "&&",
            Self::LogicalOr => "||",
            Self::BitwiseAnd => "&",
            Self::BitwiseOr => "|",
            Self::BitwiseXor => "^",
            Self::ShiftLeft => "<<",
            Self::ShiftRight => ">>",
            Self::ShiftRightLogical => ">>>",
        }
    }
}

impl TryFrom<&str> for BinaryOperator {
    type Error = ();

    fn try_from(symbol: &str) -> Result<Self, Self::Error> {
        Ok(match symbol {
            "+" => Self::Add,
            "-" => Self::Subtract,
            "*" => Self::Multiply,
            "/" => Self::Divide,
            "%" => Self::Remainder,
            "//" => Self::FloorDivide,
            "%%" => Self::FloorRemainder,
            "**" => Self::Power,
            "==" => Self::Equal,
            "!=" => Self::NotEqual,
            "<" => Self::Less,
            "<=" => Self::LessEqual,
            ">" => Self::Greater,
            ">=" => Self::GreaterEqual,
            "&&" => Self::LogicalAnd,
            "||" => Self::LogicalOr,
            "&" => Self::BitwiseAnd,
            "|" => Self::BitwiseOr,
            "^" => Self::BitwiseXor,
            "<<" => Self::ShiftLeft,
            ">>" => Self::ShiftRight,
            ">>>" => Self::ShiftRightLogical,
            _ => return Err(()),
        })
    }
}

impl std::fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.symbol())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOperator {
    Negate,
    LogicalNot,
}

impl UnaryOperator {
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::Negate => "-",
            Self::LogicalNot => "!",
        }
    }
}

impl TryFrom<&str> for UnaryOperator {
    type Error = ();

    fn try_from(symbol: &str) -> Result<Self, Self::Error> {
        match symbol {
            "-" => Ok(Self::Negate),
            "!" => Ok(Self::LogicalNot),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.symbol())
    }
}

/// The operator identity shared by EGIR's pure nodes and SSA's `InstKind::Op`.
/// The call-target type is selected by the owning IR; EGIR uses an
/// uninhabited target so calls can only be represented by its call-site arena.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OpTag<R, C> {
    /// Signed integer literal (i8, i16, i32, i64).
    Int(String),
    /// Unsigned integer literal (u8, u16, u32, u64).
    Uint(String),
    Float(String),
    Bool(bool),
    Unit,
    Global(GlobalId),
    BinOp(BinaryOperator),
    UnaryOp(UnaryOperator),
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
    /// Operand-free reference to a module-level, addressable constant value.
    AddressableConstant(AddressableConstantId),
    DynamicExtract,
    Call(C),
    Intrinsic {
        id: builtins::BuiltinId,
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
}

/// Hashable variant of `ViewSource` for use inside an `OpTag`. Drops the
/// `ValueId` from `Inherited` — that parent is stored as an operand in the
/// containing `ValueKind::Pure` or `InstKind::Op`.
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

impl<R, C> OpTag<R, C> {
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

    pub fn try_map_resource<S, E>(self, map: &mut impl FnMut(R) -> Result<S, E>) -> Result<OpTag<S, C>, E> {
        Ok(match self {
            OpTag::Int(value) => OpTag::Int(value),
            OpTag::Uint(value) => OpTag::Uint(value),
            OpTag::Float(value) => OpTag::Float(value),
            OpTag::Bool(value) => OpTag::Bool(value),
            OpTag::Unit => OpTag::Unit,
            OpTag::Global(value) => OpTag::Global(value),
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
            OpTag::AddressableConstant(value) => OpTag::AddressableConstant(value),
            OpTag::DynamicExtract => OpTag::DynamicExtract,
            OpTag::Call(value) => OpTag::Call(value),
            OpTag::Intrinsic { id, overload_idx } => OpTag::Intrinsic { id, overload_idx },
            OpTag::StorageImageLoad(resource) => OpTag::StorageImageLoad(map(resource)?),
            OpTag::StorageImageStore(resource) => OpTag::StorageImageStore(map(resource)?),
            OpTag::StorageView(source) => OpTag::StorageView(source.try_map_resource(map)?),
            OpTag::ResourceLen(resource) => OpTag::ResourceLen(map(resource)?),
            OpTag::StorageViewLen => OpTag::StorageViewLen,
        })
    }

    pub fn try_map_call<D, E>(self, map: &mut impl FnMut(C) -> Result<D, E>) -> Result<OpTag<R, D>, E> {
        Ok(match self {
            OpTag::Int(value) => OpTag::Int(value),
            OpTag::Uint(value) => OpTag::Uint(value),
            OpTag::Float(value) => OpTag::Float(value),
            OpTag::Bool(value) => OpTag::Bool(value),
            OpTag::Unit => OpTag::Unit,
            OpTag::Global(value) => OpTag::Global(value),
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
            OpTag::AddressableConstant(value) => OpTag::AddressableConstant(value),
            OpTag::DynamicExtract => OpTag::DynamicExtract,
            OpTag::Call(value) => OpTag::Call(map(value)?),
            OpTag::Intrinsic { id, overload_idx } => OpTag::Intrinsic { id, overload_idx },
            OpTag::StorageImageLoad(resource) => OpTag::StorageImageLoad(resource),
            OpTag::StorageImageStore(resource) => OpTag::StorageImageStore(resource),
            OpTag::StorageView(source) => OpTag::StorageView(source),
            OpTag::ResourceLen(resource) => OpTag::ResourceLen(resource),
            OpTag::StorageViewLen => OpTag::StorageViewLen,
        })
    }

    pub fn map_call<D>(self, mut map: impl FnMut(C) -> D) -> OpTag<R, D> {
        self.try_map_call(&mut |call| Ok::<_, std::convert::Infallible>(map(call))).unwrap()
    }
}
