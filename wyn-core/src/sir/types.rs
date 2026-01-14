//! SIR type definitions.
//!
//! SIR reuses `polytype::Type<TypeName>` for expression types, but introduces
//! its own symbolic size representation for array dimensions.

use std::fmt;

/// Unique identifier for a symbolic size variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SizeVar(pub u32);

impl From<u32> for SizeVar {
    fn from(id: u32) -> Self {
        SizeVar(id)
    }
}

impl fmt::Display for SizeVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// Symbolic array size expression.
///
/// Sizes can be concrete constants, symbolic variables, or arithmetic
/// combinations. This allows tracking size relationships through
/// transformations (e.g., map preserves input size).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Size {
    /// Concrete size known at compile time.
    Const(u64),
    /// Symbolic size variable (runtime-determined).
    Sym(SizeVar),
    /// Sum of two sizes.
    Add(Box<Size>, Box<Size>),
    /// Product of two sizes.
    Mul(Box<Size>, Box<Size>),
}

impl Size {
    /// Create a constant size.
    pub fn constant(n: u64) -> Self {
        Size::Const(n)
    }

    /// Create a symbolic size.
    pub fn symbolic(var: SizeVar) -> Self {
        Size::Sym(var)
    }

    /// Add two sizes.
    pub fn add(self, other: Size) -> Self {
        match (&self, &other) {
            (Size::Const(a), Size::Const(b)) => Size::Const(a + b),
            _ => Size::Add(Box::new(self), Box::new(other)),
        }
    }

    /// Multiply two sizes.
    pub fn mul(self, other: Size) -> Self {
        match (&self, &other) {
            (Size::Const(a), Size::Const(b)) => Size::Const(a * b),
            (Size::Const(0), _) | (_, Size::Const(0)) => Size::Const(0),
            (Size::Const(1), _) => other,
            (_, Size::Const(1)) => self,
            _ => Size::Mul(Box::new(self), Box::new(other)),
        }
    }

    /// Try to evaluate to a concrete value.
    pub fn as_const(&self) -> Option<u64> {
        match self {
            Size::Const(n) => Some(*n),
            _ => None,
        }
    }

    /// Check if this size is statically known.
    pub fn is_const(&self) -> bool {
        matches!(self, Size::Const(_))
    }
}

impl fmt::Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Size::Const(n) => write!(f, "{}", n),
            Size::Sym(v) => write!(f, "{}", v),
            Size::Add(a, b) => write!(f, "({} + {})", a, b),
            Size::Mul(a, b) => write!(f, "({} * {})", a, b),
        }
    }
}

/// Scalar type for elements of arrays in SOACs.
///
/// This is a simplified type used where we need just the element type
/// without the full polytype machinery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarTy {
    Bool,
    I32,
    I64,
    U32,
    U64,
    F32,
    F64,
}

impl fmt::Display for ScalarTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarTy::Bool => write!(f, "bool"),
            ScalarTy::I32 => write!(f, "i32"),
            ScalarTy::I64 => write!(f, "i64"),
            ScalarTy::U32 => write!(f, "u32"),
            ScalarTy::U64 => write!(f, "u64"),
            ScalarTy::F32 => write!(f, "f32"),
            ScalarTy::F64 => write!(f, "f64"),
        }
    }
}

/// Information about associativity/commutativity of a reduction operator.
///
/// This is important for determining which parallel reduction strategies
/// are valid (e.g., tree reduction requires associativity).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssocInfo {
    /// The operator is associative: (a op b) op c = a op (b op c)
    pub is_associative: bool,
    /// The operator is commutative: a op b = b op a
    pub is_commutative: bool,
}

impl AssocInfo {
    /// Unknown associativity (conservative).
    pub fn unknown() -> Self {
        AssocInfo {
            is_associative: false,
            is_commutative: false,
        }
    }

    /// Fully associative and commutative (e.g., +, *, min, max).
    pub fn fully_assoc() -> Self {
        AssocInfo {
            is_associative: true,
            is_commutative: true,
        }
    }

    /// Associative but not commutative (e.g., matrix multiply).
    pub fn assoc_only() -> Self {
        AssocInfo {
            is_associative: true,
            is_commutative: false,
        }
    }
}

impl Default for AssocInfo {
    fn default() -> Self {
        Self::unknown()
    }
}
