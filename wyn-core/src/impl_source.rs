// Implementation source for builtin functions and intrinsics
// Provides code generation implementations (SPIR-V opcodes, intrinsics)
// Types for functions are provided by modules or PolymorphicBuiltins

use crate::ast::{Type, TypeName};

/// Implementation strategy for a builtin function
///
/// Builtins are organized into three categories:
/// 1. Core primitives (PrimOp): Map fairly directly to backend operations
/// 2. Genuine intrinsics (Intrinsic): Require backend-specific lowering
/// 3. Linked SPIR-V (LinkedSpirv): External functions linked from pre-compiled SPIR-V
#[derive(Debug, Clone, PartialEq)]
pub enum BuiltinImpl {
    /// Core primitive operation: maps fairly directly to SPIR-V/backend ops
    /// Examples: f32.add, i32.mul, dot, matrix multiply
    PrimOp(PrimOp),

    /// Genuine intrinsic: needs backend-specific lowering, can't be written in language
    /// Examples: atomics, barriers, subgroup ops, uninit/poison
    Intrinsic(Intrinsic),

    /// Linked SPIR-V: function imported from external pre-compiled SPIR-V module
    /// The compiler generates an Import decoration and forward declaration,
    /// then the final binary is linked with spirv-link.
    /// Contains: linkage name (the string in OpDecorate LinkageAttributes)
    LinkedSpirv(String),
}

/// Core primitive operations that map fairly directly to SPIR-V/backend ops
#[derive(Debug, Clone, PartialEq)]
pub enum PrimOp {
    // GLSL.std.450 extended instructions
    GlslExt(u32),

    // Core SPIR-V operations
    Dot,
    OuterProduct,
    MatrixTimesMatrix,
    MatrixTimesVector,
    VectorTimesMatrix,
    VectorTimesScalar,
    MatrixTimesScalar,
    // Arithmetic ops
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
    FMod,
    IAdd,
    ISub,
    IMul,
    SDiv,
    UDiv,
    SRem,
    SMod,
    UMod,
    // Comparison ops
    FOrdEqual,
    FOrdNotEqual,
    FOrdLessThan,
    FOrdGreaterThan,
    FOrdLessThanEqual,
    FOrdGreaterThanEqual,
    IEqual,
    INotEqual,
    SLessThan,
    ULessThan,
    SGreaterThan,
    UGreaterThan,
    SLessThanEqual,
    ULessThanEqual,
    SGreaterThanEqual,
    UGreaterThanEqual,
    // Bitwise ops
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Not,
    ShiftLeftLogical,
    ShiftRightArithmetic,
    ShiftRightLogical,
    // OpIsNan / OpIsInf.
    IsNan,
    IsInf,
    // Type conversions
    // Float to signed int
    FPToSI,
    // Float to unsigned int
    FPToUI,
    // Signed int to float
    SIToFP,
    // Unsigned int to float
    UIToFP,
    // Float precision conversion
    FPConvert,
    // Signed extension
    SConvert,
    // Unsigned/zero extension
    UConvert,
    // Bitcast (reinterpret bits)
    Bitcast,
}

/// Genuine intrinsics that need backend-specific lowering
/// These cannot be written in the language itself
#[derive(Debug, Clone, PartialEq)]
pub enum Intrinsic {
    /// Placeholder for future implementations (will be desugared or moved)
    Placeholder,
    /// Uninitialized/poison value for allocation bootstrapping
    /// SAFETY: Must be fully overwritten before being read
    Uninit,
    /// Functional array update: immutable copy-with-update. Backend must
    /// preserve the source array (emit copy + patch). User-surface default.
    /// Note: Could be moved to prelude once array comprehensions or fold is implemented
    ArrayWith,
    /// In-place variant of `ArrayWith`. Caller guarantees the source array
    /// is dead after this operation (loop-carried phi, or alias-checker
    /// proved released). Backend may mutate the source buffer directly
    /// instead of copying.
    ArrayWithInPlace,
    /// `_w_intrinsic_length(arr) -> i32` — array size, distinct from
    /// vector `magnitude` (which lowers through GlslExt(66)). Can't be a
    /// PrimOp because GLSL uses method-call syntax `arr.length()` and
    /// SPIR-V uses OpArrayLength with variant-specific handling.
    Length,
}

/// Look up a vec component field type. `type_name` is a name like
/// `vec3f32`; `field_name` is one of `x`/`y`/`z`/`w` (validated against
/// the vec size). Returns the element type if the field is in range.
pub fn vec_field_type(type_name: &str, field_name: &str) -> Option<Type> {
    if !type_name.starts_with("vec") {
        return None;
    }
    let size = type_name.chars().nth(3)?.to_digit(10)? as usize;
    let valid = matches!(
        (size, field_name),
        (2, "x" | "y") | (3, "x" | "y" | "z") | (4, "x" | "y" | "z" | "w")
    );
    if !valid {
        return None;
    }
    let elem_type_str = &type_name[4..];
    let elem_type_name = match elem_type_str {
        "f32" => TypeName::Float(32),
        "f64" => TypeName::Float(64),
        "i32" => TypeName::Int(32),
        "i64" => TypeName::Int(64),
        "u32" => TypeName::UInt(32),
        "u64" => TypeName::UInt(64),
        "bool" => TypeName::Bool,
        other => TypeName::Named(other.to_string()),
    };
    Some(Type::Constructed(elem_type_name, vec![]))
}
