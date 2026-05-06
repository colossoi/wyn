//! Backend lowering shapes for builtins.
//!
//! `BuiltinLowering` is what the catalog stores per overload and what
//! the backends consume. Three shapes:
//! - `PrimOp` — direct mapping to a SPIR-V op or GLSL.std.450
//!   extended instruction.
//! - `Intrinsic` — needs backend-specific lowering (e.g. `array_with`,
//!   `length`, `uninit`).
//! - `LinkedSpirv` — function imported from a pre-compiled SPIR-V
//!   module, the inner string is the `OpDecorate LinkageAttributes`
//!   linkage name.

/// How a builtin lowers to backend operations.
#[derive(Debug, Clone)]
pub enum BuiltinLowering {
    /// Direct mapping to a `PrimOp` (GLSL.std.450 extended instruction
    /// or core SPIR-V op).
    PrimOp(PrimOp),
    /// Genuine intrinsic that needs backend-specific lowering.
    Intrinsic(Intrinsic),
    /// Function imported from a pre-compiled SPIR-V module — the string
    /// is the linkage name in `OpDecorate LinkageAttributes`.
    LinkedSpirv(&'static str),
}

/// Core primitive operations that map fairly directly to SPIR-V/backend ops.
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

/// Genuine intrinsics that need backend-specific lowering.
/// These cannot be written in the language itself.
#[derive(Debug, Clone, PartialEq)]
pub enum Intrinsic {
    /// Placeholder for future implementations (will be desugared or moved)
    Placeholder,
    /// Uninitialized/poison value for allocation bootstrapping.
    /// SAFETY: Must be fully overwritten before being read.
    Uninit,
    /// Functional array update: immutable copy-with-update. Backend must
    /// preserve the source array (emit copy + patch). User-surface default.
    ArrayWith,
    /// In-place variant of `ArrayWith`. Caller guarantees the source array
    /// is dead after this operation (loop-carried phi, or alias-checker
    /// proved released). Backend may mutate the source buffer directly
    /// instead of copying.
    ArrayWithInPlace,
    /// `_w_intrinsic_length(arr) -> i32` — array size, distinct from
    /// vector `magnitude` (which lowers through `GlslExt(66)`). Can't be a
    /// `PrimOp` because GLSL uses method-call syntax `arr.length()` and
    /// SPIR-V uses `OpArrayLength` with variant-specific handling.
    Length,
}
