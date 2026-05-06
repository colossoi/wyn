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
    /// Builtin has no backend lowering through this enum. Used for
    /// HOFs (handled by the SOAC infrastructure) and compiler-internal
    /// builtins (emitted/consumed before backend dispatch). Backend
    /// dispatch reaching this is a bug.
    NotLowered,
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
    /// `_w_intrinsic_slice(arr, start, end)` — sub-array. Three cases at
    /// the backend: view→view (new handle), view→composite (materialize
    /// elements), composite→composite (also materialize). Result variant
    /// is read from the SSA result type.
    Slice,
    /// `_w_intrinsic_storage_len(set, binding) -> i32` — runtime length
    /// of a storage buffer at the given (set, binding) coordinates. Both
    /// args are constant `u32` literals. SPIR-V lowers via `OpArrayLength`
    /// on the runtime array member of the storage buffer struct.
    StorageLen,
    /// `_w_intrinsic_thread_id() -> u32` — flattened compute-shader
    /// thread index. SPIR-V loads `GlobalInvocationId.x`.
    ThreadId,
    /// GLSL.std.450 extended instruction with operand splatting. For
    /// each position in `splat_args`, if that operand is a scalar but
    /// the result is a vec, splat it to result-vec width before emitting
    /// `OpExtInst`. Covers vec overloads of `mix`, `clamp`, `smoothstep`
    /// — where the scalar overload is a plain `PrimOp(GlslExt(N))` but
    /// the vec overload's mixed scalar args need splatting because
    /// `OpExtInst` requires operand types to match the result type.
    ExtInstSplat {
        ext: u32,
        splat_args: &'static [usize],
    },
}
