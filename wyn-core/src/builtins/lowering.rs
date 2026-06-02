//! Backend lowering shapes for builtins.
//!
//! `BuiltinLowering` is what the catalog stores per overload and what
//! the backends consume. Five variants:
//! - `PrimOp(PrimOp)` — direct mapping to a SPIR-V op or GLSL.std.450
//!   extended instruction.
//! - `LinkedSpirv(&str)` — function imported from a pre-compiled
//!   SPIR-V module; the string is the linkage name.
//! - `ExtInstSplat { ext, splat_args }` — GLSL.std.450 ext-inst with
//!   per-call operand splatting. Data-bearing because the splat
//!   metadata varies per catalog entry.
//! - `ByBuiltinId` — "ask the backend for this entry by its
//!   `BuiltinId`." Backends maintain explicit handlers keyed off
//!   `catalog::known_ids()`; this avoids a parallel `Intrinsic` enum
//!   that would shadow `BuiltinId` for the same set of entries.
//! - `NotLowered` — sentinel; reaching this at backend dispatch is a
//!   bug (HOFs, compiler-internal intrinsics consumed before backend
//!   dispatch).

/// How a builtin lowers to backend operations.
#[derive(Debug, Clone)]
pub enum BuiltinLowering {
    /// Direct mapping to a `PrimOp` (GLSL.std.450 extended instruction
    /// or core SPIR-V op).
    PrimOp(PrimOp),
    /// Function imported from a pre-compiled SPIR-V module — the string
    /// is the linkage name in `OpDecorate LinkageAttributes`.
    LinkedSpirv(&'static str),
    /// Builtin has no backend lowering through this enum. Used for
    /// HOFs (handled by the SOAC infrastructure) and compiler-internal
    /// builtins (emitted/consumed before backend dispatch). Backend
    /// dispatch reaching this is a bug.
    NotLowered,
    /// Dispatch by `BuiltinId`. The catalog's `known_ids()` exposes
    /// cached BuiltinIds for each well-known entry; backends compare
    /// the dispatch site's id against those constants and route to a
    /// per-id handler. This replaces a previous `Intrinsic` enum that
    /// duplicated `BuiltinId`-style identity.
    ByBuiltinId,
    /// GLSL.std.450 extended instruction with operand splatting. For
    /// each position in `splat_args`, if that operand is a scalar but
    /// the result is a vec, splat it to result-vec width before
    /// emitting `OpExtInst`. Covers vec overloads of `mix`, `clamp`,
    /// `smoothstep`. Inline data because each catalog entry has its
    /// own ext index and splat positions.
    ExtInstSplat {
        ext: u32,
        splat_args: &'static [usize],
    },
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
    /// Integer exponentiation by repeated squaring. SPIR-V has no
    /// native integer-pow instruction (GLSL.std.450 `Pow` is float
    /// only), so the backend emits a `OpFunctionCall` to a
    /// compiler-generated helper (`spirv::pow::emit_int_pow_helpers`).
    /// `signed = true` uses `OpSGreaterThan` / `OpShiftRightArithmetic`
    /// for the loop's exit and shift; `false` uses the unsigned ops.
    /// 32-bit only for now; other widths can be added by extending
    /// this variant with a width field.
    IntPow {
        signed: bool,
    },
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
    // Screen-space derivatives. Fragment-stage only; SPIR-V's base
    // Shader capability covers the implicit (`DPdx`/`DPdy`/`Fwidth`)
    // forms used here. Fine/coarse variants would require
    // `DerivativeControl` and are not exposed yet.
    DPdx,
    DPdy,
    Fwidth,
}
