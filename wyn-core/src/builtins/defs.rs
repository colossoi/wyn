use crate::builtins::catalog::{BuiltinDefRaw, BuiltinKind, BuiltinOverload, Purity};
use crate::builtins::lowering::BuiltinLowering;
use crate::builtins::scheme::{vec_binary_same, vec_ternary_same, vec_unary_same};
use crate::impl_source::PrimOp;

// ---------------------------------------------------------------------------
// `vec.*` module ops
// ---------------------------------------------------------------------------
//
// Vector-only counterparts to per-type scalar ops. Each is typed
// `∀n a. vec<n,a> -> ...` so passing a scalar fails at type-check.
// Codegen routes through the same GLSL.std.450 op as the scalar form
// (the SPIR-V op handles both shapes natively).

macro_rules! vec_unary {
    ($name:literal, $glsl_ext:expr) => {
        BuiltinDefRaw {
            surface_name: $name,
            internal_name: None,
            kind: BuiltinKind::ModuleBuiltin,
            purity: Purity::Pure,
            overloads: &[BuiltinOverload {
                scheme: vec_unary_same,
                lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt($glsl_ext)),
            }],
        }
    };
}

macro_rules! vec_binary {
    ($name:literal, $lowering:expr) => {
        BuiltinDefRaw {
            surface_name: $name,
            internal_name: None,
            kind: BuiltinKind::ModuleBuiltin,
            purity: Purity::Pure,
            overloads: &[BuiltinOverload {
                scheme: vec_binary_same,
                lowering: $lowering,
            }],
        }
    };
}

macro_rules! vec_ternary {
    ($name:literal, $glsl_ext:expr) => {
        BuiltinDefRaw {
            surface_name: $name,
            internal_name: None,
            kind: BuiltinKind::ModuleBuiltin,
            purity: Purity::Pure,
            overloads: &[BuiltinOverload {
                scheme: vec_ternary_same,
                lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt($glsl_ext)),
            }],
        }
    };
}

pub static ALL_BUILTINS: &[BuiltinDefRaw] = &[
    // Trig
    vec_unary!("vec.sin", 13),
    vec_unary!("vec.cos", 14),
    vec_unary!("vec.tan", 15),
    vec_unary!("vec.asin", 16),
    vec_unary!("vec.acos", 17),
    vec_unary!("vec.atan", 18),
    // Hyperbolic
    vec_unary!("vec.sinh", 19),
    vec_unary!("vec.cosh", 20),
    vec_unary!("vec.tanh", 21),
    vec_unary!("vec.asinh", 22),
    vec_unary!("vec.acosh", 23),
    vec_unary!("vec.atanh", 24),
    // Roots / exp / log
    vec_unary!("vec.sqrt", 31),
    vec_unary!("vec.rsqrt", 32),
    vec_unary!("vec.exp", 27),
    vec_unary!("vec.exp2", 29),
    vec_unary!("vec.log", 28),
    vec_unary!("vec.log2", 30),
    // Rounding / sign
    vec_unary!("vec.floor", 8),
    vec_unary!("vec.ceil", 9),
    vec_unary!("vec.round", 1),
    vec_unary!("vec.trunc", 3),
    vec_unary!("vec.fract", 10),
    vec_unary!("vec.abs", 4),
    vec_unary!("vec.sign", 6),
    // Angle conversion
    vec_unary!("vec.radians", 11),
    vec_unary!("vec.degrees", 12),
    // Binary
    vec_binary!("vec.pow", BuiltinLowering::PrimOp(PrimOp::GlslExt(26))),
    vec_binary!("vec.atan2", BuiltinLowering::PrimOp(PrimOp::GlslExt(25))),
    vec_binary!("vec.mod", BuiltinLowering::PrimOp(PrimOp::FMod)),
    vec_binary!("vec.min", BuiltinLowering::PrimOp(PrimOp::GlslExt(37))),
    vec_binary!("vec.max", BuiltinLowering::PrimOp(PrimOp::GlslExt(40))),
    // Ternary
    vec_ternary!("vec.clamp", 43),
    vec_ternary!("vec.mix", 46),
    vec_ternary!("vec.smoothstep", 49),
];
