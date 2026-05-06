use crate::builtins::catalog::{BuiltinDefRaw, BuiltinKind, BuiltinOverload, Purity};
use crate::builtins::lowering::BuiltinLowering;
use crate::builtins::lowering::PrimOp;
use crate::builtins::names::{
    INTRINSIC_ABS, INTRINSIC_ARRAY_WITH, INTRINSIC_ARRAY_WITH_INPLACE, INTRINSIC_CEIL, INTRINSIC_CLAMP,
    INTRINSIC_COS, INTRINSIC_CROSS, INTRINSIC_DETERMINANT, INTRINSIC_DISTANCE, INTRINSIC_DOT,
    INTRINSIC_FLOOR, INTRINSIC_FRACT, INTRINSIC_INVERSE, INTRINSIC_LENGTH, INTRINSIC_MAGNITUDE,
    INTRINSIC_MIX, INTRINSIC_NORMALIZE, INTRINSIC_OUTER, INTRINSIC_REFLECT, INTRINSIC_REFRACT,
    INTRINSIC_SLICE, INTRINSIC_SMOOTHSTEP, INTRINSIC_STORAGE_INDEX, INTRINSIC_STORAGE_LEN,
    INTRINSIC_STORAGE_STORE, INTRINSIC_THREAD_ID, INTRINSIC_UNINIT,
};
use crate::builtins::scheme::{
    array_to_i32, mat_square_to_mat, mat_square_to_scalar, mat_x_mat, mat_x_vec, scalar_unary, unit_to_t,
    vec_binary_same, vec_binary_to_scalar, vec_clamp_scalar_lohi, vec_mix_scalar_interp,
    vec_smoothstep_scalar_edges, vec_ternary_same, vec_to_scalar, vec_unary_same, vec_vec_outer,
    vec_vec_scalar_to_vec, vec_x_mat, vec3f32_binary,
};

// ---------------------------------------------------------------------------
// `vec.*` module ops — vector-only counterparts to per-type scalar ops
// ---------------------------------------------------------------------------

// All catalog-entry constructors are macros (not functions) because
// each entry needs `&'static` slice literals (`&[surface]`,
// `&[BuiltinOverload {..}]`) and Rust's const promotion only kicks in
// at the static-initializer site — not inside a `const fn` body.

macro_rules! vec_module_op {
    ($name:literal, $scheme:expr, $lowering:expr) => {
        BuiltinDefRaw {
            surface_name: $name,
            intrinsic_source_names: &[$name],
            impl_source_names: &[$name],
            kind: BuiltinKind::ModuleBuiltin,
            purity: Purity::Pure,
            overloads: &[BuiltinOverload {
                scheme: Some($scheme),
                lowering: $lowering,
            }],
        }
    };
}

// Single-overload UserVisible polymorphic intrinsic with surface →
// internal name renaming.
macro_rules! polymorphic_intrinsic {
    ($surface:literal, $internal:expr, $scheme:expr, $lowering:expr) => {
        BuiltinDefRaw {
            surface_name: $surface,
            intrinsic_source_names: &[$surface],
            impl_source_names: &[$internal],
            kind: BuiltinKind::UserVisible,
            purity: Purity::Pure,
            overloads: &[BuiltinOverload {
                scheme: Some($scheme),
                lowering: $lowering,
            }],
        }
    };
}

// HOF / SOAC intrinsic: published scheme, no backend lowering.
macro_rules! hof_intrinsic {
    ($name:expr, $scheme:expr) => {
        hof_intrinsic!($name, $scheme, Purity::Pure)
    };
    ($name:expr, $scheme:expr, $purity:expr) => {
        BuiltinDefRaw {
            surface_name: $name,
            intrinsic_source_names: &[$name],
            impl_source_names: &[],
            kind: BuiltinKind::InternalIntrinsic,
            purity: $purity,
            overloads: &[BuiltinOverload {
                scheme: Some($scheme),
                lowering: BuiltinLowering::NotLowered,
            }],
        }
    };
}

// Compiler-internal intrinsic: emitted by the codegen pipeline. The
// 2-arg form leaves lowering as `NotLowered` (transitional — backend
// dispatches by name). The 3-arg form takes a typed
// `BuiltinLowering::Intrinsic(_)` variant so backends can dispatch
// structurally.
macro_rules! compiler_internal {
    ($name:expr) => {
        compiler_internal!($name, Purity::Pure)
    };
    ($name:expr, $purity:expr) => {
        BuiltinDefRaw {
            surface_name: $name,
            intrinsic_source_names: &[],
            impl_source_names: &[],
            kind: BuiltinKind::InternalIntrinsic,
            purity: $purity,
            overloads: &[BuiltinOverload {
                scheme: None,
                lowering: BuiltinLowering::NotLowered,
            }],
        }
    };
    ($name:expr, $purity:expr, $intrinsic:expr) => {
        BuiltinDefRaw {
            surface_name: $name,
            intrinsic_source_names: &[],
            impl_source_names: &[],
            kind: BuiltinKind::InternalIntrinsic,
            purity: $purity,
            overloads: &[BuiltinOverload {
                scheme: None,
                lowering: BuiltinLowering::Intrinsic($intrinsic),
            }],
        }
    };
}

pub fn all_builtins() -> Vec<BuiltinDefRaw> {
    let mut defs: Vec<BuiltinDefRaw> = STATIC_BUILTINS.to_vec();
    defs.extend(generate_per_type_ops());
    defs
}

static STATIC_BUILTINS: &[BuiltinDefRaw] = &[
    // ---- vec.* trig ----
    vec_module_op!(
        "vec.sin",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(13))
    ),
    vec_module_op!(
        "vec.cos",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(14))
    ),
    vec_module_op!(
        "vec.tan",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(15))
    ),
    vec_module_op!(
        "vec.asin",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(16))
    ),
    vec_module_op!(
        "vec.acos",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(17))
    ),
    vec_module_op!(
        "vec.atan",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(18))
    ),
    // ---- vec.* hyperbolic ----
    vec_module_op!(
        "vec.sinh",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(19))
    ),
    vec_module_op!(
        "vec.cosh",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(20))
    ),
    vec_module_op!(
        "vec.tanh",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(21))
    ),
    vec_module_op!(
        "vec.asinh",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(22))
    ),
    vec_module_op!(
        "vec.acosh",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(23))
    ),
    vec_module_op!(
        "vec.atanh",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(24))
    ),
    // ---- vec.* exp/log/roots ----
    vec_module_op!(
        "vec.sqrt",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(31))
    ),
    vec_module_op!(
        "vec.rsqrt",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(32))
    ),
    vec_module_op!(
        "vec.exp",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(27))
    ),
    vec_module_op!(
        "vec.exp2",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(29))
    ),
    vec_module_op!(
        "vec.log",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(28))
    ),
    vec_module_op!(
        "vec.log2",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(30))
    ),
    // ---- vec.* rounding/sign ----
    vec_module_op!(
        "vec.floor",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(8))
    ),
    vec_module_op!(
        "vec.ceil",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(9))
    ),
    vec_module_op!(
        "vec.round",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(1))
    ),
    vec_module_op!(
        "vec.trunc",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(3))
    ),
    vec_module_op!(
        "vec.fract",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(10))
    ),
    vec_module_op!(
        "vec.abs",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(4))
    ),
    vec_module_op!(
        "vec.sign",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(6))
    ),
    // ---- vec.* angle conversion ----
    vec_module_op!(
        "vec.radians",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(11))
    ),
    vec_module_op!(
        "vec.degrees",
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(12))
    ),
    // ---- vec.* binary ----
    vec_module_op!(
        "vec.pow",
        vec_binary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(26))
    ),
    vec_module_op!(
        "vec.atan2",
        vec_binary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(25))
    ),
    vec_module_op!("vec.mod", vec_binary_same, BuiltinLowering::PrimOp(PrimOp::FMod)),
    vec_module_op!(
        "vec.min",
        vec_binary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(37))
    ),
    vec_module_op!(
        "vec.max",
        vec_binary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(40))
    ),
    // ---- vec.* ternary ----
    vec_module_op!(
        "vec.clamp",
        vec_ternary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(43))
    ),
    vec_module_op!(
        "vec.mix",
        vec_ternary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(46))
    ),
    vec_module_op!(
        "vec.smoothstep",
        vec_ternary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(49))
    ),
    // -----------------------------------------------------------------------
    // Polymorphic intrinsics — surface name (e.g. "magnitude") published
    // for type-check-time resolution; `_w_intrinsic_*` form (e.g.
    // "_w_intrinsic_magnitude") published for backend lowering. TLC's
    // INTRINSIC_RENAMES translates user calls to the internal name
    // before backend dispatch.
    // -----------------------------------------------------------------------
    polymorphic_intrinsic!(
        "magnitude",
        INTRINSIC_MAGNITUDE,
        vec_to_scalar,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(66))
    ),
    polymorphic_intrinsic!(
        "normalize",
        INTRINSIC_NORMALIZE,
        vec_unary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(69))
    ),
    polymorphic_intrinsic!(
        "dot",
        INTRINSIC_DOT,
        vec_binary_to_scalar,
        BuiltinLowering::PrimOp(PrimOp::Dot)
    ),
    polymorphic_intrinsic!(
        "cross",
        INTRINSIC_CROSS,
        vec3f32_binary,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(68))
    ),
    polymorphic_intrinsic!(
        "distance",
        INTRINSIC_DISTANCE,
        vec_binary_to_scalar,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(67))
    ),
    polymorphic_intrinsic!(
        "reflect",
        INTRINSIC_REFLECT,
        vec_binary_same,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(71))
    ),
    polymorphic_intrinsic!(
        "refract",
        INTRINSIC_REFRACT,
        vec_vec_scalar_to_vec,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(72))
    ),
    polymorphic_intrinsic!(
        "outer",
        INTRINSIC_OUTER,
        vec_vec_outer,
        BuiltinLowering::PrimOp(PrimOp::OuterProduct)
    ),
    polymorphic_intrinsic!(
        "determinant",
        INTRINSIC_DETERMINANT,
        mat_square_to_scalar,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(33))
    ),
    polymorphic_intrinsic!(
        "inverse",
        INTRINSIC_INVERSE,
        mat_square_to_mat,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(34))
    ),
    BuiltinDefRaw {
        surface_name: "mul",
        intrinsic_source_names: &["mul"],
        impl_source_names: &[],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[
            BuiltinOverload {
                scheme: Some(mat_x_mat),
                lowering: BuiltinLowering::PrimOp(PrimOp::MatrixTimesMatrix),
            },
            BuiltinOverload {
                scheme: Some(mat_x_vec),
                lowering: BuiltinLowering::PrimOp(PrimOp::MatrixTimesVector),
            },
            BuiltinOverload {
                scheme: Some(vec_x_mat),
                lowering: BuiltinLowering::PrimOp(PrimOp::VectorTimesMatrix),
            },
        ],
    },
    // ---- Scalar polymorphic math ----
    polymorphic_intrinsic!(
        "abs",
        INTRINSIC_ABS,
        scalar_unary,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(4))
    ),
    // `sign` — UserVisible polymorphic, surface name only (no
    // `_w_intrinsic_sign` alias is needed by anything today).
    BuiltinDefRaw {
        surface_name: "sign",
        intrinsic_source_names: &["sign"],
        impl_source_names: &[],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: Some(scalar_unary),
            lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(6)),
        }],
    },
    polymorphic_intrinsic!(
        "floor",
        INTRINSIC_FLOOR,
        scalar_unary,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(8))
    ),
    polymorphic_intrinsic!(
        "ceil",
        INTRINSIC_CEIL,
        scalar_unary,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(9))
    ),
    polymorphic_intrinsic!(
        "fract",
        INTRINSIC_FRACT,
        scalar_unary,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(10))
    ),
    polymorphic_intrinsic!(
        "min",
        "_w_intrinsic_min",
        crate::builtins::scheme::scalar_binary,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(37))
    ),
    polymorphic_intrinsic!(
        "max",
        "_w_intrinsic_max",
        crate::builtins::scheme::scalar_binary,
        BuiltinLowering::PrimOp(PrimOp::GlslExt(40))
    ),
    BuiltinDefRaw {
        surface_name: "clamp",
        intrinsic_source_names: &["clamp"],
        impl_source_names: &[INTRINSIC_CLAMP],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[
            BuiltinOverload {
                scheme: Some(crate::builtins::scheme::scalar_ternary),
                lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(43)),
            },
            // `clamp(vec, scalar, scalar)` — splat lo and hi to vec
            // width before `OpExtInst FClamp`.
            BuiltinOverload {
                scheme: Some(vec_clamp_scalar_lohi),
                lowering: BuiltinLowering::Intrinsic(crate::builtins::lowering::Intrinsic::ExtInstSplat {
                    ext: 43,
                    splat_args: &[1, 2],
                }),
            },
        ],
    },
    BuiltinDefRaw {
        surface_name: "mix",
        intrinsic_source_names: &["mix"],
        impl_source_names: &[INTRINSIC_MIX],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[
            BuiltinOverload {
                scheme: Some(crate::builtins::scheme::scalar_ternary),
                lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(46)),
            },
            // `mix(vec, vec, scalar)` — splat the scalar `t` to vec
            // width before `OpExtInst FMix`, which requires every
            // operand to match the result type.
            BuiltinOverload {
                scheme: Some(vec_mix_scalar_interp),
                lowering: BuiltinLowering::Intrinsic(crate::builtins::lowering::Intrinsic::ExtInstSplat {
                    ext: 46,
                    splat_args: &[2],
                }),
            },
        ],
    },
    BuiltinDefRaw {
        surface_name: "smoothstep",
        intrinsic_source_names: &["smoothstep"],
        impl_source_names: &[INTRINSIC_SMOOTHSTEP],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[
            BuiltinOverload {
                scheme: Some(crate::builtins::scheme::scalar_ternary),
                lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(49)),
            },
            // `smoothstep(scalar, scalar, vec)` — splat edge0/edge1 to
            // vec width before `OpExtInst SmoothStep`.
            BuiltinOverload {
                scheme: Some(vec_smoothstep_scalar_edges),
                lowering: BuiltinLowering::Intrinsic(crate::builtins::lowering::Intrinsic::ExtInstSplat {
                    ext: 49,
                    splat_args: &[0, 1],
                }),
            },
        ],
    },
    // ---- Internal intrinsics with real backend lowerings ----
    // `length` is user-callable as `length(arr)` — surface name distinct
    // from the internal `_w_intrinsic_length` so NameResolution can
    // classify the surface form.
    BuiltinDefRaw {
        surface_name: "length",
        intrinsic_source_names: &["length"],
        impl_source_names: &[INTRINSIC_LENGTH],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: Some(array_to_i32),
            lowering: BuiltinLowering::Intrinsic(crate::builtins::lowering::Intrinsic::Length),
        }],
    },
    BuiltinDefRaw {
        surface_name: INTRINSIC_UNINIT,
        intrinsic_source_names: &[INTRINSIC_UNINIT],
        impl_source_names: &[INTRINSIC_UNINIT],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: Some(unit_to_t),
            lowering: BuiltinLowering::Intrinsic(crate::builtins::lowering::Intrinsic::Uninit),
        }],
    },
    BuiltinDefRaw {
        surface_name: INTRINSIC_ARRAY_WITH,
        intrinsic_source_names: &[INTRINSIC_ARRAY_WITH],
        impl_source_names: &[INTRINSIC_ARRAY_WITH],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: Some(crate::builtins::scheme::array_index_value_to_array),
            lowering: BuiltinLowering::Intrinsic(crate::builtins::lowering::Intrinsic::ArrayWith),
        }],
    },
    BuiltinDefRaw {
        surface_name: INTRINSIC_ARRAY_WITH_INPLACE,
        intrinsic_source_names: &[],
        impl_source_names: &[INTRINSIC_ARRAY_WITH_INPLACE],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Effectful,
        overloads: &[BuiltinOverload {
            scheme: Some(crate::builtins::scheme::array_index_value_to_array),
            lowering: BuiltinLowering::Intrinsic(crate::builtins::lowering::Intrinsic::ArrayWithInPlace),
        }],
    },
    // ---- HOF / SOAC intrinsics: scheme only, lowered earlier in the pipeline ----
    hof_intrinsic!(
        "_w_intrinsic_replicate",
        crate::builtins::scheme::replicate_scheme,
        Purity::Pure
    ),
    hof_intrinsic!(
        "_w_intrinsic_reduce",
        crate::builtins::scheme::reduce_scheme,
        Purity::Pure
    ),
    hof_intrinsic!(
        "_w_intrinsic_filter",
        crate::builtins::scheme::filter_scheme,
        Purity::Pure
    ),
    hof_intrinsic!(
        "_w_intrinsic_scan",
        crate::builtins::scheme::scan_scheme,
        Purity::Pure
    ),
    hof_intrinsic!(
        "_w_intrinsic_map",
        crate::builtins::scheme::map_scheme,
        Purity::Pure
    ),
    hof_intrinsic!(
        "_w_intrinsic_map_into",
        crate::builtins::scheme::map_into_scheme,
        Purity::Effectful
    ),
    hof_intrinsic!(
        "_w_intrinsic_scatter",
        crate::builtins::scheme::scatter_scheme,
        Purity::Pure
    ),
    hof_intrinsic!(
        "_w_intrinsic_hist_1d",
        crate::builtins::scheme::hist_1d_scheme,
        Purity::Pure
    ),
    hof_intrinsic!(
        "_w_intrinsic_rotr32",
        crate::builtins::scheme::u32_binary,
        Purity::Pure
    ),
    // ---- Compiler-internal intrinsics: emitted by the codegen pipeline,
    // not user-facing. They have no published scheme (the compiler
    // synthesises type-correct calls) and lowering is special-cased per
    // backend. Their presence in the catalog gives every PureOp::Intrinsic
    // emission a `BuiltinId`. ----
    compiler_internal!(
        INTRINSIC_STORAGE_LEN,
        Purity::Pure,
        crate::builtins::lowering::Intrinsic::StorageLen
    ),
    compiler_internal!(INTRINSIC_STORAGE_INDEX, Purity::Pure),
    compiler_internal!(INTRINSIC_STORAGE_STORE, Purity::Effectful),
    compiler_internal!(
        INTRINSIC_SLICE,
        Purity::Pure,
        crate::builtins::lowering::Intrinsic::Slice
    ),
    compiler_internal!(
        INTRINSIC_THREAD_ID,
        Purity::Pure,
        crate::builtins::lowering::Intrinsic::ThreadId
    ),
    compiler_internal!(INTRINSIC_COS, Purity::Pure),
];

// ---------------------------------------------------------------------------
// Per-type operator generation
// ---------------------------------------------------------------------------
//
// Per-type ops (`f32.+`, `i32.<`, `f32.sin`, etc.) get their type
// schemes from prelude module signatures, so `intrinsic_source_names` is
// empty. They're registered in ImplSource under both the surface form
// (`f32.+`) and a polymorphic-suffix form (`_w_intrinsic_+_f32`) for
// prelude-module use. Names are leaked via `Box::leak` to give them
// `&'static str` lifetimes — the catalog is built once at startup so
// each name leaks once.

fn leak_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

fn leak_two(a: &'static str, b: &'static str) -> &'static [&'static str] {
    Box::leak(Box::new([a, b]))
}

fn leak_one(a: &'static str) -> &'static [&'static str] {
    Box::leak(Box::new([a]))
}

fn per_type_op(ty: &str, op: &str, internal_op: &str, lowering: BuiltinLowering) -> BuiltinDefRaw {
    let surface = leak_str(format!("{}.{}", ty, op));
    let internal = leak_str(format!("_w_intrinsic_{}_{}", internal_op, ty));
    BuiltinDefRaw {
        surface_name: surface,
        intrinsic_source_names: &[],
        impl_source_names: leak_two(surface, internal),
        kind: BuiltinKind::Operator,
        purity: Purity::Pure,
        overloads: Box::leak(Box::new([BuiltinOverload {
            scheme: None,
            lowering,
        }])),
    }
}

fn per_type_conv(ty: &str, source_ty: &str, lowering: BuiltinLowering) -> BuiltinDefRaw {
    // User-facing conversions use a single name like `f32.i32`.
    let surface = leak_str(format!("{}.{}", ty, source_ty));
    BuiltinDefRaw {
        surface_name: surface,
        intrinsic_source_names: &[],
        impl_source_names: leak_one(surface),
        kind: BuiltinKind::ModuleBuiltin,
        purity: Purity::Pure,
        overloads: Box::leak(Box::new([BuiltinOverload {
            scheme: None,
            lowering,
        }])),
    }
}

fn intrinsic_only(name: &'static str, lowering: BuiltinLowering) -> BuiltinDefRaw {
    // Names emitted only as `_w_intrinsic_*_<ty>` (no user-facing form),
    // used internally for things like float-from/to-int conversions.
    BuiltinDefRaw {
        surface_name: name,
        intrinsic_source_names: &[],
        impl_source_names: leak_one(name),
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: Box::leak(Box::new([BuiltinOverload {
            scheme: None,
            lowering,
        }])),
    }
}

fn generate_per_type_ops() -> Vec<BuiltinDefRaw> {
    use BuiltinLowering::PrimOp as L;
    use PrimOp::*;
    let mut defs = Vec::new();

    let signed_ints: &[&str] = &["i8", "i16", "i32", "i64"];
    let unsigned_ints: &[&str] = &["u8", "u16", "u32", "u64"];
    let floats: &[&str] = &["f16", "f32", "f64"];

    // ---- numeric_modules: arithmetic, comparison, min/max/abs/sign/clamp ----
    for &ty in floats {
        for (op, prim) in [
            ("+", FAdd),
            ("-", FSub),
            ("*", FMul),
            ("/", FDiv),
            ("%", FRem),
            ("**", GlslExt(26)),
        ] {
            defs.push(per_type_op(ty, op, op, L(prim)));
        }
        for (op, prim) in [
            ("<", FOrdLessThan),
            ("==", FOrdEqual),
            ("!=", FOrdNotEqual),
            (">", FOrdGreaterThan),
            ("<=", FOrdLessThanEqual),
            (">=", FOrdGreaterThanEqual),
        ] {
            defs.push(per_type_op(ty, op, op, L(prim)));
        }
        defs.push(per_type_op(ty, "min", "min", L(GlslExt(37)))); // FMin
        defs.push(per_type_op(ty, "max", "max", L(GlslExt(40)))); // FMax
        defs.push(per_type_op(ty, "abs", "abs", L(GlslExt(4)))); // FAbs
        defs.push(per_type_op(ty, "sign", "sign", L(GlslExt(6)))); // FSign
        defs.push(per_type_op(ty, "clamp", "clamp", L(GlslExt(43)))); // FClamp
    }
    for &ty in signed_ints {
        for (op, prim) in [("+", IAdd), ("-", ISub), ("*", IMul), ("/", SDiv), ("%", SRem)] {
            defs.push(per_type_op(ty, op, op, L(prim)));
        }
        for (op, prim) in [
            ("<", SLessThan),
            ("==", IEqual),
            ("!=", INotEqual),
            (">", SGreaterThan),
            ("<=", SLessThanEqual),
            (">=", SGreaterThanEqual),
        ] {
            defs.push(per_type_op(ty, op, op, L(prim)));
        }
        defs.push(per_type_op(ty, "min", "min", L(GlslExt(39)))); // SMin
        defs.push(per_type_op(ty, "max", "max", L(GlslExt(42)))); // SMax
        defs.push(per_type_op(ty, "abs", "abs", L(GlslExt(5)))); // SAbs
        defs.push(per_type_op(ty, "sign", "sign", L(GlslExt(7)))); // SSign
        defs.push(per_type_op(ty, "clamp", "clamp", L(GlslExt(45)))); // SClamp
    }
    for &ty in unsigned_ints {
        for (op, prim) in [("+", IAdd), ("-", ISub), ("*", IMul), ("/", UDiv), ("%", UMod)] {
            defs.push(per_type_op(ty, op, op, L(prim)));
        }
        for (op, prim) in [
            ("<", ULessThan),
            ("==", IEqual),
            ("!=", INotEqual),
            (">", UGreaterThan),
            ("<=", ULessThanEqual),
            (">=", UGreaterThanEqual),
        ] {
            defs.push(per_type_op(ty, op, op, L(prim)));
        }
        defs.push(per_type_op(ty, "min", "min", L(GlslExt(38)))); // UMin
        defs.push(per_type_op(ty, "max", "max", L(GlslExt(41)))); // UMax
        defs.push(per_type_op(ty, "clamp", "clamp", L(GlslExt(44)))); // UClamp
    }

    // ---- integral_modules: bitwise + shifts ----
    for &ty in signed_ints.iter().chain(unsigned_ints.iter()) {
        defs.push(per_type_op(ty, "&", "&", L(BitwiseAnd)));
        defs.push(per_type_op(ty, "|", "|", L(BitwiseOr)));
        defs.push(per_type_op(ty, "^", "^", L(BitwiseXor)));
        defs.push(per_type_op(ty, "<<", "<<", L(ShiftLeftLogical)));
        let right_shift = if ty.starts_with('i') { ShiftRightArithmetic } else { ShiftRightLogical };
        defs.push(per_type_op(ty, ">>", ">>", L(right_shift)));
    }

    // ---- integral_modules: float-to-int + int-to-int conversions ----
    for &target in signed_ints {
        for &source in floats {
            defs.push(per_type_conv(target, source, L(FPToSI)));
        }
    }
    for &target in unsigned_ints {
        for &source in floats {
            defs.push(per_type_conv(target, source, L(FPToUI)));
        }
    }
    for &target in signed_ints {
        for &source in signed_ints {
            if target != source {
                defs.push(per_type_conv(target, source, L(SConvert)));
            }
        }
    }
    for &target in unsigned_ints {
        for &source in unsigned_ints {
            if target != source {
                defs.push(per_type_conv(target, source, L(UConvert)));
            }
        }
    }
    for (i, &s_ty) in signed_ints.iter().enumerate() {
        for (j, &u_ty) in unsigned_ints.iter().enumerate() {
            let same_width = i == j;
            defs.push(per_type_conv(
                s_ty,
                u_ty,
                if same_width { L(Bitcast) } else { L(SConvert) },
            ));
            defs.push(per_type_conv(
                u_ty,
                s_ty,
                if same_width { L(Bitcast) } else { L(UConvert) },
            ));
        }
    }

    // ---- real_modules: trig, hyperbolic, exp/log, rounding, lerp/fma, isnan/isinf, ldexp ----
    for &ty in floats {
        for (op, glsl) in [
            ("sin", 13),
            ("cos", 14),
            ("tan", 15),
            ("asin", 16),
            ("acos", 17),
            ("atan", 18),
            ("sinh", 19),
            ("cosh", 20),
            ("tanh", 21),
            ("asinh", 22),
            ("acosh", 23),
            ("atanh", 24),
            ("sqrt", 31),
            ("rsqrt", 32),
            ("exp", 27),
            ("log", 28),
            ("log2", 30),
            ("radians", 11),
            ("degrees", 12),
            ("floor", 8),
            ("ceil", 9),
            ("round", 1),
            ("trunc", 3),
            ("fract", 10),
        ] {
            defs.push(per_type_op(ty, op, op, L(GlslExt(glsl))));
        }
        defs.push(per_type_op(ty, "atan2", "atan2", L(GlslExt(25))));
        defs.push(per_type_op(ty, "pow", "pow", L(GlslExt(26))));
        defs.push(per_type_op(ty, "mod", "mod", L(FMod)));
        defs.push(per_type_op(ty, "lerp", "lerp", L(GlslExt(46)))); // FMix
        defs.push(per_type_op(ty, "fma", "fma", L(GlslExt(50))));
        defs.push(per_type_op(ty, "isnan", "isnan", L(IsNan)));
        defs.push(per_type_op(ty, "isinf", "isinf", L(IsInf)));
        defs.push(per_type_op(ty, "ldexp", "ldexp", L(GlslExt(53))));
    }

    // ---- float_modules: conversions ----
    for &ty in floats {
        for &source in signed_ints {
            defs.push(per_type_conv(ty, source, L(SIToFP)));
            defs.push(intrinsic_only(
                leak_str(format!("_w_intrinsic_{}_from_{}", ty, source)),
                L(SIToFP),
            ));
        }
        for &source in unsigned_ints {
            defs.push(per_type_conv(ty, source, L(UIToFP)));
            defs.push(intrinsic_only(
                leak_str(format!("_w_intrinsic_{}_from_{}", ty, source)),
                L(UIToFP),
            ));
        }
        for &source in floats {
            if source != ty {
                defs.push(per_type_conv(ty, source, L(FPConvert)));
                defs.push(intrinsic_only(
                    leak_str(format!("_w_intrinsic_{}_from_{}", ty, source)),
                    L(FPConvert),
                ));
            }
        }
        for &target in signed_ints {
            defs.push(intrinsic_only(
                leak_str(format!("_w_intrinsic_{}_to_{}", ty, target)),
                L(FPToSI),
            ));
        }
        for &target in unsigned_ints {
            defs.push(intrinsic_only(
                leak_str(format!("_w_intrinsic_{}_to_{}", ty, target)),
                L(FPToUI),
            ));
        }
        defs.push(intrinsic_only(
            leak_str(format!("_w_intrinsic_{}_from_bits", ty)),
            L(Bitcast),
        ));
        defs.push(intrinsic_only(
            leak_str(format!("_w_intrinsic_{}_to_bits", ty)),
            L(Bitcast),
        ));
    }

    defs
}
