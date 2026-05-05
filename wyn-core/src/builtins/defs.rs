use crate::builtins::catalog::{BuiltinDefRaw, BuiltinKind, BuiltinOverload, Purity};
use crate::builtins::lowering::BuiltinLowering;
use crate::builtins::scheme::{
    array_to_i32, mat_square_to_mat, mat_square_to_scalar, mat_x_mat, mat_x_vec, scalar_unary, unit_to_t,
    vec_binary_same, vec_binary_to_scalar, vec_clamp_scalar_lohi, vec_mix_scalar_interp,
    vec_smoothstep_scalar_edges, vec_ternary_same, vec_to_scalar, vec_unary_same, vec_vec_outer,
    vec_vec_scalar_to_vec, vec_x_mat, vec3f32_binary,
};
use crate::impl_source::PrimOp;
use crate::intrinsics::{
    INTRINSIC_ABS, INTRINSIC_ARRAY_WITH, INTRINSIC_ARRAY_WITH_INPLACE, INTRINSIC_CEIL, INTRINSIC_CLAMP,
    INTRINSIC_COS, INTRINSIC_CROSS, INTRINSIC_DETERMINANT, INTRINSIC_DISTANCE, INTRINSIC_DOT,
    INTRINSIC_FLOOR, INTRINSIC_FRACT, INTRINSIC_INVERSE, INTRINSIC_LENGTH, INTRINSIC_MAGNITUDE,
    INTRINSIC_MIX, INTRINSIC_NORMALIZE, INTRINSIC_OUTER, INTRINSIC_REFLECT, INTRINSIC_REFRACT,
    INTRINSIC_SLICE, INTRINSIC_SMOOTHSTEP, INTRINSIC_STORAGE_INDEX, INTRINSIC_STORAGE_LEN,
    INTRINSIC_STORAGE_STORE, INTRINSIC_THREAD_ID, INTRINSIC_UNINIT,
};

// ---------------------------------------------------------------------------
// `vec.*` module ops — vector-only counterparts to per-type scalar ops
// ---------------------------------------------------------------------------

macro_rules! vec_unary {
    ($name:literal, $glsl_ext:expr) => {
        BuiltinDefRaw {
            surface_name: $name,
            intrinsic_source_names: &[$name],
            impl_source_names: &[$name],
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
            intrinsic_source_names: &[$name],
            impl_source_names: &[$name],
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
            intrinsic_source_names: &[$name],
            impl_source_names: &[$name],
            kind: BuiltinKind::ModuleBuiltin,
            purity: Purity::Pure,
            overloads: &[BuiltinOverload {
                scheme: vec_ternary_same,
                lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt($glsl_ext)),
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
    vec_unary!("vec.sin", 13),
    vec_unary!("vec.cos", 14),
    vec_unary!("vec.tan", 15),
    vec_unary!("vec.asin", 16),
    vec_unary!("vec.acos", 17),
    vec_unary!("vec.atan", 18),
    // ---- vec.* hyperbolic ----
    vec_unary!("vec.sinh", 19),
    vec_unary!("vec.cosh", 20),
    vec_unary!("vec.tanh", 21),
    vec_unary!("vec.asinh", 22),
    vec_unary!("vec.acosh", 23),
    vec_unary!("vec.atanh", 24),
    // ---- vec.* exp/log/roots ----
    vec_unary!("vec.sqrt", 31),
    vec_unary!("vec.rsqrt", 32),
    vec_unary!("vec.exp", 27),
    vec_unary!("vec.exp2", 29),
    vec_unary!("vec.log", 28),
    vec_unary!("vec.log2", 30),
    // ---- vec.* rounding/sign ----
    vec_unary!("vec.floor", 8),
    vec_unary!("vec.ceil", 9),
    vec_unary!("vec.round", 1),
    vec_unary!("vec.trunc", 3),
    vec_unary!("vec.fract", 10),
    vec_unary!("vec.abs", 4),
    vec_unary!("vec.sign", 6),
    // ---- vec.* angle conversion ----
    vec_unary!("vec.radians", 11),
    vec_unary!("vec.degrees", 12),
    // ---- vec.* binary ----
    vec_binary!("vec.pow", BuiltinLowering::PrimOp(PrimOp::GlslExt(26))),
    vec_binary!("vec.atan2", BuiltinLowering::PrimOp(PrimOp::GlslExt(25))),
    vec_binary!("vec.mod", BuiltinLowering::PrimOp(PrimOp::FMod)),
    vec_binary!("vec.min", BuiltinLowering::PrimOp(PrimOp::GlslExt(37))),
    vec_binary!("vec.max", BuiltinLowering::PrimOp(PrimOp::GlslExt(40))),
    // ---- vec.* ternary ----
    vec_ternary!("vec.clamp", 43),
    vec_ternary!("vec.mix", 46),
    vec_ternary!("vec.smoothstep", 49),
    // -----------------------------------------------------------------------
    // Polymorphic intrinsics — surface name (e.g. "magnitude") published
    // for type-check-time resolution; `_w_intrinsic_*` form (e.g.
    // "_w_intrinsic_magnitude") published for backend lowering. TLC's
    // INTRINSIC_RENAMES translates user calls to the internal name
    // before backend dispatch.
    // -----------------------------------------------------------------------
    BuiltinDefRaw {
        surface_name: "magnitude",
        intrinsic_source_names: &["magnitude"],
        impl_source_names: &[INTRINSIC_MAGNITUDE],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: vec_to_scalar,
            lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(66)),
        }],
    },
    BuiltinDefRaw {
        surface_name: "normalize",
        intrinsic_source_names: &["normalize"],
        impl_source_names: &[INTRINSIC_NORMALIZE],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: vec_unary_same,
            lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(69)),
        }],
    },
    BuiltinDefRaw {
        surface_name: "dot",
        intrinsic_source_names: &["dot"],
        impl_source_names: &[INTRINSIC_DOT],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: vec_binary_to_scalar,
            lowering: BuiltinLowering::PrimOp(PrimOp::Dot),
        }],
    },
    BuiltinDefRaw {
        surface_name: "cross",
        intrinsic_source_names: &["cross"],
        impl_source_names: &[INTRINSIC_CROSS],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: vec3f32_binary,
            lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(68)),
        }],
    },
    BuiltinDefRaw {
        surface_name: "distance",
        intrinsic_source_names: &["distance"],
        impl_source_names: &[INTRINSIC_DISTANCE],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: vec_binary_to_scalar,
            lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(67)),
        }],
    },
    BuiltinDefRaw {
        surface_name: "reflect",
        intrinsic_source_names: &["reflect"],
        impl_source_names: &[INTRINSIC_REFLECT],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: vec_binary_same,
            lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(71)),
        }],
    },
    BuiltinDefRaw {
        surface_name: "refract",
        intrinsic_source_names: &["refract"],
        impl_source_names: &[INTRINSIC_REFRACT],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: vec_vec_scalar_to_vec,
            lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(72)),
        }],
    },
    BuiltinDefRaw {
        surface_name: "outer",
        intrinsic_source_names: &["outer"],
        impl_source_names: &[INTRINSIC_OUTER],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: vec_vec_outer,
            lowering: BuiltinLowering::PrimOp(PrimOp::OuterProduct),
        }],
    },
    BuiltinDefRaw {
        surface_name: "determinant",
        intrinsic_source_names: &["determinant"],
        impl_source_names: &[INTRINSIC_DETERMINANT],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: mat_square_to_scalar,
            lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(33)),
        }],
    },
    BuiltinDefRaw {
        surface_name: "inverse",
        intrinsic_source_names: &["inverse"],
        impl_source_names: &[INTRINSIC_INVERSE],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: mat_square_to_mat,
            lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(34)),
        }],
    },
    BuiltinDefRaw {
        surface_name: "mul",
        intrinsic_source_names: &["mul"],
        impl_source_names: &[],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[
            BuiltinOverload {
                scheme: mat_x_mat,
                lowering: BuiltinLowering::PrimOp(PrimOp::MatrixTimesMatrix),
            },
            BuiltinOverload {
                scheme: mat_x_vec,
                lowering: BuiltinLowering::PrimOp(PrimOp::MatrixTimesVector),
            },
            BuiltinOverload {
                scheme: vec_x_mat,
                lowering: BuiltinLowering::PrimOp(PrimOp::VectorTimesMatrix),
            },
        ],
    },
    // ---- Scalar polymorphic math ----
    BuiltinDefRaw {
        surface_name: "abs",
        intrinsic_source_names: &["abs"],
        impl_source_names: &[INTRINSIC_ABS],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: scalar_unary,
            lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(4)),
        }],
    },
    BuiltinDefRaw {
        surface_name: "sign",
        intrinsic_source_names: &["sign"],
        impl_source_names: &[],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: scalar_unary,
            lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(6)),
        }],
    },
    BuiltinDefRaw {
        surface_name: "floor",
        intrinsic_source_names: &["floor"],
        impl_source_names: &[INTRINSIC_FLOOR],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: scalar_unary,
            lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(8)),
        }],
    },
    BuiltinDefRaw {
        surface_name: "ceil",
        intrinsic_source_names: &["ceil"],
        impl_source_names: &[INTRINSIC_CEIL],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: scalar_unary,
            lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(9)),
        }],
    },
    BuiltinDefRaw {
        surface_name: "fract",
        intrinsic_source_names: &["fract"],
        impl_source_names: &[INTRINSIC_FRACT],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: scalar_unary,
            lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(10)),
        }],
    },
    BuiltinDefRaw {
        surface_name: "min",
        intrinsic_source_names: &["min"],
        impl_source_names: &["_w_intrinsic_min"],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: crate::builtins::scheme::scalar_binary,
            lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(37)),
        }],
    },
    BuiltinDefRaw {
        surface_name: "max",
        intrinsic_source_names: &["max"],
        impl_source_names: &["_w_intrinsic_max"],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: crate::builtins::scheme::scalar_binary,
            lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(40)),
        }],
    },
    BuiltinDefRaw {
        surface_name: "clamp",
        intrinsic_source_names: &["clamp"],
        impl_source_names: &[INTRINSIC_CLAMP],
        kind: BuiltinKind::UserVisible,
        purity: Purity::Pure,
        overloads: &[
            BuiltinOverload {
                scheme: crate::builtins::scheme::scalar_ternary,
                lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(43)),
            },
            BuiltinOverload {
                scheme: vec_clamp_scalar_lohi,
                lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(43)),
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
                scheme: crate::builtins::scheme::scalar_ternary,
                lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(46)),
            },
            BuiltinOverload {
                scheme: vec_mix_scalar_interp,
                lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(46)),
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
                scheme: crate::builtins::scheme::scalar_ternary,
                lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(49)),
            },
            BuiltinOverload {
                scheme: vec_smoothstep_scalar_edges,
                lowering: BuiltinLowering::PrimOp(PrimOp::GlslExt(49)),
            },
        ],
    },
    // ---- HOF intrinsic: length (ImplSource has Intrinsic::Length lowering) ----
    BuiltinDefRaw {
        surface_name: INTRINSIC_LENGTH,
        intrinsic_source_names: &[INTRINSIC_LENGTH],
        impl_source_names: &[INTRINSIC_LENGTH],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: array_to_i32,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::Length),
        }],
    },
    // ---- HOF intrinsic: uninit ----
    BuiltinDefRaw {
        surface_name: INTRINSIC_UNINIT,
        intrinsic_source_names: &[INTRINSIC_UNINIT],
        impl_source_names: &[INTRINSIC_UNINIT],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: unit_to_t,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::Uninit),
        }],
    },
    // ---- HOF intrinsic: array_with (functional + in-place) ----
    BuiltinDefRaw {
        surface_name: INTRINSIC_ARRAY_WITH,
        intrinsic_source_names: &[INTRINSIC_ARRAY_WITH],
        impl_source_names: &[INTRINSIC_ARRAY_WITH],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: crate::builtins::scheme::array_index_value_to_array,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::ArrayWith),
        }],
    },
    BuiltinDefRaw {
        surface_name: INTRINSIC_ARRAY_WITH_INPLACE,
        intrinsic_source_names: &[],
        impl_source_names: &[INTRINSIC_ARRAY_WITH_INPLACE],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Effectful,
        overloads: &[BuiltinOverload {
            scheme: crate::builtins::scheme::array_index_value_to_array,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::ArrayWithInPlace),
        }],
    },
    // ---- HOF intrinsics: replicate, reduce, filter, scan, map, map_into,
    // scatter, hist_1d, rotr32. These publish polymorphic schemes but no
    // backend lowering — they're handled in TLC (SOACs) or specialized
    // earlier in the pipeline. ----
    BuiltinDefRaw {
        surface_name: "_w_intrinsic_replicate",
        intrinsic_source_names: &["_w_intrinsic_replicate"],
        impl_source_names: &[],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: crate::builtins::scheme::replicate_scheme,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::Placeholder),
        }],
    },
    BuiltinDefRaw {
        surface_name: "_w_intrinsic_reduce",
        intrinsic_source_names: &["_w_intrinsic_reduce"],
        impl_source_names: &[],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: crate::builtins::scheme::reduce_scheme,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::Placeholder),
        }],
    },
    BuiltinDefRaw {
        surface_name: "_w_intrinsic_filter",
        intrinsic_source_names: &["_w_intrinsic_filter"],
        impl_source_names: &[],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: crate::builtins::scheme::filter_scheme,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::Placeholder),
        }],
    },
    BuiltinDefRaw {
        surface_name: "_w_intrinsic_scan",
        intrinsic_source_names: &["_w_intrinsic_scan"],
        impl_source_names: &[],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: crate::builtins::scheme::scan_scheme,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::Placeholder),
        }],
    },
    BuiltinDefRaw {
        surface_name: "_w_intrinsic_map",
        intrinsic_source_names: &["_w_intrinsic_map"],
        impl_source_names: &[],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: crate::builtins::scheme::map_scheme,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::Placeholder),
        }],
    },
    BuiltinDefRaw {
        surface_name: "_w_intrinsic_map_into",
        intrinsic_source_names: &["_w_intrinsic_map_into"],
        impl_source_names: &[],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Effectful,
        overloads: &[BuiltinOverload {
            scheme: crate::builtins::scheme::map_into_scheme,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::Placeholder),
        }],
    },
    BuiltinDefRaw {
        surface_name: "_w_intrinsic_scatter",
        intrinsic_source_names: &["_w_intrinsic_scatter"],
        impl_source_names: &[],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: crate::builtins::scheme::scatter_scheme,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::Placeholder),
        }],
    },
    BuiltinDefRaw {
        surface_name: "_w_intrinsic_hist_1d",
        intrinsic_source_names: &["_w_intrinsic_hist_1d"],
        impl_source_names: &[],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: crate::builtins::scheme::hist_1d_scheme,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::Placeholder),
        }],
    },
    BuiltinDefRaw {
        surface_name: "_w_intrinsic_rotr32",
        intrinsic_source_names: &["_w_intrinsic_rotr32"],
        impl_source_names: &[],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: crate::builtins::scheme::u32_binary,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::Placeholder),
        }],
    },
    // ---- Compiler-internal intrinsics: emitted by the codegen pipeline,
    // not user-facing. They have no published scheme (the compiler
    // synthesises type-correct calls) and lowering is special-cased per
    // backend. Their presence in the catalog gives every PureOp::Intrinsic
    // emission a `BuiltinId`. ----
    BuiltinDefRaw {
        surface_name: INTRINSIC_STORAGE_LEN,
        intrinsic_source_names: &[],
        impl_source_names: &[],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: dummy_scheme,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::Placeholder),
        }],
    },
    BuiltinDefRaw {
        surface_name: INTRINSIC_STORAGE_INDEX,
        intrinsic_source_names: &[],
        impl_source_names: &[],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: dummy_scheme,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::Placeholder),
        }],
    },
    BuiltinDefRaw {
        surface_name: INTRINSIC_STORAGE_STORE,
        intrinsic_source_names: &[],
        impl_source_names: &[],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Effectful,
        overloads: &[BuiltinOverload {
            scheme: dummy_scheme,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::Placeholder),
        }],
    },
    BuiltinDefRaw {
        surface_name: INTRINSIC_SLICE,
        intrinsic_source_names: &[],
        impl_source_names: &[],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: dummy_scheme,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::Placeholder),
        }],
    },
    BuiltinDefRaw {
        surface_name: INTRINSIC_THREAD_ID,
        intrinsic_source_names: &[],
        impl_source_names: &[],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: dummy_scheme,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::Placeholder),
        }],
    },
    BuiltinDefRaw {
        surface_name: INTRINSIC_COS,
        intrinsic_source_names: &[],
        impl_source_names: &[],
        kind: BuiltinKind::InternalIntrinsic,
        purity: Purity::Pure,
        overloads: &[BuiltinOverload {
            scheme: dummy_scheme,
            lowering: BuiltinLowering::Intrinsic(crate::impl_source::Intrinsic::Placeholder),
        }],
    },
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

fn dummy_scheme(_: &mut dyn crate::type_checker::TypeVarGenerator) -> crate::ast::TypeScheme {
    crate::ast::TypeScheme::Monotype(crate::ast::Type::Constructed(crate::ast::TypeName::Unit, vec![]))
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
            scheme: dummy_scheme,
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
            scheme: dummy_scheme,
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
            scheme: dummy_scheme,
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
