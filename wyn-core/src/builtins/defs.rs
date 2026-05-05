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
    INTRINSIC_CROSS, INTRINSIC_DETERMINANT, INTRINSIC_DISTANCE, INTRINSIC_DOT, INTRINSIC_FLOOR,
    INTRINSIC_FRACT, INTRINSIC_INVERSE, INTRINSIC_LENGTH, INTRINSIC_MAGNITUDE, INTRINSIC_MIX,
    INTRINSIC_NORMALIZE, INTRINSIC_OUTER, INTRINSIC_REFLECT, INTRINSIC_REFRACT, INTRINSIC_SMOOTHSTEP,
    INTRINSIC_UNINIT,
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

pub static ALL_BUILTINS: &[BuiltinDefRaw] = &[
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
    // Polymorphic intrinsics — surface name in IntrinsicSource,
    // `_w_intrinsic_*` in ImplSource. TLC's INTRINSIC_RENAMES translates
    // user calls to the internal name before backend dispatch.
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
    // scatter, hist_1d, rotr32. These have IntrinsicSource schemes but no
    // ImplSource lowering — they're handled in TLC (SOACs) or specialized
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
];
