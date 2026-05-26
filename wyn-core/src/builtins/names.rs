// Canonical names for compiler-internal `_w_intrinsic_*` operations.
// Anywhere code emits, matches against, or compares an intrinsic name,
// reference one of these constants instead of re-spelling the literal.
//
// Names that are programmatically built via `format!("_w_intrinsic_{}_{}", ...)`
// (e.g. the float<→int conversion table) intentionally don't have
// constants — there's no place to use them.

// ---------------------------------------------------------------------------
// Array manipulation
// ---------------------------------------------------------------------------

/// Functional `with` intrinsic emitted for `arr with [i] = v`.
pub const INTRINSIC_ARRAY_WITH: &str = "_w_intrinsic_array_with";

/// In-place variant produced by the TLC promotion pass when the source
/// array is dead-after-call and its owner is mutable.
pub const INTRINSIC_ARRAY_WITH_INPLACE: &str = "_w_intrinsic_array_with_inplace";

/// `length(arr)` — runtime length of an array (any variant).
pub const INTRINSIC_LENGTH: &str = "_w_intrinsic_length";

/// Allocate an uninitialized buffer of a given array type.
pub const INTRINSIC_UNINIT: &str = "_w_intrinsic_uninit";

/// `arr[start..end]` — produces a view aliasing the source.
pub const INTRINSIC_SLICE: &str = "_w_intrinsic_slice";

/// `replicate(n, x)` — array of `n` copies of `x`.
pub const INTRINSIC_REPLICATE: &str = "_w_intrinsic_replicate";

// ---------------------------------------------------------------------------
// Storage buffer access
// ---------------------------------------------------------------------------

/// Runtime length of a storage-buffer-backed view.
pub const INTRINSIC_STORAGE_LEN: &str = "_w_intrinsic_storage_len";

/// Element pointer (`PlaceId`) into a storage view at a given index.
pub const INTRINSIC_STORAGE_INDEX: &str = "_w_intrinsic_storage_index";

/// Effectful write of a value into a storage view at a given index.
pub const INTRINSIC_STORAGE_STORE: &str = "_w_intrinsic_storage_store";

// ---------------------------------------------------------------------------
// Geometric / vector
// ---------------------------------------------------------------------------

/// Dot product.
pub const INTRINSIC_DOT: &str = "_w_intrinsic_dot";

/// 3-D cross product.
pub const INTRINSIC_CROSS: &str = "_w_intrinsic_cross";

/// Euclidean distance between two vectors.
pub const INTRINSIC_DISTANCE: &str = "_w_intrinsic_distance";

/// Vector length / magnitude.
pub const INTRINSIC_MAGNITUDE: &str = "_w_intrinsic_magnitude";

/// Normalize to unit length.
pub const INTRINSIC_NORMALIZE: &str = "_w_intrinsic_normalize";

/// Vector reflection.
pub const INTRINSIC_REFLECT: &str = "_w_intrinsic_reflect";

/// Vector refraction.
pub const INTRINSIC_REFRACT: &str = "_w_intrinsic_refract";

/// Outer product.
pub const INTRINSIC_OUTER: &str = "_w_intrinsic_outer";

// ---------------------------------------------------------------------------
// Matrix
// ---------------------------------------------------------------------------

/// Matrix determinant.
pub const INTRINSIC_DETERMINANT: &str = "_w_intrinsic_determinant";

/// Matrix inverse.
pub const INTRINSIC_INVERSE: &str = "_w_intrinsic_inverse";

// ---------------------------------------------------------------------------
// Scalar / vector math
// ---------------------------------------------------------------------------

/// Absolute value.
pub const INTRINSIC_ABS: &str = "_w_intrinsic_abs";

/// Round up.
pub const INTRINSIC_CEIL: &str = "_w_intrinsic_ceil";

/// Round down.
pub const INTRINSIC_FLOOR: &str = "_w_intrinsic_floor";

/// Fractional part.
pub const INTRINSIC_FRACT: &str = "_w_intrinsic_fract";

/// Cosine.
pub const INTRINSIC_COS: &str = "_w_intrinsic_cos";

/// Clamp to range.
pub const INTRINSIC_CLAMP: &str = "_w_intrinsic_clamp";

/// Linear blend.
pub const INTRINSIC_MIX: &str = "_w_intrinsic_mix";

/// Smoothstep interpolation.
pub const INTRINSIC_SMOOTHSTEP: &str = "_w_intrinsic_smoothstep";

// ---------------------------------------------------------------------------
// Threading / dispatch
// ---------------------------------------------------------------------------

/// Compute-shader thread id.
pub const INTRINSIC_THREAD_ID: &str = "_w_intrinsic_thread_id";

/// Compute-shader local invocation id (the `.x` index within the workgroup).
pub const INTRINSIC_LOCAL_ID: &str = "_w_intrinsic_local_id";

/// Compute-shader workgroup count (the `.x` dispatch dimension).
pub const INTRINSIC_NUM_WORKGROUPS: &str = "_w_intrinsic_num_workgroups";

// ---------------------------------------------------------------------------
// Textures / samplers
// ---------------------------------------------------------------------------

/// Raw texel fetch at an integer coordinate + mip level (no filtering).
/// Referentially transparent: result depends only on its arguments.
pub const INTRINSIC_TEXTURE_LOAD: &str = "_w_intrinsic_texture_load";

/// Filtered sample at an explicit LOD. v1 uses explicit LOD (not
/// implicit/derivative-based) to stay referentially transparent — see the
/// v2 note in the texture plan for gradient-based filtering
/// (`texture_sample_grad`).
pub const INTRINSIC_TEXTURE_SAMPLE: &str = "_w_intrinsic_texture_sample";
