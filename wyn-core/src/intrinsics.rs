// Polymorphic intrinsic registry
// Provides type schemes for truly polymorphic functions (map, magnitude, matrix ops, etc.)
// Implementations come from ImplSource

use crate::ast::{Type, TypeName, TypeScheme};
use crate::type_checker::TypeVarGenerator;
use polytype::Context;
use std::collections::HashMap;

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
// SOACs that survive past TLC normalization
// ---------------------------------------------------------------------------

/// `filter(pred, arr)` — predicate-driven array compaction.
pub const INTRINSIC_FILTER: &str = "_w_intrinsic_filter";

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

/// Entry for a intrinsic with its type scheme
#[derive(Debug, Clone)]
pub struct IntrinsicEntry {
    /// Type scheme (e.g., "forall a. a -> a")
    pub scheme: TypeScheme,
}

impl IntrinsicEntry {
    /// Compute the arity (number of arguments) of this intrinsic
    pub fn arity(&self) -> usize {
        fn count_arrows(ty: &Type) -> usize {
            match ty {
                Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => 1 + count_arrows(&args[1]),
                _ => 0,
            }
        }

        fn scheme_arity(scheme: &TypeScheme) -> usize {
            match scheme {
                TypeScheme::Monotype(ty) => count_arrows(ty),
                TypeScheme::Polytype { body, .. } => scheme_arity(body),
            }
        }

        scheme_arity(&self.scheme)
    }
}

/// Result of looking up a intrinsic - either a single entry or an overload set
pub enum IntrinsicLookup<'a> {
    /// Single intrinsic with no overloads
    Single(&'a IntrinsicEntry),
    /// Multiple overloads that need resolution based on argument types
    Overloaded(OverloadSet<'a>),
}

/// A set of overloaded intrinsics that need type-based resolution
pub struct OverloadSet<'a> {
    entries: &'a [IntrinsicEntry],
    arity: usize,
}

impl<'a> OverloadSet<'a> {
    /// Create a fresh type for this overload set: ?0 -> ?1 -> ... -> ?n
    /// Since all overloads have the same arity, we can defer resolution to unification time.
    pub fn fresh_type(&self, ctx: &mut impl TypeVarGenerator) -> Type {
        let mut ty = ctx.new_variable();
        for _ in 0..self.arity {
            let param = ctx.new_variable();
            ty = Type::arrow(param, ty);
        }
        ty
    }

    /// Choose the correct overload based on argument types.
    /// Returns the matching entry and the resolved return type.
    /// Uses backtracking: checkpoints context, tries each overload, rolls back on failure.
    pub fn choose(
        &self,
        arg_types: &[Type],
        ctx: &mut Context<TypeName>,
    ) -> Option<(&'a IntrinsicEntry, Type)> {
        for entry in self.entries {
            let checkpoint = ctx.len();

            // Instantiate this overload with fresh type variables
            let func_type = entry.scheme.instantiate(ctx);

            // Try to unify parameter types with argument types
            if let Some(return_type) = Self::try_unify(&func_type, arg_types, ctx) {
                return Some((entry, return_type));
            }

            // Roll back and try next overload
            ctx.rollback(checkpoint);
        }
        None
    }

    /// Try to unify a function type with the given argument types.
    /// Returns the return type if successful, None otherwise.
    fn try_unify(func_type: &Type, arg_types: &[Type], ctx: &mut Context<TypeName>) -> Option<Type> {
        let mut current_type = func_type.clone();

        for arg_type in arg_types {
            // Decompose the function type: should be param_ty -> rest
            let param_ty = ctx.new_variable();
            let rest_ty = ctx.new_variable();
            let expected_arrow = Type::arrow(param_ty.clone(), rest_ty.clone());

            // Unify current function type with the expected arrow type
            if ctx.unify(&current_type, &expected_arrow).is_err() {
                return None;
            }

            // Unify the parameter type with the argument type
            let param_ty = param_ty.apply(ctx);
            if ctx.unify(&param_ty, arg_type).is_err() {
                return None;
            }

            // Continue with the rest of the function type
            current_type = rest_ty.apply(ctx);
        }

        Some(current_type)
    }

    /// Get the entries (for error messages)
    pub fn entries(&self) -> &[IntrinsicEntry] {
        self.entries
    }
}

/// Registry for polymorphic intrinsic types
#[derive(Clone)]
pub struct IntrinsicSource {
    /// Maps function name to entry or entries (for overloads)
    intrinsics: HashMap<String, Vec<IntrinsicEntry>>,
}

impl IntrinsicSource {
    pub fn new(ctx: &mut impl TypeVarGenerator) -> Self {
        let mut registry = IntrinsicSource {
            intrinsics: HashMap::new(),
        };

        registry.populate_from_catalog(crate::builtins::catalog(), ctx);

        registry
    }

    /// Walk every catalog entry and register each overload under each
    /// of the entry's `intrinsic_source_names`.
    fn populate_from_catalog(
        &mut self,
        catalog: &crate::builtins::BuiltinCatalog,
        ctx: &mut impl TypeVarGenerator,
    ) {
        for def in catalog.defs() {
            for &name in def.intrinsic_source_names() {
                for ovld in def.overloads() {
                    let scheme = (ovld.scheme)(ctx);
                    self.add_overload(name.to_string(), IntrinsicEntry { scheme });
                }
            }
        }
    }

    /// Add an overload for a intrinsic.
    /// Asserts that all overloads for the same name have the same arity.
    fn add_overload(&mut self, name: String, entry: IntrinsicEntry) {
        let new_arity = entry.arity();
        let entries = self.intrinsics.entry(name.clone()).or_default();
        if let Some(existing) = entries.first() {
            let existing_arity = existing.arity();
            assert_eq!(
                existing_arity, new_arity,
                "BUG: Overload for '{}' has arity {} but existing overloads have arity {}",
                name, new_arity, existing_arity
            );
        }
        entries.push(entry);
    }

    /// Get a intrinsic by name
    pub fn get(&self, name: &str) -> Option<IntrinsicLookup<'_>> {
        self.intrinsics.get(name).map(|entries| {
            if entries.len() == 1 {
                IntrinsicLookup::Single(&entries[0])
            } else {
                let arity = entries[0].arity();
                IntrinsicLookup::Overloaded(OverloadSet { entries, arity })
            }
        })
    }

    /// Get all intrinsic names as a HashSet (for use in lambda lifting to exclude from capture)
    pub fn all_names(&self) -> std::collections::HashSet<String> {
        self.intrinsics.keys().cloned().collect()
    }
}

impl Default for IntrinsicSource {
    fn default() -> Self {
        let mut ctx = polytype::Context::<TypeName>::default();
        Self::new(&mut ctx)
    }
}
