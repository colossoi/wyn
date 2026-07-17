//! Type system definitions and utilities for Wyn.
//!
//! This module contains:
//! - `TypeName`: The type name constructors for the Wyn type system
//! - Helper functions for creating common types (i32, f32, vec, etc.)
//! - The type checker (`checker` submodule)

pub mod checker;
pub mod patterns;
pub mod run;

use crate::ast::Span;
use crate::LookupMap;
use std::hash::{Hash, Hasher};

/// Map a swizzle letter to its component index. Supports both the
/// `xyzw` (position) and `rgba` (color) sets, following WGSL — `r`/`g`/
/// `b`/`a` are aliases for `x`/`y`/`z`/`w`. Returns `None` for any other
/// character.
pub fn swizzle_component_index(c: char) -> Option<usize> {
    match c {
        'x' | 'r' => Some(0),
        'y' | 'g' => Some(1),
        'z' | 'b' => Some(2),
        'w' | 'a' => Some(3),
        _ => None,
    }
}

/// Predicate for a valid vector swizzle field: 1-4 characters, every
/// character drawn from *one* of the two swizzle sets (`xyzw` or
/// `rgba`) — mixing `.xg` is not a swizzle.
pub fn is_swizzle_field(field: &str) -> bool {
    if field.is_empty() || field.len() > 4 {
        return false;
    }
    let all_xyzw = field.chars().all(|c| matches!(c, 'x' | 'y' | 'z' | 'w'));
    let all_rgba = field.chars().all(|c| matches!(c, 'r' | 'g' | 'b' | 'a'));
    all_xyzw || all_rgba
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

// Type aliases for polytype types specialized to our TypeName
pub type Type = polytype::Type<TypeName>;
pub type TypeScheme = polytype::TypeScheme<TypeName>;

/// A bodyless external callable declaration shared by compiler IRs.
#[derive(Clone, Debug)]
pub struct ExternDecl<Ty = Type> {
    pub name: String,
    pub span: Span,
    pub linkage_name: String,
    pub params: Vec<(Ty, String)>,
    pub return_ty: Ty,
}

/// Record field names that preserve source order but have order-independent equality.
/// The actual field types are stored in Type::Constructed's type argument vector.
#[derive(Debug, Clone)]
pub struct RecordFields(pub Vec<String>);

impl RecordFields {
    pub fn new(fields: Vec<String>) -> Self {
        RecordFields(fields)
    }

    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn into_vec(self) -> Vec<String> {
        self.0
    }

    pub fn contains(&self, key: &str) -> bool {
        self.0.iter().any(|name| name == key)
    }

    pub fn get_index(&self, key: &str) -> Option<usize> {
        self.0.iter().position(|name| name == key)
    }
}

impl From<Vec<String>> for RecordFields {
    fn from(fields: Vec<String>) -> Self {
        RecordFields(fields)
    }
}

impl FromIterator<String> for RecordFields {
    fn from_iter<T: IntoIterator<Item = String>>(iter: T) -> Self {
        RecordFields(iter.into_iter().collect())
    }
}

impl PartialEq for RecordFields {
    fn eq(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }

        let mut self_fields: Vec<&str> = self.0.iter().map(String::as_str).collect();
        let mut other_fields: Vec<&str> = other.0.iter().map(String::as_str).collect();
        self_fields.sort_unstable();
        other_fields.sort_unstable();
        self_fields == other_fields
    }
}

impl Eq for RecordFields {}

impl Hash for RecordFields {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut fields: Vec<&str> = self.0.iter().map(String::as_str).collect();
        fields.sort_unstable();
        fields.hash(state);
    }
}

/// Unique identifier for skolem constants.
/// Skolems are created when opening existential types and are rigid (only unify with themselves).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SkolemId(pub u32);

impl From<u32> for SkolemId {
    fn from(id: u32) -> Self {
        SkolemId(id)
    }
}

impl std::fmt::Display for SkolemId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Type name constructors for the Wyn type system.
///
/// Note on type name variants:
/// - `Float/Int/SInt`: Numeric primitive types with bit widths (e.g., Float(32), SInt(32))
/// - `Str`: Other primitive type names hardcoded in the compiler (e.g., "->")
///   Uses static strings for efficiency
/// - `Tuple`: Tuple type constructor with arity (number of fields)
/// - `Named`: Type names parsed from user source code (e.g., "vec3", "MyType")
///   Could refer to built-in types, type aliases, or user-defined types
///   Uses owned String since the name comes from parsed input
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeName {
    /// Boolean type.
    Bool,
    /// Floating point types: f16, f32, f64, etc.
    Float(usize),
    /// Unsigned integer types: u8, u16, u32, u64
    UInt(usize),
    /// Signed integer types: i8, i16, i32, i64
    Int(usize),
    /// Unified array type constructor.
    /// Type args: `[elem, variant, dim_0, dim_1, …, dim_{rank-1}, region]`
    /// - `elem`: element type (f32, etc.)
    /// - `variant`: ArrayVariantView / Composite / Virtual / Bounded,
    ///   AddressPlaceholder, or a type variable
    /// - `dim_i`: per-dimension size term — `Size(n)` / `SizeVar(name)` /
    ///   `SizePlaceholder` / `Skolem` / `Variable`
    /// - `region`: buffer region — `Region(set,binding)` (a storage view),
    ///   `NoBuffer` (a non-view array), `AddressPlaceholder` (pre-resolution),
    ///   or a type variable (a buffer-polymorphic view function). The region
    ///   makes a view's buffer binding a statically-known property of its type.
    ///
    /// Rank is implicit: `args.len() - 3` (region is the trailing arg). All
    /// arrays are rank-1 today; the variadic dim suffix is the entry point for
    /// multi-dim representations in a future effort.
    Array,
    /// Size placeholder (for []t syntax where size is inferred). Replaced with type variable before type checking.
    SizePlaceholder,
    /// Function arrow type constructor (T1 -> T2)
    Arrow,
    /// Vector type constructor (takes size and element type)
    Vec,
    /// Matrix type constructor (takes rows, columns, and element type)
    /// Corresponds to SPIR-V matrix types
    Mat,
    /// Array size literal
    Size(usize),
    /// Size variable: [n]
    SizeVar(String),
    /// Type variable from user code: 'a, 'b (not yet bound to TypeVar)
    UserVar(String),
    /// Type names parsed from source code (user-defined types, type aliases)
    Named(String),
    /// Record type: {field1: type1, field2: type2}
    /// Preserves source order of fields, but equality is order-independent
    Record(RecordFields),
    /// Unit type: () - the empty tuple, an honest value.
    Unit,
    /// "No return value" type. Distinct from `Unit`: `Unit` IS a value
    /// (the empty tuple, `()`); `SideEffect` is the absence of a return
    /// value. It types functions whose only meaningful output is
    /// observable through side effects — imperative builtins like
    /// storage-image updates / `storage_store`. Lowering treats this as `void`
    /// in the SPIR-V / WGSL backends.
    SideEffect,
    /// Tuple type with arity (size). Field types stored in Type::Constructed args.
    Tuple(usize),
    /// Sum type: Constructor1 type* | Constructor2 type*
    Sum(Vec<(String, Vec<Type>)>),
    /// Existential size: ?[n][m]. type
    /// Inner type is stored in Type::Constructed args[0], not in the TypeName.
    Existential(Vec<String>),
    // --- Reference types (MIR only) ---
    /// Pointer type - result of Materialize, used for indexing/access.
    /// Type args: [pointee_type, addrspace]
    Pointer,
    // --- Pointer address spaces (used as args[1] in Pointer types) ---
    /// Function-local pointer (SPIR-V StorageClass::Function).
    PointerFunction,
    /// Input variable pointer (SPIR-V StorageClass::Input).
    PointerInput,
    /// Output variable pointer (SPIR-V StorageClass::Output).
    PointerOutput,
    /// Storage buffer pointer (SPIR-V StorageClass::StorageBuffer).
    PointerStorage,

    // --- Array variants (how the array is represented at runtime) ---
    /// View variant - {ptr, len} struct pointing into storage buffer.
    ArrayVariantView,
    /// Composite variant - sized array value in registers/locals.
    ArrayVariantComposite,
    /// Virtual variant - computed on-the-fly (e.g., ranges). No storage.
    ArrayVariantVirtual,
    /// Bounded variant - function-local fixed-capacity buffer plus a
    /// runtime length. Runtime layout: `{buffer: [N]T, len: u32}` where
    /// `N` is the capacity (the upper bound, encoded as `Size(N)` in the
    /// array type's size slot) and `len ≤ N` is the actual count of
    /// valid elements. Produced by `filter` and similar SOACs whose
    /// output count is data-dependent but whose upper bound is
    /// statically known.
    ArrayVariantBounded,
    /// Representation-polymorphic array. A first-class member of the
    /// variant lattice — *not* a placeholder. Used by SOAC producers
    /// (`filter`, …) whose runtime representation depends on the input
    /// (Bounded for static-capacity inputs, View for runtime-sized).
    /// Operations on Abstract arrays — `length`, `index`, `reduce`,
    /// `scan`, `map`, `slice` — are well-typed in TLC; their backend
    /// lowering dispatches on the producer's chosen concrete variant.
    /// Survives `apply_subst` in monomorphize and helper signatures.
    /// No `Array[_, Abstract, _, _]` may reach SPIR-V or WGSL emission;
    /// `egir::verify_no_abstract` enforces this at the backend boundary.
    ArrayVariantAbstract,
    /// Array variant placeholder. Replaced with type variable before type checking.
    /// Entry point params are constrained to Storage, others remain polymorphic.
    AddressPlaceholder,
    /// Buffer region `(set, binding)` carried inside a `View` variant at the
    /// EGIR/SSA level: `Array[elem, ArrayVariantView[Region(set,binding)], dim]`.
    /// Introduced at EGIR lowering (where the binding is known), so the backend
    /// reads a view's descriptor off its type instead of a per-value side-map.
    /// Never appears in source/TLC types — views are nullary (`View[]`) there.
    Buffer(crate::BindingRef),
    /// Target-independent storage identity used by allocated semantic EGIR.
    /// Physicalization rewrites this to `Buffer` before SSA elaboration.
    Resource(crate::ResourceId),
    /// The buffer slot's value for a non-view array (composite/virtual/bounded):
    /// "this array is not buffer-backed, so it has no descriptor binding." A
    /// concrete tag (not a variable) so non-view arrays carry no free type var.
    NoBuffer,

    // --- Opaque GPU resources ---
    /// A 2D, float-sampled image. Nullary (no type args): the sampled
    /// type is fixed to `f32`, matching Wyn's `vec4f32`/`mat4f32`
    /// no-angle-bracket style. An opaque handle bound via
    /// `#[texture(set, binding)]`; read with `texture_load` /
    /// `texture_sample`.
    Texture2D,
    /// A filtering sampler. Nullary opaque handle bound via
    /// `#[sampler(set, binding)]`; paired with a `Texture2D` in
    /// `texture_sample`.
    Sampler,
    /// A 2D storage image with one hidden region argument — a texture a compute entry can write via
    /// linear `with` update (and point-read via `image_load`). Bound via
    /// `#[storage_image(set, binding, format, access)]`. Element type
    /// is fixed to vec4f32; the binding's `format` attribute decides
    /// the on-GPU pixel format. The same `(set, binding)` slot can be
    /// declared as `Texture2D` in another pipeline; viz binds the same
    /// underlying `wgpu::Texture` to both, so a compute-written
    /// storage image can be sampled bilinearly by a later fragment.
    /// The region is pinned from `#[storage_image]` and participates in
    /// monomorphization just like a storage-view array region.
    StorageTexture,

    // --- Type system internals ---
    /// Rigid skolem constant for existential sizes.
    /// Created when opening existential types (?k. T). Unlike unification variables,
    /// skolems only unify with themselves (same ID), enforcing opacity.
    Skolem(SkolemId),
}

impl std::fmt::Display for TypeName {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TypeName::Bool => write!(f, "bool"),
            TypeName::Float(bits) => write!(f, "f{}", bits),
            TypeName::UInt(bits) => write!(f, "u{}", bits),
            TypeName::Int(bits) => write!(f, "i{}", bits),
            TypeName::Array => write!(f, "Array"),
            TypeName::SizePlaceholder => write!(f, ""),
            TypeName::Arrow => write!(f, "->"),
            TypeName::Vec => write!(f, "Vec"),
            TypeName::Mat => write!(f, "Mat"),
            TypeName::Size(n) => write!(f, "{}", n),
            TypeName::SizeVar(name) => write!(f, "{}", name),
            TypeName::UserVar(name) => write!(f, "'{}", name),
            TypeName::Named(name) => write!(f, "{}", name),
            TypeName::Record(fields) => {
                write!(f, "{{")?;
                for (i, name) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", name)?;
                }
                write!(f, "}}")
            }
            TypeName::Unit => write!(f, "()"),
            TypeName::SideEffect => write!(f, "!()"),
            TypeName::Tuple(n) => write!(f, "Tuple({})", n),
            TypeName::Sum(variants) => {
                for (i, (name, types)) in variants.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{}", name)?;
                    for ty in types {
                        write!(f, " {}", ty)?;
                    }
                }
                Ok(())
            }
            TypeName::Existential(vars) => {
                // Inner type is in type_args[0], not here
                write!(f, "?{}.", vars.join(" "))
            }
            TypeName::Pointer => write!(f, "Ptr"),
            TypeName::PointerFunction => write!(f, "PtrFunction"),
            TypeName::PointerInput => write!(f, "PtrInput"),
            TypeName::PointerOutput => write!(f, "PtrOutput"),
            TypeName::PointerStorage => write!(f, "PtrStorage"),
            TypeName::ArrayVariantView => write!(f, "view"),
            TypeName::ArrayVariantComposite => write!(f, "composite"),
            TypeName::ArrayVariantBounded => write!(f, "bounded"),
            TypeName::ArrayVariantVirtual => write!(f, "virtual"),
            TypeName::ArrayVariantAbstract => write!(f, "abstract"),
            TypeName::AddressPlaceholder => write!(f, "?addrspace"),
            TypeName::Buffer(b) => write!(f, "buffer(set={}, binding={})", b.set, b.binding),
            TypeName::Resource(resource) => write!(f, "resource({})", resource.0),
            TypeName::NoBuffer => write!(f, "no_buffer"),
            TypeName::Texture2D => write!(f, "texture2d"),
            TypeName::Sampler => write!(f, "sampler"),
            TypeName::StorageTexture => write!(f, "storage_image"),
            TypeName::Skolem(id) => write!(f, "{}", id),
        }
    }
}

impl polytype::Name for TypeName {
    fn arrow() -> Self {
        TypeName::Arrow
    }

    fn show(&self) -> String {
        match self {
            TypeName::Bool => "bool".to_string(),
            TypeName::Float(bits) => format!("f{}", bits),
            TypeName::UInt(bits) => format!("u{}", bits),
            TypeName::Int(bits) => format!("i{}", bits),
            TypeName::Array => "Array".to_string(),
            TypeName::SizePlaceholder => "".to_string(),
            TypeName::Arrow => "->".to_string(),
            TypeName::Vec => "Vec".to_string(),
            TypeName::Mat => "Mat".to_string(),
            TypeName::Size(n) => format!("Size({})", n),
            TypeName::SizeVar(v) => format!("[{}]", v),
            TypeName::UserVar(v) => format!("'{}", v),
            TypeName::Named(name) => name.clone(),
            TypeName::Record(fields) => {
                let field_strs: Vec<String> = fields.iter().cloned().collect();
                format!("{{{}}}", field_strs.join(", "))
            }
            TypeName::Unit => "()".to_string(),
            TypeName::SideEffect => "!()".to_string(),
            TypeName::Tuple(n) => format!("Tuple({})", n),
            TypeName::Sum(variants) => {
                let variant_strs: Vec<String> = variants
                    .iter()
                    .map(|(name, types)| {
                        if types.is_empty() {
                            name.clone()
                        } else {
                            format!(
                                "{} {}",
                                name,
                                types.iter().map(|t| format!("{}", t)).collect::<Vec<_>>().join(" ")
                            )
                        }
                    })
                    .collect();
                variant_strs.join(" | ")
            }
            TypeName::Existential(vars) => format!("?{}.", vars.join(" ")),
            TypeName::Pointer => "Ptr".to_string(),
            TypeName::PointerFunction => "PtrFunction".to_string(),
            TypeName::PointerInput => "PtrInput".to_string(),
            TypeName::PointerOutput => "PtrOutput".to_string(),
            TypeName::PointerStorage => "PtrStorage".to_string(),
            TypeName::ArrayVariantView => "view".to_string(),
            TypeName::ArrayVariantComposite => "composite".to_string(),
            TypeName::ArrayVariantVirtual => "virtual".to_string(),
            TypeName::ArrayVariantBounded => "bounded".to_string(),
            TypeName::ArrayVariantAbstract => "abstract".to_string(),
            TypeName::AddressPlaceholder => "?variant".to_string(),
            TypeName::Buffer(b) => format!("buffer_s{}_b{}", b.set, b.binding),
            TypeName::Resource(resource) => format!("resource_{}", resource.0),
            TypeName::NoBuffer => "no_buffer".to_string(),
            TypeName::Texture2D => "texture2d".to_string(),
            TypeName::Sampler => "sampler".to_string(),
            TypeName::StorageTexture => "storage_image".to_string(),
            TypeName::Skolem(id) => format!("{}", id),
        }
    }
}

// =============================================================================
// Type extension traits
// =============================================================================

/// Extension trait for common type operations.
///
/// Centralizes type queries so passes don't need to pattern-match
/// on TypeName variants directly.
///
/// ## Constructed type argument layouts
///
/// | Type    | args[0]       | args[1]          | args[2..]                |
/// |---------|---------------|------------------|--------------------------|
/// | `Vec`   | elem_type     | Size(n)          | —                        |
/// | `Mat`   | elem_type     | Size(cols)       | Size(rows)               |
/// | `Array` | elem_type     | variant          | dim_0, dim_1, …, dim_R-1 |
///
/// Array rank is implicit: `args.len() - 2`. All arrays are rank-1
/// today; the variadic dim suffix is the entry point for multi-dim
/// representations in a future effort.
pub trait TypeExt {
    /// Check if this type is an array type
    fn is_array(&self) -> bool;

    /// Check if this type is a matrix type
    fn is_mat(&self) -> bool;

    /// Check if this type is a vector type
    fn is_vec(&self) -> bool;

    /// Check if this type is a scalar numeric type (float, int, or uint)
    fn is_scalar(&self) -> bool;

    /// Get the element type (args[0]) of a Vec, Mat, or Array.
    fn elem_type(&self) -> Option<&Type>;

    /// Get the vector component count, or None if not a Vec with concrete size.
    fn vec_size(&self) -> Option<usize>;

    /// Get the vector size as a raw type (args[1]).
    fn vec_size_type(&self) -> Option<&Type>;

    /// Get the matrix column count, or None if not a Mat with concrete size.
    fn mat_cols(&self) -> Option<usize>;

    /// Get the matrix row count, or None if not a Mat with concrete size.
    fn mat_rows(&self) -> Option<usize>;

    /// Get the matrix cols as a raw type (args[1]).
    fn mat_cols_type(&self) -> Option<&Type>;

    /// Get the matrix rows as a raw type (args[2]).
    fn mat_rows_type(&self) -> Option<&Type>;

    /// Get the array's variant type (`args[1]`), or `None` if not an
    /// Array.
    fn array_variant(&self) -> Option<&Type>;

    /// Get the array's first (and, today, only) dimension size
    /// (`args[2]`), or `None` if not an Array. Equivalent to
    /// `array_dim(0)`; kept as the dominant accessor while all arrays
    /// are rank-1.
    fn array_size(&self) -> Option<&Type>;

    /// Number of array dimensions (`args.len() - 2`), or `None` if not
    /// an Array. Always 1 today; the variadic dim suffix is the entry
    /// point for multi-dim representations.
    fn array_rank(&self) -> Option<usize>;

    /// Per-dimension sizes (`&args[2..]`), or `None` if not an Array.
    /// One entry today (rank-1).
    fn array_dims(&self) -> Option<&[Type]>;

    /// Size of the i-th dimension (`&args[2 + i]`), or `None` if not
    /// an Array or if `i >= rank`.
    fn array_dim(&self, i: usize) -> Option<&Type>;

    /// The array's buffer-region type (the trailing arg), or `None` if not an
    /// Array. For a View this is `Region(set,binding)` (or a buffer variable);
    /// for non-view arrays it is `NoBuffer`.
    fn array_buffer(&self) -> Option<&Type>;
}

impl TypeExt for Type {
    fn is_array(&self) -> bool {
        matches!(self, Type::Constructed(TypeName::Array, _))
    }

    fn is_mat(&self) -> bool {
        matches!(self, Type::Constructed(TypeName::Mat, _))
    }

    fn is_vec(&self) -> bool {
        matches!(self, Type::Constructed(TypeName::Vec, _))
    }

    fn is_scalar(&self) -> bool {
        matches!(
            self,
            Type::Constructed(TypeName::Float(_) | TypeName::Int(_) | TypeName::UInt(_), _)
        )
    }

    fn elem_type(&self) -> Option<&Type> {
        match self {
            // All three have elem_type at args[0]
            Type::Constructed(TypeName::Vec, args) => {
                debug_assert_eq!(
                    args.len(),
                    2,
                    "Vec must have exactly 2 args [elem, size], got {:?}",
                    args
                );
                debug_assert!(
                    !matches!(&args[0], Type::Constructed(TypeName::Size(_), _)),
                    "Vec args[0] looks like a Size, expected elem type: {:?}",
                    args
                );
                args.first()
            }
            Type::Constructed(TypeName::Mat, args) => {
                debug_assert_eq!(
                    args.len(),
                    3,
                    "Mat must have exactly 3 args [elem, cols, rows], got {:?}",
                    args
                );
                debug_assert!(
                    !matches!(&args[0], Type::Constructed(TypeName::Size(_), _)),
                    "Mat args[0] looks like a Size, expected elem type: {:?}",
                    args
                );
                args.first()
            }
            Type::Constructed(TypeName::Array, args) => {
                debug_assert_array_args(args);
                args.first()
            }
            _ => None,
        }
    }

    fn vec_size(&self) -> Option<usize> {
        if let Type::Constructed(TypeName::Vec, args) = self {
            debug_assert_eq!(
                args.len(),
                2,
                "Vec must have exactly 2 args [elem, size], got {:?}",
                args
            );
            if let Type::Constructed(TypeName::Size(n), _) = &args[1] {
                return Some(*n);
            }
        }
        None
    }

    fn vec_size_type(&self) -> Option<&Type> {
        if let Type::Constructed(TypeName::Vec, args) = self {
            debug_assert_eq!(
                args.len(),
                2,
                "Vec must have exactly 2 args [elem, size], got {:?}",
                args
            );
            return Some(&args[1]);
        }
        None
    }

    fn mat_cols(&self) -> Option<usize> {
        if let Type::Constructed(TypeName::Mat, args) = self {
            debug_assert_eq!(
                args.len(),
                3,
                "Mat must have exactly 3 args [elem, cols, rows], got {:?}",
                args
            );
            debug_assert!(
                matches!(
                    &args[1],
                    Type::Constructed(TypeName::Size(_), _) | Type::Variable(_)
                ),
                "Mat args[1] should be Size or Variable (cols), got {:?}",
                args[1]
            );
            if let Type::Constructed(TypeName::Size(n), _) = &args[1] {
                return Some(*n);
            }
        }
        None
    }

    fn mat_rows(&self) -> Option<usize> {
        if let Type::Constructed(TypeName::Mat, args) = self {
            debug_assert_eq!(
                args.len(),
                3,
                "Mat must have exactly 3 args [elem, cols, rows], got {:?}",
                args
            );
            debug_assert!(
                matches!(
                    &args[2],
                    Type::Constructed(TypeName::Size(_), _) | Type::Variable(_)
                ),
                "Mat args[2] should be Size or Variable (rows), got {:?}",
                args[2]
            );
            if let Type::Constructed(TypeName::Size(n), _) = &args[2] {
                return Some(*n);
            }
        }
        None
    }

    fn mat_cols_type(&self) -> Option<&Type> {
        if let Type::Constructed(TypeName::Mat, args) = self {
            debug_assert_eq!(
                args.len(),
                3,
                "Mat must have exactly 3 args [elem, cols, rows], got {:?}",
                args
            );
            return Some(&args[1]);
        }
        None
    }

    fn mat_rows_type(&self) -> Option<&Type> {
        if let Type::Constructed(TypeName::Mat, args) = self {
            debug_assert_eq!(
                args.len(),
                3,
                "Mat must have exactly 3 args [elem, cols, rows], got {:?}",
                args
            );
            return Some(&args[2]);
        }
        None
    }

    fn array_size(&self) -> Option<&Type> {
        if let Type::Constructed(TypeName::Array, args) = self {
            debug_assert_array_args(args);
            return Some(&args[2]);
        }
        None
    }

    fn array_variant(&self) -> Option<&Type> {
        if let Type::Constructed(TypeName::Array, args) = self {
            debug_assert_array_args(args);
            return Some(&args[1]);
        }
        None
    }

    fn array_rank(&self) -> Option<usize> {
        if let Type::Constructed(TypeName::Array, args) = self {
            debug_assert_array_args(args);
            // [elem, variant, dim_0, …, dim_{rank-1}, region] → rank = len - 3.
            return Some(args.len() - 3);
        }
        None
    }

    fn array_dims(&self) -> Option<&[Type]> {
        if let Type::Constructed(TypeName::Array, args) = self {
            debug_assert_array_args(args);
            // Dims are the variadic middle; the trailing arg is the region.
            return Some(&args[2..args.len() - 1]);
        }
        None
    }

    fn array_dim(&self, i: usize) -> Option<&Type> {
        self.array_dims().and_then(|dims| dims.get(i))
    }

    fn array_buffer(&self) -> Option<&Type> {
        if let Type::Constructed(TypeName::Array, args) = self {
            debug_assert_array_args(args);
            return args.last();
        }
        None
    }
}

// =============================================================================
// Type helper functions
// =============================================================================

pub fn i32() -> Type {
    Type::Constructed(TypeName::Int(32), vec![])
}

pub fn f32() -> Type {
    Type::Constructed(TypeName::Float(32), vec![])
}

pub fn bool_type() -> Type {
    Type::Constructed(TypeName::Bool, vec![])
}

pub fn unit() -> Type {
    Type::Constructed(TypeName::Unit, vec![])
}

/// All valid SPIR-V scalar element types for vectors and matrices
fn spirv_element_types() -> Vec<(&'static str, Type)> {
    vec![
        ("i8", Type::Constructed(TypeName::Int(8), vec![])),
        ("i16", Type::Constructed(TypeName::Int(16), vec![])),
        ("i32", Type::Constructed(TypeName::Int(32), vec![])),
        ("i64", Type::Constructed(TypeName::Int(64), vec![])),
        ("u8", Type::Constructed(TypeName::UInt(8), vec![])),
        ("u16", Type::Constructed(TypeName::UInt(16), vec![])),
        ("u32", Type::Constructed(TypeName::UInt(32), vec![])),
        ("u64", Type::Constructed(TypeName::UInt(64), vec![])),
        ("f16", Type::Constructed(TypeName::Float(16), vec![])),
        ("f32", Type::Constructed(TypeName::Float(32), vec![])),
        ("f64", Type::Constructed(TypeName::Float(64), vec![])),
        ("bool", Type::Constructed(TypeName::Bool, vec![])),
    ]
}

// Vector types
/// Create a vector type: Vec(size, element_type)
/// Example: vec(2, f32()) creates Vec(Size(2), f32) for vec2
pub fn vec(size: usize, element_type: Type) -> Type {
    Type::Constructed(
        TypeName::Vec,
        vec![element_type, Type::Constructed(TypeName::Size(size), vec![])],
    )
}

/// Generate all vector type constructors using cartesian product of sizes and element types
/// Returns a LookupMap mapping names like "vec2f32", "vec3i32", "vec4bool" to their Type representations
pub fn vector_type_constructors() -> LookupMap<String, Type> {
    use itertools::Itertools;

    let sizes = [2, 3, 4];
    let elem_types = spirv_element_types();

    sizes
        .iter()
        .cartesian_product(elem_types.iter())
        .map(|(size, (elem_name, elem_type))| {
            let name = format!("vec{}{}", size, elem_name);
            let vec_type = Type::Constructed(
                TypeName::Vec,
                vec![
                    elem_type.clone(),
                    Type::Constructed(TypeName::Size(*size), vec![]),
                ],
            );
            (name, vec_type)
        })
        .collect()
}

// Matrix types - column-major, CxR naming
/// Create a matrix type: mat<rows, cols, elem_type>
pub fn mat(rows: usize, cols: usize, elem_type: Type) -> Type {
    Type::Constructed(
        TypeName::Mat,
        vec![
            elem_type,
            Type::Constructed(TypeName::Size(cols), vec![]),
            Type::Constructed(TypeName::Size(rows), vec![]),
        ],
    )
}

/// Generate all matrix type constructors using cartesian product of dimensions and element types
/// Returns a LookupMap mapping names like "mat2f32", "mat3x4i32" to their Type representations
pub fn matrix_type_constructors() -> LookupMap<String, Type> {
    use itertools::Itertools;

    let dims = [2, 3, 4];
    let elem_types = spirv_element_types();

    dims.iter()
        .cartesian_product(dims.iter())
        .cartesian_product(elem_types.iter())
        .flat_map(|((rows, cols), (elem_name, elem_type))| {
            let matrix_type = Type::Constructed(
                TypeName::Mat,
                vec![
                    elem_type.clone(),
                    Type::Constructed(TypeName::Size(*cols), vec![]),
                    Type::Constructed(TypeName::Size(*rows), vec![]),
                ],
            );

            if rows == cols {
                // Square matrices: add both matNf32 and matNxNf32 as aliases
                vec![
                    (format!("mat{}{}", rows, elem_name), matrix_type.clone()),
                    (format!("mat{}x{}{}", rows, cols, elem_name), matrix_type),
                ]
            } else {
                // Non-square matrices: only matRxCf32
                vec![(format!("mat{}x{}{}", rows, cols, elem_name), matrix_type)]
            }
        })
        .collect()
}

/// A concrete buffer-buffer tag `Buffer(set, binding)` for a View's buffer slot.
pub fn buffer_tag(binding: crate::BindingRef) -> Type {
    Type::Constructed(TypeName::Buffer(binding), vec![])
}

/// The `NoBuffer` tag — the buffer slot of a non-view array.
pub fn no_buffer() -> Type {
    Type::Constructed(TypeName::NoBuffer, vec![])
}

/// If `ty` is an array, return a copy with its buffer slot (the last arg)
/// replaced by `region`; otherwise return `ty` unchanged. Used where the
/// buffer authority is the buffer binding (the view-interning helpers): the
/// binding stamps the region into an array-typed view. A `StorageView`'s node
/// type is the *element* type in output-slot paths and the *array* type in
/// input-view paths; only the latter carries a region (and is what flows
/// through block params / slices), so a non-array view type is left as-is —
/// its binding rides the op tag locally at the point of creation.
pub fn array_with_buffer(ty: &Type, region: Type) -> Type {
    match ty {
        Type::Constructed(TypeName::Array, args) => {
            debug_assert_array_args(args);
            let mut args = args.clone();
            *args.last_mut().expect("array has >= 4 args") = region;
            Type::Constructed(TypeName::Array, args)
        }
        _ => ty.clone(),
    }
}

/// A view-array type carrying `region`, for a `StorageView`'s result type. If
/// `view_ty` is already an array, its buffer slot is set to `region`;
/// otherwise `view_ty` is the *element* type and is wrapped into
/// `Array[view_ty, View, SizePlaceholder, region]`. Either way the resulting
/// view value's type carries its buffer binding, so the backend recovers the
/// descriptor from the type instead of a side-map.
pub fn view_array_of(view_ty: &Type, region: Type) -> Type {
    match view_ty {
        Type::Constructed(TypeName::Array, _) => array_with_buffer(view_ty, region),
        elem => make_array1(
            elem.clone(),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            region,
        ),
    }
}

/// Read the concrete buffer off an array type's buffer slot, if it has
/// a concrete `Region` (not `NoBuffer` and not a variable).
pub fn array_view_buffer(ty: &Type) -> Option<crate::BindingRef> {
    match ty.array_buffer()? {
        Type::Constructed(TypeName::Buffer(b), _) => Some(*b),
        _ => None,
    }
}

/// Concrete descriptor carried by a monomorphized storage-image type.
/// Storage images have no runtime payload; this region is their complete
/// identity during EGIR and backend lowering.
pub fn storage_image_buffer(ty: &Type) -> Option<crate::BindingRef> {
    match ty {
        Type::Constructed(TypeName::StorageTexture, args) => match args.first() {
            Some(Type::Constructed(TypeName::Buffer(binding), _)) => Some(*binding),
            _ => None,
        },
        _ => None,
    }
}

/// Build a rank-1 `Array[elem, variant, size, region]`, validating the kind
/// schema in debug builds so a mis-ordered/short construction fails loudly
/// instead of silently producing a type that won't unify.
pub fn make_array1(elem: Type, variant: Type, size: Type, region: Type) -> Type {
    let args = vec![elem, variant, size, region];
    debug_assert!(
        validate_type_args(&TypeName::Array, &args).is_ok(),
        "malformed Array construction: {}",
        validate_type_args(&TypeName::Array, &args).unwrap_err()
    );
    Type::Constructed(TypeName::Array, args)
}

/// Create a sized array: `Array[elem, Composite, Size(n), NoBuffer]`. Defaults
/// to Composite variant for local/value arrays (not buffer-backed).
pub fn sized_array(size: usize, elem_type: Type) -> Type {
    make_array1(
        elem_type,
        Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
        Type::Constructed(TypeName::Size(size), vec![]),
        no_buffer(),
    )
}

/// Create a sized array with placeholder address space: `Array[elem,
/// AddressPlaceholder, Size(n)]`. Used for parser tests where the
/// address space hasn't been resolved yet.
pub fn sized_array_placeholder(size: usize, elem_type: Type) -> Type {
    make_array1(
        elem_type,
        Type::Constructed(TypeName::AddressPlaceholder, vec![]),
        Type::Constructed(TypeName::Size(size), vec![]),
        Type::Constructed(TypeName::AddressPlaceholder, vec![]),
    )
}

pub fn tuple(types: Vec<Type>) -> Type {
    if types.is_empty() {
        unit()
    } else {
        Type::Constructed(TypeName::Tuple(types.len()), types)
    }
}

pub fn function(arg: Type, ret: Type) -> Type {
    Type::arrow(arg, ret)
}

/// Destructure an arrow type into (param_type, result_type)
/// Returns None if the type is not an arrow type
pub fn as_arrow(ty: &Type) -> Option<(&Type, &Type)> {
    match ty {
        Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => Some((&args[0], &args[1])),
        _ => None,
    }
}

/// Count the number of real arguments a function type expects
/// e.g., (a -> b -> c) expects 2 args, (a -> b) expects 1, a expects 0
/// Note: (unit -> a) expects 0 args (unit param is implicit)
pub fn count_arrows(ty: &Type) -> usize {
    match as_arrow(ty) {
        Some((param, result)) => {
            // Unit parameter doesn't count as a real argument
            let param_count = match param {
                Type::Constructed(TypeName::Unit, _) => 0,
                _ => 1,
            };
            param_count + count_arrows(result)
        }
        None => 0,
    }
}

/// Check if a type is an integer type (signed or unsigned)
/// Per spec: array indices may be "any unsigned integer type",
/// but we also accept signed integers for compatibility
pub fn is_integer_type(ty: &Type) -> bool {
    matches!(ty, Type::Constructed(TypeName::Int(_) | TypeName::UInt(_), _))
}

/// Create a record type: {field1: type1, field2: type2}
pub fn record(fields: Vec<(String, Type)>) -> Type {
    let (field_names, field_types): (Vec<String>, Vec<Type>) = fields.into_iter().unzip();
    Type::Constructed(TypeName::Record(RecordFields::new(field_names)), field_types)
}

/// Create a sum type: Constructor1 type* | Constructor2 type*
pub fn sum(variants: Vec<(String, Vec<Type>)>) -> Type {
    Type::Constructed(TypeName::Sum(variants), vec![])
}

/// Create an existential size type: ?k. type or ?k l. type
pub fn existential(size_vars: Vec<String>, inner: Type) -> Type {
    Type::Constructed(TypeName::Existential(size_vars), vec![inner])
}

/// Create a size variable in array types: [n]
pub fn size_var(name: String) -> Type {
    Type::Constructed(TypeName::SizeVar(name), vec![])
}

/// The consumption ("diet") of a value, mirroring a type's structure but
/// carrying only its `*` markers. Uniqueness lives here, on signatures —
/// never inside `Type`. A leaf holds one `*` bit; an aggregate holds a
/// `*` bit for the whole plus a per-component diet; an arrow holds its
/// parameter and result diets, so a `*` behind an arrow (a consuming
/// function value) is still visible for the value-function ban.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Diet {
    /// A scalar / array / opaque leaf. `true` means `*` (consuming /
    /// alias-free) here.
    Leaf(bool),
    /// A tuple or record. `unique` is `*` on the whole aggregate;
    /// `components` are the per-field diets.
    Aggregate {
        unique: bool,
        components: Vec<Diet>,
    },
    /// A function type: `(param_diet, result_diet)`.
    Arrow(Box<Diet>, Box<Diet>),
}

impl Default for Diet {
    fn default() -> Self {
        Diet::Leaf(false)
    }
}

impl Diet {
    /// An all-observing diet (no `*` anywhere). Structure is irrelevant to
    /// the consuming/observing queries, so a leaf suffices.
    pub fn observing() -> Diet {
        Diet::Leaf(false)
    }

    /// Mark this whole value consuming (the effect of a leading `*`): a
    /// `*` distributes over the entire inner value, so an aggregate
    /// becomes wholly consuming.
    pub fn into_consuming(self) -> Diet {
        match self {
            Diet::Leaf(_) => Diet::Leaf(true),
            Diet::Aggregate { components, .. } => Diet::Aggregate {
                unique: true,
                components,
            },
            Diet::Arrow(_, _) => Diet::Leaf(true),
        }
    }

    /// Whether this value's own storage is consumed / alias-free: a `*`
    /// in any value position, stopping at arrows (a callback that itself
    /// consumes does not make the callback value consuming). Mirrors the
    /// former `is_consuming_parameter_type`.
    pub fn is_consuming(&self) -> bool {
        match self {
            Diet::Leaf(u) => *u,
            Diet::Aggregate { unique, components } => *unique || components.iter().any(Diet::is_consuming),
            Diet::Arrow(_, _) => false,
        }
    }

    /// Whether a `*` appears inside a function type here — a consuming
    /// function used as a value. Mirrors the former `arrow_contains_unique`.
    pub fn mentions_consuming_function(&self) -> bool {
        fn any_star(d: &Diet) -> bool {
            match d {
                Diet::Leaf(u) => *u,
                Diet::Aggregate { unique, components } => *unique || components.iter().any(any_star),
                Diet::Arrow(p, r) => any_star(p) || any_star(r),
            }
        }
        match self {
            Diet::Arrow(p, r) => any_star(p) || any_star(r),
            Diet::Aggregate { components, .. } => components.iter().any(Diet::mentions_consuming_function),
            Diet::Leaf(_) => false,
        }
    }

    /// The diet of the `i`th component of an aggregate; an observing leaf
    /// for anything else (so callers can descend uniformly).
    pub fn component(&self, i: usize) -> Diet {
        match self {
            Diet::Aggregate { components, .. } => components.get(i).cloned().unwrap_or(Diet::Leaf(false)),
            _ => Diet::Leaf(false),
        }
    }

    /// The result diet of an arrow; observing for anything else.
    pub fn arrow_result(&self) -> Diet {
        match self {
            Diet::Arrow(_, r) => (**r).clone(),
            _ => Diet::Leaf(false),
        }
    }

    /// Whether the value at this node (not its components) is marked `*` —
    /// a `*` on the whole leaf or the whole aggregate.
    pub fn is_consuming_at_root(&self) -> bool {
        match self {
            Diet::Leaf(u) => *u,
            Diet::Aggregate { unique, .. } => *unique,
            Diet::Arrow(_, _) => false,
        }
    }
}

/// Whether a parameter's diet carries a consuming `*` contract.
pub fn is_consuming_parameter_type(diet: &Diet) -> bool {
    diet.is_consuming()
}

/// Whether a type's runtime values are pure value-semantics (cheap to
/// copy, no backing-store identity). The complement is "non-copy" — the
/// alias and ownership systems track only non-copy values.
pub fn is_copy(ty: &Type) -> bool {
    match ty {
        Type::Constructed(name, args) => match name {
            TypeName::Int(_)
            | TypeName::UInt(_)
            | TypeName::Float(_)
            | TypeName::Bool
            | TypeName::Unit
            | TypeName::Vec
            | TypeName::Mat
            | TypeName::Arrow => true,
            TypeName::Array => false,
            TypeName::Tuple(_) => args.iter().all(is_copy),
            // Opaque GPU resource handles are not copyable values — they
            // must reach the backend as the original binding, not be
            // duplicated by ownership/move analysis.
            TypeName::Texture2D | TypeName::Sampler | TypeName::StorageTexture => false,
            _ => true,
        },
        Type::Variable(_) => true,
    }
}

/// Canonicalize a TLC-flavored array type for use as the declared
/// type of a storage-bound field (`EntryInput.ty`, `EntryOutput.ty`,
/// `StorageBindingDecl.elem_ty`). Strips top-level `Existential<_>`
/// (filter results and other size-quantified shapes whose size variable
/// the backend can't name).
///
/// The result is what the SPIR-V backend's `create_storage_buffer`
/// expects to receive: a concrete array shape it can compute an
/// element size for via `array_elem` + `type_byte_size`. Scalar /
/// vec / struct outputs that get packed into single-element runtime
/// arrays pass through unchanged.
pub fn canonical_storage_buffer_ty(ty: &Type) -> Type {
    match ty {
        Type::Constructed(TypeName::Existential(_), args) if !args.is_empty() => args[0].clone(),
        _ => ty.clone(),
    }
}

// --- Address space constructors ---

/// Create a storage address space type
pub fn array_variant_view() -> Type {
    Type::Constructed(TypeName::ArrayVariantView, vec![])
}

/// Create a function (local) address space type
pub fn array_variant_composite() -> Type {
    Type::Constructed(TypeName::ArrayVariantComposite, vec![])
}

/// Check if a type is the storage address space
pub fn is_array_variant_view(ty: &Type) -> bool {
    matches!(ty, Type::Constructed(TypeName::ArrayVariantView, _))
}

/// Check if a type is the function address space
pub fn is_array_variant_composite(ty: &Type) -> bool {
    matches!(ty, Type::Constructed(TypeName::ArrayVariantComposite, _))
}

/// Check if a type is a virtual array (computed on-the-fly, like ranges)
pub fn is_array_variant_virtual(ty: &Type) -> bool {
    matches!(ty, Type::Constructed(TypeName::ArrayVariantVirtual, _))
}

/// Create the Bounded array variant marker — a function-local
/// fixed-capacity buffer plus a runtime length.
pub fn array_variant_bounded() -> Type {
    Type::Constructed(TypeName::ArrayVariantBounded, vec![])
}

/// Check if a type is the Bounded array variant marker.
pub fn is_array_variant_bounded(ty: &Type) -> bool {
    matches!(ty, Type::Constructed(TypeName::ArrayVariantBounded, _))
}

/// Create the Abstract array variant marker — representation-polymorphic
/// at the TLC level, resolved to a concrete variant by the producer's
/// EGIR lowering. See `TypeName::ArrayVariantAbstract` for the full
/// semantics + invariant.
pub fn array_variant_abstract() -> Type {
    Type::Constructed(TypeName::ArrayVariantAbstract, vec![])
}

/// Check if a type is the Abstract array variant marker.
pub fn is_array_variant_abstract(ty: &Type) -> bool {
    matches!(ty, Type::Constructed(TypeName::ArrayVariantAbstract, _))
}

pub mod schema_validation;
pub use schema_validation::{
    debug_assert_array_args, validate_array_args, validate_type_args, validate_type_schema, ArgKind,
};

/// Get the array variant from an array type (returns the variant type argument)
pub fn get_array_variant(ty: &Type) -> Option<&TypeName> {
    match ty.array_variant()? {
        Type::Constructed(name, _) => Some(name),
        _ => None,
    }
}

/// Check if a type is an array variant marker (View, Composite, Virtual, or Placeholder).
/// These types should only appear as the second argument of Array types,
/// never as standalone expression types.
pub fn is_array_variant(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Constructed(
            TypeName::ArrayVariantView
                | TypeName::ArrayVariantComposite
                | TypeName::ArrayVariantVirtual
                | TypeName::ArrayVariantBounded
                | TypeName::AddressPlaceholder,
            _
        )
    )
}

/// Check if a type is an array with virtual variant (computed on-the-fly, like ranges/iota).
pub fn is_virtual_array(ty: &Type) -> bool {
    matches!(get_array_variant(ty), Some(TypeName::ArrayVariantVirtual))
}

/// Debug assertion that a type is valid as a top-level expression/variable type.
/// Panics if the type is a marker type that should only appear nested inside other types:
/// - Address space markers (AddressStorage, AddressFunction, AddressPlaceholder)
/// - Size markers (Size, SizePlaceholder, SizeVar)
#[inline]
pub fn debug_assert_top_level_type(ty: &Type, context: &str) {
    if !cfg!(debug_assertions) {
        return;
    }

    match ty {
        Type::Constructed(name, _) => match name {
            // Address space markers - only valid inside Array[elem, variant, dim_0, ...]
            // (Region lives one level deeper, inside the View variant's args.)
            TypeName::ArrayVariantView
            | TypeName::ArrayVariantComposite
            | TypeName::Buffer(_)
            | TypeName::Resource(_)
            | TypeName::NoBuffer
            | TypeName::AddressPlaceholder => {
                panic!(
                    "BUG: Address space type {:?} used as top-level type in {}. \
                     Address space markers should only appear inside Array[elem, variant, dim_0, ...].",
                    ty, context
                );
            }
            // Size markers - only valid inside Array, Vec, Mat types
            TypeName::Size(_) | TypeName::SizePlaceholder | TypeName::SizeVar(_) => {
                panic!(
                    "BUG: Size marker {:?} used as top-level type in {}. \
                     Size markers should only appear inside Array, Vec, or Mat types.",
                    ty, context
                );
            }
            _ => {}
        },
        Type::Variable(_) => {}
    }
}

// --- Pointer type helpers ---

/// Create a pointer type (MIR only): Ptr(pointee, addrspace)
pub fn pointer(pointee: Type, addrspace: Type) -> Type {
    Type::Constructed(TypeName::Pointer, vec![pointee, addrspace])
}

/// Check if a type is a pointer type
pub fn is_pointer(ty: &Type) -> bool {
    matches!(ty, Type::Constructed(TypeName::Pointer, _))
}

/// Get the pointee type from a pointer type, or None if not a pointer
pub fn pointee(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Constructed(TypeName::Pointer, args) if !args.is_empty() => Some(&args[0]),
        _ => None,
    }
}

/// Get the address space from a pointer type, or None if not a pointer
pub fn pointer_addrspace(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Constructed(TypeName::Pointer, args) if args.len() >= 2 => Some(&args[1]),
        _ => None,
    }
}

// --- Array type helpers ---
// Array[elem, variant, dim_0, ...] is the unified array type. Rank is
// implicit (args.len() - 2); all arrays are rank-1 today.

/// Create an unsized array (slice): `Array[elem, variant, SizePlaceholder]`
pub fn unsized_array(elem: Type, variant: Type, region: Type) -> Type {
    make_array1(
        elem,
        variant,
        Type::Constructed(TypeName::SizePlaceholder, vec![]),
        region,
    )
}

/// Check if a type is an unsized array (has SizePlaceholder as size arg)
pub fn is_unsized_array(ty: &Type) -> bool {
    ty.array_size()
        .map(|s| matches!(s, Type::Constructed(TypeName::SizePlaceholder, _)))
        .unwrap_or(false)
}

/// Get the element type from an Array, or None if not an Array
pub fn array_elem(ty: &Type) -> Option<&Type> {
    ty.elem_type().filter(|_| ty.is_array())
}

/// Get the address space (variant) from an Array, or None if not an Array
pub fn array_addrspace(ty: &Type) -> Option<&Type> {
    ty.array_variant()
}

/// Get the size from an Array, or None if not an Array
pub fn array_size(ty: &Type) -> Option<&Type> {
    ty.array_size()
}

/// Walk an arrow chain `P1 -> P2 -> ... -> Pn -> R` and return
/// `(vec![P1, P2, ..., Pn], R)`. For a non-arrow `ty`, returns
/// `(vec![], ty.clone())`. This is the single canonical helper for
/// peeling an entry def's signature into params plus its declared return,
/// including when EGIR wires entry parameters and output routes.
pub fn extract_function_signature(ty: &Type) -> (Vec<Type>, Type) {
    let mut params = Vec::new();
    let mut current = ty.clone();
    while let Type::Constructed(TypeName::Arrow, ref args) = current {
        if args.len() == 2 {
            params.push(args[0].clone());
            current = args[1].clone();
        } else {
            break;
        }
    }
    (params, current)
}

/// Check if a type is a storage array that requires BoundSlice-style access.
/// Storage arrays with unsized length need pointer-based indexing at runtime.
pub fn is_bound_slice_access(ty: &Type) -> bool {
    if !ty.is_array() {
        return false;
    }
    let is_storage = matches!(
        ty.array_variant().expect("Array has variant"),
        Type::Constructed(TypeName::ArrayVariantView, _)
    );
    let is_unsized = matches!(
        ty.array_size().expect("Array has size"),
        Type::Constructed(TypeName::SizePlaceholder, _)
    );
    is_storage && is_unsized
}

/// Format a type for display (standalone, without TypeChecker context)
pub fn format_type(ty: &Type) -> String {
    match ty {
        Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => {
            format!("{} -> {}", format_type(&args[0]), format_type(&args[1]))
        }
        Type::Constructed(TypeName::Tuple(_), args) => {
            let arg_strs: Vec<String> = args.iter().map(format_type).collect();
            format!("({})", arg_strs.join(", "))
        }
        // Array[elem, variant, dim_0, ...]
        _ if ty.is_array() => {
            let elem = format_type(ty.elem_type().expect("Array has elem"));
            let size = ty.array_size().expect("Array has size");
            match size {
                Type::Constructed(TypeName::Size(n), _) => format!("[{}]{}", n, elem),
                Type::Constructed(TypeName::SizePlaceholder, _) => format!("[]{}", elem),
                _ => format!("[{}]{}", format_type(size), elem),
            }
        }
        // Vec[elem, Size(n)] -> vecNelem (e.g., vec3f32)
        _ if ty.is_vec() => {
            let elem = format_type(ty.elem_type().expect("Vec has elem"));
            if let Some(n) = ty.vec_size() {
                format!("vec{}{}", n, elem)
            } else {
                format!(
                    "vec{}{}",
                    format_type(ty.vec_size_type().expect("Vec has size")),
                    elem
                )
            }
        }
        // Mat[elem, Size(cols), Size(rows)] -> matCxRelem (e.g., mat4x4f32)
        _ if ty.is_mat() => {
            let elem = format_type(ty.elem_type().expect("Mat has elem"));
            match (ty.mat_cols(), ty.mat_rows()) {
                (Some(cols), Some(rows)) => format!("mat{}x{}{}", cols, rows, elem),
                _ => format!(
                    "mat{}x{}{}",
                    format_type(ty.mat_cols_type().expect("Mat has cols")),
                    format_type(ty.mat_rows_type().expect("Mat has rows")),
                    elem
                ),
            }
        }
        Type::Constructed(name, args) if args.is_empty() => format!("{}", name),
        Type::Constructed(name, args) => {
            let arg_strs: Vec<String> = args.iter().map(format_type).collect();
            format!("{}[{}]", name, arg_strs.join(", "))
        }
        Type::Variable(id) => format!("?{}", id),
    }
}

/// Format a type scheme for display (standalone, without TypeChecker context)
pub fn format_scheme(scheme: &TypeScheme) -> String {
    match scheme {
        TypeScheme::Monotype(ty) => format_type(ty),
        TypeScheme::Polytype { variable, body } => {
            format!("∀{}. {}", variable, format_scheme(body))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn i32_ty() -> Type {
        Type::Constructed(TypeName::Int(32), vec![])
    }
    fn f32_ty() -> Type {
        Type::Constructed(TypeName::Float(32), vec![])
    }
    fn arrow(a: Type, b: Type) -> Type {
        Type::Constructed(TypeName::Arrow, vec![a, b])
    }

    fn hash_value<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn record_fields_hash_matches_order_independent_equality() {
        let left = RecordFields::new(vec!["x".to_string(), "y".to_string()]);
        let right = RecordFields::new(vec!["y".to_string(), "x".to_string()]);

        assert_eq!(left, right);
        assert_eq!(hash_value(&left), hash_value(&right));

        let duplicate = RecordFields::new(vec!["x".to_string(), "x".to_string()]);
        let distinct = RecordFields::new(vec!["x".to_string(), "y".to_string()]);

        assert_ne!(duplicate, distinct);
        assert_ne!(distinct, duplicate);
    }

    #[test]
    fn extract_function_signature_chains_arrows_in_order() {
        // `i32 -> f32 -> i32` -> ([i32, f32], i32)
        let ty = arrow(i32_ty(), arrow(f32_ty(), i32_ty()));
        let (params, ret) = extract_function_signature(&ty);
        assert_eq!(params, vec![i32_ty(), f32_ty()]);
        assert_eq!(ret, i32_ty());
    }

    #[test]
    fn extract_function_signature_on_non_arrow_returns_empty_params() {
        let (params, ret) = extract_function_signature(&i32_ty());
        assert!(params.is_empty());
        assert_eq!(ret, i32_ty());
    }

    fn rank1_arr(size: usize) -> Type {
        Type::Constructed(
            TypeName::Array,
            vec![
                f32_ty(),
                Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
                Type::Constructed(TypeName::Size(size), vec![]),
                no_buffer(),
            ],
        )
    }

    #[test]
    fn array_rank_reports_dim_count() {
        assert_eq!(rank1_arr(8).array_rank(), Some(1));
        assert_eq!(i32_ty().array_rank(), None);
    }

    #[test]
    fn array_dims_returns_dim_slice() {
        let arr = rank1_arr(8);
        let dims = arr.array_dims().unwrap();
        assert_eq!(dims.len(), 1);
        assert!(matches!(&dims[0], Type::Constructed(TypeName::Size(8), _)));
    }

    #[test]
    fn array_dim_indexes_dim_slice() {
        let arr = rank1_arr(8);
        assert!(matches!(
            arr.array_dim(0),
            Some(Type::Constructed(TypeName::Size(8), _))
        ));
        assert!(arr.array_dim(1).is_none());
    }

    #[test]
    fn array_dim_zero_matches_array_size() {
        let arr = rank1_arr(8);
        assert_eq!(arr.array_dim(0), arr.array_size());
    }

    #[test]
    fn diet_consuming_stops_at_function_arrows() {
        // A `*` nested in a tuple makes the whole aggregate consuming...
        let tuple_with_unique = Diet::Aggregate {
            unique: false,
            components: vec![Diet::Leaf(true), Diet::Leaf(false)],
        };
        assert!(tuple_with_unique.is_consuming());

        // ...but a `*` behind an arrow does not make the callback value
        // consuming (it is the callback's own parameter contract).
        let callback = Diet::Arrow(Box::new(Diet::Leaf(true)), Box::new(Diet::Leaf(false)));
        assert!(!callback.is_consuming());
        assert!(callback.mentions_consuming_function());
    }

    #[test]
    fn diet_observing_mentions_no_consuming_function() {
        let observing = Diet::Aggregate {
            unique: false,
            components: vec![Diet::Leaf(false), Diet::Leaf(false)],
        };
        assert!(!observing.is_consuming());
        assert!(!observing.mentions_consuming_function());
    }
}
