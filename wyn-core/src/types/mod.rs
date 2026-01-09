//! Type system definitions and utilities for Wyn.
//!
//! This module contains:
//! - `TypeName`: The type name constructors for the Wyn type system
//! - Helper functions for creating common types (i32, f32, vec, etc.)
//! - The type checker (`checker` submodule)

pub mod checker;

#[cfg(test)]
mod checker_tests;

use std::collections::HashMap;

// Type aliases for polytype types specialized to our TypeName
pub type Type = polytype::Type<TypeName>;
pub type TypeScheme = polytype::TypeScheme<TypeName>;

/// Record field names that preserve source order but have order-independent equality.
/// The actual field types are stored in Type::Constructed's type argument vector.
#[derive(Debug, Clone, Hash)]
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
        // Order-independent equality: check same field names exist
        if self.0.len() != other.0.len() {
            return false;
        }
        for name in &self.0 {
            if !other.0.contains(name) {
                return false;
            }
        }
        true
    }
}

impl Eq for RecordFields {}

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
/// - `Str`: Other primitive type names hardcoded in the compiler (e.g., "->", "bool")
///   Uses static strings for efficiency
/// - `Tuple`: Tuple type constructor with arity (number of fields)
/// - `Named`: Type names parsed from user source code (e.g., "vec3", "MyType")
///   Could refer to built-in types, type aliases, or user-defined types
///   Uses owned String since the name comes from parsed input
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeName {
    /// Primitive type names hardcoded in compiler: "->", "bool", etc.
    /// Numeric types use dedicated Float/Int/SInt variants instead.
    /// Tuples use the dedicated Tuple(usize) variant.
    Str(&'static str),
    /// Floating point types: f16, f32, f64, etc.
    Float(usize),
    /// Unsigned integer types: u8, u16, u32, u64
    UInt(usize),
    /// Signed integer types: i8, i16, i32, i64
    Int(usize),
    /// Array type constructor (takes size and element type)
    ValueArray,
    /// Unsized/anonymous array size placeholder (for []t syntax where size is inferred)
    Unsized,
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
    /// Uniqueness/consuming type marker (corresponds to "*" prefix)
    Unique,
    /// Record type: {field1: type1, field2: type2}
    /// Preserves source order of fields, but equality is order-independent
    Record(RecordFields),
    /// Unit type: () - the empty tuple, used for side-effect-only functions
    Unit,
    /// Tuple type with arity (size). Field types stored in Type::Constructed args.
    Tuple(usize),
    /// Sum type: Constructor1 type* | Constructor2 type*
    Sum(Vec<(String, Vec<Type>)>),
    /// Existential size: ?[n][m]. type
    /// Inner type is stored in Type::Constructed args[0], not in the TypeName.
    Existential(Vec<String>),
    /// Pointer type (MIR only) - result of Materialize, used for indexing/access.
    /// The pointee type is stored in Type::Constructed args.
    Pointer,
    /// Slice type with dynamic length (MIR only).
    /// Type args: [cap_type, elem_type] where cap is the backing buffer capacity.
    /// Runtime representation includes a dynamic length field.
    Slice,
    /// Rigid skolem constant for existential sizes.
    /// Created when opening existential types (?k. T). Unlike unification variables,
    /// skolems only unify with themselves (same ID), enforcing opacity.
    Skolem(SkolemId),
}

impl std::fmt::Display for TypeName {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TypeName::Str(s) => write!(f, "{}", s),
            TypeName::Float(bits) => write!(f, "f{}", bits),
            TypeName::UInt(bits) => write!(f, "u{}", bits),
            TypeName::Int(bits) => write!(f, "i{}", bits),
            TypeName::ValueArray => write!(f, "Array"),
            TypeName::Unsized => write!(f, ""),
            TypeName::Arrow => write!(f, "->"),
            TypeName::Vec => write!(f, "Vec"),
            TypeName::Mat => write!(f, "Mat"),
            TypeName::Size(n) => write!(f, "{}", n),
            TypeName::SizeVar(name) => write!(f, "{}", name),
            TypeName::UserVar(name) => write!(f, "'{}", name),
            TypeName::Named(name) => write!(f, "{}", name),
            TypeName::Unique => write!(f, "*"),
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
            TypeName::Slice => write!(f, "Slice"),
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
            TypeName::Str(s) => s.to_string(),
            TypeName::Float(bits) => format!("f{}", bits),
            TypeName::UInt(bits) => format!("u{}", bits),
            TypeName::Int(bits) => format!("i{}", bits),
            TypeName::ValueArray => "Array".to_string(),
            TypeName::Unsized => "".to_string(),
            TypeName::Arrow => "->".to_string(),
            TypeName::Vec => "Vec".to_string(),
            TypeName::Mat => "Mat".to_string(),
            TypeName::Size(n) => format!("Size({})", n),
            TypeName::SizeVar(v) => format!("[{}]", v),
            TypeName::UserVar(v) => format!("'{}", v),
            TypeName::Named(name) => name.clone(),
            TypeName::Unique => "*".to_string(),
            TypeName::Record(fields) => {
                let field_strs: Vec<String> = fields.iter().cloned().collect();
                format!("{{{}}}", field_strs.join(", "))
            }
            TypeName::Unit => "()".to_string(),
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
            TypeName::Slice => "Slice".to_string(),
            TypeName::Skolem(id) => format!("{}", id),
        }
    }
}

impl From<&'static str> for TypeName {
    fn from(s: &'static str) -> Self {
        TypeName::Str(s)
    }
}

// =============================================================================
// Type extension traits
// =============================================================================

/// Extension trait for common type operations.
///
/// Centralizes type queries so passes don't need to pattern-match
/// on TypeName variants directly.
pub trait TypeExt {
    /// Create a unique (consuming/alias-free) type: *T
    fn unique(inner: Type) -> Type;

    /// Check if this type is marked as unique/consuming
    fn is_unique(&self) -> bool;

    /// Get the inner type if this is a unique type, None otherwise
    fn as_unique_inner(&self) -> Option<&Type>;

    /// Strip the uniqueness marker if present, otherwise return self
    fn strip_unique(&self) -> &Type;

    /// Check if this type is an array type
    fn is_array(&self) -> bool;
}

impl TypeExt for Type {
    fn unique(inner: Type) -> Type {
        Type::Constructed(TypeName::Unique, vec![inner])
    }

    fn is_unique(&self) -> bool {
        matches!(self, Type::Constructed(TypeName::Unique, _))
    }

    fn as_unique_inner(&self) -> Option<&Type> {
        if let Type::Constructed(TypeName::Unique, args) = self {
            debug_assert_eq!(args.len(), 1, "Unique type must have exactly one inner type");
            args.first()
        } else {
            None
        }
    }

    fn strip_unique(&self) -> &Type {
        self.as_unique_inner().unwrap_or(self)
    }

    fn is_array(&self) -> bool {
        matches!(self, Type::Constructed(TypeName::ValueArray, _))
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
    Type::Constructed(TypeName::Str("bool"), vec![])
}

pub fn string() -> Type {
    Type::Constructed(TypeName::Str("string"), vec![])
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
        ("bool", Type::Constructed(TypeName::Str("bool"), vec![])),
    ]
}

// Vector types
/// Create a vector type: Vec(size, element_type)
/// Example: vec(2, f32()) creates Vec(Size(2), f32) for vec2
pub fn vec(size: usize, element_type: Type) -> Type {
    Type::Constructed(
        TypeName::Vec,
        vec![Type::Constructed(TypeName::Size(size), vec![]), element_type],
    )
}

/// Generate all vector type constructors using cartesian product of sizes and element types
/// Returns a HashMap mapping names like "vec2f32", "vec3i32", "vec4bool" to their Type representations
pub fn vector_type_constructors() -> HashMap<String, Type> {
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
                    Type::Constructed(TypeName::Size(*size), vec![]),
                    elem_type.clone(),
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
            Type::Constructed(TypeName::Size(rows), vec![]),
            Type::Constructed(TypeName::Size(cols), vec![]),
            elem_type,
        ],
    )
}

/// Generate all matrix type constructors using cartesian product of dimensions and element types
/// Returns a HashMap mapping names like "mat2f32", "mat3x4i32" to their Type representations
pub fn matrix_type_constructors() -> HashMap<String, Type> {
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
                    Type::Constructed(TypeName::Size(*rows), vec![]),
                    Type::Constructed(TypeName::Size(*cols), vec![]),
                    elem_type.clone(),
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

pub fn sized_array(size: usize, elem_type: Type) -> Type {
    Type::Constructed(
        TypeName::ValueArray,
        vec![Type::Constructed(TypeName::Size(size), vec![]), elem_type],
    )
}

pub fn tuple(types: Vec<Type>) -> Type {
    if types.is_empty() { unit() } else { Type::Constructed(TypeName::Tuple(types.len()), types) }
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
    match ty {
        Type::Constructed(TypeName::Str(name), args) if args.is_empty() => {
            matches!(*name, "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64")
        }
        _ => false,
    }
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

/// Create a unique (consuming/alias-free) type: *t
pub fn unique(inner: Type) -> Type {
    Type::Constructed(TypeName::Unique, vec![inner])
}

/// Check if a type is marked as unique/consuming
pub fn is_unique(ty: &Type) -> bool {
    matches!(ty, Type::Constructed(TypeName::Unique, _))
}

/// Strip uniqueness marker from a type, returning the inner type
pub fn strip_unique(ty: &Type) -> Type {
    match ty {
        Type::Constructed(TypeName::Unique, args) => {
            // Recursively strip from the inner type
            let inner = args.first().cloned().unwrap_or_else(|| ty.clone());
            strip_unique(&inner)
        }
        Type::Constructed(name, args) => {
            // Recursively strip from constructor arguments (e.g., arrow types)
            let stripped_args: Vec<Type> = args.iter().map(strip_unique).collect();
            Type::Constructed(name.clone(), stripped_args)
        }
        _ => ty.clone(),
    }
}

/// Create a pointer type (MIR only): Ptr(inner)
pub fn pointer(inner: Type) -> Type {
    Type::Constructed(TypeName::Pointer, vec![inner])
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

/// Create a slice type (MIR only): Slice(cap, elem)
/// cap is the capacity type (Size(n) or SizeVar), elem is the element type.
pub fn slice(cap: Type, elem: Type) -> Type {
    Type::Constructed(TypeName::Slice, vec![cap, elem])
}

/// Check if a type is a slice type
pub fn is_slice(ty: &Type) -> bool {
    matches!(ty, Type::Constructed(TypeName::Slice, _))
}

/// Get the element type from a slice type, or None if not a slice
pub fn slice_elem(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Constructed(TypeName::Slice, args) if args.len() >= 2 => Some(&args[1]),
        _ => None,
    }
}

/// Get the capacity type from a slice type, or None if not a slice
pub fn slice_cap(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Constructed(TypeName::Slice, args) if !args.is_empty() => Some(&args[0]),
        _ => None,
    }
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
        Type::Constructed(TypeName::ValueArray, args) if args.len() == 2 => {
            format!("[{}]{}", format_type(&args[0]), format_type(&args[1]))
        }
        // Vec[Size(n), elem] -> vecNelem (e.g., vec3f32)
        Type::Constructed(TypeName::Vec, args) if args.len() == 2 => {
            if let Type::Constructed(TypeName::Size(n), _) = &args[0] {
                format!("vec{}{}", n, format_type(&args[1]))
            } else {
                format!("vec{}{}", format_type(&args[0]), format_type(&args[1]))
            }
        }
        // Mat[Size(cols), Size(rows), elem] -> matCxRelem (e.g., mat4x4f32)
        Type::Constructed(TypeName::Mat, args) if args.len() == 3 => match (&args[0], &args[1]) {
            (Type::Constructed(TypeName::Size(cols), _), Type::Constructed(TypeName::Size(rows), _)) => {
                format!("mat{}x{}{}", cols, rows, format_type(&args[2]))
            }
            _ => {
                format!(
                    "mat{}x{}{}",
                    format_type(&args[0]),
                    format_type(&args[1]),
                    format_type(&args[2])
                )
            }
        },
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
