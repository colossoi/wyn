// =============================================================================
// Type-argument schema validation
//
// polytype is single-kinded: every argument of a `Type::Constructed` is just a
// `Type`, with no static guarantee that, say, an `Array`'s second argument is a
// *variant* and not a *size*. We encode each constructor's argument layout as a
// sequence of `ArgKind`s and validate built types against it, so a mis-ordered
// or wrong-arity construction fails loudly instead of silently producing a type
// that won't unify (or that a positional accessor misreads).
// =============================================================================

use super::{Type, TypeName};

/// The kind a `Type::Constructed` argument occupies. A `Type::Variable` is
/// kind-unknown and accepted in any position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgKind {
    /// An inhabited type (element types, tuple fields, fn params/returns).
    Type,
    /// An array representation variant (View/Composite/Virtual/Bounded or the
    /// `AddressPlaceholder` standing in for one).
    Variant,
    /// An array dimension size (`Size`/`SizeVar`/`SizePlaceholder`/`Skolem`).
    Size,
    /// A buffer region (`Region` / `NoBuffer`).
    Region,
    /// A pointer address space.
    AddrSpace,
}

/// The kind a concrete `TypeName` occupies when it appears as an argument. The
/// leaf marker names have a fixed non-`Type` kind; everything inhabited is
/// `Type`.
fn arg_kind_of_name(name: &TypeName) -> ArgKind {
    match name {
        TypeName::ArrayVariantView
        | TypeName::ArrayVariantComposite
        | TypeName::ArrayVariantVirtual
        | TypeName::ArrayVariantBounded
        | TypeName::ArrayVariantAbstract
        | TypeName::AddressPlaceholder => ArgKind::Variant,
        TypeName::Size(_) | TypeName::SizeVar(_) | TypeName::SizePlaceholder | TypeName::Skolem(_) => {
            ArgKind::Size
        }
        TypeName::Buffer(_) | TypeName::NoBuffer => ArgKind::Region,
        TypeName::PointerFunction
        | TypeName::PointerInput
        | TypeName::PointerOutput
        | TypeName::PointerStorage => ArgKind::AddrSpace,
        _ => ArgKind::Type,
    }
}

/// Whether `arg` is acceptable in a slot expecting `expected`. A variable is
/// accepted everywhere (its kind is not yet known).
fn arg_matches_kind(arg: &Type, expected: ArgKind) -> bool {
    match arg {
        Type::Variable(_) => true,
        // An unresolved parser placeholder stands in for the array's variant
        // *and* region metadata until `resolve_placeholders` replaces it with
        // fresh variables; accept it in either slot.
        Type::Constructed(TypeName::AddressPlaceholder, _) => {
            matches!(expected, ArgKind::Variant | ArgKind::Region)
        }
        Type::Constructed(name, _) => arg_kind_of_name(name) == expected,
    }
}

/// Validate an `Array`'s argument list against its kind-schema:
/// `[elem, variant, dim_0, …, dim_{rank-1}, region]`. Region is appended last
/// so elem/variant/size keep their indices; the dims are the variadic middle,
/// checked position-by-position. This is the single source of truth for the
/// Array shape — accessors and the construction debug-asserts call it.
pub fn validate_array_args(args: &[Type]) -> Result<(), String> {
    if args.len() < 4 {
        return Err(format!(
            "Array expects >= 4 args [elem, variant, dim.., region], got {}",
            args.len()
        ));
    }
    if !arg_matches_kind(&args[0], ArgKind::Type) {
        return Err(format!("Array arg 0 (elem) must be a Type, got {:?}", args[0]));
    }
    if !arg_matches_kind(&args[1], ArgKind::Variant) {
        return Err(format!("Array arg 1 must be a Variant, got {:?}", args[1]));
    }
    let last = args.len() - 1;
    if !arg_matches_kind(&args[last], ArgKind::Region) {
        return Err(format!("Array last arg must be a Region, got {:?}", args[last]));
    }
    for (i, dim) in args[2..last].iter().enumerate() {
        if !arg_matches_kind(dim, ArgKind::Size) {
            return Err(format!("Array dim {} must be a Size, got {:?}", i, dim));
        }
    }
    Ok(())
}

/// Debug-only guard that an `Array`'s args satisfy [`validate_array_args`].
/// Panics with the schema mismatch in debug builds; compiles away in release.
#[inline]
pub fn debug_assert_array_args(args: &[Type]) {
    debug_assert!(
        validate_array_args(args).is_ok(),
        "{}",
        validate_array_args(args).unwrap_err()
    );
}

/// Validate one constructor's argument list against the kind-schema its
/// `TypeName` expects (non-recursive; checks only this level). Constructors
/// without a fixed schema (records, sums, named, …) pass. Returns a description
/// of the first mismatch.
pub fn validate_type_args(name: &TypeName, args: &[Type]) -> Result<(), String> {
    // (fixed-prefix kinds, optional repeated tail kind, optional repeated-min)
    let schema: Option<(&[ArgKind], Option<ArgKind>)> = match name {
        TypeName::Array => return validate_array_args(args),
        TypeName::Vec => Some((&[ArgKind::Type, ArgKind::Size] as &[ArgKind], None)),
        TypeName::Mat => Some((&[ArgKind::Type, ArgKind::Size, ArgKind::Size] as &[ArgKind], None)),
        TypeName::Pointer => Some((&[ArgKind::Type, ArgKind::AddrSpace] as &[ArgKind], None)),
        TypeName::StorageTexture => Some((&[ArgKind::Region] as &[ArgKind], None)),
        _ => None,
    };

    if let Some((prefix, tail)) = schema {
        if args.len() < prefix.len() {
            return Err(format!(
                "{:?} expects >= {} args, got {}",
                name,
                prefix.len(),
                args.len()
            ));
        }
        for (i, (arg, &kind)) in args.iter().zip(prefix.iter()).enumerate() {
            if !arg_matches_kind(arg, kind) {
                return Err(format!("{:?} arg {} must be {:?}, got {:?}", name, i, kind, arg));
            }
        }
        match tail {
            Some(tail_kind) => {
                for (i, arg) in args[prefix.len()..].iter().enumerate() {
                    if !arg_matches_kind(arg, tail_kind) {
                        return Err(format!(
                            "{:?} tail arg {} must be {:?}, got {:?}",
                            name,
                            prefix.len() + i,
                            tail_kind,
                            arg
                        ));
                    }
                }
            }
            None => {
                if args.len() != prefix.len() {
                    return Err(format!(
                        "{:?} expects exactly {} args, got {}",
                        name,
                        prefix.len(),
                        args.len()
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Recursively validate `ty` and every constructed type nested within it
/// against the per-constructor kind schema (see [`validate_type_args`]).
pub fn validate_type_schema(ty: &Type) -> Result<(), String> {
    if let Type::Constructed(name, args) = ty {
        validate_type_args(name, args)?;
        for a in args {
            validate_type_schema(a)?;
        }
    }
    Ok(())
}
