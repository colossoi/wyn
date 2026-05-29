// Unified registry for builtin functions, intrinsics, and module ops.
//
// `BuiltinCatalog` is the single source of truth: each builtin has one
// `BuiltinId`, one declarative `BuiltinDef` describing its overloads
// (scheme builder + lowering), and one entry in the static `ALL_BUILTINS`
// table. Consumers query by `BuiltinId` (canonical) or by surface/internal
// name (transitional).

pub mod catalog;
pub mod defs;
pub mod lowering;
pub mod names;
pub mod overload;
pub mod scheme;

pub use catalog::{BuiltinCatalog, BuiltinDef, BuiltinId, BuiltinKind, BuiltinOverload, Purity};
pub use lowering::BuiltinLowering;
pub use scheme::SchemeBuilder;

use std::sync::OnceLock;

static CATALOG: OnceLock<BuiltinCatalog> = OnceLock::new();

/// Process-wide builtin catalog. Constructed lazily on first call by
/// concatenating the static catalog table with runtime-generated
/// per-type op entries (whose names are leaked once at startup).
pub fn catalog() -> &'static BuiltinCatalog {
    CATALOG.get_or_init(|| BuiltinCatalog::build(defs::all_builtins()))
}

/// Direct catalog lookup by id. Equivalent to `catalog().get(id)`,
/// kept short for the common access pattern.
pub fn by_id(id: BuiltinId) -> &'static BuiltinDef {
    catalog().get(id)
}

/// Arity of the catalog-defined builtin or intrinsic with this name,
/// counted from its first overload's scheme. Returns `None` if the
/// name isn't in the catalog (most operators with empty
/// `intrinsic_source_names`, ad-hoc backend-only intrinsics, etc.).
///
/// The arity comes from the same `BuiltinDefRaw` table the type
/// checker uses, so verifiers and callers see one source of truth.
/// Multi-overload entries use the first overload's shape — overloads
/// are expected to have the same arity by construction (overloads
/// differ in element type, not parameter count).
pub fn intrinsic_arity(name: &str) -> Option<usize> {
    let def = catalog().lookup_by_any_name(name)?;
    let overload = def.raw.overloads.first()?;
    let mut ctx = polytype::Context::<crate::ast::TypeName>::default();
    let scheme = (overload.scheme?)(&mut ctx);
    let mut ty = &scheme;
    let monotype = loop {
        match ty {
            polytype::TypeScheme::Monotype(t) => break t,
            polytype::TypeScheme::Polytype { body, .. } => ty = body,
        }
    };
    // Only return arity when the root is an arrow — otherwise this is a
    // placeholder/value scheme (e.g. compiler-internal intrinsics whose
    // scheme is just `Unit`) and arity is meaningless.
    match monotype {
        polytype::Type::Constructed(crate::ast::TypeName::Arrow, _) => Some(count_arrows(monotype)),
        _ => None,
    }
}

fn count_arrows(ty: &crate::ast::Type) -> usize {
    match ty {
        polytype::Type::Constructed(crate::ast::TypeName::Arrow, args) if args.len() == 2 => {
            1 + count_arrows(&args[1])
        }
        _ => 0,
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod mod_tests;
