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
pub mod scheme;

pub use catalog::{BuiltinCatalog, BuiltinDef, BuiltinId, BuiltinKind, BuiltinOverload, Purity};
pub use lowering::BuiltinLowering;
pub use scheme::SchemeBuilder;

use std::sync::OnceLock;

static CATALOG: OnceLock<BuiltinCatalog> = OnceLock::new();

/// Process-wide builtin catalog. Constructed lazily on first call from
/// the static `defs::ALL_BUILTINS` table.
pub fn catalog() -> &'static BuiltinCatalog {
    CATALOG.get_or_init(|| BuiltinCatalog::from_raw(defs::ALL_BUILTINS))
}
