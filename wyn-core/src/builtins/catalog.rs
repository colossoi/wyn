use std::collections::HashMap;

use crate::builtins::lowering::BuiltinLowering;
use crate::builtins::scheme::SchemeBuilder;

/// Canonical identity for a builtin. Allocated by position in the
/// catalog at startup; opaque to consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BuiltinId(u32);

impl BuiltinId {
    pub(crate) fn new(idx: u32) -> Self {
        BuiltinId(idx)
    }

    pub fn as_index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinKind {
    /// User-facing functions like `length`, `magnitude`, `dot`.
    UserVisible,
    /// `_w_intrinsic_*` names emitted by the compiler internally.
    InternalIntrinsic,
    /// Module-qualified ops like `f32.sin`, `vec.pow`, `i32.+`.
    ModuleBuiltin,
    /// Operator dispatch entries (e.g. `f32.+`, `i32.<`). Currently
    /// looked up by `<type>.<op>` string.
    Operator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Purity {
    Pure,
    Effectful,
}

/// One overload of a builtin. A builtin may have multiple overloads
/// that differ in scheme shape; overload selection happens at
/// type-check time via unification.
#[derive(Debug, Clone)]
pub struct BuiltinOverload {
    /// Optional scheme builder. `None` for overloads whose real
    /// scheme comes from prelude module signatures (per-type ops
    /// like `f32.add`) or for compiler-internal builtins that aren't
    /// directly callable from user code (storage_index, etc.). The
    /// type checker routes `None` to `module_schemes` by surface name.
    pub scheme: Option<SchemeBuilder>,
    pub lowering: BuiltinLowering,
}

/// The static, declarative description of one builtin. Entries are
/// produced by `defs::all_builtins()`; the catalog wraps each with its
/// assigned `BuiltinId`.
///
/// Names: `surface_name` is the canonical user-facing label (shown in
/// errors, used by namespace_hint). `intrinsic_source_names` is the set
/// of keys this entry publishes a polymorphic scheme under (consumed by
/// the type checker's name resolution). `impl_source_names` is the set
/// of keys this entry publishes a backend lowering under (consumed by
/// the backends). Per-type ops like `f32.+` have empty
/// `intrinsic_source_names` (their schemes come from prelude module
/// signatures) and two entries in `impl_source_names`: `f32.+` and
/// `_w_intrinsic_+_f32`.
#[derive(Debug, Clone)]
pub struct BuiltinDefRaw {
    pub surface_name: &'static str,
    pub intrinsic_source_names: &'static [&'static str],
    pub impl_source_names: &'static [&'static str],
    pub kind: BuiltinKind,
    pub purity: Purity,
    pub overloads: &'static [BuiltinOverload],
}

#[derive(Debug, Clone)]
pub struct BuiltinDef {
    pub id: BuiltinId,
    pub raw: BuiltinDefRaw,
}

impl BuiltinDef {
    pub fn intrinsic_source_names(&self) -> &'static [&'static str] {
        self.raw.intrinsic_source_names
    }

    pub fn impl_source_names(&self) -> &'static [&'static str] {
        self.raw.impl_source_names
    }

    pub fn overloads(&self) -> &'static [BuiltinOverload] {
        self.raw.overloads
    }

    /// The name backends key dispatch on. For polymorphic intrinsics
    /// (`magnitude` → `_w_intrinsic_magnitude`) this is the internal
    /// `_w_intrinsic_*` form; for entries whose surface name is already
    /// internal (e.g. `_w_intrinsic_storage_len`, `vec.sin`) it's the
    /// surface name.
    pub fn dispatch_name(&self) -> &'static str {
        self.raw.impl_source_names.first().copied().unwrap_or(self.raw.surface_name)
    }
}

/// Cached `BuiltinId`s for catalog entries that the IR/backends reach
/// for by name. Populated once at catalog construction; consumers
/// compare a dispatch-site id against these constants instead of
/// matching on a parallel `Intrinsic` enum.
///
/// Each field is an `Option` because the corresponding catalog entry
/// is registered by name; `None` means the catalog didn't include it
/// at construction (which would itself be a bug for any field listed
/// here, but we surface it via `expect()` in the named accessors).
#[derive(Debug)]
pub struct KnownBuiltinIds {
    pub uninit: BuiltinId,
    pub array_with: BuiltinId,
    pub array_with_in_place: BuiltinId,
    pub length: BuiltinId,
    pub slice: BuiltinId,
    pub storage_len: BuiltinId,
    pub thread_id: BuiltinId,
    pub storage_index: BuiltinId,
    pub storage_store: BuiltinId,
}

/// Indexed view over the catalog table. Built once at program startup
/// (via `OnceLock`); consumers take a `&BuiltinCatalog`.
#[derive(Debug)]
pub struct BuiltinCatalog {
    defs: Vec<BuiltinDef>,
    by_surface_name: HashMap<&'static str, BuiltinId>,
    by_internal_name: HashMap<&'static str, BuiltinId>,
    known: KnownBuiltinIds,
}

impl BuiltinCatalog {
    /// Build a catalog from a vector of raw definitions. Each entry's
    /// `BuiltinId` is its position in the slice. Per-type op entries
    /// generated at startup leak their names via `Box::leak`, giving
    /// every catalog name a `&'static str` lifetime.
    pub fn build(raw_defs: Vec<BuiltinDefRaw>) -> Self {
        let mut defs = Vec::with_capacity(raw_defs.len());
        let mut by_surface_name = HashMap::with_capacity(raw_defs.len());
        let mut by_internal_name = HashMap::new();
        for (i, raw) in raw_defs.into_iter().enumerate() {
            let id = BuiltinId::new(i as u32);
            if by_surface_name.insert(raw.surface_name, id).is_some() {
                panic!("duplicate surface_name in builtin catalog: {}", raw.surface_name);
            }
            for &impl_name in raw.impl_source_names {
                if impl_name != raw.surface_name && by_internal_name.insert(impl_name, id).is_some() {
                    panic!("duplicate impl_source_name in builtin catalog: {}", impl_name);
                }
            }
            defs.push(BuiltinDef { id, raw });
        }

        // Resolve well-known intrinsic names → BuiltinIds. Any miss
        // here is a catalog construction bug — these names are required
        // by the IR/backends for `ByBuiltinId` dispatch.
        let resolve = |n: &'static str| -> BuiltinId {
            by_surface_name
                .get(n)
                .or_else(|| by_internal_name.get(n))
                .copied()
                .unwrap_or_else(|| panic!("known intrinsic {:?} missing from catalog", n))
        };
        use crate::builtins::names as N;
        let known = KnownBuiltinIds {
            uninit: resolve(N::INTRINSIC_UNINIT),
            array_with: resolve(N::INTRINSIC_ARRAY_WITH),
            array_with_in_place: resolve(N::INTRINSIC_ARRAY_WITH_INPLACE),
            length: resolve(N::INTRINSIC_LENGTH),
            slice: resolve(N::INTRINSIC_SLICE),
            storage_len: resolve(N::INTRINSIC_STORAGE_LEN),
            thread_id: resolve(N::INTRINSIC_THREAD_ID),
            storage_index: resolve(N::INTRINSIC_STORAGE_INDEX),
            storage_store: resolve(N::INTRINSIC_STORAGE_STORE),
        };

        BuiltinCatalog {
            defs,
            by_surface_name,
            by_internal_name,
            known,
        }
    }

    /// Cached `BuiltinId`s for entries that the IR/backends dispatch on.
    pub fn known(&self) -> &KnownBuiltinIds {
        &self.known
    }

    pub fn defs(&self) -> &[BuiltinDef] {
        &self.defs
    }

    pub fn lookup_by_surface_name(&self, name: &str) -> Option<&BuiltinDef> {
        self.by_surface_name.get(name).map(|id| &self.defs[id.as_index()])
    }

    /// Look up by an internal `_w_intrinsic_*` name registered in
    /// `impl_source_names`. Returns the entry whose `impl_source_names`
    /// contains this name (excluding cases where it matches
    /// `surface_name`, which is reachable via `lookup_by_surface_name`).
    pub fn lookup_by_internal_name(&self, name: &str) -> Option<&BuiltinDef> {
        self.by_internal_name.get(name).map(|id| &self.defs[id.as_index()])
    }

    /// Convenience: try surface, then internal — useful for IR emission
    /// sites that just have a string and need a `BuiltinId`.
    pub fn lookup_by_any_name(&self, name: &str) -> Option<&BuiltinDef> {
        self.lookup_by_surface_name(name).or_else(|| self.lookup_by_internal_name(name))
    }

    pub fn get(&self, id: BuiltinId) -> &BuiltinDef {
        &self.defs[id.as_index()]
    }

    /// Resolve a name to its backend lowering. Uses the first overload's
    /// lowering — adequate for per-type ops where overloads share
    /// lowering, and for polymorphic intrinsics that have a single
    /// overload.
    ///
    /// TODO: this silently picks overload 0 with no diagnostic. Today
    /// every multi-overload entry has matching lowerings (or has only
    /// one overload), so this is correct in practice — but the
    /// invariant isn't enforced. Fix by either (a) asserting at catalog
    /// construction that all overloads of an entry share lowering, or
    /// (b) threading the overload index from the type checker through
    /// the IR so backends look up `overloads[idx]` directly.
    pub fn lookup_lowering(&self, name: &str) -> Option<&crate::builtins::lowering::BuiltinLowering> {
        let def = self.lookup_by_any_name(name)?;
        Some(&def.overloads()[0].lowering)
    }

    /// Invoke an overload's scheme builder to produce a fresh
    /// quantified `TypeScheme` against the supplied generator.
    /// Returns `None` if the overload doesn't have an embedded scheme
    /// builder — caller routes through prelude module signatures.
    pub fn build_scheme(
        &self,
        id: BuiltinId,
        overload_idx: usize,
        ctx: &mut dyn crate::type_checker::TypeVarGenerator,
    ) -> Option<crate::ast::TypeScheme> {
        let def = self.get(id);
        let ovld = &def.raw.overloads[overload_idx];
        ovld.scheme.map(|f| f(ctx))
    }
}
