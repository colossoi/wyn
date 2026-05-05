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
    pub scheme: SchemeBuilder,
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
}

/// Indexed view over the catalog table. Built once at program startup
/// (via `OnceLock`); consumers take a `&BuiltinCatalog`.
#[derive(Debug)]
pub struct BuiltinCatalog {
    defs: Vec<BuiltinDef>,
    by_surface_name: HashMap<&'static str, BuiltinId>,
    by_internal_name: HashMap<&'static str, BuiltinId>,
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
        BuiltinCatalog {
            defs,
            by_surface_name,
            by_internal_name,
        }
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

    pub fn get(&self, id: BuiltinId) -> &BuiltinDef {
        &self.defs[id.as_index()]
    }

    /// Invoke an overload's `SchemeBuilder` to produce a fresh
    /// quantified `TypeScheme` against the supplied generator.
    pub fn build_scheme(
        &self,
        id: BuiltinId,
        overload_idx: usize,
        ctx: &mut dyn crate::type_checker::TypeVarGenerator,
    ) -> crate::ast::TypeScheme {
        let def = self.get(id);
        let ovld = &def.raw.overloads[overload_idx];
        (ovld.scheme)(ctx)
    }
}
