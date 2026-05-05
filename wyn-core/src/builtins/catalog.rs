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

/// The static, declarative description of one builtin. Entries live in
/// `defs::ALL_BUILTINS` as a `&'static [BuiltinDefRaw]`; the catalog
/// wraps each with its assigned `BuiltinId`.
#[derive(Debug, Clone)]
pub struct BuiltinDefRaw {
    pub surface_name: &'static str,
    pub internal_name: Option<&'static str>,
    pub kind: BuiltinKind,
    pub purity: Purity,
    pub overloads: &'static [BuiltinOverload],
}

#[derive(Debug, Clone)]
pub struct BuiltinDef {
    pub id: BuiltinId,
    pub raw: &'static BuiltinDefRaw,
}

impl BuiltinDef {
    pub fn surface_name(&self) -> &'static str {
        self.raw.surface_name
    }

    pub fn internal_name(&self) -> Option<&'static str> {
        self.raw.internal_name
    }

    pub fn kind(&self) -> BuiltinKind {
        self.raw.kind
    }

    pub fn overloads(&self) -> &'static [BuiltinOverload] {
        self.raw.overloads
    }
}

/// Indexed view over the static `ALL_BUILTINS` table. Built once at
/// program startup (via `OnceLock`); consumers take a `&BuiltinCatalog`.
#[derive(Debug)]
pub struct BuiltinCatalog {
    defs: Vec<BuiltinDef>,
    by_surface_name: HashMap<&'static str, BuiltinId>,
    by_internal_name: HashMap<&'static str, BuiltinId>,
}

impl BuiltinCatalog {
    /// Construct from a static slice of raw definitions. Each entry's
    /// `BuiltinId` is its position in the slice.
    pub fn from_raw(raw_defs: &'static [BuiltinDefRaw]) -> Self {
        let mut defs = Vec::with_capacity(raw_defs.len());
        let mut by_surface_name = HashMap::with_capacity(raw_defs.len());
        let mut by_internal_name = HashMap::new();
        for (i, raw) in raw_defs.iter().enumerate() {
            let id = BuiltinId::new(i as u32);
            defs.push(BuiltinDef { id, raw });
            if by_surface_name.insert(raw.surface_name, id).is_some() {
                panic!("duplicate surface_name in builtin catalog: {}", raw.surface_name);
            }
            if let Some(internal) = raw.internal_name {
                if by_internal_name.insert(internal, id).is_some() {
                    panic!("duplicate internal_name in builtin catalog: {}", internal);
                }
            }
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

    pub fn lookup_by_internal_name(&self, name: &str) -> Option<&BuiltinDef> {
        self.by_internal_name.get(name).map(|id| &self.defs[id.as_index()])
    }

    pub fn get(&self, id: BuiltinId) -> &BuiltinDef {
        &self.defs[id.as_index()]
    }

    pub fn iter_by_kind(&self, kind: BuiltinKind) -> impl Iterator<Item = &BuiltinDef> {
        self.defs.iter().filter(move |d| d.raw.kind == kind)
    }
}
