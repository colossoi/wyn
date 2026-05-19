//! Unified name registry for the TLC transformer.
//!
//! Collects all top-level names from every source (builtins, modules, prelude,
//! user code) into a single `BTreeMap` for deterministic SymbolId assignment.

use std::collections::{BTreeMap, HashSet};

use crate::ast;
use crate::module_manager;

/// Category of a registered name. Used for diagnostics only —
/// resolution doesn't branch on this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameKind {
    /// SOAC/array builtins: map, reduce, scan, filter, scatter, zip, length, ...
    SoacBuiltin,
    /// Math builtins: sin, cos, tan, sqrt, abs, floor, ceil, fract, exp, log
    MathBuiltin,
    /// Type-specific operations: f32.add, i32.mul, u32.&
    ImplOp,
    /// Polymorphic intrinsics: _w_intrinsic_map, abs, sign
    PolymorphicIntrinsic,
    /// Module-qualified items: f32.sin, f32.pi, trig.sinpi
    ModuleItem,
    /// Prelude functions: unzip, all, any
    PreludeFunction,
    /// User declarations: Decl, Entry, Extern, Uniform, Storage
    UserDecl,
}

/// Authoritative registry of all names the TLC transformer will encounter.
/// Built once after type checking, consumed by `to_tlc()`.
///
/// Uses `BTreeMap` for deterministic iteration order (lexicographic),
/// ensuring SymbolId assignment is identical across runs.
pub struct NameRegistry {
    names: BTreeMap<String, NameKind>,
}

const MATH_BUILTINS: &[&str] = &[
    "sin", "cos", "tan", "sqrt", "abs", "floor", "ceil", "fract", "exp", "log", "atan", "asin", "acos",
    "radians", "degrees", "pow", "atan2", "mod",
];

impl NameRegistry {
    pub fn build(
        ast: &ast::Program,
        module_manager: &module_manager::ModuleManager,
        checker_builtins: &[String],
    ) -> Self {
        let mut names = BTreeMap::new();

        // 1+2. All catalog entries: per-source-name, classify into
        // ImplOp (entries with `impl_source_names`) and
        // PolymorphicIntrinsic (entries with `intrinsic_source_names`).
        let catalog = crate::builtins::catalog();
        for def in catalog.defs() {
            for &name in def.impl_source_names() {
                names.insert(name.to_string(), NameKind::ImplOp);
            }
            for &name in def.intrinsic_source_names() {
                names.insert(name.to_string(), NameKind::PolymorphicIntrinsic);
            }
        }

        // 3. Type-checker builtins (map, reduce, sin, cos, etc.)
        for name in checker_builtins {
            let kind = if MATH_BUILTINS.contains(&name.as_str()) {
                NameKind::MathBuiltin
            } else {
                NameKind::SoacBuiltin
            };
            names.entry(name.clone()).or_insert(kind);
        }

        // 4. Module items (specs + decls, namespaced)
        for (module_name, elaborated) in module_manager.get_elaborated_modules() {
            for item in &elaborated.items {
                let item_name = match item {
                    module_manager::ElaboratedItem::Spec(ast::Spec::Sig(n, _, _)) => Some(n.as_str()),
                    module_manager::ElaboratedItem::Spec(ast::Spec::SigOp(op, _)) => Some(op.as_str()),
                    module_manager::ElaboratedItem::Decl(decl) => Some(decl.name.as_str()),
                    _ => None,
                };
                if let Some(name) = item_name {
                    names.insert(format!("{}.{}", module_name, name), NameKind::ModuleItem);
                }
            }
        }

        // 5. Prelude functions
        for decl in module_manager.get_prelude_function_declarations() {
            names.entry(decl.name.clone()).or_insert(NameKind::PreludeFunction);
        }

        // 6. User declarations
        for decl in &ast.declarations {
            let name = match decl {
                ast::Declaration::Decl(d) => Some(d.name.clone()),
                ast::Declaration::Entry(e) => Some(e.name.clone()),
                ast::Declaration::Extern(e) => Some(e.name.clone()),
                _ => None,
            };
            if let Some(name) = name {
                names.entry(name).or_insert(NameKind::UserDecl);
            }
        }

        NameRegistry { names }
    }

    /// Deterministic iterator over all names (sorted lexicographically).
    pub fn iter(&self) -> impl Iterator<Item = (&str, NameKind)> {
        self.names.iter().map(|(k, v)| (k.as_str(), *v))
    }

    /// Collect every registered name into an owned `HashSet`.
    pub fn name_set(&self) -> HashSet<String> {
        self.names.keys().cloned().collect()
    }
}
