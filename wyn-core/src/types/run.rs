//! Top-level type-checking entry point.
//!
//! Orchestrates the three-step front-end-to-types boundary:
//! 1. `resolve_placeholders` — turn type-spec placeholders into type variables
//!    and a unification context.
//! 2. `resolve_opens` — qualify bare names introduced by `open M` against the
//!    union of module specs and the builtins catalog.
//! 3. `TypeChecker` — load builtins, run inference, collect schemes/warnings.

use crate::builtins::catalog;
use std::collections::HashMap;

use polytype::TypeScheme;

use crate::ast::{self, NodeId, TypeName};
use crate::error::Result;
use crate::module_manager::ModuleManager;
use crate::name_resolution::NameResolution;
use crate::resolve_opens;
use crate::resolve_placeholders;
use crate::types::checker::{TypeChecker, TypeWarning};

pub struct TypeCheckOutput {
    pub type_table: HashMap<NodeId, TypeScheme<TypeName>>,
    pub schemes: HashMap<String, TypeScheme<TypeName>>,
    pub warnings: Vec<TypeWarning>,
    pub builtin_names: Vec<String>,
    pub name_resolution: NameResolution,
}

pub fn run(ast: &mut ast::Program, module_manager: &mut ModuleManager) -> Result<TypeCheckOutput> {
    let mut resolver = resolve_placeholders::PlaceholderResolver::new();
    resolver.resolve(module_manager, ast);
    let (context, spec_schemes) = resolver.into_parts();

    // Build the open-resolver index once and reuse across the user
    // `Program` and every elaborated prelude module body — the index
    // is invariant across this whole run (spec_schemes + catalog
    // don't change). User-defined modules don't go through
    // `spec_schemes` (their bodies are `Decl`s, not `Sig` specs), so
    // walk the elaborated-modules table here and inject each module's
    // `Decl` member names directly — without this, `open base` for a
    // user `module base = { def foo … }` finds `base` as an unknown
    // module and bare `foo` references inside sibling module bodies
    // never resolve.
    let mut open_index = resolve_opens::build_index(&spec_schemes, catalog());
    for (module_name, elaborated) in module_manager.get_elaborated_modules() {
        for item in &elaborated.items {
            if let crate::module_manager::ElaboratedItem::Decl(decl) = item {
                open_index.add_member(module_name, &decl.name);
            }
        }
    }
    resolve_opens::run_with_index(ast, &open_index)?;

    // Prelude module decl bodies don't go through `resolve_opens::run`
    // (they aren't in the user `Program`), but their bodies reference
    // sibling defs by unqualified name (e.g. `log10`'s body calls `log`,
    // which is `f32.log` from outside). Qualify them here as if each
    // body were inside an implicit `open <module_name>`.
    let module_names: Vec<String> = module_manager.elaborated_modules_mut().keys().cloned().collect();
    for module_name in module_names {
        let elaborated =
            module_manager.elaborated_modules_mut().get_mut(&module_name).expect("module exists");
        for item in &mut elaborated.items {
            if let crate::module_manager::ElaboratedItem::Decl(decl) = item {
                resolve_opens::run_in_module_with_index(&mut decl.body, &module_name, &open_index)?;
            }
        }
    }

    let name_resolution = crate::name_resolution::build_name_resolution(ast, module_manager, catalog());

    let mut checker = TypeChecker::with_context_and_schemes(module_manager, context, spec_schemes);
    checker.set_name_resolution(name_resolution);
    checker.load_builtins()?;
    let type_table = checker.check_program(ast)?;
    let schemes = checker.get_function_schemes();
    let builtin_names = checker.builtin_names();
    let warnings: Vec<_> = checker.warnings().to_vec();
    // Read the NameResolution back: the checker may have written
    // overload-index resolutions into it during inference.
    let name_resolution = checker.name_resolution().clone();

    Ok(TypeCheckOutput {
        type_table,
        schemes,
        warnings,
        builtin_names,
        name_resolution,
    })
}
