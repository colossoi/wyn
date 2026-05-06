//! Top-level type-checking entry point.
//!
//! Orchestrates the three-step front-end-to-types boundary:
//! 1. `resolve_placeholders` — turn type-spec placeholders into type variables
//!    and a unification context.
//! 2. `resolve_opens` — qualify bare names introduced by `open M` against the
//!    union of module specs and the builtins catalog.
//! 3. `TypeChecker` — load builtins, run inference, collect schemes/warnings.

use std::collections::HashMap;

use polytype::TypeScheme;

use crate::ast::{self, NodeId, TypeName};
use crate::error::Result;
use crate::module_manager::ModuleManager;
use crate::resolve_opens;
use crate::resolve_placeholders;
use crate::types::checker::{TypeChecker, TypeWarning};

pub struct TypeCheckOutput {
    pub type_table: HashMap<NodeId, TypeScheme<TypeName>>,
    pub schemes: HashMap<String, TypeScheme<TypeName>>,
    pub warnings: Vec<TypeWarning>,
    pub builtin_names: Vec<String>,
}

pub fn run(ast: &mut ast::Program, module_manager: &mut ModuleManager) -> Result<TypeCheckOutput> {
    let mut resolver = resolve_placeholders::PlaceholderResolver::new();
    resolver.resolve(module_manager, ast);
    let (context, spec_schemes) = resolver.into_parts();

    resolve_opens::run(ast, &spec_schemes, crate::builtins::catalog())?;

    let mut checker = TypeChecker::with_context_and_schemes(module_manager, context, spec_schemes);
    checker.set_name_resolution(crate::name_resolution::build_name_resolution(
        ast,
        crate::builtins::catalog(),
    ));
    checker.load_builtins()?;
    let type_table = checker.check_program(ast)?;
    let schemes = checker.get_function_schemes();
    let builtin_names = checker.builtin_names();
    let warnings: Vec<_> = checker.warnings().to_vec();

    Ok(TypeCheckOutput {
        type_table,
        schemes,
        warnings,
        builtin_names,
    })
}
