//! Top-level AST → TLC entry point.
//!
//! Owns name-registry construction, deterministic SymbolId pre-allocation,
//! and the three-phase Transformer drive (per prelude module, per top-level
//! prelude function, then the user program). Returns a fully built
//! `tlc::Program` plus the supporting metadata (type_table, known_defs,
//! schemes, fill-hole errors).

use std::collections::HashMap;

use polytype::TypeScheme;

use crate::ast;
use crate::error::CompilerError;
use crate::module_manager::{self, ModuleManager};
use crate::name_registry::NameRegistry;
use crate::types::TypeName;
use crate::{SymbolId, SymbolTable, TypeTable};

use super::{Program, Transformer, defaults};

pub struct TlcOutput {
    pub program: Program,
    pub type_table: TypeTable,
    pub known_defs: std::collections::HashSet<String>,
    pub schemes: HashMap<SymbolId, TypeScheme<TypeName>>,
    pub fill_hole_errors: Vec<CompilerError>,
}

pub fn run(
    ast: &ast::Program,
    mut type_table: TypeTable,
    schemes: &HashMap<String, TypeScheme<TypeName>>,
    checker_builtins: &[String],
    module_manager: &ModuleManager,
    fill_holes: bool,
) -> TlcOutput {
    if fill_holes {
        defaults::default_free_vars_in_table(type_table.values_mut());
    }

    let registry = NameRegistry::build(ast, module_manager, checker_builtins);

    let mut symbols = SymbolTable::new();
    let mut top_level_symbols: HashMap<String, SymbolId> = HashMap::new();
    for (name, _kind) in registry.iter() {
        let sym = symbols.alloc(name.to_string());
        top_level_symbols.insert(name.to_string(), sym);
    }

    let mut fill_hole_errors: Vec<CompilerError> = Vec::new();

    let mut prelude_defs = Vec::new();
    for (module_name, elaborated) in module_manager.get_elaborated_modules() {
        let mut transformer = Transformer::with_namespace(
            &type_table,
            &mut symbols,
            &mut top_level_symbols,
            module_name,
            fill_holes,
            &mut fill_hole_errors,
        );
        for item in &elaborated.items {
            if let module_manager::ElaboratedItem::Decl(decl) = item {
                if let Some(def) = transformer.transform_decl(decl) {
                    prelude_defs.push(def);
                }
            }
        }
    }

    {
        let mut transformer = Transformer::new(
            &type_table,
            &mut symbols,
            &mut top_level_symbols,
            fill_holes,
            &mut fill_hole_errors,
        );
        for decl in module_manager.get_prelude_function_declarations() {
            if let Some(def) = transformer.transform_decl(decl) {
                prelude_defs.push(def);
            }
        }
    }

    let mut transformer = Transformer::new(
        &type_table,
        &mut symbols,
        &mut top_level_symbols,
        fill_holes,
        &mut fill_hole_errors,
    );
    let mut parts = transformer.transform_program(ast);

    let mut merged_defs = prelude_defs;
    merged_defs.extend(parts.defs);
    parts.defs = merged_defs;

    // Schemes the type checker recorded for non-top-level bindings
    // have no SymbolId in `top_level_symbols`; filter them out.
    let schemes_by_sym: HashMap<SymbolId, TypeScheme<TypeName>> = schemes
        .iter()
        .filter_map(|(name, scheme)| top_level_symbols.get(name).map(|&sym| (sym, scheme.clone())))
        .collect();

    let program = parts.with_symbols(symbols, top_level_symbols);

    TlcOutput {
        program,
        type_table,
        known_defs: registry.name_set(),
        schemes: schemes_by_sym,
        fill_hole_errors,
    }
}
