//! Top-level AST → TLC entry point.
//!
//! Owns name-registry construction, deterministic SymbolId pre-allocation,
//! and the three-phase Transformer drive (per prelude module, per top-level
//! prelude function, then the user program). Returns the first phase-typed TLC
//! program, with definition schemes and the remaining compiler state stored on
//! their owning nodes.

use crate::LookupMap;

use polytype::TypeScheme;

use crate::ast;
use crate::error::CompilerError;
use crate::module_manager::{self, ModuleManager};
use crate::name_registry::NameRegistry;
use crate::name_resolution::NameResolution;
use crate::types::TypeName;
use crate::{SymbolId, SymbolTable, TypeTable};

use super::{defaults, Family, Program, Stage, TermIdSource, Transformer};

/// Polymorphic TLC definitions retain their type schemes in-tree.
#[derive(Debug, Clone, Copy, Default)]
pub struct Polymorphic;

impl Family for Polymorphic {
    type DefinitionData = super::data::PolymorphicDefinition;
    type EntryData = ();
    type ClosureData = super::data::Empty;
    type SoacBodyData = super::data::Empty;
}

/// AST has been transformed to TLC.
#[derive(Debug, Clone, Copy, Default)]
pub struct Transformed;

impl Stage for Transformed {
    type Family = Polymorphic;
    type GlobalContext = super::context::TransformedGlobal;
}

pub fn run(
    ast: &ast::Program,
    mut type_table: TypeTable,
    schemes: &LookupMap<String, TypeScheme<TypeName>>,
    checker_builtins: &[String],
    name_resolution: &NameResolution,
    module_manager: &ModuleManager,
    fill_holes: bool,
) -> Program<Transformed> {
    if fill_holes {
        defaults::default_free_vars_in_table(type_table.values_mut());
    }

    let registry = NameRegistry::build(ast, module_manager, checker_builtins);

    let mut symbols = SymbolTable::new();
    let mut top_level_symbols: LookupMap<String, SymbolId> = LookupMap::new();
    for (name, _kind) in registry.iter() {
        let sym = symbols.alloc(name.to_string());
        top_level_symbols.insert(name.to_string(), sym);
    }

    let mut fill_hole_errors: Vec<CompilerError> = Vec::new();
    let mut term_ids = TermIdSource::new();

    let mut prelude_defs = Vec::new();
    for (module_name, elaborated) in module_manager.get_elaborated_modules() {
        let mut transformer = Transformer::with_namespace(
            &type_table,
            &mut symbols,
            &mut top_level_symbols,
            name_resolution,
            module_name,
            fill_holes,
            &mut fill_hole_errors,
            &mut term_ids,
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
            name_resolution,
            fill_holes,
            &mut fill_hole_errors,
            &mut term_ids,
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
        name_resolution,
        fill_holes,
        &mut fill_hole_errors,
        &mut term_ids,
    );
    let mut parts = transformer.transform_program(ast);
    drop(transformer);

    let mut merged_defs = prelude_defs;
    merged_defs.extend(parts.defs);
    parts.defs = merged_defs;

    for def in &mut parts.defs {
        let name = symbols.get(def.name).expect("BUG: transformed definition symbol is missing");
        def.data.scheme = schemes.get(name).cloned();
    }

    parts.with_symbols::<Transformed>(
        symbols,
        top_level_symbols,
        term_ids,
        super::context::TransformedGlobal {
            type_table,
            known_defs: registry.name_set(),
            fill_hole_errors,
            auto_storage_binding_ids: crate::IdSource::new(),
        },
    )
}
