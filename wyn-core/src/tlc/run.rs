//! Top-level AST → TLC transition.

use crate::ast;
use crate::name_registry::NameRegistry;
use crate::{LookupMap, SymbolId, SymbolTable};

use super::{Family, Program, Stage, TermIdSource, Transformer};

/// Polymorphic TLC definitions retain their type schemes in-tree.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnpinnedPolymorphic;

impl Family for UnpinnedPolymorphic {
    type DefinitionData = super::data::PolymorphicDefinition;
    type EntryData = ();
    type ClosureData = super::data::Empty;
    type SoacBodyData = super::data::Empty;
}

/// AST has been transformed to TLC.
#[derive(Debug, Clone, Copy, Default)]
pub struct Transformed;

impl Stage for Transformed {
    type Family = UnpinnedPolymorphic;
    type GlobalContext = super::context::TransformedGlobal;
}

pub fn lower_from_ast(ast: ast::Program<crate::ast_type_holes::HolesResolved>) -> Program<Transformed> {
    let registry = NameRegistry::build(&ast);
    let mut symbols = SymbolTable::new();
    let mut top_level_symbols: LookupMap<String, SymbolId> = LookupMap::new();
    for (name, _kind) in registry.iter() {
        let symbol = symbols.alloc(name.to_string());
        top_level_symbols.insert(name.to_string(), symbol);
    }

    let mut term_ids = TermIdSource::new();
    let mut support_defs = Vec::new();
    for support in &ast.global_context.support_definitions {
        let mut transformer = match &support.namespace {
            Some(namespace) => {
                Transformer::with_namespace(&mut symbols, &mut top_level_symbols, namespace, &mut term_ids)
            }
            None => Transformer::new(&mut symbols, &mut top_level_symbols, &mut term_ids),
        };
        if let Some(definition) = transformer.transform_decl(&support.definition) {
            support_defs.push(definition);
        }
    }

    let mut transformer = Transformer::new(&mut symbols, &mut top_level_symbols, &mut term_ids);
    let mut parts = transformer.transform_program(&ast);
    support_defs.append(&mut parts.defs);
    parts.defs = support_defs;
    let known_defs = top_level_symbols.keys().cloned().collect();

    parts.with_symbols::<Transformed>(
        symbols,
        top_level_symbols,
        term_ids,
        super::context::TransformedGlobal {
            known_defs,
            auto_storage_binding_ids: crate::IdSource::new(),
        },
    )
}
