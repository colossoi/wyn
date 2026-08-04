//! Top-level AST → TLC transition.

use super::{TermIdSource, Transformer};

/// Polymorphic TLC definitions retain their type schemes in-tree.
pub type UnpinnedPolymorphic =
    super::TreeFamily<super::data::PolymorphicDefinition, (), super::data::Empty, super::data::Empty>;

/// AST has been transformed to TLC.
#[derive(Debug, Clone, Copy)]
pub enum TransformedTag {}
pub type Transformed =
    super::Program<TransformedTag, UnpinnedPolymorphic, super::context::TransformedGlobal>;

pub fn lower_from_ast(mut ast: crate::ast_type_holes::HolesResolved) -> crate::error::Result<Transformed> {
    let mut symbols = std::mem::take(&mut ast.global_context.symbols);
    let mut term_ids = TermIdSource::new();
    let mut support_defs = Vec::new();
    let mut transformer = Transformer::new(&mut symbols, &mut term_ids);
    for support in &ast.global_context.support_definitions {
        if let Some(definition) = transformer.transform_decl(&support.definition) {
            support_defs.push(definition);
        }
    }

    let mut parts = transformer.transform_program(&ast);
    drop(transformer);
    support_defs.append(&mut parts.defs);
    parts.defs = support_defs;
    let known_defs = parts.defs.iter().map(|definition| definition.name).collect();
    let program = parts.with_symbols::<TransformedTag, _>(
        symbols,
        term_ids,
        super::context::TransformedGlobal {
            known_defs,
            auto_storage_binding_ids: crate::IdSource::new(),
        },
    );
    super::ownership::check_unextracted(&program)?;

    let super::Program {
        defs,
        mut symbols,
        mut term_ids,
        global_context,
        state: _,
    } = program;
    let mut parts = super::ProgramParts { defs };
    super::stage_extract::extract(&mut parts, &mut symbols, &mut term_ids);

    Ok(parts.with_symbols::<TransformedTag, _>(symbols, term_ids, global_context))
}
