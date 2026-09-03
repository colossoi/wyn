//! Top-level type-checking entry point.

#[cfg(test)]
#[path = "run_tests.rs"]
mod run_tests;

use crate::ast::{self, Declaration};
use crate::builtins;
use crate::err_type_at;
use crate::error::Result;
use crate::interface;
use crate::name_resolution;
use crate::name_resolution::ResolvedValueRef;
use crate::resolve_opens;
use crate::resolve_placeholders;
use crate::semantic_modules;
use crate::types::checker::{TypeChecker, TypeWarning};
use crate::{CompilerOptions, LookupMap};

pub type TypeCheckedFamily = ast::AstFamily<
    ast::TypedTree,
    ast::TypedDefinition,
    ast::TypedEntry,
    interface::ResolvedAttribute,
    ast::TypedExtern,
    std::convert::Infallible,
>;

/// AST with all inferred types, declaration schemes, and identifier
/// classifications stored on their owning tree nodes.
#[derive(Debug, Clone, Copy)]
pub enum TypeCheckedTag {}
pub type TypeChecked =
    ast::Program<TypeCheckedTag, TypeCheckedFamily, ast::TypedGlobal<ast::TypedDefinition, ast::TypedTree>>;

pub fn type_check(program: resolve_opens::OpensResolved, options: CompilerOptions) -> Result<TypeChecked> {
    let name_resolution = name_resolution::build_name_resolution(
        &program,
        &program.global_context.semantic_modules,
        builtins::catalog(),
        options,
    );
    let checked = program.try_rebuild(|declarations, global_context, _| {
        let resolve_placeholders::PlaceholdersResolvedGlobal {
            semantic_modules,
            context,
            spec_schemes,
        } = global_context;
        let mut checker = TypeChecker::with_context_and_schemes(
            &semantic_modules,
            context,
            spec_schemes,
            name_resolution,
        );
        checker.load_builtins()?;
        let type_table = checker.check_program(&declarations)?;
        let schemes = checker.get_function_schemes();
        let builtin_names = checker.builtin_names();
        let warnings: Vec<_> = checker.warnings().to_vec();
        let name_resolution = checker.name_resolution().clone();
        drop(checker);

        materialize(
            declarations,
            semantic_modules,
            type_table,
            schemes,
            warnings,
            builtin_names,
            name_resolution,
        )
    })?;
    super::stage_context::validate(&checked)?;
    Ok(checked)
}

fn materialize(
    declarations: Vec<Declaration<resolve_opens::OpensResolvedFamily>>,
    semantic_modules: semantic_modules::SemanticModules,
    mut type_table: LookupMap<ast::NodeId, ast::TypeScheme>,
    schemes: LookupMap<String, ast::TypeScheme>,
    warnings: Vec<TypeWarning>,
    builtin_names: Vec<String>,
    mut name_resolution: name_resolution::NameResolution,
) -> Result<(
    Vec<Declaration<TypeCheckedFamily>>,
    ast::TypedGlobal<ast::TypedDefinition, ast::TypedTree>,
)> {
    let mut support_definitions = Vec::new();
    for (module_name, definition) in semantic_modules.get_all_module_declarations() {
        support_definitions.push(ast::SupportDefinition {
            namespace: Some(module_name.to_string()),
            definition: materialize_definition(
                definition.clone(),
                &format!("{}.{}", module_name, definition.name),
                &mut type_table,
                &schemes,
                &mut name_resolution,
            )?,
        });
    }
    for definition in semantic_modules.get_prelude_function_declarations() {
        support_definitions.push(ast::SupportDefinition {
            namespace: None,
            definition: materialize_definition(
                definition.clone(),
                &definition.name,
                &mut type_table,
                &schemes,
                &mut name_resolution,
            )?,
        });
    }

    let mut typed_declarations = Vec::with_capacity(declarations.len());
    for declaration in declarations {
        let declaration = match declaration {
            Declaration::Decl(definition) => Some(Declaration::Decl(materialize_definition(
                definition,
                "",
                &mut type_table,
                &schemes,
                &mut name_resolution,
            )?)),
            Declaration::Entry(entry) => Some(Declaration::Entry(materialize_entry(
                entry,
                &mut type_table,
                &schemes,
                &mut name_resolution,
            )?)),
            Declaration::Extern(external) => Some(Declaration::Extern(materialize_external(
                external,
                &schemes,
                &mut name_resolution,
            )?)),
            Declaration::Frontend(_) => None,
        };
        if let Some(declaration) = declaration {
            typed_declarations.push(declaration);
        }
    }

    Ok((
        typed_declarations,
        ast::TypedGlobal {
            support_definitions,
            symbols: std::mem::take(&mut name_resolution.symbols),
            warnings,
            builtin_names,
        },
    ))
}

fn materialize_definition(
    definition: ast::Decl,
    qualified_name: &str,
    type_table: &mut LookupMap<ast::NodeId, ast::TypeScheme>,
    schemes: &LookupMap<String, ast::TypeScheme>,
    name_resolution: &mut name_resolution::NameResolution,
) -> Result<ast::Decl<ast::TypedDefinition, ast::TypedTree>> {
    let scheme_name = if qualified_name.is_empty() { definition.name.as_str() } else { qualified_name };
    let scheme = schemes.get(scheme_name).cloned().ok_or_else(|| {
        err_type_at!(
            definition.name_span,
            "type checker did not produce a scheme for '{}'",
            scheme_name
        )
    })?;
    let Some(identity) = name_resolution.take_declaration(scheme_name, definition.name_span) else {
        return Err(err_type_at!(
            definition.name_span,
            "name resolution did not assign an identity to '{}'",
            scheme_name
        ));
    };
    definition.try_rebuild(
        |syntax, _, _| {
            Ok(ast::TypedDefinition {
                source: ast::NameResolvedDefinition {
                    syntax,
                    symbol: identity.symbol,
                    package: identity.package,
                },
                scheme,
            })
        },
        |params, body| {
            Ok((
                params
                    .into_iter()
                    .map(|pattern| materialize_pattern(pattern, type_table, name_resolution))
                    .collect::<Result<_>>()?,
                materialize_expression(body, type_table, name_resolution)?,
            ))
        },
    )
}

fn materialize_entry(
    entry: ast::EntryDecl<ast::ResolvedEntry, ast::SourceTree, interface::ResolvedAttribute>,
    type_table: &mut LookupMap<ast::NodeId, ast::TypeScheme>,
    schemes: &LookupMap<String, ast::TypeScheme>,
    name_resolution: &mut name_resolution::NameResolution,
) -> Result<ast::EntryDecl<ast::TypedEntry, ast::TypedTree, interface::ResolvedAttribute>> {
    let scheme = schemes.get(&entry.name).cloned().ok_or_else(|| {
        err_type_at!(
            entry.name_span,
            "type checker did not produce a scheme for entry '{}'",
            entry.name
        )
    })?;
    let Some(identity) = name_resolution.take_declaration(&entry.name, entry.name_span) else {
        return Err(err_type_at!(
            entry.name_span,
            "name resolution did not assign an identity to entry '{}'",
            entry.name
        ));
    };
    entry.try_rebuild(
        |source, _, _| {
            Ok(ast::TypedEntry {
                source: ast::NameResolvedEntry {
                    source,
                    symbol: identity.symbol,
                    package: identity.package,
                },
                scheme,
            })
        },
        |params, body| {
            Ok((
                params
                    .into_iter()
                    .map(|pattern| materialize_pattern(pattern, type_table, name_resolution))
                    .collect::<Result<_>>()?,
                materialize_expression(body, type_table, name_resolution)?,
            ))
        },
    )
}

fn materialize_external(
    external: ast::ExternDecl,
    schemes: &LookupMap<String, ast::TypeScheme>,
    name_resolution: &mut name_resolution::NameResolution,
) -> Result<ast::ExternDecl<ast::TypedExtern>> {
    let scheme = schemes.get(&external.name).cloned().ok_or_else(|| {
        err_type_at!(
            external.data.span,
            "type checker did not produce a scheme for '{}'",
            external.name
        )
    })?;
    let Some(identity) = name_resolution.take_declaration(&external.name, external.data.span) else {
        return Err(err_type_at!(
            external.data.span,
            "name resolution did not assign an identity to extern '{}'",
            external.name
        ));
    };
    external.try_map_data(|syntax, _| {
        Ok(ast::TypedExtern {
            source: ast::NameResolvedExtern {
                syntax,
                symbol: identity.symbol,
                package: identity.package,
            },
            scheme,
        })
    })
}

fn materialize_pattern<A>(
    pattern: ast::Pattern<ast::SourceTree, A>,
    type_table: &mut LookupMap<ast::NodeId, ast::TypeScheme>,
    name_resolution: &mut name_resolution::NameResolution,
) -> Result<ast::Pattern<ast::TypedTree, A>> {
    ast::rebuild::pattern(
        pattern,
        &mut |header| typed_header(header, type_table),
        &mut |header, source| {
            let symbol = name_resolution
                .bindings
                .remove(&(header.id, source.clone()))
                .ok_or_else(|| err_type_at!(header.span, "name resolution missed binding '{}'", source))?;
            Ok(ast::ResolvedBinding { symbol, source })
        },
    )
}

fn materialize_expression(
    expression: ast::Expression,
    type_table: &mut LookupMap<ast::NodeId, ast::TypeScheme>,
    name_resolution: &mut name_resolution::NameResolution,
) -> Result<ast::Expression<ast::TypedTree>> {
    let values = &mut name_resolution.values;
    let bindings = &mut name_resolution.bindings;
    ast::rebuild::expression(
        expression,
        &mut |header| typed_header(header, type_table),
        &mut |header, identifier| {
            let resolution = match values.remove(&header.id).ok_or_else(|| {
                err_type_at!(
                    header.span,
                    "name resolution missed identifier '{}'",
                    identifier.name
                )
            })? {
                ResolvedValueRef::Symbol(symbol) => ast::IdentifierResolution::Symbol(symbol),
                ResolvedValueRef::Builtin { id, overload_idx } => ast::IdentifierResolution::Builtin {
                    id,
                    overload_idx: overload_idx.ok_or_else(|| {
                        err_type_at!(
                            header.span,
                            "builtin '{}' has no selected overload",
                            identifier.name
                        )
                    })?,
                },
                ResolvedValueRef::VecConstructor {
                    arity,
                    component_conversion,
                    ..
                } => ast::IdentifierResolution::VecConstructor {
                    arity,
                    component_conversion: component_conversion.ok_or_else(|| {
                        err_type_at!(header.span, "vector constructor conversion was not resolved")
                    })?,
                },
                ResolvedValueRef::Soac(kind) => ast::IdentifierResolution::Soac(match kind {
                    name_resolution::SoacKind::Replicate => ast::SoacKind::Replicate,
                    name_resolution::SoacKind::Map => ast::SoacKind::Map,
                    name_resolution::SoacKind::Reduce => ast::SoacKind::Reduce,
                    name_resolution::SoacKind::Scan => ast::SoacKind::Scan,
                    name_resolution::SoacKind::Filter => ast::SoacKind::Filter,
                    name_resolution::SoacKind::Zip => ast::SoacKind::Zip,
                    name_resolution::SoacKind::ReduceByIndex => ast::SoacKind::ReduceByIndex,
                    name_resolution::SoacKind::Scatter => ast::SoacKind::Scatter,
                    name_resolution::SoacKind::BucketScatter(rank) => ast::SoacKind::BucketScatter(rank),
                }),
            };
            Ok(ast::TypedIdentifier {
                source: identifier,
                resolution,
            })
        },
        &mut |header, source| {
            let symbol = bindings
                .remove(&(header.id, source.clone()))
                .ok_or_else(|| err_type_at!(header.span, "name resolution missed binding '{}'", source))?;
            Ok(ast::ResolvedBinding { symbol, source })
        },
        &mut |_header, hole| Ok(ast::ExprKind::TypeHole(hole)),
    )
}

fn typed_header(
    header: ast::Header,
    type_table: &mut LookupMap<ast::NodeId, ast::TypeScheme>,
) -> Result<ast::TypedHeader> {
    let ty = type_table.remove(&header.id).ok_or_else(|| {
        err_type_at!(
            header.span,
            "type checker did not record a type for AST node {:?}",
            header.id
        )
    })?;
    Ok(ast::TypedHeader {
        id: header.id,
        span: header.span,
        ty,
    })
}
