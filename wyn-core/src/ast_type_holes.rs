//! AST checkpoint after source `???` expressions have been handled.

#![deny(clippy::let_underscore_must_use)]

use crate::ast;
use crate::error;
use crate::error::{CompilerError, TypeHoleError, TypeHoleErrors};
use crate::interface;
use crate::types;

pub type HolesResolvedFamily = ast::AstFamily<
    ast::HolesResolvedTree,
    ast::TypedDefinition,
    ast::TypedEntry,
    interface::ResolvedAttribute,
    ast::TypedExtern,
    std::convert::Infallible,
>;

/// Fully typed AST accepted by AST-to-TLC lowering. Both rejecting a program
/// that contains holes and replacing every hole with a typed default produce
/// this single checkpoint.
#[derive(Debug, Clone, Copy)]
pub enum HolesResolvedTag {}
pub type HolesResolved = ast::Program<
    HolesResolvedTag,
    HolesResolvedFamily,
    ast::TypedGlobal<ast::TypedDefinition, ast::HolesResolvedTree>,
>;

pub fn reject_type_holes(program: types::run::TypeChecked) -> error::Result<HolesResolved> {
    let holes: Vec<_> = program
        .global_context
        .warnings
        .iter()
        .map(|warning| match warning {
            types::checker::TypeWarning::TypeHoleFilled { inferred_type, span } => (inferred_type, span),
        })
        .collect();
    if !holes.is_empty() {
        return Err(CompilerError::TypeHole(TypeHoleErrors::new(
            holes.into_iter().map(|(ty, span)| TypeHoleError {
                message: format!("type hole inferred as `{}`", types::format_type(ty)),
                span: Some(*span),
            }),
        )));
    }
    rebuild(program, &mut |header, _hole, _node_ids| {
        Err(CompilerError::TypeHole(TypeHoleErrors::single(
            "type checker omitted a warning for a stored type hole".to_owned(),
            Some(header.span),
        )))
    })
}

pub fn fill_type_holes(program: types::run::TypeChecked) -> error::Result<HolesResolved> {
    let mut errors = Vec::new();
    let rebuilt = rebuild(program, &mut |header, _hole, node_ids| {
        Ok(default_kind(header, node_ids, &mut errors))
    })?;
    if errors.is_empty() {
        Ok(rebuilt)
    } else {
        Err(CompilerError::TypeHole(TypeHoleErrors::new(errors)))
    }
}

fn rebuild(
    program: types::run::TypeChecked,
    hole: &mut impl FnMut(
        &ast::TypedHeader,
        ast::TypeHole,
        &mut ast::NodeCounter,
    ) -> error::Result<ast::ExprKind<ast::HolesResolvedTree>>,
) -> error::Result<HolesResolved> {
    program.try_rebuild(|declarations, global_context, node_ids| {
        let declarations = declarations
            .into_iter()
            .map(|declaration| match declaration {
                ast::Declaration::Decl(definition) => Ok(ast::Declaration::Decl(rebuild_definition(
                    definition, node_ids, hole,
                )?)),
                ast::Declaration::Entry(entry) => {
                    Ok(ast::Declaration::Entry(rebuild_entry(entry, node_ids, hole)?))
                }
                ast::Declaration::Extern(external) => Ok(ast::Declaration::Extern(external)),
                ast::Declaration::Frontend(never) => match never {},
            })
            .collect::<error::Result<_>>()?;
        let support_definitions = global_context
            .support_definitions
            .into_iter()
            .map(|support| {
                support.try_map_definition(|definition| rebuild_definition(definition, node_ids, hole))
            })
            .collect::<error::Result<_>>()?;
        Ok((
            declarations,
            ast::TypedGlobal {
                support_definitions,
                symbols: global_context.symbols,
                warnings: global_context.warnings,
                builtin_names: global_context.builtin_names,
            },
        ))
    })
}

fn rebuild_definition(
    definition: ast::Decl<ast::TypedDefinition, ast::TypedTree>,
    node_ids: &mut ast::NodeCounter,
    hole: &mut impl FnMut(
        &ast::TypedHeader,
        ast::TypeHole,
        &mut ast::NodeCounter,
    ) -> error::Result<ast::ExprKind<ast::HolesResolvedTree>>,
) -> error::Result<ast::Decl<ast::TypedDefinition, ast::HolesResolvedTree>> {
    definition.try_rebuild(
        |data, _, _| Ok(data),
        |params, body| {
            Ok((
                params.into_iter().map(rebuild_pattern).collect(),
                rebuild_expression(body, node_ids, hole)?,
            ))
        },
    )
}

fn rebuild_entry(
    entry: ast::EntryDecl<ast::TypedEntry, ast::TypedTree, interface::ResolvedAttribute>,
    node_ids: &mut ast::NodeCounter,
    hole: &mut impl FnMut(
        &ast::TypedHeader,
        ast::TypeHole,
        &mut ast::NodeCounter,
    ) -> error::Result<ast::ExprKind<ast::HolesResolvedTree>>,
) -> error::Result<ast::EntryDecl<ast::TypedEntry, ast::HolesResolvedTree, interface::ResolvedAttribute>> {
    entry.try_rebuild(
        |data, _, _| Ok(data),
        |params, body| {
            Ok((
                params.into_iter().map(rebuild_pattern).collect(),
                rebuild_expression(body, node_ids, hole)?,
            ))
        },
    )
}

fn rebuild_pattern<A>(pattern: ast::Pattern<ast::TypedTree, A>) -> ast::Pattern<ast::HolesResolvedTree, A> {
    ast::rebuild::pattern(pattern, &mut Ok, &mut |_header, binding| Ok(binding))
        .unwrap_or_else(|never: std::convert::Infallible| match never {})
}

fn rebuild_expression(
    expression: ast::Expression<ast::TypedTree>,
    node_ids: &mut ast::NodeCounter,
    hole: &mut impl FnMut(
        &ast::TypedHeader,
        ast::TypeHole,
        &mut ast::NodeCounter,
    ) -> error::Result<ast::ExprKind<ast::HolesResolvedTree>>,
) -> error::Result<ast::Expression<ast::HolesResolvedTree>> {
    ast::rebuild::expression(
        expression,
        &mut Ok,
        &mut |_header, identifier| Ok(identifier),
        &mut |_header, binding| Ok(binding),
        &mut |header, value| hole(header, value, node_ids),
    )
}

fn default_kind(
    header: &ast::TypedHeader,
    node_ids: &mut ast::NodeCounter,
    errors: &mut Vec<TypeHoleError>,
) -> ast::ExprKind<ast::HolesResolvedTree> {
    let ty = scheme_type(&header.ty);
    match ty {
        ast::Type::Constructed(ast::TypeName::Int(_) | ast::TypeName::UInt(_), args) if args.is_empty() => {
            ast::ExprKind::IntLiteral("0".into())
        }
        ast::Type::Constructed(ast::TypeName::Float(_), args) if args.is_empty() => {
            ast::ExprKind::FloatLiteral(0.0)
        }
        ast::Type::Constructed(ast::TypeName::Bool, args) if args.is_empty() => {
            ast::ExprKind::BoolLiteral(false)
        }
        ast::Type::Constructed(ast::TypeName::Unit, _) => ast::ExprKind::Unit,
        ast::Type::Constructed(ast::TypeName::Tuple(_), elements) => ast::ExprKind::Tuple(
            elements
                .iter()
                .map(|element| default_expression(element, header.span, node_ids, errors))
                .collect(),
        ),
        ast::Type::Constructed(ast::TypeName::Vec, args) if args.len() == 2 => {
            match size_literal(&args[1]) {
                Some(size) => ast::ExprKind::VecMatLiteral(
                    (0..size)
                        .map(|_| default_expression(&args[0], header.span, node_ids, errors))
                        .collect(),
                ),
                None => default_error(header, errors, "vector size is not a literal"),
            }
        }
        ast::Type::Constructed(ast::TypeName::Array, args) if args.len() == 4 => {
            let composite = matches!(
                &args[1],
                ast::Type::Constructed(ast::TypeName::ArrayVariantComposite, _)
            );
            match (composite, size_literal(&args[2])) {
                (true, Some(size)) => ast::ExprKind::ArrayLiteral(
                    (0..size)
                        .map(|_| default_expression(&args[0], header.span, node_ids, errors))
                        .collect(),
                ),
                (true, None) => default_error(header, errors, "array size is not a literal"),
                (false, _) => default_error(header, errors, "only Composite arrays can be default-filled"),
            }
        }
        ast::Type::Variable(_) => default_error(header, errors, "hole has an unresolved type variable"),
        ast::Type::Constructed(ast::TypeName::Arrow, _) => {
            default_error(header, errors, "cannot synthesize a default function value")
        }
        _ => default_error(header, errors, "no default value is available for this type"),
    }
}

fn default_expression(
    ty: &ast::Type,
    span: ast::Span,
    node_ids: &mut ast::NodeCounter,
    errors: &mut Vec<TypeHoleError>,
) -> ast::Expression<ast::HolesResolvedTree> {
    let header = ast::TypedHeader {
        id: node_ids.next_id(),
        span,
        ty: ast::TypeScheme::Monotype(ty.clone()),
    };
    let kind = default_kind(&header, node_ids, errors);
    ast::Node { h: header, kind }
}

fn default_error(
    header: &ast::TypedHeader,
    errors: &mut Vec<TypeHoleError>,
    reason: &str,
) -> ast::ExprKind<ast::HolesResolvedTree> {
    errors.push(TypeHoleError {
        message: format!("--fill-holes: {reason} (type: {:?})", scheme_type(&header.ty)),
        span: Some(header.span),
    });
    ast::ExprKind::IntLiteral("0".into())
}

fn scheme_type(scheme: &ast::TypeScheme) -> &ast::Type {
    match scheme {
        ast::TypeScheme::Monotype(ty) => ty,
        ast::TypeScheme::Polytype { body, .. } => scheme_type(body),
    }
}

fn size_literal(ty: &ast::Type) -> Option<usize> {
    match ty {
        ast::Type::Constructed(ast::TypeName::Size(size), _) => Some(*size),
        _ => None,
    }
}
