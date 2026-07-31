//! Mechanical reconstruction for transitions that change every recursive AST
//! node's stored header, identifier, or type-hole payload.

use super::{
    ExprKind, Expression, IfExpr, LambdaExpr, LetInExpr, LoopExpr, LoopForm, MatchCase, MatchExpr, Node,
    Pattern, PatternKind, RangeExpr, RecordPatternField, RecordPatternTarget, SliceExpr, TreeFamily,
};

#[cfg(test)]
#[path = "rebuild_tests.rs"]
mod tests;

pub fn expression<From, To, E>(
    expression: Expression<From>,
    header: &mut impl FnMut(From::Header) -> Result<To::Header, E>,
    identifier: &mut impl FnMut(&From::Header, From::Identifier) -> Result<To::Identifier, E>,
    binding: &mut impl FnMut(&From::Header, From::Binding) -> Result<To::Binding, E>,
    type_hole: &mut impl FnMut(&From::Header, From::TypeHole) -> Result<ExprKind<To>, E>,
) -> Result<Expression<To>, E>
where
    From: TreeFamily,
    To: TreeFamily,
{
    let Node { h, kind } = expression;
    let kind = expression_kind(kind, &h, header, identifier, binding, type_hole)?;
    Ok(Node { h: header(h)?, kind })
}

fn expression_kind<From, To, E>(
    kind: ExprKind<From>,
    source_header: &From::Header,
    header: &mut impl FnMut(From::Header) -> Result<To::Header, E>,
    identifier: &mut impl FnMut(&From::Header, From::Identifier) -> Result<To::Identifier, E>,
    binding: &mut impl FnMut(&From::Header, From::Binding) -> Result<To::Binding, E>,
    type_hole: &mut impl FnMut(&From::Header, From::TypeHole) -> Result<ExprKind<To>, E>,
) -> Result<ExprKind<To>, E>
where
    From: TreeFamily,
    To: TreeFamily,
{
    Ok(match kind {
        ExprKind::IntLiteral(value) => ExprKind::IntLiteral(value),
        ExprKind::FloatLiteral(value) => ExprKind::FloatLiteral(value),
        ExprKind::BoolLiteral(value) => ExprKind::BoolLiteral(value),
        ExprKind::Unit => ExprKind::Unit,
        ExprKind::Identifier(value) => ExprKind::Identifier(identifier(source_header, value)?),
        ExprKind::TypeHole(value) => type_hole(source_header, value)?,
        ExprKind::ArrayLiteral(values) => ExprKind::ArrayLiteral(
            values
                .into_iter()
                .map(|value| expression(value, header, identifier, binding, type_hole))
                .collect::<Result<_, _>>()?,
        ),
        ExprKind::VecMatLiteral(values) => ExprKind::VecMatLiteral(
            values
                .into_iter()
                .map(|value| expression(value, header, identifier, binding, type_hole))
                .collect::<Result<_, _>>()?,
        ),
        ExprKind::ArrayIndex(array, index) => ExprKind::ArrayIndex(
            Box::new(expression(*array, header, identifier, binding, type_hole)?),
            Box::new(expression(*index, header, identifier, binding, type_hole)?),
        ),
        ExprKind::ArrayWith { array, index, value } => ExprKind::ArrayWith {
            array: Box::new(expression(*array, header, identifier, binding, type_hole)?),
            index: Box::new(expression(*index, header, identifier, binding, type_hole)?),
            value: Box::new(expression(*value, header, identifier, binding, type_hole)?),
        },
        ExprKind::VecWith {
            target,
            components,
            op,
            value,
        } => ExprKind::VecWith {
            target: Box::new(expression(*target, header, identifier, binding, type_hole)?),
            components,
            op,
            value: Box::new(expression(*value, header, identifier, binding, type_hole)?),
        },
        ExprKind::RecordWith { record, path, value } => ExprKind::RecordWith {
            record: Box::new(expression(*record, header, identifier, binding, type_hole)?),
            path,
            value: Box::new(expression(*value, header, identifier, binding, type_hole)?),
        },
        ExprKind::BinaryOp(op, left, right) => ExprKind::BinaryOp(
            op,
            Box::new(expression(*left, header, identifier, binding, type_hole)?),
            Box::new(expression(*right, header, identifier, binding, type_hole)?),
        ),
        ExprKind::UnaryOp(op, value) => ExprKind::UnaryOp(
            op,
            Box::new(expression(*value, header, identifier, binding, type_hole)?),
        ),
        ExprKind::Tuple(values) => ExprKind::Tuple(
            values
                .into_iter()
                .map(|value| expression(value, header, identifier, binding, type_hole))
                .collect::<Result<_, _>>()?,
        ),
        ExprKind::RecordLiteral(fields) => ExprKind::RecordLiteral(
            fields
                .into_iter()
                .map(|(name, value)| Ok((name, expression(value, header, identifier, binding, type_hole)?)))
                .collect::<Result<_, E>>()?,
        ),
        ExprKind::Lambda(lambda) => ExprKind::Lambda(LambdaExpr {
            params: lambda
                .params
                .into_iter()
                .map(|value| pattern(value, header, binding))
                .collect::<Result<_, _>>()?,
            body: Box::new(expression(*lambda.body, header, identifier, binding, type_hole)?),
        }),
        ExprKind::Application(function, arguments) => ExprKind::Application(
            Box::new(expression(*function, header, identifier, binding, type_hole)?),
            arguments
                .into_iter()
                .map(|value| expression(value, header, identifier, binding, type_hole))
                .collect::<Result<_, _>>()?,
        ),
        ExprKind::LetIn(let_in) => ExprKind::LetIn(LetInExpr {
            pattern: pattern(let_in.pattern, header, binding)?,
            ty: let_in.ty,
            value: Box::new(expression(*let_in.value, header, identifier, binding, type_hole)?),
            body: Box::new(expression(*let_in.body, header, identifier, binding, type_hole)?),
        }),
        ExprKind::FieldAccess(value, field) => ExprKind::FieldAccess(
            Box::new(expression(*value, header, identifier, binding, type_hole)?),
            field,
        ),
        ExprKind::If(if_expression) => ExprKind::If(IfExpr {
            condition: Box::new(expression(
                *if_expression.condition,
                header,
                identifier,
                binding,
                type_hole,
            )?),
            then_branch: Box::new(expression(
                *if_expression.then_branch,
                header,
                identifier,
                binding,
                type_hole,
            )?),
            else_branch: Box::new(expression(
                *if_expression.else_branch,
                header,
                identifier,
                binding,
                type_hole,
            )?),
        }),
        ExprKind::Loop(loop_expression) => ExprKind::Loop(LoopExpr {
            pattern: pattern(loop_expression.pattern, header, binding)?,
            init: loop_expression
                .init
                .map(|value| expression(*value, header, identifier, binding, type_hole).map(Box::new))
                .transpose()?,
            form: loop_form(loop_expression.form, header, identifier, binding, type_hole)?,
            body: Box::new(expression(
                *loop_expression.body,
                header,
                identifier,
                binding,
                type_hole,
            )?),
        }),
        ExprKind::Match(match_expression) => ExprKind::Match(MatchExpr {
            scrutinee: Box::new(expression(
                *match_expression.scrutinee,
                header,
                identifier,
                binding,
                type_hole,
            )?),
            cases: match_expression
                .cases
                .into_iter()
                .map(|case| {
                    Ok(MatchCase {
                        pattern: pattern(case.pattern, header, binding)?,
                        body: Box::new(expression(*case.body, header, identifier, binding, type_hole)?),
                    })
                })
                .collect::<Result<_, E>>()?,
        }),
        ExprKind::Constructor(name, arguments) => ExprKind::Constructor(
            name,
            arguments
                .into_iter()
                .map(|value| expression(value, header, identifier, binding, type_hole))
                .collect::<Result<_, _>>()?,
        ),
        ExprKind::Range(range) => ExprKind::Range(RangeExpr {
            start: Box::new(expression(*range.start, header, identifier, binding, type_hole)?),
            step: range
                .step
                .map(|value| expression(*value, header, identifier, binding, type_hole).map(Box::new))
                .transpose()?,
            end: Box::new(expression(*range.end, header, identifier, binding, type_hole)?),
            kind: range.kind,
        }),
        ExprKind::Slice(slice) => ExprKind::Slice(SliceExpr {
            array: Box::new(expression(*slice.array, header, identifier, binding, type_hole)?),
            start: slice
                .start
                .map(|value| expression(*value, header, identifier, binding, type_hole).map(Box::new))
                .transpose()?,
            end: slice
                .end
                .map(|value| expression(*value, header, identifier, binding, type_hole).map(Box::new))
                .transpose()?,
        }),
        ExprKind::TypeAscription(value, ty) => ExprKind::TypeAscription(
            Box::new(expression(*value, header, identifier, binding, type_hole)?),
            ty,
        ),
        ExprKind::TypeCoercion(value, ty) => ExprKind::TypeCoercion(
            Box::new(expression(*value, header, identifier, binding, type_hole)?),
            ty,
        ),
    })
}

fn loop_form<From, To, E>(
    form: LoopForm<From>,
    header: &mut impl FnMut(From::Header) -> Result<To::Header, E>,
    identifier: &mut impl FnMut(&From::Header, From::Identifier) -> Result<To::Identifier, E>,
    binding: &mut impl FnMut(&From::Header, From::Binding) -> Result<To::Binding, E>,
    type_hole: &mut impl FnMut(&From::Header, From::TypeHole) -> Result<ExprKind<To>, E>,
) -> Result<LoopForm<To>, E>
where
    From: TreeFamily,
    To: TreeFamily,
{
    Ok(match form {
        LoopForm::For(pattern_value, bound) => LoopForm::For(
            pattern(pattern_value, header, binding)?,
            Box::new(expression(*bound, header, identifier, binding, type_hole)?),
        ),
        LoopForm::ForIn(item, iterable) => LoopForm::ForIn(
            pattern(item, header, binding)?,
            Box::new(expression(*iterable, header, identifier, binding, type_hole)?),
        ),
        LoopForm::While(condition) => LoopForm::While(Box::new(expression(
            *condition, header, identifier, binding, type_hole,
        )?)),
    })
}

pub fn pattern<From: TreeFamily, To: TreeFamily, A, E>(
    pattern: Pattern<From, A>,
    header: &mut impl FnMut(From::Header) -> Result<To::Header, E>,
    binding: &mut impl FnMut(&From::Header, From::Binding) -> Result<To::Binding, E>,
) -> Result<Pattern<To, A>, E> {
    let Node { h, kind } = pattern;
    let kind = match kind {
        PatternKind::Name(name) => PatternKind::Name(binding(&h, name)?),
        PatternKind::Wildcard => PatternKind::Wildcard,
        PatternKind::Literal(value) => PatternKind::Literal(value),
        PatternKind::Unit => PatternKind::Unit,
        PatternKind::Tuple(patterns) => PatternKind::Tuple(
            patterns
                .into_iter()
                .map(|value| self::pattern(value, header, binding))
                .collect::<Result<_, _>>()?,
        ),
        PatternKind::Vec(patterns) => PatternKind::Vec(
            patterns
                .into_iter()
                .map(|value| self::pattern(value, header, binding))
                .collect::<Result<_, _>>()?,
        ),
        PatternKind::Record(fields) => PatternKind::Record(
            fields
                .into_iter()
                .map(|field| {
                    Ok(RecordPatternField {
                        field: field.field,
                        target: match field.target {
                            RecordPatternTarget::Shorthand(value) => {
                                RecordPatternTarget::Shorthand(binding(&h, value)?)
                            }
                            RecordPatternTarget::Pattern(value) => {
                                RecordPatternTarget::Pattern(self::pattern(value, header, binding)?)
                            }
                        },
                    })
                })
                .collect::<Result<_, E>>()?,
        ),
        PatternKind::Constructor(name, patterns) => PatternKind::Constructor(
            name,
            patterns
                .into_iter()
                .map(|value| self::pattern(value, header, binding))
                .collect::<Result<_, _>>()?,
        ),
        PatternKind::Typed(pattern, ty) => {
            PatternKind::Typed(Box::new(self::pattern(*pattern, header, binding)?), ty)
        }
        PatternKind::Attributed(attributes, pattern) => {
            PatternKind::Attributed(attributes, Box::new(self::pattern(*pattern, header, binding)?))
        }
    };
    Ok(Node { h: header(h)?, kind })
}
