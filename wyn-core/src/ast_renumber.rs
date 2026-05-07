//! Deep-clone an AST subtree with freshly allocated `NodeId`s.
//!
//! Used by `module_manager` after a functor instantiation: each
//! instantiation needs its own NodeId space so downstream NodeId-keyed
//! side tables (e.g. `NameResolution`) don't see collisions between
//! instantiations of the same source body.
//!
//! Spans are preserved; only `Header.id` is freshened. Type annotations
//! (`crate::types::Type`) don't carry NodeIds and are cloned by value.
//!
//! Prefer these helpers over `expr.clone()` whenever the cloned subtree
//! will appear alongside the original in the same compilation unit.

use crate::ast::{
    ExprKind, Expression, Header, IfExpr, LambdaExpr, LetInExpr, LoopExpr, LoopForm, MatchCase, MatchExpr,
    Node, NodeCounter, Pattern, PatternKind, RangeExpr, SliceExpr,
};

fn fresh_header(src: &Header, nc: &mut NodeCounter) -> Header {
    Header {
        id: nc.next_id(),
        span: src.span,
    }
}

/// Deep-clone `expr` with every `Header.id` reallocated from `nc`. Span
/// information is preserved. Recurses through every sub-expression and
/// pattern.
pub fn clone_expr_fresh_ids(expr: &Expression, nc: &mut NodeCounter) -> Expression {
    let h = fresh_header(&expr.h, nc);
    let kind = match &expr.kind {
        ExprKind::IntLiteral(s) => ExprKind::IntLiteral(s.clone()),
        ExprKind::FloatLiteral(f) => ExprKind::FloatLiteral(*f),
        ExprKind::BoolLiteral(b) => ExprKind::BoolLiteral(*b),
        ExprKind::Unit => ExprKind::Unit,
        ExprKind::Identifier(quals, name) => ExprKind::Identifier(quals.clone(), name.clone()),
        ExprKind::TypeHole => ExprKind::TypeHole,

        ExprKind::ArrayLiteral(es) => {
            ExprKind::ArrayLiteral(es.iter().map(|e| clone_expr_fresh_ids(e, nc)).collect())
        }
        ExprKind::VecMatLiteral(es) => {
            ExprKind::VecMatLiteral(es.iter().map(|e| clone_expr_fresh_ids(e, nc)).collect())
        }
        ExprKind::Tuple(es) => ExprKind::Tuple(es.iter().map(|e| clone_expr_fresh_ids(e, nc)).collect()),
        ExprKind::ArrayIndex(arr, idx) => ExprKind::ArrayIndex(
            Box::new(clone_expr_fresh_ids(arr, nc)),
            Box::new(clone_expr_fresh_ids(idx, nc)),
        ),
        ExprKind::ArrayWith { array, index, value } => ExprKind::ArrayWith {
            array: Box::new(clone_expr_fresh_ids(array, nc)),
            index: Box::new(clone_expr_fresh_ids(index, nc)),
            value: Box::new(clone_expr_fresh_ids(value, nc)),
        },
        ExprKind::VecWith {
            target,
            components,
            op,
            value,
        } => ExprKind::VecWith {
            target: Box::new(clone_expr_fresh_ids(target, nc)),
            components: components.clone(),
            op: op.clone(),
            value: Box::new(clone_expr_fresh_ids(value, nc)),
        },
        ExprKind::RecordWith { record, path, value } => ExprKind::RecordWith {
            record: Box::new(clone_expr_fresh_ids(record, nc)),
            path: path.clone(),
            value: Box::new(clone_expr_fresh_ids(value, nc)),
        },
        ExprKind::BinaryOp(op, lhs, rhs) => ExprKind::BinaryOp(
            op.clone(),
            Box::new(clone_expr_fresh_ids(lhs, nc)),
            Box::new(clone_expr_fresh_ids(rhs, nc)),
        ),
        ExprKind::UnaryOp(op, inner) => {
            ExprKind::UnaryOp(op.clone(), Box::new(clone_expr_fresh_ids(inner, nc)))
        }
        ExprKind::RecordLiteral(fields) => ExprKind::RecordLiteral(
            fields.iter().map(|(name, e)| (name.clone(), clone_expr_fresh_ids(e, nc))).collect(),
        ),
        ExprKind::Lambda(lam) => ExprKind::Lambda(LambdaExpr {
            params: lam.params.iter().map(|p| clone_pattern_fresh_ids(p, nc)).collect(),
            body: Box::new(clone_expr_fresh_ids(&lam.body, nc)),
        }),
        ExprKind::Application(func, args) => ExprKind::Application(
            Box::new(clone_expr_fresh_ids(func, nc)),
            args.iter().map(|a| clone_expr_fresh_ids(a, nc)).collect(),
        ),
        ExprKind::LetIn(let_in) => ExprKind::LetIn(LetInExpr {
            pattern: clone_pattern_fresh_ids(&let_in.pattern, nc),
            ty: let_in.ty.clone(),
            value: Box::new(clone_expr_fresh_ids(&let_in.value, nc)),
            body: Box::new(clone_expr_fresh_ids(&let_in.body, nc)),
        }),
        ExprKind::FieldAccess(obj, field) => {
            ExprKind::FieldAccess(Box::new(clone_expr_fresh_ids(obj, nc)), field.clone())
        }
        ExprKind::If(if_expr) => ExprKind::If(IfExpr {
            condition: Box::new(clone_expr_fresh_ids(&if_expr.condition, nc)),
            then_branch: Box::new(clone_expr_fresh_ids(&if_expr.then_branch, nc)),
            else_branch: Box::new(clone_expr_fresh_ids(&if_expr.else_branch, nc)),
        }),
        ExprKind::Loop(loop_expr) => ExprKind::Loop(LoopExpr {
            pattern: clone_pattern_fresh_ids(&loop_expr.pattern, nc),
            init: loop_expr.init.as_ref().map(|e| Box::new(clone_expr_fresh_ids(e, nc))),
            form: clone_loop_form(&loop_expr.form, nc),
            body: Box::new(clone_expr_fresh_ids(&loop_expr.body, nc)),
        }),
        ExprKind::Match(match_expr) => ExprKind::Match(MatchExpr {
            scrutinee: Box::new(clone_expr_fresh_ids(&match_expr.scrutinee, nc)),
            cases: match_expr
                .cases
                .iter()
                .map(|case| MatchCase {
                    pattern: clone_pattern_fresh_ids(&case.pattern, nc),
                    body: Box::new(clone_expr_fresh_ids(&case.body, nc)),
                })
                .collect(),
        }),
        ExprKind::Constructor(name, args) => ExprKind::Constructor(
            name.clone(),
            args.iter().map(|a| clone_expr_fresh_ids(a, nc)).collect(),
        ),
        ExprKind::Range(range) => ExprKind::Range(RangeExpr {
            start: Box::new(clone_expr_fresh_ids(&range.start, nc)),
            step: range.step.as_ref().map(|e| Box::new(clone_expr_fresh_ids(e, nc))),
            end: Box::new(clone_expr_fresh_ids(&range.end, nc)),
            kind: range.kind.clone(),
        }),
        ExprKind::Slice(slice) => ExprKind::Slice(SliceExpr {
            array: Box::new(clone_expr_fresh_ids(&slice.array, nc)),
            start: slice.start.as_ref().map(|e| Box::new(clone_expr_fresh_ids(e, nc))),
            end: slice.end.as_ref().map(|e| Box::new(clone_expr_fresh_ids(e, nc))),
        }),
        ExprKind::TypeAscription(inner, ty) => {
            ExprKind::TypeAscription(Box::new(clone_expr_fresh_ids(inner, nc)), ty.clone())
        }
        ExprKind::TypeCoercion(inner, ty) => {
            ExprKind::TypeCoercion(Box::new(clone_expr_fresh_ids(inner, nc)), ty.clone())
        }
    };
    Node { h, kind }
}

fn clone_loop_form(form: &LoopForm, nc: &mut NodeCounter) -> LoopForm {
    match form {
        LoopForm::While(cond) => LoopForm::While(Box::new(clone_expr_fresh_ids(cond, nc))),
        LoopForm::For(name, bound) => {
            LoopForm::For(name.clone(), Box::new(clone_expr_fresh_ids(bound, nc)))
        }
        LoopForm::ForIn(pat, iter) => LoopForm::ForIn(
            clone_pattern_fresh_ids(pat, nc),
            Box::new(clone_expr_fresh_ids(iter, nc)),
        ),
    }
}

/// Deep-clone `pat` with every `Header.id` reallocated from `nc`.
pub fn clone_pattern_fresh_ids(pat: &Pattern, nc: &mut NodeCounter) -> Pattern {
    let h = fresh_header(&pat.h, nc);
    let kind = match &pat.kind {
        PatternKind::Name(s) => PatternKind::Name(s.clone()),
        PatternKind::Wildcard => PatternKind::Wildcard,
        PatternKind::Literal(lit) => PatternKind::Literal(lit.clone()),
        PatternKind::Unit => PatternKind::Unit,
        PatternKind::Tuple(pats) => {
            PatternKind::Tuple(pats.iter().map(|p| clone_pattern_fresh_ids(p, nc)).collect())
        }
        PatternKind::Record(fields) => PatternKind::Record(
            fields
                .iter()
                .map(|f| crate::ast::RecordPatternField {
                    field: f.field.clone(),
                    pattern: f.pattern.as_ref().map(|p| clone_pattern_fresh_ids(p, nc)),
                })
                .collect(),
        ),
        PatternKind::Constructor(name, sub_patterns) => PatternKind::Constructor(
            name.clone(),
            sub_patterns.iter().map(|p| clone_pattern_fresh_ids(p, nc)).collect(),
        ),
        PatternKind::Typed(inner, ty) => {
            PatternKind::Typed(Box::new(clone_pattern_fresh_ids(inner, nc)), ty.clone())
        }
        PatternKind::Attributed(attrs, inner) => {
            PatternKind::Attributed(attrs.clone(), Box::new(clone_pattern_fresh_ids(inner, nc)))
        }
    };
    Node { h, kind }
}
