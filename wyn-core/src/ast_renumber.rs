//! Deep-clone an AST subtree with freshly allocated `NodeId`s.
//!
//! Used by semantic module elaboration after a functor instantiation: each
//! instantiation needs its own NodeId space so downstream NodeId-keyed
//! side tables (e.g. `NameResolution`) don't see collisions between
//! instantiations of the same source body.
//!
//! Spans are preserved; only `Header.id` is freshened. Type annotations
//! (`crate::types::Type`) don't carry NodeIds and are cloned by value.
//!
//! Prefer these helpers over `expr.clone()` whenever the cloned subtree
//! will appear alongside the original in the same compilation unit.

use crate::ast::{self, Decl, Expression, NodeCounter, Pattern};
use ast::rebuild::VisitorMut;

struct FreshIds<'a>(&'a mut NodeCounter);
impl VisitorMut for FreshIds<'_> {
    fn expression(&mut self, e: &mut Expression) {
        e.h.id = self.0.next_id();
        ast::rebuild::walk_expression(self, e);
    }
    fn pattern<A>(&mut self, p: &mut Pattern<ast::SourceTree, A>) {
        p.h.id = self.0.next_id();
        ast::rebuild::walk_pattern(self, p);
    }
}

/// Clone an expression, preserving spans and refreshing every node ID.
pub fn clone_expr_fresh_ids(expr: &Expression, nc: &mut NodeCounter) -> Expression {
    let mut cloned = expr.clone();
    FreshIds(nc).expression(&mut cloned);
    cloned
}

pub fn clone_decl_fresh_ids(decl: &Decl, nc: &mut NodeCounter) -> Decl {
    Decl {
        data: decl.data.clone(),
        name: decl.name.clone(),
        name_span: decl.name_span,
        size_params: decl.size_params.clone(),
        type_params: decl.type_params.clone(),
        params: decl.params.iter().map(|pattern| clone_pattern_fresh_ids(pattern, nc)).collect(),
        ty: decl.ty.clone(),
        body: clone_expr_fresh_ids(&decl.body, nc),
        param_diets: decl.param_diets.clone(),
        return_diet: decl.return_diet.clone(),
    }
}

/// Clone a pattern, preserving spans and refreshing every node ID.
pub fn clone_pattern_fresh_ids(pat: &Pattern, nc: &mut NodeCounter) -> Pattern {
    let mut cloned = pat.clone();
    FreshIds(nc).pattern(&mut cloned);
    cloned
}

#[cfg(test)]
mod tests {
    use super::*;
    use ast::{ExprKind, Header, LambdaExpr, Node, NodeId, PatternKind, Span};

    #[test]
    fn cloning_refreshes_nested_expression_and_pattern_ids_in_preorder() {
        let header = Header {
            id: NodeId(999),
            span: Span::generated(),
        };
        let original: Expression = Node {
            h: header.clone(),
            kind: ExprKind::Lambda(LambdaExpr {
                params: vec![Node {
                    h: header.clone(),
                    kind: PatternKind::Tuple(vec![Node {
                        h: header.clone(),
                        kind: PatternKind::Wildcard,
                    }]),
                }],
                body: Box::new(Node {
                    h: header,
                    kind: ExprKind::Unit,
                }),
            }),
        };
        let mut ids = NodeCounter::new();
        let first = ids.peek_id().0;
        for offset in [0, 4] {
            let cloned = clone_expr_fresh_ids(&original, &mut ids);
            let ExprKind::Lambda(lambda) = &cloned.kind else {
                panic!()
            };
            let PatternKind::Tuple(children) = &lambda.params[0].kind else {
                panic!()
            };
            assert_eq!(
                [
                    cloned.h.id.0,
                    lambda.params[0].h.id.0,
                    children[0].h.id.0,
                    lambda.body.h.id.0,
                ],
                [
                    first + offset,
                    first + offset + 1,
                    first + offset + 2,
                    first + offset + 3
                ]
            );
            assert_eq!(cloned.h.span, original.h.span);
            assert_eq!(lambda.body.h.span, original.h.span);
        }
        assert_eq!(original.h.id, NodeId(999));
        assert_eq!(ids.peek_id().0, first + 8);
    }
}
