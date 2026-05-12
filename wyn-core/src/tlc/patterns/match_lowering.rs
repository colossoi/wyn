//! General `match` lowering. Replaces the previous sum-only
//! `compile_sum_match` with a pattern-kind-dispatch test-and-bind
//! lowering. Drives all match-arm pattern kinds the checker accepts
//! after coverage analysis (Maranget) ensures exhaustiveness.
//!
//! Strategy: build a single Bool condition + binding list per arm via
//! `compile_pattern_test`, then chain the arms right-to-left as nested
//! `If` terms. The last arm's condition is elided (exhaustiveness
//! guarantees it's reachable when every prior arm fails).

use crate::ast::{self, NodeId, PatternKind, PatternLiteral, Span, TypeName};
use crate::tlc::{PendingBinding, Term, TermKind, Transformer, VarRef};
use polytype::Type;

impl<'a> Transformer<'a> {
    /// Lower `match scrut case p1 -> e1 ... case pn -> en` to a chain
    /// of `If` tests over `scrut`, bound once to a fresh symbol.
    ///
    /// `result_ty` is the type each arm body produces.
    pub(in crate::tlc) fn compile_match(
        &mut self,
        match_expr: &ast::MatchExpr,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        debug_assert!(!match_expr.cases.is_empty(), "checker rejects empty match");

        let scrutinee = self.transform_expr(&match_expr.scrutinee);
        let scrut_ty = scrutinee.ty.clone();

        // Bind the scrutinee once so each arm reads it without
        // re-evaluating the source expression.
        let scrut_id = self.term_ids.next_id().0;
        let scrut_sym = self.define(&format!("_w_match_scrut_{}", scrut_id));
        let scrut_var = self.mk_term(scrut_ty.clone(), span, TermKind::Var(VarRef::Symbol(scrut_sym)));

        // Build right-to-left. Last arm's cond is elided: coverage
        // analysis upstream guarantees this arm matches whenever every
        // prior arm fails.
        let last = match_expr.cases.last().unwrap();
        let (_last_cond, last_bindings) = self.compile_pattern_test(&scrut_var, &last.pattern);
        let last_body = self.transform_expr(&last.body);
        let mut acc = self.apply_bindings_around(last_bindings, last_body, span);

        for case in match_expr.cases.iter().rev().skip(1) {
            let (cond, bindings) = self.compile_pattern_test(&scrut_var, &case.pattern);
            let body = self.transform_expr(&case.body);
            let then_branch = self.apply_bindings_around(bindings, body, span);
            acc = self.mk_term(
                result_ty.clone(),
                span,
                TermKind::If {
                    cond: Box::new(cond),
                    then_branch: Box::new(then_branch),
                    else_branch: Box::new(acc),
                },
            );
        }

        self.mk_term(
            result_ty,
            span,
            TermKind::Let {
                name: scrut_sym,
                name_ty: scrut_ty,
                rhs: Box::new(scrutinee),
                body: Box::new(acc),
            },
        )
    }

    /// Compile a pattern's `scrut`-matching condition + name bindings.
    /// Returns `(cond, bindings)`:
    ///   - `cond` is a Bool term; `true` for irrefutable patterns.
    ///   - `bindings` are the names to introduce for sub-pattern access.
    pub(super) fn compile_pattern_test(
        &mut self,
        scrut: &Term,
        pattern: &ast::Pattern,
    ) -> (Term, Vec<PendingBinding>) {
        let span = pattern.h.span;
        match &pattern.kind {
            PatternKind::Wildcard => (self.bool_lit(true, span), Vec::new()),
            PatternKind::Name(name) => {
                let sym = self.define(name);
                let bindings = vec![PendingBinding {
                    name: sym,
                    ty: scrut.ty.clone(),
                    expr: scrut.clone(),
                }];
                (self.bool_lit(true, span), bindings)
            }
            PatternKind::Typed(inner, _) | PatternKind::Attributed(_, inner) => {
                self.compile_pattern_test(scrut, inner)
            }
            PatternKind::Unit => (self.bool_lit(true, span), Vec::new()),
            PatternKind::Literal(lit) => {
                let lit_term = self.literal_term(lit, &scrut.ty, span);
                let cond = self.build_binop(
                    ast::BinaryOp { op: "==".to_string() },
                    scrut.clone(),
                    lit_term,
                    Self::bool_ty(),
                    span,
                );
                (cond, Vec::new())
            }
            PatternKind::Tuple(sub_patterns) => {
                let mut cond = self.bool_lit(true, span);
                let mut bindings = Vec::new();
                let elem_tys = match &scrut.ty {
                    Type::Constructed(TypeName::Tuple(_), args) => args.clone(),
                    other => panic!("BUG: tuple pattern against non-tuple type {:?}", other),
                };
                for (i, sub) in sub_patterns.iter().enumerate() {
                    let proj = self.mk_tuple_proj(scrut.clone(), i, elem_tys[i].clone(), span);
                    let (sub_cond, sub_bindings) = self.compile_pattern_test(&proj, sub);
                    cond = self.and_cond(cond, sub_cond, span);
                    bindings.extend(sub_bindings);
                }
                (cond, bindings)
            }
            PatternKind::Constructor(name, sub_patterns) => {
                let variants = match &scrut.ty {
                    Type::Constructed(TypeName::Tuple(_), _) => {
                        // Scrutinee already lowered from a sum to a
                        // flat tuple. Re-derive the layout from the
                        // raw type table — Constructor patterns carry
                        // their sum's raw shape via the pattern's
                        // type-table entry.
                        self.lookup_sum_variants_for_pattern(pattern.h.id).unwrap_or_else(|| {
                            panic!("BUG: Constructor pattern without sum-type entry in type table")
                        })
                    }
                    Type::Constructed(TypeName::Sum(v), _) => v.clone(),
                    other => panic!("BUG: Constructor pattern against non-sum type {:?}", other),
                };
                let layout = Self::sum_layout(&variants);
                let &(tag_value, payload_offset) =
                    layout.constructor_info.get(name).expect("BUG: constructor name not in sum layout");

                // Tag check: `scrut.0 == tag_value`.
                let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
                let tag_proj = self.mk_tuple_proj(scrut.clone(), 0, u32_ty.clone(), span);
                let tag_lit = self.mk_term(u32_ty, span, TermKind::IntLit(tag_value.to_string()));
                let mut cond = self.build_binop(
                    ast::BinaryOp { op: "==".to_string() },
                    tag_proj,
                    tag_lit,
                    Self::bool_ty(),
                    span,
                );

                let payload_types =
                    &variants.iter().find(|(n, _)| n == name).expect("BUG: constructor must exist").1;
                let mut bindings = Vec::new();
                for (i, sub) in sub_patterns.iter().enumerate() {
                    let payload_ty = Self::lower_type(payload_types[i].clone());
                    let proj = self.mk_tuple_proj(scrut.clone(), payload_offset + i, payload_ty, span);
                    let (sub_cond, sub_bindings) = self.compile_pattern_test(&proj, sub);
                    cond = self.and_cond(cond, sub_cond, span);
                    bindings.extend(sub_bindings);
                }
                (cond, bindings)
            }
            PatternKind::Record(_) => {
                // Checker rejects Record patterns today; reaching here
                // is a checker bug.
                unreachable!("Record patterns in match arms not yet supported by the checker")
            }
        }
    }

    fn bool_lit(&mut self, b: bool, span: Span) -> Term {
        self.mk_term(Self::bool_ty(), span, TermKind::BoolLit(b))
    }

    fn bool_ty() -> Type<TypeName> {
        Type::Constructed(TypeName::Bool, vec![])
    }

    /// AND two Bool-typed terms. Short-circuit isn't required — every
    /// projected sub-term has a well-defined zero from blank-fill, so
    /// eagerly evaluating the right side on a failed left is fine.
    /// Constant-folds the trivial cases to keep the IR tidy.
    fn and_cond(&mut self, lhs: Term, rhs: Term, span: Span) -> Term {
        if matches!(lhs.kind, TermKind::BoolLit(true)) {
            return rhs;
        }
        if matches!(rhs.kind, TermKind::BoolLit(true)) {
            return lhs;
        }
        self.build_binop(
            ast::BinaryOp { op: "&&".to_string() },
            lhs,
            rhs,
            Self::bool_ty(),
            span,
        )
    }

    fn literal_term(&mut self, lit: &PatternLiteral, expected_ty: &Type<TypeName>, span: Span) -> Term {
        match lit {
            PatternLiteral::Int(s) => {
                self.mk_term(expected_ty.clone(), span, TermKind::IntLit(s.0.clone()))
            }
            PatternLiteral::Float(v) => self.mk_term(expected_ty.clone(), span, TermKind::FloatLit(*v)),
            PatternLiteral::Bool(b) => self.mk_term(Self::bool_ty(), span, TermKind::BoolLit(*b)),
        }
    }

    /// Look up the raw (pre-lower) sum-type variants for a Constructor
    /// pattern. The type table stores the pattern's pre-lower type,
    /// which lets us recover the variant list even after the scrutinee
    /// has been rewritten to a flat tuple.
    fn lookup_sum_variants_for_pattern(
        &self,
        pat_id: NodeId,
    ) -> Option<Vec<(String, Vec<Type<TypeName>>)>> {
        let raw = self.lookup_type_raw(pat_id)?;
        match raw {
            Type::Constructed(TypeName::Sum(v), _) => Some(v),
            _ => None,
        }
    }
}

#[cfg(test)]
#[path = "match_lowering_tests.rs"]
mod match_lowering_tests;
