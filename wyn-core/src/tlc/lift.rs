//! Lambda lifting for TLC.
//!
//! Lifts all lambdas to top-level definitions. Lambdas with free variables
//! get extra parameters for their captures, and the lambda occurrence
//! becomes a partial application to the captured values.

use super::{Def, FunctionName, Program, Term, TermIdSource, TermKind};
use crate::ast::{Span, TypeName};
use polytype::Type;
use std::collections::HashSet;

/// Lambda lifter - transforms all lambdas to top-level definitions.
pub struct LambdaLifter {
    /// Top-level names (not captured)
    top_level: HashSet<String>,
    /// New definitions created for lifted lambdas
    new_defs: Vec<Def>,
    /// Counter for generating unique lambda names
    lambda_counter: u32,
    /// Counter for generating unique term IDs
    term_ids: TermIdSource,
    /// Scope stack for bound variables
    scope: Vec<String>,
}

impl LambdaLifter {
    /// Lift all lambdas in a program to top-level definitions.
    pub fn lift(program: Program) -> Program {
        let mut lifter = Self {
            top_level: program.defs.iter().map(|d| d.name.clone()).collect(),
            new_defs: vec![],
            lambda_counter: 0,
            term_ids: TermIdSource::new(),
            scope: vec![],
        };

        let transformed_defs: Vec<_> = program
            .defs
            .into_iter()
            .map(|def| {
                let body = lifter.lift_term(def.body);
                Def { body, ..def }
            })
            .collect();

        Program {
            defs: transformed_defs
                .into_iter()
                .chain(lifter.new_defs)
                .collect(),
        }
    }

    /// Check if a name is bound (in scope or top-level)
    fn is_bound(&self, name: &str) -> bool {
        self.scope.contains(&name.to_string()) || self.top_level.contains(name)
    }

    /// Push a name onto the scope stack
    fn push_scope(&mut self, name: &str) {
        self.scope.push(name.to_string());
    }

    /// Pop from the scope stack
    fn pop_scope(&mut self) {
        self.scope.pop();
    }

    /// Lift a term, transforming lambdas into references to lifted definitions.
    fn lift_term(&mut self, term: Term) -> Term {
        let ty = term.ty.clone();
        let span = term.span;

        match term.kind {
            TermKind::Lam {
                param,
                param_ty,
                body,
            } => {
                // Push param onto scope
                self.push_scope(&param);
                let lifted_body = self.lift_term(*body);
                self.pop_scope();

                // Compute free vars of this lambda (need to temporarily add param to scope)
                self.push_scope(&param);
                let free_vars = self.free_vars_of(&lifted_body);
                self.pop_scope();

                if free_vars.is_empty() {
                    // No captures: lift as-is
                    let name = self.fresh_name();
                    let new_lam = Term {
                        id: self.term_ids.next_id(),
                        ty: ty.clone(),
                        span,
                        kind: TermKind::Lam {
                            param,
                            param_ty,
                            body: Box::new(lifted_body),
                        },
                    };
                    self.new_defs.push(Def {
                        name: name.clone(),
                        ty: ty.clone(),
                        body: new_lam,
                    });
                    Term {
                        id: self.term_ids.next_id(),
                        ty,
                        span,
                        kind: TermKind::Var(name),
                    }
                } else {
                    // Has captures: wrap with extra lambdas, apply to captures
                    let name = self.fresh_name();
                    let wrapped =
                        self.wrap_with_captures(param, param_ty, lifted_body, &free_vars, span);
                    self.new_defs.push(Def {
                        name: name.clone(),
                        ty: wrapped.ty.clone(),
                        body: wrapped,
                    });
                    self.apply_to_captures(&name, &free_vars, ty, span)
                }
            }

            TermKind::App { func, arg } => {
                let lifted_func = self.lift_function_name(*func);
                let lifted_arg = self.lift_term(*arg);
                Term {
                    id: self.term_ids.next_id(),
                    ty,
                    span,
                    kind: TermKind::App {
                        func: Box::new(lifted_func),
                        arg: Box::new(lifted_arg),
                    },
                }
            }

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                let lifted_rhs = self.lift_term(*rhs);
                self.push_scope(&name);
                let lifted_body = self.lift_term(*body);
                self.pop_scope();
                Term {
                    id: self.term_ids.next_id(),
                    ty,
                    span,
                    kind: TermKind::Let {
                        name,
                        name_ty,
                        rhs: Box::new(lifted_rhs),
                        body: Box::new(lifted_body),
                    },
                }
            }

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let lifted_cond = self.lift_term(*cond);
                let lifted_then = self.lift_term(*then_branch);
                let lifted_else = self.lift_term(*else_branch);
                Term {
                    id: self.term_ids.next_id(),
                    ty,
                    span,
                    kind: TermKind::If {
                        cond: Box::new(lifted_cond),
                        then_branch: Box::new(lifted_then),
                        else_branch: Box::new(lifted_else),
                    },
                }
            }

            // Var and literals are unchanged
            TermKind::Var(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::StringLit(_) => term,
        }
    }

    /// Lift a function name (handles Term variant which contains a term)
    fn lift_function_name(&mut self, func: FunctionName) -> FunctionName {
        match func {
            FunctionName::Term(t) => FunctionName::Term(Box::new(self.lift_term(*t))),
            // Var, BinOp, UnOp are unchanged
            other => other,
        }
    }

    /// Compute free variables of a term (vars used but not bound in current scope or top-level)
    fn free_vars_of(&self, term: &Term) -> Vec<(String, Type<TypeName>)> {
        let mut free = Vec::new();
        let mut seen = HashSet::new();
        self.collect_free_vars(term, &mut free, &mut seen);
        free
    }

    fn collect_free_vars(
        &self,
        term: &Term,
        free: &mut Vec<(String, Type<TypeName>)>,
        seen: &mut HashSet<String>,
    ) {
        match &term.kind {
            TermKind::Var(name) => {
                if !self.is_bound(name) && !seen.contains(name) {
                    seen.insert(name.clone());
                    free.push((name.clone(), term.ty.clone()));
                }
            }
            TermKind::Lam { body, .. } => {
                // Don't descend into nested lambdas for free var analysis -
                // they'll be lifted separately and their free vars are their problem
                self.collect_free_vars(body, free, seen);
            }
            TermKind::App { func, arg } => {
                self.collect_free_vars_func(func, free, seen);
                self.collect_free_vars(arg, free, seen);
            }
            TermKind::Let { rhs, body, .. } => {
                self.collect_free_vars(rhs, free, seen);
                self.collect_free_vars(body, free, seen);
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.collect_free_vars(cond, free, seen);
                self.collect_free_vars(then_branch, free, seen);
                self.collect_free_vars(else_branch, free, seen);
            }
            // Literals have no free vars
            TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::StringLit(_) => {}
        }
    }

    fn collect_free_vars_func(
        &self,
        func: &FunctionName,
        free: &mut Vec<(String, Type<TypeName>)>,
        seen: &mut HashSet<String>,
    ) {
        match func {
            FunctionName::Term(t) => self.collect_free_vars(t, free, seen),
            FunctionName::Var(name) => {
                // Function name could be a free var if not bound
                if !self.is_bound(name) && !seen.contains(name) {
                    seen.insert(name.clone());
                    free.push((
                        name.clone(),
                        Type::Constructed(TypeName::Named("unknown".to_string()), vec![]),
                    ));
                }
            }
            // BinOp and UnOp don't introduce free vars
            FunctionName::BinOp(_) | FunctionName::UnOp(_) => {}
        }
    }

    /// Wrap a lambda body with extra lambdas for each captured variable.
    /// Build: |cap1| |cap2| ... |param| body
    fn wrap_with_captures(
        &mut self,
        param: String,
        param_ty: Type<TypeName>,
        body: Term,
        captures: &[(String, Type<TypeName>)],
        span: Span,
    ) -> Term {
        // Start with the innermost lambda (the original one)
        let inner_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), body.ty.clone()]);
        let inner = Term {
            id: self.term_ids.next_id(),
            ty: inner_ty,
            span,
            kind: TermKind::Lam {
                param,
                param_ty,
                body: Box::new(body),
            },
        };

        // Wrap with lambdas for each capture (in reverse order so first capture is outermost)
        captures.iter().rev().fold(inner, |acc, (cap_name, cap_ty)| {
            let result_ty = Type::Constructed(TypeName::Arrow, vec![cap_ty.clone(), acc.ty.clone()]);
            Term {
                id: self.term_ids.next_id(),
                ty: result_ty,
                span,
                kind: TermKind::Lam {
                    param: cap_name.clone(),
                    param_ty: cap_ty.clone(),
                    body: Box::new(acc),
                },
            }
        })
    }

    /// Build partial application: lambda_name cap1 cap2 ...
    fn apply_to_captures(
        &mut self,
        name: &str,
        captures: &[(String, Type<TypeName>)],
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        // Start with reference to the lifted lambda
        let base = Term {
            id: self.term_ids.next_id(),
            ty: Type::Constructed(TypeName::Named("unknown".to_string()), vec![]),
            span,
            kind: TermKind::Var(name.to_string()),
        };

        // Apply to each capture
        let result = captures.iter().fold(base, |acc, (cap_name, cap_ty)| {
            let arg = Term {
                id: self.term_ids.next_id(),
                ty: cap_ty.clone(),
                span,
                kind: TermKind::Var(cap_name.clone()),
            };
            Term {
                id: self.term_ids.next_id(),
                ty: Type::Constructed(TypeName::Named("unknown".to_string()), vec![]),
                span,
                kind: TermKind::App {
                    func: Box::new(FunctionName::Term(Box::new(acc))),
                    arg: Box::new(arg),
                },
            }
        });

        // Fix up the result type
        Term {
            ty: result_ty,
            ..result
        }
    }

    fn fresh_name(&mut self) -> String {
        let name = format!("_lambda_{}", self.lambda_counter);
        self.lambda_counter += 1;
        name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_span() -> Span {
        Span {
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
        }
    }

    #[test]
    fn test_lift_simple_lambda() {
        // Test: def f = |x| x
        // Should become: def f = _lambda_0, def _lambda_0 = |x| x
        let mut ids = TermIdSource::new();

        let lam = Term {
            id: ids.next_id(),
            ty: Type::Constructed(
                TypeName::Arrow,
                vec![
                    Type::Constructed(TypeName::Int(32), vec![]),
                    Type::Constructed(TypeName::Int(32), vec![]),
                ],
            ),
            span: dummy_span(),
            kind: TermKind::Lam {
                param: "x".to_string(),
                param_ty: Type::Constructed(TypeName::Int(32), vec![]),
                body: Box::new(Term {
                    id: ids.next_id(),
                    ty: Type::Constructed(TypeName::Int(32), vec![]),
                    span: dummy_span(),
                    kind: TermKind::Var("x".to_string()),
                }),
            },
        };

        let program = Program {
            defs: vec![Def {
                name: "f".to_string(),
                ty: lam.ty.clone(),
                body: lam,
            }],
        };

        let lifted = LambdaLifter::lift(program);

        // Should have 2 defs: f and _lambda_0
        assert_eq!(lifted.defs.len(), 2);
        assert_eq!(lifted.defs[0].name, "f");
        assert_eq!(lifted.defs[1].name, "_lambda_0");

        // f's body should be a Var reference to _lambda_0
        match &lifted.defs[0].body.kind {
            TermKind::Var(name) => assert_eq!(name, "_lambda_0"),
            _ => panic!("Expected Var"),
        }
    }
}
