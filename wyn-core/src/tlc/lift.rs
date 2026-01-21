//! Lambda lifting for TLC.
//!
//! Lifts all lambdas to top-level definitions. Lambdas with free variables
//! get extra parameters for their captures, and the lambda occurrence
//! becomes a partial application to the captured values.

use super::{Def, DefMeta, FunctionName, Program, Term, TermIdSource, TermKind};
use crate::ast::{Span, TypeName};
use polytype::Type;
use std::collections::HashSet;

// =============================================================================
// Standalone Free Variable Analysis
// =============================================================================

/// Compute free variables of a term, given explicit sets of bound names.
/// This is independent of the lifter's scope state.
fn compute_free_vars(
    term: &Term,
    bound: &HashSet<String>,
    top_level: &HashSet<String>,
    builtins: &HashSet<String>,
) -> Vec<(String, Type<TypeName>)> {
    let mut free = Vec::new();
    let mut seen = HashSet::new();
    collect_free_vars_standalone(term, bound, top_level, builtins, &mut free, &mut seen);
    free
}

/// Recursively collect free variables, using immutable bound set passing.
fn collect_free_vars_standalone(
    term: &Term,
    bound: &HashSet<String>,
    top_level: &HashSet<String>,
    builtins: &HashSet<String>,
    free: &mut Vec<(String, Type<TypeName>)>,
    seen: &mut HashSet<String>,
) {
    match &term.kind {
        TermKind::Var(name) => {
            if !bound.contains(name)
                && !top_level.contains(name)
                && !builtins.contains(name)
                && !name.starts_with("_w_")
                && !seen.contains(name)
            {
                seen.insert(name.clone());
                free.push((name.clone(), term.ty.clone()));
            }
        }
        TermKind::Let { name, rhs, body, .. } => {
            collect_free_vars_standalone(rhs, bound, top_level, builtins, free, seen);
            let mut inner_bound = bound.clone();
            inner_bound.insert(name.clone());
            collect_free_vars_standalone(body, &inner_bound, top_level, builtins, free, seen);
        }
        TermKind::Lam { param, body, .. } => {
            let mut inner_bound = bound.clone();
            inner_bound.insert(param.clone());
            collect_free_vars_standalone(body, &inner_bound, top_level, builtins, free, seen);
        }
        TermKind::App { func, arg } => {
            collect_free_vars_func_standalone(func, bound, top_level, builtins, free, seen);
            collect_free_vars_standalone(arg, bound, top_level, builtins, free, seen);
        }
        TermKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_free_vars_standalone(cond, bound, top_level, builtins, free, seen);
            collect_free_vars_standalone(then_branch, bound, top_level, builtins, free, seen);
            collect_free_vars_standalone(else_branch, bound, top_level, builtins, free, seen);
        }
        TermKind::IntLit(_) | TermKind::FloatLit(_) | TermKind::BoolLit(_) | TermKind::StringLit(_) => {}
    }
}

/// Collect free variables from a FunctionName.
fn collect_free_vars_func_standalone(
    func: &FunctionName,
    bound: &HashSet<String>,
    top_level: &HashSet<String>,
    builtins: &HashSet<String>,
    free: &mut Vec<(String, Type<TypeName>)>,
    seen: &mut HashSet<String>,
) {
    match func {
        FunctionName::Term(t) => collect_free_vars_standalone(t, bound, top_level, builtins, free, seen),
        FunctionName::Var(_) | FunctionName::BinOp(_) | FunctionName::UnOp(_) => {}
    }
}

// =============================================================================
// Lambda Lifter
// =============================================================================

/// Lambda lifter - transforms all lambdas to top-level definitions.
pub struct LambdaLifter<'a> {
    /// Top-level names (not captured)
    top_level: HashSet<String>,
    /// Built-in names that should not be captured (intrinsics, prelude functions)
    builtins: &'a HashSet<String>,
    /// New definitions created for lifted lambdas
    new_defs: Vec<Def>,
    /// Counter for generating unique lambda names
    lambda_counter: u32,
    /// Counter for generating unique term IDs
    term_ids: TermIdSource,
    /// Scope stack for bound variables with their types
    scope: Vec<(String, Type<TypeName>)>,
}

impl<'a> LambdaLifter<'a> {
    /// Lift all lambdas in a program to top-level definitions.
    pub fn lift(program: Program, builtins: &'a HashSet<String>) -> Program {
        let mut lifter = Self {
            top_level: program.defs.iter().map(|d| d.name.clone()).collect(),
            builtins,
            new_defs: vec![],
            lambda_counter: 0,
            term_ids: TermIdSource::new(),
            scope: vec![],
        };

        let transformed_defs: Vec<_> = program
            .defs
            .into_iter()
            .map(|def| {
                // For all defs (entry points and functions), preserve parameter lambdas
                // but lift nested lambdas inside the function body
                let body = lifter.lift_preserving_params(def.body);
                Def { body, ..def }
            })
            .collect();

        Program {
            defs: transformed_defs.into_iter().chain(lifter.new_defs).collect(),
            uniforms: program.uniforms,
            storage: program.storage,
        }
    }

    /// Lift lambdas but preserve the outermost parameter lambdas (for entry points).
    fn lift_preserving_params(&mut self, term: Term) -> Term {
        match term.kind {
            TermKind::Lam {
                param,
                param_ty,
                body,
            } => {
                // Keep this lambda, but recursively process its body
                self.push_scope(&param, param_ty.clone());
                let lifted_body = self.lift_preserving_params(*body);
                self.pop_scope();
                Term {
                    id: self.term_ids.next_id(),
                    ty: term.ty,
                    span: term.span,
                    kind: TermKind::Lam {
                        param,
                        param_ty,
                        body: Box::new(lifted_body),
                    },
                }
            }
            // Once we hit a non-lambda, lift normally
            _ => self.lift_term(term),
        }
    }

    /// Check if a name is bound (in scope, top-level, or a builtin)
    fn is_bound(&self, name: &str) -> bool {
        self.scope.iter().any(|(n, _)| n == name)
            || self.top_level.contains(name)
            || self.builtins.contains(name)
            || name.starts_with("_w_") // Internal intrinsics are always bound
    }

    /// Push a name and type onto the scope stack
    fn push_scope(&mut self, name: &str, ty: Type<TypeName>) {
        self.scope.push((name.to_string(), ty));
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
            TermKind::Lam { .. } => {
                // Extract ALL nested lambda params together (to keep multi-param functions intact)
                let (params, inner_body) = self.extract_lambda_params(term);

                // Push all params onto scope
                for (p, pty) in &params {
                    self.push_scope(p, pty.clone());
                }

                // Lift the innermost body (which is not a lambda)
                let lifted_body = self.lift_term(inner_body);

                // Pop all params
                for _ in &params {
                    self.pop_scope();
                }

                // Compute free vars with only lambda params as bound (standalone analysis)
                let bound: HashSet<String> = params.iter().map(|(p, _)| p.clone()).collect();
                let free_vars = compute_free_vars(&lifted_body, &bound, &self.top_level, self.builtins);

                // Rebuild nested lambdas from inside out
                let rebuilt_lam = self.rebuild_nested_lam(&params, lifted_body, span);

                if free_vars.is_empty() {
                    // No captures: lift as-is
                    let name = self.fresh_name();
                    self.new_defs.push(Def {
                        name: name.clone(),
                        ty: rebuilt_lam.ty.clone(),
                        body: rebuilt_lam,
                        meta: DefMeta::Function,
                        arity: params.len(),
                    });
                    Term {
                        id: self.term_ids.next_id(),
                        ty,
                        span,
                        kind: TermKind::Var(name),
                    }
                } else {
                    // Has captures: wrap with extra lambdas for captures, apply to captures
                    let name = self.fresh_name();
                    let wrapped = self.wrap_lam_with_captures(rebuilt_lam, &free_vars, span);
                    let lifted_ty = wrapped.ty.clone();
                    self.new_defs.push(Def {
                        name: name.clone(),
                        ty: lifted_ty.clone(),
                        body: wrapped,
                        meta: DefMeta::Function,
                        arity: free_vars.len() + params.len(),
                    });
                    self.apply_to_captures(&name, lifted_ty, &free_vars, ty, span)
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
                self.push_scope(&name, name_ty.clone());
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

    /// Extract all nested lambda parameters from a term.
    /// Returns (params, inner_body) where params is [(name, type), ...] outermost first.
    fn extract_lambda_params(&self, term: Term) -> (Vec<(String, Type<TypeName>)>, Term) {
        let mut params = Vec::new();
        let mut current = term;

        while let TermKind::Lam { param, param_ty, body } = current.kind {
            params.push((param, param_ty));
            current = *body;
        }

        (params, current)
    }

    /// Rebuild nested lambdas from a list of params and a body.
    fn rebuild_nested_lam(
        &mut self,
        params: &[(String, Type<TypeName>)],
        body: Term,
        span: Span,
    ) -> Term {
        // Build from inside out
        params.iter().rev().fold(body, |acc, (param, param_ty)| {
            let lam_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), acc.ty.clone()]);
            Term {
                id: self.term_ids.next_id(),
                ty: lam_ty,
                span,
                kind: TermKind::Lam {
                    param: param.clone(),
                    param_ty: param_ty.clone(),
                    body: Box::new(acc),
                },
            }
        })
    }

    /// Wrap a lambda (possibly nested) with extra lambdas for captures.
    fn wrap_lam_with_captures(
        &mut self,
        lam: Term,
        captures: &[(String, Type<TypeName>)],
        span: Span,
    ) -> Term {
        // Wrap with lambdas for each capture (in reverse order so first capture is outermost)
        captures.iter().rev().fold(lam, |acc, (cap_name, cap_ty)| {
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
        lifted_ty: Type<TypeName>,
        captures: &[(String, Type<TypeName>)],
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        // Start with reference to the lifted lambda
        let base = Term {
            id: self.term_ids.next_id(),
            ty: lifted_ty.clone(),
            span,
            kind: TermKind::Var(name.to_string()),
        };

        // Apply to each capture, threading the type through
        let (result, _) = captures.iter().fold((base, lifted_ty), |(acc, func_ty), (cap_name, cap_ty)| {
            let arg = Term {
                id: self.term_ids.next_id(),
                ty: cap_ty.clone(),
                span,
                kind: TermKind::Var(cap_name.clone()),
            };
            // func_ty should be Arrow(cap_ty, result_ty) - extract the result type
            let app_result_ty = match &func_ty {
                Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => args[1].clone(),
                _ => panic!(
                    "BUG: Expected arrow type for capture application, got {:?}",
                    func_ty
                ),
            };
            let app = Term {
                id: self.term_ids.next_id(),
                ty: app_result_ty.clone(),
                span,
                kind: TermKind::App {
                    func: Box::new(FunctionName::Term(Box::new(acc))),
                    arg: Box::new(arg),
                },
            };
            (app, app_result_ty)
        });

        // The final type should match result_ty (the original lambda type)
        debug_assert!(
            result.ty == result_ty,
            "BUG: Final type mismatch: {:?} vs {:?}",
            result.ty,
            result_ty
        );
        result
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
    fn test_lift_preserves_param_lambda() {
        // Test: def f = |x| x
        // Parameter lambdas are preserved, so f stays as |x| x (not lifted)
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
                meta: DefMeta::Function,
                arity: 1,
            }],
            uniforms: vec![],
            storage: vec![],
        };

        let builtins = HashSet::new();
        let lifted = LambdaLifter::lift(program, &builtins);

        // Should have 1 def: f with the lambda preserved
        assert_eq!(lifted.defs.len(), 1);
        assert_eq!(lifted.defs[0].name, "f");

        // f's body should still be a Lam
        match &lifted.defs[0].body.kind {
            TermKind::Lam { param, .. } => assert_eq!(param, "x"),
            _ => panic!("Expected Lam"),
        }
    }
}
