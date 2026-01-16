//! Lambda lifting for TLC.
//!
//! Lifts all lambdas to top-level definitions. Lambdas with free variables
//! get extra parameters for their captures, and the lambda occurrence
//! becomes a partial application to the captured values.

use super::{Def, DefMeta, FunctionName, Program, Term, TermIdSource, TermKind};
use crate::ast::{Span, TypeName};
use crate::scope::ScopeStack;
use polytype::Type;
use std::collections::HashSet;

/// Analyzes a lambda body to find free variables that need to be captured.
///
/// This is a separate struct because free variable analysis doesn't need access
/// to the lifting state - it only needs to know which names are globals (builtins
/// + top-level defs) that should not be captured.
struct FreeVarAnalyzer<'a> {
    globals: &'a HashSet<String>,
}

impl<'a> FreeVarAnalyzer<'a> {
    fn new(globals: &'a HashSet<String>) -> Self {
        Self { globals }
    }

    /// Analyze a lambda and return its free variables with types.
    fn analyze(&self, params: &[(String, Type<TypeName>)], body: &Term) -> Vec<(String, Type<TypeName>)> {
        let mut local_bindings: HashSet<String> = params.iter().map(|(n, _)| n.clone()).collect();
        let mut free_vars = Vec::new();
        let mut seen = HashSet::new();
        self.collect_free_vars(body, &mut local_bindings, &mut free_vars, &mut seen);
        free_vars
    }

    /// Check if a name is free (not in local bindings and not a global).
    fn is_free(&self, name: &str, local_bindings: &HashSet<String>) -> bool {
        !local_bindings.contains(name) && !self.globals.contains(name) && !name.starts_with("_w_") // Internal intrinsics are always bound
    }

    fn collect_free_vars(
        &self,
        term: &Term,
        local_bindings: &mut HashSet<String>,
        free_vars: &mut Vec<(String, Type<TypeName>)>,
        seen: &mut HashSet<String>,
    ) {
        match &term.kind {
            TermKind::Var(name) => {
                if self.is_free(name, local_bindings) && !seen.contains(name) {
                    seen.insert(name.clone());
                    free_vars.push((name.clone(), term.ty.clone()));
                }
            }
            TermKind::Lam { param, body, .. } => {
                // Mark lambda param as locally bound when checking body
                local_bindings.insert(param.clone());
                self.collect_free_vars(body, local_bindings, free_vars, seen);
                local_bindings.remove(param);
            }
            TermKind::App { func, arg } => {
                self.collect_free_vars_func(func, local_bindings, free_vars, seen);
                self.collect_free_vars(arg, local_bindings, free_vars, seen);
            }
            TermKind::Let { name, rhs, body, .. } => {
                // Check rhs first (name is not yet bound)
                self.collect_free_vars(rhs, local_bindings, free_vars, seen);
                // Then check body with name bound
                local_bindings.insert(name.clone());
                self.collect_free_vars(body, local_bindings, free_vars, seen);
                local_bindings.remove(name);
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.collect_free_vars(cond, local_bindings, free_vars, seen);
                self.collect_free_vars(then_branch, local_bindings, free_vars, seen);
                self.collect_free_vars(else_branch, local_bindings, free_vars, seen);
            }
            // Literals have no free vars
            TermKind::IntLit(_) | TermKind::FloatLit(_) | TermKind::BoolLit(_) | TermKind::StringLit(_) => {
            }
        }
    }

    fn collect_free_vars_func(
        &self,
        func: &FunctionName,
        local_bindings: &HashSet<String>,
        free_vars: &mut Vec<(String, Type<TypeName>)>,
        seen: &mut HashSet<String>,
    ) {
        match func {
            FunctionName::Term(t) => {
                self.collect_free_vars(t, &mut local_bindings.clone(), free_vars, seen)
            }
            FunctionName::Var(name) => {
                // FunctionName::Var is used for top-level functions,
                // which should always be bound. Local variables in function position
                // use FunctionName::Term(Term::Var(...)) instead.
                if self.is_free(name, local_bindings) && !seen.contains(name) {
                    panic!(
                        "BUG: Unexpected free FunctionName::Var '{}'. \
                         Local function variables should use FunctionName::Term.",
                        name
                    );
                }
            }
            // Intrinsics, BinOp, and UnOp don't introduce free vars
            FunctionName::Intrinsic(_) | FunctionName::BinOp(_) | FunctionName::UnOp(_) => {}
        }
    }
}

/// Lambda lifter - transforms all lambdas to top-level definitions.
pub struct LambdaLifter {
    /// Global names (builtins + top-level) - not captured
    globals: HashSet<String>,
    /// Scope stack: level 0 = globals, level 1+ = local bindings
    scope: ScopeStack<Type<TypeName>>,
    /// New definitions created for lifted lambdas
    new_defs: Vec<Def>,
    /// Counter for generating unique lambda names
    lambda_counter: u32,
    /// Counter for generating unique term IDs
    term_ids: TermIdSource,
}

impl LambdaLifter {
    /// Lift all lambdas in a program to top-level definitions.
    pub fn lift(program: Program, builtins: &HashSet<String>) -> Program {
        // Build globals set: builtins + top-level def names
        let mut globals: HashSet<String> = builtins.clone();
        for def in &program.defs {
            globals.insert(def.name.clone());
        }

        // Build scope with globals at level 0
        let mut scope = ScopeStack::new();
        let placeholder_ty = Type::Constructed(TypeName::Unit, vec![]);
        for name in &globals {
            scope.insert(name.clone(), placeholder_ty.clone());
        }

        let mut lifter = Self {
            globals,
            scope,
            new_defs: vec![],
            lambda_counter: 0,
            term_ids: TermIdSource::new(),
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
                self.scope.push_scope();
                self.scope.insert(param.clone(), param_ty.clone());
                let lifted_body = self.lift_preserving_params(*body);
                let popped = self.scope.pop_scope();
                assert!(popped.is_some(), "BUG: attempted to pop global scope");
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

    /// Lift a term, transforming lambdas into references to lifted definitions.
    fn lift_term(&mut self, term: Term) -> Term {
        let ty = term.ty.clone();
        let span = term.span;

        match term.kind {
            TermKind::Lam { .. } => {
                // Extract ALL nested lambda params together (to keep multi-param functions intact)
                let (params, inner_body) = self.extract_lambda_params(term);

                // Push all params onto scope for lifting the body
                self.scope.push_scope();
                for (p, pty) in &params {
                    self.scope.insert(p.clone(), pty.clone());
                }

                // Lift the innermost body (which is not a lambda)
                let lifted_body = self.lift_term(inner_body);

                // Pop the params scope
                let popped = self.scope.pop_scope();
                assert!(popped.is_some(), "BUG: attempted to pop global scope");

                // Compute free vars using FreeVarAnalyzer (no scope manipulation needed)
                let analyzer = FreeVarAnalyzer::new(&self.globals);
                let free_vars = analyzer.analyze(&params, &lifted_body);

                // Filter out unit-type captures - these don't need to be passed as parameters
                // (they carry no runtime information)
                let free_vars: Vec<_> =
                    free_vars.into_iter().filter(|(_, ty)| !Self::is_unit_type(ty)).collect();

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
                self.scope.push_scope();
                self.scope.insert(name.clone(), name_ty.clone());
                let lifted_body = self.lift_term(*body);
                let popped = self.scope.pop_scope();
                assert!(popped.is_some(), "BUG: attempted to pop global scope");
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

    /// Check if a type is the unit type `()`.
    fn is_unit_type(ty: &Type<TypeName>) -> bool {
        matches!(ty, Type::Constructed(TypeName::Unit, _))
    }

    /// Extract all nested lambda parameters from a term.
    /// Returns (params, inner_body) where params is [(name, type), ...] outermost first.
    fn extract_lambda_params(&self, term: Term) -> (Vec<(String, Type<TypeName>)>, Term) {
        let mut params = Vec::new();
        let mut current = term;

        while let TermKind::Lam {
            param,
            param_ty,
            body,
        } = current.kind
        {
            params.push((param, param_ty));
            current = *body;
        }

        (params, current)
    }

    /// Rebuild nested lambdas from a list of params and a body.
    fn rebuild_nested_lam(&mut self, params: &[(String, Type<TypeName>)], body: Term, span: Span) -> Term {
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
