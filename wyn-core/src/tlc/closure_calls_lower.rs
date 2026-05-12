//! Closure-call lowering pass (phase 3 of the closure pipeline).
//!
//! Threads captures into call sites: rewrites every
//! `App(Var(sym), args)` whose `sym` resolves through the
//! `ClosureInfo` side-table to a `Closure { code, captures }` into
//! `App(Var(code), args ++ captures)`. Direct callable values (no
//! captures) and non-callable Vars pass through unchanged.
//!
//! After this pass, every call site is "direct" — the function
//! position of every `App` resolves to a `Var`, captures have been
//! fully threaded into trailing args. The accompanying verifier
//! `verify_closure_calls_lowered` enforces this end-state invariant
//! and is strictly stronger than `assert_flat_apps` (which only
//! forbids nested `App` in func position): it additionally rejects
//! every non-`Var` func.

use super::VarRef;
use super::closure_convert::{CallableValue, ClosureInfo};
use super::{ArrayExpr, Lambda, LoopKind, Program, SoacOp, Term, TermIdSource, TermKind};
use crate::{SymbolId, SymbolTable};
use std::collections::HashMap;

#[derive(Debug)]
pub enum ClosureCallsLowerError {
    /// An `App` node has a function position that isn't a `Var(SymbolId)` —
    /// e.g. a nested App, a Lambda, or some constructed value. After
    /// closure-call lowering, every call should resolve statically.
    IndirectCall {
        def: SymbolId,
        func_kind: &'static str,
    },
    /// An `App { Var(target), args }` has the wrong arg count for its
    /// target. After defunctionalize + monomorphize + buffer_specialize,
    /// every direct call must be fully applied.
    ArityMismatch {
        def: SymbolId,
        target: SymbolId,
        expected: usize,
        actual: usize,
    },
}

pub fn verify_closure_calls_lowered(program: &Program) -> Result<(), ClosureCallsLowerError> {
    // Build an arity map for direct-call targets. Intrinsics not in
    // `program.defs` fall back to `builtins::intrinsic_arity` (the
    // catalog-derived arity for each builtin). Targets the catalog
    // doesn't know about are skipped — those are operator dispatch
    // helpers whose arity is enforced by the backend.
    let arities: HashMap<SymbolId, usize> = program.defs.iter().map(|d| (d.name, d.arity)).collect();
    for def in &program.defs {
        walk(&def.body, def.name, &arities, &program.symbols)?;
    }
    Ok(())
}

fn walk(
    term: &Term,
    def: SymbolId,
    arities: &HashMap<SymbolId, usize>,
    symbols: &SymbolTable,
) -> Result<(), ClosureCallsLowerError> {
    if let TermKind::App { func, args } = &term.kind {
        if !is_static_func(&func.kind) {
            return Err(ClosureCallsLowerError::IndirectCall {
                def,
                func_kind: discriminant_name(&func.kind),
            });
        }
        if let TermKind::Var(VarRef::Symbol(target)) = &func.kind {
            let expected = arities
                .get(target)
                .copied()
                .or_else(|| symbols.get(*target).and_then(|name| crate::builtins::intrinsic_arity(name)));
            if let Some(expected) = expected {
                if expected != args.len() {
                    return Err(ClosureCallsLowerError::ArityMismatch {
                        def,
                        target: *target,
                        expected,
                        actual: args.len(),
                    });
                }
            }
        }
    }
    let mut result = Ok(());
    term.for_each_child(&mut |child| {
        if result.is_ok() {
            result = walk(child, def, arities, symbols);
        }
    });
    result
}

/// A "static func" position is anything that resolves to a known
/// callable at compile time: a `Var` (top-level def or local
/// parameter), or an operator value (`BinOp`/`UnOp`) that the backend
/// dispatches directly via a fixed PrimOp.
fn is_static_func(kind: &TermKind) -> bool {
    matches!(
        kind,
        TermKind::Var(_) | TermKind::BinOp(_) | TermKind::UnOp(_) | TermKind::Extern(_)
    )
}

fn discriminant_name(kind: &TermKind) -> &'static str {
    match kind {
        TermKind::Var(_) => "Var",
        TermKind::BinOp(_) => "BinOp",
        TermKind::UnOp(_) => "UnOp",
        TermKind::Lambda(_) => "Lambda",
        TermKind::App { .. } => "App",
        TermKind::Let { .. } => "Let",
        TermKind::IntLit(_) => "IntLit",
        TermKind::FloatLit(_) => "FloatLit",
        TermKind::BoolLit(_) => "BoolLit",
        TermKind::Extern(_) => "Extern",
        TermKind::If { .. } => "If",
        TermKind::Loop { .. } => "Loop",
        TermKind::Soac(_) => "Soac",
        TermKind::ArrayExpr(_) => "ArrayExpr",
        TermKind::Tuple(_) => "Tuple",
        TermKind::TupleProj { .. } => "TupleProj",
        TermKind::Index { .. } => "Index",
        TermKind::VecLit(_) => "VecLit",
        TermKind::UnitLit => "UnitLit",
        TermKind::Coerce { .. } => "Coerce",
    }
}

// =============================================================================
// Closure-call lowering pass
// =============================================================================

struct CallLowerer<'a> {
    closure_info: &'a ClosureInfo,
    term_ids: TermIdSource,
}

impl<'a> CallLowerer<'a> {
    fn new(closure_info: &'a ClosureInfo) -> Self {
        Self {
            closure_info,
            term_ids: TermIdSource::new(),
        }
    }

    /// Walk a def body, preserving the outer parameter-spine `Lambda`
    /// nodes (those carry the def's named parameters). Anything below
    /// the spine routes through `lower_term`.
    fn lower_def_body(&mut self, term: Term) -> Term {
        match term.kind {
            TermKind::Lambda(Lambda { params, body, ret_ty }) => {
                let new_body = self.lower_def_body(*body);
                Term {
                    id: self.term_ids.next_id(),
                    ty: term.ty,
                    span: term.span,
                    kind: TermKind::Lambda(Lambda {
                        params,
                        body: Box::new(new_body),
                        ret_ty,
                    }),
                }
            }
            _ => self.lower_term(term),
        }
    }

    fn lower_term(&mut self, term: Term) -> Term {
        let ty = term.ty.clone();
        let span = term.span;
        match term.kind {
            TermKind::App { func, args } => {
                let new_func = self.lower_term(*func);
                let mut new_args: Vec<Term> = args.into_iter().map(|a| self.lower_term(a)).collect();

                if let TermKind::Var(VarRef::Symbol(sym)) = &new_func.kind {
                    if let Some(CallableValue::Closure {
                        code,
                        captures,
                        param_count,
                    }) = self.closure_info.resolve_callable(*sym)
                    {
                        // Idempotency: only thread captures when the
                        // App's args.len() matches the lifted def's
                        // user-facing param count. Calls inside
                        // specialized HOF bodies have already been
                        // pre-threaded by `hof_specialize` (so
                        // args.len() == param_count + captures.len())
                        // — re-threading would double the captures.
                        if !captures.is_empty() && new_args.len() == *param_count {
                            new_args.extend(captures.iter().cloned());
                            let func_term = Term {
                                id: self.term_ids.next_id(),
                                ty: new_func.ty.clone(),
                                span: new_func.span,
                                kind: TermKind::Var(VarRef::Symbol(*code)),
                            };
                            return Term {
                                id: self.term_ids.next_id(),
                                ty,
                                span,
                                kind: TermKind::App {
                                    func: Box::new(func_term),
                                    args: new_args,
                                },
                            };
                        }
                    }
                }

                Term {
                    id: self.term_ids.next_id(),
                    ty,
                    span,
                    kind: TermKind::App {
                        func: Box::new(new_func),
                        args: new_args,
                    },
                }
            }

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::Let {
                    name,
                    name_ty,
                    rhs: Box::new(self.lower_term(*rhs)),
                    body: Box::new(self.lower_term(*body)),
                },
            },

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::If {
                    cond: Box::new(self.lower_term(*cond)),
                    then_branch: Box::new(self.lower_term(*then_branch)),
                    else_branch: Box::new(self.lower_term(*else_branch)),
                },
            },

            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => {
                let init = self.lower_term(*init);
                let init_bindings: Vec<_> =
                    init_bindings.into_iter().map(|(n, ty, e)| (n, ty, self.lower_term(e))).collect();
                let kind = match kind {
                    LoopKind::For { var, var_ty, iter } => LoopKind::For {
                        var,
                        var_ty,
                        iter: Box::new(self.lower_term(*iter)),
                    },
                    LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                        var,
                        var_ty,
                        bound: Box::new(self.lower_term(*bound)),
                    },
                    LoopKind::While { cond } => LoopKind::While {
                        cond: Box::new(self.lower_term(*cond)),
                    },
                };
                Term {
                    id: self.term_ids.next_id(),
                    ty,
                    span,
                    kind: TermKind::Loop {
                        loop_var,
                        loop_var_ty,
                        init: Box::new(init),
                        init_bindings,
                        kind,
                        body: Box::new(self.lower_term(*body)),
                    },
                }
            }

            TermKind::Soac(soac) => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::Soac(self.lower_soac(soac)),
            },

            TermKind::ArrayExpr(ae) => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::ArrayExpr(self.lower_array_expr(ae)),
            },

            TermKind::Lambda(_)
            | TermKind::Var(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::UnitLit
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Extern(_) => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: term.kind,
            },

            TermKind::Coerce { inner, target_ty } => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::Coerce {
                    inner: Box::new(self.lower_term(*inner)),
                    target_ty,
                },
            },

            TermKind::Tuple(parts) => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::Tuple(parts.into_iter().map(|p| self.lower_term(p)).collect()),
            },
            TermKind::TupleProj { tuple, idx } => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::TupleProj {
                    tuple: Box::new(self.lower_term(*tuple)),
                    idx,
                },
            },
            TermKind::Index { array, index } => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::Index {
                    array: Box::new(self.lower_term(*array)),
                    index: Box::new(self.lower_term(*index)),
                },
            },
            TermKind::VecLit(parts) => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::VecLit(parts.into_iter().map(|p| self.lower_term(p)).collect()),
            },
        }
    }

    fn lower_soac(&mut self, soac: SoacOp) -> SoacOp {
        // SOAC envelope bodies are already lifted; only the captures
        // and inputs may contain calls that need lowering.
        match soac {
            SoacOp::Map {
                lam,
                inputs,
                consumes_input,
            } => SoacOp::Map {
                lam: super::SoacBody {
                    lam: lam.lam,
                    captures: lam
                        .captures
                        .into_iter()
                        .map(|(s, ty, t)| (s, ty, self.lower_term(t)))
                        .collect(),
                },
                inputs: inputs.into_iter().map(|ae| self.lower_array_expr(ae)).collect(),
                consumes_input,
            },
            SoacOp::Reduce { op, ne, input } => SoacOp::Reduce {
                op: super::SoacBody {
                    lam: op.lam,
                    captures: op
                        .captures
                        .into_iter()
                        .map(|(s, ty, t)| (s, ty, self.lower_term(t)))
                        .collect(),
                },
                ne: Box::new(self.lower_term(*ne)),
                input: self.lower_array_expr(input),
            },
            SoacOp::Scan { op, ne, input } => SoacOp::Scan {
                op: super::SoacBody {
                    lam: op.lam,
                    captures: op
                        .captures
                        .into_iter()
                        .map(|(s, ty, t)| (s, ty, self.lower_term(t)))
                        .collect(),
                },
                ne: Box::new(self.lower_term(*ne)),
                input: self.lower_array_expr(input),
            },
            SoacOp::Filter { pred, input } => SoacOp::Filter {
                pred: super::SoacBody {
                    lam: pred.lam,
                    captures: pred
                        .captures
                        .into_iter()
                        .map(|(s, ty, t)| (s, ty, self.lower_term(t)))
                        .collect(),
                },
                input: self.lower_array_expr(input),
            },
            SoacOp::Scatter {
                dest,
                indices,
                values,
            } => SoacOp::Scatter {
                dest,
                indices: self.lower_array_expr(indices),
                values: self.lower_array_expr(values),
            },
            SoacOp::ReduceByIndex {
                dest,
                op,
                ne,
                indices,
                values,
            } => SoacOp::ReduceByIndex {
                dest,
                op: super::SoacBody {
                    lam: op.lam,
                    captures: op
                        .captures
                        .into_iter()
                        .map(|(s, ty, t)| (s, ty, self.lower_term(t)))
                        .collect(),
                },
                ne: Box::new(self.lower_term(*ne)),
                indices: self.lower_array_expr(indices),
                values: self.lower_array_expr(values),
            },
            SoacOp::Redomap {
                op,
                reduce_op,
                ne,
                inputs,
            } => SoacOp::Redomap {
                op: super::SoacBody {
                    lam: op.lam,
                    captures: op
                        .captures
                        .into_iter()
                        .map(|(s, ty, t)| (s, ty, self.lower_term(t)))
                        .collect(),
                },
                reduce_op: super::SoacBody {
                    lam: reduce_op.lam,
                    captures: reduce_op
                        .captures
                        .into_iter()
                        .map(|(s, ty, t)| (s, ty, self.lower_term(t)))
                        .collect(),
                },
                ne: Box::new(self.lower_term(*ne)),
                inputs: inputs.into_iter().map(|ae| self.lower_array_expr(ae)).collect(),
            },
        }
    }

    fn lower_array_expr(&mut self, ae: ArrayExpr) -> ArrayExpr {
        match ae {
            ArrayExpr::Ref(t) => ArrayExpr::Ref(Box::new(self.lower_term(*t))),
            ArrayExpr::Zip(exprs) => {
                ArrayExpr::Zip(exprs.into_iter().map(|e| self.lower_array_expr(e)).collect())
            }
            ArrayExpr::Soac(op) => ArrayExpr::Soac(Box::new(self.lower_soac(*op))),
            ArrayExpr::Literal(terms) => {
                ArrayExpr::Literal(terms.into_iter().map(|t| self.lower_term(t)).collect())
            }
            ArrayExpr::Range { start, len, step } => ArrayExpr::Range {
                start: Box::new(self.lower_term(*start)),
                len: Box::new(self.lower_term(*len)),
                step: step.map(|s| Box::new(self.lower_term(*s))),
            },
            ArrayExpr::StorageBuffer { .. } => {
                unreachable!("StorageBuffer introduced after defunctionalization")
            }
        }
    }
}

/// Thread captures into every call site inside a single term. Used by
/// `hof_specialize` on cloned HOF bodies so the substituted callable
/// references their captures *before* the surrounding renaming step
/// rewrites outer-scope symbols to fresh per-specialization params.
/// Idempotent — re-running on an already-threaded term is a no-op.
pub fn thread_captures_in_term(
    term: Term,
    closure_info: &ClosureInfo,
    term_ids: &mut TermIdSource,
) -> Term {
    let prev = std::mem::replace(term_ids, TermIdSource::new());
    let mut lowerer = CallLowerer {
        closure_info,
        term_ids: prev,
    };
    let result = lowerer.lower_term(term);
    *term_ids = lowerer.term_ids;
    result
}

/// Run closure-call lowering. Threads captures into call sites and
/// runs the post-condition verifier.
pub fn run(program: Program, closure_info: &ClosureInfo) -> Program {
    let mut lowerer = CallLowerer::new(closure_info);
    let defs: Vec<_> = program
        .defs
        .into_iter()
        .map(|def| super::Def {
            body: lowerer.lower_def_body(def.body),
            ..def
        })
        .collect();

    let result = Program {
        defs,
        uniforms: program.uniforms,
        storage: program.storage,
        symbols: program.symbols,
        def_syms: program.def_syms,
    };

    verify_closure_calls_lowered(&result)
        .unwrap_or_else(|e| panic!("closure-calls-lowered verifier failed: {:?}", e));
    result
}

#[cfg(test)]
#[path = "closure_calls_lower_tests.rs"]
mod closure_calls_lower_tests;
