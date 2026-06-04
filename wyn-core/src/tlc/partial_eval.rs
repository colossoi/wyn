//! Stack-based partial evaluator for TLC.
//!
//! Simpler than NBE-style: collect application spines, evaluate args,
//! apply when we have enough arguments (using arity metadata).

use super::VarRef;
use super::{Def, Lambda, Program, Term, TermIdSource, TermKind};
use crate::ast::{BinaryOp, Span, TypeName, UnaryOp};
use crate::types::TypeExt;
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use std::collections::HashMap;

// =============================================================================
// Values
// =============================================================================

/// A compile-time value.
#[derive(Debug, Clone)]
pub enum Value {
    /// Known scalar: int, float, bool
    Int(i64),
    Float(f64),
    Bool(bool),

    /// Known aggregate (tuple, array, vector)
    Tuple(Vec<Value>),
    Array(Vec<Value>),
    Vector(Vec<Value>),

    /// Partial application: function waiting for more args. Each
    /// accumulated arg carries its source type so the reifier can
    /// rebuild the App term without re-deriving types from the def.
    Partial {
        sym: SymbolId,
        args: Vec<(Value, Type<TypeName>)>,
        remaining: usize,
    },

    /// Unknown at compile time - residual code
    Unknown(Term),
}

impl Value {
    fn is_known(&self) -> bool {
        !matches!(self, Value::Unknown(_))
    }
}

// =============================================================================
// Evaluator
// =============================================================================

pub struct PartialEvaluator {
    /// Symbol table for name lookup
    symbols: SymbolTable,
    /// Function definitions with their arities
    defs: HashMap<SymbolId, Def>,
    /// Term ID source for generating new terms
    term_ids: TermIdSource,
    /// Environment: symbol -> Value
    env: HashMap<SymbolId, Value>,
}

impl PartialEvaluator {
    pub fn partial_eval(program: Program) -> Program {
        program.assert_flat_apps();
        let mut eval = Self {
            symbols: program.symbols,
            defs: program.defs.iter().map(|d| (d.name, d.clone())).collect(),
            term_ids: TermIdSource::new(),
            env: HashMap::new(),
        };

        let defs = program
            .defs
            .into_iter()
            .map(|def| {
                let body_val = eval.eval(&def.body);
                let body = eval.reify(body_val, &def.body.ty, def.body.span);
                Def { body, ..def }
            })
            .collect();

        let result = Program {
            defs,
            symbols: eval.symbols,
            ..program
        };
        result.assert_flat_apps();
        result
    }

    /// Evaluate a term to a Value.
    fn eval(&mut self, term: &Term) -> Value {
        match &term.kind {
            // Literals → known values
            TermKind::IntLit(s) => Value::Int(
                s.parse().unwrap_or_else(|_| panic!("BUG: invalid integer literal from lexer: {}", s)),
            ),
            TermKind::FloatLit(f) => Value::Float(*f as f64),
            TermKind::BoolLit(b) => Value::Bool(*b),
            TermKind::UnitLit => Value::Unknown(term.clone()),
            TermKind::Coerce { inner, target_ty } => {
                let inner_val = self.eval(inner);
                let inner_term = self.reify(inner_val, &inner.ty, inner.span);
                Value::Unknown(self.mk_term(
                    term.ty.clone(),
                    term.span,
                    TermKind::Coerce {
                        inner: Box::new(inner_term),
                        target_ty: target_ty.clone(),
                    },
                ))
            }

            // Variable lookup
            TermKind::Var(VarRef::Symbol(sym)) => {
                let sym = *sym;
                if let Some(val) = self.env.get(&sym) {
                    val.clone()
                } else if let Some(def) = self.defs.get(&sym).cloned() {
                    if def.arity == 0 {
                        // Constant - evaluate it
                        self.eval(&def.body)
                    } else {
                        // Function - create partial application with 0 args applied
                        Value::Partial {
                            sym,
                            args: vec![],
                            remaining: def.arity,
                        }
                    }
                } else {
                    // Unknown variable (intrinsic or undefined)
                    Value::Unknown(term.clone())
                }
            }

            // Builtin reference: not constant-foldable on its own.
            TermKind::Var(VarRef::Builtin { .. }) => Value::Unknown(term.clone()),

            // Let binding
            TermKind::Let { name, rhs, body, .. } => {
                let rhs_val = self.eval(rhs);
                self.env.insert(*name, rhs_val);
                self.eval(body)
            }

            // If expression
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond_val = self.eval(cond);
                match cond_val {
                    Value::Bool(true) => self.eval(then_branch),
                    Value::Bool(false) => self.eval(else_branch),
                    _ => {
                        // Unknown condition - residualize
                        let then_val = self.eval(then_branch);
                        let else_val = self.eval(else_branch);
                        self.reify_if(cond_val, then_val, else_val, term)
                    }
                }
            }

            // Application - evaluate args (pairing each with its source
            // term type) and apply. Carrying types alongside values lets
            // downstream reification rebuild the App without re-deriving
            // types from the def.
            TermKind::App { func, args } => {
                let arg_vals: Vec<(Value, Type<TypeName>)> =
                    args.iter().map(|a| (self.eval(a), a.ty.clone())).collect();
                self.apply(func, arg_vals, term)
            }

            // Lambda - descend into body to fold constant sub-expressions.
            TermKind::Lambda(lam) => {
                let body = self.fold_in_body(&lam.body);
                Value::Unknown(self.mk_term(
                    term.ty.clone(),
                    term.span,
                    TermKind::Lambda(Lambda {
                        params: lam.params.clone(),
                        body: Box::new(body),
                        ret_ty: lam.ret_ty.clone(),
                    }),
                ))
            }

            // Loop - residualize (not evaluating loops at compile time)
            TermKind::Loop { .. } => Value::Unknown(term.clone()),

            // Operators as values - residualize
            TermKind::BinOp(_) | TermKind::UnOp(_) => Value::Unknown(term.clone()),

            // Extern declarations - residualize (linked at SPIR-V level)
            TermKind::Extern(_) => Value::Unknown(term.clone()),

            // SOAC nodes are opaque to partial evaluation, but their sub-
            // terms may reference env-bound `Var`s (e.g. `let m = lit in
            // map(f, m)` — the SOAC consumes `m`). Substitute those refs
            // through the SOAC's children so the dissolved `Let` doesn't
            // leave dangling free vars downstream.
            TermKind::Soac(_) | TermKind::ArrayExpr(_) => {
                let mut bound = std::collections::HashSet::new();
                let subst = self.substitute_residual_vars(term.clone(), &mut bound);
                Value::Unknown(subst)
            }

            // Structural ops: evaluate children so let-bound `Var`s
            // get substituted through, then rebuild the variant. Without
            // this an enclosing `Let` whose body references one of these
            // would leave a dangling `Var(name)` after the let is
            // dissolved by the eval pass.
            TermKind::Tuple(parts) => {
                let part_vals: Vec<Value> = parts.iter().map(|p| self.eval(p)).collect();
                let part_terms: Vec<Term> =
                    parts.iter().zip(part_vals).map(|(p, v)| self.reify(v, &p.ty, p.span)).collect();
                Value::Unknown(self.mk_term(term.ty.clone(), term.span, TermKind::Tuple(part_terms)))
            }
            TermKind::TupleProj { tuple, idx } => {
                let tuple_val = self.eval(tuple);
                let tuple_term = self.reify(tuple_val, &tuple.ty, tuple.span);
                Value::Unknown(self.mk_term(
                    term.ty.clone(),
                    term.span,
                    TermKind::TupleProj {
                        tuple: Box::new(tuple_term),
                        idx: *idx,
                    },
                ))
            }
            TermKind::Index { array, index } => {
                let array_val = self.eval(array);
                let index_val = self.eval(index);
                let array_term = self.reify(array_val, &array.ty, array.span);
                let index_term = self.reify(index_val, &index.ty, index.span);
                Value::Unknown(self.mk_term(
                    term.ty.clone(),
                    term.span,
                    TermKind::Index {
                        array: Box::new(array_term),
                        index: Box::new(index_term),
                    },
                ))
            }
            TermKind::VecLit(parts) => {
                let part_vals: Vec<Value> = parts.iter().map(|p| self.eval(p)).collect();
                let part_terms: Vec<Term> =
                    parts.iter().zip(part_vals).map(|(p, v)| self.reify(v, &p.ty, p.span)).collect();
                Value::Unknown(self.mk_term(term.ty.clone(), term.span, TermKind::VecLit(part_terms)))
            }
            TermKind::OutputSlotStore { .. } => {
                unreachable!("OutputSlotStore introduced by tlc::normalize_outputs (post-partial_eval)")
            }
        }
    }

    /// Apply a function to arguments based on the base term kind.
    fn apply(&mut self, base: &Term, args: Vec<(Value, Type<TypeName>)>, original: &Term) -> Value {
        match &base.kind {
            TermKind::BinOp(op) => {
                if args.len() >= 2 {
                    self.eval_binop(op, &args[0].0, &args[1].0, original)
                } else {
                    Value::Unknown(original.clone())
                }
            }

            TermKind::UnOp(op) => {
                if !args.is_empty() {
                    self.eval_unop(op, &args[0].0, original)
                } else {
                    Value::Unknown(original.clone())
                }
            }

            TermKind::Var(VarRef::Symbol(sym)) => self.apply_var(*sym, args, original),

            TermKind::Var(VarRef::Builtin { .. }) => {
                // Catalog builtin — opaque to partial_eval, but we
                // still residualize via the args so let-binding
                // substitutions performed by the inner `eval(arg)`
                // calls survive into the residual term.
                self.residualize_call(base.clone(), args, original)
            }

            _ => {
                // Higher-order or computed function - can't evaluate.
                // Residualize through the eval'd args, not the original
                // term, so substitutions in args don't get clobbered.
                self.residualize_call(base.clone(), args, original)
            }
        }
    }

    /// Rebuild an App from the (already-evaluated) `func` and `args`,
    /// reifying each arg back to a Term. Used when partial_eval can't
    /// reduce the call further but the args may have been substituted
    /// (e.g. a let-bound `Var` resolved to its rhs term). Cloning the
    /// original term in this position would discard those substitutions.
    fn residualize_call(
        &mut self,
        func: Term,
        args: Vec<(Value, Type<TypeName>)>,
        original: &Term,
    ) -> Value {
        let arg_terms: Vec<Term> =
            args.into_iter().map(|(arg, ty)| self.reify(arg, &ty, original.span)).collect();
        let result = self.mk_term(
            original.ty.clone(),
            original.span,
            TermKind::App {
                func: Box::new(func),
                args: arg_terms,
            },
        );
        Value::Unknown(result)
    }

    /// Apply a named function to arguments.
    fn apply_var(&mut self, sym: SymbolId, args: Vec<(Value, Type<TypeName>)>, original: &Term) -> Value {
        // Check if this is a let-bound variable aliasing a function.
        // This handles cases like `let f = g in f x` where g is a known function.
        if let Some(Value::Partial {
            sym: real_sym,
            args: partial_args,
            ..
        }) = self.env.get(&sym).cloned()
        {
            // Combine the partial application's args with the new args
            let mut combined_args = partial_args;
            combined_args.extend(args);
            // Apply to the real function (recursive to handle chains like let h = f in let g = h in g x)
            return self.apply_var(real_sym, combined_args, original);
        }

        // Check if this is a let-bound variable aliasing a function name
        // (intrinsic, builtin, or top-level def). Handles `let f = f32.sin in f x`.
        if let Some(Value::Unknown(Term {
            kind: TermKind::Var(VarRef::Symbol(real_sym)),
            ..
        })) = self.env.get(&sym)
        {
            let real_sym = *real_sym;
            return self.apply_var(real_sym, args, original);
        }

        // Check for known function
        if let Some(def) = self.defs.get(&sym).cloned() {
            let args_len = args.len();
            let all_known = args.iter().all(|(v, _)| v.is_known());
            if args_len >= def.arity && def.arity > 0 && all_known {
                // Fully applied with known args - inline
                self.inline(&def, args)
            } else if args_len < def.arity {
                // Partial application
                Value::Partial {
                    sym,
                    args,
                    remaining: def.arity - args_len,
                }
            } else {
                // Some unknown args or zero-arity - residualize
                self.reify_call(sym, args, original)
            }
        } else {
            // Unknown function - residualize
            self.reify_call(sym, args, original)
        }
    }

    /// Inline a function call.
    fn inline(&mut self, def: &Def, args: Vec<(Value, Type<TypeName>)>) -> Value {
        let mut body = &def.body;
        let mut args_iter = args.into_iter();

        loop {
            if let TermKind::Lambda(Lambda {
                params, body: inner, ..
            }) = &body.kind
            {
                // Bind as many args as this lambda has params
                let mut consumed = 0;
                for (param, _) in params {
                    if let Some((arg, _arg_ty)) = args_iter.next() {
                        self.env.insert(*param, arg);
                        consumed += 1;
                    } else {
                        break;
                    }
                }
                body = inner;
                if consumed == 0 || args_iter.len() == 0 {
                    break;
                }
            } else {
                break;
            }
        }

        self.eval(body)
    }

    /// Evaluate binary operation.
    fn eval_binop(&self, op: &BinaryOp, lhs: &Value, rhs: &Value, original: &Term) -> Value {
        match (op.op.as_str(), lhs, rhs) {
            // Integer arithmetic
            ("+", Value::Int(a), Value::Int(b)) => Value::Int(a + b),
            ("-", Value::Int(a), Value::Int(b)) => Value::Int(a - b),
            ("*", Value::Int(a), Value::Int(b)) => Value::Int(a * b),
            ("/", Value::Int(a), Value::Int(b)) if *b != 0 => Value::Int(*a / *b),

            // Float arithmetic
            ("+", Value::Float(a), Value::Float(b)) => Value::Float(a + b),
            ("-", Value::Float(a), Value::Float(b)) => Value::Float(a - b),
            ("*", Value::Float(a), Value::Float(b)) => Value::Float(a * b),
            ("/", Value::Float(a), Value::Float(b)) => Value::Float(a / b),

            // Comparisons
            ("==", Value::Int(a), Value::Int(b)) => Value::Bool(a == b),
            ("!=", Value::Int(a), Value::Int(b)) => Value::Bool(a != b),
            ("<", Value::Int(a), Value::Int(b)) => Value::Bool(a < b),
            (">", Value::Int(a), Value::Int(b)) => Value::Bool(a > b),
            ("<=", Value::Int(a), Value::Int(b)) => Value::Bool(a <= b),
            (">=", Value::Int(a), Value::Int(b)) => Value::Bool(a >= b),

            // Algebraic identities
            ("+", Value::Int(0), _) => rhs.clone(),
            ("+", _, Value::Int(0)) => lhs.clone(),
            ("*", Value::Int(1), _) => rhs.clone(),
            ("*", _, Value::Int(1)) => lhs.clone(),
            ("*", Value::Int(0), _) | ("*", _, Value::Int(0)) => Value::Int(0),

            _ => Value::Unknown(original.clone()),
        }
    }

    /// Evaluate unary operation.
    fn eval_unop(&self, op: &UnaryOp, arg: &Value, original: &Term) -> Value {
        match (op.op.as_str(), arg) {
            ("-", Value::Int(n)) => Value::Int(-n),
            ("-", Value::Float(f)) => Value::Float(-f),
            ("!", Value::Bool(b)) => Value::Bool(!b),
            _ => Value::Unknown(original.clone()),
        }
    }

    // =========================================================================
    // Structural constant folding inside opaque bodies (lambdas, loops)
    // =========================================================================

    /// Structurally descend into a term, folding constant sub-expressions
    /// (like `1.0 / 2.2` → `0.4545`) without changing the Let/Var structure.
    fn fold_in_body(&mut self, term: &Term) -> Term {
        match &term.kind {
            // Constant sub-expression: App of BinOp/UnOp on literals
            TermKind::App { func, args } if matches!(func.kind, TermKind::BinOp(_) | TermKind::UnOp(_)) => {
                let folded_args: Vec<Term> = args.iter().map(|a| self.fold_in_body(a)).collect();
                // Try to evaluate if all args are literals
                let arg_vals: Vec<(Value, Type<TypeName>)> =
                    folded_args.iter().map(|a| (self.try_literal_value(a), a.ty.clone())).collect();
                if arg_vals.iter().all(|(v, _)| !matches!(v, Value::Unknown(_))) {
                    let result = self.apply(func, arg_vals, term);
                    if !matches!(result, Value::Unknown(_)) {
                        return self.reify(result, &term.ty, term.span);
                    }
                }
                // Couldn't fold — rebuild with folded children
                self.mk_term(
                    term.ty.clone(),
                    term.span,
                    TermKind::App {
                        func: func.clone(),
                        args: folded_args,
                    },
                )
            }
            // Recurse into Let
            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                let rhs = self.fold_in_body(rhs);
                let body = self.fold_in_body(body);
                self.mk_term(
                    term.ty.clone(),
                    term.span,
                    TermKind::Let {
                        name: *name,
                        name_ty: name_ty.clone(),
                        rhs: Box::new(rhs),
                        body: Box::new(body),
                    },
                )
            }
            // Recurse into If
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond = self.fold_in_body(cond);
                let then_branch = self.fold_in_body(then_branch);
                let else_branch = self.fold_in_body(else_branch);
                self.mk_term(
                    term.ty.clone(),
                    term.span,
                    TermKind::If {
                        cond: Box::new(cond),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    },
                )
            }
            // Recurse into App (non-operator)
            TermKind::App { func, args } => {
                let func = self.fold_in_body(func);
                let args = args.iter().map(|a| self.fold_in_body(a)).collect();
                self.mk_term(
                    term.ty.clone(),
                    term.span,
                    TermKind::App {
                        func: Box::new(func),
                        args,
                    },
                )
            }
            // Recurse into Lambda
            TermKind::Lambda(lam) => {
                let body = self.fold_in_body(&lam.body);
                self.mk_term(
                    term.ty.clone(),
                    term.span,
                    TermKind::Lambda(Lambda {
                        params: lam.params.clone(),
                        body: Box::new(body),
                        ret_ty: lam.ret_ty.clone(),
                    }),
                )
            }
            // Recurse into Loop
            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => {
                let init = self.fold_in_body(init);
                let body = self.fold_in_body(body);
                self.mk_term(
                    term.ty.clone(),
                    term.span,
                    TermKind::Loop {
                        loop_var: *loop_var,
                        loop_var_ty: loop_var_ty.clone(),
                        init: Box::new(init),
                        init_bindings: init_bindings.clone(),
                        kind: kind.clone(),
                        body: Box::new(body),
                    },
                )
            }
            // Leaves and everything else — return unchanged
            _ => term.clone(),
        }
    }

    /// Try to extract a literal Value from a term. Returns Unknown for non-literals.
    fn try_literal_value(&self, term: &Term) -> Value {
        match &term.kind {
            TermKind::IntLit(s) => Value::Int(
                s.parse()
                    .unwrap_or_else(|e| panic!("lexer-produced IntLit `{s}` failed to parse as i64: {e}")),
            ),
            TermKind::FloatLit(f) => Value::Float(*f as f64),
            TermKind::BoolLit(b) => Value::Bool(*b),
            _ => Value::Unknown(term.clone()),
        }
    }

    // =========================================================================
    // Reification (Value → Term)
    // =========================================================================

    fn reify(&mut self, value: Value, ty: &Type<TypeName>, span: Span) -> Term {
        match value {
            Value::Int(n) => self.mk_term(ty.clone(), span, TermKind::IntLit(n.to_string())),
            Value::Float(f) => self.mk_term(ty.clone(), span, TermKind::FloatLit(f as f32)),
            Value::Bool(b) => self.mk_term(ty.clone(), span, TermKind::BoolLit(b)),
            Value::Unknown(t) => t,
            Value::Tuple(elems) => self.reify_tuple(elems, ty, span),
            Value::Array(elems) => self.reify_array(elems, ty, span),
            Value::Vector(elems) => self.reify_vector(elems, ty, span),
            Value::Partial { sym, args, .. } => self.reify_partial(sym, args, ty, span),
        }
    }

    /// Binder-aware walk that replaces every free `Var(VarRef::Symbol(sym))`
    /// (one not bound by an enclosing `Let`/`Lambda`) with the reified
    /// `env[sym]` value, if any. Used when residualizing nodes like `Soac`
    /// and `ArrayExpr` that the evaluator would otherwise clone wholesale —
    /// without this, an enclosing `let m = lit in soac(..., m)` dissolves
    /// the let and leaves a dangling `Var(m)` inside the SOAC.
    ///
    /// `bound` tracks symbols currently in scope (shadowing env). It's
    /// mutated in place as we descend into binders and restored on exit.
    fn substitute_residual_vars(
        &mut self,
        term: Term,
        bound: &mut std::collections::HashSet<SymbolId>,
    ) -> Term {
        let ty = term.ty.clone();
        let span = term.span;
        match term.kind {
            TermKind::Var(VarRef::Symbol(name)) => {
                if !bound.contains(&name) {
                    if let Some(val) = self.env.get(&name).cloned() {
                        return self.reify(val, &ty, span);
                    }
                }
                Term {
                    id: term.id,
                    ty,
                    span,
                    kind: TermKind::Var(VarRef::Symbol(name)),
                }
            }
            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                let new_rhs = self.substitute_residual_vars(*rhs, bound);
                let added = bound.insert(name);
                let new_body = self.substitute_residual_vars(*body, bound);
                if added {
                    bound.remove(&name);
                }
                Term {
                    id: term.id,
                    ty,
                    span,
                    kind: TermKind::Let {
                        name,
                        name_ty,
                        rhs: Box::new(new_rhs),
                        body: Box::new(new_body),
                    },
                }
            }
            TermKind::Lambda(lam) => {
                let mut added: Vec<SymbolId> = Vec::new();
                for (p, _) in &lam.params {
                    if bound.insert(*p) {
                        added.push(*p);
                    }
                }
                let new_body = self.substitute_residual_vars(*lam.body, bound);
                for p in added {
                    bound.remove(&p);
                }
                Term {
                    id: term.id,
                    ty,
                    span,
                    kind: TermKind::Lambda(Lambda {
                        params: lam.params,
                        body: Box::new(new_body),
                        ret_ty: lam.ret_ty,
                    }),
                }
            }
            other => {
                // Non-binder: recurse through every Term child via
                // `map_children` (which covers App, If, Soac, ArrayExpr,
                // Tuple, etc. recursively at the structural level).
                let outer = Term {
                    id: term.id,
                    ty: ty.clone(),
                    span,
                    kind: other,
                };
                outer.map_children(&mut |child| self.substitute_residual_vars(child, bound))
            }
        }
    }

    fn reify_tuple(&mut self, elems: Vec<Value>, ty: &Type<TypeName>, span: Span) -> Term {
        let component_types = match ty {
            Type::Constructed(TypeName::Tuple(_), args) => args.as_slice(),
            other => panic!(
                "reify_tuple dispatched on non-tuple type {other:?} — \
                 caller (`reify` for Value::Tuple) is responsible for the type shape",
            ),
        };
        assert_eq!(
            elems.len(),
            component_types.len(),
            "reify_tuple: element count {} does not match tuple type arity {}",
            elems.len(),
            component_types.len(),
        );
        let part_terms: Vec<Term> = elems
            .into_iter()
            .zip(component_types.iter())
            .map(|(elem, elem_ty)| self.reify(elem, elem_ty, span))
            .collect();
        self.mk_term(ty.clone(), span, TermKind::Tuple(part_terms))
    }

    fn reify_array(&mut self, elems: Vec<Value>, ty: &Type<TypeName>, span: Span) -> Term {
        let elem_ty = crate::types::array_elem(ty).cloned().unwrap_or_else(|| {
            panic!(
                "reify_array dispatched on non-array type {ty:?} — \
                 caller (`reify` for Value::Array) is responsible for the type shape"
            )
        });
        let part_terms: Vec<Term> =
            elems.into_iter().map(|elem| self.reify(elem, &elem_ty, span)).collect();
        self.mk_term(
            ty.clone(),
            span,
            TermKind::ArrayExpr(super::ArrayExpr::Literal(part_terms)),
        )
    }

    fn reify_vector(&mut self, elems: Vec<Value>, ty: &Type<TypeName>, span: Span) -> Term {
        let elem_ty = ty.elem_type().cloned().unwrap_or_else(|| {
            panic!(
                "reify_vector dispatched on non-vec type {ty:?} — \
                 caller (`reify` for Value::Vector) is responsible for the type shape"
            )
        });
        let part_terms: Vec<Term> =
            elems.into_iter().map(|elem| self.reify(elem, &elem_ty, span)).collect();
        self.mk_term(ty.clone(), span, TermKind::VecLit(part_terms))
    }

    fn reify_partial(
        &mut self,
        sym: SymbolId,
        args: Vec<(Value, Type<TypeName>)>,
        ty: &Type<TypeName>,
        span: Span,
    ) -> Term {
        let func_term = self.mk_term(ty.clone(), span, TermKind::Var(VarRef::Symbol(sym)));
        let arg_terms: Vec<Term> =
            args.into_iter().map(|(arg, arg_ty)| self.reify(arg, &arg_ty, span)).collect();
        self.mk_term(
            ty.clone(),
            span,
            TermKind::App {
                func: Box::new(func_term),
                args: arg_terms,
            },
        )
    }

    fn reify_call(&mut self, sym: SymbolId, args: Vec<(Value, Type<TypeName>)>, original: &Term) -> Value {
        let func_term = self.mk_term(
            original.ty.clone(),
            original.span,
            TermKind::Var(VarRef::Symbol(sym)),
        );
        let arg_terms: Vec<Term> =
            args.into_iter().map(|(arg, arg_ty)| self.reify(arg, &arg_ty, original.span)).collect();
        let result = self.mk_term(
            original.ty.clone(),
            original.span,
            TermKind::App {
                func: Box::new(func_term),
                args: arg_terms,
            },
        );
        Value::Unknown(result)
    }

    fn reify_if(&mut self, cond: Value, then_val: Value, else_val: Value, original: &Term) -> Value {
        let cond_term = self.reify(cond, &Type::Constructed(TypeName::Bool, vec![]), original.span);
        let then_term = self.reify(then_val, &original.ty, original.span);
        let else_term = self.reify(else_val, &original.ty, original.span);

        Value::Unknown(self.mk_term(
            original.ty.clone(),
            original.span,
            TermKind::If {
                cond: Box::new(cond_term),
                then_branch: Box::new(then_term),
                else_branch: Box::new(else_term),
            },
        ))
    }

    fn mk_term(&mut self, ty: Type<TypeName>, span: Span, kind: TermKind) -> Term {
        Term {
            id: self.term_ids.next_id(),
            ty,
            span,
            kind,
        }
    }
}

#[cfg(test)]
#[path = "partial_eval_tests.rs"]
mod partial_eval_tests;
