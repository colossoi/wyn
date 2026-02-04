//! Stack-based partial evaluator for TLC.
//!
//! Simpler than NBE-style: collect application spines, evaluate args,
//! apply when we have enough arguments (using arity metadata).

use super::{Def, Program, Term, TermIdSource, TermKind};
use crate::ast::{BinaryOp, Span, TypeName, UnaryOp};
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

    /// Partial application: function waiting for more args
    Partial {
        sym: SymbolId,
        args: Vec<Value>,
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

        Program {
            defs,
            uniforms: program.uniforms,
            storage: program.storage,
            symbols: eval.symbols,
        }
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
            TermKind::StringLit(_) => Value::Unknown(term.clone()),

            // Variable lookup
            TermKind::Var(sym) => {
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

            // Application - collect spine and apply
            TermKind::App { .. } => {
                let (base, args) = self.collect_spine(term);
                let arg_vals: Vec<Value> = args.iter().map(|a| self.eval(a)).collect();
                self.apply(base, arg_vals, term)
            }

            // Lambda - residualize (should be handled at def level)
            TermKind::Lam { .. } => Value::Unknown(term.clone()),

            // Loop - residualize (not evaluating loops at compile time)
            TermKind::Loop { .. } => Value::Unknown(term.clone()),

            // Operators as values - residualize
            TermKind::BinOp(_) | TermKind::UnOp(_) => Value::Unknown(term.clone()),

            // Extern declarations - residualize (linked at SPIR-V level)
            TermKind::Extern(_) => Value::Unknown(term.clone()),
        }
    }

    /// Collect the spine of an application: App(App(f, x), y) → (base_term, [x, y])
    /// Returns the base term and collected arguments.
    fn collect_spine<'a>(&self, term: &'a Term) -> (&'a Term, Vec<&'a Term>) {
        let mut args = Vec::new();
        let mut current = term;

        while let TermKind::App { func, arg } = &current.kind {
            args.push(arg.as_ref());
            current = func.as_ref();
        }

        args.reverse();
        (current, args)
    }

    /// Apply a function to arguments based on the base term kind.
    fn apply(&mut self, base: &Term, args: Vec<Value>, original: &Term) -> Value {
        match &base.kind {
            TermKind::BinOp(op) => {
                if args.len() >= 2 {
                    self.eval_binop(op, &args[0], &args[1], original)
                } else {
                    Value::Unknown(original.clone())
                }
            }

            TermKind::UnOp(op) => {
                if !args.is_empty() {
                    self.eval_unop(op, &args[0], original)
                } else {
                    Value::Unknown(original.clone())
                }
            }

            TermKind::Var(sym) => self.apply_var(*sym, args, original),

            _ => {
                // Higher-order or computed function - can't evaluate
                Value::Unknown(original.clone())
            }
        }
    }

    /// Apply a named function to arguments.
    fn apply_var(&mut self, sym: SymbolId, args: Vec<Value>, original: &Term) -> Value {
        // Check for intrinsics first (clone name to avoid borrow conflict)
        let name = self.symbols.get(sym).expect("BUG: symbol not in table").clone();
        if let Some(val) = self.try_intrinsic(&name, &args, original) {
            return val;
        }

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
            kind: TermKind::Var(real_sym),
            ..
        })) = self.env.get(&sym)
        {
            let real_sym = *real_sym;
            return self.apply_var(real_sym, args, original);
        }

        // Check for known function
        if let Some(def) = self.defs.get(&sym).cloned() {
            let args_len = args.len();
            let all_known = args.iter().all(|a| a.is_known());
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

    /// Try to evaluate an intrinsic.
    fn try_intrinsic(&mut self, name: &str, args: &[Value], _original: &Term) -> Option<Value> {
        match name {
            "_w_tuple_proj" if args.len() >= 2 => {
                if let (Value::Tuple(elems), Value::Int(idx)) = (&args[0], &args[1]) {
                    let i = *idx as usize;
                    if i < elems.len() {
                        return Some(elems[i].clone());
                    }
                }
                None
            }
            "_w_index" if args.len() >= 2 => {
                match (&args[0], &args[1]) {
                    (Value::Array(elems), Value::Int(idx)) | (Value::Vector(elems), Value::Int(idx)) => {
                        let i = *idx as usize;
                        if i < elems.len() {
                            return Some(elems[i].clone());
                        }
                    }
                    _ => {}
                }
                None
            }
            _ => None,
        }
    }

    /// Inline a function call.
    fn inline(&mut self, def: &Def, args: Vec<Value>) -> Value {
        let mut body = &def.body;

        for arg in args {
            if let TermKind::Lam {
                param, body: inner, ..
            } = &body.kind
            {
                self.env.insert(*param, arg);
                body = inner;
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

    fn reify_tuple(&mut self, elems: Vec<Value>, ty: &Type<TypeName>, span: Span) -> Term {
        // Build: _w_tuple elem0 elem1 ...
        let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
        let tuple_sym = self.symbols.alloc("_w_tuple".to_string());
        let mut result = self.mk_term(ty.clone(), span, TermKind::Var(tuple_sym));

        for elem in elems {
            let elem_term = self.reify(elem, &unit_ty, span);
            result = self.mk_term(
                ty.clone(),
                span,
                TermKind::App {
                    func: Box::new(result),
                    arg: Box::new(elem_term),
                },
            );
        }
        result
    }

    fn reify_array(&mut self, elems: Vec<Value>, ty: &Type<TypeName>, span: Span) -> Term {
        let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
        let array_sym = self.symbols.alloc("_w_array_lit".to_string());
        let mut result = self.mk_term(ty.clone(), span, TermKind::Var(array_sym));

        for elem in elems {
            let elem_term = self.reify(elem, &unit_ty, span);
            result = self.mk_term(
                ty.clone(),
                span,
                TermKind::App {
                    func: Box::new(result),
                    arg: Box::new(elem_term),
                },
            );
        }
        result
    }

    fn reify_vector(&mut self, elems: Vec<Value>, ty: &Type<TypeName>, span: Span) -> Term {
        let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
        let vec_sym = self.symbols.alloc("_w_vec_lit".to_string());
        let mut result = self.mk_term(ty.clone(), span, TermKind::Var(vec_sym));

        for elem in elems {
            let elem_term = self.reify(elem, &unit_ty, span);
            result = self.mk_term(
                ty.clone(),
                span,
                TermKind::App {
                    func: Box::new(result),
                    arg: Box::new(elem_term),
                },
            );
        }
        result
    }

    fn reify_partial(&mut self, sym: SymbolId, args: Vec<Value>, ty: &Type<TypeName>, span: Span) -> Term {
        let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
        let mut result = self.mk_term(ty.clone(), span, TermKind::Var(sym));

        for arg in args {
            let arg_term = self.reify(arg, &unit_ty, span);
            result = self.mk_term(
                ty.clone(),
                span,
                TermKind::App {
                    func: Box::new(result),
                    arg: Box::new(arg_term),
                },
            );
        }
        result
    }

    fn reify_call(&mut self, sym: SymbolId, args: Vec<Value>, original: &Term) -> Value {
        let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
        let mut result = self.mk_term(original.ty.clone(), original.span, TermKind::Var(sym));

        for arg in args {
            let arg_term = self.reify(arg, &unit_ty, original.span);
            result = self.mk_term(
                original.ty.clone(),
                original.span,
                TermKind::App {
                    func: Box::new(result),
                    arg: Box::new(arg_term),
                },
            );
        }
        Value::Unknown(result)
    }

    fn reify_if(&mut self, cond: Value, then_val: Value, else_val: Value, original: &Term) -> Value {
        let cond_term = self.reify(
            cond,
            &Type::Constructed(TypeName::Str("bool"), vec![]),
            original.span,
        );
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
