//! Stack-based partial evaluator for TLC.
//!
//! Simpler than NBE-style: collect application spines, evaluate args,
//! apply when we have enough arguments (using arity metadata).

use super::{Def, FunctionName, Program, Term, TermIdSource, TermKind};
use crate::ast::{BinaryOp, Span, TypeName, UnaryOp};
use crate::scope::ScopeStack;
use polytype::Type;
use std::collections::HashMap; // For defs lookup

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
        name: String,
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
    /// Function definitions with their arities
    defs: HashMap<String, Def>,
    /// Term ID source for generating new terms
    term_ids: TermIdSource,
    /// Scope stack for variable bindings
    env: ScopeStack<Value>,
}

impl PartialEvaluator {
    pub fn partial_eval(program: Program) -> Program {
        let mut eval = Self {
            defs: program.defs.iter().map(|d| (d.name.clone(), d.clone())).collect(),
            term_ids: TermIdSource::new(),
            env: ScopeStack::new(),
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
        }
    }

    /// Evaluate a term to a Value.
    fn eval(&mut self, term: &Term) -> Value {
        match &term.kind {
            // Literals → known values
            TermKind::IntLit(s) => Value::Int(s.parse().unwrap_or(0)),
            TermKind::FloatLit(f) => Value::Float(*f as f64),
            TermKind::BoolLit(b) => Value::Bool(*b),
            TermKind::StringLit(_) => Value::Unknown(term.clone()),

            // Variable lookup
            TermKind::Var(name) => {
                if let Some(val) = self.env.lookup(name) {
                    val.clone()
                } else if let Some(def) = self.defs.get(name).cloned() {
                    if def.arity == 0 {
                        // Constant - evaluate it
                        self.eval(&def.body)
                    } else {
                        // Function - create partial application with 0 args applied
                        Value::Partial {
                            name: name.clone(),
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
                self.env.push_scope();
                self.env.insert(name.clone(), rhs_val);
                let result = self.eval(body);
                self.env.pop_scope();
                result
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
                let (func_name, var_name, args) = self.collect_spine(term);
                let arg_vals: Vec<Value> = args.iter().map(|a| self.eval(a)).collect();

                if let Some(func) = func_name {
                    self.apply(func, arg_vals, term)
                } else if let Some(name) = var_name {
                    self.apply_var(name, arg_vals, term)
                } else {
                    // Complex function term - residualize
                    Value::Unknown(term.clone())
                }
            }

            // Lambda - residualize (should be handled at def level)
            TermKind::Lam { .. } => Value::Unknown(term.clone()),

            // Loop - residualize (not evaluating loops at compile time)
            TermKind::Loop { .. } => Value::Unknown(term.clone()),
        }
    }

    /// Collect the spine of an application: App(App(f, x), y) → (func_name, [x, y])
    /// Returns None for the function name if it's a complex term (not a simple Var/BinOp/UnOp).
    fn collect_spine<'a>(
        &self,
        term: &'a Term,
    ) -> (Option<&'a FunctionName>, Option<&'a str>, Vec<&'a Term>) {
        let mut args = Vec::new();
        let mut current = term;

        while let TermKind::App { func, arg } = &current.kind {
            args.push(arg.as_ref());
            match func.as_ref() {
                FunctionName::Term(inner) => current = inner,
                func_name => return (Some(func_name), None, args.into_iter().rev().collect()),
            }
        }

        // Reached a non-App term - check if it's a Var
        if let TermKind::Var(name) = &current.kind {
            return (None, Some(name.as_str()), args.into_iter().rev().collect());
        }

        // Complex term in function position - can't collect
        (None, None, args.into_iter().rev().collect())
    }

    /// Apply a function to arguments.
    fn apply(&mut self, func: &FunctionName, args: Vec<Value>, original: &Term) -> Value {
        match func {
            FunctionName::BinOp(op) => {
                if args.len() >= 2 {
                    self.eval_binop(op, &args[0], &args[1], original)
                } else {
                    Value::Unknown(original.clone())
                }
            }

            FunctionName::UnOp(op) => {
                if !args.is_empty() {
                    self.eval_unop(op, &args[0], original)
                } else {
                    Value::Unknown(original.clone())
                }
            }

            FunctionName::Var(name) => self.apply_var(name, args, original),

            FunctionName::Term(_) => {
                // Higher-order - shouldn't happen after spine collection
                Value::Unknown(original.clone())
            }
        }
    }

    /// Apply a named function to arguments.
    fn apply_var(&mut self, name: &str, args: Vec<Value>, original: &Term) -> Value {
        // Check for intrinsics first
        if let Some(val) = self.try_intrinsic(name, &args, original) {
            return val;
        }

        // Check if this is a let-bound variable aliasing a function.
        // This handles cases like `let f = g in f x` where g is a known function.
        if let Some(Value::Partial {
            name: real_name,
            args: partial_args,
            ..
        }) = self.env.lookup(name).cloned()
        {
            // Combine the partial application's args with the new args
            let mut combined_args = partial_args;
            combined_args.extend(args);
            // Apply to the real function (recursive to handle chains like let h = f in let g = h in g x)
            return self.apply_var(&real_name, combined_args, original);
        }

        // Check if this is a let-bound variable aliasing a function name
        // (intrinsic, builtin, or top-level def). Handles `let f = f32.sin in f x`.
        if let Some(Value::Unknown(Term {
            kind: TermKind::Var(real_name),
            ..
        })) = self.env.lookup(name)
        {
            let real_name = real_name.clone();
            return self.apply_var(&real_name, args, original);
        }

        // Check for known function
        if let Some(def) = self.defs.get(name).cloned() {
            let args_len = args.len();
            let all_known = args.iter().all(|a| a.is_known());
            if args_len >= def.arity && def.arity > 0 && all_known {
                // Fully applied with known args - inline
                self.inline(&def, args)
            } else if args_len < def.arity {
                // Partial application
                Value::Partial {
                    name: name.to_string(),
                    args,
                    remaining: def.arity - args_len,
                }
            } else {
                // Some unknown args or zero-arity - residualize
                self.reify_call(name, args, original)
            }
        } else {
            // Unknown function - residualize
            self.reify_call(name, args, original)
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
        // Push scope and bind parameters
        self.env.push_scope();
        let mut body = &def.body;

        for arg in args {
            if let TermKind::Lam {
                param, body: inner, ..
            } = &body.kind
            {
                self.env.insert(param.clone(), arg);
                body = inner;
            } else {
                break;
            }
        }

        let result = self.eval(body);
        self.env.pop_scope();
        result
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
            Value::Partial { name, args, .. } => self.reify_partial(&name, args, ty, span),
        }
    }

    fn reify_tuple(&mut self, elems: Vec<Value>, ty: &Type<TypeName>, span: Span) -> Term {
        // Build: _w_tuple elem0 elem1 ...
        let unit_ty = Type::Constructed(TypeName::Str("unit"), vec![]);
        let mut result = self.mk_term(ty.clone(), span, TermKind::Var("_w_tuple".to_string()));

        for elem in elems {
            let elem_term = self.reify(elem, &unit_ty, span);
            result = self.mk_term(
                ty.clone(),
                span,
                TermKind::App {
                    func: Box::new(FunctionName::Term(Box::new(result))),
                    arg: Box::new(elem_term),
                },
            );
        }
        result
    }

    fn reify_array(&mut self, elems: Vec<Value>, ty: &Type<TypeName>, span: Span) -> Term {
        let unit_ty = Type::Constructed(TypeName::Str("unit"), vec![]);
        let mut result = self.mk_term(ty.clone(), span, TermKind::Var("_w_array_lit".to_string()));

        for elem in elems {
            let elem_term = self.reify(elem, &unit_ty, span);
            result = self.mk_term(
                ty.clone(),
                span,
                TermKind::App {
                    func: Box::new(FunctionName::Term(Box::new(result))),
                    arg: Box::new(elem_term),
                },
            );
        }
        result
    }

    fn reify_vector(&mut self, elems: Vec<Value>, ty: &Type<TypeName>, span: Span) -> Term {
        let unit_ty = Type::Constructed(TypeName::Str("unit"), vec![]);
        let mut result = self.mk_term(ty.clone(), span, TermKind::Var("_w_vec_lit".to_string()));

        for elem in elems {
            let elem_term = self.reify(elem, &unit_ty, span);
            result = self.mk_term(
                ty.clone(),
                span,
                TermKind::App {
                    func: Box::new(FunctionName::Term(Box::new(result))),
                    arg: Box::new(elem_term),
                },
            );
        }
        result
    }

    fn reify_partial(&mut self, name: &str, args: Vec<Value>, ty: &Type<TypeName>, span: Span) -> Term {
        let unit_ty = Type::Constructed(TypeName::Str("unit"), vec![]);
        let mut result = self.mk_term(ty.clone(), span, TermKind::Var(name.to_string()));

        for arg in args {
            let arg_term = self.reify(arg, &unit_ty, span);
            result = self.mk_term(
                ty.clone(),
                span,
                TermKind::App {
                    func: Box::new(FunctionName::Term(Box::new(result))),
                    arg: Box::new(arg_term),
                },
            );
        }
        result
    }

    fn reify_call(&mut self, name: &str, args: Vec<Value>, original: &Term) -> Value {
        let unit_ty = Type::Constructed(TypeName::Str("unit"), vec![]);
        let mut result = self.mk_term(
            original.ty.clone(),
            original.span,
            TermKind::Var(name.to_string()),
        );

        for arg in args {
            let arg_term = self.reify(arg, &unit_ty, original.span);
            result = self.mk_term(
                original.ty.clone(),
                original.span,
                TermKind::App {
                    func: Box::new(FunctionName::Term(Box::new(result))),
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
