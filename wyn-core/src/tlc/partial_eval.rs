//! Partial evaluation for TLC.
//!
//! NBE-style partial evaluator that:
//! - Folds constant expressions
//! - Applies algebraic simplifications
//! - Eliminates branches with known conditions
//! - Inlines functions when arguments are known
//! - Unrolls loops with known bounds

use super::{Def, FunctionName, Program, Term, TermId, TermIdSource, TermKind};
use crate::ast::{BinaryOp, Span, TypeName, UnaryOp};
use polytype::Type;
use std::collections::HashMap;

// =============================================================================
// Value - Compile-time semantic values
// =============================================================================

/// Scalar compile-time value.
#[derive(Debug, Clone)]
pub enum ScalarValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

/// Kind of aggregate value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateKind {
    Tuple,
    Array,
    Vector,
    Matrix,
}

/// A compile-time value.
#[derive(Debug, Clone)]
pub enum Value {
    /// A known scalar value.
    Scalar(ScalarValue),

    /// A known aggregate (tuple, array, vector, matrix).
    Aggregate {
        kind: AggregateKind,
        elements: Vec<Value>,
    },

    /// A partially applied binary operator: (op lhs) waiting for rhs.
    PendingBinOp(BinaryOp, Box<Value>),

    /// A partially applied tuple projection: (_w_tuple_proj tuple) waiting for index.
    PendingTupleProj(Box<Value>),

    /// A partially applied index operation: (_w_index array) waiting for index.
    PendingIndex(Box<Value>),

    /// A closure with some arguments already applied.
    Closure {
        def_name: String,
        arity: usize,
        applied_args: Vec<Value>,
    },

    /// Unit value.
    Unit,

    /// Unknown at compile time - contains residual code.
    Unknown(Term),
}

impl Value {
    /// Check if this value is fully known at compile time.
    pub fn is_known(&self) -> bool {
        match self {
            Value::Scalar(_) | Value::Unit => true,
            Value::Aggregate { elements, .. } => elements.iter().all(|e| e.is_known()),
            Value::Closure { applied_args, .. } => applied_args.iter().all(|a| a.is_known()),
            Value::PendingBinOp(_, lhs) => lhs.is_known(),
            Value::PendingTupleProj(tuple) => tuple.is_known(),
            Value::PendingIndex(array) => array.is_known(),
            Value::Unknown(_) => false,
        }
    }
}

// =============================================================================
// Environment
// =============================================================================

/// Environment mapping variable names to values.
#[derive(Debug, Clone, Default)]
struct Env {
    bindings: HashMap<String, Value>,
}

impl Env {
    fn new() -> Self {
        Self::default()
    }

    fn get(&self, name: &str) -> Option<&Value> {
        self.bindings.get(name)
    }

    fn extend(&mut self, name: String, value: Value) {
        self.bindings.insert(name, value);
    }

    /// Create a child environment (clone for branch evaluation).
    fn child(&self) -> Self {
        self.clone()
    }
}

// =============================================================================
// Partial Evaluator
// =============================================================================

/// Cost budget for controlling code expansion.
struct CostBudget {
    max_inline_depth: usize,
    #[allow(dead_code)]
    max_loop_unroll: usize,
    current_depth: usize,
}

impl Default for CostBudget {
    fn default() -> Self {
        Self {
            max_inline_depth: 8,
            max_loop_unroll: 16,
            current_depth: 0,
        }
    }
}

/// Partial evaluator for TLC.
pub struct PartialEvaluator {
    /// Top-level definitions for inlining.
    defs: HashMap<String, Def>,
    /// Term ID generator.
    term_ids: TermIdSource,
    /// Cost budget.
    budget: CostBudget,
}

impl PartialEvaluator {
    /// Create a new partial evaluator.
    fn new(program: &Program) -> Self {
        let defs = program.defs.iter().map(|d| (d.name.clone(), d.clone())).collect();

        Self {
            defs,
            term_ids: TermIdSource::new(),
            budget: CostBudget::default(),
        }
    }

    /// Entry point: partially evaluate a program.
    pub fn partial_eval(program: Program) -> Program {
        let mut pe = Self::new(&program);

        let defs = program
            .defs
            .into_iter()
            .map(|def| {
                let mut env = Env::new();
                let body_val = pe.eval(&def.body, &mut env);
                let body = pe.reify(body_val, &def.body.ty, def.body.span);
                Def {
                    name: def.name,
                    ty: def.ty,
                    body,
                    meta: def.meta,
                    arity: def.arity,
                }
            })
            .collect();

        Program {
            defs,
            uniforms: program.uniforms,
            storage: program.storage,
        }
    }

    /// Generate a fresh term ID.
    fn fresh_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    // =========================================================================
    // Evaluation
    // =========================================================================

    /// Evaluate a term, returning a Value.
    fn eval(&mut self, term: &Term, env: &mut Env) -> Value {
        match &term.kind {
            // Literals
            TermKind::IntLit(s) => {
                if let Ok(n) = s.parse::<i64>() {
                    Value::Scalar(ScalarValue::Int(n))
                } else {
                    // Large literal - keep as unknown
                    Value::Unknown(term.clone())
                }
            }
            TermKind::FloatLit(f) => Value::Scalar(ScalarValue::Float(*f as f64)),
            TermKind::BoolLit(b) => Value::Scalar(ScalarValue::Bool(*b)),
            TermKind::StringLit(s) => Value::Scalar(ScalarValue::String(s.clone())),

            // Variable lookup
            TermKind::Var(name) => {
                if let Some(val) = env.get(name) {
                    val.clone()
                } else if let Some(def) = self.defs.get(name).cloned() {
                    // Top-level reference - check if it's a constant (no params)
                    self.try_eval_global(&def, env)
                } else {
                    // Unknown variable (parameter, intrinsic, etc.)
                    Value::Unknown(term.clone())
                }
            }

            // Let binding
            TermKind::Let { name, rhs, body, .. } => {
                let rhs_val = self.eval(rhs, env);
                env.extend(name.clone(), rhs_val);
                self.eval(body, env)
            }

            // Application
            TermKind::App { func, arg } => self.eval_app(func, arg, env, term),

            // If expression
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond_val = self.eval(cond, env);
                match &cond_val {
                    Value::Scalar(ScalarValue::Bool(true)) => self.eval(then_branch, env),
                    Value::Scalar(ScalarValue::Bool(false)) => self.eval(else_branch, env),
                    _ => {
                        // Unknown condition - evaluate both branches
                        let then_val = self.eval(then_branch, &mut env.child());
                        let else_val = self.eval(else_branch, &mut env.child());
                        self.reify_if(cond_val, then_val, else_val, term)
                    }
                }
            }

            // Lambda - create closure or residualize
            TermKind::Lam { .. } => {
                // Lambdas at this point are residualized
                // (After lifting, most lambdas are top-level defs)
                Value::Unknown(term.clone())
            }
        }
    }

    /// Try to evaluate a global definition.
    fn try_eval_global(&mut self, def: &Def, _env: &mut Env) -> Value {
        // Check if it's a constant (body is not a lambda)
        if !matches!(def.body.kind, TermKind::Lam { .. }) {
            // It's a constant - evaluate it
            if self.budget.current_depth < self.budget.max_inline_depth {
                self.budget.current_depth += 1;
                let result = self.eval(&def.body, &mut Env::new());
                self.budget.current_depth -= 1;
                return result;
            }
        }

        // It's a function or we're too deep - check arity for closure
        let arity = self.count_lambda_params(&def.body);
        if arity > 0 {
            Value::Closure {
                def_name: def.name.clone(),
                arity,
                applied_args: vec![],
            }
        } else {
            // Not a function, can't inline - return unknown
            Value::Unknown(Term {
                id: self.fresh_id(),
                ty: def.ty.clone(),
                span: def.body.span,
                kind: TermKind::Var(def.name.clone()),
            })
        }
    }

    /// Count the number of lambda parameters (curried).
    fn count_lambda_params(&self, term: &Term) -> usize {
        match &term.kind {
            TermKind::Lam { body, .. } => 1 + self.count_lambda_params(body),
            _ => 0,
        }
    }

    // =========================================================================
    // Application evaluation
    // =========================================================================

    /// Evaluate an application.
    fn eval_app(&mut self, func: &FunctionName, arg: &Term, env: &mut Env, original: &Term) -> Value {
        match func {
            FunctionName::BinOp(op) => {
                // Partial application of binary op: (op arg) returns pending
                let arg_val = self.eval(arg, env);
                Value::PendingBinOp(op.clone(), Box::new(arg_val))
            }

            FunctionName::UnOp(op) => {
                let arg_val = self.eval(arg, env);
                self.eval_unop(op, arg_val, original)
            }

            FunctionName::Var(name) => {
                // Check for loop intrinsics
                if name.starts_with("_w_loop") || name == "_w_fold" {
                    return self.eval_loop_intrinsic(name, original, env);
                }

                // Check for tuple projection (first arg - returns PendingTupleProj)
                if name == "_w_tuple_proj" {
                    let arg_val = self.eval(arg, env);
                    return Value::PendingTupleProj(Box::new(arg_val));
                }

                // Check for index operation (first arg - returns PendingIndex)
                if name == "_w_index" {
                    let arg_val = self.eval(arg, env);
                    return Value::PendingIndex(Box::new(arg_val));
                }

                // Check for tuple construction
                if name == "_w_tuple" {
                    let arg_val = self.eval(arg, env);
                    return self.eval_tuple_construction(arg_val, original);
                }

                // Check for array/vector literals
                if name == "_w_array_lit" || name == "_w_vec_lit" {
                    let arg_val = self.eval(arg, env);
                    return self.eval_array_construction(name, arg_val, original);
                }

                let arg_val = self.eval(arg, env);

                // Try to find the definition and inline
                if let Some(def) = self.defs.get(name).cloned() {
                    let arity = self.count_lambda_params(&def.body);
                    if arity > 0 {
                        // It's a function - create/extend closure
                        Value::Closure {
                            def_name: name.clone(),
                            arity,
                            applied_args: vec![arg_val],
                        }
                    } else {
                        // Not a function - residualize
                        self.reify_app_var(name, arg_val, original)
                    }
                } else {
                    // Unknown function - residualize
                    self.reify_app_var(name, arg_val, original)
                }
            }

            FunctionName::Term(inner) => {
                let func_val = self.eval(inner, env);
                let arg_val = self.eval(arg, env);

                match func_val {
                    Value::PendingBinOp(op, lhs) => {
                        // Complete the binary operation
                        self.eval_binop(&op, *lhs, arg_val, original)
                    }

                    Value::PendingTupleProj(tuple) => {
                        // Complete tuple projection: _w_tuple_proj tuple index
                        self.eval_tuple_proj(*tuple, arg_val, original)
                    }

                    Value::PendingIndex(array) => {
                        // Complete index operation: _w_index array index
                        self.eval_index(*array, arg_val, original)
                    }

                    Value::Closure {
                        def_name,
                        arity,
                        mut applied_args,
                    } => {
                        applied_args.push(arg_val);
                        if applied_args.len() >= arity {
                            // Fully applied - try to inline
                            self.try_inline(&def_name, applied_args, original)
                        } else {
                            // Still partial
                            Value::Closure {
                                def_name,
                                arity,
                                applied_args,
                            }
                        }
                    }

                    _ => self.reify_app_term(func_val, arg_val, original),
                }
            }
        }
    }

    // =========================================================================
    // Binary operations
    // =========================================================================

    /// Evaluate a binary operation with algebraic simplifications.
    fn eval_binop(&mut self, op: &BinaryOp, lhs: Value, rhs: Value, original: &Term) -> Value {
        // Try constant folding
        if let (Value::Scalar(l), Value::Scalar(r)) = (&lhs, &rhs) {
            if let Some(result) = self.try_fold_binop(op, l, r) {
                return Value::Scalar(result);
            }
        }

        // Algebraic simplifications
        let op_str = op.op.as_str();
        match (op_str, &lhs, &rhs) {
            // x + 0 = x, 0 + x = x
            ("+", _, Value::Scalar(ScalarValue::Int(0))) => return lhs,
            ("+", Value::Scalar(ScalarValue::Int(0)), _) => return rhs,
            ("+", _, Value::Scalar(ScalarValue::Float(f))) if *f == 0.0 => return lhs,
            ("+", Value::Scalar(ScalarValue::Float(f)), _) if *f == 0.0 => return rhs,

            // x - 0 = x
            ("-", _, Value::Scalar(ScalarValue::Int(0))) => return lhs,
            ("-", _, Value::Scalar(ScalarValue::Float(f))) if *f == 0.0 => return lhs,

            // x * 1 = x, 1 * x = x
            ("*", _, Value::Scalar(ScalarValue::Int(1))) => return lhs,
            ("*", Value::Scalar(ScalarValue::Int(1)), _) => return rhs,
            ("*", _, Value::Scalar(ScalarValue::Float(f))) if *f == 1.0 => return lhs,
            ("*", Value::Scalar(ScalarValue::Float(f)), _) if *f == 1.0 => return rhs,

            // x * 0 = 0, 0 * x = 0
            ("*", _, Value::Scalar(ScalarValue::Int(0))) => return Value::Scalar(ScalarValue::Int(0)),
            ("*", Value::Scalar(ScalarValue::Int(0)), _) => return Value::Scalar(ScalarValue::Int(0)),
            ("*", _, Value::Scalar(ScalarValue::Float(f))) if *f == 0.0 => {
                return Value::Scalar(ScalarValue::Float(0.0));
            }
            ("*", Value::Scalar(ScalarValue::Float(f)), _) if *f == 0.0 => {
                return Value::Scalar(ScalarValue::Float(0.0));
            }

            // x / 1 = x
            ("/", _, Value::Scalar(ScalarValue::Int(1))) => return lhs,
            ("/", _, Value::Scalar(ScalarValue::Float(f))) if *f == 1.0 => return lhs,

            // Boolean identities
            ("&&", Value::Scalar(ScalarValue::Bool(true)), _) => return rhs,
            ("&&", _, Value::Scalar(ScalarValue::Bool(true))) => return lhs,
            ("&&", Value::Scalar(ScalarValue::Bool(false)), _) => {
                return Value::Scalar(ScalarValue::Bool(false));
            }
            ("&&", _, Value::Scalar(ScalarValue::Bool(false))) => {
                return Value::Scalar(ScalarValue::Bool(false));
            }

            ("||", Value::Scalar(ScalarValue::Bool(false)), _) => return rhs,
            ("||", _, Value::Scalar(ScalarValue::Bool(false))) => return lhs,
            ("||", Value::Scalar(ScalarValue::Bool(true)), _) => {
                return Value::Scalar(ScalarValue::Bool(true));
            }
            ("||", _, Value::Scalar(ScalarValue::Bool(true))) => {
                return Value::Scalar(ScalarValue::Bool(true));
            }

            _ => {}
        }

        // Can't simplify - residualize
        self.reify_binop(op, lhs, rhs, original)
    }

    /// Try to fold a binary operation on two scalar values.
    fn try_fold_binop(&self, op: &BinaryOp, lhs: &ScalarValue, rhs: &ScalarValue) -> Option<ScalarValue> {
        let op_str = op.op.as_str();

        match (lhs, rhs) {
            // Integer operations
            (ScalarValue::Int(l), ScalarValue::Int(r)) => match op_str {
                "+" => Some(ScalarValue::Int(l.wrapping_add(*r))),
                "-" => Some(ScalarValue::Int(l.wrapping_sub(*r))),
                "*" => Some(ScalarValue::Int(l.wrapping_mul(*r))),
                "/" if *r != 0 => Some(ScalarValue::Int(l.wrapping_div(*r))),
                "%" if *r != 0 => Some(ScalarValue::Int(l.wrapping_rem(*r))),
                "==" => Some(ScalarValue::Bool(l == r)),
                "!=" => Some(ScalarValue::Bool(l != r)),
                "<" => Some(ScalarValue::Bool(l < r)),
                "<=" => Some(ScalarValue::Bool(l <= r)),
                ">" => Some(ScalarValue::Bool(l > r)),
                ">=" => Some(ScalarValue::Bool(l >= r)),
                _ => None,
            },

            // Float operations
            (ScalarValue::Float(l), ScalarValue::Float(r)) => match op_str {
                "+" => Some(ScalarValue::Float(l + r)),
                "-" => Some(ScalarValue::Float(l - r)),
                "*" => Some(ScalarValue::Float(l * r)),
                "/" if *r != 0.0 => Some(ScalarValue::Float(l / r)),
                "==" => Some(ScalarValue::Bool(l == r)),
                "!=" => Some(ScalarValue::Bool(l != r)),
                "<" => Some(ScalarValue::Bool(l < r)),
                "<=" => Some(ScalarValue::Bool(l <= r)),
                ">" => Some(ScalarValue::Bool(l > r)),
                ">=" => Some(ScalarValue::Bool(l >= r)),
                _ => None,
            },

            // Boolean operations
            (ScalarValue::Bool(l), ScalarValue::Bool(r)) => match op_str {
                "&&" => Some(ScalarValue::Bool(*l && *r)),
                "||" => Some(ScalarValue::Bool(*l || *r)),
                "==" => Some(ScalarValue::Bool(l == r)),
                "!=" => Some(ScalarValue::Bool(l != r)),
                _ => None,
            },

            _ => None,
        }
    }

    // =========================================================================
    // Unary operations
    // =========================================================================

    /// Evaluate a unary operation.
    fn eval_unop(&mut self, op: &UnaryOp, arg: Value, original: &Term) -> Value {
        if let Value::Scalar(s) = &arg {
            if let Some(result) = self.try_fold_unop(op, s) {
                return Value::Scalar(result);
            }
        }

        // Can't fold - residualize
        self.reify_unop(op, arg, original)
    }

    /// Try to fold a unary operation.
    fn try_fold_unop(&self, op: &UnaryOp, arg: &ScalarValue) -> Option<ScalarValue> {
        match (op.op.as_str(), arg) {
            ("-", ScalarValue::Int(n)) => Some(ScalarValue::Int(-n)),
            ("-", ScalarValue::Float(f)) => Some(ScalarValue::Float(-f)),
            ("!", ScalarValue::Bool(b)) => Some(ScalarValue::Bool(!b)),
            _ => None,
        }
    }

    // =========================================================================
    // Tuple projection and indexing
    // =========================================================================

    /// Evaluate tuple projection: _w_tuple_proj tuple index
    fn eval_tuple_proj(&mut self, tuple: Value, index: Value, original: &Term) -> Value {
        // Try to get the index as a constant
        let idx = match &index {
            Value::Scalar(ScalarValue::Int(n)) => Some(*n as usize),
            _ => None,
        };

        match (tuple, idx) {
            (
                Value::Aggregate {
                    kind: AggregateKind::Tuple,
                    elements,
                },
                Some(i),
            ) => {
                if i < elements.len() {
                    elements.into_iter().nth(i).unwrap()
                } else {
                    Value::Unknown(original.clone())
                }
            }
            (tuple, _) => {
                // Can't project - residualize
                self.reify_tuple_proj(tuple, index, original)
            }
        }
    }

    /// Evaluate index operation: _w_index array index
    fn eval_index(&mut self, array: Value, index: Value, original: &Term) -> Value {
        // Try to get the index as a constant
        let idx = match &index {
            Value::Scalar(ScalarValue::Int(n)) => Some(*n as usize),
            _ => None,
        };

        match (&array, idx) {
            (
                Value::Aggregate {
                    kind: AggregateKind::Array | AggregateKind::Vector,
                    elements,
                },
                Some(i),
            ) => {
                if i < elements.len() {
                    elements[i].clone()
                } else {
                    Value::Unknown(original.clone())
                }
            }
            _ => {
                // Can't index - residualize
                self.reify_index(array, index, original)
            }
        }
    }

    /// Evaluate tuple construction.
    fn eval_tuple_construction(&mut self, _arg: Value, original: &Term) -> Value {
        // _w_tuple is applied to each element in sequence
        // For now, just residualize
        Value::Unknown(original.clone())
    }

    /// Evaluate array/vector construction.
    fn eval_array_construction(&mut self, _name: &str, _arg: Value, original: &Term) -> Value {
        // For now, just residualize
        Value::Unknown(original.clone())
    }

    // =========================================================================
    // Loop intrinsics
    // =========================================================================

    /// Evaluate a loop intrinsic.
    fn eval_loop_intrinsic(&mut self, _name: &str, original: &Term, env: &mut Env) -> Value {
        // Collect all curried arguments
        let (intrinsic_name, args) = self.collect_curried_args(original);

        match intrinsic_name.as_str() {
            "_w_loop_for" => self.eval_for_loop(&args, env, original),
            "_w_fold" => self.eval_fold(&args, env, original),
            "_w_loop_while" => {
                // While loops are hard to unroll - always residualize
                Value::Unknown(original.clone())
            }
            _ => Value::Unknown(original.clone()),
        }
    }

    /// Collect curried arguments from nested applications.
    fn collect_curried_args(&self, term: &Term) -> (String, Vec<Term>) {
        let mut args = Vec::new();
        let mut current = term;

        loop {
            match &current.kind {
                TermKind::App { func, arg } => {
                    args.push((**arg).clone());
                    match &**func {
                        FunctionName::Var(name) => {
                            args.reverse();
                            return (name.clone(), args);
                        }
                        FunctionName::Term(inner) => {
                            current = inner;
                        }
                        _ => {
                            args.reverse();
                            return (String::new(), args);
                        }
                    }
                }
                TermKind::Var(name) => {
                    args.reverse();
                    return (name.clone(), args);
                }
                _ => {
                    args.reverse();
                    return (String::new(), args);
                }
            }
        }
    }

    /// Evaluate a for loop.
    fn eval_for_loop(&mut self, _args: &[Term], _env: &mut Env, original: &Term) -> Value {
        // _w_loop_for args vary:
        // - _w_loop_for bound body_lambda
        // - _w_loop_for init bound body_lambda
        // For now, always residualize (loop unrolling is complex)
        Value::Unknown(original.clone())
    }

    /// Evaluate a fold.
    fn eval_fold(&mut self, _args: &[Term], _env: &mut Env, original: &Term) -> Value {
        // _w_fold body_lambda init iterable
        // For now, always residualize
        Value::Unknown(original.clone())
    }

    // =========================================================================
    // Function inlining
    // =========================================================================

    /// Try to inline a fully-applied function.
    fn try_inline(&mut self, def_name: &str, args: Vec<Value>, original: &Term) -> Value {
        // Check inline budget
        if self.budget.current_depth >= self.budget.max_inline_depth {
            return self.reify_closure_app(def_name, args, original);
        }

        // Check if all args are known
        if !args.iter().all(|a| a.is_known()) {
            return self.reify_closure_app(def_name, args, original);
        }

        // Get the definition
        let def = match self.defs.get(def_name).cloned() {
            Some(d) => d,
            None => return self.reify_closure_app(def_name, args, original),
        };

        // Extract lambda parameters and body
        let (params, body) = self.extract_lambda_params(&def.body);

        if params.len() != args.len() {
            return self.reify_closure_app(def_name, args, original);
        }

        // Create new environment with args bound to params
        self.budget.current_depth += 1;
        let mut inline_env = Env::new();
        for (param, arg) in params.into_iter().zip(args) {
            inline_env.extend(param, arg);
        }

        // Evaluate the body
        let result = self.eval(body, &mut inline_env);
        self.budget.current_depth -= 1;

        result
    }

    /// Extract lambda parameters from a term.
    fn extract_lambda_params<'a>(&self, term: &'a Term) -> (Vec<String>, &'a Term) {
        match &term.kind {
            TermKind::Lam { param, body, .. } => {
                let (mut params, inner) = self.extract_lambda_params(body);
                params.insert(0, param.clone());
                (params, inner)
            }
            _ => (vec![], term),
        }
    }

    // =========================================================================
    // Reification (Value -> Term)
    // =========================================================================

    /// Convert a Value back to a Term.
    fn reify(&mut self, value: Value, ty: &Type<TypeName>, span: Span) -> Term {
        match value {
            Value::Scalar(ScalarValue::Int(n)) => Term {
                id: self.fresh_id(),
                ty: ty.clone(),
                span,
                kind: TermKind::IntLit(n.to_string()),
            },

            Value::Scalar(ScalarValue::Float(f)) => Term {
                id: self.fresh_id(),
                ty: ty.clone(),
                span,
                kind: TermKind::FloatLit(f as f32),
            },

            Value::Scalar(ScalarValue::Bool(b)) => Term {
                id: self.fresh_id(),
                ty: ty.clone(),
                span,
                kind: TermKind::BoolLit(b),
            },

            Value::Scalar(ScalarValue::String(s)) => Term {
                id: self.fresh_id(),
                ty: ty.clone(),
                span,
                kind: TermKind::StringLit(s),
            },

            Value::Unit => {
                // Unit is typically represented as empty tuple
                Term {
                    id: self.fresh_id(),
                    ty: ty.clone(),
                    span,
                    kind: TermKind::Var("()".to_string()),
                }
            }

            Value::Unknown(term) => term,

            Value::Aggregate { kind, elements } => self.reify_aggregate(kind, elements, ty, span),

            Value::PendingBinOp(op, lhs) => {
                // Residualize as partial application
                let lhs_term = self.reify(*lhs, ty, span);
                Term {
                    id: self.fresh_id(),
                    ty: ty.clone(),
                    span,
                    kind: TermKind::App {
                        func: Box::new(FunctionName::BinOp(op)),
                        arg: Box::new(lhs_term),
                    },
                }
            }

            Value::Closure {
                def_name,
                applied_args,
                ..
            } => self.reify_partial_closure(&def_name, applied_args, ty, span),

            Value::PendingTupleProj(tuple) => {
                // Residualize as partial application: _w_tuple_proj tuple
                let tuple_ty = ty.clone(); // TODO: extract proper type
                let tuple_term = self.reify(*tuple, &tuple_ty, span);
                Term {
                    id: self.fresh_id(),
                    ty: ty.clone(),
                    span,
                    kind: TermKind::App {
                        func: Box::new(FunctionName::Var("_w_tuple_proj".to_string())),
                        arg: Box::new(tuple_term),
                    },
                }
            }

            Value::PendingIndex(array) => {
                // Residualize as partial application: _w_index array
                let array_ty = ty.clone(); // TODO: extract proper type
                let array_term = self.reify(*array, &array_ty, span);
                Term {
                    id: self.fresh_id(),
                    ty: ty.clone(),
                    span,
                    kind: TermKind::App {
                        func: Box::new(FunctionName::Var("_w_index".to_string())),
                        arg: Box::new(array_term),
                    },
                }
            }
        }
    }

    /// Reify an aggregate value.
    fn reify_aggregate(
        &mut self,
        kind: AggregateKind,
        elements: Vec<Value>,
        ty: &Type<TypeName>,
        span: Span,
    ) -> Term {
        // For now, use intrinsic calls to construct aggregates
        let intrinsic = match kind {
            AggregateKind::Tuple => "_w_tuple",
            AggregateKind::Array => "_w_array_lit",
            AggregateKind::Vector => "_w_vec_lit",
            AggregateKind::Matrix => "_w_matrix_lit",
        };

        // Build curried application: intrinsic elem0 elem1 elem2 ...
        let unit_ty = Type::Constructed(TypeName::Str("unit"), vec![]);
        let mut result = Term {
            id: self.fresh_id(),
            ty: ty.clone(),
            span,
            kind: TermKind::Var(intrinsic.to_string()),
        };

        for elem in elements {
            let elem_term = self.reify(elem, &unit_ty, span);
            result = Term {
                id: self.fresh_id(),
                ty: ty.clone(),
                span,
                kind: TermKind::App {
                    func: Box::new(FunctionName::Term(Box::new(result))),
                    arg: Box::new(elem_term),
                },
            };
        }

        result
    }

    /// Reify a partial closure application.
    fn reify_partial_closure(
        &mut self,
        def_name: &str,
        args: Vec<Value>,
        ty: &Type<TypeName>,
        span: Span,
    ) -> Term {
        let unit_ty = Type::Constructed(TypeName::Str("unit"), vec![]);
        let mut result = Term {
            id: self.fresh_id(),
            ty: ty.clone(),
            span,
            kind: TermKind::Var(def_name.to_string()),
        };

        for arg in args {
            let arg_term = self.reify(arg, &unit_ty, span);
            result = Term {
                id: self.fresh_id(),
                ty: ty.clone(),
                span,
                kind: TermKind::App {
                    func: Box::new(FunctionName::Term(Box::new(result))),
                    arg: Box::new(arg_term),
                },
            };
        }

        result
    }

    /// Reify an if expression.
    fn reify_if(&mut self, cond: Value, then_val: Value, else_val: Value, original: &Term) -> Value {
        let cond_term = self.reify(cond, &original.ty, original.span);
        let then_term = self.reify(then_val, &original.ty, original.span);
        let else_term = self.reify(else_val, &original.ty, original.span);

        Value::Unknown(Term {
            id: self.fresh_id(),
            ty: original.ty.clone(),
            span: original.span,
            kind: TermKind::If {
                cond: Box::new(cond_term),
                then_branch: Box::new(then_term),
                else_branch: Box::new(else_term),
            },
        })
    }

    /// Reify a binary operation.
    fn reify_binop(&mut self, op: &BinaryOp, lhs: Value, rhs: Value, original: &Term) -> Value {
        let unit_ty = Type::Constructed(TypeName::Str("unit"), vec![]);
        let lhs_term = self.reify(lhs, &unit_ty, original.span);
        let rhs_term = self.reify(rhs, &unit_ty, original.span);

        // Build ((op lhs) rhs)
        let partial = Term {
            id: self.fresh_id(),
            ty: original.ty.clone(),
            span: original.span,
            kind: TermKind::App {
                func: Box::new(FunctionName::BinOp(op.clone())),
                arg: Box::new(lhs_term),
            },
        };

        Value::Unknown(Term {
            id: self.fresh_id(),
            ty: original.ty.clone(),
            span: original.span,
            kind: TermKind::App {
                func: Box::new(FunctionName::Term(Box::new(partial))),
                arg: Box::new(rhs_term),
            },
        })
    }

    /// Reify a unary operation.
    fn reify_unop(&mut self, op: &UnaryOp, arg: Value, original: &Term) -> Value {
        let unit_ty = Type::Constructed(TypeName::Str("unit"), vec![]);
        let arg_term = self.reify(arg, &unit_ty, original.span);

        Value::Unknown(Term {
            id: self.fresh_id(),
            ty: original.ty.clone(),
            span: original.span,
            kind: TermKind::App {
                func: Box::new(FunctionName::UnOp(op.clone())),
                arg: Box::new(arg_term),
            },
        })
    }

    /// Reify application with a variable function.
    fn reify_app_var(&mut self, name: &str, arg: Value, original: &Term) -> Value {
        let unit_ty = Type::Constructed(TypeName::Str("unit"), vec![]);
        let arg_term = self.reify(arg, &unit_ty, original.span);

        Value::Unknown(Term {
            id: self.fresh_id(),
            ty: original.ty.clone(),
            span: original.span,
            kind: TermKind::App {
                func: Box::new(FunctionName::Var(name.to_string())),
                arg: Box::new(arg_term),
            },
        })
    }

    /// Reify application with a term function.
    fn reify_app_term(&mut self, func: Value, arg: Value, original: &Term) -> Value {
        let unit_ty = Type::Constructed(TypeName::Str("unit"), vec![]);
        let func_term = self.reify(func, &unit_ty, original.span);
        let arg_term = self.reify(arg, &unit_ty, original.span);

        Value::Unknown(Term {
            id: self.fresh_id(),
            ty: original.ty.clone(),
            span: original.span,
            kind: TermKind::App {
                func: Box::new(FunctionName::Term(Box::new(func_term))),
                arg: Box::new(arg_term),
            },
        })
    }

    /// Reify a closure application.
    fn reify_closure_app(&mut self, def_name: &str, args: Vec<Value>, original: &Term) -> Value {
        let unit_ty = Type::Constructed(TypeName::Str("unit"), vec![]);

        let mut result = Term {
            id: self.fresh_id(),
            ty: original.ty.clone(),
            span: original.span,
            kind: TermKind::Var(def_name.to_string()),
        };

        for arg in args {
            let arg_term = self.reify(arg, &unit_ty, original.span);
            result = Term {
                id: self.fresh_id(),
                ty: original.ty.clone(),
                span: original.span,
                kind: TermKind::App {
                    func: Box::new(FunctionName::Term(Box::new(result))),
                    arg: Box::new(arg_term),
                },
            };
        }

        Value::Unknown(result)
    }

    /// Reify a tuple projection: _w_tuple_proj tuple index
    fn reify_tuple_proj(&mut self, tuple: Value, index: Value, original: &Term) -> Value {
        let unit_ty = Type::Constructed(TypeName::Str("unit"), vec![]);
        let tuple_term = self.reify(tuple, &unit_ty, original.span);
        let index_term = self.reify(
            index,
            &Type::Constructed(TypeName::Int(32), vec![]),
            original.span,
        );

        // Build: (_w_tuple_proj tuple) index
        let partial = Term {
            id: self.fresh_id(),
            ty: original.ty.clone(),
            span: original.span,
            kind: TermKind::App {
                func: Box::new(FunctionName::Var("_w_tuple_proj".to_string())),
                arg: Box::new(tuple_term),
            },
        };

        Value::Unknown(Term {
            id: self.fresh_id(),
            ty: original.ty.clone(),
            span: original.span,
            kind: TermKind::App {
                func: Box::new(FunctionName::Term(Box::new(partial))),
                arg: Box::new(index_term),
            },
        })
    }

    /// Reify an index operation: _w_index array index
    fn reify_index(&mut self, array: Value, index: Value, original: &Term) -> Value {
        let unit_ty = Type::Constructed(TypeName::Str("unit"), vec![]);
        let array_term = self.reify(array, &unit_ty, original.span);
        let index_term = self.reify(
            index,
            &Type::Constructed(TypeName::Int(32), vec![]),
            original.span,
        );

        // Build: (_w_index array) index
        let partial = Term {
            id: self.fresh_id(),
            ty: original.ty.clone(),
            span: original.span,
            kind: TermKind::App {
                func: Box::new(FunctionName::Var("_w_index".to_string())),
                arg: Box::new(array_term),
            },
        };

        Value::Unknown(Term {
            id: self.fresh_id(),
            ty: original.ty.clone(),
            span: original.span,
            kind: TermKind::App {
                func: Box::new(FunctionName::Term(Box::new(partial))),
                arg: Box::new(index_term),
            },
        })
    }
}
