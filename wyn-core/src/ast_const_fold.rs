//! AST-level constant folding and inlining for integer constants.
//!
//! This pass runs before flattening to ensure downstream passes
//! can see constant values. It handles:
//! - Folding: `2 + 7` → `9`
//! - Inlining: `def C = 9; ... C ...` → `def C = 9; ... 9 ...`
//!
//! This is intentionally limited to integer constants for simplicity.
//! Float and boolean constants can be added later if needed.

use crate::ast::{
    Decl, Declaration, EntryDecl, ExprKind, Expression, IfExpr, LetInExpr, LoopExpr, LoopForm, MatchExpr,
    Program, RangeExpr,
};
use std::collections::HashMap;

/// AST-level constant folder for integer constants.
pub struct AstConstFolder {
    /// Known integer constants: name → value
    pub(crate) constants: HashMap<String, i64>,
}

impl AstConstFolder {
    /// Add a constant for testing purposes
    #[cfg(test)]
    pub fn add_constant(&mut self, name: &str, value: i64) {
        self.constants.insert(name.to_string(), value);
    }
}

impl Default for AstConstFolder {
    fn default() -> Self {
        Self::new()
    }
}

impl AstConstFolder {
    pub fn new() -> Self {
        Self {
            constants: HashMap::new(),
        }
    }

    /// Fold constants in an entire program.
    ///
    /// Two passes:
    /// 1. Collect top-level constant definitions (parameterless defs with integer values)
    /// 2. Fold and inline in all expressions
    pub fn fold_program(&mut self, program: &mut Program) {
        // First pass: collect top-level constant definitions
        for decl in &program.declarations {
            if let Declaration::Decl(d) = decl {
                // Only parameterless definitions can be constants
                if d.params.is_empty() && d.size_params.is_empty() && d.type_params.is_empty() {
                    if let Some(val) = self.try_eval_const(&d.body) {
                        self.constants.insert(d.name.clone(), val);
                    }
                }
            }
        }

        // Second pass: fold and inline in all expressions
        for decl in &mut program.declarations {
            self.fold_declaration(decl);
        }
    }

    fn fold_declaration(&mut self, decl: &mut Declaration) {
        match decl {
            Declaration::Decl(d) => self.fold_decl(d),
            Declaration::Entry(e) => self.fold_entry_decl(e),
            // Uniform, Sig, TypeBind, ModuleBind, etc. don't contain expressions to fold
            _ => {}
        }
    }

    fn fold_decl(&mut self, d: &mut Decl) {
        self.fold_expr(&mut d.body);
    }

    fn fold_entry_decl(&mut self, e: &mut EntryDecl) {
        self.fold_expr(&mut e.body);
    }

    /// Recursively fold constants in an expression.
    /// Modifies the expression in place.
    pub fn fold_expr(&mut self, expr: &mut Expression) {
        match &mut expr.kind {
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::StringLiteral(_)
            | ExprKind::Unit
            | ExprKind::TypeHole => {
                // Leaf nodes, nothing to fold
            }

            ExprKind::Identifier(quals, name) => {
                // Inline known constants (only for unqualified names)
                if quals.is_empty() {
                    if let Some(&val) = self.constants.get(name) {
                        expr.kind = ExprKind::IntLiteral(val as i32);
                    }
                }
            }

            ExprKind::BinaryOp(op, lhs, rhs) => {
                self.fold_expr(lhs);
                self.fold_expr(rhs);
                // Try to fold after children are folded
                if let Some(val) = self.try_fold_binop(&op.op, lhs, rhs) {
                    expr.kind = ExprKind::IntLiteral(val as i32);
                }
            }

            ExprKind::UnaryOp(op, operand) => {
                self.fold_expr(operand);
                // Try to fold after child is folded
                if let Some(val) = self.try_fold_unaryop(&op.op, operand) {
                    expr.kind = ExprKind::IntLiteral(val as i32);
                }
            }

            ExprKind::ArrayLiteral(elements) | ExprKind::VecMatLiteral(elements) => {
                for elem in elements {
                    self.fold_expr(elem);
                }
            }

            ExprKind::ArrayIndex(arr, idx) => {
                self.fold_expr(arr);
                self.fold_expr(idx);
            }

            ExprKind::ArrayWith { array, index, value } => {
                self.fold_expr(array);
                self.fold_expr(index);
                self.fold_expr(value);
            }

            ExprKind::Tuple(elements) => {
                for elem in elements {
                    self.fold_expr(elem);
                }
            }

            ExprKind::RecordLiteral(fields) => {
                for (_name, value) in fields {
                    self.fold_expr(value);
                }
            }

            ExprKind::Lambda(lambda) => {
                self.fold_expr(&mut lambda.body);
            }

            ExprKind::Application(func, args) => {
                self.fold_expr(func);
                for arg in args {
                    self.fold_expr(arg);
                }
            }

            ExprKind::LetIn(let_in) => {
                self.fold_let_in(let_in);
            }

            ExprKind::FieldAccess(obj, _field) => {
                self.fold_expr(obj);
            }

            ExprKind::If(if_expr) => {
                self.fold_if(if_expr);
            }

            ExprKind::Loop(loop_expr) => {
                self.fold_loop(loop_expr);
            }

            ExprKind::Match(match_expr) => {
                self.fold_match(match_expr);
            }

            ExprKind::Range(range) => {
                self.fold_range(range);
            }

            ExprKind::Slice(slice) => {
                self.fold_expr(&mut slice.array);
                if let Some(start) = &mut slice.start {
                    self.fold_expr(start);
                }
                if let Some(end) = &mut slice.end {
                    self.fold_expr(end);
                }
            }

            ExprKind::TypeAscription(inner, _) | ExprKind::TypeCoercion(inner, _) => {
                self.fold_expr(inner);
            }

            ExprKind::Assert(cond, body) => {
                self.fold_expr(cond);
                self.fold_expr(body);
            }
        }
    }

    fn fold_let_in(&mut self, let_in: &mut LetInExpr) {
        // Fold the value first
        self.fold_expr(&mut let_in.value);

        // Check if this introduces a constant
        // For simplicity, only handle simple name patterns
        let const_binding = if let crate::ast::PatternKind::Name(name) = &let_in.pattern.kind {
            if let Some(val) = self.try_eval_const(&let_in.value) {
                Some((name.clone(), val))
            } else {
                None
            }
        } else {
            None
        };

        // If we found a constant, temporarily add it to scope
        if let Some((name, val)) = &const_binding {
            self.constants.insert(name.clone(), *val);
        }

        // Fold the body
        self.fold_expr(&mut let_in.body);

        // Remove the temporary binding (it's scoped to this let)
        if let Some((name, _)) = const_binding {
            self.constants.remove(&name);
        }
    }

    fn fold_if(&mut self, if_expr: &mut IfExpr) {
        self.fold_expr(&mut if_expr.condition);
        self.fold_expr(&mut if_expr.then_branch);
        self.fold_expr(&mut if_expr.else_branch);
    }

    fn fold_loop(&mut self, loop_expr: &mut LoopExpr) {
        if let Some(init) = &mut loop_expr.init {
            self.fold_expr(init);
        }
        match &mut loop_expr.form {
            LoopForm::For(_var, bound) => {
                self.fold_expr(bound);
            }
            LoopForm::ForIn(_pat, iter) => {
                self.fold_expr(iter);
            }
            LoopForm::While(cond) => {
                self.fold_expr(cond);
            }
        }
        self.fold_expr(&mut loop_expr.body);
    }

    fn fold_match(&mut self, match_expr: &mut MatchExpr) {
        self.fold_expr(&mut match_expr.scrutinee);
        for case in &mut match_expr.cases {
            self.fold_expr(&mut case.body);
        }
    }

    fn fold_range(&mut self, range: &mut RangeExpr) {
        self.fold_expr(&mut range.start);
        self.fold_expr(&mut range.end);
    }

    /// Try to evaluate an expression as a constant integer.
    /// Returns None if the expression is not a constant integer.
    fn try_eval_const(&self, expr: &Expression) -> Option<i64> {
        match &expr.kind {
            ExprKind::IntLiteral(n) => Some(*n as i64),
            ExprKind::Identifier(quals, name) if quals.is_empty() => self.constants.get(name).copied(),
            ExprKind::BinaryOp(op, lhs, rhs) => {
                let l = self.try_eval_const(lhs)?;
                let r = self.try_eval_const(rhs)?;
                self.eval_binop(&op.op, l, r)
            }
            ExprKind::UnaryOp(op, operand) => {
                let v = self.try_eval_const(operand)?;
                self.eval_unaryop(&op.op, v)
            }
            ExprKind::TypeAscription(inner, _) => self.try_eval_const(inner),
            _ => None,
        }
    }

    fn try_fold_binop(&self, op: &str, lhs: &Expression, rhs: &Expression) -> Option<i64> {
        let l = self.try_eval_const(lhs)?;
        let r = self.try_eval_const(rhs)?;
        self.eval_binop(op, l, r)
    }

    fn try_fold_unaryop(&self, op: &str, operand: &Expression) -> Option<i64> {
        let v = self.try_eval_const(operand)?;
        self.eval_unaryop(op, v)
    }

    fn eval_binop(&self, op: &str, l: i64, r: i64) -> Option<i64> {
        match op {
            "+" => Some(l.wrapping_add(r)),
            "-" => Some(l.wrapping_sub(r)),
            "*" => Some(l.wrapping_mul(r)),
            "/" if r != 0 => Some(l / r),
            "%" if r != 0 => Some(l % r),
            _ => None,
        }
    }

    fn eval_unaryop(&self, op: &str, v: i64) -> Option<i64> {
        match op {
            "-" => Some(-v),
            _ => None,
        }
    }
}

/// Convenience function to fold constants in a program.
pub fn fold_ast_constants(program: &mut Program) {
    let mut folder = AstConstFolder::new();
    folder.fold_program(program);
}
