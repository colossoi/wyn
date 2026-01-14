//! Desugaring pass for Range and Slice expressions.
//!
//! This pass transforms:
//! - Range expressions (`0..<n`, `a..b...c`, etc.) into `iota` and `map` calls
//! - Slice expressions (`a[i:j:s]`) into `map` and `iota` calls
//!
//! Must run after type checking and before flattening.

use crate::ast::{
    BinaryOp, Decl, Declaration, EntryDecl, ExprKind, Expression, LambdaExpr, NodeCounter, Pattern,
    PatternKind, Program, RangeExpr, RangeKind, SliceExpr, Span,
};
use crate::err_flatten_at;
use crate::error::Result;

/// Desugars range and slice expressions into map/iota constructs.
pub struct Desugarer<'a> {
    nc: &'a mut NodeCounter,
}

impl<'a> Desugarer<'a> {
    pub fn new(nc: &'a mut NodeCounter) -> Self {
        Self { nc }
    }

    /// Desugar an entire program.
    pub fn desugar_program(&mut self, program: &mut Program) -> Result<()> {
        for decl in &mut program.declarations {
            self.desugar_declaration(decl)?;
        }
        Ok(())
    }

    fn desugar_declaration(&mut self, decl: &mut Declaration) -> Result<()> {
        match decl {
            Declaration::Decl(d) => self.desugar_decl(d),
            Declaration::Entry(e) => self.desugar_entry_decl(e),
            _ => Ok(()),
        }
    }

    fn desugar_decl(&mut self, d: &mut Decl) -> Result<()> {
        self.desugar_expr(&mut d.body)
    }

    fn desugar_entry_decl(&mut self, e: &mut EntryDecl) -> Result<()> {
        self.desugar_expr(&mut e.body)
    }

    /// Recursively desugar an expression, transforming Range and Slice nodes.
    pub fn desugar_expr(&mut self, expr: &mut Expression) -> Result<()> {
        // First, recursively desugar sub-expressions
        match &mut expr.kind {
            ExprKind::ArrayLiteral(elements) | ExprKind::VecMatLiteral(elements) => {
                for elem in elements {
                    self.desugar_expr(elem)?;
                }
            }
            ExprKind::ArrayIndex(array, index) => {
                self.desugar_expr(array)?;
                self.desugar_expr(index)?;
            }
            ExprKind::ArrayWith { array, index, value } => {
                self.desugar_expr(array)?;
                self.desugar_expr(index)?;
                self.desugar_expr(value)?;
            }
            ExprKind::BinaryOp(_, left, right) => {
                self.desugar_expr(left)?;
                self.desugar_expr(right)?;
            }
            ExprKind::UnaryOp(_, operand) => {
                self.desugar_expr(operand)?;
            }
            ExprKind::Tuple(elements) => {
                for elem in elements {
                    self.desugar_expr(elem)?;
                }
            }
            ExprKind::RecordLiteral(fields) => {
                for (_, field_expr) in fields {
                    self.desugar_expr(field_expr)?;
                }
            }
            ExprKind::Lambda(lambda) => {
                self.desugar_expr(&mut lambda.body)?;
            }
            ExprKind::Application(func, args) => {
                self.desugar_expr(func)?;
                for arg in args {
                    self.desugar_expr(arg)?;
                }
                // Note: map, reduce, etc. are prelude functions that call _w_intrinsic_* intrinsics
            }
            ExprKind::LetIn(let_in) => {
                self.desugar_expr(&mut let_in.value)?;
                self.desugar_expr(&mut let_in.body)?;
            }
            ExprKind::FieldAccess(obj, _) => {
                self.desugar_expr(obj)?;
            }
            ExprKind::If(if_expr) => {
                self.desugar_expr(&mut if_expr.condition)?;
                self.desugar_expr(&mut if_expr.then_branch)?;
                self.desugar_expr(&mut if_expr.else_branch)?;
            }
            ExprKind::Loop(loop_expr) => {
                if let Some(init) = &mut loop_expr.init {
                    self.desugar_expr(init)?;
                }
                self.desugar_expr(&mut loop_expr.body)?;
            }
            ExprKind::Match(match_expr) => {
                self.desugar_expr(&mut match_expr.scrutinee)?;
                for case in &mut match_expr.cases {
                    self.desugar_expr(&mut case.body)?;
                }
            }
            ExprKind::TypeAscription(inner, _) | ExprKind::TypeCoercion(inner, _) => {
                self.desugar_expr(inner)?;
            }
            // Leaf expressions - nothing to desugar inside
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::StringLiteral(_)
            | ExprKind::Unit
            | ExprKind::Identifier(_, _)
            | ExprKind::TypeHole => {}
            // Range and Slice are handled below after recursion
            ExprKind::Range(_) | ExprKind::Slice(_) => {}
        }

        // Slice expressions are now handled in flattening as BorrowedSlice
        // Range stays as-is in the AST

        Ok(())
    }

    // NOTE: desugar_slice is no longer used - slices are now handled in flattening
    // as BorrowedSlice. This code is kept for reference but should be deleted.
    #[allow(dead_code)]
    fn desugar_slice(&mut self, slice: &SliceExpr, span: Span) -> Result<Expression> {
        let array = &slice.array;

        // Handle a[:] - full array copy (identity)
        if slice.start.is_none() && slice.end.is_none() {
            return Ok((**array).clone());
        }

        // For now, require explicit start and end for slices
        let start = slice.start.as_ref().ok_or_else(|| {
            err_flatten_at!(
                span,
                "Slice without explicit start requires array length (not yet supported)"
            )
        })?;

        let end = slice.end.as_ref().ok_or_else(|| {
            err_flatten_at!(
                span,
                "Slice without explicit end requires array length (not yet supported)"
            )
        })?;

        // Calculate count = end - start
        let diff = self.mk_binop("-", (**end).clone(), (**start).clone(), span);

        // Build lambda: |idx| array[start + idx]
        let idx_param = self.mk_pattern_name("__idx", span);
        let idx_var = self.mk_ident("__idx", span);
        let index_expr = self.mk_binop("+", (**start).clone(), idx_var, span);

        let array_access = self.mk_array_index((**array).clone(), index_expr, span);
        let lambda = self.mk_lambda(vec![idx_param], array_access, span);

        // Create range 0..<count
        let range_expr = self.mk_range(diff, span);

        Ok(self.mk_call("map", vec![lambda, range_expr], span))
    }

    /// Create a range expression: 0..<n
    fn mk_range(&mut self, end: Expression, span: Span) -> Expression {
        let zero = self.mk_int(0, span);
        self.nc.mk_node(
            ExprKind::Range(RangeExpr {
                start: Box::new(zero),
                step: None,
                end: Box::new(end),
                kind: RangeKind::ExclusiveLt,
            }),
            span,
        )
    }

    // Helper methods to construct AST nodes

    fn mk_int(&mut self, n: i32, span: Span) -> Expression {
        self.nc.mk_node(ExprKind::IntLiteral(n.to_string().into()), span)
    }

    fn mk_ident(&mut self, name: &str, span: Span) -> Expression {
        self.nc.mk_node(ExprKind::Identifier(vec![], name.to_string()), span)
    }

    fn mk_binop(&mut self, op: &str, left: Expression, right: Expression, span: Span) -> Expression {
        self.nc.mk_node(
            ExprKind::BinaryOp(BinaryOp { op: op.to_string() }, Box::new(left), Box::new(right)),
            span,
        )
    }

    fn mk_call(&mut self, func_name: &str, args: Vec<Expression>, span: Span) -> Expression {
        let func = self.mk_ident(func_name, span);
        self.nc.mk_node(ExprKind::Application(Box::new(func), args), span)
    }

    fn mk_lambda(&mut self, params: Vec<Pattern>, body: Expression, span: Span) -> Expression {
        self.nc.mk_node(
            ExprKind::Lambda(LambdaExpr {
                params,
                body: Box::new(body),
            }),
            span,
        )
    }

    fn mk_pattern_name(&mut self, name: &str, span: Span) -> Pattern {
        self.nc.mk_node(PatternKind::Name(name.to_string()), span)
    }

    fn mk_array_index(&mut self, array: Expression, index: Expression, span: Span) -> Expression {
        self.nc.mk_node(ExprKind::ArrayIndex(Box::new(array), Box::new(index)), span)
    }
}

/// Top-level function to desugar a program.
pub fn desugar_program(program: &mut Program, nc: &mut NodeCounter) -> Result<()> {
    let mut desugarer = Desugarer::new(nc);
    desugarer.desugar_program(program)
}
