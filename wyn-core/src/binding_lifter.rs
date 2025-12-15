//! Binding lifting (code motion) pass for MIR.
//!
//! This pass hoists loop-invariant bindings out of loops to reduce
//! redundant computation. A binding is loop-invariant if its value
//! depends only on variables defined outside the loop.
//!
//! ## Example Transformation
//!
//! ```text
//! // Before:
//! loop acc = 0 for i < n do
//!     let x = expensive_constant in
//!     let y = acc + x in
//!     y
//!
//! // After:
//! let x = expensive_constant in
//! loop acc = 0 for i < n do
//!     let y = acc + x in
//!     y
//! ```

use std::collections::HashSet;

use crate::ast::Span;
use crate::error::Result;
use crate::mir::{Def, Expr, ExprKind, Literal, LoopKind, Program};

/// A single binding in linear form, extracted from nested Let chains.
struct LinearBinding {
    /// The NodeId of the original Let expression
    id: crate::ast::NodeId,
    name: String,
    binding_id: u64,
    value: Expr,
    /// Set of free variables in the value expression.
    free_vars: HashSet<String>,
    span: Span,
}

/// Linearized representation of a Let chain.
struct LinearizedBody {
    /// Bindings in topological order (dependencies before uses).
    bindings: Vec<LinearBinding>,
    /// The final result expression (non-Let).
    result: Expr,
}

/// Binding lifter pass for hoisting loop-invariant bindings.
pub struct BindingLifter {}

impl BindingLifter {
    pub fn new() -> Self {
        BindingLifter {}
    }

    /// Lift bindings in all definitions in a program.
    pub fn lift_program(&mut self, program: Program) -> Result<Program> {
        let defs = program.defs.into_iter().map(|def| self.lift_def(def)).collect::<Result<Vec<_>>>()?;

        Ok(Program {
            defs,
            lambda_registry: program.lambda_registry,
        })
    }

    /// Lift bindings in a single definition.
    fn lift_def(&mut self, def: Def) -> Result<Def> {
        match def {
            Def::Function {
                id,
                name,
                params,
                ret_type,
                attributes,
                body,
                span,
            } => {
                let body = self.lift_expr(body)?;
                Ok(Def::Function {
                    id,
                    name,
                    params,
                    ret_type,
                    attributes,
                    body,
                    span,
                })
            }
            Def::Constant {
                id,
                name,
                ty,
                attributes,
                body,
                span,
            } => {
                let body = self.lift_expr(body)?;
                Ok(Def::Constant {
                    id,
                    name,
                    ty,
                    attributes,
                    body,
                    span,
                })
            }
            Def::Uniform { .. } => Ok(def),
            Def::Storage { .. } => Ok(def),
            Def::EntryPoint {
                id,
                name,
                execution_model,
                inputs,
                outputs,
                body,
                span,
            } => {
                let body = self.lift_expr(body)?;
                Ok(Def::EntryPoint {
                    id,
                    name,
                    execution_model,
                    inputs,
                    outputs,
                    body,
                    span,
                })
            }
        }
    }

    /// Main recursive driver: lift bindings in an expression.
    pub fn lift_expr(&mut self, expr: Expr) -> Result<Expr> {
        let id = expr.id;
        let ty = expr.ty.clone();
        let span = expr.span;

        match expr.kind {
            ExprKind::Loop { .. } => self.lift_loop(expr),

            ExprKind::Let {
                name,
                binding_id,
                value,
                body,
            } => {
                // Recursively lift in both value and body
                let value = self.lift_expr(*value)?;
                let body = self.lift_expr(*body)?;
                Ok(Expr::new(
                    id,
                    ty,
                    ExprKind::Let {
                        name,
                        binding_id,
                        value: Box::new(value),
                        body: Box::new(body),
                    },
                    span,
                ))
            }

            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond = self.lift_expr(*cond)?;
                let then_branch = self.lift_expr(*then_branch)?;
                let else_branch = self.lift_expr(*else_branch)?;
                Ok(Expr::new(
                    id,
                    ty,
                    ExprKind::If {
                        cond: Box::new(cond),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    },
                    span,
                ))
            }

            ExprKind::BinOp { op, lhs, rhs } => {
                let lhs = self.lift_expr(*lhs)?;
                let rhs = self.lift_expr(*rhs)?;
                Ok(Expr::new(
                    id,
                    ty,
                    ExprKind::BinOp {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    },
                    span,
                ))
            }

            ExprKind::UnaryOp { op, operand } => {
                let operand = self.lift_expr(*operand)?;
                Ok(Expr::new(
                    id,
                    ty,
                    ExprKind::UnaryOp {
                        op,
                        operand: Box::new(operand),
                    },
                    span,
                ))
            }

            ExprKind::Call { func, args } => {
                let args = args.into_iter().map(|a| self.lift_expr(a)).collect::<Result<Vec<_>>>()?;
                Ok(Expr::new(id, ty, ExprKind::Call { func, args }, span))
            }

            ExprKind::Intrinsic { name, args } => {
                let args = args.into_iter().map(|a| self.lift_expr(a)).collect::<Result<Vec<_>>>()?;
                Ok(Expr::new(id, ty, ExprKind::Intrinsic { name, args }, span))
            }

            ExprKind::Attributed {
                attributes,
                expr: inner,
            } => {
                let inner = self.lift_expr(*inner)?;
                Ok(Expr::new(
                    id,
                    ty,
                    ExprKind::Attributed {
                        attributes,
                        expr: Box::new(inner),
                    },
                    span,
                ))
            }

            ExprKind::Materialize(inner) => {
                let inner = self.lift_expr(*inner)?;
                Ok(Expr::new(id, ty, ExprKind::Materialize(Box::new(inner)), span))
            }

            ExprKind::Literal(lit) => {
                let lit = self.lift_literal(lit)?;
                Ok(Expr::new(id, ty, ExprKind::Literal(lit), span))
            }

            ExprKind::Closure {
                lambda_name,
                captures,
            } => {
                let captures =
                    captures.into_iter().map(|c| self.lift_expr(c)).collect::<Result<Vec<_>>>()?;
                Ok(Expr::new(
                    id,
                    ty,
                    ExprKind::Closure {
                        lambda_name,
                        captures,
                    },
                    span,
                ))
            }

            ExprKind::Range {
                start,
                step,
                end,
                kind,
            } => {
                let start = self.lift_expr(*start)?;
                let step = step.map(|s| self.lift_expr(*s)).transpose()?;
                let end = self.lift_expr(*end)?;
                Ok(Expr::new(
                    id,
                    ty,
                    ExprKind::Range {
                        start: Box::new(start),
                        step: step.map(Box::new),
                        end: Box::new(end),
                        kind,
                    },
                    span,
                ))
            }

            // Leaf nodes - no children to process
            ExprKind::Var(_) | ExprKind::Unit => Ok(Expr::new(id, ty, expr.kind, span)),
        }
    }

    /// Lift bindings in literals (tuples, arrays, etc. may contain expressions).
    fn lift_literal(&mut self, lit: Literal) -> Result<Literal> {
        match lit {
            Literal::Tuple(elems) => {
                let elems = elems.into_iter().map(|e| self.lift_expr(e)).collect::<Result<Vec<_>>>()?;
                Ok(Literal::Tuple(elems))
            }
            Literal::Array(elems) => {
                let elems = elems.into_iter().map(|e| self.lift_expr(e)).collect::<Result<Vec<_>>>()?;
                Ok(Literal::Array(elems))
            }
            Literal::Vector(elems) => {
                let elems = elems.into_iter().map(|e| self.lift_expr(e)).collect::<Result<Vec<_>>>()?;
                Ok(Literal::Vector(elems))
            }
            Literal::Matrix(rows) => {
                let rows = rows
                    .into_iter()
                    .map(|row| row.into_iter().map(|e| self.lift_expr(e)).collect::<Result<Vec<_>>>())
                    .collect::<Result<Vec<_>>>()?;
                Ok(Literal::Matrix(rows))
            }
            // Scalar literals have no sub-expressions
            Literal::Int(_) | Literal::Float(_) | Literal::Bool(_) | Literal::String(_) => Ok(lit),
        }
    }

    /// Lift loop-invariant bindings out of a loop.
    fn lift_loop(&mut self, loop_expr: Expr) -> Result<Expr> {
        let ExprKind::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body,
        } = loop_expr.kind
        else {
            unreachable!("lift_loop called on non-loop expression");
        };

        let id = loop_expr.id;
        let ty = loop_expr.ty;
        let span = loop_expr.span;

        // 1. Recursively lift in init expression
        let init = self.lift_expr(*init)?;

        // 2. Recursively lift in init_bindings expressions
        let init_bindings = init_bindings
            .into_iter()
            .map(|(name, expr)| Ok((name, self.lift_expr(expr)?)))
            .collect::<Result<Vec<_>>>()?;

        // 3. Recursively lift in loop kind (iter expression or condition)
        let kind = self.lift_loop_kind(kind)?;

        // 4. Recursively lift nested loops in body first
        let body = self.lift_expr(*body)?;

        // 5. Bubble up Lets from inside pure contexts (arrays, function args, etc.)
        //    so they become visible to linearize_body
        // let body = bubble_up_lets(body);  // TEMPORARILY DISABLED

        // 6. Linearize the body
        let LinearizedBody { bindings, result } = linearize_body(body);

        // If no bindings, nothing to hoist
        if bindings.is_empty() {
            return Ok(Expr::new(
                id,
                ty,
                ExprKind::Loop {
                    loop_var,
                    init: Box::new(init),
                    init_bindings,
                    kind,
                    body: Box::new(result),
                },
                span,
            ));
        }

        // 6. Compute loop-scoped variables
        let mut loop_vars: HashSet<String> = HashSet::new();
        loop_vars.insert(loop_var.clone());
        for (name, _) in &init_bindings {
            loop_vars.insert(name.clone());
        }
        match &kind {
            LoopKind::For { var, .. } | LoopKind::ForRange { var, .. } => {
                loop_vars.insert(var.clone());
            }
            LoopKind::While { .. } => {}
        }

        // 7. Partition bindings into hoistable and remaining
        let (hoistable, remaining) = partition_bindings(bindings, &loop_vars);

        // 8. Rebuild the loop body with remaining bindings
        let new_body = rebuild_nested_lets(remaining, result);

        // 9. Create the new loop
        let new_loop = Expr::new(
            id,
            ty,
            ExprKind::Loop {
                loop_var,
                init: Box::new(init),
                init_bindings,
                kind,
                body: Box::new(new_body),
            },
            span,
        );

        // 10. Wrap hoisted bindings around the loop
        Ok(rebuild_nested_lets(hoistable, new_loop))
    }

    /// Lift bindings in loop kind expressions.
    fn lift_loop_kind(&mut self, kind: LoopKind) -> Result<LoopKind> {
        match kind {
            LoopKind::For { var, iter } => {
                let iter = self.lift_expr(*iter)?;
                Ok(LoopKind::For {
                    var,
                    iter: Box::new(iter),
                })
            }
            LoopKind::ForRange { var, bound } => {
                let bound = self.lift_expr(*bound)?;
                Ok(LoopKind::ForRange {
                    var,
                    bound: Box::new(bound),
                })
            }
            LoopKind::While { cond } => {
                let cond = self.lift_expr(*cond)?;
                Ok(LoopKind::While { cond: Box::new(cond) })
            }
        }
    }
}

/// Linearize a nested Let chain into a flat list of bindings.
fn linearize_body(mut expr: Expr) -> LinearizedBody {
    let mut bindings = Vec::new();

    while let ExprKind::Let {
        name,
        binding_id,
        value,
        body,
    } = expr.kind
    {
        let free_vars = collect_free_vars(&value);
        bindings.push(LinearBinding {
            id: expr.id,
            name,
            binding_id,
            value: *value,
            free_vars,
            span: expr.span,
        });
        expr = *body;
    }

    LinearizedBody {
        bindings,
        result: expr,
    }
}

/// Partition bindings into hoistable (loop-invariant) and remaining (loop-dependent).
fn partition_bindings(
    bindings: Vec<LinearBinding>,
    loop_vars: &HashSet<String>,
) -> (Vec<LinearBinding>, Vec<LinearBinding>) {
    let mut tainted = loop_vars.clone();
    let mut hoistable = Vec::new();
    let mut remaining = Vec::new();

    for binding in bindings {
        if binding.free_vars.is_disjoint(&tainted) {
            // Can hoist - no loop dependencies
            hoistable.push(binding);
        } else {
            // Cannot hoist - mark this name as tainted for subsequent bindings
            tainted.insert(binding.name.clone());
            remaining.push(binding);
        }
    }

    (hoistable, remaining)
}

/// Rebuild a nested Let chain from linear bindings.
fn rebuild_nested_lets(bindings: Vec<LinearBinding>, result: Expr) -> Expr {
    bindings.into_iter().rev().fold(result, |body, binding| {
        Expr::new(
            binding.id,
            body.ty.clone(),
            ExprKind::Let {
                name: binding.name,
                binding_id: binding.binding_id,
                value: Box::new(binding.value),
                body: Box::new(body),
            },
            binding.span,
        )
    })
}

/// Collect free variables in an expression.
pub fn collect_free_vars(expr: &Expr) -> HashSet<String> {
    let mut free = HashSet::new();
    collect_free_vars_inner(expr, &HashSet::new(), &mut free);
    free
}

/// Inner recursive function for collecting free variables.
fn collect_free_vars_inner(expr: &Expr, bound: &HashSet<String>, free: &mut HashSet<String>) {
    match &expr.kind {
        ExprKind::Var(name) => {
            if !bound.contains(name) {
                free.insert(name.clone());
            }
        }

        ExprKind::Let {
            name, value, body, ..
        } => {
            collect_free_vars_inner(value, bound, free);
            let mut extended = bound.clone();
            extended.insert(name.clone());
            collect_free_vars_inner(body, &extended, free);
        }

        ExprKind::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body,
        } => {
            collect_free_vars_inner(init, bound, free);

            // init_bindings reference loop_var, but their expressions are evaluated
            // in the context where loop_var is bound
            let mut extended = bound.clone();
            extended.insert(loop_var.clone());

            for (name, binding_expr) in init_bindings {
                collect_free_vars_inner(binding_expr, &extended, free);
                extended.insert(name.clone());
            }

            match kind {
                LoopKind::For { var, iter } => {
                    collect_free_vars_inner(iter, bound, free);
                    extended.insert(var.clone());
                }
                LoopKind::ForRange { var, bound: upper } => {
                    collect_free_vars_inner(upper, bound, free);
                    extended.insert(var.clone());
                }
                LoopKind::While { cond } => {
                    collect_free_vars_inner(cond, &extended, free);
                }
            }

            collect_free_vars_inner(body, &extended, free);
        }

        ExprKind::BinOp { lhs, rhs, .. } => {
            collect_free_vars_inner(lhs, bound, free);
            collect_free_vars_inner(rhs, bound, free);
        }

        ExprKind::UnaryOp { operand, .. } => {
            collect_free_vars_inner(operand, bound, free);
        }

        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_free_vars_inner(cond, bound, free);
            collect_free_vars_inner(then_branch, bound, free);
            collect_free_vars_inner(else_branch, bound, free);
        }

        ExprKind::Call { args, .. } | ExprKind::Intrinsic { args, .. } => {
            for arg in args {
                collect_free_vars_inner(arg, bound, free);
            }
        }

        ExprKind::Literal(lit) => {
            collect_free_vars_in_literal(lit, bound, free);
        }

        ExprKind::Attributed { expr, .. } => {
            collect_free_vars_inner(expr, bound, free);
        }

        ExprKind::Materialize(inner) => {
            collect_free_vars_inner(inner, bound, free);
        }

        ExprKind::Closure { captures, .. } => {
            for cap in captures {
                collect_free_vars_inner(cap, bound, free);
            }
        }

        ExprKind::Range { start, step, end, .. } => {
            collect_free_vars_inner(start, bound, free);
            if let Some(s) = step {
                collect_free_vars_inner(s, bound, free);
            }
            collect_free_vars_inner(end, bound, free);
        }

        ExprKind::Unit => {}
    }
}

/// Collect free variables in literal expressions.
fn collect_free_vars_in_literal(lit: &Literal, bound: &HashSet<String>, free: &mut HashSet<String>) {
    match lit {
        Literal::Tuple(elems) | Literal::Array(elems) | Literal::Vector(elems) => {
            for elem in elems {
                collect_free_vars_inner(elem, bound, free);
            }
        }
        Literal::Matrix(rows) => {
            for row in rows {
                for elem in row {
                    collect_free_vars_inner(elem, bound, free);
                }
            }
        }
        Literal::Int(_) | Literal::Float(_) | Literal::Bool(_) | Literal::String(_) => {}
    }
}
