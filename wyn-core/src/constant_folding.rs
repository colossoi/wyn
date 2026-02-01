//! Constant folding pass for MIR.
//!
//! This pass evaluates constant expressions at compile time, reducing
//! operations on literals to their computed values.

use crate::ast::{NodeId, Span, TypeName};
use crate::error::Result;
use crate::mir::{ArrayBacking, Body, Def, Expr, ExprId, LoopKind, Program};
use crate::{bail_type_at, err_type_at};
use polytype::Type;
use std::collections::HashMap;

/// Fold constants in a MIR program.
pub fn fold_constants(program: Program) -> Result<Program> {
    let mut folder = ConstantFolder::new();
    folder.fold_program(program)
}

/// Constant folder that performs compile-time evaluation of constant expressions.
struct ConstantFolder {
    /// Mapping from old ExprId to new ExprId in the current body.
    expr_map: HashMap<ExprId, ExprId>,
}

impl ConstantFolder {
    fn new() -> Self {
        ConstantFolder {
            expr_map: HashMap::new(),
        }
    }

    fn fold_program(&mut self, program: Program) -> Result<Program> {
        let mut new_defs = Vec::new();

        for def in program.defs {
            let new_def = match def {
                Def::Function {
                    id,
                    name,
                    params,
                    ret_type,
                    attributes,
                    body,
                    span,
                    dps_output,
                } => {
                    let new_body = self.fold_body(body)?;
                    Def::Function {
                        id,
                        name,
                        params,
                        ret_type,
                        attributes,
                        body: new_body,
                        span,
                        dps_output,
                    }
                }
                Def::Constant {
                    id,
                    name,
                    ty,
                    attributes,
                    body,
                    span,
                } => {
                    let new_body = self.fold_body(body)?;
                    Def::Constant {
                        id,
                        name,
                        ty,
                        attributes,
                        body: new_body,
                        span,
                    }
                }
                Def::EntryPoint {
                    id,
                    name,
                    execution_model,
                    inputs,
                    outputs,
                    body,
                    span,
                } => {
                    let new_body = self.fold_body(body)?;
                    Def::EntryPoint {
                        id,
                        name,
                        execution_model,
                        inputs,
                        outputs,
                        body: new_body,
                        span,
                    }
                }
                // Uniforms and storage have no body to fold
                Def::Uniform { .. } | Def::Storage { .. } => def,
            };
            new_defs.push(new_def);
        }

        Ok(Program {
            defs: new_defs,
            lambda_registry: program.lambda_registry,
        })
    }

    fn fold_body(&mut self, old_body: Body) -> Result<Body> {
        self.expr_map.clear();

        let mut new_body = Body::new();

        // Copy locals (they don't change during constant folding)
        // The LocalIds will be the same since we're just copying
        for local in &old_body.locals {
            new_body.alloc_local(local.clone());
        }

        // Process expressions in order (they're stored in dependency order)
        for (old_idx, old_expr) in old_body.exprs.iter().enumerate() {
            let old_id = ExprId(old_idx as u32);
            let ty = old_body.get_type(old_id).clone();
            let span = old_body.get_span(old_id);
            let node_id = old_body.get_node_id(old_id);

            let new_id = self.fold_expr(&mut new_body, old_expr, &ty, span, node_id)?;
            self.expr_map.insert(old_id, new_id);
        }

        // Update root to point to the transformed root
        new_body.root = self.expr_map[&old_body.root];

        Ok(new_body)
    }

    fn fold_expr(
        &mut self,
        body: &mut Body,
        expr: &Expr,
        ty: &Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ExprId> {
        match expr {
            // Atoms - just copy them
            Expr::Local(local_id) => Ok(body.alloc_expr(Expr::Local(*local_id), ty.clone(), span, node_id)),
            Expr::Global(name) => {
                Ok(body.alloc_expr(Expr::Global(name.clone()), ty.clone(), span, node_id))
            }
            Expr::Extern(linkage) => {
                Ok(body.alloc_expr(Expr::Extern(linkage.clone()), ty.clone(), span, node_id))
            }
            Expr::Int(s) => Ok(body.alloc_expr(Expr::Int(s.clone()), ty.clone(), span, node_id)),
            Expr::Float(s) => Ok(body.alloc_expr(Expr::Float(s.clone()), ty.clone(), span, node_id)),
            Expr::Bool(b) => Ok(body.alloc_expr(Expr::Bool(*b), ty.clone(), span, node_id)),
            Expr::Unit => Ok(body.alloc_expr(Expr::Unit, ty.clone(), span, node_id)),
            Expr::String(s) => Ok(body.alloc_expr(Expr::String(s.clone()), ty.clone(), span, node_id)),

            // Aggregates - map child expressions
            Expr::Tuple(elems) => {
                let new_elems: Vec<_> = elems.iter().map(|e| self.expr_map[e]).collect();
                Ok(body.alloc_expr(Expr::Tuple(new_elems), ty.clone(), span, node_id))
            }
            Expr::Array { backing, size } => {
                let new_size = self.expr_map[size];
                let new_backing = match backing {
                    ArrayBacking::Literal(elems) => {
                        let new_elems: Vec<_> = elems.iter().map(|e| self.expr_map[e]).collect();
                        ArrayBacking::Literal(new_elems)
                    }
                    ArrayBacking::Range { start, step, kind } => {
                        let new_start = self.expr_map[start];
                        let new_step = step.map(|s| self.expr_map[&s]);
                        ArrayBacking::Range {
                            start: new_start,
                            step: new_step,
                            kind: *kind,
                        }
                    }
                    ArrayBacking::IndexFn { index_fn } => ArrayBacking::IndexFn {
                        index_fn: self.expr_map[index_fn],
                    },
                    ArrayBacking::View { base, offset } => ArrayBacking::View {
                        base: self.expr_map[base],
                        offset: self.expr_map[offset],
                    },
                    ArrayBacking::Owned { data } => ArrayBacking::Owned {
                        data: self.expr_map[data],
                    },
                };
                Ok(body.alloc_expr(
                    Expr::Array {
                        backing: new_backing,
                        size: new_size,
                    },
                    ty.clone(),
                    span,
                    node_id,
                ))
            }
            Expr::Vector(elems) => {
                let new_elems: Vec<_> = elems.iter().map(|e| self.expr_map[e]).collect();
                Ok(body.alloc_expr(Expr::Vector(new_elems), ty.clone(), span, node_id))
            }
            Expr::Matrix(rows) => {
                let new_rows: Vec<Vec<_>> =
                    rows.iter().map(|row| row.iter().map(|e| self.expr_map[e]).collect()).collect();
                Ok(body.alloc_expr(Expr::Matrix(new_rows), ty.clone(), span, node_id))
            }

            // Binary operations - try to fold
            Expr::BinOp { op, lhs, rhs } => {
                let new_lhs = self.expr_map[lhs];
                let new_rhs = self.expr_map[rhs];

                // Try to fold if both operands are literals
                if let Some(folded) = self.try_fold_binop(body, op, new_lhs, new_rhs, ty, span, node_id)? {
                    return Ok(folded);
                }

                Ok(body.alloc_expr(
                    Expr::BinOp {
                        op: op.clone(),
                        lhs: new_lhs,
                        rhs: new_rhs,
                    },
                    ty.clone(),
                    span,
                    node_id,
                ))
            }

            // Unary operations - try to fold
            Expr::UnaryOp { op, operand } => {
                let new_operand = self.expr_map[operand];

                // Try to fold if operand is a literal
                if let Some(folded) = self.try_fold_unaryop(body, op, new_operand, ty, span, node_id)? {
                    return Ok(folded);
                }

                Ok(body.alloc_expr(
                    Expr::UnaryOp {
                        op: op.clone(),
                        operand: new_operand,
                    },
                    ty.clone(),
                    span,
                    node_id,
                ))
            }

            // If - try to fold on constant condition
            Expr::If { cond, then_, else_ } => {
                let new_cond = self.expr_map[cond];
                let new_then = self.expr_map[then_];
                let new_else = self.expr_map[else_];

                // If condition is a constant bool, return the appropriate branch
                if let Expr::Bool(b) = body.get_expr(new_cond) {
                    return Ok(if *b { new_then } else { new_else });
                }

                Ok(body.alloc_expr(
                    Expr::If {
                        cond: new_cond,
                        then_: new_then,
                        else_: new_else,
                    },
                    ty.clone(),
                    span,
                    node_id,
                ))
            }

            // Let - map subexpressions
            Expr::Let {
                local,
                rhs,
                body: let_body,
            } => {
                let new_rhs = self.expr_map[rhs];
                let new_body = self.expr_map[let_body];
                Ok(body.alloc_expr(
                    Expr::Let {
                        local: *local,
                        rhs: new_rhs,
                        body: new_body,
                    },
                    ty.clone(),
                    span,
                    node_id,
                ))
            }

            // Loop - map subexpressions
            Expr::Loop {
                loop_var,
                init,
                init_bindings,
                kind,
                body: loop_body,
            } => {
                let new_init = self.expr_map[init];
                let new_init_bindings: Vec<_> =
                    init_bindings.iter().map(|(local, expr)| (*local, self.expr_map[expr])).collect();
                let new_kind = self.map_loop_kind(kind);
                let new_loop_body = self.expr_map[loop_body];

                Ok(body.alloc_expr(
                    Expr::Loop {
                        loop_var: *loop_var,
                        init: new_init,
                        init_bindings: new_init_bindings,
                        kind: new_kind,
                        body: new_loop_body,
                    },
                    ty.clone(),
                    span,
                    node_id,
                ))
            }

            // Call - map arguments
            Expr::Call { func, args } => {
                let new_args: Vec<_> = args.iter().map(|e| self.expr_map[e]).collect();
                Ok(body.alloc_expr(
                    Expr::Call {
                        func: func.clone(),
                        args: new_args,
                    },
                    ty.clone(),
                    span,
                    node_id,
                ))
            }

            // Intrinsic - map arguments
            Expr::Intrinsic { name, args } => {
                let new_args: Vec<_> = args.iter().map(|e| self.expr_map[e]).collect();
                Ok(body.alloc_expr(
                    Expr::Intrinsic {
                        name: name.clone(),
                        args: new_args,
                    },
                    ty.clone(),
                    span,
                    node_id,
                ))
            }

            // Materialize - map inner
            Expr::Materialize(inner) => {
                let new_inner = self.expr_map[inner];
                Ok(body.alloc_expr(Expr::Materialize(new_inner), ty.clone(), span, node_id))
            }

            // Attributed - map inner
            Expr::Attributed {
                attributes,
                expr: inner,
            } => {
                let new_inner = self.expr_map[inner];
                Ok(body.alloc_expr(
                    Expr::Attributed {
                        attributes: attributes.clone(),
                        expr: new_inner,
                    },
                    ty.clone(),
                    span,
                    node_id,
                ))
            }

            // Memory operations - map subexpressions
            Expr::Load { ptr } => {
                let new_ptr = self.expr_map[ptr];
                Ok(body.alloc_expr(Expr::Load { ptr: new_ptr }, ty.clone(), span, node_id))
            }

            Expr::Store { ptr, value } => {
                let new_ptr = self.expr_map[ptr];
                let new_value = self.expr_map[value];
                Ok(body.alloc_expr(
                    Expr::Store {
                        ptr: new_ptr,
                        value: new_value,
                    },
                    ty.clone(),
                    span,
                    node_id,
                ))
            }
        }
    }

    fn map_loop_kind(&self, kind: &LoopKind) -> LoopKind {
        match kind {
            LoopKind::For { var, iter } => LoopKind::For {
                var: *var,
                iter: self.expr_map[iter],
            },
            LoopKind::ForRange { var, bound } => LoopKind::ForRange {
                var: *var,
                bound: self.expr_map[bound],
            },
            LoopKind::While { cond } => LoopKind::While {
                cond: self.expr_map[cond],
            },
        }
    }

    /// Try to fold a binary operation on two literals.
    fn try_fold_binop(
        &self,
        body: &mut Body,
        op: &str,
        lhs: ExprId,
        rhs: ExprId,
        ty: &Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<Option<ExprId>> {
        let lhs_expr = body.get_expr(lhs);
        let rhs_expr = body.get_expr(rhs);

        match (lhs_expr, rhs_expr) {
            // Float operations
            (Expr::Float(l), Expr::Float(r)) => {
                let l: f64 = l.parse().map_err(|_| err_type_at!(span, "Invalid float literal"))?;
                let r: f64 = r.parse().map_err(|_| err_type_at!(span, "Invalid float literal"))?;

                let result = match op {
                    "+" => Some(l + r),
                    "-" => Some(l - r),
                    "*" => Some(l * r),
                    "/" => {
                        if r == 0.0 {
                            bail_type_at!(span, "Division by zero in constant expression");
                        }
                        Some(l / r)
                    }
                    _ => None,
                };

                if let Some(val) = result {
                    let lhs_ty = body.get_type(lhs).clone();
                    return Ok(Some(body.alloc_expr(
                        Expr::Float(val.to_string()),
                        lhs_ty,
                        span,
                        node_id,
                    )));
                }

                // Boolean comparison operations on floats
                let bool_result = match op {
                    "==" => Some(l == r),
                    "!=" => Some(l != r),
                    "<" => Some(l < r),
                    "<=" => Some(l <= r),
                    ">" => Some(l > r),
                    ">=" => Some(l >= r),
                    _ => None,
                };

                if let Some(val) = bool_result {
                    return Ok(Some(body.alloc_expr(Expr::Bool(val), ty.clone(), span, node_id)));
                }
            }

            // Integer operations
            (Expr::Int(l), Expr::Int(r)) => {
                let l: i64 = l.parse().map_err(|_| err_type_at!(span, "Invalid integer literal"))?;
                let r: i64 = r.parse().map_err(|_| err_type_at!(span, "Invalid integer literal"))?;

                let result = match op {
                    "+" => Some(l + r),
                    "-" => Some(l - r),
                    "*" => Some(l * r),
                    "/" => {
                        if r == 0 {
                            bail_type_at!(span, "Division by zero in constant expression");
                        }
                        Some(l / r)
                    }
                    "%" => {
                        if r == 0 {
                            bail_type_at!(span, "Modulo by zero in constant expression");
                        }
                        Some(l % r)
                    }
                    _ => None,
                };

                if let Some(val) = result {
                    let lhs_ty = body.get_type(lhs).clone();
                    return Ok(Some(body.alloc_expr(
                        Expr::Int(val.to_string()),
                        lhs_ty,
                        span,
                        node_id,
                    )));
                }

                // Boolean comparison operations on integers
                let bool_result = match op {
                    "==" => Some(l == r),
                    "!=" => Some(l != r),
                    "<" => Some(l < r),
                    "<=" => Some(l <= r),
                    ">" => Some(l > r),
                    ">=" => Some(l >= r),
                    _ => None,
                };

                if let Some(val) = bool_result {
                    return Ok(Some(body.alloc_expr(Expr::Bool(val), ty.clone(), span, node_id)));
                }
            }

            // Boolean operations
            (Expr::Bool(l), Expr::Bool(r)) => {
                let result = match op {
                    "&&" => Some(*l && *r),
                    "||" => Some(*l || *r),
                    "==" => Some(l == r),
                    "!=" => Some(l != r),
                    _ => None,
                };

                if let Some(val) = result {
                    return Ok(Some(body.alloc_expr(Expr::Bool(val), ty.clone(), span, node_id)));
                }
            }

            _ => {}
        }

        Ok(None)
    }

    /// Try to fold a unary operation on a literal.
    fn try_fold_unaryop(
        &self,
        body: &mut Body,
        op: &str,
        operand: ExprId,
        ty: &Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<Option<ExprId>> {
        let operand_expr = body.get_expr(operand);

        match (op, operand_expr) {
            // Negation of float
            ("-", Expr::Float(val)) => {
                let v: f64 = val.parse().map_err(|_| err_type_at!(span, "Invalid float literal"))?;
                Ok(Some(body.alloc_expr(
                    Expr::Float((-v).to_string()),
                    ty.clone(),
                    span,
                    node_id,
                )))
            }

            // Negation of integer
            ("-", Expr::Int(val)) => {
                let v: i64 = val.parse().map_err(|_| err_type_at!(span, "Invalid integer literal"))?;
                Ok(Some(body.alloc_expr(
                    Expr::Int((-v).to_string()),
                    ty.clone(),
                    span,
                    node_id,
                )))
            }

            // Boolean not
            ("!", Expr::Bool(val)) => {
                Ok(Some(body.alloc_expr(Expr::Bool(!val), ty.clone(), span, node_id)))
            }

            _ => Ok(None),
        }
    }
}
