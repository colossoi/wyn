//! In-place optimization rewriter for MIR.
//!
//! This pass rewrites array operations to their in-place variants when safe.
//! An operation can be done in-place when:
//! 1. The input array is not used after this operation (liveness)
//! 2. The operation preserves element type (e.g., f: a -> a for map)
//!
//! ## Rewrites
//!
//! - `_w_intrinsic_map(f, arr)` -> `_w_intrinsic_inplace_map(f, arr)`
//!   when `arr` is dead after use and `f: a -> a`

use std::collections::HashMap;

use crate::alias_checker::InPlaceInfo;
use crate::ast::{NodeId, Span, TypeName};
use crate::mir::{Body, Def, Expr, ExprId, LoopKind, Program};
use polytype::Type;

/// Rewrite eligible map operations to their in-place variants.
pub fn rewrite_inplace(program: Program, info: &InPlaceInfo) -> Program {
    let mut rewriter = InPlaceRewriter::new(info);
    rewriter.rewrite_program(program)
}

struct InPlaceRewriter<'a> {
    info: &'a InPlaceInfo,
    /// Mapping from old ExprId to new ExprId in the current body.
    expr_map: HashMap<ExprId, ExprId>,
}

impl<'a> InPlaceRewriter<'a> {
    fn new(info: &'a InPlaceInfo) -> Self {
        InPlaceRewriter {
            info,
            expr_map: HashMap::new(),
        }
    }

    fn rewrite_program(&mut self, program: Program) -> Program {
        let defs = program.defs.into_iter().map(|d| self.rewrite_def(d)).collect();
        Program {
            defs,
            lambda_registry: program.lambda_registry,
        }
    }

    fn rewrite_def(&mut self, def: Def) -> Def {
        match def {
            Def::Function {
                id,
                name,
                params,
                ret_type,
                scheme,
                attributes,
                body,
                span,
            } => {
                let new_body = self.rewrite_body(body);
                Def::Function {
                    id,
                    name,
                    params,
                    ret_type,
                    scheme,
                    attributes,
                    body: new_body,
                    span,
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
                let new_body = self.rewrite_body(body);
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
                let new_body = self.rewrite_body(body);
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
            // These don't have bodies to transform
            Def::Uniform { .. } | Def::Storage { .. } => def,
        }
    }

    fn rewrite_body(&mut self, old_body: Body) -> Body {
        // Clear mapping for fresh body
        self.expr_map.clear();

        let mut new_body = Body::new();

        // Copy locals (same locals are used)
        for local in &old_body.locals {
            new_body.alloc_local(local.clone());
        }

        // Process expressions in order (they're stored in dependency order)
        for (old_idx, old_expr) in old_body.exprs.iter().enumerate() {
            let old_id = ExprId(old_idx as u32);
            let ty = old_body.get_type(old_id).clone();
            let span = old_body.get_span(old_id);
            let node_id = old_body.get_node_id(old_id);

            let new_id = self.rewrite_expr(&mut new_body, &old_body, old_expr, &ty, span, node_id);
            self.expr_map.insert(old_id, new_id);
        }

        // Update root to point to the transformed root
        new_body.root = self.expr_map[&old_body.root];

        new_body
    }

    /// Remap an ExprId using the expr_map.
    fn remap(&self, old_id: ExprId) -> ExprId {
        self.expr_map[&old_id]
    }

    /// Rewrite a single expression, potentially transforming map -> inplace_map.
    fn rewrite_expr(
        &self,
        new_body: &mut Body,
        old_body: &Body,
        expr: &Expr,
        ty: &Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> ExprId {
        let new_expr = match expr {
            // Leaf nodes - just copy
            Expr::Local(local_id) => Expr::Local(*local_id),
            Expr::Global(name) => Expr::Global(name.clone()),
            Expr::Int(s) => Expr::Int(s.clone()),
            Expr::Float(s) => Expr::Float(s.clone()),
            Expr::Bool(b) => Expr::Bool(*b),
            Expr::Unit => Expr::Unit,
            Expr::String(s) => Expr::String(s.clone()),

            // Aggregates - remap children
            Expr::Tuple(elems) => Expr::Tuple(elems.iter().map(|e| self.remap(*e)).collect()),
            Expr::Array(elems) => Expr::Array(elems.iter().map(|e| self.remap(*e)).collect()),
            Expr::Vector(elems) => Expr::Vector(elems.iter().map(|e| self.remap(*e)).collect()),
            Expr::Matrix(rows) => {
                Expr::Matrix(rows.iter().map(|row| row.iter().map(|e| self.remap(*e)).collect()).collect())
            }

            // Operations - remap children
            Expr::BinOp { op, lhs, rhs } => Expr::BinOp {
                op: op.clone(),
                lhs: self.remap(*lhs),
                rhs: self.remap(*rhs),
            },
            Expr::UnaryOp { op, operand } => Expr::UnaryOp {
                op: op.clone(),
                operand: self.remap(*operand),
            },

            // Binding & control - remap children
            Expr::Let { local, rhs, body } => Expr::Let {
                local: *local,
                rhs: self.remap(*rhs),
                body: self.remap(*body),
            },
            Expr::If { cond, then_, else_ } => Expr::If {
                cond: self.remap(*cond),
                then_: self.remap(*then_),
                else_: self.remap(*else_),
            },
            Expr::Loop {
                loop_var,
                init,
                init_bindings,
                kind,
                body,
            } => {
                let new_kind = match kind {
                    LoopKind::For { var, iter } => LoopKind::For {
                        var: *var,
                        iter: self.remap(*iter),
                    },
                    LoopKind::ForRange { var, bound } => LoopKind::ForRange {
                        var: *var,
                        bound: self.remap(*bound),
                    },
                    LoopKind::While { cond } => LoopKind::While {
                        cond: self.remap(*cond),
                    },
                };
                Expr::Loop {
                    loop_var: *loop_var,
                    init: self.remap(*init),
                    init_bindings: init_bindings
                        .iter()
                        .map(|(local, expr)| (*local, self.remap(*expr)))
                        .collect(),
                    kind: new_kind,
                    body: self.remap(*body),
                }
            }

            // Calls - this is where we do the rewrite!
            Expr::Call { func, args } => {
                let new_args: Vec<_> = args.iter().map(|e| self.remap(*e)).collect();

                // Check if this is a map call that can be rewritten to inplace_map
                let new_func = if func == "_w_intrinsic_map" && args.len() == 2 {
                    if self.can_use_inplace(old_body, args, node_id) {
                        "_w_intrinsic_inplace_map".to_string()
                    } else {
                        func.clone()
                    }
                } else {
                    func.clone()
                };

                Expr::Call {
                    func: new_func,
                    args: new_args,
                }
            }

            Expr::Intrinsic { name, args } => Expr::Intrinsic {
                name: name.clone(),
                args: args.iter().map(|e| self.remap(*e)).collect(),
            },

            // Closures
            Expr::Closure {
                lambda_name,
                captures,
            } => Expr::Closure {
                lambda_name: lambda_name.clone(),
                captures: captures.iter().map(|c| self.remap(*c)).collect(),
            },

            // Ranges
            Expr::Range {
                start,
                step,
                end,
                kind,
            } => Expr::Range {
                start: self.remap(*start),
                step: step.map(|s| self.remap(s)),
                end: self.remap(*end),
                kind: kind.clone(),
            },

            // Special
            Expr::Materialize(inner) => Expr::Materialize(self.remap(*inner)),
            Expr::Attributed { attributes, expr } => Expr::Attributed {
                attributes: attributes.clone(),
                expr: self.remap(*expr),
            },

            // Slices
            Expr::OwnedSlice { data, len } => Expr::OwnedSlice {
                data: self.remap(*data),
                len: self.remap(*len),
            },
            Expr::BorrowedSlice { base, offset, len } => Expr::BorrowedSlice {
                base: self.remap(*base),
                offset: self.remap(*offset),
                len: self.remap(*len),
            },
        };

        new_body.alloc_expr(new_expr, ty.clone(), span, node_id)
    }

    /// Check if a map call can use the in-place variant.
    fn can_use_inplace(&self, body: &Body, args: &[ExprId], node_id: NodeId) -> bool {
        // Check 1: Is this node marked as can_reuse_input by alias analysis?
        if !self.info.can_reuse_input.contains(&node_id) {
            return false;
        }

        // Check 2: Does the mapping function preserve element type (a -> a)?
        // The closure is args[0], array is args[1]
        let closure_id = args[0];
        let closure_type = body.get_type(closure_id);

        // Closure type should be: (a -> b)
        // For in-place, we need a == b
        if let Type::Constructed(TypeName::Arrow, arrow_args) = closure_type {
            if arrow_args.len() == 2 {
                let input_elem = &arrow_args[0];
                let output_elem = &arrow_args[1];
                // Types must be equal for in-place
                return input_elem == output_elem;
            }
        }

        false
    }
}
