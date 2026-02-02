//! Binding lifting (code motion) pass for MIR.
//!
//! This pass remaps expressions in MIR bodies, copying them to a new body
//! while preserving the structure. Bindings are now handled via `body.stmts`
//! (flat statements) rather than nested `Expr::Let` constructs.

use std::collections::HashMap;

use crate::ast::{NodeId, Span, TypeName};
use crate::mir::{ArrayBacking, Block, Body, Def, Expr, ExprId, LoopKind, Program};
use polytype::Type;


/// Process a MIR program, remapping expressions in all bodies.
pub fn lift_bindings(program: Program) -> Program {
    let mut lifter = BindingLifter::new();
    lifter.lift_program(program)
}

/// Binding lifter pass for remapping expressions in MIR bodies.
struct BindingLifter {
    /// Mapping from old ExprId to new ExprId in the current body.
    expr_map: HashMap<ExprId, ExprId>,
}

impl BindingLifter {
    fn new() -> Self {
        BindingLifter {
            expr_map: HashMap::new(),
        }
    }

    /// Process all definitions in a program.
    fn lift_program(&mut self, program: Program) -> Program {
        let defs = program.defs.into_iter().map(|d| self.lift_def(d)).collect();
        Program {
            defs,
            lambda_registry: program.lambda_registry,
        }
    }

    /// Process a single definition.
    fn lift_def(&mut self, def: Def) -> Def {
        match def {
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
                let new_body = self.lift_body(body);
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
            Def::Constant { .. } => def,
            Def::Uniform { .. } => def,
            Def::Storage { .. } => def,
            Def::EntryPoint {
                id,
                name,
                execution_model,
                inputs,
                outputs,
                body,
                span,
            } => {
                let new_body = self.lift_body(body);
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
        }
    }

    /// Process a function body, remapping all expressions.
    fn lift_body(&mut self, old_body: Body) -> Body {
        self.expr_map.clear();

        let mut new_body = Body::new();

        // Copy locals (they don't change during lifting, same locals are used)
        for local in &old_body.locals {
            new_body.alloc_local(local.clone());
        }

        // Process expressions in order (they're stored in dependency order)
        for (old_idx, old_expr) in old_body.exprs.iter().enumerate() {
            let old_id = ExprId(old_idx as u32);
            let ty = old_body.get_type(old_id).clone();
            let span = old_body.get_span(old_id);
            let node_id = old_body.get_node_id(old_id);

            let new_id = self.lift_expr(&mut new_body, &old_body, old_expr, &ty, span, node_id);
            self.expr_map.insert(old_id, new_id);
        }

        // Copy statements, remapping their ExprIds
        for stmt in old_body.iter_stmts() {
            let new_rhs = self.expr_map[&stmt.rhs];
            new_body.push_stmt(stmt.local, new_rhs);
        }

        // Update root to point to the transformed root
        new_body.root = self.expr_map[&old_body.root];

        new_body
    }

    /// Remap an expression to the new body.
    fn lift_expr(
        &mut self,
        new_body: &mut Body,
        old_body: &Body,
        expr: &Expr,
        ty: &Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> ExprId {
        match expr {
            // Leaf nodes - just copy
            Expr::Local(local_id) => new_body.alloc_expr(Expr::Local(*local_id), ty.clone(), span, node_id),
            Expr::Global(name) => {
                new_body.alloc_expr(Expr::Global(name.clone()), ty.clone(), span, node_id)
            }
            Expr::Extern(linkage) => {
                new_body.alloc_expr(Expr::Extern(linkage.clone()), ty.clone(), span, node_id)
            }
            Expr::Int(s) => new_body.alloc_expr(Expr::Int(s.clone()), ty.clone(), span, node_id),
            Expr::Float(s) => new_body.alloc_expr(Expr::Float(s.clone()), ty.clone(), span, node_id),
            Expr::Bool(b) => new_body.alloc_expr(Expr::Bool(*b), ty.clone(), span, node_id),
            Expr::Unit => new_body.alloc_expr(Expr::Unit, ty.clone(), span, node_id),
            Expr::String(s) => new_body.alloc_expr(Expr::String(s.clone()), ty.clone(), span, node_id),

            // Loop - map subexpressions
            Expr::Loop { .. } => self.lift_loop(new_body, old_body, expr, ty, span, node_id),

            // If - map subexpressions
            Expr::If { cond, then_, else_ } => {
                let new_cond = self.expr_map[cond];
                let new_then = self.expr_map[&then_.result];
                let new_else = self.expr_map[&else_.result];
                new_body.alloc_expr(
                    Expr::If {
                        cond: new_cond,
                        then_: Block::new(new_then),
                        else_: Block::new(new_else),
                    },
                    ty.clone(),
                    span,
                    node_id,
                )
            }

            // BinOp - map subexpressions
            Expr::BinOp { op, lhs, rhs } => {
                let new_lhs = self.expr_map[lhs];
                let new_rhs = self.expr_map[rhs];
                new_body.alloc_expr(
                    Expr::BinOp {
                        op: op.clone(),
                        lhs: new_lhs,
                        rhs: new_rhs,
                    },
                    ty.clone(),
                    span,
                    node_id,
                )
            }

            // UnaryOp - map subexpressions
            Expr::UnaryOp { op, operand } => {
                let new_operand = self.expr_map[operand];
                new_body.alloc_expr(
                    Expr::UnaryOp {
                        op: op.clone(),
                        operand: new_operand,
                    },
                    ty.clone(),
                    span,
                    node_id,
                )
            }

            // Call - map subexpressions
            Expr::Call { func, args } => {
                let new_args: Vec<_> = args.iter().map(|a| self.expr_map[a]).collect();
                new_body.alloc_expr(
                    Expr::Call {
                        func: func.clone(),
                        args: new_args,
                    },
                    ty.clone(),
                    span,
                    node_id,
                )
            }

            // Intrinsic - map subexpressions
            Expr::Intrinsic { name, args } => {
                let new_args: Vec<_> = args.iter().map(|a| self.expr_map[a]).collect();
                new_body.alloc_expr(
                    Expr::Intrinsic {
                        name: name.clone(),
                        args: new_args,
                    },
                    ty.clone(),
                    span,
                    node_id,
                )
            }

            // Tuple - map subexpressions
            Expr::Tuple(elems) => {
                let new_elems: Vec<_> = elems.iter().map(|e| self.expr_map[e]).collect();
                new_body.alloc_expr(Expr::Tuple(new_elems), ty.clone(), span, node_id)
            }

            // Array - map subexpressions based on backing
            Expr::Array { backing, size } => {
                let new_size = self.expr_map[size];
                let new_backing = match backing {
                    ArrayBacking::Literal(elems) => {
                        let new_elems: Vec<_> = elems.iter().map(|e| self.expr_map[e]).collect();
                        ArrayBacking::Literal(new_elems)
                    }
                    ArrayBacking::Range { start, step, kind } => ArrayBacking::Range {
                        start: self.expr_map[start],
                        step: step.map(|s| self.expr_map[&s]),
                        kind: *kind,
                    },
                };
                new_body.alloc_expr(
                    Expr::Array {
                        backing: new_backing,
                        size: new_size,
                    },
                    ty.clone(),
                    span,
                    node_id,
                )
            }

            // Vector - map subexpressions
            Expr::Vector(elems) => {
                let new_elems: Vec<_> = elems.iter().map(|e| self.expr_map[e]).collect();
                new_body.alloc_expr(Expr::Vector(new_elems), ty.clone(), span, node_id)
            }

            // Matrix - map subexpressions
            Expr::Matrix(rows) => {
                let new_rows: Vec<Vec<_>> =
                    rows.iter().map(|row| row.iter().map(|e| self.expr_map[e]).collect()).collect();
                new_body.alloc_expr(Expr::Matrix(new_rows), ty.clone(), span, node_id)
            }

            // Materialize - map inner
            Expr::Materialize(inner) => {
                let new_inner = self.expr_map[inner];
                new_body.alloc_expr(Expr::Materialize(new_inner), ty.clone(), span, node_id)
            }

            // Attributed - map inner
            Expr::Attributed {
                attributes,
                expr: inner,
            } => {
                let new_inner = self.expr_map[inner];
                new_body.alloc_expr(
                    Expr::Attributed {
                        attributes: attributes.clone(),
                        expr: new_inner,
                    },
                    ty.clone(),
                    span,
                    node_id,
                )
            }

            // Memory operations - map subexpressions
            Expr::Load { ptr } => {
                let new_ptr = self.expr_map[ptr];
                new_body.alloc_expr(Expr::Load { ptr: new_ptr }, ty.clone(), span, node_id)
            }

            Expr::Store { ptr, value } => {
                let new_ptr = self.expr_map[ptr];
                let new_value = self.expr_map[value];
                new_body.alloc_expr(
                    Expr::Store {
                        ptr: new_ptr,
                        value: new_value,
                    },
                    ty.clone(),
                    span,
                    node_id,
                )
            }

            // Storage view operations
            Expr::StorageView {
                set,
                binding,
                offset,
                len,
            } => {
                let new_offset = self.expr_map[offset];
                let new_len = self.expr_map[len];
                new_body.alloc_expr(
                    Expr::StorageView {
                        set: *set,
                        binding: *binding,
                        offset: new_offset,
                        len: new_len,
                    },
                    ty.clone(),
                    span,
                    node_id,
                )
            }
            Expr::SliceStorageView { view, start, len } => {
                let new_view = self.expr_map[view];
                let new_start = self.expr_map[start];
                let new_len = self.expr_map[len];
                new_body.alloc_expr(
                    Expr::SliceStorageView {
                        view: new_view,
                        start: new_start,
                        len: new_len,
                    },
                    ty.clone(),
                    span,
                    node_id,
                )
            }
            Expr::StorageViewIndex { view, index } => {
                let new_view = self.expr_map[view];
                let new_index = self.expr_map[index];
                new_body.alloc_expr(
                    Expr::StorageViewIndex {
                        view: new_view,
                        index: new_index,
                    },
                    ty.clone(),
                    span,
                    node_id,
                )
            }
            Expr::StorageViewLen { view } => {
                let new_view = self.expr_map[view];
                new_body.alloc_expr(Expr::StorageViewLen { view: new_view }, ty.clone(), span, node_id)
            }
        }
    }

    /// Map loop expressions.
    fn lift_loop(
        &mut self,
        new_body: &mut Body,
        _old_body: &Body,
        expr: &Expr,
        ty: &Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> ExprId {
        let Expr::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body: loop_body,
        } = expr
        else {
            unreachable!("lift_loop called on non-loop expression");
        };

        // Map init expression
        let new_init = self.expr_map[init];

        // Map init_bindings
        let new_init_bindings: Vec<_> =
            init_bindings.iter().map(|(local, expr)| (*local, self.expr_map[expr])).collect();

        // Map loop kind
        let new_kind = self.map_loop_kind(kind);

        // Map loop body (access result from Block)
        let new_loop_body = self.expr_map[&loop_body.result];

        new_body.alloc_expr(
            Expr::Loop {
                loop_var: *loop_var,
                init: new_init,
                init_bindings: new_init_bindings,
                kind: new_kind,
                body: Block::new(new_loop_body),
            },
            ty.clone(),
            span,
            node_id,
        )
    }

    /// Map a loop kind to new body.
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
}
