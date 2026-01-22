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

use std::collections::{HashMap, HashSet};

use crate::ast::{NodeId, Span, TypeName};
use crate::mir::{ArrayBacking, Body, Def, Expr, ExprId, LocalId, LoopKind, Program};
use polytype::Type;

/// A single binding in linear form, extracted from nested Let chains.
struct LinearBinding {
    /// The local variable being bound.
    local: LocalId,
    /// The right-hand side expression (ExprId in new body).
    rhs: ExprId,
    /// Set of local IDs that the value depends on.
    free_locals: HashSet<LocalId>,
    /// Type of the binding.
    ty: Type<TypeName>,
    /// Span for the binding.
    span: Span,
    /// NodeId for the binding.
    node_id: NodeId,
}

/// Linearized representation of a Let chain.
struct LinearizedBody {
    /// Bindings in topological order (dependencies before uses).
    bindings: Vec<LinearBinding>,
    /// The final result expression (non-Let).
    result: ExprId,
}

/// Normalize a MIR program, hoisting loop-invariant bindings.
pub fn lift_bindings(program: Program) -> Program {
    let mut lifter = BindingLifter::new();
    lifter.lift_program(program)
}

/// Binding lifter pass for hoisting loop-invariant bindings.
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

    /// Lift bindings in all definitions in a program.
    fn lift_program(&mut self, program: Program) -> Program {
        let defs = program.defs.into_iter().map(|d| self.lift_def(d)).collect();
        Program {
            defs,
            lambda_registry: program.lambda_registry,
        }
    }

    /// Lift bindings in a single definition.
    fn lift_def(&mut self, def: Def) -> Def {
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
                let new_body = self.lift_body(body);
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

    /// Lift bindings in a function body.
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

        // Update root to point to the transformed root
        new_body.root = self.expr_map[&old_body.root];

        new_body
    }

    /// Lift bindings in an expression.
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
            Expr::Int(s) => new_body.alloc_expr(Expr::Int(s.clone()), ty.clone(), span, node_id),
            Expr::Float(s) => new_body.alloc_expr(Expr::Float(s.clone()), ty.clone(), span, node_id),
            Expr::Bool(b) => new_body.alloc_expr(Expr::Bool(*b), ty.clone(), span, node_id),
            Expr::Unit => new_body.alloc_expr(Expr::Unit, ty.clone(), span, node_id),
            Expr::String(s) => new_body.alloc_expr(Expr::String(s.clone()), ty.clone(), span, node_id),

            // Loop - this is where hoisting happens
            Expr::Loop { .. } => self.lift_loop(new_body, old_body, expr, ty, span, node_id),

            // Let - map subexpressions
            Expr::Let {
                local,
                rhs,
                body: let_body,
            } => {
                let new_rhs = self.expr_map[rhs];
                let new_let_body = self.expr_map[let_body];
                new_body.alloc_expr(
                    Expr::Let {
                        local: *local,
                        rhs: new_rhs,
                        body: new_let_body,
                    },
                    ty.clone(),
                    span,
                    node_id,
                )
            }

            // If - map subexpressions
            Expr::If { cond, then_, else_ } => {
                let new_cond = self.expr_map[cond];
                let new_then = self.expr_map[then_];
                let new_else = self.expr_map[else_];
                new_body.alloc_expr(
                    Expr::If {
                        cond: new_cond,
                        then_: new_then,
                        else_: new_else,
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
                    ArrayBacking::Storage { name, offset } => ArrayBacking::Storage {
                        name: name.clone(),
                        offset: self.expr_map[offset],
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
        }
    }

    /// Lift loop-invariant bindings out of a loop.
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

        // 1. Map init expression
        let new_init = self.expr_map[init];

        // 2. Map init_bindings
        let new_init_bindings: Vec<_> =
            init_bindings.iter().map(|(local, expr)| (*local, self.expr_map[expr])).collect();

        // 3. Map loop kind
        let new_kind = self.map_loop_kind(kind);

        // 4. Map loop body
        let new_loop_body = self.expr_map[loop_body];

        // 5. Linearize the body to extract Let chain
        let LinearizedBody { bindings, result } = linearize_body(new_body, new_loop_body);

        // If no bindings, nothing to hoist
        if bindings.is_empty() {
            return new_body.alloc_expr(
                Expr::Loop {
                    loop_var: *loop_var,
                    init: new_init,
                    init_bindings: new_init_bindings,
                    kind: new_kind,
                    body: result,
                },
                ty.clone(),
                span,
                node_id,
            );
        }

        // 6. Compute loop-scoped variables (LocalIds)
        let mut loop_locals: HashSet<LocalId> = HashSet::new();
        loop_locals.insert(*loop_var);
        for (local, _) in &new_init_bindings {
            loop_locals.insert(*local);
        }
        match &new_kind {
            LoopKind::For { var, .. } | LoopKind::ForRange { var, .. } => {
                loop_locals.insert(*var);
            }
            LoopKind::While { .. } => {}
        }

        // 7. Partition bindings into hoistable and remaining
        let (hoistable, remaining) = partition_bindings(bindings, &loop_locals);

        // 8. Rebuild the loop body with remaining bindings
        let new_loop_body_inner = rebuild_nested_lets(new_body, remaining, result);

        // 9. Create the new loop
        let new_loop = new_body.alloc_expr(
            Expr::Loop {
                loop_var: *loop_var,
                init: new_init,
                init_bindings: new_init_bindings,
                kind: new_kind,
                body: new_loop_body_inner,
            },
            ty.clone(),
            span,
            node_id,
        );

        // 10. Wrap hoisted bindings around the loop
        rebuild_nested_lets(new_body, hoistable, new_loop)
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

/// Linearize a nested Let chain into a flat list of bindings.
fn linearize_body(body: &Body, mut expr_id: ExprId) -> LinearizedBody {
    let mut bindings = Vec::new();

    while let Expr::Let {
        local,
        rhs,
        body: let_body,
    } = body.get_expr(expr_id)
    {
        let free_locals = collect_free_locals(body, *rhs);
        let ty = body.get_type(expr_id).clone();
        let span = body.get_span(expr_id);
        let node_id = body.get_node_id(expr_id);

        bindings.push(LinearBinding {
            local: *local,
            rhs: *rhs,
            free_locals,
            ty,
            span,
            node_id,
        });
        expr_id = *let_body;
    }

    LinearizedBody {
        bindings,
        result: expr_id,
    }
}

/// Partition bindings into hoistable (loop-invariant) and remaining (loop-dependent).
fn partition_bindings(
    bindings: Vec<LinearBinding>,
    loop_locals: &HashSet<LocalId>,
) -> (Vec<LinearBinding>, Vec<LinearBinding>) {
    let mut tainted = loop_locals.clone();
    let mut hoistable = Vec::new();
    let mut remaining = Vec::new();

    for binding in bindings {
        if binding.free_locals.is_disjoint(&tainted) {
            // Can hoist - no loop dependencies
            hoistable.push(binding);
        } else {
            // Cannot hoist - mark this local as tainted for subsequent bindings
            tainted.insert(binding.local);
            remaining.push(binding);
        }
    }

    (hoistable, remaining)
}

/// Rebuild a nested Let chain from linear bindings.
fn rebuild_nested_lets(body: &mut Body, bindings: Vec<LinearBinding>, result: ExprId) -> ExprId {
    bindings.into_iter().rev().fold(result, |body_expr, binding| {
        body.alloc_expr(
            Expr::Let {
                local: binding.local,
                rhs: binding.rhs,
                body: body_expr,
            },
            binding.ty,
            binding.span,
            binding.node_id,
        )
    })
}

/// Collect free local variables in an expression.
fn collect_free_locals(body: &Body, expr_id: ExprId) -> HashSet<LocalId> {
    let mut free = HashSet::new();
    collect_free_locals_inner(body, expr_id, &HashSet::new(), &mut free);
    free
}

/// Inner recursive function for collecting free local variables.
fn collect_free_locals_inner(
    body: &Body,
    expr_id: ExprId,
    bound: &HashSet<LocalId>,
    free: &mut HashSet<LocalId>,
) {
    match body.get_expr(expr_id) {
        Expr::Local(local_id) => {
            if !bound.contains(local_id) {
                free.insert(*local_id);
            }
        }

        Expr::Let {
            local,
            rhs,
            body: let_body,
        } => {
            collect_free_locals_inner(body, *rhs, bound, free);
            let mut extended = bound.clone();
            extended.insert(*local);
            collect_free_locals_inner(body, *let_body, &extended, free);
        }

        Expr::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body: loop_body,
        } => {
            collect_free_locals_inner(body, *init, bound, free);

            let mut extended = bound.clone();
            extended.insert(*loop_var);

            for (local, binding_expr) in init_bindings {
                collect_free_locals_inner(body, *binding_expr, &extended, free);
                extended.insert(*local);
            }

            match kind {
                LoopKind::For { var, iter } => {
                    collect_free_locals_inner(body, *iter, bound, free);
                    extended.insert(*var);
                }
                LoopKind::ForRange { var, bound: upper } => {
                    collect_free_locals_inner(body, *upper, bound, free);
                    extended.insert(*var);
                }
                LoopKind::While { cond } => {
                    collect_free_locals_inner(body, *cond, &extended, free);
                }
            }

            collect_free_locals_inner(body, *loop_body, &extended, free);
        }

        Expr::BinOp { lhs, rhs, .. } => {
            collect_free_locals_inner(body, *lhs, bound, free);
            collect_free_locals_inner(body, *rhs, bound, free);
        }

        Expr::UnaryOp { operand, .. } => {
            collect_free_locals_inner(body, *operand, bound, free);
        }

        Expr::If { cond, then_, else_ } => {
            collect_free_locals_inner(body, *cond, bound, free);
            collect_free_locals_inner(body, *then_, bound, free);
            collect_free_locals_inner(body, *else_, bound, free);
        }

        Expr::Call { args, .. } | Expr::Intrinsic { args, .. } => {
            for arg in args {
                collect_free_locals_inner(body, *arg, bound, free);
            }
        }

        Expr::Tuple(elems) | Expr::Vector(elems) => {
            for elem in elems {
                collect_free_locals_inner(body, *elem, bound, free);
            }
        }

        Expr::Array { backing, size } => {
            collect_free_locals_inner(body, *size, bound, free);
            match backing {
                ArrayBacking::Literal(elems) => {
                    for elem in elems {
                        collect_free_locals_inner(body, *elem, bound, free);
                    }
                }
                ArrayBacking::Range { start, step, .. } => {
                    collect_free_locals_inner(body, *start, bound, free);
                    if let Some(s) = step {
                        collect_free_locals_inner(body, *s, bound, free);
                    }
                }
                ArrayBacking::IndexFn { index_fn } => {
                    collect_free_locals_inner(body, *index_fn, bound, free);
                }
                ArrayBacking::View { base, offset } => {
                    collect_free_locals_inner(body, *base, bound, free);
                    collect_free_locals_inner(body, *offset, bound, free);
                }
                ArrayBacking::Owned { data } => {
                    collect_free_locals_inner(body, *data, bound, free);
                }
                ArrayBacking::Storage { offset, .. } => {
                    collect_free_locals_inner(body, *offset, bound, free);
                }
            }
        }

        Expr::Matrix(rows) => {
            for row in rows {
                for elem in row {
                    collect_free_locals_inner(body, *elem, bound, free);
                }
            }
        }

        Expr::Materialize(inner) => {
            collect_free_locals_inner(body, *inner, bound, free);
        }

        Expr::Attributed { expr, .. } => {
            collect_free_locals_inner(body, *expr, bound, free);
        }

        Expr::Load { ptr } => {
            collect_free_locals_inner(body, *ptr, bound, free);
        }

        Expr::Store { ptr, value } => {
            collect_free_locals_inner(body, *ptr, bound, free);
            collect_free_locals_inner(body, *value, bound, free);
        }

        // Leaf nodes - no locals to collect
        Expr::Global(_) | Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::String(_) | Expr::Unit => {}
    }
}
