//! A-Normal Form (ANF) normalization pass for MIR.
//!
//! This pass ensures that all compound expressions have atomic operands,
//! enabling code motion optimizations. After normalization:
//! - BinOp/UnaryOp operands are Local or scalar literals
//! - Call/Intrinsic args are Local or scalar literals
//! - Tuple/Array/Vector/Matrix elements are Local only
//! - Materialize inner is Local or scalar literal
//! - If/Loop conditions are Local or scalar literal

use crate::ast::{NodeId, Span, TypeName};
use crate::mir::{Body, Def, Expr, ExprId, LocalDecl, LocalId, LocalKind, LoopKind, Program};
use polytype::Type;
use std::collections::HashMap;

/// Normalize a MIR program to A-normal form.
pub fn normalize_program(program: Program) -> Program {
    let mut normalizer = Normalizer::new();
    normalizer.normalize_program(program)
}

/// Normalizer state for the ANF transformation.
struct Normalizer {
    /// Counter for generating unique temp names.
    next_temp_id: usize,
    /// Mapping from old ExprId to new ExprId in the current body.
    expr_map: HashMap<ExprId, ExprId>,
}

impl Normalizer {
    fn new() -> Self {
        Normalizer {
            next_temp_id: 0,
            expr_map: HashMap::new(),
        }
    }

    /// Generate a fresh temp variable name.
    fn fresh_temp_name(&mut self) -> String {
        let id = self.next_temp_id;
        self.next_temp_id += 1;
        format!("_w_norm_{}", id)
    }

    /// Normalize an entire program.
    fn normalize_program(&mut self, program: Program) -> Program {
        let defs = program.defs.into_iter().map(|d| self.normalize_def(d)).collect();
        Program {
            defs,
            lambda_registry: program.lambda_registry,
        }
    }

    /// Normalize a single definition.
    fn normalize_def(&mut self, def: Def) -> Def {
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
                let new_body = self.normalize_body(body);
                Def::Function {
                    id,
                    name,
                    params,
                    ret_type,
                    attributes,
                    body: new_body,
                    span,
                }
            }
            // Constants must remain compile-time literals, so don't normalize them
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
                let new_body = self.normalize_body(body);
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

    /// Normalize a function body.
    fn normalize_body(&mut self, old_body: Body) -> Body {
        self.expr_map.clear();

        let mut new_body = Body::new();

        // Copy locals (they don't change during normalization, new temps will be added)
        for local in &old_body.locals {
            new_body.alloc_local(local.clone());
        }

        // Process expressions in order (they're stored in dependency order)
        for (old_idx, old_expr) in old_body.exprs.iter().enumerate() {
            let old_id = ExprId(old_idx as u32);
            let ty = old_body.get_type(old_id).clone();
            let span = old_body.get_span(old_id);
            let node_id = old_body.get_node_id(old_id);

            let new_id = self.normalize_expr(&mut new_body, old_expr, &ty, span, node_id);
            self.expr_map.insert(old_id, new_id);
        }

        // Update root to point to the transformed root
        new_body.root = self.expr_map[&old_body.root];

        new_body
    }

    /// Normalize an expression, potentially wrapping it with Let bindings for atomization.
    fn normalize_expr(
        &mut self,
        body: &mut Body,
        expr: &Expr,
        ty: &Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> ExprId {
        match expr {
            // Already atomic - just copy
            Expr::Local(local_id) => body.alloc_expr(Expr::Local(*local_id), ty.clone(), span, node_id),
            Expr::Global(name) => body.alloc_expr(Expr::Global(name.clone()), ty.clone(), span, node_id),
            Expr::Int(s) => body.alloc_expr(Expr::Int(s.clone()), ty.clone(), span, node_id),
            Expr::Float(s) => body.alloc_expr(Expr::Float(s.clone()), ty.clone(), span, node_id),
            Expr::Bool(b) => body.alloc_expr(Expr::Bool(*b), ty.clone(), span, node_id),
            Expr::Unit => body.alloc_expr(Expr::Unit, ty.clone(), span, node_id),
            Expr::String(s) => body.alloc_expr(Expr::String(s.clone()), ty.clone(), span, node_id),

            // Binary operation - atomize both operands
            Expr::BinOp { op, lhs, rhs } => {
                let new_lhs = self.expr_map[lhs];
                let new_rhs = self.expr_map[rhs];

                let (atom_lhs, lhs_binding) = self.atomize(body, new_lhs, node_id);
                let (atom_rhs, rhs_binding) = self.atomize(body, new_rhs, node_id);

                let binop_id = body.alloc_expr(
                    Expr::BinOp {
                        op: op.clone(),
                        lhs: atom_lhs,
                        rhs: atom_rhs,
                    },
                    ty.clone(),
                    span,
                    node_id,
                );

                self.wrap_bindings(body, binop_id, ty, span, node_id, vec![rhs_binding, lhs_binding])
            }

            // Unary operation - atomize operand
            Expr::UnaryOp { op, operand } => {
                let new_operand = self.expr_map[operand];
                let (atom_operand, binding) = self.atomize(body, new_operand, node_id);

                let unop_id = body.alloc_expr(
                    Expr::UnaryOp {
                        op: op.clone(),
                        operand: atom_operand,
                    },
                    ty.clone(),
                    span,
                    node_id,
                );

                self.wrap_bindings(body, unop_id, ty, span, node_id, vec![binding])
            }

            // Tuple - atomize all elements (tuples need all elements to be vars)
            Expr::Tuple(elems) => {
                let mut new_elems = Vec::new();
                let mut bindings = Vec::new();

                for elem in elems {
                    let new_elem = self.expr_map[elem];
                    let (atom_elem, binding) = self.atomize(body, new_elem, node_id);
                    new_elems.push(atom_elem);
                    bindings.push(binding);
                }

                let tuple_id = body.alloc_expr(Expr::Tuple(new_elems), ty.clone(), span, node_id);
                bindings.reverse();
                self.wrap_bindings(body, tuple_id, ty, span, node_id, bindings)
            }

            // Array - atomize all elements
            Expr::Array(elems) => {
                let mut new_elems = Vec::new();
                let mut bindings = Vec::new();

                for elem in elems {
                    let new_elem = self.expr_map[elem];
                    let (atom_elem, binding) = self.atomize(body, new_elem, node_id);
                    new_elems.push(atom_elem);
                    bindings.push(binding);
                }

                let array_id = body.alloc_expr(Expr::Array(new_elems), ty.clone(), span, node_id);
                bindings.reverse();
                self.wrap_bindings(body, array_id, ty, span, node_id, bindings)
            }

            // Vector - atomize all elements
            Expr::Vector(elems) => {
                let mut new_elems = Vec::new();
                let mut bindings = Vec::new();

                for elem in elems {
                    let new_elem = self.expr_map[elem];
                    let (atom_elem, binding) = self.atomize(body, new_elem, node_id);
                    new_elems.push(atom_elem);
                    bindings.push(binding);
                }

                let vector_id = body.alloc_expr(Expr::Vector(new_elems), ty.clone(), span, node_id);
                bindings.reverse();
                self.wrap_bindings(body, vector_id, ty, span, node_id, bindings)
            }

            // Matrix - atomize all elements
            Expr::Matrix(rows) => {
                let mut new_rows = Vec::new();
                let mut bindings = Vec::new();

                for row in rows {
                    let mut new_row = Vec::new();
                    for elem in row {
                        let new_elem = self.expr_map[elem];
                        let (atom_elem, binding) = self.atomize(body, new_elem, node_id);
                        new_row.push(atom_elem);
                        bindings.push(binding);
                    }
                    new_rows.push(new_row);
                }

                let matrix_id = body.alloc_expr(Expr::Matrix(new_rows), ty.clone(), span, node_id);
                bindings.reverse();
                self.wrap_bindings(body, matrix_id, ty, span, node_id, bindings)
            }

            // Call - atomize args (but preserve Closures for map/reduce)
            Expr::Call { func, args } => {
                let mut new_args = Vec::new();
                let mut bindings = Vec::new();

                // For map/reduce, keep first arg (closure) as-is to preserve Closure structure
                let is_soac = func == "map" || func == "reduce" || func == "filter" || func == "scan";

                for (i, arg) in args.iter().enumerate() {
                    let new_arg = self.expr_map[arg];

                    // Don't atomize the closure argument for SOACs
                    if is_soac && i == 0 {
                        // Keep the closure expression as-is (but still map to new body)
                        new_args.push(new_arg);
                        bindings.push(None);
                    } else {
                        let (atom_arg, binding) = self.atomize(body, new_arg, node_id);
                        new_args.push(atom_arg);
                        bindings.push(binding);
                    }
                }

                let call_id = body.alloc_expr(
                    Expr::Call {
                        func: func.clone(),
                        args: new_args,
                    },
                    ty.clone(),
                    span,
                    node_id,
                );
                bindings.reverse();
                self.wrap_bindings(body, call_id, ty, span, node_id, bindings)
            }

            // Intrinsic - atomize all args
            Expr::Intrinsic { name, args } => {
                let mut new_args = Vec::new();
                let mut bindings = Vec::new();

                for arg in args {
                    let new_arg = self.expr_map[arg];
                    let (atom_arg, binding) = self.atomize(body, new_arg, node_id);
                    new_args.push(atom_arg);
                    bindings.push(binding);
                }

                let intrinsic_id = body.alloc_expr(
                    Expr::Intrinsic {
                        name: name.clone(),
                        args: new_args,
                    },
                    ty.clone(),
                    span,
                    node_id,
                );
                bindings.reverse();
                self.wrap_bindings(body, intrinsic_id, ty, span, node_id, bindings)
            }

            // Let - map subexpressions (let body creates its own scope for bindings)
            Expr::Let {
                local,
                rhs,
                body: let_body,
            } => {
                let new_rhs = self.expr_map[rhs];
                let new_body = self.expr_map[let_body];
                body.alloc_expr(
                    Expr::Let {
                        local: *local,
                        rhs: new_rhs,
                        body: new_body,
                    },
                    ty.clone(),
                    span,
                    node_id,
                )
            }

            // If - atomize condition, branches are their own scopes
            Expr::If { cond, then_, else_ } => {
                let new_cond = self.expr_map[cond];
                let new_then = self.expr_map[then_];
                let new_else = self.expr_map[else_];

                let (atom_cond, cond_binding) = self.atomize(body, new_cond, node_id);

                let if_id = body.alloc_expr(
                    Expr::If {
                        cond: atom_cond,
                        then_: new_then,
                        else_: new_else,
                    },
                    ty.clone(),
                    span,
                    node_id,
                );

                self.wrap_bindings(body, if_id, ty, span, node_id, vec![cond_binding])
            }

            // Loop - atomize init and loop kind subexpressions
            Expr::Loop {
                loop_var,
                init,
                init_bindings,
                kind,
                body: loop_body,
            } => {
                let new_init = self.expr_map[init];
                let (atom_init, init_binding) = self.atomize(body, new_init, node_id);

                let new_init_bindings: Vec<_> =
                    init_bindings.iter().map(|(local, expr)| (*local, self.expr_map[expr])).collect();

                let (new_kind, kind_bindings) = self.map_loop_kind(body, kind, node_id);
                let new_loop_body = self.expr_map[loop_body];

                let loop_id = body.alloc_expr(
                    Expr::Loop {
                        loop_var: *loop_var,
                        init: atom_init,
                        init_bindings: new_init_bindings,
                        kind: new_kind,
                        body: new_loop_body,
                    },
                    ty.clone(),
                    span,
                    node_id,
                );

                // Combine all bindings: kind bindings first (outermost), then init binding
                let mut all_bindings = kind_bindings;
                all_bindings.push(init_binding);
                self.wrap_bindings(body, loop_id, ty, span, node_id, all_bindings)
            }

            // Closure - keep captures as-is (don't wrap in Let bindings)
            // This preserves the Closure structure for SOAC lowering
            Expr::Closure {
                lambda_name,
                captures,
            } => {
                // Just map captures to new body, don't atomize
                let new_captures: Vec<_> = captures.iter().map(|cap| self.expr_map[cap]).collect();

                body.alloc_expr(
                    Expr::Closure {
                        lambda_name: lambda_name.clone(),
                        captures: new_captures,
                    },
                    ty.clone(),
                    span,
                    node_id,
                )
            }

            // Range - atomize start, step, end
            Expr::Range {
                start,
                step,
                end,
                kind,
            } => {
                let new_start = self.expr_map[start];
                let new_end = self.expr_map[end];

                let (atom_start, start_binding) = self.atomize(body, new_start, node_id);
                let (atom_end, end_binding) = self.atomize(body, new_end, node_id);

                let (new_step, step_binding) = if let Some(s) = step {
                    let new_s = self.expr_map[s];
                    let (atom_s, binding) = self.atomize(body, new_s, node_id);
                    (Some(atom_s), binding)
                } else {
                    (None, None)
                };

                let range_id = body.alloc_expr(
                    Expr::Range {
                        start: atom_start,
                        step: new_step,
                        end: atom_end,
                        kind: *kind,
                    },
                    ty.clone(),
                    span,
                    node_id,
                );

                self.wrap_bindings(
                    body,
                    range_id,
                    ty,
                    span,
                    node_id,
                    vec![end_binding, step_binding, start_binding],
                )
            }

            // Materialize - atomize inner
            Expr::Materialize(inner) => {
                let new_inner = self.expr_map[inner];
                let (atom_inner, binding) = self.atomize(body, new_inner, node_id);

                let mat_id = body.alloc_expr(Expr::Materialize(atom_inner), ty.clone(), span, node_id);
                self.wrap_bindings(body, mat_id, ty, span, node_id, vec![binding])
            }

            // Attributed - pass through (the inner expression is already normalized)
            Expr::Attributed {
                attributes,
                expr: inner,
            } => {
                let new_inner = self.expr_map[inner];
                body.alloc_expr(
                    Expr::Attributed {
                        attributes: attributes.clone(),
                        expr: new_inner,
                    },
                    ty.clone(),
                    span,
                    node_id,
                )
            }
        }
    }

    /// Map a loop kind, atomizing its subexpressions where appropriate.
    /// Returns the new LoopKind and any bindings needed for atomization.
    ///
    /// Note: While conditions are NOT atomized because they must be re-evaluated
    /// on each iteration and may reference loop variables that change.
    /// ForRange bounds and For iterators ARE atomized because they're evaluated once.
    fn map_loop_kind(
        &mut self,
        body: &mut Body,
        kind: &LoopKind,
        node_id: NodeId,
    ) -> (LoopKind, Vec<Option<(LocalId, ExprId)>>) {
        match kind {
            LoopKind::For { var, iter } => {
                let new_iter = self.expr_map[iter];
                let (atom_iter, binding) = self.atomize(body, new_iter, node_id);
                (
                    LoopKind::For {
                        var: *var,
                        iter: atom_iter,
                    },
                    vec![binding],
                )
            }
            LoopKind::ForRange { var, bound } => {
                let new_bound = self.expr_map[bound];
                let (atom_bound, binding) = self.atomize(body, new_bound, node_id);
                (
                    LoopKind::ForRange {
                        var: *var,
                        bound: atom_bound,
                    },
                    vec![binding],
                )
            }
            LoopKind::While { cond } => {
                // While conditions must NOT be atomized - they are re-evaluated each iteration
                // and may reference loop variables (like `i < 16` where `i` is the loop counter)
                let new_cond = self.expr_map[cond];
                (LoopKind::While { cond: new_cond }, vec![])
            }
        }
    }

    /// Atomize an expression if needed. Returns the atomic ExprId and optional binding.
    fn atomize(
        &mut self,
        body: &mut Body,
        expr_id: ExprId,
        node_id: NodeId,
    ) -> (ExprId, Option<(LocalId, ExprId)>) {
        if is_atomic(body.get_expr(expr_id)) {
            (expr_id, None)
        } else {
            let ty = body.get_type(expr_id).clone();
            let span = body.get_span(expr_id);

            // Create temp local
            let local_id = body.alloc_local(LocalDecl {
                name: self.fresh_temp_name(),
                span,
                ty: ty.clone(),
                kind: LocalKind::Let,
            });

            // Create reference to local
            let local_ref = body.alloc_expr(Expr::Local(local_id), ty, span, node_id);

            (local_ref, Some((local_id, expr_id)))
        }
    }

    /// Wrap an expression with Let bindings for atomization.
    fn wrap_bindings(
        &self,
        body: &mut Body,
        mut expr_id: ExprId,
        ty: &Type<TypeName>,
        span: Span,
        node_id: NodeId,
        bindings: Vec<Option<(LocalId, ExprId)>>,
    ) -> ExprId {
        for binding in bindings.into_iter().flatten() {
            let (local_id, rhs) = binding;
            expr_id = body.alloc_expr(
                Expr::Let {
                    local: local_id,
                    rhs,
                    body: expr_id,
                },
                ty.clone(),
                span,
                node_id,
            );
        }
        expr_id
    }
}

/// Check if an expression is atomic (can appear as an operand without binding).
fn is_atomic(expr: &Expr) -> bool {
    match expr {
        Expr::Local(_) | Expr::Global(_) | Expr::Unit => true,
        Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::String(_) => true,
        Expr::Closure { .. } => true,           // Closures are atomic values
        Expr::Tuple(elems) => elems.is_empty(), // Empty tuple is atomic
        _ => false,
    }
}
