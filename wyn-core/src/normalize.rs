//! A-Normal Form (ANF) normalization pass for MIR.
//!
//! This pass ensures that all compound expressions have atomic operands,
//! enabling code motion optimizations. After normalization:
//! - BinOp/UnaryOp operands are Local or scalar literals
//! - Call/Intrinsic args are Local or scalar literal
//! - Tuple/Array/Vector/Matrix elements are Local only
//! - Materialize inner is Local or scalar literal
//! - If/Loop conditions are Local or scalar literal

use crate::mir::transform::{AccumulatorStack, is_atomic, sort_stmts_by_deps};
use crate::mir::{
    ArrayBacking, Block, Body, Def, Expr, ExprId, LocalDecl, LocalId, LocalKind, LoopKind, Program, Stmt,
};
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
    /// Scoped accumulator for atomization stmts.
    /// Stmts are accumulated in the current scope and collected when the scope ends.
    stmt_stack: AccumulatorStack<Stmt>,
}

impl Normalizer {
    fn new() -> Self {
        Normalizer {
            next_temp_id: 0,
            expr_map: HashMap::new(),
            stmt_stack: AccumulatorStack::new(),
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
                dps_output,
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
                    dps_output,
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
        self.stmt_stack = AccumulatorStack::new(); // Fresh accumulator for this body

        let mut new_body = Body::new();

        // Copy locals (they don't change during normalization, new temps will be added)
        let mut local_map: HashMap<LocalId, LocalId> = HashMap::new();
        for (old_idx, local) in old_body.locals.iter().enumerate() {
            let old_id = LocalId(old_idx as u32);
            let new_id = new_body.alloc_local(local.clone());
            local_map.insert(old_id, new_id);
        }

        // Collect all stmts - we'll process them in order
        let stmts: Vec<Stmt> = old_body.iter_stmts().cloned().collect();

        // Process each stmt - atomization stmts go to stmt_stack
        for stmt in &stmts {
            // Normalize the RHS expression (atomizations go to stmt_stack)
            let new_rhs = self.normalize_expr(&mut new_body, &old_body, stmt.rhs, &local_map);

            // Add the original stmt with mapped local and new RHS
            let new_local = *local_map.get(&stmt.local).unwrap_or(&stmt.local);
            self.stmt_stack.push(Stmt {
                local: new_local,
                rhs: new_rhs,
            });
        }

        // Normalize the root expression
        let new_root = self.normalize_expr(&mut new_body, &old_body, old_body.root, &local_map);

        // Collect all stmts from the accumulator and sort by dependencies
        let all_stmts = self.stmt_stack.drain_all();
        let sorted_stmts = sort_stmts_by_deps(&new_body, &all_stmts);
        for stmt in sorted_stmts {
            new_body.push_stmt(stmt.local, stmt.rhs);
        }

        // Set root
        new_body.root = new_root;

        new_body
    }

    /// Normalize an expression, creating atomization bindings as needed.
    fn normalize_expr(
        &mut self,
        new_body: &mut Body,
        old_body: &Body,
        old_id: ExprId,
        local_map: &HashMap<LocalId, LocalId>,
    ) -> ExprId {
        // Check if already processed
        if let Some(&new_id) = self.expr_map.get(&old_id) {
            return new_id;
        }

        let ty = old_body.get_type(old_id).clone();
        let span = old_body.get_span(old_id);
        let node_id = old_body.get_node_id(old_id);

        let new_id = match old_body.get_expr(old_id).clone() {
            // Already atomic - just copy with local mapping
            Expr::Local(local_id) => {
                let new_local = *local_map.get(&local_id).unwrap_or(&local_id);
                new_body.alloc_expr(Expr::Local(new_local), ty, span, node_id)
            }
            Expr::Global(name) => new_body.alloc_expr(Expr::Global(name), ty, span, node_id),
            Expr::Extern(linkage) => new_body.alloc_expr(Expr::Extern(linkage), ty, span, node_id),
            Expr::Int(s) => new_body.alloc_expr(Expr::Int(s), ty, span, node_id),
            Expr::Float(s) => new_body.alloc_expr(Expr::Float(s), ty, span, node_id),
            Expr::Bool(b) => new_body.alloc_expr(Expr::Bool(b), ty, span, node_id),
            Expr::Unit => new_body.alloc_expr(Expr::Unit, ty, span, node_id),
            Expr::String(s) => new_body.alloc_expr(Expr::String(s), ty, span, node_id),

            // Binary operation - atomize operands
            Expr::BinOp { op, lhs, rhs } => {
                let new_lhs = self.normalize_expr(new_body, old_body, lhs, local_map);
                let new_rhs = self.normalize_expr(new_body, old_body, rhs, local_map);

                let atom_lhs = self.atomize_if_needed(new_body, new_lhs, node_id);
                let atom_rhs = self.atomize_if_needed(new_body, new_rhs, node_id);

                new_body.alloc_expr(
                    Expr::BinOp {
                        op,
                        lhs: atom_lhs,
                        rhs: atom_rhs,
                    },
                    ty,
                    span,
                    node_id,
                )
            }

            // Unary operation - atomize operand
            Expr::UnaryOp { op, operand } => {
                let new_operand = self.normalize_expr(new_body, old_body, operand, local_map);
                let atom_operand = self.atomize_if_needed(new_body, new_operand, node_id);

                new_body.alloc_expr(
                    Expr::UnaryOp {
                        op,
                        operand: atom_operand,
                    },
                    ty,
                    span,
                    node_id,
                )
            }

            // Tuple - atomize elements
            Expr::Tuple(elems) => {
                let new_elems: Vec<ExprId> = elems
                    .iter()
                    .map(|&e| {
                        let new_e = self.normalize_expr(new_body, old_body, e, local_map);
                        self.atomize_if_needed(new_body, new_e, node_id)
                    })
                    .collect();

                new_body.alloc_expr(Expr::Tuple(new_elems), ty, span, node_id)
            }

            // Vector - atomize elements
            Expr::Vector(elems) => {
                let new_elems: Vec<ExprId> = elems
                    .iter()
                    .map(|&e| {
                        let new_e = self.normalize_expr(new_body, old_body, e, local_map);
                        self.atomize_if_needed(new_body, new_e, node_id)
                    })
                    .collect();

                new_body.alloc_expr(Expr::Vector(new_elems), ty, span, node_id)
            }

            // Matrix - atomize elements
            Expr::Matrix(rows) => {
                let new_rows: Vec<Vec<ExprId>> = rows
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|&e| {
                                let new_e = self.normalize_expr(new_body, old_body, e, local_map);
                                self.atomize_if_needed(new_body, new_e, node_id)
                            })
                            .collect()
                    })
                    .collect();

                new_body.alloc_expr(Expr::Matrix(new_rows), ty, span, node_id)
            }

            // Array - atomize elements and size
            Expr::Array { backing, size } => {
                let new_size = self.normalize_expr(new_body, old_body, size, local_map);
                let atom_size = self.atomize_if_needed(new_body, new_size, node_id);

                let new_backing = match backing {
                    ArrayBacking::Literal(elems) => {
                        let new_elems: Vec<ExprId> = elems
                            .iter()
                            .map(|&e| {
                                let new_e = self.normalize_expr(new_body, old_body, e, local_map);
                                self.atomize_if_needed(new_body, new_e, node_id)
                            })
                            .collect();
                        ArrayBacking::Literal(new_elems)
                    }
                    ArrayBacking::Range { start, step, kind } => {
                        let new_start = self.normalize_expr(new_body, old_body, start, local_map);
                        let atom_start = self.atomize_if_needed(new_body, new_start, node_id);

                        let atom_step = step.map(|s| {
                            let new_s = self.normalize_expr(new_body, old_body, s, local_map);
                            self.atomize_if_needed(new_body, new_s, node_id)
                        });

                        ArrayBacking::Range {
                            start: atom_start,
                            step: atom_step,
                            kind,
                        }
                    }
                };

                new_body.alloc_expr(
                    Expr::Array {
                        backing: new_backing,
                        size: atom_size,
                    },
                    ty,
                    span,
                    node_id,
                )
            }

            // Call - atomize args (but preserve closure structure for SOACs)
            Expr::Call { func, args } => {
                let is_soac = func == "map" || func == "reduce" || func == "filter" || func == "scan";

                let new_args: Vec<ExprId> = args
                    .iter()
                    .enumerate()
                    .map(|(i, &a)| {
                        let new_a = self.normalize_expr(new_body, old_body, a, local_map);
                        // Don't atomize closure argument for SOACs
                        if is_soac && i == 0 {
                            new_a
                        } else {
                            self.atomize_if_needed(new_body, new_a, node_id)
                        }
                    })
                    .collect();

                new_body.alloc_expr(Expr::Call { func, args: new_args }, ty, span, node_id)
            }

            // Intrinsic - atomize args
            Expr::Intrinsic { name, args } => {
                let new_args: Vec<ExprId> = args
                    .iter()
                    .map(|&a| {
                        let new_a = self.normalize_expr(new_body, old_body, a, local_map);
                        self.atomize_if_needed(new_body, new_a, node_id)
                    })
                    .collect();

                new_body.alloc_expr(Expr::Intrinsic { name, args: new_args }, ty, span, node_id)
            }

            // If - atomize condition, normalize branches
            Expr::If { cond, then_, else_ } => {
                let new_cond = self.normalize_expr(new_body, old_body, cond, local_map);
                let atom_cond = self.atomize_if_needed(new_body, new_cond, node_id);

                // Process branches in their own scopes
                let new_then = self.normalize_block(new_body, old_body, &then_, local_map);
                let new_else = self.normalize_block(new_body, old_body, &else_, local_map);

                new_body.alloc_expr(
                    Expr::If {
                        cond: atom_cond,
                        then_: new_then,
                        else_: new_else,
                    },
                    ty,
                    span,
                    node_id,
                )
            }

            // Loop - atomize init and kind, normalize body
            Expr::Loop {
                loop_var,
                init,
                init_bindings,
                kind,
                body: loop_body,
            } => {
                let new_init = self.normalize_expr(new_body, old_body, init, local_map);
                let atom_init = self.atomize_if_needed(new_body, new_init, node_id);

                let new_loop_var = *local_map.get(&loop_var).unwrap_or(&loop_var);

                let new_init_bindings: Vec<(LocalId, ExprId)> = init_bindings
                    .iter()
                    .map(|(local, expr)| {
                        let new_local = *local_map.get(local).unwrap_or(local);
                        let new_expr = self.normalize_expr(new_body, old_body, *expr, local_map);
                        (new_local, new_expr)
                    })
                    .collect();

                let new_kind = match kind {
                    LoopKind::For { var, iter } => {
                        let new_var = *local_map.get(&var).unwrap_or(&var);
                        let new_iter = self.normalize_expr(new_body, old_body, iter, local_map);
                        let atom_iter = self.atomize_if_needed(new_body, new_iter, node_id);
                        LoopKind::For {
                            var: new_var,
                            iter: atom_iter,
                        }
                    }
                    LoopKind::ForRange { var, bound } => {
                        let new_var = *local_map.get(&var).unwrap_or(&var);
                        let new_bound = self.normalize_expr(new_body, old_body, bound, local_map);
                        let atom_bound = self.atomize_if_needed(new_body, new_bound, node_id);
                        LoopKind::ForRange {
                            var: new_var,
                            bound: atom_bound,
                        }
                    }
                    LoopKind::While { cond } => {
                        // Don't atomize While conditions - they're re-evaluated each iteration
                        // and may depend on loop variables that aren't in scope at this level
                        let new_cond = self.normalize_expr(new_body, old_body, cond, local_map);
                        LoopKind::While { cond: new_cond }
                    }
                };

                let new_loop_body = self.normalize_block(new_body, old_body, &loop_body, local_map);

                new_body.alloc_expr(
                    Expr::Loop {
                        loop_var: new_loop_var,
                        init: atom_init,
                        init_bindings: new_init_bindings,
                        kind: new_kind,
                        body: new_loop_body,
                    },
                    ty,
                    span,
                    node_id,
                )
            }

            // Materialize - atomize inner
            Expr::Materialize(inner) => {
                let new_inner = self.normalize_expr(new_body, old_body, inner, local_map);
                let atom_inner = self.atomize_if_needed(new_body, new_inner, node_id);

                new_body.alloc_expr(Expr::Materialize(atom_inner), ty, span, node_id)
            }

            // Attributed - pass through
            Expr::Attributed {
                attributes,
                expr: inner,
            } => {
                let new_inner = self.normalize_expr(new_body, old_body, inner, local_map);
                new_body.alloc_expr(
                    Expr::Attributed {
                        attributes,
                        expr: new_inner,
                    },
                    ty,
                    span,
                    node_id,
                )
            }

            // Memory operations - atomize subexpressions
            Expr::Load { ptr } => {
                let new_ptr = self.normalize_expr(new_body, old_body, ptr, local_map);
                let atom_ptr = self.atomize_if_needed(new_body, new_ptr, node_id);

                new_body.alloc_expr(Expr::Load { ptr: atom_ptr }, ty, span, node_id)
            }

            Expr::Store { ptr, value } => {
                let new_ptr = self.normalize_expr(new_body, old_body, ptr, local_map);
                let new_value = self.normalize_expr(new_body, old_body, value, local_map);

                let atom_ptr = self.atomize_if_needed(new_body, new_ptr, node_id);
                let atom_value = self.atomize_if_needed(new_body, new_value, node_id);

                new_body.alloc_expr(
                    Expr::Store {
                        ptr: atom_ptr,
                        value: atom_value,
                    },
                    ty,
                    span,
                    node_id,
                )
            }

            Expr::StorageView {
                set,
                binding,
                offset,
                len,
            } => {
                let new_offset = self.normalize_expr(new_body, old_body, offset, local_map);
                let new_len = self.normalize_expr(new_body, old_body, len, local_map);

                let atom_offset = self.atomize_if_needed(new_body, new_offset, node_id);
                let atom_len = self.atomize_if_needed(new_body, new_len, node_id);

                new_body.alloc_expr(
                    Expr::StorageView {
                        set,
                        binding,
                        offset: atom_offset,
                        len: atom_len,
                    },
                    ty,
                    span,
                    node_id,
                )
            }

            Expr::SliceStorageView { view, start, len } => {
                let new_view = self.normalize_expr(new_body, old_body, view, local_map);
                let new_start = self.normalize_expr(new_body, old_body, start, local_map);
                let new_len = self.normalize_expr(new_body, old_body, len, local_map);

                let atom_view = self.atomize_if_needed(new_body, new_view, node_id);
                let atom_start = self.atomize_if_needed(new_body, new_start, node_id);
                let atom_len = self.atomize_if_needed(new_body, new_len, node_id);

                new_body.alloc_expr(
                    Expr::SliceStorageView {
                        view: atom_view,
                        start: atom_start,
                        len: atom_len,
                    },
                    ty,
                    span,
                    node_id,
                )
            }

            Expr::StorageViewIndex { view, index } => {
                let new_view = self.normalize_expr(new_body, old_body, view, local_map);
                let new_index = self.normalize_expr(new_body, old_body, index, local_map);

                let atom_view = self.atomize_if_needed(new_body, new_view, node_id);
                let atom_index = self.atomize_if_needed(new_body, new_index, node_id);

                new_body.alloc_expr(
                    Expr::StorageViewIndex {
                        view: atom_view,
                        index: atom_index,
                    },
                    ty,
                    span,
                    node_id,
                )
            }

            Expr::StorageViewLen { view } => {
                let new_view = self.normalize_expr(new_body, old_body, view, local_map);
                let atom_view = self.atomize_if_needed(new_body, new_view, node_id);

                new_body.alloc_expr(Expr::StorageViewLen { view: atom_view }, ty, span, node_id)
            }
        };

        self.expr_map.insert(old_id, new_id);
        new_id
    }

    /// Normalize a block, processing stmts and result.
    fn normalize_block(
        &mut self,
        new_body: &mut Body,
        old_body: &Body,
        block: &Block,
        local_map: &HashMap<LocalId, LocalId>,
    ) -> Block {
        // Push a new scope for this block's stmts
        self.stmt_stack.push_scope();

        // Process each stmt in the block
        for stmt in &block.stmts {
            let new_rhs = self.normalize_expr(new_body, old_body, stmt.rhs, local_map);

            // Add the original stmt to this scope
            let new_local = *local_map.get(&stmt.local).unwrap_or(&stmt.local);
            self.stmt_stack.push(Stmt {
                local: new_local,
                rhs: new_rhs,
            });
        }

        // Process the result expression
        let new_result = self.normalize_expr(new_body, old_body, block.result, local_map);

        // Pop this scope's stmts and sort by dependencies
        let block_stmts = self.stmt_stack.pop_scope();
        let sorted_stmts = sort_stmts_by_deps(new_body, &block_stmts);

        Block::with_stmts(sorted_stmts, new_result)
    }

    /// Atomize an expression if it's not already atomic.
    /// Creates a new local binding and returns a reference to it.
    fn atomize_if_needed(
        &mut self,
        new_body: &mut Body,
        expr_id: ExprId,
        _node_id: crate::ast::NodeId,
    ) -> ExprId {
        let expr = new_body.get_expr(expr_id);
        if is_atomic(expr) {
            return expr_id;
        }

        // Create a new local for this expression
        let name = self.fresh_temp_name();
        let ty = new_body.get_type(expr_id).clone();
        let span = new_body.get_span(expr_id);
        let node_id = new_body.get_node_id(expr_id);

        let local_id = new_body.alloc_local(LocalDecl {
            name,
            ty: ty.clone(),
            span,
            kind: LocalKind::Let,
        });

        // Add a stmt binding the local to the accumulator (current scope)
        self.stmt_stack.push(Stmt {
            local: local_id,
            rhs: expr_id,
        });

        // Return a reference to the local
        new_body.alloc_expr(Expr::Local(local_id), ty, span, node_id)
    }
}
