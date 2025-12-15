//! A-Normal Form (ANF) normalization pass for MIR.
//!
//! This pass ensures that all compound expressions have atomic operands,
//! enabling code motion optimizations. After normalization:
//! - BinOp/UnaryOp operands are Var or scalar Literal
//! - Call/Intrinsic args are Var or scalar Literal
//! - Tuple/Array/Vector/Matrix elements are Var only
//! - Materialize inner is Var or scalar Literal
//! - If/Loop conditions are Var or scalar Literal

use crate::ast::NodeCounter;
use crate::mir::{Def, Expr, ExprKind, Literal, LoopKind, Program};

/// A pending let binding (name, binding_id, value).
type Binding = (String, u64, Expr);

/// Normalizer state for the ANF transformation.
pub struct Normalizer {
    /// Counter for generating unique binding IDs.
    next_binding_id: u64,
    /// Counter for generating unique temp names.
    next_temp_id: usize,
    /// Counter for generating unique node IDs.
    node_counter: NodeCounter,
}

impl Normalizer {
    /// Create a new normalizer with the given starting binding ID and node counter.
    pub fn new(starting_binding_id: u64, node_counter: NodeCounter) -> Self {
        Normalizer {
            next_binding_id: starting_binding_id,
            next_temp_id: 0,
            node_counter,
        }
    }

    /// Generate a fresh binding ID.
    fn fresh_binding_id(&mut self) -> u64 {
        let id = self.next_binding_id;
        self.next_binding_id += 1;
        id
    }

    /// Generate a fresh temp variable name.
    fn fresh_temp_name(&mut self) -> String {
        let id = self.next_temp_id;
        self.next_temp_id += 1;
        format!("_w_norm_{}", id)
    }

    /// Normalize an entire program.
    pub fn normalize_program(&mut self, program: Program) -> Program {
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
                let mut bindings = Vec::new();
                let body = self.normalize_expr(body, &mut bindings);
                let body = self.wrap_bindings(body, bindings);
                Def::Function {
                    id,
                    name,
                    params,
                    ret_type,
                    attributes,
                    body,
                    span,
                }
            }
            // Constants must remain compile-time literals, so don't normalize them
            Def::Constant { .. } => def,
            Def::Uniform { .. } => def, // Uniforms have no body to normalize
            Def::Storage { .. } => def, // Storage buffers have no body to normalize
            Def::EntryPoint {
                id,
                name,
                execution_model,
                inputs,
                outputs,
                body,
                span,
            } => {
                let mut bindings = Vec::new();
                let body = self.normalize_expr(body, &mut bindings);
                let body = self.wrap_bindings(body, bindings);
                Def::EntryPoint {
                    id,
                    name,
                    execution_model,
                    inputs,
                    outputs,
                    body,
                    span,
                }
            }
        }
    }

    /// Normalize an expression, collecting pending bindings.
    pub fn normalize_expr(&mut self, expr: Expr, bindings: &mut Vec<Binding>) -> Expr {
        let id = expr.id;
        let span = expr.span;
        let ty = expr.ty.clone();

        match expr.kind {
            // Already atomic - return as-is
            ExprKind::Var(_) | ExprKind::Unit => expr,

            // Scalar literals are atomic - return as-is
            ExprKind::Literal(Literal::Int(_))
            | ExprKind::Literal(Literal::Float(_))
            | ExprKind::Literal(Literal::Bool(_))
            | ExprKind::Literal(Literal::String(_)) => expr,

            // Binary operation - atomize both operands
            ExprKind::BinOp { op, lhs, rhs } => {
                let lhs = self.normalize_expr(*lhs, bindings);
                let lhs = self.atomize(lhs, bindings);
                let rhs = self.normalize_expr(*rhs, bindings);
                let rhs = self.atomize(rhs, bindings);
                Expr::new(
                    id,
                    ty,
                    ExprKind::BinOp {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    },
                    span,
                )
            }

            // Unary operation - atomize operand
            ExprKind::UnaryOp { op, operand } => {
                let operand = self.normalize_expr(*operand, bindings);
                let operand = self.atomize(operand, bindings);
                Expr::new(
                    id,
                    ty,
                    ExprKind::UnaryOp {
                        op,
                        operand: Box::new(operand),
                    },
                    span,
                )
            }

            // Function call - atomize all args
            ExprKind::Call { func, args } => {
                let args = args
                    .into_iter()
                    .map(|a| {
                        let a = self.normalize_expr(a, bindings);
                        self.atomize(a, bindings)
                    })
                    .collect();
                Expr::new(id, ty, ExprKind::Call { func, args }, span)
            }

            // Intrinsic - atomize all args
            ExprKind::Intrinsic { name, args } => {
                let args = args
                    .into_iter()
                    .map(|a| {
                        let a = self.normalize_expr(a, bindings);
                        self.atomize(a, bindings)
                    })
                    .collect();
                Expr::new(id, ty, ExprKind::Intrinsic { name, args }, span)
            }

            // Tuple literal - handle empty and non-empty cases
            ExprKind::Literal(Literal::Tuple(ref elems)) if elems.is_empty() => {
                // Empty tuples are atomic
                expr
            }
            ExprKind::Literal(Literal::Tuple(elems)) => {
                let elems = elems
                    .into_iter()
                    .map(|e| {
                        let e = self.normalize_expr(e, bindings);
                        self.atomize(e, bindings)
                    })
                    .collect();
                Expr::new(id, ty, ExprKind::Literal(Literal::Tuple(elems)), span)
            }

            // Array literal - atomize all elements
            ExprKind::Literal(Literal::Array(elems)) => {
                let elems = elems
                    .into_iter()
                    .map(|e| {
                        let e = self.normalize_expr(e, bindings);
                        self.atomize(e, bindings)
                    })
                    .collect();
                Expr::new(id, ty, ExprKind::Literal(Literal::Array(elems)), span)
            }

            // Vector literal - atomize all elements
            ExprKind::Literal(Literal::Vector(elems)) => {
                let elems = elems
                    .into_iter()
                    .map(|e| {
                        let e = self.normalize_expr(e, bindings);
                        self.atomize(e, bindings)
                    })
                    .collect();
                Expr::new(id, ty, ExprKind::Literal(Literal::Vector(elems)), span)
            }

            // Matrix literal - atomize all elements in all rows
            ExprKind::Literal(Literal::Matrix(rows)) => {
                let rows = rows
                    .into_iter()
                    .map(|row| {
                        row.into_iter()
                            .map(|e| {
                                let e = self.normalize_expr(e, bindings);
                                self.atomize(e, bindings)
                            })
                            .collect()
                    })
                    .collect();
                Expr::new(id, ty, ExprKind::Literal(Literal::Matrix(rows)), span)
            }

            // Materialize - atomize inner
            ExprKind::Materialize(inner) => {
                let inner = self.normalize_expr(*inner, bindings);
                let inner = self.atomize(inner, bindings);
                Expr::new(id, ty, ExprKind::Materialize(Box::new(inner)), span)
            }

            // Let binding - normalize value and body
            ExprKind::Let {
                name,
                binding_id,
                value,
                body,
            } => {
                // Value gets normalized with outer bindings (value is evaluated before name is bound)
                let value = self.normalize_expr(*value, bindings);

                // Body gets its own scope - bindings from body may reference `name`
                let mut body_bindings = Vec::new();
                let body = self.normalize_expr(*body, &mut body_bindings);
                let body = self.wrap_bindings(body, body_bindings);

                Expr::new(
                    id,
                    ty,
                    ExprKind::Let {
                        name,
                        binding_id,
                        value: Box::new(value),
                        body: Box::new(body),
                    },
                    span,
                )
            }

            // If expression - atomize condition, normalize branches with fresh scopes
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond = self.normalize_expr(*cond, bindings);
                let cond = self.atomize(cond, bindings);

                // Branches get their own binding scopes
                let mut then_bindings = Vec::new();
                let then_branch = self.normalize_expr(*then_branch, &mut then_bindings);
                let then_branch = self.wrap_bindings(then_branch, then_bindings);

                let mut else_bindings = Vec::new();
                let else_branch = self.normalize_expr(*else_branch, &mut else_bindings);
                let else_branch = self.wrap_bindings(else_branch, else_bindings);

                Expr::new(
                    id,
                    ty,
                    ExprKind::If {
                        cond: Box::new(cond),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    },
                    span,
                )
            }

            // Loop - atomize init, normalize init_bindings and body with fresh scopes
            ExprKind::Loop {
                loop_var,
                init,
                init_bindings: loop_init_bindings,
                kind,
                body,
            } => {
                // Init gets atomized
                let init = self.normalize_expr(*init, bindings);
                let init = self.atomize(init, bindings);

                // Init bindings get normalized
                let loop_init_bindings = loop_init_bindings
                    .into_iter()
                    .map(|(name, expr)| {
                        let mut init_b = Vec::new();
                        let expr = self.normalize_expr(expr, &mut init_b);
                        let expr = self.wrap_bindings(expr, init_b);
                        (name, expr)
                    })
                    .collect();

                // Loop kind gets normalized
                let kind = match kind {
                    LoopKind::While { cond } => {
                        let mut cond_bindings = Vec::new();
                        let cond = self.normalize_expr(*cond, &mut cond_bindings);
                        let cond = self.atomize(cond, &mut cond_bindings);
                        let cond = self.wrap_bindings(cond, cond_bindings);
                        LoopKind::While { cond: Box::new(cond) }
                    }
                    LoopKind::ForRange { var, bound } => {
                        let mut bound_bindings = Vec::new();
                        let bound = self.normalize_expr(*bound, &mut bound_bindings);
                        let bound = self.atomize(bound, &mut bound_bindings);
                        let bound = self.wrap_bindings(bound, bound_bindings);
                        LoopKind::ForRange {
                            var,
                            bound: Box::new(bound),
                        }
                    }
                    LoopKind::For { var, iter } => {
                        let mut iter_bindings = Vec::new();
                        let iter = self.normalize_expr(*iter, &mut iter_bindings);
                        let iter = self.atomize(iter, &mut iter_bindings);
                        let iter = self.wrap_bindings(iter, iter_bindings);
                        LoopKind::For {
                            var,
                            iter: Box::new(iter),
                        }
                    }
                };

                // Body gets its own scope
                let mut body_bindings = Vec::new();
                let body = self.normalize_expr(*body, &mut body_bindings);
                let body = self.wrap_bindings(body, body_bindings);

                Expr::new(
                    id,
                    ty,
                    ExprKind::Loop {
                        loop_var,
                        init: Box::new(init),
                        init_bindings: loop_init_bindings,
                        kind,
                        body: Box::new(body),
                    },
                    span,
                )
            }

            // Attributed - normalize inner
            ExprKind::Attributed {
                attributes,
                expr: inner,
            } => {
                let inner = self.normalize_expr(*inner, bindings);
                Expr::new(
                    id,
                    ty,
                    ExprKind::Attributed {
                        attributes,
                        expr: Box::new(inner),
                    },
                    span,
                )
            }

            // Closure - normalize and atomize captures
            ExprKind::Closure {
                lambda_name,
                captures,
            } => {
                let captures = captures
                    .into_iter()
                    .map(|c| {
                        let c = self.normalize_expr(c, bindings);
                        self.atomize(c, bindings)
                    })
                    .collect();
                Expr::new(
                    id,
                    ty,
                    ExprKind::Closure {
                        lambda_name,
                        captures,
                    },
                    span,
                )
            }

            // Range - normalize and atomize start, step, end
            ExprKind::Range {
                start,
                step,
                end,
                kind,
            } => {
                let start = self.normalize_expr(*start, bindings);
                let start = self.atomize(start, bindings);
                let step = step.map(|s| {
                    let s = self.normalize_expr(*s, bindings);
                    self.atomize(s, bindings)
                });
                let end = self.normalize_expr(*end, bindings);
                let end = self.atomize(end, bindings);
                Expr::new(
                    id,
                    ty,
                    ExprKind::Range {
                        start: Box::new(start),
                        step: step.map(Box::new),
                        end: Box::new(end),
                        kind,
                    },
                    span,
                )
            }
        }
    }

    /// Ensure an expression is atomic, creating a temp binding if needed.
    fn atomize(&mut self, expr: Expr, bindings: &mut Vec<Binding>) -> Expr {
        if is_atomic(&expr) {
            expr
        } else {
            let name = self.fresh_temp_name();
            let binding_id = self.fresh_binding_id();
            let ty = expr.ty.clone();
            let span = expr.span;
            bindings.push((name.clone(), binding_id, expr));
            let node_id = self.node_counter.next();
            Expr::new(node_id, ty, ExprKind::Var(name), span)
        }
    }

    /// Wrap collected bindings around an expression as nested Lets.
    fn wrap_bindings(&mut self, mut expr: Expr, bindings: Vec<Binding>) -> Expr {
        for (name, binding_id, value) in bindings.into_iter().rev() {
            let node_id = self.node_counter.next();
            let ty = expr.ty.clone();
            let span = expr.span;
            expr = Expr::new(
                node_id,
                ty,
                ExprKind::Let {
                    name,
                    binding_id,
                    value: Box::new(value),
                    body: Box::new(expr),
                },
                span,
            );
        }
        expr
    }
}

/// Check if an expression is atomic (can appear as an operand).
pub fn is_atomic(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Var(_) | ExprKind::Unit => true,
        ExprKind::Closure { .. } => true, // Closures are atomic values
        ExprKind::Literal(Literal::Tuple(elems)) => elems.is_empty(),
        ExprKind::Literal(lit) => is_scalar_literal(lit),
        _ => false,
    }
}

/// Check if a literal is a scalar (not a container).
fn is_scalar_literal(lit: &Literal) -> bool {
    matches!(
        lit,
        Literal::Int(_) | Literal::Float(_) | Literal::Bool(_) | Literal::String(_)
    )
}

/// Find the maximum binding ID in a program.
/// Used to initialize the normalizer with a safe starting ID.
pub fn find_max_binding_id(program: &Program) -> u64 {
    let mut max_id = 0;
    for def in &program.defs {
        match def {
            Def::Function { body, .. } | Def::Constant { body, .. } | Def::EntryPoint { body, .. } => {
                max_id = max_id.max(find_max_binding_id_in_expr(body));
            }
            Def::Uniform { .. } => {}
            Def::Storage { .. } => {}
        }
    }
    max_id
}

fn find_max_binding_id_in_expr(expr: &Expr) -> u64 {
    match &expr.kind {
        ExprKind::Let {
            binding_id,
            value,
            body,
            ..
        } => {
            let v = find_max_binding_id_in_expr(value);
            let b = find_max_binding_id_in_expr(body);
            (*binding_id).max(v).max(b)
        }
        ExprKind::BinOp { lhs, rhs, .. } => {
            find_max_binding_id_in_expr(lhs).max(find_max_binding_id_in_expr(rhs))
        }
        ExprKind::UnaryOp { operand, .. } => find_max_binding_id_in_expr(operand),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => find_max_binding_id_in_expr(cond)
            .max(find_max_binding_id_in_expr(then_branch))
            .max(find_max_binding_id_in_expr(else_branch)),
        ExprKind::Loop {
            init,
            init_bindings,
            kind,
            body,
            ..
        } => {
            let mut max = find_max_binding_id_in_expr(init);
            for (_, e) in init_bindings {
                max = max.max(find_max_binding_id_in_expr(e));
            }
            max = match kind {
                LoopKind::While { cond } => max.max(find_max_binding_id_in_expr(cond)),
                LoopKind::ForRange { bound, .. } => max.max(find_max_binding_id_in_expr(bound)),
                LoopKind::For { iter, .. } => max.max(find_max_binding_id_in_expr(iter)),
            };
            max.max(find_max_binding_id_in_expr(body))
        }
        ExprKind::Call { args, .. } | ExprKind::Intrinsic { args, .. } => {
            args.iter().map(find_max_binding_id_in_expr).max().unwrap_or(0)
        }
        ExprKind::Literal(lit) => match lit {
            Literal::Tuple(elems) | Literal::Array(elems) | Literal::Vector(elems) => {
                elems.iter().map(find_max_binding_id_in_expr).max().unwrap_or(0)
            }
            Literal::Matrix(rows) => {
                rows.iter().flat_map(|row| row.iter()).map(find_max_binding_id_in_expr).max().unwrap_or(0)
            }
            _ => 0,
        },
        ExprKind::Materialize(inner) => find_max_binding_id_in_expr(inner),
        ExprKind::Attributed { expr, .. } => find_max_binding_id_in_expr(expr),
        ExprKind::Closure { captures, .. } => {
            captures.iter().map(find_max_binding_id_in_expr).max().unwrap_or(0)
        }
        ExprKind::Range { start, step, end, .. } => {
            let start_max = find_max_binding_id_in_expr(start);
            let step_max = step.as_ref().map(|s| find_max_binding_id_in_expr(s)).unwrap_or(0);
            let end_max = find_max_binding_id_in_expr(end);
            start_max.max(step_max).max(end_max)
        }
        ExprKind::Var(_) | ExprKind::Unit => 0,
    }
}

/// Normalize a MIR program to A-normal form.
pub fn normalize_program(program: Program, node_counter: NodeCounter) -> (Program, NodeCounter) {
    let max_id = find_max_binding_id(&program);
    let mut normalizer = Normalizer::new(max_id + 1, node_counter);
    let program = normalizer.normalize_program(program);
    (program, normalizer.node_counter)
}
