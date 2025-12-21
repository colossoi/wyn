//! Materialize hoisting pass for MIR.
//!
//! This pass finds duplicate `Materialize` expressions within the same scope
//! (particularly in both branches of an if-then-else) and hoists them to a
//! single let binding, eliminating redundant OpStore/OpLoad in SPIR-V.
//!
//! Example transformation:
//! ```text
//! if c then @index(@materialize(f x), i) else @index(@materialize(f x), i)
//! ```
//! becomes:
//! ```text
//! let __mat_0 = @materialize(f x) in
//! if c then @index(__mat_0, i) else @index(__mat_0, i)
//! ```

use crate::mir::{Body, Def, Expr, ExprId, Program};
use std::collections::HashSet;

/// Hoist duplicate materializations in a program.
pub fn hoist_materializations(program: Program) -> Program {
    Program {
        defs: program.defs.into_iter().map(hoist_in_def).collect(),
        lambda_registry: program.lambda_registry,
    }
}

fn hoist_in_def(def: Def) -> Def {
    match def {
        Def::Function {
            id,
            name,
            params,
            ret_type,
            scheme,
            attributes,
            mut body,
            span,
        } => {
            hoist_in_body(&mut body);
            Def::Function {
                id,
                name,
                params,
                ret_type,
                scheme,
                attributes,
                body,
                span,
            }
        }
        Def::Constant {
            id,
            name,
            ty,
            attributes,
            mut body,
            span,
        } => {
            hoist_in_body(&mut body);
            Def::Constant {
                id,
                name,
                ty,
                attributes,
                body,
                span,
            }
        }
        Def::EntryPoint {
            id,
            name,
            execution_model,
            inputs,
            outputs,
            mut body,
            span,
        } => {
            hoist_in_body(&mut body);
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
        other => other,
    }
}

/// Process a body, hoisting common materializations from if branches.
fn hoist_in_body(body: &mut Body) {
    // Find all If expressions
    let if_exprs: Vec<ExprId> = body
        .exprs
        .iter()
        .enumerate()
        .filter_map(
            |(idx, expr)| {
                if matches!(expr, Expr::If { .. }) { Some(ExprId(idx as u32)) } else { None }
            },
        )
        .collect();

    // Process each If expression
    for if_id in if_exprs {
        hoist_in_if(body, if_id);
    }
}

/// Try to hoist common materializations from an If expression.
fn hoist_in_if(body: &mut Body, if_id: ExprId) {
    // Get the then/else branch IDs
    let (then_id, else_id) = match body.get_expr(if_id) {
        Expr::If { then_, else_, .. } => (*then_, *else_),
        _ => return,
    };

    // Collect Materialize expressions reachable from each branch
    let then_mats = collect_materializations(body, then_id);
    let else_mats = collect_materializations(body, else_id);

    // Find common materializations (by structural equality of inner expression)
    let common = find_common_materializations(body, &then_mats, &else_mats);

    // Hoist each common materialization
    for mat_id in common {
        body.hoist_before(mat_id, if_id, None);
    }
}

/// Collect all Materialize expression IDs reachable from an expression.
fn collect_materializations(body: &Body, root: ExprId) -> Vec<ExprId> {
    let mut result = Vec::new();
    let mut visited = HashSet::new();
    collect_materializations_rec(body, root, &mut result, &mut visited);
    result
}

fn collect_materializations_rec(
    body: &Body,
    expr_id: ExprId,
    result: &mut Vec<ExprId>,
    visited: &mut HashSet<ExprId>,
) {
    if !visited.insert(expr_id) {
        return;
    }

    let expr = body.get_expr(expr_id);

    if matches!(expr, Expr::Materialize(_)) {
        result.push(expr_id);
    }

    // Recurse into children
    match expr {
        Expr::Materialize(inner) => {
            collect_materializations_rec(body, *inner, result, visited);
        }
        Expr::BinOp { lhs, rhs, .. } => {
            collect_materializations_rec(body, *lhs, result, visited);
            collect_materializations_rec(body, *rhs, result, visited);
        }
        Expr::UnaryOp { operand, .. } => {
            collect_materializations_rec(body, *operand, result, visited);
        }
        Expr::If { cond, then_, else_ } => {
            collect_materializations_rec(body, *cond, result, visited);
            collect_materializations_rec(body, *then_, result, visited);
            collect_materializations_rec(body, *else_, result, visited);
        }
        Expr::Let {
            rhs, body: let_body, ..
        } => {
            collect_materializations_rec(body, *rhs, result, visited);
            collect_materializations_rec(body, *let_body, result, visited);
        }
        Expr::Loop {
            init,
            init_bindings,
            body: loop_body,
            ..
        } => {
            collect_materializations_rec(body, *init, result, visited);
            for (_, binding_expr) in init_bindings {
                collect_materializations_rec(body, *binding_expr, result, visited);
            }
            collect_materializations_rec(body, *loop_body, result, visited);
        }
        Expr::Call { args, .. } | Expr::Intrinsic { args, .. } => {
            for arg in args {
                collect_materializations_rec(body, *arg, result, visited);
            }
        }
        Expr::Closure { captures, .. } => {
            collect_materializations_rec(body, *captures, result, visited);
        }
        Expr::Tuple(elems) | Expr::Array(elems) | Expr::Vector(elems) => {
            for elem in elems {
                collect_materializations_rec(body, *elem, result, visited);
            }
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for elem in row {
                    collect_materializations_rec(body, *elem, result, visited);
                }
            }
        }
        Expr::Range { start, step, end, .. } => {
            collect_materializations_rec(body, *start, result, visited);
            if let Some(s) = step {
                collect_materializations_rec(body, *s, result, visited);
            }
            collect_materializations_rec(body, *end, result, visited);
        }
        Expr::Attributed { expr, .. } => {
            collect_materializations_rec(body, *expr, result, visited);
        }
        Expr::OwnedSlice { data, len } => {
            collect_materializations_rec(body, *data, result, visited);
            collect_materializations_rec(body, *len, result, visited);
        }
        Expr::BorrowedSlice { base, offset, len } => {
            collect_materializations_rec(body, *base, result, visited);
            collect_materializations_rec(body, *offset, result, visited);
            collect_materializations_rec(body, *len, result, visited);
        }
        // Atoms have no children
        Expr::Local(_)
        | Expr::Global(_)
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Bool(_)
        | Expr::Unit
        | Expr::String(_) => {}
    }
}

/// Find materializations that appear in both lists with structurally equal inner expressions.
fn find_common_materializations(body: &Body, then_mats: &[ExprId], else_mats: &[ExprId]) -> Vec<ExprId> {
    let mut common = Vec::new();

    for &then_mat in then_mats {
        // Get the inner expression of the then-branch Materialize
        let then_inner = match body.get_expr(then_mat) {
            Expr::Materialize(inner) => *inner,
            _ => continue,
        };

        for &else_mat in else_mats {
            // Get the inner expression of the else-branch Materialize
            let else_inner = match body.get_expr(else_mat) {
                Expr::Materialize(inner) => *inner,
                _ => continue,
            };

            // Check structural equality
            if exprs_equal(body, then_inner, else_inner) {
                // Check we haven't already added an equivalent one
                let already_added = common.iter().any(|&existing| {
                    let existing_inner = match body.get_expr(existing) {
                        Expr::Materialize(inner) => *inner,
                        _ => return false,
                    };
                    exprs_equal(body, existing_inner, then_inner)
                });

                if !already_added {
                    common.push(then_mat);
                }
            }
        }
    }

    common
}

/// Check if two expressions are structurally equal (ignoring IDs and spans).
fn exprs_equal(body: &Body, a: ExprId, b: ExprId) -> bool {
    // Types should match
    if body.get_type(a) != body.get_type(b) {
        return false;
    }

    match (body.get_expr(a), body.get_expr(b)) {
        (Expr::Local(la), Expr::Local(lb)) => la == lb,
        (Expr::Global(na), Expr::Global(nb)) => na == nb,
        (Expr::Int(a), Expr::Int(b)) => a == b,
        (Expr::Float(a), Expr::Float(b)) => a == b,
        (Expr::Bool(a), Expr::Bool(b)) => a == b,
        (Expr::String(a), Expr::String(b)) => a == b,
        (Expr::Unit, Expr::Unit) => true,

        (
            Expr::BinOp {
                op: opa,
                lhs: la,
                rhs: ra,
            },
            Expr::BinOp {
                op: opb,
                lhs: lb,
                rhs: rb,
            },
        ) => opa == opb && exprs_equal(body, *la, *lb) && exprs_equal(body, *ra, *rb),
        (Expr::UnaryOp { op: opa, operand: oa }, Expr::UnaryOp { op: opb, operand: ob }) => {
            opa == opb && exprs_equal(body, *oa, *ob)
        }
        (Expr::Call { func: fa, args: aa }, Expr::Call { func: fb, args: ab }) => {
            fa == fb
                && aa.len() == ab.len()
                && aa.iter().zip(ab.iter()).all(|(x, y)| exprs_equal(body, *x, *y))
        }
        (Expr::Intrinsic { name: na, args: aa }, Expr::Intrinsic { name: nb, args: ab }) => {
            na == nb
                && aa.len() == ab.len()
                && aa.iter().zip(ab.iter()).all(|(x, y)| exprs_equal(body, *x, *y))
        }
        (Expr::Materialize(ia), Expr::Materialize(ib)) => exprs_equal(body, *ia, *ib),

        (Expr::Tuple(ea), Expr::Tuple(eb))
        | (Expr::Array(ea), Expr::Array(eb))
        | (Expr::Vector(ea), Expr::Vector(eb)) => {
            ea.len() == eb.len() && ea.iter().zip(eb.iter()).all(|(x, y)| exprs_equal(body, *x, *y))
        }

        (Expr::Matrix(ra), Expr::Matrix(rb)) => {
            ra.len() == rb.len()
                && ra.iter().zip(rb.iter()).all(|(rowa, rowb)| {
                    rowa.len() == rowb.len()
                        && rowa.iter().zip(rowb.iter()).all(|(x, y)| exprs_equal(body, *x, *y))
                })
        }

        // For simplicity, don't consider if/let/loop as equal even if structurally same
        // (they're complex and unlikely to be duplicated materializations anyway)
        _ => false,
    }
}
