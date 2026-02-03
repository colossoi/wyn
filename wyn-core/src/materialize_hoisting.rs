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

use crate::mir::transform::rebuild_body_ordered;
use crate::mir::{Block, Body, Def, Expr, ExprId, LocalId, LoopKind, Program};
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
            attributes,
            mut body,
            span,
            dps_output,
        } => {
            hoist_in_body(&mut body);
            Def::Function {
                id,
                name,
                params,
                ret_type,
                attributes,
                body,
                span,
                dps_output,
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

    // Find all Loop expressions
    let loop_exprs: Vec<ExprId> = body
        .exprs
        .iter()
        .enumerate()
        .filter_map(
            |(idx, expr)| {
                if matches!(expr, Expr::Loop { .. }) { Some(ExprId(idx as u32)) } else { None }
            },
        )
        .collect();

    // Process each Loop expression
    for loop_id in loop_exprs {
        hoist_in_loop(body, loop_id);
    }

    // Use the transform primitive to rebuild body with proper ordering
    *body = rebuild_body_ordered(body);
}

/// Try to hoist common materializations from an If expression.
fn hoist_in_if(body: &mut Body, if_id: ExprId) {
    // Get the then/else branch IDs
    let (then_block, else_block) = match body.get_expr(if_id) {
        Expr::If { then_, else_, .. } => (then_.clone(), else_.clone()),
        _ => return,
    };

    // Collect Materialize expressions reachable from each branch
    let then_mats = collect_materializations(body, then_block.result);
    let else_mats = collect_materializations(body, else_block.result);

    // Also collect from block stmts
    let mut then_mats_all = then_mats;
    for stmt in &then_block.stmts {
        then_mats_all.extend(collect_materializations(body, stmt.rhs));
    }
    let mut else_mats_all = else_mats;
    for stmt in &else_block.stmts {
        else_mats_all.extend(collect_materializations(body, stmt.rhs));
    }

    // Find common materializations (by structural equality of inner expression)
    let common = find_common_materializations(body, &then_mats_all, &else_mats_all);

    // Hoist each common materialization and replace its counterpart
    for (then_mat, else_mat) in common {
        // Hoist the then-branch materialization to a statement
        let local_id = body.hoist_to_stmt(then_mat, None);
        // Replace the else-branch equivalent with a reference to the hoisted local
        *body.get_expr_mut(else_mat) = Expr::Local(local_id);
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
            collect_materializations_in_block(body, then_, result, visited);
            collect_materializations_in_block(body, else_, result, visited);
        }
        Expr::Loop {
            init,
            init_bindings,
            kind,
            body: loop_body,
            ..
        } => {
            collect_materializations_rec(body, *init, result, visited);
            for (_, expr) in init_bindings {
                collect_materializations_rec(body, *expr, result, visited);
            }
            match kind {
                LoopKind::For { iter, .. } => {
                    collect_materializations_rec(body, *iter, result, visited);
                }
                LoopKind::ForRange { bound, .. } => {
                    collect_materializations_rec(body, *bound, result, visited);
                }
                LoopKind::While { cond } => {
                    collect_materializations_rec(body, *cond, result, visited);
                }
            }
            collect_materializations_in_block(body, loop_body, result, visited);
        }
        Expr::Tuple(elems) | Expr::Vector(elems) => {
            for elem in elems {
                collect_materializations_rec(body, *elem, result, visited);
            }
        }
        Expr::Call { args, .. } | Expr::Intrinsic { args, .. } => {
            for arg in args {
                collect_materializations_rec(body, *arg, result, visited);
            }
        }
        Expr::Load { ptr } => {
            collect_materializations_rec(body, *ptr, result, visited);
        }
        Expr::Store { ptr, value } => {
            collect_materializations_rec(body, *ptr, result, visited);
            collect_materializations_rec(body, *value, result, visited);
        }
        Expr::Attributed { expr, .. } => {
            collect_materializations_rec(body, *expr, result, visited);
        }
        _ => {}
    }
}

fn collect_materializations_in_block(
    body: &Body,
    block: &Block,
    result: &mut Vec<ExprId>,
    visited: &mut HashSet<ExprId>,
) {
    for stmt in &block.stmts {
        collect_materializations_rec(body, stmt.rhs, result, visited);
    }
    collect_materializations_rec(body, block.result, result, visited);
}

/// Find pairs of Materialize expressions that are structurally identical.
fn find_common_materializations(
    body: &Body,
    then_mats: &[ExprId],
    else_mats: &[ExprId],
) -> Vec<(ExprId, ExprId)> {
    let mut common = Vec::new();
    let mut used_else: HashSet<ExprId> = HashSet::new();

    for &then_mat in then_mats {
        for &else_mat in else_mats {
            if used_else.contains(&else_mat) {
                continue;
            }
            if structurally_equal(body, then_mat, else_mat) {
                common.push((then_mat, else_mat));
                used_else.insert(else_mat);
                break;
            }
        }
    }

    common
}

/// Check if two expressions are structurally equal.
fn structurally_equal(body: &Body, a: ExprId, b: ExprId) -> bool {
    if a == b {
        return true;
    }

    let expr_a = body.get_expr(a);
    let expr_b = body.get_expr(b);

    match (expr_a, expr_b) {
        (Expr::Local(id_a), Expr::Local(id_b)) => id_a == id_b,
        (Expr::Global(name_a), Expr::Global(name_b)) => name_a == name_b,
        (Expr::Int(s_a), Expr::Int(s_b)) => s_a == s_b,
        (Expr::Float(s_a), Expr::Float(s_b)) => s_a == s_b,
        (Expr::Bool(b_a), Expr::Bool(b_b)) => b_a == b_b,
        (Expr::Unit, Expr::Unit) => true,
        (Expr::Materialize(inner_a), Expr::Materialize(inner_b)) => {
            structurally_equal(body, *inner_a, *inner_b)
        }
        (
            Expr::BinOp {
                op: op_a,
                lhs: lhs_a,
                rhs: rhs_a,
            },
            Expr::BinOp {
                op: op_b,
                lhs: lhs_b,
                rhs: rhs_b,
            },
        ) => {
            op_a == op_b
                && structurally_equal(body, *lhs_a, *lhs_b)
                && structurally_equal(body, *rhs_a, *rhs_b)
        }
        (
            Expr::UnaryOp {
                op: op_a,
                operand: operand_a,
            },
            Expr::UnaryOp {
                op: op_b,
                operand: operand_b,
            },
        ) => op_a == op_b && structurally_equal(body, *operand_a, *operand_b),
        (
            Expr::Call {
                func: func_a,
                args: args_a,
            },
            Expr::Call {
                func: func_b,
                args: args_b,
            },
        ) => {
            func_a == func_b
                && args_a.len() == args_b.len()
                && args_a.iter().zip(args_b.iter()).all(|(a, b)| structurally_equal(body, *a, *b))
        }
        (
            Expr::Intrinsic {
                name: name_a,
                args: args_a,
            },
            Expr::Intrinsic {
                name: name_b,
                args: args_b,
            },
        ) => {
            name_a == name_b
                && args_a.len() == args_b.len()
                && args_a.iter().zip(args_b.iter()).all(|(a, b)| structurally_equal(body, *a, *b))
        }
        (Expr::Tuple(elems_a), Expr::Tuple(elems_b)) => {
            elems_a.len() == elems_b.len()
                && elems_a.iter().zip(elems_b.iter()).all(|(a, b)| structurally_equal(body, *a, *b))
        }
        (Expr::Vector(elems_a), Expr::Vector(elems_b)) => {
            elems_a.len() == elems_b.len()
                && elems_a.iter().zip(elems_b.iter()).all(|(a, b)| structurally_equal(body, *a, *b))
        }
        _ => false,
    }
}

/// Try to hoist loop-invariant materializations out of a loop.
fn hoist_in_loop(body: &mut Body, loop_id: ExprId) {
    // Get the loop body
    let loop_body = match body.get_expr(loop_id) {
        Expr::Loop { body: loop_body, .. } => loop_body.clone(),
        _ => return,
    };

    // Collect Materialize expressions in the loop body
    let mut mats = Vec::new();
    for stmt in &loop_body.stmts {
        mats.extend(collect_materializations(body, stmt.rhs));
    }
    mats.extend(collect_materializations(body, loop_body.result));

    // Get loop variables that are modified in the loop
    let loop_vars = match body.get_expr(loop_id) {
        Expr::Loop {
            loop_var,
            init_bindings,
            kind,
            ..
        } => {
            let mut vars: HashSet<LocalId> = HashSet::new();
            vars.insert(*loop_var);
            for (local, _) in init_bindings {
                vars.insert(*local);
            }
            match kind {
                LoopKind::For { var, .. } | LoopKind::ForRange { var, .. } => {
                    vars.insert(*var);
                }
                _ => {}
            }
            vars
        }
        _ => return,
    };

    // Hoist materializations that don't depend on loop variables
    for mat_id in mats {
        let mat_deps = crate::mir::transform::expr_reads(body, mat_id);
        let depends_on_loop = mat_deps.iter().any(|local| loop_vars.contains(local));

        if !depends_on_loop {
            // Safe to hoist
            let local_id = body.hoist_to_stmt(mat_id, None);
            // The original expression was replaced with Local(local_id) by hoist_to_stmt
            let _ = local_id;
        }
    }
}
