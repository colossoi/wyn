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

use crate::mir::{ArrayBacking, Block, Body, Def, Expr, ExprId, LocalId, LoopKind, Program};
use std::collections::{HashMap, HashSet};

/// Hoist duplicate materializations in a program.
pub fn hoist_materializations(program: Program) -> Program {
    Program {
        defs: program.defs.into_iter().map(hoist_in_def).collect(),
        lambda_registry: program.lambda_registry,
    }
}

// =============================================================================
// Expression reordering
// =============================================================================

/// Reorder expressions in a body to restore proper dependency order.
/// After hoisting, expressions may be out of order (children after parents).
/// This pass does a post-order traversal and rebuilds the arena.
fn reorder_body(body: &mut Body) {
    // Build new body with proper order via post-order traversal
    let mut new_body = Body::new();

    // Copy locals
    for local in &body.locals {
        new_body.alloc_local(local.clone());
    }

    // Map from old ExprId to new ExprId
    let mut id_map: HashMap<ExprId, ExprId> = HashMap::new();

    // First, process statement RHS expressions (they may not be reachable from root)
    for stmt in body.iter_stmts() {
        reorder_expr(body, stmt.rhs, &mut new_body, &mut id_map);
    }

    // Then process expressions in post-order from root
    reorder_expr(body, body.root, &mut new_body, &mut id_map);

    // Copy statements, remapping their ExprIds
    for stmt in body.iter_stmts() {
        let new_rhs = id_map[&stmt.rhs];
        new_body.push_stmt(stmt.local, new_rhs);
    }

    // Update root
    new_body.root = id_map[&body.root];

    *body = new_body;
}

/// Recursively reorder an expression tree, ensuring children come before parents.
fn reorder_expr(
    old_body: &Body,
    old_id: ExprId,
    new_body: &mut Body,
    id_map: &mut HashMap<ExprId, ExprId>,
) -> ExprId {
    // If already processed, return the new ID
    if let Some(&new_id) = id_map.get(&old_id) {
        return new_id;
    }

    let ty = old_body.get_type(old_id).clone();
    let span = old_body.get_span(old_id);
    let node_id = old_body.get_node_id(old_id);

    // Process children first (post-order), then allocate this expression
    let new_expr = match old_body.get_expr(old_id).clone() {
        // Atoms - no children
        Expr::Local(id) => Expr::Local(id),
        Expr::Global(name) => Expr::Global(name),
        Expr::Extern(linkage) => Expr::Extern(linkage),
        Expr::Int(s) => Expr::Int(s),
        Expr::Float(s) => Expr::Float(s),
        Expr::Bool(b) => Expr::Bool(b),
        Expr::Unit => Expr::Unit,
        Expr::String(s) => Expr::String(s),

        // Expressions with children
        Expr::BinOp { op, lhs, rhs } => {
            let new_lhs = reorder_expr(old_body, lhs, new_body, id_map);
            let new_rhs = reorder_expr(old_body, rhs, new_body, id_map);
            Expr::BinOp {
                op,
                lhs: new_lhs,
                rhs: new_rhs,
            }
        }
        Expr::UnaryOp { op, operand } => {
            let new_operand = reorder_expr(old_body, operand, new_body, id_map);
            Expr::UnaryOp {
                op,
                operand: new_operand,
            }
        }
        Expr::Tuple(elems) => {
            let new_elems: Vec<_> =
                elems.iter().map(|e| reorder_expr(old_body, *e, new_body, id_map)).collect();
            Expr::Tuple(new_elems)
        }
        Expr::Vector(elems) => {
            let new_elems: Vec<_> =
                elems.iter().map(|e| reorder_expr(old_body, *e, new_body, id_map)).collect();
            Expr::Vector(new_elems)
        }
        Expr::Matrix(rows) => {
            let new_rows: Vec<Vec<_>> = rows
                .iter()
                .map(|row| row.iter().map(|e| reorder_expr(old_body, *e, new_body, id_map)).collect())
                .collect();
            Expr::Matrix(new_rows)
        }
        Expr::Array { backing, size } => {
            let new_size = reorder_expr(old_body, size, new_body, id_map);
            let new_backing = match backing {
                ArrayBacking::Literal(elems) => {
                    let new_elems: Vec<_> =
                        elems.iter().map(|e| reorder_expr(old_body, *e, new_body, id_map)).collect();
                    ArrayBacking::Literal(new_elems)
                }
                ArrayBacking::Range { start, step, kind } => {
                    let new_start = reorder_expr(old_body, start, new_body, id_map);
                    let new_step = step.map(|s| reorder_expr(old_body, s, new_body, id_map));
                    ArrayBacking::Range {
                        start: new_start,
                        step: new_step,
                        kind,
                    }
                }
            };
            Expr::Array {
                backing: new_backing,
                size: new_size,
            }
        }
        Expr::If { cond, then_, else_ } => {
            let new_cond = reorder_expr(old_body, cond, new_body, id_map);
            let new_then = reorder_expr(old_body, then_.result, new_body, id_map);
            let new_else = reorder_expr(old_body, else_.result, new_body, id_map);
            Expr::If {
                cond: new_cond,
                then_: Block::new(new_then),
                else_: Block::new(new_else),
            }
        }
        Expr::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body,
        } => {
            let new_init = reorder_expr(old_body, init, new_body, id_map);
            let new_bindings: Vec<_> = init_bindings
                .iter()
                .map(|(local, expr)| (*local, reorder_expr(old_body, *expr, new_body, id_map)))
                .collect();
            let new_kind = match kind {
                LoopKind::For { var, iter } => {
                    let new_iter = reorder_expr(old_body, iter, new_body, id_map);
                    LoopKind::For { var, iter: new_iter }
                }
                LoopKind::ForRange { var, bound } => {
                    let new_bound = reorder_expr(old_body, bound, new_body, id_map);
                    LoopKind::ForRange {
                        var,
                        bound: new_bound,
                    }
                }
                LoopKind::While { cond } => {
                    let new_cond = reorder_expr(old_body, cond, new_body, id_map);
                    LoopKind::While { cond: new_cond }
                }
            };
            let new_loop_body = reorder_expr(old_body, body.result, new_body, id_map);
            Expr::Loop {
                loop_var,
                init: new_init,
                init_bindings: new_bindings,
                kind: new_kind,
                body: Block::new(new_loop_body),
            }
        }
        Expr::Call { func, args } => {
            let new_args: Vec<_> =
                args.iter().map(|a| reorder_expr(old_body, *a, new_body, id_map)).collect();
            Expr::Call { func, args: new_args }
        }
        Expr::Intrinsic { name, args } => {
            let new_args: Vec<_> =
                args.iter().map(|a| reorder_expr(old_body, *a, new_body, id_map)).collect();
            Expr::Intrinsic { name, args: new_args }
        }
        Expr::Materialize(inner) => {
            let new_inner = reorder_expr(old_body, inner, new_body, id_map);
            Expr::Materialize(new_inner)
        }
        Expr::Attributed { attributes, expr } => {
            let new_expr = reorder_expr(old_body, expr, new_body, id_map);
            Expr::Attributed {
                attributes,
                expr: new_expr,
            }
        }
        Expr::Load { ptr } => {
            let new_ptr = reorder_expr(old_body, ptr, new_body, id_map);
            Expr::Load { ptr: new_ptr }
        }
        Expr::Store { ptr, value } => {
            let new_ptr = reorder_expr(old_body, ptr, new_body, id_map);
            let new_value = reorder_expr(old_body, value, new_body, id_map);
            Expr::Store {
                ptr: new_ptr,
                value: new_value,
            }
        }
        Expr::StorageView {
            set,
            binding,
            offset,
            len,
        } => {
            let new_offset = reorder_expr(old_body, offset, new_body, id_map);
            let new_len = reorder_expr(old_body, len, new_body, id_map);
            Expr::StorageView {
                set,
                binding,
                offset: new_offset,
                len: new_len,
            }
        }
        Expr::SliceStorageView { view, start, len } => {
            let new_view = reorder_expr(old_body, view, new_body, id_map);
            let new_start = reorder_expr(old_body, start, new_body, id_map);
            let new_len = reorder_expr(old_body, len, new_body, id_map);
            Expr::SliceStorageView {
                view: new_view,
                start: new_start,
                len: new_len,
            }
        }
        Expr::StorageViewIndex { view, index } => {
            let new_view = reorder_expr(old_body, view, new_body, id_map);
            let new_index = reorder_expr(old_body, index, new_body, id_map);
            Expr::StorageViewIndex {
                view: new_view,
                index: new_index,
            }
        }
        Expr::StorageViewLen { view } => {
            let new_view = reorder_expr(old_body, view, new_body, id_map);
            Expr::StorageViewLen { view: new_view }
        }
    };

    let new_id = new_body.alloc_expr(new_expr, ty, span, node_id);
    id_map.insert(old_id, new_id);
    new_id
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

/// Process a body, hoisting common materializations from if branches and loops.
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

    // Reorder expressions to restore proper dependency order after hoisting
    reorder_body(body);
}

/// Try to hoist common materializations from an If expression.
fn hoist_in_if(body: &mut Body, if_id: ExprId) {
    // Get the then/else branch IDs
    let (then_id, else_id) = match body.get_expr(if_id) {
        Expr::If { then_, else_, .. } => (then_.result, else_.result),
        _ => return,
    };

    // Collect Materialize expressions reachable from each branch
    let then_mats = collect_materializations(body, then_id);
    let else_mats = collect_materializations(body, else_id);

    // Find common materializations (by structural equality of inner expression)
    let common = find_common_materializations(body, &then_mats, &else_mats);

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
            collect_materializations_rec(body, then_.result, result, visited);
            collect_materializations_rec(body, else_.result, result, visited);
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
            collect_materializations_rec(body, loop_body.result, result, visited);
        }
        Expr::Call { args, .. } | Expr::Intrinsic { args, .. } => {
            for arg in args {
                collect_materializations_rec(body, *arg, result, visited);
            }
        }
        Expr::Tuple(elems) | Expr::Vector(elems) => {
            for elem in elems {
                collect_materializations_rec(body, *elem, result, visited);
            }
        }
        Expr::Array { backing, size } => {
            collect_materializations_rec(body, *size, result, visited);
            match backing {
                ArrayBacking::Literal(elems) => {
                    for elem in elems {
                        collect_materializations_rec(body, *elem, result, visited);
                    }
                }
                ArrayBacking::Range { start, step, .. } => {
                    collect_materializations_rec(body, *start, result, visited);
                    if let Some(s) = step {
                        collect_materializations_rec(body, *s, result, visited);
                    }
                }
            }
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for elem in row {
                    collect_materializations_rec(body, *elem, result, visited);
                }
            }
        }
        Expr::Attributed { expr, .. } => {
            collect_materializations_rec(body, *expr, result, visited);
        }
        Expr::Load { ptr } => {
            collect_materializations_rec(body, *ptr, result, visited);
        }
        Expr::Store { ptr, value } => {
            collect_materializations_rec(body, *ptr, result, visited);
            collect_materializations_rec(body, *value, result, visited);
        }
        Expr::StorageView { offset, len, .. } => {
            collect_materializations_rec(body, *offset, result, visited);
            collect_materializations_rec(body, *len, result, visited);
        }
        Expr::SliceStorageView { view, start, len } => {
            collect_materializations_rec(body, *view, result, visited);
            collect_materializations_rec(body, *start, result, visited);
            collect_materializations_rec(body, *len, result, visited);
        }
        Expr::StorageViewIndex { view, index } => {
            collect_materializations_rec(body, *view, result, visited);
            collect_materializations_rec(body, *index, result, visited);
        }
        Expr::StorageViewLen { view } => {
            collect_materializations_rec(body, *view, result, visited);
        }
        // Atoms have no children
        Expr::Local(_)
        | Expr::Global(_)
        | Expr::Extern(_)
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Bool(_)
        | Expr::Unit
        | Expr::String(_) => {}
    }
}

/// Find materializations that appear in both lists with structurally equal inner expressions.
/// Returns pairs of (then_mat, else_mat) so both can be handled during hoisting.
fn find_common_materializations(
    body: &Body,
    then_mats: &[ExprId],
    else_mats: &[ExprId],
) -> Vec<(ExprId, ExprId)> {
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
                let already_added = common.iter().any(|(existing, _)| {
                    let existing_inner = match body.get_expr(*existing) {
                        Expr::Materialize(inner) => *inner,
                        _ => return false,
                    };
                    exprs_equal(body, existing_inner, then_inner)
                });

                if !already_added {
                    common.push((then_mat, else_mat));
                    break; // Found match for this then_mat, move to next
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

        (Expr::Tuple(ea), Expr::Tuple(eb)) | (Expr::Vector(ea), Expr::Vector(eb)) => {
            ea.len() == eb.len() && ea.iter().zip(eb.iter()).all(|(x, y)| exprs_equal(body, *x, *y))
        }

        (
            Expr::Array {
                backing: ba,
                size: sa,
            },
            Expr::Array {
                backing: bb,
                size: sb,
            },
        ) => {
            if !exprs_equal(body, *sa, *sb) {
                return false;
            }
            match (ba, bb) {
                (ArrayBacking::Literal(ea), ArrayBacking::Literal(eb)) => {
                    ea.len() == eb.len() && ea.iter().zip(eb.iter()).all(|(x, y)| exprs_equal(body, *x, *y))
                }
                (
                    ArrayBacking::Range {
                        start: sta,
                        step: stepa,
                        kind: ka,
                    },
                    ArrayBacking::Range {
                        start: stb,
                        step: stepb,
                        kind: kb,
                    },
                ) => {
                    ka == kb
                        && exprs_equal(body, *sta, *stb)
                        && match (stepa, stepb) {
                            (Some(a), Some(b)) => exprs_equal(body, *a, *b),
                            (None, None) => true,
                            _ => false,
                        }
                }
                _ => false,
            }
        }

        (
            Expr::StorageView {
                set: sa,
                binding: ba,
                offset: oa,
                len: la,
            },
            Expr::StorageView {
                set: sb,
                binding: bb,
                offset: ob,
                len: lb,
            },
        ) => sa == sb && ba == bb && exprs_equal(body, *oa, *ob) && exprs_equal(body, *la, *lb),
        (
            Expr::SliceStorageView {
                view: va,
                start: sa,
                len: la,
            },
            Expr::SliceStorageView {
                view: vb,
                start: sb,
                len: lb,
            },
        ) => exprs_equal(body, *va, *vb) && exprs_equal(body, *sa, *sb) && exprs_equal(body, *la, *lb),
        (
            Expr::StorageViewIndex { view: va, index: ia },
            Expr::StorageViewIndex { view: vb, index: ib },
        ) => exprs_equal(body, *va, *vb) && exprs_equal(body, *ia, *ib),
        (Expr::StorageViewLen { view: va }, Expr::StorageViewLen { view: vb }) => {
            exprs_equal(body, *va, *vb)
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

// =============================================================================
// Loop hoisting
// =============================================================================

/// Try to hoist loop-invariant materializations before a Loop expression.
fn hoist_in_loop(body: &mut Body, loop_id: ExprId) {
    let loop_body_id = match body.get_expr(loop_id) {
        Expr::Loop { body: b, .. } => b.result,
        _ => return,
    };

    // Collect all Materialize expressions in the loop body
    let mats = collect_materializations(body, loop_body_id);

    // Get locals defined by the loop (loop_var, init_bindings, and loop kind var)
    let loop_locals = collect_loop_locals(body, loop_id);

    // Filter to loop-invariant ones (inner expr only references locals defined outside loop)
    let loop_invariant: Vec<ExprId> =
        mats.into_iter().filter(|&mat_id| is_loop_invariant(body, mat_id, &loop_locals)).collect();

    // Hoist each loop-invariant materialization to a statement
    for mat_id in loop_invariant {
        body.hoist_to_stmt(mat_id, None);
    }
}

/// Collect all locals defined by a loop expression.
fn collect_loop_locals(body: &Body, loop_id: ExprId) -> HashSet<LocalId> {
    match body.get_expr(loop_id) {
        Expr::Loop {
            loop_var,
            init_bindings,
            kind,
            ..
        } => {
            let mut locals = HashSet::new();
            locals.insert(*loop_var);
            for (local, _) in init_bindings {
                locals.insert(*local);
            }
            // Add the loop kind variable
            match kind {
                LoopKind::For { var, .. } | LoopKind::ForRange { var, .. } => {
                    locals.insert(*var);
                }
                LoopKind::While { .. } => {}
            }
            locals
        }
        _ => HashSet::new(),
    }
}

/// Check if a Materialize expression is loop-invariant.
/// It's loop-invariant if its inner expression only references:
/// - Literals (Int, Float, Bool)
/// - Globals
/// - Locals defined OUTSIDE the loop (params, outer lets)
fn is_loop_invariant(body: &Body, mat_id: ExprId, loop_locals: &HashSet<LocalId>) -> bool {
    let inner = match body.get_expr(mat_id) {
        Expr::Materialize(inner) => *inner,
        _ => return false,
    };

    // Check that inner doesn't reference any loop-local variables
    !references_any_local(body, inner, loop_locals)
}

/// Check if an expression tree references any local from the given set.
fn references_any_local(body: &Body, expr_id: ExprId, locals: &HashSet<LocalId>) -> bool {
    let mut visited = HashSet::new();
    references_any_local_rec(body, expr_id, locals, &mut visited)
}

fn references_any_local_rec(
    body: &Body,
    expr_id: ExprId,
    locals: &HashSet<LocalId>,
    visited: &mut HashSet<ExprId>,
) -> bool {
    if !visited.insert(expr_id) {
        return false;
    }

    let expr = body.get_expr(expr_id);

    match expr {
        Expr::Local(id) => locals.contains(id),

        Expr::Materialize(inner) => references_any_local_rec(body, *inner, locals, visited),

        Expr::BinOp { lhs, rhs, .. } => {
            references_any_local_rec(body, *lhs, locals, visited)
                || references_any_local_rec(body, *rhs, locals, visited)
        }
        Expr::UnaryOp { operand, .. } => references_any_local_rec(body, *operand, locals, visited),

        Expr::If { cond, then_, else_ } => {
            references_any_local_rec(body, *cond, locals, visited)
                || references_any_local_rec(body, then_.result, locals, visited)
                || references_any_local_rec(body, else_.result, locals, visited)
        }


        Expr::Loop {
            init,
            init_bindings,
            body: loop_body,
            ..
        } => {
            // Nested loop - check init and init_bindings, but not body (it's in a new scope)
            if references_any_local_rec(body, *init, locals, visited) {
                return true;
            }
            for (_, binding_expr) in init_bindings {
                if references_any_local_rec(body, *binding_expr, locals, visited) {
                    return true;
                }
            }
            // Check loop body (conservatively include it)
            references_any_local_rec(body, loop_body.result, locals, visited)
        }

        Expr::Call { args, .. } | Expr::Intrinsic { args, .. } => {
            args.iter().any(|arg| references_any_local_rec(body, *arg, locals, visited))
        }

        Expr::Tuple(elems) | Expr::Vector(elems) => {
            elems.iter().any(|elem| references_any_local_rec(body, *elem, locals, visited))
        }

        Expr::Array { backing, size } => {
            if references_any_local_rec(body, *size, locals, visited) {
                return true;
            }
            match backing {
                ArrayBacking::Literal(elems) => {
                    elems.iter().any(|elem| references_any_local_rec(body, *elem, locals, visited))
                }
                ArrayBacking::Range { start, step, .. } => {
                    references_any_local_rec(body, *start, locals, visited)
                        || step.map_or(false, |s| references_any_local_rec(body, s, locals, visited))
                }
            }
        }

        Expr::Matrix(rows) => rows
            .iter()
            .any(|row| row.iter().any(|elem| references_any_local_rec(body, *elem, locals, visited))),

        Expr::Attributed { expr, .. } => references_any_local_rec(body, *expr, locals, visited),

        Expr::Load { ptr } => references_any_local_rec(body, *ptr, locals, visited),

        Expr::Store { ptr, value } => {
            references_any_local_rec(body, *ptr, locals, visited)
                || references_any_local_rec(body, *value, locals, visited)
        }

        Expr::StorageView { offset, len, .. } => {
            references_any_local_rec(body, *offset, locals, visited)
                || references_any_local_rec(body, *len, locals, visited)
        }

        Expr::SliceStorageView { view, start, len } => {
            references_any_local_rec(body, *view, locals, visited)
                || references_any_local_rec(body, *start, locals, visited)
                || references_any_local_rec(body, *len, locals, visited)
        }

        Expr::StorageViewIndex { view, index } => {
            references_any_local_rec(body, *view, locals, visited)
                || references_any_local_rec(body, *index, locals, visited)
        }

        Expr::StorageViewLen { view } => references_any_local_rec(body, *view, locals, visited),

        // Atoms don't reference locals (except Local which is handled above)
        Expr::Global(_)
        | Expr::Extern(_)
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Bool(_)
        | Expr::Unit
        | Expr::String(_) => false,
    }
}
