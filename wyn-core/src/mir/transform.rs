//! MIR transformation operations.
//!
//! This module provides composable, correct-by-construction operations for
//! transforming MIR. The key operations are:
//!
//! - **reads**: Compute which locals an expression transitively reads
//! - **lift_stmts**: Topologically sort stmts by data dependencies
//! - **atomize**: Extract a subexpression into a new let binding
//! - **merge_identical**: CSE for structurally identical expressions
//!
//! These primitives can be composed to build passes like normalization and
//! materialization hoisting.

use super::{ArrayBacking, Block, Body, Expr, ExprId, LocalDecl, LocalId, LocalKind, LoopKind, Stmt};
use std::collections::{HashMap, HashSet};

// =============================================================================
// AccumulatorStack - scoped accumulation without name-based lookup
// =============================================================================

/// A stack-based accumulator for collecting items in nested scopes.
/// Unlike ScopeStack, this doesn't support name-based lookup - it's purely
/// for ordered accumulation where items should stay in their scope.
#[derive(Debug, Clone)]
pub struct AccumulatorStack<T> {
    scopes: Vec<Vec<T>>,
}

impl<T> Default for AccumulatorStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AccumulatorStack<T> {
    /// Create a new accumulator stack with one scope.
    pub fn new() -> Self {
        AccumulatorStack {
            scopes: vec![Vec::new()],
        }
    }

    /// Push a new scope onto the stack.
    pub fn push_scope(&mut self) {
        self.scopes.push(Vec::new());
    }

    /// Pop the current scope, returning its accumulated items.
    /// Returns empty vec if trying to pop the last scope.
    pub fn pop_scope(&mut self) -> Vec<T> {
        if self.scopes.len() > 1 {
            self.scopes.pop().unwrap_or_default()
        } else {
            // Don't pop the last scope, just drain it
            std::mem::take(self.scopes.last_mut().unwrap())
        }
    }

    /// Add an item to the current (innermost) scope.
    pub fn push(&mut self, item: T) {
        if let Some(current) = self.scopes.last_mut() {
            current.push(item);
        }
    }

    /// Get the items in the current scope (read-only).
    pub fn current(&self) -> &[T] {
        self.scopes.last().map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Drain all items from all scopes into a single vec, innermost first.
    pub fn drain_all(&mut self) -> Vec<T> {
        let mut result = Vec::new();
        for scope in &mut self.scopes {
            result.append(scope);
        }
        result
    }

    /// Check if the current scope is empty.
    pub fn is_current_empty(&self) -> bool {
        self.scopes.last().map(|v| v.is_empty()).unwrap_or(true)
    }

    /// Get current nesting depth (0 = root scope).
    pub fn depth(&self) -> usize {
        self.scopes.len().saturating_sub(1)
    }
}

// =============================================================================
// Data dependency analysis
// =============================================================================

/// Collect all locals that an expression reads (transitively).
pub fn expr_reads(body: &Body, expr_id: ExprId) -> HashSet<LocalId> {
    let mut result = HashSet::new();
    expr_reads_rec(body, expr_id, &mut result);
    result
}

fn expr_reads_rec(body: &Body, expr_id: ExprId, result: &mut HashSet<LocalId>) {
    match body.get_expr(expr_id) {
        Expr::Local(id) => {
            result.insert(*id);
        }
        Expr::BinOp { lhs, rhs, .. } => {
            expr_reads_rec(body, *lhs, result);
            expr_reads_rec(body, *rhs, result);
        }
        Expr::UnaryOp { operand, .. } => {
            expr_reads_rec(body, *operand, result);
        }
        Expr::Tuple(elems) | Expr::Vector(elems) => {
            for elem in elems {
                expr_reads_rec(body, *elem, result);
            }
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for elem in row {
                    expr_reads_rec(body, *elem, result);
                }
            }
        }
        Expr::Array { backing, size } => {
            expr_reads_rec(body, *size, result);
            match backing {
                ArrayBacking::Literal(elems) => {
                    for elem in elems {
                        expr_reads_rec(body, *elem, result);
                    }
                }
                ArrayBacking::Range { start, step, .. } => {
                    expr_reads_rec(body, *start, result);
                    if let Some(s) = step {
                        expr_reads_rec(body, *s, result);
                    }
                }
            }
        }
        Expr::Call { args, .. } | Expr::Intrinsic { args, .. } => {
            for arg in args {
                expr_reads_rec(body, *arg, result);
            }
        }
        Expr::If { cond, then_, else_ } => {
            expr_reads_rec(body, *cond, result);
            block_reads_rec(body, then_, result);
            block_reads_rec(body, else_, result);
        }
        Expr::Loop {
            init,
            init_bindings,
            kind,
            body: loop_body,
            ..
        } => {
            expr_reads_rec(body, *init, result);
            // Note: init_bindings define locals, they don't read from outside
            // But they do read from loop_var which is defined by init
            for (_, expr) in init_bindings {
                expr_reads_rec(body, *expr, result);
            }
            match kind {
                LoopKind::For { iter, .. } => expr_reads_rec(body, *iter, result),
                LoopKind::ForRange { bound, .. } => expr_reads_rec(body, *bound, result),
                LoopKind::While { cond } => expr_reads_rec(body, *cond, result),
            }
            block_reads_rec(body, loop_body, result);
        }
        Expr::Materialize(inner) | Expr::Attributed { expr: inner, .. } => {
            expr_reads_rec(body, *inner, result);
        }
        Expr::Load { ptr } => expr_reads_rec(body, *ptr, result),
        Expr::Store { ptr, value } => {
            expr_reads_rec(body, *ptr, result);
            expr_reads_rec(body, *value, result);
        }
        Expr::StorageView { offset, len, .. } => {
            expr_reads_rec(body, *offset, result);
            expr_reads_rec(body, *len, result);
        }
        Expr::SliceStorageView { view, start, len } => {
            expr_reads_rec(body, *view, result);
            expr_reads_rec(body, *start, result);
            expr_reads_rec(body, *len, result);
        }
        Expr::StorageViewIndex { view, index } => {
            expr_reads_rec(body, *view, result);
            expr_reads_rec(body, *index, result);
        }
        Expr::StorageViewLen { view } => expr_reads_rec(body, *view, result),
        // Leaves with no reads
        Expr::Global(_)
        | Expr::Extern(_)
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Bool(_)
        | Expr::Unit
        | Expr::String(_) => {}
    }
}

fn block_reads_rec(body: &Body, block: &Block, result: &mut HashSet<LocalId>) {
    for stmt in &block.stmts {
        expr_reads_rec(body, stmt.rhs, result);
    }
    expr_reads_rec(body, block.result, result);
}

/// Collect all locals that a stmt reads.
pub fn stmt_reads(body: &Body, stmt: &Stmt) -> HashSet<LocalId> {
    expr_reads(body, stmt.rhs)
}

// =============================================================================
// Topological sorting of stmts
// =============================================================================

/// Topologically sort stmts so that a stmt defining local X comes before
/// any stmt whose RHS reads local X.
///
/// This is the core operation for fixing stmt ordering after transformations.
pub fn sort_stmts_by_deps(body: &Body, stmts: &[Stmt]) -> Vec<Stmt> {
    if stmts.is_empty() {
        return Vec::new();
    }

    // Build a map from local -> stmt index that defines it
    let mut defined_by: HashMap<LocalId, usize> = HashMap::new();
    for (i, stmt) in stmts.iter().enumerate() {
        defined_by.insert(stmt.local, i);
    }

    // For each stmt, compute which other stmts it depends on
    let mut deps: Vec<HashSet<usize>> = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        let reads = stmt_reads(body, stmt);
        let mut stmt_deps = HashSet::new();
        for local in reads {
            if let Some(&def_idx) = defined_by.get(&local) {
                stmt_deps.insert(def_idx);
            }
        }
        deps.push(stmt_deps);
    }

    // Kahn's algorithm for topological sort
    let mut in_degree: Vec<usize> = vec![0; stmts.len()];
    for stmt_deps in &deps {
        for &dep in stmt_deps {
            // This stmt depends on `dep`, so `dep` has an outgoing edge to it
            // We count incoming edges, so we need to track the reverse
            let _ = dep;
        }
    }
    // Actually compute in_degree: for each stmt, count how many stmts depend on it
    // No wait, in_degree[i] = number of stmts that i depends on
    for (i, stmt_deps) in deps.iter().enumerate() {
        in_degree[i] = stmt_deps.len();
    }

    // Start with stmts that have no dependencies
    let mut queue: Vec<usize> = (0..stmts.len()).filter(|&i| in_degree[i] == 0).collect();

    let mut result = Vec::with_capacity(stmts.len());
    let mut processed: HashSet<usize> = HashSet::new();

    while let Some(idx) = queue.pop() {
        if processed.contains(&idx) {
            continue;
        }
        processed.insert(idx);
        result.push(stmts[idx].clone());

        // For all stmts that depend on this one, decrement their in_degree
        for (other_idx, other_deps) in deps.iter().enumerate() {
            if other_deps.contains(&idx) && !processed.contains(&other_idx) {
                in_degree[other_idx] -= 1;
                if in_degree[other_idx] == 0 {
                    queue.push(other_idx);
                }
            }
        }
    }

    // If there's a cycle, just append remaining stmts (shouldn't happen in valid MIR)
    for (i, stmt) in stmts.iter().enumerate() {
        if !processed.contains(&i) {
            result.push(stmt.clone());
        }
    }

    result
}

/// Lift all body-level stmts to their earliest valid positions.
pub fn lift_body_stmts(body: &mut Body) {
    let stmts: Vec<Stmt> = body.drain_all_stmts();
    let sorted = sort_stmts_by_deps(body, &stmts);
    for stmt in sorted {
        body.push_stmt(stmt.local, stmt.rhs);
    }
}

// =============================================================================
// Atomization
// =============================================================================

/// Check if an expression is atomic (a leaf that doesn't need further binding).
pub fn is_atomic(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Local(_)
            | Expr::Global(_)
            | Expr::Extern(_)
            | Expr::Int(_)
            | Expr::Float(_)
            | Expr::Bool(_)
            | Expr::Unit
            | Expr::String(_)
    )
}

/// Atomize an expression: if it's not atomic, create a new let binding for it.
/// Returns the ExprId to use (either the original if atomic, or a Local reference).
/// If a binding was created, also returns the Stmt.
pub fn atomize_expr(
    body: &mut Body,
    expr_id: ExprId,
    name_prefix: &str,
    counter: &mut usize,
) -> (ExprId, Option<Stmt>) {
    let expr = body.get_expr(expr_id);
    if is_atomic(expr) {
        return (expr_id, None);
    }

    // Create a new local for this expression
    let name = format!("{}_{}", name_prefix, *counter);
    *counter += 1;

    let ty = body.get_type(expr_id).clone();
    let span = body.get_span(expr_id);
    let node_id = body.get_node_id(expr_id);

    let local_id = body.alloc_local(LocalDecl {
        name,
        ty: ty.clone(),
        span,
        kind: LocalKind::Let,
    });

    // Create a reference to the new local
    let local_ref = body.alloc_expr(Expr::Local(local_id), ty, span, node_id);

    // Create the stmt
    let stmt = Stmt {
        local: local_id,
        rhs: expr_id,
    };

    (local_ref, Some(stmt))
}

// =============================================================================
// Expression copying with remapping
// =============================================================================

/// Copy an expression tree from one body to another, remapping ExprIds.
/// Returns the new ExprId in the destination body.
pub fn copy_expr(
    src: &Body,
    dest: &mut Body,
    expr_id: ExprId,
    expr_map: &mut HashMap<ExprId, ExprId>,
    local_map: &HashMap<LocalId, LocalId>,
) -> ExprId {
    // Check if already copied
    if let Some(&new_id) = expr_map.get(&expr_id) {
        return new_id;
    }

    let ty = src.get_type(expr_id).clone();
    let span = src.get_span(expr_id);
    let node_id = src.get_node_id(expr_id);

    let new_expr = match src.get_expr(expr_id).clone() {
        Expr::Local(id) => Expr::Local(*local_map.get(&id).unwrap_or(&id)),
        Expr::Global(name) => Expr::Global(name),
        Expr::Extern(linkage) => Expr::Extern(linkage),
        Expr::Int(s) => Expr::Int(s),
        Expr::Float(s) => Expr::Float(s),
        Expr::Bool(b) => Expr::Bool(b),
        Expr::Unit => Expr::Unit,
        Expr::String(s) => Expr::String(s),

        Expr::BinOp { op, lhs, rhs } => {
            let new_lhs = copy_expr(src, dest, lhs, expr_map, local_map);
            let new_rhs = copy_expr(src, dest, rhs, expr_map, local_map);
            Expr::BinOp {
                op,
                lhs: new_lhs,
                rhs: new_rhs,
            }
        }
        Expr::UnaryOp { op, operand } => {
            let new_operand = copy_expr(src, dest, operand, expr_map, local_map);
            Expr::UnaryOp {
                op,
                operand: new_operand,
            }
        }
        Expr::Tuple(elems) => {
            let new_elems: Vec<_> =
                elems.iter().map(|&e| copy_expr(src, dest, e, expr_map, local_map)).collect();
            Expr::Tuple(new_elems)
        }
        Expr::Vector(elems) => {
            let new_elems: Vec<_> =
                elems.iter().map(|&e| copy_expr(src, dest, e, expr_map, local_map)).collect();
            Expr::Vector(new_elems)
        }
        Expr::Matrix(rows) => {
            let new_rows: Vec<Vec<_>> = rows
                .iter()
                .map(|row| row.iter().map(|&e| copy_expr(src, dest, e, expr_map, local_map)).collect())
                .collect();
            Expr::Matrix(new_rows)
        }
        Expr::Array { backing, size } => {
            let new_size = copy_expr(src, dest, size, expr_map, local_map);
            let new_backing = match backing {
                ArrayBacking::Literal(elems) => {
                    let new_elems: Vec<_> =
                        elems.iter().map(|&e| copy_expr(src, dest, e, expr_map, local_map)).collect();
                    ArrayBacking::Literal(new_elems)
                }
                ArrayBacking::Range { start, step, kind } => {
                    let new_start = copy_expr(src, dest, start, expr_map, local_map);
                    let new_step = step.map(|s| copy_expr(src, dest, s, expr_map, local_map));
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
        Expr::Call { func, args } => {
            let new_args: Vec<_> =
                args.iter().map(|&a| copy_expr(src, dest, a, expr_map, local_map)).collect();
            Expr::Call { func, args: new_args }
        }
        Expr::Intrinsic { name, args } => {
            let new_args: Vec<_> =
                args.iter().map(|&a| copy_expr(src, dest, a, expr_map, local_map)).collect();
            Expr::Intrinsic { name, args: new_args }
        }
        Expr::If { cond, then_, else_ } => {
            let new_cond = copy_expr(src, dest, cond, expr_map, local_map);
            let new_then = copy_block(src, dest, &then_, expr_map, local_map);
            let new_else = copy_block(src, dest, &else_, expr_map, local_map);
            Expr::If {
                cond: new_cond,
                then_: new_then,
                else_: new_else,
            }
        }
        Expr::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body: loop_body,
        } => {
            let new_init = copy_expr(src, dest, init, expr_map, local_map);
            let new_loop_var = *local_map.get(&loop_var).unwrap_or(&loop_var);
            let new_init_bindings: Vec<_> = init_bindings
                .iter()
                .map(|(local, expr)| {
                    let new_local = *local_map.get(local).unwrap_or(local);
                    let new_expr = copy_expr(src, dest, *expr, expr_map, local_map);
                    (new_local, new_expr)
                })
                .collect();
            let new_kind = match kind {
                LoopKind::For { var, iter } => {
                    let new_var = *local_map.get(&var).unwrap_or(&var);
                    let new_iter = copy_expr(src, dest, iter, expr_map, local_map);
                    LoopKind::For {
                        var: new_var,
                        iter: new_iter,
                    }
                }
                LoopKind::ForRange { var, bound } => {
                    let new_var = *local_map.get(&var).unwrap_or(&var);
                    let new_bound = copy_expr(src, dest, bound, expr_map, local_map);
                    LoopKind::ForRange {
                        var: new_var,
                        bound: new_bound,
                    }
                }
                LoopKind::While { cond } => {
                    let new_cond = copy_expr(src, dest, cond, expr_map, local_map);
                    LoopKind::While { cond: new_cond }
                }
            };
            let new_body = copy_block(src, dest, &loop_body, expr_map, local_map);
            Expr::Loop {
                loop_var: new_loop_var,
                init: new_init,
                init_bindings: new_init_bindings,
                kind: new_kind,
                body: new_body,
            }
        }
        Expr::Materialize(inner) => {
            let new_inner = copy_expr(src, dest, inner, expr_map, local_map);
            Expr::Materialize(new_inner)
        }
        Expr::Attributed { attributes, expr } => {
            let new_expr = copy_expr(src, dest, expr, expr_map, local_map);
            Expr::Attributed {
                attributes,
                expr: new_expr,
            }
        }
        Expr::Load { ptr } => {
            let new_ptr = copy_expr(src, dest, ptr, expr_map, local_map);
            Expr::Load { ptr: new_ptr }
        }
        Expr::Store { ptr, value } => {
            let new_ptr = copy_expr(src, dest, ptr, expr_map, local_map);
            let new_value = copy_expr(src, dest, value, expr_map, local_map);
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
            let new_offset = copy_expr(src, dest, offset, expr_map, local_map);
            let new_len = copy_expr(src, dest, len, expr_map, local_map);
            Expr::StorageView {
                set,
                binding,
                offset: new_offset,
                len: new_len,
            }
        }
        Expr::SliceStorageView { view, start, len } => {
            let new_view = copy_expr(src, dest, view, expr_map, local_map);
            let new_start = copy_expr(src, dest, start, expr_map, local_map);
            let new_len = copy_expr(src, dest, len, expr_map, local_map);
            Expr::SliceStorageView {
                view: new_view,
                start: new_start,
                len: new_len,
            }
        }
        Expr::StorageViewIndex { view, index } => {
            let new_view = copy_expr(src, dest, view, expr_map, local_map);
            let new_index = copy_expr(src, dest, index, expr_map, local_map);
            Expr::StorageViewIndex {
                view: new_view,
                index: new_index,
            }
        }
        Expr::StorageViewLen { view } => {
            let new_view = copy_expr(src, dest, view, expr_map, local_map);
            Expr::StorageViewLen { view: new_view }
        }
    };

    let new_id = dest.alloc_expr(new_expr, ty, span, node_id);
    expr_map.insert(expr_id, new_id);
    new_id
}

/// Copy a block (stmts + result) from one body to another.
pub fn copy_block(
    src: &Body,
    dest: &mut Body,
    block: &Block,
    expr_map: &mut HashMap<ExprId, ExprId>,
    local_map: &HashMap<LocalId, LocalId>,
) -> Block {
    let new_stmts: Vec<Stmt> = block
        .stmts
        .iter()
        .map(|stmt| {
            let new_rhs = copy_expr(src, dest, stmt.rhs, expr_map, local_map);
            let new_local = *local_map.get(&stmt.local).unwrap_or(&stmt.local);
            Stmt {
                local: new_local,
                rhs: new_rhs,
            }
        })
        .collect();
    let new_result = copy_expr(src, dest, block.result, expr_map, local_map);
    Block::with_stmts(new_stmts, new_result)
}

// =============================================================================
// Body rebuilding with proper ordering
// =============================================================================

/// Rebuild a body, ensuring all expressions and stmts are in proper dependency order.
/// This is the main entry point for fixing ordering after transformations.
pub fn rebuild_body_ordered(body: &Body) -> Body {
    let mut new_body = Body::new();

    // Copy locals
    let mut local_map: HashMap<LocalId, LocalId> = HashMap::new();
    for (old_idx, local) in body.locals.iter().enumerate() {
        let old_id = LocalId(old_idx as u32);
        let new_id = new_body.alloc_local(local.clone());
        local_map.insert(old_id, new_id);
    }

    // Copy all expressions (they'll be allocated in traversal order)
    let mut expr_map: HashMap<ExprId, ExprId> = HashMap::new();

    // First, copy all stmt RHS expressions
    for stmt in body.iter_stmts() {
        copy_expr(body, &mut new_body, stmt.rhs, &mut expr_map, &local_map);
    }

    // Then copy from root
    copy_expr(body, &mut new_body, body.root, &mut expr_map, &local_map);

    // Collect and sort stmts
    let stmts: Vec<Stmt> = body
        .iter_stmts()
        .map(|stmt| Stmt {
            local: *local_map.get(&stmt.local).unwrap_or(&stmt.local),
            rhs: expr_map[&stmt.rhs],
        })
        .collect();

    let sorted_stmts = sort_stmts_by_deps(&new_body, &stmts);
    for stmt in sorted_stmts {
        new_body.push_stmt(stmt.local, stmt.rhs);
    }

    // Set root
    new_body.root = expr_map[&body.root];

    new_body
}

// =============================================================================
// Block-aware transformations
// =============================================================================

/// Process a block's stmts, sorting them by dependencies.
pub fn sort_block_stmts(body: &Body, block: &Block) -> Block {
    let sorted_stmts = sort_stmts_by_deps(body, &block.stmts);
    Block::with_stmts(sorted_stmts, block.result)
}

/// Recursively sort stmts in all nested blocks of an expression.
pub fn sort_nested_blocks(body: &mut Body, expr_id: ExprId) {
    let expr = body.get_expr(expr_id).clone();
    match expr {
        Expr::If { cond, then_, else_ } => {
            sort_nested_blocks(body, cond);
            let new_then = sort_block_stmts(body, &then_);
            let new_else = sort_block_stmts(body, &else_);
            // Recursively sort within the blocks
            for stmt in &new_then.stmts {
                sort_nested_blocks(body, stmt.rhs);
            }
            sort_nested_blocks(body, new_then.result);
            for stmt in &new_else.stmts {
                sort_nested_blocks(body, stmt.rhs);
            }
            sort_nested_blocks(body, new_else.result);
            *body.get_expr_mut(expr_id) = Expr::If {
                cond,
                then_: new_then,
                else_: new_else,
            };
        }
        Expr::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body: loop_body,
        } => {
            sort_nested_blocks(body, init);
            let new_body_block = sort_block_stmts(body, &loop_body);
            for stmt in &new_body_block.stmts {
                sort_nested_blocks(body, stmt.rhs);
            }
            sort_nested_blocks(body, new_body_block.result);
            *body.get_expr_mut(expr_id) = Expr::Loop {
                loop_var,
                init,
                init_bindings,
                kind,
                body: new_body_block,
            };
        }
        // Recurse into other compound expressions
        Expr::BinOp { lhs, rhs, .. } => {
            sort_nested_blocks(body, lhs);
            sort_nested_blocks(body, rhs);
        }
        Expr::UnaryOp { operand, .. } => {
            sort_nested_blocks(body, operand);
        }
        Expr::Tuple(elems) | Expr::Vector(elems) => {
            for elem in elems {
                sort_nested_blocks(body, elem);
            }
        }
        Expr::Call { args, .. } | Expr::Intrinsic { args, .. } => {
            for arg in args {
                sort_nested_blocks(body, arg);
            }
        }
        Expr::Materialize(inner) | Expr::Attributed { expr: inner, .. } => {
            sort_nested_blocks(body, inner);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{NodeId, Span};

    #[test]
    fn test_sort_stmts_simple() {
        // Create a body with stmts out of order
        let mut body = Body::new();

        // local_0 = 1
        // local_1 = local_0 + 2
        // But add them in reverse order

        let local_0 = body.alloc_local(LocalDecl {
            name: "a".to_string(),
            ty: polytype::Type::Constructed(crate::ast::TypeName::Int(32), vec![]),
            span: Span {
                start_line: 1,
                start_col: 1,
                end_line: 1,
                end_col: 1,
            },
            kind: LocalKind::Let,
        });
        let local_1 = body.alloc_local(LocalDecl {
            name: "b".to_string(),
            ty: polytype::Type::Constructed(crate::ast::TypeName::Int(32), vec![]),
            span: Span {
                start_line: 1,
                start_col: 1,
                end_line: 1,
                end_col: 1,
            },
            kind: LocalKind::Let,
        });

        let dummy_span = Span {
            start_line: 1,
            start_col: 1,
            end_line: 1,
            end_col: 1,
        };
        let dummy_node = NodeId(0);
        let int_ty = polytype::Type::Constructed(crate::ast::TypeName::Int(32), vec![]);

        let one = body.alloc_expr(Expr::Int("1".to_string()), int_ty.clone(), dummy_span, dummy_node);
        let two = body.alloc_expr(Expr::Int("2".to_string()), int_ty.clone(), dummy_span, dummy_node);
        let local_0_ref = body.alloc_expr(Expr::Local(local_0), int_ty.clone(), dummy_span, dummy_node);
        let add = body.alloc_expr(
            Expr::BinOp {
                op: "+".to_string(),
                lhs: local_0_ref,
                rhs: two,
            },
            int_ty.clone(),
            dummy_span,
            dummy_node,
        );

        // Add stmts in wrong order: b = a + 2, then a = 1
        let stmts = vec![
            Stmt {
                local: local_1,
                rhs: add,
            },
            Stmt {
                local: local_0,
                rhs: one,
            },
        ];

        let sorted = sort_stmts_by_deps(&body, &stmts);

        // Should be: a = 1, then b = a + 2
        assert_eq!(sorted.len(), 2);
        assert_eq!(sorted[0].local, local_0);
        assert_eq!(sorted[1].local, local_1);
    }
}
