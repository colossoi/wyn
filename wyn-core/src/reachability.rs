//! Reachability analysis for MIR
//!
//! This module performs reachability analysis on MIR to determine
//! which functions are actually called, starting from entry points.
//! It returns functions in topological order (callees before callers)
//! so the lowerer can process them without forward references.

use crate::mir::{Body, Def, Expr, Program};
use std::collections::HashSet;

/// Find all functions reachable from entry points, in topological order.
/// Returns a Vec with callees before callers (post-order DFS).
pub fn reachable_functions_ordered(program: &Program) -> Vec<String> {
    // Find entry points (Def::EntryPoint variants)
    let mut entry_points = Vec::new();
    for def in &program.defs {
        if let Def::EntryPoint { name, .. } = def {
            entry_points.push(name.clone());
        }
    }

    // Build a map of all functions for quick lookup
    let mut functions = std::collections::HashMap::new();
    for def in &program.defs {
        match def {
            Def::Function { name, body, .. } => {
                functions.insert(name.clone(), body);
            }
            Def::Constant { name, body, .. } => {
                // Constants might call functions too
                functions.insert(name.clone(), body);
            }
            Def::EntryPoint { name, body, .. } => {
                functions.insert(name.clone(), body);
            }
            Def::Uniform { .. } | Def::Storage { .. } => {
                // Uniforms and storage have no body
            }
        }
    }

    // Post-order DFS from entry points
    let mut visited = HashSet::new();
    let mut in_stack = HashSet::new();
    let mut order = Vec::new();

    for entry in entry_points {
        dfs_postorder(&entry, &functions, &mut visited, &mut in_stack, &mut order);
    }

    order
}

fn dfs_postorder(
    name: &str,
    functions: &std::collections::HashMap<String, &Body>,
    visited: &mut HashSet<String>,
    in_stack: &mut HashSet<String>,
    order: &mut Vec<String>,
) {
    if visited.contains(name) || in_stack.contains(name) {
        return;
    }

    in_stack.insert(name.to_string());

    // Visit all callees first (if they exist in the program)
    if let Some(body) = functions.get(name) {
        let callees = collect_callees(body);
        for callee in callees {
            if functions.contains_key(&callee) {
                dfs_postorder(&callee, functions, visited, in_stack, order);
            }
        }
    }

    in_stack.remove(name);
    visited.insert(name.to_string());
    order.push(name.to_string());
}

/// Find all functions reachable from entry points (unordered set).
pub fn reachable_functions(program: &Program) -> HashSet<String> {
    reachable_functions_ordered(program).into_iter().collect()
}

/// Collect all function names called in a body.
/// Walks all expressions in the body's arena.
fn collect_callees(body: &Body) -> HashSet<String> {
    let mut callees = HashSet::new();

    for expr in &body.exprs {
        match expr {
            Expr::Global(name) => {
                // Global references might refer to top-level functions or constants
                callees.insert(name.clone());
            }
            Expr::Call { func, .. } => {
                callees.insert(func.clone());
            }
            // Other expressions don't introduce new callees
            _ => {}
        }
    }

    callees
}

/// Filter a program to only include reachable definitions, in topological order.
/// Callees come before callers, so the lowerer can process them without forward references.
pub fn filter_reachable(program: Program) -> Program {
    let ordered = reachable_functions_ordered(&program);

    // Destructure to preserve lambda_registry
    let Program {
        defs,
        lambda_registry,
    } = program;

    // Build a map from name to def for reordering
    let mut def_map: std::collections::HashMap<String, Def> = defs
        .into_iter()
        .map(|def| {
            let name = match &def {
                Def::Function { name, .. } => name.clone(),
                Def::Constant { name, .. } => name.clone(),
                Def::EntryPoint { name, .. } => name.clone(),
                Def::Uniform { name, .. } => name.clone(),
                Def::Storage { name, .. } => name.clone(),
            };
            (name, def)
        })
        .collect();

    // Output defs in topological order, starting with reachable ones
    let mut output_defs = Vec::new();

    // First, add all reachable defs in order
    for name in &ordered {
        if let Some(def) = def_map.remove(name) {
            output_defs.push(def);
        }
    }

    // Uniforms and storage buffers should always be included if they exist
    // (they're external resources, not called functions)
    for (_, def) in def_map {
        match def {
            Def::Uniform { .. } | Def::Storage { .. } => {
                output_defs.push(def);
            }
            _ => {
                // Unreachable function/constant - skip it
            }
        }
    }

    Program {
        defs: output_defs,
        lambda_registry,
    }
}
