//! Dead function elimination pass.
//!
//! Removes functions that are not reachable from any entry point.

use crate::mir::ssa::{FuncBody, InstKind};
use crate::tlc::to_ssa::SsaProgram;
use std::collections::HashSet;

/// Eliminate functions that are not reachable from any entry point.
pub fn eliminate_dead_functions(mut program: SsaProgram) -> SsaProgram {
    let mut live: HashSet<String> = HashSet::new();
    let mut worklist: Vec<String> = Vec::new();

    // Seed with functions referenced from entry points
    for entry in &program.entry_points {
        collect_references(&entry.body, &mut live, &mut worklist);
    }

    // Transitively find all referenced functions
    while let Some(func_name) = worklist.pop() {
        if let Some(func) = program.functions.iter().find(|f| f.name == func_name) {
            collect_references(&func.body, &mut live, &mut worklist);
        }
    }

    // Remove dead functions (keep extern functions with linkage_name)
    program.functions.retain(|f| f.linkage_name.is_some() || live.contains(&f.name));

    program
}

/// Collect function references (calls and globals) from a function body.
fn collect_references(body: &FuncBody, live: &mut HashSet<String>, worklist: &mut Vec<String>) {
    for inst in &body.insts {
        let name = match &inst.kind {
            InstKind::Call { func, .. } => Some(func.clone()),
            InstKind::Global(name) => Some(name.clone()),
            _ => None,
        };
        if let Some(name) = name {
            if live.insert(name.clone()) {
                worklist.push(name);
            }
        }
    }
}
