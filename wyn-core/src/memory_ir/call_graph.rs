//! Direct call sites in the structured memory IR.

use super::{Definition, Function, FunctionId, MemoryIr, OperationId};
use crate::{LookupMap, LookupSet, StableMap};
use std::hash::Hash;
use wyn_graph::{dfs_postorder, for_each_reachable, WalkOrder};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallSite<F = FunctionId, O = OperationId> {
    pub operation: O,
    pub callee: F,
}

/// A snapshot of a program's calls. Repeated calls have distinct operation IDs.
#[derive(Debug)]
pub struct CallGraph<F = FunctionId, O = OperationId> {
    pub calls: StableMap<F, Vec<CallSite<F, O>>>,
}

impl CallGraph {
    /// Include sites in every attached body. External callees are leaves.
    pub fn from_ir<D>(ir: &MemoryIr<D>) -> Self {
        let mut calls = StableMap::default();
        for (function, body) in &ir.functions {
            let Function::Defined(root) = *body else {
                continue;
            };
            let mut sites = Vec::new();
            for_each_reachable(
                [root],
                WalkOrder::DepthFirst,
                |region, children| {
                    for &id in ir.regions[region].operations.iter().rev() {
                        children.extend(ir.operations[id].bodies.iter().rev().map(|edge| edge.region));
                    }
                },
                |region| {
                    for &id in &ir.regions[region].operations {
                        if let Definition::Call(callee) = &ir.operations[id].definition {
                            sites.push(CallSite {
                                operation: id,
                                callee: *callee,
                            });
                        }
                    }
                },
            );
            calls.insert(function, sites);
        }
        Self { calls }
    }
}

impl<F: Copy + Eq + Hash, O> CallGraph<F, O> {
    /// Reachable functions (including external leaves), and the reachable functions
    /// that can call themselves. Both walks in Kosaraju's algorithm are iterative.
    pub fn analyze(&self, roots: impl IntoIterator<Item = F>) -> (Vec<F>, LookupSet<F>) {
        let mut callers = LookupMap::<F, Vec<F>>::new();
        let mut recursive = LookupSet::new();
        let live = dfs_postorder(
            roots,
            |_| false,
            |caller, next| {
                if let Some(sites) = self.calls.get(&caller) {
                    for site in sites {
                        next.push(site.callee);
                        callers.entry(site.callee).or_default().push(caller);
                        if caller == site.callee {
                            recursive.insert(caller);
                        }
                    }
                }
            },
        );
        let mut visited = LookupSet::new();
        let mut component = Vec::new();
        for &root in live.iter().rev() {
            if !visited.insert(root) {
                continue;
            }
            component.clear();
            component.push(root);
            let mut index = 0;
            while index < component.len() {
                if let Some(parents) = callers.get(&component[index]) {
                    component.extend(parents.iter().copied().filter(|f| visited.insert(*f)));
                }
                index += 1;
            }
            if component.len() > 1 {
                recursive.extend(component.iter().copied());
            }
        }
        (live, recursive)
    }
}
