//! Scoped hashmap for elaboration.
//!
//! Each scope corresponds to a subtree in the dominator tree.
//! Values inserted in a child scope are invisible after popping back to the parent.

use crate::LookupMap;
use std::hash::Hash;

/// A hashmap with push/pop scope operations.
pub struct ScopedMap<K: Hash + Eq, V> {
    /// Key → stack of (depth, value). Top of stack is the most recent.
    map: LookupMap<K, Vec<(usize, V)>>,
    /// Current scope depth (0 = root).
    depth: usize,
    /// Keys inserted at each depth, for cleanup on pop.
    scope_keys: Vec<Vec<K>>,
}

impl<K: Hash + Eq + Clone, V: Copy> ScopedMap<K, V> {
    pub fn new() -> Self {
        ScopedMap {
            map: LookupMap::new(),
            depth: 0,
            scope_keys: vec![Vec::new()],
        }
    }

    /// Enter a new child scope.
    pub fn push_scope(&mut self) {
        self.depth += 1;
        self.scope_keys.push(Vec::new());
    }

    /// Leave the current scope, removing all entries added in it.
    pub fn pop_scope(&mut self) {
        let keys = self.scope_keys.pop().expect("pop_scope on root");
        for key in keys {
            if let Some(stack) = self.map.get_mut(&key) {
                stack.pop();
                if stack.is_empty() {
                    self.map.remove(&key);
                }
            }
        }
        self.depth -= 1;
    }

    /// Insert a key-value pair in the current scope.
    pub fn insert(&mut self, key: K, value: V) {
        self.map.entry(key.clone()).or_default().push((self.depth, value));
        self.scope_keys[self.depth].push(key);
    }

    /// Insert a key-value pair at a specific (ancestor) scope depth.
    ///
    /// The entry is spliced into the value stack ordered by depth so that
    /// `get()` still returns the innermost visible entry. Used by LICM to
    /// place a hoisted value in the loop-preheader's scope rather than the
    /// current scope: siblings inside the loop body still see the value via
    /// `get`, and when the loop scope pops, the value scopes out naturally.
    pub fn insert_at_depth(&mut self, depth: usize, key: K, value: V) {
        debug_assert!(depth <= self.depth);
        let stack = self.map.entry(key.clone()).or_default();
        // Find the insertion point (keep stacks sorted by depth, stable).
        let pos = stack.iter().rposition(|(d, _)| *d <= depth).map_or(0, |i| i + 1);
        stack.insert(pos, (depth, value));
        self.scope_keys[depth].push(key);
    }

    /// Look up a key. Returns the most recently inserted value visible in the current scope.
    pub fn get(&self, key: &K) -> Option<V> {
        self.map.get(key).and_then(|stack| stack.last().map(|(_, v)| *v))
    }

    /// Current scope depth (0 = root).
    pub fn depth(&self) -> usize {
        self.depth
    }
}

#[cfg(test)]
#[path = "scoped_map_tests.rs"]
mod tests;
