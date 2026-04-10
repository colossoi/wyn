//! Scoped hashmap for elaboration.
//!
//! Each scope corresponds to a subtree in the dominator tree.
//! Values inserted in a child scope are invisible after popping back to the parent.

use std::collections::HashMap;
use std::hash::Hash;

/// A hashmap with push/pop scope operations.
pub struct ScopedMap<K: Hash + Eq, V> {
    /// Key → stack of (depth, value). Top of stack is the most recent.
    map: HashMap<K, Vec<(usize, V)>>,
    /// Current scope depth (0 = root).
    depth: usize,
    /// Keys inserted at each depth, for cleanup on pop.
    scope_keys: Vec<Vec<K>>,
}

impl<K: Hash + Eq + Clone, V: Copy> ScopedMap<K, V> {
    pub fn new() -> Self {
        ScopedMap {
            map: HashMap::new(),
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
        self.map
            .entry(key.clone())
            .or_default()
            .push((self.depth, value));
        self.scope_keys[self.depth].push(key);
    }

    /// Look up a key. Returns the most recently inserted value visible in the current scope.
    pub fn get(&self, key: &K) -> Option<V> {
        self.map.get(key).and_then(|stack| stack.last().map(|(_, v)| *v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_scoping() {
        let mut m = ScopedMap::new();
        m.insert(1, "a");
        assert_eq!(m.get(&1), Some("a"));

        m.push_scope();
        assert_eq!(m.get(&1), Some("a")); // visible from parent
        m.insert(2, "b");
        assert_eq!(m.get(&2), Some("b"));

        m.pop_scope();
        assert_eq!(m.get(&1), Some("a"));
        assert_eq!(m.get(&2), None); // gone after pop
    }

    #[test]
    fn shadow_and_restore() {
        let mut m = ScopedMap::new();
        m.insert(1, 10);
        m.push_scope();
        m.insert(1, 20); // shadow
        assert_eq!(m.get(&1), Some(20));
        m.pop_scope();
        assert_eq!(m.get(&1), Some(10)); // restored
    }

    #[test]
    fn sibling_scopes_independent() {
        let mut m = ScopedMap::new();
        m.insert(0, "root");

        // Child A
        m.push_scope();
        m.insert(1, "a");
        assert_eq!(m.get(&1), Some("a"));
        m.pop_scope();

        // Child B — should not see child A's insertions
        m.push_scope();
        assert_eq!(m.get(&1), None);
        m.insert(1, "b");
        assert_eq!(m.get(&1), Some("b"));
        m.pop_scope();

        assert_eq!(m.get(&0), Some("root"));
        assert_eq!(m.get(&1), None);
    }
}
