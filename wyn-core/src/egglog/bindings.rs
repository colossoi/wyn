//! Checkpointed bindings shared by source import and code emission.
use crate::LookupMap;
use std::{hash::Hash, ops::Deref};

pub(super) struct Bindings<K, V> {
    values: LookupMap<K, V>,
    changes: Vec<(K, Option<V>)>,
}

impl<K: Copy + Eq + Hash, V> Bindings<K, V> {
    pub(super) fn checkpoint(&self) -> usize {
        self.changes.len()
    }

    pub(super) fn insert(&mut self, symbol: K, value: V) {
        self.changes.push((symbol, self.values.insert(symbol, value)));
    }

    pub(super) fn remove(&mut self, key: &K) {
        self.changes.push((*key, self.values.remove(key)));
    }

    pub(super) fn restore(&mut self, checkpoint: usize) {
        while self.changes.len() > checkpoint {
            let Some((symbol, previous)) = self.changes.pop() else {
                break;
            };
            if let Some(value) = previous {
                self.values.insert(symbol, value);
            } else {
                self.values.remove(&symbol);
            }
        }
    }
}

impl<K, V> Default for Bindings<K, V> {
    fn default() -> Self {
        Self {
            values: LookupMap::default(),
            changes: Vec::new(),
        }
    }
}
impl<K, V> Deref for Bindings<K, V> {
    type Target = LookupMap<K, V>;
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<K, V> From<LookupMap<K, V>> for Bindings<K, V> {
    fn from(values: LookupMap<K, V>) -> Self {
        Self {
            values,
            changes: Vec::new(),
        }
    }
}
