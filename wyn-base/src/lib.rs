//! Foundational, domain-independent data structures shared by Wyn crates.
//!
//! This crate sits at the bottom of the Wyn dependency graph. It must not
//! depend on compiler phases or other Wyn crates.

#![forbid(unsafe_code)]

use std::hash::Hash;
use std::marker::PhantomData;

type StableMap<K, V> = indexmap::IndexMap<K, V>;

/// Generic counter for generating unique IDs.
///
/// The ID type must implement `From<u32>` to convert the raw counter value.
#[derive(Debug, Clone)]
pub struct IdSource<Id> {
    next_id: u32,
    _phantom: PhantomData<Id>,
}

impl<Id: From<u32>> IdSource<Id> {
    pub fn new() -> Self {
        Self {
            next_id: 0,
            _phantom: PhantomData,
        }
    }

    pub fn next_id(&mut self) -> Id {
        let raw = self.next_id;
        self.next_id = self.next_id.checked_add(1).expect("compiler ID space exhausted");
        Id::from(raw)
    }

    /// Read the next ID without consuming it. Useful for "would-allocate"
    /// dry runs: peek, attempt, then commit via [`next_id`](Self::next_id)
    /// only on success.
    pub fn peek_id(&self) -> Id {
        Id::from(self.next_id)
    }
}

impl<Id: From<u32>> Default for IdSource<Id> {
    fn default() -> Self {
        Self::new()
    }
}

/// Arena that allocates IDs and stores associated items.
///
/// Combines ID generation with storage, ensuring each item gets a unique ID.
/// Uses insertion-ordered storage for deterministic iteration.
#[derive(Debug, Clone)]
pub struct IdArena<Id, T> {
    source: IdSource<Id>,
    items: StableMap<Id, T>,
}

impl<Id: From<u32> + Copy + Eq + Hash, T> IdArena<Id, T> {
    pub fn new() -> Self {
        Self {
            source: IdSource::new(),
            items: StableMap::new(),
        }
    }

    /// Allocate a new ID and store the item.
    pub fn alloc(&mut self, item: T) -> Id {
        let id = self.source.next_id();
        self.items.insert(id, item);
        id
    }

    /// Allocate a new ID without storing anything yet.
    /// Use [`insert`](Self::insert) later to store the item.
    pub fn alloc_id(&mut self) -> Id {
        self.source.next_id()
    }

    /// Insert an item with a pre-allocated ID.
    /// Panics if the ID is already in use.
    pub fn insert(&mut self, id: Id, item: T) {
        let old = self.items.insert(id, item);
        assert!(old.is_none(), "IdArena::insert called with duplicate ID");
    }

    /// Get an item by ID.
    pub fn get(&self, id: Id) -> Option<&T> {
        self.items.get(&id)
    }

    /// Get a mutable reference to an item by ID.
    pub fn get_mut(&mut self, id: Id) -> Option<&mut T> {
        self.items.get_mut(&id)
    }

    /// Iterate over all `(ID, item)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Id, &T)> {
        self.items.iter()
    }

    /// Iterate over the IDs that currently name stored items.
    pub fn ids(&self) -> impl Iterator<Item = Id> + '_ {
        self.items.keys().copied()
    }

    /// Iterate over all items without their IDs.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.items.values()
    }

    /// Number of stored items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the arena contains no stored items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl<Id: From<u32> + Copy + Eq + Hash, T> Default for IdArena<Id, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Id: From<u32> + Copy + Eq + Hash, T> std::ops::Index<Id> for IdArena<Id, T> {
    type Output = T;

    fn index(&self, id: Id) -> &Self::Output {
        &self.items[&id]
    }
}

impl<Id: From<u32> + Copy + Eq + Hash, T> std::ops::IndexMut<Id> for IdArena<Id, T> {
    fn index_mut(&mut self, id: Id) -> &mut Self::Output {
        &mut self.items[&id]
    }
}

impl<Id: From<u32> + Copy + Eq + Hash, T> IntoIterator for IdArena<Id, T> {
    type Item = (Id, T);
    type IntoIter = indexmap::map::IntoIter<Id, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a, Id: From<u32> + Copy + Eq + Hash, T> IntoIterator for &'a IdArena<Id, T> {
    type Item = (&'a Id, &'a T);
    type IntoIter = indexmap::map::Iter<'a, Id, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<'a, Id: From<u32> + Copy + Eq + Hash, T> IntoIterator for &'a mut IdArena<Id, T> {
    type Item = (&'a Id, &'a mut T);
    type IntoIter = indexmap::map::IterMut<'a, Id, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter_mut()
    }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
