//! Foundational, domain-independent data structures shared by Wyn crates.
//!
//! This crate sits at the bottom of the Wyn dependency graph. It must not
//! depend on compiler phases or other Wyn crates.

#![forbid(unsafe_code)]

pub mod persistent_sets;

use std::hash::Hash;
use std::marker::PhantomData;

/// Use for maps whose iteration order affects program output (binding
/// allocation, code emission order, etc.). Insertion order is stable
/// across compiles; `HashMap`'s randomized hasher is not.
pub type StableMap<K, V> = indexmap::IndexMap<K, V>;

/// Use for sets whose iteration follows the values' [`Ord`] ordering.
pub type SortedSet<T> = std::collections::BTreeSet<T>;

/// Use for maps consulted only via `get`/`contains_key`. Iteration
/// order doesn't escape into observable output, so `HashMap`'s
/// per-process random hash is fine — and we get the slightly faster
/// lookups in exchange.
pub type LookupMap<K, V> = std::collections::HashMap<K, V>;

/// Set companion to [`LookupMap`].
pub type LookupSet<T> = std::collections::HashSet<T>;

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

/// Value-to-ID index for an externally owned arena.
///
/// Build once per rewrite session and allocate through this index thereafter.
/// The caller must use the same arena throughout the session and must not
/// mutate indexed values or allocate behind the index's back. Rebuild the index
/// after such changes. This lets a sidecar retain ownership of its arenas.
#[derive(Debug, Clone)]
pub struct InternIndex<Id, T> {
    by_value: LookupMap<T, Id>,
}

impl<Id, T> Default for InternIndex<Id, T> {
    fn default() -> Self {
        Self {
            by_value: LookupMap::new(),
        }
    }
}

impl<Id, T> InternIndex<Id, T>
where
    Id: From<u32> + Copy + Eq + Hash,
{
    /// Index existing records without changing their IDs. If the arena contains
    /// duplicates, the last record in iteration order becomes the lookup result.
    pub fn from_arena(arena: &IdArena<Id, T>) -> Self
    where
        T: Clone + Eq + Hash,
    {
        Self {
            by_value: arena.iter().map(|(&id, value)| (value.clone(), id)).collect(),
        }
    }

    pub fn intern<Q>(&mut self, arena: &mut IdArena<Id, T>, value: &Q) -> Id
    where
        T: std::borrow::Borrow<Q> + Clone + Eq + Hash,
        Q: Eq + Hash + ToOwned<Owned = T> + ?Sized,
    {
        if let Some(id) = self.get(value) {
            return id;
        }
        let value = value.to_owned();
        let id = arena.alloc(value.clone());
        self.by_value.insert(value, id);
        id
    }

    pub fn get<Q>(&self, value: &Q) -> Option<Id>
    where
        T: std::borrow::Borrow<Q> + Eq + Hash,
        Q: Eq + Hash + ?Sized,
    {
        self.by_value.get(value).copied()
    }
}

/// Append-only bidirectional interner for values with compiler-assigned IDs.
///
/// Equal values share one ID. The arena provides ID-to-value resolution while
/// the lookup map provides value-to-ID lookup and deduplication.
#[derive(Debug, Clone)]
pub struct Interner<Id, T> {
    arena: IdArena<Id, T>,
    index: InternIndex<Id, T>,
}

impl<Id, T> Interner<Id, T>
where
    Id: From<u32> + Copy + Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            arena: IdArena::new(),
            index: InternIndex::default(),
        }
    }

    pub fn intern<Q>(&mut self, value: &Q) -> Id
    where
        T: std::borrow::Borrow<Q> + Clone + Eq + Hash,
        Q: Eq + Hash + ToOwned<Owned = T> + ?Sized,
    {
        self.index.intern(&mut self.arena, value)
    }

    pub fn get<Q>(&self, value: &Q) -> Option<Id>
    where
        T: std::borrow::Borrow<Q> + Eq + Hash,
        Q: Eq + Hash + ?Sized,
    {
        self.index.get(value)
    }

    pub fn resolve(&self, id: Id) -> &T {
        &self.arena[id]
    }

    pub fn resolve_cloned(&self, ids: impl IntoIterator<Item = Id>) -> Vec<T>
    where
        T: Clone,
    {
        ids.into_iter().map(|id| self.resolve(id).clone()).collect()
    }

    /// Borrow the immutable interned records.
    pub fn arena(&self) -> &IdArena<Id, T> {
        &self.arena
    }

    /// Finish interning and retain only the records and their IDs.
    pub fn into_arena(self) -> IdArena<Id, T> {
        self.arena
    }

    pub fn len(&self) -> usize {
        self.arena.len()
    }

    pub fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }
}

impl<Id, T> Default for Interner<Id, T>
where
    Id: From<u32> + Copy + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
