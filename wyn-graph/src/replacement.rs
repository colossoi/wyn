use std::collections::HashMap;
use std::hash::Hash;

/// A rejected directed replacement. IDs describe the requested edge.
#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ReplacementError<Id> {
    #[error("replacement {old:?} -> {survivor:?} creates a cycle")]
    Cycle {
        old: Id,
        survivor: Id,
    },
    #[error("replacement {old:?} -> {survivor:?} conflicts with survivor {existing:?}")]
    Conflict {
        old: Id,
        survivor: Id,
        existing: Id,
    },
}

/// Directed substitutions with one terminal survivor per replaced ID.
/// IDs absent from the private map survive; checked insertion keeps it acyclic.
#[derive(Debug)]
pub struct ReplacementForest<Id> {
    replacements: HashMap<Id, Id>,
}

impl<Id> Default for ReplacementForest<Id> {
    fn default() -> Self {
        Self {
            replacements: HashMap::new(),
        }
    }
}

impl<Id: Copy + Eq + Hash> ReplacementForest<Id> {
    /// Create an empty forest.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the terminal survivor, compressing every edge on the path.
    pub fn resolve(&mut self, mut id: Id) -> Id {
        let mut survivor = id;
        while let Some(&next) = self.replacements.get(&survivor) {
            survivor = next;
        }
        while let Some(next) = self.replacements.get_mut(&id) {
            id = std::mem::replace(next, survivor);
        }
        survivor
    }

    /// Assign this exact source ID, never redirecting its existing survivor.
    /// Returns whether an assignment was added. Errors preserve all resolutions.
    /// Repeating a resolved target is unchanged; a different target conflicts.
    /// Self-replacement and edges back to `old` are cycles.
    pub fn replace(&mut self, old: Id, survivor: Id) -> Result<bool, ReplacementError<Id>> {
        let target = self.resolve(survivor);
        if old == survivor || old == target {
            return Err(ReplacementError::Cycle { old, survivor });
        }
        let existing = self.resolve(old);
        if existing == target {
            return Ok(false);
        }
        if existing != old {
            return Err(ReplacementError::Conflict {
                old,
                survivor,
                existing,
            });
        }
        self.replacements.insert(old, target);
        Ok(true)
    }

    /// Export every replaced ID directly to its terminal, excluding identities.
    pub fn into_map(mut self) -> HashMap<Id, Id> {
        for id in self.replacements.keys().copied().collect::<Vec<_>>() {
            self.resolve(id);
        }
        self.replacements
    }
}

#[cfg(test)]
#[path = "replacement_tests.rs"]
mod tests;
