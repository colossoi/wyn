//! Binding allocator for automatic descriptor set/binding assignment.

use std::collections::HashMap;

/// Allocates descriptor set and binding numbers.
#[derive(Debug, Default)]
pub struct BindingAllocator {
    next_binding: HashMap<u32, u32>,
}

impl BindingAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate the next binding in the given set.
    pub fn allocate(&mut self, set: u32) -> (u32, u32) {
        let binding = self.next_binding.entry(set).or_insert(0);
        let result = (set, *binding);
        *binding += 1;
        result
    }

    /// Allocate bindings for compute entry parameters.
    /// All go to set 0, with sequential binding numbers.
    pub fn allocate_compute_params(&mut self, count: usize) -> Vec<(u32, u32)> {
        (0..count).map(|_| self.allocate(0)).collect()
    }
}
