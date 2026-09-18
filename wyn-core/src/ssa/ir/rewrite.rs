//! Structural SSA value substitution, independent of Wyn operations and types.

use super::{Function, ValueId, ValueRef};
use std::collections::HashMap;

/// An instruction whose SSA operands can be rewritten.
pub trait VisitValues {
    fn values(&self) -> Vec<ValueRef>;
    fn visit_values_mut(&mut self, visit: &mut dyn FnMut(&mut ValueRef));
}

/// A transitively resolved set of SSA value replacements.
#[derive(Default)]
pub(crate) struct Substitutions {
    values: HashMap<ValueId, ValueRef>,
}

impl Substitutions {
    pub fn insert(&mut self, value: ValueId, replacement: ValueRef) {
        self.values.insert(value, replacement);
    }

    pub fn resolve(&mut self, value: &mut ValueRef) {
        let mut cursor = *value;
        while let ValueRef::Ssa(id) = *value {
            let Some(&next) = self.values.get(&id) else {
                break;
            };
            *value = next;
        }
        while let ValueRef::Ssa(id) = cursor {
            let Some(next) = self.values.get_mut(&id) else {
                break;
            };
            cursor = std::mem::replace(next, *value);
        }
    }

    /// Rewrite every remaining use and remove the superseded value records.
    pub fn finish<I: VisitValues, T>(&mut self, function: &mut Function<I, T>) {
        for node in function.insts.values_mut() {
            node.data.visit_values_mut(&mut |value| self.resolve(value));
        }
        for block in function.blocks.values_mut() {
            block.term.visit_nodes_mut(|value| self.resolve(value));
        }
        for value in self.values.keys() {
            function.values.remove(*value);
        }
    }
}
