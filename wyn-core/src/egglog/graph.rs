//! Rust owns executable operation metadata. Egglog consumes its derived facts.
use super::dataflow::ir::Code;
use super::{output_error, OptimizeError};
use crate::ast::Span;
use crate::{LookupMap, StableMap};
use egglog_engine::{EGraph, Error, FullState, Value};
use std::collections::hash_map::Entry;
use std::ops::{Deref, DerefMut};

#[derive(Clone)]
struct Occurrence {
    code: Code,
    span: Option<Span>,
}

#[derive(Clone, Default)]
pub(super) struct Graph {
    facts: EGraph,
    operations: LookupMap<Value, Occurrence>,
    pub region_orders: StableMap<Value, Vec<Value>>,
}

impl Graph {
    pub fn code(&self, operation: Value) -> Result<Code, OptimizeError> {
        let Some(occurrence) = self.operations.get(&operation) else {
            return Err(output_error(format!(
                "missing executable operation {operation:?}"
            )));
        };
        Ok(occurrence.code.clone())
    }

    pub fn span(&self, operation: Value) -> Option<Span> {
        self.operations.get(&operation).and_then(|occurrence| occurrence.span)
    }

    pub fn region_order(&self, region: Value) -> Result<&[Value], OptimizeError> {
        let Some(operations) = self.region_orders.get(&region) else {
            return Err(output_error(format!("missing ordered region {region:?}")));
        };
        Ok(operations)
    }

    pub fn codes(&self) -> impl Iterator<Item = (Value, &Code)> {
        self.operations.iter().map(|(&key, occurrence)| (key, &occurrence.code))
    }

    pub fn update<R>(&mut self, f: impl FnOnce(State<'_, '_, '_>) -> Result<R, Error>) -> Result<R, Error> {
        let operations = &mut self.operations;
        self.facts.update(|facts| f(State { facts, operations }))
    }

    pub fn copy_occurrence(&mut self, original: Value, copy: Value) -> Result<(), OptimizeError> {
        let Some(occurrence) = self.operations.get(&original) else {
            return Err(output_error(format!("missing executable operation {original:?}")));
        };
        self.operations.insert(copy, occurrence.clone());
        Ok(())
    }
}

impl Deref for Graph {
    type Target = EGraph;

    fn deref(&self) -> &Self::Target {
        &self.facts
    }
}

impl DerefMut for Graph {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.facts
    }
}

pub(super) struct State<'a, 'db, 'operations> {
    facts: FullState<'a, 'db>,
    operations: &'operations mut LookupMap<Value, Occurrence>,
}

impl State<'_, '_, '_> {
    pub fn set_code(&mut self, operation: Value, code: Code) {
        match self.operations.entry(operation) {
            Entry::Occupied(mut entry) => entry.get_mut().code = code,
            Entry::Vacant(entry) => {
                entry.insert(Occurrence { code, span: None });
            }
        }
    }

    pub fn set_span(&mut self, operation: Value, span: Span) -> Result<(), OptimizeError> {
        let Some(occurrence) = self.operations.get_mut(&operation) else {
            return Err(output_error(format!(
                "missing executable operation {operation:?}"
            )));
        };
        occurrence.span = Some(span);
        Ok(())
    }

    pub fn remove_operation(&mut self, operation: Value) {
        self.operations.remove(&operation);
    }
}

impl<'a, 'db, 'operations> Deref for State<'a, 'db, 'operations> {
    type Target = FullState<'a, 'db>;

    fn deref(&self) -> &Self::Target {
        &self.facts
    }
}

impl DerefMut for State<'_, '_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.facts
    }
}
