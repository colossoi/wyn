//! Rust owns executable operation metadata. Egglog consumes its derived facts.
use super::dataflow::ir::Code;
use super::position_field::{Field, PositionField};
use super::{output_error, OptimizeError};
use crate::ast::Span;
use crate::{LookupMap, LookupSet, StableMap};
use egglog_engine::prelude::BaseSort;
use egglog_engine::sort::I64Sort;
use egglog_engine::{EGraph, Error, FullState, Value};
use std::collections::hash_map::Entry;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[derive(Clone)]
struct Occurrence {
    code: Code,
    span: Option<Span>,
}

#[derive(Default)]
pub(super) struct Graph {
    facts: EGraph,
    positions: Arc<Positions>,
    operations: LookupMap<Value, Occurrence>,
    pub region_orders: StableMap<Value, Vec<Value>>,
}

#[derive(Clone, Copy, Default)]
pub(super) struct Position {
    pub region: Option<Value>,
    pub order: Option<i64>,
}

#[derive(Default)]
pub(super) struct RegionPositions {
    pub operations: LookupMap<Value, Position>,
    pub by_region: LookupMap<Value, LookupSet<Value>>,
}

#[derive(Default)]
pub(super) struct Positions(RwLock<RegionPositions>);

impl Positions {
    pub fn read(&self) -> RwLockReadGuard<'_, RegionPositions> {
        self.0.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write(&self) -> RwLockWriteGuard<'_, RegionPositions> {
        self.0.write().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Graph {
    pub fn from_facts(facts: EGraph) -> Self {
        Self {
            facts,
            ..Self::default()
        }
    }

    #[cfg(test)]
    pub fn proof_snapshot(&self) -> Self {
        Self {
            facts: self.facts.clone(),
            positions: self.positions.clone(),
            operations: self.operations.clone(),
            region_orders: self.region_orders.clone(),
        }
    }

    pub fn register_positions(&mut self) -> Result<(), OptimizeError> {
        let Some(operation) = self.facts.get_sort_by_name("OperationKey").cloned() else {
            return Err(output_error("missing operation key sort"));
        };
        let Some(region) = self.facts.get_sort_by_name("RegionKey").cloned() else {
            return Err(output_error("missing region key sort"));
        };
        for (field, output) in [(Field::Region, region), (Field::Order, I64Sort.to_arcsort())] {
            self.facts.add_pure_primitive(
                PositionField {
                    positions: self.positions.clone(),
                    field,
                    operation: operation.clone(),
                    output,
                },
                None,
            );
        }
        Ok(())
    }

    pub fn positions(&self) -> &Positions {
        &self.positions
    }

    pub fn region_of(&self, operation: Value) -> Option<Value> {
        self.positions.read().operations.get(&operation).and_then(|entry| entry.region)
    }

    pub fn operation_region(&self, operation: Value) -> Result<Value, OptimizeError> {
        let Some(region) = self.region_of(operation) else {
            return Err(output_error(format!("missing region of {operation:?}")));
        };
        Ok(region)
    }

    pub fn operation_order(&self, operation: Value) -> Result<i64, OptimizeError> {
        let Some(order) = self.positions.read().operations.get(&operation).and_then(|entry| entry.order)
        else {
            return Err(output_error(format!("missing position of {operation:?}")));
        };
        Ok(order)
    }

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
        let positions = &self.positions;
        self.facts.update(|facts| {
            f(State {
                facts,
                operations,
                positions,
            })
        })
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
    positions: &'operations Positions,
}

impl State<'_, '_, '_> {
    pub fn set_region(&mut self, operation: Value, region: Value) {
        let mut positions = self.positions.write();
        if let Some(previous) = positions.operations.entry(operation).or_default().region.replace(region) {
            if previous == region {
                return;
            }
            if let Some(operations) = positions.by_region.get_mut(&previous) {
                operations.remove(&operation);
            }
        }
        positions.by_region.entry(region).or_default().insert(operation);
    }

    pub fn set_order(&mut self, operation: Value, order: i64) {
        self.positions.write().operations.entry(operation).or_default().order = Some(order);
    }

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
        let mut positions = self.positions.write();
        if let Some(region) = positions.operations.remove(&operation).and_then(|position| position.region) {
            if let Some(operations) = positions.by_region.get_mut(&region) {
                operations.remove(&operation);
            }
        }
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
