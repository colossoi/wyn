//! Rust owns executable operation metadata. Egglog consumes its derived facts.
use super::dataflow::ir::Code;
use super::executable_field::{ExecutableField, Field};
use super::{output_error, OptimizeError};
use crate::ast::Span;
use crate::{LookupMap, LookupSet, StableMap};
use egglog_engine::prelude::BaseSort;
use egglog_engine::sort::{BoolSort, I64Sort};
use egglog_engine::{EGraph, Error, FullState, Value, Write};
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
    structure: Arc<SharedStructure>,
    operations: LookupMap<Value, Occurrence>,
    pub region_orders: StableMap<Value, Vec<Value>>,
}

#[derive(Clone, Copy, Default)]
pub(super) struct Position {
    pub region: Option<Value>,
    pub order: Option<i64>,
}

#[derive(Default)]
pub(super) struct ExecutableStructure {
    pub operations: LookupMap<Value, Position>,
    pub by_region: LookupMap<Value, LookupSet<Value>>,
    pub roots: StableMap<Value, Value>,
    pub targets: StableMap<Value, Value>,
    pub controls: LookupMap<(Value, i64), ControlEdge>,
    pub argument_uses: LookupMap<Value, LookupSet<ArgumentSite>>,
    pub inputs: ValueLists,
    pub yields: ValueLists,
    arities: LookupSet<usize>,
}

pub(super) struct ControlEdge {
    pub region: Value,
    pub arguments: Vec<Value>,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub(super) struct ArgumentSite(pub Value, pub i64, pub usize);

#[derive(Default)]
pub(super) struct SharedStructure(RwLock<ExecutableStructure>);

impl SharedStructure {
    pub fn read(&self) -> RwLockReadGuard<'_, ExecutableStructure> {
        self.0.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write(&self) -> RwLockWriteGuard<'_, ExecutableStructure> {
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
            structure: self.structure.clone(),
            operations: self.operations.clone(),
            region_orders: self.region_orders.clone(),
        }
    }

    pub fn register_fields(&mut self) -> Result<(), OptimizeError> {
        let sort = |name| {
            self.facts
                .get_sort_by_name(name)
                .cloned()
                .ok_or_else(|| output_error(format!("missing executable key sort {name}")))
        };
        let operation = sort("OperationKey")?;
        let region = sort("RegionKey")?;
        let function = sort("FunctionName")?;
        let name = sort("Name")?;
        let index = I64Sort.to_arcsort();
        for (field, inputs, output) in [
            (Field::Region, vec![operation.clone()], region.clone()),
            (Field::Order, vec![operation.clone()], I64Sort.to_arcsort()),
            (Field::FunctionRoot, vec![function.clone()], region.clone()),
            (Field::CallTarget, vec![operation.clone()], function),
            (
                Field::ControlRegion,
                vec![operation.clone(), index.clone()],
                region.clone(),
            ),
            (
                Field::EdgeArgument,
                vec![operation.clone(), index.clone(), index.clone()],
                name.clone(),
            ),
            (Field::Input, vec![operation.clone(), index.clone()], name.clone()),
            (Field::InputCount, vec![operation.clone()], index.clone()),
            (Field::Yield, vec![region.clone(), index.clone()], name.clone()),
            (Field::YieldCount, vec![region.clone()], index.clone()),
            (
                Field::ControlYield,
                vec![operation, index.clone(), index],
                name.clone(),
            ),
            (Field::Yielded, vec![name.clone()], BoolSort.to_arcsort()),
            (Field::RegionYields, vec![region, name], BoolSort.to_arcsort()),
        ] {
            self.facts.add_pure_primitive(
                ExecutableField {
                    structure: self.structure.clone(),
                    field,
                    inputs,
                    output,
                },
                None,
            );
        }
        Ok(())
    }

    pub fn structure(&self) -> &SharedStructure {
        &self.structure
    }

    pub fn function_roots(&self) -> impl Iterator<Item = (Value, Value)> {
        self.structure
            .read()
            .roots
            .iter()
            .map(|(&function, &region)| (function, region))
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub fn root_of(&self, function: Value) -> Option<Value> {
        self.structure.read().roots.get(&function).copied()
    }

    pub fn function_root(&self, function: Value) -> Result<Value, OptimizeError> {
        self.root_of(function).ok_or_else(|| output_error(format!("function {function:?} has no body")))
    }

    pub fn call_target(&self, operation: Value) -> Result<Value, OptimizeError> {
        self.structure
            .read()
            .targets
            .get(&operation)
            .copied()
            .ok_or_else(|| output_error(format!("call {operation:?} has no target")))
    }

    pub fn control_at(&self, operation: Value, slot: i64) -> Option<Value> {
        self.structure.read().controls.get(&(operation, slot)).map(|edge| edge.region)
    }

    pub fn control_region(&self, operation: Value, slot: i64) -> Result<Value, OptimizeError> {
        self.control_at(operation, slot)
            .ok_or_else(|| output_error(format!("operation {operation:?} has no control region at {slot}")))
    }

    pub fn controls(&self, operation: Value) -> impl Iterator<Item = (i64, Value)> + use<> {
        let graph = self.structure.read();
        [0, 1]
            .map(|slot| graph.controls.get(&(operation, slot)).map(|edge| (slot, edge.region)))
            .into_iter()
            .flatten()
    }

    pub fn argument(&self, operation: Value, edge: i64, slot: i64) -> Result<Value, OptimizeError> {
        let slot = usize::try_from(slot).map_err(|_| output_error("negative argument slot"))?;
        self.structure
            .read()
            .controls
            .get(&(operation, edge))
            .and_then(|edge| edge.arguments.get(slot))
            .copied()
            .ok_or_else(|| {
                output_error(format!(
                    "operation {operation:?} has no argument {slot} on edge {edge}"
                ))
            })
    }

    pub fn arguments(&self, operation: Value, edge: i64) -> Vec<Value> {
        self.structure
            .read()
            .controls
            .get(&(operation, edge))
            .map(|edge| edge.arguments.clone())
            .unwrap_or_default()
    }

    pub fn inputs(&self, operation: Value) -> Vec<Value> {
        self.structure.read().inputs.values.get(&operation).cloned().unwrap_or_default()
    }

    pub fn input_count(&self, operation: Value) -> i64 {
        self.structure.read().inputs.values.get(&operation).map_or(0, |values| values.len() as i64)
    }

    pub fn input_at(&self, operation: Value, slot: i64) -> Option<Value> {
        self.structure.read().inputs.values.get(&operation)?.get(usize::try_from(slot).ok()?).copied()
    }

    pub fn input(&self, operation: Value, slot: i64) -> Result<Value, OptimizeError> {
        self.input_at(operation, slot)
            .ok_or_else(|| output_error(format!("operation {operation:?} has no input {slot}")))
    }

    pub fn yield_values(&self, region: Value) -> Vec<Value> {
        self.structure.read().yields.values.get(&region).cloned().unwrap_or_default()
    }

    pub fn yield_count(&self, region: Value) -> i64 {
        self.structure.read().yields.values.get(&region).map_or(0, |values| values.len() as i64)
    }

    pub fn yield_at(&self, region: Value, slot: i64) -> Option<Value> {
        self.structure.read().yields.values.get(&region)?.get(usize::try_from(slot).ok()?).copied()
    }

    pub fn yield_value(&self, region: Value, slot: i64) -> Result<Value, OptimizeError> {
        let Some(value) = self.yield_at(region, slot) else {
            return Err(output_error(format!("region {region:?} has no yield {slot}")));
        };
        Ok(value)
    }

    pub fn region_of(&self, operation: Value) -> Option<Value> {
        self.structure.read().operations.get(&operation).and_then(|entry| entry.region)
    }

    pub fn operation_region(&self, operation: Value) -> Result<Value, OptimizeError> {
        let Some(region) = self.region_of(operation) else {
            return Err(output_error(format!("missing region of {operation:?}")));
        };
        Ok(region)
    }

    pub fn operation_order(&self, operation: Value) -> Result<i64, OptimizeError> {
        let Some(order) = self.structure.read().operations.get(&operation).and_then(|entry| entry.order)
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

    pub fn has_operation(&self, operation: Value) -> bool {
        self.operations.contains_key(&operation)
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
        let structure = &self.structure;
        self.facts.update(|facts| {
            f(State {
                facts,
                operations,
                structure,
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
    structure: &'operations SharedStructure,
}

impl State<'_, '_, '_> {
    pub fn set_function_root(&mut self, function: Value, region: Value) {
        self.structure.write().roots.insert(function, region);
    }

    pub fn set_call_target(&mut self, operation: Value, target: Value) {
        self.structure.write().targets.insert(operation, target);
    }

    pub fn remove_call_target(&mut self, operation: Value) {
        self.structure.write().targets.shift_remove(&operation);
    }

    pub fn set_inputs(&mut self, operation: Value, inputs: Vec<Value>) -> Result<(), Error> {
        register_arity(&mut self.facts, self.structure, inputs.len())?;
        self.structure.write().inputs.replace(operation, inputs);
        Ok(())
    }

    pub fn set_input(&mut self, operation: Value, slot: i64, input: Value) -> Result<(), Error> {
        self.structure.write().inputs.redirect(operation, slot, input)
    }

    pub fn set_yields(&mut self, region: Value, values: Vec<Value>) -> Result<(), Error> {
        register_arity(&mut self.facts, self.structure, values.len())?;
        self.structure.write().yields.replace(region, values);
        Ok(())
    }

    pub fn set_yield(&mut self, region: Value, slot: i64, value: Value) -> Result<(), Error> {
        self.structure.write().yields.redirect(region, slot, value)
    }

    pub fn set_control(&mut self, operation: Value, slot: i64, child: Value) {
        assert!(
            (0..2).contains(&slot),
            "invalid control region slot {slot} on {operation:?}"
        );
        match self.structure.write().controls.entry((operation, slot)) {
            Entry::Occupied(mut edge) => edge.get_mut().region = child,
            Entry::Vacant(edge) => {
                edge.insert(ControlEdge {
                    region: child,
                    arguments: Vec::new(),
                });
            }
        }
    }

    pub fn set_argument(
        &mut self,
        ArgumentSite(operation, edge, slot): ArgumentSite,
        value: Value,
    ) -> Result<(), Error> {
        let mut graph = self.structure.write();
        let Some(control) = graph.controls.get_mut(&(operation, edge)) else {
            return Err(Error::BackendError(format!(
                "missing control edge {edge} on {operation:?}"
            )));
        };
        let args = &mut control.arguments;
        let site = ArgumentSite(operation, edge, slot);
        if slot == args.len() {
            args.push(value);
        } else if let Some(argument) = args.get_mut(slot) {
            let old = std::mem::replace(argument, value);
            if old == value {
                return Ok(());
            }
            remove_use(&mut graph.argument_uses, site, old);
        } else {
            return Err(Error::BackendError(format!(
                "missing control argument {slot} on {operation:?}"
            )));
        }
        graph.argument_uses.entry(value).or_default().insert(site);
        Ok(())
    }

    pub fn clear_control(&mut self, operation: Value) {
        let mut structure = self.structure.write();
        for slot in 0..2 {
            if let Some(edge) = structure.controls.remove(&(operation, slot)) {
                for (index, value) in edge.arguments.into_iter().enumerate() {
                    remove_use(
                        &mut structure.argument_uses,
                        ArgumentSite(operation, slot, index),
                        value,
                    );
                }
            }
        }
    }

    pub fn set_region(&mut self, operation: Value, region: Value) {
        let mut structure = self.structure.write();
        if let Some(previous) = structure.operations.entry(operation).or_default().region.replace(region) {
            if previous == region {
                return;
            }
            if let Some(operations) = structure.by_region.get_mut(&previous) {
                operations.remove(&operation);
            }
        }
        structure.by_region.entry(region).or_default().insert(operation);
    }

    pub fn set_order(&mut self, operation: Value, order: i64) {
        self.structure.write().operations.entry(operation).or_default().order = Some(order);
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
        self.clear_control(operation);
        let mut structure = self.structure.write();
        structure.targets.shift_remove(&operation);
        structure.inputs.replace(operation, Vec::new());
        if let Some(region) = structure.operations.remove(&operation).and_then(|position| position.region) {
            if let Some(operations) = structure.by_region.get_mut(&region) {
                operations.remove(&operation);
            }
        }
    }
}

#[derive(Default)]
pub(super) struct ValueLists {
    pub values: LookupMap<Value, Vec<Value>>,
    pub uses: LookupMap<Value, LookupSet<(Value, i64)>>,
}

impl ValueLists {
    fn replace(&mut self, owner: Value, values: Vec<Value>) {
        if let Some(old) = self.values.remove(&owner) {
            for (slot, value) in old.into_iter().enumerate() {
                remove_use(&mut self.uses, (owner, slot as i64), value);
            }
        }
        for (slot, &value) in values.iter().enumerate() {
            self.uses.entry(value).or_default().insert((owner, slot as i64));
        }
        if !values.is_empty() {
            self.values.insert(owner, values);
        }
    }

    fn redirect(&mut self, owner: Value, slot: i64, value: Value) -> Result<(), Error> {
        let Ok(index) = usize::try_from(slot) else {
            return Err(Error::BackendError(format!(
                "negative value-list index {slot} on {owner:?}"
            )));
        };
        let Some(current) = self.values.get_mut(&owner).and_then(|values| values.get_mut(index)) else {
            return Err(Error::BackendError(format!(
                "missing value at index {slot} on {owner:?}"
            )));
        };
        if *current != value {
            let old = std::mem::replace(current, value);
            remove_use(&mut self.uses, (owner, slot), old);
            self.uses.entry(value).or_default().insert((owner, slot));
        }
        Ok(())
    }
}

fn register_arity(
    facts: &mut FullState<'_, '_>,
    structure: &SharedStructure,
    count: usize,
) -> Result<(), Error> {
    let mut structure = structure.write();
    if !structure.arities.contains(&count) {
        for slot in 0..count {
            facts.add("AritySlot", (count as i64, slot as i64))?;
        }
        structure.arities.insert(count);
    }
    Ok(())
}

fn remove_use<Site: Eq + std::hash::Hash>(
    uses: &mut LookupMap<Value, LookupSet<Site>>,
    site: Site,
    value: Value,
) {
    if let Entry::Occupied(mut uses) = uses.entry(value) {
        uses.get_mut().remove(&site);
        if uses.get().is_empty() {
            uses.remove();
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
