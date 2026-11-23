//! Structural state owned by a parallel fusion pass until it publishes its edits.
use super::Domain;
use crate::egglog::dataflow::{body::BodyGraph, cse, edit, ir::Code};
use crate::egglog::graph::CertificateSite;
use crate::egglog::graph::{CollectiveInput, ParallelResults};
use crate::egglog::graph::{Graph, State};
use crate::egglog::query::Query;
use crate::egglog::{analysis, output_error, timing, OptimizeError};
use crate::memory_ir::{BodyRole, Definition, OperationId, RegionId, UseSite, ValueRef};
use crate::{LookupMap, LookupSet, SortedSet};
use egglog_engine::{RawValues, Value};
use slotmap::SecondaryMap;
use std::mem::take;
use wyn_graph::ReplacementForest;

mod accumulator;
mod branch;
mod guard;
mod relocation;
use guard::Guard;

fn value_name(query: Query<'_>, value: Value) -> Result<Value, OptimizeError> {
    let Some(name) = query.constructor("ValueNameRef", value)? else {
        return Err(output_error("missing value reference in fusion metadata"));
    };
    Ok(name)
}

pub(super) type ClauseId = (OperationId, i64);
type RegionOrders = Vec<(Value, Vec<Value>)>;

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub(super) enum Site {
    Operand(UseSite),
    Attribute(OperationId),
    Certificate(OperationId),
    Region(RegionId),
    Result(ClauseId),
    Neutral(ClauseId),
}

#[derive(Default)]
struct Uses {
    values: LookupMap<Site, LookupSet<Value>>,
    sites: LookupMap<Value, LookupSet<Site>>,
}

impl Uses {
    fn add(&mut self, site: Site, value: Value) {
        self.values.entry(site).or_default().insert(value);
        self.sites.entry(value).or_default().insert(site);
    }

    fn remove(&mut self, site: Site) -> impl Iterator<Item = Value> + '_ {
        self.values.remove(&site).into_iter().flatten().filter(move |name| {
            let uses = self.sites.entry(*name).or_default();
            uses.remove(&site);
            uses.is_empty()
        })
    }

    fn at(&self, site: Site) -> impl Iterator<Item = (Site, Value)> + '_ {
        self.values.get(&site).into_iter().flatten().map(move |&name| (site, name))
    }

    fn dead(&self, name: Value) -> bool {
        self.sites.get(&name).is_none_or(LookupSet::is_empty)
    }
}

#[derive(Clone, Default)]
pub(super) struct Effects {
    pub reads: LookupSet<Value>,
    pub writes: LookupSet<Value>,
    pub traps: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ClauseKind {
    Collected,
    Compacted,
    Folded,
    Contributed,
}

pub(super) struct Clause {
    owner: OperationId,
    pub kind: ClauseKind,
    pub values: Vec<Value>,
    pub neutral: Vec<Value>,
    pub predicate: Option<Value>,
}

impl Clause {
    fn can_prune(&self) -> bool {
        self.values.len() == 1 && matches!(self.kind, ClauseKind::Collected | ClauseKind::Compacted)
    }
}

struct Output {
    original: (OperationId, i64),
    clause: ClauseId,
    field: usize,
    prefix: bool,
}

#[derive(Default)]
struct Results {
    clauses: Vec<ClauseId>,
    outputs: Vec<Output>,
}

pub(super) struct Body {
    pub graph: BodyGraph,
    pub positions: SecondaryMap<OperationId, (RegionId, usize)>,
    pub regions: LookupMap<Value, RegionId>,
    pub domains: SecondaryMap<OperationId, Domain>,
    pub barriers: LookupSet<OperationId>,
    pub pinned: LookupSet<OperationId>,
    pub effects: SecondaryMap<OperationId, Effects>,
    pub unknown_aliases: bool,
    pub external: Option<Value>,
    parents: SecondaryMap<RegionId, OperationId>,
    results: SecondaryMap<OperationId, Results>,
    clauses: LookupMap<ClauseId, Clause>,
    uses: Uses,
    substitutions: ReplacementForest<Value>,
    operand_changes: LookupMap<UseSite, Value>,
    edited: SortedSet<OperationId>,
    moved: SortedSet<RegionId>,
    relocated: LookupSet<OperationId>,
    reads_before_lift: SecondaryMap<OperationId, LookupSet<Value>>,
    removed: SortedSet<OperationId>,
    pruned: LookupSet<ClauseId>,
    dirty: SortedSet<RegionId>,
    dead: Vec<Value>,
    guards: Vec<Guard>,
    changed_clauses: LookupSet<ClauseId>,
    packed_outputs: LookupSet<(OperationId, i64)>,
    accumulator_origins: SecondaryMap<OperationId, Vec<OperationId>>,
    joined_yields: SecondaryMap<RegionId, Vec<RegionId>>,
    joined_branches: SortedSet<OperationId>,
}

impl Body {
    pub fn read(
        query: Query<'_>,
        domains: LookupMap<Value, Domain>,
        barriers: LookupSet<Value>,
    ) -> Result<Self, OptimizeError> {
        let _timing = timing::span("Rust parallel fusion graph import");
        let graph = BodyGraph::read(query)?;
        let mut body = Self {
            regions: graph.regions.iter().map(|(key, &region)| (region, key)).collect(),
            domains: domains
                .into_iter()
                .filter_map(|(op, domain)| Some((*graph.operation_ids.get(&op)?, domain)))
                .collect(),
            barriers: barriers.into_iter().filter_map(|op| graph.operation_ids.get(&op).copied()).collect(),
            pinned: LookupSet::new(),
            graph,
            positions: SecondaryMap::new(),
            effects: SecondaryMap::new(),
            unknown_aliases: false,
            external: query.constructor("ExternalMemory", RawValues(vec![]))?,
            parents: SecondaryMap::new(),
            results: SecondaryMap::new(),
            clauses: LookupMap::new(),
            uses: Uses::default(),
            substitutions: ReplacementForest::new(),
            operand_changes: LookupMap::new(),
            edited: SortedSet::default(),
            moved: SortedSet::default(),
            relocated: LookupSet::new(),
            reads_before_lift: SecondaryMap::new(),
            removed: SortedSet::default(),
            pruned: LookupSet::new(),
            dirty: SortedSet::default(),
            dead: Vec::new(),
            guards: Vec::new(),
            changed_clauses: LookupSet::new(),
            packed_outputs: LookupSet::new(),
            accumulator_origins: SecondaryMap::new(),
            joined_yields: SecondaryMap::new(),
            joined_branches: SortedSet::new(),
        };
        for (region, block) in &body.graph.ir.regions {
            for (index, &op) in block.operations.iter().enumerate() {
                body.positions.insert(op, (region, index));
            }
            for (i, value) in block.yields.iter().enumerate() {
                body.uses.add(Site::Operand(UseSite::Yield(region, i)), body.graph.names[value]);
            }
        }
        for (op, operation) in &body.graph.ir.operations {
            for (i, value) in operation.inputs.iter().enumerate() {
                body.uses.add(Site::Operand(UseSite::Input(op, i)), body.graph.names[value]);
            }
            for edge in &operation.bodies {
                body.parents.insert(edge.region, op);
                for (i, value) in edge.arguments.iter().enumerate() {
                    body.uses.add(
                        Site::Operand(UseSite::Argument(op, edge.role, i)),
                        body.graph.names[value],
                    );
                }
            }
        }
        query.for_each("OperationAttributeUse", |row| {
            if let Some(&op) = body.graph.operation_ids.get(&row[0]) {
                body.uses.add(Site::Attribute(op), row[1]);
            }
            Ok(())
        })?;
        for (&operation, names) in &query.0.structure().read().certificates.operations {
            if let Some(&op) = body.graph.operation_ids.get(&operation) {
                for &value in names {
                    body.uses.add(Site::Certificate(op), value_name(query, value)?);
                }
            }
        }
        query.for_each("RegionAttributeUse", |row| {
            if let Some(&region) = body.regions.get(&row[0]) {
                body.uses.add(Site::Region(region), row[1]);
            }
            Ok(())
        })?;
        for table in ["OperationRead", "OperationWrite", "OperationTrap"] {
            query.for_each(table, |row| {
                if let Some(&op) = body.graph.operation_ids.get(&row[0]) {
                    if !body.effects.contains_key(op) {
                        body.effects.insert(op, Effects::default());
                    }
                    let effect = &mut body.effects[op];
                    match table {
                        "OperationRead" => {
                            effect.reads.insert(row[1]);
                        }
                        "OperationWrite" => {
                            effect.writes.insert(row[1]);
                        }
                        _ => effect.traps = true,
                    }
                }
                Ok(())
            })?;
        }

        for (op, _) in &body.domains {
            let key = body.graph.operations[op];
            let mut results = Results::default();
            for slot in 0..query.0.result_count(key) {
                let id = (op, slot);
                let kind = query.0.result_kind(key, slot)?;
                let values = query.0.collective_inputs(key, slot, CollectiveInput::Contribution);
                for &name in &values {
                    body.uses.add(Site::Result(id), name);
                }
                let predicate = query
                    .0
                    .collective_input_at(key, slot, CollectiveInput::Predicate, 0)
                    .map(|value| value_name(query, value))
                    .transpose()?;
                if let Some(predicate) = predicate {
                    body.uses.add(Site::Result(id), predicate);
                }
                let neutral = query.0.collective_inputs(key, slot, CollectiveInput::Neutral);
                for &name in &neutral {
                    body.uses.add(Site::Neutral(id), name);
                }
                let (kind, _) =
                    query.application(kind, &["Collected", "Compacted", "Folded", "Contributed"])?;
                let kind = match kind {
                    "Collected" => ClauseKind::Collected,
                    "Compacted" => ClauseKind::Compacted,
                    "Folded" => ClauseKind::Folded,
                    _ => ClauseKind::Contributed,
                };
                body.clauses.insert(
                    id,
                    Clause {
                        owner: op,
                        kind,
                        values,
                        neutral,
                        predicate,
                    },
                );
                results.clauses.push(id);
            }
            for slot in 0..query.0.output_count(key) {
                let clause = query.0.output_clause(key, slot)?;
                let field = query.0.output_field(key, slot)? as usize;
                let role = query.0.output_role(key, slot)?;
                results.outputs.push(Output {
                    original: (op, slot),
                    clause: (op, clause),
                    field,
                    prefix: query.enode("PrefixOutput", role)?.is_some(),
                });
            }
            body.results.insert(op, results);
        }
        Ok(body)
    }

    pub fn lane(&self, operation: OperationId) -> Result<RegionId, OptimizeError> {
        self.control_region(operation, 0)
    }

    pub fn control_region(&self, operation: OperationId, slot: usize) -> Result<RegionId, OptimizeError> {
        let Some(edge) = self.graph.ir.operations[operation]
            .bodies
            .iter()
            .find(|edge| edge.role == BodyRole::Control(slot))
        else {
            return Err(output_error("operation has no requested control region"));
        };
        Ok(edge.region)
    }

    pub fn is_code(&self, op: OperationId, code: &Code) -> bool {
        matches!(&self.graph.ir.operations[op].definition, Definition::Other(value) if value == code)
    }

    pub fn inside(&self, mut region: RegionId, scope: RegionId) -> bool {
        loop {
            if region == scope {
                return true;
            }
            let Some(owner) = self.parents.get(region) else {
                return false;
            };
            region = self.positions[*owner].0;
        }
    }

    pub fn subregions(&self, op: OperationId) -> Vec<RegionId> {
        self.graph.ir.operations[op].bodies.iter().map(|edge| edge.region).collect()
    }

    pub fn parent(&self, region: RegionId) -> Option<OperationId> {
        self.parents.get(region).copied()
    }

    pub fn users(&self, name: Value) -> impl Iterator<Item = Site> + '_ {
        self.uses.sites.get(&name).into_iter().flatten().copied()
    }

    pub fn site_operation(&self, site: Site) -> Option<OperationId> {
        match site {
            Site::Operand(UseSite::Input(op, _) | UseSite::Argument(op, _, _))
            | Site::Attribute(op)
            | Site::Certificate(op) => Some(op),
            Site::Neutral(id) => Some(self.clause(id).owner),
            Site::Operand(UseSite::Yield(..)) | Site::Region(_) | Site::Result(_) => None,
        }
    }

    pub fn demand_region(&self, site: Site) -> Result<RegionId, OptimizeError> {
        match site {
            Site::Operand(UseSite::Yield(region, _)) | Site::Region(region) => Ok(region),
            Site::Result(id) => self.lane(self.clause(id).owner),
            _ => {
                let Some(op) = self.site_operation(site) else {
                    return Err(output_error("use has no execution scope"));
                };
                Ok(self.positions[op].0)
            }
        }
    }

    pub fn captures_available_before(&self, operation: OperationId, before: OperationId) -> bool {
        let children = self.subregions(operation);
        self.tree_uses(operation).into_iter().all(|(_, name)| {
            let Some(&value) = self.graph.refs.get(&name) else {
                return false;
            };
            matches!(value, ValueRef::Result(producer, _) if producer == operation)
                || children.iter().any(|&region| self.inside(self.definition_region(value), region))
                || self.available_before(value, before)
        })
    }

    pub fn definition_region(&self, value: ValueRef) -> RegionId {
        match value {
            ValueRef::Parameter(region, _) => region,
            ValueRef::Result(producer, _) => self.positions[producer].0,
        }
    }

    pub fn available_before(&self, value: ValueRef, mut before: OperationId) -> bool {
        let definition = self.definition_region(value);
        loop {
            let (scope, position) = self.positions[before];
            if scope == definition {
                return match value {
                    ValueRef::Parameter(..) => true,
                    ValueRef::Result(producer, _) => self.positions[producer].1 < position,
                };
            }
            let Some(owner) = self.parent(scope) else {
                return false;
            };
            before = owner;
        }
    }

    pub fn has_parallel_in(&self, region: RegionId) -> bool {
        let mut pending = vec![region];
        while let Some(region) = pending.pop() {
            for &op in &self.graph.ir.regions[region].operations {
                if self.domains.contains_key(op) {
                    return true;
                }
                pending.extend(self.subregions(op));
            }
        }
        false
    }

    pub fn clauses_of(&self, op: OperationId) -> impl Iterator<Item = ClauseId> + '_ {
        self.results
            .get(op)
            .into_iter()
            .flat_map(|r| r.clauses.iter().copied())
            .filter(|id| !self.pruned.contains(id))
    }

    pub fn folded(&self, op: OperationId) -> bool {
        self.clauses_of(op).any(|id| self.clauses[&id].kind == ClauseKind::Folded)
    }

    pub fn contributes(&self, op: OperationId) -> bool {
        let mut clauses = self.clauses_of(op);
        clauses.next().is_some_and(|id| self.clauses[&id].kind == ClauseKind::Contributed)
            && clauses.next().is_none()
    }

    pub fn forwarded_value(&mut self, op: OperationId, slot: usize) -> Option<Value> {
        let (id, value) = self.output_value(op, slot)?;
        (self.clause(id).kind == ClauseKind::Collected).then(|| self.resolve(value))
    }

    pub fn output_value(&self, op: OperationId, slot: usize) -> Option<(ClauseId, Value)> {
        let output = self.results.get(op)?.outputs.get(slot)?;
        if self.pruned.contains(&output.clause) {
            return None;
        }
        Some((
            output.clause,
            *self.clause(output.clause).values.get(output.field)?,
        ))
    }

    pub fn clause(&self, id: ClauseId) -> &Clause {
        &self.clauses[&id]
    }

    pub fn reducer(&self, operation: OperationId, id: ClauseId) -> Option<RegionId> {
        let slot = self.results[operation].clauses.iter().position(|&clause| clause == id)?;
        self.graph.ir.operations[operation]
            .bodies
            .iter()
            .find_map(|edge| (edge.role == BodyRole::Reducer(slot)).then_some(edge.region))
    }

    pub fn resolve(&mut self, value: Value) -> Value {
        self.substitutions.resolve(value)
    }

    fn op_sites(&self, op: OperationId) -> Vec<Site> {
        let operation = &self.graph.ir.operations[op];
        let mut sites = vec![Site::Attribute(op), Site::Certificate(op)];
        sites.extend((0..operation.inputs.len()).map(|i| Site::Operand(UseSite::Input(op, i))));
        for edge in &operation.bodies {
            sites.extend(
                (0..edge.arguments.len()).map(|i| Site::Operand(UseSite::Argument(op, edge.role, i))),
            );
        }
        for clause in self.clauses_of(op) {
            sites.extend([Site::Result(clause), Site::Neutral(clause)]);
        }
        sites
    }

    fn region_sites(&self, region: RegionId) -> impl Iterator<Item = Site> + '_ {
        let count = self.graph.ir.regions[region].yields.len();
        (0..count).map(move |i| Site::Operand(UseSite::Yield(region, i))).chain([Site::Region(region)])
    }

    pub fn tree_uses(&self, op: OperationId) -> Vec<(Site, Value)> {
        let mut sites = self.op_sites(op);
        let mut pending = self.subregions(op);
        while let Some(region) = pending.pop() {
            sites.extend(self.region_sites(region));
            for &op in &self.graph.ir.regions[region].operations {
                sites.extend(self.op_sites(op));
                pending.extend(self.subregions(op));
            }
        }
        sites.into_iter().flat_map(|site| self.uses.at(site)).collect()
    }

    fn dirty(&mut self, mut region: RegionId) {
        loop {
            self.dirty.insert(region);
            let Some(&owner) = self.parents.get(region) else {
                break;
            };
            region = self.positions[owner].0;
        }
    }

    pub fn take_dirty(&mut self) -> SortedSet<RegionId> {
        take(&mut self.dirty)
    }

    pub fn move_within(&mut self, region: RegionId, before: usize, index: usize) {
        let operations = &mut self.graph.ir.regions[region].operations[before..=index];
        let positions = operations.iter().map(|&op| self.positions[op]).collect::<Vec<_>>();
        operations.rotate_right(1);
        for (&op, position) in operations.iter().zip(positions) {
            self.positions[op] = position;
        }
        self.moved.insert(region);
        self.dirty(region);
    }

    pub fn lift(&mut self, operation: OperationId, owner: OperationId) {
        let (source, index) = self.positions[operation];
        let (destination, before) = self.positions[owner];
        self.graph.ir.regions[source].operations.remove(index);
        self.graph.ir.regions[destination].operations.insert(before, operation);
        self.relocated.insert(operation);
        for (region, start) in [(source, index), (destination, before)] {
            for (index, &op) in self.graph.ir.regions[region].operations.iter().enumerate().skip(start) {
                self.positions[op] = (region, index);
            }
            self.moved.insert(region);
            self.dirty(region);
        }
        // These owners derive their reads entirely from their children. Their
        // enclosing region still observes the same reads after the hoist.
        if matches!(
            self.graph.ir.operations[owner].definition,
            Definition::Other(
                Code::Parallel
                    | Code::Branch
                    | Code::CountedLoop
                    | Code::WhileLoop
                    | Code::BucketScatter { .. }
            )
        ) && self.effects.get(operation).is_some_and(|effect| !effect.reads.is_empty())
        {
            let reads = self
                .subregions(owner)
                .iter()
                .flat_map(|&region| &self.graph.ir.regions[region].operations)
                .filter_map(|&op| self.effects.get(op))
                .flat_map(|effect| effect.reads.iter().copied())
                .collect();
            let previous = &mut self.effects[owner].reads;
            if *previous != reads {
                let old = take(previous);
                if !self.reads_before_lift.contains_key(owner) {
                    self.reads_before_lift.insert(owner, old);
                }
                *previous = reads;
            }
        }
    }

    fn forget(&mut self, site: Site) {
        self.dead.extend(self.uses.remove(site));
        if let Site::Operand(site) = site {
            self.operand_changes.remove(&site);
        }
    }

    fn drop_op(&mut self, op: OperationId) {
        for site in self.op_sites(op) {
            self.forget(site);
        }
        let region = self.positions[op].0;
        self.graph.ir.regions[region].operations.retain(|&other| op != other);
        // Original occurrence keys still address result and output rows at publication.
        // Keep their slots reserved while new guard operations can be allocated.
        self.removed.insert(op);
        self.dirty(region);
    }

    fn bind_operand(&mut self, site: UseSite, name: Value) {
        let value = self.graph.refs[&name];
        let region = match site {
            UseSite::Input(op, index) => {
                self.graph.ir.operations[op].inputs[index] = value;
                self.positions[op].0
            }
            UseSite::Argument(op, role, index) => {
                for edge in &mut self.graph.ir.operations[op].bodies {
                    if edge.role == role {
                        edge.arguments[index] = value;
                    }
                }
                self.positions[op].0
            }
            UseSite::Yield(region, index) => {
                self.graph.ir.regions[region].yields[index] = value;
                region
            }
        };
        self.dirty(region);
    }

    fn redirect(&mut self, old: Value, new: Value) -> Result<(), OptimizeError> {
        let new = self.substitutions.resolve(new);
        if old == new {
            return Ok(());
        }
        self.substitutions.replace(old, new).map_err(|e| output_error(e.to_string()))?;
        for site in self.uses.sites.remove(&old).into_iter().flatten() {
            if let Some(values) = self.uses.values.get_mut(&site) {
                values.remove(&old);
            }
            self.uses.add(site, new);
            if let Site::Operand(site) = site {
                self.bind_operand(site, new);
                self.operand_changes.insert(site, new);
            }
        }
        Ok(())
    }

    fn move_value(&mut self, old: ValueRef, new: ValueRef) {
        let name = self.graph.names[&old];
        self.graph.names.insert(new, name);
        self.graph.refs.insert(name, new);
        let sites = self.uses.sites.get(&name).cloned().unwrap_or_default();
        for site in sites {
            if let Site::Operand(site) = site {
                self.bind_operand(site, name);
            }
        }
    }

    fn append_results(&mut self, first: OperationId, second: OperationId) {
        let offset = self.graph.ir.operations[first].result_types.len();
        let result_types = take(&mut self.graph.ir.operations[second].result_types);
        for i in 0..result_types.len() {
            self.move_value(ValueRef::Result(second, i), ValueRef::Result(first, offset + i));
        }
        self.graph.ir.operations[first].result_types.extend(result_types);
        let copied = self.uses.at(Site::Certificate(second)).map(|(_, name)| name).collect::<Vec<_>>();
        for name in copied {
            self.uses.add(Site::Certificate(first), name);
        }
    }

    fn append_operations(&mut self, destination: RegionId, source: RegionId) {
        let start =
            self.graph.ir.regions[destination].operations.last().map_or(0, |op| self.positions[*op].1 + 1);
        let moved = take(&mut self.graph.ir.regions[source].operations);
        for (index, &op) in moved.iter().enumerate() {
            self.positions[op] = (destination, start + index);
            self.relocated.insert(op);
        }
        self.graph.ir.regions[destination].operations.extend(moved);
        self.moved.insert(destination);
        self.dirty(destination);
    }

    fn absorb_effects(&mut self, first: OperationId, second: OperationId) {
        let b = self.effects.remove(second).unwrap_or_default();
        let mut a = self.effects.remove(first).unwrap_or_default();
        a.reads.extend(b.reads);
        a.writes.extend(b.writes);
        a.traps |= b.traps;
        self.effects.insert(first, a);
        if self.barriers.contains(&second) {
            self.barriers.insert(first);
        }
        if self.pinned.remove(&second) {
            self.pinned.insert(first);
        }
    }

    fn prune(&mut self) {
        while let Some(name) = self.dead.pop() {
            if !self.uses.dead(name) {
                continue;
            }
            let Some(&ValueRef::Result(op, slot)) = self.graph.refs.get(&name) else {
                continue;
            };
            if self.removed.contains(&op) {
                continue;
            }
            let Some(operation) = self.graph.ir.operations.get(op) else {
                continue;
            };
            if let Some(result) = self.results.get(op) {
                let clause = result.outputs[slot].clause;
                if self.clauses[&clause].can_prune()
                    && result
                        .outputs
                        .iter()
                        .enumerate()
                        .filter(|(_, output)| output.clause == clause)
                        .all(|(i, _)| self.uses.dead(self.graph.names[&ValueRef::Result(op, i)]))
                    && self.pruned.insert(clause)
                {
                    self.forget(Site::Result(clause));
                    self.forget(Site::Neutral(clause));
                    self.edited.insert(op);
                    self.dirty(self.positions[op].0);
                }
            } else if operation.bodies.is_empty()
                && !self.barriers.contains(&op)
                && (0..operation.result_types.len())
                    .all(|i| self.uses.dead(self.graph.names[&ValueRef::Result(op, i)]))
            {
                self.drop_op(op);
            }
        }
    }

    pub fn simplify(&mut self, query: Query<'_>) -> Result<(), OptimizeError> {
        let mut regions = self.moved.clone();
        for &site in self.operand_changes.keys() {
            regions.insert(self.demand_region(Site::Operand(site))?);
        }
        let mut reuse = cse::Reuse::default();
        for &region in &regions {
            let mut prefixes = Vec::new();
            let mut child = region;
            while let Some(owner) = self.parent(child) {
                let (ancestor, index) = self.positions[owner];
                if self.removed.contains(&owner) || regions.contains(&ancestor) {
                    break;
                }
                let end = self.graph.ir.regions[ancestor]
                    .operations
                    .partition_point(|&op| self.positions[op].1 < index);
                prefixes.push((ancestor, end));
                child = ancestor;
            }
            if self.parent(child).is_none() {
                prefixes.reverse();
                reuse.scope(query, &self.graph, region, &prefixes)?;
            }
        }
        for (old, new) in reuse.replacements {
            if let ValueRef::Result(op, _) = old {
                self.redirect(self.graph.names[&old], self.graph.names[&new])?;
                self.drop_op(op);
            }
        }
        self.prune();
        Ok(())
    }

    pub fn splice(
        &mut self,
        query: Query<'_>,
        first: OperationId,
        second: OperationId,
        loads: &[(OperationId, Value)],
    ) -> Result<(), OptimizeError> {
        let destination = self.lane(first)?;
        let source = self.lane(second)?;
        let width = self.graph.ir.regions[destination].parameters.len();
        if width != self.graph.ir.regions[source].parameters.len() {
            return Err(output_error("fused lane parameter counts differ"));
        }
        for i in 0..width {
            self.redirect(
                self.graph.names[&ValueRef::Parameter(source, i)],
                self.graph.names[&ValueRef::Parameter(destination, i)],
            )?;
        }
        // Captured scalar values keep their original meaning. A forwarded view may
        // require a fresh alias proof before another write can move across it.
        for &(load, value) in loads {
            let name = self.graph.names[&ValueRef::Result(load, 0)];
            if query.lookup("ValuePrimitive", name)?.is_none() {
                self.unknown_aliases = true;
            }
            self.redirect(name, value)?;
            self.drop_op(load);
        }
        self.append_results(first, second);
        let clause_offset = self.results[first].clauses.len();
        let results = take(&mut self.results[second]);
        for id in &results.clauses {
            if let Some(clause) = self.clauses.get_mut(id) {
                clause.owner = first;
            }
        }
        self.results[first].clauses.extend(results.clauses);
        self.results[first].outputs.extend(results.outputs);
        // Only reducer bodies remain attached to their clauses after the lane splice.
        let reducers = self.graph.ir.operations[second]
            .bodies
            .iter()
            .filter_map(|edge| {
                let BodyRole::Reducer(index) = edge.role else {
                    return None;
                };
                let mut edge = edge.clone();
                edge.role = BodyRole::Reducer(clause_offset + index);
                Some(edge)
            })
            .collect::<Vec<_>>();
        for edge in &reducers {
            self.parents.insert(edge.region, first);
        }
        self.graph.ir.operations[first].bodies.extend(reducers);
        self.append_operations(destination, source);
        let dropped = self.region_sites(source).collect::<Vec<_>>();
        for site in dropped {
            self.forget(site);
        }
        self.absorb_effects(first, second);
        // Its result sites now belong to first, even though their original keys
        // are retained for publication.
        self.drop_op(second);
        self.domains.remove(second);
        self.edited.insert(first);
        self.dirty(destination);
        self.prune();
        Ok(())
    }

    pub fn publish(self, graph: &mut Graph) -> Result<(), OptimizeError> {
        let _timing = timing::span("Rust parallel fusion publication");
        self.publish_joins(graph)?;
        self.publish_accumulators(graph)?;
        // Motion and accumulator-only batches do not rewrite parallel clauses.
        if self.edited.is_empty() {
            graph.update(|mut sink| self.publish_order(&mut sink))?;
            if self.removed.is_empty() {
                return edit::changed(graph);
            }
            return edit::remove(
                graph,
                &self.removed.into_iter().map(|op| self.graph.operations[op]).collect::<Vec<_>>(),
            );
        }
        self.publish_guards(graph)?;
        let query = Query(graph);
        let mut changed = Vec::new();
        let mut edges = Vec::new();
        for &op in &self.edited {
            if self.removed.contains(&op) {
                continue;
            }
            let key = self.graph.operations[op];
            let mut results = ParallelResults::default();
            let mut clauses = LookupMap::new();
            let mut reducers = Vec::new();
            for id in self.clauses_of(op) {
                let original = self.graph.operations[id.0];
                let clause = results.kinds.len() as i64;
                clauses.insert(id, clause);
                results.kinds.push(Some(graph.result_kind(original, id.1)?));
                if let Some(region) = graph.reducer_at(original, id.1) {
                    reducers.push((clause, region));
                }
                for kind in [
                    CollectiveInput::Contribution,
                    CollectiveInput::Neutral,
                    CollectiveInput::Predicate,
                ] {
                    let values = graph.collective_inputs(original, id.1, kind);
                    if !values.is_empty() {
                        edges.push((key, clause, kind, values));
                    }
                }
            }
            let mut names = Vec::new();
            for output in &self.results[op].outputs {
                let Some(&clause) = clauses.get(&output.clause) else {
                    continue;
                };
                let original = self.graph.operations[output.original.0];
                let slot = output.original.1;
                let mut binding = graph.output_binding(original, slot)?;
                binding.clause = Some(clause);
                results.bindings.push(binding);
                names.push(graph.output(original, slot)?);
            }
            changed.push((key, results, names, reducers));
        }
        let certificates = self
            .edited
            .iter()
            .flat_map(|&op| self.uses.at(Site::Certificate(op)).map(move |(_, name)| (op, name)))
            .map(|(op, name)| Ok((self.graph.operations[op], query.name(name)?)))
            .collect::<Result<Vec<_>, OptimizeError>>()?;
        graph.update(|mut sink| {
            for (op, results, names, reducers) in changed {
                sink.clear_collective_inputs(op);
                sink.set_reducers(op, reducers);
                sink.set_parallel_results(op, results);
                sink.set_outputs(op, names)?;
            }
            for (op, group, kind, values) in edges {
                sink.set_collective_inputs(op, group, kind, values)?;
            }
            self.publish_order(&mut sink)?;
            for (operation, certificate) in certificates {
                sink.add_certificate(CertificateSite::Operation(operation), certificate);
            }
            Ok(())
        })?;
        let changes =
            self.operand_changes.into_iter().map(|(site, name)| (site, self.graph.refs[&name])).collect();
        self.graph.publish_edges(graph, changes)?;
        let replacements = self.substitutions.into_map().into_iter().collect::<Vec<_>>();
        if !replacements.is_empty() {
            edit::replace_attributes(graph, &replacements)?;
        }
        edit::remove(
            graph,
            &self.removed.into_iter().map(|op| self.graph.operations[op]).collect::<Vec<_>>(),
        )
    }

    /// Same-region motion preserves definitions, actual uses, ownership and effects.
    pub fn publish_sibling_order(self, graph: &mut Graph) -> Result<(), OptimizeError> {
        let _timing = timing::span("Rust sibling order publication");
        let (previous, updated) = self.orders(Query(graph))?;
        graph.update(|mut sink| self.publish_order(&mut sink))?;
        analysis::replace_order(graph, previous, updated)?;
        edit::preserve_revisions(graph)
    }

    fn orders(&self, query: Query<'_>) -> Result<(RegionOrders, RegionOrders), OptimizeError> {
        let mut previous = Vec::new();
        let mut updated = Vec::new();
        for &region in &self.moved {
            let key = self.graph.regions[region];
            let old = query.0.region_order(key)?.to_vec();
            let order = self.graph.ir.regions[region]
                .operations
                .iter()
                .map(|&op| self.graph.operations[op])
                .collect::<Vec<_>>();
            previous.push((key, old));
            updated.push((key, order));
        }
        Ok((previous, updated))
    }

    fn publish_order(&self, sink: &mut State<'_, '_, '_>) -> Result<(), egglog_engine::Error> {
        for &region in &self.moved {
            let Some(body) = self.graph.ir.regions.get(region) else {
                continue;
            };
            for (order, &op) in body.operations.iter().enumerate() {
                let key = self.graph.operations[op];
                if self.relocated.contains(&op) {
                    sink.set_region(key, self.graph.regions[region]);
                }
                sink.set_order(key, order as i64);
            }
        }
        Ok(())
    }
}
