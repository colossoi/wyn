//! Structural state owned by a parallel fusion pass until it publishes its edits.
use super::Domain;
use crate::egglog::analysis::count;
use crate::egglog::dataflow::{body::BodyGraph, edit, ir::Code};
use crate::egglog::query::Query;
use crate::egglog::{output_error, timing, OptimizeError};
use crate::memory_ir::{BodyRole, Definition, OperationId, RegionId, UseSite, ValueRef};
use crate::{LookupMap, LookupSet, SortedSet};
use egglog_engine::{EGraph, RawValues, Value, Write};
use slotmap::SecondaryMap;
use std::mem::take;
use wyn_graph::ReplacementForest;

fn value_name(query: Query<'_>, value: Value) -> Result<Value, OptimizeError> {
    let Some(name) = query.constructor("ValueNameRef", value)? else {
        return Err(output_error("missing value reference in fusion metadata"));
    };
    Ok(name)
}

type ClauseId = (OperationId, i64);

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

struct Clause {
    can_prune: bool,
    folded: bool,
    contributes: bool,
}

struct Output {
    original: (OperationId, i64),
    clause: ClauseId,
    forwarded: Option<Value>,
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
    removed: SortedSet<OperationId>,
    pruned: LookupSet<ClauseId>,
    dirty: SortedSet<RegionId>,
    dead: Vec<Value>,
}

impl Body {
    pub fn read(
        query: Query<'_>,
        new_regions: &[(Value, Vec<Value>)],
        domains: LookupMap<Value, Domain>,
        barriers: LookupSet<Value>,
    ) -> Result<Self, OptimizeError> {
        let _timing = timing::span("Rust parallel fusion graph import");
        let graph = BodyGraph::read(query, new_regions)?;
        let mut body = Self {
            regions: graph.regions.iter().map(|(key, &region)| (region, key)).collect(),
            domains: domains
                .into_iter()
                .filter_map(|(op, domain)| Some((*graph.operation_ids.get(&op)?, domain)))
                .collect(),
            barriers: barriers.into_iter().filter_map(|op| graph.operation_ids.get(&op).copied()).collect(),
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
            removed: SortedSet::default(),
            pruned: LookupSet::new(),
            dirty: SortedSet::default(),
            dead: Vec::new(),
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
        for (table, kind) in [
            (
                "OperationAttributeUse",
                Site::Attribute as fn(OperationId) -> Site,
            ),
            ("OperationCertificate", Site::Certificate),
        ] {
            query.for_each(table, |row| {
                if let Some(&op) = body.graph.operation_ids.get(&row[0]) {
                    let name =
                        if table == "OperationCertificate" { value_name(query, row[1])? } else { row[1] };
                    body.uses.add(kind(op), name);
                }
                Ok(())
            })?;
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
            let mut fields = LookupMap::new();
            for slot in 0..count(query, "ResultCount", key)? {
                let id = (op, slot);
                let kind = query.required("ResultKindOf", (key, slot))?;
                let values = query.indexed(
                    "ResultValue",
                    &[key, query.0.base_to_value(slot)],
                    count(query, "ResultValueCount", (key, slot))?,
                )?;
                for (index, &name) in values.iter().enumerate() {
                    fields.insert((slot, index as i64), name);
                    body.uses.add(Site::Result(id), name);
                }
                if let Some(predicate) = query.lookup("ResultPredicate", (key, slot))? {
                    body.uses.add(Site::Result(id), value_name(query, predicate)?);
                }
                if let Some(n) = query.lookup("ResultNeutralCount", (key, slot))? {
                    for name in query.indexed(
                        "ResultNeutral",
                        &[key, query.0.base_to_value(slot)],
                        query.0.value_to_base(n),
                    )? {
                        body.uses.add(Site::Neutral(id), name);
                    }
                }
                let can_prune = values.len() == 1
                    && (query.enode("Collected", kind)?.is_some()
                        || query.enode("Compacted", kind)?.is_some());
                body.clauses.insert(
                    id,
                    Clause {
                        can_prune,
                        folded: query.enode("Folded", kind)?.is_some(),
                        contributes: query.enode("Contributed", kind)?.is_some(),
                    },
                );
                results.clauses.push(id);
            }
            for slot in 0..count(query, "OutputCount", key)? {
                let clause = count(query, "OutputClause", (key, slot))?;
                let field = count(query, "OutputField", (key, slot))?;
                let role = query.required("OutputRole", (key, slot))?;
                let forwarded = if query.enode("CollectedOutput", role)?.is_some() {
                    fields.get(&(clause, field)).copied()
                } else {
                    None
                };
                results.outputs.push(Output {
                    original: (op, slot),
                    clause: (op, clause),
                    forwarded,
                });
            }
            body.results.insert(op, results);
        }
        Ok(body)
    }

    pub fn lane(&self, operation: OperationId) -> Result<RegionId, OptimizeError> {
        let Some(edge) = self.graph.ir.operations[operation]
            .bodies
            .iter()
            .find(|edge| edge.role == BodyRole::Control(0))
        else {
            return Err(output_error("parallel operation has no lane body"));
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

    fn clauses_of(&self, op: OperationId) -> impl Iterator<Item = ClauseId> + '_ {
        self.results
            .get(op)
            .into_iter()
            .flat_map(|r| r.clauses.iter().copied())
            .filter(|id| !self.pruned.contains(id))
    }

    pub fn folded(&self, op: OperationId) -> bool {
        self.clauses_of(op).any(|id| self.clauses[&id].folded)
    }

    pub fn contributes(&self, op: OperationId) -> bool {
        let mut clauses = self.clauses_of(op);
        clauses.next().is_some_and(|id| self.clauses[&id].contributes) && clauses.next().is_none()
    }

    pub fn forwarded_value(&mut self, op: OperationId, slot: usize) -> Option<Value> {
        let output = &self.results[op].outputs[slot];
        let value = output.forwarded?;
        (!self.pruned.contains(&output.clause)).then(|| self.substitutions.resolve(value))
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
        self.graph.ir.operations.remove(op);
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

    fn move_output(&mut self, old: ValueRef, new: ValueRef) {
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

    fn prune(&mut self) {
        while let Some(name) = self.dead.pop() {
            if !self.uses.dead(name) {
                continue;
            }
            let Some(&ValueRef::Result(op, slot)) = self.graph.refs.get(&name) else {
                continue;
            };
            let Some(operation) = self.graph.ir.operations.get(op) else {
                continue;
            };
            if let Some(result) = self.results.get(op) {
                let clause = result.outputs[slot].clause;
                if self.clauses[&clause].can_prune
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
        let offset = self.graph.ir.operations[first].result_types.len();
        let result_types = take(&mut self.graph.ir.operations[second].result_types);
        for i in 0..result_types.len() {
            self.move_output(ValueRef::Result(second, i), ValueRef::Result(first, offset + i));
        }
        self.graph.ir.operations[first].result_types.extend(result_types);
        let clause_offset = self.results[first].clauses.len();
        let results = take(&mut self.results[second]);
        self.results[first].clauses.extend(results.clauses);
        self.results[first].outputs.extend(results.outputs);
        let certificates =
            self.uses.at(Site::Certificate(second)).map(|(_, name)| name).collect::<Vec<_>>();
        for name in certificates {
            self.uses.add(Site::Certificate(first), name);
        }
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
        let start =
            self.graph.ir.regions[destination].operations.last().map_or(0, |op| self.positions[*op].1 + 1);
        let moved = take(&mut self.graph.ir.regions[source].operations);
        for (index, &op) in moved.iter().enumerate() {
            self.positions[op] = (destination, start + index);
        }
        self.graph.ir.regions[destination].operations.extend(moved);
        self.moved.insert(destination);
        let dropped = self.region_sites(source).collect::<Vec<_>>();
        for site in dropped {
            self.forget(site);
        }
        self.graph.ir.regions.remove(source);
        let b = self.effects.remove(second).unwrap_or_default();
        let mut a = self.effects.remove(first).unwrap_or_default();
        a.reads.extend(b.reads);
        a.writes.extend(b.writes);
        a.traps |= b.traps;
        self.effects.insert(first, a);
        if self.barriers.contains(&second) {
            self.barriers.insert(first);
        }
        self.drop_op(second);
        self.domains.remove(second);
        self.edited.insert(first);
        self.dirty(destination);
        self.prune();
        Ok(())
    }

    pub fn publish(self, graph: &mut EGraph) -> Result<(), OptimizeError> {
        let _timing = timing::span("Rust parallel fusion publication");
        // Pure motion changes neither result clauses nor operands.
        if self.edited.is_empty() && self.removed.is_empty() {
            graph.update(|mut sink| {
                for region in self.moved {
                    for (order, &op) in self.graph.ir.regions[region].operations.iter().enumerate() {
                        let op = self.graph.operations[op];
                        sink.remove("OperationOrder", op)?;
                        sink.set("OperationOrder", op, order as i64)?;
                    }
                }
                Ok(())
            })?;
            return edit::changed(graph);
        }
        let query = Query(graph);
        let mut clauses = LookupMap::new();
        let mut outputs = LookupMap::new();
        let mut counts = Vec::new();
        for &op in &self.edited {
            if self.removed.contains(&op) {
                continue;
            }
            let key = self.graph.operations[op];
            let mut clause_count = 0_i64;
            for id in self.clauses_of(op) {
                clauses.insert((self.graph.operations[id.0], id.1), (key, clause_count));
                clause_count += 1;
            }
            let mut output_count = 0_i64;
            for output in &self.results[op].outputs {
                let Some(&(_, clause)) =
                    clauses.get(&(self.graph.operations[output.clause.0], output.clause.1))
                else {
                    continue;
                };
                outputs.insert(
                    (self.graph.operations[output.original.0], output.original.1),
                    (key, output_count, clause),
                );
                output_count += 1;
            }
            counts.push((key, clause_count, output_count));
        }
        let edited: LookupSet<_> = counts.iter().map(|&(op, _, _)| op).collect();
        let mut rows = Vec::new();
        let mut old = Vec::new();
        for &table in edit::OPERATION_CELLS {
            let result = table.starts_with("Result") && table != "ResultCount";
            let output = table.starts_with("Output") && table != "OutputCount";
            if !result && !output {
                continue;
            }
            query.for_function(table, |keys, value| {
                if edited.contains(&keys[0]) {
                    old.push((table, keys.to_vec()));
                }
                let original = (keys[0], graph.value_to_base::<i64>(keys[1]));
                let target = if result {
                    clauses.get(&original).map(|&(op, slot)| (op, slot, value))
                } else {
                    outputs.get(&original).map(|&(op, slot, clause)| {
                        let value =
                            if table == "OutputClause" { graph.base_to_value(clause) } else { value };
                        (op, slot, value)
                    })
                };
                if let Some((op, slot, value)) = target {
                    let mut keys = keys.to_vec();
                    keys[0] = op;
                    keys[1] = graph.base_to_value(slot);
                    rows.push((table, keys, value));
                }
                Ok(())
            })?;
        }
        let certificates = self
            .edited
            .iter()
            .flat_map(|&op| self.uses.at(Site::Certificate(op)).map(move |(_, name)| (op, name)))
            .map(|(op, name)| Ok((self.graph.operations[op], query.name(name)?)))
            .collect::<Result<Vec<_>, OptimizeError>>()?;
        graph.update(|mut sink| {
            for (table, keys) in old {
                sink.remove(table, RawValues(keys))?;
            }
            for (table, keys, value) in rows {
                sink.set(table, RawValues(keys), value)?;
            }
            for (op, clauses, outputs) in counts {
                for (table, count) in [("ResultCount", clauses), ("OutputCount", outputs)] {
                    sink.remove(table, op)?;
                    sink.set(table, op, count)?;
                }
            }
            for region in self.moved {
                let Some(body) = self.graph.ir.regions.get(region) else {
                    continue;
                };
                for (order, &op) in body.operations.iter().enumerate() {
                    let op = self.graph.operations[op];
                    sink.remove("OperationRegion", op)?;
                    sink.remove("OperationOrder", op)?;
                    sink.set("OperationRegion", op, self.graph.regions[region])?;
                    sink.set("OperationOrder", op, order as i64)?;
                }
            }
            for certificate in certificates {
                sink.add("OperationCertificate", certificate)?;
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
}
