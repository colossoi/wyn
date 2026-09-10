//! Owned, checked contraction of scoped operations and their value boundaries.
//!
//! Composition algebra belongs to the caller. Groups own caller-defined facts;
//! scope and resource keys use the caller's existing identity types.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::hash::Hash;
use thiserror::Error;
use wyn_base::{IdArena, IdSource, LookupMap, SortedSet, StableMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupId(u32);

impl From<u32> for GroupId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PortId(u32);

impl From<u32> for PortId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstraintId(u32);

impl From<u32> for ConstraintId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OrderingReason<R> {
    Effect,
    Resource(R),
    Opaque,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group<S, P> {
    scope: S,
    members: SortedSet<GroupId>,
    inputs: Vec<PortId>,
    outputs: Vec<PortId>,
    payload: P,
}

impl<S: Copy, P> Group<S, P> {
    pub fn scope(&self) -> S {
        self.scope
    }
    pub fn members(&self) -> &SortedSet<GroupId> {
        &self.members
    }
    pub fn inputs(&self) -> &[PortId] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[PortId] {
        &self.outputs
    }
    pub fn payload(&self) -> &P {
        &self.payload
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Value {
    Input,
    Result(GroupId),
    Pure(Vec<PortId>),
    Alias(PortId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint<R> {
    pub before: GroupId,
    pub after: GroupId,
    pub reason: OrderingReason<R>,
}

/// Computed from use incidences, never supplied by a composition rule.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Boundary {
    pub inputs: Vec<PortId>,
    pub outputs: Vec<PortId>,
    /// Source input occurrences whose value comes from inside this boundary.
    pub internal: Vec<PortId>,
    pub constraints: Vec<ConstraintId>,
}

/// Caller-checked composition, with the old ports represented by each result.
/// Empty result origins allow new, unobserved results; multiple origins alias
/// the same result. The graph derives and routes the required boundary.
#[derive(Clone, Debug)]
pub struct Proposal<P> {
    pub sources: Vec<GroupId>,
    pub absorbed_values: Vec<PortId>,
    pub results: Vec<Vec<PortId>>,
    pub accounted_constraints: Vec<ConstraintId>,
    pub payload: P,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Action {
    pub sources: Vec<GroupId>,
    pub target: GroupId,
    pub outputs: Vec<(PortId, PortId)>,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("contraction references an inactive group")]
    StaleGroup,
    #[error("contraction requires distinct source groups")]
    Membership,
    #[error("contraction crosses a scope boundary")]
    Scope,
    #[error("value port is absent or cannot be absorbed")]
    Port,
    #[error("composition does not route its complete boundary")]
    Routing,
    #[error("composition does not account for its internal ordering constraints")]
    Ordering,
    #[error("contraction creates a dependency cycle")]
    Cycle,
    #[error("every source group requires exactly one payload")]
    Payload,
}

#[derive(Clone, Debug)]
pub struct Builder<S, R> {
    graph: Graph<S, R, ()>,
    accesses: LookupMap<(S, R), Vec<(GroupId, bool)>>,
}

impl<S: Copy + Eq + Hash, R: Copy + Eq + Hash> Builder<S, R> {
    pub fn new() -> Self {
        Self {
            graph: Graph::default(),
            accesses: LookupMap::new(),
        }
    }

    pub fn input(&mut self) -> PortId {
        self.graph.alloc_value(Value::Input)
    }

    /// Pure expression incidences preserve paths through projections, indices,
    /// lengths, and other caller-owned expressions without storing payloads.
    pub fn value(&mut self, dependencies: Vec<PortId>) -> Result<PortId, Error> {
        self.graph.check_ports(&dependencies)?;
        Ok(self.graph.alloc_value(Value::Pure(dependencies)))
    }

    pub fn operation(&mut self, scope: S, inputs: Vec<PortId>, results: usize) -> Result<GroupId, Error> {
        self.graph.check_ports(&inputs)?;
        let id = self.graph.group_ids.next_id();
        let outputs = (0..results).map(|_| self.graph.alloc_value(Value::Result(id))).collect();
        self.graph.groups.insert(
            id,
            Group {
                scope,
                members: SortedSet::from([id]),
                inputs,
                outputs,
                payload: (),
            },
        );
        Ok(id)
    }

    pub fn outputs(&self, group: GroupId) -> Result<&[PortId], Error> {
        Ok(self.graph.group(group)?.outputs())
    }

    /// Supply forward value incidences after allocating all skeleton effects.
    pub fn set_inputs(&mut self, group: GroupId, inputs: Vec<PortId>) -> Result<(), Error> {
        self.graph.check_ports(&inputs)?;
        self.graph.groups.get_mut(&group).ok_or(Error::StaleGroup)?.inputs = inputs;
        Ok(())
    }

    pub fn observe(&mut self, value: PortId) -> Result<(), Error> {
        self.graph.check_ports(&[value])?;
        self.graph.observers.insert(value);
        Ok(())
    }

    pub fn order(
        &mut self,
        before: GroupId,
        after: GroupId,
        reason: OrderingReason<R>,
    ) -> Result<ConstraintId, Error> {
        if self.graph.group(before)?.scope != self.graph.group(after)?.scope {
            return Err(Error::Scope);
        }
        if let Some((id, _)) = self
            .graph
            .constraints
            .iter()
            .find(|(_, edge)| edge.before == before && edge.after == after && edge.reason == reason)
        {
            return Ok(*id);
        }
        let id = self.graph.constraint_ids.next_id();
        self.graph.constraints.insert(
            id,
            Constraint {
                before,
                after,
                reason,
            },
        );
        Ok(id)
    }

    /// Record a canonical resource access in source order. `writes` is true
    /// for writes and read-modify-writes; read/read pairs remain independent.
    pub fn access(&mut self, group: GroupId, resource: R, writes: bool) -> Result<(), Error> {
        let scope = self.graph.group(group)?.scope;
        let accesses = self.accesses.entry((scope, resource)).or_default();
        let predecessors = accesses
            .iter()
            .filter_map(|(before, other_writes)| {
                (*before != group && (writes || *other_writes)).then_some(*before)
            })
            .collect::<Vec<_>>();
        if let Some((_, old)) = accesses.iter_mut().find(|(id, _)| *id == group) {
            *old |= writes;
        } else {
            accesses.push((group, writes));
        }
        for before in predecessors {
            self.order(before, group, OrderingReason::Resource(resource))?;
        }
        Ok(())
    }

    pub fn finish<P>(
        mut self,
        payloads: impl IntoIterator<Item = (GroupId, P)>,
    ) -> Result<Graph<S, R, P>, Error> {
        self.graph.order()?;
        let mut groups = BTreeMap::new();
        for (id, payload) in payloads {
            let source = self.graph.groups.remove(&id).ok_or(Error::Payload)?;
            let group = Group {
                scope: source.scope,
                members: source.members,
                inputs: source.inputs,
                outputs: source.outputs,
                payload,
            };
            groups.insert(id, group);
        }
        if !self.graph.groups.is_empty() {
            return Err(Error::Payload);
        }
        Ok(Graph {
            groups,
            group_ids: self.graph.group_ids,
            constraint_ids: self.graph.constraint_ids,
            values: self.graph.values,
            observers: self.graph.observers,
            constraints: self.graph.constraints,
            actions: self.graph.actions,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Graph<S, R, P> {
    group_ids: IdSource<GroupId>,
    constraint_ids: IdSource<ConstraintId>,
    groups: BTreeMap<GroupId, Group<S, P>>,
    values: IdArena<PortId, Value>,
    observers: SortedSet<PortId>,
    constraints: BTreeMap<ConstraintId, Constraint<R>>,
    actions: Vec<Action>,
}

impl<S, R, P> Default for Graph<S, R, P> {
    fn default() -> Self {
        Self {
            group_ids: IdSource::new(),
            constraint_ids: IdSource::new(),
            groups: BTreeMap::new(),
            values: IdArena::new(),
            observers: SortedSet::new(),
            constraints: BTreeMap::new(),
            actions: vec![],
        }
    }
}

impl<S: Copy + Eq + Hash, R: Copy + Eq + Hash> Default for Builder<S, R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Copy + Eq, R: Copy + Eq, P> Graph<S, R, P> {
    fn alloc_value(&mut self, value: Value) -> PortId {
        self.values.alloc(value)
    }

    fn check_ports(&self, ports: &[PortId]) -> Result<(), Error> {
        if ports.iter().all(|port| self.values.get(*port).is_some()) {
            Ok(())
        } else {
            Err(Error::Port)
        }
    }

    pub fn group(&self, group: GroupId) -> Result<&Group<S, P>, Error> {
        self.groups.get(&group).ok_or(Error::StaleGroup)
    }

    pub fn groups(&self) -> impl Iterator<Item = (GroupId, &Group<S, P>)> {
        self.groups.iter().map(|(id, group)| (*id, group))
    }

    pub fn constraint(&self, id: ConstraintId) -> Option<&Constraint<R>> {
        self.constraints.get(&id)
    }

    pub fn canonical(&self, mut port: PortId) -> Result<PortId, Error> {
        self.check_ports(&[port])?;
        while let Value::Alias(next) = self.values[port] {
            port = next;
        }
        Ok(port)
    }

    pub fn actions(&self) -> &[Action] {
        &self.actions
    }

    /// Producers reached through pure expressions, stopping at effect results.
    pub fn producers(&self, port: PortId) -> Result<SortedSet<GroupId>, Error> {
        self.check_ports(&[port])?;
        let mut producers = SortedSet::new();
        let mut visited = SortedSet::new();
        let mut pending = vec![port];
        while let Some(port) = pending.pop() {
            let port = self.canonical(port)?;
            if !visited.insert(port) {
                continue;
            }
            match &self.values[port] {
                Value::Result(group) => {
                    producers.insert(*group);
                }
                Value::Pure(inputs) => pending.extend(inputs),
                Value::Alias(input) => pending.push(*input),
                Value::Input => {}
            }
        }
        Ok(producers)
    }

    /// Whether a value-use incidence reaches a particular result boundary.
    pub fn depends_on(&self, value: PortId, target: PortId) -> Result<bool, Error> {
        self.check_ports(&[value, target])?;
        let target = self.canonical(target)?;
        let mut pending = vec![value];
        let mut visited = SortedSet::new();
        while let Some(port) = pending.pop() {
            let port = self.canonical(port)?;
            if !visited.insert(port) {
                continue;
            }
            if port == target {
                return Ok(true);
            }
            match &self.values[port] {
                Value::Pure(inputs) => pending.extend(inputs),
                Value::Alias(input) => pending.push(*input),
                _ => {}
            }
        }
        Ok(false)
    }

    /// Active groups consuming any selected result, through pure incidences.
    pub fn consumers(&self, ports: &[PortId]) -> Result<SortedSet<GroupId>, Error> {
        self.check_ports(ports)?;
        let mut consumers = SortedSet::new();
        for (id, group) in &self.groups {
            for input in &group.inputs {
                for port in ports {
                    if self.depends_on(*input, *port)? {
                        consumers.insert(*id);
                    }
                }
            }
        }
        Ok(consumers)
    }

    fn sources(&self, sources: &[GroupId]) -> Result<SortedSet<GroupId>, Error> {
        let selected = sources.iter().copied().collect::<SortedSet<_>>();
        if selected.is_empty() || selected.len() != sources.len() {
            return Err(Error::Membership);
        }
        let scope = self.group(sources[0])?.scope;
        for source in sources {
            if self.group(*source)?.scope != scope {
                return Err(Error::Scope);
            }
        }
        Ok(selected)
    }

    pub fn boundary(&self, sources: &[GroupId], absorbed_values: &[PortId]) -> Result<Boundary, Error> {
        let selected = self.sources(sources)?;
        let absorbed = absorbed_values.iter().copied().collect::<SortedSet<_>>();
        if absorbed.len() != absorbed_values.len() {
            return Err(Error::Port);
        }
        for port in &absorbed {
            if !matches!(self.values.get(*port), Some(Value::Pure(_)))
                || self.producers(*port)?.is_disjoint(&selected)
            {
                return Err(Error::Port);
            }
        }
        let mut outputs = SortedSet::new();
        let mut pending = self.observers.iter().copied().collect::<Vec<_>>();
        for (id, group) in &self.groups {
            if !selected.contains(id) {
                pending.extend(&group.inputs);
            }
        }
        let mut visited = SortedSet::new();
        while let Some(port) = pending.pop() {
            let port = self.canonical(port)?;
            if !visited.insert(port) {
                continue;
            }
            if absorbed.contains(&port) {
                outputs.insert(port);
                continue;
            }
            match &self.values[port] {
                Value::Result(group) if selected.contains(group) => {
                    outputs.insert(port);
                }
                Value::Pure(inputs) => pending.extend(inputs),
                Value::Alias(input) => pending.push(*input),
                _ => {}
            }
        }
        let mut inputs = SortedSet::new();
        let mut internal = SortedSet::new();
        let mut pending =
            sources.iter().flat_map(|id| self.groups[id].inputs.iter().copied()).collect::<Vec<_>>();
        for port in &absorbed {
            if let Value::Pure(dependencies) = &self.values[*port] {
                pending.extend(dependencies);
            }
        }
        let mut visited = SortedSet::new();
        while let Some(port) = pending.pop() {
            let port = self.canonical(port)?;
            if !visited.insert(port) {
                continue;
            }
            if absorbed.contains(&port) {
                internal.insert(port);
                continue;
            }
            if self.producers(port)?.is_disjoint(&selected) {
                inputs.insert(port);
                continue;
            }
            internal.insert(port);
            match &self.values[port] {
                Value::Pure(dependencies) => pending.extend(dependencies),
                Value::Alias(input) => pending.push(*input),
                _ => {}
            }
        }
        let constraints = self
            .constraints
            .iter()
            .filter_map(|(id, edge)| {
                (selected.contains(&edge.before) && selected.contains(&edge.after)).then_some(*id)
            })
            .collect();
        Ok(Boundary {
            inputs: inputs.into_iter().collect(),
            outputs: outputs.into_iter().collect(),
            internal: internal.into_iter().collect(),
            constraints,
        })
    }

    pub fn contract(&mut self, proposal: Proposal<P>) -> Result<Action, Error>
    where
        P: Clone,
    {
        if proposal.sources.len() < 2 {
            return Err(Error::Membership);
        }
        self.apply(proposal)
    }

    /// Unary rewrites share exactly the same boundary and ordering checks.
    /// The caller's recipe algebra must retire the matched unary form.
    pub fn rewrite(&mut self, proposal: Proposal<P>) -> Result<Action, Error>
    where
        P: Clone,
    {
        if proposal.sources.len() != 1 {
            return Err(Error::Membership);
        }
        self.apply(proposal)
    }

    fn apply(&mut self, proposal: Proposal<P>) -> Result<Action, Error>
    where
        P: Clone,
    {
        let boundary = self.boundary(&proposal.sources, &proposal.absorbed_values)?;
        fn same<T: Copy + Ord>(left: &[T], right: &[T]) -> bool {
            let set = left.iter().copied().collect::<SortedSet<_>>();
            set.len() == left.len() && set == right.iter().copied().collect()
        }
        let mut origins = LookupMap::new();
        for (slot, ports) in proposal.results.iter().enumerate() {
            for port in ports {
                let port = self.canonical(*port)?;
                if origins.insert(port, slot).is_some_and(|old| old != slot) {
                    return Err(Error::Routing);
                }
            }
        }
        let routes = boundary
            .outputs
            .iter()
            .map(|port| origins.get(port).copied().map(|slot| (*port, slot)).ok_or(Error::Routing))
            .collect::<Result<Vec<_>, _>>()?;
        if !same(&proposal.accounted_constraints, &boundary.constraints) {
            return Err(Error::Ordering);
        }

        // Stage the entire small owned graph. No identity or incidence changes
        // escape a rejected proposal, including a rejected quotient DAG.
        let mut next = self.clone();
        let target = next.group_ids.next_id();
        let scope = next.groups[&proposal.sources[0]].scope;
        let members = proposal
            .sources
            .iter()
            .flat_map(|source| next.groups[source].members.iter().copied())
            .collect();
        let outputs = (0..proposal.results.len())
            .map(|_| next.alloc_value(Value::Result(target)))
            .collect::<Vec<_>>();
        let mapping = routes.iter().map(|(port, slot)| (*port, outputs[*slot])).collect::<Vec<_>>();
        for (old, new) in &mapping {
            next.values[*old] = Value::Alias(*new);
        }
        // Unobserved retired results must not keep an inactive producer alive.
        for source in &proposal.sources {
            let retired = next.groups.remove(source).ok_or(Error::StaleGroup)?;
            for port in retired.outputs {
                if matches!(next.values[port], Value::Result(_)) {
                    next.values[port] = Value::Input;
                }
            }
        }
        next.groups.insert(
            target,
            Group {
                scope,
                members,
                inputs: boundary.inputs,
                outputs,
                payload: proposal.payload,
            },
        );
        next.constraints.retain(|id, _| !boundary.constraints.contains(id));
        for edge in next.constraints.values_mut() {
            if proposal.sources.contains(&edge.before) {
                edge.before = target;
            }
            if proposal.sources.contains(&edge.after) {
                edge.after = target;
            }
        }
        next.order()?;
        let action = Action {
            sources: proposal.sources,
            target,
            outputs: mapping,
        };
        next.actions.push(action.clone());
        *self = next;
        Ok(action)
    }

    pub fn order(&self) -> Result<Vec<GroupId>, Error> {
        let mut groups = self.groups.keys().copied().collect::<Vec<_>>();
        groups.sort_by_key(|id| self.groups[id].members.first().copied());
        let mut dependencies = BTreeMap::<GroupId, SortedSet<GroupId>>::new();
        for (id, group) in &self.groups {
            for port in &group.inputs {
                for producer in self.producers(*port)? {
                    if self.group(producer)?.scope != group.scope {
                        return Err(Error::Scope);
                    }
                    dependencies.entry(*id).or_default().insert(producer);
                }
            }
        }
        for edge in self.constraints.values() {
            dependencies.entry(edge.after).or_default().insert(edge.before);
        }
        wyn_graph::topo_sort_by_dependencies(groups, |id, out| {
            out.extend(dependencies.get(&id).into_iter().flatten().copied());
        })
        .map_err(|_| Error::Cycle)
    }

    pub fn finalize(mut self) -> Result<Plan<S, R, P>, Error>
    where
        S: std::hash::Hash,
    {
        let order = self.order()?;
        let mut scopes = StableMap::<S, Vec<GroupId>>::new();
        for id in &order {
            scopes.entry(self.groups[id].scope).or_default().push(*id);
        }
        let aliases = self
            .values
            .iter()
            .filter_map(|(port, value)| matches!(value, Value::Alias(_)).then_some(*port))
            .collect::<Vec<_>>();
        for port in aliases {
            self.values[port] = Value::Alias(self.canonical(port)?);
        }
        Ok(Plan {
            graph: self,
            order,
            scopes,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Plan<S, R, P> {
    graph: Graph<S, R, P>,
    order: Vec<GroupId>,
    scopes: StableMap<S, Vec<GroupId>>,
}

impl<S: Copy + Eq, R: Copy + Eq, P> Plan<S, R, P> {
    pub fn group(&self, id: GroupId) -> Result<&Group<S, P>, Error> {
        self.graph.group(id)
    }
    pub fn canonical(&self, port: PortId) -> Result<PortId, Error> {
        self.graph.canonical(port)
    }
    pub fn groups(&self) -> impl Iterator<Item = (GroupId, &Group<S, P>)> {
        self.order.iter().map(|id| (*id, &self.graph.groups[id]))
    }
    pub fn actions(&self) -> &[Action] {
        self.graph.actions()
    }
    pub fn boundary(&self, group: GroupId) -> Result<Boundary, Error> {
        self.graph.boundary(&[group], &[])
    }

    pub fn scopes(&self) -> impl Iterator<Item = (S, &[GroupId])> {
        self.scopes.iter().map(|(scope, groups)| (*scope, groups.as_slice()))
    }

    pub fn changed(&self, id: GroupId) -> bool {
        self.graph.groups.get(&id).is_some_and(|group| !group.members.contains(&id))
    }

    pub fn changed_scopes(&self) -> impl Iterator<Item = S> + '_ {
        self.scopes()
            .filter(|(_, groups)| groups.iter().any(|id| self.changed(*id)))
            .map(|(scope, _)| scope)
    }

    /// Direct mappings to surviving results, including absorbed pure values.
    pub fn replacements(&self) -> impl Iterator<Item = (PortId, PortId)> + '_ {
        self.graph.values.iter().filter_map(|(port, value)| match value {
            Value::Alias(target) if matches!(self.graph.values[*target], Value::Result(_)) => {
                Some((*port, *target))
            }
            _ => None,
        })
    }

    pub fn survives(&self, port: PortId) -> Result<bool, Error> {
        Ok(matches!(
            self.graph.values[self.canonical(port)?],
            Value::Result(_)
        ))
    }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod lib_tests;
