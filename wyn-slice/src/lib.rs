//! Checked dependency closure and executable boundaries for one immutable snapshot.
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use wyn_graph::{walk_reachable, WalkDecision, WalkOrder};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Node<V, O> {
    Value(V),
    Operation(O),
}

#[derive(Clone, Copy)]
pub enum Observer<O, T> {
    Operation(O),
    Terminator(T),
}

pub enum Definition<V, O> {
    Input,
    Pure(Vec<V>),
    Flow(Vec<V>),
    Produced(O),
}

pub struct Graph<V, O, T> {
    dependencies: HashMap<Node<V, O>, Vec<Node<V, O>>>,
    users: HashMap<Node<V, O>, Vec<Node<V, O>>>,
    pure_users: HashMap<Node<V, O>, Vec<Node<V, O>>>,
    observers: HashMap<V, Vec<T>>,
    results: HashMap<O, Vec<V>>,
    flows: HashSet<V>,
    inputs: HashSet<V>,
}

#[derive(Clone, Debug)]
pub struct LiveSlice<V, O> {
    values: HashSet<V>,
    operations: HashSet<O>,
    inputs: HashSet<V>,
    requested: Vec<V>,
    live_outs: Vec<V>,
}

impl<V, O> LiveSlice<V, O> {
    pub fn values(&self) -> &HashSet<V> {
        &self.values
    }
    pub fn operations(&self) -> &HashSet<O> {
        &self.operations
    }
    pub fn inputs(&self) -> &HashSet<V> {
        &self.inputs
    }
    pub fn requested(&self) -> &[V] {
        &self.requested
    }
    pub fn live_outs(&self) -> &[V] {
        &self.live_outs
    }
    pub fn outputs(&self) -> impl Iterator<Item = &V> {
        self.requested.iter().chain(&self.live_outs)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error<V, O> {
    Missing(Node<V, O>),
    Unsupplied(V),
    OutsideRegion(Node<V, O>),
}

impl<V: Copy + Eq + Hash + Ord, O: Copy + Eq + Hash, T: Copy + Eq + Hash> Graph<V, O, T> {
    pub fn new(
        values: impl IntoIterator<Item = (V, Definition<V, O>)>,
        operations: impl IntoIterator<Item = (O, Vec<V>)>,
        observers: impl IntoIterator<Item = (T, Vec<V>)>,
    ) -> Self {
        let mut graph = Self {
            dependencies: HashMap::new(),
            users: HashMap::new(),
            pure_users: HashMap::new(),
            observers: HashMap::new(),
            results: HashMap::new(),
            flows: HashSet::new(),
            inputs: HashSet::new(),
        };
        for (value, definition) in values {
            let pure = matches!(definition, Definition::Pure(_));
            let dependencies = match definition {
                Definition::Input => {
                    graph.inputs.insert(value);
                    vec![]
                }
                Definition::Pure(inputs) => inputs.into_iter().map(Node::Value).collect(),
                Definition::Flow(inputs) => {
                    graph.flows.insert(value);
                    inputs.into_iter().map(Node::Value).collect()
                }
                Definition::Produced(operation) => {
                    graph.results.entry(operation).or_default().push(value);
                    vec![Node::Operation(operation)]
                }
            };
            graph.insert(Node::Value(value), dependencies, pure);
        }
        for (operation, inputs) in operations {
            graph.insert(
                Node::Operation(operation),
                inputs.into_iter().map(Node::Value).collect(),
                true,
            );
        }
        for (observer, values) in observers {
            for value in values {
                graph.observers.entry(value).or_default().push(observer);
            }
        }
        graph
    }

    fn insert(&mut self, node: Node<V, O>, dependencies: Vec<Node<V, O>>, pure: bool) {
        assert!(
            !self.dependencies.contains_key(&node),
            "duplicate dependency definition"
        );
        for &dependency in &dependencies {
            self.users.entry(dependency).or_default().push(node);
            if pure {
                self.pure_users.entry(dependency).or_default().push(node);
            }
        }
        self.dependencies.insert(node, dependencies);
    }

    pub fn inputs(&self, node: Node<V, O>) -> impl Iterator<Item = V> + '_ {
        self.dependencies.get(&node).into_iter().flatten().filter_map(|node| match node {
            Node::Value(value) => Some(*value),
            _ => None,
        })
    }

    pub fn observers(&self, source: V, full: bool) -> (HashSet<O>, HashSet<T>) {
        let mut operations = HashSet::new();
        let mut observers = HashSet::new();
        let users = if full { &self.users } else { &self.pure_users };
        wyn_graph::for_each_reachable(
            [Node::Value(source)],
            WalkOrder::DepthFirst,
            |node, out| out.extend(users.get(&node).into_iter().flatten().copied()),
            |node| match node {
                Node::Operation(operation) => {
                    operations.insert(operation);
                }
                Node::Value(value) => observers.extend(self.observers.get(&value).into_iter().flatten()),
            },
        );
        (operations, observers)
    }

    pub fn pure_reaches(&self, source: V, target: V) -> bool {
        wyn_graph::reaches_ordered(
            Node::Value(source),
            Node::Value(target),
            WalkOrder::DepthFirst,
            |node, out| out.extend(self.pure_users.get(&node).into_iter().flatten().copied()),
        )
    }

    pub fn select(
        &self,
        requested: impl IntoIterator<Item = V>,
        demands: impl IntoIterator<Item = Node<V, O>>,
        supplied: &HashSet<V>,
        allowed: impl Fn(Node<V, O>) -> bool,
        retained: impl Fn(Observer<O, T>) -> bool,
        external: &[V],
    ) -> Result<LiveSlice<V, O>, Error<V, O>> {
        let mut seen = HashSet::new();
        let requested = requested.into_iter().filter(|value| seen.insert(*value)).collect();
        let mut slice = LiveSlice {
            values: HashSet::new(),
            operations: HashSet::new(),
            inputs: HashSet::new(),
            requested,
            live_outs: Vec::new(),
        };
        let roots = slice.requested.iter().copied().map(Node::Value).chain(demands);
        let error = walk_reachable(
            roots,
            WalkOrder::DepthFirst,
            |node, out| out.extend(self.dependencies.get(&node).into_iter().flatten().copied()),
            |node| {
                if !self.dependencies.contains_key(&node) {
                    return WalkDecision::Break(Error::Missing(node));
                }
                if let Node::Value(value) = node {
                    slice.values.insert(value);
                    if supplied.contains(&value) {
                        slice.inputs.insert(value);
                        return WalkDecision::Prune;
                    }
                }
                if !allowed(node) {
                    return WalkDecision::Break(Error::OutsideRegion(node));
                }
                if let Node::Value(value) = node {
                    if self.inputs.contains(&value) {
                        return WalkDecision::Break(Error::Unsupplied(value));
                    }
                }
                if let Node::Operation(operation) = node {
                    slice.operations.insert(operation);
                }
                WalkDecision::Continue
            },
        );
        if let Some(error) = error {
            return Err(error);
        }
        let candidates = slice
            .operations
            .iter()
            .flat_map(|operation| self.results.get(operation).into_iter().flatten())
            .chain(slice.values.intersection(&self.flows));
        for &value in candidates {
            if slice.requested.contains(&value) || slice.inputs.contains(&value) {
                continue;
            }
            let (operations, observers) = self.observers(value, false);
            if operations
                .iter()
                .any(|op| !slice.operations.contains(op) && retained(Observer::Operation(*op)))
                || observers.into_iter().any(|observer| retained(Observer::Terminator(observer)))
                || external.iter().any(|root| self.pure_reaches(value, *root))
            {
                if !allowed(Node::Value(value)) {
                    return Err(Error::OutsideRegion(Node::Value(value)));
                }
                slice.live_outs.push(value);
            }
        }
        slice.live_outs.sort_unstable();
        slice.values.extend(&slice.live_outs);
        Ok(slice)
    }
}

#[cfg(test)]
mod tests;
