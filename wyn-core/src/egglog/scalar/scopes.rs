//! Indexed lexical forests for placement. Scope identities remain opaque.
use crate::egglog::OptimizeError;
use crate::{LookupMap, LookupSet};
use egglog_engine::Value;
use wyn_graph::{forest_intervals, DfsInterval};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Scope(usize);

struct Node {
    value: Value,
    parent: Option<Scope>,
    depth: usize,
    root: Scope,
    interval: DfsInterval,
}

pub(super) struct Forest {
    nodes: Vec<Node>,
    indices: LookupMap<Value, Scope>,
}

impl Forest {
    pub fn new(
        values: impl IntoIterator<Item = Value>,
        parent: impl Fn(Value) -> Result<Option<Value>, OptimizeError>,
    ) -> Result<Self, OptimizeError> {
        let mut values: Vec<_> = values.into_iter().collect();
        values.sort();
        values.dedup();
        let indices: LookupMap<_, _> = values.iter().enumerate().map(|(i, &v)| (v, Scope(i))).collect();
        let mut nodes: Vec<_> = values
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                Ok(Node {
                    value,
                    parent: parent(value)?.and_then(|p| indices.get(&p).copied()),
                    depth: 0,
                    root: Scope(i),
                    interval: DfsInterval { start: 0, end: 0 },
                })
            })
            .collect::<Result<_, OptimizeError>>()?;
        let mut children = vec![Vec::new(); nodes.len()];
        let mut roots = Vec::new();
        for (i, node) in nodes.iter().enumerate() {
            if let Some(parent) = node.parent {
                children[parent.0].push(Scope(i));
            } else {
                roots.push(Scope(i));
            }
        }
        let intervals = forest_intervals(roots, |scope: Scope, out| {
            out.extend_from_slice(&children[scope.0])
        });
        if intervals.len() != nodes.len() {
            return Err(OptimizeError::Output("cyclic lexical scope forest".into()));
        }
        let mut preorder: Vec<_> = intervals.into_iter().collect();
        preorder.sort_by_key(|(_, interval)| interval.start);
        for (scope, interval) in preorder {
            nodes[scope.0].interval = interval;
            if let Some(parent) = nodes[scope.0].parent {
                nodes[scope.0].depth = nodes[parent.0].depth + 1;
                nodes[scope.0].root = nodes[parent.0].root;
            }
        }
        Ok(Self { nodes, indices })
    }

    pub fn find(&self, value: Value) -> Option<Scope> {
        self.indices.get(&value).copied()
    }
    pub fn value(&self, scope: Scope) -> Value {
        self.nodes[scope.0].value
    }
    pub fn parent(&self, scope: Scope) -> Option<Scope> {
        self.nodes[scope.0].parent
    }
    pub fn root(&self, scope: Scope) -> Scope {
        self.nodes[scope.0].root
    }
    pub fn ancestor(&self, a: Scope, b: Scope) -> bool {
        let (a, b) = (&self.nodes[a.0], &self.nodes[b.0]);
        a.interval.contains(b.interval.start)
    }
    pub fn common(&self, mut a: Scope, mut b: Scope) -> Option<Scope> {
        if self.root(a) != self.root(b) {
            return None;
        }
        while a != b {
            if self.nodes[a.0].depth >= self.nodes[b.0].depth {
                a = self.parent(a)?;
            } else {
                b = self.parent(b)?;
            }
        }
        Some(a)
    }
    /// Ancestor requirements are implied by a requirement in a deeper scope.
    pub fn require(&self, requirements: &mut Vec<Scope>, scope: Scope) {
        if requirements.iter().any(|&old| self.ancestor(scope, old)) {
            return;
        }
        requirements.retain(|&old| !self.ancestor(old, scope));
        requirements.push(scope);
    }
}

pub(super) struct Scopes {
    definitions: Forest,
    contexts: LookupMap<Value, Forest>,
    loops: LookupSet<Value>,
}

impl Scopes {
    pub fn new(definitions: Forest, contexts: LookupMap<Value, Forest>, loops: LookupSet<Value>) -> Self {
        Self {
            definitions,
            contexts,
            loops,
        }
    }
    pub fn definition(&self, value: Value) -> Option<Scope> {
        self.definitions.find(value)
    }
    pub fn require(&self, required: &mut Vec<Scope>, value: Scope) {
        self.definitions.require(required, value);
    }
    pub fn available(&self, required: &[Scope], scope: Value) -> bool {
        self.definitions
            .find(scope)
            .is_some_and(|scope| required.iter().all(|&d| self.definitions.ancestor(d, scope)))
    }
    pub fn common(&self, context: Value, a: Value, b: Value) -> Option<Value> {
        let tree = self.contexts.get(&context)?;
        Some(tree.value(tree.common(tree.find(a)?, tree.find(b)?)?))
    }
    pub fn root(&self, context: Value, value: Value) -> Value {
        self.contexts.get(&context).and_then(|t| t.find(value).map(|s| t.value(t.root(s)))).unwrap_or(value)
    }
    pub fn outside_loops(&self, context: Value, scope: Value, required: &[Scope]) -> Value {
        let Some(tree) = self.contexts.get(&context) else {
            return scope;
        };
        let Some(mut current) = tree.find(scope) else {
            return scope;
        };
        let mut selected = scope;
        while let Some(parent) = tree.parent(current) {
            if self.loops.contains(&tree.value(current)) && self.available(required, tree.value(parent)) {
                selected = tree.value(parent);
            }
            current = parent;
        }
        selected
    }
}
