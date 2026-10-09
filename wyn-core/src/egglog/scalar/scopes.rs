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
            .map(|value| {
                Ok(Node {
                    value,
                    parent: parent(value)?.and_then(|p| indices.get(&p).copied()),
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
        for (scope, interval) in intervals {
            nodes[scope.0].interval = interval;
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
    pub fn ancestor(&self, a: Scope, b: Scope) -> bool {
        let (a, b) = (&self.nodes[a.0], &self.nodes[b.0]);
        a.interval.contains(b.interval.start)
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
    /// Legal lexical ancestors, annotated with loop depth and lexical depth.
    /// The policy choosing among these proofs belongs to evaluation.egg.
    pub fn sites(&self, context: Value, scope: Value, required: &[Scope]) -> Vec<(Value, usize, usize)> {
        let Some(tree) = self.contexts.get(&context) else {
            return Vec::new();
        };
        let Some(mut current) = tree.find(scope) else {
            return Vec::new();
        };
        let mut ancestors = vec![current];
        while let Some(parent) = tree.parent(current) {
            ancestors.push(parent);
            current = parent;
        }
        let mut loops = 0;
        ancestors
            .into_iter()
            .rev()
            .enumerate()
            .filter_map(|(depth, scope)| {
                let scope = tree.value(scope);
                loops += usize::from(self.loops.contains(&scope));
                self.available(required, scope).then_some((scope, loops, depth))
            })
            .collect()
    }
}
