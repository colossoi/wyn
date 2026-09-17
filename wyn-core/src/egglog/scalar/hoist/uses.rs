//! Shared expression closures and reachable structured scope uses.
use crate::egglog::data::{ExprId, ExprKind, Ir, OperationId, OperationKind, RegionId};
use std::collections::{BTreeMap, BTreeSet};
use wyn_base::persistent_sets::{Set, Sets, EMPTY};

#[derive(Default)]
pub(super) struct Dag {
    pub sets: Sets,
    pub descendants: BTreeMap<ExprId, Set>,
    pub lambdas: BTreeMap<ExprId, Set>,
    pub order: Vec<ExprId>,
    pub roots: BTreeSet<ExprId>,
}
impl Dag {
    pub fn include(&mut self, data: &Ir, roots: &[ExprId]) -> Set {
        self.roots.extend(roots);
        let order = wyn_graph::dag_postorder(
            roots.iter().copied(),
            |e| self.descendants.contains_key(&e),
            |e, out| out.extend(data.expressions[e].kind.children()),
        );
        for e in order {
            let mut set = self.sets.singleton(e.as_u32());
            let mut lambdas = match data.expressions[e].kind {
                ExprKind::Lambda(r) => self.sets.singleton(r.as_u32()),
                _ => EMPTY,
            };
            for child in data.expressions[e].kind.children() {
                set = self.sets.union(set, self.descendants[&child]);
                lambdas = self.sets.union(lambdas, self.lambdas[&child]);
            }
            self.descendants.insert(e, set);
            self.lambdas.insert(e, lambdas);
            self.order.push(e);
        }
        let mut result = EMPTY;
        for root in roots {
            result = self.sets.union(result, self.descendants[root]);
        }
        result
    }
}
#[derive(Default)]
pub(super) struct Uses {
    pub dag: Dag,
    pub operations: BTreeSet<OperationId>,
    pub scopes: BTreeMap<RegionId, Set>,
}
impl Uses {
    pub fn region(&mut self, data: &Ir, region: RegionId, live: &BTreeSet<OperationId>) {
        let mut roots = data.regions[region].results.clone();
        let mut nested = vec![];
        for &op in data.regions[region].members.intersection(live) {
            data.operations[op].kind.operands(&mut roots, &mut vec![]);
            nested.extend(data.operations[op].kind.structured_regions());
        }
        let mut set = self.dag.include(data, &roots);
        for r in nested {
            set = self.dag.sets.union(set, self.scopes[&r]);
        }
        self.scopes.insert(region, set);
    }
}

pub(super) fn analyze(data: &Ir, live: &BTreeSet<OperationId>) -> Uses {
    let symbols: BTreeMap<_, _> = data.definitions.values().map(|d| (d.symbol, d.body)).collect();
    let mut pending: Vec<_> = data.entries.values().map(|e| data.definitions[e.definition].body).collect();
    let mut uses = Uses::default();
    let mut structured: BTreeMap<RegionId, Vec<RegionId>> = BTreeMap::new();
    while let Some(region) = pending.pop() {
        if uses.scopes.contains_key(&region) {
            continue;
        }
        let source = &data.regions[region];
        let mut roots = source.results.clone();
        for &op in source.members.intersection(live) {
            uses.operations.insert(op);
            let kind = &data.operations[op].kind;
            kind.operands(&mut roots, &mut pending);
            if let OperationKind::EvalGlobal(s) = kind {
                pending.extend(symbols.get(s).copied());
            }
            structured.entry(region).or_default().extend(kind.structured_regions());
        }
        let old = uses.dag.order.len();
        let set = uses.dag.include(data, &roots);
        for &e in &uses.dag.order[old..] {
            match data.expressions[e].kind {
                ExprKind::Lambda(r) => pending.push(r),
                ExprKind::Global(s) | ExprKind::Closure { code: s, .. } => {
                    pending.extend(symbols.get(&s).copied())
                }
                ExprKind::OperationResult(op) => pending.push(data.operations[op].region),
                _ => {}
            }
        }
        uses.scopes.insert(region, set);
    }
    let order = wyn_graph::dag_postorder(
        uses.scopes.keys().copied(),
        |_| false,
        |r, out| out.extend(structured.get(&r).into_iter().flatten().copied()),
    );
    for r in order {
        let mut set = uses.scopes[&r];
        for c in structured.get(&r).into_iter().flatten() {
            set = uses.dag.sets.union(set, uses.scopes[c]);
        }
        uses.scopes.insert(r, set);
    }
    uses
}
