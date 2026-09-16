//! Specialize callees before callers, so new captures flow outward in one visit.
use super::super::super::{regions::Regions, snapshot};
use super::*;

pub(super) struct Context<'a> {
    pub placements: &'a mut Placements,
    pub rewrite: rewrite::Rewriter,
    pub uses: uses::Uses,
    pub live: BTreeSet<OperationId>,
    pub lexical: Regions,
    pub originals: BTreeMap<RegionId, RegionId>,
    pub specializations: BTreeMap<(RegionId, Vec<ExprId>), RegionId>,
    done: BTreeSet<RegionId>,
    visiting: BTreeSet<RegionId>,
    total: BTreeMap<ExprId, bool>,
    choices: BTreeMap<(RegionId, Vec<ParameterId>), Vec<ExprId>>,
}

pub(super) fn run(data: &mut AssociatedData, placements: &mut Placements) -> Result<(), OptimizeError> {
    let summary = timing::time("analyze dependencies", || snapshot::analyze(data));
    timing::time("validate dependency order", || summary.schedules(data))?;
    let uses = timing::time("collect expression uses", || uses::analyze(data, &summary.live));
    let regions: Vec<_> = uses.scopes.keys().copied().collect();
    let mut context = Context {
        placements,
        rewrite: timing::time("index expressions", || rewrite::Rewriter::new(data)),
        uses,
        live: summary.live,
        lexical: Regions::new(data),
        originals: BTreeMap::new(),
        specializations: BTreeMap::new(),
        done: BTreeSet::new(),
        visiting: BTreeSet::new(),
        total: BTreeMap::new(),
        choices: BTreeMap::new(),
    };
    let _timing = timing::span("specialize bodies");
    for r in regions {
        context.region(data, r)?;
    }
    Ok(())
}

impl Context<'_> {
    pub fn original(&self, r: RegionId) -> RegionId {
        self.originals.get(&r).copied().unwrap_or(r)
    }

    fn region(&mut self, data: &mut AssociatedData, r: RegionId) -> Result<(), OptimizeError> {
        if self.done.contains(&r) {
            return Ok(());
        }
        if !self.visiting.insert(r) {
            return Err(error("recursive SOAC region during capture placement"));
        }
        let lexical = self.lexical.children.get(&r).cloned().unwrap_or_default();
        for child in lexical {
            if self.uses.scopes.contains_key(&child) {
                self.region(data, child)?;
            }
        }
        let operations: Vec<_> = data.regions[r].members.intersection(&self.live).copied().collect();
        for op in operations {
            let mut nested = vec![];
            data.operations[op].kind.operands(&mut vec![], &mut nested);
            for child in nested {
                self.region(data, child)?;
            }
            self.operation(data, op)?;
        }
        self.uses.region(data, r, &self.live);
        self.visiting.remove(&r);
        self.done.insert(r);
        Ok(())
    }

    fn totals(&mut self, data: &AssociatedData, roots: &[ExprId]) {
        self.uses.dag.include(data, roots);
        // Immutable expression IDs make these proofs reusable across invocations.
        for &e in &self.uses.dag.order[self.total.len()..] {
            let total =
                total_node(data, e) && data.expressions[e].kind.children().iter().all(|x| self.total[x]);
            self.total.insert(e, total);
        }
    }

    fn operation(&mut self, data: &mut AssociatedData, op: OperationId) -> Result<(), OptimizeError> {
        let bodies = data.operations[op].kind.callbacks();
        let mut bindings: BTreeMap<RegionId, BTreeSet<ParameterId>> = BTreeMap::new();
        for body in bodies {
            let SoacBody::Apply {
                region,
                parameters,
                captures,
                ..
            } = body
            else {
                continue;
            };
            let formals = &data.regions[*region].parameters;
            if formals.len() != parameters.len() + captures.len() {
                return Err(error("SOAC capture arity"));
            }
            self.totals(data, captures);
            let safe: BTreeSet<_> = formals[parameters.len()..]
                .iter()
                .zip(captures)
                .filter_map(|(&p, e)| self.total[e].then_some(p))
                .collect();
            bindings.entry(*region).and_modify(|known| known.retain(|p| safe.contains(p))).or_insert(safe);
        }
        for (region, safe) in bindings {
            let key = (region, safe.iter().copied().collect());
            if !self.choices.contains_key(&key) {
                let expressions: BTreeSet<_> =
                    self.uses.dag.sets.iter(self.uses.scopes[&region]).map(ExprId::from).collect();
                let mut invariant = BTreeMap::new();
                let mut selected = vec![];
                for e in ordered(data, &expressions) {
                    let safe = match data.expressions[e].kind {
                        ExprKind::Parameter(p) => safe.contains(&p),
                        ExprKind::OperationResult(_) => false,
                        _ => {
                            total_node(data, e)
                                && data.expressions[e].kind.children().iter().all(|x| invariant[x])
                        }
                    };
                    invariant.insert(e, safe);
                    if safe && analysis::computation(data, e) {
                        selected.push(e);
                    }
                }
                self.choices.insert(key.clone(), selected);
            }
            let values = self.choices[&key].clone();
            if !values.is_empty() {
                specialize::apply(data, op, region, &values, self)?;
            }
        }
        Ok(())
    }
}
