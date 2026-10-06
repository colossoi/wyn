//! Place selected scalar terms while emitting SSA; no temporary control-flow tree.
use super::{Body, OptimizeError, TermId, Typed, Value};
use crate::egglog::query::Query;
use crate::egglog::scalar::extract::operands;
use crate::{LookupMap, LookupSet};
use egglog_engine::{Read, Term};

impl Body<'_, '_, '_> {
    pub(super) fn scalar_body(&mut self, scope: Value, root: TermId) -> Result<Typed, OptimizeError> {
        if let Some(value) = self.available_scalar(root)? {
            return Ok(value);
        }
        self.scalar_roots(scope, &[root], &mut LookupSet::default())?;
        self.scalar(scope, root)
    }

    fn scalar_dependencies(&self, term: TermId) -> Vec<TermId> {
        let selected = &self.compiler.program.stage.selected;
        let Term::App(name, fields) = selected.dag.get(term) else {
            return Vec::new();
        };
        if name == "ScalarLeaf" {
            let source = selected.values[fields[2]];
            if !self.values.contains_key(&source) {
                if let Some(&root) = selected.roots.get(&(selected.values[fields[0]], source)) {
                    if root != term {
                        return vec![root];
                    }
                }
            }
        }
        if name == "ScalarChoice" {
            return vec![fields[2]];
        }
        operands(name, fields)
    }

    fn scalar_demands(&self, roots: &[TermId]) -> Vec<TermId> {
        wyn_graph::dag_postorder(
            roots.iter().copied(),
            |_| false,
            |term, out| {
                out.extend(self.scalar_dependencies(term));
            },
        )
    }

    fn scalar_readonly(&self, roots: &[TermId]) -> Result<bool, OptimizeError> {
        let scalars = &self.compiler.program.stage.scalars;
        for &term in roots {
            let (name, _) = self.compiler.program.stage.selected.app(term)?;
            let table = if matches!(name, "ScalarCons" | "ScalarNil") {
                "ScalarArgsReadOnly"
            } else {
                "ScalarReadOnly"
            };
            let value = self.compiler.program.stage.selected.values[term];
            if !scalars
                .read(|r| r.lookup(table, value))?
                .is_some_and(|value| scalars.value_to_base::<bool>(value))
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn scalar_enclosing_bindings(
        &self,
        scope: Value,
        roots: &[TermId],
    ) -> Result<LookupSet<TermId>, OptimizeError> {
        let selected = &self.compiler.program.stage.selected;
        let terms = wyn_graph::dag_postorder(
            roots.iter().copied(),
            |_| false,
            |term, out| {
                out.extend(self.scalar_dependencies(term));
                if let Term::App(name, fields) = selected.dag.get(term) {
                    if name == "ScalarChoice" {
                        out.extend_from_slice(&fields[3..5]);
                    }
                }
            },
        );
        let mut bindings = LookupSet::default();
        for term in terms {
            let (name, fields) = selected.app(term)?;
            if name != "ScalarLeaf" {
                continue;
            }
            // A value defined in an enclosing source scope was evaluated before
            // these consumers. Restore that materialization, including its own
            // guards, without speculating expressions defined inside an arm.
            let source = selected.values[fields[2]];
            if let Some(owner) = Query(&self.compiler.program.graph).lookup("ScalarOwner", (source,))? {
                if self.compiler.placements.available(term, owner, scope) {
                    bindings.insert(term);
                }
            }
        }
        Ok(bindings)
    }

    pub(super) fn scalar_common(
        &self,
        scope: Value,
        arms: [&[TermId]; 2],
    ) -> Result<Vec<TermId>, OptimizeError> {
        if !self.scalar_readonly(arms[0])? || !self.scalar_readonly(arms[1])? {
            return Ok(Vec::new());
        }
        let yes: LookupSet<_> = self.scalar_demands(arms[0]).into_iter().collect();
        let mut common: LookupSet<_> =
            self.scalar_demands(arms[1]).into_iter().filter(|term| yes.contains(term)).collect();
        let yes = self.scalar_enclosing_bindings(scope, arms[0])?;
        common.extend(
            self.scalar_enclosing_bindings(scope, arms[1])?.into_iter().filter(|term| yes.contains(term)),
        );
        Ok(common.into_iter().collect())
    }

    pub(super) fn scalar_roots(
        &mut self,
        scope: Value,
        roots: &[TermId],
        available: &mut LookupSet<TermId>,
    ) -> Result<(), OptimizeError> {
        let program = self.compiler.program;
        let selected = &program.stage.selected;
        let mut order = self.scalar_demands(roots);
        let mut groups = LookupMap::<TermId, Vec<TermId>>::default();
        if self.scalar_readonly(roots)? {
            // Mandatory dependencies precede choices, so guarded uses do not
            // duplicate a producer also demanded unconditionally in this block.
            let mut levels = LookupMap::default();
            for &term in &order {
                let (name, fields) = selected.app(term)?;
                let level =
                    self.scalar_dependencies(term).iter().map(|child| levels[child]).max().unwrap_or(0);
                levels.insert(term, level + usize::from(name == "ScalarChoice"));
                if name == "ScalarChoice" {
                    groups.entry(fields[2]).or_default().push(term);
                }
            }
            order.sort_by_key(|term| levels[term]);
        }
        for term in order {
            if !available.insert(term) {
                continue;
            }
            let (name, fields) = selected.app(term)?;
            if matches!(name, "ScalarCons" | "ScalarNil") {
                continue;
            }
            if name == "ScalarChoice" {
                let terms = groups.remove(&fields[2]).unwrap_or_else(|| vec![term]);
                if terms
                    .iter()
                    .map(|&term| self.available_scalar(term))
                    .collect::<Result<Vec<_>, _>>()?
                    .iter()
                    .all(Option::is_some)
                {
                    available.extend(terms);
                    continue;
                }
                let [yes, no] = [3, 4].map(|arm| {
                    terms
                        .iter()
                        .map(|&term| selected.app(term).map(|(_, f)| f[arm]))
                        .collect::<Result<Vec<_>, _>>()
                });
                let (yes, no) = (yes?, no?);
                let common = self.scalar_common(scope, [&yes, &no])?;
                let condition = self.scalar(scope, fields[2])?;
                self.scalar_roots(scope, &common, available)?;
                let values = self.branch(
                    scope,
                    condition,
                    |body| body.choice_results(scope, &terms, &yes, &mut available.clone(), 3),
                    |body| body.choice_results(scope, &terms, &no, &mut available.clone(), 4),
                    None,
                )?;
                for (index, &term) in terms.iter().enumerate() {
                    let value =
                        if terms.len() == 1 { values.clone() } else { self.field(values.clone(), index)? };
                    let block = self.current()?;
                    self.scalar_values.entry(term).or_default().push((block, value));
                }
                available.extend(terms);
            } else if self.available_scalar(term)?.is_none() {
                let context = selected.values[fields[0]];
                let total =
                    program.stage.scalars.read(|r| r.contains("ScalarTotal", selected.values[term]))?;
                let target = total.then(|| self.compiler.placements.outside_loops(context, scope, term));
                let target =
                    target.and_then(|scope| self.scopes.get(&scope).copied()).unwrap_or(self.current()?);
                self.scalar_at(scope, term, target)?;
            }
        }
        Ok(())
    }

    fn choice_results(
        &mut self,
        scope: Value,
        terms: &[TermId],
        roots: &[TermId],
        available: &mut LookupSet<TermId>,
        arm: usize,
    ) -> Result<Typed, OptimizeError> {
        self.scalar_roots(scope, roots, available)?;
        let mut values = Vec::new();
        for &term in terms {
            let selected = &self.compiler.program.stage.selected;
            let (_, fields) = selected.app(term)?;
            let layout = self.compiler.facts.layout(selected.values[fields[1]])?;
            let value = self.scalar(scope, fields[arm])?;
            let value = self.materialize(value, layout)?;
            if terms.len() == 1 {
                return Ok(value);
            }
            let block = self.current()?;
            self.scalar_values.entry(term).or_default().push((block, value.clone()));
            values.push(value);
        }
        self.tuple(values)
    }
}
