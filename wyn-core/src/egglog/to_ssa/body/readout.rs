//! Place selected scalar terms while emitting SSA; no temporary control-flow tree.
use super::{Body, OptimizeError, TermId, Typed, Value};
use crate::LookupSet;

impl Body<'_, '_, '_> {
    pub(super) fn scalar_body(&mut self, scope: Value, root: TermId) -> Result<Typed, OptimizeError> {
        if let Some(value) = self.available_scalar(root)? {
            return Ok(value);
        }
        self.scalar_roots(scope, &[root], &mut LookupSet::default())?;
        self.scalar(scope, root)
    }

    pub(super) fn planned_boundary(&mut self, scope: Value, source: Value) -> Result<(), OptimizeError> {
        let schedule = self.compiler.evaluations.before(self.context, source).to_vec();
        for term in schedule {
            self.scalar_body(scope, term)?;
        }
        Ok(())
    }

    pub(super) fn scalar_common(&self, scope: Value, arms: [&[TermId]; 2]) -> Vec<TermId> {
        self.compiler.evaluations.common(&self.compiler.placements, scope, arms)
    }

    pub(super) fn scalar_roots(
        &mut self,
        scope: Value,
        roots: &[TermId],
        available: &mut LookupSet<TermId>,
    ) -> Result<(), OptimizeError> {
        let program = self.compiler.program;
        let selected = &program.stage.selected;
        let schedule = self.compiler.evaluations.schedule(roots)?.to_vec();
        for terms in schedule {
            let term = terms[0];
            if !available.insert(term) {
                continue;
            }
            let (name, fields) = selected.app(term)?;
            if name == "ScalarChoice" {
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
                let common = self.scalar_common(scope, [&yes, &no]);
                let condition = self.scalar(scope, fields[2])?;
                for term in common {
                    self.scalar_roots(scope, &[term], available)?;
                }
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
                let target = self.compiler.evaluations.target(scope, term);
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
