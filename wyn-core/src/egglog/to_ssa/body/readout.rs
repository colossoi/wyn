//! Emit the materialization locations selected during scalar readout.
use super::{Body, OptimizeError, TermId, Typed, Value};
use crate::egglog::scalar::placement::Materialization;

impl Body<'_, '_, '_> {
    pub(super) fn scalar_body(&mut self, scope: Value, root: TermId) -> Result<Typed, OptimizeError> {
        if let Some(value) = self.available_scalar(root)? {
            return Ok(value);
        }
        let plan = self.compiler.placements.readout(self.compiler.program, scope, &[root], |source| {
            self.values.contains_key(&source)
        })?;
        self.materializations(scope, &plan)?;
        self.scalar(scope, root)
    }

    pub(super) fn materializations(
        &mut self,
        scope: Value,
        plan: &[Materialization],
    ) -> Result<(), OptimizeError> {
        for step in plan {
            match step {
                Materialization::Value(term, location) => {
                    if self.available_scalar(*term)?.is_none() {
                        let target = location
                            .and_then(|scope| self.scopes.get(&scope).copied())
                            .unwrap_or(self.current()?);
                        self.scalar_at(scope, *term, target)?;
                    }
                }
                Materialization::Choice { terms, shared, arms } => {
                    if terms
                        .iter()
                        .map(|&term| self.available_scalar(term))
                        .collect::<Result<Vec<_>, _>>()?
                        .iter()
                        .all(Option::is_some)
                    {
                        continue;
                    }
                    let (_, fields) = self.compiler.program.stage.selected.app(terms[0])?;
                    let condition = self.scalar(scope, fields[2])?;
                    self.materializations(scope, shared)?;
                    let values = self.branch(
                        scope,
                        condition,
                        |body| body.choice_results(scope, terms, &arms[0], 3),
                        |body| body.choice_results(scope, terms, &arms[1], 4),
                        None,
                    )?;
                    for (index, &term) in terms.iter().enumerate() {
                        let value = if terms.len() == 1 {
                            values.clone()
                        } else {
                            self.field(values.clone(), index)?
                        };
                        let block = self.current()?;
                        self.scalar_values.entry(term).or_default().push((block, value));
                    }
                }
            }
        }
        Ok(())
    }

    fn choice_results(
        &mut self,
        scope: Value,
        terms: &[TermId],
        plan: &[Materialization],
        arm: usize,
    ) -> Result<Typed, OptimizeError> {
        self.materializations(scope, plan)?;
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
