//! Materialize values demanded by both arms before creating their control split.
use super::{Body, LookupSet, OptimizeError, TermId, Value};
use crate::egglog::scalar::extract::operands;

impl Body<'_, '_, '_> {
    fn branch_operands(&self, term: TermId, guarded: bool) -> Result<Vec<TermId>, OptimizeError> {
        let selected = &self.compiler.program.stage.selected;
        let (name, fields) = selected.app(term)?;
        if name == "ScalarLeaf" {
            if let Some(&root) = selected.roots.get(&(self.context, selected.values[fields[2]])) {
                if root != term {
                    return Ok(vec![root]);
                }
            }
        }
        // An arm's condition is demanded, but its results may remain guarded.
        if name == "ScalarChoice" && guarded {
            return Ok(vec![fields[2]]);
        }
        Ok(operands(name, fields))
    }

    fn branch_demands(&self, root: TermId) -> Result<LookupSet<TermId>, OptimizeError> {
        let mut seen = LookupSet::default();
        let mut pending = vec![root];
        while let Some(term) = pending.pop() {
            if seen.insert(term) {
                pending.extend(self.branch_operands(term, true)?);
            }
        }
        Ok(seen)
    }

    fn branch_readonly(&self, root: TermId) -> Result<bool, OptimizeError> {
        let selected = &self.compiler.program.stage.selected;
        let mut seen = LookupSet::default();
        let mut pending = vec![root];
        while let Some(term) = pending.pop() {
            if !seen.insert(term) {
                continue;
            }
            let (name, fields) = selected.app(term)?;
            let proof = match name {
                "ScalarLeaf" if self.values.contains_key(&selected.values[fields[2]]) => None,
                "ScalarLeaf" | "ScalarExecute" | "ScalarInstruction" | "ScalarCall" => {
                    Some(("SourceReadOnly", selected.values[fields[2]]))
                }
                "ScalarInvoke" => Some(("SourceRegionReadOnly", selected.values[fields[2]])),
                _ => None,
            };
            if let Some((table, key)) = proof {
                if !self
                    .compiler
                    .facts
                    .lookup(table, (key,))
                    .is_some_and(|value| self.compiler.program.graph.value_to_base::<bool>(value))
                {
                    return Ok(false);
                }
            }
            pending.extend(self.branch_operands(term, false)?);
        }
        Ok(true)
    }

    pub(super) fn share_branch_values(
        &mut self,
        scope: Value,
        yes: TermId,
        no: TermId,
    ) -> Result<(), OptimizeError> {
        let required = self.branch_demands(yes)?;
        let mut pending = vec![no];
        let mut seen = LookupSet::default();
        while let Some(term) = pending.pop() {
            if !seen.insert(term) {
                continue;
            }
            let (name, _) = self.compiler.program.stage.selected.app(term)?;
            if required.contains(&term)
                && !matches!(name, "ScalarCons" | "ScalarNil")
                && self.branch_readonly(term)?
            {
                // The whole producer runs once, including any guard or loop of
                // its own. Its result dominates both consumers without making
                // the producer's conditional operands unconditional.
                self.scalar(scope, term)?;
            } else {
                pending.extend(self.branch_operands(term, true)?);
            }
        }
        Ok(())
    }
}
