//! Scalar formulas for coordinate maps and scoped update bodies.
use super::{error, Body, OptimizeError, Typed, Value};
use crate::{op::BinaryOperator, types};
use egglog_engine::sort::S;

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn formula(
        &mut self,
        scope: Value,
        value: Value,
        input: &Typed,
        bound: &[Typed],
    ) -> Result<Typed, OptimizeError> {
        if let Some(fields) = self.compiler.facts.enode("FormulaInput", value) {
            return self.field(
                input.clone(),
                self.compiler.facts.unsigned(fields[0], "formula argument")? as usize,
            );
        }
        if let Some(fields) = self.compiler.facts.enode("FormulaBound", value) {
            let index = self.compiler.facts.unsigned(fields[0], "formula binding")? as usize;
            let Some(value) = bound.iter().rev().nth(index) else {
                return Err(error("formula binding is out of scope"));
            };
            return Ok(value.clone());
        }
        if let Some(fields) = self.compiler.facts.enode("FormulaConstant", value) {
            return self.literal(self.compiler.facts.integer(fields[0]), &types::i32());
        }
        if let Some(fields) = self.compiler.facts.enode("FormulaExtent", value) {
            return self.extent(scope, fields[0]);
        }
        let Some(fields) = self.compiler.facts.enode("FormulaBinary", value) else {
            return Err(error("invalid scalar formula"));
        };
        let name = self.compiler.program.graph.value_to_base::<S>(fields[0]);
        let operator = BinaryOperator::try_from(name.as_str())
            .map_err(|_| error("invalid formula binary operator"))?;
        let left = self.formula(scope, fields[1], input, bound)?;
        let right = self.formula(scope, fields[2], input, bound)?;
        self.binary(operator, left, right)
    }
}
