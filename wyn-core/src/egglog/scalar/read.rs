//! Scalar admission reads source identities directly from the structural graph.
use crate::egglog::{query::Query, OptimizeError};
use egglog_engine::{sort::VecContainer, Value};

impl Query<'_> {
    pub(super) fn duplicable(&self, source: Value) -> Result<bool, OptimizeError> {
        // Only summarized operation boundaries carry this proof. Ordinary
        // scalar expressions acquire their safety proofs in scalar-analysis.
        Ok(self.lookup("SourceDuplicable", (source,))?.is_some_and(|v| self.0.value_to_base::<bool>(v)))
    }
    pub(super) fn parameter(&self, source: Value) -> Result<Option<(Value, i64)>, OptimizeError> {
        let Some(region) = self.lookup("SourceParameterRegion", (source,))? else {
            return Ok(None);
        };
        let index = self.required("SourceParameterIndex", (source,))?;
        Ok(Some((region, self.0.value_to_base::<i64>(index))))
    }

    pub(super) fn alias(&self, source: Value) -> Result<Option<Value>, OptimizeError> {
        self.lookup("SourceAlias", (source,))
    }

    pub(super) fn callable(&self, source: Value) -> Result<Option<Value>, OptimizeError> {
        Ok(self.row("SourceCallable", |r| r[0] == source)?.map(|r| r[1]))
    }

    pub(super) fn call(&self, source: Value) -> Result<Option<(Value, Vec<Value>)>, OptimizeError> {
        let Some(fields) = self.enode("SourceInvocation", source)? else {
            return Ok(None);
        };
        let Some(callee) = self.callable(fields[1])? else {
            return Err(OptimizeError::Output("callback invocation has no callee".into()));
        };
        let Some(arguments) = self.0.value_to_container::<VecContainer>(fields[2]) else {
            return Err(OptimizeError::Output(
                "callback invocation has no arguments".into(),
            ));
        };
        Ok(Some((callee, arguments.data.clone())))
    }

    pub(super) fn operation(&self, source: Value) -> Result<Option<Value>, OptimizeError> {
        self.lookup("SourceValueOperation", (source,))
    }

    pub(super) fn rematerialized(&self, source: Value) -> Result<bool, OptimizeError> {
        match self.operation(source)? {
            Some(op) => Ok(self
                .lookup("Rematerialize", (op,))?
                .is_some_and(|value| self.0.value_to_base::<bool>(value))),
            None => Ok(false),
        }
    }
}
