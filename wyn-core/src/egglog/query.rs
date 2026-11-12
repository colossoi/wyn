//! Borrowed access to native tables. No facts are retained outside the egraph.
use super::OptimizeError;
use egglog_engine::sort::VecContainer;
use egglog_engine::{EGraph, IntoValues, Read, Value};

#[derive(Clone, Copy)]
pub(super) struct Query<'a>(pub &'a EGraph);

impl Query<'_> {
    pub fn fields(&self, name: &str, value: Value) -> Result<Vec<Value>, OptimizeError> {
        let Some(fields) = self.enode(name, value)? else {
            return Err(super::output_error(format!("expected {name}")));
        };
        Ok(fields)
    }

    pub fn application<'a>(
        &self,
        value: Value,
        names: &'a [&str],
    ) -> Result<(&'a str, Vec<Value>), OptimizeError> {
        for &name in names {
            if let Some(fields) = self.enode(name, value)? {
                return Ok((name, fields));
            }
        }
        Err(super::output_error(format!(
            "unexpected memory IR node {value:?}"
        )))
    }

    pub fn vector(&self, value: Value) -> Result<Vec<Value>, OptimizeError> {
        let Some(values) = self.0.value_to_container::<VecContainer>(value) else {
            return Err(OptimizeError::Output("expected a memory IR vector".into()));
        };
        Ok(values.data.clone())
    }

    pub fn for_function(
        &self,
        name: &str,
        mut f: impl FnMut(&[Value], Value) -> Result<(), OptimizeError>,
    ) -> Result<(), OptimizeError> {
        let mut result = Ok(());
        self.0.read(|r| {
            r.function_entries_while(name, |row| {
                result = f(row.inputs, row.output);
                result.is_ok()
            })
        })?;
        result
    }
    pub fn for_each(
        &self,
        name: &str,
        mut f: impl FnMut(&[Value]) -> Result<(), OptimizeError>,
    ) -> Result<(), OptimizeError> {
        let mut result = Ok(());
        self.0.read(|r| {
            r.constructor_enodes_while(name, |row| {
                result = f(row.children);
                result.is_ok()
            })
        })?;
        result
    }
    pub fn lookup(&self, name: &str, keys: impl IntoValues) -> Result<Option<Value>, OptimizeError> {
        Ok(self.0.read(|r| r.lookup(name, keys))?)
    }

    pub fn constructor(&self, name: &str, keys: impl IntoValues) -> Result<Option<Value>, OptimizeError> {
        Ok(self.0.read(|r| r.eclass_of(name, keys))?)
    }

    pub fn required(&self, name: &str, keys: impl IntoValues) -> Result<Value, OptimizeError> {
        let Some(value) = self.lookup(name, keys)? else {
            return Err(OptimizeError::Output(format!("missing selected {name}")));
        };
        Ok(value)
    }

    pub fn contains(&self, name: &str, keys: impl IntoValues) -> Result<bool, OptimizeError> {
        Ok(self.0.read(|r| r.contains(name, keys))?)
    }

    pub fn enode(&self, name: &str, key: Value) -> Result<Option<Vec<Value>>, OptimizeError> {
        let mut fields = None;
        let mut ambiguous = false;
        self.0.read(|r| {
            r.enodes_for_eclass(name, key, |row| {
                ambiguous |= fields.is_some();
                fields = Some(row.children.to_vec());
            })
        })?;
        if ambiguous {
            return Err(OptimizeError::Output(format!("ambiguous {name} for {key:?}")));
        }
        Ok(fields)
    }

    pub fn row(
        &self,
        name: &str,
        matches: impl Fn(&[Value]) -> bool,
    ) -> Result<Option<Vec<Value>>, OptimizeError> {
        let mut fields = None;
        let mut ambiguous = false;
        self.for_each(name, |row| {
            if matches(row) {
                ambiguous |= fields.is_some();
                fields = Some(row.to_vec());
            }
            Ok(())
        })?;
        if ambiguous {
            return Err(OptimizeError::Output(format!("ambiguous {name}")));
        }
        Ok(fields)
    }

    #[cfg(test)]
    pub fn flag(&self, name: &str, keys: impl IntoValues) -> Result<bool, OptimizeError> {
        Ok(self.0.value_to_base::<bool>(self.required(name, keys)?))
    }
}
