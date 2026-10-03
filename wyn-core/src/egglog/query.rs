//! Borrowed access to native tables. No facts are retained outside the egraph.
use super::OptimizeError;
use egglog_engine::{EGraph, IntoValues, Read, Value};

#[derive(Clone, Copy)]
pub(super) struct Query<'a>(pub &'a EGraph);

impl Query<'_> {
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

    pub fn inverse(&self, name: &str, value: Value) -> Result<Option<Vec<Value>>, OptimizeError> {
        let mut keys = None;
        self.for_function(name, |inputs, output| {
            if output == value {
                if keys.is_some() {
                    return Err(OptimizeError::Output(format!(
                        "ambiguous inverse {name} for {value:?}"
                    )));
                }
                keys = Some(inputs.to_vec());
            }
            Ok(())
        })?;
        Ok(keys)
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
        self.0.constructor_enodes(name, |row| {
            if matches(row.children) {
                ambiguous |= fields.is_some();
                fields = Some(row.children.to_vec());
            }
        })?;
        if ambiguous {
            return Err(OptimizeError::Output(format!("ambiguous {name}")));
        }
        Ok(fields)
    }

    pub fn flag(&self, name: &str, keys: impl IntoValues) -> Result<bool, OptimizeError> {
        Ok(self.0.value_to_base::<bool>(self.required(name, keys)?))
    }
}
