//! Borrowed access to native tables. No facts are retained outside the egraph.
use super::OptimizeError;
use crate::egglog::graph::Graph;
use egglog_engine::ast::FunctionSubtype;
use egglog_engine::sort::VecContainer;
use egglog_engine::{IntoValues, RawValues, Read, Value};

#[derive(Clone, Copy)]
pub(super) struct Query<'a>(pub &'a Graph);

impl Query<'_> {
    pub fn indexed(&self, table: &str, keys: &[Value], count: i64) -> Result<Vec<Value>, OptimizeError> {
        if count < 0 {
            return Err(super::output_error(format!("negative {table} count")));
        }
        (0..count)
            .map(|slot| {
                let mut keys = keys.to_vec();
                keys.push(self.0.base_to_value(slot));
                self.required(table, RawValues(keys))
            })
            .collect()
    }

    pub fn operations(&self, region: Value) -> Result<Vec<Value>, OptimizeError> {
        let mut operations = Vec::new();
        let positions = self.0.structure().read();
        for &operation in positions.by_region.get(&region).into_iter().flatten() {
            if self.0.has_operation(operation) {
                let Some(order) = positions.operations[&operation].order else {
                    return Err(super::output_error("operation in a region has no position"));
                };
                operations.push((order, operation));
            }
        }
        operations.sort_by_key(|&(order, _)| order);
        Ok(operations.into_iter().map(|(_, operation)| operation).collect())
    }

    pub fn loop_parameter(&self, operation: Value) -> Result<Value, OptimizeError> {
        let body = self.0.control_region(operation, 0)?;
        // The induction variable and while predicate are the last body parameters.
        let parameter = self.0.parameter(body, self.0.parameter_count(body) - 1)?;
        let Some(value) = self.value_name(parameter)? else {
            return Err(super::output_error(
                "loop control parameter has no value identity",
            ));
        };
        Ok(value)
    }

    pub fn name(&self, name: Value) -> Result<Value, OptimizeError> {
        let (_, fields) = self.application(name, &["ValueNameRef", "AccNameRef"])?;
        Ok(fields[0])
    }

    pub fn value_name(&self, name: Value) -> Result<Option<Value>, OptimizeError> {
        Ok(self.enode("ValueNameRef", name)?.map(|fields| fields[0]))
    }
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
        self.0.read(|r| match r.table_subtype(name) {
            Some(FunctionSubtype::Constructor) => r.constructor_enodes_while(name, |row| {
                result = f(row.children);
                result.is_ok()
            }),
            Some(FunctionSubtype::Custom) => r.function_entries_while(name, |row| {
                let mut fields = row.inputs.to_vec();
                fields.push(row.output);
                result = f(&fields);
                result.is_ok()
            }),
            None => {
                result = Err(super::output_error(format!("unregistered graph table {name}")));
                Ok(())
            }
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

    pub fn first(&self, name: &str) -> Result<Option<Vec<Value>>, OptimizeError> {
        let mut first = None;
        self.for_each(name, |row| {
            if first.is_none() {
                first = Some(row.to_vec());
            }
            Ok(())
        })?;
        Ok(first)
    }

    pub fn any(&self, name: &str, matches: impl Fn(&[Value]) -> bool) -> Result<bool, OptimizeError> {
        let mut found = false;
        self.for_each(name, |row| {
            found |= matches(row);
            Ok(())
        })?;
        Ok(found)
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

    pub fn flag(&self, name: &str, keys: impl IntoValues) -> Result<bool, OptimizeError> {
        Ok(self.0.value_to_base::<bool>(self.required(name, keys)?))
    }
}
