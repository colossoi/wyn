//! Borrowed access to native tables. No facts are retained outside the egraph.
use super::OptimizeError;
use egglog_engine::ast::FunctionSubtype;
use egglog_engine::sort::VecContainer;
use egglog_engine::{EGraph, IntoValues, RawValues, Read, Value};

#[derive(Clone, Copy)]
pub(super) struct Query<'a>(pub &'a EGraph);

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

    pub fn sequence(&self, table: &str, owner: Value) -> Result<Vec<Value>, OptimizeError> {
        let count = self.required(&format!("{table}Count"), owner)?;
        self.indexed(table, &[owner], self.0.value_to_base::<i64>(count))
    }

    pub fn operations(&self, region: Value) -> Result<Vec<Value>, OptimizeError> {
        let mut operations = Vec::new();
        self.for_function("OperationRegion", |keys, owner| {
            if owner == region && self.contains("Operation", keys[0])? {
                let order = self.required("OperationOrder", keys[0])?;
                operations.push((self.0.value_to_base::<i64>(order), keys[0]));
            }
            Ok(())
        })?;
        operations.sort_by_key(|&(order, _)| order);
        Ok(operations.into_iter().map(|(_, operation)| operation).collect())
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
