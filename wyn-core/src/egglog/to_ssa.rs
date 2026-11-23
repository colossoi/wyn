//! Read the final memory program once to emit SSA and its public interfaces.
use super::query::Query;
use super::representation::return_type;
use super::representation::same_value_structure;
pub(super) use super::representation::{
    domain_axes, reducer_state_type, runtime_binding, value_type, Axis,
};
use super::{execution::Execution, InterfacesPrepared, OptimizeError, Program};
use crate::ast::Span;
use crate::ssa::builder::BuilderError;
use crate::ssa::stage::Elaborated;
use crate::ssa::types::{Function, ValueRef};
use crate::types::{self, Type};
use crate::{CodegenTarget, FunctionId, LookupMap};
use egglog_engine::{sort::S, Value};
use wyn_base::IdSource;

mod body;
mod entry;
pub(super) use super::host;
mod kernels;
use body::Body;

pub(super) fn lower(
    program: &Program<'_, InterfacesPrepared>,
    execution: &Execution,
    target: CodegenTarget,
) -> Result<Elaborated, OptimizeError> {
    let query = Query(&program.graph);
    let definitions = query.0.function_roots().collect();
    let mut externals = LookupMap::default();
    query.for_each("External", |row| {
        externals.insert(row[0], row[0]);
        Ok(())
    })?;
    let mut entries = Vec::new();
    query.for_function("EntryOrder", |keys, order| {
        entries.push((query.0.value_to_base::<i64>(order), keys[0]));
        Ok(())
    })?;
    entries.sort_by_key(|&(order, _)| order);
    let mut compiler = Compiler {
        program,
        definitions,
        externals,
        execution,
        functions: Vec::new(),
        callable_ids: LookupMap::default(),
        function_ids: IdSource::new(),
    };
    entry::lower(
        &mut compiler,
        entries.into_iter().map(|(_, function)| function).collect(),
        target,
    )
}

pub(super) struct Compiler<'a, 'source> {
    pub program: &'a Program<'source, InterfacesPrepared>,
    pub definitions: LookupMap<Value, Value>,
    externals: LookupMap<Value, Value>,
    pub functions: Vec<Function>,
    pub execution: &'a Execution,
    callable_ids: LookupMap<Value, (FunctionId, Vec<Type>, Type)>,
    function_ids: IdSource<FunctionId>,
}

impl Compiler<'_, '_> {
    fn function(&mut self, name: Value) -> Result<(FunctionId, Vec<Type>, Type), OptimizeError> {
        if let Some(selected) = self.callable_ids.get(&name) {
            return Ok(selected.clone());
        }
        let query = Query(&self.program.graph);
        let root = self.definitions.get(&name).copied();
        let external = self.externals.contains_key(&name);
        if root.is_none() && !external {
            return Err(error("unknown graph function"));
        }
        let name_fields = query.fields("FunctionId", name)?;
        let text = self.program.graph.value_to_base::<S>(name_fields[1]).to_string();
        let params = if let Some(root) = root {
            query.0.parameters(root).into_iter().try_fold(Vec::new(), |mut out, name| {
                let Some(binding) = runtime_binding(query, name)? else {
                    return Err(error(
                        "memory-bearing shader helper must be inlined before emission",
                    ));
                };
                out.push(binding);
                Ok::<_, OptimizeError>(out)
            })?
        } else {
            let count = super::analysis::count(query, "ExternalParameterCount", name)?;
            (0..count)
                .map(|slot| {
                    let parameter = query.required("ExternalParameterName", (name, slot))?;
                    let info = query.required("ExternalParameterType", (name, slot))?;
                    Ok((parameter, value_type(query, info)?))
                })
                .collect::<Result<Vec<_>, OptimizeError>>()?
        };
        let count = super::analysis::count(query, "FunctionResultCount", name)?;
        let result_types = (0..count)
            .filter_map(|slot| {
                match query
                    .required("FunctionResultType", (name, slot))
                    .and_then(|info| return_type(query, info))
                {
                    Ok(Some(ty)) => Some(Ok(ty)),
                    Ok(None) => None,
                    Err(error) => Some(Err(error)),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let result_type =
            if let [ty] = result_types.as_slice() { ty.clone() } else { types::tuple(result_types) };
        let id = self.function_ids.next_id();
        let signature = (
            id,
            params.iter().map(|(_, ty)| ty.clone()).collect(),
            result_type.clone(),
        );
        self.callable_ids.insert(name, signature.clone());
        let mut body = Body::new(self, name, params, result_type.clone());
        let body = if external {
            body.builder.terminate(crate::ssa::types::Terminator::Unreachable).map_err(builder_error)?;
            body.builder.finish().map_err(builder_error)?
        } else {
            let Some(root) = root else {
                return Err(error("function body is missing"));
            };
            let values = body.block(root)?;
            let result = if values.len() == 1 {
                values.into_iter().next().ok_or_else(|| error("call result missing"))?
            } else {
                body.tuple(values)?
            };
            body.finish(result)?
        };
        self.functions.push(Function {
            id,
            name: text.clone(),
            body,
            span: Span::generated(),
            linkage_name: external.then_some(text),
        });
        Ok(signature)
    }
}

/// Logical values become SSA values; accumulator tokens remain compile-time capabilities.
/// SSA values carry their concrete storage identity; logical bindings constrain
/// their element structure without replacing that identity with a placeholder.

#[derive(Clone)]
struct Typed {
    value: ValueRef,
    ty: Type,
}

pub(super) fn error(message: impl Into<String>) -> OptimizeError {
    OptimizeError::Output(format!("SSA lowering: {}", message.into()))
}
fn builder_error(error: BuilderError) -> OptimizeError {
    OptimizeError::Output(error.to_string())
}

#[cfg(test)]
#[path = "to_ssa_tests.rs"]
mod tests;
