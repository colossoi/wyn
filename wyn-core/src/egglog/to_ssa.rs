//! Read the final memory program once to emit SSA and its public interfaces.
use super::query::Query;
use super::{execution::Execution, OptimizeError, Optimized, Program};
use crate::ast::Span;
use crate::ssa::builder::BuilderError;
use crate::ssa::layout::storage_value_type;
use crate::ssa::stage::Elaborated;
use crate::ssa::types::{Function, ValueRef};
use crate::types::{self, Type, TypeExt, TypeName};
use crate::{CodegenTarget, FunctionId, LookupMap};
use egglog_engine::{sort::S, Value};
use wyn_base::IdSource;

mod body;
mod entry;
mod host;
mod kernels;
use body::Body;

pub(super) fn lower(
    program: &Program<'_, Optimized>,
    target: CodegenTarget,
) -> Result<Elaborated, OptimizeError> {
    let query = Query(&program.graph);
    let mut definitions = LookupMap::default();
    query.for_function("FunctionRoot", |keys, root| {
        definitions.insert(keys[0], root);
        Ok(())
    })?;
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
        execution: Execution::select(&program.graph)?,
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

#[derive(Clone, Copy)]
pub(super) struct Axis {
    pub name: Value,
    pub extent: Value,
}

pub(super) fn domain_axes(query: Query<'_>, domain: Value) -> Result<Vec<Axis>, OptimizeError> {
    (0..super::analysis::count(query, "DomainRank", domain)?)
        .map(|slot| {
            Ok(Axis {
                name: query.required("DomainAxis", (domain, slot))?,
                extent: query.required("DomainExtent", (domain, slot))?,
            })
        })
        .collect()
}

pub(super) struct Compiler<'a, 'source> {
    pub program: &'a Program<'source, Optimized>,
    pub definitions: LookupMap<Value, Value>,
    externals: LookupMap<Value, Value>,
    pub functions: Vec<Function>,
    pub execution: Execution,
    callable_ids: LookupMap<Value, (FunctionId, Type)>,
    function_ids: IdSource<FunctionId>,
}

impl Compiler<'_, '_> {
    fn function(&mut self, name: Value) -> Result<(FunctionId, Type), OptimizeError> {
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
            query.sequence("RegionParameter", root)?.into_iter().try_fold(Vec::new(), |mut out, name| {
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
        self.callable_ids.insert(name, (id, result_type.clone()));
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
        Ok((id, result_type))
    }
}

/// Logical values become SSA values; accumulator tokens remain compile-time capabilities.
pub(super) fn runtime_binding(
    query: Query<'_>,
    binding: Value,
) -> Result<Option<(Value, Type)>, OptimizeError> {
    let Some(name) = query.value_name(binding)? else {
        return Ok(None);
    };
    let info = query.required("ValueType", name)?;
    Ok(Some((name, value_type(query, info)?)))
}

pub(super) fn value_type(query: Query<'_>, info: Value) -> Result<Type, OptimizeError> {
    let (kind, fields) = query.application(info, &["ScalarInfo", "ResourceInfo", "LogicalValue"])?;
    match kind {
        "ScalarInfo" => element_type(query, fields[0]),
        "ResourceInfo" => resource_type(query, fields[0]),
        "LogicalValue" => logical_type(query, fields[0]),
        _ => unreachable!(),
    }
}

fn return_type(query: Query<'_>, info: Value) -> Result<Option<Type>, OptimizeError> {
    let (kind, fields) = query.application(
        info,
        &[
            "ReturnScalar",
            "ReturnAccumulator",
            "ReturnResource",
            "LogicalReturn",
        ],
    )?;
    match kind {
        "ReturnScalar" => element_type(query, fields[0]).map(Some),
        "ReturnAccumulator" => Ok(None),
        "ReturnResource" => resource_type(query, fields[0]).map(Some),
        "LogicalReturn" => logical_type(query, fields[0]).map(Some),
        _ => unreachable!(),
    }
}

fn logical_type(query: Query<'_>, ty: Value) -> Result<Type, OptimizeError> {
    let (kind, fields) =
        query.application(ty, &["ElementValue", "ProductType", "ArrayType", "ResourceValue"])?;
    match kind {
        "ElementValue" => element_type(query, fields[0]),
        "ResourceValue" => resource_type(query, fields[0]),
        "ProductType" => Ok(types::tuple(
            query
                .vector(fields[0])?
                .into_iter()
                .map(|field| logical_type(query, field))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        "ArrayType" => {
            let mut element = logical_type(query, fields[2])?;
            for extent in query.vector(fields[1])?.into_iter().rev() {
                let size = if let Some(fixed) = query.enode("FixedExtent", extent)? {
                    let size = usize::try_from(query.0.value_to_base::<i64>(fixed[0]))
                        .map_err(|_| error("negative array extent"))?;
                    Type::Constructed(TypeName::Size(size), vec![])
                } else {
                    Type::Constructed(TypeName::SizePlaceholder, vec![])
                };
                element = types::make_array1(
                    element,
                    types::array_variant_composite(),
                    size,
                    types::no_buffer(),
                );
            }
            Ok(element)
        }
        _ => unreachable!(),
    }
}

/// A fixed neutral and a length-preserving reducer define fixed SSA state,
/// including when transport has erased that length from a buffer-view type.
pub(super) fn reducer_state_type(
    query: Query<'_>,
    _function: Value,
    reducer: Value,
    slot: usize,
    neutral: Value,
    ty: Type,
) -> Result<Type, OptimizeError> {
    if !ty.is_array()
        || matches!(ty.array_size(), Some(Type::Constructed(TypeName::Size(_), _)))
        || !query.contains("ReducerPreservesOuterLength", (reducer, slot as i64))?
    {
        return Ok(ty);
    }
    if let Some(name) = query.value_name(neutral)? {
        if let Some(count) = query.lookup("LogicalOuterCount", name)? {
            if let Some(size) = query.lookup("StaticIndex", count)? {
                let size = usize::try_from(query.0.value_to_base::<i64>(size))
                    .map_err(|_| error("negative reducer state extent"))?;
                let Some(element) = ty.elem_type() else {
                    return Err(error("array reducer element type missing"));
                };
                return Ok(types::sized_array(size, element.clone()));
            }
        }
    }
    Ok(ty)
}

/// SSA values carry their concrete storage identity; logical bindings constrain
/// their element structure without replacing that identity with a placeholder.
fn same_value_structure(actual: &Type, expected: &Type) -> bool {
    match (actual, expected) {
        (
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields),
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), expected),
        ) => {
            fields.len() == expected.len()
                && fields.iter().zip(expected).all(|(a, b)| {
                    same_value_structure(a, b) || storage_value_type(a) == storage_value_type(b)
                })
        }
        _ if actual.is_array() && expected.is_array() => {
            if actual.array_rank().is_some_and(|rank| rank > 1)
                && actual.array_variant().is_some_and(types::is_array_variant_view)
            {
                let mut remaining = actual.array_rank().unwrap_or(0);
                let mut element = expected;
                while remaining > 0 {
                    let Some(rank) = element.array_rank() else {
                        return false;
                    };
                    let Some(child) = element.elem_type() else {
                        return false;
                    };
                    if rank > remaining {
                        return false;
                    }
                    remaining -= rank;
                    element = child;
                }
                return actual.elem_type().is_some_and(|actual| {
                    same_value_structure(actual, element)
                        || storage_value_type(actual) == storage_value_type(element)
                });
            }
            actual.array_rank() == expected.array_rank()
                && actual.elem_type().zip(expected.elem_type()).is_some_and(|(a, b)| {
                    same_value_structure(a, b) || storage_value_type(a) == storage_value_type(b)
                })
        }
        _ => actual == expected,
    }
}

fn resource_type(query: Query<'_>, resource: Value) -> Result<Type, OptimizeError> {
    let (kind, fields) = query.application(resource, &["TextureType", "SamplerType", "ImageType"])?;
    Ok(match kind {
        "TextureType" => Type::Constructed(TypeName::Texture2D, vec![]),
        "SamplerType" => Type::Constructed(TypeName::Sampler, vec![]),
        "ImageType" => {
            let set = u32::try_from(query.0.value_to_base::<i64>(fields[0]))
                .map_err(|_| error("invalid image descriptor set"))?;
            let slot = u32::try_from(query.0.value_to_base::<i64>(fields[1]))
                .map_err(|_| error("invalid image descriptor slot"))?;
            Type::Constructed(
                TypeName::StorageTexture,
                vec![types::buffer_tag(crate::BindingRef::new(set, slot))],
            )
        }
        _ => unreachable!(),
    })
}

fn element_type(query: Query<'_>, element: Value) -> Result<Type, OptimizeError> {
    if let Some(fields) = query.enode("MatrixElement", element)? {
        let rows = usize::try_from(query.0.value_to_base::<i64>(fields[1]))
            .map_err(|_| error("invalid matrix rows"))?;
        let columns = usize::try_from(query.0.value_to_base::<i64>(fields[2]))
            .map_err(|_| error("invalid matrix columns"))?;
        return Ok(types::mat(rows, columns, primitive_type(query, fields[0])?));
    }
    if let Some(fields) = query.enode("ProductElement", element)? {
        return Ok(types::tuple(
            query
                .vector(fields[0])?
                .into_iter()
                .map(|field| element_type(query, field))
                .collect::<Result<_, _>>()?,
        ));
    }
    let (name, fields) = query.application(element, &["ScalarElement", "VectorElement"])?;
    let primitive = primitive_type(query, fields[0])?;
    if name == "ScalarElement" {
        return Ok(primitive);
    }
    let width = usize::try_from(query.0.value_to_base::<i64>(fields[1]))
        .map_err(|_| error("invalid vector width"))?;
    Ok(Type::Constructed(
        TypeName::Vec,
        vec![primitive, Type::Constructed(TypeName::Size(width), vec![])],
    ))
}

fn primitive_type(query: Query<'_>, primitive: Value) -> Result<Type, OptimizeError> {
    for (name, ty) in [
        ("I8", TypeName::Int(8)),
        ("I16", TypeName::Int(16)),
        ("I32", TypeName::Int(32)),
        ("I64", TypeName::Int(64)),
        ("U8", TypeName::UInt(8)),
        ("U16", TypeName::UInt(16)),
        ("U32", TypeName::UInt(32)),
        ("U64", TypeName::UInt(64)),
        ("F16", TypeName::Float(16)),
        ("F32", TypeName::Float(32)),
        ("F64", TypeName::Float(64)),
        ("Bool", TypeName::Bool),
        ("UnitType", TypeName::Unit),
    ] {
        if query.enode(name, primitive)?.is_some() {
            return Ok(Type::Constructed(ty, vec![]));
        }
    }
    Err(error("unknown memory primitive type"))
}

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
