//! Runtime types derived from logical values before physical planning.
use super::analysis::count;
use super::query::Query;
use super::{output_error as error, OptimizeError};
use crate::ssa::layout::storage_value_type;
use crate::types::{self, Type, TypeExt, TypeName};
use crate::BindingRef;
use egglog_engine::Value;

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

pub(super) fn return_type(query: Query<'_>, info: Value) -> Result<Option<Type>, OptimizeError> {
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
    let Some((name, neutral_ty)) = runtime_binding(query, neutral)? else {
        return Err(error("collective neutral has no runtime type"));
    };
    let size = if let Some(count) = query.lookup("LogicalOuterCount", name)? {
        query
            .lookup("StaticIndex", count)?
            .map(|size| {
                usize::try_from(query.0.value_to_base::<i64>(size))
                    .map_err(|_| error("negative reducer state extent"))
            })
            .transpose()?
    } else {
        None
    };
    let size = size.or_else(|| match neutral_ty.array_size() {
        Some(Type::Constructed(TypeName::Size(size), _)) => Some(*size),
        _ => None,
    });
    if let Some(size) = size {
        let Some(element) = ty.elem_type() else {
            return Err(error("array reducer element type missing"));
        };
        return Ok(types::sized_array(size, element.clone()));
    }
    Ok(ty)
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
                vec![types::buffer_tag(BindingRef::new(set, slot))],
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

pub(super) fn primitive_type(query: Query<'_>, primitive: Value) -> Result<Type, OptimizeError> {
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

#[derive(Clone, Copy)]
pub(super) struct Axis {
    pub name: Value,
    pub extent: Value,
}

pub(super) fn domain_axes(query: Query<'_>, domain: Value) -> Result<Vec<Axis>, OptimizeError> {
    (0..count(query, "DomainRank", domain)?)
        .map(|slot| {
            Ok(Axis {
                name: query.required("DomainAxis", (domain, slot))?,
                extent: query.required("DomainExtent", (domain, slot))?,
            })
        })
        .collect()
}

pub(super) fn same_value_structure(actual: &Type, expected: &Type) -> bool {
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
