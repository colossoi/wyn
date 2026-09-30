//! Publish source result types using the shader's storage layout rules.
use crate::host::{ResultField, ResultLayout, ResultScalar};
use crate::ssa::layout::{
    std430_matrix_stride, std430_struct_layout, storage_elem_stride, storage_value_type, type_byte_size,
};
use crate::types::{strip_existentials, Type, TypeExt, TypeName};

pub(crate) fn result_layout(ty: &Type) -> ResultLayout {
    let ty = strip_existentials(ty);
    // A root SoA result is published as one array of logical tuple elements.
    // Nested fields inside a scalar storage record retain their own arrays.
    fn array_element(ty: &Type) -> Option<(Type, Type)> {
        if let Some(fields) = crate::types::as_soa_tuple(ty) {
            let fields = fields.iter().map(array_element).collect::<Option<Vec<_>>>()?;
            let size = fields.first()?.1.clone();
            if !fields.iter().all(|(_, n)| *n == size) {
                return None;
            }
            Some((
                crate::types::tuple(fields.into_iter().map(|(t, _)| t).collect()),
                size,
            ))
        } else {
            Some((ty.elem_type()?.clone(), ty.array_size()?.clone()))
        }
    }
    if crate::types::as_soa_tuple(ty).is_some() {
        if let Some((element, size)) = array_element(ty) {
            let array = crate::types::make_array1(
                element,
                crate::types::array_variant_composite(),
                size,
                Type::Constructed(TypeName::NoBuffer, vec![]),
            );
            return layout(&array, true).unwrap_or_else(|| ResultLayout::Unsupported(ty.to_string()));
        }
    }
    layout(ty, true).unwrap_or_else(|| ResultLayout::Unsupported(ty.to_string()))
}

fn layout(ty: &Type, root: bool) -> Option<ResultLayout> {
    let ty = strip_existentials(ty);
    let scalar = match ty {
        Type::Constructed(TypeName::Int(8), _) => Some(ResultScalar::I8),
        Type::Constructed(TypeName::Int(16), _) => Some(ResultScalar::I16),
        Type::Constructed(TypeName::Int(32), _) => Some(ResultScalar::I32),
        Type::Constructed(TypeName::Int(64), _) => Some(ResultScalar::I64),
        Type::Constructed(TypeName::UInt(8), _) => Some(ResultScalar::U8),
        Type::Constructed(TypeName::UInt(16), _) => Some(ResultScalar::U16),
        Type::Constructed(TypeName::UInt(32), _) => Some(ResultScalar::U32),
        Type::Constructed(TypeName::UInt(64), _) => Some(ResultScalar::U64),
        Type::Constructed(TypeName::Float(32), _) => Some(ResultScalar::F32),
        Type::Constructed(TypeName::Float(64), _) => Some(ResultScalar::F64),
        Type::Constructed(TypeName::Bool, _) => Some(ResultScalar::Bool),
        _ => None,
    };
    if let Some(scalar) = scalar {
        return Some(ResultLayout::Scalar(scalar));
    }
    if ty.is_vec() {
        return Some(ResultLayout::Sequence {
            element: Box::new(layout(ty.elem_type()?, false)?),
            count: u32::try_from(ty.vec_size()?).ok()?,
            stride: type_byte_size(ty.elem_type()?)?,
        });
    }
    if ty.is_mat() {
        let scalar = ty.as_tensor()?.elem;
        return Some(ResultLayout::Sequence {
            element: Box::new(ResultLayout::Sequence {
                element: Box::new(layout(scalar, false)?),
                count: u32::try_from(ty.mat_rows()?).ok()?,
                stride: type_byte_size(scalar)?,
            }),
            count: u32::try_from(ty.mat_cols()?).ok()?,
            stride: std430_matrix_stride(ty)?,
        });
    }
    if ty.is_array() {
        let element = ty.elem_type()?;
        let stride = storage_elem_stride(&storage_value_type(element))?;
        let length = match ty.array_size()? {
            Type::Constructed(TypeName::Size(n), _) => u32::try_from(*n).ok(),
            _ => None,
        };
        let element = Box::new(layout(element, false)?);
        return Some(if root {
            ResultLayout::Array {
                element,
                stride,
                length,
            }
        } else {
            ResultLayout::Sequence {
                element,
                count: length?,
                stride,
            }
        });
    }
    match ty {
        Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), types) => {
            let physical = types.iter().map(storage_value_type).collect::<Vec<_>>();
            let storage = std430_struct_layout(&physical.iter().collect::<Vec<_>>())?;
            let mut fields = Vec::new();
            for (index, (ty, offset)) in types.iter().zip(storage.member_offsets).enumerate() {
                let name = format!("result_{index}");
                fields.push(ResultField {
                    name,
                    offset,
                    layout: layout(ty, false)?,
                });
            }
            if let Type::Constructed(TypeName::Record(names), _) = ty {
                for (field, name) in fields.iter_mut().zip(&names.0) {
                    field.name = name.clone();
                }
                Some(ResultLayout::Record {
                    fields,
                    size: storage.size,
                })
            } else {
                Some(ResultLayout::Tuple {
                    fields,
                    size: storage.size,
                })
            }
        }
        _ => None,
    }
}
