//! Publish source result types using the shader's storage layout rules.
use crate::host::{ResultField, ResultLayout, ResultScalar};
use crate::ssa::layout::{std430_matrix_stride, std430_struct_layout, storage_elem_stride, type_byte_size};
use crate::types::{strip_existentials, Type, TypeExt, TypeName};

pub(crate) fn result_layout(ty: &Type) -> ResultLayout {
    let ty = strip_existentials(ty);
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
        let stride = storage_elem_stride(element)?;
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
            let storage = std430_struct_layout(&types.iter().collect::<Vec<_>>())?;
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
