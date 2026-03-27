//! SoA-aware array operation helpers for SSA construction.
//!
//! These helpers abstract over plain arrays vs SoA tuple-of-arrays,
//! so loop expansion code doesn't need to know about SoA layout.
//! Used by both `to_ssa.rs` (for non-SOAC array ops) and `ssa_soac_lower.rs`.

use crate::ast::TypeName;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::ValueId;
use crate::types::TypeExt;
use polytype::Type;

/// Check if a type is an SoA tuple-of-arrays (result of SoA transform on `[n](A,B)` → `([n]A, [n]B)`).
/// Returns the component types if so.
pub fn is_soa_tuple(ty: &Type<TypeName>) -> Option<&[Type<TypeName>]> {
    match ty {
        Type::Constructed(TypeName::Tuple(_), component_types) if !component_types.is_empty() => {
            let all_soa = component_types.iter().all(|ct| {
                matches!(ct, Type::Constructed(TypeName::Array, args) if args.len() == 3)
                    || is_soa_tuple(ct).is_some()
            });
            if all_soa { Some(component_types) } else { None }
        }
        _ => None,
    }
}

/// Given an SoA tuple type `([n]A, [n]B)`, return the element tuple type `(A, B)`.
pub fn soa_elem_type(soa_ty: &Type<TypeName>) -> Type<TypeName> {
    match soa_ty {
        Type::Constructed(TypeName::Tuple(n), component_types) => {
            let elem_types: Vec<Type<TypeName>> = component_types
                .iter()
                .map(|ct| match ct {
                    ty if ty.is_array() => ty.elem_type().expect("Array has elem").clone(),
                    ty if is_soa_tuple(ty).is_some() => soa_elem_type(ty),
                    _ => ct.clone(),
                })
                .collect();
            Type::Constructed(TypeName::Tuple(*n), elem_types)
        }
        _ => panic!("BUG: soa_elem_type called on non-SoA type: {:?}", soa_ty),
    }
}

/// Extract compile-time array size from a type (handles both plain arrays and SoA tuples).
pub fn extract_array_size(ty: &Type<TypeName>) -> Option<usize> {
    match ty {
        _ if ty.is_array() => match ty.array_size().expect("Array has size") {
            Type::Constructed(TypeName::Size(n), _) => Some(*n),
            _ => None,
        },
        Type::Constructed(TypeName::Tuple(_), components) if !components.is_empty() => {
            extract_array_size(&components[0])
        }
        _ => None,
    }
}

/// Emit length of an array-like value (plain array or SoA tuple-of-arrays).
pub fn soa_length(
    builder: &mut FuncBuilder,
    arr: ValueId,
    arr_ty: &Type<TypeName>,
) -> Result<ValueId, String> {
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    if let Some(components) = is_soa_tuple(arr_ty) {
        let first = builder.push_project(arr, 0, components[0].clone()).map_err(|e| e.to_string())?;
        soa_length(builder, first, &components[0])
    } else {
        builder.push_intrinsic("_w_intrinsic_length", vec![arr], i32_ty).map_err(|e| e.to_string())
    }
}

/// Emit index into an array-like value.
/// For SoA tuple-of-arrays: project each component, index each, repack as element tuple.
pub fn soa_index(
    builder: &mut FuncBuilder,
    arr: ValueId,
    index: ValueId,
    arr_ty: &Type<TypeName>,
    elem_ty: &Type<TypeName>,
) -> Result<ValueId, String> {
    if let Some(components) = is_soa_tuple(arr_ty) {
        let mut elem_values = Vec::with_capacity(components.len());
        for (i, comp_ty) in components.iter().enumerate() {
            let comp_arr =
                builder.push_project(arr, i as u32, comp_ty.clone()).map_err(|e| e.to_string())?;
            let comp_elem_ty = match comp_ty {
                ty if ty.is_array() => ty.elem_type().expect("Array has elem").clone(),
                ty if is_soa_tuple(ty).is_some() => soa_elem_type(ty),
                _ => comp_ty.clone(),
            };
            let elem = soa_index(builder, comp_arr, index, comp_ty, &comp_elem_ty)?;
            elem_values.push(elem);
        }
        builder.push_tuple(elem_values, elem_ty.clone()).map_err(|e| e.to_string())
    } else {
        builder.push_index(arr, index, elem_ty.clone()).map_err(|e| e.to_string())
    }
}

/// Emit array update on an array-like value.
/// For SoA tuple-of-arrays: project each component array and element, array_with each, repack.
pub fn soa_array_with(
    builder: &mut FuncBuilder,
    arr: ValueId,
    index: ValueId,
    elem: ValueId,
    arr_ty: &Type<TypeName>,
) -> Result<ValueId, String> {
    if let Some(components) = is_soa_tuple(arr_ty) {
        let mut new_components = Vec::with_capacity(components.len());
        for (i, comp_ty) in components.iter().enumerate() {
            let comp_arr =
                builder.push_project(arr, i as u32, comp_ty.clone()).map_err(|e| e.to_string())?;
            let comp_elem_ty = match comp_ty {
                ty if ty.is_array() => ty.elem_type().expect("Array has elem").clone(),
                ty if is_soa_tuple(ty).is_some() => soa_elem_type(ty),
                _ => comp_ty.clone(),
            };
            let comp_elem =
                builder.push_project(elem, i as u32, comp_elem_ty).map_err(|e| e.to_string())?;
            let new_comp = soa_array_with(builder, comp_arr, index, comp_elem, comp_ty)?;
            new_components.push(new_comp);
        }
        builder.push_tuple(new_components, arr_ty.clone()).map_err(|e| e.to_string())
    } else {
        builder
            .push_call("_w_intrinsic_array_with", vec![arr, index, elem], arr_ty.clone())
            .map_err(|e| e.to_string())
    }
}

/// Emit uninitialized array-like value.
/// For SoA tuple-of-arrays: create one uninit per component array, pack as tuple.
pub fn soa_uninit(builder: &mut FuncBuilder, ty: &Type<TypeName>) -> Result<ValueId, String> {
    if let Some(components) = is_soa_tuple(ty) {
        let mut uninit_components = Vec::with_capacity(components.len());
        for comp_ty in components {
            let uninit = soa_uninit(builder, comp_ty)?;
            uninit_components.push(uninit);
        }
        builder.push_tuple(uninit_components, ty.clone()).map_err(|e| e.to_string())
    } else {
        builder.push_call("_w_intrinsic_uninit", vec![], ty.clone()).map_err(|e| e.to_string())
    }
}
