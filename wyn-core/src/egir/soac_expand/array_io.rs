//! Array length and element-access helpers for SOAC expansion.

use super::*;
use crate::egir::types::soac_element_type;

pub(super) fn emit_seg_space_len(
    graph: &mut EGraph,
    space: &SegSpace,
    fallback: &(ValueId, Type<TypeName>),
    i32_ty: &Type<TypeName>,
) -> ValueId {
    let dimensions = emit_seg_space_dimensions(graph, space, fallback, i32_ty);
    let Some(first) = dimensions.first().copied() else {
        return emit_length(graph, fallback.0, &fallback.1, i32_ty);
    };
    dimensions.into_iter().skip(1).fold(first, |product, dimension| {
        graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Multiply),
            smallvec![product, dimension],
            i32_ty.clone(),
            None,
        )
    })
}

pub(super) fn emit_seg_space_dimensions(
    graph: &mut EGraph,
    space: &SegSpace,
    fallback: &(ValueId, Type<TypeName>),
    i32_ty: &Type<TypeName>,
) -> Vec<ValueId> {
    use crate::egir::types::SegExtent;

    let mut dimensions = Vec::with_capacity(space.dims().len());
    for extent in space.dims() {
        let dimension = match extent {
            SegExtent::Fixed(count) => {
                graph.intern_pure(PureOp::Int(count.to_string()), smallvec![], i32_ty.clone(), None)
            }
            SegExtent::PushConstant { node, .. } => *node,
            SegExtent::Value(node) => {
                let ty = graph.nodes[*node].ty.clone();
                if is_plain_array_source(&ty) {
                    emit_length(graph, *node, &ty, i32_ty)
                } else {
                    *node
                }
            }
            SegExtent::ResourceLength { view, .. } => {
                let node = view.value();
                let ty = graph.nodes[node].ty.clone();
                emit_length(graph, node, &ty, i32_ty)
            }
        };
        dimensions.push(dimension);
    }
    if dimensions.is_empty() {
        dimensions.push(emit_length(graph, fallback.0, &fallback.1, i32_ty));
    }
    dimensions
}

/// Decode a row-major flattened lane into one coordinate per logical domain
/// dimension.
pub(super) fn emit_flat_domain_coordinates(
    graph: &mut EGraph,
    lane: ValueId,
    domain_dimensions: &[ValueId],
    i32_ty: &Type<TypeName>,
) -> Vec<ValueId> {
    (0..domain_dimensions.len())
        .map(|dimension| {
            let suffix =
                domain_dimensions.iter().copied().skip(dimension + 1).fold(None, |product, extent| {
                    Some(match product {
                        None => extent,
                        Some(product) => graph.intern_pure(
                            PureOp::BinOp(crate::op::BinaryOperator::Multiply),
                            smallvec![product, extent],
                            i32_ty.clone(),
                            None,
                        ),
                    })
                });
            let divided = suffix.map_or(lane, |stride| {
                graph.intern_pure(
                    PureOp::BinOp(crate::op::BinaryOperator::Divide),
                    smallvec![lane, stride],
                    i32_ty.clone(),
                    None,
                )
            });
            let extent = domain_dimensions[dimension];
            graph.intern_pure(
                PureOp::BinOp(crate::op::BinaryOperator::Remainder),
                smallvec![divided, extent],
                i32_ty.clone(),
                None,
            )
        })
        .collect()
}

/// Emit the length of an input array in the requested integer type.
/// Composite, view, and virtual arrays share `_w_intrinsic_length`. For a SoA
/// tuple, the length is the length of component 0 (all components share it
/// post-`tlc::soa`).
pub(super) fn emit_length(
    graph: &mut EGraph,
    arr_nid: ValueId,
    arr_ty: &Type<TypeName>,
    result_ty: &Type<TypeName>,
) -> ValueId {
    let actual_arr_ty =
        graph.nodes.get(arr_nid).map(|node| &node.ty).filter(|ty| is_plain_array_source(ty)).cloned();
    let arr_ty = actual_arr_ty.as_ref().unwrap_or(arr_ty);
    if let Some(components) = as_soa_tuple(arr_ty) {
        let first_arr = graph.intern_pure(
            PureOp::Project { index: 0 },
            smallvec![arr_nid],
            components[0].clone(),
            None,
        );
        return emit_length(graph, first_arr, &components[0], result_ty);
    }
    let length_id = catalog().known().length;
    graph.intern_pure(
        PureOp::Intrinsic {
            id: length_id,
            overload_idx: 0,
        },
        smallvec![arr_nid],
        result_ty.clone(),
        None,
    )
}

/// Emit a per-iteration read of `arr[idx]` at the given body block.
/// Composite arrays use a pure `Index`; view arrays use `StorageViewIndex` +
/// effectful `Load`.
pub(super) fn emit_read_element(
    graph: &mut EGraph,
    body: BlockId,
    arr_nid: ValueId,
    idx_nid: ValueId,
    arr_ty: &Type<TypeName>,
    elem_ty: &Type<TypeName>,
    next_effect: &mut crate::IdSource<EffectToken>,
) -> ValueId {
    let actual_arr_ty =
        graph.nodes.get(arr_nid).map(|node| &node.ty).filter(|ty| is_plain_array_source(ty)).cloned();
    let arr_ty = actual_arr_ty.as_ref().unwrap_or(arr_ty);
    // SoA tuple: project each component array, recursively read element i
    // from each, repack as the element tuple.
    if let Some(components) = as_soa_tuple(arr_ty) {
        let elem_components: Vec<Type<TypeName>> = components
            .iter()
            .map(|ct| {
                if ct.is_array() {
                    ct.elem_type().expect("Array has elem").clone()
                } else if as_soa_tuple(ct).is_some() {
                    soac_element_type(ct)
                } else {
                    ct.clone()
                }
            })
            .collect();
        let mut elem_nids: SmallVec<[ValueId; 4]> = SmallVec::with_capacity(components.len());
        for (i, (comp_ty, comp_elem_ty)) in components.iter().zip(elem_components.iter()).enumerate() {
            let comp_arr = graph.intern_pure(
                PureOp::Project { index: i as u32 },
                smallvec![arr_nid],
                comp_ty.clone(),
                None,
            );
            let e = emit_read_element(graph, body, comp_arr, idx_nid, comp_ty, comp_elem_ty, next_effect);
            elem_nids.push(e);
        }
        return graph.intern_pure(PureOp::Tuple(components.len()), elem_nids, elem_ty.clone(), None);
    }
    if is_view_node(graph, arr_nid, arr_ty) {
        let place = graph.add_view_index_place(graph.view_id(arr_nid), idx_nid, elem_ty.clone(), None);
        if <WynLanguage as super::super::types::Language>::is_materialized_aggregate(elem_ty) {
            let region = arr_ty.array_buffer().cloned().unwrap_or_else(crate::types::no_buffer);
            let view_ty = crate::types::view_array_of(elem_ty, region);
            return graph.add_place_view(place, view_ty, None).value();
        }
        let load_result = graph.alloc_side_effect_result(elem_ty.clone());
        let eff_in = alloc_effect(next_effect);
        let eff_out = alloc_effect(next_effect);
        let result = graph.value_result(load_result);
        graph.skeleton.blocks[body].side_effects.push(SideEffect {
            kind: SideEffectKind::Effect(EffectOp::Load { place }),
            operands: smallvec![],
            result: Some(result),
            effects: Some((eff_in, eff_out)),
            span: None,
        });
        load_result
    } else if is_virtual_source(arr_ty) {
        // Virtual {start, step, len}: elem = start + i * step.
        let start_nid = graph.intern_pure(
            PureOp::Project { index: 0 },
            smallvec![arr_nid],
            elem_ty.clone(),
            None,
        );
        let step_nid = graph.intern_pure(
            PureOp::Project { index: 1 },
            smallvec![arr_nid],
            elem_ty.clone(),
            None,
        );
        let mul_nid = graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Multiply),
            smallvec![idx_nid, step_nid],
            elem_ty.clone(),
            None,
        );
        graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Add),
            smallvec![start_nid, mul_nid],
            elem_ty.clone(),
            None,
        )
    } else {
        graph.intern_pure(PureOp::Index, smallvec![arr_nid, idx_nid], elem_ty.clone(), None)
    }
}

/// Read one leaf from a regular rank-N array using a row-major flattened lane
/// index. Storage-backed arrays retain nested `ViewIndex`/`PlaceIndex`
/// addressing, while composite arrays use nested pure `Index` operations.
pub(super) fn emit_read_ranked_element(
    graph: &mut EGraph,
    body: BlockId,
    arr_nid: ValueId,
    flat_index: ValueId,
    arr_ty: &Type<TypeName>,
    leaf_ty: &Type<TypeName>,
    rank: u8,
    layout: &ArrayLayout,
    next_effect: &mut crate::IdSource<EffectToken>,
) -> ValueId {
    if rank == 1 {
        return emit_read_ranked_coordinates(
            graph,
            body,
            arr_nid,
            &[flat_index],
            arr_ty,
            leaf_ty,
            layout,
            next_effect,
        );
    }
    let inner_extents = ranked_inner_extents(arr_ty, rank);
    let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
    let mut remaining = flat_index;
    let mut coordinates = SmallVec::<[ValueId; 4]>::with_capacity(rank as usize);
    for dimension in 0..rank as usize {
        if dimension + 1 == rank as usize {
            coordinates.push(remaining);
            continue;
        }
        let stride = inner_extents[dimension..]
            .iter()
            .copied()
            .try_fold(1u32, u32::checked_mul)
            .expect("ranked SOAC inner dimensions are too large");
        let stride_node = graph.intern_pure(
            PureOp::Int(stride.to_string()),
            smallvec![],
            i32_type.clone(),
            None,
        );
        coordinates.push(graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Divide),
            smallvec![remaining, stride_node],
            i32_type.clone(),
            None,
        ));
        remaining = graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Remainder),
            smallvec![remaining, stride_node],
            i32_type.clone(),
            None,
        );
    }

    emit_read_ranked_coordinates(
        graph,
        body,
        arr_nid,
        &coordinates,
        arr_ty,
        leaf_ty,
        layout,
        next_effect,
    )
}

/// Read one leaf using an explicit coordinate for every regular array axis.
/// This avoids flattening large tiled domains into a single scalar index.
pub(super) fn emit_read_ranked_coordinates(
    graph: &mut EGraph,
    body: BlockId,
    arr_nid: ValueId,
    coordinates: &[ValueId],
    arr_ty: &Type<TypeName>,
    leaf_ty: &Type<TypeName>,
    layout: &ArrayLayout,
    next_effect: &mut crate::IdSource<EffectToken>,
) -> ValueId {
    assert!(!coordinates.is_empty(), "ranked SOAC read requires a coordinate");
    if coordinates.len() == 1 && !matches!(layout, ArrayLayout::StorageAos) {
        return emit_read_element(graph, body, arr_nid, coordinates[0], arr_ty, leaf_ty, next_effect);
    }
    let actual_arr_ty =
        graph.nodes.get(arr_nid).map(|node| &node.ty).filter(|ty| is_plain_array_source(ty)).cloned();
    let arr_ty = actual_arr_ty.as_ref().unwrap_or(arr_ty);
    if let Some(components) = as_soa_tuple(arr_ty) {
        let mut leaves = SmallVec::<[ValueId; 4]>::with_capacity(components.len());
        for (component_index, component_ty) in components.iter().enumerate() {
            let component = graph.intern_pure(
                PureOp::Project {
                    index: component_index as u32,
                },
                smallvec![arr_nid],
                component_ty.clone(),
                None,
            );
            let component_leaf = super::super::types::soac_leaf_type(
                component_ty,
                u8::try_from(coordinates.len()).expect("SOAC input rank exceeds u8"),
            );
            leaves.push(emit_read_ranked_coordinates(
                graph,
                body,
                component,
                coordinates,
                component_ty,
                &component_leaf,
                &ArrayLayout::Composite,
                next_effect,
            ));
        }
        return graph.intern_pure(PureOp::Tuple(components.len()), leaves, leaf_ty.clone(), None);
    }

    if matches!(layout, ArrayLayout::StorageAos) || is_view_node(graph, arr_nid, arr_ty) {
        let mut current_ty = arr_ty.clone();
        let first_ty = current_ty.elem_type().expect("ranked SOAC input must be an array").clone();
        let mut place =
            graph.add_view_index_place(graph.view_id(arr_nid), coordinates[0], first_ty.clone(), None);
        current_ty = first_ty;
        for coordinate in coordinates.iter().skip(1) {
            let next_ty = current_ty.elem_type().expect("ranked SOAC input rank exceeds its type").clone();
            place = graph.add_index_place(place, *coordinate, next_ty.clone(), None);
            current_ty = next_ty;
        }
        if <WynLanguage as super::super::types::Language>::is_materialized_aggregate(leaf_ty) {
            let region = arr_ty.array_buffer().cloned().unwrap_or_else(crate::types::no_buffer);
            let view_ty = crate::types::view_array_of(leaf_ty, region);
            return graph.add_place_view(place, view_ty, None).value();
        }
        return emit_load(graph, body, place, leaf_ty.clone(), next_effect, None);
    }

    let mut value = arr_nid;
    let mut current_ty = arr_ty.clone();
    for coordinate in coordinates {
        let next_ty = current_ty.elem_type().expect("ranked SOAC input rank exceeds its type").clone();
        value = graph.intern_pure(
            PureOp::Index,
            smallvec![value, *coordinate],
            next_ty.clone(),
            None,
        );
        current_ty = next_ty;
    }
    value
}

/// Storage-backed entry arrays can retain their fixed composite surface type
/// while the value itself is a `StorageView`. Addressing follows the producer
/// operation in that case; relying only on the array variant would try to
/// materialize the view's `(offset, length)` handle as the full fixed array.
fn is_view_node(graph: &EGraph, arr_nid: ValueId, arr_ty: &Type<TypeName>) -> bool {
    is_view_source(arr_ty)
        || matches!(
            graph.nodes[arr_nid].kind,
            crate::egir::ir::ValueKind::Pure {
                op: PureOp::StorageView(_),
                ..
            }
        )
}

fn ranked_inner_extents(arr_ty: &Type<TypeName>, rank: u8) -> Vec<u32> {
    let mut ty = arr_ty;
    let mut extents = Vec::with_capacity(rank.saturating_sub(1) as usize);
    for dimension in 0..rank {
        while let Some(components) = as_soa_tuple(ty) {
            ty = components.first().expect("structure-of-arrays tuple has a component");
        }
        if dimension > 0 {
            let Type::Constructed(TypeName::Size(size), _) =
                ty.array_size().expect("ranked SOAC inner dimension has no size")
            else {
                panic!("ranked SOAC inner dimensions must be fixed")
            };
            extents.push(u32::try_from(*size).expect("ranked SOAC dimension is too large"));
        }
        ty = ty.elem_type().expect("ranked SOAC input rank exceeds its type");
    }
    extents
}
