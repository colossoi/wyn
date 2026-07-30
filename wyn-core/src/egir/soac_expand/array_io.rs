//! Array length and element-access helpers for SOAC expansion.

use super::*;

pub(super) fn emit_seg_space_len(
    graph: &mut EGraph,
    space: &SegSpace,
    fallback: &(NodeId, Type<TypeName>),
    i32_ty: &Type<TypeName>,
) -> NodeId {
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
            SegExtent::ResourceLength { node, .. } => {
                let ty = graph.nodes[*node].ty.clone();
                emit_length(graph, *node, &ty, i32_ty)
            }
        };
        dimensions.push(dimension);
    }
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

/// Emit the length of an input array in the requested integer type.
/// Composite, view, and virtual arrays share `_w_intrinsic_length`. For a SoA
/// tuple, the length is the length of component 0 (all components share it
/// post-`tlc::soa`).
pub(super) fn emit_length(
    graph: &mut EGraph,
    arr_nid: NodeId,
    arr_ty: &Type<TypeName>,
    result_ty: &Type<TypeName>,
) -> NodeId {
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
    arr_nid: NodeId,
    idx_nid: NodeId,
    arr_ty: &Type<TypeName>,
    elem_ty: &Type<TypeName>,
    next_effect: &mut crate::IdSource<EffectToken>,
) -> NodeId {
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
        let mut elem_nids: SmallVec<[NodeId; 4]> = SmallVec::with_capacity(components.len());
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
    if is_view_source(arr_ty) {
        // View array: ViewIndex (pure, PlaceId) + Load (effectful).
        let ptr_nid = graph.intern_pure(
            PureOp::ViewIndex,
            smallvec![arr_nid, idx_nid],
            elem_ty.clone(),
            None,
        );
        let load_result = graph.alloc_side_effect_result(elem_ty.clone());
        let eff_in = alloc_effect(next_effect);
        let eff_out = alloc_effect(next_effect);
        graph.skeleton.blocks[body].side_effects.push(SideEffect {
            kind: SideEffectKind::Effect(EffectOp::Load),
            operand_nodes: smallvec![ptr_nid],
            result: Some(load_result),
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

/// Emit a per-iteration write `arr[idx] = val`, producing the new array node.
///
/// `elem_ty` must be the logical element type of `arr_ty`:
/// - Plain composite array: `arr_ty.elem_type()`.
/// - SoA tuple: `soac_element_type(arr_ty)` (a tuple whose components line
///   up with `as_soa_tuple(arr_ty)`).
///
/// For a SoA tuple, this projects each component array out of `arr_nid`,
/// projects the matching component out of `val_nid`, recursively writes,
/// and repacks a `PureOp::Tuple`. For a plain composite array, this emits
/// `_w_intrinsic_array_with_inplace` directly. Any other `arr_ty` (view,
/// virtual, tuple whose elements aren't all arrays) is a bug in the caller
/// — soac_expand's output arrays are always freshly-built composites.
pub(super) fn emit_write_element(
    graph: &mut EGraph,
    arr_nid: NodeId,
    idx_nid: NodeId,
    val_nid: NodeId,
    arr_ty: &Type<TypeName>,
    elem_ty: &Type<TypeName>,
) -> NodeId {
    // Invariant: the supplied elem_ty must match what arr_ty implies.
    // A mismatch means an upstream pass produced inconsistent types.
    // Hard panic — emitting silently-wrong IR in release is worse than
    // crashing loudly.
    let expected_elem_ty = derive_elem_ty(arr_ty);
    if elem_ty != &expected_elem_ty {
        panic!(
            "emit_write_element: elem_ty {:?} disagrees with arr_ty {:?} (expected elem {:?})",
            elem_ty, arr_ty, expected_elem_ty
        );
    }

    if let Some(components) = as_soa_tuple(arr_ty) {
        let Type::Constructed(TypeName::Tuple(_), elem_components) = elem_ty else {
            panic!(
                "emit_write_element: SoA-tuple arr_ty {:?} paired with non-tuple elem_ty {:?}",
                arr_ty, elem_ty
            );
        };
        if components.len() != elem_components.len() {
            panic!(
                "emit_write_element: SoA tuple arity mismatch — arr_ty has {} components, elem_ty has {}",
                components.len(),
                elem_components.len()
            );
        }
        let mut new_component_arrs: SmallVec<[NodeId; 4]> = SmallVec::with_capacity(components.len());
        for (i, (comp_arr_ty, comp_elem_ty)) in components.iter().zip(elem_components.iter()).enumerate() {
            let comp_arr = graph.intern_pure(
                PureOp::Project { index: i as u32 },
                smallvec![arr_nid],
                comp_arr_ty.clone(),
                None,
            );
            let comp_val = graph.intern_pure(
                PureOp::Project { index: i as u32 },
                smallvec![val_nid],
                comp_elem_ty.clone(),
                None,
            );
            let new_comp =
                emit_write_element(graph, comp_arr, idx_nid, comp_val, comp_arr_ty, comp_elem_ty);
            new_component_arrs.push(new_comp);
        }
        return graph.intern_pure(
            PureOp::Tuple(components.len()),
            new_component_arrs,
            arr_ty.clone(),
            None,
        );
    }

    let inplace_id = catalog().known().array_with_in_place;
    graph.intern_pure(
        PureOp::Intrinsic {
            id: inplace_id,
            overload_idx: 0,
        },
        smallvec![arr_nid, idx_nid, val_nid],
        arr_ty.clone(),
        None,
    )
}

/// The logical element type implied by `arr_ty`: `arr_ty.elem_type()` for
/// composite arrays, `soac_element_type(arr_ty)` for SoA tuples. Only used
/// by `emit_write_element`'s debug_assert.
fn derive_elem_ty(arr_ty: &Type<TypeName>) -> Type<TypeName> {
    if as_soa_tuple(arr_ty).is_some() {
        soac_element_type(arr_ty)
    } else {
        arr_ty.elem_type().expect("composite array has elem").clone()
    }
}
