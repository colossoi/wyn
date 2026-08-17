//! Shared EGraph emission primitives. The three EGIR-construction
//! contexts — `from_tlc::Converter`, `egir::builder::EntryBuilder`, and
//! the in-place rewrite helpers in `egir::parallelize` — all need to
//! intern the same set of pure ops (literals, intrinsics, BinOps,
//! StorageViews) and push the same shapes of side-effects (`Store`,
//! semantic `Soac` effects). This module owns those primitives so the three
//! contexts don't drift in their representation.
//!
//! The functions take `Option<Span>` for span attachment; pass
//! `None` when no source span is available, otherwise the caller's
//! current span. Bigger stateful helpers (`emit_store_through_view`,
//! `emit_pending_soac`) also take the target `BlockId` and a mutable
//! effect-token counter.

use crate::LookupMap;
use polytype::Type;
use smallvec::{smallvec, SmallVec};
use std::collections::{HashMap, HashSet};

use crate::ast::{Span, TypeName};
use crate::builtins::{catalog, BuiltinId};
use crate::flow::BlockId;
use crate::ssa::types::ConstantValue;
use crate::types::TypeExt;
use crate::BindingRef;

use super::ir::{Family, Language, PlaceOp, ResultTree, Value};
use super::types::{
    CallSiteId, EGraph, EffectOp, EffectToken, FuncParam, GraphResource, OperandRef, OperandType, Physical,
    PlaceAccess, PlaceDestination, PlaceId, PlaceRegion, PlaceType, PureOp, PureViewSource, Raw,
    ResourceAccess, ResultBinding, ResultDestination, SegBody, SegResourceAccess, Semantic, SideEffect,
    SideEffectKind, SideEffectSite, SkeletonTerminator, Soac, SoacEffect, ValueId, ValueKind, ViewId,
    WynLanguage, WynSoacPhase,
};

#[cfg(test)]
#[path = "graph_ops_tests.rs"]
mod graph_ops_tests;

pub fn bind_by_value_result<P: Family>(
    graph: &mut EGraph<P>,
    abi: &super::types::FunctionResult<Type<TypeName>>,
    value: ValueId,
) -> ResultBinding<Type<TypeName>> {
    fn bind<P: Family>(
        graph: &mut EGraph<P>,
        abi: &super::types::FunctionResult<Type<TypeName>>,
        value: ValueId,
    ) -> ResultBinding<Type<TypeName>> {
        let value = graph.canonical_value(value);
        if !abi.is_product() {
            return ResultBinding::destination(abi.ty().clone(), ResultDestination::ReturnValue(value));
        }
        ResultBinding::product(
            abi.ty().clone(),
            abi.top_level_fields().into_iter().enumerate().map(|(index, field)| {
                let op = PureOp::Project { index: index as u32 };
                let operands = smallvec![value];
                let projected = graph
                    .try_algebraic_fold(&op, &operands, field.ty())
                    .unwrap_or_else(|| graph.intern_pure(op, operands, field.ty().clone(), None));
                bind(graph, &field, projected)
            }),
        )
    }

    bind(graph, abi, value)
}

pub fn bind_physical_result_value<P: Family>(
    graph: &mut EGraph<P>,
    ty: Type<TypeName>,
    value: ValueId,
) -> ResultBinding<Type<TypeName>> {
    let abi = super::types::by_value_function_result::<WynLanguage>(ty);
    let binding = bind_by_value_result(graph, &abi, value);
    binding.map_destinations(|_, destination| match destination {
        ResultDestination::ReturnValue(value) => {
            let value = graph.canonical_value(*value);
            match graph.nodes[value].kind() {
                ValueKind::PlaceView { place } => ResultDestination::Place(PlaceDestination::Fixed(*place)),
                _ => ResultDestination::ReturnValue(value),
            }
        }
        ResultDestination::Place(destination) => ResultDestination::Place(destination.clone()),
    })
}

/// Recover the place denoted by a view or by a projection/index spine rooted
/// in an addressable value. This operation follows representation already
/// present in the graph; it never materializes a value or inserts effects.
pub(crate) fn addressable_value_place<P: Family>(
    graph: &mut EGraph<P>,
    value: ValueId,
    pointee: &Type<TypeName>,
) -> Option<PlaceId> {
    let value = graph.canonical_value(value);
    match graph.nodes[value].kind().clone() {
        ValueKind::PlaceView { place } => Some(place),
        ValueKind::Pure {
            op: PureOp::Index,
            operands,
        } if operands.len() == 2 => {
            let base = graph.canonical_value(operands[0]);
            let index = graph.canonical_value(operands[1]);
            let span = graph.nodes[value].span();
            if let OperandRef::View(view) = graph.operand_ref(base) {
                Some(graph.add_view_index_place(view, index, pointee.clone(), span))
            } else {
                let base_ty = graph.nodes[base].ty().clone();
                let base = addressable_value_place(graph, base, &base_ty)?;
                Some(graph.add_index_place(base, index, pointee.clone(), span))
            }
        }
        ValueKind::Pure {
            op: PureOp::Project { index },
            operands,
        } if operands.len() == 1 => {
            let base = graph.canonical_value(operands[0]);
            let base_ty = graph.nodes[base].ty().clone();
            let base = addressable_value_place(graph, base, &base_ty)?;
            let coordinate = graph.intern_pure(
                PureOp::Int(index.to_string()),
                smallvec![],
                Type::Constructed(TypeName::Int(32), vec![]),
                graph.nodes[value].span(),
            );
            Some(graph.add_index_place(base, coordinate, pointee.clone(), graph.nodes[value].span()))
        }
        _ => match graph.operand_ref(value) {
            OperandRef::View(view) => Some(graph.add_view_place(
                view,
                pointee.clone(),
                PlaceAccess::ReadWrite,
                graph.nodes[value].span(),
            )),
            OperandRef::Value(_) | OperandRef::Place(_) => None,
        },
    }
}

pub(crate) fn addressable_operand_place<P: Family>(
    graph: &mut EGraph<P>,
    operand: OperandRef,
    pointee: &Type<TypeName>,
) -> Option<PlaceId> {
    match graph.canonical_operand(operand) {
        OperandRef::Place(place) => Some(place),
        OperandRef::View(view) => addressable_value_place(graph, view.value(), pointee).or_else(|| {
            Some(graph.add_view_place(
                view,
                pointee.clone(),
                PlaceAccess::ReadOnly,
                graph.nodes[view.value()].span(),
            ))
        }),
        OperandRef::Value(value) => addressable_value_place(graph, value, pointee),
    }
}

pub(crate) fn adapt_physical_call_argument(
    graph: &mut super::program::PhysicalEGraph,
    argument: OperandRef,
    parameter: &FuncParam<super::program::PhysicalResourceRef, Type<TypeName>>,
    callee: crate::FunctionId,
    index: usize,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(OperandRef, Vec<super::program::PhysicalSideEffect>), String> {
    let argument = graph.canonical_operand(argument);
    let mut effects = Vec::new();
    let argument = match parameter.representation() {
        OperandType::Value(expected) => match argument {
            OperandRef::Value(value) if graph.nodes[value].ty() == expected => OperandRef::Value(value),
            _ => {
                return Err(format!(
                    "call to {callee:?} argument {index} does not match value parameter {expected:?}"
                ))
            }
        },
        OperandType::View(expected) => match argument {
            argument @ OperandRef::View(view)
                if WynLanguage::view_argument_matches(&expected.array, graph.nodes[view.value()].ty()) =>
            {
                argument
            }
            OperandRef::Value(value) => {
                let pointee = graph.nodes[value].ty().clone();
                let span = graph.nodes[value].span();
                let place = if let Some(place) = addressable_value_place(graph, value, &pointee) {
                    place
                } else {
                    let (place, allocation) = detached_alloca(graph, pointee.clone(), effect_ids, span);
                    effects.push(allocation);
                    effects.push(detached_store(place, value, effect_ids, span));
                    place
                };
                OperandRef::View(graph.add_place_view(
                    place,
                    view_type_for_place(graph, place, &pointee),
                    span,
                ))
            }
            OperandRef::Place(place) => {
                let pointee = graph.place(place).ty().pointee.clone();
                OperandRef::View(graph.add_place_view(
                    place,
                    view_type_for_place(graph, place, &pointee),
                    None,
                ))
            }
            argument => {
                return Err(format!(
                    "call to {callee:?} argument {index} {argument:?} does not match view parameter {:?}",
                    expected.array
                ))
            }
        },
        OperandType::Place(expected) => {
            let place = if let Some(place) = addressable_operand_place(graph, argument, &expected.pointee) {
                place
            } else {
                let value = argument.value().ok_or_else(|| {
                    format!(
                        "call to {callee:?} argument {index} cannot be materialized as {:?}",
                        expected.pointee
                    )
                })?;
                let value = graph.canonical_value(value);
                let actual = graph.nodes[value].ty().clone();
                if !WynLanguage::view_argument_matches(&expected.pointee, &actual) {
                    return Err(format!(
                        "call to {callee:?} argument {index} has type {:?}, expected place pointee {:?}",
                        actual, expected.pointee
                    ));
                }
                let span = graph.nodes[value].span();
                let (place, allocation) = detached_alloca(graph, actual, effect_ids, span);
                effects.push(allocation);
                effects.push(detached_store(place, value, effect_ids, span));
                place
            };
            OperandRef::Place(place)
        }
    };
    Ok((argument, effects))
}

fn view_type_for_place(
    graph: &super::program::PhysicalEGraph,
    place: PlaceId,
    pointee: &Type<TypeName>,
) -> Type<TypeName> {
    let region = match &graph.place(place).ty().region {
        PlaceRegion::Resource(binding) => crate::types::buffer_tag(*binding),
        PlaceRegion::Function | PlaceRegion::Workgroup | PlaceRegion::Parametric | PlaceRegion::Output => {
            crate::types::no_buffer()
        }
    };
    crate::types::view_array_of(pointee, region)
}

pub fn alloc_by_value_effect_result<P: Family>(
    graph: &mut EGraph<P>,
    ty: Type<TypeName>,
) -> ResultBinding<Type<TypeName>> {
    super::ir::by_value_function_result::<super::types::WynLanguage>(ty).bind(
        |_, ty| graph.alloc_side_effect_result(ty.clone()),
        |_| unreachable!("a by-value effect result has no destination parameters"),
    )
}

pub(crate) fn retype_projection_tree<P: Family>(
    graph: &mut EGraph<P>,
    source: ValueId,
    ty: &Type<TypeName>,
) {
    graph.retype_node(source, ty.clone());
    let Some(fields) = WynLanguage::product_fields(ty).map(<[_]>::to_vec) else {
        return;
    };
    let source = graph.canonical_value(source);
    let projections = graph
        .nodes
        .iter()
        .filter_map(|(value, definition)| match definition.kind() {
            ValueKind::Pure {
                op: PureOp::Project { index },
                operands,
            } if operands.len() == 1 && graph.canonical_value(operands[0]) == source => {
                Some((value, *index as usize))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    for (projection, index) in projections {
        if let Some(field) = fields.get(index) {
            retype_projection_tree(graph, projection, field);
        }
    }
}

pub(crate) fn normalize_place_backed_value_consumers<P: Family>(graph: &mut EGraph<P>, value: ValueId) {
    let value = graph.canonical_value(value);
    let ValueKind::PlaceView { place } = graph.nodes[value].kind() else {
        return;
    };
    let place = *place;
    let fixed_length = match graph.nodes[value].ty().array_size() {
        Some(Type::Constructed(TypeName::Size(length), _)) => Some(*length),
        _ => None,
    };
    let length = catalog().known().length;
    let consumers = graph
        .nodes
        .iter()
        .filter_map(|(consumer, node)| match node.kind() {
            ValueKind::Pure {
                op: PureOp::Intrinsic { id, .. },
                operands,
            } if *id == length && operands.len() == 1 && graph.canonical_value(operands[0]) == value => {
                Some(consumer)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    for consumer in consumers {
        let replacement = match (fixed_length, graph.nodes[consumer].ty()) {
            (Some(length), Type::Constructed(TypeName::UInt(_), _)) => ValueKind::Constant(
                ConstantValue::U32(length.try_into().expect("fixed array length exceeds u32")),
            ),
            (Some(length), _) => ValueKind::Constant(ConstantValue::I32(
                length.try_into().expect("fixed array length exceeds i32"),
            )),
            (None, _) => ValueKind::PlaceLength { place },
        };
        graph.replace_node_preserving_type(consumer, replacement);
    }

    let slice = catalog().known().slice;
    let slices = graph
        .nodes
        .iter()
        .filter_map(|(consumer, node)| match node.kind() {
            ValueKind::Pure {
                op: PureOp::Intrinsic { id, .. },
                operands,
            } if *id == slice && operands.len() == 3 && graph.canonical_value(operands[0]) == value => {
                Some((consumer, operands[1], operands[2], node.ty().clone(), node.span()))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    for (consumer, start, end, ty, span) in slices {
        let length = graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Subtract),
            smallvec![end, start],
            graph.nodes[end].ty().clone(),
            span,
        );
        let slice_place = graph.add_slice_place(place, start, length, ty.clone(), span);
        let view_ty = crate::types::view_array_of(&ty, crate::types::no_buffer());
        let slice_view = graph.add_place_view(slice_place, view_ty, span).value();
        graph.replace_value_references(consumer, slice_view);
        graph.install_aliases([(consumer, slice_view)]);
        normalize_place_backed_value_consumers(graph, slice_view);
    }
}

/// Fold projections exposed by graph rewrites that replaced an opaque value
/// with a concrete tuple, vector, or array literal. Construction-time folding
/// cannot see these shapes because the consumer can predate the replacement.
pub(crate) fn fold_exposed_projections<P: Family>(graph: &mut EGraph<P>) -> usize {
    let mut folded = 0;
    loop {
        let candidates = graph
            .nodes
            .iter()
            .filter_map(|(value, node)| match node.kind() {
                ValueKind::Pure {
                    op: PureOp::Project { .. } | PureOp::Index,
                    operands,
                } if node.alias().is_none() => {
                    Some((value, node.kind().clone(), operands.clone(), node.ty().clone()))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut changed = false;
        for (value, kind, operands, ty) in candidates {
            let ValueKind::Pure { op, .. } = kind else {
                unreachable!()
            };
            let operands = operands
                .into_iter()
                .map(|operand| graph.canonical_value(operand))
                .collect::<SmallVec<[ValueId; 4]>>();
            let Some(replacement) = graph.try_algebraic_fold(&op, &operands, &ty) else {
                continue;
            };
            let replacement = graph.canonical_value(replacement);
            if value == replacement {
                continue;
            }
            graph.replace_value_references(value, replacement);
            graph.install_aliases([(value, replacement)]);
            folded += 1;
            changed = true;
        }
        if !changed {
            return folded;
        }
    }
}

/// Resolve direct element projections of a completed place-backed aggregate
/// at the block that begins consuming it. Aggregate projections remain
/// addressable; scalar projections become explicit element loads.
pub(crate) fn materialize_place_backed_projections<P: Family>(
    graph: &mut EGraph<P>,
    value: ValueId,
    block: BlockId,
    effect_ids: &mut crate::IdSource<EffectToken>,
) {
    let mut pending = vec![graph.canonical_value(value)];
    let mut loads = Vec::new();
    while let Some(value) = pending.pop() {
        normalize_place_backed_value_consumers(graph, value);
        let ValueKind::PlaceView { place } = graph.nodes[value].kind() else {
            continue;
        };
        let place = *place;
        let consumers = graph
            .nodes
            .iter()
            .filter_map(|(consumer, node)| match node.kind() {
                ValueKind::Pure {
                    op: PureOp::Project { index },
                    operands,
                } if operands.len() == 1 && graph.canonical_value(operands[0]) == value => {
                    Some((consumer, None, *index, node.ty().clone(), node.span()))
                }
                ValueKind::Pure {
                    op: PureOp::Index,
                    operands,
                } if operands.len() == 2 && graph.canonical_value(operands[0]) == value => {
                    Some((consumer, Some(operands[1]), 0, node.ty().clone(), node.span()))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        for (consumer, dynamic_index, static_index, ty, span) in consumers {
            let index = dynamic_index.unwrap_or_else(|| {
                graph.intern_pure(
                    PureOp::Int(static_index.to_string()),
                    smallvec![],
                    Type::Constructed(TypeName::Int(32), vec![]),
                    span,
                )
            });
            let element = graph.add_index_place(place, index, ty.clone(), span);
            let replacement = if ty.array_variant().is_some() {
                let view_ty = crate::types::view_array_of(&ty, crate::types::no_buffer());
                let view = graph.add_place_view(element, view_ty, span).value();
                pending.push(view);
                view
            } else {
                let (loaded, effect) = detached_load(graph, element, ty, effect_ids, span);
                loads.push(effect);
                loaded
            };
            graph.replace_value_references(consumer, replacement);
            graph.install_aliases([(consumer, replacement)]);
        }
    }
    graph.skeleton.blocks[block].side_effects.splice(0..0, loads);
}

pub fn pack_result_values<P: Family>(
    graph: &mut EGraph<P>,
    binding: &ResultBinding<Type<TypeName>>,
) -> Result<ValueId, String> {
    if binding.destination_count() != binding.values().len() {
        return Err("place-backed result requires an explicit load before value materialization".into());
    }

    if !binding.is_product() {
        return binding
            .single_value()
            .map(|value| graph.canonical_value(value))
            .ok_or_else(|| "by-value result leaf has no value".into());
    }

    let fields = binding
        .top_level_fields()
        .iter()
        .map(|field| pack_result_values(graph, field))
        .collect::<Result<SmallVec<[ValueId; 4]>, _>>()?;
    let result = graph.intern_pure(PureOp::Tuple(fields.len()), fields, binding.ty().clone(), None);
    register_result_origin_tree(graph, result, binding);
    Ok(result)
}

/// Produce the value-channel handle used when a completed logical result is
/// passed to another operation. A destination-backed aggregate remains a
/// view of its place; only an entirely by-value product is packed.
pub(crate) fn result_argument_value<P: Family>(
    graph: &mut EGraph<P>,
    binding: &ResultBinding<Type<TypeName>>,
) -> Result<ValueId, String> {
    if binding.is_product() {
        return pack_result_values(graph, binding);
    }
    let (ty, destination) = binding
        .single_destination()
        .ok_or_else(|| "result argument has no physical destination".to_owned())?;
    match destination {
        ResultDestination::ReturnValue(value) => Ok(graph.canonical_value(*value)),
        ResultDestination::Place(PlaceDestination::Fixed(place)) => {
            let view_ty = crate::types::view_array_of(ty, crate::types::no_buffer());
            Ok(graph.add_place_view(*place, view_ty, None).value())
        }
        ResultDestination::Place(PlaceDestination::Bounded { .. }) => {
            Err("bounded result arguments require a length-carrying view".to_owned())
        }
    }
}

/// Explicitly load a complete logical result into the value channel. This is
/// reserved for boundaries whose source-language type is itself passed by
/// value, such as a fixed aggregate nested in a reduction accumulator.
pub(crate) fn load_result_value<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    binding: &ResultBinding<Type<TypeName>>,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> ValueId {
    if binding.is_product() {
        let fields = binding
            .top_level_fields()
            .iter()
            .map(|field| load_result_value(graph, block, field, effect_ids))
            .collect::<SmallVec<[ValueId; 4]>>();
        let value = graph.intern_pure(PureOp::Tuple(fields.len()), fields, binding.ty().clone(), None);
        register_result_origin_tree(graph, value, binding);
        return value;
    }
    let (ty, destination) =
        binding.single_destination().expect("a result leaf has one physical destination");
    match destination {
        ResultDestination::ReturnValue(value) => graph.canonical_value(*value),
        ResultDestination::Place(PlaceDestination::Fixed(place))
        | ResultDestination::Place(PlaceDestination::Bounded { storage: place, .. }) => {
            emit_load(graph, block, *place, ty.clone(), effect_ids, None)
        }
    }
}

pub fn register_result_origin_tree<P: Family>(
    graph: &mut EGraph<P>,
    value: ValueId,
    binding: &ResultBinding<Type<TypeName>>,
) {
    graph.register_result_origin(value, binding.clone());
    if !binding.is_product() {
        return;
    }
    for (index, field) in binding.top_level_fields().into_iter().enumerate() {
        let projection = graph.intern_pure(
            PureOp::Project { index: index as u32 },
            smallvec![value],
            field.ty().clone(),
            None,
        );
        register_result_origin_tree(graph, projection, &field);
    }
}

pub fn rebind_result_value_references<P: Family>(
    graph: &mut EGraph<P>,
    old: &ResultBinding<Type<TypeName>>,
    new: &ResultBinding<Type<TypeName>>,
) -> Result<Vec<(ValueId, ValueId)>, String> {
    if old.ty() != new.ty() {
        return Err(format!(
            "cannot rebind result values with different types: {:?} and {:?}",
            old.ty(),
            new.ty()
        ));
    }
    let old_values = result_leaf_values(graph, old)?;
    let new_values = result_leaf_values(graph, new)?;
    if old_values.len() != new_values.len() {
        return Err("result fields have different by-value leaf counts".into());
    }
    let replacements = old_values.into_iter().zip(new_values).collect::<Vec<_>>();
    for (old, new) in &replacements {
        graph.replace_value_references(*old, *new);
    }
    graph.install_aliases(replacements.iter().copied().filter(|(old, new)| old != new));
    Ok(replacements)
}

pub(crate) fn bind_result_to_view<P: Family, R: Clone, D: Clone>(
    graph: &mut EGraph<P>,
    result: &ResultTree<Type<TypeName>, R, D>,
    view: ValueId,
) -> Result<ResultBinding<Type<TypeName>>, String> {
    let leaves = result.destination_leaves();
    if leaves.len() == 1 {
        return Ok(result.map_destinations(|_, _| ResultDestination::ReturnValue(view)));
    }
    let parent_ty = graph.nodes[view].ty().clone();
    let parent_elem = crate::types::array_elem(&parent_ty)
        .ok_or_else(|| "structured result destination is not an array view".to_owned())?;
    let parent_elem_bytes = crate::ssa::layout::type_byte_size(parent_elem)
        .ok_or_else(|| "structured result destination has no physical element size".to_owned())?;
    let region = parent_ty
        .array_buffer()
        .cloned()
        .ok_or_else(|| "structured result destination has no storage region".to_owned())?;
    let mut offset_bytes = 0u32;
    let mut views = Vec::with_capacity(leaves.len());
    for leaf in &leaves {
        let bytes = crate::ssa::layout::type_byte_size(leaf.ty())
            .ok_or_else(|| "structured result component has no fixed physical size".to_owned())?;
        if offset_bytes % parent_elem_bytes != 0 {
            return Err("structured result component is not aligned to its storage element".into());
        }
        let Type::Constructed(TypeName::Size(length), _) = leaf
            .ty()
            .array_size()
            .ok_or_else(|| "structured result component is not an array".to_owned())?
        else {
            return Err("structured result component has no fixed length".into());
        };
        let offset = intern_u32(graph, offset_bytes / parent_elem_bytes, None);
        let length = intern_u32(
            graph,
            u32::try_from(*length).map_err(|_| "structured result length exceeds u32")?,
            None,
        );
        let view_ty = crate::types::view_array_of(leaf.ty(), region.clone());
        views.push(intern_inherited_view(graph, view, offset, length, view_ty, None));
        offset_bytes = offset_bytes
            .checked_add(bytes)
            .ok_or_else(|| "structured result offsets overflow u32".to_owned())?;
    }
    let mut views = views.into_iter();
    Ok(result.map_destinations(|_, _| {
        ResultDestination::ReturnValue(views.next().expect("one view was built for every result leaf"))
    }))
}

fn result_leaf_values<P: Family>(
    graph: &mut EGraph<P>,
    result: &ResultBinding<Type<TypeName>>,
) -> Result<Vec<ValueId>, String> {
    result
        .destination_leaves()
        .into_iter()
        .map(|leaf| {
            let (ty, destination) = leaf
                .single_destination()
                .ok_or_else(|| "result leaf has no physical destination".to_owned())?;
            Ok(match destination {
                ResultDestination::ReturnValue(value) => graph.canonical_value(*value),
                ResultDestination::Place(PlaceDestination::Fixed(place))
                | ResultDestination::Place(PlaceDestination::Bounded { storage: place, .. }) => {
                    let view_ty = crate::types::view_array_of(ty, crate::types::no_buffer());
                    graph.add_place_view(*place, view_ty, None).value()
                }
            })
        })
        .collect()
}

pub fn rebind_result_projection_references<P: Family>(
    graph: &mut EGraph<P>,
    source: ValueId,
    result: &ResultBinding<Type<TypeName>>,
) -> Result<(), String> {
    register_result_origin_tree(graph, source, result);
    if result.places().is_empty() {
        let replacement = pack_result_values(graph, result)?;
        if source != replacement {
            graph.replace_value_references(source, replacement);
            graph.install_aliases([(source, replacement)]);
        }
        return Ok(());
    }
    if result.is_product() {
        for (index, field) in result.top_level_fields().into_iter().enumerate() {
            let projection = graph.intern_pure(
                PureOp::Project { index: index as u32 },
                smallvec![source],
                field.ty().clone(),
                None,
            );
            rebind_result_projection_references(graph, projection, &field)?;
        }
        return Ok(());
    }

    let replacement = result_leaf_values(graph, result)?[0];
    if source != replacement {
        graph.replace_value_references(source, replacement);
        graph.install_aliases([(source, replacement)]);
    }
    Ok(())
}

/// Phase-specific SOAC metadata that contributes to a produced value.
///
/// Raw SOACs have captures and operator seeds but no resolved segmented
/// iteration space. Semantic SOACs additionally expose their resolved space.
pub(crate) trait ValueProducerPhase: Family {
    fn effect_metadata_inputs(effect: &SideEffect<Self>) -> Vec<ValueId>;

    fn effect_value_inputs(graph: &EGraph<Self>, effect: &SideEffect<Self>) -> Vec<ValueId> {
        let mut values = graph.effect_boundary_value_dependencies(effect);
        values.extend(Self::effect_metadata_inputs(effect));
        values
    }
}

impl<R: GraphResource> ValueProducerPhase for Raw<R> {
    fn effect_metadata_inputs(effect: &SideEffect<Self>) -> Vec<ValueId> {
        let mut nodes = Vec::new();
        let SideEffectKind::Soac(SoacEffect(_, soac)) = &effect.kind else {
            return nodes;
        };
        nodes.extend(soac.seg_bodies().into_iter().flat_map(SegBody::capture_values));
        if let Soac::Screma(op) = soac {
            nodes.extend(op.form.scans.iter().flat_map(|scan| scan.neutral.iter().copied()));
            nodes.extend(op.form.reductions.iter().flat_map(|reduction| reduction.neutral.iter().copied()));
        }
        nodes
    }
}

impl<R: GraphResource> ValueProducerPhase for Semantic<R> {
    fn effect_metadata_inputs(effect: &SideEffect<Self>) -> Vec<ValueId> {
        effect.semantic_metadata_inputs()
    }
}

pub(crate) fn effect_value_inputs<P: ValueProducerPhase>(
    graph: &EGraph<P>,
    effect: &SideEffect<P>,
) -> Vec<ValueId> {
    P::effect_value_inputs(graph, effect)
}

/// The complete value-producing closure behind one or more EGIR values.
///
/// `ValueKind::children` covers floating pure expressions, but intentionally has
/// no edges for effect results or block parameters.  Analyses that need the
/// actual producer must also follow an effect result to its anchored effect and
/// a block parameter to every incoming CFG argument.  Keeping both visited
/// sets makes loop-carried values finite even though those additional edges can
/// form cycles.
#[derive(Debug, Default)]
pub(crate) struct ValueProducerClosure {
    pub(crate) nodes: HashSet<ValueId>,
    pub(crate) effects: HashSet<SideEffectSite>,
}

impl ValueProducerClosure {
    pub(crate) fn contains_node(&self, node: ValueId) -> bool {
        self.nodes.contains(&node)
    }
}

/// Executable graph locations whose values depend on a source value.
///
/// Locations are stable only for the graph snapshot used to build the
/// corresponding [`ValueUseIndex`].
#[derive(Debug, Default)]
pub(crate) struct ValueObservers {
    effects: HashSet<SideEffectSite>,
    terminators: HashSet<BlockId>,
}

impl ValueObservers {
    pub(crate) fn effect_sites(&self) -> impl Iterator<Item = SideEffectSite> + '_ {
        self.effects.iter().copied()
    }

    pub(crate) fn terminator_blocks(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.terminators.iter().copied()
    }
}

/// Reverse value-flow and executable-use index for one immutable graph
/// snapshot.
///
/// Pure successors follow only floating pure/union operands. Value successors
/// additionally cross side-effect results and CFG block arguments, mirroring
/// [`value_producer_closure`] in the opposite direction. This lets passes ask
/// centralized observer and liveness questions instead of repeatedly scanning
/// every effect and terminator with a producer-reachability query.
///
/// Rebuild the index after inserting, removing, reordering, or rewriting graph
/// structure. In particular, the [`SideEffectSite`] values it returns must not
/// survive a skeleton mutation.
pub(crate) struct ValueUseIndex {
    pure_successors: LookupMap<ValueId, Vec<ValueId>>,
    value_successors: LookupMap<ValueId, Vec<ValueId>>,
    effect_observers: LookupMap<ValueId, Vec<SideEffectSite>>,
    terminator_observers: LookupMap<ValueId, Vec<BlockId>>,
}

impl ValueUseIndex {
    pub(crate) fn build<P: ValueProducerPhase>(graph: &EGraph<P>) -> Self {
        let mut index = Self {
            pure_successors: LookupMap::new(),
            value_successors: LookupMap::new(),
            effect_observers: LookupMap::new(),
            terminator_observers: LookupMap::new(),
        };

        for (user, definition) in &graph.nodes {
            for source in definition.kind.children() {
                index.pure_successors.entry(source).or_default().push(user);
                index.value_successors.entry(source).or_default().push(user);
            }
        }

        for (block, body) in &graph.skeleton.blocks {
            for (effect_index, effect) in body.side_effects.iter().enumerate() {
                let site = SideEffectSite {
                    block,
                    index: effect_index,
                };
                for source in P::effect_value_inputs(graph, effect) {
                    index.effect_observers.entry(source).or_default().push(site);
                    if let Some(result) = graph.effect_result_binding(effect) {
                        for result in result.values() {
                            index.value_successors.entry(source).or_default().push(result);
                        }
                    }
                }
            }
            for source in body.term.referenced_nodes() {
                index.terminator_observers.entry(source).or_default().push(block);
            }
            index_block_argument_successors(graph, &mut index.value_successors, &body.term);
        }

        index
    }

    /// Effects and terminators reached through floating pure/union users.
    pub(crate) fn pure_observers(&self, source: ValueId) -> ValueObservers {
        self.observers(source, &self.pure_successors)
    }

    /// Effects and terminators reached through complete value flow, including
    /// effect results and incoming CFG block arguments.
    pub(crate) fn value_observers(&self, source: ValueId) -> ValueObservers {
        self.observers(source, &self.value_successors)
    }

    /// Whether `user` consumes `source` through floating pure/union nodes.
    pub(crate) fn pure_reaches(&self, source: ValueId, user: ValueId) -> bool {
        self.reaches(source, user, &self.pure_successors)
    }

    fn observers(&self, source: ValueId, successors: &LookupMap<ValueId, Vec<ValueId>>) -> ValueObservers {
        let mut observers = ValueObservers::default();
        self.walk_users(source, successors, |user| {
            observers.effects.extend(self.effect_observers.get(&user).into_iter().flatten().copied());
            observers
                .terminators
                .extend(self.terminator_observers.get(&user).into_iter().flatten().copied());
            false
        });
        observers
    }

    fn reaches(
        &self,
        source: ValueId,
        target: ValueId,
        successors: &LookupMap<ValueId, Vec<ValueId>>,
    ) -> bool {
        self.walk_users(source, successors, |user| user == target)
    }

    fn walk_users(
        &self,
        source: ValueId,
        successors: &LookupMap<ValueId, Vec<ValueId>>,
        mut visit: impl FnMut(ValueId) -> bool,
    ) -> bool {
        let mut seen = HashSet::new();
        let mut pending = vec![source];
        while let Some(user) = pending.pop() {
            if !seen.insert(user) {
                continue;
            }
            if visit(user) {
                return true;
            }
            pending.extend(successors.get(&user).into_iter().flatten().copied());
        }
        false
    }
}

fn index_block_argument_successors<P: Family>(
    graph: &EGraph<P>,
    successors: &mut LookupMap<ValueId, Vec<ValueId>>,
    term: &SkeletonTerminator,
) {
    let mut add_edge = |target: BlockId, args: &[super::types::FlowValueId], condition: Option<ValueId>| {
        let Some(target_block) = graph.skeleton.blocks.get(target) else {
            return;
        };
        for (&argument, &parameter) in args.iter().zip(&target_block.params) {
            successors.entry(argument.value()).or_default().push(parameter.value());
            if let Some(condition) = condition {
                successors.entry(condition).or_default().push(parameter.value());
            }
        }
    };
    match term {
        SkeletonTerminator::Branch { target, args } => add_edge(*target, args, None),
        SkeletonTerminator::CondBranch {
            cond,
            then_target,
            then_args,
            else_target,
            else_args,
        } => {
            add_edge(*then_target, then_args, Some(*cond));
            add_edge(*else_target, else_args, Some(*cond));
        }
        SkeletonTerminator::Return(_) | SkeletonTerminator::Unreachable => {}
    }
}

/// Follow pure tails, value-producing effects, and CFG block arguments to the
/// values that can contribute to `roots`.
pub(crate) fn value_producer_closure<P: ValueProducerPhase>(
    graph: &EGraph<P>,
    roots: impl IntoIterator<Item = ValueId>,
) -> ValueProducerClosure {
    let producer_index = graph.side_effect_index();
    let mut closure = ValueProducerClosure::default();
    let mut pending = roots.into_iter().collect::<Vec<_>>();

    while let Some(node) = pending.pop() {
        if !closure.nodes.insert(node) {
            continue;
        }
        let Some(definition) = graph.nodes.get(node) else {
            continue;
        };
        if let Some(alias) = definition.alias {
            pending.push(alias);
            continue;
        }
        if let Some(field) = projected_tuple_field(graph, node) {
            pending.push(field);
            continue;
        }
        match &definition.kind {
            ValueKind::Pure { operands, .. } => pending.extend(operands.iter().copied()),
            ValueKind::Union { left, right } => pending.extend([*left, *right]),
            ValueKind::BlockParam { block, index } => {
                extend_incoming_block_args(graph, *block, *index, &mut pending);
            }
            ValueKind::SideEffectResult => {
                let Some(site) = producer_index.site(node) else {
                    continue;
                };
                if closure.effects.insert(site) {
                    pending.extend(P::effect_value_inputs(graph, graph.skeleton.effect(site)));
                }
            }
            ValueKind::CallResult { call, .. } => {
                pending
                    .extend(graph.call(*call).arguments().iter().filter_map(|argument| argument.value()));
            }
            ValueKind::PlaceLength { place } | ValueKind::PlaceView { place } => {
                pending.extend(graph.place_value_dependencies(*place));
            }
            ValueKind::FuncParam { .. } | ValueKind::Constant(_) => {}
        }
    }

    closure
}

/// Return the selected field when a projection is applied directly to a
/// structural tuple. Pure value-flow consumers need only that field.
pub(crate) fn projected_tuple_field<P: Family>(graph: &EGraph<P>, node: ValueId) -> Option<ValueId> {
    let ValueKind::Pure {
        op: PureOp::Project { index },
        operands,
    } = graph.nodes.get(node)?.kind()
    else {
        return None;
    };
    let [tuple] = operands.as_slice() else {
        return None;
    };
    let ValueKind::Pure {
        op: PureOp::Tuple(arity),
        operands: fields,
    } = graph.nodes.get(*tuple)?.kind()
    else {
        return None;
    };
    (*arity == fields.len()).then(|| fields.get(*index as usize).copied()).flatten()
}

/// Follow every value used by executable graph structure, together with
/// caller-supplied result roots. This is the common reachability boundary for
/// analyses of a projected recipe: block effects and terminators are executed,
/// while projection-preserved but unused metadata is not.
pub(crate) fn execution_value_producer_closure<P: ValueProducerPhase>(
    graph: &EGraph<P>,
    result_roots: impl IntoIterator<Item = ValueId>,
) -> ValueProducerClosure {
    value_producer_closure(
        graph,
        execution_value_roots(graph).into_iter().chain(result_roots),
    )
}

/// Values referenced directly by executable effects and terminators.
///
/// The phase adapter includes SOAC captures and other producer metadata that
/// the phase-agnostic IR cannot see through `P::Soac`.
pub(crate) fn execution_value_roots<P: ValueProducerPhase>(graph: &EGraph<P>) -> Vec<ValueId> {
    graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, block)| {
            block
                .side_effects
                .iter()
                .flat_map(|effect| P::effect_value_inputs(graph, effect))
                .chain(block.term.referenced_nodes())
        })
        .collect()
}

/// Pure-graph reachability from every executable effect and terminator.
pub(crate) fn reachable_execution_values<P: ValueProducerPhase>(graph: &EGraph<P>) -> Vec<ValueId> {
    reachable_execution_values_with_roots(graph, std::iter::empty())
}

/// Pure-graph reachability from executable graph structure and values rooted
/// by metadata owned outside the graph, such as entry output routes.
pub(crate) fn reachable_execution_values_with_roots<P: ValueProducerPhase>(
    graph: &EGraph<P>,
    roots: impl IntoIterator<Item = ValueId>,
) -> Vec<ValueId> {
    wyn_graph::reachable_from_ordered(
        execution_value_roots(graph).into_iter().chain(roots),
        wyn_graph::WalkOrder::DepthFirst,
        |node, out| {
            if let Some(definition) = graph.nodes.get(node) {
                out.extend(definition.kind.children());
            }
        },
    )
}

/// Whether `target` is reachable from `root` through floating pure/union
/// operands only. This is the common dependency predicate for use-site and
/// fusion analyses that deliberately stop at effect results and CFG params.
pub(crate) fn pure_depends_on<P: Family>(graph: &EGraph<P>, root: ValueId, target: ValueId) -> bool {
    wyn_graph::reaches_ordered(root, target, wyn_graph::WalkOrder::DepthFirst, |node, out| {
        if let Some(definition) = graph.nodes.get(node) {
            out.extend(definition.kind.children());
        }
    })
}

/// Whether the complete value-producing closure behind `root` contains
/// `target`, crossing effect results and incoming block arguments as needed.
pub(crate) fn value_depends_on<P: ValueProducerPhase>(
    graph: &EGraph<P>,
    root: ValueId,
    target: ValueId,
) -> bool {
    value_producer_closure(graph, [root]).contains_node(target)
}

/// Maximal movable values at the boundary of executable graph structure.
///
/// A value belongs to the frontier when it is movable and is either used
/// directly by an effect/terminator or consumed by a non-movable value. The
/// predicate owns the meaning of "movable" (loop invariant, stage invariant,
/// cloneable, and so on), while this helper owns the shared graph boundary
/// calculation.
pub(crate) fn maximal_execution_frontier<P: ValueProducerPhase>(
    graph: &EGraph<P>,
    mut movable: impl FnMut(ValueId) -> bool,
) -> Vec<ValueId> {
    let reachable = reachable_execution_values(graph);
    let reachable_set = reachable.iter().copied().collect::<HashSet<_>>();
    let movable = reachable.iter().map(|node| (*node, movable(*node))).collect::<HashMap<_, _>>();
    let mut boundary = execution_value_roots(graph).into_iter().collect::<HashSet<_>>();
    for node in &reachable {
        if movable[node] {
            continue;
        }
        if let Some(definition) = graph.nodes.get(*node) {
            boundary.extend(
                definition.kind.children().into_iter().filter(|child| reachable_set.contains(child)),
            );
        }
    }
    let mut frontier =
        reachable.into_iter().filter(|node| boundary.contains(node) && movable[node]).collect::<Vec<_>>();
    frontier.sort_unstable();
    frontier.dedup();
    frontier
}

/// Storage resources read by the complete producer closure behind `roots`.
pub(crate) fn read_storage_resources<P>(
    graph: &EGraph<P>,
    roots: impl IntoIterator<Item = ValueId>,
) -> Vec<SegResourceAccess<super::program::SemanticResourceRef>>
where
    P: ValueProducerPhase + Family<Resource = super::program::SemanticResourceRef>,
{
    let resources = value_producer_closure(graph, roots)
        .nodes
        .into_iter()
        .filter_map(|node| extract_storage_view_source(graph, node))
        .collect::<HashSet<_>>();
    let mut resources = resources
        .into_iter()
        .map(|resource| SegResourceAccess {
            resource,
            access: ResourceAccess::Read,
        })
        .collect::<Vec<_>>();
    resources.sort_by_key(|resource| resource.resource);
    resources
}

/// Index effectful SOAC writers by the logical resources named by their
/// explicit destination views.
pub(crate) fn resource_effect_writers<P>(
    graph: &EGraph<P>,
) -> crate::LookupMap<super::program::SemanticResourceRef, Vec<EffectToken>>
where
    P: ValueProducerPhase
        + super::types::WynSoacPhase
        + Family<Resource = super::program::SemanticResourceRef>,
{
    let mut writers = crate::LookupMap::<_, Vec<_>>::new();
    for block in graph.skeleton.blocks.values() {
        for effect in &block.side_effects {
            let SideEffectKind::Soac(super::types::SoacEffect(_, soac)) = &effect.kind else {
                continue;
            };
            let Some((_, effect_out)) = effect.effects() else {
                continue;
            };
            for access in read_storage_resources(graph, soac.written_views().map(ViewId::value)) {
                writers.entry(access.resource).or_default().push(effect_out);
            }
        }
    }
    writers
}

/// Return the output selected by a direct projection of `root`.
pub(crate) fn projection_index<P: Family>(
    graph: &EGraph<P>,
    node: ValueId,
    root: ValueId,
) -> Option<usize> {
    let node = graph.canonical_value(node);
    let root = graph.canonical_value(root);
    if node == root {
        return Some(0);
    }
    match &graph.nodes.get(node)?.kind {
        ValueKind::Pure {
            op: PureOp::Project { index },
            operands,
        } if operands.first().is_some_and(|operand| graph.canonical_value(*operand) == root) => {
            Some(*index as usize)
        }
        _ => None,
    }
}

fn extend_incoming_block_args<P: Family>(
    graph: &EGraph<P>,
    target: BlockId,
    index: usize,
    pending: &mut Vec<ValueId>,
) {
    for (_, predecessor) in &graph.skeleton.blocks {
        match &predecessor.term {
            SkeletonTerminator::Branch {
                target: branch_target,
                args,
            } if *branch_target == target => {
                pending.extend(args.get(index).map(|argument| argument.value()));
            }
            SkeletonTerminator::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
                ..
            } => {
                let mut reaches_target = false;
                if *then_target == target {
                    pending.extend(then_args.get(index).map(|argument| argument.value()));
                    reaches_target = true;
                }
                if *else_target == target {
                    pending.extend(else_args.get(index).map(|argument| argument.value()));
                    reaches_target = true;
                }
                if reaches_target {
                    pending.push(*cond);
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Pure ops
// ---------------------------------------------------------------------------

/// `u32` literal — the helper most code reaches for. Same canonical
/// shape (`PureOp::Uint(n.to_string())`) as `from_tlc` produces from
/// `TermKind::IntLit` so hash-consing deduplicates across the two
/// emission paths.
pub fn intern_u32<P: Family>(graph: &mut EGraph<P>, n: u32, span: Option<Span>) -> ValueId {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    graph.intern_pure(PureOp::Uint(n.to_string()), smallvec![], u32_ty, span)
}

/// Constant via `EGraph::intern_constant` (canonical `ValueKind::Constant`
/// form). Use this when the value comes through a `ConstantValue`
/// already (e.g. carrying a reduce's neutral element across passes).
/// For freshly-typed-out integer/float literals from terms, prefer the
/// `PureOp::Uint`/`Int`/`Float` form via the other helpers.
pub fn intern_constant<P: Family>(
    graph: &mut EGraph<P>,
    value: ConstantValue,
    ty: Type<TypeName>,
) -> ValueId {
    graph.intern_constant(value, ty)
}

/// Generic intrinsic call (`PureOp::Intrinsic` with `overload_idx: 0`).
pub fn intern_intrinsic<P: Family>(
    graph: &mut EGraph<P>,
    id: BuiltinId,
    operands: SmallVec<[ValueId; 4]>,
    ty: Type<TypeName>,
    span: Option<Span>,
) -> ValueId {
    graph.intern_pure(PureOp::Intrinsic { id, overload_idx: 0 }, operands, ty, span)
}

/// Binary op (`PureOp::BinOp`). `op` is the operator string (`"+"`,
/// `"-"`, etc.) — matches the convention `from_tlc` uses.
pub fn intern_binop<P: Family>(
    graph: &mut EGraph<P>,
    op: crate::op::BinaryOperator,
    lhs: ValueId,
    rhs: ValueId,
    ty: Type<TypeName>,
    span: Option<Span>,
) -> ValueId {
    graph.intern_pure(PureOp::BinOp(op), smallvec![lhs, rhs], ty, span)
}

/// `StorageView(Storage(br))` with the default
/// `[0, _w_intrinsic_storage_len(set, binding)]` operand pair.
pub fn intern_storage_view(
    graph: &mut EGraph<Physical>,
    br: BindingRef,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> ValueId {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let set_nid = intern_u32(graph, br.set, span);
    let binding_nid = intern_u32(graph, br.binding, span);
    let storage_len_id = catalog().known().storage_len;
    let len_nid = intern_intrinsic(
        graph,
        storage_len_id,
        smallvec![set_nid, binding_nid],
        u32_ty,
        span,
    );
    let zero_nid = intern_u32(graph, 0, span);
    let view_ty = crate::types::view_array_of(&view_ty, crate::types::buffer_tag(br));
    graph.intern_pure(
        PureOp::StorageView(PureViewSource::Storage(br)),
        smallvec![zero_nid, len_nid],
        view_ty,
        span,
    )
}

/// Target-independent storage view used after logical-resource allocation.
pub fn intern_resource_view<P: Family<Resource = super::program::SemanticResourceRef>>(
    graph: &mut EGraph<P>,
    resource: crate::ResourceId,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> ValueId {
    let len = intern_resource_len(graph, resource, span);
    let zero = intern_u32(graph, 0, span);
    intern_chunked_resource_view(graph, resource, zero, len, view_ty, span)
}

/// Target-independent logical-resource length.
pub fn intern_resource_len<P: Family<Resource = super::program::SemanticResourceRef>>(
    graph: &mut EGraph<P>,
    resource: crate::ResourceId,
    span: Option<Span>,
) -> ValueId {
    graph.intern_pure(
        PureOp::ResourceLen(super::program::SemanticResourceRef(resource)),
        smallvec![],
        Type::Constructed(TypeName::UInt(32), vec![]),
        span,
    )
}

pub fn intern_chunked_resource_view<P: Family<Resource = super::program::SemanticResourceRef>>(
    graph: &mut EGraph<P>,
    resource: crate::ResourceId,
    offset: ValueId,
    len: ValueId,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> ValueId {
    let view_ty =
        crate::types::view_array_of(&view_ty, Type::Constructed(TypeName::Resource(resource), vec![]));
    graph.intern_pure(
        PureOp::StorageView(PureViewSource::Storage(super::program::SemanticResourceRef(
            resource,
        ))),
        smallvec![offset, len],
        view_ty,
        span,
    )
}

/// Retarget every view of one logical resource while preserving its slice.
/// Returns the value substitutions needed by metadata stored outside the graph.
pub(crate) fn retarget_resource_views<P>(
    graph: &mut EGraph<P>,
    source: crate::ResourceId,
    destination: crate::ResourceId,
) -> Vec<(ValueId, ValueId)>
where
    P: Family<Resource = super::program::SemanticResourceRef>,
{
    if source == destination {
        return Vec::new();
    }
    let source = super::program::SemanticResourceRef(source);
    let lengths = graph
        .nodes
        .iter()
        .filter_map(|(value, definition)| {
            matches!(
                definition.kind(),
                ValueKind::Pure {
                    op: PureOp::ResourceLen(resource),
                    operands,
                } if *resource == source && operands.is_empty()
            )
            .then_some((value, definition.span()))
        })
        .collect::<Vec<_>>();
    let mut replacements = Vec::new();
    for (length, span) in lengths {
        let replacement = intern_resource_len(graph, destination, span);
        graph.replace_value_references(length, replacement);
        replacements.push((length, replacement));
    }
    let views = graph
        .nodes
        .iter()
        .filter_map(|(value, definition)| match definition.kind() {
            ValueKind::Pure {
                op: PureOp::StorageView(PureViewSource::Storage(resource)),
                operands,
            } if *resource == source && operands.len() == 2 => Some((
                value,
                operands[0],
                operands[1],
                definition.ty().clone(),
                definition.span(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    replacements.reserve(views.len());
    for (view, offset, length, ty, span) in views {
        let replacement = intern_chunked_resource_view(graph, destination, offset, length, ty, span);
        graph.replace_value_references(view, replacement);
        replacements.push((view, replacement));
    }
    replacements
}

/// Construct an addressable sub-view whose offset is relative to an existing
/// view. The parent is part of the node itself, so addressability survives
/// graph rewrites without recovering provenance from an intrinsic opcode.
pub fn intern_inherited_view<P: Family>(
    graph: &mut EGraph<P>,
    parent: ValueId,
    offset: ValueId,
    len: ValueId,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> ValueId {
    let parent = graph.canonical_value(parent);
    if let ValueKind::PlaceView { place } = graph.nodes[parent].kind() {
        let place = *place;
        let slice = graph.add_slice_place(place, offset, len, view_ty.clone(), span);
        return graph.add_place_view(slice, view_ty, span).value();
    }
    graph.intern_pure(
        PureOp::StorageView(PureViewSource::Inherited),
        smallvec![offset, len, parent],
        view_ty,
        span,
    )
}

/// A workgroup-shared array view: `StorageView(Workgroup{id, count})` with
/// `[offset=0, len=count]`. `view_ty` is the array type `[count]elem`; the
/// backends recover the element type from it to declare a module-scope
/// `array<elem, count>` in workgroup storage. Indexed with the same
/// `ViewIndex` + `Load`/`Store` machinery as storage views.
pub fn emit_workgroup_view<P: Family>(
    graph: &mut EGraph<P>,
    id: u32,
    count: u32,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> ValueId {
    let zero_nid = intern_u32(graph, 0, span);
    let count_nid = intern_u32(graph, count, span);
    // Workgroup-shared memory is not descriptor-bound: no (set, binding) region.
    let view_ty = crate::types::view_array_of(&view_ty, crate::types::no_buffer());
    graph.intern_pure(
        PureOp::StorageView(PureViewSource::Workgroup { id, count }),
        smallvec![zero_nid, count_nid],
        view_ty,
        span,
    )
}

/// `StorageView(Storage(br))` with caller-supplied `offset` and `len`.
/// Builds a chunked sub-view of a larger storage buffer (phase1 of
/// parallel reduce/scan).
pub fn intern_chunked_storage_view(
    graph: &mut EGraph<Physical>,
    br: BindingRef,
    offset: ValueId,
    len: ValueId,
    view_ty: Type<TypeName>,
    span: Option<Span>,
) -> ValueId {
    let view_ty = crate::types::view_array_of(&view_ty, crate::types::buffer_tag(br));
    graph.intern_pure(
        PureOp::StorageView(PureViewSource::Storage(br)),
        smallvec![offset, len],
        view_ty,
        span,
    )
}

// ---------------------------------------------------------------------------
// Side effects
// ---------------------------------------------------------------------------

pub fn alloc_effect(effect_ids: &mut crate::IdSource<EffectToken>) -> EffectToken {
    effect_ids.next_id()
}

/// Emit a `Store` side-effect in `block`. `place_nid` must be a place-
/// producing pure op (`ViewIndex`, `OutputSlot`). Returns the produced
/// effect-out token.
pub fn emit_store<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    place: PlaceId,
    value_nid: ValueId,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> EffectToken {
    let effect_in = alloc_effect(effect_ids);
    let effect_out = alloc_effect(effect_ids);
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Store { place }),
        operands: smallvec![OperandRef::Value(value_nid)],
        result: None,
        effects: Some((effect_in, effect_out)),
        span,
    });
    effect_out
}

/// Construct a `Store` without choosing its position in a skeleton block.
pub fn detached_store<P: Family>(
    place: PlaceId,
    value: ValueId,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> SideEffect<P> {
    SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Store { place }),
        operands: smallvec![OperandRef::Value(value)],
        result: None,
        effects: Some((alloc_effect(effect_ids), alloc_effect(effect_ids))),
        span,
    }
}

/// Emit an atomic integer update through an addressable place. The returned
/// node is the value observed before the update.
pub fn emit_atomic<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    place: PlaceId,
    op: crate::ssa::types::AtomicOp,
    values: &[ValueId],
    result_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> ValueId {
    assert_eq!(values.len(), op.value_arity());
    let effect_in = alloc_effect(effect_ids);
    let effect_out = alloc_effect(effect_ids);
    let result = graph.alloc_side_effect_result(result_ty);
    let operands = values.iter().copied().map(OperandRef::Value).collect();
    let result_binding = graph.value_result(result);
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Atomic { place, op }),
        operands,
        result: Some(result_binding),
        effects: Some((effect_in, effect_out)),
        span,
    });
    result
}
/// Emit a workgroup execution+memory barrier
/// in `block`. No operands or result; the effect token keeps it ordered
/// against the workgroup-shared loads/stores it synchronizes. Returns the
/// produced effect-out token.
pub fn emit_workgroup_barrier<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> EffectToken {
    let effect_in = alloc_effect(effect_ids);
    let effect_out = alloc_effect(effect_ids);
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::ControlBarrier),
        operands: smallvec![],
        result: None,
        effects: Some((effect_in, effect_out)),
        span: None,
    });
    effect_out
}

/// Emit a store through a `StorageView` at `index_nid`. Builds the
/// `ViewIndex` pure node and the `Store` side-effect.
pub fn emit_storage_store<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    view_nid: ValueId,
    index_nid: ValueId,
    value_nid: ValueId,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> EffectToken {
    let view = graph.view_id(view_nid);
    let place = graph.add_view_index_place(view, index_nid, elem_ty, span);
    emit_store(graph, block, place, value_nid, effect_ids, span)
}

/// Write one logical result value to a resource-backed destination. Fixed
/// arrays are copied elementwise; scalar and aggregate storage elements use
/// slot zero. Runtime-sized arrays must already be destination-backed.
pub fn emit_resource_write<P: Family<Resource = super::program::SemanticResourceRef>>(
    graph: &mut EGraph<P>,
    block: BlockId,
    resource: crate::ResourceId,
    value: ValueId,
    ty: &Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> Result<Vec<EffectToken>, String> {
    let view = intern_resource_view(graph, resource, ty.clone(), span);
    if let Some(Type::Constructed(TypeName::Size(length), _)) = ty.array_size() {
        let elem_ty =
            ty.elem_type().cloned().ok_or_else(|| "resource result is not an array".to_owned())?;
        let source_is_view = matches!(graph.operand_ref(value), OperandRef::View(_));
        let mut writers = Vec::with_capacity(*length);
        for index in 0..*length {
            let coordinate = intern_u32(
                graph,
                u32::try_from(index).map_err(|_| "resource result index exceeds u32")?,
                span,
            );
            let element = if source_is_view {
                emit_view_load(graph, block, value, coordinate, elem_ty.clone(), effect_ids, span)
            } else {
                graph.intern_pure(
                    PureOp::Project { index: index as u32 },
                    smallvec![value],
                    elem_ty.clone(),
                    span,
                )
            };
            writers.push(emit_storage_store(
                graph,
                block,
                view,
                coordinate,
                element,
                elem_ty.clone(),
                effect_ids,
                span,
            ));
        }
        Ok(writers)
    } else if ty.array_size().is_some() {
        Err(
            "runtime-sized result has no destination-backed writer; wrap its producer in a map so it can write the destination elementwise"
                .to_owned(),
        )
    } else {
        let zero = intern_u32(graph, 0, span);
        Ok(vec![emit_storage_store(
            graph,
            block,
            view,
            zero,
            value,
            ty.clone(),
            effect_ids,
            span,
        )])
    }
}

/// Emit a typed `Load` from an addressable place in `block`.
pub fn emit_load<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    place: PlaceId,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> ValueId {
    let (result, effect) = detached_load(graph, place, elem_ty, effect_ids, span);
    graph.skeleton.blocks[block].side_effects.push(effect);
    result
}

#[derive(Clone, Copy)]
enum AddressableSource {
    Place(PlaceId),
    View(super::types::ViewId),
}

/// Write a logical result tree into one addressable destination. Product
/// nodes are projected structurally, and addressable array leaves are copied
/// element-by-element without constructing an aggregate value.
pub fn emit_result_to_place<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    result: &ResultBinding<Type<TypeName>>,
    destination: PlaceId,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> Result<BlockId, String> {
    if result.is_product() {
        let mut tail = block;
        for (index, field) in result.top_level_fields().into_iter().enumerate() {
            let coordinate = graph.intern_pure(
                PureOp::Int(index.to_string()),
                smallvec![],
                Type::Constructed(TypeName::Int(32), vec![]),
                span,
            );
            let field_place = graph.add_index_place(destination, coordinate, field.ty().clone(), span);
            tail = emit_result_to_place(graph, tail, &field, field_place, effect_ids, span)?;
        }
        return Ok(tail);
    }

    let (ty, source) =
        result.single_destination().ok_or_else(|| "result leaf has no physical destination".to_owned())?;
    match source {
        ResultDestination::ReturnValue(value) => {
            let value = graph.canonical_value(*value);
            let array_update = match graph.nodes[value].kind() {
                ValueKind::Pure {
                    op: PureOp::Intrinsic { id, .. },
                    operands,
                } if (*id == catalog().known().array_with
                    || *id == catalog().known().array_with_in_place)
                    && operands.len() == 3 =>
                {
                    Some((operands[0], operands[1], operands[2]))
                }
                _ => None,
            };
            if let Some((base, index, element)) = array_update {
                let base = ResultBinding::destination(ty.clone(), ResultDestination::ReturnValue(base));
                let tail = emit_result_to_place(graph, block, &base, destination, effect_ids, span)?;
                let element_ty = ty
                    .elem_type()
                    .cloned()
                    .ok_or_else(|| "array update result is not an array".to_owned())?;
                emit_place_index_store(
                    graph,
                    tail,
                    destination,
                    index,
                    element,
                    element_ty,
                    effect_ids,
                    span,
                );
                let view_ty = crate::types::view_array_of(ty, crate::types::no_buffer());
                let replacement = graph.add_place_view(destination, view_ty, span).value();
                graph.replace_value_references(value, replacement);
                graph.install_aliases([(value, replacement)]);
                return Ok(tail);
            }
            let addressable = match graph.operand_ref(value) {
                OperandRef::Place(place) => Some(AddressableSource::Place(place)),
                OperandRef::View(view) => Some(AddressableSource::View(view)),
                OperandRef::Value(_) => None,
            };
            if let Some(source) = addressable {
                if matches!(source, AddressableSource::Place(place) if place == destination) {
                    Ok(block)
                } else {
                    emit_addressable_copy(graph, block, source, destination, ty, effect_ids, span)
                }
            } else {
                emit_store(graph, block, destination, value, effect_ids, span);
                Ok(block)
            }
        }
        ResultDestination::Place(PlaceDestination::Fixed(source)) => {
            if *source == destination {
                Ok(block)
            } else {
                emit_addressable_copy(
                    graph,
                    block,
                    AddressableSource::Place(*source),
                    destination,
                    ty,
                    effect_ids,
                    span,
                )
            }
        }
        ResultDestination::Place(PlaceDestination::Bounded { storage, .. }) => {
            if *storage == destination {
                Ok(block)
            } else {
                emit_addressable_copy(
                    graph,
                    block,
                    AddressableSource::Place(*storage),
                    destination,
                    ty,
                    effect_ids,
                    span,
                )
            }
        }
    }
}

pub fn rewrite_result_store_consumers<P: Family>(
    graph: &mut EGraph<P>,
    source: ValueId,
    result: &ResultBinding<Type<TypeName>>,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), String> {
    loop {
        let candidate = graph.skeleton.blocks.iter().find_map(|(block, contents)| {
            contents.side_effects.iter().enumerate().find_map(|(index, effect)| {
                let SideEffectKind::Effect(EffectOp::Store { place }) = effect.kind() else {
                    return None;
                };
                let [operand] = effect.operands() else {
                    return None;
                };
                (operand.value() == Some(source)).then_some((block, index, *place, effect.span()))
            })
        });
        let Some((block, index, destination, span)) = candidate else {
            return Ok(());
        };
        let continuation = graph.skeleton.split_block_before_effect(block, index);
        graph.skeleton.remove_effect_splicing_dependencies(SideEffectSite {
            block: continuation,
            index: 0,
        });
        let tail = emit_result_to_place(graph, block, result, destination, effect_ids, span)?;
        graph.skeleton.blocks[tail].term = SkeletonTerminator::Branch {
            target: continuation,
            args: Vec::new(),
        };
    }
}

fn emit_addressable_copy<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    source: AddressableSource,
    destination: PlaceId,
    ty: &Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> Result<BlockId, String> {
    if let Some(fields) = WynLanguage::product_fields(ty) {
        let mut tail = block;
        for (index, field_ty) in fields.iter().enumerate() {
            let coordinate = graph.intern_pure(
                PureOp::Int(index.to_string()),
                smallvec![],
                Type::Constructed(TypeName::Int(32), vec![]),
                span,
            );
            let source = index_addressable(graph, source, coordinate, field_ty.clone(), span);
            let destination = graph.add_index_place(destination, coordinate, field_ty.clone(), span);
            tail = emit_addressable_copy(graph, tail, source, destination, field_ty, effect_ids, span)?;
        }
        return Ok(tail);
    }

    if ty.array_variant().is_some() {
        let Type::Constructed(TypeName::Size(length), _) =
            ty.array_size().ok_or_else(|| "addressable copy requires a fixed array extent".to_owned())?
        else {
            return Err("addressable copy requires a fixed array extent".into());
        };
        let element_ty =
            ty.elem_type().cloned().ok_or_else(|| "addressable copy source is not an array".to_owned())?;
        let header = graph.skeleton.create_block();
        let body = graph.skeleton.create_block();
        let after = graph.skeleton.create_block();
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let zero = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone(), span);
        graph.skeleton.blocks[block].term = SkeletonTerminator::Branch {
            target: header,
            args: graph.admit_flow_values([zero]),
        };
        let index = graph.add_block_param(header, i32_ty.clone());
        let extent = graph.intern_pure(PureOp::Int(length.to_string()), smallvec![], i32_ty.clone(), span);
        let condition = graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Less),
            smallvec![index, extent],
            Type::Constructed(TypeName::Bool, vec![]),
            span,
        );
        graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
            cond: condition,
            then_target: body,
            then_args: vec![],
            else_target: after,
            else_args: vec![],
        };
        graph.skeleton.blocks[header].control_header = Some(crate::flow::ControlHeader::Loop {
            merge: after,
            continue_block: body,
        });

        let source = index_addressable(graph, source, index, element_ty.clone(), span);
        let destination = graph.add_index_place(destination, index, element_ty.clone(), span);
        let tail = emit_addressable_copy(graph, body, source, destination, &element_ty, effect_ids, span)?;
        let one = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone(), span);
        let next = graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Add),
            smallvec![index, one],
            i32_ty,
            span,
        );
        graph.skeleton.blocks[tail].term = SkeletonTerminator::Branch {
            target: header,
            args: graph.admit_flow_values([next]),
        };
        return Ok(after);
    }

    let AddressableSource::Place(source) = source else {
        return Err("a view source must be indexed before scalar copy".into());
    };
    let value = emit_load(graph, block, source, ty.clone(), effect_ids, span);
    emit_store(graph, block, destination, value, effect_ids, span);
    Ok(block)
}

fn index_addressable<P: Family>(
    graph: &mut EGraph<P>,
    source: AddressableSource,
    index: ValueId,
    pointee: Type<TypeName>,
    span: Option<Span>,
) -> AddressableSource {
    AddressableSource::Place(match source {
        AddressableSource::Place(place) => graph.add_index_place(place, index, pointee, span),
        AddressableSource::View(view) => graph.add_view_index_place(view, index, pointee, span),
    })
}

/// Construct a `Load` and its result without choosing its position in a
/// block. Rewriters use this when a synthesized load must be inserted before
/// an existing scheduled operation instead of appended to the block tail.
pub fn detached_load<P: Family>(
    graph: &mut EGraph<P>,
    place: PlaceId,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> (ValueId, SideEffect<P>) {
    let effect_in = alloc_effect(effect_ids);
    let effect_out = alloc_effect(effect_ids);
    let result = graph.alloc_side_effect_result(elem_ty);
    let effect = SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Load { place }),
        operands: smallvec![],
        result: Some(graph.value_result(result)),
        effects: Some((effect_in, effect_out)),
        span,
    };
    (result, effect)
}

/// Emit a function-local allocation and return its addressable place.
pub fn emit_alloca<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> PlaceId {
    let (place, effect) = detached_alloca(graph, elem_ty, effect_ids, span);
    graph.skeleton.blocks[block].side_effects.push(effect);
    place
}

/// Construct a function-local allocation without choosing its position in a
/// block.
pub fn detached_alloca<P: Family>(
    graph: &mut EGraph<P>,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> (PlaceId, SideEffect<P>) {
    let effect_in = alloc_effect(effect_ids);
    let effect_out = alloc_effect(effect_ids);
    let place = graph.add_alloca_place(
        PlaceType {
            pointee: elem_ty,
            region: PlaceRegion::Function,
            access: PlaceAccess::ReadWrite,
        },
        span,
    );
    let effect = SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Alloca { result: place }),
        operands: smallvec![],
        result: None,
        effects: Some((effect_in, effect_out)),
        span,
    };
    (place, effect)
}

/// Index into an existing place to produce a
/// sub-place addressing one element. The parent place can be an `Alloca`'d
/// array or any other place-producing node; the result has element type
/// `elem_ty` (e.g. `T` for an `[T;N]` parent).
pub fn intern_place_index<P: Family>(
    graph: &mut EGraph<P>,
    parent_place: PlaceId,
    index_nid: ValueId,
    elem_ty: Type<TypeName>,
    span: Option<Span>,
) -> PlaceId {
    graph.add_index_place(parent_place, index_nid, elem_ty, span)
}

/// Emit `place[index] = value` as a `PlaceIndex` sub-place + `Store` in
/// `block`. Companion to `emit_storage_store` for function-local Alloca'd
/// arrays — no whole-array `Load`/`Store` round-trip.
pub fn emit_place_index_store<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    parent_place: PlaceId,
    index_nid: ValueId,
    value_nid: ValueId,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) {
    let elem_place = intern_place_index(graph, parent_place, index_nid, elem_ty, span);
    let _ = emit_store(graph, block, elem_place, value_nid, effect_ids, span);
}

/// Emit `view[index]` as a `ViewIndex` place + `Load` in `block`; returns the
/// loaded value. Companion to `emit_storage_store`.
pub fn emit_view_load<P: Family>(
    graph: &mut EGraph<P>,
    block: BlockId,
    view_nid: ValueId,
    index_nid: ValueId,
    elem_ty: Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> ValueId {
    let view = graph.view_id(view_nid);
    let place = graph.add_view_index_place(view, index_nid, elem_ty.clone(), span);
    emit_load(graph, block, place, elem_ty, effect_ids, span)
}

/// Push a SOAC side effect into `block` with its complete result routes.
pub fn emit_pending_soac<P: WynSoacPhase>(
    graph: &mut EGraph<P>,
    block: BlockId,
    id: P::SoacId,
    soac: Soac<P>,
    operands: SmallVec<[OperandRef; 4]>,
    result: ResultBinding<Type<TypeName>>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    span: Option<Span>,
) -> ResultBinding<Type<TypeName>> {
    let effect_in = alloc_effect(effect_ids);
    let effect_out = alloc_effect(effect_ids);
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Soac(SoacEffect(id, soac)),
        operands: operands,
        result: Some(result.clone()),
        effects: Some((effect_in, effect_out)),
        span,
    });
    result
}

// ---------------------------------------------------------------------------
// Read-side inspection
// ---------------------------------------------------------------------------

/// Return the semantic identity carried by a storage-view node.
pub fn extract_storage_view_source<P: Family<Resource = super::program::SemanticResourceRef>>(
    graph: &EGraph<P>,
    view_nid: ValueId,
) -> Option<super::program::SemanticResourceRef> {
    let view_nid = graph.canonical_value(view_nid);
    match &graph.nodes[view_nid].kind {
        ValueKind::Pure {
            op: PureOp::StorageView(PureViewSource::Storage(resource)),
            ..
        } => Some(*resource),
        _ => None,
    }
}

/// If `nid` is a `PureOp::ArrayRange`, return `(start, len, step?)`
/// ValueNodeIds. Otherwise `None`.
pub fn extract_array_range_operands<P: Family>(
    graph: &EGraph<P>,
    nid: ValueId,
) -> Option<(ValueId, ValueId, Option<ValueId>)> {
    match &graph.nodes[nid].kind {
        ValueKind::Pure {
            op: PureOp::ArrayRange { has_step },
            operands,
            ..
        } => {
            let step = if *has_step { Some(operands[2]) } else { None };
            Some((operands[0], operands[1], step))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Cross-graph cloning
// ---------------------------------------------------------------------------

/// Recursively clone a pure subgraph rooted at `root` from `src` into
/// `dst`, returning the new root `ValueId`. Copies a reduce's neutral
/// element (or any pure value) from one entry's EGraph into another's —
/// phase2 needs a fresh copy of phase1's NE since EGraph ValueNodeIds don't
/// cross entries.
///
/// Only pure nodes and constants are cloned; encountering a
/// `SideEffectResult` or a `BlockParam` returns `Err` because those
/// reference cross-block / cross-effect data that doesn't translate.
pub fn clone_pure_subgraph<P: Family>(
    src: &EGraph<P>,
    dst: &mut EGraph<P>,
    root: ValueId,
) -> Result<ValueId, String> {
    let mut memo: LookupMap<ValueId, ValueId> = LookupMap::new();
    clone_value_subgraph(
        src,
        dst,
        root,
        &mut memo,
        ConstantCopy::Intern,
        false,
        PureCopy::Preserve,
    )
}

/// Clone an addressable place and the pure value/place dependencies that
/// define its address into another graph.
pub fn clone_place_subgraph<P: Family>(
    src: &EGraph<P>,
    dst: &mut EGraph<P>,
    root: PlaceId,
) -> Result<PlaceId, String> {
    fn clone_place<P: Family>(
        src: &EGraph<P>,
        dst: &mut EGraph<P>,
        source: PlaceId,
        values: &mut LookupMap<ValueId, ValueId>,
        places: &mut LookupMap<PlaceId, PlaceId>,
    ) -> Result<PlaceId, String> {
        if let Some(&target) = places.get(&source) {
            return Ok(target);
        }
        let place = src
            .places
            .get(source)
            .ok_or_else(|| format!("clone_place_subgraph: missing place {source:?}"))?
            .clone();
        let ty = place.ty().clone();
        let span = place.span();
        let clone_value = |value, dst: &mut EGraph<P>, values: &mut LookupMap<ValueId, ValueId>| {
            clone_value_subgraph(
                src,
                dst,
                value,
                values,
                ConstantCopy::Intern,
                false,
                PureCopy::Preserve,
            )
        };
        let target = match place.op() {
            PlaceOp::Parameter { parameter } => dst.add_place_parameter(*parameter, ty),
            PlaceOp::View { view } => {
                let view = clone_value(view.value(), dst, values)?;
                dst.add_view_place(dst.view_id(view), ty.pointee, ty.access, span)
            }
            PlaceOp::AllocaResult => dst.add_alloca_place(ty, span),
            PlaceOp::Index { base, index } => {
                let base = clone_place(src, dst, *base, values, places)?;
                let index = clone_value(*index, dst, values)?;
                dst.add_index_place(base, index, ty.pointee, span)
            }
            PlaceOp::Slice { base, start, length } => {
                let base = clone_place(src, dst, *base, values, places)?;
                let start = clone_value(*start, dst, values)?;
                let length = clone_value(*length, dst, values)?;
                dst.add_slice_place(base, start, length, ty.pointee, span)
            }
            PlaceOp::ViewIndex { view, index } => {
                let view = clone_value(view.value(), dst, values)?;
                let index = clone_value(*index, dst, values)?;
                let view = dst.view_id(view);
                dst.add_view_index_place(view, index, ty.pointee, span)
            }
            PlaceOp::OutputSlot { index } => dst.add_output_place(*index, ty),
        };
        places.insert(source, target);
        Ok(target)
    }

    clone_place(src, dst, root, &mut LookupMap::new(), &mut LookupMap::new())
}

/// Clone one typed boundary operand without collapsing its value, view, or
/// place representation.
pub fn clone_operand_subgraph<P: Family>(
    src: &EGraph<P>,
    dst: &mut EGraph<P>,
    operand: OperandRef,
) -> Result<OperandRef, String> {
    Ok(match operand {
        OperandRef::Value(value) => OperandRef::Value(clone_pure_subgraph(src, dst, value)?),
        OperandRef::View(view) => {
            let value = clone_pure_subgraph(src, dst, view.value())?;
            OperandRef::View(dst.view_id(value))
        }
        OperandRef::Place(place) => OperandRef::Place(clone_place_subgraph(src, dst, place)?),
    })
}

/// Clone a pure subgraph of `src` into `dst`, but substitute the given `src`
/// nodes for already-existing `dst` nodes: any `(from, to)` pre-seeds the clone
/// memo, so a reference to `from` in `src` becomes `to` in `dst`. Lets a value
/// rooted at a non-pure node (e.g. a SOAC result) be re-expressed over a
/// replacement `dst` value without rebuilding its projection structure by hand.
pub fn clone_pure_subgraph_substituting<P: Family>(
    src: &EGraph<P>,
    dst: &mut EGraph<P>,
    root: ValueId,
    subs: &[(ValueId, ValueId)],
) -> Result<ValueId, String> {
    let mut memo: LookupMap<ValueId, ValueId> = subs.iter().copied().collect();
    clone_value_subgraph(
        src,
        dst,
        root,
        &mut memo,
        ConstantCopy::Intern,
        false,
        PureCopy::Preserve,
    )
}

#[derive(Clone, Copy)]
pub(crate) enum ConstantCopy {
    Intern,
    PreserveIdentity,
}

#[derive(Clone, Copy)]
pub(crate) enum PureCopy {
    /// Reproduce the source DAG exactly apart from hash-consing.
    Preserve,
    /// Re-run algebraic folds after operands have been substituted.
    Fold,
}

pub(crate) fn clone_value_subgraph<P: Family>(
    src: &EGraph<P>,
    dst: &mut EGraph<P>,
    nid: ValueId,
    memo: &mut LookupMap<ValueId, ValueId>,
    constants: ConstantCopy,
    allow_unions: bool,
    pure: PureCopy,
) -> Result<ValueId, String> {
    clone_value_subgraph_inner(src, dst, nid, memo, constants, allow_unions, pure)
}

fn clone_value_subgraph_inner<P: Family>(
    src: &EGraph<P>,
    dst: &mut EGraph<P>,
    nid: ValueId,
    memo: &mut LookupMap<ValueId, ValueId>,
    constants: ConstantCopy,
    allow_unions: bool,
    pure: PureCopy,
) -> Result<ValueId, String> {
    if let Some(&existing) = memo.get(&nid) {
        return Ok(existing);
    }
    let canonical = src.canonical_value(nid);
    if canonical != nid {
        let target = clone_value_subgraph_inner(src, dst, canonical, memo, constants, allow_unions, pure)?;
        memo.insert(nid, target);
        return Ok(target);
    }
    let source = src.nodes.get(nid).ok_or_else(|| format!("clone_value_subgraph: missing node {nid:?}"))?;
    let ty = source.ty.clone();
    let new_nid = match &source.kind {
        ValueKind::Constant(c) => match constants {
            ConstantCopy::Intern => dst.intern_constant(*c, ty),
            ConstantCopy::PreserveIdentity => {
                let target = dst.nodes.insert(Value {
                    kind: ValueKind::Constant(*c),
                    ty,
                    span: source.span,
                    alias: None,
                    result_origins: Vec::new(),
                });
                target
            }
        },
        ValueKind::Pure { op, operands, .. } => {
            let new_ops: SmallVec<[ValueId; 4]> = operands
                .iter()
                .map(|&operand| {
                    clone_value_subgraph_inner(src, dst, operand, memo, constants, allow_unions, pure)
                })
                .collect::<Result<_, _>>()?;
            if matches!(pure, PureCopy::Fold) {
                if let Some(folded) = dst.try_algebraic_fold(op, &new_ops, &ty) {
                    folded
                } else {
                    dst.intern_pure(op.clone(), new_ops, ty, source.span)
                }
            } else {
                dst.intern_pure(op.clone(), new_ops, ty, source.span)
            }
        }
        ValueKind::Union { left, right } if allow_unions => {
            let left = clone_value_subgraph_inner(src, dst, *left, memo, constants, allow_unions, pure)?;
            let right = clone_value_subgraph_inner(src, dst, *right, memo, constants, allow_unions, pure)?;
            dst.add_union(left, right)
        }
        ValueKind::CallResult { call, .. } => {
            let source_call = src.call(*call).clone();
            if !source_call.result().places().is_empty() {
                return Err("clone call requires explicit destination-place substitutions".into());
            }
            let arguments = source_call
                .arguments()
                .iter()
                .map(|argument| match *argument {
                    OperandRef::Value(value) => {
                        clone_value_subgraph_inner(src, dst, value, memo, constants, allow_unions, pure)
                            .map(OperandRef::Value)
                    }
                    OperandRef::View(view) => clone_value_subgraph_inner(
                        src,
                        dst,
                        view.value(),
                        memo,
                        constants,
                        allow_unions,
                        pure,
                    )
                    .map(|value| OperandRef::View(dst.view_id(value))),
                    OperandRef::Place(_) => {
                        Err("clone call requires explicit place-argument substitutions".into())
                    }
                })
                .collect::<Result<Vec<_>, String>>()?
                .into_boxed_slice();
            let (_, _, mappings) = dst.add_projected_call(
                &source_call,
                arguments,
                |source_result| {
                    let source = &src.nodes[source_result];
                    let ValueKind::CallResult { slot, .. } = source.kind() else {
                        unreachable!("call result binding contains a non-call value")
                    };
                    (*slot, source.ty().clone(), source.span())
                },
                |_| unreachable!("place-backed call was rejected before cloning"),
            );
            for (source, target) in mappings {
                memo.insert(source, target);
            }
            memo[&nid]
        }
        other => {
            let producer = src.side_effect_index().site(nid).map(|site| {
                format!(
                    " produced by {:?}",
                    src.skeleton.blocks[site.block].side_effects[site.index].kind
                )
            });
            return Err(format!(
                "clone_value_subgraph cannot copy {nid:?} ({ty:?}): {other:?}{}",
                producer.unwrap_or_default()
            ));
        }
    };
    let origins = source
        .result_origins()
        .iter()
        .cloned()
        .map(|origin| {
            origin.try_map(
                &mut |ty| Ok::<_, String>(ty),
                &mut |value| {
                    clone_value_subgraph_inner(src, dst, value, memo, constants, allow_unions, pure)
                },
                &mut |_| Err("clone result origin requires explicit place substitutions".into()),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    dst.nodes[new_nid].result_origins = origins;
    memo.insert(nid, new_nid);
    Ok(new_nid)
}

pub(crate) struct ClonedBody {
    pub entry: BlockId,
    pub returns: Vec<(BlockId, ResultBinding<Type<TypeName>>)>,
    pub node_count: usize,
    pub block_count: usize,
}

/// Clone a complete reachable function body into another graph while binding
/// its physical parameters to one fully applied call boundary. Values, places,
/// calls, effects, block parameters, aliases, and structured control metadata
/// are remapped together; callers only orchestrate where the cloned entry and
/// returns splice into their surrounding CFG.
pub(crate) fn clone_body_substituting<P: Family>(
    source: &EGraph<P>,
    target: &mut EGraph<P>,
    arguments: &[OperandRef],
    place_bindings: &[(PlaceId, PlaceId)],
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<ClonedBody, String> {
    let blocks = wyn_graph::reachable_from_ordered(
        [source.skeleton.entry],
        wyn_graph::WalkOrder::DepthFirst,
        |block, out| out.extend(source.skeleton.blocks[block].term.successors()),
    );
    let mut cloner = BodyCloner {
        source,
        target,
        arguments,
        values: LookupMap::new(),
        places: place_bindings.iter().copied().collect(),
        bound_places: place_bindings.iter().map(|(source, _)| *source).collect(),
        calls: LookupMap::new(),
        blocks: LookupMap::new(),
        effects: LookupMap::new(),
        effect_ids,
    };

    for source_block in &blocks {
        cloner.blocks.insert(*source_block, cloner.target.skeleton.create_block());
    }
    for source_block in &blocks {
        let target_block = cloner.blocks[source_block];
        for parameter in &source.skeleton.blocks[*source_block].params {
            let source_value = parameter.value();
            let target_value =
                cloner.target.add_block_param(target_block, source.nodes[source_value].ty.clone());
            cloner.values.insert(source_value, target_value);
        }
    }

    let mut returns = Vec::new();
    for source_block in &blocks {
        let target_block = cloner.blocks[source_block];
        for effect in &source.skeleton.blocks[*source_block].side_effects {
            if let Some(effect) = cloner.clone_effect(effect)? {
                cloner.target.skeleton.blocks[target_block].side_effects.push(effect);
            }
        }

        let term = match &source.skeleton.blocks[*source_block].term {
            SkeletonTerminator::Return(Some(result)) => {
                let result = cloner.clone_result(result)?;
                returns.push((target_block, result.clone()));
                SkeletonTerminator::Return(Some(result))
            }
            SkeletonTerminator::Return(None) => SkeletonTerminator::Return(None),
            SkeletonTerminator::Branch { target, args } => SkeletonTerminator::Branch {
                target: cloner.blocks[target],
                args: args
                    .iter()
                    .map(|value| {
                        cloner.clone_value(value.value()).map(|value| cloner.target.admit_flow_value(value))
                    })
                    .collect::<Result<_, _>>()?,
            },
            SkeletonTerminator::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
            } => SkeletonTerminator::CondBranch {
                cond: cloner.clone_value(*cond)?,
                then_target: cloner.blocks[then_target],
                then_args: then_args
                    .iter()
                    .map(|value| {
                        cloner.clone_value(value.value()).map(|value| cloner.target.admit_flow_value(value))
                    })
                    .collect::<Result<_, _>>()?,
                else_target: cloner.blocks[else_target],
                else_args: else_args
                    .iter()
                    .map(|value| {
                        cloner.clone_value(value.value()).map(|value| cloner.target.admit_flow_value(value))
                    })
                    .collect::<Result<_, _>>()?,
            },
            SkeletonTerminator::Unreachable => SkeletonTerminator::Unreachable,
        };
        cloner.target.skeleton.blocks[target_block].term = term;
        cloner.target.skeleton.blocks[target_block].control_header = source.skeleton.blocks[*source_block]
            .control_header
            .as_ref()
            .map(|header| header.remap(&|block| cloner.blocks[&block]));
    }

    let entry = cloner.blocks[&source.skeleton.entry];
    let node_count = cloner.values.len();
    cloner.target.verify_hash_cons()?;
    cloner.target.skeleton.verify_branch_arities()?;
    Ok(ClonedBody {
        entry,
        returns,
        node_count,
        block_count: blocks.len(),
    })
}

struct BodyCloner<'a, P: Family> {
    source: &'a EGraph<P>,
    target: &'a mut EGraph<P>,
    arguments: &'a [OperandRef],
    values: LookupMap<ValueId, ValueId>,
    places: LookupMap<PlaceId, PlaceId>,
    bound_places: HashSet<PlaceId>,
    calls: LookupMap<CallSiteId, CallSiteId>,
    blocks: LookupMap<BlockId, BlockId>,
    effects: LookupMap<EffectToken, EffectToken>,
    effect_ids: &'a mut crate::IdSource<EffectToken>,
}

impl<P: Family> BodyCloner<'_, P> {
    fn clone_value(&mut self, source: ValueId) -> Result<ValueId, String> {
        if let Some(target) = self.values.get(&source) {
            return Ok(*target);
        }
        let canonical = self.source.canonical_value(source);
        if canonical != source {
            let target = self.clone_value(canonical)?;
            self.values.insert(source, target);
            return Ok(target);
        }
        let definition = self
            .source
            .nodes
            .get(source)
            .ok_or_else(|| format!("body clone references missing value {source:?}"))?;
        let target = match &definition.kind {
            ValueKind::FuncParam { parameter } => {
                self.arguments.get(parameter.index()).and_then(|argument| argument.value()).ok_or_else(
                    || {
                        format!(
                            "body clone parameter {} requires a value or view argument",
                            parameter.index()
                        )
                    },
                )?
            }
            ValueKind::BlockParam { .. } => *self
                .values
                .get(&source)
                .ok_or_else(|| format!("body clone omitted block parameter {source:?}"))?,
            ValueKind::Constant(value) => self.target.intern_constant(value.clone(), definition.ty.clone()),
            ValueKind::Pure { op, operands } => {
                let operands = operands
                    .iter()
                    .map(|value| self.clone_value(*value))
                    .collect::<Result<SmallVec<[ValueId; 4]>, _>>()?;
                self.target.try_algebraic_fold(op, &operands, &definition.ty).unwrap_or_else(|| {
                    self.target.intern_pure(op.clone(), operands, definition.ty.clone(), definition.span)
                })
            }
            ValueKind::Union { left, right } => {
                let left = self.clone_value(*left)?;
                let right = self.clone_value(*right)?;
                self.target.add_union(left, right)
            }
            ValueKind::SideEffectResult => self.target.alloc_side_effect_result(definition.ty.clone()),
            ValueKind::CallResult { call, .. } => {
                self.clone_call(*call)?;
                *self
                    .values
                    .get(&source)
                    .ok_or_else(|| format!("body clone omitted call result {source:?}"))?
            }
            ValueKind::PlaceLength { place } => {
                let place = self.clone_place(*place)?;
                self.target.add_place_length(place, definition.ty.clone(), definition.span)
            }
            ValueKind::PlaceView { place } => {
                let place = self.clone_place(*place)?;
                self.target.add_place_view(place, definition.ty.clone(), definition.span).value()
            }
        };
        self.values.insert(source, target);
        Ok(target)
    }

    fn clone_place(&mut self, source: PlaceId) -> Result<PlaceId, String> {
        if let Some(target) = self.places.get(&source) {
            return Ok(*target);
        }
        let definition = self
            .source
            .places()
            .get(source)
            .ok_or_else(|| format!("body clone references missing place {source:?}"))?;
        let target = match definition.op() {
            PlaceOp::Parameter { parameter } => {
                self.arguments.get(parameter.index()).and_then(|argument| argument.place()).ok_or_else(
                    || {
                        format!(
                            "body clone parameter {} requires a place argument",
                            parameter.index()
                        )
                    },
                )?
            }
            PlaceOp::View { view } => {
                let view = self.clone_value(view.value())?;
                self.target.add_view_place(
                    self.target.view_id(view),
                    definition.ty().pointee.clone(),
                    definition.ty().access,
                    definition.span(),
                )
            }
            PlaceOp::AllocaResult => {
                self.target.add_alloca_place(definition.ty().clone(), definition.span())
            }
            PlaceOp::Index { base, index } => {
                let base = self.clone_place(*base)?;
                let index = self.clone_value(*index)?;
                self.target.add_index_place(base, index, definition.ty().pointee.clone(), definition.span())
            }
            PlaceOp::Slice { base, start, length } => {
                let base = self.clone_place(*base)?;
                let start = self.clone_value(*start)?;
                let length = self.clone_value(*length)?;
                self.target.add_slice_place(
                    base,
                    start,
                    length,
                    definition.ty().pointee.clone(),
                    definition.span(),
                )
            }
            PlaceOp::ViewIndex { view, index } => {
                let view = self.clone_value(view.value())?;
                let index = self.clone_value(*index)?;
                self.target.add_view_index_place(
                    self.target.view_id(view),
                    index,
                    definition.ty().pointee.clone(),
                    definition.span(),
                )
            }
            PlaceOp::OutputSlot { index } => self.target.add_output_place(*index, definition.ty().clone()),
        };
        self.places.insert(source, target);
        Ok(target)
    }

    fn clone_operand(&mut self, source: OperandRef) -> Result<OperandRef, String> {
        Ok(match source {
            OperandRef::Value(value) => {
                let value = self.clone_value(value)?;
                self.target.operand_ref(value)
            }
            OperandRef::View(view) => {
                let value = self.clone_value(view.value())?;
                OperandRef::View(self.target.view_id(value))
            }
            OperandRef::Place(place) => OperandRef::Place(self.clone_place(place)?),
        })
    }

    fn clone_call(&mut self, source: CallSiteId) -> Result<CallSiteId, String> {
        if let Some(target) = self.calls.get(&source) {
            return Ok(*target);
        }
        let call = self.source.call(source).clone();
        let arguments = call
            .arguments()
            .iter()
            .map(|argument| self.clone_operand(*argument))
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        for place in call.result().places() {
            self.clone_place(place)?;
        }
        let places = &self.places;
        let (target, _, values) = self.target.add_projected_call(
            &call,
            arguments,
            |source_value| {
                let definition = &self.source.nodes[source_value];
                let ValueKind::CallResult { slot, .. } = definition.kind() else {
                    unreachable!("call result binding contains a non-call value")
                };
                (*slot, definition.ty().clone(), definition.span())
            },
            |place| places[&place],
        );
        self.calls.insert(source, target);
        self.values.extend(values);
        Ok(target)
    }

    fn clone_result(
        &mut self,
        source: &ResultBinding<Type<TypeName>>,
    ) -> Result<ResultBinding<Type<TypeName>>, String> {
        let mut values = LookupMap::new();
        for source in source.values() {
            values.insert(source, self.clone_value(source)?);
        }
        let mut places = LookupMap::new();
        for source in source.places() {
            places.insert(source, self.clone_place(source)?);
        }
        Ok(source.clone().map(|ty| ty, |value| values[&value], |place| places[&place]))
    }

    fn clone_effect(&mut self, source: &SideEffect<P>) -> Result<Option<SideEffect<P>>, String> {
        if matches!(source.kind(), SideEffectKind::Soac(_)) {
            return Err("body clone requires SOAC expansion before effectful inlining".into());
        }
        if matches!(
            source.kind(),
            SideEffectKind::Effect(EffectOp::Alloca { result }) if self.bound_places.contains(result)
        ) {
            return Ok(None);
        }
        let operands = source
            .operands()
            .iter()
            .map(|operand| self.clone_operand(*operand))
            .collect::<Result<SmallVec<[OperandRef; 4]>, _>>()?;
        let result = source.result().map(|result| self.clone_result(result)).transpose()?;
        let SideEffectKind::Effect(operation) = source.kind() else {
            unreachable!()
        };
        let operation = match operation {
            EffectOp::Call { site } => EffectOp::Call {
                site: self.clone_call(*site)?,
            },
            EffectOp::Op { tag } => EffectOp::Op { tag: tag.clone() },
            EffectOp::Alloca { result } => EffectOp::Alloca {
                result: self.clone_place(*result)?,
            },
            EffectOp::Load { place } => EffectOp::Load {
                place: self.clone_place(*place)?,
            },
            EffectOp::Store { place } => EffectOp::Store {
                place: self.clone_place(*place)?,
            },
            EffectOp::Atomic { place, op } => EffectOp::Atomic {
                place: self.clone_place(*place)?,
                op: *op,
            },
            EffectOp::ControlBarrier => EffectOp::ControlBarrier,
        };
        let effects = source
            .effects()
            .map(|(input, output)| (self.clone_effect_token(input), self.clone_effect_token(output)));
        Ok(Some(SideEffect::new(
            SideEffectKind::Effect(operation),
            operands,
            result,
            effects,
            source.span(),
        )))
    }

    fn clone_effect_token(&mut self, source: EffectToken) -> EffectToken {
        *self.effects.entry(source).or_insert_with(|| alloc_effect(self.effect_ids))
    }
}
