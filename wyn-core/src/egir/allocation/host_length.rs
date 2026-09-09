//! Preserve logical array lengths whose allocation capacity must be supplied
//! by the host. This follows semantic spaces, never physical workgroup counts.

use std::collections::BTreeSet;

use super::super::graph_ops;
use super::super::program::Entry;
use super::super::types::{EGraph, PureOp, SegExtent, ValueKind};
use super::*;
use crate::interface::StorageLayout;
use crate::pipeline_descriptor::{HostSizeInput, HostSizeScalar};
use crate::ssa;

pub(super) fn retain_output_lengths(program: &mut Optimized) -> Result<(), ConvertError> {
    for entry in &mut program.entry_points {
        let mut lengths: HashMap<_, (BTreeSet<HostSizeInput>, u32)> = HashMap::new();
        let mut host_extents = HashMap::new();

        for block in entry.graph.skeleton.blocks.values() {
            for effect in &block.side_effects {
                let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                    continue;
                };
                let screma::SemanticState::Segmented {
                    space, output_slots, ..
                } = op.semantic_state()
                else {
                    continue;
                };

                for dim in space.dims() {
                    if let SegExtent::Value(node) = dim {
                        let extent_node = *node;
                        let canonical = entry.graph.canonical_value(extent_node);
                        if scalar(entry.graph.nodes[canonical].ty()).is_some() {
                            host_extents
                                .insert(extent_node, host_dependencies(&entry.graph, entry, extent_node));
                        }
                    }
                }

                // Array-valued extents carry view provenance and are resolved by
                // the existing residency rules. Scalar values may use arbitrary
                // shader computation, so the host must provide their capacity.
                if !space.dims().iter().any(
                    |dim| matches!(dim, SegExtent::Value(node) if scalar(entry.graph.nodes[entry.graph.canonical_value(*node)].ty()).is_some()),
                ) {
                    continue;
                }
                if space.dims().iter().any(
                    |dim| matches!(dim, SegExtent::Value(node) if scalar(entry.graph.nodes[entry.graph.canonical_value(*node)].ty()).is_none()),
                ) {
                    continue;
                }

                let inputs = space
                    .dims()
                    .iter()
                    .filter_map(|dim| match dim {
                        SegExtent::Value(node) => Some(host_dependencies(&entry.graph, entry, *node)),
                        _ => None,
                    })
                    .flatten()
                    .collect::<BTreeSet<_>>();

                for slot in output_slots {
                    let output = &entry.outputs[slot.0];
                    let EntryOutputKind::Storage {
                        length: Some(BufferLen::SameAsDispatch { elem_bytes }),
                        ..
                    } = &output.kind
                    else {
                        continue;
                    };
                    let (known_inputs, known_elem_bytes) =
                        lengths.entry(slot.0).or_insert_with(|| (BTreeSet::new(), *elem_bytes));
                    if *known_elem_bytes != *elem_bytes {
                        return Err(ConvertError::GraphError(format!(
                            "output {} of {} has conflicting storage strides",
                            slot.0, entry.name
                        )));
                    }
                    known_inputs.extend(inputs.iter().cloned());
                }
            }
        }

        // Compiler-created materializations carry the same host-provided
        // dependency metadata through LogicalSize::for_space.
        for block in entry.graph.skeleton.blocks.values_mut() {
            for effect in &mut block.side_effects {
                let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &mut effect.kind else {
                    continue;
                };
                let screma::SemanticState::Segmented { space, .. } = op.semantic_state_mut() else {
                    continue;
                };
                let dims = space
                    .clone()
                    .into_dims()
                    .into_iter()
                    .map(|dim| match dim {
                        SegExtent::Value(node) if host_extents.contains_key(&node) => {
                            SegExtent::HostProvided {
                                node,
                                inputs: host_extents[&node].clone(),
                            }
                        }
                        other => other,
                    })
                    .collect();
                *space = super::super::types::SegSpace::from_dims(dims).expect("nonempty space");
            }
        }

        for (slot, (inputs, elem_bytes)) in lengths {
            if let EntryOutputKind::Storage { length, .. } = &mut entry.outputs[slot].kind {
                *length = Some(BufferLen::HostProvided {
                    inputs: inputs.into_iter().collect(),
                    elem_bytes,
                });
            }
        }
    }
    Ok(())
}

fn scalar(ty: &Type<TypeName>) -> Option<HostSizeScalar> {
    match ty {
        Type::Constructed(TypeName::Int(32), _) => Some(HostSizeScalar::I32),
        Type::Constructed(TypeName::UInt(32), _) => Some(HostSizeScalar::U32),
        Type::Constructed(TypeName::Float(32), _) => Some(HostSizeScalar::F32),
        _ => None,
    }
}

fn host_dependencies(
    graph: &EGraph<Semantic>,
    entry: &Entry<Semantic>,
    node: ValueId,
) -> Vec<HostSizeInput> {
    graph_ops::value_producer_closure(graph, [node])
        .nodes
        .into_iter()
        .filter_map(|dependency| {
            let dependency = graph.canonical_value(dependency);
            let scalar = scalar(graph.nodes[dependency].ty())?;
            let (binding, offset, name) = uniform_location(graph, entry, dependency, 0)?;
            Some(HostSizeInput {
                name,
                set: binding.set,
                binding: binding.binding,
                offset,
                scalar,
            })
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn uniform_location(
    graph: &EGraph<Semantic>,
    entry: &Entry<Semantic>,
    node: ValueId,
    depth: usize,
) -> Option<(BindingRef, u32, String)> {
    if depth > 128 {
        return None;
    }
    let node = graph.canonical_value(node);
    match graph.nodes[node].kind() {
        ValueKind::FuncParam { parameter } => {
            let position = entry.params().abi_position(*parameter)?;
            let [slot] = entry.parameter_inputs.get(position)?.as_slice() else {
                return None;
            };
            let EntryInputKind::Uniform { binding } = entry.inputs.get(slot.0)?.kind else {
                return None;
            };
            Some((binding, 0, entry.inputs.get(slot.0)?.name.clone()))
        }
        ValueKind::Pure {
            op: PureOp::Project { index },
            operands,
        } => {
            let [base] = operands.as_slice() else {
                return None;
            };
            let (binding, offset, name) = uniform_location(graph, entry, *base, depth + 1)?;
            let ty = graph.nodes[graph.canonical_value(*base)].ty();
            let field_offset = if ty.is_vec() {
                u32::try_from(*index).ok()?.checked_mul(ssa::layout::type_byte_size(ty.elem_type()?)?)?
            } else {
                *ssa::layout::block_layout(ty, StorageLayout::Std140)?
                    .member_offsets
                    .get(*index as usize)?
            };
            let field = match ty {
                Type::Constructed(TypeName::Record(fields), _) => fields.0.get(*index as usize)?.clone(),
                _ if ty.is_vec() => ["x", "y", "z", "w"].get(*index as usize)?.to_string(),
                _ => index.to_string(),
            };
            Some((
                binding,
                offset.checked_add(field_offset)?,
                format!("{name}_{field}"),
            ))
        }
        ValueKind::Pure {
            op: PureOp::DynamicExtract | PureOp::Index,
            operands,
        } => {
            let [base, index] = operands.as_slice() else {
                return None;
            };
            let ty = graph.nodes[graph.canonical_value(*base)].ty();
            if !ty.is_vec() {
                return None;
            }
            let ValueKind::Pure {
                op: PureOp::Int(index) | PureOp::Uint(index),
                ..
            } = graph.nodes[graph.canonical_value(*index)].kind()
            else {
                return None;
            };
            let (binding, offset, name) = uniform_location(graph, entry, *base, depth + 1)?;
            let index = index.parse::<usize>().ok()?;
            let stride = ssa::layout::type_byte_size(ty.elem_type()?)?;
            Some((
                binding,
                offset.checked_add(u32::try_from(index).ok()?.checked_mul(stride)?)?,
                format!("{name}_{}", ["x", "y", "z", "w"].get(index)?),
            ))
        }
        _ => None,
    }
}
