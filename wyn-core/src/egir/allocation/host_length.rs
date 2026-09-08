//! Preserve host-known logical array lengths on output resources before scheduling.
//! This follows semantic spaces, never physical workgroup counts.
use super::super::program::Entry;
use super::super::types::{EGraph, PureOp, SegExtent, ValueKind};
use super::*;
use crate::builtins::lowering::PrimOp;
use crate::builtins::{by_id, BuiltinLowering};
use crate::interface::StorageLayout;
use crate::op::BinaryOperator;
use crate::pipeline_descriptor::{HostBinary, HostExpression, HostScalar};
use crate::ssa;

pub(super) fn retain_output_lengths(program: &mut Optimized) -> Result<(), ConvertError> {
    for entry in &mut program.entry_points {
        let mut lengths = HashMap::new();
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
                        if let Some(count) = expression(&entry.graph, entry, *node, 0) {
                            host_extents.insert(*node, count);
                        }
                    }
                }
                // Array-valued extents carry view provenance, resolved by existing
                // residency rules. Only scalar lengths are host expressions.
                if space.dims().iter().any(|d| matches!(d, SegExtent::Value(node) if scalar(entry.graph.nodes[entry.graph.canonical_value(*node)].ty()).is_none())) { continue; }
                // Fixed and resource-backed domains retain their existing policies.
                if !space.dims().iter().any(|d| matches!(d, SegExtent::Value(_))) {
                    continue;
                }
                for slot in output_slots {
                    let output = &entry.outputs[slot.0];
                    let EntryOutputKind::Storage {
                        length: Some(BufferLen::SameAsDispatch { elem_bytes }),
                        ..
                    } = &output.kind
                    else {
                        continue;
                    };
                    let count = space.dims().iter().try_fold(None, |count, extent| {
                        let dim = match extent {
                            SegExtent::Fixed(n) => Some(HostExpression::Constant { scalar: HostScalar::I32, bits: *n }),
                            SegExtent::Value(node) => expression(&entry.graph, entry, *node, 0),
                            _ => None,
                        }?;
                        Some(Some(match count {
                            None => dim,
                            Some(left) => HostExpression::Binary { op: HostBinary::Multiply, left: Box::new(left), right: Box::new(dim) },
                        }))
                    }).flatten().ok_or_else(|| ConvertError::GraphError(format!(
                        "output {} of {} has a logical length that cannot be evaluated from host uniforms; use a fixed or input-derived capacity", slot.0, entry.name)))?;
                    let policy = BufferLen::HostExpression {
                        count,
                        elem_bytes: *elem_bytes,
                    };
                    if let Some(previous) = lengths.insert(slot.0, policy.clone()) {
                        if previous != policy {
                            return Err(ConvertError::GraphError(format!(
                                "output {} of {} has conflicting logical lengths",
                                slot.0, entry.name
                            )));
                        }
                    }
                }
            }
        }
        // Compiler-created materializations use the same logical expression as
        // public outputs through LogicalSize::for_space.
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
                        SegExtent::Value(node) if host_extents.contains_key(&node) => SegExtent::Host {
                            node,
                            count: host_extents[&node].clone(),
                        },
                        other => other,
                    })
                    .collect();
                *space = super::super::types::SegSpace::from_dims(dims).expect("nonempty space");
            }
        }
        for (slot, length) in lengths {
            if let EntryOutputKind::Storage { length: policy, .. } = &mut entry.outputs[slot].kind {
                *policy = Some(length);
            }
        }
    }
    Ok(())
}

fn scalar(ty: &Type<TypeName>) -> Option<HostScalar> {
    match ty {
        Type::Constructed(TypeName::Int(32), _) => Some(HostScalar::I32),
        Type::Constructed(TypeName::UInt(32), _) => Some(HostScalar::U32),
        Type::Constructed(TypeName::Float(32), _) => Some(HostScalar::F32),
        _ => None,
    }
}

fn expression(
    graph: &EGraph<Semantic>,
    entry: &Entry<Semantic>,
    node: ValueId,
    depth: usize,
) -> Option<HostExpression> {
    if depth > 128 {
        return None;
    }
    let node = graph.canonical_value(node);
    let ty = scalar(graph.nodes[node].ty())?;
    if let Some((binding, offset)) = uniform_location(graph, entry, node, 0) {
        return Some(HostExpression::Uniform {
            set: binding.set,
            binding: binding.binding,
            offset,
            scalar: ty,
        });
    }
    let ValueKind::Pure { op, operands } = graph.nodes[node].kind() else {
        return None;
    };
    let recurse = |n| expression(graph, entry, n, depth + 1);
    let binary = |op| {
        let [a, b] = operands.as_slice() else { return None };
        Some(HostExpression::Binary {
            op,
            left: Box::new(recurse(*a)?),
            right: Box::new(recurse(*b)?),
        })
    };
    match op {
        PureOp::Int(n) => Some(HostExpression::Constant {
            scalar: ty,
            bits: n.parse::<i32>().ok()? as u32,
        }),
        PureOp::Uint(n) => Some(HostExpression::Constant {
            scalar: ty,
            bits: n.parse().ok()?,
        }),
        PureOp::Float(n) => Some(HostExpression::Constant {
            scalar: ty,
            bits: n.parse::<f32>().ok()?.to_bits(),
        }),
        PureOp::BinOp(op) => binary(match op {
            BinaryOperator::Add => HostBinary::Add,
            BinaryOperator::Subtract => HostBinary::Subtract,
            BinaryOperator::Multiply => HostBinary::Multiply,
            BinaryOperator::Divide => HostBinary::Divide,
            BinaryOperator::Remainder => HostBinary::Remainder,
            _ => return None,
        }),
        PureOp::Intrinsic { id, overload_idx } => {
            match &by_id(*id).overloads().get(*overload_idx)?.lowering {
                BuiltinLowering::PrimOp(op) => match op {
                    PrimOp::FPToSI | PrimOp::FPToUI | PrimOp::SIToFP | PrimOp::UIToFP | PrimOp::Bitcast => {
                        let [value] = operands.as_slice() else { return None };
                        // Numeric i32/u32 casts reinterpret bits. Float bitcasts are not numeric casts.
                        let from = scalar(graph.nodes[*value].ty())?;
                        if *op == PrimOp::Bitcast && (from == HostScalar::F32) != (ty == HostScalar::F32) {
                            return None;
                        }
                        Some(HostExpression::Convert {
                            to: ty,
                            value: Box::new(recurse(*value)?),
                        })
                    }
                    PrimOp::IAdd | PrimOp::FAdd => binary(HostBinary::Add),
                    PrimOp::ISub | PrimOp::FSub => binary(HostBinary::Subtract),
                    PrimOp::IMul | PrimOp::FMul => binary(HostBinary::Multiply),
                    PrimOp::SDiv | PrimOp::UDiv | PrimOp::FDiv => binary(HostBinary::Divide),
                    PrimOp::SRem | PrimOp::UMod | PrimOp::FRem => binary(HostBinary::Remainder),
                    _ => None,
                },
                _ => None,
            }
        }
        _ => None,
    }
}

fn uniform_location(
    graph: &EGraph<Semantic>,
    entry: &Entry<Semantic>,
    node: ValueId,
    depth: usize,
) -> Option<(BindingRef, u32)> {
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
            Some((binding, 0))
        }
        ValueKind::Pure {
            op: PureOp::Project { index },
            operands,
        } => {
            let [base] = operands.as_slice() else { return None };
            let (binding, offset) = uniform_location(graph, entry, *base, depth + 1)?;
            let ty = graph.nodes[graph.canonical_value(*base)].ty();
            let field_offset = if ty.is_vec() {
                u32::try_from(*index).ok()?.checked_mul(ssa::layout::type_byte_size(ty.elem_type()?)?)?
            } else {
                *ssa::layout::block_layout(ty, StorageLayout::Std140)?
                    .member_offsets
                    .get(*index as usize)?
            };
            Some((binding, offset.checked_add(field_offset)?))
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
            let (binding, offset) = uniform_location(graph, entry, *base, depth + 1)?;
            let stride = ssa::layout::type_byte_size(ty.elem_type()?)?;
            Some((
                binding,
                offset.checked_add(index.parse::<u32>().ok()?.checked_mul(stride)?)?,
            ))
        }
        _ => None,
    }
}
