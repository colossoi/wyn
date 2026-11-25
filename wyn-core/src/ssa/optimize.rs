//! Concrete array storage preparation shared by both compiler routes.
use super::ir::ValueDef;
use super::stage::{Elaborated, Optimized};
use super::types::{
    ConstantValue, FuncBody, InstId, InstKind, PlaceInfo, PlaceOrigin, ValueId, ValueRef, WynFunction,
};
use crate::builtins::{by_id, Purity};
use crate::host::{Binding, BufferUsage, ModuleInterface, Pipeline, ScalarSource};
use crate::op::{BinaryOperator, OpTag};
use crate::types::{array_view_buffer, is_array_variant_view, is_virtual_array, Type, TypeExt, TypeName};
use crate::{BindingRef, LookupMap, LookupSet};
use std::collections::HashMap;

/// Prepare concrete storage introduced during SSA lowering.
/// Source helper expansion, folding, expression sharing and hoisting belong to Egglog.
pub fn optimize(mut program: Elaborated) -> Optimized {
    let stable = stable_storage(&program.global_context.pipeline);
    let empty = LookupSet::new();
    for function in &mut program.functions {
        prepare_values(&mut function.body, &empty);
    }
    for entry in &mut program.entry_points {
        let bindings = stable.get(entry.name.as_str()).unwrap_or(&empty);
        prepare_values(&mut entry.body, bindings);
    }
    program.retag()
}

fn stable_storage(module: &ModuleInterface) -> LookupMap<&str, LookupSet<BindingRef>> {
    let graph = &module.frame_graph;
    let resources: LookupMap<_, _> = graph
        .resources
        .iter()
        .enumerate()
        .flat_map(|(id, resource)| {
            resource.bindings.iter().filter_map(move |binding| {
                Some(((binding.pipeline_index, binding.set?, binding.binding?), id))
            })
        })
        .collect();

    // Buffer joins and loop carries can supply a second descriptor for the
    // same allocation at dispatch time. Keep their resources out of this proof.
    let joins = module.buffer_joins.iter().flat_map(|join| {
        std::iter::once(&join.binding).chain(join.source.bindings()).map(|source| (join.pipeline, source))
    });
    let carries = module.dispatch_loops.iter().flat_map(|region| {
        region.carried.iter().flat_map(|carry| {
            std::iter::once(&carry.binding)
                .chain(carry.initial.bindings())
                .chain(carry.next.bindings())
                .map(|source| (region.pipeline, source))
        })
    });
    let mut rebound = LookupSet::new();
    for (pipeline, source) in joins.chain(carries) {
        let ScalarSource::Binding { set, binding } = *source else {
            return LookupMap::new();
        };
        if let Some(&resource) = resources.get(&(pipeline, set, binding)) {
            rebound.insert(resource);
        }
    }

    let mut stable = LookupMap::new();
    for pass in &graph.passes {
        let Some(Pipeline::Compute(pipeline)) = module.pipelines.get(pass.pipeline_index) else {
            continue;
        };
        let Some(stage) = pipeline.stages.get(pass.stage_index) else {
            continue;
        };
        let writes: LookupSet<_> = pass.writes.iter().map(|access| access.resource).collect();
        let bindings = pass
            .reads
            .iter()
            .filter(|access| !writes.contains(&access.resource) && !rebound.contains(&access.resource))
            .filter_map(|access| graph.resources.get(access.resource))
            .flat_map(|resource| &resource.bindings)
            .filter(|binding| binding.pipeline_index == pass.pipeline_index)
            .filter_map(|binding| {
                let Binding::StorageBuffer {
                    set,
                    binding,
                    usage: BufferUsage::Intermediate,
                    ..
                } = pipeline.bindings.get(binding.binding_index)?
                else {
                    return None;
                };
                Some(BindingRef::new(*set, *binding))
            })
            .collect::<LookupSet<_>>();
        if !bindings.is_empty() {
            stable.insert(stage.entry_point.as_str(), bindings);
        }
    }
    stable
}

pub(crate) fn is_speculatable(data: &InstKind) -> bool {
    match data {
        InstKind::Op {
            tag: OpTag::Intrinsic { id, overload_idx },
            ..
        } => {
            let builtin = by_id(*id);
            builtin.raw.purity == Purity::Pure
                && builtin
                    .overloads()
                    .get(*overload_idx)
                    .is_some_and(|overload| overload.lowering.is_speculatable())
        }
        InstKind::Op {
            tag: OpTag::BinOp(op),
            ..
        } => !matches!(
            op,
            BinaryOperator::Divide
                | BinaryOperator::Remainder
                | BinaryOperator::FloorDivide
                | BinaryOperator::FloorRemainder
                | BinaryOperator::Power
                | BinaryOperator::ShiftLeft
                | BinaryOperator::ShiftRight
                | BinaryOperator::ShiftRightLogical
        ),
        InstKind::Op {
            tag:
                OpTag::Int(_)
                | OpTag::Uint(_)
                | OpTag::Float(_)
                | OpTag::Bool(_)
                | OpTag::Unit
                | OpTag::UnaryOp(_)
                | OpTag::Tuple
                | OpTag::Vector
                | OpTag::ArrayLit
                | OpTag::Project { .. }
                | OpTag::Materialize,
            ..
        } => true,
        _ => false,
    }
}

fn materialize_dynamic_index(
    function: &mut WynFunction,
    instruction: InstId,
    expressions: &mut HashMap<ValueId, ValueId>,
) {
    let InstKind::Op {
        tag: OpTag::Index,
        operands,
    } = &function.insts[instruction].data
    else {
        return;
    };
    let [ValueRef::Ssa(array), ValueRef::Ssa(index)] = operands.as_slice() else {
        return;
    };
    if let ValueDef::Inst { inst } = function.values[*index].def {
        if matches!(
            function.insts[inst].data,
            InstKind::Op {
                tag: OpTag::Int(_) | OpTag::Uint(_),
                ..
            }
        ) {
            return;
        }
    }
    let ty = &function.values[*array].ty;
    let is_scalar_array = ty.is_array()
        && ty.elem_type().is_some_and(|element| {
            matches!(
                element,
                Type::Constructed(
                    TypeName::Int(_) | TypeName::UInt(_) | TypeName::Float(_) | TypeName::Bool,
                    _
                )
            )
        });
    if !is_scalar_array || ty.array_variant().is_some_and(is_array_variant_view) || is_virtual_array(ty) {
        return;
    }
    let array = *array;
    let ty = ty.clone();
    // The array is an immutable SSA value. Place its addressable copy with
    // the definition so nested loops share it without crossing its lifetime.
    let materialized = *expressions.entry(array).or_insert_with(|| {
        function.append_floating_inst(
            function.entry,
            InstKind::Op {
                tag: OpTag::Materialize,
                operands: vec![array.into()],
            },
            ty,
        )
    });
    if let InstKind::Op { tag, operands } = &mut function.insts[instruction].data {
        *tag = OpTag::DynamicExtract;
        operands[0] = materialized.into();
    }
}

fn prepare_values(body: &mut FuncBody, stable: &LookupSet<BindingRef>) {
    // Concrete storage views carry authored writes, not optional copy elision.
    // Normalize them here, alongside the other representation lowering.
    for node in body.inner.insts.values_mut() {
        if let InstKind::Op { tag, operands } = &mut node.data {
            if let Some(source) = operands.first().and_then(|v| v.as_ssa()) {
                super::storage::lower_array_update(tag, &body.inner.values[source].ty);
            }
        }
    }
    forward_aggregate_reads(body, stable);
    // Backend preparation folds the control flow generated by storage lowering.
    let mut expressions = HashMap::new();
    let blocks = body.inner.blocks.keys().collect::<Vec<_>>();
    for block in blocks {
        for instruction in body.inner.blocks[block].insts.clone() {
            materialize_dynamic_index(&mut body.inner, instruction, &mut expressions);
        }
    }
}

/// A projection can read the original place while its contents are stable.
fn forward_aggregate_reads(body: &mut FuncBody, stable: &LookupSet<BindingRef>) {
    let mut stable_reads = HashMap::new();
    if !stable.is_empty()
        && !body.inner.insts.values().any(|node| {
            matches!(
                node.data,
                InstKind::Op {
                    tag: OpTag::Call(_),
                    ..
                }
            )
        })
    {
        let places: LookupSet<_> = body
            .inner
            .insts
            .values()
            .filter_map(|node| {
                let InstKind::ViewIndex { view, result, .. } = node.data else {
                    return None;
                };
                let binding = array_view_buffer(&body.value_ref_type(view))?;
                stable.contains(&binding).then_some(result)
            })
            .collect();
        for node in body.inner.insts.values() {
            if let (Some(value), InstKind::Load { place }) = (node.result, &node.data) {
                if places.contains(place) {
                    stable_reads.insert(value, *place);
                }
            }
        }
    }
    let mut forwarded = Vec::new();
    let blocks = body.inner.blocks.keys().collect::<Vec<_>>();
    for block in blocks {
        let mut loaded = HashMap::new();
        for instruction in std::mem::take(&mut body.inner.blocks[block].insts) {
            let node = body.inner.insts[instruction].clone();
            if let InstKind::Op { tag, operands } = &node.data {
                let aggregate = operands.first().and_then(|value| value.as_ssa());
                let projection = match tag {
                    // Vectors already load in one instruction. Forwarding their
                    // components can duplicate that load, or split it into several.
                    OpTag::Project { index }
                        if aggregate.is_some_and(|value| {
                            !matches!(&body.inner.values[value].ty, Type::Constructed(TypeName::Vec, _))
                        }) =>
                    {
                        Some(ValueRef::Const(ConstantValue::U32(*index)))
                    }
                    OpTag::Index | OpTag::DynamicExtract if operands.len() == 2 => {
                        let ty = body.value_ref_type(operands[0]);
                        ty.array_variant()
                            .is_some_and(crate::types::is_array_variant_composite)
                            .then_some(operands[1])
                    }
                    _ => None,
                };
                if let (Some(index), Some(&place), Some(result)) = (
                    projection,
                    aggregate.and_then(|value| loaded.get(&value).or_else(|| stable_reads.get(&value))),
                    node.result,
                ) {
                    let ty = body.inner.values[result].ty.clone();
                    let element = body.places.insert(PlaceInfo {
                        elem_ty: ty,
                        origin: PlaceOrigin::Instruction,
                    });
                    body.inner.append_void_inst(
                        block,
                        InstKind::PlaceIndex {
                            place,
                            index,
                            result: element,
                        },
                    );
                    body.inner.insts[instruction].data = InstKind::Load { place: element };
                    if aggregate.is_some_and(|value| stable_reads.contains_key(&value)) {
                        stable_reads.insert(result, element);
                    }
                    forwarded.extend(aggregate);
                }
            }
            let data = &body.inner.insts[instruction].data;
            if let (InstKind::Load { place }, Some(result)) = (data, node.result) {
                loaded.insert(result, *place);
            } else if matches!(
                data,
                InstKind::Op {
                    tag: OpTag::Intrinsic { .. },
                    ..
                }
            ) || (!is_speculatable(data)
                && !matches!(
                    data,
                    InstKind::Alloca { .. }
                        | InstKind::ViewIndex { .. }
                        | InstKind::PlaceIndex { .. }
                        | InstKind::OutputSlot { .. }
                ))
            {
                loaded.clear();
            }
            body.inner.blocks[block].insts.push(instruction);
        }
    }
    let uses = super::ValueUses::analyze(&body.inner);
    let dead = forwarded
        .into_iter()
        .filter(|value| uses.count(*value) == 0)
        .filter_map(|value| body.inner.inst_of_value(value))
        .collect::<crate::LookupSet<_>>();
    for block in body.inner.blocks.values_mut() {
        block.insts.retain(|inst| !dead.contains(inst));
    }
    for inst in dead {
        if let Some(node) = body.inner.insts.remove(inst) {
            if let Some(value) = node.result {
                body.inner.values.remove(value);
            }
        }
    }
}

#[cfg(test)]
#[path = "optimize_tests.rs"]
mod tests;
