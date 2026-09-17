//! Source interface facts and final ABI readout. No generated bodies are re-imported.
use crate::egglog::abi::{error, storage_type};
use crate::egglog::blocks::Storage;
use crate::egglog::data::{
    BlockId, BufferId, DispatchId, EntryData, EntryId, OutputData, OutputId, ParameterId, TypeData, TypeId,
};
use crate::egglog::planning::{number, rows};
use crate::egglog::{OptimizeError, Program, Scheduled};
use crate::interface::{
    Attribute, EntryInput, EntryInputKind, StorageBindingDecl, StorageLayout, StorageRole,
};
use crate::pipeline_descriptor::{BufferLen, DispatchSize, HostSizeInput, HostSizeScalar};
use crate::ssa::layout::{block_layout, storage_elem_stride, type_byte_size};
use crate::types::{Type, TypeExt, TypeName};
use crate::{BindingRef, ResourceAccess};
use egglog_engine::sort::S;
use egglog_engine::{EGraph, Error, FullState, Value as EggValue, Write};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use wyn_base::IdArena;

pub(in crate::egglog) fn facts(
    inputs: &BTreeMap<ParameterId, Vec<EntryInput>>,
    outputs: &IdArena<OutputId, OutputData>,
    entries: &IdArena<EntryId, EntryData>,
    types: &IdArena<TypeId, TypeData>,
    sink: &mut FullState<'_, '_>,
    uniforms: &mut Vec<HostSizeInput>,
) -> Result<(), Error> {
    for (&parameter, inputs) in inputs {
        let parent = sink.add("AbiParameter", i64::from(parameter.as_u32()))?;
        for (i, input) in inputs.iter().enumerate() {
            let value = if inputs.len() == 1 { parent } else { sink.add("AbiField", (parent, i as i64))? };
            if inputs.len() > 1 {
                sink.add("AbiFieldValue", (parent, i as i64, value))?;
            }
            if let Some(binding) = input.storage_binding() {
                if let Some(stride) = input.ty.elem_type().and_then(storage_elem_stride) {
                    let binding_key = sink.add(
                        "InputBinding",
                        (i64::from(binding.set), i64::from(binding.binding)),
                    )?;
                    sink.add("AbiStorage", (value, binding_key, i64::from(stride)))?;
                }
            }
            if let EntryInputKind::Uniform { binding } = input.kind {
                uniform(sink, value, &input.ty, binding, 0, &input.name, uniforms)?;
            }
        }
    }
    for (&id, output) in outputs {
        if let Some(Attribute::Storage { set, binding, .. }) =
            entries[output.entry].declaration.outputs.get(output.index).and_then(|o| o.attribute.as_ref())
        {
            let name = format!(
                "{}_output_{}",
                entries[output.entry].declaration.name, output.index
            );
            sink.add(
                "AbiOutputBinding",
                (
                    i64::from(id.as_u32()),
                    i64::from(*set),
                    i64::from(*binding),
                    name.as_str(),
                ),
            )?;
        }
    }
    for (&id, ty) in types {
        // Most source types are not buffer elements. Validate actual storage
        // types at readout; only import layouts that have a physical form.
        let Ok(ty) = storage_type(&ty.ty) else {
            continue;
        };
        if let Some(stride) = storage_elem_stride(&ty) {
            let ty = sink.add("TypeId", i64::from(id.as_u32()))?;
            sink.add("TypeStride", (ty, i64::from(stride)))?;
        }
    }
    Ok(())
}

fn uniform(
    sink: &mut FullState<'_, '_>,
    value: EggValue,
    ty: &Type,
    binding: BindingRef,
    offset: u32,
    name: &str,
    inputs: &mut Vec<HostSizeInput>,
) -> Result<(), Error> {
    let scalar = match ty {
        Type::Constructed(TypeName::Int(32), _) => Some(HostSizeScalar::I32),
        Type::Constructed(TypeName::UInt(32), _) => Some(HostSizeScalar::U32),
        Type::Constructed(TypeName::Float(32), _) => Some(HostSizeScalar::F32),
        _ => None,
    };
    if let Some(scalar) = scalar {
        sink.add("AbiUniform", (value, inputs.len() as i64))?;
        inputs.push(HostSizeInput {
            name: name.into(),
            set: binding.set,
            binding: binding.binding,
            offset,
            scalar,
        });
        return Ok(());
    }
    let Type::Constructed(kind, args) = ty else {
        return Ok(());
    };
    let fields: Vec<_> = if ty.is_vec() {
        let Some(element) = ty.elem_type() else {
            return Ok(());
        };
        (0..ty.vec_size().unwrap_or(0)).map(|_| element).collect()
    } else {
        args.iter().collect()
    };
    let layout = block_layout(ty, StorageLayout::Std140);
    for (i, field) in fields.into_iter().enumerate() {
        let delta = if ty.is_vec() {
            type_byte_size(field).and_then(|size| size.checked_mul(i as u32))
        } else {
            layout.as_ref().and_then(|l| l.member_offsets.get(i)).copied()
        };
        let Some(offset) = delta.and_then(|d| offset.checked_add(d)) else {
            continue;
        };
        let label = match kind {
            TypeName::Record(names) => names.0.get(i).cloned().unwrap_or_else(|| i.to_string()),
            _ if ty.is_vec() => ["x", "y", "z", "w"].get(i).copied().unwrap_or("?").to_string(),
            _ => i.to_string(),
        };
        let child = sink.add("AbiField", (value, i as i64))?;
        sink.add("AbiFieldValue", (value, i as i64, child))?;
        uniform(
            sink,
            child,
            field,
            binding,
            offset,
            &format!("{name}_{label}"),
            inputs,
        )?;
    }
    Ok(())
}

pub(in crate::egglog) fn read(
    graph: &EGraph,
    data: &mut Program<Scheduled>,
    buffers: &HashMap<EggValue, BufferId>,
    stages: &HashMap<EggValue, DispatchId>,
    entry_roots: &BTreeMap<EntryId, BlockId>,
    uniforms: &[HostSizeInput],
) -> Result<HashMap<EggValue, BlockId>, OptimizeError> {
    let mut pinned = BTreeMap::new();
    rows(graph, "AbiPinnedBinding", |a| {
        let id = buffers[&a[0]];
        let binding = BindingRef::new(number(graph, a[1])?, number(graph, a[2])?);
        let name = graph.value_to_base::<S>(a[3]).to_string();
        if pinned.insert(id, (binding, name)).is_some_and(|(old, _)| old != binding) {
            return Err(error(
                "shared output requires copies to distinct declared bindings",
            ));
        }
        Ok(())
    })?;
    let mut reserved: BTreeSet<_> =
        data.state.abi.inputs.values().flatten().filter_map(|i| i.descriptor_binding()).collect();
    reserved.extend(pinned.values().map(|(b, _)| *b));
    let mut next = data.programs.values().map(|p| p.next_auto_storage_binding).max().unwrap_or(0);
    // Numbering is deterministic serialization of the allocation and pin facts.
    for (&id, buffer) in &data.state.buffers {
        if buffer.storage != Storage::Device {
            continue;
        }
        let (binding, name) = if let Some(pinned) = pinned.remove(&id) {
            pinned
        } else {
            while reserved.contains(&BindingRef::new(0, next)) {
                next = next.checked_add(1).ok_or_else(|| error("too many bindings"))?;
            }
            let binding = BindingRef::new(0, next);
            next = next.checked_add(1).ok_or_else(|| error("too many bindings"))?;
            (binding, format!("egg_resource{}", id.as_u32()))
        };
        data.state.abi.bindings.insert(
            id,
            StorageBindingDecl {
                binding,
                elem_ty: storage_type(&buffer.element)?,
                role: StorageRole::Intermediate,
                logical_resource: Some(name),
                length: None,
            },
        );
    }
    let mut binding_ids = HashMap::new();
    let mut status = Ok(());
    graph.constructor_enodes_while("InputBinding", |e| {
        status = (|| {
            let binding = BindingRef::new(number(graph, e.children[0])?, number(graph, e.children[1])?);
            if binding_ids.insert(e.eclass, binding).is_some_and(|old| old != binding) {
                return Err(error("conflicting explicit bindings"));
            }
            Ok(())
        })();
        status.is_ok()
    })?;
    status?;
    graph.constructor_enodes("ResourceBinding", |e| {
        if let Some(id) = buffers.get(&e.children[0]) {
            if let Some(binding) = data.state.abi.bindings.get(id) {
                binding_ids.entry(e.eclass).or_insert(binding.binding);
            }
        }
    })?;
    let mut roots = HashMap::new();
    let mut status = Ok(());
    graph.constructor_enodes_while("EntryRoot", |e| {
        status = number(graph, e.children[0]).map(|entry| {
            roots.insert(e.eclass, entry_roots[&EntryId::from(entry)]);
        });
        status.is_ok()
    })?;
    status?;
    graph.constructor_enodes("KernelRoot", |e| {
        if let Some(&id) = stages.get(&e.children[0]) {
            roots.insert(e.eclass, data.state.dispatches[id].kernel);
        }
    })?;
    rows(graph, "AbiInvalidHost", |a| {
        data.state.unsupported_host = Some(entry_roots[&EntryId::from(number(graph, a[0])?)]);
        Ok(())
    })?;
    rows(graph, "AbiRoot", |a| {
        data.state.abi.root_accesses.entry(roots[&a[0]]).or_default();
        data.state.abi.roots.push((
            roots[&a[0]],
            EntryId::from(number(graph, a[1])?),
            [number(graph, a[2])?, number(graph, a[3])?, number(graph, a[4])?],
            graph.value_to_base::<bool>(a[5]),
        ));
        Ok(())
    })?;
    data.state.abi.roots.sort_by_key(|r| r.0);
    rows(graph, "AbiBufferBinding", |a| {
        if let Some(&id) = buffers.get(&a[0]) {
            data.state.abi.buffer_bindings.insert(id, binding_ids[&a[1]]);
        }
        Ok(())
    })?;
    let mut status = Ok(());
    graph.function_entries_while("AbiAccess", |entry| {
        status = (|| {
            let a = entry.inputs;
            let access = match graph.value_to_base::<i64>(entry.output) {
                1 => ResourceAccess::Read,
                2 => ResourceAccess::Write,
                3 => ResourceAccess::ReadWrite,
                _ => return Err(error("invalid access flags")),
            };
            data.state
                .abi
                .root_accesses
                .entry(roots[&a[0]])
                .or_default()
                .insert(binding_ids[&a[1]], access);
            Ok(())
        })();
        status.is_ok()
    })?;
    status?;
    let mut capacities = BTreeMap::new();
    rows(graph, "AbiFixedCapacity", |a| {
        capacities.insert(
            buffers[&a[0]],
            BufferLen::Fixed {
                bytes: graph.value_to_base::<i64>(a[1]) as u64,
            },
        );
        Ok(())
    })?;
    rows(graph, "AbiInputCapacity", |a| {
        capacities.insert(
            buffers[&a[0]],
            BufferLen::LikeInput {
                set: binding_ids[&a[1]].set,
                binding: binding_ids[&a[1]].binding,
                elem_bytes: number(graph, a[2])?,
                src_elem_bytes: number(graph, a[3])?,
            },
        );
        Ok(())
    })?;
    let mut capacity_inputs: BTreeMap<BufferId, BTreeSet<HostSizeInput>> = BTreeMap::new();
    rows(graph, "AbiCapacityInput", |a| {
        capacity_inputs
            .entry(buffers[&a[0]])
            .or_default()
            .insert(uniforms[number(graph, a[1])? as usize].clone());
        Ok(())
    })?;
    for (&id, binding) in &mut data.state.abi.bindings {
        let Some(stride) = storage_elem_stride(&binding.elem_ty) else {
            return Err(error("buffer element has no byte layout"));
        };
        // TODO: extend the shared ABI with checked capacity formulas; this is
        // the explicit host-provided policy for formulas it cannot yet encode.
        binding.length = Some(capacities.remove(&id).unwrap_or_else(|| BufferLen::HostProvided {
            inputs: capacity_inputs.remove(&id).unwrap_or_default().into_iter().collect(),
            elem_bytes: stride,
        }));
    }
    rows(graph, "AbiFixedGrid", |a| {
        data.state.dispatches[stages[&a[0]]].size = DispatchSize::Fixed {
            x: number(graph, a[1])?,
            y: number(graph, a[2])?,
            z: number(graph, a[3])?,
            explicit: true,
        };
        Ok(())
    })?;
    Ok(roots)
}
