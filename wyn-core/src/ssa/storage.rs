//! Concrete SSA storage semantics for representation lowering and local reuse.

use super::dataflow::Dataflow;
use super::ir::ValueDef;
use super::types::{FuncBody, InstKind, Terminator, ValueRef};
use crate::builtins::catalog;
use crate::op::{OpTag, PureViewSource};
use crate::types::{
    is_array_variant_bounded, is_array_variant_composite, is_array_variant_view, Type, TypeExt, TypeName,
};
use crate::{FunctionId, LookupMap, ResourceAccess};
use egglog_engine::{EGraph, FullState, RawValues, Read, Value, Write};
use slotmap::Key;

/// How an instruction obtains storage for an array-containing result.
pub(super) enum ResultStorage {
    /// A value copy with independent local contents (descriptors are excluded).
    Fresh,
    /// The backend may retain the selected operand's storage.
    Alias(usize),
    /// Aggregate construction may retain storage from any operand.
    AliasAll,
    /// No local allocation/alias contract is available.
    Unknown,
}

pub(super) enum StorageUse {
    /// Observes contents at the instruction, without retaining them.
    Read,
    /// Observes only shape, which an element update preserves.
    Metadata,
    /// Forwards storage into the result; its consumers determine liveness.
    Forward,
    /// May retain storage through a call, place, or an unmodeled operation.
    Escape,
}

impl ResultStorage {
    pub(super) fn aliases_operand(&self, index: usize) -> bool {
        matches!(self, Self::AliasAll) || matches!(self, Self::Alias(i) if *i == index)
    }
}

impl InstKind {
    pub(super) fn result_storage(&self) -> ResultStorage {
        match self {
            Self::Load { .. } => ResultStorage::Fresh,
            Self::Op { tag, .. } => match tag {
                OpTag::ArrayLit => ResultStorage::Fresh,
                OpTag::Tuple => ResultStorage::AliasAll,
                OpTag::Materialize | OpTag::Project { .. } | OpTag::Index | OpTag::DynamicExtract => {
                    ResultStorage::Alias(0)
                }
                OpTag::StorageView(PureViewSource::Inherited) => {
                    ResultStorage::Alias(self.value_uses().len() - 1)
                }
                OpTag::Intrinsic { id, .. } if *id == catalog().known().slice => ResultStorage::Alias(0),
                OpTag::Intrinsic { id, .. } if *id == catalog().known().array_with => ResultStorage::Fresh,
                OpTag::Intrinsic { id, .. } if *id == catalog().known().array_with_in_place => {
                    ResultStorage::Alias(0)
                }
                _ => ResultStorage::Unknown,
            },
            _ => ResultStorage::Unknown,
        }
    }

    /// Classify each operand independently of the result's allocation behavior.
    pub(super) fn storage_use(&self, operand: usize) -> StorageUse {
        match self {
            Self::Copy { .. } => {
                if operand == 1 {
                    StorageUse::Read
                } else {
                    StorageUse::Metadata
                }
            }
            Self::ViewIndex { .. } => StorageUse::Forward,
            Self::Op { tag, .. } => match tag {
                OpTag::Materialize | OpTag::Tuple => StorageUse::Forward,
                OpTag::StorageView(_) => StorageUse::Forward,
                OpTag::Intrinsic { id, .. } if *id == catalog().known().slice => StorageUse::Forward,
                OpTag::StorageViewLen => StorageUse::Metadata,
                OpTag::Index | OpTag::DynamicExtract | OpTag::Project { .. } | OpTag::ArrayLit => {
                    StorageUse::Read
                }
                OpTag::Intrinsic { id, .. }
                    if operand == 0
                        && (*id == catalog().known().length || *id == catalog().known().storage_len) =>
                {
                    StorageUse::Metadata
                }
                OpTag::Intrinsic { id, .. }
                    if *id == catalog().known().array_with
                        || *id == catalog().known().array_with_in_place =>
                {
                    StorageUse::Read
                }
                _ => StorageUse::Escape,
            },
            _ => StorageUse::Escape,
        }
    }
}

/// View updates already mean writes. Make that explicit when lowering concrete
/// representations, before any optional local-storage reuse is considered.
pub(super) fn lower_array_update<R, C>(tag: &mut OpTag<R, C>, source: &Type) {
    if let OpTag::Intrinsic { id, .. } = tag {
        if *id == catalog().known().array_with && source.array_variant().is_some_and(is_array_variant_view)
        {
            *id = catalog().known().array_with_in_place;
        }
    }
}

pub(super) fn contains_array(ty: &Type) -> bool {
    ty.is_array()
        || matches!(ty,
            Type::Constructed(TypeName::Tuple(_), fields)
                if fields.iter().any(contains_array))
}

/// Local value storage can be copied. Storage descriptors and unresolved array
/// representations cannot acquire mutation permission from this analysis.
pub(super) fn is_local_value(ty: &Type) -> bool {
    if let Some(variant) = ty.array_variant() {
        return (is_array_variant_composite(variant) || is_array_variant_bounded(variant))
            && ty.elem_type().is_some_and(is_local_value);
    }
    match ty {
        Type::Constructed(TypeName::Tuple(_), fields) => fields.iter().all(is_local_value),
        _ => true,
    }
}

pub(super) fn memory_id(key: impl Key) -> i64 {
    key.data().as_ffi() as i64
}

pub(super) fn local_array(ty: &Type) -> bool {
    ty.is_array()
        && is_local_value(ty)
        && ty.array_variant().is_some_and(is_array_variant_composite)
        && matches!(ty.array_size(), Some(Type::Constructed(TypeName::Size(_), _)))
}

fn contains_view(ty: &Type) -> bool {
    ty.array_variant().is_some_and(is_array_variant_view)
        || ty.elem_type().is_some_and(contains_view)
        || matches!(ty, Type::Constructed(TypeName::Tuple(_), xs) if xs.iter().any(contains_view))
}

impl Dataflow {
    pub fn memory(body: &FuncBody) -> Result<Self, egglog_engine::Error> {
        let mut graph = EGraph::default();
        graph.parse_and_run_program(None, include_str!("../egglog/flow.egg"))?;
        graph.parse_and_run_program(None, include_str!("storage.egg"))?;
        graph.update(|mut sink| {
            for (v, info) in &body.inner.values {
                if !contains_view(&info.ty) {
                    continue;
                }
                let value = sink.add("SsaValue", memory_id(v))?;
                sink.add("MemoryObject", value)?;
                let root = match info.def {
                    ValueDef::Inst { inst } => match &body.inner.insts[inst].data {
                        InstKind::Op {
                            tag: OpTag::StorageView(PureViewSource::Workgroup { id, .. }),
                            ..
                        } => Some(sink.add("MemoryWorkgroup", i64::from(*id))?),
                        InstKind::Op {
                            tag: OpTag::StorageView(PureViewSource::Storage(_)),
                            ..
                        } => Some(sink.add("MemoryExternal", RawValues(vec![]))?),
                        InstKind::Op { operands, .. } => {
                            let storage = body.inner.insts[inst].data.result_storage();
                            for (i, operand) in operands.iter().enumerate() {
                                if storage.aliases_operand(i) {
                                    if let Some(v) = operand.as_ssa() {
                                        let parent = sink.add("SsaValue", memory_id(v))?;
                                        sink.add("FlowFrom", (value, parent))?;
                                    }
                                }
                            }
                            None
                        }
                        _ => None,
                    },
                    ValueDef::FunctionParam { .. }
                        if crate::types::array_view_buffer(&info.ty).is_some() =>
                    {
                        Some(sink.add("MemoryExternal", RawValues(vec![]))?)
                    }
                    _ => None,
                };
                if let Some(root) = root {
                    sink.add("MemoryOrigin", (value, root))?;
                }
            }
            for place in body.places.keys() {
                let value = sink.add("SsaPlace", memory_id(place))?;
                sink.add("MemoryObject", value)?;
            }
            for node in body.inner.insts.values() {
                if let InstKind::Load { place } = node.data {
                    if node.result.is_some_and(|v| local_array(body.get_value_type(v))) {
                        let place = sink.add("SsaPlace", memory_id(place))?;
                        sink.add("MemoryLoaded", place)?;
                    }
                }
                match node.data {
                    InstKind::Alloca { result } | InstKind::OutputSlot { result, .. } => {
                        let place = sink.add("SsaPlace", memory_id(result))?;
                        let root = if matches!(node.data, InstKind::Alloca { .. }) {
                            sink.add("MemoryAllocation", memory_id(result))?
                        } else {
                            sink.add("MemoryExternal", RawValues(vec![]))?
                        };
                        sink.add("MemoryOrigin", (place, root))?;
                    }
                    InstKind::ViewIndex { view, result, .. } => {
                        if let Some(view) = view.as_ssa() {
                            let parent = sink.add("SsaValue", memory_id(view))?;
                            let place = sink.add("SsaPlace", memory_id(result))?;
                            sink.add("FlowFrom", (place, parent))?;
                        }
                    }
                    InstKind::PlaceIndex { place, result, .. } => {
                        let parent = sink.add("SsaPlace", memory_id(place))?;
                        let place = sink.add("SsaPlace", memory_id(result))?;
                        sink.add("FlowFrom", (place, parent))?;
                    }
                    _ => {}
                }
            }
            for block in body.inner.blocks.values() {
                for (target, args) in block.term.edges() {
                    for (&v, arg) in body.inner.blocks[target].params.iter().zip(args) {
                        if contains_view(body.get_value_type(v)) {
                            if let Some(arg) = arg.as_ssa() {
                                let value = sink.add("SsaValue", memory_id(v))?;
                                let parent = sink.add("SsaValue", memory_id(arg))?;
                                sink.add("FlowFrom", (value, parent))?;
                            }
                        }
                    }
                }
            }
            Ok::<_, egglog_engine::Error>(())
        })?;
        graph.parse_and_run_program(
            None,
            "(run-schedule (saturate memory-origins) memory-unknown (saturate memory-origins) memory-loaded)",
        )?;
        Ok(Self {
            graph,
            ..Self::default()
        })
    }

    pub fn memory_accesses(
        &mut self,
        body: &FuncBody,
        calls: &LookupMap<FunctionId, Option<ResourceAccess>>,
    ) -> Result<(), egglog_engine::Error> {
        self.graph.update(|mut sink| {
            for (inst, node) in &body.inner.insts {
                let point = sink.add("SsaInstruction", memory_id(inst))?;
                let mut value_access =
                    |value: ValueRef, access: ResourceAccess| -> Result<(), egglog_engine::Error> {
                        if let Some(value) = value.as_ssa() {
                            let value = sink.add("SsaValue", memory_id(value))?;
                            memory_use(&mut sink, point, value, access)?;
                        }
                        Ok(())
                    };
                match &node.data {
                    InstKind::Copy {
                        destination, source, ..
                    } => {
                        value_access(*source, ResourceAccess::Read)?;
                        value_access(*destination, ResourceAccess::Write)?;
                    }
                    InstKind::Op {
                        tag: OpTag::Index | OpTag::DynamicExtract | OpTag::Materialize,
                        operands,
                    } => value_access(operands[0], ResourceAccess::Read)?,
                    InstKind::Op {
                        tag: OpTag::Intrinsic { id, .. },
                        operands,
                    } if *id != catalog().known().length
                        && *id != catalog().known().storage_len
                        && *id != catalog().known().slice =>
                    {
                        for &v in operands {
                            value_access(v, ResourceAccess::ReadWrite)?;
                        }
                    }
                    InstKind::Op {
                        tag: OpTag::Call(callee),
                        ..
                    } => {
                        if let Some(access) =
                            calls.get(callee).copied().unwrap_or(Some(ResourceAccess::ReadWrite))
                        {
                            let root = sink.add("MemoryUnknown", RawValues(vec![]))?;
                            if access.reads() {
                                sink.add("MemoryAccess", (point, root, false))?;
                            }
                            if access.writes() {
                                sink.add("MemoryAccess", (point, root, true))?;
                            }
                        }
                    }
                    InstKind::Load { place }
                    | InstKind::Store { place, .. }
                    | InstKind::Atomic { place, .. } => {
                        let value = sink.add("SsaPlace", memory_id(*place))?;
                        let access = match node.data {
                            InstKind::Load { .. } => ResourceAccess::Read,
                            InstKind::Store { .. } => ResourceAccess::Write,
                            _ => ResourceAccess::ReadWrite,
                        };
                        memory_use(&mut sink, point, value, access)?;
                    }
                    _ => {}
                }
                for (operand, value) in node.data.value_uses().into_iter().enumerate() {
                    if matches!(node.data.storage_use(operand), StorageUse::Escape) {
                        if let Some(value) = value.as_ssa() {
                            let value = sink.add("SsaValue", memory_id(value))?;
                            sink.add("MemoryEscapes", value)?;
                        }
                    }
                }
            }
            for (block, data) in &body.inner.blocks {
                if let Terminator::Return(Some(value)) = data.term {
                    if let Some(value) = value.as_ssa() {
                        let point = sink.add("SsaReturn", memory_id(block))?;
                        let value = sink.add("SsaValue", memory_id(value))?;
                        memory_use(&mut sink, point, value, ResourceAccess::Read)?;
                        sink.add("MemoryEscapes", value)?;
                    }
                }
            }
            Ok::<_, egglog_engine::Error>(())
        })?;
        self.graph.parse_and_run_program(None, "(run-schedule (saturate memory-access))")?;
        Ok(())
    }
}

fn memory_use(
    sink: &mut FullState<'_, '_>,
    point: Value,
    value: Value,
    access: ResourceAccess,
) -> Result<(), egglog_engine::Error> {
    if access.reads() {
        sink.add("MemoryUse", (point, value, false))?;
    }
    if access.writes() {
        sink.add("MemoryUse", (point, value, true))?;
    }
    Ok(())
}

/// Transitive workgroup effects keep scalar helpers transparent to memory analysis.
pub(super) fn workgroup_call_effects<Tag, Context>(
    program: &super::types::Program<Tag, Context>,
) -> Result<LookupMap<FunctionId, Option<ResourceAccess>>, egglog_engine::Error> {
    let mut effects = LookupMap::default();
    let no_calls = program.functions.iter().map(|f| (f.id, None)).collect();
    for function in &program.functions {
        let mut flow = Dataflow::memory(&function.body)?;
        flow.memory_accesses(&function.body, &no_calls)?;
        let mut access = function.linkage_name.as_ref().map(|_| ResourceAccess::ReadWrite);
        for (write, effect) in [(false, ResourceAccess::Read), (true, ResourceAccess::Write)] {
            if flow.graph.read(|r| r.contains("MemorySharedAccess", write))? {
                access = Some(access.map_or(effect, |old| old.merge(effect)));
            }
        }
        effects.insert(function.id, access);
    }
    loop {
        let mut changed = false;
        for function in &program.functions {
            for node in function.body.inner.insts.values() {
                if let InstKind::Op {
                    tag: OpTag::Call(callee),
                    ..
                } = node.data
                {
                    if let Some(effect) =
                        effects.get(&callee).copied().unwrap_or(Some(ResourceAccess::ReadWrite))
                    {
                        let old = effects[&function.id];
                        let new = Some(old.map_or(effect, |old| old.merge(effect)));
                        changed |= old != new;
                        effects.insert(function.id, new);
                    }
                }
            }
        }
        if !changed {
            return Ok(effects);
        }
    }
}
