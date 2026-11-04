//! Concrete SSA storage semantics for representation lowering and local reuse.

use super::ir::ValueDef;
use super::types::{FuncBody, InstKind, PlaceId, Terminator, ValueId, ValueRef};
use crate::builtins::catalog;
use crate::op::{OpTag, PureViewSource};
use crate::types::{
    is_array_variant_bounded, is_array_variant_composite, is_array_variant_view, Type, TypeExt, TypeName,
};
use crate::{FunctionId, LookupMap, LookupSet, ResourceAccess};

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

/// Physical provenance shared by allocation reuse and memory synchronization.
/// Unknown descriptors may name shared memory; external bindings cannot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum MemoryRoot {
    Allocation(PlaceId),
    Workgroup(u32),
    External,
    Unknown,
}

/// Transitive workgroup effects keep scalar helpers transparent to memory
/// analysis. Unknown callees retain conservative read/write effects.
pub(super) fn workgroup_call_effects<Tag, Context>(
    program: &super::types::Program<Tag, Context>,
) -> LookupMap<FunctionId, Option<ResourceAccess>> {
    let mut effects = LookupMap::default();
    for function in &program.functions {
        let roots = MemoryRoots::analyze(&function.body);
        let mut access = function.linkage_name.as_ref().map(|_| ResourceAccess::ReadWrite);
        for node in function.body.inner.insts.values() {
            if matches!(
                node.data,
                InstKind::Op {
                    tag: OpTag::Call(_),
                    ..
                }
            ) {
                continue;
            }
            let (reads, writes) = roots.access(&node.data, &LookupMap::default());
            for (xs, effect) in [(reads, ResourceAccess::Read), (writes, ResourceAccess::Write)] {
                if xs.iter().any(|r| matches!(r, MemoryRoot::Workgroup(_) | MemoryRoot::Unknown)) {
                    access = Some(access.map_or(effect, |old| old.merge(effect)));
                }
            }
        }
        // Returning a descriptor publishes its contents to the caller.
        if function.body.inner.blocks.values().any(|block| {
            roots
                .returned(&block.term)
                .iter()
                .any(|r| matches!(r, MemoryRoot::Workgroup(_) | MemoryRoot::Unknown))
        }) {
            access = Some(access.map_or(ResourceAccess::Read, |old| old.merge(ResourceAccess::Read)));
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
            return effects;
        }
    }
}

#[derive(Default)]
pub(super) struct MemoryRoots {
    values: LookupMap<ValueId, LookupSet<MemoryRoot>>,
    pub places: LookupMap<PlaceId, LookupSet<MemoryRoot>>,
}

fn contains_view(ty: &Type) -> bool {
    ty.array_variant().is_some_and(is_array_variant_view)
        || ty.elem_type().is_some_and(contains_view)
        || matches!(ty, Type::Constructed(TypeName::Tuple(_), xs) if xs.iter().any(contains_view))
}

fn extend<K: Copy + Eq + std::hash::Hash>(
    map: &mut LookupMap<K, LookupSet<MemoryRoot>>,
    key: K,
    roots: impl IntoIterator<Item = MemoryRoot>,
) -> bool {
    let entry = map.entry(key).or_default();
    let old = entry.len();
    entry.extend(roots);
    entry.len() != old
}

impl MemoryRoots {
    pub fn analyze(body: &FuncBody) -> Self {
        let mut result = Self::default();
        for (v, info) in &body.inner.values {
            if !contains_view(&info.ty) {
                continue;
            }
            let root = match info.def {
                ValueDef::Inst { inst } => match body.inner.insts[inst].data {
                    InstKind::Op {
                        tag: OpTag::StorageView(PureViewSource::Workgroup { id, .. }),
                        ..
                    } => Some(MemoryRoot::Workgroup(id)),
                    InstKind::Op {
                        tag: OpTag::StorageView(PureViewSource::Storage(_)),
                        ..
                    } => Some(MemoryRoot::External),
                    InstKind::Load { .. } => Some(MemoryRoot::Unknown),
                    _ => None,
                },
                ValueDef::FunctionParam { .. } => {
                    Some(if crate::types::array_view_buffer(&info.ty).is_some() {
                        MemoryRoot::External
                    } else {
                        MemoryRoot::Unknown
                    })
                }
                _ => None,
            };
            if let Some(root) = root {
                extend(&mut result.values, v, [root]);
            }
        }
        for node in body.inner.insts.values() {
            match node.data {
                InstKind::Alloca { result: place } => {
                    extend(&mut result.places, place, [MemoryRoot::Allocation(place)]);
                }
                InstKind::OutputSlot { result: place, .. } => {
                    extend(&mut result.places, place, [MemoryRoot::External]);
                }
                _ => {}
            }
        }
        // Resolve descriptor joins and derived places together, including loop backedges.
        loop {
            let mut changed = false;
            for node in body.inner.insts.values() {
                if let Some(v) = node.result.filter(|v| contains_view(body.get_value_type(*v))) {
                    if let InstKind::Op { tag, operands } = &node.data {
                        let storage = node.data.result_storage();
                        if matches!(storage, ResultStorage::Unknown)
                            && !matches!(tag, OpTag::StorageView(_))
                        {
                            changed |= extend(&mut result.values, v, [MemoryRoot::Unknown]);
                        }
                        let roots: Vec<_> = operands
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| storage.aliases_operand(*i))
                            .flat_map(|(_, v)| result.value(*v))
                            .collect();
                        changed |= extend(&mut result.values, v, roots);
                    }
                }
                let derived = match node.data {
                    InstKind::ViewIndex {
                        view, result: place, ..
                    } => Some((place, result.value(view))),
                    InstKind::PlaceIndex {
                        place, result: child, ..
                    } => Some((
                        child,
                        result.places.get(&place).into_iter().flatten().copied().collect(),
                    )),
                    _ => None,
                };
                if let Some((place, roots)) = derived {
                    changed |= extend(&mut result.places, place, roots);
                }
            }
            for block in body.inner.blocks.values() {
                let edges: Vec<_> = match &block.term {
                    Terminator::Branch { target, args } => vec![(*target, args)],
                    Terminator::CondBranch {
                        then_target,
                        then_args,
                        else_target,
                        else_args,
                        ..
                    } => vec![(*then_target, then_args), (*else_target, else_args)],
                    _ => vec![],
                };
                for (target, args) in edges {
                    for (&v, &arg) in body.inner.blocks[target].params.iter().zip(args) {
                        if contains_view(body.get_value_type(v)) {
                            let roots = result.value(arg);
                            changed |= extend(&mut result.values, v, roots);
                        }
                    }
                }
            }
            if !changed {
                for (v, info) in &body.inner.values {
                    if contains_view(&info.ty) && result.values.get(&v).is_none_or(|xs| xs.is_empty()) {
                        changed |= extend(&mut result.values, v, [MemoryRoot::Unknown]);
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        result
    }

    pub fn value(&self, value: ValueRef) -> Vec<MemoryRoot> {
        value.as_ssa().and_then(|v| self.values.get(&v)).into_iter().flatten().copied().collect()
    }

    pub fn returned(&self, term: &Terminator) -> Vec<MemoryRoot> {
        match term {
            Terminator::Return(Some(value)) => self.value(*value),
            _ => Vec::new(),
        }
    }

    pub fn access(
        &self,
        data: &InstKind,
        calls: &LookupMap<FunctionId, Option<ResourceAccess>>,
    ) -> (LookupSet<MemoryRoot>, LookupSet<MemoryRoot>) {
        let mut reads = LookupSet::default();
        let mut writes = LookupSet::default();
        let place =
            |p| self.places.get(&p).cloned().unwrap_or_else(|| [MemoryRoot::Unknown].into_iter().collect());
        match data {
            InstKind::Copy {
                destination, source, ..
            } => {
                reads.extend(self.value(*source));
                writes.extend(self.value(*destination));
            }
            InstKind::Load { place: p } => reads = place(*p),
            InstKind::Store { place: p, .. } => writes = place(*p),
            InstKind::Atomic { place: p, .. } => {
                reads = place(*p);
                writes = reads.clone();
            }
            InstKind::Op {
                tag: OpTag::Index | OpTag::DynamicExtract | OpTag::Materialize,
                operands,
            } => reads.extend(self.value(operands[0])),
            InstKind::Op {
                tag: OpTag::Call(callee),
                ..
            } => {
                if let Some(access) = calls.get(callee).copied().unwrap_or(Some(ResourceAccess::ReadWrite))
                {
                    if access.reads() {
                        reads.insert(MemoryRoot::Unknown);
                    }
                    if access.writes() {
                        writes.insert(MemoryRoot::Unknown);
                    }
                }
            }
            InstKind::Op {
                tag: OpTag::Intrinsic { id, .. },
                operands,
            } if *id != catalog().known().length
                && *id != catalog().known().storage_len
                && *id != catalog().known().slice =>
            {
                for &v in operands {
                    reads.extend(self.value(v));
                }
                writes = reads.clone();
            }
            _ => {}
        }
        (reads, writes)
    }
}
