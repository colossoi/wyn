//! Whole-program EGIR container + per-body records.
//!
//! These are plain (non-generic) structs. State tracking happens at the
//! public API boundary via the semantic `EgirRaw` / `EgirSegmented` /
//! `EgirOptimized` / `EgirAllocated` newtypes in `crate::lib`, each
//! of which wraps an `SemanticProgram`.
//!
//! `SemanticProgram` carries, for each function and entry point, a per-body
//! `EGraph` + control-headers + alias map, plus program-level metadata
//! (constants, uniforms, storage decls, pipeline descriptor, extern stubs).

use crate::LookupMap;

use polytype::Type;

use crate::ast::{Span, TypeName};
use crate::interface;
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::ssa::types::{
    BlockId, Constant, ControlHeader, EntryInput, EntryOutput, ExecutionModel, Function,
};
use crate::types::TypeExt;

use std::collections::HashMap;

use super::parallelize::schedule::{KernelPlan, ValidatedKernelPlan};
use super::types::{EGraph, EffectToken, NodeId, RegionId};

#[cfg(test)]
#[path = "program_tests.rs"]
mod program_tests;

/// Name ↔ arena-index interner for callable regions.
///
/// Region identity is the assigned `RegionId` (a dense index). The textual
/// name is retained because it is the SSA `Call` ABI — a region lowers to a
/// named function, and operator/lane Calls reference it by that name. Interning
/// the same name twice returns the same index, so SegBody construction and the
/// function arena agree without a separate resolution pass.
#[derive(Clone, Debug, Default)]
pub struct RegionInterner {
    by_name: HashMap<String, RegionId>,
    names: Vec<String>,
}

impl RegionInterner {
    pub fn intern(&mut self, name: impl AsRef<str>) -> RegionId {
        let name = name.as_ref();
        if let Some(id) = self.by_name.get(name) {
            return *id;
        }
        let id = RegionId::from_index(self.names.len() as u32);
        self.names.push(name.to_string());
        self.by_name.insert(name.to_string(), id);
        id
    }

    pub fn get(&self, name: &str) -> Option<RegionId> {
        self.by_name.get(name).copied()
    }

    /// Recover the SSA function name backing a region index.
    pub fn name(&self, id: RegionId) -> &str {
        &self.names[id.index() as usize]
    }

    /// Recover the owned SSA names for a sequence of regions — e.g. a SOAC's
    /// map lanes or a reduction's per-operator combiners, which lower to
    /// `PureOp::Call`s by name.
    pub fn names(&self, ids: impl IntoIterator<Item = RegionId>) -> Vec<String> {
        ids.into_iter().map(|id| self.name(id).to_string()).collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SemanticOpId(pub u32);

/// Assign stable operation identities once, immediately after segmentation.
/// Graph projection copies these ids unchanged, so resource ownership and
/// dependency edges never depend on arena-local `NodeId`s.
pub fn assign_semantic_op_ids(inner: &mut SemanticProgram) {
    let mut next = inner
        .entry_points
        .iter()
        .flat_map(|entry| entry.graph.skeleton.blocks.iter())
        .flat_map(|(_, block)| block.side_effects.iter())
        .chain(
            inner
                .functions
                .iter()
                .flat_map(|function| function.graph.skeleton.blocks.iter())
                .flat_map(|(_, block)| block.side_effects.iter()),
        )
        .filter_map(|effect| effect.semantic_id.map(|id| id.0))
        .max()
        .map_or(0, |id| id + 1);
    let graphs = inner
        .entry_points
        .iter_mut()
        .map(|entry| &mut entry.graph)
        .chain(inner.functions.iter_mut().map(|function| &mut function.graph));
    for graph in graphs {
        for (_, block) in graph.skeleton.blocks.iter_mut() {
            for effect in &mut block.side_effects {
                if effect.semantic_id.is_none() {
                    effect.semantic_id = Some(SemanticOpId(next));
                    next += 1;
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SemanticDependencyKind {
    Value,
    Effect,
    Resource,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SemanticDependency {
    pub producer: SemanticOpId,
    pub consumer: SemanticOpId,
    pub kind: SemanticDependencyKind,
}

/// Callable body arena entry used by semantic SegOps.
#[derive(Clone)]
pub struct EgirRegion {
    pub name: String,
    pub params: Vec<(Type<TypeName>, String)>,
    pub return_ty: Type<TypeName>,
    pub graph: EGraph,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
}

impl EgirRegion {
    pub fn from_function(function: &EgirFunc) -> Self {
        Self {
            name: function.name.clone(),
            params: function.params.clone(),
            return_ty: function.return_ty.clone(),
            graph: function.graph.clone(),
            control_headers: function.control_headers.clone(),
        }
    }
}

pub use crate::ResourceId;

/// Stable identity of an entry while the program is still semantic EGIR.
/// Textual entry names are publication metadata and are deliberately not used
/// to connect plans back to their source entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SemanticEntryId(pub u32);

/// Stable identity of an entry input position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InputSlotId(pub usize);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogicalSize {
    FixedBytes(u64),
    PendingLikeBinding {
        binding: crate::BindingRef,
        elem_bytes: u32,
        src_elem_bytes: u32,
    },
    LikeResource {
        resource: ResourceId,
        elem_bytes: u32,
        src_elem_bytes: u32,
    },
    SameAsDispatch {
        elem_bytes: u32,
    },
    Unspecified,
}

/// Storage identity while EGIR crosses the allocation boundary. Raw and
/// optimized EGIR may still contain a descriptor-shaped reference emitted by
/// TLC. `plan_logical_resources` rewrites every such value to `Resource`
/// before constructing `EgirAllocated`; target planning rejects `Binding`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SemanticResourceRef {
    Binding(crate::BindingRef),
    Resource(ResourceId),
}

impl SemanticResourceRef {
    pub fn resource(self) -> Option<ResourceId> {
        match self {
            SemanticResourceRef::Binding(_) => None,
            SemanticResourceRef::Resource(resource) => Some(resource),
        }
    }

    pub fn binding(self) -> Option<crate::BindingRef> {
        match self {
            SemanticResourceRef::Binding(binding) => Some(binding),
            SemanticResourceRef::Resource(_) => None,
        }
    }
}

/// Entry-local use of a logical resource. Unlike `StorageBindingDecl`, this is
/// target independent after allocation and cannot assign a descriptor binding
/// to a compiler-created resource.
#[derive(Clone, Debug)]
pub struct SemanticResourceDecl {
    pub resource: SemanticResourceRef,
    pub role: interface::StorageRole,
    pub elem_ty: Type<TypeName>,
    pub size: LogicalSize,
}

impl SemanticResourceDecl {
    pub(crate) fn pending(declaration: interface::StorageBindingDecl) -> Self {
        Self {
            resource: SemanticResourceRef::Binding(declaration.binding),
            role: declaration.role,
            elem_ty: declaration.elem_ty,
            size: pending_logical_size(declaration.length.as_ref()),
        }
    }
}

/// Why a compiler-introduced resource exists. The kind fixes its physical
/// storage role and lets descriptor publication build the right
/// `StorageBindingDecl` without re-deriving it from the lowering site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompilerResourceKind {
    /// A pre-existing intermediate surfaced from a `StorageBindingDecl`
    /// (gather prepass / generic staging) — not owned by a Seg op.
    Staging,
    /// One per-accumulator partial buffer of a parallel `SegRed`.
    ReducePartial,
    /// The two scratch buffers of a parallel `SegScan`.
    ScanBlockSums,
    ScanBlockOffsets,
    /// A runtime `filter`'s compaction buffer and its paired length cell.
    FilterScratch,
    FilterLenCell,
    FilterFlags,
    FilterOffsets,
    FilterScanBlockSums,
    FilterScanBlockOffsets,
    /// Scalar result produced by a compiler-hoisted prepass and consumed by a
    /// later source entry phase.
    ScalarHandoff,
    /// One shared materialization for an array-valued SegMap with more than
    /// one semantic consumer.
    MultiConsumerArray,
}

impl CompilerResourceKind {
    /// The physical storage role a resource of this kind lowers to.
    pub fn role(self) -> interface::StorageRole {
        match self {
            CompilerResourceKind::FilterScratch
            | CompilerResourceKind::ScalarHandoff
            | CompilerResourceKind::MultiConsumerArray => interface::StorageRole::Output,
            _ => interface::StorageRole::Intermediate,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompilerResource {
    pub kind: CompilerResourceKind,
    pub role: interface::StorageRole,
    /// Semantic operation that owns the resource. Generic staging resources
    /// introduced before segmentation have no single owner.
    pub owner: Option<SemanticOpId>,
    /// Stable resource position within the owner (accumulator/lane/scratch
    /// index, depending on `kind`).
    pub slot: usize,
}

impl CompilerResource {
    pub fn new(kind: CompilerResourceKind, owner: Option<SemanticOpId>, slot: usize) -> Self {
        Self {
            role: kind.role(),
            kind,
            owner,
            slot,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ResourceOrigin {
    Host(crate::BindingRef),
    Compiler(CompilerResource),
}

#[derive(Clone, Debug)]
pub struct LogicalResource {
    pub id: ResourceId,
    pub origin: ResourceOrigin,
    pub elem_ty: Type<TypeName>,
    pub size: LogicalSize,
}

/// Build the authoritative logical-resource manifest at the allocation
/// boundary. Host storage (entry inputs/outputs plus pre-existing compiler
/// intermediates) is mirrored as today; then every parallel `SegRed`/`SegScan`
/// contributes owner-tagged scratch `Compiler` resources. Terminal lowering
/// resolves them directly from the manifest instead of storing phase-local
/// resource ids on semantic Seg operations.
pub fn plan_logical_resources(inner: &mut SemanticProgram, binding_ids: &mut crate::IdSource<u32>) {
    let mut compiler_origins = super::multi_consumer::run(inner, binding_ids);
    compiler_origins.extend(gather_prepass_resources(inner));
    let scalar_handoffs = scalar_handoff_resources(inner);
    compiler_origins.extend(scalar_handoffs);
    let filter_kinds = filter_resource_kinds(inner);
    compiler_origins.extend(filter_kinds);
    let mut filter_work = allocate_filter_work_resources(inner, binding_ids, &mut compiler_origins);
    let mut pending = mirror_storage_resources(inner, &compiler_origins);
    pending.sort_by_key(|(binding, _)| (binding.set, binding.binding));
    for (index, (_, resource)) in pending.iter_mut().enumerate() {
        resource.id = ResourceId(index as u32);
    }
    for (binding, mut resource) in filter_work.drain(..) {
        resource.id = ResourceId(pending.len() as u32);
        pending.push((binding, resource));
    }
    let mut scratch = super::parallelize::enumerate_seg_scratch(inner, binding_ids, pending.len() as u32);
    pending.append(&mut scratch);
    let by_binding =
        pending.iter().map(|(binding, resource)| (*binding, resource.id)).collect::<HashMap<_, _>>();
    inner.resources = pending.into_iter().map(|(_, resource)| resource).collect();
    normalize_semantic_resource_references(inner, &by_binding);
    if cfg!(debug_assertions) {
        verify_allocated_resources(inner).expect("invalid allocated semantic resources");
    }
}

fn gather_prepass_resources(inner: &SemanticProgram) -> HashMap<crate::BindingRef, CompilerResource> {
    let mut resources = HashMap::new();
    for entry in &inner.entry_points {
        if entry.origin != interface::EntryOrigin::GatherPrepass {
            continue;
        }
        for (slot, declaration) in entry.resource_declarations.iter().enumerate() {
            if declaration.role != interface::StorageRole::Output {
                continue;
            }
            if let Some(binding) = pending_binding(declaration.resource) {
                resources.insert(
                    binding,
                    CompilerResource::new(CompilerResourceKind::Staging, None, slot),
                );
            }
        }
    }
    resources
}

/// Replace descriptor-shaped storage identities in semantic graphs with their
/// target-independent `ResourceId`. Entry ABI declarations retain host
/// bindings; executable graph values do not.
fn normalize_semantic_resource_references(
    inner: &mut SemanticProgram,
    by_binding: &HashMap<crate::BindingRef, ResourceId>,
) {
    let compiler_resources = inner
        .resources
        .iter()
        .filter_map(|resource| {
            matches!(resource.origin, ResourceOrigin::Compiler(_)).then_some(resource.id)
        })
        .collect::<std::collections::HashSet<_>>();
    for entry in &mut inner.entry_points {
        for input in &entry.inputs {
            let Some(binding) = input.storage_binding else {
                continue;
            };
            let resource = *by_binding.get(&binding).expect("entry input must be in manifest");
            if !entry
                .resource_declarations
                .iter()
                .any(|declaration| resolves_to_resource(declaration.resource, resource, by_binding))
            {
                entry.resource_declarations.push(SemanticResourceDecl {
                    resource: SemanticResourceRef::Resource(resource),
                    role: interface::StorageRole::Input,
                    elem_ty: input.ty.elem_type().cloned().unwrap_or_else(|| input.ty.clone()),
                    size: pending_logical_size(input.length.as_ref()),
                });
            }
        }
        for output in &entry.outputs {
            let Some(binding) = output.storage_binding else {
                continue;
            };
            let resource = *by_binding.get(&binding).expect("entry output must be in manifest");
            if !entry
                .resource_declarations
                .iter()
                .any(|declaration| resolves_to_resource(declaration.resource, resource, by_binding))
            {
                entry.resource_declarations.push(SemanticResourceDecl {
                    resource: SemanticResourceRef::Resource(resource),
                    role: interface::StorageRole::Output,
                    elem_ty: output.ty.elem_type().cloned().unwrap_or_else(|| output.ty.clone()),
                    size: pending_logical_size(output.length.as_ref()),
                });
            }
        }
        for declaration in &mut entry.resource_declarations {
            if let SemanticResourceRef::Binding(binding) = declaration.resource {
                declaration.resource = SemanticResourceRef::Resource(
                    *by_binding.get(&binding).expect("entry resource declaration must be in manifest"),
                );
            }
            normalize_logical_size(&mut declaration.size, by_binding);
        }
        for input in &mut entry.inputs {
            if input
                .storage_binding
                .and_then(|binding| by_binding.get(&binding))
                .is_some_and(|resource| compiler_resources.contains(resource))
            {
                input.storage_binding = None;
            }
        }
        let mut output_slots = vec![None; entry.outputs.len()];
        let mut host_outputs = Vec::with_capacity(entry.outputs.len());
        for (slot, output) in std::mem::take(&mut entry.outputs).into_iter().enumerate() {
            let compiler_output = output
                .storage_binding
                .and_then(|binding| by_binding.get(&binding))
                .is_some_and(|resource| compiler_resources.contains(resource));
            if !compiler_output {
                output_slots[slot] = Some(host_outputs.len());
                host_outputs.push(output);
            }
        }
        entry.outputs = host_outputs;
        entry.output_routes.retain_mut(|route| {
            let Some(slot) = output_slots.get(route.slot.0).copied().flatten() else {
                return false;
            };
            route.slot = OutputSlotId(slot);
            true
        });
    }
    for entry in &mut inner.entry_points {
        normalize_graph_resources(&mut entry.graph, by_binding);
    }
    for function in &mut inner.functions {
        normalize_graph_resources(&mut function.graph, by_binding);
    }
    for region in inner.regions.values_mut() {
        normalize_graph_resources(&mut region.graph, by_binding);
    }
    normalize_structural_resources(inner, by_binding);
}

fn resolves_to_resource(
    reference: SemanticResourceRef,
    expected: ResourceId,
    by_binding: &HashMap<crate::BindingRef, ResourceId>,
) -> bool {
    match reference {
        SemanticResourceRef::Binding(binding) => by_binding.get(&binding).copied() == Some(expected),
        SemanticResourceRef::Resource(resource) => resource == expected,
    }
}

fn normalize_logical_size(size: &mut LogicalSize, by_binding: &HashMap<crate::BindingRef, ResourceId>) {
    let LogicalSize::PendingLikeBinding {
        binding,
        elem_bytes,
        src_elem_bytes,
    } = *size
    else {
        return;
    };
    *size = LogicalSize::LikeResource {
        resource: *by_binding.get(&binding).expect("resource-size source must be in manifest"),
        elem_bytes,
        src_elem_bytes,
    };
}

fn normalize_graph_resources(graph: &mut EGraph, by_binding: &HashMap<crate::BindingRef, ResourceId>) {
    let logical_lens = by_binding
        .iter()
        .map(|(binding, resource)| {
            let len = graph.intern_pure(
                super::types::PureOp::ResourceLen(*resource),
                smallvec::smallvec![],
                Type::Constructed(TypeName::UInt(32), vec![]),
            );
            (*binding, len)
        })
        .collect::<HashMap<_, _>>();
    let pure_nodes = graph.nodes.keys().collect::<Vec<_>>();
    for node in pure_nodes {
        graph.update_pure_node(node, |op, operands| {
            if let super::types::PureOp::StorageView(crate::op::PureViewSource::Storage(binding)) = op {
                if let Some(resource) = by_binding.get(binding) {
                    if operands.len() >= 2 {
                        operands[1] = logical_lens[binding];
                    }
                    *op = super::types::PureOp::StorageView(crate::op::PureViewSource::Resource(*resource));
                }
            }
        });
    }
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            if let super::types::SideEffectKind::Inst(crate::ssa::types::InstKind::Op { tag, .. }) =
                &mut effect.kind
            {
                if let crate::op::OpTag::StorageView(crate::op::PureViewSource::Storage(binding)) = tag {
                    if let Some(resource) = by_binding.get(binding) {
                        if effect.operand_nodes.len() >= 2 {
                            effect.operand_nodes[1] = logical_lens[binding];
                        }
                        *tag =
                            crate::op::OpTag::StorageView(crate::op::PureViewSource::Resource(*resource));
                    }
                }
            }
            if let super::types::SideEffectKind::Soac(soac) = &mut effect.kind {
                soac.visit_resource_refs_mut(|reference| {
                    if let SemanticResourceRef::Binding(binding) = *reference {
                        *reference = SemanticResourceRef::Resource(
                            *by_binding.get(&binding).expect("SOAC resource must be in manifest"),
                        );
                    }
                });
                soac.visit_types_mut(|ty| normalize_type_resources(ty, by_binding));
            }
        }
    }
    let nodes = graph.types.keys().copied().collect::<Vec<_>>();
    for node in nodes {
        let mut ty = graph.types[&node].clone();
        normalize_type_resources(&mut ty, by_binding);
        graph.retype_node(node, ty);
    }
}

fn normalize_structural_resources(
    inner: &mut SemanticProgram,
    by_binding: &HashMap<crate::BindingRef, ResourceId>,
) {
    for resource in &mut inner.resources {
        normalize_logical_size(&mut resource.size, by_binding);
        normalize_type_resources(&mut resource.elem_ty, by_binding);
    }
    for entry in &mut inner.entry_points {
        for input in &mut entry.inputs {
            normalize_type_resources(&mut input.ty, by_binding);
        }
        for output in &mut entry.outputs {
            normalize_type_resources(&mut output.ty, by_binding);
        }
        for (ty, _) in &mut entry.params {
            normalize_type_resources(ty, by_binding);
        }
        normalize_type_resources(&mut entry.return_ty, by_binding);
        for declaration in &mut entry.resource_declarations {
            normalize_type_resources(&mut declaration.elem_ty, by_binding);
        }
    }
    for function in &mut inner.functions {
        for (ty, _) in &mut function.params {
            normalize_type_resources(ty, by_binding);
        }
        normalize_type_resources(&mut function.return_ty, by_binding);
    }
    for region in inner.regions.values_mut() {
        for (ty, _) in &mut region.params {
            normalize_type_resources(ty, by_binding);
        }
        normalize_type_resources(&mut region.return_ty, by_binding);
    }
}

fn normalize_type_resources(ty: &mut Type<TypeName>, by_binding: &HashMap<crate::BindingRef, ResourceId>) {
    let Type::Constructed(name, arguments) = ty else {
        return;
    };
    if let TypeName::Buffer(binding) = *name {
        *name = TypeName::Resource(
            *by_binding.get(&binding).expect("buffer type resource must be in manifest"),
        );
    }
    if let TypeName::Sum(variants) = name {
        for (_, fields) in variants {
            for field in fields {
                normalize_type_resources(field, by_binding);
            }
        }
    }
    for argument in arguments {
        normalize_type_resources(argument, by_binding);
    }
}

/// Resolve semantic resource references immediately after validation and
/// before any physical graph transformation or backend pass runs.
pub fn physicalize_resource_references(
    inner: &mut SemanticProgram,
    bindings: &PhysicalResourceTable,
) -> Result<(), String> {
    for entry in &mut inner.entry_points {
        physicalize_graph_resources(&mut entry.graph, bindings)?;
    }
    for function in &mut inner.functions {
        physicalize_graph_resources(&mut function.graph, bindings)?;
    }
    for region in inner.regions.values_mut() {
        physicalize_graph_resources(&mut region.graph, bindings)?;
    }
    physicalize_structural_types(inner, bindings)?;
    Ok(())
}

fn physicalize_graph_resources(graph: &mut EGraph, bindings: &PhysicalResourceTable) -> Result<(), String> {
    let pure_nodes = graph.nodes.keys().collect::<Vec<_>>();
    let mut missing = None;
    for node in pure_nodes {
        let resource_len = match graph.nodes.get(node) {
            Some(super::types::ENode::Pure {
                op: super::types::PureOp::ResourceLen(resource),
                ..
            }) => Some(*resource),
            _ => None,
        };
        if let Some(resource) = resource_len {
            let Some(binding) = bindings.binding(resource) else {
                missing = Some(resource);
                continue;
            };
            let set = super::graph_ops::intern_u32(graph, binding.set, None);
            let slot = super::graph_ops::intern_u32(graph, binding.binding, None);
            graph.replace_pure_node(
                node,
                super::types::PureOp::Intrinsic {
                    id: crate::builtins::catalog().known().storage_len,
                    overload_idx: 0,
                },
                smallvec::smallvec![set, slot],
            );
            continue;
        }
        graph.update_pure_node(node, |op, _| {
            if let super::types::PureOp::StorageView(crate::op::PureViewSource::Resource(resource)) = op {
                if let Some(binding) = bindings.binding(*resource) {
                    *op = super::types::PureOp::StorageView(crate::op::PureViewSource::Storage(binding));
                } else {
                    missing = Some(*resource);
                }
            }
        });
    }
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            if let super::types::SideEffectKind::Inst(crate::ssa::types::InstKind::Op { tag, .. }) =
                &mut effect.kind
            {
                if let crate::op::OpTag::StorageView(crate::op::PureViewSource::Resource(resource)) = tag {
                    let Some(binding) = bindings.binding(*resource) else {
                        return Err(format!(
                            "semantic resource {:?} has no physical binding",
                            resource
                        ));
                    };
                    *tag = crate::op::OpTag::StorageView(crate::op::PureViewSource::Storage(binding));
                }
            }
            if let super::types::SideEffectKind::Soac(soac) = &mut effect.kind {
                let mut error = None;
                soac.visit_resource_refs_mut(|reference| {
                    if let SemanticResourceRef::Resource(resource) = *reference {
                        if let Some(binding) = bindings.binding(resource) {
                            *reference = SemanticResourceRef::Binding(binding);
                        } else {
                            error = Some(resource);
                        }
                    }
                });
                if let Some(resource) = error {
                    return Err(format!(
                        "semantic resource {:?} has no physical binding",
                        resource
                    ));
                }
                soac.visit_types_mut(|ty| physicalize_type_resources(ty, bindings));
            }
        }
    }
    if let Some(resource) = missing {
        return Err(format!(
            "semantic resource {:?} has no physical binding",
            resource
        ));
    }
    let nodes = graph.types.keys().copied().collect::<Vec<_>>();
    for node in nodes {
        let mut ty = graph.types[&node].clone();
        physicalize_type_resources(&mut ty, bindings);
        graph.retype_node(node, ty);
    }
    Ok(())
}

fn physicalize_structural_types(
    inner: &mut SemanticProgram,
    bindings: &PhysicalResourceTable,
) -> Result<(), String> {
    for entry in &mut inner.entry_points {
        for input in &mut entry.inputs {
            physicalize_type_resources(&mut input.ty, bindings);
        }
        for output in &mut entry.outputs {
            physicalize_type_resources(&mut output.ty, bindings);
        }
        for (ty, _) in &mut entry.params {
            physicalize_type_resources(ty, bindings);
        }
        physicalize_type_resources(&mut entry.return_ty, bindings);
        for declaration in &mut entry.resource_declarations {
            physicalize_type_resources(&mut declaration.elem_ty, bindings);
        }
    }
    for function in &mut inner.functions {
        for (ty, _) in &mut function.params {
            physicalize_type_resources(ty, bindings);
        }
        physicalize_type_resources(&mut function.return_ty, bindings);
    }
    for region in inner.regions.values_mut() {
        for (ty, _) in &mut region.params {
            physicalize_type_resources(ty, bindings);
        }
        physicalize_type_resources(&mut region.return_ty, bindings);
    }
    Ok(())
}

fn physicalize_type_resources(ty: &mut Type<TypeName>, bindings: &PhysicalResourceTable) {
    let Type::Constructed(name, arguments) = ty else {
        return;
    };
    if let TypeName::Resource(resource) = *name {
        *name = TypeName::Buffer(
            bindings.binding(resource).expect("semantic type resource must have a physical binding"),
        );
    }
    if let TypeName::Sum(variants) = name {
        for (_, fields) in variants {
            for field in fields {
                physicalize_type_resources(field, bindings);
            }
        }
    }
    for argument in arguments {
        physicalize_type_resources(argument, bindings);
    }
}

fn allocate_filter_work_resources(
    inner: &mut SemanticProgram,
    binding_ids: &mut crate::IdSource<u32>,
    compiler_origins: &mut HashMap<crate::BindingRef, CompilerResource>,
) -> Vec<(crate::BindingRef, LogicalResource)> {
    use super::types::{EgirSoac, FilterOutput, FilterState, FilterWorkBuffers, SegExtent, SideEffectKind};
    let mut used: std::collections::HashSet<_> = inner
        .entry_points
        .iter()
        .flat_map(|entry| {
            entry
                .inputs
                .iter()
                .filter_map(|input| input.storage_binding)
                .chain(entry.outputs.iter().filter_map(|output| output.storage_binding))
                .chain(
                    entry
                        .resource_declarations
                        .iter()
                        .filter_map(|declaration| pending_binding(declaration.resource)),
                )
        })
        .collect();
    let mut resources = Vec::new();
    for entry in &inner.entry_points {
        for (_, block) in &entry.graph.skeleton.blocks {
            for effect in &block.side_effects {
                let SideEffectKind::Soac(EgirSoac::Filter {
                    state: FilterState::Semantic { space },
                    output: FilterOutput::Runtime { .. },
                    ..
                }) = &effect.kind
                else {
                    continue;
                };
                let mut next_binding = || loop {
                    let binding =
                        crate::BindingRef::new(super::from_tlc::AUTO_STORAGE_SET, binding_ids.next_id());
                    if used.insert(binding) {
                        break binding;
                    }
                };
                let buffers = FilterWorkBuffers {
                    flags: SemanticResourceRef::Binding(next_binding()),
                    offsets: SemanticResourceRef::Binding(next_binding()),
                    block_sums: SemanticResourceRef::Binding(next_binding()),
                    block_offsets: SemanticResourceRef::Binding(next_binding()),
                };
                let element_count_size = match space.dims.first() {
                    Some(SegExtent::Fixed(count)) if space.dims.len() == 1 => {
                        LogicalSize::FixedBytes(*count as u64 * 4)
                    }
                    Some(SegExtent::ResourceLength {
                        resource, elem_bytes, ..
                    }) if space.dims.len() == 1 => LogicalSize::PendingLikeBinding {
                        binding: pending_binding(*resource)
                            .expect("filter work is allocated before resource normalization"),
                        elem_bytes: 4,
                        src_elem_bytes: *elem_bytes,
                    },
                    _ => LogicalSize::SameAsDispatch { elem_bytes: 4 },
                };
                // The scan phase runs a fixed worker grid
                // (`FILTER_SCAN_GROUPS * REDUCE_PHASE1_WIDTH` workers), so its
                // per-worker `block_sums`/`block_offsets` have a fixed length
                // independent of the input — which bounds the serial phase-2
                // scan and decouples the buffer from any stage's dispatch.
                let worker_count_size = LogicalSize::FixedBytes(
                    (super::parallelize::FILTER_SCAN_GROUPS * super::parallelize::REDUCE_PHASE1_WIDTH)
                        as u64
                        * 4,
                );
                let owner = effect.semantic_id;
                for (slot, (binding, kind, size)) in [
                    (
                        buffers.flags,
                        CompilerResourceKind::FilterFlags,
                        element_count_size.clone(),
                    ),
                    (
                        buffers.offsets,
                        CompilerResourceKind::FilterOffsets,
                        element_count_size.clone(),
                    ),
                    (
                        buffers.block_sums,
                        CompilerResourceKind::FilterScanBlockSums,
                        worker_count_size.clone(),
                    ),
                    (
                        buffers.block_offsets,
                        CompilerResourceKind::FilterScanBlockOffsets,
                        worker_count_size.clone(),
                    ),
                ]
                .into_iter()
                .enumerate()
                {
                    let binding = pending_binding(binding)
                        .expect("filter work is allocated before resource normalization");
                    let compiler = CompilerResource::new(kind, owner, slot);
                    compiler_origins.insert(binding, compiler.clone());
                    resources.push((
                        binding,
                        LogicalResource {
                            id: ResourceId(0),
                            origin: ResourceOrigin::Compiler(compiler),
                            elem_ty: Type::Constructed(TypeName::UInt(32), vec![]),
                            size,
                        },
                    ));
                }
            }
        }
    }
    resources
}

/// Bindings introduced by TLC scalar-prepass hoisting already have a stable
/// physical ABI, but are compiler-owned logical resources rather than host
/// inputs. Record that ownership at the allocation boundary so schedule and
/// descriptor publication do not have to infer it again.
fn scalar_handoff_resources(inner: &SemanticProgram) -> HashMap<crate::BindingRef, CompilerResource> {
    let consumer_inputs: std::collections::HashSet<_> = inner
        .entry_points
        .iter()
        .flat_map(|entry| {
            entry.resource_declarations.iter().filter_map(|declaration| {
                (declaration.role == interface::StorageRole::Input)
                    .then(|| pending_binding(declaration.resource))
                    .flatten()
            })
        })
        .collect();
    let mut resources = HashMap::new();
    for entry in &inner.entry_points {
        if entry.origin != interface::EntryOrigin::ScalarPrepass {
            continue;
        }
        let owner = entry
            .graph
            .skeleton
            .blocks
            .iter()
            .find_map(|(_, block)| block.side_effects.iter().find_map(|effect| effect.semantic_id));
        for declaration in &entry.resource_declarations {
            if declaration.role == interface::StorageRole::Output
                && matches!(declaration.size, LogicalSize::Unspecified)
                && pending_binding(declaration.resource)
                    .is_some_and(|binding| consumer_inputs.contains(&binding))
            {
                let binding = pending_binding(declaration.resource).unwrap();
                resources.insert(
                    binding,
                    CompilerResource::new(CompilerResourceKind::ScalarHandoff, owner.clone(), 0),
                );
            }
        }
    }
    resources
}

/// Runtime `filter` bindings, classified so the mirror gives them a precise
/// `CompilerResourceKind` rather than generic `Staging`.
fn filter_resource_kinds(inner: &SemanticProgram) -> HashMap<crate::BindingRef, CompilerResource> {
    let mut kinds = HashMap::new();
    for entry in &inner.entry_points {
        for (_, block) in &entry.graph.skeleton.blocks {
            for effect in &block.side_effects {
                if let super::types::SideEffectKind::Soac(super::types::EgirSoac::Filter {
                    output,
                    state,
                    ..
                }) = &effect.kind
                {
                    let owner = effect.semantic_id;
                    if let super::types::FilterOutput::Runtime { scratch, length } = output {
                        if let Some(scratch) = pending_binding(*scratch) {
                            kinds.insert(
                                scratch,
                                CompilerResource::new(CompilerResourceKind::FilterScratch, owner, 0),
                            );
                        }
                        if let super::types::RuntimeFilterLength::EntryOutput(len) = length {
                            if let Some(len) = pending_binding(*len) {
                                kinds.insert(
                                    len,
                                    CompilerResource::new(CompilerResourceKind::FilterLenCell, owner, 1),
                                );
                            }
                        }
                    }
                    if let super::types::FilterState::Scheduled { plan, .. } = state {
                        let work = match plan {
                            super::types::FilterPlan::Serial => None,
                            super::types::FilterPlan::Flags(work)
                            | super::types::FilterPlan::Scan(work)
                            | super::types::FilterPlan::Scatter(work) => Some(work),
                        };
                        if let Some(work) = work {
                            for (resource, kind, slot) in [
                                (work.flags, CompilerResourceKind::FilterFlags, 2),
                                (work.offsets, CompilerResourceKind::FilterOffsets, 3),
                                (work.block_sums, CompilerResourceKind::FilterScanBlockSums, 4),
                                (
                                    work.block_offsets,
                                    CompilerResourceKind::FilterScanBlockOffsets,
                                    5,
                                ),
                            ] {
                                if let Some(binding) = pending_binding(resource) {
                                    kinds.insert(binding, CompilerResource::new(kind, owner, slot));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    kinds
}

/// Mirror entry inputs, outputs, and declared storage bindings into logical
/// resources (deduped by binding). `Intermediate`-role declarations become
/// `Compiler` resources; a filter length cell is tagged precisely.
fn mirror_storage_resources(
    inner: &SemanticProgram,
    filter_kinds: &HashMap<crate::BindingRef, CompilerResource>,
) -> Vec<(crate::BindingRef, LogicalResource)> {
    let mut resources: LookupMap<crate::BindingRef, LogicalResource> = LookupMap::new();
    let resource =
        |binding: crate::BindingRef, elem_ty: Type<TypeName>, size: LogicalSize| LogicalResource {
            id: ResourceId(0),
            origin: filter_kinds
                .get(&binding)
                .cloned()
                .map(ResourceOrigin::Compiler)
                .unwrap_or(ResourceOrigin::Host(binding)),
            elem_ty,
            size,
        };
    for entry in &inner.entry_points {
        for input in &entry.inputs {
            if let Some(binding) = input.storage_binding {
                resources.entry(binding).or_insert_with(|| {
                    let mut logical = resource(
                        binding,
                        input.ty.elem_type().cloned().unwrap_or_else(|| input.ty.clone()),
                        pending_logical_size(input.length.as_ref()),
                    );
                    if !filter_kinds.contains_key(&binding) {
                        logical.origin = ResourceOrigin::Host(binding);
                    }
                    logical
                });
            }
            if let Some((binding, _, _, _)) = input.storage_image_binding {
                resources.entry(binding).or_insert_with(|| {
                    let mut logical = resource(binding, input.ty.clone(), LogicalSize::Unspecified);
                    logical.origin = ResourceOrigin::Host(binding);
                    logical
                });
            }
        }
        for output in &entry.outputs {
            if let Some(binding) = output.storage_binding {
                resources.entry(binding).or_insert_with(|| {
                    let mut logical = resource(
                        binding,
                        output.ty.elem_type().cloned().unwrap_or_else(|| output.ty.clone()),
                        pending_logical_size(output.length.as_ref()),
                    );
                    if entry.origin == interface::EntryOrigin::Source
                        || !filter_kinds.contains_key(&binding)
                    {
                        logical.origin = ResourceOrigin::Host(binding);
                    }
                    logical
                });
            }
        }
        for declaration in &entry.resource_declarations {
            let Some(binding) = pending_binding(declaration.resource) else {
                continue;
            };
            let origin = if let Some(kind) = filter_kinds.get(&binding) {
                ResourceOrigin::Compiler(kind.clone())
            } else if declaration.role == interface::StorageRole::Intermediate {
                ResourceOrigin::Compiler(CompilerResource::new(CompilerResourceKind::Staging, None, 0))
            } else {
                ResourceOrigin::Host(binding)
            };
            resources.entry(binding).or_insert(LogicalResource {
                id: ResourceId(0),
                origin,
                elem_ty: declaration.elem_ty.clone(),
                size: declaration.size.clone(),
            });
        }
    }
    resources.into_iter().collect()
}

/// Verify the allocation typestate. From this boundary through validation,
/// every executable storage identity is a `ResourceId`; bindings survive only
/// in the host ABI fields and `ResourceOrigin::Host` constraints.
pub(crate) fn verify_allocated_resources(inner: &SemanticProgram) -> Result<(), String> {
    let covered =
        inner.resources.iter().map(|resource| resource.id).collect::<std::collections::HashSet<_>>();
    if covered.len() != inner.resources.len() {
        return Err("resource manifest contains duplicate ids".to_string());
    }
    for (index, resource) in inner.resources.iter().enumerate() {
        let expected = ResourceId(index as u32);
        if resource.id != expected {
            return Err(format!(
                "resource manifest is not dense: position {index} contains {:?}",
                resource.id
            ));
        }
        verify_allocated_size(&resource.size, &covered)?;
        verify_allocated_type(&resource.elem_ty, &covered)?;
    }
    for entry in &inner.entry_points {
        for declaration in &entry.resource_declarations {
            let id = declaration
                .resource
                .resource()
                .ok_or_else(|| format!("allocated entry `{}` contains a pending binding", entry.name))?;
            if !covered.contains(&id) {
                return Err(format!(
                    "entry `{}` references resource {:?}, which is missing from the manifest",
                    entry.name, id
                ));
            }
            verify_allocated_size(&declaration.size, &covered)?;
            verify_allocated_type(&declaration.elem_ty, &covered)?;
            if declaration.role == interface::StorageRole::Intermediate {
                let resource = inner
                    .resources
                    .iter()
                    .find(|resource| resource.id == id)
                    .expect("resource coverage checked above");
                if !matches!(resource.origin, ResourceOrigin::Compiler(_)) {
                    return Err(format!("intermediate resource {:?} is not compiler-owned", id));
                }
            }
        }
        verify_allocated_type(&entry.return_ty, &covered)?;
        for input in &entry.inputs {
            verify_allocated_type(&input.ty, &covered)?;
        }
        for output in &entry.outputs {
            verify_allocated_type(&output.ty, &covered)?;
        }
        for (ty, _) in &entry.params {
            verify_allocated_type(ty, &covered)?;
        }
        verify_allocated_graph(&entry.graph, &covered, &format!("entry `{}`", entry.name))?;
    }
    for function in &inner.functions {
        verify_allocated_type(&function.return_ty, &covered)?;
        for (ty, _) in &function.params {
            verify_allocated_type(ty, &covered)?;
        }
        verify_allocated_graph(
            &function.graph,
            &covered,
            &format!("function `{}`", function.name),
        )?;
    }
    for region in inner.regions.values() {
        verify_allocated_type(&region.return_ty, &covered)?;
        for (ty, _) in &region.params {
            verify_allocated_type(ty, &covered)?;
        }
        verify_allocated_graph(&region.graph, &covered, &format!("region `{}`", region.name))?;
    }
    Ok(())
}

fn verify_allocated_size(
    size: &LogicalSize,
    covered: &std::collections::HashSet<ResourceId>,
) -> Result<(), String> {
    match size {
        LogicalSize::PendingLikeBinding { .. } => {
            Err("allocated resource has a pending binding-based size".to_string())
        }
        LogicalSize::LikeResource { resource, .. } if !covered.contains(resource) => {
            Err(format!("resource size references missing source {:?}", resource))
        }
        _ => Ok(()),
    }
}

fn verify_allocated_type(
    ty: &Type<TypeName>,
    covered: &std::collections::HashSet<ResourceId>,
) -> Result<(), String> {
    let Type::Constructed(name, arguments) = ty else {
        return Ok(());
    };
    match name {
        TypeName::Buffer(binding) => {
            return Err(format!(
                "allocated semantic type still names storage binding {binding:?}"
            ));
        }
        TypeName::Resource(resource) if !covered.contains(resource) => {
            return Err(format!("semantic type references missing resource {resource:?}"));
        }
        TypeName::Sum(variants) => {
            for (_, fields) in variants {
                for field in fields {
                    verify_allocated_type(field, covered)?;
                }
            }
        }
        _ => {}
    }
    for argument in arguments {
        verify_allocated_type(argument, covered)?;
    }
    Ok(())
}

fn verify_allocated_graph(
    graph: &EGraph,
    covered: &std::collections::HashSet<ResourceId>,
    owner: &str,
) -> Result<(), String> {
    let verify_resource = |resource: ResourceId| {
        if covered.contains(&resource) {
            Ok(())
        } else {
            Err(format!("{owner} references missing resource {resource:?}"))
        }
    };
    for (_, node) in &graph.nodes {
        if let super::types::ENode::Pure { op, .. } = node {
            match op {
                super::types::PureOp::StorageView(crate::op::PureViewSource::Storage(binding)) => {
                    return Err(format!("{owner} still contains storage binding {binding:?}"));
                }
                super::types::PureOp::StorageView(crate::op::PureViewSource::Resource(resource))
                | super::types::PureOp::ResourceLen(resource) => verify_resource(*resource)?,
                _ => {}
            }
        }
    }
    for ty in graph.types.values() {
        verify_allocated_type(ty, covered)?;
    }
    for (_, block) in &graph.skeleton.blocks {
        for effect in &block.side_effects {
            match &effect.kind {
                super::types::SideEffectKind::Inst(crate::ssa::types::InstKind::Op { tag, .. }) => {
                    match tag {
                        crate::op::OpTag::StorageView(crate::op::PureViewSource::Storage(binding)) => {
                            return Err(format!("{owner} still contains storage binding {binding:?}"));
                        }
                        crate::op::OpTag::StorageView(crate::op::PureViewSource::Resource(resource))
                        | crate::op::OpTag::ResourceLen(resource) => verify_resource(*resource)?,
                        _ => {}
                    }
                }
                super::types::SideEffectKind::Soac(soac) => {
                    let mut soac = soac.clone();
                    let mut error = None;
                    soac.visit_resource_refs_mut(|reference| match *reference {
                        SemanticResourceRef::Binding(binding) => {
                            error.get_or_insert_with(|| {
                                format!("{owner} SOAC still contains storage binding {binding:?}")
                            });
                        }
                        SemanticResourceRef::Resource(resource) if !covered.contains(&resource) => {
                            error.get_or_insert_with(|| {
                                format!("{owner} SOAC references missing resource {resource:?}")
                            });
                        }
                        SemanticResourceRef::Resource(_) => {}
                    });
                    soac.visit_types_mut(|ty| {
                        if error.is_none() {
                            error = verify_allocated_type(ty, covered).err();
                        }
                    });
                    if let Some(error) = error {
                        return Err(error);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

pub fn pending_logical_size(length: Option<&crate::pipeline_descriptor::BufferLen>) -> LogicalSize {
    match length {
        Some(crate::pipeline_descriptor::BufferLen::Fixed { bytes }) => LogicalSize::FixedBytes(*bytes),
        Some(crate::pipeline_descriptor::BufferLen::LikeInput {
            set,
            binding,
            elem_bytes,
            src_elem_bytes,
        }) => LogicalSize::PendingLikeBinding {
            binding: crate::BindingRef::new(*set, *binding),
            elem_bytes: *elem_bytes,
            src_elem_bytes: *src_elem_bytes,
        },
        Some(crate::pipeline_descriptor::BufferLen::SameAsDispatch { elem_bytes }) => {
            LogicalSize::SameAsDispatch {
                elem_bytes: *elem_bytes,
            }
        }
        None => LogicalSize::Unspecified,
    }
}

/// Physical `BufferLen` for a logical size, or `None` for `Unspecified` (a
/// host-supplied length). Inverse of `logical_size`, used when a compiler
/// resource is published as a `StorageBindingDecl`.
pub fn buffer_len(
    size: &LogicalSize,
    resources: &PhysicalResourceTable,
) -> Option<crate::pipeline_descriptor::BufferLen> {
    use crate::pipeline_descriptor::BufferLen;
    match size {
        LogicalSize::FixedBytes(bytes) => Some(BufferLen::Fixed { bytes: *bytes }),
        LogicalSize::PendingLikeBinding { .. } => {
            panic!("pending binding size reached physical publication")
        }
        LogicalSize::LikeResource {
            resource,
            elem_bytes,
            src_elem_bytes,
        } => {
            let binding =
                resources.binding(*resource).expect("size source resource must have a physical binding");
            Some(BufferLen::LikeInput {
                set: binding.set,
                binding: binding.binding,
                elem_bytes: *elem_bytes,
                src_elem_bytes: *src_elem_bytes,
            })
        }
        LogicalSize::SameAsDispatch { elem_bytes } => Some(BufferLen::SameAsDispatch {
            elem_bytes: *elem_bytes,
        }),
        LogicalSize::Unspecified => None,
    }
}

fn pending_binding(resource: SemanticResourceRef) -> Option<crate::BindingRef> {
    match resource {
        SemanticResourceRef::Binding(binding) => Some(binding),
        SemanticResourceRef::Resource(_) => None,
    }
}

pub struct EgirFunc {
    pub name: String,
    pub span: Span,
    pub linkage_name: Option<String>,
    pub params: Vec<(Type<TypeName>, String)>,
    pub return_ty: Type<TypeName>,
    pub graph: EGraph,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
    pub aliases: LookupMap<NodeId, NodeId>,
}

impl EgirFunc {
    pub fn new(
        name: String,
        span: Span,
        linkage_name: Option<String>,
        params: Vec<(Type<TypeName>, String)>,
        return_ty: Type<TypeName>,
        graph: EGraph,
        control_headers: LookupMap<BlockId, ControlHeader>,
    ) -> Self {
        EgirFunc {
            name,
            span,
            linkage_name,
            params,
            return_ty,
            graph,
            control_headers,
            aliases: LookupMap::new(),
        }
    }
}

/// One write site for an entry output slot: the block in which the
/// store fires and the value produced there. A slot can have multiple
/// sources when different CFG paths each write it (e.g. both arms of
/// an `If` whose result flows into the slot).
///
/// The `block` is load-bearing for any pass that emits side-effect
/// stores at the producer site — retargeting a `Map`'s destination is
/// metadata-only on the node, but emitting `Store` for a scalar
/// requires knowing the block to insert it into.
#[derive(Debug, Clone, Copy)]
pub struct SlotSource {
    pub block: BlockId,
    pub value: NodeId,
}

/// Stable identity of a declared entry-output position.
///
/// Keeping this distinct from a raw vector index makes output ownership
/// explicit in semantic records and prevents callers from confusing a result
/// lane, a storage binding, and an entry-output position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OutputSlotId(pub usize);

/// The concrete side effect that fulfils an output route after realization.
/// Value-producing effects (SOACs) are named by their result; stores, which do
/// not produce an EGIR value, are named by their effect token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputWriter {
    Value(NodeId),
    Effect(EffectToken),
}

/// Declared output ownership carried from `OutputSlotStore` conversion through
/// physicalization. `source.value` is the user-level value, `slot` is the
/// declared output it fulfils, and `writers` are populated by output
/// realization. The slot's `EntryOutput::storage_binding` then identifies the
/// host resource until logical-resource allocation assigns a `ResourceId`.
#[derive(Debug, Clone)]
pub struct OutputRoute {
    pub source: SlotSource,
    pub slot: OutputSlotId,
    pub writers: Vec<OutputWriter>,
}

pub struct SemanticEntry {
    /// Source/compiler provenance. Generated names are not semantic tags.
    pub origin: interface::EntryOrigin,
    pub name: String,
    pub span: Span,
    pub execution_model: ExecutionModel,
    pub inputs: Vec<EntryInput>,
    pub outputs: Vec<EntryOutput>,
    pub resource_declarations: Vec<SemanticResourceDecl>,
    pub params: Vec<(Type<TypeName>, String)>,
    pub return_ty: Type<TypeName>,
    pub graph: EGraph,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
    pub aliases: LookupMap<NodeId, NodeId>,
    /// Explicit value-to-output routes. A slot can have several routes when
    /// distinct CFG paths write it. Output realization fills `writers`; later
    /// semantic passes consume these declarations instead of reconstructing
    /// ownership from storage-view provenance and effect shape.
    pub output_routes: Vec<OutputRoute>,
}

impl SemanticEntry {
    pub fn new(
        origin: interface::EntryOrigin,
        name: String,
        span: Span,
        execution_model: ExecutionModel,
        inputs: Vec<EntryInput>,
        outputs: Vec<EntryOutput>,
        storage_bindings: Vec<interface::StorageBindingDecl>,
        params: Vec<(Type<TypeName>, String)>,
        return_ty: Type<TypeName>,
        graph: EGraph,
        control_headers: LookupMap<BlockId, ControlHeader>,
    ) -> Self {
        Self::new_with_resources(
            origin,
            name,
            span,
            execution_model,
            inputs,
            outputs,
            storage_bindings.into_iter().map(SemanticResourceDecl::pending).collect(),
            params,
            return_ty,
            graph,
            control_headers,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_resources(
        origin: interface::EntryOrigin,
        name: String,
        span: Span,
        execution_model: ExecutionModel,
        inputs: Vec<EntryInput>,
        outputs: Vec<EntryOutput>,
        resource_declarations: Vec<SemanticResourceDecl>,
        params: Vec<(Type<TypeName>, String)>,
        return_ty: Type<TypeName>,
        graph: EGraph,
        control_headers: LookupMap<BlockId, ControlHeader>,
    ) -> Self {
        SemanticEntry {
            origin,
            name,
            span,
            execution_model,
            inputs,
            outputs,
            resource_declarations,
            params,
            return_ty,
            graph,
            control_headers,
            aliases: LookupMap::new(),
            output_routes: Vec::new(),
        }
    }

    pub fn publication(&self) -> PlannedEntryPublication {
        PlannedEntryPublication {
            name: self.name.clone(),
            execution_model: self.execution_model.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            resources: self.resource_declarations.clone(),
        }
    }
}

/// The exact entry ABI consumed by descriptor publication. It deliberately
/// contains no graph, routes, or mutable lowering state.
#[derive(Clone, Debug)]
pub struct PlannedEntryPublication {
    pub name: String,
    pub execution_model: ExecutionModel,
    pub inputs: Vec<EntryInput>,
    pub outputs: Vec<EntryOutput>,
    pub resources: Vec<SemanticResourceDecl>,
}

impl PlannedEntryPublication {
    pub fn physicalize(&self, resources: &PhysicalResourceTable) -> Result<EntryPublication, String> {
        let storage_bindings =
            self.resources
                .iter()
                .map(|declaration| {
                    let resource = declaration.resource.resource().ok_or_else(|| {
                        format!("entry `{}` contains a pending resource binding", self.name)
                    })?;
                    if !resources.is_compiler(resource) {
                        return Ok(None);
                    }
                    let binding = resources.binding(resource).ok_or_else(|| {
                        format!(
                            "entry `{}` references unallocated resource {:?}",
                            self.name, resource
                        )
                    })?;
                    Ok(Some(interface::StorageBindingDecl {
                        binding,
                        role: declaration.role.clone(),
                        elem_ty: declaration.elem_ty.clone(),
                        length: buffer_len(&declaration.size, resources),
                    }))
                })
                .collect::<Result<Vec<_>, String>>()?
                .into_iter()
                .flatten()
                .collect();
        Ok(EntryPublication {
            name: self.name.clone(),
            execution_model: self.execution_model.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            storage_bindings,
        })
    }
}

#[derive(Clone, Debug)]
pub struct EntryPublication {
    pub name: String,
    pub execution_model: ExecutionModel,
    pub inputs: Vec<EntryInput>,
    pub outputs: Vec<EntryOutput>,
    pub storage_bindings: Vec<interface::StorageBindingDecl>,
}

/// A complete entry after a validated kernel recipe has been physicalized.
/// This is intentionally a distinct type from `SemanticEntry`: downstream
/// codegen passes cannot receive an entry that is still legal to reschedule.
pub struct PhysicalEntry {
    pub origin: interface::EntryOrigin,
    pub name: String,
    pub span: Span,
    pub execution_model: ExecutionModel,
    pub inputs: Vec<EntryInput>,
    pub outputs: Vec<EntryOutput>,
    pub storage_bindings: Vec<interface::StorageBindingDecl>,
    pub params: Vec<(Type<TypeName>, String)>,
    pub return_ty: Type<TypeName>,
    pub graph: EGraph,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
    pub aliases: LookupMap<NodeId, NodeId>,
    pub output_routes: Vec<OutputRoute>,
}

/// Deterministic allocation of logical resources to backend bindings.
#[derive(Clone, Debug, Default)]
pub struct PhysicalResourceTable {
    bindings: Vec<crate::BindingRef>,
    compiler_owned: Vec<bool>,
}

impl PhysicalResourceTable {
    pub fn allocate(resources: &[LogicalResource], ids: &mut crate::IdSource<u32>) -> Self {
        let mut ordered = resources.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|resource| resource.id.0);
        let mut used = resources
            .iter()
            .filter_map(|resource| match resource.origin {
                ResourceOrigin::Host(binding) => Some(binding),
                ResourceOrigin::Compiler(_) => None,
            })
            .collect::<std::collections::HashSet<_>>();
        let mut bindings = Vec::with_capacity(ordered.len());
        let mut compiler_owned = Vec::with_capacity(ordered.len());
        for resource in ordered {
            compiler_owned.push(matches!(resource.origin, ResourceOrigin::Compiler(_)));
            let binding = match resource.origin {
                ResourceOrigin::Host(binding) => binding,
                ResourceOrigin::Compiler(_) => loop {
                    let candidate =
                        crate::BindingRef::new(super::from_tlc::AUTO_STORAGE_SET, ids.next_id());
                    if used.insert(candidate) {
                        break candidate;
                    }
                },
            };
            bindings.push(binding);
        }
        Self {
            bindings,
            compiler_owned,
        }
    }

    pub fn binding(&self, resource: ResourceId) -> Option<crate::BindingRef> {
        self.bindings.get(resource.0 as usize).copied()
    }

    pub fn is_compiler(&self, resource: ResourceId) -> bool {
        self.compiler_owned.get(resource.0 as usize).copied().unwrap_or(false)
    }
}

/// Whole-program EGIR container. Wrapped by the semantic `EgirRaw` /
/// `EgirSegmented` / `EgirOptimized` / `EgirAllocated` newtypes at
/// the public-API layer (see `crate::lib`).
pub struct SemanticProgram {
    pub functions: Vec<EgirFunc>,
    /// Extern function stubs. These don't have a body that flows through EGIR;
    /// they're already `Function` records with a 1-block Unreachable body and
    /// pass straight through.
    pub externs: Vec<Function>,
    pub entry_points: Vec<SemanticEntry>,
    pub constants: Vec<Constant>,
    pub pipeline: PipelineDescriptor,
    /// Source names retained until the descriptor is published atomically at
    /// terminal lowering.
    pub input_names: LookupMap<(u32, u32), String>,
    /// Complete callable regions referenced by semantic Seg bodies, keyed by
    /// their arena index.
    pub regions: LookupMap<RegionId, EgirRegion>,
    /// Name ↔ index interner shared with construction. Synthesized regions
    /// (e.g. scan offset wrappers) intern here to obtain a fresh index.
    pub region_interner: RegionInterner,
    /// Logical host and compiler resources. Compiler resources receive a
    /// physical binding only during target-aware lowering.
    pub resources: Vec<LogicalResource>,
    /// Whole-program semantic dependency DAG. Edges come from values, effect
    /// tokens, and conflicting logical resource accesses.
    pub semantic_dependencies: Vec<SemanticDependency>,
    /// Concrete compute schedule produced after segmented-operation lowering.
    /// It remains attached to EGIR so later passes and descriptor publication
    /// consume the same phase/resource graph instead of rediscovering it.
    pub kernel_plan: KernelPlan,
}

/// EGIR after the plan has validated and every physical entry has been
/// constructed. Only this type is accepted by expansion and SSA elaboration.
pub struct PhysicalProgram {
    pub functions: Vec<EgirFunc>,
    pub externs: Vec<Function>,
    pub entry_points: Vec<PhysicalEntry>,
    pub constants: Vec<Constant>,
    pub pipeline: PipelineDescriptor,
    pub input_names: LookupMap<(u32, u32), String>,
    pub regions: LookupMap<RegionId, EgirRegion>,
    pub region_interner: RegionInterner,
    pub resources: Vec<LogicalResource>,
    pub semantic_dependencies: Vec<SemanticDependency>,
    pub plan: ValidatedKernelPlan,
    pub physical_resources: PhysicalResourceTable,
}

impl PhysicalProgram {
    pub fn from_validated(
        program: SemanticProgram,
        plan: ValidatedKernelPlan,
        physical_resources: PhysicalResourceTable,
    ) -> Result<Self, String> {
        let entry_points = program
            .entry_points
            .into_iter()
            .map(|entry| {
                super::builder::PhysicalEntryBuilder::from_planned_entry(entry, &physical_resources).build()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            functions: program.functions,
            externs: program.externs,
            entry_points,
            constants: program.constants,
            pipeline: program.pipeline,
            input_names: program.input_names,
            regions: program.regions,
            region_interner: program.region_interner,
            resources: program.resources,
            semantic_dependencies: program.semantic_dependencies,
            plan,
            physical_resources,
        })
    }
}

/// Give `function` its region index and record its body under it. The index is
/// the interned name, so calling this twice for one name refreshes the body
/// rather than allocating a second region.
fn record_region(
    interner: &mut RegionInterner,
    regions: &mut LookupMap<RegionId, EgirRegion>,
    function: &EgirFunc,
) -> RegionId {
    let id = interner.intern(&function.name);
    regions.insert(id, EgirRegion::from_function(function));
    id
}

impl SemanticProgram {
    pub fn new(
        functions: Vec<EgirFunc>,
        externs: Vec<Function>,
        entry_points: Vec<SemanticEntry>,
        constants: Vec<Constant>,
        pipeline: PipelineDescriptor,
        mut region_interner: RegionInterner,
    ) -> Self {
        // Every function is callable, so it owns a region index. Names already
        // interned during construction keep their index; the rest are assigned
        // here. The arena is then keyed by that index.
        let mut regions = LookupMap::new();
        for function in &functions {
            record_region(&mut region_interner, &mut regions, function);
        }
        SemanticProgram {
            functions,
            externs,
            entry_points,
            constants,
            pipeline,
            input_names: LookupMap::new(),
            regions,
            region_interner,
            resources: Vec::new(),
            semantic_dependencies: Vec::new(),
            kernel_plan: KernelPlan::default(),
        }
    }

    /// Convenience: build an EGIR program wrapping a single function body.
    /// Used by the probe path in `from_tlc`.
    pub fn single_function(func: EgirFunc) -> Self {
        Self::new(
            vec![func],
            vec![],
            vec![],
            vec![],
            PipelineDescriptor::default(),
            RegionInterner::default(),
        )
    }

    /// Intern (or look up) the region backing a callable name. Synthesized
    /// regions created after construction obtain their index this way.
    pub fn intern_region(&mut self, name: impl AsRef<str>) -> RegionId {
        self.region_interner.intern(name)
    }

    /// Add a synthesized function to the program: record its body in the region
    /// arena and make it callable. The returned index is the one a `SegBody`
    /// must name to call it, and it equals `intern_region(&function.name)`, so a
    /// caller that needed the index before the body existed may use either.
    pub fn define_region(&mut self, function: EgirFunc) -> RegionId {
        let id = record_region(&mut self.region_interner, &mut self.regions, &function);
        self.functions.push(function);
        id
    }

    /// SSA function name backing a region index (the `PureOp::Call` ABI).
    pub fn region_name(&self, id: RegionId) -> &str {
        self.region_interner.name(id)
    }
}
