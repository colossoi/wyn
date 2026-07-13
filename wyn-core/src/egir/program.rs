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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SemanticOpId {
    pub scope: String,
    pub result: NodeId,
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
    LikeBinding {
        binding: crate::BindingRef,
        elem_bytes: u32,
        src_elem_bytes: u32,
    },
    SameAsDispatch {
        elem_bytes: u32,
    },
    Unspecified,
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
    /// Transitional identity used by storage-view types until terminal
    /// lowering rewrites compiler resources to their allocated binding.
    pub legacy_binding: crate::BindingRef,
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
    super::multi_consumer::run(inner, binding_ids);
    let mut filter_work = allocate_filter_work_resources(inner, binding_ids);
    let scalar_handoffs = scalar_handoff_resources(inner);
    inner.resource_hints.extend(scalar_handoffs);
    let filter_kinds = filter_resource_kinds(inner);
    inner.resource_hints.extend(filter_kinds);
    let mut host = mirror_storage_resources(inner, &inner.resource_hints);
    host.sort_by_key(|resource| (resource.legacy_binding.set, resource.legacy_binding.binding));
    for (index, resource) in host.iter_mut().enumerate() {
        resource.id = ResourceId(index as u32);
    }
    attach_materialization_resources(inner, &host);
    let mut resources = host;
    for resource in &mut filter_work {
        resource.id = ResourceId(resources.len() as u32);
        resources.push(resource.clone());
    }
    let mut scratch = super::parallelize::enumerate_seg_scratch(inner, binding_ids, resources.len() as u32);
    resources.append(&mut scratch);
    inner.resources = resources;
    normalize_semantic_resource_references(inner);
    if cfg!(debug_assertions) {
        verify_manifest_covers_storage(inner);
    }
}

/// Replace descriptor-shaped storage identities in semantic graphs with their
/// target-independent `ResourceId`. Entry ABI declarations retain host
/// bindings; executable graph values do not.
fn normalize_semantic_resource_references(inner: &mut SemanticProgram) {
    let by_binding = inner
        .resources
        .iter()
        .map(|resource| (resource.legacy_binding, resource.id))
        .collect::<HashMap<_, _>>();
    for entry in &mut inner.entry_points {
        normalize_graph_resources(&mut entry.graph, &by_binding);
    }
    for function in &mut inner.functions {
        normalize_graph_resources(&mut function.graph, &by_binding);
    }
    for region in inner.regions.values_mut() {
        normalize_graph_resources(&mut region.graph, &by_binding);
    }
}

fn normalize_graph_resources(graph: &mut EGraph, by_binding: &HashMap<crate::BindingRef, ResourceId>) {
    let pure_nodes = graph.nodes.keys().collect::<Vec<_>>();
    for node in pure_nodes {
        graph.update_pure_node(node, |op, _| {
            if let super::types::PureOp::StorageView(crate::op::PureViewSource::Storage(binding)) = op {
                if let Some(resource) = by_binding.get(binding) {
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
                        *tag =
                            crate::op::OpTag::StorageView(crate::op::PureViewSource::Resource(*resource));
                    }
                }
            }
        }
    }
}

/// Resolve semantic resource references immediately after validation and
/// before any physical graph transformation or backend pass runs.
pub fn physicalize_resource_references(inner: &mut SemanticProgram) -> Result<(), String> {
    let bindings = PhysicalResourceTable::from_resources(&inner.resources);
    for entry in &mut inner.entry_points {
        physicalize_graph_resources(&mut entry.graph, &bindings)?;
    }
    for function in &mut inner.functions {
        physicalize_graph_resources(&mut function.graph, &bindings)?;
    }
    for region in inner.regions.values_mut() {
        physicalize_graph_resources(&mut region.graph, &bindings)?;
    }
    Ok(())
}

fn physicalize_graph_resources(graph: &mut EGraph, bindings: &PhysicalResourceTable) -> Result<(), String> {
    let pure_nodes = graph.nodes.keys().collect::<Vec<_>>();
    let mut missing = None;
    for node in pure_nodes {
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
        }
    }
    if let Some(resource) = missing {
        return Err(format!(
            "semantic resource {:?} has no physical binding",
            resource
        ));
    }
    Ok(())
}

fn allocate_filter_work_resources(
    inner: &mut SemanticProgram,
    binding_ids: &mut crate::IdSource<u32>,
) -> Vec<LogicalResource> {
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
                .chain(entry.storage_bindings.iter().map(|declaration| declaration.binding))
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
                    flags: next_binding(),
                    offsets: next_binding(),
                    block_sums: next_binding(),
                    block_offsets: next_binding(),
                };
                let element_count_size = match space.dims.first() {
                    Some(SegExtent::Fixed(count)) if space.dims.len() == 1 => {
                        LogicalSize::FixedBytes(*count as u64 * 4)
                    }
                    Some(SegExtent::ResourceLength {
                        binding, elem_bytes, ..
                    }) if space.dims.len() == 1 => LogicalSize::LikeBinding {
                        binding: *binding,
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
                let owner = effect.result.map(|result| SemanticOpId {
                    scope: entry.name.clone(),
                    result,
                });
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
                    let compiler = CompilerResource::new(kind, owner.clone(), slot);
                    inner.resource_hints.insert(binding, compiler.clone());
                    resources.push(LogicalResource {
                        id: ResourceId(0),
                        origin: ResourceOrigin::Compiler(compiler),
                        legacy_binding: binding,
                        elem_ty: Type::Constructed(TypeName::UInt(32), vec![]),
                        size,
                    });
                }
            }
        }
    }
    resources
}

fn attach_materialization_resources(_inner: &mut SemanticProgram, _resources: &[LogicalResource]) {
    // Resource ownership lives in the manifest; Seg operations no longer
    // mirror resource ids in a phase-dependent scratch field.
}

/// Refresh the manifest after terminal lowering has introduced phase entries
/// and physical declarations.  Unlike `plan_logical_resources`, this never
/// allocates a binding or rewrites an owning SegOp; it preserves the precise
/// compiler origin assigned at the allocation boundary.
pub fn refresh_logical_resources(inner: &mut SemanticProgram) {
    let previous: HashMap<_, _> =
        inner.resources.drain(..).map(|resource| (resource.legacy_binding, resource)).collect();
    let filter_kinds = filter_resource_kinds(inner);
    let scalar_handoffs = scalar_handoff_resources(inner);
    inner.resource_hints.extend(scalar_handoffs);
    inner.resource_hints.extend(filter_kinds);
    let mut refreshed = mirror_storage_resources(inner, &inner.resource_hints);
    for resource in &mut refreshed {
        if let Some(old) = previous.get(&resource.legacy_binding) {
            if matches!(old.origin, ResourceOrigin::Compiler(_)) {
                resource.origin = old.origin.clone();
                resource.elem_ty = old.elem_ty.clone();
                resource.size = old.size.clone();
            }
        }
    }
    // A compiler resource may not be repeated on every synthesized phase, but
    // remains part of the program-level manifest.
    for (binding, resource) in previous {
        if matches!(resource.origin, ResourceOrigin::Compiler(_))
            && !refreshed.iter().any(|candidate| candidate.legacy_binding == binding)
        {
            refreshed.push(resource);
        }
    }
    refreshed.sort_by_key(|resource| (resource.legacy_binding.set, resource.legacy_binding.binding));
    for (index, resource) in refreshed.iter_mut().enumerate() {
        resource.id = ResourceId(index as u32);
    }
    inner.resources = refreshed;
    if cfg!(debug_assertions) {
        verify_manifest_covers_storage(inner);
    }
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
            entry.storage_bindings.iter().filter_map(|declaration| {
                (declaration.role == interface::StorageRole::Input).then_some(declaration.binding)
            })
        })
        .collect();
    let mut resources = HashMap::new();
    for entry in &inner.entry_points {
        if entry.origin != interface::EntryOrigin::ScalarPrepass {
            continue;
        }
        let owner = entry.graph.skeleton.blocks.iter().find_map(|(_, block)| {
            block.side_effects.iter().find_map(|effect| {
                effect.result.map(|result| SemanticOpId {
                    scope: entry.name.clone(),
                    result,
                })
            })
        });
        for declaration in &entry.storage_bindings {
            if declaration.role == interface::StorageRole::Output
                && declaration.length.is_none()
                && consumer_inputs.contains(&declaration.binding)
            {
                resources.insert(
                    declaration.binding,
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
                    let owner = effect.result.map(|result| SemanticOpId {
                        scope: entry.name.clone(),
                        result,
                    });
                    if let super::types::FilterOutput::Runtime { scratch, length } = output {
                        kinds.insert(
                            *scratch,
                            CompilerResource::new(CompilerResourceKind::FilterScratch, owner.clone(), 0),
                        );
                        if let super::types::RuntimeFilterLength::EntryOutput(len) = length {
                            kinds.insert(
                                *len,
                                CompilerResource::new(
                                    CompilerResourceKind::FilterLenCell,
                                    owner.clone(),
                                    1,
                                ),
                            );
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
                            kinds.insert(
                                work.flags,
                                CompilerResource::new(CompilerResourceKind::FilterFlags, owner.clone(), 2),
                            );
                            kinds.insert(
                                work.offsets,
                                CompilerResource::new(
                                    CompilerResourceKind::FilterOffsets,
                                    owner.clone(),
                                    3,
                                ),
                            );
                            kinds.insert(
                                work.block_sums,
                                CompilerResource::new(
                                    CompilerResourceKind::FilterScanBlockSums,
                                    owner.clone(),
                                    4,
                                ),
                            );
                            kinds.insert(
                                work.block_offsets,
                                CompilerResource::new(
                                    CompilerResourceKind::FilterScanBlockOffsets,
                                    owner,
                                    5,
                                ),
                            );
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
) -> Vec<LogicalResource> {
    let mut resources: LookupMap<crate::BindingRef, LogicalResource> = LookupMap::new();
    let resource =
        |binding: crate::BindingRef, elem_ty: Type<TypeName>, size: LogicalSize| LogicalResource {
            id: ResourceId(0),
            origin: filter_kinds
                .get(&binding)
                .cloned()
                .map(ResourceOrigin::Compiler)
                .unwrap_or(ResourceOrigin::Host(binding)),
            legacy_binding: binding,
            elem_ty,
            size,
        };
    for entry in &inner.entry_points {
        for input in &entry.inputs {
            if let Some(binding) = input.storage_binding {
                resources.entry(binding).or_insert_with(|| {
                    resource(
                        binding,
                        input.ty.elem_type().cloned().unwrap_or_else(|| input.ty.clone()),
                        logical_size(input.length.as_ref()),
                    )
                });
            }
        }
        for output in &entry.outputs {
            if let Some(binding) = output.storage_binding {
                resources.entry(binding).or_insert_with(|| {
                    resource(
                        binding,
                        output.ty.elem_type().cloned().unwrap_or_else(|| output.ty.clone()),
                        logical_size(output.length.as_ref()),
                    )
                });
            }
        }
        for declaration in &entry.storage_bindings {
            let origin = if let Some(kind) = filter_kinds.get(&declaration.binding) {
                ResourceOrigin::Compiler(kind.clone())
            } else if declaration.role == interface::StorageRole::Intermediate {
                ResourceOrigin::Compiler(CompilerResource::new(CompilerResourceKind::Staging, None, 0))
            } else {
                ResourceOrigin::Host(declaration.binding)
            };
            resources.entry(declaration.binding).or_insert(LogicalResource {
                id: ResourceId(0),
                origin,
                legacy_binding: declaration.binding,
                elem_ty: declaration.elem_ty.clone(),
                size: logical_size(declaration.length.as_ref()),
            });
        }
    }
    resources.into_values().collect()
}

/// Debug invariant: every physical storage binding the program declares is
/// covered by a logical resource. Catches a scratch site that escaped the
/// manifest before it can silently corrupt the descriptor.
fn verify_manifest_covers_storage(inner: &SemanticProgram) {
    let covered: std::collections::HashSet<crate::BindingRef> =
        inner.resources.iter().map(|resource| resource.legacy_binding).collect();
    for entry in &inner.entry_points {
        for declaration in &entry.storage_bindings {
            debug_assert!(
                covered.contains(&declaration.binding),
                "storage binding {:?} ({:?}) is missing from the logical-resource manifest",
                declaration.binding,
                declaration.role,
            );
            if declaration.role == interface::StorageRole::Intermediate {
                let resource = inner
                    .resources
                    .iter()
                    .find(|resource| resource.legacy_binding == declaration.binding)
                    .expect("coverage checked above");
                debug_assert!(
                    matches!(resource.origin, ResourceOrigin::Compiler(_)),
                    "intermediate binding {:?} is not compiler-owned",
                    declaration.binding,
                );
            }
        }
    }
    let unique = inner
        .resources
        .iter()
        .map(|resource| resource.legacy_binding)
        .collect::<std::collections::HashSet<_>>();
    debug_assert_eq!(
        unique.len(),
        inner.resources.len(),
        "resource manifest contains duplicate bindings: {:?}",
        inner
            .resources
            .iter()
            .map(|resource| (resource.legacy_binding, &resource.origin))
            .collect::<Vec<_>>()
    );
}

pub fn logical_size(length: Option<&crate::pipeline_descriptor::BufferLen>) -> LogicalSize {
    match length {
        Some(crate::pipeline_descriptor::BufferLen::Fixed { bytes }) => LogicalSize::FixedBytes(*bytes),
        Some(crate::pipeline_descriptor::BufferLen::LikeInput {
            set,
            binding,
            elem_bytes,
            src_elem_bytes,
        }) => LogicalSize::LikeBinding {
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
pub fn buffer_len(size: &LogicalSize) -> Option<crate::pipeline_descriptor::BufferLen> {
    use crate::pipeline_descriptor::BufferLen;
    match size {
        LogicalSize::FixedBytes(bytes) => Some(BufferLen::Fixed { bytes: *bytes }),
        LogicalSize::LikeBinding {
            binding,
            elem_bytes,
            src_elem_bytes,
        } => Some(BufferLen::LikeInput {
            set: binding.set,
            binding: binding.binding,
            elem_bytes: *elem_bytes,
            src_elem_bytes: *src_elem_bytes,
        }),
        LogicalSize::SameAsDispatch { elem_bytes } => Some(BufferLen::SameAsDispatch {
            elem_bytes: *elem_bytes,
        }),
        LogicalSize::Unspecified => None,
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
    pub storage_bindings: Vec<interface::StorageBindingDecl>,
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
        SemanticEntry {
            origin,
            name,
            span,
            execution_model,
            inputs,
            outputs,
            storage_bindings,
            params,
            return_ty,
            graph,
            control_headers,
            aliases: LookupMap::new(),
            output_routes: Vec::new(),
        }
    }
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

impl From<SemanticEntry> for PhysicalEntry {
    fn from(entry: SemanticEntry) -> Self {
        Self {
            origin: entry.origin,
            name: entry.name,
            span: entry.span,
            execution_model: entry.execution_model,
            inputs: entry.inputs,
            outputs: entry.outputs,
            storage_bindings: entry.storage_bindings,
            params: entry.params,
            return_ty: entry.return_ty,
            graph: entry.graph,
            control_headers: entry.control_headers,
            aliases: entry.aliases,
            output_routes: entry.output_routes,
        }
    }
}

/// Deterministic allocation of logical resources to backend bindings.
#[derive(Clone, Debug, Default)]
pub struct PhysicalResourceTable {
    bindings: Vec<crate::BindingRef>,
}

impl PhysicalResourceTable {
    pub fn from_resources(resources: &[LogicalResource]) -> Self {
        let mut ordered = resources.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|resource| resource.id.0);
        Self {
            bindings: ordered.into_iter().map(|resource| resource.legacy_binding).collect(),
        }
    }

    pub fn binding(&self, resource: ResourceId) -> Option<crate::BindingRef> {
        self.bindings.get(resource.0 as usize).copied()
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
    /// Precise compiler ownership for physical declarations that exist before
    /// the numbered logical-resource table is rebuilt.
    pub resource_hints: LookupMap<crate::BindingRef, CompilerResource>,
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
    pub fn from_validated(program: SemanticProgram, plan: ValidatedKernelPlan) -> Self {
        let physical_resources = PhysicalResourceTable::from_resources(&program.resources);
        Self {
            functions: program.functions,
            externs: program.externs,
            entry_points: program.entry_points.into_iter().map(PhysicalEntry::from).collect(),
            constants: program.constants,
            pipeline: program.pipeline,
            input_names: program.input_names,
            regions: program.regions,
            region_interner: program.region_interner,
            resources: program.resources,
            semantic_dependencies: program.semantic_dependencies,
            plan,
            physical_resources,
        }
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
            resource_hints: LookupMap::new(),
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

    /// Physical binding currently backing a logical resource. Resources are
    /// stored in `ResourceId` order, so this is a direct index. Until terminal
    /// lowering rewrites compiler resources to allocated bindings, this returns
    /// the `legacy_binding`.
    pub fn binding_of(&self, id: ResourceId) -> crate::BindingRef {
        self.resources[id.0 as usize].legacy_binding
    }
}
