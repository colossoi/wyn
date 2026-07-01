//! Whole-program EGIR container + per-body records.
//!
//! These are plain (non-generic) structs. State tracking happens at the
//! public API boundary via the semantic `EgirRaw` / `EgirSegmented` /
//! `EgirOptimized` / `EgirAllocated` newtypes in `crate::lib`, each
//! of which wraps an `EgirInner`.
//!
//! `EgirInner` carries, for each function and entry point, a per-body
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

use super::parallelize::schedule::KernelSchedule;
use super::types::{EGraph, NodeId, RegionId};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceId(pub u32);

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
}

impl CompilerResourceKind {
    /// The physical storage role a resource of this kind lowers to.
    pub fn role(self) -> interface::StorageRole {
        match self {
            CompilerResourceKind::FilterScratch => interface::StorageRole::Output,
            _ => interface::StorageRole::Intermediate,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ResourceOrigin {
    Host(crate::BindingRef),
    Compiler(CompilerResourceKind),
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
/// contributes its scratch as `Compiler` resources — fresh bindings are drawn
/// here and their `ResourceId`s recorded on the owning Seg op, so terminal
/// lowering consumes the manifest instead of allocating scratch ad hoc.
pub fn plan_logical_resources(inner: &mut EgirInner, binding_ids: &mut crate::IdSource<u32>) {
    let filter_kinds = filter_resource_kinds(inner);
    let mut host = mirror_storage_resources(inner, &filter_kinds);
    host.sort_by_key(|resource| (resource.legacy_binding.set, resource.legacy_binding.binding));
    for (index, resource) in host.iter_mut().enumerate() {
        resource.id = ResourceId(index as u32);
    }
    let mut scratch = super::parallelize::enumerate_seg_scratch(inner, binding_ids, host.len() as u32);
    let mut resources = host;
    resources.append(&mut scratch);
    inner.resources = resources;
    if cfg!(debug_assertions) {
        verify_manifest_covers_storage(inner);
    }
}

/// Runtime `filter` bindings, classified so the mirror gives them a precise
/// `CompilerResourceKind` rather than generic `Staging`.
fn filter_resource_kinds(inner: &EgirInner) -> HashMap<crate::BindingRef, CompilerResourceKind> {
    let mut kinds = HashMap::new();
    for entry in &inner.entry_points {
        for (_, block) in &entry.graph.skeleton.blocks {
            for effect in &block.side_effects {
                if let super::types::SideEffectKind::Soac(super::types::EgirSoac::Filter {
                    len_out, ..
                }) = &effect.kind
                {
                    if let Some(len) = len_out {
                        kinds.insert(*len, CompilerResourceKind::FilterLenCell);
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
    inner: &EgirInner,
    filter_kinds: &HashMap<crate::BindingRef, CompilerResourceKind>,
) -> Vec<LogicalResource> {
    let mut resources: LookupMap<crate::BindingRef, LogicalResource> = LookupMap::new();
    let host = |binding: crate::BindingRef, elem_ty: Type<TypeName>, size: LogicalSize| LogicalResource {
        id: ResourceId(0),
        origin: ResourceOrigin::Host(binding),
        legacy_binding: binding,
        elem_ty,
        size,
    };
    for entry in &inner.entry_points {
        for input in &entry.inputs {
            if let Some(binding) = input.storage_binding {
                resources.entry(binding).or_insert_with(|| {
                    host(
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
                    host(
                        binding,
                        output.ty.elem_type().cloned().unwrap_or_else(|| output.ty.clone()),
                        logical_size(output.length.as_ref()),
                    )
                });
            }
        }
        for declaration in &entry.storage_bindings {
            let origin = if let Some(kind) = filter_kinds.get(&declaration.binding) {
                ResourceOrigin::Compiler(*kind)
            } else if declaration.role == interface::StorageRole::Intermediate {
                ResourceOrigin::Compiler(CompilerResourceKind::Staging)
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
fn verify_manifest_covers_storage(inner: &EgirInner) {
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
        }
    }
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

#[derive(Clone)]
pub struct EgirEntry {
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
    /// Per-slot list of (producing-block, value) pairs. Indexed by
    /// declared output slot. A slot with one source has `vec![one]`;
    /// a slot written from both arms of an `If` has two. Empty for
    /// unit-returning entries. Phase 1 of the DPS migration: populated
    /// only by code added in later phases; today's `from_tlc` leaves
    /// it untouched.
    pub slot_sources: Vec<Vec<SlotSource>>,
}

impl EgirEntry {
    pub fn new(
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
        EgirEntry {
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
            slot_sources: Vec::new(),
        }
    }
}

/// Whole-program EGIR container. Wrapped by the semantic `EgirRaw` /
/// `EgirSegmented` / `EgirOptimized` / `EgirAllocated` newtypes at
/// the public-API layer (see `crate::lib`).
pub struct EgirInner {
    pub functions: Vec<EgirFunc>,
    /// Extern function stubs. These don't have a body that flows through EGIR;
    /// they're already `Function` records with a 1-block Unreachable body and
    /// pass straight through.
    pub externs: Vec<Function>,
    pub entry_points: Vec<EgirEntry>,
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
    pub kernel_schedule: KernelSchedule,
}

impl EgirInner {
    pub fn new(
        functions: Vec<EgirFunc>,
        externs: Vec<Function>,
        entry_points: Vec<EgirEntry>,
        constants: Vec<Constant>,
        pipeline: PipelineDescriptor,
        mut region_interner: RegionInterner,
    ) -> Self {
        // Every function is callable, so it owns a region index. Names already
        // interned during construction keep their index; the rest are assigned
        // here. The arena is then keyed by that index.
        let mut regions = LookupMap::new();
        for function in &functions {
            let id = region_interner.intern(&function.name);
            regions.insert(id, EgirRegion::from_function(function));
        }
        EgirInner {
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
            kernel_schedule: KernelSchedule::default(),
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
