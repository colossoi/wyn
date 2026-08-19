//! Compiler-specific EGIR program data and per-body records.
//!
//! EGIR typestate aliases explicitly select their recursive graph family,
//! entry-boundary data, program-owned data, and global context. This module
//! defines the concrete resource arenas, identifiers, and program data used
//! by those states.

use crate::builtins;
use crate::pipeline_descriptor;
use crate::ssa;
use crate::LoweringProfile;
use crate::ResourceAccess;
use crate::SortedSet;
use crate::{BindingRef, EntryId, FunctionId, GlobalId, IdArena, IdSource, LookupMap};

use polytype::Type;

use crate::ast::{Span, TypeName};
use crate::flow::{BlockId, ExecutionModel};
use crate::interface::{self, EntryInput, EntryOutput};
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::types::TypeExt;
use std::collections::{HashMap, HashSet};
use std::ops::{Deref, Index, IndexMut};

use super::soac::{filter, hist, screma};
use super::types::{
    EGraph, Family, Physical, Raw, Scheduled, SegExtent, SegSpace, Semantic, SideEffectKind, Soac,
    SoacEffect, ValueId, ValueKind, WynLanguage,
};

pub use super::ir::{OutputSlotId, OutputWriter, RealizedOutputRoute, SlotSource};
pub type ConstantDef<P = Semantic, Lang = WynLanguage> = super::ir::ConstantDef<P, Lang>;
pub type AllocatedConstantDef<Lang = WynLanguage> =
    super::ir::ConstantDef<Semantic<SemanticResourceRef>, Lang>;
pub use crate::types::ExternDecl;
pub type Func<P = Semantic, Lang = WynLanguage> = super::ir::Func<P, Lang>;
pub type AllocatedFunc<Lang = WynLanguage> = super::ir::Func<Semantic<SemanticResourceRef>, Lang>;
pub type Entry<
    P = Semantic,
    ResourceDecl = NoStorageDeclaration,
    Route = RealizedOutputRoute,
    Lang = WynLanguage,
> = super::ir::Entry<P, ResourceDecl, Route, Lang>;
pub type RawEntry<Route = RealizedOutputRoute> = Entry<Raw, NoStorageDeclaration, Route>;
pub type AllocatedEntry<Route = RealizedOutputRoute> =
    Entry<Semantic<SemanticResourceRef>, SemanticResourceDecl, Route>;
pub type Program<Tag, Shape, GlobalContext, Lang = WynLanguage> =
    super::ir::Program<Tag, Shape, GlobalContext, Lang>;

pub(crate) fn fresh_region_name(identities: &ProgramIdentities, base: &str) -> String {
    let is_available = |candidate: &str| identities.function_names().all(|name| name != candidate);
    if is_available(base) {
        return base.to_string();
    }
    for suffix in 1.. {
        let candidate = format!("{base}_{suffix}");
        if is_available(&candidate) {
            return candidate;
        }
    }
    unreachable!()
}

impl<Tag, Shape, GlobalContext> super::ir::Program<Tag, Shape, GlobalContext, WynLanguage>
where
    Shape: super::ir::ProgramShape,
{
    pub fn contains_region(&self, region: FunctionId) -> bool {
        self.functions.iter().any(|function| function.region == region)
    }

    pub fn region(&self, region: FunctionId) -> Option<&super::ir::Func<Shape::Family, WynLanguage>> {
        self.functions.iter().find(|function| function.region == region)
    }

    pub fn iter_regions(
        &self,
    ) -> impl Iterator<Item = (FunctionId, &super::ir::Func<Shape::Family, WynLanguage>)> {
        self.functions.iter().map(|function| (function.region, function))
    }
}

impl<Tag>
    super::ir::Program<
        Tag,
        super::ir::ProgramFamily<
            Semantic<SemanticResourceRef>,
            SemanticResourceDecl,
            RealizedOutputRoute,
            ResourceProgramData,
        >,
        RewriteGlobal,
        WynLanguage,
    >
{
    pub fn region_name(&self, region: FunctionId) -> &str {
        self.data.identities.function_name(region)
    }
}

impl<Tag> Index<EntryId>
    for super::ir::Program<
        Tag,
        super::ir::ProgramFamily<
            Semantic<SemanticResourceRef>,
            SemanticResourceDecl,
            RealizedOutputRoute,
            ResourceProgramData,
        >,
        RewriteGlobal,
        WynLanguage,
    >
{
    type Output = AllocatedEntry;

    fn index(&self, id: EntryId) -> &Self::Output {
        &self.entry_points[id.index()]
    }
}

#[cfg(test)]
#[path = "program_tests.rs"]
mod program_tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SemanticOpId {
    source: u32,
    implementation: Option<u32>,
}

impl SemanticOpId {
    /// Dense identity of the source semantic operation. This remains stable
    /// across semantic rewrites and is suitable for diagnostics and tooling.
    pub const fn source_index(self) -> u32 {
        self.source
    }

    /// Compiler-created helper slot, when this operation implements another
    /// semantic operation rather than corresponding directly to source work.
    pub const fn implementation_slot(self) -> Option<u32> {
        self.implementation
    }

    #[cfg(test)]
    pub(crate) const fn for_test(index: u32) -> Self {
        Self {
            source: index,
            implementation: None,
        }
    }

    /// Identify a compiler-created operation by the semantic operation whose
    /// implementation requires it. The slot distinguishes multiple helpers
    /// without reopening or reconstructing the source ID sequence.
    pub(crate) const fn implementation(self, slot: u32) -> Self {
        Self {
            source: self.source,
            implementation: Some(slot),
        }
    }
}

impl From<u32> for SemanticOpId {
    fn from(index: u32) -> Self {
        Self {
            source: index,
            implementation: None,
        }
    }
}

pub(crate) type SemanticOpIdSource = IdSource<SemanticOpId>;

/// Uninhabited entry-declaration payload before logical-resource allocation.
/// Authored storage remains in the entry interface; compiler-created storage
/// cannot be represented in `Converted`, `Segmented`, or `Optimized` EGIR.
#[derive(Clone, Debug)]
pub enum NoStorageDeclaration {}

/// Target-independent identity of a semantic storage resource. Identities are
/// issued only by logical-resource allocation and committed by the arena;
/// callers can observe an id's dense index but cannot manufacture one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResourceId(u32);

impl ResourceId {
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Reserve the dense identity that logical-resource allocation will
    /// commit at this arena position.
    pub(in crate::egir) const fn for_allocation(index: usize) -> Self {
        Self(index as u32)
    }

    #[cfg(test)]
    pub(crate) const fn for_test(index: u32) -> Self {
        Self(index)
    }
}

/// Stable identity of a semantic requirement to materialize a shared value.
/// It is deliberately distinct from `EntryId`: a requirement is not
/// an entry point and cannot be mutated by semantic entry passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MaterializationId(pub u32);

impl From<u32> for MaterializationId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl MaterializationId {
    /// Backend-visible name for the synthetic entry owned by this
    /// materialization. Keeping the authored owner as the prefix preserves a
    /// useful naming convention for existing tooling; explicit stage-owner
    /// metadata remains the authoritative relationship.
    pub(crate) fn entry_name(self, source: &str, role: &str) -> String {
        format!("{source}_{role}_{}", self.0)
    }
}

/// Stable identity of an entry input position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InputSlotId(pub usize);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogicalSize {
    FixedBytes(u64),
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

impl LogicalSize {
    /// Size storage for one value per point in a semantic segmented space.
    /// Returns `None` when the element has no legal storage layout.
    pub(crate) fn for_space(
        space: &SegSpace<SemanticResourceRef>,
        elem_ty: &Type<TypeName>,
    ) -> Option<Self> {
        let elem_bytes = ssa::layout::storage_elem_stride(elem_ty)?;
        if let Some(count) = space.dims().iter().try_fold(1u64, |count, extent| match extent {
            SegExtent::Fixed(length) => count.checked_mul(u64::from(*length)),
            _ => None,
        }) {
            return Some(Self::FixedBytes(count.saturating_mul(u64::from(elem_bytes))));
        }
        Some(match space.dims() {
            [SegExtent::ResourceLength {
                resource,
                elem_bytes: source_elem_bytes,
                ..
            }] => Self::LikeResource {
                resource: resource.0,
                elem_bytes,
                src_elem_bytes: *source_elem_bytes,
            },
            _ => Self::SameAsDispatch { elem_bytes },
        })
    }
}

/// A semantic storage identity. It cannot represent a backend binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SemanticResourceRef(pub ResourceId);

/// Entry-local use of a logical resource. Unlike `StorageBindingDecl`, this is
/// target independent after allocation and cannot assign a descriptor binding
/// to a compiler-created resource.
#[derive(Clone, Debug)]
pub struct SemanticResourceDecl {
    pub resource: SemanticResourceRef,
    pub role: interface::StorageRole,
}

/// Why a compiler-introduced resource exists. The kind fixes its physical
/// storage role and lets descriptor publication build the right
/// `StorageBindingDecl` without re-deriving it from the lowering site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompilerResourceKind {
    /// A pre-existing generic intermediate surfaced from a
    /// `StorageBindingDecl` and not owned by a Seg op.
    Staging,
    /// Array result produced by a compiler-hoisted gather prepass.
    GatherHandoff,
    /// One per-accumulator partial buffer of a parallel `SegRed`.
    ReducePartial,
    /// Block-level scratch buffers of a parallel `SegScan`.
    ScanBlockSums,
    ScanBlockOffsets,
    /// Per-element local prefixes retained until phase 3 applies global offsets.
    ScanPrefixes,
    /// A runtime `filter`'s compaction buffer and its paired length cell.
    FilterScratch,
    FilterLenCell,
    FilterFlags,
    FilterOffsets,
    FilterScanBlockSums,
    FilterScanBlockOffsets,
    /// Per-bucket populations and the one-cell overflow flag used by
    /// capacity-bounded bucket insertion.
    BucketCounts,
    BucketOverflow,
    /// Scalar result produced by a compiler-hoisted prepass and consumed by a
    /// later source entry phase.
    ScalarHandoff,
    /// One shared materialization for an array-valued SegMap with more than
    /// one semantic consumer.
    MultiConsumerArray,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompilerResource {
    pub kind: CompilerResourceKind,
    /// Semantic operation that owns the resource. Generic staging resources
    /// introduced before segmentation have no single owner.
    pub owner: Option<SemanticOpId>,
    /// Stable resource position within the owner (accumulator/lane/scratch
    /// index, depending on `kind`).
    pub slot: usize,
}

/// Arena key for an operation-owned compiler resource. The arena assigns at
/// most one logical resource to each key, so target recipes can retain the
/// returned id instead of rediscovering it from the manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CompilerResourceKey {
    pub(crate) owner: SemanticOpId,
    pub(crate) kind: CompilerResourceKind,
    pub(crate) slot: usize,
}

impl CompilerResource {
    pub fn new(kind: CompilerResourceKind, owner: Option<SemanticOpId>, slot: usize) -> Self {
        Self { kind, owner, slot }
    }

    pub(crate) fn key(&self) -> Option<CompilerResourceKey> {
        Some(CompilerResourceKey {
            owner: self.owner?,
            kind: self.kind,
            slot: self.slot,
        })
    }
}

#[derive(Clone, Debug)]
pub struct HostResource {
    pub binding: BindingRef,
    /// Source-level name published for this binding, when one exists.
    pub name: Option<String>,
}

#[derive(Clone, Debug)]
pub enum ResourceOrigin {
    Host(HostResource),
    Compiler(CompilerResource),
}

impl ResourceOrigin {
    pub fn host(binding: BindingRef) -> Self {
        Self::Host(HostResource { binding, name: None })
    }
}

#[derive(Clone, Debug)]
pub struct LogicalResource {
    /// Dense planning-session identity. Compiler-owned ids may change when
    /// target recipes change and must not be treated as host ABI bindings.
    id: ResourceId,
    pub origin: ResourceOrigin,
    pub elem_ty: Type<TypeName>,
    pub size: LogicalSize,
}

impl LogicalResource {
    pub fn id(&self) -> ResourceId {
        self.id
    }

    pub fn host_binding(&self) -> Option<BindingRef> {
        match &self.origin {
            ResourceOrigin::Host(host) => Some(host.binding),
            ResourceOrigin::Compiler(_) => None,
        }
    }
}

/// Dense logical-resource storage. Resource identities are assigned only by
/// this arena, so a manifest cannot contain duplicate, sparse, or mismatched
/// ids. The resource payload remains mutable, but its identity does not.
#[derive(Clone, Debug, Default)]
pub struct LogicalResourceArena {
    resources: Vec<LogicalResource>,
    host: HashMap<BindingRef, ResourceId>,
    compiler: HashMap<CompilerResourceKey, ResourceId>,
}

impl LogicalResourceArena {
    pub(crate) fn allocate(
        &mut self,
        origin: ResourceOrigin,
        elem_ty: Type<TypeName>,
        size: LogicalSize,
    ) -> ResourceId {
        let existing = match &origin {
            ResourceOrigin::Host(host) => self.host.get(&host.binding).copied(),
            ResourceOrigin::Compiler(compiler) => {
                compiler.key().and_then(|key| self.compiler.get(&key).copied())
            }
        };
        if let Some(id) = existing {
            return id;
        }
        let id = ResourceId(self.resources.len() as u32);
        match &origin {
            ResourceOrigin::Host(host) => {
                self.host.insert(host.binding, id);
            }
            ResourceOrigin::Compiler(compiler) => {
                if let Some(key) = compiler.key() {
                    self.compiler.insert(key, id);
                }
            }
        }
        self.resources.push(LogicalResource {
            id,
            origin,
            elem_ty,
            size,
        });
        id
    }

    pub(crate) fn host_resource(&self, binding: BindingRef) -> Option<ResourceId> {
        self.host.get(&binding).copied()
    }

    pub(crate) fn host_bindings(&self) -> impl Iterator<Item = BindingRef> + '_ {
        self.host.keys().copied()
    }

    #[cfg(test)]
    pub(crate) fn compiler_resource(
        &self,
        owner: SemanticOpId,
        kind: CompilerResourceKind,
        slot: usize,
    ) -> Option<ResourceId> {
        self.compiler.get(&CompilerResourceKey { owner, kind, slot }).copied()
    }

    pub(crate) fn reclassify_as_compiler(&mut self, id: ResourceId, compiler: CompilerResource) {
        let resource = &mut self.resources[id.index()];
        if let ResourceOrigin::Host(host) = &resource.origin {
            self.host.remove(&host.binding);
        }
        if let Some(key) = compiler.key() {
            self.compiler.insert(key, id);
        }
        resource.origin = ResourceOrigin::Compiler(compiler);
    }

    pub(crate) fn contains(&self, id: ResourceId) -> bool {
        id.index() < self.resources.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, LogicalResource> {
        self.resources.iter()
    }

    pub fn ids(&self) -> impl Iterator<Item = ResourceId> + '_ {
        self.resources.iter().map(LogicalResource::id)
    }

    pub fn len(&self) -> usize {
        self.resources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }
}

impl Index<ResourceId> for LogicalResourceArena {
    type Output = LogicalResource;

    fn index(&self, id: ResourceId) -> &Self::Output {
        &self.resources[id.index()]
    }
}

impl IndexMut<ResourceId> for LogicalResourceArena {
    fn index_mut(&mut self, id: ResourceId) -> &mut Self::Output {
        &mut self.resources[id.index()]
    }
}

impl Index<usize> for LogicalResourceArena {
    type Output = LogicalResource;

    fn index(&self, index: usize) -> &Self::Output {
        &self.resources[index]
    }
}

impl Deref for LogicalResourceArena {
    type Target = [LogicalResource];

    fn deref(&self) -> &Self::Target {
        &self.resources
    }
}

impl<'a> IntoIterator for &'a LogicalResourceArena {
    type Item = &'a LogicalResource;
    type IntoIter = std::slice::Iter<'a, LogicalResource>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut LogicalResourceArena {
    type Item = &'a mut LogicalResource;
    type IntoIter = std::slice::IterMut<'a, LogicalResource>;

    fn into_iter(self) -> Self::IntoIter {
        self.resources.iter_mut()
    }
}

pub(crate) fn host_resource_names(resources: &[LogicalResource]) -> LookupMap<(u32, u32), String> {
    resources
        .iter()
        .filter_map(|resource| match &resource.origin {
            ResourceOrigin::Host(host) => {
                Some(((host.binding.set, host.binding.binding), host.name.clone()?))
            }
            ResourceOrigin::Compiler(_) => None,
        })
        .collect()
}

fn semantic_type_resource(ty: &Type<TypeName>) -> Option<SemanticResourceRef> {
    let Type::Constructed(TypeName::Resource(resource), _) = ty.array_buffer()? else {
        return None;
    };
    Some(SemanticResourceRef(*resource))
}

pub(crate) fn visit_type_names_mut(ty: &mut Type<TypeName>, mut visit: impl FnMut(&mut TypeName)) {
    fn recurse(ty: &mut Type<TypeName>, visit: &mut impl FnMut(&mut TypeName)) {
        let Type::Constructed(name, arguments) = ty else {
            return;
        };
        visit(name);
        if let TypeName::Sum(variants) = name {
            for field in variants.iter_mut().flat_map(|(_, fields)| fields) {
                recurse(field, visit);
            }
        }
        for argument in arguments {
            recurse(argument, visit);
        }
    }
    recurse(ty, &mut visit);
}

fn rewrite_physical_graph_types(
    graph: &mut EGraph<Physical>,
    mut rewrite: impl FnMut(&mut Type<TypeName>),
) {
    for block in graph.skeleton.blocks.values_mut() {
        for effect in &mut block.side_effects {
            if let super::types::SideEffectKind::Soac(SoacEffect(_, soac)) = &mut effect.kind {
                soac.for_each_type_mut(&mut rewrite);
            }
        }
    }
    rewrite_node_types(graph, rewrite);
}

pub(crate) fn rewrite_graph_types<R: super::types::GraphResource>(
    graph: &mut EGraph<Semantic<R>>,
    mut rewrite: impl FnMut(&mut Type<TypeName>),
) {
    for block in graph.skeleton.blocks.values_mut() {
        for effect in &mut block.side_effects {
            if let super::types::SideEffectKind::Soac(SoacEffect(_, soac)) = &mut effect.kind {
                soac.for_each_type_mut(&mut rewrite);
            }
        }
    }
    rewrite_node_types(graph, rewrite);
}

fn rewrite_node_types<P: Family>(graph: &mut EGraph<P>, mut rewrite: impl FnMut(&mut Type<TypeName>)) {
    for node in graph.nodes.keys().collect::<Vec<_>>() {
        let mut ty = graph.nodes[node].ty.clone();
        rewrite(&mut ty);
        graph.retype_node(node, ty);
    }
}

fn physicalize_soac(
    soac: Soac<Scheduled>,
    nodes: &LookupMap<ValueId, ValueId>,
    places: &LookupMap<super::ir::PlaceId, super::ir::PlaceId>,
    bindings: &PhysicalResourceTable,
) -> Result<Soac<Physical>, String> {
    let mut remap = super::soac::remap::Remap::new(nodes, places, |reference: SemanticResourceRef| {
        Ok::<_, String>(bindings.binding(reference.0))
    });
    Ok(match soac {
        Soac::Screma(screma::Op {
            inputs,
            form,
            result_state,
            state,
        }) => {
            let map_shaped = form.scans.is_empty() && form.reductions.is_empty();
            let state = match state {
                screma::ScheduledState::Serial => screma::PhysicalState::Serial,
                screma::ScheduledState::Segmented(segment) if map_shaped => {
                    screma::PhysicalState::Segmented(remap.segment(segment)?)
                }
                screma::ScheduledState::Segmented(_) => {
                    return Err("scheduled parallel fold reached physicalization; split it into physical kernels first".into());
                }
            };
            Soac::Screma(screma::Op {
                inputs,
                form: remap.screma_form(form),
                result_state,
                state,
            })
        }
        Soac::Filter(filter::Op { body, state }) => {
            let state = match state {
                filter::ScheduledState::Loop {
                    space: iteration_space,
                    storage,
                } => filter::ScheduledState::Loop {
                    space: remap.space(iteration_space)?,
                    storage: remap.filter_output(storage)?,
                },
                filter::ScheduledState::Pipeline {
                    space: iteration_space,
                    storage,
                    plan,
                } => filter::ScheduledState::Pipeline {
                    space: remap.space(iteration_space)?,
                    storage: remap.runtime_storage(storage)?,
                    plan: filter::ParallelPlan {
                        stage: plan.stage,
                        buffers: remap.work_buffers(plan.buffers)?,
                        scan_workgroup_width: plan.scan_workgroup_width,
                    },
                },
            };
            Soac::Filter(filter::Op {
                body: remap.filter_body(body),
                state,
            })
        }
        Soac::Hist(hist::Op { inputs, form, state }) => {
            let state = match state {
                hist::ScheduledState::Serial => hist::ScheduledState::Serial,

                hist::ScheduledState::Atomic {
                    space: iteration_space,
                    operations,
                } => hist::ScheduledState::Atomic {
                    space: remap.space(iteration_space)?,
                    operations,
                },
                hist::ScheduledState::Bucket {
                    space: iteration_space,
                    stage,
                    topology,
                    storage,
                } => hist::ScheduledState::Bucket {
                    space: remap.space(iteration_space)?,
                    stage,
                    topology,
                    storage: remap.bucket_storage(storage)?,
                },
            };
            Soac::Hist(hist::Op {
                inputs,
                form: remap.hist_form(form),
                state,
            })
        }
    })
}

pub(crate) fn physicalize_graph_resources(
    graph: EGraph<Scheduled>,
    bindings: &PhysicalResourceTable,
) -> Result<
    (
        EGraph<Physical>,
        LookupMap<ValueId, ValueId>,
        LookupMap<BlockId, BlockId>,
    ),
    String,
> {
    let (mut graph, node_map, block_map) = graph.try_map_resources_and_phase(
        |reference| {
            let resource = reference.0;
            Ok::<_, String>(bindings.binding(resource))
        },
        |id, soac, nodes, places| physicalize_soac(soac, nodes, places, bindings).map(|soac| (id, soac)),
    )?;
    let pure_nodes = graph.nodes.keys().collect::<Vec<_>>();
    for node in pure_nodes {
        let resource_len = match graph.nodes.get(node).map(|node| &node.kind) {
            Some(super::types::ValueKind::Pure {
                op: super::types::PureOp::ResourceLen(binding),
                ..
            }) => Some(*binding),
            _ => None,
        };
        if let Some(binding) = resource_len {
            let set = super::graph_ops::intern_u32(&mut graph, binding.set, None);
            let slot = super::graph_ops::intern_u32(&mut graph, binding.binding, None);
            graph.replace_pure_node(
                node,
                super::types::PureOp::Intrinsic {
                    id: builtins::catalog().known().storage_len,
                    overload_idx: 0,
                },
                smallvec::smallvec![set, slot],
            );
            continue;
        }
    }
    rewrite_physical_graph_types(&mut graph, |ty| physicalize_type_resources(ty, bindings));
    graph.canonicalize_boundary_operands();
    Ok((graph, node_map, block_map))
}

pub(crate) fn physicalize_type_resources(ty: &mut Type<TypeName>, bindings: &PhysicalResourceTable) {
    visit_type_names_mut(ty, |name| {
        if let TypeName::Resource(resource) = *name {
            *name = TypeName::Buffer(bindings.binding(resource));
        }
    });
}

/// Verify the allocation typestate. From this boundary through validation,
/// every executable storage identity is a `ResourceId`; bindings survive only
/// in the host ABI fields and `ResourceOrigin::Host` constraints.
/// Physical `BufferLen` for a logical size, or `None` for `Unspecified` (a
/// host-supplied length). Inverse of `logical_size`, used when a compiler
/// resource is published as a `StorageBindingDecl`.
pub fn buffer_len(
    size: &LogicalSize,
    resources: &PhysicalResourceTable,
) -> Option<pipeline_descriptor::BufferLen> {
    use crate::pipeline_descriptor::BufferLen;
    match size {
        LogicalSize::FixedBytes(bytes) => Some(BufferLen::Fixed { bytes: *bytes }),
        LogicalSize::LikeResource {
            resource,
            elem_bytes,
            src_elem_bytes,
        } => {
            let binding = resources.binding(*resource);
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

#[cfg(test)]
pub(crate) fn semantic_program_for_test(
    functions: Vec<Func<Semantic>>,
    externs: Vec<ExternDecl<Type<TypeName>>>,
    entry_points: Vec<Entry<Semantic>>,
    constants: Vec<ConstantDef<Semantic>>,
    pipeline: PipelineDescriptor,
    identities: ProgramIdentities,
) -> super::reify::Segmented {
    Program::from_parts(
        functions,
        externs,
        entry_points,
        constants,
        SemanticProgramData {
            pipeline,
            stage_entries: Vec::new(),
            identities,
        },
        RewriteGlobal {
            binding_ids: IdSource::new(),
            effect_ids: IdSource::new(),
            semantic_ids: IdSource::new(),
        },
    )
}

impl AllocatedEntry {
    /// Resource identities referenced by a set of values in `graph`, including
    /// resource-backed entry parameters whose identity is carried by the
    /// interface rather than by a storage-view node.
    pub(crate) fn resources_referenced_by_nodes(
        &self,
        graph: &EGraph<Semantic<SemanticResourceRef>>,
        nodes: impl IntoIterator<Item = ValueId>,
    ) -> HashSet<ResourceId> {
        let mut resources = HashSet::new();
        for node in nodes {
            if let Some(ValueKind::Pure { op, .. }) = graph.nodes.get(node).map(|node| &node.kind) {
                if let Some(resource) = op.referenced_resource() {
                    resources.insert(resource.0);
                }
            }
            if let Some(ValueKind::FuncParam { parameter }) = graph.nodes.get(node).map(|node| &node.kind) {
                resources.extend(
                    self.params()
                        .abi_position(*parameter)
                        .and_then(|position| self.inputs.get(position))
                        .and_then(|input| input.resource.or_else(|| semantic_type_resource(&input.ty)))
                        .map(|resource| resource.0),
                );
            }
        }
        resources
    }

    /// Resource identities retained by a graph projection. The projection is
    /// the authority for which source values and effects survived, so callers
    /// do not need to rediscover that boundary from the projected graph.
    pub(crate) fn resources_referenced_by_projection(
        &self,
        projection: &super::graph_projector::GraphProjection<SemanticResourceRef>,
    ) -> HashSet<ResourceId> {
        let mut resources = self.resources_referenced_by_nodes(&self.graph, projection.source_nodes());
        for site in projection.source_effects() {
            let effect = self.graph.skeleton.effect(*site);
            if let SideEffectKind::Effect(operation) = &effect.kind {
                if let Some(resource) = operation.referenced_resource() {
                    resources.insert(resource.0);
                }
            }
            resources.extend(
                super::semantic_graph::read_resources(&self.graph, effect)
                    .into_iter()
                    .map(|access| access.resource.0),
            );
            if let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind {
                if let screma::SemanticState::Segmented {
                    resources: accesses, ..
                } = op.semantic_state()
                {
                    resources.extend(accesses.iter().map(|access| access.resource.0));
                }
            }
        }
        resources
    }

    pub(crate) fn parameter_indices_referenced_by_projection(
        &self,
        projection: &super::graph_projector::GraphProjection<SemanticResourceRef>,
        resources: &HashSet<ResourceId>,
    ) -> SortedSet<usize> {
        let mut parameters = projection
            .source_nodes()
            .filter_map(|node| match self.graph.nodes.get(node).map(|node| &node.kind) {
                Some(ValueKind::FuncParam { parameter }) => self.params().abi_position(*parameter),
                _ => None,
            })
            .collect::<SortedSet<_>>();
        for (index, input) in self.inputs.iter().enumerate() {
            if input
                .resource
                .or_else(|| semantic_type_resource(&input.ty))
                .is_some_and(|resource| resources.contains(&resource.0))
            {
                parameters.insert(index);
            }
        }
        parameters
    }

    pub(crate) fn resource_declarations_for(
        &self,
        resources: &HashSet<ResourceId>,
    ) -> Vec<SemanticResourceDecl> {
        self.resource_declarations
            .iter()
            .filter(|declaration| resources.contains(&declaration.resource.0))
            .cloned()
            .collect()
    }

    pub(crate) fn set_resource_declaration(&mut self, resource: ResourceId, role: interface::StorageRole) {
        if let Some(declaration) =
            self.resource_declarations.iter_mut().find(|declaration| declaration.resource.0 == resource)
        {
            declaration.role = role;
        } else {
            self.resource_declarations.push(SemanticResourceDecl {
                resource: SemanticResourceRef(resource),
                role,
            });
        }
    }

    pub(crate) fn declare_resource_view(
        &mut self,
        resource: ResourceId,
        role: interface::StorageRole,
        elem_ty: &Type<TypeName>,
    ) -> ValueId {
        let view = super::graph_ops::intern_resource_view(&mut self.graph, resource, elem_ty.clone(), None);
        self.set_resource_declaration(resource, role);
        view
    }

    /// Remove entry parameters and input resource declarations that the graph
    /// and output routes cannot observe.
    pub(crate) fn compact_interface(&mut self) {
        let mut roots = self
            .graph
            .skeleton
            .blocks
            .iter()
            .flat_map(|(_, block)| {
                block
                    .side_effects
                    .iter()
                    .flat_map(|effect| super::graph_ops::effect_value_inputs(&self.graph, effect))
                    .chain(block.term.referenced_nodes())
            })
            .collect::<Vec<_>>();
        for route in self.routes() {
            roots.push(route.source.value);
            roots.extend(route.writers.iter().filter_map(|writer| match writer {
                OutputWriter::Value(value) => Some(*value),
                OutputWriter::Effect(_) => None,
            }));
        }
        let reachable = super::graph_ops::execution_value_producer_closure(&self.graph, roots).nodes;
        let mut reachable_resources =
            self.resources_referenced_by_nodes(&self.graph, reachable.iter().copied());
        for (_, block) in &self.graph.skeleton.blocks {
            for effect in &block.side_effects {
                if let SideEffectKind::Effect(operation) = &effect.kind {
                    if let Some(resource) = operation.referenced_resource() {
                        reachable_resources.insert(resource.0);
                    }
                }
            }
        }
        let mut kept_indices = reachable
            .iter()
            .filter_map(|node| match self.graph.nodes.get(*node).map(|node| &node.kind) {
                Some(ValueKind::FuncParam { parameter }) => self.params().abi_position(*parameter),
                _ => None,
            })
            .collect::<SortedSet<_>>();
        for (index, input) in self.inputs.iter().enumerate() {
            if input.resource.is_some_and(|resource| reachable_resources.contains(&resource.0)) {
                kept_indices.insert(index);
            }
        }
        self.retain_parameter_indices(&kept_indices);

        let mut used_resources = self
            .inputs
            .iter()
            .filter_map(|input| input.resource.map(|resource| resource.0))
            .chain(self.outputs.iter().filter_map(|output| output.resource.map(|resource| resource.0)))
            .collect::<HashSet<_>>();
        for (_, block) in &self.graph.skeleton.blocks {
            for effect in &block.side_effects {
                used_resources.extend(
                    super::semantic_graph::read_resources(&self.graph, effect)
                        .into_iter()
                        .map(|access| access.resource.0),
                );
                if let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind {
                    if let screma::SemanticState::Segmented { resources, .. } = op.semantic_state() {
                        used_resources.extend(resources.iter().map(|access| access.resource.0));
                    }
                }
            }
        }
        self.resource_declarations.retain(|declaration| {
            declaration.role != interface::StorageRole::Input
                || used_resources.contains(&declaration.resource.0)
        });
    }
}

/// A complete, fresh entry projection owned by a kernel recipe.
#[derive(Clone, Debug)]
pub struct PlannedEntry<P: Family = Semantic<SemanticResourceRef>> {
    entry: super::ir::Entry<P, SemanticResourceDecl, RealizedOutputRoute, WynLanguage>,
    parallel_scremas: HashSet<SemanticOpId>,
}

impl<P: Family> PlannedEntry<P> {
    pub(crate) fn new(
        entry: super::ir::Entry<P, SemanticResourceDecl, RealizedOutputRoute, WynLanguage>,
    ) -> Self {
        Self {
            entry,
            parallel_scremas: HashSet::new(),
        }
    }

    pub(crate) fn with_parallel_scremas(
        mut self,
        operations: impl IntoIterator<Item = SemanticOpId>,
    ) -> Self {
        self.parallel_scremas = operations.into_iter().collect();
        self
    }

    pub(crate) fn parallel_scremas(&self) -> &HashSet<SemanticOpId> {
        &self.parallel_scremas
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        super::ir::Entry<P, SemanticResourceDecl, RealizedOutputRoute, WynLanguage>,
        HashSet<SemanticOpId>,
    ) {
        (self.entry, self.parallel_scremas)
    }

    pub(crate) fn into_inner(
        self,
    ) -> super::ir::Entry<P, SemanticResourceDecl, RealizedOutputRoute, WynLanguage> {
        self.entry
    }
}

impl<P: Family> Deref for PlannedEntry<P> {
    type Target = super::ir::Entry<P, SemanticResourceDecl, RealizedOutputRoute, WynLanguage>;

    fn deref(&self) -> &Self::Target {
        &self.entry
    }
}

impl<P: Family> std::ops::DerefMut for PlannedEntry<P> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.entry
    }
}

/// Backend-visible entry metadata retained by the plan without retaining a
/// second copy of the semantic graph.
#[derive(Clone, Debug)]
pub struct PlannedPublication {
    pub id: EntryId,
    pub name: String,
    pub execution_model: ExecutionModel,
    pub inputs: Vec<EntryInput>,
    pub outputs: Vec<EntryOutput>,
    pub resource_declarations: Vec<SemanticResourceDecl>,
}

impl PlannedPublication {
    pub fn from_semantic(entry: &AllocatedEntry) -> Self {
        Self {
            id: entry.id,
            name: entry.name.clone(),
            execution_model: entry.execution_model.clone(),
            inputs: entry.inputs.iter().map(|input| input.inner.clone()).collect(),
            outputs: entry.outputs.iter().map(|output| output.inner.clone()).collect(),
            resource_declarations: entry.resource_declarations.clone(),
        }
    }

    pub fn publication(&self, resources: &PhysicalResourceTable) -> Result<EntryPublication, String> {
        publish_entry(
            self.id,
            &self.name,
            &self.execution_model,
            &self.inputs,
            &self.outputs,
            &self.resource_declarations,
            resources,
        )
    }
}

impl AllocatedEntry {
    pub(crate) fn resource_for_result(
        &self,
        result: &super::ir::ResultBinding<Type<TypeName>>,
    ) -> Option<SemanticResourceRef> {
        resource_for_result(self, result)
    }

    pub(crate) fn bind_mapped_output_destinations(&mut self) -> Result<(), String> {
        let results = self
            .graph
            .skeleton
            .blocks
            .values()
            .flat_map(|block| &block.side_effects)
            .filter_map(|effect| {
                let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = effect.kind() else {
                    return None;
                };
                let result = effect.result()?;
                Some(
                    result
                        .top_level_fields()
                        .into_iter()
                        .enumerate()
                        .filter_map(|(field, result)| {
                            matches!(op.form.result_id(field), Some(screma::ResultId::Post(_)))
                                .then_some(result)
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .flatten()
            .collect::<Vec<_>>();
        let bindings = results
            .into_iter()
            .filter_map(|result| resource_for_result(self, &result).map(|resource| (resource, result)))
            .collect::<Vec<_>>();

        let mut replacements = Vec::new();
        for (resource, result) in bindings {
            let view = super::graph_ops::intern_resource_view(
                &mut self.graph,
                resource.0,
                result.ty().clone(),
                Some(self.span),
            );
            let destination = super::graph_ops::bind_result_to_view(&mut self.graph, &result, view)?;
            replacements.extend(super::graph_ops::rebind_result_value_references(
                &mut self.graph,
                &result,
                &destination,
            )?);
        }
        for route in self.routes_mut() {
            route.replace_values(&replacements);
        }
        Ok(())
    }
}

impl PlannedEntry {
    pub fn project(entry: &AllocatedEntry) -> Result<Self, String> {
        let projection = super::graph_projector::GraphProjector::new(&entry.graph)
            .all_with_values(entry.routes().map(|route| route.source.value).collect())
            .map_err(|error| format!("could not project semantic entry '{}': {error}", entry.name))?;
        Self::from_projection(
            projection,
            entry.id,
            entry.name.clone(),
            entry.span,
            entry.execution_model.clone(),
            entry.inputs.clone(),
            entry.parameter_inputs.clone(),
            entry.outputs.clone(),
            entry.internal_results.clone(),
            entry.resource_declarations.clone(),
            entry.params.clone(),
            entry.result.clone(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_projection(
        projection: super::graph_projector::GraphProjection<SemanticResourceRef>,
        id: EntryId,
        name: String,
        span: Span,
        execution_model: ExecutionModel,
        inputs: Vec<super::ir::EntryInput<SemanticResourceRef, WynLanguage>>,
        parameter_inputs: Vec<Vec<InputSlotId>>,
        outputs: Vec<super::ir::EntryOutput<SemanticResourceRef, RealizedOutputRoute, WynLanguage>>,
        internal_results: Vec<super::ir::InternalResultRoute<SemanticResourceRef, RealizedOutputRoute>>,
        resource_declarations: Vec<SemanticResourceDecl>,
        params: super::ir::Parameters<SemanticResourceRef, Type<TypeName>>,
        result: super::ir::FunctionResult<Type<TypeName>>,
    ) -> Result<Self, String> {
        let outputs = outputs
            .into_iter()
            .map(|mut output| {
                output.routes = projection.remap_output_routes(output.routes)?;
                Ok(output)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let internal_results = internal_results
            .into_iter()
            .map(|mut result| {
                result.route = projection.remap_output_routes(vec![result.route])?.remove(0);
                Ok(result)
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self::new(Entry {
            id,
            name,
            span,
            execution_model,
            inputs,
            parameter_inputs,
            outputs,
            internal_results,
            resource_declarations,
            params,
            result,
            graph: projection.graph,
        }))
    }
}

fn resource_for_result<P: super::ir::Family<Resource = SemanticResourceRef>>(
    entry: &super::ir::Entry<P, SemanticResourceDecl, RealizedOutputRoute, WynLanguage>,
    result: &super::ir::ResultBinding<Type<TypeName>>,
) -> Option<SemanticResourceRef> {
    entry.resource_routes().find_map(|(resource, route)| {
        route
            .referenced_values()
            .any(|value| entry.graph.value_has_result_origin(value, result))
            .then_some(*resource)
    })
}

impl<P, Route> super::ir::Entry<P, SemanticResourceDecl, Route, WynLanguage>
where
    P: Family<Resource = SemanticResourceRef>,
    Route: Clone + std::fmt::Debug,
{
    pub fn publication(&self, resources: &PhysicalResourceTable) -> Result<EntryPublication, String> {
        let inputs = self.inputs.iter().map(|input| input.inner.clone()).collect::<Vec<_>>();
        let outputs = self.outputs.iter().map(|output| output.inner.clone()).collect::<Vec<_>>();
        publish_entry(
            self.id,
            &self.name,
            &self.execution_model,
            &inputs,
            &outputs,
            &self.resource_declarations,
            resources,
        )
    }
}

fn publish_entry(
    id: EntryId,
    name: &str,
    execution_model: &ExecutionModel,
    inputs: &[EntryInput],
    outputs: &[EntryOutput],
    declarations: &[SemanticResourceDecl],
    resources: &PhysicalResourceTable,
) -> Result<EntryPublication, String> {
    let storage_bindings = declarations
        .iter()
        .filter(|declaration| resources.is_compiler(declaration.resource.0))
        .map(|declaration| interface::StorageBindingDecl {
            binding: resources.binding(declaration.resource.0),
            role: declaration.role.clone(),
            logical_resource: resources.logical_name(declaration.resource.0),
            elem_ty: resources.elem_ty(declaration.resource.0).clone(),
            length: buffer_len(resources.size(declaration.resource.0), resources),
        })
        .collect();
    Ok(EntryPublication {
        id,
        name: name.to_string(),
        execution_model: execution_model.clone(),
        inputs: inputs.to_vec(),
        outputs: outputs.to_vec(),
        storage_bindings,
    })
}

/// A semantic shared-value requirement. Nesting the single semantic entry
/// representation avoids maintaining another entry-shaped record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterializationKind {
    SharedArray,
    Gather,
    Scalar,
    /// Runtime-sized array plus a stored logical-length cell.  Producers such
    /// as filter require this layout when their result crosses a scheduling
    /// boundary; future variable-cardinality producers can reuse it.
    RuntimeArray,
}

#[derive(Debug)]
pub enum MaterializationRequirement {
    SharedArray {
        space: SegSpace<SemanticResourceRef>,
        entry: AllocatedEntry,
    },
    Gather {
        space: SegSpace<SemanticResourceRef>,
        entry: AllocatedEntry,
    },
    RuntimeArray {
        space: SegSpace<SemanticResourceRef>,
        entry: AllocatedEntry,
    },
    Scalar {
        entry: AllocatedEntry,
    },
}

impl MaterializationRequirement {
    pub fn kind(&self) -> MaterializationKind {
        match self {
            Self::SharedArray { .. } => MaterializationKind::SharedArray,
            Self::Gather { .. } => MaterializationKind::Gather,
            Self::RuntimeArray { .. } => MaterializationKind::RuntimeArray,
            Self::Scalar { .. } => MaterializationKind::Scalar,
        }
    }

    pub fn space(&self) -> Option<&SegSpace<SemanticResourceRef>> {
        match self {
            Self::SharedArray { space, .. }
            | Self::Gather { space, .. }
            | Self::RuntimeArray { space, .. } => Some(space),
            Self::Scalar { .. } => None,
        }
    }

    pub fn entry(&self) -> &AllocatedEntry {
        match self {
            Self::SharedArray { entry, .. }
            | Self::Gather { entry, .. }
            | Self::RuntimeArray { entry, .. }
            | Self::Scalar { entry } => entry,
        }
    }

    pub fn entry_mut(&mut self) -> &mut AllocatedEntry {
        match self {
            Self::SharedArray { entry, .. }
            | Self::Gather { entry, .. }
            | Self::RuntimeArray { entry, .. }
            | Self::Scalar { entry } => entry,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EntryPublication {
    /// Compiler identity. The name below remains emitted host ABI metadata.
    pub id: EntryId,
    pub name: String,
    pub execution_model: ExecutionModel,
    pub inputs: Vec<EntryInput>,
    pub outputs: Vec<EntryOutput>,
    pub storage_bindings: Vec<interface::StorageBindingDecl>,
}

/// A complete entry after a validated kernel recipe has been physicalized.
pub type PhysicalEntry =
    super::ir::Entry<Physical, interface::StorageBindingDecl, RealizedOutputRoute, WynLanguage>;

/// Deterministic allocation of logical resources to backend bindings.
#[derive(Clone, Debug, Default)]
pub struct PhysicalResourceTable {
    bindings: Vec<BindingRef>,
    compiler_owned: Vec<bool>,
    elem_types: Vec<Type<TypeName>>,
    sizes: Vec<LogicalSize>,
}

impl PhysicalResourceTable {
    /// Assign backend bindings deterministically. Host resources retain their
    /// declared ABI identities; only compiler-owned resources draw automatic
    /// bindings from `ids`.
    pub fn allocate(resources: &LogicalResourceArena, ids: &mut IdSource<u32>) -> Self {
        Self::allocate_avoiding(resources, ids, std::iter::empty())
    }

    /// Assign bindings while also reserving descriptor slots occupied by
    /// non-resource interfaces such as textures and samplers.
    pub fn allocate_avoiding(
        resources: &LogicalResourceArena,
        ids: &mut IdSource<u32>,
        reserved: impl IntoIterator<Item = BindingRef>,
    ) -> Self {
        let mut used = resources.host_bindings().collect::<std::collections::HashSet<_>>();
        used.extend(reserved);
        let mut bindings = Vec::with_capacity(resources.len());
        let mut compiler_owned = Vec::with_capacity(resources.len());
        let mut elem_types = Vec::with_capacity(resources.len());
        let mut sizes = Vec::with_capacity(resources.len());
        for resource in resources {
            compiler_owned.push(matches!(resource.origin, ResourceOrigin::Compiler(_)));
            elem_types.push(resource.elem_ty.clone());
            sizes.push(resource.size.clone());
            let binding = match &resource.origin {
                ResourceOrigin::Host(host) => host.binding,
                ResourceOrigin::Compiler(_) => loop {
                    let candidate = BindingRef::new(super::from_tlc::AUTO_STORAGE_SET, ids.next_id());
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
            elem_types,
            sizes,
        }
    }

    pub fn binding(&self, resource: ResourceId) -> BindingRef {
        self.bindings[resource.index()]
    }

    pub fn is_compiler(&self, resource: ResourceId) -> bool {
        self.compiler_owned[resource.index()]
    }

    pub fn elem_ty(&self, resource: ResourceId) -> &Type<TypeName> {
        &self.elem_types[resource.index()]
    }

    pub fn size(&self, resource: ResourceId) -> &LogicalSize {
        &self.sizes[resource.index()]
    }

    /// Descriptor-stable identity for one compiler-owned logical resource.
    /// Physical descriptor slots and entry-local names are access paths.
    pub fn logical_name(&self, resource: ResourceId) -> Option<String> {
        self.is_compiler(resource).then(|| format!("_w_resource_{}", resource.index()))
    }
}

/// Program-owned EGIR data shared by logical and physical checkpoints.
#[derive(Clone, Debug)]
pub struct ProgramIdentities {
    /// One identity realm for user, extern, lifted, and compiler-generated callables.
    functions: IdArena<FunctionId, String>,
    /// Program-level values, independent of how a backend materializes them.
    globals: IdArena<GlobalId, String>,
    /// Entry identity separate from its host-visible symbol.
    entries: IdArena<EntryId, String>,
}

impl ProgramIdentities {
    pub(crate) fn new() -> Self {
        Self {
            functions: IdArena::default(),
            globals: IdArena::default(),
            entries: IdArena::default(),
        }
    }
    pub(crate) fn alloc_function(&mut self, name: String) -> FunctionId {
        self.functions.alloc(name)
    }

    pub(crate) fn alloc_global(&mut self, name: String) -> GlobalId {
        self.globals.alloc(name)
    }

    pub(crate) fn alloc_entry(&mut self, name: String) -> EntryId {
        self.entries.alloc(name)
    }

    pub(crate) fn function_name(&self, id: FunctionId) -> &str {
        self.functions.get(id).expect("unknown function identity")
    }

    pub(crate) fn global_name(&self, id: GlobalId) -> &str {
        self.globals.get(id).expect("unknown global identity")
    }

    pub(crate) fn entry_name(&self, id: EntryId) -> &str {
        self.entries.get(id).expect("unknown entry identity")
    }

    pub(crate) fn function_names(&self) -> impl Iterator<Item = &str> {
        self.functions.values().map(String::as_str)
    }

    pub(crate) fn contains_function(&self, id: FunctionId) -> bool {
        self.functions.get(id).is_some()
    }

    pub(crate) fn contains_global(&self, id: GlobalId) -> bool {
        self.globals.get(id).is_some()
    }

    pub(crate) fn contains_entry(&self, id: EntryId) -> bool {
        self.entries.get(id).is_some()
    }
}

#[cfg(test)]
impl Default for ProgramIdentities {
    fn default() -> Self {
        Self::new()
    }
}
/// Program-owned semantic data before logical-resource allocation.
///
/// Descriptor bindings and authored storage remain in the entry interface.
/// This shape deliberately has no logical-resource arena.
#[derive(Debug)]
pub struct SemanticProgramData {
    pub pipeline: PipelineDescriptor,
    /// Structural entry identity for each descriptor pipeline stage.
    pub stage_entries: Vec<Vec<EntryId>>,
    pub identities: ProgramIdentities,
}

/// Program-owned data after logical-resource allocation. This is the first
/// program shape allowed to own a logical-resource arena.
#[derive(Debug)]
pub struct ResourceProgramData {
    pub pipeline: PipelineDescriptor,
    /// Structural entry identity for each descriptor pipeline stage.
    pub stage_entries: Vec<Vec<EntryId>>,
    pub resources: LogicalResourceArena,
    pub identities: ProgramIdentities,
}

/// Program-owned data after materialization requirements have been planned.
#[derive(Debug)]
pub struct AllocatedProgramData {
    pub core: ResourceProgramData,
    pub materializations: IdArena<MaterializationId, MaterializationRequirement>,
}

impl AllocatedProgramData {
    pub(crate) fn alloc_compiler_resource(
        &mut self,
        compiler: CompilerResource,
        elem_ty: Type<TypeName>,
        size: LogicalSize,
    ) -> ResourceId {
        self.core.resources.allocate(ResourceOrigin::Compiler(compiler), elem_ty, size)
    }
}

/// Allocators carried while EGIR graphs and logical resources are rebuilt.
#[derive(Debug)]
pub struct RewriteGlobal {
    pub binding_ids: IdSource<u32>,
    pub effect_ids: IdSource<super::types::EffectToken>,
    pub semantic_ids: SemanticOpIdSource,
}

/// Non-tree state retained after target-specific planning.
#[derive(Debug)]
pub struct PlannedGlobal {
    pub kernel_plan: super::parallelize::KernelPlanSummary,
    pub profile: LoweringProfile,
    pub effect_ids: IdSource<super::types::EffectToken>,
    pub semantic_ids: SemanticOpIdSource,
}

fn physicalize_function(
    function: AllocatedFunc,
    resources: &PhysicalResourceTable,
    serial: bool,
) -> Result<Func<Physical>, String> {
    let Func {
        region,
        name,
        span,
        linkage_name,
        params,
        result,
        effects,
        graph,
    } = function;
    let (graph, _) = super::parallelize::prepare::graph(graph, serial)?;
    let (graph, _, _) = physicalize_graph_resources(graph, resources)?;
    let params = params.map(
        |resource| resources.binding(resource.0),
        |mut ty| {
            physicalize_type_resources(&mut ty, resources);
            ty
        },
    );
    let result = result.map(
        |mut ty| {
            physicalize_type_resources(&mut ty, resources);
            ty
        },
        |slot| slot,
        |parameter| parameter,
    );
    Ok(Func {
        region,
        name,
        span,
        linkage_name,
        params,
        result,
        effects,
        graph,
    })
}

fn physicalize_constant(
    constant: AllocatedConstantDef,
    resources: &PhysicalResourceTable,
) -> Result<ConstantDef<Physical>, String> {
    let ConstantDef {
        id,
        name,
        span,
        mut return_ty,
        graph,
    } = constant;
    let (graph, _) = super::parallelize::prepare::graph(graph, false)?;
    let (graph, _, _) = physicalize_graph_resources(graph, resources)?;
    physicalize_type_resources(&mut return_ty, resources);
    Ok(ConstantDef {
        id,
        name,
        span,
        return_ty,
        graph,
    })
}

fn route_writes_resource(
    graph: &EGraph<Scheduled>,
    route: &RealizedOutputRoute,
    resource: SemanticResourceRef,
) -> bool {
    graph.skeleton.blocks.values().flat_map(|block| &block.side_effects).any(|effect| {
        let token_is_writer = effect
            .effects()
            .is_some_and(|(_, output)| route.writers.contains(&OutputWriter::Effect(output)));
        let value_is_writer = graph.effect_result_binding(effect).is_some_and(|result| {
            result.values().into_iter().any(|value| route.writers.contains(&OutputWriter::Value(value)))
        });
        if !token_is_writer && !value_is_writer {
            return false;
        }
        match effect.kind() {
            SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => {
                matches!(&op.state, screma::ScheduledState::Segmented(segment)
                if segment.resources.iter().any(|access| {
                    access.resource == resource && access.access != ResourceAccess::Read
                }))
            }
            SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) => match &op.state {
                filter::ScheduledState::Loop { storage, .. } => {
                    matches!(storage, filter::Output::Runtime(runtime)
                        if runtime.backing == filter::RuntimeBacking::Bound(resource))
                }
                filter::ScheduledState::Pipeline { storage, .. } => storage.data == resource,
            },
            _ => false,
        }
    })
}

fn emit_entry_output_writes(
    entry: &mut PlannedEntry<Scheduled>,
    effect_ids: &mut IdSource<super::types::EffectToken>,
) -> Result<(), String> {
    for slot in 0..entry.outputs.len() {
        let entry_span = entry.span;
        let output_ty = entry.outputs[slot].ty.clone();
        let resource = entry.outputs[slot].resource;
        let routes = entry.outputs[slot].routes.clone();
        let mut writers_by_route = Vec::with_capacity(routes.len());
        for route in routes {
            let source = entry.graph.canonical_value(route.source.value);
            let writes_resource =
                resource.is_some_and(|resource| route_writes_resource(&entry.graph, &route, resource));
            let mut writers = route.writers;
            if let Some(resource) = resource {
                if super::graph_ops::extract_storage_view_source(&entry.graph, source) == Some(resource)
                    || writes_resource
                {
                    writers.push(OutputWriter::Value(source));
                    let mut seen = HashSet::new();
                    writers.retain(|writer| seen.insert(*writer));
                    writers_by_route.push(writers);
                    continue;
                }
                writers.extend(
                    super::graph_ops::emit_resource_write(
                        &mut entry.graph,
                        route.source.block,
                        resource.0,
                        source,
                        &output_ty,
                        effect_ids,
                        Some(entry_span),
                    )
                    .map_err(|error| format!("entry output {slot}: {error}"))?
                    .into_iter()
                    .map(OutputWriter::Effect),
                );
            } else {
                let place = entry.graph.add_output_place(
                    slot,
                    super::types::PlaceType {
                        pointee: output_ty.clone(),
                        region: super::types::PlaceRegion::Output,
                        access: super::types::PlaceAccess::WriteOnly,
                    },
                );
                writers.push(OutputWriter::Effect(super::graph_ops::emit_store(
                    &mut entry.graph,
                    route.source.block,
                    place,
                    source,
                    effect_ids,
                    Some(entry_span),
                )));
            }
            let mut seen = HashSet::new();
            writers.retain(|writer| seen.insert(*writer));
            writers_by_route.push(writers);
        }
        for (route, writers) in entry.outputs[slot].routes.iter_mut().zip(writers_by_route) {
            route.writers = writers;
        }
    }
    for (_, block) in &mut entry.graph.skeleton.blocks {
        if matches!(block.term, super::types::SkeletonTerminator::Return(Some(_))) {
            block.term = super::types::SkeletonTerminator::Return(None);
        }
    }
    entry.result = super::types::by_value_function_result::<WynLanguage>(Type::Constructed(
        TypeName::Unit,
        Vec::new(),
    ));
    Ok(())
}

fn physicalize_entry(
    mut entry: PlannedEntry<Scheduled>,
    resources: &PhysicalResourceTable,
    effect_ids: &mut IdSource<super::types::EffectToken>,
) -> Result<PhysicalEntry, String> {
    emit_entry_output_writes(&mut entry, effect_ids)?;
    let Entry {
        id,
        name,
        span,
        execution_model,
        inputs,
        parameter_inputs,
        outputs,
        internal_results: _,
        resource_declarations: declarations,
        params,
        result,
        graph,
    } = entry.into_inner();
    let (graph, nodes, blocks) = physicalize_graph_resources(graph, resources)?;
    let inputs = inputs
        .into_iter()
        .map(|mut input| {
            physicalize_type_resources(&mut input.ty, resources);
            let resource = input.resource.map(|resource| resources.binding(resource.0));
            super::ir::EntryInput {
                inner: input.inner,
                resource,
            }
        })
        .collect();
    let outputs = outputs
        .into_iter()
        .map(|mut output| {
            physicalize_type_resources(&mut output.ty, resources);
            let resource = output.resource.map(|resource| resources.binding(resource.0));
            let routes = super::graph_projector::remap_output_routes(
                output.routes,
                |node| nodes.get(&node).copied(),
                |block| blocks.get(&block).copied(),
                Some,
                true,
                "physicalization",
            )?;
            Ok(super::ir::EntryOutput {
                inner: output.inner,
                resource,
                routes,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let params = params.map(
        |resource| resources.binding(resource.0),
        |mut ty| {
            physicalize_type_resources(&mut ty, resources);
            ty
        },
    );
    let result = result.map(
        |mut ty| {
            physicalize_type_resources(&mut ty, resources);
            ty
        },
        |slot| slot,
        |parameter| parameter,
    );
    let resource_declarations = declarations
        .into_iter()
        .map(|declaration| {
            let mut elem_ty = resources.elem_ty(declaration.resource.0).clone();
            physicalize_type_resources(&mut elem_ty, resources);
            interface::StorageBindingDecl {
                binding: resources.binding(declaration.resource.0),
                role: declaration.role,
                logical_resource: resources.logical_name(declaration.resource.0),
                elem_ty,
                length: buffer_len(resources.size(declaration.resource.0), resources),
            }
        })
        .collect();
    Ok(PhysicalEntry {
        id,
        name,
        span,
        execution_model,
        inputs,
        parameter_inputs,
        outputs,
        internal_results: Vec::new(),
        resource_declarations,
        params,
        result,
        graph,
    })
}

pub(in crate::egir) fn physicalize_program(
    program: super::allocation::ResourcesAllocated,
    entries: impl IntoIterator<Item = PlannedEntry<Scheduled>>,
    physical_resources: &PhysicalResourceTable,
    serial: bool,
    kernel_plan: super::parallelize::KernelPlanSummary,
    profile: LoweringProfile,
) -> Result<super::parallelize::Planned, String> {
    let Program {
        functions,
        externs,
        entry_points: _,
        constants,
        data,
        mut global_context,
        state: _,
    } = program;
    let entry_points = entries
        .into_iter()
        .map(|entry| physicalize_entry(entry, physical_resources, &mut global_context.effect_ids))
        .collect::<Result<Vec<_>, _>>()?;
    let functions = functions
        .into_iter()
        .map(|function| physicalize_function(function, physical_resources, serial))
        .collect::<Result<Vec<_>, _>>()?;
    let constants = constants
        .into_iter()
        .map(|constant| physicalize_constant(constant, physical_resources))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Program::from_parts(
        functions,
        externs,
        entry_points,
        constants,
        ResourceProgramData {
            pipeline: data.core.pipeline,
            stage_entries: data.core.stage_entries,
            resources: data.core.resources,
            identities: data.core.identities,
        },
        PlannedGlobal {
            kernel_plan,
            profile,
            effect_ids: global_context.effect_ids,
            semantic_ids: global_context.semantic_ids,
        },
    ))
}
