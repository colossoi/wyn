//! Compiler-specific EGIR program data and per-body records.
//!
//! EGIR typestate aliases explicitly select their recursive graph family,
//! entry-boundary data, program-owned data, and global context. This module
//! defines the concrete resource arenas, identifiers, and program data used
//! by those states.

use crate::LookupMap;

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
    EGraph, ENode, Family, NodeId, Physical, Raw, RegionId, Scheduled, SegBody, SegExtent, SegSpace,
    Semantic, SideEffectKind, Soac, SoacEffect, WynLanguage,
};

pub use super::ir::{
    OutputSlotId, OutputWriter, RealizedOutputRoute, RegionInterner, SlotSource, UnrealizedOutputRoute,
};
pub type ConstantDef<P = Semantic, Lang = WynLanguage> = super::ir::ConstantDef<P, Lang>;
pub use crate::types::ExternDecl;
pub type Func<P = Semantic, Lang = WynLanguage> = super::ir::Func<P, Lang>;
pub type Entry<
    P = Semantic,
    ResourceDecl = SemanticResourceDecl,
    Route = RealizedOutputRoute,
    Lang = WynLanguage,
> = super::ir::Entry<P, ResourceDecl, Route, Lang>;
pub type Program<Tag, Shape, GlobalContext, Lang = WynLanguage> =
    super::ir::Program<Tag, Shape, GlobalContext, Lang>;

pub(crate) fn fresh_region_name(region_interner: &RegionInterner, base: &str) -> String {
    if region_interner.get(base).is_none() {
        return base.to_string();
    }
    for suffix in 1.. {
        let candidate = format!("{base}_{suffix}");
        if region_interner.get(&candidate).is_none() {
            return candidate;
        }
    }
    unreachable!()
}

impl<Tag, Shape, GlobalContext> super::ir::Program<Tag, Shape, GlobalContext, WynLanguage>
where
    Shape: super::ir::ProgramShape,
{
    pub fn contains_region(&self, region: RegionId) -> bool {
        self.functions.iter().any(|function| function.region == region)
    }

    pub fn region(&self, region: RegionId) -> Option<&super::ir::Func<Shape::Family, WynLanguage>> {
        self.functions.iter().find(|function| function.region == region)
    }

    pub fn iter_regions(
        &self,
    ) -> impl Iterator<Item = (RegionId, &super::ir::Func<Shape::Family, WynLanguage>)> {
        self.functions.iter().map(|function| (function.region, function))
    }
}

impl<Tag>
    super::ir::Program<
        Tag,
        super::ir::ProgramFamily<Semantic, SemanticResourceDecl, RealizedOutputRoute, CoreProgramData>,
        RewriteGlobal,
        WynLanguage,
    >
{
    pub fn region_name(&self, region: RegionId) -> &str {
        self.data.region_interner.resolve(region)
    }
}

impl<Tag> Index<SemanticEntryId>
    for super::ir::Program<
        Tag,
        super::ir::ProgramFamily<Semantic, SemanticResourceDecl, RealizedOutputRoute, CoreProgramData>,
        RewriteGlobal,
        WynLanguage,
    >
{
    type Output = SemanticEntry;

    fn index(&self, id: SemanticEntryId) -> &Self::Output {
        &self.entry_points[id.index()]
    }
}

#[cfg(test)]
#[path = "program_tests.rs"]
mod program_tests;

impl<P: Family, Route> Entry<P, SemanticResourceDecl, Route> {
    pub(super) fn visit_types_mut(&mut self, mut visit: impl FnMut(&mut Type<TypeName>)) {
        for input in &mut self.inputs {
            visit(&mut input.ty);
        }
        for output in &mut self.outputs {
            visit(&mut output.ty);
        }
        for (ty, _) in &mut self.params {
            visit(ty);
        }
        visit(&mut self.return_ty);
        for declaration in &mut self.resource_declarations {
            visit(&mut declaration.elem_ty);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SemanticOpId {
    source: u32,
    implementation: Option<u32>,
}

impl SemanticOpId {
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

pub(crate) type SemanticOpIdSource = crate::IdSource<SemanticOpId>;

/// Target-independent identity of a semantic storage resource. Identities are
/// issued only by the logical-resource arena and its conversion-time builder;
/// callers can observe an id's dense index but cannot manufacture one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResourceId(u32);

impl ResourceId {
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[cfg(test)]
    pub(crate) const fn for_test(index: u32) -> Self {
        Self(index)
    }
}

/// Opaque index into the fixed semantic-entry table. Textual entry names are
/// publication metadata and are deliberately not used to connect plans back
/// to their source entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SemanticEntryId(usize);

impl SemanticEntryId {
    pub(crate) const fn from_index(index: usize) -> Self {
        Self(index)
    }

    pub(crate) const fn index(self) -> usize {
        self.0
    }

    #[cfg(test)]
    pub(crate) const fn for_test(index: usize) -> Self {
        Self(index)
    }
}

/// Stable identity of a semantic requirement to materialize a shared value.
/// It is deliberately distinct from `SemanticEntryId`: a requirement is not
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
        let elem_bytes = crate::ssa::layout::storage_elem_stride(elem_ty)?;
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

pub type PhysicalResourceRef = crate::BindingRef;
pub type PhysicalEGraph = EGraph<Physical>;
pub type PhysicalSoac = super::types::Soac<Physical>;
pub type PhysicalSideEffect = super::types::SideEffect<Physical>;
pub type PhysicalSideEffectKind = super::types::SideEffectKind<Physical>;
pub type PhysicalSegSpace = super::types::SegSpace<PhysicalResourceRef>;
pub type PhysicalFilterWorkBuffers = super::soac::filter::WorkBuffers<PhysicalResourceRef>;
pub type PhysicalFilterOutput = super::soac::filter::Output<PhysicalResourceRef>;
pub type PhysicalPureOp = super::types::PureOp<PhysicalResourceRef>;

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
    pub binding: crate::BindingRef,
    /// Source-level name published for this binding, when one exists.
    pub name: Option<String>,
}

#[derive(Clone, Debug)]
pub enum ResourceOrigin {
    Host(HostResource),
    Compiler(CompilerResource),
}

impl ResourceOrigin {
    pub fn host(binding: crate::BindingRef) -> Self {
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

    pub fn host_binding(&self) -> Option<crate::BindingRef> {
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
    host: HashMap<crate::BindingRef, ResourceId>,
    compiler: HashMap<CompilerResourceKey, ResourceId>,
}

/// Conversion-time resource arena. Host resources may be referenced before
/// their declarations are encountered, so this builder reserves their stable
/// identities and requires every reservation to be defined before `finish`.
#[derive(Default)]
pub(crate) struct LogicalResourceArenaBuilder {
    by_binding: HashMap<crate::BindingRef, ResourceId>,
    compiler: HashMap<CompilerResourceKey, ResourceId>,
    resources: Vec<Option<LogicalResourceDraft>>,
}

struct LogicalResourceDraft {
    origin: ResourceOrigin,
    elem_ty: Type<TypeName>,
    size: LogicalSize,
}

impl LogicalResourceArenaBuilder {
    pub(crate) fn host_id(&mut self, binding: crate::BindingRef) -> ResourceId {
        if let Some(resource) = self.by_binding.get(&binding) {
            return *resource;
        }
        let resource = ResourceId(self.resources.len() as u32);
        self.by_binding.insert(binding, resource);
        self.resources.push(None);
        resource
    }

    pub(crate) fn declare_host(
        &mut self,
        binding: crate::BindingRef,
        elem_ty: Type<TypeName>,
        size: LogicalSize,
    ) -> ResourceId {
        let resource = self.host_id(binding);
        let slot = &mut self.resources[resource.index()];
        match slot {
            Some(existing) => {
                if matches!(existing.size, LogicalSize::Unspecified)
                    && !matches!(size, LogicalSize::Unspecified)
                {
                    existing.size = size;
                }
            }
            None => {
                *slot = Some(LogicalResourceDraft {
                    origin: ResourceOrigin::Host(HostResource { binding, name: None }),
                    elem_ty,
                    size,
                });
            }
        }
        resource
    }

    pub(crate) fn allocate_compiler(
        &mut self,
        compiler: CompilerResource,
        elem_ty: Type<TypeName>,
        size: LogicalSize,
    ) -> ResourceId {
        if let Some(resource) = compiler.key().and_then(|key| self.compiler.get(&key).copied()) {
            return resource;
        }
        let resource = ResourceId(self.resources.len() as u32);
        if let Some(key) = compiler.key() {
            self.compiler.insert(key, resource);
        }
        self.resources.push(Some(LogicalResourceDraft {
            origin: ResourceOrigin::Compiler(compiler),
            elem_ty,
            size,
        }));
        resource
    }

    pub(crate) fn finish(
        self,
    ) -> Result<(HashMap<crate::BindingRef, ResourceId>, LogicalResourceArena), ResourceId> {
        let Self {
            by_binding,
            compiler,
            resources,
        } = self;
        let resources = resources
            .into_iter()
            .enumerate()
            .map(|(index, resource)| {
                let id = ResourceId(index as u32);
                resource
                    .map(|resource| LogicalResource {
                        id,
                        origin: resource.origin,
                        elem_ty: resource.elem_ty,
                        size: resource.size,
                    })
                    .ok_or(id)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let host = by_binding.clone();
        Ok((
            by_binding,
            LogicalResourceArena {
                resources,
                host,
                compiler,
            },
        ))
    }
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

pub(crate) fn host_resource_map(resources: &[LogicalResource]) -> HashMap<crate::BindingRef, ResourceId> {
    resources.iter().filter_map(|resource| Some((resource.host_binding()?, resource.id))).collect()
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

/// Finish the TLC conversion boundary by installing its authoritative
/// resource arena and replacing descriptor-shaped identities inside the
/// just-built graphs and types. No later semantic pass is allowed to perform
/// this rewrite or to introduce a binding-backed semantic resource.
pub(crate) fn finalize_converted_resources(
    inner: &mut super::from_tlc::Converted,
    by_binding: &HashMap<crate::BindingRef, ResourceId>,
) {
    for entry in &mut inner.entry_points {
        normalize_converted_graph_types(&mut entry.graph, by_binding);
    }
    for function in &mut inner.functions {
        normalize_converted_graph_types(&mut function.graph, by_binding);
    }
    for constant in &mut inner.constants {
        normalize_converted_graph_types(&mut constant.graph, by_binding);
    }
    normalize_structural_resources(inner, by_binding);
    for entry in &mut inner.entry_points {
        for input in &mut entry.inputs {
            input.resource = input
                .storage_binding()
                .or_else(|| input.storage_image_binding().map(|(binding, ..)| binding))
                .and_then(|binding| by_binding.get(&binding).copied())
                .map(SemanticResourceRef)
                .or_else(|| semantic_type_resource(&input.ty));
        }
        for output in &mut entry.outputs {
            output.resource = output
                .storage_binding()
                .and_then(|binding| by_binding.get(&binding).copied())
                .map(SemanticResourceRef)
                .or_else(|| semantic_type_resource(&output.ty));
        }
    }
}

fn semantic_type_resource(ty: &Type<TypeName>) -> Option<SemanticResourceRef> {
    let Type::Constructed(TypeName::Resource(resource), _) = ty.array_buffer()? else {
        return None;
    };
    Some(SemanticResourceRef(*resource))
}

fn normalize_converted_graph_types(
    graph: &mut EGraph<Raw>,
    by_binding: &HashMap<crate::BindingRef, ResourceId>,
) {
    rewrite_raw_graph_types(graph, |ty| normalize_type_resources(ty, by_binding));
}

fn normalize_structural_resources(
    inner: &mut super::from_tlc::Converted,
    by_binding: &HashMap<crate::BindingRef, ResourceId>,
) {
    for resource in &mut inner.data.resources {
        normalize_type_resources(&mut resource.elem_ty, by_binding);
    }
    for entry in &mut inner.entry_points {
        entry.visit_types_mut(|ty| normalize_type_resources(ty, by_binding));
    }
    for function in &mut inner.functions {
        for (ty, _) in &mut function.params {
            normalize_type_resources(ty, by_binding);
        }
        normalize_type_resources(&mut function.return_ty, by_binding);
    }
}

fn normalize_type_resources(ty: &mut Type<TypeName>, by_binding: &HashMap<crate::BindingRef, ResourceId>) {
    visit_type_names_mut(ty, |name| {
        if let TypeName::Buffer(binding) = *name {
            *name = TypeName::Resource(
                *by_binding.get(&binding).expect("buffer type resource must be in manifest"),
            );
        }
    });
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

fn rewrite_raw_graph_types(graph: &mut EGraph<Raw>, mut rewrite: impl FnMut(&mut Type<TypeName>)) {
    for block in graph.skeleton.blocks.values_mut() {
        for effect in &mut block.side_effects {
            if let super::types::SideEffectKind::Soac(SoacEffect(_, soac)) = &mut effect.kind {
                soac.for_each_type_mut(&mut rewrite);
            }
        }
    }
    rewrite_node_types(graph, rewrite);
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

fn rewrite_node_types<P: Family>(graph: &mut EGraph<P>, mut rewrite: impl FnMut(&mut Type<TypeName>)) {
    for node in graph.nodes.keys().collect::<Vec<_>>() {
        let mut ty = graph.nodes[node].ty.clone();
        rewrite(&mut ty);
        graph.retype_node(node, ty);
    }
}

fn physicalize_soac(
    soac: Soac<Scheduled>,
    nodes: &LookupMap<NodeId, NodeId>,
    bindings: &PhysicalResourceTable,
) -> Result<Soac<Physical>, String> {
    fn binding(reference: SemanticResourceRef, bindings: &PhysicalResourceTable) -> PhysicalResourceRef {
        bindings.binding(reference.0)
    }

    fn seg_body(mut body: SegBody, nodes: &LookupMap<NodeId, NodeId>) -> SegBody {
        for capture in &mut body.captures {
            *capture = nodes[capture];
        }
        body
    }

    fn space(
        space: SegSpace,
        nodes: &LookupMap<NodeId, NodeId>,
        bindings: &PhysicalResourceTable,
    ) -> Result<PhysicalSegSpace, String> {
        let dims = space
            .into_dims()
            .into_iter()
            .map(|extent| {
                Ok(match extent {
                    SegExtent::Fixed(value) => SegExtent::Fixed(value),
                    SegExtent::PushConstant { node, offset } => SegExtent::PushConstant {
                        node: nodes[&node],
                        offset,
                    },
                    SegExtent::ResourceLength {
                        node,
                        resource,
                        elem_bytes,
                    } => SegExtent::ResourceLength {
                        node: nodes[&node],
                        resource: binding(resource, bindings),
                        elem_bytes,
                    },
                    SegExtent::Value(node) => SegExtent::Value(nodes[&node]),
                })
            })
            .collect::<Result<_, String>>()?;
        SegSpace::from_dims(dims).ok_or_else(|| "physicalized segmented space was empty".to_string())
    }

    fn operator(mut operator: screma::Operator, nodes: &LookupMap<NodeId, NodeId>) -> screma::Operator {
        operator.step = seg_body(operator.step, nodes);
        operator.combine = seg_body(operator.combine, nodes);
        operator.neutral = nodes[&operator.neutral];
        for node in &mut operator.shape {
            *node = nodes[node];
        }
        operator
    }

    fn operators(
        operators: screma::NonEmpty<screma::Operator>,
        nodes: &LookupMap<NodeId, NodeId>,
    ) -> screma::NonEmpty<screma::Operator> {
        screma::NonEmpty {
            first: operator(operators.first, nodes),
            rest: operators.rest.into_iter().map(|value| operator(value, nodes)).collect(),
        }
    }

    fn screma_lanes(mut lanes: screma::Lanes, nodes: &LookupMap<NodeId, NodeId>) -> screma::Lanes {
        for map in &mut lanes.maps {
            map.body = seg_body(map.body.clone(), nodes);
        }
        lanes
    }

    fn composite_operators(
        values: screma::NonEmpty<screma::CompositeOperator>,
        nodes: &LookupMap<NodeId, NodeId>,
    ) -> screma::NonEmpty<screma::CompositeOperator> {
        let map = |value| match value {
            screma::CompositeOperator::Reduce(value) => {
                screma::CompositeOperator::Reduce(operator(value, nodes))
            }
            screma::CompositeOperator::Scan(value) => {
                screma::CompositeOperator::Scan(operator(value, nodes))
            }
        };
        screma::NonEmpty {
            first: map(values.first),
            rest: values.rest.into_iter().map(map).collect(),
        }
    }

    fn physical_segment(
        segment: screma::Segmented<SemanticResourceRef>,
        nodes: &LookupMap<NodeId, NodeId>,
        bindings: &PhysicalResourceTable,
    ) -> Result<screma::Segmented<PhysicalResourceRef>, String> {
        Ok(screma::Segmented {
            space: space(segment.space, nodes, bindings)?,
            output_slots: segment.output_slots,
            resources: segment
                .resources
                .into_iter()
                .map(|resource| {
                    Ok(super::types::SegResourceAccess {
                        resource: binding(resource.resource, bindings),
                        access: resource.access,
                    })
                })
                .collect::<Result<_, String>>()?,
        })
    }

    fn filter_output(
        output: filter::Output,
        bindings: &PhysicalResourceTable,
    ) -> Result<PhysicalFilterOutput, String> {
        Ok(match output {
            filter::Output::Local {
                capacity,
                destination,
            } => filter::Output::Local {
                capacity,
                destination,
            },
            filter::Output::Runtime { scratch, length } => filter::Output::Runtime {
                scratch: binding(scratch, bindings),
                length: match length {
                    filter::RuntimeLength::ViewOnly => filter::RuntimeLength::ViewOnly,
                    filter::RuntimeLength::Stored(resource) => {
                        filter::RuntimeLength::Stored(binding(resource, bindings))
                    }
                },
            },
        })
    }

    fn work_buffers(
        buffers: filter::WorkBuffers,
        bindings: &PhysicalResourceTable,
    ) -> Result<PhysicalFilterWorkBuffers, String> {
        Ok(filter::WorkBuffers {
            flags: binding(buffers.flags, bindings),
            offsets: binding(buffers.offsets, bindings),
            block_sums: binding(buffers.block_sums, bindings),
            block_offsets: binding(buffers.block_offsets, bindings),
        })
    }

    Ok(match soac {
        Soac::Screma(screma::Op::Map { lanes, state }) => {
            let state = match state {
                screma::ScheduledState::Serial => screma::ScheduledState::Serial,
                screma::ScheduledState::Segmented(segment) => {
                    screma::ScheduledState::Segmented(physical_segment(segment, nodes, bindings)?)
                }
            };
            Soac::Screma(screma::Op::Map {
                lanes: screma_lanes(lanes, nodes),
                state,
            })
        }
        Soac::Screma(screma::Op::Reduce {
            lanes,
            operators: values,
            state,
        }) => {
            if matches!(state, screma::ScheduledState::Segmented(_)) {
                return Err(
                    "scheduled SegRed reached physicalization; split it into physical kernels first".into(),
                );
            }
            Soac::Screma(screma::Op::Reduce {
                lanes: screma_lanes(lanes, nodes),
                operators: operators(values, nodes),
                state: screma::PhysicalSerialState,
            })
        }
        Soac::Screma(screma::Op::Scan {
            lanes,
            operators: values,
            state,
        }) => {
            if matches!(state, screma::ScheduledState::Segmented(_)) {
                return Err(
                    "scheduled SegScan reached physicalization; split it into physical kernels first"
                        .into(),
                );
            }
            Soac::Screma(screma::Op::Scan {
                lanes: screma_lanes(lanes, nodes),
                operators: operators(values, nodes),
                state: screma::PhysicalSerialState,
            })
        }
        Soac::Screma(screma::Op::Composite {
            lanes,
            operators: values,
            state,
        }) => {
            if matches!(state, screma::ScheduledState::Segmented(_)) {
                return Err(
                    "scheduled SegComposite reached physicalization; split it into physical kernels first"
                        .into(),
                );
            }
            Soac::Screma(screma::Op::Composite {
                lanes: screma_lanes(lanes, nodes),
                operators: composite_operators(values, nodes),
                state: screma::PhysicalSerialState,
            })
        }
        Soac::Filter(filter::Op { mut body, state }) => {
            if let filter::Input::Mapped { body, .. } = &mut body.input {
                *body = seg_body(body.clone(), nodes);
            }
            body.predicate = seg_body(body.predicate, nodes);
            let state = match state {
                filter::ScheduledState::Loop {
                    space: iteration_space,
                    storage,
                } => filter::ScheduledState::Loop {
                    space: space(iteration_space, nodes, bindings)?,
                    storage: filter_output(storage, bindings)?,
                },
                filter::ScheduledState::Pipeline {
                    space: iteration_space,
                    storage,
                    plan,
                } => filter::ScheduledState::Pipeline {
                    space: space(iteration_space, nodes, bindings)?,
                    storage: filter::RuntimeStorage {
                        scratch: binding(storage.scratch, bindings),
                        length: match storage.length {
                            filter::RuntimeLength::ViewOnly => filter::RuntimeLength::ViewOnly,
                            filter::RuntimeLength::Stored(resource) => {
                                filter::RuntimeLength::Stored(binding(resource, bindings))
                            }
                        },
                    },
                    plan: filter::ParallelPlan {
                        stage: plan.stage,
                        buffers: work_buffers(plan.buffers, bindings)?,
                        scan_workgroup_width: plan.scan_workgroup_width,
                    },
                },
            };
            Soac::Filter(filter::Op { body, state })
        }
        Soac::Hist(hist::Op { mut body, state }) => {
            body.body = seg_body(body.body, nodes);
            let state = match state {
                hist::State::Serial => hist::State::Serial,
                hist::State::Segmented(iteration_space) => {
                    hist::State::Segmented(space(iteration_space, nodes, bindings)?)
                }
            };
            Soac::Hist(hist::Op { body, state })
        }
    })
}

pub(crate) fn physicalize_graph_resources(
    graph: EGraph<Scheduled>,
    bindings: &PhysicalResourceTable,
) -> Result<
    (
        EGraph<Physical>,
        LookupMap<NodeId, NodeId>,
        LookupMap<BlockId, BlockId>,
    ),
    String,
> {
    let (mut graph, node_map, block_map) = graph.try_map_resources_and_phase(
        |reference| {
            let resource = reference.0;
            Ok::<_, String>(bindings.binding(resource))
        },
        |id, soac, nodes| physicalize_soac(soac, nodes, bindings).map(|soac| (id, soac)),
    )?;
    let pure_nodes = graph.nodes.keys().collect::<Vec<_>>();
    for node in pure_nodes {
        let resource_len = match graph.nodes.get(node).map(|node| &node.kind) {
            Some(super::types::ENode::Pure {
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
                    id: crate::builtins::catalog().known().storage_len,
                    overload_idx: 0,
                },
                smallvec::smallvec![set, slot],
            );
            continue;
        }
    }
    rewrite_physical_graph_types(&mut graph, |ty| physicalize_type_resources(ty, bindings));
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
) -> Option<crate::pipeline_descriptor::BufferLen> {
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

pub type RawFunc = Func<Raw>;
pub type SemanticFunc = Func<Semantic>;
pub type ScheduledFunc = Func<Scheduled>;
pub type PhysicalFunc = Func<Physical>;

pub type RawEntry<Route = super::ir::UnrealizedOutputRoute> = Entry<Raw, SemanticResourceDecl, Route>;
pub type SemanticEntry = Entry<Semantic>;
pub type ScheduledEntry = Entry<Scheduled>;

#[cfg(test)]
pub(crate) fn semantic_program_for_test(
    mut functions: Vec<SemanticFunc>,
    externs: Vec<ExternDecl<Type<TypeName>>>,
    entry_points: Vec<SemanticEntry>,
    constants: Vec<ConstantDef<Semantic>>,
    pipeline: PipelineDescriptor,
    mut region_interner: RegionInterner,
) -> super::reify::Segmented {
    for function in &mut functions {
        function.region = region_interner.intern(&function.name);
    }
    Program::from_parts(
        functions,
        externs,
        entry_points,
        constants,
        CoreProgramData {
            pipeline,
            resources: LogicalResourceArena::default(),
            region_interner,
        },
        RewriteGlobal {
            binding_ids: crate::IdSource::new(),
            effect_ids: crate::IdSource::new(),
            semantic_ids: crate::IdSource::new(),
        },
    )
}

impl SemanticEntry {
    /// Resource identities referenced by a set of values in `graph`, including
    /// resource-backed entry parameters whose identity is carried by the
    /// interface rather than by a storage-view node.
    pub(crate) fn resources_referenced_by_nodes(
        &self,
        graph: &EGraph,
        nodes: impl IntoIterator<Item = NodeId>,
    ) -> HashSet<ResourceId> {
        let mut resources = HashSet::new();
        for node in nodes {
            if let Some(ENode::Pure { op, .. }) = graph.nodes.get(node).map(|node| &node.kind) {
                if let Some(resource) = op.referenced_resource() {
                    resources.insert(resource.0);
                }
            }
            if let Some(ENode::FuncParam { index }) = graph.nodes.get(node).map(|node| &node.kind) {
                resources.extend(
                    self.inputs
                        .get(*index)
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
        projection: &super::graph_projector::GraphProjection,
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
        projection: &super::graph_projector::GraphProjection,
        resources: &HashSet<ResourceId>,
    ) -> crate::SortedSet<usize> {
        let mut parameters = projection
            .source_nodes()
            .filter_map(|node| match self.graph.nodes.get(node).map(|node| &node.kind) {
                Some(ENode::FuncParam { index }) => Some(*index),
                _ => None,
            })
            .collect::<crate::SortedSet<_>>();
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

    pub(crate) fn set_resource_declaration(
        &mut self,
        resource: ResourceId,
        role: interface::StorageRole,
        elem_ty: &Type<TypeName>,
        size: &LogicalSize,
    ) {
        if let Some(declaration) =
            self.resource_declarations.iter_mut().find(|declaration| declaration.resource.0 == resource)
        {
            declaration.role = role;
            declaration.elem_ty = elem_ty.clone();
            declaration.size = size.clone();
        } else {
            self.resource_declarations.push(SemanticResourceDecl {
                resource: SemanticResourceRef(resource),
                role,
                elem_ty: elem_ty.clone(),
                size: size.clone(),
            });
        }
    }

    pub(crate) fn declare_resource_view(
        &mut self,
        resource: ResourceId,
        role: interface::StorageRole,
        elem_ty: &Type<TypeName>,
        size: &LogicalSize,
    ) -> NodeId {
        let view = super::graph_ops::intern_resource_view(&mut self.graph, resource, elem_ty.clone(), None);
        self.set_resource_declaration(resource, role, elem_ty, size);
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
                    .flat_map(|effect| effect.referenced_nodes())
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
                Some(ENode::FuncParam { index }) => Some(*index),
                _ => None,
            })
            .collect::<crate::SortedSet<_>>();
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
pub type PlannedEntry<P = Semantic> =
    super::ir::Entry<P, SemanticResourceDecl, RealizedOutputRoute, WynLanguage>;

/// Backend-visible entry metadata retained by the plan without retaining a
/// second copy of the semantic graph.
#[derive(Clone, Debug)]
pub struct PlannedPublication {
    pub name: String,
    pub execution_model: ExecutionModel,
    pub inputs: Vec<EntryInput>,
    pub outputs: Vec<EntryOutput>,
    pub resource_declarations: Vec<SemanticResourceDecl>,
}

impl PlannedPublication {
    pub fn from_semantic(entry: &SemanticEntry) -> Self {
        Self {
            name: entry.name.clone(),
            execution_model: entry.execution_model.clone(),
            inputs: entry.inputs.iter().map(|input| input.inner.clone()).collect(),
            outputs: entry.outputs.iter().map(|output| output.inner.clone()).collect(),
            resource_declarations: entry.resource_declarations.clone(),
        }
    }

    pub fn publication(&self, resources: &PhysicalResourceTable) -> Result<EntryPublication, String> {
        publish_entry(
            &self.name,
            &self.execution_model,
            &self.inputs,
            &self.outputs,
            &self.resource_declarations,
            resources,
        )
    }
}

impl SemanticEntry {
    pub fn project(entry: &SemanticEntry) -> Result<Self, String> {
        let projection = super::graph_projector::GraphProjector::new(&entry.graph)
            .all_with_values(entry.routes().map(|route| route.source.value).collect())?;
        Self::from_projection(
            projection,
            entry.name.clone(),
            entry.span,
            entry.execution_model.clone(),
            entry.inputs.clone(),
            entry.outputs.clone(),
            entry.resource_declarations.clone(),
            entry.params.clone(),
            entry.return_ty.clone(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_projection(
        projection: super::graph_projector::GraphProjection,
        name: String,
        span: Span,
        execution_model: ExecutionModel,
        inputs: Vec<super::ir::EntryInput<SemanticResourceRef, WynLanguage>>,
        outputs: Vec<super::ir::EntryOutput<SemanticResourceRef, RealizedOutputRoute, WynLanguage>>,
        resource_declarations: Vec<SemanticResourceDecl>,
        params: Vec<(Type<TypeName>, String)>,
        return_ty: Type<TypeName>,
    ) -> Result<Self, String> {
        let outputs = outputs
            .into_iter()
            .map(|mut output| {
                output.routes = projection.remap_output_routes(output.routes)?;
                Ok(output)
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            name,
            span,
            execution_model,
            inputs,
            outputs,
            resource_declarations,
            params,
            return_ty,
            graph: projection.graph,
        })
    }
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
            elem_ty: declaration.elem_ty.clone(),
            length: buffer_len(&declaration.size, resources),
        })
        .collect();
    Ok(EntryPublication {
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
        entry: SemanticEntry,
    },
    Gather {
        space: SegSpace<SemanticResourceRef>,
        entry: SemanticEntry,
    },
    RuntimeArray {
        space: SegSpace<SemanticResourceRef>,
        entry: SemanticEntry,
    },
    Scalar {
        entry: SemanticEntry,
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

    pub fn entry(&self) -> &SemanticEntry {
        match self {
            Self::SharedArray { entry, .. }
            | Self::Gather { entry, .. }
            | Self::RuntimeArray { entry, .. }
            | Self::Scalar { entry } => entry,
        }
    }

    pub fn entry_mut(&mut self) -> &mut SemanticEntry {
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
    bindings: Vec<crate::BindingRef>,
    compiler_owned: Vec<bool>,
}

impl PhysicalResourceTable {
    /// Assign backend bindings deterministically. Host resources retain their
    /// declared ABI identities; only compiler-owned resources draw automatic
    /// bindings from `ids`.
    pub fn allocate(resources: &LogicalResourceArena, ids: &mut crate::IdSource<u32>) -> Self {
        Self::allocate_avoiding(resources, ids, std::iter::empty())
    }

    /// Assign bindings while also reserving descriptor slots occupied by
    /// non-resource interfaces such as textures and samplers.
    pub fn allocate_avoiding(
        resources: &LogicalResourceArena,
        ids: &mut crate::IdSource<u32>,
        reserved: impl IntoIterator<Item = crate::BindingRef>,
    ) -> Self {
        let mut used = host_resource_map(resources).into_keys().collect::<std::collections::HashSet<_>>();
        used.extend(reserved);
        let mut bindings = Vec::with_capacity(resources.len());
        let mut compiler_owned = Vec::with_capacity(resources.len());
        for resource in resources {
            compiler_owned.push(matches!(resource.origin, ResourceOrigin::Compiler(_)));
            let binding = match &resource.origin {
                ResourceOrigin::Host(host) => host.binding,
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

    pub fn binding(&self, resource: ResourceId) -> crate::BindingRef {
        self.bindings[resource.index()]
    }

    pub fn is_compiler(&self, resource: ResourceId) -> bool {
        self.compiler_owned[resource.index()]
    }

    /// Descriptor-stable identity for one compiler-owned logical resource.
    /// Physical descriptor slots and entry-local names are access paths.
    pub fn logical_name(&self, resource: ResourceId) -> Option<String> {
        self.is_compiler(resource).then(|| format!("_w_resource_{}", resource.index()))
    }
}

/// Program-owned EGIR data shared by logical and physical checkpoints.
#[derive(Debug)]
pub struct CoreProgramData {
    pub pipeline: PipelineDescriptor,
    pub resources: LogicalResourceArena,
    pub region_interner: RegionInterner,
}

/// Program-owned data after materialization requirements have been planned.
#[derive(Debug)]
pub struct AllocatedProgramData {
    pub core: CoreProgramData,
    pub materializations: crate::IdArena<MaterializationId, MaterializationRequirement>,
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
    pub binding_ids: crate::IdSource<u32>,
    pub effect_ids: crate::IdSource<super::types::EffectToken>,
    pub semantic_ids: SemanticOpIdSource,
}

/// Non-tree state retained after target-specific planning.
#[derive(Debug)]
pub struct PlannedGlobal {
    pub kernel_plan: super::parallelize::KernelPlanSummary,
    pub profile: crate::LoweringProfile,
    pub effect_ids: crate::IdSource<super::types::EffectToken>,
    pub semantic_ids: SemanticOpIdSource,
}

fn physicalize_function(
    function: SemanticFunc,
    resources: &PhysicalResourceTable,
    serial: bool,
) -> Result<PhysicalFunc, String> {
    let SemanticFunc {
        region,
        name,
        span,
        linkage_name,
        mut params,
        mut return_ty,
        graph,
    } = function;
    let (graph, _) = super::parallelize::prepare::graph(graph, serial)?;
    let (graph, _, _) = physicalize_graph_resources(graph, resources)?;
    for (ty, _) in &mut params {
        physicalize_type_resources(ty, resources);
    }
    physicalize_type_resources(&mut return_ty, resources);
    Ok(PhysicalFunc {
        region,
        name,
        span,
        linkage_name,
        params,
        return_ty,
        graph,
    })
}

fn physicalize_constant(
    constant: ConstantDef<Semantic>,
    resources: &PhysicalResourceTable,
) -> Result<ConstantDef<Physical>, String> {
    let ConstantDef {
        name,
        span,
        mut return_ty,
        graph,
    } = constant;
    let (graph, _) = super::parallelize::prepare::graph(graph, false)?;
    let (graph, _, _) = physicalize_graph_resources(graph, resources)?;
    physicalize_type_resources(&mut return_ty, resources);
    Ok(ConstantDef {
        name,
        span,
        return_ty,
        graph,
    })
}

fn physicalize_entry(
    entry: PlannedEntry<Scheduled>,
    resources: &PhysicalResourceTable,
) -> Result<PhysicalEntry, String> {
    let super::ir::Entry {
        name,
        span,
        execution_model,
        inputs,
        outputs,
        resource_declarations: mut declarations,
        mut params,
        mut return_ty,
        graph,
    } = entry;
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
    for (ty, _) in &mut params {
        physicalize_type_resources(ty, resources);
    }
    physicalize_type_resources(&mut return_ty, resources);
    for declaration in &mut declarations {
        physicalize_type_resources(&mut declaration.elem_ty, resources);
    }
    let resource_declarations = declarations
        .into_iter()
        .map(|declaration| interface::StorageBindingDecl {
            binding: resources.binding(declaration.resource.0),
            role: declaration.role,
            logical_resource: resources.logical_name(declaration.resource.0),
            elem_ty: declaration.elem_ty,
            length: buffer_len(&declaration.size, resources),
        })
        .collect();
    Ok(PhysicalEntry {
        name,
        span,
        execution_model,
        inputs,
        outputs,
        resource_declarations,
        params,
        return_ty,
        graph,
    })
}

pub(in crate::egir) fn physicalize_program(
    program: super::allocation::ResourcesAllocated,
    entries: impl IntoIterator<Item = PlannedEntry<Scheduled>>,
    physical_resources: &PhysicalResourceTable,
    serial: bool,
    kernel_plan: super::parallelize::KernelPlanSummary,
    profile: crate::LoweringProfile,
) -> Result<super::parallelize::Planned, String> {
    let Program {
        functions,
        externs,
        entry_points: _,
        constants,
        data,
        global_context,
        state: _,
    } = program;
    let entry_points = entries
        .into_iter()
        .map(|entry| physicalize_entry(entry, physical_resources))
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
        CoreProgramData {
            pipeline: data.core.pipeline,
            resources: data.core.resources,
            region_interner: data.core.region_interner,
        },
        PlannedGlobal {
            kernel_plan,
            profile,
            effect_ids: global_context.effect_ids,
            semantic_ids: global_context.semantic_ids,
        },
    ))
}
