//! Whole-program EGIR container + per-body records.
//!
//! Compiler state is explicit at both boundaries: program wrappers carry the
//! metadata available at each pipeline checkpoint, while each graph is
//! parameterized by its phase-specific resource identity. Physicalization
//! rebuilds those graphs as `EGraph<Physical>` inside a distinct
//! `PhysicalProgram`.
//!
//! The underlying [`super::ir::Program`] carries only low-level IR. Logical
//! resources, semantic dependencies, and allocation requirements live in the
//! compiler-facing wrappers in this module.

use crate::LookupMap;

use polytype::Type;

use crate::ast::{Span, TypeName};
use crate::flow::{BlockId, ControlHeader, ExecutionModel};
use crate::interface::{self, EntryInput, EntryOutput};
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::types::TypeExt;
use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut, Index, IndexMut};

use super::parallelize::KernelPlan;
use super::soac::{filter, hist, screma};
use super::types::{
    EGraph, ENode, EgirPhase, NodeId, Physical, Raw, Scheduled, SegBody, SegExtent, SegSpace, Semantic,
    SideEffectKind, SideEffectSite, Soac, SoacEffect, WynLanguage,
};

pub use super::ir::{OutputRoute, OutputSlotId, OutputWriter, RegionInterner, SlotSource};
pub type ConstantDef<P = Semantic, Lang = WynLanguage> = super::ir::ConstantDef<P, Lang>;
pub use crate::types::ExternDecl;
pub type Func<P = Semantic, Lang = WynLanguage> = super::ir::Func<P, Lang>;
pub type Entry<P = Semantic, Lang = WynLanguage> = super::ir::Entry<P, Lang>;
pub type Program<P = Semantic, Lang = WynLanguage> = super::ir::Program<P, Lang>;

#[cfg(test)]
#[path = "program_tests.rs"]
mod program_tests;

impl<P: EgirPhase<ResourceDecl = SemanticResourceDecl>> Entry<P> {
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
    /// materialization. The compiler-reserved `_w_` namespace keeps generated
    /// entries distinct from source declarations, while the arena identity
    /// makes names unique without searching every existing name.
    pub(crate) fn entry_name(self, source: &str, role: &str) -> String {
        format!("_w_materialization_{}_{source}_{role}", self.0)
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
    /// Explicit producer/consumer relationship established at allocation.
    /// Target planning consumes this edge directly instead of rediscovering
    /// physical requirements from explicit semantic materialization records.
    pub flow: Option<CompilerResourceFlow>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompilerResourceFlow {
    pub producer: CompilerFlowEndpoint,
    pub consumers: Vec<CompilerFlowEndpoint>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CompilerFlowEndpoint {
    Entry(SemanticEntryId),
    Materialization(MaterializationId),
}

impl CompilerResource {
    pub fn new(kind: CompilerResourceKind, owner: Option<SemanticOpId>, slot: usize) -> Self {
        Self {
            kind,
            owner,
            slot,
            flow: None,
        }
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
pub enum ResourceOrigin {
    Host(crate::BindingRef),
    Compiler(CompilerResource),
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
        match self.origin {
            ResourceOrigin::Host(binding) => Some(binding),
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
                    origin: ResourceOrigin::Host(binding),
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
            ResourceOrigin::Host(binding) => self.host.get(binding).copied(),
            ResourceOrigin::Compiler(compiler) => {
                compiler.key().and_then(|key| self.compiler.get(&key).copied())
            }
        };
        if let Some(id) = existing {
            return id;
        }
        let id = ResourceId(self.resources.len() as u32);
        match &origin {
            ResourceOrigin::Host(binding) => {
                self.host.insert(*binding, id);
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
        if let ResourceOrigin::Host(binding) = resource.origin {
            self.host.remove(&binding);
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

/// Establish target-independent logical residency and materialization
/// resources. Algorithm-specific reduce/scan/filter work buffers are selected
/// later by target planning, before physical bindings are allocated.
pub fn plan_logical_resources(
    inner: SemanticProgram,
    effect_ids: &mut crate::IdSource<super::types::EffectToken>,
) -> Result<AllocatedProgram, String> {
    let mut allocated = AllocatedProgram {
        semantic: inner,
        materializations: crate::IdArena::new(),
    };
    classify_existing_compiler_resources(&mut allocated);
    super::in_place::resolve_destinations(&mut allocated);
    super::residency::run(&mut allocated, effect_ids)?;
    super::soac::filter::resolve_scratch_sizes(&mut allocated);
    strip_compiler_abi(&mut allocated);
    record_compiler_resource_flows(&mut allocated);
    if cfg!(debug_assertions) {
        verify_allocated_resources(&allocated).expect("invalid allocated semantic resources");
    }
    Ok(allocated)
}

pub(crate) fn verify_allocated_resources(inner: &AllocatedProgram) -> Result<(), String> {
    let check_size = |size: &LogicalSize| match size {
        LogicalSize::LikeResource { resource, .. } if !inner.resources.contains(*resource) => {
            Err(format!("resource size references missing source {resource:?}"))
        }
        _ => Ok(()),
    };
    for resource in &inner.resources {
        check_size(&resource.size)?;
    }
    for declaration in inner.entries_with_endpoints().flat_map(|(_, entry)| &entry.resource_declarations) {
        if !inner.resources.contains(declaration.resource.0) {
            return Err(format!(
                "entry references missing resource {:?}",
                declaration.resource.0
            ));
        }
        check_size(&declaration.size)?;
    }
    Ok(())
}

fn classify_existing_compiler_resources(inner: &mut AllocatedProgram) {
    let mut classifications = HashMap::new();
    for entry in &inner.entry_points {
        for declaration in &entry.resource_declarations {
            if declaration.role == interface::StorageRole::Intermediate {
                let resource = declaration.resource.0;
                classifications
                    .entry(resource)
                    .or_insert_with(|| CompilerResource::new(CompilerResourceKind::Staging, None, 0));
            }
        }
    }
    let source_outputs = inner
        .entry_points
        .iter()
        .flat_map(|entry| {
            entry.outputs.iter().filter_map(|output| output.resource.map(|resource| resource.0))
        })
        .collect::<std::collections::HashSet<_>>();
    classifications.extend(
        super::soac::filter::resource_kinds(inner)
            .into_iter()
            .filter(|(resource, _)| !source_outputs.contains(resource)),
    );
    for (resource, compiler) in classifications {
        inner.resources.reclassify_as_compiler(resource, compiler);
    }
}

fn record_compiler_resource_flows(inner: &mut AllocatedProgram) {
    let mut producers: HashMap<ResourceId, Vec<CompilerFlowEndpoint>> = HashMap::new();
    let mut consumers: HashMap<ResourceId, Vec<CompilerFlowEndpoint>> = HashMap::new();
    for (endpoint, entry) in inner.entries_with_endpoints() {
        for declaration in &entry.resource_declarations {
            let resource = declaration.resource.0;
            match declaration.role {
                interface::StorageRole::Output => producers.entry(resource).or_default().push(endpoint),
                interface::StorageRole::Input => consumers.entry(resource).or_default().push(endpoint),
                interface::StorageRole::Intermediate => {}
            }
        }
    }
    for resource in &mut inner.resources {
        let ResourceOrigin::Compiler(compiler) = &mut resource.origin else {
            continue;
        };
        if !matches!(
            compiler.kind,
            CompilerResourceKind::GatherHandoff
                | CompilerResourceKind::ScalarHandoff
                | CompilerResourceKind::MultiConsumerArray
                | CompilerResourceKind::FilterScratch
                | CompilerResourceKind::FilterLenCell
        ) {
            continue;
        }
        let mut resource_producers = producers.remove(&resource.id).unwrap_or_default();
        resource_producers.sort_unstable();
        resource_producers.dedup();
        let [producer] = resource_producers.as_slice() else {
            continue;
        };
        let mut resource_consumers = consumers.remove(&resource.id).unwrap_or_default();
        resource_consumers.retain(|consumer| consumer != producer);
        resource_consumers.sort_unstable();
        resource_consumers.dedup();
        compiler.flow = Some(CompilerResourceFlow {
            producer: *producer,
            consumers: resource_consumers,
        });
    }
}

/// Finish the TLC conversion boundary by installing its authoritative
/// resource arena and replacing descriptor-shaped identities inside the
/// just-built graphs and types. No later semantic pass is allowed to perform
/// this rewrite or to introduce a binding-backed semantic resource.
pub(crate) fn finalize_converted_resources(
    inner: &mut RawProgram,
    resources: LogicalResourceArena,
    by_binding: &HashMap<crate::BindingRef, ResourceId>,
) {
    inner.resources = resources;
    for entry in &mut inner.entry_points {
        normalize_converted_graph_types(&mut entry.graph, by_binding);
    }
    for function in &mut inner.functions {
        normalize_converted_graph_types(&mut function.graph, by_binding);
    }
    normalize_structural_resources(inner, by_binding);
    for entry in &mut inner.entry_points {
        for input in &mut entry.inputs {
            input.resource = input
                .storage_binding()
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

fn strip_compiler_abi(inner: &mut AllocatedProgram) {
    let compiler_resources = inner
        .resources
        .iter()
        .filter_map(|resource| {
            matches!(resource.origin, ResourceOrigin::Compiler(_)).then_some(resource.id)
        })
        .collect::<std::collections::HashSet<_>>();
    let strip = |inputs: &mut Vec<super::ir::EntryInput<SemanticResourceRef, WynLanguage>>,
                 outputs: &mut Vec<super::ir::EntryOutput<SemanticResourceRef, WynLanguage>>,
                 routes: &mut Vec<OutputRoute>| {
        for input in inputs.iter_mut() {
            if input.resource.is_some_and(|resource| compiler_resources.contains(&resource.0)) {
                input.make_storage_internal();
            }
        }
        let mut output_slots = vec![None; outputs.len()];
        let mut host_outputs = Vec::with_capacity(outputs.len());
        for (slot, output) in std::mem::take(outputs).into_iter().enumerate() {
            let compiler_output =
                output.resource.is_some_and(|resource| compiler_resources.contains(&resource.0));
            if !compiler_output {
                output_slots[slot] = Some(host_outputs.len());
                host_outputs.push(output);
            }
        }
        *outputs = host_outputs;
        routes.retain_mut(|route| {
            let Some(slot) = output_slots.get(route.slot.0).copied().flatten() else {
                return false;
            };
            route.slot = OutputSlotId(slot);
            true
        });
    };
    for entry in &mut inner.entry_points {
        strip(&mut entry.inputs, &mut entry.outputs, &mut entry.output_routes);
    }
    for (_, requirement) in &mut inner.materializations {
        let entry = requirement.entry_mut();
        strip(&mut entry.inputs, &mut entry.outputs, &mut entry.output_routes);
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
    inner: &mut RawProgram,
    by_binding: &HashMap<crate::BindingRef, ResourceId>,
) {
    for resource in &mut inner.resources {
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

fn rewrite_node_types<P: EgirPhase>(graph: &mut EGraph<P>, mut rewrite: impl FnMut(&mut Type<TypeName>)) {
    for node in graph.types.keys().copied().collect::<Vec<_>>() {
        let mut ty = graph.types[&node].clone();
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
        let resource_len = match graph.nodes.get(node) {
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

pub type RawEntry = Entry<Raw>;
pub type SemanticEntry = Entry<Semantic>;
pub type ScheduledEntry = Entry<Scheduled>;

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
            if let Some(resource) = super::graph_ops::extract_storage_view_source(graph, node) {
                resources.insert(resource.0);
            }
            if let Some(ENode::FuncParam { index }) = graph.nodes.get(node) {
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
            .filter_map(|node| match self.graph.nodes.get(node) {
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
        for route in &self.output_routes {
            roots.push(route.source.value);
            roots.extend(route.writers.iter().filter_map(|writer| match writer {
                OutputWriter::Value(value) => Some(*value),
                OutputWriter::Effect(_) => None,
            }));
        }
        let reachable = super::graph_ops::execution_value_producer_closure(&self.graph, roots).nodes;
        let reachable_resources =
            self.resources_referenced_by_nodes(&self.graph, reachable.iter().copied());
        let mut kept_indices = reachable
            .iter()
            .filter_map(|node| match self.graph.nodes.get(*node) {
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
pub type PlannedEntry<P = Semantic> = super::ir::Entry<P, WynLanguage>;

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

impl super::ir::Entry<Semantic, WynLanguage> {
    pub fn project(entry: &SemanticEntry) -> Result<Self, String> {
        let projection = super::graph_projector::GraphProjector::new(&entry.graph, &entry.control_headers)
            .all_with_values(entry.output_routes.iter().map(|route| route.source.value).collect())?;
        Self::from_projection(
            projection,
            entry.name.clone(),
            entry.span,
            entry.execution_model.clone(),
            entry.inputs.iter().map(|input| input.inner.clone()).collect(),
            entry.outputs.iter().map(|output| output.inner.clone()).collect(),
            entry.resource_declarations.clone(),
            entry.params.clone(),
            entry.return_ty.clone(),
            &entry.aliases,
            entry.output_routes.clone(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_projection(
        projection: super::graph_projector::GraphProjection,
        name: String,
        span: Span,
        execution_model: ExecutionModel,
        inputs: Vec<EntryInput>,
        outputs: Vec<EntryOutput>,
        resource_declarations: Vec<SemanticResourceDecl>,
        params: Vec<(Type<TypeName>, String)>,
        return_ty: Type<TypeName>,
        aliases: &LookupMap<NodeId, NodeId>,
        output_routes: Vec<OutputRoute>,
    ) -> Result<Self, String> {
        let aliases = projection.remap_aliases(aliases);
        let output_routes = projection.remap_output_routes(output_routes)?;
        Ok(Self {
            name,
            span,
            execution_model,
            inputs: inputs
                .into_iter()
                .map(|inner| super::ir::EntryInput {
                    inner,
                    resource: None,
                })
                .collect(),
            outputs: outputs
                .into_iter()
                .map(|inner| super::ir::EntryOutput {
                    inner,
                    resource: None,
                })
                .collect(),
            resource_declarations,
            params,
            return_ty,
            graph: projection.graph,
            control_headers: projection.control_headers,
            aliases,
            output_routes,
        })
    }
}

impl<P: EgirPhase<ResourceDecl = SemanticResourceDecl>> super::ir::Entry<P, WynLanguage> {
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
pub type PhysicalEntry = super::ir::Entry<Physical, WynLanguage>;

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

    pub fn binding(&self, resource: ResourceId) -> crate::BindingRef {
        self.bindings[resource.index()]
    }

    pub fn is_compiler(&self, resource: ResourceId) -> bool {
        self.compiler_owned[resource.index()]
    }
}

/// Low-level raw EGIR plus the logical resources established during TLC
/// conversion. Raw EGIR has neither semantic dependencies nor allocation
/// requirements.
pub struct RawProgram {
    pub ir: Program<Raw>,
    pub resources: LogicalResourceArena,
}

impl RawProgram {
    pub fn new(
        functions: Vec<RawFunc>,
        externs: Vec<ExternDecl<Type<TypeName>>>,
        entry_points: Vec<RawEntry>,
        constants: Vec<ConstantDef<Raw>>,
        pipeline: PipelineDescriptor,
        region_interner: RegionInterner,
    ) -> Self {
        Self {
            ir: Program::new(
                functions,
                externs,
                entry_points,
                constants,
                pipeline,
                region_interner,
            ),
            resources: LogicalResourceArena::default(),
        }
    }
}

impl Deref for RawProgram {
    type Target = Program<Raw>;

    fn deref(&self) -> &Self::Target {
        &self.ir
    }
}

impl DerefMut for RawProgram {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ir
    }
}

/// Semantic EGIR plus its logical-resource manifest and dependency DAG.
pub struct SemanticProgram {
    pub ir: Program<Semantic>,
    pub resources: LogicalResourceArena,
    pub semantic_dependencies: Vec<SemanticDependency>,
    /// Array-producing operations whose values cross an indexing or captured
    /// execution boundary. Rebuilt beside semantic dependencies so residency
    /// consumes construction-time graph facts instead of rescanning patterns.
    pub(crate) array_residency_demands: HashSet<SemanticOpId>,
}

impl SemanticProgram {
    pub fn new(
        functions: Vec<SemanticFunc>,
        externs: Vec<ExternDecl<Type<TypeName>>>,
        entry_points: Vec<SemanticEntry>,
        constants: Vec<ConstantDef<Semantic>>,
        pipeline: PipelineDescriptor,
        region_interner: RegionInterner,
    ) -> Self {
        Self {
            ir: Program::new(
                functions,
                externs,
                entry_points,
                constants,
                pipeline,
                region_interner,
            ),
            resources: LogicalResourceArena::default(),
            semantic_dependencies: Vec::new(),
            array_residency_demands: HashSet::new(),
        }
    }

    pub(crate) fn entry_ids(&self) -> impl Iterator<Item = SemanticEntryId> + '_ {
        (0..self.ir.entry_points.len()).map(SemanticEntryId)
    }

    /// Mutably select one segmented body inside an entry-point effect.
    pub(crate) fn entry_seg_body_mut(
        &mut self,
        entry: usize,
        effect: SideEffectSite,
        body: usize,
    ) -> Option<&mut SegBody> {
        self.ir.entry_points.get_mut(entry)?.graph.skeleton.get_effect_mut(effect)?.seg_body_mut(body)
    }
}

impl Index<SemanticEntryId> for SemanticProgram {
    type Output = SemanticEntry;

    fn index(&self, id: SemanticEntryId) -> &Self::Output {
        &self.ir.entry_points[id.index()]
    }
}

impl Deref for SemanticProgram {
    type Target = Program<Semantic>;

    fn deref(&self) -> &Self::Target {
        &self.ir
    }
}

impl DerefMut for SemanticProgram {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ir
    }
}

/// Semantic EGIR after logical resource planning has introduced every
/// materialization requirement.
pub struct AllocatedProgram {
    pub semantic: SemanticProgram,
    pub materializations: crate::IdArena<MaterializationId, MaterializationRequirement>,
}

impl Deref for AllocatedProgram {
    type Target = SemanticProgram;

    fn deref(&self) -> &Self::Target {
        &self.semantic
    }
}

impl DerefMut for AllocatedProgram {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.semantic
    }
}

pub type ScheduledProgram = Program<Scheduled>;

/// Fully constructed physical EGIR. Schedule validation and physical binding
/// verification are complete before this value is returned; transient
/// planning state is not retained.
pub struct PhysicalProgram {
    ir: Program<Physical>,
    resources: LogicalResourceArena,
}

impl Deref for PhysicalProgram {
    type Target = Program<Physical>;

    fn deref(&self) -> &Self::Target {
        &self.ir
    }
}

impl DerefMut for PhysicalProgram {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ir
    }
}

pub(crate) fn remap_control_headers(
    headers: &LookupMap<BlockId, ControlHeader>,
    map: impl Fn(BlockId) -> BlockId + Copy,
) -> LookupMap<BlockId, ControlHeader> {
    headers.iter().map(|(block, header)| (map(*block), header.remap(&map))).collect()
}

fn physicalize_function(
    function: SemanticFunc,
    resources: &PhysicalResourceTable,
    serial: bool,
) -> Result<PhysicalFunc, String> {
    let SemanticFunc {
        name,
        span,
        linkage_name,
        mut params,
        mut return_ty,
        graph,
        control_headers,
        aliases,
    } = function;
    let (graph, scheduled_blocks) = super::parallelize::prepare::graph(graph, serial)?;
    let control_headers = remap_control_headers(&control_headers, |block| scheduled_blocks[&block]);
    let (graph, node_map, block_map) = physicalize_graph_resources(graph, resources)?;
    for (ty, _) in &mut params {
        physicalize_type_resources(ty, resources);
    }
    physicalize_type_resources(&mut return_ty, resources);
    Ok(PhysicalFunc {
        name,
        span,
        linkage_name,
        params,
        return_ty,
        graph,
        control_headers: remap_control_headers(&control_headers, |block| block_map[&block]),
        aliases: aliases.into_iter().map(|(from, to)| (node_map[&from], node_map[&to])).collect(),
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
        control_headers,
        aliases,
    } = constant;
    let (graph, scheduled_blocks) = super::parallelize::prepare::graph(graph, false)?;
    let control_headers = remap_control_headers(&control_headers, |block| scheduled_blocks[&block]);
    let (graph, node_map, block_map) = physicalize_graph_resources(graph, resources)?;
    physicalize_type_resources(&mut return_ty, resources);
    Ok(ConstantDef {
        name,
        span,
        return_ty,
        graph,
        control_headers: remap_control_headers(&control_headers, |block| block_map[&block]),
        aliases: aliases.into_iter().map(|(from, to)| (node_map[&from], node_map[&to])).collect(),
    })
}

fn physicalize_entry(
    entry: &PlannedEntry<Scheduled>,
    resources: &PhysicalResourceTable,
) -> Result<PhysicalEntry, String> {
    let inputs = entry
        .inputs
        .iter()
        .cloned()
        .map(|mut input| {
            physicalize_type_resources(&mut input.ty, resources);
            let resource = input.resource.map(|resource| resources.binding(resource.0));
            super::ir::EntryInput {
                inner: input.inner,
                resource,
            }
        })
        .collect();
    let outputs = entry
        .outputs
        .iter()
        .cloned()
        .map(|mut output| {
            physicalize_type_resources(&mut output.ty, resources);
            let resource = output.resource.map(|resource| resources.binding(resource.0));
            super::ir::EntryOutput {
                inner: output.inner,
                resource,
            }
        })
        .collect();
    let mut declarations = entry.resource_declarations.clone();
    let mut params = entry.params.clone();
    let mut return_ty = entry.return_ty.clone();
    let (graph, nodes, blocks) = physicalize_graph_resources(entry.graph.clone(), resources)?;
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
            elem_ty: declaration.elem_ty,
            length: buffer_len(&declaration.size, resources),
        })
        .collect();
    let output_routes = super::graph_projector::remap_output_routes(
        entry.output_routes.clone(),
        |node| nodes.get(&node).copied(),
        |block| blocks.get(&block).copied(),
        Some,
        true,
        "physicalization",
    )?;
    Ok(PhysicalEntry {
        name: entry.name.clone(),
        span: entry.span,
        execution_model: entry.execution_model.clone(),
        inputs,
        outputs,
        resource_declarations,
        params,
        return_ty,
        graph,
        control_headers: remap_control_headers(&entry.control_headers, |block| blocks[&block]),
        aliases: entry.aliases.iter().map(|(from, to)| (nodes[from], nodes[to])).collect(),
        output_routes,
    })
}

impl PhysicalProgram {
    pub(in crate::egir) fn from_plan(
        program: AllocatedProgram,
        plan: &KernelPlan,
        physical_resources: &PhysicalResourceTable,
        serial: bool,
        pipeline: PipelineDescriptor,
    ) -> Result<Self, String> {
        let AllocatedProgram {
            semantic,
            materializations: _,
        } = program;
        let SemanticProgram {
            ir,
            resources,
            semantic_dependencies: _,
            array_residency_demands: _,
        } = semantic;
        let Program {
            functions,
            externs,
            entry_points: _,
            constants,
            pipeline: _,
            input_names,
            regions,
            region_interner,
        } = ir;
        let entry_points = plan
            .physical_entries()
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
        let mut regions = regions;
        for (index, function) in functions.iter().enumerate() {
            let id = region_interner
                .get(&function.name)
                .ok_or_else(|| format!("physical callable `{}` has no region identity", function.name))?;
            regions.insert(id, index);
        }
        Ok(Self {
            ir: Program {
                functions,
                externs,
                entry_points,
                constants,
                pipeline,
                input_names,
                regions,
                region_interner,
            },
            resources,
        })
    }

    pub(crate) fn logical_resources(&self) -> &[LogicalResource] {
        &self.resources
    }

    pub(in crate::egir) fn ir_mut(&mut self) -> &mut Program<Physical> {
        &mut self.ir
    }

    pub(in crate::egir) fn into_ir(self) -> Program<Physical> {
        self.ir
    }
}

impl AllocatedProgram {
    pub(crate) fn alloc_compiler_resource(
        &mut self,
        compiler: CompilerResource,
        elem_ty: Type<TypeName>,
        size: LogicalSize,
    ) -> ResourceId {
        self.resources.allocate(ResourceOrigin::Compiler(compiler), elem_ty, size)
    }

    pub(crate) fn entries_with_endpoints(
        &self,
    ) -> impl Iterator<Item = (CompilerFlowEndpoint, &SemanticEntry)> {
        self.semantic.entry_ids().map(|id| (CompilerFlowEndpoint::Entry(id), &self.semantic[id])).chain(
            self.materializations.ids().map(|id| {
                (
                    CompilerFlowEndpoint::Materialization(id),
                    self.materializations[id].entry(),
                )
            }),
        )
    }
}
