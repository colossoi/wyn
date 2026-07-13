//! Whole-program EGIR container + per-body records.
//!
//! Compiler state is explicit at both boundaries: public semantic pipeline
//! newtypes wrap `SemanticProgram`, while each graph is parameterized by its
//! phase-specific resource identity. Physicalization rebuilds those graphs as
//! `EGraph<BindingRef>` inside a distinct `PhysicalProgram`.
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

use super::parallelize::schedule::ValidatedKernelPlan;
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
pub struct SemanticRegion {
    pub name: String,
    pub params: Vec<(Type<TypeName>, String)>,
    pub return_ty: Type<TypeName>,
    pub graph: EGraph,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
}

impl SemanticRegion {
    pub fn from_function(function: &SemanticFunc) -> Self {
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

/// Stable identity of a semantic requirement to materialize a shared value.
/// It is deliberately distinct from `SemanticEntryId`: a requirement is not
/// an entry point and cannot be mutated by semantic entry passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MaterializationId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PrepassId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrepassKind {
    Scalar,
    Gather,
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

/// A semantic storage identity. It cannot represent a backend binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SemanticResourceRef(pub ResourceId);

pub type PhysicalResourceRef = crate::BindingRef;

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
    pub(crate) fn from_abi(
        declaration: interface::StorageBindingDecl,
        resource: ResourceId,
        size: LogicalSize,
    ) -> Self {
        Self {
            resource: SemanticResourceRef(resource),
            role: declaration.role,
            elem_ty: declaration.elem_ty,
            size,
        }
    }
}

/// Why a compiler-introduced resource exists. The kind fixes its physical
/// storage role and lets descriptor publication build the right
/// `StorageBindingDecl` without re-deriving it from the lowering site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

impl CompilerResourceKind {
    /// The physical storage role a resource of this kind lowers to.
    pub fn role(self) -> interface::StorageRole {
        match self {
            CompilerResourceKind::FilterScratch
            | CompilerResourceKind::GatherHandoff
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
    /// Explicit producer/consumer relationship established at allocation.
    /// Target planning consumes this edge directly instead of rediscovering
    /// prepasses from entry provenance or storage roles.
    pub flow: Option<CompilerResourceFlow>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompilerResourceFlow {
    pub producer: CompilerFlowEndpoint,
    pub consumers: Vec<CompilerFlowEndpoint>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CompilerFlowEndpoint {
    Entry(SemanticEntryId),
    Prepass(PrepassId),
    Materialization(MaterializationId),
}

impl CompilerResource {
    pub fn new(kind: CompilerResourceKind, owner: Option<SemanticOpId>, slot: usize) -> Self {
        Self {
            role: kind.role(),
            kind,
            owner,
            slot,
            flow: None,
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

/// Complete the authoritative logical-resource manifest at the allocation
/// boundary. Host resources and TLC-originated compiler resources already
/// have stable identities; this pass classifies compiler ownership, adds the
/// scratch required by segmented operations, extracts typed prepasses, and
/// records explicit producer/consumer flows.
pub fn plan_logical_resources(inner: &mut SemanticProgram) {
    classify_existing_compiler_resources(inner);
    super::multi_consumer::run(inner);
    allocate_filter_work_resources(inner);
    let mut scratch = super::parallelize::enumerate_seg_scratch(inner, inner.resources.len() as u32);
    inner.resources.append(&mut scratch);
    extract_prepass_requirements(inner);
    strip_compiler_abi(inner);
    record_compiler_resource_flows(inner);
    if cfg!(debug_assertions) {
        verify_allocated_resources(inner).expect("invalid allocated semantic resources");
    }
}

pub(crate) fn verify_allocated_resources(inner: &SemanticProgram) -> Result<(), String> {
    let ids = inner.resources.iter().map(|resource| resource.id).collect::<std::collections::HashSet<_>>();
    if ids.len() != inner.resources.len()
        || inner.resources.iter().enumerate().any(|(index, resource)| resource.id.0 as usize != index)
    {
        return Err("resource manifest is not dense and unique".into());
    }
    let check_size = |size: &LogicalSize| match size {
        LogicalSize::LikeResource { resource, .. } if !ids.contains(resource) => {
            Err(format!("resource size references missing source {resource:?}"))
        }
        _ => Ok(()),
    };
    for resource in &inner.resources {
        check_size(&resource.size)?;
    }
    let declarations = inner
        .entry_points
        .iter()
        .map(|entry| entry.resource_declarations.as_slice())
        .chain(inner.prepasses.iter().map(|prepass| prepass.entry.resource_declarations.as_slice()))
        .chain(
            inner
                .materializations
                .iter()
                .map(|requirement| requirement.entry.resource_declarations.as_slice()),
        );
    for declaration in declarations.flatten() {
        if !ids.contains(&declaration.resource.0) {
            return Err(format!(
                "entry references missing resource {:?}",
                declaration.resource.0
            ));
        }
        check_size(&declaration.size)?;
    }
    Ok(())
}

fn gather_prepass_resources(inner: &SemanticProgram) -> HashMap<ResourceId, CompilerResource> {
    let mut resources = HashMap::new();
    for entry in &inner.entry_points {
        if inner.prepass_roles.get(&entry.name) != Some(&PrepassKind::Gather) {
            continue;
        }
        for (slot, declaration) in entry.resource_declarations.iter().enumerate() {
            if declaration.role != interface::StorageRole::Output {
                continue;
            }
            resources.insert(
                declaration.resource.0,
                CompilerResource::new(CompilerResourceKind::GatherHandoff, None, slot),
            );
        }
    }
    resources
}

fn classify_existing_compiler_resources(inner: &mut SemanticProgram) {
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
    classifications.extend(gather_prepass_resources(inner));
    classifications.extend(scalar_handoff_resources(inner));
    let source_outputs = inner
        .entry_points
        .iter()
        .filter(|entry| !inner.prepass_roles.contains_key(&entry.name))
        .flat_map(|entry| entry.resource_abi.outputs.iter().flatten().copied())
        .collect::<std::collections::HashSet<_>>();
    classifications.extend(
        filter_resource_kinds(inner).into_iter().filter(|(resource, _)| !source_outputs.contains(resource)),
    );
    for (resource, compiler) in classifications {
        let logical = inner
            .resources
            .get_mut(resource.0 as usize)
            .filter(|logical| logical.id == resource)
            .expect("compiler classification references a missing resource");
        logical.origin = ResourceOrigin::Compiler(compiler);
    }
}

fn extract_prepass_requirements(inner: &mut SemanticProgram) {
    let mut source_entries = Vec::with_capacity(inner.entry_points.len());
    let mut prepasses = Vec::with_capacity(inner.prepass_roles.len());
    for entry in inner.entry_points.drain(..) {
        let Some(kind) = inner.prepass_roles.remove(&entry.name) else {
            source_entries.push(entry);
            continue;
        };
        prepasses.push(PrepassRequirement {
            id: PrepassId(prepasses.len() as u32),
            kind,
            entry,
        });
    }
    assert!(
        inner.prepass_roles.is_empty(),
        "prepass classification references a missing entry"
    );
    inner.entry_points = source_entries;
    inner.prepasses = prepasses;
}

fn record_compiler_resource_flows(inner: &mut SemanticProgram) {
    let mut producers: HashMap<ResourceId, Vec<CompilerFlowEndpoint>> = HashMap::new();
    let mut consumers: HashMap<ResourceId, Vec<CompilerFlowEndpoint>> = HashMap::new();
    for (index, entry) in inner.entry_points.iter().enumerate() {
        let entry_id = SemanticEntryId(index as u32);
        for declaration in &entry.resource_declarations {
            let resource = declaration.resource.0;
            match declaration.role {
                interface::StorageRole::Output => {
                    producers.entry(resource).or_default().push(CompilerFlowEndpoint::Entry(entry_id))
                }
                interface::StorageRole::Input => {
                    consumers.entry(resource).or_default().push(CompilerFlowEndpoint::Entry(entry_id))
                }
                interface::StorageRole::Intermediate => {}
            }
        }
    }
    for prepass in &inner.prepasses {
        let endpoint = CompilerFlowEndpoint::Prepass(prepass.id);
        for declaration in &prepass.entry.resource_declarations {
            let resource = declaration.resource.0;
            match declaration.role {
                interface::StorageRole::Output => producers.entry(resource).or_default().push(endpoint),
                interface::StorageRole::Input => consumers.entry(resource).or_default().push(endpoint),
                interface::StorageRole::Intermediate => {}
            }
        }
    }
    for requirement in &inner.materializations {
        let endpoint = CompilerFlowEndpoint::Materialization(requirement.id);
        for declaration in &requirement.entry.resource_declarations {
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
    let flows = inner
        .resources
        .iter()
        .filter_map(|resource| match &resource.origin {
            ResourceOrigin::Compiler(compiler) => {
                compiler.flow.as_ref().map(|flow| (resource.id, flow.consumers.clone()))
            }
            ResourceOrigin::Host(_) => None,
        })
        .collect::<HashMap<_, _>>();
    for requirement in &mut inner.materializations {
        for substitution in &mut requirement.substitutions {
            let resource = substitution.resource.0;
            substitution.consumers = flows.get(&resource).cloned().unwrap_or_default();
        }
    }
}

/// Finish the TLC conversion boundary by installing its authoritative
/// resource arena and replacing descriptor-shaped identities inside the
/// just-built graphs and types. No later semantic pass is allowed to perform
/// this rewrite or to introduce a binding-backed semantic resource.
pub(crate) fn finalize_converted_resources(
    inner: &mut SemanticProgram,
    resources: Vec<LogicalResource>,
    by_binding: &HashMap<crate::BindingRef, ResourceId>,
) {
    inner.resources = resources;
    for entry in &mut inner.entry_points {
        normalize_converted_graph_types(&mut entry.graph, by_binding);
    }
    for function in &mut inner.functions {
        normalize_converted_graph_types(&mut function.graph, by_binding);
    }
    for region in inner.regions.values_mut() {
        normalize_converted_graph_types(&mut region.graph, by_binding);
    }
    normalize_structural_resources(inner, by_binding);
    for entry in &mut inner.entry_points {
        entry.resource_abi = EntryResourceAbi {
            inputs: entry
                .inputs
                .iter()
                .map(|input| {
                    input
                        .storage_binding
                        .and_then(|binding| by_binding.get(&binding).copied())
                        .or_else(|| semantic_type_resource(&input.ty))
                })
                .collect(),
            outputs: entry
                .outputs
                .iter()
                .map(|output| {
                    output
                        .storage_binding
                        .and_then(|binding| by_binding.get(&binding).copied())
                        .or_else(|| semantic_type_resource(&output.ty))
                })
                .collect(),
        };
    }
}

fn strip_compiler_abi(inner: &mut SemanticProgram) {
    let compiler_resources = inner
        .resources
        .iter()
        .filter_map(|resource| {
            matches!(resource.origin, ResourceOrigin::Compiler(_)).then_some(resource.id)
        })
        .collect::<std::collections::HashSet<_>>();
    let strip = |inputs: &mut Vec<EntryInput>,
                 outputs: &mut Vec<EntryOutput>,
                 resource_abi: &mut EntryResourceAbi,
                 routes: &mut Vec<OutputRoute>| {
        for (input, resource) in inputs.iter_mut().zip(&resource_abi.inputs) {
            if resource.is_some_and(|resource| compiler_resources.contains(&resource)) {
                input.storage_binding = None;
            }
        }
        let mut output_slots = vec![None; outputs.len()];
        let mut host_outputs = Vec::with_capacity(outputs.len());
        let mut host_resources = Vec::with_capacity(resource_abi.outputs.len());
        for (slot, (output, resource)) in
            std::mem::take(outputs).into_iter().zip(std::mem::take(&mut resource_abi.outputs)).enumerate()
        {
            let compiler_output = resource.is_some_and(|resource| compiler_resources.contains(&resource));
            if !compiler_output {
                output_slots[slot] = Some(host_outputs.len());
                host_outputs.push(output);
                host_resources.push(resource);
            }
        }
        *outputs = host_outputs;
        resource_abi.outputs = host_resources;
        routes.retain_mut(|route| {
            let Some(slot) = output_slots.get(route.slot.0).copied().flatten() else {
                return false;
            };
            route.slot = OutputSlotId(slot);
            true
        });
    };
    for entry in &mut inner.entry_points {
        strip(
            &mut entry.inputs,
            &mut entry.outputs,
            &mut entry.resource_abi,
            &mut entry.output_routes,
        );
    }
    for prepass in &mut inner.prepasses {
        strip(
            &mut prepass.entry.inputs,
            &mut prepass.entry.outputs,
            &mut prepass.entry.resource_abi,
            &mut prepass.entry.output_routes,
        );
    }
}

fn semantic_type_resource(ty: &Type<TypeName>) -> Option<ResourceId> {
    let Type::Constructed(TypeName::Resource(resource), _) = ty.array_buffer()? else {
        return None;
    };
    Some(*resource)
}

fn normalize_converted_graph_types(
    graph: &mut EGraph,
    by_binding: &HashMap<crate::BindingRef, ResourceId>,
) {
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            if let super::types::SideEffectKind::Soac(soac) = &mut effect.kind {
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
    for requirement in &mut inner.materializations {
        for input in &mut requirement.entry.inputs {
            normalize_type_resources(&mut input.ty, by_binding);
        }
        for (ty, _) in &mut requirement.entry.params {
            normalize_type_resources(ty, by_binding);
        }
        normalize_type_resources(&mut requirement.entry.return_ty, by_binding);
        for declaration in &mut requirement.entry.resource_declarations {
            normalize_type_resources(&mut declaration.elem_ty, by_binding);
        }
    }
    for prepass in &mut inner.prepasses {
        for input in &mut prepass.entry.inputs {
            normalize_type_resources(&mut input.ty, by_binding);
        }
        for output in &mut prepass.entry.outputs {
            normalize_type_resources(&mut output.ty, by_binding);
        }
        for (ty, _) in &mut prepass.entry.params {
            normalize_type_resources(ty, by_binding);
        }
        normalize_type_resources(&mut prepass.entry.return_ty, by_binding);
        for declaration in &mut prepass.entry.resource_declarations {
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

pub(crate) fn physicalize_graph_resources(
    graph: EGraph,
    bindings: &PhysicalResourceTable,
) -> Result<
    (
        EGraph<PhysicalResourceRef>,
        LookupMap<NodeId, NodeId>,
        LookupMap<BlockId, BlockId>,
    ),
    String,
> {
    let (mut graph, node_map, block_map) = graph.try_map_resources(|reference| {
        let resource = reference.0;
        bindings
            .binding(resource)
            .ok_or_else(|| format!("semantic resource {:?} has no physical binding", resource))
    })?;
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
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            if let super::types::SideEffectKind::Soac(soac) = &mut effect.kind {
                soac.visit_types_mut(|ty| physicalize_type_resources(ty, bindings));
            }
        }
    }
    let nodes = graph.types.keys().copied().collect::<Vec<_>>();
    for node in nodes {
        let mut ty = graph.types[&node].clone();
        physicalize_type_resources(&mut ty, bindings);
        graph.retype_node(node, ty);
    }
    Ok((graph, node_map, block_map))
}

pub(crate) fn physicalize_type_resources(ty: &mut Type<TypeName>, bindings: &PhysicalResourceTable) {
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

fn allocate_filter_work_resources(inner: &mut SemanticProgram) {
    use super::types::{EgirSoac, FilterOutput, FilterState, SegExtent, SideEffectKind};
    let mut pending = Vec::new();
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
                let element_count_size = match space.dims.first() {
                    Some(SegExtent::Fixed(count)) if space.dims.len() == 1 => {
                        LogicalSize::FixedBytes(*count as u64 * 4)
                    }
                    Some(SegExtent::ResourceLength {
                        resource, elem_bytes, ..
                    }) if space.dims.len() == 1 => LogicalSize::LikeResource {
                        resource: resource.0,
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
                for (slot, (kind, size)) in [
                    (CompilerResourceKind::FilterFlags, element_count_size.clone()),
                    (CompilerResourceKind::FilterOffsets, element_count_size.clone()),
                    (
                        CompilerResourceKind::FilterScanBlockSums,
                        worker_count_size.clone(),
                    ),
                    (
                        CompilerResourceKind::FilterScanBlockOffsets,
                        worker_count_size.clone(),
                    ),
                ]
                .into_iter()
                .enumerate()
                {
                    let compiler = CompilerResource::new(kind, owner, slot);
                    pending.push((compiler, size));
                }
            }
        }
    }
    for (compiler, size) in pending {
        let id = ResourceId(inner.resources.len() as u32);
        inner.resources.push(LogicalResource {
            id,
            origin: ResourceOrigin::Compiler(compiler),
            elem_ty: Type::Constructed(TypeName::UInt(32), vec![]),
            size,
        });
    }
}

/// Classify the scalar resources owned by TLC-originated prepass requirements.
fn scalar_handoff_resources(inner: &SemanticProgram) -> HashMap<ResourceId, CompilerResource> {
    let consumer_inputs: std::collections::HashSet<_> = inner
        .entry_points
        .iter()
        .flat_map(|entry| {
            entry.resource_declarations.iter().filter_map(|declaration| {
                (declaration.role == interface::StorageRole::Input).then_some(declaration.resource.0)
            })
        })
        .collect();
    let mut resources = HashMap::new();
    for entry in &inner.entry_points {
        if inner.prepass_roles.get(&entry.name) != Some(&PrepassKind::Scalar) {
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
                && consumer_inputs.contains(&declaration.resource.0)
            {
                let resource = declaration.resource.0;
                resources.insert(
                    resource,
                    CompilerResource::new(CompilerResourceKind::ScalarHandoff, owner.clone(), 0),
                );
            }
        }
    }
    resources
}

/// Runtime `filter` bindings, classified so the mirror gives them a precise
/// `CompilerResourceKind` rather than generic `Staging`.
fn filter_resource_kinds(inner: &SemanticProgram) -> HashMap<ResourceId, CompilerResource> {
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
                        kinds.insert(
                            scratch.0,
                            CompilerResource::new(CompilerResourceKind::FilterScratch, owner, 0),
                        );
                        if let super::types::RuntimeFilterLength::EntryOutput(len) = length {
                            kinds.insert(
                                len.0,
                                CompilerResource::new(CompilerResourceKind::FilterLenCell, owner, 1),
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
                                kinds.insert(resource.0, CompilerResource::new(kind, owner, slot));
                            }
                        }
                    }
                }
            }
        }
    }
    kinds
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

#[derive(Clone, Debug)]
pub struct SemanticFunc {
    pub name: String,
    pub span: Span,
    pub linkage_name: Option<String>,
    pub params: Vec<(Type<TypeName>, String)>,
    pub return_ty: Type<TypeName>,
    pub graph: EGraph,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
    pub aliases: LookupMap<NodeId, NodeId>,
}

impl SemanticFunc {
    pub fn new(
        name: String,
        span: Span,
        linkage_name: Option<String>,
        params: Vec<(Type<TypeName>, String)>,
        return_ty: Type<TypeName>,
        graph: EGraph,
        control_headers: LookupMap<BlockId, ControlHeader>,
    ) -> Self {
        SemanticFunc {
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

/// Callable after the semantic-to-physical resource boundary. Its graph can
/// carry backend bindings only; constructing it requires consuming an
/// `SemanticFunc` through physicalization.
#[derive(Clone, Debug)]
pub struct PhysicalFunc {
    pub name: String,
    pub span: Span,
    pub linkage_name: Option<String>,
    pub params: Vec<(Type<TypeName>, String)>,
    pub return_ty: Type<TypeName>,
    pub graph: EGraph<PhysicalResourceRef>,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
    pub aliases: LookupMap<NodeId, NodeId>,
}

/// Callable-region projection owned by a physical program.
#[derive(Clone)]
pub struct PhysicalRegion {
    pub name: String,
    pub params: Vec<(Type<TypeName>, String)>,
    pub return_ty: Type<TypeName>,
    pub graph: EGraph<PhysicalResourceRef>,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
}

impl PhysicalRegion {
    pub fn from_function(function: &PhysicalFunc) -> Self {
        Self {
            name: function.name.clone(),
            params: function.params.clone(),
            return_ty: function.return_ty.clone(),
            graph: function.graph.clone(),
            control_headers: function.control_headers.clone(),
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
    pub name: String,
    pub span: Span,
    pub execution_model: ExecutionModel,
    pub inputs: Vec<EntryInput>,
    pub outputs: Vec<EntryOutput>,
    /// Logical resources associated with ABI slots. Host bindings are only a
    /// publication constraint; semantic passes use these identities directly.
    pub resource_abi: EntryResourceAbi,
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

#[derive(Clone, Debug, Default)]
pub struct EntryResourceAbi {
    pub inputs: Vec<Option<ResourceId>>,
    pub outputs: Vec<Option<ResourceId>>,
}

/// TLC-originated scalar or gather producer removed from the semantic entry
/// arena at allocation. Its graph remains target-independent until the
/// planner selects and commits a kernel recipe.
pub struct PrepassRequirement {
    pub id: PrepassId,
    pub kind: PrepassKind,
    pub entry: SemanticEntry,
}

impl SemanticEntry {
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_resources(
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
            name,
            span,
            execution_model,
            inputs,
            outputs,
            resource_abi: EntryResourceAbi::default(),
            resource_declarations,
            params,
            return_ty,
            graph,
            control_headers,
            aliases: LookupMap::new(),
            output_routes: Vec::new(),
        }
    }
}

/// A complete, fresh entry projection owned by a kernel recipe. This is the
/// sole entry-shaped planner record: publication and physical construction
/// both read it, so fields cannot drift between parallel representations.
#[derive(Clone, Debug)]
pub struct PlannedEntry {
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
    pub output_routes: Vec<OutputRoute>,
}

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
            inputs: entry.inputs.clone(),
            outputs: entry.outputs.clone(),
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

impl PlannedEntry {
    pub fn project(entry: &SemanticEntry) -> Result<Self, String> {
        let projection = super::graph_projector::GraphProjector::new(&entry.graph, &entry.control_headers)
            .all_with_values(entry.output_routes.iter().map(|route| route.source.value).collect())?;
        let output_routes = entry
            .output_routes
            .iter()
            .map(|route| {
                Ok(OutputRoute {
                    source: SlotSource {
                        block: projection
                            .block(route.source.block)
                            .ok_or_else(|| "planned route block was not projected".to_string())?,
                        value: projection
                            .node(route.source.value)
                            .ok_or_else(|| "planned route value was not projected".to_string())?,
                    },
                    slot: route.slot,
                    writers: route
                        .writers
                        .iter()
                        .filter_map(|writer| match writer {
                            OutputWriter::Value(value) => projection.node(*value).map(OutputWriter::Value),
                            OutputWriter::Effect(effect) => {
                                projection.effect(*effect).map(OutputWriter::Effect)
                            }
                        })
                        .collect(),
                })
            })
            .collect::<Result<_, String>>()?;
        Ok(Self {
            name: entry.name.clone(),
            span: entry.span,
            execution_model: entry.execution_model.clone(),
            inputs: entry.inputs.clone(),
            outputs: entry.outputs.clone(),
            resource_declarations: entry.resource_declarations.clone(),
            params: entry.params.clone(),
            return_ty: entry.return_ty.clone(),
            aliases: projection.remap_aliases(&entry.aliases),
            graph: projection.graph,
            control_headers: projection.control_headers,
            output_routes,
        })
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
        .map(|declaration| {
            let binding = resources
                .binding(declaration.resource.0)
                .ok_or_else(|| format!("entry `{name}` references an unallocated resource"))?;
            Ok(interface::StorageBindingDecl {
                binding,
                role: declaration.role.clone(),
                elem_ty: declaration.elem_ty.clone(),
                length: buffer_len(&declaration.size, resources),
            })
        })
        .collect::<Result<_, String>>()?;
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
pub struct MaterializationRequirement {
    pub id: MaterializationId,
    pub producer: SemanticOpId,
    pub entry: SemanticEntry,
    pub substitutions: Vec<MaterializationSubstitution>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterializationSubstitution {
    pub resource: SemanticResourceRef,
    pub consumers: Vec<CompilerFlowEndpoint>,
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
    pub name: String,
    pub span: Span,
    pub execution_model: ExecutionModel,
    pub inputs: Vec<EntryInput>,
    pub outputs: Vec<EntryOutput>,
    pub storage_bindings: Vec<interface::StorageBindingDecl>,
    pub params: Vec<(Type<TypeName>, String)>,
    pub return_ty: Type<TypeName>,
    pub graph: EGraph<PhysicalResourceRef>,
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
    pub functions: Vec<SemanticFunc>,
    /// Extern function stubs. These don't have a body that flows through EGIR;
    /// they're already `Function` records with a 1-block Unreachable body and
    /// pass straight through.
    pub externs: Vec<Function>,
    pub entry_points: Vec<SemanticEntry>,
    /// TLC classification retained only until allocation extracts typed
    /// prepass requirements from the semantic entry arena.
    pub prepass_roles: LookupMap<String, PrepassKind>,
    pub prepasses: Vec<PrepassRequirement>,
    /// Shared-producer requirements discovered during logical allocation.
    /// These are planned and physicalized directly; they never join the
    /// semantic entry arena.
    pub materializations: Vec<MaterializationRequirement>,
    pub constants: Vec<Constant>,
    pub pipeline: PipelineDescriptor,
    /// Source names retained until the descriptor is published atomically at
    /// terminal lowering.
    pub input_names: LookupMap<(u32, u32), String>,
    /// Complete callable regions referenced by semantic Seg bodies, keyed by
    /// their arena index.
    pub regions: LookupMap<RegionId, SemanticRegion>,
    /// Name ↔ index interner shared with construction. Synthesized regions
    /// (e.g. scan offset wrappers) intern here to obtain a fresh index.
    pub region_interner: RegionInterner,
    /// Logical host and compiler resources. Compiler resources receive a
    /// physical binding only during target-aware lowering.
    pub resources: Vec<LogicalResource>,
    /// Whole-program semantic dependency DAG. Edges come from values, effect
    /// tokens, and conflicting logical resource accesses.
    pub semantic_dependencies: Vec<SemanticDependency>,
}

/// EGIR after the plan has validated and every physical entry has been
/// constructed. Only this type is accepted by expansion and SSA elaboration.
pub struct PhysicalProgram {
    pub functions: Vec<PhysicalFunc>,
    pub externs: Vec<Function>,
    pub entry_points: Vec<PhysicalEntry>,
    pub constants: Vec<Constant>,
    pub pipeline: PipelineDescriptor,
    pub input_names: LookupMap<(u32, u32), String>,
    pub regions: LookupMap<RegionId, PhysicalRegion>,
    pub region_interner: RegionInterner,
    pub resources: Vec<LogicalResource>,
    pub semantic_dependencies: Vec<SemanticDependency>,
    pub plan: ValidatedKernelPlan,
    pub physical_resources: PhysicalResourceTable,
}

fn physicalize_control_headers(
    headers: LookupMap<BlockId, ControlHeader>,
    block_map: &LookupMap<BlockId, BlockId>,
) -> LookupMap<BlockId, ControlHeader> {
    headers
        .into_iter()
        .map(|(block, header)| {
            let block = block_map[&block];
            let header = header.remap(&|target| block_map[&target]);
            (block, header)
        })
        .collect()
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
    let (mut graph, node_map, block_map) = physicalize_graph_resources(graph, resources)?;
    for (ty, _) in &mut params {
        physicalize_type_resources(ty, resources);
    }
    physicalize_type_resources(&mut return_ty, resources);
    super::parallelize::prepare_executable_graph(&mut graph, serial);
    Ok(PhysicalFunc {
        name,
        span,
        linkage_name,
        params,
        return_ty,
        graph,
        control_headers: physicalize_control_headers(control_headers, &block_map),
        aliases: aliases.into_iter().map(|(from, to)| (node_map[&from], node_map[&to])).collect(),
    })
}

fn physicalize_region(
    region: SemanticRegion,
    resources: &PhysicalResourceTable,
) -> Result<PhysicalRegion, String> {
    let SemanticRegion {
        name,
        mut params,
        mut return_ty,
        graph,
        control_headers,
    } = region;
    let (graph, _, block_map) = physicalize_graph_resources(graph, resources)?;
    for (ty, _) in &mut params {
        physicalize_type_resources(ty, resources);
    }
    physicalize_type_resources(&mut return_ty, resources);
    Ok(PhysicalRegion {
        name,
        params,
        return_ty,
        graph,
        control_headers: physicalize_control_headers(control_headers, &block_map),
    })
}

fn physicalize_entry(
    entry: &PlannedEntry,
    resources: &PhysicalResourceTable,
) -> Result<PhysicalEntry, String> {
    let mut inputs = entry.inputs.clone();
    let mut outputs = entry.outputs.clone();
    let mut declarations = entry.resource_declarations.clone();
    let mut params = entry.params.clone();
    let mut return_ty = entry.return_ty.clone();
    let (graph, nodes, blocks) = physicalize_graph_resources(entry.graph.clone(), resources)?;
    for input in &mut inputs {
        physicalize_type_resources(&mut input.ty, resources);
    }
    for output in &mut outputs {
        physicalize_type_resources(&mut output.ty, resources);
    }
    for (ty, _) in &mut params {
        physicalize_type_resources(ty, resources);
    }
    physicalize_type_resources(&mut return_ty, resources);
    for declaration in &mut declarations {
        physicalize_type_resources(&mut declaration.elem_ty, resources);
    }
    let storage_bindings = declarations
        .into_iter()
        .map(|declaration| {
            let binding = resources
                .binding(declaration.resource.0)
                .ok_or_else(|| format!("entry `{}` references an unallocated resource", entry.name))?;
            Ok(interface::StorageBindingDecl {
                binding,
                role: declaration.role,
                elem_ty: declaration.elem_ty,
                length: buffer_len(&declaration.size, resources),
            })
        })
        .collect::<Result<_, String>>()?;
    let output_routes = entry
        .output_routes
        .iter()
        .map(|route| {
            Ok(OutputRoute {
                source: SlotSource {
                    block: *blocks
                        .get(&route.source.block)
                        .ok_or_else(|| "planned output block was not physicalized".to_string())?,
                    value: *nodes
                        .get(&route.source.value)
                        .ok_or_else(|| "planned output value was not physicalized".to_string())?,
                },
                slot: route.slot,
                writers: route
                    .writers
                    .iter()
                    .map(|writer| match writer {
                        OutputWriter::Value(value) => nodes
                            .get(value)
                            .copied()
                            .map(OutputWriter::Value)
                            .ok_or_else(|| "planned output writer was not physicalized".to_string()),
                        OutputWriter::Effect(effect) => Ok(OutputWriter::Effect(*effect)),
                    })
                    .collect::<Result<_, String>>()?,
            })
        })
        .collect::<Result<_, String>>()?;
    Ok(PhysicalEntry {
        name: entry.name.clone(),
        span: entry.span,
        execution_model: entry.execution_model.clone(),
        inputs,
        outputs,
        storage_bindings,
        params,
        return_ty,
        graph,
        control_headers: physicalize_control_headers(entry.control_headers.clone(), &blocks),
        aliases: entry.aliases.iter().map(|(from, to)| (nodes[from], nodes[to])).collect(),
        output_routes,
    })
}

impl PhysicalProgram {
    pub fn from_validated(
        program: SemanticProgram,
        plan: ValidatedKernelPlan,
        physical_resources: PhysicalResourceTable,
        serial: bool,
        pipeline: PipelineDescriptor,
    ) -> Result<Self, String> {
        let entry_points = plan
            .physical_kernels()
            .map(|phase| physicalize_entry(phase.recipe.entry(), &physical_resources))
            .collect::<Result<Vec<_>, _>>()?;
        let mut functions = program
            .functions
            .into_iter()
            .map(|function| physicalize_function(function, &physical_resources, serial))
            .collect::<Result<Vec<_>, _>>()?;
        for generated in plan.generated_callables() {
            functions.push(physicalize_function(
                generated.clone(),
                &physical_resources,
                serial,
            )?);
        }
        let region_interner = plan.region_interner().clone();
        let mut regions = program
            .regions
            .into_iter()
            .map(|(id, region)| Ok((id, physicalize_region(region, &physical_resources)?)))
            .collect::<Result<LookupMap<_, _>, String>>()?;
        for function in &functions {
            let id = region_interner
                .get(&function.name)
                .ok_or_else(|| format!("physical callable `{}` has no region identity", function.name))?;
            regions.insert(id, PhysicalRegion::from_function(function));
        }
        Ok(Self {
            functions,
            externs: program.externs,
            entry_points,
            constants: program.constants,
            pipeline,
            input_names: program.input_names,
            regions,
            region_interner,
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
    regions: &mut LookupMap<RegionId, SemanticRegion>,
    function: &SemanticFunc,
) -> RegionId {
    let id = interner.intern(&function.name);
    regions.insert(id, SemanticRegion::from_function(function));
    id
}

impl SemanticProgram {
    pub fn new(
        functions: Vec<SemanticFunc>,
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
            prepass_roles: LookupMap::new(),
            prepasses: Vec::new(),
            materializations: Vec::new(),
            constants,
            pipeline,
            input_names: LookupMap::new(),
            regions,
            region_interner,
            resources: Vec::new(),
            semantic_dependencies: Vec::new(),
        }
    }

    /// Convenience: build an EGIR program wrapping a single function body.
    /// Used by the probe path in `from_tlc`.
    pub fn single_function(func: SemanticFunc) -> Self {
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
    pub fn define_region(&mut self, function: SemanticFunc) -> RegionId {
        let id = record_region(&mut self.region_interner, &mut self.regions, &function);
        self.functions.push(function);
        id
    }

    /// SSA function name backing a region index (the `PureOp::Call` ABI).
    pub fn region_name(&self, id: RegionId) -> &str {
        self.region_interner.name(id)
    }
}
