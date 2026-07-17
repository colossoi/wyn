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
use crate::interface;
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::ssa::types::{
    BlockId, Constant, ControlHeader, EntryInput, EntryOutput, ExecutionModel, Function,
};
use crate::types::TypeExt;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use super::parallelize::schedule::ValidatedKernelPlan;
use super::soac::{filter, hist, screma};
use super::types::{
    EGraph, EgirPhase, NodeId, Physical, Raw, RegionId, Scheduled, SegBody, SegExtent, SegSpace, Semantic,
    Soac, WynLanguage,
};

pub use super::ir::{OutputRoute, OutputSlotId, OutputWriter, RegionInterner, SlotSource};
pub type Region<P = Semantic, Lang = WynLanguage> = super::ir::Region<P, Lang>;
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
pub struct SemanticOpId(pub u32);

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
pub type RawRegion = Region<Raw>;
pub type SemanticRegion = Region<Semantic>;
pub type ScheduledRegion = Region<Scheduled>;
pub type PhysicalRegion = Region<Physical>;
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
    /// physical requirements from explicit semantic materialization records.
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

impl LogicalResource {
    pub fn host_binding(&self) -> Option<crate::BindingRef> {
        match self.origin {
            ResourceOrigin::Host(binding) => Some(binding),
            ResourceOrigin::Compiler(_) => None,
        }
    }
}

pub(crate) fn host_resource_map(resources: &[LogicalResource]) -> HashMap<crate::BindingRef, ResourceId> {
    resources.iter().filter_map(|resource| Some((resource.host_binding()?, resource.id))).collect()
}

/// Complete the authoritative logical-resource manifest at the allocation
/// boundary. Host resources already have stable identities; this pass adds
/// semantic residency requirements and segmented-operation scratch, then
/// records explicit producer/consumer flows.
pub fn plan_logical_resources(inner: SemanticProgram) -> AllocatedProgram {
    let mut allocated = AllocatedProgram {
        semantic: inner,
        materializations: Vec::new(),
    };
    classify_existing_compiler_resources(&mut allocated);
    super::residency::run(&mut allocated);
    super::soac::filter::resolve_scratch_sizes(&mut allocated);
    super::soac::filter::allocate_work_resources(&mut allocated);
    let mut scratch =
        super::parallelize::enumerate_seg_scratch(&allocated, allocated.resources.len() as u32);
    allocated.resources.append(&mut scratch);
    strip_compiler_abi(&mut allocated);
    record_compiler_resource_flows(&mut allocated);
    if cfg!(debug_assertions) {
        verify_allocated_resources(&allocated).expect("invalid allocated semantic resources");
    }
    allocated
}

pub(crate) fn verify_allocated_resources(inner: &AllocatedProgram) -> Result<(), String> {
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
    for declaration in inner.entries_with_endpoints().flat_map(|(_, entry)| &entry.resource_declarations) {
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
        let logical = inner
            .resources
            .get_mut(resource.0 as usize)
            .filter(|logical| logical.id == resource)
            .expect("compiler classification references a missing resource");
        logical.origin = ResourceOrigin::Compiler(compiler);
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
    inner: &mut RawProgram,
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
        for input in &mut entry.inputs {
            input.resource = input
                .storage_binding
                .and_then(|binding| by_binding.get(&binding).copied())
                .map(SemanticResourceRef)
                .or_else(|| semantic_type_resource(&input.ty));
        }
        for output in &mut entry.outputs {
            output.resource = output
                .storage_binding
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
    let strip = |inputs: &mut Vec<super::ir::EntryInput<SemanticResourceRef>>,
                 outputs: &mut Vec<super::ir::EntryOutput<SemanticResourceRef>>,
                 routes: &mut Vec<OutputRoute>| {
        for input in inputs.iter_mut() {
            if input.resource.is_some_and(|resource| compiler_resources.contains(&resource.0)) {
                input.storage_binding = None;
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
    for requirement in &mut inner.materializations {
        strip(
            &mut requirement.entry.inputs,
            &mut requirement.entry.outputs,
            &mut requirement.entry.output_routes,
        );
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
    for region in inner.regions.values_mut() {
        for (ty, _) in &mut region.params {
            normalize_type_resources(ty, by_binding);
        }
        normalize_type_resources(&mut region.return_ty, by_binding);
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
            if let super::types::SideEffectKind::Soac(_, soac) = &mut effect.kind {
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
            if let super::types::SideEffectKind::Soac(_, soac) = &mut effect.kind {
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
    fn binding(
        reference: SemanticResourceRef,
        bindings: &PhysicalResourceTable,
    ) -> Result<PhysicalResourceRef, String> {
        bindings
            .binding(reference.0)
            .ok_or_else(|| format!("semantic resource {:?} has no physical binding", reference.0))
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
        Ok(SegSpace {
            level: space.level,
            dims: space
                .dims
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
                            resource: binding(resource, bindings)?,
                            elem_bytes,
                        },
                        SegExtent::Value(node) => SegExtent::Value(nodes[&node]),
                    })
                })
                .collect::<Result<_, String>>()?,
        })
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
                        resource: binding(resource.resource, bindings)?,
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
                scratch: binding(scratch, bindings)?,
                length: match length {
                    filter::RuntimeLength::ViewOnly => filter::RuntimeLength::ViewOnly,
                    filter::RuntimeLength::Stored(resource) => {
                        filter::RuntimeLength::Stored(binding(resource, bindings)?)
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
            flags: binding(buffers.flags, bindings)?,
            offsets: binding(buffers.offsets, bindings)?,
            block_sums: binding(buffers.block_sums, bindings)?,
            block_offsets: binding(buffers.block_offsets, bindings)?,
        })
    }

    Ok(match soac {
        Soac::Screma(screma::Op::Map { lanes, state }) => {
            let state = match state {
                screma::ScheduledState::Serial => screma::PhysicalMapState::Serial,
                screma::ScheduledState::Segmented(segment) => {
                    screma::PhysicalMapState::Segmented(physical_segment(segment, nodes, bindings)?)
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
                filter::ScheduledState::Serial {
                    space: iteration_space,
                    storage,
                } => filter::ScheduledState::Serial {
                    space: space(iteration_space, nodes, bindings)?,
                    storage: filter_output(storage, bindings)?,
                },
                filter::ScheduledState::Parallel {
                    space: iteration_space,
                    storage,
                    plan,
                } => filter::ScheduledState::Parallel {
                    space: space(iteration_space, nodes, bindings)?,
                    storage: filter::RuntimeStorage {
                        scratch: binding(storage.scratch, bindings)?,
                        length: match storage.length {
                            filter::RuntimeLength::ViewOnly => filter::RuntimeLength::ViewOnly,
                            filter::RuntimeLength::Stored(resource) => {
                                filter::RuntimeLength::Stored(binding(resource, bindings)?)
                            }
                        },
                    },
                    plan: filter::ParallelPlan {
                        stage: plan.stage,
                        buffers: work_buffers(plan.buffers, bindings)?,
                    },
                },
            };
            Soac::Filter(filter::Op { body, state })
        }
        Soac::Hist(hist::Op { mut body, state }) => {
            body.body = seg_body(body.body, nodes);
            let state = match state {
                hist::ScheduledState::Serial => hist::ScheduledState::Serial,
                hist::ScheduledState::Segmented(iteration_space) => {
                    hist::ScheduledState::Segmented(space(iteration_space, nodes, bindings)?)
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
            bindings
                .binding(resource)
                .ok_or_else(|| format!("semantic resource {:?} has no physical binding", resource))
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
            *name = TypeName::Buffer(
                bindings.binding(resource).expect("semantic type resource must have a physical binding"),
            );
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

pub type RawFunc = Func<Raw>;
pub type SemanticFunc = Func<Semantic>;
pub type ScheduledFunc = Func<Scheduled>;
pub type PhysicalFunc = Func<Physical>;

pub type RawEntry = Entry<Raw>;
pub type SemanticEntry = Entry<Semantic>;
pub type ScheduledEntry = Entry<Scheduled>;

/// A complete, fresh entry projection owned by a kernel recipe. This is the
/// sole entry-shaped planner record: publication and physical construction
/// both read it, so fields cannot drift between parallel representations.
#[derive(Clone, Debug)]
pub struct PlannedEntry<P: EgirPhase = Semantic> {
    pub name: String,
    pub span: Span,
    pub execution_model: ExecutionModel,
    pub inputs: Vec<EntryInput>,
    pub outputs: Vec<EntryOutput>,
    pub resource_declarations: Vec<SemanticResourceDecl>,
    pub params: Vec<(Type<TypeName>, String)>,
    pub return_ty: Type<TypeName>,
    pub graph: EGraph<P>,
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
            inputs: entry.inputs.iter().map(|input| input.interface.clone()).collect(),
            outputs: entry.outputs.iter().map(|output| output.interface.clone()).collect(),
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

impl PlannedEntry<Semantic> {
    pub fn project(entry: &SemanticEntry) -> Result<Self, String> {
        let projection = super::graph_projector::GraphProjector::new(&entry.graph, &entry.control_headers)
            .all_with_values(entry.output_routes.iter().map(|route| route.source.value).collect())?;
        Self::from_projection(
            projection,
            entry.name.clone(),
            entry.span,
            entry.execution_model.clone(),
            entry.inputs.iter().map(|input| input.interface.clone()).collect(),
            entry.outputs.iter().map(|output| output.interface.clone()).collect(),
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
            inputs,
            outputs,
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

impl<P: EgirPhase> PlannedEntry<P> {
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

pub struct MaterializationRequirement {
    pub id: MaterializationId,
    pub kind: MaterializationKind,
    /// SOAC provenance when the source is a semantic operation. Captured
    /// parallel preludes intentionally do not receive synthetic operation ids.
    pub producer: Option<SemanticOpId>,
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
    pub graph: EGraph<Physical>,
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
        let mut used = host_resource_map(resources).into_keys().collect::<std::collections::HashSet<_>>();
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

/// Low-level raw EGIR plus the logical resources established during TLC
/// conversion. Raw EGIR has neither semantic dependencies nor allocation
/// requirements.
pub struct RawProgram {
    pub ir: Program<Raw>,
    pub resources: Vec<LogicalResource>,
}

impl RawProgram {
    pub fn new(
        functions: Vec<RawFunc>,
        externs: Vec<Function>,
        entry_points: Vec<RawEntry>,
        constants: Vec<Constant>,
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
            resources: Vec::new(),
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
    pub resources: Vec<LogicalResource>,
    pub semantic_dependencies: Vec<SemanticDependency>,
}

impl SemanticProgram {
    pub fn new(
        functions: Vec<SemanticFunc>,
        externs: Vec<Function>,
        entry_points: Vec<SemanticEntry>,
        constants: Vec<Constant>,
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
            resources: Vec::new(),
            semantic_dependencies: Vec::new(),
        }
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
    pub materializations: Vec<MaterializationRequirement>,
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
    let (graph, scheduled_blocks) = super::parallelize::prepare::graph(graph, false)?;
    let control_headers = remap_control_headers(&control_headers, |block| scheduled_blocks[&block]);
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
        control_headers: remap_control_headers(&control_headers, |block| block_map[&block]),
    })
}

fn physicalize_entry(
    entry: &PlannedEntry<Scheduled>,
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
        storage_bindings,
        params,
        return_ty,
        graph,
        control_headers: remap_control_headers(&entry.control_headers, |block| blocks[&block]),
        aliases: entry.aliases.iter().map(|(from, to)| (nodes[from], nodes[to])).collect(),
        output_routes,
    })
}

impl PhysicalProgram {
    pub fn from_validated(
        program: AllocatedProgram,
        plan: ValidatedKernelPlan,
        physical_resources: PhysicalResourceTable,
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
            semantic_dependencies,
        } = semantic;
        let Program {
            functions,
            externs,
            entry_points: _,
            constants,
            pipeline: _,
            input_names,
            regions,
            region_interner: _,
        } = ir;
        let entry_points = plan
            .physical_kernels()
            .map(|phase| physicalize_entry(phase.recipe.entry(), &physical_resources))
            .collect::<Result<Vec<_>, _>>()?;
        let mut functions = functions
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
        let mut regions = regions
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
            externs,
            entry_points,
            constants,
            pipeline,
            input_names,
            regions,
            region_interner,
            resources,
            semantic_dependencies,
            plan,
            physical_resources,
        })
    }
}

impl AllocatedProgram {
    pub(crate) fn alloc_compiler_resource(
        &mut self,
        compiler: CompilerResource,
        elem_ty: Type<TypeName>,
        size: LogicalSize,
    ) -> ResourceId {
        let id = ResourceId(self.resources.len() as u32);
        self.resources.push(LogicalResource {
            id,
            origin: ResourceOrigin::Compiler(compiler),
            elem_ty,
            size,
        });
        id
    }

    pub(crate) fn entries_with_endpoints(
        &self,
    ) -> impl Iterator<Item = (CompilerFlowEndpoint, &SemanticEntry)> {
        self.semantic
            .ir
            .entry_points
            .iter()
            .enumerate()
            .map(|(index, entry)| (CompilerFlowEndpoint::Entry(SemanticEntryId(index as u32)), entry))
            .chain(self.materializations.iter().map(|requirement| {
                (
                    CompilerFlowEndpoint::Materialization(requirement.id),
                    &requirement.entry,
                )
            }))
    }
}
