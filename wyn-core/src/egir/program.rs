//! Whole-program EGIR container + per-body records.
//!
//! Compiler state is explicit at both boundaries: public semantic pipeline
//! newtypes wrap `SemanticProgram`, while each graph is parameterized by its
//! phase-specific resource identity. Physicalization rebuilds those graphs as
//! `EGraph<Physical>` inside a distinct `PhysicalProgram`.
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
use super::soac::{filter, hist, screma};
use super::types::{
    EGraph, EffectToken, EgirPhase, NodeId, Physical, Raw, RegionId, Scheduled, SegBody, SegExtent,
    SegSpace, Semantic, Soac,
};

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

/// Callable body arena entry used by SegOps.
#[derive(Clone)]
pub struct Region<P: EgirPhase = Semantic> {
    pub name: String,
    pub params: Vec<(Type<TypeName>, String)>,
    pub return_ty: Type<TypeName>,
    pub graph: EGraph<P>,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
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
pub fn plan_logical_resources(inner: &mut SemanticProgram) {
    classify_existing_compiler_resources(inner);
    super::residency::run(inner);
    super::soac::filter::resolve_scratch_sizes(inner);
    super::soac::filter::allocate_work_resources(inner);
    let mut scratch = super::parallelize::enumerate_seg_scratch(inner, inner.resources.len() as u32);
    inner.resources.append(&mut scratch);
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
    let source_outputs = inner
        .entry_points
        .iter()
        .flat_map(|entry| entry.resource_abi.outputs.iter().flatten().copied())
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

fn record_compiler_resource_flows(inner: &mut SemanticProgram) {
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
    for requirement in &mut inner.materializations {
        strip(
            &mut requirement.entry.inputs,
            &mut requirement.entry.outputs,
            &mut requirement.entry.resource_abi,
            &mut requirement.entry.output_routes,
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
    for (_, entry) in inner.entries_with_endpoints_mut() {
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
            if let super::types::SideEffectKind::Soac(soac) = &mut effect.kind {
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
            if let super::types::SideEffectKind::Soac(soac) = &mut effect.kind {
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

    fn screma_body(mut body: screma::Body, nodes: &LookupMap<NodeId, NodeId>) -> screma::Body {
        for map in &mut body.maps {
            map.body = seg_body(map.body.clone(), nodes);
        }
        body.kind = match body.kind {
            screma::Kind::Map => screma::Kind::Map,
            screma::Kind::Reduce(values) => screma::Kind::Reduce(operators(values, nodes)),
            screma::Kind::Scan(values) => screma::Kind::Scan(operators(values, nodes)),
            screma::Kind::Composite(values) => {
                let map = |value| match value {
                    screma::CompositeOperator::Reduce(value) => {
                        screma::CompositeOperator::Reduce(operator(value, nodes))
                    }
                    screma::CompositeOperator::Scan(value) => {
                        screma::CompositeOperator::Scan(operator(value, nodes))
                    }
                };
                screma::Kind::Composite(screma::NonEmpty {
                    first: map(values.first),
                    rest: values.rest.into_iter().map(map).collect(),
                })
            }
        };
        body
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
                    filter::RuntimeLength::EntryOutput(resource) => {
                        filter::RuntimeLength::EntryOutput(binding(resource, bindings)?)
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
        Soac::Screma(screma::Op { body, state }) => {
            let segmented_map = matches!(body.kind, screma::Kind::Map);
            let body = screma_body(body, nodes);
            let state = match state {
                screma::ScheduledState::Serial => screma::PhysicalState::Serial,
                screma::ScheduledState::Segmented {
                    space: iteration_space,
                    output_slots,
                    resources,
                } => {
                    if !segmented_map {
                        return Err("scheduled segmented reduce/scan reached physicalization".into());
                    }
                    screma::PhysicalState::SegMap {
                        space: space(iteration_space, nodes, bindings)?,
                        output_slots,
                        resources: resources
                            .into_iter()
                            .map(|resource| {
                                Ok(super::types::SegResourceAccess {
                                    resource: binding(resource.resource, bindings)?,
                                    access: resource.access,
                                })
                            })
                            .collect::<Result<_, String>>()?,
                    }
                }
            };
            Soac::Screma(screma::Op { body, state })
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
                            filter::RuntimeLength::EntryOutput(resource) => {
                                filter::RuntimeLength::EntryOutput(binding(resource, bindings)?)
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
        |soac, nodes| physicalize_soac(soac, nodes, bindings),
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

#[derive(Clone, Debug)]
pub struct Func<P: EgirPhase = Semantic> {
    pub name: String,
    pub span: Span,
    pub linkage_name: Option<String>,
    pub params: Vec<(Type<TypeName>, String)>,
    pub return_ty: Type<TypeName>,
    pub graph: EGraph<P>,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
    pub aliases: LookupMap<NodeId, NodeId>,
}

impl<P: EgirPhase> Func<P> {
    pub fn new(
        name: String,
        span: Span,
        linkage_name: Option<String>,
        params: Vec<(Type<TypeName>, String)>,
        return_ty: Type<TypeName>,
        graph: EGraph<P>,
        control_headers: LookupMap<BlockId, ControlHeader>,
    ) -> Self {
        Self {
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

pub type RawFunc = Func<Raw>;
pub type SemanticFunc = Func<Semantic>;
pub type ScheduledFunc = Func<Scheduled>;
pub type PhysicalFunc = Func<Physical>;

impl<P: EgirPhase> Region<P> {
    pub fn from_function(function: &Func<P>) -> Self {
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

/// Declared output ownership derived during TLC-to-EGIR conversion and carried
/// through physicalization. `source.value` is the user-level value, `slot` is the
/// declared output it fulfils, and `writers` are populated by output
/// realization. The slot's `EntryOutput::storage_binding` then identifies the
/// host resource until logical-resource allocation assigns a `ResourceId`.
#[derive(Debug, Clone)]
pub struct OutputRoute {
    pub source: SlotSource,
    pub slot: OutputSlotId,
    pub writers: Vec<OutputWriter>,
}

pub struct Entry<P: EgirPhase = Semantic> {
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
    pub graph: EGraph<P>,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
    pub aliases: LookupMap<NodeId, NodeId>,
    /// Explicit value-to-output routes. A slot can have several routes when
    /// distinct CFG paths write it. Output realization fills `writers`; later
    /// semantic passes consume these declarations instead of reconstructing
    /// ownership from storage-view provenance and effect shape.
    pub output_routes: Vec<OutputRoute>,
}

pub type RawEntry = Entry<Raw>;
pub type SemanticEntry = Entry<Semantic>;
pub type ScheduledEntry = Entry<Scheduled>;

#[derive(Clone, Debug, Default)]
pub struct EntryResourceAbi {
    pub inputs: Vec<Option<ResourceId>>,
    pub outputs: Vec<Option<ResourceId>>,
}

impl<P: EgirPhase> Entry<P> {
    fn visit_types_mut(&mut self, mut visit: impl FnMut(&mut Type<TypeName>)) {
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
        graph: EGraph<P>,
        control_headers: LookupMap<BlockId, ControlHeader>,
    ) -> Self {
        Self {
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

impl PlannedEntry<Semantic> {
    pub fn project(entry: &SemanticEntry) -> Result<Self, String> {
        let projection = super::graph_projector::GraphProjector::new(&entry.graph, &entry.control_headers)
            .all_with_values(entry.output_routes.iter().map(|route| route.source.value).collect())?;
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
}

pub struct MaterializationRequirement<P: EgirPhase = Semantic> {
    pub id: MaterializationId,
    pub kind: MaterializationKind,
    pub producer: SemanticOpId,
    pub entry: Entry<P>,
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

/// Whole-program EGIR container. Wrapped by the semantic `EgirRaw` /
/// `EgirOutputsRealized` / `EgirSegmented` / `EgirOptimized` /
/// `EgirAllocated` newtypes at the public-API layer (see `crate::lib`).
pub struct Program<P: EgirPhase> {
    pub functions: Vec<Func<P>>,
    /// Extern function stubs. These don't have a body that flows through EGIR;
    /// they're already `Function` records with a 1-block Unreachable body and
    /// pass straight through.
    pub externs: Vec<Function>,
    pub entry_points: Vec<Entry<P>>,
    /// Residency requirements discovered during logical allocation.
    /// These are planned and physicalized directly; they never join the
    /// semantic entry arena.
    pub materializations: Vec<MaterializationRequirement<P>>,
    pub constants: Vec<Constant>,
    pub pipeline: PipelineDescriptor,
    /// Source names retained until the descriptor is published atomically at
    /// terminal lowering.
    pub input_names: LookupMap<(u32, u32), String>,
    /// Complete callable regions referenced by semantic Seg bodies, keyed by
    /// their arena index.
    pub regions: LookupMap<RegionId, Region<P>>,
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

pub type RawProgram = Program<Raw>;
pub type SemanticProgram = Program<Semantic>;
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
fn record_region<P: EgirPhase>(
    interner: &mut RegionInterner,
    regions: &mut LookupMap<RegionId, Region<P>>,
    function: &Func<P>,
) -> RegionId {
    let id = interner.intern(&function.name);
    regions.insert(id, Region::<P>::from_function(function));
    id
}

impl<P: EgirPhase> Program<P> {
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

    pub(crate) fn entries_with_endpoints(&self) -> impl Iterator<Item = (CompilerFlowEndpoint, &Entry<P>)> {
        self.entry_points
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

    fn entries_with_endpoints_mut(
        &mut self,
    ) -> impl Iterator<Item = (CompilerFlowEndpoint, &mut Entry<P>)> {
        self.entry_points
            .iter_mut()
            .enumerate()
            .map(|(index, entry)| (CompilerFlowEndpoint::Entry(SemanticEntryId(index as u32)), entry))
            .chain(self.materializations.iter_mut().map(|requirement| {
                (
                    CompilerFlowEndpoint::Materialization(requirement.id),
                    &mut requirement.entry,
                )
            }))
    }

    pub fn new(
        functions: Vec<Func<P>>,
        externs: Vec<Function>,
        entry_points: Vec<Entry<P>>,
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
        Self {
            functions,
            externs,
            entry_points,
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
    pub fn single_function(func: Func<P>) -> Self {
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
    pub fn define_region(&mut self, function: Func<P>) -> RegionId {
        let id = record_region(&mut self.region_interner, &mut self.regions, &function);
        self.functions.push(function);
        id
    }

    /// SSA function name backing a region index (the `PureOp::Call` ABI).
    pub fn region_name(&self, id: RegionId) -> &str {
        self.region_interner.name(id)
    }
}
