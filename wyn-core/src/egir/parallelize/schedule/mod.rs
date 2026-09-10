//! First-class compute schedule produced by EGIR parallelization.
//!
//! A schedule owns the relationship between generated entry points, dispatch
//! domains, resource accesses, and phase dependencies.  The pipeline
//! descriptor is a publication format derived from this graph after every
//! lowering has finished; it is not mutated while an individual lowering is
//! still speculative.

use crate::egir;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::egir::program::{
    GeneratedStageKind, LogicalResourceArena, OutputSlotId, PlannedEntry, PlannedPublication,
    SemanticResourceRef, StageOrigin, StagedProgram,
};
use crate::egir::soac::filter;
use crate::egir::types::{Scheduled, SegExtent, SegResourceAccess};
use crate::flow::ExecutionModel;
use crate::pipeline_descriptor::{
    Binding, ComputePipeline, ComputeStage, DispatchLen, DispatchSize, Pipeline, PipelineDescriptor,
};
use crate::{BindingRef, EntryId, ResourceId};
use wyn_base::IdSource;
pub use wyn_kernel_graph::KernelId;
use wyn_kernel_graph::{Builder, Fragment, Plan};
use wyn_staged_ir::StageId;

use super::declared_resources;

mod finalize;
mod validation;

#[cfg(test)]
mod tests;

type Topology = Plan<StageId, ResourceId, PipelineId>;
type TopologyBuilder = Builder<StageId, ResourceId, PipelineId>;

/// The immutable topology and separately owned EGIR body catalog.
#[derive(Debug)]
pub(in crate::egir) struct KernelPlan {
    topology: Topology,
    catalog: BTreeMap<KernelId, PreparedKernel>,
    pipelines: Vec<ScheduledPipeline>,
    source_entries: BTreeMap<EntryId, PlannedPublication>,
}

pub(super) struct ScheduleBuilder {
    topology: TopologyBuilder,
    catalog: BTreeMap<KernelId, PreparedKernel>,
    stages: BTreeMap<StageId, StageMetadata>,
    pipelines: BTreeMap<PipelineId, Pipeline>,
    graphics_associations: BTreeMap<PipelineId, Vec<EntryId>>,
    source_entries: BTreeMap<EntryId, PlannedPublication>,
    kernel_ids: IdSource<KernelId>,
}

/// Registration retains only stage policy and ABI facts. Recipes own all bodies.
pub(super) struct StageMetadata {
    primary: KernelId,
    compute: bool,
    source_entry: Option<EntryId>,
    dispatch: KernelDispatch,
    output_routes: Vec<OutputRouteProjection>,
    required_elements: Option<u32>,
    generated_kind: Option<GeneratedStageKind>,
}

impl KernelPlan {
    pub(in crate::egir::parallelize) fn debug_assert_authored_only(
        &self,
        authored_stage_count: usize,
        generated_callable_count: usize,
    ) {
        debug_assert_eq!(
            generated_callable_count, 0,
            "authored-only scheduling unexpectedly produced a callable"
        );
        debug_assert_eq!(
            self.catalog.len(),
            authored_stage_count,
            "authored-only scheduling unexpectedly changed the stage count"
        );
    }
    pub(in crate::egir) fn into_physical_entries(mut self) -> Vec<PlannedEntry<Scheduled>> {
        self.topology
            .kernel_order()
            .iter()
            .map(|id| {
                self.catalog.remove(id).unwrap_or_else(|| unreachable!("validated kernel catalog")).entry
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(super) enum KernelMutationError {
    #[error("{0}")]
    InvalidKernel(String),
    #[error(transparent)]
    Topology(#[from] wyn_kernel_graph::Error),
}
impl From<String> for KernelMutationError {
    fn from(error: String) -> Self {
        Self::InvalidKernel(error)
    }
}

/// A body plus the target facts established by the code that created it.
/// Preparing the body only changes EGIR phase; it never re-inspects the graph
/// to infer dispatch, resources, or output ownership.
pub(super) struct PhaseSpec {
    body: PlannedEntry,
    label: &'static str,
    filter_plan: Option<super::prepare::ParallelFilterPlan>,
    hist_plan: Option<super::prepare::ParallelHistPlan>,
    expected_compute: bool,
    dispatch: KernelDispatch,
    resources: Vec<SegResourceAccess<ResourceId>>,
    serial_single_workgroup: bool,
    output_projection: Option<Vec<OutputSlotId>>,
}

impl PhaseSpec {
    pub(super) fn compute(body: PlannedEntry, dispatch: KernelDispatch, label: &'static str) -> Self {
        let resources = declared_resources(&body.resource_declarations);
        Self {
            body,
            label,
            filter_plan: None,
            hist_plan: None,
            expected_compute: true,
            dispatch,
            resources,
            serial_single_workgroup: false,
            output_projection: None,
        }
    }

    pub(super) fn graphics(body: PlannedEntry, dispatch: KernelDispatch) -> Self {
        let resources = declared_resources(&body.resource_declarations);
        Self {
            body,
            label: "graphics_passthrough",
            filter_plan: None,
            hist_plan: None,
            expected_compute: false,
            dispatch,
            resources,
            serial_single_workgroup: false,
            output_projection: None,
        }
    }

    pub(super) fn filter(
        body: PlannedEntry,
        dispatch: KernelDispatch,
        stage: filter::ParallelStage,
        config: filter::ParallelConfig<SemanticResourceRef>,
        storage: filter::RuntimeStorage<SemanticResourceRef>,
    ) -> Self {
        let label = match stage {
            filter::ParallelStage::Flags => "filter_flags",
            filter::ParallelStage::Scan => "filter_scan",
            filter::ParallelStage::Scatter => "filter_scatter",
        };
        let resources = declared_resources(&body.resource_declarations);
        Self {
            body,
            label,
            filter_plan: Some(super::prepare::ParallelFilterPlan::new(stage, config, storage)),
            hist_plan: None,
            expected_compute: true,
            dispatch,
            resources,
            serial_single_workgroup: false,
            output_projection: None,
        }
    }

    pub(super) fn hist(
        body: PlannedEntry,
        dispatch: KernelDispatch,
        owner: super::super::program::SemanticOpId,
        operations: Vec<egir::soac::hist::AtomicUpdate>,
    ) -> Self {
        let resources = declared_resources(&body.resource_declarations);
        Self {
            body,
            label: "hist_atomic",
            filter_plan: None,
            hist_plan: Some(super::prepare::ParallelHistPlan::new(owner, operations)),
            expected_compute: true,
            dispatch,
            resources,
            serial_single_workgroup: false,
            output_projection: None,
        }
    }

    pub(super) fn bucket(
        body: PlannedEntry,
        dispatch: KernelDispatch,
        owner: super::super::program::SemanticOpId,
        stage: egir::soac::hist::ParallelStage,
        topology: Option<egir::soac::hist::DispatchTopology>,
        storage: egir::soac::hist::BucketStorage<SemanticResourceRef>,
    ) -> Self {
        let label = match stage {
            egir::soac::hist::ParallelStage::Init => "bucket_init",
            egir::soac::hist::ParallelStage::Insert => "bucket_insert",
            egir::soac::hist::ParallelStage::Finish => "bucket_finish",
        };
        let resources = declared_resources(&body.resource_declarations);
        Self {
            body,
            label,
            filter_plan: None,
            hist_plan: Some(super::prepare::ParallelHistPlan::bucket(
                owner, stage, topology, storage,
            )),
            expected_compute: true,
            dispatch,
            resources,
            serial_single_workgroup: false,
            output_projection: None,
        }
    }
    pub(super) fn with_resources(mut self, resources: Vec<SegResourceAccess<ResourceId>>) -> Self {
        self.resources = resources;
        self
    }

    pub(super) fn with_output_projection(mut self, outputs: Option<Vec<usize>>) -> Self {
        self.output_projection = outputs.map(|outputs| outputs.into_iter().map(OutputSlotId).collect());
        self
    }

    fn prepare(self) -> Result<PreparedKernel, String> {
        let entry = super::prepare::entry(self.body, self.filter_plan, self.hist_plan)?;
        if entry.execution_model.is_compute() != self.expected_compute {
            let expected = if self.expected_compute { "compute" } else { "graphics" };
            return Err(format!("entry `{}` cannot use a {expected} body", entry.name));
        }
        let required_elements = self.dispatch.required_elements();
        let projected = self.output_projection.is_some();
        let output_routes = self
            .output_projection
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(physical, semantic_slot)| OutputRouteProjection {
                semantic_slot,
                physical_slot: OutputSlotId(physical),
            })
            .collect();
        Ok(PreparedKernel {
            label: self.label,
            entry,
            source_entry: None,
            output_routes,
            projected,
            dispatch: self.dispatch,
            resources: self.resources,
            serial_single_workgroup: self.serial_single_workgroup,
            required_elements,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputRouteProjection {
    pub semantic_slot: OutputSlotId,
    pub physical_slot: OutputSlotId,
}

/// Final descriptor indices are assigned exactly once, after grouping.
#[derive(Debug)]
struct ScheduledPipeline {
    id: PipelineId,
    template: Pipeline,
    /// Graphics stage associations retain authored order.
    graphics_entries: Vec<EntryId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PipelineId(u32);

/// A prepared body and creator-supplied metadata; it contains no topology.
#[derive(Debug)]
struct PreparedKernel {
    label: &'static str,
    entry: PlannedEntry<Scheduled>,
    source_entry: Option<EntryId>,
    output_routes: Vec<OutputRouteProjection>,
    projected: bool,
    dispatch: KernelDispatch,
    resources: Vec<SegResourceAccess<ResourceId>>,
    serial_single_workgroup: bool,
    required_elements: Option<u32>,
}

impl PreparedKernel {
    fn entry_point(&self) -> &str {
        &self.entry.name
    }
    fn workgroup_size(&self) -> (u32, u32, u32) {
        execution_workgroup(&self.entry.execution_model)
    }
    fn resources(&self) -> &[SegResourceAccess<ResourceId>] {
        &self.resources
    }
}

/// Lowerers finish body preparation before returning a composable fragment.
pub(super) struct PreparedRecipe {
    fragment: Fragment,
    primary: KernelId,
    bodies: BTreeMap<KernelId, PreparedKernel>,
}

impl PreparedRecipe {
    pub(super) fn single(id: KernelId, spec: PhaseSpec) -> Result<Self, KernelMutationError> {
        Ok(Self {
            fragment: Fragment::kernel(id, 0),
            primary: id,
            bodies: [(id, spec.prepare()?)].into_iter().collect(),
        })
    }
    pub(super) fn unchanged(
        id: KernelId,
        body: PlannedEntry,
        stage: &StageMetadata,
    ) -> Result<Self, KernelMutationError> {
        let mut spec = if stage.compute {
            let label = match stage.generated_kind {
                Some(GeneratedStageKind::SharedArray) => "shared_array_materialization",
                Some(GeneratedStageKind::Gather) => "gather_prepass",
                Some(GeneratedStageKind::Scalar) => "scalar_prepass",
                Some(GeneratedStageKind::RuntimeArray) => "runtime_array_materialization",
                None => "serial_compute",
            };
            PhaseSpec::compute(body, stage.dispatch.clone(), label)
        } else {
            PhaseSpec::graphics(body, stage.dispatch.clone())
        };
        spec.serial_single_workgroup = stage.generated_kind.is_some();
        let mut recipe = Self::single(id, spec)?;
        recipe.bodies.get_mut(&id).unwrap_or_else(|| unreachable!("single kernel")).required_elements =
            stage.required_elements;
        Ok(recipe)
    }
    pub(super) fn sequence(
        specs: Vec<(KernelId, PhaseSpec)>,
        primary: KernelId,
    ) -> Result<Self, KernelMutationError> {
        let mut bodies = BTreeMap::new();
        let mut fragments = Vec::new();
        for (rank, (id, spec)) in specs.into_iter().enumerate() {
            let body = spec.prepare()?;
            if bodies.insert(id, body).is_some() {
                return Err(wyn_kernel_graph::Error::DuplicateKernel(id).into());
            }
            fragments.push(Fragment::kernel(id, rank as u64));
        }
        if !bodies.contains_key(&primary) {
            return Err("recipe primary has no prepared body".to_string().into());
        }
        Ok(Self {
            fragment: Fragment::sequence(fragments)?,
            primary,
            bodies,
        })
    }
    pub(super) fn parallel(recipes: Vec<Self>, primary: KernelId) -> Result<Self, KernelMutationError> {
        let mut fragments = Vec::new();
        let mut bodies = BTreeMap::new();
        for recipe in recipes {
            fragments.push(recipe.fragment);
            for (id, body) in recipe.bodies {
                if bodies.insert(id, body).is_some() {
                    return Err(wyn_kernel_graph::Error::DuplicateKernel(id).into());
                }
            }
        }
        if !bodies.contains_key(&primary) {
            return Err("recipe primary has no prepared body".to_string().into());
        }
        Ok(Self {
            fragment: Fragment::parallel(fragments)?,
            primary,
            bodies,
        })
    }
}

/// Persistent physical kernel graph.
///
/// Kernel bodies remain in the surrounding physical program's entry arena and
/// are named here by [`EntryId`]. This keeps the existing EGIR body traversal
/// machinery reusable while retaining kernel identity, physical dependencies,
/// dispatch, resource access, provenance, and output routing as first-class
/// program structure.
#[derive(Clone, Debug, Default)]
pub struct PhysicalKernelGraph {
    kernels: Vec<PhysicalKernel>,
}

impl PhysicalKernelGraph {
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    pub fn kernels(&self) -> impl ExactSizeIterator<Item = &PhysicalKernel> {
        self.kernels.iter()
    }

    /// Compatibility name for callers that present kernels as schedule
    /// phases. New compiler code should prefer [`Self::kernels`].
    pub fn phases(&self) -> impl ExactSizeIterator<Item = &PhysicalKernel> {
        self.kernels()
    }

    pub fn kernel(&self, id: KernelId) -> Option<&PhysicalKernel> {
        self.kernels.iter().find(|kernel| kernel.id == id)
    }

    /// Kernel identities in the immutable finalized dependency order.
    pub fn topological_kernel_ids(&self) -> Vec<KernelId> {
        self.kernels.iter().map(|kernel| kernel.id).collect()
    }

    /// Check the adapter-owned entry identities. Topology is established by
    /// the finalized kernel plan and cannot be mutated through this graph.
    pub fn validate(&self) -> Result<(), String> {
        let mut entries = HashSet::new();
        for kernel in &self.kernels {
            if !entries.insert(kernel.entry) {
                return Err(format!(
                    "physical entry {:?} is owned by multiple kernels",
                    kernel.entry
                ));
            }
        }
        Ok(())
    }

    pub(in crate::egir) fn validate_entry_ids(
        &self,
        entry_ids: impl IntoIterator<Item = EntryId>,
    ) -> Result<(), String> {
        let expected = self.kernels.iter().map(|kernel| kernel.entry).collect::<HashSet<_>>();
        let actual_ids = entry_ids.into_iter().collect::<Vec<_>>();
        let actual = actual_ids.iter().copied().collect::<HashSet<_>>();
        if actual.len() != actual_ids.len() {
            return Err("physical body arena repeats an entry identity".into());
        }
        if expected != actual {
            let mut missing = expected.difference(&actual).copied().collect::<Vec<_>>();
            let mut unowned = actual.difference(&expected).copied().collect::<Vec<_>>();
            missing.sort_unstable();
            unowned.sort_unstable();
            return Err(format!(
                "physical kernel/body ownership mismatch; missing bodies: {missing:?}; unowned bodies: {unowned:?}"
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct PhysicalKernel {
    pub id: KernelId,
    pub entry: EntryId,
    pub entry_point: String,
    pub label: String,
    pub source_entry: Option<EntryId>,
    pub output_routes: Vec<OutputRouteProjection>,
    pub workgroup_size: (u32, u32, u32),
    pub domain: KernelDomain,
    pub resources: Vec<SegResourceAccess<ResourceId>>,
    pub dependencies: Vec<KernelId>,
}

impl From<&KernelPlan> for PhysicalKernelGraph {
    fn from(plan: &KernelPlan) -> Self {
        Self {
            kernels: plan
                .topology
                .kernel_order()
                .iter()
                .map(|&id| {
                    let phase = &plan.catalog[&id];
                    PhysicalKernel {
                        id,
                        entry: phase.entry.id,
                        entry_point: phase.entry_point().to_owned(),
                        label: phase.label.to_owned(),
                        source_entry: phase.source_entry,
                        output_routes: phase.output_routes.clone(),
                        workgroup_size: phase.workgroup_size(),
                        domain: phase.dispatch.domain.clone(),
                        resources: phase.resources.clone(),
                        dependencies: plan
                            .topology
                            .kernel(id)
                            .unwrap_or_else(|| unreachable!("finalized kernel"))
                            .dependencies()
                            .to_vec(),
                    }
                })
                .collect(),
        }
    }
}

/// Concrete host-visible launch domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelDomain {
    /// Exactly the workgroup count recorded here.
    Fixed {
        x: u32,
        y: u32,
        z: u32,
    },
    /// One logical invocation per element of a concrete length source.
    Elements(DispatchLen),
    /// One invocation per element of a logical storage resource. The
    /// descriptor binding is resolved only while publishing a validated plan.
    ResourceElements {
        resource: ResourceId,
        elem_bytes: u32,
    },
}

/// A launch domain and whether the host explicitly fixed it. Recipe-provided
/// domains replace inferred descriptor placeholders but never explicit host
/// dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct KernelDispatch {
    domain: KernelDomain,
    explicit: bool,
}

impl KernelDispatch {
    fn required_elements(&self) -> Option<u32> {
        match &self.domain {
            KernelDomain::Elements(DispatchLen::Fixed { count }) => Some(*count),
            _ => None,
        }
    }
    pub(super) fn inferred(baseline: KernelDomain) -> Self {
        Self {
            domain: baseline,
            explicit: false,
        }
    }

    pub(super) fn explicit(domain: KernelDomain) -> Self {
        Self {
            domain,
            explicit: true,
        }
    }
}

impl KernelPlan {
    fn phases(&self) -> impl Iterator<Item = &PreparedKernel> {
        self.topology.kernel_order().iter().map(|id| &self.catalog[id])
    }
    fn phase(&self, id: KernelId) -> &PreparedKernel {
        &self.catalog[&id]
    }
    fn phase_ids_in(&self, group: PipelineId) -> &[KernelId] {
        self.topology
            .groups()
            .iter()
            .find(|candidate| candidate.id() == group)
            .map_or(&[], |group| group.kernels())
    }
}

impl ScheduleBuilder {
    pub(super) fn from_descriptor(
        descriptor: &PipelineDescriptor,
        stage_entries: &[Vec<EntryId>],
        resources: &LogicalResourceArena,
        stages: &StagedProgram,
    ) -> Result<Self, KernelMutationError> {
        if stage_entries.len() != descriptor.pipelines.len() {
            return Err(
                "descriptor pipeline stages are missing structural entry associations".to_string().into(),
            );
        }
        let authored = stages
            .stages()
            .filter(|(_, stage)| matches!(stage.origin(), StageOrigin::Authored))
            .collect::<Vec<_>>();
        let entries = authored.iter().map(|(id, stage)| (stage.body().id, *id)).collect::<HashMap<_, _>>();
        let mut topology = TopologyBuilder::default();
        let mut pipelines = BTreeMap::new();
        let mut graphics_associations = BTreeMap::new();
        let mut associations = HashMap::new();
        let mut dispatch = HashMap::new();
        let mut sequencing = Vec::new();
        for (index, (pipeline, associated)) in descriptor.pipelines.iter().zip(stage_entries).enumerate() {
            let id = PipelineId(index as u32);
            topology.register_group(id, index as u64, matches!(pipeline, Pipeline::Compute(_)))?;
            pipelines.insert(id, pipeline.clone());
            if matches!(pipeline, Pipeline::Graphics(_)) {
                graphics_associations.insert(id, associated.clone());
            }
            let count = match pipeline {
                Pipeline::Compute(p) => p.stages.len(),
                Pipeline::Graphics(p) => p.stages.len(),
            };
            if count != associated.len() {
                return Err("pipeline stage association count differs from descriptor".to_string().into());
            }
            let mut previous = None;
            for (position, entry) in associated.iter().enumerate() {
                let stage = *entries
                    .get(entry)
                    .ok_or_else(|| format!("descriptor stage has unknown semantic entry {entry:?}"))?;
                if associations.insert(stage, id).is_some() {
                    return Err(
                        format!("semantic entry {entry:?} is assigned to multiple pipelines").into(),
                    );
                }
                if let Pipeline::Compute(compute) = pipeline {
                    dispatch.insert(
                        stage,
                        domain_selection_from_stage(&compute.stages[position], resources)?,
                    );
                    if let Some(before) = previous {
                        sequencing.push((before, stage));
                    }
                    previous = Some(stage);
                }
            }
        }
        let ordered = stages.topological_stages();
        let ranks = ordered
            .iter()
            .enumerate()
            .map(|(rank, &stage)| (stage, rank as u64))
            .collect::<HashMap<_, _>>();
        // Attribution follows the earliest downstream consumer in the staged
        // DAG. It has no dependency on the order recipes happen to be built.
        fn attributed_source(
            stage: StageId,
            stages: &StagedProgram,
            ranks: &HashMap<StageId, u64>,
        ) -> Result<StageId, String> {
            let body = stages.stage(stage).ok_or_else(|| format!("missing staged body {stage:?}"))?;
            if matches!(body.origin(), StageOrigin::Authored) {
                return Ok(stage);
            }
            let consumer = body
                .outgoing_flows()
                .iter()
                .filter_map(|flow| stages.flow(*flow))
                .flat_map(|flow| flow.consumers())
                .copied()
                .min_by_key(|id| ranks[id])
                .ok_or_else(|| format!("generated stage {stage:?} has no downstream consumer"))?;
            attributed_source(consumer, stages, ranks)
        }
        let mut metadata = BTreeMap::new();
        let mut kernel_ids = IdSource::new();
        for &stage in &ordered {
            let staged = stages.stage(stage).ok_or_else(|| format!("missing staged body {stage:?}"))?;
            let source = attributed_source(stage, stages, &ranks)?;
            let source_body =
                stages.stage(source).ok_or_else(|| format!("missing attributed stage {source:?}"))?.body();
            let mut group = associations.get(&source).copied();
            let generated = matches!(staged.origin(), StageOrigin::Generated { .. });
            // Generated publication starts independently of source attribution.
            // Surviving topology edges decide which compute groups coalesce.
            if generated && group.is_some() {
                let id = PipelineId(pipelines.len() as u32);
                topology.register_group(id, id.0 as u64, true)?;
                pipelines.insert(
                    id,
                    Pipeline::Compute(ComputePipeline {
                        bindings: Vec::new(),
                        stages: Vec::new(),
                        default_total_threads: None,
                    }),
                );
                group = Some(id);
            }
            topology.register_stage(stage, group, ranks[&stage])?;
            let entry = staged.body();
            let selection = if generated {
                KernelDispatch::explicit(
                    staged.origin().space().and_then(domain_from_space).unwrap_or(KernelDomain::Fixed {
                        x: 1,
                        y: 1,
                        z: 1,
                    }),
                )
            } else if entry.execution_model.is_compute() {
                let mut selection = dispatch
                    .remove(&stage)
                    .unwrap_or_else(|| KernelDispatch::inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 }));
                if !selection.explicit {
                    if let Some(domain) = storage_image_domain_inputs(&entry.inputs, &selection.domain) {
                        selection.domain = domain;
                    }
                }
                selection
            } else {
                KernelDispatch::inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 })
            };
            metadata.insert(
                stage,
                StageMetadata {
                    primary: kernel_ids.next_id(),
                    compute: entry.execution_model.is_compute(),
                    source_entry: Some(source_body.id),
                    required_elements: selection.required_elements(),
                    dispatch: selection,
                    output_routes: if generated { Vec::new() } else { output_projection(entry) },
                    generated_kind: staged.origin().generated_kind(),
                },
            );
        }
        for (before, after) in sequencing {
            topology.sequence_stages(before, after)?;
        }
        let source_entries = authored
            .iter()
            .map(|(_, stage)| (stage.body().id, PlannedPublication::from_semantic(stage.body())))
            .collect();
        Ok(Self {
            topology,
            catalog: BTreeMap::new(),
            stages: metadata,
            pipelines,
            graphics_associations,
            source_entries,
            kernel_ids,
        })
    }

    pub(super) fn primary_kernel(&self, stage: StageId) -> KernelId {
        self.stages[&stage].primary
    }
    pub(super) fn stage_metadata(&self, stage: StageId) -> &StageMetadata {
        &self.stages[&stage]
    }
    pub(super) fn allocate_kernel(&mut self) -> KernelId {
        self.kernel_ids.next_id()
    }
    pub(super) fn set_required_elements(&mut self, stage: StageId, count: Option<u32>) {
        if let Some(metadata) = self.stages.get_mut(&stage) {
            metadata.required_elements = count;
        }
    }
    pub(super) fn install_stage(
        &mut self,
        stage: StageId,
        mut recipe: PreparedRecipe,
    ) -> Result<(), KernelMutationError> {
        let metadata = self
            .stages
            .get(&stage)
            .ok_or_else(|| format!("stage {stage:?} is not awaiting installation"))?;
        if recipe.primary != metadata.primary {
            return Err("stage recipe changed its primary kernel identity".to_string().into());
        }
        for (&id, body) in &mut recipe.bodies {
            if body.entry.execution_model.is_compute() != metadata.compute {
                return Err("recipe execution model differs from its registered stage".to_string().into());
            }
            if self.catalog.contains_key(&id) {
                return Err(wyn_kernel_graph::Error::DuplicateKernel(id).into());
            }
            body.source_entry = metadata.source_entry;
            if id == metadata.primary {
                if metadata.dispatch.explicit {
                    body.dispatch = metadata.dispatch.clone();
                }
                if !body.projected {
                    body.output_routes = metadata.output_routes.clone();
                }
                // Generated stages retain their semantic materialization
                // coverage policy. Scalar prepasses have no element grid.
                if metadata.generated_kind.is_some() {
                    body.required_elements = metadata.required_elements;
                }
            }
        }
        validate_routes(&recipe.bodies, &self.source_entries)?;
        self.topology.bind_stage(stage, recipe.fragment)?;
        // No fallible operation follows the checked topology installation.
        self.stages.remove(&stage);
        self.catalog.extend(recipe.bodies);
        Ok(())
    }

    pub(super) fn finish(
        mut self,
        stages: &StagedProgram,
        serial: bool,
    ) -> Result<KernelPlan, KernelMutationError> {
        // All stages are bound before projecting resident flows onto kernels.
        let mut readers = HashMap::<_, Vec<_>>::new();
        let mut writers = HashMap::<_, Vec<_>>::new();
        for (&id, kernel) in &self.catalog {
            let stage = self.topology.stage_of(id).unwrap_or_else(|| unreachable!("bound kernel"));
            for access in &kernel.resources {
                for (index, includes) in [
                    (&mut readers, access.access.reads()),
                    (&mut writers, access.access.writes()),
                ] {
                    if includes {
                        let kernels = index.entry((stage, access.resource)).or_default();
                        // Catalog order preserves deterministic edges; repeated
                        // declarations still identify each kernel only once.
                        if kernels.last() != Some(&id) {
                            kernels.push(id);
                        }
                    }
                }
            }
        }
        for (_, flow) in stages.flows() {
            let producer = flow.producer();
            for &consumer in flow.consumers() {
                let mut surviving = false;
                for resource in [Some(flow.storage().data), flow.storage().length].into_iter().flatten() {
                    let Some(readers) = readers.get(&(consumer, resource)) else {
                        continue;
                    };
                    surviving = true;
                    let Some(writers) = writers.get(&(producer, resource)) else {
                        return Err(format!(
                            "flow producer {producer:?} does not declare a writer for {resource:?}"
                        )
                        .into());
                    };
                    for &reader in readers {
                        for &writer in writers {
                            if writer != reader {
                                self.topology.connect_resource(resource, writer, reader)?;
                            }
                        }
                    }
                }
                if surviving
                    && stages
                        .stage(producer)
                        .is_some_and(|stage| matches!(stage.origin(), StageOrigin::Generated { .. }))
                {
                    self.topology.sequence_stages(producer, consumer)?;
                }
            }
        }
        if serial {
            for body in self.catalog.values_mut().filter(|body| body.entry.execution_model.is_compute()) {
                super::prepare::force_serial(&mut body.entry.graph);
                if body.serial_single_workgroup {
                    body.dispatch = KernelDispatch::explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 });
                } else {
                    body.label = "serial_compute";
                }
            }
        }
        let topology = self.topology.finalize()?;
        let mut pipelines = Vec::new();
        for group in topology.groups() {
            let mut templates = group
                .members()
                .iter()
                .map(|id| self.pipelines.remove(id).unwrap_or_else(|| unreachable!("registered group")));
            let mut template = templates.next().unwrap_or_else(|| unreachable!("nonempty group"));
            for member in templates {
                match (&mut template, member) {
                    (Pipeline::Compute(target), Pipeline::Compute(source)) => {
                        merge_bindings(&mut target.bindings, source.bindings)
                    }
                    _ => return Err("graphics pipelines cannot be coalesced".to_string().into()),
                }
            }
            let graphics_entries = self.graphics_associations.remove(&group.id()).unwrap_or_default();
            pipelines.push(ScheduledPipeline {
                id: group.id(),
                template,
                graphics_entries,
            });
        }
        let plan = KernelPlan {
            topology,
            catalog: self.catalog,
            pipelines,
            source_entries: self.source_entries,
        };
        plan.validate()?;
        Ok(plan)
    }
}

fn validate_routes(
    bodies: &BTreeMap<KernelId, PreparedKernel>,
    sources: &BTreeMap<EntryId, PlannedPublication>,
) -> Result<(), String> {
    let mut owned = HashSet::new();
    for (&id, body) in bodies {
        let Some(source) = body.source_entry else {
            continue;
        };
        let source_plan = sources
            .get(&source)
            .ok_or_else(|| format!("kernel {id:?} references missing semantic entry {source:?}"))?;
        for route in &body.output_routes {
            if route.semantic_slot.0 >= source_plan.outputs.len()
                || !owned.insert((source, route.semantic_slot))
            {
                return Err(format!(
                    "kernel {id:?} has invalid or duplicate output slot {:?}",
                    route.semantic_slot
                ));
            }
            // Physical slots name the projected publication ABI. Collective
            // recipes may move its stores into helpers and retire body outputs.
        }
    }
    Ok(())
}

fn output_projection(entry: &egir::program::AllocatedEntry) -> Vec<OutputRouteProjection> {
    entry
        .outputs
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.routes.is_empty())
        .map(|(slot, _)| OutputSlotId(slot))
        .enumerate()
        .map(|(physical, semantic_slot)| OutputRouteProjection {
            semantic_slot,
            physical_slot: OutputSlotId(physical),
        })
        .collect()
}

/// A compute entry with no SOAC-derived domain and a `#[storage_image]` param
/// runs one thread per texel of the image (the mountains / one_weekend
/// per-pixel pass shape) — the host resolves the size from the bound texture's
/// extent. Incidental storage-buffer inputs (e.g. mountains' keyboard buffer)
/// don't opt out; the image is the domain. Only upgrades the single-workgroup
/// placeholder domain; an explicit fixed grid stays as scheduled.
///
fn storage_image_domain_inputs(
    inputs: &[egir::ir::EntryInput<SemanticResourceRef, egir::types::WynLanguage>],
    baseline: &KernelDomain,
) -> Option<KernelDomain> {
    if !matches!(baseline, KernelDomain::Fixed { x: 1, y: 1, z: 1 }) {
        return None;
    }
    let (binding, ..) = inputs.iter().find_map(|input| input.storage_image_binding())?;
    Some(KernelDomain::Elements(DispatchLen::StorageImage {
        set: binding.set,
        binding: binding.binding,
    }))
}

fn execution_workgroup(model: &ExecutionModel) -> (u32, u32, u32) {
    match model {
        ExecutionModel::Compute { local_size } => *local_size,
        _ => (1, 1, 1),
    }
}

fn domain_from_dispatch(
    dispatch: &DispatchSize,
    resources: &LogicalResourceArena,
) -> Result<KernelDomain, String> {
    Ok(match dispatch {
        DispatchSize::Fixed { x, y, z, .. } => KernelDomain::Fixed { x: *x, y: *y, z: *z },
        DispatchSize::DerivedFrom {
            len:
                DispatchLen::InputBinding {
                    set,
                    binding,
                    elem_bytes,
                },
            ..
        } => KernelDomain::ResourceElements {
            resource: resources.host_resource(BindingRef::new(*set, *binding)).ok_or_else(|| {
                format!("descriptor dispatch binding {set}:{binding} is absent from the resource manifest")
            })?,
            elem_bytes: *elem_bytes,
        },
        DispatchSize::DerivedFrom { len, .. } => KernelDomain::Elements(len.clone()),
    })
}

fn domain_selection_from_stage(
    stage: &ComputeStage,
    resources: &LogicalResourceArena,
) -> Result<KernelDispatch, String> {
    let domain = domain_from_dispatch(&stage.dispatch_size, resources)?;
    Ok(match stage.dispatch_size {
        // Honor the source's explicit intent: a user-pinned `#[dispatch]` grid
        // (including `1x1x1`) stays `Explicit` and is never re-inferred. Only
        // the unpinned default `1x1x1` placeholder is `Inferred`.
        DispatchSize::Fixed { explicit: true, .. } => KernelDispatch::explicit(domain),
        DispatchSize::Fixed { explicit: false, .. } => KernelDispatch::inferred(domain),
        DispatchSize::DerivedFrom { .. } => KernelDispatch::inferred(domain),
    })
}

pub(super) fn domain_from_space(
    space: &egir::types::SegSpace<SemanticResourceRef>,
) -> Option<KernelDomain> {
    if space.dims().iter().all(|extent| matches!(extent, SegExtent::Fixed(_))) {
        let count = space.dims().iter().try_fold(1u32, |product, extent| match extent {
            SegExtent::Fixed(n) => product.checked_mul(*n),
            _ => None,
        })?;
        return Some(KernelDomain::Elements(DispatchLen::Fixed { count }));
    }
    match space.dims() {
        [SegExtent::PushConstant { offset, .. }] => {
            Some(KernelDomain::Elements(DispatchLen::PushConstant {
                offset: *offset,
            }))
        }
        [SegExtent::ResourceLength {
            resource, elem_bytes, ..
        }] => Some(KernelDomain::ResourceElements {
            resource: resource.0,
            elem_bytes: *elem_bytes,
        }),
        _ => None,
    }
}

fn merge_bindings(target: &mut Vec<Binding>, source: Vec<Binding>) {
    for binding in source {
        if !target.iter().any(|existing| same_binding_slot(existing, &binding)) {
            target.push(binding);
        }
    }
}

fn same_binding_slot(left: &Binding, right: &Binding) -> bool {
    if std::mem::discriminant(left) == std::mem::discriminant(right)
        && left.slot().is_some_and(|slot| right.slot() == Some(slot))
    {
        return true;
    }
    match (left, right) {
        (
            Binding::PushConstant {
                offset: left_offset,
                size: left_size,
                ..
            },
            Binding::PushConstant {
                offset: right_offset,
                size: right_size,
                ..
            },
        ) => left_offset == right_offset && left_size == right_size,
        _ => false,
    }
}
