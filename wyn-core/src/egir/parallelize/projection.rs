//! Physical-entry projection, interface compaction, and output-domain split.

use super::*;
use crate::egir::{graph_projector, program};

pub(super) struct ProjectionSpec {
    name: String,
    execution_model: ExecutionModel,
    outputs: Vec<crate::interface::EntryOutput>,
    output_routes: Vec<program::OutputRoute>,
    resource_declarations: Vec<SemanticResourceDecl>,
    return_ty: Type<TypeName>,
}

impl ProjectionSpec {
    pub(super) fn unit(
        name: String,
        execution_model: ExecutionModel,
        resource_declarations: Vec<SemanticResourceDecl>,
    ) -> Self {
        Self {
            name,
            execution_model,
            outputs: Vec::new(),
            output_routes: Vec::new(),
            resource_declarations,
            return_ty: Type::Constructed(TypeName::Unit, vec![]),
        }
    }

    pub(super) fn preserving_interface(
        source: &program::PlannedEntry,
        resource_declarations: Vec<SemanticResourceDecl>,
    ) -> Self {
        Self {
            name: source.name.clone(),
            execution_model: source.execution_model.clone(),
            outputs: source.outputs.iter().map(|output| output.inner.clone()).collect(),
            output_routes: source.output_routes.clone(),
            resource_declarations,
            return_ty: source.return_ty.clone(),
        }
    }
}

pub(super) fn project_kernel_body(
    source: &program::PlannedEntry,
    spec: ProjectionSpec,
) -> Result<program::PlannedEntry, String> {
    let ProjectionSpec {
        name,
        execution_model,
        outputs,
        output_routes,
        resource_declarations,
        return_ty,
    } = spec;
    let route_values = output_routes.iter().map(|route| route.source.value).collect();
    let projection = graph_projector::GraphProjector::new(&source.graph, &source.control_headers)
        .all_with_values(route_values)?;
    program::PlannedEntry::from_projection(
        projection,
        name,
        source.span,
        execution_model,
        source.inputs.iter().map(|input| input.inner.clone()).collect(),
        outputs,
        resource_declarations,
        source.params.clone(),
        return_ty,
        &source.aliases,
        output_routes,
    )
}

fn project_kernel_body_effects(
    source: &program::PlannedEntry,
    selected: HashSet<SideEffectSite>,
    spec: ProjectionSpec,
) -> Result<(program::PlannedEntry, LookupMap<SideEffectSite, SideEffectSite>), String> {
    let ProjectionSpec {
        name,
        execution_model,
        outputs,
        output_routes,
        resource_declarations,
        return_ty,
    } = spec;
    let route_values = output_routes.iter().map(|route| route.source.value).collect();
    let projection = graph_projector::GraphProjector::new(&source.graph, &source.control_headers)
        .selected_component_with_values(selected, route_values)?;
    let effect_sites = projection
        .source_effects()
        .iter()
        .filter_map(|source| Some((*source, projection.effect_site(*source)?)))
        .collect();
    let retained_resources = source.resources_referenced_by_projection(&projection);
    let retained_parameters =
        source.parameter_indices_referenced_by_projection(&projection, &retained_resources);
    let mut entry = program::PlannedEntry::from_projection(
        projection,
        name,
        source.span,
        execution_model,
        source.inputs.iter().map(|input| input.inner.clone()).collect(),
        outputs,
        resource_declarations,
        source.params.clone(),
        return_ty,
        &source.aliases,
        output_routes,
    )?;
    entry.retain_parameter_indices(&retained_parameters);
    entry.resource_declarations.retain(|declaration| match declaration.role {
        crate::interface::StorageRole::Input | crate::interface::StorageRole::Output => {
            retained_resources.contains(&declaration.resource.0)
        }
        crate::interface::StorageRole::Intermediate => true,
    });
    Ok((entry, effect_sites))
}

/// Resolve side-effect ownership from explicit output routes.
pub(super) fn side_effect_output_slots(entry: &program::PlannedEntry, effect: &SideEffect) -> Vec<usize> {
    if let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind {
        if let screma::SemanticState::Segmented { output_slots, .. } = op.semantic_state() {
            return output_slots.iter().map(|slot| slot.0).collect();
        }
    }
    let value_writer = effect.result.map(OutputWriter::Value);
    let effect_writer = effect.effects.map(|(_, output)| OutputWriter::Effect(output));
    let mut slots = entry
        .output_routes
        .iter()
        .filter(|route| {
            route
                .writers
                .iter()
                .any(|writer| Some(*writer) == value_writer || Some(*writer) == effect_writer)
        })
        .map(|route| route.slot.0)
        .collect::<Vec<_>>();
    slots.sort_unstable();
    slots.dedup();
    slots
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputExecutionDomain {
    Singleton,
    ScheduledParallel,
}

impl OutputExecutionDomain {
    fn join(self, other: Self) -> Self {
        if self == Self::ScheduledParallel || other == Self::ScheduledParallel {
            Self::ScheduledParallel
        } else {
            Self::Singleton
        }
    }

    fn is_scheduled_parallel(self) -> bool {
        self == Self::ScheduledParallel
    }
}

fn output_execution_domain(effect: &SideEffect) -> OutputExecutionDomain {
    match &effect.kind {
        SideEffectKind::Soac(SoacEffect(_, soac)) if soac.scheduling_space().is_some() => {
            OutputExecutionDomain::ScheduledParallel
        }
        _ => OutputExecutionDomain::Singleton,
    }
}

fn requires_unrouted_owner(effect: &SideEffect) -> bool {
    match &effect.kind {
        SideEffectKind::Soac(SoacEffect(_, Soac::Hist(_) | Soac::Filter(_))) => true,
        SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => op
            .lanes()
            .maps
            .iter()
            .map(|map| map.destination)
            .chain(op.operators().into_iter().map(|operator| operator.destination))
            .any(|destination| !destination.is_unplaced()),
        SideEffectKind::Effect(EffectOp::Store) => true,
        _ => false,
    }
}

pub(super) struct SplitEntry {
    pub(super) entry: program::PlannedEntry,
    pub(super) semantic_slots: Vec<usize>,
    pub(super) effect_sites: LookupMap<SideEffectSite, SideEffectSite>,
}

pub(super) struct EntrySplit {
    pub(super) primary: SplitEntry,
    pub(super) siblings: Vec<SplitEntry>,
}

struct OutputGroup {
    root: usize,
    slots: Vec<usize>,
    first_slot: usize,
    domain: OutputExecutionDomain,
}

impl OutputGroup {
    fn new(root: usize, first_slot: usize) -> Self {
        Self {
            root,
            slots: Vec::new(),
            first_slot,
            domain: OutputExecutionDomain::Singleton,
        }
    }

    fn include(&mut self, producer: OutputProducer) {
        self.first_slot = self.first_slot.min(producer.first_slot);
        self.domain = self.domain.join(producer.domain);
        for slot in producer.slots {
            if !self.slots.contains(&slot) {
                self.slots.push(slot);
            }
        }
    }
}

struct OutputProducer {
    site: SideEffectSite,
    slots: Vec<usize>,
    first_slot: usize,
    domain: OutputExecutionDomain,
}

struct OutputPartition {
    primary: OutputGroup,
    siblings: Vec<OutputGroup>,
    effect_groups: LookupMap<SideEffectSite, usize>,
    unrouted_host: usize,
}

impl OutputPartition {
    fn from_producers(output_count: usize, producers: Vec<OutputProducer>) -> Option<Self> {
        let mut output_domains = DisjointSets::new(output_count);
        for producer in &producers {
            for slot in producer.slots.iter().copied().skip(1) {
                output_domains.merge(producer.first_slot, slot);
            }
        }

        let mut effect_groups = LookupMap::new();
        let mut groups = LookupMap::<usize, OutputGroup>::new();
        for producer in producers {
            let root = output_domains.representative(producer.first_slot);
            effect_groups.insert(producer.site, root);
            groups
                .entry(root)
                .or_insert_with(|| OutputGroup::new(root, producer.first_slot))
                .include(producer);
        }

        let mut groups = groups.into_iter().map(|(_, group)| group).collect::<Vec<_>>();
        for group in &mut groups {
            group.slots.sort_unstable();
        }
        groups.sort_by_key(|group| group.first_slot);

        if groups.len() < 2 {
            return None;
        }
        let unrouted_host = groups.iter().find(|group| group.domain.is_scheduled_parallel())?.root;
        let mut groups = groups.into_iter();
        Some(Self {
            primary: groups.next()?,
            siblings: groups.collect(),
            effect_groups,
            unrouted_host,
        })
    }

    fn claim_unrouted_effects(&mut self, entry: &program::PlannedEntry) {
        for (block_id, block) in &entry.graph.skeleton.blocks {
            for (index, effect) in block.side_effects.iter().enumerate() {
                if side_effect_output_slots(entry, effect).is_empty() && requires_unrouted_owner(effect) {
                    self.effect_groups.insert(
                        SideEffectSite {
                            block: block_id,
                            index,
                        },
                        self.unrouted_host,
                    );
                }
            }
        }
    }
}

fn output_producers(entry: &program::PlannedEntry) -> Result<Vec<OutputProducer>, String> {
    let output_count = entry.outputs.len();
    let mut producers = Vec::new();
    for (block_id, block) in &entry.graph.skeleton.blocks {
        for (index, effect) in block.side_effects.iter().enumerate() {
            let slots = side_effect_output_slots(entry, effect);
            let Some(first_slot) = slots.first().copied() else {
                continue;
            };
            if slots.iter().any(|slot| *slot >= output_count) {
                return Err(format!(
                    "entry `{}` has an output route outside its interface",
                    entry.name
                ));
            }
            producers.push(OutputProducer {
                site: SideEffectSite {
                    block: block_id,
                    index,
                },
                slots,
                first_slot,
                domain: output_execution_domain(effect),
            });
        }
    }
    Ok(producers)
}

fn output_partition(entry: &program::PlannedEntry) -> Result<Option<OutputPartition>, String> {
    if entry.outputs.len() <= 1 {
        return Ok(None);
    }

    let producers = output_producers(entry)?;
    let Some(mut partition) = OutputPartition::from_producers(entry.outputs.len(), producers) else {
        return Ok(None);
    };
    partition.claim_unrouted_effects(entry);
    Ok(Some(partition))
}

fn project_output_group(
    entry: &program::PlannedEntry,
    group: &OutputGroup,
    effect_groups: &LookupMap<SideEffectSite, usize>,
    name: String,
) -> Result<SplitEntry, String> {
    let selected = effect_groups
        .iter()
        .filter_map(|(&site, &root)| (root == group.root).then_some(site))
        .collect::<HashSet<_>>();
    let outputs = group
        .slots
        .iter()
        .map(|slot| {
            entry
                .outputs
                .get(*slot)
                .map(|output| output.inner.clone())
                .ok_or_else(|| format!("split output slot {slot} is out of bounds"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let projected_slots = group
        .slots
        .iter()
        .enumerate()
        .map(|(projected, semantic)| (*semantic, projected))
        .collect::<LookupMap<_, _>>();
    let output_routes = entry
        .output_routes
        .iter()
        .filter_map(|route| {
            let projected_slot = projected_slots.get(&route.slot.0)?;
            let mut route = route.clone();
            route.slot = program::OutputSlotId(*projected_slot);
            Some(route)
        })
        .collect();
    let execution_model = match (&entry.execution_model, group.domain) {
        (ExecutionModel::Compute { .. }, OutputExecutionDomain::Singleton) => ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        _ => entry.execution_model.clone(),
    };
    let spec = ProjectionSpec {
        name,
        execution_model,
        outputs,
        output_routes,
        resource_declarations: entry.resource_declarations.clone(),
        return_ty: entry.return_ty.clone(),
    };

    let (entry, effect_sites) = project_kernel_body_effects(entry, selected, spec)?;
    Ok(SplitEntry {
        entry,
        semantic_slots: group.slots.clone(),
        effect_sites,
    })
}

/// Partition a multi-output entry into one projected kernel per independently
/// owned output component whenever at least one component has a scheduled
/// parallel domain. Singleton outputs therefore do not inherit a sibling
/// operation's per-invocation execution frequency. Output routes remain the
/// authority for effect ownership, and retained interface facts carry
/// semantic-operation ownership into each projection.
pub(super) fn partition_entry_output_domains(
    entry: &program::PlannedEntry,
) -> Result<Option<EntrySplit>, String> {
    let Some(partition) = output_partition(entry)? else {
        return Ok(None);
    };
    let OutputPartition {
        primary: primary_group,
        siblings: sibling_groups,
        effect_groups,
        ..
    } = partition;

    let primary = project_output_group(entry, &primary_group, &effect_groups, entry.name.clone())?;
    let siblings = sibling_groups
        .iter()
        .map(|group| {
            project_output_group(
                entry,
                group,
                &effect_groups,
                format!("{}_dispatch_{}", entry.name, group.first_slot),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(EntrySplit { primary, siblings }))
}
