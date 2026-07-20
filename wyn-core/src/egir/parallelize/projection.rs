//! Physical-entry projection, interface compaction, and output-domain split.

use super::*;
use crate::egir::soac::filter as filter_soac;
use crate::egir::{graph_projector, program, semantic_graph};

/// Retained-interface facts owned by one projection result. Interface
/// compaction consumes them immediately; graph-local ids never outlive the
/// projected entry they describe.
struct RetainedInterface {
    parameters: crate::SortedSet<usize>,
    resources: HashSet<ResourceId>,
}

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

fn reachable_projection_nodes(
    projection: &graph_projector::GraphProjection,
    output_routes: &[program::OutputRoute],
) -> HashSet<NodeId> {
    let mut roots = projection
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
    for route in output_routes {
        if let Some(value) = projection.node(route.source.value) {
            roots.push(value);
        }
        roots.extend(route.writers.iter().filter_map(|writer| match writer {
            OutputWriter::Value(value) => projection.node(*value),
            OutputWriter::Effect(_) => None,
        }));
    }

    graph_ops::execution_value_producer_closure(&projection.graph, roots).nodes
}

fn retained_resources(graph: &EGraph, reachable: &HashSet<NodeId>) -> HashSet<ResourceId> {
    let mut resources = reachable
        .iter()
        .filter_map(|node| graph_ops::extract_storage_view_source(graph, *node))
        .map(|resource| resource.0)
        .collect::<HashSet<_>>();

    for (_, block) in &graph.skeleton.blocks {
        for effect in &block.side_effects {
            resources.extend(
                semantic_graph::read_resources(graph, effect).into_iter().map(|access| access.resource.0),
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
    }

    resources
}

fn retained_parameters(
    source: &program::PlannedEntry,
    graph: &EGraph,
    reachable: &HashSet<NodeId>,
    resources: &HashSet<ResourceId>,
) -> crate::SortedSet<usize> {
    let mut parameters = reachable
        .iter()
        .filter_map(|node| match graph.nodes.get(*node) {
            Some(ENode::FuncParam { index }) => Some(index).copied(),
            _ => None,
        })
        .collect::<crate::SortedSet<_>>();

    for (index, input) in source.inputs.iter().enumerate() {
        let Some(Type::Constructed(TypeName::Resource(resource), _)) = input.ty.array_buffer() else {
            continue;
        };
        if resources.contains(resource) {
            parameters.insert(index);
        }
    }

    parameters
}

fn retained_interface(
    source: &program::PlannedEntry,
    projection: &graph_projector::GraphProjection,
    output_routes: &[program::OutputRoute],
) -> RetainedInterface {
    let reachable = reachable_projection_nodes(projection, output_routes);
    let resources = retained_resources(&projection.graph, &reachable);
    let parameters = retained_parameters(source, &projection.graph, &reachable, &resources);

    RetainedInterface {
        parameters,
        resources,
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
        .selected_with_values(selected, route_values)?;
    let retained = retained_interface(source, &projection, &output_routes);
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
    entry.retain_parameter_indices(&retained.parameters);
    entry.resource_declarations.retain(|declaration| {
        declaration.role != crate::interface::StorageRole::Input
            || retained.resources.contains(&declaration.resource.0)
    });
    Ok(entry)
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

fn starts_parallel_output_domain(effect: &SideEffect) -> bool {
    matches!(
        &effect.kind,
        SideEffectKind::Soac(SoacEffect(
            _,
            Soac::Screma(screma::Op::Map {
                state: screma::SemanticState::Segmented { .. },
                ..
            })
        ))
    ) || matches!(
        &effect.kind,
        SideEffectKind::Soac(SoacEffect(
            _,
            Soac::Filter(filter_soac::Op {
                state: filter_soac::SemanticState {
                    storage: filter_soac::Output::Runtime { .. },
                    ..
                },
                ..
            })
        ))
    )
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
}

pub(super) struct EntrySplit {
    pub(super) primary: SplitEntry,
    pub(super) siblings: Vec<SplitEntry>,
}

struct OutputGroup {
    root: usize,
    slots: Vec<usize>,
    first_slot: usize,
    starts_parallel_domain: bool,
}

impl OutputGroup {
    fn new(root: usize, first_slot: usize) -> Self {
        Self {
            root,
            slots: Vec::new(),
            first_slot,
            starts_parallel_domain: false,
        }
    }

    fn include(&mut self, producer: OutputProducer) {
        self.first_slot = self.first_slot.min(producer.first_slot);
        self.starts_parallel_domain |= producer.starts_parallel_domain;
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
    starts_parallel_domain: bool,
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

        let unrouted_host = {
            let mut parallel_groups = groups.iter().filter(|group| group.starts_parallel_domain);
            let host = parallel_groups.next()?.root;
            parallel_groups.next()?;
            host
        };
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
                starts_parallel_domain: starts_parallel_output_domain(effect),
            });
        }
    }
    Ok(producers)
}

fn output_partition(entry: &program::PlannedEntry) -> Result<Option<OutputPartition>, String> {
    if entry.outputs.len() <= 1 {
        return Ok(None);
    }

    let Some(mut partition) =
        OutputPartition::from_producers(entry.outputs.len(), output_producers(entry)?)
    else {
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
    let spec = ProjectionSpec {
        name,
        execution_model: entry.execution_model.clone(),
        outputs,
        output_routes,
        resource_declarations: entry.resource_declarations.clone(),
        return_ty: entry.return_ty.clone(),
    };

    Ok(SplitEntry {
        entry: project_kernel_body_effects(entry, selected, spec)?,
        semantic_slots: group.slots.clone(),
    })
}

/// Split a multi-output entry into one projected kernel per output domain.
/// Output routes remain the authority for effect ownership, and retained
/// interface facts carry semantic-operation ownership into each projection.
pub(super) fn split_multidomain_seg_maps(
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
