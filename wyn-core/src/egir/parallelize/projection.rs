//! Physical-entry projection, interface compaction, and output-domain split.

use super::*;
use crate::egir::soac::filter as filter_soac;
use crate::egir::{graph_projector, program, semantic_graph};

pub(super) struct UnionFind {
    parents: Vec<usize>,
}

impl UnionFind {
    pub(super) fn new(len: usize) -> Self {
        Self {
            parents: (0..len).collect(),
        }
    }

    pub(super) fn find(&mut self, index: usize) -> usize {
        if self.parents[index] != index {
            self.parents[index] = self.find(self.parents[index]);
        }
        self.parents[index]
    }

    pub(super) fn union(&mut self, left: usize, right: usize) {
        let (left, right) = (self.find(left), self.find(right));
        if left != right {
            self.parents[right] = left;
        }
    }
}

/// Retained-interface facts owned by one projection result. Interface
/// compaction consumes them immediately; graph-local ids never outlive the
/// projected entry they describe.
struct ProjectionMetadata {
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

    fn selected_outputs(
        source: &program::PlannedEntry,
        name: String,
        outputs: Vec<crate::interface::EntryOutput>,
        output_routes: Vec<program::OutputRoute>,
    ) -> Self {
        Self {
            name,
            execution_model: source.execution_model.clone(),
            outputs,
            output_routes,
            resource_declarations: source.resource_declarations.clone(),
            return_ty: source.return_ty.clone(),
        }
    }
}

fn projection_metadata(
    source: &program::PlannedEntry,
    projection: &graph_projector::GraphProjection,
    output_routes: &[program::OutputRoute],
) -> Result<ProjectionMetadata, String> {
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
    let reachable = graph_ops::execution_value_producer_closure(&projection.graph, roots).nodes;
    let mut parameters = reachable
        .iter()
        .filter_map(|node| match projection.graph.nodes.get(*node) {
            Some(ENode::FuncParam { index }) => Some(index).copied(),
            _ => None,
        })
        .collect::<crate::SortedSet<_>>();
    let mut resources = reachable
        .iter()
        .filter_map(|node| graph_ops::extract_storage_view_source(&projection.graph, *node))
        .map(|resource| resource.0)
        .collect::<HashSet<_>>();

    for (index, input) in source.inputs.iter().enumerate() {
        let Some(Type::Constructed(TypeName::Resource(resource), _)) = input.ty.array_buffer() else {
            continue;
        };
        if resources.contains(resource) {
            parameters.insert(index);
        }
    }

    for (_, block) in &projection.graph.skeleton.blocks {
        for effect in &block.side_effects {
            resources.extend(
                semantic_graph::read_resources(&projection.graph, effect)
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
    }

    Ok(ProjectionMetadata {
        parameters,
        resources,
    })
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
        .selected_with_values(selected.clone(), route_values)?;
    let metadata = projection_metadata(source, &projection, &output_routes)?;
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
    compact_projected_entry_interface(&mut entry, &metadata);
    Ok(entry)
}

fn compact_projected_entry_interface(entry: &mut program::PlannedEntry, metadata: &ProjectionMetadata) {
    entry.retain_parameter_indices(&metadata.parameters);
    entry.resource_declarations.retain(|declaration| {
        declaration.role != crate::interface::StorageRole::Input
            || metadata.resources.contains(&declaration.resource.0)
    });
}

/// Resolve side-effect ownership from explicit output routes.
pub(super) fn side_effect_output_slots_from_routes(
    routes: &[program::OutputRoute],
    effect: &SideEffect,
) -> Vec<usize> {
    if let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind {
        if let screma::SemanticState::Segmented { output_slots, .. } = op.semantic_state() {
            return output_slots.iter().map(|slot| slot.0).collect();
        }
    }
    let value_writer = effect.result.map(OutputWriter::Value);
    let effect_writer = effect.effects.map(|(_, output)| OutputWriter::Effect(output));
    let mut slots = routes
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

fn side_effect_output_slots(entry: &program::PlannedEntry, effect: &SideEffect) -> Vec<usize> {
    side_effect_output_slots_from_routes(&entry.output_routes, effect)
}

fn is_parallel_output(effect: &SideEffect) -> bool {
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

fn is_write_effectful(effect: &SideEffect) -> bool {
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
}

/// Split a multi-output entry into one projected kernel per output domain.
/// Output routes remain the authority for effect ownership, and projection
/// metadata carries semantic-operation ownership into the resulting entries.
pub(super) fn split_multidomain_seg_maps(
    entry: &program::PlannedEntry,
) -> Result<Option<EntrySplit>, String> {
    let output_count = entry.outputs.len();
    if output_count <= 1 {
        return Ok(None);
    }

    let mut union = UnionFind::new(output_count);
    let mut producers = Vec::<((BlockId, usize), Vec<usize>, bool)>::new();
    for (block_id, block) in &entry.graph.skeleton.blocks {
        for (index, effect) in block.side_effects.iter().enumerate() {
            let slots = side_effect_output_slots(entry, effect);
            if slots.is_empty() {
                continue;
            }
            if slots.iter().any(|slot| *slot >= output_count) {
                return Err(format!(
                    "entry `{}` has an output route outside its interface",
                    entry.name
                ));
            }
            let first = slots[0];
            for slot in slots.iter().copied().skip(1) {
                union.union(first, slot);
            }
            producers.push(((block_id, index), slots, is_parallel_output(effect)));
        }
    }

    let mut group_of = LookupMap::<(BlockId, usize), usize>::new();
    let mut group_slots = LookupMap::<usize, Vec<usize>>::new();
    let mut parallel_groups = HashSet::<usize>::new();
    for (location, slots, is_parallel) in &producers {
        let Some(first) = slots.first().copied() else {
            continue;
        };
        let root = union.find(first);
        group_of.insert(*location, root);
        let entry_slots = group_slots.entry(root).or_default();
        for slot in slots {
            if !entry_slots.contains(slot) {
                entry_slots.push(*slot);
            }
        }
        if *is_parallel {
            parallel_groups.insert(root);
        }
    }
    if parallel_groups.len() < 2 {
        return Ok(None);
    }

    let mut groups = group_slots
        .into_iter()
        .filter_map(|(root, mut slots)| {
            slots.sort_unstable();
            slots.first().copied().map(|first_slot| OutputGroup {
                root,
                slots,
                first_slot,
            })
        })
        .collect::<Vec<_>>();
    groups.sort_by_key(|group| group.first_slot);

    let host_root = groups
        .iter()
        .map(|group| group.root)
        .find(|root| parallel_groups.contains(root))
        .ok_or_else(|| "multi-domain split has no parallel host group".to_string())?;
    for (block_id, block) in &entry.graph.skeleton.blocks {
        for (index, effect) in block.side_effects.iter().enumerate() {
            if side_effect_output_slots(entry, effect).is_empty() && is_write_effectful(effect) {
                group_of.insert((block_id, index), host_root);
            }
        }
    }

    let base_name = entry.name.clone();
    let project_group = |root: usize, slots: &[usize], name: String| -> Result<_, String> {
        let selected = group_of
            .iter()
            .filter_map(|(&(block, index), &group)| {
                (group == root).then_some(SideEffectSite { block, index })
            })
            .collect::<HashSet<_>>();
        let outputs = slots
            .iter()
            .map(|slot| {
                entry
                    .outputs
                    .get(*slot)
                    .map(|output| output.inner.clone())
                    .ok_or_else(|| format!("split output slot {slot} is out of bounds"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut routes = entry
            .output_routes
            .iter()
            .filter(|route| slots.contains(&route.slot.0))
            .cloned()
            .collect::<Vec<_>>();
        for route in &mut routes {
            let projected_slot = slots
                .iter()
                .position(|slot| *slot == route.slot.0)
                .ok_or_else(|| format!("selected route {} is absent from split group", route.slot.0))?;
            route.slot = program::OutputSlotId(projected_slot);
        }
        let spec = ProjectionSpec::selected_outputs(entry, name, outputs, routes);
        project_kernel_body_effects(entry, selected, spec)
    };

    let Some((primary_group, sibling_groups)) = groups.split_first() else {
        return Ok(None);
    };
    let mut siblings = Vec::new();
    for group in sibling_groups {
        let projected = project_group(
            group.root,
            &group.slots,
            format!("{base_name}_dispatch_{}", group.first_slot),
        )?;
        siblings.push(SplitEntry {
            entry: projected,
            semantic_slots: group.slots.clone(),
        });
    }

    let primary = SplitEntry {
        entry: project_group(primary_group.root, &primary_group.slots, entry.name.clone())?,
        semantic_slots: primary_group.slots.clone(),
    };
    Ok(Some(EntrySplit { primary, siblings }))
}
