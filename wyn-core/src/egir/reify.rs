//! Raw EGIR to semantic EGIR.
//!
//! This is the single boundary that constructs semantic SOAC state. The
//! operation family is preserved by the direct match in `reify_soac`.

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;

use polytype::Type;

use crate::ast::TypeName;
use crate::flow::{BlockId, ControlHeader, ExecutionModel};
use crate::types::TypeExt;
use crate::LookupMap;

use super::graph_ops;
use super::program::{
    ConstantDef, Entry, Func, OutputRoute, OutputSlotId, OutputWriter, Program, RawEntry, RawProgram,
    SemanticOpId, SemanticProgram, SemanticResourceRef,
};
use super::soac::{filter, hist, screma};
use super::types::{
    EGraph, ENode, NodeId, PureOp, Raw, ResourceAccess, SegExtent, SegLevel, SegResourceAccess, SegSpace,
    Semantic, SideEffect, SideEffectKind, Soac, SoacDestination, SoacInputType,
};

struct Facts {
    space: SegSpace<SemanticResourceRef>,
    placement: screma::Placement,
    output_slots: Vec<OutputSlotId>,
    resources: Vec<SegResourceAccess<SemanticResourceRef>>,
    entry: bool,
}

pub fn run(raw: RawProgram) -> SemanticProgram {
    let RawProgram { ir, resources } = raw;
    let Program {
        functions,
        externs,
        entry_points,
        constants,
        pipeline,
        input_names,
        regions,
        region_interner,
    } = ir;

    let mut next_semantic_id = 0;
    let entry_points =
        entry_points.into_iter().map(|entry| reify_entry(entry, &mut next_semantic_id)).collect();
    let functions =
        functions.into_iter().map(|function| reify_func(function, &mut next_semantic_id)).collect();
    let constants =
        constants.into_iter().map(|constant| reify_constant(constant, &mut next_semantic_id)).collect();

    let mut semantic = SemanticProgram {
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
        semantic_dependencies: Vec::new(),
    };
    super::semantic_graph::rebuild_dependencies(&mut semantic);
    semantic
}

fn reify_constant(constant: ConstantDef<Raw>, next_semantic_id: &mut u32) -> ConstantDef<Semantic> {
    let facts = function_facts(&constant.graph);
    let ConstantDef {
        name,
        span,
        return_ty,
        graph,
        control_headers,
        aliases,
    } = constant;
    let (graph, blocks) = map_graph(graph, facts, next_semantic_id);
    ConstantDef {
        name,
        span,
        return_ty,
        graph,
        control_headers: remap_headers(control_headers, &blocks),
        aliases,
    }
}

fn reify_func(function: Func<Raw>, next_semantic_id: &mut u32) -> Func<Semantic> {
    let facts = function_facts(&function.graph);
    let Func {
        name,
        span,
        linkage_name,
        params,
        return_ty,
        graph,
        control_headers,
        aliases,
    } = function;
    let (graph, blocks) = map_graph(graph, facts, next_semantic_id);
    Func {
        name,
        span,
        linkage_name,
        params,
        return_ty,
        graph,
        control_headers: remap_headers(control_headers, &blocks),
        aliases,
    }
}

fn reify_entry(entry: Entry<Raw>, next_semantic_id: &mut u32) -> Entry<Semantic> {
    let facts = entry_facts(&entry);
    let Entry {
        name,
        span,
        execution_model,
        inputs,
        outputs,
        resource_declarations,
        params,
        return_ty,
        graph,
        control_headers,
        aliases,
        output_routes,
    } = entry;
    let (graph, blocks) = map_graph(graph, facts, next_semantic_id);
    Entry {
        name,
        span,
        execution_model,
        inputs,
        outputs,
        resource_declarations,
        params,
        return_ty,
        graph,
        control_headers: remap_headers(control_headers, &blocks),
        aliases,
        output_routes: remap_routes(output_routes, &blocks),
    }
}

fn map_graph(
    graph: EGraph<Raw>,
    mut facts: HashMap<(BlockId, usize), Facts>,
    next_semantic_id: &mut u32,
) -> (EGraph<Semantic>, LookupMap<BlockId, BlockId>) {
    match graph.try_map_phase(|block, index, (), soac| {
        let facts = facts.remove(&(block, index)).expect("every raw SOAC must have semantic facts");
        let id = SemanticOpId(*next_semantic_id);
        *next_semantic_id += 1;
        Ok::<_, Infallible>((id, reify_soac(soac, facts)))
    }) {
        Ok(mapped) => mapped,
        Err(never) => match never {},
    }
}

fn reify_soac(soac: Soac<Raw>, facts: Facts) -> Soac<Semantic> {
    match soac {
        Soac::Screma(screma::Op::Map { lanes, .. }) => {
            let mut placement = facts.placement;
            if placement == screma::Placement::Kernel
                && (lanes.maps.is_empty()
                    || !lanes.maps.iter().all(|map| {
                        matches!(
                            map.destination,
                            SoacDestination::OutputView | SoacDestination::InputBuffer
                        )
                    }))
            {
                placement = screma::Placement::LaneLocal;
            }
            if placement == screma::Placement::Kernel && facts.output_slots.is_empty() {
                placement = screma::Placement::LaneLocal;
            }
            Soac::Screma(screma::Op::Map {
                lanes,
                state: screma::SemanticState::Segmented {
                    space: facts.space,
                    placement,
                    output_slots: facts.output_slots,
                    resources: facts.resources,
                },
            })
        }
        Soac::Screma(screma::Op::Reduce { lanes, operators, .. }) => Soac::Screma(screma::Op::Reduce {
            lanes,
            operators,
            state: screma::SemanticState::Segmented {
                space: facts.space,
                placement: facts.placement,
                output_slots: facts.output_slots,
                resources: facts.resources,
            },
        }),
        Soac::Screma(screma::Op::Scan { lanes, operators, .. }) => Soac::Screma(screma::Op::Scan {
            lanes,
            operators,
            state: screma::SemanticState::Segmented {
                space: facts.space,
                placement: facts.placement,
                output_slots: facts.output_slots,
                resources: facts.resources,
            },
        }),
        Soac::Screma(screma::Op::Composite { lanes, operators, .. }) => {
            Soac::Screma(screma::Op::Composite {
                lanes,
                operators,
                state: screma::SemanticState::Segmented {
                    space: facts.space,
                    placement: facts.placement,
                    output_slots: facts.output_slots,
                    resources: facts.resources,
                },
            })
        }
        Soac::Filter(op) => {
            let storage = match op.state.storage {
                filter::RawStorage::Local {
                    capacity,
                    destination,
                } => filter::Output::Local {
                    capacity,
                    destination,
                },
                filter::RawStorage::Runtime { scratch, length } => {
                    filter::Output::Runtime { scratch, length }
                }
            };
            Soac::Filter(filter::Op {
                body: op.body,
                state: filter::SemanticState {
                    space: facts.space,
                    storage,
                },
            })
        }
        Soac::Hist(op) => Soac::Hist(hist::Op {
            body: op.body,
            state: if facts.entry {
                hist::SemanticState::Segmented(facts.space)
            } else {
                hist::SemanticState::Serial
            },
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egir::types::EffectOp;
    use smallvec::SmallVec;

    fn raw_map() -> SideEffect<Raw> {
        SideEffect {
            kind: SideEffectKind::Soac(
                (),
                Soac::Screma(screma::Op::Map {
                    lanes: screma::Lanes {
                        inputs: vec![],
                        maps: vec![],
                    },
                    state: screma::RawState,
                }),
            ),
            operand_nodes: SmallVec::new(),
            result: None,
            effects: None,
            span: None,
        }
    }

    fn facts() -> Facts {
        Facts {
            space: SegSpace {
                level: SegLevel::Thread,
                dims: vec![],
            },
            placement: screma::Placement::LaneLocal,
            output_slots: vec![],
            resources: vec![],
            entry: false,
        }
    }

    #[test]
    fn phase_boundary_assigns_ids_to_soacs_but_not_instructions() {
        let mut graph = EGraph::<Raw>::new();
        let block = graph.skeleton.entry;
        graph.skeleton.blocks[block].side_effects.push(raw_map());
        graph.skeleton.blocks[block].side_effects.push(SideEffect {
            kind: SideEffectKind::Effect(EffectOp::ControlBarrier),
            operand_nodes: SmallVec::new(),
            result: None,
            effects: None,
            span: None,
        });
        graph.skeleton.blocks[block].side_effects.push(raw_map());

        let mut next = 7;
        let (graph, _) = map_graph(
            graph,
            HashMap::from([((block, 0), facts()), ((block, 2), facts())]),
            &mut next,
        );
        let ids: Vec<_> = graph.skeleton.blocks[graph.skeleton.entry]
            .side_effects
            .iter()
            .map(|effect| effect.kind.soac_id().copied())
            .collect();

        assert_eq!(ids, vec![Some(SemanticOpId(7)), None, Some(SemanticOpId(8))]);
        assert_eq!(next, 9);
    }
}

fn function_facts(graph: &EGraph<Raw>) -> HashMap<(BlockId, usize), Facts> {
    graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(block, contents)| {
            contents.side_effects.iter().enumerate().filter_map(move |(index, effect)| {
                semantic_facts(graph, None, effect, screma::Placement::LaneLocal)
                    .map(|facts| ((block, index), facts))
            })
        })
        .collect()
}

fn entry_facts(entry: &RawEntry) -> HashMap<(BlockId, usize), Facts> {
    let consumed = soac_consumed_nodes(&entry.graph);
    let kernel_scope = matches!(entry.execution_model, ExecutionModel::Compute { .. });
    let mut facts_by_location = HashMap::new();
    for (block, contents) in &entry.graph.skeleton.blocks {
        for (index, effect) in contents.side_effects.iter().enumerate() {
            let placement =
                if kernel_scope && !effect.result.is_some_and(|result| consumed.contains(&result)) {
                    screma::Placement::Kernel
                } else {
                    screma::Placement::LaneLocal
                };
            if let Some(facts) = semantic_facts(&entry.graph, Some(entry), effect, placement) {
                facts_by_location.insert((block, index), facts);
            }
        }
    }

    let kernel_accumulators = entry
        .graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(block, contents)| {
            contents.side_effects.iter().enumerate().map(move |(index, effect)| (block, index, effect))
        })
        .filter(|(block, index, effect)| {
            let SideEffectKind::Soac(_, Soac::Screma(op)) = &effect.kind else {
                return false;
            };
            matches!(op, screma::Op::Reduce { .. } | screma::Op::Scan { .. })
                && matches!(
                    facts_by_location.get(&(*block, *index)),
                    Some(Facts {
                        placement: screma::Placement::Kernel,
                        ..
                    })
                )
        })
        .map(|(block, index, _)| (block, index))
        .collect::<Vec<_>>();
    if kernel_accumulators.len() > 1 {
        for location in kernel_accumulators {
            if let Some(facts) = facts_by_location.get_mut(&location) {
                facts.placement = screma::Placement::LaneLocal;
            }
        }
    }
    facts_by_location
}

fn semantic_facts(
    graph: &EGraph<Raw>,
    entry: Option<&RawEntry>,
    effect: &SideEffect<Raw>,
    requested_placement: screma::Placement,
) -> Option<Facts> {
    let SideEffectKind::Soac(_, soac) = &effect.kind else {
        return None;
    };
    let (input, operand_index, is_screma) = match soac {
        Soac::Screma(op) => (op.lanes().inputs.first(), 0, true),
        Soac::Filter(op) => (Some(filter_input_type(&op.body.input)), 0, false),
        Soac::Hist(op) => (op.body.inputs.first(), 1, false),
    };
    let output_slots = if is_screma {
        entry.map_or_else(Vec::new, |entry| output_slots(entry, effect))
    } else {
        Vec::new()
    };
    let resources =
        if is_screma { semantic_resources(graph, entry, effect, &output_slots) } else { Vec::new() };
    Some(Facts {
        space: space(graph, entry, effect, input, operand_index),
        placement: requested_placement,
        output_slots,
        resources,
        entry: entry.is_some(),
    })
}

fn filter_input_type(input: &filter::Input) -> &SoacInputType {
    match input {
        filter::Input::Plain(input) | filter::Input::Mapped { input, .. } => input,
    }
}

fn space(
    graph: &EGraph<Raw>,
    entry: Option<&RawEntry>,
    effect: &SideEffect<Raw>,
    input: Option<&SoacInputType>,
    operand_index: usize,
) -> SegSpace<SemanticResourceRef> {
    let extent = effect.operand_nodes.get(operand_index).copied().map(|node| {
        if let Some(resource) = graph_ops::extract_storage_view_source(graph, node) {
            let elem_bytes = input
                .and_then(|input| crate::ssa::layout::type_byte_size(&input.element))
                .unwrap_or(1) as u32;
            return SegExtent::ResourceLength {
                node,
                resource,
                elem_bytes,
            };
        }
        if let Some((_, len, _)) = graph_ops::extract_array_range_operands(graph, node) {
            return extent_from_node(graph, entry, len);
        }
        if let Some(Type::Constructed(TypeName::Size(size), _)) =
            input.and_then(|input| input.array.array_size())
        {
            return SegExtent::Fixed(*size as u32);
        }
        SegExtent::Value(node)
    });
    SegSpace {
        level: SegLevel::Thread,
        dims: extent.into_iter().collect(),
    }
}

fn extent_from_node(
    graph: &EGraph<Raw>,
    entry: Option<&RawEntry>,
    node: NodeId,
) -> SegExtent<SemanticResourceRef> {
    match &graph.nodes[node] {
        ENode::Pure {
            op: PureOp::Int(value) | PureOp::Uint(value),
            ..
        } => value.parse().map(SegExtent::Fixed).unwrap_or(SegExtent::Value(node)),
        ENode::FuncParam { index } => entry
            .and_then(|entry| entry.inputs.get(*index))
            .and_then(|input| input.push_constant())
            .map(|slot| SegExtent::PushConstant {
                node,
                offset: slot.offset,
            })
            .unwrap_or(SegExtent::Value(node)),
        _ => SegExtent::Value(node),
    }
}

fn output_slots(entry: &RawEntry, effect: &SideEffect<Raw>) -> Vec<OutputSlotId> {
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
        .map(|route| route.slot)
        .collect::<Vec<_>>();
    slots.sort_unstable();
    slots.dedup();
    slots
}

fn semantic_resources(
    graph: &EGraph<Raw>,
    entry: Option<&RawEntry>,
    effect: &SideEffect<Raw>,
    output_slots: &[OutputSlotId],
) -> Vec<SegResourceAccess<SemanticResourceRef>> {
    let mut accesses = read_resources(graph, effect)
        .into_iter()
        .map(|resource| (resource.resource, resource.access))
        .collect::<HashMap<_, _>>();
    if let Some(entry) = entry {
        for slot in output_slots {
            if let Some(resource) = entry.outputs.get(slot.0).and_then(|output| output.resource) {
                accesses
                    .entry(resource)
                    .and_modify(|access| *access = ResourceAccess::ReadWrite)
                    .or_insert(ResourceAccess::Write);
            }
        }
    }
    let mut resources = accesses
        .into_iter()
        .map(|(resource, access)| SegResourceAccess { resource, access })
        .collect::<Vec<_>>();
    resources.sort_by_key(|resource| resource.resource);
    resources
}

fn read_resources(
    graph: &EGraph<Raw>,
    effect: &SideEffect<Raw>,
) -> Vec<SegResourceAccess<SemanticResourceRef>> {
    let resources = graph_ops::value_producer_closure(graph, referenced_nodes(effect))
        .nodes
        .into_iter()
        .filter_map(|node| graph_ops::extract_storage_view_source(graph, node))
        .collect::<HashSet<_>>();
    let mut resources = resources
        .into_iter()
        .map(|resource| SegResourceAccess {
            resource,
            access: ResourceAccess::Read,
        })
        .collect::<Vec<_>>();
    resources.sort_by_key(|resource| resource.resource);
    resources
}

fn soac_consumed_nodes(graph: &EGraph<Raw>) -> HashSet<NodeId> {
    let roots = graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, block)| &block.side_effects)
        .filter(|effect| matches!(effect.kind, SideEffectKind::Soac(_, _)))
        .flat_map(referenced_nodes)
        .collect::<Vec<_>>();
    graph_ops::value_producer_closure(graph, roots).nodes
}

fn referenced_nodes(effect: &SideEffect<Raw>) -> Vec<NodeId> {
    let mut nodes = effect.operand_nodes.to_vec();
    let SideEffectKind::Soac(_, soac) = &effect.kind else {
        return nodes;
    };
    nodes.extend(soac.seg_bodies().into_iter().flat_map(|body| body.captures.iter().copied()));
    if let Soac::Screma(op) = soac {
        for operator in op.operators() {
            nodes.push(operator.neutral);
            nodes.extend(operator.shape.iter().copied());
        }
    }
    nodes
}

fn remap_headers(
    headers: LookupMap<BlockId, ControlHeader>,
    blocks: &LookupMap<BlockId, BlockId>,
) -> LookupMap<BlockId, ControlHeader> {
    headers
        .into_iter()
        .map(|(block, header)| (blocks[&block], header.remap(&|target| blocks[&target])))
        .collect()
}

fn remap_routes(mut routes: Vec<OutputRoute>, blocks: &LookupMap<BlockId, BlockId>) -> Vec<OutputRoute> {
    for route in &mut routes {
        route.source.block = blocks[&route.source.block];
    }
    routes
}
