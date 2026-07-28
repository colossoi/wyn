//! Raw EGIR to semantic EGIR.
//!
//! This is the single boundary that constructs semantic SOAC state. The
//! operation family is preserved by the direct match in `reify_soac`.

/// EGIR whose higher-order array operations have semantic segmented form.
#[derive(Debug, Clone, Copy)]
pub enum SegmentedTag {}
pub type Segmented = super::program::Program<
    SegmentedTag,
    super::ir::ProgramFamily<
        super::types::Semantic,
        super::program::SemanticResourceDecl,
        super::ir::RealizedOutputRoute,
        super::program::CoreProgramData,
    >,
    super::program::RewriteGlobal,
>;

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;

use polytype::Type;

use crate::ast::TypeName;
use crate::flow::BlockId;
use crate::types::TypeExt;
use crate::LookupMap;

use super::graph_ops;
use super::program::RealizedOutputRoute;
use super::program::{
    ConstantDef, Entry, Func, OutputSlotId, OutputWriter, Program, RawEntry, SemanticOpIdSource,
    SemanticResourceRef,
};
use super::realize_outputs::OutputsRealized;
use super::soac::{filter, hist, screma};
use super::types::{
    EGraph, ENode, NodeId, PureOp, Raw, ResourceAccess, SegExtent, SegResourceAccess, SegSpace, Semantic,
    SideEffect, SideEffectKind, Soac, SoacEffect, SoacInputType,
};

struct Facts {
    space: SegSpace<SemanticResourceRef>,
    placement: screma::Placement,
    output_slots: Vec<OutputSlotId>,
    resources: Vec<SegResourceAccess<SemanticResourceRef>>,
    entry: bool,
}

pub fn reify_soacs(program: OutputsRealized) -> Segmented {
    let Program {
        functions,
        externs,
        entry_points,
        constants,
        data,
        mut global_context,
        state: _,
    } = program;

    let entry_points = entry_points
        .into_iter()
        .map(|entry| reify_entry(entry, &mut global_context.semantic_ids))
        .collect();
    let functions = functions
        .into_iter()
        .map(|function| reify_func(function, &mut global_context.semantic_ids))
        .collect();
    let constants = constants
        .into_iter()
        .map(|constant| reify_constant(constant, &mut global_context.semantic_ids))
        .collect();

    Program::from_parts(functions, externs, entry_points, constants, data, global_context)
}

fn reify_constant(
    constant: ConstantDef<Raw>,
    semantic_ids: &mut SemanticOpIdSource,
) -> ConstantDef<Semantic> {
    let facts = function_facts(&constant.graph);
    let ConstantDef {
        name,
        span,
        return_ty,
        graph,
    } = constant;
    let (graph, _) = map_graph(graph, facts, semantic_ids);
    ConstantDef {
        name,
        span,
        return_ty,
        graph,
    }
}

fn reify_func(function: Func<Raw>, semantic_ids: &mut SemanticOpIdSource) -> Func<Semantic> {
    let facts = function_facts(&function.graph);
    let Func {
        region,
        name,
        span,
        linkage_name,
        params,
        return_ty,
        graph,
    } = function;
    let (graph, _) = map_graph(graph, facts, semantic_ids);
    Func {
        region,
        name,
        span,
        linkage_name,
        params,
        return_ty,
        graph,
    }
}

fn reify_entry(
    entry: Entry<Raw, super::program::SemanticResourceDecl, RealizedOutputRoute>,
    semantic_ids: &mut SemanticOpIdSource,
) -> Entry<Semantic> {
    let mut facts = entry_facts(&entry);
    match entry.try_map_phase(|block, index, (), soac| {
        let facts = facts.remove(&(block, index)).expect("every raw SOAC must have semantic facts");
        let id = semantic_ids.next_id();
        Ok::<_, Infallible>((id, reify_soac(soac, facts)))
    }) {
        Ok(entry) => entry,
        Err(never) => match never {},
    }
}

fn map_graph(
    graph: EGraph<Raw>,
    mut facts: HashMap<(BlockId, usize), Facts>,
    semantic_ids: &mut SemanticOpIdSource,
) -> (EGraph<Semantic>, LookupMap<BlockId, BlockId>) {
    match graph.try_map_phase(|block, index, (), soac| {
        let facts = facts.remove(&(block, index)).expect("every raw SOAC must have semantic facts");
        let id = semantic_ids.next_id();
        Ok::<_, Infallible>((id, reify_soac(soac, facts)))
    }) {
        Ok(mapped) => mapped,
        Err(never) => match never {},
    }
}

fn reify_soac(soac: Soac<Raw>, facts: Facts) -> Soac<Semantic> {
    match soac {
        Soac::Screma(screma::Op {
            inputs,
            form,
            result_state,
            ..
        }) => {
            let mut placement = facts.placement;
            let map_only = form.scans.is_empty() && form.reductions.is_empty();
            if map_only
                && placement == screma::Placement::Kernel
                && (form.post.result_types.is_empty()
                    || result_state.iter().all(|result| !result.destination.is_unplaced()))
            {
                placement = screma::Placement::LaneLocal;
            }
            if placement == screma::Placement::Kernel && facts.output_slots.is_empty() {
                placement = screma::Placement::LaneLocal;
            }
            Soac::Screma(screma::Op {
                inputs,
                form,
                result_state,
                state: screma::SemanticState::Segmented {
                    space: facts.space,
                    placement,
                    output_slots: facts.output_slots,
                    resources: facts.resources,
                },
            })
        }
        Soac::Filter(op) => {
            let storage = op.state.storage;
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
            state: if facts.entry { hist::State::Segmented(facts.space) } else { hist::State::Serial },
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egir::program::SemanticOpId;
    use crate::egir::types::EffectOp;
    use smallvec::SmallVec;

    fn raw_map() -> SideEffect<Raw> {
        SideEffect {
            kind: SideEffectKind::Soac(SoacEffect(
                (),
                Soac::Screma(screma::Op {
                    inputs: vec![],
                    form: screma::ScremaForm {
                        pre: screma::Lambda::identity(vec![]),
                        scans: vec![],
                        reductions: vec![],
                        post: screma::Lambda::identity(vec![]),
                    },
                    result_state: vec![],
                    state: screma::RawState,
                }),
            )),
            operand_nodes: SmallVec::new(),
            result: None,
            effects: None,
            span: None,
        }
    }

    fn facts() -> Facts {
        Facts {
            space: SegSpace::new(SegExtent::Fixed(1)),
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

        let mut semantic_ids = SemanticOpIdSource::default();
        for _ in 0..7 {
            semantic_ids.next_id();
        }
        let (graph, _) = map_graph(
            graph,
            HashMap::from([((block, 0), facts()), ((block, 2), facts())]),
            &mut semantic_ids,
        );
        let ids: Vec<_> = graph.skeleton.blocks[graph.skeleton.entry]
            .side_effects
            .iter()
            .map(|effect| effect.kind.soac_id().copied())
            .collect();

        assert_eq!(
            ids,
            vec![
                Some(SemanticOpId::for_test(7)),
                None,
                Some(SemanticOpId::for_test(8))
            ]
        );
        assert_eq!(semantic_ids.next_id(), SemanticOpId::for_test(9));
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

fn entry_facts(entry: &RawEntry<RealizedOutputRoute>) -> HashMap<(BlockId, usize), Facts> {
    let consumed = soac_consumed_nodes(&entry.graph);
    let kernel_scope = entry.execution_model.is_compute();
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
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                return false;
            };
            !op.is_map()
                && !op.is_mixed()
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
    entry: Option<&RawEntry<RealizedOutputRoute>>,
    effect: &SideEffect<Raw>,
    requested_placement: screma::Placement,
) -> Option<Facts> {
    let SideEffectKind::Soac(SoacEffect(_, soac)) = &effect.kind else {
        return None;
    };
    let (input, operand_index, is_screma) = match soac {
        Soac::Screma(op) => (op.inputs.first(), 0, true),
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
    entry: Option<&RawEntry<RealizedOutputRoute>>,
    effect: &SideEffect<Raw>,
    input: Option<&SoacInputType>,
    operand_index: usize,
) -> SegSpace<SemanticResourceRef> {
    let extent = effect.operand_nodes.get(operand_index).copied().map(|node| {
        if let Some(resource) = graph_ops::extract_storage_view_source(graph, node) {
            let elem_bytes = input
                .and_then(|input| crate::ssa::layout::storage_elem_stride(&input.element()))
                .expect("resource-backed SOAC input must have a storable element type");
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
    SegSpace::new(extent.expect("semantic SOAC must have a domain operand"))
}

fn extent_from_node(
    graph: &EGraph<Raw>,
    entry: Option<&RawEntry<RealizedOutputRoute>>,
    node: NodeId,
) -> SegExtent<SemanticResourceRef> {
    match &graph.nodes[node].kind {
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

fn output_slots(entry: &RawEntry<RealizedOutputRoute>, effect: &SideEffect<Raw>) -> Vec<OutputSlotId> {
    let value_writer = effect.result.map(OutputWriter::Value);
    let effect_writer = effect.effects.map(|(_, output)| OutputWriter::Effect(output));
    let mut slots = entry
        .outputs
        .iter()
        .enumerate()
        .filter(|(_, output)| {
            output.routes.iter().any(|route| {
                route
                    .writers
                    .iter()
                    .any(|writer| Some(*writer) == value_writer || Some(*writer) == effect_writer)
            })
        })
        .map(|(slot, _)| OutputSlotId(slot))
        .collect::<Vec<_>>();
    slots.sort_unstable();
    slots.dedup();
    slots
}

fn semantic_resources(
    graph: &EGraph<Raw>,
    entry: Option<&RawEntry<RealizedOutputRoute>>,
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
    graph_ops::read_storage_resources(graph, referenced_nodes(effect))
}

fn soac_consumed_nodes(graph: &EGraph<Raw>) -> HashSet<NodeId> {
    let roots = graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, block)| &block.side_effects)
        .filter(|effect| matches!(effect.kind, SideEffectKind::Soac(SoacEffect(_, _))))
        .flat_map(referenced_nodes)
        .collect::<Vec<_>>();
    graph_ops::value_producer_closure(graph, roots).nodes
}

fn referenced_nodes(effect: &SideEffect<Raw>) -> Vec<NodeId> {
    let mut nodes = effect.operand_nodes.to_vec();
    let SideEffectKind::Soac(SoacEffect(_, soac)) = &effect.kind else {
        return nodes;
    };
    nodes.extend(soac.seg_bodies().into_iter().flat_map(|body| body.captures.iter().copied()));
    if let Soac::Screma(op) = soac {
        nodes.extend(op.form.scans.iter().flat_map(|scan| scan.neutral.iter().copied()));
        nodes.extend(op.form.reductions.iter().flat_map(|reduction| reduction.neutral.iter().copied()));
    }
    nodes
}
