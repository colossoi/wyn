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
    EGraph, PureOp, Raw, ResourceAccess, SegExtent, SegResourceAccess, SegSpace, Semantic, SideEffect,
    SideEffectKind, Soac, SoacEffect, SoacInputType, ValueId, ValueKind,
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
        id,
        name,
        span,
        return_ty,
        graph,
    } = constant;
    let (graph, _) = map_graph(graph, facts, semantic_ids);
    ConstantDef {
        id,
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
        result,
        effects,
        graph,
    } = function;
    let (graph, _) = map_graph(graph, facts, semantic_ids);
    Func {
        region,
        name,
        span,
        linkage_name,
        params,
        result,
        effects,
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
            inputs: op.inputs,
            form: op.form,
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
            operands: SmallVec::new(),
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
            operands: SmallVec::new(),
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
                if kernel_scope
                    && !effect
                        .result
                        .as_ref()
                        .is_some_and(|result| result.values().iter().any(|value| consumed.contains(value)))
                {
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
            op.form.operator_input_count() != 0
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
    let (inputs, is_screma) = match soac {
        Soac::Screma(op) => (op.inputs.as_slice(), true),
        Soac::Filter(op) => (op.body.inputs.as_slice(), false),
        Soac::Hist(op) => (op.inputs.as_slice(), false),
    };
    let output_slots = if is_screma {
        entry.map_or_else(Vec::new, |entry| output_slots(entry, effect))
    } else {
        Vec::new()
    };
    let resources =
        if is_screma { semantic_resources(graph, entry, effect, &output_slots) } else { Vec::new() };
    Some(Facts {
        space: space(graph, entry, effect, inputs),
        placement: requested_placement,
        output_slots,
        resources,
        entry: entry.is_some(),
    })
}

fn space(
    graph: &EGraph<Raw>,
    entry: Option<&RawEntry<RealizedOutputRoute>>,
    effect: &SideEffect<Raw>,
    inputs: &[SoacInputType],
) -> SegSpace<SemanticResourceRef> {
    let domain_rank = inputs
        .iter()
        .flat_map(|input| input.dimensions.iter().copied())
        .max()
        .map(|dimension| usize::from(dimension) + 1)
        .expect("semantic SOAC must have a domain input");
    let mut dims = Vec::with_capacity(domain_rank);
    for logical_dimension in 0..domain_rank {
        let (operand_index, input, array_axis) = inputs
            .iter()
            .enumerate()
            .find_map(|(operand_index, input)| {
                input
                    .dimensions
                    .iter()
                    .position(|dimension| usize::from(*dimension) == logical_dimension)
                    .map(|array_axis| (operand_index, input, array_axis))
            })
            .unwrap_or_else(|| panic!("SOAC domain dimension {logical_dimension} has no input"));
        let node = effect.operands[operand_index]
            .value()
            .expect("SOAC input uses the value or view channel");
        let mut dimension_ty = &input.array;
        while let Some(components) = super::types::as_soa_tuple(dimension_ty) {
            dimension_ty = components.first().expect("structure-of-arrays tuple must have a component");
        }
        for _ in 0..array_axis {
            dimension_ty = dimension_ty
                .elem_type()
                .unwrap_or_else(|| panic!("SOAC input rank exceeds array type {:?}", input.array));
            while let Some(components) = super::types::as_soa_tuple(dimension_ty) {
                dimension_ty = components.first().expect("structure-of-arrays tuple must have a component");
            }
        }
        let extent = if array_axis == 0 {
            if let Some(Type::Constructed(TypeName::Size(size), _)) = input.array.array_size() {
                SegExtent::Fixed(u32::try_from(*size).expect("ranked SOAC dimension is too large"))
            } else if let Some(resource) = graph_ops::extract_storage_view_source(graph, node) {
                let elem_bytes = crate::ssa::layout::storage_elem_stride(
                    input.array.elem_type().expect("resource-backed SOAC input must be an array"),
                )
                .expect("resource-backed SOAC input must have a storable element type");
                SegExtent::ResourceLength {
                    view: graph.view_id(node),
                    resource,
                    elem_bytes,
                }
            } else if let Some((_, len, _)) = graph_ops::extract_array_range_operands(graph, node) {
                extent_from_node(graph, entry, len)
            } else {
                SegExtent::Value(node)
            }
        } else {
            let Type::Constructed(TypeName::Size(size), _) = dimension_ty
            .array_size()
            .unwrap_or_else(|| {
                panic!(
                    "ranked SOAC array axis {array_axis} has no size: input={:?}, dimension={dimension_ty:?}",
                    input.array
                )
            })
            else {
                panic!("ranked SOAC inner dimensions must be fixed")
            };
            SegExtent::Fixed(u32::try_from(*size).expect("ranked SOAC dimension is too large"))
        };
        dims.push(extent);
    }
    SegSpace::from_dims(dims).expect("ranked SOAC space is non-empty")
}

fn extent_from_node(
    graph: &EGraph<Raw>,
    entry: Option<&RawEntry<RealizedOutputRoute>>,
    node: ValueId,
) -> SegExtent<SemanticResourceRef> {
    match &graph.nodes[node].kind {
        ValueKind::Pure {
            op: PureOp::Int(value) | PureOp::Uint(value),
            ..
        } => value.parse().map(SegExtent::Fixed).unwrap_or(SegExtent::Value(node)),
        ValueKind::FuncParam { parameter } => entry
            .and_then(|entry| entry.inputs.get(parameter.index()))
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
    let value_writers = effect
        .result
        .as_ref()
        .map(|result| result.values())
        .unwrap_or_default();
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
                    .any(|writer| {
                        matches!(writer, OutputWriter::Value(value) if value_writers.contains(value))
                            || Some(*writer) == effect_writer
                    })
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

fn soac_consumed_nodes(graph: &EGraph<Raw>) -> HashSet<ValueId> {
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

fn referenced_nodes(effect: &SideEffect<Raw>) -> Vec<ValueId> {
    let mut nodes = effect.operand_values().collect::<Vec<_>>();
    let SideEffectKind::Soac(SoacEffect(_, soac)) = &effect.kind else {
        return nodes;
    };
    nodes.extend(soac.seg_bodies().into_iter().flat_map(|body| body.capture_values()));
    match soac {
        Soac::Screma(op) => {
            nodes.extend(op.form.scans.iter().flat_map(|scan| scan.neutral.iter().copied()));
            nodes.extend(op.form.reductions.iter().flat_map(|reduction| reduction.neutral.iter().copied()));
        }
        Soac::Hist(op) => nodes.extend(op.referenced_nodes()),
        Soac::Filter(_) => {}
    }
    nodes
}
