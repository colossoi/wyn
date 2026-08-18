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
        super::program::NoStorageDeclaration,
        super::ir::RealizedOutputRoute,
        super::program::SemanticProgramData,
    >,
    super::program::RewriteGlobal,
>;

use crate::ssa;
use std::collections::HashMap;
use std::convert::Infallible;

use polytype::Type;

use crate::ast::TypeName;
use crate::flow::BlockId;
use crate::types::TypeExt;
use crate::{BindingRef, LookupMap, LookupSet};

use super::from_tlc::Converted;
use super::graph_ops;
use super::program::{
    ConstantDef, Entry, Func, OutputSlotId, OutputWriter, Program, RawEntry, SemanticOpIdSource,
};
use super::soac::{filter, hist, screma};
use super::types::{
    EGraph, EffectToken, PureOp, Raw, ResourceAccess, SegExtent, SegResourceAccess, SegSpace, Semantic,
    SideEffect, SideEffectKind, Soac, SoacEffect, SoacInputType, ValueId, ValueKind,
};

struct Facts {
    space: SegSpace<BindingRef>,
    output_slots: Vec<OutputSlotId>,
    resources: Vec<SegResourceAccess<BindingRef>>,
    entry: bool,
}

pub fn reify_soacs(program: Converted) -> Segmented {
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

fn reify_entry(mut entry: RawEntry, semantic_ids: &mut SemanticOpIdSource) -> Entry<Semantic> {
    link_output_producers(&mut entry);
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
        }) => Soac::Screma(screma::Op {
            inputs,
            form,
            result_state,
            state: screma::SemanticState::Segmented {
                space: facts.space,
                output_slots: facts.output_slots,
                resources: facts.resources,
            },
        }),
        Soac::Filter(op) => {
            let output = match op.state.output {
                filter::RawOutput::Local { capacity, ownership } => {
                    filter::Output::Local { capacity, ownership }
                }
                filter::RawOutput::Runtime { capacity } => filter::Output::Runtime(filter::RuntimeOutput {
                    capacity,
                    backing: filter::RuntimeBacking::Deferred,
                    length: filter::RuntimeLength::Implicit,
                }),
            };
            Soac::Filter(filter::Op {
                body: op.body,
                state: filter::SemanticState {
                    space: facts.space,
                    output,
                    output_slots: facts.output_slots,
                    resources: facts.resources,
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
#[path = "reify_tests.rs"]
mod tests;

fn function_facts(graph: &EGraph<Raw>) -> HashMap<(BlockId, usize), Facts> {
    graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(block, contents)| {
            contents.side_effects.iter().enumerate().filter_map(move |(index, effect)| {
                semantic_facts(graph, None, effect).map(|facts| ((block, index), facts))
            })
        })
        .collect()
}

fn entry_facts(entry: &RawEntry) -> HashMap<(BlockId, usize), Facts> {
    entry
        .graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(block, contents)| {
            contents.side_effects.iter().enumerate().filter_map(move |(index, effect)| {
                semantic_facts(&entry.graph, Some(entry), effect).map(|facts| ((block, index), facts))
            })
        })
        .collect()
}

fn semantic_facts(
    graph: &EGraph<Raw>,
    entry: Option<&RawEntry>,
    effect: &SideEffect<Raw>,
) -> Option<Facts> {
    let SideEffectKind::Soac(SoacEffect(_, soac)) = &effect.kind else {
        return None;
    };
    let inputs = match soac {
        Soac::Screma(op) => op.inputs.as_slice(),
        Soac::Filter(op) => op.body.inputs.as_slice(),
        Soac::Hist(op) => op.inputs.as_slice(),
    };
    let output_slots = match (entry, soac) {
        (Some(entry), Soac::Screma(_)) => output_slots(entry, effect),
        (Some(entry), Soac::Filter(_)) => direct_output_slots(entry, effect),
        _ => Vec::new(),
    };
    let resources = if matches!(soac, Soac::Screma(_) | Soac::Filter(_)) {
        semantic_resources(graph, entry, effect, &output_slots)
    } else {
        Vec::new()
    };
    Some(Facts {
        space: space(graph, entry, effect, inputs),
        output_slots,
        resources,
        entry: entry.is_some(),
    })
}

fn space(
    graph: &EGraph<Raw>,
    entry: Option<&RawEntry>,
    effect: &SideEffect<Raw>,
    inputs: &[SoacInputType],
) -> SegSpace<BindingRef> {
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
        let node =
            effect.operands[operand_index].value().expect("SOAC input uses the value or view channel");
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
                let elem_bytes = ssa::layout::storage_elem_stride(
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

fn extent_from_node(graph: &EGraph<Raw>, entry: Option<&RawEntry>, node: ValueId) -> SegExtent<BindingRef> {
    match &graph.nodes[node].kind {
        ValueKind::Pure {
            op: PureOp::Int(value) | PureOp::Uint(value),
            ..
        } => value.parse().map(SegExtent::Fixed).unwrap_or(SegExtent::Value(node)),
        ValueKind::FuncParam { parameter } => entry
            .and_then(|entry| {
                entry.params().abi_position(*parameter).and_then(|position| entry.inputs.get(position))
            })
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
    let value_writers = effect.result_values();
    let effect_writer = effect.effects.map(|(_, output)| OutputWriter::Effect(output));
    let mut slots = entry
        .outputs
        .iter()
        .enumerate()
        .filter(|(_, output)| {
            output.routes.iter().any(|route| {
                route.writers.iter().any(|writer| {
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

/// A Filter publishes its compacted representation only when the route's
/// returned source is the Filter result itself. Route writer provenance is
/// transitive for DCE and fusion, so using every writer here would make an
/// upstream Filter claim downstream map or length slots.
fn direct_output_slots(entry: &RawEntry, effect: &SideEffect<Raw>) -> Vec<OutputSlotId> {
    let results = effect.result_values();
    entry
        .outputs
        .iter()
        .enumerate()
        .filter(|(_, output)| output.routes.iter().any(|route| results.contains(&route.source.value)))
        .map(|(slot, _)| OutputSlotId(slot))
        .collect()
}

/// Link completed output routes to the raw semantic producers that can fulfil
/// them. This is intentionally private to the raw-to-semantic boundary: route
/// construction belongs to conversion, while producer discovery requires the
/// complete graph and is consumed immediately by reification, fusion, and DCE.
fn link_output_producers(entry: &mut RawEntry) {
    let graph = &entry.graph;
    let effect_index = graph.side_effect_index();
    let resource_writers = graph_ops::resource_effect_writers(graph);
    for output in &mut entry.outputs {
        for route in &mut output.routes {
            let mut writers =
                source_value_writers(graph, &effect_index, &resource_writers, route.source.value);
            writers.push(OutputWriter::Value(route.source.value));
            let mut seen = LookupSet::new();
            writers.retain(|writer| seen.insert(*writer));
            route.writers = writers;
        }
    }
}

fn source_value_writers(
    graph: &EGraph<Raw>,
    effect_index: &super::types::SideEffectIndex,
    resource_writers: &LookupMap<BindingRef, Vec<EffectToken>>,
    source: ValueId,
) -> Vec<OutputWriter> {
    let mut writers = Vec::new();
    wyn_graph::for_each_reachable(
        [source],
        wyn_graph::WalkOrder::DepthFirst,
        |node, dependencies| {
            if effect_index.site(node).is_none() {
                dependencies.extend(graph.nodes[node].kind.children());
            }
        },
        |node| {
            if effect_index
                .effect(graph, node)
                .is_some_and(|effect| matches!(effect.kind, SideEffectKind::Soac(SoacEffect(_, _))))
            {
                writers.push(OutputWriter::Value(node));
            }
        },
    );
    writers.extend(
        graph_ops::read_storage_resources(graph, [source])
            .into_iter()
            .filter_map(|access| resource_writers.get(&access.resource))
            .flatten()
            .copied()
            .map(OutputWriter::Effect),
    );
    writers
}

fn semantic_resources(
    graph: &EGraph<Raw>,
    entry: Option<&RawEntry>,
    effect: &SideEffect<Raw>,
    output_slots: &[OutputSlotId],
) -> Vec<SegResourceAccess<BindingRef>> {
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

fn read_resources(graph: &EGraph<Raw>, effect: &SideEffect<Raw>) -> Vec<SegResourceAccess<BindingRef>> {
    graph_ops::read_storage_resources(graph, referenced_nodes(effect))
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
