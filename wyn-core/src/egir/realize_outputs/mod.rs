//! Record the semantic producers of each host-visible entry output.
//!
//! This phase does not select storage or mutate the graph representation.
//! Fusion, residency, and scheduling retain the original result values while
//! updating route provenance. Concrete places and writes are installed only
//! when the planned program is physicalized.

use crate::ast;
use crate::flow::BlockId;
use crate::interface;
use crate::pipeline_descriptor;
use crate::ssa;
use crate::types;
use crate::types::TypeExt;
use crate::LookupMap;
use crate::LookupSet;
use polytype::Type;

use super::from_tlc::ConvertError;
use super::ir::{RealizedOutputRoute, ResultBinding};
use super::program::{OutputWriter, RawEntry, SlotSource};
use super::types::{EGraph, EffectToken, Raw, SideEffectKind, SkeletonTerminator, SoacEffect, ValueId};
use super::{soac::filter, types::Soac};

pub mod verify;

pub fn realize_outputs(
    mut program: super::from_tlc::Converted,
) -> Result<super::from_tlc::Converted, ConvertError> {
    for entry in &mut program.entry_points {
        record_entry_outputs(entry)?;
    }
    Ok(program)
}

fn record_entry_outputs(entry: &mut RawEntry) -> Result<(), ConvertError> {
    if entry.outputs.is_empty() {
        return Ok(());
    }
    if entry.routes().next().is_none() {
        synthesize_routes(entry)?;
    }

    let graph = &entry.graph;
    let effect_index = graph.side_effect_index();
    let resource_writers = super::graph_ops::resource_effect_writers(graph);
    for (slot, output) in entry.outputs.iter_mut().enumerate() {
        if output.routes.is_empty() {
            return Err(ConvertError::Unsupported(format!(
                "entry output #{slot} has no source"
            )));
        }
        for route in &mut output.routes {
            let mut writers =
                source_value_writers(graph, &effect_index, &resource_writers, route.source.value);
            writers.push(OutputWriter::Value(route.source.value));
            let mut seen = LookupSet::new();
            writers.retain(|writer| seen.insert(*writer));
            route.writers = writers;
        }

        let [route] = output.routes.as_slice() else {
            continue;
        };
        let Some(source_resource) =
            super::graph_ops::extract_storage_view_source(graph, route.source.value)
        else {
            continue;
        };
        let length = output.storage_length().cloned();
        output.kind = interface::EntryOutputKind::Storage {
            exposure: interface::BindingExposure::Host(source_resource),
            length,
        };
        output.resource = Some(source_resource);
    }
    for slot in 0..entry.outputs.len() {
        bind_runtime_filter_output(entry, slot)?;
    }
    Ok(())
}

fn bind_runtime_filter_output(entry: &mut RawEntry, slot: usize) -> Result<(), ConvertError> {
    let Some(output_binding) = entry.outputs[slot].resource else {
        return Ok(());
    };
    let [route] = entry.outputs[slot].routes.as_slice() else {
        return Ok(());
    };
    let effect_index = entry.graph.side_effect_index();
    let Some((effect, result, _)) = effect_index.effect_result_field(&entry.graph, route.source.value)
    else {
        return Ok(());
    };
    let SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) = effect.kind() else {
        return Ok(());
    };
    let filter::Output::Runtime(runtime) = &op.state.output else {
        return Ok(());
    };
    if !matches!(runtime.backing, filter::RuntimeBacking::Deferred) {
        return Ok(());
    }
    let input =
        op.body.inputs.first().ok_or_else(|| ConvertError::Internal("Filter has no array input".into()))?;
    let input_array = input.array.clone();
    let input_element = input.element();
    let output_element = op.body.output_element_type();
    let effect = effect_index
        .effect_mut(&mut entry.graph, result)
        .ok_or_else(|| ConvertError::Internal("Filter result lost its producer".into()))?;
    let SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) = &mut effect.kind else {
        unreachable!("located Filter producer changed")
    };
    let filter::Output::Runtime(runtime) = &mut op.state.output else {
        unreachable!()
    };
    runtime.backing = filter::RuntimeBacking::Bound(output_binding);
    runtime.length = filter::RuntimeLength::Required;

    let length = input_array.array_buffer().and_then(|region| {
        let Type::Constructed(ast::TypeName::Buffer(binding), _) = region else {
            return None;
        };
        Some(pipeline_descriptor::BufferLen::LikeInput {
            set: binding.set,
            binding: binding.binding,
            elem_bytes: ssa::layout::storage_elem_stride(&output_element)?,
            src_elem_bytes: ssa::layout::storage_elem_stride(&input_element)?,
        })
    });
    *entry.outputs[slot].storage_length_mut().expect("runtime Filter output is storage") = length;
    Ok(())
}

fn synthesize_routes(entry: &mut RawEntry) -> Result<(), ConvertError> {
    let Some((return_block, result)) = unique_value_return(&entry.graph) else {
        return Ok(());
    };
    let sources =
        output_sources(&mut entry.graph, &result, entry.outputs.len()).map_err(ConvertError::Internal)?;
    for (output, source) in entry.outputs.iter_mut().zip(sources) {
        output.routes.push(RealizedOutputRoute {
            source: SlotSource {
                block: return_block,
                value: source,
            },
            writers: Vec::new(),
        });
    }
    Ok(())
}

fn unique_value_return(graph: &EGraph<Raw>) -> Option<(BlockId, ResultBinding<types::Type>)> {
    let mut returns = graph.skeleton.blocks.iter().filter_map(|(block, body)| {
        let SkeletonTerminator::Return(Some(result)) = &body.term else {
            return None;
        };
        Some((block, result.clone()))
    });
    let result = returns.next();
    assert!(
        returns.next().is_none(),
        "entry body has more than one value-returning terminator"
    );
    result
}

fn source_value_writers(
    graph: &EGraph<Raw>,
    effect_index: &super::types::SideEffectIndex,
    resource_writers: &LookupMap<crate::BindingRef, Vec<EffectToken>>,
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
        super::graph_ops::read_storage_resources(graph, [source])
            .into_iter()
            .filter_map(|access| resource_writers.get(&access.resource))
            .flatten()
            .copied()
            .map(OutputWriter::Effect),
    );
    writers
}

fn output_sources(
    graph: &mut EGraph<Raw>,
    result: &ResultBinding<types::Type>,
    output_count: usize,
) -> Result<Vec<ValueId>, String> {
    if output_count == 1 {
        return super::graph_ops::pack_result_values(graph, result).map(|source| vec![source]);
    }
    let fields = result.top_level_fields();
    if fields.len() != output_count {
        return Err(format!(
            "entry result has {} logical fields for {output_count} declared outputs",
            fields.len()
        ));
    }
    fields.iter().map(|field| super::graph_ops::pack_result_values(graph, field)).collect()
}
