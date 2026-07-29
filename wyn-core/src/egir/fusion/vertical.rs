//! Same-space producer/consumer fusion for canonical Scremas.
//!
//! A pure map producer can be folded into any downstream Screma by composing
//! its pre-lambda with the consumer's pre-lambda. This preserves the consumer's
//! single collective barrier and never reconstructs the former lane graph.

use polytype::Type;
use smallvec::SmallVec;

use super::graph_and_span;
use super::horizontal;
use super::screma as fusion_screma;
use super::support;
use crate::ast::TypeName;
use crate::egir::graph_ops;
use crate::egir::ir::{splice_effect_tokens, BodySite};
use crate::egir::program::CoreProgramData;
use crate::egir::reify::Segmented;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::soac::screma;
use crate::egir::types::{EGraph, ResourceAccess, SegResourceAccess, SideEffectKind, Soac, SoacEffect};
use crate::flow::BlockId;
use crate::types::TypeExt;
use crate::LookupMap;

#[derive(Clone)]
pub(crate) struct Candidate {
    site: BodySite,
    block: BlockId,
    producer: usize,
    consumer: usize,
    routes: Vec<fusion_screma::InputRoute>,
    retained_producer_outputs: Vec<usize>,
}

pub(super) fn analyze(inner: &Segmented, oracle: &SemanticGraph) -> Option<Candidate> {
    for (index, entry) in inner.entry_points.iter().enumerate() {
        if let Some(candidate) = find_in_graph(inner, &entry.graph, BodySite::Entry(index), oracle) {
            return Some(candidate);
        }
    }
    for function in &inner.functions {
        if let Some(candidate) = find_in_graph(
            inner,
            &function.graph,
            BodySite::Function(function.region),
            oracle,
        ) {
            return Some(candidate);
        }
    }
    None
}

fn find_in_graph(
    inner: &Segmented,
    graph: &EGraph,
    site: BodySite,
    oracle: &SemanticGraph,
) -> Option<Candidate> {
    for (block_id, block) in &graph.skeleton.blocks {
        for producer_index in 0..block.side_effects.len().saturating_sub(1) {
            let producer = &block.side_effects[producer_index];
            let SideEffectKind::Soac(SoacEffect(producer_id, Soac::Screma(producer_op))) = &producer.kind
            else {
                continue;
            };
            let screma::SemanticState::Segmented { resources, .. } = producer_op.semantic_state() else {
                continue;
            };
            let Some(producer_result) = producer.result else {
                continue;
            };
            let value_consumers =
                oracle.value_consumers(producer_id).collect::<std::collections::HashSet<_>>();
            let value_consumer_kinds = block
                .side_effects
                .iter()
                .filter_map(|effect| match &effect.kind {
                    SideEffectKind::Soac(SoacEffect(id, Soac::Screma(op)))
                        if value_consumers.contains(id) =>
                    {
                        Some(op.is_map())
                    }
                    SideEffectKind::Soac(SoacEffect(id, _)) if value_consumers.contains(id) => Some(false),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let all_value_consumers_are_maps = value_consumers.len() > 1
                && value_consumer_kinds.len() == value_consumers.len()
                && value_consumer_kinds.into_iter().all(|is_map| is_map);

            for consumer_index in (producer_index + 1)..block.side_effects.len() {
                let consumer = &block.side_effects[consumer_index];
                let SideEffectKind::Soac(SoacEffect(consumer_id, Soac::Screma(consumer_op))) =
                    &consumer.kind
                else {
                    continue;
                };
                let screma::SemanticState::Segmented {
                    resources: consumer_resources,
                    ..
                } = consumer_op.semantic_state()
                else {
                    continue;
                };
                if consumer.result.is_none() {
                    continue;
                }

                if !((producer_index + 1)..consumer_index).all(|index| {
                    let effect = &block.side_effects[index];
                    match (&effect.kind, effect.result) {
                        (SideEffectKind::Soac(SoacEffect(intervening, Soac::Screma(_))), Some(_)) => {
                            !oracle.conflicts(producer_id, intervening)
                        }
                        _ => effect.effects.is_none(),
                    }
                }) {
                    continue;
                }

                let producer_input_count = producer_op.inputs.len();
                let mut producer_output_operands =
                    producer.operand_nodes[producer_input_count..].iter().copied();
                let producer_output_nodes = (0..producer_op.result_count())
                    .map(|field| {
                        producer_op
                            .destination(field)
                            .filter(|destination| destination.is_output_view())
                            .map(|_| {
                                producer_output_operands
                                    .next()
                                    .expect("missing producer output-view operand")
                            })
                    })
                    .collect::<Vec<_>>();
                debug_assert!(producer_output_operands.next().is_none());
                let consumer_input_count = consumer_op.inputs.len();
                let projected = consumer.operand_nodes[..consumer_input_count]
                    .iter()
                    .enumerate()
                    .filter_map(|(input, &operand)| {
                        graph_ops::projection_index(graph, operand, producer_result)
                            .or_else(|| {
                                producer_output_nodes.iter().position(|output| *output == Some(operand))
                            })
                            .map(|field| (input, field))
                    })
                    .collect::<Vec<_>>();
                if projected.is_empty() {
                    continue;
                }
                let routes = projected
                    .iter()
                    .map(|(consumer_input, producer_field)| {
                        let screma::ResultId::Post(producer_post_output) =
                            producer_op.form.result_id(*producer_field)?
                        else {
                            return None;
                        };
                        Some(fusion_screma::InputRoute {
                            consumer_input: *consumer_input,
                            producer_post_output,
                        })
                    })
                    .collect::<Option<Vec<_>>>();
                let Some(routes) = routes else {
                    continue;
                };
                let routed_resources = routes
                    .iter()
                    .filter_map(|route| consumer_op.inputs[route.consumer_input].array.array_buffer())
                    .filter_map(|buffer| match buffer {
                        Type::Constructed(TypeName::Resource(resource), _) => Some(*resource),
                        _ => None,
                    })
                    .collect::<std::collections::HashSet<_>>();
                let unrouteable_resource_conflict = resources.iter().any(|producer_resource| {
                    consumer_resources.iter().any(|consumer_resource| {
                        producer_resource.resource == consumer_resource.resource
                            && (producer_resource.access != ResourceAccess::Read
                                || consumer_resource.access != ResourceAccess::Read)
                            && !routed_resources.contains(&producer_resource.resource.0)
                    })
                });
                let direct_effect_chain = matches!(
                    (producer.effects, consumer.effects),
                    (Some((_, producer_out)), Some((consumer_in, _))) if producer_out == consumer_in
                );
                if unrouteable_resource_conflict
                    || (oracle.conflicts(producer_id, consumer_id)
                        && !direct_effect_chain
                        && routed_resources.is_empty())
                {
                    continue;
                }
                let projected_roots = projected
                    .iter()
                    .map(|(input, _)| consumer.operand_nodes[*input])
                    .collect::<std::collections::HashSet<_>>();
                let semantic_roots = consumer_op
                    .capture_nodes()
                    .into_iter()
                    .chain(consumer_op.form.scans.iter().flat_map(|scan| scan.neutral.iter().copied()))
                    .chain(
                        consumer_op
                            .form
                            .reductions
                            .iter()
                            .flat_map(|reduction| reduction.neutral.iter().copied()),
                    );
                if semantic_roots.into_iter().any(|root| {
                    graph_ops::pure_depends_on(graph, root, producer_result)
                        && !projected_roots.contains(&root)
                }) {
                    continue;
                }
                if !fusion_screma::can_fuse_vertical(inner, &producer_op.form, &consumer_op.form, &routes) {
                    continue;
                }
                let consumer_count = oracle.value_consumer_count(producer_id);
                if consumer_count > 1 && !all_value_consumers_are_maps {
                    continue;
                }
                let retained_producer_outputs = retained_producer_outputs(
                    graph,
                    block_id,
                    producer_index,
                    consumer_index,
                    producer_result,
                    producer_op,
                );
                let retains_unplaced_fresh = retained_producer_outputs.iter().any(|field| {
                    matches!(
                        producer_op.form.result_id(*field),
                        Some(screma::ResultId::Post(_))
                    ) && producer_op
                        .destination(*field)
                        .is_some_and(|destination| destination.is_unplaced_fresh())
                });
                if retains_unplaced_fresh && consumer_op.form.operator_input_count() > 0 {
                    continue;
                }
                let retains_placed_array = retained_producer_outputs.iter().any(|field| {
                    matches!(
                        producer_op.form.result_id(*field),
                        Some(screma::ResultId::Post(_))
                    ) && producer_op
                        .destination(*field)
                        .is_some_and(|destination| !destination.is_unplaced())
                });
                let consumer_has_unplaced_array = (0..consumer_op.result_count()).any(|field| {
                    matches!(consumer_op.form.result_id(field), Some(screma::ResultId::Post(_)))
                        && consumer_op
                            .destination(field)
                            .is_some_and(|destination| destination.is_unplaced())
                });
                // A fused map with both externally placed and unplaced array
                // results has no parallel lowering recipe. Keeping these
                // operations separate also leaves the consumer available for
                // fusion into a later output envelope.
                if retains_placed_array && consumer_has_unplaced_array {
                    continue;
                }
                return Some(Candidate {
                    site,
                    block: block_id,
                    producer: producer_index,
                    consumer: consumer_index,
                    routes,
                    retained_producer_outputs,
                });
            }
        }
    }
    None
}

fn retained_producer_outputs(
    graph: &EGraph,
    producer_block: BlockId,
    producer_index: usize,
    consumer_index: usize,
    producer_result: crate::egir::types::NodeId,
    producer: &screma::Op<crate::egir::types::Semantic>,
) -> Vec<usize> {
    let projects = graph
        .nodes
        .iter()
        .filter_map(|(node, definition)| {
            let crate::egir::types::ENode::Pure {
                op: crate::egir::types::PureOp::Project { index },
                operands,
            } = &definition.kind
            else {
                return None;
            };
            (operands.first() == Some(&producer_result)).then_some((node, *index as usize))
        })
        .collect::<Vec<_>>();

    (0..producer.result_count())
        .filter(|field| {
            if producer.destination(*field).is_some_and(|destination| !destination.is_unplaced()) {
                return true;
            }
            let field_projects = projects
                .iter()
                .filter_map(|(project, project_field)| (*project_field == *field).then_some(*project))
                .collect::<Vec<_>>();
            if field_projects.is_empty() {
                return false;
            }
            for (block_id, block) in &graph.skeleton.blocks {
                for (index, effect) in block.side_effects.iter().enumerate() {
                    if block_id == producer_block && (index == producer_index || index == consumer_index) {
                        continue;
                    }
                    if effect.referenced_nodes().any(|root| {
                        graph_ops::projection_index(graph, root, producer_result) == Some(*field)
                            || field_projects
                                .iter()
                                .any(|project| graph_ops::pure_depends_on(graph, root, *project))
                    }) {
                        return true;
                    }
                }
                if block.term.referenced_nodes().into_iter().any(|root| {
                    graph_ops::projection_index(graph, root, producer_result) == Some(*field)
                        || field_projects
                            .iter()
                            .any(|project| graph_ops::pure_depends_on(graph, root, *project))
                }) {
                    return true;
                }
            }
            false
        })
        .collect()
}
pub(super) fn apply(inner: Segmented, candidate: Candidate) -> Segmented {
    let (graph, span, scope) = graph_and_span(&inner, candidate.site);
    let outer_types =
        graph.nodes.iter().map(|(node, data)| (node, data.ty.clone())).collect::<LookupMap<_, _>>();
    let producer = horizontal::extract_screma(graph, candidate.block, candidate.producer);
    let consumer = horizontal::extract_screma(graph, candidate.block, candidate.consumer);
    let mut interner = inner.data.region_interner.clone();
    let mut context = fusion_screma::Context {
        program: &inner,
        interner: &mut interner,
        scope: &scope,
        span,
        outer_types: &outer_types,
    };
    let normalized = fusion_screma::fuse_vertical(
        &mut context,
        fusion_screma::Source {
            input_nodes: &producer.input_nodes,
            inputs: &producer.op.inputs,
            form: &producer.op.form,
        },
        fusion_screma::Source {
            input_nodes: &consumer.input_nodes,
            inputs: &consumer.op.inputs,
            form: &consumer.op.form,
        },
        &candidate.routes,
        &candidate.retained_producer_outputs,
    )
    .expect("analyzed SuperScrema no longer normalizes");

    let mut producer_mapping = vec![usize::MAX; producer.result_types.len()];
    let mut consumer_mapping = vec![usize::MAX; consumer.result_types.len()];
    let mut result_state = Vec::with_capacity(normalized.outputs.len());
    let mut result_types = Vec::with_capacity(normalized.outputs.len());
    let mut output_nodes = Vec::with_capacity(normalized.outputs.len());
    for (fused_field, origin) in normalized.outputs.iter().copied().enumerate() {
        let (source_field, source_state, source_types, source_outputs, mapping) = match origin {
            fusion_screma::OutputOrigin::Producer(field) => (
                field,
                &producer.op.result_state,
                &producer.result_types,
                &producer.output_nodes,
                &mut producer_mapping,
            ),
            fusion_screma::OutputOrigin::Consumer(field) => (
                field,
                &consumer.op.result_state,
                &consumer.result_types,
                &consumer.output_nodes,
                &mut consumer_mapping,
            ),
        };
        mapping[source_field] = fused_field;
        result_state.push(source_state[source_field]);
        result_types.push(source_types[source_field].clone());
        output_nodes.push(source_outputs[source_field]);
    }

    debug_assert!(consumer_mapping.iter().all(|field| *field != usize::MAX));

    let mut output_slots = producer.output_slots.clone();
    output_slots.extend(consumer.output_slots.iter().copied());
    output_slots.sort_unstable();
    output_slots.dedup();
    let resources = SegResourceAccess::merge(&producer.resources, &consumer.resources);
    let fused_op = screma::Op {
        inputs: normalized.inputs,
        form: normalized.form,
        result_state,
        state: screma::SemanticState::Segmented {
            space: producer.space,
            placement: if producer.placement == screma::Placement::Kernel
                || consumer.placement == screma::Placement::Kernel
            {
                screma::Placement::Kernel
            } else {
                screma::Placement::LaneLocal
            },
            output_slots,
            resources,
        },
    };
    debug_assert!(
        fused_op.validate().is_ok(),
        "invalid vertically fused Screma: {:?}",
        fused_op.validate()
    );

    let mut operands = SmallVec::new();
    operands.extend(normalized.input_nodes);
    operands.extend(output_nodes.into_iter().flatten());
    let synthesized = normalized.synthesized;
    let producer_result = producer.result;
    let consumer_result = consumer.result;

    let consumer_result_types = consumer.result_types;
    let consumer_id = consumer.id;
    let site = candidate.site;
    let rebuilt = inner.rewrite_body(site, |body| {
        let rewrite = |graph: &mut EGraph| {
            let tuple_type = Type::Constructed(TypeName::Tuple(result_types.len()), result_types.clone());
            let fused_result = graph.alloc_side_effect_result(tuple_type);
            if producer_mapping.iter().any(|field| *field != usize::MAX) {
                let retained_mapping = producer_mapping
                    .iter()
                    .map(|field| (*field != usize::MAX).then_some(*field))
                    .collect::<Vec<_>>();
                support::retarget_projects(graph, producer_result, fused_result, &retained_mapping);
            }
            horizontal::reproject_fields(
                graph,
                consumer_result,
                fused_result,
                &consumer_mapping,
                &consumer_result_types,
            );

            let block = &mut graph.skeleton.blocks[candidate.block];
            let effects = splice_effect_tokens(
                block.side_effects[candidate.producer].effects,
                block.side_effects[candidate.consumer].effects,
            );
            let consumer = &mut block.side_effects[candidate.consumer];
            consumer.kind = SideEffectKind::Soac(SoacEffect(consumer_id, Soac::Screma(fused_op.clone())));
            consumer.operand_nodes = operands.clone();
            consumer.result = Some(fused_result);
            consumer.effects = effects;
            block.side_effects.remove(candidate.producer);
        };
        support::rewrite_body_graph(body, rewrite)
    });
    rebuilt.extend_functions(synthesized).map_data(|data| CoreProgramData {
        region_interner: interner,
        ..data
    })
}
