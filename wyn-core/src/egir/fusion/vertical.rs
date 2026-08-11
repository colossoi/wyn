//! Same-space producer/consumer fusion for canonical Scremas.
//!
//! A pure map producer can be folded into any downstream Screma by composing
//! its pre-lambda with the consumer's pre-lambda while preserving the consumer's
//! single collective barrier.

use polytype::Type;
use smallvec::{smallvec, SmallVec};

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
use crate::egir::types::{
    EGraph, ENode, NodeId, PureOp, ResourceAccess, SegResourceAccess, SideEffectKind, Soac, SoacEffect,
    SoacInputType,
};
use crate::flow::BlockId;
use crate::types::TypeExt;
use crate::LookupMap;

/// Ordered array-to-array transforms on a routed producer result. Slice is the
/// only such operation in Wyn's current pure IR; scalar `Index` transforms are
/// handled by the point-demand pass because they do not remain SOAC inputs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct InputTransform {
    slices: Vec<SliceTransform>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SliceTransform {
    op: PureOp,
    start: NodeId,
    end: NodeId,
    size: Type<TypeName>,
}

impl InputTransform {
    fn route(
        graph: &EGraph,
        operand: NodeId,
        producer_result: NodeId,
        producer_outputs: &[Option<NodeId>],
    ) -> Option<(usize, Self)> {
        let mut current = operand;
        let mut slices = Vec::new();
        loop {
            if let Some(field) = graph_ops::projection_index(graph, current, producer_result) {
                slices.reverse();
                return Some((field, Self { slices }));
            }
            if let Some(field) = producer_outputs.iter().position(|output| *output == Some(current)) {
                slices.reverse();
                return Some((field, Self { slices }));
            }
            let ENode::Pure { op, operands } = &graph.nodes.get(current)?.kind else {
                return None;
            };
            let PureOp::Intrinsic { id, .. } = op else {
                return None;
            };
            if *id != crate::builtins::catalog().known().slice {
                return None;
            }
            let [base, start, end] = operands.as_slice() else {
                return None;
            };
            slices.push(SliceTransform {
                op: op.clone(),
                start: *start,
                end: *end,
                size: graph.nodes[current].ty.array_size()?.clone(),
            });
            current = *base;
        }
    }

    fn is_identity(&self) -> bool {
        self.slices.is_empty()
    }

    fn apply(
        &self,
        graph: &mut EGraph,
        mut node: NodeId,
        input: &SoacInputType,
    ) -> Option<(NodeId, SoacInputType)> {
        let mut array = input.array.clone();
        for slice in &self.slices {
            array = array_with_outer_size(&array, &slice.size)?;
            node = graph.intern_pure(
                slice.op.clone(),
                smallvec![node, slice.start, slice.end],
                array.clone(),
                None,
            );
        }
        let mut input = input.clone();
        input.array = array;
        Some((node, input))
    }
}

fn array_with_outer_size(array: &Type<TypeName>, size: &Type<TypeName>) -> Option<Type<TypeName>> {
    match array {
        Type::Constructed(TypeName::Array, arguments) if arguments.len() >= 4 => {
            let mut arguments = arguments.clone();
            arguments[2] = size.clone();
            Some(Type::Constructed(TypeName::Array, arguments))
        }
        Type::Constructed(TypeName::Tuple(arity), fields) => Some(Type::Constructed(
            TypeName::Tuple(*arity),
            fields.iter().map(|field| array_with_outer_size(field, size)).collect::<Option<Vec<_>>>()?,
        )),
        _ => None,
    }
}

#[derive(Clone)]
pub(super) struct Candidate {
    site: BodySite,
    block: BlockId,
    producer: usize,
    consumer: usize,
    routes: Vec<fusion_screma::InputRoute>,
    transform: InputTransform,
    retained_producer_outputs: Vec<usize>,
}

pub(super) fn analyze(inner: &Segmented, oracle: &SemanticGraph) -> Option<Candidate> {
    super::bodies(inner).find_map(|(site, graph, _)| find_in_graph(inner, graph, site, oracle))
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
                let routed = consumer.operand_nodes[..consumer_input_count]
                    .iter()
                    .enumerate()
                    .filter_map(|(input, &operand)| {
                        InputTransform::route(graph, operand, producer_result, &producer_output_nodes)
                            .map(|(field, transform)| (input, field, transform))
                    })
                    .collect::<Vec<_>>();
                if routed.is_empty() {
                    continue;
                }
                let routed_inputs =
                    routed.iter().map(|(input, _, _)| *input).collect::<std::collections::HashSet<_>>();
                let has_unrouteable_input = consumer.operand_nodes[..consumer_input_count]
                    .iter()
                    .enumerate()
                    .any(|(input, &operand)| {
                        !routed_inputs.contains(&input)
                            && (graph_ops::pure_depends_on(graph, operand, producer_result)
                                || producer_output_nodes
                                    .iter()
                                    .flatten()
                                    .any(|output| graph_ops::pure_depends_on(graph, operand, *output)))
                    });
                if has_unrouteable_input {
                    continue;
                }
                let transform = routed[0].2.clone();
                if routed.iter().any(|(_, _, candidate_transform)| *candidate_transform != transform)
                    || (!transform.is_identity() && !producer_op.is_map())
                {
                    continue;
                }
                let routes = routed
                    .iter()
                    .map(|(consumer_input, producer_field, _)| {
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
                let routed_roots = routed
                    .iter()
                    .map(|(input, _, _)| consumer.operand_nodes[*input])
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
                        && !routed_roots.contains(&root)
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
                if !transform.is_identity() && !retained_producer_outputs.is_empty() {
                    continue;
                }
                return Some(Candidate {
                    site,
                    block: block_id,
                    producer: producer_index,
                    consumer: consumer_index,
                    routes,
                    transform,
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
pub(super) fn apply(mut inner: Segmented, candidate: Candidate) -> Segmented {
    let transform = candidate.transform.clone();
    let mut transformed_source = None;
    inner = inner.rewrite_body(candidate.site, |body| {
        support::rewrite_body_graph(body, |graph| {
            let (input_nodes, inputs) = {
                let effect = &graph.skeleton.blocks[candidate.block].side_effects[candidate.producer];
                let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                    unreachable!("vertical fusion selected a non-Screma producer")
                };
                (
                    effect.operand_nodes[..op.inputs.len()].to_vec(),
                    op.inputs.clone(),
                )
            };
            let (input_nodes, inputs): (Vec<_>, Vec<_>) = input_nodes
                .into_iter()
                .zip(&inputs)
                .map(|(node, input)| {
                    transform
                        .apply(graph, node, input)
                        .expect("analyzed producer input transform became invalid")
                })
                .unzip();
            transformed_source = Some((input_nodes, inputs));
        })
    });
    let (producer_input_nodes, producer_inputs) =
        transformed_source.expect("vertical fusion body was not rewritten");

    let (graph, span, scope) = graph_and_span(&inner, candidate.site);
    let outer_types =
        graph.nodes.iter().map(|(node, data)| (node, data.ty.clone())).collect::<LookupMap<_, _>>();
    let producer = horizontal::extract_screma(graph, candidate.block, candidate.producer);
    let consumer = horizontal::extract_screma(graph, candidate.block, candidate.consumer);
    let mut identities = inner.data.identities.clone();
    let mut context = fusion_screma::Context {
        program: &inner,
        identities: &mut identities,
        scope: &scope,
        span,
        outer_types: &outer_types,
    };
    let normalized = fusion_screma::fuse_vertical(
        &mut context,
        fusion_screma::Source {
            input_nodes: &producer_input_nodes,
            inputs: &producer_inputs,
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
            space: if candidate.transform.is_identity() { producer.space } else { consumer.space },
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
        identities: identities,
        ..data
    })
}
