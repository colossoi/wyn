//! Same-space producer/consumer fusion for canonical Scremas.
//!
//! A pure map producer can be folded into any downstream Screma by composing
//! its pre-lambda with the consumer's pre-lambda. This preserves the consumer's
//! single collective barrier and never reconstructs the former lane graph.

use smallvec::SmallVec;

use super::screma as fusion_screma;
use super::{graph_and_span, producer_is_used_only_by};
use crate::egir::graph_ops;
use crate::egir::ir::{splice_effect_tokens, Body, BodySite};
use crate::egir::program::CoreProgramData;
use crate::egir::reify::Segmented;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::soac::screma;
use crate::egir::types::{EGraph, ResourceAccess, SegResourceAccess, SideEffectKind, Soac, SoacEffect};
use crate::flow::BlockId;
use crate::LookupMap;

#[derive(Clone)]
pub(crate) struct Candidate {
    site: BodySite,
    block: BlockId,
    producer: usize,
    consumer: usize,
    consumer_inputs: Vec<usize>,
    producer_output: usize,
}

pub(super) fn analyze(inner: &Segmented, oracle: &SemanticGraph) -> Option<Candidate> {
    for (index, entry) in inner.entry_points.iter().enumerate() {
        if let Some(candidate) = find_in_graph(&entry.graph, BodySite::Entry(index), oracle) {
            return Some(candidate);
        }
    }
    for function in &inner.functions {
        if let Some(candidate) = find_in_graph(&function.graph, BodySite::Function(function.region), oracle)
        {
            return Some(candidate);
        }
    }
    None
}

fn find_in_graph(graph: &EGraph, site: BodySite, oracle: &SemanticGraph) -> Option<Candidate> {
    for (block_id, block) in &graph.skeleton.blocks {
        for producer_index in 0..block.side_effects.len().saturating_sub(1) {
            let producer = &block.side_effects[producer_index];
            let SideEffectKind::Soac(SoacEffect(producer_id, Soac::Screma(producer_op))) = &producer.kind
            else {
                continue;
            };
            let screma::SemanticState::Segmented {
                placement,
                output_slots,
                resources,
                ..
            } = producer_op.semantic_state()
            else {
                continue;
            };
            if *placement != screma::Placement::LaneLocal
                || !output_slots.is_empty()
                || resources.iter().any(|resource| resource.access != ResourceAccess::Read)
                || producer_op.result_state.iter().any(|result| !result.destination.is_unplaced())
            {
                continue;
            }
            let Some(producer_result) = producer.result else {
                continue;
            };
            if oracle.value_consumer_count(producer_id) != 1 {
                continue;
            }

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
                if consumer.result.is_none()
                    || !fusion_screma::can_fuse_vertical(&producer_op.form, &consumer_op.form)
                {
                    continue;
                }
                if oracle.conflicts(producer_id, consumer_id)
                    && !matches!(
                        (producer.effects, consumer.effects),
                        (Some((_, producer_out)), Some((consumer_in, _))) if producer_out == consumer_in
                    )
                {
                    continue;
                }
                if resources.iter().any(|producer_resource| {
                    consumer_resources.iter().any(|consumer_resource| {
                        producer_resource.resource == consumer_resource.resource
                            && (producer_resource.access != ResourceAccess::Read
                                || consumer_resource.access != ResourceAccess::Read)
                    })
                }) {
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

                let consumer_input_count = consumer_op.inputs.len();
                let projected = consumer.operand_nodes[..consumer_input_count]
                    .iter()
                    .enumerate()
                    .filter_map(|(input, &operand)| {
                        graph_ops::projection_index(graph, operand, producer_result)
                            .map(|field| (input, field))
                    })
                    .collect::<Vec<_>>();
                let Some(&(_, producer_field)) = projected.first() else {
                    continue;
                };
                let Some(screma::ResultId::Post(producer_output)) =
                    producer_op.form.result_id(producer_field)
                else {
                    continue;
                };
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
                if projected.iter().any(|(_, field)| *field != producer_field)
                    || semantic_roots.into_iter().any(|root| {
                        graph_ops::pure_depends_on(graph, root, producer_result)
                            && !projected_roots.contains(&root)
                    })
                    || !producer_is_used_only_by(
                        graph,
                        block_id,
                        producer_index,
                        consumer_index,
                        producer_result,
                    )
                {
                    continue;
                }
                return Some(Candidate {
                    site,
                    block: block_id,
                    producer: producer_index,
                    consumer: consumer_index,
                    consumer_inputs: projected.into_iter().map(|(input, _)| input).collect(),
                    producer_output,
                });
            }
        }
    }
    None
}

pub(super) fn apply(inner: Segmented, candidate: Candidate) -> Segmented {
    let (graph, span, scope) = graph_and_span(&inner, candidate.site);
    let outer_types =
        graph.nodes.iter().map(|(node, data)| (node, data.ty.clone())).collect::<LookupMap<_, _>>();
    let producer_effect = graph.skeleton.blocks[candidate.block].side_effects[candidate.producer].clone();
    let consumer_effect = graph.skeleton.blocks[candidate.block].side_effects[candidate.consumer].clone();
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(producer_op))) = &producer_effect.kind else {
        unreachable!();
    };
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(consumer_op))) = &consumer_effect.kind else {
        unreachable!();
    };

    let producer_input_count = producer_op.inputs.len();
    let consumer_input_count = consumer_op.inputs.len();
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
            input_nodes: &producer_effect.operand_nodes[..producer_input_count],
            inputs: &producer_op.inputs,
            form: &producer_op.form,
        },
        fusion_screma::Source {
            input_nodes: &consumer_effect.operand_nodes[..consumer_input_count],
            inputs: &consumer_op.inputs,
            form: &consumer_op.form,
        },
        &candidate.consumer_inputs,
        candidate.producer_output,
    )
    .expect("analyzed SuperScrema no longer normalizes");
    debug_assert!(normalized
        .outputs
        .iter()
        .enumerate()
        .all(|(field, origin)| { *origin == fusion_screma::OutputOrigin::Consumer(field) }));

    let mut fused_op = consumer_op.clone();
    fused_op.inputs = normalized.inputs;
    fused_op.form = normalized.form;
    let producer_resources = match producer_op.semantic_state() {
        screma::SemanticState::Segmented { resources, space, .. } => (resources.clone(), space.clone()),
        screma::SemanticState::Serial => unreachable!(),
    };
    let screma::SemanticState::Segmented { space, resources, .. } = fused_op.semantic_state_mut() else {
        unreachable!();
    };
    *space = producer_resources.1;
    *resources = SegResourceAccess::merge(resources, &producer_resources.0);
    debug_assert!(
        fused_op.validate().is_ok(),
        "invalid vertically fused Screma: {:?}",
        fused_op.validate()
    );

    let tail = consumer_effect.operand_nodes[consumer_input_count..].to_vec();
    let operands = normalized.input_nodes.into_iter().chain(tail).collect::<SmallVec<[_; 4]>>();
    let synthesized = normalized.synthesized;
    let site = candidate.site;
    let rebuilt = inner.rewrite_body(site, |body| {
        let rewrite = |graph: &mut EGraph| {
            let block = &mut graph.skeleton.blocks[candidate.block];
            let effects = splice_effect_tokens(
                block.side_effects[candidate.producer].effects,
                block.side_effects[candidate.consumer].effects,
            );
            let consumer = &mut block.side_effects[candidate.consumer];
            let SideEffectKind::Soac(SoacEffect(id, _)) = consumer.kind else {
                unreachable!();
            };
            consumer.kind = SideEffectKind::Soac(SoacEffect(id, Soac::Screma(fused_op.clone())));
            consumer.operand_nodes = operands.clone();
            consumer.effects = effects;
            block.side_effects.remove(candidate.producer);
        };
        match body {
            Body::Entry(mut entry) => {
                rewrite(&mut entry.graph);
                Body::Entry(entry)
            }
            Body::Function(mut function) => {
                rewrite(&mut function.graph);
                Body::Function(function)
            }
            Body::Constant(_) => unreachable!("vertical fusion never targets constants"),
        }
    });
    rebuilt.extend_functions(synthesized).map_data(|data| CoreProgramData {
        region_interner: interner,
        ..data
    })
}
