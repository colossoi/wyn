//! Fuse pure map producers into the callable map of a Filter envelope.
//!
//! The Filter owns compaction and output storage.  Folding the producer's
//! complete result into its canonical map lambda removes the intermediate
//! array without weakening the Filter's effect or placement semantics.

use std::collections::HashSet;

use polytype::Type;
use smallvec::SmallVec;

use super::{graph_and_span, horizontal, screma as fusion_screma, support};
use crate::ast::TypeName;
use crate::egir::graph_ops;
use crate::egir::ir::{splice_effect_tokens, BodySite};
use crate::egir::program::CoreProgramData;
use crate::egir::reify::Segmented;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::soac::{filter, screma};
use crate::egir::types::{EGraph, ResourceAccess, SideEffectKind, Soac, SoacEffect};
use crate::flow::BlockId;
use crate::LookupMap;

pub(super) struct Candidate {
    site: BodySite,
    block: BlockId,
    producer: usize,
    consumer: usize,
    routes: Vec<fusion_screma::InputRoute>,
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
                output_slots,
                resources,
                ..
            } = producer_op.semantic_state()
            else {
                continue;
            };
            if !producer_op.is_map()
                || !output_slots.is_empty()
                || resources.iter().any(|resource| resource.access != ResourceAccess::Read)
                || producer_op.result_state.iter().any(|result| !result.destination.is_unplaced())
            {
                continue;
            }
            let Some(producer_result) = producer.result else {
                continue;
            };

            for consumer_index in producer_index + 1..block.side_effects.len() {
                let consumer = &block.side_effects[consumer_index];
                let SideEffectKind::Soac(SoacEffect(consumer_id, Soac::Filter(consumer_op))) =
                    &consumer.kind
                else {
                    continue;
                };
                if consumer.result.is_none() || oracle.conflicts(producer_id, consumer_id) {
                    continue;
                }
                if !((producer_index + 1)..consumer_index).all(|index| {
                    let effect = &block.side_effects[index];
                    match &effect.kind {
                        SideEffectKind::Soac(SoacEffect(intervening, _)) => {
                            !oracle.conflicts(producer_id, intervening)
                        }
                        _ => effect.effects.is_none(),
                    }
                }) {
                    continue;
                }

                let input_count = consumer_op.body.inputs.len();
                if consumer.operand_nodes.len() < input_count {
                    continue;
                }
                let input_nodes = &consumer.operand_nodes[..input_count];
                let routes = input_nodes
                    .iter()
                    .enumerate()
                    .filter_map(|(consumer_input, &operand)| {
                        let field = graph_ops::projection_index(graph, operand, producer_result)?;
                        let screma::ResultId::Post(producer_post_output) =
                            producer_op.form.result_id(field)?
                        else {
                            return None;
                        };
                        Some(fusion_screma::InputRoute {
                            consumer_input,
                            producer_post_output,
                        })
                    })
                    .collect::<Vec<_>>();
                let routed_outputs =
                    routes.iter().map(|route| route.producer_post_output).collect::<HashSet<_>>();
                if routes.is_empty()
                    || routed_outputs.len() != producer_op.form.post.result_types.len()
                    || !support::result_used_only_by_effect_pair(
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
                    routes,
                });
            }
        }
    }
    None
}

pub(super) fn apply(inner: Segmented, candidate: Candidate) -> Segmented {
    let (graph, span, scope) = graph_and_span(&inner, candidate.site);
    let outer_types = graph
        .nodes
        .iter()
        .map(|(node, data)| (node, data.ty.clone()))
        .collect::<LookupMap<_, Type<TypeName>>>();
    let producer = horizontal::extract_screma(graph, candidate.block, candidate.producer);
    let consumer_effect = graph.skeleton.blocks[candidate.block].side_effects[candidate.consumer].clone();
    let SideEffectKind::Soac(SoacEffect(consumer_id, Soac::Filter(mut consumer_op))) = consumer_effect.kind
    else {
        unreachable!();
    };
    let input_count = consumer_op.body.inputs.len();
    let input_nodes = consumer_effect.operand_nodes[..input_count].to_vec();
    let consumer_lambda = consumer_op.body.map.clone();
    let mut interner = inner.data.region_interner.clone();
    let mut context = fusion_screma::Context {
        program: &inner,
        interner: &mut interner,
        scope: &scope,
        span,
        outer_types: &outer_types,
    };
    let normalized = fusion_screma::fuse_map_into_lambda(
        &mut context,
        fusion_screma::Source {
            input_nodes: &producer.input_nodes,
            inputs: &producer.op.inputs,
            form: &producer.op.form,
        },
        fusion_screma::LambdaSource {
            input_nodes: &input_nodes,
            inputs: &consumer_op.body.inputs,
            lambda: &consumer_lambda,
        },
        &candidate.routes,
    )
    .expect("analyzed map-to-Filter fusion no longer composes");
    consumer_op.body.inputs = normalized.inputs;
    consumer_op.body.map = normalized.lambda;
    consumer_op.state.space = producer.space.clone();
    if let filter::Output::Local { destination, .. } = &mut consumer_op.state.storage {
        if destination.is_unplaced_unique_input() {
            destination.make_fresh();
        }
    }
    debug_assert!(consumer_op.body.validate().is_ok());
    let operands = SmallVec::from_vec(normalized.input_nodes);
    let synthesized = normalized.synthesized;

    let rebuilt = inner.rewrite_body(candidate.site, |body| {
        let rewrite = |graph: &mut EGraph| {
            let block = &mut graph.skeleton.blocks[candidate.block];
            let effects = splice_effect_tokens(
                block.side_effects[candidate.producer].effects,
                block.side_effects[candidate.consumer].effects,
            );
            let consumer = &mut block.side_effects[candidate.consumer];
            consumer.kind =
                SideEffectKind::Soac(SoacEffect(consumer_id, Soac::Filter(consumer_op.clone())));
            consumer.operand_nodes = operands.clone();
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
