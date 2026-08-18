//! Shared graph plumbing for fusing a pure map into an anchored SOAC.
//!
//! Filters and histograms retain their own state and result semantics, but
//! candidate discovery, map/lambda composition, and effect replacement are
//! identical. This module keeps those invariants in one place while the
//! consumer modules provide only their shape-specific adapters.

use crate::egir;
use std::collections::HashSet;

use polytype::Type;
use smallvec::SmallVec;

use super::{graph_and_span, horizontal, screma as fusion_screma, support};
use crate::ast::TypeName;
use crate::egir::graph_ops;
use crate::egir::ir::{splice_effect_tokens, BodySite};
use crate::egir::program::{CoreProgramData, Func, ProgramIdentities};
use crate::egir::reify::Segmented;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::soac::screma;
use crate::egir::types::{
    EGraph, ResourceAccess, SegSpace, Semantic, SideEffect, SideEffectKind, Soac, SoacEffect,
    SoacInputType, ValueId,
};
use crate::flow::BlockId;
use crate::LookupMap;

pub(super) struct Candidate {
    pub site: BodySite,
    pub block: BlockId,
    pub producer: usize,
    pub consumer: usize,
    pub routes: Vec<fusion_screma::InputRoute>,
}

pub(super) struct Composition {
    pub producer_space: SegSpace,
    pub normalized: fusion_screma::NormalizedLambda,
    pub identities: ProgramIdentities,
}

/// Find a pure map whose complete result feeds one anchored consumer.
///
/// `consumer_inputs` recognizes the consumer kind, validates any
/// consumer-specific state and operand-layout invariant, and returns the
/// number of leading operands that are parallel array inputs.
pub(super) fn analyze(
    inner: &Segmented,
    oracle: &SemanticGraph,
    consumer_inputs: impl Fn(&SideEffect) -> Option<usize>,
) -> Option<Candidate> {
    super::bodies(inner).find_map(|(site, graph, _)| find_in_graph(graph, site, oracle, &consumer_inputs))
}

fn find_in_graph(
    graph: &EGraph,
    site: BodySite,
    oracle: &SemanticGraph,
    consumer_inputs: &impl Fn(&SideEffect) -> Option<usize>,
) -> Option<Candidate> {
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
            {
                continue;
            }
            let Some(producer_result) = producer.value_result() else {
                continue;
            };

            for consumer_index in producer_index + 1..block.side_effects.len() {
                let consumer = &block.side_effects[consumer_index];
                let Some(input_count) = consumer_inputs(consumer) else {
                    continue;
                };
                let SideEffectKind::Soac(SoacEffect(consumer_id, _)) = &consumer.kind else {
                    unreachable!("anchored consumer recognizer accepted a non-SOAC");
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

                let routes = consumer.operands[..input_count]
                    .iter()
                    .enumerate()
                    .filter_map(|(consumer_input, operand)| {
                        let operand = operand.value().expect("anchored SOAC inputs are values or views");
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

pub(super) fn compose(
    inner: &Segmented,
    candidate: &Candidate,
    consumer_input_nodes: &[ValueId],
    consumer_inputs: &[SoacInputType],
    consumer_lambda: &screma::Lambda,
) -> Option<Composition> {
    let (graph, span, scope) = graph_and_span(inner, candidate.site);
    let outer_types = graph
        .nodes
        .iter()
        .map(|(node, data)| (node, data.ty.clone()))
        .collect::<LookupMap<_, Type<TypeName>>>();
    let producer = horizontal::extract_screma(graph, candidate.block, candidate.producer);
    let mut identities = inner.data.identities.clone();
    let mut context = fusion_screma::Context {
        program: inner,
        identities: &mut identities,
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
            input_nodes: consumer_input_nodes,
            inputs: consumer_inputs,
            lambda: consumer_lambda,
        },
        &candidate.routes,
    )?;
    Some(Composition {
        producer_space: producer.space,
        normalized,
        identities,
    })
}

pub(super) fn finish(
    inner: Segmented,
    candidate: Candidate,
    consumer_id: egir::program::SemanticOpId,
    consumer_op: Soac<Semantic>,
    input_nodes: Vec<ValueId>,
    synthesized: Vec<Func<Semantic>>,
    identities: ProgramIdentities,
) -> Segmented {
    let rebuilt = inner.rewrite_body(candidate.site, |body| {
        let rewrite = |graph: &mut EGraph| {
            let operands =
                input_nodes.iter().map(|input| graph.operand_ref(*input)).collect::<SmallVec<_>>();
            let block = &mut graph.skeleton.blocks[candidate.block];
            let effects = splice_effect_tokens(
                block.side_effects[candidate.producer].effects,
                block.side_effects[candidate.consumer].effects,
            );
            let consumer = &mut block.side_effects[candidate.consumer];
            consumer.kind = SideEffectKind::Soac(SoacEffect(consumer_id, consumer_op.clone()));
            consumer.operands = operands;
            consumer.effects = effects;
            block.side_effects.remove(candidate.producer);
        };
        support::rewrite_body_graph(body, rewrite)
    });
    rebuilt.extend_functions(synthesized).map_data(|data| CoreProgramData { identities, ..data })
}
