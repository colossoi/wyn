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

use super::{graph_and_span, horizontal, screma as fusion_screma, support, FusionEffect, FusionInput};
use crate::ast::TypeName;
use crate::egir::graph_ops;
use crate::egir::ir::splice_effect_tokens;
use crate::egir::program::{Func, ProgramIdentities, SemanticProgramData};
use crate::egir::reify::Segmented;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::soac::screma;
use crate::egir::types::{
    EGraph, ResourceAccess, SegSpace, Semantic, SideEffect, SideEffectKind, Soac, SoacEffect,
};
use crate::LookupMap;

pub(super) struct Candidate {
    pub producer: FusionEffect,
    pub consumer: FusionEffect,
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
    super::bodies(inner).find_map(|(_, graph, _)| find_in_graph(graph, oracle, &consumer_inputs))
}

fn find_in_graph(
    graph: &EGraph,
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

                let Some(input_operands) = consumer.operands.get(..input_count) else {
                    continue;
                };
                let routes = input_operands
                    .iter()
                    .enumerate()
                    .filter_map(|(consumer_input, operand)| {
                        let operand = operand.value()?;
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
                    producer: FusionEffect(*producer_id),
                    consumer: FusionEffect(*consumer_id),
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
    consumer_inputs: &[FusionInput],
    consumer_lambda: &screma::Lambda,
) -> super::FusionResult<Option<Composition>> {
    let (producer_location, _) = super::resolve_pair(inner, candidate.producer, candidate.consumer)?;
    let (graph, span, scope) = graph_and_span(inner, producer_location.body)?;
    let outer_types = graph
        .nodes
        .iter()
        .map(|(node, data)| (node, data.ty.clone()))
        .collect::<LookupMap<_, Type<TypeName>>>();
    let producer = horizontal::extract_screma(graph, producer_location.block, producer_location.index)?;
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
            inputs: &producer.inputs,
            form: &producer.op.form,
        },
        fusion_screma::LambdaSource {
            inputs: consumer_inputs,
            lambda: consumer_lambda,
        },
        &candidate.routes,
    );
    let Some(normalized) = normalized else {
        return Ok(None);
    };
    Ok(Some(Composition {
        producer_space: producer.space,
        normalized,
        identities,
    }))
}

pub(super) fn finish(
    inner: Segmented,
    candidate: Candidate,
    consumer_id: egir::program::SemanticOpId,
    consumer_op: Soac<Semantic>,
    inputs: Vec<FusionInput>,
    synthesized: Vec<Func<Semantic>>,
    identities: ProgramIdentities,
) -> super::FusionResult<Segmented> {
    let (producer, consumer) = super::resolve_pair(&inner, candidate.producer, candidate.consumer)?;
    let rebuilt = inner.try_rewrite_body(producer.body, |body| {
        support::try_rewrite_body_graph(body, |graph| {
            let operands =
                inputs.iter().map(|input| graph.operand_ref(input.node)).collect::<SmallVec<_>>();
            let Some(block) = graph.skeleton.blocks.get_mut(producer.block) else {
                return Err(super::FusionError::MissingEffect(candidate.producer.0));
            };
            let Some(producer_effect) = block.side_effects.get(producer.index) else {
                return Err(super::FusionError::MissingEffect(candidate.producer.0));
            };
            let Some(consumer_effect) = block.side_effects.get(consumer.index) else {
                return Err(super::FusionError::MissingEffect(candidate.consumer.0));
            };
            let effects = splice_effect_tokens(producer_effect.effects, consumer_effect.effects);
            let Some(consumer_effect) = block.side_effects.get_mut(consumer.index) else {
                return Err(super::FusionError::MissingEffect(candidate.consumer.0));
            };
            consumer_effect.kind = SideEffectKind::Soac(SoacEffect(consumer_id, consumer_op.clone()));
            consumer_effect.operands = operands;
            consumer_effect.effects = effects;
            block.side_effects.remove(producer.index);
            Ok(())
        })
    })?;
    Ok(rebuilt.extend_functions(synthesized).map_data(|data| SemanticProgramData { identities, ..data }))
}
