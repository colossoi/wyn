//! Fuse pure map producers into the callable map of a Filter envelope.
//!
//! The Filter owns compaction and output storage. Folding the producer's
//! complete result into its canonical map lambda removes the intermediate
//! array without weakening the Filter's effect or placement semantics.

use super::map_anchor::Candidate;
use super::{map_anchor, FusionInput};
use crate::egir::reify::Segmented;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::soac::filter;
use crate::egir::types::{SideEffectKind, Soac, SoacEffect};
use crate::types;

pub(super) fn analyze(inner: &Segmented, oracle: &SemanticGraph) -> Option<Candidate> {
    map_anchor::analyze(inner, oracle, |effect| {
        let SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) = &effect.kind else {
            return None;
        };
        let input_count = op.body.inputs.len();
        (effect.operands.len() >= input_count).then_some(input_count)
    })
}

pub(super) fn apply(inner: Segmented, candidate: Candidate) -> super::FusionResult<Segmented> {
    let consumer_location = candidate.consumer.resolve(&inner)?;
    let Some(graph) = inner.body_graph(consumer_location.body) else {
        return Err(super::FusionError::MissingEffect(candidate.consumer.0));
    };
    let Some(block) = graph.skeleton.blocks.get(consumer_location.block) else {
        return Err(super::FusionError::MissingEffect(candidate.consumer.0));
    };
    let Some(consumer_effect) = block.side_effects.get(consumer_location.index).cloned() else {
        return Err(super::FusionError::MissingEffect(candidate.consumer.0));
    };
    let SideEffectKind::Soac(SoacEffect(consumer_id, Soac::Filter(mut consumer_op))) = consumer_effect.kind
    else {
        return Err(super::FusionError::InvalidCandidate(
            "map-to-Filter consumer changed kind after candidate analysis".to_owned(),
        ));
    };
    let input_count = consumer_op.body.inputs.len();
    let Some(input_operands) = consumer_effect.operands.get(..input_count) else {
        return Err(super::FusionError::InvalidCandidate(
            "Filter has fewer operands than input types".to_owned(),
        ));
    };
    let input_nodes = input_operands.iter().map(|operand| operand.value()).collect::<Option<Vec<_>>>();
    let Some(input_nodes) = input_nodes else {
        return Err(super::FusionError::InvalidCandidate(
            "Filter input uses the place channel during semantic fusion".to_owned(),
        ));
    };
    let Some(inputs) = FusionInput::join(&input_nodes, &consumer_op.body.inputs) else {
        return Err(super::FusionError::InvalidCandidate(
            "Filter input nodes and types have different lengths".to_owned(),
        ));
    };
    let Some(composition) = map_anchor::compose(&inner, &candidate, &inputs, &consumer_op.body.map)? else {
        return Err(super::FusionError::InvalidCandidate(
            "map-to-Filter composition failed after candidate analysis".to_owned(),
        ));
    };
    let map_anchor::Composition {
        producer_space,
        normalized,
        identities,
    } = composition;
    let super::screma::NormalizedLambda {
        inputs,
        lambda,
        synthesized,
    } = normalized;

    consumer_op.body.inputs = inputs.iter().map(|input| input.ty.clone()).collect();
    consumer_op.body.map = lambda;
    consumer_op.state.space = producer_space;
    if let filter::Output::Local { ownership, .. } = &mut consumer_op.state.output {
        if *ownership == types::SoacOwnership::UniqueInput {
            *ownership = types::SoacOwnership::Fresh;
        }
    }
    debug_assert!(consumer_op.body.validate().is_ok());

    map_anchor::finish(
        inner,
        candidate,
        consumer_id,
        Soac::Filter(consumer_op),
        inputs,
        synthesized,
        identities,
    )
}
