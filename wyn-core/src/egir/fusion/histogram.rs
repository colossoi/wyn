//! Futhark-style map-to-histogram vertical fusion.
//!
//! A pure map whose complete result is consumed as histogram inputs is folded
//! into the histogram element lambda. The histogram remains the anchored
//! side-effect, so destination ordering and overwrite semantics are unchanged.

use super::map_anchor::Candidate;
use super::{map_anchor, FusionInput};
use crate::egir::reify::Segmented;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::soac::hist;
use crate::egir::types::{SideEffectKind, Soac, SoacEffect};

pub(super) fn analyze(inner: &Segmented, oracle: &SemanticGraph) -> Option<Candidate> {
    map_anchor::analyze(inner, oracle, |effect| {
        let SideEffectKind::Soac(SoacEffect(_, Soac::Hist(op))) = &effect.kind else {
            return None;
        };
        let hist::SemanticState::Segmented(_) = &op.state else {
            return None;
        };
        let input_count = op.inputs.len();
        (effect.operands.len() == input_count).then_some(input_count)
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
    let SideEffectKind::Soac(SoacEffect(consumer_id, Soac::Hist(mut consumer_op))) = consumer_effect.kind
    else {
        return Err(super::FusionError::InvalidCandidate(
            "map-to-histogram consumer changed kind after candidate analysis".to_owned(),
        ));
    };
    let input_count = consumer_op.inputs.len();
    let Some(input_operands) = consumer_effect.operands.get(..input_count) else {
        return Err(super::FusionError::InvalidCandidate(
            "histogram has fewer operands than input types".to_owned(),
        ));
    };
    let input_nodes = input_operands.iter().map(|operand| operand.value()).collect::<Option<Vec<_>>>();
    let Some(input_nodes) = input_nodes else {
        return Err(super::FusionError::InvalidCandidate(
            "histogram input uses the place channel during semantic fusion".to_owned(),
        ));
    };
    let Some(inputs) = FusionInput::join(&input_nodes, &consumer_op.inputs) else {
        return Err(super::FusionError::InvalidCandidate(
            "histogram input nodes and types have different lengths".to_owned(),
        ));
    };
    let Some(composition) = map_anchor::compose(&inner, &candidate, &inputs, &consumer_op.form.bucket)?
    else {
        return Err(super::FusionError::InvalidCandidate(
            "map-to-histogram composition failed after candidate analysis".to_owned(),
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

    consumer_op.inputs = inputs.iter().map(|input| input.ty.clone()).collect();
    consumer_op.form.bucket = lambda;
    if candidate.routes.iter().any(|route| route.consumer_input == 0) {
        consumer_op.state = hist::SemanticState::Segmented(producer_space);
    }

    map_anchor::finish(
        inner,
        candidate,
        consumer_id,
        Soac::Hist(consumer_op),
        inputs,
        synthesized,
        identities,
    )
}
