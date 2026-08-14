//! Futhark-style map-to-histogram vertical fusion.
//!
//! A pure map whose complete result is consumed as histogram inputs is folded
//! into the histogram element lambda. The histogram remains the anchored
//! side-effect, so destination ordering and overwrite semantics are unchanged.

use super::map_anchor;
use crate::egir::reify::Segmented;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::soac::hist;
use crate::egir::types::{SideEffectKind, Soac, SoacEffect};

pub(super) type Candidate = map_anchor::Candidate;

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

pub(super) fn apply(inner: Segmented, candidate: Candidate) -> Segmented {
    let graph = inner.body_graph(candidate.site).expect("semantic fusion body");
    let consumer_effect = graph.skeleton.blocks[candidate.block].side_effects[candidate.consumer].clone();
    let SideEffectKind::Soac(SoacEffect(consumer_id, Soac::Hist(mut consumer_op))) = consumer_effect.kind
    else {
        unreachable!();
    };
    let input_count = consumer_op.inputs.len();
    let input_nodes = consumer_effect.operands[..input_count]
        .iter()
        .map(|operand| operand.value().expect("Hist input uses the value or view channel"))
        .collect::<Vec<_>>();
    let composition = map_anchor::compose(
        &inner,
        &candidate,
        &input_nodes,
        &consumer_op.inputs,
        &consumer_op.form.bucket,
    )
    .expect("analyzed map-to-histogram fusion no longer composes");
    let map_anchor::Composition {
        producer_space,
        normalized,
        identities,
    } = composition;
    let super::screma::NormalizedLambda {
        input_nodes,
        inputs,
        lambda,
        synthesized,
    } = normalized;

    consumer_op.inputs = inputs;
    consumer_op.form.bucket = lambda;
    if candidate.routes.iter().any(|route| route.consumer_input == 0) {
        consumer_op.state = hist::SemanticState::Segmented(producer_space);
    }

    map_anchor::finish(
        inner,
        candidate,
        consumer_id,
        Soac::Hist(consumer_op),
        input_nodes,
        synthesized,
        identities,
    )
}
