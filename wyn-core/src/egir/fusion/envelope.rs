//! Fuse pure map producers into the callable map of a Filter envelope.
//!
//! The Filter owns compaction and output storage. Folding the producer's
//! complete result into its canonical map lambda removes the intermediate
//! array without weakening the Filter's effect or placement semantics.

use super::map_anchor;
use super::map_anchor::Candidate;
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

pub(super) fn apply(inner: Segmented, candidate: Candidate) -> Segmented {
    let graph = inner.body_graph(candidate.site).expect("semantic fusion body");
    let consumer_effect = graph.skeleton.blocks[candidate.block].side_effects[candidate.consumer].clone();
    let SideEffectKind::Soac(SoacEffect(consumer_id, Soac::Filter(mut consumer_op))) = consumer_effect.kind
    else {
        unreachable!();
    };
    let input_count = consumer_op.body.inputs.len();
    let input_nodes = consumer_effect.operands[..input_count]
        .iter()
        .map(|operand| operand.value().expect("Filter input uses the value or view channel"))
        .collect::<Vec<_>>();
    let composition = map_anchor::compose(
        &inner,
        &candidate,
        &input_nodes,
        &consumer_op.body.inputs,
        &consumer_op.body.map,
    )
    .expect("analyzed map-to-Filter fusion no longer composes");
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

    consumer_op.body.inputs = inputs;
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
        input_nodes,
        synthesized,
        identities,
    )
}
