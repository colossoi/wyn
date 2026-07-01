//! Producer→consumer (vertical) fusion: when a `SegMap` producer's array result
//! feeds a *single* same-space consumer SegOp, inline the producer's per-element
//! body into the consumer so the intermediate array is never materialized.
//!
//! Multi-consumer producer arrays are deliberately left in place — the
//! allocation manifest materializes those into a real buffer. Single-consumer is
//! the only case where eliminating the array is unconditionally a win, and it is
//! the explicit hand-off contract with Phase A.
//!
//! One fusion per call; the driver rebuilds the DAG between calls.

use super::legality::SemanticGraph;
use super::space::seg_space_fusable;
use crate::egir::program::{EgirInner, SemanticOpId};
use crate::egir::types::{
    EGraph, EgirSoac, NodeId, SegOpKind, SegResourceAccessKind, SideEffectKind, SoacDestination,
};
use crate::ssa::framework::BlockId;

/// Find one legal producer→consumer pair anywhere in the program and fuse it.
/// Returns whether a fusion happened.
pub fn fuse_producer_into_consumer(inner: &mut EgirInner, oracle: &SemanticGraph) -> bool {
    // Region composition needs the program-level interner, so operate over
    // `inner` rather than an isolated graph.
    for idx in 0..inner.entry_points.len() {
        let scope = inner.entry_points[idx].name.clone();
        if let Some(candidate) = find_candidate(&inner.entry_points[idx].graph, &scope, oracle) {
            return apply_fusion(inner, FusionSite::Entry(idx), candidate);
        }
    }
    for idx in 0..inner.functions.len() {
        let scope = inner.functions[idx].name.clone();
        if let Some(candidate) = find_candidate(&inner.functions[idx].graph, &scope, oracle) {
            return apply_fusion(inner, FusionSite::Function(idx), candidate);
        }
    }
    false
}

enum FusionSite {
    Entry(usize),
    Function(usize),
}

/// A legal producer→consumer pair within one graph: the producer/consumer
/// side-effect indices, the consumer input slot that reads the producer, and the
/// producer's result node.
struct Candidate {
    producer: usize,
    consumer: usize,
    consumer_input: usize,
    producer_result: NodeId,
    block: BlockId,
}

fn find_candidate(graph: &EGraph, scope: &str, oracle: &SemanticGraph) -> Option<Candidate> {
    let block_ids: Vec<BlockId> = graph.skeleton.blocks.iter().map(|(id, _)| id).collect();
    for block_id in block_ids {
        let block = &graph.skeleton.blocks[block_id];
        for (p_idx, producer) in block.side_effects.iter().enumerate() {
            // Producer must be a pure, single-consumer, output-free SegMap.
            let SideEffectKind::Soac(EgirSoac::Seg {
                kind: SegOpKind::SegMap,
                placement: p_placement,
                space: p_space,
                map_destinations,
                output_slots,
                resources,
                ..
            }) = &producer.kind
            else {
                continue;
            };
            let Some(p_result) = producer.result else {
                continue;
            };
            let writes_output = !output_slots.is_empty()
                || !map_destinations.iter().all(|d| matches!(d, SoacDestination::Fresh))
                || resources.iter().any(|r| r.access != SegResourceAccessKind::Read);
            if writes_output {
                continue;
            }
            let p_id = SemanticOpId {
                scope: scope.to_string(),
                result: p_result,
            };
            if oracle.value_consumer_count(&p_id) != 1 {
                continue;
            }

            // The unique consumer must be a same-space, non-conflicting SegOp in
            // this block that reads `p_result` as one of its input operands.
            for (c_idx, consumer) in block.side_effects.iter().enumerate() {
                if c_idx == p_idx {
                    continue;
                }
                let SideEffectKind::Soac(EgirSoac::Seg {
                    placement: c_placement,
                    space: c_space,
                    input_array_types,
                    ..
                }) = &consumer.kind
                else {
                    continue;
                };
                if p_placement != c_placement || !seg_space_fusable(p_space, c_space) {
                    continue;
                }
                let Some(c_result) = consumer.result else {
                    continue;
                };
                let c_id = SemanticOpId {
                    scope: scope.to_string(),
                    result: c_result,
                };
                if oracle.conflicts(&p_id, &c_id) {
                    continue;
                }
                let n_inputs = input_array_types.len();
                if let Some(consumer_input) =
                    consumer.operand_nodes[..n_inputs].iter().position(|&n| n == p_result)
                {
                    return Some(Candidate {
                        producer: p_idx,
                        consumer: c_idx,
                        consumer_input,
                        producer_result: p_result,
                        block: block_id,
                    });
                }
            }
        }
    }
    None
}

/// Inline the producer body into the consumer.
///
/// The rewrite composes regions: the consumer input slot `consumer_input`
/// (which read `producer_result`) is replaced by the producer's inputs, and each
/// consumer body that read that slot is redirected to a synthesized region whose
/// body is `consumer_body(producer_body(x), ...)`, interned through
/// `inner.region_interner`. The producer side-effect and its now-dead
/// intermediate are then removed.
///
/// This composition step is the remaining work; the candidate analysis above —
/// the DAG-driven legality that makes it safe — is complete. Until the rewrite
/// lands, recognizing a candidate is a no-op so the pipeline stays correct
/// (the producer array is simply materialized as before).
fn apply_fusion(inner: &mut EgirInner, site: FusionSite, candidate: Candidate) -> bool {
    let _graph = match site {
        FusionSite::Entry(idx) => &mut inner.entry_points[idx].graph,
        FusionSite::Function(idx) => &mut inner.functions[idx].graph,
    };
    let Candidate {
        producer,
        consumer,
        consumer_input,
        producer_result,
        block,
    } = candidate;
    let _ = (producer, consumer, consumer_input, producer_result, block);
    // TODO(milestone-5 F3): synthesize the composed region via
    // `inner.region_interner`, rewire the consumer's input list and map bodies,
    // and drop the producer side-effect. Returning false keeps the producer
    // array materialized (correct, unfused) until then.
    false
}
