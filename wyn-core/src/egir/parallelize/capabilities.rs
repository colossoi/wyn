//! Parallel recipe eligibility for canonical Scremas.
//!
//! This classifies lowering strategies, not Screma representation variants.
//! Eligibility is expressed directly in terms of the canonical pre/operators/post
//! form; graph-local cloning and storage checks remain in each recipe analyzer.

use crate::egir::soac::screma;
use crate::egir::types::WynSoacPhase;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Strategy {
    Map,
    Reduce,
    Scan,
    Serial,
}

pub(super) fn classify<P: WynSoacPhase>(op: &screma::Op<P>) -> Strategy {
    let map_ready = op.is_map() && op.form.post.is_identity();
    let reduction_results = op.form.reduction_result_count();
    let reduce_ready = op.is_reduce()
        && op.form.post.is_identity()
        && !op.inputs.is_empty()
        && op
            .form
            .reductions
            .iter()
            .all(|reduction| !reduction.neutral.is_empty() && reduction.operator.seg_body().is_some())
        && (0..reduction_results)
            .all(|field| op.destination(field).is_some_and(|destination| destination.is_unplaced_fresh()))
        && (reduction_results..op.result_count())
            .all(|field| op.destination(field).is_some_and(|destination| destination.is_output_view()));
    let scan_ready = op.is_scan_only()
        && !op.inputs.is_empty()
        && !op.form.scans.is_empty()
        && op.form.scans.iter().all(|scan| !scan.neutral.is_empty() && scan.operator.seg_body().is_some())
        && (0..op.result_count())
            .all(|field| op.destination(field).is_some_and(|destination| destination.is_output_view()));

    if map_ready {
        Strategy::Map
    } else if reduce_ready {
        Strategy::Reduce
    } else if scan_ready {
        Strategy::Scan
    } else {
        Strategy::Serial
    }
}
