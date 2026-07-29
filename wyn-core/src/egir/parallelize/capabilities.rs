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
    let reduction_results = op.form.reduction_result_count();
    let reductions_ready = op
        .form
        .reductions
        .iter()
        .all(|reduction| !reduction.neutral.is_empty() && reduction.operator.seg_body().is_some());
    let scans_ready =
        op.form.scans.iter().all(|scan| !scan.neutral.is_empty() && scan.operator.seg_body().is_some());
    let reductions_are_fresh = (0..reduction_results)
        .all(|field| op.destination(field).is_some_and(|destination| destination.is_unplaced_fresh()));
    let post_results_are_views = (reduction_results..op.result_count())
        .all(|field| op.destination(field).is_some_and(|destination| destination.is_output_view()));

    if op.is_map() && op.form.post.is_identity() {
        Strategy::Map
    } else if op.is_reduce()
        && op.form.post.is_identity()
        && !op.inputs.is_empty()
        && reductions_ready
        && reductions_are_fresh
        && post_results_are_views
    {
        Strategy::Reduce
    } else if !op.form.scans.is_empty()
        && !op.inputs.is_empty()
        && scans_ready
        && reductions_ready
        && reductions_are_fresh
        && post_results_are_views
    {
        // Scan lowering uses the product of every scan and reduction operator.
        // The block-scan total supplies reductions while its prefixes feed post.
        Strategy::Scan
    } else {
        Strategy::Serial
    }
}
