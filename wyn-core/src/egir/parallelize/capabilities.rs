//! Parallel recipe eligibility for canonical Scremas.
//!
//! This classifies lowering strategies, not Screma representation variants.
//! Eligibility is expressed directly in terms of the canonical pre/operators/post
//! form; graph-local cloning and storage checks remain in each recipe analyzer.

use crate::egir::soac::screma;
use crate::egir::types::{ResourceAccess, Semantic};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Strategy {
    Map,
    Reduce,
    Scan,
    Serial,
}

pub(super) fn classify(op: &screma::Op<Semantic>) -> Strategy {
    let reduction_results = op.form.reduction_result_count();
    let reductions_ready = op
        .form
        .reductions
        .iter()
        .all(|reduction| !reduction.neutral.is_empty() && reduction.operator.seg_body().is_some());
    let scans_ready =
        op.form.scans.iter().all(|scan| !scan.neutral.is_empty() && scan.operator.seg_body().is_some());
    let reductions_are_fresh =
        (0..reduction_results).all(|field| op.ownership(field) == Some(crate::types::SoacOwnership::Fresh));
    let routed_post_results = match op.semantic_state() {
        screma::SemanticState::Segmented {
            output_slots,
            resources,
            ..
        } => output_slots
            .len()
            .max(resources.iter().filter(|resource| resource.access != ResourceAccess::Read).count()),
        screma::SemanticState::Serial => 0,
    };
    let post_results_are_views = routed_post_results >= op.result_count() - reduction_results;

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
