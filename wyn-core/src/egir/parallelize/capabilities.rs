//! Structural classification for canonical Scremas.
//!
//! Recipe eligibility is deliberately conservative while the parallel
//! algorithms are migrated to consume whole pre/operator/post lambdas.  The
//! shape remains available to scheduling policy, but no legacy mini-map
//! recipe is selected.

use crate::egir::soac::screma;
use crate::egir::types::WynSoacPhase;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ScremaShape {
    Map,
    Reduce,
    Scan,
    Mixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ScremaRecipeClass {
    Map,
    Reduce,
    Scan,
    Serial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ScremaRecipeCapabilities {
    shape: ScremaShape,
    recipe: ScremaRecipeClass,
}

impl ScremaRecipeCapabilities {
    pub(super) fn analyze<P: WynSoacPhase>(op: &screma::Op<P>) -> Self {
        let shape = if op.is_map() {
            ScremaShape::Map
        } else if op.is_reduce() {
            ScremaShape::Reduce
        } else if op.is_scan_only() {
            ScremaShape::Scan
        } else {
            ScremaShape::Mixed
        };
        let map_ready = shape == ScremaShape::Map && op.form.post.is_identity();
        let reduction_results = op.form.reduction_result_count();
        let reduce_ready = shape == ScremaShape::Reduce
            && op.form.post.is_identity()
            && !op.inputs.is_empty()
            && op.form.reductions.iter().all(|reduction| {
                reduction.neutral.len() == 1
                    && reduction.operator.seg_body().is_some_and(|body| body.captures.is_empty())
            })
            && (0..reduction_results).all(|field| {
                op.destination(field).is_some_and(|destination| destination.is_unplaced_fresh())
            })
            && (reduction_results..op.result_count())
                .all(|field| op.destination(field).is_some_and(|destination| destination.is_output_view()));
        Self {
            shape,
            recipe: if map_ready {
                ScremaRecipeClass::Map
            } else if reduce_ready {
                ScremaRecipeClass::Reduce
            } else {
                ScremaRecipeClass::Serial
            },
        }
    }

    pub(super) fn shape(self) -> ScremaShape {
        self.shape
    }

    pub(super) fn recipe_class(self) -> ScremaRecipeClass {
        self.recipe
    }
}
