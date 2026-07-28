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
        let map_ready = shape == ScremaShape::Map
            && op.form.post.is_identity()
            && op.result_count() != 0
            && (0..op.result_count()).all(|field| {
                op.destination(field).is_some_and(|destination| {
                    destination.is_output_view() || destination.is_input_buffer()
                })
            });
        Self {
            shape,
            recipe: if map_ready { ScremaRecipeClass::Map } else { ScremaRecipeClass::Serial },
        }
    }

    pub(super) fn shape(self) -> ScremaShape {
        self.shape
    }

    pub(super) fn recipe_class(self) -> ScremaRecipeClass {
        self.recipe
    }
}
