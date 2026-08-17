//! Output-route invariant check.
//!
//! Every host-visible output has an explicit source and at least one producer
//! recorded before scheduling. Aggregate values may remain logical here;
//! physicalization binds those routes to views and places.

use super::super::allocation::{entries_with_endpoints, CompilerFlowEndpoint, ResourcesAllocated};
use super::super::from_tlc::ConvertError;
use super::super::program::SemanticEntry;

/// Verify the post-realization invariant for every entry. Returns
/// `ConvertError::Internal` on the first violation, naming the entry
/// and offending ValueId.
pub fn check(inner: &ResourcesAllocated) -> Result<(), ConvertError> {
    for (endpoint, entry) in entries_with_endpoints(inner) {
        if matches!(endpoint, CompilerFlowEndpoint::Entry(_)) {
            check_routes(entry)?;
        }
    }
    Ok(())
}

fn check_routes(entry: &SemanticEntry) -> Result<(), ConvertError> {
    for (slot, output) in entry.outputs.iter().enumerate() {
        if output.routes.is_empty() {
            return Err(ConvertError::Internal(format!(
                "entry `{}` output slot {} has no explicit route",
                entry.name, slot
            )));
        }
        for route in &output.routes {
            if route.writers.is_empty() {
                return Err(ConvertError::Internal(format!(
                    "entry `{}` output slot {} has a source value but no producer",
                    entry.name, slot
                )));
            }
        }
    }
    Ok(())
}
