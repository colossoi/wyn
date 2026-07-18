//! Target-independent defaults used by the portable parallel scheduler.
//!
//! A selected recipe carries every value needed by later lowering stages;
//! consumers must not re-derive these choices from module globals.

#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

/// Scheduling choices shared by candidate analysis, scratch sizing, and
/// physical kernel construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ParallelPolicy {
    pub reduce_phase1_width: u32,
    pub reduce_phase2_width: u32,
    pub filter_scan_groups: u32,
}

impl Default for ParallelPolicy {
    fn default() -> Self {
        Self {
            reduce_phase1_width: REDUCE_PHASE1_WIDTH,
            reduce_phase2_width: PHASE2_WIDTH,
            filter_scan_groups: FILTER_SCAN_GROUPS,
        }
    }
}

/// Per-workgroup width of a synthesized phase-2 tree reduce.
pub const PHASE2_WIDTH: u32 = 256;
/// Per-workgroup width used to chunk a phase-1 partial reduce or scan.
pub(crate) const REDUCE_PHASE1_WIDTH: u32 = 64;
/// Workgroup count for the runtime-filter chunk scan.
pub(crate) const FILTER_SCAN_GROUPS: u32 = 4;
