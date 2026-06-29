//! EGIR-side SOAC parallelization. From each entry's recognition, tags the tail
//! SOAC as a reified `Seg` (or parallel `Scatter`) so `soac_expand` emits the
//! lane-indexed kernel.
use crate::LookupMap;

use super::program::{EgirEntry, EgirFunc, EgirInner};
use super::types::{PendingSoac, SegLevel, SegSpace, SideEffectKind, SoacDestination};
use crate::tlc::parallelize::EntryRecognition;

/// Walk every entry; for each entry that has a plan, dispatch to
/// `parallelize_entry`. It internally dispatches on the Screma's
/// accumulator-kind shape:
///
/// - **0 accumulators** (pointwise Screma): wrap the SOAC in
///   `PendingSoac::Parallel` so `soac_expand` emits the lane-indexed
///   kernel via `build_parallel_maps`.
/// - **N Reduce accumulators**: phase 1 chunked rewrite (chunked input +
///   chunked map outputs + store-to-`partials_i[tid]`) plus one
///   synthesized phase-2 tree-reduce per accumulator.
/// - **1 Scan accumulator**: phase 1 chunked rewrite (chunked input +
///   chunked output view + appended chunked reduce → `block_sums[tid]`),
///   plus synthesized phase 2 (sequential scan of `block_sums`) and
///   phase 3 (apply offsets via a swap-args wrapper).
pub fn run(inner: &mut EgirInner, recognitions: &LookupMap<String, EntryRecognition>) {
    use crate::tlc::parallelize::ParallelStrategy;
    let new_entries: Vec<EgirEntry> = Vec::new();
    let new_functions: Vec<EgirFunc> = Vec::new();
    for entry in inner.entry_points.iter_mut() {
        let Some(rec) = recognitions.get(&entry.name) else {
            continue;
        };
        match rec.strategy {
            // Pointwise map / Screma: tag the tail SOAC as a `Seg` so
            // `soac_expand` emits the lane-indexed kernel. `rewrite_tail_map`
            // only converts the no-accumulator case, so a mixed `Screma` here
            // is left for the reduce/scan path below.
            ParallelStrategy::Map | ParallelStrategy::Screma => {
                rewrite_tail_map(entry);
            }
            // Reduce/scan two-/three-phase lowering allocates scratch bindings
            // and synthesizes phase entries; wired in a follow-up increment.
            ParallelStrategy::Reduce | ParallelStrategy::Scan => {}
        }
    }
    inner.entry_points.extend(new_entries);
    inner.functions.extend(new_functions);
}

fn rewrite_tail_map(entry: &mut super::program::EgirEntry) {
    // Walk every block and locate the (unique) tail pointwise Screma.
    for (_, block) in entry.graph.skeleton.blocks.iter_mut() {
        for se in block.side_effects.iter_mut() {
            let SideEffectKind::Pending(ref pending) = se.kind else {
                continue;
            };
            match pending {
                PendingSoac::Screma {
                    accumulators,
                    map_destinations,
                    ..
                } if accumulators.is_empty()
                    && !map_destinations.is_empty()
                    && map_destinations.iter().all(|dest| {
                        matches!(dest, SoacDestination::OutputView | SoacDestination::InputBuffer)
                    }) =>
                {
                    let SideEffectKind::Pending(PendingSoac::Screma {
                        map_funcs,
                        input_array_types,
                        input_elem_types,
                        map_output_elem_types,
                        map_input_indices,
                        map_capture_counts,
                        map_destinations,
                        acc_destinations,
                        ..
                    }) = se.kind.clone()
                    else {
                        unreachable!()
                    };
                    let result_types = map_output_elem_types.clone();
                    se.kind = SideEffectKind::Pending(PendingSoac::Seg {
                        space: SegSpace {
                            level: SegLevel::Thread,
                            len: None,
                        },
                        map_funcs,
                        accumulators: vec![],
                        input_array_types,
                        input_elem_types,
                        map_output_elem_types,
                        map_input_indices,
                        map_capture_counts,
                        map_destinations,
                        acc_destinations,
                        result_types,
                    });
                    return;
                }
                PendingSoac::Scatter { space: None, .. } => {
                    if let SideEffectKind::Pending(PendingSoac::Scatter { space, .. }) = &mut se.kind {
                        *space = Some(SegSpace {
                            level: SegLevel::Thread,
                            len: None,
                        });
                    }
                    return;
                }
                _ => continue,
            }
        }
    }
}
