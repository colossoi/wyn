//! The semantic-to-scheduled graph boundary.
//!
//! Kernel construction works with semantic graphs. Closing a kernel recipe
//! consumes that graph and records the execution decisions made by the
//! scheduler in phase-specific SOAC states.

use crate::ssa::types::BlockId;

use super::super::program::{remap_control_headers, PlannedEntry, SemanticResourceRef};
use super::super::soac::{filter, hist, screma};
use super::super::types::{EGraph, Scheduled, Semantic, SideEffect, SideEffectKind, Soac};

pub(crate) fn entry(
    entry: PlannedEntry<Semantic>,
    filter_plan: Option<filter::Plan<SemanticResourceRef>>,
) -> Result<PlannedEntry<Scheduled>, String> {
    let PlannedEntry {
        name,
        span,
        execution_model,
        inputs,
        outputs,
        resource_declarations,
        params,
        return_ty,
        graph,
        control_headers,
        aliases,
        mut output_routes,
    } = entry;
    let (graph, block_map) = graph.try_map_phase(|_, _, soac| schedule_soac(soac, filter_plan))?;
    for route in &mut output_routes {
        route.source.block = block_map[&route.source.block];
    }
    Ok(PlannedEntry {
        name,
        span,
        execution_model,
        inputs,
        outputs,
        resource_declarations,
        params,
        return_ty,
        graph,
        control_headers: remap_control_headers(&control_headers, |block| block_map[&block]),
        aliases,
        output_routes,
    })
}

pub(crate) fn graph(
    graph: EGraph<Semantic>,
    serial: bool,
) -> Result<(EGraph<Scheduled>, crate::LookupMap<BlockId, BlockId>), String> {
    graph.try_map_phase(|_, _, soac| schedule_soac_with_mode(soac, None, serial))
}

fn schedule_soac(
    soac: Soac<Semantic>,
    filter_plan: Option<filter::Plan<SemanticResourceRef>>,
) -> Result<Soac<Scheduled>, String> {
    schedule_soac_with_mode(soac, filter_plan, false)
}

fn schedule_soac_with_mode(
    soac: Soac<Semantic>,
    filter_plan: Option<filter::Plan<SemanticResourceRef>>,
    serial: bool,
) -> Result<Soac<Scheduled>, String> {
    Ok(match soac {
        Soac::Screma(screma::Op { body, state }) => {
            let state = match state {
                screma::SemanticState::Serial => screma::ScheduledState::Serial,
                screma::SemanticState::Segmented {
                    space,
                    placement,
                    output_slots,
                    resources,
                } if !serial && placement == screma::Placement::Kernel => match body.kind {
                    screma::Kind::Map => screma::ScheduledState::SegMap {
                        space,
                        output_slots,
                        resources,
                    },
                    screma::Kind::Reduce(_) => screma::ScheduledState::SegRed {
                        space,
                        output_slots,
                        resources,
                    },
                    screma::Kind::Scan(_) => screma::ScheduledState::SegScan {
                        space,
                        output_slots,
                        resources,
                    },
                    screma::Kind::Composite(_) => screma::ScheduledState::SegComposite {
                        space,
                        output_slots,
                        resources,
                    },
                },
                screma::SemanticState::Segmented { .. } => screma::ScheduledState::Serial,
            };
            Soac::Screma(screma::Op { body, state })
        }
        Soac::Filter(filter::Op { body, state }) => {
            let filter::SemanticState { space, storage } = state;
            let state = match filter_plan.unwrap_or(filter::Plan::Serial) {
                filter::Plan::Serial => filter::ScheduledState::Serial { space, storage },
                plan => {
                    let filter::Output::Runtime { scratch, length } = storage else {
                        return Err("parallel filter plan requires runtime output storage".into());
                    };
                    let (stage, buffers) = match plan {
                        filter::Plan::Flags(buffers) => (filter::ParallelStage::Flags, buffers),
                        filter::Plan::Scan(buffers) => (filter::ParallelStage::Scan, buffers),
                        filter::Plan::Scatter(buffers) => (filter::ParallelStage::Scatter, buffers),
                        filter::Plan::Serial => unreachable!(),
                    };
                    filter::ScheduledState::Parallel {
                        space,
                        storage: filter::RuntimeStorage { scratch, length },
                        plan: filter::ParallelPlan { stage, buffers },
                    }
                }
            };
            Soac::Filter(filter::Op { body, state })
        }
        Soac::Hist(hist::Op { body, state }) => {
            let state = match state {
                hist::SemanticState::Serial => hist::ScheduledState::Serial,
                hist::SemanticState::Segmented(space) if !serial => hist::ScheduledState::Segmented(space),
                hist::SemanticState::Segmented(_) => hist::ScheduledState::Serial,
            };
            Soac::Hist(hist::Op { body, state })
        }
    })
}

pub(crate) fn force_serial(graph: &mut EGraph<Scheduled>) {
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            let SideEffectKind::Soac(Soac::Screma(op)) = &mut effect.kind else {
                continue;
            };
            if matches!(
                op.state,
                screma::ScheduledState::SegMap { .. }
                    | screma::ScheduledState::SegRed { .. }
                    | screma::ScheduledState::SegScan { .. }
                    | screma::ScheduledState::SegComposite { .. }
            ) {
                op.state = screma::ScheduledState::Serial;
            }
        }
    }
}

pub(crate) fn kernel_effect(graph: &EGraph<Scheduled>) -> Option<(BlockId, usize, &SideEffect<Scheduled>)> {
    graph.skeleton.blocks.iter().find_map(|(block, contents)| {
        contents.side_effects.iter().enumerate().find_map(|(index, effect)| {
            matches!(
                &effect.kind,
                SideEffectKind::Soac(Soac::Screma(screma::Op {
                    state: screma::ScheduledState::SegMap { .. }
                        | screma::ScheduledState::SegRed { .. }
                        | screma::ScheduledState::SegScan { .. }
                        | screma::ScheduledState::SegComposite { .. },
                    ..
                })) | SideEffectKind::Soac(Soac::Filter(filter::Op {
                    state: filter::ScheduledState::Parallel { .. },
                    ..
                }))
            )
            .then_some((block, index, effect))
        })
    })
}
