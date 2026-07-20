//! The semantic-to-scheduled graph boundary.
//!
//! Kernel construction works with semantic graphs. Closing a kernel recipe
//! consumes that graph and records the execution decisions made by the
//! scheduler in phase-specific SOAC states.

use crate::flow::BlockId;

use super::super::program::{PlannedEntry, SemanticResourceRef};
use super::super::soac::{filter, hist, screma};
use super::super::types::{EGraph, Scheduled, Semantic, SideEffect, SideEffectKind, Soac, SoacEffect};

#[derive(Clone, Copy)]
pub(super) struct ParallelFilterPlan {
    pub(super) stage: filter::ParallelStage,
    config: filter::ParallelConfig<SemanticResourceRef>,
    storage: filter::RuntimeStorage<SemanticResourceRef>,
}

impl ParallelFilterPlan {
    pub(super) fn new(
        stage: filter::ParallelStage,
        config: filter::ParallelConfig<SemanticResourceRef>,
        storage: filter::RuntimeStorage<SemanticResourceRef>,
    ) -> Self {
        Self {
            stage,
            config,
            storage,
        }
    }
}

pub(super) fn entry(
    entry: PlannedEntry<Semantic>,
    filter_plan: Option<ParallelFilterPlan>,
) -> Result<PlannedEntry<Scheduled>, String> {
    entry.try_map_phase(|_, _, id, soac| {
        schedule_soac_with_mode(soac, filter_plan, false).map(|soac| (id, soac))
    })
}

pub(in crate::egir) fn graph(
    graph: EGraph<Semantic>,
    serial: bool,
) -> Result<(EGraph<Scheduled>, crate::LookupMap<BlockId, BlockId>), String> {
    graph.try_map_phase(|_, _, id, soac| schedule_soac_with_mode(soac, None, serial).map(|soac| (id, soac)))
}

fn schedule_soac_with_mode(
    soac: Soac<Semantic>,
    filter_plan: Option<ParallelFilterPlan>,
    serial: bool,
) -> Result<Soac<Scheduled>, String> {
    Ok(match soac {
        Soac::Screma(screma::Op::Map { lanes, state }) => Soac::Screma(screma::Op::Map {
            lanes,
            state: schedule_screma_state(state, serial),
        }),
        Soac::Screma(screma::Op::Reduce {
            lanes,
            operators,
            state,
        }) => Soac::Screma(screma::Op::Reduce {
            lanes,
            operators,
            state: schedule_screma_state(state, serial),
        }),
        Soac::Screma(screma::Op::Scan {
            lanes,
            operators,
            state,
        }) => Soac::Screma(screma::Op::Scan {
            lanes,
            operators,
            state: schedule_screma_state(state, serial),
        }),
        Soac::Screma(screma::Op::Composite {
            lanes,
            operators,
            state,
        }) => Soac::Screma(screma::Op::Composite {
            lanes,
            operators,
            state: schedule_screma_state(state, serial),
        }),
        Soac::Filter(filter::Op { body, state }) => {
            let filter::SemanticState { space, storage } = state;
            let state = match filter_plan {
                None => filter::ScheduledState::Loop { space, storage },
                Some(ParallelFilterPlan {
                    stage,
                    config,
                    storage,
                }) => filter::ScheduledState::Pipeline {
                    space,
                    storage,
                    plan: filter::ParallelPlan {
                        stage,
                        buffers: config.buffers,
                        scan_workgroup_width: config.scan_workgroup_width,
                    },
                },
            };
            Soac::Filter(filter::Op { body, state })
        }
        Soac::Hist(hist::Op { body, state }) => {
            let state = match state {
                hist::State::Serial => hist::State::Serial,
                hist::State::Segmented(space) if !serial => hist::State::Segmented(space),
                hist::State::Segmented(_) => hist::State::Serial,
            };
            Soac::Hist(hist::Op { body, state })
        }
    })
}

fn schedule_screma_state(
    state: screma::SemanticState<SemanticResourceRef>,
    serial: bool,
) -> screma::ScheduledState<SemanticResourceRef> {
    match state {
        screma::SemanticState::Serial => screma::ScheduledState::Serial,
        screma::SemanticState::Segmented {
            space,
            placement,
            output_slots,
            resources,
        } if !serial && placement == screma::Placement::Kernel => {
            screma::ScheduledState::Segmented(screma::Segmented {
                space,
                output_slots,
                resources,
            })
        }
        screma::SemanticState::Segmented { .. } => screma::ScheduledState::Serial,
    }
}

pub(super) fn force_serial(graph: &mut EGraph<Scheduled>) {
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &mut effect.kind else {
                continue;
            };
            match op {
                screma::Op::Map { state, .. }
                | screma::Op::Reduce { state, .. }
                | screma::Op::Scan { state, .. }
                | screma::Op::Composite { state, .. }
                    if matches!(state, screma::ScheduledState::Segmented(_)) =>
                {
                    *state = screma::ScheduledState::Serial;
                }
                _ => {}
            }
        }
    }
}

pub(super) fn parallel_effect(
    graph: &EGraph<Scheduled>,
) -> Option<(BlockId, usize, &SideEffect<Scheduled>)> {
    graph.skeleton.blocks.iter().find_map(|(block, contents)| {
        contents.side_effects.iter().enumerate().find_map(|(index, effect)| {
            matches!(
                &effect.kind,
                SideEffectKind::Soac(SoacEffect(
                    _,
                    Soac::Screma(
                        screma::Op::Map {
                            state: screma::ScheduledState::Segmented(_),
                            ..
                        } | screma::Op::Reduce {
                            state: screma::ScheduledState::Segmented(_),
                            ..
                        } | screma::Op::Scan {
                            state: screma::ScheduledState::Segmented(_),
                            ..
                        } | screma::Op::Composite {
                            state: screma::ScheduledState::Segmented(_),
                            ..
                        }
                    )
                )) | SideEffectKind::Soac(SoacEffect(
                    _,
                    Soac::Filter(filter::Op {
                        state: filter::ScheduledState::Pipeline { .. },
                        ..
                    })
                ))
            )
            .then_some((block, index, effect))
        })
    })
}
