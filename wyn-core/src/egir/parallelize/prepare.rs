//! The semantic-to-scheduled graph boundary.
//!
//! Kernel construction works with semantic graphs. Closing a kernel recipe
//! consumes that graph and records the execution decisions made by the
//! scheduler in phase-specific SOAC states.

use crate::flow::BlockId;

use super::super::program::{PlannedEntry, SemanticResourceRef};
use super::super::soac::{filter, hist, screma};
use super::super::types::{EGraph, Scheduled, Semantic, SideEffectKind, Soac, SoacEffect};

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

#[derive(Clone)]
pub(super) struct ParallelHistPlan {
    owner: super::super::program::SemanticOpId,
    kind: ParallelHistKind,
}

#[derive(Clone)]
enum ParallelHistKind {
    Atomic(Vec<crate::egir::soac::hist::AtomicUpdate>),
    Bucket {
        stage: crate::egir::soac::hist::ParallelStage,
        topology: Option<crate::egir::soac::hist::DispatchTopology>,
    },
}

impl ParallelHistPlan {
    pub(super) fn new(
        owner: super::super::program::SemanticOpId,
        operations: Vec<crate::egir::soac::hist::AtomicUpdate>,
    ) -> Self {
        Self {
            owner,
            kind: ParallelHistKind::Atomic(operations),
        }
    }

    pub(super) fn bucket(
        owner: super::super::program::SemanticOpId,
        stage: crate::egir::soac::hist::ParallelStage,
        topology: Option<crate::egir::soac::hist::DispatchTopology>,
    ) -> Self {
        Self {
            owner,
            kind: ParallelHistKind::Bucket { stage, topology },
        }
    }
}
pub(super) fn entry(
    entry: PlannedEntry<Semantic>,
    filter_plan: Option<ParallelFilterPlan>,
    hist_plan: Option<ParallelHistPlan>,
) -> Result<PlannedEntry<Scheduled>, String> {
    entry
        .into_inner()
        .try_map_phase(|_, _, id, soac| {
            schedule_soac_with_mode(id, soac, filter_plan, hist_plan.clone(), false).map(|soac| (id, soac))
        })
        .map(PlannedEntry::new)
}

pub(in crate::egir) fn graph(
    graph: EGraph<Semantic>,
    serial: bool,
) -> Result<(EGraph<Scheduled>, crate::LookupMap<BlockId, BlockId>), String> {
    graph.try_map_phase(|_, _, id, soac| {
        schedule_soac_with_mode(id, soac, None, None, serial).map(|soac| (id, soac))
    })
}

fn schedule_soac_with_mode(
    id: super::super::program::SemanticOpId,
    soac: Soac<Semantic>,
    filter_plan: Option<ParallelFilterPlan>,
    hist_plan: Option<ParallelHistPlan>,
    serial: bool,
) -> Result<Soac<Scheduled>, String> {
    Ok(match soac {
        Soac::Screma(screma::Op {
            inputs,
            form,
            result_state,
            state,
        }) => Soac::Screma(screma::Op {
            inputs,
            form,
            result_state,
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
        Soac::Hist(hist::Op { inputs, form, state }) => {
            let state = match (state, hist_plan) {
                (hist::SemanticState::Segmented(space), Some(plan)) if !serial && plan.owner == id => {
                    match plan.kind {
                        ParallelHistKind::Atomic(operations) => {
                            hist::ScheduledState::Atomic { space, operations }
                        }
                        ParallelHistKind::Bucket { stage, topology } => hist::ScheduledState::Bucket {
                            space,
                            stage,
                            topology,
                        },
                    }
                }
                _ => hist::ScheduledState::Serial,
            };
            Soac::Hist(hist::Op { inputs, form, state })
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
            match &mut effect.kind {
                SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op)))
                    if matches!(op.state, screma::ScheduledState::Segmented(_)) =>
                {
                    op.state = screma::ScheduledState::Serial;
                }
                SideEffectKind::Soac(SoacEffect(_, Soac::Hist(op)))
                    if !matches!(op.state, hist::ScheduledState::Serial) =>
                {
                    op.state = hist::ScheduledState::Serial;
                }
                _ => {}
            }
        }
    }
}
