//! The semantic-to-scheduled graph boundary.
//!
//! Kernel construction works with semantic graphs. Closing a kernel recipe
//! consumes that graph and records the execution decisions made by the
//! scheduler in phase-specific SOAC states.

use crate::egir;
use crate::flow::BlockId;
use crate::LookupMap;

use super::super::program::{PlannedEntry, SemanticResourceRef};
use super::super::soac::{filter, hist, screma};
use super::super::types::{EGraph, Scheduled, Semantic, SideEffectKind, Soac, SoacEffect};

type AllocatedSemantic = Semantic<SemanticResourceRef>;

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
    Atomic(Vec<egir::soac::hist::AtomicUpdate>),
    Bucket {
        stage: egir::soac::hist::ParallelStage,
        topology: Option<egir::soac::hist::DispatchTopology>,
        storage: egir::soac::hist::BucketStorage<SemanticResourceRef>,
    },
}

impl ParallelHistPlan {
    pub(super) fn new(
        owner: super::super::program::SemanticOpId,
        operations: Vec<egir::soac::hist::AtomicUpdate>,
    ) -> Self {
        Self {
            owner,
            kind: ParallelHistKind::Atomic(operations),
        }
    }

    pub(super) fn bucket(
        owner: super::super::program::SemanticOpId,
        stage: egir::soac::hist::ParallelStage,
        topology: Option<egir::soac::hist::DispatchTopology>,
        storage: egir::soac::hist::BucketStorage<SemanticResourceRef>,
    ) -> Self {
        Self {
            owner,
            kind: ParallelHistKind::Bucket {
                stage,
                topology,
                storage,
            },
        }
    }
}
pub(super) fn entry(
    entry: PlannedEntry<AllocatedSemantic>,
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
    graph: EGraph<AllocatedSemantic>,
    serial: bool,
) -> Result<(EGraph<Scheduled>, LookupMap<BlockId, BlockId>), String> {
    graph.try_map_phase(|_, _, id, soac| {
        schedule_soac_with_mode(id, soac, None, None, serial).map(|soac| (id, soac))
    })
}

fn schedule_soac_with_mode(
    id: super::super::program::SemanticOpId,
    soac: Soac<AllocatedSemantic>,
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
            let filter::SemanticState { space, output, .. } = state;
            let state = match filter_plan {
                None => filter::ScheduledState::Loop {
                    space,
                    storage: output,
                },
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
                        ParallelHistKind::Bucket {
                            stage,
                            topology,
                            storage,
                        } => hist::ScheduledState::Bucket {
                            space,
                            stage,
                            topology,
                            storage,
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
