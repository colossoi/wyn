//! Shared pass loop over complete relational graph snapshots.

use super::{data::AssociatedData, emit, extract, from_tlc::Converted, snapshot, timing};
use egglog_engine::{ast::Parser, EGraph};

const FUSION: &str = include_str!("fusion.egg");
const DEPENDENCIES: &str = include_str!("dependencies.egg");

#[derive(Debug, thiserror::Error)]
pub enum OptimizeError {
    #[error("egglog optimization: {0}")]
    Engine(#[from] egglog_engine::Error),
    #[error("egglog extraction: {0}")]
    Extraction(String),
    #[error("egglog output: {0}")]
    Output(String),
}

pub(super) fn analyze(data: &AssociatedData) -> Result<EGraph, OptimizeError> {
    let mut graph = EGraph::default();
    let source = emit::program(data);
    timing::time("load fusion graph", || {
        graph.parse_and_run_program(Some("wyn-selected.egg".into()), &source)?;
        graph.parse_and_run_program(Some("dependencies.egg".into()), DEPENDENCIES)
    })?;
    timing::time("dependency rules", || {
        graph.parse_and_run_program(None, "(run-schedule (saturate (run dependencies)))")
    })?;
    Ok(graph)
}

/// Derive legal candidates in egglog, select one whole candidate, and repeat.
/// Selection is deterministic and greedy, not a global minimum-cost search.
/// Rebuilding the snapshot prevents monotone facts about an old action from
/// leaking into its replacement. Each current fusion removes one live execution;
/// dead records remain in the sidecar until readout selects reachable work.
/// The returned egglog program summarizes the last selected graph.
pub fn optimize(mut converted: Converted) -> Result<Converted, OptimizeError> {
    let _timing = timing::span("fusion");
    if converted.expression_program.is_some() || !converted.data.blocks.is_empty() {
        return Err(OptimizeError::Output(
            "fusion must precede expression insertion and scheduling".into(),
        ));
    }
    loop {
        let mut graph = analyze(&converted.data)?;
        timing::time("validate dependency order", || {
            snapshot::analyze(&converted.data).schedules(&converted.data)
        })?;
        timing::time("load fusion rules", || {
            graph.parse_and_run_program(Some("fusion.egg".into()), FUSION)
        })?;
        // Candidates are terminal facts: no fusion rule consumes another
        // candidate. One match round suffices; only dependency closure saturates.
        timing::time("fusion rules", || {
            graph.parse_and_run_program(None, "(run-schedule (run fusion))")
        })?;
        if timing::time("select and compose", || {
            extract::fuse_one(&graph, &mut converted.data)
        })? {
            continue;
        }
        let output = emit::program(&converted.data);
        let _parse = timing::span("parse final fusion facts");
        converted.program = Parser::default()
            .get_program_from_string(Some("wyn-optimized.egg".into()), &output)
            .map_err(|error| OptimizeError::Output(error.to_string()))?;
        return Ok(converted);
    }
}
