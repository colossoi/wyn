//! Shared pass loop over complete relational graph snapshots.

use super::{data::AssociatedData, emit, extract, from_tlc::Converted};
use egglog_engine::{ast::Parser, EGraph};

const FUSION: &str = include_str!("fusion.egg");
const DEPENDENCIES: &str = include_str!("dependencies.egg");
const EFFECTS: &str = include_str!("effects.egg");
const REACHABILITY: &str = include_str!("reachability.egg");
// Membership must finish before safety's all-members check; safety must finish
// before required effects; liveness must finish before closed-world use checks.
const ANALYSIS: &str = "(run-schedule (seq
    (saturate (run dependencies)) (saturate (run effects))
    (saturate (run reachability)) (saturate (run uses))))";

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
    graph.parse_and_run_program(Some("wyn-selected.egg".into()), &emit::program(data))?;
    for (name, rules) in [
        ("dependencies.egg", DEPENDENCIES),
        ("effects.egg", EFFECTS),
        ("reachability.egg", REACHABILITY),
    ] {
        graph.parse_and_run_program(Some(name.into()), rules)?;
    }
    graph.parse_and_run_program(None, ANALYSIS)?;
    Ok(graph)
}

/// Derive legal candidates in egglog, select one whole candidate, and repeat.
/// Selection is deterministic and greedy, not a global minimum-cost search.
/// Rebuilding the snapshot prevents monotone facts about an old action from
/// leaking into its replacement. Each current fusion removes one live execution;
/// dead records remain in the fact base until readout selects reachable work.
/// Both the returned program and sidecar describe the last selected graph.
pub fn optimize(mut converted: Converted) -> Result<Converted, OptimizeError> {
    loop {
        let mut graph = analyze(&converted.data)?;
        extract::schedules(&graph, &converted.data)?;
        graph.parse_and_run_program(Some("fusion.egg".into()), FUSION)?;
        graph.parse_and_run_program(None, "(run-schedule (saturate (run fusion)))")?;
        if extract::fuse_one(&graph, &mut converted.data)? {
            continue;
        }
        let output = emit::program(&converted.data) + &extract::facts(&graph)?;
        converted.program = Parser::default()
            .get_program_from_string(Some("wyn-optimized.egg".into()), &output)
            .map_err(|error| OptimizeError::Output(error.to_string()))?;
        return Ok(converted);
    }
}
