//! Compiler mid-end using [egglog](https://github.com/egraphs-good/egglog).
//! Pass order is enforced by typestate:
//! [`from_tlc`] → [`fuse`] → [`place`] → [`schedule`] → [`prepare_interfaces`]
//! → [`optimize`] → [`prepare_host`] → [`to_ssa`].
use crate::ssa::stage::Elaborated;
use crate::tlc::stage::OwnershipApplied;
use crate::{CodegenTarget, PipelineTopologyPolicy, SymbolId};
use egglog_engine::ast::{Command, Parser};
use egglog_engine::EGraph;
use egglog_engine::Error;
use wyn_base::Interner;

mod analysis;
mod bindings;
pub mod cost;
pub(crate) mod dataflow;
mod execution;
mod from_tlc;
mod fusion;
mod memory;
mod planning;
mod query;
mod scalar;
mod timing;
mod to_ssa;

pub use timing::with_timings;

#[derive(Debug, thiserror::Error)]
pub enum OptimizeError {
    #[error("egglog optimization: {0}")]
    Engine(#[from] Error),
    #[error("egglog extraction: {0}")]
    Extraction(String),
    #[error("egglog output: {0}")]
    Output(String),
}

/// Internalize typed TLC into normalized components and explicit lexical bodies.
pub fn from_tlc(source: &OwnershipApplied) -> Result<Program<'_, Imported>, OptimizeError> {
    let _timing = timing::span("egglog import");
    let (graph, symbols) = from_tlc::import(source)?;
    Ok(Program {
        source,
        graph,
        symbols,
        _stage: Imported,
    })
}

/// Simplify regions and fuse parallel statements in the common IR.
pub fn fuse(mut program: Program<'_, Imported>) -> Result<Program<'_, Fused>, OptimizeError> {
    timing::time("egglog structural analysis", || analysis::run(&mut program.graph))?;
    memory::inline(&mut program.graph)?;
    memory::simplify(&mut program.graph, false)?;
    fusion::run(&mut program.graph)?;
    memory::simplify(&mut program.graph, true)?;
    Ok(program.advance(Fused))
}

/// Choose execution domains and rematerialization. This is execution placement;
/// scalar loop-invariant and common-branch hoisting belongs to the local optimizer.
pub fn place(
    program: Program<'_, Fused>,
    topology: PipelineTopologyPolicy,
) -> Result<Program<'_, Placed>, OptimizeError> {
    place_with_cost_model(program, topology, cost::CostModel::default())
}

pub fn place_with_cost_model(
    mut program: Program<'_, Fused>,
    topology: PipelineTopologyPolicy,
    costs: cost::CostModel,
) -> Result<Program<'_, Placed>, OptimizeError> {
    let costs = costs.validate().map_err(output_error)?;
    let _timing = timing::span("egglog placement");
    planning::place(&mut program.graph, topology, costs)?;
    Ok(program.advance(Placed))
}

/// Select dispatch recipes, resource requirements, and ordering constraints.
pub fn schedule(program: Program<'_, Placed>) -> Result<Program<'_, Scheduled>, OptimizeError> {
    schedule_with_subgroups(program, false)
}

/// Enable native subgroup arithmetic on devices supporting that shader feature.
/// Without it, scheduling retains portable workgroup collectives.
pub fn schedule_with_subgroups(
    program: Program<'_, Placed>,
    enabled: bool,
) -> Result<Program<'_, Scheduled>, OptimizeError> {
    schedule_with_cost_model(program, enabled, cost::CostModel::default())
}

pub fn schedule_with_cost_model(
    mut program: Program<'_, Placed>,
    enabled: bool,
    costs: cost::CostModel,
) -> Result<Program<'_, Scheduled>, OptimizeError> {
    let costs = costs.validate().map_err(output_error)?;
    program.graph.parse_and_run_program(None, &format!("(set (SubgroupsEnabled) {enabled})"))?;
    let _timing = timing::span("egglog scheduling");
    planning::schedule(&mut program.graph, costs)?;
    Ok(program.advance(Scheduled))
}

/// Validate memory bindings before final scalar simplification.
pub fn prepare_interfaces(
    mut program: Program<'_, Scheduled>,
) -> Result<Program<'_, InterfacesPrepared>, OptimizeError> {
    let _timing = timing::span("egglog interface preparation");
    memory::validate(&mut program.graph)?;
    program.graph.parse_and_run_program(None, "(run-schedule (saturate simplify-analysis))")?;
    Ok(program.advance(InterfacesPrepared))
}

/// Simplify the authoritative statement bodies.
pub fn optimize(program: Program<'_, InterfacesPrepared>) -> Result<Program<'_, Optimized>, OptimizeError> {
    optimize_with_policy(program, ScalarOptimization::Full)
}

/// Basic retains reducing rewrites and safety analysis; Full additionally
/// inlines nonrecursive helpers through the common region renamer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarOptimization {
    Basic,
    Full,
}

pub fn optimize_with_policy(
    mut program: Program<'_, InterfacesPrepared>,
    policy: ScalarOptimization,
) -> Result<Program<'_, Optimized>, OptimizeError> {
    scalar::run(&mut program.graph, policy)?;
    program.graph.parse_and_run_program(None, "(run-schedule (saturate simplify-analysis))")?;
    Ok(program.advance(Optimized))
}

/// Validate the program before host and shader emission.
pub fn prepare_host(mut program: Program<'_, Optimized>) -> Result<HostPrepared<'_>, OptimizeError> {
    let _timing = timing::span("egglog host preparation");
    program.graph.parse_and_run_program(
        None,
        "(run-schedule (saturate (seq
        (saturate fusion-index memory-names) (saturate memory-scopes)
        (saturate schedule-cost memory-origin) (saturate memory-effects simplify-analysis)
        (saturate placement-context) (saturate placement-scopes)
        (saturate kernel-prelude-analysis memory-effects memory-rename) (saturate kernel-prelude-select)
        (saturate memory-rewrite memory-rename))))",
    )?;
    memory::validate(&mut program.graph)?;
    program.graph.parse_and_run_program(None, "(run-schedule (saturate memory-origin schedule-cost) (saturate memory-allocation) (saturate memory-publication))")?;
    Ok(HostPrepared { program })
}

/// Plan evaluation and lower shader SSA together with its host interfaces.
pub fn to_ssa(program: HostPrepared<'_>, target: CodegenTarget) -> Result<Elaborated, OptimizeError> {
    to_ssa::lower(&program.program, target)
}

/// TLC retains source and public-interface metadata. The graph owns all
/// executable bodies, transformations, and allocation decisions.
pub struct Program<'source, Stage> {
    source: &'source OwnershipApplied,
    graph: EGraph,
    symbols: Interner<i64, SymbolId>,
    _stage: Stage,
}

impl<'source, Stage> Program<'source, Stage> {
    fn advance<Next>(self, stage: Next) -> Program<'source, Next> {
        Program {
            source: self.source,
            graph: self.graph,
            symbols: self.symbols,
            _stage: stage,
        }
    }
}

pub struct Imported;
pub struct Fused;
pub struct Placed;

/// Dispatch recipes, execution domains, storage, and effect order are fixed.
pub struct Scheduled;

pub struct InterfacesPrepared;

pub struct Optimized;

pub struct HostPrepared<'source> {
    program: Program<'source, Optimized>,
}

#[cfg(test)]
mod pipeline_tests;

#[cfg(test)]
mod descriptor_tests;

#[cfg(test)]
fn optimize_for_test(program: Program<'_, Scheduled>) -> Result<Program<'_, Optimized>, OptimizeError> {
    optimize(prepare_interfaces(program)?)
}

#[cfg(test)]
fn finish_for_test(
    program: Program<'_, Optimized>,
    target: CodegenTarget,
) -> Result<Elaborated, OptimizeError> {
    let program = prepare_host(program)?;
    to_ssa(program, target)
}

fn parse_program(filename: &str, source: &str) -> Result<Vec<Command>, OptimizeError> {
    Parser::default()
        .get_program_from_string(Some(filename.into()), source)
        .map_err(|error| OptimizeError::Output(error.to_string()))
}

pub(super) fn output_error(message: impl Into<String>) -> OptimizeError {
    OptimizeError::Output(message.into())
}
