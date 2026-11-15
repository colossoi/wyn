//! Compiler mid-end using [egglog](https://github.com/egraphs-good/egglog).
//! Pass order is enforced by typestate:
//! [`from_tlc`] → [`fuse`] → [`place`] → [`schedule`] → [`prepare_interfaces`]
//! → [`optimize`] → [`prepare_host`] → [`to_ssa`].
use self::dataflow::edit;
use crate::ssa::stage::Elaborated;
use crate::tlc::stage::OwnershipApplied;
use crate::{CodegenTarget, PipelineTopologyPolicy, SymbolId};
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
mod payload;
mod planning;
mod query;
mod scalar;
#[cfg(test)]
mod test_graph;
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

/// Internalize typed TLC, preserving logical products and array nesting.
pub fn from_tlc(source: &OwnershipApplied) -> Result<Program<'_, Imported>, OptimizeError> {
    let _timing = timing::span("egglog import");
    let (graph, symbols) = from_tlc::import(source)?;
    timing::tables(&graph, "after-import");
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
    // Fuse adjacent producers before hoisting invariant work between them.
    loop {
        let revision = edit::revision(&program.graph)?;
        fusion::run(&mut program.graph)?;
        memory::simplify(&mut program.graph, true)?;
        if edit::revision(&program.graph)? == revision {
            break;
        }
    }
    memory::validate(&mut program.graph)?;
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

/// Select logical execution strategies. SSA emission completes physical scheduling.
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
    timing::evaluate(
        &mut program.graph,
        None,
        &format!("(set (SubgroupsEnabled) {enabled})"),
    )?;
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
    analysis::run(&mut program.graph)?;
    Ok(program.advance(InterfacesPrepared))
}

/// Simplify the authoritative statement bodies.
pub fn optimize(program: Program<'_, InterfacesPrepared>) -> Result<Program<'_, Optimized>, OptimizeError> {
    optimize_with_policy(program, ScalarOptimization::Full)
}

/// Basic retains reducing rewrites and safety analysis; Full additionally
/// inlines nonrecursive helpers with fresh graph identities.
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
    analysis::run(&mut program.graph)?;
    timing::tables(&program.graph, "after-optimization");
    Ok(program.advance(Optimized))
}

/// Validate the program before host and shader emission.
pub fn prepare_host(mut program: Program<'_, Optimized>) -> Result<HostPrepared<'_>, OptimizeError> {
    let _timing = timing::span("egglog host preparation");
    analysis::run(&mut program.graph)?;
    planning::epilogues(&mut program.graph)?;
    Ok(HostPrepared { program })
}

/// Plan evaluation and lower shader SSA together with its host interfaces.
pub fn to_ssa(program: HostPrepared<'_>, target: CodegenTarget) -> Result<Elaborated, OptimizeError> {
    let _timing = timing::span("SSA emission");
    to_ssa::lower(&program.program, target)
}

/// TLC retains source and public-interface metadata. The graph owns the logical
/// program and execution choices until direct SSA emission.
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

/// Logical execution strategies are selected; storage is determined during SSA emission.
pub struct Scheduled;

pub struct InterfacesPrepared;

pub struct Optimized;

pub struct HostPrepared<'source> {
    program: Program<'source, Optimized>,
}

#[cfg(test)]
mod pipeline_tests;

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

pub(super) fn output_error(message: impl Into<String>) -> OptimizeError {
    OptimizeError::Output(message.into())
}
