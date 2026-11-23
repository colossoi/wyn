//! Compiler mid-end using [egglog](https://github.com/egraphs-good/egglog).
//! Pass order is enforced by typestate:
//! [`from_tlc`] → [`fuse`] → [`optimize`] → [`place`] → [`schedule`]
//! → [`prepare_interfaces`] → [`prepare_host`] → [`to_ssa`].
//!
//! Egglog owns logical shapes, effects, aliasing, and transformation legality.
//! Host preparation reads those proofs to freeze dispatch recipes and storage requirements.
//! SSA emission instantiates the recipes with concrete bindings and operands;
//! physical eligibility and kernel policies remain queryable in egglog.
use self::dataflow::edit;
use self::query::Query;
use crate::egglog::graph::Graph;
use crate::ssa::stage::Elaborated;
use crate::tlc::stage::OwnershipApplied;
use crate::{CodegenTarget, PipelineTopologyPolicy, SymbolId};
use egglog_engine::Error;
use wyn_base::Interner;

mod analysis;
mod bindings;
pub mod cost;
pub(crate) mod dataflow;
mod execution;
mod from_tlc;
mod fusion;
mod graph;
mod host;
mod memory;
mod payload;
mod planning;
mod position_field;
mod query;
mod representation;
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
        costs: cost::CostModel::default(),
        _stage: Imported,
    })
}

/// Simplify regions and fuse parallel statements in the common IR.
pub fn fuse(mut program: Program<'_, Imported>) -> Result<Program<'_, Fused>, OptimizeError> {
    memory::inline(&mut program.graph)?;
    memory::simplify(&mut program.graph, false)?;
    planning::analyze(&mut program.graph)?;
    program.graph.parse_and_run_program(
        None,
        "(run-schedule (saturate regularity-index) (saturate regularity))",
    )?;
    if query::Query(&program.graph).first("IrregularResult")?.is_some() {
        return Err(output_error(
            "map callback results must have the same array shape for every input element",
        ));
    }
    // Fuse adjacent producers before hoisting invariant work between them.
    loop {
        let revision = edit::revision(&program.graph)?;
        fusion::run(&mut program.graph)?;
        memory::simplify(&mut program.graph, true)?;
        if edit::revision(&program.graph)? == revision {
            break;
        }
    }
    analysis::ensure(&mut program.graph)?;
    Ok(program.advance(Fused))
}

/// Choose execution domains and rematerialization. This is execution placement;
/// scalar loop-invariant and common-branch hoisting belongs to the local optimizer.
pub fn place(
    program: Program<'_, Optimized>,
    topology: PipelineTopologyPolicy,
) -> Result<Program<'_, Placed>, OptimizeError> {
    place_with_cost_model(program, topology, cost::CostModel::default())
}

pub fn place_with_cost_model(
    mut program: Program<'_, Optimized>,
    topology: PipelineTopologyPolicy,
    costs: cost::CostModel,
) -> Result<Program<'_, Placed>, OptimizeError> {
    let costs = costs.validate().map_err(output_error)?;
    let _timing = timing::span("egglog placement");
    planning::place(&mut program.graph, topology, costs)?;
    Ok(program.advance(Placed))
}

/// Select logical execution strategies; host preparation completes physical planning.
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
    planning::schedule(&mut program.graph)?;
    program.costs = costs;
    Ok(program.advance(Scheduled))
}

/// Validate memory bindings after executable rewrites and logical scheduling.
pub fn prepare_interfaces(
    program: Program<'_, Scheduled>,
) -> Result<Program<'_, InterfacesPrepared>, OptimizeError> {
    let _timing = timing::span("egglog interface preparation");
    // Scheduling only changes execution annotations; placement's dataflow proofs remain valid.
    analysis::validate(&program.graph)?;
    Ok(program.advance(InterfacesPrepared))
}

/// Simplify the authoritative statement bodies.
pub fn optimize(program: Program<'_, Fused>) -> Result<Program<'_, Optimized>, OptimizeError> {
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
    mut program: Program<'_, Fused>,
    policy: ScalarOptimization,
) -> Result<Program<'_, Optimized>, OptimizeError> {
    scalar::run(&mut program.graph, policy)?;
    analysis::ensure(&mut program.graph)?;
    timing::tables(&program.graph, "after-optimization");
    Ok(program.advance(Optimized))
}

/// Finish graph analysis and freeze physical execution and storage choices.
pub fn prepare_host(
    mut program: Program<'_, InterfacesPrepared>,
) -> Result<HostPrepared<'_>, OptimizeError> {
    let _timing = timing::span("egglog host preparation");
    let query = Query(&program.graph);
    let mut roots = Vec::new();
    query.for_function("EntryOrder", |keys, _| {
        roots.push(query.required("FunctionRoot", keys[0])?);
        Ok(())
    })?;
    analysis::liveness(&mut program.graph, &roots)?;
    planning::epilogues(&mut program.graph)?;
    let execution = execution::Execution::prepare(&mut program)?;
    Ok(HostPrepared { program, execution })
}

/// Emit shader SSA and host interfaces from the prepared physical plan.
pub fn to_ssa(program: HostPrepared<'_>, target: CodegenTarget) -> Result<Elaborated, OptimizeError> {
    let _timing = timing::span("SSA emission");
    to_ssa::lower(&program.program, &program.execution, target)
}

/// TLC retains source and public-interface metadata. The graph owns the logical
/// program and logical execution choices. Host preparation freezes the physical plan.
pub struct Program<'source, Stage> {
    source: &'source OwnershipApplied,
    graph: Graph,
    symbols: Interner<i64, SymbolId>,
    costs: cost::CostModel,
    _stage: Stage,
}

impl<'source, Stage> Program<'source, Stage> {
    fn advance<Next>(self, stage: Next) -> Program<'source, Next> {
        Program {
            source: self.source,
            graph: self.graph,
            symbols: self.symbols,
            costs: self.costs,
            _stage: stage,
        }
    }
}

pub struct Imported;
pub struct Fused;
pub struct Placed;

/// Executable rewrites are complete and logical execution strategies are selected.
pub struct Scheduled;

pub struct InterfacesPrepared;

pub struct Optimized;

pub struct HostPrepared<'source> {
    program: Program<'source, InterfacesPrepared>,
    execution: execution::Execution,
}

#[cfg(test)]
mod pipeline_tests;

#[cfg(test)]
fn optimize_for_test(program: Program<'_, Fused>) -> Result<Program<'_, Optimized>, OptimizeError> {
    optimize_with_policy(program, ScalarOptimization::Basic)
}

#[cfg(test)]
fn finish_for_test(
    program: Program<'_, InterfacesPrepared>,
    target: CodegenTarget,
) -> Result<Elaborated, OptimizeError> {
    let program = prepare_host(program)?;
    to_ssa(program, target)
}

pub(super) fn output_error(message: impl Into<String>) -> OptimizeError {
    OptimizeError::Output(message.into())
}
