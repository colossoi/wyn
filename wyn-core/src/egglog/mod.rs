//! Compiler mid-end using [egglog](https://github.com/egraphs-good/egglog).
//! Pass order is enforced by typestate:
//! [`from_tlc`] → [`fuse`] → [`place`] → [`schedule`] → [`prepare_interfaces`]
//! → [`optimize`] → [`prepare_host`] → [`to_ssa`].
use crate::host::ScalarExpr;
use crate::ssa::stage::Elaborated;
use crate::tlc::stage::OwnershipApplied;
use crate::{CodegenTarget, LookupMap, PipelineTopologyPolicy};
use egglog_engine::ast::{Command, Parser};
use egglog_engine::Error;
use egglog_engine::{EGraph, Value};

mod abi;
mod analysis;
mod bindings;
mod facts;
mod fusion;
mod host;
mod planning;
mod query;
mod scalar;
mod source;
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

/// Summarize semantic values, operations, and execution scopes from typed TLC.
/// Resolve lexical bindings directly; scalar bodies remain source references.
pub fn from_tlc(source: &OwnershipApplied) -> Result<Program<'_, Imported>, OptimizeError> {
    let _timing = timing::span("egglog import");
    let (graph, identities) = source::import(source)?;
    Ok(Program {
        source,
        graph,
        identities,
        stage: Imported,
    })
}

/// Apply the retained fusion rules and record callback composition in egglog.
pub fn fuse(mut program: Program<'_, Imported>) -> Result<Program<'_, Fused>, OptimizeError> {
    timing::time("egglog structural analysis", || analysis::run(&mut program.graph))?;
    fusion::run(&mut program.graph)?;
    Ok(program.advance(Fused))
}

/// Choose execution domains and rematerialization. This is execution placement;
/// scalar loop-invariant and common-branch hoisting belongs to the local optimizer.
pub fn place(
    mut program: Program<'_, Fused>,
    topology: PipelineTopologyPolicy,
) -> Result<Program<'_, Placed>, OptimizeError> {
    let _timing = timing::span("egglog placement");
    planning::place(&mut program.graph, topology)?;
    Ok(program.advance(Placed))
}

/// Select dispatch recipes, resource requirements, and ordering constraints.
pub fn schedule(mut program: Program<'_, Placed>) -> Result<Program<'_, Scheduled>, OptimizeError> {
    let _timing = timing::span("egglog scheduling");
    planning::schedule(&mut program.graph)?;
    Ok(program.advance(Scheduled))
}

/// Resolve demanded scalar inputs and the physical ABI before scalar optimization.
pub fn prepare_interfaces(
    mut program: Program<'_, Scheduled>,
) -> Result<Program<'_, InterfacesPrepared>, OptimizeError> {
    let _timing = timing::span("egglog interface preparation");
    scalar::admit(&mut program.graph)?;
    // Physical ABI and publication rules consume the scalar demand facts.
    abi::prepare(&mut program.graph)?;
    Ok(program.advance(InterfacesPrepared))
}

/// Optimize and extract demanded scalar regions.
pub fn optimize(program: Program<'_, InterfacesPrepared>) -> Result<Program<'_, Optimized>, OptimizeError> {
    optimize_with_policy(program, ScalarOptimization::Full)
}

/// Basic retains reducing rewrites and safety analysis; Full additionally searches
/// helper expansions and compares compact and inlining-preferred extractions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarOptimization {
    Basic,
    Full,
}

pub fn optimize_with_policy(
    program: Program<'_, InterfacesPrepared>,
    policy: ScalarOptimization,
) -> Result<Program<'_, Optimized>, OptimizeError> {
    let (selected, scalars) = scalar::run(&program.graph, &program.identities, policy)?;
    Ok(program.advance(Optimized {
        selected,
        scalars,
        host: LookupMap::default(),
        captures: LookupMap::default(),
    }))
}

/// Lower supported host scalar expressions and captures from the selected DAG.
pub fn prepare_host(mut program: Program<'_, Optimized>) -> Result<HostPrepared<'_>, OptimizeError> {
    let _timing = timing::span("egglog host preparation");
    facts::prepare(&mut program)?;
    host::prepare(&mut program)?;
    Ok(HostPrepared { program })
}

/// Plan evaluation and lower shader SSA together with its host interfaces.
pub fn to_ssa(program: HostPrepared<'_>, target: CodegenTarget) -> Result<Elaborated, OptimizeError> {
    to_ssa::lower(&program.program, target)
}

/// Original TLC supplies untouched bodies and source metadata. The graph owns
/// structural transformations and planning facts; Rust does not mirror its terms.
pub struct Program<'source, Stage> {
    source: &'source OwnershipApplied,
    graph: EGraph,
    identities: source::Identities<'source>,
    stage: Stage,
}

impl<'source, Stage> Program<'source, Stage> {
    fn advance<Next>(self, stage: Next) -> Program<'source, Next> {
        Program {
            source: self.source,
            graph: self.graph,
            identities: self.identities,
            stage,
        }
    }
}

pub struct Imported;
pub struct Fused;
pub struct Placed;

/// Dispatch recipes, execution domains, storage, and effect order are fixed.
pub struct Scheduled;

pub struct InterfacesPrepared;

pub struct Optimized {
    captures: LookupMap<egglog_engine::TermId, host::Capture>,
    host: LookupMap<(Value, Value), ScalarExpr>,
    selected: scalar::Selected,
    // Shares immutable source/context identities with the structural graph.
    // Rewrites and placement proofs belong to this scalar snapshot alone.
    scalars: EGraph,
}

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
