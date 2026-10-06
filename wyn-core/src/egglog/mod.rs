//! Compiler mid-end using [egglog](https://github.com/egraphs-good/egglog).
//! Pass order is enforced by typestate:
//! [`from_tlc`] → [`fuse`] → [`place`] → [`schedule`] → [`optimize`] → [`to_ssa`].
use crate::host::ScalarExpr;
use crate::ssa::stage::Elaborated;
use crate::tlc::stage::InputSliceBoundsInferred;
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

/// Summarize semantic values, operations, and execution scopes in one TLC walk.
/// Resolve lexical bindings directly; scalar bodies remain source references.
pub fn from_tlc(source: &InputSliceBoundsInferred) -> Result<Program<'_, Imported>, OptimizeError> {
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

/// Optimize demanded scalar regions after structural scheduling.
pub fn optimize(program: Program<'_, Scheduled>) -> Result<Program<'_, Optimized>, OptimizeError> {
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
    mut program: Program<'_, Scheduled>,
    policy: ScalarOptimization,
) -> Result<Program<'_, Optimized>, OptimizeError> {
    let (selected, scalars) = scalar::run(&mut program.graph, &program.identities, policy)?;
    let mut program = program.advance(Optimized {
        selected,
        scalars,
        host: LookupMap::default(),
        captures: LookupMap::default(),
    });
    facts::prepare(&mut program)?;
    host::prepare(&mut program)?;
    Ok(program)
}

/// Emit the optimized scalar graph directly into SSA.
pub fn to_ssa(program: Program<'_, Optimized>, target: CodegenTarget) -> Result<Elaborated, OptimizeError> {
    to_ssa::lower(&program, target)
}

/// Original TLC supplies untouched bodies and source metadata. The graph owns
/// structural transformations and planning facts; Rust does not mirror its terms.
pub struct Program<'source, Stage> {
    source: &'source InputSliceBoundsInferred,
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
/// Scalar optimization finalizes host execution and the physical ABI.
pub struct Scheduled;

pub struct Optimized {
    captures: LookupMap<egglog_engine::TermId, host::Capture>,
    host: LookupMap<(Value, Value), ScalarExpr>,
    selected: scalar::Selected,
    // Shares immutable source/context identities with the structural graph.
    // Rewrites and placement proofs belong to this scalar snapshot alone.
    scalars: EGraph,
}

#[cfg(test)]
mod pipeline_tests;

fn parse_program(filename: &str, source: &str) -> Result<Vec<Command>, OptimizeError> {
    Parser::default()
        .get_program_from_string(Some(filename.into()), source)
        .map_err(|error| OptimizeError::Output(error.to_string()))
}

pub(super) fn output_error(message: impl Into<String>) -> OptimizeError {
    OptimizeError::Output(message.into())
}
