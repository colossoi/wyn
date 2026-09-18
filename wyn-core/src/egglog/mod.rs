//! Compiler mid-end using [egglog](https://github.com/egraphs-good/egglog).
//!
//! [`from_tlc()`] accepts normalized TLC with inferred input bounds and
//! returns an egglog fusion graph together with [`Ir`]. [`fuse`]
//! applies fusion decisions to the sidecar. [`insert_expressions`] adds a
//! separate typed expression DAG with region uses and dependency facts.
//! [`simplify`] uses equality saturation for scalar algebra. [`place`] records
//! common-branch and loop-invariant placements, including SOAC captures.
//! [`schedule`] derives logical stages, residency, backing, scratch requirements,
//! and dispatch constraints in egglog before building functions and blocks.
//! Resources, aliases, capacities, accesses and dispatch order are read from
//! that plan. Generated scalar instructions remain opaque sidecar bodies.
//! [`to_ssa`] preserves source inputs and publishes static compute pipelines
//! through the shared shader/runtime ABI. Runtime capacity formulas, host
//! control flow and graphics publication still have explicit limitations.
//! Map, reduce, and scan are constructed as Scremas during import. Types and
//! pure values are interned in the sidecar. Fusion exports only SOAC layouts,
//! uses, dependencies and motion constraints. Each pass owns its egglog graph;
//! extracted expressions and scheduled blocks remain in the IR.
//!
//! The schemas are defined in `fusion/schema.egg`, `expressions.egg`, and
//! `planning.egg`. Identities are local to one conversion.
use egglog_engine::ast::{Command, Parser};
use egglog_engine::Error;

mod abi;
mod blocks;
mod data;
mod dependencies;
mod expressions;
pub mod from_tlc;
mod fusion;
mod planning;
mod publish;
mod regions;
mod rewrite;
mod scalar;
mod schedule;
mod stage;
pub use stage::{Expressions, Fused, Imported, Placed, Program, Scheduled, Simplified};
mod term;
mod timing;
mod to_ssa;
mod visit;

pub use blocks::{
    BlockData, BodyData, BufferData, DispatchData, Edge, Exit, Function, FunctionKind, GridData,
    Instruction, Storage, Value,
};
pub use data::{
    Array, BlockId, BodyId, BucketShapeData, BucketShapeId, BufferId, BuiltinData, BuiltinId,
    DefinitionData, DefinitionId, DefinitionKind, DispatchId, EntryData, EntryId, EntryParamData,
    EntryParamId, ExprData, ExprId, ExprKind, ExternData, ExternId, GridId, InputBoundData, InputBoundId,
    Ir, LoopKind, OperationData, OperationId, OperationKind, OriginData, OriginId, OutputData, OutputId,
    ParameterData, ParameterId, Place, PlacementData, PlacementId, PlacementSite, ProgramData, ProgramId,
    Reduction, RegionData, RegionId, Scan, ScremaForm, SoacBody, SymbolData, SymbolId, TypeData, TypeId,
};
pub use expressions::insert_expressions;
pub use from_tlc::{from_tlc, ConvertError};
pub use fusion::fuse;
pub use scalar::{place, simplify};
pub use schedule::schedule;
pub use timing::with_timings;
pub use to_ssa::to_ssa;

#[derive(Debug, thiserror::Error)]
pub enum OptimizeError {
    #[error("egglog optimization: {0}")]
    Engine(#[from] Error),
    #[error("egglog extraction: {0}")]
    Extraction(String),
    #[error("egglog output: {0}")]
    Output(String),
}

/// Declarations included at the start of fusion fact programs (egglog 3.0).
pub const SCHEMA: &str = concat!(include_str!("ids.egg"), "\n", include_str!("fusion/schema.egg"));

#[cfg(test)]
mod from_tlc_tests;
#[cfg(test)]
mod graph_tests;

#[cfg(test)]
pub(super) fn simplify_and_place(program: Program<Expressions>) -> Result<Program<Placed>, OptimizeError> {
    place(simplify(program, true)?)
}

fn parse_program(filename: &str, source: &str) -> Result<Vec<Command>, OptimizeError> {
    Parser::default()
        .get_program_from_string(Some(filename.into()), source)
        .map_err(|error| OptimizeError::Output(error.to_string()))
}
