//! Experimental TLC import into [egglog](https://github.com/egraphs-good/egglog).
//!
//! [`from_tlc::convert_program`] accepts the same TLC checkpoint as EGIR and
//! returns an egglog fusion graph together with [`AssociatedData`]. [`optimize`]
//! applies fusion decisions to the sidecar. [`insert_expressions`] adds a
//! separate typed expression DAG with region uses and dependency facts.
//! [`optimize_expressions`] uses equality saturation for scalar algebra and
//! records common-branch and loop-invariant placements, including SOAC captures.
//! [`schedule`] then selects parallel GPU recipes and builds functions, blocks,
//! buffers and dispatches, retaining the expression layer. Generated scalar
//! instructions remain opaque sidecar bodies. [`to_ssa`] hands those bodies to
//! the existing shader backend with a provisional storage interface.
//! Map, reduce, and scan are constructed as Scremas during import. Types and
//! pure values are interned in the sidecar. Fusion exports only SOAC layouts,
//! uses, dependencies and motion constraints; scheduling exports block topology.
//!
//! The languages are documented in `schema.egg`, `expressions.egg`, and
//! `blocks.egg`. Identities are local to one conversion; keep each fact program
//! and its sidecar together.

mod blocks;
mod data;
mod emit;
mod expressions;
mod extract;
pub mod from_tlc;
mod fusion;
mod optimize;
mod rewrite;
mod scalar;
mod schedule;
mod snapshot;
mod timing;
mod to_ssa;

pub use blocks::{
    BlockData, BodyData, BufferData, DispatchData, Edge, Exit, Function, FunctionKind, GridData,
    Instruction, Storage, Value,
};
pub use data::*;
pub use expressions::insert_expressions;
pub use from_tlc::{convert_program, ConvertError, Converted};
pub use optimize::{optimize, OptimizeError};
pub use scalar::optimize_expressions;
pub use schedule::schedule;
pub use timing::with_timings;
pub use to_ssa::to_ssa;

/// Declarations included at the start of fusion fact programs (egglog 3.0).
pub const SCHEMA: &str = concat!(include_str!("ids.egg"), "\n", include_str!("schema.egg"));

#[cfg(test)]
mod from_tlc_tests;
#[cfg(test)]
mod fusion_parity_tests;
#[cfg(test)]
mod fusion_tests;
#[cfg(test)]
mod graph_tests;
