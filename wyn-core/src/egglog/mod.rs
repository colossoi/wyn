//! Experimental TLC import into [egglog](https://github.com/egraphs-good/egglog).
//!
//! [`from_tlc::convert_program`] accepts the same TLC checkpoint as EGIR and
//! returns an egglog fusion graph together with [`AssociatedData`]. [`optimize`]
//! applies fusion decisions to the sidecar. [`schedule`] then selects parallel
//! GPU recipes and builds functions, blocks, buffers and dispatches. Scalar
//! bodies remain opaque in both fact languages. Shader emission is deferred.
//! Map, reduce, and scan are constructed as Scremas during import. Types and
//! pure values are interned in the sidecar. Fusion exports only SOAC layouts,
//! uses, dependencies and motion constraints; scheduling exports block topology.
//!
//! The languages are documented in `schema.egg` and `blocks.egg`. Identities are
//! local to one conversion; keep each fact program and its sidecar together.

mod blocks;
mod data;
mod emit;
mod extract;
pub mod from_tlc;
mod optimize;
mod schedule;
mod snapshot;

pub use blocks::{
    BlockData, BodyData, BufferData, DispatchData, Edge, Exit, Function, FunctionKind, GridData,
    Instruction, Storage, Value,
};
pub use data::*;
pub use from_tlc::{convert_program, ConvertError, Converted};
pub use optimize::{optimize, OptimizeError};
pub use schedule::{readout, schedule};

/// Declarations included at the start of fusion fact programs (egglog 3.0).
pub const SCHEMA: &str = include_str!("schema.egg");

#[cfg(test)]
mod from_tlc_tests;
#[cfg(test)]
mod fusion_tests;
#[cfg(test)]
mod graph_tests;
