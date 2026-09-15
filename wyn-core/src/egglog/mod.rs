//! Experimental TLC import into [egglog](https://github.com/egraphs-good/egglog).
//!
//! [`from_tlc::convert_program`] accepts the same TLC checkpoint as EGIR and
//! returns a standalone egglog program together with [`AssociatedData`]. No
//! optimization, extraction, or backend lowering is performed here.
//!
//! The emitted language is documented in `schema.egg`. Metadata identities are
//! local to one conversion; keep the program and its sidecar together. Each ID's
//! `egglog()` method renders its typed egglog constructor for use in queries.

mod data;
pub mod from_tlc;

pub use data::*;
pub use from_tlc::{convert_program, ConvertError, Converted};

/// Declarations included at the start of every emitted program (egglog 3.0).
pub const SCHEMA: &str = include_str!("schema.egg");

#[cfg(test)]
mod from_tlc_tests;
