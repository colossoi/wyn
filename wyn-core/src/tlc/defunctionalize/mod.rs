//! TLC defunctionalization.
//!
//! This is one externally visible phase transition with three internal
//! algorithms:
//!
//! 1. lambda lifting and explicit closure/capture construction;
//! 2. higher-order-function specialization;
//! 3. lowering explicit closure applications to direct calls.
//!
//! No analysis sidecar crosses those boundaries. Callable environments are
//! stored on `TermKind::Closure` and SOAC bodies.

mod closure_convert;
mod hof_specialize;
mod lower_calls;

use super::{Family, Stage};

pub(crate) use hof_specialize::verify_hof_specialized;

/// TLC whose closure values and SOAC environments are explicit.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClosureConverted;

impl Family for ClosureConverted {
    type DefinitionData = ();
    type EntryData = ();
    type ClosureData = super::data::ExplicitClosurePayload;
    type SoacBodyData = super::data::ExplicitCapturesPayload;
}

/// TLC after all higher-order values have been replaced by direct-callable
/// code plus explicit environments.
#[derive(Debug, Clone, Copy, Default)]
pub struct Defunctionalized;

impl Stage for Defunctionalized {
    type Family = ClosureConverted;
    type GlobalContext = super::context::PostClosureGlobal;
}

pub fn run(
    program: super::Program<super::stage::RuntimeIndexProducersFloated>,
) -> super::Program<Defunctionalized> {
    let mut program = closure_convert::run(program);
    hof_specialize::run(&mut program);
    lower_calls::run(&mut program);
    program
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod pipeline_tests;
