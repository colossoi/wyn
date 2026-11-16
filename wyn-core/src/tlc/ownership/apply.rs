//! Finish the validated TLC boundary while retaining parameter consumption diets.
use crate::tlc;
use crate::tlc::stage::GeneratedLambdasFolded;

#[derive(Debug, Clone, Copy)]
pub enum OwnershipAppliedTag {}

pub type OwnershipApplied =
    tlc::Program<OwnershipAppliedTag, tlc::family::ClosureConverted, tlc::context::PostClosureGlobal>;

pub fn apply_ownership(program: GeneratedLambdasFolded) -> OwnershipApplied {
    program.rebuild(|source| source, |def, _| def)
}
