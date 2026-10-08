//! Export ownership permission at the immutable TLC boundary.

use super::facts::{collect, OwnershipFacts};
use crate::tlc;
use crate::tlc::stage::GeneratedLambdasFolded;

#[derive(Debug, Clone, Copy)]
pub enum OwnershipAppliedTag {}

#[derive(Debug, Clone)]
pub struct OwnershipGlobal {
    pub source: tlc::context::PostClosureGlobal,
    pub ownership: OwnershipFacts,
}

pub type OwnershipApplied =
    tlc::Program<OwnershipAppliedTag, tlc::family::ClosureConverted, OwnershipGlobal>;

/// Preserve the functional program and export permission on values. Fusion
/// cannot invalidate this permission by eliminating an intermediate SOAC.
/// Egglog checks physical reuse against the selected plan; local array updates
/// are committed only after SSA placement and alias/liveness analysis.
pub fn apply_ownership(program: GeneratedLambdasFolded) -> OwnershipApplied {
    let ownership = collect(&program);
    program.rebuild(|source| OwnershipGlobal { source, ownership }, |def, _| def)
}
