//! HOF-specialization phase boundary (phase 2 of the defunctionalization split).
//!
//! Owns the post-condition verifier for HOF specialization: every
//! reachable top-level def has zero function-typed parameters. Wired in
//! post-`filter_reachable`, where dead unspecialized HOF definitions
//! have already been removed.
//!
//! The actual specialization logic still lives inside
//! `tlc::defunctionalize` (tightly coupled with lambda lifting and
//! StaticVal tracking). This module owns the architectural seam — the
//! verifier guarantees the existing pipeline produces well-shaped output
//! and gives a target shape for any future extraction of the
//! specialization phase.

use super::{Def, Program};
use crate::SymbolId;
use crate::ast::TypeName;
use polytype::Type;

#[derive(Debug)]
pub enum HofSpecializeError {
    /// A top-level def has a function-typed parameter. Every HOF should
    /// have been specialized away into monomorphic copies by this point.
    FunctionTypedParam {
        def: SymbolId,
        param_index: usize,
    },
}

pub fn verify_hof_specialized(program: &Program) -> Result<(), HofSpecializeError> {
    for def in &program.defs {
        verify_def_no_arrow_params(def)?;
    }
    Ok(())
}

fn verify_def_no_arrow_params(def: &Def) -> Result<(), HofSpecializeError> {
    let mut current = &def.ty;
    let mut param_index = 0;
    while let Type::Constructed(TypeName::Arrow, args) = current {
        if args.len() != 2 {
            break;
        }
        if is_arrow_type(&args[0]) {
            return Err(HofSpecializeError::FunctionTypedParam {
                def: def.name,
                param_index,
            });
        }
        current = &args[1];
        param_index += 1;
    }
    Ok(())
}

fn is_arrow_type(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::Arrow, _))
}

#[cfg(test)]
#[path = "hof_specialize_tests.rs"]
mod hof_specialize_tests;
