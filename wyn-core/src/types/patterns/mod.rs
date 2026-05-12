//! Pattern matching subsystem: type-check pattern binding, coverage
//! analysis (exhaustiveness + redundancy), and refutability checks.
//!
//! Pattern logic is its own subsystem with no dependence on
//! value-handling concerns elsewhere in the type checker. Everything
//! in this submodule operates on `ast::Pattern` and `Type<TypeName>`.

pub mod bind;
pub mod coverage;
pub mod refutability;
