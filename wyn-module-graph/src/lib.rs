//! Syntax-independent package and source-module graph construction.
//!
//! This crate owns physical source identity and import topology. Parsing Wyn
//! syntax remains the responsibility of an injected frontend, while source I/O
//! remains the responsibility of an injected provider.

#![forbid(unsafe_code)]

mod error;
mod graph;
mod ids;
mod path;
mod plan;
mod source;

pub use error::{BuildError, BuildFailure};
pub use graph::{
    load_modules, BuildResult, ImportEdge, ImportTarget, ImportTraceFrame, LoadedModule, ModuleFrontend,
    ModuleGraph, SourceProvider,
};
pub use ids::{ImportSiteId, ModuleId, PackageId};
pub use path::{ModulePath, PathError, RelativeModulePath};
pub use plan::{
    AliasError, Dependency, DependencyAlias, IdentityError, ModuleKey, Package, PackageIdentity,
    PackagePlan, PackagePlanBuilder, PlanError, SourceFingerprint,
};
pub use source::{SourceLocation, SourceTextError, Span, SpanError, TextRange};

#[cfg(test)]
mod tests;
