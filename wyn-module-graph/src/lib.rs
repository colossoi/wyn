//! Syntax-independent package and source-module graph construction.
//!
//! This crate owns physical source identity and import topology. Parsing Wyn
//! syntax remains the responsibility of an injected frontend, while source I/O
//! remains the responsibility of an injected reader.

#![forbid(unsafe_code)]

mod error;
mod graph;
mod ids;
mod input;
mod local_sources;
mod path;
mod plan;
mod source;

pub use error::{BuildError, BuildFailure};
pub use graph::{
    ImportEdge, ImportTarget, ImportTraceFrame, LoadedModule, ModuleGraph, ModuleParser, SourceGraph,
    SourceReader,
};
pub use ids::{ImportSiteId, ModuleId, PackageId};
pub use input::PackagePlan;
pub use local_sources::{LocalSourceError, LocalSources};
pub use path::{ModulePath, PathError, RelativeModulePath};
pub use plan::{
    AliasError, Dependency, DependencyAlias, IdentityError, ModuleKey, Package, PackageGraph,
    PackageGraphBuilder, PackageGraphError, PackageIdentity,
};
pub use source::{SourceLocation, SourceTextError, Span, SpanError, TextRange};

#[cfg(test)]
mod tests;
