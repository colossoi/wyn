use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use crate::graph::ImportTraceFrame;
use crate::ids::{ImportSiteId, ModuleId};
use crate::path::PathError;
use crate::plan::{DependencyAlias, ModuleKey, PackageGraph};
use crate::source::{SourceLocation, SourceMap, SourceTextError, Span, SpanError};

/// A graph-construction error together with every source buffer loaded before
/// construction stopped.
///
/// Keeping the partial source map makes all spans in the error and its import
/// trace usable by diagnostics, including parse and transitive load failures.
#[derive(Debug)]
pub struct BuildFailure<FrontendError, ProviderError> {
    error: BuildError<FrontendError, ProviderError>,
    context: Box<FailureContext>,
}

#[derive(Debug)]
struct FailureContext {
    packages: PackageGraph,
    module_ids: HashMap<ModuleKey, ModuleId>,
    sources: SourceMap,
}

impl<FrontendError, ProviderError> BuildFailure<FrontendError, ProviderError> {
    pub(crate) fn new(
        error: BuildError<FrontendError, ProviderError>,
        packages: PackageGraph,
        module_ids: HashMap<ModuleKey, ModuleId>,
        sources: SourceMap,
    ) -> Self {
        Self {
            error,
            context: Box::new(FailureContext {
                packages,
                module_ids,
                sources,
            }),
        }
    }

    /// The structured reason graph construction stopped.
    pub const fn error(&self) -> &BuildError<FrontendError, ProviderError> {
        &self.error
    }

    /// The closed package graph used by the failed module-graph build.
    pub const fn package_graph(&self) -> &PackageGraph {
        &self.context.packages
    }

    /// The stable package-relative identity assigned to a discovered module.
    pub fn module_key(&self, module: ModuleId) -> Option<&ModuleKey> {
        self.context.module_ids.iter().find_map(|(key, &id)| (id == module).then_some(key))
    }

    /// Source text loaded for a module before construction stopped.
    pub fn source_text(&self, module: ModuleId) -> Option<&str> {
        self.context.sources.source(module)
    }

    /// Convert the start of a retained span to a user-facing location.
    pub fn location(&self, span: Span) -> Result<SourceLocation, SpanError> {
        self.context.sources.location(span)
    }

    /// Return the retained source text covered by a span.
    pub fn snippet(&self, span: Span) -> Result<&str, SpanError> {
        self.context.sources.snippet(span)
    }

    fn fmt_module(&self, formatter: &mut fmt::Formatter<'_>, key: &ModuleKey) -> fmt::Result {
        let root_package = self.context.packages.root().package();
        if key.package() == root_package {
            return write!(formatter, "{}", key.path());
        }

        let Some(package) = self.context.packages.package(key.package()) else {
            return write!(formatter, "{}", key.path());
        };
        write!(
            formatter,
            "{}@{}:{}",
            package.identity().canonical_name(),
            package.identity().version(),
            key.path()
        )
    }

    fn fmt_module_id(&self, formatter: &mut fmt::Formatter<'_>, module: ModuleId) -> fmt::Result {
        let Some(key) = self.module_key(module) else {
            return formatter.write_str("unknown source module");
        };
        self.fmt_module(formatter, key)
    }

    fn fmt_span(&self, formatter: &mut fmt::Formatter<'_>, span: Span) -> fmt::Result {
        let Some(module) = span.module() else {
            return formatter.write_str("generated syntax");
        };
        self.fmt_module_id(formatter, module)?;
        if let Ok(location) = self.location(span) {
            write!(formatter, ":{}:{}", location.line, location.column)?;
        }
        Ok(())
    }

    fn fmt_trace(&self, formatter: &mut fmt::Formatter<'_>, trace: &[ImportTraceFrame]) -> fmt::Result {
        for frame in trace {
            formatter.write_str("\n  imported from ")?;
            self.fmt_span(formatter, frame.span)?;
        }
        Ok(())
    }
}

impl<FrontendError: fmt::Display, ProviderError: fmt::Display> fmt::Display
    for BuildFailure<FrontendError, ProviderError>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.error {
            BuildError::Load {
                module,
                trace,
                source,
                ..
            } => {
                formatter.write_str("failed to load ")?;
                self.fmt_module(formatter, module)?;
                write!(formatter, ": {source}")?;
                self.fmt_trace(formatter, trace)
            }
            BuildError::SourceText {
                module,
                trace,
                source,
                ..
            } => {
                formatter.write_str("failed to store ")?;
                self.fmt_module(formatter, module)?;
                write!(formatter, ": {source}")?;
                self.fmt_trace(formatter, trace)
            }
            BuildError::Parse {
                module,
                trace,
                source,
            } => {
                formatter.write_str("failed to parse ")?;
                self.fmt_module_id(formatter, *module)?;
                write!(formatter, ": {source}")?;
                self.fmt_trace(formatter, trace)
            }
            BuildError::UnknownDependency {
                alias, span, trace, ..
            } => {
                self.fmt_span(formatter, *span)?;
                write!(formatter, ": unknown package dependency alias `{alias}`")?;
                self.fmt_trace(formatter, trace)
            }
            BuildError::InvalidPath {
                span, trace, source, ..
            } => {
                self.fmt_span(formatter, *span)?;
                write!(formatter, ": invalid import path: {source}")?;
                self.fmt_trace(formatter, trace)
            }
            BuildError::InvalidImportSpan {
                span, trace, source, ..
            } => {
                self.fmt_span(formatter, *span)?;
                write!(formatter, ": invalid import source range: {source}")?;
                self.fmt_trace(formatter, trace)
            }
            BuildError::DuplicateImportSite {
                site, span, trace, ..
            } => {
                self.fmt_span(formatter, *span)?;
                write!(formatter, ": duplicate import site {site:?}")?;
                self.fmt_trace(formatter, trace)
            }
            BuildError::Cycle { edges } => {
                formatter.write_str("source import cycle")?;
                for edge in edges {
                    formatter.write_str("\n  ")?;
                    self.fmt_span(formatter, edge.span)?;
                    formatter.write_str(" imports ")?;
                    self.fmt_module(formatter, &edge.requested)?;
                }
                Ok(())
            }
        }
    }
}

impl<FrontendError, ProviderError> Error for BuildFailure<FrontendError, ProviderError>
where
    FrontendError: Error + 'static,
    ProviderError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.error)
    }
}

/// Failure produced while constructing a physical source-module graph.
#[derive(Debug)]
pub enum BuildError<FrontendError, ProviderError> {
    Load {
        module: ModuleKey,
        trace: Box<[ImportTraceFrame]>,
        source: ProviderError,
    },
    SourceText {
        module: ModuleKey,
        trace: Box<[ImportTraceFrame]>,
        source: SourceTextError,
    },
    Parse {
        module: ModuleId,
        trace: Box<[ImportTraceFrame]>,
        source: FrontendError,
    },
    UnknownDependency {
        alias: DependencyAlias,
        span: Span,
        trace: Box<[ImportTraceFrame]>,
    },
    InvalidPath {
        span: Span,
        trace: Box<[ImportTraceFrame]>,
        source: PathError,
    },
    InvalidImportSpan {
        site: ImportSiteId,
        span: Span,
        trace: Box<[ImportTraceFrame]>,
        source: SpanError,
    },
    DuplicateImportSite {
        site: ImportSiteId,
        span: Span,
        trace: Box<[ImportTraceFrame]>,
    },
    Cycle {
        edges: Box<[ImportTraceFrame]>,
    },
}

impl<FrontendError: fmt::Display, ProviderError: fmt::Display> fmt::Display
    for BuildError<FrontendError, ProviderError>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load { module, source, .. } => {
                write!(formatter, "failed to load {}: {source}", module.path())
            }
            Self::SourceText { module, source, .. } => {
                write!(formatter, "failed to store {}: {source}", module.path())
            }
            Self::Parse { module, source, .. } => {
                write!(formatter, "failed to parse {module:?}: {source}")
            }
            Self::UnknownDependency { alias, .. } => {
                write!(formatter, "unknown package dependency alias `{alias}`")
            }
            Self::InvalidPath { source, .. } => write!(formatter, "invalid import path: {source}"),
            Self::InvalidImportSpan { source, .. } => {
                write!(formatter, "invalid import source range: {source}")
            }
            Self::DuplicateImportSite { site, .. } => {
                write!(formatter, "duplicate import site {site:?}")
            }
            Self::Cycle { .. } => formatter.write_str("source import cycle"),
        }
    }
}

impl<FrontendError, ProviderError> Error for BuildError<FrontendError, ProviderError>
where
    FrontendError: Error + 'static,
    ProviderError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Load { source, .. } => Some(source),
            Self::SourceText { source, .. } => Some(source),
            Self::Parse { source, .. } => Some(source),
            Self::InvalidPath { source, .. } => Some(source),
            Self::InvalidImportSpan { source, .. } => Some(source),
            Self::UnknownDependency { .. } | Self::DuplicateImportSite { .. } | Self::Cycle { .. } => None,
        }
    }
}

#[cfg(test)]
#[path = "error_tests.rs"]
mod error_tests;
