use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use crate::graph::ImportTraceFrame;
use crate::ids::{ImportSiteId, ModuleId};
use crate::path::PathError;
use crate::plan::{DependencyAlias, ModuleKey, PackagePlan};
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
    plan: PackagePlan,
    module_ids: HashMap<ModuleKey, ModuleId>,
    sources: SourceMap,
}

impl<FrontendError, ProviderError> BuildFailure<FrontendError, ProviderError> {
    pub(crate) fn new(
        error: BuildError<FrontendError, ProviderError>,
        plan: PackagePlan,
        module_ids: HashMap<ModuleKey, ModuleId>,
        sources: SourceMap,
    ) -> Self {
        Self {
            error,
            context: Box::new(FailureContext {
                plan,
                module_ids,
                sources,
            }),
        }
    }

    /// The structured reason graph construction stopped.
    pub const fn error(&self) -> &BuildError<FrontendError, ProviderError> {
        &self.error
    }

    /// The closed package plan used by the failed graph build.
    pub const fn plan(&self) -> &PackagePlan {
        &self.context.plan
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
}

impl<FrontendError: fmt::Display, ProviderError: fmt::Display> fmt::Display
    for BuildFailure<FrontendError, ProviderError>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, formatter)
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
        requested_at: Option<Span>,
        trace: Box<[ImportTraceFrame]>,
        source: ProviderError,
    },
    SourceText {
        module: ModuleKey,
        requested_at: Option<Span>,
        trace: Box<[ImportTraceFrame]>,
        source: SourceTextError,
    },
    Parse {
        module: ModuleId,
        trace: Box<[ImportTraceFrame]>,
        source: FrontendError,
    },
    UnknownDependency {
        from: ModuleId,
        site: ImportSiteId,
        alias: DependencyAlias,
        span: Span,
    },
    InvalidPath {
        from: ModuleId,
        site: ImportSiteId,
        span: Span,
        source: PathError,
    },
    InvalidImportSpan {
        from: ModuleId,
        site: ImportSiteId,
        source: SpanError,
    },
    DuplicateImportSite {
        from: ModuleId,
        site: ImportSiteId,
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
