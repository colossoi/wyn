use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use wyn_base::IdArena;

use crate::error::{BuildError, BuildFailure};
use crate::source::SourceMap;
use crate::{
    DependencyAlias, ImportSiteId, ModuleId, ModuleKey, PackageGraph, PackageId, PackagePlan,
    RelativeModulePath, SourceLocation, Span, SpanError, TextRange,
};

/// Supplies verified source text for a module in a closed package plan.
pub trait SourceReader {
    type Error;

    fn load(&mut self, module: &ModuleKey) -> Result<Arc<str>, Self::Error>;
}

/// Parses one source file and extracts its physical import requests.
pub trait ModuleParser {
    type Parsed;
    type Error;

    fn parse(
        &mut self,
        module: ModuleId,
        source: &str,
        report_import: &mut dyn FnMut(ImportSiteId, ImportTarget, TextRange),
    ) -> Result<Self::Parsed, Self::Error>;
}

/// One physical import discovered by a frontend.
struct ImportRequest {
    site: ImportSiteId,
    target: ImportTarget,
    range: TextRange,
}

/// Syntax-independent destination of a source import.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImportTarget {
    Local(RelativeModulePath),
    Dependency {
        alias: DependencyAlias,
        module: Option<RelativeModulePath>,
    },
}

/// A resolved source-module import edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportEdge {
    site: ImportSiteId,
    span: Span,
    target: ModuleId,
}

impl ImportEdge {
    pub const fn site(&self) -> ImportSiteId {
        self.site
    }

    pub const fn span(&self) -> Span {
        self.span
    }

    pub const fn target(&self) -> ModuleId {
        self.target
    }
}

/// One fully loaded source module and its frontend payload.
#[derive(Clone, Debug)]
pub struct LoadedModule<T> {
    key: ModuleKey,
    syntax: T,
    imports: Vec<ImportEdge>,
}

impl<T> LoadedModule<T> {
    pub const fn key(&self) -> &ModuleKey {
        &self.key
    }

    pub const fn syntax(&self) -> &T {
        &self.syntax
    }

    pub fn imports(&self) -> impl ExactSizeIterator<Item = &ImportEdge> {
        self.imports.iter()
    }
}

/// Immutable physical module graph produced from a closed package graph.
#[derive(Clone, Debug)]
pub struct ModuleGraph<T> {
    packages: PackageGraph,
    root: ModuleId,
    modules: IdArena<ModuleId, LoadedModule<T>>,
    dependency_order: Vec<ModuleId>,
    sources: SourceMap,
}

/// Physical source and import topology with frontend syntax discarded.
pub type SourceGraph = ModuleGraph<()>;

impl<T> ModuleGraph<T> {
    pub const fn package_graph(&self) -> &PackageGraph {
        &self.packages
    }

    pub const fn root(&self) -> ModuleId {
        self.root
    }

    pub fn module(&self, id: ModuleId) -> Option<&LoadedModule<T>> {
        self.modules.get(id)
    }

    pub fn modules(&self) -> impl Iterator<Item = (ModuleId, &LoadedModule<T>)> {
        self.modules.iter().map(|(&id, module)| (id, module))
    }

    pub fn package_of(&self, id: ModuleId) -> Option<PackageId> {
        self.module(id).map(|module| module.key.package())
    }

    pub fn source(&self, id: ModuleId) -> Option<&str> {
        self.sources.source(id)
    }

    pub fn import_target(&self, from: ModuleId, site: ImportSiteId) -> Option<ModuleId> {
        self.module(from)?.imports.iter().find(|edge| edge.site == site).map(|edge| edge.target)
    }

    pub fn modules_in_dependency_order(&self) -> impl ExactSizeIterator<Item = ModuleId> + '_ {
        self.dependency_order.iter().copied()
    }

    pub fn location(&self, span: Span) -> Result<SourceLocation, SpanError> {
        self.sources.location(span)
    }

    /// Format a source span using package-relative identity and a one-based
    /// line and column. Dependency modules include their package and version.
    pub fn display_location(&self, span: Span) -> Result<impl fmt::Display + '_, SpanError> {
        let module = span.module().ok_or(SpanError::GeneratedSpan)?;
        let loaded = self.module(module).ok_or(SpanError::UnknownModule { module })?;
        let location = self.location(span)?;
        Ok(DisplaySourceLocation {
            packages: &self.packages,
            module: loaded.key(),
            location,
        })
    }

    pub fn snippet(&self, span: Span) -> Result<&str, SpanError> {
        self.sources.snippet(span)
    }

    /// Discard frontend syntax while retaining package, module, import, and
    /// source provenance.
    pub fn erase_syntax(self) -> SourceGraph {
        let Self {
            packages,
            root,
            modules,
            dependency_order,
            sources,
        } = self;
        let mut source_modules = IdArena::new();
        for (module_id, module) in modules {
            let LoadedModule {
                key,
                syntax: _,
                imports,
            } = module;
            let copied_id = source_modules.alloc(LoadedModule {
                key,
                syntax: (),
                imports,
            });
            debug_assert_eq!(copied_id, module_id);
        }
        SourceGraph {
            packages,
            root,
            modules: source_modules,
            dependency_order,
            sources,
        }
    }
}

struct DisplaySourceLocation<'a> {
    packages: &'a PackageGraph,
    module: &'a ModuleKey,
    location: SourceLocation,
}

impl fmt::Display for DisplaySourceLocation<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.module.package() != self.packages.root().package() {
            if let Some(package) = self.packages.package(self.module.package()) {
                write!(
                    formatter,
                    "{}@{}:",
                    package.identity().canonical_name(),
                    package.identity().version()
                )?;
            }
        }
        write!(
            formatter,
            "{}:{}:{}",
            self.module.path(),
            self.location.line,
            self.location.column
        )
    }
}

/// One edge in an import provenance or cycle trace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportTraceFrame {
    pub span: Span,
    pub requested: ModuleKey,
}

/// Result type returned by physical module-graph construction.
pub type BuildResult<Parsed, FrontendError, ProviderError> =
    Result<ModuleGraph<Parsed>, BuildFailure<FrontendError, ProviderError>>;

impl<S: SourceReader> PackagePlan<S> {
    /// Load the complete source closure using the supplied language frontend.
    pub fn load<F: ModuleParser>(self, frontend: &mut F) -> BuildResult<F::Parsed, F::Error, S::Error> {
        let PackagePlan {
            package_graph,
            mut sources,
        } = self;
        let mut traversal = GraphTraversal {
            modules: IdArena::new(),
            by_key: HashMap::new(),
            active: Vec::new(),
            dependency_order: Vec::new(),
            source_map: SourceMap::default(),
        };

        let root_key = package_graph.root().clone();
        let root = match traversal.visit_module(&package_graph, &mut sources, frontend, root_key, None) {
            Ok(root) => root,
            Err(error) => {
                return Err(BuildFailure::new(
                    error,
                    package_graph,
                    traversal.by_key,
                    traversal.source_map,
                ));
            }
        };
        let mut modules = IdArena::new();
        for (id, module) in traversal.modules {
            let module = match module {
                Some(module) => module,
                None => panic!("module loader completed with an unfinished module"),
            };
            let copied_id = modules.alloc(module);
            debug_assert_eq!(copied_id, id);
        }
        Ok(ModuleGraph {
            packages: package_graph,
            root,
            modules,
            dependency_order: traversal.dependency_order,
            sources: traversal.source_map,
        })
    }
}

struct ActiveFrame {
    module: ModuleId,
    incoming: Option<ImportTraceFrame>,
}

struct GraphTraversal<T> {
    modules: IdArena<ModuleId, Option<LoadedModule<T>>>,
    by_key: HashMap<ModuleKey, ModuleId>,
    active: Vec<ActiveFrame>,
    dependency_order: Vec<ModuleId>,
    source_map: SourceMap,
}

impl<T> GraphTraversal<T> {
    fn visit_module<F, S>(
        &mut self,
        packages: &PackageGraph,
        sources: &mut S,
        frontend: &mut F,
        key: ModuleKey,
        incoming: Option<ImportTraceFrame>,
    ) -> Result<ModuleId, BuildError<F::Error, S::Error>>
    where
        F: ModuleParser<Parsed = T>,
        S: SourceReader,
    {
        if let Some(&module) = self.by_key.get(&key) {
            return match self.modules.get(module) {
                Some(Some(_)) => Ok(module),
                Some(None) => Err(BuildError::Cycle {
                    edges: self.cycle_trace(module, incoming).into_boxed_slice(),
                }),
                None => panic!("module key table refers to an unallocated module"),
            };
        }

        let module = self.modules.alloc(None);
        self.by_key.insert(key.clone(), module);
        self.active.push(ActiveFrame { module, incoming });

        let trace = self.import_trace();
        let text = sources.load(&key).map_err(|source| BuildError::Load {
            module: key.clone(),
            trace: trace.clone().into_boxed_slice(),
            source,
        })?;
        self.source_map.insert(module, text).map_err(|source| BuildError::SourceText {
            module: key.clone(),
            trace: trace.clone().into_boxed_slice(),
            source,
        })?;
        let mut requests = Vec::new();
        let source_text = self
            .source_map
            .source(module)
            .unwrap_or_else(|| panic!("source map lost a newly inserted module"));
        let syntax = frontend
            .parse(module, source_text, &mut |site, target, range| {
                requests.push(ImportRequest { site, target, range });
            })
            .map_err(|source| BuildError::Parse {
                module,
                trace: trace.into_boxed_slice(),
                source,
            })?;

        let mut sites = HashSet::new();
        let mut imports = Vec::with_capacity(requests.len());
        for request in requests {
            if !sites.insert(request.site) {
                return Err(BuildError::DuplicateImportSite {
                    site: request.site,
                    span: Span::new(module, request.range),
                    trace: self.import_trace().into_boxed_slice(),
                });
            }

            let span = Span::new(module, request.range);
            self.source_map.snippet(span).map_err(|source| BuildError::InvalidImportSpan {
                site: request.site,
                span,
                trace: self.import_trace().into_boxed_slice(),
                source,
            })?;
            let target_key = self.resolve_target(packages, &key, &request, span)?;
            let frame = ImportTraceFrame {
                span,
                requested: target_key.clone(),
            };
            let target = self.visit_module(packages, sources, frontend, target_key, Some(frame))?;
            imports.push(ImportEdge {
                site: request.site,
                span,
                target,
            });
        }

        let finished = self.active.pop();
        debug_assert_eq!(finished.as_ref().map(|frame| frame.module), Some(module));
        self.modules[module] = Some(LoadedModule { key, syntax, imports });
        self.dependency_order.push(module);
        Ok(module)
    }

    fn resolve_target<FrontendError, ProviderError>(
        &self,
        packages: &PackageGraph,
        from_key: &ModuleKey,
        request: &ImportRequest,
        span: Span,
    ) -> Result<ModuleKey, BuildError<FrontendError, ProviderError>> {
        match &request.target {
            ImportTarget::Local(relative) => {
                let path = from_key.path().resolve(relative).map_err(|source| BuildError::InvalidPath {
                    span,
                    trace: self.import_trace().into_boxed_slice(),
                    source,
                })?;
                Ok(ModuleKey::new(from_key.package(), path))
            }
            ImportTarget::Dependency { alias, module } => {
                let Some(package) =
                    packages.package(from_key.package()).and_then(|package| package.dependency(alias))
                else {
                    return Err(BuildError::UnknownDependency {
                        alias: alias.clone(),
                        span,
                        trace: self.import_trace().into_boxed_slice(),
                    });
                };
                let Some(target_package) = packages.package(package) else {
                    return Err(BuildError::UnknownDependency {
                        alias: alias.clone(),
                        span,
                        trace: self.import_trace().into_boxed_slice(),
                    });
                };
                let path = match module {
                    Some(relative) => {
                        target_package.library_root().resolve(relative).map_err(|source| {
                            BuildError::InvalidPath {
                                span,
                                trace: self.import_trace().into_boxed_slice(),
                                source,
                            }
                        })?
                    }
                    None => target_package.library_root().clone(),
                };
                Ok(ModuleKey::new(package, path))
            }
        }
    }

    fn import_trace(&self) -> Vec<ImportTraceFrame> {
        self.active.iter().filter_map(|frame| frame.incoming.clone()).collect()
    }

    fn cycle_trace(&self, target: ModuleId, incoming: Option<ImportTraceFrame>) -> Vec<ImportTraceFrame> {
        let start = self.active.iter().position(|frame| frame.module == target).unwrap_or_default();
        let mut edges: Vec<_> =
            self.active[start + 1..].iter().filter_map(|frame| frame.incoming.clone()).collect();
        if let Some(incoming) = incoming {
            edges.push(incoming);
        }
        edges
    }
}
