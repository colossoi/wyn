use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use wyn_base::IdArena;

use crate::error::{BuildError, BuildFailure};
use crate::source::SourceMap;
use crate::{
    DependencyAlias, ImportSiteId, ModuleId, ModuleKey, PackageId, PackagePlan, RelativeModulePath,
    SourceLocation, Span, SpanError, TextRange,
};

/// Supplies verified source text for a module in a closed package plan.
pub trait SourceProvider {
    type Error;

    fn load(&mut self, module: &ModuleKey) -> Result<Arc<str>, Self::Error>;
}

/// Parses one source file and extracts its physical import requests.
pub trait ModuleFrontend {
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
#[derive(Clone, Debug, PartialEq, Eq)]
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

/// Immutable physical module graph produced from a closed package plan.
#[derive(Clone, Debug)]
pub struct ModuleGraph<T> {
    plan: PackagePlan,
    root: ModuleId,
    modules: IdArena<ModuleId, LoadedModule<T>>,
    dependency_order: Vec<ModuleId>,
    sources: SourceMap,
}

/// Physical source and import topology with frontend syntax discarded.
pub type SourceGraph = ModuleGraph<()>;

impl<T> ModuleGraph<T> {
    pub const fn plan(&self) -> &PackagePlan {
        &self.plan
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

    pub fn snippet(&self, span: Span) -> Result<&str, SpanError> {
        self.sources.snippet(span)
    }

    /// Discard frontend syntax while retaining package, module, import, and
    /// source provenance.
    pub fn erase_syntax(self) -> SourceGraph {
        let Self {
            plan,
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
            plan,
            root,
            modules: source_modules,
            dependency_order,
            sources,
        }
    }
}

/// One edge in an import provenance or cycle trace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportTraceFrame {
    pub from: ModuleId,
    pub span: Span,
    pub requested: ModuleKey,
}

/// Result type returned by physical module-graph construction.
pub type BuildResult<Parsed, FrontendError, ProviderError> =
    Result<ModuleGraph<Parsed>, BuildFailure<FrontendError, ProviderError>>;

/// Load, parse, and resolve every source module reachable from the plan root.
pub fn load_modules<F, S>(
    plan: PackagePlan,
    sources: &mut S,
    frontend: &mut F,
) -> BuildResult<F::Parsed, F::Error, S::Error>
where
    F: ModuleFrontend,
    S: SourceProvider,
{
    GraphBuilder::new(plan, sources, frontend).build()
}

#[derive(Clone, Debug)]
struct ActiveFrame {
    module: ModuleId,
    incoming: Option<ImportTraceFrame>,
}

struct GraphBuilder<'a, F: ModuleFrontend, S: SourceProvider> {
    plan: PackagePlan,
    sources: &'a mut S,
    frontend: &'a mut F,
    modules: IdArena<ModuleId, Option<LoadedModule<F::Parsed>>>,
    by_key: HashMap<ModuleKey, ModuleId>,
    active: Vec<ActiveFrame>,
    dependency_order: Vec<ModuleId>,
    source_map: SourceMap,
}

impl<'a, F: ModuleFrontend, S: SourceProvider> GraphBuilder<'a, F, S> {
    fn new(plan: PackagePlan, sources: &'a mut S, frontend: &'a mut F) -> Self {
        Self {
            plan,
            sources,
            frontend,
            modules: IdArena::new(),
            by_key: HashMap::new(),
            active: Vec::new(),
            dependency_order: Vec::new(),
            source_map: SourceMap::default(),
        }
    }

    fn build(mut self) -> BuildResult<F::Parsed, F::Error, S::Error> {
        let root_key = self.plan.root().clone();
        let root = match self.load(root_key, None) {
            Ok(root) => root,
            Err(error) => {
                return Err(BuildFailure::new(error, self.plan, self.by_key, self.source_map));
            }
        };
        let mut modules = IdArena::new();
        for (id, module) in self.modules {
            let module = match module {
                Some(module) => module,
                None => panic!("module builder completed with an unfinished module"),
            };
            let copied_id = modules.alloc(module);
            debug_assert_eq!(copied_id, id);
        }
        Ok(ModuleGraph {
            plan: self.plan,
            root,
            modules,
            dependency_order: self.dependency_order,
            sources: self.source_map,
        })
    }

    fn load(
        &mut self,
        key: ModuleKey,
        incoming: Option<ImportTraceFrame>,
    ) -> Result<ModuleId, BuildError<F::Error, S::Error>> {
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

        let requested_at =
            self.active.last().and_then(|frame| frame.incoming.as_ref()).map(|edge| edge.span);
        let trace = self.import_trace();
        let text = self.sources.load(&key).map_err(|source| BuildError::Load {
            module: key.clone(),
            requested_at,
            trace: trace.clone().into_boxed_slice(),
            source,
        })?;
        self.source_map.insert(module, text).map_err(|source| BuildError::SourceText {
            module: key.clone(),
            requested_at,
            trace: trace.clone().into_boxed_slice(),
            source,
        })?;
        let mut requests = Vec::new();
        let source_text = self
            .source_map
            .source(module)
            .unwrap_or_else(|| panic!("source map lost a newly inserted module"));
        let syntax = self
            .frontend
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
                    from: module,
                    site: request.site,
                });
            }

            let span = Span::new(module, request.range);
            self.source_map.snippet(span).map_err(|source| BuildError::InvalidImportSpan {
                from: module,
                site: request.site,
                source,
            })?;
            let target_key = self.resolve_target(module, &key, &request, span)?;
            let frame = ImportTraceFrame {
                from: module,
                span,
                requested: target_key.clone(),
            };
            let target = self.load(target_key, Some(frame))?;
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

    fn resolve_target(
        &self,
        from: ModuleId,
        from_key: &ModuleKey,
        request: &ImportRequest,
        span: Span,
    ) -> Result<ModuleKey, BuildError<F::Error, S::Error>> {
        match &request.target {
            ImportTarget::Local(relative) => {
                let path = from_key.path().resolve(relative).map_err(|source| BuildError::InvalidPath {
                    from,
                    site: request.site,
                    span,
                    source,
                })?;
                Ok(ModuleKey::new(from_key.package(), path))
            }
            ImportTarget::Dependency { alias, module } => {
                let package = self
                    .plan
                    .package(from_key.package())
                    .and_then(|package| package.dependency(alias))
                    .ok_or_else(|| BuildError::UnknownDependency {
                        from,
                        site: request.site,
                        alias: alias.clone(),
                        span,
                    })?;
                let target_package =
                    self.plan.package(package).ok_or_else(|| BuildError::UnknownDependency {
                        from,
                        site: request.site,
                        alias: alias.clone(),
                        span,
                    })?;
                let path = match module {
                    Some(relative) => {
                        target_package.library_root().resolve(relative).map_err(|source| {
                            BuildError::InvalidPath {
                                from,
                                site: request.site,
                                span,
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
