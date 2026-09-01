use std::sync::OnceLock;

use crate::ast::{NodeCounter, SourceImport};
use crate::error::{CompilerError, Result};
use crate::module_manager::{ModuleManager, PreElaboratedPrelude};
use crate::parser::{self, ParsedFile};
use crate::{err_parse_at, CompilerOptions};
use wyn_module_graph::{
    BuildFailure, DependencyAlias, ImportSiteId, ImportTarget, ModuleFrontend, ModuleGraph, ModuleId,
    PackagePlan, RelativeModulePath, SourceProvider, TextRange,
};

static COMPILER_PRELUDE_CACHE: OnceLock<(PreElaboratedPrelude, NodeCounter)> = OnceLock::new();

/// State owned by one compilation before its source-module graph is loaded.
pub struct Compiler {
    pub(crate) node_ids: NodeCounter,
    pub(crate) semantic_modules: ModuleManager,
}

/// Parsed source modules together with the compiler state that produced them.
///
/// This is an opaque compiler checkpoint. Later frontend phases consume it so
/// parsed syntax cannot be separated from its module graph or ID allocators.
pub struct ParsedModules {
    pub(crate) graph: ModuleGraph<ParsedFile>,
    pub(crate) node_ids: NodeCounter,
    pub(crate) semantic_modules: ModuleManager,
}

impl Compiler {
    /// Create a compiler with a cached, pre-elaborated standard prelude.
    pub fn new(options: CompilerOptions) -> Result<Self> {
        let (prelude, node_ids) = compiler_prelude()?;
        Ok(Self {
            node_ids,
            semantic_modules: ModuleManager::from_prelude_with_options(prelude, options),
        })
    }

    /// Load and parse every source module reachable from a closed package plan.
    pub fn load_modules<S>(
        mut self,
        plan: PackagePlan,
        sources: &mut S,
    ) -> std::result::Result<ParsedModules, BuildFailure<CompilerError, S::Error>>
    where
        S: SourceProvider,
    {
        let options = self.semantic_modules.options();
        let mut frontend = WynFrontend::new(&mut self.node_ids, options);
        let graph = wyn_module_graph::load_modules(plan, sources, &mut frontend)?;
        Ok(ParsedModules {
            graph,
            node_ids: self.node_ids,
            semantic_modules: self.semantic_modules,
        })
    }
}

fn compiler_prelude() -> Result<(PreElaboratedPrelude, NodeCounter)> {
    if let Some((prelude, node_ids)) = COMPILER_PRELUDE_CACHE.get() {
        return Ok((prelude.clone(), node_ids.clone()));
    }

    let mut node_ids = NodeCounter::new();
    let prelude = ModuleManager::create_prelude(&mut node_ids)?;
    let (prelude, node_ids) = COMPILER_PRELUDE_CACHE.get_or_init(|| (prelude, node_ids));
    Ok((prelude.clone(), node_ids.clone()))
}

/// The Wyn-specific adapter at the syntax-independent module graph boundary.
pub(crate) struct WynFrontend<'a> {
    node_ids: &'a mut NodeCounter,
    options: CompilerOptions,
}

impl<'a> WynFrontend<'a> {
    pub(crate) const fn new(node_ids: &'a mut NodeCounter, options: CompilerOptions) -> Self {
        Self { node_ids, options }
    }
}

impl ModuleFrontend for WynFrontend<'_> {
    type Parsed = ParsedFile;
    type Error = CompilerError;

    fn parse(
        &mut self,
        module: ModuleId,
        source: &str,
        report_import: &mut dyn FnMut(ImportSiteId, ImportTarget, TextRange),
    ) -> Result<ParsedFile> {
        let parsed = parser::parse_file(module, source, self.node_ids, self.options.graphics)?;
        for import in &parsed.imports {
            report_import(import.site, decode_import(import)?, import.span.range());
        }
        Ok(parsed)
    }
}

fn decode_import(import: &SourceImport) -> Result<ImportTarget> {
    if let Some(package_path) = import.path.strip_prefix("pkg:") {
        let (alias, module) = match package_path.split_once('/') {
            Some((alias, module)) => (alias, Some(module)),
            None => (package_path, None),
        };
        let alias = DependencyAlias::new(alias)
            .map_err(|error| err_parse_at!(import.span, "invalid package import: {error}"))?;
        let module = module
            .map(RelativeModulePath::from_import)
            .transpose()
            .map_err(|error| err_parse_at!(import.span, "invalid package import path: {error}"))?;
        Ok(ImportTarget::Dependency { alias, module })
    } else {
        let path = RelativeModulePath::from_import(&import.path)
            .map_err(|error| err_parse_at!(import.span, "invalid import path: {error}"))?;
        Ok(ImportTarget::Local(path))
    }
}

#[cfg(test)]
#[path = "frontend_tests.rs"]
mod frontend_tests;
