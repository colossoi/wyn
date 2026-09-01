use std::sync::{Arc, OnceLock};

use crate::ast::{NodeCounter, SourceImport};
use crate::error::{CompilerError, FrontendFailure, Result};
use crate::parser::{self, ParsedFile};
use crate::semantic_modules::{PreElaboratedPrelude, SemanticModules};
use crate::{
    ast_const_fold, elaborate_modules, err_parse_at, name_resolution, resolve_imports, resolve_opens,
    resolve_placeholders, resolve_resources, types, CompilerOptions,
};
use wyn_module_graph::{
    BuildFailure, DependencyAlias, ImportSiteId, ImportTarget, ModuleFrontend, ModuleGraph, ModuleId,
    PackagePlan, RelativeModulePath, SourceProvider, TextRange,
};

static COMPILER_PRELUDE_CACHE: OnceLock<(PreElaboratedPrelude, NodeCounter)> = OnceLock::new();

/// State owned by one compilation before its source-module graph is loaded.
pub struct Compiler {
    pub(crate) node_ids: NodeCounter,
    pub(crate) semantic_modules: SemanticModules,
}

/// Parsed source modules together with the compiler state that produced them.
///
/// This is an opaque compiler checkpoint. Later frontend phases consume it so
/// parsed syntax cannot be separated from its module graph or ID allocators.
pub struct ParsedModules {
    pub(crate) graph: ModuleGraph<ParsedFile>,
    pub(crate) node_ids: NodeCounter,
    pub(crate) semantic_modules: SemanticModules,
}

impl ParsedModules {
    /// Run the complete semantic frontend through type checking.
    ///
    /// Failures retain the source graph so callers can render package-aware
    /// locations without consulting the filesystem.
    pub fn type_check(self) -> std::result::Result<types::run::TypeChecked, FrontendFailure> {
        let program = resolve_imports::resolve_imports(self)?;
        let source_graph = Arc::clone(&program.source_graph);
        let result = (|| {
            let program = elaborate_modules::elaborate_modules(program)?;
            let program = name_resolution::resolve_names(program);
            let program = resolve_resources::resolve_resources(program)?;
            let program = ast_const_fold::fold_constants(program);
            let program = resolve_placeholders::resolve_type_placeholders(program);
            let program = resolve_opens::resolve_opens(program)?;
            types::run::type_check(program)
        })();
        result.map_err(|error| FrontendFailure::new(error, source_graph))
    }
}

impl Compiler {
    /// Create a compiler with a cached, pre-elaborated standard prelude.
    pub fn new(options: CompilerOptions) -> Result<Self> {
        let (prelude, node_ids) = compiler_prelude()?;
        Ok(Self {
            node_ids,
            semantic_modules: SemanticModules::from_prelude_with_options(prelude, options),
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
    let prelude = SemanticModules::create_prelude(&mut node_ids)?;
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
