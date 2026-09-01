use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;

use crate::source::SourceFile;
use crate::{
    load_modules, BuildError, DependencyAlias, ImportSiteId, ImportTarget, ModuleFrontend, ModuleId,
    ModuleKey, ModulePath, PackageIdentity, PackagePlan, PackagePlanBuilder, PathError, RelativeModulePath,
    SourceFingerprint, SourceLocation, SourceProvider, Span, SpanError, TextRange,
};

#[derive(Clone, Debug, Error, PartialEq, Eq)]
enum TestProviderError {
    #[error("missing test source")]
    Missing,
}

#[derive(Default)]
struct MemoryProvider {
    sources: HashMap<ModuleKey, Arc<str>>,
    loads: HashMap<ModuleKey, usize>,
}

impl MemoryProvider {
    fn insert(&mut self, package: crate::PackageId, path: &str, source: &str) {
        self.sources.insert(
            ModuleKey::new(package, module_path(path)),
            Arc::<str>::from(source),
        );
    }

    fn load_count(&self, package: crate::PackageId, path: &str) -> usize {
        self.loads.get(&ModuleKey::new(package, module_path(path))).copied().unwrap_or_default()
    }
}

impl SourceProvider for MemoryProvider {
    type Error = TestProviderError;

    fn load(&mut self, module: &ModuleKey) -> Result<Arc<str>, Self::Error> {
        *self.loads.entry(module.clone()).or_default() += 1;
        self.sources.get(module).cloned().ok_or(TestProviderError::Missing)
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
enum TestFrontendError {
    #[error("requested parse failure")]
    Requested,
    #[error("invalid test import: {0}")]
    InvalidImport(String),
}

#[derive(Default)]
struct TestFrontend;

impl ModuleFrontend for TestFrontend {
    type Parsed = String;
    type Error = TestFrontendError;

    fn parse(
        &mut self,
        _module: ModuleId,
        source: &str,
        report_import: &mut dyn FnMut(ImportSiteId, ImportTarget, TextRange),
    ) -> Result<Self::Parsed, Self::Error> {
        if source.trim() == "parse-error" {
            return Err(TestFrontendError::Requested);
        }

        let mut next_site = 0u32;
        let mut offset = 0usize;
        for line in source.split_inclusive('\n') {
            let text = line.trim();
            if !text.is_empty() {
                let start_in_line = line.find(text).unwrap_or_default();
                let start = u32::try_from(offset + start_in_line)
                    .map_err(|_| TestFrontendError::InvalidImport(text.to_owned()))?;
                let end = start
                    .checked_add(
                        u32::try_from(text.len())
                            .map_err(|_| TestFrontendError::InvalidImport(text.to_owned()))?,
                    )
                    .ok_or_else(|| TestFrontendError::InvalidImport(text.to_owned()))?;
                let target = parse_test_import(text)?;
                report_import(ImportSiteId::from(next_site), target, text_range(start, end));
                next_site += 1;
            }
            offset += line.len();
        }

        Ok(source.to_owned())
    }
}

fn parse_test_import(text: &str) -> Result<ImportTarget, TestFrontendError> {
    if let Some(path) = text.strip_prefix("local:") {
        let path = RelativeModulePath::from_import(path)
            .map_err(|_| TestFrontendError::InvalidImport(text.to_owned()))?;
        return Ok(ImportTarget::Local(path));
    }

    let Some(dependency) = text.strip_prefix("dep:") else {
        return Err(TestFrontendError::InvalidImport(text.to_owned()));
    };
    let (alias, module) = match dependency.split_once(':') {
        Some((alias, module)) => (
            alias,
            Some(
                RelativeModulePath::from_import(module)
                    .map_err(|_| TestFrontendError::InvalidImport(text.to_owned()))?,
            ),
        ),
        None => (dependency, None),
    };
    let alias =
        DependencyAlias::new(alias).map_err(|_| TestFrontendError::InvalidImport(text.to_owned()))?;
    Ok(ImportTarget::Dependency { alias, module })
}

fn text_range(start: u32, end: u32) -> TextRange {
    TextRange::new(start, end).unwrap_or_else(|error| panic!("invalid test range: {error}"))
}

fn module_path(path: &str) -> ModulePath {
    ModulePath::new(path).unwrap_or_else(|error| panic!("invalid test module path: {error}"))
}

fn alias(alias: &str) -> DependencyAlias {
    DependencyAlias::new(alias).unwrap_or_else(|error| panic!("invalid test alias: {error}"))
}

fn identity(name: &str) -> PackageIdentity {
    let fingerprint = SourceFingerprint::new(format!("test:{name}"))
        .unwrap_or_else(|error| panic!("invalid fingerprint: {error}"));
    PackageIdentity::new(name, "v1.0.0", fingerprint)
        .unwrap_or_else(|error| panic!("invalid package identity: {error}"))
}

fn one_package_plan() -> (PackagePlan, crate::PackageId) {
    let mut builder = PackagePlanBuilder::new();
    let package = builder
        .add_package(identity("test/root"), module_path("src/lib.wyn"))
        .unwrap_or_else(|error| panic!("add package: {error}"));
    builder
        .set_root(ModuleKey::new(package, module_path("src/main.wyn")))
        .unwrap_or_else(|error| panic!("set root: {error}"));
    let plan = builder.build().unwrap_or_else(|error| panic!("build plan: {error}"));
    (plan, package)
}

#[test]
fn module_paths_normalize_and_confine_relative_imports() {
    assert_eq!(module_path("src\\nested/./../lib.wyn").as_str(), "src/lib.wyn");
    assert_eq!(
        ModulePath::from_import("src/math")
            .unwrap_or_else(|error| panic!("module import: {error}"))
            .as_str(),
        "src/math.wyn"
    );
    assert_eq!(ModulePath::from_import(""), Err(PathError::Empty));
    assert_eq!(RelativeModulePath::from_import(".."), Err(PathError::Empty));

    let importer = module_path("src/nested/main.wyn");
    let sibling = RelativeModulePath::from_import("../shared")
        .unwrap_or_else(|error| panic!("relative import: {error}"));
    assert_eq!(
        importer.resolve(&sibling).unwrap_or_else(|error| panic!("resolve import: {error}")).as_str(),
        "src/shared.wyn"
    );

    let escape = RelativeModulePath::from_import("../../../outside")
        .unwrap_or_else(|error| panic!("relative import: {error}"));
    assert_eq!(importer.resolve(&escape), Err(PathError::EscapesPackageRoot));
    assert_eq!(ModulePath::new("C:\\source\\lib.wyn"), Err(PathError::Absolute));
}

#[test]
fn source_file_maps_utf8_spans_to_snippets_and_locations() {
    let module = ModuleId::from(0);
    let other = ModuleId::from(1);
    let source =
        SourceFile::new(module, Arc::from("αβ\nz")).unwrap_or_else(|error| panic!("source file: {error}"));
    let beta = Span::new(module, text_range(2, 4));
    let z = Span::new(module, text_range(5, 6));

    assert_eq!(source.snippet(beta), Ok("β"));
    assert_eq!(source.location(beta), Ok(SourceLocation { line: 1, column: 2 }));
    assert_eq!(source.location(z), Ok(SourceLocation { line: 2, column: 1 }));
    assert_eq!(
        source.snippet(Span::new(other, text_range(0, 0))),
        Err(SpanError::WrongModule {
            expected: module,
            actual: other,
        })
    );
    assert_eq!(
        source.snippet(Span::new(module, text_range(1, 2))),
        Err(SpanError::NotCharBoundary { offset: 1 })
    );
}

#[test]
fn package_plan_has_deterministic_ids_and_package_local_aliases() {
    let mut builder = PackagePlanBuilder::new();
    let root = builder
        .add_package(identity("test/root"), module_path("src/lib.wyn"))
        .unwrap_or_else(|error| panic!("root package: {error}"));
    let first = builder
        .add_package(identity("test/first"), module_path("src/lib.wyn"))
        .unwrap_or_else(|error| panic!("first package: {error}"));
    assert_eq!(root, crate::PackageId::from(0));
    assert_eq!(first, crate::PackageId::from(1));

    builder
        .add_dependency(root, alias("util"), first)
        .unwrap_or_else(|error| panic!("dependency: {error}"));
    assert!(builder.add_dependency(root, alias("util"), first).is_err());
    assert!(builder.add_package(identity("test/root"), module_path("other/lib.wyn")).is_err());
}

#[test]
fn diamond_import_loads_shared_source_once_in_dependency_order() {
    let (plan, package) = one_package_plan();
    let mut provider = MemoryProvider::default();
    provider.insert(package, "src/main.wyn", "local:left\nlocal:right\n");
    provider.insert(package, "src/left.wyn", "local:shared\n");
    provider.insert(package, "src/right.wyn", "local:shared\n");
    provider.insert(package, "src/shared.wyn", "");

    let graph = load_modules(plan, &mut provider, &mut TestFrontend)
        .unwrap_or_else(|error| panic!("load graph: {error}"));
    assert_eq!(graph.modules().count(), 4);
    assert_eq!(provider.load_count(package, "src/shared.wyn"), 1);
    assert_eq!(
        graph
            .modules_in_dependency_order()
            .map(|module| graph.module(module).map(|loaded| loaded.key().path().as_str()))
            .collect::<Vec<_>>(),
        vec![
            Some("src/shared.wyn"),
            Some("src/left.wyn"),
            Some("src/right.wyn"),
            Some("src/main.wyn"),
        ]
    );
}

#[test]
fn syntax_erasure_preserves_source_and_resolved_imports() {
    let (plan, package) = one_package_plan();
    let mut provider = MemoryProvider::default();
    provider.insert(package, "src/main.wyn", "local:dependency\n");
    provider.insert(package, "src/dependency.wyn", "");

    let graph = load_modules(plan, &mut provider, &mut TestFrontend)
        .unwrap_or_else(|error| panic!("load graph: {error}"));
    let root = graph.root();
    let target = graph
        .import_target(root, ImportSiteId::from(0))
        .unwrap_or_else(|| panic!("missing resolved import"));
    let graph = graph.erase_syntax();

    assert_eq!(graph.source(root), Some("local:dependency\n"));
    assert_eq!(graph.source(target), Some(""));
    assert_eq!(graph.import_target(root, ImportSiteId::from(0)), Some(target));
    assert_eq!(graph.package_of(target), Some(package));
}

#[test]
fn same_relative_path_in_distinct_packages_has_distinct_module_identity() {
    let mut builder = PackagePlanBuilder::new();
    let root = builder
        .add_package(identity("test/root"), module_path("src/lib.wyn"))
        .unwrap_or_else(|error| panic!("root package: {error}"));
    let one = builder
        .add_package(identity("test/one"), module_path("src/lib.wyn"))
        .unwrap_or_else(|error| panic!("one package: {error}"));
    let two = builder
        .add_package(identity("test/two"), module_path("src/lib.wyn"))
        .unwrap_or_else(|error| panic!("two package: {error}"));
    builder
        .add_dependency(root, alias("one"), one)
        .unwrap_or_else(|error| panic!("one dependency: {error}"));
    builder
        .add_dependency(root, alias("two"), two)
        .unwrap_or_else(|error| panic!("two dependency: {error}"));
    builder
        .set_root(ModuleKey::new(root, module_path("src/main.wyn")))
        .unwrap_or_else(|error| panic!("set root: {error}"));
    let plan = builder.build().unwrap_or_else(|error| panic!("build plan: {error}"));

    let mut provider = MemoryProvider::default();
    provider.insert(root, "src/main.wyn", "dep:one\ndep:two\n");
    provider.insert(one, "src/lib.wyn", "");
    provider.insert(two, "src/lib.wyn", "");
    let graph = load_modules(plan, &mut provider, &mut TestFrontend)
        .unwrap_or_else(|error| panic!("load graph: {error}"));

    let imported: Vec<_> = graph
        .module(graph.root())
        .into_iter()
        .flat_map(|module| module.imports())
        .map(|edge| (edge.target(), graph.package_of(edge.target())))
        .collect();
    assert_eq!(imported.len(), 2);
    assert_ne!(imported[0].0, imported[1].0);
    assert_eq!(imported[0].1, Some(one));
    assert_eq!(imported[1].1, Some(two));
}

#[test]
fn dependency_aliases_are_resolved_in_the_importing_package() {
    let mut builder = PackagePlanBuilder::new();
    let root = builder
        .add_package(identity("test/root"), module_path("src/lib.wyn"))
        .unwrap_or_else(|error| panic!("root package: {error}"));
    let middle = builder
        .add_package(identity("test/middle"), module_path("src/lib.wyn"))
        .unwrap_or_else(|error| panic!("middle package: {error}"));
    let leaf = builder
        .add_package(identity("test/leaf"), module_path("src/lib.wyn"))
        .unwrap_or_else(|error| panic!("leaf package: {error}"));
    builder
        .add_dependency(root, alias("util"), middle)
        .unwrap_or_else(|error| panic!("root dependency: {error}"));
    builder
        .add_dependency(middle, alias("util"), leaf)
        .unwrap_or_else(|error| panic!("middle dependency: {error}"));
    builder
        .set_root(ModuleKey::new(root, module_path("src/main.wyn")))
        .unwrap_or_else(|error| panic!("set root: {error}"));
    let plan = builder.build().unwrap_or_else(|error| panic!("build plan: {error}"));

    let mut provider = MemoryProvider::default();
    provider.insert(root, "src/main.wyn", "dep:util\n");
    provider.insert(middle, "src/lib.wyn", "dep:util\n");
    provider.insert(leaf, "src/lib.wyn", "");
    let graph = load_modules(plan, &mut provider, &mut TestFrontend)
        .unwrap_or_else(|error| panic!("load graph: {error}"));

    let root_target = graph
        .import_target(graph.root(), ImportSiteId::from(0))
        .unwrap_or_else(|| panic!("missing root import"));
    let leaf_target = graph
        .import_target(root_target, ImportSiteId::from(0))
        .unwrap_or_else(|| panic!("missing middle import"));
    assert_eq!(graph.package_of(root_target), Some(middle));
    assert_eq!(graph.package_of(leaf_target), Some(leaf));
}

#[test]
fn import_cycle_reports_only_the_ordered_cycle_edges() {
    let (plan, package) = one_package_plan();
    let mut provider = MemoryProvider::default();
    provider.insert(package, "src/main.wyn", "local:a\n");
    provider.insert(package, "src/a.wyn", "local:b\n");
    provider.insert(package, "src/b.wyn", "local:a\n");

    let failure = load_modules(plan, &mut provider, &mut TestFrontend).unwrap_err();
    let BuildError::Cycle { edges } = failure.error() else {
        panic!("expected cycle error");
    };
    assert_eq!(edges.len(), 2);
    assert_eq!(edges[0].requested.path().as_str(), "src/b.wyn");
    assert_eq!(edges[1].requested.path().as_str(), "src/a.wyn");
    assert_eq!(failure.snippet(edges[0].span), Ok("local:b"));
    assert_eq!(failure.snippet(edges[1].span), Ok("local:a"));
}

#[test]
fn load_failure_retains_the_complete_import_chain() {
    let (plan, package) = one_package_plan();
    let mut provider = MemoryProvider::default();
    provider.insert(package, "src/main.wyn", "local:a\n");
    provider.insert(package, "src/a.wyn", "local:missing\n");

    let failure = load_modules(plan, &mut provider, &mut TestFrontend).unwrap_err();
    let BuildError::Load {
        module,
        requested_at,
        trace,
        source: TestProviderError::Missing,
    } = failure.error()
    else {
        panic!("expected load error");
    };
    assert_eq!(module.path().as_str(), "src/missing.wyn");
    assert!(requested_at.is_some());
    assert_eq!(trace.len(), 2);
    assert_eq!(trace[0].requested.path().as_str(), "src/a.wyn");
    assert_eq!(trace[1].requested.path().as_str(), "src/missing.wyn");
    assert_eq!(failure.snippet(requested_at.unwrap()), Ok("local:missing"));
}

#[test]
fn undeclared_dependency_reports_its_alias_and_source_span() {
    let (plan, package) = one_package_plan();
    let mut provider = MemoryProvider::default();
    provider.insert(package, "src/main.wyn", "dep:missing\n");

    let failure = load_modules(plan, &mut provider, &mut TestFrontend).unwrap_err();
    let BuildError::UnknownDependency {
        from,
        site,
        alias,
        span,
    } = failure.error()
    else {
        panic!("expected unknown dependency error");
    };
    assert_eq!(*from, ModuleId::from(0));
    assert_eq!(*site, ImportSiteId::from(0));
    assert_eq!(alias.as_str(), "missing");
    assert_eq!(span.range(), text_range(0, 11));
    assert_eq!(failure.snippet(*span), Ok("dep:missing"));
}

#[test]
fn parse_failure_retains_the_source_that_failed_to_parse() {
    let (plan, package) = one_package_plan();
    let mut provider = MemoryProvider::default();
    provider.insert(package, "src/main.wyn", "parse-error");

    let failure = load_modules(plan, &mut provider, &mut TestFrontend).unwrap_err();
    let BuildError::Parse {
        module,
        trace,
        source: TestFrontendError::Requested,
    } = failure.error()
    else {
        panic!("expected parse error");
    };
    assert!(trace.is_empty());
    assert_eq!(
        failure.module_key(*module).map(|key| key.path().as_str()),
        Some("src/main.wyn")
    );
    assert_eq!(
        failure.plan().package(package).map(|package| package.identity().canonical_name()),
        Some("test/root")
    );
    assert_eq!(failure.source_text(*module), Some("parse-error"));
    let span = Span::new(*module, text_range(0, 5));
    assert_eq!(failure.snippet(span), Ok("parse"));
    assert_eq!(failure.location(span), Ok(SourceLocation { line: 1, column: 1 }));
}
