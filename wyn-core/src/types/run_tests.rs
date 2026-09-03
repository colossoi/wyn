use super::type_check;
use crate::ast::Declaration;
use crate::{
    ast_const_fold, ast_type_holes, elaborate_modules, name_resolution, resolve_imports, resolve_opens,
    resolve_placeholders, resolve_resources, symbol_name_or_bug, tlc, CompilerOptions, ParsedModules,
};
use wyn_module_graph::{
    DependencyAlias, LocalSources, ModuleKey, ModulePath, PackageGraphBuilder, PackageIdentity,
    PackagePlan, SourceFingerprint,
};

#[test]
fn source_package_identity_reaches_typed_and_tlc_definitions() {
    let fingerprint = SourceFingerprint::new("package-provenance-test").expect("valid fingerprint");
    let root_identity =
        PackageIdentity::new("test/root", "v0.0.0", fingerprint.clone()).expect("valid root identity");
    let dependency_identity =
        PackageIdentity::new("test/dependency", "v1.0.0", fingerprint).expect("valid dependency identity");
    let root_path = ModulePath::new("main.wyn").expect("valid root path");
    let dependency_path = ModulePath::new("lib.wyn").expect("valid dependency path");
    let mut builder = PackageGraphBuilder::new();
    let root_package =
        builder.add_package(root_identity, root_path.clone()).expect("root package should be unique");
    let dependency_package = builder
        .add_package(dependency_identity, dependency_path.clone())
        .expect("dependency package should be unique");
    builder
        .add_dependency(
            root_package,
            DependencyAlias::new("dependency").expect("valid dependency alias"),
            dependency_package,
        )
        .expect("dependency should be unique");
    let root = ModuleKey::new(root_package, root_path);
    let dependency = ModuleKey::new(dependency_package, dependency_path);
    builder.set_root(root.clone()).expect("root module should belong to the plan");
    let plan = builder.build().expect("package plan should be complete");

    let mut sources = LocalSources::new();
    sources
        .add_override(
            root,
            concat!(
                "module Dependency = import \"pkg:dependency\"\n",
                "type root_value = i32\n",
                "entry compute_main(value: root_value) root_value = Dependency.identity(value)\n",
            ),
        )
        .expect("root override should be unique");
    sources
        .add_override(dependency, "def identity<T>(value: T) T = value")
        .expect("dependency override should be unique");

    let modules = ParsedModules::load(PackagePlan::new(plan, sources), CompilerOptions::default())
        .expect("module graph should load");
    let program = resolve_imports::resolve_imports(modules).expect("imports should resolve");
    let program = elaborate_modules::elaborate_modules(program).expect("modules should elaborate");
    let program = name_resolution::resolve_names(program);
    let program = resolve_resources::resolve_resources(program).expect("resources should resolve");
    let program = ast_const_fold::fold_constants(program);
    let program = resolve_placeholders::resolve_type_placeholders(program);
    let program = resolve_opens::resolve_opens(program).expect("opens should resolve");
    let typed = type_check(program, CompilerOptions::default()).expect("program should type check");

    let entry = typed
        .declarations
        .iter()
        .find_map(|declaration| match declaration {
            Declaration::Entry(entry) if entry.name == "compute_main" => Some(entry),
            Declaration::Decl(_) | Declaration::Entry(_) | Declaration::Extern(_) => None,
            Declaration::Frontend(never) => match *never {},
        })
        .expect("root entry should remain in the typed AST");
    assert_eq!(entry.data.source.package, Some(root_package));

    let dependency_definition = typed
        .global_context
        .support_definitions
        .iter()
        .find(|support| {
            support.namespace.as_deref() == Some("Dependency") && support.definition.name == "identity"
        })
        .expect("dependency definition should be retained for TLC lowering");
    assert_eq!(
        dependency_definition.definition.data.source.package,
        Some(dependency_package)
    );

    let holes_resolved =
        ast_type_holes::reject_type_holes(typed).expect("program should have no type holes");
    let lowered = tlc::lower_from_ast(holes_resolved).expect("AST should lower to TLC");
    let package_of = |name: &str| {
        lowered
            .defs
            .iter()
            .find(|definition| symbol_name_or_bug(&lowered.symbols, definition.name) == name)
            .and_then(|definition| definition.package)
    };
    assert_eq!(package_of("Dependency.identity"), Some(dependency_package));
    assert_eq!(package_of("compute_main"), Some(root_package));

    let pinned = tlc::pin_entry_buffers(lowered).expect("entry buffers should pin");
    let ownership = tlc::validate_ownership(pinned).expect("ownership should validate");
    let partial = tlc::partial_eval(ownership);
    let normalized = tlc::normalize_soacs(partial);
    let monomorphic = tlc::monomorphize(normalized).expect("TLC should monomorphize");
    let dependency_specialization = monomorphic
        .defs
        .iter()
        .find(|definition| {
            symbol_name_or_bug(&monomorphic.symbols, definition.name).starts_with("Dependency.identity$")
        })
        .expect("dependency generic should be specialized for the root call");
    assert_eq!(dependency_specialization.package, Some(dependency_package));
    let root_entry = monomorphic
        .defs
        .iter()
        .find(|definition| symbol_name_or_bug(&monomorphic.symbols, definition.name) == "compute_main")
        .expect("root entry should remain after monomorphization");
    assert_eq!(root_entry.package, Some(root_package));
}

#[test]
fn dependency_local_aliases_have_independent_semantic_namespaces() {
    let fingerprint = SourceFingerprint::new("transitive-alias-test").expect("valid fingerprint");
    let root_path = ModulePath::new("main.wyn").expect("valid root path");
    let library_path = ModulePath::new("lib.wyn").expect("valid library path");
    let mut builder = PackageGraphBuilder::new();
    let root_package = builder
        .add_package(
            PackageIdentity::new("test/root", "v0.0.0", fingerprint.clone()).expect("valid root identity"),
            root_path.clone(),
        )
        .expect("root package should be unique");
    let package_a = builder
        .add_package(
            PackageIdentity::new("test/a", "v1.0.0", fingerprint.clone()).expect("valid package identity"),
            library_path.clone(),
        )
        .expect("package should be unique");
    let package_b = builder
        .add_package(
            PackageIdentity::new("test/b", "v1.0.0", fingerprint.clone()).expect("valid package identity"),
            library_path.clone(),
        )
        .expect("package should be unique");
    let utility_a = builder
        .add_package(
            PackageIdentity::new("test/utility-a", "v1.0.0", fingerprint.clone())
                .expect("valid package identity"),
            library_path.clone(),
        )
        .expect("package should be unique");
    let utility_b = builder
        .add_package(
            PackageIdentity::new("test/utility-b", "v1.0.0", fingerprint).expect("valid package identity"),
            library_path.clone(),
        )
        .expect("package should be unique");
    builder
        .add_dependency(
            root_package,
            DependencyAlias::new("a").expect("valid dependency alias"),
            package_a,
        )
        .expect("dependency should be unique");
    builder
        .add_dependency(
            root_package,
            DependencyAlias::new("b").expect("valid dependency alias"),
            package_b,
        )
        .expect("dependency should be unique");
    builder
        .add_dependency(
            package_a,
            DependencyAlias::new("util").expect("valid dependency alias"),
            utility_a,
        )
        .expect("dependency should be unique");
    builder
        .add_dependency(
            package_b,
            DependencyAlias::new("util").expect("valid dependency alias"),
            utility_b,
        )
        .expect("dependency should be unique");

    let root = ModuleKey::new(root_package, root_path);
    let source_a = ModuleKey::new(package_a, library_path.clone());
    let source_b = ModuleKey::new(package_b, library_path.clone());
    let source_utility_a = ModuleKey::new(utility_a, library_path.clone());
    let source_utility_b = ModuleKey::new(utility_b, library_path);
    builder.set_root(root.clone()).expect("root module should belong to the plan");
    let plan = builder.build().expect("package plan should be complete");

    let mut sources = LocalSources::new();
    sources
        .add_override(
            root,
            concat!(
                "module A = import \"pkg:a\"\n",
                "module B = import \"pkg:b\"\n",
                "entry compute_main(value: i32) i32 = A.compute(value) + B.compute(value)\n",
            ),
        )
        .expect("root override should be unique");
    sources
        .add_override(
            source_a,
            concat!(
                "module Util = import \"pkg:util\"\n",
                "def compute(value: Util.value) Util.value = Util.adjust(value)\n",
            ),
        )
        .expect("package override should be unique");
    sources
        .add_override(
            source_b,
            concat!(
                "module Util = import \"pkg:util\"\n",
                "def compute(value: Util.value) Util.value = Util.adjust(value)\n",
            ),
        )
        .expect("package override should be unique");
    sources
        .add_override(
            source_utility_a,
            concat!(
                "type value = i32\n",
                "def adjust(value: value) value = value + 1\n",
            ),
        )
        .expect("package override should be unique");
    sources
        .add_override(
            source_utility_b,
            concat!(
                "type value = i32\n",
                "def adjust(value: value) value = value * 2\n",
            ),
        )
        .expect("package override should be unique");

    let modules = ParsedModules::load(PackagePlan::new(plan, sources), CompilerOptions::default())
        .expect("module graph should load");
    let program = resolve_imports::resolve_imports(modules).expect("imports should resolve");
    let program = elaborate_modules::elaborate_modules(program).expect("modules should elaborate");
    let program = name_resolution::resolve_names(program);
    let program = resolve_resources::resolve_resources(program).expect("resources should resolve");
    let program = ast_const_fold::fold_constants(program);
    let program = resolve_placeholders::resolve_type_placeholders(program);
    let program = resolve_opens::resolve_opens(program).expect("opens should resolve");
    let typed = type_check(program, CompilerOptions::default()).expect("program should type check");

    let package_of = |namespace: &str, name: &str| {
        typed
            .global_context
            .support_definitions
            .iter()
            .find(|support| {
                support.namespace.as_deref() == Some(namespace) && support.definition.name == name
            })
            .and_then(|support| support.definition.data.source.package)
    };
    assert_eq!(package_of("A.Util", "adjust"), Some(utility_a));
    assert_eq!(package_of("B.Util", "adjust"), Some(utility_b));
    assert_eq!(package_of("A", "compute"), Some(package_a));
    assert_eq!(package_of("B", "compute"), Some(package_b));
}
