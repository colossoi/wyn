use super::type_check;
use crate::ast::Declaration;
use crate::{
    ast_const_fold, ast_type_holes, elaborate_modules, name_resolution, resolve_imports, resolve_opens,
    resolve_placeholders, resolve_resources, symbol_name_or_bug, tlc, Compiler, CompilerOptions,
};
use wyn_module_graph::{
    DependencyAlias, LocalSources, ModuleKey, ModulePath, PackageIdentity, PackagePlanBuilder,
    SourceFingerprint,
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
    let mut builder = PackagePlanBuilder::new();
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
                "entry compute_main(value: i32) i32 = Dependency.increment(value)\n",
            ),
        )
        .expect("root override should be unique");
    sources
        .add_override(dependency, "def increment(value: i32) i32 = value + 1")
        .expect("dependency override should be unique");

    let compiler = Compiler::new(CompilerOptions::default()).expect("compiler should initialize");
    let modules = compiler.load_modules(plan, &mut sources).expect("module graph should load");
    let program = resolve_imports::resolve_imports(modules).expect("imports should resolve");
    let program = elaborate_modules::elaborate_modules(program).expect("modules should elaborate");
    let program = name_resolution::resolve_names(program);
    let program = resolve_resources::resolve_resources(program).expect("resources should resolve");
    let program = ast_const_fold::fold_constants(program);
    let program = resolve_placeholders::resolve_type_placeholders(program);
    let program = resolve_opens::resolve_opens(program).expect("opens should resolve");
    let typed = type_check(program).expect("program should type check");

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
            support.namespace.as_deref() == Some("Dependency") && support.definition.name == "increment"
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
    assert_eq!(package_of("Dependency.increment"), Some(dependency_package));
    assert_eq!(package_of("compute_main"), Some(root_package));

    let pinned = tlc::pin_entry_buffers(lowered).expect("entry buffers should pin");
    let ownership = tlc::validate_ownership(pinned).expect("ownership should validate");
    let partial = tlc::partial_eval(ownership);
    let normalized = tlc::normalize_soacs(partial);
    let monomorphic = tlc::monomorphize(normalized).expect("TLC should monomorphize");
    let package_of = |name: &str| {
        monomorphic
            .defs
            .iter()
            .find(|definition| symbol_name_or_bug(&monomorphic.symbols, definition.name) == name)
            .and_then(|definition| definition.package)
    };
    assert_eq!(package_of("Dependency.increment"), Some(dependency_package));
    assert_eq!(package_of("compute_main"), Some(root_package));
}
