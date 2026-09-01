use super::*;
use crate::{Compiler, CompilerOptions};
use wyn_module_graph::{
    LocalSources, ModuleKey, ModulePath, PackageIdentity, PackagePlan, PackagePlanBuilder,
    SourceFingerprint,
};

fn local_plan() -> (PackagePlan, ModuleKey, ModuleKey) {
    let fingerprint = SourceFingerprint::new("test-sources").expect("valid fingerprint");
    let identity =
        PackageIdentity::new("test/root", "v0.0.0", fingerprint).expect("valid package identity");
    let root_path = ModulePath::new("main.wyn").expect("valid root path");
    let dependency_path = ModulePath::new("dependency.wyn").expect("valid dependency path");
    let mut builder = PackagePlanBuilder::new();
    let package = builder.add_package(identity, root_path.clone()).expect("package should be unique");
    let root = ModuleKey::new(package, root_path);
    let dependency = ModuleKey::new(package, dependency_path);
    builder.set_root(root.clone()).expect("root package should exist");
    (
        builder.build().expect("plan should be complete"),
        root,
        dependency,
    )
}

#[test]
fn module_binding_import_becomes_loaded_module_body() {
    let (plan, root, dependency) = local_plan();
    let mut sources = LocalSources::new();
    sources
        .add_override(root, "module Dependency = import \"dependency\"")
        .expect("root override should be unique");
    sources
        .add_override(dependency, "def value: i32 = 42")
        .expect("dependency override should be unique");
    let compiler = Compiler::new(CompilerOptions::default()).expect("compiler should initialize");
    let modules = compiler.load_modules(plan, &mut sources).expect("module graph should load");

    let program = resolve_imports(modules).expect("imports should resolve");

    assert!(matches!(
        &program.declarations[..],
        [Declaration::Frontend(ImportsResolvedFrontend::Module(ModuleDecl::Module {
            body: ModuleExpression::Struct(declarations),
            ..
        }))] if matches!(&declarations[..], [NestedDeclaration::Decl(declaration)] if declaration.name == "value")
    ));
}
