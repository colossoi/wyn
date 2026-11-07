use super::*;
use crate::semantic_modules::ElaboratedItem;
use crate::{elaborate_modules, CompilerOptions, ParsedModules};
use wyn_module_graph::{
    LocalSources, ModuleKey, ModulePath, PackageGraph, PackageGraphBuilder, PackageIdentity, PackagePlan,
};

fn local_plan() -> (PackageGraph, ModuleKey, ModuleKey) {
    let identity = PackageIdentity::new("test/root", "v0.0.0").expect("valid package identity");
    let root_path = ModulePath::new("main.wyn").expect("valid root path");
    let dependency_path = ModulePath::new("dependency.wyn").expect("valid dependency path");
    let mut builder = PackageGraphBuilder::new();
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
fn imported_map_callback_with_opened_sqrt_and_runtime_index_normalizes() {
    assert_imported_callback_normalizes(
        "module Helper = import \"dependency\"\nentry repro() [2]f32 = Helper.values",
        "open f32\ndef values:[2]f32 = map(|i| f32.i32([i,i+1][i%2])+sqrt(2.0),iota(2))",
    );
}

#[test]
fn imported_nested_map_callback_with_opened_sqrt_normalizes() {
    assert_imported_callback_normalizes(
        "module Wrapper = import \"dependency\"\nentry repro() [2]f32 = Wrapper.values",
        "module Helper = { open f32\ndef values:[2]f32 = map(|i| f32.i32([i,i+1][i%2])+sqrt(2.0),iota(2)) }\ndef values:[2]f32 = Helper.values",
    );
}

#[test]
fn imported_builtin_can_be_reexported_and_called_by_qualified_name() {
    assert_imported_callback_normalizes(
        "module Helper = import \"dependency\"\nmodule Again = { open Helper }\nentry repro(x:f32) f32 = Again.sqrt(x)",
        "open f32",
    );
}

#[test]
fn imported_definition_shadows_opened_builtin() {
    assert_imported_callback_normalizes(
        "module Helper = import \"dependency\"\nentry repro(x:bool) bool = Helper.sqrt(x)",
        "open f32\ndef sqrt(x:bool) bool = !x",
    );
}

fn assert_imported_callback_normalizes(root_source: &str, helper_source: &str) {
    let (plan, root, dependency) = local_plan();
    let mut sources = LocalSources::new();
    sources.add_override(root, root_source).unwrap();
    sources.add_override(dependency, helper_source).unwrap();
    let checked = ParsedModules::load(PackagePlan::new(plan, sources), CompilerOptions::default())
        .unwrap()
        .type_check()
        .unwrap();
    let program = crate::ast_type_holes::reject_type_holes(checked).unwrap();
    let program = crate::tlc::lower_from_ast(program).unwrap();
    let program = crate::tlc::validate_ownership(program).unwrap();
    crate::optimize_tlc_for_test(program).expect("imported callback must normalize");
}

fn imported_module_definitions(module_source: &str, leaf_source: Option<&str>) -> Vec<String> {
    let identity = PackageIdentity::new("test/root", "v0.0.0").expect("valid package identity");
    let root_path = ModulePath::new("main.wyn").expect("valid root path");
    let module_path = ModulePath::new("module.wyn").expect("valid module path");
    let leaf_path = ModulePath::new("leaf.wyn").expect("valid leaf path");
    let mut builder = PackageGraphBuilder::new();
    let package = builder.add_package(identity, root_path.clone()).expect("package should be unique");
    let root = ModuleKey::new(package, root_path);
    let imported = ModuleKey::new(package, module_path);
    let leaf = ModuleKey::new(package, leaf_path);
    builder.set_root(root.clone()).expect("root package should exist");

    let mut sources = LocalSources::new();
    sources
        .add_override(root, "module Imported = import \"module\"")
        .expect("root override should be unique");
    sources.add_override(imported, module_source).expect("module override should be unique");
    if let Some(leaf_source) = leaf_source {
        sources.add_override(leaf, leaf_source).expect("leaf override should be unique");
    }

    let input = PackagePlan::new(builder.build().expect("plan should be complete"), sources);
    let modules = ParsedModules::load(input, CompilerOptions::default()).expect("module graph should load");
    let program = resolve_imports(modules).expect("imports should resolve");
    let program = elaborate_modules::elaborate_modules(program).expect("modules should elaborate");
    let imported = program
        .global_context
        .get_elaborated_module("Imported")
        .expect("imported module should be elaborated");

    imported
        .items
        .iter()
        .filter_map(|item| match item {
            ElaboratedItem::Decl(declaration) => Some(declaration.name.clone()),
            ElaboratedItem::Spec(_) | ElaboratedItem::TypeAlias(_, _) => None,
        })
        .collect()
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
    let modules = ParsedModules::load(PackagePlan::new(plan, sources), CompilerOptions::default())
        .expect("module graph should load");

    let program = resolve_imports(modules).expect("imports should resolve");

    assert_eq!(
        program.source_graph().source(program.source_graph().root()),
        Some("module Dependency = import \"dependency\"")
    );

    assert!(matches!(
        &program.declarations[..],
        [Declaration::Frontend(ImportsResolvedFrontend::Module(ModuleDecl::Module {
            body: ModuleExpression::Struct(declarations),
            ..
        }))] if matches!(&declarations[..], [NestedDeclaration::Decl(declaration)] if declaration.name == "value")
    ));
}

#[test]
fn imported_nested_semantic_module_is_preserved() {
    let (plan, root, dependency) = local_plan();
    let mut sources = LocalSources::new();
    sources
        .add_override(root, "module Dependency = import \"dependency\"")
        .expect("root override should be unique");
    sources
        .add_override(dependency, "module Nested = { def value: i32 = 42 }")
        .expect("dependency override should be unique");
    let modules = ParsedModules::load(PackagePlan::new(plan, sources), CompilerOptions::default())
        .expect("module graph should load");
    let program = resolve_imports(modules).expect("imports should resolve");

    let program = crate::elaborate_modules::elaborate_modules(program)
        .expect("nested semantic module should elaborate");

    let nested = program
        .global_context
        .get_elaborated_module("Dependency.Nested")
        .expect("nested module should have a qualified semantic namespace");
    assert!(nested.items.iter().any(
        |item| matches!(item, crate::semantic_modules::ElaboratedItem::Decl(declaration) if declaration.name == "value")
    ));
}

fn import_resolution_error(root_source: &str, dependency_source: &str) -> String {
    let (plan, root, dependency) = local_plan();
    let mut sources = LocalSources::new();
    sources.add_override(root, root_source).expect("root override should be unique");
    sources.add_override(dependency, dependency_source).expect("dependency override should be unique");
    let modules = ParsedModules::load(PackagePlan::new(plan, sources), CompilerOptions::default())
        .expect("module graph should load");

    resolve_imports(modules).expect_err("imported entry should be rejected").to_string()
}

#[test]
fn bare_import_rejects_entry_from_non_root_source() {
    let error = import_resolution_error("import \"dependency\"", "entry imported(value: i32) i32 = value");

    assert!(error.contains("entry `imported` is not declared directly in the root source module"));
}

#[test]
fn qualified_import_rejects_entry_from_non_root_source() {
    let error = import_resolution_error(
        "module Dependency = import \"dependency\"",
        "entry imported(value: i32) i32 = value",
    );

    assert!(error.contains("entry `imported` is not declared directly in the root source module"));
}

#[test]
fn nested_entry_is_rejected_in_root_source() {
    let error = import_resolution_error(
        "module Nested = { entry nested(value: i32) i32 = value }",
        "def unused(value: i32) i32 = value",
    );

    assert!(error.contains("entry `nested` is not declared directly in the root source module"));
}

#[test]
#[ignore = "`local` declarations are specified but deferred"]
fn local_declaration_is_not_exported_from_imported_module() {
    let definitions = imported_module_definitions(
        concat!(
            "local def hidden(value: i32) i32 = value + 1\n",
            "def visible(value: i32) i32 = hidden(value)\n",
        ),
        None,
    );

    assert!(!definitions.iter().any(|name| name == "hidden"));
    assert!(definitions.iter().any(|name| name == "visible"));
}

#[test]
#[ignore = "bare imports do not yet apply implicit `local open` visibility"]
fn bare_import_does_not_reexport_declarations() {
    let definitions = imported_module_definitions(
        concat!(
            "import \"leaf\"\n",
            "def visible(value: i32) i32 = hidden(value)\n",
        ),
        Some("def hidden(value: i32) i32 = value + 1"),
    );

    assert!(!definitions.iter().any(|name| name == "hidden"));
    assert!(definitions.iter().any(|name| name == "visible"));
}

#[test]
fn open_import_reexports_declarations() {
    let definitions = imported_module_definitions(
        "open import \"leaf\"",
        Some("def exposed(value: i32) i32 = value + 1"),
    );

    assert!(definitions.iter().any(|name| name == "exposed"));
}
