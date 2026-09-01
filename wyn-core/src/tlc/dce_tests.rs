use crate::{
    ast_const_fold, ast_type_holes, compile_thru_tlc, elaborate_modules, name_resolution, resolve_imports,
    resolve_opens, resolve_placeholders, resolve_resources, symbol_name_or_bug, Compiler, CompilerOptions,
};
use wyn_module_graph::{
    DependencyAlias, LocalSources, ModuleKey, ModulePath, PackageIdentity, PackagePlanBuilder,
    SourceFingerprint,
};

fn has_definition(program: &super::super::stage::Reachable, name: &str) -> bool {
    program.defs.iter().any(|definition| symbol_name_or_bug(&program.symbols, definition.name) == name)
}

#[test]
fn unused_extern_is_not_a_reachability_root() {
    let program = compile_thru_tlc(concat!(
        "#[linked(\"unused_link\")] extern unused(value: i32) i32\n",
        "entry main(value: i32) i32 = value\n",
    ))
    .expect("program should compile through reachability");

    assert!(!has_definition(&program, "unused"));
    assert!(has_definition(&program, "main"));
}

#[test]
fn called_extern_remains_reachable() {
    let program = compile_thru_tlc(concat!(
        "#[linked(\"used_link\")] extern linked(value: i32) i32\n",
        "entry main(value: i32) i32 = linked(value)\n",
    ))
    .expect("program should compile through reachability");

    assert!(has_definition(&program, "linked"));
    assert!(has_definition(&program, "main"));
}

#[test]
fn every_root_entry_remains_and_unreachable_definitions_do_not() {
    let program = compile_thru_tlc(concat!(
        "def used(value: i32) i32 = value + 1\n",
        "def unused(value: i32) i32 = value - 1\n",
        "entry first(value: i32) i32 = used(value)\n",
        "entry second(value: i32) i32 = value\n",
    ))
    .expect("program should compile through reachability");

    assert!(has_definition(&program, "first"));
    assert!(has_definition(&program, "second"));
    assert!(!has_definition(&program, "unused"));
}

#[test]
fn unused_dependency_contributes_no_reachable_definitions() {
    let fingerprint = SourceFingerprint::new("dce-package-test").expect("valid fingerprint");
    let root_path = ModulePath::new("main.wyn").expect("valid root path");
    let dependency_path = ModulePath::new("lib.wyn").expect("valid dependency path");
    let mut builder = PackagePlanBuilder::new();
    let root_package = builder
        .add_package(
            PackageIdentity::new("test/root", "v0.0.0", fingerprint.clone()).expect("valid root identity"),
            root_path.clone(),
        )
        .expect("root package should be unique");
    let dependency_package = builder
        .add_package(
            PackageIdentity::new("test/dependency", "v1.0.0", fingerprint)
                .expect("valid dependency identity"),
            dependency_path.clone(),
        )
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
                "entry main(value: i32) i32 = value\n",
            ),
        )
        .expect("root override should be unique");
    sources
        .add_override(dependency, "def unused(value: i32) i32 = value + 1")
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
    let program = crate::types::run::type_check(program).expect("program should type check");
    let program = ast_type_holes::reject_type_holes(program).expect("program should have no holes");
    let program = super::super::lower_from_ast(program).expect("AST should lower to TLC");
    let program = super::super::pin_entry_buffers(program).expect("entry buffers should pin");
    let program = super::super::validate_ownership(program).expect("ownership should validate");
    let program = crate::optimize_tlc_for_test(program).expect("TLC should optimize");

    assert!(!has_definition(&program, "Dependency.unused"));
    assert!(has_definition(&program, "main"));
}
