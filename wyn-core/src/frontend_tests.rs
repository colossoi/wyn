use super::*;
use wyn_module_graph::{
    DependencyAlias, LocalSources, ModuleKey, ModulePath, PackageIdentity, PackagePlanBuilder,
    SourceFingerprint,
};

fn reported_imports(source: &str) -> Vec<(ImportSiteId, ImportTarget, TextRange)> {
    let mut node_ids = NodeCounter::new();
    let mut frontend = WynFrontend::new(&mut node_ids, CompilerOptions::default());
    let mut imports = Vec::new();
    frontend
        .parse(ModuleId::from(7), source, &mut |site, target, range| {
            imports.push((site, target, range));
        })
        .expect("source should parse");
    imports
}

#[test]
fn reports_every_import_form_in_source_order() {
    let source = concat!(
        "import \"first\"\n",
        "module Nested = { import \"second\" }\n",
        "module Bound = import \"third\"\n",
    );
    let imports = reported_imports(source);

    assert_eq!(imports.len(), 3);
    for (index, (site, _, range)) in imports.iter().enumerate() {
        assert_eq!(*site, ImportSiteId::from(index as u32));
        assert!(range.start() < range.end());
        assert!(source[range.start() as usize..range.end() as usize].starts_with("import"));
    }
}

#[test]
fn decodes_local_and_package_imports() {
    let imports = reported_imports(concat!(
        "import \"local/module\"\n",
        "module Root = import \"pkg:rng\"\n",
        "module Child = import \"pkg:rng/algorithm/xoshiro\"\n",
    ));

    assert!(matches!(
        &imports[0].1,
        ImportTarget::Local(path) if path.as_str() == "local/module.wyn"
    ));
    assert!(matches!(
        &imports[1].1,
        ImportTarget::Dependency { alias, module: None } if alias.as_str() == "rng"
    ));
    assert!(matches!(
        &imports[2].1,
        ImportTarget::Dependency { alias, module: Some(path) }
            if alias.as_str() == "rng" && path.as_str() == "algorithm/xoshiro.wyn"
    ));
}

#[test]
fn semantic_failure_retains_dependency_source_identity() {
    let fingerprint = SourceFingerprint::new("frontend-failure-test").expect("valid fingerprint");
    let root_path = ModulePath::new("main.wyn").expect("valid root path");
    let dependency_path = ModulePath::new("lib.wyn").expect("valid dependency path");
    let mut builder = PackagePlanBuilder::new();
    let root_package = builder
        .add_package(
            PackageIdentity::new("example/root", "v0.1.0", fingerprint.clone())
                .expect("valid root identity"),
            root_path.clone(),
        )
        .expect("root package should be unique");
    let dependency_package = builder
        .add_package(
            PackageIdentity::new("example/dependency", "v2.3.4", fingerprint)
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
    builder.set_root(root.clone()).expect("root should belong to plan");
    let plan = builder.build().expect("complete package plan");
    let mut sources = LocalSources::new();
    sources
        .add_override(root, "module Dependency = import \"pkg:dependency\"")
        .expect("root override should be unique");
    sources
        .add_override(dependency, "def broken: i32 = true")
        .expect("dependency override should be unique");

    let compiler = Compiler::new(CompilerOptions::default()).expect("compiler should initialize");
    let modules = compiler.load_modules(plan, &mut sources).expect("source graph should load");
    let failure = modules.type_check().expect_err("dependency should fail type checking");
    let span = failure.error().span().expect("type error should retain its source span");
    let module = span.module().expect("type error should belong to physical source");

    assert_eq!(
        failure.source_graph().package_of(module),
        Some(dependency_package)
    );
    assert!(failure
        .source_graph()
        .source(module)
        .is_some_and(|source| source.contains("def broken: i32 = true")));
    let message = failure.to_string();
    assert!(
        message.starts_with("example/dependency@v2.3.4:lib.wyn:1:"),
        "unexpected dependency diagnostic: {message}"
    );
    assert!(
        message.contains("Type error:"),
        "unexpected dependency diagnostic: {message}"
    );
    assert!(
        !message.contains("ModuleId("),
        "diagnostic leaked an internal module ID: {message}"
    );
}
