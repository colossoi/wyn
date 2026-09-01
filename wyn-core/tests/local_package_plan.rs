use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use wyn_core::{Compiler, CompilerOptions};
use wyn_module_graph::{
    DependencyAlias, LocalSources, ModuleKey, ModulePath, PackageIdentity, PackagePlanBuilder,
    SourceFingerprint,
};

struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn new() -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should follow the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "wyn_local_package_plan_{}_{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&path).expect("test directory should be created");
        Self { path }
    }

    fn package(&self, name: &str) -> PathBuf {
        let path = self.path.join(name);
        fs::create_dir_all(&path).expect("package directory should be created");
        path
    }

    fn write(root: &Path, relative: &str, source: &str) {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("source directory should be created");
        }
        fs::write(path, source).expect("source file should be written");
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.path) {
            eprintln!(
                "failed to remove test directory `{}`: {error}",
                self.path.display()
            );
        }
    }
}

#[test]
fn verified_local_package_roots_compile_as_one_program() {
    let directory = TestDirectory::new();
    let root_directory = directory.package("root");
    let dependency_directory = directory.package("dependency");
    TestDirectory::write(
        &root_directory,
        "main.wyn",
        concat!(
            "module Dependency = import \"pkg:dependency\"\n",
            "entry compute_main(value: i32) i32 = Dependency.identity(value)\n",
        ),
    );
    TestDirectory::write(
        &dependency_directory,
        "src/lib.wyn",
        "def identity<T>(value: T) T = value\n",
    );

    let fingerprint = SourceFingerprint::new("local-package-plan-test").expect("valid fingerprint");
    let root_path = ModulePath::new("main.wyn").expect("valid root path");
    let dependency_path = ModulePath::new("src/lib.wyn").expect("valid dependency path");
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
    builder.set_root(root).expect("root module should belong to the plan");
    let plan = builder.build().expect("package plan should be complete");

    let mut sources = LocalSources::new();
    sources
        .add_package_root(root_package, &root_directory)
        .expect("root source tree should be verified");
    sources
        .add_package_root(dependency_package, &dependency_directory)
        .expect("dependency source tree should be verified");

    Compiler::new(CompilerOptions::default())
        .expect("compiler should initialize")
        .load_modules(plan, &mut sources)
        .expect("local package graph should load")
        .type_check()
        .expect("local packages should type check as one program");
}
