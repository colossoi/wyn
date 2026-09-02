use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use wyn_module_graph::SourceProvider;

use super::{load_local_build, load_local_input, LocalBuildError};

struct TestTree {
    root: PathBuf,
}

impl TestTree {
    fn new() -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should follow the Unix epoch")
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("wyn_package_manager_{}_{}", std::process::id(), unique));
        fs::create_dir_all(&root).expect("test tree should be created");
        Self { root }
    }

    fn package(&self, relative: &str, name: &str, version: &str, dependencies: &str) -> PathBuf {
        let root = self.root.join(relative);
        fs::create_dir_all(root.join("src")).expect("package source directory should be created");
        fs::write(
            root.join("wyn.toml"),
            format!(
                concat!(
                    "manifest-version = 1\n",
                    "[package]\n",
                    "name = \"{}\"\n",
                    "version = \"{}\"\n",
                    "wyn = \"v0.1.0\"\n",
                    "library = \"src/lib.wyn\"\n",
                    "{}",
                ),
                name, version, dependencies
            ),
        )
        .expect("manifest should be written");
        fs::write(
            root.join("src/lib.wyn"),
            format!("def package_name: i32 = {}\n", name.len()),
        )
        .expect("source should be written");
        root
    }
}

impl Drop for TestTree {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root) {
            eprintln!("failed to remove test tree `{}`: {error}", self.root.display());
        }
    }
}

fn dependency(alias: &str, package: &str, version: &str, path: &str) -> String {
    format!(
        concat!(
            "[dependencies]\n",
            "{} = {{ package = \"{}\", version = \"{}\", path = \"{}\" }}\n",
        ),
        alias, package, version, path
    )
}

fn package_id(plan: &wyn_module_graph::PackagePlan, name: &str) -> wyn_module_graph::PackageId {
    plan.packages()
        .find_map(|(id, package)| (package.identity().canonical_name() == name).then_some(id))
        .expect("package should belong to the plan")
}

#[test]
fn local_manifests_produce_a_closed_plan_and_confined_sources() {
    let tree = TestTree::new();
    let dependency_root = tree.package("dependency", "test/dependency", "v1.3.0", "");
    let root = tree.package(
        "root",
        "test/root",
        "v1.0.0",
        &dependency("dependency", "test/dependency", "v1.2.0", "../dependency"),
    );

    let build = load_local_build(root).expect("local graph should load");
    let (plan, mut sources) = build.into_parts();
    assert_eq!(plan.packages().count(), 2);
    let dependency = package_id(&plan, "test/dependency");
    let root_package = plan.package(plan.root().package()).expect("root package should exist");
    let edge = root_package.dependencies().next().expect("dependency edge should exist");
    assert_eq!(edge.alias().as_str(), "dependency");
    assert_eq!(edge.package(), dependency);

    let source = sources
        .load(&wyn_module_graph::ModuleKey::new(
            dependency,
            wyn_module_graph::ModulePath::new("src/lib.wyn").expect("valid source path"),
        ))
        .expect("verified dependency source should load");
    assert!(source.contains("package_name"));
    assert!(dependency_root.is_dir());
}

#[test]
fn local_dependency_must_satisfy_its_minimum() {
    let tree = TestTree::new();
    tree.package("dependency", "test/dependency", "v1.1.0", "");
    let root = tree.package(
        "root",
        "test/root",
        "v1.0.0",
        &dependency("dependency", "test/dependency", "v1.2.0", "../dependency"),
    );

    assert!(matches!(
        load_local_build(root),
        Err(LocalBuildError::VersionMismatch { .. })
    ));
}

#[test]
fn one_package_name_cannot_come_from_two_local_roots() {
    let tree = TestTree::new();
    tree.package("first", "test/shared", "v1.0.0", "");
    tree.package("second", "test/shared", "v1.0.0", "");
    let dependencies = format!(
        "[dependencies]\nfirst = {{ package = \"test/shared\", version = \"v1.0.0\", path = \"../first\" }}\nsecond = {{ package = \"test/shared\", version = \"v1.0.0\", path = \"../second\" }}\n"
    );
    let root = tree.package("root", "test/root", "v1.0.0", &dependencies);

    assert!(matches!(
        load_local_build(root),
        Err(LocalBuildError::ConflictingPackageRoots { .. })
    ));
}

#[test]
fn package_dependency_cycles_are_representable() {
    let tree = TestTree::new();
    let first = tree.package(
        "first",
        "test/first",
        "v1.0.0",
        &dependency("second", "test/second", "v1.0.0", "../second"),
    );
    tree.package(
        "second",
        "test/second",
        "v1.0.0",
        &dependency("first", "test/first", "v1.0.0", "../first"),
    );

    let build = load_local_build(first).expect("package dependency cycle should close");
    assert_eq!(build.into_parts().0.packages().count(), 2);
}

#[test]
fn local_input_recognizes_package_directories_manifests_and_source_roots() {
    let tree = TestTree::new();
    let root = tree.package("root", "test/root", "v1.0.0", "");

    let directory = load_local_input(&root)
        .expect("package directory should load")
        .expect("package directory should be recognized");
    assert_eq!(directory.into_parts().0.packages().count(), 1);

    let manifest = load_local_input(root.join("wyn.toml"))
        .expect("package manifest should load")
        .expect("package manifest should be recognized");
    assert_eq!(manifest.into_parts().0.packages().count(), 1);

    let example = root.join("test/example.wyn");
    fs::create_dir_all(example.parent().expect("example should have a parent"))
        .expect("example directory should be created");
    fs::write(&example, "def example: i32 = 1\n").expect("example source should be written");
    let source = load_local_input(&example)
        .expect("package source should load")
        .expect("package source should be recognized");
    assert_eq!(source.into_parts().0.root().path().as_str(), "test/example.wyn");

    let standalone = tree.root.join("standalone.wyn");
    fs::write(&standalone, "def standalone: i32 = 1\n").expect("standalone source should be written");
    assert!(
        load_local_input(standalone)
            .expect("standalone source should not fail package recognition")
            .is_none(),
        "standalone source should remain available to the compiler driver",
    );
}
