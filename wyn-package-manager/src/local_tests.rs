use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use wyn_module_graph::SourceReader;

use super::{prepare_package, PreparationError};

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

fn package_id(packages: &wyn_module_graph::PackageGraph, name: &str) -> wyn_module_graph::PackageId {
    packages
        .packages()
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

    let input = prepare_package(root, None).expect("local graph should load");
    let (packages, mut sources) = input.into_parts();
    assert_eq!(packages.packages().count(), 2);
    let dependency = package_id(&packages, "test/dependency");
    let root_package = packages.package(packages.root().package()).expect("root package should exist");
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
        prepare_package(root, None),
        Err(PreparationError::VersionMismatch { .. })
    ));
}

#[test]
fn git_dependency_requires_materialization_before_compilation() {
    let tree = TestTree::new();
    let root = tree.package(
        "root",
        "test/root",
        "v1.0.0",
        concat!(
            "[dependencies]\n",
            "dependency = { package = \"test/dependency\", version = \"v1.2.0\", ",
            "git = \"https://example.invalid/dependency\" }\n",
        ),
    );

    assert!(matches!(
        prepare_package(root, None),
        Err(PreparationError::MaterializationUnavailable { repository, .. })
            if repository == "https://example.invalid/dependency"
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
        prepare_package(root, None),
        Err(PreparationError::ConflictingPackageRoots { .. })
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

    let input = prepare_package(first, None).expect("package dependency cycle should close");
    assert_eq!(input.into_parts().0.packages().count(), 2);
}
