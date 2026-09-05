use std::convert::Infallible;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use wyn_module_graph::{DependencyAlias, ImportSiteId, ImportTarget, ModuleId, ModuleParser, TextRange};

use super::{prepare_package, PreparationError};

static TEST_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TestParser;

impl ModuleParser for TestParser {
    type Parsed = ();
    type Error = Infallible;

    fn parse(
        &mut self,
        _module: ModuleId,
        source: &str,
        report_import: &mut dyn FnMut(ImportSiteId, ImportTarget, TextRange),
    ) -> Result<Self::Parsed, Self::Error> {
        if source == "load-dependency\n" {
            let end = u32::try_from(source.len()).expect("test source should fit in a byte range");
            report_import(
                ImportSiteId::from(0),
                ImportTarget::Dependency {
                    alias: DependencyAlias::new("dependency").expect("valid dependency alias"),
                    module: None,
                },
                TextRange::new(0, end).expect("valid import range"),
            );
        }
        Ok(())
    }
}

struct TestTree {
    root: PathBuf,
}

impl TestTree {
    fn new() -> Self {
        loop {
            let sequence = TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let root =
                std::env::temp_dir().join(format!("wyn_package_manager_{}_{sequence}", std::process::id()));
            match fs::create_dir(&root) {
                Ok(()) => return Self { root },
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("test tree should be created: {error}"),
            }
        }
    }

    fn package(&self, relative: &str, name: &str, version: &str, dependencies: &str) -> PathBuf {
        self.package_with_wyn(relative, name, version, "v0.1.0", dependencies)
    }

    fn package_with_wyn(
        &self,
        relative: &str,
        name: &str,
        version: &str,
        wyn: &str,
        dependencies: &str,
    ) -> PathBuf {
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
                    "wyn = \"{}\"\n",
                    "library = \"src/lib.wyn\"\n",
                    "{}",
                ),
                name, version, wyn, dependencies
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

#[test]
fn packages_cannot_require_a_newer_wyn_version() {
    let tree = TestTree::new();
    let root = tree.package_with_wyn("root", "test/root", "v1.0.0", "v999.0.0", "");

    assert!(matches!(
        prepare_package(root, None),
        Err(PreparationError::UnsupportedWynVersion { .. })
    ));
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
    fs::write(root.join("src/lib.wyn"), "load-dependency\n").expect("root source should be written");

    let input = prepare_package(root, None).expect("local package should prepare");
    let graph = input.load(&mut TestParser).expect("local graph should load");
    let packages = graph.package_graph();
    assert_eq!(packages.packages().count(), 2);
    let dependency = package_id(packages, "test/dependency");
    let root_package = packages.package(packages.root().package()).expect("root package should exist");
    let edge = root_package.dependencies().next().expect("dependency edge should exist");
    assert_eq!(edge.alias().as_str(), "dependency");
    assert_eq!(edge.package(), dependency);

    let dependency_module = graph
        .modules()
        .find_map(|(id, module)| (module.key().package() == dependency).then_some(id))
        .expect("dependency module should load");
    let source = graph.source(dependency_module).expect("verified dependency source should load");
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
fn unsupported_github_host_is_rejected_before_compilation() {
    let tree = TestTree::new();
    let root = tree.package(
        "root",
        "test/root",
        "v1.0.0",
        concat!(
            "[dependencies]\n",
            "dependency = { package = \"test/dependency\", version = \"v1.2.0\", ",
            "github = \"https://example.invalid/dependency\" }\n",
        ),
    );

    assert!(matches!(
        prepare_package(root, None),
        Err(PreparationError::DependencyMaterialization {
            detail,
            ..
        }) if detail.contains("https://example.invalid/dependency")
    ));
}

#[test]
fn one_package_name_cannot_come_from_two_local_roots() {
    let tree = TestTree::new();
    tree.package("first", "test/shared", "v1.0.0", "");
    tree.package("second", "test/shared", "v1.0.0", "");
    let dependencies = "[dependencies]\nfirst = { package = \"test/shared\", version = \"v1.0.0\", path = \"../first\" }\nsecond = { package = \"test/shared\", version = \"v1.0.0\", path = \"../second\" }\n".to_string();
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
    let graph = input.load(&mut TestParser).expect("root source should load");
    assert_eq!(graph.package_graph().packages().count(), 2);
}
