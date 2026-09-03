use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    root: PathBuf,
    command: String,
    expect: String,
    diagnostic: Option<String>,
}

struct CaseCopy {
    root: PathBuf,
}

impl CaseCopy {
    fn new(source: &Path) -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should follow the Unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "wyn_package_functional_{}_{}",
            std::process::id(),
            unique
        ));
        copy_tree(source, &root);
        Self { root }
    }
}

impl Drop for CaseCopy {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root) {
            eprintln!("failed to remove test case `{}`: {error}", self.root.display());
        }
    }
}

fn copy_tree(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).expect("fixture destination should be created");
    for entry in fs::read_dir(source).expect("fixture directory should be readable") {
        let entry = entry.expect("fixture entry should be readable");
        let target = destination.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).expect("fixture file should be copied");
        }
    }
}

fn assert_local_manifests(root: &Path) {
    for entry in fs::read_dir(root).expect("fixture directory should be readable") {
        let entry = entry.expect("fixture entry should be readable");
        let path = entry.path();
        if path.is_dir() {
            assert_local_manifests(&path);
            continue;
        }
        if path.file_name().and_then(|name| name.to_str()) != Some("wyn.toml") {
            continue;
        }
        let source = fs::read_to_string(&path).expect("fixture manifest should be readable");
        let manifest: toml::Value = toml::from_str(&source).expect("fixture manifest should be TOML");
        let Some(dependencies) = manifest.get("dependencies").and_then(toml::Value::as_table) else {
            continue;
        };
        for (alias, dependency) in dependencies {
            let dependency = dependency.as_table().expect("fixture dependency should be an inline table");
            assert!(
                dependency.contains_key("path"),
                "dependency `{alias}` in `{}` is not a local path dependency",
                path.display(),
            );
            for source_key in ["git", "url", "registry"] {
                assert!(
                    !dependency.contains_key(source_key),
                    "dependency `{alias}` in `{}` uses forbidden source `{source_key}`",
                    path.display(),
                );
            }
        }
    }
}

fn run_case(case_directory: &Path) -> (Case, Output) {
    assert_local_manifests(case_directory);
    let source = fs::read_to_string(case_directory.join("case.toml"))
        .expect("functional case should contain case.toml");
    let case: Case = toml::from_str(&source).expect("functional case metadata should be valid");
    let copied = CaseCopy::new(case_directory);
    let output = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg(&case.command)
        .arg(copied.root.join(&case.root))
        .output()
        .expect("Wyn compiler should run");
    (case, output)
}

#[test]
fn local_package_cases() {
    let cases = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/module-packages/cases");
    let mut case_directories: Vec<_> = fs::read_dir(cases)
        .expect("functional case directory should be readable")
        .map(|entry| entry.expect("functional case should be readable").path())
        .filter(|path| path.is_dir())
        .collect();
    case_directories.sort();
    assert!(
        !case_directories.is_empty(),
        "at least one functional case is required"
    );

    for case_directory in case_directories {
        let (case, output) = run_case(&case_directory);
        let error = String::from_utf8_lossy(&output.stderr);
        match case.expect.as_str() {
            "success" => assert!(output.status.success(), "case `{}` failed:\n{error}", case.name,),
            "failure" => assert!(
                !output.status.success(),
                "case `{}` unexpectedly succeeded",
                case.name,
            ),
            expectation => panic!("case `{}` has unknown expectation `{expectation}`", case.name),
        }
        if let Some(diagnostic) = case.diagnostic {
            assert!(
                error.contains(&diagnostic),
                "case `{}` did not report `{diagnostic}`:\n{error}",
                case.name,
            );
        }
    }
}

#[test]
fn package_manifest_is_not_a_cli_input() {
    let case =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/module-packages/cases/local-dependency");
    assert_local_manifests(&case);
    let copied = CaseCopy::new(&case);
    let output = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("check")
        .arg(copied.root.join("app/wyn.toml"))
        .output()
        .expect("Wyn compiler should run");

    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "manifest input unexpectedly succeeded");
    assert!(
        error.contains("must be a package directory or `.wyn` source file"),
        "unexpected manifest-input diagnostic:\n{error}",
    );
}

#[test]
fn source_inside_a_package_uses_that_packages_dependency_plan() {
    let case =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/module-packages/cases/local-dependency");
    assert_local_manifests(&case);
    let copied = CaseCopy::new(&case);
    let output = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("check")
        .arg(copied.root.join("app/test/alternate.wyn"))
        .output()
        .expect("Wyn compiler should run");

    assert!(
        output.status.success(),
        "package source input failed:\n{}",
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn package_compiles_with_a_local_dependency() {
    let case =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/module-packages/cases/local-dependency");
    assert_local_manifests(&case);
    let copied = CaseCopy::new(&case);
    let output_path = copied.root.join("package.wgsl");
    let output = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("build")
        .arg(copied.root.join("app"))
        .args(["--target", "wgsl", "--output"])
        .arg(&output_path)
        .output()
        .expect("Wyn compiler should run");

    assert!(
        output.status.success(),
        "package compilation failed:\n{}",
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(output_path.is_file(), "package output should be written");
}
