use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct LocalPackage {
    directory: PathBuf,
}

impl LocalPackage {
    fn new() -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should follow the Unix epoch")
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("wyn_local_modules_{}_{}", std::process::id(), unique));
        fs::create_dir_all(&directory).expect("test package directory should be created");
        Self { directory }
    }

    fn write(&self, relative: impl AsRef<Path>, source: &str) -> PathBuf {
        let path = self.directory.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("source directory should be created");
        }
        fs::write(&path, source).expect("test source should be written");
        path
    }
}

impl Drop for LocalPackage {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.directory) {
            eprintln!(
                "failed to remove test package directory `{}`: {error}",
                self.directory.display()
            );
        }
    }
}

#[test]
fn check_loads_transitive_local_modules() {
    let package = LocalPackage::new();
    let root = package.write(
        "main.wyn",
        concat!(
            "module dependency = import \"library/dependency\"\n",
            "def answer: i32 = dependency.value\n",
        ),
    );
    package.write(
        "library/dependency.wyn",
        concat!("import \"shared\"\n", "def value: i32 = shared_value\n",),
    );
    package.write("library/shared.wyn", "def shared_value: i32 = 42\n");

    let output = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("check")
        .arg(root)
        .output()
        .expect("Wyn compiler should run");

    assert!(
        output.status.success(),
        "local module check failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn check_accepts_nested_semantic_module_in_imported_source() {
    let package = LocalPackage::new();
    let root = package.write(
        "main.wyn",
        concat!(
            "module dependency = import \"library/dependency\"\n",
            "def answer: i32 = dependency.nested.value\n",
        ),
    );
    package.write(
        "library/dependency.wyn",
        "module nested = { def value: i32 = 42 }\n",
    );

    let output = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("check")
        .arg(root)
        .output()
        .expect("Wyn compiler should run");

    assert!(
        output.status.success(),
        "nested semantic module check failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn check_rejects_entry_in_imported_source() {
    let package = LocalPackage::new();
    let root = package.write("main.wyn", "import \"library/dependency\"\n");
    package.write(
        "library/dependency.wyn",
        "entry imported(value: i32) i32 = value\n",
    );

    let output = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("check")
        .arg(root)
        .output()
        .expect("Wyn compiler should run");
    let error = String::from_utf8_lossy(&output.stderr);

    assert!(!output.status.success(), "imported entry should be rejected");
    assert!(
        error.contains("entry `imported` is not declared directly in the root source module"),
        "unexpected imported-entry diagnostic:\n{error}"
    );
}

#[test]
fn check_reports_undeclared_package_alias_at_import() {
    let package = LocalPackage::new();
    let root = package.write("main.wyn", "import \"pkg:missing\"\n");

    let output = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("check")
        .arg(root)
        .output()
        .expect("Wyn compiler should run");
    let error = String::from_utf8_lossy(&output.stderr);

    assert!(
        !output.status.success(),
        "undeclared package alias should be rejected"
    );
    assert!(
        error.contains("main.wyn:1:1: unknown package dependency alias `missing`"),
        "unexpected package-alias diagnostic:\n{error}"
    );
    assert!(
        !error.contains("ModuleId("),
        "diagnostic leaked an internal module ID:\n{error}"
    );
    assert!(
        !error.contains(&package.directory.to_string_lossy().to_string()),
        "diagnostic leaked the temporary package root:\n{error}"
    );
}

#[test]
fn check_rejects_import_outside_package_root() {
    let package = LocalPackage::new();
    let root = package.write("main.wyn", "import \"../outside\"\n");

    let output = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("check")
        .arg(root)
        .output()
        .expect("Wyn compiler should run");
    let error = String::from_utf8_lossy(&output.stderr);

    assert!(!output.status.success(), "escaping import should be rejected");
    assert!(
        error.contains("main.wyn:1:1: invalid import path: module path escapes its package root"),
        "unexpected escaping-import diagnostic:\n{error}"
    );
}

#[test]
fn wgsl_output_is_independent_of_source_directory() {
    let first = LocalPackage::new();
    let second = LocalPackage::new();
    let source = concat!(
        "module Dependency = import \"library/dependency\"\n",
        "entry compute(value: i32) i32 = Dependency.identity(value)\n",
    );
    let dependency = "def identity<T>(value: T) T = value\n";
    let first_root = first.write("main.wyn", source);
    let second_root = second.write("main.wyn", source);
    first.write("library/dependency.wyn", dependency);
    second.write("library/dependency.wyn", dependency);
    let first_output = first.directory.join("output.wgsl");
    let second_output = second.directory.join("output.wgsl");

    for (root, output) in [(&first_root, &first_output), (&second_root, &second_output)] {
        let result = Command::new(env!("CARGO_BIN_EXE_wyn"))
            .arg("build")
            .arg(root)
            .args(["--target", "wgsl", "--output"])
            .arg(output)
            .output()
            .expect("Wyn compiler should run");
        assert!(
            result.status.success(),
            "local module compilation failed:\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }

    assert_eq!(
        fs::read(first_output).expect("first WGSL output should exist"),
        fs::read(second_output).expect("second WGSL output should exist"),
        "generated WGSL should not depend on the source cache directory",
    );
}

#[test]
fn check_reports_semantic_error_in_imported_source() {
    let package = LocalPackage::new();
    let root = package.write("main.wyn", "module Dependency = import \"library/dependency\"\n");
    package.write("library/dependency.wyn", "def broken: i32 = true\n");

    let output = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("check")
        .arg(root)
        .output()
        .expect("Wyn compiler should run");
    let error = String::from_utf8_lossy(&output.stderr);

    assert!(
        !output.status.success(),
        "dependency type error should fail checking"
    );
    assert!(
        error.contains("library/dependency.wyn:1:") && error.contains("Type error:"),
        "unexpected dependency diagnostic:\n{error}"
    );
    assert!(
        !error.contains("ModuleId("),
        "diagnostic leaked an internal module ID:\n{error}"
    );
    assert!(
        !error.contains(&package.directory.to_string_lossy().to_string()),
        "diagnostic leaked the temporary package root:\n{error}"
    );
}

#[test]
fn check_reports_type_hole_in_imported_source() {
    let package = LocalPackage::new();
    let root = package.write("main.wyn", "module Dependency = import \"library/dependency\"\n");
    package.write("library/dependency.wyn", "def incomplete: i32 = ???\n");

    let output = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("check")
        .arg(root)
        .output()
        .expect("Wyn compiler should run");
    let error = String::from_utf8_lossy(&output.stderr);

    assert_eq!(
        output.status.code(),
        Some(2),
        "type holes use their dedicated exit code"
    );
    assert!(
        error.contains("at library/dependency.wyn:1:") && error.contains("inferred `i32`"),
        "unexpected type-hole diagnostic:\n{error}"
    );
    assert!(
        !error.contains("ModuleId("),
        "diagnostic leaked an internal module ID:\n{error}"
    );
    assert!(
        !error.contains(&package.directory.to_string_lossy().to_string()),
        "diagnostic leaked the temporary package root:\n{error}"
    );
}

#[test]
fn compile_reports_backend_error_in_imported_source() {
    let package = LocalPackage::new();
    let root = package.write(
        "main.wyn",
        concat!(
            "module Dependency = import \"library/dependency\"\n",
            "entry compute_main(value: u32) u32 = Dependency.multiply(value)\n",
        ),
    );
    package.write(
        "library/dependency.wyn",
        "def multiply(value: u32) u32 = u32.u64(u64.u32(value) * 2u64)\n",
    );
    let output_path = package.directory.join("output.wgsl");

    let output = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("build")
        .arg(root)
        .args(["--target", "wgsl", "--wgsl-emulate-u64", "--output"])
        .arg(output_path)
        .output()
        .expect("Wyn compiler should run");
    let error = String::from_utf8_lossy(&output.stderr);

    assert!(
        !output.status.success(),
        "unsupported WGSL source should fail lowering"
    );
    assert!(
        error.contains("library/dependency.wyn:1:") && error.contains("u64 operator '*'"),
        "unexpected backend diagnostic:\n{error}"
    );
    assert!(
        !error.contains("ModuleId("),
        "diagnostic leaked an internal module ID:\n{error}"
    );
    assert!(
        !error.contains(&package.directory.to_string_lossy().to_string()),
        "diagnostic leaked the temporary package root:\n{error}"
    );
}
