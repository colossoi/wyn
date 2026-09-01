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
fn check_reports_nested_semantic_module_in_imported_source() {
    let package = LocalPackage::new();
    let root = package.write("main.wyn", "module dependency = import \"library/dependency\"\n");
    package.write(
        "library/dependency.wyn",
        "module nested = { def value: i32 = 42 }\n",
    );

    let output = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("check")
        .arg(root)
        .output()
        .expect("Wyn compiler should run");
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !output.status.success(),
        "nested semantic module should fail checking"
    );
    assert!(
        stderr.contains("module 'dependency' contains nested semantic module 'nested'"),
        "unexpected diagnostic:\n{stderr}"
    );
}
