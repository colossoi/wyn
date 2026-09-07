use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

const UNUSED_PARAMETERS: &str = "entry main(first: i32, second: i32, third: i32) i32 = 0\n";

struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn new() -> Self {
        loop {
            let sequence = TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("wyn_warning_output_{}_{sequence}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Self { path },
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("test directory should be created: {error}"),
            }
        }
    }

    fn source(&self, contents: &str) -> PathBuf {
        let path = self.path.join("warnings.wyn");
        fs::write(&path, contents).expect("test source should be written");
        path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.path) {
            eprintln!("failed to remove `{}`: {error}", self.path.display());
        }
    }
}

fn check(source: &Path, warning_limit: usize) -> Output {
    Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("check")
        .arg(source)
        .arg("--max-warnings")
        .arg(warning_limit.to_string())
        .output()
        .expect("Wyn compiler should run")
}

#[test]
fn check_prints_every_warning_below_the_limit() {
    let directory = TestDirectory::new();
    let source = directory.source(UNUSED_PARAMETERS);
    let output = check(&source, 10);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "check failed:\n{stderr}");
    assert_eq!(stderr.matches("warning: unused parameter").count(), 3);
    let first = stderr.find("`first`").expect("first warning should be printed");
    let second = stderr.find("`second`").expect("second warning should be printed");
    let third = stderr.find("`third`").expect("third warning should be printed");
    assert!(
        first < second && second < third,
        "warnings are not in source order:\n{stderr}"
    );
    assert!(!stderr.contains("additional warnings omitted"));
}

#[test]
fn check_caps_warning_output_and_reports_the_omitted_count() {
    let directory = TestDirectory::new();
    let source = directory.source(UNUSED_PARAMETERS);
    let output = check(&source, 2);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "check failed:\n{stderr}");
    assert_eq!(stderr.matches("warning: unused parameter").count(), 2);
    assert!(stderr.contains("`first`"));
    assert!(stderr.contains("`second`"));
    assert!(!stderr.contains("`third`"));
    assert!(
        stderr.contains("note: 1 additional warnings omitted (limit 2; use --max-warnings to show more)")
    );
}

#[test]
fn zero_warning_limit_suppresses_warning_output() {
    let directory = TestDirectory::new();
    let source = directory.source(UNUSED_PARAMETERS);
    let output = check(&source, 0);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "check failed:\n{stderr}");
    assert!(
        !stderr.contains("warning:"),
        "warnings were not suppressed:\n{stderr}"
    );
    assert!(!stderr.contains("additional warnings omitted"));
}

#[test]
fn warning_limit_does_not_hide_type_hole_errors() {
    let directory = TestDirectory::new();
    let source = directory.source("entry main(unused: i32) i32 = ???\n");
    let output = check(&source, 0);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert_eq!(output.status.code(), Some(2), "unexpected status:\n{stderr}");
    assert!(stderr.contains("error: type hole inferred as `i32`"));
    assert!(
        !stderr.contains("warning:"),
        "warnings were not suppressed:\n{stderr}"
    );
}

#[test]
fn build_prints_warnings_without_preventing_output() {
    let directory = TestDirectory::new();
    let source = directory.source(UNUSED_PARAMETERS);
    let artifact = directory.path.join("warnings.spv");
    let output = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("build")
        .arg(&source)
        .arg("--output")
        .arg(&artifact)
        .arg("--max-warnings")
        .arg("10")
        .output()
        .expect("Wyn compiler should run");
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "build failed:\n{stderr}");
    assert_eq!(stderr.matches("warning: unused parameter").count(), 3);
    assert!(artifact.is_file(), "build output was not written");
}
