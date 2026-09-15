use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_CASE: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        loop {
            let sequence = NEXT_CASE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("wyn_egglog_{}_{sequence}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create test directory: {error}"),
            }
        }
    }

    fn source(&self, contents: &str) -> PathBuf {
        let path = self.0.join("input.wyn");
        fs::write(&path, contents).unwrap();
        path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove test directory");
    }
}

#[test]
fn egglog_saves_egg_output_and_prints_program_to_stdout() {
    let directory = TestDirectory::new();
    let source = directory.source("entry mapped(xs: []i32) []i32 = map(|x: i32| x + 17, xs)");
    let tlc = directory.0.join("input.tlc");
    let egg = directory.0.join("input.egg");
    let output = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("build")
        .arg(&source)
        .args(["--egglog", "--verbose", "--output-tlc"])
        .arg(&tlc)
        .arg("--egg-out")
        .arg(&egg)
        .output()
        .expect("run wyn");
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(output.status.success(), "{stderr}");
    assert!(!stdout.is_empty());
    let egg = fs::read_to_string(egg).unwrap();
    assert!(egg.starts_with("(datatype ProgramKey"), "{egg}");
    assert!(egg.contains("(Definition (DefinitionId"));
    assert!(egg.contains("(Screma "));
    assert!(egg.contains("(Expression (ExprId"));
    assert!(egg.contains("(Do (OperationId"));
    assert!(!egg.contains("TermId"));
    assert!(!egg.contains("OriginalSoac"));
    assert!(!egg.contains("(Let "));
    assert!(egg.contains("(Int \"17\")"));
    assert!(!egg.contains("mapped"));
    assert!(stderr.contains("from_tlc_egglog:"));
    assert!(!stderr.contains("to_egraph:"));
    assert!(!stderr.contains("egir_plan:"));
    assert!(!stderr.contains("wgsl_lower:"));
    assert!(tlc.is_file());
    for extension in ["spv", "wgsl", "json"] {
        assert!(!source.with_extension(extension).exists());
    }
}

#[test]
fn egglog_reports_source_errors_without_printing_a_partial_program() {
    let directory = TestDirectory::new();
    let source = directory.source("entry invalid() i32 = not_defined_here");
    let output =
        Command::new(env!("CARGO_BIN_EXE_wyn")).arg("build").arg(source).arg("--egglog").output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("not_defined_here"));
}

#[test]
fn normal_build_still_writes_backend_artifacts() {
    let directory = TestDirectory::new();
    let source = directory.source("entry scalar(x: i32) i32 = x + 1");
    let output = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("build")
        .arg(&source)
        .args(["--target", "wgsl"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    assert!(source.with_extension("wgsl").is_file());
    assert!(source.with_extension("json").is_file());
}

#[test]
fn egglog_rejects_backend_output_paths() {
    for option in ["--output", "--output-mir"] {
        let output = Command::new(env!("CARGO_BIN_EXE_wyn"))
            .args(["build", "unused.wyn", "--egglog", option, "unused.out"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("cannot be used with"));
    }
}
