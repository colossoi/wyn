use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is before Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("wyn-wgsl-descriptor-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&path).expect("create test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn wgsl_compile_emits_multistage_pipeline_descriptor() {
    let dir = TestDir::new();
    let input = dir.path().join("reduce.wyn");
    let output = dir.path().join("reduce.wgsl");
    let descriptor = dir.path().join("reduce.json");
    fs::write(
        &input,
        r#"
#[compute]
entry sum(xs: []i32) i32 =
  reduce(|a: i32, b: i32| a + b, 0i32, xs)
"#,
    )
    .expect("write Wyn source");

    let result = Command::new(env!("CARGO_BIN_EXE_wyn"))
        .args([
            "compile",
            input.to_str().unwrap(),
            "-t",
            "wgsl",
            "-o",
            output.to_str().unwrap(),
        ])
        .output()
        .expect("run wyn compiler");
    assert!(
        result.status.success(),
        "WGSL compilation failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.is_file(), "WGSL output was not written");
    assert!(
        descriptor.is_file(),
        "sibling pipeline descriptor was not written"
    );

    let json: serde_json::Value = serde_json::from_slice(&fs::read(&descriptor).expect("read descriptor"))
        .expect("valid descriptor JSON");
    let pipelines = json["pipelines"].as_array().expect("pipelines array");
    assert_eq!(pipelines.len(), 1, "one source entry should produce one pipeline");
    let stages = pipelines[0]["stages"].as_array().expect("compute stages array");
    assert!(
        stages.len() >= 2,
        "parallel reduction must publish its multi-stage dispatch schedule"
    );

    let wgsl = fs::read_to_string(&output).expect("read WGSL output");
    for stage in stages {
        let entry_point = stage["entry_point"].as_str().expect("stage entry point");
        assert!(
            wgsl.contains(&format!("fn {entry_point}(")),
            "descriptor stage {entry_point:?} is absent from the WGSL module"
        );
    }
}
