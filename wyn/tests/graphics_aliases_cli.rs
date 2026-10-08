//! CLI wiring only. Graphics semantics and serialization live in
//! wyn-core/src/graphics_host_tests.rs and tlc/stage_extract_tests.rs.
use std::fs;
use std::path::PathBuf;
use std::process::Command;

struct TestDirectory(PathBuf);

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn build_graphics_artifacts(target: &str, optimize: bool) {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    let directory = TestDirectory(std::env::temp_dir().join(format!(
        "wyn graphics CLI {}_{target}_{optimize}_{timestamp}",
        std::process::id()
    )));
    fs::create_dir(&directory.0).expect("create test directory");
    fs::create_dir(directory.0.join("source files")).expect("create source directory");
    fs::create_dir(directory.0.join("generated shaders")).expect("create output directory");
    fs::write(
        directory.0.join("source files/scene.wyn"),
        include_str!("../../testfiles/regressions/render_helper_tuple.wyn"),
    )
    .expect("write source");

    // Relative paths with spaces exercise argument/path handling. The neutral
    // extension ensures --target, rather than the output suffix, selects codegen.
    let output = "generated shaders/compiled graphics.shader";
    let mut command = Command::new(env!("CARGO_BIN_EXE_wyn"));
    command.current_dir(&directory.0).args([
        "build",
        "--graphics",
        "--target",
        target,
        "--max-warnings",
        "0",
        "source files/scene.wyn",
        "--output",
        output,
    ]);
    if optimize {
        command.arg("-O");
    }
    let result = command.output().expect("run compiler");
    assert!(
        result.status.success(),
        "{target}, -O={optimize}: {}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let output = directory.0.join(output);
    let shader = fs::read(&output).expect("shader artifact at requested path");
    match target {
        "spirv" => {
            assert!(shader.len() >= 20, "SPIR-V header");
            assert_eq!(shader.len() % 4, 0, "SPIR-V word alignment");
            assert_eq!(&shader[..4], &0x0723_0203u32.to_le_bytes(), "SPIR-V magic");
        }
        "wgsl" => {
            let shader = String::from_utf8(shader).expect("WGSL text");
            for stage in ["@compute", "@vertex", "@fragment"] {
                assert!(shader.contains(stage), "missing {stage} entry point");
            }
        }
        _ => unreachable!("test uses concrete targets"),
    }
    let host = fs::read_to_string(output.with_extension("wynhost"))
        .expect("host artifact beside requested shader");
    assert!(host.starts_with("(define-host-program :version 1)"), "{host}");
    assert!(
        host.contains(&format!(
            "(define-gpu-module 'shaders :format :{target} :path \"compiled graphics.shader\")"
        )),
        "host must reference the emitted format and sibling shader: {host}"
    );
    assert!(host.contains("(define-host-entry "), "host entry was emitted");
}

#[test]
fn graphics_build_writes_spirv_artifacts() {
    build_graphics_artifacts("spirv", false);
}

#[test]
fn optimized_graphics_build_writes_spirv_artifacts() {
    build_graphics_artifacts("spirv", true);
}

#[test]
fn graphics_build_writes_wgsl_artifacts() {
    build_graphics_artifacts("wgsl", false);
}

#[test]
fn optimized_graphics_build_writes_wgsl_artifacts() {
    build_graphics_artifacts("wgsl", true);
}
