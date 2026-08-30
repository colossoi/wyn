use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const GRAPHICS_SOURCE: &str = r#"
def vertex_main(vertex: vertex_invocation) vertex<vec2f32> =
  let x = if vertex.vertex_index == 0u32 then -1.0 else if vertex.vertex_index == 1u32 then 3.0 else -1.0 in
  let y = if vertex.vertex_index == 0u32 then -1.0 else if vertex.vertex_index == 1u32 then -1.0 else 3.0 in
  vertex_output(@[x, y, 0.0, 1.0], @[0.0, 0.0])

entry frame(screen: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(direct_draw(3u32, 1u32), vertex_main) in
  shade(screen, covered, |fragment| @[fragment.position.x, 0.0, 0.0, 1.0])
"#;

fn temp_case(extension: &str) -> (PathBuf, PathBuf, PathBuf) {
    let unique = SystemTime::now().duration_since(UNIX_EPOCH).expect("clock").as_nanos();
    let directory = std::env::temp_dir().join(format!("wyn_direct_{}_{}", std::process::id(), unique));
    fs::create_dir_all(&directory).expect("create test directory");
    let source = directory.join("direct.wyn");
    let output = directory.join(format!("direct.{extension}"));
    fs::write(&source, GRAPHICS_SOURCE).expect("write test source");
    (directory, source, output)
}

fn compile(source: &PathBuf, output: &PathBuf, target: &str, direct_flag: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_wyn"))
        .arg("compile")
        .arg(source)
        .arg("--graphics")
        .arg("--target")
        .arg(target)
        .arg("--output")
        .arg(output)
        .arg(direct_flag)
        .output()
        .expect("run wyn compiler")
}

#[test]
fn direct_compiles_authored_graphics_for_both_backends() {
    for (target, extension) in [("spirv", "spv"), ("wgsl", "wgsl")] {
        let (directory, source, output) = temp_case(extension);
        let result = compile(&source, &output, target, "--direct");
        assert!(
            result.status.success(),
            "direct {target} compilation failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(output.is_file(), "direct {target} output was not written");

        let descriptor = fs::read_to_string(output.with_extension("json")).expect("pipeline descriptor");
        assert!(descriptor.contains(r#""kind": "graphics""#));
        assert!(!descriptor.contains(r#""kind": "compute""#));
        assert!(!descriptor.contains(r#""usage": "intermediate""#));

        fs::remove_dir_all(directory).expect("remove test directory");
    }
}

#[test]
fn removed_direct_wgsl_spelling_is_rejected() {
    let (directory, source, output) = temp_case("wgsl");
    let result = compile(&source, &output, "wgsl", "--direct-wgsl");
    assert!(
        !result.status.success(),
        "removed flag unexpectedly remained accepted"
    );
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("--direct-wgsl"),
        "unexpected diagnostic: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    fs::remove_dir_all(directory).expect("remove test directory");
}
