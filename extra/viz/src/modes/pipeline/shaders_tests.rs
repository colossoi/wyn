use super::load;
use std::fs;
use std::path::Path;
use wyn_host_interp::Program;

const HOST: &str = r#"
(define-host-program :version 1)
(define-gpu-module 'shaders :format :wgsl :path "unavailable-default.wgsl")
"#;

#[test]
fn explicit_shader_path_supplies_the_module_bytes() {
    let program = Program::parse(HOST).unwrap();
    let path = std::env::temp_dir().join(format!("wyn-viz-selected-{}.wgsl", std::process::id()));
    let shader = b"@compute @workgroup_size(1) fn main() {}";
    fs::write(&path, shader).unwrap();
    let loaded = load(&program, &path);
    fs::remove_file(&path).unwrap();
    let sources = loaded.unwrap().unwrap();
    assert_eq!(sources["shaders"], shader);
}

#[test]
fn missing_explicit_shader_does_not_fall_back_to_the_host_path() {
    let program = Program::parse(HOST).unwrap();
    let path = std::env::temp_dir().join(format!("wyn-viz-missing-{}.wgsl", std::process::id()));
    let error = load(&program, &path).unwrap_err().to_string();
    assert!(error.contains(&path.display().to_string()), "{error}");
}

#[test]
fn shader_and_host_formats_must_agree() {
    let program = Program::parse(HOST).unwrap();
    let error = load(&program, Path::new("shader.spv")).unwrap_err().to_string();
    assert!(error.contains("format :spirv"), "{error}");
    assert!(error.contains("declares :wgsl"), "{error}");
}

#[test]
fn direct_host_path_uses_declared_module_paths() {
    let program = Program::parse(HOST).unwrap();
    assert!(load(&program, Path::new("program.wynhost")).unwrap().is_none());
}

#[test]
fn shader_path_rejects_ambiguous_module_selection() {
    let source = format!("{HOST}\n(define-gpu-module 'other :format :wgsl :path \"other.wgsl\")");
    let program = Program::parse(&source).unwrap();
    let error = load(&program, Path::new("shader.wgsl")).unwrap_err().to_string();
    assert!(error.contains("one GPU module"), "{error}");
}
