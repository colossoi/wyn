mod support;

use std::fs;
use std::path::PathBuf;
use std::process::Command;

struct TestDirectory(PathBuf);

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn validators_accept_compiled_shaders_and_reject_invalid_output() {
    let directory =
        TestDirectory(std::env::temp_dir().join(format!("wyn_shader_validation_{}", std::process::id())));
    fs::create_dir(&directory.0).unwrap();
    let source = directory.0.join("input.wyn");
    fs::write(&source, "entry main(xs: []i32) []i32 = map(|x| x + 1, xs)").unwrap();
    let missing_validator = directory.0.join("missing-spirv-val");

    for target in ["spirv", "wgsl"] {
        let shader = directory.0.join(format!("output.{target}"));
        let result = Command::new(env!("CARGO_BIN_EXE_wyn"))
            .arg("build")
            .arg(&source)
            .args(["--target", target, "--output"])
            .arg(&shader)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        support::validate_shader(&shader, target);
        // Exercise fallback even on machines that have SPIRV-Tools installed.
        support::validate_shader_with(&shader, target, &missing_validator).unwrap();

        #[cfg(unix)]
        if target == "spirv" {
            use std::os::unix::fs::PermissionsExt;
            let validator = directory.0.join("rejecting-spirv-val");
            fs::write(&validator, "#!/bin/sh\necho validation-rejected >&2\nexit 1\n").unwrap();
            fs::set_permissions(&validator, fs::Permissions::from_mode(0o700)).unwrap();
            let error = support::validate_shader_with(&shader, target, &validator).unwrap_err();
            assert!(error.contains("validation-rejected"), "{error}");
            fs::write(&validator, "#!/bin/sh\nexit 0\n").unwrap();
            support::validate_shader_with(&shader, target, &validator).unwrap();
            let valid_shader = fs::read(&shader).unwrap();
            fs::write(&shader, "").unwrap();
            let error = support::validate_shader_with(&shader, target, &validator).unwrap_err();
            assert!(error.contains("shader output is empty"), "{error}");
            fs::write(&shader, valid_shader).unwrap();
            // Launch errors other than NotFound must not silently fall back either.
            fs::set_permissions(&validator, fs::Permissions::from_mode(0o600)).unwrap();
            let error = support::validate_shader_with(&shader, target, &validator).unwrap_err();
            assert!(error.contains("could not run spirv-val"), "{error}");
        }

        fs::write(&shader, "invalid shader").unwrap();
        assert!(support::validate_shader_with(&shader, target, &missing_validator).is_err());

        for empty in ["", " \n\t"] {
            fs::write(&shader, empty).unwrap();
            let error = support::validate_shader_with(&shader, target, &missing_validator).unwrap_err();
            assert!(error.contains("shader output is empty"), "{error}");
        }

        if target == "wgsl" {
            // Parseable WGSL that fails semantic validation (wrong return type).
            fs::write(&shader, "fn wrong() -> i32 { return true; }").unwrap();
            let error = support::validate_shader_with(&shader, target, &missing_validator).unwrap_err();
            assert!(error.contains("Naga validation failed"), "{error}");
        }
    }
}
