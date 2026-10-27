use std::fs;
use std::io::ErrorKind;
use std::path::Path;
use std::process::Command;

/// Validate the explicit target, since CLI tests also use neutral file extensions.
pub fn validate_shader(path: &Path, target: &str) {
    validate_shader_with(path, target, Path::new("spirv-val"))
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
}

pub fn validate_shader_with(path: &Path, target: &str, spirv_val: &Path) -> Result<(), String> {
    let source = fs::read(path).map_err(|error| format!("read shader: {error}"))?;
    if source.is_empty() || source.iter().all(u8::is_ascii_whitespace) {
        return Err("shader output is empty".into());
    }
    if target == "spirv" {
        match Command::new(spirv_val).arg(path).output() {
            Ok(output) if output.status.success() => return Ok(()),
            Ok(output) => {
                return Err(format!(
                    "spirv-val failed ({}):\n{}\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr),
                ));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(format!("could not run spirv-val: {error}")),
        }
    }

    // Naga supports a subset of SPIR-V; unsupported features fail rather than skip validation.
    let module = match target {
        "spirv" => naga::front::spv::parse_u8_slice(&source, &Default::default())
            .map_err(|error| format!("Naga SPIR-V parse failed: {error}"))?,
        "wgsl" => {
            let source = std::str::from_utf8(&source).map_err(|error| error.to_string())?;
            naga::front::wgsl::parse_str(source).map_err(|error| error.emit_to_string(source))?
        }
        _ => return Err(format!("unknown shader target: {target}")),
    };
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| format!("Naga validation failed: {error:?}"))?;
    Ok(())
}
