use std::fs;
use std::io::ErrorKind;
use std::path::Path;
use std::process::Command;

/// Resolve authored entries through WHL instead of depending on generated shader names.
pub fn source_shaders(source: &str, entry: &str) -> std::collections::BTreeSet<String> {
    use wyn_host_interp::{Form, Program};
    let program = Program::parse(source).expect("valid host ABI");
    let forms = wyn_host_interp::read(source).unwrap();
    let mut shaders = std::collections::BTreeSet::new();
    let Ok(entry) = program.entry(entry) else {
        return shaders;
    };
    let mut pending = vec![entry.function.clone()];
    let mut seen = std::collections::BTreeSet::new();
    fn visit(
        form: &Form,
        program: &Program,
        shaders: &mut std::collections::BTreeSet<String>,
        pending: &mut Vec<String>,
    ) {
        let Form::List(values) = form else {
            return;
        };
        if let Some(Form::Symbol(op)) = values.first() {
            if op == "quote" {
                return;
            }
            pending.push(op.clone());
            if matches!(op.as_str(), "gpu-draw" | "gpu-dispatch") {
                if let Some(Form::List(quoted)) = values.get(1) {
                    if let [Form::Symbol(quote), Form::Symbol(name)] = quoted.as_slice() {
                        assert_eq!(quote, "quote");
                        if let Some(kernel) = program.kernels.get(name) {
                            shaders.insert(kernel.options.text(":entry").unwrap().into());
                        }
                        if let Some(graphics) = program.graphics.get(name) {
                            for stage in [":vertex", ":fragment"] {
                                if let Some(value) = graphics
                                    .options
                                    .optional(stage)
                                    .and_then(|value| value.list().ok())
                                    .and_then(|values| values.get(1))
                                {
                                    shaders.insert(value.text().unwrap().into());
                                }
                            }
                        }
                    }
                }
            }
        }
        for value in values {
            visit(value, program, shaders, pending);
        }
    }
    while let Some(function) = pending.pop() {
        if !seen.insert(function.clone()) {
            continue;
        }
        if let Some(form) = forms.iter().find(|form| matches!(form, Form::List(values)
            if matches!(values.as_slice(), [Form::Symbol(kind), Form::Symbol(name), ..] if kind == "defun" && name == &function))) {
            visit(form, &program, &mut shaders, &mut pending);
        }
    }
    shaders
}

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
