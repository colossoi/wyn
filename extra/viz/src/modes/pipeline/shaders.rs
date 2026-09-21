use anyhow::{anyhow, Context, Result};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use wyn_host_interp::Program;

pub(super) fn load(program: &Program, path: &Path) -> Result<Option<BTreeMap<String, Vec<u8>>>> {
    let extension = path.extension().and_then(|extension| extension.to_str()).unwrap_or("");
    if extension.eq_ignore_ascii_case("wynhost") {
        return Ok(None);
    }
    if program.modules.len() > 1 {
        return Err(anyhow!(
            "a shader path requires a host program with one GPU module; \
             pass the .wynhost path to run a program with multiple modules"
        ));
    }
    let Some((name, module)) = program.modules.first_key_value() else {
        return Err(anyhow!("host program has no GPU module"));
    };
    let format = if extension.eq_ignore_ascii_case("wgsl") { ":wgsl" } else { ":spirv" };
    let declared = module.options.text(":format")?;
    if format != declared {
        return Err(anyhow!(
            "shader {} has format {format}, but the host program declares {declared}; \
             rebuild the shader and host program with the same target",
            path.display()
        ));
    }
    let bytes = fs::read(path).with_context(|| format!("reading shader {}", path.display()))?;
    Ok(Some(BTreeMap::from([(name.clone(), bytes)])))
}

#[cfg(test)]
#[path = "shaders_tests.rs"]
mod shaders_tests;
