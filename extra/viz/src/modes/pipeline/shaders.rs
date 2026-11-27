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
    let mut external =
        program.modules.iter().filter(|(_, module)| module.options.optional(":path").is_some());
    let Some((name, module)) = external.next() else {
        return Err(anyhow!("host program has no external GPU module"));
    };
    if external.next().is_some() {
        return Err(anyhow!(
            "a shader path requires a host program with one GPU module; \
             pass the .wynhost path to run a program with multiple modules"
        ));
    }
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
