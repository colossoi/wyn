//! `validate` subcommand — headless naga / wgpu validation of a WGSL
//! or SPIR-V module. WGSL is parsed directly via naga (no GPU needed);
//! SPIR-V goes through wgpu so the same validation path that a real
//! pipeline build would take is exercised.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use rspirv::binary::parse_words;
use rspirv::dr::{Loader, Operand};
use wgpu::{
    DeviceDescriptor, Instance, InstanceDescriptor, InstanceFlags, PowerPreference,
    RequestAdapterOptions, Trace,
};

use crate::spirv::load_spirv_module;

/// Parse and validate a WGSL source file via naga. Synchronous — naga
/// doesn't need a GPU device, unlike the SPIR-V path.
pub fn validate_wgsl_file(path: &Path, verbose: bool) -> Result<()> {
    eprintln!("[validate] Loading {}", path.display());
    let src = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let module =
        naga::front::wgsl::parse_str(&src).map_err(|e| anyhow::anyhow!("WGSL parse error:\n{}", e))?;
    if verbose {
        for (_, ep) in module.entry_points.iter().enumerate() {
            eprintln!("[validate] Entry point: {} ({:?})", ep.name, ep.stage);
        }
    }
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator.validate(&module).map_err(|e| anyhow::anyhow!("WGSL validation failed:\n{:?}", e))?;
    eprintln!("[validate] OK — module parsed and validated");
    Ok(())
}

pub async fn validate_spirv(path: &Path, verbose: bool) -> Result<()> {
    let instance = Instance::new(&InstanceDescriptor {
        flags: InstanceFlags::VALIDATION,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .context("No suitable GPU adapter found")?;

    if verbose {
        let info = adapter.get_info();
        eprintln!("[validate] Adapter: {} ({:?})", info.name, info.backend);
    }

    let (device, _queue) = adapter
        .request_device(&DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: Trace::Off,
        })
        .await
        .context("Failed to create GPU device")?;

    if verbose {
        // Show detected entry points
        let bytes = fs::read(path)?;
        if bytes.len() >= 20 && bytes.len() % 4 == 0 {
            let words: Vec<u32> =
                bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let mut loader = Loader::new();
            if parse_words(&words, &mut loader).is_ok() {
                let module = loader.module();
                for ep in &module.entry_points {
                    let name = ep.operands.iter().find_map(|op| {
                        if let Operand::LiteralString(s) = op { Some(s.as_str()) } else { None }
                    });
                    let model = ep.operands.iter().find_map(|op| {
                        if let Operand::ExecutionModel(m) = op { Some(format!("{:?}", m)) } else { None }
                    });
                    eprintln!(
                        "[validate] Entry point: {} ({})",
                        name.unwrap_or("?"),
                        model.as_deref().unwrap_or("?")
                    );
                }
            }
        }
    }

    eprintln!("[validate] Loading {}", path.display());
    match load_spirv_module(&device, path) {
        Ok(_) => {
            eprintln!("[validate] OK — shader module created successfully");
            Ok(())
        }
        Err(e) => Err(e),
    }
}
