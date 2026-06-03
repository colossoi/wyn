//! `ShaderModule` loading with passthrough-or-naga fallback.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use wgpu::ShaderModuleDescriptorPassthrough;

pub fn load_spirv_module(device: &wgpu::Device, path: &Path) -> Result<wgpu::ShaderModule> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;

    // Basic SPIR-V validation
    if bytes.len() < 20 {
        return Err(anyhow!("SPIR-V file too small ({} bytes)", bytes.len()));
    }
    if bytes.len() % 4 != 0 {
        return Err(anyhow!(
            "SPIR-V file size {} is not aligned to 4-byte words",
            bytes.len()
        ));
    }

    // Check magic number
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != 0x07230203 {
        return Err(anyhow!(
            "Invalid SPIR-V magic number: 0x{:08x} (expected 0x07230203)",
            magic
        ));
    }

    // Check if SPIRV_SHADER_PASSTHROUGH is supported
    if device.features().contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH) {
        // Convert bytes to u32 words for SPIR-V passthrough
        let mut spirv_data = Vec::new();
        for chunk in bytes.chunks_exact(4) {
            let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            spirv_data.push(word);
        }

        // Use create_shader_module_passthrough to bypass wgpu's SPIR-V validation
        // This allows loading SPIR-V with unsupported capabilities like Linkage
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        let shader_module = unsafe {
            device.create_shader_module_passthrough(ShaderModuleDescriptorPassthrough::SpirV(
                wgpu::ShaderModuleDescriptorSpirV {
                    label: Some(&format!("{}", path.display())),
                    source: std::borrow::Cow::Borrowed(&spirv_data),
                },
            ))
        };

        // Check for validation errors even with passthrough
        let error_option = pollster::block_on(device.pop_error_scope());
        if let Some(error) = error_option {
            eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
            eprintln!("║                  SHADER COMPILATION ERROR                    ║");
            eprintln!("╚══════════════════════════════════════════════════════════════╝");
            eprintln!("File: {}", path.display());
            eprintln!("Mode: SPIR-V passthrough");
            eprintln!();
            return Err(anyhow::Error::msg(format!(
                "Shader validation failed (passthrough): {}",
                error
            )));
        }

        Ok(shader_module)
    } else {
        // Fall back to regular shader module creation with validation
        let source = wgpu::util::make_spirv(&bytes);

        // Push an error scope to catch shader validation errors
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{}", path.display())),
            source,
        });

        // Check for validation errors
        let error_option = pollster::block_on(device.pop_error_scope());
        if let Some(error) = error_option {
            eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
            eprintln!("║                  SHADER COMPILATION ERROR                    ║");
            eprintln!("╚══════════════════════════════════════════════════════════════╝");
            eprintln!("File: {}", path.display());
            eprintln!("Mode: naga validation");
            eprintln!();
            return Err(anyhow::Error::msg(format!("Shader validation failed: {}", error)));
        }

        Ok(shader_module)
    }
}
