//! SPIR-V parsing helpers: entry-point detection, descriptor access
//! decoration scraping, auto-detect helpers, and ShaderModule loading
//! (with passthrough-or-naga fallback).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rspirv::binary::parse_words;
use rspirv::dr::{Loader, Operand};
use rspirv::spirv::{Decoration, ExecutionModel};
use wgpu::ShaderModuleDescriptorPassthrough;

pub fn detect_entry_points(spirv_words: &[u32]) -> Result<Vec<(String, ExecutionModel)>> {
    let mut loader = Loader::new();
    parse_words(spirv_words, &mut loader).map_err(|e| anyhow!("Failed to parse SPIR-V: {:?}", e))?;
    let module = loader.module();

    let mut entry_points = Vec::new();

    // module.entry_points contains OpEntryPoint instructions
    // Operands: [0] ExecutionModel, [1] IdRef (function), [2] LiteralString (name), [3..] interfaces
    for instruction in &module.entry_points {
        if instruction.operands.len() < 3 {
            continue;
        }

        let execution_model = match &instruction.operands[0] {
            Operand::ExecutionModel(model) => *model,
            _ => continue,
        };

        let name = match &instruction.operands[2] {
            Operand::LiteralString(s) => s.clone(),
            _ => continue,
        };

        entry_points.push((name, execution_model));
    }

    Ok(entry_points)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpirvAccess {
    ReadOnly,
    ReadWrite,
    WriteOnly,
}

/// Walk the SPIR-V annotations and produce a `(set, binding) → access`
/// map for every `OpVariable` decorated with both `DescriptorSet` and
/// `Binding`. The pipeline layout this drives must match the shader's
/// declared access exactly — naga rejects pipelines that grant more
/// access than the shader permits (e.g. `LOAD|STORE` pipeline against
/// a `NonWritable` shader binding).
pub fn detect_storage_access(spirv_words: &[u32]) -> Result<HashMap<(u32, u32), SpirvAccess>> {
    let mut loader = Loader::new();
    parse_words(spirv_words, &mut loader).map_err(|e| anyhow!("Failed to parse SPIR-V: {:?}", e))?;
    let module = loader.module();

    // First pass: per-id, collect (DescriptorSet, Binding, NonWritable, NonReadable).
    #[derive(Default)]
    struct Decor {
        set: Option<u32>,
        binding: Option<u32>,
        non_writable: bool,
        non_readable: bool,
    }
    let mut by_id: HashMap<u32, Decor> = HashMap::new();
    for ann in &module.annotations {
        // OpDecorate operands: [0] IdRef target, [1] Decoration, [2..] literals.
        if ann.operands.len() < 2 {
            continue;
        }
        let id = match ann.operands[0] {
            Operand::IdRef(id) => id,
            _ => continue,
        };
        let dec = match ann.operands[1] {
            Operand::Decoration(d) => d,
            _ => continue,
        };
        let entry = by_id.entry(id).or_default();
        match dec {
            Decoration::DescriptorSet => {
                if let Some(Operand::LiteralBit32(n)) = ann.operands.get(2) {
                    entry.set = Some(*n);
                }
            }
            Decoration::Binding => {
                if let Some(Operand::LiteralBit32(n)) = ann.operands.get(2) {
                    entry.binding = Some(*n);
                }
            }
            Decoration::NonWritable => entry.non_writable = true,
            Decoration::NonReadable => entry.non_readable = true,
            _ => {}
        }
    }

    let mut out: HashMap<(u32, u32), SpirvAccess> = HashMap::new();
    for d in by_id.values() {
        let (Some(set), Some(binding)) = (d.set, d.binding) else {
            continue;
        };
        let access = match (d.non_writable, d.non_readable) {
            (true, false) => SpirvAccess::ReadOnly,
            (false, true) => SpirvAccess::WriteOnly,
            _ => SpirvAccess::ReadWrite,
        };
        // If the same (set, binding) is decorated by multiple `OpVariable`s
        // (e.g. uniform and storage view of the same slot — shouldn't
        // happen with the new descriptor-set convention, but be safe),
        // narrow to the more permissive access.
        out.entry((set, binding))
            .and_modify(|existing| {
                if access == SpirvAccess::ReadWrite {
                    *existing = SpirvAccess::ReadWrite;
                }
            })
            .or_insert(access);
    }
    Ok(out)
}

/// Resolve vertex and fragment entry point names.
/// If both provided, use them. If neither, auto-detect. If only one, error.
pub fn resolve_entry_points(
    path: &Path,
    vertex_arg: Option<String>,
    fragment_arg: Option<String>,
) -> Result<(String, String)> {
    match (vertex_arg, fragment_arg) {
        (Some(v), Some(f)) => Ok((v, f)),
        (None, None) => auto_detect_entry_points(path),
        (Some(_), None) => Err(anyhow!(
            "--vertex was provided but --fragment was not. Provide both or neither for auto-detection."
        )),
        (None, Some(_)) => Err(anyhow!(
            "--fragment was provided but --vertex was not. Provide both or neither for auto-detection."
        )),
    }
}

/// Auto-detect entry points from a SPIR-V file.
/// Succeeds if exactly one Vertex and one Fragment entry point exist (and no others).
pub fn auto_detect_entry_points(path: &Path) -> Result<(String, String)> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;

    if bytes.len() % 4 != 0 {
        return Err(anyhow!("SPIR-V file size is not aligned to 4-byte words"));
    }

    let spirv_words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let entry_points = detect_entry_points(&spirv_words)?;

    let vertex_entries: Vec<_> =
        entry_points.iter().filter(|(_, m)| *m == ExecutionModel::Vertex).collect();

    let fragment_entries: Vec<_> =
        entry_points.iter().filter(|(_, m)| *m == ExecutionModel::Fragment).collect();

    let other_entries: Vec<_> = entry_points
        .iter()
        .filter(|(_, m)| *m != ExecutionModel::Vertex && *m != ExecutionModel::Fragment)
        .collect();

    match (vertex_entries.len(), fragment_entries.len(), other_entries.len()) {
        (1, 1, 0) => {
            let vertex_name = vertex_entries[0].0.clone();
            let fragment_name = fragment_entries[0].0.clone();
            eprintln!(
                "Auto-detected entry points: vertex='{}', fragment='{}'",
                vertex_name, fragment_name
            );
            Ok((vertex_name, fragment_name))
        }
        _ => {
            let mut msg = String::from("Cannot auto-detect entry points.\n\nFound entry points:\n");

            if !vertex_entries.is_empty() {
                msg.push_str("  Vertex:\n");
                for (name, _) in &vertex_entries {
                    msg.push_str(&format!("    - {}\n", name));
                }
            }

            if !fragment_entries.is_empty() {
                msg.push_str("  Fragment:\n");
                for (name, _) in &fragment_entries {
                    msg.push_str(&format!("    - {}\n", name));
                }
            }

            if !other_entries.is_empty() {
                msg.push_str("  Other:\n");
                for (name, model) in &other_entries {
                    msg.push_str(&format!("    - {} ({:?})\n", name, model));
                }
            }

            if entry_points.is_empty() {
                msg.push_str("  (none found)\n");
            }

            msg.push_str("\nExpected exactly 1 Vertex and 1 Fragment entry point.\n");
            msg.push_str("Use --vertex and --fragment to specify entry points manually.");

            Err(anyhow!(msg))
        }
    }
}

/// Auto-detect a compute entry point from a SPIR-V file.
/// Succeeds if exactly one GLCompute entry point exists.
pub fn auto_detect_compute_entry_point(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;

    if bytes.len() % 4 != 0 {
        return Err(anyhow!("SPIR-V file size is not aligned to 4-byte words"));
    }

    let spirv_words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let entry_points = detect_entry_points(&spirv_words)?;

    let compute_entries: Vec<_> =
        entry_points.iter().filter(|(_, m)| *m == ExecutionModel::GLCompute).collect();

    match compute_entries.len() {
        1 => {
            let name = compute_entries[0].0.clone();
            eprintln!("Auto-detected compute entry point: '{}'", name);
            Ok(name)
        }
        0 => Err(anyhow!("No GLCompute entry point found in SPIR-V module")),
        _ => {
            let mut msg = String::from("Multiple compute entry points found:\n");
            for (name, _) in &compute_entries {
                msg.push_str(&format!("  - {}\n", name));
            }
            msg.push_str("Use --entry to specify which one to run.");
            Err(anyhow!(msg))
        }
    }
}

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
