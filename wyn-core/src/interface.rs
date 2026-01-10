//! Shader interface extraction for JSON output.
//!
//! Extracts binding information, entry points, and buffer layouts from a compiled program.

use serde::Serialize;

use crate::ast::{StorageAccess, StorageLayout};
use crate::mir::{BufferBlock, Def, ExecutionModel, Program};
use crate::pipeline;

/// Complete shader interface description.
#[derive(Debug, Clone, Serialize)]
pub struct ShaderInterface {
    /// Interface format version.
    pub version: String,
    /// Entry points in the shader.
    pub entry_points: Vec<EntryPointInfo>,
    /// Buffer bindings.
    pub buffers: Vec<BufferBindingInfo>,
}

/// Information about a shader entry point.
#[derive(Debug, Clone, Serialize)]
pub struct EntryPointInfo {
    /// Entry point name.
    pub name: String,
    /// Execution model (vertex, fragment, compute).
    pub execution_model: String,
    /// Workgroup size for compute shaders.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workgroup_size: Option<[u32; 3]>,
}

/// Information about a buffer binding.
#[derive(Debug, Clone, Serialize)]
pub struct BufferBindingInfo {
    /// Buffer name.
    pub name: String,
    /// Descriptor set number.
    pub set: u32,
    /// Binding number within the set.
    pub binding: u32,
    /// Buffer type (storage, uniform).
    pub buffer_type: String,
    /// Memory layout (std430, std140).
    pub layout: String,
    /// Access mode (readonly, writeonly, readwrite).
    pub access: String,
    /// Fields in the buffer.
    pub fields: Vec<BufferFieldInfo>,
}

/// Information about a field within a buffer.
#[derive(Debug, Clone, Serialize)]
pub struct BufferFieldInfo {
    /// Field name.
    pub name: String,
    /// Type description.
    pub type_name: String,
    /// Byte offset within the buffer.
    pub offset: u32,
}

/// Extract shader interface from a compiled MIR program.
pub fn extract_interface(program: &Program) -> ShaderInterface {
    let mut entry_points = Vec::new();
    let mut buffers = Vec::new();

    for def in &program.defs {
        match def {
            Def::EntryPoint {
                name,
                execution_model,
                ..
            } => {
                let (model_str, workgroup_size) = match execution_model {
                    ExecutionModel::Vertex => ("vertex".to_string(), None),
                    ExecutionModel::Fragment => ("fragment".to_string(), None),
                    ExecutionModel::Compute { local_size } => (
                        "compute".to_string(),
                        Some([local_size.0, local_size.1, local_size.2]),
                    ),
                };

                entry_points.push(EntryPointInfo {
                    name: name.clone(),
                    execution_model: model_str,
                    workgroup_size,
                });

                // For compute shaders, extract buffer bindings from pipeline
                if matches!(execution_model, ExecutionModel::Compute { .. }) {
                    if let Some(pipeline) = pipeline::build_pipeline(def) {
                        for buffer in pipeline.buffers {
                            buffers.push(buffer_block_to_info(&buffer));
                        }
                    }
                }
            }
            Def::Storage {
                name,
                set,
                binding,
                layout,
                access,
                ty,
                ..
            } => {
                // Legacy storage definitions
                buffers.push(BufferBindingInfo {
                    name: name.clone(),
                    set: *set,
                    binding: *binding,
                    buffer_type: "storage".to_string(),
                    layout: layout_to_string(layout),
                    access: access_to_string(access),
                    fields: vec![BufferFieldInfo {
                        name: "data".to_string(),
                        type_name: format_type(ty),
                        offset: 0,
                    }],
                });
            }
            _ => {}
        }
    }

    ShaderInterface {
        version: "0.1.0".to_string(),
        entry_points,
        buffers,
    }
}

fn buffer_block_to_info(block: &BufferBlock) -> BufferBindingInfo {
    let fields = block
        .fields
        .iter()
        .enumerate()
        .map(|(i, field)| {
            // TODO: Calculate actual offsets using layout rules
            let offset = i as u32 * 4; // Placeholder
            BufferFieldInfo {
                name: field.name.clone(),
                type_name: format_type(&field.ty),
                offset,
            }
        })
        .collect();

    BufferBindingInfo {
        name: block.name.clone(),
        set: block.set,
        binding: block.binding,
        buffer_type: "storage".to_string(),
        layout: layout_to_string(&block.layout),
        access: access_to_string(&block.access),
        fields,
    }
}

fn layout_to_string(layout: &StorageLayout) -> String {
    match layout {
        StorageLayout::Std430 => "std430".to_string(),
        StorageLayout::Std140 => "std140".to_string(),
    }
}

fn access_to_string(access: &StorageAccess) -> String {
    match access {
        StorageAccess::ReadOnly => "readonly".to_string(),
        StorageAccess::WriteOnly => "writeonly".to_string(),
        StorageAccess::ReadWrite => "readwrite".to_string(),
    }
}

fn format_type(ty: &polytype::Type<crate::ast::TypeName>) -> String {
    use crate::ast::TypeName;
    use polytype::Type;

    match ty {
        Type::Constructed(TypeName::Int(32), _) => "i32".to_string(),
        Type::Constructed(TypeName::UInt(32), _) => "u32".to_string(),
        Type::Constructed(TypeName::Float(32), _) => "f32".to_string(),
        Type::Constructed(TypeName::Float(64), _) => "f64".to_string(),
        Type::Constructed(TypeName::Vec, args) if args.len() >= 2 => {
            if let Type::Constructed(TypeName::Size(n), _) = &args[0] {
                let elem = format_type(&args[1]);
                format!("vec{}{}", n, elem)
            } else {
                format!("{:?}", ty)
            }
        }
        Type::Constructed(TypeName::Mat, args) if args.len() >= 3 => {
            if let (
                Type::Constructed(TypeName::Size(cols), _),
                Type::Constructed(TypeName::Size(rows), _),
            ) = (&args[0], &args[1])
            {
                let elem = format_type(&args[2]);
                format!("mat{}x{}{}", cols, rows, elem)
            } else {
                format!("{:?}", ty)
            }
        }
        Type::Constructed(TypeName::Array, args) => {
            assert!(args.len() == 3);
            let elem = format_type(&args[0]);
            match &args[2] {
                Type::Constructed(TypeName::Size(n), _) => format!("[{}]{}", n, elem),
                Type::Constructed(TypeName::Unsized, _) => format!("[]{}", elem),
                _ => format!("[?]{}", elem),
            }
        }
        _ => format!("{:?}", ty),
    }
}

/// Serialize interface to JSON string.
pub fn to_json(interface: &ShaderInterface) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(interface)
}
