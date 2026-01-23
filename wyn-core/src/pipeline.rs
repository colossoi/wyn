//! Compute pipeline representation.
//!
//! A Pipeline wraps one or more compute shader stages with their buffer bindings.
//! This is a separate IR level between MIR and SPIR-V for compute workloads.

use crate::ast::{StorageAccess, StorageLayout, TypeName};
use crate::mir::{self, BufferBlock, BufferField};
use polytype::Type;

/// A compute pipeline with buffer bindings and shader stages.
#[derive(Debug, Clone)]
pub struct Pipeline {
    /// Name of the pipeline (from the entry point).
    pub name: String,
    /// Buffer blocks with their bindings.
    pub buffers: Vec<BufferBlock>,
    /// Shader stages in execution order.
    pub stages: Vec<PipelineStage>,
}

/// A single shader stage in the pipeline.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Name of this stage.
    pub name: String,
    /// Workgroup size.
    pub local_size: (u32, u32, u32),
    /// The MIR body for this stage.
    pub body: mir::Body,
    /// Mapping from parameter names to buffer indices.
    pub param_buffers: Vec<(String, usize)>,
}

/// Build a Pipeline from a compute entry point.
pub fn build_pipeline(def: &mir::Def) -> Option<Pipeline> {
    let (name, execution_model, inputs, body) = match def {
        mir::Def::EntryPoint {
            name,
            execution_model,
            inputs,
            body,
            ..
        } => (name, execution_model, inputs, body),
        _ => return None,
    };

    let local_size = match execution_model {
        mir::ExecutionModel::Compute { local_size } => *local_size,
        _ => return None,
    };

    // Create buffer blocks for each slice-typed input
    let mut buffers = Vec::new();
    let mut param_buffers = Vec::new();
    let mut binding_num = 0u32;

    for input in inputs {
        if is_slice_type(&input.ty) {
            let elem_type = get_slice_element_type(&input.ty)?;
            let buffer = BufferBlock {
                id: crate::ast::NodeId(0), // TODO: proper id allocation
                name: input.name.clone(),
                layout: StorageLayout::Std430,
                set: 0,
                binding: binding_num,
                access: StorageAccess::ReadOnly,
                fields: vec![BufferField {
                    name: "data".to_string(),
                    ty: elem_type,
                    is_runtime_sized: true,
                }],
            };
            param_buffers.push((input.name.clone(), buffers.len()));
            buffers.push(buffer);
            binding_num += 1;
        }
    }

    // Check if the return type is a slice - if so, add an output buffer
    let return_type = body.get_type(body.root);
    if is_output_slice_type(return_type) {
        let output_elem_type = get_slice_element_type(return_type)?;
        let output_buffer = BufferBlock {
            id: crate::ast::NodeId(0),
            name: "_output".to_string(),
            layout: StorageLayout::Std430,
            set: 0,
            binding: binding_num,
            access: StorageAccess::WriteOnly,
            fields: vec![BufferField {
                name: "data".to_string(),
                ty: output_elem_type,
                is_runtime_sized: true,
            }],
        };
        buffers.push(output_buffer);
    }

    let stage = PipelineStage {
        name: name.clone(),
        local_size,
        body: body.clone(),
        param_buffers,
    };

    Some(Pipeline {
        name: name.clone(),
        buffers,
        stages: vec![stage],
    })
}

/// Check if a type is an output slice (unsized array that can be in any address space).
/// This is more permissive than is_slice_type since the return value doesn't need
/// an explicit Storage address space annotation.
fn is_output_slice_type(ty: &Type<TypeName>) -> bool {
    match ty {
        Type::Constructed(TypeName::Array, args) => {
            assert!(args.len() == 3);
            // Accept SizePlaceholder or type variable (polymorphic size = runtime determined)
            matches!(
                &args[2],
                Type::Constructed(TypeName::SizePlaceholder, _) | Type::Variable(_)
            )
        }
        _ => false,
    }
}

/// Check if a type is a storage slice (unsized array in storage address space).
/// Array[elem, addrspace, size] where addrspace is Storage and size is SizePlaceholder or a type variable.
fn is_slice_type(ty: &Type<TypeName>) -> bool {
    match ty {
        Type::Constructed(TypeName::Array, args) => {
            assert!(args.len() == 3);
            let is_storage = matches!(&args[1], Type::Constructed(TypeName::AddressStorage, _));
            // Accept SizePlaceholder or type variable (polymorphic size = runtime determined)
            let is_unsized = matches!(
                &args[2],
                Type::Constructed(TypeName::SizePlaceholder, _) | Type::Variable(_)
            );
            is_storage && is_unsized
        }
        _ => false,
    }
}

/// Get the element type of an array.
/// Array[elem, addrspace, size] - element is at args[0].
fn get_slice_element_type(ty: &Type<TypeName>) -> Option<Type<TypeName>> {
    match ty {
        Type::Constructed(TypeName::Array, args) => {
            assert!(args.len() == 3);
            Some(args[0].clone())
        }
        _ => None,
    }
}
