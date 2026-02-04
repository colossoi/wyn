//! Common utilities shared between lowering backends (SPIR-V, GLSL, etc.)

use crate::ast::TypeName;
use crate::tlc::to_ssa::ExecutionModel;
use polytype::Type as PolyType;

/// Shader stage for entry points
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute {
        local_size: (u32, u32, u32),
    },
}

impl From<ExecutionModel> for ShaderStage {
    fn from(model: ExecutionModel) -> Self {
        match model {
            ExecutionModel::Vertex => ShaderStage::Vertex,
            ExecutionModel::Fragment => ShaderStage::Fragment,
            ExecutionModel::Compute { local_size } => ShaderStage::Compute { local_size },
        }
    }
}

impl From<&ExecutionModel> for ShaderStage {
    fn from(model: &ExecutionModel) -> Self {
        match model {
            ExecutionModel::Vertex => ShaderStage::Vertex,
            ExecutionModel::Fragment => ShaderStage::Fragment,
            ExecutionModel::Compute { local_size } => ShaderStage::Compute {
                local_size: *local_size,
            },
        }
    }
}

/// Check if a type represents an empty closure (no captured variables)
/// With the new closure format (_w_lambda_name, captures), the captures tuple
/// is what gets passed to the lambda function. Empty captures = empty tuple or unit.
/// Note: types::tuple([]) returns Unit, not Tuple(0, []), so we must check both.
pub fn is_empty_closure_type(ty: &PolyType<TypeName>) -> bool {
    match ty {
        // Empty tuple with no args
        PolyType::Constructed(TypeName::Tuple(_), args) => args.is_empty(),
        // Unit type (what types::tuple([]) returns)
        PolyType::Constructed(TypeName::Unit, _) => true,
        _ => false,
    }
}
