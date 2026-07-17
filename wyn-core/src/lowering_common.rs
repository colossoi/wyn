//! Common utilities shared between lowering backends (SPIR-V, WGSL).

use crate::ast::TypeName;
use polytype::Type as PolyType;

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
