//! Shared select operand and type contract. Evaluating operand expressions is
//! separate from selecting between their already computed values.

use crate::types::{Type, TypeExt, TypeName};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Selection<T> {
    pub condition: T,
    pub yes: T,
    pub no: T,
}

impl<T: Copy> Selection<T> {
    pub fn from_operands(operands: &[T]) -> Option<Self> {
        let &[no, yes, condition] = operands else {
            return None;
        };
        Some(Self { condition, yes, no })
    }
}

/// Common native select types for both shader targets. Wider and emulated
/// values require explicit legalization before being admitted here.
pub(crate) fn supported_type(ty: &Type) -> bool {
    match ty {
        Type::Constructed(
            TypeName::Bool | TypeName::Int(32) | TypeName::UInt(32) | TypeName::Float(16 | 32),
            args,
        ) => args.is_empty(),
        Type::Constructed(TypeName::Vec, _) => {
            ty.vec_size().is_some_and(|n| (2..=4).contains(&n))
                && ty.elem_type().is_some_and(|element| !element.is_vec() && supported_type(element))
        }
        _ => false,
    }
}
