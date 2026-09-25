//! Shared select operand and type contract. Evaluating operand expressions is
//! separate from selecting between their already computed values.

use super::lowering::PrimOp;
use crate::types::{Type, TypeExt, TypeName};

pub(crate) const ADDED_WORK_LIMIT: usize = 4;

/// Cheap primitive operations admitted by both early and late if-conversion.
/// The caller must also prove speculation safety and account for vector width.
pub(crate) fn cheap_primop(prim: &PrimOp) -> bool {
    matches!(
        prim,
        PrimOp::Select
            | PrimOp::FAdd
            | PrimOp::FSub
            | PrimOp::FMul
            | PrimOp::IAdd
            | PrimOp::ISub
            | PrimOp::IMul
            | PrimOp::FOrdEqual
            | PrimOp::FOrdNotEqual
            | PrimOp::FOrdLessThan
            | PrimOp::FOrdGreaterThan
            | PrimOp::FOrdLessThanEqual
            | PrimOp::FOrdGreaterThanEqual
            | PrimOp::IEqual
            | PrimOp::INotEqual
            | PrimOp::SLessThan
            | PrimOp::ULessThan
            | PrimOp::SGreaterThan
            | PrimOp::UGreaterThan
            | PrimOp::SLessThanEqual
            | PrimOp::ULessThanEqual
            | PrimOp::SGreaterThanEqual
            | PrimOp::UGreaterThanEqual
            | PrimOp::BitwiseAnd
            | PrimOp::BitwiseOr
            | PrimOp::BitwiseXor
            | PrimOp::Not
            | PrimOp::Bitcast
            | PrimOp::SIToFP
            | PrimOp::UIToFP
            | PrimOp::SConvert
            | PrimOp::UConvert
            | PrimOp::FPConvert
    )
}

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
