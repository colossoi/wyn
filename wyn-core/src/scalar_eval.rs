//! Typed scalar evaluation shared by compile-time IR simplifiers.

use polytype::Type;

use crate::ast::TypeName;
use crate::op::{BinaryOperator, UnaryOperator};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Scalar {
    Int(i64),
    Float(f64),
    Bool(bool),
}

pub(crate) fn binary(
    op: BinaryOperator,
    lhs: Scalar,
    rhs: Scalar,
    operand_ty: &Type<TypeName>,
) -> Option<Scalar> {
    match (lhs, rhs) {
        (Scalar::Int(lhs), Scalar::Int(rhs)) => integer_binary(op, lhs, rhs, operand_ty),
        (Scalar::Float(lhs), Scalar::Float(rhs)) => float_binary(op, lhs, rhs, operand_ty),
        (Scalar::Bool(lhs), Scalar::Bool(rhs)) => bool_binary(op, lhs, rhs),
        _ => None,
    }
}

pub(crate) fn unary(op: UnaryOperator, value: Scalar, ty: &Type<TypeName>) -> Option<Scalar> {
    match (op, value) {
        (UnaryOperator::Negate, Scalar::Int(value)) => Some(Scalar::Int(wrap_int(-(value as i128), ty))),
        (UnaryOperator::Negate, Scalar::Float(value)) => Some(Scalar::Float(match ty {
            Type::Constructed(TypeName::Float(32), _) => (-(value as f32)) as f64,
            Type::Constructed(TypeName::Float(64), _) => -value,
            _ => return None,
        })),
        (UnaryOperator::LogicalNot, Scalar::Bool(value)) => Some(Scalar::Bool(!value)),
        _ => None,
    }
}

fn integer_binary(op: BinaryOperator, lhs: i64, rhs: i64, ty: &Type<TypeName>) -> Option<Scalar> {
    let (signed, bits) = integer_layout(ty)?;
    let lhs = wrap_int(lhs as i128, ty);
    let rhs = wrap_int(rhs as i128, ty);
    let arithmetic = |value| Some(Scalar::Int(wrap_int(value, ty)));
    match op {
        BinaryOperator::Add => arithmetic(lhs as i128 + rhs as i128),
        BinaryOperator::Subtract => arithmetic(lhs as i128 - rhs as i128),
        BinaryOperator::Multiply => arithmetic(lhs as i128 * rhs as i128),
        BinaryOperator::Divide | BinaryOperator::Remainder => {
            if signed {
                if rhs == 0 {
                    return None;
                }
                let value = if op == BinaryOperator::Divide {
                    lhs as i128 / rhs as i128
                } else {
                    lhs as i128 % rhs as i128
                };
                arithmetic(value)
            } else {
                let lhs = unsigned(lhs, bits);
                let rhs = unsigned(rhs, bits);
                if rhs == 0 {
                    return None;
                }
                let value = if op == BinaryOperator::Divide { lhs / rhs } else { lhs % rhs };
                arithmetic(value as i128)
            }
        }
        BinaryOperator::BitwiseAnd => arithmetic((lhs & rhs) as i128),
        BinaryOperator::BitwiseOr => arithmetic((lhs | rhs) as i128),
        BinaryOperator::BitwiseXor => arithmetic((lhs ^ rhs) as i128),
        BinaryOperator::Equal => Some(Scalar::Bool(lhs == rhs)),
        BinaryOperator::NotEqual => Some(Scalar::Bool(lhs != rhs)),
        BinaryOperator::Less
        | BinaryOperator::LessEqual
        | BinaryOperator::Greater
        | BinaryOperator::GreaterEqual => {
            let ordering =
                if signed { lhs.cmp(&rhs) } else { unsigned(lhs, bits).cmp(&unsigned(rhs, bits)) };
            Some(Scalar::Bool(match op {
                BinaryOperator::Less => ordering.is_lt(),
                BinaryOperator::LessEqual => ordering.is_le(),
                BinaryOperator::Greater => ordering.is_gt(),
                BinaryOperator::GreaterEqual => ordering.is_ge(),
                _ => unreachable!(),
            }))
        }
        _ => None,
    }
}

fn float_binary(op: BinaryOperator, lhs: f64, rhs: f64, ty: &Type<TypeName>) -> Option<Scalar> {
    let result = match op {
        BinaryOperator::Equal => return Some(Scalar::Bool(lhs == rhs)),
        BinaryOperator::NotEqual => {
            return Some(Scalar::Bool(!lhs.is_nan() && !rhs.is_nan() && lhs != rhs));
        }
        BinaryOperator::Less => return Some(Scalar::Bool(lhs < rhs)),
        BinaryOperator::LessEqual => return Some(Scalar::Bool(lhs <= rhs)),
        BinaryOperator::Greater => return Some(Scalar::Bool(lhs > rhs)),
        BinaryOperator::GreaterEqual => return Some(Scalar::Bool(lhs >= rhs)),
        _ => match ty {
            Type::Constructed(TypeName::Float(32), _) => {
                let (lhs, rhs) = (lhs as f32, rhs as f32);
                (match op {
                    BinaryOperator::Add => lhs + rhs,
                    BinaryOperator::Subtract => lhs - rhs,
                    BinaryOperator::Multiply => lhs * rhs,
                    BinaryOperator::Divide => lhs / rhs,
                    BinaryOperator::Remainder => lhs % rhs,
                    _ => return None,
                }) as f64
            }
            Type::Constructed(TypeName::Float(64), _) => match op {
                BinaryOperator::Add => lhs + rhs,
                BinaryOperator::Subtract => lhs - rhs,
                BinaryOperator::Multiply => lhs * rhs,
                BinaryOperator::Divide => lhs / rhs,
                BinaryOperator::Remainder => lhs % rhs,
                _ => return None,
            },
            _ => return None,
        },
    };
    Some(Scalar::Float(result))
}

fn bool_binary(op: BinaryOperator, lhs: bool, rhs: bool) -> Option<Scalar> {
    Some(Scalar::Bool(match op {
        BinaryOperator::Equal => lhs == rhs,
        BinaryOperator::NotEqual => lhs != rhs,
        BinaryOperator::LogicalAnd => lhs && rhs,
        BinaryOperator::LogicalOr => lhs || rhs,
        _ => return None,
    }))
}

fn integer_layout(ty: &Type<TypeName>) -> Option<(bool, usize)> {
    match ty {
        Type::Constructed(TypeName::Int(bits), _) => Some((true, *bits)),
        Type::Constructed(TypeName::UInt(bits), _) => Some((false, *bits)),
        _ => None,
    }
}

fn unsigned(value: i64, bits: usize) -> u128 {
    if bits >= 64 {
        value as u64 as u128
    } else {
        (value as u64 & ((1_u64 << bits) - 1)) as u128
    }
}

/// Wrap an integer result to the scalar type's two's-complement width.
pub(crate) fn wrap_int(value: i128, ty: &Type<TypeName>) -> i64 {
    match ty {
        Type::Constructed(TypeName::UInt(8), _) => (value as u8) as i64,
        Type::Constructed(TypeName::UInt(16), _) => (value as u16) as i64,
        Type::Constructed(TypeName::UInt(32), _) => (value as u32) as i64,
        Type::Constructed(TypeName::UInt(64), _) => (value as u64) as i64,
        Type::Constructed(TypeName::Int(8), _) => (value as i8) as i64,
        Type::Constructed(TypeName::Int(16), _) => (value as i16) as i64,
        Type::Constructed(TypeName::Int(32), _) => (value as i32) as i64,
        Type::Constructed(TypeName::Int(64), _) => value as i64,
        _ => value as i64,
    }
}
