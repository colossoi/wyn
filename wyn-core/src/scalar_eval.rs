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
        BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight | BinaryOperator::ShiftRightLogical => {
            // Only fold counts for which WGSL and SPIR-V agree. Out-of-range
            // counts have target-specific behavior and must remain residual.
            if rhs < 0 || rhs as usize >= bits {
                return None;
            }
            match op {
                BinaryOperator::ShiftLeft => arithmetic((unsigned(lhs, bits) << rhs) as i128),
                BinaryOperator::ShiftRight if signed => arithmetic((lhs >> rhs) as i128),
                _ => arithmetic((unsigned(lhs, bits) >> rhs) as i128),
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_arithmetic_wraps_at_each_declared_width() {
        for bits in [8, 16, 32, 64] {
            let signed = Type::Constructed(TypeName::Int(bits), vec![]);
            let unsigned = Type::Constructed(TypeName::UInt(bits), vec![]);
            let min = (-(1_i128 << (bits - 1))) as i64;
            let max = ((1_i128 << (bits - 1)) - 1) as i64;
            let mask = ((1_u128 << bits) - 1) as i64;
            for (ty, op, lhs, rhs, expected) in [
                (&signed, BinaryOperator::Add, max, 1, min),
                (&signed, BinaryOperator::Subtract, min, 1, max),
                (&signed, BinaryOperator::Multiply, min, -1, min),
                (&unsigned, BinaryOperator::Add, mask, 1, 0),
                (&unsigned, BinaryOperator::Subtract, 0, 1, mask),
                (&unsigned, BinaryOperator::Multiply, mask, mask, 1),
            ] {
                assert_eq!(
                    binary(op, Scalar::Int(lhs), Scalar::Int(rhs), ty),
                    Some(Scalar::Int(expected)),
                    "{ty:?}: {lhs} {op:?} {rhs}"
                );
            }
        }
    }

    #[test]
    fn integer_shifts_respect_signedness_and_width() {
        for bits in [8, 16, 32, 64] {
            let signed = Type::Constructed(TypeName::Int(bits), vec![]);
            let unsigned = Type::Constructed(TypeName::UInt(bits), vec![]);
            let high_bit = (1_u64 << (bits - 1)) as i64;
            let mask = ((1_u128 << bits) - 1) as i64;
            for (ty, op, lhs, rhs, expected) in [
                (&unsigned, BinaryOperator::ShiftLeft, 1, 0, 1),
                (&unsigned, BinaryOperator::ShiftLeft, 1, bits as i64 - 1, high_bit),
                (&unsigned, BinaryOperator::ShiftLeft, high_bit, 1, 0),
                (&unsigned, BinaryOperator::ShiftRight, mask, bits as i64 - 1, 1),
                (&signed, BinaryOperator::ShiftRight, -2, 1, -1),
                (&signed, BinaryOperator::ShiftRightLogical, -1, bits as i64 - 1, 1),
            ] {
                assert_eq!(
                    binary(op, Scalar::Int(lhs), Scalar::Int(rhs), ty),
                    Some(Scalar::Int(expected)),
                    "{ty:?}: {lhs} {op:?} {rhs}"
                );
            }
            for ty in [&signed, &unsigned] {
                for op in [
                    BinaryOperator::ShiftLeft,
                    BinaryOperator::ShiftRight,
                    BinaryOperator::ShiftRightLogical,
                ] {
                    for count in [-1, bits as i64, bits as i64 + 1] {
                        assert_eq!(binary(op, Scalar::Int(1), Scalar::Int(count), ty), None);
                    }
                }
            }
        }
    }
}
