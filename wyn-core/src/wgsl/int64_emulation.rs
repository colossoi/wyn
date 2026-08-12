//! Optional unsigned 64-bit integer legalization for WGSL.
//!
//! The typed compiler IR continues to carry scalar `u64` values. This module
//! owns the backend-only representation and operations used when
//! [`WgslInt64Mode::EmulateU64`](super::WgslInt64Mode::EmulateU64) is selected.

use std::collections::BTreeSet;

use polytype::Type;

use crate::ast::TypeName;
use crate::op::BinaryOperator;

pub(crate) const WGSL_U64_TYPE: &str = "vec2<u32>";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Helper {
    Add,
    Subtract,
    ShiftLeft,
    ShiftRight,
}

#[derive(Default)]
pub(crate) struct U64Emulation {
    helpers: BTreeSet<Helper>,
}

impl U64Emulation {
    /// Emit a branch-free shift when the source count is known. This is the
    /// hot path for cryptographic rotates and avoids relying on driver inlining
    /// and constant propagation through the general helper functions.
    pub(crate) fn lower_constant_shift(
        &self,
        op: BinaryOperator,
        value: &str,
        count: u32,
    ) -> Option<String> {
        let shift = count & 63;
        let expression = match op {
            BinaryOperator::ShiftLeft if shift == 0 => value.to_string(),
            BinaryOperator::ShiftLeft if shift < 32 => format!(
                "vec2<u32>(({value}).x << {shift}u, (({value}).y << {shift}u) | (({value}).x >> {}u))",
                32 - shift
            ),
            BinaryOperator::ShiftLeft => format!("vec2<u32>(0u, ({value}).x << {}u)", shift - 32),
            BinaryOperator::ShiftRight | BinaryOperator::ShiftRightLogical if shift == 0 => {
                value.to_string()
            }
            BinaryOperator::ShiftRight | BinaryOperator::ShiftRightLogical if shift < 32 => {
                format!(
                    "vec2<u32>((({value}).x >> {shift}u) | (({value}).y << {}u), ({value}).y >> {shift}u)",
                    32 - shift
                )
            }
            BinaryOperator::ShiftRight | BinaryOperator::ShiftRightLogical => {
                format!("vec2<u32>(({value}).y >> {}u, 0u)", shift - 32)
            }
            _ => return None,
        };
        Some(expression)
    }

    pub(crate) fn lower_binary(
        &mut self,
        op: BinaryOperator,
        lhs: &str,
        rhs: &str,
    ) -> Result<String, String> {
        let expression = match op {
            BinaryOperator::Add => {
                self.helpers.insert(Helper::Add);
                format!("_wyn_u64_add({lhs}, {rhs})")
            }
            BinaryOperator::Subtract => {
                self.helpers.insert(Helper::Subtract);
                format!("_wyn_u64_sub({lhs}, {rhs})")
            }
            BinaryOperator::BitwiseAnd | BinaryOperator::BitwiseOr | BinaryOperator::BitwiseXor => {
                format!("({lhs} {} {rhs})", op.symbol())
            }
            BinaryOperator::ShiftLeft => {
                self.helpers.insert(Helper::ShiftLeft);
                format!("_wyn_u64_shl({lhs}, ({rhs}).x)")
            }
            BinaryOperator::ShiftRight | BinaryOperator::ShiftRightLogical => {
                self.helpers.insert(Helper::ShiftRight);
                format!("_wyn_u64_shr({lhs}, ({rhs}).x)")
            }
            BinaryOperator::Equal => format!("all({lhs} == {rhs})"),
            BinaryOperator::NotEqual => format!("any({lhs} != {rhs})"),
            BinaryOperator::Less => unsigned_less(lhs, rhs),
            BinaryOperator::LessEqual => format!("!{}", unsigned_less(rhs, lhs)),
            BinaryOperator::Greater => unsigned_less(rhs, lhs),
            BinaryOperator::GreaterEqual => format!("!{}", unsigned_less(lhs, rhs)),
            BinaryOperator::Multiply
            | BinaryOperator::Divide
            | BinaryOperator::Remainder
            | BinaryOperator::FloorDivide
            | BinaryOperator::FloorRemainder
            | BinaryOperator::Power
            | BinaryOperator::LogicalAnd
            | BinaryOperator::LogicalOr => {
                return Err(format!(
                    "u64 operator '{}' is not supported by WGSL u64 emulation",
                    op.symbol()
                ));
            }
        };
        Ok(expression)
    }

    pub(crate) fn emit_helpers(&self, output: &mut String) {
        for helper in &self.helpers {
            match helper {
                Helper::Add => output.push_str(ADD_HELPER),
                Helper::Subtract => output.push_str(SUBTRACT_HELPER),
                Helper::ShiftLeft => output.push_str(SHIFT_LEFT_HELPER),
                Helper::ShiftRight => output.push_str(SHIFT_RIGHT_HELPER),
            }
            output.push('\n');
        }
    }
}

pub(crate) fn is_u64(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::UInt(64), _))
}

/// Split the decimal spelling carried by TLC/EGIR into little-endian lanes.
///
/// Constant folding stores integer bit patterns in an `i64`, so high-bit u64
/// values can arrive either as their positive source spelling or as a negative
/// signed residual. Both spellings must reproduce the same 64 bits.
pub(crate) fn lower_literal(value: &str) -> Result<String, String> {
    let bits = match value.parse::<u64>() {
        Ok(bits) => bits,
        Err(unsigned_error) => {
            value.parse::<i64>().map(|signed| signed as u64).map_err(|signed_error| {
                format!(
                "invalid u64 literal '{value}' ({unsigned_error}; signed residual parse: {signed_error})"
            )
            })?
        }
    };
    let low = bits as u32;
    let high = (bits >> 32) as u32;
    Ok(format!("vec2<u32>({low}u, {high}u)"))
}

/// Lower a conversion involving an emulated u64. `None` means neither side is
/// u64 and ordinary WGSL conversion lowering should continue.
pub(crate) fn lower_conversion(
    source: &Type<TypeName>,
    target: &Type<TypeName>,
    argument: &str,
) -> Option<Result<String, String>> {
    if !is_u64(source) && !is_u64(target) {
        return None;
    }

    Some(match (source, target) {
        (source, target) if is_u64(source) && is_u64(target) => Ok(argument.to_string()),
        (Type::Constructed(TypeName::UInt(32), _), target) if is_u64(target) => {
            Ok(format!("vec2<u32>({argument}, 0u)"))
        }
        (Type::Constructed(TypeName::Int(32), _), target) if is_u64(target) => {
            Ok(format!("vec2<u32>(bitcast<u32>({argument}), 0u)"))
        }
        (source, Type::Constructed(TypeName::UInt(32), _)) if is_u64(source) => {
            Ok(format!("({argument}).x"))
        }
        (source, Type::Constructed(TypeName::Int(32), _)) if is_u64(source) => {
            Ok(format!("bitcast<i32>(({argument}).x)"))
        }
        _ => Err(format!(
            "conversion from {source:?} to {target:?} is not supported by WGSL u64 emulation"
        )),
    })
}

fn unsigned_less(lhs: &str, rhs: &str) -> String {
    format!("((({lhs}).y < ({rhs}).y) || ((({lhs}).y == ({rhs}).y) && (({lhs}).x < ({rhs}).x)))")
}

const ADD_HELPER: &str = r#"fn _wyn_u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let low = a.x + b.x;
    let carry = select(0u, 1u, low < a.x);
    return vec2<u32>(low, a.y + b.y + carry);
}
"#;

const SUBTRACT_HELPER: &str = r#"fn _wyn_u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let borrow = select(0u, 1u, a.x < b.x);
    return vec2<u32>(a.x - b.x, a.y - b.y - borrow);
}
"#;

const SHIFT_LEFT_HELPER: &str = r#"fn _wyn_u64_shl(value: vec2<u32>, count: u32) -> vec2<u32> {
    let shift = count & 63u;
    if (shift == 0u) {
        return value;
    }
    if (shift < 32u) {
        return vec2<u32>(
            value.x << shift,
            (value.y << shift) | (value.x >> (32u - shift)));
    }
    return vec2<u32>(0u, value.x << (shift - 32u));
}
"#;

const SHIFT_RIGHT_HELPER: &str = r#"fn _wyn_u64_shr(value: vec2<u32>, count: u32) -> vec2<u32> {
    let shift = count & 63u;
    if (shift == 0u) {
        return value;
    }
    if (shift < 32u) {
        return vec2<u32>(
            (value.x >> shift) | (value.y << (32u - shift)),
            value.y >> shift);
    }
    return vec2<u32>(value.y >> (shift - 32u), 0u);
}
"#;

#[cfg(test)]
#[path = "int64_emulation_tests.rs"]
mod int64_emulation_tests;
