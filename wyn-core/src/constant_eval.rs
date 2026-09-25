//! Constant evaluation for catalog builtins. Vector lifting is deliberately
//! limited to componentwise operations; reductions and geometry stay residual.

use crate::builtins::lowering::{BuiltinLowering, PrimOp};
use crate::op::BinaryOperator;
use crate::scalar_eval::{self, wrap_int, Scalar};
use crate::types::{bool_type, Type, TypeExt, TypeName};
use spirv::GLOp;

/// Scalar or small componentwise vector value, independent of any compiler IR.
#[derive(Debug, Clone)]
pub(crate) enum Constant {
    Int(i64),
    Float(f64),
    Bool(bool),
    Vector(Vec<Scalar>),
}
impl Constant {
    pub(crate) fn as_scalar(&self) -> Option<Scalar> {
        match self {
            Self::Int(v) => Some(Scalar::Int(*v)),
            Self::Float(v) => Some(Scalar::Float(*v)),
            Self::Bool(v) => Some(Scalar::Bool(*v)),
            Self::Vector(_) => None,
        }
    }
    pub(crate) fn from_scalar(value: Scalar) -> Self {
        match value {
            Scalar::Int(v) => Self::Int(v),
            Scalar::Float(v) => Self::Float(v),
            Scalar::Bool(v) => Self::Bool(v),
        }
    }
}
/// Evaluate in source order with rounding at each operation. Vector comparisons
/// and reductions are deliberately excluded from this componentwise path.
pub(crate) fn binary(
    op: BinaryOperator,
    lhs: &Constant,
    rhs: &Constant,
    lhs_ty: &Type,
) -> Option<Constant> {
    if let (Some(lhs), Some(rhs)) = (lhs.as_scalar(), rhs.as_scalar()) {
        return scalar_eval::binary(op, lhs, rhs, lhs_ty).map(Constant::from_scalar);
    }
    if !matches!(
        op,
        BinaryOperator::Add
            | BinaryOperator::Subtract
            | BinaryOperator::Multiply
            | BinaryOperator::Divide
            | BinaryOperator::Remainder
    ) {
        return None;
    }
    let width = match (lhs, rhs) {
        (Constant::Vector(a), Constant::Vector(b)) if a.len() == b.len() => a.len(),
        (Constant::Vector(a), b) if b.as_scalar().is_some() => a.len(),
        (a, Constant::Vector(b)) if a.as_scalar().is_some() => b.len(),
        _ => return None,
    };
    if !(2..=4).contains(&width) {
        return None;
    }
    let ty = if lhs_ty.is_vec() { lhs_ty.elem_type()? } else { lhs_ty };
    let lane = |v: &Constant, i: usize| match v {
        Constant::Vector(values) => values.get(i).copied(),
        v => v.as_scalar(),
    };
    (0..width)
        .map(|i| scalar_eval::binary(op, lane(lhs, i)?, lane(rhs, i)?, ty))
        .collect::<Option<Vec<_>>>()
        .map(Constant::Vector)
}

pub(crate) fn builtin(
    lowering: &BuiltinLowering,
    args: &[(Constant, Type)],
    result_ty: &Type,
) -> Option<Constant> {
    if matches!(lowering, BuiltinLowering::PrimOp(PrimOp::Select)) {
        let [(no, no_ty), (yes, yes_ty), (Constant::Bool(condition), condition_ty)] = args else {
            return None;
        };
        if no_ty != result_ty || yes_ty != result_ty || *condition_ty != bool_type() {
            return None;
        }
        return Some(if *condition { yes } else { no }.clone());
    }
    let (prim, splat_args) = match lowering {
        BuiltinLowering::PrimOp(prim) => (prim.clone(), &[][..]),
        BuiltinLowering::ExtInstSplat { ext, splat_args } => (PrimOp::GlslExt(*ext), *splat_args),
        _ => return None,
    };
    if !result_ty.is_vec() {
        return fold_scalar(&prim, args, result_ty);
    }

    let width = result_ty.vec_size()?;
    if !(2..=4).contains(&width) {
        return None;
    }
    let element_ty = result_ty.elem_type()?;
    let mut components = Vec::with_capacity(width);
    for lane in 0..width {
        let lane_args = args
            .iter()
            .enumerate()
            .map(|(index, (value, ty))| match value {
                Constant::Vector(values) if values.len() == width && ty.vec_size() == Some(width) => {
                    Some((Constant::from_scalar(values[lane]), ty.elem_type()?.clone()))
                }
                value if splat_args.contains(&index) && value.as_scalar().is_some() => {
                    Some((value.clone(), ty.clone()))
                }
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        components.push(fold_scalar(&prim, &lane_args, element_ty)?.as_scalar()?);
    }
    // Preserve the whole call if any lane is unsupported or outside its domain.
    Some(Constant::Vector(components))
}

fn fold_scalar(prim: &PrimOp, args: &[(Constant, Type)], result_ty: &Type) -> Option<Constant> {
    if args.iter().any(|(value, _)| value.as_scalar().is_none()) {
        return None;
    }
    match prim {
        PrimOp::GlslExt(op) => fold_glsl(GLOp::from_u32(*op)?, args, result_ty),
        PrimOp::FMod if args.len() == 2 => {
            let a = float_arg(args, 0, result_ty)?;
            let b = float_arg(args, 1, result_ty)?;
            if b == 0.0 {
                return None;
            }
            float_result(a - b * (a / b).floor(), result_ty)
        }
        PrimOp::IsNan | PrimOp::IsInf if args.len() == 1 => {
            let (Constant::Float(value), ty) = &args[0] else {
                return None;
            };
            if !matches!(result_ty, Type::Constructed(TypeName::Bool, _)) {
                return None;
            }
            let value = round_float(*value as f32, ty)?;
            Some(Constant::Bool(if *prim == PrimOp::IsNan {
                value.is_nan()
            } else {
                value.is_infinite()
            }))
        }
        _ if args.len() == 1 => fold_scalar_conversion(prim, args, result_ty),
        _ => None,
    }
}

fn fold_glsl(op: GLOp, args: &[(Constant, Type)], result_ty: &Type) -> Option<Constant> {
    use GLOp::*;
    let arity = match op {
        Round | RoundEven | Trunc | FAbs | SAbs | FSign | SSign | Floor | Ceil | Fract | Radians
        | Degrees | Sin | Cos | Tan | Asin | Acos | Atan | Sinh | Cosh | Tanh | Asinh | Acosh | Atanh
        | Exp | Log | Exp2 | Log2 | Sqrt | InverseSqrt => 1,
        Atan2 | Pow | FMin | UMin | SMin | FMax | UMax | SMax | Step => 2,
        FClamp | UClamp | SClamp | FMix | SmoothStep | Fma => 3,
        _ => return None,
    };
    if args.len() != arity {
        return None;
    }
    if matches!(op, SAbs | SSign | UMin | SMin | UMax | SMax | UClamp | SClamp) {
        return fold_integer(op, args, result_ty);
    }

    let a = float_arg(args, 0, result_ty)?;
    let b = if arity >= 2 { float_arg(args, 1, result_ty)? } else { 0.0 };
    let c = if arity >= 3 { float_arg(args, 2, result_ty)? } else { 0.0 };
    let result = match op {
        // WGSL requires ties-to-even; GLSL Round permits it.
        Round | RoundEven => a.round_ties_even(),
        Trunc => a.trunc(),
        FAbs => a.abs(),
        FSign => {
            if a == 0.0 {
                0.0
            } else {
                a.signum()
            }
        }
        Floor => a.floor(),
        Ceil => a.ceil(),
        // Rust's fract uses truncation and differs for negative inputs.
        Fract => a - a.floor(),
        Radians => a.to_radians(),
        Degrees => a.to_degrees(),
        Sin => a.sin(),
        Cos => a.cos(),
        Tan => a.tan(),
        Asin => a.asin(),
        Acos => a.acos(),
        Atan => a.atan(),
        Sinh => a.sinh(),
        Cosh => a.cosh(),
        Tanh => a.tanh(),
        Asinh => a.asinh(),
        Acosh => a.acosh(),
        Atanh => a.atanh(),
        Atan2 if a != 0.0 || b != 0.0 => a.atan2(b),
        Pow if a >= 0.0 && (a != 0.0 || b > 0.0) => a.powf(b),
        Exp => a.exp(),
        Log => a.ln(),
        Exp2 => a.exp2(),
        Log2 => a.log2(),
        Sqrt => a.sqrt(),
        InverseSqrt => a.sqrt().recip(),
        FMin => float_min(a, b),
        FMax => float_max(a, b),
        FClamp if b <= c => float_min(float_max(a, b), c),
        FMix => a * (1.0 - c) + b * c,
        Step => {
            if b < a {
                0.0
            } else {
                1.0
            }
        }
        SmoothStep if a < b => {
            // Check intermediates before clamp can mask an overflow/NaN.
            let distance = c - a;
            let width = b - a;
            let t = distance / width;
            if !distance.is_finite() || !width.is_finite() || !t.is_finite() {
                return None;
            }
            let t = t.clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        }
        Fma => a.mul_add(b, c),
        _ => return None,
    };
    float_result(result, result_ty)
}

fn fold_integer(op: GLOp, args: &[(Constant, Type)], result_ty: &Type) -> Option<Constant> {
    let signed = matches!(result_ty, Type::Constructed(TypeName::Int(_), _));
    if !signed && !matches!(result_ty, Type::Constructed(TypeName::UInt(_), _)) {
        return None;
    }
    let unsigned_op = matches!(op, GLOp::UMin | GLOp::UMax | GLOp::UClamp);
    if signed == unsigned_op {
        return None;
    }
    let values = args
        .iter()
        .map(|(value, ty)| {
            let Constant::Int(value) = value else { return None };
            if ty != result_ty {
                return None;
            }
            let value = wrap_int(*value as i128, ty);
            Some(if signed { value as i128 } else { value as u64 as i128 })
        })
        .collect::<Option<Vec<_>>>()?;
    let a = values[0];
    let result = match op {
        GLOp::SAbs => a.abs(),
        GLOp::SSign => a.signum(),
        GLOp::SMin | GLOp::UMin => a.min(values[1]),
        GLOp::SMax | GLOp::UMax => a.max(values[1]),
        GLOp::SClamp | GLOp::UClamp if values[1] <= values[2] => a.clamp(values[1], values[2]),
        _ => return None,
    };
    Some(Constant::Int(wrap_int(result, result_ty)))
}

fn float_arg(args: &[(Constant, Type)], index: usize, result_ty: &Type) -> Option<f32> {
    let (Constant::Float(value), ty) = args.get(index)? else {
        return None;
    };
    if ty != result_ty {
        return None;
    }
    let value = round_float(*value as f32, ty)?;
    value.is_finite().then_some(value)
}

fn float_result(value: f32, ty: &Type) -> Option<Constant> {
    let value = round_float(value, ty)?;
    // Preserve domain errors and non-finite results as runtime calls.
    value.is_finite().then_some(Constant::Float(value as f64))
}

fn round_float(value: f32, ty: &Type) -> Option<f32> {
    match ty {
        Type::Constructed(TypeName::Float(16), _) => Some(half::f16::from_f32(value).to_f32()),
        Type::Constructed(TypeName::Float(32), _) => Some(value),
        // TLC FloatLit stores f32: don't pretend to evaluate f64 arithmetic.
        _ => None,
    }
}

// GLSL min/max order -0 below +0. Rust min/max need not preserve that order.
fn float_min(a: f32, b: f32) -> f32 {
    if a.total_cmp(&b).is_le() {
        a
    } else {
        b
    }
}

fn float_max(a: f32, b: f32) -> f32 {
    if a.total_cmp(&b).is_ge() {
        a
    } else {
        b
    }
}

fn fold_scalar_conversion(prim: &PrimOp, args: &[(Constant, Type)], result_ty: &Type) -> Option<Constant> {
    let (arg, arg_ty) = args.first()?;
    match prim {
        PrimOp::FPToSI => {
            let Constant::Float(v) = arg else { return None };
            let Type::Constructed(TypeName::Int(bits), _) = result_ty else {
                return None;
            };
            let truncated = v.trunc();
            let (min, max) = signed_float_bounds(*bits)?;
            (v.is_finite() && truncated >= min && truncated <= max)
                .then(|| Constant::Int(wrap_int(truncated as i128, result_ty)))
        }
        PrimOp::FPToUI => {
            let Constant::Float(v) = arg else { return None };
            let Type::Constructed(TypeName::UInt(bits), _) = result_ty else {
                return None;
            };
            let truncated = v.trunc();
            let max = unsigned_float_max(*bits)?;
            (v.is_finite() && truncated >= 0.0 && truncated <= max)
                .then(|| Constant::Int(wrap_int(truncated as i128, result_ty)))
        }
        PrimOp::SIToFP => {
            let Constant::Int(v) = arg else { return None };
            int_to_float(*v as i128, result_ty).map(Constant::Float)
        }
        PrimOp::UIToFP => {
            let Constant::Int(v) = arg else { return None };
            int_to_float(*v as u64 as i128, result_ty).map(Constant::Float)
        }
        PrimOp::FPConvert => {
            let Constant::Float(v) = arg else { return None };
            match result_ty {
                Type::Constructed(TypeName::Float(32), _) => Some(Constant::Float((*v as f32) as f64)),
                Type::Constructed(TypeName::Float(64), _) => Some(Constant::Float(*v)),
                _ => None,
            }
        }
        PrimOp::SConvert | PrimOp::UConvert => {
            let Constant::Int(v) = arg else { return None };
            Some(Constant::Int(wrap_int(*v as i128, result_ty)))
        }
        PrimOp::Bitcast => fold_scalar_bitcast(arg, arg_ty, result_ty),
        _ => None,
    }
}

fn int_to_float(v: i128, ty: &Type) -> Option<f64> {
    match ty {
        Type::Constructed(TypeName::Float(32), _) => Some((v as f32) as f64),
        Type::Constructed(TypeName::Float(64), _) => Some(v as f64),
        _ => None,
    }
}

fn signed_float_bounds(bits: usize) -> Option<(f64, f64)> {
    match bits {
        8 => Some((i8::MIN as f64, i8::MAX as f64)),
        16 => Some((i16::MIN as f64, i16::MAX as f64)),
        32 => Some((i32::MIN as f64, i32::MAX as f64)),
        64 => Some((i64::MIN as f64, i64::MAX as f64)),
        _ => None,
    }
}

fn unsigned_float_max(bits: usize) -> Option<f64> {
    match bits {
        8 => Some(u8::MAX as f64),
        16 => Some(u16::MAX as f64),
        32 => Some(u32::MAX as f64),
        64 => Some(u64::MAX as f64),
        _ => None,
    }
}

fn fold_scalar_bitcast(arg: &Constant, arg_ty: &Type, result_ty: &Type) -> Option<Constant> {
    match (arg, arg_ty, result_ty) {
        (v, a, b) if a == b => Some(v.clone()),
        (
            Constant::Float(v),
            Type::Constructed(TypeName::Float(32), _),
            Type::Constructed(TypeName::Int(32), _),
        ) => Some(Constant::Int((*v as f32).to_bits() as i32 as i64)),
        (
            Constant::Float(v),
            Type::Constructed(TypeName::Float(32), _),
            Type::Constructed(TypeName::UInt(32), _),
        ) => Some(Constant::Int((*v as f32).to_bits() as i64)),
        (
            Constant::Int(v),
            Type::Constructed(TypeName::Int(32) | TypeName::UInt(32), _),
            Type::Constructed(TypeName::Float(32), _),
        ) => Some(Constant::Float(f32::from_bits(*v as u32) as f64)),
        (
            Constant::Int(v),
            Type::Constructed(TypeName::Int(32), _),
            Type::Constructed(TypeName::UInt(32), _),
        )
        | (
            Constant::Int(v),
            Type::Constructed(TypeName::UInt(32), _),
            Type::Constructed(TypeName::Int(32), _),
        ) => Some(Constant::Int(*v)),
        _ => None,
    }
}
