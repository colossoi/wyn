//! Typed literal evaluation shared with EGIR; algebraic identities live in .egg.
use super::{intern_expr, name};
use crate::builtins::lowering::{BuiltinLowering, PrimOp};
use crate::builtins::{by_id, Purity};
use crate::egglog::data::{Array, ExprData, ExprId, ExprKind, Ir, OperationKind, TypeId};
use crate::op::{BinaryOperator, UnaryOperator};
use crate::scalar_eval::{binary, unary, wrap_int, Scalar};
use crate::types::{Type, TypeName};

pub(super) fn facts(data: &Ir, id: ExprId, out: &mut String) {
    let flag = match literal(data, id) {
        Some(Scalar::Int(0)) => Some("Zero"),
        Some(Scalar::Int(1)) => Some("One"),
        Some(Scalar::Float(x)) if x == 0.0 => Some("Zero"),
        Some(Scalar::Float(x)) if x == 1.0 => Some("One"),
        _ => None,
    };
    if let Some(flag) = flag {
        out.push_str(&format!("({flag} {})\n", name(id)));
    }
    if matches!(lowering(data, id), Some(BuiltinLowering::PrimOp(PrimOp::Bitcast))) {
        out.push_str(&format!("(Bitcast {})\n", name(id)));
    }
}

pub(super) fn lowering(data: &Ir, f: ExprId) -> Option<&BuiltinLowering> {
    let ExprKind::Builtin(b) = data.expressions[f].kind else {
        return None;
    };
    let b = &data.builtins[b];
    let def = by_id(b.builtin);
    (def.raw.purity == Purity::Pure).then_some(())?;
    Some(&def.overloads().get(b.overload_idx)?.lowering)
}

pub(super) fn literal(data: &Ir, id: ExprId) -> Option<Scalar> {
    let e = &data.expressions[id];
    match (&data.types[e.ty].ty, &e.kind) {
        (Type::Constructed(TypeName::Int(_), _), ExprKind::Int(s)) => Some(Scalar::Int(s.parse().ok()?)),
        (Type::Constructed(TypeName::UInt(_), _), ExprKind::Int(s)) => {
            Some(Scalar::Int(s.parse::<u64>().ok()? as i64))
        }
        (Type::Constructed(TypeName::Float(32), _), ExprKind::FloatBits(b)) => {
            Some(Scalar::Float(f32::from_bits(*b) as f64))
        }
        (Type::Constructed(TypeName::Bool, _), ExprKind::Bool(b)) => Some(Scalar::Bool(*b)),
        _ => None,
    }
}

fn constant(data: &mut Ir, ty: TypeId, value: Scalar) -> Option<ExprId> {
    let kind = match (value, &data.types[ty].ty) {
        (Scalar::Int(v), t @ Type::Constructed(TypeName::Int(_), _)) => {
            ExprKind::Int(wrap_int(v as i128, t).to_string())
        }
        (Scalar::Int(v), t @ Type::Constructed(TypeName::UInt(_), _)) => {
            ExprKind::Int((wrap_int(v as i128, t) as u64).to_string())
        }
        (Scalar::Float(v), Type::Constructed(TypeName::Float(32), _)) => {
            ExprKind::FloatBits((v as f32).to_bits())
        }
        (Scalar::Bool(v), Type::Constructed(TypeName::Bool, _)) => ExprKind::Bool(v),
        _ => return None,
    };
    Some(intern_expr(data, ty, kind))
}

pub(super) fn evaluate(data: &mut Ir, id: ExprId) -> Option<ExprId> {
    let ExprData { ty, kind } = data.expressions[id].clone();
    if let ExprKind::OperationResult(op) = kind {
        let OperationKind::Index { array, index } = data.operations[op].kind else {
            return None;
        };
        let Scalar::Int(i) = literal(data, index)? else {
            return None;
        };
        let items = match &data.expressions[array].kind {
            ExprKind::Vector(xs) | ExprKind::Array(Array::Literal(xs)) => xs,
            _ => return None, // A tuple may describe a buffer view, not its elements.
        };
        let value = *items.get(usize::try_from(i).ok()?)?;
        return (data.expressions[value].ty == ty).then_some(value);
    }
    let ExprKind::PureApp { function, args } = kind else {
        return None;
    };
    let value = match (&data.expressions[function].kind, args.as_slice()) {
        (ExprKind::BinOp(op), &[a, b]) => {
            let op = BinaryOperator::try_from(op.as_str()).ok()?;
            if let (Some(a_value), Some(b_value)) = (literal(data, a), literal(data, b)) {
                binary(op, a_value, b_value, &data.types[data.expressions[a].ty].ty)?
            } else {
                return None;
            }
        }
        (ExprKind::UnOp(op), &[a]) => unary(
            UnaryOperator::try_from(op.as_str()).ok()?,
            literal(data, a)?,
            &data.types[data.expressions[a].ty].ty,
        )?,
        (ExprKind::Builtin(_), &[a]) => {
            let BuiltinLowering::PrimOp(prim) = lowering(data, function)? else {
                return None;
            };
            if *prim == PrimOp::Bitcast {
                return bitcast(data, ty, a);
            }
            conversion(data, prim, ty, a)?
        }
        _ => return None,
    };
    constant(data, ty, value)
}

fn conversion(data: &Ir, prim: &PrimOp, ty: TypeId, a: ExprId) -> Option<Scalar> {
    let value = literal(data, a)?;
    let result = &data.types[ty].ty;
    Some(match (prim, result, value) {
        (PrimOp::GlslExt(8), _, Scalar::Float(v)) => Scalar::Float((v as f32).floor() as f64),
        (PrimOp::GlslExt(9), _, Scalar::Float(v)) => Scalar::Float((v as f32).ceil() as f64),
        (PrimOp::FPToSI, Type::Constructed(TypeName::Int(32), _), Scalar::Float(v))
            if v.is_finite() && v.trunc() >= -2147483648.0 && v.trunc() < 2147483648.0 =>
        {
            Scalar::Int(v.trunc() as i64)
        }
        (PrimOp::FPToUI, Type::Constructed(TypeName::UInt(32), _), Scalar::Float(v))
            if v.is_finite() && v.trunc() >= 0.0 && v.trunc() < 4294967296.0 =>
        {
            Scalar::Int(v.trunc() as i64)
        }
        (PrimOp::SIToFP, Type::Constructed(TypeName::Float(32), _), Scalar::Int(v)) => {
            Scalar::Float(v as f32 as f64)
        }
        (PrimOp::UIToFP, Type::Constructed(TypeName::Float(32), _), Scalar::Int(v)) => {
            Scalar::Float((v as u64) as f32 as f64)
        }
        (PrimOp::SConvert | PrimOp::UConvert, _, Scalar::Int(v)) => {
            Scalar::Int(wrap_int(v as i128, result))
        }
        (PrimOp::FPConvert, Type::Constructed(TypeName::Float(32), _), Scalar::Float(v)) => {
            Scalar::Float(v)
        }
        _ => return None,
    })
}

fn bitcast(data: &mut Ir, ty: TypeId, a: ExprId) -> Option<ExprId> {
    let input = &data.expressions[a];
    let bits = match (&data.types[input.ty].ty, &input.kind) {
        (Type::Constructed(TypeName::Float(32), _), ExprKind::FloatBits(bits)) => *bits,
        (Type::Constructed(TypeName::Int(32), _), ExprKind::Int(s)) => s.parse::<i32>().ok()? as u32,
        (Type::Constructed(TypeName::UInt(32), _), ExprKind::Int(s)) => s.parse::<u32>().ok()?,
        _ => return None,
    };
    let value = match &data.types[ty].ty {
        Type::Constructed(TypeName::Float(32), _) => ExprKind::FloatBits(bits),
        Type::Constructed(TypeName::Int(32), _) => ExprKind::Int((bits as i32).to_string()),
        Type::Constructed(TypeName::UInt(32), _) => ExprKind::Int(bits.to_string()),
        _ => return None,
    };
    Some(intern_expr(data, ty, value))
}
