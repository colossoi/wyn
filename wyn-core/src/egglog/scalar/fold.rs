//! Typed scalar primitives. Egglog owns matching, propagation, and saturation.
use crate::builtins::lowering::{BuiltinLowering, PrimOp};
use crate::builtins::{by_id, Purity};
use crate::egglog::data::{ExprId, ExprKind, Ir};
use crate::op::{BinaryOperator, UnaryOperator};
use crate::scalar_eval::{binary, unary, wrap_int, Scalar};
use crate::types::{Type, TypeName};
use egglog_engine::ast::Span;
use egglog_engine::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog_engine::prelude::BaseSort;
use egglog_engine::sort::{I64Sort, StringSort, S};
use egglog_engine::{Core, EGraph, Error, FullState, Primitive, PurePrim, PureState, Read, Value, Write};

pub(super) fn register(graph: &mut EGraph) {
    for primitive in [
        ScalarPrimitive::Binary,
        ScalarPrimitive::Unary,
        ScalarPrimitive::Integer,
    ] {
        graph.add_pure_primitive(primitive, None);
    }
}

#[derive(Clone)]
enum ScalarPrimitive {
    Binary,
    Unary,
    Integer,
}
impl Primitive for ScalarPrimitive {
    fn name(&self) -> &str {
        match self {
            Self::Binary => "wyn-binary",
            Self::Unary => "wyn-unary",
            Self::Integer => "wyn-i64",
        }
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let arity = match self {
            Self::Binary => 5,
            Self::Unary => 4,
            Self::Integer => 1,
        };
        let mut sorts = vec![StringSort.to_arcsort(); arity];
        sorts.push(if matches!(self, Self::Integer) {
            I64Sort.to_arcsort()
        } else {
            StringSort.to_arcsort()
        });
        SimpleTypeConstraint::new(self.name(), sorts, span.clone()).into_box()
    }
}
impl PurePrim for ScalarPrimitive {
    fn apply<'a, 'db>(&self, state: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        // Sort constraints guarantee these are strings; this unwrap decodes an
        // egglog base value, not an Option or Result.
        let string = |v| state.base_values().unwrap::<S>(v);
        let result = match (self, args) {
            (Self::Integer, &[value]) => {
                return string(value).as_str().parse::<i64>().ok().map(|v| state.base_values().get(v))
            }
            (Self::Binary, &[op, input, output, a, b]) => {
                let input = scalar_type(string(input).as_str())?;
                let output = scalar_type(string(output).as_str())?;
                let value = binary(
                    BinaryOperator::try_from(string(op).as_str()).ok()?,
                    decode(&input, string(a).as_str())?,
                    decode(&input, string(b).as_str())?,
                    &input,
                )?;
                encode(&output, value)?
            }
            (Self::Unary, &[op, input, output, a]) => evaluate_unary(
                string(op).as_str(),
                string(input).as_str(),
                string(output).as_str(),
                string(a).as_str(),
            )?,
            _ => unreachable!("primitive {} received {} arguments", self.name(), args.len()),
        };
        Some(state.base_values().get(S::new(result)))
    }
}

pub(super) fn facts(data: &Ir, mut sink: FullState<'_, '_>) -> Result<(), Error> {
    for (&id, t) in &data.types {
        let tag = match t.ty {
            Type::Constructed(TypeName::Int(bits), _) => format!("i{bits}"),
            Type::Constructed(TypeName::UInt(bits), _) => format!("u{bits}"),
            Type::Constructed(TypeName::Float(32), _) => "f32".into(),
            Type::Constructed(TypeName::Bool, _) => "bool".into(),
            _ => continue,
        };
        let ty = sink.add("TypeId", (i64::from(id.as_u32()),))?;
        sink.add("ScalarType", (ty, S::new(tag)))?;
        if matches!(t.ty, Type::Constructed(TypeName::Int(_) | TypeName::UInt(_), _)) {
            sink.add("IntegerType", (ty,))?;
        }
    }
    for (&id, _) in &data.expressions {
        let Some(BuiltinLowering::PrimOp(prim)) = lowering(data, id) else {
            continue;
        };
        let op = match prim {
            PrimOp::Bitcast => "bitcast",
            PrimOp::GlslExt(8) => "floor",
            PrimOp::GlslExt(9) => "ceil",
            PrimOp::FPToSI => "fptosi",
            PrimOp::FPToUI => "fptoui",
            PrimOp::SIToFP => "sitofp",
            PrimOp::UIToFP => "uitofp",
            PrimOp::SConvert | PrimOp::UConvert => "int-convert",
            PrimOp::FPConvert => "float-convert",
            _ => continue,
        };
        let Some(value) = sink.lookup("SourceExpression", (i64::from(id.as_u32()),))? else {
            continue;
        };
        sink.add("FoldUnary", (value, S::new(op.to_owned())))?;
        if *prim == PrimOp::Bitcast {
            sink.add("Bitcast", (value,))?;
        }
    }
    Ok(())
}

fn scalar_type(name: &str) -> Option<Type> {
    let kind = if name == "bool" {
        TypeName::Bool
    } else if name == "f32" {
        TypeName::Float(32)
    } else {
        let bits = name.get(1..)?.parse().ok()?;
        match name.get(..1)? {
            "i" => TypeName::Int(bits),
            "u" => TypeName::UInt(bits),
            _ => return None,
        }
    };
    Some(Type::Constructed(kind, vec![]))
}

fn decode(ty: &Type, text: &str) -> Option<Scalar> {
    Some(match ty {
        Type::Constructed(TypeName::Int(_), _) => Scalar::Int(text.parse().ok()?),
        Type::Constructed(TypeName::UInt(_), _) => Scalar::Int(text.parse::<u64>().ok()? as i64),
        Type::Constructed(TypeName::Float(32), _) => {
            Scalar::Float(f32::from_bits(text.parse().ok()?) as f64)
        }
        Type::Constructed(TypeName::Bool, _) => Scalar::Bool(text.parse().ok()?),
        _ => return None,
    })
}
fn encode(ty: &Type, value: Scalar) -> Option<String> {
    Some(match (value, ty) {
        (Scalar::Int(v), Type::Constructed(TypeName::Int(_), _)) => wrap_int(v as i128, ty).to_string(),
        (Scalar::Int(v), Type::Constructed(TypeName::UInt(_), _)) => {
            (wrap_int(v as i128, ty) as u64).to_string()
        }
        (Scalar::Float(v), Type::Constructed(TypeName::Float(32), _)) => (v as f32).to_bits().to_string(),
        (Scalar::Bool(v), Type::Constructed(TypeName::Bool, _)) => v.to_string(),
        _ => return None,
    })
}

fn evaluate_unary(op: &str, input: &str, output: &str, text: &str) -> Option<String> {
    let input = scalar_type(input)?;
    let output = scalar_type(output)?;
    if op == "bitcast" {
        let bits = match input {
            Type::Constructed(TypeName::Float(32) | TypeName::UInt(32), _) => text.parse::<u32>().ok()?,
            Type::Constructed(TypeName::Int(32), _) => text.parse::<i32>().ok()? as u32,
            _ => return None,
        };
        return Some(match output {
            Type::Constructed(TypeName::Float(32) | TypeName::UInt(32), _) => bits.to_string(),
            Type::Constructed(TypeName::Int(32), _) => (bits as i32).to_string(),
            _ => return None,
        });
    }
    let value = decode(&input, text)?;
    if let Ok(op) = UnaryOperator::try_from(op) {
        return encode(&output, unary(op, value, &input)?);
    }
    encode(&output, conversion(op, &output, value)?)
}
fn conversion(op: &str, result: &Type, value: Scalar) -> Option<Scalar> {
    Some(match (op, result, value) {
        ("floor", _, Scalar::Float(v)) => Scalar::Float((v as f32).floor() as f64),
        ("ceil", _, Scalar::Float(v)) => Scalar::Float((v as f32).ceil() as f64),
        ("fptosi", Type::Constructed(TypeName::Int(32), _), Scalar::Float(v))
            if v.is_finite() && v.trunc() >= -2147483648.0 && v.trunc() < 2147483648.0 =>
        {
            Scalar::Int(v.trunc() as i64)
        }
        ("fptoui", Type::Constructed(TypeName::UInt(32), _), Scalar::Float(v))
            if v.is_finite() && v.trunc() >= 0.0 && v.trunc() < 4294967296.0 =>
        {
            Scalar::Int(v.trunc() as i64)
        }
        ("sitofp", Type::Constructed(TypeName::Float(32), _), Scalar::Int(v)) => {
            Scalar::Float(v as f32 as f64)
        }
        ("uitofp", Type::Constructed(TypeName::Float(32), _), Scalar::Int(v)) => {
            Scalar::Float((v as u64) as f32 as f64)
        }
        ("int-convert", _, Scalar::Int(v)) => Scalar::Int(wrap_int(v as i128, result)),
        ("float-convert", Type::Constructed(TypeName::Float(32), _), Scalar::Float(v)) => Scalar::Float(v),
        _ => return None,
    })
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
