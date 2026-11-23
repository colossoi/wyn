//! Typed scalar primitives. Egglog owns matching, propagation, and saturation.
use crate::builtins::by_id;
use crate::constant_eval::{self, Constant};
use crate::egglog::payload::{Payload, PayloadSort};
use crate::scalar_eval::{binary, unary, wrap_int, Scalar};
use crate::types::{Type, TypeName};
use egglog_engine::ast::Span;
use egglog_engine::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog_engine::prelude::BaseSort;
use egglog_engine::sort::{Boxed, I64Sort, StringSort, S};
use egglog_engine::{Core, EGraph, Primitive, PurePrim, PureState, Value};

pub(super) fn register(graph: &mut EGraph) {
    for primitive in [ScalarPrimitive::Binary, ScalarPrimitive::Unary] {
        graph.add_pure_primitive(primitive, None);
    }
}

#[derive(Clone)]
enum ScalarPrimitive {
    Binary,
    Unary,
}
impl Primitive for ScalarPrimitive {
    fn name(&self) -> &str {
        match self {
            Self::Binary => "wyn-binary",
            Self::Unary => "wyn-unary",
        }
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let arity = match self {
            Self::Binary => 5,
            Self::Unary => 4,
        };
        let mut sorts = vec![StringSort.to_arcsort(); arity];
        sorts[0] = PayloadSort.to_arcsort();
        sorts.push(StringSort.to_arcsort());
        sorts[3..].fill(I64Sort.to_arcsort());
        SimpleTypeConstraint::new(self.name(), sorts, span.clone()).into_box()
    }
}
impl PurePrim for ScalarPrimitive {
    fn apply<'a, 'db>(&self, state: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let string = |v| state.base_values().unwrap::<S>(v);
        let result = match (self, args) {
            (Self::Binary, &[op, input, output, a, b]) => {
                let Payload::Binary(op) = state.value_to_base::<Boxed<Payload>>(op).0 else {
                    return None;
                };
                let input = scalar_type(string(input).as_str())?;
                let output = scalar_type(string(output).as_str())?;
                evaluate_typed(
                    Payload::Binary(op),
                    &input,
                    &output,
                    &[
                        state.base_values().unwrap::<i64>(a),
                        state.base_values().unwrap::<i64>(b),
                    ],
                )?
            }
            (Self::Unary, &[op, input, output, a]) => evaluate_unary(
                state.value_to_base::<Boxed<Payload>>(op).0,
                string(input).as_str(),
                string(output).as_str(),
                state.base_values().unwrap::<i64>(a),
            )?,
            _ => unreachable!("primitive {} received {} arguments", self.name(), args.len()),
        };
        Some(state.base_values().get(result))
    }
}

pub(super) fn scalar_type(name: &str) -> Option<Type> {
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

fn decode(ty: &Type, bits: i64) -> Option<Scalar> {
    Some(match ty {
        Type::Constructed(TypeName::Int(_) | TypeName::UInt(_), _) => Scalar::Int(bits),
        Type::Constructed(TypeName::Float(32), _) => Scalar::Float(f32::from_bits(bits as u32) as f64),
        Type::Constructed(TypeName::Bool, _) if matches!(bits, 0 | 1) => Scalar::Bool(bits != 0),
        _ => return None,
    })
}
fn encode(ty: &Type, value: Scalar) -> Option<i64> {
    Some(match (value, ty) {
        (Scalar::Int(v), Type::Constructed(TypeName::Int(_) | TypeName::UInt(_), _)) => {
            wrap_int(v as i128, ty)
        }
        (Scalar::Float(v), Type::Constructed(TypeName::Float(32), _)) => (v as f32).to_bits() as i64,
        (Scalar::Bool(v), Type::Constructed(TypeName::Bool, _)) => i64::from(v),
        _ => return None,
    })
}

fn evaluate_unary(payload: Payload, input: &str, output: &str, bits: i64) -> Option<i64> {
    let input = scalar_type(input)?;
    let output = scalar_type(output)?;
    evaluate_typed(payload, &input, &output, &[bits])
}

/// Evaluate the same scalar operations through either a graph edge or a rule.
pub(super) fn evaluate_typed(payload: Payload, input: &Type, output: &Type, bits: &[i64]) -> Option<i64> {
    let result = match (payload, bits) {
        (Payload::Binary(op), &[a, b]) => binary(op, decode(input, a)?, decode(input, b)?, input)?,
        (Payload::Unary(op), &[value]) => unary(op, decode(input, value)?, input)?,
        (Payload::Intrinsic { id, overload }, &[value]) => {
            let overload = by_id(id).overloads().get(overload)?;
            constant_eval::builtin(
                &overload.lowering,
                &[(Constant::from_scalar(decode(input, value)?), input.clone())],
                output,
            )?
            .as_scalar()?
        }
        _ => return None,
    };
    encode(output, result)
}

#[cfg(test)]
#[path = "fold_tests.rs"]
mod tests;
