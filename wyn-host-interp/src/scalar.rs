//! Wyn scalar operations retain device widths and IEEE floating-point values.
use crate::{Error, Number, Result, Value};

pub(crate) fn evaluate(name: &str, args: &[Value]) -> Result<Value> {
    if name == "wyn-f32-bits" {
        let [bits] = args else {
            return Err(invalid(name));
        };
        return Ok(Value::Number(Number::F32(f32::from_bits(bits.u32()?))));
    }
    let Some((ty, op)) = name.strip_prefix("wyn-").and_then(|s| s.split_once('-')) else {
        return Err(invalid(name));
    };
    // Function arguments have already been evaluated eagerly by the caller.
    if op == "select" {
        let [no, yes, condition] = args else {
            return Err(invalid(name));
        };
        let correct_type = |value: &Value| {
            matches!(
                (ty, value),
                ("bool", Value::True | Value::Nil)
                    | ("i32", Value::Number(Number::I32(_)))
                    | ("u32", Value::Number(Number::U32(_)))
                    | ("f32", Value::Number(Number::F32(_)))
            )
        };
        if !correct_type(no) || !correct_type(yes) {
            return Err(invalid(name));
        }
        return match condition {
            Value::True => Ok(yes.clone()),
            Value::Nil => Ok(no.clone()),
            _ => Err(invalid(name)),
        };
    }
    if ty == "bool" {
        let values = args
            .iter()
            .map(|value| match value {
                Value::True => Ok(true),
                Value::Nil => Ok(false),
                _ => Err(invalid(name)),
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(Value::boolean(match (op, values.as_slice()) {
            ("not", [a]) => !a,
            ("and", [a, b]) => a & b,
            ("or", [a, b]) => a | b,
            ("xor", [a, b]) => a ^ b,
            ("eq", [a, b]) => a == b,
            ("ne", [a, b]) => a != b,
            _ => return Err(invalid(name)),
        }));
    }
    let numbers = args.iter().map(Value::number).collect::<Result<Vec<_>>>()?;
    macro_rules! integer {
        ($kind:ident, $type:ty) => {{
            let values = numbers
                .iter()
                .map(|n| match n {
                    Number::$kind(n) => Ok(*n),
                    _ => Err(invalid(name)),
                })
                .collect::<Result<Vec<_>>>()?;
            let result = match (op, values.as_slice()) {
                ("add", [a, b]) => a.wrapping_add(*b),
                ("sub", [a, b]) => a.wrapping_sub(*b),
                ("mul", [a, b]) => a.wrapping_mul(*b),
                ("div", [a, b]) => {
                    let Some(n) = a.checked_div(*b) else {
                        return Err(invalid(name));
                    };
                    n
                }
                ("rem", [a, b]) => {
                    let Some(n) = a.checked_rem(*b) else {
                        return Err(invalid(name));
                    };
                    n
                }
                ("neg", [a]) => a.wrapping_neg(),
                ("not", [a]) => !a,
                ("and", [a, b]) => a & b,
                ("or", [a, b]) => a | b,
                ("xor", [a, b]) => a ^ b,
                ("shl", [a, b]) => a.wrapping_shl(*b as u32),
                ("shr", [a, b]) => a.wrapping_shr(*b as u32),
                ("min", [a, b]) => *a.min(b),
                ("max", [a, b]) => *a.max(b),
                ("eq", [a, b]) => return Ok(Value::boolean(a == b)),
                ("ne", [a, b]) => return Ok(Value::boolean(a != b)),
                ("lt", [a, b]) => return Ok(Value::boolean(a < b)),
                ("le", [a, b]) => return Ok(Value::boolean(a <= b)),
                ("gt", [a, b]) => return Ok(Value::boolean(a > b)),
                ("ge", [a, b]) => return Ok(Value::boolean(a >= b)),
                ("to-i32", [a]) => return Ok(Value::Number(Number::I32(*a as i32))),
                ("to-u32", [a]) => return Ok(Value::Number(Number::U32(*a as u32))),
                ("to-f32", [a]) => return Ok(Value::Number(Number::F32(*a as f32))),
                _ => return Err(invalid(name)),
            };
            Ok(Value::Number(Number::$kind(result as $type)))
        }};
    }
    match ty {
        "i32" => {
            if let [Number::I32(n)] = numbers.as_slice() {
                match op {
                    "abs" => return Ok(Value::Number(Number::I32(n.wrapping_abs()))),
                    "sign" => return Ok(Value::Number(Number::I32(n.signum()))),
                    _ => {}
                }
            }
            integer!(I32, i32)
        }
        "u32" => integer!(U32, u32),
        "f32" => {
            let values = numbers
                .iter()
                .map(|n| match n {
                    Number::F32(n) => Ok(*n),
                    _ => Err(invalid(name)),
                })
                .collect::<Result<Vec<_>>>()?;
            let result = match (op, values.as_slice()) {
                ("add", [a, b]) => a + b,
                ("sub", [a, b]) => a - b,
                ("mul", [a, b]) => a * b,
                ("div", [a, b]) => a / b,
                ("rem", [a, b]) => a % b,
                ("neg", [a]) => -a,
                ("eq", [a, b]) => return Ok(Value::boolean(a == b)),
                ("ne", [a, b]) => return Ok(Value::boolean(!a.is_nan() && !b.is_nan() && a != b)),
                ("lt", [a, b]) => return Ok(Value::boolean(a < b)),
                ("le", [a, b]) => return Ok(Value::boolean(a <= b)),
                ("gt", [a, b]) => return Ok(Value::boolean(a > b)),
                ("ge", [a, b]) => return Ok(Value::boolean(a >= b)),
                ("isnan", [a]) => return Ok(Value::boolean(a.is_nan())),
                ("isinf", [a]) => return Ok(Value::boolean(a.is_infinite())),
                ("to-i32", [a]) => return Ok(Value::Number(Number::I32(*a as i32))),
                ("to-u32", [a]) => return Ok(Value::Number(Number::U32(*a as u32))),
                ("to-f32", [a]) => *a,
                ("round", [a]) => a.round(),
                ("round-even", [a]) => a.round_ties_even(),
                ("trunc", [a]) => a.trunc(),
                ("abs", [a]) => a.abs(),
                ("sign", [a]) => {
                    if *a == 0.0 {
                        0.0
                    } else {
                        a.signum()
                    }
                }
                ("floor", [a]) => a.floor(),
                ("ceil", [a]) => a.ceil(),
                ("fract", [a]) => a - a.floor(),
                ("radians", [a]) => a.to_radians(),
                ("degrees", [a]) => a.to_degrees(),
                ("sin", [a]) => a.sin(),
                ("cos", [a]) => a.cos(),
                ("tan", [a]) => a.tan(),
                ("asin", [a]) => a.asin(),
                ("acos", [a]) => a.acos(),
                ("atan", [a]) => a.atan(),
                ("sinh", [a]) => a.sinh(),
                ("cosh", [a]) => a.cosh(),
                ("tanh", [a]) => a.tanh(),
                ("asinh", [a]) => a.asinh(),
                ("acosh", [a]) => a.acosh(),
                ("atanh", [a]) => a.atanh(),
                ("atan2", [a, b]) => a.atan2(*b),
                ("pow", [a, b]) => a.powf(*b),
                ("exp", [a]) => a.exp(),
                ("log", [a]) => a.ln(),
                ("exp2", [a]) => a.exp2(),
                ("log2", [a]) => a.log2(),
                ("sqrt", [a]) => a.sqrt(),
                ("rsqrt", [a]) => 1.0 / a.sqrt(),
                ("min", [a, b]) => a.min(*b),
                ("max", [a, b]) => a.max(*b),
                _ => return Err(invalid(name)),
            };
            Ok(Value::Number(Number::F32(result)))
        }
        _ => Err(invalid(name)),
    }
}

fn invalid(name: &str) -> Error {
    Error::Invalid(format!(
        "invalid operands or unsupported Wyn scalar operation {name}"
    ))
}
