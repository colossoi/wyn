use crate::{Error, Result};
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumberType {
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

impl NumberType {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "i32" => Some(Self::I32),
            "u32" => Some(Self::U32),
            "i64" => Some(Self::I64),
            "u64" => Some(Self::U64),
            "f32" => Some(Self::F32),
            "f64" => Some(Self::F64),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Number {
    Literal(i128),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
}

fn overflow() -> Error {
    Error::Invalid("numeric overflow or out-of-range conversion".into())
}

impl Number {
    pub fn kind(self) -> Option<NumberType> {
        match self {
            Self::Literal(_) => None,
            Self::I32(_) => Some(NumberType::I32),
            Self::U32(_) => Some(NumberType::U32),
            Self::I64(_) => Some(NumberType::I64),
            Self::U64(_) => Some(NumberType::U64),
            Self::F32(_) => Some(NumberType::F32),
            Self::F64(_) => Some(NumberType::F64),
        }
    }
    pub fn integer(self) -> Result<i128> {
        match self {
            Self::Literal(n) => Ok(n),
            Self::I32(n) => Ok(n.into()),
            Self::U32(n) => Ok(n.into()),
            Self::I64(n) => Ok(n.into()),
            Self::U64(n) => Ok(n.into()),
            Self::F32(_) | Self::F64(_) => Err(Error::Invalid("expected an integer".into())),
        }
    }
    pub fn real(self) -> f64 {
        match self {
            Self::Literal(n) => n as f64,
            Self::I32(n) => n.into(),
            Self::U32(n) => n.into(),
            Self::I64(n) => n as f64,
            Self::U64(n) => n as f64,
            Self::F32(n) => n.into(),
            Self::F64(n) => n,
        }
    }
    pub fn f32(n: f32) -> Result<Self> {
        if n.is_finite() {
            Ok(Self::F32(n))
        } else {
            Err(overflow())
        }
    }
    pub fn f64(n: f64) -> Result<Self> {
        if n.is_finite() {
            Ok(Self::F64(n))
        } else {
            Err(overflow())
        }
    }
    pub fn convert(self, kind: NumberType) -> Result<Self> {
        Ok(match kind {
            NumberType::I32 => Self::I32(i32::try_from(self.integer()?).map_err(|_| overflow())?),
            NumberType::U32 => Self::U32(u32::try_from(self.integer()?).map_err(|_| overflow())?),
            NumberType::I64 => Self::I64(i64::try_from(self.integer()?).map_err(|_| overflow())?),
            NumberType::U64 => Self::U64(u64::try_from(self.integer()?).map_err(|_| overflow())?),
            NumberType::F32 => return Self::f32(self.real() as f32),
            NumberType::F64 => return Self::f64(self.real()),
        })
    }
}

fn common(values: &[Number]) -> Result<NumberType> {
    let kind = values.iter().find_map(|n| n.kind()).unwrap_or(NumberType::I32);
    if values.iter().any(|n| n.kind().is_some_and(|k| k != kind)) {
        return Err(Error::Invalid(
            "mixed numeric types require explicit conversions".into(),
        ));
    }
    Ok(kind)
}

pub(crate) fn compare(op: &str, values: &[Number]) -> Result<bool> {
    if values.is_empty() {
        return Err(Error::Invalid(format!("{op} needs at least one operand")));
    }
    let kind = common(values)?;
    let numbers = values.iter().map(|n| n.convert(kind)).collect::<Result<Vec<_>>>()?;
    let ordered = |a: Number, b: Number| -> Result<Ordering> {
        if matches!(kind, NumberType::F32 | NumberType::F64) {
            let Some(order) = a.real().partial_cmp(&b.real()) else {
                return Err(overflow());
            };
            Ok(order)
        } else {
            Ok(a.integer()?.cmp(&b.integer()?))
        }
    };
    if op == "/=" {
        for (i, a) in numbers.iter().enumerate() {
            for b in &numbers[i + 1..] {
                if ordered(*a, *b)? == Ordering::Equal {
                    return Ok(false);
                }
            }
        }
        return Ok(true);
    }
    for pair in numbers.windows(2) {
        let order = ordered(pair[0], pair[1])?;
        let matches = match op {
            "=" => order == Ordering::Equal,
            "<" => order == Ordering::Less,
            ">" => order == Ordering::Greater,
            "<=" => order != Ordering::Greater,
            ">=" => order != Ordering::Less,
            _ => return Err(Error::Invalid(format!("unknown comparison {op}"))),
        };
        if !matches {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) fn calculate(op: &str, values: &[Number]) -> Result<Number> {
    let wrapping = op.starts_with("i32-") || op.starts_with("u32-");
    if wrapping {
        let [a, b] = values else {
            return Err(Error::Invalid(format!("{op} needs two operands")));
        };
        let kind = if op.starts_with('i') { NumberType::I32 } else { NumberType::U32 };
        if values.iter().any(|n| n.kind().is_some_and(|k| k != kind)) {
            return Err(Error::Invalid("wrapping operands have the wrong type".into()));
        }
        let a = a.convert(kind)?.integer()?;
        let b = b.convert(kind)?.integer()?;
        let n = match &op[4..] {
            "add" => a + b,
            "sub" => a - b,
            "mul" => a * b,
            _ => return Err(Error::Invalid(format!("unknown operation {op}"))),
        };
        return Ok(if kind == NumberType::I32 { Number::I32(n as i32) } else { Number::U32(n as u32) });
    }
    if values.is_empty() {
        return match op {
            "+" => Ok(Number::I32(0)),
            "*" => Ok(Number::I32(1)),
            _ => Err(Error::Invalid(format!("{op} needs operands"))),
        };
    }
    if matches!(op, "floor" | "ceiling") && values.len() > 2 || op == "mod" && values.len() != 2 {
        return Err(Error::Invalid(format!("wrong operand count for {op}")));
    }
    let kind = common(values)?;
    let mut numbers = values.iter().map(|n| n.convert(kind)).collect::<Result<Vec<_>>>()?;
    if numbers.len() == 1 {
        match op {
            "-" => numbers.insert(0, Number::Literal(0).convert(kind)?),
            "/" => numbers.insert(0, Number::Literal(1).convert(kind)?),
            "floor" | "ceiling" => numbers.push(Number::Literal(1).convert(kind)?),
            _ => return Ok(numbers[0]),
        }
    }
    let mut result = numbers[0];
    for &next in &numbers[1..] {
        if matches!(kind, NumberType::F32 | NumberType::F64) {
            let a = result.real();
            let b = next.real();
            if matches!(op, "/" | "floor" | "ceiling" | "mod") && b == 0.0 {
                return Err(Error::Invalid("division by zero".into()));
            }
            let value = match op {
                "+" => a + b,
                "-" => a - b,
                "*" => a * b,
                "/" => a / b,
                "floor" => (a / b).floor(),
                "ceiling" => (a / b).ceil(),
                "mod" => a - (a / b).floor() * b,
                "min" => a.min(b),
                "max" => a.max(b),
                _ => return Err(Error::Invalid(format!("unknown arithmetic {op}"))),
            };
            if matches!(op, "floor" | "ceiling") {
                let (min, max, target) = if kind == NumberType::F32 {
                    (i32::MIN as f64, 2147483648.0, NumberType::I32)
                } else {
                    (i64::MIN as f64, 9223372036854775808.0, NumberType::I64)
                };
                if !value.is_finite() || value < min || value >= max {
                    return Err(overflow());
                }
                result = Number::Literal(value as i128).convert(target)?;
            } else {
                result = Number::f64(value)?.convert(kind)?;
            }
        } else {
            let a = result.integer()?;
            let b = next.integer()?;
            if matches!(op, "/" | "floor" | "ceiling" | "mod") && b == 0 {
                return Err(Error::Invalid("division by zero".into()));
            }
            let value = match op {
                "+" => a.checked_add(b),
                "-" => a.checked_sub(b),
                "*" => a.checked_mul(b),
                "/" if a % b == 0 => Some(a / b),
                "/" => return Err(Error::Invalid("integer division is not exact".into())),
                "floor" | "ceiling" | "mod" => {
                    let q = a / b;
                    let r = a % b;
                    let floor = q - i128::from(r != 0 && (r < 0) != (b < 0));
                    let ceil = q + i128::from(r != 0 && (r < 0) == (b < 0));
                    Number::Literal(if op == "ceiling" { ceil } else { floor }).convert(kind)?;
                    Some(match op {
                        "floor" => floor,
                        "ceiling" => ceil,
                        _ => a - floor * b,
                    })
                }
                "min" => Some(a.min(b)),
                "max" => Some(a.max(b)),
                _ => return Err(Error::Invalid(format!("unknown arithmetic {op}"))),
            };
            let Some(value) = value else {
                return Err(overflow());
            };
            result = Number::Literal(value).convert(kind)?;
        }
    }
    Ok(result)
}
