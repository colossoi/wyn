use super::{HostBinary, HostExpression, HostScalar};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    set: u32,
    binding: u32,
    offset: u32,
    #[serde(rename = "type")]
    scalar: HostScalar,
}

#[derive(Serialize, Deserialize)]
struct Expression {
    inputs: BTreeMap<String, Input>,
    count: String,
}

impl Serialize for HostExpression {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut inputs = BTreeMap::new();
        let count = format_expression(self, &mut inputs, 0, 0).map_err(serde::ser::Error::custom)?;
        Expression { inputs, count }.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for HostExpression {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let expression = Expression::deserialize(deserializer)?;
        parse(&expression).map_err(serde::de::Error::custom)
    }
}

fn scalar_name(scalar: HostScalar) -> &'static str {
    match scalar {
        HostScalar::I32 => "i32",
        HostScalar::U32 => "u32",
        HostScalar::F32 => "f32",
    }
}

fn identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn format_expression(
    expr: &HostExpression,
    inputs: &mut BTreeMap<String, Input>,
    parent: u8,
    depth: usize,
) -> Result<String, String> {
    if depth > 64 {
        return Err("host expression nesting exceeds 64".into());
    }
    Ok(match expr {
        HostExpression::Uniform {
            name,
            set,
            binding,
            offset,
            scalar,
        } => {
            let input = Input {
                set: *set,
                binding: *binding,
                offset: *offset,
                scalar: *scalar,
            };
            let base =
                if identifier(name) { name.clone() } else { format!("input_{set}_{binding}_{offset}") };
            let mut alias = base.clone();
            let mut suffix = 1;
            while inputs.get(&alias).is_some_and(|previous| *previous != input) {
                alias = format!("{base}_{suffix}");
                suffix += 1;
            }
            inputs.insert(alias.clone(), input);
            alias
        }
        HostExpression::Constant { scalar, bits } => match scalar {
            HostScalar::I32 => (*bits as i32).to_string(),
            HostScalar::U32 => format!("{bits}u32"),
            HostScalar::F32 => {
                let value = f32::from_bits(*bits);
                if !value.is_finite() {
                    return Err("host expression literal must be finite".into());
                }
                format!("{value:?}f32")
            }
        },
        HostExpression::Convert { to, value } => format!(
            "{}({})",
            scalar_name(*to),
            format_expression(value, inputs, 0, depth + 1)?
        ),
        HostExpression::Binary { op, left, right } => {
            let (symbol, precedence) = match op {
                HostBinary::Add => ("+", 1),
                HostBinary::Subtract => ("-", 1),
                HostBinary::Multiply => ("*", 2),
                HostBinary::Divide => ("/", 2),
                HostBinary::Remainder => ("%", 2),
            };
            let text = format!(
                "{} {symbol} {}",
                format_expression(left, inputs, precedence, depth + 1)?,
                format_expression(right, inputs, precedence + 1, depth + 1)?
            );
            if precedence < parent {
                format!("({text})")
            } else {
                text
            }
        }
    })
}

fn parse(expression: &Expression) -> Result<HostExpression, String> {
    if expression.count.len() > 16384 {
        return Err("host expression exceeds 16384 bytes".into());
    }
    if expression.inputs.keys().any(|name| !identifier(name)) {
        return Err("invalid host input identifier".into());
    }
    let mut parser = Parser {
        rest: &expression.count,
        inputs: &expression.inputs,
        nodes: 0,
    };
    let (value, _) = parser.expression(0, 0)?;
    if !parser.rest.trim().is_empty() {
        return Err(format!("unexpected host expression token: {}", parser.rest));
    }
    Ok(value)
}

struct Parser<'a> {
    rest: &'a str,
    inputs: &'a BTreeMap<String, Input>,
    nodes: usize,
}

impl Parser<'_> {
    fn consume(&mut self, token: &str) -> bool {
        self.rest = self.rest.trim_start();
        if let Some(rest) = self.rest.strip_prefix(token) {
            self.rest = rest;
            true
        } else {
            false
        }
    }

    fn expression(&mut self, minimum: u8, depth: usize) -> Result<(HostExpression, HostScalar), String> {
        self.nodes += 1;
        if depth > 64 || self.nodes > 256 {
            return Err("host expression is too complex".into());
        }
        let (mut left, scalar) = self.atom(depth + 1)?;
        loop {
            self.rest = self.rest.trim_start();
            let (op, precedence) = match self.rest.as_bytes().first() {
                Some(b'+') => (HostBinary::Add, 1),
                Some(b'-') => (HostBinary::Subtract, 1),
                Some(b'*') => (HostBinary::Multiply, 2),
                Some(b'/') => (HostBinary::Divide, 2),
                Some(b'%') => (HostBinary::Remainder, 2),
                _ => break,
            };
            if precedence < minimum {
                break;
            }
            self.rest = &self.rest[1..];
            let (right, other) = self.expression(precedence + 1, depth + 1)?;
            if scalar != other {
                return Err(
                    "host expression operand types differ; use an explicit cast or typed literal".into(),
                );
            }
            left = HostExpression::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok((left, scalar))
    }

    fn atom(&mut self, depth: usize) -> Result<(HostExpression, HostScalar), String> {
        self.rest = self.rest.trim_start();
        if self.consume("(") {
            let value = self.expression(0, depth)?;
            if !self.consume(")") {
                return Err("expected ')' in host expression".into());
            }
            return Ok(value);
        }
        let bytes = self.rest.as_bytes();
        if bytes.first().is_some_and(u8::is_ascii_digit)
            || (bytes.first() == Some(&b'-') && bytes.get(1).is_some_and(u8::is_ascii_digit))
        {
            let length = self
                .rest
                .char_indices()
                .take_while(|(index, c)| {
                    c.is_ascii_alphanumeric()
                        || *c == '.'
                        || ((*c == '-' || *c == '+')
                            && (*index == 0 || matches!(bytes.get(index - 1), Some(b'e' | b'E'))))
                })
                .last()
                .map_or(0, |(i, c)| i + c.len_utf8());
            let token = &self.rest[..length];
            self.rest = &self.rest[length..];
            let invalid = || format!("invalid host expression literal '{token}'");
            let (scalar, bits) = if let Some(n) = token.strip_suffix("u32") {
                (HostScalar::U32, n.parse::<u32>().map_err(|_| invalid())?)
            } else if let Some(n) = token.strip_suffix("f32") {
                let value = n.parse::<f32>().map_err(|_| invalid())?;
                if !value.is_finite() {
                    return Err(invalid());
                }
                (HostScalar::F32, value.to_bits())
            } else {
                let n = token.strip_suffix("i32").unwrap_or(token);
                (HostScalar::I32, n.parse::<i32>().map_err(|_| invalid())? as u32)
            };
            return Ok((HostExpression::Constant { scalar, bits }, scalar));
        }
        if self.consume("-") {
            let (value, scalar) = self.expression(3, depth)?;
            return Ok((
                HostExpression::Binary {
                    op: HostBinary::Subtract,
                    left: Box::new(HostExpression::Constant { scalar, bits: 0 }),
                    right: Box::new(value),
                },
                scalar,
            ));
        }
        let length = self.rest.bytes().take_while(|c| c.is_ascii_alphanumeric() || *c == b'_').count();
        let name = &self.rest[..length];
        self.rest = &self.rest[length..];
        if !identifier(name) {
            return Err("expected a literal, input name, cast, or '(' in host expression".into());
        }
        if self.consume("(") {
            let to = match name {
                "i32" => HostScalar::I32,
                "u32" => HostScalar::U32,
                "f32" => HostScalar::F32,
                _ => return Err(format!("unknown host expression cast '{name}'")),
            };
            let (value, _) = self.expression(0, depth)?;
            if !self.consume(")") {
                return Err("expected ')' after host expression cast".into());
            }
            Ok((
                HostExpression::Convert {
                    to,
                    value: Box::new(value),
                },
                to,
            ))
        } else {
            let input = self.inputs.get(name).ok_or_else(|| format!("unknown host input '{name}'"))?;
            Ok((
                HostExpression::Uniform {
                    name: name.into(),
                    set: input.set,
                    binding: input.binding,
                    offset: input.offset,
                    scalar: input.scalar,
                },
                input.scalar,
            ))
        }
    }
}
