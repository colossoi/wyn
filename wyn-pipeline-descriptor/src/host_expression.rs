//! Host-known scalar expressions used for logical buffer capacities.
//! Integer arithmetic is checked: values that overflow the shader's scalar
//! width are rejected, never widened into a different shader computation.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostScalar {
    I32,
    U32,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostBinary {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
}

/// Leaves identify ABI bytes, not application names. Float literals use their
/// IEEE bits so the descriptor round-trips exactly (including signed zero).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HostExpression {
    Constant {
        scalar: HostScalar,
        bits: u32,
    },
    Uniform {
        set: u32,
        binding: u32,
        offset: u32,
        scalar: HostScalar,
    },
    Convert {
        to: HostScalar,
        value: Box<HostExpression>,
    },
    Binary {
        op: HostBinary,
        left: Box<HostExpression>,
        right: Box<HostExpression>,
    },
}

impl HostExpression {
    /// Evaluate with a lookup of a 32-bit word in the host's uniform snapshot.
    /// No GPU readback is needed. Missing bytes and invalid arithmetic are errors.
    pub fn evaluate(
        &self,
        uniform: &impl Fn(u32, u32, u32) -> Option<u32>,
    ) -> Result<(HostScalar, u32), String> {
        use HostScalar::*;
        Ok(match self {
            Self::Constant { scalar, bits } => (*scalar, *bits),
            Self::Uniform {
                set,
                binding,
                offset,
                scalar,
            } => (
                *scalar,
                uniform(*set, *binding, *offset)
                    .ok_or_else(|| format!("missing uniform word at {set}:{binding}+{offset}"))?,
            ),
            Self::Convert { to, value } => {
                let (from, bits) = value.evaluate(uniform)?;
                let bits = match (from, *to) {
                    (F32, I32 | U32) => {
                        let n = f32::from_bits(bits).trunc() as f64;
                        let (min, max) = if *to == I32 {
                            (i32::MIN as f64, i32::MAX as f64)
                        } else {
                            (0.0, u32::MAX as f64)
                        };
                        if !n.is_finite() || n < min || n > max {
                            return Err("allocation expression float conversion is out of range".into());
                        }
                        if *to == I32 {
                            (n as i32) as u32
                        } else {
                            n as u32
                        }
                    }
                    (I32, F32) => (bits as i32 as f32).to_bits(),
                    (U32, F32) => (bits as f32).to_bits(),
                    _ => bits,
                };
                (*to, bits)
            }
            Self::Binary { op, left, right } => {
                let (ty, a) = left.evaluate(uniform)?;
                let (other, b) = right.evaluate(uniform)?;
                if ty != other {
                    return Err("allocation expression operand types differ".into());
                }
                use HostBinary::*;
                let overflow = || "allocation expression overflows or divides by zero".to_owned();
                let bits = match ty {
                    I32 => {
                        let (a, b) = (a as i32, b as i32);
                        match op {
                            Add => a.checked_add(b),
                            Subtract => a.checked_sub(b),
                            Multiply => a.checked_mul(b),
                            Divide => a.checked_div(b),
                            Remainder => a.checked_rem(b),
                        }
                        .ok_or_else(overflow)? as u32
                    }
                    U32 => match op {
                        Add => a.checked_add(b),
                        Subtract => a.checked_sub(b),
                        Multiply => a.checked_mul(b),
                        Divide => a.checked_div(b),
                        Remainder => a.checked_rem(b),
                    }
                    .ok_or_else(overflow)?,
                    F32 => {
                        let (a, b) = (f32::from_bits(a), f32::from_bits(b));
                        let n = match op {
                            Add => a + b,
                            Subtract => a - b,
                            Multiply => a * b,
                            Divide => a / b,
                            Remainder => a % b,
                        };
                        if !n.is_finite() {
                            return Err("non-finite allocation expression".into());
                        }
                        n.to_bits()
                    }
                };
                (ty, bits)
            }
        })
    }

    pub fn element_count(&self, uniform: &impl Fn(u32, u32, u32) -> Option<u32>) -> Result<u64, String> {
        let (ty, bits) = self.evaluate(uniform)?;
        match ty {
            HostScalar::U32 => Ok(u64::from(bits)),
            HostScalar::I32 if bits as i32 >= 0 => Ok(u64::from(bits)),
            HostScalar::I32 => Err("negative allocation length".into()),
            HostScalar::F32 => Err("allocation length must be an integer".into()),
        }
    }
}
