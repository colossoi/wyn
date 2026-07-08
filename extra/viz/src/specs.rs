//! CLI spec parser for `--push-constant` on the `pipeline` subcommand.

use anyhow::{anyhow, Result};

#[derive(Debug, Clone)]
pub struct PushConstantSpec {
    pub name: String,
    pub offset: u32,
    pub data: Vec<u8>,
}

impl PushConstantSpec {
    /// Parse from "name:type=value" format
    /// Examples: "n:i32=64", "header_base:u32x19=0,0,0,..."
    pub fn parse(spec: &str) -> Result<Self> {
        let (name_type, value) =
            spec.split_once('=').ok_or_else(|| anyhow!("Push constant spec must contain '=': {}", spec))?;
        let (name, ty) = name_type
            .split_once(':')
            .ok_or_else(|| anyhow!("Push constant spec must have format name:type=value: {}", spec))?;

        let data = parse_push_constant_value(ty, value)?;

        Ok(Self {
            name: name.to_string(),
            offset: 0, // filled in later
            data,
        })
    }

    pub fn byte_size(&self) -> u32 {
        self.data.len() as u32
    }
}

fn parse_push_constant_value(ty: &str, value: &str) -> Result<Vec<u8>> {
    // Check for array types like u32x19, i32x4, f32x3
    if let Some(rest) = ty.strip_prefix("u32x") {
        let count: usize = rest.parse().map_err(|_| anyhow!("Invalid array size: {}", rest))?;
        let values: Vec<u32> = value
            .split(',')
            .map(|v| {
                let v = v.trim();
                if let Some(hex) = v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
                    u32::from_str_radix(hex, 16).map_err(|e| anyhow!("Invalid hex u32: {}: {}", v, e))
                } else {
                    v.parse::<u32>().map_err(|e| anyhow!("Invalid u32: {}: {}", v, e))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        if values.len() != count {
            return Err(anyhow!(
                "Expected {} values for u32x{}, got {}",
                count,
                count,
                values.len()
            ));
        }
        return Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect());
    }

    if let Some(rest) = ty.strip_prefix("i32x") {
        let count: usize = rest.parse().map_err(|_| anyhow!("Invalid array size: {}", rest))?;
        let values: Vec<i32> = value
            .split(',')
            .map(|v| v.trim().parse::<i32>().map_err(|e| anyhow!("Invalid i32: {}: {}", v, e)))
            .collect::<Result<Vec<_>>>()?;
        if values.len() != count {
            return Err(anyhow!(
                "Expected {} values for i32x{}, got {}",
                count,
                count,
                values.len()
            ));
        }
        return Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect());
    }

    if let Some(rest) = ty.strip_prefix("f32x") {
        let count: usize = rest.parse().map_err(|_| anyhow!("Invalid array size: {}", rest))?;
        let values: Vec<f32> = value
            .split(',')
            .map(|v| v.trim().parse::<f32>().map_err(|e| anyhow!("Invalid f32: {}: {}", v, e)))
            .collect::<Result<Vec<_>>>()?;
        if values.len() != count {
            return Err(anyhow!(
                "Expected {} values for f32x{}, got {}",
                count,
                count,
                values.len()
            ));
        }
        return Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect());
    }

    match ty {
        "i32" => {
            let v: i32 = value.parse().map_err(|e| anyhow!("Invalid i32: {}: {}", value, e))?;
            Ok(v.to_le_bytes().to_vec())
        }
        "u32" => {
            let v: u32 = if let Some(hex) = value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
                u32::from_str_radix(hex, 16).map_err(|e| anyhow!("Invalid hex u32: {}: {}", value, e))?
            } else {
                value.parse::<u32>().map_err(|e| anyhow!("Invalid u32: {}: {}", value, e))?
            };
            Ok(v.to_le_bytes().to_vec())
        }
        "f32" => {
            let v: f32 = value.parse().map_err(|e| anyhow!("Invalid f32: {}: {}", value, e))?;
            Ok(v.to_le_bytes().to_vec())
        }
        _ => Err(anyhow!("Unknown push constant type: {}", ty)),
    }
}

/// CLI spec for `--uniform` on the `pipeline` subcommand: one member
/// value of a uniform block, placed via the descriptor's published
/// member layout.
#[derive(Debug, Clone)]
pub struct UniformSpec {
    /// Uniform binding name (the entry parameter name).
    pub name: String,
    /// Block member name; `None` addresses a bare scalar/vector
    /// uniform's whole value.
    pub member: Option<String>,
    pub data: Vec<u8>,
}

impl UniformSpec {
    /// Parse from "name.member:type=value" (or "name:type=value" for a
    /// bare scalar/vector uniform). Value syntax matches
    /// `--push-constant`: `c.radius:f32=0.35`, `c.tint:f32x2=0.9,0.2`.
    pub fn parse(spec: &str) -> Result<Self> {
        let (name_type, value) =
            spec.split_once('=').ok_or_else(|| anyhow!("Uniform spec must contain '=': {}", spec))?;
        let (path, ty) = name_type
            .split_once(':')
            .ok_or_else(|| anyhow!("Uniform spec must have format name[.member]:type=value: {}", spec))?;
        let (name, member) = match path.split_once('.') {
            Some((n, m)) => (n.to_string(), Some(m.to_string())),
            None => (path.to_string(), None),
        };
        let data = parse_push_constant_value(ty, value)?;
        Ok(Self { name, member, data })
    }
}
