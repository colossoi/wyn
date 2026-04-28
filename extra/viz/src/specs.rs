//! CLI spec parsers for storage buffers, push constants, and uniforms.
//!
//! These are the `--storage`, `--push-constant`, and `--uniform`
//! tokens from the `compute` and `run` subcommands; they describe how
//! to size, initialize, and encode each binding's bytes.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};

#[derive(Debug, Clone)]
pub struct StorageBufferSpec {
    pub set: u32,
    pub binding: u32,
    pub size_elements: u32,
    pub element_type: StorageElementType,
    /// Optional path to JSON file with initial data
    pub input_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy)]
pub enum StorageElementType {
    I32,
    U32,
    F32,
}

pub fn parse_storage_element_type(s: &str) -> Result<StorageElementType> {
    match s.to_lowercase().as_str() {
        "i32" => Ok(StorageElementType::I32),
        "u32" => Ok(StorageElementType::U32),
        "f32" => Ok(StorageElementType::F32),
        other => Err(anyhow!(
            "Unknown element type: {}. Expected i32, u32, or f32",
            other
        )),
    }
}

impl StorageBufferSpec {
    /// Parse one of:
    ///   `binding:size:type`                  (set defaults to 0)
    ///   `set:binding:size:type`              (explicit set)
    ///   `set:binding:size:type:input.json`   (explicit set, with input)
    ///
    /// The 3-part form keeps backward-compat with old call sites that
    /// pre-date the descriptor-set convention. To attach an input file,
    /// the spec must include an explicit `set:` prefix — the
    /// hypothetical 4-part `binding:size:type:input.json` form is
    /// rejected because it collides with `set:binding:size:type`.
    ///
    /// Examples:
    ///   "1:64:i32"             - set 0, binding 1, 64 i32s, zero-initialized
    ///   "0:1:64:i32"           - set 0, binding 1 (explicit)
    ///   "1:0:64:f32"           - set 1, binding 0
    ///   "0:0:8:f32:data.json"  - set 0, binding 0, 8 f32s from data.json
    pub fn parse(spec: &str) -> Result<Self> {
        let parts: Vec<&str> = spec.split(':').collect();
        let (set, binding, size, ty_str, input_file) = match parts.len() {
            3 => (0u32, parts[0], parts[1], parts[2], None),
            4 => {
                let set = parts[0].parse::<u32>().map_err(|_| anyhow!("Invalid set: {}", parts[0]))?;
                (set, parts[1], parts[2], parts[3], None)
            }
            5 => {
                let set = parts[0].parse::<u32>().map_err(|_| anyhow!("Invalid set: {}", parts[0]))?;
                (set, parts[1], parts[2], parts[3], Some(parts[4]))
            }
            _ => {
                return Err(anyhow!(
                    "Invalid storage buffer spec '{}'. Expected `binding:size:type`, `set:binding:size:type`, or `set:binding:size:type:input.json`",
                    spec
                ));
            }
        };

        let binding = binding.parse::<u32>().map_err(|_| anyhow!("Invalid binding: {}", binding))?;
        let size_elements = size.parse::<u32>().map_err(|_| anyhow!("Invalid size: {}", size))?;
        let element_type = parse_storage_element_type(ty_str)?;
        let input_file = input_file.map(PathBuf::from);

        Ok(Self {
            set,
            binding,
            size_elements,
            element_type,
            input_file,
        })
    }

    pub fn byte_size(&self) -> u64 {
        self.size_elements as u64 * 4
    }

    /// Load initial data from JSON file or return zeros
    pub fn load_initial_data(&self) -> Result<Vec<u8>> {
        match &self.input_file {
            Some(path) => {
                let content = fs::read_to_string(path)
                    .with_context(|| format!("Failed to read input file: {}", path.display()))?;
                let json: serde_json::Value = serde_json::from_str(&content)
                    .with_context(|| format!("Failed to parse JSON from: {}", path.display()))?;

                let array = json.as_array().ok_or_else(|| anyhow!("JSON input must be an array"))?;

                if array.len() != self.size_elements as usize {
                    return Err(anyhow!(
                        "JSON array has {} elements but buffer expects {}",
                        array.len(),
                        self.size_elements
                    ));
                }

                let mut bytes = Vec::with_capacity(self.byte_size() as usize);
                match self.element_type {
                    StorageElementType::I32 => {
                        for (i, val) in array.iter().enumerate() {
                            let n =
                                val.as_i64().ok_or_else(|| anyhow!("Element {} is not an integer", i))?
                                    as i32;
                            bytes.extend_from_slice(&n.to_le_bytes());
                        }
                    }
                    StorageElementType::U32 => {
                        for (i, val) in array.iter().enumerate() {
                            let n = val
                                .as_u64()
                                .ok_or_else(|| anyhow!("Element {} is not a positive integer", i))?
                                as u32;
                            bytes.extend_from_slice(&n.to_le_bytes());
                        }
                    }
                    StorageElementType::F32 => {
                        for (i, val) in array.iter().enumerate() {
                            let n = val.as_f64().ok_or_else(|| anyhow!("Element {} is not a number", i))?
                                as f32;
                            bytes.extend_from_slice(&n.to_le_bytes());
                        }
                    }
                }
                Ok(bytes)
            }
            None => Ok(vec![0u8; self.byte_size() as usize]),
        }
    }
}

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

    /// Assign sequential offsets across `specs` (each one immediately
    /// after the previous, no padding) and return the total byte
    /// size of the laid-out range. For the descriptor-driven path the
    /// offsets come from the descriptor; for the `compute` mode (no
    /// descriptor), the caller must lay them out before calling
    /// `build_push_constant_bytes`.
    pub fn lay_out_sequential(specs: &mut [PushConstantSpec]) -> u32 {
        let mut offset = 0u32;
        for spec in specs {
            spec.offset = offset;
            offset += spec.byte_size();
        }
        offset
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
                "Expected {} values for {}, got {}",
                count,
                ty,
                values.len()
            ));
        }
        Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
    } else if let Some(rest) = ty.strip_prefix("i32x") {
        let count: usize = rest.parse().map_err(|_| anyhow!("Invalid array size: {}", rest))?;
        let values: Vec<i32> = value
            .split(',')
            .map(|v| v.trim().parse::<i32>().map_err(|e| anyhow!("Invalid i32: {}: {}", v, e)))
            .collect::<Result<Vec<_>>>()?;
        if values.len() != count {
            return Err(anyhow!(
                "Expected {} values for {}, got {}",
                count,
                ty,
                values.len()
            ));
        }
        Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
    } else if let Some(rest) = ty.strip_prefix("f32x") {
        let count: usize = rest.parse().map_err(|_| anyhow!("Invalid array size: {}", rest))?;
        let values: Vec<f32> = value
            .split(',')
            .map(|v| v.trim().parse::<f32>().map_err(|e| anyhow!("Invalid f32: {}: {}", v, e)))
            .collect::<Result<Vec<_>>>()?;
        if values.len() != count {
            return Err(anyhow!(
                "Expected {} values for {}, got {}",
                count,
                ty,
                values.len()
            ));
        }
        Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
    } else {
        match ty {
            "i32" => {
                let v: i32 = value.parse().map_err(|e| anyhow!("Invalid i32 '{}': {}", value, e))?;
                Ok(v.to_le_bytes().to_vec())
            }
            "u32" => {
                let v = if let Some(hex) = value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
                    u32::from_str_radix(hex, 16)
                        .map_err(|e| anyhow!("Invalid hex u32: {}: {}", value, e))?
                } else {
                    value.parse::<u32>().map_err(|e| anyhow!("Invalid u32: {}: {}", value, e))?
                };
                Ok(v.to_le_bytes().to_vec())
            }
            "f32" => {
                let v: f32 = value.parse().map_err(|e| anyhow!("Invalid f32 '{}': {}", value, e))?;
                Ok(v.to_le_bytes().to_vec())
            }
            other => Err(anyhow!(
                "Unknown push constant type: {}. Use i32, u32, f32, u32xN, i32xN, f32xN",
                other
            )),
        }
    }
}

/// Uniform buffer specification parsed from CLI.
///
/// Format: `set:binding:type=value[,value...]`. The type is required —
/// uniforms aren't sized like storage buffers; the type tells us how
/// many bytes to allocate and how to encode the value.
#[derive(Debug, Clone)]
pub struct UniformSpec {
    pub set: u32,
    pub binding: u32,
    /// Encoded little-endian bytes of the value, padded to the WGSL
    /// std140-ish layout where vec3 occupies 16 bytes.
    pub data: Vec<u8>,
}

impl UniformSpec {
    /// Parse `set:binding:type=value[,value...]`.
    ///
    /// Examples:
    ///   "1:0:f32=0.5"
    ///   "1:1:vec3f32=1920,1080,1"
    ///   "0:2:i32=42"
    pub fn parse(spec: &str) -> Result<Self> {
        let (head, value) =
            spec.split_once('=').ok_or_else(|| anyhow!("Uniform spec must contain '=': {}", spec))?;
        let head_parts: Vec<&str> = head.split(':').collect();
        if head_parts.len() != 3 {
            return Err(anyhow!(
                "Uniform spec head must be set:binding:type, got '{}'",
                head
            ));
        }
        let set = head_parts[0].parse::<u32>().map_err(|_| anyhow!("Invalid set: {}", head_parts[0]))?;
        let binding =
            head_parts[1].parse::<u32>().map_err(|_| anyhow!("Invalid binding: {}", head_parts[1]))?;
        let ty = head_parts[2];
        let data = encode_uniform_value(ty, value)?;
        Ok(Self { set, binding, data })
    }
}

/// Encode a uniform value to little-endian bytes per WGSL std140-ish
/// layout: scalars are 4 bytes; vec2 is 8; vec3 pads to 16; vec4 is 16.
fn encode_uniform_value(ty: &str, value: &str) -> Result<Vec<u8>> {
    let (base, count, pad_to_16): (&str, usize, bool) = match ty.to_lowercase().as_str() {
        "i32" | "u32" | "f32" => (ty, 1, false),
        "vec2i32" | "vec2u32" | "vec2f32" => (&ty[4..], 2, false),
        "vec3i32" | "vec3u32" | "vec3f32" => (&ty[4..], 3, true),
        "vec4i32" | "vec4u32" | "vec4f32" => (&ty[4..], 4, false),
        other => {
            return Err(anyhow!(
                "Unknown uniform type '{}'. Expected i32/u32/f32 or vec[2-4]<…>",
                other
            ));
        }
    };
    let parts: Vec<&str> = value.split(',').map(|s| s.trim()).collect();
    if parts.len() != count {
        return Err(anyhow!(
            "Uniform type {} expects {} value(s), got {}",
            ty,
            count,
            parts.len()
        ));
    }
    let mut bytes = Vec::with_capacity(count * 4);
    for v in parts {
        let four = match base.to_lowercase().as_str() {
            "i32" => v.parse::<i32>().map_err(|e| anyhow!("Invalid i32 '{}': {}", v, e))?.to_le_bytes(),
            "u32" => v.parse::<u32>().map_err(|e| anyhow!("Invalid u32 '{}': {}", v, e))?.to_le_bytes(),
            "f32" => v.parse::<f32>().map_err(|e| anyhow!("Invalid f32 '{}': {}", v, e))?.to_le_bytes(),
            _ => unreachable!(),
        };
        bytes.extend_from_slice(&four);
    }
    // vec3 occupies 16 bytes per WGSL std140 / SPIR-V vec3 layout.
    if pad_to_16 {
        bytes.extend_from_slice(&[0u8; 4]);
    }
    Ok(bytes)
}
