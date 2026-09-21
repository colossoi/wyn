use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};

pub fn load_f32_json(path: &Path) -> Result<Vec<f32>> {
    let content =
        fs::read_to_string(path).with_context(|| format!("Failed to read: {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse JSON: {}", path.display()))?;
    let array = json.as_array().ok_or_else(|| anyhow!("JSON input must be an array"))?;
    array
        .iter()
        .enumerate()
        .map(|(i, v)| v.as_f64().map(|f| f as f32).ok_or_else(|| anyhow!("Element {} is not a number", i)))
        .collect()
}
