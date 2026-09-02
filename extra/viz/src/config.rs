//! Runtime configuration sidecar for `viz pipeline`.
//!
//! The compiler-owned `<shader>.json` descriptor describes executable GPU
//! topology. The optional `<shader>.viz.json` file describes host policy that
//! is not part of the Wyn program, beginning with cross-frame feedback.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

pub const VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Sidecar {
    version: u32,
    #[serde(default)]
    feedback: Vec<FeedbackSpec>,
}

/// Feed one authored entry result back into one authored entry input on the
/// next frame. Neither selector depends on generated descriptor binding names.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeedbackSpec {
    pub entry: String,
    pub input: String,
    pub result: usize,
    #[serde(default)]
    pub initial: FeedbackInitial,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeedbackInitial {
    #[default]
    Zero,
    Rng,
    File {
        path: PathBuf,
    },
}

#[derive(Debug, Clone, Default)]
pub struct LoadedConfig {
    pub path: Option<PathBuf>,
    pub feedback: Vec<FeedbackSpec>,
}

pub fn automatic_path(shader_path: &Path) -> PathBuf {
    shader_path.with_extension("viz.json")
}

pub fn load(shader_path: &Path, explicit: Option<&Path>, disabled: bool) -> Result<LoadedConfig> {
    if disabled {
        return Ok(LoadedConfig::default());
    }

    let path = explicit.map(Path::to_path_buf).unwrap_or_else(|| automatic_path(shader_path));
    if explicit.is_none() && !path.is_file() {
        return Ok(LoadedConfig::default());
    }

    let text = fs::read_to_string(&path)
        .with_context(|| format!("failed to read viz config: {}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let feedback =
        parse(&text, base).with_context(|| format!("failed to parse viz config: {}", path.display()))?;
    Ok(LoadedConfig {
        path: Some(path),
        feedback,
    })
}

fn parse(text: &str, base: &Path) -> Result<Vec<FeedbackSpec>> {
    let mut sidecar: Sidecar = serde_json::from_str(text)?;
    if sidecar.version != VERSION {
        return Err(anyhow!(
            "unsupported viz config version {}; expected {}",
            sidecar.version,
            VERSION
        ));
    }

    let mut seen = HashSet::new();
    for spec in &mut sidecar.feedback {
        if spec.entry.is_empty() || spec.input.is_empty() {
            return Err(anyhow!("feedback entry and input must not be empty"));
        }
        if !seen.insert((spec.entry.clone(), spec.input.clone())) {
            return Err(anyhow!(
                "duplicate feedback destination '{}:{}'",
                spec.entry,
                spec.input
            ));
        }
        if let FeedbackInitial::File { path } = &mut spec.initial {
            if path.is_relative() {
                *path = base.join(&*path);
            }
        }
    }
    Ok(sidecar.feedback)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_source_level_feedback_and_resolves_relative_seed() {
        let feedback = parse(
            r#"{
                "version": 1,
                "feedback": [{
                    "entry": "pulse",
                    "input": "previous",
                    "result": 0,
                    "initial": { "kind": "file", "path": "seed.bin" }
                }]
            }"#,
            Path::new("examples"),
        )
        .expect("valid sidecar");

        assert_eq!(feedback.len(), 1);
        assert_eq!(feedback[0].entry, "pulse");
        assert_eq!(feedback[0].input, "previous");
        assert_eq!(feedback[0].result, 0);
        assert_eq!(
            feedback[0].initial,
            FeedbackInitial::File {
                path: PathBuf::from("examples").join("seed.bin")
            }
        );
    }

    #[test]
    fn rejects_unknown_versions_and_duplicate_destinations() {
        let bad_version = parse(r#"{"version": 2}"#, Path::new("."));
        assert!(bad_version.unwrap_err().to_string().contains("unsupported viz config version"));

        let duplicate = parse(
            r#"{
                "version": 1,
                "feedback": [
                    {"entry":"pulse","input":"previous","result":0},
                    {"entry":"pulse","input":"previous","result":1}
                ]
            }"#,
            Path::new("."),
        );
        assert!(duplicate.unwrap_err().to_string().contains("duplicate feedback destination"));
    }

    #[test]
    fn derives_sibling_sidecar_name() {
        assert_eq!(
            automatic_path(Path::new("tmp/particles.spv")),
            PathBuf::from("tmp/particles.viz.json")
        );
    }
}
