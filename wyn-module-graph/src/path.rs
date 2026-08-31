use std::fmt;
use std::sync::Arc;

use thiserror::Error;

/// Error produced while validating or resolving a package-relative path.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PathError {
    #[error("module path is empty")]
    Empty,
    #[error("module path must be relative to its package")]
    Absolute,
    #[error("module path escapes its package root")]
    EscapesPackageRoot,
    #[error("module path contains a NUL byte")]
    NulByte,
}

/// A normalized UTF-8 source path rooted within one package.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModulePath(Arc<str>);

impl ModulePath {
    /// Normalize separators and dot segments, then validate package confinement.
    pub fn new(path: impl AsRef<str>) -> Result<Self, PathError> {
        let normalized = normalize(path.as_ref(), ParentPolicy::Confined)?;
        Ok(Self(normalized.into()))
    }

    /// Construct an import path, adding the `.wyn` source extension when omitted.
    pub fn from_import(path: impl AsRef<str>) -> Result<Self, PathError> {
        let normalized = normalize(path.as_ref(), ParentPolicy::Confined)?;
        Ok(Self(add_source_extension(normalized).into()))
    }

    /// Return the stable slash-separated representation.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Resolve a local import relative to this source file's parent directory.
    pub(crate) fn resolve(&self, relative: &RelativeModulePath) -> Result<Self, PathError> {
        let mut components: Vec<&str> = self.0.split('/').collect();
        components.pop();

        for component in relative.0.split('/') {
            if component == ".." {
                if components.pop().is_none() {
                    return Err(PathError::EscapesPackageRoot);
                }
            } else {
                components.push(component);
            }
        }

        Self::new(components.join("/"))
    }
}

impl fmt::Debug for ModulePath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("ModulePath").field(&self.as_str()).finish()
    }
}

impl fmt::Display for ModulePath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A normalized path interpreted relative to an importing source module.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RelativeModulePath(Arc<str>);

impl RelativeModulePath {
    /// Normalize separators and dot segments while retaining required leading
    /// parents for resolution against an importing file.
    pub fn new(path: impl AsRef<str>) -> Result<Self, PathError> {
        let normalized = normalize(path.as_ref(), ParentPolicy::RetainLeading)?;
        Ok(Self(normalized.into()))
    }

    /// Construct an import path, adding the `.wyn` source extension when omitted.
    pub fn from_import(path: impl AsRef<str>) -> Result<Self, PathError> {
        let normalized = normalize(path.as_ref(), ParentPolicy::RetainLeading)?;
        if normalized.rsplit('/').next() == Some("..") {
            return Err(PathError::Empty);
        }
        Ok(Self(add_source_extension(normalized).into()))
    }

    /// Return the stable slash-separated representation.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for RelativeModulePath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("RelativeModulePath").field(&self.as_str()).finish()
    }
}

impl fmt::Display for RelativeModulePath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Copy)]
enum ParentPolicy {
    Confined,
    RetainLeading,
}

fn normalize(path: &str, parent_policy: ParentPolicy) -> Result<String, PathError> {
    if path.contains('\0') {
        return Err(PathError::NulByte);
    }

    let path = path.replace('\\', "/");
    if path.starts_with('/') || has_windows_drive_prefix(&path) {
        return Err(PathError::Absolute);
    }

    let mut components = Vec::new();
    for component in path.split('/') {
        match component {
            "" | "." => {}
            ".." => match components.last().copied() {
                Some(previous) if previous != ".." => {
                    components.pop();
                }
                _ if matches!(parent_policy, ParentPolicy::RetainLeading) => {
                    components.push(component);
                }
                _ => return Err(PathError::EscapesPackageRoot),
            },
            _ => components.push(component),
        }
    }

    if components.is_empty() {
        return Err(PathError::Empty);
    }

    Ok(components.join("/"))
}

fn has_windows_drive_prefix(path: &str) -> bool {
    let bytes = path.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

fn add_source_extension(path: String) -> String {
    let final_component = path.rsplit('/').next().unwrap_or_default();
    if final_component.ends_with(".wyn") {
        path
    } else {
        format!("{path}.wyn")
    }
}
