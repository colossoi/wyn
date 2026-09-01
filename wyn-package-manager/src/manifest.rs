use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use thiserror::Error;
use wyn_module_graph::{AliasError, DependencyAlias, ModulePath, PathError};

use crate::{PackageVersion, VersionError};

/// A validated canonical ecosystem package name.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PackageName(Arc<str>);

impl PackageName {
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, PackageNameError> {
        let value = value.into();
        if value.is_empty() {
            return Err(PackageNameError::Empty);
        }
        if value.starts_with('/') || value.ends_with('/') {
            return Err(PackageNameError::EmptyComponent);
        }
        for component in value.split('/') {
            if component.is_empty() {
                return Err(PackageNameError::EmptyComponent);
            }
            if let Some(character) = component.chars().find(|character| {
                !character.is_ascii_lowercase()
                    && !character.is_ascii_digit()
                    && !matches!(character, '-' | '_' | '.')
            }) {
                return Err(PackageNameError::InvalidCharacter { character });
            }
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PackageName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PackageNameError {
    #[error("package name is empty")]
    Empty,
    #[error("package name contains an empty path component")]
    EmptyComponent,
    #[error("package name contains the invalid character `{character}`")]
    InvalidCharacter {
        character: char,
    },
}

/// One local-path dependency declared by a package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dependency {
    package: PackageName,
    minimum: PackageVersion,
    path: PathBuf,
}

impl Dependency {
    pub const fn package(&self) -> &PackageName {
        &self.package
    }

    pub const fn minimum(&self) -> &PackageVersion {
        &self.minimum
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Strict, validated `wyn.toml` data for the local-only implementation stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Manifest {
    package: PackageName,
    version: PackageVersion,
    minimum_wyn: PackageVersion,
    library: ModulePath,
    dependencies: BTreeMap<DependencyAlias, Dependency>,
}

impl Manifest {
    pub fn parse(source: &str) -> Result<Self, ManifestError> {
        let raw: RawManifest = toml::from_str(source).map_err(ManifestError::Toml)?;
        if raw.manifest_version != 1 {
            return Err(ManifestError::UnsupportedFormat {
                version: raw.manifest_version,
            });
        }
        let package = PackageName::new(raw.package.name).map_err(ManifestError::PackageName)?;
        let version =
            PackageVersion::parse(&raw.package.version).map_err(|source| ManifestError::Version {
                field: "package.version",
                source,
            })?;
        let minimum_wyn =
            PackageVersion::parse(&raw.package.wyn).map_err(|source| ManifestError::Version {
                field: "package.wyn",
                source,
            })?;
        let library = ModulePath::new(raw.package.library).map_err(ManifestError::LibraryPath)?;
        let mut dependencies = BTreeMap::new();
        for (alias, raw_dependency) in raw.dependencies {
            let alias = DependencyAlias::new(alias).map_err(ManifestError::DependencyAlias)?;
            let package = PackageName::new(raw_dependency.package).map_err(ManifestError::PackageName)?;
            let minimum = PackageVersion::parse(&raw_dependency.version).map_err(|source| {
                ManifestError::Version {
                    field: "dependencies.version",
                    source,
                }
            })?;
            dependencies.insert(
                alias,
                Dependency {
                    package,
                    minimum,
                    path: raw_dependency.path,
                },
            );
        }
        Ok(Self {
            package,
            version,
            minimum_wyn,
            library,
            dependencies,
        })
    }

    pub const fn package(&self) -> &PackageName {
        &self.package
    }

    pub const fn version(&self) -> &PackageVersion {
        &self.version
    }

    pub const fn minimum_wyn(&self) -> &PackageVersion {
        &self.minimum_wyn
    }

    pub const fn library(&self) -> &ModulePath {
        &self.library
    }

    pub fn dependencies(&self) -> impl ExactSizeIterator<Item = (&DependencyAlias, &Dependency)> {
        self.dependencies.iter()
    }
}

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("invalid TOML manifest: {0}")]
    Toml(toml::de::Error),
    #[error("unsupported manifest version {version}; expected 1")]
    UnsupportedFormat {
        version: u32,
    },
    #[error("invalid package name: {0}")]
    PackageName(PackageNameError),
    #[error("invalid {field}: {source}")]
    Version {
        field: &'static str,
        #[source]
        source: VersionError,
    },
    #[error("invalid library path: {0}")]
    LibraryPath(PathError),
    #[error("invalid dependency alias: {0}")]
    DependencyAlias(AliasError),
}

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
struct RawManifest {
    manifest_version: u32,
    package: RawPackage,
    #[serde(default)]
    dependencies: BTreeMap<String, RawDependency>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPackage {
    name: String,
    version: String,
    wyn: String,
    library: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDependency {
    package: String,
    version: String,
    path: PathBuf,
}

#[cfg(test)]
#[path = "manifest_tests.rs"]
mod manifest_tests;
