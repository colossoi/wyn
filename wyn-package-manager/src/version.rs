use std::fmt;
use std::str::FromStr;

use semver::Version;
use thiserror::Error;

/// A canonical Wyn package version written as `vMAJOR.MINOR.PATCH`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PackageVersion(Version);

impl PackageVersion {
    pub fn parse(value: &str) -> Result<Self, VersionError> {
        let Some(version) = value.strip_prefix('v') else {
            return Err(VersionError::MissingPrefix);
        };
        let version = Version::parse(version).map_err(VersionError::InvalidSemver)?;
        if !version.build.is_empty() {
            return Err(VersionError::BuildMetadata);
        }
        Ok(Self(version))
    }

    pub const fn major(&self) -> u64 {
        self.0.major
    }

    /// Whether this exact release can satisfy a minimum under Wyn's
    /// one-major-per-package rule.
    pub fn satisfies(&self, minimum: &Self) -> bool {
        self.major() == minimum.major() && self >= minimum
    }

    pub fn is_prerelease(&self) -> bool {
        !self.0.pre.is_empty()
    }
}

impl fmt::Display for PackageVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "v{}", self.0)
    }
}

impl FromStr for PackageVersion {
    type Err = VersionError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

#[derive(Debug, Error)]
pub enum VersionError {
    #[error("package versions must begin with `v`")]
    MissingPrefix,
    #[error("invalid semantic version: {0}")]
    InvalidSemver(semver::Error),
    #[error("package versions do not support build metadata")]
    BuildMetadata,
}

#[cfg(test)]
#[path = "version_tests.rs"]
mod version_tests;
