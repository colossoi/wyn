use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use thiserror::Error;
use wyn_base::IdArena;

use crate::{ModulePath, PackageId};

/// Error produced while constructing package identity data.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum IdentityError {
    #[error("package {field} is empty")]
    Empty {
        field: &'static str,
    },
}

/// Immutable source identity supplied by the package manager.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceFingerprint(Arc<str>);

impl SourceFingerprint {
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, IdentityError> {
        let value = value.into();
        if value.is_empty() {
            Err(IdentityError::Empty {
                field: "source fingerprint",
            })
        } else {
            Ok(Self(value))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Stable identity for one exact package release.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PackageIdentity {
    canonical_name: Arc<str>,
    version: Arc<str>,
    source_fingerprint: SourceFingerprint,
}

impl PackageIdentity {
    pub fn new(
        canonical_name: impl Into<Arc<str>>,
        version: impl Into<Arc<str>>,
        source_fingerprint: SourceFingerprint,
    ) -> Result<Self, IdentityError> {
        let canonical_name = canonical_name.into();
        if canonical_name.is_empty() {
            return Err(IdentityError::Empty {
                field: "canonical name",
            });
        }
        let version = version.into();
        if version.is_empty() {
            return Err(IdentityError::Empty { field: "version" });
        }
        Ok(Self {
            canonical_name,
            version,
            source_fingerprint,
        })
    }

    pub fn canonical_name(&self) -> &str {
        &self.canonical_name
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub const fn source_fingerprint(&self) -> &SourceFingerprint {
        &self.source_fingerprint
    }
}

/// Error produced while validating a package-local dependency alias.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AliasError {
    #[error("dependency alias is empty")]
    Empty,
    #[error("dependency alias contains the reserved character `{character}`")]
    ReservedCharacter {
        character: char,
    },
}

/// The package-local name used by source imports.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DependencyAlias(Arc<str>);

impl DependencyAlias {
    pub fn new(alias: impl Into<Arc<str>>) -> Result<Self, AliasError> {
        let alias = alias.into();
        if alias.is_empty() {
            return Err(AliasError::Empty);
        }
        if let Some(character) = alias
            .chars()
            .find(|character| character.is_whitespace() || matches!(character, '/' | '\\' | ':' | '\0'))
        {
            return Err(AliasError::ReservedCharacter { character });
        }
        Ok(Self(alias))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for DependencyAlias {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("DependencyAlias").field(&self.as_str()).finish()
    }
}

impl fmt::Display for DependencyAlias {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One package-local dependency edge from an alias to a selected package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dependency {
    alias: DependencyAlias,
    package: PackageId,
}

impl Dependency {
    pub const fn alias(&self) -> &DependencyAlias {
        &self.alias
    }

    pub const fn package(&self) -> PackageId {
        self.package
    }
}

/// One exact package in a closed compilation plan.
#[derive(Clone, Debug)]
pub struct Package {
    identity: PackageIdentity,
    library_root: ModulePath,
    dependencies: Vec<Dependency>,
}

impl Package {
    pub const fn identity(&self) -> &PackageIdentity {
        &self.identity
    }

    pub const fn library_root(&self) -> &ModulePath {
        &self.library_root
    }

    pub fn dependencies(&self) -> impl ExactSizeIterator<Item = &Dependency> {
        self.dependencies.iter()
    }

    pub(crate) fn dependency(&self, alias: &DependencyAlias) -> Option<PackageId> {
        self.dependencies.iter().find(|dependency| dependency.alias() == alias).map(Dependency::package)
    }
}

/// Stable key for a source module before a session-local `ModuleId` is assigned.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModuleKey {
    package: PackageId,
    path: ModulePath,
}

impl ModuleKey {
    pub const fn new(package: PackageId, path: ModulePath) -> Self {
        Self { package, path }
    }

    pub const fn package(&self) -> PackageId {
        self.package
    }

    pub const fn path(&self) -> &ModulePath {
        &self.path
    }
}

/// A validated, closed package plan consumed by module loading.
#[derive(Clone, Debug)]
pub struct PackagePlan {
    root: ModuleKey,
    packages: IdArena<PackageId, Package>,
}

impl PackagePlan {
    pub const fn root(&self) -> &ModuleKey {
        &self.root
    }

    pub fn package(&self, id: PackageId) -> Option<&Package> {
        self.packages.get(id)
    }

    pub fn packages(&self) -> impl Iterator<Item = (PackageId, &Package)> {
        self.packages.iter().map(|(&id, package)| (id, package))
    }
}

/// Error produced while assembling a closed package plan.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PlanError {
    #[error("package `{canonical_name}` already belongs to the plan")]
    DuplicatePackage {
        canonical_name: Arc<str>,
    },
    #[error("package ID {package:?} does not belong to this plan")]
    UnknownPackage {
        package: PackageId,
    },
    #[error("dependency alias `{alias}` already exists in package {package:?}")]
    DuplicateAlias {
        package: PackageId,
        alias: DependencyAlias,
    },
    #[error("the package plan has no root module")]
    MissingRoot,
}

/// Incrementally constructs and validates a deterministic package plan.
#[derive(Clone, Debug, Default)]
pub struct PackagePlanBuilder {
    root: Option<ModuleKey>,
    packages: IdArena<PackageId, Package>,
    names: HashMap<Arc<str>, PackageId>,
}

impl PackagePlanBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_package(
        &mut self,
        identity: PackageIdentity,
        library_root: ModulePath,
    ) -> Result<PackageId, PlanError> {
        if self.names.contains_key(identity.canonical_name()) {
            return Err(PlanError::DuplicatePackage {
                canonical_name: identity.canonical_name.clone(),
            });
        }

        let canonical_name = identity.canonical_name.clone();
        let id = self.packages.alloc(Package {
            identity,
            library_root,
            dependencies: Vec::new(),
        });
        self.names.insert(canonical_name, id);
        Ok(id)
    }

    pub fn add_dependency(
        &mut self,
        from: PackageId,
        alias: DependencyAlias,
        target: PackageId,
    ) -> Result<(), PlanError> {
        if self.packages.get(target).is_none() {
            return Err(PlanError::UnknownPackage { package: target });
        }
        let package = self.packages.get_mut(from).ok_or(PlanError::UnknownPackage { package: from })?;
        if package.dependencies.iter().any(|dependency| dependency.alias == alias) {
            return Err(PlanError::DuplicateAlias { package: from, alias });
        }
        package.dependencies.push(Dependency {
            alias,
            package: target,
        });
        Ok(())
    }

    pub fn set_root(&mut self, root: ModuleKey) -> Result<(), PlanError> {
        if self.packages.get(root.package()).is_none() {
            return Err(PlanError::UnknownPackage {
                package: root.package(),
            });
        }
        self.root = Some(root);
        Ok(())
    }

    pub fn build(self) -> Result<PackagePlan, PlanError> {
        let root = self.root.ok_or(PlanError::MissingRoot)?;
        Ok(PackagePlan {
            root,
            packages: self.packages,
        })
    }
}
