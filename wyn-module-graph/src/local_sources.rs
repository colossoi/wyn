use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use thiserror::Error;

use crate::{ModuleKey, PackageId, SourceReader};

/// Failure while reading a verified local package source tree.
#[derive(Debug, Error)]
pub enum LocalSourceError {
    #[error("package {package:?} already has a local source root")]
    DuplicatePackageRoot {
        package: PackageId,
    },
    #[error("source module {} already has an in-memory override", module.path())]
    DuplicateOverride {
        module: ModuleKey,
    },
    #[error("failed to resolve local package root: {source}")]
    PackageRoot {
        root: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("source module {} has no configured local package root", module.path())]
    MissingPackageRoot {
        module: ModuleKey,
    },
    #[error("failed to resolve source module: {source}")]
    ResolveModule {
        module: ModuleKey,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("source module resolves outside its local package root")]
    EscapesPackageRoot {
        module: ModuleKey,
        path: PathBuf,
        root: PathBuf,
    },
    #[error("failed to read source module: {source}")]
    ReadModule {
        module: ModuleKey,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

/// Local source trees with optional in-memory module overlays.
///
/// Package roots are canonicalized when registered. Each filesystem load is
/// canonicalized again and confined to its package root, including through
/// symbolic links. An exact in-memory override takes precedence and requires
/// no filesystem root, which supports editors and browser-hosted compilers.
#[derive(Debug, Default)]
pub struct LocalSources {
    package_roots: HashMap<PackageId, PathBuf>,
    overrides: HashMap<ModuleKey, Arc<str>>,
}

impl LocalSources {
    pub(crate) fn from_override(module: ModuleKey, source: impl Into<Arc<str>>) -> Self {
        let mut overrides = HashMap::new();
        overrides.insert(module, source.into());
        Self {
            package_roots: HashMap::new(),
            overrides,
        }
    }

    pub fn new() -> Self {
        Self::default()
    }

    /// Register the verified local root containing one package's source tree.
    pub fn add_package_root(
        &mut self,
        package: PackageId,
        root: impl AsRef<Path>,
    ) -> Result<(), LocalSourceError> {
        if self.package_roots.contains_key(&package) {
            return Err(LocalSourceError::DuplicatePackageRoot { package });
        }
        let root = root.as_ref();
        let canonical = root.canonicalize().map_err(|source| LocalSourceError::PackageRoot {
            root: root.to_owned(),
            source,
        })?;
        self.package_roots.insert(package, canonical);
        Ok(())
    }

    /// Supply source text for one exact module without consulting a filesystem.
    pub fn add_override(
        &mut self,
        module: ModuleKey,
        source: impl Into<Arc<str>>,
    ) -> Result<(), LocalSourceError> {
        if self.overrides.contains_key(&module) {
            return Err(LocalSourceError::DuplicateOverride { module });
        }
        self.overrides.insert(module, source.into());
        Ok(())
    }
}

impl SourceReader for LocalSources {
    type Error = LocalSourceError;

    fn load(&mut self, module: &ModuleKey) -> Result<Arc<str>, Self::Error> {
        if let Some(source) = self.overrides.get(module) {
            return Ok(source.clone());
        }

        let Some(root) = self.package_roots.get(&module.package()) else {
            return Err(LocalSourceError::MissingPackageRoot {
                module: module.clone(),
            });
        };
        let requested = root.join(module.path().as_str());
        let resolved = requested.canonicalize().map_err(|source| LocalSourceError::ResolveModule {
            module: module.clone(),
            path: requested,
            source,
        })?;
        if !resolved.starts_with(root) {
            return Err(LocalSourceError::EscapesPackageRoot {
                module: module.clone(),
                path: resolved,
                root: root.clone(),
            });
        }
        let source = std::fs::read_to_string(&resolved).map_err(|source| LocalSourceError::ReadModule {
            module: module.clone(),
            path: resolved,
            source,
        })?;
        Ok(Arc::from(source))
    }
}

#[cfg(test)]
#[path = "local_sources_tests.rs"]
mod local_sources_tests;
