use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use thiserror::Error;
use wyn_module_graph::{
    DependencyAlias, IdentityError, LocalSourceError, LocalSources, ModuleKey, ModulePath,
    PackageGraphBuilder, PackageGraphError, PackageIdentity, PackagePlan, PathError, SourceFingerprint,
};

use crate::materialize::{GitHubArchiveFetcher, GitHubRepository, PackageCache};
use crate::{Dependency, DependencySource, Manifest, ManifestError, PackageName, PackageVersion};

/// Prepare compiler inputs for the package rooted beside `wyn.toml`.
///
/// `root_module` selects a source module in the root package. When omitted,
/// the package manifest's library module is used. Every reachable dependency
/// must be materialized before this returns.
pub fn prepare_package(
    root: impl AsRef<Path>,
    root_module: Option<ModulePath>,
) -> Result<PackagePlan, PreparationError> {
    let mut preparer = PackagePreparer::new();
    let root = preparer.load_package(root.as_ref())?;
    let root_module = match root_module {
        Some(module) => module,
        None => preparer.packages[root].manifest.library().clone(),
    };
    preparer.finish(root, root_module)
}

/// Prepare compiler inputs for one source file outside a package.
pub fn prepare_standalone(source: impl AsRef<Path>) -> Result<PackagePlan, PreparationError> {
    let source = source.as_ref();
    let source = source.canonicalize().map_err(|error| PreparationError::ResolveStandaloneSource {
        path: source.to_owned(),
        source: error,
    })?;
    let Some(root) = source.parent() else {
        return Err(PreparationError::StandaloneSourceWithoutParent { path: source });
    };
    let Some(root_file) = source.file_name().and_then(|name| name.to_str()) else {
        return Err(PreparationError::NonUtf8StandaloneSource { path: source });
    };

    let root_file = ModulePath::new(root_file).map_err(|error| PreparationError::StandaloneSourcePath {
        path: source.clone(),
        source: error,
    })?;
    let fingerprint = SourceFingerprint::new("direct-local-source")?;
    let identity = PackageIdentity::new("direct/root", "v0.0.0", fingerprint)?;
    let mut packages = PackageGraphBuilder::new();
    let package = packages.add_package(identity, root_file.clone())?;
    packages.set_root(ModuleKey::new(package, root_file))?;

    let mut sources = LocalSources::new();
    sources.add_package_root(package, root)?;
    Ok(PackagePlan::new(packages.build()?, sources))
}

#[derive(Debug, Error)]
pub enum PreparationError {
    #[error("failed to resolve standalone source `{path}`: {source}")]
    ResolveStandaloneSource {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("standalone source `{path}` has no parent directory")]
    StandaloneSourceWithoutParent {
        path: PathBuf,
    },
    #[error("standalone source path `{path}` is not UTF-8")]
    NonUtf8StandaloneSource {
        path: PathBuf,
    },
    #[error("invalid standalone source path `{path}`: {source}")]
    StandaloneSourcePath {
        path: PathBuf,
        #[source]
        source: PathError,
    },
    #[error("failed to resolve local package directory `{path}`: {source}")]
    ResolvePackage {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to read package manifest `{path}`: {source}")]
    ReadManifest {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("invalid package manifest `{path}`: {source}")]
    Manifest {
        path: PathBuf,
        #[source]
        source: Box<ManifestError>,
    },
    #[error("dependency `{alias}` in `{package}` uses an absolute local path `{path}`")]
    AbsoluteDependencyPath {
        package: PackageName,
        alias: DependencyAlias,
        path: PathBuf,
    },
    #[error("failed to materialize dependency `{alias}` (`{dependency}`) of `{package}`: {detail}")]
    DependencyMaterialization {
        package: PackageName,
        alias: DependencyAlias,
        dependency: PackageName,
        detail: String,
    },
    #[error(
        "dependency `{alias}` in `{package}` declares `{expected}` but the local package is `{actual}`"
    )]
    PackageMismatch {
        package: PackageName,
        alias: DependencyAlias,
        expected: PackageName,
        actual: PackageName,
    },
    #[error("dependency `{alias}` in `{package}` requires {minimum}, but the local package is {actual}")]
    VersionMismatch {
        package: PackageName,
        alias: DependencyAlias,
        minimum: PackageVersion,
        actual: PackageVersion,
    },
    #[error(
        "dependency `{alias}` in `{package}` selected GitHub tag {selected}, but its manifest declares {actual}"
    )]
    GitHubTagVersionMismatch {
        package: PackageName,
        alias: DependencyAlias,
        selected: PackageVersion,
        actual: PackageVersion,
    },
    #[error("package `{package}` is supplied by both `{first}` and `{second}`")]
    ConflictingPackageRoots {
        package: PackageName,
        first: PathBuf,
        second: PathBuf,
    },
    #[error("invalid package identity: {0}")]
    Identity(#[from] IdentityError),
    #[error("invalid package plan: {0}")]
    PackageGraph(#[from] PackageGraphError),
    #[error("invalid local source root: {0}")]
    Sources(#[from] LocalSourceError),
}

#[derive(Debug)]
struct LoadedPackage {
    root: PathBuf,
    manifest: Manifest,
    dependencies: Vec<(DependencyAlias, usize)>,
}

struct PackagePreparer {
    packages: Vec<LoadedPackage>,
    by_root: HashMap<PathBuf, usize>,
    by_name: HashMap<PackageName, usize>,
    cache: PackageCache,
    github: GitHubArchiveFetcher,
}

impl PackagePreparer {
    fn new() -> Self {
        Self {
            packages: Vec::new(),
            by_root: HashMap::new(),
            by_name: HashMap::new(),
            cache: PackageCache::from_environment(),
            github: GitHubArchiveFetcher::new(),
        }
    }

    fn load_package(&mut self, root: &Path) -> Result<usize, PreparationError> {
        let root = root.canonicalize().map_err(|source| PreparationError::ResolvePackage {
            path: root.to_owned(),
            source,
        })?;
        if let Some(&package) = self.by_root.get(&root) {
            return Ok(package);
        }

        let manifest_path = root.join("wyn.toml");
        let source =
            fs::read_to_string(&manifest_path).map_err(|source| PreparationError::ReadManifest {
                path: manifest_path.clone(),
                source,
            })?;
        let manifest = Manifest::parse(&source).map_err(|source| PreparationError::Manifest {
            path: manifest_path,
            source: Box::new(source),
        })?;
        if let Some(&existing) = self.by_name.get(manifest.package()) {
            return Err(PreparationError::ConflictingPackageRoots {
                package: manifest.package().clone(),
                first: self.packages[existing].root.clone(),
                second: root,
            });
        }

        let package = self.packages.len();
        self.by_root.insert(root.clone(), package);
        self.by_name.insert(manifest.package().clone(), package);
        let dependencies: Vec<_> = manifest
            .dependencies()
            .map(|(alias, dependency)| (alias.clone(), dependency.clone()))
            .collect();
        self.packages.push(LoadedPackage {
            root: root.clone(),
            manifest,
            dependencies: Vec::with_capacity(dependencies.len()),
        });

        for (alias, dependency) in dependencies {
            let child = self.load_dependency(&root, package, &alias, &dependency)?;
            self.packages[package].dependencies.push((alias, child));
        }
        Ok(package)
    }

    fn load_dependency(
        &mut self,
        root: &Path,
        package: usize,
        alias: &DependencyAlias,
        dependency: &Dependency,
    ) -> Result<usize, PreparationError> {
        let child_root = self.materialize_dependency(root, package, alias, dependency)?;
        let child = self.load_package(&child_root)?;
        let actual = &self.packages[child].manifest;
        if actual.package() != dependency.package() {
            return Err(PreparationError::PackageMismatch {
                package: self.packages[package].manifest.package().clone(),
                alias: alias.clone(),
                expected: dependency.package().clone(),
                actual: actual.package().clone(),
            });
        }
        if matches!(dependency.source(), DependencySource::GitHub { .. })
            && actual.version() != dependency.minimum()
        {
            return Err(PreparationError::GitHubTagVersionMismatch {
                package: self.packages[package].manifest.package().clone(),
                alias: alias.clone(),
                selected: dependency.minimum().clone(),
                actual: actual.version().clone(),
            });
        }
        if !actual.version().satisfies(dependency.minimum()) {
            return Err(PreparationError::VersionMismatch {
                package: self.packages[package].manifest.package().clone(),
                alias: alias.clone(),
                minimum: dependency.minimum().clone(),
                actual: actual.version().clone(),
            });
        }
        Ok(child)
    }

    fn materialize_dependency(
        &mut self,
        root: &Path,
        package: usize,
        alias: &DependencyAlias,
        dependency: &Dependency,
    ) -> Result<PathBuf, PreparationError> {
        match dependency.source() {
            DependencySource::LocalPath(path) => {
                if path.is_absolute() {
                    return Err(PreparationError::AbsoluteDependencyPath {
                        package: self.packages[package].manifest.package().clone(),
                        alias: alias.clone(),
                        path: path.clone(),
                    });
                }
                Ok(root.join(path))
            }
            DependencySource::GitHub { repository } => {
                let materialized = GitHubRepository::parse(repository).and_then(|repository| {
                    let cache_key = repository.cache_key(dependency.minimum());
                    let github = &mut self.github;
                    self.cache.get_or_insert(&cache_key, |destination| {
                        github.fetch(&repository, dependency.minimum(), destination)
                    })
                });
                materialized.map_err(|error| PreparationError::DependencyMaterialization {
                    package: self.packages[package].manifest.package().clone(),
                    alias: alias.clone(),
                    dependency: dependency.package().clone(),
                    detail: error.to_string(),
                })
            }
        }
    }

    fn finish(self, root: usize, root_module: ModulePath) -> Result<PackagePlan, PreparationError> {
        let fingerprint = SourceFingerprint::new("local-path")?;
        let mut packages = PackageGraphBuilder::new();
        let mut ids = Vec::with_capacity(self.packages.len());
        for package in &self.packages {
            let identity = PackageIdentity::new(
                package.manifest.package().as_str(),
                package.manifest.version().to_string(),
                fingerprint.clone(),
            )?;
            ids.push(packages.add_package(identity, package.manifest.library().clone())?);
        }
        for (index, package) in self.packages.iter().enumerate() {
            for (alias, dependency) in &package.dependencies {
                packages.add_dependency(ids[index], alias.clone(), ids[*dependency])?;
            }
        }
        packages.set_root(ModuleKey::new(ids[root], root_module))?;
        let packages = packages.build()?;

        let mut sources = LocalSources::new();
        for (id, package) in ids.into_iter().zip(self.packages) {
            sources.add_package_root(id, package.root)?;
        }
        Ok(PackagePlan::new(packages, sources))
    }
}

#[cfg(test)]
#[path = "local_tests.rs"]
mod local_tests;
