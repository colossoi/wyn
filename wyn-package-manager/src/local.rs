use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use thiserror::Error;
use wyn_module_graph::{
    DependencyAlias, IdentityError, LocalSourceError, LocalSources, ModuleKey, PackageIdentity,
    PackagePlan, PackagePlanBuilder, PlanError, SourceFingerprint,
};

use crate::{Dependency, Manifest, ManifestError, PackageName, PackageVersion};

/// A closed local package plan and its confined filesystem source provider.
#[derive(Debug)]
pub struct LocalBuild {
    plan: PackagePlan,
    sources: LocalSources,
}

impl LocalBuild {
    pub fn into_parts(self) -> (PackagePlan, LocalSources) {
        (self.plan, self.sources)
    }
}

/// Read a root `wyn.toml` and every reachable local-path dependency.
pub fn load_local_build(root: impl AsRef<Path>) -> Result<LocalBuild, LocalBuildError> {
    let mut loader = LocalLoader::new();
    let root = loader.load(root.as_ref())?;
    loader.finish(root)
}

#[derive(Debug, Error)]
pub enum LocalBuildError {
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
        source: ManifestError,
    },
    #[error("dependency `{alias}` in `{package}` uses an absolute local path `{path}`")]
    AbsoluteDependencyPath {
        package: PackageName,
        alias: DependencyAlias,
        path: PathBuf,
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
    #[error("package `{package}` is supplied by both `{first}` and `{second}`")]
    ConflictingPackageRoots {
        package: PackageName,
        first: PathBuf,
        second: PathBuf,
    },
    #[error("invalid package identity: {0}")]
    Identity(#[from] IdentityError),
    #[error("invalid package plan: {0}")]
    Plan(#[from] PlanError),
    #[error("invalid local source root: {0}")]
    Sources(#[from] LocalSourceError),
}

#[derive(Debug)]
struct LoadedPackage {
    root: PathBuf,
    manifest: Manifest,
    dependencies: Vec<(DependencyAlias, usize)>,
}

struct LocalLoader {
    packages: Vec<LoadedPackage>,
    by_root: HashMap<PathBuf, usize>,
    by_name: HashMap<PackageName, usize>,
}

impl LocalLoader {
    fn new() -> Self {
        Self {
            packages: Vec::new(),
            by_root: HashMap::new(),
            by_name: HashMap::new(),
        }
    }

    fn load(&mut self, root: &Path) -> Result<usize, LocalBuildError> {
        let root = root.canonicalize().map_err(|source| LocalBuildError::ResolvePackage {
            path: root.to_owned(),
            source,
        })?;
        if let Some(&package) = self.by_root.get(&root) {
            return Ok(package);
        }

        let manifest_path = root.join("wyn.toml");
        let source =
            fs::read_to_string(&manifest_path).map_err(|source| LocalBuildError::ReadManifest {
                path: manifest_path.clone(),
                source,
            })?;
        let manifest = Manifest::parse(&source).map_err(|source| LocalBuildError::Manifest {
            path: manifest_path,
            source,
        })?;
        if let Some(&existing) = self.by_name.get(manifest.package()) {
            return Err(LocalBuildError::ConflictingPackageRoots {
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
    ) -> Result<usize, LocalBuildError> {
        if dependency.path().is_absolute() {
            return Err(LocalBuildError::AbsoluteDependencyPath {
                package: self.packages[package].manifest.package().clone(),
                alias: alias.clone(),
                path: dependency.path().to_owned(),
            });
        }
        let child = self.load(&root.join(dependency.path()))?;
        let actual = &self.packages[child].manifest;
        if actual.package() != dependency.package() {
            return Err(LocalBuildError::PackageMismatch {
                package: self.packages[package].manifest.package().clone(),
                alias: alias.clone(),
                expected: dependency.package().clone(),
                actual: actual.package().clone(),
            });
        }
        if !actual.version().satisfies(dependency.minimum()) {
            return Err(LocalBuildError::VersionMismatch {
                package: self.packages[package].manifest.package().clone(),
                alias: alias.clone(),
                minimum: dependency.minimum().clone(),
                actual: actual.version().clone(),
            });
        }
        Ok(child)
    }

    fn finish(self, root: usize) -> Result<LocalBuild, LocalBuildError> {
        let fingerprint = SourceFingerprint::new("local-path")?;
        let mut plan = PackagePlanBuilder::new();
        let mut ids = Vec::with_capacity(self.packages.len());
        for package in &self.packages {
            let identity = PackageIdentity::new(
                package.manifest.package().as_str(),
                package.manifest.version().to_string(),
                fingerprint.clone(),
            )?;
            ids.push(plan.add_package(identity, package.manifest.library().clone())?);
        }
        for (index, package) in self.packages.iter().enumerate() {
            for (alias, dependency) in &package.dependencies {
                plan.add_dependency(ids[index], alias.clone(), ids[*dependency])?;
            }
        }
        plan.set_root(ModuleKey::new(
            ids[root],
            self.packages[root].manifest.library().clone(),
        ))?;
        let plan = plan.build()?;

        let mut sources = LocalSources::new();
        for (id, package) in ids.into_iter().zip(self.packages) {
            sources.add_package_root(id, package.root)?;
        }
        Ok(LocalBuild { plan, sources })
    }
}

#[cfg(test)]
#[path = "local_tests.rs"]
mod local_tests;
