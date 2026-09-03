use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};

use flate2::read::GzDecoder;
use thiserror::Error;

use crate::PackageVersion;

const CACHE_ENVIRONMENT_VARIABLE: &str = "WYN_PKG_CACHE";
const HTTP_TIMEOUT_SECONDS: u64 = 60;
const MAX_ARCHIVE_ENTRIES: usize = 100_000;
const MAX_UNPACKED_BYTES: u64 = 512 * 1024 * 1024;

static STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Failure while obtaining an unpacked source package.
#[derive(Debug, Error)]
pub(crate) enum MaterializationError {
    #[error(
        "unsupported GitHub repository specifier `{repository}`; expected `github.com/OWNER/REPOSITORY`"
    )]
    UnsupportedRepository {
        repository: String,
    },
    #[error(
        "no package cache directory is available; set {CACHE_ENVIRONMENT_VARIABLE} to an absolute directory"
    )]
    CacheUnavailable,
    #[error("package cache directory `{path}` must be absolute")]
    RelativeCache {
        path: PathBuf,
    },
    #[error("failed to create package cache directory `{path}`: {source}")]
    CreateCache {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("cached package `{path}` is incomplete")]
    IncompleteCache {
        path: PathBuf,
    },
    #[error("failed to download GitHub package archive `{url}`: {detail}")]
    Download {
        url: String,
        detail: String,
    },
    #[error("GitHub package archive `{url}` returned HTTP status {status}")]
    HttpStatus {
        url: String,
        status: u16,
    },
    #[error("failed to create package staging directory `{path}`: {source}")]
    CreateStaging {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to read GitHub package archive from `{repository}`: {source}")]
    ReadArchive {
        repository: String,
        #[source]
        source: io::Error,
    },
    #[error("GitHub package archive from `{repository}` contains an unsafe path `{path}`")]
    UnsafeArchivePath {
        repository: String,
        path: PathBuf,
    },
    #[error("GitHub package archive from `{repository}` contains multiple top-level directories")]
    MultipleArchiveRoots {
        repository: String,
    },
    #[error("GitHub package archive from `{repository}` contains too many entries")]
    TooManyArchiveEntries {
        repository: String,
    },
    #[error("GitHub package archive from `{repository}` expands beyond 512 MiB")]
    ArchiveTooLarge {
        repository: String,
    },
    #[error("GitHub package archive from `{repository}` contains unsupported entry `{path}`")]
    UnsupportedArchiveEntry {
        repository: String,
        path: PathBuf,
    },
    #[error("GitHub package archive from `{repository}` has no package directory")]
    MissingArchiveRoot {
        repository: String,
    },
    #[error("materialized package from `{repository}` has no `wyn.toml` at its root")]
    MissingManifest {
        repository: String,
    },
    #[error("failed to install materialized package at `{path}`: {source}")]
    Install {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

pub(crate) struct HttpResponse {
    status: u16,
    body: Box<dyn Read>,
}

impl HttpResponse {
    #[cfg(test)]
    pub(crate) fn new(status: u16, body: impl Read + 'static) -> Self {
        Self {
            status,
            body: Box::new(body),
        }
    }
}

pub(crate) trait HttpClient {
    fn get(&mut self, url: &str) -> Result<HttpResponse, String>;
}

struct MinreqHttpClient;

impl HttpClient for MinreqHttpClient {
    fn get(&mut self, url: &str) -> Result<HttpResponse, String> {
        let response = minreq::get(url)
            .with_header("User-Agent", "wyn-package-manager/0.1")
            .with_timeout(HTTP_TIMEOUT_SECONDS)
            .with_max_redirects(5)
            .send_lazy()
            .map_err(|error| error.to_string())?;
        Ok(HttpResponse {
            status: response.status_code,
            body: Box::new(response),
        })
    }
}

/// Source-agnostic storage for unpacked package source trees.
pub(crate) struct PackageCache {
    root_override: Option<PathBuf>,
}

impl PackageCache {
    pub(crate) fn from_environment() -> Self {
        Self { root_override: None }
    }

    #[cfg(test)]
    pub(crate) fn at(root: PathBuf) -> Self {
        Self {
            root_override: Some(root),
        }
    }

    pub(crate) fn get_or_insert(
        &self,
        key: &Path,
        fetch: impl FnOnce(&Path) -> Result<PathBuf, MaterializationError>,
    ) -> Result<PathBuf, MaterializationError> {
        let cache_root = match &self.root_override {
            Some(root) => root.clone(),
            None => default_cache_root()?,
        };
        let destination = cache_root.join(key);
        if destination.is_dir() {
            if destination.join("wyn.toml").is_file() {
                return Ok(destination);
            }
            return Err(MaterializationError::IncompleteCache { path: destination });
        }

        let Some(parent) = destination.parent() else {
            return Err(MaterializationError::CacheUnavailable);
        };
        fs::create_dir_all(parent).map_err(|source| MaterializationError::CreateCache {
            path: parent.to_owned(),
            source,
        })?;
        let staging = StagingDirectory::create(parent)?;
        let source_root = fetch(staging.path())?;
        if !source_root.join("wyn.toml").is_file() {
            return Err(MaterializationError::MissingManifest {
                repository: source_root.display().to_string(),
            });
        }
        match fs::rename(&source_root, &destination) {
            Ok(()) => Ok(destination),
            Err(_) if destination.join("wyn.toml").is_file() => Ok(destination),
            Err(source) => Err(MaterializationError::Install {
                path: destination,
                source,
            }),
        }
    }
}

/// Fetches and unpacks GitHub's generated archive for a repository tag.
pub(crate) struct GitHubArchiveFetcher {
    http: Box<dyn HttpClient>,
}

impl GitHubArchiveFetcher {
    pub(crate) fn new() -> Self {
        Self {
            http: Box::new(MinreqHttpClient),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_client(http: impl HttpClient + 'static) -> Self {
        Self { http: Box::new(http) }
    }

    pub(crate) fn fetch(
        &mut self,
        repository: &GitHubRepository,
        version: &PackageVersion,
        destination: &Path,
    ) -> Result<PathBuf, MaterializationError> {
        let url = repository.archive_url(version);
        let response = self.http.get(&url).map_err(|detail| MaterializationError::Download {
            url: url.clone(),
            detail,
        })?;
        if response.status != 200 {
            return Err(MaterializationError::HttpStatus {
                url,
                status: response.status,
            });
        }
        unpack_archive(response.body, destination, repository.original())
    }
}

pub(crate) struct GitHubRepository {
    original: String,
    owner: String,
    repository: String,
}

impl GitHubRepository {
    pub(crate) fn parse(value: &str) -> Result<Self, MaterializationError> {
        const PREFIX: &str = "github.com/";
        let Some(path) = value.strip_prefix(PREFIX) else {
            return Err(MaterializationError::UnsupportedRepository {
                repository: value.to_owned(),
            });
        };
        let Some((owner, repository)) = path.split_once('/') else {
            return Err(MaterializationError::UnsupportedRepository {
                repository: value.to_owned(),
            });
        };
        if owner.is_empty()
            || repository.is_empty()
            || repository.contains('/')
            || repository.ends_with(".git")
            || !valid_github_component(owner)
            || !valid_github_component(repository)
        {
            return Err(MaterializationError::UnsupportedRepository {
                repository: value.to_owned(),
            });
        }
        Ok(Self {
            original: value.to_owned(),
            owner: owner.to_ascii_lowercase(),
            repository: repository.to_ascii_lowercase(),
        })
    }

    fn original(&self) -> &str {
        &self.original
    }

    fn archive_url(&self, version: &PackageVersion) -> String {
        format!(
            "https://github.com/{}/{}/archive/refs/tags/{version}.tar.gz",
            self.owner, self.repository
        )
    }

    pub(crate) fn cache_key(&self, version: &PackageVersion) -> PathBuf {
        PathBuf::from("github.com").join(&self.owner).join(&self.repository).join(version.to_string())
    }
}

fn valid_github_component(component: &str) -> bool {
    component != "."
        && component != ".."
        && component
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.'))
}

fn default_cache_root() -> Result<PathBuf, MaterializationError> {
    if let Some(root) = nonempty_environment_path(CACHE_ENVIRONMENT_VARIABLE) {
        return absolute_cache_root(root);
    }

    #[cfg(target_os = "windows")]
    if let Some(root) = nonempty_environment_path("LOCALAPPDATA") {
        return absolute_cache_root(root.join("wyn").join("packages"));
    }

    #[cfg(target_os = "macos")]
    if let Some(root) = nonempty_environment_path("HOME") {
        return absolute_cache_root(root.join("Library").join("Caches").join("wyn").join("packages"));
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        if let Some(root) = nonempty_environment_path("XDG_CACHE_HOME") {
            return absolute_cache_root(root.join("wyn").join("packages"));
        }
        if let Some(root) = nonempty_environment_path("HOME") {
            return absolute_cache_root(root.join(".cache").join("wyn").join("packages"));
        }
    }

    Err(MaterializationError::CacheUnavailable)
}

fn absolute_cache_root(path: PathBuf) -> Result<PathBuf, MaterializationError> {
    if path.is_absolute() {
        Ok(path)
    } else {
        Err(MaterializationError::RelativeCache { path })
    }
}

fn nonempty_environment_path(name: &str) -> Option<PathBuf> {
    env::var_os(name).filter(|value| !value.is_empty()).map(PathBuf::from)
}

fn unpack_archive(
    body: Box<dyn Read>,
    staging: &Path,
    repository: &str,
) -> Result<PathBuf, MaterializationError> {
    let decoder = GzDecoder::new(body);
    let mut archive = tar::Archive::new(decoder);
    let entries = archive.entries().map_err(|source| MaterializationError::ReadArchive {
        repository: repository.to_owned(),
        source,
    })?;
    let mut root: Option<OsString> = None;
    let mut entry_count = 0usize;
    let mut unpacked_bytes = 0u64;

    for entry in entries {
        entry_count += 1;
        if entry_count > MAX_ARCHIVE_ENTRIES {
            return Err(MaterializationError::TooManyArchiveEntries {
                repository: repository.to_owned(),
            });
        }
        let mut entry = entry.map_err(|source| MaterializationError::ReadArchive {
            repository: repository.to_owned(),
            source,
        })?;
        let entry_type = entry.header().entry_type();
        let size = entry.header().size().map_err(|source| MaterializationError::ReadArchive {
            repository: repository.to_owned(),
            source,
        })?;
        let Some(total) = unpacked_bytes.checked_add(size) else {
            return Err(MaterializationError::ArchiveTooLarge {
                repository: repository.to_owned(),
            });
        };
        unpacked_bytes = total;
        if unpacked_bytes > MAX_UNPACKED_BYTES {
            return Err(MaterializationError::ArchiveTooLarge {
                repository: repository.to_owned(),
            });
        }
        if is_archive_metadata(entry_type) {
            continue;
        }

        let path = entry
            .path()
            .map_err(|source| MaterializationError::ReadArchive {
                repository: repository.to_owned(),
                source,
            })?
            .into_owned();
        let Some(archive_root) = safe_archive_root(&path) else {
            return Err(MaterializationError::UnsafeArchivePath {
                repository: repository.to_owned(),
                path,
            });
        };
        match &root {
            Some(root) if root != &archive_root => {
                return Err(MaterializationError::MultipleArchiveRoots {
                    repository: repository.to_owned(),
                });
            }
            Some(_) => {}
            None => root = Some(archive_root),
        }

        if !entry_type.is_file() && !entry_type.is_dir() {
            return Err(MaterializationError::UnsupportedArchiveEntry {
                repository: repository.to_owned(),
                path,
            });
        }
        let unpacked = entry.unpack_in(staging).map_err(|source| MaterializationError::ReadArchive {
            repository: repository.to_owned(),
            source,
        })?;
        if !unpacked {
            return Err(MaterializationError::UnsafeArchivePath {
                repository: repository.to_owned(),
                path,
            });
        }
    }

    let Some(root) = root else {
        return Err(MaterializationError::MissingArchiveRoot {
            repository: repository.to_owned(),
        });
    };
    let root = staging.join(root);
    if !root.is_dir() {
        return Err(MaterializationError::MissingArchiveRoot {
            repository: repository.to_owned(),
        });
    }
    Ok(root)
}

fn is_archive_metadata(entry_type: tar::EntryType) -> bool {
    entry_type.is_pax_global_extensions()
        || entry_type.is_pax_local_extensions()
        || entry_type.is_gnu_longname()
        || entry_type.is_gnu_longlink()
}

fn safe_archive_root(path: &Path) -> Option<OsString> {
    let mut root = None;
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(component) => {
                if root.is_none() {
                    root = Some(component.to_owned());
                }
            }
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    root
}

struct StagingDirectory {
    path: PathBuf,
}

impl StagingDirectory {
    fn create(parent: &Path) -> Result<Self, MaterializationError> {
        for _ in 0..32 {
            let sequence = STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(".tmp-{}-{sequence}", process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(source) => {
                    return Err(MaterializationError::CreateStaging { path, source });
                }
            }
        }
        let path = parent.join(format!(".tmp-{}", process::id()));
        Err(MaterializationError::CreateStaging {
            path,
            source: io::Error::new(
                io::ErrorKind::AlreadyExists,
                "could not allocate a unique staging directory",
            ),
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for StagingDirectory {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.path) {
            if error.kind() != io::ErrorKind::NotFound {
                eprintln!(
                    "failed to remove package staging directory `{}`: {error}",
                    self.path.display()
                );
            }
        }
    }
}

#[cfg(test)]
#[path = "materialize_tests.rs"]
mod materialize_tests;
