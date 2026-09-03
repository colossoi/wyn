use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use flate2::write::GzEncoder;
use flate2::Compression;

use super::{
    absolute_cache_root, GitHubArchiveFetcher, GitHubRepository, HttpClient, HttpResponse,
    MaterializationError, PackageCache,
};
use crate::PackageVersion;

struct TestCache {
    root: PathBuf,
}

impl TestCache {
    fn new() -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should follow the Unix epoch")
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("wyn_materialize_test_{}_{}", std::process::id(), unique));
        fs::create_dir(&root).expect("test cache should be created");
        Self { root }
    }
}

impl Drop for TestCache {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root) {
            eprintln!("failed to remove test cache `{}`: {error}", self.root.display());
        }
    }
}

struct FakeHttpClient {
    status: u16,
    archive: Vec<u8>,
    requests: Arc<AtomicUsize>,
}

impl HttpClient for FakeHttpClient {
    fn get(&mut self, _url: &str) -> Result<HttpResponse, String> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        Ok(HttpResponse::new(self.status, Cursor::new(self.archive.clone())))
    }
}

fn package_archive(root: &str, files: &[(&str, &str)]) -> Vec<u8> {
    let encoder = GzEncoder::new(Vec::new(), Compression::default());
    let mut archive = tar::Builder::new(encoder);
    let metadata = b"18 comment=github\n";
    let mut metadata_header = tar::Header::new_gnu();
    metadata_header.set_entry_type(tar::EntryType::XGlobalHeader);
    metadata_header.set_size(metadata.len() as u64);
    metadata_header.set_mode(0o644);
    metadata_header.set_cksum();
    archive
        .append_data(&mut metadata_header, "pax_global_header", &metadata[..])
        .expect("test archive metadata should be written");
    for (path, source) in files {
        let bytes = source.as_bytes();
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        archive
            .append_data(&mut header, format!("{root}/{path}"), bytes)
            .expect("test archive entry should be written");
    }
    let encoder = archive.into_inner().expect("test tar archive should finish");
    encoder.finish().expect("test gzip archive should finish")
}

fn version(value: &str) -> PackageVersion {
    PackageVersion::parse(value).expect("test package version should be valid")
}

#[test]
fn github_repository_maps_tag_to_archive_and_cache_paths() {
    let repository =
        GitHubRepository::parse("github.com/Example/Noise").expect("GitHub repository should be accepted");
    let version = version("v1.2.3");

    assert_eq!(
        repository.archive_url(&version),
        "https://github.com/example/noise/archive/refs/tags/v1.2.3.tar.gz"
    );
    assert_eq!(
        repository.cache_key(&version),
        Path::new("github.com").join("example").join("noise").join("v1.2.3")
    );
}

#[test]
fn materialization_unpacks_package_and_reuses_completed_cache() {
    let cache = TestCache::new();
    let requests = Arc::new(AtomicUsize::new(0));
    let archive = package_archive(
        "noise-v1.2.3",
        &[
            (
                "wyn.toml",
                concat!(
                    "manifest-version = 1\n",
                    "[package]\n",
                    "name = \"example/noise\"\n",
                    "version = \"v1.2.3\"\n",
                    "wyn = \"v0.1.0\"\n",
                    "library = \"src/lib.wyn\"\n",
                ),
            ),
            ("src/lib.wyn", "def answer: i32 = 42\n"),
        ],
    );
    let http = FakeHttpClient {
        status: 200,
        archive,
        requests: requests.clone(),
    };
    let package_cache = PackageCache::at(cache.root.clone());
    let mut github = GitHubArchiveFetcher::with_client(http);
    let repository =
        GitHubRepository::parse("github.com/example/noise").expect("GitHub repository should be accepted");
    let package_version = version("v1.2.3");
    let cache_key = repository.cache_key(&package_version);

    let first = package_cache
        .get_or_insert(&cache_key, |destination| {
            github.fetch(&repository, &package_version, destination)
        })
        .expect("package should materialize");
    assert_eq!(
        fs::read_to_string(first.join("src/lib.wyn")).expect("materialized source should be readable"),
        "def answer: i32 = 42\n"
    );

    let second = package_cache
        .get_or_insert(&cache_key, |destination| {
            github.fetch(&repository, &package_version, destination)
        })
        .expect("completed cache should be reusable");
    assert_eq!(first, second);
    assert_eq!(requests.load(Ordering::Relaxed), 1);
}

#[test]
fn materialization_rejects_archives_with_multiple_roots() {
    let cache = TestCache::new();
    let requests = Arc::new(AtomicUsize::new(0));
    let mut first = package_archive("first", &[("wyn.toml", "first")]);
    let second = package_archive("second", &[("src/lib.wyn", "second")]);

    let first_decoder = flate2::read::GzDecoder::new(Cursor::new(first));
    let second_decoder = flate2::read::GzDecoder::new(Cursor::new(second));
    let encoder = GzEncoder::new(Vec::new(), Compression::default());
    let mut combined = tar::Builder::new(encoder);
    for source in [first_decoder, second_decoder] {
        let mut source = tar::Archive::new(source);
        for entry in source.entries().expect("test archive should be readable") {
            let mut entry = entry.expect("test entry should be readable");
            let path = entry.path().expect("test path should be readable").into_owned();
            combined.append(&entry.header().clone(), &mut entry).expect("test entry should be copied");
            let _ = path;
        }
    }
    let encoder = combined.into_inner().expect("combined tar should finish");
    first = encoder.finish().expect("combined gzip should finish");

    let http = FakeHttpClient {
        status: 200,
        archive: first,
        requests,
    };
    let package_cache = PackageCache::at(cache.root.clone());
    let mut github = GitHubArchiveFetcher::with_client(http);
    let repository =
        GitHubRepository::parse("github.com/example/noise").expect("GitHub repository should be accepted");
    let package_version = version("v1.2.3");
    assert!(matches!(
        package_cache.get_or_insert(&repository.cache_key(&package_version), |destination| {
            github.fetch(&repository, &package_version, destination)
        }),
        Err(MaterializationError::MultipleArchiveRoots { .. })
    ));
}

#[test]
fn github_repository_rejects_urls_and_unsupported_hosts() {
    assert!(matches!(
        GitHubRepository::parse("https://example.com/noise"),
        Err(MaterializationError::UnsupportedRepository { .. })
    ));
    assert!(matches!(
        GitHubRepository::parse("https://github.com/example/noise"),
        Err(MaterializationError::UnsupportedRepository { .. })
    ));
    assert!(matches!(
        GitHubRepository::parse("github.com/example/noise.git"),
        Err(MaterializationError::UnsupportedRepository { .. })
    ));
}

#[test]
fn configured_cache_root_must_be_absolute() {
    assert!(matches!(
        absolute_cache_root(PathBuf::from("relative/cache")),
        Err(MaterializationError::RelativeCache { .. })
    ));
}
