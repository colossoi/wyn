use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;
use wyn_module_graph::TextRange;

static TEST_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn new() -> Self {
        loop {
            let sequence = TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("wyn_analyzer_{}_{sequence}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Self { path },
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("test directory should be created: {error}"),
            }
        }
    }

    fn write(&self, relative: impl AsRef<Path>, contents: &str) -> PathBuf {
        let path = self.path.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("test source directory should be created");
        }
        fs::write(&path, contents).expect("test source should be written");
        path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.path) {
            eprintln!(
                "failed to remove analyzer test directory `{}`: {error}",
                self.path.display()
            );
        }
    }
}

#[test]
fn lsp_positions_round_trip_through_utf8_offsets() {
    let source = "a\nβ😀z";

    let positions = [
        (Position::new(0, 0), 0),
        (Position::new(0, 1), 1),
        (Position::new(1, 0), 2),
        (Position::new(1, 1), 4),
        (Position::new(1, 3), 8),
        (Position::new(1, 4), 9),
    ];
    for (position, offset) in positions {
        assert_eq!(position_to_offset(source, position), Some(offset));
        assert_eq!(offset_to_position(source, offset), Some(position));
    }

    assert_eq!(position_to_offset(source, Position::new(1, 2)), None);
}

#[test]
fn source_span_maps_to_an_lsp_range() {
    let source = "a\nβ😀z";
    let span = Span::new(ModuleId::from(0), TextRange::new(2, 8).expect("valid range"));

    assert_eq!(
        span_to_range(source, span),
        Some(Range::new(Position::new(1, 0), Position::new(1, 3)))
    );
    assert_eq!(span_to_range(source, Span::generated()), None);
}

#[test]
fn source_graph_uses_the_enclosing_package_and_document_overlay() {
    let directory = TestDirectory::new();
    directory.write(
        "dependency/wyn.toml",
        concat!(
            "manifest-version = 1\n",
            "[package]\n",
            "name = \"test/dependency\"\n",
            "version = \"v1.0.0\"\n",
            "wyn = \"v0.1.0\"\n",
            "library = \"src/lib.wyn\"\n",
        ),
    );
    directory.write("dependency/src/lib.wyn", "def identity<T>(value: T) T = value\n");
    directory.write(
        "application/wyn.toml",
        concat!(
            "manifest-version = 1\n",
            "[package]\n",
            "name = \"test/application\"\n",
            "version = \"v1.0.0\"\n",
            "wyn = \"v0.1.0\"\n",
            "library = \"src/lib.wyn\"\n",
            "[dependencies]\n",
            "dependency = { package = \"test/dependency\", version = \"v1.0.0\", path = \"../dependency\" }\n",
        ),
    );
    directory.write("application/src/lib.wyn", "def library: i32 = 0\n");
    let document = directory.write("application/test/editor.wyn", "this is stale text\n");
    let source = concat!(
        "module Dependency = import \"pkg:dependency\"\n",
        "entry main(value: i32) i32 = Dependency.identity(value)\n",
    );

    let modules = load_source_graph(Some(&document), source).expect("package source graph should load");
    modules.type_check().expect("overlaid package source should type check");
}

#[test]
fn unsaved_standalone_document_uses_its_parent_directory() {
    let directory = TestDirectory::new();
    let document = directory.path.join("new.wyn");
    let source = "entry main(value: i32) i32 = value\n";

    let modules = load_source_graph(Some(&document), source).expect("standalone source graph should load");
    modules.type_check().expect("standalone source should type check");
}
