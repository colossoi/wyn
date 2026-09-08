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

#[test]
fn semantic_tokens_recognize_current_keywords() {
    assert_eq!(token_type_index(&lexer::Token::Resource), Some(0));
    assert_eq!(token_type_index(&lexer::Token::TypeSizeLifted), Some(0));
    assert_eq!(token_type_index(&lexer::Token::TypeFullyLifted), Some(0));
}

#[test]
fn semantic_tokens_prefer_the_open_document_buffer() {
    let directory = TestDirectory::new();
    let path = directory.write("buffer.wyn", "def saved: i32 = 0\n");
    let uri = Url::from_file_path(path).expect("test path should convert to a file URL");
    let texts = RwLock::new(HashMap::new());

    assert_eq!(
        document_text(&texts, &uri).as_deref(),
        Some("def saved: i32 = 0\n")
    );
    texts
        .write()
        .expect("document text lock should be available")
        .insert(uri.clone(), "def unsaved: i32 = 1\n".to_string());
    assert_eq!(
        document_text(&texts, &uri).as_deref(),
        Some("def unsaved: i32 = 1\n")
    );
}

#[test]
fn failed_edits_clear_the_last_type_checked_document() {
    let uri = Url::parse("file:///editor-buffer.wyn").expect("valid test URL");
    let source = "def value: i32 = 0\n";
    let ast = load_source_graph(None, source)
        .expect("source graph should load")
        .type_check()
        .expect("source should type check");
    let documents = RwLock::new(HashMap::from([(
        uri.clone(),
        DocumentState {
            ast,
            text: source.to_string(),
            roots: HashMap::new(),
        },
    )]));

    update_document_state(&documents, uri.clone(), None);
    assert!(!documents.read().expect("document lock should be available").contains_key(&uri));
}

#[test]
fn navigation_distinguishes_shadowed_symbols_and_top_level_functions() {
    let source = "def first(value: i32) i32 = value\ndef second(value: i32) i32 = first(value)\n";
    let program = load_source_graph(None, source).unwrap().type_check().unwrap();
    let index = Navigation::new(&program);
    let module = program.source_graph().root();
    let first_param = index.symbol_at(module, source.find("value:").unwrap() as u32).unwrap();
    let second_param = index.symbol_at(module, source.rfind("value:").unwrap() as u32).unwrap();
    assert_ne!(first_param, second_param);
    for symbol in [first_param, second_param] {
        assert_eq!(
            index.occurrences.iter().filter(|item| item.symbol == symbol && !item.declaration).count(),
            1
        );
    }
    let call = index.symbol_at(module, source.rfind("first(").unwrap() as u32).unwrap();
    assert_eq!(index.definition(call).unwrap().range().start(), 4);
}

#[test]
fn navigation_uses_dependency_buffer_and_physical_source_location() {
    let directory = TestDirectory::new();
    for name in ["app", "dependency"] {
        let mut manifest = format!("manifest-version = 1\n[package]\nname = \"test/{name}\"\nversion = \"v1.0.0\"\nwyn = \"v0.1.0\"\nlibrary = \"src/lib.wyn\"\n");
        if name == "app" {
            manifest.push_str("[dependencies]\ndep = { package = \"test/dependency\", version = \"v1.0.0\", path = \"../dependency\" }\n");
        }
        directory.write(format!("{name}/wyn.toml"), &manifest);
    }
    let dependency = directory.write("dependency/src/lib.wyn", "def old(value: i32) i32 = value\n");
    let source = "module D = import \"pkg:dep\"\nentry main(value: i32) i32 = D.fresh(value)\n";
    let path = directory.write("app/src/lib.wyn", source);
    let overlay = "-- 😀 unsaved line\ndef fresh(value: i32) i32 = value\n";
    let buffers = HashMap::from([(normalize_path(&dependency), overlay.to_owned())]);
    let (modules, roots) = load_editor_graph(Some(&path), source, &buffers).unwrap();
    let ast = modules.type_check().unwrap();
    let index = Navigation::new(&ast);
    let symbol =
        index.symbol_at(ast.source_graph().root(), source.find("D.fresh").unwrap() as u32).unwrap();
    let span = index.definition(symbol).unwrap();
    let doc = DocumentState {
        ast,
        roots,
        text: source.to_owned(),
    };
    let location = doc.location(&Url::from_file_path(path).unwrap(), span).unwrap();
    assert_eq!(
        location.uri,
        Url::from_file_path(normalize_path(&dependency)).unwrap()
    );
    assert_eq!(
        location.range,
        Range::new(Position::new(1, 4), Position::new(1, 9))
    );
}

#[test]
fn navigation_retains_each_folded_constant_occurrence() {
    let source = "def amount: i32 = 42\nentry main(value: i32) i32 = value + amount + amount\n";
    let ast = load_source_graph(None, source).unwrap().type_check().unwrap();
    let index = Navigation::new(&ast);
    let module = ast.source_graph().root();
    let declaration = index.symbol_at(module, source.find("amount").unwrap() as u32).unwrap();
    let references: Vec<_> =
        index.occurrences.iter().filter(|item| item.symbol == declaration && !item.declaration).collect();
    assert_eq!(references.len(), 2);
    for reference in references {
        assert_eq!(ast.source_graph().snippet(reference.span).unwrap(), "amount");
        assert_eq!(
            index.symbol_at(module, reference.span.range().start()),
            Some(declaration)
        );
    }
}
