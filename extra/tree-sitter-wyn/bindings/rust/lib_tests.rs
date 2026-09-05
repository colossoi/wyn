use super::*;

use std::path::{Path, PathBuf};

fn wyn_files_below(directory: &Path) -> Vec<PathBuf> {
    fn visit(directory: &Path, files: &mut Vec<PathBuf>) {
        let mut entries = std::fs::read_dir(directory)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", directory.display()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|error| panic!("failed to enumerate {}: {error}", directory.display()));
        entries.sort_by_key(std::fs::DirEntry::path);

        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                visit(&path, files);
            } else if path.extension().is_some_and(|extension| extension == "wyn") {
                files.push(path);
            }
        }
    }

    let mut files = Vec::new();
    visit(directory, &mut files);
    files
}

#[test]
fn test_can_load_grammar() {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&LANGUAGE.into()).expect("Error loading Wyn grammar");
}

#[test]
fn test_queries_compile() {
    let language = LANGUAGE.into();
    tree_sitter::Query::new(&language, HIGHLIGHTS_QUERY).expect("invalid highlights query");
    tree_sitter::Query::new(&language, LOCALS_QUERY).expect("invalid locals query");
}

#[test]
fn test_parse_simple_function() {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&LANGUAGE.into()).expect("Error loading Wyn grammar");

    let source = "def add(x: i32, y: i32) i32 = x + y";
    let tree = parser.parse(source, None).unwrap();
    let root = tree.root_node();

    assert!(!root.has_error());
    assert_eq!(root.kind(), "source_file");
}

#[test]
fn test_parse_entry_point() {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&LANGUAGE.into()).expect("Error loading Wyn grammar");

    let source = r#"
            entry vertex_main() [4]f32 = @[0.0, 0.0, 0.0, 1.0]
        "#;
    let tree = parser.parse(source, None).unwrap();
    let root = tree.root_node();

    assert!(!root.has_error());
}

#[test]
fn test_parse_lambda() {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&LANGUAGE.into()).expect("Error loading Wyn grammar");

    let source = "def double = |x: i32| x * 2";
    let tree = parser.parse(source, None).unwrap();
    let root = tree.root_node();

    assert!(!root.has_error());
}

#[test]
fn test_parse_custom_operator_definition() {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&LANGUAGE.into()).expect("Error loading Wyn grammar");

    let source = "def (+^)((a: i32, b: i32), (c: i32, d: i32)) = (a + c, b + d)";
    let tree = parser.parse(source, None).unwrap();
    let root = tree.root_node();

    assert!(!root.has_error(), "{}", root.to_sexp());
    let declaration = root.named_child(0).expect("expected a declaration");
    let name = declaration.child_by_field_name("name").expect("expected an operator name");
    assert_eq!(name.kind(), "operator_name");
    assert_eq!(name.utf8_text(source.as_bytes()).unwrap(), "(+^)");
}

#[test]
fn test_with_value_contains_the_full_binary_expression() {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&LANGUAGE.into()).expect("Error loading Wyn grammar");

    let source = "def modify(a: []i32, i: i32, x: i32) []i32 = a with [i] = a[i] + x";
    let tree = parser.parse(source, None).unwrap();
    let root = tree.root_node();

    assert!(!root.has_error(), "{}", root.to_sexp());
    let declaration = root.named_child(0).expect("expected a declaration");
    let body = declaration.child_by_field_name("body").expect("expected a definition body");
    assert_eq!(body.kind(), "array_with", "{}", body.to_sexp());
    let value = body.child_by_field_name("value").expect("expected an update value");
    assert_eq!(value.kind(), "binary_expression", "{}", body.to_sexp());
}

#[test]
fn test_with_updates_remain_left_associative() {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&LANGUAGE.into()).expect("Error loading Wyn grammar");

    let source = "def update(a: []i32, i: i32, j: i32, x: i32, y: i32) []i32 = a with [i] = x with [j] = y";
    let tree = parser.parse(source, None).unwrap();
    let root = tree.root_node();

    assert!(!root.has_error(), "{}", root.to_sexp());
    let declaration = root.named_child(0).expect("expected a declaration");
    let body = declaration.child_by_field_name("body").expect("expected a definition body");
    assert_eq!(body.kind(), "array_with", "{}", body.to_sexp());
    let array = body.child_by_field_name("array").expect("expected an updated array");
    assert_eq!(array.kind(), "array_with", "{}", body.to_sexp());
}

#[test]
fn parse_all_repository_testfiles() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let Some(repository_root) = manifest_dir.ancestors().nth(2) else {
        panic!("Tree-sitter package is not nested under a repository root");
    };
    let testfiles_dir = repository_root.join("testfiles");

    // The published crate does not contain the repository-level fixtures.
    if !testfiles_dir.is_dir() {
        return;
    }

    let files = wyn_files_below(&testfiles_dir);
    assert!(
        !files.is_empty(),
        "no .wyn files found under {}",
        testfiles_dir.display()
    );

    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&LANGUAGE.into()).expect("Error loading Wyn grammar");

    let mut failures = Vec::new();
    for path in &files {
        let source = std::fs::read(path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let tree = parser
            .parse(&source, None)
            .unwrap_or_else(|| panic!("Tree-sitter cancelled parsing {}", path.display()));
        if tree.root_node().has_error() {
            failures.push(path.strip_prefix(repository_root).unwrap_or(path).display().to_string());
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} Wyn test files contain Tree-sitter errors:\n{}",
        failures.len(),
        files.len(),
        failures.join("\n"),
    );
}
