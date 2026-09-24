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
    // Both Cargo manifests share these tests, but have different depths.
    // Published crates do not contain the repository-level fixtures.
    let Some(repository_root) = manifest_dir.ancestors().find(|path| {
        path.join("SPECIFICATION.md").is_file() && path.join("wyn-core/src/parser.rs").is_file()
    }) else {
        return;
    };
    let files: Vec<_> = ["testfiles", "pkg", "tests"]
        .into_iter()
        .flat_map(|directory| wyn_files_below(&repository_root.join(directory)))
        .collect();
    assert!(!files.is_empty(), "no repository Wyn fixtures found");

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

fn parse_source(source: &str) -> tree_sitter::Tree {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&LANGUAGE.into()).unwrap();
    let tree = parser.parse(source, None).unwrap();
    assert!(
        !tree.root_node().has_error(),
        "{source}: {}",
        tree.root_node().to_sexp()
    );
    tree
}

#[test]
fn compiler_supported_forms_parse() {
    for source in [
        "def bits = 1 | 2 & 3 ^ 4",
        "def add = (+)",
        "def x = (+)(1, 2)",
        "def x = M.(+)(1, 2)",
        "def x = M.(+^)(1, 2)",
        r"module F = \((X: S)) -> X",
        r"module F = \((X: S), (Y: T)) -> { module Z = X }",
        "def f(#[size_hint(4)] a: []i32) = a",
        "def f(#[size_hint(4)] #[size_hint(8)] a: []i32) = a",
        "def f = |#[size_hint(4)] a: []i32| a",
        "#[size_hint(4)] type t = i32",
        "#[size_hint(4)] module M = { let x = 1 }",
        r#"module type S = { #[linked("foo")] sig f: i32 }"#,
        "def x = 1e1_0",
        "def x = 1.5e-1_0f64",
        "def x = (1,)",
        "def f((x,)) = x",
        "def x = 1 -- comment without final newline",
    ] {
        parse_source(source);
    }
}

#[test]
fn malformed_literals_and_singleton_tuple_types_are_rejected() {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&LANGUAGE.into()).unwrap();
    for source in [
        "def x = 0x_FF",
        "def x = 0b_10",
        r#"#[linked("a\b")] extern f() i32"#,
        "#[linked(\"a\nb\")] extern f() i32",
        "type t = (i32,)",
    ] {
        let tree = parser.parse(source, None).unwrap();
        assert!(tree.root_node().has_error(), "unexpectedly accepted {source:?}");
    }
}

#[test]
fn underscore_field_is_not_part_of_a_float() {
    let tree = parse_source("def x = 1._2e3");
    let body = tree.root_node().named_child(0).unwrap().child_by_field_name("body").unwrap();
    assert_eq!(body.kind(), "field_expression");
}

fn expression_shape(node: tree_sitter::Node<'_>, source: &str) -> String {
    let child = |field| expression_shape(node.child_by_field_name(field).unwrap(), source);
    match node.kind() {
        "binary_expression" => {
            let operator =
                node.child_by_field_name("operator").unwrap().utf8_text(source.as_bytes()).unwrap();
            format!("({} {operator} {})", child("left"), child("right"))
        }
        "array_with" => format!(
            "({} with [{}] = {})",
            child("array"),
            child("index"),
            child("value")
        ),
        "vec_with" => format!(
            "({} with .{} = {})",
            child("vector"),
            child("swizzle"),
            child("value")
        ),
        "record_with" => format!(
            "({} with {} = {})",
            child("record"),
            child("field"),
            child("value")
        ),
        "type_ascription" => format!("({} : {})", child("expression"), child("type")),
        "type_coercion" => format!("({} :> {})", child("expression"), child("type")),
        "parenthesized_expression" => expression_shape(node.named_child(0).unwrap(), source),
        _ => node.utf8_text(source.as_bytes()).unwrap().to_owned(),
    }
}

#[test]
fn expression_grouping_matches_the_compiler() {
    for (expression, expected) in [
        ("a + b |> f", "((a + b) |> f)"),
        ("a |> f |> g", "((a |> f) |> g)"),
        ("a | b & c", "((a | b) & c)"),
        ("a ** b ** c", "((a ** b) ** c)"),
        ("0 .. 2 < 3", "(0 .. (2 < 3))"),
        ("0 ..= 2 |> f", "(0 ..= (2 |> f))"),
        ("a + b with [i] = v", "(a + (b with [i] = v))"),
        ("a + b with .xy = v", "(a + (b with .xy = v))"),
        ("a + b with field = v", "(a + (b with field = v))"),
        ("a with [i] = b + c", "(a with [i] = (b + c))"),
        ("a with [i] = b |> f", "(a with [i] = (b |> f))"),
        ("a with [i] = b .. c", "((a with [i] = b) .. c)"),
        ("a with [i] = b : T", "((a with [i] = b) : T)"),
        ("a with [i] = b :> T", "((a with [i] = b) :> T)"),
        ("-a with [i] = b", "(-a with [i] = b)"),
        ("a + b with [i] = c + d", "(a + (b with [i] = (c + d)))"),
        ("(a + b) with [i] = v", "((a + b) with [i] = v)"),
        ("a with [i] = x with [j] = y", "((a with [i] = x) with [j] = y)"),
        (
            "a with [i] = b + c with [j] = y",
            "((a with [i] = (b + c)) with [j] = y)",
        ),
    ] {
        let source = format!("def result = {expression}");
        let tree = parse_source(&source);
        let body = tree.root_node().named_child(0).unwrap().child_by_field_name("body").unwrap();
        assert_eq!(expression_shape(body, &source), expected, "{source}");
    }
}

#[test]
fn test_literal_expansion_and_inclusive_range() {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&LANGUAGE.into()).unwrap();
    for source in [
        "let v = @[nrm..., 1.0]",
        "let v = @[nrm.yx..., nrm.xx...,]",
        "let v = [nrm..., 1.0]",
        "let r = 0..=3",
        "let r = 0..2..=4",
    ] {
        let tree = parser.parse(source, None).unwrap();
        assert!(
            !tree.root_node().has_error(),
            "{source}: {}",
            tree.root_node().to_sexp()
        );
    }
}
