use super::*;

#[test]
fn test_can_load_grammar() {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&LANGUAGE.into()).expect("Error loading Wyn grammar");
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
            #[vertex]
            entry vertex_main() #[builtin(position)] [4]f32 = @[0.0, 0.0, 0.0, 1.0]
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
