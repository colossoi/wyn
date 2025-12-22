use super::*;

// Helper to extract just the tokens from LocatedToken for testing
fn tokens_only(input: &str) -> Vec<Token> {
    tokenize(input).unwrap().into_iter().map(|lt| lt.token).collect()
}

#[test]
fn test_tokenize_keywords() {
    let input = "let def";
    let tokens = tokens_only(input);
    assert_eq!(tokens, vec![Token::Let, Token::Def]);
}

#[test]
fn test_tokenize_types() {
    let input = "i32 f32";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::Identifier("i32".to_string()),
            Token::Identifier("f32".to_string())
        ]
    );
}

#[test]
fn test_tokenize_identifiers() {
    let input = "vertex_main SKY_RGBA verts";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::Identifier("vertex_main".to_string()),
            Token::Identifier("SKY_RGBA".to_string()),
            Token::Identifier("verts".to_string()),
        ]
    );
}

#[test]
fn test_tokenize_literals() {
    let input = "-1.0f32 42 3.14f32";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::FloatLiteral(-1.0),
            Token::IntLiteral(42),
            Token::FloatLiteral(3.14),
        ]
    );
}

#[test]
fn test_all_literal_types() {
    // literal ::= intnumber | floatnumber | "true" | "false"
    let input = "true false 123 45.67f32";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::True,
            Token::False,
            Token::IntLiteral(123),
            Token::FloatLiteral(45.67),
        ]
    );
}

#[test]
fn test_integer_literal_formats() {
    // Test decimal integers (basic support)
    let input = "42 -10 0";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::IntLiteral(42),
            Token::IntLiteral(-10),
            Token::IntLiteral(0),
        ]
    );
}

// Hexadecimal, binary, underscore, and exponent tests moved to literal::tests submodule

#[test]
fn test_tokenize_with_comments() {
    let input = "-- This is a comment\nlet x = 42";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::Let,
            Token::Identifier("x".to_string()),
            Token::Assign,
            Token::IntLiteral(42),
        ]
    );
}

#[test]
fn test_tokenize_array_syntax() {
    let input = "[3][4]f32";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::LeftBracketSpaced, // At start of input, counts as having whitespace
            Token::IntLiteral(3),
            Token::RightBracket,
            Token::LeftBracket, // No space before this one
            Token::IntLiteral(4),
            Token::RightBracket,
            Token::Identifier("f32".to_string()),
        ]
    );
}

#[test]
fn test_tokenize_division() {
    let input = "135.0f32/255.0f32";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::FloatLiteral(135.0),
            Token::BinOp("/".to_string()),
            Token::FloatLiteral(255.0),
        ]
    );
}

#[test]
fn test_tokenize_binary_operators() {
    let input = "a + b - c * d / e";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::Identifier("a".to_string()),
            Token::BinOp("+".to_string()),
            Token::Identifier("b".to_string()),
            Token::BinOp("-".to_string()),
            Token::Identifier("c".to_string()),
            Token::BinOp("*".to_string()),
            Token::Identifier("d".to_string()),
            Token::BinOp("/".to_string()),
            Token::Identifier("e".to_string()),
        ]
    );
}

#[test]
fn test_tokenize_attributes() {
    let input = "#[vertex]";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::AttributeStart,
            Token::Identifier("vertex".to_string()),
            Token::RightBracket,
        ]
    );
}

// Tests for grammar rules: name with constituent characters
#[test]
fn test_name_with_prime() {
    // constituent ::= letter | digit | "_" | "'"
    let input = "acc' uv' x'' my_var'";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::Identifier("acc'".to_string()),
            Token::Identifier("uv'".to_string()),
            Token::Identifier("x''".to_string()),
            Token::Identifier("my_var'".to_string()),
        ]
    );
}

#[test]
fn test_name_starting_with_underscore() {
    // name ::= lowercase constituent* | "_" constituent*
    let input = "_foo _bar123 _x'";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::Identifier("_foo".to_string()),
            Token::Identifier("_bar123".to_string()),
            Token::Identifier("_x'".to_string()),
        ]
    );
}

#[test]
fn test_constructor_names() {
    // constructor ::= uppercase constituent*
    let input = "Some None True' MyType123";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::Identifier("Some".to_string()),
            Token::Identifier("None".to_string()),
            Token::Identifier("True'".to_string()),
            Token::Identifier("MyType123".to_string()),
        ]
    );
}

#[test]
fn test_new_operators() {
    // Test |>, .., ..., ..<, ..>, |, !, ?, @
    let input = "|> .. ... ..< ..>";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::PipeOp,
            Token::DotDot,
            Token::Ellipsis,
            Token::DotDotLt,
            Token::DotDotGt,
        ]
    );
}

#[test]
fn test_bang_and_pipe() {
    let input = "! | !x |> y";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::Bang,
            Token::Pipe,
            Token::Bang,
            Token::Identifier("x".to_string()),
            Token::PipeOp,
            Token::Identifier("y".to_string()),
        ]
    );
}

#[test]
fn test_question_and_at() {
    let input = "? @ x ?[n]";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::QuestionMark,
            Token::At,
            Token::Identifier("x".to_string()),
            Token::QuestionMark,
            Token::LeftBracket,
            Token::Identifier("n".to_string()),
            Token::RightBracket,
        ]
    );
}

#[test]
fn test_new_keywords() {
    let input = "loop for while do match case";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::Loop,
            Token::For,
            Token::While,
            Token::Do,
            Token::Match,
            Token::Case,
        ]
    );
}

#[test]
fn test_qualified_names_tokenization() {
    // quals ::= (name ".")+
    // qualname ::= name | quals name
    // The parser will handle this, but lexer should produce: Identifier, Dot, Identifier
    let input = "f32.sin i32.max std.math.sqrt";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::Identifier("f32".to_string()),
            Token::Dot,
            Token::Identifier("sin".to_string()),
            Token::Identifier("i32".to_string()),
            Token::Dot,
            Token::Identifier("max".to_string()),
            Token::Identifier("std".to_string()),
            Token::Dot,
            Token::Identifier("math".to_string()),
            Token::Dot,
            Token::Identifier("sqrt".to_string()),
        ]
    );
}

#[test]
fn test_range_operators_in_context() {
    // Test that range operators work in realistic context
    let input = "a..b a...b a..<b a..>b";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::Identifier("a".to_string()),
            Token::DotDot,
            Token::Identifier("b".to_string()),
            Token::Identifier("a".to_string()),
            Token::Ellipsis,
            Token::Identifier("b".to_string()),
            Token::Identifier("a".to_string()),
            Token::DotDotLt,
            Token::Identifier("b".to_string()),
            Token::Identifier("a".to_string()),
            Token::DotDotGt,
            Token::Identifier("b".to_string()),
        ]
    );
}

#[test]
fn test_comparison_operators() {
    let input = "== != <= >= < >";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::BinOp("==".to_string()),
            Token::BinOp("!=".to_string()),
            Token::BinOp("<=".to_string()),
            Token::BinOp(">=".to_string()),
            Token::BinOp("<".to_string()),
            Token::BinOp(">".to_string()),
        ]
    );
}

#[test]
fn test_mixed_identifiers_and_operators() {
    // Test realistic expression with new features
    let input = "acc' |> map (\\x -> -x) arr[0]";
    let tokens = tokens_only(input);
    assert_eq!(
        tokens,
        vec![
            Token::Identifier("acc'".to_string()),
            Token::PipeOp,
            Token::Identifier("map".to_string()),
            Token::LeftParen,
            Token::Backslash,
            Token::Identifier("x".to_string()),
            Token::Arrow,
            Token::BinOp("-".to_string()),
            Token::Identifier("x".to_string()),
            Token::RightParen,
            Token::Identifier("arr".to_string()),
            Token::LeftBracket,
            Token::IntLiteral(0),
            Token::RightBracket,
        ]
    );
}
