mod literal;

use crate::ast::Span;
use nom::{
    IResult,
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alpha1, alphanumeric1, char, multispace0, multispace1, one_of},
    combinator::{eof, map, peek, recognize, value},
    multi::many0,
    sequence::{pair, preceded, terminated, tuple},
};

use literal::{parse_char_literal, parse_float_literal, parse_int_literal, parse_string_literal};

/// Token with source location information
#[derive(Debug, Clone, PartialEq)]
pub struct LocatedToken {
    pub token: Token,
    pub span: Span,
}

impl LocatedToken {
    pub fn new(token: Token, span: Span) -> Self {
        LocatedToken { token, span }
    }

    /// Create a located token with dummy span (for testing)
    #[cfg(test)]
    pub fn dummy(token: Token) -> Self {
        LocatedToken {
            token,
            span: Span::dummy(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Let,
    Def,
    Sig,
    In,
    If,
    Then,
    Else,
    Loop,
    For,
    While,
    Do,
    Match,
    Case,
    Module,
    Open,
    Import,
    Type,
    Include,
    With,
    Assert,

    // Identifiers and literals
    Identifier(String),
    IntLiteral(i32),
    SuffixedLiteral(Box<Token>, String), // literal token with type suffix (e.g., "u32", "i64", "f64")
    FloatLiteral(f32),
    CharLiteral(char),
    StringLiteral(String),

    // Boolean literals
    True,
    False,

    // Operators
    Assign,
    BinOp(String), // Binary operators: +, -, *, /
    Arrow,
    TypeCoercion, // :> for type coercion
    Backslash,    // \ for lambda expressions
    Dot,          // . for field access
    DotDot,       // .. for ranges
    DotDotLt,     // ..< for ranges
    DotDotGt,     // ..> for ranges
    Ellipsis,     // ... for ranges
    Pipe,         // | for pattern matching
    PipeOp,       // |> for pipe operator
    Star,         // * for uniqueness types (prefix)
    Minus,        // - (can be unary or binary)
    Underscore,   // _ for wildcard patterns
    Bang,         // ! for negation
    QuestionMark, // ? for existential types
    TypeHole,     // ??? for type holes (placeholder expressions)
    At,           // @ for as-patterns (future)
    DollarSign,   // $ for curried/partial application

    // Delimiters
    LeftParen,
    RightParen,
    LeftBracket,       // [ without preceding whitespace (for indexing: arr[0])
    LeftBracketSpaced, // [ with preceding whitespace (for array literals: f [1,2,3])
    RightBracket,
    LeftBrace,
    RightBrace,
    Colon,
    Comma,

    // Attributes
    AttributeStart, // #[

    // Vector/Matrix literals
    AtBracket, // @[ for vector/matrix literals

    // Comments (to be skipped)
    Comment(String),
}

fn parse_comment(input: &str) -> IResult<&str, Token> {
    map(preceded(tag("--"), take_until("\n")), |s: &str| {
        Token::Comment(s.to_string())
    })(input)
}

fn parse_keyword(input: &str) -> IResult<&str, Token> {
    // Helper function to match a keyword with word boundaries
    let keyword = |kw: &'static str, token: Token| {
        map(
            terminated(
                tag(kw),
                peek(alt((eof, recognize(one_of(" \t\n\r()[]{}=:,+-/*#<>"))))),
            ),
            move |_| token.clone(),
        )
    };

    alt((
        alt((
            keyword("module", Token::Module),
            keyword("import", Token::Import),
            keyword("include", Token::Include),
            keyword("assert", Token::Assert),
            keyword("match", Token::Match),
            keyword("case", Token::Case),
        )),
        alt((
            keyword("let", Token::Let),
            keyword("loop", Token::Loop),
            keyword("while", Token::While),
            keyword("def", Token::Def),
            keyword("sig", Token::Sig),
            keyword("type", Token::Type),
            keyword("with", Token::With),
            keyword("open", Token::Open),
        )),
        alt((
            keyword("then", Token::Then),
            keyword("else", Token::Else),
            keyword("true", Token::True),
            keyword("false", Token::False),
            keyword("for", Token::For),
            keyword("in", Token::In),
            keyword("if", Token::If),
            keyword("do", Token::Do),
        )),
    ))(input)
}

// Removed parse_type - i32 and f32 are now treated as regular identifiers
// They are only special in type position (handled by parser) and as suffixes on literals

fn parse_type_variable(input: &str) -> IResult<&str, Token> {
    map(
        recognize(tuple((
            tag("'"),
            alt((alpha1, tag("_"))),
            many0(alt((alphanumeric1, tag("_")))),
        ))),
        |s: &str| Token::Identifier(s.to_string()),
    )(input)
}

fn parse_identifier(input: &str) -> IResult<&str, Token> {
    map(
        recognize(pair(
            alt((alpha1, tag("_"))),
            many0(alt((alphanumeric1, tag("_"), tag("'")))),
        )),
        |s: &str| {
            // Handle underscore specially - it's a wildcard pattern, not an identifier
            if s == "_" { Token::Underscore } else { Token::Identifier(s.to_string()) }
        },
    )(input)
}

// Float, integer, string, and char literal parsing moved to literal submodule

fn parse_operator(input: &str) -> IResult<&str, Token> {
    alt((
        alt((
            value(Token::Arrow, tag("->")),
            value(Token::TypeCoercion, tag(":>")),
            // Type hole (must come before Ellipsis and QuestionMark)
            value(Token::TypeHole, tag("???")),
            // Range operators (must come before ..)
            value(Token::Ellipsis, tag("...")),
            value(Token::DotDotLt, tag("..<")),
            value(Token::DotDotGt, tag("..>")),
            value(Token::DotDot, tag("..")),
            // Pipe operator (must come before |)
            value(Token::PipeOp, tag("|>")),
        )),
        alt((
            // Multi-character operators (must come before single-char versions)
            map(tag(">>>"), |s: &str| Token::BinOp(s.to_string())), // Logical right shift
            map(tag("**"), |s: &str| Token::BinOp(s.to_string())),  // Exponentiation
            map(tag("//"), |s: &str| Token::BinOp(s.to_string())),  // Integer division
            map(tag("%%"), |s: &str| Token::BinOp(s.to_string())),  // Integer modulo
            map(tag(">>"), |s: &str| Token::BinOp(s.to_string())),  // Right shift
            map(tag("<<"), |s: &str| Token::BinOp(s.to_string())),  // Left shift
            // Comparison operators (must come before single =, <, >)
            map(tag("=="), |s: &str| Token::BinOp(s.to_string())),
            map(tag("!="), |s: &str| Token::BinOp(s.to_string())),
            map(tag("<="), |s: &str| Token::BinOp(s.to_string())),
            map(tag(">="), |s: &str| Token::BinOp(s.to_string())),
        )),
        alt((
            // Single-character comparison
            map(tag("<"), |s: &str| Token::BinOp(s.to_string())),
            map(tag(">"), |s: &str| Token::BinOp(s.to_string())),
            // Assignment (must come after ==)
            value(Token::Assign, tag("=")),
            // Arithmetic operators
            map(tag("/"), |s: &str| Token::BinOp(s.to_string())),
            map(char('+'), |c| Token::BinOp(c.to_string())),
            map(char('-'), |c| Token::BinOp(c.to_string())),
            map(char('*'), |c| Token::BinOp(c.to_string())),
            map(char('%'), |c| Token::BinOp(c.to_string())),
            // Bitwise operators
            map(char('&'), |c| Token::BinOp(c.to_string())),
            map(char('^'), |c| Token::BinOp(c.to_string())),
            value(Token::Backslash, char('\\')),
        )),
        alt((
            value(Token::Dot, char('.')),
            value(Token::Pipe, char('|')),
            value(Token::Bang, char('!')),
            value(Token::QuestionMark, char('?')),
            value(Token::At, char('@')),
            value(Token::DollarSign, char('$')),
        )),
    ))(input)
}

fn parse_delimiter(input: &str) -> IResult<&str, Token> {
    alt((
        value(Token::AttributeStart, tag("#[")),
        value(Token::AtBracket, tag("@[")),
        value(Token::LeftParen, char('(')),
        value(Token::RightParen, char(')')),
        value(Token::LeftBracket, char('[')),
        value(Token::RightBracket, char(']')),
        value(Token::LeftBrace, char('{')),
        value(Token::RightBrace, char('}')),
        value(Token::Colon, char(':')),
        value(Token::Comma, char(',')),
    ))(input)
}

fn parse_token(input: &str) -> IResult<&str, Token> {
    preceded(
        multispace0,
        alt((
            parse_comment,
            parse_string_literal,
            parse_char_literal,
            parse_keyword,
            parse_float_literal,
            parse_type_variable,
            parse_identifier,
            parse_int_literal,
            parse_delimiter, // Must come before parse_operator to match @[ before @
            parse_operator,
        )),
    )(input)
}

pub fn tokenize(input: &str) -> Result<Vec<LocatedToken>, String> {
    let mut remaining = input;
    let mut tokens = Vec::new();
    let mut had_whitespace = true; // Start of input counts as having whitespace

    // Build line offset table once for O(log n) span calculations
    let line_offsets = LineOffsets::new(input);

    while !remaining.is_empty() {
        // Check for and skip leading whitespace
        if let Ok((rest, _)) = multispace1::<&str, nom::error::Error<&str>>(remaining) {
            remaining = rest;
            had_whitespace = true;
            continue;
        }

        match parse_token(remaining) {
            Ok((rest, mut token)) => {
                // Skip comments
                if matches!(token, Token::Comment(_)) {
                    remaining = rest;
                    had_whitespace = true; // Comments act like whitespace
                    continue;
                }

                // Convert LeftBracket based on whitespace
                if matches!(token, Token::LeftBracket) {
                    token = if had_whitespace { Token::LeftBracketSpaced } else { Token::LeftBracket };
                }

                // Calculate span based on position in original input
                let span = calculate_span(input, &line_offsets, remaining, rest);
                tokens.push(LocatedToken::new(token, span));
                remaining = rest;
                had_whitespace = false; // Next token won't have whitespace unless we skip some
            }
            Err(_) if remaining.trim().is_empty() => break,
            Err(e) => return Err(format!("Tokenization error: {:?}", e)),
        }
    }

    Ok(tokens)
}

/// Precomputed line offset table for efficient offset-to-line-column conversion.
/// Built once per input, then used with binary search for O(log n) lookups.
struct LineOffsets {
    /// Byte offsets where each line starts. line_starts[0] = 0 (line 1 starts at offset 0).
    line_starts: Vec<usize>,
}

impl LineOffsets {
    /// Build a line offset table from the input string. O(n) one-time cost.
    fn new(input: &str) -> Self {
        let mut line_starts = vec![0];
        for (i, ch) in input.char_indices() {
            if ch == '\n' {
                line_starts.push(i + 1);
            }
        }
        LineOffsets { line_starts }
    }

    /// Convert byte offset to (line, column) - both 1-indexed. O(log n) per lookup.
    fn offset_to_line_col(&self, offset: usize) -> (usize, usize) {
        // Binary search to find which line this offset is on
        let line_idx = match self.line_starts.binary_search(&offset) {
            Ok(idx) => idx,      // Exact match: offset is at start of line
            Err(idx) => idx - 1, // offset is within line (idx-1)
        };
        let line = line_idx + 1; // 1-indexed
        let col = offset - self.line_starts[line_idx] + 1; // 1-indexed
        (line, col)
    }
}

/// Calculate the span of a token given the original input and the before/after strings
fn calculate_span(original: &str, line_offsets: &LineOffsets, before: &str, after: &str) -> Span {
    // Calculate byte offsets
    let start_offset = original.len() - before.len();
    let end_offset = original.len() - after.len();

    // Calculate line and column for start and end positions
    let (start_line, start_col) = line_offsets.offset_to_line_col(start_offset);
    let (end_line, end_col) = line_offsets.offset_to_line_col(end_offset);

    Span::new(start_line, start_col, end_line, end_col)
}

#[cfg(test)]
mod tests {
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
        let input = "loop for while do match case assert";
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
                Token::Assert,
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
}
