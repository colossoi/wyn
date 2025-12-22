use super::Token;
use super::literal::*;

#[test]
fn test_hexadecimal_integers() {
    assert_eq!(parse_int_literal("0x10"), Ok(("", Token::IntLiteral(16))));
    assert_eq!(parse_int_literal("0xFF"), Ok(("", Token::IntLiteral(255))));
    assert_eq!(parse_int_literal("0x1A_2B"), Ok(("", Token::IntLiteral(0x1A2B))));
    assert_eq!(parse_int_literal("0X00FF"), Ok(("", Token::IntLiteral(255))));
}

#[test]
fn test_binary_integers() {
    assert_eq!(parse_int_literal("0b1010"), Ok(("", Token::IntLiteral(10))));
    assert_eq!(parse_int_literal("0B1111"), Ok(("", Token::IntLiteral(15))));
    assert_eq!(parse_int_literal("0b10_11"), Ok(("", Token::IntLiteral(11))));
}

#[test]
fn test_integers_with_underscores() {
    assert_eq!(
        parse_int_literal("1_000_000"),
        Ok(("", Token::IntLiteral(1_000_000)))
    );
    assert_eq!(parse_int_literal("42_42"), Ok(("", Token::IntLiteral(4242))));
    assert_eq!(parse_int_literal("123_456"), Ok(("", Token::IntLiteral(123456))));
}

#[test]
fn test_basic_decimals() {
    assert_eq!(parse_int_literal("42"), Ok(("", Token::IntLiteral(42))));
    assert_eq!(parse_int_literal("-17"), Ok(("", Token::IntLiteral(-17))));
}

#[test]
fn test_pointfloat() {
    assert_eq!(
        parse_float_literal("3.14f32"),
        Ok(("", Token::FloatLiteral(3.14)))
    );
    assert_eq!(parse_float_literal("0.5f32"), Ok(("", Token::FloatLiteral(0.5))));
    // Note: .5f32 no longer parses as float (conflicts with tuple field access xy.0)
    assert!(parse_float_literal(".5f32").is_err());
}

#[test]
fn test_float_exponent_notation() {
    assert_eq!(
        parse_float_literal("1.5e10f32"),
        Ok(("", Token::FloatLiteral(1.5e10)))
    );
    assert_eq!(
        parse_float_literal("2E-5f32"),
        Ok(("", Token::FloatLiteral(2e-5)))
    );
    assert_eq!(
        parse_float_literal("3.14e2f32"),
        Ok(("", Token::FloatLiteral(314.0)))
    );
}

#[test]
fn test_floats_with_underscores() {
    assert_eq!(
        parse_float_literal("3.14_159f32"),
        Ok(("", Token::FloatLiteral(3.14159)))
    );
    assert_eq!(
        parse_float_literal("1_000.5f32"),
        Ok(("", Token::FloatLiteral(1000.5)))
    );
}

#[test]
fn test_string_literals() {
    assert_eq!(
        parse_string_literal("\"hello\""),
        Ok(("", Token::StringLiteral("hello".to_string())))
    );
    assert_eq!(
        parse_string_literal("\"hello world\""),
        Ok(("", Token::StringLiteral("hello world".to_string())))
    );
    assert_eq!(
        parse_string_literal("\"\""),
        Ok(("", Token::StringLiteral("".to_string())))
    );
    assert_eq!(
        parse_string_literal("\"foo123\""),
        Ok(("", Token::StringLiteral("foo123".to_string())))
    );
}

#[test]
fn test_string_rejects_backslash() {
    // Strings with backslashes should fail (no escape sequences according to grammar)
    assert!(parse_string_literal("\"hello\\nworld\"").is_err());
    assert!(parse_string_literal("\"test\\\"quote\"").is_err());
}

#[test]
fn test_string_rejects_newline() {
    // Strings with newlines should fail
    assert!(parse_string_literal("\"hello\nworld\"").is_err());
}

#[test]
fn test_char_literals() {
    assert_eq!(parse_char_literal("'a'"), Ok(("", Token::CharLiteral('a'))));
    assert_eq!(parse_char_literal("'Z'"), Ok(("", Token::CharLiteral('Z'))));
    assert_eq!(parse_char_literal("'0'"), Ok(("", Token::CharLiteral('0'))));
    assert_eq!(parse_char_literal("' '"), Ok(("", Token::CharLiteral(' '))));
    assert_eq!(parse_char_literal("'?'"), Ok(("", Token::CharLiteral('?'))));
}

#[test]
fn test_char_rejects_backslash() {
    // Chars with backslashes should fail (no escape sequences according to grammar)
    assert!(parse_char_literal("'\\n'").is_err());
    assert!(parse_char_literal("'\\''").is_err());
}

#[test]
fn test_char_rejects_newline() {
    // Chars with newlines should fail
    assert!(parse_char_literal("'\n'").is_err());
}

#[test]
fn test_char_rejects_empty() {
    // Empty char literals should fail
    assert!(parse_char_literal("''").is_err());
}

#[test]
fn test_float_without_suffix() {
    // Float literals without suffix should parse as f32
    assert_eq!(parse_float_literal("3.14"), Ok(("", Token::FloatLiteral(3.14))));
    assert_eq!(parse_float_literal("0.5"), Ok(("", Token::FloatLiteral(0.5))));
    // Note: .5 no longer parses as float (conflicts with tuple field access xy.0)
    assert!(parse_float_literal(".5").is_err());
    assert_eq!(parse_float_literal("7.0"), Ok(("", Token::FloatLiteral(7.0))));
    assert_eq!(parse_float_literal("-3.14"), Ok(("", Token::FloatLiteral(-3.14))));
}

#[test]
fn test_float_exponent_without_suffix() {
    // Float exponent notation without suffix
    assert_eq!(
        parse_float_literal("1.5e10"),
        Ok(("", Token::FloatLiteral(1.5e10)))
    );
    assert_eq!(parse_float_literal("2E-5"), Ok(("", Token::FloatLiteral(2e-5))));
    assert_eq!(
        parse_float_literal("3.14e2"),
        Ok(("", Token::FloatLiteral(314.0)))
    );
}
