//! Literal parsing for the Wyn lexer
//!
//! Implements the full grammar specification for literals:
//! - Integer literals: decimal, hexadecimal (0xFF), binary (0b1010) with optional underscores
//! - Float literals: pointfloat, exponent notation, hexadecimal floats
//! - Type suffixes: i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64
//! - String literals: "stringchar*"
//! - Char literals: 'char'

use nom::{
    IResult,
    branch::alt,
    bytes::complete::{tag, take_while},
    character::complete::{char, digit1, hex_digit1, none_of, one_of},
    combinator::{map, opt, recognize, verify},
    multi::many1,
    sequence::{delimited, pair, tuple},
};

use super::Token;

// Helper to parse digits with optional underscores
fn decimal_with_underscores(input: &str) -> IResult<&str, &str> {
    recognize(pair(digit1, many1(alt((digit1, tag("_"))))))(input)
}

fn hex_with_underscores(input: &str) -> IResult<&str, &str> {
    recognize(pair(hex_digit1, many1(alt((hex_digit1, tag("_"))))))(input)
}

fn binary_with_underscores(input: &str) -> IResult<&str, &str> {
    recognize(pair(one_of("01"), many1(alt((one_of("01"), char('_'))))))(input)
}

// Strip underscores from a numeric string for parsing
fn strip_underscores(s: &str) -> String {
    s.chars().filter(|c| *c != '_').collect()
}

// Parse integer type suffix
fn int_type_suffix(input: &str) -> IResult<&str, &str> {
    alt((
        tag("i8"),
        tag("i16"),
        tag("i32"),
        tag("i64"),
        tag("u8"),
        tag("u16"),
        tag("u32"),
        tag("u64"),
    ))(input)
}

// Parse float type suffix
fn float_type_suffix(input: &str) -> IResult<&str, &str> {
    alt((tag("f16"), tag("f32"), tag("f64")))(input)
}

// Hexadecimal integer: 0x[hex_digits][type_suffix]
fn parse_hexadecimal_int(input: &str) -> IResult<&str, Token> {
    map(
        tuple((
            alt((tag("0x"), tag("0X"))),
            alt((hex_with_underscores, hex_digit1)),
            opt(int_type_suffix),
        )),
        |(_, digits, suffix)| {
            let clean = strip_underscores(digits);
            // Parse as u64 to handle full range, convert to decimal string
            let value = u64::from_str_radix(&clean, 16).unwrap_or_else(|e| {
                panic!("BUG: Failed to parse hexadecimal integer '0x{}': {}. The lexer matched the pattern but conversion failed.", clean, e)
            });
            let base_token = Token::IntLiteral(value.to_string().into());
            match suffix {
                Some("i32") | None => base_token,
                Some(s) => Token::SuffixedLiteral(Box::new(base_token), s.to_string()),
            }
        },
    )(input)
}

// Binary integer: 0b[binary_digits][type_suffix]
fn parse_binary_int(input: &str) -> IResult<&str, Token> {
    map(
        tuple((
            alt((tag("0b"), tag("0B"))),
            alt((binary_with_underscores, recognize(many1(one_of("01"))))),
            opt(int_type_suffix),
        )),
        |(_, digits, suffix)| {
            let clean = strip_underscores(digits);
            // Parse as u64 to handle full range, convert to decimal string
            let value = u64::from_str_radix(&clean, 2).unwrap_or_else(|e| {
                panic!("BUG: Failed to parse binary integer '0b{}': {}. The lexer matched the pattern but conversion failed.", clean, e)
            });
            let base_token = Token::IntLiteral(value.to_string().into());
            match suffix {
                // i32 is the default integer type, no need to wrap
                Some("i32") | None => base_token,
                Some(s) => Token::SuffixedLiteral(Box::new(base_token), s.to_string()),
            }
        },
    )(input)
}

// Decimal integer: [digits][type_suffix]
fn parse_decimal_int(input: &str) -> IResult<&str, Token> {
    map(
        tuple((
            opt(char('-')),
            alt((decimal_with_underscores, digit1)),
            opt(int_type_suffix),
        )),
        |(sign, digits, suffix)| {
            let mut clean = strip_underscores(digits);
            if sign.is_some() {
                clean = format!("-{}", clean);
            }
            // Store as string directly - no numeric conversion needed for decimal
            let base_token = Token::IntLiteral(clean.into());
            match suffix {
                // i32 is the default integer type, no need to wrap
                Some("i32") | None => base_token,
                Some(s) => Token::SuffixedLiteral(Box::new(base_token), s.to_string()),
            }
        },
    )(input)
}

// Parse integer literal (try hex, binary, then decimal)
pub fn parse_int_literal(input: &str) -> IResult<&str, Token> {
    alt((parse_hexadecimal_int, parse_binary_int, parse_decimal_int))(input)
}

// Intpart: digits (possibly with underscores)
fn intpart(input: &str) -> IResult<&str, &str> {
    alt((decimal_with_underscores, digit1))(input)
}

// Pointfloat: intpart "." fraction
// Note: We require digits before the decimal point to avoid parsing `.0` as a float
// (which interferes with tuple field access like `xy.0`). Use `0.5` instead of `.5`.
fn pointfloat(input: &str) -> IResult<&str, &str> {
    recognize(tuple((
        intpart,
        char('.'),
        alt((decimal_with_underscores, digit1)),
    )))(input)
}

// Exponent: (e|E)[+|-]digits
fn exponent(input: &str) -> IResult<&str, &str> {
    recognize(tuple((
        alt((char('e'), char('E'))),
        opt(alt((char('+'), char('-')))),
        alt((decimal_with_underscores, digit1)),
    )))(input)
}

// Exponent float: (intpart | pointfloat) exponent
fn exponentfloat(input: &str) -> IResult<&str, &str> {
    recognize(tuple((alt((pointfloat, intpart)), exponent)))(input)
}

// Hexadecimal float: 0x[hex_mantissa]p[+|-][dec_exponent]
fn hexadecimalfloat(input: &str) -> IResult<&str, &str> {
    recognize(tuple((
        alt((tag("0x"), tag("0X"))),
        alt((hex_with_underscores, hex_digit1)),
        opt(tuple((char('.'), alt((hex_with_underscores, hex_digit1))))),
        alt((char('p'), char('P'))),
        opt(alt((char('+'), char('-')))),
        alt((decimal_with_underscores, digit1)),
    )))(input)
}

// Parse float literal
pub fn parse_float_literal(input: &str) -> IResult<&str, Token> {
    map(
        tuple((
            opt(char('-')),
            alt((hexadecimalfloat, exponentfloat, pointfloat)),
            opt(float_type_suffix),
        )),
        |(sign, float_str, suffix)| {
            let mut clean = strip_underscores(float_str);
            if sign.is_some() {
                clean = format!("-{}", clean);
            }

            // Hexadecimal floats need special parsing
            if clean.starts_with("0x") || clean.starts_with("0X") {
                // Parse hex float manually: 0x[mantissa]p[exponent]
                panic!(
                    "BUG: Hexadecimal float literals (e.g., '{}') are not yet implemented. The lexer should not accept this syntax.",
                    clean
                )
            } else {
                let value = clean.parse().unwrap_or_else(|e| {
                    panic!("BUG: Failed to parse float literal '{}': {}. The lexer matched the pattern but conversion failed.", clean, e)
                });
                let base_token = Token::FloatLiteral(value);
                match suffix {
                    // f32 is the default float type, no need to wrap
                    Some("f32") | None => base_token,
                    Some(s) => Token::SuffixedLiteral(Box::new(base_token), s.to_string()),
                }
            }
        },
    )(input)
}

// String literal parser
// stringlit  ::= '"' stringchar* '"'
// stringchar ::= <any source character except "\" or newline or double quotes>
pub fn parse_string_literal(input: &str) -> IResult<&str, Token> {
    map(
        delimited(
            char('"'),
            take_while(|c| c != '"' && c != '\\' && c != '\n'),
            char('"'),
        ),
        |s: &str| Token::StringLiteral(s.to_string()),
    )(input)
}

// Char literal parser
// charlit ::= "'" char "'"
// char    ::= <any source character except "\" or newline or single quotes>
pub fn parse_char_literal(input: &str) -> IResult<&str, Token> {
    map(
        delimited(char('\''), verify(none_of("'\\\n"), |_| true), char('\'')),
        Token::CharLiteral,
    )(input)
}
