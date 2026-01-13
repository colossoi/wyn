use crate::ast::*;
use crate::error::Result;
use crate::lexer::Token;
use crate::parser::Parser;
use crate::{bail_parse, err_parse};
use log::trace;

impl Parser<'_> {
    /// Parse a pattern in function parameter position.
    /// In this context, type annotations must be inside parentheses: (x : i32)
    /// This avoids ambiguity with function return types.
    pub fn parse_function_parameter(&mut self) -> Result<Pattern> {
        trace!("parse_function_parameter: next token = {:?}", self.peek());
        let start_span = self.current_span();

        // Parse optional attributes
        let attributes = self.parse_attributes()?;

        let pattern = if !attributes.is_empty() {
            // #[attr] pat
            let inner = self.parse_pattern_without_attributes()?;
            let span = start_span.merge(&inner.h.span);
            self.node_counter.mk_node(PatternKind::Attributed(attributes, Box::new(inner)), span)
        } else {
            self.parse_pattern_without_attributes()?
        };

        // In function parameter context, don't allow bare `: type` syntax
        // Type annotations must be inside parentheses: (x : i32)
        Ok(pattern)
    }

    /// Parse a pattern according to the grammar:
    /// ```text
    /// pat ::= name
    ///       | pat_literal
    ///       | "_"
    ///       | "(" ")"
    ///       | "(" pat ")"
    ///       | "(" pat ("," pat)+ [","] ")"
    ///       | "{" "}"
    ///       | "{" fieldid ["=" pat] ("," fieldid ["=" pat])* [","] "}"
    ///       | constructor pat*
    ///       | pat ":" type
    ///       | "#[" attr "]" pat
    /// ```
    pub fn parse_pattern(&mut self) -> Result<Pattern> {
        trace!("parse_pattern: next token = {:?}", self.peek());
        let start_span = self.current_span();

        // Parse optional attributes
        let attributes = self.parse_attributes()?;

        trace!("done parsing attributes: next token = {:?}", self.peek());
        let pattern = if !attributes.is_empty() {
            // #[attr] pat
            let inner = self.parse_pattern_without_attributes()?;
            let span = start_span.merge(&inner.h.span);
            self.node_counter.mk_node(PatternKind::Attributed(attributes, Box::new(inner)), span)
        } else {
            self.parse_pattern_without_attributes()?
        };
        trace!("done parsing primary pattern: next token = {:?}", self.peek());

        // Check for type annotation (pat : type)
        if self.check(&Token::Colon) {
            self.advance();
            trace!("parsing pattern type suffix: next token = {:?}", self.peek());
            let ty = self.parse_type()?;
            // Use the pattern's span since Type doesn't have a span field
            let span = pattern.h.span;
            return Ok(self.node_counter.mk_node(PatternKind::Typed(Box::new(pattern), ty), span));
        }

        Ok(pattern)
    }

    fn parse_pattern_without_attributes(&mut self) -> Result<Pattern> {
        match self.peek() {
            Some(Token::Underscore) => {
                let span = self.current_span();
                self.advance();
                Ok(self.node_counter.mk_node(PatternKind::Wildcard, span))
            }

            Some(Token::LeftParen) => self.parse_paren_pattern(),

            Some(Token::LeftBrace) => self.parse_record_pattern(),

            Some(Token::Identifier(name)) => {
                // Check if it's a constructor (starts with uppercase)
                if name.chars().next().is_some_and(|c| c.is_uppercase()) {
                    self.parse_constructor_pattern()
                } else {
                    // Simple name binding
                    let span = self.current_span();
                    let name = self.expect_identifier()?;
                    Ok(self.node_counter.mk_node(PatternKind::Name(name), span))
                }
            }

            Some(Token::IntLiteral(_))
            | Some(Token::FloatLiteral(_))
            | Some(Token::True)
            | Some(Token::False)
            | Some(Token::CharLiteral(_)) => self.parse_pattern_literal(),

            Some(Token::Minus) => {
                // Negative literal
                self.parse_pattern_literal()
            }

            _ => Err(err_parse!("Expected pattern, got {:?}", self.peek())),
        }
    }

    fn parse_paren_pattern(&mut self) -> Result<Pattern> {
        let start_span = self.current_span();
        self.expect(Token::LeftParen)?;

        if self.check(&Token::RightParen) {
            // () unit pattern
            let end_span = self.current_span();
            self.advance();
            let span = start_span.merge(&end_span);
            return Ok(self.node_counter.mk_node(PatternKind::Unit, span));
        }

        let first = self.parse_pattern()?;

        if self.check(&Token::Comma) {
            // Tuple pattern: (pat, pat, ...)
            let mut patterns = vec![first];

            while self.check(&Token::Comma) {
                self.advance();
                // Allow trailing comma
                if self.check(&Token::RightParen) {
                    break;
                }
                patterns.push(self.parse_pattern()?);
            }

            let end_span = self.current_span();
            self.expect(Token::RightParen)?;
            let span = start_span.merge(&end_span);
            Ok(self.node_counter.mk_node(PatternKind::Tuple(patterns), span))
        } else {
            // Single pattern in parens: (pat)
            self.expect(Token::RightParen)?;
            Ok(first)
        }
    }

    fn parse_record_pattern(&mut self) -> Result<Pattern> {
        let start_span = self.current_span();
        self.expect(Token::LeftBrace)?;

        if self.check(&Token::RightBrace) {
            // {} empty record
            let end_span = self.current_span();
            self.advance();
            let span = start_span.merge(&end_span);
            return Ok(self.node_counter.mk_node(PatternKind::Record(vec![]), span));
        }

        let mut fields = Vec::new();

        loop {
            let field_name = self.expect_identifier()?;

            let pattern = if self.check(&Token::Assign) {
                // field = pat
                self.advance();
                Some(self.parse_pattern()?)
            } else {
                // Shorthand: just field name
                None
            };

            fields.push(RecordPatternField {
                field: field_name,
                pattern,
            });

            if !self.check(&Token::Comma) {
                break;
            }
            self.advance();

            // Allow trailing comma
            if self.check(&Token::RightBrace) {
                break;
            }
        }

        let end_span = self.current_span();
        self.expect(Token::RightBrace)?;
        let span = start_span.merge(&end_span);
        Ok(self.node_counter.mk_node(PatternKind::Record(fields), span))
    }

    fn parse_constructor_pattern(&mut self) -> Result<Pattern> {
        let start_span = self.current_span();
        let constructor = self.expect_identifier()?;

        // Parse constructor arguments (zero or more patterns)
        let mut args = Vec::new();
        let mut end_span = start_span;

        // Keep parsing patterns as long as the next token can start a pattern
        // but stop if we see tokens that indicate the end of the constructor pattern
        while self.can_start_pattern() && !self.is_pattern_terminator() {
            let arg = self.parse_pattern_without_attributes()?;
            end_span = arg.h.span;
            args.push(arg);
        }

        let span = start_span.merge(&end_span);
        Ok(self.node_counter.mk_node(PatternKind::Constructor(constructor, args), span))
    }

    fn can_start_pattern(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::Identifier(_))
                | Some(Token::Underscore)
                | Some(Token::LeftParen)
                | Some(Token::LeftBrace)
                | Some(Token::IntLiteral(_))
                | Some(Token::FloatLiteral(_))
                | Some(Token::CharLiteral(_))
                | Some(Token::True)
                | Some(Token::False)
                | Some(Token::Minus)
        )
    }

    fn is_pattern_terminator(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::Assign)
                | Some(Token::In)
                | Some(Token::Arrow)
                | Some(Token::RightParen)
                | Some(Token::RightBrace)
                | Some(Token::Comma)
                | Some(Token::Colon)
        )
    }

    /// Parse a pattern literal:
    /// ```text
    /// pat_literal ::= [ "-" ] intnumber
    ///               | [ "-" ] floatnumber
    ///               | charlit
    ///               | "true"
    ///               | "false"
    /// ```
    fn parse_pattern_literal(&mut self) -> Result<Pattern> {
        trace!("parse_pattern_literal: next token = {:?}", self.peek());
        let start_span = self.current_span();

        // Check for negative sign
        let is_negative = if self.check(&Token::Minus) {
            self.advance();
            true
        } else {
            false
        };

        let literal = match self.peek() {
            Some(Token::IntLiteral(n)) => {
                let value = n.clone();
                self.advance();
                PatternLiteral::Int(if is_negative { value.negated() } else { value })
            }

            Some(Token::FloatLiteral(f)) => {
                let value = *f;
                self.advance();
                PatternLiteral::Float(if is_negative { -value } else { value })
            }

            Some(Token::CharLiteral(c)) => {
                if is_negative {
                    bail_parse!("Character literals cannot be negative");
                }
                let ch = *c;
                self.advance();
                PatternLiteral::Char(ch)
            }

            Some(Token::True) => {
                if is_negative {
                    bail_parse!("Boolean literals cannot be negative");
                }
                self.advance();
                PatternLiteral::Bool(true)
            }

            Some(Token::False) => {
                if is_negative {
                    bail_parse!("Boolean literals cannot be negative");
                }
                self.advance();
                PatternLiteral::Bool(false)
            }

            _ => {
                bail_parse!("Expected literal in pattern, got {:?}", self.peek());
            }
        };

        let end_span = self.current_span();
        let span = start_span.merge(&end_span);
        Ok(self.node_counter.mk_node(PatternKind::Literal(literal), span))
    }
}
