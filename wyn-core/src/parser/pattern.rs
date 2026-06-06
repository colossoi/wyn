use crate::ast::*;
use crate::error::Result;
use crate::lexer::Token;
use crate::parser::Parser;
use crate::{bail_parse, bail_parse_at, err_parse};
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

            Some(Token::AtBracket) => self.parse_vec_pattern(),

            Some(Token::Identifier(_)) => {
                // Simple name binding. (Sum-type constructor patterns
                // start with `Token::Constructor`, not an identifier.)
                let span = self.current_span();
                let name = self.expect_identifier()?;
                Ok(self.node_counter.mk_node(PatternKind::Name(name), span))
            }

            Some(Token::Constructor(_)) => self.parse_constructor_pattern(),

            Some(Token::IntLiteral(_))
            | Some(Token::FloatLiteral(_))
            | Some(Token::True)
            | Some(Token::False) => self.parse_pattern_literal(),

            // Negative literal — the lexer emits unary minus as
            // `BinOp("-")`, never the dedicated `Token::Minus`.
            Some(Token::BinOp(op)) if op == "-" => self.parse_pattern_literal(),

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

    /// Parse `@[pat1, pat2, …]` — the positional inverse of the `@[…]`
    /// vec constructor. All sub-patterns bind components of a vec
    /// scrutinee in order; the type checker validates arity against
    /// the scrutinee's vec size.
    fn parse_vec_pattern(&mut self) -> Result<Pattern> {
        let start_span = self.current_span();
        self.expect(Token::AtBracket)?;

        let mut patterns = Vec::new();
        if !self.check(&Token::RightBracket) {
            loop {
                patterns.push(self.parse_pattern()?);
                if !self.check(&Token::Comma) {
                    break;
                }
                self.advance();
                // Allow trailing comma before `]`.
                if self.check(&Token::RightBracket) {
                    break;
                }
            }
        }

        let end_span = self.current_span();
        self.expect(Token::RightBracket)?;
        let span = start_span.merge(&end_span);
        Ok(self.node_counter.mk_node(PatternKind::Vec(patterns), span))
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
        let constructor = match self.peek() {
            Some(Token::Constructor(name)) => {
                let n = name.clone();
                self.advance();
                n
            }
            other => bail_parse_at!(self.current_span(), "Expected `#name`, got {:?}", other),
        };

        // Optional payload list: `#name(p1, p2, ...)`. Bare `#name` is
        // a nullary constructor pattern.
        let mut args = Vec::new();
        let mut end_span = start_span;
        if self.check(&Token::LeftParen) {
            self.advance(); // consume `(`
            if !self.check(&Token::RightParen) {
                loop {
                    args.push(self.parse_pattern()?);
                    if self.check(&Token::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            end_span = self.current_span();
            self.expect(Token::RightParen)?;
        }

        let span = start_span.merge(&end_span);
        Ok(self.node_counter.mk_node(PatternKind::Constructor(constructor, args), span))
    }

    /// Parse a pattern literal:
    /// ```text
    /// pat_literal ::= [ "-" ] intnumber
    ///               | [ "-" ] floatnumber
    ///               | "true"
    ///               | "false"
    /// ```
    fn parse_pattern_literal(&mut self) -> Result<Pattern> {
        trace!("parse_pattern_literal: next token = {:?}", self.peek());
        let start_span = self.current_span();

        // Check for negative sign (lexer emits unary minus as `BinOp("-")`).
        let is_negative = match self.peek() {
            Some(Token::BinOp(op)) if op == "-" => {
                self.advance();
                true
            }
            _ => false,
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
