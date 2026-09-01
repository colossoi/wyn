use crate::ast::*;
use crate::error::Result;
use crate::interface::{Attribute, EntryKind, EntryOutputDecl};
use crate::lexer;
use crate::lexer::{LocatedToken, Token};
use crate::op;
use crate::types;
use crate::LookupMap;
use crate::{bail_parse_at, err_parse, err_parse_at};
use log::trace;
use std::sync::OnceLock;
use wyn_base::IdSource;
use wyn_module_graph::{ImportSiteId, ModuleId, TextRange};

mod module;
mod pattern;
#[cfg(test)]
mod tests;

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
enum ParsedTag {}

#[cfg(test)]
type Parsed = Program<ParsedTag, ParsedFamily, crate::module_manager::ModuleManager>;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParsedFile {
    pub(crate) declarations: Vec<Declaration<ParsedFamily>>,
    pub(crate) imports: Vec<SourceImport>,
}

pub(crate) fn parse_file(
    module: ModuleId,
    source: &str,
    node_ids: &mut NodeCounter,
    graphics: bool,
) -> Result<ParsedFile> {
    let tokens = lexer::tokenize(module, source).map_err(|error| err_parse!("{}", error))?;
    let mut parser = Parser::with_graphics(module, tokens, node_ids, graphics);
    let declarations = parser.parse()?;
    Ok(ParsedFile {
        declarations,
        imports: parser.imports,
    })
}

// Lazily initialized type constructor maps
static VECTOR_TYPES: OnceLock<LookupMap<String, Type>> = OnceLock::new();
static MATRIX_TYPES: OnceLock<LookupMap<String, Type>> = OnceLock::new();

fn get_vector_types() -> &'static LookupMap<String, Type> {
    VECTOR_TYPES.get_or_init(types::vector_type_constructors)
}

fn get_matrix_types() -> &'static LookupMap<String, Type> {
    MATRIX_TYPES.get_or_init(types::matrix_type_constructors)
}

/// Argument in a function call: either a call-section placeholder (`_`) or a
/// regular expression.
enum CallArg {
    Placeholder,
    Expr(Expression),
}

/// Convert a type suffix string (e.g., "u32", "f64") to its Type representation
fn suffix_to_type(suffix: &str) -> Type {
    let type_name = match suffix {
        "f16" => TypeName::Float(16),
        "f32" => TypeName::Float(32),
        "f64" => TypeName::Float(64),
        "i8" => TypeName::Int(8),
        "i16" => TypeName::Int(16),
        "i32" => TypeName::Int(32),
        "i64" => TypeName::Int(64),
        "u8" => TypeName::UInt(8),
        "u16" => TypeName::UInt(16),
        "u32" => TypeName::UInt(32),
        "u64" => TypeName::UInt(64),
        _ => TypeName::Named(suffix.to_string()),
    };
    Type::Constructed(type_name, vec![])
}

pub(crate) struct Parser<'a> {
    module: ModuleId,
    tokens: Vec<LocatedToken>,
    current: usize,
    node_counter: &'a mut NodeCounter,
    import_sites: IdSource<ImportSiteId>,
    imports: Vec<SourceImport>,
    graphics: bool,
}

impl<'a> Parser<'a> {
    #[cfg(test)]
    pub(crate) fn new(tokens: Vec<LocatedToken>, node_counter: &'a mut NodeCounter) -> Self {
        let module =
            tokens.first().and_then(|token| token.span.module()).unwrap_or_else(|| ModuleId::from(0));
        Self::with_graphics(module, tokens, node_counter, true)
    }

    pub(crate) fn with_graphics(
        module: ModuleId,
        tokens: Vec<LocatedToken>,
        node_counter: &'a mut NodeCounter,
        graphics: bool,
    ) -> Self {
        Parser {
            module,
            tokens,
            current: 0,
            node_counter,
            import_sites: IdSource::new(),
            imports: Vec::new(),
            graphics,
        }
    }

    /// Get the span of the current token, or a zero-width span at the
    /// end of the last token when we've consumed everything (so error
    /// diagnostics at EOF point at where the missing token *would* be).
    /// Returns the "before-beginning" span `(0,0,0,0)` only on empty
    /// input.
    fn current_span(&self) -> Span {
        if let Some(t) = self.tokens.get(self.current) {
            return t.span;
        }
        match self.tokens.last() {
            Some(last) => Span::new(
                self.module,
                TextRange::new(last.span.range().end(), last.span.range().end())
                    .unwrap_or_else(|error| panic!("invalid end-of-file span: {error}")),
            ),
            None => self.start_of_file_span(),
        }
    }

    /// Get the span of the previous token. Returns the "before-beginning"
    /// span `(0,0,0,0)` when no token has been consumed yet.
    fn previous_span(&self) -> Span {
        if self.current > 0 {
            // `current > 0` and the parser never advances past `tokens.len()`,
            // so this index is always in-bounds.
            self.tokens[self.current - 1].span
        } else {
            self.start_of_file_span()
        }
    }

    fn start_of_file_span(&self) -> Span {
        Span::new(
            self.module,
            TextRange::new(0, 0).unwrap_or_else(|error| panic!("invalid start-of-file span: {error}")),
        )
    }

    pub(crate) fn parse(&mut self) -> Result<Vec<Declaration<ParsedFamily>>> {
        let mut declarations = Vec::new();

        while !self.is_at_end() {
            declarations.push(self.parse_declaration()?);
        }

        Ok(declarations)
    }

    fn parse_declaration(&mut self) -> Result<Declaration<ParsedFamily>> {
        trace!("parse_declaration: next token = {:?}", self.peek());
        // Parse optional attributes
        let attributes = self.parse_attributes()?;
        match self.peek() {
            Some(Token::Let) => self.parse_decl("let", attributes),
            Some(Token::Def) => self.parse_decl("def", attributes),
            Some(Token::Entry) => self.parse_entry_decl(attributes),
            Some(Token::Sig) => {
                let mut decl = self.parse_sig_decl()?;
                decl.attributes = attributes;
                Ok(Declaration::Frontend(ParsedFrontend::Sig(decl)))
            }
            Some(Token::Type) | Some(Token::TypeSizeLifted) | Some(Token::TypeFullyLifted) => {
                let type_bind = self.parse_type_bind()?;
                Ok(Declaration::Frontend(ParsedFrontend::TypeBind(type_bind)))
            }
            Some(Token::Module) => {
                // Check if it's "module type" or just "module"
                let saved_pos = self.current;
                self.advance();
                if self.check(&Token::Type) {
                    // module type declaration
                    self.current = saved_pos;
                    let mod_type_bind = self.parse_module_type_bind()?;
                    Ok(Declaration::Frontend(ParsedFrontend::ModuleTypeBind(
                        mod_type_bind,
                    )))
                } else {
                    // module declaration
                    self.current = saved_pos;
                    Ok(Declaration::Frontend(ParsedFrontend::Module(
                        self.parse_module_decl()?,
                    )))
                }
            }
            Some(Token::Functor) => Ok(Declaration::Frontend(ParsedFrontend::Module(
                self.parse_functor_decl()?,
            ))),
            Some(Token::Open) => {
                self.advance();
                let mod_exp = self.parse_module_expression()?;
                Ok(Declaration::Frontend(ParsedFrontend::Open(mod_exp)))
            }
            Some(Token::Import) => {
                let import = self.parse_source_import()?;
                Ok(Declaration::Frontend(ParsedFrontend::Import(import)))
            }
            Some(Token::Extern) => self.parse_extern_decl(attributes),
            Some(Token::Resource) => Err(err_parse_at!(
                self.current_span(),
                "resource declarations are not part of the language; resources are values supplied through entry parameters"
            )),
            _ => Err(err_parse_at!(
                self.current_span(),
                "Expected declaration, got {:?}",
                self.peek()
            )),
        }
    }

    fn parse_nested_declaration(&mut self) -> Result<NestedDeclaration> {
        Ok(match self.parse_declaration()? {
            Declaration::Decl(decl) => NestedDeclaration::Decl(decl),
            Declaration::Entry(entry) => NestedDeclaration::Entry(entry),
            Declaration::Extern(extern_decl) => NestedDeclaration::Extern(extern_decl),
            Declaration::Frontend(frontend) => match frontend {
                ParsedFrontend::Sig(decl) => NestedDeclaration::Sig(decl),
                ParsedFrontend::TypeBind(decl) => NestedDeclaration::TypeBind(decl),
                ParsedFrontend::Module(decl) => NestedDeclaration::Module(decl),
                ParsedFrontend::ModuleTypeBind(decl) => NestedDeclaration::ModuleTypeBind(decl),
                ParsedFrontend::Open(expression) => NestedDeclaration::Open(expression),
                ParsedFrontend::Import(import) => NestedDeclaration::Import(import),
                ParsedFrontend::Resource(decl) => NestedDeclaration::Resource(decl),
            },
        })
    }

    fn expect_string_literal(&mut self) -> Result<String> {
        match self.peek() {
            Some(Token::StringLiteral(s)) => {
                let s = s.clone();
                self.advance();
                Ok(s)
            }
            _ => Err(err_parse!("Expected string literal")),
        }
    }

    fn parse_source_import(&mut self) -> Result<SourceImport> {
        let start = self.current_span();
        self.expect(Token::Import)?;
        let path = self.expect_string_literal()?;
        let span = start.merge(&self.previous_span());
        let import = SourceImport {
            site: self.import_sites.next_id(),
            path,
            span,
        };
        self.imports.push(import.clone());
        Ok(import)
    }

    fn parse_decl(
        &mut self,
        keyword: &'static str,
        attributes: Vec<Attribute>,
    ) -> Result<Declaration<ParsedFamily>> {
        trace!("parse_decl({}): next token = {:?}", keyword, self.peek());

        {
            // Regular declaration (let or def)
            match keyword {
                "let" => self.expect(Token::Let)?,
                "def" => self.expect(Token::Def)?,
                _ => bail_parse_at!(self.current_span(), "Invalid keyword: {}", keyword),
            }

            // Parse name - either an identifier or an operator in parentheses like (+) or (+^)
            let name = if self.peek() == Some(&Token::LeftParen) {
                let op = self.parse_operator_section()?;
                format!("({})", op)
            } else {
                self.expect_identifier()?
            };
            let name_span = self.previous_span();

            // Rust-style generics: <[n], A, B> (optional, only for def)
            let (size_params, type_params) = if keyword == "def" && self.check_binop("<") {
                self.parse_generic_params()?
            } else {
                (vec![], vec![])
            };

            // For def: either function syntax or constant binding
            //   - def foo(x: T, y: U) R = ...  (function with params)
            //   - def foo: T = ...              (constant binding)
            // For let: type annotation with colon: let x: type = expr - no params
            let (params, param_diets, ty, return_diet) = if keyword == "def" {
                if self.check(&Token::LeftParen) {
                    // Function: def foo(params) R = ...
                    let (params, param_diets) = self.parse_comma_separated_params()?;

                    // Reject zero-argument functions - use constant syntax instead
                    if params.is_empty() {
                        bail_parse_at!(
                            self.current_span(),
                            "Zero-argument functions are not allowed. Use constant syntax instead: `def {} = ...`",
                            name
                        );
                    }

                    // Return type directly after params (no arrow): def foo(x: T) R = ...
                    let (ty, return_diet) = if !self.check(&Token::Assign) {
                        let (ty, diet) = self.parse_return_type_simple()?;
                        (Some(ty), diet)
                    } else {
                        (None, Diet::observing())
                    };
                    (params, param_diets, ty, return_diet)
                } else if self.check(&Token::Colon) {
                    // Constant binding with type: def foo: T = ...
                    self.advance();
                    let (ty, diet) = self.parse_type()?;
                    (vec![], vec![], Some(ty), diet)
                } else if self.check(&Token::Assign) {
                    // Constant binding without type: def foo = ...
                    (vec![], vec![], None, Diet::observing())
                } else {
                    bail_parse_at!(self.current_span(), "Expected '(' or ':' after def name");
                }
            } else {
                // let declarations don't have params, just optional type annotation
                let (ty, diet) = if self.check(&Token::Colon) {
                    self.advance();
                    let (ty, diet) = self.parse_type()?;
                    (Some(ty), diet)
                } else {
                    (None, Diet::observing())
                };
                (vec![], vec![], ty, diet)
            };

            self.expect(Token::Assign)?;
            let body = self.parse_expression()?;

            Ok(Declaration::Decl(Decl {
                data: DefinitionSyntax { keyword, attributes },
                name,
                name_span,
                size_params,
                type_params,
                params,
                ty,
                body,
                param_diets,
                return_diet,
            }))
        }
    }

    fn parse_sig_decl(&mut self) -> Result<SigDecl> {
        trace!("parse_sig_decl: next token = {:?}", self.peek());
        self.expect(Token::Sig)?;

        // Parse name - either an identifier or an operator in parentheses like (+) or (**)
        let name = if self.check(&Token::LeftParen) {
            self.parse_operator_section()?
        } else {
            self.expect_identifier()?
        };

        // Rust-style generics: <[n], A, B> (optional)
        let (size_params, type_params) =
            if self.check_binop("<") { self.parse_generic_params()? } else { (vec![], vec![]) };

        let (ty, param_diets, return_diet) = self.parse_sig_type()?;

        Ok(SigDecl {
            attributes: vec![],
            name,
            size_params,
            type_params,
            ty,
            param_diets,
            return_diet,
        })
    }

    /// Parse the type tail of a `sig`, written like the `def` it describes —
    /// a function declaration without its `= body`:
    ///   `(p: A, q: B) R`  → builds the curried arrow `A -> B -> R`
    ///   `: T`             → a constant of type `T` (nullary, like `def x: T`)
    /// There is no arrow-chain spelling; a `sig` mirrors the function's own
    /// parameter list so the two read in parallel.
    fn parse_sig_type(&mut self) -> Result<(Type, Vec<Diet>, Diet)> {
        if self.check(&Token::LeftParen) {
            let (params, param_diets) = self.parse_extern_params()?;
            if params.is_empty() {
                bail_parse_at!(
                    self.current_span(),
                    "Zero-argument sig must use constant syntax: `sig name: T`"
                );
            }
            let (ret, return_diet) = self.parse_return_type_simple()?;
            // Fold params right-to-left into the curried arrow representation.
            let ty = params.into_iter().rev().fold(ret, |acc, (_, pty)| types::function(pty, acc));
            Ok((ty, param_diets, return_diet))
        } else if self.check(&Token::Colon) {
            self.advance();
            let (ty, diet) = self.parse_type()?;
            Ok((ty, vec![], diet))
        } else {
            bail_parse_at!(self.current_span(), "Expected '(' or ':' after sig name")
        }
    }

    /// Parse an extern declaration for linked SPIR-V functions.
    /// Syntax: `#[linked("linkage_name")] extern name(param: Type, ...) ReturnType`
    fn parse_extern_decl(&mut self, attributes: Vec<Attribute>) -> Result<Declaration<ParsedFamily>> {
        trace!("parse_extern_decl: next token = {:?}", self.peek());
        let start_span = self.current_span();

        // Find the linked attribute
        let linkage_name = attributes
            .iter()
            .find_map(
                |attr| {
                    if let Attribute::Linked(name) = attr {
                        Some(name.clone())
                    } else {
                        None
                    }
                },
            )
            .ok_or_else(|| err_parse!("extern declaration requires #[linked(\"name\")] attribute"))?;

        self.expect(Token::Extern)?;
        let name = self.expect_identifier()?;

        // Parse optional type parameters: <[n], [m], T>
        let (size_params, type_params) =
            if self.check_binop("<") { self.parse_generic_params()? } else { (vec![], vec![]) };

        // Parse parameters: (param: Type, ...)
        let (params, param_diets) = self.parse_extern_params()?;

        // Parse return type (required for extern functions)
        let (ret_type, return_diet) = self.parse_return_type_simple()?;

        let end_span = self.current_span();

        // Build function type from params and return type
        let ty = if params.is_empty() {
            ret_type
        } else {
            // Build curried function type: T1 -> T2 -> ... -> Ret
            params.into_iter().rev().fold(ret_type, |acc, (_, param_ty)| types::function(param_ty, acc))
        };

        Ok(Declaration::Extern(ExternDecl {
            name,
            data: ExternSyntax {
                linkage_name,
                size_params,
                type_params,
                ty,
                span: start_span.merge(&end_span),
                param_diets,
                return_diet,
            },
        }))
    }

    /// Parse extern function parameters: (name: Type, ...)
    /// Returns (name, type) pairs.
    fn parse_extern_params(&mut self) -> Result<(Vec<(String, Type)>, Vec<Diet>)> {
        self.expect(Token::LeftParen)?;
        let params = self.parse_delimited_list(&Token::RightParen, false, |parser| {
            let param_name = parser.expect_identifier()?;
            parser.expect(Token::Colon)?;
            let (ty, diet) = parser.parse_type()?;
            Ok(((param_name, ty), diet))
        })?;
        self.expect(Token::RightParen)?;
        Ok(params.into_iter().unzip())
    }

    /// Parse a host-visible root entry.
    /// Entries have restrictive syntax: only `id: type` parameters, not general patterns.
    /// Syntax is `entry name(id: type, ...) return_type = body`.
    fn parse_entry_decl(&mut self, attributes: Vec<Attribute>) -> Result<Declaration<ParsedFamily>> {
        trace!("parse_entry_decl: next token = {:?}", self.peek());

        if !attributes.is_empty() {
            bail_parse_at!(
                self.current_span(),
                "attributes are not valid on an entry declaration"
            );
        }
        let entry_kind = EntryKind::Root;
        let compute_dispatch = None;

        self.expect(Token::Entry)?;
        let name = self.expect_identifier()?;
        let name_span = self.previous_span();

        // Parse optional type parameters: <[n], [m], T>
        let (size_params, type_params) =
            if self.check_binop("<") { self.parse_generic_params()? } else { (vec![], vec![]) };

        // Parse restrictive parameters: (id: type, id: type, ...)
        // Only typed identifiers allowed, not general patterns
        let (params, param_diets) = self.parse_entry_params()?;

        // Parse return type (which may have optional attributes) - no arrow required
        let (return_types, return_attributes, return_diets) =
            if self.check(&Token::AttributeStart) || self.check(&Token::LeftParen) {
                // Attributed return type(s)
                self.parse_return_type()?
            } else if !self.check(&Token::Assign) {
                // Simple unattributed return type
                let (ty, diet) = self.parse_type()?;
                (vec![ty], vec![None], vec![diet])
            } else {
                bail_parse_at!(
                    self.current_span(),
                    "Entry point declarations must have an explicit return type"
                );
            };
        // One output → its diet; several → an aggregate mirroring the tuple.
        let return_diet = if let [diet] = return_diets.as_slice() {
            diet.clone()
        } else {
            Diet::Aggregate {
                unique: false,
                components: return_diets,
            }
        };

        // Combine into EntryOutputDecl structs
        let outputs: Vec<EntryOutputDecl> = return_types
            .into_iter()
            .zip(return_attributes)
            .map(|(ty, attribute)| EntryOutputDecl { ty, attribute })
            .collect();

        self.expect(Token::Assign)?;
        let body = self.parse_expression()?;

        Ok(Declaration::Entry(EntryDecl {
            data: EntrySyntax {
                entry_kind,
                compute_dispatch,
                outputs,
                param_diets,
                return_diet,
            },
            name,
            name_span,
            size_params,
            type_params,
            params,
            body,
        }))
    }

    /// Parse entry point parameters with restrictive syntax.
    /// Only allows `id: type` or `#[attr] id: type`, not general patterns.
    fn parse_entry_params(&mut self) -> Result<(Vec<Pattern>, Vec<Diet>)> {
        trace!("parse_entry_params: next token = {:?}", self.peek());
        let _start_span = self.current_span();
        self.expect(Token::LeftParen)?;
        let params = self.parse_delimited_list(&Token::RightParen, false, |parser| {
            let param_start = parser.current_span();
            let attrs =
                if parser.check(&Token::AttributeStart) { parser.parse_attributes()? } else { vec![] };
            let name = parser.expect_identifier()?;
            let name_span = parser.previous_span();
            parser.expect(Token::Colon)?;
            let (ty, diet) = parser.parse_type()?;
            let name_pat = parser.node_counter.mk_node(PatternKind::Name(name), name_span);
            let inner_pat = if attrs.is_empty() {
                name_pat
            } else {
                let span = param_start.merge(&name_span);
                parser.node_counter.mk_node(PatternKind::Attributed(attrs, Box::new(name_pat)), span)
            };
            let span = param_start.merge(&parser.previous_span());
            let typed_pat = parser.node_counter.mk_node(PatternKind::Typed(Box::new(inner_pat), ty), span);
            Ok((typed_pat, diet))
        })?;
        self.expect(Token::RightParen)?;
        Ok(params.into_iter().unzip())
    }

    /// Parse return type with optional attributes, returning parallel arrays
    /// Returns (return_types, return_attributes)
    fn parse_return_type(&mut self) -> Result<(Vec<Type>, Vec<Option<Attribute>>, Vec<Diet>)> {
        trace!("parse_return_type: next token = {:?}", self.peek());

        // Check if it's a tuple: ([attr1] type1, [attr2] type2, ...)
        if self.check(&Token::LeftParen) {
            self.advance(); // consume '('
            let returns = self.parse_delimited_list(&Token::RightParen, false, |parser| {
                let attribute = if parser.check(&Token::AttributeStart) {
                    parser.advance(); // consume '#['
                    let attribute = parser.parse_attribute()?;
                    Some(attribute)
                } else {
                    None
                };
                let (ty, diet) = parser.parse_type()?;
                Ok(((ty, attribute), diet))
            })?;
            self.expect(Token::RightParen)?;
            let (returns, diets): (Vec<_>, Vec<_>) = returns.into_iter().unzip();
            let (types, attributes) = returns.into_iter().unzip();
            Ok((types, attributes, diets))
        } else if self.check(&Token::AttributeStart) {
            // Single attributed type: #[attribute] type
            self.advance(); // consume '#['
            let attribute = self.parse_attribute()?;
            let (ty, diet) = self.parse_type()?;

            Ok((vec![ty], vec![Some(attribute)], vec![diet]))
        } else {
            // Regular single return type without attributes
            let (ty, diet) = self.parse_type()?;
            Ok((vec![ty], vec![None], vec![diet]))
        }
    }

    fn parse_attribute(&mut self) -> Result<Attribute> {
        trace!("parse_attribute: next token = {:?}", self.peek());
        let attr_name = self.expect_identifier()?;

        match attr_name.as_str() {
            "vertex" | "fragment" | "compute" => bail_parse_at!(
                self.current_span(),
                "#[{}] is not part of the language; a stage context is determined by the invocation operation receiving its callback",
                attr_name
            ),
            "dispatch" => bail_parse_at!(
                self.current_span(),
                "#[dispatch(...)] is not part of the language; array operations determine their own execution",
            ),
            "uniform" | "storage" | "texture" | "sampler" | "storage_image" | "view" => {
                bail_parse_at!(
                    self.current_span(),
                    "#[{}(...)] is not part of the language; external resources have no source-language bindings or views",
                    attr_name
                )
            }
            "builtin" | "target" | "vertex_slot" | "varying" => bail_parse_at!(
                self.current_span(),
                "#[{}(...)] is not part of the language; invocation arguments, payloads, and render targets are expressed by typed values",
                attr_name
            ),
            "size_hint" => {
                // Parse size hint for dynamic arrays: #[size_hint(N)]
                self.expect(Token::LeftParen)?;
                let hint = self.expect_integer()?;
                self.expect(Token::RightParen)?;
                self.expect(Token::RightBracket)?;
                let hint = std::num::NonZeroU32::new(hint)
                    .ok_or_else(|| err_parse!("#[size_hint(N)] requires N > 0"))?;
                Ok(Attribute::SizeHint(hint))
            }
            "linked" => {
                // Parse linked SPIR-V function: #[linked("linkage_name")]
                self.expect(Token::LeftParen)?;
                let linkage_name = if let Some(Token::StringLiteral(s)) = self.advance() {
                    s.to_string()
                } else {
                    bail_parse_at!(self.current_span(), "Expected string literal for linkage name");
                };
                self.expect(Token::RightParen)?;
                self.expect(Token::RightBracket)?;
                Ok(Attribute::Linked(linkage_name))
            }
            _ => Err(err_parse!("Unknown attribute: {}", attr_name)),
        }
    }

    fn parse_attributes(&mut self) -> Result<Vec<Attribute>> {
        trace!("parse_attributes: next token = {:?}", self.peek());
        let mut attributes = Vec::new();

        while self.check(&Token::AttributeStart) {
            self.advance(); // consume '#['
            let attribute = self.parse_attribute()?;
            attributes.push(attribute);
        }

        Ok(attributes)
    }

    /// Parse generic parameters: <[n], [m], A, B>
    /// Returns (size_params, type_params)
    fn parse_generic_params(&mut self) -> Result<(Vec<String>, Vec<String>)> {
        trace!("parse_generic_params: next token = {:?}", self.peek());
        self.expect_binop("<")?;
        let mut size_params = Vec::new();
        let mut type_params = Vec::new();

        if !self.check_binop(">") {
            loop {
                if self.check(&Token::LeftBracket) || self.check(&Token::LeftBracketSpaced) {
                    // Size param: [n]
                    self.advance();
                    size_params.push(self.expect_identifier()?);
                    self.expect(Token::RightBracket)?;
                } else if let Some(Token::Identifier(name)) = self.peek() {
                    // Type param: must be uppercase
                    let name = name.clone();
                    if !name.chars().next().is_some_and(|c| c.is_uppercase()) {
                        bail_parse_at!(
                            self.current_span(),
                            "Type parameters must be uppercase (got '{}')",
                            name
                        );
                    }
                    self.advance();
                    type_params.push(name);
                } else {
                    bail_parse_at!(
                        self.current_span(),
                        "Expected size parameter [n] or type parameter in generics"
                    );
                }

                if !self.check(&Token::Comma) {
                    break;
                }
                self.advance(); // consume comma
            }
        }

        self.expect_binop(">")?;
        Ok((size_params, type_params))
    }

    /// Parse comma-separated function parameters: (x: T, y: U)
    fn parse_comma_separated_params(&mut self) -> Result<(Vec<Pattern>, Vec<Diet>)> {
        trace!("parse_comma_separated_params: next token = {:?}", self.peek());
        self.expect(Token::LeftParen)?;
        let params = self.parse_delimited_list(&Token::RightParen, false, |parser| {
            parser.parse_pattern_with_diet()
        })?;
        self.expect(Token::RightParen)?;
        Ok(params.into_iter().unzip())
    }

    fn parse_type(&mut self) -> Result<(Type, Diet)> {
        trace!("parse_type: next token = {:?}", self.peek());

        // Existential size quantifier `?[n].T` (spec line 243):
        //   existential_size ::= "?" ("[" name "]")+ "." type
        // Valid anywhere a type can appear (the type's existential sizes
        // become fresh during checking). Lifted-type declarations require
        // this — `type~ bag = ?[n].[n]i32`.
        if self.check(&Token::QuestionMark) {
            return self.parse_existential_type();
        }

        // Check for named parameter syntax: (name: type) -> ...
        // We parse this for documentation but drop the name, keeping just the type
        if self.check(&Token::LeftParen) {
            let saved_pos = self.current;
            self.advance(); // consume '('

            // Try to parse as named parameter
            if let Some(Token::Identifier(_name)) = self.peek() {
                self.advance();

                if self.check(&Token::Colon) {
                    // It's a named parameter - parse but drop the name
                    self.advance(); // consume ':'
                    let (param_type, param_diet) = self.parse_type()?;
                    self.expect(Token::RightParen)?;

                    // Must be followed by ->
                    if self.check(&Token::Arrow) {
                        self.advance();
                        let (return_type, return_diet) = self.parse_type()?;
                        // Just use the param_type directly, ignoring the name
                        return Ok((
                            types::function(param_type, return_type),
                            Diet::Arrow(Box::new(param_diet), Box::new(return_diet)),
                        ));
                    } else {
                        bail_parse_at!(self.current_span(), "Named parameter must be followed by ->");
                    }
                }
            }

            // Not a named parameter, restore position and parse normally
            self.current = saved_pos;
        }

        // Regular function type or type application
        let (left, left_diet) = self.parse_type_application()?;

        // Handle function arrows: T1 -> T2 -> T3
        // Arrow is right-associative: a -> b -> c means a -> (b -> c)
        if self.check(&Token::Arrow) {
            self.advance();
            let (right, right_diet) = self.parse_type()?; // Recursive call for right-associativity
            Ok((
                types::function(left, right),
                Diet::Arrow(Box::new(left_diet), Box::new(right_diet)),
            ))
        } else {
            Ok((left, left_diet))
        }
    }

    /// Parse a return type, which may include an existential quantifier.
    /// Existential types (?k. [k]T) are only valid in return position.
    fn parse_return_type_simple(&mut self) -> Result<(Type, Diet)> {
        trace!("parse_return_type_simple: next token = {:?}", self.peek());
        // Check for existential size: ?k. type or ?k l. type
        if self.check(&Token::QuestionMark) {
            return self.parse_existential_type();
        }
        self.parse_type()
    }

    fn parse_existential_type(&mut self) -> Result<(Type, Diet)> {
        self.expect(Token::QuestionMark)?;
        let mut size_vars = Vec::new();

        // Parse one or more bare identifiers: ?k. or ?k l. or ?k l m.
        while let Some(Token::Identifier(name)) = self.peek().cloned() {
            size_vars.push(name);
            self.advance();
        }

        if size_vars.is_empty() {
            bail_parse_at!(
                self.current_span(),
                "Existential type must have at least one size variable"
            );
        }

        self.expect(Token::Dot)?;
        // Uniqueness sits on the inner value; the existential just quantifies
        // a size, so the diet passes through.
        let (inner_type, inner_diet) = self.parse_type()?;

        Ok((types::existential(size_vars, inner_type), inner_diet))
    }

    fn parse_type_application(&mut self) -> Result<(Type, Diet)> {
        trace!("parse_type_application: next token = {:?}", self.peek());

        let (mut base, base_diet) = self.parse_array_or_base_type()?;

        // Explicit application of a parameterised type abbreviation or
        // spellable generic builtin: `pair<i32, bool>`, `vector<[4], f32>`,
        // or `raster<V>`. Constructors are first-order and applications must
        // be saturated; arity and kinds are checked during alias expansion.
        if self.check_binop("<") {
            let args = self.parse_type_arguments()?;
            match &mut base {
                Type::Constructed(TypeName::Named(_), base_args)
                | Type::Constructed(TypeName::Raster, base_args)
                | Type::Constructed(TypeName::Vertex, base_args)
                | Type::Constructed(TypeName::FragmentInvocation, base_args)
                | Type::Constructed(TypeName::FragmentOutput, base_args)
                | Type::Constructed(TypeName::RenderTarget, base_args)
                    if base_args.is_empty() =>
                {
                    *base_args = args;
                }
                Type::Constructed(name, _) => {
                    bail_parse_at!(
                        self.previous_span(),
                        "type '{}' does not accept explicit type arguments",
                        name
                    );
                }
                Type::Variable(_) => {
                    bail_parse_at!(
                        self.previous_span(),
                        "a type variable cannot be applied as a type constructor"
                    );
                }
            }
        }

        // Type application loop: keep applying type arguments
        // Grammar: type_application ::= type type_arg | "*" type
        //          type_arg         ::= "[" [dim] "]" | type
        // Track whether any array dimension wraps `base`: if so, the outer
        // value is an array (a diet leaf, consuming iff a `*` is buried in
        // it); otherwise the base's own diet stands.
        let mut wrapped_in_array = false;
        loop {
            if self.is_at_type_boundary() {
                break;
            }

            match self.peek() {
                // Array dimension application: [n] or []
                Some(Token::LeftBracket) | Some(Token::LeftBracketSpaced) => {
                    self.advance();
                    wrapped_in_array = true;

                    if self.check(&Token::RightBracket) {
                        // Empty brackets [] - unsized array with placeholder address space
                        // Array[elem, variant, size]
                        self.advance();
                        base = Type::Constructed(
                            TypeName::Array,
                            vec![
                                base,
                                Type::Constructed(TypeName::AddressPlaceholder, vec![]),
                                Type::Constructed(TypeName::SizePlaceholder, vec![]),
                                // region placeholder, resolved to a fresh var by resolve_placeholders.
                                Type::Constructed(TypeName::AddressPlaceholder, vec![]),
                            ],
                        );
                    } else if let Some(Token::Identifier(name)) = self.peek() {
                        // Size variable [n]
                        let size_var = name.clone();
                        self.advance();
                        self.expect(Token::RightBracket)?;
                        // Array[elem, variant, size]
                        base = Type::Constructed(
                            TypeName::Array,
                            vec![
                                base,
                                Type::Constructed(TypeName::AddressPlaceholder, vec![]),
                                types::size_var(size_var),
                                // region placeholder, resolved to a fresh var by resolve_placeholders.
                                Type::Constructed(TypeName::AddressPlaceholder, vec![]),
                            ],
                        );
                    } else if let Some(Token::IntLiteral(n)) = self.peek() {
                        // Size literal [3]
                        let size = usize::try_from(n).map_err(|_| err_parse!("Invalid array size"))?;
                        self.advance();
                        self.expect(Token::RightBracket)?;
                        // Array[elem, variant, size]
                        base = Type::Constructed(
                            TypeName::Array,
                            vec![
                                base,
                                Type::Constructed(TypeName::AddressPlaceholder, vec![]),
                                Type::Constructed(TypeName::Size(size), vec![]),
                                // region placeholder, resolved to a fresh var by resolve_placeholders.
                                Type::Constructed(TypeName::AddressPlaceholder, vec![]),
                            ],
                        );
                    } else {
                        bail_parse_at!(self.current_span(), "Expected size in array type application");
                    }
                }
                // Regular type argument application - not yet supported
                Some(Token::Identifier(_)) | Some(Token::LeftParen) | Some(Token::LeftBrace) => {
                    bail_parse_at!(
                        self.current_span(),
                        "Type constructor application (e.g., 'F T') is not yet supported"
                    );
                }
                _ => break,
            }
        }

        let diet = if wrapped_in_array { Diet::Leaf(base_diet.is_consuming()) } else { base_diet };
        Ok((base, diet))
    }

    /// Parse `<...>` arguments on a first-order type constructor.
    /// Size arguments retain brackets so their kind remains explicit.
    fn parse_type_arguments(&mut self) -> Result<Vec<Type>> {
        self.expect_binop("<")?;
        let mut args = Vec::new();
        if !self.check_type_argument_close() {
            loop {
                args.push(self.parse_type_argument()?);
                if !self.check(&Token::Comma) {
                    break;
                }
                self.advance();
            }
        }
        self.expect_type_argument_close()?;
        Ok(args)
    }

    fn parse_type_argument(&mut self) -> Result<Type> {
        if self.check(&Token::LeftBracket) || self.check(&Token::LeftBracketSpaced) {
            let start = self.current;
            self.advance();
            let size = match self.peek() {
                Some(Token::RightBracket) => {
                    self.advance();
                    Type::Constructed(TypeName::SizePlaceholder, vec![])
                }
                Some(Token::Identifier(name)) => {
                    let name = name.clone();
                    self.advance();
                    self.expect(Token::RightBracket)?;
                    types::size_var(name)
                }
                Some(Token::IntLiteral(n)) => {
                    let size = usize::try_from(n).map_err(|_| err_parse!("Invalid type size argument"))?;
                    self.advance();
                    self.expect(Token::RightBracket)?;
                    Type::Constructed(TypeName::Size(size), vec![])
                }
                _ => bail_parse_at!(
                    self.current_span(),
                    "expected a size literal, size name, or ']' in type argument"
                ),
            };

            // A bracketed token followed by comma/close is a size argument.
            // If a type follows the bracket, rewind and parse the whole array
            // type as one ordinary argument.
            if self.check(&Token::Comma) || self.check_type_argument_close() || self.peek().is_none() {
                Ok(size)
            } else {
                self.current = start;
                self.parse_type().map(|(ty, _)| ty)
            }
        } else {
            self.parse_type().map(|(ty, _)| ty)
        }
    }

    // `>>` and `>>>` are lexer tokens for expression shifts. Inside nested
    // type applications they are consecutive closes, so consume one `>` and
    // leave the remainder for the enclosing application.
    fn check_type_argument_close(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::BinOp(op)) if !op.is_empty() && op.chars().all(|c| c == '>')
        )
    }

    fn expect_type_argument_close(&mut self) -> Result<()> {
        let Some(Token::BinOp(op)) = self.peek().cloned() else {
            bail_parse_at!(self.current_span(), "expected '>' after type arguments");
        };
        if op.is_empty() || !op.chars().all(|c| c == '>') {
            bail_parse_at!(self.current_span(), "expected '>' after type arguments");
        }
        if op.len() == 1 {
            self.advance();
        } else {
            self.tokens[self.current].token = Token::BinOp(op[1..].to_string());
        }
        Ok(())
    }

    // Helper to check if current token can start a type
    fn can_start_type(&self) -> bool {
        match self.peek() {
            Some(Token::LeftParen) => true, // Tuple type
            Some(Token::LeftBrace) => true, // Record type
            Some(Token::LeftBracket) | Some(Token::LeftBracketSpaced) => true, // Array type
            Some(Token::BinOp(op)) if op == "*" => true, // Unique type
            Some(Token::Identifier(name)) => {
                // Grammar allows qualname which includes any lowercase identifier
                // Uppercase = constructor/sum type
                // Lowercase = base types (i32/f32), vector/matrix types, or user-defined type aliases
                name.chars().next().is_some_and(|c| c.is_uppercase() || c.is_lowercase() || c == '\'')
            }
            _ => false,
        }
    }

    // Helper to check if we're at a type boundary (don't continue type application)
    fn is_at_type_boundary(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::Arrow)
                | Some(Token::RightParen)
                | Some(Token::RightBrace)
                | Some(Token::Comma)
                | Some(Token::Assign)
                | Some(Token::Pipe)
                | Some(Token::Colon)
                | None
        ) || !self.can_start_type()
    }

    fn parse_array_or_base_type(&mut self) -> Result<(Type, Diet)> {
        trace!("parse_array_or_base_type: next token = {:?}", self.peek());
        // Uniqueness prefix `*`: strip it off the type and mark the diet
        // at this position consuming. `*` distributes over the whole inner
        // value, so an aggregate becomes wholly consuming.
        if matches!(self.peek(), Some(Token::BinOp(op)) if op == "*") {
            self.advance(); // consume '*'
            let (inner_type, inner_diet) = self.parse_array_or_base_type()?;
            return Ok((inner_type, inner_diet.into_consuming()));
        }

        // Check for array type [dim]baseType (Futhark style)
        // Accept both LeftBracket and LeftBracketSpaced in type position
        if self.check(&Token::LeftBracket) || self.check(&Token::LeftBracketSpaced) {
            self.advance(); // consume '['

            // An array is one storage unit: its diet is a leaf, consuming
            // only if a `*` was buried inside it.
            let build = |elem_type: Type, elem_diet: Diet, size: Type| {
                (
                    Type::Constructed(
                        TypeName::Array,
                        vec![
                            elem_type,
                            Type::Constructed(TypeName::AddressPlaceholder, vec![]),
                            size,
                            // region placeholder, resolved to a fresh var by resolve_placeholders.
                            Type::Constructed(TypeName::AddressPlaceholder, vec![]),
                        ],
                    ),
                    Diet::Leaf(elem_diet.is_consuming()),
                )
            };

            // Check for empty brackets [] - unsized array with unknown address space
            if self.check(&Token::RightBracket) {
                self.advance();
                let (elem_type, elem_diet) = self.parse_array_or_base_type()?;
                return Ok(build(
                    elem_type,
                    elem_diet,
                    Type::Constructed(TypeName::SizePlaceholder, vec![]),
                ));
            }

            // Parse dimension - could be integer literal or identifier (size variable)
            if let Some(Token::IntLiteral(n)) = self.peek() {
                let size = usize::try_from(n).map_err(|_| err_parse!("Invalid array size"))?;
                self.advance();
                self.expect(Token::RightBracket)?;
                let (elem_type, elem_diet) = self.parse_array_or_base_type()?; // Allow nested arrays
                Ok(build(
                    elem_type,
                    elem_diet,
                    Type::Constructed(TypeName::Size(size), vec![]),
                ))
            } else if let Some(Token::Identifier(name)) = self.peek() {
                // Size variable [n]
                let size_var = name.clone();
                self.advance();
                self.expect(Token::RightBracket)?;
                let (elem_type, elem_diet) = self.parse_array_or_base_type()?;
                Ok(build(elem_type, elem_diet, types::size_var(size_var)))
            } else {
                Err(err_parse!("Expected size literal or variable in array type"))
            }
        } else {
            self.parse_base_type()
        }
    }

    fn parse_base_type(&mut self) -> Result<(Type, Diet)> {
        trace!("parse_base_type: next token = {:?}", self.peek());

        // Check for vector/matrix types first to avoid borrow issues
        if let Some(Token::Identifier(name)) = self.peek() {
            let name_str = name.clone();
            if let Some(ty) = get_vector_types().get(&name_str) {
                self.advance();
                return Ok((ty.clone(), Diet::Leaf(false)));
            }
            if let Some(ty) = get_matrix_types().get(&name_str) {
                self.advance();
                return Ok((ty.clone(), Diet::Leaf(false)));
            }
        }

        match self.peek() {
            Some(Token::Identifier(name)) if name == "i32" => {
                self.advance();
                Ok((types::i32(), Diet::Leaf(false)))
            }
            Some(Token::Identifier(name)) if name == "f32" => {
                self.advance();
                Ok((types::f32(), Diet::Leaf(false)))
            }
            Some(Token::Identifier(name)) if name.chars().next().is_some_and(char::is_lowercase) => {
                let type_name = name.clone();
                self.advance();

                // Check for qualified type name (module.typename)
                if self.check(&Token::Dot) {
                    self.advance(); // consume '.'
                    let inner_name = self.expect_identifier()?;
                    let qualified = format!("{}.{}", type_name, inner_name);
                    return Ok((
                        Type::Constructed(TypeName::Named(qualified), vec![]),
                        Diet::Leaf(false),
                    ));
                }

                // Check if this is a builtin primitive type
                let type_name_variant = match type_name.as_str() {
                    // Floating point types
                    "f16" => TypeName::Float(16),
                    "f32" => TypeName::Float(32),
                    "f64" => TypeName::Float(64),
                    // Signed integer types
                    "i8" => TypeName::Int(8),
                    "i16" => TypeName::Int(16),
                    "i32" => TypeName::Int(32),
                    "i64" => TypeName::Int(64),
                    // Unsigned integer types
                    "u8" => TypeName::UInt(8),
                    "u16" => TypeName::UInt(16),
                    "u32" => TypeName::UInt(32),
                    "u64" => TypeName::UInt(64),
                    // Boolean
                    "bool" => TypeName::Bool,
                    // Opaque GPU resources. Storage images carry a hidden
                    // buffer slot so their descriptor binding survives
                    // capture and monomorphization.
                    "texture2d" => TypeName::Texture2D,
                    "sampler" => TypeName::Sampler,
                    "raster" if self.graphics => TypeName::Raster,
                    "vertex_invocation" if self.graphics => TypeName::VertexInvocation,
                    "vertex" if self.graphics => TypeName::Vertex,
                    "fragment_invocation" if self.graphics => TypeName::FragmentInvocation,
                    "fragment_output" if self.graphics => TypeName::FragmentOutput,
                    "draw" if self.graphics => TypeName::Draw,
                    "render_target" if self.graphics => TypeName::RenderTarget,
                    "storage_image" => {
                        return Ok((
                            Type::Constructed(
                                TypeName::StorageTexture,
                                vec![Type::Constructed(TypeName::AddressPlaceholder, vec![])],
                            ),
                            Diet::Leaf(false),
                        ));
                    }
                    // User-defined type alias or unrecognized type
                    _ => TypeName::Named(type_name),
                };
                Ok((Type::Constructed(type_name_variant, vec![]), Diet::Leaf(false)))
            }
            Some(Token::LeftParen) => {
                // Tuple type (T1, T2, T3), empty tuple (), or parenthesized type (T)
                self.advance(); // consume '('
                let mut tuple_types = Vec::new();
                let mut tuple_diets = Vec::new();
                let mut has_comma = false;

                if !self.check(&Token::RightParen) {
                    loop {
                        let (ty, diet) = self.parse_type()?;
                        tuple_types.push(ty);
                        tuple_diets.push(diet);
                        if !self.check(&Token::Comma) {
                            break;
                        }
                        has_comma = true;
                        self.advance(); // consume ','
                    }
                }

                self.expect(Token::RightParen)?;

                // If exactly one type with no comma, it's just grouping parens, not a tuple
                if tuple_types.len() == 1 && !has_comma {
                    let Some(ty) = tuple_types.pop() else {
                        return Err(err_parse_at!(self.previous_span(), "missing parenthesized type"));
                    };
                    let Some(diet) = tuple_diets.pop() else {
                        return Err(err_parse_at!(
                            self.previous_span(),
                            "missing parenthesized type diet"
                        ));
                    };
                    Ok((ty, diet))
                } else {
                    let diet = Diet::Aggregate {
                        unique: false,
                        components: tuple_diets,
                    };
                    Ok((types::tuple(tuple_types), diet))
                }
            }
            Some(Token::LeftBrace) => {
                // Record type {field1: type1, field2: type2} or empty record {}
                self.parse_record_type()
            }
            Some(Token::Identifier(name)) if name.chars().next().is_some_and(char::is_uppercase) => {
                let name = name.clone();
                self.advance();

                // Check for qualified type (e.g., R.t for functor param module member)
                if self.check(&Token::Dot) {
                    self.advance();
                    let member = self.expect_identifier()?;
                    let qualified = format!("{}.{}", name, member);
                    return Ok((
                        Type::Constructed(TypeName::Named(qualified), vec![]),
                        Diet::Leaf(false),
                    ));
                }

                // Uppercase identifiers in type position are type variables
                // (e.g. `T`, `UV`). Sum types use `#name` constructors and
                // dispatch via the Token::Constructor arm below.
                Ok((
                    Type::Constructed(TypeName::UserVar(name), vec![]),
                    Diet::Leaf(false),
                ))
            }
            Some(Token::Constructor(_)) => self.parse_sum_type(),
            _ => {
                let span = self.current_span();
                Err(err_parse_at!(span, "Expected type"))
            }
        }
    }

    fn parse_record_type(&mut self) -> Result<(Type, Diet)> {
        self.expect(Token::LeftBrace)?;
        let fields = self.parse_delimited_list(&Token::RightBrace, true, |parser| {
            let field_name = match parser.peek() {
                Some(Token::Identifier(name)) => {
                    let name = name.clone();
                    parser.advance();
                    name
                }
                Some(Token::IntLiteral(number)) => {
                    let number = number.to_string();
                    parser.advance();
                    number
                }
                _ => bail_parse_at!(parser.current_span(), "Expected field name or number"),
            };
            parser.expect(Token::Colon)?;
            let (field_type, field_diet) = parser.parse_type()?;
            Ok(((field_name, field_type), field_diet))
        })?;
        self.expect(Token::RightBrace)?;
        let (fields, field_diets) = fields.into_iter().unzip();
        let diet = Diet::Aggregate {
            unique: false,
            components: field_diets,
        };
        Ok((types::record(fields), diet))
    }

    fn parse_sum_type(&mut self) -> Result<(Type, Diet)> {
        let mut variants = Vec::new();

        loop {
            // `#name` constructor token from the lexer.
            let constructor_name = match self.peek() {
                Some(Token::Constructor(name)) => {
                    let n = name.clone();
                    self.advance();
                    n
                }
                _ => bail_parse_at!(self.current_span(), "Expected `#name` constructor"),
            };

            // Optional payload list: `#name(t1, t2, ...)`. A bare `#name`
            // (no parens) is a nullary constructor.
            let arg_types = if self.check(&Token::LeftParen) {
                self.advance(); // consume `(`
                let args = self.parse_delimited_list(&Token::RightParen, false, |parser| {
                    parser.parse_type().map(|(ty, _)| ty)
                })?;
                self.expect(Token::RightParen)?;
                args
            } else {
                Vec::new()
            };

            variants.push((constructor_name, arg_types));

            if self.check(&Token::Pipe) {
                self.advance(); // consume `|`
            } else {
                break;
            }
        }

        // A sum type is a value leaf; `*` on its payloads is not meaningful.
        Ok((types::sum(variants), Diet::Leaf(false)))
    }

    fn parse_expression(&mut self) -> Result<Expression> {
        trace!("parse_expression: next token = {:?}", self.peek());
        self.parse_type_ascription()
    }

    // Parse type ascription and coercion (lowest precedence)
    fn parse_type_ascription(&mut self) -> Result<Expression> {
        let mut expr = self.parse_range_expression()?;

        // Check for type ascription (:) or type coercion (:>)
        match self.peek() {
            Some(Token::Colon) => {
                let start_span = expr.h.span;
                self.advance();
                let (ty, _diet) = self.parse_type()?;
                let end_span = self.previous_span();
                let span = start_span.merge(&end_span);
                expr = self.node_counter.mk_node(ExprKind::TypeAscription(Box::new(expr), ty), span);
            }
            Some(Token::TypeCoercion) => {
                let start_span = expr.h.span;
                self.advance();
                let (ty, _diet) = self.parse_type()?;
                let end_span = self.previous_span();
                let span = start_span.merge(&end_span);
                expr = self.node_counter.mk_node(ExprKind::TypeCoercion(Box::new(expr), ty), span);
            }
            _ => {}
        }

        Ok(expr)
    }

    // Parse range expressions: a..b, a..<b, a..>b, a...b, a..step..end
    fn parse_range_expression(&mut self) -> Result<Expression> {
        let mut start = self.parse_binary_expression()?;

        // Check if we have a range operator
        match self.peek() {
            Some(Token::DotDot) | Some(Token::DotDotLt) | Some(Token::DotDotGt) | Some(Token::Ellipsis) => {
                let start_span = start.h.span;
                self.advance();
                let first_op = self.tokens[self.current - 1].token.clone();

                // Check if there's a step value (for a..step..end)
                let (step, end_op) = if matches!(first_op, Token::DotDot) {
                    // Parse potential step
                    let step_expr = self.parse_binary_expression()?;

                    // Check if there's another range operator
                    match self.peek() {
                        Some(Token::DotDotLt) | Some(Token::DotDotGt) | Some(Token::Ellipsis) => {
                            self.advance();
                            let second_op = self.tokens[self.current - 1].token.clone();
                            (Some(Box::new(step_expr)), second_op)
                        }
                        _ => {
                            // No second operator, step_expr is actually the end
                            let end_span = step_expr.h.span;
                            let span = start_span.merge(&end_span);
                            return Ok(self.node_counter.mk_node(
                                ExprKind::Range(RangeExpr {
                                    start: Box::new(start),
                                    step: None,
                                    end: Box::new(step_expr),
                                    kind: RangeKind::Exclusive,
                                }),
                                span,
                            ));
                        }
                    }
                } else {
                    (None, first_op)
                };

                // Parse the end expression
                let end = self.parse_binary_expression()?;
                let end_span = end.h.span;

                // Determine range kind
                let kind = match end_op {
                    Token::Ellipsis => RangeKind::Inclusive,
                    Token::DotDotLt => RangeKind::ExclusiveLt,
                    Token::DotDotGt => RangeKind::ExclusiveGt,
                    Token::DotDot => RangeKind::Exclusive,
                    _ => unreachable!(),
                };

                let span = start_span.merge(&end_span);
                start = self.node_counter.mk_node(
                    ExprKind::Range(RangeExpr {
                        start: Box::new(start),
                        step,
                        end: Box::new(end),
                        kind,
                    }),
                    span,
                );
            }
            _ => {}
        }

        Ok(start)
    }

    fn parse_binary_expression(&mut self) -> Result<Expression> {
        trace!("parse_binary_expression: next token = {:?}", self.peek());
        self.parse_binary_expression_with_precedence(0)
    }

    /// Parse the RHS of `with [i] = e`, `with .swizzle [op]= e`, or
    /// `with field = e`. Accepts the full binary expression (so
    /// `v with .y = v.y + s` lowers `s` *inside* the with-value), but
    /// rejects a trailing `with` at the same level so chained forms
    /// like `a with .x = e1 with .y = e2` retain their left-
    /// associative interpretation `(a with .x = e1) with .y = e2`.
    fn parse_with_value(&mut self) -> Result<Expression> {
        self.parse_binary_expression_with_precedence_ex(0, false)
    }

    fn get_operator_precedence(op: &str) -> Option<(u32, bool)> {
        // Returns (precedence, is_left_associative)
        // Dominating precedence (higher number) binds tighter than dominated precedence
        // Based on SPECIFICATION.md operator precedence table:
        //   |> (dominated) < || < && < comparisons < bitwise < shifts < +- < */% < ** (dominating)
        match op {
            "|>" => Some((0, true)), // Pipe operator (lowest precedence, left-associative)
            "||" => Some((1, true)), // Logical or
            "&&" => Some((2, true)), // Logical and
            "==" | "!=" | "<" | ">" | "<=" | ">=" => Some((3, true)), // Comparison operators
            "&" | "^" | "|" => Some((4, true)), // Bitwise operators
            "<<" | ">>" | ">>>" => Some((5, true)), // Bitwise shifts
            "+" | "-" => Some((6, true)), // Addition and subtraction
            "*" | "/" | "%" | "//" | "%%" => Some((7, true)), // Multiplication, division, modulo
            "**" => Some((9, true)), // Exponentiation (most dominating binary)
            _ => None,
        }
    }

    fn parse_binary_expression_with_precedence(&mut self, dominated_by: u32) -> Result<Expression> {
        self.parse_binary_expression_with_precedence_ex(dominated_by, true)
    }

    /// Like `parse_binary_expression_with_precedence`, but `allow_with`
    /// gates whether a top-level `with` operator is consumed here.
    /// Setting it to `false` keeps `with` from being swallowed when we're
    /// already inside the RHS of an outer `with`, preserving the left-
    /// associative chained semantics — see `parse_with_value`.
    fn parse_binary_expression_with_precedence_ex(
        &mut self,
        dominated_by: u32,
        allow_with: bool,
    ) -> Result<Expression> {
        trace!(
            "parse_binary_expression_with_precedence({}): next token = {:?}",
            dominated_by,
            self.peek()
        );
        let mut left = self.parse_unary_expression()?;

        loop {
            // Handle 'with' as a special left-associative operator
            // with has precedence 10 (dominates all binary operators).
            // Two LHS forms are accepted:
            //   array form:    a with [i] = e
            //   swizzle form:  v with .yz = e   (or `.yz *= e` for compound)
            if allow_with && self.check(&Token::With) && dominated_by <= 10 {
                let start_span = left.h.span;
                self.advance(); // consume 'with'

                if self.check(&Token::LeftBracket) || self.check(&Token::LeftBracketSpaced) {
                    self.advance();
                    let index = self.parse_expression()?;
                    self.expect(Token::RightBracket)?;
                    self.expect(Token::Assign)?;

                    let value = self.parse_with_value()?;
                    let end_span = self.previous_span();
                    let span = start_span.merge(&end_span);

                    left = self.node_counter.mk_node(
                        ExprKind::ArrayWith {
                            array: Box::new(left),
                            index: Box::new(index),
                            value: Box::new(value),
                        },
                        span,
                    );
                    continue;
                } else if self.check(&Token::Dot) {
                    self.advance(); // consume '.'
                    let (swizzle_str, swizzle_span) = match self.peek() {
                        Some(Token::Identifier(s)) => (s.clone(), self.current_span()),
                        _ => bail_parse_at!(
                            self.current_span(),
                            "Expected swizzle (e.g. `.xy`) after `with .`"
                        ),
                    };
                    self.advance();
                    if !types::is_swizzle_field(&swizzle_str) {
                        bail_parse_at!(
                            swizzle_span,
                            "`.{}` is not a valid swizzle (must be 1-4 chars from `xyzw` or `rgba`)",
                            swizzle_str
                        );
                    }
                    let mut components: Vec<u8> = Vec::with_capacity(swizzle_str.len());
                    for c in swizzle_str.chars() {
                        let Some(idx) = types::swizzle_component_index(c) else {
                            bail_parse_at!(swizzle_span, "invalid swizzle component `{c}`");
                        };
                        if components.contains(&(idx as u8)) {
                            bail_parse_at!(
                                swizzle_span,
                                "swizzle `.{}` repeats component `{}` — \
                                 distinct components are required on the left of `with`",
                                swizzle_str,
                                c
                            );
                        }
                        components.push(idx as u8);
                    }

                    // After the swizzle: either plain `=`, or a
                    // compound form. The lexer keeps `*=` etc. as
                    // two adjacent tokens (`BinOp("*")` then
                    // `Assign`), so peek at the operator first and
                    // only consume it when an `=` follows.
                    let op = match self.peek() {
                        Some(Token::Assign) => {
                            self.advance();
                            None
                        }
                        _ if {
                            // Look ahead for compound `op=` form.
                            matches!(
                                self.peek2(),
                                Some((Token::BinOp(s), Token::Assign))
                                    if matches!(s.as_str(), "*" | "+" | "-" | "/")
                            )
                        } =>
                        {
                            let prefix = match self.peek() {
                                Some(Token::BinOp(s)) => {
                                    op::BinaryOperator::try_from(s.as_str()).map_err(|_| {
                                        err_parse_at!(
                                            self.current_span(),
                                            "Unsupported compound operator '{}'",
                                            s
                                        )
                                    })?
                                }
                                _ => unreachable!("peek2 just matched BinOp"),
                            };
                            self.advance(); // consume binop
                            self.advance(); // consume `=`
                            Some(prefix)
                        }
                        _ => bail_parse_at!(
                            self.current_span(),
                            "Expected `=`, `*=`, `+=`, `-=`, or `/=` after `with .{}`",
                            swizzle_str
                        ),
                    };

                    let value = self.parse_with_value()?;
                    let end_span = self.previous_span();
                    let span = start_span.merge(&end_span);

                    left = self.node_counter.mk_node(
                        ExprKind::VecWith {
                            target: Box::new(left),
                            components,
                            op,
                            value: Box::new(value),
                        },
                        span,
                    );
                    continue;
                } else if matches!(self.peek(), Some(Token::Identifier(_))) {
                    // Record field path: `r with x = e` or `r with a.x = e`.
                    // Spec grammar: `exp "with" fieldid ("." fieldid)* "=" exp`
                    // (SPECIFICATION.md:597). Distinguished from
                    // swizzle-with by the absence of a leading dot.
                    let mut path: Vec<String> = Vec::new();
                    let Some(Token::Identifier(first)) = self.peek().cloned() else {
                        unreachable!("matched Token::Identifier above")
                    };
                    path.push(first);
                    self.advance();
                    while self.check(&Token::Dot) {
                        self.advance();
                        match self.peek().cloned() {
                            Some(Token::Identifier(s)) => {
                                path.push(s);
                                self.advance();
                            }
                            _ => bail_parse_at!(
                                self.current_span(),
                                "Expected field name after `.` in `with` field path"
                            ),
                        }
                    }
                    self.expect(Token::Assign)?;

                    let value = self.parse_with_value()?;
                    let end_span = self.previous_span();
                    let span = start_span.merge(&end_span);

                    left = self.node_counter.mk_node(
                        ExprKind::RecordWith {
                            record: Box::new(left),
                            path,
                            value: Box::new(value),
                        },
                        span,
                    );
                    continue;
                } else {
                    bail_parse_at!(
                        self.current_span(),
                        "Expected `[`, `.swizzle`, or `field` after `with`"
                    );
                }
            }

            // Check if we have a binary operator or pipe operator
            let op_string = match self.peek() {
                Some(Token::BinOp(op)) => op.clone(),
                Some(Token::PipeOp) => "|>".to_string(),
                Some(Token::Pipe) => "|".to_string(),
                _ => break,
            };

            // Get operator precedence
            let (precedence, is_left_assoc) = match Self::get_operator_precedence(&op_string) {
                Some(p) => p,
                None => break,
            };

            // Check if this operator dominates our current context
            if precedence < dominated_by {
                break;
            }

            // Consume the operator
            self.advance();

            // Parse right side: left-associative ops require dominating precedence on right
            let right_dominated_by = if is_left_assoc {
                precedence + 1 // Left-assoc: right side dominated by this op's level + 1
            } else {
                precedence // Right-assoc: right side at same level
            };

            let right = self.parse_binary_expression_with_precedence_ex(right_dominated_by, allow_with)?;

            // Build the appropriate operation with span from left to right
            let span = left.h.span.merge(&right.h.span);
            left = if op_string == "|>" {
                // Desugar pipe by splicing the left operand as the final
                // argument of the right-hand call — a fully-saturated
                // application, never a partial one:
                //   x |> f(a, b)  =>  f(a, b, x)
                //   x |> f        =>  f(x)   (bare callee / lambda fallback)
                match right.kind {
                    ExprKind::Application(callee, mut args) => {
                        args.push(left);
                        self.node_counter.mk_node(ExprKind::Application(callee, args), span)
                    }
                    _ => {
                        self.node_counter.mk_node(ExprKind::Application(Box::new(right), vec![left]), span)
                    }
                }
            } else {
                // Regular binary operation
                let op = op::BinaryOperator::try_from(op_string.as_str()).map_err(|_| {
                    err_parse_at!(
                        self.previous_span(),
                        "Unsupported primitive operator '{}'",
                        op_string
                    )
                })?;
                self.node_counter.mk_node(
                    ExprKind::BinaryOp(BinaryOp { op }, Box::new(left), Box::new(right)),
                    span,
                )
            };
        }

        Ok(left)
    }

    // Function application uses tuple-style syntax: f(x, y, z).
    fn parse_application_expression(&mut self) -> Result<Expression> {
        trace!("parse_application_expression: next token = {:?}", self.peek());
        // Postfix parsing owns parenthesized function calls.
        self.parse_postfix_expression()
    }

    /// Parse comma-separated arguments for function calls: `(x, y, z)`.
    /// A bare `_` is a call-section placeholder owned by this call; underscores
    /// inside a nested call are therefore consumed by that nested call instead.
    fn parse_call_arguments(&mut self) -> Result<Vec<CallArg>> {
        trace!("parse_call_arguments: next token = {:?}", self.peek());
        self.parse_delimited_list(&Token::RightParen, true, |parser| {
            if parser.check(&Token::Underscore) {
                parser.advance();
                Ok(CallArg::Placeholder)
            } else {
                Ok(CallArg::Expr(parser.parse_expression()?))
            }
        })
    }

    fn parse_postfix_expression(&mut self) -> Result<Expression> {
        trace!("parse_postfix_expression: next token = {:?}", self.peek());
        let mut expr = self.parse_primary_expression()?;

        loop {
            match self.peek() {
                Some(Token::LeftBracket) => {
                    // Array indexing or slicing (no space before [): arr[0] or arr[i:j:s]
                    let start_span = expr.h.span;
                    self.advance();
                    expr = self.parse_index_or_slice(expr, start_span)?;
                }
                Some(Token::LeftBracketSpaced) => {
                    // Space before [ means it's not array indexing, it's a new expression
                    // Stop postfix parsing and let the caller handle it
                    break;
                }
                Some(Token::Dot) => {
                    // Field access (e.g., v.x, v.y, v.z, v.w, t.0, t.1)
                    let start_span = expr.h.span;
                    self.advance();
                    let field_name = self.expect_field_name()?;
                    let end_span = self.previous_span();
                    let span = start_span.merge(&end_span);
                    expr =
                        self.node_counter.mk_node(ExprKind::FieldAccess(Box::new(expr), field_name), span);
                }
                Some(Token::LeftParen) => {
                    // Function call: f(x, y, z)
                    let start_span = expr.h.span;
                    self.advance(); // consume '('
                    let args = self.parse_call_arguments()?;
                    self.expect(Token::RightParen)?;
                    let end_span = self.previous_span();
                    let span = start_span.merge(&end_span);
                    expr = self.build_call_or_section(expr, args, span);
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    /// Build an ordinary call, or desugar a call containing placeholders into
    /// a lambda. Each `_` introduces a distinct parameter in left-to-right
    /// order:
    ///
    /// `f(a, _, c, _)` becomes `|_0_, _1_| f(a, _0_, c, _1_)`.
    fn build_call_or_section(&mut self, func: Expression, args: Vec<CallArg>, span: Span) -> Expression {
        if !args.iter().any(|arg| matches!(arg, CallArg::Placeholder)) {
            let args = args
                .into_iter()
                .map(|arg| match arg {
                    CallArg::Expr(expr) => expr,
                    CallArg::Placeholder => unreachable!(),
                })
                .collect();
            return self.node_counter.mk_node(ExprKind::Application(Box::new(func), args), span);
        }

        let mut params = Vec::new();
        let mut next_param = 0;
        let call_args = args
            .into_iter()
            .map(|arg| match arg {
                CallArg::Placeholder => {
                    let name = format!("_{}_", next_param);
                    next_param += 1;
                    params.push(self.node_counter.mk_node(PatternKind::Name(name.clone()), span));
                    self.node_counter.mk_node(
                        ExprKind::Identifier(Identifier {
                            qualifiers: vec![],
                            name,
                        }),
                        span,
                    )
                }
                CallArg::Expr(expr) => expr,
            })
            .collect();

        let body = self.node_counter.mk_node(ExprKind::Application(Box::new(func), call_args), span);
        self.node_counter.mk_node(
            ExprKind::Lambda(LambdaExpr {
                params,
                body: Box::new(body),
            }),
            span,
        )
    }

    /// Parse either array indexing `a[i]` or array slicing `a[start..end]`
    /// Called after consuming the `[` token
    fn parse_index_or_slice(&mut self, array: Expression, start_span: Span) -> Result<Expression> {
        // Check if we have a DotDot immediately ([..end])
        let start_expr = if self.check(&Token::DotDot) {
            None
        } else if self.check(&Token::RightBracket) {
            // Empty brackets a[] - this is an error
            bail_parse_at!(self.current_span(), "Expected index or slice expression");
        } else {
            // Parse the first expression
            Some(Box::new(self.parse_binary_expression()?))
        };

        // Check if this is a slice (DotDot present) or regular index
        if self.check(&Token::DotDot) {
            // This is a slice: a[start..end] or a[..end] etc.
            self.advance(); // consume '..'

            // Parse optional end expression
            let end_expr = if self.check(&Token::RightBracket) {
                None
            } else {
                Some(Box::new(self.parse_binary_expression()?))
            };

            self.expect(Token::RightBracket)?;
            let end_span = self.previous_span();
            let span = start_span.merge(&end_span);

            Ok(self.node_counter.mk_node(
                ExprKind::Slice(SliceExpr {
                    array: Box::new(array),
                    start: start_expr,
                    end: end_expr,
                }),
                span,
            ))
        } else {
            // Regular array indexing
            self.expect(Token::RightBracket)?;
            let end_span = self.previous_span();
            let span = start_span.merge(&end_span);

            // start_expr must be Some here since we didn't see DotDot
            let Some(index) = start_expr else {
                return Err(err_parse_at!(start_span, "array index expression is missing"));
            };
            Ok(self.node_counter.mk_node(ExprKind::ArrayIndex(Box::new(array), index), span))
        }
    }

    fn parse_unary_expression(&mut self) -> Result<Expression> {
        trace!("parse_unary_expression: next token = {:?}", self.peek());
        // Check for unary operators: - and !
        // Postfix operators ([], .) bind tighter than unary, so we parse postfix for the operand
        match self.peek() {
            Some(Token::BinOp(op)) if op == "-" => {
                let start_span = self.current_span();
                self.advance();
                let operand = self.parse_unary_expression()?; // Right-associative for chaining: --x
                let span = start_span.merge(&operand.h.span);
                // Negation of a numeric literal is itself a literal (the lexer's
                // `floatnumber` / `intnumber` carry no sign). Fold it at parse
                // time so a negative constant — e.g. an array element `-0.5` —
                // stays a compile-time constant rather than a runtime negation.
                let negated = match &operand.kind {
                    ExprKind::IntLiteral(s) => Some(ExprKind::IntLiteral(s.negated())),
                    ExprKind::FloatLiteral(f) => Some(ExprKind::FloatLiteral(-*f)),
                    _ => None,
                };
                Ok(match negated {
                    Some(kind) => self.node_counter.mk_node(kind, span),
                    None => self.node_counter.mk_node(
                        ExprKind::UnaryOp(
                            UnaryOp {
                                op: op::UnaryOperator::Negate,
                            },
                            Box::new(operand),
                        ),
                        span,
                    ),
                })
            }
            Some(Token::Bang) => {
                let start_span = self.current_span();
                self.advance();
                let operand = self.parse_unary_expression()?; // Right-associative for chaining: !!x
                let span = start_span.merge(&operand.h.span);
                Ok(self.node_counter.mk_node(
                    ExprKind::UnaryOp(
                        UnaryOp {
                            op: op::UnaryOperator::LogicalNot,
                        },
                        Box::new(operand),
                    ),
                    span,
                ))
            }
            _ => self.parse_application_expression(),
        }
    }

    fn parse_primary_expression(&mut self) -> Result<Expression> {
        trace!("parse_primary_expression: next token = {:?}", self.peek());
        match self.peek() {
            Some(Token::TypeHole) => {
                let span = self.current_span();
                self.advance();
                Ok(self.node_counter.mk_node(ExprKind::TypeHole(TypeHole), span))
            }
            Some(Token::IntLiteral(n)) => {
                let n = n.clone();
                let span = self.current_span();
                self.advance();
                Ok(self.node_counter.mk_node(ExprKind::IntLiteral(n), span))
            }
            Some(Token::SuffixedLiteral(inner, suffix)) => {
                let inner = inner.clone();
                let suffix = suffix.clone();
                let span = self.current_span();
                self.advance();
                // Convert suffixed literal to TypeAscription(literal, type)
                let inner_expr = match *inner {
                    Token::IntLiteral(n) => ExprKind::IntLiteral(n),
                    Token::FloatLiteral(f) => ExprKind::FloatLiteral(f),
                    _ => bail_parse_at!(self.current_span(), "Invalid suffixed literal"),
                };
                let inner_node = self.node_counter.mk_node(inner_expr, span);
                // Convert suffix string to Type
                let ty = suffix_to_type(&suffix);
                Ok(self.node_counter.mk_node(ExprKind::TypeAscription(Box::new(inner_node), ty), span))
            }
            Some(Token::FloatLiteral(f)) => {
                let f = *f;
                let span = self.current_span();
                self.advance();
                Ok(self.node_counter.mk_node(ExprKind::FloatLiteral(f), span))
            }
            Some(Token::True) => {
                let span = self.current_span();
                self.advance();
                Ok(self.node_counter.mk_node(ExprKind::BoolLiteral(true), span))
            }
            Some(Token::False) => {
                let span = self.current_span();
                self.advance();
                Ok(self.node_counter.mk_node(ExprKind::BoolLiteral(false), span))
            }
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                let span = self.current_span();
                self.advance();
                Ok(self.node_counter.mk_node(
                    ExprKind::Identifier(Identifier {
                        qualifiers: vec![],
                        name,
                    }),
                    span,
                ))
            }
            Some(Token::Constructor(name)) => {
                let name = name.clone();
                let span = self.current_span();
                self.advance();
                // Optional payload: `#name(arg1, arg2, ...)`. A bare
                // `#name` is a nullary constructor.
                let args = if self.check(&Token::LeftParen) {
                    self.advance(); // consume `(`
                    let args = self.parse_delimited_list(&Token::RightParen, false, |parser| {
                        parser.parse_expression()
                    })?;
                    self.expect(Token::RightParen)?;
                    args
                } else {
                    Vec::new()
                };
                Ok(self.node_counter.mk_node(ExprKind::Constructor(name, args), span))
            }
            Some(Token::LeftBracket) | Some(Token::LeftBracketSpaced) => self.parse_array_literal(),
            Some(Token::AtBracket) => self.parse_vec_mat_literal(),
            Some(Token::LeftParen) => {
                let start_span = self.current_span();
                self.advance(); // consume '('

                // Check for unit ()
                if self.check(&Token::RightParen) {
                    self.advance();
                    let end_span = self.previous_span();
                    let span = start_span.merge(&end_span);
                    return Ok(self.node_counter.mk_node(ExprKind::Unit, span));
                }

                // Check for operator section: (+), (-), (*), etc.
                // Use peek2 to check if we have (BinOp, RightParen) pattern
                // Desugar to lambda: (+) => \x y -> x + y
                if let Some((Token::BinOp(op), Token::RightParen)) = self.peek2() {
                    let op = op::BinaryOperator::try_from(op.as_str()).map_err(|_| {
                        err_parse_at!(self.current_span(), "Unsupported primitive operator '{}'", op)
                    })?;
                    self.advance(); // consume operator
                    self.advance(); // consume )
                    let end_span = self.previous_span();
                    let span = start_span.merge(&end_span);

                    // Create patterns for parameters: x, y
                    let x_pattern = self.node_counter.mk_node(PatternKind::Name("x".to_string()), span);
                    let y_pattern = self.node_counter.mk_node(PatternKind::Name("y".to_string()), span);

                    // Create identifier expressions for body: x, y
                    let x_expr = self.node_counter.mk_node(
                        ExprKind::Identifier(Identifier {
                            qualifiers: vec![],
                            name: "x".to_string(),
                        }),
                        span,
                    );
                    let y_expr = self.node_counter.mk_node(
                        ExprKind::Identifier(Identifier {
                            qualifiers: vec![],
                            name: "y".to_string(),
                        }),
                        span,
                    );

                    // Create body: x op y
                    let body = self.node_counter.mk_node(
                        ExprKind::BinaryOp(BinaryOp { op }, Box::new(x_expr), Box::new(y_expr)),
                        span,
                    );

                    // Create lambda: |x, y| x op y
                    let lambda = LambdaExpr {
                        params: vec![x_pattern, y_pattern],
                        body: Box::new(body),
                    };

                    return Ok(self.node_counter.mk_node(ExprKind::Lambda(lambda), span));
                }

                // Parse first expression
                let first_expr = self.parse_expression()?;

                // Check if it's a tuple or just a parenthesized expression
                if self.check(&Token::Comma) {
                    // It's a tuple
                    let mut elements = vec![first_expr];
                    self.advance(); // consume first comma
                    elements.extend(self.parse_delimited_list(&Token::RightParen, true, |parser| {
                        parser.parse_expression()
                    })?);
                    self.expect(Token::RightParen)?;
                    let end_span = self.previous_span();
                    let span = start_span.merge(&end_span);
                    Ok(self.node_counter.mk_node(ExprKind::Tuple(elements), span))
                } else {
                    // Just a parenthesized expression
                    self.expect(Token::RightParen)?;
                    Ok(first_expr)
                }
            }
            Some(Token::LeftBrace) => self.parse_record_literal(),
            Some(Token::Pipe) => self.parse_lambda(),
            Some(Token::BinOp(op)) if op == "||" => self.parse_lambda(), // Empty lambda: || body
            Some(Token::Let) => self.parse_let_in(),
            Some(Token::If) => self.parse_if_then_else(),
            Some(Token::Loop) => self.parse_loop(),
            Some(Token::Match) => self.parse_match(),
            _ => {
                let span = self.current_span();
                Err(err_parse_at!(span, "Expected expression, got {:?}", self.peek()))
            }
        }
    }

    fn parse_array_literal(&mut self) -> Result<Expression> {
        trace!("parse_array_literal: next token = {:?}", self.peek());
        // Accept either LeftBracket or LeftBracketSpaced
        let start_span = self.current_span();
        match self.peek() {
            Some(Token::LeftBracket) | Some(Token::LeftBracketSpaced) => {
                self.advance();
            }
            _ => bail_parse_at!(self.current_span(), "Expected '['"),
        }

        let elements =
            self.parse_delimited_list(&Token::RightBracket, true, |parser| parser.parse_expression())?;
        self.expect(Token::RightBracket)?;
        let end_span = self.previous_span();
        let span = start_span.merge(&end_span);
        Ok(self.node_counter.mk_node(ExprKind::ArrayLiteral(elements), span))
    }

    /// Parse @[...] vector/matrix literal
    /// - @[1.0, 2.0, 3.0] -> vec3 (elements are scalars)
    /// - @[[1,2,3], [4,5,6]] -> mat2x3 (elements are row arrays)
    fn parse_vec_mat_literal(&mut self) -> Result<Expression> {
        trace!("parse_vec_mat_literal: next token = {:?}", self.peek());
        let start_span = self.current_span();
        self.expect(Token::AtBracket)?;

        let elements =
            self.parse_delimited_list(&Token::RightBracket, true, |parser| parser.parse_expression())?;
        self.expect(Token::RightBracket)?;
        let end_span = self.previous_span();
        let span = start_span.merge(&end_span);
        Ok(self.node_counter.mk_node(ExprKind::VecMatLiteral(elements), span))
    }

    fn parse_record_literal(&mut self) -> Result<Expression> {
        trace!("parse_record_literal: next token = {:?}", self.peek());
        let start_span = self.current_span();
        self.expect(Token::LeftBrace)?;
        let fields = self.parse_delimited_list(&Token::RightBrace, true, |parser| {
            let field_name = if let Some(Token::Identifier(name)) = parser.peek() {
                let name = name.clone();
                parser.advance();
                name
            } else {
                bail_parse_at!(
                    parser.current_span(),
                    "Expected field name in record literal, got {:?} at {}",
                    parser.peek(),
                    parser.current_span()
                );
            };
            parser.expect(Token::Assign)?;
            Ok((field_name, parser.parse_expression()?))
        })?;
        self.expect(Token::RightBrace)?;
        let end_span = self.previous_span();
        let span = start_span.merge(&end_span);
        Ok(self.node_counter.mk_node(ExprKind::RecordLiteral(fields), span))
    }

    /// Parse lambda: |x, y| body or |x: i32| -> i32 body or || body (empty params)
    fn parse_lambda(&mut self) -> Result<Expression> {
        trace!("parse_lambda: next token = {:?}", self.peek());
        let start_span = self.current_span();

        // Handle empty lambda || (tokenized as BinOp("||")) vs regular |params|
        let params = if self.check(&Token::BinOp(String::from("||"))) {
            self.advance(); // consume ||
            vec![] // empty params
        } else {
            self.expect(Token::Pipe)?; // Opening |
            let params = self.parse_delimited_list(&Token::Pipe, false, |parser| parser.parse_pattern())?;
            self.expect(Token::Pipe)?; // Closing |
            params
        };

        // Parse body expression
        let body = Box::new(self.parse_expression()?);
        let span = start_span.merge(&body.h.span);

        Ok(self.node_counter.mk_node(ExprKind::Lambda(LambdaExpr { params, body }), span))
    }

    fn parse_let_in(&mut self) -> Result<Expression> {
        trace!("parse_let_in: next token = {:?}", self.peek());
        use crate::ast::LetInExpr;

        let start_span = self.current_span();
        self.expect(Token::Let)?;
        let pattern = self.parse_pattern()?;

        // `let name(params) = body in rest` — a let-bound function.
        // Desugar to `let name = |params| body in rest` at parse time
        // so downstream sees the existing `LetIn { value: Lambda }`
        // shape, no new AST variant required. Recursion is not allowed
        // in Wyn; the desugar honours that by construction — the
        // lambda is constructed before `name` enters scope, so `name`
        // is structurally absent from the lambda body and present
        // only in `rest`. Only fires for simple-name patterns; a
        // tuple destructuring `let (a, b) = …` already consumed its
        // `(...)` inside `parse_pattern` above.
        if pattern.simple_name().is_some() && self.check(&Token::LeftParen) {
            let params_span = self.current_span();
            let (params, _param_diets) = self.parse_comma_separated_params()?;
            self.expect(Token::Assign)?;
            let body_expr = self.parse_expression()?;
            let lam_span = params_span.merge(&body_expr.h.span);
            let lambda = self.node_counter.mk_node(
                ExprKind::Lambda(LambdaExpr {
                    params,
                    body: Box::new(body_expr),
                }),
                lam_span,
            );

            if !self.check(&Token::Let) {
                self.expect(Token::In)?;
            }
            let body = Box::new(self.parse_expression()?);
            let span = start_span.merge(&body.h.span);

            return Ok(self.node_counter.mk_node(
                ExprKind::LetIn(LetInExpr {
                    pattern,
                    ty: None,
                    value: Box::new(lambda),
                    body,
                }),
                span,
            ));
        }

        // Optional type annotation
        let ty = if self.check(&Token::Colon) {
            self.advance(); // consume ':'
            let (ty, _diet) = self.parse_type()?;
            Some(ty)
        } else {
            None
        };

        self.expect(Token::Assign)?;
        let value = Box::new(self.parse_expression()?);
        // `in` may be omitted when the body is itself a `let` expression,
        // so chained bindings can be written without a sea of `in`
        // keywords: `let x = 1 let y = 2 in x + y`.
        if !self.check(&Token::Let) {
            self.expect(Token::In)?;
        }
        let body = Box::new(self.parse_expression()?);
        let span = start_span.merge(&body.h.span);

        Ok(self.node_counter.mk_node(
            ExprKind::LetIn(LetInExpr {
                pattern,
                ty,
                value,
                body,
            }),
            span,
        ))
    }

    fn parse_if_then_else(&mut self) -> Result<Expression> {
        trace!("parse_if_then_else: next token = {:?}", self.peek());
        use crate::ast::IfExpr;

        let start_span = self.current_span();
        self.expect(Token::If)?;
        let condition = Box::new(self.parse_expression()?);
        self.expect(Token::Then)?;
        let then_branch = Box::new(self.parse_expression()?);
        self.expect(Token::Else)?;
        let else_branch = Box::new(self.parse_expression()?);
        let span = start_span.merge(&else_branch.h.span);

        Ok(self.node_counter.mk_node(
            ExprKind::If(IfExpr {
                condition,
                then_branch,
                else_branch,
            }),
            span,
        ))
    }

    fn parse_loop(&mut self) -> Result<Expression> {
        trace!("parse_loop: next token = {:?}", self.peek());
        use crate::ast::{LoopExpr, LoopForm};

        let start_span = self.current_span();
        self.expect(Token::Loop)?;
        let pattern = self.parse_pattern()?;

        // Check for optional initialization: = exp
        let init = if self.check(&Token::Assign) {
            self.advance();
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        // Parse loop form
        let form = if self.check(&Token::For) {
            self.advance();
            // Check if it's "for name < exp" or "for pat in exp"
            let saved_pos = self.current;

            // Try to parse as pattern first
            if let Ok(pat) = self.parse_pattern() {
                if self.check(&Token::In) {
                    // It's "for pat in exp"
                    self.advance();
                    let iter_expr = Box::new(self.parse_expression()?);
                    LoopForm::ForIn(pat, iter_expr)
                } else {
                    // Backtrack and try as "for name < exp"
                    self.current = saved_pos;
                    let name = self.expect_identifier()?;
                    let name_span = self.previous_span();
                    self.expect(Token::BinOp("<".to_string()))?;
                    let bound = Box::new(self.parse_expression()?);
                    let pattern = self.node_counter.mk_node(PatternKind::Name(name), name_span);
                    LoopForm::For(pattern, bound)
                }
            } else {
                bail_parse_at!(self.current_span(), "Expected pattern in for loop");
            }
        } else if self.check(&Token::While) {
            self.advance();
            let condition = Box::new(self.parse_expression()?);
            LoopForm::While(condition)
        } else {
            bail_parse_at!(self.current_span(), "Expected 'for' or 'while' in loop");
        };

        self.expect(Token::Do)?;
        let body = Box::new(self.parse_expression()?);
        let span = start_span.merge(&body.h.span);

        Ok(self.node_counter.mk_node(
            ExprKind::Loop(LoopExpr {
                pattern,
                init,
                form,
                body,
            }),
            span,
        ))
    }

    fn parse_match(&mut self) -> Result<Expression> {
        trace!("parse_match: next token = {:?}", self.peek());
        use crate::ast::{MatchCase, MatchExpr};

        let start_span = self.current_span();
        self.expect(Token::Match)?;
        let scrutinee = Box::new(self.parse_expression()?);

        // Parse one or more case branches
        let mut cases = Vec::new();
        let mut last_span = scrutinee.h.span;
        while self.check(&Token::Case) {
            self.advance();
            let pattern = self.parse_pattern()?;
            self.expect(Token::Arrow)?;
            let body = Box::new(self.parse_expression()?);
            last_span = body.h.span;
            cases.push(MatchCase { pattern, body });
        }

        if cases.is_empty() {
            bail_parse_at!(
                self.current_span(),
                "Match expression must have at least one case"
            );
        }

        let span = start_span.merge(&last_span);
        Ok(self.node_counter.mk_node(ExprKind::Match(MatchExpr { scrutinee, cases }), span))
    }

    // Helper methods
    /// Parse comma-separated elements up to (but not including) `closing`.
    ///
    /// The caller consumes both delimiters. Keeping the closing token in place
    /// lets callers include it in their source span.
    fn parse_delimited_list<T>(
        &mut self,
        closing: &Token,
        allow_trailing_comma: bool,
        mut parse_element: impl FnMut(&mut Self) -> Result<T>,
    ) -> Result<Vec<T>> {
        let mut elements = Vec::new();
        if self.check(closing) {
            return Ok(elements);
        }
        loop {
            elements.push(parse_element(self)?);
            if !self.check(&Token::Comma) {
                break;
            }
            self.advance();
            if allow_trailing_comma && self.check(closing) {
                break;
            }
        }
        Ok(elements)
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.current).map(|lt| &lt.token)
    }

    fn peek2(&self) -> Option<(&Token, &Token)> {
        let first = self.tokens.get(self.current)?;
        let second = self.tokens.get(self.current + 1)?;
        Some((&first.token, &second.token))
    }

    fn advance(&mut self) -> Option<&Token> {
        if !self.is_at_end() {
            self.current += 1;
            self.tokens.get(self.current - 1).map(|lt| &lt.token)
        } else {
            None
        }
    }

    fn check(&self, token: &Token) -> bool {
        if let Some(t) = self.peek() {
            std::mem::discriminant(t) == std::mem::discriminant(token)
        } else {
            false
        }
    }

    fn check_binop(&self, op: &str) -> bool {
        matches!(self.peek(), Some(Token::BinOp(s)) if s == op)
    }

    fn expect_binop(&mut self, op: &str) -> Result<()> {
        if self.check_binop(op) {
            self.advance();
            Ok(())
        } else {
            let span = self.current_span();
            Err(err_parse_at!(span, "Expected '{}', got {:?}", op, self.peek()))
        }
    }

    fn expect(&mut self, token: Token) -> Result<()> {
        if self.check(&token) {
            self.advance();
            Ok(())
        } else {
            let span = self.current_span();
            Err(err_parse_at!(span, "Expected {:?}, got {:?}", token, self.peek()))
        }
    }

    /// Parse an operator section: (op) where op is a sequence of operator characters.
    /// Valid operator characters are: +-*/%=!><&^|
    /// Examples: (+), (|), (+^), (**), (>>)
    fn parse_operator_section(&mut self) -> Result<String> {
        self.expect(Token::LeftParen)?;

        let mut operator = String::new();

        // Accumulate all operator characters until we hit RightParen
        loop {
            match self.peek() {
                Some(Token::RightParen) => {
                    self.advance();
                    break;
                }
                Some(Token::BinOp(op)) => {
                    operator.push_str(op);
                    self.advance();
                }
                Some(Token::Pipe) => {
                    operator.push('|');
                    self.advance();
                }
                Some(Token::Bang) => {
                    operator.push('!');
                    self.advance();
                }
                Some(Token::Assign) => {
                    operator.push('=');
                    self.advance();
                }
                _ => {
                    bail_parse_at!(
                        self.current_span(),
                        "Expected operator or ) in operator section at {}",
                        self.current_span()
                    );
                }
            }
        }

        if operator.is_empty() {
            bail_parse_at!(self.current_span(), "Operator section cannot be empty");
        }

        // Validate that all characters are valid operator characters
        const VALID_OP_CHARS: &str = "+-*/%=!><&^|";
        for ch in operator.chars() {
            if !VALID_OP_CHARS.contains(ch) {
                bail_parse_at!(
                    self.current_span(),
                    "Invalid operator character '{}' in operator section",
                    ch
                );
            }
        }

        Ok(operator)
    }

    fn expect_identifier(&mut self) -> Result<String> {
        let span = self.current_span();
        match self.advance() {
            Some(Token::Identifier(name)) => Ok(name.clone()),
            _ => Err(err_parse_at!(span, "Expected identifier")),
        }
    }

    /// Like `expect_identifier`, but also accepts integer literals for tuple
    /// field access (`.0`, `.1`, …) and a parenthesized operator naming a
    /// module's operator member (`m.(+)`, `m.(<<)`).
    fn expect_field_name(&mut self) -> Result<String> {
        let span = self.current_span();
        // `m.(+)` — the member is named by an operator; reuse the operator-
        // section parser, which consumes the `( … )` and returns the operator.
        // Wrap it in parens to match how an operator `def` is named (`(+)`).
        if matches!(self.peek(), Some(Token::LeftParen)) {
            let op = self.parse_operator_section()?;
            return Ok(format!("({})", op));
        }
        match self.advance() {
            Some(Token::Identifier(name)) => Ok(name.clone()),
            Some(Token::IntLiteral(n)) => Ok(n.0.clone()),
            _ => Err(err_parse_at!(
                span,
                "Expected field name, tuple index, or `(operator)`"
            )),
        }
    }

    fn expect_integer(&mut self) -> Result<u32> {
        let span = self.current_span();
        match self.advance() {
            Some(Token::IntLiteral(n)) => {
                u32::try_from(n).map_err(|_| err_parse_at!(span, "Invalid integer"))
            }
            _ => Err(err_parse_at!(span, "Expected integer")),
        }
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len()
    }
}
