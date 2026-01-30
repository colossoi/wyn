use crate::ast::*;
use crate::error::Result;
use crate::lexer::{LocatedToken, Token};
use crate::types;
use crate::{bail_parse, bail_parse_at, err_parse, err_parse_at};
use log::trace;
use std::sync::OnceLock;

mod module;
mod pattern;
#[cfg(test)]
mod tests;

// Lazily initialized type constructor maps
static VECTOR_TYPES: OnceLock<std::collections::HashMap<String, Type>> = OnceLock::new();
static MATRIX_TYPES: OnceLock<std::collections::HashMap<String, Type>> = OnceLock::new();

fn get_vector_types() -> &'static std::collections::HashMap<String, Type> {
    VECTOR_TYPES.get_or_init(types::vector_type_constructors)
}

fn get_matrix_types() -> &'static std::collections::HashMap<String, Type> {
    MATRIX_TYPES.get_or_init(types::matrix_type_constructors)
}

/// Argument in a curry expression - either a placeholder (_) or a real expression
enum CurryArg {
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

pub struct Parser<'a> {
    tokens: Vec<LocatedToken>,
    current: usize,
    node_counter: &'a mut NodeCounter,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: Vec<LocatedToken>, node_counter: &'a mut NodeCounter) -> Self {
        Parser {
            tokens,
            current: 0,
            node_counter,
        }
    }

    /// Get the span of the current token
    fn current_span(&self) -> Span {
        self.tokens.get(self.current).map(|t| t.span).unwrap_or(Span::new(0, 0, 0, 0))
    }

    /// Get the span of the previous token
    fn previous_span(&self) -> Span {
        if self.current > 0 {
            self.tokens.get(self.current - 1).map(|t| t.span).unwrap_or(Span::new(0, 0, 0, 0))
        } else {
            Span::new(0, 0, 0, 0)
        }
    }

    pub fn parse(&mut self) -> Result<Program> {
        let mut declarations = Vec::new();

        while !self.is_at_end() {
            declarations.push(self.parse_declaration()?);
        }

        Ok(Program { declarations })
    }

    fn parse_declaration(&mut self) -> Result<Declaration> {
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
                Ok(Declaration::Sig(decl))
            }
            Some(Token::Type) => {
                let type_bind = self.parse_type_bind()?;
                Ok(Declaration::TypeBind(type_bind))
            }
            Some(Token::Module) => {
                // Check if it's "module type" or just "module"
                let saved_pos = self.current;
                self.advance();
                if self.check(&Token::Type) {
                    // module type declaration
                    self.current = saved_pos;
                    let mod_type_bind = self.parse_module_type_bind()?;
                    Ok(Declaration::ModuleTypeBind(mod_type_bind))
                } else {
                    // module declaration
                    self.current = saved_pos;
                    Ok(Declaration::Module(self.parse_module_decl()?))
                }
            }
            Some(Token::Functor) => Ok(Declaration::Module(self.parse_functor_decl()?)),
            Some(Token::Open) => {
                self.advance();
                let mod_exp = self.parse_module_expression()?;
                Ok(Declaration::Open(mod_exp))
            }
            Some(Token::Import) => {
                self.advance();
                let path = self.expect_string_literal()?;
                Ok(Declaration::Import(path))
            }
            Some(Token::Extern) => self.parse_extern_decl(attributes),
            _ => Err(err_parse_at!(
                self.current_span(),
                "Expected declaration, got {:?}",
                self.peek()
            )),
        }
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

    fn parse_decl(&mut self, keyword: &'static str, attributes: Vec<Attribute>) -> Result<Declaration> {
        trace!("parse_decl({}): next token = {:?}", keyword, self.peek());

        // Check for special attributes that require specific declaration types
        let has_entry_attr = attributes
            .iter()
            .any(|attr| matches!(attr, Attribute::Vertex | Attribute::Fragment | Attribute::Compute));
        let uniform_attr = attributes.iter().find_map(|attr| {
            if let Attribute::Uniform { set, binding } = attr { Some((*set, *binding)) } else { None }
        });
        let storage_attr = attributes.iter().find_map(|attr| {
            if let Attribute::Storage {
                set,
                binding,
                layout,
                access,
            } = attr
            {
                Some((*set, *binding, *layout, *access))
            } else {
                None
            }
        });

        // Entry attributes require the 'entry' keyword, not 'def' or 'let'
        if has_entry_attr {
            bail_parse!(
                "Entry point attributes (#[vertex], #[fragment], #[compute]) require 'entry' keyword, not '{}'",
                keyword
            );
        }

        if let Some((set, binding)) = uniform_attr {
            // Uniform declaration - delegate to helper
            if keyword != "def" {
                bail_parse!("Uniform declarations must use 'def', not 'let'");
            }
            self.parse_uniform_decl(set, binding)
        } else if let Some((set, binding, layout, access)) = storage_attr {
            // Storage buffer declaration - delegate to helper
            if keyword != "def" {
                bail_parse!("Storage declarations must use 'def', not 'let'");
            }
            self.parse_storage_decl(set, binding, layout, access)
        } else {
            // Regular declaration (let or def)
            match keyword {
                "let" => self.expect(Token::Let)?,
                "def" => self.expect(Token::Def)?,
                _ => bail_parse!("Invalid keyword: {}", keyword),
            }

            // Parse name - either an identifier or an operator in parentheses like (+) or (+^)
            let name = if self.peek() == Some(&Token::LeftParen) {
                let op = self.parse_operator_section()?;
                format!("({})", op)
            } else {
                self.expect_identifier()?
            };

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
            let (params, ty) = if keyword == "def" {
                if self.check(&Token::LeftParen) {
                    // Function: def foo(params) R = ...
                    let params = self.parse_comma_separated_params()?;

                    // Reject zero-argument functions - use constant syntax instead
                    if params.is_empty() {
                        bail_parse!(
                            "Zero-argument functions are not allowed. Use constant syntax instead: `def {} = ...`",
                            name
                        );
                    }

                    // Return type directly after params (no arrow): def foo(x: T) R = ...
                    let ty = if !self.check(&Token::Assign) {
                        Some(self.parse_return_type_simple()?)
                    } else {
                        None
                    };
                    (params, ty)
                } else if self.check(&Token::Colon) {
                    // Constant binding with type: def foo: T = ...
                    self.advance();
                    let ty = Some(self.parse_type()?);
                    (vec![], ty)
                } else if self.check(&Token::Assign) {
                    // Constant binding without type: def foo = ...
                    (vec![], None)
                } else {
                    bail_parse!("Expected '(' or ':' after def name");
                }
            } else {
                // let declarations don't have params, just optional type annotation
                let ty = if self.check(&Token::Colon) {
                    self.advance();
                    Some(self.parse_type()?)
                } else {
                    None
                };
                (vec![], ty)
            };

            self.expect(Token::Assign)?;
            let body = self.parse_expression()?;

            Ok(Declaration::Decl(Decl {
                keyword,
                attributes,
                name,
                size_params,
                type_params,
                params,
                ty,
                body,
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

        self.expect(Token::Colon)?;
        let ty = self.parse_type()?;

        Ok(SigDecl {
            attributes: vec![],
            name,
            size_params,
            type_params,
            ty,
        })
    }

    /// Parse an extern declaration for linked SPIR-V functions.
    /// Syntax: `#[linked("linkage_name")] extern name(param: Type, ...) ReturnType`
    fn parse_extern_decl(&mut self, attributes: Vec<Attribute>) -> Result<Declaration> {
        trace!("parse_extern_decl: next token = {:?}", self.peek());
        let start_span = self.current_span();

        // Find the linked attribute
        let linkage_name = attributes
            .iter()
            .find_map(
                |attr| {
                    if let Attribute::Linked(name) = attr { Some(name.clone()) } else { None }
                },
            )
            .ok_or_else(|| err_parse!("extern declaration requires #[linked(\"name\")] attribute"))?;

        self.expect(Token::Extern)?;
        let name = self.expect_identifier()?;

        // Parse optional type parameters: <[n], [m], T>
        let (size_params, type_params) =
            if self.check_binop("<") { self.parse_generic_params()? } else { (vec![], vec![]) };

        // Parse parameters: (param: Type, ...)
        let params = self.parse_extern_params()?;

        // Parse return type (required for extern functions)
        let ret_type = self.parse_return_type_simple()?;

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
            linkage_name,
            size_params,
            type_params,
            ty,
            span: start_span.merge(&end_span),
        }))
    }

    /// Parse extern function parameters: (name: Type, ...)
    /// Returns (name, type) pairs.
    fn parse_extern_params(&mut self) -> Result<Vec<(String, Type)>> {
        self.expect(Token::LeftParen)?;
        let mut params = Vec::new();

        if !self.check(&Token::RightParen) {
            loop {
                let param_name = self.expect_identifier()?;
                self.expect(Token::Colon)?;
                let ty = self.parse_type()?;
                params.push((param_name, ty));

                if !self.check(&Token::Comma) {
                    break;
                }
                self.advance(); // consume comma
            }
        }

        self.expect(Token::RightParen)?;
        Ok(params)
    }

    /// Parse an entry point declaration.
    /// Entry points have restrictive syntax: only `id: type` parameters, not general patterns.
    /// Syntax: `#[vertex|fragment|compute(x,y,z)] entry name(id: type, ...) return_type = body`
    fn parse_entry_decl(&mut self, attributes: Vec<Attribute>) -> Result<Declaration> {
        trace!("parse_entry_decl: next token = {:?}", self.peek());

        // Find the entry type attribute
        let entry_type = attributes
            .iter()
            .find(|attr| matches!(attr, Attribute::Vertex | Attribute::Fragment | Attribute::Compute))
            .ok_or_else(|| {
                err_parse!(
                    "Entry declarations require #[vertex], #[fragment], or #[compute(...)] attribute"
                )
            })?
            .clone();

        self.expect(Token::Entry)?;
        let name = self.expect_identifier()?;

        // Parse optional type parameters: <[n], [m], T>
        let (size_params, type_params) =
            if self.check_binop("<") { self.parse_generic_params()? } else { (vec![], vec![]) };

        // Parse restrictive parameters: (id: type, id: type, ...)
        // Only typed identifiers allowed, not general patterns
        let params = self.parse_entry_params()?;

        // Compute entry params cannot have explicit bindings
        if matches!(entry_type, Attribute::Compute) {
            for param in &params {
                if let PatternKind::Typed(inner, _) = &param.kind {
                    if let PatternKind::Attributed(attrs, _) = &inner.kind {
                        if attrs.iter().any(|a| matches!(a, Attribute::Storage { .. })) {
                            bail_parse!("Compute entry parameters cannot have explicit bindings");
                        }
                    }
                }
            }
        }

        // Parse return type (which may have optional attributes) - no arrow required
        let (return_types, return_attributes) =
            if self.check(&Token::AttributeStart) || self.check(&Token::LeftParen) {
                // Attributed return type(s)
                self.parse_return_type()?
            } else if !self.check(&Token::Assign) {
                // Simple unattributed return type
                let ty = self.parse_type()?;
                (vec![ty], vec![None])
            } else {
                bail_parse!("Entry point declarations must have an explicit return type");
            };

        // Combine into EntryOutput structs
        let outputs: Vec<EntryOutput> = return_types
            .into_iter()
            .zip(return_attributes)
            .map(|(ty, attribute)| EntryOutput { ty, attribute })
            .collect();

        self.expect(Token::Assign)?;
        let body = self.parse_expression()?;

        Ok(Declaration::Entry(EntryDecl {
            entry_type,
            name,
            size_params,
            type_params,
            params,
            outputs,
            body,
        }))
    }

    /// Parse entry point parameters with restrictive syntax.
    /// Only allows `id: type` or `#[attr] id: type`, not general patterns.
    fn parse_entry_params(&mut self) -> Result<Vec<Pattern>> {
        trace!("parse_entry_params: next token = {:?}", self.peek());
        let _start_span = self.current_span();
        self.expect(Token::LeftParen)?;
        let mut params = Vec::new();

        if !self.check(&Token::RightParen) {
            loop {
                let param_start = self.current_span();

                // Parse optional attributes (can be multiple)
                let attrs =
                    if self.check(&Token::AttributeStart) { self.parse_attributes()? } else { vec![] };

                // Must be an identifier
                let name = self.expect_identifier()?;
                let name_span = self.previous_span();

                // Must have : type
                self.expect(Token::Colon)?;
                let ty = self.parse_type()?;

                // Build pattern: if attrs present, Typed(Attributed(attrs, Name), ty)
                // otherwise just Typed(Name, ty)
                let name_pat = self.node_counter.mk_node(PatternKind::Name(name), name_span);

                let inner_pat = if !attrs.is_empty() {
                    let span = param_start.merge(&name_span);
                    self.node_counter.mk_node(PatternKind::Attributed(attrs, Box::new(name_pat)), span)
                } else {
                    name_pat
                };

                let span = param_start.merge(&self.previous_span());
                let typed_pat =
                    self.node_counter.mk_node(PatternKind::Typed(Box::new(inner_pat), ty), span);

                params.push(typed_pat);

                if !self.check(&Token::Comma) {
                    break;
                }
                self.advance(); // consume comma
            }
        }

        self.expect(Token::RightParen)?;
        Ok(params)
    }

    /// Parse return type with optional attributes, returning parallel arrays
    /// Returns (return_types, return_attributes)
    fn parse_return_type(&mut self) -> Result<(Vec<Type>, Vec<Option<Attribute>>)> {
        trace!("parse_return_type: next token = {:?}", self.peek());

        // Check if it's a tuple: ([attr1] type1, [attr2] type2, ...)
        if self.check(&Token::LeftParen) {
            self.advance(); // consume '('
            let mut types = Vec::new();
            let mut attributes = Vec::new();

            if !self.check(&Token::RightParen) {
                loop {
                    // Parse optional #[attribute]
                    let attr = if self.check(&Token::AttributeStart) {
                        self.advance(); // consume '#['
                        let attribute = self.parse_attribute()?;
                        Some(attribute)
                    } else {
                        None
                    };

                    // Parse the type
                    let ty = self.parse_type()?;

                    types.push(ty);
                    attributes.push(attr);

                    if !self.check(&Token::Comma) {
                        break;
                    }
                    self.advance(); // consume ','
                }
            }

            self.expect(Token::RightParen)?;
            Ok((types, attributes))
        } else if self.check(&Token::AttributeStart) {
            // Single attributed type: #[attribute] type
            self.advance(); // consume '#['
            let attribute = self.parse_attribute()?;
            let ty = self.parse_type()?;

            Ok((vec![ty], vec![Some(attribute)]))
        } else {
            // Regular single return type without attributes
            let ty = self.parse_type()?;
            Ok((vec![ty], vec![None]))
        }
    }

    fn parse_uniform_decl(&mut self, set: u32, binding: u32) -> Result<Declaration> {
        // Consume 'def' keyword
        self.expect(Token::Def)?;

        let name = self.expect_identifier()?;

        // Uniforms must have an explicit type annotation
        self.expect(Token::Colon)?;
        let ty = self.parse_type()?;

        // Uniforms must NOT have initializers
        if self.check(&Token::Assign) {
            bail_parse!("Uniform declarations cannot have initializer values");
        }

        Ok(Declaration::Uniform(UniformDecl {
            name,
            ty,
            set,
            binding,
        }))
    }

    fn parse_storage_decl(
        &mut self,
        set: u32,
        binding: u32,
        layout: StorageLayout,
        access: StorageAccess,
    ) -> Result<Declaration> {
        // Consume 'def' keyword
        self.expect(Token::Def)?;

        let name = self.expect_identifier()?;

        // Storage buffers must have an explicit type annotation
        self.expect(Token::Colon)?;
        let ty = self.parse_type()?;

        // Storage buffers must NOT have initializers
        if self.check(&Token::Assign) {
            bail_parse!("Storage buffer declarations cannot have initializer values");
        }

        Ok(Declaration::Storage(StorageDecl {
            name,
            ty,
            set,
            binding,
            layout,
            access,
        }))
    }

    fn parse_attribute(&mut self) -> Result<Attribute> {
        trace!("parse_attribute: next token = {:?}", self.peek());
        let attr_name = self.expect_identifier()?;

        match attr_name.as_str() {
            "vertex" => {
                self.expect(Token::RightBracket)?;
                Ok(Attribute::Vertex)
            }
            "fragment" => {
                self.expect(Token::RightBracket)?;
                Ok(Attribute::Fragment)
            }
            "compute" => {
                if self.check(&Token::LeftParen) {
                    bail_parse!("#[compute] takes no parameters");
                }
                self.expect(Token::RightBracket)?;
                Ok(Attribute::Compute)
            }
            "uniform" => {
                // Parse uniform attribute: #[uniform(binding=N)] or #[uniform(set=M, binding=N)]
                self.expect(Token::LeftParen)?;

                let mut set: u32 = 0;
                let mut binding: Option<u32> = None;

                loop {
                    let param_name = self.expect_identifier()?;
                    self.expect(Token::Assign)?;

                    match param_name.as_str() {
                        "set" => {
                            set = self.expect_integer()?;
                        }
                        "binding" => {
                            binding = Some(self.expect_integer()?);
                        }
                        _ => bail_parse!("Unknown uniform parameter: {}", param_name),
                    }

                    // Check for comma or end
                    if self.check(&Token::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }

                self.expect(Token::RightParen)?;
                self.expect(Token::RightBracket)?;

                let binding =
                    binding.ok_or_else(|| err_parse!("uniform attribute requires 'binding' parameter"))?;

                Ok(Attribute::Uniform { set, binding })
            }
            "builtin" => {
                self.expect(Token::LeftParen)?;
                let builtin_name = self.expect_identifier()?;
                self.expect(Token::RightParen)?;
                self.expect(Token::RightBracket)?;

                let builtin = match builtin_name.as_str() {
                    // Vertex shader builtins
                    "position" => spirv::BuiltIn::Position,
                    "vertex_index" => spirv::BuiltIn::VertexIndex,
                    "instance_index" => spirv::BuiltIn::InstanceIndex,
                    // Fragment shader builtins
                    "front_facing" => spirv::BuiltIn::FrontFacing,
                    "frag_coord" => spirv::BuiltIn::FragCoord,
                    "frag_depth" => spirv::BuiltIn::FragDepth,
                    // Compute shader builtins
                    "global_invocation_id" => spirv::BuiltIn::GlobalInvocationId,
                    "local_invocation_id" => spirv::BuiltIn::LocalInvocationId,
                    "workgroup_id" => spirv::BuiltIn::WorkgroupId,
                    "num_workgroups" => spirv::BuiltIn::NumWorkgroups,
                    _ => {
                        bail_parse!("Unknown builtin: {}", builtin_name);
                    }
                };
                Ok(Attribute::BuiltIn(builtin))
            }
            "location" => {
                self.expect(Token::LeftParen)?;
                let location = if let Some(Token::IntLiteral(location)) = self.advance() {
                    u32::try_from(location).map_err(|_| err_parse!("Invalid location number"))?
                } else {
                    bail_parse!("Expected location number");
                };
                self.expect(Token::RightParen)?;
                self.expect(Token::RightBracket)?;
                Ok(Attribute::Location(location))
            }
            "storage" => {
                // Parse storage attribute: #[storage(binding=N)] or #[storage(set=M, binding=N)]
                // Optional: layout=std430|std140, access=read|write|readwrite
                self.expect(Token::LeftParen)?;

                let mut set: u32 = 0;
                let mut binding: Option<u32> = None;
                let mut layout = StorageLayout::default();
                let mut access = StorageAccess::default();

                loop {
                    let param_name = self.expect_identifier()?;
                    self.expect(Token::Assign)?;

                    match param_name.as_str() {
                        "set" => {
                            set = self.expect_integer()?;
                        }
                        "binding" => {
                            binding = Some(self.expect_integer()?);
                        }
                        "layout" => {
                            let layout_name = self.expect_identifier()?;
                            layout = match layout_name.as_str() {
                                "std430" => StorageLayout::Std430,
                                "std140" => StorageLayout::Std140,
                                _ => bail_parse!("Unknown storage layout: {}", layout_name),
                            };
                        }
                        "access" => {
                            let access_name = self.expect_identifier()?;
                            access = match access_name.as_str() {
                                "read" => StorageAccess::ReadOnly,
                                "write" => StorageAccess::WriteOnly,
                                "readwrite" => StorageAccess::ReadWrite,
                                _ => bail_parse!("Unknown storage access: {}", access_name),
                            };
                        }
                        _ => bail_parse!("Unknown storage parameter: {}", param_name),
                    }

                    // Check for comma or end
                    if self.check(&Token::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }

                self.expect(Token::RightParen)?;
                self.expect(Token::RightBracket)?;

                let binding =
                    binding.ok_or_else(|| err_parse!("storage attribute requires 'binding' parameter"))?;

                Ok(Attribute::Storage {
                    set,
                    binding,
                    layout,
                    access,
                })
            }
            "size_hint" => {
                // Parse size hint for dynamic arrays: #[size_hint(N)]
                self.expect(Token::LeftParen)?;
                let hint = self.expect_integer()?;
                self.expect(Token::RightParen)?;
                self.expect(Token::RightBracket)?;
                Ok(Attribute::SizeHint(hint))
            }
            "linked" => {
                // Parse linked SPIR-V function: #[linked("linkage_name")]
                self.expect(Token::LeftParen)?;
                let linkage_name = if let Some(Token::StringLiteral(s)) = self.advance() {
                    s.to_string()
                } else {
                    bail_parse!("Expected string literal for linkage name");
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
                        bail_parse!("Type parameters must be uppercase (got '{}')", name);
                    }
                    self.advance();
                    type_params.push(name);
                } else {
                    bail_parse!("Expected size parameter [n] or type parameter in generics");
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
    fn parse_comma_separated_params(&mut self) -> Result<Vec<Pattern>> {
        trace!("parse_comma_separated_params: next token = {:?}", self.peek());
        self.expect(Token::LeftParen)?;
        let mut params = Vec::new();

        if !self.check(&Token::RightParen) {
            loop {
                params.push(self.parse_pattern()?);
                if !self.check(&Token::Comma) {
                    break;
                }
                self.advance(); // consume comma
            }
        }

        self.expect(Token::RightParen)?;
        Ok(params)
    }

    fn parse_type(&mut self) -> Result<Type> {
        trace!("parse_type: next token = {:?}", self.peek());
        // Note: existential types (?k. [k]T) are NOT allowed here.
        // Use parse_return_type_simple for return types that allow existentials.

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
                    let param_type = self.parse_type()?;
                    self.expect(Token::RightParen)?;

                    // Must be followed by ->
                    if self.check(&Token::Arrow) {
                        self.advance();
                        let return_type = self.parse_type()?;
                        // Just use the param_type directly, ignoring the name
                        return Ok(types::function(param_type, return_type));
                    } else {
                        bail_parse!("Named parameter must be followed by ->");
                    }
                }
            }

            // Not a named parameter, restore position and parse normally
            self.current = saved_pos;
        }

        // Regular function type or type application
        let left = self.parse_type_application()?;

        // Handle function arrows: T1 -> T2 -> T3
        // Arrow is right-associative: a -> b -> c means a -> (b -> c)
        if self.check(&Token::Arrow) {
            self.advance();
            let right = self.parse_type()?; // Recursive call for right-associativity
            Ok(types::function(left, right))
        } else {
            Ok(left)
        }
    }

    /// Parse a return type, which may include an existential quantifier.
    /// Existential types (?k. [k]T) are only valid in return position.
    fn parse_return_type_simple(&mut self) -> Result<Type> {
        trace!("parse_return_type_simple: next token = {:?}", self.peek());
        // Check for existential size: ?k. type or ?k l. type
        if self.check(&Token::QuestionMark) {
            return self.parse_existential_type();
        }
        self.parse_type()
    }

    fn parse_existential_type(&mut self) -> Result<Type> {
        self.expect(Token::QuestionMark)?;
        let mut size_vars = Vec::new();

        // Parse one or more bare identifiers: ?k. or ?k l. or ?k l m.
        while let Some(Token::Identifier(name)) = self.peek().cloned() {
            size_vars.push(name);
            self.advance();
        }

        if size_vars.is_empty() {
            bail_parse!("Existential type must have at least one size variable");
        }

        self.expect(Token::Dot)?;
        let inner_type = self.parse_type()?;

        Ok(types::existential(size_vars, inner_type))
    }

    fn parse_type_application(&mut self) -> Result<Type> {
        trace!("parse_type_application: next token = {:?}", self.peek());

        let mut base = self.parse_array_or_base_type()?;

        // Type application loop: keep applying type arguments
        // Grammar: type_application ::= type type_arg | "*" type
        //          type_arg         ::= "[" [dim] "]" | type
        loop {
            if self.is_at_type_boundary() {
                break;
            }

            match self.peek() {
                // Array dimension application: [n] or []
                Some(Token::LeftBracket) | Some(Token::LeftBracketSpaced) => {
                    self.advance();

                    if self.check(&Token::RightBracket) {
                        // Empty brackets [] - unsized array with placeholder address space
                        // Array[elem, AddressPlaceholder, SizePlaceholder]
                        self.advance();
                        base = Type::Constructed(
                            TypeName::Array,
                            vec![
                                base,
                                Type::Constructed(TypeName::AddressPlaceholder, vec![]),
                                Type::Constructed(TypeName::SizePlaceholder, vec![]),
                            ],
                        );
                    } else if let Some(Token::Identifier(name)) = self.peek() {
                        // Size variable [n]
                        let size_var = name.clone();
                        self.advance();
                        self.expect(Token::RightBracket)?;
                        // Array[elem, AddressPlaceholder, SizeVar(n)]
                        base = Type::Constructed(
                            TypeName::Array,
                            vec![
                                base,
                                Type::Constructed(TypeName::AddressPlaceholder, vec![]),
                                types::size_var(size_var),
                            ],
                        );
                    } else if let Some(Token::IntLiteral(n)) = self.peek() {
                        // Size literal [3]
                        let size = usize::try_from(n).map_err(|_| err_parse!("Invalid array size"))?;
                        self.advance();
                        self.expect(Token::RightBracket)?;
                        // Array[elem, AddressPlaceholder, Size(n)]
                        base = Type::Constructed(
                            TypeName::Array,
                            vec![
                                base,
                                Type::Constructed(TypeName::AddressPlaceholder, vec![]),
                                Type::Constructed(TypeName::Size(size), vec![]),
                            ],
                        );
                    } else {
                        bail_parse!("Expected size in array type application");
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

        Ok(base)
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

    fn parse_array_or_base_type(&mut self) -> Result<Type> {
        trace!("parse_array_or_base_type: next token = {:?}", self.peek());
        // Check for uniqueness prefix *
        if matches!(self.peek(), Some(Token::BinOp(op)) if op == "*") {
            self.advance(); // consume '*'
            let inner_type = self.parse_array_or_base_type()?;
            return Ok(types::unique(inner_type));
        }

        // Check for array type [dim]baseType (Futhark style)
        // Accept both LeftBracket and LeftBracketSpaced in type position
        if self.check(&Token::LeftBracket) || self.check(&Token::LeftBracketSpaced) {
            self.advance(); // consume '['

            // Check for empty brackets [] - unsized array with unknown address space
            if self.check(&Token::RightBracket) {
                self.advance();
                let elem_type = self.parse_array_or_base_type()?;
                return Ok(Type::Constructed(
                    TypeName::Array,
                    vec![
                        elem_type,
                        Type::Constructed(TypeName::AddressPlaceholder, vec![]),
                        Type::Constructed(TypeName::SizePlaceholder, vec![]),
                    ],
                ));
            }

            // Parse dimension - could be integer literal or identifier (size variable)
            if let Some(Token::IntLiteral(n)) = self.peek() {
                let size = usize::try_from(n).map_err(|_| err_parse!("Invalid array size"))?;
                self.advance();
                self.expect(Token::RightBracket)?;
                let elem_type = self.parse_array_or_base_type()?; // Allow nested arrays
                // Array[elem, AddressPlaceholder, Size(n)]
                Ok(Type::Constructed(
                    TypeName::Array,
                    vec![
                        elem_type,
                        Type::Constructed(TypeName::AddressPlaceholder, vec![]),
                        Type::Constructed(TypeName::Size(size), vec![]),
                    ],
                ))
            } else if let Some(Token::Identifier(name)) = self.peek() {
                // Size variable [n]
                let size_var = name.clone();
                self.advance();
                self.expect(Token::RightBracket)?;
                let elem_type = self.parse_array_or_base_type()?;
                // Array[elem, AddressPlaceholder, SizeVar(n)]
                Ok(Type::Constructed(
                    TypeName::Array,
                    vec![
                        elem_type,
                        Type::Constructed(TypeName::AddressPlaceholder, vec![]),
                        types::size_var(size_var),
                    ],
                ))
            } else {
                Err(err_parse!("Expected size literal or variable in array type"))
            }
        } else {
            self.parse_base_type()
        }
    }

    fn parse_base_type(&mut self) -> Result<Type> {
        trace!("parse_base_type: next token = {:?}", self.peek());

        // Check for vector/matrix types first to avoid borrow issues
        if let Some(Token::Identifier(name)) = self.peek() {
            let name_str = name.clone();
            if let Some(ty) = get_vector_types().get(&name_str) {
                self.advance();
                return Ok(ty.clone());
            }
            if let Some(ty) = get_matrix_types().get(&name_str) {
                self.advance();
                return Ok(ty.clone());
            }
        }

        match self.peek() {
            Some(Token::Identifier(name)) if name == "i32" => {
                self.advance();
                Ok(types::i32())
            }
            Some(Token::Identifier(name)) if name == "f32" => {
                self.advance();
                Ok(types::f32())
            }
            Some(Token::Identifier(name)) if name.chars().next().unwrap().is_lowercase() => {
                let type_name = name.clone();
                self.advance();

                // Check for qualified type name (module.typename)
                if self.check(&Token::Dot) {
                    self.advance(); // consume '.'
                    let inner_name = self.expect_identifier()?;
                    let qualified = format!("{}.{}", type_name, inner_name);
                    return Ok(Type::Constructed(TypeName::Named(qualified), vec![]));
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
                    "bool" => TypeName::Str("bool"),
                    // User-defined type alias or unrecognized type
                    _ => TypeName::Named(type_name),
                };
                Ok(Type::Constructed(type_name_variant, vec![]))
            }
            Some(Token::LeftParen) => {
                // Tuple type (T1, T2, T3), empty tuple (), or parenthesized type (T)
                self.advance(); // consume '('
                let mut tuple_types = Vec::new();
                let mut has_comma = false;

                if !self.check(&Token::RightParen) {
                    loop {
                        tuple_types.push(self.parse_type()?);
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
                    Ok(tuple_types.into_iter().next().unwrap())
                } else {
                    Ok(types::tuple(tuple_types))
                }
            }
            Some(Token::LeftBrace) => {
                // Record type {field1: type1, field2: type2} or empty record {}
                self.parse_record_type()
            }
            Some(Token::Identifier(name)) if name.chars().next().unwrap().is_uppercase() => {
                let name = name.clone();
                self.advance();

                // Check for qualified type (e.g., R.t for functor param module member)
                if self.check(&Token::Dot) {
                    self.advance();
                    let member = self.expect_identifier()?;
                    let qualified = format!("{}.{}", name, member);
                    return Ok(Type::Constructed(TypeName::Named(qualified), vec![]));
                }

                // All-caps: type variable (T, UV, R1). CamelCase: sum type (Some, None)
                let is_type_var = name.chars().all(|c| c.is_uppercase() || c.is_ascii_digit());
                if is_type_var {
                    Ok(Type::Constructed(TypeName::UserVar(name), vec![]))
                } else {
                    // Back up for parse_sum_type which expects to see the identifier
                    self.current -= 1;
                    self.parse_sum_type()
                }
            }
            _ => {
                let span = self.current_span();
                Err(err_parse_at!(span, "Expected type"))
            }
        }
    }

    fn parse_record_type(&mut self) -> Result<Type> {
        self.expect(Token::LeftBrace)?;
        let mut fields = Vec::new();

        if !self.check(&Token::RightBrace) {
            loop {
                // Parse field identifier (can be a number or name)
                let field_name = match self.peek() {
                    Some(Token::Identifier(name)) => {
                        let n = name.clone();
                        self.advance();
                        n
                    }
                    Some(Token::IntLiteral(n)) => {
                        let num = n.to_string();
                        self.advance();
                        num
                    }
                    _ => {
                        bail_parse!("Expected field name or number");
                    }
                };

                self.expect(Token::Colon)?;
                let field_type = self.parse_type()?;
                fields.push((field_name, field_type));

                if !self.check(&Token::Comma) {
                    break;
                }
                self.advance(); // consume ','

                // Allow trailing comma
                if self.check(&Token::RightBrace) {
                    break;
                }
            }
        }

        self.expect(Token::RightBrace)?;
        Ok(types::record(fields))
    }

    fn parse_sum_type(&mut self) -> Result<Type> {
        let mut variants = Vec::new();

        loop {
            // Parse constructor name (uppercase identifier)
            let constructor_name = match self.peek() {
                Some(Token::Identifier(name)) if name.chars().next().unwrap().is_uppercase() => {
                    let n = name.clone();
                    self.advance();
                    n
                }
                _ => bail_parse!("Expected constructor name"),
            };

            // Parse zero or more type arguments for this constructor
            let mut arg_types = Vec::new();
            while !self.check(&Token::Pipe)
                && !self.check(&Token::RightParen)
                && !self.check(&Token::RightBracket)
                && !self.check(&Token::RightBrace)
                && !self.check(&Token::Comma)
                && !self.check(&Token::Arrow)
                && self.current < self.tokens.len()
            {
                // Try to parse a type argument
                // This is tricky - we need to avoid consuming tokens that aren't part of the sum type
                // For now, we'll be conservative and only parse simple types
                match self.peek() {
                    Some(Token::Identifier(_))
                    | Some(Token::LeftParen)
                    | Some(Token::LeftBrace)
                    | Some(Token::LeftBracket)
                    | Some(Token::LeftBracketSpaced) => {
                        arg_types.push(self.parse_array_or_base_type()?);
                    }
                    Some(Token::BinOp(op)) if op == "*" => {
                        arg_types.push(self.parse_array_or_base_type()?);
                    }
                    _ => break,
                }
            }

            variants.push((constructor_name, arg_types));

            // Check for more variants
            if self.check(&Token::Pipe) {
                self.advance(); // consume '|'
            } else {
                break;
            }
        }

        Ok(types::sum(variants))
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
                let ty = self.parse_type()?;
                let end_span = self.previous_span();
                let span = start_span.merge(&end_span);
                expr = self.node_counter.mk_node(ExprKind::TypeAscription(Box::new(expr), ty), span);
            }
            Some(Token::TypeCoercion) => {
                let start_span = expr.h.span;
                self.advance();
                let ty = self.parse_type()?;
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

    fn get_operator_precedence(op: &str) -> Option<(u32, bool)> {
        // Returns (precedence, is_left_associative)
        // Dominating precedence (higher number) binds tighter than dominated precedence
        // Based on SPECIFICATION.md operator precedence table:
        //   || (dominated) < && < comparisons < bitwise < shifts < +- < */% < |> < ** (dominating)
        match op {
            "||" => Some((1, true)), // Logical or (most dominated)
            "&&" => Some((2, true)), // Logical and
            "==" | "!=" | "<" | ">" | "<=" | ">=" => Some((3, true)), // Comparison operators
            "&" | "^" | "|" => Some((4, true)), // Bitwise operators
            "<<" | ">>" | ">>>" => Some((5, true)), // Bitwise shifts
            "+" | "-" => Some((6, true)), // Addition and subtraction
            "*" | "/" | "%" | "//" | "%%" => Some((7, true)), // Multiplication, division, modulo
            "|>" => Some((8, true)), // Pipe operator
            "**" => Some((9, true)), // Exponentiation (most dominating binary)
            _ => None,
        }
    }

    fn parse_binary_expression_with_precedence(&mut self, dominated_by: u32) -> Result<Expression> {
        trace!(
            "parse_binary_expression_with_precedence({}): next token = {:?}",
            dominated_by,
            self.peek()
        );
        let mut left = self.parse_unary_expression()?;

        loop {
            // Handle 'with' as a special left-associative operator
            // with has precedence 10 (dominates all binary operators)
            // arr with [i] = v with [j] = w parses as ((arr with [i] = v) with [j] = w)
            if self.check(&Token::With) && dominated_by <= 10 {
                let start_span = left.h.span;
                self.advance(); // consume 'with'

                // Accept either LeftBracket or LeftBracketSpaced after 'with'
                if self.check(&Token::LeftBracket) || self.check(&Token::LeftBracketSpaced) {
                    self.advance();
                } else {
                    bail_parse!("Expected '[' after 'with'");
                }
                let index = self.parse_expression()?;
                self.expect(Token::RightBracket)?;
                self.expect(Token::Assign)?;

                // Parse value dominated by 11 for left-associativity
                let value = self.parse_binary_expression_with_precedence(11)?;
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

            let right = self.parse_binary_expression_with_precedence(right_dominated_by)?;

            // Build the appropriate operation with span from left to right
            let span = left.h.span.merge(&right.h.span);
            left = if op_string == "|>" {
                // Desugar pipe: a |> f  =>  f(a)
                self.node_counter.mk_node(ExprKind::Application(Box::new(right), vec![left]), span)
            } else {
                // Regular binary operation
                self.node_counter.mk_node(
                    ExprKind::BinaryOp(BinaryOp { op: op_string }, Box::new(left), Box::new(right)),
                    span,
                )
            };
        }

        Ok(left)
    }

    // Function application: f(x, y, z)
    // Now uses tuple-style syntax with parentheses
    fn parse_application_expression(&mut self) -> Result<Expression> {
        trace!("parse_application_expression: next token = {:?}", self.peek());
        // Postfix expression now handles function calls with ()
        self.parse_postfix_expression()
    }

    /// Parse comma-separated arguments for function calls: (x, y, z)
    /// Returns empty vec for (), single element for (x), etc.
    fn parse_call_arguments(&mut self) -> Result<Vec<Expression>> {
        trace!("parse_call_arguments: next token = {:?}", self.peek());
        let mut args = Vec::new();

        // Handle empty argument list: f()
        if self.check(&Token::RightParen) {
            return Ok(args);
        }

        // Parse first argument
        args.push(self.parse_expression()?);

        // Parse remaining arguments separated by commas
        while self.check(&Token::Comma) {
            self.advance(); // consume ','
            // Allow trailing comma: f(x, y,)
            if self.check(&Token::RightParen) {
                break;
            }
            args.push(self.parse_expression()?);
        }

        Ok(args)
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
                    // Field access (e.g., v.x, v.y, v.z, v.w)
                    let start_span = expr.h.span;
                    self.advance();
                    let field_name = self.expect_identifier()?;
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
                    expr = self.node_counter.mk_node(ExprKind::Application(Box::new(expr), args), span);
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    /// Parse either array indexing `a[i]` or array slicing `a[start..end]`
    /// Called after consuming the `[` token
    fn parse_index_or_slice(&mut self, array: Expression, start_span: Span) -> Result<Expression> {
        // Check if we have a DotDot immediately ([..end])
        let start_expr = if self.check(&Token::DotDot) {
            None
        } else if self.check(&Token::RightBracket) {
            // Empty brackets a[] - this is an error
            bail_parse!("Expected index or slice expression");
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
            let index = start_expr.expect("index expression should exist for array indexing");
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
                Ok(self.node_counter.mk_node(
                    ExprKind::UnaryOp(UnaryOp { op: "-".to_string() }, Box::new(operand)),
                    span,
                ))
            }
            Some(Token::Bang) => {
                let start_span = self.current_span();
                self.advance();
                let operand = self.parse_unary_expression()?; // Right-associative for chaining: !!x
                let span = start_span.merge(&operand.h.span);
                Ok(self.node_counter.mk_node(
                    ExprKind::UnaryOp(UnaryOp { op: "!".to_string() }, Box::new(operand)),
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
                Ok(self.node_counter.mk_node(ExprKind::TypeHole, span))
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
                    _ => bail_parse!("Invalid suffixed literal"),
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
            Some(Token::StringLiteral(s)) => {
                let s = s.clone();
                let span = self.current_span();
                self.advance();
                Ok(self.node_counter.mk_node(ExprKind::StringLiteral(s), span))
            }
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                let span = self.current_span();
                self.advance();
                Ok(self.node_counter.mk_node(ExprKind::Identifier(vec![], name), span))
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
                    let op = op.clone();
                    self.advance(); // consume operator
                    self.advance(); // consume )
                    let end_span = self.previous_span();
                    let span = start_span.merge(&end_span);

                    // Create patterns for parameters: x, y
                    let x_pattern = self.node_counter.mk_node(PatternKind::Name("x".to_string()), span);
                    let y_pattern = self.node_counter.mk_node(PatternKind::Name("y".to_string()), span);

                    // Create identifier expressions for body: x, y
                    let x_expr =
                        self.node_counter.mk_node(ExprKind::Identifier(vec![], "x".to_string()), span);
                    let y_expr =
                        self.node_counter.mk_node(ExprKind::Identifier(vec![], "y".to_string()), span);

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
                    while self.check(&Token::Comma) {
                        self.advance(); // consume ','
                        if self.check(&Token::RightParen) {
                            break; // Allow trailing comma
                        }
                        elements.push(self.parse_expression()?);
                    }
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
            Some(Token::DollarSign) => self.parse_curry_expression(),
            _ => {
                let span = self.current_span();
                Err(err_parse!(
                    "Expected expression, got {:?} at {}",
                    self.peek(),
                    span
                ))
            }
        }
    }

    /// Parse a curry expression: $expr(args) where args may contain _ placeholders
    /// Desugars to a lambda if there are placeholders
    fn parse_curry_expression(&mut self) -> Result<Expression> {
        trace!("parse_curry_expression: next token = {:?}", self.peek());
        let start_span = self.current_span();
        self.expect(Token::DollarSign)?;

        // Parse the function expression (identifier, field access, array index, but NOT call)
        let func = self.parse_curry_base()?;

        // Require parenthesized arguments
        self.expect(Token::LeftParen)?;
        let args = self.parse_curry_arguments()?;
        self.expect(Token::RightParen)?;
        let end_span = self.previous_span();
        let span = start_span.merge(&end_span);

        // Check for underscore placeholders
        let placeholder_indices: Vec<usize> = args
            .iter()
            .enumerate()
            .filter_map(
                |(i, arg)| {
                    if matches!(arg, CurryArg::Placeholder) { Some(i) } else { None }
                },
            )
            .collect();

        if placeholder_indices.is_empty() {
            // No placeholders - just a normal application
            let real_args: Vec<Expression> = args
                .into_iter()
                .map(|a| match a {
                    CurryArg::Expr(e) => e,
                    CurryArg::Placeholder => unreachable!(),
                })
                .collect();
            return Ok(self.node_counter.mk_node(ExprKind::Application(Box::new(func), real_args), span));
        }

        // Desugar to lambda
        self.desugar_curry(func, args, span)
    }

    /// Parse the base of a curry expression (function without the call)
    /// Handles identifiers, field access, and array indexing, but stops before (
    fn parse_curry_base(&mut self) -> Result<Expression> {
        let mut expr = self.parse_primary_expression()?;

        loop {
            match self.peek() {
                Some(Token::LeftBracket) => {
                    // Array indexing: arr[0]
                    let start_span = expr.h.span;
                    self.advance();
                    let index = self.parse_expression()?;
                    self.expect(Token::RightBracket)?;
                    let end_span = self.previous_span();
                    let span = start_span.merge(&end_span);
                    expr = self
                        .node_counter
                        .mk_node(ExprKind::ArrayIndex(Box::new(expr), Box::new(index)), span);
                }
                Some(Token::Dot) => {
                    // Field access (v.x)
                    let start_span = expr.h.span;
                    self.advance();
                    let field_name = self.expect_identifier()?;
                    let end_span = self.previous_span();
                    let span = start_span.merge(&end_span);
                    expr =
                        self.node_counter.mk_node(ExprKind::FieldAccess(Box::new(expr), field_name), span);
                }
                // Stop before ( - that's handled by parse_curry_expression
                _ => break,
            }
        }

        Ok(expr)
    }

    /// Parse curry arguments, treating _ as a placeholder
    fn parse_curry_arguments(&mut self) -> Result<Vec<CurryArg>> {
        let mut args = Vec::new();

        // Handle empty argument list
        if self.check(&Token::RightParen) {
            return Ok(args);
        }

        loop {
            if self.check(&Token::Underscore) {
                self.advance();
                args.push(CurryArg::Placeholder);
            } else {
                args.push(CurryArg::Expr(self.parse_expression()?));
            }

            if !self.check(&Token::Comma) {
                break;
            }
            self.advance();
            // Allow trailing comma
            if self.check(&Token::RightParen) {
                break;
            }
        }

        Ok(args)
    }

    /// Desugar curry expression to lambda
    /// $f(a, _, b, _) -> |_0_, _1_| f(a, _0_, b, _1_)
    fn desugar_curry(&mut self, func: Expression, args: Vec<CurryArg>, span: Span) -> Result<Expression> {
        // Generate lambda params: _0_, _1_, ...
        let mut params = Vec::new();
        let mut param_idx = 0;

        for arg in &args {
            if matches!(arg, CurryArg::Placeholder) {
                let name = format!("_{}_", param_idx);
                let param = self.node_counter.mk_node(PatternKind::Name(name), span);
                params.push(param);
                param_idx += 1;
            }
        }

        // Build argument list, replacing placeholders with param references
        let mut param_idx = 0;
        let call_args: Vec<Expression> = args
            .into_iter()
            .map(|arg| match arg {
                CurryArg::Placeholder => {
                    let name = format!("_{}_", param_idx);
                    param_idx += 1;
                    self.node_counter.mk_node(ExprKind::Identifier(vec![], name), span)
                }
                CurryArg::Expr(e) => e,
            })
            .collect();

        // Build the function call: func(args...)
        let body = self.node_counter.mk_node(ExprKind::Application(Box::new(func), call_args), span);

        // Build the lambda: |_0_, _1_, ...| body
        let lambda = LambdaExpr {
            params,
            body: Box::new(body),
        };

        Ok(self.node_counter.mk_node(ExprKind::Lambda(lambda), span))
    }

    fn parse_array_literal(&mut self) -> Result<Expression> {
        trace!("parse_array_literal: next token = {:?}", self.peek());
        // Accept either LeftBracket or LeftBracketSpaced
        let start_span = self.current_span();
        match self.peek() {
            Some(Token::LeftBracket) | Some(Token::LeftBracketSpaced) => {
                self.advance();
            }
            _ => bail_parse!("Expected '['"),
        }

        let mut elements = Vec::new();
        if !self.check(&Token::RightBracket) {
            loop {
                elements.push(self.parse_expression()?);
                if !self.check(&Token::Comma) {
                    break;
                }
                self.advance();
            }
        }

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

        let mut elements = Vec::new();
        if !self.check(&Token::RightBracket) {
            loop {
                elements.push(self.parse_expression()?);
                if !self.check(&Token::Comma) {
                    break;
                }
                self.advance();
            }
        }

        self.expect(Token::RightBracket)?;
        let end_span = self.previous_span();
        let span = start_span.merge(&end_span);
        Ok(self.node_counter.mk_node(ExprKind::VecMatLiteral(elements), span))
    }

    fn parse_record_literal(&mut self) -> Result<Expression> {
        trace!("parse_record_literal: next token = {:?}", self.peek());
        let start_span = self.current_span();
        self.expect(Token::LeftBrace)?;

        let mut fields = Vec::new();

        // Empty record: {}
        if self.check(&Token::RightBrace) {
            self.advance();
            let end_span = self.previous_span();
            let span = start_span.merge(&end_span);
            return Ok(self.node_counter.mk_node(ExprKind::RecordLiteral(fields), span));
        }

        // Parse field: value pairs
        loop {
            // Parse field name
            let field_name = if let Some(Token::Identifier(name)) = self.peek() {
                let name = name.clone();
                self.advance();
                name
            } else {
                bail_parse!(
                    "Expected field name in record literal, got {:?} at {}",
                    self.peek(),
                    self.current_span()
                );
            };

            // Expect colon
            self.expect(Token::Colon)?;

            // Parse field value
            let field_value = self.parse_expression()?;

            fields.push((field_name, field_value));

            // Check for comma or end of record
            if self.check(&Token::Comma) {
                self.advance();
                // Allow trailing comma
                if self.check(&Token::RightBrace) {
                    break;
                }
            } else {
                break;
            }
        }

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

            let mut params = Vec::new();
            if !self.check(&Token::Pipe) {
                loop {
                    params.push(self.parse_pattern()?);
                    if !self.check(&Token::Comma) {
                        break;
                    }
                    self.advance(); // consume comma
                }
            }

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

        // Optional type annotation
        let ty = if self.check(&Token::Colon) {
            self.advance(); // consume ':'
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(Token::Assign)?;
        let value = Box::new(self.parse_expression()?);
        self.expect(Token::In)?;
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
                    self.expect(Token::BinOp("<".to_string()))?;
                    let bound = Box::new(self.parse_expression()?);
                    LoopForm::For(name, bound)
                }
            } else {
                bail_parse!("Expected pattern in for loop");
            }
        } else if self.check(&Token::While) {
            self.advance();
            let condition = Box::new(self.parse_expression()?);
            LoopForm::While(condition)
        } else {
            bail_parse!("Expected 'for' or 'while' in loop");
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
            bail_parse!("Match expression must have at least one case");
        }

        let span = start_span.merge(&last_span);
        Ok(self.node_counter.mk_node(ExprKind::Match(MatchExpr { scrutinee, cases }), span))
    }

    // Helper methods
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
                    bail_parse!(
                        "Expected operator or ) in operator section at {}",
                        self.current_span()
                    );
                }
            }
        }

        if operator.is_empty() {
            bail_parse!("Operator section cannot be empty");
        }

        // Validate that all characters are valid operator characters
        const VALID_OP_CHARS: &str = "+-*/%=!><&^|";
        for ch in operator.chars() {
            if !VALID_OP_CHARS.contains(ch) {
                bail_parse!("Invalid operator character '{}' in operator section", ch);
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
