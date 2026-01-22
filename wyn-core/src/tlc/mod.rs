//! Typed Lambda Calculus (TLC) representation.
//!
//! A minimal typed lambda calculus IR for SOAC fusion analysis.
//! Lambdas remain as values (not yet defunctionalized).

pub mod defunctionalize;
#[cfg(test)]
mod defunctionalize_tests;
pub mod partial_eval;
#[cfg(test)]
mod partial_eval_tests;
pub mod specialize;
pub mod to_mir;
#[cfg(test)]
mod to_mir_tests;

use crate::TypeTable;
use crate::ast::{self, NodeId, Span, TypeName};
use polytype::Type;
use std::collections::HashMap;

// =============================================================================
// TLC Terms
// =============================================================================

/// A unique identifier for TLC terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TermId(pub u32);

/// Counter for generating unique TermIds.
#[derive(Debug, Clone, Default)]
pub struct TermIdSource {
    next: u32,
}

impl TermIdSource {
    pub fn new() -> Self {
        Self { next: 0 }
    }

    pub fn next_id(&mut self) -> TermId {
        let id = TermId(self.next);
        self.next += 1;
        id
    }
}

/// A typed term in the lambda calculus.
#[derive(Debug, Clone)]
pub struct Term {
    pub id: TermId,
    pub ty: Type<TypeName>,
    pub span: Span,
    pub kind: TermKind,
}

/// The kind of term.
#[derive(Debug, Clone)]
pub enum TermKind {
    /// Variable reference.
    Var(String),

    /// Binary operator as a value: +, -, *, /, ==, etc.
    BinOp(ast::BinaryOp),

    /// Unary operator as a value: -, !
    UnOp(ast::UnaryOp),

    /// Lambda abstraction: λ(x:T). body
    Lam {
        param: String,
        param_ty: Type<TypeName>,
        body: Box<Term>,
    },

    /// Application: f x
    App {
        func: Box<Term>,
        arg: Box<Term>,
    },

    /// Let binding: let x:T = rhs in body
    Let {
        name: String,
        name_ty: Type<TypeName>,
        rhs: Box<Term>,
        body: Box<Term>,
    },

    /// Integer literal.
    IntLit(String),

    /// Float literal.
    FloatLit(f32),

    /// Boolean literal.
    BoolLit(bool),

    /// String literal.
    StringLit(String),

    /// Conditional: if cond then t else e
    If {
        cond: Box<Term>,
        then_branch: Box<Term>,
        else_branch: Box<Term>,
    },

    /// Loop construct (mirrors MIR::Loop).
    Loop {
        /// The loop accumulator variable name.
        loop_var: String,
        /// Type of the loop variable.
        loop_var_ty: Type<TypeName>,
        /// Initial value for the accumulator.
        init: Box<Term>,
        /// Bindings that extract from loop_var (e.g., for tuple destructuring).
        /// Each is (name, type, extraction_expr).
        init_bindings: Vec<(String, Type<TypeName>, Term)>,
        /// The kind of loop.
        kind: LoopKind,
        /// Loop body expression.
        body: Box<Term>,
    },
}

/// The kind of loop (mirrors MIR::LoopKind).
#[derive(Debug, Clone)]
pub enum LoopKind {
    /// For loop over an array: `for x in arr`.
    For {
        var: String,
        var_ty: Type<TypeName>,
        iter: Box<Term>,
    },
    /// For loop with range bound: `for i < n`.
    ForRange {
        var: String,
        var_ty: Type<TypeName>,
        bound: Box<Term>,
    },
    /// While loop: `while cond`.
    While {
        cond: Box<Term>,
    },
}

// =============================================================================
// TLC Program
// =============================================================================

/// Metadata about how a definition should be lowered to MIR.
#[derive(Debug, Clone)]
pub enum DefMeta {
    /// A regular function or constant.
    Function,
    /// A shader entry point - stores the original AST entry for metadata.
    EntryPoint(Box<ast::EntryDecl>),
}

/// A top-level definition in TLC.
#[derive(Debug, Clone)]
pub struct Def {
    pub name: String,
    pub ty: Type<TypeName>,
    pub body: Term,
    pub meta: DefMeta,
    /// Number of arguments this function expects (for uncurrying).
    pub arity: usize,
}

/// A TLC program (collection of definitions).
#[derive(Debug, Clone)]
pub struct Program {
    pub defs: Vec<Def>,
    /// Uniform declarations (no bodies, just metadata).
    pub uniforms: Vec<ast::UniformDecl>,
    /// Storage buffer declarations (no bodies, just metadata).
    pub storage: Vec<ast::StorageDecl>,
}

// =============================================================================
// AST to TLC Transformation
// =============================================================================

/// A pending let-binding to be applied after all lambdas are created.
#[derive(Debug, Clone)]
struct PendingBinding {
    name: String,
    ty: Type<TypeName>,
    expr: Term,
}

/// Context for transforming AST to TLC.
pub struct Transformer<'a> {
    type_table: &'a TypeTable,
    term_ids: TermIdSource,
    /// Optional namespace prefix for definition names (e.g., "f32" -> "f32.pi")
    namespace: Option<String>,
}

impl<'a> Transformer<'a> {
    pub fn new(type_table: &'a TypeTable) -> Self {
        Self {
            type_table,
            term_ids: TermIdSource::new(),
            namespace: None,
        }
    }

    /// Create a transformer with a namespace prefix for definition names.
    pub fn with_namespace(type_table: &'a TypeTable, namespace: &str) -> Self {
        Self {
            type_table,
            term_ids: TermIdSource::new(),
            namespace: Some(namespace.to_string()),
        }
    }

    /// Transform an AST program to TLC.
    pub fn transform_program(&mut self, program: &ast::Program) -> Program {
        let mut defs = Vec::new();
        let mut uniforms = Vec::new();
        let mut storage = Vec::new();

        for decl in &program.declarations {
            match decl {
                ast::Declaration::Decl(d) => {
                    if let Some(def) = self.transform_decl(d) {
                        defs.push(def);
                    }
                }
                ast::Declaration::Entry(e) => {
                    if let Some(def) = self.transform_entry(e) {
                        defs.push(def);
                    }
                }
                ast::Declaration::Uniform(u) => {
                    uniforms.push(u.clone());
                }
                ast::Declaration::Storage(s) => {
                    storage.push(s.clone());
                }
                ast::Declaration::Sig(_)
                | ast::Declaration::TypeBind(_)
                | ast::Declaration::Module(_)
                | ast::Declaration::ModuleTypeBind(_)
                | ast::Declaration::Open(_)
                | ast::Declaration::Import(_) => {}
            }
        }

        Program {
            defs,
            uniforms,
            storage,
        }
    }

    pub fn transform_decl(&mut self, decl: &ast::Decl) -> Option<Def> {
        let body_ty = self.lookup_type(decl.body.h.id)?;
        let full_ty = self.build_function_type(&decl.params, &body_ty);
        let body = self.transform_with_params(&decl.params, &decl.body);

        // Apply namespace prefix if set (e.g., "f32" + "pi" -> "f32.pi")
        let name = match &self.namespace {
            Some(ns) => format!("{}.{}", ns, decl.name),
            None => decl.name.clone(),
        };

        Some(Def {
            name,
            ty: full_ty,
            body,
            meta: DefMeta::Function,
            arity: decl.params.len(),
        })
    }

    fn transform_entry(&mut self, entry: &ast::EntryDecl) -> Option<Def> {
        let body_ty = self.lookup_type(entry.body.h.id)?;
        let full_ty = self.build_function_type(&entry.params, &body_ty);
        let body = self.transform_with_params(&entry.params, &entry.body);

        Some(Def {
            name: entry.name.clone(),
            ty: full_ty,
            body,
            meta: DefMeta::EntryPoint(Box::new(entry.clone())),
            arity: entry.params.len(),
        })
    }

    fn build_function_type(&self, params: &[ast::Pattern], ret_ty: &Type<TypeName>) -> Type<TypeName> {
        let mut ty = ret_ty.clone();

        for param in params.iter().rev() {
            let param_ty = self.pattern_type(param);
            ty = Type::Constructed(TypeName::Arrow, vec![param_ty, ty]);
        }

        ty
    }

    fn pattern_type(&self, pattern: &ast::Pattern) -> Type<TypeName> {
        match &pattern.kind {
            // For attributed patterns, recurse into the inner pattern
            ast::PatternKind::Attributed(_, inner) => self.pattern_type(inner),
            // Always look up from type_table - the type checker has substituted UserVars
            // with Type::Variables. Using the AST type directly would retain UserVars.
            _ => self.lookup_type(pattern.h.id).expect("Pattern must have type in type table"),
        }
    }

    fn transform_with_params(&mut self, params: &[ast::Pattern], body: &ast::Expression) -> Term {
        if params.is_empty() {
            return self.transform_expr(body);
        }

        let inner = self.transform_with_params(&params[1..], body);
        let param = &params[0];
        let param_ty = self.pattern_type(param);

        self.pattern_to_lambda(param, param_ty, inner)
    }

    fn pattern_to_lambda(&mut self, pattern: &ast::Pattern, param_ty: Type<TypeName>, body: Term) -> Term {
        match &pattern.kind {
            ast::PatternKind::Name(name) => {
                let lam_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), body.ty.clone()]);
                self.mk_term(
                    lam_ty,
                    pattern.h.span,
                    TermKind::Lam {
                        param: name.clone(),
                        param_ty,
                        body: Box::new(body),
                    },
                )
            }

            ast::PatternKind::Wildcard => {
                let fresh = format!("_wild_{}", self.term_ids.next_id().0);
                let lam_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), body.ty.clone()]);
                self.mk_term(
                    lam_ty,
                    pattern.h.span,
                    TermKind::Lam {
                        param: fresh,
                        param_ty,
                        body: Box::new(body),
                    },
                )
            }

            ast::PatternKind::Typed(inner, _) => self.pattern_to_lambda(inner, param_ty, body),

            ast::PatternKind::Attributed(_, inner) => self.pattern_to_lambda(inner, param_ty, body),

            ast::PatternKind::Tuple(patterns) => {
                let fresh = format!("_tup_{}", self.term_ids.next_id().0);
                let component_types = self.extract_tuple_types(&param_ty, patterns.len());
                let inner =
                    self.flatten_tuple_pattern(&fresh, patterns, &component_types, body, pattern.h.span);

                let lam_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), inner.ty.clone()]);
                self.mk_term(
                    lam_ty,
                    pattern.h.span,
                    TermKind::Lam {
                        param: fresh,
                        param_ty,
                        body: Box::new(inner),
                    },
                )
            }

            ast::PatternKind::Record(fields) => {
                let fresh = format!("_rec_{}", self.term_ids.next_id().0);
                let field_types = self.extract_record_types(&param_ty);
                let inner = self.flatten_record_pattern(
                    &fresh,
                    &param_ty,
                    fields,
                    &field_types,
                    body,
                    pattern.h.span,
                );

                let lam_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), inner.ty.clone()]);
                self.mk_term(
                    lam_ty,
                    pattern.h.span,
                    TermKind::Lam {
                        param: fresh,
                        param_ty,
                        body: Box::new(inner),
                    },
                )
            }

            ast::PatternKind::Unit => {
                todo!("Unit patterns")
            }

            ast::PatternKind::Literal(_) => {
                todo!("Literal patterns in lambdas")
            }

            ast::PatternKind::Constructor(_, _) => {
                todo!("Constructor patterns in lambdas")
            }
        }
    }

    /// Collect bindings from a pattern without wrapping in let expressions.
    /// Returns (fresh_param_name, list_of_pending_bindings).
    /// Used by transform_lambda to defer bindings until after all lambdas are created.
    fn collect_pattern_bindings(
        &mut self,
        pattern: &ast::Pattern,
        param_ty: &Type<TypeName>,
        span: Span,
    ) -> (String, Vec<PendingBinding>) {
        match &pattern.kind {
            ast::PatternKind::Name(name) => {
                // Simple name - no extra bindings needed
                (name.clone(), vec![])
            }

            ast::PatternKind::Wildcard => {
                let fresh = format!("_wild_{}", self.term_ids.next_id().0);
                (fresh, vec![])
            }

            ast::PatternKind::Typed(inner, _) => self.collect_pattern_bindings(inner, param_ty, span),

            ast::PatternKind::Attributed(_, inner) => self.collect_pattern_bindings(inner, param_ty, span),

            ast::PatternKind::Tuple(patterns) => {
                let fresh = format!("_tup_{}", self.term_ids.next_id().0);
                let component_types = self.extract_tuple_types(param_ty, patterns.len());
                let bindings =
                    self.collect_tuple_bindings(&fresh, param_ty, patterns, &component_types, span);
                (fresh, bindings)
            }

            ast::PatternKind::Record(fields) => {
                let fresh = format!("_rec_{}", self.term_ids.next_id().0);
                let field_types = self.extract_record_types(param_ty);
                let bindings = self.collect_record_bindings(&fresh, param_ty, fields, &field_types, span);
                (fresh, bindings)
            }

            ast::PatternKind::Unit => {
                todo!("Unit patterns")
            }

            ast::PatternKind::Literal(_) => {
                todo!("Literal patterns in lambdas")
            }

            ast::PatternKind::Constructor(_, _) => {
                todo!("Constructor patterns in lambdas")
            }
        }
    }

    /// Collect bindings for a tuple pattern without wrapping in let.
    fn collect_tuple_bindings(
        &mut self,
        tuple_var: &str,
        tuple_ty: &Type<TypeName>,
        patterns: &[ast::Pattern],
        component_types: &[Type<TypeName>],
        span: Span,
    ) -> Vec<PendingBinding> {
        let mut bindings = Vec::new();

        for (i, pattern) in patterns.iter().enumerate() {
            let comp_ty = component_types
                .get(i)
                .cloned()
                .expect("BUG: Tuple pattern has more elements than tuple type");

            // Build projection: _w_tuple_proj tuple_var i
            let tuple_ref = self.mk_term(tuple_ty.clone(), span, TermKind::Var(tuple_var.to_string()));
            let index_lit = self.mk_term(
                Type::Constructed(TypeName::Int(32), vec![]),
                span,
                TermKind::IntLit(i.to_string()),
            );
            let proj = self.build_app2("_w_tuple_proj", tuple_ref, index_lit, comp_ty.clone(), span);

            self.collect_pattern_bindings_into(pattern, &comp_ty, proj, &mut bindings, span);
        }

        bindings
    }

    /// Collect bindings for a record pattern without wrapping in let.
    fn collect_record_bindings(
        &mut self,
        record_var: &str,
        record_ty: &Type<TypeName>,
        fields: &[ast::RecordPatternField],
        field_types: &HashMap<String, Type<TypeName>>,
        span: Span,
    ) -> Vec<PendingBinding> {
        let mut bindings = Vec::new();

        for field in fields {
            let field_ty = field_types
                .get(&field.field)
                .cloned()
                .unwrap_or_else(|| panic!("BUG: Record field '{}' not found in type", field.field));

            let field_idx = self
                .resolve_field_index(record_ty, &field.field)
                .unwrap_or_else(|| panic!("BUG: field '{}' not in record type", field.field));

            let record_ref = self.mk_term(record_ty.clone(), span, TermKind::Var(record_var.to_string()));
            let index_lit = self.mk_term(
                Type::Constructed(TypeName::Int(32), vec![]),
                span,
                TermKind::IntLit(field_idx.to_string()),
            );
            let field_access =
                self.build_app2("_w_tuple_proj", record_ref, index_lit, field_ty.clone(), span);

            if let Some(pat) = &field.pattern {
                self.collect_pattern_bindings_into(pat, &field_ty, field_access, &mut bindings, span);
            } else {
                bindings.push(PendingBinding {
                    name: field.field.clone(),
                    ty: field_ty,
                    expr: field_access,
                });
            }
        }

        bindings
    }

    /// Recursively collect bindings from a pattern into a list.
    fn collect_pattern_bindings_into(
        &mut self,
        pattern: &ast::Pattern,
        pat_ty: &Type<TypeName>,
        expr: Term,
        bindings: &mut Vec<PendingBinding>,
        span: Span,
    ) {
        match &pattern.kind {
            ast::PatternKind::Name(name) => {
                bindings.push(PendingBinding {
                    name: name.clone(),
                    ty: pat_ty.clone(),
                    expr,
                });
            }

            ast::PatternKind::Wildcard => {
                let fresh = format!("_wild_{}", self.term_ids.next_id().0);
                bindings.push(PendingBinding {
                    name: fresh,
                    ty: pat_ty.clone(),
                    expr,
                });
            }

            ast::PatternKind::Typed(inner, _) => {
                self.collect_pattern_bindings_into(inner, pat_ty, expr, bindings, span)
            }

            ast::PatternKind::Attributed(_, inner) => {
                self.collect_pattern_bindings_into(inner, pat_ty, expr, bindings, span)
            }

            ast::PatternKind::Tuple(patterns) => {
                let fresh = format!("_tup_{}", self.term_ids.next_id().0);
                let component_types = self.extract_tuple_types(pat_ty, patterns.len());

                // First bind the tuple to a fresh name
                bindings.push(PendingBinding {
                    name: fresh.clone(),
                    ty: pat_ty.clone(),
                    expr,
                });

                // Then recursively collect bindings for each component
                for (i, sub_pattern) in patterns.iter().enumerate() {
                    let comp_ty = component_types
                        .get(i)
                        .cloned()
                        .expect("BUG: Tuple pattern has more elements than tuple type");

                    let tuple_ref = self.mk_term(pat_ty.clone(), span, TermKind::Var(fresh.clone()));
                    let index_lit = self.mk_term(
                        Type::Constructed(TypeName::Int(32), vec![]),
                        span,
                        TermKind::IntLit(i.to_string()),
                    );
                    let proj =
                        self.build_app2("_w_tuple_proj", tuple_ref, index_lit, comp_ty.clone(), span);

                    self.collect_pattern_bindings_into(sub_pattern, &comp_ty, proj, bindings, span);
                }
            }

            ast::PatternKind::Record(fields) => {
                let fresh = format!("_rec_{}", self.term_ids.next_id().0);
                let field_types = self.extract_record_types(pat_ty);

                // First bind the record to a fresh name
                bindings.push(PendingBinding {
                    name: fresh.clone(),
                    ty: pat_ty.clone(),
                    expr,
                });

                // Then recursively collect bindings for each field
                for field in fields {
                    let field_ty = field_types
                        .get(&field.field)
                        .cloned()
                        .unwrap_or_else(|| panic!("BUG: Record field '{}' not found in type", field.field));

                    let field_idx = self
                        .resolve_field_index(pat_ty, &field.field)
                        .unwrap_or_else(|| panic!("BUG: field '{}' not in record type", field.field));

                    let record_ref = self.mk_term(pat_ty.clone(), span, TermKind::Var(fresh.clone()));
                    let index_lit = self.mk_term(
                        Type::Constructed(TypeName::Int(32), vec![]),
                        span,
                        TermKind::IntLit(field_idx.to_string()),
                    );
                    let field_access =
                        self.build_app2("_w_tuple_proj", record_ref, index_lit, field_ty.clone(), span);

                    if let Some(pat) = &field.pattern {
                        self.collect_pattern_bindings_into(pat, &field_ty, field_access, bindings, span);
                    } else {
                        bindings.push(PendingBinding {
                            name: field.field.clone(),
                            ty: field_ty,
                            expr: field_access,
                        });
                    }
                }
            }

            ast::PatternKind::Unit => {
                todo!("Unit patterns in let bindings")
            }

            ast::PatternKind::Literal(_) => {
                todo!("Literal patterns in let bindings")
            }

            ast::PatternKind::Constructor(_, _) => {
                todo!("Constructor patterns in let bindings")
            }
        }
    }

    fn extract_tuple_types(&self, ty: &Type<TypeName>, _expected_len: usize) -> Vec<Type<TypeName>> {
        match ty {
            Type::Constructed(TypeName::Tuple(_), args) => args.clone(),
            _ => panic!("BUG: Expected tuple type, got {:?}", ty),
        }
    }

    /// Resolve a field name to its index in a record type
    fn resolve_field_index(&self, ty: &Type<TypeName>, field: &str) -> Option<usize> {
        match ty {
            Type::Constructed(TypeName::Record(fields), _) => fields.iter().position(|f| f == field),
            // Vec swizzle: x=0, y=1, z=2, w=3
            Type::Constructed(TypeName::Vec, _) => match field {
                "x" => Some(0),
                "y" => Some(1),
                "z" => Some(2),
                "w" => Some(3),
                _ => None,
            },
            _ => None,
        }
    }

    fn extract_record_types(&self, ty: &Type<TypeName>) -> HashMap<String, Type<TypeName>> {
        match ty {
            Type::Constructed(TypeName::Record(fields), args) => {
                fields.iter().cloned().zip(args.iter().cloned()).collect()
            }
            _ => HashMap::new(),
        }
    }

    fn flatten_tuple_pattern(
        &mut self,
        tuple_var: &str,
        patterns: &[ast::Pattern],
        component_types: &[Type<TypeName>],
        body: Term,
        span: Span,
    ) -> Term {
        let mut result = body;

        for (i, pattern) in patterns.iter().enumerate().rev() {
            let comp_ty = component_types
                .get(i)
                .cloned()
                .expect("BUG: Tuple pattern has more elements than tuple type");

            // proj_i(tuple_var) as function application
            let tuple_ref = self.mk_term(
                Type::Constructed(TypeName::Tuple(patterns.len()), component_types.to_vec()),
                span,
                TermKind::Var(tuple_var.to_string()),
            );
            let index_lit = self.mk_term(
                Type::Constructed(TypeName::Int(32), vec![]),
                span,
                TermKind::IntLit(i.to_string()),
            );
            let proj = self.build_app2("_w_tuple_proj", tuple_ref, index_lit, comp_ty.clone(), span);

            result = self.bind_pattern_to_expr(pattern, comp_ty, proj, result, span);
        }

        result
    }

    fn flatten_record_pattern(
        &mut self,
        record_var: &str,
        record_ty: &Type<TypeName>,
        fields: &[ast::RecordPatternField],
        field_types: &HashMap<String, Type<TypeName>>,
        body: Term,
        span: Span,
    ) -> Term {
        let mut result = body;

        for field in fields.iter().rev() {
            let field_ty = field_types
                .get(&field.field)
                .cloned()
                .unwrap_or_else(|| panic!("BUG: Record field '{}' not found in type", field.field));

            // Resolve field name to index, treat record as tuple
            let field_idx = self
                .resolve_field_index(record_ty, &field.field)
                .unwrap_or_else(|| panic!("BUG: field '{}' not in record type", field.field));

            let record_ref = self.mk_term(record_ty.clone(), span, TermKind::Var(record_var.to_string()));
            let index_lit = self.mk_term(
                Type::Constructed(TypeName::Int(32), vec![]),
                span,
                TermKind::IntLit(field_idx.to_string()),
            );
            let field_access =
                self.build_app2("_w_tuple_proj", record_ref, index_lit, field_ty.clone(), span);

            if let Some(pat) = &field.pattern {
                result = self.bind_pattern_to_expr(pat, field_ty, field_access, result, span);
            } else {
                result = self.mk_term(
                    result.ty.clone(),
                    span,
                    TermKind::Let {
                        name: field.field.clone(),
                        name_ty: field_ty,
                        rhs: Box::new(field_access),
                        body: Box::new(result),
                    },
                );
            }
        }

        result
    }

    fn bind_pattern_to_expr(
        &mut self,
        pattern: &ast::Pattern,
        pat_ty: Type<TypeName>,
        expr: Term,
        body: Term,
        span: Span,
    ) -> Term {
        match &pattern.kind {
            ast::PatternKind::Name(name) => self.mk_term(
                body.ty.clone(),
                span,
                TermKind::Let {
                    name: name.clone(),
                    name_ty: pat_ty,
                    rhs: Box::new(expr),
                    body: Box::new(body),
                },
            ),

            ast::PatternKind::Wildcard => {
                let fresh = format!("_wild_{}", self.term_ids.next_id().0);
                self.mk_term(
                    body.ty.clone(),
                    span,
                    TermKind::Let {
                        name: fresh,
                        name_ty: pat_ty,
                        rhs: Box::new(expr),
                        body: Box::new(body),
                    },
                )
            }

            ast::PatternKind::Typed(inner, _) => self.bind_pattern_to_expr(inner, pat_ty, expr, body, span),

            ast::PatternKind::Attributed(_, inner) => {
                self.bind_pattern_to_expr(inner, pat_ty, expr, body, span)
            }

            ast::PatternKind::Tuple(patterns) => {
                let fresh = format!("_tup_{}", self.term_ids.next_id().0);
                let component_types = self.extract_tuple_types(&pat_ty, patterns.len());
                let inner = self.flatten_tuple_pattern(&fresh, patterns, &component_types, body, span);

                self.mk_term(
                    inner.ty.clone(),
                    span,
                    TermKind::Let {
                        name: fresh,
                        name_ty: pat_ty,
                        rhs: Box::new(expr),
                        body: Box::new(inner),
                    },
                )
            }

            ast::PatternKind::Record(fields) => {
                let fresh = format!("_rec_{}", self.term_ids.next_id().0);
                let field_types = self.extract_record_types(&pat_ty);
                let inner = self.flatten_record_pattern(&fresh, &pat_ty, fields, &field_types, body, span);

                self.mk_term(
                    inner.ty.clone(),
                    span,
                    TermKind::Let {
                        name: fresh,
                        name_ty: pat_ty,
                        rhs: Box::new(expr),
                        body: Box::new(inner),
                    },
                )
            }

            ast::PatternKind::Unit => {
                todo!("Unit patterns in let bindings")
            }

            ast::PatternKind::Literal(_) => {
                todo!("Literal patterns in let bindings")
            }

            ast::PatternKind::Constructor(_, _) => {
                todo!("Constructor patterns in let bindings")
            }
        }
    }

    fn transform_expr(&mut self, expr: &ast::Expression) -> Term {
        let ty = self.lookup_type(expr.h.id).expect("BUG: Expression must have type in type table");
        let span = expr.h.span;

        match &expr.kind {
            ast::ExprKind::IntLiteral(s) => self.mk_term(ty, span, TermKind::IntLit(s.0.clone())),

            ast::ExprKind::FloatLiteral(f) => self.mk_term(ty, span, TermKind::FloatLit(*f)),

            ast::ExprKind::BoolLiteral(b) => self.mk_term(ty, span, TermKind::BoolLit(*b)),

            ast::ExprKind::StringLiteral(s) => self.mk_term(ty, span, TermKind::StringLit(s.clone())),

            ast::ExprKind::Unit => {
                // Unit value represented as _w_unit intrinsic call
                self.build_intrinsic_call("_w_unit", &[], ty, span)
            }

            ast::ExprKind::Identifier(qualifiers, name) => {
                let full_name = if qualifiers.is_empty() {
                    name.clone()
                } else {
                    format!("{}.{}", qualifiers.join("."), name)
                };
                self.mk_term(ty, span, TermKind::Var(full_name))
            }

            ast::ExprKind::ArrayLiteral(elements) => {
                // array_lit(e1, e2, ...) as curried application
                self.build_intrinsic_call("_w_array_lit", elements, ty, span)
            }

            ast::ExprKind::VecMatLiteral(elements) => {
                // For matrices, columns are vectors not arrays
                // Check if result type is Mat and transform columns accordingly
                if let Type::Constructed(TypeName::Mat, args) = &ty {
                    // Mat<rows, cols, elem_ty> - column type is Vec<rows, elem_ty>
                    if args.len() >= 3 {
                        let col_ty =
                            Type::Constructed(TypeName::Vec, vec![args[0].clone(), args[2].clone()]);
                        // Transform elements, treating ArrayLiterals as vectors
                        let col_terms: Vec<Term> =
                            elements.iter().map(|e| self.transform_as_vector(e, col_ty.clone())).collect();
                        return self.build_vec_lit_from_terms(&col_terms, ty, span);
                    }
                }
                self.build_intrinsic_call("_w_vec_lit", elements, ty, span)
            }

            ast::ExprKind::ArrayIndex(array, index) => {
                let arr = self.transform_expr(array);
                let idx = self.transform_expr(index);
                self.build_app2("_w_index", arr, idx, ty, span)
            }

            ast::ExprKind::ArrayWith { array, index, value } => {
                let arr = self.transform_expr(array);
                let idx = self.transform_expr(index);
                let val = self.transform_expr(value);
                self.build_app3("_w_array_with", arr, idx, val, ty, span)
            }

            ast::ExprKind::BinaryOp(op, lhs, rhs) => {
                let l = self.transform_expr(lhs);
                let r = self.transform_expr(rhs);
                self.build_binop(op.clone(), l, r, ty, span)
            }

            ast::ExprKind::UnaryOp(op, operand) => {
                let arg = self.transform_expr(operand);
                self.build_unop(op.clone(), arg, ty, span)
            }

            ast::ExprKind::Tuple(elements) => self.build_intrinsic_call("_w_tuple", elements, ty, span),

            ast::ExprKind::RecordLiteral(fields) => {
                // Records are tuples - reorder fields to match type's field order
                let field_map: HashMap<&str, &ast::Expression> =
                    fields.iter().map(|(name, expr)| (name.as_str(), expr)).collect();

                let ordered_exprs: Vec<ast::Expression> = match &ty {
                    Type::Constructed(TypeName::Record(type_fields), _) => type_fields
                        .iter()
                        .filter_map(|f| field_map.get(f.as_str()).map(|e| (*e).clone()))
                        .collect(),
                    _ => fields.iter().map(|(_, e)| e.clone()).collect(),
                };

                self.build_intrinsic_call("_w_tuple", &ordered_exprs, ty, span)
            }

            ast::ExprKind::Lambda(lam) => self.transform_lambda(&lam.params, &lam.body, ty, span),

            ast::ExprKind::Application(func, args) => self.transform_application(func, args, ty, span),

            ast::ExprKind::LetIn(let_in) => {
                let rhs = self.transform_expr(&let_in.value);
                let body = self.transform_expr(&let_in.body);
                let rhs_ty = rhs.ty.clone();

                self.bind_pattern_to_expr(&let_in.pattern, rhs_ty, rhs, body, span)
            }

            ast::ExprKind::FieldAccess(record, field) => {
                let rec = self.transform_expr(record);
                // Resolve field name to index, treat record as tuple
                let field_idx = self
                    .resolve_field_index(&rec.ty, field)
                    .unwrap_or_else(|| panic!("BUG: field '{}' not in record type", field));
                let index_lit = self.mk_term(
                    Type::Constructed(TypeName::Int(32), vec![]),
                    span,
                    TermKind::IntLit(field_idx.to_string()),
                );
                self.build_app2("_w_tuple_proj", rec, index_lit, ty, span)
            }

            ast::ExprKind::If(if_expr) => {
                let cond = self.transform_expr(&if_expr.condition);
                let then_branch = self.transform_expr(&if_expr.then_branch);
                let else_branch = self.transform_expr(&if_expr.else_branch);
                self.mk_term(
                    ty,
                    span,
                    TermKind::If {
                        cond: Box::new(cond),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    },
                )
            }

            ast::ExprKind::Loop(loop_expr) => self.transform_loop(loop_expr, ty, span),

            ast::ExprKind::Match(match_expr) => self.transform_match(match_expr, ty, span),

            ast::ExprKind::Range(range) => {
                // Transform range to _w_range intrinsic
                let start = self.transform_expr(&range.start);
                let end = self.transform_expr(&range.end);
                let kind_val = match range.kind {
                    ast::RangeKind::Inclusive => 0,
                    ast::RangeKind::Exclusive => 1,
                    ast::RangeKind::ExclusiveLt => 2,
                    ast::RangeKind::ExclusiveGt => 3,
                };
                let kind_lit = self.mk_term(
                    Type::Constructed(TypeName::Int(32), vec![]),
                    span,
                    TermKind::IntLit(kind_val.to_string()),
                );

                match &range.step {
                    Some(step_expr) => {
                        let step = self.transform_expr(step_expr);
                        // _w_range_step start step end kind
                        self.build_app4("_w_range_step", start, step, end, kind_lit, ty, span)
                    }
                    None => {
                        // _w_range start end kind
                        self.build_app3("_w_range", start, end, kind_lit, ty, span)
                    }
                }
            }

            ast::ExprKind::Slice(slice) => {
                // Transform slice to _w_slice(arr, start, end)
                // This represents a view into the array - aliases the source
                let arr = self.transform_expr(&slice.array);

                // Default start to 0 if not specified
                let start = slice
                    .start
                    .as_ref()
                    .map(|e| self.transform_expr(e))
                    .unwrap_or_else(|| self.mk_i32(0, span));

                // End is required for now (would need array length otherwise)
                let end = slice
                    .end
                    .as_ref()
                    .map(|e| self.transform_expr(e))
                    .expect("Slice without end not yet supported");

                self.build_app3("_w_slice", arr, start, end, ty, span)
            }

            ast::ExprKind::TypeAscription(inner, _) => self.transform_expr(inner),

            ast::ExprKind::TypeCoercion(inner, _) => {
                let term = self.transform_expr(inner);
                self.build_app1("_w_coerce", term, ty, span)
            }

            ast::ExprKind::TypeHole => {
                todo!("Type holes")
            }
        }
    }

    fn transform_lambda(
        &mut self,
        params: &[ast::Pattern],
        body: &ast::Expression,
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        if params.is_empty() {
            return self.transform_expr(body);
        }

        // Collect all lambda parameters and their pending bindings
        // This avoids creating let-bindings between nested lambdas which causes captures
        let mut lambda_info: Vec<(String, Type<TypeName>, Vec<PendingBinding>)> = Vec::new();
        let mut current_ty = ty.clone();

        for param in params {
            let param_ty = self.get_param_type(&current_ty);
            let (fresh_name, bindings) = self.collect_pattern_bindings(param, &param_ty, span);
            lambda_info.push((fresh_name, param_ty.clone(), bindings));
            current_ty = self.get_body_type(&current_ty);
        }

        // Transform the body expression
        let mut result = self.transform_expr(body);

        // Apply all bindings in reverse order (innermost first, so outermost ends up innermost)
        for (_, _, bindings) in lambda_info.iter().rev() {
            for binding in bindings.iter().rev() {
                result = self.mk_term(
                    result.ty.clone(),
                    span,
                    TermKind::Let {
                        name: binding.name.clone(),
                        name_ty: binding.ty.clone(),
                        rhs: Box::new(binding.expr.clone()),
                        body: Box::new(result),
                    },
                );
            }
        }

        // Build nested lambdas from inside-out
        for (fresh_name, param_ty, _) in lambda_info.into_iter().rev() {
            let lam_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), result.ty.clone()]);
            result = self.mk_term(
                lam_ty,
                span,
                TermKind::Lam {
                    param: fresh_name,
                    param_ty,
                    body: Box::new(result),
                },
            );
        }

        result
    }

    fn get_param_type(&self, ty: &Type<TypeName>) -> Type<TypeName> {
        match ty {
            Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => args[0].clone(),
            _ => panic!("BUG: Expected arrow type for function param, got {:?}", ty),
        }
    }

    fn get_body_type(&self, ty: &Type<TypeName>) -> Type<TypeName> {
        match ty {
            Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => args[1].clone(),
            _ => ty.clone(),
        }
    }

    fn transform_application(
        &mut self,
        func: &ast::Expression,
        args: &[ast::Expression],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let func_term = self.transform_expr(func);

        if args.is_empty() {
            return func_term;
        }

        // First application
        let first_arg = self.transform_expr(&args[0]);
        let mut result = self.mk_term(
            self.get_body_type(&func_term.ty),
            span,
            TermKind::App {
                func: Box::new(func_term),
                arg: Box::new(first_arg),
            },
        );

        // Subsequent applications chain
        for arg in &args[1..] {
            let arg_term = self.transform_expr(arg);
            let app_ty = self.get_body_type(&result.ty);
            result = self.mk_term(
                app_ty,
                span,
                TermKind::App {
                    func: Box::new(result),
                    arg: Box::new(arg_term),
                },
            );
        }

        Term { ty, ..result }
    }

    fn transform_loop(&mut self, loop_expr: &ast::LoopExpr, ty: Type<TypeName>, span: Span) -> Term {
        // Get the init expression and accumulator type
        let init_term = loop_expr.init.as_ref().map(|e| self.transform_expr(e)).unwrap_or_else(|| {
            // No accumulator - use unit
            self.build_intrinsic_call("_w_unit", &[], Type::Constructed(TypeName::Unit, vec![]), span)
        });
        let acc_ty = init_term.ty.clone();

        // Build loop_var and init_bindings from the pattern
        let (loop_var, loop_var_ty, init_bindings) =
            self.build_loop_var_and_bindings(&loop_expr.pattern, &acc_ty, span);

        // Transform body (pattern bindings are handled via init_bindings)
        let body = self.transform_expr(&loop_expr.body);

        match &loop_expr.form {
            ast::LoopForm::For(idx_var, bound) => {
                let bound_term = self.transform_expr(bound);
                let index_ty = Type::Constructed(TypeName::Int(32), vec![]);

                self.mk_term(
                    ty,
                    span,
                    TermKind::Loop {
                        loop_var,
                        loop_var_ty,
                        init: Box::new(init_term),
                        init_bindings,
                        kind: LoopKind::ForRange {
                            var: idx_var.clone(),
                            var_ty: index_ty,
                            bound: Box::new(bound_term),
                        },
                        body: Box::new(body),
                    },
                )
            }

            ast::LoopForm::ForIn(elem_pattern, iter) => {
                let iter_term = self.transform_expr(iter);
                let elem_ty = self.get_array_element_type(&iter_term.ty);
                let elem_var = elem_pattern.simple_name().unwrap_or("_elem").to_string();

                self.mk_term(
                    ty,
                    span,
                    TermKind::Loop {
                        loop_var,
                        loop_var_ty,
                        init: Box::new(init_term),
                        init_bindings,
                        kind: LoopKind::For {
                            var: elem_var,
                            var_ty: elem_ty,
                            iter: Box::new(iter_term),
                        },
                        body: Box::new(body),
                    },
                )
            }

            ast::LoopForm::While(cond) => {
                let cond_term = self.transform_expr(cond);

                self.mk_term(
                    ty,
                    span,
                    TermKind::Loop {
                        loop_var,
                        loop_var_ty,
                        init: Box::new(init_term),
                        init_bindings,
                        kind: LoopKind::While {
                            cond: Box::new(cond_term),
                        },
                        body: Box::new(body),
                    },
                )
            }
        }
    }

    /// Build loop variable name and init_bindings from a pattern.
    fn build_loop_var_and_bindings(
        &mut self,
        pattern: &ast::Pattern,
        acc_ty: &Type<TypeName>,
        span: Span,
    ) -> (String, Type<TypeName>, Vec<(String, Type<TypeName>, Term)>) {
        use crate::pattern::binding_paths;

        // For a simple name pattern, use it directly
        if let ast::PatternKind::Name(name) = &pattern.kind {
            return (name.clone(), acc_ty.clone(), vec![]);
        }

        // For complex patterns, create a fresh loop_var and build projections
        let loop_var = format!("_loop_{}", self.term_ids.next_id().0);
        let paths = binding_paths(pattern);

        let init_bindings = paths
            .into_iter()
            .filter_map(|bp| {
                if bp.path.is_empty() {
                    // This is the root binding - shouldn't happen for complex patterns
                    None
                } else {
                    let binding_ty = self.type_at_path(acc_ty, &bp.path);
                    let proj_term = self.build_projection_chain(&loop_var, acc_ty, &bp.path, span);
                    Some((bp.name, binding_ty, proj_term))
                }
            })
            .collect();

        (loop_var, acc_ty.clone(), init_bindings)
    }

    /// Get the type at a given projection path within a tuple type.
    fn type_at_path(&self, ty: &Type<TypeName>, path: &[usize]) -> Type<TypeName> {
        let mut current = ty.clone();
        for &idx in path {
            current = match &current {
                Type::Constructed(TypeName::Tuple(_), args) => {
                    args.get(idx).cloned().unwrap_or(current.clone())
                }
                Type::Constructed(TypeName::Record(_), args) => {
                    args.get(idx).cloned().unwrap_or(current.clone())
                }
                _ => current.clone(),
            };
        }
        current
    }

    /// Build a chain of tuple projections: proj[path[n-1]](...proj[path[0]](var))
    fn build_projection_chain(
        &mut self,
        var: &str,
        var_ty: &Type<TypeName>,
        path: &[usize],
        span: Span,
    ) -> Term {
        let mut current_ty = var_ty.clone();
        let mut current = self.mk_term(current_ty.clone(), span, TermKind::Var(var.to_string()));

        for &idx in path {
            let elem_ty = self.type_at_path(&current_ty, &[idx]);
            let index_lit = self.mk_term(
                Type::Constructed(TypeName::Int(32), vec![]),
                span,
                TermKind::IntLit(idx.to_string()),
            );
            current = self.build_app2("_w_tuple_proj", current, index_lit, elem_ty.clone(), span);
            current_ty = elem_ty;
        }

        current
    }

    fn get_array_element_type(&self, ty: &Type<TypeName>) -> Type<TypeName> {
        match ty {
            Type::Constructed(TypeName::Array, args) if !args.is_empty() => args[0].clone(),
            _ => panic!("BUG: Expected array type, got {:?}", ty),
        }
    }

    fn transform_match(&mut self, match_expr: &ast::MatchExpr, ty: Type<TypeName>, span: Span) -> Term {
        let scrutinee = self.transform_expr(&match_expr.scrutinee);

        if match_expr.cases.is_empty() {
            todo!("Empty match")
        }

        self.compile_match_cases(&scrutinee, &match_expr.cases, ty, span)
    }

    fn compile_match_cases(
        &mut self,
        scrutinee: &Term,
        cases: &[ast::MatchCase],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        if cases.is_empty() {
            let fail_fn = self.mk_term(ty.clone(), span, TermKind::Var("_w_match_fail".to_string()));
            return fail_fn;
        }

        let case = &cases[0];
        let rest = &cases[1..];

        match &case.pattern.kind {
            ast::PatternKind::Wildcard | ast::PatternKind::Name(_) => {
                let body = self.transform_expr(&case.body);
                self.bind_pattern_to_expr(
                    &case.pattern,
                    scrutinee.ty.clone(),
                    scrutinee.clone(),
                    body,
                    span,
                )
            }

            ast::PatternKind::Literal(lit) => {
                let lit_term = self.literal_to_term(lit, span);
                let eq_op = ast::BinaryOp { op: "==".to_string() };
                let cond = self.build_binop(
                    eq_op,
                    scrutinee.clone(),
                    lit_term,
                    Type::Constructed(TypeName::Str("bool"), vec![]),
                    span,
                );
                let then_branch = self.transform_expr(&case.body);
                let else_branch = self.compile_match_cases(scrutinee, rest, ty.clone(), span);

                self.mk_term(
                    ty,
                    span,
                    TermKind::If {
                        cond: Box::new(cond),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    },
                )
            }

            ast::PatternKind::Tuple(patterns) => {
                let body = self.transform_expr(&case.body);
                let component_types = self.extract_tuple_types(&scrutinee.ty, patterns.len());

                let fresh = format!("_match_{}", self.term_ids.next_id().0);
                let inner = self.flatten_tuple_pattern(&fresh, patterns, &component_types, body, span);

                self.mk_term(
                    ty,
                    span,
                    TermKind::Let {
                        name: fresh,
                        name_ty: scrutinee.ty.clone(),
                        rhs: Box::new(scrutinee.clone()),
                        body: Box::new(inner),
                    },
                )
            }

            ast::PatternKind::Constructor(ctor_name, patterns) => {
                let is_ctor = self.build_app1(
                    &format!("_w_is_{}", ctor_name),
                    scrutinee.clone(),
                    Type::Constructed(TypeName::Str("bool"), vec![]),
                    span,
                );

                let body = self.transform_expr(&case.body);
                let mut bound_body = body;
                for (i, pat) in patterns.iter().enumerate().rev() {
                    let field_ty =
                        self.lookup_type(pat.h.id).expect("BUG: Constructor field pattern must have type");
                    let extract = self.build_app1(
                        &format!("_w_extract_{}_{}", ctor_name, i),
                        scrutinee.clone(),
                        field_ty.clone(),
                        span,
                    );
                    bound_body = self.bind_pattern_to_expr(pat, field_ty, extract, bound_body, span);
                }

                let else_branch = self.compile_match_cases(scrutinee, rest, ty.clone(), span);

                self.mk_term(
                    ty,
                    span,
                    TermKind::If {
                        cond: Box::new(is_ctor),
                        then_branch: Box::new(bound_body),
                        else_branch: Box::new(else_branch),
                    },
                )
            }

            ast::PatternKind::Typed(inner, _) => {
                let adjusted_case = ast::MatchCase {
                    pattern: (**inner).clone(),
                    body: case.body.clone(),
                };
                let mut adjusted_cases = vec![adjusted_case];
                adjusted_cases.extend(rest.iter().cloned());
                self.compile_match_cases(scrutinee, &adjusted_cases, ty, span)
            }

            ast::PatternKind::Attributed(_, inner) => {
                let adjusted_case = ast::MatchCase {
                    pattern: (**inner).clone(),
                    body: case.body.clone(),
                };
                let mut adjusted_cases = vec![adjusted_case];
                adjusted_cases.extend(rest.iter().cloned());
                self.compile_match_cases(scrutinee, &adjusted_cases, ty, span)
            }

            ast::PatternKind::Record(fields) => {
                let body = self.transform_expr(&case.body);
                let field_types = self.extract_record_types(&scrutinee.ty);

                let fresh = format!("_match_{}", self.term_ids.next_id().0);
                let inner =
                    self.flatten_record_pattern(&fresh, &scrutinee.ty, fields, &field_types, body, span);

                self.mk_term(
                    ty,
                    span,
                    TermKind::Let {
                        name: fresh,
                        name_ty: scrutinee.ty.clone(),
                        rhs: Box::new(scrutinee.clone()),
                        body: Box::new(inner),
                    },
                )
            }

            ast::PatternKind::Unit => {
                todo!("Unit patterns in match")
            }
        }
    }

    fn literal_to_term(&mut self, lit: &ast::PatternLiteral, span: Span) -> Term {
        match lit {
            ast::PatternLiteral::Int(s) => self.mk_term(
                Type::Constructed(TypeName::Int(32), vec![]),
                span,
                TermKind::IntLit(s.0.clone()),
            ),
            ast::PatternLiteral::Float(f) => self.mk_term(
                Type::Constructed(TypeName::Float(32), vec![]),
                span,
                TermKind::FloatLit(*f),
            ),
            ast::PatternLiteral::Bool(b) => self.mk_term(
                Type::Constructed(TypeName::Str("bool"), vec![]),
                span,
                TermKind::BoolLit(*b),
            ),
            ast::PatternLiteral::Char(c) => {
                // Represent char as int for now
                self.mk_term(
                    Type::Constructed(TypeName::Int(32), vec![]),
                    span,
                    TermKind::IntLit((*c as u32).to_string()),
                )
            }
        }
    }

    // Helper: build App(Var(name), arg)
    fn build_app1(&mut self, name: &str, arg: Term, result_ty: Type<TypeName>, span: Span) -> Term {
        // Build the function type for the Var
        let func_ty = Type::Constructed(TypeName::Arrow, vec![arg.ty.clone(), result_ty.clone()]);
        let func_term = self.mk_term(func_ty, span, TermKind::Var(name.to_string()));
        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(func_term),
                arg: Box::new(arg),
            },
        )
    }

    // Helper: build App(App(Var(name), arg1), arg2)
    fn build_app2(
        &mut self,
        name: &str,
        arg1: Term,
        arg2: Term,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let app1_result_ty = Type::Constructed(TypeName::Arrow, vec![arg2.ty.clone(), result_ty.clone()]);
        let func_ty = Type::Constructed(TypeName::Arrow, vec![arg1.ty.clone(), app1_result_ty.clone()]);
        let func_term = self.mk_term(func_ty, span, TermKind::Var(name.to_string()));
        let app1 = self.mk_term(
            app1_result_ty,
            span,
            TermKind::App {
                func: Box::new(func_term),
                arg: Box::new(arg1),
            },
        );
        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(app1),
                arg: Box::new(arg2),
            },
        )
    }

    // Helper: build App(App(App(Var(name), arg1), arg2), arg3)
    fn build_app3(
        &mut self,
        name: &str,
        arg1: Term,
        arg2: Term,
        arg3: Term,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        // Type of app2: arg3.ty -> result_ty
        let app2_ty = Type::Constructed(TypeName::Arrow, vec![arg3.ty.clone(), result_ty.clone()]);
        // Type of app1: arg2.ty -> app2_ty
        let app1_ty = Type::Constructed(TypeName::Arrow, vec![arg2.ty.clone(), app2_ty.clone()]);
        // Type of func: arg1.ty -> app1_ty
        let func_ty = Type::Constructed(TypeName::Arrow, vec![arg1.ty.clone(), app1_ty.clone()]);
        let func_term = self.mk_term(func_ty, span, TermKind::Var(name.to_string()));
        let app1 = self.mk_term(
            app1_ty,
            span,
            TermKind::App {
                func: Box::new(func_term),
                arg: Box::new(arg1),
            },
        );
        let app2 = self.mk_term(
            app2_ty,
            span,
            TermKind::App {
                func: Box::new(app1),
                arg: Box::new(arg2),
            },
        );
        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(app2),
                arg: Box::new(arg3),
            },
        )
    }

    // Helper: build App(App(App(App(Var(name), arg1), arg2), arg3), arg4)
    fn build_app4(
        &mut self,
        name: &str,
        arg1: Term,
        arg2: Term,
        arg3: Term,
        arg4: Term,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let app3_ty = Type::Constructed(TypeName::Arrow, vec![arg4.ty.clone(), result_ty.clone()]);
        let app2_ty = Type::Constructed(TypeName::Arrow, vec![arg3.ty.clone(), app3_ty.clone()]);
        let app1_ty = Type::Constructed(TypeName::Arrow, vec![arg2.ty.clone(), app2_ty.clone()]);
        let func_ty = Type::Constructed(TypeName::Arrow, vec![arg1.ty.clone(), app1_ty.clone()]);
        let func_term = self.mk_term(func_ty, span, TermKind::Var(name.to_string()));
        let app1 = self.mk_term(
            app1_ty,
            span,
            TermKind::App {
                func: Box::new(func_term),
                arg: Box::new(arg1),
            },
        );
        let app2 = self.mk_term(
            app2_ty,
            span,
            TermKind::App {
                func: Box::new(app1),
                arg: Box::new(arg2),
            },
        );
        let app3 = self.mk_term(
            app3_ty,
            span,
            TermKind::App {
                func: Box::new(app2),
                arg: Box::new(arg3),
            },
        );
        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(app3),
                arg: Box::new(arg4),
            },
        )
    }

    // Helper: build binary op application
    fn build_binop(
        &mut self,
        op: ast::BinaryOp,
        lhs: Term,
        rhs: Term,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        // Build the binop type: lhs.ty -> rhs.ty -> result_ty
        let app1_result_ty = Type::Constructed(TypeName::Arrow, vec![rhs.ty.clone(), result_ty.clone()]);
        let binop_ty = Type::Constructed(TypeName::Arrow, vec![lhs.ty.clone(), app1_result_ty.clone()]);
        let binop_term = self.mk_term(binop_ty, span, TermKind::BinOp(op));
        let app1 = self.mk_term(
            app1_result_ty,
            span,
            TermKind::App {
                func: Box::new(binop_term),
                arg: Box::new(lhs),
            },
        );
        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(app1),
                arg: Box::new(rhs),
            },
        )
    }

    // Helper: build unary op application
    fn build_unop(&mut self, op: ast::UnaryOp, arg: Term, result_ty: Type<TypeName>, span: Span) -> Term {
        let unop_ty = Type::Constructed(TypeName::Arrow, vec![arg.ty.clone(), result_ty.clone()]);
        let unop_term = self.mk_term(unop_ty, span, TermKind::UnOp(op));
        self.mk_term(
            result_ty,
            span,
            TermKind::App {
                func: Box::new(unop_term),
                arg: Box::new(arg),
            },
        )
    }

    // Helper: build curried application for variable number of args
    fn build_intrinsic_call(
        &mut self,
        name: &str,
        args: &[ast::Expression],
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        if args.is_empty() {
            return self.mk_term(result_ty, span, TermKind::Var(name.to_string()));
        }

        // Transform all args first to get their types
        let arg_terms: Vec<Term> = args.iter().map(|a| self.transform_expr(a)).collect();

        // Compute intermediate types working backwards from result_ty
        // For f(a, b, c) with result R: f(a) : B -> C -> R, f(a)(b) : C -> R, f(a)(b)(c) : R
        let mut intermediate_types = vec![result_ty.clone()];
        for arg in arg_terms.iter().rev().skip(1) {
            let prev_ty = intermediate_types.last().unwrap().clone();
            let cur_ty = Type::Constructed(TypeName::Arrow, vec![arg.ty.clone(), prev_ty]);
            intermediate_types.push(cur_ty);
        }
        intermediate_types.reverse();

        // Build curried applications
        // Compute the function type
        let func_ty = Type::Constructed(
            TypeName::Arrow,
            vec![arg_terms[0].ty.clone(), intermediate_types[0].clone()],
        );
        let func_term = self.mk_term(func_ty, span, TermKind::Var(name.to_string()));
        let mut result = self.mk_term(
            intermediate_types[0].clone(),
            span,
            TermKind::App {
                func: Box::new(func_term),
                arg: Box::new(arg_terms[0].clone()),
            },
        );

        for (i, arg_term) in arg_terms.iter().enumerate().skip(1) {
            let app_ty = intermediate_types.get(i).cloned().unwrap_or_else(|| result_ty.clone());
            result = self.mk_term(
                app_ty,
                span,
                TermKind::App {
                    func: Box::new(result),
                    arg: Box::new(arg_term.clone()),
                },
            );
        }

        result
    }

    fn lookup_type(&self, node_id: NodeId) -> Option<Type<TypeName>> {
        self.type_table.get(&node_id).map(|scheme| self.extract_monotype(scheme))
    }

    fn extract_monotype(&self, scheme: &polytype::TypeScheme<TypeName>) -> Type<TypeName> {
        match scheme {
            polytype::TypeScheme::Monotype(ty) => ty.clone(),
            polytype::TypeScheme::Polytype { body, .. } => self.extract_monotype(body),
        }
    }

    fn mk_term(&mut self, ty: Type<TypeName>, span: Span, kind: TermKind) -> Term {
        Term {
            id: self.term_ids.next_id(),
            ty,
            span,
            kind,
        }
    }

    fn mk_i32(&mut self, value: i32, span: Span) -> Term {
        self.mk_term(
            Type::Constructed(TypeName::Int(32), vec![]),
            span,
            TermKind::IntLit(value.to_string()),
        )
    }

    /// Transform an expression as a vector, converting ArrayLiteral to _w_vec_lit
    fn transform_as_vector(&mut self, expr: &ast::Expression, vec_ty: Type<TypeName>) -> Term {
        let span = expr.h.span;
        match &expr.kind {
            ast::ExprKind::ArrayLiteral(elements) => {
                // Convert array literal syntax to vector literal
                self.build_intrinsic_call("_w_vec_lit", elements, vec_ty, span)
            }
            _ => {
                // For other expressions, just transform normally
                self.transform_expr(expr)
            }
        }
    }

    /// Build a _w_vec_lit from already-transformed terms
    fn build_vec_lit_from_terms(&mut self, terms: &[Term], result_ty: Type<TypeName>, span: Span) -> Term {
        if terms.is_empty() {
            return self.mk_term(result_ty, span, TermKind::Var("_w_vec_lit".to_string()));
        }

        // Compute intermediate types working backwards from result_ty
        let mut intermediate_types = vec![result_ty.clone()];
        for term in terms.iter().rev().skip(1) {
            let prev_ty = intermediate_types.last().unwrap().clone();
            let cur_ty = Type::Constructed(TypeName::Arrow, vec![term.ty.clone(), prev_ty]);
            intermediate_types.push(cur_ty);
        }
        intermediate_types.reverse();

        // Build curried applications
        let func_ty = Type::Constructed(
            TypeName::Arrow,
            vec![terms[0].ty.clone(), intermediate_types[0].clone()],
        );
        let func_term = self.mk_term(func_ty, span, TermKind::Var("_w_vec_lit".to_string()));
        let mut result = self.mk_term(
            intermediate_types[0].clone(),
            span,
            TermKind::App {
                func: Box::new(func_term),
                arg: Box::new(terms[0].clone()),
            },
        );

        for (i, term) in terms.iter().enumerate().skip(1) {
            let app_ty = intermediate_types.get(i).cloned().unwrap_or_else(|| result_ty.clone());
            result = self.mk_term(
                app_ty,
                span,
                TermKind::App {
                    func: Box::new(result),
                    arg: Box::new(term.clone()),
                },
            );
        }

        result
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Transform an AST program to TLC.
pub fn transform(program: &ast::Program, type_table: &TypeTable) -> Program {
    let mut transformer = Transformer::new(type_table);
    transformer.transform_program(program)
}
