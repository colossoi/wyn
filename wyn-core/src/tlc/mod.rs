//! Typed Lambda Calculus (TLC) representation.
//!
//! A minimal typed lambda calculus IR for SOAC fusion analysis.
//! Lambdas remain as values (not yet defunctionalized).

pub mod defunctionalize;
#[cfg(test)]
mod defunctionalize_tests;
pub mod monomorphize;
pub mod partial_eval;
#[cfg(test)]
mod partial_eval_tests;
pub mod soac_analysis;
#[cfg(test)]
mod soac_analysis_tests;
pub mod specialize;
pub mod to_mir;
#[cfg(test)]
mod to_mir_tests;

use crate::TypeTable;
use crate::ast::{self, NodeId, Span, TypeName};
use polytype::Type;
use std::collections::{HashMap, HashSet};

/// Mapping from user-facing SOAC names to their internal intrinsic names.
/// These are registered as builtins in the type checker and renamed here during TLC transformation.
const FUNDAMENTAL_SOACS: &[(&str, &str)] = &[
    ("map", "_w_intrinsic_map"),
    ("reduce", "_w_intrinsic_reduce"),
    ("scan", "_w_intrinsic_scan"),
    ("filter", "_w_intrinsic_filter"),
    ("scatter", "_w_intrinsic_scatter"),
    ("zip", "_w_intrinsic_zip"),
    ("length", "_w_intrinsic_length"),
    ("replicate", "_w_intrinsic_replicate"),
    ("reduce_by_index", "_w_intrinsic_hist_1d"),
];

// =============================================================================
// Helper functions
// =============================================================================

/// Count the arity of a function type by counting the number of arrow constructors.
/// For `A -> B -> C`, returns 2.
/// For non-function types, returns 0.
fn count_function_arity(ty: &Type<TypeName>) -> usize {
    match ty {
        Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => 1 + count_function_arity(&args[1]),
        _ => 0,
    }
}

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

    /// External function reference (linked SPIR-V).
    /// The string is the linkage name for spirv-link.
    /// The Wyn-visible name comes from the parent Def.
    Extern(String),

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
    /// Track locally bound names to prevent renaming shadowed SOAC identifiers
    bound_names: HashSet<String>,
}

impl<'a> Transformer<'a> {
    pub fn new(type_table: &'a TypeTable) -> Self {
        Self {
            type_table,
            term_ids: TermIdSource::new(),
            namespace: None,
            bound_names: HashSet::new(),
        }
    }

    /// Create a transformer with a namespace prefix for definition names.
    pub fn with_namespace(type_table: &'a TypeTable, namespace: &str) -> Self {
        Self {
            type_table,
            term_ids: TermIdSource::new(),
            namespace: Some(namespace.to_string()),
            bound_names: HashSet::new(),
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
                ast::Declaration::Extern(e) => {
                    // Create a forward declaration for linked SPIR-V function
                    let body = self.mk_term(e.ty.clone(), e.span, TermKind::Extern(e.linkage_name.clone()));
                    // Compute arity by counting arrows in the function type
                    let arity = count_function_arity(&e.ty);
                    defs.push(Def {
                        name: e.name.clone(),
                        ty: e.ty.clone(),
                        body,
                        meta: DefMeta::Function,
                        arity,
                    });
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
        // Clear bound names for each definition to ensure fresh scope
        self.bound_names.clear();
        let body_ty = self.lookup_type(decl.body.h.id)?;
        let full_ty = self.build_function_type(&decl.params, &body_ty);
        let body = self.transform_with_params(&decl.params, &decl.body, full_ty.clone());

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
        // Clear bound names for each entry to ensure fresh scope
        self.bound_names.clear();
        let body_ty = self.lookup_type(entry.body.h.id)?;
        let full_ty = self.build_function_type(&entry.params, &body_ty);
        let body = self.transform_with_params(&entry.params, &entry.body, full_ty.clone());

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

    fn transform_with_params(
        &mut self,
        params: &[ast::Pattern],
        body: &ast::Expression,
        full_ty: Type<TypeName>,
    ) -> Term {
        let span = params.first().map(|p| p.h.span).unwrap_or(body.h.span);
        self.build_lambda_chain(params, body, full_ty, span)
    }

    /// Build a chain of nested lambdas from patterns, deferring all let-bindings
    /// until after all lambdas are created. This ensures no let-bindings appear
    /// between nested lambdas, which is important for consistent capture analysis.
    fn build_lambda_chain(
        &mut self,
        params: &[ast::Pattern],
        body: &ast::Expression,
        full_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        if params.is_empty() {
            return self.transform_expr(body);
        }

        // Collect all lambda parameters and their pending bindings
        let mut lambda_info: Vec<(String, Type<TypeName>, Vec<PendingBinding>)> = Vec::new();
        let mut current_ty = full_ty;

        for param in params {
            let param_ty = self.get_param_type(&current_ty);

            // Use a placeholder scrutinee - we need to call compute_pattern_bindings to get
            // the param name and projection bindings, but the actual lambda param value
            // won't exist until runtime
            let placeholder =
                self.mk_term(param_ty.clone(), span, TermKind::Var("_placeholder".to_string()));
            let (param_name, mut bindings) = self.compute_pattern_bindings(param, placeholder, span);

            // For complex patterns (Tuple/Record), compute_pattern_bindings returns bindings that
            // include the top-level binding (fresh = scrutinee). For lambdas, we don't want this
            // since the lambda param IS the fresh name. Skip the first binding if it matches.
            if !bindings.is_empty() && bindings[0].name == param_name {
                bindings.remove(0);
            }

            lambda_info.push((param_name, param_ty.clone(), bindings));
            current_ty = self.get_body_type(&current_ty);
        }

        // Add all lambda param names and binding names to bound_names before transforming body
        for (param_name, _, bindings) in &lambda_info {
            self.bound_names.insert(param_name.clone());
            for binding in bindings {
                self.bound_names.insert(binding.name.clone());
            }
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
        for (param_name, param_ty, _) in lambda_info.into_iter().rev() {
            let lam_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), result.ty.clone()]);
            result = self.mk_term(
                lam_ty,
                span,
                TermKind::Lam {
                    param: param_name,
                    param_ty,
                    body: Box::new(result),
                },
            );
        }

        result
    }

    /// Compute bindings for a pattern matched against a scrutinee variable.
    /// Returns (bound_name, list_of_pending_bindings).
    ///
    /// The bound_name is either:
    /// - The pattern's name (for simple Name patterns)
    /// - A fresh variable name (for complex patterns like Tuple/Record)
    ///
    /// For Name/Wildcard patterns, no bindings are returned - the caller is responsible
    /// for creating the top-level binding if needed (e.g., for let-in).
    ///
    /// For Tuple/Record patterns, bindings include:
    /// - The top-level binding (fresh_name = scrutinee)
    /// - All projection bindings
    ///
    /// This is the single source of truth for pattern → binding plan transformation.
    fn compute_pattern_bindings(
        &mut self,
        pattern: &ast::Pattern,
        scrutinee: Term,
        span: Span,
    ) -> (String, Vec<PendingBinding>) {
        self.compute_pattern_bindings_inner(pattern, scrutinee, span, true)
    }

    /// Inner implementation that tracks whether we're at the top level.
    /// At top level, Name/Wildcard don't create bindings (caller handles).
    /// Nested Name/Wildcard DO create bindings (needed for tuple component extraction).
    fn compute_pattern_bindings_inner(
        &mut self,
        pattern: &ast::Pattern,
        scrutinee: Term,
        span: Span,
        is_top_level: bool,
    ) -> (String, Vec<PendingBinding>) {
        match &pattern.kind {
            ast::PatternKind::Name(name) => {
                if is_top_level {
                    // Top-level Name: no binding needed, caller will use scrutinee directly
                    // or wrap with Let as appropriate
                    (name.clone(), vec![])
                } else {
                    // Nested Name (e.g., inside tuple): need binding for projection result
                    let binding = PendingBinding {
                        name: name.clone(),
                        ty: scrutinee.ty.clone(),
                        expr: scrutinee,
                    };
                    (name.clone(), vec![binding])
                }
            }

            ast::PatternKind::Wildcard => {
                let fresh = format!("_wild_{}", self.term_ids.next_id().0);
                if is_top_level {
                    // Top-level Wildcard: no binding needed
                    (fresh, vec![])
                } else {
                    // Nested Wildcard: need binding to evaluate projection
                    let binding = PendingBinding {
                        name: fresh.clone(),
                        ty: scrutinee.ty.clone(),
                        expr: scrutinee,
                    };
                    (fresh, vec![binding])
                }
            }

            ast::PatternKind::Typed(inner, _) | ast::PatternKind::Attributed(_, inner) => {
                self.compute_pattern_bindings_inner(inner, scrutinee, span, is_top_level)
            }

            ast::PatternKind::Tuple(patterns) => {
                let fresh = format!("_tup_{}", self.term_ids.next_id().0);
                let tuple_ty = scrutinee.ty.clone();
                let component_types = self.extract_tuple_types(&tuple_ty, patterns.len());

                // First bind the scrutinee to the fresh name
                let mut bindings = vec![PendingBinding {
                    name: fresh.clone(),
                    ty: tuple_ty.clone(),
                    expr: scrutinee,
                }];

                // Then recursively compute bindings for each component (NOT top-level)
                for (i, sub_pattern) in patterns.iter().enumerate() {
                    let comp_ty = component_types
                        .get(i)
                        .cloned()
                        .expect("BUG: Tuple pattern has more elements than tuple type");

                    let proj = self.build_tuple_projection(&fresh, &tuple_ty, i, comp_ty, span);
                    let (_, sub_bindings) =
                        self.compute_pattern_bindings_inner(sub_pattern, proj, span, false);
                    bindings.extend(sub_bindings);
                }

                (fresh, bindings)
            }

            ast::PatternKind::Record(fields) => {
                let fresh = format!("_rec_{}", self.term_ids.next_id().0);
                let record_ty = scrutinee.ty.clone();
                let field_types = self.extract_record_types(&record_ty);

                // First bind the scrutinee to the fresh name
                let mut bindings = vec![PendingBinding {
                    name: fresh.clone(),
                    ty: record_ty.clone(),
                    expr: scrutinee,
                }];

                // Then recursively compute bindings for each field (NOT top-level)
                for field in fields {
                    let field_ty = field_types
                        .get(&field.field)
                        .cloned()
                        .unwrap_or_else(|| panic!("BUG: Record field '{}' not found in type", field.field));

                    let field_idx = self
                        .resolve_field_index(&record_ty, &field.field)
                        .unwrap_or_else(|| panic!("BUG: field '{}' not in record type", field.field));

                    let field_access =
                        self.build_tuple_projection(&fresh, &record_ty, field_idx, field_ty.clone(), span);

                    if let Some(pat) = &field.pattern {
                        let (_, sub_bindings) =
                            self.compute_pattern_bindings_inner(pat, field_access, span, false);
                        bindings.extend(sub_bindings);
                    } else {
                        bindings.push(PendingBinding {
                            name: field.field.clone(),
                            ty: field_ty,
                            expr: field_access,
                        });
                    }
                }

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

    /// Build a tuple projection: _w_tuple_proj(var, index)
    fn build_tuple_projection(
        &mut self,
        var_name: &str,
        var_ty: &Type<TypeName>,
        index: usize,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let var_term = self.mk_term(var_ty.clone(), span, TermKind::Var(var_name.to_string()));
        let index_lit = self.mk_term(
            Type::Constructed(TypeName::Int(32), vec![]),
            span,
            TermKind::IntLit(index.to_string()),
        );
        self.build_app2("_w_tuple_proj", var_term, index_lit, result_ty, span)
    }

    /// Apply a list of bindings around a body term, creating nested let expressions.
    /// Bindings are applied in reverse order so the first binding is outermost.
    fn apply_bindings_around(&mut self, bindings: Vec<PendingBinding>, body: Term, span: Span) -> Term {
        bindings.into_iter().rev().fold(body, |acc, b| {
            self.mk_term(
                acc.ty.clone(),
                span,
                TermKind::Let {
                    name: b.name,
                    name_ty: b.ty,
                    rhs: Box::new(b.expr),
                    body: Box::new(acc),
                },
            )
        })
    }

    /// Returns Some(name) for simple patterns (Name, Wildcard, or wrapped versions),
    /// None for complex patterns that need destructuring.
    fn simple_pattern_name(&mut self, pattern: &ast::Pattern) -> Option<String> {
        match &pattern.kind {
            ast::PatternKind::Name(name) => Some(name.clone()),
            ast::PatternKind::Wildcard => Some(format!("_wild_{}", self.term_ids.next_id().0)),
            ast::PatternKind::Typed(inner, _) | ast::PatternKind::Attributed(_, inner) => {
                self.simple_pattern_name(inner)
            }
            ast::PatternKind::Tuple(_)
            | ast::PatternKind::Record(_)
            | ast::PatternKind::Unit
            | ast::PatternKind::Literal(_)
            | ast::PatternKind::Constructor(_, _) => None,
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

    fn transform_expr(&mut self, expr: &ast::Expression) -> Term {
        let ty = self.lookup_type(expr.h.id).unwrap_or_else(|| {
            panic!(
                "BUG: Expression must have type in type table. NodeId={:?}, kind={:?}, span={:?}",
                expr.h.id, expr.kind, expr.h.span
            )
        });
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
                    // Check if this is a fundamental SOAC not shadowed by a local binding
                    if !self.bound_names.contains(name) {
                        if let Some((_, intrinsic)) = FUNDAMENTAL_SOACS.iter().find(|(s, _)| *s == name) {
                            intrinsic.to_string()
                        } else {
                            name.clone()
                        }
                    } else {
                        name.clone()
                    }
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
                // Check pattern kind to avoid redundant transforms for simple patterns
                let simple_name = match self.simple_pattern_name(&let_in.pattern) {
                    Some(name) => Some(name),
                    None => None,
                };

                if let Some(bound_name) = simple_name {
                    // Simple Name/Wildcard pattern - single Let binding
                    let rhs = self.transform_expr(&let_in.value);
                    // Track bound name before transforming body
                    self.bound_names.insert(bound_name.clone());
                    let body = self.transform_expr(&let_in.body);
                    self.mk_term(
                        body.ty.clone(),
                        span,
                        TermKind::Let {
                            name: bound_name,
                            name_ty: rhs.ty.clone(),
                            rhs: Box::new(rhs),
                            body: Box::new(body),
                        },
                    )
                } else {
                    // Complex pattern - use compute_pattern_bindings
                    let rhs = self.transform_expr(&let_in.value);
                    let (_, bindings) = self.compute_pattern_bindings(&let_in.pattern, rhs, span);
                    // Track all binding names before transforming body
                    for binding in &bindings {
                        self.bound_names.insert(binding.name.clone());
                    }
                    let body = self.transform_expr(&let_in.body);
                    self.apply_bindings_around(bindings, body, span)
                }
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
        self.build_lambda_chain(params, body, ty, span)
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

    /// Get the type at a given projection path within a tuple/record type.
    fn type_at_path(&self, ty: &Type<TypeName>, path: &[usize]) -> Type<TypeName> {
        let mut current = ty.clone();
        for &idx in path {
            current = match &current {
                Type::Constructed(TypeName::Tuple(_), args) => {
                    args.get(idx).cloned().unwrap_or_else(|| {
                        panic!(
                            "BUG: tuple projection index {} out of bounds for {:?}",
                            idx, current
                        )
                    })
                }
                Type::Constructed(TypeName::Record(fields), args) => {
                    args.get(idx).cloned().unwrap_or_else(|| {
                        panic!(
                            "BUG: record projection index {} out of bounds for {:?} (fields: {:?})",
                            idx, current, fields
                        )
                    })
                }
                _ => panic!("BUG: projection on non-tuple/record type: {:?}", current),
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
                // Simple pattern - bind scrutinee to pattern name
                let bound_name = self.simple_pattern_name(&case.pattern).unwrap();
                let body = self.transform_expr(&case.body);
                self.mk_term(
                    ty,
                    span,
                    TermKind::Let {
                        name: bound_name,
                        name_ty: scrutinee.ty.clone(),
                        rhs: Box::new(scrutinee.clone()),
                        body: Box::new(body),
                    },
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

            ast::PatternKind::Tuple(_) | ast::PatternKind::Record(_) => {
                // Complex pattern - use compute_pattern_bindings
                let (_, bindings) = self.compute_pattern_bindings(&case.pattern, scrutinee.clone(), span);
                let body = self.transform_expr(&case.body);
                self.apply_bindings_around(bindings, body, span)
            }

            ast::PatternKind::Constructor(ctor_name, patterns) => {
                let is_ctor = self.build_app1(
                    &format!("_w_is_{}", ctor_name),
                    scrutinee.clone(),
                    Type::Constructed(TypeName::Str("bool"), vec![]),
                    span,
                );

                // Collect all constructor field bindings
                let mut all_bindings = Vec::new();
                for (i, pat) in patterns.iter().enumerate() {
                    let field_ty =
                        self.lookup_type(pat.h.id).expect("BUG: Constructor field pattern must have type");
                    let extract = self.build_app1(
                        &format!("_w_extract_{}_{}", ctor_name, i),
                        scrutinee.clone(),
                        field_ty.clone(),
                        span,
                    );
                    let (_, bindings) = self.compute_pattern_bindings(pat, extract, span);
                    if bindings.is_empty() {
                        // Simple pattern - need to create binding manually
                        let bound_name = self.simple_pattern_name(pat).unwrap();
                        all_bindings.push(PendingBinding {
                            name: bound_name,
                            ty: field_ty,
                            expr: self.build_app1(
                                &format!("_w_extract_{}_{}", ctor_name, i),
                                scrutinee.clone(),
                                self.lookup_type(pat.h.id).unwrap(),
                                span,
                            ),
                        });
                    } else {
                        all_bindings.extend(bindings);
                    }
                }

                let body = self.transform_expr(&case.body);
                let bound_body = self.apply_bindings_around(all_bindings, body, span);
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

            ast::PatternKind::Typed(inner, _) | ast::PatternKind::Attributed(_, inner) => {
                let adjusted_case = ast::MatchCase {
                    pattern: (**inner).clone(),
                    body: case.body.clone(),
                };
                let mut adjusted_cases = vec![adjusted_case];
                adjusted_cases.extend(rest.iter().cloned());
                self.compile_match_cases(scrutinee, &adjusted_cases, ty, span)
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
    /// Build a curried call from a function name and already-transformed argument terms.
    /// For f(a, b, c) with result R: builds f(a) : B -> C -> R, then f(a)(b) : C -> R, then f(a)(b)(c) : R
    fn build_curried_call_terms(
        &mut self,
        func_name: &str,
        args: &[Term],
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        if args.is_empty() {
            return self.mk_term(result_ty, span, TermKind::Var(func_name.to_string()));
        }

        // Compute intermediate types working backwards from result_ty
        // For f(a, b, c) with result R: after f(a) we have B -> C -> R, after f(a)(b) we have C -> R
        let mut intermediate_types = vec![result_ty.clone()];
        for arg in args.iter().rev().skip(1) {
            let prev_ty = intermediate_types.last().unwrap().clone();
            let cur_ty = Type::Constructed(TypeName::Arrow, vec![arg.ty.clone(), prev_ty]);
            intermediate_types.push(cur_ty);
        }
        intermediate_types.reverse();

        // Build curried applications
        let func_ty = Type::Constructed(
            TypeName::Arrow,
            vec![args[0].ty.clone(), intermediate_types[0].clone()],
        );
        let func_term = self.mk_term(func_ty, span, TermKind::Var(func_name.to_string()));
        let mut result = self.mk_term(
            intermediate_types[0].clone(),
            span,
            TermKind::App {
                func: Box::new(func_term),
                arg: Box::new(args[0].clone()),
            },
        );

        for (i, arg) in args.iter().enumerate().skip(1) {
            let app_ty = intermediate_types.get(i).cloned().unwrap_or_else(|| result_ty.clone());
            result = self.mk_term(
                app_ty,
                span,
                TermKind::App {
                    func: Box::new(result),
                    arg: Box::new(arg.clone()),
                },
            );
        }

        result
    }

    fn build_intrinsic_call(
        &mut self,
        name: &str,
        args: &[ast::Expression],
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let arg_terms: Vec<Term> = args.iter().map(|a| self.transform_expr(a)).collect();
        self.build_curried_call_terms(name, &arg_terms, result_ty, span)
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
        self.build_curried_call_terms("_w_vec_lit", terms, result_ty, span)
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
