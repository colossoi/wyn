//! Pass to resolve type placeholders into fresh type variables.
//!
//! This pass runs after parsing and before type checking. It walks the AST
//! and replaces `SizePlaceholder` and `AddressPlaceholder` markers with fresh
//! type variables from a polytype Context.
//!
//! The Context is then passed to the type checker so that the same type
//! variables can be unified during type inference.

use crate::ast::{self, Declaration, EntryDecl, Expression, Pattern, PatternKind, Program};
use crate::types::TypeName;
use polytype::{Context, Type};

#[cfg(test)]
#[path = "resolve_placeholders_tests.rs"]
mod tests;

/// Resolver that transforms placeholder types into type variables.
pub struct PlaceholderResolver {
    context: Context<TypeName>,
}

impl PlaceholderResolver {
    /// Create a new resolver with a fresh Context.
    pub fn new() -> Self {
        Self {
            context: Context::default(),
        }
    }

    /// Create a resolver with an existing Context (e.g., from prelude parsing).
    pub fn with_context(context: Context<TypeName>) -> Self {
        Self { context }
    }

    /// Consume the resolver and return the Context for use in type checking.
    pub fn into_context(self) -> Context<TypeName> {
        self.context
    }

    /// Resolve all placeholders in a program.
    pub fn resolve_program(&mut self, program: &mut Program) {
        for decl in &mut program.declarations {
            self.resolve_declaration(decl);
        }
    }

    fn resolve_declaration(&mut self, decl: &mut Declaration) {
        match decl {
            Declaration::Decl(d) => self.resolve_decl(d),
            Declaration::Entry(e) => self.resolve_entry(e),
            Declaration::Storage(s) => {
                s.ty = self.resolve_type(&s.ty);
            }
            Declaration::Uniform(u) => {
                u.ty = self.resolve_type(&u.ty);
            }
            Declaration::Sig(s) => {
                s.ty = self.resolve_type(&s.ty);
            }
            Declaration::TypeBind(_)
            | Declaration::Import(_)
            | Declaration::Module(_)
            | Declaration::ModuleTypeBind(_)
            | Declaration::Open(_) => {
                // No type annotations to resolve
            }
        }
    }

    fn resolve_decl(&mut self, decl: &mut ast::Decl) {
        // Resolve parameter patterns
        for param in &mut decl.params {
            self.resolve_pattern(param);
        }

        // Resolve return type annotation if present
        if let Some(ref mut ty) = decl.ty {
            *ty = self.resolve_type(ty);
        }

        // Resolve types in body expression
        self.resolve_expression(&mut decl.body);
    }

    fn resolve_entry(&mut self, entry: &mut EntryDecl) {
        // Resolve parameter patterns
        for param in &mut entry.params {
            self.resolve_pattern(param);
        }

        // Resolve output types
        for output in &mut entry.outputs {
            output.ty = self.resolve_type(&output.ty);
        }

        // Resolve types in body expression
        self.resolve_expression(&mut entry.body);
    }

    fn resolve_pattern(&mut self, pattern: &mut Pattern) {
        match &mut pattern.kind {
            PatternKind::Typed(inner, ty) => {
                self.resolve_pattern(inner);
                *ty = self.resolve_type(ty);
            }
            PatternKind::Tuple(patterns) => {
                for p in patterns {
                    self.resolve_pattern(p);
                }
            }
            PatternKind::Attributed(_, inner) => {
                self.resolve_pattern(inner);
            }
            PatternKind::Record(fields) => {
                for field in fields {
                    if let Some(ref mut pat) = field.pattern {
                        self.resolve_pattern(pat);
                    }
                }
            }
            PatternKind::Constructor(_, patterns) => {
                for p in patterns {
                    self.resolve_pattern(p);
                }
            }
            PatternKind::Name(_) | PatternKind::Wildcard | PatternKind::Literal(_) | PatternKind::Unit => {
                // No types to resolve
            }
        }
    }

    fn resolve_expression(&mut self, expr: &mut Expression) {
        match &mut expr.kind {
            ast::ExprKind::Lambda(lambda) => {
                for param in &mut lambda.params {
                    self.resolve_pattern(param);
                }
                self.resolve_expression(&mut lambda.body);
            }
            ast::ExprKind::Application(func, args) => {
                self.resolve_expression(func);
                for arg in args {
                    self.resolve_expression(arg);
                }
            }
            ast::ExprKind::LetIn(let_in) => {
                self.resolve_pattern(&mut let_in.pattern);
                if let Some(ref mut ty) = let_in.ty {
                    *ty = self.resolve_type(ty);
                }
                self.resolve_expression(&mut let_in.value);
                self.resolve_expression(&mut let_in.body);
            }
            ast::ExprKind::If(if_expr) => {
                self.resolve_expression(&mut if_expr.condition);
                self.resolve_expression(&mut if_expr.then_branch);
                self.resolve_expression(&mut if_expr.else_branch);
            }
            ast::ExprKind::BinaryOp(_, left, right) => {
                self.resolve_expression(left);
                self.resolve_expression(right);
            }
            ast::ExprKind::UnaryOp(_, operand) => {
                self.resolve_expression(operand);
            }
            ast::ExprKind::ArrayIndex(array, index) => {
                self.resolve_expression(array);
                self.resolve_expression(index);
            }
            ast::ExprKind::ArrayWith { array, index, value } => {
                self.resolve_expression(array);
                self.resolve_expression(index);
                self.resolve_expression(value);
            }
            ast::ExprKind::Slice(slice) => {
                self.resolve_expression(&mut slice.array);
                if let Some(ref mut s) = slice.start {
                    self.resolve_expression(s);
                }
                if let Some(ref mut e) = slice.end {
                    self.resolve_expression(e);
                }
            }
            ast::ExprKind::ArrayLiteral(elements) => {
                for elem in elements {
                    self.resolve_expression(elem);
                }
            }
            ast::ExprKind::VecMatLiteral(elements) => {
                for elem in elements {
                    self.resolve_expression(elem);
                }
            }
            ast::ExprKind::Tuple(elements) => {
                for elem in elements {
                    self.resolve_expression(elem);
                }
            }
            ast::ExprKind::RecordLiteral(fields) => {
                for (_, value) in fields {
                    self.resolve_expression(value);
                }
            }
            ast::ExprKind::FieldAccess(expr, _) => {
                self.resolve_expression(expr);
            }
            ast::ExprKind::Range(range) => {
                self.resolve_expression(&mut range.start);
                if let Some(ref mut step) = range.step {
                    self.resolve_expression(step);
                }
                self.resolve_expression(&mut range.end);
            }
            ast::ExprKind::Match(match_expr) => {
                self.resolve_expression(&mut match_expr.scrutinee);
                for case in &mut match_expr.cases {
                    self.resolve_pattern(&mut case.pattern);
                    self.resolve_expression(&mut case.body);
                }
            }
            ast::ExprKind::TypeAscription(expr, ty) => {
                self.resolve_expression(expr);
                *ty = self.resolve_type(ty);
            }
            ast::ExprKind::TypeCoercion(expr, ty) => {
                self.resolve_expression(expr);
                *ty = self.resolve_type(ty);
            }
            ast::ExprKind::Loop(loop_expr) => {
                self.resolve_pattern(&mut loop_expr.pattern);
                if let Some(ref mut init) = loop_expr.init {
                    self.resolve_expression(init);
                }
                match &mut loop_expr.form {
                    ast::LoopForm::For(_, bound) => {
                        self.resolve_expression(bound);
                    }
                    ast::LoopForm::ForIn(pat, iter) => {
                        self.resolve_pattern(pat);
                        self.resolve_expression(iter);
                    }
                    ast::LoopForm::While(cond) => {
                        self.resolve_expression(cond);
                    }
                }
                self.resolve_expression(&mut loop_expr.body);
            }
            // Leaf expressions with no nested types
            ast::ExprKind::Identifier(_, _)
            | ast::ExprKind::IntLiteral(_)
            | ast::ExprKind::FloatLiteral(_)
            | ast::ExprKind::BoolLiteral(_)
            | ast::ExprKind::StringLiteral(_)
            | ast::ExprKind::Unit
            | ast::ExprKind::TypeHole => {}
        }
    }

    /// Resolve placeholders in a type, replacing them with fresh type variables.
    pub fn resolve_type(&mut self, ty: &Type<TypeName>) -> Type<TypeName> {
        match ty {
            Type::Constructed(TypeName::SizePlaceholder, _) => self.context.new_variable(),
            Type::Constructed(TypeName::AddressPlaceholder, _) => self.context.new_variable(),
            Type::Constructed(name, args) => {
                let new_args: Vec<_> = args.iter().map(|a| self.resolve_type(a)).collect();
                Type::Constructed(name.clone(), new_args)
            }
            Type::Variable(_) => ty.clone(),
        }
    }
}

impl Default for PlaceholderResolver {
    fn default() -> Self {
        Self::new()
    }
}
