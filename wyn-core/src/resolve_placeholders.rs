//! Pass to resolve type placeholders into fresh type variables.
//!
//! This pass runs after parsing and before type checking. It walks the AST
//! and replaces placeholder types with fresh type variables from a polytype Context:
//!
//! - `SizePlaceholder` (unnamed `[]`) → fresh variable for each occurrence
//! - `AddressPlaceholder` → fresh variable for each occurrence
//! - `SizeVar("n")` (named `[n]`) → same variable for all occurrences within a declaration
//! - `UserVar("T")` (named type param) → same variable for all occurrences within a declaration
//!
//! The Context is then passed to the type checker so that the same type
//! variables can be unified during type inference.
//!
//! This pass also builds TypeSchemes for module specs (like f32.sin), so that
//! the type checker doesn't need to re-convert type parameters to variables.

use crate::ast::{self, Declaration, EntryDecl, Expression, Pattern, PatternKind, Program, TypeParam};
use crate::types::TypeName;
use polytype::{Context, Type, TypeScheme};
use std::collections::HashMap;

#[cfg(test)]
#[path = "resolve_placeholders_tests.rs"]
mod tests;

/// Resolver that transforms placeholder types into type variables.
pub struct PlaceholderResolver {
    context: Context<TypeName>,
    /// Bindings for named type parameters within the current declaration scope.
    /// Maps size param names (e.g., "n") and type param names (e.g., "T") to type variables.
    type_param_bindings: HashMap<String, Type<TypeName>>,
    /// Pre-built TypeSchemes for module specs (e.g., "f32.sin" -> its polytype).
    /// Built during resolve_elaborated_modules so the type checker can use them directly.
    spec_schemes: HashMap<String, TypeScheme<TypeName>>,
}

impl PlaceholderResolver {
    /// Create a new resolver with a fresh Context.
    pub fn new() -> Self {
        Self {
            context: Context::default(),
            type_param_bindings: HashMap::new(),
            spec_schemes: HashMap::new(),
        }
    }

    /// Create a resolver with an existing Context (e.g., from prelude parsing).
    pub fn with_context(context: Context<TypeName>) -> Self {
        Self {
            context,
            type_param_bindings: HashMap::new(),
            spec_schemes: HashMap::new(),
        }
    }

    /// Consume the resolver and return the Context for use in type checking.
    pub fn into_context(self) -> Context<TypeName> {
        self.context
    }

    /// Consume the resolver and return both the Context and spec schemes.
    pub fn into_parts(self) -> (Context<TypeName>, HashMap<String, TypeScheme<TypeName>>) {
        (self.context, self.spec_schemes)
    }

    /// Get the pre-built spec schemes (for module functions like f32.sin).
    pub fn spec_schemes(&self) -> &HashMap<String, TypeScheme<TypeName>> {
        &self.spec_schemes
    }

    /// Resolve all placeholders in a program and its dependencies.
    /// This is the main entry point - it handles prelude, modules, and the program.
    pub fn resolve(
        &mut self,
        module_manager: &mut crate::module_manager::ModuleManager,
        program: &mut Program,
    ) {
        self.resolve_prelude(module_manager.prelude_functions_mut());
        self.resolve_elaborated_modules(module_manager.elaborated_modules_mut());
        self.resolve_program(program);
    }

    /// Resolve all placeholders in a program.
    fn resolve_program(&mut self, program: &mut Program) {
        for decl in &mut program.declarations {
            self.resolve_declaration(decl);
        }
    }

    /// Resolve all placeholders in prelude function declarations.
    fn resolve_prelude(&mut self, prelude_functions: &mut indexmap::IndexMap<String, ast::Decl>) {
        for decl in prelude_functions.values_mut() {
            self.resolve_decl(decl);
        }
    }

    /// Resolve all placeholders in elaborated modules (e.g., f32.sum, i32.max).
    /// Also builds TypeSchemes for Spec::Sig items and stores them in spec_schemes.
    fn resolve_elaborated_modules(
        &mut self,
        modules: &mut std::collections::HashMap<String, crate::module_manager::ElaboratedModule>,
    ) {
        for (module_name, module) in modules.iter_mut() {
            for item in &mut module.items {
                self.resolve_elaborated_item(module_name, item);
            }
        }
    }

    fn resolve_elaborated_item(
        &mut self,
        module_name: &str,
        item: &mut crate::module_manager::ElaboratedItem,
    ) {
        use crate::module_manager::ElaboratedItem;
        match item {
            ElaboratedItem::Spec(spec) => self.resolve_spec_and_build_scheme(module_name, spec),
            ElaboratedItem::Decl(decl) => self.resolve_decl(decl),
            ElaboratedItem::TypeAlias(_, ty) => {
                *ty = self.resolve_type(ty);
            }
        }
    }

    /// Resolve a Spec and build its TypeScheme, storing it in spec_schemes.
    fn resolve_spec_and_build_scheme(&mut self, module_name: &str, spec: &mut ast::Spec) {
        use ast::Spec;
        match spec {
            Spec::Sig(name, type_params, ty) => {
                // Collect variable IDs as we create them for type params
                let mut var_ids: Vec<usize> = Vec::new();

                // Set up bindings for type params in the signature
                for param in type_params.iter() {
                    let var = self.context.new_variable();
                    if let Type::Variable(id) = &var {
                        var_ids.push(*id);
                    }
                    match param {
                        TypeParam::Size(param_name) => {
                            self.type_param_bindings.insert(param_name.clone(), var);
                        }
                        TypeParam::Type(param_name) => {
                            self.type_param_bindings.insert(param_name.clone(), var);
                        }
                        _ => {}
                    }
                }

                // Resolve the type (converts SizeVar/UserVar to Variables)
                *ty = self.resolve_type(ty);

                // Build the TypeScheme using the variables we created
                let scheme = Self::build_polytype(ty.clone(), &var_ids);

                // Store the scheme
                let qualified_name = format!("{}.{}", module_name, name);
                self.spec_schemes.insert(qualified_name, scheme);

                self.type_param_bindings.clear();
            }
            Spec::SigOp(name, ty) => {
                *ty = self.resolve_type(ty);
                // SigOp has no type params, so it's a monotype
                let qualified_name = format!("{}.{}", module_name, name);
                self.spec_schemes.insert(qualified_name, TypeScheme::Monotype(ty.clone()));
            }
            Spec::Type(_, _, _, Some(ty)) => {
                *ty = self.resolve_type(ty);
            }
            Spec::Type(_, _, _, None) | Spec::Module(_, _) | Spec::Include(_) => {
                // No types to resolve
            }
        }
    }

    /// Build a TypeScheme by wrapping a type in nested Polytype layers.
    fn build_polytype(ty: Type<TypeName>, var_ids: &[usize]) -> TypeScheme<TypeName> {
        let mut result = TypeScheme::Monotype(ty);
        // Wrap in reverse order so the first param is the outermost quantifier
        for &var_id in var_ids.iter().rev() {
            result = TypeScheme::Polytype {
                variable: var_id,
                body: Box::new(result),
            };
        }
        result
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
            Declaration::Extern(e) => {
                e.ty = self.resolve_type(&e.ty);
            }
        }
    }

    fn resolve_decl(&mut self, decl: &mut ast::Decl) {
        // Set up bindings for named type parameters
        // Each size param (e.g., "n" from [n]) and type param (e.g., "T") gets a fresh variable
        // that is shared across all occurrences within this declaration
        for size_param in &decl.size_params {
            let var = self.context.new_variable();
            self.type_param_bindings.insert(size_param.clone(), var);
        }
        for type_param in &decl.type_params {
            let var = self.context.new_variable();
            self.type_param_bindings.insert(type_param.clone(), var);
        }

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

        // Clear bindings after processing this declaration
        self.type_param_bindings.clear();
    }

    fn resolve_entry(&mut self, entry: &mut EntryDecl) {
        // Set up bindings for named type parameters
        for size_param in &entry.size_params {
            let var = self.context.new_variable();
            self.type_param_bindings.insert(size_param.clone(), var);
        }
        for type_param in &entry.type_params {
            let var = self.context.new_variable();
            self.type_param_bindings.insert(type_param.clone(), var);
        }

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

        // Clear bindings after processing this entry
        self.type_param_bindings.clear();
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
            // Unnamed placeholders get fresh variables each time
            Type::Constructed(TypeName::SizePlaceholder, _) => self.context.new_variable(),
            Type::Constructed(TypeName::AddressPlaceholder, _) => self.context.new_variable(),
            // Named size/type variables use the binding from the current declaration scope
            Type::Constructed(TypeName::SizeVar(name), _) => {
                if let Some(var) = self.type_param_bindings.get(name) {
                    var.clone()
                } else {
                    // No binding found - leave as-is (will be caught by type checker)
                    ty.clone()
                }
            }
            Type::Constructed(TypeName::UserVar(name), _) => {
                if let Some(var) = self.type_param_bindings.get(name) {
                    var.clone()
                } else {
                    // No binding found - leave as-is (will be caught by type checker)
                    ty.clone()
                }
            }
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
