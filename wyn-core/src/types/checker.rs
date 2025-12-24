use super::{SkolemId, Type, TypeExt, TypeName, TypeScheme};
use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::scope::ScopeStack;
use crate::{bail_module, bail_type_at, err_module, err_type_at, err_undef_at};
use log::debug;
use polytype::Context;
use std::collections::{BTreeSet, HashMap};

// Import type helper functions from parent module
use super::{
    as_arrow, bool_type, f32, function, i32, mat, record, sized_array, string, strip_unique, tuple, unit,
    vec,
};

/// Trait for generating fresh type variables
pub trait TypeVarGenerator {
    fn new_variable(&mut self) -> Type;
}

// Implement TypeVarGenerator for Context
impl TypeVarGenerator for Context<TypeName> {
    fn new_variable(&mut self) -> Type {
        Context::new_variable(self)
    }
}

/// A warning produced during type checking
#[derive(Debug, Clone)]
pub enum TypeWarning {
    /// A type hole was filled with an inferred type
    TypeHoleFilled {
        inferred_type: Type,
        span: Span,
    },
}

impl TypeWarning {
    /// Get the span for this warning
    pub fn span(&self) -> &Span {
        match self {
            TypeWarning::TypeHoleFilled { span, .. } => span,
        }
    }

    /// Format the warning as a display message
    pub fn message(&self, formatter: &dyn Fn(&Type) -> String) -> String {
        match self {
            TypeWarning::TypeHoleFilled { inferred_type, .. } => {
                format!("Hole of type {}", formatter(inferred_type))
            }
        }
    }
}

pub struct TypeChecker<'a> {
    scope_stack: ScopeStack<TypeScheme>, // Store polymorphic types
    context: Context<TypeName>,          // Polytype unification context
    record_field_map: HashMap<(String, String), Type>, // Map (type_name, field_name) -> field_type
    impl_source: crate::impl_source::ImplSource, // Implementation source for code generation
    intrinsics: crate::intrinsics::IntrinsicSource, // Type registry for polymorphic builtins
    module_manager: &'a crate::module_manager::ModuleManager, // Lazy module loading
    type_table: HashMap<crate::ast::NodeId, TypeScheme>, // Maps NodeId to type scheme
    warnings: Vec<TypeWarning>,          // Collected warnings
    type_holes: Vec<(NodeId, Span)>,     // Track type hole locations for warning emission
    arity_map: HashMap<String, usize>,   // function name -> required arity (number of params)
    /// Scope stack for type parameter bindings (e.g., A -> var_1, B -> var_2).
    /// Used to substitute UserVar/SizeVar in nested lambda type annotations.
    /// Follows same scoping rules as value bindings - inner defs can shadow outer type params.
    type_param_scope: ScopeStack<Type>,
    /// ID source for generating unique skolem constants when opening existential types.
    skolem_ids: crate::IdSource<SkolemId>,
    /// Current module context for resolving unqualified type aliases in expressions.
    /// Set during check_decl_as_in_module for module function checking.
    current_module: Option<String>,
}

/// Compute free type variables in a Type
fn fv_type(ty: &Type) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    fn go(t: &Type, acc: &mut BTreeSet<usize>) {
        match t {
            Type::Variable(n) => {
                acc.insert(*n);
            }
            Type::Constructed(_, args) => {
                for a in args {
                    go(a, acc);
                }
            }
        }
    }
    go(ty, &mut out);
    out
}

/// Compute free type variables in a TypeScheme
fn fv_scheme(s: &TypeScheme) -> BTreeSet<usize> {
    match s {
        TypeScheme::Monotype(t) => fv_type(t),
        TypeScheme::Polytype { variable, body } => {
            let mut set = fv_scheme(body);
            set.remove(variable);
            set
        }
    }
}

/// Wrap a TypeScheme in nested Polytype quantifiers for the given variables
fn quantify(mut body: TypeScheme, vars: &BTreeSet<usize>) -> TypeScheme {
    // Quantify in descending order so the smallest variable ends up outermost
    for v in vars.iter().rev() {
        body = TypeScheme::Polytype {
            variable: *v,
            body: Box::new(body),
        };
    }
    body
}

/// Represents candidate function types for a callee expression.
/// For overloaded intrinsics, there may be multiple candidates.
struct CalleeCandidates {
    candidates: Vec<Candidate>,
    display_name: String,
}

/// A single candidate function type.
struct Candidate {
    /// Fresh instantiated monotype to unify against
    ty: Type,
    /// Original scheme (for storing in type_table)
    scheme: Option<TypeScheme>,
}

impl<'a> TypeChecker<'a> {
    /// Try to extract a constant integer value from an expression.
    /// Returns None if the expression is not a constant.
    fn try_extract_const_int(expr: &Expression) -> Option<i32> {
        match &expr.kind {
            ExprKind::IntLiteral(n) => Some(*n),
            _ => None,
        }
    }

    /// Pure structural equality without applying substitution.
    /// Used internally after types have already been resolved.
    fn types_equal_structural(left: &Type, right: &Type) -> bool {
        match (left, right) {
            (Type::Constructed(l_name, l_args), Type::Constructed(r_name, r_args)) => {
                l_name == r_name
                    && l_args.len() == r_args.len()
                    && l_args.iter().zip(r_args.iter()).all(|(l, r)| Self::types_equal_structural(l, r))
            }
            (Type::Variable(l_id), Type::Variable(r_id)) => l_id == r_id,
            _ => false,
        }
    }

    /// Format a type for error messages by applying current substitution
    pub fn format_type(&self, ty: &Type) -> String {
        let applied = ty.apply(&self.context);
        match &applied {
            Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => {
                // Special case for arrow types
                format!("{} -> {}", self.format_type(&args[0]), self.format_type(&args[1]))
            }
            Type::Constructed(TypeName::Tuple(_), args) => {
                // Special case for tuple types
                let arg_strs: Vec<String> = args.iter().map(|a| self.format_type(a)).collect();
                format!("({})", arg_strs.join(", "))
            }
            Type::Constructed(TypeName::Str(s), args) if *s == "Array" && args.len() == 2 => {
                // Special case for array types [size]elem
                format!("[{}]{}", self.format_type(&args[0]), self.format_type(&args[1]))
            }
            Type::Constructed(name, args) if args.is_empty() => format!("{}", name),
            Type::Constructed(name, args) => {
                let arg_strs: Vec<String> = args.iter().map(|a| self.format_type(a)).collect();
                format!("{}[{}]", name, arg_strs.join(", "))
            }
            Type::Variable(id) => format!("?{}", id),
        }
    }

    /// Format a type scheme for error messages
    pub fn format_scheme(&self, scheme: &TypeScheme) -> String {
        match scheme {
            TypeScheme::Monotype(ty) => self.format_type(ty),
            TypeScheme::Polytype { variable, body } => {
                // For display, we can show quantified vars or just the body
                // For now, just show the body
                format!("∀{}. {}", variable, self.format_scheme(body))
            }
        }
    }

    /// Check if a type contains an existential quantifier.
    /// Existential types are only valid in function return types, not parameters.
    fn contains_existential(ty: &Type) -> bool {
        match ty {
            Type::Constructed(TypeName::Existential(_), _) => true,
            Type::Constructed(_, args) => args.iter().any(Self::contains_existential),
            Type::Variable(_) => false,
        }
    }

    /// Check if a type is numeric (Int, UInt, or Float).
    /// Returns None if the type is a variable (not yet resolved).
    fn is_numeric_type(ty: &Type) -> Option<bool> {
        match ty {
            Type::Constructed(TypeName::Int(_), _) => Some(true),
            Type::Constructed(TypeName::UInt(_), _) => Some(true),
            Type::Constructed(TypeName::Float(_), _) => Some(true),
            Type::Variable(_) => None,
            _ => Some(false),
        }
    }

    /// Open an existential type: ?k. T becomes T[#n/k] where #n is a fresh skolem.
    /// This is called when binding an existential value to a variable.
    /// Skolems are rigid constants that only unify with themselves, enforcing opacity.
    fn open_existential(&mut self, ty: Type) -> Type {
        match ty {
            Type::Constructed(TypeName::Existential(vars), args) if !args.is_empty() => {
                let mut inner = args.into_iter().next().unwrap();
                // Substitute each bound size variable with a fresh skolem constant
                for var_name in vars {
                    let skolem_id = self.skolem_ids.next_id();
                    let skolem = Type::Constructed(TypeName::Skolem(skolem_id), vec![]);
                    inner = Self::substitute_size_var(&inner, &var_name, &skolem);
                }
                inner
            }
            _ => ty,
        }
    }

    /// Substitute a size variable name with a type in a type expression.
    fn substitute_size_var(ty: &Type, var_name: &str, replacement: &Type) -> Type {
        match ty {
            Type::Variable(v) => Type::Variable(*v),
            Type::Constructed(TypeName::SizeVar(name), _) if name == var_name => replacement.clone(),
            Type::Constructed(name, args) => {
                let new_args: Vec<Type> =
                    args.iter().map(|arg| Self::substitute_size_var(arg, var_name, replacement)).collect();
                Type::Constructed(name.clone(), new_args)
            }
        }
    }

    /// Resolve type aliases in a type with optional module context.
    /// - Qualified names (containing '.') are looked up as-is
    /// - Unqualified names are qualified with current_module if provided
    /// - Recursively resolves nested aliases
    fn resolve_type_aliases_scoped(&self, ty: &Type, current_module: Option<&str>) -> Type {
        let mut visited = Vec::new();
        self.resolve_type_aliases_impl(ty, current_module, &mut visited).unwrap_or_else(|cycle_err| {
            // Return the original type on cycle - error will be caught elsewhere
            // or we could log it. For now, just return unresolved.
            log::error!("{}", cycle_err);
            ty.clone()
        })
    }

    fn resolve_type_aliases_impl(
        &self,
        ty: &Type,
        current_module: Option<&str>,
        visited: &mut Vec<String>,
    ) -> Result<Type> {
        match ty {
            Type::Constructed(TypeName::Named(name), args) => {
                // First resolve args recursively
                let resolved_args: Result<Vec<Type>> = args
                    .iter()
                    .map(|a| self.resolve_type_aliases_impl(a, current_module, visited))
                    .collect();
                let resolved_args = resolved_args?;

                // Build lookup keys based on qualification
                let mut keys = Vec::new();
                if name.contains('.') {
                    // Already qualified - try as-is
                    keys.push(name.clone());
                } else if let Some(m) = current_module {
                    // Unqualified in module context - try qualified
                    keys.push(format!("{}.{}", m, name));
                }
                // No prelude fallback - unqualified without context stays unresolved

                for key in keys {
                    // Check for cycles before resolving
                    if let Some(cycle_err) = Self::check_alias_cycle(visited, &key) {
                        return Err(cycle_err);
                    }

                    if let Some(underlying) = self.module_manager.resolve_type_alias(&key) {
                        // Track this alias as visited
                        visited.push(key);
                        // Recursively resolve in same module context
                        let result = self.resolve_type_aliases_impl(underlying, current_module, visited);
                        visited.pop();
                        return result;
                    }
                }

                // Not an alias - keep as-is with resolved args
                Ok(Type::Constructed(TypeName::Named(name.clone()), resolved_args))
            }
            Type::Constructed(name, args) => {
                // Non-Named constructor - just resolve args
                let resolved_args: Result<Vec<Type>> = args
                    .iter()
                    .map(|a| self.resolve_type_aliases_impl(a, current_module, visited))
                    .collect();
                Ok(Type::Constructed(name.clone(), resolved_args?))
            }
            Type::Variable(id) => Ok(Type::Variable(*id)),
        }
    }

    /// Check if adding `key` to visited would create a cycle.
    /// Returns an error if a cycle is detected.
    fn check_alias_cycle(visited: &[String], key: &str) -> Option<CompilerError> {
        if visited.contains(&key.to_string()) {
            let mut cycle_path = visited.to_vec();
            cycle_path.push(key.to_string());
            Some(err_type_at!(
                Span::new(0, 0, 0, 0),
                "type alias cycle detected: {}",
                cycle_path.join(" -> ")
            ))
        } else {
            None
        }
    }

    /// Look up a variable in the scope stack (for testing)
    pub fn lookup(&self, name: &str) -> Option<TypeScheme> {
        self.scope_stack.lookup(name).cloned()
    }

    /// Get a reference to the context (for testing)
    pub fn context(&self) -> &Context<TypeName> {
        &self.context
    }

    /// Compute all free type variables in the current environment (scope stack)
    fn env_free_type_vars(&self) -> BTreeSet<usize> {
        let mut acc = BTreeSet::new();
        self.scope_stack.for_each_binding(|_name, sch| {
            acc.extend(fv_scheme(sch));
        });
        acc
    }

    /// HM-style generalization at let: ∀(fv(ty) \ fv(env) \ ascription_vars). ty
    /// Quantifies over type variables that are free in ty but not free in the environment
    /// and not in the set of ascription variables (which must remain monomorphic)
    fn generalize(&self, ty: &Type) -> TypeScheme {
        // Always generalize the *solved* view
        let applied = ty.apply(&self.context);

        // Assert no unsubstituted UserVar/SizeVar remain - these should have been
        // substituted with type Variables before generalization
        debug_assert!(
            !Self::contains_user_or_size_var(&applied),
            "Type contains unsubstituted UserVar/SizeVar before generalization: {:?}",
            applied
        );

        // Free vars in type
        let mut fv_ty = fv_type(&applied);

        // Free vars in environment
        let fv_env = self.env_free_type_vars();

        // vars to quantify = fv(ty) \ fv(env)
        for v in fv_env {
            fv_ty.remove(&v);
        }

        // Wrap in nested Polytype quantifiers
        quantify(TypeScheme::Monotype(applied), &fv_ty)
    }

    /// Check if a type contains any unsubstituted UserVar or SizeVar.
    /// These should be substituted with proper type Variables before generalization.
    /// Skips checking inside Existential types, as those intentionally contain SizeVar
    /// for existentially quantified sizes.
    fn contains_user_or_size_var(ty: &Type) -> bool {
        match ty {
            Type::Constructed(TypeName::UserVar(_), _) => true,
            Type::Constructed(TypeName::SizeVar(_), _) => true,
            // Skip checking inside Existential - those intentionally use SizeVar
            Type::Constructed(TypeName::Existential(_), _) => false,
            Type::Constructed(_, args) => args.iter().any(Self::contains_user_or_size_var),
            Type::Variable(_) => false,
        }
    }

    /// Try to extract a qualified name from a FieldAccess expression chain
    /// Returns Some(QualName) if the expression is a chain of Identifier + FieldAccess
    /// E.g., M.N.x -> QualName { qualifiers: ["M", "N"], name: "x" }
    ///       f32.cos -> QualName { qualifiers: ["f32"], name: "cos" }
    fn try_extract_qual_name(expr: &Expression, final_field: &str) -> Option<crate::ast::QualName> {
        let mut qualifiers = Vec::new();
        let mut current = expr;

        // Walk up the FieldAccess chain collecting qualifiers
        loop {
            match &current.kind {
                ExprKind::Identifier(quals, name) if quals.is_empty() => {
                    // Base case: found the root identifier
                    qualifiers.push(name.clone());
                    qualifiers.reverse();
                    return Some(crate::ast::QualName::new(qualifiers, final_field.to_string()));
                }
                ExprKind::FieldAccess(base, field) => {
                    // Intermediate field access - this is a qualifier
                    qualifiers.push(field.clone());
                    current = base;
                }
                _ => {
                    // Not a simple qualified name chain (e.g., function call, literal, etc.)
                    return None;
                }
            }
        }
    }

    /// Create a new TypeChecker with a reference to a ModuleManager
    pub fn new(module_manager: &'a crate::module_manager::ModuleManager) -> Self {
        Self::with_type_table(module_manager, HashMap::new())
    }

    /// Create a TypeChecker with an empty type table (for building prelude)
    pub fn new_empty(module_manager: &'a crate::module_manager::ModuleManager) -> Self {
        Self::with_type_table(module_manager, HashMap::new())
    }

    /// Create a TypeChecker with a given initial type table
    fn with_type_table(
        module_manager: &'a crate::module_manager::ModuleManager,
        type_table: HashMap<NodeId, TypeScheme>,
    ) -> Self {
        let mut context = Context::default();
        let impl_source = crate::impl_source::ImplSource::new();
        let poly_builtins = crate::intrinsics::IntrinsicSource::new(&mut context);

        TypeChecker {
            scope_stack: ScopeStack::new(),
            context,
            record_field_map: HashMap::new(),
            impl_source,
            intrinsics: poly_builtins,
            module_manager,
            type_table,
            warnings: Vec::new(),
            type_holes: Vec::new(),
            arity_map: HashMap::new(),
            type_param_scope: ScopeStack::new(),
            skolem_ids: crate::IdSource::new(),
            current_module: None,
        }
    }

    /// Get all warnings collected during type checking
    pub fn warnings(&self) -> &[TypeWarning] {
        &self.warnings
    }

    /// Create a fresh type for a pattern based on its structure
    /// For tuple patterns, creates a tuple of fresh type variables
    /// For simple patterns, creates a single fresh type variable
    fn fresh_type_for_pattern(&mut self, pattern: &Pattern) -> Type {
        match &pattern.kind {
            PatternKind::Tuple(patterns) => {
                // Create a tuple type with fresh type variable for each element
                let elem_types: Vec<Type> =
                    patterns.iter().map(|p| self.fresh_type_for_pattern(p)).collect();
                tuple(elem_types)
            }
            PatternKind::Typed(_, annotated_type) => {
                // Pattern has explicit type, resolve any module type aliases
                // Uses current_module for context (e.g., lambda params inside module functions)
                self.resolve_type_aliases_scoped(annotated_type, self.current_module.as_deref())
            }
            PatternKind::Attributed(_, inner_pattern) => {
                // Ignore attributes, recurse on inner pattern
                self.fresh_type_for_pattern(inner_pattern)
            }
            _ => {
                // For simple patterns (Name, Wildcard, etc.), create a fresh type variable
                self.context.new_variable()
            }
        }
    }

    /// Bind a pattern with a given type, adding bindings to the current scope
    /// Returns the actual type that the pattern matches (for type checking)
    /// If generalize is true, generalizes types for polymorphism (used in let bindings)
    fn bind_pattern(&mut self, pattern: &Pattern, expected_type: &Type, generalize: bool) -> Result<Type> {
        match &pattern.kind {
            PatternKind::Name(name) => {
                // Simple name binding
                let type_scheme = if generalize {
                    self.generalize(expected_type)
                } else {
                    TypeScheme::Monotype(expected_type.clone())
                };
                self.scope_stack.insert(name.clone(), type_scheme);
                // Store resolved type in type_table for mirize
                self.type_table.insert(
                    pattern.h.id,
                    TypeScheme::Monotype(expected_type.apply(&self.context)),
                );
                Ok(expected_type.clone())
            }
            PatternKind::Wildcard => {
                // Wildcard doesn't bind anything
                self.type_table.insert(
                    pattern.h.id,
                    TypeScheme::Monotype(expected_type.apply(&self.context)),
                );
                Ok(expected_type.clone())
            }
            PatternKind::Tuple(patterns) => {
                // Expected type should be a tuple with matching arity
                let expected_applied = expected_type.apply(&self.context);

                match expected_applied {
                    Type::Constructed(TypeName::Tuple(_), ref elem_types) => {
                        if elem_types.len() != patterns.len() {
                            bail_type_at!(
                                pattern.h.span,
                                "Tuple pattern has {} elements but type has {}",
                                patterns.len(),
                                elem_types.len()
                            );
                        }

                        // Bind each sub-pattern with its corresponding element type
                        for (sub_pattern, elem_type) in patterns.iter().zip(elem_types.iter()) {
                            self.bind_pattern(sub_pattern, elem_type, generalize)?;
                        }

                        self.type_table.insert(
                            pattern.h.id,
                            TypeScheme::Monotype(expected_type.apply(&self.context)),
                        );
                        Ok(expected_type.clone())
                    }
                    _ => Err(err_type_at!(
                        pattern.h.span,
                        "Expected tuple type for tuple pattern, got {}",
                        self.format_type(&expected_applied)
                    )),
                }
            }
            PatternKind::Typed(inner_pattern, annotated_type) => {
                // Resolve module type aliases in the annotation (e.g., rand.state -> f32)
                // Uses current_module for context (e.g., lambda params inside module functions)
                let resolved_annotation =
                    self.resolve_type_aliases_scoped(annotated_type, self.current_module.as_deref());
                // Substitute any UserVar/SizeVar from enclosing function's type parameters
                let substituted_annotation = self.substitute_from_type_param_scope(&resolved_annotation);
                // Pattern has a type annotation - unify with expected type
                self.context.unify(&substituted_annotation, expected_type).map_err(|_| {
                    err_type_at!(
                        pattern.h.span,
                        "Pattern type annotation {} doesn't match expected type {}",
                        self.format_type(&substituted_annotation),
                        self.format_type(expected_type)
                    )
                })?;
                // Bind the inner pattern with the substituted type
                let result = self.bind_pattern(inner_pattern, &substituted_annotation, generalize)?;
                // Also store type for the outer Typed pattern
                let resolved = substituted_annotation.apply(&self.context);
                self.type_table.insert(pattern.h.id, TypeScheme::Monotype(resolved));
                Ok(result)
            }
            PatternKind::Attributed(_, inner_pattern) => {
                // Ignore attributes, bind the inner pattern
                let result = self.bind_pattern(inner_pattern, expected_type, generalize)?;
                // Also store type for the outer Attributed pattern
                self.type_table.insert(
                    pattern.h.id,
                    TypeScheme::Monotype(expected_type.apply(&self.context)),
                );
                Ok(result)
            }
            PatternKind::Unit => {
                // Unit pattern should match unit type
                let unit_type = tuple(vec![]);
                self.context.unify(&unit_type, expected_type).map_err(|_| {
                    err_type_at!(
                        pattern.h.span,
                        "Unit pattern doesn't match expected type {}",
                        self.format_type(expected_type)
                    )
                })?;
                self.type_table.insert(pattern.h.id, TypeScheme::Monotype(unit_type.apply(&self.context)));
                Ok(unit_type)
            }
            _ => {
                // Other patterns not yet supported in lambda parameters
                Err(err_type_at!(
                    pattern.h.span,
                    "Pattern {:?} not yet supported in lambda parameters",
                    pattern.kind
                ))
            }
        }
    }

    /// Try to unify an overload's function type with the given argument types
    /// Returns the return type if successful, None if unification fails
    fn try_unify_overload(
        func_type: &Type,
        arg_types: &[Type],
        ctx: &mut Context<TypeName>,
    ) -> Option<Type> {
        let mut current_type = func_type.clone();

        for arg_type in arg_types {
            // Decompose the function type: should be param_ty -> rest
            let param_ty = ctx.new_variable();
            let rest_ty = ctx.new_variable();
            let expected_arrow = Type::arrow(param_ty.clone(), rest_ty.clone());

            // Unify current function type with the expected arrow type
            if ctx.unify(&current_type, &expected_arrow).is_err() {
                return None;
            }

            // Unify the parameter type with the argument type
            let param_ty = param_ty.apply(ctx);
            if ctx.unify(&param_ty, arg_type).is_err() {
                return None;
            }

            // Continue with the rest of the function type
            current_type = rest_ty.apply(ctx);
        }

        // After processing all arguments, current_type should be the return type
        Some(current_type)
    }

    /// Check an expression against an expected type (bidirectional checking mode)
    /// Returns the actual type (which should unify with expected_type)
    fn check_expression(&mut self, expr: &Expression, expected_type: &Type) -> Result<Type> {
        match &expr.kind {
            // Integer literals can be checked against any integer type
            ExprKind::IntLiteral(_) => {
                // Accept the expected type if it's an integer type
                let applied = expected_type.apply(&self.context);
                match &applied {
                    Type::Constructed(TypeName::Int(_), _) | Type::Constructed(TypeName::UInt(_), _) => {
                        // Store the type in the type table
                        self.type_table.insert(expr.h.id, TypeScheme::Monotype(expected_type.clone()));
                        Ok(expected_type.clone())
                    }
                    _ => {
                        // Fall back to inference (will produce i32)
                        let inferred = self.infer_expression(expr)?;
                        self.context.unify(&inferred, expected_type).map_err(|_| {
                            err_type_at!(
                                expr.h.span,
                                "Expected {}, got {}",
                                self.format_type(expected_type),
                                self.format_type(&inferred)
                            )
                        })?;
                        Ok(inferred)
                    }
                }
            }
            // Float literals can be checked against any float type
            ExprKind::FloatLiteral(_) => {
                let applied = expected_type.apply(&self.context);
                match &applied {
                    Type::Constructed(TypeName::Float(_), _) => {
                        // Store the type in the type table
                        self.type_table.insert(expr.h.id, TypeScheme::Monotype(expected_type.clone()));
                        Ok(expected_type.clone())
                    }
                    _ => {
                        let inferred = self.infer_expression(expr)?;
                        self.context.unify(&inferred, expected_type).map_err(|_| {
                            err_type_at!(
                                expr.h.span,
                                "Expected {}, got {}",
                                self.format_type(expected_type),
                                self.format_type(&inferred)
                            )
                        })?;
                        Ok(inferred)
                    }
                }
            }
            ExprKind::Lambda(lambda) => self.type_lambda(lambda, Some(expected_type), expr),
            _ => {
                // For non-lambdas, infer and unify with expected
                let actual_type = self.infer_expression(expr)?;
                self.context.unify(&actual_type, expected_type).map_err(|_| {
                    err_type_at!(
                        expr.h.span,
                        "Type mismatch: expected {}, got {}",
                        self.format_type(expected_type),
                        self.format_type(&actual_type)
                    )
                })?;
                Ok(actual_type)
            }
        }
    }

    /// Substitute UserVars and SizeVars with bound type variables (recursive helper)
    /// Generic substitution: walk the type tree and replace any TypeName for which `f` returns Some.
    fn substitute_named_vars<F>(ty: &Type, f: &F) -> Type
    where
        F: Fn(&TypeName) -> Option<Type>,
    {
        match ty {
            Type::Constructed(name, args) => {
                if let Some(replacement) = f(name) {
                    replacement
                } else {
                    let new_args: Vec<Type> =
                        args.iter().map(|arg| Self::substitute_named_vars(arg, f)).collect();
                    Type::Constructed(name.clone(), new_args)
                }
            }
            Type::Variable(_) => ty.clone(),
        }
    }

    /// Substitute UserVar/SizeVar using the current type parameter scope stack.
    fn substitute_from_type_param_scope(&self, ty: &Type) -> Type {
        Self::substitute_named_vars(ty, &|tn| match tn {
            TypeName::UserVar(name) | TypeName::SizeVar(name) => {
                self.type_param_scope.lookup(name).cloned()
            }
            _ => None,
        })
    }

    fn substitute_type_params_static(ty: &Type, bindings: &HashMap<String, Type>) -> Type {
        Self::substitute_named_vars(ty, &|tn| match tn {
            TypeName::UserVar(name) | TypeName::SizeVar(name) => bindings.get(name).cloned(),
            _ => None,
        })
    }

    /// Allocate a fresh type variable and return its ID.
    fn fresh_var(&mut self) -> polytype::Variable {
        match self.context.new_variable() {
            Type::Variable(id) => id,
            _ => unreachable!(),
        }
    }

    /// Convert a variable ID to a Type.
    fn var(id: polytype::Variable) -> Type {
        Type::Variable(id)
    }

    /// Build a chain of arrows: p1 -> p2 -> ... -> ret
    fn arrow_chain(params: &[Type], ret: Type) -> Type {
        params.iter().rev().fold(ret, |acc, param| Type::arrow(param.clone(), acc))
    }

    /// Wrap a type in nested ∀ quantifiers.
    fn forall(ids: &[polytype::Variable], body: Type) -> TypeScheme {
        ids.iter().rev().fold(TypeScheme::Monotype(body), |acc, &id| TypeScheme::Polytype {
            variable: id,
            body: Box::new(acc),
        })
    }

    /// Build an array type: [n]elem
    fn array_ty(n: polytype::Variable, elem: Type) -> Type {
        Type::Constructed(TypeName::Array, vec![Self::var(n), elem])
    }

    /// Build a Vec type: Vec(n, elem)
    fn vec_ty(n: polytype::Variable, elem: Type) -> Type {
        Type::Constructed(TypeName::Vec, vec![Self::var(n), elem])
    }

    /// Type a lambda expression, optionally with an expected type for bidirectional checking.
    fn type_lambda(
        &mut self,
        lambda: &crate::ast::LambdaExpr,
        expected: Option<&Type>,
        expr: &crate::ast::Expression,
    ) -> Result<Type> {
        // Extract expected parameter types if we have an expected function type
        let expected_param_types: Option<Vec<Type>> = expected.and_then(|exp| {
            let mut result = Vec::new();
            let mut ty = exp.clone();
            for _ in 0..lambda.params.len() {
                let applied = ty.apply(&self.context);
                if let Some((param_type, result_type)) = as_arrow(&applied) {
                    result.push(param_type.clone());
                    ty = result_type.clone();
                } else {
                    return None; // Expected type doesn't match lambda structure
                }
            }
            Some(result)
        });

        self.scope_stack.push_scope();

        // Determine parameter types and bind them
        let mut param_types = Vec::new();
        for (i, param) in lambda.params.iter().enumerate() {
            let param_type = if let Some(annotated_type) = param.pattern_type() {
                // Explicit annotation takes precedence
                // First resolve type aliases, then substitute type params
                let resolved =
                    self.resolve_type_aliases_scoped(annotated_type, self.current_module.as_deref());
                self.substitute_from_type_param_scope(&resolved)
            } else if let Some(ref expected_params) = expected_param_types {
                // Use expected type from bidirectional checking
                expected_params[i].clone()
            } else {
                // No annotation and no expected type - create fresh type variable
                self.fresh_type_for_pattern(param)
            };

            param_types.push(param_type.clone());
            self.bind_pattern(param, &param_type, false)?;
        }

        // Type check the body
        let body_type = self.infer_expression(&lambda.body)?;

        // Handle return type annotation
        let return_type = if let Some(annotated_return_type) = &lambda.return_type {
            let substituted = self.substitute_from_type_param_scope(annotated_return_type);
            self.context.unify(&body_type, &substituted).map_err(|_| {
                err_type_at!(
                    lambda.body.h.span,
                    "Lambda body type {} does not match return type annotation {}",
                    self.format_type(&body_type),
                    self.format_type(&substituted)
                )
            })?;
            substituted
        } else {
            body_type
        };

        self.scope_stack.pop_scope();

        // Build the function type
        let func_type = Self::arrow_chain(&param_types, return_type);

        // If we had an expected type, unify with it
        if let Some(exp) = expected {
            self.context.unify(&func_type, exp).map_err(|_| {
                err_type_at!(
                    expr.h.span,
                    "Lambda type {} doesn't match expected type {}",
                    self.format_type(&func_type),
                    self.format_type(exp)
                )
            })?;
            // Store in type table when checking against expected type
            self.type_table.insert(expr.h.id, TypeScheme::Monotype(func_type.clone()));
        }

        Ok(func_type)
    }

    pub fn load_builtins(&mut self) -> Result<()> {
        // length: ∀n a. [n]a -> i32
        let (n, a) = (self.fresh_var(), self.fresh_var());
        let body = Self::arrow_chain(&[Self::array_ty(n, Self::var(a))], i32());
        self.scope_stack.insert("length".to_string(), Self::forall(&[n, a], body));

        // map: ∀a b n. (a -> b) -> [n]a -> [n]b
        let (a, b, n) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let body = Self::arrow_chain(
            &[
                Type::arrow(Self::var(a), Self::var(b)),
                Self::array_ty(n, Self::var(a)),
            ],
            Self::array_ty(n, Self::var(b)),
        );
        self.scope_stack.insert("map".to_string(), Self::forall(&[a, b, n], body));

        // zip: ∀n a b. [n]a -> [n]b -> [n](a, b)
        let (n, a, b) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let body = Self::arrow_chain(
            &[Self::array_ty(n, Self::var(a)), Self::array_ty(n, Self::var(b))],
            Self::array_ty(n, tuple(vec![Self::var(a), Self::var(b)])),
        );
        self.scope_stack.insert("zip".to_string(), Self::forall(&[n, a, b], body));

        // to_vec: ∀n a. [n]a -> Vec(n, a)
        let (n, a) = (self.fresh_var(), self.fresh_var());
        let body = Self::arrow_chain(&[Self::array_ty(n, Self::var(a))], Self::vec_ty(n, Self::var(a)));
        self.scope_stack.insert("to_vec".to_string(), Self::forall(&[n, a], body));

        // replicate: ∀size a. i32 -> a -> [size]a
        let (size, a) = (self.fresh_var(), self.fresh_var());
        let body = Self::arrow_chain(&[i32(), Self::var(a)], Self::array_ty(size, Self::var(a)));
        self.scope_stack.insert("replicate".to_string(), Self::forall(&[size, a], body));

        // _w_alloc_array: ∀n t. i32 -> [n]t
        let (n, t) = (self.fresh_var(), self.fresh_var());
        let body = Self::arrow_chain(&[i32()], Self::array_ty(n, Self::var(t)));
        self.scope_stack.insert("_w_alloc_array".to_string(), Self::forall(&[n, t], body));

        // dot: ∀n t. Vec(n, t) -> Vec(n, t) -> t
        let (n, t) = (self.fresh_var(), self.fresh_var());
        let vec = Self::vec_ty(n, Self::var(t));
        let body = Self::arrow_chain(&[vec.clone(), vec], Self::var(t));
        self.scope_stack.insert("dot".to_string(), Self::forall(&[n, t], body));

        // Trigonometric functions: f32 -> f32
        let trig_type = Type::arrow(f32(), f32());
        self.scope_stack.insert("sin".to_string(), TypeScheme::Monotype(trig_type.clone()));
        self.scope_stack.insert("cos".to_string(), TypeScheme::Monotype(trig_type.clone()));
        self.scope_stack.insert("tan".to_string(), TypeScheme::Monotype(trig_type));

        // Register vector field mappings
        self.register_vector_fields();

        // Note: Prelude files are automatically loaded when ModuleManager is created

        Ok(())
    }

    fn register_vector_fields(&mut self) {
        // Vector field access is now handled directly in the FieldAccess case
        // Vec(size, element_type) fields (x, y, z, w) return element_type
    }

    /// Register a record type with its field mappings
    pub fn register_record_type(&mut self, type_name: &str, fields: Vec<(String, Type)>) {
        for (field_name, field_type) in fields {
            self.record_field_map.insert((type_name.to_string(), field_name), field_type);
        }
    }

    pub fn check_program(&mut self, program: &Program) -> Result<HashMap<crate::ast::NodeId, TypeScheme>> {
        // Type-check prelude functions first (needed so they're in scope for user code)
        self.check_prelude_functions()?;

        // Type-check module function bodies (e.g., rand.init, rand.int)
        // This ensures module functions have type table entries for flattening
        self.check_module_functions()?;

        // Process user declarations
        for decl in &program.declarations {
            self.check_declaration(decl)?;
        }

        // Emit warnings for all type holes now that types are fully inferred
        self.emit_hole_warnings();

        // Apply the context to all types in the type table to resolve type variables
        let resolved_table: HashMap<crate::ast::NodeId, TypeScheme> = self
            .type_table
            .iter()
            .map(|(node_id, scheme)| {
                let resolved = match scheme {
                    TypeScheme::Monotype(ty) => {
                        let resolved_ty = ty.apply(&self.context);
                        TypeScheme::Monotype(resolved_ty)
                    }
                    TypeScheme::Polytype { variable, body } => {
                        // For polytypes, apply context to the body but preserve quantified variables
                        TypeScheme::Polytype {
                            variable: *variable,
                            body: Box::new(match body.as_ref() {
                                TypeScheme::Monotype(ty) => TypeScheme::Monotype(ty.apply(&self.context)),
                                other => other.clone(), // Nested polytypes stay as-is for now
                            }),
                        }
                    }
                };
                (*node_id, resolved)
            })
            .collect();

        Ok(resolved_table)
    }

    /// Emit warnings for all type holes showing their inferred types
    fn emit_hole_warnings(&mut self) {
        // Clone the holes list to avoid borrow checker issues
        let holes = self.type_holes.clone();
        for (node_id, span) in holes {
            if let Some(hole_scheme) = self.type_table.get(&node_id) {
                let resolved_type = match hole_scheme {
                    TypeScheme::Monotype(ty) => ty.apply(&self.context),
                    TypeScheme::Polytype { body, .. } => {
                        // For polytypes, just show the body type
                        match body.as_ref() {
                            TypeScheme::Monotype(ty) => ty.apply(&self.context),
                            _ => continue, // Skip nested polytypes for now
                        }
                    }
                };
                self.warnings.push(TypeWarning::TypeHoleFilled {
                    inferred_type: resolved_type,
                    span,
                });
            }
        }
    }

    /// Helper to type check a function body with parameters in scope
    /// Returns (param_types, body_type)
    /// If type_param_bindings is provided, UserVars in parameter types will be substituted
    fn check_function_with_params(
        &mut self,
        params: &[Pattern],
        body: &Expression,
        type_param_bindings: &HashMap<String, Type>,
        module_name: Option<&str>,
    ) -> Result<(Vec<Type>, Type)> {
        // Create type variables or use explicit types for parameters
        let param_types: Vec<Type> = params
            .iter()
            .map(|p| {
                let ty = p.pattern_type().cloned().unwrap_or_else(|| self.context.new_variable());
                // Resolve module type aliases (e.g., rand.state -> f32)
                let resolved = self.resolve_type_aliases_scoped(&ty, module_name);
                // Substitute UserVars with bound type variables
                Self::substitute_type_params_static(&resolved, type_param_bindings)
            })
            .collect();

        // Validate that no parameter types contain existential quantifiers
        // Existential types are only valid in return types, not parameters
        for (param, param_type) in params.iter().zip(param_types.iter()) {
            if Self::contains_existential(param_type) {
                bail_type_at!(
                    param.h.span,
                    "Existential types (?k. ...) are only allowed in return types, not parameter types"
                );
            }
        }

        // Push new scope for function parameters
        self.scope_stack.push_scope();

        // Add parameters to scope
        for (param, param_type) in params.iter().zip(param_types.iter()) {
            // Skip unit patterns (no parameters)
            if matches!(param.kind, PatternKind::Unit) {
                continue;
            }

            let param_name = param
                .simple_name()
                .ok_or_else(|| {
                    err_type_at!(
                        param.h.span,
                        "Complex patterns in function parameters not yet supported"
                    )
                })?
                .to_string();
            let type_scheme = TypeScheme::Monotype(param_type.clone());

            // Store resolved type in type_table for mirize
            // Need to insert for the outer pattern node ID
            let resolved_param_type = param_type.apply(&self.context);
            self.type_table.insert(param.h.id, TypeScheme::Monotype(resolved_param_type));

            debug!(
                "Adding parameter '{}' to scope with type: {:?}",
                param_name, param_type
            );
            self.scope_stack.insert(param_name, type_scheme);
        }

        // Infer body type
        let body_type = self.infer_expression(body)?;

        // Pop parameter scope
        self.scope_stack.pop_scope();

        Ok((param_types, body_type))
    }

    /// Type-check function bodies from modules (e.g., rand.init, rand.int)
    /// This populates the type table so these functions can be flattened to MIR.
    fn check_module_functions(&mut self) -> Result<()> {
        // Collect all module function declarations to avoid borrowing issues
        let module_functions: Vec<(String, crate::ast::Decl)> = self
            .module_manager
            .get_module_function_declarations()
            .into_iter()
            .map(|(module_name, decl)| (module_name.to_string(), decl.clone()))
            .collect();

        // Type-check each module function with module context for alias resolution
        for (module_name, decl) in module_functions {
            let qualified_name = format!("{}.{}", module_name, decl.name);
            debug!("Type-checking module function: {}", qualified_name);

            // Type-check the declaration with module context for alias resolution
            self.check_decl_as_in_module(&decl, &qualified_name, Some(&module_name))?;
        }

        Ok(())
    }

    /// Type-check all prelude functions.
    /// Called during prelude creation to populate the type table for prelude function bodies.
    pub fn check_prelude_functions(&mut self) -> Result<()> {
        // Collect all prelude function declarations to avoid borrowing issues
        let prelude_functions: Vec<crate::ast::Decl> =
            self.module_manager.get_prelude_function_declarations().into_iter().cloned().collect();

        // Type-check each prelude function
        for decl in prelude_functions {
            debug!("Type-checking prelude function: {}", decl.name);
            self.check_decl(&decl)?;
        }

        Ok(())
    }

    // =========================================================================
    // Module/prelude function type lookup (moved from ModuleManager)
    // =========================================================================

    /// Get the TypeScheme for a module function.
    /// This replaces ModuleManager::get_module_function_type, keeping polytype logic in TypeChecker.
    pub fn get_module_function_type_scheme(
        &mut self,
        module_name: &str,
        function_name: &str,
    ) -> Result<TypeScheme> {
        use crate::module_manager::ElaboratedItem;

        // Look up the elaborated module
        let elaborated = self
            .module_manager
            .get_elaborated_module(module_name)
            .ok_or_else(|| err_module!("Module '{}' not found", module_name))?;

        // Search for the function in the elaborated items
        for item in &elaborated.items {
            match item {
                ElaboratedItem::Spec(spec) => match spec {
                    Spec::Sig(name, type_params, ty) if name == function_name => {
                        // Convert to TypeScheme if there are type/size parameters
                        return Ok(self.convert_to_polytype(ty, type_params));
                    }
                    Spec::SigOp(op, ty) if op == function_name => {
                        // Operators currently don't have type parameters, return as Monotype
                        return Ok(TypeScheme::Monotype(ty.clone()));
                    }
                    _ => {}
                },
                ElaboratedItem::Decl(decl) if decl.name == function_name => {
                    // Build the full function type from parameters and return type
                    return self.build_function_type_from_decl(decl, module_name);
                }
                _ => {}
            }
        }

        Err(err_module!(
            "Function '{}' not found in module '{}'",
            function_name,
            module_name
        ))
    }

    /// Get the TypeScheme for a top-level prelude function.
    /// This replaces ModuleManager::get_prelude_function_type.
    pub fn get_prelude_function_type_scheme(&mut self, name: &str) -> Option<TypeScheme> {
        let decl = self.module_manager.get_prelude_function(name)?.clone();

        // Build the full function type from parameters and return type
        let mut param_types = Vec::new();
        for param in &decl.params {
            if let Some(param_ty) = self.extract_type_from_pattern(param) {
                param_types.push(param_ty);
            } else {
                // Parameter lacks type annotation, can't determine type
                return None;
            }
        }

        // Get return type (default to unit if not specified)
        let return_type = decl.ty.clone().unwrap_or_else(|| Type::Constructed(TypeName::Unit, vec![]));

        // Build function type by folding right-to-left
        let mut result_type = return_type;
        for param_ty in param_types.into_iter().rev() {
            result_type = Type::Constructed(TypeName::Arrow, vec![param_ty, result_type]);
        }

        // Convert to TypeScheme if there are type/size parameters
        let type_params = self.extract_type_params_from_type(&result_type);
        Some(self.convert_to_polytype(&result_type, &type_params))
    }

    /// Build the full function type from a declaration's parameters and return type.
    fn build_function_type_from_decl(&mut self, decl: &Decl, module_name: &str) -> Result<TypeScheme> {
        // Extract parameter types and resolve any type aliases within the module
        let mut param_types = Vec::new();
        for param in &decl.params {
            if let Some(param_ty) = self.extract_type_from_pattern(param) {
                // Resolve type aliases (e.g., "state" -> "f32" within the rand module)
                let resolved_ty = self.resolve_type_aliases_scoped(&param_ty, Some(module_name));
                param_types.push(resolved_ty);
            } else {
                bail_module!("Function parameter in '{}' lacks type annotation", decl.name);
            }
        }

        // Get return type (default to unit if not specified) and resolve aliases
        let return_type = decl.ty.clone().unwrap_or_else(|| Type::Constructed(TypeName::Unit, vec![]));
        let return_type = self.resolve_type_aliases_scoped(&return_type, Some(module_name));

        // Build function type by folding right-to-left
        // f32 -> f32 -> f32 is represented as f32 -> (f32 -> f32)
        let mut result_type = return_type;
        for param_ty in param_types.into_iter().rev() {
            result_type = Type::Constructed(TypeName::Arrow, vec![param_ty, result_type]);
        }

        // Convert to TypeScheme if there are type/size parameters
        let type_params = self.extract_type_params_from_type(&result_type);
        Ok(self.convert_to_polytype(&result_type, &type_params))
    }

    /// Convert a type with type parameters to a polymorphic TypeScheme.
    /// Converts SizeVar("n") and UserVar("t") to fresh Type::Variables
    /// and wraps the result in nested TypeScheme::Polytype layers.
    fn convert_to_polytype(&mut self, ty: &Type, type_params: &[TypeParam]) -> TypeScheme {
        if type_params.is_empty() {
            return TypeScheme::Monotype(ty.clone());
        }

        // Create fresh variables for each parameter and build substitution map
        let mut substitutions: HashMap<String, polytype::Variable> = HashMap::new();
        let mut var_ids = Vec::new();

        for param in type_params {
            let var = self.context.new_variable();
            if let Type::Variable(id) = var {
                var_ids.push(id);
                match param {
                    TypeParam::Size(name) => {
                        substitutions.insert(name.clone(), id);
                    }
                    TypeParam::Type(name) => {
                        substitutions.insert(name.clone(), id);
                    }
                    _ => {} // Ignore other param types for now
                }
            }
        }

        // Substitute SizeVar/UserVar with Variable in the type
        let substituted_ty = Self::substitute_type_params(ty, &substitutions);

        // Wrap in nested Polytype layers
        let mut result = TypeScheme::Monotype(substituted_ty);
        for &var_id in var_ids.iter().rev() {
            result = TypeScheme::Polytype {
                variable: var_id,
                body: Box::new(result),
            };
        }

        result
    }

    /// Recursively substitute SizeVar and UserVar with Variable.
    fn substitute_type_params(ty: &Type, substitutions: &HashMap<String, polytype::Variable>) -> Type {
        Self::substitute_named_vars(ty, &|tn| match tn {
            TypeName::UserVar(name) | TypeName::SizeVar(name) => {
                substitutions.get(name).map(|&var_id| Type::Variable(var_id))
            }
            _ => None,
        })
    }

    /// Extract type parameters from a type by finding all UserVar and SizeVar.
    fn extract_type_params_from_type(&self, ty: &Type) -> Vec<TypeParam> {
        let mut params = std::collections::HashSet::new();
        self.collect_type_params(ty, &mut params);
        params.into_iter().collect()
    }

    /// Recursively collect type parameters from a type.
    fn collect_type_params(&self, ty: &Type, params: &mut std::collections::HashSet<TypeParam>) {
        match ty {
            Type::Constructed(TypeName::UserVar(name), args) => {
                params.insert(TypeParam::Type(name.clone()));
                for arg in args {
                    self.collect_type_params(arg, params);
                }
            }
            Type::Constructed(TypeName::SizeVar(name), args) => {
                params.insert(TypeParam::Size(name.clone()));
                for arg in args {
                    self.collect_type_params(arg, params);
                }
            }
            Type::Constructed(_, args) => {
                for arg in args {
                    self.collect_type_params(arg, params);
                }
            }
            Type::Variable(_) => {}
        }
    }

    /// Extract type annotation from a pattern.
    fn extract_type_from_pattern(&self, pattern: &Pattern) -> Option<Type> {
        match &pattern.kind {
            PatternKind::Typed(_, ty) => Some(ty.clone()),
            PatternKind::Tuple(pats) => {
                // For tuple patterns, extract types from each element
                let elem_types: Option<Vec<Type>> =
                    pats.iter().map(|p| self.extract_type_from_pattern(p)).collect();
                elem_types.map(|types| Type::Constructed(TypeName::Tuple(types.len()), types))
            }
            _ => None,
        }
    }

    /// Consume the type checker and return the type table.
    /// Used to extract the prelude type table after type-checking prelude functions.
    pub fn into_type_table(self) -> std::collections::HashMap<crate::ast::NodeId, TypeScheme> {
        self.type_table
    }

    /// Get all function type schemes from the scope stack.
    /// Used to extract canonical schemes for prelude functions during prelude creation.
    /// This ensures monomorphization has consistent type variable IDs across params/return.
    pub fn get_function_schemes(&self) -> std::collections::HashMap<String, TypeScheme> {
        let mut schemes = std::collections::HashMap::new();
        self.scope_stack.for_each_binding(|name, scheme| {
            schemes.insert(name.to_string(), scheme.clone());
        });
        schemes
    }

    /// Consume the type checker and return all parts needed by FrontEnd.
    /// Returns (context, type_table, intrinsics, schemes).
    pub fn into_parts(
        self,
    ) -> (
        Context<TypeName>,
        std::collections::HashMap<crate::ast::NodeId, TypeScheme>,
        crate::intrinsics::IntrinsicSource,
        std::collections::HashMap<String, TypeScheme>,
    ) {
        // Extract schemes from scope_stack before consuming self
        let mut schemes = std::collections::HashMap::new();
        self.scope_stack.for_each_binding(|name, scheme| {
            schemes.insert(name.to_string(), scheme.clone());
        });
        (self.context, self.type_table, self.intrinsics, schemes)
    }

    fn check_declaration(&mut self, decl: &Declaration) -> Result<()> {
        match decl {
            Declaration::Decl(decl_node) => {
                debug!("Checking {} declaration: {}", decl_node.keyword, decl_node.name);
                self.check_decl(decl_node)
            }
            Declaration::Entry(entry) => {
                debug!("Checking entry point: {}", entry.name);
                let (_param_types, body_type) =
                    self.check_function_with_params(&entry.params, &entry.body, &HashMap::new(), None)?;
                debug!("Entry point '{}' body type: {:?}", entry.name, body_type);

                // Build expected return type from declared outputs
                let expected_type = match entry.outputs.len() {
                    0 => Type::Constructed(TypeName::Unit, vec![]),
                    1 => entry.outputs[0].ty.clone(),
                    n => Type::Constructed(
                        TypeName::Tuple(n),
                        entry.outputs.iter().map(|o| o.ty.clone()).collect(),
                    ),
                };
                // Resolve type aliases (e.g., rand.state -> f32)
                let expected_type = self.resolve_type_aliases_scoped(&expected_type, None);

                // Validate body type matches declared outputs
                self.context.unify(&body_type, &expected_type).map_err(|_| {
                    err_type_at!(
                        entry.body.h.span,
                        "Entry point '{}' return type mismatch: declared {}, inferred {}",
                        entry.name,
                        self.format_type(&expected_type),
                        self.format_type(&body_type)
                    )
                })?;

                Ok(())
            }
            Declaration::Uniform(uniform_decl) => {
                debug!("Checking Uniform declaration: {}", uniform_decl.name);
                self.check_uniform_decl(uniform_decl)
            }
            Declaration::Storage(storage_decl) => {
                debug!("Checking Storage declaration: {}", storage_decl.name);
                self.check_storage_decl(storage_decl)
            }
            Declaration::Sig(sig_decl) => {
                debug!("Checking Sig declaration: {}", sig_decl.name);
                self.check_sig_decl(sig_decl)
            }
            Declaration::TypeBind(type_bind) => {
                debug!("Processing TypeBind: {}", type_bind.name);
                // Type bindings are registered in the environment during elaboration
                // For now, just skip them in type checking
                Ok(())
            }
            Declaration::ModuleBind(_) => {
                // Module bindings should be elaborated away before type checking
                // If we encounter one here, it means elaboration wasn't run or failed
                Err(err_module!(
                    "Module bindings should be elaborated before type checking"
                ))
            }
            Declaration::ModuleTypeBind(_) => {
                // Module type bindings are erased during elaboration
                // If we see one, elaboration wasn't run
                Ok(())
            }
            Declaration::Open(_) => {
                // Open declarations should be elaborated away
                Ok(())
            }
            Declaration::Import(_) => {
                // Import declarations should be resolved during elaboration
                Ok(())
            }
        }
    }

    fn check_uniform_decl(&mut self, decl: &UniformDecl) -> Result<()> {
        // Add the uniform to scope with its declared type
        let type_scheme = TypeScheme::Monotype(decl.ty.clone());
        self.scope_stack.insert(decl.name.clone(), type_scheme);
        debug!("Inserting uniform variable '{}' into scope", decl.name);
        Ok(())
    }

    fn check_storage_decl(&mut self, decl: &StorageDecl) -> Result<()> {
        // Add the storage buffer to scope with its declared type
        let type_scheme = TypeScheme::Monotype(decl.ty.clone());
        self.scope_stack.insert(decl.name.clone(), type_scheme);
        debug!("Inserting storage variable '{}' into scope", decl.name);
        Ok(())
    }

    fn check_decl(&mut self, decl: &Decl) -> Result<()> {
        self.check_decl_as_in_module(decl, &decl.name, None)
    }

    fn check_decl_as_in_module(
        &mut self,
        decl: &Decl,
        scope_name: &str,
        module_name: Option<&str>,
    ) -> Result<()> {
        // Set current module context for alias resolution in nested expressions
        let saved_module = self.current_module.take();
        self.current_module = module_name.map(|s| s.to_string());

        // Push a new scope for type parameters
        self.type_param_scope.push_scope();

        // Bind type parameters to fresh type variables
        // This ensures all occurrences of 'a in the function signature refer to the same variable
        let mut type_param_bindings: HashMap<String, Type> = HashMap::new();
        for type_param in &decl.type_params {
            let fresh_var = self.context.new_variable();
            type_param_bindings.insert(type_param.clone(), fresh_var.clone());
            self.type_param_scope.insert(type_param.clone(), fresh_var);
        }

        // Bind size parameters to fresh type variables
        // Size parameters like [n] in "def f [n] (xs: [n]i32): i32" are treated as type variables
        // that can unify with concrete sizes (Size(8)) or other size variables
        for size_param in &decl.size_params {
            let fresh_var = self.context.new_variable();
            type_param_bindings.insert(size_param.clone(), fresh_var.clone());
            self.type_param_scope.insert(size_param.clone(), fresh_var);
        }

        // Note: substitution function defined as static method below

        if decl.params.is_empty() {
            // Variable or entry point declaration: let/def name: type = value or let/def name = value
            // Resolve type aliases in declared type (e.g., rand.state -> f32)
            let resolved_declared_type =
                decl.ty.as_ref().map(|ty| self.resolve_type_aliases_scoped(ty, module_name));

            let expr_type = if let Some(ref declared_type) = resolved_declared_type {
                // Use bidirectional checking when type annotation is present
                self.check_expression(&decl.body, declared_type)?
            } else {
                // No type annotation, infer the type
                self.infer_expression(&decl.body)?
            };

            if let Some(ref declared_type) = resolved_declared_type {
                if !self.types_match(&expr_type, declared_type) {
                    bail_type_at!(
                        decl.body.h.span,
                        "Type mismatch: expected {}, got {}",
                        self.format_type(declared_type),
                        self.format_type(&expr_type)
                    );
                }
            }

            // Add to scope - use declared type if available, otherwise inferred type
            let stored_type = resolved_declared_type.unwrap_or(expr_type.clone());
            // Generalize the type to enable polymorphism
            let type_scheme = self.generalize(&stored_type);
            debug!("Inserting variable '{}' into scope", scope_name);
            self.scope_stack.insert(scope_name.to_string(), type_scheme);
            debug!("Inferred type for {}: {}", scope_name, stored_type);
        } else {
            // Function declaration: let/def name param1 param2 = body

            // Special handling for _w_applyN dispatchers (generated by defunctionalization)
            // These functions route to different lambda functions based on closure tags
            // and can't be properly type-checked because different branches have different closure types
            if decl.name.starts_with("_w_apply") {
                // Create a polymorphic type: closure -> arg1 -> ... -> argN -> result
                // All types are fresh variables to allow maximum flexibility
                let mut param_types = Vec::new();
                for _ in &decl.params {
                    param_types.push(self.context.new_variable());
                }
                let result_type = self.context.new_variable();

                let func_type = param_types
                    .into_iter()
                    .rev()
                    .fold(result_type, |acc, param_ty| function(param_ty, acc));

                // Register the dispatcher with its polymorphic type
                let type_scheme = self.generalize(&func_type);
                self.scope_stack.insert(scope_name.to_string(), type_scheme);
                debug!(
                    "Registered _w_apply dispatcher '{}' with polymorphic type",
                    scope_name
                );
                self.type_param_scope.pop_scope();
                self.current_module = saved_module;
                return Ok(());
            }

            let (param_types, body_type) = self.check_function_with_params(
                &decl.params,
                &decl.body,
                &type_param_bindings,
                module_name,
            )?;
            debug!(
                "Successfully inferred body type for '{}': {:?}",
                decl.name, body_type
            );

            // Build function type: param1 -> param2 -> ... -> body_type
            let func_type = param_types
                .into_iter()
                .rev()
                .fold(body_type.clone(), |acc, param_ty| function(param_ty, acc));

            // Check against declared type if provided
            if let Some(declared_type) = &decl.ty {
                // Resolve type aliases in the declared return type (e.g., rand.state -> f32)
                let resolved_return_type = self.resolve_type_aliases_scoped(declared_type, module_name);
                // Substitute UserVars in the declared return type
                let substituted_return_type =
                    Self::substitute_type_params_static(&resolved_return_type, &type_param_bindings);

                // When a function has parameters, decl.ty is just the return type annotation
                // Unify the body type with the declared return type
                if !decl.params.is_empty() {
                    self.context.unify(&body_type, &substituted_return_type).map_err(|e| {
                        err_type_at!(
                            decl.body.h.span,
                            "Function return type mismatch for '{}': {}",
                            decl.name,
                            e
                        )
                    })?;
                } else {
                    // For functions without parameters, ty should be the full type
                    // But currently we're storing just the value type
                    // Since func_type for parameterless functions is just the body type,
                    // we can just check body_type against substituted declared_type
                    self.context.unify(&body_type, &substituted_return_type).map_err(|_| {
                        err_type_at!(
                            decl.body.h.span,
                            "Type mismatch for '{}': declared {}, inferred {}",
                            decl.name,
                            self.format_type(declared_type),
                            self.format_type(&body_type)
                        )
                    })?;
                }
            }

            // Entry points are now handled separately via Declaration::Entry
            // Regular Decl no longer has attributed return types

            // Update scope with inferred type using generalization
            let type_scheme = self.generalize(&func_type);
            self.scope_stack.insert(scope_name.to_string(), type_scheme);

            // Track arity for partial application checking
            self.arity_map.insert(scope_name.to_string(), decl.params.len());

            debug!("Inferred type for {}: {}", scope_name, func_type);
        }

        // Pop type parameter scope
        self.type_param_scope.pop_scope();

        // Restore previous module context
        self.current_module = saved_module;

        Ok(())
    }

    fn check_sig_decl(&mut self, decl: &SigDecl) -> Result<()> {
        // Sig declarations are just type signatures - register them in scope
        let type_scheme = TypeScheme::Monotype(decl.ty.clone());
        self.scope_stack.insert(decl.name.clone(), type_scheme);
        Ok(())
    }

    fn infer_expression(&mut self, expr: &Expression) -> Result<Type> {
        let ty = match &expr.kind {
            ExprKind::RecordLiteral(fields) => {
                let mut field_types = Vec::new();
                for (field_name, field_expr) in fields {
                    let field_ty = self.infer_expression(field_expr)?;
                    field_types.push((field_name.clone(), field_ty));
                }
                Ok(record(field_types))
            }
            ExprKind::TypeHole => {
                // Record this hole for warning emission after type inference completes
                self.type_holes.push((expr.h.id, expr.h.span));
                Ok(self.context.new_variable())
            }
            ExprKind::IntLiteral(_) => Ok(i32()),
            ExprKind::FloatLiteral(_) => Ok(f32()),
            ExprKind::BoolLiteral(_) => Ok(bool_type()),
            ExprKind::StringLiteral(_) => Ok(string()),
            ExprKind::Unit => Ok(unit()),
            ExprKind::Identifier(quals, name) => {
                let full_name = if quals.is_empty() {
                    name.clone()
                } else {
                    format!("{}.{}", quals.join("."), name)
                };

                debug!("Looking up identifier '{}'", full_name);
                debug!("Current scope depth: {}", self.scope_stack.depth());

                // For qualified names, check builtins with full name first
                if !quals.is_empty() {
                    if let Some(lookup) = self.intrinsics.get(&full_name) {
                        use crate::intrinsics::IntrinsicLookup;
                        let (scheme, ty) = match lookup {
                            IntrinsicLookup::Single(entry) => {
                                let ty = entry.scheme.instantiate(&mut self.context);
                                (entry.scheme.clone(), ty)
                            }
                            IntrinsicLookup::Overloaded(overloads) => {
                                // Overloaded intrinsics get fresh type vars; store as monotype
                                let ty = overloads.fresh_type(&mut self.context);
                                (TypeScheme::Monotype(ty.clone()), ty)
                            }
                        };
                        self.type_table.insert(expr.h.id, scheme);
                        return Ok(ty);
                    }

                    // Try to query from elaborated modules
                    let module_name = &quals[0];
                    if let Ok(type_scheme) = self.get_module_function_type_scheme(module_name, name) {
                        debug!("Found '{}' in elaborated module '{}' with type: {:?}", name, module_name, type_scheme);
                        let ty = type_scheme.instantiate(&mut self.context);
                        self.type_table.insert(expr.h.id, type_scheme);
                        return Ok(ty);
                    }
                }

                // Check scope stack for variables (use full_name for qualified lookups)
                if let Some(type_scheme) = self.scope_stack.lookup(&full_name) {
                    debug!("Found '{}' in scope stack with type: {:?}", full_name, type_scheme);
                    let ty = type_scheme.instantiate(&mut self.context);
                    self.type_table.insert(expr.h.id, type_scheme.clone());
                    return Ok(ty);
                } else if let Some(lookup) = self.intrinsics.get(&full_name) {
                    // Check polymorphic intrinsics for polymorphic function types
                    use crate::intrinsics::IntrinsicLookup;
                    debug!("'{}' is a polymorphic intrinsic", full_name);
                    let (scheme, func_type) = match lookup {
                        IntrinsicLookup::Single(entry) => {
                            let ty = entry.scheme.instantiate(&mut self.context);
                            (entry.scheme.clone(), ty)
                        }
                        IntrinsicLookup::Overloaded(overloads) => {
                            // Overloaded intrinsics get fresh type vars; store as monotype
                            let ty = overloads.fresh_type(&mut self.context);
                            (TypeScheme::Monotype(ty.clone()), ty)
                        }
                    };
                    debug!("Built function type for intrinsic '{}': {:?}", full_name, func_type);
                    self.type_table.insert(expr.h.id, scheme);
                    return Ok(func_type);
                } else if let Some(type_scheme) = self.get_prelude_function_type_scheme(&full_name) {
                    // Check top-level prelude functions (auto-imported)
                    debug!("'{}' is a prelude function with type: {:?}", full_name, type_scheme);
                    let ty = type_scheme.instantiate(&mut self.context);
                    self.type_table.insert(expr.h.id, type_scheme);
                    return Ok(ty);
                } else {
                    // Not found anywhere
                    debug!("Variable lookup failed for '{}' - not in scope, intrinsics, or prelude", full_name);
                    debug!("Scope stack contents: {:?}", self.scope_stack);
                    return Err(err_undef_at!(expr.h.span, "{}", full_name));
                }
            }
            ExprKind::ArrayLiteral(elements) => {
                if elements.is_empty() {
                    Err(err_type_at!(expr.h.span, "Cannot infer type of empty array"))
                } else {
                    let first_type = self.infer_expression(&elements[0])?;

                    // Futhark restriction: arrays of functions are not permitted
                    // Strip uniqueness to catch Unique(Arrow(...)) as well
                    let resolved_first = strip_unique(&first_type.apply(&self.context));
                    if as_arrow(&resolved_first).is_some() {
                        bail_type_at!(
                            expr.h.span,
                            "Arrays of functions are not permitted"
                        );
                    }

                    for elem in &elements[1..] {
                        let elem_type = self.infer_expression(elem)?;
                        self.context.unify(&elem_type, &first_type).map_err(|_| {
                            err_type_at!(
                                elem.h.span,
                                "Array elements must have the same type, expected {}, got {}",
                                self.format_type(&first_type),
                                self.format_type(&elem_type)
                            )
                        })?;
                    }

                    // Array literals have concrete sizes: [1, 2, 3] has type [3]i32
                    // Variable sizes require explicit type parameters: def f[n]: [n]i32 = ...
                    Ok(sized_array(elements.len(), first_type))
                }
            }
            ExprKind::VecMatLiteral(elements) => {
                // @[...] vector/matrix literal - type inferred from context
                // Vector: @[1.0, 2.0, 3.0] - elements are scalars
                // Matrix: @[[1,2,3], [4,5,6]] - elements are array literals (rows)
                if elements.is_empty() {
                    bail_type_at!(
                        expr.h.span,
                        "Cannot infer type of empty vector/matrix literal"
                    );
                }

                // Infer type of first element to determine if vector or matrix
                let first_type = self.infer_expression(&elements[0])?;

                // Check if first element is an array (matrix) or scalar (vector)
                let is_matrix = matches!(
                    first_type.apply(&self.context),
                    Type::Constructed(TypeName::Array, _)
                );

                // Unify all element types
                for elem in &elements[1..] {
                    let elem_type = self.infer_expression(elem)?;
                    self.context.unify(&elem_type, &first_type).map_err(|_| {
                        err_type_at!(
                            elem.h.span,
                            "All elements must have the same type, expected {}, got {}",
                            self.format_type(&first_type),
                            self.format_type(&elem_type)
                        )
                    })?;
                }

                if is_matrix {
                    // Matrix: extract row size and element type from the array type
                    let resolved = first_type.apply(&self.context);
                    if let Type::Constructed(TypeName::Array, args) = resolved {
                        if args.len() == 2 {
                            if let Type::Constructed(TypeName::Size(cols), _) = &args[0] {
                                let rows = elements.len();
                                let elem_type = args[1].clone();
                                Ok(mat(rows, *cols, elem_type))
                            } else {
                                Err(err_type_at!(
                                    expr.h.span,
                                    "Matrix rows must be fixed-size arrays"
                                ))
                            }
                        } else {
                            Err(err_type_at!(
                                expr.h.span,
                                "Matrix rows must be fixed-size arrays"
                            ))
                        }
                    } else {
                        Err(err_type_at!(
                            expr.h.span,
                            "Matrix rows must be fixed-size arrays"
                        ))
                    }
                } else {
                    // Vector literal
                    let size = elements.len();
                    if !(2..=4).contains(&size) {
                        Err(err_type_at!(
                            expr.h.span,
                            "Vector size must be 2, 3, or 4, got {}",
                            size
                        ))
                    } else {
                        Ok(vec(size, first_type))
                    }
                }
            }
            ExprKind::ArrayIndex(array_expr, index_expr) => {
                let array_type = self.infer_expression(array_expr)?;
                let index_type = self.infer_expression(index_expr)?;

                // Unify index type with i32
                // Per spec: array index may be "any unsigned integer type"
                // We use i32 for now for compatibility
                self.context.unify(&index_type, &i32()).map_err(|_| {
                    err_type_at!(
                        index_expr.h.span,
                        "Array index must be an integer type, got {}",
                        self.format_type(&index_type.apply(&self.context))
                    )
                })?;

                // Constrain array type to be Array(n, a) even if it's currently unknown
                // This allows indexing arrays whose type is a meta-variable
                // Strip uniqueness marker - indexing a *[n]T should work like indexing [n]T
                let array_type_stripped = strip_unique(&array_type);
                let size_var = self.context.new_variable();
                let elem_var = self.context.new_variable();
                let want_array = Type::Constructed(TypeName::Array, vec![size_var, elem_var.clone()]);

                self.context.unify(&array_type_stripped, &want_array).map_err(|_| {
                    err_type_at!(
                        array_expr.h.span,
                        "Cannot index non-array type: got {}",
                        self.format_type(&array_type.apply(&self.context))
                    )
                })?;

                // Return the element type, resolved through the context
                Ok(elem_var.apply(&self.context))
            }
            ExprKind::ArrayWith { array, index, value } => {
                // Type check: array must be [n]T, index must be i32, value must be T
                // Result type is [n]T
                let array_type = self.infer_expression(array)?;
                let index_type = self.infer_expression(index)?;
                let value_type = self.infer_expression(value)?;

                // Unify index type with i32
                self.context.unify(&index_type, &i32()).map_err(|_| {
                    err_type_at!(
                        index.h.span,
                        "Array index must be an integer type, got {}",
                        self.format_type(&index_type.apply(&self.context))
                    )
                })?;

                // Constrain array type to be Array(n, a)
                let array_type_stripped = strip_unique(&array_type);
                let size_var = self.context.new_variable();
                let elem_var = self.context.new_variable();
                let want_array = Type::Constructed(TypeName::Array, vec![size_var.clone(), elem_var.clone()]);

                self.context.unify(&array_type_stripped, &want_array).map_err(|_| {
                    err_type_at!(
                        array.h.span,
                        "Cannot update non-array type: got {}",
                        self.format_type(&array_type.apply(&self.context))
                    )
                })?;

                // Unify value type with element type
                self.context.unify(&value_type, &elem_var).map_err(|_| {
                    err_type_at!(
                        value.h.span,
                        "Array element type mismatch: expected {}, got {}",
                        self.format_type(&elem_var.apply(&self.context)),
                        self.format_type(&value_type.apply(&self.context))
                    )
                })?;

                // Return the array type (same type as input)
                Ok(array_type.apply(&self.context))
            }
            ExprKind::BinaryOp(op, left, right) => {
                let left_type = self.infer_expression(left)?;
                let right_type = self.infer_expression(right)?;

                // Check that both operands have compatible types
                self.context.unify(&left_type, &right_type).map_err(|_| {
                    err_type_at!(
                        expr.h.span,
                        "Binary operator '{}' requires operands of the same type, got {} and {}",
                        op.op,
                        left_type,
                        right_type
                    )
                })?;

                // Determine return type based on operator
                match op.op.as_str() {
                    "==" | "!=" | "<" | ">" | "<=" | ">=" => {
                        // Comparison operators return boolean
                        Ok(Type::Constructed(TypeName::Str("bool"), vec![]))
                    }
                    "&&" | "||" => {
                        // Logical operators require boolean operands and return boolean
                        let bool_type = Type::Constructed(TypeName::Str("bool"), vec![]);
                        self.context.unify(&left_type, &bool_type).map_err(|_| {
                            err_type_at!(
                                expr.h.span,
                                "Logical operator '{}' requires boolean operands, got {}",
                                op.op,
                                self.format_type(&left_type)
                            )
                        })?;
                        Ok(bool_type)
                    }
                    "+" | "-" | "*" | "/" | "%" | "**" => {
                        // Arithmetic operators require numeric operands
                        let resolved = left_type.apply(&self.context);
                        if let Some(false) = Self::is_numeric_type(&resolved) {
                            return Err(err_type_at!(
                                expr.h.span,
                                "Arithmetic operator '{}' requires numeric operands, got {}",
                                op.op,
                                self.format_type(&resolved)
                            ));
                        }
                        Ok(resolved)
                    }
                    _ => Err(err_type_at!(expr.h.span, "Unknown binary operator: {}", op.op)),
                }
            }
            ExprKind::Tuple(elements) => {
                let elem_types: Result<Vec<Type>> =
                    elements.iter().map(|e| self.infer_expression(e)).collect();

                Ok(tuple(elem_types?))
            }
            ExprKind::Lambda(lambda) => self.type_lambda(lambda, None, expr),
            ExprKind::LetIn(let_in) => {
                // Infer type of the value expression
                let value_type = self.infer_expression(&let_in.value)?;

                // Resolve type annotation if present
                let resolved_annotation = let_in.ty.as_ref().map(|ty| {
                    let resolved = self.resolve_type_aliases_scoped(ty, self.current_module.as_deref());
                    self.substitute_from_type_param_scope(&resolved)
                });

                // Check type annotation if present
                if let Some(declared_type) = &resolved_annotation {
                    self.context.unify(&value_type, declared_type).map_err(|_| {
                        err_type_at!(
                            let_in.value.h.span,
                            "Type mismatch in let binding: expected {}, got {}",
                            self.format_type(declared_type),
                            self.format_type(&value_type)
                        )
                    })?;
                }

                // Push new scope and bind pattern
                self.scope_stack.push_scope();
                let bound_type = resolved_annotation.unwrap_or_else(|| value_type.clone());

                // Open existential types: ?k. T becomes T[k'/k] where k' is fresh
                let bound_type = self.open_existential(bound_type);

                // Bind all names in the pattern
                // Let bindings should be generalized for polymorphism
                self.bind_pattern(&let_in.pattern, &bound_type, true)?;

                // Infer type of body expression
                let body_type = self.infer_expression(&let_in.body)?;

                // Pop scope
                self.scope_stack.pop_scope();

                Ok(body_type)
            }
            ExprKind::Application(func, args) => {
                // Resolve callee to candidate function types
                let callee = self.resolve_callee_candidates(func)?;

                // For single candidate, use apply_two_pass (handles lambdas correctly)
                // For multiple candidates (overloads), use inference-first approach
                if callee.candidates.len() == 1 {
                    let cand = callee.candidates.into_iter().next().unwrap();

                    // Use two-pass for bidirectional lambda checking
                    let result_ty = self.apply_two_pass(cand.ty.clone(), args)?;

                    // Check for partial application AFTER application (when types are resolved)
                    self.ensure_not_partial(&result_ty, &expr.h.span)?;

                    // Store types in type table
                    if let Some(s) = cand.scheme {
                        self.type_table.insert(func.h.id, s);
                    } else {
                        let resolved = cand.ty.apply(&self.context);
                        self.type_table
                            .insert(func.h.id, TypeScheme::Monotype(resolved));
                    }
                    self.type_table
                        .insert(expr.h.id, TypeScheme::Monotype(result_ty.clone()));
                    Ok(result_ty)
                } else {
                    // Multiple candidates: infer argument types first, then try overloads
                    let mut arg_types = Vec::new();
                    for arg in args {
                        arg_types.push(self.infer_expression(arg)?);
                    }

                    // Try each overload with backtracking
                    for cand in callee.candidates {
                        let saved_context = self.context.clone();

                        if let Some(result_ty) =
                            Self::try_unify_overload(&cand.ty, &arg_types, &mut self.context)
                        {
                            // Check for partial application
                            if self.ensure_not_partial(&result_ty, &expr.h.span).is_ok() {
                                let resolved_func_ty = cand.ty.apply(&self.context);
                                if let Some(s) = cand.scheme {
                                    self.type_table.insert(func.h.id, s);
                                } else {
                                    self.type_table
                                        .insert(func.h.id, TypeScheme::Monotype(resolved_func_ty));
                                }
                                self.type_table
                                    .insert(expr.h.id, TypeScheme::Monotype(result_ty.clone()));
                                return Ok(result_ty);
                            }
                        }

                        self.context = saved_context;
                    }

                    bail_type_at!(
                        expr.h.span,
                        "No matching overload for '{}' with argument types: {}",
                        callee.display_name,
                        arg_types
                            .iter()
                            .map(|t| self.format_type(t))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
            }
            ExprKind::FieldAccess(inner_expr, field) => {
                // Try to extract a qualified name (e.g., f32.cos, M.N.x)
                if let Some(qual_name) = Self::try_extract_qual_name(inner_expr, field) {
                    let dotted = qual_name.to_dotted();
                    let mangled = qual_name.mangle();

                    // Check if this is a module-qualified name (mangled name exists in scope)
                    if let Some(scheme) = self.scope_stack.lookup(&mangled) {
                        // Instantiate the type scheme
                        let ty = scheme.instantiate(&mut self.context);
                        self.type_table.insert(expr.h.id, TypeScheme::Monotype(ty.clone()));
                        return Ok(ty);
                    }

                    // Check if this is a polymorphic builtin (e.g., magnitude, normalize)
                    if let Some(lookup) = self.intrinsics.get(&dotted) {
                        use crate::intrinsics::IntrinsicLookup;
                        let ty = match lookup {
                            IntrinsicLookup::Single(entry) => entry.scheme.instantiate(&mut self.context),
                            IntrinsicLookup::Overloaded(overloads) => overloads.fresh_type(&mut self.context),
                        };
                        self.type_table.insert(expr.h.id, TypeScheme::Monotype(ty.clone()));
                        return Ok(ty);
                    }

                    // Try to look up in elaborated modules (e.g., f32.i32 for type conversion)
                    let module_name = &qual_name.qualifiers[0];
                    if let Ok(type_scheme) = self.get_module_function_type_scheme(module_name, &qual_name.name) {
                        let ty = type_scheme.instantiate(&mut self.context);
                        self.type_table.insert(expr.h.id, type_scheme);
                        return Ok(ty);
                    }

                    // Qualified name not found - fall through to field access
                }

                // Not a qualified name (or wasn't found), treat as normal field access
                {
                    // Not a qualified name, proceed with normal field access
                    let expr_type = self.infer_expression(inner_expr)?;

                    // Check if this is a _w_lambda_name field access (closure lambda name for direct dispatch)
                    // Allow it on any type variable and return string type
                    if field == "_w_lambda_name" {
                        // The type checker can't verify this is actually a closure record,
                        // but the defunctionalizer guarantees it. Just return string type.
                        let ty = Type::Constructed(TypeName::Str("string"), vec![]);
                        self.type_table.insert(expr.h.id, TypeScheme::Monotype(ty.clone()));
                        return Ok(ty);
                    }

                    // Apply context to resolve any type variables that have been unified
                    let expr_type = expr_type.apply(&self.context);

                    // Strip uniqueness for field access - the inner type's fields are accessible
                    let expr_type = expr_type.strip_unique().clone();

                    // Extract the type name from the expression type
                    // First check if it's a record with the requested field
                    if let Type::Constructed(TypeName::Record(fields), field_types) = &expr_type {
                        if let Some(field_index) = fields.get_index(field) {
                            if field_index < field_types.len() {
                                let field_type = &field_types[field_index];
                                self.type_table.insert(expr.h.id, TypeScheme::Monotype(field_type.clone()));
                                return Ok(field_type.clone());
                            }
                        }
                    }

                    // Check if this is a tuple numeric field access (0, 1, 2, etc.)
                    if let Ok(index) = field.parse::<usize>() {
                        // Tuple field access: t.0, t.1, etc.
                        // The expr_type should be a tuple
                        if let Type::Constructed(TypeName::Tuple(_), elem_types) = &expr_type {
                            if index < elem_types.len() {
                                let field_type = elem_types[index].clone();
                                self.type_table.insert(expr.h.id, TypeScheme::Monotype(field_type.clone()));
                                return Ok(field_type);
                            } else {
                                bail_type_at!(
                                    expr.h.span,
                                    "Tuple index {} out of bounds (tuple has {} elements)",
                                    index,
                                    elem_types.len()
                                );
                            }
                        } else {
                            bail_type_at!(
                                expr.h.span,
                                "Numeric field access '.{}' requires a tuple type, got {}",
                                index,
                                self.format_type(&expr_type)
                            );
                        }
                    }

                    // Check if this is a vector field access (x, y, z, w)
                    // If so, constrain the type to be a Vec even if it's currently unknown
                    if matches!(field.as_str(), "x" | "y" | "z" | "w") {
                        // Create a Vec type with unknown size and element type
                        let size_var = self.context.new_variable();
                        let elem_var = self.context.new_variable();
                        let want_vec = Type::Constructed(TypeName::Vec, vec![size_var, elem_var.clone()]);

                        // Unify to constrain expr_type to be a Vec
                        self.context.unify(&expr_type, &want_vec).map_err(|_| {
                            err_type_at!(
                                expr.h.span,
                                "Field access '{}' requires a vector type, got {}",
                                field,
                                self.format_type(&expr_type.apply(&self.context))
                            )
                        })?;

                        // Return the element type
                        let result_ty = elem_var.apply(&self.context);
                        self.type_table.insert(expr.h.id, TypeScheme::Monotype(result_ty.clone()));
                        return Ok(result_ty);
                    }

                    // Extract the type name from the expression type
                    match expr_type {
                        Type::Constructed(type_name, ref args) => {
                            // Handle Vec type specially for field access
                            if matches!(type_name, TypeName::Vec) {
                                // Vec(size, element_type) - must have exactly 2 args
                                if args.len() != 2 {
                                    bail_type_at!(
                                        expr.h.span,
                                        "Malformed Vec type - expected 2 arguments (size, element), got {}",
                                        args.len()
                                    );
                                }

                                // Fields x, y, z, w return the element type (args[1])
                                let element_type = &args[1];

                                // Check if field is valid (x, y, z, w)
                                if matches!(field.as_str(), "x" | "y" | "z" | "w") {
                                    Ok(element_type.clone())
                                } else {
                                    Err(err_type_at!(expr.h.span, "Vector type has no field '{}'", field))
                                }
                            } else if let TypeName::Record(fields) = &type_name {
                                // Handle Record type specially - look up field in the record's field map
                                if let Some(field_index) = fields.get_index(field) {
                                    if field_index < args.len() {
                                        let field_type = &args[field_index];
                                        self.type_table.insert(expr.h.id, TypeScheme::Monotype(field_type.clone()));
                                        return Ok(field_type.clone());
                                    }
                                }
                                // Field not found in record
                                bail_type_at!(
                                    expr.h.span,
                                    "Record type has no field '{}'. Available fields: {}",
                                    field,
                                    fields.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                                );
                            } else {
                                // Get the type name as a string for other types
                                let type_name_str = match &type_name {
                                    TypeName::Str(s) => s.to_string(),
                                    TypeName::Float(bits) => format!("f{}", bits),
                                    TypeName::UInt(bits) => format!("u{}", bits),
                                    TypeName::Int(bits) => format!("i{}", bits),
                                    TypeName::Array => "array".to_string(),
                                    TypeName::Unsized => "unsized".to_string(),
                                    TypeName::Arrow => "function".to_string(),
                                    TypeName::Vec => "vec".to_string(),
                                    TypeName::Mat => "mat".to_string(),
                                    TypeName::Size(n) => n.to_string(),
                                    TypeName::SizeVar(name) => name.clone(),
                                    TypeName::UserVar(name) => format!("'{}", name),
                                    TypeName::Named(name) => name.clone(),
                                    TypeName::Unique => unreachable!("Uniqueness stripped above"),
                                    TypeName::Record(_) => "record".to_string(),
                                    TypeName::Unit => "unit".to_string(),
                                    TypeName::Tuple(_) => "tuple".to_string(),
                                    TypeName::Sum(_) => "sum".to_string(),
                                    TypeName::Existential(_) => "existential".to_string(),
                                    TypeName::Pointer => "pointer".to_string(),
                                    TypeName::Slice => "slice".to_string(),
                                    TypeName::Skolem(id) => format!("{}", id),
                                };

                                // Look up field in builtin registry (for vector types)
                                if let Some(field_type) =
                                    self.impl_source.get_field_type(&type_name_str, field)
                                {
                                    Ok(field_type)
                                } else if let Some(field_type) =
                                    self.record_field_map.get(&(type_name_str.clone(), field.clone()))
                                {
                                    Ok(field_type.clone())
                                } else {
                                    Err(err_type_at!(
                                        expr.h.span,
                                        "Type '{}' has no field '{}'",
                                        type_name_str,
                                        field
                                    ))
                                }
                            }
                        }
                        _ => Err(err_type_at!(
                            expr.h.span,
                            "Field access '{}' not supported on type {}",
                            field,
                            expr_type
                        )),
                    }
                }
            }
            ExprKind::If(if_expr) => {
                // Infer condition type - should be bool
                let condition_ty = self.infer_expression(&if_expr.condition)?;
                let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);

                // Unify condition with bool type
                self.context.unify(&condition_ty, &bool_ty).map_err(|_| {
                    err_type_at!(
                        if_expr.condition.h.span,
                        "If condition must be boolean, got: {}",
                        self.format_type(&condition_ty)
                    )
                })?;

                // Infer then and else branch types - they must be the same
                let then_ty = self.infer_expression(&if_expr.then_branch)?;
                let else_ty = self.infer_expression(&if_expr.else_branch)?;

                // Unify then and else types
                self.context.unify(&then_ty, &else_ty).map_err(|_| {
                    err_type_at!(
                        if_expr.else_branch.h.span,
                        "If branches have incompatible types: then={}, else={}",
                        then_ty,
                        else_ty
                    )
                })?;

                // Futhark restriction: functions cannot be returned from if expressions
                let resolved_then = then_ty.apply(&self.context);
                if as_arrow(&resolved_then).is_some() {
                    bail_type_at!(
                        expr.h.span,
                        "Functions cannot be returned from if expressions"
                    );
                }

                Ok(then_ty)
            }

            ExprKind::UnaryOp(op, operand) => {
                let operand_type = self.infer_expression(operand)?;
                let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);
                match op.op.as_str() {
                    "-" => {
                        // Numeric negation - operand must be numeric
                        let resolved = operand_type.apply(&self.context);
                        if let Some(false) = Self::is_numeric_type(&resolved) {
                            return Err(err_type_at!(
                                operand.h.span,
                                "Unary minus requires numeric operand, got {}",
                                self.format_type(&resolved)
                            ));
                        }
                        Ok(resolved)
                    }
                    "!" => {
                        // Logical not - operand must be bool, returns bool
                        self.context.unify(&operand_type, &bool_ty).map_err(|_| {
                            err_type_at!(
                                operand.h.span,
                                "Logical not (!) requires bool operand, got {:?}",
                                operand_type
                            )
                        })?;
                        Ok(bool_ty)
                    }
                    _ => Err(err_type_at!(expr.h.span, "Unknown unary operator: {}", op.op)),
                }
            }

            ExprKind::Loop(loop_expr) => {
                // Push a new scope for loop variables
                self.scope_stack.push_scope();

                // Get or infer the type of the loop variable from init
                let loop_var_type = if let Some(init) = &loop_expr.init {
                    self.infer_expression(init)?
                } else {
                    // No init - create a fresh type variable
                    self.context.new_variable()
                };

                // Futhark restriction: loop parameters cannot be functions
                let resolved_loop_var = loop_var_type.apply(&self.context);
                if as_arrow(&resolved_loop_var).is_some() {
                    bail_type_at!(
                        expr.h.span,
                        "Loop parameters cannot be functions"
                    );
                }

                // Bind pattern to the loop variable type
                self.bind_pattern(&loop_expr.pattern, &loop_var_type, false)?;

                // Type check the loop form
                match &loop_expr.form {
                    LoopForm::While(cond) => {
                        // Condition must be bool
                        let cond_type = self.infer_expression(cond)?;
                        self.context.unify(&cond_type, &bool_type()).map_err(|_| {
                            err_type_at!(
                                cond.h.span,
                                "While condition must be bool, got {}",
                                self.format_type(&cond_type)
                            )
                        })?;
                    }
                    LoopForm::For(var_name, bound) => {
                        // Iteration variable is i32
                        self.scope_stack
                            .insert(var_name.clone(), TypeScheme::Monotype(i32()));

                        // Bound must be integer
                        let bound_type = self.infer_expression(bound)?;
                        self.context.unify(&bound_type, &i32()).map_err(|_| {
                            err_type_at!(
                                bound.h.span,
                                "Loop bound must be i32, got {}",
                                self.format_type(&bound_type)
                            )
                        })?;
                    }
                    LoopForm::ForIn(pat, arr) => {
                        // Array must be an array type
                        let arr_type = self.infer_expression(arr)?;
                        let elem_type = self.context.new_variable();
                        let size_type = self.context.new_variable();
                        let expected_arr = Type::Constructed(TypeName::Array, vec![size_type, elem_type.clone()]);

                        self.context.unify(&arr_type, &expected_arr).map_err(|_| {
                            err_type_at!(
                                arr.h.span,
                                "for-in requires an array, got {}",
                                self.format_type(&arr_type)
                            )
                        })?;

                        // Bind pattern to element type
                        self.bind_pattern(pat, &elem_type, false)?;
                    }
                }

                // Type check the body - its type must match the loop variable type
                let body_type = self.infer_expression(&loop_expr.body)?;
                self.context.unify(&body_type, &loop_var_type).map_err(|e| {
                    err_type_at!(
                        loop_expr.body.h.span,
                        "Loop body type must match loop variable type: {:?}",
                        e
                    )
                })?;

                // Pop the scope
                self.scope_stack.pop_scope();

                // The loop returns the loop variable type
                Ok(loop_var_type)
            }

            ExprKind::Match(_) => {
                Err(err_type_at!(
                    expr.h.span,
                    "match expressions are not yet supported"
                ))
            }

            ExprKind::Range(range) => {
                // Range expressions produce an array of integers
                // All operands must be the same integer type
                let start_type = self.infer_expression(&range.start)?;
                let end_type = self.infer_expression(&range.end)?;

                // Unify start and end types
                self.context.unify(&start_type, &end_type).map_err(|_| {
                    err_type_at!(
                        expr.h.span,
                        "Range start and end must have the same type: {} vs {}",
                        self.format_type(&start_type.apply(&self.context)),
                        self.format_type(&end_type.apply(&self.context))
                    )
                })?;

                // Check start is an integer type (unify with i32)
                self.context.unify(&start_type, &i32()).map_err(|_| {
                    err_type_at!(
                        range.start.h.span,
                        "Range operands must be integer types, got {}",
                        self.format_type(&start_type.apply(&self.context))
                    )
                })?;

                // Check step type if present
                let step_val = if let Some(step) = &range.step {
                    let step_type = self.infer_expression(step)?;
                    self.context.unify(&step_type, &start_type).map_err(|_| {
                        err_type_at!(
                            step.h.span,
                            "Range step must match start/end type: expected {}, got {}",
                            self.format_type(&start_type.apply(&self.context)),
                            self.format_type(&step_type.apply(&self.context))
                        )
                    })?;
                    Self::try_extract_const_int(step)
                } else {
                    Some(1) // Default step is 1
                };

                // Try to compute a concrete size if bounds are constant
                let start_val = Self::try_extract_const_int(&range.start);
                let end_val = Self::try_extract_const_int(&range.end);

                let size_type = match (start_val, end_val, step_val) {
                    (Some(start), Some(end), Some(step)) if step != 0 => {
                        // Calculate size based on range kind
                        let size = match range.kind {
                            RangeKind::Inclusive => (end - start) / step + 1,
                            RangeKind::Exclusive | RangeKind::ExclusiveLt => (end - start) / step,
                            RangeKind::ExclusiveGt => (start - end) / step,
                        };
                        if size > 0 {
                            Type::Constructed(TypeName::Size(size as usize), vec![])
                        } else {
                            self.context.new_variable()
                        }
                    }
                    _ => self.context.new_variable(),
                };

                let elem_type = start_type.apply(&self.context);
                Ok(Type::Constructed(TypeName::Array, vec![size_type, elem_type]))
            }

            ExprKind::Slice(slice) => {
                // Slice expression: array[start:end]
                // - array must be [n]T
                // - start/end (if present) must be integers
                // - result is [m]T where m = end - start (concrete if bounds are literals)

                let array_type = self.infer_expression(&slice.array)?;
                let array_type_stripped = strip_unique(&array_type);

                // Constrain array to be Array(n, elem)
                let size_var = self.context.new_variable();
                let elem_var = self.context.new_variable();
                let want_array = Type::Constructed(TypeName::Array, vec![size_var, elem_var.clone()]);

                self.context.unify(&array_type_stripped, &want_array).map_err(|_| {
                    err_type_at!(
                        slice.array.h.span,
                        "Cannot slice non-array type: got {}",
                        self.format_type(&array_type.apply(&self.context))
                    )
                })?;

                // Extract integer literal value from start (default 0)
                let start_val: Option<i32> = match &slice.start {
                    Some(start) => {
                        let start_type = self.infer_expression(start)?;
                        self.context.unify(&start_type, &i32()).map_err(|_| {
                            err_type_at!(
                                start.h.span,
                                "Slice start must be an integer, got {}",
                                self.format_type(&start_type.apply(&self.context))
                            )
                        })?;
                        // Try to extract literal value
                        match &start.kind {
                            ExprKind::IntLiteral(n) => Some(*n),
                            _ => None,
                        }
                    }
                    None => Some(0), // Default start is 0
                };

                // Extract integer literal value from end
                let end_val: Option<i32> = match &slice.end {
                    Some(end) => {
                        let end_type = self.infer_expression(end)?;
                        self.context.unify(&end_type, &i32()).map_err(|_| {
                            err_type_at!(
                                end.h.span,
                                "Slice end must be an integer, got {}",
                                self.format_type(&end_type.apply(&self.context))
                            )
                        })?;
                        // Try to extract literal value
                        match &end.kind {
                            ExprKind::IntLiteral(n) => Some(*n),
                            _ => None,
                        }
                    }
                    None => None, // No default for end - would need array length
                };

                // Compute result size: concrete if both bounds are known, otherwise fresh variable
                let result_size = match (start_val, end_val) {
                    (Some(s), Some(e)) if e >= s => {
                        // Both bounds are known literals - compute concrete size
                        Type::Constructed(TypeName::Size((e - s) as usize), vec![])
                    }
                    _ => {
                        // Dynamic bounds - use fresh type variable
                        self.context.new_variable()
                    }
                };

                let elem_type = elem_var.apply(&self.context);
                Ok(Type::Constructed(TypeName::Array, vec![result_size, elem_type]))
            }

            ExprKind::TypeAscription(expr, ascribed_ty) => {
                // Type ascription: check the inner expression against the ascribed type
                // This allows integer literals to take on the ascribed type (e.g., 42u32)
                let resolved = self.resolve_type_aliases_scoped(ascribed_ty, self.current_module.as_deref());
                let substituted = self.substitute_from_type_param_scope(&resolved);
                self.check_expression(expr, &substituted)?;
                Ok(substituted)
            }

            ExprKind::TypeCoercion(_, _) => {
                Err(err_type_at!(
                    expr.h.span,
                    "type coercion (:>) is not yet supported"
                ))
            }
        } // NEWCASESHERE - add new cases before this closing brace
        ?;

        // Store the inferred type in the type table
        self.type_table.insert(expr.h.id, TypeScheme::Monotype(ty.clone()));
        Ok(ty)
    }

    // Removed: fresh_var - now using polytype's context.new_variable()

    /// Resolve a callee expression to a set of candidate function types.
    /// For overloaded intrinsics, returns multiple candidates.
    /// For normal functions, returns a single candidate.
    fn resolve_callee_candidates(&mut self, func: &Expression) -> Result<CalleeCandidates> {
        use crate::intrinsics::IntrinsicLookup;

        match &func.kind {
            ExprKind::Identifier(quals, name) => {
                let full_name =
                    if quals.is_empty() { name.clone() } else { format!("{}.{}", quals.join("."), name) };

                // 1) Lexical scope first (local variables shadow intrinsics)
                if let Some(scheme) = self.scope_stack.lookup(&full_name).cloned() {
                    let ty = scheme.instantiate(&mut self.context);
                    return Ok(CalleeCandidates {
                        candidates: vec![Candidate {
                            ty,
                            scheme: Some(scheme),
                        }],
                        display_name: full_name,
                    });
                }

                // 2) Overloaded intrinsic? => multiple candidates
                if let Some(lookup) = self.intrinsics.get(&full_name) {
                    match lookup {
                        IntrinsicLookup::Overloaded(set) => {
                            let mut candidates = Vec::new();
                            for entry in set.entries() {
                                let ty = entry.scheme.instantiate(&mut self.context);
                                candidates.push(Candidate {
                                    ty,
                                    scheme: Some(entry.scheme.clone()),
                                });
                            }
                            return Ok(CalleeCandidates {
                                candidates,
                                display_name: full_name,
                            });
                        }
                        IntrinsicLookup::Single(entry) => {
                            let scheme = entry.scheme.clone();
                            let ty = scheme.instantiate(&mut self.context);
                            return Ok(CalleeCandidates {
                                candidates: vec![Candidate {
                                    ty,
                                    scheme: Some(scheme),
                                }],
                                display_name: full_name,
                            });
                        }
                    }
                }

                // 3) Module function?
                if !quals.is_empty() {
                    let module_name = &quals[0];
                    if let Ok(scheme) = self.get_module_function_type_scheme(module_name, name) {
                        let ty = scheme.instantiate(&mut self.context);
                        return Ok(CalleeCandidates {
                            candidates: vec![Candidate {
                                ty,
                                scheme: Some(scheme),
                            }],
                            display_name: full_name,
                        });
                    }
                }

                // 4) Prelude?
                if let Some(scheme) = self.get_prelude_function_type_scheme(&full_name) {
                    let ty = scheme.instantiate(&mut self.context);
                    return Ok(CalleeCandidates {
                        candidates: vec![Candidate {
                            ty,
                            scheme: Some(scheme),
                        }],
                        display_name: full_name,
                    });
                }

                Err(err_undef_at!(func.h.span, "{}", full_name))
            }

            // Any non-identifier callee: infer its type as single candidate
            _ => {
                let ty = self.infer_expression(func)?;
                Ok(CalleeCandidates {
                    candidates: vec![Candidate { ty, scheme: None }],
                    display_name: "<expr>".to_string(),
                })
            }
        }
    }

    /// Ensure the result type is not a function (no partial application).
    /// Note: () -> T is allowed (unit functions are not considered partial application).
    fn ensure_not_partial(&self, result_ty: &Type, call_span: &Span) -> Result<()> {
        let r = result_ty.apply(&self.context);
        if let Some((param, _)) = as_arrow(&r) {
            // Unit parameter doesn't count as partial application
            if !matches!(param, Type::Constructed(TypeName::Unit, _)) {
                bail_type_at!(
                    *call_span,
                    "Partial application not allowed: result is function type {}",
                    self.format_type(&r)
                );
            }
        }
        Ok(())
    }

    /// Two-pass function application for better lambda inference
    ///
    /// Pass 1: Process non-lambda arguments to constrain type variables
    /// Pass 2: Process lambda arguments with bidirectionally checked expected types
    ///
    /// This allows map (\x -> ...) arr to infer properly regardless of argument order
    fn apply_two_pass(&mut self, mut func_type: Type, args: &[Expression]) -> Result<Type> {
        // Collect argument types and expected types for lambdas
        let mut arg_types: Vec<Option<Type>> = vec![None; args.len()];
        let mut lambda_expected_types: Vec<Option<Type>> = vec![None; args.len()];

        // First pass: process arguments to constrain type variables
        for (i, arg) in args.iter().enumerate() {
            if let ExprKind::Lambda(lambda) = &arg.kind {
                // For lambdas with k params: build k arrows for the expected lambda type
                // This ensures check_expression can extract all k parameter types
                let num_params = lambda.params.len();
                let final_result = self.context.new_variable();

                // Build expected lambda type: α1 -> α2 -> ... -> αk -> β
                let mut expected_lambda_type = final_result;
                for _ in 0..num_params {
                    let param_var = self.context.new_variable();
                    expected_lambda_type = Type::arrow(param_var, expected_lambda_type);
                }

                // Peel one arrow from func_type: (expected_lambda_type -> result)
                let result_type = self.context.new_variable();
                let expected_func_type = Type::arrow(expected_lambda_type.clone(), result_type.clone());

                self.context.unify(&func_type, &expected_func_type).map_err(|_| {
                    err_type_at!(
                        arg.h.span,
                        "Cannot apply {} as a function",
                        self.format_type(&func_type)
                    )
                })?;

                // Store the whole expected lambda type (not just first param)
                lambda_expected_types[i] = Some(expected_lambda_type);

                func_type = result_type;
            } else {
                // For non-lambda argument: infer type and unify
                let arg_type = self.infer_expression(arg)?;
                arg_types[i] = Some(arg_type.clone());

                // Peel the head with a fresh arrow
                let param_type_var = self.context.new_variable();
                let result_type = self.context.new_variable();
                let expected_func_type = Type::arrow(param_type_var.clone(), result_type.clone());

                self.context.unify(&func_type, &expected_func_type).map_err(|_| {
                    err_type_at!(
                        arg.h.span,
                        "Cannot apply {} as a function",
                        self.format_type(&func_type)
                    )
                })?;

                // Extract the expected parameter type
                let expected_param_type = param_type_var.apply(&self.context);

                // Strip uniqueness for unification
                let arg_type_for_unify = strip_unique(&arg_type);
                let expected_param_for_unify = strip_unique(&expected_param_type);

                self.context.unify(&arg_type_for_unify, &expected_param_for_unify).map_err(|e| {
                    let error_msg = if arg.h.span.is_generated() {
                        format!(
                            "Function argument type mismatch at argument {}: {:?}\n\
                             Expected param type: {}\n\
                             Actual arg type: {}\n\
                             Generated expression: {:#?}",
                            i + 1,
                            e,
                            self.format_type(&expected_param_for_unify),
                            self.format_type(&arg_type_for_unify),
                            arg
                        )
                    } else {
                        format!("Function argument type mismatch: {:?}", e)
                    };
                    err_type_at!(arg.h.span, "{}", error_msg)
                })?;

                func_type = result_type;
            }
        }

        // Second pass: process lambda arguments with bidirectional checking
        for (i, arg) in args.iter().enumerate() {
            if !matches!(&arg.kind, ExprKind::Lambda(_)) {
                continue;
            }

            // Get the expected type from first pass
            let expected_param_type = lambda_expected_types[i].as_ref().map(|t| t.apply(&self.context));

            // Use bidirectional checking for lambdas
            if let Some(ref expected) = expected_param_type {
                self.check_expression(arg, expected)?;
            } else {
                self.infer_expression(arg)?;
            }
        }

        Ok(func_type.apply(&self.context))
    }

    /// Check if two types match structurally after applying substitutions.
    fn types_match(&self, t1: &Type, t2: &Type) -> bool {
        let a = t1.apply(&self.context);
        let b = t2.apply(&self.context);
        Self::types_equal_structural(&a, &b)
    }
}
