use super::{Type, TypeExt, TypeName, TypeScheme};
use crate::ast::*;
use crate::error::Result;
use crate::scope::ScopeStack;
use crate::{bail_type_at, err_module, err_type_at, err_undef_at};
use log::debug;
use polytype::Context;
use std::collections::{BTreeSet, HashMap};

// Import type helper functions from parent module
use super::{
    as_arrow, bool_type, count_arrows, f32, function, i32, mat, record, sized_array, string, strip_unique,
    tuple, unit, vec,
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
    poly_builtins: crate::poly_builtins::PolyBuiltins, // Type registry for polymorphic builtins
    module_manager: &'a crate::module_manager::ModuleManager, // Lazy module loading
    type_table: HashMap<crate::ast::NodeId, TypeScheme>, // Maps NodeId to type scheme
    warnings: Vec<TypeWarning>,          // Collected warnings
    type_holes: Vec<(NodeId, Span)>,     // Track type hole locations for warning emission
    arity_map: HashMap<String, usize>,   // function name -> required arity (number of params)
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

impl<'a> TypeChecker<'a> {
    /// Try to extract a constant integer value from an expression.
    /// Returns None if the expression is not a constant.
    fn try_extract_const_int(expr: &Expression) -> Option<i32> {
        match &expr.kind {
            ExprKind::IntLiteral(n) => Some(*n),
            _ => None,
        }
    }

    fn types_equal(&self, left: &Type, right: &Type) -> bool {
        match (left, right) {
            (Type::Constructed(l_name, l_args), Type::Constructed(r_name, r_args)) => {
                l_name == r_name
                    && l_args.len() == r_args.len()
                    && l_args.iter().zip(r_args.iter()).all(|(l, r)| self.types_equal(l, r))
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

    /// Resolve type aliases in a type (e.g., "rand.state" -> "f32")
    /// This expands qualified type names like "module.typename" to their underlying types.
    fn resolve_type_aliases(&self, ty: &Type) -> Type {
        match ty {
            Type::Constructed(TypeName::Named(name), args) if name.contains('.') => {
                // Qualified type name - check if it's a type alias
                if let Some(underlying) = self.module_manager.resolve_type_alias(name) {
                    // Recursively resolve in case of nested aliases
                    self.resolve_type_aliases(underlying)
                } else {
                    // Not a known alias, keep as-is but resolve args
                    let resolved_args: Vec<Type> =
                        args.iter().map(|a| self.resolve_type_aliases(a)).collect();
                    Type::Constructed(TypeName::Named(name.clone()), resolved_args)
                }
            }
            Type::Constructed(name, args) => {
                // Resolve aliases in type arguments
                let resolved_args: Vec<Type> = args.iter().map(|a| self.resolve_type_aliases(a)).collect();
                Type::Constructed(name.clone(), resolved_args)
            }
            Type::Variable(id) => Type::Variable(*id),
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
        let mut context = Context::default();
        let impl_source = crate::impl_source::ImplSource::new();
        let poly_builtins = crate::poly_builtins::PolyBuiltins::new(&mut context);

        TypeChecker {
            scope_stack: ScopeStack::new(),
            context,
            record_field_map: HashMap::new(),
            impl_source,
            poly_builtins,
            module_manager,
            type_table: HashMap::new(),
            warnings: Vec::new(),
            type_holes: Vec::new(),
            arity_map: HashMap::new(),
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
                self.resolve_type_aliases(annotated_type)
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
                let resolved_annotation = self.resolve_type_aliases(annotated_type);
                // Pattern has a type annotation - unify with expected type
                self.context.unify(&resolved_annotation, expected_type).map_err(|_| {
                    err_type_at!(
                        pattern.h.span,
                        "Pattern type annotation {} doesn't match expected type {}",
                        self.format_type(&resolved_annotation),
                        self.format_type(expected_type)
                    )
                })?;
                // Bind the inner pattern with the resolved type
                let result = self.bind_pattern(inner_pattern, &resolved_annotation, generalize)?;
                // Also store type for the outer Typed pattern
                let resolved = resolved_annotation.apply(&self.context);
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
            ExprKind::Lambda(lambda) => {
                // Special handling for lambdas in check mode
                // Extract parameter types from the expected function type
                let original_expected_type = expected_type.clone();
                let mut expected_type = expected_type.clone();
                let mut expected_param_types = Vec::new();

                // Unwrap nested function types to get parameter types
                for _ in 0..lambda.params.len() {
                    let applied = expected_type.apply(&self.context);
                    if let Some((param_type, result_type)) = as_arrow(&applied) {
                        expected_param_types.push(param_type.clone());
                        expected_type = result_type.clone();
                    } else {
                        // Expected type doesn't match lambda structure, fall back to inference
                        return self.infer_expression(expr);
                    }
                }

                // Now check the lambda with known parameter types
                self.scope_stack.push_scope();

                let mut param_types = Vec::new();
                for (param, expected_param_type) in lambda.params.iter().zip(expected_param_types.iter()) {
                    // If parameter has a type annotation, trust it
                    // Otherwise use the expected type from bidirectional checking
                    let param_type = if let Some(annotated_type) = param.pattern_type() {
                        annotated_type.clone()
                    } else {
                        // Use the expected type for the parameter
                        expected_param_type.clone()
                    };

                    param_types.push(param_type.clone());

                    // Bind the pattern (handles tuples, wildcards, etc.)
                    // Lambda parameters are not generalized
                    self.bind_pattern(param, &param_type, false)?;
                }

                // Check the body
                let body_type = self.infer_expression(&lambda.body)?;

                // If return type annotation exists, unify it with the body type
                let return_type = if let Some(annotated_return_type) = &lambda.return_type {
                    self.context.unify(&body_type, annotated_return_type).map_err(|_| {
                        err_type_at!(
                            lambda.body.h.span,
                            "Lambda body type {} does not match return type annotation {}",
                            self.format_type(&body_type),
                            self.format_type(annotated_return_type)
                        )
                    })?;
                    annotated_return_type.clone()
                } else {
                    body_type
                };

                self.scope_stack.pop_scope();

                // Build the function type
                let mut func_type = return_type;
                for param_type in param_types.iter().rev() {
                    func_type = function(param_type.clone(), func_type);
                }

                // Unify the built function type with the original expected type
                self.context.unify(&func_type, &original_expected_type).map_err(|_| {
                    err_type_at!(
                        expr.h.span,
                        "Lambda type {} doesn't match expected type {}",
                        self.format_type(&func_type),
                        self.format_type(&original_expected_type)
                    )
                })?;

                // Store the checked type in the type table
                self.type_table.insert(expr.h.id, TypeScheme::Monotype(func_type.clone()));
                Ok(func_type)
            }
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
    fn substitute_type_params_static(ty: &Type, bindings: &HashMap<String, Type>) -> Type {
        match ty {
            Type::Constructed(TypeName::UserVar(name), _) => {
                // Replace UserVar with the bound type variable
                bindings.get(name).cloned().unwrap_or_else(|| ty.clone())
            }
            Type::Constructed(TypeName::SizeVar(name), _) => {
                // Replace SizeVar with the bound size variable
                bindings.get(name).cloned().unwrap_or_else(|| ty.clone())
            }
            Type::Constructed(name, args) => {
                let new_args: Vec<Type> =
                    args.iter().map(|arg| Self::substitute_type_params_static(arg, bindings)).collect();
                Type::Constructed(name.clone(), new_args)
            }
            Type::Variable(_) => ty.clone(),
        }
    }

    pub fn load_builtins(&mut self) -> Result<()> {
        // Add builtin function types directly using manual construction

        // length: ∀a n. [n]a -> i32
        let var_n = self.context.new_variable();
        let var_a = self.context.new_variable();

        let var_n_id = if let Type::Variable(id) = var_n { id } else { panic!("Expected Type::Variable") };
        let var_a_id = if let Type::Variable(id) = var_a { id } else { panic!("Expected Type::Variable") };

        let array_type = Type::Constructed(
            TypeName::Array,
            vec![Type::Variable(var_n_id), Type::Variable(var_a_id)],
        );
        let length_body = Type::arrow(array_type, i32());

        // Create Polytype ∀n a. [n]a -> i32
        let length_scheme = TypeScheme::Polytype {
            variable: var_n_id,
            body: Box::new(TypeScheme::Polytype {
                variable: var_a_id,
                body: Box::new(TypeScheme::Monotype(length_body)),
            }),
        };
        self.scope_stack.insert("length".to_string(), length_scheme);

        // map: ∀a b n. (a -> b) -> [n]a -> [n]b
        // The input array is borrowed (read-only), output is fresh
        // Build the type using fresh type variables for proper polymorphism
        let var_a = self.context.new_variable();
        let var_b = self.context.new_variable();
        let var_n = self.context.new_variable();

        // Extract the variable IDs for the TypeScheme
        let var_a_id = if let Type::Variable(id) = var_a { id } else { panic!("Expected Type::Variable") };
        let var_b_id = if let Type::Variable(id) = var_b { id } else { panic!("Expected Type::Variable") };
        let var_n_id = if let Type::Variable(id) = var_n { id } else { panic!("Expected Type::Variable") };

        let func_type = Type::arrow(Type::Variable(var_a_id), Type::Variable(var_b_id));
        let input_array_type = Type::Constructed(
            TypeName::Array,
            vec![Type::Variable(var_n_id), Type::Variable(var_a_id)],
        );
        let output_array_type = Type::Constructed(
            TypeName::Array,
            vec![Type::Variable(var_n_id), Type::Variable(var_b_id)],
        );
        let map_arrow1 = Type::arrow(input_array_type, output_array_type);
        let map_body = Type::arrow(func_type, map_arrow1);
        // Create nested Polytype for ∀a b n
        let map_scheme = TypeScheme::Polytype {
            variable: var_a_id,
            body: Box::new(TypeScheme::Polytype {
                variable: var_b_id,
                body: Box::new(TypeScheme::Polytype {
                    variable: var_n_id,
                    body: Box::new(TypeScheme::Monotype(map_body)),
                }),
            }),
        };
        self.scope_stack.insert("map".to_string(), map_scheme);

        // zip: ∀a b n. [n]a -> [n]b -> [n](a, b)
        let var_n = self.context.new_variable();
        let var_a = self.context.new_variable();
        let var_b = self.context.new_variable();

        let var_n_id = if let Type::Variable(id) = var_n { id } else { panic!("Expected Type::Variable") };
        let var_a_id = if let Type::Variable(id) = var_a { id } else { panic!("Expected Type::Variable") };
        let var_b_id = if let Type::Variable(id) = var_b { id } else { panic!("Expected Type::Variable") };

        let array_a_type = Type::Constructed(
            TypeName::Array,
            vec![Type::Variable(var_n_id), Type::Variable(var_a_id)],
        );
        let array_b_type = Type::Constructed(
            TypeName::Array,
            vec![Type::Variable(var_n_id), Type::Variable(var_b_id)],
        );
        let tuple_type = tuple(vec![Type::Variable(var_a_id), Type::Variable(var_b_id)]);
        let result_array_type =
            Type::Constructed(TypeName::Array, vec![Type::Variable(var_n_id), tuple_type]);
        let zip_arrow1 = Type::arrow(array_b_type, result_array_type);
        let zip_body = Type::arrow(array_a_type, zip_arrow1);

        let zip_scheme = TypeScheme::Polytype {
            variable: var_n_id,
            body: Box::new(TypeScheme::Polytype {
                variable: var_a_id,
                body: Box::new(TypeScheme::Polytype {
                    variable: var_b_id,
                    body: Box::new(TypeScheme::Monotype(zip_body)),
                }),
            }),
        };
        self.scope_stack.insert("zip".to_string(), zip_scheme);

        // Array to vector conversion: to_vec
        // to_vec: ∀n a. [n]a -> Vec(n, a)
        let var_n = self.context.new_variable();
        let var_a = self.context.new_variable();

        let var_n_id = if let Type::Variable(id) = var_n { id } else { panic!("Expected Type::Variable") };
        let var_a_id = if let Type::Variable(id) = var_a { id } else { panic!("Expected Type::Variable") };

        let array_input = Type::Constructed(
            TypeName::Array,
            vec![Type::Variable(var_n_id), Type::Variable(var_a_id)],
        );
        let vec_output = Type::Constructed(
            TypeName::Vec,
            vec![Type::Variable(var_n_id), Type::Variable(var_a_id)],
        );
        let to_vec_body = Type::arrow(array_input, vec_output);

        let to_vec_scheme = TypeScheme::Polytype {
            variable: var_n_id,
            body: Box::new(TypeScheme::Polytype {
                variable: var_a_id,
                body: Box::new(TypeScheme::Monotype(to_vec_body)),
            }),
        };
        self.scope_stack.insert("to_vec".to_string(), to_vec_scheme);

        // replicate: ∀size a. i32 -> a -> [size]a
        // Creates an array of length n filled with the given value
        // Note: The size is determined by type inference from context
        let var_a = self.context.new_variable();
        let var_size = self.context.new_variable(); // Size will be inferred

        let var_a_id = if let Type::Variable(id) = var_a { id } else { panic!("Expected Type::Variable") };
        let var_size_id =
            if let Type::Variable(id) = var_size { id } else { panic!("Expected Type::Variable") };

        let output_array = Type::Constructed(
            TypeName::Array,
            vec![Type::Variable(var_size_id), Type::Variable(var_a_id)],
        );
        let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
        let replicate_body = Type::arrow(i32_type, Type::arrow(Type::Variable(var_a_id), output_array));

        let replicate_scheme = TypeScheme::Polytype {
            variable: var_size_id,
            body: Box::new(TypeScheme::Polytype {
                variable: var_a_id,
                body: Box::new(TypeScheme::Monotype(replicate_body)),
            }),
        };
        self.scope_stack.insert("replicate".to_string(), replicate_scheme);

        // _w_alloc_array: ∀n t. i32 -> [n]t
        // Allocates an uninitialized array of the given size
        // Used by map desugaring; size n and element type t are inferred from usage
        let var_n = self.context.new_variable();
        let var_t = self.context.new_variable();
        let var_n_id = if let Type::Variable(id) = var_n { id } else { panic!("Expected Type::Variable") };
        let var_t_id = if let Type::Variable(id) = var_t { id } else { panic!("Expected Type::Variable") };

        let array_type = Type::Constructed(
            TypeName::Array,
            vec![Type::Variable(var_n_id), Type::Variable(var_t_id)],
        );
        let alloc_array_body = Type::arrow(i32(), array_type);

        let alloc_array_scheme = TypeScheme::Polytype {
            variable: var_n_id,
            body: Box::new(TypeScheme::Polytype {
                variable: var_t_id,
                body: Box::new(TypeScheme::Monotype(alloc_array_body)),
            }),
        };
        self.scope_stack.insert("_w_alloc_array".to_string(), alloc_array_scheme);

        // Vector operations
        // dot: ∀n t. Vec(n, t) -> Vec(n, t) -> t
        // Takes two vectors of the same size and element type, returns the element type
        let var_n = self.context.new_variable();
        let var_t = self.context.new_variable();

        let var_n_id = if let Type::Variable(id) = var_n { id } else { panic!("Expected Type::Variable") };
        let var_t_id = if let Type::Variable(id) = var_t { id } else { panic!("Expected Type::Variable") };

        let vec_type = Type::Constructed(
            TypeName::Vec,
            vec![Type::Variable(var_n_id), Type::Variable(var_t_id)],
        );
        let dot_body = Type::arrow(vec_type.clone(), Type::arrow(vec_type, Type::Variable(var_t_id)));

        let dot_scheme = TypeScheme::Polytype {
            variable: var_n_id,
            body: Box::new(TypeScheme::Polytype {
                variable: var_t_id,
                body: Box::new(TypeScheme::Monotype(dot_body)),
            }),
        };
        self.scope_stack.insert("dot".to_string(), dot_scheme);

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
    ) -> Result<(Vec<Type>, Type)> {
        // Create type variables or use explicit types for parameters
        let param_types: Vec<Type> = params
            .iter()
            .map(|p| {
                let ty = p.pattern_type().cloned().unwrap_or_else(|| self.context.new_variable());
                // Resolve module type aliases (e.g., rand.state -> f32)
                let resolved = self.resolve_type_aliases(&ty);
                // Substitute UserVars with bound type variables
                Self::substitute_type_params_static(&resolved, type_param_bindings)
            })
            .collect();

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

        // Type-check each module function
        for (module_name, decl) in module_functions {
            let qualified_name = format!("{}.{}", module_name, decl.name);
            debug!("Type-checking module function: {}", qualified_name);

            // Resolve type aliases in the declaration (e.g., "state" -> "f32" within rand module)
            let resolved_decl = self.resolve_decl_type_aliases(&decl, &module_name);

            // Type-check the declaration body
            self.check_decl(&resolved_decl)?;

            // Register with qualified name so it can be found during flattening
            if let Some(type_scheme) = self.scope_stack.lookup(&decl.name) {
                self.scope_stack.insert(qualified_name.clone(), type_scheme.clone());
            }
        }

        Ok(())
    }

    /// Resolve type aliases in a declaration within a module context
    fn resolve_decl_type_aliases(&self, decl: &crate::ast::Decl, module_name: &str) -> crate::ast::Decl {
        // Resolve type aliases in the return type
        let resolved_ty = decl.ty.as_ref().map(|ty| self.resolve_type_aliases_in_module(ty, module_name));

        // Resolve type aliases in parameter patterns
        let resolved_params: Vec<_> =
            decl.params.iter().map(|p| self.resolve_pattern_type_aliases(p, module_name)).collect();

        crate::ast::Decl {
            keyword: decl.keyword,
            attributes: decl.attributes.clone(),
            name: decl.name.clone(),
            size_params: decl.size_params.clone(),
            type_params: decl.type_params.clone(),
            params: resolved_params,
            ty: resolved_ty,
            body: decl.body.clone(),
        }
    }

    /// Resolve type aliases in a pattern within a module context
    fn resolve_pattern_type_aliases(&self, pattern: &Pattern, module_name: &str) -> Pattern {
        use crate::ast::PatternKind;

        let resolved_kind = match &pattern.kind {
            PatternKind::Typed(inner, ty) => {
                let resolved_ty = self.resolve_type_aliases_in_module(ty, module_name);
                PatternKind::Typed(inner.clone(), resolved_ty)
            }
            PatternKind::Tuple(pats) => {
                let resolved_pats: Vec<_> =
                    pats.iter().map(|p| self.resolve_pattern_type_aliases(p, module_name)).collect();
                PatternKind::Tuple(resolved_pats)
            }
            other => other.clone(),
        };

        Pattern {
            kind: resolved_kind,
            h: pattern.h.clone(),
        }
    }

    /// Resolve type aliases in a type within a module context (unqualified names like "state")
    fn resolve_type_aliases_in_module(&self, ty: &Type, module_name: &str) -> Type {
        match ty {
            Type::Constructed(TypeName::Named(name), args) => {
                // Try to resolve as a module-local type alias
                let qualified_name = if name.contains('.') {
                    name.clone() // Already qualified
                } else {
                    format!("{}.{}", module_name, name) // Make qualified
                };

                if let Some(underlying) = self.module_manager.resolve_type_alias(&qualified_name) {
                    // Recursively resolve in case of nested aliases
                    self.resolve_type_aliases_in_module(underlying, module_name)
                } else {
                    // Not a known alias, keep as-is but resolve args
                    let resolved_args: Vec<Type> =
                        args.iter().map(|a| self.resolve_type_aliases_in_module(a, module_name)).collect();
                    Type::Constructed(TypeName::Named(name.clone()), resolved_args)
                }
            }
            Type::Constructed(name, args) => {
                let resolved_args: Vec<Type> =
                    args.iter().map(|a| self.resolve_type_aliases_in_module(a, module_name)).collect();
                Type::Constructed(name.clone(), resolved_args)
            }
            Type::Variable(v) => Type::Variable(*v),
        }
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
                    self.check_function_with_params(&entry.params, &entry.body, &HashMap::new())?;
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
        // Bind type parameters to fresh type variables
        // This ensures all occurrences of 'a in the function signature refer to the same variable
        let mut type_param_bindings: HashMap<String, Type> = HashMap::new();
        for type_param in &decl.type_params {
            let fresh_var = self.context.new_variable();
            type_param_bindings.insert(type_param.clone(), fresh_var);
        }

        // Bind size parameters to fresh type variables
        // Size parameters like [n] in "def f [n] (xs: [n]i32): i32" are treated as type variables
        // that can unify with concrete sizes (Size(8)) or other size variables
        for size_param in &decl.size_params {
            let fresh_var = self.context.new_variable();
            type_param_bindings.insert(size_param.clone(), fresh_var);
        }

        // Note: substitution function defined as static method below

        if decl.params.is_empty() {
            // Variable or entry point declaration: let/def name: type = value or let/def name = value
            let expr_type = if let Some(declared_type) = &decl.ty {
                // Use bidirectional checking when type annotation is present
                self.check_expression(&decl.body, declared_type)?
            } else {
                // No type annotation, infer the type
                self.infer_expression(&decl.body)?
            };

            if let Some(declared_type) = &decl.ty {
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
            let stored_type = decl.ty.as_ref().unwrap_or(&expr_type).clone();
            // Generalize the type to enable polymorphism
            let type_scheme = self.generalize(&stored_type);
            debug!("Inserting variable '{}' into scope", decl.name);
            self.scope_stack.insert(decl.name.clone(), type_scheme);
            debug!("Inferred type for {}: {}", decl.name, stored_type);
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
                self.scope_stack.insert(decl.name.clone(), type_scheme);
                debug!(
                    "Registered _w_apply dispatcher '{}' with polymorphic type",
                    decl.name
                );
                return Ok(());
            }

            let (param_types, body_type) =
                self.check_function_with_params(&decl.params, &decl.body, &type_param_bindings)?;
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
                let resolved_return_type = self.resolve_type_aliases(declared_type);
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
            self.scope_stack.insert(decl.name.clone(), type_scheme);

            // Track arity for partial application checking
            self.arity_map.insert(decl.name.clone(), decl.params.len());

            debug!("Inferred type for {}: {}", decl.name, func_type);
        }

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
                self.type_holes.push((expr.h.id, expr.h.span.clone()));
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
                    if let Some(lookup) = self.poly_builtins.get(&full_name) {
                        use crate::poly_builtins::BuiltinLookup;
                        let ty = match lookup {
                            BuiltinLookup::Single(entry) => entry.scheme.instantiate(&mut self.context),
                            BuiltinLookup::Overloaded(overloads) => overloads.fresh_type(&mut self.context),
                        };
                        self.type_table.insert(expr.h.id, polytype::TypeScheme::Monotype(ty.clone()));
                        return Ok(ty);
                    }

                    // Try to query from elaborated modules
                    let module_name = &quals[0];
                    if let Ok(type_scheme) = self.module_manager.get_module_function_type(module_name, name, &mut self.context) {
                        debug!("Found '{}' in elaborated module '{}' with type: {:?}", name, module_name, type_scheme);
                        let ty = type_scheme.instantiate(&mut self.context);
                        self.type_table.insert(expr.h.id, type_scheme);
                        return Ok(ty);
                    }
                }

                // Check scope stack for variables (use full_name for qualified lookups)
                if let Some(type_scheme) = self.scope_stack.lookup(&full_name) {
                    debug!("Found '{}' in scope stack with type: {:?}", full_name, type_scheme);
                    // Instantiate the type scheme to get a concrete type
                    Ok(type_scheme.instantiate(&mut self.context))
                } else if let Some(lookup) = self.poly_builtins.get(&full_name) {
                    // Check polymorphic builtins for polymorphic function types
                    use crate::poly_builtins::BuiltinLookup;
                    debug!("'{}' is a polymorphic builtin", full_name);
                    let func_type = match lookup {
                        BuiltinLookup::Single(entry) => entry.scheme.instantiate(&mut self.context),
                        BuiltinLookup::Overloaded(overloads) => overloads.fresh_type(&mut self.context),
                    };
                    debug!("Built function type for builtin '{}': {:?}", full_name, func_type);
                    Ok(func_type)
                } else {
                    // Not found anywhere
                    debug!("Variable lookup failed for '{}' - not in scope or builtins", full_name);
                    debug!("Scope stack contents: {:?}", self.scope_stack);
                    Err(err_undef_at!(expr.h.span, "{}", full_name))
                }
            }
            ExprKind::ArrayLiteral(elements) => {
                if elements.is_empty() {
                    Err(err_type_at!(expr.h.span, "Cannot infer type of empty array"))
                } else {
                    let first_type = self.infer_expression(&elements[0])?;

                    // Futhark restriction: arrays of functions are not permitted
                    let resolved_first = first_type.apply(&self.context);
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
                    if size < 2 || size > 4 {
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
                    "+" | "-" | "*" | "/" | "%" | "**" => {
                        // Arithmetic operators return the same type as operands
                        Ok(left_type.apply(&self.context))
                    }
                    _ => Err(err_type_at!(expr.h.span, "Unknown binary operator: {}", op.op)),
                }
            }
            ExprKind::Tuple(elements) => {
                let elem_types: Result<Vec<Type>> =
                    elements.iter().map(|e| self.infer_expression(e)).collect();

                Ok(tuple(elem_types?))
            }
            ExprKind::Lambda(lambda) => {
                // Push new scope for lambda parameters
                self.scope_stack.push_scope();

                // Add parameters to scope with their types (or fresh type variables)
                // Save the parameter types so we can reuse them when building the function type
                let mut param_types = Vec::new();
                for param in &lambda.params {
                    let param_type = param.pattern_type().cloned().unwrap_or_else(|| {
                        // No explicit type annotation - infer from pattern shape
                        self.fresh_type_for_pattern(param)
                    });
                    param_types.push(param_type.clone());

                    // Bind the pattern (handles tuples, wildcards, etc.)
                    // Lambda parameters are not generalized
                    self.bind_pattern(param, &param_type, false)?;
                }

                // Type check the lambda body with parameters in scope
                let body_type = self.infer_expression(&lambda.body)?;

                // If return type annotation exists, unify it with the body type
                let return_type = if let Some(annotated_return_type) = &lambda.return_type {
                    self.context.unify(&body_type, annotated_return_type).map_err(|_| {
                        err_type_at!(
                            lambda.body.h.span,
                            "Lambda body type {} does not match return type annotation {}",
                            self.format_type(&body_type),
                            self.format_type(annotated_return_type)
                        )
                    })?;
                    annotated_return_type.clone()
                } else {
                    body_type
                };

                // Pop parameter scope
                self.scope_stack.pop_scope();

                // For multiple parameters, create nested function types using the SAME type variables
                // we used when adding parameters to scope
                let mut func_type = return_type;
                for param_type in param_types.iter().rev() {
                    func_type = function(param_type.clone(), func_type);
                }

                Ok(func_type)
            }
            ExprKind::LetIn(let_in) => {
                // Infer type of the value expression
                let value_type = self.infer_expression(&let_in.value)?;

                // Check type annotation if present
                if let Some(declared_type) = &let_in.ty {
                    self.context.unify(&value_type, declared_type).map_err(|_| {
                        err_type_at!(
                            let_in.value.h.span,
                            "Type mismatch in let binding: expected {}, got {}",
                            declared_type,
                            value_type
                        )
                    })?;
                }

                // Push new scope and bind pattern
                self.scope_stack.push_scope();
                let bound_type = let_in.ty.as_ref().unwrap_or(&value_type).clone();

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
                // Check arity for partial application prevention
                let required_arity: Option<usize> = match &func.kind {
                    ExprKind::Identifier(quals, name) => {
                        let full_name = if quals.is_empty() {
                            name.clone()
                        } else {
                            format!("{}.{}", quals.join("."), name)
                        };
                        // First check user-defined functions
                        if let Some(arity) = self.arity_map.get(&full_name).copied() {
                            Some(arity)
                        } else if self.scope_stack.lookup(&full_name).is_some() {
                            // Name is bound locally - don't check builtins
                            // (local variables shadow builtins, and we can't know their arity)
                            None
                        } else {
                            // Check builtins - extract arity from type scheme
                            use crate::poly_builtins::BuiltinLookup;
                            if let Some(lookup) = self.poly_builtins.get(&full_name) {
                                let ty = match lookup {
                                    BuiltinLookup::Single(entry) => entry.scheme.instantiate(&mut self.context),
                                    BuiltinLookup::Overloaded(overloads) => overloads.fresh_type(&mut self.context),
                                };
                                Some(count_arrows(&ty))
                            } else if !quals.is_empty() {
                                // Check module functions (e.g., f32.min)
                                let module_name = &quals[0];
                                if let Ok(type_scheme) = self.module_manager.get_module_function_type(module_name, name, &mut self.context) {
                                    let ty = type_scheme.instantiate(&mut self.context);
                                    Some(count_arrows(&ty))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                    }
                    ExprKind::Lambda(lambda) => Some(lambda.params.len()),
                    _ => None,
                };

                if let Some(arity) = required_arity {
                    if args.len() < arity {
                        bail_type_at!(
                            expr.h.span,
                            "Partial application not allowed: function requires {} argument(s), but {} provided",
                            arity,
                            args.len()
                        );
                    }
                }

                // Check if the function is an overloaded builtin identifier
                // If so, perform overload resolution based on argument types
                if let ExprKind::Identifier(quals, name) = &func.kind {
                    let full_name = if quals.is_empty() {
                        name.clone()
                    } else {
                        format!("{}.{}", quals.join("."), name)
                    };
                    use crate::poly_builtins::BuiltinLookup;
                    // Clone entries to release the borrow on self.poly_builtins
                    let overload_entries = match self.poly_builtins.get(&full_name) {
                        Some(BuiltinLookup::Overloaded(overload_set)) => {
                            Some(overload_set.entries().to_vec())
                        }
                        _ => None,
                    };

                    if let Some(entries) = overload_entries {
                        // Infer argument types first
                        let mut arg_types = Vec::new();
                        for arg in args {
                            arg_types.push(self.infer_expression(arg)?);
                        }

                        // Try each overload with backtracking
                        for entry in &entries {
                            let saved_context = self.context.clone();
                            let func_type = entry.scheme.instantiate(&mut self.context);

                            if let Some(return_type) = Self::try_unify_overload(&func_type, &arg_types, &mut self.context) {
                                // Store the types in the type table
                                let resolved_func_type = func_type.apply(&self.context);
                                self.type_table.insert(func.h.id, TypeScheme::Monotype(resolved_func_type));
                                self.type_table.insert(expr.h.id, TypeScheme::Monotype(return_type.clone()));
                                return Ok(return_type);
                            }

                            self.context = saved_context;
                        }

                        bail_type_at!(
                            expr.h.span,
                            "No matching overload for '{}' with argument types: {}",
                            name,
                            arg_types.iter().map(|t| self.format_type(t)).collect::<Vec<_>>().join(", ")
                        );
                    }
                }

                // Not an overloaded builtin, use standard application
                let func_type = self.infer_expression(func)?;

                // Use two-pass application for better lambda inference
                // This enables proper inference for expressions like (map (\x -> ...) arr)
                // or (|>) operators with lambdas
                self.apply_two_pass(func_type, args)
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
                    if let Some(lookup) = self.poly_builtins.get(&dotted) {
                        use crate::poly_builtins::BuiltinLookup;
                        let ty = match lookup {
                            BuiltinLookup::Single(entry) => entry.scheme.instantiate(&mut self.context),
                            BuiltinLookup::Overloaded(overloads) => overloads.fresh_type(&mut self.context),
                        };
                        self.type_table.insert(expr.h.id, TypeScheme::Monotype(ty.clone()));
                        return Ok(ty);
                    }

                    // Try to look up in elaborated modules (e.g., f32.i32 for type conversion)
                    let module_name = &qual_name.qualifiers[0];
                    if let Ok(type_scheme) = self.module_manager.get_module_function_type(module_name, &qual_name.name, &mut self.context) {
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
                        let ty = Type::Constructed(TypeName::Str("string".into()), vec![]);
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
                                    TypeName::Existential(_, _) => "existential".to_string(),
                                    TypeName::Pointer => "pointer".to_string(),
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
                        // Numeric negation - operand must be numeric, returns same type
                        Ok(operand_type)
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
                        self.context.unify(&cond_type, &bool_type()).map_err(|e| {
                            err_type_at!(cond.h.span, "While condition must be bool: {:?}", e)
                        })?;
                    }
                    LoopForm::For(var_name, bound) => {
                        // Iteration variable is i32
                        self.scope_stack
                            .insert(var_name.clone(), TypeScheme::Monotype(i32()));

                        // Bound must be integer
                        let bound_type = self.infer_expression(bound)?;
                        self.context.unify(&bound_type, &i32()).map_err(|e| {
                            err_type_at!(bound.h.span, "Loop bound must be i32: {:?}", e)
                        })?;
                    }
                    LoopForm::ForIn(pat, arr) => {
                        // Array must be an array type
                        let arr_type = self.infer_expression(arr)?;
                        let elem_type = self.context.new_variable();
                        let size_type = self.context.new_variable();
                        let expected_arr = Type::Constructed(TypeName::Array, vec![size_type, elem_type.clone()]);

                        self.context.unify(&arr_type, &expected_arr).map_err(|e| {
                            err_type_at!(arr.h.span, "for-in requires an array: {:?}", e)
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
                todo!("Match not yet implemented in type checker")
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
                // Slice expression: array[start:end:step]
                // - array must be [n]T
                // - start/end/step (if present) must be integers
                // - result is [?m]T (existential size)

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

                // Check start, end, step are integers (if present)
                if let Some(start) = &slice.start {
                    let start_type = self.infer_expression(start)?;
                    self.context.unify(&start_type, &i32()).map_err(|_| {
                        err_type_at!(
                            start.h.span,
                            "Slice start must be an integer, got {}",
                            self.format_type(&start_type.apply(&self.context))
                        )
                    })?;
                }

                if let Some(end) = &slice.end {
                    let end_type = self.infer_expression(end)?;
                    self.context.unify(&end_type, &i32()).map_err(|_| {
                        err_type_at!(
                            end.h.span,
                            "Slice end must be an integer, got {}",
                            self.format_type(&end_type.apply(&self.context))
                        )
                    })?;
                }

                if let Some(step) = &slice.step {
                    let step_type = self.infer_expression(step)?;
                    self.context.unify(&step_type, &i32()).map_err(|_| {
                        err_type_at!(
                            step.h.span,
                            "Slice step must be an integer, got {}",
                            self.format_type(&step_type.apply(&self.context))
                        )
                    })?;
                }

                // Result has a fresh (existential) size and the same element type
                let result_size = self.context.new_variable();
                let elem_type = elem_var.apply(&self.context);
                Ok(Type::Constructed(TypeName::Array, vec![result_size, elem_type]))
            }

            ExprKind::TypeAscription(expr, ascribed_ty) => {
                // Type ascription: check the inner expression against the ascribed type
                // This allows integer literals to take on the ascribed type (e.g., 42u32)
                self.check_expression(expr, ascribed_ty)?;
                Ok(ascribed_ty.clone())
            }

            ExprKind::TypeCoercion(_, _) => {
                todo!("TypeCoercion not yet implemented in type checker")
            }

            ExprKind::Assert(_, _) => {
                todo!("Assert not yet implemented in type checker")
            }
        } // NEWCASESHERE - add new cases before this closing brace
        ?;

        // Store the inferred type in the type table
        self.type_table.insert(expr.h.id, TypeScheme::Monotype(ty.clone()));
        Ok(ty)
    }

    // Removed: fresh_var - now using polytype's context.new_variable()

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
            if matches!(&arg.kind, ExprKind::Lambda(_)) {
                // For lambdas: peel the head with a fresh arrow (α -> β) and unify
                let param_type_var = self.context.new_variable();
                let result_type = self.context.new_variable();
                let expected_func_type = Type::arrow(param_type_var.clone(), result_type.clone());

                self.context
                    .unify(&func_type, &expected_func_type)
                    .map_err(|e| err_type_at!(arg.h.span, "Function application type error: {:?}", e))?;

                // Extract the parameter type by applying context
                let param_type = param_type_var.apply(&self.context);
                lambda_expected_types[i] = Some(param_type);

                func_type = result_type;
            } else {
                // For non-lambda argument: infer type and unify
                let arg_type = self.infer_expression(arg)?;
                arg_types[i] = Some(arg_type.clone());

                // Peel the head with a fresh arrow
                let param_type_var = self.context.new_variable();
                let result_type = self.context.new_variable();
                let expected_func_type = Type::arrow(param_type_var.clone(), result_type.clone());

                self.context
                    .unify(&func_type, &expected_func_type)
                    .map_err(|e| err_type_at!(arg.h.span, "Function application type error: {:?}", e))?;

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

    /// Check if two types match, treating tuple and attributed_tuple as compatible.
    ///
    /// This allows attributed_tuple (used in entry point return types) to match
    /// plain tuple types. The attributes are metadata for code generation and don't
    /// affect type compatibility.
    fn types_match(&self, t1: &Type, t2: &Type) -> bool {
        // Apply current substitution without mutating context
        let a = t1.apply(&self.context);
        let b = t2.apply(&self.context);

        // Handle attributed_tuple vs tuple matching (symmetric)
        match (&a, &b) {
            // tuple matches attributed_tuple if component types match
            (
                Type::Constructed(TypeName::Tuple(_), types1),
                Type::Constructed(TypeName::Str("attributed_tuple"), types2),
            )
            | (
                Type::Constructed(TypeName::Str("attributed_tuple"), types1),
                Type::Constructed(TypeName::Tuple(_), types2),
            ) => {
                types1.len() == types2.len()
                    && types1.iter().zip(types2.iter()).all(|(t1, t2)| self.types_equal(t1, t2))
            }
            // Regular case - use structural equality after applying substitution
            _ => self.types_equal(&a, &b),
        }
    }
}
