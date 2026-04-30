use super::{SkolemId, Type, TypeExt, TypeName, TypeScheme};
use crate::ast::*;
use crate::error::{CompilerError, Result};
use crate::interface::{StorageDecl, UniformDecl};
use crate::scope::{IdentifierKind, ScopeEntry, ScopeStack};
use crate::{bail_type_at, err_module, err_type, err_type_at, err_undef_at};
use log::debug;
use polytype::Context;
use std::collections::{BTreeSet, HashMap};

// Import type helper functions from parent module
use super::{
    as_arrow, bool_type, f32, function, i32, mat, record, sized_array, strip_unique, tuple, unit, vec,
};

/// Render a single swizzle slot index as its `xyzw` letter. Used by
/// VecWith diagnostics so error messages match the user's source.
fn swizzle_letter(idx: u8) -> char {
    match idx {
        0 => 'x',
        1 => 'y',
        2 => 'z',
        3 => 'w',
        _ => '?',
    }
}

/// Render a swizzle component list (`[1, 2]`) as `yz`.
fn format_swizzle_str(components: &[u8]) -> String {
    components.iter().map(|&c| swizzle_letter(c)).collect()
}

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
    scope_stack: ScopeStack<ScopeEntry<TypeScheme>>,
    context: Context<TypeName>, // Polytype unification context
    record_field_map: HashMap<(String, String), Type>, // Map (type_name, field_name) -> field_type
    impl_source: crate::impl_source::ImplSource, // Implementation source for code generation
    intrinsics: crate::intrinsics::IntrinsicSource, // Type registry for polymorphic builtins
    module_manager: &'a crate::module_manager::ModuleManager, // Lazy module loading
    type_table: HashMap<crate::ast::NodeId, TypeScheme>, // Maps NodeId to type scheme
    warnings: Vec<TypeWarning>, // Collected warnings
    type_holes: Vec<(NodeId, Span)>, // Track type hole locations for warning emission
    arity_map: HashMap<String, usize>, // function name -> required arity (number of params)
    /// ID source for generating unique skolem constants when opening existential types.
    skolem_ids: crate::IdSource<SkolemId>,
    /// Current module context for resolving unqualified type aliases in expressions.
    /// Set during check_decl_as_in_module for module function checking.
    current_module: Option<String>,
    /// Cached module function schemes (key: "module.function", e.g., "rand.init").
    /// Populated during check_module_functions to avoid rebuilding schemes on each lookup.
    module_schemes: HashMap<String, TypeScheme>,
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

/// Free type variables, skipping any that occur in the size or variant
/// positions of Array types — so let-polymorphism cannot accidentally
/// abstract over representation-critical array metadata. Size and
/// variant are compile-time invariants that must be pinned by
/// unification at the definition, not re-instantiated at every use.
/// Without this, a sized array flowing through a let binding gets its
/// size generalized, every use instantiates a fresh size var,
/// unification pins the per-use var, and the original definition stays
/// unresolved all the way into monomorphization / SPIR-V lowering.
fn fv_type_generalizable(ty: &Type) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    fn go(t: &Type, acc: &mut BTreeSet<usize>) {
        match t {
            Type::Variable(n) => {
                acc.insert(*n);
            }
            Type::Constructed(TypeName::Array, args) => {
                // Only element type position is generalizable. Wyn's Array layout
                // is [elem, size, variant]; size and variant are compile-time
                // invariants that must be concrete before monomorphization.
                if let Some(elem) = args.first() {
                    go(elem, acc);
                }
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
}

/// Result of resolving a value name (identifier or qualified name).
/// Centralizes name resolution to avoid precedence drift between different call sites.
struct ResolvedValue {
    /// Display name for error messages
    display_name: String,
    /// Type scheme to store in type_table at the identifier node
    scheme_for_table: TypeScheme,
    /// Instantiated monotype (with fresh type variables)
    instantiated: Type,
    /// For overloaded intrinsics: all available schemes (for callee resolution)
    overloads: Option<Vec<TypeScheme>>,
}

/// Unified scheme lookup result, matching the pattern from IntrinsicLookup.
/// All scheme providers (scope, intrinsics, modules) feed into this.
enum SchemeLookup {
    Single(TypeScheme),
    Overloaded(Vec<TypeScheme>),
}

impl<'a> TypeChecker<'a> {
    /// Try to extract a constant integer value from an expression.
    /// Returns None if the expression is not a constant.
    fn try_extract_const_int(expr: &Expression) -> Option<i32> {
        match &expr.kind {
            ExprKind::IntLiteral(n) => i32::try_from(n).ok(),
            _ => None,
        }
    }

    /// Try to extract constant size from a slice (e.g., `arr[0..4]` → size 4).
    /// Returns None if end is missing or bounds aren't constant.
    // TODO: Currently AST constant folding does not cross function boundaries,
    // so slices with bounds from function parameters won't have constant sizes.
    fn try_extract_slice_size(slice: &SliceExpr) -> Option<usize> {
        let start = match &slice.start {
            Some(s) => Self::try_extract_const_int(s)?,
            None => 0,
        };
        let end = Self::try_extract_const_int(slice.end.as_ref()?)?;
        if end >= start { Some((end - start) as usize) } else { None }
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

    /// Constrain the address space of an array type to Storage.
    /// Used for entry point parameters where []f32 means storage buffer.
    /// Sized arrays (e.g. [19]u32) stay Composite — only unsized arrays become View.
    fn constrain_array_to_storage(&mut self, ty: &Type) -> Result<()> {
        let resolved = ty.apply(&self.context);
        if !resolved.is_array() {
            return Ok(());
        }

        let size = resolved.array_size().expect("Array has size");
        let variant = resolved.array_variant().expect("Array has variant");
        let elem = resolved.elem_type().expect("Array has elem");

        // Check if this is a sized array (Size constant)
        let is_sized = matches!(size.apply(&self.context), Type::Constructed(TypeName::Size(_), _));

        if is_sized {
            // Sized arrays → Composite (actual data, passed as push constants)
            let composite = Type::Constructed(TypeName::ArrayVariantComposite, vec![]);
            self.context
                .unify(variant, &composite)
                .map_err(|_| err_type!("Sized entry point array must be Composite"))?;
        } else {
            // Unsized arrays → View (storage buffer)
            let storage = Type::Constructed(TypeName::ArrayVariantView, vec![]);
            self.context
                .unify(variant, &storage)
                .map_err(|_| err_type!("Entry point array parameter must have Storage address space"))?;
        }

        // Recursively constrain nested arrays in element type
        self.constrain_array_to_storage(elem)?;
        Ok(())
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

    /// Check if a type is numeric (Int, UInt, Float, or Vec with numeric elements).
    /// Returns None if the type is a variable (not yet resolved).
    fn is_numeric_type(ty: &Type) -> Option<bool> {
        match ty {
            Type::Constructed(TypeName::Int(_), _) => Some(true),
            Type::Constructed(TypeName::UInt(_), _) => Some(true),
            Type::Constructed(TypeName::Float(_), _) => Some(true),
            // Vec types are numeric if their element type is numeric
            _ if ty.is_vec() => Self::is_numeric_type(ty.elem_type().expect("Vec has elem")),
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

    /// Insert a name into the scope with a given identifier kind.
    fn define(&mut self, name: String, kind: IdentifierKind, value: TypeScheme) {
        self.scope_stack.insert(name, ScopeEntry { kind, value });
    }

    /// Look up a variable in the scope stack (for testing)
    pub fn lookup(&self, name: &str) -> Option<TypeScheme> {
        self.scope_stack.lookup(name).map(|e| e.value.clone())
    }

    /// Get a reference to the context (for testing)
    pub fn context(&self) -> &Context<TypeName> {
        &self.context
    }

    /// Compute all free type variables in the current environment (scope stack)
    fn env_free_type_vars(&self) -> BTreeSet<usize> {
        let mut acc = BTreeSet::new();
        self.scope_stack.for_each_binding(|_name, entry| {
            acc.extend(fv_scheme(&entry.value));
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

        // Free vars in type — but skip size / variant positions of Array
        // types. Those are compile-time invariants that must be pinned by
        // unification, not let-generalized (see `fv_type_generalizable`).
        let mut fv_ty = fv_type_generalizable(&applied);

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

    /// Unified name resolution for identifiers and qualified names.
    ///
    /// Precedence (enforced consistently across all call sites):
    /// - Unqualified names: scope > intrinsics (locals shadow builtins)
    /// - Qualified names: intrinsics > modules (explicit qualification)
    ///
    /// Returns `None` if the name is not found (caller should produce error with span).
    fn resolve_value_name(&mut self, full_name: &str, is_qualified: bool) -> Option<ResolvedValue> {
        let lookup = if is_qualified {
            // Qualified: intrinsics > modules (cache, then on-demand)
            self.lookup_intrinsic(full_name).or_else(|| self.lookup_module_scheme(full_name))
        } else {
            // Unqualified: scope > intrinsics
            self.scope_stack
                .lookup(full_name)
                .map(|e| SchemeLookup::Single(e.value.clone()))
                .or_else(|| self.lookup_intrinsic(full_name))
        };

        lookup.map(|scheme_lookup| self.resolve_scheme_lookup(full_name, scheme_lookup))
    }

    fn lookup_module_scheme(&self, qualified_name: &str) -> Option<SchemeLookup> {
        self.module_schemes.get(qualified_name).cloned().map(SchemeLookup::Single)
    }

    fn lookup_intrinsic(&mut self, name: &str) -> Option<SchemeLookup> {
        use crate::intrinsics::IntrinsicLookup;
        self.intrinsics.get(name).map(|lookup| match lookup {
            IntrinsicLookup::Single(entry) => SchemeLookup::Single(entry.scheme.clone()),
            IntrinsicLookup::Overloaded(set) => {
                SchemeLookup::Overloaded(set.entries().iter().map(|e| e.scheme.clone()).collect())
            }
        })
    }

    fn resolve_scheme_lookup(&mut self, name: &str, lookup: SchemeLookup) -> ResolvedValue {
        match lookup {
            SchemeLookup::Single(scheme) => {
                let ty = scheme.instantiate(&mut self.context);
                ResolvedValue {
                    display_name: name.to_string(),
                    scheme_for_table: scheme,
                    instantiated: ty,
                    overloads: None,
                }
            }
            SchemeLookup::Overloaded(schemes) => {
                let ty = self.context.new_variable();
                ResolvedValue {
                    display_name: name.to_string(),
                    scheme_for_table: TypeScheme::Monotype(ty.clone()),
                    instantiated: ty,
                    overloads: Some(schemes),
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

    /// Create a TypeChecker with an existing Context and spec_schemes (from resolve_placeholders pass).
    pub fn with_context_and_schemes(
        module_manager: &'a crate::module_manager::ModuleManager,
        context: Context<TypeName>,
        spec_schemes: HashMap<String, TypeScheme>,
    ) -> Self {
        Self::with_context_and_type_table(module_manager, context, HashMap::new(), spec_schemes)
    }

    /// Create a TypeChecker with a given initial type table
    fn with_type_table(
        module_manager: &'a crate::module_manager::ModuleManager,
        type_table: HashMap<NodeId, TypeScheme>,
    ) -> Self {
        Self::with_context_and_type_table(module_manager, Context::default(), type_table, HashMap::new())
    }

    /// Create a TypeChecker with both an existing Context and type table.
    fn with_context_and_type_table(
        module_manager: &'a crate::module_manager::ModuleManager,
        mut context: Context<TypeName>,
        type_table: HashMap<NodeId, TypeScheme>,
        spec_schemes: HashMap<String, TypeScheme>,
    ) -> Self {
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
            skolem_ids: crate::IdSource::new(),
            current_module: None,
            module_schemes: spec_schemes,
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
                self.normalize_annotation_type(annotated_type, self.current_module.as_deref())
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
                self.define(name.clone(), IdentifierKind::Local, type_scheme);
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
                let normalized =
                    self.normalize_annotation_type(annotated_type, self.current_module.as_deref());
                // Unify annotation with expected type
                self.context.unify(&normalized, expected_type).map_err(|_| {
                    err_type_at!(
                        pattern.h.span,
                        "Pattern type annotation {} doesn't match expected type {}",
                        self.format_type(&normalized),
                        self.format_type(expected_type)
                    )
                })?;
                // Bind the inner pattern
                let result = self.bind_pattern(inner_pattern, &normalized, generalize)?;
                let resolved = normalized.apply(&self.context);
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
            PatternKind::Constructor(name, args) => {
                // The expected type must be a sum that contains this
                // constructor at matching payload arity.
                let expected_applied = expected_type.apply(&self.context);
                match expected_applied {
                    Type::Constructed(TypeName::Sum(ref variants), _) => {
                        let payload_types = match variants.iter().find(|(n, _)| n == name) {
                            Some((_, payload)) => payload,
                            None => bail_type_at!(
                                pattern.h.span,
                                "constructor `#{}` not found in sum type {}",
                                name,
                                self.format_type(&expected_applied)
                            ),
                        };
                        if args.len() != payload_types.len() {
                            bail_type_at!(
                                pattern.h.span,
                                "constructor `#{}` expects {} payload value{}, got {}",
                                name,
                                payload_types.len(),
                                if payload_types.len() == 1 { "" } else { "s" },
                                args.len()
                            );
                        }
                        for (sub_pattern, payload_ty) in args.iter().zip(payload_types.iter()) {
                            self.bind_pattern(sub_pattern, payload_ty, generalize)?;
                        }
                        self.type_table.insert(
                            pattern.h.id,
                            TypeScheme::Monotype(expected_type.apply(&self.context)),
                        );
                        Ok(expected_type.clone())
                    }
                    _ => Err(err_type_at!(
                        pattern.h.span,
                        "constructor pattern `#{}` requires a sum-typed scrutinee, got {}",
                        name,
                        self.format_type(&expected_applied)
                    )),
                }
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
            // Sum-type constructor application uses the expected type to
            // resolve which sum type the constructor belongs to.
            // Bare-inference (`infer_expression`) can't disambiguate.
            ExprKind::Constructor(name, args) => {
                let applied = expected_type.apply(&self.context);
                let variants = match &applied {
                    Type::Constructed(TypeName::Sum(variants), _) => variants,
                    _ => bail_type_at!(
                        expr.h.span,
                        "constructor `#{}` requires a sum type from context, but expected {}",
                        name,
                        self.format_type(&applied)
                    ),
                };
                let payload_types = match variants.iter().find(|(n, _)| n == name) {
                    Some((_, payload)) => payload,
                    None => bail_type_at!(
                        expr.h.span,
                        "constructor `#{}` not found in sum type {}",
                        name,
                        self.format_type(&applied)
                    ),
                };
                if args.len() != payload_types.len() {
                    bail_type_at!(
                        expr.h.span,
                        "constructor `#{}` expects {} payload value{}, got {}",
                        name,
                        payload_types.len(),
                        if payload_types.len() == 1 { "" } else { "s" },
                        args.len()
                    );
                }
                for (arg, expected_arg_ty) in args.iter().zip(payload_types.iter()) {
                    self.check_expression(arg, expected_arg_ty)?;
                }
                self.type_table.insert(expr.h.id, TypeScheme::Monotype(expected_type.clone()));
                Ok(expected_type.clone())
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

    /// Resolve type aliases in a type annotation. SizeVar/UserVar
    /// substitution is handled upstream by `resolve_placeholders`.
    fn normalize_annotation_type(&self, ty: &Type, module: Option<&str>) -> Type {
        self.resolve_type_aliases_scoped(ty, module)
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

    /// Apply the checker's substitution context to the innermost
    /// `Monotype` of a scheme, preserving the surrounding `Polytype`
    /// shells. `forall` builds one shell per quantified variable, so a
    /// scheme with N type variables walks through N `Polytype` cases
    /// before reaching the `Monotype`.
    fn apply_scheme(&self, scheme: &TypeScheme) -> TypeScheme {
        match scheme {
            TypeScheme::Monotype(ty) => TypeScheme::Monotype(ty.apply(&self.context)),
            TypeScheme::Polytype { variable, body } => TypeScheme::Polytype {
                variable: *variable,
                body: Box::new(self.apply_scheme(body)),
            },
        }
    }

    /// Walk through any number of `Polytype` shells to the innermost
    /// `Monotype` and return it with `self.context` applied.
    fn scheme_inner(&self, scheme: &TypeScheme) -> Type {
        match scheme {
            TypeScheme::Monotype(ty) => ty.apply(&self.context),
            TypeScheme::Polytype { body, .. } => self.scheme_inner(body),
        }
    }

    /// Build an array type: Array[elem, size, variant]
    fn array_ty(elem: Type, addrspace: polytype::Variable, size: polytype::Variable) -> Type {
        Type::Constructed(TypeName::Array, vec![elem, Self::var(size), Self::var(addrspace)])
    }

    /// Register zipN: ∀n a1..aN s. [a1,s,n] -> ... -> [aN,s,n] -> [(a1,...,aN),s,n]
    fn register_zip_n(&mut self, arity: usize) {
        let n = self.fresh_var();
        let s = self.fresh_var();
        let elem_vars: Vec<polytype::Variable> = (0..arity).map(|_| self.fresh_var()).collect();

        let params: Vec<Type> = elem_vars.iter().map(|&v| Self::array_ty(Self::var(v), s, n)).collect();
        let tuple_ty = Type::Constructed(
            TypeName::Tuple(arity),
            elem_vars.iter().map(|&v| Self::var(v)).collect(),
        );
        let ret = Self::array_ty(tuple_ty, s, n);
        let body = Self::arrow_chain(&params, ret);

        let mut all_vars = vec![n, s];
        all_vars.extend(&elem_vars);
        self.define(
            format!("zip{}", arity),
            IdentifierKind::Builtin,
            Self::forall(&all_vars, body),
        );
    }

    /// Build a Vec type: Vec[elem, size]
    fn vec_ty(n: polytype::Variable, elem: Type) -> Type {
        Type::Constructed(TypeName::Vec, vec![elem, Self::var(n)])
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
                self.normalize_annotation_type(annotated_type, self.current_module.as_deref())
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

        self.scope_stack.pop_scope();

        // Build the function type
        let func_type = Self::arrow_chain(&param_types, body_type);

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

    fn define_builtin(&mut self, name: &str, scheme: TypeScheme) {
        self.define(name.to_string(), IdentifierKind::Builtin, scheme);
    }

    pub fn load_builtins(&mut self) -> Result<()> {
        // length: ∀n a s. Array[a, s, n] -> i32
        let (n, a, s) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let body = Self::arrow_chain(&[Self::array_ty(Self::var(a), s, n)], i32());
        self.define_builtin("length", Self::forall(&[n, a, s], body));

        // map: ∀a b n s. (a -> b) -> Array[a, s, n] -> Array[b, n, Composite]
        //
        // The output variant is pinned to Composite because `egir::soac_expand`
        // always materializes the map result via `_w_intrinsic_uninit` +
        // `_w_intrinsic_array_with_inplace`. Preserving the input variant `s`
        // in the output type would be a lie: post-expand the representation
        // is always a composite buffer. The lie leaked into loop back-edges
        // — a loop carrying `map(…, iota(N))` ends up with a Virtual-variant
        // block parameter that SPIR-V `ArrayWith` can't write to.
        let (a, b, n, s) = (
            self.fresh_var(),
            self.fresh_var(),
            self.fresh_var(),
            self.fresh_var(),
        );
        let composite = Type::Constructed(TypeName::ArrayVariantComposite, vec![]);
        let body = Self::arrow_chain(
            &[
                Type::arrow(Self::var(a), Self::var(b)),
                Self::array_ty(Self::var(a), s, n),
            ],
            Type::Constructed(TypeName::Array, vec![Self::var(b), Self::var(n), composite]),
        );
        self.define_builtin("map", Self::forall(&[a, b, n, s], body));

        // zip: ∀n a b s. Array[a, s, n] -> Array[b, s, n] -> Array[(a, b), s, n]
        let (n, a, b, s) = (
            self.fresh_var(),
            self.fresh_var(),
            self.fresh_var(),
            self.fresh_var(),
        );
        let body = Self::arrow_chain(
            &[
                Self::array_ty(Self::var(a), s, n),
                Self::array_ty(Self::var(b), s, n),
            ],
            Self::array_ty(tuple(vec![Self::var(a), Self::var(b)]), s, n),
        );
        self.define_builtin("zip", Self::forall(&[n, a, b, s], body));

        // zip3..zip5: ∀n a1..aN s. [a1,s,n] -> ... -> [aN,s,n] -> [(a1,...,aN),s,n]
        for arity in 3..=5 {
            self.register_zip_n(arity);
        }

        // to_vec: ∀n a s. Array[a, s, n] -> Vec(n, a)
        let (n, a, s) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let body = Self::arrow_chain(
            &[Self::array_ty(Self::var(a), s, n)],
            Self::vec_ty(n, Self::var(a)),
        );
        self.define_builtin("to_vec", Self::forall(&[n, a, s], body));

        // replicate: ∀size a s. i32 -> a -> Array[a, s, size]
        let (size, a, s) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let body = Self::arrow_chain(&[i32(), Self::var(a)], Self::array_ty(Self::var(a), s, size));
        self.define_builtin("replicate", Self::forall(&[size, a, s], body));

        // reduce: ∀a n s. (a -> a -> a) -> a -> Array[a, s, n] -> a
        let (a, n, s) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let op_ty = Type::arrow(Self::var(a), Type::arrow(Self::var(a), Self::var(a)));
        let body = Self::arrow_chain(
            &[op_ty, Self::var(a), Self::array_ty(Self::var(a), s, n)],
            Self::var(a),
        );
        self.define_builtin("reduce", Self::forall(&[a, n, s], body));

        // scan: ∀a n s. (a -> a -> a) -> a -> Array[a, s, n] -> Array[a, n, Composite]
        // Output variant pinned to Composite for the same reason as `map` above.
        let (a, n, s) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let op_ty = Type::arrow(Self::var(a), Type::arrow(Self::var(a), Self::var(a)));
        let composite = Type::Constructed(TypeName::ArrayVariantComposite, vec![]);
        let body = Self::arrow_chain(
            &[op_ty, Self::var(a), Self::array_ty(Self::var(a), s, n)],
            Type::Constructed(TypeName::Array, vec![Self::var(a), Self::var(n), composite]),
        );
        self.define_builtin("scan", Self::forall(&[a, n, s], body));

        // filter: ∀a n s. (a -> bool) -> Array[a, s, n] -> ?k. Array[a, k, Composite]
        // Output variant pinned to Composite for the same reason as `map` / `scan`.
        let (a, n, s) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
        let pred_ty = Type::arrow(Self::var(a), bool_ty);
        let array_a = Self::array_ty(Self::var(a), s, n);
        // Existential return type: ?k. Array[a, k, Composite]
        let k = "k".to_string();
        let k_var = Type::Constructed(TypeName::SizeVar(k.clone()), vec![]);
        let composite = Type::Constructed(TypeName::ArrayVariantComposite, vec![]);
        let result_array = Type::Constructed(TypeName::Array, vec![Self::var(a), k_var, composite]);
        let existential_result = Type::Constructed(TypeName::Existential(vec![k]), vec![result_array]);
        let body = Self::arrow_chain(&[pred_ty, array_a], existential_result);
        self.define_builtin("filter", Self::forall(&[a, n, s], body));

        // scatter: ∀a n m s1 s2 s3. Array[a, s1, n] -> Array[i32, s2, m] -> Array[a, s3, m] -> Array[a, s1, n]
        let (a, n, m) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let (s1, s2, s3) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let dest_array = Self::array_ty(Self::var(a), s1, n);
        let indices_array = Self::array_ty(i32(), s2, m);
        let values_array = Self::array_ty(Self::var(a), s3, m);
        let body = Self::arrow_chain(&[dest_array.clone(), indices_array, values_array], dest_array);
        self.define_builtin("scatter", Self::forall(&[a, n, m, s1, s2, s3], body));

        // reduce_by_index: ∀a n m s1 s2 s3. Array[a, s1, n] -> (a -> a -> a) -> a -> Array[i32, s2, m] -> Array[a, s3, m] -> Array[a, s1, n]
        let (a, n, m) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let (s1, s2, s3) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let op_ty = Type::arrow(Self::var(a), Type::arrow(Self::var(a), Self::var(a)));
        let dest_array = Self::array_ty(Self::var(a), s1, n);
        let indices_array = Self::array_ty(i32(), s2, m);
        let values_array = Self::array_ty(Self::var(a), s3, m);
        let body = Self::arrow_chain(
            &[
                dest_array.clone(),
                op_ty,
                Self::var(a),
                indices_array,
                values_array,
            ],
            dest_array,
        );
        self.define_builtin("reduce_by_index", Self::forall(&[a, n, m, s1, s2, s3], body));

        // _w_alloc_array: ∀n t s. i32 -> Array[t, s, n]
        let (n, t, s) = (self.fresh_var(), self.fresh_var(), self.fresh_var());
        let body = Self::arrow_chain(&[i32()], Self::array_ty(Self::var(t), s, n));
        self.define_builtin("_w_alloc_array", Self::forall(&[n, t, s], body));

        // dot: ∀n t. Vec(n, t) -> Vec(n, t) -> t
        let (n, t) = (self.fresh_var(), self.fresh_var());
        let vec = Self::vec_ty(n, Self::var(t));
        let body = Self::arrow_chain(&[vec.clone(), vec], Self::var(t));
        self.define_builtin("dot", Self::forall(&[n, t], body));

        // Math functions: ∀t. t -> t (works on f32, vec2f32, vec3f32, vec4f32).
        // `sqrt`/`abs`/`floor`/`ceil`/`fract` keep a top-level binding because
        // they're still in `INTRINSIC_RENAMES` (no per-type-only home).
        let t = self.fresh_var();
        let math_unary = Self::forall(&[t], Type::arrow(Self::var(t), Self::var(t)));
        for name in &["abs", "floor", "ceil", "fract"] {
            self.define_builtin(name, math_unary.clone());
        }
        // sin/cos/tan/asin/acos/atan/sqrt/exp/log/radians/degrees and the
        // binary `pow`/`atan2`/`mod` no longer have top-level bindings —
        // bare names only resolve via `open f32` (scalar) or `open vec`
        // (vector). The schemes for `f32.cos`/`vec.cos`/etc. come from
        // prelude module sigs and `IntrinsicSource::register_vec_module_ops`.

        // Register vector field mappings
        self.register_vector_fields();

        // Note: Prelude files are automatically loaded when ModuleManager is created

        Ok(())
    }

    /// Names registered by `load_builtins`, queried from the scope by kind.
    pub fn builtin_names(&self) -> Vec<String> {
        self.scope_stack.names_by_kind(IdentifierKind::Builtin)
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
        // Type-check module functions first to populate the module_schemes cache.
        // This must happen before prelude functions since they may reference module functions.
        self.check_module_functions()?;

        // Type-check prelude functions (needed so they're in scope for user code)
        self.check_prelude_functions()?;

        // Process user declarations
        for decl in &program.declarations {
            self.check_declaration(decl)?;
        }

        // Emit warnings for all type holes now that types are fully inferred
        self.emit_hole_warnings();

        // Apply the context to all types in the type table to resolve type variables
        let resolved_table: HashMap<crate::ast::NodeId, TypeScheme> =
            self.type_table.iter().map(|(node_id, scheme)| (*node_id, self.apply_scheme(scheme))).collect();

        Ok(resolved_table)
    }

    /// Emit warnings for all type holes showing their inferred types
    fn emit_hole_warnings(&mut self) {
        // Clone the holes list to avoid borrow checker issues
        let holes = self.type_holes.clone();
        for (node_id, span) in holes {
            if let Some(hole_scheme) = self.type_table.get(&node_id) {
                self.warnings.push(TypeWarning::TypeHoleFilled {
                    inferred_type: self.scheme_inner(hole_scheme),
                    span,
                });
            }
        }
    }

    /// Helper to type check a function body with parameters in scope
    /// Returns (param_types, body_type)
    fn check_function_with_params(
        &mut self,
        params: &[Pattern],
        body: &Expression,
        module_name: Option<&str>,
    ) -> Result<(Vec<Type>, Type)> {
        self.check_function_with_params_inner(params, body, module_name, false)
    }

    fn check_entry_with_params(
        &mut self,
        params: &[Pattern],
        body: &Expression,
    ) -> Result<(Vec<Type>, Type)> {
        self.check_function_with_params_inner(params, body, None, true)
    }

    fn check_function_with_params_inner(
        &mut self,
        params: &[Pattern],
        body: &Expression,
        module_name: Option<&str>,
        is_entry: bool,
    ) -> Result<(Vec<Type>, Type)> {
        // Create type variables or use explicit types for parameters
        let param_types: Vec<Type> = params
            .iter()
            .map(|p| {
                let ty = p.pattern_type().cloned().unwrap_or_else(|| self.context.new_variable());
                self.normalize_annotation_type(&ty, module_name)
            })
            .collect();

        // For entry point parameters, constrain array address spaces to Storage
        if is_entry {
            for param_type in &param_types {
                self.constrain_array_to_storage(param_type)?;
            }
        }

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
            self.define(param_name, IdentifierKind::Local, type_scheme);
        }

        // Infer body type
        let body_type = self.infer_expression(body)?;

        // Pop parameter scope
        self.scope_stack.pop_scope();

        Ok((param_types, body_type))
    }

    /// Type-check function bodies from modules (e.g., rand.init, rand.int, f32.pi)
    /// This populates the type table so these functions can be flattened to MIR.
    /// Note: module_schemes for Spec::Sig items are pre-built by resolve_placeholders
    /// and passed to the constructor.
    pub fn check_module_functions(&mut self) -> Result<()> {
        // Collect all module declarations that need flattening (includes constants like f32.pi)
        let module_functions: Vec<(String, crate::ast::Decl)> = self
            .module_manager
            .get_all_module_declarations()
            .into_iter()
            .map(|(module_name, decl)| (module_name.to_string(), decl.clone()))
            .collect();

        // Type-check each module function with module context for alias resolution
        for (module_name, decl) in module_functions {
            let qualified_name = format!("{}.{}", module_name, decl.name);
            debug!("Type-checking module function: {}", qualified_name);

            // Type-check the declaration with module context for alias resolution
            self.check_decl_as_in_module(&decl, &qualified_name, Some(&module_name))?;

            // Cache the scheme for fast lookup during name resolution
            if let Some(entry) = self.scope_stack.lookup(&qualified_name) {
                self.module_schemes.insert(qualified_name, entry.value.clone());
            }
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
        self.scope_stack.for_each_binding(|name, entry| {
            let resolved = self.apply_context_to_scheme(&entry.value);
            schemes.insert(name.to_string(), resolved);
        });
        schemes
    }

    /// Apply the unification context to a TypeScheme, resolving type variables.
    fn apply_context_to_scheme(&self, scheme: &TypeScheme) -> TypeScheme {
        match scheme {
            TypeScheme::Monotype(ty) => TypeScheme::Monotype(ty.apply(&self.context)),
            TypeScheme::Polytype { variable, body } => {
                // Check if this variable was resolved to a concrete type
                let var_ty = Type::Variable(*variable);
                let resolved = var_ty.apply(&self.context);
                if let Type::Variable(v) = resolved {
                    if v == *variable {
                        // Variable is still free, keep the quantifier
                        TypeScheme::Polytype {
                            variable: *variable,
                            body: Box::new(self.apply_context_to_scheme(body)),
                        }
                    } else {
                        // Variable was unified with another variable, use the resolved ID
                        TypeScheme::Polytype {
                            variable: v,
                            body: Box::new(self.apply_context_to_scheme(body)),
                        }
                    }
                } else {
                    // Variable was resolved to a concrete type, remove the quantifier
                    // and substitute in the body
                    self.apply_context_to_scheme(body)
                }
            }
        }
    }

    /// Get a module function scheme from the cache.
    /// Requires `check_module_functions()` to have been called first.
    pub fn get_module_scheme(&self, qualified_name: &str) -> Option<&TypeScheme> {
        self.module_schemes.get(qualified_name)
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
        self.scope_stack.for_each_binding(|name, entry| {
            schemes.insert(name.to_string(), entry.value.clone());
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
                let (_param_types, body_type) = self.check_entry_with_params(&entry.params, &entry.body)?;
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
            Declaration::Module(_) => {
                // Module/functor declarations should be elaborated away before type checking
                // If we encounter one here, it means elaboration wasn't run or failed
                Err(err_module!(
                    "Module declarations should be elaborated before type checking"
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
            Declaration::Extern(extern_decl) => {
                debug!("Checking Extern declaration: {}", extern_decl.name);
                self.check_extern_decl(extern_decl)
            }
        }
    }

    fn check_uniform_decl(&mut self, decl: &UniformDecl) -> Result<()> {
        // Add the uniform to scope with its declared type
        let type_scheme = TypeScheme::Monotype(decl.ty.clone());
        self.define(decl.name.clone(), IdentifierKind::UserDecl, type_scheme);
        debug!("Inserting uniform variable '{}' into scope", decl.name);
        Ok(())
    }

    fn check_storage_decl(&mut self, decl: &StorageDecl) -> Result<()> {
        // Add the storage buffer to scope with its declared type
        let type_scheme = TypeScheme::Monotype(decl.ty.clone());
        self.define(decl.name.clone(), IdentifierKind::UserDecl, type_scheme);
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

        // Note: SizeVar/UserVar substitution is now handled by resolve_placeholders pass
        // before type checking. Type parameter names are already converted to type variables.

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
            self.define(scope_name.to_string(), IdentifierKind::UserDecl, type_scheme);
            debug!("Inferred type for {}: {}", scope_name, stored_type);
        } else {
            // Function declaration: let/def name param1 param2 = body

            let (param_types, body_type) =
                self.check_function_with_params(&decl.params, &decl.body, module_name)?;
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
                let normalized_return_type = self.normalize_annotation_type(declared_type, module_name);

                // When a function has parameters, decl.ty is just the return type annotation
                // Unify the body type with the declared return type
                if !decl.params.is_empty() {
                    self.context.unify(&body_type, &normalized_return_type).map_err(|e| {
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
                    self.context.unify(&body_type, &normalized_return_type).map_err(|_| {
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

            // Entry points go through `Declaration::Entry`; `Decl` has
            // no attributed return types.

            // Update scope with inferred type using generalization
            let type_scheme = self.generalize(&func_type);
            self.define(scope_name.to_string(), IdentifierKind::UserDecl, type_scheme);

            // Track arity for partial application checking
            self.arity_map.insert(scope_name.to_string(), decl.params.len());

            debug!("Inferred type for {}: {}", scope_name, func_type);
        }

        // Restore previous module context
        self.current_module = saved_module;

        Ok(())
    }

    fn check_sig_decl(&mut self, decl: &SigDecl) -> Result<()> {
        // Sig declarations are just type signatures - register them in scope
        let type_scheme = TypeScheme::Monotype(decl.ty.clone());
        self.define(decl.name.clone(), IdentifierKind::UserDecl, type_scheme);
        Ok(())
    }

    fn check_extern_decl(&mut self, decl: &ExternDecl) -> Result<()> {
        // Extern declarations register a type signature for a linked SPIR-V function
        let type_scheme = TypeScheme::Monotype(decl.ty.clone());
        self.define(decl.name.clone(), IdentifierKind::UserDecl, type_scheme);
        debug!(
            "Registered extern function '{}' with linkage '{}'",
            decl.name, decl.linkage_name
        );
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
            ExprKind::Unit => Ok(unit()),
            ExprKind::Identifier(quals, name) => {
                let full_name = if quals.is_empty() {
                    name.clone()
                } else {
                    format!("{}.{}", quals.join("."), name)
                };
                let is_qualified = !quals.is_empty();

                debug!("Looking up identifier '{}'", full_name);

                if let Some(resolved) = self.resolve_value_name(&full_name, is_qualified) {
                    // Store instantiated type with substitutions applied
                    let applied = resolved.instantiated.apply(&self.context);
                    self.type_table
                        .insert(expr.h.id, TypeScheme::Monotype(applied));
                    return Ok(resolved.instantiated);
                }

                debug!("Variable lookup failed for '{}' - not in scope, intrinsics, or prelude", full_name);
                return Err(err_undef_at!(expr.h.span, "{}", full_name));
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
                    // Array[elem_type, size, variant]
                    let resolved = first_type.apply(&self.context);
                    if resolved.is_array() {
                        if let Type::Constructed(TypeName::Size(cols), _) = resolved.array_size().expect("Array has size") {
                            let rows = elements.len();
                            let elem_type = resolved.elem_type().expect("Array has elem").clone();
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
                self.context.unify(&index_type, &i32()).map_err(|_| {
                    err_type_at!(
                        index_expr.h.span,
                        "Array index must be an integer type, got {}",
                        self.format_type(&index_type.apply(&self.context))
                    )
                })?;

                // Constrain array type - strip uniqueness (indexing *[n]T works like [n]T)
                let array_type_stripped = strip_unique(&array_type);
                let (elem_var, _, _) = self.constrain_array_type(
                    &array_type_stripped,
                    &array_expr.h.span,
                    "Cannot index non-array type",
                )?;

                Ok(elem_var.apply(&self.context))
            }
            ExprKind::ArrayWith { array, index, value, .. } => {
                // Type check: array must be Array[elem, addrspace, size], index must be i32, value must be elem
                // Result type is Array[elem, addrspace, size]
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

                // Constrain array type - strip uniqueness
                let array_type_stripped = strip_unique(&array_type);
                let (elem_var, _, _) = self.constrain_array_type(
                    &array_type_stripped,
                    &array.h.span,
                    "Cannot update non-array type",
                )?;

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
            ExprKind::VecWith {
                target,
                components,
                op,
                value,
            } => {
                // Target must resolve to a Vec<elem, size>.
                let target_type = self.infer_expression(target)?;
                let elem_var = self.context.new_variable();
                let size_var = self.context.new_variable();
                let want_vec =
                    Type::Constructed(TypeName::Vec, vec![elem_var.clone(), size_var.clone()]);
                self.context.unify(&target_type, &want_vec).map_err(|_| {
                    err_type_at!(
                        target.h.span,
                        "`with .swizzle` requires a vector target, got {}",
                        self.format_type(&target_type.apply(&self.context))
                    )
                })?;

                // Range-check components against the resolved vec size,
                // when known. (Parser already enforced distinctness.)
                let resolved_target = target_type.apply(&self.context);
                if let Type::Constructed(TypeName::Vec, args) = &resolved_target {
                    if let Some(Type::Constructed(TypeName::Size(n), _)) = args.get(1) {
                        for &c in components {
                            if (c as usize) >= *n {
                                bail_type_at!(
                                    expr.h.span,
                                    "swizzle component `{}` is out of range for vec{}",
                                    swizzle_letter(c),
                                    n
                                );
                            }
                        }
                    }
                }

                // What type does `target.swizzle` have? Single-component
                // is the elem type; multi-component is a vec of swizzle
                // length. This is what the lhs slot reads/writes.
                let elem_ty = elem_var.apply(&self.context);
                let swizzle_ty = if components.len() == 1 {
                    elem_ty.clone()
                } else {
                    Type::Constructed(
                        TypeName::Vec,
                        vec![
                            elem_ty.clone(),
                            Type::Constructed(TypeName::Size(components.len()), vec![]),
                        ],
                    )
                };

                let rhs_type = self.infer_expression(value)?;

                match op {
                    None => {
                        // Plain `=`: rhs must match the swizzle slot's type.
                        self.context.unify(&rhs_type, &swizzle_ty).map_err(|_| {
                            err_type_at!(
                                value.h.span,
                                "`with .{}` expects {}, got {}",
                                format_swizzle_str(components),
                                self.format_type(&swizzle_ty),
                                self.format_type(&rhs_type.apply(&self.context))
                            )
                        })?;
                    }
                    Some(binop) => {
                        // Compound `op=`: type `swizzle_ty <op> rhs` and
                        // require the result to equal swizzle_ty.
                        let combined = self.infer_binop_result(
                            binop,
                            swizzle_ty.clone(),
                            rhs_type,
                            expr.h.span,
                        )?;
                        self.context.unify(&combined, &swizzle_ty).map_err(|_| {
                            err_type_at!(
                                expr.h.span,
                                "compound `with .{} {}=` must produce {}, got {}",
                                format_swizzle_str(components),
                                binop,
                                self.format_type(&swizzle_ty),
                                self.format_type(&combined.apply(&self.context))
                            )
                        })?;
                    }
                }

                Ok(target_type.apply(&self.context))
            }
            ExprKind::BinaryOp(op, left, right) => {
                let left_type = self.infer_expression(left)?;
                let right_type = self.infer_expression(right)?;
                self.infer_binop_result(&op.op, left_type, right_type, expr.h.span)
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

                let resolved_annotation = let_in
                    .ty
                    .as_ref()
                    .map(|ty| self.normalize_annotation_type(ty, self.current_module.as_deref()));
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

                    // Store resolved type in type table (apply substitutions)
                    let resolved = cand.ty.apply(&self.context);
                    self.type_table
                        .insert(func.h.id, TypeScheme::Monotype(resolved));
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
                        let checkpoint = self.context.len();

                        if let Some(result_ty) =
                            Self::try_unify_overload(&cand.ty, &arg_types, &mut self.context)
                        {
                            // Check for partial application
                            if self.ensure_not_partial(&result_ty, &expr.h.span).is_ok() {
                                // Store resolved type (always apply substitutions)
                                let resolved_func_ty = cand.ty.apply(&self.context);
                                self.type_table
                                    .insert(func.h.id, TypeScheme::Monotype(resolved_func_ty));
                                self.type_table
                                    .insert(expr.h.id, TypeScheme::Monotype(result_ty.clone()));
                                return Ok(result_ty);
                            }
                        }

                        self.context.rollback(checkpoint);
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

                    if let Some(resolved) = self.resolve_value_name(&dotted, true) {
                        self.type_table.insert(expr.h.id, resolved.scheme_for_table);
                        return Ok(resolved.instantiated);
                    }
                    // Qualified name not found - fall through to field access
                }

                // Infer base expression type
                let base_type = self.infer_expression(inner_expr)?;

                // Apply context and strip uniqueness
                let base_type = base_type.apply(&self.context).strip_unique().clone();

                // Use unified field access helper
                let field_type = self.infer_field_access(&base_type, field, &expr.h.span)?;
                self.type_table.insert(expr.h.id, TypeScheme::Monotype(field_type.clone()));
                Ok(field_type)
            }
            ExprKind::If(if_expr) => {
                // Infer condition type - should be bool
                let condition_ty = self.infer_expression(&if_expr.condition)?;
                let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

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
                let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
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
                        self.define(var_name.clone(), IdentifierKind::Local, TypeScheme::Monotype(i32()));

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
                        // Array must be an array type: Array[elem, addrspace, size]
                        let arr_type = self.infer_expression(arr)?;
                        let (elem_type, _, _) =
                            self.constrain_array_type(&arr_type, &arr.h.span, "for-in requires an array")?;

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

            ExprKind::Match(match_expr) => {
                // Type the scrutinee. It must resolve to a sum type so we
                // can check arms against its variants.
                let scrutinee_ty = self.infer_expression(&match_expr.scrutinee)?;
                let scrutinee_applied = scrutinee_ty.apply(&self.context);
                let variants = match &scrutinee_applied {
                    Type::Constructed(TypeName::Sum(variants), _) => variants,
                    _ => bail_type_at!(
                        match_expr.scrutinee.h.span,
                        "match scrutinee must be a sum type, got {}",
                        self.format_type(&scrutinee_applied)
                    ),
                };

                // All arms produce the same type; pick a fresh variable
                // and unify each arm's body against it.
                let result_var = self.context.new_variable();

                let mut covered: std::collections::HashSet<&str> =
                    std::collections::HashSet::new();
                for case in &match_expr.cases {
                    self.scope_stack.push_scope();

                    // Track top-level constructor coverage for the
                    // exhaustiveness check below. (Wildcards / nested
                    // patterns aren't supported yet, so anything other
                    // than a top-level Constructor pattern is rejected.)
                    match &case.pattern.kind {
                        PatternKind::Constructor(name, _) => {
                            covered.insert(name.as_str());
                        }
                        _ => bail_type_at!(
                            case.pattern.h.span,
                            "match arms must be top-level `#name(...)` constructor patterns"
                        ),
                    }

                    self.bind_pattern(&case.pattern, &scrutinee_ty, false)?;
                    self.check_expression(&case.body, &result_var)?;

                    self.scope_stack.pop_scope();
                }

                // Exhaustiveness: every constructor in the scrutinee's
                // sum type must appear in some arm.
                for (name, _) in variants {
                    if !covered.contains(name.as_str()) {
                        bail_type_at!(
                            expr.h.span,
                            "non-exhaustive match: missing constructor `#{}`",
                            name
                        );
                    }
                }

                Ok(result_var)
            }

            // A bare `#name(args)` doesn't pin down which sum type it
            // belongs to — `#some(3)` is a value of any sum type that
            // declares a `#some` variant carrying an `i32` payload.
            // Use a type ascription, an annotated `let`, or context
            // (function argument, return type) to disambiguate, which
            // routes through `check_expression`'s Constructor arm.
            ExprKind::Constructor(name, _) => Err(err_type_at!(
                expr.h.span,
                "ambiguous constructor `#{}`: cannot determine which sum type it belongs to. \
                 Add a type annotation or use it where a sum type is expected.",
                name
            )),

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

                // Check start is an integer type (i32, u32, etc.)
                let resolved_start = start_type.apply(&self.context);
                if !crate::types::is_integer_type(&resolved_start) {
                    // If still a type variable, default to i32
                    if matches!(resolved_start, Type::Variable(_)) {
                        self.context.unify(&start_type, &i32()).map_err(|_| {
                            err_type_at!(
                                range.start.h.span,
                                "Range operands must be integer types, got {}",
                                self.format_type(&start_type.apply(&self.context))
                            )
                        })?;
                    } else {
                        return Err(err_type_at!(
                            range.start.h.span,
                            "Range operands must be integer types, got {}",
                            self.format_type(&resolved_start)
                        ));
                    }
                }

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
                // Range literals produce virtual arrays (struct {start, step, len})
                let addrspace = Type::Constructed(TypeName::ArrayVariantVirtual, vec![]);
                Ok(Type::Constructed(TypeName::Array, vec![elem_type, size_type, addrspace]))
            }

            ExprKind::Slice(slice) => {
                // Slice expression: array[start:end]
                // - array must be Array[elem, addrspace, size]
                // - start/end (if present) must be integers
                // - result is Array[elem, addrspace, size'] where size' = end - start

                let array_type = self.infer_expression(&slice.array)?;
                let array_type_stripped = strip_unique(&array_type);

                // Constrain array to be Array[elem, addrspace, size]
                let (elem_var, addrspace_var, _) = self.constrain_array_type(
                    &array_type_stripped,
                    &slice.array.h.span,
                    "Cannot slice non-array type",
                )?;

                // Type-check start bound if present
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

                // Type-check end bound if present
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

                // Compute result size from constant bounds, or use fresh variable
                let result_size = match Self::try_extract_slice_size(slice) {
                    Some(size) => Type::Constructed(TypeName::Size(size), vec![]),
                    None => self.context.new_variable(),
                };

                // Slice result: if the source is a view but the result has known size,
                // the elements will be materialized into a composite array.
                let elem_type = elem_var.apply(&self.context);
                let addrspace = addrspace_var.apply(&self.context);
                let result_variant = if super::is_array_variant_view(&addrspace)
                    && matches!(result_size, Type::Constructed(TypeName::Size(_), _))
                {
                    super::array_variant_composite()
                } else {
                    addrspace
                };
                Ok(Type::Constructed(TypeName::Array, vec![elem_type, result_size, result_variant]))
            }

            ExprKind::TypeAscription(expr, ascribed_ty) => {
                // Type ascription: check the inner expression against the ascribed type
                // This allows integer literals to take on the ascribed type (e.g., 42u32)
                let normalized =
                    self.normalize_annotation_type(ascribed_ty, self.current_module.as_deref());
                self.check_expression(expr, &normalized)?;
                Ok(normalized)
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
        match &func.kind {
            ExprKind::Identifier(quals, name) => {
                let full_name =
                    if quals.is_empty() { name.clone() } else { format!("{}.{}", quals.join("."), name) };
                let is_qualified = !quals.is_empty();

                if let Some(resolved) = self.resolve_value_name(&full_name, is_qualified) {
                    // For overloaded intrinsics, expand into multiple candidates
                    let candidates = if let Some(overloads) = resolved.overloads {
                        overloads
                            .into_iter()
                            .map(|scheme| {
                                let ty = scheme.instantiate(&mut self.context);
                                Candidate { ty }
                            })
                            .collect()
                    } else {
                        vec![Candidate {
                            ty: resolved.instantiated,
                        }]
                    };

                    return Ok(CalleeCandidates {
                        candidates,
                        display_name: resolved.display_name,
                    });
                }

                Err(err_undef_at!(func.h.span, "{}", full_name))
            }

            // Any non-identifier callee: infer its type as single candidate
            _ => {
                let ty = self.infer_expression(func)?;
                Ok(CalleeCandidates {
                    candidates: vec![Candidate { ty }],
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

    /// Unify two types or produce a "<context>, got A and B" type error.
    fn unify_or_err(&mut self, a: &Type, b: &Type, span: Span, ctx: &str) -> Result<()> {
        self.context.unify(a, b).map_err(|_| {
            err_type_at!(
                span,
                "{}, got {} and {}",
                ctx,
                self.format_type(&a.apply(&self.context)),
                self.format_type(&b.apply(&self.context))
            )
        })
    }

    /// Destructure a `Mat` type into `(elem, cols, rows)`. None for non-Mat.
    fn mat_parts(t: &Type) -> Option<(&Type, &Type, &Type)> {
        match t {
            Type::Constructed(TypeName::Mat, args) if args.len() == 3 => {
                Some((&args[0], &args[1], &args[2]))
            }
            _ => None,
        }
    }

    /// Destructure a `Vec` type into `(elem, size)`. None for non-Vec.
    fn vec_parts(t: &Type) -> Option<(&Type, &Type)> {
        match t {
            Type::Constructed(TypeName::Vec, args) if args.len() == 2 => Some((&args[0], &args[1])),
            _ => None,
        }
    }

    /// Compute the result type of a binary operator given pre-inferred
    /// operand types. Shared between `ExprKind::BinaryOp` and the
    /// compound-form arm of `ExprKind::VecWith`, which needs to know
    /// the type of `target.swizzle <op> rhs` without rebuilding an AST.
    fn infer_binop_result(
        &mut self,
        op: &str,
        left_type: Type,
        right_type: Type,
        span: Span,
    ) -> Result<Type> {
        let bool_ty = || Type::Constructed(TypeName::Bool, vec![]);

        match op {
            "==" | "!=" | "<" | ">" | "<=" | ">=" => {
                self.unify_or_err(
                    &left_type,
                    &right_type,
                    span,
                    &format!("Operator '{}' requires same-typed operands", op),
                )?;
                Ok(bool_ty())
            }
            "&&" | "||" => {
                let bt = bool_ty();
                let ctx = format!("Logical operator '{}' requires bool operands", op);
                self.unify_or_err(&left_type, &bt, span, &ctx)?;
                self.unify_or_err(&right_type, &bt, span, &ctx)?;
                Ok(bt)
            }
            "+" | "-" | "*" | "/" | "%" | "**" => {
                self.infer_arith_op_result(op, left_type, right_type, span)
            }
            _ => Err(err_type_at!(span, "Unknown binary operator: {}", op)),
        }
    }

    /// Result type for arithmetic operators (`+ - * / % **`). Handles
    /// matrix-mul (only for `*`), vec-vec / vec-scalar / scalar-scalar
    /// dispatch, and the numeric-elem check.
    fn infer_arith_op_result(
        &mut self,
        op: &str,
        left_type: Type,
        right_type: Type,
        span: Span,
    ) -> Result<Type> {
        let l = left_type.apply(&self.context);
        let r = right_type.apply(&self.context);

        // `*` covers matrix products that don't reduce to plain
        // component-wise arithmetic. Try those first.
        if op == "*" {
            if let Some(ty) = self.try_mat_mul(&l, &r, span)? {
                return Ok(ty);
            }
        }

        // Vec-vec, vec-scalar, scalar-vec, or scalar-scalar.
        let result = match (Self::vec_parts(&l), Self::vec_parts(&r)) {
            (Some(_), Some(_)) => {
                self.unify_or_err(
                    &left_type,
                    &right_type,
                    span,
                    "Vector arithmetic requires matching vector types",
                )?;
                l.clone()
            }
            (Some((le, _)), None) => {
                self.unify_or_err(le, &r, span, "Vector-scalar element type must match scalar")?;
                l.clone()
            }
            (None, Some((re, _))) => {
                self.unify_or_err(&l, re, span, "Scalar-vector scalar must match element type")?;
                r.clone()
            }
            (None, None) => {
                self.unify_or_err(
                    &left_type,
                    &right_type,
                    span,
                    &format!("Operator '{}' requires same-typed operands", op),
                )?;
                l.clone()
            }
        };

        // Operands (or vec elements) must be numeric.
        let check = Self::vec_parts(&result).map(|(e, _)| e).unwrap_or(&result);
        if let Some(false) = Self::is_numeric_type(check) {
            return Err(err_type_at!(
                span,
                "Arithmetic operator '{}' requires numeric operands, got {}",
                op,
                self.format_type(check)
            ));
        }

        Ok(result)
    }

    /// Try the matrix-product family for `*`. Returns `Ok(Some(_))` if
    /// the operands match a Mat-Mat / Mat-Vec / Vec-Mat / Mat-scalar /
    /// scalar-Mat case (with unification performed); `Ok(None)` if
    /// neither operand is a Mat, leaving the caller to dispatch to
    /// component-wise arithmetic.
    fn try_mat_mul(&mut self, l: &Type, r: &Type, span: Span) -> Result<Option<Type>> {
        let lm = Self::mat_parts(l);
        let rm = Self::mat_parts(r);
        let lv = Self::vec_parts(l);
        let rv = Self::vec_parts(r);

        let result = match (lm, rm, lv, rv) {
            // Mat × Mat: inner dims must match (left rows = right cols).
            (Some((le, lc, lr)), Some((re, rc, rr)), _, _) => {
                self.unify_or_err(lr, rc, span, "Matrix multiply inner dimensions must match")?;
                self.unify_or_err(le, re, span, "Matrix multiply element types must match")?;
                Some(Type::Constructed(
                    TypeName::Mat,
                    vec![le.clone(), lc.clone(), rr.clone()],
                ))
            }
            // Mat × Vec: matrix rows must equal vec size; result is Vec[cols].
            (Some((le, lc, lr)), _, _, Some((re, rs))) => {
                self.unify_or_err(
                    lr,
                    rs,
                    span,
                    "Matrix-vector multiply: matrix rows must match vector size",
                )?;
                self.unify_or_err(le, re, span, "Matrix-vector multiply element types must match")?;
                Some(Type::Constructed(TypeName::Vec, vec![le.clone(), lc.clone()]))
            }
            // Vec × Mat: vec size must equal matrix cols; result is Vec[rows].
            (_, Some((re, rc, rr)), Some((le, ls)), _) => {
                self.unify_or_err(
                    ls,
                    rc,
                    span,
                    "Vector-matrix multiply: vector size must match matrix cols",
                )?;
                self.unify_or_err(le, re, span, "Vector-matrix multiply element types must match")?;
                Some(Type::Constructed(TypeName::Vec, vec![le.clone(), rr.clone()]))
            }
            // Mat × scalar / scalar × Mat: element types must match.
            (Some((le, _, _)), None, _, None) => {
                self.unify_or_err(le, r, span, "Matrix-scalar element type must match scalar")?;
                Some(l.clone())
            }
            (None, Some((re, _, _)), None, _) => {
                self.unify_or_err(l, re, span, "Scalar-matrix scalar must match element type")?;
                Some(r.clone())
            }
            _ => None,
        };

        Ok(result)
    }

    /// Peel one arrow from a function type and unify an argument with the expected param.
    ///
    /// Returns the result type after application.
    fn unify_apply_arg(
        &mut self,
        func_ty: &Type,
        arg_ty: &Type,
        arg: &Expression,
        arg_index: usize,
    ) -> Result<Type> {
        // Create fresh variables for param and result
        let param_var = self.context.new_variable();
        let result_var = self.context.new_variable();
        let arrow_type = Type::arrow(param_var.clone(), result_var.clone());

        // Unify function type with arrow
        self.context.unify(func_ty, &arrow_type).map_err(|_| {
            err_type_at!(
                arg.h.span,
                "Cannot apply {} as a function",
                self.format_type(func_ty)
            )
        })?;

        // Get expected param type after unification
        let expected_param = param_var.apply(&self.context);

        // Strip uniqueness for unification
        let arg_stripped = strip_unique(arg_ty);
        let param_stripped = strip_unique(&expected_param);

        // Unify argument with expected param
        self.context.unify(&arg_stripped, &param_stripped).map_err(|e| {
            let error_msg = if arg.h.span.is_generated() {
                format!(
                    "Function argument type mismatch at argument {}: {:?}\n\
                         Expected param type: {}\n\
                         Actual arg type: {}\n\
                         Generated expression: {:#?}",
                    arg_index + 1,
                    e,
                    self.format_type(&param_stripped),
                    self.format_type(&arg_stripped),
                    arg
                )
            } else {
                format!("Function argument type mismatch: {:?}", e)
            };
            err_type_at!(arg.h.span, "{}", error_msg)
        })?;

        Ok(result_var)
    }

    /// Constrain a type to be an Array and return its components.
    ///
    /// Creates fresh type variables for elem, addrspace, and size, then unifies
    /// the given type with Array[elem, addrspace, size]. Returns the three
    /// component type variables.
    ///
    /// Caller should strip uniqueness before calling if needed.
    fn constrain_array_type(
        &mut self,
        array_ty: &Type,
        span: &Span,
        error_context: &str,
    ) -> Result<(Type, Type, Type)> {
        let elem_var = self.context.new_variable();
        let addrspace_var = self.context.new_variable();
        let size_var = self.context.new_variable();
        let want_array = Type::Constructed(
            TypeName::Array,
            vec![elem_var.clone(), size_var.clone(), addrspace_var.clone()],
        );

        self.context.unify(array_ty, &want_array).map_err(|_| {
            err_type_at!(
                *span,
                "{}: got {}",
                error_context,
                self.format_type(&array_ty.apply(&self.context))
            )
        })?;

        Ok((elem_var, addrspace_var, size_var))
    }

    /// Infer the type of a field access on a base type.
    ///
    /// Precedence:
    /// 1. Record fields (by TypeName::Record field map)
    /// 2. Tuple numeric indices (.0, .1, etc.)
    /// 3. Vec swizzle (x/y/z/w) - constrains to Vec if type is unknown
    /// 4. Known type fields via impl_source / record_field_map
    ///
    /// The base_ty should already be applied and uniqueness-stripped.
    fn infer_field_access(&mut self, base_ty: &Type, field: &str, span: &Span) -> Result<Type> {
        // 1. Record field access
        if let Type::Constructed(TypeName::Record(fields), field_types) = base_ty {
            if let Some(field_index) = fields.get_index(field) {
                if field_index < field_types.len() {
                    return Ok(field_types[field_index].clone());
                }
            }
            bail_type_at!(
                *span,
                "Record type has no field '{}'. Available: {}",
                field,
                fields.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            );
        }

        // 2. Tuple numeric index (.0, .1, etc.)
        if let Ok(index) = field.parse::<usize>() {
            if let Type::Constructed(TypeName::Tuple(_), elem_types) = base_ty {
                if index < elem_types.len() {
                    return Ok(elem_types[index].clone());
                }
                bail_type_at!(
                    *span,
                    "Tuple index {} out of bounds (tuple has {} elements)",
                    index,
                    elem_types.len()
                );
            }
            bail_type_at!(
                *span,
                "Numeric field access '.{}' requires a tuple type, got {}",
                index,
                self.format_type(base_ty)
            );
        }

        // 3. Vec swizzle: single-letter returns a scalar, 2–4-letter
        //    returns a vec of that length. Each letter is drawn from
        //    ONE of the two swizzle sets — `xyzw` (position) or `rgba`
        //    (color, aliased to position: r=x, g=y, b=z, a=w). Mixing
        //    sets (`.xg`) is rejected at the predicate level. Each
        //    letter must also be in range for the source vec's length
        //    (`.z` on a vec2 is a range error, caught here when the
        //    size is known, otherwise only that the base is a vec).
        if super::is_swizzle_field(field) {
            // Constrain base to some Vec<_, _>.
            let elem_var = self.context.new_variable();
            let size_var = self.context.new_variable();
            let want_vec = Type::Constructed(TypeName::Vec, vec![elem_var.clone(), size_var.clone()]);
            self.context.unify(base_ty, &want_vec).map_err(|_| {
                err_type_at!(
                    *span,
                    "Field '{}' requires a vector type, got {}",
                    field,
                    self.format_type(&base_ty.apply(&self.context))
                )
            })?;

            // If the size is resolved, bounds-check the swizzle letters.
            let resolved_base = base_ty.apply(&self.context);
            if let Type::Constructed(TypeName::Vec, args) = &resolved_base {
                if let Some(Type::Constructed(TypeName::Size(n), _)) = args.get(1) {
                    for c in field.chars() {
                        let idx = super::swizzle_component_index(c)
                            .expect("is_swizzle_field already accepted this letter");
                        if idx >= *n {
                            bail_type_at!(
                                *span,
                                "Swizzle '.{}' uses component '{}' (index {}) but source is vec{}",
                                field,
                                c,
                                idx,
                                n
                            );
                        }
                    }
                }
            }

            let elem = elem_var.apply(&self.context);
            if field.len() == 1 {
                return Ok(elem);
            }
            // Multi-letter swizzle → a new vec of that length.
            return Ok(Type::Constructed(
                TypeName::Vec,
                vec![elem, Type::Constructed(TypeName::Size(field.len()), vec![])],
            ));
        }

        // 4. Known type fields via impl_source / record_field_map
        if let Type::Constructed(type_name, _) = base_ty {
            let type_name_str = match type_name {
                TypeName::Bool => "bool".to_string(),
                TypeName::Float(bits) => format!("f{}", bits),
                TypeName::UInt(bits) => format!("u{}", bits),
                TypeName::Int(bits) => format!("i{}", bits),
                TypeName::Vec => "vec".to_string(),
                TypeName::Mat => "mat".to_string(),
                TypeName::Array => "array".to_string(),
                TypeName::Named(name) => name.clone(),
                _ => {
                    bail_type_at!(
                        *span,
                        "Type {} has no field '{}'",
                        self.format_type(base_ty),
                        field
                    );
                }
            };

            if let Some(field_type) = self.impl_source.get_field_type(&type_name_str, field) {
                return Ok(field_type);
            }
            if let Some(field_type) = self.record_field_map.get(&(type_name_str.clone(), field.to_string()))
            {
                return Ok(field_type.clone());
            }

            bail_type_at!(*span, "Type '{}' has no field '{}'", type_name_str, field);
        }

        bail_type_at!(
            *span,
            "Field access '{}' not supported on type {}",
            field,
            self.format_type(base_ty)
        )
    }

    /// Two-pass function application for better lambda inference
    ///
    /// Pass 1: Process non-lambda arguments to constrain type variables
    /// Pass 2: Process lambda arguments with bidirectionally checked expected types
    ///
    /// This allows map (\x -> ...) arr to infer properly regardless of argument order
    fn apply_two_pass(&mut self, mut func_type: Type, args: &[Expression]) -> Result<Type> {
        // Track expected types for lambda arguments (for second pass)
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
            } else if matches!(&arg.kind, ExprKind::Constructor(_, _)) {
                // Constructor expressions need bidirectional checking against
                // the expected parameter type to disambiguate which sum type
                // they belong to — `infer_expression` produces an "ambiguous
                // constructor" error.
                let param_var = self.context.new_variable();
                let result_var = self.context.new_variable();
                let arrow_type = Type::arrow(param_var.clone(), result_var.clone());
                self.context.unify(&func_type, &arrow_type).map_err(|_| {
                    err_type_at!(
                        arg.h.span,
                        "Cannot apply {} as a function",
                        self.format_type(&func_type)
                    )
                })?;
                let expected_param = param_var.apply(&self.context);
                self.check_expression(arg, &expected_param)?;
                func_type = result_var;
            } else {
                // For non-lambda argument: infer type and unify with expected param
                let arg_type = self.infer_expression(arg)?;
                func_type = self.unify_apply_arg(&func_type, &arg_type, arg, i)?;
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

#[cfg(test)]
#[path = "checker_tests.rs"]
mod checker_tests;
