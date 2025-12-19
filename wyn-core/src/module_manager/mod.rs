//! Module manager for lazy loading and caching module definitions

use crate::ast::{
    Decl, Declaration, ModuleExpression, ModuleTypeExpression, Node, NodeCounter, NodeId, Pattern,
    PatternKind, Program, Spec, Type, TypeName, TypeParam,
};
use crate::error::Result;
use crate::lexer;
use crate::parser::Parser;
use crate::scope::ScopeStack;
use crate::types::checker::TypeChecker;
use crate::{bail_module, err_module, err_parse};
use polytype::{Context, TypeScheme};
use std::collections::{HashMap, HashSet};

/// Name resolver for tracking opened modules and resolving unqualified names
/// TODO: Integrate with elaboration to handle `open` declarations
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct NameResolver {
    /// Modules currently opened (via `open` declarations)
    opened_modules: Vec<String>,
    /// Local definitions in scope (for shadowing)
    local_scope: ScopeStack<()>,
}

#[allow(dead_code)]
impl NameResolver {
    fn new() -> Self {
        NameResolver {
            opened_modules: Vec::new(),
            local_scope: ScopeStack::new(),
        }
    }

    /// Resolve an unqualified name by checking opened modules
    /// Returns None if the name can't be resolved, or Some(qualified_name) if found
    fn resolve_name(&self, name: &str, module_manager: &ModuleManager) -> Option<String> {
        // 1. Check if it's locally defined (shadows everything)
        if self.local_scope.is_defined(name) {
            return None; // Keep unqualified, it's a local binding
        }

        // 2. Try each opened module in reverse order (most recent first)
        for module_name in self.opened_modules.iter().rev() {
            // Check if this module has this function
            if let Some(elaborated) = module_manager.elaborated_modules.get(module_name) {
                for item in &elaborated.items {
                    let item_name = match item {
                        ElaboratedItem::Spec(Spec::Sig(n, _, _)) => Some(n.as_str()),
                        ElaboratedItem::Spec(Spec::SigOp(op, _)) => Some(op.as_str()),
                        ElaboratedItem::Decl(decl) => Some(decl.name.as_str()),
                        _ => None,
                    };

                    if item_name == Some(name) {
                        return Some(format!("{}.{}", module_name, name));
                    }
                }
            }
        }

        // 3. Not found in any opened module
        None
    }

    /// Open a module (bring its names into scope)
    fn open_module(&mut self, module_name: String) {
        self.opened_modules.push(module_name);
    }

    /// Close the most recently opened module
    fn close_module(&mut self) {
        self.opened_modules.pop();
    }

    /// Push a new scope for local definitions
    fn push_scope(&mut self) {
        self.local_scope.push_scope();
    }

    /// Pop the current scope
    fn pop_scope(&mut self) {
        self.local_scope.pop_scope();
    }

    /// Add a local definition (for shadowing)
    fn add_local(&mut self, name: String) {
        self.local_scope.insert(name, ());
    }
}

/// Represents a single item in an elaborated module
#[derive(Debug, Clone)]
pub enum ElaboratedItem {
    /// A signature spec (from module type) with substitutions applied
    Spec(Spec),
    /// A declaration (def/let) from module body with substitutions and resolved names
    Decl(Decl),
    /// A type alias from module body (e.g., `type state = f32`)
    TypeAlias(String, Type),
}

/// Represents a fully elaborated module with all includes expanded, type substitutions applied,
/// and names resolved. Contains both signature specs and body declarations in source order.
#[derive(Debug, Clone)]
pub struct ElaboratedModule {
    pub name: String,
    /// Items in source order (specs first, then body declarations)
    pub items: Vec<ElaboratedItem>,
}

/// Pre-elaborated prelude data that can be shared across compilations (for test performance)
#[derive(Clone)]
pub struct PreElaboratedPrelude {
    /// Module type registry: type name -> ModuleTypeExpression
    pub module_type_registry: HashMap<String, ModuleTypeExpression>,
    /// Elaborated modules: module_name -> ElaboratedModule
    pub elaborated_modules: HashMap<String, ElaboratedModule>,
    /// Set of known module names (for name resolution)
    pub known_modules: HashSet<String>,
    /// Type aliases from modules: "module.typename" -> underlying Type
    pub type_aliases: HashMap<String, Type>,
    /// Top-level prelude function declarations (auto-imported)
    pub prelude_functions: HashMap<String, Decl>,
    /// Type table for prelude function bodies (from type-checking during prelude creation)
    pub prelude_type_table: HashMap<NodeId, TypeScheme<TypeName>>,
}

/// Manages lazy loading of module files
pub struct ModuleManager {
    /// Module type registry: type name -> ModuleTypeExpression
    module_type_registry: HashMap<String, ModuleTypeExpression>,
    /// Elaborated modules: module_name -> ElaboratedModule
    pub(crate) elaborated_modules: HashMap<String, ElaboratedModule>,
    /// Set of known module names (for name resolution)
    known_modules: HashSet<String>,
    /// Type aliases from modules: "module.typename" -> underlying Type
    type_aliases: HashMap<String, Type>,
    /// Top-level prelude function declarations (auto-imported)
    prelude_functions: HashMap<String, Decl>,
    /// Type table for prelude function bodies
    prelude_type_table: HashMap<NodeId, TypeScheme<TypeName>>,
}

impl ModuleManager {
    /// Create a new module manager and load prelude files using the provided counter
    pub fn new(node_counter: &mut NodeCounter) -> Self {
        let mut manager = Self::new_empty();
        if let Err(e) = manager.load_prelude_files(node_counter) {
            eprintln!("ERROR loading prelude files: {:?}", e);
        }
        manager
    }

    /// Create an empty module manager without loading prelude (internal helper)
    fn new_empty() -> Self {
        let known_modules = [
            "f32",
            "f64",
            "f16",
            "i8",
            "i16",
            "i32",
            "i64",
            "u8",
            "u16",
            "u32",
            "u64",
            "bool",
            "graphics32",
            "graphics64",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        ModuleManager {
            module_type_registry: HashMap::new(),
            elaborated_modules: HashMap::new(),
            known_modules,
            type_aliases: HashMap::new(),
            prelude_functions: HashMap::new(),
            prelude_type_table: HashMap::new(),
        }
    }

    /// Load all prelude files automatically
    fn load_prelude_files(&mut self, node_counter: &mut NodeCounter) -> Result<()> {
        // Load all prelude files using include_str!
        self.load_str(include_str!("../../../prelude/math.wyn"), node_counter)?;
        self.load_str(include_str!("../../../prelude/graphics.wyn"), node_counter)?;
        self.load_str(include_str!("../../../prelude/rand.wyn"), node_counter)?;
        self.load_str(include_str!("../../../prelude/soacs.wyn"), node_counter)?;
        Ok(())
    }

    /// Create a pre-elaborated prelude by loading all prelude files
    /// This can be cached and reused across compilations
    pub fn create_prelude(node_counter: &mut NodeCounter) -> Result<PreElaboratedPrelude> {
        let mut manager = Self::new_empty();
        manager.load_prelude_files(node_counter)?;

        // Type-check prelude functions to populate the type table
        let mut checker = TypeChecker::new(&manager);
        checker.load_builtins()?;
        checker.check_prelude_functions()?;
        let prelude_type_table = checker.into_type_table();

        Ok(PreElaboratedPrelude {
            module_type_registry: manager.module_type_registry,
            elaborated_modules: manager.elaborated_modules,
            known_modules: manager.known_modules,
            type_aliases: manager.type_aliases,
            prelude_functions: manager.prelude_functions,
            prelude_type_table,
        })
    }

    /// Create a ModuleManager using a pre-elaborated prelude (avoids re-parsing)
    /// Advances the node_counter to start after all prelude NodeIds
    pub fn from_prelude(prelude: &PreElaboratedPrelude) -> Self {
        ModuleManager {
            module_type_registry: prelude.module_type_registry.clone(),
            elaborated_modules: prelude.elaborated_modules.clone(),
            known_modules: prelude.known_modules.clone(),
            type_aliases: prelude.type_aliases.clone(),
            prelude_functions: prelude.prelude_functions.clone(),
            prelude_type_table: prelude.prelude_type_table.clone(),
        }
    }

    /// Check if a name is a known module
    pub fn is_known_module(&self, name: &str) -> bool {
        self.known_modules.contains(name)
    }

    /// Resolve a qualified type alias (e.g., "rand.state" -> underlying type)
    pub fn resolve_type_alias(&self, qualified_name: &str) -> Option<&Type> {
        self.type_aliases.get(qualified_name)
    }

    /// Resolve type aliases in a type, given the module context
    /// For unqualified type names like "state", looks them up as "module_name.state"
    fn resolve_type_aliases_in_module(&self, ty: &Type, module_name: &str) -> Type {
        match ty {
            Type::Constructed(TypeName::Named(name), args) => {
                // Try to resolve as a module-local type alias
                let qualified_name = if name.contains('.') {
                    name.clone() // Already qualified
                } else {
                    format!("{}.{}", module_name, name) // Make qualified
                };
                if let Some(underlying) = self.type_aliases.get(&qualified_name) {
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
                // Resolve aliases in type arguments
                let resolved_args: Vec<Type> =
                    args.iter().map(|a| self.resolve_type_aliases_in_module(a, module_name)).collect();
                Type::Constructed(name.clone(), resolved_args)
            }
            Type::Variable(id) => Type::Variable(*id),
        }
    }

    /// Load and elaborate modules from a source string
    pub fn load_str(&mut self, source: &str, node_counter: &mut NodeCounter) -> Result<()> {
        // Parse the source
        let tokens = lexer::tokenize(source).map_err(|e| err_parse!("{}", e))?;
        let mut parser = Parser::new(tokens, node_counter);
        let program = parser.parse()?;

        // Register module types first
        self.register_module_types(&program)?;

        // Elaborate all modules from the program
        self.elaborate_all_modules(&program)?;

        Ok(())
    }

    /// Elaborate all module bindings and top-level declarations from a parsed program
    fn elaborate_all_modules(&mut self, program: &Program) -> Result<()> {
        for decl in &program.declarations {
            // Collect top-level function declarations for prelude
            if let Declaration::Decl(d) = decl {
                self.prelude_functions.insert(d.name.clone(), d.clone());
                continue;
            }

            if let Declaration::ModuleBind(mb) = decl {
                if self.elaborated_modules.contains_key(&mb.name) {
                    bail_module!("Module '{}' is already defined", mb.name);
                }

                // Extract type substitutions from the signature
                let substitutions = if let Some(signature) = &mb.signature {
                    self.extract_substitutions(signature)?
                } else {
                    HashMap::new()
                };

                let mut items = Vec::new();

                // Elaborate the module signature if it exists
                if let Some(signature) = &mb.signature {
                    let specs = self.elaborate_module_type(signature, &HashMap::new())?;
                    // Wrap specs in ElaboratedItem::Spec
                    items.extend(specs.into_iter().map(ElaboratedItem::Spec));
                }

                // Elaborate the module body
                let body_items = self.elaborate_module_body(&mb.body, &mb.name, &substitutions)?;
                items.extend(body_items);

                let elaborated = ElaboratedModule {
                    name: mb.name.clone(),
                    items,
                };

                // Register type aliases from this module
                for item in &elaborated.items {
                    if let ElaboratedItem::TypeAlias(type_name, underlying_type) = item {
                        let qualified_name = format!("{}.{}", mb.name, type_name);
                        self.type_aliases.insert(qualified_name, underlying_type.clone());
                    }
                }

                self.elaborated_modules.insert(mb.name.clone(), elaborated);
                // Also register as a known module for name resolution
                self.known_modules.insert(mb.name.clone());
            }
        }
        Ok(())
    }

    /// Extract type substitutions from a module signature
    /// e.g., (float with t = f32 with int_t = u32) -> {t: f32, int_t: u32}
    fn extract_substitutions(&self, mte: &ModuleTypeExpression) -> Result<HashMap<String, Type>> {
        let mut substitutions = HashMap::new();
        let mut current = mte;

        // Walk through nested With expressions
        loop {
            match current {
                ModuleTypeExpression::With(inner, type_name, _type_params, type_value) => {
                    substitutions.insert(type_name.clone(), type_value.clone());
                    current = inner;
                }
                _ => break,
            }
        }

        Ok(substitutions)
    }

    /// Register all module type definitions from a parsed program
    fn register_module_types(&mut self, program: &Program) -> Result<()> {
        for decl in &program.declarations {
            if let Declaration::ModuleTypeBind(mtb) = decl {
                if self.module_type_registry.contains_key(&mtb.name) {
                    bail_module!("Module type '{}' is already defined", mtb.name);
                }
                self.module_type_registry.insert(mtb.name.clone(), mtb.definition.clone());
            }
        }
        Ok(())
    }

    /// Elaborate a module type expression into a flat list of specs
    /// Recursively expands includes and applies type substitutions
    fn elaborate_module_type(
        &self,
        mte: &ModuleTypeExpression,
        substitutions: &HashMap<String, Type>,
    ) -> Result<Vec<Spec>> {
        match mte {
            ModuleTypeExpression::Name(name) => {
                // Look up the module type in the registry
                let definition = self
                    .module_type_registry
                    .get(name)
                    .ok_or_else(|| err_module!("Module type '{}' not found", name))?;
                // Recurse on the definition
                self.elaborate_module_type(definition, substitutions)
            }

            ModuleTypeExpression::Signature(specs) => {
                // Process each spec, expanding includes and applying substitutions
                let mut result = Vec::new();
                for spec in specs {
                    match spec {
                        Spec::Include(inner_mte) => {
                            // Recursively elaborate the included module type
                            let included_specs = self.elaborate_module_type(inner_mte, substitutions)?;
                            result.extend(included_specs);
                        }
                        _ => {
                            // Apply type substitutions to the spec and add it
                            let substituted_spec = self.substitute_in_spec(spec, substitutions);
                            result.push(substituted_spec);
                        }
                    }
                }
                Ok(result)
            }

            ModuleTypeExpression::With(inner, type_name, _type_params, type_value) => {
                // Add the type substitution and recurse on the inner expression
                let mut new_substitutions = substitutions.clone();
                new_substitutions.insert(type_name.clone(), type_value.clone());
                self.elaborate_module_type(inner, &new_substitutions)
            }

            ModuleTypeExpression::Arrow(_, _, _) | ModuleTypeExpression::FunctorType(_, _) => {
                // Functor types not yet supported
                Err(err_module!("Functor types are not yet supported"))
            }
        }
    }

    /// Apply type substitutions to a spec
    fn substitute_in_spec(&self, spec: &Spec, substitutions: &HashMap<String, Type>) -> Spec {
        match spec {
            Spec::Sig(name, type_params, ty) => {
                let substituted_ty = self.substitute_in_type(ty, substitutions);
                Spec::Sig(name.clone(), type_params.clone(), substituted_ty)
            }
            Spec::SigOp(op, ty) => {
                let substituted_ty = self.substitute_in_type(ty, substitutions);
                Spec::SigOp(op.clone(), substituted_ty)
            }
            Spec::Type(kind, name, type_params, maybe_ty) => {
                let substituted_ty = maybe_ty.as_ref().map(|ty| self.substitute_in_type(ty, substitutions));
                Spec::Type(kind.clone(), name.clone(), type_params.clone(), substituted_ty)
            }
            Spec::Module(name, mte) => {
                // Don't substitute in nested module signatures for now
                Spec::Module(name.clone(), mte.clone())
            }
            Spec::Include(_) => {
                // Includes should have been expanded by now
                spec.clone()
            }
        }
    }

    /// Apply type substitutions to a type
    fn substitute_in_type(&self, ty: &Type, substitutions: &HashMap<String, Type>) -> Type {
        use crate::ast::TypeName;

        match ty {
            Type::Constructed(name, args) => {
                // Check if this is a named type that should be substituted
                if let TypeName::Named(type_name) = name {
                    if args.is_empty() {
                        if let Some(replacement) = substitutions.get(type_name) {
                            return replacement.clone();
                        }
                    }
                }

                // Recursively substitute in type arguments
                let new_args: Vec<Type> =
                    args.iter().map(|arg| self.substitute_in_type(arg, substitutions)).collect();
                Type::Constructed(name.clone(), new_args)
            }
            Type::Variable(_) => ty.clone(),
        }
    }

    /// Convert a type with type parameters to a polymorphic TypeScheme
    /// Converts SizeVar("n") and UserVar("t") to fresh Type::Variables
    /// and wraps the result in nested TypeScheme::Polytype layers
    fn convert_to_polytype(
        &self,
        ty: &Type,
        type_params: &[TypeParam],
        context: &mut Context<TypeName>,
    ) -> TypeScheme<TypeName> {
        if type_params.is_empty() {
            return TypeScheme::Monotype(ty.clone());
        }

        // Create fresh variables for each parameter and build substitution map
        let mut substitutions: HashMap<String, polytype::Variable> = HashMap::new();
        let mut var_ids = Vec::new();

        for param in type_params {
            let var = context.new_variable();
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
        let substituted_ty = self.substitute_params(&ty, &substitutions);

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

    /// Recursively substitute SizeVar and UserVar with Variable
    fn substitute_params(&self, ty: &Type, substitutions: &HashMap<String, polytype::Variable>) -> Type {
        match ty {
            Type::Constructed(TypeName::SizeVar(name), args) => {
                if let Some(&var_id) = substitutions.get(name) {
                    Type::Variable(var_id)
                } else {
                    // Not in our substitution map, keep as-is
                    Type::Constructed(
                        TypeName::SizeVar(name.clone()),
                        args.iter().map(|a| self.substitute_params(a, substitutions)).collect(),
                    )
                }
            }
            Type::Constructed(TypeName::UserVar(name), args) => {
                if let Some(&var_id) = substitutions.get(name) {
                    Type::Variable(var_id)
                } else {
                    Type::Constructed(
                        TypeName::UserVar(name.clone()),
                        args.iter().map(|a| self.substitute_params(a, substitutions)).collect(),
                    )
                }
            }
            Type::Constructed(name, args) => Type::Constructed(
                name.clone(),
                args.iter().map(|a| self.substitute_params(a, substitutions)).collect(),
            ),
            Type::Variable(_) => ty.clone(),
        }
    }

    /// Query the type of a function in a specific module
    /// Returns a TypeScheme for polymorphic functions (with type/size params)
    /// e.g., get_module_function_type("f32", "sum") -> TypeScheme::Polytype for [n] param
    pub fn get_module_function_type(
        &self,
        module_name: &str,
        function_name: &str,
        context: &mut Context<TypeName>,
    ) -> Result<TypeScheme<TypeName>> {
        // Look up the elaborated module
        let elaborated = self
            .elaborated_modules
            .get(module_name)
            .ok_or_else(|| err_module!("Module '{}' not found", module_name))?;

        // Search for the function in the elaborated items
        for item in &elaborated.items {
            match item {
                ElaboratedItem::Spec(spec) => match spec {
                    Spec::Sig(name, type_params, ty) if name == function_name => {
                        // Convert to TypeScheme if there are type/size parameters
                        return Ok(self.convert_to_polytype(ty, type_params, context));
                    }
                    Spec::SigOp(op, ty) if op == function_name => {
                        // Operators currently don't have type parameters, return as Monotype
                        return Ok(TypeScheme::Monotype(ty.clone()));
                    }
                    _ => {}
                },
                ElaboratedItem::Decl(decl) if decl.name == function_name => {
                    // Build the full function type from parameters and return type
                    // For def min (x: f32) (y: f32): f32, we need to construct f32 -> f32 -> f32
                    return self.build_function_type_from_decl(decl, module_name, context);
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

    /// Build the full function type from a declaration's parameters and return type
    fn build_function_type_from_decl(
        &self,
        decl: &Decl,
        module_name: &str,
        context: &mut Context<TypeName>,
    ) -> Result<TypeScheme<TypeName>> {
        // Extract parameter types and resolve any type aliases within the module
        let mut param_types = Vec::new();
        for param in &decl.params {
            if let Some(param_ty) = self.extract_type_from_pattern(param) {
                // Resolve type aliases (e.g., "state" -> "f32" within the rand module)
                let resolved_ty = self.resolve_type_aliases_in_module(&param_ty, module_name);
                param_types.push(resolved_ty);
            } else {
                bail_module!("Function parameter in '{}' lacks type annotation", decl.name);
            }
        }

        // Get return type (default to unit if not specified) and resolve aliases
        let return_type = decl.ty.clone().unwrap_or_else(|| Type::Constructed(TypeName::Unit, vec![]));
        let return_type = self.resolve_type_aliases_in_module(&return_type, module_name);

        // Build function type by folding right-to-left
        // f32 -> f32 -> f32 is represented as f32 -> (f32 -> f32)
        let mut result_type = return_type;
        for param_ty in param_types.into_iter().rev() {
            result_type = Type::Constructed(TypeName::Arrow, vec![param_ty, result_type]);
        }

        // Convert to TypeScheme if there are type/size parameters
        // For declarations, check if the type contains UserVar or SizeVar
        let type_params = self.extract_type_params_from_type(&result_type);
        Ok(self.convert_to_polytype(&result_type, &type_params, context))
    }

    /// Extract type parameters from a type by finding all UserVar and SizeVar
    fn extract_type_params_from_type(&self, ty: &Type) -> Vec<TypeParam> {
        let mut params = HashSet::new();
        self.collect_type_params(ty, &mut params);

        params.into_iter().collect()
    }

    /// Recursively collect type parameters from a type
    fn collect_type_params(&self, ty: &Type, params: &mut HashSet<TypeParam>) {
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

    /// Extract type annotation from a pattern
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

    /// Builtin/intrinsic modules that shouldn't be type-checked
    /// (their implementations use internal __builtin_* functions)
    const BUILTIN_MODULES: &'static [&'static str] = &[
        "f32",
        "f64",
        "f16",
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
        "bool",
        "graphics32",
        "graphics64",
    ];

    /// Check if a module is a builtin primitive module
    fn is_builtin_module(name: &str) -> bool {
        Self::BUILTIN_MODULES.contains(&name)
    }

    /// Get all module function declarations for type-checking
    /// Returns (module_name, decl) pairs for user-defined modules only
    /// (excludes builtin primitive modules like f32, i32, etc.)
    pub fn get_module_function_declarations(&self) -> Vec<(&str, &Decl)> {
        self.elaborated_modules
            .iter()
            .filter(|(name, _)| !Self::is_builtin_module(name))
            .flat_map(|(module_name, elaborated)| {
                elaborated.items.iter().filter_map(move |item| {
                    if let ElaboratedItem::Decl(decl) = item {
                        Some((module_name.as_str(), decl))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Get all top-level prelude function declarations for flattening
    /// These are auto-imported functions from prelude files
    pub fn get_prelude_function_declarations(&self) -> Vec<&Decl> {
        self.prelude_functions.values().collect()
    }

    /// Get the type table for prelude function bodies
    pub fn get_prelude_type_table(&self) -> &HashMap<NodeId, TypeScheme<TypeName>> {
        &self.prelude_type_table
    }

    /// Check if a name is a qualified module reference (e.g., "f32.sum")
    pub fn is_qualified_name(name: &str) -> bool {
        name.contains('.')
    }

    /// Split a qualified name into (module, function) parts
    /// E.g., "f32.sum" -> Some(("f32", "sum"))
    pub fn split_qualified_name(name: &str) -> Option<(&str, &str)> {
        let parts: Vec<&str> = name.splitn(2, '.').collect();
        if parts.len() == 2 { Some((parts[0], parts[1])) } else { None }
    }


    /// Check if a name is a top-level prelude function (auto-imported)
    pub fn is_prelude_function(&self, name: &str) -> bool {
        self.prelude_functions.contains_key(name)
    }

    /// Get a top-level prelude function declaration
    pub fn get_prelude_function(&self, name: &str) -> Option<&Decl> {
        self.prelude_functions.get(name)
    }

    /// Get the type of a top-level prelude function
    /// Returns a TypeScheme for polymorphic functions
    pub fn get_prelude_function_type(
        &self,
        name: &str,
        context: &mut Context<TypeName>,
    ) -> Option<TypeScheme<TypeName>> {
        let decl = self.prelude_functions.get(name)?;

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
        Some(self.convert_to_polytype(&result_type, &type_params, context))
    }

    /// Elaborate a module body expression into a list of elaborated items
    /// Applies type substitutions to declaration signatures
    fn elaborate_module_body(
        &self,
        module_expr: &ModuleExpression,
        module_name: &str,
        substitutions: &HashMap<String, Type>,
    ) -> Result<Vec<ElaboratedItem>> {
        match module_expr {
            ModuleExpression::Struct(declarations) => {
                // First pass: collect all function names in this module (for intra-module resolution)
                let mut module_functions: HashSet<String> = HashSet::new();
                for decl in declarations {
                    match decl {
                        Declaration::Decl(d) => {
                            module_functions.insert(d.name.clone());
                        }
                        Declaration::Sig(sig_decl) => {
                            module_functions.insert(sig_decl.name.clone());
                        }
                        _ => {}
                    }
                }

                let mut items = Vec::new();

                // Second pass: elaborate declarations with name resolution
                for decl in declarations {
                    match decl {
                        Declaration::Decl(d) => {
                            // Apply type substitutions and resolve names
                            let elaborated_decl = self.elaborate_decl_signature(
                                d,
                                module_name,
                                &module_functions,
                                substitutions,
                            );
                            items.push(ElaboratedItem::Decl(elaborated_decl));
                        }
                        Declaration::Sig(sig_decl) => {
                            // Apply type substitutions to sig declarations
                            let substituted_ty = self.substitute_in_type(&sig_decl.ty, substitutions);

                            // Convert size_params and type_params to Vec<TypeParam>
                            use crate::ast::TypeParam;
                            let mut type_params_vec = Vec::new();
                            for size_param in &sig_decl.size_params {
                                type_params_vec.push(TypeParam::Size(size_param.clone()));
                            }
                            for type_param in &sig_decl.type_params {
                                type_params_vec.push(TypeParam::Type(type_param.clone()));
                            }

                            let spec = Spec::Sig(sig_decl.name.clone(), type_params_vec, substituted_ty);
                            items.push(ElaboratedItem::Spec(spec));
                        }
                        Declaration::Open(_) => {
                            // TODO: Handle open declarations for name resolution
                        }
                        Declaration::TypeBind(type_bind) => {
                            // Record type aliases (e.g., `type state = f32`)
                            let substituted_ty =
                                self.substitute_in_type(&type_bind.definition, substitutions);
                            items.push(ElaboratedItem::TypeAlias(type_bind.name.clone(), substituted_ty));
                        }
                        _ => {
                            // Skip other declaration types (ModuleTypeBind, etc.)
                        }
                    }
                }

                Ok(items)
            }
            _ => {
                // For now, only handle struct module expressions
                Err(err_module!("Only struct module expressions are supported"))
            }
        }
    }

    /// Elaborate a declaration's signature (params and return type) with type substitutions
    fn elaborate_decl_signature(
        &self,
        decl: &Decl,
        module_name: &str,
        module_functions: &HashSet<String>,
        substitutions: &HashMap<String, Type>,
    ) -> Decl {
        // Apply type substitutions to params
        let new_params: Vec<Pattern> =
            decl.params.iter().map(|p| self.substitute_in_pattern(p, substitutions)).collect();

        // Apply type substitutions to return type
        let new_ty = decl.ty.as_ref().map(|ty| self.substitute_in_type(ty, substitutions));

        // Collect parameter names to avoid qualifying them as module functions
        let param_names: HashSet<String> =
            decl.params.iter().flat_map(|p| self.collect_pattern_names(p)).collect();

        // Resolve names in body (convert FieldAccess to QualifiedName, qualify intra-module refs)
        let mut new_body = decl.body.clone();
        self.resolve_names_in_expr(&mut new_body, module_name, module_functions, &param_names);

        Decl {
            keyword: decl.keyword,
            attributes: decl.attributes.clone(),
            name: decl.name.clone(),
            size_params: decl.size_params.clone(),
            type_params: decl.type_params.clone(),
            params: new_params,
            ty: new_ty,
            body: new_body,
        }
    }

    /// Collect all bound names from a pattern
    fn collect_pattern_names(&self, pattern: &Pattern) -> Vec<String> {
        use crate::ast::PatternKind;
        let mut names = Vec::new();
        match &pattern.kind {
            PatternKind::Name(name) => names.push(name.clone()),
            PatternKind::Typed(inner, _) => names.extend(self.collect_pattern_names(inner)),
            PatternKind::Tuple(pats) => {
                for p in pats {
                    names.extend(self.collect_pattern_names(p));
                }
            }
            PatternKind::Constructor(_, pats) => {
                for p in pats {
                    names.extend(self.collect_pattern_names(p));
                }
            }
            PatternKind::Record(fields) => {
                for f in fields {
                    if let Some(p) = &f.pattern {
                        names.extend(self.collect_pattern_names(p));
                    } else {
                        names.push(f.field.clone());
                    }
                }
            }
            PatternKind::Attributed(_, inner) => names.extend(self.collect_pattern_names(inner)),
            PatternKind::Wildcard | PatternKind::Unit | PatternKind::Literal(_) => {}
        }
        names
    }

    /// Resolve names in an expression within a module context
    /// - Converts FieldAccess to QualifiedName for external module references
    /// - Qualifies intra-module function references
    fn resolve_names_in_expr(
        &self,
        expr: &mut crate::ast::Expression,
        module_name: &str,
        module_functions: &HashSet<String>,
        local_bindings: &HashSet<String>,
    ) {
        use crate::ast::ExprKind;

        match &mut expr.kind {
            ExprKind::Identifier(quals, name) => {
                // Check if this is an intra-module function reference (not shadowed by local binding)
                if quals.is_empty() && !local_bindings.contains(name) && module_functions.contains(name) {
                    // Convert to qualified identifier: next -> rand.next
                    *quals = vec![module_name.to_string()];
                }
            }
            ExprKind::FieldAccess(obj, field) => {
                // Check if this is module.name pattern
                if let ExprKind::Identifier(quals, name) = &obj.kind {
                    if quals.is_empty() && self.known_modules.contains(name) {
                        // Convert to qualified Identifier
                        expr.kind = ExprKind::Identifier(vec![name.clone()], field.clone());
                        return;
                    }
                }
                // Otherwise recurse into object
                self.resolve_names_in_expr(obj, module_name, module_functions, local_bindings);
            }
            ExprKind::Application(func, args) => {
                self.resolve_names_in_expr(func, module_name, module_functions, local_bindings);
                for arg in args {
                    self.resolve_names_in_expr(arg, module_name, module_functions, local_bindings);
                }
            }
            ExprKind::Lambda(lambda) => {
                // Collect lambda parameter names
                let mut inner_bindings = local_bindings.clone();
                for p in &lambda.params {
                    inner_bindings.extend(self.collect_pattern_names(p));
                }
                self.resolve_names_in_expr(
                    &mut lambda.body,
                    module_name,
                    module_functions,
                    &inner_bindings,
                );
            }
            ExprKind::LetIn(let_in) => {
                self.resolve_names_in_expr(
                    &mut let_in.value,
                    module_name,
                    module_functions,
                    local_bindings,
                );
                // Collect let binding names
                let mut inner_bindings = local_bindings.clone();
                inner_bindings.extend(self.collect_pattern_names(&let_in.pattern));
                self.resolve_names_in_expr(
                    &mut let_in.body,
                    module_name,
                    module_functions,
                    &inner_bindings,
                );
            }
            ExprKind::If(if_expr) => {
                self.resolve_names_in_expr(
                    &mut if_expr.condition,
                    module_name,
                    module_functions,
                    local_bindings,
                );
                self.resolve_names_in_expr(
                    &mut if_expr.then_branch,
                    module_name,
                    module_functions,
                    local_bindings,
                );
                self.resolve_names_in_expr(
                    &mut if_expr.else_branch,
                    module_name,
                    module_functions,
                    local_bindings,
                );
            }
            ExprKind::BinaryOp(_, lhs, rhs) => {
                self.resolve_names_in_expr(lhs, module_name, module_functions, local_bindings);
                self.resolve_names_in_expr(rhs, module_name, module_functions, local_bindings);
            }
            ExprKind::UnaryOp(_, operand) => {
                self.resolve_names_in_expr(operand, module_name, module_functions, local_bindings);
            }
            ExprKind::Tuple(exprs) | ExprKind::ArrayLiteral(exprs) | ExprKind::VecMatLiteral(exprs) => {
                for e in exprs {
                    self.resolve_names_in_expr(e, module_name, module_functions, local_bindings);
                }
            }
            ExprKind::ArrayIndex(arr, idx) => {
                self.resolve_names_in_expr(arr, module_name, module_functions, local_bindings);
                self.resolve_names_in_expr(idx, module_name, module_functions, local_bindings);
            }
            ExprKind::ArrayWith { array, index, value } => {
                self.resolve_names_in_expr(array, module_name, module_functions, local_bindings);
                self.resolve_names_in_expr(index, module_name, module_functions, local_bindings);
                self.resolve_names_in_expr(value, module_name, module_functions, local_bindings);
            }
            ExprKind::RecordLiteral(fields) => {
                for (_, e) in fields {
                    self.resolve_names_in_expr(e, module_name, module_functions, local_bindings);
                }
            }
            ExprKind::Loop(loop_expr) => {
                if let Some(ref mut init) = loop_expr.init {
                    self.resolve_names_in_expr(init, module_name, module_functions, local_bindings);
                }
                // Collect loop variable binding
                let mut inner_bindings = local_bindings.clone();
                inner_bindings.extend(self.collect_pattern_names(&loop_expr.pattern));
                match &mut loop_expr.form {
                    crate::ast::LoopForm::While(cond) => {
                        self.resolve_names_in_expr(cond, module_name, module_functions, &inner_bindings);
                    }
                    crate::ast::LoopForm::For(var, bound) => {
                        inner_bindings.insert(var.clone());
                        self.resolve_names_in_expr(bound, module_name, module_functions, local_bindings);
                    }
                    crate::ast::LoopForm::ForIn(var_pat, iter) => {
                        inner_bindings.extend(self.collect_pattern_names(var_pat));
                        self.resolve_names_in_expr(iter, module_name, module_functions, local_bindings);
                    }
                }
                self.resolve_names_in_expr(
                    &mut loop_expr.body,
                    module_name,
                    module_functions,
                    &inner_bindings,
                );
            }
            ExprKind::Match(match_expr) => {
                self.resolve_names_in_expr(
                    &mut match_expr.scrutinee,
                    module_name,
                    module_functions,
                    local_bindings,
                );
                for case in &mut match_expr.cases {
                    let mut inner_bindings = local_bindings.clone();
                    inner_bindings.extend(self.collect_pattern_names(&case.pattern));
                    self.resolve_names_in_expr(
                        &mut case.body,
                        module_name,
                        module_functions,
                        &inner_bindings,
                    );
                }
            }
            ExprKind::Range(range_expr) => {
                self.resolve_names_in_expr(
                    &mut range_expr.start,
                    module_name,
                    module_functions,
                    local_bindings,
                );
                self.resolve_names_in_expr(
                    &mut range_expr.end,
                    module_name,
                    module_functions,
                    local_bindings,
                );
                if let Some(step) = &mut range_expr.step {
                    self.resolve_names_in_expr(step, module_name, module_functions, local_bindings);
                }
            }
            ExprKind::Slice(slice_expr) => {
                self.resolve_names_in_expr(
                    &mut slice_expr.array,
                    module_name,
                    module_functions,
                    local_bindings,
                );
                if let Some(start) = &mut slice_expr.start {
                    self.resolve_names_in_expr(start, module_name, module_functions, local_bindings);
                }
                if let Some(end) = &mut slice_expr.end {
                    self.resolve_names_in_expr(end, module_name, module_functions, local_bindings);
                }
            }
            ExprKind::TypeAscription(inner, _) | ExprKind::TypeCoercion(inner, _) => {
                self.resolve_names_in_expr(inner, module_name, module_functions, local_bindings);
            }
            ExprKind::Assert(cond, body) => {
                self.resolve_names_in_expr(cond, module_name, module_functions, local_bindings);
                self.resolve_names_in_expr(body, module_name, module_functions, local_bindings);
            }
            // Leaf expressions - nothing to resolve
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::StringLiteral(_)
            | ExprKind::Unit
            | ExprKind::TypeHole => {}
        }
    }

    /// Apply type substitutions to a pattern
    fn substitute_in_pattern(&self, pattern: &Pattern, substitutions: &HashMap<String, Type>) -> Pattern {
        let new_kind = match &pattern.kind {
            PatternKind::Typed(inner, ty) => {
                let new_ty = self.substitute_in_type(ty, substitutions);
                PatternKind::Typed(inner.clone(), new_ty)
            }
            PatternKind::Tuple(pats) => {
                let new_pats: Vec<Pattern> =
                    pats.iter().map(|p| self.substitute_in_pattern(p, substitutions)).collect();
                PatternKind::Tuple(new_pats)
            }
            PatternKind::Record(fields) => {
                let new_fields = fields
                    .iter()
                    .map(|field| crate::ast::RecordPatternField {
                        field: field.field.clone(),
                        pattern: field
                            .pattern
                            .as_ref()
                            .map(|p| self.substitute_in_pattern(p, substitutions)),
                    })
                    .collect();
                PatternKind::Record(new_fields)
            }
            PatternKind::Constructor(name, pats) => {
                let new_pats: Vec<Pattern> =
                    pats.iter().map(|p| self.substitute_in_pattern(p, substitutions)).collect();
                PatternKind::Constructor(name.clone(), new_pats)
            }
            PatternKind::Attributed(attrs, inner) => {
                let new_inner = self.substitute_in_pattern(inner, substitutions);
                PatternKind::Attributed(attrs.clone(), Box::new(new_inner))
            }
            // Name, Wildcard, Literal, Unit don't contain types
            _ => pattern.kind.clone(),
        };

        Node {
            h: pattern.h.clone(),
            kind: new_kind,
        }
    }
}

#[cfg(test)]
mod tests;
